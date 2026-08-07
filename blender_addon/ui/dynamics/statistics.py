# File: blender_addon/ui/dynamics/statistics.py
# Code: GitHub Copilot
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Current-frame solver statistics panel and scalar CSV export."""

from __future__ import annotations

import csv
import os

import bpy  # pyright: ignore
from bpy.app.handlers import persistent  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_
from bpy.props import StringProperty  # pyright: ignore
from bpy.types import Operator, Panel  # pyright: ignore
from bpy_extras.io_utils import ExportHelper  # pyright: ignore

from ...core.encoder import resolve_start_frame
from ...core.statistics_cache import (
    CHANNELS,
    CHANNEL_BY_ID,
    StatisticsCacheError,
    iter_scalar_records,
    load_manifest,
    manifest_object,
    read_record,
    scalar_value,
)
from ...core.utils import get_category_name
from ...models.enum_props import EnumProperty, dynamic_enum_items
from ...models.groups import get_addon_data, has_addon_data, iterate_active_object_groups
from ...core.uuid_registry import get_object_by_uuid


@persistent
def statistics_frame_change_handler(_scene, _depsgraph=None):
    from ...core.utils import redraw_all_windows

    redraw_all_windows("VIEW_3D")


def _assigned_objects(scene):
    for group in iterate_active_object_groups(scene):
        for assigned in group.assigned_objects:
            if not assigned.included or not assigned.uuid:
                continue
            yield group, assigned, get_object_by_uuid(assigned.uuid)


def _format_value(value, channel_id):
    if value is None:
        return iface_("N/A")
    if channel_id == "CONTACT_COUNT":
        count = int(value)
        for threshold, suffix in (
            (1_000_000_000, "B"),
            (1_000_000, "M"),
            (1_000, "K"),
        ):
            if count >= threshold:
                return f"{count / threshold:.3g}{suffix}"
        return str(count)
    unit = CHANNEL_BY_ID[channel_id][2]
    if unit == "ratio":
        return f"{float(value):.3g} ({(float(value) - 1.0) * 100.0:+.3g}%)"
    return f"{float(value):.3g}"


def _label_with_unit(label, unit):
    if unit in ("axis", "count"):
        return iface_(label)
    return iface_("{label} ({unit})").format(label=label, unit=unit)


_VECTOR_ROWS = (
    ("Location", ("LOCATION_X", "LOCATION_Y", "LOCATION_Z")),
    ("Velocity", ("VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z")),
    (
        "Acceleration",
        ("ACCELERATION_X", "ACCELERATION_Y", "ACCELERATION_Z"),
    ),
    (
        "Angular Velocity",
        ("ANGULAR_VELOCITY_X", "ANGULAR_VELOCITY_Y", "ANGULAR_VELOCITY_Z"),
    ),
    ("Angular Axis", ("ANGULAR_AXIS_X", "ANGULAR_AXIS_Y", "ANGULAR_AXIS_Z")),
)
_VECTOR_CHANNELS = frozenset(
    channel for _label, channels in _VECTOR_ROWS for channel in channels
)
_MEASURE_STRETCH = {
    "SURFACE_AREA": "AREA_STRETCH",
    "VOLUME": "VOLUME_STRETCH",
    "ROD_LENGTH": "LENGTH_STRETCH",
}
_COMBINED_STRETCH_CHANNELS = frozenset(_MEASURE_STRETCH.values())
_EXPORT_METRICS = (
    ("LOCATION", "Location", ("LOCATION_X", "LOCATION_Y", "LOCATION_Z")),
    ("VOLUME", "Volume", ("VOLUME",)),
    ("SURFACE_AREA", "Surface Area", ("SURFACE_AREA",)),
    ("ROD_LENGTH", "Rod Length", ("ROD_LENGTH",)),
    ("VELOCITY", "Velocity", ("VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z")),
    ("SPEED", "Speed", ("SPEED",)),
    (
        "ACCELERATION",
        "Acceleration",
        ("ACCELERATION_X", "ACCELERATION_Y", "ACCELERATION_Z"),
    ),
    (
        "ACCELERATION_MAGNITUDE",
        "Acceleration Magnitude",
        ("ACCELERATION_MAGNITUDE",),
    ),
    (
        "ANGULAR_VELOCITY",
        "Angular Velocity",
        ("ANGULAR_VELOCITY_X", "ANGULAR_VELOCITY_Y", "ANGULAR_VELOCITY_Z"),
    ),
    ("ANGULAR_SPEED", "Angular Speed", ("ANGULAR_SPEED",)),
    (
        "ANGULAR_AXIS",
        "Angular Axis",
        ("ANGULAR_AXIS_X", "ANGULAR_AXIS_Y", "ANGULAR_AXIS_Z"),
    ),
    ("CONTACT_COUNT", "Contact Count", ("CONTACT_COUNT",)),
)
_EXPORT_CHANNELS = {
    identifier: channels for identifier, _label, channels in _EXPORT_METRICS
}


def _format_vector(record, channels):
    values = [scalar_value(record, channel) for channel in channels]
    if any(value is None for value in values):
        return iface_("N/A")
    formatted = ", ".join(f"{float(value):.3g}" for value in values)
    return f"[{formatted}]"


def _format_measure(record, channel_id):
    value = scalar_value(record, channel_id)
    if value is None:
        return iface_("N/A")
    text = f"{float(value):.3g}"
    stretch = scalar_value(record, _MEASURE_STRETCH[channel_id])
    if stretch is not None:
        text += f" ({float(stretch) * 100.0:.3g}%)"
    return text


@dynamic_enum_items
def _statistics_object_items(_self, context):
    items = []
    for number, (group, assigned, obj) in enumerate(
        _assigned_objects(context.scene)
    ):
        object_name = obj.name if obj is not None else assigned.name
        group_name = group.name or iface_("Dynamics Group")
        items.append((
            assigned.uuid,
            object_name,
            iface_("{object} in {group}").format(
                object=object_name, group=group_name
            ),
            "NONE",
            number,
        ))
    if not items:
        return [("NONE", iface_("(No Dynamics Objects)"), "", "ERROR", 1000)]
    return items


class SCENE_OT_SelectStatisticsObject(Operator):
    bl_idname = "scene.select_statistics_object"
    bl_label = "Select Statistics Object"
    bl_options = {"INTERNAL"}

    object_uuid: EnumProperty(
        name="Object",
        items=_statistics_object_items,
        options={"SKIP_SAVE"},
    )  # pyright: ignore

    def execute(self, context):
        if self.object_uuid == "NONE":
            return {"CANCELLED"}
        get_addon_data(context.scene).state.statistics_object_uuid = self.object_uuid
        return {"FINISHED"}


@dynamic_enum_items
def _statistics_channel_items(self, context):
    object_uuid = self.object_uuid or _selected_statistics_uuid(context)
    obj = manifest_object(object_uuid)
    if obj is None:
        return [("NONE", iface_("(No Statistics)"), "", "ERROR", 1000)]
    supported = obj["supported_channels"]
    items = []
    for number, (identifier, label, channels) in enumerate(_EXPORT_METRICS):
        if not all(
            supported & (1 << CHANNEL_BY_ID[channel][3])
            for channel in channels
        ):
            continue
        unit = CHANNEL_BY_ID[channels[0]][2]
        items.append((
            identifier,
            _label_with_unit(label, unit),
            iface_("Export {label} for the selected object").format(label=label),
            "NONE",
            number,
        ))
    if items:
        return items
    return [("NONE", iface_("(No Statistics)"), "", "ERROR", 1000)]


def _selected_statistics_uuid(context):
    state = get_addon_data(context.scene).state
    objects = list(_assigned_objects(context.scene))
    object_uuids = {assigned.uuid for _group, assigned, _obj in objects}
    if state.statistics_object_uuid in object_uuids:
        return state.statistics_object_uuid
    return objects[0][1].uuid if objects else ""


class SOLVER_OT_ExportStatisticsCSV(Operator, ExportHelper):
    bl_idname = "solver.export_statistics_csv"
    bl_label = "Export Statistics CSV"
    bl_options = {"REGISTER"}

    filename_ext = ".csv"
    filter_glob: StringProperty(default="*.csv", options={"HIDDEN"})  # pyright: ignore
    object_uuid: StringProperty(options={"HIDDEN"})  # pyright: ignore
    channel: EnumProperty(name="Value", items=_statistics_channel_items)  # pyright: ignore

    def invoke(self, context, event):
        if not self.object_uuid:
            self.object_uuid = _selected_statistics_uuid(context)
        if self.channel == "NONE":
            self.report({"ERROR"}, iface_("No statistics channel is available"))
            return {"CANCELLED"}
        obj = get_object_by_uuid(self.object_uuid)
        object_name = obj.name if obj is not None else self.object_uuid
        safe_name = object_name.replace(" ", "_").replace("/", "_")
        self.filepath = f"{safe_name}_{self.channel.lower()}.csv"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        if not self.object_uuid:
            self.object_uuid = _selected_statistics_uuid(context)
        if manifest_object(self.object_uuid) is None:
            self.report({"ERROR"}, iface_("Statistics are unavailable for this object"))
            return {"CANCELLED"}
        start_frame = resolve_start_frame(get_addon_data(context.scene).state)
        path = bpy.path.abspath(self.filepath)
        try:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, lineterminator="\n")
                writer.writerow(("frame", "time_s", "value"))
                channels = _EXPORT_CHANNELS[self.channel]
                rows = [
                    list(iter_scalar_records(self.object_uuid, channel))
                    for channel in channels
                ]
                for entries in zip(*rows, strict=True):
                    solver_frame = entries[0][0]
                    time_seconds = entries[0][1]
                    if any(
                        entry[0] != solver_frame or entry[1] != time_seconds
                        for entry in entries[1:]
                    ):
                        raise StatisticsCacheError(
                            "statistics vector channels are not frame-aligned"
                        )
                    values = [entry[2] for entry in entries]
                    if any(value is None for value in values):
                        value = ""
                    elif len(values) == 1:
                        value = values[0]
                    else:
                        value = "[" + ",".join(
                            f"{float(component):.9g}" for component in values
                        ) + "]"
                    writer.writerow((
                        solver_frame + start_frame,
                        f"{time_seconds:.9g}",
                        value,
                    ))
        except (OSError, StatisticsCacheError) as exc:
            self.report({"ERROR"}, iface_("CSV export failed: {error}").format(error=exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            iface_("Exported statistics to {path}").format(path=os.path.abspath(path)),
        )
        return {"FINISHED"}


class STATISTICS_PT_Statistics(Panel):
    bl_label = "Object Statistics"
    bl_idname = "STATISTICS_PT_Statistics"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 110

    @classmethod
    def poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        layout = self.layout
        state = get_addon_data(context.scene).state
        manifest = load_manifest()

        objects = list(_assigned_objects(context.scene))
        if not objects:
            layout.label(text=iface_("No dynamics objects"), icon="INFO")
            return

        object_uuid = state.statistics_object_uuid
        object_uuids = {assigned.uuid for _group, assigned, _obj in objects}
        if object_uuid not in object_uuids:
            object_uuid = objects[0][1].uuid
        selected_obj = next(
            (
                obj
                for _group, assigned, obj in objects
                if assigned.uuid == object_uuid
            ),
            None,
        )
        object_name = selected_obj.name if selected_obj is not None else object_uuid
        row = layout.row()
        row.label(text=iface_("Object"))
        row.operator_menu_enum(
            SCENE_OT_SelectStatisticsObject.bl_idname,
            "object_uuid",
            text=object_name,
            icon="DOWNARROW_HLT",
        )

        selected_manifest = manifest_object(object_uuid, manifest)
        if selected_manifest is None:
            layout.label(
                text=iface_("Statistics unavailable; rerun the simulation"),
                icon="INFO",
            )
            return

        solver_frame = context.scene.frame_current - resolve_start_frame(state)
        try:
            record = read_record(object_uuid, solver_frame)
        except StatisticsCacheError as exc:
            layout.label(text=str(exc), icon="ERROR")
            return

        details = layout.box()
        supported = selected_manifest["supported_channels"]
        col = details.column(align=True)
        row = col.row(align=True)
        row.label(text=iface_("Frame"))
        row.label(text=str(context.scene.frame_current))
        row = col.row(align=True)
        row.label(text=iface_("Time (s)"))
        row.label(
            text=(
                f"{float(record['time_seconds']):.3g}"
                if record is not None
                else iface_("N/A")
            )
        )
        for label, channels in _VECTOR_ROWS:
            if not any(
                supported & (1 << CHANNEL_BY_ID[channel][3])
                for channel in channels
            ):
                continue
            row = col.row(align=True)
            row.label(
                text=_label_with_unit(label, CHANNEL_BY_ID[channels[0]][2])
            )
            row.label(text=_format_vector(record, channels))
        for channel_id, label, _unit, bit in CHANNELS:
            if (
                channel_id in _VECTOR_CHANNELS
                or channel_id in _COMBINED_STRETCH_CHANNELS
                or supported & (1 << bit) == 0
            ):
                continue
            row = col.row(align=True)
            row.label(text=_label_with_unit(label, _unit))
            row.label(
                text=(
                    _format_measure(record, channel_id)
                    if channel_id in _MEASURE_STRETCH
                    else _format_value(
                        scalar_value(record, channel_id), channel_id
                    )
                )
            )
        if supported & (1 << CHANNEL_BY_ID["CONTACT_COUNT"][3]) == 0:
            row = col.row(align=True)
            row.label(text=iface_("Contact Count"))
            row.label(text=iface_("N/A"))

        export_row = layout.row()
        export_row.enabled = record is not None
        op = export_row.operator_menu_enum(
            SOLVER_OT_ExportStatisticsCSV.bl_idname,
            "channel",
            text=iface_("Export CSV"),
            icon="EXPORT",
        )
        op.object_uuid = object_uuid


classes = (
    SCENE_OT_SelectStatisticsObject,
    SOLVER_OT_ExportStatisticsCSV,
    STATISTICS_PT_Statistics,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    for handler in list(bpy.app.handlers.frame_change_post):
        if getattr(handler, "__name__", "") == "statistics_frame_change_handler":
            bpy.app.handlers.frame_change_post.remove(handler)
    bpy.app.handlers.frame_change_post.append(statistics_frame_change_handler)


def unregister():
    for handler in list(bpy.app.handlers.frame_change_post):
        if getattr(handler, "__name__", "") == "statistics_frame_change_handler":
            bpy.app.handlers.frame_change_post.remove(handler)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
