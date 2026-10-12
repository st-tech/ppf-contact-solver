# File: dynamics/material_map_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Add and remove operators for a group's spatial material maps.

import bpy  # pyright: ignore
from bpy.props import IntProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_  # pyright: ignore

from ...models.collection_utils import (
    safe_update_index,
    sort_keyframes_by_frame,
    validate_no_duplicate_frame,
)


def _group_from_slot(context, slot: int):
    """The group in `object_group_<slot>`, addressed by SLOT.

    A group's slot is not its `index`: the index numbers active groups
    consecutively for display, and the two diverge once any group is deleted.
    """
    root = context.scene.zozo_contact_solver
    return getattr(root, f"object_group_{slot}", None)


def _holds_only_hair(group) -> bool:
    """True when `group` holds objects and every one is a Hair Curves object."""
    from ...core.hair_curves import is_hair_curves
    from ...core.uuid_registry import get_object_by_uuid

    objects = [
        get_object_by_uuid(assigned.uuid)
        for assigned in group.assigned_objects
        if assigned.uuid
    ]
    return bool(objects) and all(is_hair_curves(obj) for obj in objects)


class OBJECT_OT_AddMaterialMap(Operator):
    """Add a spatial material map to this group"""

    bl_idname = "object.ppf_add_material_map"
    bl_label = "Add Material Map"
    bl_options = {"REGISTER", "UNDO"}

    slot: IntProperty(default=0, options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        group = _group_from_slot(context, self.slot)
        if group is None:
            self.report({"ERROR"}, "No group in that slot")
            return {"CANCELLED"}
        entry = group.material_maps.add()
        if _holds_only_hair(group):
            # Hair Curves have no vertex groups, so the one source such a
            # group can read is a point attribute.
            entry.source_type = "ATTRIBUTE"
        group.material_maps_index = len(group.material_maps) - 1
        return {"FINISHED"}


class OBJECT_OT_RemoveMaterialMap(Operator):
    """Remove the selected spatial material map"""

    bl_idname = "object.ppf_remove_material_map"
    bl_label = "Remove Material Map"
    bl_options = {"REGISTER", "UNDO"}

    slot: IntProperty(default=0, options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        group = _group_from_slot(context, self.slot)
        if group is None:
            self.report({"ERROR"}, "No group in that slot")
            return {"CANCELLED"}
        i = group.material_maps_index
        if not (0 <= i < len(group.material_maps)):
            self.report({"ERROR"}, "No material map selected")
            return {"CANCELLED"}
        group.material_maps.remove(i)
        # Keep the selection on a real row rather than one past the end, and on
        # row 0 rather than -1 once the last row is gone.
        group.material_maps_index = safe_update_index(i, len(group.material_maps))
        return {"FINISHED"}


def _selected_map(context, slot: int):
    """The selected map on the group in `slot`, or None."""
    group = _group_from_slot(context, slot)
    if group is None:
        return None
    i = group.material_maps_index
    if not (0 <= i < len(group.material_maps)):
        return None
    return group.material_maps[i]


class OBJECT_OT_AddMaterialMapSample(Operator):
    """Add a later weight source to the selected material map"""

    bl_idname = "object.ppf_add_material_map_sample"
    bl_label = "Add Map Sample"
    bl_options = {"REGISTER", "UNDO"}

    slot: IntProperty(default=0, options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        entry = _selected_map(context, self.slot)
        if entry is None:
            self.report({"ERROR"}, "No material map selected")
            return {"CANCELLED"}
        # A sample at or before the start frame is refused at encode: the map's
        # own source already IS the weights at the start frame. Seeding the
        # playhead blindly makes the very first click author a row the encoder
        # will reject, so it is moved and reported instead.
        from ...core.encoder import resolve_start_frame
        from ...models.groups import get_addon_data
        start_frame = resolve_start_frame(get_addon_data(context.scene).state)
        frame = int(context.scene.frame_current)
        if frame <= start_frame:
            frame = start_frame + 1
            self.report(
                {"INFO"},
                iface_(
                    "Sample placed at frame {frame}: the map's own source is "
                    "already the weights at the start frame."
                ).format(frame=frame),
            )
        try:
            validate_no_duplicate_frame(entry.samples, frame)
        except ValueError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        sample = entry.samples.add()
        sample.frame = frame
        sample.source_type = entry.source_type
        entry.samples_index = sort_keyframes_by_frame(entry.samples)
        return {"FINISHED"}


class OBJECT_OT_RemoveMaterialMapSample(Operator):
    """Remove the selected weight source from the material map"""

    bl_idname = "object.ppf_remove_material_map_sample"
    bl_label = "Remove Map Sample"
    bl_options = {"REGISTER", "UNDO"}

    slot: IntProperty(default=0, options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        entry = _selected_map(context, self.slot)
        if entry is None:
            self.report({"ERROR"}, "No material map selected")
            return {"CANCELLED"}
        i = entry.samples_index
        if not (0 <= i < len(entry.samples)):
            self.report({"ERROR"}, "No map sample selected")
            return {"CANCELLED"}
        entry.samples.remove(i)
        entry.samples_index = safe_update_index(i, len(entry.samples))
        return {"FINISHED"}


classes = (
    OBJECT_OT_AddMaterialMap,
    OBJECT_OT_RemoveMaterialMap,
    OBJECT_OT_AddMaterialMapSample,
    OBJECT_OT_RemoveMaterialMapSample,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
