# File: ui_lists.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from bpy.types import UIList  # pyright: ignore

from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_

from ...models.groups import GROUP_TYPE_ICONS
from ..state import decode_vertex_group_identifier


class OBJECT_UL_AssignedObjectsList(UIList):
    """UI List for assigned objects"""

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_property, index
    ):
        # Determine icon based on group type
        group = data  # data is the group object
        object_icon = GROUP_TYPE_ICONS.get(group.object_type, {}).get(
            "object", "MESH_DATA"
        )

        if self.layout_type in {"DEFAULT", "COMPACT"}:
            from ...core.uuid_registry import get_object_by_uuid
            obj = get_object_by_uuid(item.uuid) if item.uuid else None
            if obj:
                row = layout.row()
                row.prop(item, "included", text="")
                row.prop(obj, "name", text="", emboss=False, icon=object_icon)
            else:
                row = layout.row()
                row.prop(item, "included", text="")
                row.label(text=iface_("{name} (Missing)").format(name=item.name), icon="ERROR")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon=object_icon)


class OBJECT_UL_PinVertexGroupsList(UIList):
    """UI List for pin vertex groups"""

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_property, index
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            from ...core.uuid_registry import resolve_pin, get_object_by_uuid
            # Resolve object + VG renames before drawing
            resolve_pin(item)
            _, vg_name = decode_vertex_group_identifier(item.name)
            obj = get_object_by_uuid(item.object_uuid) if item.object_uuid else None
            row = layout.row(align=True)
            if obj and vg_name:
                has_pin = False
                if obj.type == "MESH" and obj.vertex_groups.get(vg_name):
                    has_pin = True
                elif obj.type == "CURVE" and obj.get(f"_pin_{vg_name}"):
                    has_pin = True
                if has_pin:
                    row.label(text=f"[{obj.name}][{vg_name}]", icon="GROUP_VERTEX")
                else:
                    row.label(
                        text=iface_("[{name}][{group}] (Missing)").format(name=obj.name, group=vg_name), icon="ERROR"
                    )
            else:
                row.label(text=item.name, icon="ERROR")
            # Per-pin viewport visibility (eye), default open. Replaces the
            # former group-level "Show Pins" checkbox.
            row.prop(
                item, "show_overlay", text="", emboss=False,
                icon="HIDE_OFF" if item.show_overlay else "HIDE_ON",
            )
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="GROUP_VERTEX")


class OBJECT_UL_PinOperationsList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        row = layout.row(align=True)
        # EMBEDDED_MOVE is the sentinel that signals "this pin has
        # animation"; the encoder splices a per-vertex pin_anim track
        # at this op's index but never executes the op itself.
        #
        # Two motion sources reach this row:
        #   - Manual Make Keyframe writes vertex-co fcurves on the
        #     mesh action; the label stays ``[Embedded] Move``.
        #   - Capture Deformation writes a ``_pindeform.pc2`` cache;
        #     the label switches to ``[Embedded] Move (Captured)`` so
        #     the user can tell which path is live. The pin item's
        #     ``has_captured_anim`` flag is the source of truth.
        if item.op_type == "EMBEDDED_MOVE":
            pin_item = data
            if getattr(pin_item, "has_captured_anim", False):
                row.label(text="[Embedded] Move (Captured)", icon="KEYFRAME")
            else:
                row.label(text="[Embedded] Move", icon="KEYFRAME")
            return
        if item.op_type == "MOVE_BY":
            d = item.delta
            row.label(text=iface_("Move ({x:.1f}, {y:.1f}, {z:.1f})").format(x=d[0], y=d[1], z=d[2]), icon="ORIENTATION_LOCAL")
        elif item.op_type == "SPIN":
            row.label(text=iface_("Spin \u03c9={speed:.0f}\u00b0/s").format(speed=item.spin_angular_velocity), icon="DRIVER_ROTATIONAL_DIFFERENCE")
        elif item.op_type == "SCALE":
            row.label(text=iface_("Scale \u00d7{factor:.2f}").format(factor=item.scale_factor), icon="FULLSCREEN_ENTER")
        elif item.op_type == "TORQUE":
            row.label(text=iface_("Torque {magnitude:.1f} N\u00b7m").format(magnitude=item.torque_magnitude), icon="FORCE_MAGNETIC")
        eye_icon = "HIDE_OFF" if item.show_overlay else "HIDE_ON"
        row.prop(item, "show_overlay", text="", icon=eye_icon, emboss=False)


class OBJECT_UL_StaticOpsList(bpy.types.UIList):
    """UIList for UI-assigned static ops (move/spin/scale)."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        row = layout.row(align=True)
        if item.op_type == "MOVE_BY":
            d = item.delta
            row.label(
                text=iface_("Move ({x:.2f}, {y:.2f}, {z:.2f}) [{start}-{end}]").format(
                    x=d[0], y=d[1], z=d[2], start=item.frame_start, end=item.frame_end
                ),
                icon="ORIENTATION_LOCAL",
            )
        elif item.op_type == "SPIN":
            row.label(
                text=iface_("Spin \u03c9={speed:.0f}\u00b0/s [{start}-{end}]").format(
                    speed=item.spin_angular_velocity, start=item.frame_start, end=item.frame_end
                ),
                icon="DRIVER_ROTATIONAL_DIFFERENCE",
            )
        elif item.op_type == "SCALE":
            row.label(
                text=iface_("Scale \u00d7{factor:.2f} [{start}-{end}]").format(
                    factor=item.scale_factor, start=item.frame_start, end=item.frame_end
                ),
                icon="FULLSCREEN_ENTER",
            )
        eye_icon = "HIDE_OFF" if item.show_overlay else "HIDE_ON"
        row.prop(item, "show_overlay", text="", icon=eye_icon, emboss=False)


class OBJECT_UL_MergePairsList(bpy.types.UIList):
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_property, index
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            from ...core.uuid_registry import get_object_by_uuid
            row = layout.row(align=True)
            obj_a = get_object_by_uuid(item.object_a_uuid) if item.object_a_uuid else None
            obj_b = get_object_by_uuid(item.object_b_uuid) if item.object_b_uuid else None
            label_a = obj_a.name if obj_a else (item.object_a or iface_("(missing)"))
            label_b = obj_b.name if obj_b else (item.object_b or iface_("(missing)"))
            ico = "AUTOMERGE_ON" if (obj_a and obj_b) else "ERROR"
            row.label(text=f"{label_a} \u2194 {label_b}", icon=ico)
            eye_icon = "HIDE_OFF" if item.show_stitch else "HIDE_ON"
            row.prop(item, "show_stitch", text="", icon=eye_icon, emboss=False)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="AUTOMERGE_ON")


_DYN_PARAM_ICONS = {
    "GRAVITY": "FORCE_FORCE",
    "WIND": "FORCE_WIND",
    "AIR_DENSITY": "MOD_FLUID",
    "AIR_FRICTION": "FORCE_DRAG",
    "VERTEX_AIR_DAMP": "MOD_SMOOTH",
}

_DYN_PARAM_LABELS = {
    "GRAVITY": "Gravity",
    "WIND": "Wind",
    "AIR_DENSITY": "Air Density",
    "AIR_FRICTION": "Air Friction",
    "VERTEX_AIR_DAMP": "Vertex Air Damping",
}


class SCENE_UL_DynParamsList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            ico = _DYN_PARAM_ICONS.get(item.param_type, "PREFERENCES")
            label = iface_(_DYN_PARAM_LABELS.get(item.param_type, item.param_type))
            n_kf = len(item.keyframes)
            layout.label(text=iface_("{label} ({count} kf)").format(label=label, count=n_kf), icon=ico)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="PREFERENCES")


class SCENE_UL_DynParamKeyframesList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            if index == 0:
                layout.label(text=iface_("Frame {frame} (Initial)").format(frame=item.frame), icon="DECORATE_KEYFRAME")
            else:
                layout.label(text=iface_("Frame {frame}").format(frame=item.frame), icon="KEYFRAME")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="KEYFRAME")


class SOLVER_UL_CheckpointFrames(bpy.types.UIList):
    """Saved-checkpoint frames offered in the Resume-From dialog."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            # item.frame is the remote 0-based checkpoint index (kept raw for
            # the solver's --load on resume). Display it as the Blender 1-based
            # frame so it matches "Last Saved" in Scene Info.
            from ..main_panel import remote_frame_to_blender
            layout.label(
                text=iface_("Frame {frame}").format(frame=remote_frame_to_blender(item.frame)), icon="KEYFRAME"
            )
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="KEYFRAME")


class SCENE_UL_SaveCheckpointFrames(bpy.types.UIList):
    """User-requested frames at which the solver saves a resumable state."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            # item.frame is the Blender 1-based frame the artist entered, so
            # show it verbatim (the encoder converts to the solver's 0-based
            # index at upload time).
            layout.label(text=iface_("Frame {frame}").format(frame=item.frame), icon="KEYFRAME")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="KEYFRAME")


class SCENE_UL_InvisibleCollidersList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            if item.collider_type == "WALL":
                ico = "MESH_PLANE"
            else:
                ico = "MESH_UVSPHERE"
            label = item.name
            flags = []
            if item.collider_type == "SPHERE":
                if item.invert:
                    flags.append(iface_("Inv"))
                if item.hemisphere:
                    flags.append(iface_("Hemi"))
            if flags:
                label += f" ({', '.join(flags)})"
            row.label(text=label, icon=ico)
            eye_icon = "HIDE_OFF" if item.show_preview else "HIDE_ON"
            row.prop(item, "show_preview", text="", icon=eye_icon, emboss=False)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="MESH_UVSPHERE")


class SCENE_UL_ColliderKeyframesList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            if index == 0:
                layout.label(text=iface_("Frame {frame} (Initial)").format(frame=item.frame), icon="DECORATE_KEYFRAME")
            else:
                layout.label(text=iface_("Frame {frame}").format(frame=item.frame), icon="KEYFRAME")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="KEYFRAME")


class OBJECT_UL_VelocityKeyframesList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            d = item.direction
            spin = (
                f"  ω={item.angular_speed:.0f}°/s {item.angular_axis}"
                if item.enable_angular and item.angular_speed != 0.0
                else ""
            )
            layout.label(
                text=iface_("Frame {frame}  ({speed:.1f} m/s  [{x:.1f}, {y:.1f}, {z:.1f}]){spin}").format(
                    frame=item.frame, speed=item.speed, x=d[0], y=d[1], z=d[2], spin=spin
                ),
                icon="KEYFRAME",
            )
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="KEYFRAME")


class OBJECT_UL_CollisionWindowsList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.label(
                text=iface_("Frame {start} - {end}").format(start=item.frame_start, end=item.frame_end),
                icon="TIME",
            )
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="TIME")


classes = (
    OBJECT_UL_AssignedObjectsList,
    OBJECT_UL_PinVertexGroupsList,
    OBJECT_UL_PinOperationsList,
    OBJECT_UL_StaticOpsList,
    OBJECT_UL_MergePairsList,
    SCENE_UL_DynParamsList,
    SCENE_UL_DynParamKeyframesList,
    SOLVER_UL_CheckpointFrames,
    SCENE_UL_SaveCheckpointFrames,
    SCENE_UL_InvisibleCollidersList,
    SCENE_UL_ColliderKeyframesList,
    OBJECT_UL_VelocityKeyframesList,
    OBJECT_UL_CollisionWindowsList,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
