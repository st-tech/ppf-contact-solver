# File: ui_lists.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from bpy.types import UIList  # pyright: ignore

from ..state import decode_vertex_group_identifier


class OBJECT_UL_AssignedObjectsList(UIList):
    """UI List for assigned objects"""

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_property, index
    ):
        # Determine icon based on group type
        group = data  # data is the group object
        if group.object_type == "SOLID":
            object_icon = "MESH_CUBE"
        elif group.object_type == "SHELL":
            object_icon = "OUTLINER_OB_SURFACE"
        elif group.object_type == "ROD":
            object_icon = "VIEW_ORTHO"
        elif group.object_type == "STATIC":
            object_icon = "OBJECT_ORIGIN"
        else:
            object_icon = "MESH_DATA"

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
                row.label(text=f"{item.name} (Missing)", icon="ERROR")
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
            if obj and vg_name:
                has_pin = False
                if obj.type == "MESH" and obj.vertex_groups.get(vg_name):
                    has_pin = True
                elif obj.type == "CURVE" and obj.get(f"_pin_{vg_name}"):
                    has_pin = True
                if has_pin:
                    row = layout.row()
                    row.label(text=f"[{obj.name}][{vg_name}]", icon="GROUP_VERTEX")
                else:
                    layout.label(
                        text=f"[{obj.name}][{vg_name}] (Missing)", icon="ERROR"
                    )
            else:
                layout.label(text=item.name, icon="ERROR")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="GROUP_VERTEX")


class OBJECT_UL_PinOperationsList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        row = layout.row(align=True)
        if item.op_type == "EMBEDDED_MOVE":
            row.label(text="[Embedded] Move", icon="KEYFRAME")
            return
        if item.op_type == "MOVE_BY":
            d = item.delta
            row.label(text=f"Move ({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f})", icon="ORIENTATION_LOCAL")
        elif item.op_type == "SPIN":
            row.label(text=f"Spin \u03c9={item.spin_angular_velocity:.0f}\u00b0/s", icon="DRIVER_ROTATIONAL_DIFFERENCE")
        elif item.op_type == "SCALE":
            row.label(text=f"Scale \u00d7{item.scale_factor:.2f}", icon="FULLSCREEN_ENTER")
        elif item.op_type == "TORQUE":
            row.label(text=f"Torque {item.torque_magnitude:.1f} N\u00b7m", icon="FORCE_MAGNETIC")
        eye_icon = "HIDE_OFF" if item.show_overlay else "HIDE_ON"
        row.prop(item, "show_overlay", text="", icon=eye_icon, emboss=False)


class OBJECT_UL_StaticOpsList(bpy.types.UIList):
    """UIList for UI-assigned static ops (move/spin/scale)."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        row = layout.row(align=True)
        if item.op_type == "MOVE_BY":
            d = item.delta
            row.label(
                text=f"Move ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}) "
                     f"[{item.frame_start}-{item.frame_end}]",
                icon="ORIENTATION_LOCAL",
            )
        elif item.op_type == "SPIN":
            row.label(
                text=f"Spin \u03c9={item.spin_angular_velocity:.0f}\u00b0/s "
                     f"[{item.frame_start}-{item.frame_end}]",
                icon="DRIVER_ROTATIONAL_DIFFERENCE",
            )
        elif item.op_type == "SCALE":
            row.label(
                text=f"Scale \u00d7{item.scale_factor:.2f} "
                     f"[{item.frame_start}-{item.frame_end}]",
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
            label_a = obj_a.name if obj_a else (item.object_a or "(missing)")
            label_b = obj_b.name if obj_b else (item.object_b or "(missing)")
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
            label = _DYN_PARAM_LABELS.get(item.param_type, item.param_type)
            n_kf = len(item.keyframes)
            layout.label(text=f"{label} ({n_kf} kf)", icon=ico)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="PREFERENCES")


class SCENE_UL_DynParamKeyframesList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            if index == 0:
                layout.label(text=f"Frame {item.frame} (Initial)", icon="DECORATE_KEYFRAME")
            else:
                layout.label(text=f"Frame {item.frame}", icon="KEYFRAME")
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
                    flags.append("Inv")
                if item.hemisphere:
                    flags.append("Hemi")
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
                layout.label(text=f"Frame {item.frame} (Initial)", icon="DECORATE_KEYFRAME")
            else:
                layout.label(text=f"Frame {item.frame}", icon="KEYFRAME")
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="KEYFRAME")


class OBJECT_UL_VelocityKeyframesList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            d = item.direction
            layout.label(
                text=f"Frame {item.frame}  ({item.speed:.1f} m/s  [{d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}])",
                icon="KEYFRAME",
            )
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="KEYFRAME")


class OBJECT_UL_CollisionWindowsList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.label(
                text=f"Frame {item.frame_start} - {item.frame_end}",
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
