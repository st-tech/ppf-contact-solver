# File: static_ops.py
# License: Apache v2.0
#
# Add/remove/reorder operators for UI-assigned static ops on moving
# static objects. Mirrors the pin-op pattern: ops live on the
# AssignedObject currently selected in the assigned-objects UIList.

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ...core.utils import has_transform_fcurves, redraw_all_areas
from ...models.collection_utils import safe_update_index
from ...models.groups import invalidate_overlays
from .utils import get_group_from_index


def _get_selected_assigned(group):
    """Return the AssignedObject at group.assigned_objects_index, or None."""
    idx = group.assigned_objects_index
    if 0 <= idx < len(group.assigned_objects):
        return group.assigned_objects[idx]
    return None


class OBJECT_OT_AddStaticOp(Operator):
    """Add a move/spin/scale op to the selected static object"""

    bl_idname = "object.add_static_op"
    bl_label = "Add Static Op"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={"HIDDEN"})  # pyright: ignore
    op_type: bpy.props.EnumProperty(
        items=[
            ("MOVE_BY", "Move By", "Translate by a delta over a time range"),
            ("SPIN", "Spin", "Rotate around an axis over a time range"),
            ("SCALE", "Scale", "Scale from a center over a time range"),
        ],
    )  # pyright: ignore

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None or group.object_type != "STATIC":
            return {"CANCELLED"}
        assigned = _get_selected_assigned(group)
        if assigned is None:
            self.report({"ERROR"}, "No assigned object selected")
            return {"CANCELLED"}

        from ...core.uuid_registry import get_object_by_uuid
        obj = get_object_by_uuid(assigned.uuid) if assigned.uuid else None
        if obj is not None and has_transform_fcurves(obj):
            # UI and warning label tell the user the op will be ignored;
            # allow adding so the data survives removing the fcurves.
            self.report(
                {"WARNING"},
                "Object has Blender keyframes — ops will be ignored at simulate time",
            )

        op = assigned.static_ops.add()
        op.op_type = self.op_type
        assigned.static_ops.move(len(assigned.static_ops) - 1, 0)
        assigned.static_ops_index = 0
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_RemoveStaticOp(Operator):
    """Remove the selected static op"""

    bl_idname = "object.remove_static_op"
    bl_label = "Remove Static Op"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        assigned = _get_selected_assigned(group)
        if assigned is None:
            return {"CANCELLED"}
        idx = assigned.static_ops_index
        if idx < 0 or idx >= len(assigned.static_ops):
            return {"CANCELLED"}
        assigned.static_ops.remove(idx)
        assigned.static_ops_index = safe_update_index(idx, len(assigned.static_ops))
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_MoveStaticOp(Operator):
    """Reorder the selected static op"""

    bl_idname = "object.move_static_op"
    bl_label = "Move Static Op"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={"HIDDEN"})  # pyright: ignore
    direction: bpy.props.IntProperty()  # pyright: ignore  # -1 up, +1 down

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        assigned = _get_selected_assigned(group)
        if assigned is None:
            return {"CANCELLED"}
        idx = assigned.static_ops_index
        new_idx = idx + self.direction
        if idx < 0 or idx >= len(assigned.static_ops):
            return {"CANCELLED"}
        if new_idx < 0 or new_idx >= len(assigned.static_ops):
            return {"CANCELLED"}
        assigned.static_ops.move(idx, new_idx)
        assigned.static_ops_index = new_idx
        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    OBJECT_OT_AddStaticOp,
    OBJECT_OT_RemoveStaticOp,
    OBJECT_OT_MoveStaticOp,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
