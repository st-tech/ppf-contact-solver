# File: collision_window_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ...core.utils import redraw_all_areas
from ...models.collection_utils import safe_update_index
from .utils import get_assigned_by_selection_uuid, get_group_from_index

MAX_COLLISION_WINDOWS = 8


def _get_selected_assigned(group):
    return get_assigned_by_selection_uuid(group, "collision_window_object_selection")


class OBJECT_OT_AddCollisionWindow(Operator):
    """Add a collision active window"""

    bl_idname = "object.add_collision_window"
    bl_label = "Add"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        assigned = _get_selected_assigned(group)
        if assigned is None:
            self.report({"ERROR"}, "No object selected")
            return {"CANCELLED"}

        if len(assigned.collision_windows) >= MAX_COLLISION_WINDOWS:
            self.report({"ERROR"}, f"Maximum {MAX_COLLISION_WINDOWS} collision windows per object")
            return {"CANCELLED"}

        item = assigned.collision_windows.add()
        item.frame_start = 1
        item.frame_end = 60
        # Keep list ordered so overlap/range checks on the encoder side get
        # entries in timeline order.
        new_idx = len(assigned.collision_windows) - 1
        assigned.collision_windows_index = new_idx

        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_RemoveCollisionWindow(Operator):
    """Remove the selected collision window"""

    bl_idname = "object.remove_collision_window"
    bl_label = "Remove"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={'HIDDEN'})  # pyright: ignore

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        assigned = _get_selected_assigned(group)
        if assigned is None:
            self.report({"ERROR"}, "No object selected")
            return {"CANCELLED"}

        idx = assigned.collision_windows_index
        if 0 <= idx < len(assigned.collision_windows):
            assigned.collision_windows.remove(idx)
            assigned.collision_windows_index = safe_update_index(
                idx, len(assigned.collision_windows)
            )

        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    OBJECT_OT_AddCollisionWindow,
    OBJECT_OT_RemoveCollisionWindow,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
