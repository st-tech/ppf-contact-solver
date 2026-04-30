# File: invisible_collider_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
from bpy.props import EnumProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ...models.collection_utils import (
    generate_unique_name,
    safe_update_index,
    sort_keyframes_by_frame,
)
from ...models.groups import get_addon_data, invalidate_overlays
from ...core.utils import redraw_all_areas


COLLIDER_TYPE_ITEMS = [
    ("WALL", "Wall", "Infinite plane collider"),
    ("SPHERE", "Sphere", "Sphere collider"),
]


def _next_collider_name(state, collider_type):
    """Generate auto-name like 'Wall 1', 'Sphere 2'."""
    prefix = "Wall" if collider_type == "WALL" else "Sphere"
    existing = [
        c.name for c in state.invisible_colliders
        if c.collider_type == collider_type
    ]
    return generate_unique_name(prefix, existing)


class SCENE_OT_AddInvisibleCollider(Operator):
    """Add an invisible collider"""

    bl_idname = "scene.add_invisible_collider"
    bl_label = "Add Invisible Collider"
    bl_options = {"UNDO"}

    collider_type: EnumProperty(items=COLLIDER_TYPE_ITEMS)  # pyright: ignore

    def execute(self, context):
        state = get_addon_data(context.scene).state
        item = state.invisible_colliders.add()
        item.collider_type = self.collider_type
        item.name = _next_collider_name(state, self.collider_type)
        item.show_preview = True
        # Add initial keyframe at frame 1
        kf = item.keyframes.add()
        kf.frame = 1
        state.invisible_colliders_index = len(state.invisible_colliders) - 1
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_RemoveInvisibleCollider(Operator):
    """Remove the selected invisible collider"""

    bl_idname = "scene.remove_invisible_collider"
    bl_label = "Remove Invisible Collider"
    bl_options = {"UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        idx = state.invisible_colliders_index
        if idx < 0 or idx >= len(state.invisible_colliders):
            return {"CANCELLED"}
        state.invisible_colliders.remove(idx)
        state.invisible_colliders_index = safe_update_index(idx, len(state.invisible_colliders))
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_AddColliderKeyframe(Operator):
    """Add a keyframe to the selected invisible collider"""

    bl_idname = "scene.add_collider_keyframe"
    bl_label = "Add Keyframe"
    bl_options = {"UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        idx = state.invisible_colliders_index
        if idx < 0 or idx >= len(state.invisible_colliders):
            return {"CANCELLED"}
        item = state.invisible_colliders[idx]
        current_frame = max(1, context.scene.frame_current)
        # Reject duplicate frames
        for kf in item.keyframes:
            if kf.frame == current_frame:
                self.report({"WARNING"}, f"Keyframe at frame {current_frame} already exists")
                return {"CANCELLED"}
        kf = item.keyframes.add()
        kf.frame = current_frame
        kf.position = tuple(item.position)
        kf.radius = item.radius
        new_idx = sort_keyframes_by_frame(item.keyframes)
        item.keyframes_index = new_idx
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_RemoveColliderKeyframe(Operator):
    """Remove the selected keyframe (cannot remove the initial keyframe)"""

    bl_idname = "scene.remove_collider_keyframe"
    bl_label = "Remove Keyframe"
    bl_options = {"UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        idx = state.invisible_colliders_index
        if idx < 0 or idx >= len(state.invisible_colliders):
            return {"CANCELLED"}
        item = state.invisible_colliders[idx]
        kf_idx = item.keyframes_index
        if kf_idx <= 0:
            self.report({"WARNING"}, "Cannot remove the initial keyframe")
            return {"CANCELLED"}
        if kf_idx >= len(item.keyframes):
            return {"CANCELLED"}
        item.keyframes.remove(kf_idx)
        item.keyframes_index = safe_update_index(kf_idx, len(item.keyframes))
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    SCENE_OT_AddInvisibleCollider,
    SCENE_OT_RemoveInvisibleCollider,
    SCENE_OT_AddColliderKeyframe,
    SCENE_OT_RemoveColliderKeyframe,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
