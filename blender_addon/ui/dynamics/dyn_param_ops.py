# File: dyn_param_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
from bpy.props import EnumProperty, IntProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ...models.collection_utils import safe_update_index, sort_keyframes_by_frame
from ...models.groups import get_addon_data, invalidate_overlays
from ...core.utils import redraw_all_areas


DYN_PARAM_ITEMS = [
    ("GRAVITY", "Gravity", "Dynamic gravity"),
    ("WIND", "Wind", "Dynamic wind"),
    ("AIR_DENSITY", "Air Density", "Dynamic air density"),
    ("AIR_FRICTION", "Air Friction", "Dynamic air friction"),
    ("VERTEX_AIR_DAMP", "Vertex Air Damping", "Dynamic vertex air damping"),
]


class SCENE_OT_AddDynParam(Operator):
    """Add a dynamic scene parameter"""

    bl_idname = "scene.add_dyn_param"
    bl_label = "Add Dynamic Parameter"
    bl_options = {"REGISTER", "UNDO"}

    param_type: EnumProperty(items=DYN_PARAM_ITEMS)  # pyright: ignore

    def execute(self, context):
        state = get_addon_data(context.scene).state
        # Reject duplicates
        for item in state.dyn_params:
            if item.param_type == self.param_type:
                self.report({"WARNING"}, f"{self.param_type} already added")
                return {"CANCELLED"}
        # Create new dynamic param entry
        item = state.dyn_params.add()
        item.param_type = self.param_type
        # Add initial keyframe at frame 1
        kf = item.keyframes.add()
        kf.frame = 1
        state.dyn_params_index = len(state.dyn_params) - 1
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_RemoveDynParam(Operator):
    """Remove the selected dynamic scene parameter"""

    bl_idname = "scene.remove_dyn_param"
    bl_label = "Remove Dynamic Parameter"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        idx = state.dyn_params_index
        if idx < 0 or idx >= len(state.dyn_params):
            return {"CANCELLED"}
        state.dyn_params.remove(idx)
        state.dyn_params_index = safe_update_index(idx, len(state.dyn_params))
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_AddDynParamKeyframe(Operator):
    """Add a keyframe to the selected dynamic parameter at the current frame"""

    bl_idname = "scene.add_dyn_param_keyframe"
    bl_label = "Add Keyframe"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        idx = state.dyn_params_index
        if idx < 0 or idx >= len(state.dyn_params):
            return {"CANCELLED"}
        dyn_item = state.dyn_params[idx]
        current_frame = context.scene.frame_current
        if current_frame < 1:
            current_frame = 1
        # Reject duplicate frames
        for kf in dyn_item.keyframes:
            if kf.frame == current_frame:
                self.report({"WARNING"}, f"Keyframe at frame {current_frame} already exists")
                return {"CANCELLED"}
        # Add keyframe with default values from global params
        kf = dyn_item.keyframes.add()
        kf.frame = current_frame
        if dyn_item.param_type == "GRAVITY":
            kf.gravity_value = tuple(state.gravity_3d)
        elif dyn_item.param_type == "WIND":
            kf.wind_direction_value = tuple(state.wind_direction)
            kf.wind_strength_value = state.wind_strength
        elif dyn_item.param_type == "AIR_DENSITY":
            kf.scalar_value = state.air_density
        elif dyn_item.param_type == "AIR_FRICTION":
            kf.scalar_value = state.air_friction
        elif dyn_item.param_type == "VERTEX_AIR_DAMP":
            kf.scalar_value = state.vertex_air_damp
        new_idx = sort_keyframes_by_frame(dyn_item.keyframes)
        dyn_item.keyframes_index = new_idx
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_RemoveDynParamKeyframe(Operator):
    """Remove the selected keyframe (cannot remove the initial keyframe)"""

    bl_idname = "scene.remove_dyn_param_keyframe"
    bl_label = "Remove Keyframe"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        idx = state.dyn_params_index
        if idx < 0 or idx >= len(state.dyn_params):
            return {"CANCELLED"}
        dyn_item = state.dyn_params[idx]
        kf_idx = dyn_item.keyframes_index
        if kf_idx <= 0:
            self.report({"WARNING"}, "Cannot remove the initial keyframe")
            return {"CANCELLED"}
        if kf_idx >= len(dyn_item.keyframes):
            return {"CANCELLED"}
        dyn_item.keyframes.remove(kf_idx)
        dyn_item.keyframes_index = safe_update_index(kf_idx, len(dyn_item.keyframes))
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    SCENE_OT_AddDynParam,
    SCENE_OT_RemoveDynParam,
    SCENE_OT_AddDynParamKeyframe,
    SCENE_OT_RemoveDynParamKeyframe,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
