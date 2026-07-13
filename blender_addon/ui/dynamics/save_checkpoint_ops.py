# File: save_checkpoint_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_  # pyright: ignore

from ...models.collection_utils import safe_update_index, sort_keyframes_by_frame
from ...models.groups import get_addon_data
from ...core.utils import redraw_all_areas


class SCENE_OT_AddSaveCheckpoint(Operator):
    """Add a save checkpoint at the current timeline frame"""

    bl_idname = "scene.add_save_checkpoint"
    bl_label = "Add"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        # Use the timeline's current frame; the artist scrubs to the frame
        # they want a checkpoint at and clicks Add. Clamp to the Blender
        # 1-based minimum so frame 0 / negatives never reach the encoder.
        frame = max(1, int(context.scene.frame_current))
        # Reject duplicates: the list is a set of frames, and a repeated
        # frame would only produce a redundant save at the same index.
        for item in state.save_checkpoint_frames:
            if item.frame == frame:
                self.report(
                    {"WARNING"},
                    iface_("Checkpoint at frame {frame} already exists").format(
                        frame=frame
                    ),
                )
                return {"CANCELLED"}
        item = state.save_checkpoint_frames.add()
        item.frame = frame
        # Keep the list ascending; sort_keyframes_by_frame bubbles the new
        # tail entry into place and returns its final index.
        new_idx = sort_keyframes_by_frame(state.save_checkpoint_frames)
        state.save_checkpoint_frames_index = new_idx
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_RemoveSaveCheckpoint(Operator):
    """Remove the selected save checkpoint"""

    bl_idname = "scene.remove_save_checkpoint"
    bl_label = "Remove"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        state = get_addon_data(context.scene).state
        idx = state.save_checkpoint_frames_index
        if idx < 0 or idx >= len(state.save_checkpoint_frames):
            return {"CANCELLED"}
        state.save_checkpoint_frames.remove(idx)
        state.save_checkpoint_frames_index = safe_update_index(
            idx, len(state.save_checkpoint_frames)
        )
        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    SCENE_OT_AddSaveCheckpoint,
    SCENE_OT_RemoveSaveCheckpoint,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
