# File: velocity_keyframe_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
from bpy.props import BoolProperty, PointerProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ...core.param_introspect import copy_scalar_props
from ...core.utils import redraw_all_areas
from ...models.collection_utils import safe_update_index, sort_keyframes_by_frame
from ...models.groups import invalidate_overlays
from .utils import get_assigned_by_selection_uuid, get_group_from_index

# VelocityKeyframe.preview is a per-keyframe viewport toggle, not
# simulation state — don't shuttle it across objects.
_VELOCITY_KF_CLIPBOARD_EXCLUDE = frozenset({"preview"})


def _copy_velocity_keyframes_collection(src_kfs, dst_kfs):
    """Replace *dst_kfs* with a scalar-wise copy of *src_kfs*."""
    dst_kfs.clear()
    for src_kf in src_kfs:
        new_kf = dst_kfs.add()
        copy_scalar_props(src_kf, new_kf, exclude=_VELOCITY_KF_CLIPBOARD_EXCLUDE)


def _get_selected_assigned(group):
    return get_assigned_by_selection_uuid(group, "velocity_object_selection")


class OBJECT_OT_AddVelocityKeyframe(Operator):
    """Add a velocity keyframe at the current frame"""

    bl_idname = "object.add_velocity_keyframe"
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

        frame = max(1, context.scene.frame_current)

        for kf in assigned.velocity_keyframes:
            if kf.frame == frame:
                self.report({"WARNING"}, f"Frame {frame} already has a keyframe")
                return {"CANCELLED"}

        item = assigned.velocity_keyframes.add()
        item.frame = frame
        assigned.velocity_keyframes_index = sort_keyframes_by_frame(
            assigned.velocity_keyframes
        )

        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_RemoveVelocityKeyframe(Operator):
    """Remove the selected velocity keyframe"""

    bl_idname = "object.remove_velocity_keyframe"
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

        idx = assigned.velocity_keyframes_index
        if 0 <= idx < len(assigned.velocity_keyframes):
            assigned.velocity_keyframes.remove(idx)
            assigned.velocity_keyframes_index = safe_update_index(
                idx, len(assigned.velocity_keyframes)
            )

        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_CopyVelocityKeyframes(Operator):
    """Copy velocity keyframes of the selected object to clipboard"""

    bl_idname = "object.copy_velocity_keyframes"
    bl_label = "Copy Velocity Keyframes"

    group_index: bpy.props.IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        assigned = _get_selected_assigned(group)
        if assigned is None:
            self.report({"WARNING"}, "No object selected")
            return {"CANCELLED"}

        wm = context.window_manager
        _copy_velocity_keyframes_collection(
            assigned.velocity_keyframes,
            wm.velocity_keyframes_clipboard.velocity_keyframes,
        )
        wm.velocity_keyframes_clipboard_valid = True
        self.report(
            {"INFO"}, f"Copied {len(assigned.velocity_keyframes)} velocity keyframe(s)"
        )
        return {"FINISHED"}


class OBJECT_OT_PasteVelocityKeyframes(Operator):
    """Paste velocity keyframes from clipboard, replacing existing ones"""

    bl_idname = "object.paste_velocity_keyframes"
    bl_label = "Paste Velocity Keyframes"
    bl_options = {"REGISTER", "UNDO"}

    group_index: bpy.props.IntProperty(options={"HIDDEN"})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return getattr(context.window_manager, "velocity_keyframes_clipboard_valid", False)

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            self.report({"ERROR"}, "Group not found")
            return {"CANCELLED"}

        assigned = _get_selected_assigned(group)
        if assigned is None:
            self.report({"WARNING"}, "No object selected")
            return {"CANCELLED"}

        wm = context.window_manager
        _copy_velocity_keyframes_collection(
            wm.velocity_keyframes_clipboard.velocity_keyframes,
            assigned.velocity_keyframes,
        )
        assigned.velocity_keyframes_index = sort_keyframes_by_frame(
            assigned.velocity_keyframes
        )
        invalidate_overlays()
        redraw_all_areas(context)
        self.report(
            {"INFO"}, f"Pasted {len(assigned.velocity_keyframes)} velocity keyframe(s)"
        )
        return {"FINISHED"}


classes = (
    OBJECT_OT_AddVelocityKeyframe,
    OBJECT_OT_RemoveVelocityKeyframe,
    OBJECT_OT_CopyVelocityKeyframes,
    OBJECT_OT_PasteVelocityKeyframes,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # Clipboard is a PointerProperty to AssignedObject so we reuse the
    # existing VelocityKeyframe schema — any future VelocityKeyframe
    # field flows through copy/paste via bl_rna introspection.
    from ..state_types import AssignedObject
    bpy.types.WindowManager.velocity_keyframes_clipboard = PointerProperty(type=AssignedObject)
    bpy.types.WindowManager.velocity_keyframes_clipboard_valid = BoolProperty(default=False)


def unregister():
    for attr in (
        "velocity_keyframes_clipboard",
        "velocity_keyframes_clipboard_valid",
    ):
        if hasattr(bpy.types.WindowManager, attr):
            delattr(bpy.types.WindowManager, attr)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
