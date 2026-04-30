# File: merge_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore

from ..models.collection_utils import safe_update_index
from ..models.groups import get_addon_data
from ..models.groups import iterate_active_object_groups


def cleanup_stale_merge_pairs(scene):
    """Remove merge pairs where either object is not in any dynamics
    group, and clear stored cross_stitch JSON when the source mesh
    topology has changed since the pair was created."""
    import json

    from ..core.uuid_registry import get_object_by_uuid

    state = get_addon_data(scene).state
    assigned_uuids = set()
    removed = False
    invalidated = False
    for group in iterate_active_object_groups(scene):
        for obj_ref in group.assigned_objects:
            if obj_ref.uuid:
                assigned_uuids.add(obj_ref.uuid)
    for i in range(len(state.merge_pairs) - 1, -1, -1):
        pair = state.merge_pairs[i]
        if not pair.object_a_uuid or not pair.object_b_uuid:
            state.merge_pairs.remove(i)
            removed = True
            continue
        if pair.object_a_uuid not in assigned_uuids or pair.object_b_uuid not in assigned_uuids:
            state.merge_pairs.remove(i)
            removed = True
            continue
        if pair.cross_stitch_json:
            try:
                data = json.loads(pair.cross_stitch_json)
            except (ValueError, json.JSONDecodeError):
                pair.cross_stitch_json = ""
                invalidated = True
                continue
            a_obj = get_object_by_uuid(pair.object_a_uuid)
            b_obj = get_object_by_uuid(pair.object_b_uuid)
            expected_a = data.get("a_vert_count")
            expected_b = data.get("b_vert_count")
            if (
                (a_obj is not None and expected_a is not None
                 and len(a_obj.data.vertices) != expected_a)
                or (b_obj is not None and expected_b is not None
                    and len(b_obj.data.vertices) != expected_b)
            ):
                pair.cross_stitch_json = ""
                invalidated = True
    state.merge_pairs_index = safe_update_index(state.merge_pairs_index, len(state.merge_pairs))
    if removed or invalidated:
        from ..ui.dynamics.overlay import apply_object_overlays

        apply_object_overlays()


class OBJECT_OT_RemoveMergePair(Operator):
    """Remove the selected merge pair"""

    bl_idname = "object.remove_merge_pair"
    bl_label = "Remove"

    def execute(self, context):
        state = get_addon_data(context.scene).state
        index = state.merge_pairs_index
        if 0 <= index < len(state.merge_pairs):
            state.merge_pairs.remove(index)
            state.merge_pairs_index = safe_update_index(index, len(state.merge_pairs))
            from ..ui.dynamics.overlay import apply_object_overlays

            apply_object_overlays()
        return {"FINISHED"}


classes = (OBJECT_OT_RemoveMergePair,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
