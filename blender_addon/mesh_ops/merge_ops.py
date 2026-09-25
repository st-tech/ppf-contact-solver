# File: merge_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from bpy.types import Operator  # pyright: ignore

from ..models.collection_utils import safe_update_index
from ..models.groups import get_addon_data
from ..models.groups import iterate_active_object_groups


def _member_types(scene) -> dict:
    """Every object UUID assigned to an active group, mapped to its group type."""
    types = {}
    for group in iterate_active_object_groups(scene):
        for obj_ref in group.assigned_objects:
            if obj_ref.uuid:
                types[obj_ref.uuid] = group.object_type
    return types


def pair_label(pair) -> str:
    """``A <-> B`` for a merge pair, by its objects' current names."""
    from ..core.uuid_registry import get_object_by_uuid

    def _name(uuid, stored):
        obj = get_object_by_uuid(uuid) if uuid else None
        return obj.name if obj is not None else (stored or "(missing)")

    return (
        f"{_name(pair.object_a_uuid, pair.object_a)} <-> "
        f"{_name(pair.object_b_uuid, pair.object_b)}"
    )


def merge_pair_problem(scene, pair, member_types=None):
    """Why *pair*'s stitch cannot reach the solver, or ``None`` when it can.

    THE ONE ANSWER for the Transfer check, the encoder and the panel, so none
    of them can pass a pair another drops. Nothing here writes to the pair: a
    stale or unassigned pair is REFUSED with a reason the artist can act on
    (Re-snap, or remove the pair) rather than cleared or deleted during an
    encode, which is what left a seam missing from the solve with nothing
    saying why. A pair is removed when an object's membership ends
    (``cleanup_group_references_for_object``), not by scanning for it later.
    """
    import json

    from ..core.uuid_registry import get_object_by_uuid

    if member_types is None:
        member_types = _member_types(scene)
    if not pair.object_a_uuid or not pair.object_b_uuid:
        return "one of its objects has no UUID; remove the pair and snap again"
    for uuid, stored in ((pair.object_a_uuid, pair.object_a),
                         (pair.object_b_uuid, pair.object_b)):
        if uuid not in member_types:
            obj = get_object_by_uuid(uuid)
            return (
                f"'{obj.name if obj is not None else stored}' is in no active "
                "dynamics group; add it back or remove the pair"
            )
    if not pair.cross_stitch_json:
        return "it has no stitch points; click Re-snap or remove the pair"
    try:
        data = json.loads(pair.cross_stitch_json)
    except (ValueError, json.JSONDecodeError):
        return "its stored stitch cannot be read; click Re-snap"
    if not isinstance(data, dict) or not data:
        return "it has no stitch points; click Re-snap or remove the pair"
    for uuid, key in ((pair.object_a_uuid, "a_vert_count"),
                      (pair.object_b_uuid, "b_vert_count")):
        expected = data.get(key)
        obj = get_object_by_uuid(uuid)
        if (expected is not None and obj is not None
                and obj.type == "MESH" and len(obj.data.vertices) != expected):
            return (
                f"the mesh of '{obj.name}' changed since the pair was snapped "
                f"({expected} vertices then, {len(obj.data.vertices)} now); "
                "click Re-snap"
            )
    source, target = data.get("source_uuid"), data.get("target_uuid")
    if not source or not target:
        return "its stored stitch names no source or target; click Re-snap"
    if source not in member_types or target not in member_types:
        return "its stored stitch names an object outside the pair; click Re-snap"
    ind, w = data.get("ind") or [], data.get("w") or []
    if not ind or not w:
        return "it has no stitch points; click Re-snap or remove the pair"
    if len(ind) != len(w):
        return "its stored stitch has unequal index and weight rows; click Re-snap"
    width = len(ind[0])
    if width not in (4, 6) or any(
        len(r) != width for r in ind
    ) or any(len(r) != width for r in w):
        return "its stored stitch rows are malformed; click Re-snap"
    for uuid, key in ((source, "source_points"), (target, "target_points")):
        if member_types.get(uuid) != "SOLID":
            continue
        # A SOLID side is placed on its tetrahedral surface from the points
        # recorded at snap time, which the Blender indices cannot replace.
        points = data.get(key) or []
        if width == 4 or len(points) != len(ind):
            return (
                "it was snapped before a SOLID side recorded its stitch points, "
                "so that side cannot be placed on the tetrahedral surface; "
                "click Re-snap"
            )
    return None


def pair_stitch_row_count(pair) -> int:
    """The number of stitch rows a merge pair's stored stitch carries.

    For the panel's count. Whether those rows can ship is
    :func:`merge_pair_problem`'s question.
    """
    import json

    if not pair.cross_stitch_json:
        return 0
    try:
        data = json.loads(pair.cross_stitch_json)
    except (ValueError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    ind = data.get("ind")
    return len(ind) if ind else 0


def pair_has_stitch(pair) -> bool:
    """True when a merge pair carries at least one usable cross-stitch row."""
    return pair_stitch_row_count(pair) > 0


class OBJECT_OT_RemoveMergePair(Operator):
    """Remove the selected merge pair"""

    bl_idname = "object.remove_merge_pair"
    bl_label = "Remove"
    bl_options = {"REGISTER", "UNDO"}

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
