# File: animation.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ..models.groups import get_addon_data, iterate_active_object_groups
from .pc2 import cleanup_mesh_cache, has_mesh_cache
from .utils import _get_fcurves


def clear_animation_data(context, move_to_frame: bool = True):
    if move_to_frame:
        context.scene.frame_set(1)
    get_addon_data(context.scene).state.clear_fetched_frames()
    for group in iterate_active_object_groups(context.scene):
        for assigned in group.assigned_objects:
            from .uuid_registry import resolve_assigned
            obj = resolve_assigned(assigned)
            if not obj:
                continue
            if group.object_type == "STATIC":
                # UI-op static objects get MESH_CACHE when they're
                # animated via move/spin/scale ops; drop it here. Rest-
                # pose and fcurve-driven static objects have no cache,
                # so cleanup_mesh_cache is a no-op for them.
                if obj.type in ("MESH", "CURVE"):
                    cleanup_mesh_cache(obj)
                continue
            # Clear curve animation (MESH_CACHE + PC2)
            if obj.type == "CURVE":
                cleanup_mesh_cache(obj)
                continue
            if obj.type != "MESH":
                continue
            # Remove the output MESH_CACHE modifier and its PC2 file.
            # Keyframed-pin PC2 caches use a separate key and are left
            # intact — their playback handler keeps driving the mesh.
            cleanup_mesh_cache(obj)
    # Clear fetched frame tracking since keyframes were removed
    get_addon_data(context.scene).state.clear_fetched_frames()


def _iterate_dynamic_objects(scene):
    for group in iterate_active_object_groups(scene):
        if group.object_type == "STATIC":
            continue
        for assigned in group.assigned_objects:
            from .uuid_registry import resolve_assigned
            obj = resolve_assigned(assigned)
            if not obj:
                continue
            if obj.type in ("MESH", "CURVE"):
                yield obj


def _capture_object_geometry(obj):
    if obj.type == "MESH":
        return ("MESH", [vert.co.copy() for vert in obj.data.vertices])
    spline_snapshot = []
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            spline_snapshot.append(
                (
                    "BEZIER",
                    [
                        (
                            bp.co.copy(),
                            bp.handle_left.copy(),
                            bp.handle_right.copy(),
                            bp.handle_left_type,
                            bp.handle_right_type,
                        )
                        for bp in spline.bezier_points
                    ],
                )
            )
        else:
            spline_snapshot.append(("POINTS", [tuple(pt.co) for pt in spline.points]))
    return ("CURVE", spline_snapshot)


def _restore_object_geometry(obj, snapshot):
    obj_type, data = snapshot
    if obj_type == "MESH":
        for vert, co in zip(obj.data.vertices, data, strict=False):
            vert.co = co
        return
    for spline, spline_snapshot in zip(obj.data.splines, data, strict=False):
        snapshot_type, points = spline_snapshot
        if snapshot_type == "BEZIER":
            for bp, point in zip(spline.bezier_points, points, strict=False):
                co, hl, hr, hl_type, hr_type = point
                bp.co = co
                bp.handle_left = hl
                bp.handle_right = hr
                bp.handle_left_type = hl_type
                bp.handle_right_type = hr_type
        else:
            for pt, point in zip(spline.points, points, strict=False):
                pt.co = point


# Mesh ID-property that records "the current shape in
# ``vertices[i].co`` is the preserved frame-1 anchor". Written by
# ``_insert_frame_one_keys`` and read by ``_has_frame_one_geometry_keys``.
# Keying every pinned vertex's ``vertices[N].co`` here instead would
# fan out to hundreds of per-axis fcurves across several meshes and
# tip Blender 5.x's parallel animation evaluator into heap corruption
# on the next ``frame_set``. The pose already lives in the mesh's
# vertex buffer, so a single bool property carries the same signal
# with zero per-frame evaluator load.
_FRAME_ONE_MARKER = "_pcs_frame_one_preserved"


def _insert_frame_one_keys(obj, context):
    if obj.type == "MESH":
        # The pose was just placed into ``vertices[i].co`` by
        # ``_restore_object_geometry``; setting the marker tells
        # ``_has_frame_one_geometry_keys`` not to re-snapshot on the
        # next ``prepare_animation_targets`` pass. The pin encoder
        # gates on the ``EMBEDDED_MOVE`` sentinel for fcurve reads,
        # so it sees nothing here either way.
        obj.data[_FRAME_ONE_MARKER] = True
    # Curves use PC2; no frame-1 keyframes needed.


def _has_frame_one_geometry_keys(obj):
    # Objects with a MESH_CACHE modifier are considered "ready":
    # their geometry is already driven.
    if has_mesh_cache(obj):
        return True
    if obj.type == "MESH" and obj.data.get(_FRAME_ONE_MARKER):
        return True
    if not obj.data.animation_data or not obj.data.animation_data.action:
        return False
    action = obj.data.animation_data.action
    if obj.type == "MESH":
        path_prefixes = ("vertices[",)
    elif obj.type == "CURVE":
        path_prefixes = ("splines[",)
    else:
        return False
    # Curves anchor on ``splines[N].co`` fcurves (no heap-corruption
    # issue at this volume) and a marker-less mesh file may carry the
    # anchor as a frame-1 keyframe on each pinned vertex's ``co``
    # fcurve. Both are still recognized so the check is correct on
    # those layouts; the next ``prepare_animation_targets`` pass on a
    # marker-less mesh writes the marker, after which this branch
    # short-circuits above.
    for fc in _get_fcurves(action):
        if not fc.data_path.startswith(path_prefixes):
            continue
        for kp in fc.keyframe_points:
            if round(kp.co[0]) == 1:
                return True
    return False


def prepare_animation_targets(context, *, clear_existing: bool):
    """Preserve the current visible shape as frame 1 before run/fetch work begins."""
    from .uuid_registry import get_or_create_object_uuid

    scene = context.scene
    snapshots = {}
    for obj in _iterate_dynamic_objects(scene):
        uid = get_or_create_object_uuid(obj)
        if uid:
            snapshots[uid] = _capture_object_geometry(obj)
    if clear_existing:
        clear_animation_data(context, move_to_frame=False)
    for obj in _iterate_dynamic_objects(scene):
        uid = get_or_create_object_uuid(obj)
        if not uid:
            continue
        snapshot = snapshots.get(uid)
        if snapshot is None:
            continue
        if clear_existing or not _has_frame_one_geometry_keys(obj):
            _restore_object_geometry(obj, snapshot)
            _insert_frame_one_keys(obj, context)


