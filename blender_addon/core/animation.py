# File: animation.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from mathutils import Vector  # pyright: ignore

from ..models.groups import decode_vertex_group_identifier, get_addon_data, iterate_active_object_groups
from .pc2 import cleanup_mesh_cache, has_mesh_cache
from .utils import (
    _get_fcurves,
    get_pin_vertex_indices,
    get_vertices_in_group,
    parse_vertex_index,
    set_linear_interpolation,
)


def save_pin_keyframes(context):
    """Save all keyframes on pinned vertices to persistent Blender storage.

    Saves per (object_uuid, vertex_group) so lookups are fast.
    """
    state = get_addon_data(context.scene).state
    state.saved_pin_keyframes.clear()
    for group in iterate_active_object_groups(context.scene):
        if group.object_type == "STATIC":
            continue
        for pin_item in group.pin_vertex_groups:
            from .uuid_registry import resolve_pin
            obj = resolve_pin(pin_item)
            obj_name, vg_name = decode_vertex_group_identifier(pin_item.name)
            if not obj_name or not vg_name:
                continue
            if not obj or obj.type != "MESH":
                continue
            if not obj.data.animation_data or not obj.data.animation_data.action:
                continue
            vg = obj.vertex_groups.get(vg_name)
            if not vg:
                continue
            pin_vis = set(get_vertices_in_group(obj, vg))
            if not pin_vis:
                continue
            grp_entry = None
            for fc in _get_fcurves(obj.data.animation_data.action):
                if fc.data_path.startswith("vertices[") and ".co" in fc.data_path:
                    idx = parse_vertex_index(fc.data_path)
                    if idx is not None and idx in pin_vis:
                        if grp_entry is None:
                            from .uuid_registry import get_or_create_object_uuid, compute_vg_hash
                            grp_entry = state.saved_pin_keyframes.add()
                            grp_entry.object_name = obj_name
                            grp_entry.object_uuid = get_or_create_object_uuid(obj)
                            grp_entry.vertex_group = vg_name
                            grp_entry.vg_hash = str(compute_vg_hash(obj, vg_name))
                        fc_entry = grp_entry.fcurves.add()
                        fc_entry.data_path = fc.data_path
                        fc_entry.array_index = fc.array_index
                        for kp in fc.keyframe_points:
                            pt = fc_entry.points.add()
                            pt.frame = int(kp.co[0])
                            pt.value = kp.co[1]


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
            # Remove MESH_CACHE modifier and delete PC2 file
            vert_save = [Vector(v.co) for v in obj.data.vertices]
            cleanup_mesh_cache(obj)
            # Restore saved pin keyframes for all pinned vertices
            saved_col = get_addon_data(context.scene).state.saved_pin_keyframes
            restored = False
            from .uuid_registry import get_object_uuid
            obj_uuid = get_object_uuid(obj)
            for grp_entry in saved_col:
                if grp_entry.object_uuid == obj_uuid:
                    for fc_entry in grp_entry.fcurves:
                        idx = parse_vertex_index(fc_entry.data_path)
                        for pt in fc_entry.points:
                            obj.data.vertices[idx].co[fc_entry.array_index] = pt.value
                            obj.data.vertices[idx].keyframe_insert(
                                data_path="co", index=fc_entry.array_index,
                                frame=pt.frame,
                            )
                    restored = True
            if restored:
                if obj.data.animation_data and obj.data.animation_data.action:
                    set_linear_interpolation(obj.data.animation_data.action)
            for i in range(len(obj.data.vertices)):
                obj.data.vertices[i].co = vert_save[i]
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


def _insert_frame_one_keys(obj, context):
    if obj.type == "MESH":
        # Only insert frame-1 keys for pinned vertices (simulation uses PC2)
        pinned_vert_index = set(get_pin_vertex_indices(obj, context, frame=1))
        for i in pinned_vert_index:
            if i < len(obj.data.vertices):
                obj.data.vertices[i].keyframe_insert(data_path="co", frame=1)
        if obj.data.animation_data and obj.data.animation_data.action:
            set_linear_interpolation(obj.data.animation_data.action)
    # Curves use PC2 — no frame-1 keyframes needed


def _has_frame_one_geometry_keys(obj):
    # Objects with MESH_CACHE modifier are considered "ready"
    if has_mesh_cache(obj):
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


