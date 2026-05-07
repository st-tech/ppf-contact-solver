# File: pins.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json

import gpu  # pyright: ignore

from gpu_extras.batch import batch_for_shader  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ...state import decode_vertex_group_identifier, iterate_active_object_groups
from ....core.utils import get_moving_vertex_indices
from ....models.groups import get_addon_data

from .primitives import _line_to_tris


def _build_rod_batches(scene, depsgraph):
    """Build GPU batches for all rod overlays. Returns list of (batch, color)."""
    batches = []
    thickness = 0.003

    # Collect triangles grouped by color
    color_triangles = {}

    for group in iterate_active_object_groups(scene):
        if group.object_type == "ROD" and group.show_overlay_color:
            color = tuple(group.color)

            for obj_ref in group.assigned_objects:
                if not obj_ref.included:
                    continue
                from ....core.uuid_registry import resolve_assigned
                obj = resolve_assigned(obj_ref)
                if obj and obj.type == "MESH":
                    eval_obj = obj.evaluated_get(depsgraph)
                    mesh = eval_obj.data
                    world_matrix = eval_obj.matrix_world

                    if color not in color_triangles:
                        color_triangles[color] = []
                    tris = color_triangles[color]

                    for edge in mesh.edges:
                        v1 = world_matrix @ mesh.vertices[edge.vertices[0]].co
                        v2 = world_matrix @ mesh.vertices[edge.vertices[1]].co

                        edge_dir = (v2 - v1).normalized()

                        if abs(edge_dir.z) < 0.9:
                            perp1 = edge_dir.cross(Vector((0, 0, 1))).normalized()
                        else:
                            perp1 = edge_dir.cross(Vector((1, 0, 0))).normalized()
                        perp2 = edge_dir.cross(perp1).normalized()

                        offset1 = perp1 * thickness
                        offset2 = perp2 * thickness

                        p1 = v1 + offset1
                        p2 = v1 - offset1
                        p3 = v2 - offset1
                        p4 = v2 + offset1

                        tris.extend([p1, p2, p3, p1, p3, p4])

                        p5 = v1 + offset2
                        p6 = v1 - offset2
                        p7 = v2 - offset2
                        p8 = v2 + offset2

                        tris.extend([p5, p6, p7, p5, p7, p8])

    for color, tris in color_triangles.items():
        if tris:
            batch = batch_for_shader(
                gpu.shader.from_builtin("UNIFORM_COLOR"), "TRIS", {"pos": tris}
            )
            batches.append((batch, color))

    return batches


def _build_pin_data(scene, depsgraph):
    """Build pin vertex data. Returns list of (vertex, size, color) tuples."""
    pin_data = []

    for group in iterate_active_object_groups(scene):
        if group.show_pin_overlay and group.object_type != "STATIC":
            for obj_ref in group.assigned_objects:
                if not obj_ref.included:
                    continue
                from ....core.uuid_registry import resolve_assigned, resolve_pin
                obj = resolve_assigned(obj_ref)
                if not obj:
                    continue
                _obj_uuid = obj_ref.uuid
                current_frame = scene.frame_current

                if obj.type == "MESH":
                    eval_obj = obj.evaluated_get(depsgraph)
                    mesh = eval_obj.data
                    world_matrix = eval_obj.matrix_world
                    animated_indices = set(get_moving_vertex_indices(obj))

                    for pin_ref in group.pin_vertex_groups:
                        if pin_ref.use_pin_duration and current_frame > pin_ref.pin_duration:
                            continue
                        resolve_pin(pin_ref)
                        _, pin_vg_name = decode_vertex_group_identifier(
                            pin_ref.name
                        )
                        if pin_ref.object_uuid == _obj_uuid:
                            vg = obj.vertex_groups.get(pin_vg_name)
                            if vg:
                                has_ops = any(
                                    op.op_type in ("SPIN", "SCALE", "MOVE_BY", "TORQUE")
                                    for op in pin_ref.operations
                                )
                                vg_index = vg.index
                                for vert in mesh.vertices:
                                    for vg_elem in vert.groups:
                                        if vg_elem.group == vg_index:
                                            world_co = world_matrix @ vert.co
                                            if has_ops or vert.index in animated_indices:
                                                color = (0.5, 0.5, 1.0, 0.8)
                                            else:
                                                color = (1.0, 1.0, 1.0, 0.8)
                                            pin_data.append((
                                                world_co,
                                                group.pin_overlay_size,
                                                color,
                                            ))

                elif obj.type == "CURVE":
                    world_matrix = obj.matrix_world
                    for pin_ref in group.pin_vertex_groups:
                        if pin_ref.use_pin_duration and current_frame > pin_ref.pin_duration:
                            continue
                        resolve_pin(pin_ref)
                        _, pin_vg_name = decode_vertex_group_identifier(
                            pin_ref.name
                        )
                        if pin_ref.object_uuid == _obj_uuid:
                            key = f"_pin_{pin_vg_name}"
                            raw = obj.get(key)
                            if not raw:
                                continue
                            cp_indices = set(json.loads(raw))
                            has_ops = any(
                                op.op_type in ("SPIN", "SCALE", "MOVE_BY", "TORQUE")
                                for op in pin_ref.operations
                            )
                            color = (0.5, 0.5, 1.0, 0.8) if has_ops else (1.0, 1.0, 1.0, 0.8)
                            idx = 0
                            for spline in obj.data.splines:
                                if spline.type == "BEZIER":
                                    for bp in spline.bezier_points:
                                        if idx in cp_indices:
                                            pin_data.append((
                                                world_matrix @ bp.co,
                                                group.pin_overlay_size,
                                                color,
                                            ))
                                        idx += 1
                                else:
                                    for pt in spline.points:
                                        if idx in cp_indices:
                                            from mathutils import Vector  # pyright: ignore
                                            pin_data.append((
                                                world_matrix @ Vector((pt.co[0], pt.co[1], pt.co[2])),
                                                group.pin_overlay_size,
                                                color,
                                            ))
                                        idx += 1

    return pin_data


def _overlay_object_points(scene, obj_uuid):
    from ....core.uuid_registry import get_object_by_uuid
    obj = get_object_by_uuid(obj_uuid) if obj_uuid else None
    if obj is None:
        return None
    if obj.type == "MESH":
        # Read animated positions from PC2 file (no depsgraph access)
        from ....core.pc2 import (
            get_pc2_path,
            object_pc2_key,
            read_pc2_frame_count,
            read_pc2_frame,
        )
        import os
        pc2 = get_pc2_path(object_pc2_key(obj))
        if os.path.exists(pc2):
            n_verts = len(obj.data.vertices)
            n_frames = read_pc2_frame_count(pc2)
            frame_idx = scene.frame_current - 1
            if n_frames > 0 and 0 <= frame_idx < n_frames:
                verts = read_pc2_frame(pc2, frame_idx, n_verts)
                return [obj.matrix_world @ Vector(v) for v in verts]
        return [obj.matrix_world @ v.co for v in obj.data.vertices]
    if obj.type == "CURVE":
        from ....core.curve_rod import sample_curve

        vert, _, _ = sample_curve(obj, obj.matrix_world)
        return [Vector(v) for v in vert]
    return None


def _build_snap_batches(scene):
    state = get_addon_data(scene).state
    all_tris = []
    snap_points = []
    thickness = 0.0018

    for pair in state.merge_pairs:
        if not pair.show_stitch:
            continue
        if not pair.cross_stitch_json:
            continue
        try:
            stitch = json.loads(pair.cross_stitch_json)
        except json.JSONDecodeError:
            continue

        source_uuid = stitch.get("source_uuid", "")
        target_uuid = stitch.get("target_uuid", "")
        ind = stitch.get("ind", [])
        w = stitch.get("w", [])
        if not source_uuid or not target_uuid or not ind or not w:
            continue

        source_points = _overlay_object_points(scene, source_uuid)
        target_points = _overlay_object_points(scene, target_uuid)
        if source_points is None or target_points is None:
            continue

        for row, weight in zip(ind, w, strict=False):
            if len(row) < 4 or len(weight) < 4:
                continue
            src_i = int(row[0])
            if src_i < 0 or src_i >= len(source_points):
                continue
            target_indices = [int(row[1]), int(row[2]), int(row[3])]
            if any(idx < 0 or idx >= len(target_points) for idx in target_indices):
                continue

            source_pos = source_points[src_i]
            target_pos = (
                float(weight[1]) * target_points[target_indices[0]]
                + float(weight[2]) * target_points[target_indices[1]]
                + float(weight[3]) * target_points[target_indices[2]]
            )
            all_tris.extend(_line_to_tris(source_pos, target_pos, thickness))
            snap_points.append((source_pos, 7.0, (1.0, 0.7, 0.2, 1.0)))
            snap_points.append((target_pos, 5.0, (1.0, 1.0, 0.2, 1.0)))

    if not all_tris:
        return [], []

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
    return [(batch, (1.0, 0.9, 0.2, 0.95))], snap_points
