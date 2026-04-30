# File: overlay_geometry.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json
import math

import bpy  # pyright: ignore
import gpu  # pyright: ignore

from gpu_extras.batch import batch_for_shader  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ..state import decode_vertex_group_identifier, iterate_active_object_groups
from ...core.utils import get_moving_vertex_indices
from ...models.groups import get_addon_data


_OP_COLORS = {
    "SPIN": (1.0, 0.6, 0.2),
    "MOVE_BY": (0.3, 0.9, 0.9),
    "SCALE": (0.9, 0.3, 0.8),
    "TORQUE": (0.2, 0.8, 0.4),
}


def _line_to_tris(v1, v2, thickness):
    """Convert a line segment into two triangles (a thin quad) for guaranteed thickness."""
    d = (v2 - v1)
    if d.length < 1e-9:
        return []
    d_norm = d.normalized()
    if abs(d_norm.z) < 0.9:
        perp = d_norm.cross(Vector((0, 0, 1))).normalized()
    else:
        perp = d_norm.cross(Vector((1, 0, 0))).normalized()
    perp2 = d_norm.cross(perp).normalized()
    off1 = perp * thickness
    off2 = perp2 * thickness
    # Two quads (4 faces) for cross-shaped thickness visible from any angle
    tris = []
    for off in (off1, off2):
        p1 = v1 + off
        p2 = v1 - off
        p3 = v2 - off
        p4 = v2 + off
        tris.extend([p1, p2, p3, p1, p3, p4])
    return tris


def _generate_sphere_wireframe(radius=1.0, thickness=0.005, segments=16, rings=8):
    """Generate triangle-based wireframe for a UV sphere (TRIS primitive).

    Uses _line_to_tris so line thickness is guaranteed on all platforms.
    """
    grid = []
    for i in range(rings + 1):
        theta = math.pi * i / rings - math.pi / 2
        ring_row = []
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            x = radius * math.cos(theta) * math.cos(phi)
            y = radius * math.cos(theta) * math.sin(phi)
            z = radius * math.sin(theta)
            ring_row.append(Vector((x, y, z)))
        grid.append(ring_row)
    tris = []
    # Ring lines (latitude)
    for i in range(rings + 1):
        for j in range(segments):
            tris.extend(_line_to_tris(grid[i][j], grid[i][(j + 1) % segments], thickness))
    # Meridian lines (longitude)
    for j in range(segments):
        for i in range(rings):
            tris.extend(_line_to_tris(grid[i][j], grid[i + 1][j], thickness))
    return tris


def _generate_sphere_fill(radius=1.0, segments=16, rings=8):
    """Generate vertex list for a filled UV sphere (TRIS primitive)."""
    grid = []
    for i in range(rings + 1):
        theta = math.pi * i / rings - math.pi / 2
        ring_row = []
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            x = radius * math.cos(theta) * math.cos(phi)
            y = radius * math.cos(theta) * math.sin(phi)
            z = radius * math.sin(theta)
            ring_row.append(Vector((x, y, z)))
        grid.append(ring_row)
    tris = []
    for i in range(rings):
        for j in range(segments):
            j_next = (j + 1) % segments
            tris.append(grid[i][j])
            tris.append(grid[i + 1][j])
            tris.append(grid[i + 1][j_next])
            tris.append(grid[i][j])
            tris.append(grid[i + 1][j_next])
            tris.append(grid[i][j_next])
    return tris


def _max_towards_centroid(vertices, direction, eps=1e-3):
    """Compute centroid of vertices furthest towards a direction."""
    import numpy as _np
    if not vertices or direction.length < 1e-6:
        return sum(vertices, Vector((0, 0, 0))) / max(len(vertices), 1)
    d = direction.normalized()
    positions = _np.array([list(v) for v in vertices])
    projections = positions @ _np.array(d)
    max_val = projections.max()
    mask = projections > max_val - eps
    selected = positions[mask]
    c = selected.mean(axis=0)
    return Vector((c[0], c[1], c[2]))


def _generate_arrow(direction, shaft_length=1.2, shaft_thickness=0.01,
                     cone_length=0.15, cone_radius=0.06):
    """Generate shaft (TRIS) and cone (TRIS) vertices for a directional arrow."""
    d = direction.normalized()
    origin = Vector((0, 0, 0))
    tip = d * shaft_length
    shaft_tris = _line_to_tris(origin, tip, shaft_thickness)

    # Perpendicular vectors for cone base
    if abs(d.z) < 0.9:
        perp1 = d.cross(Vector((0, 0, 1))).normalized()
    else:
        perp1 = d.cross(Vector((1, 0, 0))).normalized()
    perp2 = d.cross(perp1).normalized()

    apex = d * (shaft_length + cone_length)
    cone_segments = 8
    cone_tris = []
    base_points = []
    for i in range(cone_segments):
        angle = 2 * math.pi * i / cone_segments
        offset = perp1 * (cone_radius * math.cos(angle)) + perp2 * (
            cone_radius * math.sin(angle)
        )
        base_points.append(tip + offset)
    for i in range(cone_segments):
        cone_tris.append(apex)
        cone_tris.append(base_points[i])
        cone_tris.append(base_points[(i + 1) % cone_segments])
    return shaft_tris, cone_tris


def _generate_circle(center, axis, radius, segments=32, thickness=0.005):
    """Generate a 3D circle perpendicular to axis as triangle-based lines."""
    axis_n = axis.normalized()
    if abs(axis_n.z) < 0.9:
        perp1 = axis_n.cross(Vector((0, 0, 1))).normalized()
    else:
        perp1 = axis_n.cross(Vector((1, 0, 0))).normalized()
    perp2 = axis_n.cross(perp1).normalized()

    points = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        p = center + perp1 * (radius * math.cos(angle)) + perp2 * (radius * math.sin(angle))
        points.append(p)

    tris = []
    for i in range(segments):
        tris.extend(_line_to_tris(points[i], points[(i + 1) % segments], thickness))
    return tris


def _generate_rotation_arc(center, axis, radius, angular_velocity, thickness=0.008):
    """Generate an arc with arrow head showing rotation direction."""
    axis_n = axis.normalized()
    if abs(axis_n.z) < 0.9:
        perp1 = axis_n.cross(Vector((0, 0, 1))).normalized()
    else:
        perp1 = axis_n.cross(Vector((1, 0, 0))).normalized()
    perp2 = axis_n.cross(perp1).normalized()

    arc_angle = math.radians(270)
    sign = 1 if angular_velocity >= 0 else -1

    segments = 24
    points = []
    for i in range(segments + 1):
        t = sign * arc_angle * i / segments
        p = center + perp1 * (radius * math.cos(t)) + perp2 * (radius * math.sin(t))
        points.append(p)

    tris = []
    for i in range(len(points) - 1):
        tris.extend(_line_to_tris(points[i], points[i + 1], thickness))

    # Arrow cone at the end of the arc
    if len(points) >= 2:
        tangent = (points[-1] - points[-2]).normalized()
        cone_length = radius * 0.15
        cone_radius = radius * 0.06
        apex = points[-1] + tangent * cone_length
        if abs(tangent.z) < 0.9:
            cp1 = tangent.cross(Vector((0, 0, 1))).normalized()
        else:
            cp1 = tangent.cross(Vector((1, 0, 0))).normalized()
        cp2 = tangent.cross(cp1).normalized()
        base_points = []
        for i in range(8):
            angle = 2 * math.pi * i / 8
            offset = cp1 * (cone_radius * math.cos(angle)) + cp2 * (cone_radius * math.sin(angle))
            base_points.append(points[-1] + offset)
        for i in range(8):
            tris.append(apex)
            tris.append(base_points[i])
            tris.append(base_points[(i + 1) % 8])

    return tris


def _generate_vertex_arrow(start, end, thickness=0.003):
    """Generate a small directional arrow from start to end."""
    diff = end - start
    length = diff.length
    if length < 1e-6:
        return []

    direction = diff.normalized()
    cone_length = thickness * 5
    cone_radius = thickness * 3
    shaft_end = end - direction * cone_length
    tris = _line_to_tris(start, shaft_end, thickness)
    if abs(direction.z) < 0.9:
        cp1 = direction.cross(Vector((0, 0, 1))).normalized()
    else:
        cp1 = direction.cross(Vector((1, 0, 0))).normalized()
    cp2 = direction.cross(cp1).normalized()
    base_center = end - direction * cone_length
    base_points = []
    for i in range(6):
        angle = 2 * math.pi * i / 6
        offset = cp1 * (cone_radius * math.cos(angle)) + cp2 * (cone_radius * math.sin(angle))
        base_points.append(base_center + offset)
    for i in range(6):
        tris.append(end)
        tris.append(base_points[i])
        tris.append(base_points[(i + 1) % 6])

    return tris


def _compute_pca_axes(vertices):
    """Compute PCA eigenvectors from vertex positions.

    Returns (centroid, eigenvectors) where eigenvectors columns are
    PC1 (major), PC2 (middle), PC3 (minor) in descending eigenvalue order.
    Returns None if fewer than 2 vertices.
    """
    import numpy as np

    if len(vertices) < 2:
        return None
    coords = np.array([[v.x, v.y, v.z] for v in vertices])
    centroid_np = coords.mean(axis=0)
    centered = coords - centroid_np
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    centroid_vec = Vector((float(centroid_np[0]), float(centroid_np[1]), float(centroid_np[2])))
    return centroid_vec, eigenvectors


def _build_static_op_batches(group, depsgraph, shader, batches, labels):
    """Append overlay batches/labels for UI-assigned static ops on *group*.

    Mirrors the pin-op overlay shapes — MOVE_BY arrows from each vertex,
    SPIN circle + sweep arc, SCALE arrows from each vertex — but walks
    per-object because static ops live on AssignedObject.static_ops.
    """
    from ...core.uuid_registry import get_object_by_uuid

    for assigned in group.assigned_objects:
        if not assigned.included or not assigned.uuid:
            continue
        if not any(op.show_overlay for op in assigned.static_ops):
            continue
        obj = get_object_by_uuid(assigned.uuid)
        if obj is None or obj.type != "MESH":
            continue

        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.data
        world_matrix = eval_obj.matrix_world
        vertices = [world_matrix @ v.co for v in mesh.vertices]
        if not vertices:
            continue
        centroid = sum(vertices, Vector((0, 0, 0))) / len(vertices)

        for op in assigned.static_ops:
            if not op.show_overlay:
                continue
            rgb = _OP_COLORS.get(op.op_type, (0.8, 0.8, 0.8))

            if op.op_type == "MOVE_BY":
                delta = Vector(op.delta)
                if delta.length < 1e-6:
                    continue
                # Single arrow from the object's origin — static objects
                # translate as a rigid whole, one arrow reads clearer
                # than a per-vertex forest.
                origin = world_matrix.translation.copy()
                thickness = max(0.01, delta.length * 0.02)
                all_tris = _generate_vertex_arrow(
                    origin, origin + delta, thickness=thickness,
                )
                if all_tris:
                    batches.append((
                        batch_for_shader(shader, "TRIS", {"pos": all_tris}),
                        (*rgb, 0.8),
                    ))
                labels.append({
                    "text": f"\u0394 ({delta.x:.2f}, {delta.y:.2f}, {delta.z:.2f})",
                    "pos_3d": origin + delta * 0.5,
                    "color": (*rgb, 0.9),
                })

            elif op.op_type == "SPIN":
                # Static spin pivots around the object origin.
                center = world_matrix.translation.copy()
                axis = Vector(op.spin_axis)
                if axis.length < 1e-6:
                    continue
                axis = axis.normalized()
                if getattr(op, "spin_flip", False):
                    axis = -axis
                total = 0.0
                for v in vertices:
                    d = v - center
                    total += (d - axis * d.dot(axis)).length
                avg_radius = max(total / len(vertices), 0.1)
                thickness = max(0.002, avg_radius * 0.01)
                all_tris = (
                    _generate_circle(center, axis, avg_radius, thickness=thickness)
                    + _generate_rotation_arc(
                        center, axis, avg_radius, op.spin_angular_velocity,
                        thickness=thickness * 2,
                    )
                )
                if all_tris:
                    batches.append((
                        batch_for_shader(shader, "TRIS", {"pos": all_tris}),
                        (*rgb, 0.7),
                    ))
                labels.append({
                    "text": f"\u03c9 = {op.spin_angular_velocity:.0f}\u00b0/s",
                    "pos_3d": center + axis * avg_radius * 0.1,
                    "color": (*rgb, 0.9),
                })

            elif op.op_type == "SCALE":
                # Static scale pivots around the object origin.
                center = world_matrix.translation.copy()
                factor = op.scale_factor
                all_tris = []
                for v in vertices:
                    target = center + (v - center) * factor
                    diff = target - v
                    if diff.length < 1e-6:
                        continue
                    t = max(0.001, diff.length * 0.015)
                    all_tris.extend(_generate_vertex_arrow(v, target, thickness=t))
                if all_tris:
                    batches.append((
                        batch_for_shader(shader, "TRIS", {"pos": all_tris}),
                        (*rgb, 0.6),
                    ))
                labels.append({
                    "text": f"\u00d7{factor:.2f}",
                    "pos_3d": center,
                    "color": (*rgb, 0.9),
                })


def _build_operation_batches(scene, depsgraph, view_distance=10.0):
    """Build GPU batches and labels for pin operation overlays."""
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batches = []
    labels = []

    for group in iterate_active_object_groups(scene):
        if group.object_type == "STATIC":
            _build_static_op_batches(
                group, depsgraph, shader, batches, labels,
            )
            continue
        from ...core.uuid_registry import resolve_pin
        for pin_ref in group.pin_vertex_groups:
            obj = resolve_pin(pin_ref)
            _, pin_vg_name = decode_vertex_group_identifier(pin_ref.name)
            if not obj or not pin_vg_name:
                continue

            has_visible = any(
                op.show_overlay and op.op_type != "EMBEDDED_MOVE"
                for op in pin_ref.operations
            )
            has_max_towards = any(
                (op.op_type == "SPIN" and op.spin_center_mode == "MAX_TOWARDS" and op.show_max_towards_spin)
                or (op.op_type == "SCALE" and op.scale_center_mode == "MAX_TOWARDS" and op.show_max_towards_scale)
                for op in pin_ref.operations
            )
            has_show_vertex = any(
                (op.op_type == "SPIN" and op.spin_center_mode == "VERTEX" and op.show_vertex_spin)
                or (op.op_type == "SCALE" and op.scale_center_mode == "VERTEX" and op.show_vertex_scale)
                for op in pin_ref.operations
            )
            if not has_visible and not has_max_towards and not has_show_vertex:
                continue

            if obj.type != "MESH":
                continue

            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.data
            world_matrix = eval_obj.matrix_world

            vg = obj.vertex_groups.get(pin_vg_name)
            if not vg:
                continue
            vg_index = vg.index

            vertices = []
            for vert in mesh.vertices:
                for vg_elem in vert.groups:
                    if vg_elem.group == vg_index:
                        vertices.append(world_matrix @ vert.co)
                        break

            if not vertices:
                continue

            centroid = sum(vertices, Vector((0, 0, 0))) / len(vertices)

            for op in pin_ref.operations:
                if not op.show_overlay or op.op_type == "EMBEDDED_MOVE":
                    continue

                rgb = _OP_COLORS.get(op.op_type, (0.8, 0.8, 0.8))

                if op.op_type == "SPIN":
                    if op.spin_center_mode == "ABSOLUTE":
                        center = Vector(op.spin_center)
                    elif op.spin_center_mode == "MAX_TOWARDS":
                        center = _max_towards_centroid(vertices, Vector(op.spin_center_direction))
                    elif op.spin_center_mode == "VERTEX" and op.spin_center_vertex >= 0:
                        center = world_matrix @ obj.data.vertices[op.spin_center_vertex].co
                    else:
                        center = centroid
                    axis = Vector(op.spin_axis)
                    if axis.length < 1e-6:
                        continue
                    axis = axis.normalized()
                    if getattr(op, "spin_flip", False):
                        axis = -axis

                    total_dist = 0
                    for v in vertices:
                        diff = v - center
                        projected = diff - axis * diff.dot(axis)
                        total_dist += projected.length
                    avg_radius = total_dist / len(vertices)
                    if avg_radius < 1e-4:
                        avg_radius = 0.1

                    thickness = max(0.002, avg_radius * 0.01)
                    circle_tris = _generate_circle(center, axis, avg_radius, thickness=thickness)
                    arc_tris = _generate_rotation_arc(
                        center, axis, avg_radius, op.spin_angular_velocity,
                        thickness=thickness * 2,
                    )
                    all_tris = circle_tris + arc_tris
                    if all_tris:
                        batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
                        batches.append((batch, (*rgb, 0.7)))

                    labels.append({
                        "text": f"\u03c9 = {op.spin_angular_velocity:.0f}\u00b0/s",
                        "pos_3d": center + axis * avg_radius * 0.1,
                        "color": (*rgb, 0.9),
                    })

                elif op.op_type == "MOVE_BY":
                    delta = Vector(op.delta)
                    if delta.length < 1e-6:
                        continue

                    thickness = 0.015
                    all_tris = []
                    for v in vertices:
                        all_tris.extend(_generate_vertex_arrow(v, v + delta, thickness=thickness))

                    if all_tris:
                        batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
                        batches.append((batch, (*rgb, 0.6)))

                    labels.append({
                        "text": f"\u0394 ({delta.x:.2f}, {delta.y:.2f}, {delta.z:.2f})",
                        "pos_3d": centroid + delta * 0.5,
                        "color": (*rgb, 0.9),
                    })

                elif op.op_type == "SCALE":
                    if op.scale_center_mode == "ABSOLUTE":
                        center = Vector(op.scale_center)
                    elif op.scale_center_mode == "MAX_TOWARDS":
                        center = _max_towards_centroid(vertices, Vector(op.scale_center_direction))
                    elif op.scale_center_mode == "VERTEX" and op.scale_center_vertex >= 0:
                        center = world_matrix @ obj.data.vertices[op.scale_center_vertex].co
                    else:
                        center = centroid
                    factor = op.scale_factor

                    all_tris = []
                    for v in vertices:
                        target = center + (v - center) * factor
                        diff = target - v
                        if diff.length < 1e-6:
                            continue
                        t = max(0.001, diff.length * 0.015)
                        all_tris.extend(_generate_vertex_arrow(v, target, thickness=t))

                    if all_tris:
                        batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
                        batches.append((batch, (*rgb, 0.6)))

                    labels.append({
                        "text": f"\u00d7{factor:.2f}",
                        "pos_3d": center,
                        "color": (*rgb, 0.9),
                    })

                elif op.op_type == "TORQUE":
                    pca_result = _compute_pca_axes(vertices)
                    if pca_result is None:
                        continue
                    center, eigvecs = pca_result
                    comp_idx = {"PC1": 0, "PC2": 1, "PC3": 2}.get(
                        op.torque_axis_component, 2,
                    )
                    axis = Vector((
                        float(eigvecs[0, comp_idx]),
                        float(eigvecs[1, comp_idx]),
                        float(eigvecs[2, comp_idx]),
                    ))
                    if axis.length < 1e-6:
                        continue
                    # Compute everything in solver space to match GPU exactly
                    import numpy as _np
                    coords = _np.array([[v.x, v.y, v.z] for v in vertices])
                    # Blender Z-up -> solver Y-up: [x, z, -y]
                    sv = coords[:, [0, 2, 1]].copy()
                    sv[:, 2] *= -1
                    sc = sv.mean(axis=0)
                    sd = sv - sc
                    scov = _np.cov(sd, rowvar=False)
                    sevals, sevecs = _np.linalg.eigh(scov)
                    sidx = _np.argsort(sevals)[::-1]
                    sevecs = sevecs[:, sidx]
                    sax = sevecs[:, comp_idx]
                    # Pick hint vertex in solver space (same as encoder picks
                    # in Blender space then maps -- use solver projections directly)
                    sproj = sd @ sax
                    if op.torque_flip:
                        hint_idx = int(_np.argmin(sproj))
                    else:
                        hint_idx = int(_np.argmax(sproj))
                    # Orient axis using hint
                    if _np.dot(sax, sd[hint_idx]) < 0:
                        sax = -sax
                    # Transform back to Blender: solver [x,y,z] -> Blender [x,-z,y]
                    axis = Vector((float(sax[0]), float(-sax[2]), float(sax[1]))).normalized()

                    total_dist = 0
                    for v in vertices:
                        diff = v - center
                        projected = diff - axis * diff.dot(axis)
                        total_dist += projected.length
                    avg_radius = total_dist / len(vertices)
                    if avg_radius < 1e-4:
                        avg_radius = 0.1

                    thickness = max(0.002, avg_radius * 0.01)
                    sign = 1
                    circle_tris = _generate_circle(
                        center, axis, avg_radius, thickness=thickness,
                    )
                    arc_tris = _generate_rotation_arc(
                        center, axis, avg_radius, sign * 360.0,
                        thickness=thickness * 2,
                    )
                    all_tris = circle_tris + arc_tris
                    if all_tris:
                        batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
                        batches.append((batch, (*rgb, 0.7)))

                    labels.append({
                        "text": f"\u03c4 = {op.torque_magnitude:.1f} N\u00b7m ({op.torque_axis_component})",
                        "pos_3d": center + axis * avg_radius * 0.1,
                        "color": (*rgb, 0.9),
                    })

            # Max Towards visualization: highlight selected vertices + direction arrow
            import numpy as _np
            for op in pin_ref.operations:
                show = False
                direction = None
                if op.op_type == "SPIN" and op.spin_center_mode == "MAX_TOWARDS" and op.show_max_towards_spin:
                    show = True
                    direction = Vector(op.spin_center_direction)
                elif op.op_type == "SCALE" and op.scale_center_mode == "MAX_TOWARDS" and op.show_max_towards_scale:
                    show = True
                    direction = Vector(op.scale_center_direction)
                if not show or direction is None or direction.length < 1e-6:
                    continue
                d = direction.normalized()
                positions = _np.array([list(v) for v in vertices])
                projections = positions @ _np.array(d)
                max_val = projections.max()
                eps = 1e-3
                mask = projections > max_val - eps
                selected_verts = [vertices[i] for i in range(len(vertices)) if mask[i]]
                if not selected_verts:
                    continue
                # Highlighted points batch (drawn as small spheres via POINTS)
                point_batch = batch_for_shader(shader, "POINTS", {"pos": selected_verts})
                batches.append((point_batch, (1.0, 0.8, 0.0, 0.9)))
                # Centroid of selected
                sel_centroid = sum(selected_verts, Vector((0, 0, 0))) / len(selected_verts)
                # Direction arrow from centroid (view-scaled)
                arrow_len = view_distance * 0.06
                shaft_tris, cone_tris = _generate_arrow(
                    d, shaft_length=arrow_len,
                    shaft_thickness=arrow_len * 0.015,
                    cone_length=arrow_len * 0.15,
                    cone_radius=arrow_len * 0.05,
                )
                shaft_tris = [v + sel_centroid for v in shaft_tris]
                cone_tris = [v + sel_centroid for v in cone_tris]
                if shaft_tris:
                    batches.append((batch_for_shader(shader, "TRIS", {"pos": shaft_tris}), (1.0, 0.8, 0.0, 0.7)))
                if cone_tris:
                    batches.append((batch_for_shader(shader, "TRIS", {"pos": cone_tris}), (1.0, 0.8, 0.0, 0.9)))
                labels.append({
                    "text": f"Max Towards ({len(selected_verts)} verts)",
                    "pos_3d": sel_centroid + d * arrow_len * 0.5,
                    "color": (1.0, 0.8, 0.0, 0.9),
                })

            # Vertex center visualization: highlight the single picked vertex
            for op in pin_ref.operations:
                vi = -1
                if op.op_type == "SPIN" and op.spin_center_mode == "VERTEX" and op.show_vertex_spin:
                    vi = op.spin_center_vertex
                elif op.op_type == "SCALE" and op.scale_center_mode == "VERTEX" and op.show_vertex_scale:
                    vi = op.scale_center_vertex
                if vi < 0 or vi >= len(mesh.vertices):
                    continue
                pos = world_matrix @ mesh.vertices[vi].co
                point_batch = batch_for_shader(shader, "POINTS", {"pos": [pos]})
                batches.append((point_batch, (0.0, 1.0, 0.5, 0.9)))
                labels.append({
                    "text": f"Center (v{vi})",
                    "pos_3d": pos,
                    "color": (0.0, 1.0, 0.5, 0.9),
                })

    return batches, labels


class DirectionPreviewManager:
    """Manages multiple direction preview visualizations in the viewport."""

    def __init__(self):
        self._entries = {}

    def add(self, key, direction, color, label="", strength=0.0, unit="", raw_direction=None):
        """Add or update a direction preview entry."""
        self._entries[key] = {
            "direction": direction,
            "color": color,
            "label": label,
            "strength": strength,
            "unit": unit,
            "raw_direction": raw_direction or direction,
        }

    def remove(self, key):
        """Remove a direction preview entry."""
        self._entries.pop(key, None)

    def clear(self):
        """Remove all entries."""
        self._entries.clear()

    def build_batches(self, radius=1.0):
        """Build GPU batches for all active entries, spaced along X to avoid overlap.

        Returns list of dicts with batches, colors, label info, and center position.
        """
        if not self._entries:
            return []

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        keys = sorted(self._entries.keys())
        n = len(keys)
        spacing = 2.5 * radius
        results = []

        for idx, key in enumerate(keys):
            entry = self._entries[key]
            offset_x = (idx - (n - 1) / 2.0) * spacing
            offset = Vector((offset_x, 0, 0))

            wire_thickness = radius * 0.005
            shaft_thickness = radius * 0.015

            # Filled sphere background translated to offset
            fill_verts = _generate_sphere_fill(radius)
            fill_verts = [v + offset for v in fill_verts]

            # Sphere wireframe (triangle-based for thickness)
            sphere_verts = _generate_sphere_wireframe(radius, thickness=wire_thickness)
            sphere_verts = [v + offset for v in sphere_verts]

            # Arrow -- length proportional to strength
            strength = entry.get("strength", 1.0)
            arrow_scale = min(strength, 20.0) / 5.0 if strength > 0 else 1.0
            shaft_verts, cone_verts = _generate_arrow(
                entry["direction"],
                shaft_length=1.2 * radius * arrow_scale,
                shaft_thickness=shaft_thickness,
                cone_length=0.15 * radius,
                cone_radius=0.06 * radius,
            )
            shaft_verts = [v + offset for v in shaft_verts]
            cone_verts = [v + offset for v in cone_verts]

            r, g, b = entry["color"]
            # Store sphere center and radius for 2D label projection
            label_pos = Vector(offset)
            results.append(
                {
                    "fill_batch": batch_for_shader(
                        shader, "TRIS", {"pos": fill_verts}
                    ),
                    "sphere_batch": batch_for_shader(
                        shader, "TRIS", {"pos": sphere_verts}
                    ),
                    "shaft_batch": batch_for_shader(
                        shader, "TRIS", {"pos": shaft_verts}
                    ),
                    "cone_batch": batch_for_shader(
                        shader, "TRIS", {"pos": cone_verts}
                    ),
                    "fill_color": (r, g, b, 0.08),
                    "sphere_color": (r, g, b, 0.15),
                    "arrow_color": (r, g, b, 0.8),
                    "label": entry["label"],
                    "strength": entry["strength"],
                    "unit": entry.get("unit", ""),
                    "raw_direction": entry.get("raw_direction", Vector((0, 0, 0))),
                    "label_pos": label_pos,
                    "radius": radius,
                    "label_color": (r, g, b, 0.9),
                }
            )
        return results


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
                from ...core.uuid_registry import resolve_assigned
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
                from ...core.uuid_registry import resolve_assigned, resolve_pin
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
    from ...core.uuid_registry import get_object_by_uuid
    obj = get_object_by_uuid(obj_uuid) if obj_uuid else None
    if obj is None:
        return None
    if obj.type == "MESH":
        # Read animated positions from PC2 file (no depsgraph access)
        from ...core.pc2 import (
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
        from ...core.curve_rod import sample_curve

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


def _resolve_scene_dyn_params(state, current_frame):
    """Resolve gravity and wind at the given frame, considering dynamic parameter keyframes.

    Returns (gravity_vec, wind_direction_vec, wind_strength).
    """
    gravity = list(state.gravity_3d)
    wind_dir = list(state.wind_direction)
    wind_strength = state.wind_strength

    for dyn_item in state.dyn_params:
        if len(dyn_item.keyframes) < 2:
            continue

        # Build resolved keyframe list
        resolved = []
        for i, kf in enumerate(dyn_item.keyframes):
            if i == 0:
                if dyn_item.param_type == "GRAVITY":
                    resolved.append((kf.frame, list(state.gravity_3d), 0.0))
                elif dyn_item.param_type == "WIND":
                    resolved.append((kf.frame, list(state.wind_direction), state.wind_strength))
                else:
                    continue
            elif kf.use_hold and resolved:
                prev = resolved[-1]
                resolved.append((kf.frame, list(prev[1]), prev[2]))
            else:
                if dyn_item.param_type == "GRAVITY":
                    resolved.append((kf.frame, list(kf.gravity_value), 0.0))
                elif dyn_item.param_type == "WIND":
                    resolved.append((kf.frame, list(kf.wind_direction_value), kf.wind_strength_value))
                else:
                    continue

        if not resolved:
            continue

        # Interpolate at current_frame
        if current_frame <= resolved[0][0]:
            val, extra = resolved[0][1], resolved[0][2]
        elif current_frame >= resolved[-1][0]:
            val, extra = resolved[-1][1], resolved[-1][2]
        else:
            val, extra = resolved[0][1], resolved[0][2]
            for j in range(len(resolved) - 1):
                f0, v0, e0 = resolved[j]
                f1, v1, e1 = resolved[j + 1]
                if f0 <= current_frame <= f1:
                    dt = f1 - f0
                    w = (current_frame - f0) / dt if dt > 0 else 1.0
                    val = [v0[k] * (1 - w) + v1[k] * w for k in range(3)]
                    extra = e0 * (1 - w) + e1 * w
                    break

        if dyn_item.param_type == "GRAVITY":
            gravity = val
        elif dyn_item.param_type == "WIND":
            wind_dir = val
            wind_strength = extra

    return gravity, wind_dir, wind_strength


def _resolve_collider_state(item, current_frame):
    """Resolve collider position and radius at the given frame by interpolating keyframes.

    Frame 0 (index 0) uses base properties. Subsequent keyframes override.
    Interpolates linearly between keyframes; hold keyframes repeat previous value.

    Returns (position: Vector, radius: float).
    """
    base_pos = Vector(item.position)
    base_radius = item.radius if item.collider_type == "SPHERE" else 0.0

    if len(item.keyframes) <= 1:
        return base_pos, base_radius

    # Build resolved list: (frame, position, radius)
    resolved = []
    for i, kf in enumerate(item.keyframes):
        if i == 0:
            resolved.append((kf.frame, base_pos.copy(), base_radius))
        elif kf.use_hold and resolved:
            prev = resolved[-1]
            resolved.append((kf.frame, prev[1].copy(), prev[2]))
        else:
            resolved.append((kf.frame, Vector(kf.position), kf.radius))

    # Before first keyframe
    if current_frame <= resolved[0][0]:
        return resolved[0][1], resolved[0][2]
    # After last keyframe
    if current_frame >= resolved[-1][0]:
        return resolved[-1][1], resolved[-1][2]

    # Find surrounding keyframes and interpolate
    for i in range(len(resolved) - 1):
        f0, p0, r0 = resolved[i]
        f1, p1, r1 = resolved[i + 1]
        if f0 <= current_frame <= f1:
            dt = f1 - f0
            if dt == 0:
                return p1, r1
            w = (current_frame - f0) / dt
            pos = p0.lerp(p1, w)
            r = r0 * (1 - w) + r1 * w
            return pos, r

    return base_pos, base_radius


def _collider_hue_color(index, collider_type, saturation=0.6, value=0.85, alpha=1.0):
    """Generate a distinct color by rotating hue for each collider index.

    Walls start at blue (hue=0.6), spheres start at green (hue=0.33).
    """
    import colorsys
    base_hue = 0.6 if collider_type == "WALL" else 0.33
    hue = (base_hue + index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b, alpha)


def _build_collider_batches(view_distance):
    """Build GPU batches for invisible collider previews (walls and spheres).

    All sizes are proportional to view_distance so they appear constant on screen.
    """
    batches = []
    try:
        state = get_addon_data(bpy.context.scene).state
    except Exception:
        return batches

    scale = view_distance * 0.15
    thickness = scale * 0.006
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    current_frame = bpy.context.scene.frame_current
    wall_idx = 0
    sphere_idx = 0

    for item in state.invisible_colliders:
        if not item.show_preview:
            continue
        # Honor the "Active Until (frame)" cutoff: once the timeline has
        # passed it, the encoder drops this collider from the contact set,
        # so the overlay should hide too.
        if (
            item.enable_active_duration
            and current_frame >= item.active_duration
        ):
            continue

        pos, radius = _resolve_collider_state(item, current_frame)
        if item.collider_type == "WALL":
            base_r, base_g, base_b, _ = _collider_hue_color(wall_idx, "WALL")
            wall_idx += 1
        else:
            base_r, base_g, base_b, _ = _collider_hue_color(sphere_idx, "SPHERE")
            sphere_idx += 1

        if item.collider_type == "WALL":
            normal = Vector(item.normal)
            if normal.length < 1e-9:
                continue
            normal = normal.normalized()

            # Orthonormal basis on the plane
            if abs(normal.z) < 0.9:
                tangent = normal.cross(Vector((0, 0, 1))).normalized()
            else:
                tangent = normal.cross(Vector((1, 0, 0))).normalized()
            bitangent = normal.cross(tangent).normalized()

            # Core grid on the plane (view-scaled)
            grid_size = scale
            grid_steps = 8
            tris = []
            for i in range(grid_steps + 1):
                t = -grid_size + (2 * grid_size * i / grid_steps)
                p1 = pos + tangent * t - bitangent * grid_size
                p2 = pos + tangent * t + bitangent * grid_size
                tris.extend(_line_to_tris(p1, p2, thickness))
                p1 = pos - tangent * grid_size + bitangent * t
                p2 = pos + tangent * grid_size + bitangent * t
                tris.extend(_line_to_tris(p1, p2, thickness))

            if tris:
                batch = batch_for_shader(shader, "TRIS", {"pos": tris})
                batches.append((batch, (base_r, base_g, base_b, 0.3)))

            # Expanding dashed square outlines (infinite plane hint)
            n_rings = 4
            dash_count = 24
            for ring in range(1, n_rings + 1):
                ring_size = grid_size * (1 + ring * 0.3)
                alpha = 0.25 / (ring * 0.8)
                dash_tris = []
                # Four edges of the square, each dashed
                corners = [
                    (pos + tangent * (-ring_size) + bitangent * (-ring_size),
                     pos + tangent * ring_size + bitangent * (-ring_size)),
                    (pos + tangent * ring_size + bitangent * (-ring_size),
                     pos + tangent * ring_size + bitangent * ring_size),
                    (pos + tangent * ring_size + bitangent * ring_size,
                     pos + tangent * (-ring_size) + bitangent * ring_size),
                    (pos + tangent * (-ring_size) + bitangent * ring_size,
                     pos + tangent * (-ring_size) + bitangent * (-ring_size)),
                ]
                for edge_start, edge_end in corners:
                    edge_vec = edge_end - edge_start
                    for d in range(dash_count):
                        t0 = d / dash_count
                        t1 = (d + 0.5) / dash_count
                        p1 = edge_start + edge_vec * t0
                        p2 = edge_start + edge_vec * t1
                        dash_tris.extend(_line_to_tris(p1, p2, thickness * 0.7))
                if dash_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": dash_tris})
                    batches.append((batch, (base_r, base_g, base_b, alpha)))

            # Normal arrow (view-scaled)
            arrow_len = scale * 0.8
            shaft_tris, cone_tris = _generate_arrow(
                normal,
                shaft_length=arrow_len,
                shaft_thickness=thickness * 2,
                cone_length=arrow_len * 0.2,
                cone_radius=thickness * 5,
            )
            shaft_tris = [v + pos for v in shaft_tris]
            cone_tris = [v + pos for v in cone_tris]
            arrow_color = (base_r, base_g, base_b, 0.8)
            if shaft_tris:
                batch = batch_for_shader(shader, "TRIS", {"pos": shaft_tris})
                batches.append((batch, arrow_color))
            if cone_tris:
                batch = batch_for_shader(shader, "TRIS", {"pos": cone_tris})
                batches.append((batch, arrow_color))

        elif item.collider_type == "SPHERE":
            segments = 24
            rings = 16

            if item.hemisphere:
                # Hemisphere = lower half-sphere + cylinder extending upward.
                # The solver projects the center Y to vertex Y for vertices
                # above the equator, creating a cylindrical extension.

                # Lower hemisphere wireframe (only bottom half: rings below equator)
                half_rings = rings // 2
                grid = []
                for i in range(half_rings + 1):
                    theta = -math.pi / 2 + math.pi / 2 * i / half_rings
                    ring_row = []
                    for j in range(segments):
                        phi = 2 * math.pi * j / segments
                        x = radius * math.cos(theta) * math.cos(phi)
                        y = radius * math.cos(theta) * math.sin(phi)
                        z = radius * math.sin(theta)
                        ring_row.append(Vector((x, y, z)))
                    grid.append(ring_row)
                hemi_tris = []
                for i in range(half_rings + 1):
                    for j in range(segments):
                        hemi_tris.extend(_line_to_tris(
                            grid[i][j], grid[i][(j + 1) % segments], thickness,
                        ))
                for j in range(segments):
                    for i in range(half_rings):
                        hemi_tris.extend(_line_to_tris(
                            grid[i][j], grid[i + 1][j], thickness,
                        ))
                hemi_tris = [v + pos for v in hemi_tris]

                # Cylinder extending upward from equator
                cyl_height = scale * 1.5
                cyl_tris = []
                for j in range(segments):
                    phi = 2 * math.pi * j / segments
                    bx = radius * math.cos(phi)
                    by = radius * math.sin(phi)
                    bottom = pos + Vector((bx, by, 0))
                    top = pos + Vector((bx, by, cyl_height))
                    # Vertical lines
                    cyl_tris.extend(_line_to_tris(bottom, top, thickness))
                # Top and bottom rings
                for z_off in (0.0, cyl_height):
                    ring_pts = []
                    for j in range(segments):
                        phi = 2 * math.pi * j / segments
                        ring_pts.append(pos + Vector((
                            radius * math.cos(phi),
                            radius * math.sin(phi),
                            z_off,
                        )))
                    for j in range(segments):
                        cyl_tris.extend(_line_to_tris(
                            ring_pts[j], ring_pts[(j + 1) % segments], thickness,
                        ))

                all_tris = hemi_tris + cyl_tris
                if all_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
                    batches.append((batch, (base_r, base_g, base_b, 0.4)))
            else:
                # Full sphere wireframe
                wire_tris = _generate_sphere_wireframe(
                    radius=radius, thickness=thickness,
                    segments=segments, rings=rings,
                )
                wire_tris = [v + pos for v in wire_tris]
                if wire_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": wire_tris})
                    batches.append((batch, (base_r, base_g, base_b, 0.4)))

            # Normal arrows at 6 cardinal surface points
            # Direction: outward for normal sphere, inward for inverted
            sign = -1.0 if item.invert else 1.0
            arrow_len = scale * 0.4
            arrow_thick = thickness * 1.5
            normal_color = (base_r, base_g, base_b, 0.7)
            cardinal_dirs = [
                Vector((1, 0, 0)), Vector((-1, 0, 0)),
                Vector((0, 1, 0)), Vector((0, -1, 0)),
                Vector((0, 0, -1)),  # bottom always
            ]
            if not item.hemisphere:
                cardinal_dirs.append(Vector((0, 0, 1)))  # top only for full sphere
            for d in cardinal_dirs:
                surface_pt = pos + d * radius
                arrow_dir = d * sign
                shaft_tris, cone_tris = _generate_arrow(
                    arrow_dir,
                    shaft_length=arrow_len,
                    shaft_thickness=arrow_thick,
                    cone_length=arrow_len * 0.25,
                    cone_radius=arrow_thick * 4,
                )
                shaft_tris = [v + surface_pt for v in shaft_tris]
                cone_tris = [v + surface_pt for v in cone_tris]
                if shaft_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": shaft_tris})
                    batches.append((batch, normal_color))
                if cone_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": cone_tris})
                    batches.append((batch, normal_color))

    return batches


def _build_velocity_arrow_batches(scene, view_distance):
    """Build arrow + filled mesh overlay for per-object initial velocity visualization.

    Returns (batches, labels) where labels is a list of dicts for 2D text rendering.
    """
    batches = []
    labels = []
    current_frame = bpy.context.scene.frame_current
    scale = view_distance * 0.12
    thickness = scale * 0.008
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    for group in iterate_active_object_groups(scene):
        r, g, b = group.color[:3]
        arrow_color = (r, g, b, 0.8)
        fill_color = (r, g, b, 0.15)

        if not group.preview_velocity:
            continue
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            preview_kfs = [kf for kf in assigned.velocity_keyframes if kf.frame == current_frame]
            if not preview_kfs:
                continue
            from ...core.uuid_registry import resolve_assigned
            obj = resolve_assigned(assigned)
            if not obj or obj.type != "MESH":
                continue

            for kf in preview_kfs:
                vel_dir = Vector(kf.direction)
                strength = kf.speed
                if vel_dir.length < 1e-6 or strength < 1e-6:
                    continue
                vel_dir = vel_dir.normalized()

                mesh = obj.data
                mat = obj.matrix_world
                tris = []
                for poly in mesh.polygons:
                    verts = [mat @ mesh.vertices[vi].co for vi in poly.vertices]
                    for i in range(1, len(verts) - 1):
                        tris.extend([verts[0], verts[i], verts[i + 1]])
                if tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": tris})
                    batches.append((batch, fill_color))

                center = mat.translation.copy()
                arrow_len = scale * min(strength, 20.0) / 5.0
                shaft_tris, cone_tris = _generate_arrow(
                    vel_dir,
                    shaft_length=arrow_len,
                    shaft_thickness=thickness * 2,
                    cone_length=arrow_len * 0.18,
                    cone_radius=thickness * 5,
                )
                shaft_tris = [v + center for v in shaft_tris]
                cone_tris = [v + center for v in cone_tris]
                if shaft_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": shaft_tris})
                    batches.append((batch, arrow_color))
                if cone_tris:
                    batch = batch_for_shader(shader, "TRIS", {"pos": cone_tris})
                    batches.append((batch, arrow_color))

                tip_3d = center + vel_dir * arrow_len * 1.2
                d = vel_dir
                labels.append({
                    "pos_3d": tip_3d,
                    "text": f"F{kf.frame} {strength:.1f} m/s ({d.x:.1f}, {d.y:.1f}, {d.z:.1f})",
                    "color": (r, g, b, 0.9),
                })
    return batches, labels


def _build_violation_batches(scene, depsgraph, violations):
    """Build GPU batches to highlight validation violations.

    Returns list of (batch, primitive_type, color) tuples.
    """
    from gpu_extras.batch import batch_for_shader

    batches = []
    if not violations:
        return batches

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    COLORS = {
        "self_intersection": (1.0, 0.1, 0.1, 0.85),
        "contact_offset": (1.0, 0.5, 0.0, 0.85),
        "wall": (1.0, 0.1, 0.1, 0.9),
        "sphere": (0.8, 0.1, 1.0, 0.9),
        "runtime_intersection": (1.0, 0.05, 0.05, 0.9),
    }
    LABELS = {
        "self_intersection": "Self-Intersections",
        "contact_offset": "Contact-Offset Violations",
        "wall": "Wall Violations",
        "sphere": "Sphere Violations",
        "runtime_intersection": "Runtime Intersection(s)",
    }
    EDGE_THICKNESS = 0.006

    def _solver_to_blender(pos):
        """Convert solver Y-up position to Blender Z-up."""
        return Vector((pos[0], -pos[2], pos[1]))

    labels = []

    for violation in violations:
        vtype = violation.get("type", "")
        color = COLORS.get(vtype, (1.0, 0.0, 0.0, 0.9))
        label_text = LABELS.get(vtype, "Violation")
        count = violation.get("count", 0)

        if vtype in ("wall", "sphere"):
            points = []
            for entry in violation.get("vertices", []):
                pos = entry.get("pos")
                if pos:
                    points.append(_solver_to_blender(pos))
            if points:
                batch = batch_for_shader(shader, "POINTS", {"pos": points})
                batches.append((batch, "POINTS", color))
                c = points[0]
                labels.append({
                    "pos_3d": c + Vector((0, 0, 0.05)),
                    "text": f"{count} {label_text}",
                    "color": color[:3] + (1.0,),
                })

        elif vtype == "contact_offset":
            tri_verts = []
            centers = []
            for pair in violation.get("pairs", []):
                for key in ("ei", "ej"):
                    etype = pair.get(f"{key}_type", "")
                    pos_list = pair.get(f"{key}_pos", [])
                    if not pos_list:
                        continue
                    bverts = [_solver_to_blender(p) for p in pos_list]
                    if etype == "triangle" and len(bverts) >= 3:
                        tri_verts.extend([bverts[0], bverts[1], bverts[2]])
                        if not centers:
                            centers.append((bverts[0] + bverts[1] + bverts[2]) / 3.0)
                    elif etype == "edge" and len(bverts) >= 2:
                        tri_verts.extend(_line_to_tris(bverts[0], bverts[1], EDGE_THICKNESS))
                        if not centers:
                            centers.append((bverts[0] + bverts[1]) / 2.0)
            if tri_verts:
                batch = batch_for_shader(shader, "TRIS", {"pos": tri_verts})
                batches.append((batch, "TRIS", color))
                if centers:
                    labels.append({
                        "pos_3d": centers[0] + Vector((0, 0, 0.05)),
                        "text": f"{count} {label_text}",
                        "color": color[:3] + (1.0,),
                    })

        elif vtype == "self_intersection":
            tri_verts = []
            centers = []
            for tri_pair in violation.get("tris", []):
                for tri_pos in tri_pair:
                    if len(tri_pos) >= 3:
                        bv = [_solver_to_blender(p) for p in tri_pos]
                        tri_verts.extend([bv[0], bv[1], bv[2]])
                        if not centers:
                            centers.append((bv[0] + bv[1] + bv[2]) / 3.0)
            if tri_verts:
                batch = batch_for_shader(shader, "TRIS", {"pos": tri_verts})
                batches.append((batch, "TRIS", color))
                if centers:
                    labels.append({
                        "pos_3d": centers[0] + Vector((0, 0, 0.05)),
                        "text": f"{count} {label_text}",
                        "color": color[:3] + (1.0,),
                    })

        elif vtype == "runtime_intersection":
            tri_verts = []
            centers = []
            for entry in violation.get("entries", []):
                itype = entry.get("itype", "")
                pos0 = entry.get("positions0", [])
                pos1 = entry.get("positions1", [])
                if itype in ("face_edge", "collision_mesh"):
                    # pos0 = face (3 verts), pos1 = edge (2 verts)
                    if len(pos0) >= 3:
                        bv = [_solver_to_blender(p) for p in pos0]
                        tri_verts.extend([bv[0], bv[1], bv[2]])
                        if not centers:
                            centers.append((bv[0] + bv[1] + bv[2]) / 3.0)
                    if len(pos1) >= 2:
                        bv = [_solver_to_blender(p) for p in pos1]
                        tri_verts.extend(_line_to_tris(bv[0], bv[1], EDGE_THICKNESS))
                elif itype == "edge_edge":
                    # pos0 = edge0 (2 verts), pos1 = edge1 (2 verts)
                    for edge_pos in (pos0, pos1):
                        if len(edge_pos) >= 2:
                            bv = [_solver_to_blender(p) for p in edge_pos]
                            tri_verts.extend(_line_to_tris(bv[0], bv[1], EDGE_THICKNESS))
                            if not centers:
                                centers.append((bv[0] + bv[1]) / 2.0)
            if tri_verts:
                batch = batch_for_shader(shader, "TRIS", {"pos": tri_verts})
                batches.append((batch, "TRIS", color))
                if centers:
                    labels.append({
                        "pos_3d": centers[0] + Vector((0, 0, 0.05)),
                        "text": f"{count} {label_text}",
                        "color": color[:3] + (1.0,),
                    })

    return batches, labels
