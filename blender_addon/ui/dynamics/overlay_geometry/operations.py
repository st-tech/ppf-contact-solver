# File: operations.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import gpu  # pyright: ignore

from gpu_extras.batch import batch_for_shader  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ...state import decode_vertex_group_identifier, iterate_active_object_groups

from .primitives import (
    _compute_pca_axes,
    _generate_arrow,
    _generate_circle,
    _generate_rotation_arc,
    _generate_vertex_arrow,
    _max_towards_centroid,
    _max_towards_selected,
)


_OP_COLORS = {
    "SPIN": (1.0, 0.6, 0.2),
    "MOVE_BY": (0.3, 0.9, 0.9),
    "SCALE": (0.9, 0.3, 0.8),
    "TORQUE": (0.2, 0.8, 0.4),
}


def _build_static_op_batches(group, depsgraph, shader, batches, labels):
    """Append overlay batches/labels for UI-assigned static ops on *group*.

    Mirrors the pin-op overlay shapes — MOVE_BY arrows from each vertex,
    SPIN circle + sweep arc, SCALE arrows from each vertex — but walks
    per-object because static ops live on AssignedObject.static_ops.
    """
    from ....core.uuid_registry import get_object_by_uuid

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
                        "TRIS",
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
                        "TRIS",
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
                        "TRIS",
                        (*rgb, 0.6),
                    ))
                labels.append({
                    "text": f"\u00d7{factor:.2f}",
                    "pos_3d": center,
                    "color": (*rgb, 0.9),
                })


def _build_operation_batches(scene, depsgraph, view_distance=10.0):
    """Build GPU batches and labels for pin operation overlays.

    Returns list of (batch, primitive_type, color) tuples. POINTS batches
    are built with POINT_UNIFORM_COLOR so the shader writes gl_PointSize;
    Metal (Blender 5.x macOS) has no fixed-function point size and
    gpu.state.point_size_set is a no-op.
    """
    from ....core.transform import _to_blender

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    point_shader = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
    batches = []
    labels = []

    for group in iterate_active_object_groups(scene):
        if group.object_type == "STATIC":
            _build_static_op_batches(
                group, depsgraph, shader, batches, labels,
            )
            continue
        from ....core.uuid_registry import resolve_pin
        for pin_ref in group.pin_vertex_groups:
            obj = resolve_pin(pin_ref)
            _, pin_vg_name = decode_vertex_group_identifier(pin_ref.name)
            if not obj or not pin_vg_name:
                continue

            has_visible = any(
                op.show_overlay for op in pin_ref.operations
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
                if not op.show_overlay:
                    continue

                rgb = _OP_COLORS.get(op.op_type, (0.8, 0.8, 0.8))

                if op.op_type == "SPIN":
                    if op.spin_center_mode == "ABSOLUTE":
                        center = Vector(op.spin_center)
                    elif op.spin_center_mode == "MAX_TOWARDS":
                        center = _max_towards_centroid(vertices, Vector(op.spin_center_direction))
                    elif op.spin_center_mode == "VERTEX" and 0 <= op.spin_center_vertex < len(mesh.vertices):
                        center = world_matrix @ mesh.vertices[op.spin_center_vertex].co
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
                        batches.append((batch, "TRIS", (*rgb, 0.7)))

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
                        batches.append((batch, "TRIS", (*rgb, 0.6)))

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
                    elif op.scale_center_mode == "VERTEX" and 0 <= op.scale_center_vertex < len(mesh.vertices):
                        center = world_matrix @ mesh.vertices[op.scale_center_vertex].co
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
                        batches.append((batch, "TRIS", (*rgb, 0.6)))

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
                    axis = Vector(_to_blender(sax)).normalized()

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
                        batches.append((batch, "TRIS", (*rgb, 0.7)))

                    labels.append({
                        "text": f"\u03c4 = {op.torque_magnitude:.1f} N\u00b7m ({op.torque_axis_component})",
                        "pos_3d": center + axis * avg_radius * 0.1,
                        "color": (*rgb, 0.9),
                    })

            # Max Towards visualization: highlight selected vertices + direction arrow
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
                # Same selection rule as the resolved center (eps lives in the helper).
                selected_verts = _max_towards_selected(vertices, direction)
                if not selected_verts:
                    continue
                # Highlighted points batch (drawn as small spheres via POINTS)
                point_batch = batch_for_shader(point_shader, "POINTS", {"pos": selected_verts})
                batches.append((point_batch, "POINTS", (1.0, 0.8, 0.0, 0.9)))
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
                    batches.append((batch_for_shader(shader, "TRIS", {"pos": shaft_tris}), "TRIS", (1.0, 0.8, 0.0, 0.7)))
                if cone_tris:
                    batches.append((batch_for_shader(shader, "TRIS", {"pos": cone_tris}), "TRIS", (1.0, 0.8, 0.0, 0.9)))
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
                point_batch = batch_for_shader(point_shader, "POINTS", {"pos": [pos]})
                batches.append((point_batch, "POINTS", (0.0, 1.0, 0.5, 0.9)))
                labels.append({
                    "text": f"Center (v{vi})",
                    "pos_3d": pos,
                    "color": (0.0, 1.0, 0.5, 0.9),
                })

    return batches, labels


def _build_pdrd_hinge_batches(scene, depsgraph, view_distance=10.0):
    """Build hinge-axle gizmos for PDRD bodies with the hinge enabled.

    For each included PDRD body whose ``pdrd_hinge_enable`` is set (a per-object
    setting), draw a ring (plus a full-turn arc) in the plane perpendicular to
    the chosen principal (PCA) axis through the body centroid. This is exactly
    the axle the solver pins (see ``Object.hinge`` and the frontend PCA bake),
    and the axis ordering matches (``_compute_pca_axes`` returns PC1/PC2/PC3 in
    descending eigenvalue order, the same convention the frontend uses).

    Returns ``(batches, labels)`` in the same shape as
    :func:`_build_operation_batches`.
    """
    from ....core.uuid_registry import resolve_assigned

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batches = []
    labels = []
    rgb = (0.2, 0.7, 1.0)  # cyan axle

    for group in iterate_active_object_groups(scene):
        if group.object_type != "PDRD":
            continue
        if not getattr(group, "pdrd_hinge_visualize", True):
            continue
        for assigned in group.assigned_objects:
            if not getattr(assigned, "included", True):
                continue
            if not getattr(assigned, "pdrd_hinge_enable", False):
                continue
            comp_idx = {"0": 0, "1": 1, "2": 2}.get(
                str(assigned.pdrd_hinge_axis), 2
            )
            obj = resolve_assigned(assigned)
            if obj is None or obj.type != "MESH":
                continue
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.data
            world_matrix = eval_obj.matrix_world
            vertices = [world_matrix @ v.co for v in mesh.vertices]
            if len(vertices) < 2:
                continue
            pca = _compute_pca_axes(vertices)
            if pca is None:
                continue
            center, eigvecs = pca
            axis = Vector((
                float(eigvecs[0, comp_idx]),
                float(eigvecs[1, comp_idx]),
                float(eigvecs[2, comp_idx]),
            ))
            if axis.length < 1e-6:
                continue
            axis = axis.normalized()
            # Ring radius = mean perpendicular distance to the axle (matches
            # the TORQUE gizmo so the two read consistently).
            total = 0.0
            for v in vertices:
                diff = v - center
                total += (diff - axis * diff.dot(axis)).length
            avg_radius = total / len(vertices)
            if avg_radius < 1e-4:
                avg_radius = 0.1
            thickness = max(0.002, avg_radius * 0.01)
            circle_tris = _generate_circle(
                center, axis, avg_radius, thickness=thickness,
            )
            arc_tris = _generate_rotation_arc(
                center, axis, avg_radius, 360.0, thickness=thickness * 2,
            )
            all_tris = circle_tris + arc_tris
            if all_tris:
                batch = batch_for_shader(shader, "TRIS", {"pos": all_tris})
                batches.append((batch, "TRIS", (*rgb, 0.7)))
            labels.append({
                "text": f"hinge (PC{comp_idx + 1})",
                "pos_3d": center + axis * avg_radius * 0.1,
                "color": (*rgb, 0.9),
            })

    return batches, labels
