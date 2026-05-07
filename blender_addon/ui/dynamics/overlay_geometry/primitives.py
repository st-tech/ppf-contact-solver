# File: primitives.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import math

from mathutils import Vector  # pyright: ignore


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
