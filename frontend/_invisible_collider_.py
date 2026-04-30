# File: _invisible_collider_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Check dynamic vertices for violations against invisible colliders (walls and spheres)."""

import warnings

from typing import Optional

import numpy as np

from numba import njit, prange


@njit(cache=True)
def _check_wall_violation_single(
    vertex: np.ndarray,
    wall_pos: np.ndarray,
    wall_normal: np.ndarray,
) -> bool:
    """Check whether a single vertex violates a wall constraint.

    A vertex violates the wall when it lies on the side opposite the
    wall's outer normal (i.e. its signed distance to the plane is
    negative).

    Args:
        vertex: Vertex position, shape (3,).
        wall_pos: A point on the wall plane, shape (3,).
        wall_normal: Unit outer normal of the wall, shape (3,).

    Returns:
        True if the vertex violates the wall constraint.
    """
    # Signed distance from vertex to plane
    # Positive means vertex is on the normal side (correct side)
    diff = vertex - wall_pos
    signed_dist = diff[0] * wall_normal[0] + diff[1] * wall_normal[1] + diff[2] * wall_normal[2]

    # Violation if signed distance is negative (on wrong side of plane)
    return signed_dist < 0.0


@njit(parallel=True, cache=True)
def _check_wall_violations_parallel(
    vertices: np.ndarray,
    is_pinned: np.ndarray,
    wall_pos: np.ndarray,
    wall_normal: np.ndarray,
    violations: np.ndarray,
) -> int:
    """Check all vertices against a single wall in parallel.

    Args:
        vertices: All vertices, shape (N, 3).
        is_pinned: Boolean array marking pinned vertices, shape (N,).
            Pinned vertices are skipped.
        wall_pos: A point on the wall plane, shape (3,).
        wall_normal: Unit outer normal of the wall, shape (3,).
        violations: Output buffer, shape (N,). Entries corresponding to
            violating vertices are set to True; other entries are left
            unchanged.

    Returns:
        Total number of True entries in ``violations`` after the check.
    """
    n_verts = len(vertices)

    # Check violations in parallel
    for i in prange(n_verts):
        if not is_pinned[i] and _check_wall_violation_single(
            vertices[i], wall_pos, wall_normal
        ):
            violations[i] = True

    # Sequential count: a second ``prange`` here would create a back-
    # to-back parallel region within the same function call, which the
    # workqueue threading layer (numba's only option on macOS arm64
    # without TBB) flags as "concurrent access" and aborts the process.
    # The counting step is O(n) trivial work, so the loss of
    # parallelism is irrelevant at any realistic mesh size.
    count = 0
    for i in range(n_verts):
        if violations[i]:
            count += 1
    return count


@njit(cache=True)
def _check_sphere_violation(
    vertex: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius_sq: float,
    is_inverted: bool,
    is_hemisphere: bool,
) -> bool:
    """Check whether a single vertex violates a sphere constraint.

    Uses squared distance to avoid a square root. For a hemisphere, the
    effective center is lifted to the vertex's y-level whenever the
    vertex is above the sphere center, so the top of the hemisphere
    acts as an open cylinder.

    Args:
        vertex: Vertex position, shape (3,).
        sphere_center: Sphere center, shape (3,).
        sphere_radius_sq: Squared sphere radius.
        is_inverted: If True, the collider is the interior of the
            sphere (vertex must stay inside).
        is_hemisphere: If True, only the bottom half of the sphere is
            active (bowl shape).

    Returns:
        True if the vertex violates the sphere constraint.
    """
    # For hemisphere, if vertex is above center.y, project center to vertex's y-level
    if is_hemisphere and vertex[1] > sphere_center[1]:
        cx = sphere_center[0]
        cy = vertex[1]
        cz = sphere_center[2]
    else:
        cx = sphere_center[0]
        cy = sphere_center[1]
        cz = sphere_center[2]

    dx = vertex[0] - cx
    dy = vertex[1] - cy
    dz = vertex[2] - cz
    dist_sq = dx*dx + dy*dy + dz*dz

    if is_inverted:
        # For inverted sphere, vertex must be inside (distance <= radius)
        # Violation if distance > radius (outside the sphere)
        return dist_sq > sphere_radius_sq
    else:
        # For normal sphere, vertex must be outside (distance >= radius)
        # Violation if distance < radius (inside the sphere)
        return dist_sq < sphere_radius_sq


@njit(parallel=True, cache=True)
def _check_sphere_violations_parallel(
    vertices: np.ndarray,
    is_pinned: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
    is_inverted: bool,
    is_hemisphere: bool,
    violations: np.ndarray,
) -> int:
    """Check all vertices against a single sphere in parallel.

    Args:
        vertices: All vertices, shape (N, 3).
        is_pinned: Boolean array marking pinned vertices, shape (N,).
            Pinned vertices are skipped.
        sphere_center: Sphere center, shape (3,).
        sphere_radius: Sphere radius.
        is_inverted: If True, the collider is the interior of the
            sphere (vertex must stay inside).
        is_hemisphere: If True, only the bottom half of the sphere is
            active (bowl shape).
        violations: Output buffer, shape (N,). Entries corresponding to
            violating vertices are set to True; other entries are left
            unchanged.

    Returns:
        Total number of True entries in ``violations`` after the check.
    """
    n_verts = len(vertices)
    sphere_radius_sq = sphere_radius * sphere_radius

    # Check violations in parallel using squared distance
    for i in prange(n_verts):
        if not is_pinned[i] and _check_sphere_violation(
            vertices[i], sphere_center, sphere_radius_sq, is_inverted, is_hemisphere
        ):
            violations[i] = True

    # Sequential count (see _check_wall_violations_parallel for the
    # rationale: a second ``prange`` here trips workqueue's
    # concurrent-access detector on macOS arm64).
    count = 0
    for i in range(n_verts):
        if violations[i]:
            count += 1
    return count


def check_wall_violations(
    vertices: np.ndarray,
    walls: list,  # List of Wall objects
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
) -> list[tuple[int, int, float]]:
    """Check vertices against a list of static wall colliders.

    Walls with more than one keyframe entry (kinematic walls) are
    skipped; those are handled elsewhere. For static walls, only the
    initial keyframe position is used.

    Args:
        vertices: Vertex positions, shape (N, 3).
        walls: List of wall objects. Each wall must expose a
            ``get_entry()`` method returning a sequence of keyframes
            and a ``normal`` attribute.
        pinned_vertices: Optional set of vertex indices to skip.
        verbose: If True, print a per-wall violation count.

    Returns:
        List of ``(vertex_index, wall_index, signed_distance)`` tuples,
        one entry per violating (vertex, wall) pair. ``signed_distance``
        is negative for vertices on the wrong side of the wall.
    """
    if not walls:
        return []

    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    n_verts = len(vertices)

    # Build pinned array
    is_pinned = np.zeros(n_verts, dtype=np.bool_)
    if pinned_vertices:
        for vi in pinned_vertices:
            if 0 <= vi < n_verts:
                is_pinned[vi] = True

    all_violations: list[tuple[int, int, float]] = []

    for wall_idx, wall in enumerate(walls):
        entry = wall.get_entry()
        if not entry:
            continue

        # Skip kinematic walls (those with multiple keyframes)
        # Kinematic colliders use a different code path that handles violations gracefully
        if len(entry) > 1:
            continue

        # Use initial position (time=0)
        wall_pos = np.array(entry[0][0], dtype=np.float64)
        wall_normal = np.array(wall.normal, dtype=np.float64)
        # Normalize the normal
        norm = np.linalg.norm(wall_normal)
        if norm > 0:
            wall_normal = wall_normal / norm

        violations = np.zeros(n_verts, dtype=np.bool_)
        count = _check_wall_violations_parallel(
            vertices, is_pinned, wall_pos, wall_normal, violations
        )

        if count > 0:
            for vi in range(n_verts):
                if violations[vi]:
                    diff = vertices[vi] - wall_pos
                    signed_dist = np.dot(diff, wall_normal)
                    all_violations.append((vi, wall_idx, float(signed_dist)))

            if verbose:
                print(f"  Wall {wall_idx}: {count} vertex violations")

    return all_violations


def check_sphere_violations(
    vertices: np.ndarray,
    spheres: list,  # List of Sphere objects
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
) -> list[tuple[int, int, float]]:
    """Check vertices against a list of static sphere colliders.

    Spheres with more than one keyframe entry (kinematic spheres) are
    skipped; those are handled elsewhere. For static spheres, only the
    initial keyframe state is used.

    Args:
        vertices: Vertex positions, shape (N, 3).
        spheres: List of sphere objects. Each sphere must expose a
            ``get_entry()`` method returning a sequence of
            ``(position, radius, ...)`` keyframes, along with
            ``is_inverted`` and ``is_hemisphere`` attributes.
        pinned_vertices: Optional set of vertex indices to skip.
        verbose: If True, print a per-sphere violation count.

    Returns:
        List of ``(vertex_index, sphere_index, distance_to_surface)``
        tuples, one entry per violating (vertex, sphere) pair.
        ``distance_to_surface`` is computed as ``radius - distance`` to
        the (possibly hemisphere-adjusted) center, so it is positive
        when the vertex is inside the sphere and negative when outside.
    """
    if not spheres:
        return []

    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    n_verts = len(vertices)

    # Build pinned array
    is_pinned = np.zeros(n_verts, dtype=np.bool_)
    if pinned_vertices:
        for vi in pinned_vertices:
            if 0 <= vi < n_verts:
                is_pinned[vi] = True

    all_violations: list[tuple[int, int, float]] = []

    for sphere_idx, sphere in enumerate(spheres):
        entry = sphere.get_entry()
        if not entry:
            continue

        # Skip kinematic spheres (those with multiple keyframes)
        # Kinematic colliders use a different code path that handles violations gracefully
        if len(entry) > 1:
            continue

        # Use initial state (time=0)
        pos, radius, _ = entry[0]
        sphere_center = np.array(pos, dtype=np.float64)
        sphere_radius = float(radius)
        is_inverted = sphere.is_inverted
        is_hemisphere = sphere.is_hemisphere

        violations = np.zeros(n_verts, dtype=np.bool_)
        count = _check_sphere_violations_parallel(
            vertices, is_pinned, sphere_center, sphere_radius,
            is_inverted, is_hemisphere, violations
        )

        if count > 0:
            for vi in range(n_verts):
                if violations[vi]:
                    vertex = vertices[vi]
                    # For hemisphere, adjust center like in the check
                    if is_hemisphere and vertex[1] > sphere_center[1]:
                        center = np.array([sphere_center[0], vertex[1], sphere_center[2]])
                    else:
                        center = sphere_center

                    dist = np.linalg.norm(vertex - center)
                    # Distance to surface (positive = inside, negative = outside)
                    dist_to_surface = sphere_radius - dist
                    all_violations.append((vi, sphere_idx, float(dist_to_surface)))

            if verbose:
                mode = []
                if is_inverted:
                    mode.append("inverted")
                if is_hemisphere:
                    mode.append("hemisphere")
                mode_str = f" ({', '.join(mode)})" if mode else ""
                print(f"  Sphere {sphere_idx}{mode_str}: {count} vertex violations")

    return all_violations


def check_invisible_collider_violations(
    vertices: np.ndarray,
    walls: list,
    spheres: list,
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Check vertices against both wall and sphere invisible colliders.

    Args:
        vertices: Vertex positions, shape (N, 3).
        walls: List of wall objects (see :func:`check_wall_violations`).
        spheres: List of sphere objects (see
            :func:`check_sphere_violations`).
        pinned_vertices: Optional set of vertex indices to skip.
        verbose: If True, print per-collider violation counts.

    Returns:
        A tuple ``(wall_violations, sphere_violations)`` as produced by
        :func:`check_wall_violations` and
        :func:`check_sphere_violations` respectively.
    """
    wall_violations = check_wall_violations(vertices, walls, pinned_vertices, verbose)
    sphere_violations = check_sphere_violations(vertices, spheres, pinned_vertices, verbose)

    return wall_violations, sphere_violations
