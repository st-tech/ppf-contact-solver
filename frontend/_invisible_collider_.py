# File: _invisible_collider_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Check for violations of dynamic vertices against invisible colliders (walls and spheres)."""

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
    """Check if a single vertex violates wall constraint.

    A vertex violates the wall if it's on the wrong side of the plane
    (opposite to the normal direction).

    Args:
        vertex: Vertex position (3,)
        wall_pos: A point on the wall plane (3,)
        wall_normal: Outer normal of the wall (3,)

    Returns:
        True if vertex violates the wall constraint
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
        vertices: All vertices (N, 3)
        is_pinned: Boolean array indicating pinned vertices (N,)
        wall_pos: A point on the wall plane (3,)
        wall_normal: Outer normal of the wall (3,)
        violations: Output array for violation flags (N,)

    Returns:
        Number of violations found
    """
    n_verts = len(vertices)

    # Check violations in parallel
    for i in prange(n_verts):
        if not is_pinned[i] and _check_wall_violation_single(
            vertices[i], wall_pos, wall_normal
        ):
            violations[i] = True

    # Parallel reduction for counting
    count = 0
    for i in prange(n_verts):
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
    """Check if a single vertex violates sphere constraint using squared distance.

    This version avoids sqrt for the check, only computing it when needed.
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
        vertices: All vertices (N, 3)
        is_pinned: Boolean array indicating pinned vertices (N,)
        sphere_center: Center of sphere (3,)
        sphere_radius: Radius of sphere
        is_inverted: If True, collision is with inside of sphere
        is_hemisphere: If True, top half is open (bowl shape)
        violations: Output array for violation flags (N,)

    Returns:
        Number of violations found
    """
    n_verts = len(vertices)
    sphere_radius_sq = sphere_radius * sphere_radius

    # Check violations in parallel using squared distance
    for i in prange(n_verts):
        if not is_pinned[i] and _check_sphere_violation(
            vertices[i], sphere_center, sphere_radius_sq, is_inverted, is_hemisphere
        ):
            violations[i] = True

    # Parallel reduction for counting
    count = 0
    for i in prange(n_verts):
        if violations[i]:
            count += 1
    return count


def check_wall_violations(
    vertices: np.ndarray,
    walls: list,  # List of Wall objects
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
) -> list[tuple[int, int, float]]:
    """Check for wall constraint violations.

    Args:
        vertices: Vertex positions (N, 3)
        walls: List of Wall objects with position and normal
        pinned_vertices: Set of pinned vertex indices to skip
        verbose: If True, print violation details

    Returns:
        List of (vertex_index, wall_index, signed_distance) tuples for violations
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
                warnings.warn(
                    f"Wall {wall_idx}: {count} vertex violations", stacklevel=1
                )

    return all_violations


def check_sphere_violations(
    vertices: np.ndarray,
    spheres: list,  # List of Sphere objects
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
) -> list[tuple[int, int, float]]:
    """Check for sphere constraint violations.

    Args:
        vertices: Vertex positions (N, 3)
        spheres: List of Sphere objects
        pinned_vertices: Set of pinned vertex indices to skip
        verbose: If True, print violation details

    Returns:
        List of (vertex_index, sphere_index, distance_to_surface) tuples for violations.
        Positive distance means inside sphere, negative means outside.
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
                warnings.warn(
                    f"Sphere {sphere_idx}{mode_str}: {count} vertex violations",
                    stacklevel=1,
                )

    return all_violations


def check_invisible_collider_violations(
    vertices: np.ndarray,
    walls: list,
    spheres: list,
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Check for all invisible collider violations.

    Args:
        vertices: Vertex positions (N, 3)
        walls: List of Wall objects
        spheres: List of Sphere objects
        pinned_vertices: Set of pinned vertex indices to skip
        verbose: If True, print violation details

    Returns:
        Tuple of (wall_violations, sphere_violations)
    """
    wall_violations = check_wall_violations(vertices, walls, pinned_vertices, verbose)
    sphere_violations = check_sphere_violations(vertices, spheres, pinned_vertices, verbose)

    return wall_violations, sphere_violations
