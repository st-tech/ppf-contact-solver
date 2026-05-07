# File: _invisible_collider_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Check dynamic vertices for violations against invisible colliders (walls and spheres)."""

from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]


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
    is_pinned = _rust.build_pinned_mask(
        n_verts,
        list(pinned_vertices) if pinned_vertices else None,
    )

    # Filter + scan in one Rust call. Static-wall extraction (entry[0],
    # normal normalization, kinematic-skip) and the per-wall scan are
    # both inside Rust so no Python list grows in this hot path.
    all_violations, static_indices = _rust.check_walls_violations_for_objs(
        vertices, is_pinned, walls,
    )
    if verbose:
        from collections import Counter

        counts = Counter(v[1] for v in all_violations)
        for wall_idx in static_indices:
            print(
                f"  Wall {wall_idx}: {counts.get(wall_idx, 0)} vertex violations"
            )

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
    is_pinned = _rust.build_pinned_mask(
        n_verts,
        list(pinned_vertices) if pinned_vertices else None,
    )

    # Filter + scan + mode-tag formatting in one Rust call. The
    # per-sphere static filter, the violation scan, and the
    # ``" (inverted, hemisphere)"`` tag string all live in Rust so
    # nothing grows a Python list here.
    all_violations, static_indices, mode_tags = (
        _rust.check_spheres_violations_for_objs(vertices, is_pinned, spheres)
    )
    if verbose:
        from collections import Counter

        counts = Counter(v[1] for v in all_violations)
        for sphere_idx, mode_str in zip(static_indices, mode_tags):
            print(
                f"  Sphere {sphere_idx}{mode_str}: "
                f"{counts.get(sphere_idx, 0)} vertex violations"
            )

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
