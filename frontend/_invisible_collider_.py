# File: _invisible_collider_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Check dynamic vertices for violations against invisible colliders (walls and spheres).

These wrappers are now used only by the test suite; production scene
validation runs the same checks inside ``_rust.scene_fixed_scene_assemble``.
"""

from collections import Counter
from itertools import repeat
from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]
from ._utils_ import _as_c


def _coerce_inputs(
    vertices: np.ndarray,
    pinned_vertices: Optional[set[int]],
    is_pinned: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous float64 vertices and a pinned mask.

    When ``is_pinned`` is already provided (combined path), the vertex
    array is still coerced but the mask is reused so the per-vertex
    mask build runs only once. ``build_pinned_mask`` is a pure function
    of ``(n_verts, pinned_vertices)``.
    """
    vertices = _as_c(vertices, np.float64)
    if is_pinned is None:
        is_pinned = _rust.build_pinned_mask(
            len(vertices),
            list(pinned_vertices) if pinned_vertices else None,
        )
    return vertices, is_pinned


def _print_violation_counts(
    label: str,
    all_violations: list,
    static_indices: list,
    mode_tags: Optional[list] = None,
) -> None:
    """Print a per-collider violation count line for each static index.

    ``mode_tags`` supplies an optional per-collider tag string (used by
    spheres for the ``" (inverted, hemisphere)"`` mode suffix); when
    omitted every tag is empty so the output matches the untagged form.
    """
    counts = Counter(v[1] for v in all_violations)
    for idx, tag in zip(static_indices, mode_tags or repeat("")):
        print(f"  {label} {idx}{tag}: {counts.get(idx, 0)} vertex violations")


def check_wall_violations(
    vertices: np.ndarray,
    walls: list,  # List of Wall objects
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
    is_pinned: Optional[np.ndarray] = None,
) -> list[tuple[int, int, float]]:
    """Check vertices against a list of static wall colliders.

    Walls for which ``Wall.is_static_collider()`` is ``False`` (empty or
    kinematic walls) are skipped; those are handled elsewhere. For static
    walls, only the initial keyframe position is used.

    Args:
        vertices: Vertex positions, shape (N, 3).
        walls: List of wall objects. Each wall must expose a
            ``get_entry()`` method returning a sequence of keyframes
            and a ``normal`` attribute.
        pinned_vertices: Optional set of vertex indices to skip.
        verbose: If True, print a per-wall violation count.
        is_pinned: Optional pre-built pinned mask. When provided it is
            reused as-is; otherwise it is built from ``pinned_vertices``.

    Returns:
        List of ``(vertex_index, wall_index, signed_distance)`` tuples,
        one entry per violating (vertex, wall) pair. ``signed_distance``
        is negative for vertices on the wrong side of the wall.
    """
    if not walls:
        return []

    vertices, is_pinned = _coerce_inputs(vertices, pinned_vertices, is_pinned)

    # Static-wall extraction (entry[0], normal normalization,
    # kinematic-skip) and the per-wall scan run in a single Rust call.
    all_violations, static_indices = _rust.check_walls_violations_for_objs(
        vertices, is_pinned, walls,
    )
    if verbose:
        _print_violation_counts("Wall", all_violations, static_indices)

    return all_violations


def check_sphere_violations(
    vertices: np.ndarray,
    spheres: list,  # List of Sphere objects
    pinned_vertices: Optional[set[int]] = None,
    verbose: bool = False,
    is_pinned: Optional[np.ndarray] = None,
) -> list[tuple[int, int, float]]:
    """Check vertices against a list of static sphere colliders.

    Spheres for which ``Sphere.is_static_collider()`` is ``False`` (empty
    or kinematic spheres) are skipped; those are handled elsewhere. For
    static spheres, only the initial keyframe state is used.

    Args:
        vertices: Vertex positions, shape (N, 3).
        spheres: List of sphere objects. Each sphere must expose a
            ``get_entry()`` method returning a sequence of
            ``(position, radius, ...)`` keyframes, along with
            ``is_inverted`` and ``is_hemisphere`` attributes.
        pinned_vertices: Optional set of vertex indices to skip.
        verbose: If True, print a per-sphere violation count.
        is_pinned: Optional pre-built pinned mask. When provided it is
            reused as-is; otherwise it is built from ``pinned_vertices``.

    Returns:
        List of ``(vertex_index, sphere_index, distance_to_surface)``
        tuples, one entry per violating (vertex, sphere) pair.
        ``distance_to_surface`` is computed as ``radius - distance`` to
        the (possibly hemisphere-adjusted) center, so it is positive
        when the vertex is inside the sphere and negative when outside.
    """
    if not spheres:
        return []

    vertices, is_pinned = _coerce_inputs(vertices, pinned_vertices, is_pinned)

    # The per-sphere static filter, the violation scan, and the
    # ``" (inverted, hemisphere)"`` mode-tag formatting run in a single
    # Rust call.
    all_violations, static_indices, mode_tags = (
        _rust.check_spheres_violations_for_objs(vertices, is_pinned, spheres)
    )
    if verbose:
        _print_violation_counts(
            "Sphere", all_violations, static_indices, mode_tags
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
    # Coerce vertices and build the pinned mask once (only when at least
    # one collider list is non-empty), then forward both to the
    # sub-checks so the contiguous copy and the per-vertex mask build do
    # not run twice over identical inputs.
    is_pinned: Optional[np.ndarray] = None
    if walls or spheres:
        vertices, is_pinned = _coerce_inputs(vertices, pinned_vertices, None)

    wall_violations = check_wall_violations(
        vertices, walls, pinned_vertices, verbose, is_pinned=is_pinned
    )
    sphere_violations = check_sphere_violations(
        vertices, spheres, pinned_vertices, verbose, is_pinned=is_pinned
    )

    return wall_violations, sphere_violations
