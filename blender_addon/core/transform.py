# File: transform.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np

from mathutils import Matrix  # pyright: ignore


def _swap_axes(v) -> list[float]:
    """Convert a direction from Blender (Z-up) to solver (Y-up) coordinates.

    Use for directions only (gravity, wind, spin axis, normals).
    For positions, use ``_to_solver()`` instead.
    """
    return [float(v[0]), float(v[2]), float(-v[1])]


def _to_solver(v) -> list[float]:
    """Convert a position from Blender world space to solver space.

    Axis swap only (Z-up → Y-up).  Use for any absolute coordinate
    (centers, wall/sphere positions, deltas, etc.).
    """
    return [float(v[0]), float(v[2]), float(-v[1])]


def _normalize_and_scale(direction, strength) -> list[float]:
    """Normalize a direction vector and multiply by strength."""
    d = np.array([float(direction[i]) for i in range(3)], dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm > 0:
        d = d / norm
    return (d * float(strength)).tolist()


def zup_to_yup():
    return Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def world_matrix(obj):
    return zup_to_yup() @ obj.matrix_world if obj else Matrix.Identity(4)


def inv_world_matrix(obj):
    return world_matrix(obj).inverted() if obj else Matrix.Identity(4)
