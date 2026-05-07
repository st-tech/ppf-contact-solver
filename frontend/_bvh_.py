# File: _bvh_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""BVH-backed embedding helpers.

Thin wrappers over the Rust kernel that build a triangle BVH internally
and run the closest-triangle / frame-reconstruction queries needed by
:mod:`frontend._mesh_`.
"""

import numpy as np

from . import _rust  # type: ignore[attr-defined]


def frame_mapping(
    orig_vert: np.ndarray,
    new_vert: np.ndarray,
    new_tri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed each original vertex in the local frame of its closest new-mesh triangle.

    Args:
        orig_vert: Points to embed (N, 3).
        new_vert: Target mesh vertices (P, 3).
        new_tri: Target mesh triangles (Q, 3).

    Returns:
        tri_indices: Closest triangle index per input point (N,).
        coefs: Frame coefficients (c1, c2, c3) per input point (N, 3).
    """
    return _rust.frame_mapping(
        np.ascontiguousarray(orig_vert, dtype=np.float64),
        np.ascontiguousarray(new_vert, dtype=np.float64),
        np.ascontiguousarray(new_tri, dtype=np.int32),
    )


def interpolate_surface(
    deformed_vert: np.ndarray,
    surf_tri: np.ndarray,
    tri_indices: np.ndarray,
    coefs: np.ndarray,
) -> np.ndarray:
    """Reconstruct embedded point positions from a deformed host mesh.

    Args:
        deformed_vert: Deformed host mesh vertices (P, 3).
        surf_tri: Host mesh triangles (Q, 3).
        tri_indices: Closest triangle index per embedded point (N,).
        coefs: Frame coefficients (c1, c2, c3) per embedded point (N, 3).

    Returns:
        Reconstructed point positions (N, 3).
    """
    return _rust.interpolate_surface(
        np.ascontiguousarray(deformed_vert, dtype=np.float64),
        np.ascontiguousarray(surf_tri, dtype=np.int32),
        np.ascontiguousarray(tri_indices, dtype=np.int32),
        np.ascontiguousarray(coefs, dtype=np.float64),
    )
