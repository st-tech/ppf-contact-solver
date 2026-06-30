# File: _intersection_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Self-intersection detection backed by the Rust kernel."""

from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]
from ._utils_ import _as_c


def check_self_intersection(
    V: np.ndarray,
    F: np.ndarray,
    is_collider: Optional[np.ndarray] = None,
    rod_edges: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """Check for self-intersections in a triangle mesh, optionally including rod edges.

    Uses edge-triangle intersection tests: when two triangles intersect
    in general position, at least one edge of one triangle pierces the
    other. Coplanar overlap is handled as a separate fallback test.

    Rod edges (from open curves) are tested against every triangle.
    Their hits are reported separately as ``(-1, tri_idx)`` pairs.

    Args:
        V: Vertices (N, 3).
        F: Triangle faces (M, 3).
        is_collider: Optional boolean array (M,) marking triangles that
            belong to collider meshes. Pairs where both triangles are
            colliders are skipped.
        rod_edges: Optional edge array (K, 2) for rod/curve edges to
            test against triangles.

    Returns:
        Sorted list of intersecting pairs. Triangle-triangle pairs are
        ``(i, j)`` with ``i < j``; rod-triangle hits appear as
        ``(-1, tri_idx)`` and are appended after the triangle-triangle
        pairs. Empty list if no intersections are found.
    """
    return _rust.check_self_intersection(
        _as_c(V, np.float64),
        _as_c(F, np.int32),
        _as_c(is_collider, bool),
        _as_c(rod_edges, np.int32)
        if rod_edges is not None and len(rod_edges) > 0
        else None,
    )
