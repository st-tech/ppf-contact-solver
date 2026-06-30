# File: _proximity_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Contact-offset proximity detection backed by the Rust kernel."""

from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]
from ._utils_ import _as_c


def check_contact_offset_violation(
    V: np.ndarray,
    F: Optional[np.ndarray] = None,
    E: Optional[np.ndarray] = None,
    is_collider: Optional[np.ndarray] = None,
    contact_offset: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """Check for contact-offset violations between mesh elements.

    Finds pairs of elements whose distance is smaller than the sum of
    their contact offsets.

    Args:
        V: Vertex positions, shape (N, 3).
        F: Triangle faces, shape (M, 3). Optional.
        E: Edge segments, shape (K, 2). Optional.
        is_collider: Boolean array of length M + K (triangles first, then
            edges) marking collider elements. Pairs where both elements
            are colliders are skipped. Defaults to all False.
        contact_offset: Per-element contact offset, shape (M + K,).
            Defaults to 0.0 for every element.

    Returns:
        A list of element index pairs (i, j) whose distance is strictly
        less than offset_i + offset_j. Indices 0..M-1 refer to triangles
        and M..M+K-1 refer to edges.
    """
    return _rust.check_contact_offset_violation(
        _as_c(V, np.float64),
        _as_c(F, np.int32),
        _as_c(E, np.int32),
        _as_c(is_collider, bool),
        _as_c(contact_offset, np.float64),
    )
