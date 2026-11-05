# File: _tie_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""Least-norm affine weights for :meth:`Scene.tie`."""

import numpy as np

from ._scene_ import TIE_SLOTS


def tie_weights(points, candidates, k: int = TIE_SLOTS, max_weight: float = 4.0):
    """Affine weights that hold each point on its nearest candidate vertices.

    For each point ``p`` the ``k`` nearest ``candidates`` are taken and the
    least-norm weights solving ``sum_i w_i c_i = p``, ``sum_i w_i = 1`` are
    returned. The combination reproduces ``p`` exactly when the ``k``
    candidates span 3D affinely; a candidate set that is flat (coplanar, or
    nearly so) cannot reach a point off its plane, and one that is nearly flat
    reaches it only with large weights. Both are refused: a residual above
    round-off, or a weight larger than ``max_weight`` in magnitude, names the
    point.

    Args:
        points: ``(N, 3)`` positions to hold, in the coordinates the scene is
            built in.
        candidates: ``(M, 3)`` positions of the vertices that may control them.
        k (int, optional): Controllers per point, at most 8.
        max_weight (float, optional): The largest admissible ``|w|``.

    Returns:
        tuple: ``(ind, w)``, each ``(N, k)``: indices into ``candidates`` and
        the weights, ready for :meth:`Scene.tie`.

    Raises:
        ValueError: If ``k`` is out of range, or a point's candidate set is
            degenerate or needs a weight beyond ``max_weight``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    if not 4 <= k <= TIE_SLOTS:
        raise ValueError(f"tie_weights takes 4 <= k <= {TIE_SLOTS}, got {k}")
    if len(candidates) < k:
        raise ValueError(f"tie_weights needs at least {k} candidates, got {len(candidates)}")
    try:
        from scipy.spatial import KDTree

        _, ind = KDTree(candidates).query(points, k=k)
    except ImportError:
        d = np.linalg.norm(points[:, None, :] - candidates[None, :, :], axis=2)
        ind = np.argsort(d, axis=1)[:, :k]
    ind = np.asarray(ind, dtype=np.int64).reshape(len(points), k)
    w = np.empty((len(points), k), dtype=np.float64)
    scale = max(1.0, float(np.abs(candidates).max()))
    for n, p in enumerate(points):
        c = candidates[ind[n]]
        center = c.mean(axis=0)
        # Centered, so the affine row is well conditioned against the others.
        a = np.vstack([(c - center).T, np.ones(k)])
        b = np.r_[p - center, 1.0]
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        miss = np.linalg.norm(a @ sol - b)
        if miss > 1e-9 * scale:
            raise ValueError(
                f"tie_weights: point {n} is off the affine span of its {k} nearest "
                f"candidates (residual {miss:.3e}); they are flat, so take more of "
                "them or candidates from both sides"
            )
        if np.abs(sol).max() > max_weight:
            raise ValueError(
                f"tie_weights: point {n} needs a weight of {np.abs(sol).max():.2f} on its "
                f"{k} nearest candidates, above {max_weight}; they are nearly flat"
            )
        w[n] = sol
    return ind, w
