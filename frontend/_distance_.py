# File: _distance_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Distance computation functions for triangle meshes using Numba."""

import numpy as np

from numba import njit


@njit(cache=True)
def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _norm_sq(a: np.ndarray) -> float:
    """Compute squared norm of a 3D vector."""
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]


@njit(cache=True)
def _solve_2x2(a00: float, a01: float, a10: float, a11: float, b0: float, b1: float):
    """Solve 2x2 linear system using Cramer's rule. Returns (x0, x1, det)."""
    det = a00 * a11 - a10 * a01
    x0 = a11 * b0 - a01 * b1
    x1 = -a10 * b0 + a00 * b1
    return x0, x1, det


@njit(cache=True)
def point_edge_distance_coeff(
    p: np.ndarray, e0: np.ndarray, e1: np.ndarray
) -> np.ndarray:
    """Find barycentric coordinates of closest point on edge to point p."""
    r = e1 - e0
    d = _dot(r, r)
    if d > 0.0:
        t = _dot(r, p - e0) / d
        return np.array([1.0 - t, t])
    else:
        return np.array([0.5, 0.5])


@njit(cache=True)
def point_edge_distance_coeff_clamped(
    p: np.ndarray, e0: np.ndarray, e1: np.ndarray
) -> np.ndarray:
    """Find clamped barycentric coordinates of closest point on edge segment."""
    c = point_edge_distance_coeff(p, e0, e1)
    if c[0] >= 0.0 and c[0] <= 1.0:
        return c
    elif c[0] > 1.0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])


@njit(cache=True)
def point_triangle_distance_coeff(
    p: np.ndarray, t0: np.ndarray, t1: np.ndarray, t2: np.ndarray
) -> np.ndarray:
    """Find barycentric coordinates of closest point on triangle plane to point p."""
    r0 = t1 - t0
    r1 = t2 - t0
    dp = p - t0

    a00 = _dot(r0, r0)
    a01 = _dot(r0, r1)
    a10 = a01
    a11 = _dot(r1, r1)

    b0 = _dot(r0, dp)
    b1 = _dot(r1, dp)

    x0, x1, det = _solve_2x2(a00, a01, a10, a11, b0, b1)

    if abs(det) > 1e-14:
        c0 = x0 / det
        c1 = x1 / det
        return np.array([1.0 - c0 - c1, c0, c1])
    else:
        len0 = _norm_sq(r0)
        len1 = _norm_sq(t2 - t1)
        len2 = _norm_sq(t0 - t2)
        if len0 >= len1 and len0 >= len2:
            w = point_edge_distance_coeff(p, t0, t1)
            return np.array([w[0], w[1], 0.0])
        elif len1 >= len2:
            w = point_edge_distance_coeff(p, t1, t2)
            return np.array([0.0, w[0], w[1]])
        else:
            w = point_edge_distance_coeff(p, t2, t0)
            return np.array([w[1], 0.0, w[0]])


@njit(cache=True)
def point_triangle_distance_coeff_clamped(
    p: np.ndarray, t0: np.ndarray, t1: np.ndarray, t2: np.ndarray
) -> np.ndarray:
    """Find clamped barycentric coordinates of closest point on triangle."""
    c = point_triangle_distance_coeff(p, t0, t1, t2)

    if (
        c[0] >= 0.0 and c[1] >= 0.0 and c[2] >= 0.0
        and c[0] <= 1.0 and c[1] <= 1.0 and c[2] <= 1.0
    ):
        return c

    if c[0] < 0.0:
        w = point_edge_distance_coeff_clamped(p, t1, t2)
        return np.array([0.0, w[0], w[1]])
    elif c[1] < 0.0:
        w = point_edge_distance_coeff_clamped(p, t0, t2)
        return np.array([w[0], 0.0, w[1]])
    else:
        w = point_edge_distance_coeff_clamped(p, t0, t1)
        return np.array([w[0], w[1], 0.0])


@njit(cache=True)
def edge_edge_distance_coeff(
    ea0: np.ndarray, ea1: np.ndarray, eb0: np.ndarray, eb1: np.ndarray
) -> np.ndarray:
    """Find coefficients for closest points between two edges."""
    r0 = ea1 - ea0
    r1 = eb1 - eb0
    d = eb0 - ea0

    a00 = _dot(r0, r0)
    a01 = -_dot(r0, r1)
    a10 = a01
    a11 = _dot(r1, r1)

    b0 = _dot(r0, d)
    b1 = -_dot(r1, d)

    x0, x1, det = _solve_2x2(a00, a01, a10, a11, b0, b1)

    c = point_edge_distance_coeff(ea0, eb0, eb1)
    result = np.array([1.0, 0.0, c[0], c[1]])
    pa = ea0.copy()
    pb = c[0] * eb0 + c[1] * eb1
    diff = pa - pb
    min_dist = _dot(diff, diff)

    if abs(det) > 1e-14:
        ta = x0 / det
        tb = x1 / det
        direct = np.array([1.0 - ta, ta, 1.0 - tb, tb])
        pa_direct = direct[0] * ea0 + direct[1] * ea1
        pb_direct = direct[2] * eb0 + direct[3] * eb1
        diff_direct = pa_direct - pb_direct
        dist = _dot(diff_direct, diff_direct)
        if dist < min_dist:
            result = direct

    x = np.array([result[1], result[3]])
    q0 = eb0 - ea0
    q1 = eb1 - ea0
    p0 = ea0 - eb0
    p1 = ea1 - eb0

    for _ in range(4):
        w1 = point_edge_distance_coeff(x[1] * r1, p0, p1)
        x[0] = w1[1]
        w0 = point_edge_distance_coeff(x[0] * r0, q0, q1)
        x[1] = w0[1]

    return np.array([1.0 - x[0], x[0], 1.0 - x[1], x[1]])


@njit(cache=True)
def edge_edge_distance_coeff_clamped(
    ea0: np.ndarray, ea1: np.ndarray, eb0: np.ndarray, eb1: np.ndarray
) -> np.ndarray:
    """Find clamped coefficients for closest points between two edge segments."""
    c = edge_edge_distance_coeff(ea0, ea1, eb0, eb1)

    if (
        c[0] >= 0.0 and c[1] >= 0.0 and c[2] >= 0.0 and c[3] >= 0.0
        and c[0] <= 1.0 and c[1] <= 1.0 and c[2] <= 1.0 and c[3] <= 1.0
    ):
        return c

    c1 = point_edge_distance_coeff_clamped(ea0, eb0, eb1)
    c2 = point_edge_distance_coeff_clamped(ea1, eb0, eb1)
    c3 = point_edge_distance_coeff_clamped(eb0, ea0, ea1)
    c4 = point_edge_distance_coeff_clamped(eb1, ea0, ea1)

    types = np.zeros((4, 4))
    types[0] = np.array([1.0, 0.0, c1[0], c1[1]])
    types[1] = np.array([0.0, 1.0, c2[0], c2[1]])
    types[2] = np.array([c3[0], c3[1], 1.0, 0.0])
    types[3] = np.array([c4[0], c4[1], 0.0, 1.0])

    min_dist = np.inf
    best_idx = 0

    for i in range(4):
        coeff = types[i]
        pa = coeff[0] * ea0 + coeff[1] * ea1
        pb = coeff[2] * eb0 + coeff[3] * eb1
        diff = pa - pb
        dist = _dot(diff, diff)
        if dist < min_dist:
            min_dist = dist
            best_idx = i

    return types[best_idx]


@njit(cache=True)
def point_point_distance_sq(p0: np.ndarray, p1: np.ndarray) -> float:
    """Compute squared distance between two points."""
    diff = p0 - p1
    return _dot(diff, diff)


@njit(cache=True)
def point_edge_distance_sq(
    p: np.ndarray, e0: np.ndarray, e1: np.ndarray
) -> float:
    """Compute squared distance from point to edge segment."""
    c = point_edge_distance_coeff_clamped(p, e0, e1)
    closest = c[0] * e0 + c[1] * e1
    diff = p - closest
    return _dot(diff, diff)


@njit(cache=True)
def point_triangle_distance_sq(
    p: np.ndarray, t0: np.ndarray, t1: np.ndarray, t2: np.ndarray
) -> float:
    """Compute squared distance from point to triangle."""
    c = point_triangle_distance_coeff_clamped(p, t0, t1, t2)
    closest = c[0] * t0 + c[1] * t1 + c[2] * t2
    diff = p - closest
    return _dot(diff, diff)


@njit(cache=True)
def edge_edge_distance_sq(
    ea0: np.ndarray, ea1: np.ndarray, eb0: np.ndarray, eb1: np.ndarray
) -> float:
    """Compute squared distance between two edge segments."""
    c = edge_edge_distance_coeff_clamped(ea0, ea1, eb0, eb1)
    pa = c[0] * ea0 + c[1] * ea1
    pb = c[2] * eb0 + c[3] * eb1
    diff = pa - pb
    return _dot(diff, diff)
