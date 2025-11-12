# File: eigsolve3x3.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
import numpy.linalg as LA


def eigvalues(A):
    p1 = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
    q = np.trace(A) / 3.0
    p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    if abs(p) < 1e-8:
        return None
    else:
        B = (1.0 / p) * (A - q * np.eye(3))
        r = np.linalg.det(B) / 2.0
        if r <= -1.0:
            phi = np.pi / 3.0
        elif r >= 1.0:
            phi = 0.0
        else:
            phi = np.arccos(r) / 3.0
        eig1 = q + 2.0 * p * np.cos(phi)
        eig3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
        eig2 = 3.0 * q - eig1 - eig3
        return np.array([eig1, eig2, eig3])


def pick_largest(a, b, c):
    if a.dot(a) > b.dot(b):
        if a.dot(a) > c.dot(c):
            return a
        else:
            return c
    else:
        if b.dot(b) > c.dot(c):
            return b
        else:
            return c


def find_ortho(A):
    eps = 1e-8
    u, v, w = A[:, 0], A[:, 1], A[:, 2]
    uv, vw, wu = np.cross(u, v), np.cross(v, w), np.cross(w, u)
    q = pick_largest(uv, vw, wu)
    if q.dot(q) < eps:
        p = pick_largest(u, v, w)
        x = LA.cross(p, np.array([1, 0, 0]))
        if x.dot(x) < eps:
            x = LA.cross(p, np.array([0, 1, 0]))
        y = LA.cross(p, x)
        return x / LA.norm(x), y / LA.norm(y)
    else:
        return q / LA.norm(q), np.zeros(3)


def eigvectors(A, lmd):
    u, v = find_ortho(A - lmd[0] * np.eye(3))
    if v.dot(v) == 0:
        v, _ = find_ortho(A - lmd[1] * np.eye(3))
    w = np.cross(u, v)
    return np.array([u, v, w]).T


def sym_eigsolve_3x3(A):
    scale = LA.norm(A)
    A = A / scale
    lmd = eigvalues(A)
    if lmd is not None:
        eigvecs = eigvectors(A, lmd)
    else:
        eigvecs = np.eye(3)
        val = sum([LA.norm(A[:, i]) for i in range(3)]) / 3
        lmd = np.array([val, val, val])
    return scale * lmd, eigvecs
