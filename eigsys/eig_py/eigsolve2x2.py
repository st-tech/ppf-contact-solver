# File: eigsolve2x2.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
import numpy.linalg as LA


# fmt: off
def eigvalues(A):
    a00, a01, a11 = A[0,0], A[0,1], A[1,1]
    d = float(np.sqrt((a00-a11)**2+4*a01**2)/2)
    mid = float(a00+a11)/2
    return np.array([mid-d, mid+d])
# fmt: on


def rot90(x):
    return np.array([x[1], -x[0]])


def find_ortho(A, x):
    eps = 1e-8
    u, v = rot90(A[:, 0]), rot90(A[:, 1])
    if np.dot(u, u) > eps**2:
        return u / LA.norm(u)
    elif np.dot(v, v) > eps**2:
        return v / LA.norm(v)
    else:
        return rot90(x)


def eigvectors(A, lmd):
    u = find_ortho(A - lmd[0] * np.eye(2), np.array([0, 1]))
    v = find_ortho(A - lmd[1] * np.eye(2), -u)
    return np.array([u, v]).T


def sym_eigsolve_2x2(A):
    scale = LA.norm(A)
    A = A / scale
    lmd = eigvalues(A)
    eigvecs = eigvectors(A, lmd)
    return scale * lmd, eigvecs
