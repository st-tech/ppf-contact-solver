# File: eigsys_3.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
import numpy.linalg as LA
from eig_py.eigsolve3x3 import sym_eigsolve_3x3

h = 1e-4
F = np.random.rand(3, 3)
verbose = False
models = [
    "ARAP",
    "SymDirichlet",
    "MIPS",
    "Ogden",
    "Yeoh",
]

# ==== uncommnent below to try a case two singular values are almost the same =====
# a, b, c = 0.73000001, 0.73, 0.780002
# U, _, Vt = LA.svd(F)
# F = U @ np.diag([a, b, c]) @ Vt


def svd3x3(F):
    A = F.T @ F
    lmd, eigvecs = sym_eigsolve_3x3(A)
    sigma = np.sqrt(lmd)
    U = F @ eigvecs
    for i in range(3):
        U[:, i] /= np.linalg.norm(U[:, i])
    return U, sigma, eigvecs.T


def energy_s(a, b, c, m):
    if m == "ARAP":
        return sum([(s - 1) ** 2 for s in [a, b, c]])
    elif m == "SymDirichlet":
        return sum([s**2 + 1 / s**2 for s in [a, b, c]])
    elif m == "MIPS":
        return sum([s**2 for s in [a, b, c]]) / (a * b * c)
    elif m == "Ogden":
        return sum(
            [a ** (0.5**k) + b ** (0.5**k) + c ** (0.5**k) - 3 for k in range(5)]
        )
    elif m == "Yeoh":
        return sum([(a**2 + b**2 + c**2 - 3) ** (k + 1) for k in range(3)])
    else:
        return 0.0


def energy_F(F, m):
    _, (a, b, c), _ = svd3x3(F)
    return energy_s(a, b, c, m)


def approx_grad_s(a, b, c, m):
    return np.array(
        [
            (energy_s(a + h, b, c, m) - energy_s(a - h, b, c, m)) / (2 * h),
            (energy_s(a, b + h, c, m) - energy_s(a, b - h, c, m)) / (2 * h),
            (energy_s(a, b, c + h, m) - energy_s(a, b, c - h, m)) / (2 * h),
        ]
    )


def approx_hess_s(a, b, c, m):
    H = np.zeros((3, 3))
    H[:, 0] = (approx_grad_s(a + h, b, c, m) - approx_grad_s(a - h, b, c, m)) / (2 * h)
    H[:, 1] = (approx_grad_s(a, b + h, c, m) - approx_grad_s(a, b - h, c, m)) / (2 * h)
    H[:, 2] = (approx_grad_s(a, b, c + h, m) - approx_grad_s(a, b, c - h, m)) / (2 * h)
    return H


def approx_grad_F(F, dF, m):
    return (energy_F(F + h * dF, m) - energy_F(F - h * dF, m)) / (2 * h)


def approx_hess_F(F, dF, m):
    H = np.zeros((len(dF), len(dF)))
    for i, dFi in enumerate(dF):
        for j, dFj in enumerate(dF):
            H[i][j] = (
                approx_grad_F(F + h * dFi, dFj, m) - approx_grad_F(F - h * dFi, dFj, m)
            ) / (2 * h)
    return H


def gen_dF(i, j):
    dF = np.zeros((3, 3))
    dF[i][j] = 1
    return dF


def mat2vec(A):
    x = []
    for j in range(3):
        for i in range(3):
            x.append(A[i][j])
    return np.array(x)


dF = []
for j in range(3):
    for i in range(3):
        dF.append(gen_dF(i, j))

U, (a, b, c), Vt = svd3x3(F)
errors = {}
for m in models:
    g_F = np.zeros(9)
    for i, dFi in enumerate(dF):
        g_F[i] = approx_grad_F(F, dFi, m)
    g_F = np.reshape(g_F, (3, 3)).T
    if verbose:
        print(f"---- ({m}) numerical gradient ----")
        print(g_F)

    H_F = approx_hess_F(F, dF, m)
    if verbose:
        print(f"---- ({m}) numerical hessian ----")
        print(H_F)

    ####### Analytical Eigen Decomposition #######
    H_s, g_s = approx_hess_s(a, b, c, m), approx_grad_s(a, b, c, m)
    S_s, U_s = sym_eigsolve_3x3(H_s)

    g_F_rebuilt = np.zeros(9)
    for i, dFi in enumerate(dF):
        g_F_rebuilt[i] = sum(
            g_s[k] * np.dot(U[:, k], np.dot(dFi, Vt[k, :])) for k in range(3)
        )
    g_F_rebuilt = np.reshape(g_F_rebuilt, (3, 3)).T
    if verbose:
        print(f"---- ({m}) analytical gradient ----")
        print(g_F_rebuilt)

    Qs = [
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]) / np.sqrt(2),
        np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]) / np.sqrt(2),
        np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]) / np.sqrt(2),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / np.sqrt(2),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / np.sqrt(2),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / np.sqrt(2),
        np.diag(U_s[:, 0]),
        np.diag(U_s[:, 1]),
        np.diag(U_s[:, 2]),
    ]
    lmds = [
        (g_s[0] + g_s[1]) / (a + b),
        (g_s[0] + g_s[2]) / (a + c),
        (g_s[1] + g_s[2]) / (b + c),
        (g_s[0] - g_s[1]) / (a - b) if abs(a - b) > 1e-2 else H_s[0][0] - H_s[0][1],
        (g_s[0] - g_s[2]) / (a - c) if abs(a - c) > 1e-2 else H_s[0][0] - H_s[0][2],
        (g_s[1] - g_s[2]) / (b - c) if abs(b - c) > 1e-2 else H_s[1][1] - H_s[1][2],
        S_s[0],
        S_s[1],
        S_s[2],
    ]

    Qmat = np.zeros((9, 9))
    for i, w in enumerate(Qs):
        Qmat[:, i] = mat2vec(U @ w @ Vt)
    H_rebuilt = Qmat @ np.diag(lmds) @ Qmat.T
    if verbose:
        print(f"--- ({m}) analytical hessian ---")
        print(H_rebuilt)

    ###############################################

    errors[m] = [
        LA.norm(g_F - g_F_rebuilt) / LA.norm(g_F),
        LA.norm(H_F - H_rebuilt) / LA.norm(H_F),
    ]

print("===== error summary =====")
for name, err in errors.items():
    print(f"{name}: grad: {err[0]:.3e}, Hess: {err[1]:.3e}")
