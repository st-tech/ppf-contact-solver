# File: eigsys_2.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
import numpy.linalg as LA
from eig_py.eigsolve2x2 import sym_eigsolve_2x2

h = 1e-4
F = np.random.rand(3, 2)

# ==== uncommnent below to try a case two singular values are almost the same =====
# a, b = 0.75, 0.75
# U, _, Vt = LA.svd(F)
# F = U[:, 0:2] @ np.diag([a, b]) @ Vt


def svd3x2(F):
    A = F.T @ F
    lmd, eigvecs = sym_eigsolve_2x2(A)
    sigma = np.sqrt(lmd)
    U = F @ eigvecs
    for i in range(2):
        U[:, i] /= np.linalg.norm(U[:, i])
    return U, sigma, eigvecs.T


def extend(U):
    uv = np.cross(U[:, 0], U[:, 1])
    return np.array([U[:, 0], U[:, 1], uv]).T


def energy_s(a, b):
    return np.log(a**2) + np.log(b**2) + a * b + a**2 * b**2


def energy_F(F):
    _, sigma, _ = svd3x2(F)
    return energy_s(*sigma)


def approx_grad_s(a, b):
    return np.array(
        [
            (energy_s(a + h, b) - energy_s(a - h, b)) / (2 * h),
            (energy_s(a, b + h) - energy_s(a, b - h)) / (2 * h),
        ]
    )


def approx_hess_s(a, b):
    H = np.zeros((2, 2))
    H[:, 0] = (approx_grad_s(a + h, b) - approx_grad_s(a - h, b)) / (2 * h)
    H[:, 1] = (approx_grad_s(a, b + h) - approx_grad_s(a, b - h)) / (2 * h)
    return H


def approx_grad_F(F, dF):
    return (energy_F(F + h * dF) - energy_F(F - h * dF)) / (2 * h)


def approx_hess_F(F, dF):
    H = np.zeros((len(dF), len(dF)))
    for i, dFi in enumerate(dF):
        for j, dFj in enumerate(dF):
            H[i][j] = (
                approx_grad_F(F + h * dFi, dFj) - approx_grad_F(F - h * dFi, dFj)
            ) / (2 * h)
    return H


def gen_dF(i, j):
    dF = np.zeros((3, 2))
    dF[i][j] = 1
    return dF


def mat2vec(A):
    x = []
    for j in range(2):
        for i in range(3):
            x.append(A[i][j])
    return np.array(x)


dF = []
for j in range(2):
    for i in range(3):
        dF.append(gen_dF(i, j))

print("---- numerical gradient ----")
g_F = np.zeros(6)
for i, dFi in enumerate(dF):
    g_F[i] = approx_grad_F(F, dFi)
g_F = np.reshape(g_F, (2, 3)).T
print(g_F)

print("---- numerical hessian ----")
H_F = approx_hess_F(F, dF)
print(H_F)

####### Analytical Eigen Decomposition #######

U, (a, b), Vt = svd3x2(F)
U = extend(U)
print("singular values:", a, b)

H_s, g_s = approx_hess_s(a, b), approx_grad_s(a, b)
S_s, U_s = sym_eigsolve_2x2(H_s)

print("---- analytical gradient ----")
g_F_rebuilt = np.zeros(6)
for i, dFi in enumerate(dF):
    g_F_rebuilt[i] = sum(
        g_s[k] * np.dot(U[:, k], np.dot(dFi, Vt[k, :])) for k in range(2)
    )
g_F_rebuilt = np.reshape(g_F_rebuilt, (2, 3)).T
print(g_F_rebuilt)

Q = [
    np.array([[0, 1], [-1, 0], [0, 0]]) / np.sqrt(2),
    np.array([[0, 1], [1, 0], [0, 0]]) / np.sqrt(2),
    np.array([[0, 0], [0, 0], [1, 0]]),
    np.array([[0, 0], [0, 0], [0, 1]]),
    np.concatenate((np.diag(U_s[:, 0]), np.zeros((1, 2)))),
    np.concatenate((np.diag(U_s[:, 1]), np.zeros((1, 2)))),
]
lmds = [
    (g_s[0] + g_s[1]) / (a + b),
    (g_s[0] - g_s[1]) / (a - b) if abs(a - b) > 1e-2 else H_s[0][0] - H_s[0][1],
    g_s[0] / a,
    g_s[1] / b,
    S_s[0],
    S_s[1],
]

print("--- analytical hessian ---")
Qmat = np.zeros((6, 6))
for i, m in enumerate(Q):
    Qmat[:, i] = mat2vec(U @ m @ Vt)
H_rebuilt = Qmat @ np.diag(lmds) @ Qmat.T
print(H_rebuilt)

###############################################

print("--- error ---")
print("gradient:", LA.norm(g_F - g_F_rebuilt) / LA.norm(g_F))
print("hessian:", LA.norm(H_F - H_rebuilt) / LA.norm(H_F))
