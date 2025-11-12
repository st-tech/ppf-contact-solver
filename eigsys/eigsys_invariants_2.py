# File: eigsys_invariants_2.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
import numpy.linalg as LA
from eig_py.eigsolve2x2 import sym_eigsolve_2x2

h = 1e-4
F = np.random.rand(3, 2)
np.set_printoptions(
    precision=2, suppress=False, formatter={"float_kind": "{:0.2e}".format}
)


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


def energy_I(I1, I2):
    return (I1 + I2 - 6) ** 2 / I1 + np.sqrt(I1 / I2 + 1) - 2


def get_invariants(F):
    C = F.T @ F
    I1 = np.trace(C)
    I2 = LA.det(C)
    return I1, I2


def energy_F(F):
    I1, I2 = get_invariants(F)
    return energy_I(I1, I2)


def sym_dIda(a, b):
    gI1 = 2 * np.array([a, b])
    gI2 = 2 * np.array([a * b**2, a**2 * b])
    return np.array([gI1, gI2]).T


def adjugate(matrix):
    return np.array([[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]])


def sym_dIdF(F):
    C = F.T @ F
    gI1 = 2 * F
    gI2 = 2 * F @ adjugate(C).T
    return [gI1, gI2]


def sym_d2Ida2(a, b):
    H1 = 2 * np.eye(2)
    H2 = np.zeros((2, 2))
    H2[:, 0] = [2 * b**2, 4 * a * b]
    H2[:, 1] = [4 * a * b, 2 * a**2]
    return H1, H2


def approx_dEdI(I1, I2):
    return np.array(
        [
            (energy_I(I1 + h, I2) - energy_I(I1 - h, I2)) / (2 * h),
            (energy_I(I1, I2 + h) - energy_I(I1, I2 - h)) / (2 * h),
        ]
    )


def approx_d2EdI2(I1, I2):
    H = np.zeros((2, 2))
    H[:, 0] = (approx_dEdI(I1 + h, I2) - approx_dEdI(I1 - h, I2)) / (2 * h)
    H[:, 1] = (approx_dEdI(I1, I2 + h) - approx_dEdI(I1, I2 - h)) / (2 * h)
    return H


def sym_dEda(a, b, dEdI):
    dIda = sym_dIda(a, b)
    return dIda @ dEdI


def sym_d2Eda2(a, b, dEdI, d2EdI2):
    H = np.zeros((2, 2))
    dIda = sym_dIda(a, b)
    d2Ida2 = sym_d2Ida2(a, b)
    for i in range(2):
        H += dEdI[i] * d2Ida2[i]
    H += dIda @ d2EdI2 @ dIda.T
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

H_F = approx_hess_F(F, dF)
print("---- numerical hessian ----")
print(H_F)

####### Analytical Eigen Decomposition using Invariants #######

U, (a, b), Vt = svd3x2(F)
U = extend(U)
print("singular values:", a, b)

I1, I2 = get_invariants(F)
dEdI = approx_dEdI(I1, I2)
dE2dI2 = approx_d2EdI2(I1, I2)

print("---- analytical gradient ----")
g_F_rebuilt = np.zeros((3, 2))
dIdF = sym_dIdF(F)
for dIk, dEk in zip(dIdF, dEdI):
    g_F_rebuilt += dIk * dEk
print(g_F_rebuilt)

H_s = sym_d2Eda2(a, b, dEdI, dE2dI2)
S_s, U_s = sym_eigsolve_2x2(H_s)
Q = [
    np.array([[0, 1], [-1, 0], [0, 0]]) / np.sqrt(2),
    np.array([[0, 1], [1, 0], [0, 0]]) / np.sqrt(2),
    np.array([[0, 0], [0, 0], [1, 0]]),
    np.array([[0, 0], [0, 0], [0, 1]]),
    np.concatenate((np.diag(U_s[:, 0]), np.zeros((1, 2)))),
    np.concatenate((np.diag(U_s[:, 1]), np.zeros((1, 2)))),
]
lmds = [
    2 * (dEdI[0] + a * b * dEdI[1]),
    2 * (dEdI[0] - a * b * dEdI[1]),
    2 * (dEdI[0] + b * b * dEdI[1]),
    2 * (dEdI[0] + a * a * dEdI[1]),
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
print(f"gradient error: {LA.norm(g_F - g_F_rebuilt) / LA.norm(g_F):.3e}")
print(f"hessian error: {LA.norm(H_F - H_rebuilt) / LA.norm(H_F):.3e}")
