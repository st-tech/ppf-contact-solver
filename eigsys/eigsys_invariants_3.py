# File: eigsys_invariants_3.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import numpy as np
import numpy.linalg as LA
from eig_py.eigsolve3x3 import sym_eigsolve_3x3

h = 1e-4
F = np.random.rand(3, 3)
np.set_printoptions(
    precision=2, suppress=False, formatter={"float_kind": "{:0.2e}".format}
)


def svd3x3(F):
    A = F.T @ F
    lmd, eigvecs = sym_eigsolve_3x3(A)
    sigma = np.sqrt(lmd)
    U = F @ eigvecs
    for i in range(3):
        U[:, i] /= np.linalg.norm(U[:, i])
    return U, sigma, eigvecs.T


def energy_I(I1, I2, I3):
    return (I1 + I2 - 6) ** 2 / I1 + np.sqrt(I3 / I1 + 1) - 2


def get_invariants(F):
    C = F.T @ F
    I1 = np.trace(C)
    I2 = 0.5 * (I1**2 - np.trace(C @ C))
    I3 = LA.det(C)
    return I1, I2, I3


def energy_F(F):
    I1, I2, I3 = get_invariants(F)
    return energy_I(I1, I2, I3)


def sym_dIda(a, b, c):
    gI1 = 2 * np.array([a, b, c])
    gI2 = 2 * np.array([a * b**2 + a * c**2, b * c**2 + b * a**2, c * a**2 + c * b**2])
    gI3 = 2 * np.array([a * b**2 * c**2, b * c**2 * a**2, c * a**2 * b**2])
    return np.array([gI1, gI2, gI3]).T


def adjugate(matrix):
    result = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            result[i, j] = ((-1) ** (i + j)) * LA.det(minor)
    return result.T


def sym_dIdF(F):
    C = F.T @ F
    gI1 = 2 * F
    gI2 = 2 * (F * np.trace(C) - F @ C)
    gI3 = 2 * F @ adjugate(C).T
    return [gI1, gI2, gI3]


def sym_d2Ida2(a, b, c):
    H1 = 2 * np.eye(3)
    H2 = np.zeros((3, 3))
    H2[:, 0] = 2 * np.array([b**2 + c**2, 2 * a * b, 2 * a * c])
    H2[:, 1] = 2 * np.array([2 * a * b, c**2 + a**2, 2 * b * c])
    H2[:, 2] = 2 * np.array([2 * a * c, 2 * b * c, a**2 + b**2])
    H3 = np.zeros((3, 3))
    H3[:, 0] = 2 * np.array([b**2 * c**2, 2 * a * b * c**2, 2 * a * b**2 * c])
    H3[:, 1] = 2 * np.array([2 * a * b * c**2, c**2 * a**2, 2 * a**2 * b * c])
    H3[:, 2] = 2 * np.array([2 * a * b**2 * c, 2 * a**2 * b * c, a**2 * b**2])
    return H1, H2, H3


def approx_dEdI(I1, I2, I3):
    return np.array(
        [
            (energy_I(I1 + h, I2, I3) - energy_I(I1 - h, I2, I3)) / (2 * h),
            (energy_I(I1, I2 + h, I3) - energy_I(I1, I2 - h, I3)) / (2 * h),
            (energy_I(I1, I2, I3 + h) - energy_I(I1, I2, I3 - h)) / (2 * h),
        ]
    )


def approx_d2EdI2(I1, I2, I3):
    H = np.zeros((3, 3))
    H[:, 0] = (approx_dEdI(I1 + h, I2, I3) - approx_dEdI(I1 - h, I2, I3)) / (2 * h)
    H[:, 1] = (approx_dEdI(I1, I2 + h, I3) - approx_dEdI(I1, I2 - h, I3)) / (2 * h)
    H[:, 2] = (approx_dEdI(I1, I2, I3 + h) - approx_dEdI(I1, I2, I3 - h)) / (2 * h)
    return H


def sym_dEda(a, b, c, dEdI):
    dIda = sym_dIda(a, b, c)
    return dIda @ dEdI


def sym_d2Eda2(a, b, c, dEdI, d2EdI2):
    H = np.zeros((3, 3))
    dIda = sym_dIda(a, b, c)
    d2Ida2 = sym_d2Ida2(a, b, c)
    for i in range(3):
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

print("---- numerical gradient ----")
g_F = np.zeros(9)
for i, dFi in enumerate(dF):
    g_F[i] = approx_grad_F(F, dFi)
g_F = np.reshape(g_F, (3, 3)).T
print(g_F)

U, (a, b, c), Vt = svd3x3(F)
H_F = approx_hess_F(F, dF)
print("--- numerical hessian ---")
print(H_F)

####### Analytical Eigen Decomposition using Invariants #######

I1, I2, I3 = get_invariants(F)
dEdI = approx_dEdI(I1, I2, I3)
dE2dI2 = approx_d2EdI2(I1, I2, I3)

print("---- analytical gradient ----")
g_F_rebuilt = np.zeros((3, 3))
dIdF = sym_dIdF(F)
for dIk, dEk in zip(dIdF, dEdI):
    g_F_rebuilt += dIk * dEk
print(g_F_rebuilt)

H_s = sym_d2Eda2(a, b, c, dEdI, dE2dI2)
S_s, U_s = sym_eigsolve_3x3(H_s)
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
abc = a * b * c
lmds = [
    2 * (dEdI[0] + (a * b + c**2) * dEdI[1] + (c * abc) * dEdI[2]),
    2 * (dEdI[0] + (a * c + b**2) * dEdI[1] + (b * abc) * dEdI[2]),
    2 * (dEdI[0] + (b * c + a**2) * dEdI[1] + (a * abc) * dEdI[2]),
    2 * (dEdI[0] + (c**2 - a * b) * dEdI[1] - (c * abc) * dEdI[2]),
    2 * (dEdI[0] + (b**2 - a * c) * dEdI[1] - (b * abc) * dEdI[2]),
    2 * (dEdI[0] + (a**2 - b * c) * dEdI[1] - (a * abc) * dEdI[2]),
    S_s[0],
    S_s[1],
    S_s[2],
]

Qmat = np.zeros((9, 9))
for i, w in enumerate(Qs):
    Qmat[:, i] = mat2vec(U @ w @ Vt)
H_rebuilt = Qmat @ np.diag(lmds) @ Qmat.T
print("--- analytical hessian ---")
print(H_rebuilt)

###############################################

print(f"gradient error: {LA.norm(g_F - g_F_rebuilt) / LA.norm(g_F):.3e}")
print(f"hessian error: {LA.norm(H_F - H_rebuilt) / LA.norm(H_F):.3e}")
