// File: inflate.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef INFLATE_HPP
#define INFLATE_HPP

#include "../../data.hpp"

namespace inflate {

// Pressure potential: E = -(P/6) * x0 · (x1 × x2)
// Gradient: dE/dx0 = -(P/6)(x1 × x2), etc.
__device__ Mat3x3f face_gradient(float pressure, const Vec3f &v0,
                                 const Vec3f &v1, const Vec3f &v2) {
    float coeff = -pressure / 6.0f;
    Mat3x3f g;
    g.col(0) = coeff * v1.cross(v2);
    g.col(1) = coeff * v2.cross(v0);
    g.col(2) = coeff * v0.cross(v1);
    return g;
}

// Analytically PSD-projected Hessian of the pressure potential.
//
// Under the SVD X = U*diag(s0,s1,s2)*V^T, the 9x9 Hessian of det(X)
// decomposes into:
//
//   Swap pairs:  eigenvalues ±s0, ±s1, ±s2  (6 eigenvalues)
//     eigvecs for -si: symmetric (e_jk + e_kj)/√2
//   Diagonal subspace: eigenvalues of the 3x3 matrix
//     A = [[0,s2,s1],[s2,0,s0],[s1,s0,0]]
//     which has 1 positive + 2 negative eigenvalues
//
// H = -(P/6)*M flips signs, giving 5 positive eigenvalues to keep:
//   3 from swap pairs  +  2 from the diagonal subspace.
// The projected Hessian is rank 5.
//
__device__ Mat9x9f face_hessian(float pressure, const Vec3f &v0,
                                const Vec3f &v1, const Vec3f &v2) {

    Mat3x3f X;
    X.col(0) = v0;
    X.col(1) = v1;
    X.col(2) = v2;
    Svd3x3 svd = utility::svd3x3_rv(X);

    float s0 = svd.S[0], s1 = svd.S[1], s2 = svd.S[2];
    Mat3x3f V = svd.Vt.transpose();
    Vec3f u0 = svd.U.col(0), u1 = svd.U.col(1), u2 = svd.U.col(2);

    // --- Eigendecomposition of diagonal-subspace matrix A ---
    // A = [[0,s2,s1],[s2,0,s0],[s1,s0,0]]
    // Has 1 positive + 2 negative eigenvalues.
    Mat3x3f A;
    A <<  0, s2, s1,
         s2,  0, s0,
         s1, s0,  0;
    Eigen::SelfAdjointEigenSolver<Mat3x3f> eigsolver;
    eigsolver.computeDirect(A);
    Vec3f eigvals_A = eigsolver.eigenvalues();
    Mat3x3f eigvecs_A = eigsolver.eigenvectors();

    // --- Collect kept eigenvalues and eigenvectors ---
    // We keep modes where the H = -(P/6)*M eigenvalue is positive:
    //   swap pair eigenvalue -si  → H eigenvalue (P/6)*si  (positive)
    //   A eigenvalue < 0          → H eigenvalue (P/6)*|λ| (positive)
    float c = pressure / 6.0f;
    float inv_sqrt2 = 0.7071067811865475f;

    // Up to 5 kept eigenpairs: 3 swap + 2 diagonal
    float weights[5];
    Vec3f Qcols[5][3];  // Qcols[k][vertex] = 3-vector
    int n_kept = 0;

    // Swap pair eigenvectors (symmetric combinations):
    // Q = (u_i * v_j^T + u_j * v_i^T) / √2
    // Column(k) = (V[k,j]*u_i + V[k,i]*u_j) / √2
    int swap_ij[3][2] = {{1, 0}, {2, 0}, {2, 1}};  // (i,j) pairs
    float swap_s[3] = {s2, s1, s0};                 // corresponding σ
    for (int p = 0; p < 3; ++p) {
        float w = c * swap_s[p];
        if (w > 0.0f) {
            int i = swap_ij[p][0], j = swap_ij[p][1];
            Vec3f ui = svd.U.col(i), uj = svd.U.col(j);
            weights[n_kept] = w;
            for (int k = 0; k < 3; ++k) {
                Qcols[n_kept][k] = inv_sqrt2 * (V(k, j) * ui + V(k, i) * uj);
            }
            n_kept++;
        }
    }

    // Diagonal subspace eigenvectors (negative eigenvalues of A)
    for (int d = 0; d < 3; ++d) {
        if (eigvals_A[d] < -1e-12f) {
            float w = c * fabsf(eigvals_A[d]);
            if (w > 0.0f) {
                Vec3f abc = eigvecs_A.col(d);
                weights[n_kept] = w;
                for (int k = 0; k < 3; ++k) {
                    Qcols[n_kept][k] = abc[0] * V(k, 0) * u0 +
                                        abc[1] * V(k, 1) * u1 +
                                        abc[2] * V(k, 2) * u2;
                }
                n_kept++;
            }
        }
    }

    // --- Assemble 9x9 projected Hessian via rank-1 outer products ---
    Mat9x9f H = Mat9x9f::Zero();
    for (int i = 0; i < n_kept; ++i) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                H.block<3, 3>(3 * a, 3 * b) +=
                    weights[i] * Qcols[i][a] * Qcols[i][b].transpose();
            }
        }
    }
    return H;
}

} // namespace inflate

#endif
