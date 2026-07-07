// File: utility.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../kernels/reduce.hpp"
#include "../linalg/eigsolve.hpp"
#include "dispatcher.hpp"
#include "utility.hpp"
#include <limits>

namespace utility {

__device__ Vec3f compute_vertex_normal(const DataSet &data,
                                       const Vec<Vec3f> &vertex, unsigned i) {
    Vec3f normal = Vec3f::Zero();
    if (data.mesh.neighbor.vertex.face.size) {
        for (unsigned j = 0; j < data.mesh.neighbor.vertex.face.count(i); ++j) {
            const Vec3u &face =
                data.mesh.mesh.face[data.mesh.neighbor.vertex.face(i, j)];
            const Vec3f &z0 = vertex[face[0]];
            const Vec3f &z1 = vertex[face[1]];
            const Vec3f &z2 = vertex[face[2]];
            normal += (z1 - z0).cross(z2 - z0);
        }
        if (normal.squaredNorm()) {
            normal.normalize();
        }
    }
    return normal;
}

__device__ void solve_symm_eigen2x2(const Mat2x2f &matrix, Vec2f &eigenvalues,
                                    Mat2x2f &eigenvectors) {
    linalg::eig::symm2x2(matrix, eigenvalues, eigenvectors);
}

__device__ void solve_symm_eigen3x3(const Mat3x3f &matrix, Vec3f &eigenvalues,
                                    Mat3x3f &eigenvectors) {
    linalg::eig::symm3x3(matrix, eigenvalues, eigenvectors);
}

// One-sided Jacobi SVD of a 3xN matrix (N in {2,3}) in float32, computed in a
// FIXED, unrolled op count: no convergence branch, so the time is deterministic
// and there is no warp divergence. Orthogonalizing F's columns directly, rather
// than eigensolving the normal equations F^T F, keeps full float relative
// accuracy in the small singular values: forming F^T F squares the condition
// number and caps accuracy at ~sqrt(eps) ~ 3e-4 (worst-case singular value ~1%,
// 3x3 reconstruction ~1e-3), whereas this reaches ~1e-6. One rotation
// orthogonalizes two columns exactly, so N=2 needs a single sweep; N=3 needs
// five (three already give float-precise singular values and reconstruction,
// but a near-degenerate triple needs five to bring U and V to float-orthonormal
// ~1e-6, vs ~3e-4 at three). Output: F = U diag(sigma) V^T with sigma >= 0,
// sorted ASCENDING (the prior symm-eigen convention: deda pairing, the
// svd3x3_rv min-index, the strain-limit maxCoeff). U and V have orthonormal
// columns; column signs are arbitrary but consistent between U and V (every
// consumer is sign-invariant: v v^T Hessian modes, U diag V^T force, U S V^T
// reconstruction).
template <int N, int SWEEPS>
__device__ void jacobi_svd(const SMat<float, 3, N> &F, SMat<float, 3, N> &U,
                           SVec<float, N> &sigma, SMat<float, N, N> &V) {
    float A[3][N], Vv[N][N];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < N; ++c) A[r][c] = F(r, c);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) Vv[i][j] = (i == j) ? 1.0f : 0.0f;
#pragma unroll
    for (int sweep = 0; sweep < SWEEPS; ++sweep) {
#pragma unroll
        for (int p = 0; p < N; ++p) {
#pragma unroll
            for (int q = p + 1; q < N; ++q) {
                float a = 0.0f, b = 0.0f, g = 0.0f;
                for (int r = 0; r < 3; ++r) {
                    a += A[r][p] * A[r][p];
                    b += A[r][q] * A[r][q];
                    g += A[r][p] * A[r][q];
                }
                // Rotation that zeros the (p,q) column inner product. g == 0
                // gives t = 0 (identity), keeping the op count fixed and the
                // path branch-free across the warp.
                float zeta = (b - a) / (2.0f * g);
                float t = (g == 0.0f)
                              ? 0.0f
                              : (zeta >= 0.0f ? 1.0f : -1.0f) /
                                    (fabsf(zeta) + sqrtf(1.0f + zeta * zeta));
                float c = 1.0f / sqrtf(1.0f + t * t), s = c * t;
                for (int r = 0; r < 3; ++r) {
                    float ap = A[r][p], aq = A[r][q];
                    A[r][p] = c * ap - s * aq;
                    A[r][q] = s * ap + c * aq;
                }
                for (int r = 0; r < N; ++r) {
                    float vp = Vv[r][p], vq = Vv[r][q];
                    Vv[r][p] = c * vp - s * vq;
                    Vv[r][q] = s * vp + c * vq;
                }
            }
        }
    }
    // Singular values are the column norms of the orthogonalized F (accurate to
    // float precision, unlike sqrt of an F^T F eigenvalue); U is the normalized
    // columns.
    float s_[N], u_[3][N];
    for (int c = 0; c < N; ++c) {
        float n2 = 0.0f;
        for (int r = 0; r < 3; ++r) n2 += A[r][c] * A[r][c];
        float n = sqrtf(n2);
        s_[c] = n;
        float inv = (n > 0.0f) ? 1.0f / n : 0.0f;
        for (int r = 0; r < 3; ++r) u_[r][c] = A[r][c] * inv;
    }
    int idx[N];
    for (int i = 0; i < N; ++i) idx[i] = i;
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int j = i + 1; j < N; ++j)
            if (s_[idx[j]] < s_[idx[i]]) {
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
    for (int c = 0; c < N; ++c) {
        sigma[c] = s_[idx[c]];
        for (int r = 0; r < 3; ++r) U(r, c) = u_[r][idx[c]];
        for (int r = 0; r < N; ++r) V(r, c) = Vv[r][idx[c]];
    }
}

__device__ Vec2f singular_vals_minus_one(const Mat3x2f &F) {
    Mat3x2f U;
    Vec2f sigma;
    Mat2x2f V;
    jacobi_svd<2, 1>(F, U, sigma, V);
    sigma[0] -= 1.0f;
    sigma[1] -= 1.0f;
    return sigma;
}

__device__ Svd3x2 svd3x2_shifted(const Mat3x2f &F) {
    Mat3x2f U;
    Vec2f sigma;
    Mat2x2f V;
    jacobi_svd<2, 1>(F, U, sigma, V);
    sigma[0] -= 1.0f;
    sigma[1] -= 1.0f;
    return {U, sigma, V.transpose()};
}

__device__ Svd3x2 svd3x2(const Mat3x2f &F) {
    Mat3x2f U;
    Vec2f sigma;
    Mat2x2f V;
    jacobi_svd<2, 1>(F, U, sigma, V);
    return {U, sigma, V.transpose()};
}

__device__ Svd3x3 svd3x3(const Mat3x3f &F) {
    Mat3x3f U, V;
    Vec3f sigma;
    jacobi_svd<3, 5>(F, U, sigma, V);
    return {U, sigma, V.transpose()};
}

__device__ Svd3x3 svd3x3_rv(const Mat3x3f &F) {
    Svd3x3 svd = svd3x3(F);
    float det_u = svd.U.determinant();
    float det_vt = svd.Vt.determinant();
    Mat3x3f L = Mat3x3f::Identity();
    unsigned min_index;
    svd.S.minCoeff(&min_index);
    L(min_index, min_index) = -1.0f;
    if (det_u < 0.0f && det_vt > 0.0f) {
        svd.U = svd.U * L;
        svd.S[min_index] *= -1.0f;
    } else if (det_u > 0.0f && det_vt < 0.0f) {
        svd.Vt = L * svd.Vt;
        svd.S[min_index] *= -1.0f;
    }
    return svd;
}

__device__ Mat3x3f convert_force(const Mat3x2f &dedF,
                                 const Mat2x2f &inv_rest2x2) {
    Vec2f g0 = -inv_rest2x2.row(0) - inv_rest2x2.row(1);
    Vec2f g1 = inv_rest2x2.row(0);
    Vec2f g2 = inv_rest2x2.row(1);

    Mat3x3f result;
    for (unsigned dim = 0; dim < 3; ++dim) {
        result(dim, 0) = g0.dot(dedF.row(dim));
        result(dim, 1) = g1.dot(dedF.row(dim));
        result(dim, 2) = g2.dot(dedF.row(dim));
    }
    return result;
}

__device__ Mat3x4f convert_force(const Mat3x3f &dedF,
                                 const Mat3x3f &inv_rest3x3) {
    Vec3f g0 = -inv_rest3x3.row(0) - inv_rest3x3.row(1) - inv_rest3x3.row(2);
    Vec3f g1 = inv_rest3x3.row(0);
    Vec3f g2 = inv_rest3x3.row(1);
    Vec3f g3 = inv_rest3x3.row(2);

    Mat3x4f result;
    for (unsigned dim = 0; dim < 3; ++dim) {
        result(dim, 0) = g0.dot(dedF.row(dim));
        result(dim, 1) = g1.dot(dedF.row(dim));
        result(dim, 2) = g2.dot(dedF.row(dim));
        result(dim, 3) = g3.dot(dedF.row(dim));
    }
    return result;
}

__device__ Mat9x9f convert_hessian(const Mat6x6f &d2ed2f,
                                   const Mat2x2f &inv_rest2x2) {
    Vec2f g[3];
    g[0] = -inv_rest2x2.row(0) - inv_rest2x2.row(1);
    g[1] = inv_rest2x2.row(0);
    g[2] = inv_rest2x2.row(1);

    // The deformation map dfdx = G (x) I_3 (with G[d][a] = g_a[d]) is sparse, so
    // the dense result = dfdx^T d2ed2f dfdx collapses to per-vertex 3x3 blocks:
    //   result.block(a,b) = sum_{d,e} g_a[d] g_b[e] * d2ed2f.block(3d, 3e).
    // This skips materializing the 6x9 dfdx and the 36 9-vector outer products
    // (fewer MACs, far less local-memory pressure). Every (a,b) block is filled
    // (no upper-triangular shortcut) so add_stiffness_damping's dense K*u and
    // the CSR scatter see the full matrix. Verified bit-close to the dense form.
    Mat9x9f result;
    for (unsigned a = 0; a < 3; ++a) {
        for (unsigned b = 0; b < 3; ++b) {
            Mat3x3f blk = Mat3x3f::Zero();
            for (unsigned d = 0; d < 2; ++d) {
                for (unsigned e = 0; e < 2; ++e) {
                    blk += (g[a][d] * g[b][e]) *
                           d2ed2f.block<3, 3>(3 * d, 3 * e);
                }
            }
            result.block<3, 3>(3 * a, 3 * b) = blk;
        }
    }
    return result;
}

__device__ Mat12x12f convert_hessian(const Mat9x9f &d2ed2f,
                                     const Mat3x3f &inv_rest3x3) {
    Vec3f g0 = -inv_rest3x3.row(0) - inv_rest3x3.row(1) - inv_rest3x3.row(2);
    Vec3f g1 = inv_rest3x3.row(0);
    Vec3f g2 = inv_rest3x3.row(1);
    Vec3f g3 = inv_rest3x3.row(2);

    Vec3f g[4] = {g0, g1, g2, g3};

    // Same (G (x) I_3) factorization as the 6x6 overload, here 4 vertices x 3
    // deformation-gradient columns. result.block(a,b) = sum_{d,e} g_a[d] g_b[e]
    // * d2ed2f.block(3d, 3e), avoiding the 9x12 dfdx and the 81 12-vector outer
    // products. All 16 blocks filled so downstream dense uses see a full matrix.
    Mat12x12f result;
    for (unsigned a = 0; a < 4; ++a) {
        for (unsigned b = 0; b < 4; ++b) {
            Mat3x3f blk = Mat3x3f::Zero();
            for (unsigned d = 0; d < 3; ++d) {
                for (unsigned e = 0; e < 3; ++e) {
                    blk += (g[a][d] * g[b][e]) *
                           d2ed2f.block<3, 3>(3 * d, 3 * e);
                }
            }
            result.block<3, 3>(3 * a, 3 * b) = blk;
        }
    }
    return result;
}

__device__ Mat3x2f compute_deformation_grad(const Mat3x3f &x,
                                            const Mat2x2f &inv_rest2x2) {
    Mat3x2f dx;
    dx.col(0) = x.col(1) - x.col(0);
    dx.col(1) = x.col(2) - x.col(0);
    return dx * inv_rest2x2;
}

__device__ Mat3x3f compute_deformation_grad(const Mat3x4f &x,
                                            const Mat3x3f &inv_rest3x3) {
    Mat3x3f dx;
    dx.col(0) = x.col(1) - x.col(0);
    dx.col(1) = x.col(2) - x.col(0);
    dx.col(2) = x.col(3) - x.col(0);
    return dx * inv_rest3x3;
}

__device__ float compute_face_area(const Mat3x3f &vertex) {
    const Vec3f v0 = vertex.col(0);
    const Vec3f v1 = vertex.col(1);
    const Vec3f v2 = vertex.col(2);
    return 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

void compute_svd(DataSet data, Vec<Vec3f> curr, Vec<Svd3x2> svd,
                 ParamSet param) {
    unsigned shell_face_count = data.shell_face_count;
    auto mesh_face = data.mesh.mesh.face.data;
    auto curr_data = curr.data;
    auto svd_data = svd.data;
    auto inv_rest2x2 = data.inv_rest2x2.data;
    DISPATCH_START(shell_face_count)
    [mesh_face, curr_data, svd_data, inv_rest2x2] __device__(unsigned i) mutable {
        Vec3u face = mesh_face[i];
        Mat3x3f x;
        x << curr_data[face[0]], curr_data[face[1]], curr_data[face[2]];
        const Mat3x2f F =
            utility::compute_deformation_grad(x, inv_rest2x2[i]);
        svd_data[i] = utility::svd3x2(F);
    } DISPATCH_END;
}

__device__ float get_wind_weight(float time) {
    float angle = 30.0f * time;
    float t = 0.25f;
    return t * (0.5f * (1.0f + sinf(angle))) + (1.0f - t);
}

} // namespace utility
