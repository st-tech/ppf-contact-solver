// File: eigenanalysis.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../utility/utility.hpp"
#include "eigenanalysis.hpp"

namespace eigenanalysis {

__device__ Mat3x3f expand_U(const Mat3x2f &U) {
    Vec3f cross = U.col(0).cross(U.col(1));
    Mat3x3f result;
    result << U.col(0), U.col(1), cross;
    return result;
}

__device__ Mat3x2f compute_force(const DiffTable2 &table, const Svd3x2 &svd) {
    // Closed form of the loop it replaces (same identity as the 3x3 overload):
    // dadf(k) for df = e_i e_j^T is U(i,k) V(j,k), so result(i,j) =
    // sum_k deda[k] U(i,k) V(j,k) = U * diag(deda) * V^T.
    return svd.U * table.deda.asDiagonal() * svd.Vt;
}

__device__ Mat6x6f compute_hessian(const DiffTable2 &table, const Svd3x2 &svd,
                                   float eps) {
    Vec2f eigvalues;
    Mat2x2f eigvectors;
    utility::solve_symm_eigen2x2(table.d2ed2a, eigvalues, eigvectors);
    float inv_sqrt2 = 0.7071067811865475f;
    Mat3x3f U = expand_U(svd.U);
    Mat2x2f Vt = svd.Vt;
    Mat3x2f Q[6];
    Q[0] = inv_sqrt2 * Mat3x2f({{0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, 0.0f}});
    // Flip mode: SYMMETRIC off-diagonal, paired with the difference eigenvalue
    // (deda0 - deda1) / (a0 - a1) below, exactly like the 3x3 overload's
    // Q[3..5]. It must NOT be the antisymmetric twist matrix Q[0], which would
    // degenerate the mode basis (vec(U Q0 V^T) == vec(U Q1 V^T)): the twist
    // direction would receive lambda0 + lambda1 and the flip direction zero,
    // i.e. the projected Hessian would be wrong in the flip subspace whenever
    // the singular values differ. Affects ARAP / StVK / SNHk shells only
    // (BaraffWitkin does not use this path).
    Q[1] = inv_sqrt2 * Mat3x2f({{0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}});
    Q[2] = Mat3x2f({{0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}});
    Q[3] = Mat3x2f({{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}});
    Q[4] = Mat3x2f::Zero();
    Q[5] = Mat3x2f::Zero();
    for (int i = 0; i < 2; ++i) {
        Q[4](i, i) = eigvectors(i, 0);
        Q[5](i, i) = eigvectors(i, 1);
    }
    float a0 = svd.S[0];
    float a1 = svd.S[1];
    float denom = a0 - a1;
    float lambda[6];
    lambda[0] = fmaxf(0.0f, (table.deda[0] + table.deda[1]) / (a0 + a1));
    lambda[1] =
        fmaxf(0.0f, fabsf(denom) > eps
                       ? (table.deda[0] - table.deda[1]) / denom
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(1, 1)) -
                             0.5f * (table.d2ed2a(0, 1) + table.d2ed2a(1, 0)));
    lambda[2] = fmaxf(0.0f, table.deda[0] / a0);
    lambda[3] = fmaxf(0.0f, table.deda[1] / a1);
    lambda[4] = fmaxf(0.0f, eigvalues[0]);
    lambda[5] = fmaxf(0.0f, eigvalues[1]);
    Mat6x6f result = Mat6x6f::Zero();
    for (int i = 0; i < 6; ++i) {
        if (lambda[i]) {
            Mat3x2f tmp = U * Q[i] * Vt;
            Vec6f q = Map<Vec6f>(tmp.data());
            result += lambda[i] * q * q.transpose();
        }
    }
    return result;
}

__device__ Mat3x3f compute_force(const DiffTable3 &table, const Svd3x3 &svd) {
    // Closed form of the loop it replaces: dadf(k) for df = e_i e_j^T is
    // U(i,k) V(j,k), so result(i,j) = sum_k deda[k] U(i,k) V(j,k), i.e. exactly
    // U * diag(deda) * V^T (the same device-safe diagonal-product expression
    // plasticity.cu ships). Drops 9 zero-matrix builds and 18 3x3 matmuls.
    return svd.U * table.deda.asDiagonal() * svd.Vt;
}

// Fused tet Hessian: build the 12x12 element Hessian directly from the nine
// eigenmodes, never materializing the 9x9 dF-space Hessian. Algebra: the 9x9 is
// sum_i lambda_i q_i q_i^T with q_i = vec(M_i), M_i = U Q_i V^T; the chain-rule
// conversion contracts each dF index with the shape-gradient vectors g_a, so
//   d2edx2.block(a,b) = sum_i lambda_i (M_i g_a)(M_i g_b)^T = sum_i lambda_i
//   vec(P_i) vec(P_i)^T,  P_i = M_i G  (3x4, G columns = g_a).
// With W = V^T G precomputed, each twist/flip mode's P_i is TWO rank-1 outer
// products (Q_i has two nonzeros) and each scaling mode's is three, so the 9x9
// intermediate (324 B of thread-local traffic) and the 16x9-block conversion
// loop disappear. The lambda clamping (SPD projection) is byte-identical to
// compute_hessian above. This kernel is latency-bound at low occupancy, so the
// shorter dependency chain and smaller local footprint are the point.
__device__ void accumulate_hessian_tet_fused(const DiffTable3 &table,
                                             const Svd3x3 &svd, float eps,
                                             const Mat3x3f &inv_rest3x3,
                                             float mass, Mat12x12f &out) {
    Vec3f eigvalues;
    Mat3x3f eigvectors;
    utility::solve_symm_eigen3x3(table.d2ed2a, eigvalues, eigvectors);
    float a = svd.S[0];
    float b = svd.S[1];
    float c = svd.S[2];
    float denom_ab = a - b;
    float denom_ac = a - c;
    float denom_bc = b - c;
    Vec9f lambda;
    lambda[0] = fmaxf(0.0f, (table.deda[0] + table.deda[1]) / (a + b));
    lambda[1] = fmaxf(0.0f, (table.deda[0] + table.deda[2]) / (a + c));
    lambda[2] = fmaxf(0.0f, (table.deda[1] + table.deda[2]) / (b + c));
    lambda[3] =
        fmaxf(0.0f, fabsf(denom_ab) > eps
                       ? (table.deda[0] - table.deda[1]) / denom_ab
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(1, 1)) -
                             0.5f * (table.d2ed2a(0, 1) + table.d2ed2a(1, 0)));
    lambda[4] =
        fmaxf(0.0f, fabsf(denom_ac) > eps
                       ? (table.deda[0] - table.deda[2]) / denom_ac
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(2, 2)) -
                             0.5f * (table.d2ed2a(0, 2) + table.d2ed2a(2, 0)));
    lambda[5] =
        fmaxf(0.0f, fabsf(denom_bc) > eps
                       ? (table.deda[1] - table.deda[2]) / denom_bc
                       : 0.5f * (table.d2ed2a(1, 1) + table.d2ed2a(2, 2)) -
                             0.5f * (table.d2ed2a(1, 2) + table.d2ed2a(2, 1)));
    lambda[6] = fmaxf(0.0f, eigvalues[0]);
    lambda[7] = fmaxf(0.0f, eigvalues[1]);
    lambda[8] = fmaxf(0.0f, eigvalues[2]);

    // Shape-gradient columns g_a (same construction as utility::convert_*).
    Vec3f g0 = -inv_rest3x3.row(0) - inv_rest3x3.row(1) - inv_rest3x3.row(2);
    Mat3x4f G;
    G.col(0) = g0;
    G.col(1) = inv_rest3x3.row(0);
    G.col(2) = inv_rest3x3.row(1);
    G.col(3) = inv_rest3x3.row(2);
    const Mat3x4f W = svd.Vt * G;

    const float inv_sqrt2 = 0.7071067811865475f;
    // Twist (antisymmetric, s2 = -1) then flip (symmetric, s2 = +1) index pairs,
    // matching Q[0..5] above.
    const unsigned R[6] = {0, 0, 1, 0, 0, 1};
    const unsigned C[6] = {1, 2, 2, 1, 2, 2};
    for (int i = 0; i < 9; ++i) {
        if (lambda[i] == 0.0f) {
            continue;
        }
        Mat3x4f P;
        if (i < 6) {
            const float s2 = (i < 3) ? -1.0f : 1.0f;
            const unsigned r = R[i], cc = C[i];
            // P = inv_sqrt2 * (U.col(r) W.row(cc) + s2 * U.col(cc) W.row(r))
            for (unsigned col = 0; col < 4; ++col) {
                P.col(col) = inv_sqrt2 * (W(cc, col) * svd.U.col(r) +
                                          s2 * W(r, col) * svd.U.col(cc));
            }
        } else {
            const Vec3f v = eigvectors.col(i - 6);
            for (unsigned col = 0; col < 4; ++col) {
                P.col(col) = v[0] * W(0, col) * svd.U.col(0) +
                             v[1] * W(1, col) * svd.U.col(1) +
                             v[2] * W(2, col) * svd.U.col(2);
            }
        }
        // Accumulate (mass * lambda_i) p p^T directly into the caller's element
        // Hessian: no internal 12x12 accumulator, no 12x12 return temp, no
        // mass-scaling expression temp -- the peak device-stack footprint of the
        // tet chain stays at one 12x12 (the caller's), which the non-fused path
        // exceeded only marginally. Scalar loop, symmetric fill.
        const float s = mass * lambda[i];
        const float *pd = P.data();
        for (int r = 0; r < 12; ++r) {
            const float sr = s * pd[r];
            for (int c = 0; c < 12; ++c) {
                out(r, c) += sr * pd[c];
            }
        }
    }
}

__device__ Mat9x9f compute_hessian(const DiffTable3 &table, const Svd3x3 &svd,
                                   float eps) {
    Vec3f eigvalues;
    Mat3x3f eigvectors;
    utility::solve_symm_eigen3x3(table.d2ed2a, eigvalues, eigvectors);
    float inv_sqrt2 = 0.7071067811865475f;
    Mat3x3f Q[9];
    Q[0] = inv_sqrt2 * Mat3x3f({{0., 1., 0.}, {-1., 0., 0.}, {0., 0., 0.}});
    Q[1] = inv_sqrt2 * Mat3x3f({{0., 0., 1.}, {0., 0., 0.}, {-1., 0., 0.}});
    Q[2] = inv_sqrt2 * Mat3x3f({{0., 0., 0.}, {0., 0., 1.}, {0., -1., 0.}});
    Q[3] = inv_sqrt2 * Mat3x3f({{0., 1., 0.}, {1., 0., 0.}, {0., 0., 0.}});
    Q[4] = inv_sqrt2 * Mat3x3f({{0., 0., 1.}, {0., 0., 0.}, {1., 0., 0.}});
    Q[5] = inv_sqrt2 * Mat3x3f({{0., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}});
    Q[6] = Mat3x3f::Zero();
    Q[7] = Mat3x3f::Zero();
    Q[8] = Mat3x3f::Zero();
    for (int i = 0; i < 3; ++i) {
        Q[6](i, i) = eigvectors(i, 0);
        Q[7](i, i) = eigvectors(i, 1);
        Q[8](i, i) = eigvectors(i, 2);
    }
    float a = svd.S[0];
    float b = svd.S[1];
    float c = svd.S[2];
    float denom_ab = a - b;
    float denom_ac = a - c;
    float denom_bc = b - c;
    Vec9f lambda;
    lambda[0] = fmaxf(0.0f, (table.deda[0] + table.deda[1]) / (a + b));
    lambda[1] = fmaxf(0.0f, (table.deda[0] + table.deda[2]) / (a + c));
    lambda[2] = fmaxf(0.0f, (table.deda[1] + table.deda[2]) / (b + c));
    lambda[3] =
        fmaxf(0.0f, fabsf(denom_ab) > eps
                       ? (table.deda[0] - table.deda[1]) / denom_ab
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(1, 1)) -
                             0.5f * (table.d2ed2a(0, 1) + table.d2ed2a(1, 0)));
    lambda[4] =
        fmaxf(0.0f, fabsf(denom_ac) > eps
                       ? (table.deda[0] - table.deda[2]) / denom_ac
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(2, 2)) -
                             0.5f * (table.d2ed2a(0, 2) + table.d2ed2a(2, 0)));
    lambda[5] =
        fmaxf(0.0f, fabsf(denom_bc) > eps
                       ? (table.deda[1] - table.deda[2]) / denom_bc
                       : 0.5f * (table.d2ed2a(1, 1) + table.d2ed2a(2, 2)) -
                             0.5f * (table.d2ed2a(1, 2) + table.d2ed2a(2, 1)));
    lambda[6] = fmaxf(0.0f, eigvalues[0]);
    lambda[7] = fmaxf(0.0f, eigvalues[1]);
    lambda[8] = fmaxf(0.0f, eigvalues[2]);
    Mat9x9f result = Mat9x9f::Zero();
    for (int i = 0; i < 9; ++i) {
        if (lambda[i]) {
            Mat3x3f tmp = svd.U * Q[i] * svd.Vt;
            Vec9f q = Map<Vec9f>(tmp.data());
            result += lambda[i] * q * q.transpose();
        }
    }
    return result;
}

} // namespace eigenanalysis
