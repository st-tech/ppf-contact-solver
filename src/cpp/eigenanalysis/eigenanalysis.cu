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
    Mat3x2f result;
    Mat2x3f Ut = svd.U.transpose();
    Mat2x2f V = svd.Vt.transpose();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            Mat3x2f df = Mat3x2f::Zero();
            df(i, j) = 1.0f;
            Vec2f dadf = (Ut * df * V).diagonal();
            result(i, j) = table.deda.dot(dadf);
        }
    }
    return result;
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
    Q[1] = inv_sqrt2 * Mat3x2f({{0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, 0.0f}});
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
    lambda[0] = fmax(0.0f, (table.deda[0] + table.deda[1]) / (a0 + a1));
    lambda[1] =
        fmax(0.0f, fabs(denom) > eps
                       ? (table.deda[0] - table.deda[1]) / denom
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(1, 1)) -
                             0.5f * (table.d2ed2a(0, 1) + table.d2ed2a(1, 0)));
    lambda[2] = fmax(0.0f, table.deda[0] / a0);
    lambda[3] = fmax(0.0f, table.deda[1] / a1);
    lambda[4] = fmax(0.0f, eigvalues[0]);
    lambda[5] = fmax(0.0f, eigvalues[1]);
    Mat6x6f result = Mat6x6f::Zero();
    for (int i = 0; i < 6; ++i) {
        if (lambda[i]) {
            Mat3x2f tmp = U * Q[i] * Vt;
            Map<Vec6f> q(tmp.data());
            result += lambda[i] * q * q.transpose();
        }
    }
    return result;
}

__device__ Mat3x3f compute_force(const DiffTable3 &table, const Svd3x3 &svd) {
    Mat3x3f result;
    const Mat3x3f Ut = svd.U.transpose();
    const Mat3x3f V = svd.Vt.transpose();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Mat3x3f df = Mat3x3f::Zero();
            df(i, j) = 1.0f;
            const Vec3f dadf = (Ut * df * V).diagonal();
            result(i, j) = table.deda.dot(dadf);
        }
    }
    return result;
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
    lambda[0] = fmax(0.0f, (table.deda[0] + table.deda[1]) / (a + b));
    lambda[1] = fmax(0.0f, (table.deda[0] + table.deda[2]) / (a + c));
    lambda[2] = fmax(0.0f, (table.deda[1] + table.deda[2]) / (b + c));
    lambda[3] =
        fmax(0.0f, fabs(denom_ab) > eps
                       ? (table.deda[0] - table.deda[1]) / denom_ab
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(1, 1)) -
                             0.5f * (table.d2ed2a(0, 1) + table.d2ed2a(1, 0)));
    lambda[4] =
        fmax(0.0f, fabs(denom_ac) > eps
                       ? (table.deda[0] - table.deda[2]) / denom_ac
                       : 0.5f * (table.d2ed2a(0, 0) + table.d2ed2a(2, 2)) -
                             0.5f * (table.d2ed2a(0, 2) + table.d2ed2a(2, 0)));
    lambda[5] =
        fmax(0.0f, fabs(denom_bc) > eps
                       ? (table.deda[1] - table.deda[2]) / denom_bc
                       : 0.5f * (table.d2ed2a(1, 1) + table.d2ed2a(2, 2)) -
                             0.5f * (table.d2ed2a(1, 2) + table.d2ed2a(2, 1)));
    lambda[6] = fmax(0.0f, eigvalues[0]);
    lambda[7] = fmax(0.0f, eigvalues[1]);
    lambda[8] = fmax(0.0f, eigvalues[2]);
    Mat9x9f result = Mat9x9f::Zero();
    for (int i = 0; i < 9; ++i) {
        if (lambda[i]) {
            Mat3x3f tmp = svd.U * Q[i] * svd.Vt;
            Map<Vec9f> q(tmp.data());
            result += lambda[i] * q * q.transpose();
        }
    }
    return result;
}

} // namespace eigenanalysis
