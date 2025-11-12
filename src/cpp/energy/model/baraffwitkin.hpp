// File: baraffwitkin.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef BARAFFWITKIN_HPP
#define BARAFFWITKIN_HPP

#include "../../common.hpp"
#include "../../data.hpp"

namespace BaraffWitkin {

__device__ float sqr(float x) { return x * x; }

__device__ float stretch_energy(const Mat3x2f &F, float mu) {
    float i5u = F.col(0).squaredNorm();
    float i5v = F.col(1).squaredNorm();
    return 0.5f * mu * (sqr(sqrtf(i5u) - 1.0f) + sqr(sqrtf(i5v) - 1.0f));
}

__device__ float shear_energy(const Mat3x2f &F, float lmd) {
    float i6 = F.col(0).dot(F.col(1));
    return 0.5f * lmd * sqr(i6);
}

__device__ Mat3x2f stretch_gradient(const Mat3x2f &F, float mu) {
    const Vec3f fu = F.col(0);
    const Vec3f fv = F.col(1);
    float norm_u = fu.norm();
    float norm_v = fv.norm();
    Mat3x2f result;
    result << fu * (norm_u - 1.0f) / norm_u, fv * (norm_v - 1.0f) / norm_v;
    return mu * result;
}

__device__ Mat6x6f stretch_hessian(const Mat3x2f &F, float mu) {
    float f1 = F(0, 0), f2 = F(1, 0), f3 = F(2, 0);
    float f4 = F(0, 1), f5 = F(1, 1), f6 = F(2, 1);
    float norm_1 = F.col(0).norm();
    float norm_2 = F.col(1).norm();

    Vec3f q1(f1, f2, f3);
    Vec3f q2, q2_1, q2_2, q2_3;
    q2_1 = Vec3f(f3, 0.0f, -f1).cross(q1);
    q2_2 = Vec3f(0.0f, f3, -f2).cross(q1);
    q2_3 = Vec3f(f2, -f1, 0.0f).cross(q1);
    float q2_sqr_norms[] = {q2_1.squaredNorm(), q2_2.squaredNorm(),
                            q2_3.squaredNorm()};
    if (q2_sqr_norms[0] > q2_sqr_norms[1] &&
        q2_sqr_norms[0] > q2_sqr_norms[2]) {
        q2 = q2_1;
    } else if (q2_sqr_norms[1] > q2_sqr_norms[0] &&
               q2_sqr_norms[1] > q2_sqr_norms[2]) {
        q2 = q2_2;
    } else {
        q2 = q2_3;
    }
    Vec3f q3 = q1.cross(q2);

    q1.normalize();
    q2.normalize();
    q3.normalize();

    float lmd1 = 1.0f;
    float lmd2 = 1.0f - 1.0f / norm_1;
    float lmd3 = 1.0f - 1.0f / norm_1;

    Vec3f q4(f4, f5, f6);
    Vec3f q5, q5_1, q5_2, q5_3;
    q5_1 = Vec3f(f6, 0.0f, -f4).cross(q4);
    q5_2 = Vec3f(0.0f, f6, -f5).cross(q4);
    q5_3 = Vec3f(f5, -f4, 0.0f).cross(q4);
    float sqr_norms[] = {q5_1.squaredNorm(), q5_2.squaredNorm(),
                         q5_3.squaredNorm()};
    if (sqr_norms[0] > sqr_norms[1] && sqr_norms[0] > sqr_norms[2]) {
        q5 = q5_1;
    } else if (sqr_norms[1] > sqr_norms[0] && sqr_norms[1] > sqr_norms[2]) {
        q5 = q5_2;
    } else {
        q5 = q5_3;
    }
    Vec3f q6 = q4.cross(q5);

    q4.normalize();
    q5.normalize();
    q6.normalize();

    float lmd4 = 1.0f;
    float lmd5 = 1.0f - 1.0f / norm_2;
    float lmd6 = 1.0f - 1.0f / norm_2;

    Mat3x3f H1 = Mat3x3f::Zero();
    Mat3x3f H2 = Mat3x3f::Zero();

    H1 += std::max(0.0f, lmd1) * q1 * q1.transpose();
    H1 += std::max(0.0f, lmd2) * q2 * q2.transpose();
    H1 += std::max(0.0f, lmd3) * q3 * q3.transpose();

    H2 += std::max(0.0f, lmd4) * q4 * q4.transpose();
    H2 += std::max(0.0f, lmd5) * q5 * q5.transpose();
    H2 += std::max(0.0f, lmd6) * q6 * q6.transpose();

    Mat6x6f H = Mat6x6f::Zero();
    H.block<3, 3>(0, 0) = H1;
    H.block<3, 3>(3, 3) = H2;

    return mu * H;
}

__device__ Mat3x2f shear_gradient(const Mat3x2f &F, float lmd) {
    float w = F.col(0).dot(F.col(1));
    Mat3x2f result;
    result << w * F.col(1), w * F.col(0);
    return lmd * result;
}

__device__ Mat2x2f helper_adj2x2(const Mat2x2f &A) {
    Mat2x2f result;
    result << A(1, 1), -A(0, 1), -A(1, 0), A(0, 0);
    return result;
}

__device__ Mat6x6f shear_hessian(const Mat3x2f &F, float lmd) {
    float I6 = F.col(0).dot(F.col(1));
    float I2 = (F.array() * F.array()).sum();
    Mat6x6f H = Mat6x6f::Zero();
    H.block<3, 3>(0, 3) = Mat3x3f::Identity();
    H.block<3, 3>(3, 0) = Mat3x3f::Identity();

    Vec6f g;
    g.segment<3>(0) = F.col(1);
    g.segment<3>(3) = F.col(0);

    float lmd0 = I2 + sqrt(I2 * I2 + 12.0f * I6 * I6);
    Vec6f q0 = 2.0f * I6 * (H * g) + lmd0 * g;

    Vec6f z;
    float lmd1, lmd2;
    Vec6f w1, w2, w3;

    if (I6 > 0) {
        z = q0;
        lmd1 = lmd2 = 2.0f * I6;
        w1 << 1, 1, 0, 0, 0, 0;
        w2 << 0, 0, 1, 1, 0, 0;
        w3 << 0, 0, 0, 0, 1, 1;
    } else {
        lmd1 = lmd2 = -2.0f * I6;
        float tmp = I2 - sqrt(I2 * I2 + 12.0f * I6 * I6);
        z = 2.0f * I6 * (H * g) + tmp * g;
        w1 << 1, -1, 0, 0, 0, 0;
        w2 << 0, 0, 1, -1, 0, 0;
        w3 << 0, 0, 0, 0, 1, -1;
    }

    float a1 = z.dot(w1);
    float a2 = z.dot(w2);
    float a3 = z.dot(w3);

    Vec6f q1, q1_1, q1_2, q1_3;
    q1_1 = a3 * w1 + a3 * w2 - (a1 - a2) * w3;
    q1_2 = a2 * w1 - (a1 - a3) * w2 + a2 * w3;
    q1_3 = -(a2 - a3) * w1 + a1 * w2 + a1 * w3;
    float q1_sqr_norms[] = {q1_1.squaredNorm(), q1_2.squaredNorm(),
                            q1_3.squaredNorm()};
    if (q1_sqr_norms[0] > q1_sqr_norms[1] &&
        q1_sqr_norms[0] > q1_sqr_norms[2]) {
        q1 = q1_1;
    } else if (q1_sqr_norms[1] > q1_sqr_norms[0] &&
               q1_sqr_norms[1] > q1_sqr_norms[2]) {
        q1 = q1_2;
    } else {
        q1 = q1_3;
    }

    float b1 = q1.dot(w1);
    float b2 = q1.dot(w2);
    float b3 = q1.dot(w3);

    Mat2x2f A1, A2, A3;
    A1 << a2, a3, b2, b3;
    A2 << a3, a1, b3, b1;
    A3 << a1, a2, b1, b2;

    Vec2f rhs1(-a1, -b1), rhs2(-a2, -b2), rhs3(-a3, -b3);

    float detA1 = A1.determinant();
    float detA2 = A2.determinant();
    float detA3 = A3.determinant();

    Vec6f q2, q2_1, q2_2, q2_3;
    Vec2f c45 = helper_adj2x2(A3) * rhs3;
    Vec2f c56 = helper_adj2x2(A1) * rhs1;
    Vec2f c64 = helper_adj2x2(A2) * rhs2;
    q2_1 = c45[0] * w1 + c45[1] * w2 + detA3 * w3;
    q2_2 = detA1 * w1 + c56[0] * w2 + c56[1] * w3;
    q2_3 = c64[1] * w1 + detA2 * w2 + c64[0] * w3;
    float q2_sqr_norms[] = {q2_1.squaredNorm(), q2_2.squaredNorm(),
                            q2_3.squaredNorm()};
    if (q2_sqr_norms[0] > q2_sqr_norms[1] &&
        q2_sqr_norms[0] > q2_sqr_norms[2]) {
        q2 = q2_1;
    } else if (q2_sqr_norms[1] > q2_sqr_norms[0] &&
               q2_sqr_norms[1] > q2_sqr_norms[2]) {
        q2 = q2_2;
    } else {
        q2 = q2_3;
    }
    q0.normalize();
    q1.normalize();
    q2.normalize();

    Mat6x6f result = Mat6x6f::Zero();
    result += std::max(0.0f, lmd0) * (q0 * q0.transpose());
    result += std::max(0.0f, lmd1) * (q1 * q1.transpose());
    result += std::max(0.0f, lmd2) * (q2 * q2.transpose());

    return 0.5f * lmd * result;
}

} // namespace BaraffWitkin

#endif
