// File: rigid_core.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Body-agnostic SO(3) rotation math (skew, inverse-skew, the exponential map)
// shared by PDRD exact-rigid bodies (pdrd_rigid.hpp) and SAND grain spin
// (sand_rigid.hpp). Factoring it here lets grain code reuse PDRD's rotation
// primitives without pulling in the per-body reduced-solve machinery. The pure
// quaternion/polar math is in pdrd_polar.hpp (host-testable, Eigen-free); this
// header holds the Eigen (Mat3x3f / Vec3f) device math used inside the CUDA
// kernels.

#ifndef RIGID_CORE_HPP
#define RIGID_CORE_HPP

#include "../../data.hpp"
#include <cmath>

namespace RigidCore {

// 3x3 skew-symmetric matrix of a vector: skew(v) w = v x w.
__device__ __host__ inline Mat3x3f rigid_skew(const Vec3f &v) {
    Mat3x3f S;
    S(0, 0) = 0.0f;   S(0, 1) = -v[2]; S(0, 2) = v[1];
    S(1, 0) = v[2];   S(1, 1) = 0.0f;  S(1, 2) = -v[0];
    S(2, 0) = -v[1];  S(2, 1) = v[0];  S(2, 2) = 0.0f;
    return S;
}

// Inverse of skew: the axial vector of the antisymmetric part of A.
__device__ __host__ inline Vec3f rigid_skew_inv(const Mat3x3f &A) {
    return Vec3f(0.5f * (A(2, 1) - A(1, 2)), 0.5f * (A(0, 2) - A(2, 0)),
                 0.5f * (A(1, 0) - A(0, 1)));
}

// Exponential map so(3) -> SO(3) (Rodrigues), numerically safe near zero.
__device__ __host__ inline Mat3x3f rigid_exp_so3(const Vec3f &theta) {
    float a2 = theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2];
    float a = std::sqrt(a2);
    Mat3x3f K = rigid_skew(theta);
    Mat3x3f I = Mat3x3f::Identity();
    // sinc(a) = sin(a)/a, (1-cos a)/a^2, with Taylor fallback near 0.
    float s, c;
    if (a < 1e-5f) {
        s = 1.0f - a2 / 6.0f;          // sin(a)/a
        c = 0.5f - a2 / 24.0f;         // (1 - cos a)/a^2
    } else {
        s = std::sin(a) / a;
        c = (1.0f - std::cos(a)) / a2;
    }
    return I + s * K + c * (K * K);
}

} // namespace RigidCore

#endif // RIGID_CORE_HPP
