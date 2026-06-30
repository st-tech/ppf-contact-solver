// File: pdrd_polar.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pure-float rotation math for the PDRD exact-rigid fit: quaternion <-> matrix
// conversion and the best-fit (polar) rotation of a near-rotation 3x3 matrix.
// Deliberately free of CUDA, Eigen, and buffer dependencies so it can be unit
// tested on a plain host compiler with no nvcc (see tests/test_pdrd_polar.cpp)
// and is shared verbatim with the CUDA fit kernel in pdrd_rigid.hpp.

#ifndef PDRD_POLAR_HPP
#define PDRD_POLAR_HPP

#include <cmath>

// Host/device portability: nvcc compiles these for both host and device; a plain
// host compiler (the unit test) sees no annotation.
#ifdef __CUDACC__
#define PDRD_POLAR_HD __device__ __host__
#else
#define PDRD_POLAR_HD
#endif

namespace PDRD {

// Quaternion (x, y, z, w) to a column-major 3x3 rotation matrix.
PDRD_POLAR_HD inline void rigid_quat_to_mat(const float q[4], float R[9]) {
    float x = q[0], y = q[1], z = q[2], w = q[3];
    R[0] = 1.0f - 2.0f * (y * y + z * z); // col 0
    R[1] = 2.0f * (x * y + z * w);
    R[2] = 2.0f * (x * z - y * w);
    R[3] = 2.0f * (x * y - z * w);        // col 1
    R[4] = 1.0f - 2.0f * (x * x + z * z);
    R[5] = 2.0f * (y * z + x * w);
    R[6] = 2.0f * (x * z + y * w);        // col 2
    R[7] = 2.0f * (y * z - x * w);
    R[8] = 1.0f - 2.0f * (x * x + y * y);
}

// Quaternion (x, y, z, w) from a proper column-major rotation matrix R, via
// Shepperd's largest-diagonal branch so it is exact at 180 degrees (where the
// naive trace branch divides by ~0).
PDRD_POLAR_HD inline void rigid_mat_to_quat(const float R[9], float q[4]) {
    // Column-major: R(row,col) = R[row + 3*col].
    const float r00 = R[0], r10 = R[1], r20 = R[2];
    const float r01 = R[3], r11 = R[4], r21 = R[5];
    const float r02 = R[6], r12 = R[7], r22 = R[8];
    const float tr = r00 + r11 + r22;
    if (tr > 0.0f) {
        float s = std::sqrt(tr + 1.0f) * 2.0f; // s = 4w
        q[3] = 0.25f * s;
        q[0] = (r21 - r12) / s;
        q[1] = (r02 - r20) / s;
        q[2] = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
        float s = std::sqrt(1.0f + r00 - r11 - r22) * 2.0f; // s = 4x
        q[3] = (r21 - r12) / s;
        q[0] = 0.25f * s;
        q[1] = (r01 + r10) / s;
        q[2] = (r02 + r20) / s;
    } else if (r11 > r22) {
        float s = std::sqrt(1.0f + r11 - r00 - r22) * 2.0f; // s = 4y
        q[3] = (r02 - r20) / s;
        q[0] = (r01 + r10) / s;
        q[1] = 0.25f * s;
        q[2] = (r12 + r21) / s;
    } else {
        float s = std::sqrt(1.0f + r22 - r00 - r11) * 2.0f; // s = 4z
        q[3] = (r10 - r01) / s;
        q[0] = (r02 + r20) / s;
        q[1] = (r12 + r21) / s;
        q[2] = 0.25f * s;
    }
}

// Best-fit rotation (polar factor) of a near-rotation 3x3 M (column-major), via
// the quaternion fixed-point iteration of Mueller et al., "A Robust Method to
// Extract the Rotational Part of Deformations" (2016): omega = (sum_c R_c x M_c)
// / (|sum_c R_c . M_c| + eps), q <- exp(omega) q. Pure float, no matrix inverse,
// so it is allocation-free and device-safe (the rest of this code avoids Eigen
// in hot kernels for the same reason).
//
// SEED: a Gram-Schmidt orthonormalization of M's columns (not the identity).
// The identity seed is degenerate at the 180-degree antipodal singularity of
// SO(3): for a symmetric body (e.g. a cube, whose rest Gram is isotropic so
// M ~ R*sigma) settled near a half-turn from its rest pose, M is nearly diagonal
// with negative trace, the cross-product torque at the identity is ~0, and the
// fixed 20-iteration crawl can return a grossly wrong rotation. That wrong R
// then drives the rigidify snap to lerp the body across a large angle and
// collapse it. The Gram-Schmidt seed lands on (or very near) the true rotation
// for any orientation, including 180 degrees, so the Mueller iteration only
// polishes. Falls back to the identity seed if M is degenerate (near-zero
// columns), which is not the antipodal case.
PDRD_POLAR_HD inline void rigid_polar_quat(const float M[9], float Rout[9]) {
    float q[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    {
        // Gram-Schmidt on the columns m0,m1,m2 of M -> orthonormal seed R_s.
        float m0[3] = {M[0], M[1], M[2]};
        float m1[3] = {M[3], M[4], M[5]};
        float n0 = std::sqrt(m0[0] * m0[0] + m0[1] * m0[1] + m0[2] * m0[2]);
        if (n0 > 1e-12f) {
            float u0[3] = {m0[0] / n0, m0[1] / n0, m0[2] / n0};
            float d = m1[0] * u0[0] + m1[1] * u0[1] + m1[2] * u0[2];
            float t1[3] = {m1[0] - d * u0[0], m1[1] - d * u0[1],
                           m1[2] - d * u0[2]};
            float n1 = std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
            if (n1 > 1e-12f) {
                float u1[3] = {t1[0] / n1, t1[1] / n1, t1[2] / n1};
                // u2 = u0 x u1 (right-handed -> proper rotation, det +1).
                float u2[3] = {u0[1] * u1[2] - u0[2] * u1[1],
                               u0[2] * u1[0] - u0[0] * u1[2],
                               u0[0] * u1[1] - u0[1] * u1[0]};
                float Rs[9] = {u0[0], u0[1], u0[2], u1[0], u1[1],
                               u1[2], u2[0], u2[1], u2[2]};
                rigid_mat_to_quat(Rs, q);
            }
        }
    }
    for (unsigned iter = 0; iter < 20; ++iter) {
        float R[9];
        rigid_quat_to_mat(q, R);
        float on0 = 0.0f, on1 = 0.0f, on2 = 0.0f, od = 0.0f;
        for (unsigned c = 0; c < 3; ++c) {
            float r0 = R[3 * c], r1 = R[3 * c + 1], r2 = R[3 * c + 2];
            float a0 = M[3 * c], a1 = M[3 * c + 1], a2 = M[3 * c + 2];
            on0 += r1 * a2 - r2 * a1;
            on1 += r2 * a0 - r0 * a2;
            on2 += r0 * a1 - r1 * a0;
            od += r0 * a0 + r1 * a1 + r2 * a2;
        }
        float denom = std::fabs(od) + 1e-9f;
        float w0 = on0 / denom, w1 = on1 / denom, w2 = on2 / denom;
        float ang = std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
        if (ang < 1e-9f) break;
        float s = std::sin(0.5f * ang) / ang;
        float dq[4] = {s * w0, s * w1, s * w2, std::cos(0.5f * ang)};
        // q <- dq * q  (quaternion product), then renormalize.
        float nx = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
        float ny = dq[3] * q[1] - dq[0] * q[2] + dq[1] * q[3] + dq[2] * q[0];
        float nz = dq[3] * q[2] + dq[0] * q[1] - dq[1] * q[0] + dq[2] * q[3];
        float nw = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];
        float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
        q[0] = nx * inv;
        q[1] = ny * inv;
        q[2] = nz * inv;
        q[3] = nw * inv;
    }
    rigid_quat_to_mat(q, Rout);
}

} // namespace PDRD

#endif // PDRD_POLAR_HPP
