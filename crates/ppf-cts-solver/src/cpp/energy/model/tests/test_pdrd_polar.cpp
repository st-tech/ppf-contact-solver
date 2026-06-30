// File: test_pdrd_polar.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Host (no-CUDA) regression test for the PDRD exact-rigid polar fit. It runs on
// plain clang++/g++ (e.g. macOS, no nvcc) because pdrd_polar.hpp is free of
// CUDA/Eigen deps. It guards the rigid-body collapse bug (wreck scene,
// Box_4_0_5): a cube that settled exactly 180 degrees from its rest pose lost
// half its volume in one frame because the identity-warm-started polar fit
// returned a wrong rotation at the 180-degree antipodal singularity of SO(3),
// and the rigidify partial-snap then lerped the body linearly across that wrong
// rotation. The fix seeds the fit from a Gram-Schmidt orthonormalization of M.
//
// What is asserted:
//   1. rigid_polar_quat recovers the true rotation for the exact wreck M and for
//      exact 180-degree rotations about X/Y/Z and a tilted axis (the bug case).
//   2. It recovers random rotations (isotropic + anisotropic stretch) to <0.5deg
//      and always returns a proper rotation (R^T R = I, det = +1).
//   3. The end-to-end rigidity property: fitting a 180-degree-rotated cube and
//      partial-snapping toward the fitted rigid target keeps det(F) ~ 1 (no
//      collapse). With the old identity seed this det dropped to ~0.49; the test
//      shows the old path failing and the fixed path passing.

#include "../pdrd_polar.hpp"

#include <cmath>
#include <cstdio>

using PDRD::rigid_polar_quat;
using PDRD::rigid_quat_to_mat;

namespace {

int g_failures = 0;
constexpr float PI = 3.14159265358979323846f;

// ---- tiny column-major 3x3 helpers (M[i + 3*j] = row i, col j) -------------
void matmul(const float A[9], const float B[9], float C[9]) {
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) s += A[i + 3 * k] * B[k + 3 * j];
            C[i + 3 * j] = s;
        }
}
float det3(const float M[9]) {
    return M[0] * (M[4] * M[8] - M[5] * M[7]) -
           M[3] * (M[1] * M[8] - M[2] * M[7]) +
           M[6] * (M[1] * M[5] - M[2] * M[4]);
}
bool inv3(const float M[9], float out[9]) {
    float d = det3(M);
    if (std::fabs(d) < 1e-20f) return false;
    float id = 1.0f / d;
    out[0] = (M[4] * M[8] - M[5] * M[7]) * id;
    out[1] = (M[2] * M[7] - M[1] * M[8]) * id;
    out[2] = (M[1] * M[5] - M[2] * M[4]) * id;
    out[3] = (M[5] * M[6] - M[3] * M[8]) * id;
    out[4] = (M[0] * M[8] - M[2] * M[6]) * id;
    out[5] = (M[2] * M[3] - M[0] * M[5]) * id;
    out[6] = (M[3] * M[7] - M[4] * M[6]) * id;
    out[7] = (M[1] * M[6] - M[0] * M[7]) * id;
    out[8] = (M[0] * M[4] - M[1] * M[3]) * id;
    return true;
}
float angle_deg(const float R[9]) {
    float t = (R[0] + R[4] + R[8] - 1.0f) * 0.5f;
    if (t > 1.0f) t = 1.0f;
    if (t < -1.0f) t = -1.0f;
    return std::acos(t) * 180.0f / PI;
}
// Rotation about a unit-ish axis by angle (radians), column-major (Rodrigues).
void axis_angle(float ax, float ay, float az, float th, float R[9]) {
    float n = std::sqrt(ax * ax + ay * ay + az * az);
    ax /= n; ay /= n; az /= n;
    float c = std::cos(th), s = std::sin(th), C = 1.0f - c;
    R[0] = c + ax * ax * C;       R[1] = ay * ax * C + az * s; R[2] = az * ax * C - ay * s;
    R[3] = ax * ay * C - az * s;  R[4] = c + ay * ay * C;      R[5] = az * ay * C + ax * s;
    R[6] = ax * az * C + ay * s;  R[7] = ay * az * C - ax * s; R[8] = c + az * az * C;
}
// Relative rotation angle between A and B (degrees): angle(A * B^T).
float rot_err_deg(const float A[9], const float B[9]) {
    float Bt[9] = {B[0], B[3], B[6], B[1], B[4], B[7], B[2], B[5], B[8]};
    float D[9];
    matmul(A, Bt, D);
    return angle_deg(D);
}
float orthonormality_err(const float R[9]) {
    float Rt[9] = {R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]};
    float R2[9];
    matmul(Rt, R, R2);
    float e = 0.0f;
    for (int i = 0; i < 9; ++i) {
        float id = (i % 4 == 0) ? 1.0f : 0.0f; // identity diagonal
        e = std::fmax(e, std::fabs(R2[i] - id));
    }
    return e;
}

void check(bool ok, const char *name, float val, float tol) {
    if (ok) {
        std::printf("  PASS  %-46s (%.4g <= %.4g)\n", name, val, tol);
    } else {
        std::printf("  FAIL  %-46s (%.4g  !<= %.4g)\n", name, val, tol);
        ++g_failures;
    }
}

// Unit cube rest vertices, rest-centered (ybar). Rest Gram = 0.5*I.
const float YBAR[8][3] = {
    {-0.25f, -0.25f, -0.25f}, {-0.25f, -0.25f, 0.25f}, {-0.25f, 0.25f, -0.25f},
    {-0.25f, 0.25f, 0.25f},   {0.25f, -0.25f, -0.25f}, {0.25f, -0.25f, 0.25f},
    {0.25f, 0.25f, -0.25f},   {0.25f, 0.25f, 0.25f}};

// Cross-covariance M = sum_k (R ybar_k) ybar_k^T, column-major, for a cube
// rotated by R (the configuration the fit sees when the body is rigid at R).
void cube_M(const float R[9], float M[9]) {
    for (int e = 0; e < 9; ++e) M[e] = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const float *yb = YBAR[k];
        float y[3] = {R[0] * yb[0] + R[3] * yb[1] + R[6] * yb[2],
                      R[1] * yb[0] + R[4] * yb[1] + R[7] * yb[2],
                      R[2] * yb[0] + R[5] * yb[1] + R[8] * yb[2]};
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) M[i + 3 * j] += y[i] * yb[j];
    }
}

// Old buggy fit: identity-warm-started Mueller, no Gram-Schmidt seed. Kept local
// so the test can show the regression it guards against.
void rigid_polar_quat_idseed(const float M[9], float Rout[9]) {
    float q[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (int it = 0; it < 20; ++it) {
        float R[9];
        rigid_quat_to_mat(q, R);
        float on0 = 0, on1 = 0, on2 = 0, od = 0;
        for (int c = 0; c < 3; ++c) {
            float r0 = R[3 * c], r1 = R[3 * c + 1], r2 = R[3 * c + 2];
            float a0 = M[3 * c], a1 = M[3 * c + 1], a2 = M[3 * c + 2];
            on0 += r1 * a2 - r2 * a1; on1 += r2 * a0 - r0 * a2;
            on2 += r0 * a1 - r1 * a0; od += r0 * a0 + r1 * a1 + r2 * a2;
        }
        float denom = std::fabs(od) + 1e-9f;
        float w0 = on0 / denom, w1 = on1 / denom, w2 = on2 / denom;
        float ang = std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
        if (ang < 1e-9f) break;
        float s = std::sin(0.5f * ang) / ang;
        float dq[4] = {s * w0, s * w1, s * w2, std::cos(0.5f * ang)};
        float nx = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
        float ny = dq[3] * q[1] - dq[0] * q[2] + dq[1] * q[3] + dq[2] * q[0];
        float nz = dq[3] * q[2] + dq[0] * q[1] - dq[1] * q[0] + dq[2] * q[3];
        float nw = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];
        float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
        q[0] = nx * inv; q[1] = ny * inv; q[2] = nz * inv; q[3] = nw * inv;
    }
    rigid_quat_to_mat(q, Rout);
}

// Deformation gradient det of a cube whose vertices are `pts` (rest-centered map
// ybar -> pts_centered): F = (sum pts_c ybar^T)(sum ybar ybar^T)^{-1}, det(F).
float cube_defgrad_det(const float pts[8][3]) {
    float c[3] = {0, 0, 0};
    for (int k = 0; k < 8; ++k)
        for (int i = 0; i < 3; ++i) c[i] += pts[k][i] / 8.0f;
    float A[9] = {0}, S[9] = {0};
    for (int k = 0; k < 8; ++k) {
        const float *yb = YBAR[k];
        float p[3] = {pts[k][0] - c[0], pts[k][1] - c[1], pts[k][2] - c[2]};
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                A[i + 3 * j] += p[i] * yb[j];
                S[i + 3 * j] += yb[i] * yb[j];
            }
    }
    float Sinv[9], F[9];
    if (!inv3(S, Sinv)) return 0.0f;
    matmul(A, Sinv, F);
    return det3(F);
}

// The end-to-end rigidity property at one orientation: take a cube rigidly
// rotated by Rtrue, fit R from its M, build the rigid target rigid_tgt = R*ybar,
// and partial-snap eval_x toward it (eval_x += t*(rigid_tgt - eval_x)) as the
// rigidify does. Return det(F) of the snapped cube (1 = rigid, <1 = collapse).
float snap_det(const float Rtrue[9], float t,
               void (*fit)(const float[9], float[9])) {
    float M[9];
    cube_M(Rtrue, M);
    float Rfit[9];
    fit(M, Rfit);
    float eval_x[8][3], rigid_tgt[8][3], snapped[8][3];
    for (int k = 0; k < 8; ++k) {
        const float *yb = YBAR[k];
        for (int i = 0; i < 3; ++i) {
            eval_x[k][i] = Rtrue[i] * yb[0] + Rtrue[i + 3] * yb[1] + Rtrue[i + 6] * yb[2];
            rigid_tgt[k][i] = Rfit[i] * yb[0] + Rfit[i + 3] * yb[1] + Rfit[i + 6] * yb[2];
            snapped[k][i] = eval_x[k][i] + t * (rigid_tgt[k][i] - eval_x[k][i]);
        }
    }
    return cube_defgrad_det(snapped);
}

} // namespace

int main() {
    std::printf("PDRD polar-fit regression (host, no CUDA)\n");

    // ---- Test 1: exact wreck M (Box_4_0_5 settled ~180deg about a Y-ish axis).
    std::printf("[1] wreck M (settled 180 deg)\n");
    {
        // Column-major, measured from the wreck PC2 (sample 351).
        float M[9] = {-0.49227f, 0.08758f, 0.00031f, 0.08758f, 0.49227f,
                      -0.00006f, -0.00032f, 0.0f, -0.5f};
        float R[9];
        rigid_polar_quat(M, R);
        check(std::fabs(angle_deg(R) - 180.0f) < 1.0f, "fit angle ~ 180 deg",
              std::fabs(angle_deg(R) - 180.0f), 1.0f);
        check(orthonormality_err(R) < 1e-4f, "R is orthonormal", orthonormality_err(R), 1e-4f);
        check(std::fabs(det3(R) - 1.0f) < 1e-4f, "det(R) ~ +1", std::fabs(det3(R) - 1.0f), 1e-4f);
    }

    // ---- Test 2: exact 180-deg rotations about X/Y/Z and a tilted axis. This is
    // the precise failure case: a symmetric cube's M is diagonal with negative
    // trace, so the identity-seeded fit stalls. The Gram-Schmidt seed must give 180.
    std::printf("[2] exact 180-deg rotations (the bug case)\n");
    {
        const float axes[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.088f, 0.996f, 0.0f}};
        const char *names[4] = {"180 about X", "180 about Y", "180 about Z", "180 about tilted axis"};
        for (int a = 0; a < 4; ++a) {
            float Rt[9], M[9], R[9];
            axis_angle(axes[a][0], axes[a][1], axes[a][2], PI, Rt);
            cube_M(Rt, M);
            rigid_polar_quat(M, R);
            check(rot_err_deg(R, Rt) < 0.5f, names[a], rot_err_deg(R, Rt), 0.5f);
        }
    }

    // ---- Test 3: random rotations, isotropic + anisotropic stretch -----------
    std::printf("[3] random rotations (isotropic + anisotropic stretch)\n");
    {
        unsigned seed = 0x12345u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)(seed >> 8) / (float)(1u << 24); // [0,1)
        };
        float worst_err = 0.0f, worst_orth = 0.0f;
        for (int trial = 0; trial < 4000; ++trial) {
            float ax = rnd() * 2 - 1, ay = rnd() * 2 - 1, az = rnd() * 2 - 1;
            float th = rnd() * PI; // [0, pi]
            if (ax * ax + ay * ay + az * az < 1e-6f) continue;
            float Rt[9];
            axis_angle(ax, ay, az, th, Rt);
            // M = Rt * diag(sx,sy,sz) (anisotropic stretch); column-major.
            float sx = 0.3f + rnd() * 1.5f, sy = 0.3f + rnd() * 1.5f, sz = 0.3f + rnd() * 1.5f;
            float M[9] = {Rt[0] * sx, Rt[1] * sx, Rt[2] * sx, Rt[3] * sy, Rt[4] * sy,
                          Rt[5] * sy, Rt[6] * sz, Rt[7] * sz, Rt[8] * sz};
            float R[9];
            rigid_polar_quat(M, R);
            worst_err = std::fmax(worst_err, rot_err_deg(R, Rt));
            worst_orth = std::fmax(worst_orth, orthonormality_err(R));
        }
        check(worst_err < 0.5f, "worst rotation error over 4000 trials", worst_err, 0.5f);
        check(worst_orth < 1e-3f, "worst orthonormality error", worst_orth, 1e-3f);
    }

    // ---- Test 4: end-to-end rigidity, fit + partial rigidify snap ------------
    // The actual bug surfaced as a half-volume collapse (det(F) ~ 0.49) when the
    // 180-deg fit was wrong and the snap lerped linearly across it. At an exact
    // antipode (cube rotated 180 deg about a coordinate axis) the cross-
    // covariance is diagonal with zero identity-seed torque, so the OLD fit
    // stalls and the snap collapses the body. The FIXED fit must keep det(F) ~ 1
    // for every snap fraction; we also assert the OLD path collapses, so the test
    // self-verifies that it exercises the bug regime (not a benign orientation).
    std::printf("[4] rigidity under partial snap (det(F) ~ 1, no collapse)\n");
    {
        // Exact diagonal 180-deg rotations about X, Y, Z (built directly, not via
        // axis_angle, to avoid a sin(pi) fp residual; these give an exactly
        // diagonal M with zero identity-seed torque, the true degenerate case).
        const float ROTS[3][9] = {
            {1, 0, 0, 0, -1, 0, 0, 0, -1},  // 180 about X = diag(1,-1,-1)
            {-1, 0, 0, 0, 1, 0, 0, 0, -1},  // 180 about Y = diag(-1,1,-1)
            {-1, 0, 0, 0, -1, 0, 0, 0, 1}}; // 180 about Z = diag(-1,-1,1)
        float worst_dev_fixed = 0.0f, worst_collapse_old = 0.0f;
        for (int a = 0; a < 3; ++a) {
            const float *Rt = ROTS[a];
            for (int i = 1; i <= 9; ++i) {
                float t = 0.1f * i; // snap fraction in (0,1]
                worst_dev_fixed = std::fmax(worst_dev_fixed,
                                            std::fabs(snap_det(Rt, t, rigid_polar_quat) - 1.0f));
                // collapse = volume lost (1 - det); worst over fractions/axes.
                worst_collapse_old = std::fmax(worst_collapse_old,
                                               1.0f - snap_det(Rt, t, rigid_polar_quat_idseed));
            }
        }
        check(worst_dev_fixed < 1e-2f, "fixed: |det(F)-1| over snaps/axes",
              worst_dev_fixed, 1e-2f);
        // Self-check: the old identity-seed fit must actually collapse here,
        // confirming the test reproduces the bug the fix resolves.
        bool old_collapses = worst_collapse_old > 0.3f;
        if (old_collapses) {
            std::printf("  PASS  %-46s (vol loss %.3f >= 0.3)\n",
                        "old identity-seed collapses (test in regime)", worst_collapse_old);
        } else {
            std::printf("  FAIL  %-46s (vol loss %.3f < 0.3; test not exercising bug)\n",
                        "old identity-seed collapses (test in regime)", worst_collapse_old);
            ++g_failures;
        }
    }

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL PASS" : "FAILED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
