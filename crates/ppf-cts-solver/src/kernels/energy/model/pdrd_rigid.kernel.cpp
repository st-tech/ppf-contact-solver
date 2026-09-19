// File: pdrd_rigid.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A NEUTRAL KERNEL SOURCE holding the PDRD exact-rigid arithmetic. This file is
// plain C++ and belongs to no backend: no preprocessor conditional, no macro of
// its own, and no spelling that only nvcc or only the Metal shader compiler
// accepts. ppf-cts-compute/seam/kernelgen.py renders it into the three forms the three
// compilers read. The two facts a backend cannot infer are written as C++
// attributes: `[[seam::host_device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a pointer or reference parameter,
// which MSL requires on every one of them.
//
// Everything here operates on thread-space VALUES: no launch geometry, no
// reduction, no atomic, no address space other than `[[seam::thread]]`. The
// reductions (the per-body centroid and cross-covariance, the reduced
// restriction, the 6x6 assembly) stay in each backend's kernel entry point,
// because a threadgroup barrier and a shared-memory accumulator have no common
// spelling and sharing them would buy nothing.
//
// The split follows the rest of the port: pdrd_rigid.hpp keeps the CUDA launches
// and calls these bodies, and the Metal backend's PDRD entry points call the
// same bytes.
// A change to the rigid kinematics therefore lands on both backends at once, or
// on neither.
//
// Layout conventions carried over unchanged from the CUDA implementation:
//   - a rotation is 9 floats, COLUMN major, matching Mat3x3f's storage;
//   - a quaternion is 4 floats ordered (x, y, z, w);
//   - a reduced per-body block is 6 floats (dx_b, dtheta_b);
//   - a reduced 6x6 self-block is 36 floats, ROW major.
//
// ON THE TWO TRANSCENDENTAL SITES. The rotation increment's sine and cosine are
// spelled through fmath::sin_bounded / fmath::cos_bounded rather than a plain
// sin/cos, because on CUDA the choice is load bearing: the library sinf and cosf
// reduce their argument in 64-bit integer and double arithmetic, so a kernel
// that merely calls one emits FP64 and violates the solver's float32-only rule
// (the release build's SASS guard fails on it). The CUDA forms behind those two
// names are therefore fmath's special-function-unit ones, which are float
// throughout, and the _bounded spelling records the precondition each call site
// can prove: both arguments here are one Newton iteration's rotation increment,
// and the result is re-orthogonalized by the next iteration's rigid fit, which
// is far coarser than the intrinsic's error. MSL has no double at all, so the
// Metal prologue maps these to plain sin and cos; that is slightly MORE accurate
// than the CUDA side, which makes these the two sites in this file where the
// backends are not bit-identical. Every other operation here is.

#include "../../csrmat/fixed_csr.kernel.cpp"

#pragma once

// THREE DEPENDENCIES, ALL NAMED HERE, because this file stands on its own:
// tests/test_pdrd_polar.cpp compiles it with a plain C++ compiler and no nvcc.
// `seam/seam.hpp` supplies the `fmath::` arithmetic the bodies call,
// float_math.hpp the single-precision sine and cosine that fmath::sin_bounded
// and fmath::cos_bounded resolve to there, and smat.hpp the header-only linear
// algebra. The Metal backend resolves all three itself, splicing the segments
// it needs in dependency order.
#include "../../float_math.hpp"
#include "../../linalg/smat.hpp"
#include "../../seam/seam.hpp"

// The two vector/matrix names the bodies below use. data.hpp declares the same
// aliases for the same underlying types, and the shader's alias segment
// declares them again, so any arrival order is fine: an alias redeclaration
// naming an identical type is legal.
using Vec3f = linalg::SVec<float, 3>;
using Mat3x3f = linalg::SMat<float, 3, 3>;

// The bodies below are reached from the host as well as from a kernel: the CUDA
// rigid-fit kernels call them on the device, and pdrd_polar.hpp wraps them for
// the host, which is how tests/test_pdrd_polar.cpp exercises them. So they take
// the host-and-device spelling rather than the device-only one.

// The reduced solve's per-body DOF count, its 6x6 block size, and the per-vertex
// body id that means "this vertex belongs to no body". Enumerators rather than
// `constexpr` variables: MSL rejects a program-scope constexpr variable outright
// ("program scope variable must reside in constant address space"), and an
// enumerator carries the same value in both languages.
//
// The cloth marker is here rather than beside the reduced map on either backend
// because both backends fill and test the same array: PDRD::RIGID_UNSET
// (pdrd_lock_projector.hpp) and the Metal reduced map are both defined from this
// enumerator, so there is one value to move rather than three to keep in step.
//
// The per-body joint modes are here for the same reason. PdrdBodyProp::joint_mode
// is read by the CUDA projector (pdrd_lock_projector.hpp, whose PDRD_JOINT_*
// names are now defined from these) and by the Metal backend's capability gate,
// which cannot include that header at all because it declares a __global__
// kernel.
enum : unsigned {
    PDRD_BODY_DOFS = 6u,
    PDRD_BLOCK_FLOATS = 36u,
    PDRD_CLOTH_MARKER = 0xffffffffu,
    PDRD_JOINT_FREE = 0u,   // full 6-DOF rigid body
    PDRD_JOINT_HINGE = 1u,  // translation locked, spin about the axle
};

// One exact-rigid state per body: centroid x_b, rotation R_b, the reference
// inertia I_ref, total mass, and the vertex count. Plain arrays rather than
// Vec3f / Mat3x3f members so the device array's layout is trivial and identical
// under every compiler that builds this header.
//
// The centroid is stored RELATIVE to an anchor vertex of the body, never as an
// absolute coordinate. Summing N absolute coordinates rounds at eps*|x| per term
// before the divide, and the reconstruction then stamps that error onto every
// vertex of the body. Anchoring keeps every sum and difference at body-extent
// scale, so the fit is as accurate far from the origin as it is near it. The
// anchor is a plain component triple rather than a Vec3f member, so the device
// array's layout stays trivial.
struct PdrdRigidState {
    float anchor[3];  // anchor vertex position
    float x[3];     // current centroid, RELATIVE to anchor
    float R[9];     // best-fit rotation, column-major (Mat3x3f data layout)
    float Iref[9];  // reference inertia, column-major
    float mass_total;
    unsigned N;
};

// ---------------------------------------------------------------- quaternions

// Quaternion (x, y, z, w) to a column-major 3x3 rotation matrix.
[[seam::host_device_fn]] inline void
pdrd_quat_to_mat(const float *q,
                     float *R) {
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
[[seam::host_device_fn]] inline void
pdrd_mat_to_quat(const float *R,
                     float *q) {
    // Column-major: R(row,col) = R[row + 3*col].
    const float r00 = R[0], r10 = R[1], r20 = R[2];
    const float r01 = R[3], r11 = R[4], r21 = R[5];
    const float r02 = R[6], r12 = R[7], r22 = R[8];
    const float tr = r00 + r11 + r22;
    if (tr > 0.0f) {
        float s = fmath::sqrt(tr + 1.0f) * 2.0f; // s = 4w
        q[3] = 0.25f * s;
        q[0] = fmath::div(r21 - r12, s);
        q[1] = fmath::div(r02 - r20, s);
        q[2] = fmath::div(r10 - r01, s);
    } else if (r00 > r11 && r00 > r22) {
        float s = fmath::sqrt(1.0f + r00 - r11 - r22) * 2.0f; // s = 4x
        q[3] = fmath::div(r21 - r12, s);
        q[0] = 0.25f * s;
        q[1] = fmath::div(r01 + r10, s);
        q[2] = fmath::div(r02 + r20, s);
    } else if (r11 > r22) {
        float s = fmath::sqrt(1.0f + r11 - r00 - r22) * 2.0f; // s = 4y
        q[3] = fmath::div(r02 - r20, s);
        q[0] = fmath::div(r01 + r10, s);
        q[1] = 0.25f * s;
        q[2] = fmath::div(r12 + r21, s);
    } else {
        float s = fmath::sqrt(1.0f + r22 - r00 - r11) * 2.0f; // s = 4z
        q[3] = fmath::div(r10 - r01, s);
        q[0] = fmath::div(r02 + r20, s);
        q[1] = fmath::div(r12 + r21, s);
        q[2] = 0.25f * s;
    }
}

// Best-fit rotation (polar factor) of a near-rotation 3x3 M (column-major), via
// the quaternion fixed-point iteration of Mueller et al., "A Robust Method to
// Extract the Rotational Part of Deformations" (2016): omega = (sum_c R_c x M_c)
// / (|sum_c R_c . M_c| + eps), q <- exp(omega) q. Pure float, no matrix inverse,
// so it is allocation-free and needs no scratch on either backend.
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
[[seam::host_device_fn]] inline void
pdrd_polar_quat(const float *M,
                    float *Rout) {
    float q[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    {
        // Gram-Schmidt on the columns m0,m1,m2 of M -> orthonormal seed R_s.
        float m0[3] = {M[0], M[1], M[2]};
        float m1[3] = {M[3], M[4], M[5]};
        float n0 = fmath::sqrt(m0[0] * m0[0] + m0[1] * m0[1] + m0[2] * m0[2]);
        if (n0 > 1e-12f) {
            float u0[3] = {fmath::div(m0[0], n0), fmath::div(m0[1], n0),
                           fmath::div(m0[2], n0)};
            float d = m1[0] * u0[0] + m1[1] * u0[1] + m1[2] * u0[2];
            float t1[3] = {m1[0] - d * u0[0], m1[1] - d * u0[1],
                           m1[2] - d * u0[2]};
            float n1 =
                fmath::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
            if (n1 > 1e-12f) {
                float u1[3] = {fmath::div(t1[0], n1), fmath::div(t1[1], n1),
                               fmath::div(t1[2], n1)};
                // u2 = u0 x u1 (right-handed -> proper rotation, det +1).
                float u2[3] = {u0[1] * u1[2] - u0[2] * u1[1],
                               u0[2] * u1[0] - u0[0] * u1[2],
                               u0[0] * u1[1] - u0[1] * u1[0]};
                float Rs[9] = {u0[0], u0[1], u0[2], u1[0], u1[1],
                               u1[2], u2[0], u2[1], u2[2]};
                pdrd_mat_to_quat(Rs, q);
            }
        }
    }
    for (unsigned iter = 0; iter < 20u; ++iter) {
        float R[9];
        pdrd_quat_to_mat(q, R);
        float on0 = 0.0f, on1 = 0.0f, on2 = 0.0f, od = 0.0f;
        for (unsigned c = 0; c < 3u; ++c) {
            float r0 = R[3 * c], r1 = R[3 * c + 1], r2 = R[3 * c + 2];
            float a0 = M[3 * c], a1 = M[3 * c + 1], a2 = M[3 * c + 2];
            on0 += r1 * a2 - r2 * a1;
            on1 += r2 * a0 - r0 * a2;
            on2 += r0 * a1 - r1 * a0;
            od += r0 * a0 + r1 * a1 + r2 * a2;
        }
        float denom = fmath::abs(od) + 1e-9f;
        float w0 = fmath::div(on0, denom), w1 = fmath::div(on1, denom),
              w2 = fmath::div(on2, denom);
        float ang = fmath::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
        if (ang < 1e-9f) {
            break;
        }
        // A polar-decomposition increment, so `ang` is small and bounded.
        float s = fmath::div(fmath::sin_bounded(0.5f * ang), ang);
        float dq[4] = {s * w0, s * w1, s * w2, fmath::cos_bounded(0.5f * ang)};
        // q <- dq * q  (quaternion product), then renormalize.
        float nx = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
        float ny = dq[3] * q[1] - dq[0] * q[2] + dq[1] * q[3] + dq[2] * q[0];
        float nz = dq[3] * q[2] + dq[0] * q[1] - dq[1] * q[0] + dq[2] * q[3];
        float nw = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];
        float inv = fmath::div(1.0f, fmath::sqrt(nx * nx + ny * ny + nz * nz +
                                         nw * nw));
        q[0] = nx * inv;
        q[1] = ny * inv;
        q[2] = nz * inv;
        q[3] = nw * inv;
    }
    pdrd_quat_to_mat(q, Rout);
}

// ------------------------------------------------------------------ SO(3) math

// 3x3 skew-symmetric matrix of a vector: skew(v) w = v x w.
[[seam::host_device_fn]] inline Mat3x3f
pdrd_skew(const Vec3f &v) {
    Mat3x3f S;
    S(0, 0) = 0.0f;   S(0, 1) = -v[2]; S(0, 2) = v[1];
    S(1, 0) = v[2];   S(1, 1) = 0.0f;  S(1, 2) = -v[0];
    S(2, 0) = -v[1];  S(2, 1) = v[0];  S(2, 2) = 0.0f;
    return S;
}

// Inverse of skew: the axial vector of the antisymmetric part of A.
[[seam::host_device_fn]] inline Vec3f
pdrd_skew_inv(const Mat3x3f &A) {
    return Vec3f(0.5f * (A(2, 1) - A(1, 2)), 0.5f * (A(0, 2) - A(2, 0)),
                 0.5f * (A(1, 0) - A(0, 1)));
}

// Exponential map so(3) -> SO(3) (Rodrigues), numerically safe near zero.
[[seam::host_device_fn]] inline Mat3x3f
pdrd_exp_so3(const Vec3f &theta) {
    float a2 = theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2];
    float a = fmath::sqrt(a2);
    Mat3x3f K = pdrd_skew(theta);
    Mat3x3f I = Mat3x3f::Identity();
    // sinc(a) = sin(a)/a, (1-cos a)/a^2, with Taylor fallback near 0.
    float s, c;
    if (a < 1e-5f) {
        s = 1.0f - a2 / 6.0f;          // sin(a)/a
        c = 0.5f - a2 / 24.0f;         // (1 - cos a)/a^2
    } else {
        // A rotation increment of one Newton iteration, so `a` is small and
        // bounded; the result is re-orthogonalized by the rigid fit on the
        // next iteration, which is far coarser than the intrinsics' error.
        s = fmath::div(fmath::sin_bounded(a), a);
        c = fmath::div(1.0f - fmath::cos_bounded(a), a2);
    }
    return I + s * K + c * (K * K);
}

// -------------------------------------------------------- reduced kinematics

// The rigid kinematic Jacobian applied to one body step:
//   J_k = [ I3 | -skew(p_k) ],  dx_k = dx_b - p_k x dtheta_b,
// with p_k = R_b ybar_k the rotated rest-centered vertex.
[[seam::host_device_fn]] inline Vec3f
pdrd_prolong(const Vec3f &body_translation,
                 const Vec3f &body_rotation,
                 const Vec3f &rotated_rest) {
    return body_translation - rotated_rest.cross(body_rotation);
}

// The transpose of the same Jacobian applied to one per-vertex force:
//   J_k^T y = [ y ; -skew(p_k)^T y ] = [ y ; p_k x y ].
// The translation half is `y` itself, so only the torque half needs a helper.
[[seam::host_device_fn]] inline Vec3f
pdrd_restrict_torque(const Vec3f &rotated_rest,
                         const Vec3f &force) {
    return rotated_rest.cross(force);
}

// The per-body joint projector Pi applied to a reduced rotation triple. For a
// hinge, Pi = blockdiag(0, a a^T): the caller zeroes the translation and this
// keeps only the spin about the axle. Restricting the reduced search direction
// (and the rhs, and the preconditioned residual) this way is what stops the
// linear solve from ever moving a hinged body off its axle.
[[seam::host_device_fn]] inline Vec3f
pdrd_project_hinge(const Vec3f &body_rotation,
                       const Vec3f &axis) {
    const float spin = body_rotation[0] * axis[0] +
                       body_rotation[1] * axis[1] +
                       body_rotation[2] * axis[2];
    return Vec3f(spin * axis[0], spin * axis[1], spin * axis[2]);
}

// The reference inertia I_ref = m (tr(Sbar) I - Sbar), with Sbar the rest Gram
// sum_k ybar_k ybar_k^T and m the (mass-scaled) per-vertex mass. Reducing the
// per-vertex inertia through J is block diagonal precisely because the
// rest-centered vertices satisfy sum_k m_k ybar_k = 0, so no cross term needs
// forming here.
[[seam::host_device_fn]] inline Mat3x3f
pdrd_reference_inertia(const Mat3x3f &rest_gram,
                           float mass_per_vertex) {
    const float trace = rest_gram(0, 0) + rest_gram(1, 1) + rest_gram(2, 2);
    Mat3x3f inertia;
    for (unsigned j = 0; j < 3u; ++j) {
        for (unsigned i = 0; i < 3u; ++i) {
            inertia(int(i), int(j)) =
                mass_per_vertex *
                ((i == j ? trace : 0.0f) - rest_gram(int(i), int(j)));
        }
    }
    return inertia;
}

// ------------------------------------------------- reduced 6x6 self-block math

// One vertex's contribution to the reduced self-block, J^T H J, written (not
// accumulated) into 36 row-major floats. The caller owns the accumulation, which
// is where the two backends legitimately differ: CUDA adds these into a
// threadgroup accumulator with atomicAdd, and so does Metal, but the spelling of
// a threadgroup atomic is not shareable and the arithmetic here is.
[[seam::host_device_fn]] inline void
pdrd_sandwich(const Vec3f &rotated_rest,
                  const Mat3x3f &hessian,
                  float *block) {
    float J[3][6];
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned c = 0; c < 6u; ++c) {
            J[i][c] = 0.0f;
        }
    }
    for (unsigned i = 0; i < 3u; ++i) {
        J[i][i] = 1.0f;
    }
    // rotation columns = -skew(p).
    J[0][4] = rotated_rest[2];  J[0][5] = -rotated_rest[1];
    J[1][3] = -rotated_rest[2]; J[1][5] = rotated_rest[0];
    J[2][3] = rotated_rest[1];  J[2][4] = -rotated_rest[0];
    for (unsigned a = 0; a < 6u; ++a) {
        for (unsigned b = 0; b < 6u; ++b) {
            float s = 0.0f;
            for (unsigned i = 0; i < 3u; ++i) {
                for (unsigned k = 0; k < 3u; ++k) {
                    s += J[i][a] * hessian(int(i), int(k)) * J[k][b];
                }
            }
            block[a * 6u + b] = s;
        }
    }
}

// The analytic rigid inertia half of the reduced self-block, written into 36
// row-major floats: blockdiag(m_total I3, R I_ref R^T) / dt^2. The mass enters
// here analytically so it is not double counted from the assembled matrix, whose
// only remaining contribution on a body vertex is contact.
[[seam::host_device_fn]] inline void
pdrd_inertia_block(float mass_total,
                       const Mat3x3f &world_inertia,
                       float inverse_dt_squared, float *block) {
    for (unsigned i = 0; i < PDRD_BLOCK_FLOATS; ++i) {
        block[i] = 0.0f;
    }
    for (unsigned i = 0; i < 3u; ++i) {
        block[i * 6u + i] = mass_total * inverse_dt_squared;
    }
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            block[(3u + i) * 6u + (3u + j)] =
                world_inertia(int(i), int(j)) * inverse_dt_squared;
        }
    }
}

// z = G^T (G r) for one body, with G the fp32 lower-triangular inverse factor of
// the reduced 6x6 (so the block inverse is G^T G, symmetric and positive by
// construction whatever the factor's conditioning). Both halves run in one
// thread: 36 multiply-adds is far below the cost of any cross-thread staging,
// and a single thread keeps the summation order identical on both backends.
[[seam::host_device_fn]] inline void
pdrd_precond_apply(const float *factor,
                       const float *residual,
                       float *out) {
    float gr[6];
    for (unsigned p = 0; p < 6u; ++p) {
        float acc = 0.0f;
        for (unsigned j = 0; j <= p; ++j) {
            acc += factor[p * 6u + j] * residual[j];
        }
        gr[p] = acc;
    }
    for (unsigned p = 0; p < 6u; ++p) {
        float acc = 0.0f;
        for (unsigned i = p; i < 6u; ++i) {
            acc += factor[i * 6u + p] * gr[i];
        }
        out[p] = acc;
    }
}

// ------------------------------------------------------------- entry points
//
// The two element-wise rows of the committed-rotation path, one thread per
// body. The physics above is what they compose; what they add is the indexing,
// which is the entry point's business rather than a helper's.

// Copy each body's fitted rotation into a flat [9 * nb] buffer. This is what
// seeds the persistent committed rotation from the absolute fit, once per body
// lifetime, on the first frame after an init or a scene load.
[[seam::entry(b)]]
[[seam::device_fn]] inline void
pdrd_copy_state_rotation_row(const PdrdRigidState *state,
                             float *rotation, unsigned b) {
    // The element is copied into thread space before its members are read: a
    // `[[seam::device]]` base pointer yields a device-space lvalue, which one of
    // the three backends cannot bind where a body names a member.
    const PdrdRigidState body = state[b];
    for (unsigned e = 0; e < 9u; ++e) {
        rotation[9u * b + e] = body.R[e];
    }
}

// running[b] <- exp(scale * dtheta_b) * running[b], the rotation increment this
// Newton iteration actually took, `scale` being the fraction of the reduced step
// the line search accepted. Carrying it across frames is what keeps the
// anchored rigidify target from re-fitting, and accumulating drift from, the
// contact-sheared iterate.
[[seam::entry(b)]]
[[seam::device_fn]] inline void
pdrd_compose_running_rotation_row(float *running,
                                  const float *dtheta,
                                  float scale, unsigned b) {
    const Vec3f increment(scale * dtheta[3u * b + 0u],
                          scale * dtheta[3u * b + 1u],
                          scale * dtheta[3u * b + 2u]);
    const Mat3x3f delta = pdrd_exp_so3(increment);
    Mat3x3f previous;
    for (unsigned e = 0; e < 9u; ++e) {
        previous.data()[e] = running[9u * b + e];
    }
    const Mat3x3f composed = delta * previous;
    for (unsigned e = 0; e < 9u; ++e) {
        running[9u * b + e] = composed.data()[e];
    }
}

// ------------------------------------------------- the reduced-space transfer
//
// `P` maps the reduced vector (cloth xyz blocks, then six DOFs per body) onto
// the per-vertex space. A cloth vertex is a copy; a body vertex goes through the
// rigid Jacobian. The two rows below are `P` and `P^T`.

// x = P u. Cloth entries copy, body entries apply J = [I | -skew(p)].
[[seam::entry(v)]]
[[seam::device_fn]] inline void pdrd_prolong_row(
    const unsigned *vertex_body,
    const unsigned *cloth_offset,
    const Vec3f *rotated_rest,
    const float *reduced,
    float *full, unsigned body_base, unsigned v) {
    const unsigned b = vertex_body[v];
    if (b == PDRD_CLOTH_MARKER) {
        const unsigned o = cloth_offset[v];
        full[3u * v + 0u] = reduced[o + 0u];
        full[3u * v + 1u] = reduced[o + 1u];
        full[3u * v + 2u] = reduced[o + 2u];
        return;
    }
    const unsigned base = body_base + 6u * b;
    const Vec3f p = rotated_rest[v];
    const Vec3f body_translation(reduced[base + 0u], reduced[base + 1u],
                                 reduced[base + 2u]);
    const Vec3f body_rotation(reduced[base + 3u], reduced[base + 4u],
                              reduced[base + 5u]);
    const Vec3f dx = pdrd_prolong(body_translation, body_rotation, p);
    full[3u * v + 0u] = dx[0];
    full[3u * v + 1u] = dx[1];
    full[3u * v + 2u] = dx[2];
}

// u = P^T y. Cloth entries are disjoint and written directly; a body's six DOFs
// are accumulated from every vertex it owns, so `reduced` must arrive zeroed on
// the body region.
[[seam::entry(v)]]
[[seam::device_fn]] inline void pdrd_restrict_row(
    const unsigned *vertex_body,
    const unsigned *cloth_offset,
    const Vec3f *rotated_rest,
    const float *full,
    float *cloth_out,
    compute::atomic_float_t *reduced, unsigned body_base,
    unsigned v) {
    const Vec3f y(full[3u * v + 0u], full[3u * v + 1u], full[3u * v + 2u]);
    const unsigned b = vertex_body[v];
    if (b == PDRD_CLOTH_MARKER) {
        // THE SAME ALLOCATION AS `reduced`, BOUND A SECOND TIME AS PLAIN
        // FLOATS. Cloth offsets are disjoint, so those words are STORED rather
        // than accumulated, and the seam has no float atomic store: a value
        // written once by one thread does not need one.
        const unsigned o = cloth_offset[v];
        cloth_out[o + 0u] = y[0];
        cloth_out[o + 1u] = y[1];
        cloth_out[o + 2u] = y[2];
        return;
    }
    const unsigned base = body_base + 6u * b;
    const Vec3f p = rotated_rest[v];
    const Vec3f torque = pdrd_restrict_torque(p, y);
    compute::atomic_add(reduced + base + 0u, y[0]);
    compute::atomic_add(reduced + base + 1u, y[1]);
    compute::atomic_add(reduced + base + 2u, y[2]);
    compute::atomic_add(reduced + base + 3u, torque[0]);
    compute::atomic_add(reduced + base + 4u, torque[1]);
    compute::atomic_add(reduced + base + 5u, torque[2]);
}

// The reduced image of an INITIAL GUESS, which is deliberately NOT `P^T`. The
// force restriction accumulates `sum_v J_v^T y_v` on a body block, which is
// right for a force and wrong for a guess: a body's six DOFs are seeded by the
// caller, and only the cloth rows are copied here.
[[seam::entry(v)]]
[[seam::device_fn]] inline void pdrd_seed_restrict_row(
    const unsigned *vertex_body,
    const unsigned *cloth_offset,
    const float *full,
    float *reduced, unsigned v) {
    if (vertex_body[v] != PDRD_CLOTH_MARKER) {
        return;
    }
    const unsigned o = cloth_offset[v];
    reduced[o + 0u] = full[3u * v + 0u];
    reduced[o + 1u] = full[3u * v + 1u];
    reduced[o + 2u] = full[3u * v + 2u];
}

// Copy only the cloth coordinates of a full-space vector back into a reduced
// one. This is the coordinate image of the deformable lock projector: cloth
// rows are identity-mapped by `P`, while a body's coordinates stay in their
// six-DOF representation and are handled by the body projector.
[[seam::entry(v)]]
[[seam::device_fn]] inline void pdrd_copy_projected_cloth_row(
    const unsigned *vertex_body,
    const unsigned *cloth_offset,
    const float *full,
    float *reduced, unsigned v) {
    if (vertex_body[v] != PDRD_CLOTH_MARKER) {
        return;
    }
    const unsigned o = cloth_offset[v];
    reduced[o + 0u] = full[3u * v + 0u];
    reduced[o + 1u] = full[3u * v + 1u];
    reduced[o + 2u] = full[3u * v + 2u];
}

// The affine particular correction for a locked free body. The drift holds
// `sum(m B (x - x_initial))`, so dividing by the group's total mass gives the
// translation the Newton correction has to remove. A HINGE already has no
// translation DOF, and its compatibility is checked on the host before this
// runs, so its zero translation is left alone.
[[seam::entry(b)]]
[[seam::device_fn]] inline void pdrd_translation_lock_particular_row(
    const unsigned *body_lock,
    const unsigned *joint_mode,
    const TranslationLock *locks,
    const float *drift,
    float *reduced, unsigned body_base, unsigned b) {
    const unsigned li = body_lock[b];
    if (li == PDRD_CLOTH_MARKER || joint_mode[b] == PDRD_JOINT_HINGE) {
        return;
    }
    const TranslationLock lock = locks[li];
    const float inverse_mass = 1.0f / lock.total_mass;
    const unsigned base = body_base + 6u * b;
    // Three floats per group: the aggregate lock accumulates the drift with
    // float atomics, so this reads the layout it is written in.
    reduced[base + 0u] = drift[3u * li + 0u] * inverse_mass;
    reduced[base + 1u] = drift[3u * li + 1u] * inverse_mass;
    reduced[base + 2u] = drift[3u * li + 2u] * inverse_mass;
}

// The per-body rotation DOFs of the reduced solution, lifted out for the
// integrator that composes them onto the running rotation.
[[seam::entry(b)]]
[[seam::device_fn]] inline void pdrd_extract_body_rotation_row(
    const float *reduced,
    float *rotation_out, unsigned body_base, unsigned b) {
    const unsigned base = body_base + 6u * b + 3u;
    rotation_out[3u * b + 0u] = reduced[base + 0u];
    rotation_out[3u * b + 1u] = reduced[base + 1u];
    rotation_out[3u * b + 2u] = reduced[base + 2u];
}

// The body's anchor position, read from the state the fit stored. Kept beside
// the state it reads rather than in the shim, so the rows below and the host
// both spell it once.
[[seam::host_device_fn]] inline Vec3f
pdrd_rigid_anchor(const PdrdRigidState &s) {
    return Vec3f(s.anchor[0], s.anchor[1], s.anchor[2]);
}

// ------------------------------------------- reconstruction, per PDRD vertex
//
// BOTH ROWS BELOW ARE DISPATCHED FLAT over `pdrd_vert_list`, not one group per
// body. The kernels they replace were one BLOCK per body striding over that
// body's vertices, with no shared memory and no barrier between them, so the
// grouping carried no information: a thread's body is recoverable from its own
// vertex, and the rest-centered array is parallel to the vertex list.

// p_v = R_b ybar_k, the rotated rest vector the rigid Jacobian is built from.
[[seam::entry(j)]]
[[seam::device_fn]] inline void pdrd_scatter_rotated_rest_row(
    const unsigned *vert_list,
    const VertexProp *prop,
    const PdrdRigidState *state,
    const Vec3f *rest_centered,
    Vec3f *rotated_rest, unsigned j) {
    const unsigned v = vert_list[j];
    const unsigned body = prop[v].pdrd_body_index;
    if (body == 0u) {
        return;
    }
    const PdrdRigidState s = state[body - 1u];
    if (s.N == 0u) {
        return;
    }
    Mat3x3f rotation;
    for (unsigned e = 0; e < 9u; ++e) {
        rotation.data()[e] = s.R[e];
    }
    const Vec3f rest = rest_centered[j];
    rotated_rest[v] = rotation * rest;
}

// ------------------------------------------------------- the preconditioner
//
// A body's six reduced DOFs get the analytic rigid 6x6 block; a cloth vertex
// keeps its 3x3 block-Jacobi. The two rows are dispatched over different
// things, which is why they are two.

// z_body = G^-1 r_body, through the lower-triangular factor the host built.
[[seam::entry(b)]]
[[seam::device_fn]] inline void pdrd_precond_body_row(
    const float *factor,
    const float *residual,
    float *out, unsigned body_base, unsigned b) {
    float r[PDRD_BODY_DOFS];
    float z[PDRD_BODY_DOFS];
    const unsigned base = body_base + PDRD_BODY_DOFS * b;
    for (unsigned p = 0; p < PDRD_BODY_DOFS; ++p) {
        r[p] = residual[base + p];
    }
    // The factor is a base pointer offset by the body: a record addresses one
    // allocation per field, and the per-body slab is a run inside it.
    float block[PDRD_BLOCK_FLOATS];
    for (unsigned e = 0; e < PDRD_BLOCK_FLOATS; ++e) {
        block[e] = factor[PDRD_BLOCK_FLOATS * b + e];
    }
    pdrd_precond_apply(block, r, z);
    for (unsigned p = 0; p < PDRD_BODY_DOFS; ++p) {
        out[base + p] = z[p];
    }
}

// z_cloth = inv_diag * r, the 3x3 block-Jacobi a cloth vertex keeps. A body's
// vertex is skipped: its DOFs live in the six-vector the row above handles.
[[seam::entry(v)]]
[[seam::device_fn]] inline void pdrd_precond_cloth_row(
    const unsigned *vertex_body,
    const unsigned *cloth_offset,
    const Mat3x3f *inverse_diagonal,
    const float *residual,
    float *out, unsigned v) {
    if (vertex_body[v] != PDRD_CLOTH_MARKER) {
        return;
    }
    const unsigned o = cloth_offset[v];
    const Vec3f r(residual[o + 0u], residual[o + 1u], residual[o + 2u]);
    const Mat3x3f inverse = inverse_diagonal[v];
    const Vec3f z = inverse * r;
    out[o + 0u] = z[0];
    out[o + 1u] = z[1];
    out[o + 2u] = z[2];
}

// ------------------------------------- the anchored rigidify, in two passes
//
// THESE TWO REPLACE ONE COOPERATIVE KERNEL, and the split is what removes the
// need for a threadgroup reduction: the first row accumulates each body's
// centroid and the second reads it back, with KERNEL COMPLETION between them
// as the barrier. That is a full memory barrier on every backend, where
// `compute::threadgroup_barrier()` orders threadgroup memory only.
//
// Both are dispatched flat over `pdrd_vert_list`. A thread finds its body from
// its own vertex, and the body's anchor from that body's FIRST vertex, which is
// the same anchoring the fit uses: the sum is built from (x - anchor)
// differences so neither it nor the written position carries |x|-proportional
// rounding.

// The body's centroid sum, relative to its anchor vertex.
[[seam::entry(j)]]
[[seam::device_fn]] inline void pdrd_rigidify_centroid_row(
    const unsigned *vert_list,
    const VertexProp *prop,
    const PdrdBodyProp *body_prop,
    const Vec3f *positions,
    compute::atomic_float_t *centroid, unsigned j) {
    const unsigned v = vert_list[j];
    const unsigned body = prop[v].pdrd_body_index;
    if (body == 0u) {
        return;
    }
    const unsigned b = body - 1u;
    const PdrdBodyProp bp = body_prop[b];
    if (bp.vertex_count == 0u) {
        return;
    }
    const Vec3f anchor = positions[vert_list[bp.vertex_start]];
    const Vec3f here = positions[v];
    const Vec3f d = (here - anchor).cast<float>();
    compute::atomic_add(centroid + 3u * b + 0u, d[0]);
    compute::atomic_add(centroid + 3u * b + 1u, d[1]);
    compute::atomic_add(centroid + 3u * b + 2u, d[2]);
}

// x_v = anchor + (centroid + R_run ybar_k). Only the ROTATION is anchored: the
// centroid is taken from the iterate, so translation stays exact.
[[seam::entry(j)]]
[[seam::device_fn]] inline void pdrd_rigidify_write_row(
    const unsigned *vert_list,
    const VertexProp *prop,
    const PdrdBodyProp *body_prop,
    const Vec3f *positions,
    const float *centroid,
    const float *running_rotation,
    const Vec3f *rest_centered,
    Vec3f *out, unsigned j) {
    const unsigned v = vert_list[j];
    const unsigned body = prop[v].pdrd_body_index;
    if (body == 0u) {
        return;
    }
    const unsigned b = body - 1u;
    const PdrdBodyProp bp = body_prop[b];
    if (bp.vertex_count == 0u) {
        return;
    }
    const Vec3f anchor = positions[vert_list[bp.vertex_start]];
    const float inverse_count = 1.0f / float(bp.vertex_count);
    const Vec3f c(centroid[3u * b + 0u] * inverse_count,
                  centroid[3u * b + 1u] * inverse_count,
                  centroid[3u * b + 2u] * inverse_count);
    Mat3x3f rotation;
    for (unsigned e = 0; e < 9u; ++e) {
        rotation.data()[e] = running_rotation[9u * b + e];
    }
    const Vec3f rest = rest_centered[j];
    out[v] = anchor + (c + rotation * rest);
}

// --------------------------------------------- the best-fit rigid, in three
//
// THESE THREE REPLACE THE LAST COOPERATIVE FIT. The scratch is twelve floats
// per body in DEVICE memory, the centroid in [0, 3) and the cross-covariance in
// [3, 12), and kernel completion between the passes is the barrier.
//
// EVERY QUANTITY IS BUILT FROM (x - anchor) DIFFERENCES, which are body-extent
// sized. Subtracting a centroid accumulated from absolute coordinates instead
// would cancel the leading digits of two nearby large values, leaving rounding
// that grows with distance from the origin, and the difference feeds the
// cross-covariance whose polar factor IS the body rotation, so that noise would
// land on the orientation.

// Pass one: the body's centroid sum, relative to its anchor vertex.
[[seam::entry(j)]]
[[seam::device_fn]] inline void pdrd_fit_centroid_row(
    const unsigned *vert_list,
    const VertexProp *prop,
    const PdrdBodyProp *body_prop,
    const Vec3f *positions,
    compute::atomic_float_t *scratch, unsigned j) {
    const unsigned v = vert_list[j];
    const unsigned body = prop[v].pdrd_body_index;
    if (body == 0u) {
        return;
    }
    const unsigned b = body - 1u;
    const PdrdBodyProp bp = body_prop[b];
    if (bp.vertex_count == 0u) {
        return;
    }
    const Vec3f anchor = positions[vert_list[bp.vertex_start]];
    const Vec3f here = positions[v];
    const Vec3f d = (here - anchor).cast<float>();
    compute::atomic_add(scratch + 12u * b + 0u, d[0]);
    compute::atomic_add(scratch + 12u * b + 1u, d[1]);
    compute::atomic_add(scratch + 12u * b + 2u, d[2]);
}

// Pass two: the cross-covariance sum(y ybar^T), y being the vertex relative to
// the centroid pass one built. The divide happens on READ rather than in a pass
// of its own, which is one dispatch fewer for the same arithmetic.
[[seam::entry(j)]]
[[seam::device_fn]] inline void pdrd_fit_covariance_row(
    const unsigned *vert_list,
    const VertexProp *prop,
    const PdrdBodyProp *body_prop,
    const Vec3f *positions,
    const Vec3f *rest_centered,
    const float *centroid,
    compute::atomic_float_t *scratch, unsigned j) {
    const unsigned v = vert_list[j];
    const unsigned body = prop[v].pdrd_body_index;
    if (body == 0u) {
        return;
    }
    const unsigned b = body - 1u;
    const PdrdBodyProp bp = body_prop[b];
    if (bp.vertex_count == 0u) {
        return;
    }
    const float inverse_count = 1.0f / float(bp.vertex_count);
    const Vec3f c(centroid[12u * b + 0u] * inverse_count,
                  centroid[12u * b + 1u] * inverse_count,
                  centroid[12u * b + 2u] * inverse_count);
    const Vec3f anchor = positions[vert_list[bp.vertex_start]];
    const Vec3f here = positions[v];
    const Vec3f y = (here - anchor).cast<float>() - c;
    const Vec3f rest = rest_centered[j];
    for (unsigned col = 0; col < 3u; ++col) {
        for (unsigned row = 0; row < 3u; ++row) {
            compute::atomic_add(scratch + 12u * b + 3u + row + 3u * col,
                                y[row] * rest[col]);
        }
    }
}

// Pass three, per BODY: the polar factor of the cross-covariance is the fitted
// rotation, and the rest of the state is arithmetic on the body's own props.
//
// A HINGE BODY TAKES THE SAME FREE BEST-FIT AS A FREE ONE. The joint is
// enforced in the reduced solve by the per-body DOF projector, which restricts
// the Newton direction to joint-admissible rigid motions, so the body never
// accrues translational or off-axle rotational velocity. Enforcing it here
// instead would make the rigidify target fight contact: snapping a
// contact-pushed body back onto its axle re-creates the penetration contact
// just resolved, the rigidify CCD then blocks it and the next step starts
// interpenetrating. The free fit keeps the target reachable.
[[seam::entry(b)]]
[[seam::device_fn]] inline void pdrd_fit_finish_row(
    const unsigned *vert_list,
    const PdrdBodyProp *body_prop,
    const Vec3f *positions,
    const float *scratch,
    PdrdRigidState *state, unsigned b) {
    const PdrdBodyProp bp = body_prop[b];
    PdrdRigidState s{};
    if (bp.vertex_count == 0u) {
        s.N = 0u;
        state[b] = s;
        return;
    }
    const float inverse_count = 1.0f / float(bp.vertex_count);
    const Vec3f anchor = positions[vert_list[bp.vertex_start]];
    float covariance[9];
    for (unsigned e = 0; e < 9u; ++e) {
        covariance[e] = scratch[12u * b + 3u + e];
    }
    s.anchor[0] = anchor[0];
    s.anchor[1] = anchor[1];
    s.anchor[2] = anchor[2];
    s.x[0] = scratch[12u * b + 0u] * inverse_count;
    s.x[1] = scratch[12u * b + 1u] * inverse_count;
    s.x[2] = scratch[12u * b + 2u] * inverse_count;
    pdrd_polar_quat(covariance, s.R);
    // Iref = m (tr(Sbar) I - Sbar). SMat is column-major, which is the order
    // PdrdRigidState::Iref declares, so the copy is a straight run.
    const Mat3x3f rest_gram = bp.rest_gram_inv.inverse();
    const Mat3x3f reference = pdrd_reference_inertia(rest_gram, bp.mass_per_vertex);
    for (unsigned e = 0; e < 9u; ++e) {
        s.Iref[e] = reference.data()[e];
    }
    s.mass_total = bp.mass_per_vertex * float(bp.vertex_count);
    s.N = bp.vertex_count;
    state[b] = s;
}

// ------------------------------------ the per-body reduced self-block, in two
//
// K_b = the analytic rigid inertia, blockdiag(m_total/dt^2 I3, R Iref R^T/dt^2),
// plus sum_k J_k^T (A(v,v) + B(v,v)) J_k over the body's own vertices. The
// inertia carries the mass analytically so it is not double counted, and the
// contact diagonal is read straight from the assembled matrix.
//
// TWO PASSES, and the order is load-bearing: the first WRITES the inertia into
// the block and the second ACCUMULATES the contact sandwich into it, so the
// first must complete before the second begins. Kernel completion is that
// barrier.

// Pass one, per BODY: the analytic inertia, or the identity for an empty body
// so its 6x6 stays invertible.
[[seam::entry(b)]]
[[seam::device_fn]] inline void pdrd_assemble_inertia_row(
    const PdrdRigidState *state,
    float *blocks, float dt, unsigned b) {
    const PdrdRigidState s = state[b];
    const unsigned base = PDRD_BLOCK_FLOATS * b;
    for (unsigned i = 0; i < PDRD_BLOCK_FLOATS; ++i) {
        blocks[base + i] = 0.0f;
    }
    if (s.N == 0u) {
        for (unsigned i = 0; i < PDRD_BODY_DOFS; ++i) {
            blocks[base + i * PDRD_BODY_DOFS + i] = 1.0f;
        }
        return;
    }
    Mat3x3f rotation;
    Mat3x3f reference;
    for (unsigned e = 0; e < 9u; ++e) {
        rotation.data()[e] = s.R[e];
        reference.data()[e] = s.Iref[e];
    }
    const Mat3x3f rotated = rotation * reference * rotation.transpose();
    float block[PDRD_BLOCK_FLOATS];
    pdrd_inertia_block(s.mass_total, rotated, 1.0f / (dt * dt), block);
    for (unsigned i = 0; i < PDRD_BLOCK_FLOATS; ++i) {
        blocks[base + i] = block[i];
    }
}

// Pass two, per PDRD VERTEX: the contact sandwich J^T (A + B) J, accumulated.
//
// THE CONTACT DIAGONAL IS READ FROM FLAT ARRAYS, which is what lets this be a
// neutral body at all: a `DynCSRMat` row owns its own columns and blocks
// through handles stored INSIDE an element, and a record addresses one
// allocation per field. The compacted `dyn_*` arrays and the fixed table's own
// flat data and offsets are the same matrix, reachable.
[[seam::entry(j)]]
[[seam::device_fn]] inline void pdrd_assemble_sandwich_row(
    const unsigned *vert_list,
    const VertexProp *prop,
    const PdrdRigidState *state,
    const Vec3f *rest_centered,
    const unsigned *dyn_index,
    const unsigned *dyn_offset,
    const float *dyn_value,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    compute::atomic_float_t *blocks, unsigned j) {
    const unsigned v = vert_list[j];
    const unsigned body = prop[v].pdrd_body_index;
    if (body == 0u) {
        return;
    }
    const unsigned b = body - 1u;
    const PdrdRigidState s = state[b];
    if (s.N == 0u) {
        return;
    }
    // The dynamic row's diagonal is SCANNED rather than indexed: that pattern
    // is built by insertion and promises nothing about where a row's diagonal
    // block sits.
    Mat3x3f contact = Mat3x3f::Zero();
    for (unsigned slot = dyn_offset[v]; slot < dyn_offset[v + 1u]; ++slot) {
        if (dyn_index[slot] != v) {
            continue;
        }
        for (unsigned k = 0; k < 9u; ++k) {
            contact.m[k] += dyn_value[9u * slot + k];
        }
    }
    const Mat3x3f fixed_block =
        fixed_csr_read(fixed_index, fixed_offset, fixed_value, row_count, v, v);
    for (unsigned k = 0; k < 9u; ++k) {
        contact.m[k] += fixed_block.m[k];
    }
    Mat3x3f rotation;
    for (unsigned e = 0; e < 9u; ++e) {
        rotation.data()[e] = s.R[e];
    }
    const Vec3f rest = rest_centered[j];
    const Vec3f rotated_rest = rotation * rest;
    float block[PDRD_BLOCK_FLOATS];
    pdrd_sandwich(rotated_rest, contact, block);
    const unsigned base = PDRD_BLOCK_FLOATS * b;
    for (unsigned i = 0; i < PDRD_BLOCK_FLOATS; ++i) {
        compute::atomic_add(blocks + base + i, block[i]);
    }
}
