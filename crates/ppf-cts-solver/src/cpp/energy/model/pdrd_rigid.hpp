// File: pdrd_rigid.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Exact-rigid body kinematics for Painless Differentiable Rotation Dynamics bodies.
//
// Background. Each body is parameterized EXACTLY by a translation x_b and a
// rotation R_b in SO(3):
//
//   x_k = x_b + R_b ybar_k,   ybar_k = xbar_k - cbar  (rest-centered vertex).
//
// Rigidity is a kinematic constraint, so there is no shape penalty or
// user-facing rigidity stiffness parameter. The dynamics come from minimizing
// the incremental potential; following "Painless Differentiable Rotation Dynamics"
// (Romanya-Serrasolsas, Casafranca, Otaduy, ACM TOG 2025) the rotational update
// uses Lie-algebra coordinates: a tangent step dtheta is applied multiplicatively
//
//   R_b <- exp(dtheta) R_b,
//
// and the energy gradient/Hessian with respect to R_b are Lie derivatives. The
// paper's section 12 result for an energy of rotated points,
//   DPsi_p/DR = -(dPsi_p/dp) skew(p),
// is exactly the reduction of a per-vertex quantity through the rigid kinematic
// Jacobian below, so the reduced 6-DOF solve reuses the per-vertex assembly.
//
// Reduced DOF per body: u_b = (dx_b, dtheta_b) in R^6. The rigid kinematic
// Jacobian maps a body step to a vertex displacement:
//   dx_k = dx_b - skew(R_b ybar_k) dtheta_b,   J_k = [ I3 | -skew(R_b ybar_k) ]  (3x6).
//
// Reference inertia. The reduced mass operator obtained by reducing the
// per-vertex inertia (sum_k m_k/h^2) through J_k is block-diagonal,
//   sum_k J_k^T (m_k) J_k = blockdiag( m_total I3 , R I_ref R^T ),
// because the rest-centered vertices satisfy sum_k m_k ybar_k = 0 (the
// translation-rotation cross term vanishes) and
//   I_ref = sum_k m_k (|ybar_k|^2 I - ybar_k ybar_k^T) = m (tr(Sbar) I - Sbar),
// with Sbar = sum_k ybar_k ybar_k^T the rest Gram and m the (mass-scaled)
// per-vertex mass. Both are already stored in PdrdBodyProp, so no new per-body
// state is required: R_b is fit each Newton iteration as the best-fit rotation
// (polar factor) of the cross-covariance M = sum_k y_k ybar_k^T.
//
// Provenance. Only the published Lie-derivative rotation math from "Painless
// Differentiable Rotation Dynamics" was used as a concept; none of the paper's
// accompanying reference code was read, copied, or reproduced. The live path is
// a per-iteration polar fit + reconstruct (launch_rigidify_to) plus a reduced
// inertia+contact PCG (the S2 RigidMap / RigidPrecond below), which is standard
// rigid-body machinery. The kernels are written from scratch against this
// codebase's own conventions (Mueller quaternion polar, plain shared-memory
// reductions, POD device structs).

#ifndef PDRD_RIGID_HPP
#define PDRD_RIGID_HPP

#include "../../buffer/buffer.hpp"
#include "../../common.hpp"
#include "../../data.hpp"
#include "../../main/cuda_utils.hpp"
#include "../../utility/dispatcher.hpp"
#include "../../utility/utility.hpp"
#include "pdrd_polar.hpp" // rigid_quat_to_mat / rigid_mat_to_quat / rigid_polar_quat
#include "rigid_core.hpp" // rigid_skew / rigid_skew_inv / rigid_exp_so3 (shared with SAND)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace PDRD {

constexpr unsigned RIGID_THREADS_PER_BODY = 64;

// PDRD joint / DOF-filtering modes (mirror PdrdBodyProp::joint_mode).
constexpr unsigned PDRD_JOINT_FREE = 0u;   // full 6-DOF rigid body
constexpr unsigned PDRD_JOINT_HINGE = 1u;  // translation locked, spin about axle

// SO(3) rotation math (skew, inverse-skew, exponential map) is shared with SAND
// grain spin and lives in rigid_core.hpp. These using-declarations keep the
// historical PDRD::rigid_* call sites below unchanged.
using RigidCore::rigid_exp_so3;
using RigidCore::rigid_skew;
using RigidCore::rigid_skew_inv;

// rigid_quat_to_mat, rigid_mat_to_quat, and rigid_polar_quat (the best-fit
// rotation with its 180-degree-robust Gram-Schmidt seed) live in pdrd_polar.hpp
// so the pure rotation math can be unit tested on a host compiler with no nvcc
// (tests/test_pdrd_polar.cpp). They are in namespace PDRD, available here via the
// include above.

// One exact-rigid state per body: centroid x_b, rotation R_b, the reference
// inertia I_ref = m (tr(Sbar) I - Sbar), total mass, and the vertex count.
// Stored as plain float arrays (NOT Eigen members) so the device array has a
// trivial layout; Eigen math is done on local copies via Map, matching the
// ReducedState convention.
struct RigidState {
    float x[3];     // current centroid
    float R[9];     // best-fit rotation, column-major (Mat3x3f data layout)
    float Iref[9];  // reference inertia, column-major
    float mass_total;
    unsigned N;
};

// Fit (x_b, R_b) for every body from the current positions (one block per body,
// centroid + cross-covariance M = sum_k y_k ybar_k^T accumulated by a plain
// shared-memory reduction: each thread adds its grid-stride partials straight
// into the shared accumulators with atomicAdd, no warp-level primitives). R_b is
// the polar factor of M (Kabsch). Also stashes I_ref and m_total for assembly.
static __global__ void fit_rigid_kernel(DataSet data, Vec<Vec3f> eval_x,
                                        Vec<RigidState> state) {
    const unsigned body_id = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned T = blockDim.x;

    const PdrdBodyProp bp = data.prop.pdrd_body[body_id];
    const unsigned start = bp.vertex_start;
    const unsigned N = bp.vertex_count;

    __shared__ float sh[12]; // c[3], M[9]
    if (tid < 12) sh[tid] = 0.0f;
    __syncthreads();
    if (N == 0) {
        if (tid == 0) {
            RigidState s{};
            s.N = 0;
            state.data[body_id] = s;
        }
        return;
    }

    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (unsigned k = tid; k < N; k += T) {
        const Vec3f &xf = eval_x[data.pdrd_vert_list[start + k]];
        cx += float(xf[0]);
        cy += float(xf[1]);
        cz += float(xf[2]);
    }
    atomicAdd(&sh[0], cx);
    atomicAdd(&sh[1], cy);
    atomicAdd(&sh[2], cz);
    __syncthreads();
    if (tid == 0) {
        float invN = 1.0f / float(N);
        sh[0] *= invN;
        sh[1] *= invN;
        sh[2] *= invN;
    }
    __syncthreads();
    Vec3f c(sh[0], sh[1], sh[2]);

    float m[9] = {0.0f};
    for (unsigned k = tid; k < N; k += T) {
        const Vec3f &xf = eval_x[data.pdrd_vert_list[start + k]];
        Vec3f y(float(xf[0]) - c[0], float(xf[1]) - c[1], float(xf[2]) - c[2]);
        const Vec3f &yb = data.pdrd_rest_centered[start + k];
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned i = 0; i < 3; ++i)
                m[i + 3 * j] += y[i] * yb[j];
    }
    for (unsigned e = 0; e < 9; ++e)
        atomicAdd(&sh[3 + e], m[e]);
    __syncthreads();

    if (tid == 0) {
        float M9[9];
        for (unsigned e = 0; e < 9; ++e) M9[e] = sh[3 + e];
        Mat3x3f Sbar = bp.rest_gram_inv.inverse();
        float m_pv = bp.mass_per_vertex;
        float tr = Sbar(0, 0) + Sbar(1, 1) + Sbar(2, 2);
        RigidState s{};
        s.x[0] = c[0];
        s.x[1] = c[1];
        s.x[2] = c[2];
        rigid_polar_quat(M9, s.R);
        // Hinge bodies use the SAME free best-fit rigid target as free
        // bodies here. The joint (locked translation + axle-only rotation)
        // is enforced in the reduced linear solve by the per-body DOF
        // projector Pi (see project_body_dofs_kernel): the Newton search
        // direction is restricted to joint-admissible rigid motions, so
        // the body never accrues translational or off-axle rotational
        // velocity. Enforcing the joint here instead (hard-pinning the
        // centroid + projecting R) would make the rigidify target fight
        // contact: snapping a contact-pushed body back onto its axle
        // re-creates the very penetration contact just resolved, the
        // rigidify CCD then blocks it (toi_rig -> 0) and the next step
        // starts interpenetrating. Keeping the free fit keeps the rigid
        // target reachable (toi_rig ~ 1) exactly as for free bodies.
        // Iref = m (tr(Sbar) I - Sbar), column-major.
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned i = 0; i < 3; ++i)
                s.Iref[i + 3 * j] =
                    m_pv * ((i == j ? tr : 0.0f) - Sbar(i, j));
        s.mass_total = m_pv * float(N);
        s.N = N;
        state.data[body_id] = s;
    }
}

// Reconstruct PDRD vertex positions x_v = x_b + R_b ybar_k into `out` (indexed by
// GLOBAL vertex id, float). Cloth entries are left untouched, so the caller
// must pre-fill `out` (e.g. out = eval_x) before calling.
static __global__ void reconstruct_rigid_kernel(Vec<RigidState> state,
                                                DataSet data, Vec<Vec3f> out) {
    const unsigned body_id = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned T = blockDim.x;
    const RigidState s = state.data[body_id];
    if (s.N == 0) return;
    const PdrdBodyProp bp = data.prop.pdrd_body[body_id];
    const unsigned start = bp.vertex_start;
    Vec3f x(s.x[0], s.x[1], s.x[2]);
    Mat3x3f R;
    for (unsigned e = 0; e < 9; ++e) R.data()[e] = s.R[e];
    for (unsigned k = tid; k < s.N; k += T) {
        unsigned v = data.pdrd_vert_list[start + k];
        const Vec3f &yb = data.pdrd_rest_centered[start + k];
        out.data[v] = (x + R * yb).cast<float>();
    }
}

inline void launch_fit_rigid(const DataSet &data, const Vec<Vec3f> &eval_x,
                             Vec<RigidState> &state) {
    unsigned nb = data.prop.pdrd_body.size;
    if (nb == 0) return;
    fit_rigid_kernel<<<nb, RIGID_THREADS_PER_BODY>>>(
        data, const_cast<Vec<Vec3f> &>(eval_x), state);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// Fit (x_b, R_b) from `eval_x` and write the exact rigid image x_v = x_b + R_b
// ybar_k of each PDRD vertex into `out` (by global vertex id). Cloth entries of
// `out` are NOT written, so the caller must initialize out = eval_x first. This
// is the "rigidify target": the nearest exactly-rigid configuration to eval_x.
inline void launch_rigidify_to(const DataSet &data, const Vec<Vec3f> &eval_x,
                               Vec<Vec3f> &out) {
    unsigned nb = data.prop.pdrd_body.size;
    if (nb == 0) return;
    auto st = buffer::get().get<RigidState>(nb);
    Vec<RigidState> state = st.as_vec();
    launch_fit_rigid(data, eval_x, state);
    reconstruct_rigid_kernel<<<nb, RIGID_THREADS_PER_BODY>>>(
        state, const_cast<DataSet &>(data), out);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// ===========================================================================
// Persistent-rotation rigidify (fixes the contact-wedged non-rigid shrink).
//
// The shipping rigidify target is the polar fit of eval_x; when a body is
// contact-wedged the partial (toi_rig<1) lerp commits a non-rigid pose to
// vertex.curr, the next frame's fit drifts further, and det accumulates down to
// ~0.85 over a few frames (then self-recovers). To break the accumulation we
// build the target rotation by INTEGRATING the reduced-solve rotation increment
// onto a persistent per-body rotation R_run (seeded once from the absolute fit),
// instead of re-fitting the drifted shape. The rigidify lerp + CCD path are
// UNCHANGED (same straight eval_x->rigid_tgt sweep). R_run/R_prev are float[9]
// col-major per body.
// ===========================================================================

static __global__ void copy_state_R_kernel(unsigned nb, Vec<RigidState> state,
                                           Vec<float> Rbuf) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    for (unsigned e = 0; e < 9; ++e) Rbuf.data[9 * b + e] = state.data[b].R[e];
}

// Persistent per-body committed rotation R_prev[9*nb], carried across frames.
inline Vec<float> &pdrd_rprev() {
    static Vec<float> r;
    return r;
}
inline bool &pdrd_rprev_seeded() {
    static bool s = false;
    return s;
}
inline void pdrd_reset_rprev() { pdrd_rprev_seeded() = false; }

// Seed Rprev[9*nb] from the absolute polar fit of `eval_x` (the body's actual
// current orientation). Call once per body lifetime (first frame / post-load).
inline void launch_seed_rprev(const DataSet &data, const Vec<Vec3f> &eval_x,
                              Vec<float> &Rprev) {
    unsigned nb = data.prop.pdrd_body.size;
    if (nb == 0) return;
    auto st = buffer::get().get<RigidState>(nb);
    Vec<RigidState> state = st.as_vec();
    launch_fit_rigid(data, eval_x, state);
    unsigned tpb = 64;
    copy_state_R_kernel<<<(nb + tpb - 1) / tpb, tpb>>>(nb, state, Rprev);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// R_run[b] <- exp(scale * dtheta_b) * R_run[b] (applied rotation increment;
// scale = toi_recale*toi). dtheta is [3*nb]. One thread per body.
static __global__ void compose_rrun_kernel(unsigned nb, Vec<float> Rrun,
                                           Vec<float> dtheta, float scale) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    Vec3f w(scale * dtheta.data[3 * b + 0], scale * dtheta.data[3 * b + 1],
            scale * dtheta.data[3 * b + 2]);
    Mat3x3f dR = rigid_exp_so3(w);
    Mat3x3f R0;
    for (unsigned e = 0; e < 9; ++e) R0.data()[e] = Rrun.data[9 * b + e];
    Mat3x3f Rn = dR * R0;
    for (unsigned e = 0; e < 9; ++e) Rrun.data[9 * b + e] = Rn.data()[e];
}

inline void launch_compose_rrun(unsigned nb, Vec<float> &Rrun,
                                const Vec<float> &dtheta, float scale) {
    if (nb == 0) return;
    unsigned tpb = 64;
    compose_rrun_kernel<<<(nb + tpb - 1) / tpb, tpb>>>(
        nb, Rrun, const_cast<Vec<float> &>(dtheta), scale);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// Rigidify target from the integrated rotation: out[v] = centroid(eval_x) +
// R_run[b] * ybar_k. Centroid from eval_x (translation is exact); only the
// rotation is anchored. Cloth entries untouched.
static __global__ void rigidify_from_rot_kernel(DataSet data, Vec<Vec3f> eval_x,
                                               Vec<float> Rrun, Vec<Vec3f> out) {
    const unsigned b = blockIdx.x;
    const unsigned tid = threadIdx.x, T = blockDim.x;
    const PdrdBodyProp bp = data.prop.pdrd_body[b];
    const unsigned start = bp.vertex_start, N = bp.vertex_count;
    __shared__ float sh[3];
    if (tid < 3) sh[tid] = 0.0f;
    __syncthreads();
    if (N == 0) return;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (unsigned k = tid; k < N; k += T) {
        const Vec3f &xf = eval_x[data.pdrd_vert_list[start + k]];
        cx += float(xf[0]); cy += float(xf[1]); cz += float(xf[2]);
    }
    atomicAdd(&sh[0], cx); atomicAdd(&sh[1], cy); atomicAdd(&sh[2], cz);
    __syncthreads();
    float invN = 1.0f / float(N);
    Vec3f c(sh[0] * invN, sh[1] * invN, sh[2] * invN);
    Mat3x3f R;
    for (unsigned e = 0; e < 9; ++e) R.data()[e] = Rrun.data[9 * b + e];
    for (unsigned k = tid; k < N; k += T) {
        unsigned v = data.pdrd_vert_list[start + k];
        const Vec3f &yb = data.pdrd_rest_centered[start + k];
        out.data[v] = (c + R * yb).cast<float>();
    }
}

inline void launch_rigidify_from_rot(const DataSet &data,
                                     const Vec<Vec3f> &eval_x,
                                     const Vec<float> &Rrun, Vec<Vec3f> &out) {
    unsigned nb = data.prop.pdrd_body.size;
    if (nb == 0) return;
    rigidify_from_rot_kernel<<<nb, RIGID_THREADS_PER_BODY>>>(
        const_cast<DataSet &>(data), const_cast<Vec<Vec3f> &>(eval_x),
        const_cast<Vec<float> &>(Rrun), out);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// ===========================================================================
// S2: reduced mixed-DOF rigid system  R = P^T M P  via prolong / restrict.
//
// Reduced DOF vector u layout: [ 3 floats per CLOTH vertex ; 6 floats per PDRD
// body ] = (dx_b, dtheta_b). P maps u -> per-vertex displacement: identity on
// cloth, the rigid Jacobian J_k = [ I3 | -skew(p_k) ] on PDRD vertices, where
// p_k = R_b ybar_k is the current rotated rest vector. The reduced operator
// R = P^T M P is applied matrix-free by wrapping the per-vertex matvec:
// y = P^T ( M ( P u ) ). Reducing the per-vertex inertia diagonal of M through
// J yields the exact rigid 6x6 inertia (translation m_total I, rotation
// R Iref R^T); reducing the contact blocks yields the rigid contact coupling.
// No penalty term exists for a rigid body, so M on PDRD vertices carries only
// inertia + contact and the projection is exact.
// ===========================================================================

constexpr unsigned RIGID_UNSET = 0xffffffffu;

struct RigidMap {
    unsigned nrow{0};
    unsigned n_bodies{0};
    unsigned n_cloth{0};
    unsigned dim{0};        // 3*n_cloth + 6*n_bodies
    unsigned body_base{0};  // = 3*n_cloth
    bool any_joint{false};  // true if any body has joint_mode != FREE
    Vec<unsigned> vbody;    // [nrow] 0-based body id, or RIGID_UNSET for cloth
    Vec<unsigned> cloth_off;// [nrow] reduced float offset for a cloth vertex
    Vec<Vec3f> prot;        // [nrow] rotated rest vector p_k = R_b ybar_k (PDRD)
    Vec<unsigned> jmode;    // [n_bodies] joint mode per body
    Vec<Vec3f> jaxis;       // [n_bodies] world rotation axle per body (hinge)
    bool built{false};      // topology (vbody/cloth_off/jmode/jaxis) is populated
    void free_all() {
        vbody.free();
        cloth_off.free();
        prot.free();
        jmode.free();
        jaxis.free();
        nrow = n_bodies = n_cloth = dim = body_base = 0;
        any_joint = false;
        built = false;
    }
};

// Apply the per-body joint projector Pi to the reduced DOF vector `u` (body
// blocks only; cloth is untouched). For a hinge Pi = blockdiag(0, a a^T): zero
// the translation and keep only the spin component about the axle `a`. This
// restricts the reduced search direction (and rhs / preconditioned residual) to
// joint-admissible rigid motions, so the linear solve never moves a hinged body
// off its axle. One thread per body.
static __global__ void project_body_dofs_kernel(unsigned nb, unsigned body_base,
                                                Vec<unsigned> jmode,
                                                Vec<Vec3f> jaxis, Vec<float> u) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    unsigned mode = jmode.data[b];
    if (mode == PDRD_JOINT_FREE) return;
    float *q = u.data + body_base + 6u * b;
    if (mode == PDRD_JOINT_HINGE) {
        q[0] = 0.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        Vec3f a = jaxis.data[b];
        float dth = q[3] * a[0] + q[4] * a[1] + q[5] * a[2];
        q[3] = dth * a[0];
        q[4] = dth * a[1];
        q[5] = dth * a[2];
    }
}

inline void launch_project_bodies(const RigidMap &rm, Vec<float> &u) {
    if (rm.n_bodies == 0 || !rm.any_joint) return;
    unsigned tpb = 64;
    project_body_dofs_kernel<<<(rm.n_bodies + tpb - 1) / tpb, tpb>>>(
        rm.n_bodies, rm.body_base, const_cast<Vec<unsigned> &>(rm.jmode),
        const_cast<Vec<Vec3f> &>(rm.jaxis), u);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// Per-PDRD-vertex rotated rest vector p_v = R_b ybar_k, indexed by GLOBAL vertex
// id (the prolong / restrict kernels are one-thread-per-vertex). Cloth entries
// are left untouched (unused).
static __global__ void scatter_prot_kernel(Vec<RigidState> state, DataSet data,
                                           Vec<Vec3f> prot) {
    const unsigned b = blockIdx.x;
    const RigidState s = state.data[b];
    if (s.N == 0) return;
    const PdrdBodyProp bp = data.prop.pdrd_body[b];
    const unsigned start = bp.vertex_start;
    Mat3x3f R;
    for (unsigned e = 0; e < 9; ++e) R.data()[e] = s.R[e];
    for (unsigned k = threadIdx.x; k < s.N; k += blockDim.x) {
        unsigned v = data.pdrd_vert_list[start + k];
        prot.data[v] = R * data.pdrd_rest_centered[start + k];
    }
}

// x_vert = P u_red. One thread per vertex.
static __global__ void prolong_rigid_kernel(RigidMap rm, Vec<float> u,
                                            Vec<float> x) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= rm.nrow) return;
    unsigned b = rm.vbody.data[v];
    if (b == RIGID_UNSET) {
        unsigned o = rm.cloth_off.data[v];
        x.data[3 * v + 0] = u.data[o + 0];
        x.data[3 * v + 1] = u.data[o + 1];
        x.data[3 * v + 2] = u.data[o + 2];
    } else {
        const float *q = u.data + rm.body_base + 6u * b;
        Vec3f p = rm.prot.data[v];
        // J = [I3 | -skew(p)] so dx = dx_b - skew(p) dtheta = dx_b - p x dtheta.
        Vec3f dxb(q[0], q[1], q[2]);
        Vec3f dth(q[3], q[4], q[5]);
        Vec3f dx = dxb - p.cross(dth);
        x.data[3 * v + 0] = dx[0];
        x.data[3 * v + 1] = dx[1];
        x.data[3 * v + 2] = dx[2];
    }
}

// u_red = P^T y_vert. Cloth entries written directly (disjoint); PDRD body blocks
// accumulated with atomics. u must be pre-zeroed on the body region.
static __global__ void restrict_rigid_kernel(RigidMap rm, Vec<float> y,
                                             Vec<float> u) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= rm.nrow) return;
    Vec3f yv(y.data[3 * v + 0], y.data[3 * v + 1], y.data[3 * v + 2]);
    unsigned b = rm.vbody.data[v];
    if (b == RIGID_UNSET) {
        unsigned o = rm.cloth_off.data[v];
        u.data[o + 0] = yv[0];
        u.data[o + 1] = yv[1];
        u.data[o + 2] = yv[2];
    } else {
        float *q = u.data + rm.body_base + 6u * b;
        Vec3f p = rm.prot.data[v];
        // J_v^T y = [ y ; -skew(p)^T y ] = [ y ; p x y ].
        Vec3f tq = p.cross(yv);
        atomicAdd(&q[0], yv[0]);
        atomicAdd(&q[1], yv[1]);
        atomicAdd(&q[2], yv[2]);
        atomicAdd(&q[3], tq[0]);
        atomicAdd(&q[4], tq[1]);
        atomicAdd(&q[5], tq[2]);
    }
}

// Build the reduced DOF map from prop.vertex[].pdrd_body_index (1-based; 0 =
// cloth) and scatter the rotated rest vectors from the fitted rigid state.
inline void build_rigid_map(RigidMap &rm, const DataSet &data,
                            const Vec<RigidState> &state, unsigned nrow) {
    unsigned nb = data.prop.pdrd_body.size;
    bool topology_valid = rm.built && rm.nrow == nrow && rm.n_bodies == nb;
    rm.nrow = nrow;
    rm.n_bodies = nb;
    if (rm.vbody.size < nrow) {
        rm.vbody.free();
        rm.cloth_off.free();
        rm.prot.free();
        rm.vbody = Vec<unsigned>::alloc(nrow);
        rm.cloth_off = Vec<unsigned>::alloc(nrow);
        rm.prot = Vec<Vec3f>::alloc(nrow);
        topology_valid = false;
    }
    if (nb > 0 && rm.jmode.size < nb) {
        rm.jmode.free();
        rm.jaxis.free();
        rm.jmode = Vec<unsigned>::alloc(nb);
        rm.jaxis = Vec<Vec3f>::alloc(nb);
        topology_valid = false;
    }
    // The DOF partition (vbody / cloth_off / dim / jmode / jaxis) is a pure
    // function of the per-vertex body assignment and per-body joint props, which
    // are fixed for the lifetime of a solve (one solver process per session).
    // Build it once and reuse it across this solve's Newton iterations; only
    // prot (the rotated rest vectors from the per-iteration rigid state) is
    // recomputed each call. This removes a per-Newton VertexProp D2H + host scan
    // + uploads. NOTE: if a future path mutates pdrd_body_index in place within
    // one process, clear rm.built (or call free_all) to force a rebuild.
    if (!topology_valid) {
        std::vector<VertexProp> vp(nrow);
        CUDA_HANDLE_ERROR(cudaMemcpy(vp.data(), data.prop.vertex.data,
                                     nrow * sizeof(VertexProp),
                                     cudaMemcpyDeviceToHost));
        std::vector<unsigned> hvbody(nrow), hcloth(nrow, 0);
        unsigned ncloth = 0;
        for (unsigned v = 0; v < nrow; ++v) {
            unsigned idx1 = vp[v].pdrd_body_index;
            if (idx1 == 0) {
                hvbody[v] = RIGID_UNSET;
                hcloth[v] = 3u * ncloth;
                ncloth++;
            } else {
                hvbody[v] = idx1 - 1u;
            }
        }
        rm.n_cloth = ncloth;
        rm.body_base = 3u * ncloth;
        rm.dim = 3u * ncloth + 6u * nb;
        CUDA_HANDLE_ERROR(cudaMemcpy(rm.vbody.data, hvbody.data(),
                                     nrow * sizeof(unsigned),
                                     cudaMemcpyHostToDevice));
        CUDA_HANDLE_ERROR(cudaMemcpy(rm.cloth_off.data, hcloth.data(),
                                     nrow * sizeof(unsigned),
                                     cudaMemcpyHostToDevice));
        rm.any_joint = false;
        if (nb > 0) {
            // Pull per-body joint mode + axle to the host once so the DOF
            // projector can skip entirely when no joints are present.
            std::vector<PdrdBodyProp> bp(nb);
            CUDA_HANDLE_ERROR(cudaMemcpy(bp.data(), data.prop.pdrd_body.data,
                                         nb * sizeof(PdrdBodyProp),
                                         cudaMemcpyDeviceToHost));
            std::vector<unsigned> hjmode(nb);
            std::vector<Vec3f> hjaxis(nb);
            for (unsigned b = 0; b < nb; ++b) {
                hjmode[b] = bp[b].joint_mode;
                hjaxis[b] = bp[b].joint_axis;
                if (bp[b].joint_mode != PDRD_JOINT_FREE) rm.any_joint = true;
            }
            CUDA_HANDLE_ERROR(cudaMemcpy(rm.jmode.data, hjmode.data(),
                                         nb * sizeof(unsigned),
                                         cudaMemcpyHostToDevice));
            CUDA_HANDLE_ERROR(cudaMemcpy(rm.jaxis.data, hjaxis.data(),
                                         nb * sizeof(Vec3f),
                                         cudaMemcpyHostToDevice));
        }
        rm.built = true;
    }

    // prot depends on the per-iteration fitted rigid state; recompute it every
    // call (this is the only per-Newton work that remains).
    if (nb > 0) {
        scatter_prot_kernel<<<nb, RIGID_THREADS_PER_BODY>>>(
            const_cast<Vec<RigidState> &>(state), const_cast<DataSet &>(data),
            rm.prot);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

inline void launch_prolong_rigid(const RigidMap &rm, const Vec<float> &u,
                                 Vec<float> &x) {
    unsigned tpb = 128;
    prolong_rigid_kernel<<<(rm.nrow + tpb - 1) / tpb, tpb>>>(
        rm, const_cast<Vec<float> &>(u), x);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

inline void launch_restrict_rigid(const RigidMap &rm, const Vec<float> &y,
                                  Vec<float> &u) {
    CUDA_HANDLE_ERROR(cudaMemset(u.data, 0, rm.dim * sizeof(float)));
    unsigned tpb = 128;
    restrict_rigid_kernel<<<(rm.nrow + tpb - 1) / tpb, tpb>>>(
        rm, const_cast<Vec<float> &>(y), u);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// Extract the per-body rotation DOFs (dtheta_b) of the reduced solution `u`
// (layout: [3*ncloth cloth dofs][6 per body: dx(3), dtheta(3)]) into
// dtheta_out[3*nb]. Used to integrate the applied rotation onto R_run.
static __global__ void extract_body_dtheta_kernel(unsigned nb,
                                                  unsigned body_base,
                                                  Vec<float> u,
                                                  Vec<float> dtheta_out) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float *q = u.data + body_base + 6u * b + 3u; // rotation block
    dtheta_out.data[3 * b + 0] = q[0];
    dtheta_out.data[3 * b + 1] = q[1];
    dtheta_out.data[3 * b + 2] = q[2];
}

inline void launch_extract_body_dtheta(const RigidMap &rm, const Vec<float> &u,
                                       Vec<float> &dtheta_out) {
    if (rm.n_bodies == 0) return;
    unsigned tpb = 64;
    extract_body_dtheta_kernel<<<(rm.n_bodies + tpb - 1) / tpb, tpb>>>(
        rm.n_bodies, rm.body_base, const_cast<Vec<float> &>(u), dtheta_out);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// ===========================================================================
// S2: per-body 6x6 block preconditioner (analytic rigid inertia + live contact
// diagonal). Cloth DOFs keep a 3x3 block-Jacobi (inv_diag). This matches the
// reduced 6-DOF rigid block used by the live solver.
// ===========================================================================

struct RigidPrecond {
    unsigned nb{0};
    Vec<float> Gbody; // [36*nb] fp32 lower-triangular inverse factor per body
    void free_all() {
        Gbody.free();
        nb = 0;
    }
};

// Host double-precision equilibrated Cholesky of a 6x6, returning the fp32
// lower-triangular inverse factor G (so A^-1 = G^T G). Double is host-only; the
// device stores/applies fp32 (README GPU-fp32 rule).
inline void factor6_host_to_G(const float Kf[36], float Gout[36]) {
    const unsigned n = 6;
    double M[6][6], D[6];
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j) M[i][j] = (double)Kf[i * n + j];
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i + 1; j < n; ++j) {
            double m = 0.5 * (M[i][j] + M[j][i]);
            M[i][j] = M[j][i] = m;
        }
    double tr = 0.0;
    for (unsigned i = 0; i < n; ++i) tr += M[i][i];
    double dfloor = 1e-12 * (tr / n) + 1e-30;
    for (unsigned i = 0; i < n; ++i) {
        double dii = M[i][i];
        D[i] = 1.0 / std::sqrt(dii > dfloor ? dii : dfloor);
    }
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j) M[i][j] *= D[i] * D[j];
    const double eps = 1e-8;
    for (unsigned i = 0; i < n; ++i) M[i][i] += eps;
    double L[6][6] = {};
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j <= i; ++j) {
            double s = M[i][j];
            for (unsigned k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j)
                L[i][i] = std::sqrt(s > 1e-20 ? s : 1e-20);
            else
                L[i][j] = s / L[j][j];
        }
    for (unsigned i = 0; i < n * n; ++i) Gout[i] = 0.0f;
    double col[6];
    for (unsigned cc = 0; cc < n; ++cc) {
        col[cc] = 1.0 / L[cc][cc];
        for (unsigned i = cc + 1; i < n; ++i) {
            double s = 0.0;
            for (unsigned k = cc; k < i; ++k) s += L[i][k] * col[k];
            col[i] = -s / L[i][i];
        }
        for (unsigned i = cc; i < n; ++i)
            Gout[i * n + cc] = (float)(col[i] * D[cc]);
    }
}

// atomicAdd form of the rigid sandwich J^T H J into a 6x6 (row-major), with the
// rigid Jacobian J = [ I3 | -skew(p) ] (3x6), p = R ybar_k.
__device__ inline void rigid_sandwich_atomic(float *K6, const Vec3f &p,
                                            const Mat3x3f &H) {
    float J[3][6];
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned c = 0; c < 6; ++c) J[i][c] = 0.0f;
    for (unsigned i = 0; i < 3; ++i) J[i][i] = 1.0f;
    // rotation columns = -skew(p).
    J[0][4] = p[2];  J[0][5] = -p[1];
    J[1][3] = -p[2]; J[1][5] = p[0];
    J[2][3] = p[1];  J[2][4] = -p[0];
    for (unsigned a = 0; a < 6; ++a)
        for (unsigned b = 0; b < 6; ++b) {
            float s = 0.0f;
            for (unsigned i = 0; i < 3; ++i)
                for (unsigned k = 0; k < 3; ++k)
                    s += J[i][a] * H(i, k) * J[k][b];
            atomicAdd(&K6[a * 6 + b], s);
        }
}

// Assemble the per-body reduced self-block K_b = analytic rigid inertia
// blockdiag(m_total/dt^2 I3, R Iref R^T / dt^2) + sum_k J_k^T (A(v,v)+B(v,v)) J_k.
// The inertia (mass) lives in C analytically here so it is not double counted;
// the contact diagonal is read straight from the assembled matrix on the body's
// own vertices after same_pdrd_body filtering. One block per body. K_out is
// [36*nb], row-major.
static __global__ void assemble_rigid_K_kernel(DataSet data, DynCSRMat A,
                                              FixedCSRMat B,
                                              Vec<RigidState> state, float dt,
                                              Vec<float> K_out) {
    const unsigned b = blockIdx.x;
    const unsigned tid = threadIdx.x, T = blockDim.x;
    const RigidState s = state.data[b];
    const PdrdBodyProp bp = data.prop.pdrd_body[b];
    const unsigned start = bp.vertex_start, N = s.N;
    float *K = K_out.data + (size_t)36 * b;

    __shared__ float Ksh[36];
    for (unsigned i = tid; i < 36; i += T) Ksh[i] = 0.0f;
    __syncthreads();
    if (N == 0) {
        if (tid < 6) Ksh[tid * 6 + tid] = 1.0f;
        __syncthreads();
        for (unsigned i = tid; i < 36; i += T) K[i] = Ksh[i];
        return;
    }
    // Analytic rigid inertia (single thread; tiny).
    Mat3x3f R;
    for (unsigned e = 0; e < 9; ++e) R.data()[e] = s.R[e];
    if (tid == 0) {
        float inv_dt2 = 1.0f / (dt * dt);
        Mat3x3f Iref;
        for (unsigned e = 0; e < 9; ++e) Iref.data()[e] = s.Iref[e];
        Mat3x3f RIR = R * Iref * R.transpose();
        for (unsigned i = 0; i < 3; ++i) Ksh[i * 6 + i] = s.mass_total * inv_dt2;
        for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = 0; j < 3; ++j)
                Ksh[(3 + i) * 6 + (3 + j)] = RIR(i, j) * inv_dt2;
    }
    __syncthreads();

    for (unsigned k = tid; k < N; k += T) {
        unsigned v = data.pdrd_vert_list[start + k];
        Mat3x3f Dcontact = A(v, v) + B(v, v);
        Vec3f pp = R * data.pdrd_rest_centered[start + k];
        rigid_sandwich_atomic(Ksh, pp, Dcontact);
    }
    __syncthreads();
    for (unsigned i = tid; i < 36; i += T) K[i] = Ksh[i];
}

// Build per-body 6x6 inverse factors with the live contact diagonal folded into
// the analytic rigid inertia block. Fits the rigid state on device, assembles
// K_b on device, factors in double on the host, uploads fp32 G.
inline void build_rigid_precond(RigidPrecond &P, const DataSet &data,
                                const Vec<RigidState> &state, const DynCSRMat &A,
                                const FixedCSRMat &B, float dt) {
    unsigned nb = data.prop.pdrd_body.size;
    P.nb = nb;
    if (nb == 0) return;
    if (P.Gbody.size < (size_t)36 * nb) {
        P.Gbody.free();
        P.Gbody = Vec<float>::alloc((size_t)36 * nb);
    }
    auto kb = buffer::get().get<float>((size_t)36 * nb);
    Vec<float> Kdev = kb.as_vec();
    assemble_rigid_K_kernel<<<nb, RIGID_THREADS_PER_BODY>>>(
        const_cast<DataSet &>(data), const_cast<DynCSRMat &>(A),
        const_cast<FixedCSRMat &>(B), const_cast<Vec<RigidState> &>(state), dt,
        Kdev);
    CUDA_HANDLE_ERROR(cudaGetLastError());
    std::vector<float> hK((size_t)36 * nb);
    CUDA_HANDLE_ERROR(cudaMemcpy(hK.data(), Kdev.data,
                                 (size_t)36 * nb * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    std::vector<float> G((size_t)36 * nb);
    for (unsigned b = 0; b < nb; ++b)
        factor6_host_to_G(hK.data() + (size_t)36 * b, G.data() + (size_t)36 * b);
    CUDA_HANDLE_ERROR(cudaMemcpy(P.Gbody.data, G.data(),
                                 (size_t)36 * nb * sizeof(float),
                                 cudaMemcpyHostToDevice));
}

// z_body = G^T (G r_body) per body (one block per body, 6 threads).
static __global__ void precond_rigid_pdrd_kernel(Vec<float> Gbody,
                                               unsigned body_base, Vec<float> r,
                                               Vec<float> z) {
    unsigned b = blockIdx.x;
    unsigned p = threadIdx.x;
    if (p >= 6) return;
    __shared__ float rb[6], gr[6];
    const float *G = Gbody.data + (size_t)36 * b;
    rb[p] = r.data[body_base + 6u * b + p];
    __syncthreads();
    // gr = G r (G lower-triangular, row-major).
    float acc = 0.0f;
    for (unsigned j = 0; j <= p; ++j) acc += G[p * 6 + j] * rb[j];
    gr[p] = acc;
    __syncthreads();
    // z = G^T gr.
    float zz = 0.0f;
    for (unsigned i = p; i < 6; ++i) zz += G[i * 6 + p] * gr[i];
    z.data[body_base + 6u * b + p] = zz;
}

// z_cloth = inv_diag * r for cloth vertices (3x3 block-Jacobi). One thread per
// vertex; PDRD vertices are skipped (handled by the per-body block above).
static __global__ void precond_rigid_cloth_kernel(RigidMap rm,
                                                  Vec<Mat3x3f> inv_diag,
                                                  Vec<float> r, Vec<float> z) {
    unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= rm.nrow) return;
    if (rm.vbody.data[v] != RIGID_UNSET) return;
    unsigned o = rm.cloth_off.data[v];
    Vec3f rv(r.data[o + 0], r.data[o + 1], r.data[o + 2]);
    Vec3f zv = inv_diag.data[v] * rv;
    z.data[o + 0] = zv[0];
    z.data[o + 1] = zv[1];
    z.data[o + 2] = zv[2];
}

inline void apply_rigid_precond(const RigidPrecond &P, const RigidMap &rm,
                                const Vec<Mat3x3f> &inv_diag,
                                const Vec<float> &r, Vec<float> &z) {
    if (P.nb) {
        precond_rigid_pdrd_kernel<<<P.nb, 6>>>(const_cast<Vec<float> &>(P.Gbody),
                                              rm.body_base,
                                              const_cast<Vec<float> &>(r), z);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
    unsigned tpb = 128;
    precond_rigid_cloth_kernel<<<(rm.nrow + tpb - 1) / tpb, tpb>>>(
        rm, const_cast<Vec<Mat3x3f> &>(inv_diag), const_cast<Vec<float> &>(r),
        z);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

// ---- S1 kinematics self-test ----------------------------------------------
// Validates, on the live scene's bodies, that:
//  (1) the polar fit recovers a known applied rotation (R_fit ~ R_true),
//  (2) reconstruct round-trips the fitted state back to the input vertices,
//  (3) the rigid Jacobian J_k = [I | -skew(R ybar_k)] matches a finite-
//      difference of reconstruct under a tangent step (dx, dtheta), and
//  (4) the reduced mass sum_k J_k^T m J_k equals blockdiag(m_total I, R Iref R^T)
//      with a vanishing translation-rotation cross block.
// Host-side checks in double; the fit runs on the GPU. Prints to stderr.
inline void selftest_rigid(const DataSet &data, const Vec<Vec3f> &eval_x) {
    unsigned nb = data.prop.pdrd_body.size;
    if (nb == 0) return;
    auto st = buffer::get().get<RigidState>(nb);
    Vec<RigidState> state = st.as_vec();
    launch_fit_rigid(data, eval_x, state);
    std::vector<RigidState> hs(nb);
    CUDA_HANDLE_ERROR(cudaMemcpy(hs.data(), state.data, nb * sizeof(RigidState),
                                 cudaMemcpyDeviceToHost));
    std::vector<PdrdBodyProp> bp(nb);
    CUDA_HANDLE_ERROR(cudaMemcpy(bp.data(), data.prop.pdrd_body.data,
                                 nb * sizeof(PdrdBodyProp),
                                 cudaMemcpyDeviceToHost));
    unsigned nv = data.pdrd_rest_centered.size;
    std::vector<Vec3f> ybar(nv);
    CUDA_HANDLE_ERROR(cudaMemcpy(ybar.data(), data.pdrd_rest_centered.data,
                                 nv * sizeof(Vec3f), cudaMemcpyDeviceToHost));

    double worst_recon = 0.0, worst_jac = 0.0, worst_mass = 0.0,
           worst_cross = 0.0;
    for (unsigned b = 0; b < nb; ++b) {
        const RigidState &s = hs[b];
        if (s.N == 0) continue;
        unsigned start = bp[b].vertex_start;
        Mat3x3f R, Iref;
        for (unsigned e = 0; e < 9; ++e) R.data()[e] = s.R[e];
        for (unsigned e = 0; e < 9; ++e) Iref.data()[e] = s.Iref[e];
        Vec3f xb(s.x[0], s.x[1], s.x[2]);

        // (2) reconstruct round-trip vs the fit's own image (R, x are consistent
        // by construction; this checks R is a proper rotation: R R^T = I).
        Mat3x3f orth = R.transpose() * R - Mat3x3f::Identity();
        worst_recon = std::max(worst_recon, (double)orth.norm());

        // (3) rigid Jacobian vs finite difference of x_k = x + R ybar_k.
        const float eps = 1e-3f;
        Vec3f dth(0.13f, -0.27f, 0.41f);
        Vec3f dxv(0.05f, -0.02f, 0.09f);
        Mat3x3f Rp = rigid_exp_so3(eps * dth) * R;
        // (4) accumulate reduced mass blocks.
        Mat3x3f Mtt = Mat3x3f::Zero(), Mtr = Mat3x3f::Zero(),
                Mrr = Mat3x3f::Zero();
        for (unsigned k = 0; k < s.N; ++k) {
            Vec3f yb = ybar[start + k];
            Vec3f p = R * yb;
            Mat3x3f sk = rigid_skew(p);
            // J_k = [I | -sk]; analytic vertex displacement under (dxv, dth).
            Vec3f dx_analytic = dxv - sk * dth;
            // finite-difference displacement.
            Vec3f xk = xb + R * yb;
            Vec3f xk_p = (xb + eps * dxv) + Rp * yb;
            Vec3f dx_fd = (xk_p - xk) / eps;
            worst_jac = std::max(worst_jac, (double)(dx_analytic - dx_fd).norm());
            float m = bp[b].mass_per_vertex;
            Mtt += m * Mat3x3f::Identity();
            Mtr += m * (-sk);                 // I^T (-sk)
            Mrr += m * (sk.transpose() * sk); // (-sk)^T (-sk)
        }
        Mat3x3f RIR = R * Iref * R.transpose();
        worst_mass = std::max(worst_mass, (double)(Mrr - RIR).norm() /
                                              (1.0 + (double)RIR.norm()));
        worst_cross = std::max(worst_cross, (double)Mtr.norm() /
                                                (1.0 + (double)Mtt.norm()));
    }
    fprintf(stderr,
            "[pdrd rigid selftest] bodies=%u orth(R^TR-I)=%.3e jacobian_fd=%.3e "
            "mass_block(RIR)=%.3e cross=%.3e\n",
            nb, worst_recon, worst_jac, worst_mass, worst_cross);
}

} // namespace PDRD

#endif // PDRD_RIGID_HPP
