// File: contact.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../barrier/barrier.hpp"
#include "../buffer/buffer.hpp"
#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../lbvh/bvh_storage.hpp"
#include "../energy/model/fix.hpp"
#include "../energy/model/friction.hpp"
#include "../energy/model/push.hpp"
#include "../energy/model/rigid_core.hpp"
#include "../kernels/reduce.hpp"
#include "../kernels/vec_ops.hpp"
#include "../main/cuda_utils.hpp"
#include "../simplelog/SimpleLog.h"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "aabb.hpp"
#include "accd.hpp"
#include "contact.hpp"
#include "distance.hpp"
#include <cassert>

namespace contact {

namespace storage {
Vec<AABB> face_aabb, edge_aabb, vertex_aabb;
Vec<AABB> collision_mesh_face_aabb, collision_mesh_edge_aabb;
IntersectionRecord host_intersection_records[MAX_INTERSECTION_RECORDS];
unsigned host_intersection_count = 0;
} // namespace storage

__device__ __forceinline__ float combine_friction(float a, float b, FrictionMode mode) {
    switch (mode) {
    case FrictionMode::Max:
        return fmaxf(a, b);
    case FrictionMode::Mean:
        return 0.5f * (a + b);
    case FrictionMode::Min:
    default:
        return fminf(a, b);
    }
}

__device__ void record_intersection(
    IntersectionRecord *records, unsigned *counter,
    unsigned type, unsigned elem0, unsigned elem1,
    const Vec3f *verts0, unsigned n0,
    const Vec3f *verts1, unsigned n1) {
    unsigned slot = atomicAdd(counter, 1);
    if (slot < MAX_INTERSECTION_RECORDS) {
        IntersectionRecord &r = records[slot];
        r.type = type;
        r.elem0 = elem0;
        r.elem1 = elem1;
        r.num_verts0 = n0;
        r.num_verts1 = n1;
        unsigned k = 0;
        for (unsigned i = 0; i < n0; i++)
            for (unsigned d = 0; d < 3; d++)
                r.positions[k++] = (float)verts0[i][d];
        for (unsigned i = 0; i < n1; i++)
            for (unsigned d = 0; d < 3; d++)
                r.positions[k++] = (float)verts1[i][d];
    }
}

static unsigned max_of_two(unsigned a, unsigned b) { return a > b ? a : b; }
__device__ inline float sqr(float x) { return x * x; }

void initialize(const DataSet &data, const ParamSet &param) {
    // Initialize AABB buffers (kept in storage namespace)
    // Calculate BVH node counts: 2n-1 nodes for n primitives
    unsigned n_faces = data.mesh.mesh.face.size;
    unsigned n_edges = data.mesh.mesh.edge.size;
    unsigned n_verts = data.surface_vert_count;
    unsigned face_bvh_nodes = n_faces > 0 ? 2 * n_faces - 1 : 0;
    unsigned edge_bvh_nodes = n_edges > 0 ? 2 * n_edges - 1 : 0;
    unsigned vertex_bvh_nodes = n_verts > 0 ? 2 * n_verts - 1 : 0;

    storage::face_aabb = Vec<AABB>::alloc(face_bvh_nodes);
    storage::edge_aabb = Vec<AABB>::alloc(edge_bvh_nodes);
    storage::vertex_aabb = Vec<AABB>::alloc(vertex_bvh_nodes);

    unsigned cm_faces = data.constraint.mesh.face.size;
    unsigned cm_edges = data.constraint.mesh.edge.size;
    unsigned cm_face_bvh_nodes = cm_faces > 0 ? 2 * cm_faces - 1 : 0;
    unsigned cm_edge_bvh_nodes = cm_edges > 0 ? 2 * cm_edges - 1 : 0;

    storage::collision_mesh_face_aabb = Vec<AABB>::alloc(cm_face_bvh_nodes);
    storage::collision_mesh_edge_aabb = Vec<AABB>::alloc(cm_edge_bvh_nodes);
}

Vec<AABB> &get_face_aabb() { return storage::face_aabb; }
Vec<AABB> &get_edge_aabb() { return storage::edge_aabb; }
Vec<AABB> &get_vertex_aabb() { return storage::vertex_aabb; }
Vec<AABB> &get_collision_mesh_face_aabb() { return storage::collision_mesh_face_aabb; }
Vec<AABB> &get_collision_mesh_edge_aabb() { return storage::collision_mesh_edge_aabb; }

__device__ bool edge_has_shared_vert(const Vec2u &e0, const Vec2u &e1) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (e0[i] == e1[j])
                return true;
        }
    }
    return false;
}

void check_success(Vec<char> array) {
    DISPATCH_START(array.size)[array] __device__(unsigned i) {
        assert(array[i] == 0);
    }
    DISPATCH_END;
}

template <typename F> struct AABB_AABB_Tester {
    __device__ AABB_AABB_Tester(F &op) : op(op) {}
    __device__ bool operator()(unsigned index) { return op(index); }
    __device__ bool test(const AABB &a, const AABB &b) {
        return aabb::overlap(a, b);
    }
    F &op;
};

template <unsigned N>
__device__ void
extend_contact_force_hess(const Proximity<N> &prox, const Vec3f &force,
                          const Mat3x3f &hess, SMatf<3, N> &ext_force,
                          SMatf<N * 3, N * 3> &ext_hess) {
    for (unsigned i = 0; i < N; ++i) {
        ext_force.col(i) = prox.value[i] * force;
    }
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            ext_hess.template block<3, 3>(3 * i, 3 * j) =
                prox.value[i] * prox.value[j] * hess;
        }
    }
}

template <unsigned N>
__device__ static void
atomic_embed_hessian(const Eigen::Vector<unsigned, N> &index,
                     const Eigen::Matrix<float, N * 3, N * 3> &H,
                     FixedCSRMat &fixed, DynCSRMat &dyn) {
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            unsigned i = index[ii];
            unsigned j = index[jj];
            if (i <= j) {
                Mat3x3f val = H.template block<3, 3>(ii * 3, jj * 3);
                if (!fixed.push(i, j, val)) {
                    dyn.push(i, j, val);
                }
            }
        }
    }
}

template <unsigned N>
__device__ static void
dry_atomic_embed_hessian(const Eigen::Vector<unsigned, N> &index,
                         const FixedCSRMat &fixed, DynCSRMat &dyn) {
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            unsigned i = index[ii];
            unsigned j = index[jj];
            if (i <= j) {
                if (!fixed.exists(i, j)) {
                    dyn.dry_push(i, j);
                }
            }
        }
    }
}

// SAND rolling spin clamp (the bound the spin term must satisfy; the clamp itself
// runs in post-solve, see the NOTE below and sand_rigid.hpp). The spin-induced
// contact displacement must never exceed the realized tangential center
// displacement |P dx| (P = I - n n^T). This keeps the friction slip
// fdx = dx - spin from reversing past no-slip (over-roll),
// so kinetic friction can never flip into a propelling direction, the energy-
// pumping ratchet. At the cap the slip is zero (free rolling); below it friction
// resists, so travel is bounded by a frictionless slide regardless of how the
// lagged omega was integrated. n is the grain's outward contact normal.
__device__ inline Vec3f grain_clamp_spin(const Vec3f &spin, const Vec3f &dx,
                                         const Vec3f &n) {
    // NOTE: clamping the spin displacement to exactly |P dx| in the embed
    // dead-locks the grain at stick (to roll it must move, but it cannot move
    // until it rolls). The anti-pump bound is instead applied to omega in the
    // post-solve integrate (under-roll clamp, see sand_rigid.hpp), so this is a
    // pass-through; kept as a single point of control for the spin term.
    (void)dx;
    (void)n;
    return spin;
}

// SAND implicit (Schur-condensed) rolling: accumulate one analytic contact's
// Schur blocks for a grain.
// fric_hess is the friction Hessian lambda*P (P = I - n n^T, trace 2*lambda); dx
// the grain's center slip; n the OUTWARD contact normal; r the grain radius. With
// the slip u = P(dx - r(dtheta x n)) = P dx + r P [n]x dtheta, the per-contact
// friction energy 1/2 lambda |u|^2 gives:
//   A_c = lambda r^2 (P [n]x)^T (P [n]x)   (angular, SPD; symmetric by construction)
//   B_c = lambda r   P [n]x                 (translation<->rotation coupling)
//   grot_c = B_c^T (P dx)                   (rotational gradient at dtheta = 0)
__device__ inline void accumulate_grain_schur(const Mat3x3f &fric_hess,
                                              const Vec3f &dx, const Vec3f &n,
                                              float r, Mat3x3f &A_loc,
                                              Mat3x3f &B_loc, Vec3f &grot_loc) {
    float lambda =
        (fric_hess(0, 0) + fric_hess(1, 1) + fric_hess(2, 2)) * 0.5f;
    if (lambda <= 0.0f) {
        return; // no friction (mu = 0 or zero normal force): no coupling
    }
    Mat3x3f P = Mat3x3f::Identity() - n * n.transpose();
    Mat3x3f PS = P * RigidCore::rigid_skew(n);
    Mat3x3f B_c = (lambda * r) * PS;
    A_loc += (lambda * r * r) * (PS.transpose() * PS);
    B_loc += B_c;
    grot_loc += B_c.transpose() * (P * dx);
}

template <unsigned N>
__device__ void embed_contact_force_hess(
    const Proximity<N> &prox, const Vec<Vec3f> &x0, const Vec<Vec3f> &x,
    const FixedCSRMat &fixed_in, FixedCSRMat &fixed_out, Vec<float> &force_out,
    const Vec<float> &force_in, DynCSRMat &dyn_out, float ghat, float offset,
    const Vec<VertexProp> &vert_prop, Barrier barrier, float dt, float friction,
    unsigned count, const ParamSet &param, int stage, bool include_friction,
    unsigned i, const Vec3f *fdx_override = nullptr,
    Vec3f *out_fric_grad = nullptr, float *out_lambda = nullptr) {

    // Stage 0 only reserves CSR slots from prox.index; the per-pair prologue
    // below (mass / ex / normal / compute_stiffness, the last of which streams
    // fixed_in) is computed and discarded in this stage. Take the dry path
    // before it. Release byte-identical: the prologue is side-effect-free and
    // the asserts are NDEBUG-stripped.
    if (stage == 0) {
        dry_atomic_embed_hessian<N>(prox.index, fixed_out, dyn_out);
        return;
    }

    SVecf<N> mass;
    Vec3f ex0 = Vec3f::Zero();
    Vec3f ex = Vec3f::Zero();
    float area = 0.0f;
    float wsum = 0.0f;
    for (int ii = 0; ii < N; ++ii) {
        unsigned index = prox.index[ii];
        mass[ii] = vert_prop[index].mass;
        float w = prox.value[ii];
        area += fabsf(w) * vert_prop[index].area;
        wsum += fabsf(w);
        ex0 += float(w) * x0[index];
        ex += float(w) * x[index];
    }

    Vec3f ex_fp = ex;
    assert(ex0.squaredNorm() > sqr(offset));
    assert(ex_fp.squaredNorm() > sqr(offset));

    Vec3f dx = ex - ex0;
    if (wsum) {
        area /= wsum;
    }
    Vec3f normal = ex_fp.normalized();
    float stiff_k = barrier::compute_stiffness<N>(prox, mass, fixed_in, ex_fp,
                                                  ghat, offset, param);

    Vec3f f = stiff_k *
              barrier::compute_edge_gradient(ex_fp, ghat, offset, barrier);
    Mat3x3f H = stiff_k *
                barrier::compute_edge_hessian(ex_fp, ghat, offset, barrier);

    if (include_friction) {
        // SAND grains override the friction slip with the rolling-adjusted
        // contact-point slip (both grains' lagged spin folded in by the
        // caller); other contacts use the plain relative displacement dx.
        const Vec3f &slip = fdx_override ? *fdx_override : dx;
        Friction _friction(f, slip, normal, friction, param.friction_eps);
        Vec3f fg = _friction.gradient();
        Mat3x3f fh = _friction.hessian();
        f += fg;
        H += fh;
        // Export the friction gradient + stiffness by value (Friction has a
        // reference member and no default ctor, so it cannot be returned).
        if (out_fric_grad) {
            *out_fric_grad = fg;
        }
        if (out_lambda) {
            *out_lambda = (fh(0, 0) + fh(1, 1) + fh(2, 2)) * 0.5f;
        }
    }

    SMatf<3, N> ext_force;
    SMatf<N * 3, N * 3> ext_hess;
    extend_contact_force_hess<N>(prox, f, H, ext_force, ext_hess);
    utility::atomic_embed_force<N>(prox.index, count * ext_force, force_out);
    atomic_embed_hessian<N>(prox.index, count * ext_hess, fixed_out, dyn_out);
}

// Detect-once cache for one contact type. Stage 0 appends each active
// (query, candidate) primitive pair into a flat list so stage 1 can fill it
// without a second BVH traversal (the dominant assembly cost). A null `data`
// disables recording (used by the stage-1 replay and the overflow fallback).
// Candidates come from a shared BVH, so the slot counter is atomic; each write
// lands in its own slot and is race-free. If the list overflows its capacity
// `overflow` is set and the caller falls back to a full BVH fill, so
// correctness never depends on `cap`.
struct ContactPairCache {
    Vec2u *data{nullptr};
    unsigned *cnt{nullptr};
    unsigned *overflow{nullptr};
    unsigned cap{0};
    __device__ void record(unsigned a, unsigned b) const {
        unsigned slot = atomicAdd(cnt, 1u);
        if (slot < cap) {
            data[slot] = Vec2u(a, b);
        } else {
            *overflow = 1u;
        }
    }
};

struct PointPointContactForceHessEmbed {

    unsigned vertex_index;
    unsigned rod_count;
    const Vec<Vec2u> &edge;
    const Vec<Vec3u> &face;
    const VertexNeighbor &neighbor;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &prop;
    const Vec<VertexParam> &vertex_params;
    int stage;
    float dt;
    const ParamSet &param;
    // Full dataset, for reading grain spin state (grain_inv_inertia, grain_omega)
    // and atomic-accumulating grain_torque / grain_ang_stiff / grain_contact_normal.
    const DataSet &data;
    ContactPairCache cache; // detect-once recording (stage 0); {} = disabled

    __device__ bool operator()(unsigned index) {
        bool either_dyn =
            prop[vertex_index].fix_index == 0 || prop[index].fix_index == 0;
        // Skip intra-PDRD-body vertex pairs: vertices on the same body share one
        // exact rigid transform, so internal collisions are physically
        // meaningless and just inflate the linear system.
        unsigned bid_a = prop[vertex_index].pdrd_body_index;
        unsigned bid_b = prop[index].pdrd_body_index;
        bool same_pdrd_body = bid_a != 0 && bid_a == bid_b;
        if (index < vertex_index && either_dyn && !same_pdrd_body) {
            const Vec3f &x = eval_x[vertex_index];
            const Vec3f &y = eval_x[index];
            Vec3f e = x - y;
            const VertexParam &vparam_vertex =
                vertex_params[prop[vertex_index].param_index];
            const VertexParam &vparam_index =
                vertex_params[prop[index].param_index];
            float offset = vparam_vertex.offset + vparam_index.offset;
            float friction =
                combine_friction(vparam_vertex.friction, vparam_index.friction, param.friction_mode);
            float ghat = 0.5f * (vparam_vertex.ghat + vparam_index.ghat);
            if (e.squaredNorm() < sqr(ghat + offset)) {
                unsigned count = 0;
                bool include_friction = true;
                if (neighbor.edge.count(index)) {
                    for (unsigned j = neighbor.edge.offset[index];
                         j < neighbor.edge.offset[index + 1]; ++j) {
                        Vec2u f = edge[neighbor.edge.data[j]];
                        if (f[0] == vertex_index || f[1] == vertex_index) {
                            ++count;
                            include_friction = false;
                        } else {
                            Vec2f c =
                                distance::point_edge_distance_coeff<float,
                                                                    float>(
                                    x, eval_x[f[0]], eval_x[f[1]]);
                            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                                continue;
                            } else {
                                ++count;
                            }
                        }
                    }
                    if (neighbor.face.count(index)) {
                        for (unsigned j = neighbor.face.offset[index];
                             j < neighbor.face.offset[index + 1]; ++j) {
                            Vec3u f = face[neighbor.face.data[j]];
                            if (f[0] == vertex_index || f[1] == vertex_index ||
                                f[2] == vertex_index) {
                                ++count;
                                include_friction = false;
                            } else {
                                Vec3f c =
                                    distance::point_triangle_distance_coeff<
                                        float, float>(x, eval_x[f[0]],
                                                           eval_x[f[1]],
                                                           eval_x[f[2]]);
                                if (c.maxCoeff() < 1.0f &&
                                    c.minCoeff() > 0.0f) {
                                    continue;
                                } else {
                                    ++count;
                                }
                            }
                        }
                    }
                } else {
                    count = 1u;
                }
                if (count) {
                    assert(e.squaredNorm() > sqr(offset));
                    Proximity<2> prox;
                    prox.index = Vec2u(vertex_index, index);
                    prox.value = Vec2f(1.0f, -1.0f);

                    // SAND grain rolling. If either endpoint is a grain, fold the
                    // lagged angular velocity of BOTH grains into the contact-point
                    // slip (no-slip relation u = dx_rel - dt*[r_a omega_a x n +
                    // r_b omega_b x n], n the a-side outward normal) so the contact
                    // rolls, and after the friction is built atomic-accumulate the
                    // torque tau = r*(n x g) onto each grain (g = friction
                    // gradient, +g on a / -g on b cancels the -n on b, so both get
                    // r*(n x g) = genuine counter-rotation = rolling). Gated so
                    // non-grain pairs and the dry (stage 0) / friction-off passes
                    // are byte-identical (fdx_ptr stays null -> embed uses dx).
                    unsigned a = vertex_index, b = index;
                    bool ga = data.grain_inv_inertia[a] > 0.0f;
                    bool gb = data.grain_inv_inertia[b] > 0.0f;
                    Vec3f fdx;
                    const Vec3f *fdx_ptr = nullptr;
                    Vec3f fric_grad = Vec3f::Zero();
                    float fric_lambda = 0.0f;
                    Vec3f n = Vec3f::Zero();
                    float ra = 0.0f, rb = 0.0f;
                    if ((ga || gb) && stage != 0 && include_friction) {
                        ra = vparam_vertex.offset;
                        rb = vparam_index.offset;
                        // Match the embed's internal slip: ex = x - y,
                        // ex0 = vertex[a] - vertex[b], dx = ex - ex0,
                        // normal = ex.normalized().
                        n = e.normalized();
                        Vec3f dx_rel = (x - y) - (vertex[a] - vertex[b]);
                        Vec3f spin = Vec3f::Zero();
                        if (ga) {
                            spin += ra * data.grain_omega[a].cross(n);
                        }
                        if (gb) {
                            spin += rb * data.grain_omega[b].cross(n);
                        }
                        // Clamp the combined spin displacement to the tangential
                        // relative step so the pair can never over-roll and let
                        // friction propel them apart (energy pump).
                        fdx = dx_rel - grain_clamp_spin(dt * spin, dx_rel, n);
                        fdx_ptr = &fdx;
                    }

                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, ghat, offset, prop, param.barrier,
                        dt, friction, count, param, stage, include_friction,
                        vertex_index, fdx_ptr, fdx_ptr ? &fric_grad : nullptr,
                        fdx_ptr ? &fric_lambda : nullptr);

                    if (fdx_ptr) {
                        // count matches the actually-deposited friction force
                        // (count * ext_force) so the torque tracks the applied
                        // force; count == 1 for pure grain-grain pile contacts.
                        Vec3f f_t = float(count) * fric_grad;
                        float lam = float(count) * fric_lambda;
                        if (ga) {
                            Vec3f tau = ra * n.cross(f_t);
                            atomicAdd(&data.grain_torque.data[a][0], tau[0]);
                            atomicAdd(&data.grain_torque.data[a][1], tau[1]);
                            atomicAdd(&data.grain_torque.data[a][2], tau[2]);
                            atomicAdd(&data.grain_ang_stiff.data[a], ra * ra * lam);
                            atomicAdd(&data.grain_contact_normal.data[a][0], n[0]);
                            atomicAdd(&data.grain_contact_normal.data[a][1], n[1]);
                            atomicAdd(&data.grain_contact_normal.data[a][2], n[2]);
                        }
                        if (gb) {
                            Vec3f tau = rb * n.cross(f_t);
                            atomicAdd(&data.grain_torque.data[b][0], tau[0]);
                            atomicAdd(&data.grain_torque.data[b][1], tau[1]);
                            atomicAdd(&data.grain_torque.data[b][2], tau[2]);
                            atomicAdd(&data.grain_ang_stiff.data[b], rb * rb * lam);
                            atomicAdd(&data.grain_contact_normal.data[b][0], -n[0]);
                            atomicAdd(&data.grain_contact_normal.data[b][1], -n[1]);
                            atomicAdd(&data.grain_contact_normal.data[b][2], -n[2]);
                        }
                    }
                    if (stage == 0 && cache.data) {
                        cache.record(vertex_index, index);
                    }
                    return true;
                }
            }
        }
        return false;
    }
};

struct PointEdgeContactForceHessEmbed {

    unsigned vertex_index;
    const Vec<Vec2u> &edge;
    const Vec<Vec3u> &face;
    const EdgeNeighbor &neighbor;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &vert_prop;
    const Vec<EdgeProp> &edge_prop;
    const Vec<VertexParam> &vertex_params;
    const Vec<EdgeParam> &edge_params;
    int stage;
    float dt;
    const ParamSet &param;
    ContactPairCache cache; // detect-once recording (stage 0); {} = disabled

    __device__ bool operator()(unsigned index) {
        Vec2u f = edge[index];
        bool either_dyn = vert_prop[vertex_index].fix_index == 0 ||
                          edge_prop[index].fixed == false;
        unsigned bid_v = vert_prop[vertex_index].pdrd_body_index;
        unsigned bid_e = vert_prop[f[0]].pdrd_body_index;
        bool same_pdrd_body = bid_v != 0 && bid_v == bid_e;
        if (f[0] != vertex_index && f[1] != vertex_index && either_dyn &&
            !same_pdrd_body) {
            const Vec3f &p = eval_x[vertex_index];
            const Vec3f &t0 = eval_x[f[0]];
            const Vec3f &t1 = eval_x[f[1]];
            Vec2f c = distance::point_edge_distance_coeff<float, float>(
                p, t0, t1);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f e = c[0] * (p - t0) + c[1] * (p - t1);
                const VertexParam &vparam =
                    vertex_params[vert_prop[vertex_index].param_index];
                const EdgeParam &eparam =
                    edge_params[edge_prop[index].param_index];
                float offset = vparam.offset + eparam.offset;
                float ghat = 0.5f * (vparam.ghat + eparam.ghat);
                float friction = combine_friction(vparam.friction, eparam.friction, param.friction_mode);
                if (e.squaredNorm() < sqr(ghat + offset)) {
                    unsigned count = 0;
                    bool include_friction = true;
                    if (neighbor.face.count(index)) {
                        for (unsigned j = neighbor.face.offset[index];
                             j < neighbor.face.offset[index + 1]; ++j) {
                            Vec3u f = face[neighbor.face.data[j]];
                            if (f[0] == vertex_index || f[1] == vertex_index ||
                                f[2] == vertex_index) {
                                ++count;
                                include_friction = false;
                            } else {
                                Vec3f c =
                                    distance::point_triangle_distance_coeff<
                                        float, float>(p, eval_x[f[0]],
                                                           eval_x[f[1]],
                                                           eval_x[f[2]]);
                                if (c.maxCoeff() < 1.0f &&
                                    c.minCoeff() > 0.0f) {
                                    continue;
                                } else {
                                    ++count;
                                }
                            }
                        }
                    } else {
                        count = 1u;
                    }
                    if (count) {
                        assert(e.squaredNorm() > sqr(offset));
                        Proximity<3> prox;
                        prox.index = Vec3u(vertex_index, f[0], f[1]);
                        prox.value = Vec3f(1.0f, -c[0], -c[1]);
                        embed_contact_force_hess(
                            prox, vertex, eval_x, fixed_hess_in, fixed_out,
                            force, force_in, dyn_out, ghat, offset, vert_prop,
                            param.barrier, dt, friction, count, param, stage,
                            include_friction, vertex_index);
                        if (stage == 0 && cache.data) {
                            cache.record(vertex_index, index);
                        }
                        return true;
                    }
                }
            }
        }
        return false;
    }
};

struct PointFaceContactForceHessEmbed {

    unsigned vertex_index;
    const Vec<Vec3u> &face;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &vert_prop;
    const Vec<FaceProp> &face_prop;
    const Vec<VertexParam> &vertex_params;
    const Vec<FaceParam> &face_params;
    int stage;
    float dt;
    const ParamSet &param;
    ContactPairCache cache; // detect-once recording (stage 0); {} = disabled

    __device__ bool operator()(unsigned index) {
        Vec3u f = face[index];
        bool either_dyn = vert_prop[vertex_index].fix_index == 0 ||
                          face_prop[index].fixed == false;
        unsigned bid_v = vert_prop[vertex_index].pdrd_body_index;
        unsigned bid_f = vert_prop[f[0]].pdrd_body_index;
        bool same_pdrd_body = bid_v != 0 && bid_v == bid_f;
        if (f[0] != vertex_index && f[1] != vertex_index &&
            f[2] != vertex_index && either_dyn && !same_pdrd_body) {
            const Vec3f &p = eval_x[vertex_index];
            const Vec3f &t0 = eval_x[f[0]];
            const Vec3f &t1 = eval_x[f[1]];
            const Vec3f &t2 = eval_x[f[2]];
            Vec3f c =
                distance::point_triangle_distance_coeff<float, float>(
                    p, t0, t1, t2);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f e = c[0] * (p - t0) + c[1] * (p - t1) + c[2] * (p - t2);
                const VertexParam &vparam =
                    vertex_params[vert_prop[vertex_index].param_index];
                const FaceParam &fparam =
                    face_params[face_prop[index].param_index];
                float offset = vparam.offset + fparam.offset;
                float ghat = 0.5f * (vparam.ghat + fparam.ghat);
                float friction = combine_friction(vparam.friction, fparam.friction, param.friction_mode);
                if (e.squaredNorm() < sqr(ghat + offset)) {
                    assert(e.squaredNorm() > sqr(offset));
                    Proximity<4> prox;
                    prox.index = Vec4u(vertex_index, f[0], f[1], f[2]);
                    prox.value = Vec4f(1.0f, -c[0], -c[1], -c[2]);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, ghat, offset, vert_prop,
                        param.barrier, dt, friction, 1, param, stage, true,
                        vertex_index);
                    if (stage == 0 && cache.data) {
                        cache.record(vertex_index, index);
                    }
                    return true;
                }
            }
        }
        return false;
    }
};

struct EdgeEdgeContactForceHessEmbed {

    unsigned edge_index;
    const Vec<Vec2u> &edge;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    Vec<float> &force;
    const Vec<float> &force_in;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    DynCSRMat &dyn_out;
    const Vec<VertexProp> &vert_prop;
    const Vec<EdgeProp> &edge_prop;
    const Vec<EdgeParam> &edge_params;
    int stage;
    float dt;
    const ParamSet &param;
    ContactPairCache cache; // detect-once recording (stage 0); {} = disabled

    __device__ bool operator()(unsigned index) {
        const Vec2u &e0 = edge[edge_index];
        const Vec2u &e1 = edge[index];
        bool either_dyn = edge_prop[edge_index].fixed == false ||
                          edge_prop[index].fixed == false;
        unsigned bid_a = vert_prop[e0[0]].pdrd_body_index;
        unsigned bid_b = vert_prop[e1[0]].pdrd_body_index;
        bool same_pdrd_body = bid_a != 0 && bid_a == bid_b;
        if (edge_index < index && edge_has_shared_vert(e0, e1) == false &&
            either_dyn && !same_pdrd_body) {
            const Vec3f &p0 = eval_x[e0[0]];
            const Vec3f &p1 = eval_x[e0[1]];
            const Vec3f &q0 = eval_x[e1[0]];
            const Vec3f &q1 = eval_x[e1[1]];
            Vec4f c = distance::edge_edge_distance_coeff<float, float>(
                p0, p1, q0, q1);
            if (c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f) {
                Vec3f x0 = float(c[0]) * p0 + float(c[1]) * p1;
                Vec3f x1 = float(c[2]) * q0 + float(c[3]) * q1;
                Vec3f e = x0 - x1;
                const EdgeParam &eparam_edge =
                    edge_params[edge_prop[edge_index].param_index];
                const EdgeParam &eparam_index =
                    edge_params[edge_prop[index].param_index];
                float offset = eparam_edge.offset + eparam_index.offset;
                float ghat = 0.5f * (eparam_edge.ghat + eparam_index.ghat);
                float friction =
                    combine_friction(eparam_edge.friction, eparam_index.friction, param.friction_mode);
                if (e.squaredNorm() < sqr(ghat + offset)) {
                    assert(e.squaredNorm() > sqr(offset));
                    Proximity<4> prox;
                    prox.index = Vec4u(e0[0], e0[1], e1[0], e1[1]);
                    prox.value = Vec4f(c[0], c[1], -c[2], -c[3]);
                    embed_contact_force_hess(
                        prox, vertex, eval_x, fixed_hess_in, fixed_out, force,
                        force_in, dyn_out, ghat, offset, vert_prop,
                        param.barrier, dt, friction, 1, param, stage, true,
                        edge_index);
                    if (stage == 0 && cache.data) {
                        cache.record(edge_index, index);
                    }
                    return true;
                }
            }
        }
        return false;
    }
};

struct CollisionHessForceEmbedArgs {
    const Vec<Vec3f> collision_mesh_vertex;
    const Vec<Vec3u> collision_mesh_face;
    const Vec<Vec2u> collision_mesh_edge;
    const BVH collision_mesh_face_bvh;
    const BVH collision_mesh_edge_bvh;
    const Vec<AABB> collision_mesh_face_aabb;
    const Vec<AABB> collision_mesh_edge_aabb;
};

struct CollisionMeshVertexFaceContactForceHessEmbed_M2C {

    unsigned vertex_index;
    Vec<float> force;
    const Mat3x3f &local_hess;
    FixedCSRMat &dyn_out;
    const Vec<Vec3u> &collision_mesh_face;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<FaceProp> &static_face_prop;
    const Vec<VertexParam> &dyn_vert_params;
    const Vec<FaceParam> &static_face_params;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (dyn_vert_prop[vertex_index].fix_index == 0 &&
            dyn_vert_prop[vertex_index].mass > 0.0f) {
            const Vec3u &f = collision_mesh_face[index];
            Vec3f p = eval_x[vertex_index];
            Vec3f q = vertex[vertex_index] - p;
            Vec3f t0 = collision_mesh_vertex[f[0]] - p;
            Vec3f t1 = collision_mesh_vertex[f[1]] - p;
            Vec3f t2 = collision_mesh_vertex[f[2]] - p;
            Vec3f zero = Vec3f::Zero();
            Vec3f c = distance::point_triangle_distance_coeff_unclassified<
                float, float>(zero, t0, t1, t2);
            Vec3f y = c[0] * t0 + c[1] * t1 + c[2] * t2;
            Vec3f e = -y;
            const VertexParam &dyn_vparam =
                dyn_vert_params[dyn_vert_prop[vertex_index].param_index];
            const FaceParam &static_fparam =
                static_face_params[static_face_prop[index].param_index];
            float offset = dyn_vparam.offset + static_fparam.offset;
            float ghat = 0.5f * (dyn_vparam.ghat + static_fparam.ghat);
            float friction =
                combine_friction(dyn_vparam.friction, static_fparam.friction, param.friction_mode);
            if (e.squaredNorm() < sqr(offset + ghat)) {
                assert(e.squaredNorm() > sqr(offset));
                Vec3f normal = e.normalized();
                float mass = dyn_vert_prop[vertex_index].mass;
                Vec3f proj_x = y + normal * (offset + ghat);
                float gap_squared = sqr(e.norm() - offset);
                float stiff_k =
                    normal.dot(local_hess * normal) + mass / gap_squared;
                Vec3f f = stiff_k * push::gradient(-proj_x, normal, ghat);
                Mat3x3f H = stiff_k * push::hessian(-proj_x, normal, ghat);
                Friction _friction(f, -q, normal, friction, param.friction_eps);
                f += _friction.gradient();
                H += _friction.hessian();

                utility::atomic_embed_force<1>(Vec1u(vertex_index), f, force);
                utility::atomic_embed_hessian<1>(Vec1u(vertex_index), H,
                                                 dyn_out);
                return true;
            }
        }
        return false;
    }
};

struct CollisionMeshVertexFaceContactForceHessEmbed_C2M {

    const DataSet &data;
    unsigned vertex_index;
    const FixedCSRMat &fixed_hess_in;
    FixedCSRMat &fixed_out;
    Vec<float> force;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<FaceProp> &dyn_face_prop;
    const Vec<VertexProp> &static_vert_prop;
    const Vec<FaceParam> &dyn_face_params;
    const Vec<VertexParam> &static_vert_params;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (!dyn_face_prop[index].fixed && dyn_face_prop[index].mass > 0.0f) {
            const Vec3u &fc = data.mesh.mesh.face[index];
            const Vec3f &y = data.constraint.mesh.vertex[vertex_index];
            const Vec3f t0 = eval_x[fc[0]] - y;
            const Vec3f t1 = eval_x[fc[1]] - y;
            const Vec3f t2 = eval_x[fc[2]] - y;
            const Vec3f s0 = vertex[fc[0]] - y;
            const Vec3f s1 = vertex[fc[1]] - y;
            const Vec3f s2 = vertex[fc[2]] - y;
            const Vec3f zero = Vec3f::Zero();
            const Vec3f c =
                distance::point_triangle_distance_coeff_unclassified<float,
                                                                     float>(
                    zero, t0, t1, t2);
            Vec3f p = c[0] * t0 + c[1] * t1 + c[2] * t2;
            Vec3f q = c[0] * s0 + c[1] * s1 + c[2] * s2;
            const Vec3f &e = p;
            Vec3f dx = p - q;
            const FaceParam &dyn_fparam =
                dyn_face_params[dyn_face_prop[index].param_index];
            const VertexParam &static_vparam =
                static_vert_params[static_vert_prop[vertex_index].param_index];
            float offset = dyn_fparam.offset + static_vparam.offset;
            float ghat = 0.5f * (dyn_fparam.ghat + static_vparam.ghat);
            float friction =
                combine_friction(dyn_fparam.friction, static_vparam.friction, param.friction_mode);
            if (e.squaredNorm() < sqr(offset + ghat)) {
                assert(e.squaredNorm() > sqr(offset));
                Vec3f normal = e.normalized();
                Mat9x9f local_hess = Mat9x9f::Zero();
                float gap_squared = sqr(e.norm() - offset);
                for (unsigned ii = 0; ii < 3; ++ii) {
                    for (unsigned jj = 0; jj < 3; ++jj) {
                        local_hess.block<3, 3>(3 * ii, 3 * jj) =
                            fixed_hess_in(fc[ii], fc[jj]);
                    }
                    local_hess.block<3, 3>(3 * ii, 3 * ii) +=
                        (dyn_vert_prop[fc[ii]].mass / gap_squared) *
                        Mat3x3f::Identity();
                }
                Vec9f normal_ext;
                for (int j = 0; j < 9; ++j) {
                    normal_ext[j] = c[j / 3] * normal[j % 3];
                }
                float stiff_k = (local_hess * normal_ext).dot(normal_ext) /
                                normal_ext.squaredNorm();
                float area = c[0] * dyn_vert_prop[fc[0]].area +
                             c[1] * dyn_vert_prop[fc[1]].area +
                             c[2] * dyn_vert_prop[fc[2]].area;

                Vec3f proj_x = normal * (offset + ghat);
                Vec3f f = stiff_k * push::gradient(p - proj_x, normal, ghat);
                Mat3x3f H = stiff_k * push::hessian(p - proj_x, normal, ghat);
                Friction _friction(f, p - q, normal, friction,
                                   param.friction_eps);
                f += _friction.gradient();
                H += _friction.hessian();
                Mat3x3f ff;
                Mat9x9f HH;
                for (int i = 0; i < 3; ++i) {
                    ff.col(i) = c[i] * f;
                }
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        HH.block<3, 3>(i * 3, j * 3) = c[i] * c[j] * H;
                    }
                }
                utility::atomic_embed_force<3>(fc, ff, force);
                utility::atomic_embed_hessian<3>(fc, HH, fixed_out);
                return true;
            }
        }
        return false;
    }
};

struct CollisionMeshEdgeEdgeContactForceHessEmbed {

    unsigned i;
    const Vec<Vec2u> &mesh_edge;
    Vec<float> &force;
    const Mat6x6f &local_hess;
    FixedCSRMat &dyn_out;
    const Vec<Vec2u> &collision_mesh_edge;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3f> &eval_x;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<EdgeProp> &dyn_edge_prop;
    const Vec<EdgeProp> &static_edge_prop;
    const Vec<EdgeParam> &dyn_edge_params;
    const Vec<EdgeParam> &static_edge_params;
    float dt;
    const ParamSet &param;

    __device__ bool operator()(unsigned index) {
        if (!dyn_edge_prop[i].fixed && dyn_edge_prop[i].mass > 0.0f) {
            const Vec2u &mesh_edge = this->mesh_edge[i];
            const Vec2u &coll_edge = collision_mesh_edge[index];
            Vec3f p0 = eval_x[mesh_edge[0]];
            Vec3f p1 = eval_x[mesh_edge[1]];
            Vec3f q0 = collision_mesh_vertex[coll_edge[0]];
            Vec3f q1 = collision_mesh_vertex[coll_edge[1]];
            Vec3f b0 = vertex[mesh_edge[0]];
            Vec3f b1 = vertex[mesh_edge[1]];
            Vec4f c =
                distance::edge_edge_distance_coeff_unclassified<float, float>(
                    p0, p1, q0, q1);
            Vec3f x = c[0] * p0 + c[1] * p1;
            Vec3f y = c[2] * q0 + c[3] * q1;
            Vec3f z = c[0] * b0 + c[1] * b1;
            Vec3f e = x - y;
            Vec3f dx = x - z;
            const EdgeParam &dyn_eparam =
                dyn_edge_params[dyn_edge_prop[i].param_index];
            const EdgeParam &static_eparam =
                static_edge_params[static_edge_prop[index].param_index];
            float offset = dyn_eparam.offset + static_eparam.offset;
            float ghat = 0.5f * (dyn_eparam.ghat + static_eparam.ghat);
            float friction =
                combine_friction(dyn_eparam.friction, static_eparam.friction, param.friction_mode);
            if (e.squaredNorm() < sqr(offset + ghat)) {
                assert(e.squaredNorm() > sqr(offset));
                Vec3f normal = e.normalized();
                Vec3f proj_x = (offset + ghat) * normal;
                Vec6f normal_ext;
                for (int j = 0; j < 6; ++j) {
                    normal_ext[j] = c[j / 3] * normal[j % 3];
                }
                Mat6x6f mass_diag = Mat6x6f::Zero();
                float gap_squared = sqr(e.norm() - offset);
                mass_diag.block<3, 3>(0, 0) =
                    (dyn_vert_prop[mesh_edge[0]].mass / gap_squared) *
                    Mat3x3f::Identity();
                mass_diag.block<3, 3>(3, 3) =
                    (dyn_vert_prop[mesh_edge[1]].mass / gap_squared) *
                    Mat3x3f::Identity();
                float stiff_k =
                    ((local_hess + mass_diag) * normal_ext).dot(normal_ext) /
                    normal_ext.squaredNorm();
                float area = c[0] * dyn_vert_prop[mesh_edge[0]].area +
                             c[1] * dyn_vert_prop[mesh_edge[1]].area;
                Vec3f f = stiff_k * push::gradient(e - proj_x, normal, ghat);
                Mat3x3f H = stiff_k * push::hessian(e - proj_x, normal, ghat);

                Friction _friction(f, x - z, normal, friction,
                                   param.friction_eps);
                f += _friction.gradient();
                H += _friction.hessian();

                Mat3x2f ff;
                Mat6x6f HH;
                for (int i = 0; i < 2; ++i) {
                    ff.col(i) = c[i] * f;
                }
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        HH.block<3, 3>(i * 3, j * 3) = c[i] * c[j] * H;
                    }
                }
                utility::atomic_embed_force<2>(mesh_edge, ff, force);
                utility::atomic_embed_hessian<2>(mesh_edge, HH, dyn_out);
                return true;
            }
        }
        return false;
    }
};

__device__ unsigned embed_vertex_constraint_force_hessian(
    const DataSet &data, const Vec<Vec3f> &eval_x, Vec<float> &force,
    const FixedCSRMat &fixed_hess_in, FixedCSRMat &fixed_out, float dt,
    const ParamSet &param, unsigned i) {

    const VertexProp &prop = data.prop.vertex[i];
    const VertexParam &vparam = data.param_arrays.vertex[prop.param_index];
    float mass = prop.mass;
    unsigned num_contact = 0;

    // SAND grain spin: a grain (inv_inertia > 0) rolls when the contact-point
    // velocity v_center + omega x r (arm r = -grain_radius*normal, normal the
    // contact push direction) feeds friction, and the torque
    // tau = r x F = (grain_radius*normal) x F spins it. Summed over this
    // vertex's per-vertex contacts (floor/sphere/wall) below, then written to
    // grain_torque[i] for the post-solve integrate (sand_rigid.hpp).
    bool grain = data.grain_inv_inertia[i] > 0.0f;
    float grain_radius = vparam.offset;
    // Implicit (Schur-condensed) rolling: accumulate this grain's floor/sphere
    // Schur blocks locally (one thread per vertex), then write once below.
    // grain_A is the SPD angular block, grain_B the translation<->rotation
    // coupling, and grain_grot the rotational gradient (sand_rigid.hpp condenses
    // them).
    Mat3x3f grain_A_loc = Mat3x3f::Zero();
    Mat3x3f grain_B_loc = Mat3x3f::Zero();
    Vec3f grain_grot_loc = Vec3f::Zero();

    Mat3x3f local_hess = fixed_hess_in(i, i);
    Mat3x3f H = Mat3x3f::Zero();
    Vec3f f = Vec3f::Zero();
    const Vec3f &x = eval_x[i];
    const Vec3f &dx = (x - data.vertex.curr[i]);

    if (prop.fix_index > 0) {
        const FixPair &fix = data.constraint.fix[prop.fix_index - 1];
        const Vec3f &y = fix.position;
        Vec3f w = (x - y);
        float d = w.norm();
        float gap = fix.ghat - d;
        if (fix.kinematic) {
            gap = fmaxf(gap, param.constraint_tol * fix.ghat);
        } else {
            assert(gap >= 0.0f);
        }
        float tmp =
            w.squaredNorm() ? (local_hess * w).dot(w) / w.squaredNorm() : 0.0f;
        float stiff_k = tmp + mass / (gap * gap);
        // Per-pin stiffness scales the moving-pin pull (force + Hessian).
        // Static pins keep stiffness == 1.0 effect (left unscaled).
        if (fix.kinematic) {
            stiff_k *= fix.stiffness;
        }
        f += stiff_k * fix::gradient(x, y);
        H += stiff_k * fix::hessian();
    } else if (mass > 0.0f) {
        // Zero-mass vertices are static solids; skip walls and spheres
        // (both are static themselves; no contact between static solids).
        for (unsigned j = 0; j < data.constraint.sphere.size; ++j) {
            const Sphere &sphere = data.constraint.sphere[j];
            float ghat = sphere.ghat;
            float friction = combine_friction(sphere.friction, vparam.friction, param.friction_mode);
            bool bowl = sphere.bowl;
            bool reverse = sphere.reverse;
            float radius = sphere.radius;
            Vec3f center = sphere.center;
            center = (bowl && (x[1] > center[1]))
                         ? Vec3f(center[0], x[1], center[2])
                         : center;
            float d2 = (x - center).squaredNorm();
            if (sphere.kinematic) {
                if (d2) {
                    Vec3f normal = (x - center).normalized();
                    float eff_radius = reverse ? (radius - ghat) : (radius + ghat);
                    Vec3f target = eff_radius * normal;
                    Vec3f o = (x - center);
                    if (bowl) {
                        reverse = true;
                    }
                    float stiff_k = mass / (ghat * ghat);
                    float r2 = eff_radius * eff_radius;
                    // Pass-through when penetration depth exceeds thickness.
                    float dist = sqrtf(d2);
                    float depth = reverse ? (dist - sphere.radius)
                                          : (sphere.radius - dist);
                    if (sphere.thickness > 0.0f && depth > sphere.thickness) {
                        continue;
                    }
                    if (reverse == true && d2 > r2) {
                        num_contact += 1;
                        f +=
                            stiff_k * push::gradient(o - target, -normal, ghat);
                        H += stiff_k * push::hessian(o - target, -normal, ghat);
                    } else if (reverse == false && d2 < r2) {
                        num_contact += 1;
                        f += stiff_k * push::gradient(o - target, normal, ghat);
                        H += stiff_k * push::hessian(o - target, normal, ghat);
                    }
                }
            } else {
                if (reverse) {
                    radius -= ghat;
                } else {
                    radius += ghat;
                }
                float r2 = radius * radius;
                bool intersected = reverse ? d2 > r2 : d2 < r2;
                if (intersected) {
                    // Pass-through when penetration depth exceeds thickness.
                    float dist = sqrtf(d2);
                    float depth = reverse ? (dist - sphere.radius)
                                          : (sphere.radius - dist);
                    if (sphere.thickness > 0.0f && depth > sphere.thickness) {
                        continue;
                    }
                    num_contact += 1;
                    Vec3f normal = (x - center).normalized();
                    Vec3f projected_x = radius * normal;
                    Vec3f o = (x - center);
                    if (reverse) {
                        normal = -normal;
                    }
                    float gap;
                    if (reverse) {
                        gap = sphere.radius - sqrtf(d2);
                    } else {
                        gap = sqrtf(d2) - sphere.radius;
                    }
                    assert(gap >= 0.0f);
                    float stiff_k =
                        (normal.dot(local_hess * normal) + mass / (gap * gap));
                    Vec3f f_push =
                        stiff_k * push::gradient(o - projected_x, normal, ghat);
                    f += f_push;
                    H += stiff_k * push::hessian(o - projected_x, normal, ghat);

                    // Implicit (Schur-condensed) rolling on the sphere: plain-dx
                    // translation friction (rotation solved implicitly), grain
                    // angular coupling via the Schur blocks. `normal` is the
                    // grain's outward push direction.
                    Friction _friction(f_push, dx, normal, friction,
                                       param.friction_eps);
                    Vec3f fric_grad = _friction.gradient();
                    Mat3x3f fric_hess = _friction.hessian();
                    f += fric_grad;
                    H += fric_hess;
                    if (grain) {
                        accumulate_grain_schur(fric_hess, dx, normal, grain_radius,
                                               grain_A_loc, grain_B_loc,
                                               grain_grot_loc);
                    }
                }
            }
        }

        for (unsigned j = 0; j < data.constraint.floor.size; ++j) {
            const Floor &floor = data.constraint.floor[j];
            float ghat = floor.ghat;
            float friction = combine_friction(floor.friction, vparam.friction, param.friction_mode);
            const Vec3f &up = floor.up;
            Vec3f ground =
                floor.ground + float(ghat) * up;

            Vec3f e = (x - ground);
            if (e.dot(up) < 0.0f) {
                // Pass-through when penetration depth exceeds thickness.
                float depth = -e.dot(up);
                if (floor.thickness > 0.0f && depth > floor.thickness) {
                    continue;
                }
                num_contact += 1;
                Vec3f projected_x = -e.dot(up) * up;
                float gap = (x - floor.ground).dot(up);
                if (floor.kinematic) {
                    gap = fmaxf(gap, param.constraint_tol * ghat);
                }
                assert(gap >= 0.0f);
                float stiff_k = (up.dot(local_hess * up) + mass / (gap * gap));
                Vec3f f_push = stiff_k * push::gradient(-projected_x, up, ghat);
                Mat3x3f H_push =
                    stiff_k * push::hessian(-projected_x, up, ghat);
                f += f_push;
                H += H_push;
                // Implicit (Schur-condensed) rolling: the translation friction
                // uses the PLAIN center slip dx (the rotation coupling is solved
                // implicitly, not lagged into the slip), so f/H here are exactly a
                // normal contact's. The grain's angular DOF enters via the Schur
                // blocks accumulated below.
                Friction _friction(f_push, dx, up, friction, param.friction_eps);
                Vec3f fric_grad = _friction.gradient();
                Mat3x3f fric_hess = _friction.hessian();
                f += fric_grad;
                H += fric_hess;
                if (grain) {
                    accumulate_grain_schur(fric_hess, dx, up, grain_radius,
                                           grain_A_loc, grain_B_loc,
                                           grain_grot_loc);
                }
            }
        }
    }

    // Write this grain's accumulated floor/sphere Schur blocks for the implicit
    // (Schur-condensed) rolling path. One
    // thread per vertex owns grain_A/B/grot[i] (grain-grain uses the disjoint
    // grain_torque/ang_stiff/contact_normal), so a plain write is race-free; the
    // buffers were zeroed at the Newton-iteration top, so a grain with no analytic
    // contact writes zero and the condense/recover skip it.
    if (grain) {
        data.grain_A.data[i] = grain_A_loc;
        data.grain_B.data[i] = grain_B_loc;
        data.grain_grot.data[i] = grain_grot_loc;
    }

    utility::atomic_embed_force<1>(Vec1u(i), f, force);
    utility::atomic_embed_hessian<1>(Vec1u(i), H, fixed_out);

    return num_contact;
}

unsigned embed_contact_force_hessian(const DataSet &data,
                                     const Vec<Vec3f> &eval_x,
                                     Vec<float> force,
                                     const FixedCSRMat &fixed_hess_in,
                                     FixedCSRMat &fixed_out, DynCSRMat &dyn_out,
                                     unsigned &max_nnz_row, float &dyn_consumed,
                                     float dt, const ParamSet &param) {
    const bool *vert_active = get_vert_collision_active();
    const bool *edge_active = get_edge_collision_active();
    const bool *face_active = get_face_collision_active();

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned rod_count = data.rod_count;
    unsigned edge_count = data.mesh.mesh.edge.size;
    unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
    unsigned max_contact_vert = (surface_vert_count > collision_mesh_vert_count)
                                    ? surface_vert_count
                                    : collision_mesh_vert_count;
    const BVH face_bvh = bvh_storage::get_bvh().face;
    const BVH edge_bvh = bvh_storage::get_bvh().edge;
    const BVH vertex_bvh = bvh_storage::get_bvh().vertex;
    const Vec<AABB> face_aabb = storage::face_aabb;
    const Vec<AABB> edge_aabb = storage::edge_aabb;
    const Vec<AABB> vertex_aabb = storage::vertex_aabb;

    Vec<VertexParam> vertex_params = data.param_arrays.vertex;
    Vec<EdgeParam> edge_params = data.param_arrays.edge;
    Vec<FaceParam> face_params = data.param_arrays.face;

    buffer::MemoryPool &pool = buffer::get();
    auto contact_force = pool.get<float>(3 * surface_vert_count);
    auto num_contact_vtf = pool.get<unsigned>(max_contact_vert);
    auto num_contact_ee = pool.get<unsigned>(edge_count);

    num_contact_vtf.clear(0);
    num_contact_ee.clear(0);
    contact_force.clear(0.0f);

    Vec<float> contact_force_vec = contact_force.as_vec();
    Vec<unsigned> num_contact_vtf_vec = num_contact_vtf.as_vec();
    Vec<unsigned> num_contact_ee_vec = num_contact_ee.as_vec();

    // Detect-once caches (one per contact type). Stage 0 records every active
    // primitive pair into these flat lists; stage 1 fills them directly,
    // skipping a second BVH traversal (the dominant assembly cost). A shared
    // overflow flag triggers a full-BVH fallback in stage 1, so correctness
    // never depends on the capacity.
    const unsigned CC_CAP = 1u << 21; // 2M pairs per type (16 MB)
    auto pf_buf = pool.get<Vec2u>(CC_CAP); // point-face
    auto pe_buf = pool.get<Vec2u>(CC_CAP); // point-edge
    auto pp_buf = pool.get<Vec2u>(CC_CAP); // point-point
    auto ee_buf = pool.get<Vec2u>(CC_CAP); // edge-edge
    auto cc_cnt = pool.get<unsigned>(4);   // per-type active counts
    auto cc_overflow = pool.get<unsigned>(1);
    cc_cnt.clear(0);
    cc_overflow.clear(0);
    Vec<Vec2u> pf_vec = pf_buf.as_vec(), pe_vec = pe_buf.as_vec(),
               pp_vec = pp_buf.as_vec(), ee_vec = ee_buf.as_vec();
    Vec<unsigned> cc_cnt_vec = cc_cnt.as_vec();
    Vec<unsigned> cc_overflow_vec = cc_overflow.as_vec();
    ContactPairCache pf_cache{pf_vec.data, cc_cnt_vec.data + 0,
                              cc_overflow_vec.data, CC_CAP};
    ContactPairCache pe_cache{pe_vec.data, cc_cnt_vec.data + 1,
                              cc_overflow_vec.data, CC_CAP};
    ContactPairCache pp_cache{pp_vec.data, cc_cnt_vec.data + 2,
                              cc_overflow_vec.data, CC_CAP};
    ContactPairCache ee_cache{ee_vec.data, cc_cnt_vec.data + 3,
                              cc_overflow_vec.data, CC_CAP};

    for (int stage = 0; stage < 2; ++stage) {
        if (stage == 0) {
            dyn_out.start_rebuild_buffer();
        }

        // At the fill stage, replay the detect-once caches (no second BVH
        // traversal) unless stage 0 overflowed a cache, in which case fall back
        // to a full BVH fill so correctness never depends on the capacity.
        bool replay = false;
        unsigned cc_n[4] = {0, 0, 0, 0};
        if (stage == 1) {
            unsigned of = 0;
            CUDA_HANDLE_ERROR(cudaMemcpy(&of, cc_overflow_vec.data,
                                         sizeof(unsigned),
                                         cudaMemcpyDeviceToHost));
            CUDA_HANDLE_ERROR(cudaMemcpy(cc_n, cc_cnt_vec.data,
                                         4 * sizeof(unsigned),
                                         cudaMemcpyDeviceToHost));
            replay = (of == 0);
        }

        if (replay) {
            // ---- point-contact replays (stage 1 fill, no BVH) ----
            if (cc_n[0]) {
                DISPATCH_START(cc_n[0])
                [data, eval_x, contact_force_vec, force, fixed_hess_in,
                 fixed_out, dyn_out, vertex_params, face_params, dt, param,
                 pf_vec] __device__(unsigned e) mutable {
                    PointFaceContactForceHessEmbed embed = {
                        pf_vec[e][0],      data.mesh.mesh.face,
                        data.vertex.curr,  eval_x,
                        contact_force_vec, force,
                        fixed_hess_in,     fixed_out,
                        dyn_out,           data.prop.vertex,
                        data.prop.face,    vertex_params,
                        face_params,       1,
                        dt,                param};
                    embed(pf_vec[e][1]);
                } DISPATCH_END;
            }
            if (cc_n[1]) {
                DISPATCH_START(cc_n[1])
                [data, eval_x, contact_force_vec, force, fixed_hess_in,
                 fixed_out, dyn_out, vertex_params, edge_params, dt, param,
                 pe_vec] __device__(unsigned e) mutable {
                    PointEdgeContactForceHessEmbed embed = {
                        pe_vec[e][0],            data.mesh.mesh.edge,
                        data.mesh.mesh.face,     data.mesh.neighbor.edge,
                        data.vertex.curr,        eval_x,
                        contact_force_vec,       force,
                        fixed_hess_in,           fixed_out,
                        dyn_out,                 data.prop.vertex,
                        data.prop.edge,          vertex_params,
                        edge_params,             1,
                        dt,                      param};
                    embed(pe_vec[e][1]);
                } DISPATCH_END;
            }
            if (cc_n[2]) {
                DISPATCH_START(cc_n[2])
                [data, eval_x, rod_count, contact_force_vec, force,
                 fixed_hess_in, fixed_out, dyn_out, vertex_params, dt, param,
                 pp_vec] __device__(unsigned e) mutable {
                    PointPointContactForceHessEmbed embed = {
                        pp_vec[e][0],              rod_count,
                        data.mesh.mesh.edge,       data.mesh.mesh.face,
                        data.mesh.neighbor.vertex, data.vertex.curr,
                        eval_x,                    contact_force_vec,
                        force,                     fixed_hess_in,
                        fixed_out,                 dyn_out,
                        data.prop.vertex,          vertex_params,
                        1,                         dt,
                        param,                     data};
                    embed(pp_vec[e][1]);
                } DISPATCH_END;
            }
        } else {
            // Stage 0 detection (records the caches) and the stage-1 overflow
            // fallback share this BVH dispatch: recording is gated on stage 0,
            // so with stage 1 it simply fills.
            DISPATCH_START(surface_vert_count)
            [data, eval_x, rod_count, contact_force_vec, force, fixed_hess_in,
             fixed_out, dyn_out, face_bvh, face_aabb, edge_bvh, edge_aabb,
             vertex_bvh, vertex_aabb, num_contact_vtf_vec, vertex_params,
             edge_params, face_params, stage, dt, vert_active, param, pf_cache,
             pe_cache, pp_cache] __device__(unsigned i) mutable {
                unsigned count(0);
                const VertexParam &vparam =
                    vertex_params[data.prop.vertex[i].param_index];
                float ext_eps = 0.5f * vparam.ghat + vparam.offset;
                AABB pt_aabb = aabb::make(eval_x[i], ext_eps);
                if (vert_active && !vert_active[i]) pt_aabb.active = false;

                PointFaceContactForceHessEmbed embed_0 = {
                    i,                 data.mesh.mesh.face,
                    data.vertex.curr,  eval_x,
                    contact_force_vec, force,
                    fixed_hess_in,     fixed_out,
                    dyn_out,           data.prop.vertex,
                    data.prop.face,    vertex_params,
                    face_params,       stage,
                    dt,                param,
                    pf_cache};
                AABB_AABB_Tester<PointFaceContactForceHessEmbed> op_0(embed_0);
                count += aabb::query(face_bvh, face_aabb, op_0, pt_aabb);

                PointEdgeContactForceHessEmbed embed_1 = {
                    i,                   data.mesh.mesh.edge,
                    data.mesh.mesh.face, data.mesh.neighbor.edge,
                    data.vertex.curr,    eval_x,
                    contact_force_vec,   force,
                    fixed_hess_in,       fixed_out,
                    dyn_out,             data.prop.vertex,
                    data.prop.edge,      vertex_params,
                    edge_params,         stage,
                    dt,                  param,
                    pe_cache};
                AABB_AABB_Tester<PointEdgeContactForceHessEmbed> op_1(embed_1);
                count += aabb::query(edge_bvh, edge_aabb, op_1, pt_aabb);

                PointPointContactForceHessEmbed embed_2 = {
                    i,                         rod_count,
                    data.mesh.mesh.edge,       data.mesh.mesh.face,
                    data.mesh.neighbor.vertex, data.vertex.curr,
                    eval_x,                    contact_force_vec,
                    force,                     fixed_hess_in,
                    fixed_out,                 dyn_out,
                    data.prop.vertex,          vertex_params,
                    stage,                     dt,
                    param,                     data,
                    pp_cache};
                AABB_AABB_Tester<PointPointContactForceHessEmbed> op_2(embed_2);
                count += aabb::query(vertex_bvh, vertex_aabb, op_2, pt_aabb);
                if (stage == 0) {
                    num_contact_vtf_vec[i] += count;
                }
            } DISPATCH_END;
        }

        if (replay) {
            // ---- edge-edge replay (stage 1 fill, no BVH) ----
            if (cc_n[3]) {
                DISPATCH_START(cc_n[3])
                [data, eval_x, contact_force_vec, force, fixed_hess_in,
                 fixed_out, dyn_out, edge_params, dt, param,
                 ee_vec] __device__(unsigned e) mutable {
                    EdgeEdgeContactForceHessEmbed embed = {
                        ee_vec[e][0],      data.mesh.mesh.edge,
                        data.vertex.curr,  eval_x,
                        contact_force_vec, force,
                        fixed_hess_in,     fixed_out,
                        dyn_out,           data.prop.vertex,
                        data.prop.edge,    edge_params,
                        1,                 dt,
                        param};
                    embed(ee_vec[e][1]);
                } DISPATCH_END;
            }
        } else {
            // Stage 0 detection (records ee_cache) and the stage-1 overflow
            // fallback share this BVH dispatch (recording is gated on stage 0).
            DISPATCH_START(edge_count)
            [data, eval_x, contact_force_vec, force, fixed_hess_in, fixed_out,
             dyn_out, edge_bvh, edge_aabb, num_contact_ee_vec, edge_params,
             stage, dt, edge_active, param, ee_cache] __device__(unsigned i)
                mutable {
                Vec2u edge = data.mesh.mesh.edge[i];
                const EdgeParam &eparam =
                    edge_params[data.prop.edge[i].param_index];
                float ext_eps = 0.5f * eparam.ghat + eparam.offset;
                AABB aabb =
                    aabb::make(eval_x[edge[0]], eval_x[edge[1]], ext_eps);
                if (edge_active && !edge_active[i]) aabb.active = false;
                EdgeEdgeContactForceHessEmbed embed = {
                    i,                 data.mesh.mesh.edge,
                    data.vertex.curr,  eval_x,
                    contact_force_vec, force,
                    fixed_hess_in,     fixed_out,
                    dyn_out,           data.prop.vertex,
                    data.prop.edge,    edge_params,
                    stage,             dt,
                    param,             ee_cache};
                AABB_AABB_Tester<EdgeEdgeContactForceHessEmbed> op(embed);
                unsigned count = aabb::query(edge_bvh, edge_aabb, op, aabb);
                if (stage == 0) {
                    num_contact_ee_vec[i] += count;
                }
            } DISPATCH_END;
        }

        if (stage == 0) {
            // After the dry pass, recompute the dynamic contact-matrix memory
            // layout so the fill pass can assemble into it.
            dyn_out.finish_rebuild_buffer(max_nnz_row, dyn_consumed);
        } else {
            // After the fill pass, compress the contact matrix (dedup entries).
            if (dyn_consumed) {
                dyn_out.finalize();
            }
        }
    }

    DISPATCH_START(3 * surface_vert_count)
    [force, contact_force_vec] __device__(unsigned i) mutable {
        force[i] += contact_force_vec[i];
    } DISPATCH_END;

    unsigned n_contact =
        kernels::sum_array(num_contact_vtf.data, num_contact_vtf.size) +
        kernels::sum_array(num_contact_ee.data, num_contact_ee.size);

    // PooledVec buffers auto-release when exiting function
    return n_contact;
}

unsigned embed_constraint_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> force,
                                        const FixedCSRMat &fixed_hess_in,
                                        FixedCSRMat &fixed_out, float dt,
                                        const ParamSet &param) {

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned edge_count = data.mesh.mesh.edge.size;
    unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
    unsigned max_contact_vert = (surface_vert_count > collision_mesh_vert_count)
                                    ? surface_vert_count
                                    : collision_mesh_vert_count;

    const BVH &face_bvh = bvh_storage::get_bvh().face;
    const Vec<AABB> face_aabb = storage::face_aabb;

    Vec<VertexParam> vertex_params = data.param_arrays.vertex;
    Vec<EdgeParam> edge_params = data.param_arrays.edge;
    Vec<FaceParam> face_params = data.param_arrays.face;
    Vec<VertexParam> collision_vertex_params =
        data.constraint.mesh.param_arrays.vertex;
    Vec<EdgeParam> collision_edge_params =
        data.constraint.mesh.param_arrays.edge;
    Vec<FaceParam> collision_face_params =
        data.constraint.mesh.param_arrays.face;

    buffer::MemoryPool &pool = buffer::get();
    auto num_contact_vtf = pool.get<unsigned>(max_contact_vert);
    auto num_contact_ee = pool.get<unsigned>(edge_count);

    num_contact_vtf.clear(0);
    num_contact_ee.clear(0);

    Vec<unsigned> num_contact_vtf_vec = num_contact_vtf.as_vec();
    Vec<unsigned> num_contact_ee_vec = num_contact_ee.as_vec();

    DISPATCH_START(surface_vert_count)
    [data, eval_x, force, fixed_hess_in, fixed_out, dt, num_contact_vtf_vec,
     param] __device__(unsigned i) mutable {
        num_contact_vtf_vec[i] += embed_vertex_constraint_force_hessian(
            data, eval_x, force, fixed_hess_in, fixed_out, dt, param, i);
    } DISPATCH_END;

    const bool *vert_active = get_vert_collision_active();
    const bool *edge_active = get_edge_collision_active();

    if (!param.disable_contact) {
        CollisionHessForceEmbedArgs args = {
            data.constraint.mesh.vertex,      data.constraint.mesh.face,
            data.constraint.mesh.edge,        bvh_storage::get_collision_mesh_bvh().face,
            bvh_storage::get_collision_mesh_bvh().edge,    storage::collision_mesh_face_aabb,
            storage::collision_mesh_edge_aabb};

        DISPATCH_START(surface_vert_count)
        [data, eval_x, force, fixed_hess_in, fixed_out, args, dt,
         num_contact_vtf_vec, vertex_params, collision_face_params,
         vert_active, param] __device__(unsigned i) mutable {
            Mat3x3f local_hess = fixed_hess_in(i, i);
            const VertexProp &prop = data.prop.vertex[i];
            const VertexParam &vparam = vertex_params[prop.param_index];
            CollisionMeshVertexFaceContactForceHessEmbed_M2C embed = {
                i,
                force,
                local_hess,
                fixed_out,
                args.collision_mesh_face,
                args.collision_mesh_vertex,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                data.constraint.mesh.prop.face,
                vertex_params,
                collision_face_params,
                dt,
                param};
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            AABB pt_aabb = aabb::make(eval_x[i], ext_eps);
            if (vert_active && !vert_active[i]) pt_aabb.active = false;
            AABB_AABB_Tester<CollisionMeshVertexFaceContactForceHessEmbed_M2C>
                op(embed);
            num_contact_vtf_vec[i] +=
                aabb::query(args.collision_mesh_face_bvh,
                            args.collision_mesh_face_aabb, op, pt_aabb);
        } DISPATCH_END;

        DISPATCH_START(data.constraint.mesh.vertex.size)
        [data, eval_x, force, fixed_hess_in, fixed_out, args,
         num_contact_vtf_vec, face_bvh, face_aabb, face_params,
         collision_vertex_params, dt, param] __device__(unsigned i) mutable {
            const VertexProp &prop = data.constraint.mesh.prop.vertex[i];
            const VertexParam &vparam =
                collision_vertex_params[prop.param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            CollisionMeshVertexFaceContactForceHessEmbed_C2M embed = {
                data,
                i,
                fixed_hess_in,
                fixed_out,
                force,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                data.prop.face,
                data.constraint.mesh.prop.vertex,
                face_params,
                collision_vertex_params,
                dt,
                param};
            AABB_AABB_Tester<CollisionMeshVertexFaceContactForceHessEmbed_C2M>
                op(embed);
            num_contact_vtf_vec[i] += aabb::query(
                face_bvh, face_aabb, op,
                aabb::make(data.constraint.mesh.vertex[i], ext_eps));
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, eval_x, force, fixed_hess_in, fixed_out, args,
         num_contact_ee_vec, edge_params, collision_edge_params, dt,
         edge_active, param] __device__(unsigned i) mutable {
            const Vec2u &edge = data.mesh.mesh.edge[i];
            const EdgeParam &eparam =
                edge_params[data.prop.edge[i].param_index];
            float ext_eps = 0.5f * eparam.ghat + eparam.offset;
            Mat6x6f local_hess = Mat6x6f::Zero();
            for (unsigned ii = 0; ii < 2; ++ii) {
                for (unsigned jj = 0; jj < 2; ++jj) {
                    local_hess.block<3, 3>(3 * ii, 3 * jj) =
                        fixed_hess_in(edge[ii], edge[jj]);
                }
            }
            CollisionMeshEdgeEdgeContactForceHessEmbed embed = {
                i,
                data.mesh.mesh.edge,
                force,
                local_hess,
                fixed_out,
                data.constraint.mesh.edge,
                data.constraint.mesh.vertex,
                data.vertex.curr,
                eval_x,
                data.prop.vertex,
                data.prop.edge,
                data.constraint.mesh.prop.edge,
                edge_params,
                collision_edge_params,
                dt,
                param};
            AABB aabb = aabb::make(eval_x[edge[0]], eval_x[edge[1]], ext_eps);
            if (edge_active && !edge_active[i]) aabb.active = false;
            AABB_AABB_Tester<CollisionMeshEdgeEdgeContactForceHessEmbed> op(
                embed);
            num_contact_ee_vec[i] +=
                aabb::query(args.collision_mesh_edge_bvh,
                            args.collision_mesh_edge_aabb, op, aabb);
        } DISPATCH_END;
    }

    unsigned n_contact =
        kernels::sum_array(num_contact_vtf_vec.data, num_contact_vtf_vec.size) +
        kernels::sum_array(num_contact_ee_vec.data, num_contact_ee_vec.size);

    return n_contact;
}

struct CollisionMeshPointFaceCCD_M2C {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<Vec3u> &collision_mesh_face;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<VertexProp> &dyn_vert_prop;
    const Vec<FaceProp> &static_face_prop;
    const Vec<VertexParam> &dyn_vert_params;
    const Vec<FaceParam> &static_face_params;
    unsigned vertex_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        // Zero-mass vertices (static solids, incl. moving-static shells)
        // never collide with the collision mesh (itself a static solid).
        if (dyn_vert_prop[vertex_index].mass == 0.0f) return false;
        const Vec3u &f = collision_mesh_face[index];
        const Vec3f &t0 = collision_mesh_vertex[f[0]];
        const Vec3f &t1 = collision_mesh_vertex[f[1]];
        const Vec3f &t2 = collision_mesh_vertex[f[2]];
        const Vec3f &p0 = x0[vertex_index];
        const Vec3f &p1 = x1[vertex_index];
        const VertexParam &dyn_vparam =
            dyn_vert_params[dyn_vert_prop[vertex_index].param_index];
        const FaceParam &static_fparam =
            static_face_params[static_face_prop[index].param_index];
        float offset = dyn_vparam.offset + static_fparam.offset;
        float result = accd::point_triangle_ccd(p0, p1, t0, t1, t2, t0, t1, t2,
                                                offset, param);
        if (result < param.line_search_max_t) {
            toi = fminf(toi, result);
            assert(toi > 0.0f);
            return true;
        }
        return false;
    }
};

struct CollisionMeshPointFaceCCD_C2M {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<FaceProp> &dyn_face_prop;
    const Vec<VertexProp> &static_vert_prop;
    const Vec<FaceParam> &dyn_face_params;
    const Vec<VertexParam> &static_vert_params;
    unsigned vertex_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        if (!dyn_face_prop[index].fixed && dyn_face_prop[index].mass > 0.0f) {
            const Vec3u &f = face[index];
            const Vec3f &t00 = x0[f[0]];
            const Vec3f &t01 = x0[f[1]];
            const Vec3f &t02 = x0[f[2]];
            const Vec3f &t10 = x1[f[0]];
            const Vec3f &t11 = x1[f[1]];
            const Vec3f &t12 = x1[f[2]];
            const Vec3f &p = collision_mesh_vertex[vertex_index];
            const FaceParam &dyn_fparam =
                dyn_face_params[dyn_face_prop[index].param_index];
            const VertexParam &static_vparam =
                static_vert_params[static_vert_prop[vertex_index].param_index];
            float offset = dyn_fparam.offset + static_vparam.offset;
            float result = accd::point_triangle_ccd(p, p, t00, t01, t02, t10,
                                                    t11, t12, offset, param);
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

struct CollisionMeshEdgeEdgeCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec2u> &edge;
    const Vec<Vec2u> &collision_mesh_edge;
    const Vec<Vec3f> &collision_mesh_vertex;
    const Vec<EdgeProp> &dyn_edge_prop;
    const Vec<EdgeProp> &static_edge_prop;
    const Vec<EdgeParam> &dyn_edge_params;
    const Vec<EdgeParam> &static_edge_params;
    unsigned edge_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        if (!dyn_edge_prop[edge_index].fixed && dyn_edge_prop[edge_index].mass > 0.0f) {
            const Vec2u &e0 = edge[edge_index];
            const Vec2u &e1 = collision_mesh_edge[index];
            const Vec3f &p00 = x0[e0[0]];
            const Vec3f &p01 = x0[e0[1]];
            const Vec3f &p10 = x1[e0[0]];
            const Vec3f &p11 = x1[e0[1]];
            const Vec3f &q0 = collision_mesh_vertex[e1[0]];
            const Vec3f &q1 = collision_mesh_vertex[e1[1]];
            const EdgeParam &dyn_eparam =
                dyn_edge_params[dyn_edge_prop[edge_index].param_index];
            const EdgeParam &static_eparam =
                static_edge_params[static_edge_prop[index].param_index];
            float offset = dyn_eparam.offset + static_eparam.offset;
            float result = accd::edge_edge_ccd(p00, p01, q0, q1, p10, p11, q0,
                                               q1, offset, param);
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

struct PointFaceCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec3u> &face;
    const Vec<VertexProp> &vertex_prop;
    const Vec<FaceProp> &face_prop;
    const Vec<VertexParam> &vertex_params;
    const Vec<FaceParam> &face_params;
    unsigned vertex_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        const Vec3u &f = face[index];
        bool either_dyn = vertex_prop[vertex_index].fix_index == 0 ||
                          face_prop[index].fixed == false;
        unsigned bid_v = vertex_prop[vertex_index].pdrd_body_index;
        unsigned bid_f = vertex_prop[f[0]].pdrd_body_index;
        bool same_pdrd_body = bid_v != 0 && bid_v == bid_f;
        if (either_dyn && !same_pdrd_body) {
            int dup_i = -1;
            for (int i = 0; i < 3; ++i) {
                if (f[i] == vertex_index) {
                    dup_i = i;
                    break;
                }
            }
            float result = param.line_search_max_t;
            if (dup_i == -1) {
                const VertexParam &vparam =
                    vertex_params[vertex_prop[vertex_index].param_index];
                const FaceParam &fparam =
                    face_params[face_prop[index].param_index];
                float offset = vparam.offset + fparam.offset;
                const Vec3f &p0 = x0[vertex_index];
                const Vec3f &p1 = x1[vertex_index];
                const Vec3f &t00 = x0[f[0]];
                const Vec3f &t01 = x0[f[1]];
                const Vec3f &t02 = x0[f[2]];
                const Vec3f &t10 = x1[f[0]];
                const Vec3f &t11 = x1[f[1]];
                const Vec3f &t12 = x1[f[2]];
                result = accd::point_triangle_ccd(p0, p1, t00, t01, t02, t10,
                                                  t11, t12, offset, param);
            } else {
                const VertexParam &vparam =
                    vertex_params[vertex_prop[vertex_index].param_index];
                float offset = 2.0f * vparam.offset;
                unsigned i = dup_i;
                unsigned j = (i + 1) % 3;
                unsigned k = (i + 2) % 3;
                const Vec3f &p0 = x0[f[i]];
                const Vec3f &p1 = x1[f[i]];
                const Vec3f &q00 = x0[f[j]];
                const Vec3f &q10 = x1[f[j]];
                const Vec3f &q01 = x0[f[k]];
                const Vec3f &q11 = x1[f[k]];
                result = accd::point_edge_ccd(p0, p1, q00, q01, q10, q11,
                                              offset, param);
            }
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

struct EdgeEdgeCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<Vec2u> &edge;
    const Vec<EdgeProp> &edge_prop;
    const Vec<EdgeParam> &edge_params;
    const Vec<VertexProp> &vert_prop;
    unsigned edge_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        bool either_dyn = edge_prop[edge_index].fixed == false ||
                          edge_prop[index].fixed == false;
        const Vec2u &e0 = edge[edge_index];
        const Vec2u &e1 = edge[index];
        unsigned bid_a = vert_prop[e0[0]].pdrd_body_index;
        unsigned bid_b = vert_prop[e1[0]].pdrd_body_index;
        bool same_pdrd_body = bid_a != 0 && bid_a == bid_b;
        if (edge_index < index && either_dyn && !same_pdrd_body) {
            float result = param.line_search_max_t;
            const EdgeParam &eparam_edge =
                edge_params[edge_prop[edge_index].param_index];
            const EdgeParam &eparam_index =
                edge_params[edge_prop[index].param_index];
            float offset = eparam_edge.offset + eparam_index.offset;
            if (!edge_has_shared_vert(e0, e1)) {
                const Vec3f &p00 = x0[e0[0]];
                const Vec3f &p01 = x0[e0[1]];
                const Vec3f &q00 = x0[e1[0]];
                const Vec3f &q01 = x0[e1[1]];
                const Vec3f &p10 = x1[e0[0]];
                const Vec3f &p11 = x1[e0[1]];
                const Vec3f &q10 = x1[e1[0]];
                const Vec3f &q11 = x1[e1[1]];
                result = accd::edge_edge_ccd(p00, p01, q00, q01, p10, p11, q10,
                                             q11, offset, param);
            } else {
                const Vec2u ij[] = {Vec2u(0, 0), Vec2u(0, 1), Vec2u(1, 0),
                                    Vec2u(1, 1)};
                for (unsigned k = 0; k < 4; ++k) {
                    unsigned i = ij[k][0];
                    unsigned j = ij[k][1];
                    if (e0[i] == e1[j]) {
                        unsigned idx0 = e0[i];
                        unsigned idx1 = e0[1 - i];
                        unsigned idx2 = e1[1 - j];
                        const Vec3f &q00 = x0[idx0];
                        const Vec3f &q10 = x1[idx0];
                        const Vec3f &q01 = x0[idx1];
                        const Vec3f &q11 = x1[idx1];
                        const Vec3f &q02 = x0[idx2];
                        const Vec3f &q12 = x1[idx2];
                        float toi_0 = accd::point_edge_ccd(
                            q01, q11, q00, q02, q10, q12, offset, param);
                        float toi_1 = accd::point_edge_ccd(
                            q02, q12, q00, q01, q10, q11, offset, param);
                        result = fminf(toi_0, toi_1);
                        break;
                    }
                }
            }
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

// Continuous collision detection for two free grains (point-point). A faceless
// SAND cloud has no point-point CCD candidate among the face/edge primitives,
// so without this the line search cannot bound a Newton step that pushes two
// grains through each other; under pile load that drives a pair inside the
// contact wall and trips the barrier precondition. Mirrors PointFaceCCD and
// shares the same pair filters as PointPointContactForceHessEmbed
// (upper-triangular, at least one free vertex, not the same PDRD body).
struct PointPointCCD {
    const Vec<Vec3f> &x0;
    const Vec<Vec3f> &x1;
    const Vec<VertexProp> &vertex_prop;
    const Vec<VertexParam> &vertex_params;
    unsigned vertex_index;
    float &toi;
    const ParamSet &param;
    __device__ bool operator()(unsigned index) {
        bool either_dyn = vertex_prop[vertex_index].fix_index == 0 ||
                          vertex_prop[index].fix_index == 0;
        unsigned bid_a = vertex_prop[vertex_index].pdrd_body_index;
        unsigned bid_b = vertex_prop[index].pdrd_body_index;
        bool same_pdrd_body = bid_a != 0 && bid_a == bid_b;
        if (index < vertex_index && either_dyn && !same_pdrd_body) {
            const VertexParam &vparam_a =
                vertex_params[vertex_prop[vertex_index].param_index];
            const VertexParam &vparam_b =
                vertex_params[vertex_prop[index].param_index];
            float offset = vparam_a.offset + vparam_b.offset;
            float result = accd::point_point_ccd(x0[vertex_index], x1[vertex_index],
                                                 x0[index], x1[index], offset, param);
            if (result < param.line_search_max_t) {
                toi = fminf(toi, result);
                assert(toi > 0.0f);
                return true;
            }
        }
        return false;
    }
};

__device__ void vertex_constraint_line_search(const DataSet &data,
                                              const Vec<Vec3f> &y0,
                                              const Vec<Vec3f> &y1,
                                              Vec<float> toi_vert,
                                              ParamSet param, unsigned i) {
    const Vec3f x1 =
        float(param.line_search_max_t) * (y1[i] - y0[i]) + y0[i];
    const Vec3f &x0 = y0[i];
    const VertexProp &prop = data.prop.vertex[i];
    if (prop.fix_index > 0) {
        const FixPair &fix = data.constraint.fix[prop.fix_index - 1];
        if (fix.kinematic == false) {
            const Vec3f &position = fix.position;
            float r0 = (x0 - position).norm();
            float r1 = (x1 - position).norm();
            assert(r0 < fix.ghat);
            float r = fix.ghat;
            if (r1 > r) {
                // (1.0f - t) r0 + t r1 = r
                // r0 - t r0 + t r1 = r
                // t (r1 - r0) = r - r0
                // t = (r - r0) / (r1 - r0)
                float denom = r1 - r0;
                if (denom) {
                    float t = (r - r0) / denom;
                    toi_vert[i] =
                        fminf(toi_vert[i], param.line_search_max_t * t);
                    assert(toi_vert[i] > 0.0f);
                }
            }
        }
    } else if (prop.mass > 0.0f) {
        // Zero-mass vertices are static solids; skip walls and spheres
        // (both are static themselves).
        for (unsigned j = 0; j < data.constraint.sphere.size; ++j) {
            const Sphere &sphere = data.constraint.sphere[j];
            if (!sphere.kinematic) {
                bool reverse = sphere.reverse;
                bool bowl = sphere.bowl;
                const Vec3f &center = sphere.center;
                const Vec3f center0 = (bowl && (x0[1] > center[1]))
                                           ? Vec3f(center[0], x0[1], center[2])
                                           : center;
                const Vec3f center1 = (bowl && (x1[1] > center[1]))
                                           ? Vec3f(center[0], x1[1], center[2])
                                           : center;
                float r = sphere.radius;
                float r0 = (x0 - center0).norm();
                float r1 = (x1 - center1).norm();
                // Skip if x0 is already embedded past the thickness cutoff
                // (also loosens the "must be on feasible side" invariant so
                // an ill-placed initial vertex doesn't crash CCD).
                float depth0 = reverse ? (r0 - r) : (r - r0);
                if (sphere.thickness > 0.0f && depth0 > sphere.thickness) {
                    continue;
                }
                bool intersected = (r0 - r) * (r1 - r) <= 0.0f;
                if (intersected) {
                    // (1.0f - t) r0 + t r1 = r
                    // r0 - t r0 + t r1 = r
                    // t (r1 - r0) = radius - r0
                    // t = (r - r0) / (r1 - r0)
                    float t = (r - r0) / (r1 - r0);
                    toi_vert[i] =
                        fminf(toi_vert[i], param.line_search_max_t * t);
                    assert(toi_vert[i] > 0.0f);
                }
            }
        }

        for (unsigned j = 0; j < data.constraint.floor.size; ++j) {
            const Floor &floor = data.constraint.floor[j];
            if (!floor.kinematic) {
                const Vec3f &up = floor.up;
                const Vec3f &ground = floor.ground;
                float h0 = up.dot((x0 - ground));
                float h1 = up.dot((x1 - ground));
                // Skip if x0 is already below the wall past the thickness
                // cutoff (replaces a hard assert; if the vertex is that
                // far inside, we treat it as pass-through).
                if (floor.thickness > 0.0f && -h0 > floor.thickness) {
                    continue;
                }
                if (h1 < 0.0f) {
                    // (1.0f - t) h0 + t h1 = 0
                    // h0 - t h0 + t h1 = 0
                    // t (h1 - h0) = - h0
                    // t = - h0 / (h1 - h0)
                    float t = -h0 / (h1 - h0);
                    toi_vert[i] =
                        fminf(toi_vert[i], param.line_search_max_t * t);
                    assert(toi_vert[i] > 0.0f);
                }
            }
        }
    }
}

float line_search(const DataSet &data, const Vec<Vec3f> &x0,
                  const Vec<Vec3f> &x1, const ParamSet &param) {

    const MeshInfo &mesh = data.mesh;
    const BVHSet &bvhset = bvh_storage::get_bvh();

    unsigned surface_vert_count = data.surface_vert_count;
    unsigned edge_count = mesh.mesh.edge.size;
    unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
    unsigned max_contact_vert = (surface_vert_count > collision_mesh_vert_count)
                                    ? surface_vert_count
                                    : collision_mesh_vert_count;

    const BVH &face_bvh = bvhset.face;
    const BVH &edge_bvh = bvhset.edge;
    const BVH &collision_mesh_face_bvh = bvh_storage::get_collision_mesh_bvh().face;
    const BVH &collision_mesh_edge_bvh = bvh_storage::get_collision_mesh_bvh().edge;
    const Vec<Vec2u> &collision_mesh_edge = data.constraint.mesh.edge;
    const Vec<Vec3u> &collision_mesh_face = data.constraint.mesh.face;
    const Vec<Vec3f> &collision_mesh_vertex = data.constraint.mesh.vertex;
    const Vec<AABB> face_aabb = storage::face_aabb;
    const Vec<AABB> edge_aabb = storage::edge_aabb;
    const Vec<AABB> collision_mesh_face_aabb =
        storage::collision_mesh_face_aabb;
    const Vec<AABB> collision_mesh_edge_aabb =
        storage::collision_mesh_edge_aabb;

    buffer::MemoryPool &pool = buffer::get();
    auto intersection_flag = pool.get<char>(edge_count);
    auto toi_vtf = pool.get<float>(max_contact_vert);
    auto toi_ee = pool.get<float>(edge_count);

    intersection_flag.clear(0);
    toi_vtf.clear(param.line_search_max_t);
    toi_ee.clear(param.line_search_max_t);

    Vec<char> intersection_flag_vec = intersection_flag.as_vec();
    Vec<float> toi_vtf_vec = toi_vtf.as_vec();
    Vec<float> toi_ee_vec = toi_ee.as_vec();

    Vec<VertexParam> vertex_params = data.param_arrays.vertex;
    Vec<EdgeParam> edge_params = data.param_arrays.edge;
    Vec<FaceParam> face_params = data.param_arrays.face;
    Vec<VertexParam> collision_vertex_params =
        data.constraint.mesh.param_arrays.vertex;
    Vec<EdgeParam> collision_edge_params =
        data.constraint.mesh.param_arrays.edge;
    Vec<FaceParam> collision_face_params =
        data.constraint.mesh.param_arrays.face;

    DISPATCH_START(surface_vert_count)
    [data, x0, x1, toi_vtf_vec, param] __device__(unsigned i) mutable {
        vertex_constraint_line_search(data, x0, x1, toi_vtf_vec, param, i);
    } DISPATCH_END;

    const bool *vert_active = get_vert_collision_active();
    const bool *edge_active = get_edge_collision_active();

    if (!param.disable_contact) {
        DISPATCH_START(surface_vert_count)
        [data, mesh, x0, x1, face_bvh, face_aabb, toi_vtf_vec, vertex_params,
         face_params, vert_active, param] __device__(unsigned i) mutable {
            const VertexParam &vparam =
                vertex_params[data.prop.vertex[i].param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            float toi = param.line_search_max_t;
            PointFaceCCD ccd = {x0,
                                x1,
                                mesh.mesh.face,
                                data.prop.vertex,
                                data.prop.face,
                                vertex_params,
                                face_params,
                                i,
                                toi,
                                param};
            AABB_AABB_Tester<PointFaceCCD> op(ccd);
            AABB aabb = aabb::make(
                x0[i], float(toi) * (x1[i] - x0[i]) + x0[i], ext_eps);
            if (vert_active && !vert_active[i]) aabb.active = false;
            aabb::query(face_bvh, face_aabb, op, aabb);
            toi_vtf_vec[i] = fmin(toi_vtf_vec[i], toi);
        } DISPATCH_END;

        // Grain-grain (point-point) CCD over the vertex BVH, bounding the step
        // so two free grains never cross the contact wall. Without it a faceless
        // SAND cloud has no point-point CCD candidate and dense piles penetrate.
        const BVH &vertex_bvh = bvhset.vertex;
        const Vec<AABB> vertex_aabb = storage::vertex_aabb;
        DISPATCH_START(surface_vert_count)
        [data, x0, x1, vertex_bvh, vertex_aabb, toi_vtf_vec, vertex_params,
         vert_active, param] __device__(unsigned i) mutable {
            const VertexParam &vparam =
                vertex_params[data.prop.vertex[i].param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            float toi = param.line_search_max_t;
            PointPointCCD ccd = {x0,   x1, data.prop.vertex, vertex_params,
                                 i,    toi, param};
            AABB_AABB_Tester<PointPointCCD> op(ccd);
            AABB aabb = aabb::make(
                x0[i], float(toi) * (x1[i] - x0[i]) + x0[i], ext_eps);
            if (vert_active && !vert_active[i]) aabb.active = false;
            aabb::query(vertex_bvh, vertex_aabb, op, aabb);
            toi_vtf_vec[i] = fmin(toi_vtf_vec[i], toi);
        } DISPATCH_END;

        DISPATCH_START(surface_vert_count)
        [data, mesh, x0, x1, collision_mesh_face, collision_mesh_face_bvh,
         collision_mesh_face_aabb, collision_mesh_vertex, toi_vtf_vec,
         vertex_params, collision_face_params, vert_active,
         param] __device__(unsigned i) mutable {
            if (data.prop.vertex[i].fix_index == 0) {
                const VertexParam &vparam =
                    vertex_params[data.prop.vertex[i].param_index];
                float ext_eps = 0.5f * vparam.ghat + vparam.offset;
                float toi = param.line_search_max_t;
                CollisionMeshPointFaceCCD_M2C ccd = {
                    x0,
                    x1,
                    mesh.mesh.face,
                    collision_mesh_face,
                    collision_mesh_vertex,
                    data.prop.vertex,
                    data.constraint.mesh.prop.face,
                    vertex_params,
                    collision_face_params,
                    i,
                    toi,
                    param};
                AABB_AABB_Tester<CollisionMeshPointFaceCCD_M2C> op(ccd);
                AABB aabb = aabb::make(
                    x0[i], float(toi) * (x1[i] - x0[i]) + x0[i], ext_eps);
                if (vert_active && !vert_active[i]) aabb.active = false;
                aabb::query(collision_mesh_face_bvh, collision_mesh_face_aabb,
                            op, aabb);
                toi_vtf_vec[i] = fmin(toi_vtf_vec[i], toi);
            }
        } DISPATCH_END;

        unsigned collision_mesh_vert_count = data.constraint.mesh.vertex.size;
        DISPATCH_START(collision_mesh_vert_count)
        [data, mesh, x0, x1, collision_mesh_face, collision_mesh_vertex,
         toi_vtf_vec, face_bvh, face_aabb, face_params, collision_vertex_params,
         param] __device__(unsigned i) mutable {
            float toi = param.line_search_max_t;
            CollisionMeshPointFaceCCD_C2M ccd = {
                x0,
                x1,
                mesh.mesh.face,
                collision_mesh_vertex,
                data.prop.face,
                data.constraint.mesh.prop.vertex,
                face_params,
                collision_vertex_params,
                i,
                toi,
                param};
            AABB_AABB_Tester<CollisionMeshPointFaceCCD_C2M> op(ccd);
            Vec3f q = collision_mesh_vertex[i];
            const VertexParam &vparam =
                collision_vertex_params[data.constraint.mesh.prop.vertex[i]
                                            .param_index];
            float ext_eps = 0.5f * vparam.ghat + vparam.offset;
            aabb::query(face_bvh, face_aabb, op, aabb::make(q, ext_eps));
            toi_vtf_vec[i] = fmin(toi_vtf_vec[i], toi);
        } DISPATCH_END;

        // Seed the edge-edge CCD with the vertex/face time-of-impact minimum.
        // An edge collision later than T_vf cannot beat an already-found earlier
        // vertex/face hit, so bounding the edge sweep to [0, T_vf] prunes far
        // BVH nodes (this kernel is global-load-latency bound) while leaving the
        // final fmin over all toi bit-identical. The ext_eps spatial margin is
        // untouched, so coverage of [0, T_vf] stays conservative (no missed
        // swept pair, penetration-free).
        float T_vf_seed = kernels::min_array(
            toi_vtf_vec.data, toi_vtf_vec.size, param.line_search_max_t);

        DISPATCH_START(edge_count)
        [data, mesh, x0, x1, edge_bvh, edge_aabb, toi_ee_vec, edge_params,
         edge_active, T_vf_seed, param] __device__(unsigned i) mutable {
            Vec2u edge = mesh.mesh.edge[i];
            float toi = T_vf_seed;
            const EdgeParam &eparam =
                edge_params[data.prop.edge[i].param_index];
            float ext_eps = 0.5f * eparam.ghat + eparam.offset;
            AABB aabb0 = aabb::make(x0[edge[0]], x0[edge[1]], ext_eps);
            AABB aabb1 = aabb::make(
                float(toi) * (x1[edge[0]] - x0[edge[0]]) + x0[edge[0]],
                float(toi) * (x1[edge[1]] - x0[edge[1]]) + x0[edge[1]],
                ext_eps);
            AABB aabb = aabb::join(aabb0, aabb1);
            if (edge_active && !edge_active[i]) aabb.active = false;
            EdgeEdgeCCD ccd = {
                x0, x1,  mesh.mesh.edge, data.prop.edge, edge_params,
                data.prop.vertex, i,  toi, param};
            AABB_AABB_Tester<EdgeEdgeCCD> op(ccd);
            aabb::query(edge_bvh, edge_aabb, op, aabb);
            toi_ee_vec[i] = fmin(toi, toi_ee_vec[i]);
        } DISPATCH_END;

        DISPATCH_START(edge_count)
        [data, mesh, x0, x1, collision_mesh_edge_bvh, collision_mesh_edge,
         collision_mesh_vertex, collision_mesh_edge_aabb, toi_ee_vec,
         edge_params, collision_edge_params, edge_active, T_vf_seed,
         param] __device__(unsigned i) mutable {
            Vec2u edge = mesh.mesh.edge[i];
            float toi = T_vf_seed;
            const EdgeParam &eparam =
                edge_params[data.prop.edge[i].param_index];
            float ext_eps = 0.5f * eparam.ghat + eparam.offset;
            AABB aabb0 = aabb::make(x0[edge[0]], x0[edge[1]], ext_eps);
            AABB aabb1 = aabb::make(
                float(toi) * (x1[edge[0]] - x0[edge[0]]) + x0[edge[0]],
                float(toi) * (x1[edge[1]] - x0[edge[1]]) + x0[edge[1]],
                ext_eps);
            AABB aabb = aabb::join(aabb0, aabb1);
            if (edge_active && !edge_active[i]) aabb.active = false;
            CollisionMeshEdgeEdgeCCD ccd = {x0,
                                            x1,
                                            mesh.mesh.edge,
                                            collision_mesh_edge,
                                            collision_mesh_vertex,
                                            data.prop.edge,
                                            data.constraint.mesh.prop.edge,
                                            edge_params,
                                            collision_edge_params,
                                            i,
                                            toi,
                                            param};

            AABB_AABB_Tester<CollisionMeshEdgeEdgeCCD> op(ccd);
            aabb::query(collision_mesh_edge_bvh, collision_mesh_edge_aabb, op,
                        aabb);
            toi_ee_vec[i] = fmin(toi, toi_ee_vec[i]);
        } DISPATCH_END;
    }

    float toi = fminf(kernels::min_array(toi_vtf_vec.data, toi_vtf_vec.size,
                                         param.line_search_max_t),
                      kernels::min_array(toi_ee_vec.data, toi_ee_vec.size,
                                         param.line_search_max_t));

    return toi / param.line_search_max_t;
}

template <class T, class Y>
__device__ bool point_triangle_inside(const Vec3<T> &p, const Vec3<T> &t0,
                                      const Vec3<T> &t1, const Vec3<T> &t2) {
    Vec3<Y> r0 = (t1 - t0).template cast<Y>();
    Vec3<Y> r1 = (t2 - t0).template cast<Y>();
    Mat3x2<Y> a;
    a << r0, r1;
    Eigen::Transpose<Mat3x2<Y>> a_t = a.transpose();
    Y det;
    Vec2<Y> c;
    distance::solve<Y>(a_t * a, a_t * (p - t0).template cast<Y>(), c, det);
    if (det) {
        Y w0 = c[0] / det;
        Y w1 = c[1] / det;
        Y w2 = Y(1.0) - w0 - w1;
        Y wmin = fmin(fmin(w0, w1), w2);
        Y wmax = fmax(fmax(w0, w1), w2);
        return wmin >= 0.0f && wmax <= 1.0f;
    } else {
        return false;
    }
}

__device__ bool edge_triangle_intersect(const Vec3f &a0, const Vec3f &a1,
                                        const Vec3f &b0, const Vec3f &b1,
                                        const Vec3f &b2) {
    Vec3f d1 = (b1 - b0);
    Vec3f d2 = (b2 - b0);
    Vec3f e0 = (a0 - b0);
    Vec3f e1 = (a1 - b0);
    Vec3f n = d1.cross(d2);
    float s1 = e0.dot(n);
    float s2 = e1.dot(n);
    if (s1 * s2 < 0.0f) {
        float t = s1 / (s1 - s2);
        Vec3f r = (1.0f - t) * e0 + t * e1;
        return point_triangle_inside<float, float>(r, Vec3f::Zero(), d1, d2);
    }
    return false;
}

class EdgeEdgeIntersectTester {
  public:
    __device__
    EdgeEdgeIntersectTester(const Vec<EdgeProp> &prop,
                            const Vec<EdgeParam> &edge_params,
                            const Vec<Vec3f> &vertex, const Vec<Vec2u> &edge,
                            const Vec<VertexProp> &vert_prop,
                            const ParamSet &param, unsigned edge_index,
                            IntersectionRecord *records, unsigned *record_counter)
        : prop(prop), edge_params(edge_params), vertex(vertex), edge(edge),
          vert_prop(vert_prop), param(param), edge_index(edge_index),
          records(records), record_counter(record_counter) {}
    __device__ bool operator()(unsigned index) {
        if (index < edge_index) {
            const Vec2u &e0 = edge[edge_index];
            const Vec2u &e1 = edge[index];
            bool either_dyn =
                prop[edge_index].fixed == false || prop[index].fixed == false;
            // Two zero-mass edges (both static solids) never intersect.
            bool either_nonzero =
                prop[edge_index].mass > 0.0f || prop[index].mass > 0.0f;
            unsigned bid_a = vert_prop[e0[0]].pdrd_body_index;
            unsigned bid_b = vert_prop[e1[0]].pdrd_body_index;
            bool same_pdrd_body = bid_a != 0 && bid_a == bid_b;
            if (either_dyn && either_nonzero && !same_pdrd_body) {
                const EdgeParam &eparam_edge =
                    edge_params[prop[edge_index].param_index];
                const EdgeParam &eparam_index =
                    edge_params[prop[index].param_index];
                float offset = eparam_edge.offset + eparam_index.offset;
                if (!edge_has_shared_vert(e0, e1)) {
                    Vec3f p0 = vertex[e0[0]];
                    Vec3f p1 = vertex[e0[1]];
                    Vec3f q0 = vertex[e1[0]];
                    Vec3f q1 = vertex[e1[1]];
                    Vec4f c = distance::edge_edge_distance_coeff_unclassified<
                                   float, float>(p0, p1, q0, q1)
                                   ;
                    Vec3f x0 = c[0] * p0 + c[1] * p1;
                    Vec3f x1 = c[2] * q0 + c[3] * q1;
                    Vec3f e = x0 - x1;
                    if (e.dot(e) < offset * offset) {
                        Vec3f ev0[2] = {p0, p1};
                        Vec3f ev1[2] = {q0, q1};
                        record_intersection(records, record_counter,
                                            1, edge_index, index, ev0, 2, ev1, 2);
                        return true;
                    }
                } else {
                    const Vec2u ij[] = {Vec2u(0, 0), Vec2u(0, 1), Vec2u(1, 0),
                                        Vec2u(1, 1)};
                    for (unsigned k = 0; k < 4; ++k) {
                        unsigned i = ij[k][0];
                        unsigned j = ij[k][1];
                        if (e0[i] == e1[j]) {
                            unsigned idx0 = e0[i];
                            unsigned idx1 = e0[1 - i];
                            unsigned idx2 = e1[1 - j];
                            const Vec3f &q0 = vertex[idx0];
                            const Vec3f &q1 = vertex[idx1];
                            const Vec3f &q2 = vertex[idx2];
                            Vec2f c_0 =
                                distance::
                                    point_edge_distance_coeff_unclassified<
                                        float, float>(q1, q0, q2)
                                        ;
                            Vec2f c_1 =
                                distance::
                                    point_edge_distance_coeff_unclassified<
                                        float, float>(q2, q0, q1)
                                        ;
                            Vec3f e_0 = ((c_0[0] * q0 + c_0[1] * q2) - q1)
                                            ;
                            Vec3f e_1 = ((c_1[0] * q0 + c_1[1] * q1) - q2)
                                            ;
                            float sqr_d0 = e_0.dot(e_0);
                            float sqr_d1 = e_1.dot(e_1);
                            if (std::min(sqr_d0, sqr_d1) < offset * offset) {
                                Vec3f ev0[2] = {vertex[e0[0]], vertex[e0[1]]};
                                Vec3f ev1[2] = {vertex[e1[0]], vertex[e1[1]]};
                                record_intersection(records, record_counter,
                                                    1, edge_index, index, ev0, 2, ev1, 2);
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }
    const Vec<EdgeProp> &prop;
    const Vec<EdgeParam> &edge_params;
    const Vec<Vec3f> &vertex;
    const Vec<Vec2u> &edge;
    const Vec<VertexProp> &vert_prop;
    const ParamSet &param;
    unsigned edge_index;
    IntersectionRecord *records;
    unsigned *record_counter;
};

class FaceEdgeIntersectTester {
  public:
    __device__
    FaceEdgeIntersectTester(const Vec<FaceProp> &face_prop,
                            const Vec<EdgeProp> &edge_prop,
                            const Vec<VertexProp> &vert_prop,
                            const Vec<Vec3f> &vertex, const Vec<Vec3u> &face,
                            const Vec<Vec2u> &edge, unsigned edge_index,
                            IntersectionRecord *records, unsigned *record_counter)
        : face_prop(face_prop), edge_prop(edge_prop), vert_prop(vert_prop),
          vertex(vertex), face(face), edge(edge), edge_index(edge_index),
          records(records), record_counter(record_counter) {}
    __device__ bool operator()(unsigned index) {
        bool either_dyn = face_prop[index].fixed == false ||
                          edge_prop[edge_index].fixed == false;
        // Two zero-mass elements (both static solids) never intersect.
        bool either_nonzero = face_prop[index].mass > 0.0f ||
                              edge_prop[edge_index].mass > 0.0f;
        // Skip intra-PDRD-body edge/face pairs: a rigid body never deforms, so
        // an edge piercing a face of the SAME body is a fixed, physically
        // meaningless self-intersection that must be tolerated even when the
        // body starts self-tangled. Mirrors the same_pdrd_body filter in the
        // edge-edge / point-point intersect testers and the contact embeds.
        unsigned bid_e = vert_prop[edge[edge_index][0]].pdrd_body_index;
        unsigned bid_f = vert_prop[face[index][0]].pdrd_body_index;
        bool same_pdrd_body = bid_e != 0 && bid_e == bid_f;
        if (either_dyn && either_nonzero && !same_pdrd_body) {
            Vec3u f = face[index];
            unsigned e0 = edge[edge_index][0];
            unsigned e1 = edge[edge_index][1];
            if (e0 != f[0] && e0 != f[1] && e0 != f[2] && //
                e1 != f[0] && e1 != f[1] && e1 != f[2]) {
                const Vec3f &x0 = vertex[f[0]];
                const Vec3f &x1 = vertex[f[1]];
                const Vec3f &x2 = vertex[f[2]];
                const Vec3f &y0 = vertex[e0];
                const Vec3f &y1 = vertex[e1];
                if (edge_triangle_intersect(y0, y1, x0, x1, x2)) {
                    Vec3f fv[3] = {x0, x1, x2};
                    Vec3f ev[2] = {y0, y1};
                    record_intersection(records, record_counter,
                                        0, index, edge_index, fv, 3, ev, 2);
                    return true;
                }
            }
        }
        return false;
    }
    const Vec<FaceProp> &face_prop;
    const Vec<EdgeProp> &edge_prop;
    const Vec<VertexProp> &vert_prop;
    const Vec<Vec3f> &vertex;
    const Vec<Vec3u> &face;
    const Vec<Vec2u> &edge;
    unsigned edge_index;
    IntersectionRecord *records;
    unsigned *record_counter;
};

class CollisionMeshFaceEdgeIntersectTester {
  public:
    __device__ CollisionMeshFaceEdgeIntersectTester(const Vec<Vec3f> &vertex,
                                                    const Vec<Vec3u> &face,
                                                    const Vec3f &y0,
                                                    const Vec3f &y1,
                                                    unsigned edge_index,
                                                    IntersectionRecord *records,
                                                    unsigned *record_counter)
        : vertex(vertex), face(face), y0(y0), y1(y1),
          edge_index(edge_index), records(records),
          record_counter(record_counter) {}
    __device__ bool operator()(unsigned index) {
        Vec3u f = face[index];
        const Vec3f &x0 = vertex[f[0]];
        const Vec3f &x1 = vertex[f[1]];
        const Vec3f &x2 = vertex[f[2]];
        if (edge_triangle_intersect(y0, y1, x0, x1, x2)) {
            Vec3f fv[3] = {x0, x1, x2};
            Vec3f ev[2] = {y0, y1};
            record_intersection(records, record_counter,
                                2, index, edge_index, fv, 3, ev, 2);
            return true;
        }
        return false;
    }
    const Vec<Vec3f> &vertex;
    const Vec<Vec3u> &face;
    Vec3f y0, y1;
    unsigned edge_index;
    IntersectionRecord *records;
    unsigned *record_counter;
};

// Detects two grains (free vertices) that begin the simulation already inside
// each other's contact radius (centers closer than offset_i + offset_j). The
// smooth contact barrier is undefined inside the wall, so such a start state
// would otherwise abort mid-advance with an opaque kernel assertion; recording
// it here lets initialization fail with a clear "point_point" violation, the
// same way edge/triangle self-intersection is reported. Pair filtering matches
// PointPointContactForceHessEmbed (upper-triangular, at least one free vertex,
// not the same PDRD body).
class PointPointIntersectTester {
  public:
    __device__
    PointPointIntersectTester(const Vec<VertexProp> &vert_prop,
                              const Vec<VertexParam> &vertex_params,
                              const Vec<Vec3f> &vertex, unsigned vertex_index,
                              IntersectionRecord *records,
                              unsigned *record_counter)
        : vert_prop(vert_prop), vertex_params(vertex_params), vertex(vertex),
          vertex_index(vertex_index), records(records),
          record_counter(record_counter) {}
    __device__ bool operator()(unsigned index) {
        if (index < vertex_index) {
            bool either_dyn = vert_prop[vertex_index].fix_index == 0 ||
                              vert_prop[index].fix_index == 0;
            unsigned bid_a = vert_prop[vertex_index].pdrd_body_index;
            unsigned bid_b = vert_prop[index].pdrd_body_index;
            bool same_pdrd_body = bid_a != 0 && bid_a == bid_b;
            if (either_dyn && !same_pdrd_body) {
                const VertexParam &vp_a =
                    vertex_params[vert_prop[vertex_index].param_index];
                const VertexParam &vp_b =
                    vertex_params[vert_prop[index].param_index];
                float offset = vp_a.offset + vp_b.offset;
                Vec3f p = vertex[vertex_index];
                Vec3f q = vertex[index];
                Vec3f e = (p - q).cast<float>();
                if (e.dot(e) < offset * offset) {
                    Vec3f ev0[1] = {p};
                    Vec3f ev1[1] = {q};
                    record_intersection(records, record_counter, 3, vertex_index,
                                        index, ev0, 1, ev1, 1);
                    return true;
                }
            }
        }
        return false;
    }
    const Vec<VertexProp> &vert_prop;
    const Vec<VertexParam> &vertex_params;
    const Vec<Vec3f> &vertex;
    unsigned vertex_index;
    IntersectionRecord *records;
    unsigned *record_counter;
};

bool check_intersection(const DataSet &data, const Vec<Vec3f> &vertex,
                        const ParamSet &param) {

    unsigned edge_count = data.mesh.mesh.edge.size;
    const BVH &face_bvh = bvh_storage::get_bvh().face;
    const BVH &edge_bvh = bvh_storage::get_bvh().edge;
    const BVH &collision_mesh_face_bvh = bvh_storage::get_collision_mesh_bvh().face;
    const MeshInfo &mesh = data.mesh;
    const Vec<Vec3u> &collision_mesh_face = data.constraint.mesh.face;
    const Vec<AABB> face_aabb = storage::face_aabb;
    const Vec<AABB> edge_aabb = storage::edge_aabb;
    const Vec<AABB> collision_mesh_face_aabb =
        storage::collision_mesh_face_aabb;

    buffer::MemoryPool &pool = buffer::get();
    auto intersection_flag = pool.get<char>(edge_count);
    intersection_flag.clear(0);
    const Vec<Vec3f> &collision_mesh_vertex = data.constraint.mesh.vertex;

    auto records_pool = pool.get<IntersectionRecord>(MAX_INTERSECTION_RECORDS);
    auto counter_pool = pool.get<unsigned>(1);
    counter_pool.clear(0);
    Vec<IntersectionRecord> records_vec = records_pool.as_vec();
    Vec<unsigned> counter_vec = counter_pool.as_vec();

    Vec<char> intersection_flag_vec = intersection_flag.as_vec();
    Vec<EdgeParam> edge_params = data.param_arrays.edge;

    const bool *edge_active = get_edge_collision_active();

    DISPATCH_START(edge_count)
    [data, mesh, vertex, collision_mesh_vertex, face_bvh, face_aabb, edge_bvh,
     edge_aabb, collision_mesh_face_bvh, collision_mesh_face_aabb,
     collision_mesh_face, intersection_flag_vec, edge_params,
     records_vec, counter_vec, edge_active,
     param] __device__(unsigned i) mutable {
        const Vec2u &edge = mesh.mesh.edge[i];
        Vec3f y0 = vertex[edge[0]];
        Vec3f y1 = vertex[edge[1]];
        AABB aabb = aabb::make(y0, y1, 0.0f);
        if (edge_active && !edge_active[i]) aabb.active = false;
        FaceEdgeIntersectTester tester_0(data.prop.face, data.prop.edge,
                                         data.prop.vertex, vertex,
                                         mesh.mesh.face, mesh.mesh.edge, i,
                                         records_vec.data, counter_vec.data);
        AABB_AABB_Tester<FaceEdgeIntersectTester> op_0(tester_0);
        if (aabb::query(face_bvh, face_aabb, op_0, aabb)) {
            intersection_flag_vec[i] = 1;
        }
        EdgeEdgeIntersectTester tester_1(data.prop.edge, edge_params, vertex,
                                         mesh.mesh.edge, data.prop.vertex,
                                         param, i,
                                         records_vec.data, counter_vec.data);
        AABB_AABB_Tester<EdgeEdgeIntersectTester> op_1(tester_1);
        if (aabb::query(edge_bvh, edge_aabb, op_1, aabb)) {
            intersection_flag_vec[i] = 1;
        }
        // Zero-mass edges (static solids, incl. moving-static shells)
        // never intersect the collision mesh (itself a static solid).
        if (data.prop.edge[i].mass > 0.0f) {
            CollisionMeshFaceEdgeIntersectTester tester_2(
                collision_mesh_vertex, collision_mesh_face, y0, y1, i,
                records_vec.data, counter_vec.data);
            AABB_AABB_Tester<CollisionMeshFaceEdgeIntersectTester> op_2(tester_2);
            if (aabb::query(collision_mesh_face_bvh, collision_mesh_face_aabb, op_2,
                            aabb)) {
                intersection_flag_vec[i] = 1;
            }
        }
    } DISPATCH_END;

    // Grain-grain (point-point) initial overlap. The edge loop above never runs
    // for a faceless particle cloud (no edges), so without this pass an
    // overlapping SAND cloud passes the init check and then aborts mid-advance.
    unsigned surface_vert_count = data.surface_vert_count;
    auto vert_flag = pool.get<char>(max_of_two(surface_vert_count, 1u));
    vert_flag.clear(0);
    Vec<char> vert_flag_vec = vert_flag.as_vec();
    if (surface_vert_count > 0) {
        const BVH &vertex_bvh = bvh_storage::get_bvh().vertex;
        const Vec<AABB> vertex_aabb = storage::vertex_aabb;
        Vec<VertexParam> vertex_params = data.param_arrays.vertex;
        const bool *vert_active = get_vert_collision_active();
        DISPATCH_START(surface_vert_count)
        [data, vertex, vertex_bvh, vertex_aabb, vertex_params, vert_flag_vec,
         records_vec, counter_vec, vert_active] __device__(unsigned i) mutable {
            const VertexParam &vparam =
                vertex_params[data.prop.vertex[i].param_index];
            AABB aabb = aabb::make(vertex[i], vparam.offset);
            if (vert_active && !vert_active[i]) aabb.active = false;
            PointPointIntersectTester tester(data.prop.vertex, vertex_params,
                                             vertex, i, records_vec.data,
                                             counter_vec.data);
            AABB_AABB_Tester<PointPointIntersectTester> op(tester);
            if (aabb::query(vertex_bvh, vertex_aabb, op, aabb)) {
                vert_flag_vec[i] = 1;
            }
        } DISPATCH_END;
    }

    // Copy intersection records from GPU to host
    unsigned gpu_count = 0;
    CUDA_HANDLE_ERROR(cudaMemcpy(&gpu_count, counter_vec.data,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
    storage::host_intersection_count =
        std::min(gpu_count, (unsigned)MAX_INTERSECTION_RECORDS);
    if (storage::host_intersection_count > 0) {
        CUDA_HANDLE_ERROR(cudaMemcpy(
            storage::host_intersection_records, records_vec.data,
            storage::host_intersection_count * sizeof(IntersectionRecord),
            cudaMemcpyDeviceToHost));
    }

    bool edge_clear = kernels::max_array(intersection_flag_vec.data, edge_count,
                                         char(0)) == 0;
    bool vert_clear =
        surface_vert_count == 0 ||
        kernels::max_array(vert_flag_vec.data, surface_vert_count, char(0)) == 0;
    return edge_clear && vert_clear;
}

unsigned get_intersection_count() {
    return storage::host_intersection_count;
}

const IntersectionRecord *get_intersection_records() {
    return storage::host_intersection_records;
}

} // namespace contact
