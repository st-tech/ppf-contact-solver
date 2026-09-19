// File: collision_narrow.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: the three static-collision-mesh visitors.
//
// Three passes, in the order the assembly dispatches them: a dynamic vertex
// against a collider triangle (M2C), a collider vertex against a dynamic
// triangle (C2M), and a dynamic edge against a collider edge. The order is part
// of the answer, because the force and both Hessians are fp32 running sums.
//
// THE COLLIDER SIDE CONTRIBUTES NO ROW. It is a rest-pose contact-only pool
// outside the solved namespace, so only the DYNAMIC vertices of a pair carry
// weights, and the mass, the elastic snapshot and the friction slip are
// gathered over those alone. That is what makes the collider a boundary
// condition rather than a body: it pushes and is never pushed.
//
// THE DIAGNOSTIC IS PART OF THE PHYSICS HERE, as in `contact_narrow`: the
// separation assert is the penetration-free guarantee, so each entry declares a
// `[[seam::diag]]` lane.

// `distance.hpp` FIRST, and the order is load-bearing rather than tidy. A
// neutral kernel body includes no header of its own, so the type vocabulary
// (`Vec3f`, `Mat3x3f`, the property and parameter records) reaches it from
// whichever SHARED HEADER an includer pulls in first; `distance.hpp` is the one this file needs anyway and
// it includes `data.hpp` unconditionally. Put a `.kernel.cpp` ahead of it and
// the file compiles only for a translation unit that had already established
// the vocabulary, which the host build does and a bare `nvcc` on the rendering
// does not.
// THE TRAVERSAL, because the fused passes below ARE its visitors. A kernel body
// brings no header of its own, so a rendering compiled ALONE, which is what
// `--features cuda-abi` hands nvcc, sees only what this file includes.
#include "aabb_traversal.kernel.cpp"
#include "distance.hpp"

#include "../barrier/contact_stiffness.kernel.cpp"
#include "../energy/model/friction.kernel.cpp"
#include "../energy/model/push.kernel.cpp"
#include "../primitives/vec_ops.kernel.cpp"
#include "analytic_contact.kernel.cpp"
#include "contact_assembly.kernel.cpp"
#include "contact_statistics.kernel.cpp"

// One collision-mesh contact's extended force and Hessian over the N DYNAMIC
// vertices its weights name.
//
// `separation` is measured by the caller, because the two sides live in
// different position arrays and how they combine is what distinguishes the
// three passes. Everything after it is the general contact form: the
// elasticity-inclusive dynamic stiffness, the one-sided push barrier, the
// friction cone, and the congruence that widens a 3x3 onto 3N x 3N.
template <unsigned N>
[[seam::device_fn]] inline bool embed_collision(
    const Vec3f *x0, const Vec3f *x,
    const VertexProp *vertex_prop,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, and the substep it is
    // divided by. A COPY of the force vector taken before this pass: all three
    // collision-mesh passes deposit into that vector through atomics, so
    // reading it directly would hand one pair a residual another had moved.
    const float *residual, float dt, const unsigned *index,
    const SVecf<N> &weight,
    const Vec3f &separation, float ghat, float offset,
    float friction, unsigned left, unsigned right,
    // THREAD SPACE, NOT STAGING. The pair form copies these into its own slot
    // and a fused traversal deposits them straight into the force and the
    // matrix, so the shared contact math has ONE home and neither caller
    // re-derives it.
    SMatf<3, N> &out_force,
    SMatf<3 * N, 3 * N> &out_hessian,
    // The collapsed-separation report. This body fills the MEASUREMENT
    // (flag, world-space squared separation, offset); the caller names the
    // pair and the kind, because only it knows which of the three
    // collision-mesh paths it is and which vertex ids the reference reports.
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    SVecf<N> mass;
    SMatf<3 * N, 3 * N> local_hessian = SMatf<3 * N, 3 * N>::Zero();
    Vec3f slip = Vec3f::Zero();
    for (unsigned i = 0; i < N; ++i) {
        mass[i] = vertex_prop[index[i]].mass;
        for (unsigned j = 0; j < N; ++j) {
            local_hessian.template block<3, 3>(3 * i, 3 * j) =
                fixed_csr_read(fixed_index, fixed_offset, fixed_value,
                                   row_count, index[i], index[j]);
        }
        // THE SLIP IS A DIFFERENCE, NEVER A COORDINATE. Each vertex's step is
        // formed as `x - x0` before anything scales it, so no absolute
        // coordinate enters the sum; the weighted sum of those steps is the
        // change in the separation vector, because the static side does not
        // move.
        // THE COPIES ARE A METAL REQUIREMENT rather than a style. Reached
        // through a `[[seam::device]]` base pointer, `x[index[i]]` is a
        // DEVICE-space lvalue, and the vector type's `operator-` is
        // declared for thread space alone, so MSL finds no viable overload
        // for the difference. The copy is what the generator emits for a
        // `[[seam::gather]]` parameter anyway, so it costs nothing, and the
        // subtraction it feeds is the same one on the same values.
        const Vec3f x_i = x[index[i]];
        const Vec3f x0_i = x0[index[i]];
        slip += weight[i] * (x_i - x0_i).cast<float>();
    }
    // A PAIR WHOSE SEPARATION HAS ALREADY CLOSED TO THE CONTACT OFFSET. The
    // barrier's gap is non-positive there and `normal` below is a 0/0
    // normalize, so the pair carries no usable force: it contributes nothing
    // and is REPORTED rather than asserted, for the reason `embed_contact`
    // gives (an assert differs by build; a record the host reads does not).
    // Mirrors the three `accd::report_contact_overlap` sites in `contact.cu`
    // (kinds 7, 8, 9), whose lengths are world units.
    // THE SAME TEST AS `embed_contact`, the reference's `!(d2 > offset^2)`,
    // which fires on a pair exactly touching; see the note there for why the
    // boundary is load-bearing and why a stricter one buys nothing.
    const float d2 = separation.squaredNorm();
    if (!(d2 > offset * offset)) {
        out_overlap.flagged = 1u;
        out_overlap.d2 = d2;
        out_overlap.offset = offset;
        (void)left;
        (void)right;
        return false;
    }
    const Vec3f normal = separation.normalized();
    const float stiffness = contact_stiffness<N>(local_hessian, weight, mass,
                                                     separation, offset, diag);
    // THE ONE-SIDED PUSH BARRIER, which is what the reference implementation
    // assembles at all three collision-mesh sites and is NOT the barrier family
    // the self-contact path selects through `ParamSet::barrier`. The two are not
    // interchangeable: the cubic family's energy is twice the push barrier's, so
    // reading the family here would make a collider push twice as hard. The
    // signed distance is the separation's length past the combined offset and
    // gap.
    const Vec3f projected = (offset + ghat) * normal;
    const float signed_distance = (separation - projected).dot(normal);
    Vec3f force = stiffness * push_gradient(signed_distance, normal, ghat);
    Mat3x3f hessian = stiffness * push_hessian(signed_distance, normal, ghat);
    Vec3f friction_gradient;
    Mat3x3f friction_hessian;
    Vec3f tangent;
    Mat3x3f projection;
    float lambda;
    float friction_stiffness;
    float normal_force;
    // A VERTEX WITHOUT A DEGREE OF FREEDOM CARRIES WEIGHT ZERO IN THE
    // PREDICTION, an exact fix pin or a massless vertex of a static solid: its
    // row is eliminated, so it does not move however hard it is pushed. THE
    // COLLIDER SIDE IS NOT IN `index` AT ALL, which is what makes one form
    // serve all three passes: the dynamic side is one vertex against a collider
    // face, the three vertices of a mesh face under a collider vertex, or the
    // two of a mesh edge, and `weight` is already the barycentric weights of
    // whichever it is.
    SVecf<N> free_weight;
    for (unsigned i = 0; i < N; ++i) {
        const bool owns_dof =
            vertex_prop[index[i]].fix_index == 0u && mass[i] > 0.0f;
        free_weight[i] = owns_dof ? weight[i] : 0.0f;
    }
    Vec3f drive = Vec3f::Zero();
    float drive_stiffness = 0.0f;
    if (!friction_slip_prediction<N>(index, free_weight, mass, local_hessian,
                                         residual, normal, dt, drive,
                                         drive_stiffness)) {
        drive_stiffness = 0.0f;
    }
    friction_evaluate(force, slip, normal, friction, friction_eps, drive,
                          drive_stiffness, friction_gradient, friction_hessian,
                          lambda, friction_stiffness, tangent, projection,
                          normal_force);
    force += friction_gradient;
    hessian += friction_hessian;
    out_force = SMatf<3, N>::Zero();
    out_hessian = SMatf<3 * N, 3 * N>::Zero();
    extend_contact_force_hessian<N>(weight, force, hessian, out_force,
                                        out_hessian);
    return true;
}

// One accepted collision contact's force and Hessian, deposited straight into
// the force vector and the fixed matrix.
//
// THIS IS WHAT THE TWO STAGED EMBED KERNELS DO, per pair, with the staging read
// replaced by the thread-space result the core just produced. The block walk,
// the `row > column` skip and the dynamic-slot fallback are the same, so a
// fused pass and the pair path deposit identically.
//
// `dyn_capacity` IS ZERO ON THE COLLIDER PATH and that is a rule rather than a
// size: a collision-mesh stencil is a vertex's own diagonal, a face's three
// vertices or an edge's two, all of which `builder.rs` registers, so a block the
// fixed pattern cannot hold means the pattern does not describe this mesh. The
// claim still counts, and the HOST turns a non-zero claim into a fatal.
template <unsigned N>
[[seam::device_fn]] inline void deposit_collision(
    const unsigned *slot,
    const SMatf<3, N> &local_force,
    const SMatf<3 * N, 3 * N> &local_hessian,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *out_fixed_value,
    unsigned row_count,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity) {
    for (unsigned i = 0u; i < N; ++i) {
        for (unsigned d = 0u; d < 3u; ++d) {
            const float term = local_force.m[3u * i + d];
            if (term != 0.0f) {
                compute::atomic_add(out_vertex_force + 3u * slot[i] + d, term);
            }
        }
    }
    for (unsigned ii = 0u; ii < N; ++ii) {
        for (unsigned jj = 0u; jj < N; ++jj) {
            const unsigned row = slot[ii];
            const unsigned column = slot[jj];
            if (row > column) {
                continue;
            }
            Mat3x3f block;
            for (unsigned c = 0u; c < 3u; ++c) {
                for (unsigned r = 0u; r < 3u; ++r) {
                    block.m[3u * c + r] =
                        local_hessian.m[3u * N * (3u * jj + c) + 3u * ii + r];
                }
            }
            if (fixed_csr_atomic_push(fixed_index, fixed_offset, out_fixed_value,
                                      row_count, row, column, block)) {
                continue;
            }
            const unsigned claimed = compute::atomic_add(dyn_claim, 1u);
            if (claimed >= dyn_capacity) {
                continue;
            }
            dyn_row[claimed] = row;
            dyn_column[claimed] = column;
            for (unsigned e = 0u; e < 9u; ++e) {
                dyn_block[9u * claimed + e] = block.m[e];
            }
        }
    }
}

// A dynamic vertex against a collider triangle, evaluated at ONE pair of
// indices.
//
// THE PAIR FORM BELOW AND THE FUSED TRAVERSAL BOTH CALL THIS, so the contact
// math and the per-object statistics record have one home and the two callers
// differ only in where the result is deposited: the pair form copies it into
// its staging slot, the traversal deposits it into the force and the matrix.
// The result comes back in THREAD space for that reason.
[[seam::device_fn]] inline bool collision_point_face_m2c_at(
    const Vec3f *x0, const Vec3f *x,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const Vec3f *static_x,
    const Vec3u *static_face,
    const FaceProp *static_face_prop,
    const FaceParam *static_face_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt, unsigned vertex_index,
    unsigned face_index, unsigned *out_slot,
    SMatf<3, 1> &out_local_force,
    SMatf<3, 3> &out_local_hessian,
    // THE PER-OBJECT STATISTICS CHANNEL. The dynamic and static sides resolve
    // through DIFFERENT maps because they are different index spaces, which is
    // why both arrays are here rather than one indexed twice.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    // A pinned or zero-mass vertex cannot yield, and the collider cannot
    // either, so the pair has no way to resolve.
    if (vertex_prop[vertex_index].fix_index != 0u ||
        !(vertex_prop[vertex_index].mass > 0.0f)) {
        return false;
    }
    const Vec3u fc = static_face[face_index];
    const Vec3f p = x[vertex_index];
    const Vec3f t0 = static_x[fc[0]];
    const Vec3f t1 = static_x[fc[1]];
    const Vec3f t2 = static_x[fc[2]];
    const Vec3f c =
        proximity::point_triangle_distance_coeff_unclassified<float, float>(
            p, t0, t1, t2);
    const Vec3f separation = c[0] * (p - t0).cast<float>() +
                             c[1] * (p - t1).cast<float>() +
                             c[2] * (p - t2).cast<float>();
    const VertexParam vparam =
        vertex_param[vertex_prop[vertex_index].param_index];
    const FaceParam fparam =
        static_face_param[static_face_prop[face_index].param_index];
    const float offset = vparam.offset + fparam.offset;
    const float ghat = 0.5f * (vparam.ghat + fparam.ghat);
    const float friction = combine_friction_values(
        vparam.friction, fparam.friction, friction_mode);
    const float reach = ghat + offset;
    if (!(separation.squaredNorm() < reach * reach)) {
        return false;
    }
    unsigned index[1];
    index[0] = vertex_index;
    out_slot[0] = index[0];
    SVecf<1> weight;
    weight[0] = 1.0f;
    const bool accepted = embed_collision<1>(
        x0, x, vertex_prop, fixed_index, fixed_offset, fixed_value, row_count,
        friction_eps, residual, dt, index, weight, separation, ghat, offset, friction,
        vertex_index, face_index, out_local_force, out_local_hessian, out_overlap, diag);
    if (out_overlap.flagged != 0u) {
        // Kind 7: the reference names this pair as (vertex_index, fc[0]); the
        // second index is in the static collision-mesh vertex space.
        out_overlap.kind = 7u;
        out_overlap.elem0 = vertex_index;
        out_overlap.elem1 = fc[0];
    }
    // Counted at the ACCEPTED exit, where `contact.cu:960` records it.
    if (accepted) {
        statistics_record_dynamic_static_contact(
            statistics_contact_count, statistics_contact_count_size,
            statistics_object_index, statistics_object_index_size,
            statistics_static_object_index,
            statistics_static_object_index_size, vertex_index, fc[0], diag);
    }
    return accepted;
}

// The pair form. `pair` is (vertex, collider face).
//
// EACH PASS WRITES ONLY THE `out_index` SLOTS ITS OWN ARITY NAMES, which is one
// here, three for C2M and two for edge-edge. The staging arrays are the
// fixed-width shape the self-contact passes use so one scatter serves both, and
// that scatter reads `arity` to know how far into the four slots to look.
[[seam::entry(k)]]
[[seam::device_fn]] inline void collision_point_face_m2c(
    const Vec3f *x0, const Vec3f *x,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const Vec3f *static_x,
    const Vec3u *static_face,
    const FaceProp *static_face_prop,
    const FaceParam *static_face_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    const unsigned *pair, unsigned k,
    unsigned *active, unsigned *arity,
    unsigned *out_index, float *out_force,
    float *out_hessian,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    CcdOverlapRecord *out_overlap, DiagHandle diag) {
    active[k] = 0u;
    arity[k] = 1u;
    unsigned slot[1];
    SMatf<3, 1> local_force;
    SMatf<3, 3> local_hessian;
    CcdOverlapRecord overlap;
    overlap.flagged = 0u;
    overlap.kind = 0u;
    overlap.elem0 = 0u;
    overlap.elem1 = 0u;
    overlap.d2 = 0.0f;
    overlap.offset = 0.0f;
    const bool accepted = collision_point_face_m2c_at(
        x0, x, vertex_prop, vertex_param, static_x, static_face,
        static_face_prop, static_face_param, fixed_index, fixed_offset,
        fixed_value, row_count, friction_mode, friction_eps, residual, dt,
        pair[2 * k], pair[2 * k + 1], slot, local_force, local_hessian,
        statistics_contact_count, statistics_contact_count_size,
        statistics_object_index, statistics_object_index_size,
        statistics_static_object_index, statistics_static_object_index_size,
        overlap, diag);
    if (overlap.flagged != 0u && out_overlap[k].flagged == 0u) {
        out_overlap[k] = overlap;
    }
    // COPIED ONLY ON THE ACCEPTED PATH, which is what the staging slots have
    // always held: a rejected pair leaves its slot untouched and the embed
    // kernels skip it on `active`.
    if (accepted) {
        out_index[4 * k] = slot[0];
        for (unsigned i = 0; i < 3 * 1; ++i) {
            out_force[12 * k + i] = local_force.m[i];
        }
        for (unsigned i = 0; i < 9 * 1 * 1; ++i) {
            out_hessian[144 * k + i] = local_hessian.m[i];
        }
    }
    active[k] = accepted ? 1u : 0u;
}
// A collider vertex against a dynamic triangle, evaluated at ONE pair of
// indices. `collision_point_face_m2c_at` states why the result comes back in
// thread space.
[[seam::device_fn]] inline bool collision_point_face_c2m_at(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const FaceParam *face_param,
    const Vec3f *static_x,
    const VertexProp *static_vertex_prop,
    const VertexParam *static_vertex_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    unsigned vertex_index, unsigned face_index,
    unsigned *out_slot,
    SMatf<3, 3> &out_local_force,
    SMatf<9, 9> &out_local_hessian,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    if (face_prop[face_index].fixed || !(face_prop[face_index].mass > 0.0f)) {
        return false;
    }
    const Vec3u fc = face[face_index];
    const Vec3f y = static_x[vertex_index];
    const Vec3f t0 = x[fc[0]];
    const Vec3f t1 = x[fc[1]];
    const Vec3f t2 = x[fc[2]];
    const Vec3f c =
        proximity::point_triangle_distance_coeff_unclassified<float, float>(
            y, t0, t1, t2);
    const Vec3f separation = c[0] * (t0 - y).cast<float>() +
                             c[1] * (t1 - y).cast<float>() +
                             c[2] * (t2 - y).cast<float>();
    const FaceParam fparam = face_param[face_prop[face_index].param_index];
    const VertexParam vparam =
        static_vertex_param[static_vertex_prop[vertex_index].param_index];
    const float offset = fparam.offset + vparam.offset;
    const float ghat = 0.5f * (fparam.ghat + vparam.ghat);
    const float friction = combine_friction_values(
        fparam.friction, vparam.friction, friction_mode);
    const float reach = ghat + offset;
    if (!(separation.squaredNorm() < reach * reach)) {
        return false;
    }
    unsigned index[3];
    index[0] = fc[0];
    index[1] = fc[1];
    index[2] = fc[2];
    for (unsigned i = 0; i < 3; ++i) {
        out_slot[i] = index[i];
    }
    SVecf<3> weight;
    weight[0] = c[0];
    weight[1] = c[1];
    weight[2] = c[2];
    const bool accepted = embed_collision<3>(
        x0, x, vertex_prop, fixed_index, fixed_offset, fixed_value, row_count,
        friction_eps, residual, dt, index, weight, separation, ghat, offset, friction, fc[0],
        vertex_index, out_local_force, out_local_hessian, out_overlap, diag);
    if (out_overlap.flagged != 0u) {
        // Kind 8: the reference names this pair as (fc[0], vertex_index); the
        // second index is in the static collision-mesh vertex space.
        out_overlap.kind = 8u;
        out_overlap.elem0 = fc[0];
        out_overlap.elem1 = vertex_index;
    }
    // Counted at the ACCEPTED exit, where `contact.cu:960` records it.
    if (accepted) {
        statistics_record_dynamic_static_contact(
            statistics_contact_count, statistics_contact_count_size,
            statistics_object_index, statistics_object_index_size,
            statistics_static_object_index,
            statistics_static_object_index_size, index[0], vertex_index, diag);
    }
    return accepted;
}

// The pair form. `pair` is (collider vertex, face).
[[seam::entry(k)]]
[[seam::device_fn]] inline void collision_point_face_c2m(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const FaceParam *face_param,
    const Vec3f *static_x,
    const VertexProp *static_vertex_prop,
    const VertexParam *static_vertex_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    const unsigned *pair, unsigned k,
    unsigned *active, unsigned *arity,
    unsigned *out_index, float *out_force,
    float *out_hessian,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    CcdOverlapRecord *out_overlap, DiagHandle diag) {
    active[k] = 0u;
    arity[k] = 3u;
    unsigned slot[3];
    SMatf<3, 3> local_force;
    SMatf<9, 9> local_hessian;
    CcdOverlapRecord overlap;
    overlap.flagged = 0u;
    overlap.kind = 0u;
    overlap.elem0 = 0u;
    overlap.elem1 = 0u;
    overlap.d2 = 0.0f;
    overlap.offset = 0.0f;
    const bool accepted = collision_point_face_c2m_at(
        x0, x, face, vertex_prop, face_prop, face_param, static_x,
        static_vertex_prop, static_vertex_param, fixed_index, fixed_offset,
        fixed_value, row_count, friction_mode, friction_eps, residual, dt, pair[2 * k], pair[2 * k + 1], slot, local_force, local_hessian,
        statistics_contact_count, statistics_contact_count_size,
        statistics_object_index, statistics_object_index_size,
        statistics_static_object_index, statistics_static_object_index_size,
        overlap, diag);
    if (overlap.flagged != 0u && out_overlap[k].flagged == 0u) {
        out_overlap[k] = overlap;
    }
    if (accepted) {
        for (unsigned i = 0; i < 3; ++i) {
            out_index[4 * k + i] = slot[i];
        }
        for (unsigned i = 0; i < 3 * 3; ++i) {
            out_force[12 * k + i] = local_force.m[i];
        }
        for (unsigned i = 0; i < 9 * 3 * 3; ++i) {
            out_hessian[144 * k + i] = local_hessian.m[i];
        }
    }
    active[k] = accepted ? 1u : 0u;
}

// A dynamic edge against a collider edge, evaluated at ONE pair of indices.
// `collision_point_face_m2c_at` states why the result comes back in thread
// space.
[[seam::device_fn]] inline bool collision_edge_edge_at(
    const Vec3f *x0, const Vec3f *x,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const Vec3f *static_x,
    const Vec2u *static_edge,
    const EdgeProp *static_edge_prop,
    const EdgeParam *static_edge_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    unsigned edge_index, unsigned other_index,
    unsigned *out_slot,
    SMatf<3, 2> &out_local_force,
    SMatf<6, 6> &out_local_hessian,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    if (edge_prop[edge_index].fixed || !(edge_prop[edge_index].mass > 0.0f)) {
        return false;
    }
    const Vec2u me = edge[edge_index];
    const Vec2u ce = static_edge[other_index];
    const Vec3f q0 = static_x[ce[0]];
    const Vec3f q1 = static_x[ce[1]];
    // Centering on the collider edge makes every coordinate below a
    // displacement, so the scene's absolute magnitude never enters the
    // coefficients and cannot cancel against itself.
    const float midpoint_weight(0.5f);
    const Vec3f cog = midpoint_weight * q0 + midpoint_weight * q1;
    // Thread-space copies, for the reason `embed_collision` states above:
    // `x[me[0]]` is a DEVICE-space lvalue and the vector type's
    // `operator-` is declared for thread space alone. `q0` and `q1` are
    // already copies, which is why only the dynamic side needs this.
    const Vec3f p0 = x[me[0]];
    const Vec3f p1 = x[me[1]];
    const Vec3f p0c = p0 - cog;
    const Vec3f p1c = p1 - cog;
    const Vec3f q0c = q0 - cog;
    const Vec3f q1c = q1 - cog;
    const Vec4f c =
        proximity::edge_edge_distance_coeff<float, float>(p0c, p1c, q0c, q1c);
    const Vec3f left = c[0] * p0c.cast<float>() + c[1] * p1c.cast<float>();
    const Vec3f right = c[2] * q0c.cast<float>() + c[3] * q1c.cast<float>();
    const Vec3f separation = left - right;
    const EdgeParam eparam = edge_param[edge_prop[edge_index].param_index];
    const EdgeParam sparam =
        static_edge_param[static_edge_prop[other_index].param_index];
    const float offset = eparam.offset + sparam.offset;
    const float ghat = 0.5f * (eparam.ghat + sparam.ghat);
    const float friction = combine_friction_values(
        eparam.friction, sparam.friction, friction_mode);
    const float reach = ghat + offset;
    if (!(separation.squaredNorm() < reach * reach)) {
        return false;
    }
    unsigned index[2];
    index[0] = me[0];
    index[1] = me[1];
    for (unsigned i = 0; i < 2; ++i) {
        out_slot[i] = index[i];
    }
    SVecf<2> weight;
    weight[0] = c[0];
    weight[1] = c[1];
    const bool accepted = embed_collision<2>(
        x0, x, vertex_prop, fixed_index, fixed_offset, fixed_value, row_count,
        friction_eps, residual, dt, index, weight, separation, ghat, offset, friction,
        edge_index, other_index, out_local_force, out_local_hessian, out_overlap, diag);
    if (out_overlap.flagged != 0u) {
        // Kind 9: the reference names this pair as (me[0], ce[0]); the
        // second index is in the static collision-mesh vertex space.
        out_overlap.kind = 9u;
        out_overlap.elem0 = me[0];
        out_overlap.elem1 = ce[0];
    }
    // Counted at the ACCEPTED exit, where `contact.cu:960` records it.
    if (accepted) {
        statistics_record_dynamic_static_contact(
            statistics_contact_count, statistics_contact_count_size,
            statistics_object_index, statistics_object_index_size,
            statistics_static_object_index,
            statistics_static_object_index_size, index[0], ce[0], diag);
    }
    return accepted;
}

// The pair form. `pair` is (dynamic edge, collider edge).
[[seam::entry(k)]]
[[seam::device_fn]] inline void collision_edge_edge(
    const Vec3f *x0, const Vec3f *x,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const Vec3f *static_x,
    const Vec2u *static_edge,
    const EdgeProp *static_edge_prop,
    const EdgeParam *static_edge_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    const unsigned *pair, unsigned k,
    unsigned *active, unsigned *arity,
    unsigned *out_index, float *out_force,
    float *out_hessian,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    CcdOverlapRecord *out_overlap, DiagHandle diag) {
    active[k] = 0u;
    arity[k] = 2u;
    unsigned slot[2];
    SMatf<3, 2> local_force;
    SMatf<6, 6> local_hessian;
    CcdOverlapRecord overlap;
    overlap.flagged = 0u;
    overlap.kind = 0u;
    overlap.elem0 = 0u;
    overlap.elem1 = 0u;
    overlap.d2 = 0.0f;
    overlap.offset = 0.0f;
    const bool accepted = collision_edge_edge_at(
        x0, x, edge, vertex_prop, edge_prop, edge_param, static_x,
        static_edge, static_edge_prop, static_edge_param, fixed_index,
        fixed_offset, fixed_value, row_count, friction_mode, friction_eps, residual, dt, pair[2 * k], pair[2 * k + 1], slot, local_force, local_hessian,
        statistics_contact_count, statistics_contact_count_size,
        statistics_object_index, statistics_object_index_size,
        statistics_static_object_index, statistics_static_object_index_size,
        overlap, diag);
    if (overlap.flagged != 0u && out_overlap[k].flagged == 0u) {
        out_overlap[k] = overlap;
    }
    if (accepted) {
        for (unsigned i = 0; i < 2; ++i) {
            out_index[4 * k + i] = slot[i];
        }
        for (unsigned i = 0; i < 3 * 2; ++i) {
            out_force[12 * k + i] = local_force.m[i];
        }
        for (unsigned i = 0; i < 9 * 2 * 2; ++i) {
            out_hessian[144 * k + i] = local_hessian.m[i];
        }
    }
    active[k] = accepted ? 1u : 0u;
}

// The three entry points, one per pass.
//
// EACH NAMES ONLY THE ARRAYS ITS OWN PASS READS, which is why the static side
// appears in three different shapes: M2C reads the collider's faces, C2M reads
// its vertices, and edge-edge reads its edges. One record naming all of them
// would carry, for every pass, the arrays two other passes read.
//
// THE DYNAMIC `VertexProp` IS IN ALL THREE and the static one only in C2M,
// because the mass and the elastic snapshot are gathered over the DYNAMIC
// vertices alone. That asymmetry is the collider's definition, not an omission.
// A dynamic vertex against every collider face its box overlaps.
//
// THE FUSED FORM: the broad phase and the narrow phase are one dispatch, which
// is the reference's shape. `contact.cu` invokes its per-hit functor from
// inside the traversal and never builds a pair list, so nothing here is read
// back to the host between finding a candidate and depositing it.
struct CollisionPointFaceM2cEmbed {
    const Vec3f *x0;
    const Vec3f *x;
    const VertexProp *vertex_prop;
    const VertexParam *vertex_param;
    const Vec3f *static_x;
    const Vec3u *static_face;
    const FaceProp *static_face_prop;
    const FaceParam *static_face_param;
    const unsigned *fixed_index;
    const unsigned *fixed_offset;
    const float *fixed_value;
    unsigned row_count;
    unsigned friction_mode;
    float friction_eps;
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual;
    float dt;
    compute::atomic_float_t *out_vertex_force;
    compute::atomic_float_t *out_fixed_value;
    compute::atomic_uint_t *dyn_claim;
    unsigned *dyn_row;
    unsigned *dyn_column;
    float *dyn_block;
    unsigned dyn_capacity;
    compute::atomic_uint_t *statistics_contact_count;
    unsigned statistics_contact_count_size;
    const unsigned *statistics_object_index;
    unsigned statistics_object_index_size;
    const unsigned *statistics_static_object_index;
    unsigned statistics_static_object_index_size;
    // The per-query collapsed-separation slot; first writer wins per slot.
    CcdOverlapRecord *out_overlap;
    unsigned query_index;
    DiagHandle diag;
    unsigned count;
    // CANDIDATES VISITED, beside `count`'s ACCEPTED. The broad-phase number is
    // what the collision-mesh diagnostic reports, and a pass that found
    // candidates and accepted none is a different fact from a pass that found
    // nothing, which is the miswiring that line exists to catch.
    unsigned visited;

    // BOTH METHODS CARRY THE EXECUTION SPACE, for the reason `AabbPairCollect`
    // states: a member with no annotation is a HOST function to nvcc, and only
    // `--features cuda-abi` catches it.
    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        ++visited;
        unsigned slot[1];
        SMatf<3, 1> local_force;
        SMatf<3, 3> local_hessian;
        CcdOverlapRecord overlap;
        overlap.flagged = 0u;
        overlap.kind = 0u;
        overlap.elem0 = 0u;
        overlap.elem1 = 0u;
        overlap.d2 = 0.0f;
        overlap.offset = 0.0f;
        if (collision_point_face_m2c_at(
                x0, x, vertex_prop, vertex_param, static_x, static_face,
                static_face_prop, static_face_param, fixed_index, fixed_offset,
                fixed_value, row_count, friction_mode, friction_eps, residual, dt, query_index, primitive, slot, local_force,
                local_hessian, statistics_contact_count, statistics_contact_count_size,
                statistics_object_index, statistics_object_index_size,
                statistics_static_object_index,
                statistics_static_object_index_size, overlap, diag)) {
            deposit_collision<1>(slot, local_force, local_hessian, fixed_index,
                                fixed_offset, out_fixed_value, row_count,
                                out_vertex_force, dyn_claim, dyn_row,
                                dyn_column, dyn_block, dyn_capacity);
            ++count;
        }
        if (overlap.flagged != 0u && out_overlap[query_index].flagged == 0u) {
            out_overlap[query_index] = overlap;
        }
        return true;
    }
};

// A collider vertex against every dynamic face its box overlaps.
//
// THE FUSED FORM: the broad phase and the narrow phase are one dispatch, which
// is the reference's shape. `contact.cu` invokes its per-hit functor from
// inside the traversal and never builds a pair list, so nothing here is read
// back to the host between finding a candidate and depositing it.
struct CollisionPointFaceC2mEmbed {
    const Vec3f *x0;
    const Vec3f *x;
    const Vec3u *face;
    const VertexProp *vertex_prop;
    const FaceProp *face_prop;
    const FaceParam *face_param;
    const Vec3f *static_x;
    const VertexProp *static_vertex_prop;
    const VertexParam *static_vertex_param;
    const unsigned *fixed_index;
    const unsigned *fixed_offset;
    const float *fixed_value;
    unsigned row_count;
    unsigned friction_mode;
    float friction_eps;
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual;
    float dt;
    compute::atomic_float_t *out_vertex_force;
    compute::atomic_float_t *out_fixed_value;
    compute::atomic_uint_t *dyn_claim;
    unsigned *dyn_row;
    unsigned *dyn_column;
    float *dyn_block;
    unsigned dyn_capacity;
    compute::atomic_uint_t *statistics_contact_count;
    unsigned statistics_contact_count_size;
    const unsigned *statistics_object_index;
    unsigned statistics_object_index_size;
    const unsigned *statistics_static_object_index;
    unsigned statistics_static_object_index_size;
    // The per-query collapsed-separation slot; first writer wins per slot.
    CcdOverlapRecord *out_overlap;
    unsigned query_index;
    DiagHandle diag;
    unsigned count;
    // CANDIDATES VISITED, beside `count`'s ACCEPTED. The broad-phase number is
    // what the collision-mesh diagnostic reports, and a pass that found
    // candidates and accepted none is a different fact from a pass that found
    // nothing, which is the miswiring that line exists to catch.
    unsigned visited;

    // BOTH METHODS CARRY THE EXECUTION SPACE, for the reason `AabbPairCollect`
    // states: a member with no annotation is a HOST function to nvcc, and only
    // `--features cuda-abi` catches it.
    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        ++visited;
        unsigned slot[3];
        SMatf<3, 3> local_force;
        SMatf<9, 9> local_hessian;
        CcdOverlapRecord overlap;
        overlap.flagged = 0u;
        overlap.kind = 0u;
        overlap.elem0 = 0u;
        overlap.elem1 = 0u;
        overlap.d2 = 0.0f;
        overlap.offset = 0.0f;
        if (collision_point_face_c2m_at(
                x0, x, face, vertex_prop, face_prop, face_param, static_x,
                static_vertex_prop, static_vertex_param, fixed_index,
                fixed_offset, fixed_value, row_count, friction_mode,
                friction_eps, residual, dt, query_index, primitive, slot,
                local_force,
                local_hessian, statistics_contact_count, statistics_contact_count_size,
                statistics_object_index, statistics_object_index_size,
                statistics_static_object_index,
                statistics_static_object_index_size, overlap, diag)) {
            deposit_collision<3>(slot, local_force, local_hessian, fixed_index,
                                fixed_offset, out_fixed_value, row_count,
                                out_vertex_force, dyn_claim, dyn_row,
                                dyn_column, dyn_block, dyn_capacity);
            ++count;
        }
        if (overlap.flagged != 0u && out_overlap[query_index].flagged == 0u) {
            out_overlap[query_index] = overlap;
        }
        return true;
    }
};

// A dynamic edge against every collider edge its box overlaps.
//
// THE FUSED FORM: the broad phase and the narrow phase are one dispatch, which
// is the reference's shape. `contact.cu` invokes its per-hit functor from
// inside the traversal and never builds a pair list, so nothing here is read
// back to the host between finding a candidate and depositing it.
struct CollisionEdgeEdgeEmbed {
    const Vec3f *x0;
    const Vec3f *x;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const EdgeProp *edge_prop;
    const EdgeParam *edge_param;
    const Vec3f *static_x;
    const Vec2u *static_edge;
    const EdgeProp *static_edge_prop;
    const EdgeParam *static_edge_param;
    const unsigned *fixed_index;
    const unsigned *fixed_offset;
    const float *fixed_value;
    unsigned row_count;
    unsigned friction_mode;
    float friction_eps;
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual;
    float dt;
    compute::atomic_float_t *out_vertex_force;
    compute::atomic_float_t *out_fixed_value;
    compute::atomic_uint_t *dyn_claim;
    unsigned *dyn_row;
    unsigned *dyn_column;
    float *dyn_block;
    unsigned dyn_capacity;
    compute::atomic_uint_t *statistics_contact_count;
    unsigned statistics_contact_count_size;
    const unsigned *statistics_object_index;
    unsigned statistics_object_index_size;
    const unsigned *statistics_static_object_index;
    unsigned statistics_static_object_index_size;
    // The per-query collapsed-separation slot; first writer wins per slot.
    CcdOverlapRecord *out_overlap;
    unsigned query_index;
    DiagHandle diag;
    unsigned count;
    // CANDIDATES VISITED, beside `count`'s ACCEPTED. The broad-phase number is
    // what the collision-mesh diagnostic reports, and a pass that found
    // candidates and accepted none is a different fact from a pass that found
    // nothing, which is the miswiring that line exists to catch.
    unsigned visited;

    // BOTH METHODS CARRY THE EXECUTION SPACE, for the reason `AabbPairCollect`
    // states: a member with no annotation is a HOST function to nvcc, and only
    // `--features cuda-abi` catches it.
    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        ++visited;
        unsigned slot[2];
        SMatf<3, 2> local_force;
        SMatf<6, 6> local_hessian;
        CcdOverlapRecord overlap;
        overlap.flagged = 0u;
        overlap.kind = 0u;
        overlap.elem0 = 0u;
        overlap.elem1 = 0u;
        overlap.d2 = 0.0f;
        overlap.offset = 0.0f;
        if (collision_edge_edge_at(
                x0, x, edge, vertex_prop, edge_prop, edge_param, static_x,
                static_edge, static_edge_prop, static_edge_param, fixed_index,
                fixed_offset, fixed_value, row_count, friction_mode,
                friction_eps, residual, dt, query_index, primitive, slot,
                local_force,
                local_hessian, statistics_contact_count, statistics_contact_count_size,
                statistics_object_index, statistics_object_index_size,
                statistics_static_object_index,
                statistics_static_object_index_size, overlap, diag)) {
            deposit_collision<2>(slot, local_force, local_hessian, fixed_index,
                                fixed_offset, out_fixed_value, row_count,
                                out_vertex_force, dyn_claim, dyn_row,
                                dyn_column, dyn_block, dyn_capacity);
            ++count;
        }
        if (overlap.flagged != 0u && out_overlap[query_index].flagged == 0u) {
            out_overlap[query_index] = overlap;
        }
        return true;
    }
};

[[seam::entry(element)]]
[[seam::device_fn]] inline void collision_point_face_m2c_traverse(
    const Vec3f *x0, const Vec3f *x,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const Vec3f *static_x,
    const Vec3u *static_face,
    const FaceProp *static_face_prop,
    const FaceParam *static_face_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    const unsigned *node, unsigned node_count,
    const AABB *tree_aabb, unsigned root,
    const AABB *query,
    compute::atomic_uint_t *assembled,
    CcdOverlapRecord *out_overlap, DiagHandle diag,
    unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return;
    }
    CollisionPointFaceM2cEmbed embed{x0, x, vertex_prop, vertex_param, static_x, static_face, static_face_prop, static_face_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, friction_eps, residual, dt, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, statistics_static_object_index, statistics_static_object_index_size, out_overlap, element, diag, 0u, 0u};
    aabb_query(node, node_count, tree_aabb, root, embed, box, diag);
    if (embed.count > 0u) {
        compute::atomic_add(assembled, embed.count);
    }
    // SLOT 1 IS THE CANDIDATE COUNT. One buffer rather than two, because the
    // driver reads both back in the one download it already makes.
    if (embed.visited > 0u) {
        compute::atomic_add(assembled + 1, embed.visited);
    }
}

[[seam::entry(element)]]
[[seam::device_fn]] inline void collision_point_face_c2m_traverse(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const FaceParam *face_param,
    const Vec3f *static_x,
    const VertexProp *static_vertex_prop,
    const VertexParam *static_vertex_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    const unsigned *node, unsigned node_count,
    const AABB *tree_aabb, unsigned root,
    const AABB *query,
    compute::atomic_uint_t *assembled,
    CcdOverlapRecord *out_overlap, DiagHandle diag,
    unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return;
    }
    CollisionPointFaceC2mEmbed embed{x0, x, face, vertex_prop, face_prop, face_param, static_x, static_vertex_prop, static_vertex_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, friction_eps, residual, dt, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, statistics_static_object_index, statistics_static_object_index_size, out_overlap, element, diag, 0u, 0u};
    aabb_query(node, node_count, tree_aabb, root, embed, box, diag);
    if (embed.count > 0u) {
        compute::atomic_add(assembled, embed.count);
    }
    // SLOT 1 IS THE CANDIDATE COUNT. One buffer rather than two, because the
    // driver reads both back in the one download it already makes.
    if (embed.visited > 0u) {
        compute::atomic_add(assembled + 1, embed.visited);
    }
}

[[seam::entry(element)]]
[[seam::device_fn]] inline void collision_edge_edge_traverse(
    const Vec3f *x0, const Vec3f *x,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const Vec3f *static_x,
    const Vec2u *static_edge,
    const EdgeProp *static_edge_prop,
    const EdgeParam *static_edge_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    const float *residual, float dt,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    const unsigned *statistics_static_object_index,
    unsigned statistics_static_object_index_size,
    const unsigned *node, unsigned node_count,
    const AABB *tree_aabb, unsigned root,
    const AABB *query,
    compute::atomic_uint_t *assembled,
    CcdOverlapRecord *out_overlap, DiagHandle diag,
    unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return;
    }
    CollisionEdgeEdgeEmbed embed{x0, x, edge, vertex_prop, edge_prop, edge_param, static_x, static_edge, static_edge_prop, static_edge_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, friction_eps, residual, dt, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, statistics_static_object_index, statistics_static_object_index_size, out_overlap, element, diag, 0u, 0u};
    aabb_query(node, node_count, tree_aabb, root, embed, box, diag);
    if (embed.count > 0u) {
        compute::atomic_add(assembled, embed.count);
    }
    // SLOT 1 IS THE CANDIDATE COUNT. One buffer rather than two, because the
    // driver reads both back in the one download it already makes.
    if (embed.visited > 0u) {
        compute::atomic_add(assembled + 1, embed.visited);
    }
}
