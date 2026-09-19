// File: contact_narrow.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: the four narrow-phase contact visitors, one per pair
// kind, each turning a broad-phase candidate into an extended force and Hessian
// or into nothing.
//
// A COLLAPSED SEPARATION IS REPORTED, NOT ASSERTED. Every visitor reaches
// `embed_contact`, which requires the pair to be separated by more than the
// contact offset: at a non-positive gap the barrier is evaluated outside the
// domain it is defined on and the normal is a 0/0 normalize. Such a pair
// contributes nothing and writes a `CcdOverlapRecord` into its query's slot,
// the same record the CCD sweep writes, so the host ends the advance with a
// structured OverlappingStart naming the pair, which is the contract the rig
// pins. An assert would trap the CUDA device, be compiled out of the Windows
// build and not exist on Metal, so the outcome would differ by build; the
// record is the same on all three. The `DiagHandle` the bodies still take is
// for the checks that ARE invariants, the index bounds among them.
//
// THE ADJACENCY ARRAYS ARE GATED BY A RECORD SCALAR, NOT BY A NULL TEST. A
// scene either has a vertex-edge table or it does not, which is a property of
// the scene the driver knows before it dispatches, so it arrives as a flag and
// the body never asks whether a pointer is null. That keeps the branch a
// configuration the driver owns rather than a fact a body discovers, and it is
// what lets the same body compile for a target on which a buffer cannot be
// null at all.

// `distance.hpp` FIRST, and the order is load-bearing rather than tidy. A
// neutral kernel body includes no header of its own, so the type vocabulary
// (`Vec3f`, `Mat3x3f`, the property and parameter records) reaches it from
// whichever SHARED HEADER an includer pulls in first. Put a `.kernel.cpp` ahead of every shared header and
// the file compiles only for a translation unit that had already established
// the vocabulary, which the host build does and a bare `nvcc` on the rendering
// does not.
// THE TRAVERSAL, because the fused pass below IS its visitor. A kernel body
// brings no header of its own, so a rendering compiled ALONE, which is what
// `--features cuda-abi` hands nvcc, sees only what this file includes. The
// host build cannot show the omission: `entrypoints/entries.cpp` includes
// `data.hpp` first and every rendering it compiles inherits the vocabulary
// whatever its own order is.
#include "aabb_traversal.kernel.cpp"
#include "distance.hpp"
#include "pair_filter.kernel.cpp"

// The three barrier families `barrier_*` dispatches over, before the body
// that dispatches over them: it names them unqualified, so an includer supplies
// them, exactly as `strainlimiting/shell_strain.kernel.cpp` does.
#include "../barrier/cubic.hpp"
#include "../barrier/logarithm.hpp"
#include "../barrier/quadratic.hpp"

#include "../barrier/contact_barrier.kernel.cpp"
#include "../barrier/contact_stiffness.kernel.cpp"
// The SAND grain's contact-point slip and its share of the friction force. The
// bodies are inert for every pair that is not two grains, and this file names
// them, so it includes them.
#include "grain_pair.kernel.cpp"
#include "../energy/model/friction.kernel.cpp"
#include "analytic_contact.kernel.cpp"
#include "contact_assembly.kernel.cpp"
#include "contact_statistics.kernel.cpp"

// Two edges that meet at a vertex are adjacent rather than in contact: their
// closest points coincide at the shared vertex, so the separation is zero and
// no barrier is defined there.
[[seam::device_fn]] inline bool
edges_share_a_vertex(const Vec2u &a,
                         const Vec2u &b) {
    return a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1];
}

// `x[index] - x[anchor]`, summed against the proximity weights.
//
// THE SEPARATION IS A SUM OF DIFFERENCES, NEVER OF COORDINATES. The weights sum
// to zero, so the result is the separation between the two primitives and is
// invariant under translating the pair. Anchoring every term at `x[index[0]]`
// is what makes that invariance survive in fp32: no absolute coordinate enters
// the accumulation, so nothing large has to cancel at the end, and the result
// resolves to the float spacing at the separation rather than at the scene's
// own magnitude.
template <unsigned N>
[[seam::device_fn]] inline Vec3f
weighted_separation(const Vec3f *x,
                        const unsigned *index,
                        const SVecf<N> &weight) {
    Vec3f sum = Vec3f::Zero();
    const Vec3f anchor = x[index[0]];
    for (unsigned i = 0; i < N; ++i) {
        // THE ELEMENT IS COPIED INTO THREAD SPACE BEFORE IT IS DIFFERENCED. An
        // array reached through a `[[seam::device]]` base pointer yields a
        // DEVICE-space lvalue, and MSL applies `operator-` only to two
        // operands of one address space. The copy is what the generator emits
        // for a `[[seam::gather]]` parameter anyway, so it costs nothing, and
        // `x` stays a base pointer because `index` names its elements.
        const Vec3f xi = x[index[i]];
        sum += float(weight[i]) * (xi - anchor);
    }
    return sum;
}

// One contact's extended force and Hessian, spread over the N vertices its
// weights name and scaled by the contact multiplicity.
//
// The order of the four contributions is fixed, because `force` and `hessian`
// are fp32 running sums and reordering them is a different answer.
template <unsigned N>
[[seam::device_fn]] inline bool embed_contact(
    const Vec3f *x0, const Vec3f *x,
    const VertexProp *vertex_prop,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned barrier_id, float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, and the substep it is
    // divided by. It is a COPY of the force vector taken before this pass, not
    // the force vector itself: every contact here deposits into that vector
    // through atomics, so reading it directly would hand one pair a residual
    // another pair had already moved. The copy holds the momentum, elastic and
    // strain-limit terms plus the analytic and collision-mesh contacts, which
    // assemble first for exactly this reason.
    const float *residual, float dt,
    const unsigned *index,
    const SVecf<N> &weight, float ghat, float offset,
    float friction, unsigned multiplicity, bool include_friction, unsigned left,
    unsigned right,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // THE GRAIN PATH, inert for every pair that is not two SAND grains.
    // `spin_displacement` is `dt * (ra omega_a + rb omega_b) x n`, already
    // formed by the caller because only it knows which endpoints are grains;
    // `has_grain` is what turns the substitution on. The friction gradient and
    // its stiffness come back out because the torque is built from the force
    // that was actually deposited and the Schur block from the curvature that
    // was actually assembled.
    bool has_grain, const Vec3f &spin_displacement,
    Vec3f &out_friction_gradient,
    float &out_friction_stiffness,
    Vec3f &out_normal,
    // THE COLLAPSED-SEPARATION REPORT, written here and carried out by the
    // visitor into its query's slot. Same record the CCD sweep writes, so the
    // host reads both through one decoder; the assembly's kinds are 6 to 9
    // and its lengths are WORLD units where the sweep's are its rescaled ones.
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    SVecf<N> mass;
    SMatf<3 * N, 3 * N> local_hessian = SMatf<3 * N, 3 * N>::Zero();
    for (unsigned i = 0; i < N; ++i) {
        mass[i] = vertex_prop[index[i]].mass;
        for (unsigned j = 0; j < N; ++j) {
            local_hessian.template block<3, 3>(3 * i, 3 * j) =
                fixed_csr_read(fixed_index, fixed_offset, fixed_value,
                                   row_count, index[i], index[j]);
        }
    }
    const Vec3f ex0 = weighted_separation<N>(x0, index, weight);
    const Vec3f ex = weighted_separation<N>(x, index, weight);
    const Vec3f separation = ex.cast<float>();
    // A PAIR THAT HAS ALREADY CLOSED TO THE CONTACT OFFSET, AT THE START POSE OR
    // AT THE ITERATE, IS REPORTED, NEVER ASSERTED. The barrier is singular there
    // and the pair carries no usable force: it contributes nothing and writes
    // the record the host reads, so the advance ends with a structured
    // OverlappingStart naming the pair, which is the contract
    // `rig_coincident_contact_pair` pins. An assert would trap the CUDA device,
    // be compiled out of the Windows build and not exist on Metal, so the
    // outcome would differ by build; a record is the same on all three. The
    // record is kind 6 and names the pair by its first and last VERTEX,
    // `index[0]` and `index[N - 1]`.
    //
    // THE TEST IS `!(d2 > offset^2)`, AND IT FIRES ON A PAIR EXACTLY TOUCHING. That boundary is load-bearing, not incidental: two
    // coincident sheets at the shell default offset of zero sit exactly on it,
    // and the frontend's proximity scan refuses any positive offset for such a
    // pair, so exact coincidence at offset zero is the only "inside" state a
    // scene can be built in, and `<` would make the contract unobservable. It
    // is also the same value the CCD sweep's entry check fires on in the same
    // step (`accd.hpp`, "d2 <= offset^2"), so a pair reported here would have
    // ended the step at the line search a moment later: a stricter test buys
    // nothing, which was measured on `metal_allow_intersection_fixture`, whose
    // crossing pairs have a true distance of zero and fail the sweep on
    // exactly the runs a `<` here let past. The skip below shares the test,
    // `separation` being the vector the normal is `normalized()` from.
    const float ex0_d2 = ex0.cast<float>().squaredNorm();
    const float ex_d2 = separation.squaredNorm();
    const float offset2 = offset * offset;
    if (!(ex0_d2 > offset2) || !(ex_d2 > offset2)) {
        out_overlap.flagged = 1u;
        out_overlap.kind = 6u;
        out_overlap.elem0 = index[0];
        out_overlap.elem1 = index[N - 1];
        out_overlap.d2 = fmath::min(ex0_d2, ex_d2);
        out_overlap.offset = offset;
        (void)left;
        (void)right;
        return false;
    }
    const Vec3f relative_step = (ex - ex0).cast<float>();
    const Vec3f normal = separation.normalized();
    // A GRAIN PAIR'S FRICTION SEES THE CONTACT-POINT SLIP, not the relative
    // center step: the two surfaces meet at a point that each grain's own spin
    // is already carrying, and exactly that is subtracted here before the
    // friction term. The clamp inside keeps the pair from over-rolling and
    // letting friction propel them apart.
    const Vec3f slip =
        has_grain ? relative_step - grain_pair_clamp_spin(spin_displacement,
                                                              relative_step,
                                                              normal)
                  : relative_step;
    out_normal = normal;
    out_friction_gradient = Vec3f::Zero();
    out_friction_stiffness = 0.0f;
    const Barrier barrier = static_cast<Barrier>(barrier_id);
    const float stiffness = contact_stiffness<N>(local_hessian, weight, mass,
                                                     separation, offset, diag);
    Vec3f force =
        stiffness * barrier_edge_gradient(separation, ghat, offset, barrier);
    Mat3x3f hessian =
        stiffness * barrier_edge_hessian(separation, ghat, offset, barrier);
    if (include_friction) {
        Vec3f friction_gradient;
        Mat3x3f friction_hessian;
        Vec3f tangent;
        Mat3x3f projection;
        float lambda;
        float friction_stiffness;
        float normal_force;
        // A VERTEX WITHOUT A DEGREE OF FREEDOM CARRIES WEIGHT ZERO IN THE
        // PREDICTION. An exact fix pin has its row eliminated and a massless
        // vertex belongs to a static solid; neither moves however hard it is
        // pushed, so neither takes part in the slip the anchor predicts. This
        // is a separate weight vector from the embed's, which still spreads the
        // force over every vertex the pair names.
        SVecf<N> free_weight;
        for (unsigned i = 0; i < N; ++i) {
            const bool owns_dof =
                vertex_prop[index[i]].fix_index == 0u && mass[i] > 0.0f;
            free_weight[i] = owns_dof ? weight[i] : 0.0f;
        }
        Vec3f drive = Vec3f::Zero();
        float drive_stiffness = 0.0f;
        if (!friction_slip_prediction<N>(index, free_weight, mass,
                                             local_hessian, residual, normal,
                                             dt, drive, drive_stiffness)) {
            drive_stiffness = 0.0f;
        }
        friction_evaluate(force, slip, normal, friction, friction_eps, drive,
                              drive_stiffness, friction_gradient,
                              friction_hessian, lambda, friction_stiffness,
                              tangent, projection, normal_force);
        force += friction_gradient;
        hessian += friction_hessian;
        out_friction_gradient = friction_gradient;
        // THE SURROGATE'S STIFFNESS, NOT THE FORCE'S SECANT. The grain Schur
        // block pairs with the friction HESSIAN, so it must carry the curvature
        // that was assembled; `analytic_grain_schur` reads the Hessian itself
        // and is consistent for free.
        out_friction_stiffness = friction_stiffness;
    }
    SMatf<3, N> extended_force = SMatf<3, N>::Zero();
    SMatf<3 * N, 3 * N> extended_hessian = SMatf<3 * N, 3 * N>::Zero();
    extend_contact_force_hessian<N>(weight, force, hessian, extended_force,
                                        extended_hessian);
    // The multiplicity scale. A matrix times a scalar is that scalar applied
    // to each of its stored floats, and an `SMat`'s whole storage is one flat
    // `float[R * C]`, so the elementwise form is the same expression on the
    // same bytes.
    //
    // IT IS SPELLED HERE RATHER THAN TAKEN FROM `vec_add_scaled`, because the
    // source is a THREAD-space local while that body names a
    // `[[seam::device]]` source, and MSL has no conversion between the two.
    // The accumulate-into-zero shape is the one the shared body has, kept
    // rather than reduced to a plain store: the destination opens at zero and
    // a second contributor to the same element would need it.
    const float count = (float)multiplicity;
    // THE EMBED HAPPENS HERE, IN THIS THREAD: the force and the Hessian are
    // deposited out of the same registers that evaluated them. Nothing is
    // stored between evaluating a pair and depositing it, so this pair's 12
    // floats of force and 144 of Hessian never reach a buffer and never cross
    // the seam.
    //
    // THE READ AND THE WRITE NAME TWO DIFFERENT ALLOCATIONS. This body READS
    // the fixed matrix through `fixed_value` to build the elasticity-inclusive
    // stiffness above; it WRITES through `out_fixed_value`. Folding the two
    // into one would let a pair's own deposit change the stiffness a pair
    // evaluated later in the same pass reads.
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned d = 0; d < 3; ++d) {
            const float term = count * extended_force.m[3 * i + d];
            if (term != 0.0f) {
                compute::atomic_add(out_vertex_force + 3 * index[i] + d, term);
            }
        }
    }
    // Only the upper triangle is offered: the matrix stores one of each pair
    // and reaches the other through its transpose index. A block the fixed
    // pattern has no slot for belongs to the dynamic matrix, and losing that
    // routing loses a Hessian coupling.
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            const unsigned row = index[ii];
            const unsigned column = index[jj];
            if (row > column) {
                continue;
            }
            Mat3x3f block;
            for (unsigned c = 0; c < 3; ++c) {
                for (unsigned r = 0; r < 3; ++r) {
                    block.m[3 * c + r] =
                        count *
                        extended_hessian.m[(3 * N) * (3 * jj + c) + 3 * ii + r];
                }
            }
            if (fixed_csr_atomic_push(fixed_index, fixed_offset,
                                      out_fixed_value, row_count, row, column,
                                      block)) {
                continue;
            }
            const unsigned slot = compute::atomic_add(dyn_claim, 1u);
            if (slot >= dyn_capacity) {
                continue;
            }
            dyn_row[slot] = row;
            dyn_column[slot] = column;
            for (unsigned e = 0; e < 9; ++e) {
                dyn_block[9 * slot + e] = block.m[e];
            }
        }
    }
    return true;
}

// A vertex against a triangle. `pair` is (vertex, face).
//
// Every visitor writes `active` and `arity` for a REJECTED pair too, so a
// reader never has to guess a stride, and `arity` is the pair kind's own
// constant rather than a count of anything discovered here.
[[seam::device_fn]] inline bool contact_point_face_at(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const VertexParam *vertex_param,
    const FaceParam *face_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    const float *residual, float dt,
    unsigned first_index, unsigned second_index,
    unsigned *out_slot,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    for (unsigned i = 0; i < 4u; ++i) {
        out_slot[i] = 0u;
    }
    const unsigned vertex_index = first_index;
    const unsigned face_index = second_index;
    const Vec3u f = face[face_index];
    const bool either_dyn = vertex_prop[vertex_index].fix_index == 0u ||
                            face_prop[face_index].fixed == false;
    if (!contact_pair_admitted(either_dyn,
                                   vertex_prop[vertex_index].pdrd_body_index,
                                   vertex_prop[f[0]].pdrd_body_index,
                                   vertex_prop[vertex_index].collider,
                                   vertex_prop[f[0]].collider)) {
        return false;
    }
    if (f[0] == vertex_index || f[1] == vertex_index || f[2] == vertex_index) {
        return false;
    }
    const Vec3f p = x[vertex_index];
    const Vec3f t0 = x[f[0]];
    const Vec3f t1 = x[f[1]];
    const Vec3f t2 = x[f[2]];
    const Vec3f c =
        proximity::point_triangle_distance_coeff<float, float>(p, t0, t1,
                                                                   t2);
    if (!(c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f)) {
        return false;
    }
    const Vec3f e = c[0] * (p - t0).cast<float>() +
                    c[1] * (p - t1).cast<float>() +
                    c[2] * (p - t2).cast<float>();
    const VertexParam vparam =
        vertex_param[vertex_prop[vertex_index].param_index];
    const FaceParam fparam = face_param[face_prop[face_index].param_index];
    const float offset = vparam.offset + fparam.offset;
    const float ghat = 0.5f * (vparam.ghat + fparam.ghat);
    const float friction = combine_friction_values(
        vparam.friction, fparam.friction, friction_mode);
    const float reach = ghat + offset;
    if (!(e.squaredNorm() < reach * reach)) {
        return false;
    }
    unsigned index[4];
    index[0] = vertex_index;
    index[1] = f[0];
    index[2] = f[1];
    index[3] = f[2];
    SVecf<4> weight;
    weight[0] = 1.0f;
    weight[1] = -c[0];
    weight[2] = -c[1];
    weight[3] = -c[2];
    for (unsigned i = 0; i < 4; ++i) {
        out_slot[i] = index[i];
    }
    // The grain path's out-parameters need lvalues even where it is off. A
    // pair that is not two grains leaves all four untouched.
    const Vec3f no_spin = Vec3f::Zero();
    Vec3f unused_friction;
    float unused_lambda;
    Vec3f unused_normal;
    const bool accepted = embed_contact<4>(
                    x0, x, vertex_prop, fixed_index, fixed_offset, fixed_value,
                    row_count, barrier_id, friction_eps, residual, dt, index,
                    weight, ghat,
                    offset, friction, 1u, true, vertex_index, face_index,
                    out_vertex_force, out_fixed_value, dyn_claim, dyn_row,
                    dyn_column, dyn_block, dyn_capacity,
                    // INERT GRAIN ARGUMENTS: this pair is not two grains, so
                    // the slip substitution is off and the friction outputs are
                    // discarded. Only `contact_point_point` sets them live.
                    false, no_spin, unused_friction,
                    unused_lambda, unused_normal, out_overlap, diag)
                    ;
    // THE CONTACT IS COUNTED AT THE ACCEPTED EXIT, not at every candidate: the
    // embed returns false on a pair the proximity test rejects, and such a pair
    // is charged to nobody.
    if (accepted) {
        statistics_record_dynamic_contact(
            statistics_contact_count, statistics_contact_count_size,
            statistics_object_index, statistics_object_index_size, index[0],
            index[1], diag);
    }
    return accepted;
}

// A vertex against an edge. `pair` is (vertex, edge).
//
// THE MULTIPLICITY. A vertex sitting over an edge also sits over the faces that
// edge belongs to, and the same contact would otherwise be counted once per
// incident face. The edge carries the contact exactly where the face does not,
// counted by the faces whose own point-triangle region does NOT contain the
// vertex; a face that touches the vertex disables friction, because the face's
// own contact already carries it.
[[seam::device_fn]] inline bool contact_point_edge_at(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face, const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const VertexParam *vertex_param,
    const EdgeParam *edge_param,
    const unsigned *edge_face_index,
    const unsigned *edge_face_offset, unsigned has_edge_face,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    const float *residual, float dt,
    unsigned first_index, unsigned second_index,
    unsigned *out_slot,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    for (unsigned i = 0; i < 4u; ++i) {
        out_slot[i] = 0u;
    }
    const unsigned vertex_index = first_index;
    const unsigned edge_index = second_index;
    const Vec2u f = edge[edge_index];
    const bool either_dyn = vertex_prop[vertex_index].fix_index == 0u ||
                            edge_prop[edge_index].fixed == false;
    if (!contact_pair_admitted(either_dyn,
                                   vertex_prop[vertex_index].pdrd_body_index,
                                   vertex_prop[f[0]].pdrd_body_index,
                                   vertex_prop[vertex_index].collider,
                                   vertex_prop[f[0]].collider)) {
        return false;
    }
    if (f[0] == vertex_index || f[1] == vertex_index) {
        return false;
    }
    const Vec3f p = x[vertex_index];
    const Vec3f t0 = x[f[0]];
    const Vec3f t1 = x[f[1]];
    const Vec2f c =
        proximity::point_edge_distance_coeff<float, float>(p, t0, t1);
    if (!(c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f)) {
        return false;
    }
    const Vec3f e =
        c[0] * (p - t0).cast<float>() + c[1] * (p - t1).cast<float>();
    const VertexParam vparam =
        vertex_param[vertex_prop[vertex_index].param_index];
    const EdgeParam eparam = edge_param[edge_prop[edge_index].param_index];
    const float offset = vparam.offset + eparam.offset;
    const float ghat = 0.5f * (vparam.ghat + eparam.ghat);
    const float friction = combine_friction_values(
        vparam.friction, eparam.friction, friction_mode);
    const float reach = ghat + offset;
    if (!(e.squaredNorm() < reach * reach)) {
        return false;
    }
    unsigned multiplicity = 0u;
    bool include_friction = true;
    if (has_edge_face != 0u &&
        edge_face_offset[edge_index + 1] > edge_face_offset[edge_index]) {
        for (unsigned j = edge_face_offset[edge_index];
             j < edge_face_offset[edge_index + 1]; ++j) {
            const Vec3u g = face[edge_face_index[j]];
            if (g[0] == vertex_index || g[1] == vertex_index ||
                g[2] == vertex_index) {
                ++multiplicity;
                include_friction = false;
                continue;
            }
            const Vec3f g0 = x[g[0]];
            const Vec3f g1 = x[g[1]];
            const Vec3f g2 = x[g[2]];
            const Vec3f gc =
                proximity::point_triangle_distance_coeff<float, float>(
                    p, g0, g1, g2);
            if (gc.maxCoeff() < 1.0f && gc.minCoeff() > 0.0f) {
                continue;
            }
            ++multiplicity;
        }
    } else {
        multiplicity = 1u;
    }
    if (multiplicity == 0u) {
        return false;
    }
    unsigned index[4];
    index[0] = vertex_index;
    index[1] = f[0];
    index[2] = f[1];
    index[3] = f[1];
    SVecf<3> weight;
    weight[0] = 1.0f;
    weight[1] = -c[0];
    weight[2] = -c[1];
    for (unsigned i = 0; i < 4; ++i) {
        out_slot[i] = index[i];
    }
    // The grain path's out-parameters need lvalues even where it is off. A
    // pair that is not two grains leaves all four untouched.
    const Vec3f no_spin = Vec3f::Zero();
    Vec3f unused_friction;
    float unused_lambda;
    Vec3f unused_normal;
    const bool accepted = embed_contact<3>(
                    x0, x, vertex_prop, fixed_index, fixed_offset, fixed_value,
                    row_count, barrier_id, friction_eps, residual, dt, index,
                    weight, ghat,
                    offset, friction, multiplicity, include_friction,
                    vertex_index, edge_index,
                    out_vertex_force, out_fixed_value, dyn_claim, dyn_row,
                    dyn_column, dyn_block, dyn_capacity,
                    // INERT GRAIN ARGUMENTS: see the point-face site.
                    false, no_spin, unused_friction,
                    unused_lambda, unused_normal, out_overlap, diag)
                    ;
    // THE CONTACT IS COUNTED AT THE ACCEPTED EXIT, not at every candidate: the
    // embed returns false on a pair the proximity test rejects, and such a pair
    // is charged to nobody.
    if (accepted) {
        statistics_record_dynamic_contact(
            statistics_contact_count, statistics_contact_count_size,
            statistics_object_index, statistics_object_index_size, index[0],
            index[1], diag);
    }
    return accepted;
}

// A vertex against a vertex. `pair` is (query vertex, found vertex), and only
// the strictly-lower found index is taken, so each unordered pair is offered
// once.
//
// THE MULTIPLICITY, as for point-edge: the pair carries the contact only where
// the incident edges and faces of the other vertex do not, and an incident
// element that touches this vertex disables friction because that element's own
// contact already carries it. A vertex with no incident edge at all is a free
// grain, which carries it alone.
[[seam::device_fn]] inline bool contact_point_point_at(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face, const Vec2u *edge,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const unsigned *vertex_edge_index,
    const unsigned *vertex_edge_offset,
    unsigned has_vertex_edge,
    const unsigned *vertex_face_index,
    const unsigned *vertex_face_offset,
    unsigned has_vertex_face,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    // `dt` IS NOT REPEATED HERE. The grain block below already carries it, and
    // the friction anchor divides the pair's mass by the same substep.
    const float *residual,
    unsigned first_index, unsigned second_index,
    // THE SAND GRAIN INPUTS. A non-zero inverse inertia is what makes a vertex
    // a grain, and `dt` turns its angular velocity into the surface
    // displacement the friction term sees.
    const float *grain_inv_inertia,
    const Vec3f *grain_omega, float dt,
    unsigned *out_slot,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // PER PAIR SLOT, two endpoints each, which a scatter folds into the
    // per-vertex arrays the integrate reads. Written for every pair so the
    // scatter reads this iteration's values; a pair with no grain leaves zeros.
    float *out_grain_torque,
    float *out_grain_stiffness,
    float *out_grain_normal,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    for (unsigned i = 0; i < 4u; ++i) {
        out_slot[i] = 0u;
    }
    const unsigned vertex_index = first_index;
    const unsigned other = second_index;
    if (!(other < vertex_index)) {
        return false;
    }
    const bool either_dyn = vertex_prop[vertex_index].fix_index == 0u ||
                            vertex_prop[other].fix_index == 0u;
    if (!contact_pair_admitted(either_dyn,
                                   vertex_prop[vertex_index].pdrd_body_index,
                                   vertex_prop[other].pdrd_body_index,
                                   vertex_prop[vertex_index].collider,
                                   vertex_prop[other].collider)) {
        return false;
    }
    const Vec3f p = x[vertex_index];
    const Vec3f q = x[other];
    const Vec3f e = (p - q).cast<float>();
    const VertexParam vparam_a =
        vertex_param[vertex_prop[vertex_index].param_index];
    const VertexParam vparam_b = vertex_param[vertex_prop[other].param_index];
    const float offset = vparam_a.offset + vparam_b.offset;
    const float ghat = 0.5f * (vparam_a.ghat + vparam_b.ghat);
    const float friction = combine_friction_values(
        vparam_a.friction, vparam_b.friction, friction_mode);
    const float reach = ghat + offset;
    if (!(e.squaredNorm() < reach * reach)) {
        return false;
    }
    unsigned multiplicity = 0u;
    bool include_friction = true;
    const bool has_edges =
        has_vertex_edge != 0u &&
        vertex_edge_offset[other + 1] > vertex_edge_offset[other];
    if (has_edges) {
        for (unsigned j = vertex_edge_offset[other];
             j < vertex_edge_offset[other + 1]; ++j) {
            const Vec2u g = edge[vertex_edge_index[j]];
            if (g[0] == vertex_index || g[1] == vertex_index) {
                ++multiplicity;
                include_friction = false;
                continue;
            }
            const Vec3f g0 = x[g[0]];
            const Vec3f g1 = x[g[1]];
            const Vec2f gc =
                proximity::point_edge_distance_coeff<float, float>(
                    p, g0, g1);
            if (gc.maxCoeff() < 1.0f && gc.minCoeff() > 0.0f) {
                continue;
            }
            ++multiplicity;
        }
        if (has_vertex_face != 0u &&
            vertex_face_offset[other + 1] > vertex_face_offset[other]) {
            for (unsigned j = vertex_face_offset[other];
                 j < vertex_face_offset[other + 1]; ++j) {
                const Vec3u g = face[vertex_face_index[j]];
                if (g[0] == vertex_index || g[1] == vertex_index ||
                    g[2] == vertex_index) {
                    ++multiplicity;
                    include_friction = false;
                    continue;
                }
                const Vec3f g0 = x[g[0]];
                const Vec3f g1 = x[g[1]];
                const Vec3f g2 = x[g[2]];
                const Vec3f gc =
                    proximity::point_triangle_distance_coeff<float, float>(
                        p, g0, g1, g2);
                if (gc.maxCoeff() < 1.0f && gc.minCoeff() > 0.0f) {
                    continue;
                }
                ++multiplicity;
            }
        }
    } else {
        multiplicity = 1u;
    }
    if (multiplicity == 0u) {
        return false;
    }
    unsigned index[4];
    index[0] = vertex_index;
    index[1] = other;
    index[2] = other;
    index[3] = other;
    SVecf<2> weight;
    weight[0] = 1.0f;
    weight[1] = -1.0f;
    // THE ONLY PAIR TYPE A SAND GRAIN TAKES PART IN, because a grain is a
    // point. The slip substitution is gated on TWO things: at least one
    // endpoint being a grain, and the friction being included. A pass that
    // counted pairs before computing them would owe a third gate on being past
    // that counting pass; this narrow phase walks once and has none.
    const bool grain_left = grain_inv_inertia[vertex_index] > 0.0f;
    const bool grain_right = grain_inv_inertia[other] > 0.0f;
    const bool has_grain = (grain_left || grain_right) && include_friction;
    // THE MATERIAL TABLE IS INDEXED BY `param_index`, NEVER BY A VERTEX INDEX.
    // `vertex_param` is DEDUPLICATED across objects that share a material, so
    // it is shorter than the vertex array by however much sharing the scene
    // has, and subscripting it by a vertex id reads past its end on any scene
    // with more vertices than distinct materials. The two records were already
    // resolved through `param_index` above, so the radius is read off those.
    const float radius_left = vparam_a.offset;
    const float radius_right = vparam_b.offset;
    Vec3f spin_displacement = Vec3f::Zero();
    if (has_grain) {
        // The normal is formed the way the pair function forms it, from the
        // SAME difference of the same two positions, so the two cannot
        // disagree about which way the contact points.
        // COPIED INTO THREAD SPACE BEFORE THEY ARE DIFFERENCED. Under MSL
        // `x[i]` is a `const device` lvalue and there is no subtraction across
        // address spaces; the same applies to each grain's own omega below.
        const Vec3f left_position = x[vertex_index];
        const Vec3f right_position = x[other];
        const Vec3f pair_normal =
            (left_position - right_position).template cast<float>().normalized();
        Vec3f spin = Vec3f::Zero();
        if (grain_left) {
            const Vec3f omega_left = grain_omega[vertex_index];
            spin += radius_left * omega_left.cross(pair_normal);
        }
        if (grain_right) {
            const Vec3f omega_right = grain_omega[other];
            spin += radius_right * omega_right.cross(pair_normal);
        }
        spin_displacement = dt * spin;
    }
    Vec3f grain_friction = Vec3f::Zero();
    // THE SURROGATE'S STIFFNESS, which is what the angular Schur block pairs
    // with: the block is built from the friction HESSIAN, so it carries the
    // curvature that was assembled rather than the force's secant.
    float grain_stiffness_pair = 0.0f;
    Vec3f grain_normal = Vec3f::Zero();
    for (unsigned i = 0; i < 4; ++i) {
        out_slot[i] = index[i];
    }
    const bool accepted = embed_contact<2>(
                    x0, x, vertex_prop, fixed_index, fixed_offset, fixed_value,
                    row_count, barrier_id, friction_eps, residual, dt, index,
                    weight, ghat,
                    offset, friction, multiplicity, include_friction,
                    vertex_index, other, out_vertex_force, out_fixed_value,
                    dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity,
                    has_grain, spin_displacement,
                    grain_friction, grain_stiffness_pair, grain_normal, out_overlap, diag)
                    ;
    // THE CONTACT IS COUNTED AT THE ACCEPTED EXIT, not at every candidate: the
    // embed returns false on a pair the proximity test rejects, and such a pair
    // is charged to nobody.
    if (accepted) {
        statistics_record_dynamic_contact(
            statistics_contact_count, statistics_contact_count_size,
            statistics_object_index, statistics_object_index_size, index[0],
            index[1], diag);
    }

    // EACH GRAIN ENDPOINT'S SHARE OF THE FRICTION FORCE, turned back into the
    // torque, the angular stiffness and the contact normal the post-solve
    // integrate consumes. The multiplicity is the count the embed actually
    // deposited with, so the torque tracks the force that was applied rather
    // than the force of one nominal pair.
    //
    // THE SIGN THAT LOOKS WRONG AND IS NOT: both endpoints take `r (n x g)`
    // with the SAME normal and the SAME gradient, and `flip_normal` distinguishes
    // them, for the reason `grain_pair.kernel.cpp` states above the body.
    Vec3f torque_left = Vec3f::Zero();
    Vec3f torque_right = Vec3f::Zero();
    Vec3f normal_left = Vec3f::Zero();
    Vec3f normal_right = Vec3f::Zero();
    float stiffness_left = 0.0f;
    float stiffness_right = 0.0f;
    if (has_grain && accepted) {
        if (grain_left) {
            grain_pair_contribution(radius_left, false, grain_normal,
                                        grain_friction, grain_stiffness_pair,
                                        float(multiplicity), torque_left,
                                        stiffness_left, normal_left);
        }
        if (grain_right) {
            grain_pair_contribution(radius_right, true, grain_normal,
                                        grain_friction, grain_stiffness_pair,
                                        float(multiplicity), torque_right,
                                        stiffness_right, normal_right);
        }
    }
    for (unsigned c = 0; c < 3; ++c) {
        out_grain_torque[c] = torque_left[c];
        out_grain_torque[3 + c] = torque_right[c];
        out_grain_normal[c] = normal_left[c];
        out_grain_normal[3 + c] = normal_right[c];
    }
    out_grain_stiffness[0] = stiffness_left;
    out_grain_stiffness[1] = stiffness_right;
    return accepted;
}

// An edge against an edge. `pair` is (query edge, found edge), and only the
// strictly-greater found index is taken.
[[seam::device_fn]] inline bool contact_edge_edge_at(
    const Vec3f *x0, const Vec3f *x,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    const float *residual, float dt,
    unsigned first_index, unsigned second_index,
    unsigned *out_slot,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord &out_overlap, DiagHandle diag) {
    for (unsigned i = 0; i < 4u; ++i) {
        out_slot[i] = 0u;
    }
    const unsigned edge_index = first_index;
    const unsigned other = second_index;
    if (!(edge_index < other)) {
        return false;
    }
    const Vec2u e0 = edge[edge_index];
    const Vec2u e1 = edge[other];
    if (edges_share_a_vertex(e0, e1)) {
        return false;
    }
    const bool either_dyn = edge_prop[edge_index].fixed == false ||
                            edge_prop[other].fixed == false;
    if (!contact_pair_admitted(either_dyn,
                                   vertex_prop[e0[0]].pdrd_body_index,
                                   vertex_prop[e1[0]].pdrd_body_index,
                                   vertex_prop[e0[0]].collider,
                                   vertex_prop[e1[0]].collider)) {
        return false;
    }
    const Vec3f p0 = x[e0[0]];
    const Vec3f p1 = x[e0[1]];
    const Vec3f q0 = x[e1[0]];
    const Vec3f q1 = x[e1[1]];
    const Vec4f c =
        proximity::edge_edge_distance_coeff<float, float>(p0, p1, q0, q1);
    if (!(c.maxCoeff() < 1.0f && c.minCoeff() > 0.0f)) {
        return false;
    }
    const Vec3f y0 = float(c[0]) * p0 + float(c[1]) * p1;
    const Vec3f y1 = float(c[2]) * q0 + float(c[3]) * q1;
    const Vec3f e = (y0 - y1).cast<float>();
    const EdgeParam eparam_a = edge_param[edge_prop[edge_index].param_index];
    const EdgeParam eparam_b = edge_param[edge_prop[other].param_index];
    const float offset = eparam_a.offset + eparam_b.offset;
    const float ghat = 0.5f * (eparam_a.ghat + eparam_b.ghat);
    const float friction = combine_friction_values(
        eparam_a.friction, eparam_b.friction, friction_mode);
    const float reach = ghat + offset;
    if (!(e.squaredNorm() < reach * reach)) {
        return false;
    }
    unsigned index[4];
    index[0] = e0[0];
    index[1] = e0[1];
    index[2] = e1[0];
    index[3] = e1[1];
    SVecf<4> weight;
    weight[0] = c[0];
    weight[1] = c[1];
    weight[2] = -c[2];
    weight[3] = -c[3];
    for (unsigned i = 0; i < 4; ++i) {
        out_slot[i] = index[i];
    }
    // The grain path's out-parameters need lvalues even where it is off. A
    // pair that is not two grains leaves all four untouched.
    const Vec3f no_spin = Vec3f::Zero();
    Vec3f unused_friction;
    float unused_lambda;
    Vec3f unused_normal;
    const bool accepted = embed_contact<4>(
                    x0, x, vertex_prop, fixed_index, fixed_offset, fixed_value,
                    row_count, barrier_id, friction_eps, residual, dt, index,
                    weight, ghat,
                    offset, friction, 1u, true, edge_index, other,
                    out_vertex_force, out_fixed_value, dyn_claim, dyn_row,
                    dyn_column, dyn_block, dyn_capacity,
                    // INERT GRAIN ARGUMENTS: this pair is not two grains, so
                    // the slip substitution is off and the friction outputs are
                    // discarded. Only `contact_point_point` sets them live.
                    false, no_spin, unused_friction,
                    unused_lambda, unused_normal, out_overlap, diag)
                    ;
    // THE CONTACT IS COUNTED AT THE ACCEPTED EXIT, not at every candidate: the
    // embed returns false on a pair the proximity test rejects, and such a pair
    // is charged to nobody.
    if (accepted) {
        statistics_record_dynamic_contact(
            statistics_contact_count, statistics_contact_count_size,
            statistics_object_index, statistics_object_index_size, index[0],
            index[2], diag);
    }
    return accepted;
}

// The four entry points, one per pair kind.
//
// EVERY BUFFER IS A BASE POINTER. A visitor addresses `pair` at `2 * k` and
// `out_index` at `4 * k`, stages no force or Hessian at `144 * k`, and reaches
// the mesh, property and parameter arrays at indices the pair NAMES, so none of
// them is an element gather. What the record still
// carries per dispatch is the scene's own scalars: the CSR row count, the two
// `ParamSet` selectors and the friction floor, and the adjacency flags.
//
// THE `ParamSet` IS NOT IN THE RECORD, its three read fields are. A record
// holds 4-byte scalars, so passing the selectors directly costs three fields
// against one buffer and removes the record's dependency on that struct's
// layout entirely; nothing here needs the rest of it.
// The PAIR form, which is what the entry point dispatches: it reads this
// pair's two indices and hands them to the body above, so a caller that
// already HAS the two indices, such as a traversal visitor, reaches the
// same physics without a pair array existing at all.
//
// A NAME OF ITS OWN, because `check-shared-wiring.py` keys a neutral body
// on its name and cannot tell two rows apart. C++ would take both under
// one name by resolving on arity, and the gate would refuse it.
// ---------------------------------------------------------------------------
// The narrow phase AS THE TRAVERSAL'S VISITOR: the struct below is handed to
// `aabb_query`, so a hit is embedded where it is found and no pair list exists
// at any point.
// ---------------------------------------------------------------------------

struct ContactPointFaceEmbed {
    const Vec3f *x0;
    const Vec3f *x;
    const Vec3u *face;
    const VertexProp *vertex_prop;
    const FaceProp *face_prop;
    const VertexParam *vertex_param;
    const FaceParam *face_param;
    const unsigned *fixed_index;
    const unsigned *fixed_offset;
    const float *fixed_value;
    unsigned row_count;
    unsigned friction_mode;
    unsigned barrier_id;
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
    // The per-query collapsed-separation slot, first writer wins per slot
    // and a slot belongs to exactly one thread, as the CCD sweep's does.
    CcdOverlapRecord *out_overlap;
    unsigned query_index;
    DiagHandle diag;
    unsigned count;

    // BOTH METHODS CARRY THE EXECUTION SPACE, for the reason `AabbPairCollect`
    // states: a member with no annotation is a HOST function to nvcc, and only
    // `--features cuda-abi` catches it.
    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        unsigned slot[4];
        CcdOverlapRecord overlap;
        overlap.flagged = 0u;
        overlap.kind = 0u;
        overlap.elem0 = 0u;
        overlap.elem1 = 0u;
        overlap.d2 = 0.0f;
        overlap.offset = 0.0f;
        if (contact_point_face_at(x0, x, face, vertex_prop, face_prop, vertex_param, face_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, query_index, primitive, slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag)) {
            ++count;
        }
        if (overlap.flagged != 0u && out_overlap[query_index].flagged == 0u) {
            out_overlap[query_index] = overlap;
        }
        return true;
    }
};

// ONE THREAD PER QUERY VERTEX, traversing the face tree and embedding every hit
// where it finds it. NO PAIR ARRAY EXISTS IN THIS PATH: walking the tree into
// per-query slots, downloading them, compacting them on the host and uploading
// the compacted list back per chunk would move the whole query set across the
// bus twice per chunk, and none of it is needed to embed a hit at the point it
// is found.
//
// THE COUNT IS AN ATOMIC because there is no per-pair slot to sum over, and the
// assembly tally is all anything does with it.
[[seam::entry(element)]]
[[seam::device_fn]] inline void contact_point_face_traverse(
    const Vec3f *x0,
    const Vec3f *x,
    const Vec3u *face,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const VertexParam *vertex_param,
    const FaceParam *face_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    unsigned row_count,
    unsigned friction_mode,
    unsigned barrier_id,
    float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual, float dt,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row,
    unsigned *dyn_column,
    float *dyn_block,
    unsigned dyn_capacity,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
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
    ContactPointFaceEmbed embed{x0, x, face, vertex_prop, face_prop, vertex_param, face_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, out_overlap, element, diag, 0u};
    aabb_query(node, node_count, tree_aabb, root, embed, box, diag);
    if (embed.count > 0u) {
        compute::atomic_add(assembled, embed.count);
    }
}

[[seam::entry(k)]]
[[seam::device_fn]] inline void contact_point_face(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const VertexParam *vertex_param,
    const FaceParam *face_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual, float dt,
    const unsigned *pair, unsigned k,
    unsigned *active, unsigned *arity,
    unsigned *out_index,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord *out_overlap, DiagHandle diag) {
    arity[k] = 4u;
    unsigned slot[4];
    CcdOverlapRecord overlap;
    overlap.flagged = 0u;
    overlap.kind = 0u;
    overlap.elem0 = 0u;
    overlap.elem1 = 0u;
    overlap.d2 = 0.0f;
    overlap.offset = 0.0f;
    const bool accepted = contact_point_face_at(x0, x, face, vertex_prop, face_prop, vertex_param, face_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, pair[2 * k], pair[2 * k + 1], slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag);
    if (overlap.flagged != 0u && out_overlap[k].flagged == 0u) {
        out_overlap[k] = overlap;
    }
    for (unsigned i = 0; i < 4u; ++i) {
        out_index[4u * k + i] = slot[i];
    }
    active[k] = accepted ? 1u : 0u;
}

// The PAIR form, which is what the entry point dispatches: it reads this
// pair's two indices and hands them to the body above, so a caller that
// already HAS the two indices, such as a traversal visitor, reaches the
// same physics without a pair array existing at all.
//
// A NAME OF ITS OWN, because `check-shared-wiring.py` keys a neutral body
// on its name and cannot tell two rows apart. C++ would take both under
// one name by resolving on arity, and the gate would refuse it.
// The same fusion as point-face, for this kind. See `ContactPointFaceEmbed`.
struct ContactPointEdgeEmbed {
    const Vec3f *x0;
    const Vec3f *x;
    const Vec3u *face;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const EdgeProp *edge_prop;
    const VertexParam *vertex_param;
    const EdgeParam *edge_param;
    const unsigned *edge_face_index;
    const unsigned *edge_face_offset;
    unsigned has_edge_face;
    const unsigned *fixed_index;
    const unsigned *fixed_offset;
    const float *fixed_value;
    unsigned row_count;
    unsigned friction_mode;
    unsigned barrier_id;
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
    // The per-query collapsed-separation slot, first writer wins per slot
    // and a slot belongs to exactly one thread, as the CCD sweep's does.
    CcdOverlapRecord *out_overlap;
    unsigned query_index;
    DiagHandle diag;
    unsigned count;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        unsigned slot[4];
        CcdOverlapRecord overlap;
        overlap.flagged = 0u;
        overlap.kind = 0u;
        overlap.elem0 = 0u;
        overlap.elem1 = 0u;
        overlap.d2 = 0.0f;
        overlap.offset = 0.0f;
        if (contact_point_edge_at(x0, x, face, edge, vertex_prop, edge_prop, vertex_param, edge_param, edge_face_index, edge_face_offset, has_edge_face, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, query_index, primitive, slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag)) {
            ++count;
        }
        if (overlap.flagged != 0u && out_overlap[query_index].flagged == 0u) {
            out_overlap[query_index] = overlap;
        }
        return true;
    }
};

[[seam::entry(element)]]
[[seam::device_fn]] inline void contact_point_edge_traverse(
    const Vec3f *x0,
    const Vec3f *x,
    const Vec3u *face,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const VertexParam *vertex_param,
    const EdgeParam *edge_param,
    const unsigned *edge_face_index,
    const unsigned *edge_face_offset,
    unsigned has_edge_face,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    unsigned row_count,
    unsigned friction_mode,
    unsigned barrier_id,
    float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual, float dt,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row,
    unsigned *dyn_column,
    float *dyn_block,
    unsigned dyn_capacity,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
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
    ContactPointEdgeEmbed embed{x0, x, face, edge, vertex_prop, edge_prop, vertex_param, edge_param, edge_face_index, edge_face_offset, has_edge_face, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, out_overlap, element, diag, 0u};
    aabb_query(node, node_count, tree_aabb, root, embed, box, diag);
    if (embed.count > 0u) {
        compute::atomic_add(assembled, embed.count);
    }
}

[[seam::entry(k)]]
[[seam::device_fn]] inline void contact_point_edge(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face, const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const VertexParam *vertex_param,
    const EdgeParam *edge_param,
    const unsigned *edge_face_index,
    const unsigned *edge_face_offset, unsigned has_edge_face,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual, float dt,
    const unsigned *pair, unsigned k,
    unsigned *active, unsigned *arity,
    unsigned *out_index,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord *out_overlap, DiagHandle diag) {
    arity[k] = 3u;
    unsigned slot[4];
    CcdOverlapRecord overlap;
    overlap.flagged = 0u;
    overlap.kind = 0u;
    overlap.elem0 = 0u;
    overlap.elem1 = 0u;
    overlap.d2 = 0.0f;
    overlap.offset = 0.0f;
    const bool accepted = contact_point_edge_at(x0, x, face, edge, vertex_prop, edge_prop, vertex_param, edge_param, edge_face_index, edge_face_offset, has_edge_face, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, pair[2 * k], pair[2 * k + 1], slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag);
    if (overlap.flagged != 0u && out_overlap[k].flagged == 0u) {
        out_overlap[k] = overlap;
    }
    for (unsigned i = 0; i < 4u; ++i) {
        out_index[4u * k + i] = slot[i];
    }
    active[k] = accepted ? 1u : 0u;
}

// The PAIR form, which is what the entry point dispatches: it reads this
// pair's two indices and hands them to the body above, so a caller that
// already HAS the two indices, such as a traversal visitor, reaches the
// same physics without a pair array existing at all.
//
// A NAME OF ITS OWN, because `check-shared-wiring.py` keys a neutral body
// on its name and cannot tell two rows apart. C++ would take both under
// one name by resolving on arity, and the gate would refuse it.
// The same fusion as the other three, plus the SAND grain triple. See
// `ContactPointFaceEmbed`.
//
// THE GRAIN ARRAYS ARE PER-VERTEX AND ATOMIC: this visitor accumulates the
// three straight into the per-vertex arrays rather than writing per-pair slots
// a host pass then folds. Such a fold is what would need a pair list to exist.
struct ContactPointPointEmbed {
    const Vec3f *x0;
    const Vec3f *x;
    const Vec3u *face;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const VertexParam *vertex_param;
    const unsigned *vertex_edge_index;
    const unsigned *vertex_edge_offset;
    unsigned has_vertex_edge;
    const unsigned *vertex_face_index;
    const unsigned *vertex_face_offset;
    unsigned has_vertex_face;
    const unsigned *fixed_index;
    const unsigned *fixed_offset;
    const float *fixed_value;
    unsigned row_count;
    unsigned friction_mode;
    unsigned barrier_id;
    float friction_eps;
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual;
    float dt;
    const float *grain_inv_inertia;
    const Vec3f *grain_omega;
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
    compute::atomic_float_t *grain_torque_vertex;
    compute::atomic_float_t *grain_stiffness_vertex;
    compute::atomic_float_t *grain_normal_vertex;
    unsigned grains_present;
    // The per-query collapsed-separation slot, first writer wins per slot
    // and a slot belongs to exactly one thread, as the CCD sweep's does.
    CcdOverlapRecord *out_overlap;
    unsigned query_index;
    DiagHandle diag;
    unsigned count;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        unsigned slot[4];
        float torque[6];
        float stiffness[2];
        float normal[6];
        CcdOverlapRecord overlap;
        overlap.flagged = 0u;
        overlap.kind = 0u;
        overlap.elem0 = 0u;
        overlap.elem1 = 0u;
        overlap.d2 = 0.0f;
        overlap.offset = 0.0f;
        if (!contact_point_point_at(x0, x, face, edge, vertex_prop, vertex_param, vertex_edge_index, vertex_edge_offset, has_vertex_edge, vertex_face_index, vertex_face_offset, has_vertex_face, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, query_index, primitive, grain_inv_inertia, grain_omega, dt, slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, torque, stiffness, normal, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag)) {
            if (overlap.flagged != 0u && out_overlap[query_index].flagged == 0u) {
            out_overlap[query_index] = overlap;
        }
        return true;
        }
        ++count;
        if (grains_present == 0u) {
            return true;
        }
        // THE TWO ENDPOINTS, charged to their own vertices. The host fold this
        // replaces walked `index[4k + endpoint]` for endpoint 0 and 1, which is
        // point-point's arity, and added into the same three arrays.
        for (unsigned e = 0; e < 2u; ++e) {
            // `vert`, not `vertex`: MSL reserves that as a shader-stage
            // qualifier and the generator refuses it in a neutral body.
            const unsigned vert = slot[e];
            for (unsigned c = 0; c < 3u; ++c) {
                compute::atomic_add(grain_torque_vertex + 3u * vert + c,
                                    torque[3u * e + c]);
                compute::atomic_add(grain_normal_vertex + 3u * vert + c,
                                    normal[3u * e + c]);
            }
            compute::atomic_add(grain_stiffness_vertex + vert, stiffness[e]);
        }
        return true;
    }
};

[[seam::entry(element)]]
[[seam::device_fn]] inline void contact_point_point_traverse(
    const Vec3f *x0,
    const Vec3f *x,
    const Vec3u *face,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const unsigned *vertex_edge_index,
    const unsigned *vertex_edge_offset,
    unsigned has_vertex_edge,
    const unsigned *vertex_face_index,
    const unsigned *vertex_face_offset,
    unsigned has_vertex_face,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    unsigned row_count,
    unsigned friction_mode,
    unsigned barrier_id,
    float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual,
    const float *grain_inv_inertia,
    const Vec3f *grain_omega,
    float dt,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row,
    unsigned *dyn_column,
    float *dyn_block,
    unsigned dyn_capacity,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    compute::atomic_float_t *grain_torque_vertex,
    compute::atomic_float_t *grain_stiffness_vertex,
    compute::atomic_float_t *grain_normal_vertex,
    unsigned grains_present,
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
    ContactPointPointEmbed embed{x0, x, face, edge, vertex_prop, vertex_param, vertex_edge_index, vertex_edge_offset, has_vertex_edge, vertex_face_index, vertex_face_offset, has_vertex_face, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, grain_inv_inertia, grain_omega, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, grain_torque_vertex,
                                 grain_stiffness_vertex, grain_normal_vertex,
                                 grains_present, out_overlap, element, diag, 0u};
    aabb_query(node, node_count, tree_aabb, root, embed, box, diag);
    if (embed.count > 0u) {
        compute::atomic_add(assembled, embed.count);
    }
}

[[seam::entry(k)]]
[[seam::device_fn]] inline void contact_point_point(
    const Vec3f *x0, const Vec3f *x,
    const Vec3u *face, const Vec2u *edge,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const unsigned *vertex_edge_index,
    const unsigned *vertex_edge_offset,
    unsigned has_vertex_edge,
    const unsigned *vertex_face_index,
    const unsigned *vertex_face_offset,
    unsigned has_vertex_face,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual,
    const unsigned *pair, unsigned k,
    // THE SAND GRAIN INPUTS. A non-zero inverse inertia is what makes a vertex
    // a grain, and `dt` turns its angular velocity into the surface
    // displacement the friction term sees.
    const float *grain_inv_inertia,
    const Vec3f *grain_omega, float dt,
    unsigned *active, unsigned *arity,
    unsigned *out_index,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // PER PAIR SLOT, two endpoints each, which a scatter folds into the
    // per-vertex arrays the integrate reads. Written for every pair so the
    // scatter reads this iteration's values; a pair with no grain leaves zeros.
    float *out_grain_torque,
    float *out_grain_stiffness,
    float *out_grain_normal,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord *out_overlap, DiagHandle diag) {
    arity[k] = 2u;
    unsigned slot[4];
    // THE GRAIN TRIPLE, into this pair's own slot. The body yields it in
    // thread space so a caller with no pair slot, a traversal visitor, can
    // accumulate it into the per-vertex arrays with atomics instead.
    float grain_torque[6];
    float grain_stiffness[2];
    float grain_normal[6];
    CcdOverlapRecord overlap;
    overlap.flagged = 0u;
    overlap.kind = 0u;
    overlap.elem0 = 0u;
    overlap.elem1 = 0u;
    overlap.d2 = 0.0f;
    overlap.offset = 0.0f;
    const bool accepted = contact_point_point_at(x0, x, face, edge, vertex_prop, vertex_param, vertex_edge_index, vertex_edge_offset, has_vertex_edge, vertex_face_index, vertex_face_offset, has_vertex_face, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, pair[2 * k], pair[2 * k + 1], grain_inv_inertia, grain_omega, dt, slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, grain_torque, grain_stiffness, grain_normal, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag);
    if (overlap.flagged != 0u && out_overlap[k].flagged == 0u) {
        out_overlap[k] = overlap;
    }
    for (unsigned i = 0; i < 4u; ++i) {
        out_index[4u * k + i] = slot[i];
    }
    for (unsigned c = 0; c < 6u; ++c) {
        out_grain_torque[6u * k + c] = grain_torque[c];
        out_grain_normal[6u * k + c] = grain_normal[c];
    }
    out_grain_stiffness[2u * k] = grain_stiffness[0];
    out_grain_stiffness[2u * k + 1u] = grain_stiffness[1];
    active[k] = accepted ? 1u : 0u;
}

// The PAIR form, which is what the entry point dispatches: it reads this
// pair's two indices and hands them to the body above, so a caller that
// already HAS the two indices, such as a traversal visitor, reaches the
// same physics without a pair array existing at all.
//
// A NAME OF ITS OWN, because `check-shared-wiring.py` keys a neutral body
// on its name and cannot tell two rows apart. C++ would take both under
// one name by resolving on arity, and the gate would refuse it.
// The same fusion as point-face, for this kind. See `ContactPointFaceEmbed`.
struct ContactEdgeEdgeEmbed {
    const Vec3f *x0;
    const Vec3f *x;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const EdgeProp *edge_prop;
    const EdgeParam *edge_param;
    const unsigned *fixed_index;
    const unsigned *fixed_offset;
    const float *fixed_value;
    unsigned row_count;
    unsigned friction_mode;
    unsigned barrier_id;
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
    // The per-query collapsed-separation slot, first writer wins per slot
    // and a slot belongs to exactly one thread, as the CCD sweep's does.
    CcdOverlapRecord *out_overlap;
    unsigned query_index;
    DiagHandle diag;
    unsigned count;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        unsigned slot[4];
        CcdOverlapRecord overlap;
        overlap.flagged = 0u;
        overlap.kind = 0u;
        overlap.elem0 = 0u;
        overlap.elem1 = 0u;
        overlap.d2 = 0.0f;
        overlap.offset = 0.0f;
        if (contact_edge_edge_at(x0, x, edge, vertex_prop, edge_prop, edge_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, query_index, primitive, slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag)) {
            ++count;
        }
        if (overlap.flagged != 0u && out_overlap[query_index].flagged == 0u) {
            out_overlap[query_index] = overlap;
        }
        return true;
    }
};

[[seam::entry(element)]]
[[seam::device_fn]] inline void contact_edge_edge_traverse(
    const Vec3f *x0,
    const Vec3f *x,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    unsigned row_count,
    unsigned friction_mode,
    unsigned barrier_id,
    float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual, float dt,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row,
    unsigned *dyn_column,
    float *dyn_block,
    unsigned dyn_capacity,
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
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
    ContactEdgeEdgeEmbed embed{x0, x, edge, vertex_prop, edge_prop, edge_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, out_overlap, element, diag, 0u};
    aabb_query(node, node_count, tree_aabb, root, embed, box, diag);
    if (embed.count > 0u) {
        compute::atomic_add(assembled, embed.count);
    }
}

[[seam::entry(k)]]
[[seam::device_fn]] inline void contact_edge_edge(
    const Vec3f *x0, const Vec3f *x,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, unsigned barrier_id, float friction_eps,
    // THE RESIDUAL THE FRICTION ANCHOR IS READ OFF, a copy of the force vector
    // taken before this pass, and the substep it is divided by.
    const float *residual, float dt,
    const unsigned *pair, unsigned k,
    unsigned *active, unsigned *arity,
    unsigned *out_index,
    compute::atomic_float_t *out_vertex_force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *dyn_claim,
    unsigned *dyn_row, unsigned *dyn_column,
    float *dyn_block, unsigned dyn_capacity,
    // THE PER-OBJECT STATISTICS CHANNEL. Its two index arrays are its OWN and
    // are NOT `VertexProp::object_index`: a DIFFERENT index space, so handing
    // one of them the other's ids reads an entry that looks valid and charges
    // the contact to an unrelated object. `contact_statistics.kernel.cpp`
    // checks every subscript against the size passed beside it for that
    // reason. Absent configuration leaves the counter
    // zero-length, `statistics_enabled` reads false, and every recorder call
    // below is a predictable early return rather than a branch on a null.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    CcdOverlapRecord *out_overlap, DiagHandle diag) {
    arity[k] = 4u;
    unsigned slot[4];
    CcdOverlapRecord overlap;
    overlap.flagged = 0u;
    overlap.kind = 0u;
    overlap.elem0 = 0u;
    overlap.elem1 = 0u;
    overlap.d2 = 0.0f;
    overlap.offset = 0.0f;
    const bool accepted = contact_edge_edge_at(x0, x, edge, vertex_prop, edge_prop, edge_param, fixed_index, fixed_offset, fixed_value, row_count, friction_mode, barrier_id, friction_eps, residual, dt, pair[2 * k], pair[2 * k + 1], slot, out_vertex_force, out_fixed_value, dyn_claim, dyn_row, dyn_column, dyn_block, dyn_capacity, statistics_contact_count, statistics_contact_count_size, statistics_object_index, statistics_object_index_size, overlap, diag);
    if (overlap.flagged != 0u && out_overlap[k].flagged == 0u) {
        out_overlap[k] = overlap;
    }
    for (unsigned i = 0; i < 4u; ++i) {
        out_index[4u * k + i] = slot[i];
    }
    active[k] = accepted ? 1u : 0u;
}
