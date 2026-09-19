// File: plasticity.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read, and the build hands
// each compiler its own form. The two facts a backend cannot infer are written
// as C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a reference parameter. MSL
// requires the second on every reference and pointer type; CUDA and the host
// have one address space and are handed the same declarations with it removed.
//
// Every reference below is the caller's own value, a singular-value vector or a
// rest angle held in registers or on its stack, so all of them are
// `[[seam::thread]]`. Nothing here reads global or threadgroup memory: the
// entry point gathers the rest shape, calls these, and writes the result back.
//
// `fmath::exp` is the float-throughout exponential, not the library `expf`. The
// library form reduces and scales its argument in double arithmetic, so a
// kernel that merely calls it emits FP64 in the SASS with no `double` anywhere
// in the source, which the release build's guard rejects. What that trades is
// accuracy the creep rate does not need: the argument is `-plasticity * dt`
// and the result only sets how fast the rest shape follows the deformed one.

#include "../energy/model/rod_bend_stiffness.kernel.cpp"
#include "../float_math.hpp"

[[seam::device_fn]] inline float plasticity_alpha(float plasticity,
                                                     float dt) {
    return 1.0f - fmath::exp(-plasticity * dt);
}

[[seam::device_fn]] inline float
dead_zone_creep(float delta, float threshold, float alpha) {
    const float target = delta > 0.0f ? threshold : -threshold;
    return alpha * (delta - target);
}

template <unsigned N>
[[seam::device_fn]] inline bool plasticity_update_singular_values(
    const SVecf<N> &singular_values, float threshold,
    float alpha, SVecf<N> &updated) {
    updated = singular_values;
    bool changed = false;
    for (unsigned i = 0; i < N; ++i) {
        const float deviation = fmath::abs(singular_values[i] - 1.0f);
        if (deviation > threshold) {
            const float target =
                singular_values[i] < 1.0f ? 1.0f - threshold
                                          : 1.0f + threshold;
            updated[i] =
                singular_values[i] + alpha * (target - singular_values[i]);
            changed = true;
        }
    }
    return changed;
}

[[seam::device_fn]] inline bool plasticity_update_rest_angle(
    float angle, float threshold, float alpha,
    float &rest_angle) {
    const float delta = angle - rest_angle;
    if (fmath::abs(delta) <= threshold) {
        return false;
    }
    rest_angle += dead_zone_creep(delta, threshold, alpha);
    return true;
}

// The rest shape a crept singular-value spectrum implies, one body per element
// arity. Each takes the element's positions and the SVD of its deformation
// gradient with the crept singular values substituted, and returns the inverse
// rest matrix that reproduces that spectrum from the current pose.
//
// The positions are differenced before anything else is done with them, so the
// rest matrix is built out of EDGE vectors: the absolute coordinate cancels in
// `x1[d] - x0[d]` and never scales a product further down.

[[seam::device_fn]] inline Mat2x2f plasticity_face_inverse_rest(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Mat3x2f &left,
    const Vec2f &singular,
    const Mat2x2f &right) {
    Mat3x2f edges;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        edges(dimension, 0) = static_cast<float>(x1[dimension] - x0[dimension]);
        edges(dimension, 1) = static_cast<float>(x2[dimension] - x0[dimension]);
    }
    const Mat3x2f target = left * singular.asDiagonal() * right;
    // A 3x2 deformation gradient has no 3x3 inverse to undo, so the new rest
    // shape is expressed in the face's own tangent frame. Only the frame's ROW
    // SPACE is load-bearing: replacing the frame by any invertible combination
    // of its rows leaves the result unchanged, since the same factor appears in
    // the inverse and cancels. A frame whose rows leave the tangent plane does
    // not cancel, and is what this arrangement has to avoid.
    const Vec3f area_vector = edges.col(0).cross(edges.col(1));
    const Vec3f face_normal = area_vector.normalized();
    const Vec3f tangent = edges.col(0).normalized();
    const Vec3f bitangent = face_normal.cross(tangent).normalized();
    SMatf<2, 3> frame;
    frame.row(0) = tangent.transpose();
    frame.row(1) = bitangent.transpose();
    const Mat2x2f projected_edges = frame * edges;
    return projected_edges.inverse() * (frame * target);
}

[[seam::device_fn]] inline Mat3x3f plasticity_tet_inverse_rest(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    const Mat3x3f &left, const Vec3f &singular,
    const Mat3x3f &right) {
    Mat3x3f edges;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        edges(dimension, 0) = static_cast<float>(x1[dimension] - x0[dimension]);
        edges(dimension, 1) = static_cast<float>(x2[dimension] - x0[dimension]);
        edges(dimension, 2) = static_cast<float>(x3[dimension] - x0[dimension]);
    }
    const Mat3x3f target = left * singular.asDiagonal() * right;
    // A tet's edge matrix is square and invertible for any non-degenerate
    // element, so the new rest shape needs no frame.
    return edges.inverse() * target;
}

// The entry point for the creep rate, declared once and rendered for four
// targets. One element gather, one scalar and one element scatter: `dt` is the
// step and is the same for every element, so it arrives in the record itself
// rather than through a buffer.
//
// The three bodies still declaring no entry return a bool the launcher widens
// to a byte, which is a conversion an entry point cannot express.
[[seam::entry(count)]] void plasticity_alpha(
    const float *plasticity, float dt,
    float *alpha,
    unsigned count);

// The two inverse-rest entry points, which read their positions THROUGH the
// element's own index list.
//
// Each writes the rest shape a crept spectrum implies to a SEPARATE output
// rather than over `inv_rest*` in place. The reference kernel returns before
// this point when no singular value crossed its threshold, so an element that
// did not yield must keep the exact bytes it had; recomputing its rest matrix
// from the SVD would return a value differing in the last bits. The caller
// copies only where `changed` says the element yielded.
//
// `x` is [[seam::through]] the [[seam::indices(N)]] list, so the entry reads
// the element's N slots, checks each against `vertex_count`, and hands the body
// the N positions they name as N arguments. The body is unchanged by that and
// does no index arithmetic: it takes the positions, which is also how the fused
// CUDA and Metal kernels call it, so one body serves both arrangements.
//
// THE BOUND IS WHAT THE ENTRY FORM ADDS over passing `x` as a base pointer and
// letting the body subscript it. A slot is data rather than the thread index,
// so the [[seam::count]] guard says nothing about it; Metal returns 0.0 for an
// out-of-bounds read rather than faulting, which would make a corrupt index
// list a plausible rest shape instead of a stopped run.
[[seam::entry(count)]] void plasticity_face_inverse_rest(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const Mat3x2f *u,
    const Vec2f *singular,
    const Mat2x2f *vt,
    Mat2x2f *inverse_rest,
    unsigned count);

[[seam::entry(count)]] void plasticity_tet_inverse_rest(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(4)]] const unsigned *tet,
    [[seam::bound]] unsigned vertex_count,
    const Mat3x3f *u,
    const Vec3f *singular,
    const Mat3x3f *vt,
    Mat3x3f *inverse_rest,
    unsigned count);

// THE THREE CREEP PASSES AS ENTRY POINTS, each wrapping one of the templates
// above at the arity its element has.
//
// THE VERDICT IS AN `unsigned`, which is what the two other per-element
// verdicts in this tree already are: the tet material table's `accepted` and
// the face table's `dispatch`. A creep pass answers the same kind of question
// they do, one word per element saying which way the branch went, so it answers
// it in the same width. The templates return `bool` because they are also
// called from the fused arrangements, so the widening happens here, once, at
// the boundary where the answer becomes memory.
//
// EACH IS A NON-TEMPLATE WRAPPER because an entry point is one launch shape:
// there is nowhere in a declaration to put a template argument, and a second
// arity is a second declaration rather than a parameter. `SVecf<2>` and
// `SVecf<3>` are spelled `Vec2f` and `Vec3f` here, which are the same types
// under aliases a parameter list may name.
[[seam::device_fn]] inline unsigned plasticity_creep_singular2(
    const Vec2f &singular_values, float threshold, float alpha,
    Vec2f &updated) {
    return plasticity_update_singular_values<2>(singular_values, threshold,
                                                    alpha, updated)
               ? 1u
               : 0u;
}

[[seam::device_fn]] inline unsigned plasticity_creep_singular3(
    const Vec3f &singular_values, float threshold, float alpha,
    Vec3f &updated) {
    return plasticity_update_singular_values<3>(singular_values, threshold,
                                                    alpha, updated)
               ? 1u
               : 0u;
}

// `rest_angle` is READ AND WRITTEN at the same element, which is the in-place
// update the shared body performs: a value inside the dead zone is left exactly
// as it was. That is `[[seam::gather]]` and `[[seam::scatter]]` would be on one
// parameter for a returned value, but the return here is the verdict, so the
// angle is a non-const gather and the verdict is the scatter.
[[seam::device_fn]] inline unsigned plasticity_creep_rest_angle(
    float angle, float threshold, float alpha,
    float &rest_angle) {
    return plasticity_update_rest_angle(angle, threshold, alpha, rest_angle)
               ? 1u
               : 0u;
}

[[seam::entry(count)]] void plasticity_creep_singular2(
    const Vec2f *singular_values,
    const float *threshold,
    const float *alpha,
    Vec2f *updated,
    [[seam::scatter]] unsigned *changed,
    unsigned count);

[[seam::entry(count)]] void plasticity_creep_singular3(
    const Vec3f *singular_values,
    const float *threshold,
    const float *alpha,
    Vec3f *updated,
    [[seam::scatter]] unsigned *changed,
    unsigned count);

[[seam::entry(count)]] void plasticity_creep_rest_angle(
    const float *angle,
    const float *threshold,
    const float *alpha,
    float *rest_angle,
    [[seam::scatter]] unsigned *changed,
    unsigned count);

// ---------------------------------------------------------------------------
// THE COMMIT, which publishes a crept row into the rest shape the elastic
// kernels read.
//
// IT RUNS ON THE DEVICE, like every other pass over a per-element array. The
// host spelling would download the verdict and the crept rows, walk the active
// list serially, copy the changed rows into the destination and upload it,
// three times over. This one runs once per frame rather than once per Newton
// iteration, so those transfers would be cheap, and it would still be a
// per-element array coming down for a host loop to rewrite.
//
// TWO OF THE THREE CREEP PASSES COMMIT HERE, and the hinge does not, which is a
// data layout rather than an omission: its destination is `HingeProp::rest_angle`,
// a MEMBER of another array's element, and a record addresses one allocation per
// field. Writing the kernel anyway would leave one compiled, wired and reached
// by nothing, which is worse than the host copy it would replace.
//
// TWO GATES, NOT ONE, and both are the host loop's. `plasticity > 0` is what
// made an element ACTIVE, and `changed` is what the creep decided; an element
// that fails either keeps the bytes it had. Reading only `changed` would be
// right for every scene whose inactive elements leave it zero and would depend
// on that rather than state it.

[[seam::entry(face)]]
[[seam::device_fn]] inline void plasticity_commit_face(
    const float *plasticity,
    const unsigned *changed,
    const float *inverse_rest,
    float *destination, unsigned face) {
    if (!(plasticity[face] > 0.0f) || changed[face] == 0u) {
        return;
    }
    for (unsigned k = 0; k < 4u; ++k) {
        destination[4u * face + k] = inverse_rest[4u * face + k];
    }
}

[[seam::entry(tet)]]
[[seam::device_fn]] inline void plasticity_commit_tet(
    const float *plasticity,
    const unsigned *changed,
    const float *inverse_rest,
    float *destination, unsigned tet) {
    if (!(plasticity[tet] > 0.0f) || changed[tet] == 0u) {
        return;
    }
    for (unsigned k = 0; k < 9u; ++k) {
        destination[9u * tet + k] = inverse_rest[9u * tet + k];
    }
}


// THE CREEP MATERIAL, GATHERED FROM THE ELEMENT'S OWN RECORDS.
//
// `plasticity.cu` reads `param_face[prop_face[i].param_index]` INSIDE its
// dispatch and gates there on `plasticity <= 0.0f`; this driver walked every
// element on the host every step, gathered the same two floats through the
// same indirection, and uploaded two arrays. The walk ran even on a scene
// where no material creeps, because the early-out that skips the dispatches
// sat AFTER it.
//
// FOUR BODIES RATHER THAN ONE, because the record types differ and a neutral
// body cannot be generic over them. The arithmetic is the same three lines in
// each, which is the price of the four record vocabularies.
//
// THE SEED IS WRITTEN UNCONDITIONALLY: a zero rate is what the downstream
// alpha pass turns into a zero creep, so an element whose material does not
// creep must read zero rather than the previous step's value.
[[seam::entry(element)]]
[[seam::device_fn]] inline void plasticity_face_from_records(
    const FaceProp &prop,
    const FaceParam *face_param,
    float *plasticity, float *threshold,
    unsigned element) {
    const FaceParam material = face_param[prop.param_index];
    const bool creeps = material.plasticity > 0.0f;
    plasticity[element] = creeps ? material.plasticity : 0.0f;
    threshold[element] = creeps ? material.plasticity_threshold : 0.0f;
}

[[seam::device_fn]] inline void plasticity_tet_from_records(
    const TetProp &prop,
    const TetParam *tet_param,
    float *plasticity, float *threshold,
    unsigned element) {
    const TetParam material = tet_param[prop.param_index];
    const bool creeps = material.plasticity > 0.0f;
    plasticity[element] = creeps ? material.plasticity : 0.0f;
    threshold[element] = creeps ? material.plasticity_threshold : 0.0f;
}

// THE ROD ONE IS PER SITE, NOT PER EDGE, and that is not a detail. A bending
// site sits BETWEEN two edges and its creep rate is the mean of theirs, which
// is what `rod_bend_segment_average` states for the stiffness; a per-edge
// gather would be a different quantity. It reads the site's pair through the
// same list `rod_bend_embed` walks.
[[seam::device_fn]] inline void plasticity_rod_from_records(
    const unsigned *site_edge,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    float *plasticity, float *threshold,
    unsigned element) {
    const unsigned first = site_edge[2u * element];
    const unsigned second = site_edge[2u * element + 1u];
    const EdgeParam a = edge_param[edge_prop[first].param_index];
    const EdgeParam b = edge_param[edge_prop[second].param_index];
    const float rate = rod_bend_segment_average(a.plasticity, b.plasticity);
    const bool creeps = rate > 0.0f;
    plasticity[element] = creeps ? rate : 0.0f;
    threshold[element] =
        creeps ? rod_bend_segment_average(a.plasticity_threshold,
                                          b.plasticity_threshold)
               : 0.0f;
}

// THE HINGE GATE CONSULTS `fixed` AND THE FACE GATE DOES NOT, which is a real
// difference between the two layers rather than an oversight in either. A
// prescribed hinge has no rest angle to creep toward, while a fixed FACE still
// has a rest shape a later unpinning springs back to, so the face pass takes
// `plasticity > 0` as its only test. Dropping this here passes every test
// except `a_fixed_hinge_does_not_creep`, which is the one that states it.
[[seam::entry(element)]]
[[seam::device_fn]] inline void plasticity_hinge_from_records(
    const HingeProp &prop,
    const HingeParam *hinge_param,
    float *plasticity, float *threshold,
    unsigned element) {
    if (prop.fixed) {
        plasticity[element] = 0.0f;
        threshold[element] = 0.0f;
        return;
    }
    const HingeParam material = hinge_param[prop.param_index];
    const bool creeps = material.plasticity > 0.0f;
    plasticity[element] = creeps ? material.plasticity : 0.0f;
    threshold[element] = creeps ? material.plasticity_threshold : 0.0f;
}

[[seam::entry(count, element)]] void plasticity_tet_from_records(
    const TetProp *prop,
    const TetParam *tet_param,
    float *plasticity,
    float *threshold,
    unsigned element,
    unsigned count);

[[seam::entry(count, element)]] void plasticity_rod_from_records(
    const unsigned *site_edge,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    float *plasticity,
    float *threshold,
    unsigned element,
    unsigned count);

