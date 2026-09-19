// File: rod_bend.kernel.cpp
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
// EVERY DIVISION AND EVERY ROOT HERE GOES THROUGH `fmath::div` AND
// `fmath::sqrt`, and the turning angle through `fmath::atan2`, which the
// backend prologue defines. That is a correctness spelling rather than a style:
// MSL's plain `/` and `sqrt` are not correctly rounded even at mathMode Safe,
// while `precise::divide` and `precise::sqrt` are in every math mode and CUDA's
// are already.

#include "../../contact/distance.hpp"
#include "../../csrmat/fixed_csr.kernel.cpp"
#include "../../utility/face_scatter.kernel.cpp"
#include "../../utility/rod_bend_damping.kernel.cpp"
#include "rod_bend_stiffness.kernel.cpp"

[[seam::device_fn]] inline float
rod_bend_angle(const Vec3f &x0,
                   const Vec3f &x1,
                   const Vec3f &x2) {
    const Vec3f edge0 = proximity::difference<float, float>(x0, x1);
    const Vec3f edge1 = proximity::difference<float, float>(x2, x1);
    return fmath::atan2(edge0.cross(edge1).norm(), edge0.dot(edge1));
}

[[seam::device_fn]] inline Mat3x2f
rod_bend_angle_gradient(const Vec3f &edge0,
                            const Vec3f &edge1) {
    const Vec3f cross = edge0.cross(edge1);
    const float cross_squared = cross.squaredNorm();
    const float edge0_squared = edge0.dot(edge0);
    const float edge1_squared = edge1.dot(edge1);
    if (cross_squared <= 1.0e-12f * edge0_squared * edge1_squared) {
        return Mat3x2f::Zero();
    }
    // DIVIDED, not scaled by a reciprocal, which is how `dihedral_angle.hpp`
    // writes all three: one IEEE rounding per component instead of two. The
    // shell path in this same tree already divides, so this also stops the two
    // bending arms from spelling the same normalization two ways.
    const Vec3f normal = cross / fmath::sqrt(cross_squared);
    const Vec3f gradient0 = edge0.cross(normal) / edge0_squared;
    const Vec3f gradient1 = -(edge1.cross(normal) / edge1_squared);
    Mat3x2f result;
    result.col(0) = gradient0;
    result.col(1) = gradient1;
    return result;
}

[[seam::device_fn]] inline Mat3x3f
rod_bend_force(const Vec3f &x0,
                   const Vec3f &x1,
                   const Vec3f &x2, float rest_angle) {
    const Vec3f edge0 = proximity::difference<float, float>(x0, x1);
    const Vec3f edge1 = proximity::difference<float, float>(x2, x1);
    const float angle =
        fmath::atan2(edge0.cross(edge1).norm(), edge0.dot(edge1));
    const Mat3x2f angle_gradient =
        rod_bend_angle_gradient(edge0, edge1);
    Mat3x3f result;
    result.col(0) = (angle - rest_angle) * angle_gradient.col(0);
    result.col(1) = (angle - rest_angle) *
                    (-angle_gradient.col(0) - angle_gradient.col(1));
    result.col(2) = (angle - rest_angle) * angle_gradient.col(1);
    return result;
}

[[seam::device_fn]] inline void rod_bend_accumulate_mode(
    float eigenvalue, const Vec3f &qa,
    const Vec3f &qb, Mat9x9f &hessian) {
    if (eigenvalue <= 0.0f) {
        return;
    }
    const float norm_squared = qa.dot(qa) + qb.dot(qb);
    if (norm_squared <= 1.0e-20f) {
        return;
    }
    float vector[9];
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        vector[dimension] = qa[dimension];
        vector[3 + dimension] = -qa[dimension] - qb[dimension];
        vector[6 + dimension] = qb[dimension];
    }
    const float scale = fmath::div(eigenvalue, norm_squared);
    for (unsigned row = 0; row < 9; ++row) {
        for (unsigned column = 0; column < 9; ++column) {
            hessian(row, column) +=
                scale * vector[row] * vector[column];
        }
    }
}

[[seam::device_fn]] inline void rod_bend_force_hessian(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, float rest_angle,
    Mat3x3f &force, Mat9x9f &hessian) {
    const Vec3f edge0 = proximity::difference<float, float>(x0, x1);
    const Vec3f edge1 = proximity::difference<float, float>(x2, x1);
    const Vec3f cross = edge0.cross(edge1);
    const float cross_norm = cross.norm();
    const float edge0_norm = edge0.norm();
    const float edge1_norm = edge1.norm();
    const float angle = fmath::atan2(cross_norm, edge0.dot(edge1));
    force = rod_bend_force(x0, x1, x2, rest_angle);
    hessian = Mat9x9f::Zero();
    if (cross_norm <= 1.0e-9f * edge0_norm * edge1_norm ||
        edge0_norm <= 1.0e-9f || edge1_norm <= 1.0e-9f) {
        return;
    }
    // COMPONENT-WISE DIVISION, NOT A PRECOMPUTED RECIPROCAL. `cross /
    // cross_norm` rounds once per component, where multiplying by a
    // precomputed `1 / cross_norm` would round the reciprocal first and then
    // each product. `length_hessian` further down is deliberately the other
    // shape, two separate `1 / (l * l)` reciprocals added, so do not fold it
    // into one division either.
    const Vec3f binormal = cross / cross_norm;
    const Vec3f edge0_perpendicular = edge0.cross(binormal);
    const Vec3f edge1_perpendicular = edge1.cross(binormal);
    const float cosine =
        fmath::div(edge0.dot(edge1), edge0_norm * edge1_norm);
    const float sine = fmath::div(cross_norm, edge0_norm * edge1_norm);
    const float angle_difference = angle - rest_angle;
    const float gamma =
        fmath::div(edge1_norm * edge1_norm, edge0_norm * edge0_norm);
    const float gamma_minus_one = gamma - 1.0f;
    const float gamma_plus_one = gamma + 1.0f;
    const float ratio = fmath::div(gamma_minus_one, gamma_plus_one);
    const float radical = fmath::sqrt(
        4.0f * angle_difference * angle_difference * ratio * ratio + 1.0f);
    const float r_minus = fmath::sqrt(fmath::max(
        0.0f, 2.0f *
                  (2.0f * angle_difference * angle_difference + 1.0f -
                   radical)));
    const float r_plus = fmath::sqrt(fmath::max(
        0.0f, 2.0f *
                  (2.0f * angle_difference * angle_difference + 1.0f +
                   radical)));
    const float length_hessian =
        fmath::div(1.0f, edge0_norm * edge0_norm) +
        fmath::div(1.0f, edge1_norm * edge1_norm);
    const float coupling =
        fmath::div(4.0f * angle_difference, gamma_plus_one);
    const int radical_sign[4] = {-1, -1, 1, 1};
    const float root[4] = {r_minus, r_minus, r_plus, r_plus};
    const int root_sign[4] = {-1, 1, -1, 1};
    for (unsigned mode = 0; mode < 4; ++mode) {
        const float root_value = root[mode];
        const float a =
            angle_difference *
            (fmath::div(gamma_minus_one * gamma_minus_one, gamma_plus_one) +
             static_cast<float>(radical_sign[mode]) * gamma_plus_one *
                 radical +
             static_cast<float>(root_sign[mode]) * gamma_minus_one *
                 root_value);
        const float b =
            -0.5f * gamma * root_value * root_value +
            2.0f * angle_difference * angle_difference -
            static_cast<float>(root_sign[mode]) * 0.5f *
                (gamma_minus_one +
                 static_cast<float>(radical_sign[mode]) * gamma_plus_one *
                     radical) *
                root_value;
        const float d =
            1.0f + static_cast<float>(radical_sign[mode]) * radical +
            static_cast<float>(root_sign[mode]) * root_value;
        rod_bend_accumulate_mode(
            0.25f * length_hessian * d,
            a * edge0 + b * edge0_perpendicular,
            coupling * edge1 + d * edge1_perpendicular, hessian);
    }
    const float beta =
        (fmath::div(edge1_norm, edge0_norm) -
         fmath::div(edge0_norm, edge1_norm)) *
        cosine;
    const float alpha =
        0.5f * (-beta + fmath::sqrt(beta * beta + 4.0f));
    const float inverse_sine_angle = fmath::div(angle_difference, sine);
    const float eigenvalue4 =
        inverse_sine_angle *
        (fmath::div(cosine, edge1_norm * edge1_norm) -
         fmath::div(alpha, edge0_norm * edge1_norm));
    const float eigenvalue5 =
        inverse_sine_angle *
        (fmath::div(cosine, edge1_norm * edge1_norm) +
         fmath::div(1.0f, alpha * edge0_norm * edge1_norm));
    rod_bend_accumulate_mode(eigenvalue4, alpha * binormal, binormal,
                                 hessian);
    rod_bend_accumulate_mode(eigenvalue5, binormal, -alpha * binormal,
                                 hessian);
}


// THE TURNING ANGLE AND THE BENDING PAIR, each reading its site's three nodes
// THROUGH the site's own index list. The bound beside that list is not
// bookkeeping: a slot is DATA rather than the thread index, so the count guard
// says nothing about it, and Metal answers an out-of-bounds read with 0.0 rather
// than faulting, which the range shims these replace could not check.
//
// THE ANGLE IS A SCATTER AND THE BENDING PAIR ARE NON-CONST GATHERS, which is
// the difference between a body that RETURNS its answer and one that writes two
// through references. A scatter carries one return value, so a body with two
// outputs cannot use it; a gather hands the body `buffer[index]`, an lvalue, so
// a non-const one IS the write. The pair are ASSIGNED rather than accumulated,
// which is why nothing has to be cleared before the dispatch.
[[seam::entry(count)]] void rod_bend_angle(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(3)]] const unsigned *node_index,
    [[seam::bound]] unsigned vertex_count,
    float *angle,
    unsigned count);

[[seam::entry(count)]] void rod_bend_force_hessian(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(3)]] const unsigned *node_index,
    [[seam::bound]] unsigned vertex_count,
    const float *rest_angle,
    Mat3x3f *force,
    Mat9x9f *hessian,
    unsigned count);

// THE WHOLE ROD BENDING TERM IN ONE THREAD: it reads the site's two incident
// edges and its interior vertex, forms the stiffness and the damping from their
// materials, evaluates the turning-angle force and Hessian, adds the lagged
// Rayleigh block and scatters both, under one dispatch.
//
// TWO GATES DECIDE WHICH SITES CONTRIBUTE. A prescribed interior vertex carries
// no bending, its row leaving the Newton system entirely, and a site whose
// stiffness comes out non-positive contributes nothing. Those two decide the
// set `builder.rs` registered stencils for.
//
// THE MATERIAL IS AVERAGED OVER THE TWO SEGMENTS, which is what
// `rod_bend_segment_average` states: a site sits BETWEEN two edges and its
// stiffness is a property of the pair rather than of either one.
[[seam::device_fn]] inline void rod_bend_embed(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const unsigned *node_slots,
    const unsigned *site_edge,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const VertexProp *vertex_prop, float dt,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness,
    const unsigned *hess_slots, unsigned has_hess_slots,
    unsigned element) {
    const unsigned interior = node_slots[3u * element + 1u];
    const VertexProp center = vertex_prop[interior];
    if (center.fix_index != 0u) {
        return;
    }
    const unsigned first = site_edge[2u * element];
    const unsigned second = site_edge[2u * element + 1u];
    const EdgeProp prop0 = edge_prop[first];
    const EdgeProp prop1 = edge_prop[second];
    const EdgeParam material0 = edge_param[prop0.param_index];
    const EdgeParam material1 = edge_param[prop1.param_index];
    const float bend =
        rod_bend_segment_average(material0.bend, material1.bend);
    const float stiffness =
        rod_bend_stiffness(bend, center.mass, prop0.length, prop1.length);
    if (!(stiffness > 0.0f)) {
        return;
    }
    Mat3x3f gradient;
    Mat9x9f hessian;
    rod_bend_force_hessian(x0, x1, x2, center.rest_bend_angle, gradient,
                           hessian);
    gradient *= stiffness;
    hessian *= stiffness;
    const float damping = rod_bend_segment_average(material0.bend_damping,
                                                  material1.bend_damping);
    if (damping > 0.0f) {
        // THE LAGGED POSE, scaled by the SAME stiffness, which is what makes
        // the turning-angle damping unconditionally dissipative. Its own force
        // is discarded exactly as `embed_rod_bend_force_hessian` discards
        // `f_lag`.
        Mat3x3f lagged_gradient;
        Mat9x9f lagged;
        rod_bend_force_hessian(current0, current1, current2,
                               center.rest_bend_angle, lagged_gradient, lagged);
        lagged *= stiffness;
        const Vec3f iterate[3] = {x0, x1, x2};
        const Vec3f start[3] = {current0, current1, current2};
        rod_bend_add_stiffness_damping_lagged(iterate, start, damping, dt,
                                              gradient, hessian, lagged);
    }
    Vec3u nodes;
    unsigned vertex_slots[3];
    for (unsigned k = 0; k < 3u; ++k) {
        vertex_slots[k] = node_slots[3u * element + k];
        nodes[k] = vertex_slots[k];
    }
    face_atomic_embed_force(nodes, gradient, force);
    if (has_hess_slots != 0u) {
        // THE TABLE REACHES THIS KERNEL ALREADY IN SITE ORDER.
        // `builder.rs` keys it by surface VERTEX, and `SolverState::allocate`
        // repacks it into site order because this kernel walks a compacted site
        // list. So the element index is the right key HERE, and re-deriving the
        // interior vertex would apply the repack twice.
        fixed_push_blocks_thread_at(hess_slots, hessian.m, 3u, fixed_value,
                                    element);
    } else {
        fixed_push_blocks_thread(vertex_slots, hessian.m, 3u, fixed_index,
                                 fixed_offset, fixed_value, row_count, refused,
                                 witness);
    }
}

[[seam::entry(count, element)]] void rod_bend_embed(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(3)]] const unsigned *node,
    [[seam::bound]] unsigned vertex_count,
    const unsigned *node_slots,
    const unsigned *site_edge,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const VertexProp *vertex_prop,
    float dt,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value,
    unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness,
    const unsigned *hess_slots,
    unsigned has_hess_slots,
    unsigned element,
    unsigned count);

