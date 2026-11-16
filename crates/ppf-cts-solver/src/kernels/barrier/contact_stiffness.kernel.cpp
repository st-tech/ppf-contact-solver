// File: contact_stiffness.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::thread]]` is the address space of a reference parameter,
// which MSL requires on every reference and pointer type. The attribute sits
// after the template parameter list, where the declaration begins.
//
// No include of its own. `SMatf`, `SVecf`, `Vec3f` and `DiagHandle` arrive from
// whatever declares them for the backend that is compiling, which is data.hpp
// under nvcc and on the host and the shader prologue's aliases under MSL.
//
// `DiagHandle` is taken BY VALUE and never dereferenced here. A neutral body
// cannot name the diagnostic channel, the three targets binding three different
// things to it, so it carries the one alias through to `DIAG_ASSERT4` and the
// entry that owns the lane declares `[[seam::diag]]`.
//
// The division goes through `fmath::div` rather than the operator, because MSL
// spells the correctly rounded quotient `precise::divide` and the plain
// operator is a different function there.

template <unsigned N>
[[seam::device_fn]] inline float contact_stiffness(
    const SMatf<3 * N, 3 * N> &local_hessian,
    const SVecf<N> &weight,
    const SVecf<N> &mass,
    const Vec3f &edge, float offset, DiagHandle diag) {
    const float gap = edge.norm() - offset;
    // THE GUARD IS ON THE GAP THIS FUNCTION DIVIDES BY, not on its square.
    // Callers assert `squaredNorm() > offset * offset` before reaching here,
    // and that predicate is strictly weaker in fp32: a squared norm one ulp
    // above the rounded `offset * offset` has a square root that rounds back to
    // `offset` exactly, leaving `gap` at zero and `mass / gap_squared`
    // infinite. `barrier.cu:57` asserts this quantity for the same reason, and
    // that assert is live in the reference's release build.
    DIAG_ASSERT4(diag, gap > 0.0f, gap, edge.norm(), offset,
                 static_cast<float>(N));
    const float gap_squared = gap * gap;
    float maximum_mass = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        if (mass[i] > maximum_mass) {
            maximum_mass = mass[i];
        }
    }
    SVecf<3 * N> direction;
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            direction[3 * i + dimension] = weight[i] * edge[dimension];
        }
    }
    direction.normalize();
    float result = 0.0f;
    for (unsigned row = 0; row < 3 * N; ++row) {
        float product = 0.0f;
        for (unsigned column = 0; column < 3 * N; ++column) {
            product += local_hessian(row, column) * direction[column];
        }
        result += product * direction[row];
    }
    for (unsigned i = 0; i < N; ++i) {
        const float effective_mass =
            mass[i] > 0.0f ? mass[i] : maximum_mass;
        const float inertia = fmath::div(effective_mass, gap_squared);
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            const float component = direction[3 * i + dimension];
            result += inertia * component * component;
        }
    }
    return result;
}

// A RESTITUTION PAIR'S STIFFNESS, KEPT IN ITS TWO PARTS: the elastic
// `dir . H dir`, as `contact_stiffness` forms it, and an inertial mass to divide
// by the squared gap. A restitution contact integrates the barrier over the
// step's sweep of the gap (`floor_secant_push`), and the inertial part varies
// with the gap along that sweep while the elastic part is held, so it needs
// them apart. The inertial mass is the lighter side's (below), not the weighted
// average `contact_stiffness` uses, so only restitution pairs see it.
template <unsigned N>
[[seam::device_fn]] inline void contact_stiffness_parts(
    const SMatf<3 * N, 3 * N> &local_hessian,
    const SVecf<N> &weight,
    const SVecf<N> &mass,
    const Vec3f &edge, float &elastic, float &inertial_mass) {
    float maximum_mass = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        if (mass[i] > maximum_mass) {
            maximum_mass = mass[i];
        }
    }
    SVecf<3 * N> direction;
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            direction[3 * i + dimension] = weight[i] * edge[dimension];
        }
    }
    direction.normalize();
    elastic = 0.0f;
    for (unsigned row = 0; row < 3 * N; ++row) {
        float product = 0.0f;
        for (unsigned column = 0; column < 3 * N; ++column) {
            product += local_hessian(row, column) * direction[column];
        }
        elastic += product * direction[row];
    }
    // THE LIGHTER SIDE'S MASS, over the pair's two SIDES (the vertices of each
    // sign of `weight`; a restitution pair carries each body vertex's whole
    // body mass, so a side's mass is its largest), a massless side being
    // immovable and never the lighter. This is the mass the barrier must stop
    // within the gap when the other
    // side is held: at the floor or a static mesh it is the moving body's own
    // mass, for a light body on a heavy one about its own, and for equal
    // bodies the same as the vertex average used before. The average instead
    // sizes a light ball's barrier by the heavy slab under it, hundreds of
    // times too stiff, and the step no longer converges; the two-body reduced
    // mass is half as stiff for equal bodies, too soft for one resting on the
    // floor (measured: a cube stack gained 16% at dt 1e-2). It is per SIDE
    // rather than per vertex weight so that it stays fixed while the closest
    // point slides across a face: a stiffness that follows the weights
    // changes the stored barrier energy between steps, which the secant then
    // hands back as a gain.
    float side_mass[2] = {0.0f, 0.0f};
    for (unsigned i = 0; i < N; ++i) {
        if (weight[i] != 0.0f) {
            const unsigned side = weight[i] > 0.0f ? 0u : 1u;
            side_mass[side] = fmath::max(side_mass[side], mass[i]);
        }
    }
    if (side_mass[0] > 0.0f && side_mass[1] > 0.0f) {
        inertial_mass = fmath::min(side_mass[0], side_mass[1]);
    } else if (side_mass[0] > 0.0f || side_mass[1] > 0.0f) {
        inertial_mass = fmath::max(side_mass[0], side_mass[1]);
    } else {
        inertial_mass = maximum_mass;
    }
}
