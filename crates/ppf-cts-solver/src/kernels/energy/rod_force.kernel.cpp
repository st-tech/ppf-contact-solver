// File: rod_force.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. `[[seam::device_fn]]` is the
// execution space and `[[seam::thread]]` is the address space MSL requires on
// every reference.
//
// THE ROD SEGMENT'S STRETCH TERM: a Hookean spring on one rod edge, plus
// the Rayleigh damping block built on the same already-PSD stiffness.

#include "../csrmat/fixed_csr.kernel.cpp"
#include "../utility/rod_damping.kernel.cpp"
#include "../utility/rod_scatter.kernel.cpp"
#include "model/hook.hpp"

// The stretch gradient and Hessian of one rod segment.
//
// `weight` is the force factor the caller composes, which for a rod edge is its
// stiffness times its mass. It is formed by the caller rather than here because
// what multiplies what is an authoring decision about the material, and this
// body is the arithmetic that follows from it.
//
// `gradient` and `hessian` are WRITTEN, not accumulated: hook::make_diff_table
// assigns both, so a caller that has something else in them would lose it. The
// damping call afterward is the accumulating half, and the order is
// load-bearing: the damping operator IS the elastic Hessian just built, so it
// has to run after that Hessian is complete and before the scatter.
[[seam::device_fn]] inline void rod_hook_force_hessian(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &current0,
    const Vec3f &current1, float rest_length, float weight,
    float deform_damping, float dt, Mat3x2f &gradient,
    Mat6x6f &hessian) {
    hook::make_diff_table(x0, x1, rest_length, weight, gradient, hessian);
    rod_add_stiffness_damping(x0, x1, current0, current1, deform_damping,
                                  dt, gradient, hessian);
}

// THE STRETCH TERM ALONE, without the damping half the composition above adds.
//
// IT IS NOT TOTAL, AND THAT IS THE CALLER'S PROBLEM. It divides by the
// segment's CURRENT length and by its rest length and guards neither, so a
// segment whose endpoints coincide, or whose rest length is zero, writes a
// non-finite gradient and Hessian. `weight` cannot recover that: zero times a
// non-finite value is not zero. The caller must therefore evaluate only the
// segments it will scatter, which is what walking the active list does.
//
// It carries the declaration's name because the generated entry calls that
// name, and it is a body of its own rather than an arity change on
// `rod_hook_force_hessian`: that composition already has callers, and the
// damping it folds in is a separate dispatch here.
[[seam::device_fn]] inline void rod_stretch_diff_table(
    const Vec3f &x0, const Vec3f &x1,
    float rest_length, float weight, Mat3x2f &gradient,
    Mat6x6f &hessian) {
    hook::make_diff_table(x0, x1, rest_length, weight, gradient, hessian);
}

// TWO POSITIONS READ THROUGH THE EDGE'S OWN INDEX LIST, with the bound each
// slot is checked against carried in the record. The rest length and the
// weight are per-element gathers, and the gradient and the Hessian are
// non-const gathers, which is the write: this body ASSIGNS both rather than
// accumulating into them.
[[seam::entry(count)]] void rod_stretch_diff_table(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    const float *rest_length,
    const float *weight,
    Mat3x2f *gradient,
    Mat6x6f *hessian,
    unsigned count);

// THE WHOLE ROD STRETCH TERM IN ONE THREAD. One dispatch over the rod prefix
// reads the edge, its prop and its material, gates on the stiffness, builds the
// diff table, adds the damping and scatters both the force and the Hessian,
// with no intermediate array between the stages and so no second pass to keep
// in step with this one.
//
// TWO GATES, AND THE ORDER IS LOAD-BEARING: `prop.fixed` first, then
// `stiffness > 0.0f` on the material the first gate's `param_index` selects.
// Together they decide the element set `builder.rs` registered stencils for, so
// widening either would push blocks the fixed pattern has no slot for and
// `fixed_csr_atomic_push` would refuse them, which is a fatal rather than a
// silent drop.
//
// WHAT THIS BODY DOES NOT CHECK is the rest length. `rod_stretch_diff_table`
// divides by it and guards nothing, and a zero would write a non-finite
// gradient that no later multiply can recover. That check stays on the HOST,
// where the props already live and reading them costs no transfer, and it
// raises a fatal naming the rod rather than producing a number.
[[seam::device_fn]] inline void rod_stretch_embed(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &current0,
    const Vec3f &current1,
    const unsigned *edge_slots,
    const EdgeProp &prop,
    const EdgeParam *edge_param, float dt,
    const unsigned *hess_slots, unsigned has_hess_slots,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned element) {
    if (prop.fixed) {
        return;
    }
    const EdgeParam material = edge_param[prop.param_index];
    if (!(material.stiffness > 0.0f)) {
        return;
    }
    Mat3x2f gradient;
    Mat6x6f hessian;
    rod_hook_force_hessian(x0, x1, current0, current1, prop.length,
                           material.stiffness * prop.mass,
                           material.deform_damping, dt, gradient, hessian);
    // THE SLOTS INTO THREAD SPACE, which the push reads from registers and
    // which Metal requires be thread-space rather than a `const device`
    // lvalue. `shell_strain_embed` takes its three the same way.
    Vec2u edge;
    unsigned vertex_slots[2];
    for (unsigned k = 0; k < 2u; ++k) {
        vertex_slots[k] = edge_slots[2u * element + k];
        edge[k] = vertex_slots[k];
    }
    rod_atomic_embed_force(edge, gradient, force);
    // TWO DEPOSIT PATHS UNDER ONE BRANCH: deposit at the precomputed slot when
    // the scene carries the table, and search the row when it does not.
    // `PPF_SLOT_REPLAY=0` ships an empty table (`builder.rs`), which is the A/B
    // arm, so both paths stay exercised and must agree.
    if (has_hess_slots != 0u) {
        fixed_push_blocks_thread_at(hess_slots, hessian.m, 2u, fixed_value,
                                    element);
    } else {
        fixed_push_blocks_thread(vertex_slots, hessian.m, 2u, fixed_index,
                                 fixed_offset, fixed_value, row_count, refused,
                                 witness);
    }
}

// TWO POSITION ARRAYS THROUGH ONE INDEX LIST, as the strain limiter's fused
// entry takes them: `edge` is read for the bound-checked slots and
// `edge_slots` is the same array in its other role, walked by the body to give
// the scatter and the CSR lookup their row and column.
[[seam::entry(count, element)]] void rod_stretch_embed(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    const unsigned *edge_slots,
    const EdgeProp *prop,
    const EdgeParam *edge_param,
    float dt,
    const unsigned *hess_slots,
    unsigned has_hess_slots,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value,
    unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness,
    unsigned element,
    unsigned count);
