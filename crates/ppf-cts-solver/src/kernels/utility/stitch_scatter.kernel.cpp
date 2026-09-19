// File: stitch_scatter.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ that belongs to no backend, with no
// preprocessor conditional, no macro of its own, and no spelling that only one
// of the three compilers accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// form each compiler reads. `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` and `[[seam::device]]` are the address spaces MSL requires
// on every reference and pointer type. The seam functions and types come
// from the backend prologue: cpp/seam under nvcc and on the host, and
// `kMslMacroSeam` in metal/shader_compiler.mm under MSL.

// The six-node force scatter. `index` is the stitch's slot-to-vertex map in the
// order `gradient`'s columns were built in, which is the authored order of
// Stitch::index: slots 0..2 the source barycentric, 3..5 the target. The CUDA
// site spells the same two loops as utility::atomic_embed_force<6>.
//
// A DEGENERATE STITCH NAMES ONE VERTEX THREE TIMES, and nothing here special
// cases it: the three folds land on the same three floats through the same
// atomic add, which is the sum the barycentric form is asking for when all the
// weight sits on one slot. That is also why the adds cannot be replaced by a
// gather.
//
// A zero component is added rather than skipped, matching the three- and
// four-node scatters beside this one. The accumulator is cleared to +0.0 before
// the pass and every contribution folds onto it, and IEEE round-to-nearest
// gives +0.0 for (+0.0) + (-0.0) and for x + (-x), so the buffer can never come
// to hold -0.0, which is the only float a +0.0 addition would alter.
// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `stitch_atomic_embed_force_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// THE THREAD INDEX REACHES THE BODY THROUGH THE GATHERS AND NOWHERE ELSE, as
// in the arity-1 form (utility/vertex_scatter.kernel.cpp). This body takes the
// six weighted slots the contribution lands on and the contribution itself,
// so the declaration carries no `[[seam::index]]` and the generator
// names the index it guards on. Both lists are compacted, one entry per
// contributing element, so the two gathers read the same element of each.
//
// THE SCATTER IS THE BODY'S, AND IT STAYS SERIAL. `force` arrives as a base
// pointer because the destination is chosen by the stitch's own six
// vertices, not by the thread index, and a degenerate stitch repeats an
// index, so two slots of one stitch can land on one destination.
// `compute::atomic_add` is a plain read, add and write back on the host seam,
// which makes a parallel pass over these elements a data race rather than a
// different fold order. Nothing in this declaration says otherwise: the
// entry point covers whatever range it is handed, and the kernel table's
// `Scatter::Atomic` is what keeps that range one ascending pass.
//
// `compute::atomic_float_t` is the seam's own name for the destination's
// element type, `float` under nvcc and on the host and `atomic_float` under
// MSL: a `float *` here would compile on two backends and fail on the third.
[[seam::entry]]
[[seam::device_fn]] inline void stitch_atomic_embed_force(
    const Vec6u &index,
    const Mat3x6f &gradient,
    compute::atomic_float_t *force) {
    for (unsigned slot = 0; slot < 6; ++slot) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            compute::atomic_add(force + 3 * index[slot] + dimension,
                            gradient(dimension, slot));
        }
    }
}

// The six-node Hessian scatter into precomputed fixed-CSR value slots. `slots`
// points at this stitch's 36-entry table (row major, ii * 6 + jj), built by
// builder.rs in the same order Stitch::index is read above, so slots[ii * 6 +
// jj] addresses the block (index[ii], index[jj]).
//
// A 0xFFFFFFFF entry marks a block the CSR does not store, which is every
// lower-triangle block, and is skipped. A degenerate stitch's repeated indices
// resolve to the SAME slot for several (ii, jj) pairs, and those blocks fold
// onto one CSR entry through the atomic, exactly as the repeated push() calls
// they replace did. The per-component zero test and the column-major element
// order are the ones FixedCSRMat::push_at uses, so the float fold order is
// unchanged by going through the slot table.
[[seam::device_fn]] inline void stitch_atomic_embed_hessian_slots(
    const unsigned *slots,
    const SMatf<18, 18> &hessian,
    compute::atomic_float_t *values) {
    for (unsigned row_slot = 0; row_slot < 6; ++row_slot) {
        for (unsigned column_slot = 0; column_slot < 6; ++column_slot) {
            const unsigned slot = slots[6 * row_slot + column_slot];
            if (slot != 0xFFFFFFFFu) {
                for (unsigned column_dimension = 0; column_dimension < 3;
                     ++column_dimension) {
                    for (unsigned row_dimension = 0; row_dimension < 3;
                         ++row_dimension) {
                        const float value =
                            hessian(3 * row_slot + row_dimension,
                                    3 * column_slot + column_dimension);
                        if (value != 0.0f) {
                            const unsigned element =
                                row_dimension + 3 * column_dimension;
                            compute::atomic_add(
                                values + 9 * slot + element, value);
                        }
                    }
                }
            }
        }
    }
}
