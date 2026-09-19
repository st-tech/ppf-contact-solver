// File: rod_scatter.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++, no preprocessor conditional and no macro
// of its own. ppf-cts-compute/seam/kernelgen.py renders it into the three forms the three
// compilers read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space, and `[[seam::thread]]` and
// `[[seam::device]]` are the address spaces MSL requires on every reference and
// pointer. Everything else is an ordinary call into the `ppf` namespace each
// backend prologue defines.

// The two-node scatter, for a rod segment. This is the arity-2 member of the
// set the three-node face scatter (face_scatter.kernel.cpp,
// face_hessian_scatter.kernel.cpp) and the four-node hinge scatter
// (hinge_scatter.kernel.cpp) already belong to: one body per node count, each
// written once and called by both backends. Two terms come through here, the
// rod stretch energy and the rod strain-limit barrier, and builder.rs emits one
// `edge_hess_slots` table for both because both address the CSR through
// `data.mesh.mesh.edge[i]` in the same order.
//
// `edge` is the vertex index pair in the SAME order `gradient`'s columns were
// built in, which for a rod segment is the mesh order.
//
// A zero component is added rather than skipped, matching the face and hinge
// scatters. That is not a value change: the accumulator is cleared to +0.0
// before the pass and every contribution folds onto it, and IEEE
// round-to-nearest gives +0.0 for (+0.0) + (-0.0) and for x + (-x), so the
// buffer can never come to hold -0.0, which is the only float a +0.0 addition
// would alter.
// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `rod_atomic_embed_force_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// THE THREAD INDEX REACHES THE BODY THROUGH THE GATHERS AND NOWHERE ELSE, as
// in the arity-1 form (utility/vertex_scatter.kernel.cpp). This body takes the
// two vertices the contribution lands on and the contribution itself, so
// the declaration carries no `[[seam::index]]` and the generator
// names the index it guards on. Both lists are compacted, one entry per
// contributing element, so the two gathers read the same element of each.
//
// THE SCATTER IS THE BODY'S, AND IT STAYS SERIAL. `force` arrives as a base
// pointer because the destination is chosen by the element's own vertices, not
// by the thread index, so two elements sharing a vertex land on one
// destination. `compute::atomic_add` is a plain read, add and write back on the
// host seam, which makes a parallel pass over these elements a data race rather
// than a different fold order. Nothing in this declaration says otherwise: the
// entry point covers whatever range it is handed, and the kernel table's
// `Scatter::Atomic` is what keeps that range one ascending pass.
//
// `compute::atomic_float_t` is the seam's own name for the destination's
// element type, `float` under nvcc and on the host and `atomic_float` under
// MSL: a `float *` here would compile on two backends and fail on the third.
[[seam::entry]]
[[seam::device_fn]] inline void
rod_atomic_embed_force(const Vec2u &edge,
                           const Mat3x2f &gradient,
                           compute::atomic_float_t *force) {
    for (unsigned vertex_index = 0; vertex_index < 2; ++vertex_index) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            compute::atomic_add(force + 3 * edge[vertex_index] + dimension,
                            gradient(dimension, vertex_index));
        }
    }
}

// The same scatter over an ACTIVE SUBSET, reached through `active` rather than
// through a host-compacted run. `hinge_active_embed_force` states the argument
// in full: the Hessian half of these passes already takes this shape, the order
// is unchanged because `active` is ascending and the slot is the thread index,
// and the thread-space copies are this body's own work because its element is
// DATA rather than the thread index.
[[seam::device_fn]] inline void
rod_active_embed_force(const unsigned *active,
                             const unsigned *edge,
                             const float *gradient,
                             compute::atomic_float_t *force,
                             unsigned slot) {
    const unsigned element = active[slot];
    Vec2u nodes;
    for (unsigned i = 0; i < 2u; ++i) {
        nodes[i] = edge[2u * element + i];
    }
    Mat3x2f local;
    for (unsigned k = 0; k < 6u; ++k) {
        local.m[k] = gradient[6u * element + k];
    }
    rod_atomic_embed_force(nodes, local, force);
}

// THE SAME SCATTER WITH THE ELEMENT AS THE THREAD INDEX AND THE RUN DECIDED
// HERE, which is `energy.cu`'s shape and `hinge_live_embed_force`'s: dispatch
// over the full element count and gate on a verdict the device already holds,
// rather than over a list the host compacted out of a downloaded array.
//
// THE ORDER IS THE COMPACTED RUN'S ORDER. The gate skips exactly the rods the
// run omitted and the survivors are visited in the same ascending sequence, so
// the `Scatter::Atomic` row's serial ascending pass on the host arm reaches the
// two-node embeds in the order it did before.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
rod_live_embed_force(const unsigned *live,
                     const unsigned *edge,
                     const float *gradient,
                     compute::atomic_float_t *force,
                     unsigned element) {
    if (live[element] == 0u) {
        return;
    }
    Vec2u nodes;
    for (unsigned i = 0; i < 2u; ++i) {
        nodes[i] = edge[2u * element + i];
    }
    Mat3x2f local;
    for (unsigned k = 0; k < 6u; ++k) {
        local.m[k] = gradient[6u * element + k];
    }
    rod_atomic_embed_force(nodes, local, force);
}

[[seam::entry(count, slot)]] void rod_active_embed_force(
    const unsigned *active,
    const unsigned *edge,
    const float *gradient,
    compute::atomic_float_t *force,
    unsigned slot,
    unsigned count);

// The two-node scatter where the GRADIENT is already packed by slot and only
// the EDGE needs the indirection.
//
// THE ROD STRETCH DIFF TABLE WRITES ONE COMPACTED ASCENDING RUN over the active
// rods, so this rod's six floats sit at its position in that run rather than at
// its own index, while its two vertices still come from the mesh edge array at
// the rod's own index. That asymmetry is why this cannot be
// `rod_active_embed_force`, which reads both by element.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void
rod_packed_embed_force(const unsigned *active,
                           const unsigned *edge,
                           const float *gradient,
                           compute::atomic_float_t *force,
                           unsigned slot) {
    const unsigned element = active[slot];
    Vec2u nodes;
    for (unsigned i = 0; i < 2u; ++i) {
        nodes[i] = edge[2u * element + i];
    }
    Mat3x2f local;
    for (unsigned k = 0; k < 6u; ++k) {
        // BY SLOT, not by element: the packed run's own position.
        local.m[k] = gradient[6u * slot + k];
    }
    rod_atomic_embed_force(nodes, local, force);
}

// The two-node Hessian scatter into precomputed fixed-CSR value slots. `slots`
// points at this segment's 4-entry table (row major, ii * 2 + jj), so
// slot[ii * 2 + jj] addresses the block (edge[ii], edge[jj]).
//
// A 0xFFFFFFFF entry marks a block the CSR does not store, which is the
// lower-triangle block (1, 0), and is skipped. THE SENTINEL IS THE ONLY THING
// THAT MAY BE SKIPPED: FixedCSRMat::push returns false and drops a block that
// is out of pattern without telling anyone, which leaves an indefinite matrix,
// so a caller whose slot table could address a block builder.rs never
// registered must refuse rather than scatter. The per-component zero test and
// the column-major element order are the ones FixedCSRMat::push_at uses, so the
// float fold order is unchanged by going through the slot table.
[[seam::device_fn]] inline void rod_atomic_embed_hessian_slots(
    const unsigned *slots,
    const Mat6x6f &hessian,
    compute::atomic_float_t *values) {
    for (unsigned row_vertex = 0; row_vertex < 2; ++row_vertex) {
        for (unsigned column_vertex = 0; column_vertex < 2; ++column_vertex) {
            const unsigned slot = slots[2 * row_vertex + column_vertex];
            if (slot != 0xFFFFFFFFu) {
                for (unsigned column_dimension = 0; column_dimension < 3;
                     ++column_dimension) {
                    for (unsigned row_dimension = 0; row_dimension < 3;
                         ++row_dimension) {
                        const float value =
                            hessian(3 * row_vertex + row_dimension,
                                    3 * column_vertex + column_dimension);
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

// The Hessian scatter's entry point, declared once and rendered for four
// targets: the `__global__` nvcc compiles, the `kernel void` the Metal shader
// compiler compiles, the `rod_atomic_embed_hessian_slots_entry` shim a
// host C++ compiler compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// An edge contributes four 3x3 blocks and so owns four consecutive slot ids,
// which is the stride; `values` is the CSR value array and stays a BASE
// pointer, because the destination is chosen by the slot table rather than by
// the thread index. Like every scatter it stays SERIAL, for the reason the
// face and hinge forms state.
//
// ONLY THE HESSIAN, for the reason the hinge form states: the force scatter
// beside it has a live driver call site that does not go through the `Device`
// seam.
[[seam::entry(count)]] void rod_atomic_embed_hessian_slots(
    [[seam::stride(4)]] const unsigned *slots,
    const Mat6x6f *hessian,
    compute::atomic_float_t *values,
    unsigned count);
