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
// set the three-node face scatter (face_scatter.kernel.cpp) and the four-node
// hinge scatter
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

