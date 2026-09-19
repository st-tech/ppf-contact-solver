// File: face_scatter.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. `[[seam::device_fn]]` is the
// execution space and `[[seam::thread]]` / `[[seam::device]]` are the address
// spaces MSL requires on every reference and pointer type; CUDA and the host
// have one address space and are handed the same declarations with it removed.

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `face_atomic_embed_force_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// THE THREAD INDEX REACHES THE BODY THROUGH THE GATHERS AND NOWHERE ELSE, as
// in the arity-1 form (utility/vertex_scatter.kernel.cpp). This body takes the
// three vertices the contribution lands on and the contribution itself, so the
// declaration carries no `[[seam::index]]` and the generator names the index it
// guards on. Both lists are compacted, one entry per contributing face, so the
// two gathers read the same element of each.
//
// THE SCATTER IS THE BODY'S, AND IT STAYS SERIAL. `force` arrives as a base
// pointer because the destination is chosen by the face's own vertices, not by
// the thread index, so two faces sharing a vertex land on one destination.
// `compute::atomic_add` is a plain read, add and write back on the host seam,
// which makes a parallel pass over these elements a data race rather than a
// different fold order. Nothing in this declaration says otherwise: the entry
// point covers whatever range it is handed, and the kernel table's
// `Scatter::Atomic` is what keeps that range one ascending pass.
//
// `compute::atomic_float_t` is the seam's own name for the destination's
// element type, `float` under nvcc and on the host and `atomic_float` under
// MSL: a `float *` here would compile on two backends and fail on the third.
[[seam::entry]]
[[seam::device_fn]] inline void face_atomic_embed_force(
    const Vec3u &face,
    const Mat3x3f &gradient,
    compute::atomic_float_t *force) {
    for (unsigned vertex_index = 0; vertex_index < 3; ++vertex_index) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            compute::atomic_add(force + 3 * face[vertex_index] + dimension,
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
face_active_embed_force(const unsigned *active,
                             const unsigned *face,
                             const float *gradient,
                             compute::atomic_float_t *force,
                             unsigned slot) {
    const unsigned element = active[slot];
    Vec3u nodes;
    for (unsigned i = 0; i < 3u; ++i) {
        nodes[i] = face[3u * element + i];
    }
    Mat3x3f local;
    for (unsigned k = 0; k < 9u; ++k) {
        local.m[k] = gradient[9u * element + k];
    }
    face_atomic_embed_force(nodes, local, force);
}

// THE SAME EMBED WITH THE ELEMENT AS THE THREAD INDEX AND THE RUN DECIDED HERE,
// which is `strainlimiting.cu:57`'s shape: one dispatch over the shell face
// count, gated on the face's own records and on the SVD's verdict, with no list
// compacted anywhere.
//
// THE ORDER IS THE COMPACTED RUN'S ORDER. The gate skips exactly the faces the
// run omitted and the survivors are visited in the same ascending sequence, so
// the `Scatter::Atomic` row's serial ascending pass on the host arm reaches the
// three-vertex embeds in the order it did before.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
face_live_embed_force(const float *live,
                      const unsigned *face,
                      const float *gradient,
                      compute::atomic_float_t *force,
                      unsigned element) {
    if (live[element] <= 0.0f) {
        return;
    }
    Vec3u nodes;
    for (unsigned i = 0; i < 3u; ++i) {
        nodes[i] = face[3u * element + i];
    }
    Mat3x3f local;
    for (unsigned k = 0; k < 9u; ++k) {
        local.m[k] = gradient[9u * element + k];
    }
    face_atomic_embed_force(nodes, local, force);
}

[[seam::entry(count, slot)]] void face_active_embed_force(
    const unsigned *active,
    const unsigned *face,
    const float *gradient,
    compute::atomic_float_t *force,
    unsigned slot,
    unsigned count);
