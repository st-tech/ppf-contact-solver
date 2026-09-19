// File: hinge_scatter.kernel.cpp
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

// The four-node force scatter. `hinge` is the vertex index quadruple in the
// SAME order `gradient`'s columns were built in, which for the shell hinge is
// the (2,1,0,3) order the dihedral math wants and for a tet is the mesh order
// of `data.mesh.mesh.tet`. Both four-node elements come through here on the
// Metal backend; the CUDA sites spell the same two loops as
// utility::atomic_embed_force<4> and utility::atomic_embed_hessian_at<4>.
//
// A zero component is added rather than skipped, matching the three-node face
// scatter beside this one. That is not an oversight and it is not a value
// change: the accumulator is cleared to +0.0 before the pass and every
// contribution folds onto it, and IEEE round-to-nearest gives +0.0 for
// (+0.0) + (-0.0) and for x + (-x), so the buffer can never come to hold -0.0,
// which is the only float a +0.0 addition would alter. Adding zero is therefore
// a genuine no-op at every interleaving of the concurrent folds.
// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `hinge_atomic_embed_force_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// THE THREAD INDEX REACHES THE BODY THROUGH THE GATHERS AND NOWHERE ELSE, as
// in the arity-1 form (utility/vertex_scatter.kernel.cpp). This body takes the
// four vertices the contribution lands on and the contribution itself, so
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
hinge_atomic_embed_force(const Vec4u &hinge,
                             const Mat3x4f &gradient,
                             compute::atomic_float_t *force) {
    for (unsigned vertex_index = 0; vertex_index < 4; ++vertex_index) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            compute::atomic_add(force + 3 * hinge[vertex_index] + dimension,
                            gradient(dimension, vertex_index));
        }
    }
}

// The same scatter over an ACTIVE SUBSET, reached through `active` rather than
// through a host-compacted run.
//
// THE HESSIAN SIDE ALREADY TAKES THIS SHAPE. `fixed_push_element_blocks` is
// dispatched over the active list with the element read from it, and this is
// the force half of the same pass, which had stayed on the host: the driver
// downloaded the whole gradient array, compacted the active run into staging,
// and uploaded it again. The order is unchanged, because `active` is ascending
// and the slot is the thread index, so the fold reaches the four-vertex embeds
// in exactly the order the compacted run did.
//
// THE THREAD-SPACE COPIES ARE THIS BODY'S OWN WORK. Its element is DATA rather
// than the thread index, so no `[[seam::gather]]` brought them in and a
// generated entry made no copy; reached through a `[[seam::device]]` base
// pointer these are device-space lvalues, which MSL will not bind to the
// thread-space reference the scatter takes.
[[seam::device_fn]] inline void
hinge_active_embed_force(const unsigned *active,
                             const unsigned *hinge,
                             const float *gradient,
                             compute::atomic_float_t *force,
                             unsigned slot) {
    const unsigned element = active[slot];
    Vec4u quad;
    for (unsigned i = 0; i < 4; ++i) {
        quad[i] = hinge[4u * element + i];
    }
    Mat3x4f local;
    for (unsigned k = 0; k < 12u; ++k) {
        local.m[k] = gradient[12u * element + k];
    }
    hinge_atomic_embed_force(quad, local, force);
}

// THE SAME EMBED WITH THE ELEMENT AS THE THREAD INDEX AND THE RUN DECIDED HERE,
// which is `energy.cu`'s shape: it dispatches over the full element count and
// gates on the element's own material rather than over a list the host
// compacted out of a downloaded array.
//
// THE ORDER IS THE COMPACTED RUN'S ORDER. The gate skips exactly the elements
// the run omitted and the surviving ones are visited in the same ascending
// sequence, so the `Scatter::Atomic` row's serial ascending pass on the host arm
// reaches the four-vertex embeds in the order it did before.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
hinge_live_embed_force(const float *live,
                       const unsigned *hinge,
                       const float *gradient,
                       compute::atomic_float_t *force,
                       unsigned element) {
    if (live[element] <= 0.0f) {
        return;
    }
    Vec4u quad;
    for (unsigned i = 0; i < 4; ++i) {
        quad[i] = hinge[4u * element + i];
    }
    Mat3x4f local;
    for (unsigned k = 0; k < 12u; ++k) {
        local.m[k] = gradient[12u * element + k];
    }
    hinge_atomic_embed_force(quad, local, force);
}

[[seam::entry(count, slot)]] void hinge_active_embed_force(
    const unsigned *active,
    const unsigned *hinge,
    const float *gradient,
    compute::atomic_float_t *force,
    unsigned slot,
    unsigned count);

// The four-node Hessian scatter into precomputed fixed-CSR value slots.
// `slots` points at this element's 16-entry table (row major, ii * 4 + jj),
// built in the same order the quadruple is passed in, so slot[ii * 4 + jj]
// addresses the block (hinge[ii], hinge[jj]).
//
// A 0xFFFFFFFF entry marks a block the CSR does not store, which is every
// lower-triangle block, and is skipped. The per-component zero test and the
// column-major element order are the ones the CSR's own slot push uses, so the
// float fold order is unchanged by going through the slot table.
[[seam::device_fn]] inline void hinge_atomic_embed_hessian_slots(
    const unsigned *slots,
    const Mat12x12f &hessian,
    compute::atomic_float_t *values) {
    for (unsigned row_vertex = 0; row_vertex < 4; ++row_vertex) {
        for (unsigned column_vertex = 0; column_vertex < 4; ++column_vertex) {
            const unsigned slot = slots[4 * row_vertex + column_vertex];
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
// compiler compiles, the `hinge_atomic_embed_hessian_slots_entry` shim
// a host C++ compiler compiles, and the Rust `#[repr(C)]` twin the driver
// fills.
//
// A hinge contributes sixteen 3x3 blocks and so owns sixteen consecutive slot
// ids, which is the stride; `values` is the CSR value array and stays a BASE
// pointer, because the destination is chosen by the slot table rather than by
// the thread index. Like every scatter it stays SERIAL: `compute::atomic_add`
// is a plain read, add and write back on the host seam, so two hinges sharing a
// block are a data race and not a fold-order difference.
//
// ONLY THE HESSIAN. The force scatter beside it writes to per-vertex
// destinations named by the hinge's own index vector, and its launcher has a
// live driver call site that does not go through the `Device` seam, so
// converting it is a call-site change rather than a declaration.
[[seam::entry(count)]] void hinge_atomic_embed_hessian_slots(
    [[seam::stride(16)]] const unsigned *slots,
    const Mat12x12f *hessian,
    compute::atomic_float_t *values,
    unsigned count);
