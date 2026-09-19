// File: face_hessian_scatter.kernel.cpp
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
// `[[seam::thread]]` / `[[seam::device]]` are the address spaces of the
// reference and pointer parameters. MSL requires the second on every reference
// and pointer type; CUDA and the host have one address space and are handed the
// same declarations with it removed.
//
// No include of its own. `Mat9x9f` arrives from the includer rather than from
// here, which is data.hpp under nvcc and on the host and the shader prologue's
// aliases under MSL.
//
// The atomic add is the one name here whose spelling genuinely differs per
// backend rather than expanding to nothing: an intrinsic on the GPU, a read,
// add and write back on the host, where one thread runs the body and the
// read-modify-write needs no hardware support. `compute::atomic_float_t` and
// `compute::atomic_add` are the backend prologue's names for the two halves of
// it.

[[seam::device_fn]] inline void face_atomic_embed_hessian_slots(
    const unsigned *slots,
    const Mat9x9f &hessian,
    compute::atomic_float_t *values) {
    for (unsigned row_vertex = 0; row_vertex < 3; ++row_vertex) {
        for (unsigned column_vertex = 0; column_vertex < 3; ++column_vertex) {
            const unsigned slot = slots[3 * row_vertex + column_vertex];
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

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `face_atomic_embed_hessian_slots_entry` shim a host C++
// compiler compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// THE SLOT TABLE IS A STRIDE, NOT A GATHER. A face contributes nine 3x3 blocks
// and so owns nine consecutive slot ids, which is why the body is handed
// `slots + 9 * index` rather than one element: the run's own indexing is the
// body's. `values` is the CSR value array and stays a BASE pointer, because the
// destination is chosen by the slot table and not by the thread index.
//
// IT IS A SCATTER AND IT STAYS SERIAL. Two faces can share a block, so
// `compute::atomic_add`, a plain read, add and write back on the host seam,
// makes a parallel pass over these elements a data race rather than a different
// fold order. A generated entry point covers whatever range it is handed and
// cannot say otherwise; the kernel table's `Scatter::Atomic` is what keeps that
// range one ascending pass.
//
// `compute::atomic_float_t` is the seam's own name for the destination's
// element type, `float` under nvcc and on the host and `atomic_float` under
// MSL: a `float *` here would compile on two backends and fail on the third.
[[seam::entry(count)]] void face_atomic_embed_hessian_slots(
    [[seam::stride(9)]] const unsigned *slots,
    const Mat9x9f *hessian,
    compute::atomic_float_t *values,
    unsigned count);
