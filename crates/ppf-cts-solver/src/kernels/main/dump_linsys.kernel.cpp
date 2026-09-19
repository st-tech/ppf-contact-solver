// File: dump_linsys.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the forms the three compilers read. The one fact a compiler
// cannot infer is written as a C++ attribute: `[[seam::device]]` is the address
// space of a pointer parameter, which MSL requires on every one of them.
//
// No include of its own. `compute::atomic_uint_t` and `compute::atomic_add`
// come from whichever backend prologue is in scope: cpp/seam under nvcc and on
// the host, the prologue in metal/shader_compiler.mm under MSL.
//
// THE EXECUTION SPACE IS DEVICE-ONLY, for the reason the two transpose bodies
// in csrmat/dynamic_csr.kernel.cpp give: the body reaches an atomic, which is
// `__device__` alone under nvcc, so marking it for both spaces makes nvcc
// reject the atomic rather than the caller.
//
// WHAT THIS BODY IS. One thread owns one row of a block-CSR matrix and appends
// that row's stored blocks to a shared coordinate (COO) buffer through an
// atomic cursor. It is the whole of the offline linear-system dumper's device
// work, and both of the dumper's passes are this one body: the dynamic
// contact matrix and the fixed matrix differ only in which arrays they hand
// it (main/dump_linsys.hpp).
//
// THE ROW'S EXTENT ARRIVES AS TWO NUMBERS RATHER THAN AS A PATTERN THIS BODY
// WALKS, and that is what lets one body serve both matrices. A fixed CSR row
// is `[offset[i], offset[i + 1])` of one flat pattern, while a dynamic row is
// a slab of its own inside two flat buffers whose width the row itself
// records. Neither shape is a special case of the other, so the caller states
// the half-open slot range and this body reads it.
//
// ONLY THE UPPER TRIANGLE IS STORED by either matrix (the lower triangle is
// the implied transpose), so a row carries diagonal blocks and strictly-upper
// blocks and nothing else. The diagonal is summed into a per-row block that
// the caller adds the separate diagonal term to; the strictly-upper blocks go
// to the COO. A block reached at `column[slot] < row` cannot occur and is not
// emitted, which is the same reading the two launchers this body replaced had.
//
// THE CAPACITY IS CHECKED AND THE CURSOR IS NOT CLAMPED, deliberately: the
// cursor keeps counting past `capacity` so the caller sees the demand rather
// than the truncation, and compares it against the capacity itself. Metal
// never faults on an out-of-bounds write and drops it silently, so the check
// is what makes an undersized buffer visible there at all.

[[seam::device_fn]] inline void dump_linsys_row_to_coo(
    unsigned row_begin, unsigned row_end,
    const unsigned *column,
    const float *block,
    compute::atomic_uint_t *cursor,
    unsigned *out_row, unsigned *out_column,
    float *out_block, float *diagonal,
    unsigned capacity, unsigned row) {
    for (unsigned slot = row_begin; slot < row_end; ++slot) {
        const unsigned other = column[slot];
        if (other == row) {
            for (unsigned element = 0; element < 9u; ++element) {
                diagonal[9u * row + element] += block[9u * slot + element];
            }
        } else if (other > row) {
            const unsigned position = compute::atomic_add(cursor, 1u);
            if (position < capacity) {
                out_row[position] = row;
                out_column[position] = other;
                for (unsigned element = 0; element < 9u; ++element) {
                    out_block[9u * position + element] =
                        block[9u * slot + element];
                }
            }
        }
    }
}

// The row pass as an entry point, declared once and rendered for four targets:
// the `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the range shim a host C++ compiler compiles, and the Rust
// `#[repr(C)]` twin. A hand-written entry point is a mirror pair per backend
// with nothing linking the halves; this declaration is the one place the
// argument record exists.
//
// `row_begin` and `row_end` are GATHERS: they are this row's own slot range
// and address nothing, so there is no slot to bound. Every other buffer stays
// a BASE pointer, because none of them is reached at the thread index: the
// pattern and the blocks are reached at a slot the row range names, and the
// three COO arrays at a position the atomic cursor hands back. That position
// is data rather than the thread index, which is why the capacity is a
// declared field the body tests and not a guard this form could supply.
//
// A block is nine floats rather than a `[[seam::pod(36)]]` matrix, and the
// blocks in and out are spelled the same way, so the copy is nine float
// stores in the order the file format writes them.
[[seam::entry(count, row)]] void dump_linsys_row_to_coo(
    const unsigned *row_begin,
    const unsigned *row_end,
    const unsigned *column,
    const float *block,
    compute::atomic_uint_t *cursor,
    unsigned *out_row, unsigned *out_column,
    float *out_block, float *diagonal,
    unsigned capacity, unsigned count,
    unsigned row);
