// File: contact_assembly.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ belonging to no backend, rendered into the
// three forms the three compilers read by ppf-cts-compute/seam/kernelgen.py. The two facts
// a compiler cannot infer are C++ attributes, `[[seam::device_fn]]` for the
// execution space and `[[seam::device]]` / `[[seam::thread]]` for the address
// space of a pointer or reference parameter.
//
// THE TWO ADDRESS SPACES HERE ARE NOT INTERCHANGEABLE. The sparsity arrays are
// buffers the whole grid reads, so they are `[[seam::device]]`; the force, the
// Hessian and the extended blocks are one thread's own registers and stack, so
// they are `[[seam::thread]]`. The extended Hessian is `SMatf<3 * N, 3 * N>`,
// 144 floats at the N = 4 the callers instantiate, and it is written through a
// reference rather than returned: a large matrix returned by value is what
// exhausts the small default device stack, so this tree accumulates in place.

#include "../csrmat/fixed_csr.kernel.cpp"

// The slot a contact block belongs in, or 0xFFFFFFFF when the pair is below the
// diagonal. Both CSR matrices store the upper triangle only, so a contribution
// arriving with `row > column` is the transpose of one already accounted for
// and has no slot of its own.
[[seam::device_fn]] inline unsigned
contact_fixed_slot(const unsigned *index,
                       const unsigned *offset,
                       unsigned row_count, unsigned row, unsigned column) {
    if (row > column) {
        return 0xFFFFFFFFu;
    }
    return fixed_csr_find(index, offset, row_count, row, column).slot;
}

// One contact's force and Hessian spread over the N vertices its barycentric
// weights name. The Hessian block for the pair (i, j) is `w_i w_j H`, so the
// extended matrix is the outer product of the weight vector with itself against
// H, which is PSD whenever H is: the weights enter as a congruence and cannot
// introduce a negative mode.
template <unsigned N>
[[seam::device_fn]] inline void extend_contact_force_hessian(
    const SVecf<N> &weight, const Vec3f &force,
    const Mat3x3f &hessian,
    SMatf<3, N> &extended_force,
    SMatf<3 * N, 3 * N> &extended_hessian) {
    for (unsigned i = 0; i < N; ++i) {
        extended_force.col(i) = weight[i] * force;
    }
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            extended_hessian.template block<3, 3>(3 * i, 3 * j) =
                weight[i] * weight[j] * hessian;
        }
    }
}

// The slot lookup's entry point, declared once and rendered for four targets:
// the `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `contact_fixed_slot_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// The CSR pattern arrives as BASE pointers, because the row's own search is
// the body's; the pair being looked up is two element gathers, and the slot it
// resolves to is one element scatter. `row_count` is the matrix's, one value
// for the whole dispatch, so it arrives in the record.
//
// THE EXTENSION BODY BESIDE IT DECLARES NO ENTRY. It is a template over the
// vertex arity whose Hessian is an `SMatf<3 * N, 3 * N>`, and a record field
// may address a plain identifier or one seam name, never a templated
// one: the parameter split does not track template angle brackets, so the
// comma inside would read as a parameter separator. Naming those sizes in the
// shared type vocabulary is what unblocks it.
[[seam::entry(count)]] void contact_fixed_slot(
    const unsigned *index,
    const unsigned *offset, unsigned row_count,
    const unsigned *row,
    const unsigned *column,
    unsigned *slot,
    unsigned count);

// THE HESSIAN EMBED, ON THE DEVICE: each block is folded into the fixed or the
// dynamic matrix by the thread that formed it, so no staged Hessian crosses the
// bus. Embedding on the host instead means downloading the whole staged
// Hessian: a chunk is 16,384 pairs and a pair's extended Hessian is 144 floats,
// so one chunk moves 9.4 MB down and the unpacked blocks 9.4 MB back up. Item
// 41 measured that across a run, 11.35 GB against 51 MB for the device embed.
//
// THE INDEXING IS THE STAGED LAYOUT'S AND IS READ RATHER THAN DERIVED: `n`
// is the pair's arity, the extended matrix is `3n` on a side, the pair's blocks
// begin at `144 * pair`, and block `(ii, jj)` reads
// `base + stride * (3 * jj + c) + 3 * ii + r` into the block's own `3 * c + r`.
// Only the upper triangle is offered, `row > column` skipped, because the
// matrix stores one of each pair and reaches the other through its transpose.
//
// THE ROUTING IS THE POINT AND IT IS GUARANTEE-CLASS. A block the FIXED
// pattern has no slot for is not an error here, unlike on the elastic path: it
// belongs to the dynamic matrix, which has no pattern to be outside of. So the
// verdict `fixed_csr_atomic_push` returns is read, and a false one appends the
// block to the dynamic matrix's own staging. Losing that routing loses a
// Hessian coupling and leaves an indefinite matrix, which `pAp <= 0` reports
// only some of the time.
//
// THE APPEND CLAIMS ITS SLOT, WHICH IS WHY THE ROW IS `Scatter::Claim`. A
// claim numbers the output ascending on the host arm, where one thread walks
// the span, and in arrival order on a GPU. Nothing downstream depends on the
// order:
// the fixed push folds through float atomics and the dynamic push appends under
// its own atomic and compacts afterwards.
//
// AN OVERFLOWING CLAIM IS COUNTED AND THE BLOCK IS DROPPED, and the caller MUST
// compare the claim against the capacity it passed. It cannot assert here: the
// count is the only place the overflow is visible, and a device assert would
// take the run down where the host can report it and abandon one step.
[[seam::entry(pair)]]
[[seam::device_fn]] inline void contact_embed_hessian_blocks(
    const unsigned *active,
    const unsigned *arity,
    const unsigned *index,
    const float *hessian,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *claim,
    unsigned *stage_row,
    unsigned *stage_column,
    float *stage_block, unsigned capacity, unsigned pair) {
    if (active[pair] == 0u) {
        return;
    }
    const unsigned n = arity[pair];
    const unsigned stride = 3u * n;
    const unsigned base = 144u * pair;
    for (unsigned ii = 0u; ii < n; ++ii) {
        for (unsigned jj = 0u; jj < n; ++jj) {
            const unsigned row = index[4u * pair + ii];
            const unsigned column = index[4u * pair + jj];
            if (row > column) {
                continue;
            }
            Mat3x3f block;
            for (unsigned c = 0u; c < 3u; ++c) {
                for (unsigned r = 0u; r < 3u; ++r) {
                    block.m[3u * c + r] =
                        hessian[base + stride * (3u * jj + c) + 3u * ii + r];
                }
            }
            if (fixed_csr_atomic_push(fixed_index, fixed_offset, fixed_value,
                                      row_count, row, column, block)) {
                continue;
            }
            const unsigned slot = compute::atomic_add(claim, 1u);
            if (slot >= capacity) {
                continue;
            }
            stage_row[slot] = row;
            stage_column[slot] = column;
            for (unsigned element = 0u; element < 9u; ++element) {
                stage_block[9u * slot + element] = block.m[element];
            }
        }
    }
}

// THE FORCE EMBED, ON THE DEVICE: each pair's terms are folded into the
// per-vertex force by the thread that formed them, so the staged force never
// leaves the device. Embedding on the host instead means COMPACTING: download
// the staged force, gather the pairs of one arity into a dense run, upload that
// run and dispatch a scatter over it, three times, once per arity.
//
// NO COMPACTION IS NEEDED AND THAT IS THE WHOLE SIMPLIFICATION. A pass over the
// whole span that returns for an inactive pair reaches exactly the pairs those
// compacted runs would hold. The arity is not a dispatch parameter either: it is
// `arity[pair]`, read in the thread, so one kernel covers what three did.
//
// THE FOLD IS A FLOAT ATOMIC into the per-vertex force, so the row is NOT
// `Scatter::Disjoint`: two contacts sharing a vertex write the same three
// slots, and the host seam spells a float atomic as a plain read, add and write
// back. `check-atomic-scatter.py` gates that pairing.
[[seam::entry(pair)]]
[[seam::device_fn]] inline void contact_embed_force_terms(
    const unsigned *active,
    const unsigned *arity,
    const unsigned *index,
    const float *gradient,
    compute::atomic_float_t *force, unsigned pair) {
    if (active[pair] == 0u) {
        return;
    }
    const unsigned n = arity[pair];
    for (unsigned i = 0u; i < n; ++i) {
        // `vert`, not `vertex`: MSL reserves that as a shader-stage qualifier.
        const unsigned vert = index[4u * pair + i];
        for (unsigned d = 0u; d < 3u; ++d) {
            const float term = gradient[12u * pair + 3u * i + d];
            if (term != 0.0f) {
                compute::atomic_add(force + 3u * vert + d, term);
            }
        }
    }
}
