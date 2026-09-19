// File: fixed_csr.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The two facts a backend cannot infer are written as
// C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::device]]` and `[[seam::thread]]` are the address spaces of a pointer
// or a reference parameter. MSL requires an address space on every pointer and
// every reference; CUDA and the host have one address space and are handed the
// same declarations with it removed.
//
// No include of its own. `Mat3x3f` comes from whatever declares it for the
// backend that is compiling, which is data.hpp under nvcc and on the host and
// the shader prologue's aliases under MSL. `compute::atomic_float_t` and
// `compute::atomic_add` come from that backend's prologue in the same way:
// ppf-cts-compute/cuda/seam_cuda.cuh under nvcc, kernels/seam/seam_host.h on
// the host, and the prologue in ppf-cts-compute/metal/shader_compiler.mm under
// MSL.
//
// THE ADDRESS SPACES HERE ARE NOT INTERCHANGEABLE, and MSL is the only backend
// that can tell. The four buffers are `[[seam::device]]` because a matrix lives
// in the arena every threadgroup reads, while the 3x3 block a caller pushes is
// `[[seam::thread]]` because it is that thread's own value. On MSL the address
// space is part of the pointer type and selects the `compute::atomic_add`
// overload, and the float accumulate has a `device` overload only, so a wrong
// address space on `value` is a compile error naming the missing overload. CUDA
// and the host drop the attribute and cannot raise that error, which is why it
// is written here rather than left to the one backend that checks it.

struct FixedCsrLookup {
    unsigned slot;
    bool transpose;
};

[[seam::device_fn]] inline FixedCsrLookup
fixed_csr_find(const unsigned *index,
                   const unsigned *offset, unsigned row_count,
                   unsigned i, unsigned j) {
    FixedCsrLookup result{0xFFFFFFFFu, false};
    if (i > j) {
        const unsigned temporary = i;
        i = j;
        j = temporary;
        result.transpose = true;
    }
    if (i >= row_count) {
        return result;
    }
    for (unsigned slot = offset[i]; slot < offset[i + 1]; ++slot) {
        if (index[slot] == j) {
            result.slot = slot;
            break;
        }
        if (index[slot] > j) {
            break;
        }
    }
    return result;
}

[[seam::device_fn]] inline Mat3x3f
fixed_csr_read(const unsigned *index,
                   const unsigned *offset,
                   const float *value, unsigned row_count,
                   unsigned i, unsigned j) {
    const FixedCsrLookup lookup =
        fixed_csr_find(index, offset, row_count, i, j);
    Mat3x3f result = Mat3x3f::Zero();
    if (lookup.slot == 0xFFFFFFFFu) {
        return result;
    }
    for (unsigned element = 0; element < 9; ++element) {
        result.m[element] = value[9 * lookup.slot + element];
    }
    return lookup.transpose ? result.transpose() : result;
}

[[seam::device_fn]] inline void fixed_csr_atomic_push_slot(
    compute::atomic_float_t *value, unsigned slot,
    const Mat3x3f &block) {
    for (unsigned element = 0; element < 9; ++element) {
        const float contribution = block.m[element];
        if (contribution != 0.0f) {
            compute::atomic_add(value + 9 * slot + element, contribution);
        }
    }
}

[[seam::device_fn]] inline bool
fixed_csr_atomic_push(const unsigned *index,
                          const unsigned *offset,
                          compute::atomic_float_t *value,
                          unsigned row_count, unsigned i, unsigned j,
                          const Mat3x3f &block) {
    if (i > j) {
        return false;
    }
    const FixedCsrLookup lookup =
        fixed_csr_find(index, offset, row_count, i, j);
    if (lookup.slot == 0xFFFFFFFFu) {
        return false;
    }
    fixed_csr_atomic_push_slot(value, lookup.slot, block);
    return true;
}

// EVERY ELEMENT'S HESSIAN BLOCKS, OFFERED TO THE FIXED PATTERN IN ITS OWN
// THREAD. The host spelling of it would download the element Hessians, compact
// them into `push_row`, `push_column` and `push_block`, upload those three and
// push from them, and that staging alone would be 56 percent of the
// host-to-device bytes this solver moves.
//
// ONE BODY FOR SIX STENCILS, because they differ only in ARITY. A tet and a
// hinge name four vertices, a rod-bend and a face-strain three, a rod and a
// rod-strain two, and every one of them stores its element Hessian as a dense
// column-major `(3 * arity)` square at `(3 * arity)^2 * element`. So the span
// and the base are arithmetic on `arity` rather than six separate kernels, and
// the loops are bounded by data the caller states rather than by a template
// parameter.
//
// THE LOWER TRIANGLE IS DECLINED, NOT DROPPED. The matrix stores one of each
// pair and reaches the other through its transpose index, so `row > column` is
// the block already accounted for. A pass that pushed those too and ignored the
// refusal would store the same set.
//
// A REFUSAL IS COUNTED AND THE HOST RAISES. An elastic stencil the fixed
// pattern cannot hold is a `builder.rs` defect rather than a routing, which is
// the case the rod-bend `(j, k)` blocks lost once: the coupling went missing and
// damping masked it. The counter makes it loud without the kernel needing a
// verdict of its own.
[[seam::device_fn]] inline void fixed_push_blocks_at(
    const unsigned *index,
    const float *hessian, unsigned arity,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned element) {
    const unsigned span = 3u * arity;
    const unsigned base = span * span * element;
    const unsigned first = arity * element;
    for (unsigned ii = 0; ii < arity; ++ii) {
        for (unsigned jj = 0; jj < arity; ++jj) {
            const unsigned row = index[first + ii];
            const unsigned column = index[first + jj];
            if (row > column) {
                continue;
            }
            Mat3x3f block;
            for (unsigned c = 0; c < 3u; ++c) {
                for (unsigned r = 0; r < 3u; ++r) {
                    block.m[3u * c + r] =
                        hessian[base + span * (3u * jj + c) + 3u * ii + r];
                }
            }
            if (!fixed_csr_atomic_push(fixed_index, fixed_offset, fixed_value,
                                           row_count, row, column, block)) {
                // THE COUNT AND ONE WITNESS. A caller cannot act on a
                // per-block verdict, which is why this is a counter rather
                // than a per-block verdict array, but a fatal that names no
                // block sends its reader looking through a whole
                // stencil. Slots 1 and 2 carry A refused pair: several threads
                // may write them and the last wins, so the message says "one
                // of them" rather than "the first".
                compute::atomic_add(refused, 1u);
                // THE WITNESS IS A PLAIN ARRAY, NOT THE ATOMIC ONE. The host
                // seam DELETES assignment to an atomic slot, and rightly: a
                // plain store to one is a non-atomic write. This pair is
                // racy on purpose and does not need to be anything else,
                // because any thread that gets here has a genuine refused
                // block and the message says "one of them".
                witness[0] = row;
                witness[1] = column;
            }
        }
    }
}

// THE SAME PUSH FROM THREAD SPACE, for a kernel that formed the element's
// Hessian itself and never wrote it to memory. `fixed_push_blocks_at` reads one
// element's blocks out of a device array that a previous pass staged; this
// reads them out of the caller's registers, which is what a fused kernel wants
// when it hands the Hessian it just formed straight to the push. The block loop
// appears twice because the two read
// different address spaces, and a body that copied the device array into thread
// space to share this one would spend the registers the fused caller is saving.
//
// `hessian` is `span * span` floats column-major with `span = 3 * arity`, which
// is `Mat12x12f::m` for a four-vertex element, and `index` is the element's own
// vertex list, `arity` long.
[[seam::device_fn]] inline void fixed_push_blocks_thread(
    const unsigned *index,
    const float *hessian, unsigned arity,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness) {
    const unsigned span = 3u * arity;
    for (unsigned ii = 0; ii < arity; ++ii) {
        for (unsigned jj = 0; jj < arity; ++jj) {
            const unsigned row = index[ii];
            const unsigned column = index[jj];
            if (row > column) {
                continue;
            }
            Mat3x3f block;
            for (unsigned c = 0; c < 3u; ++c) {
                for (unsigned r = 0; r < 3u; ++r) {
                    block.m[3u * c + r] = hessian[span * (3u * jj + c) + 3u * ii + r];
                }
            }
            if (!fixed_csr_atomic_push(fixed_index, fixed_offset, fixed_value,
                                           row_count, row, column, block)) {
                // The count and one witness, as `fixed_push_blocks_at` keeps
                // them and for the reasons it gives.
                compute::atomic_add(refused, 1u);
                witness[0] = row;
                witness[1] = column;
            }
        }
    }
}

// THE SAME PUSH FROM REGISTERS AT A PRECOMPUTED SLOT, which is the DEFAULT
// deposit rather than an optimization layered on top of the search. It reads
// `slots[arity * ii + jj]` out of a table `builder.rs` computed once for the
// scene, skips the sentinel that table carries for a lower-triangle no-op, and
// adds the block. `fixed_push_blocks_thread` above reaches the same slot by
// SEARCHING the row, which is O(row width) where this is O(1), and the width
// scales with a vertex's incident element count.
//
// THE SENTINEL IS THE LOWER TRIANGLE, so this loop needs no `row > column`
// test: the table already encodes which of the arity-squared blocks are
// stored. It must match the `0xFFFFFFFFu` literal `builder.rs` writes.
//
// NO REFUSAL COUNTER, and that is deliberate rather than an omission. A slot
// came from the pattern the matrix was built over, so there is nothing to look
// up and fail; what the search path's `refused` and `witness` catch is a block
// whose stencil `builder.rs` never registered, and that question is asked when
// the table is BUILT, not when it is replayed.
[[seam::device_fn]] inline void fixed_push_blocks_thread_at(
    const unsigned *slots,
    const float *hessian, unsigned arity,
    compute::atomic_float_t *fixed_value, unsigned element) {
    const unsigned span = 3u * arity;
    const unsigned base = arity * arity * element;
    for (unsigned ii = 0; ii < arity; ++ii) {
        for (unsigned jj = 0; jj < arity; ++jj) {
            const unsigned slot = slots[base + arity * ii + jj];
            if (slot == 0xFFFFFFFFu) {
                continue;
            }
            Mat3x3f block;
            for (unsigned c = 0; c < 3u; ++c) {
                for (unsigned r = 0; r < 3u; ++r) {
                    block.m[3u * c + r] =
                        hessian[span * (3u * jj + c) + 3u * ii + r];
                }
            }
            fixed_csr_atomic_push_slot(fixed_value, slot, block);
        }
    }
}

// THE SLOT DEPOSIT FROM A DEVICE ARRAY, for a caller whose Hessian a
// previous pass staged rather than formed in registers. It is to
// `fixed_push_blocks_at` what `fixed_push_blocks_thread_at` is to
// `fixed_push_blocks_thread`, and the block loop appears twice for the reason
// stated there: the two read different address spaces.
[[seam::device_fn]] inline void fixed_push_blocks_device_at(
    const unsigned *slots,
    const float *hessian, unsigned arity,
    compute::atomic_float_t *fixed_value, unsigned element) {
    const unsigned span = 3u * arity;
    const unsigned base = span * span * element;
    const unsigned first = arity * arity * element;
    for (unsigned ii = 0; ii < arity; ++ii) {
        for (unsigned jj = 0; jj < arity; ++jj) {
            const unsigned slot = slots[first + arity * ii + jj];
            if (slot == 0xFFFFFFFFu) {
                continue;
            }
            Mat3x3f block;
            for (unsigned c = 0; c < 3u; ++c) {
                for (unsigned r = 0; r < 3u; ++r) {
                    block.m[3u * c + r] =
                        hessian[base + span * (3u * jj + c) + 3u * ii + r];
                }
            }
            fixed_csr_atomic_push_slot(fixed_value, slot, block);
        }
    }
}

// THE COMPACTED RUN'S ELEMENT, deposited at its precomputed slots. `item` is
// the thread index and the element is data, exactly as in
// `fixed_push_element_blocks`.
[[seam::entry(item)]]
[[seam::device_fn]] inline void fixed_push_element_blocks_at(
    const unsigned *active,
    const unsigned *slots,
    const float *hessian, unsigned arity,
    compute::atomic_float_t *fixed_value, unsigned item) {
    fixed_push_blocks_device_at(slots, hessian, arity, fixed_value,
                                active[item]);
}

// THE SLOT DEPOSIT WITH THE RUN DECIDED IN THE THREAD, which is the pairing a
// caller needs once it stops compacting an active list on the host: the
// element is the thread index, the gate is the element's own scale, and the
// blocks land at their precomputed slots.
//
// THE GATE IS NOT AN OPTIMIZATION. An element outside the run has a Hessian
// left at the inert seed, and its slot row is real, so depositing it would add
// zeros to blocks that belong to it; but an element `builder.rs` never
// registered has an all-sentinel row and would silently deposit NOTHING, which
// is the failure this gate makes impossible rather than merely unlikely.
[[seam::entry(element)]]
[[seam::device_fn]] inline void fixed_push_element_blocks_gated_at(
    const float *live,
    const unsigned *slots,
    const float *hessian, unsigned arity,
    compute::atomic_float_t *fixed_value, unsigned element) {
    if (live[element] <= 0.0f) {
        return;
    }
    fixed_push_blocks_device_at(slots, hessian, arity, fixed_value, element);
}

// THE ELEMENT NAMED BY A COMPACTED RUN. `item` is the thread index and the
// element is data, which is why the caller states the run's length as the
// dispatch extent rather than the element count.
[[seam::device_fn]] inline void fixed_push_element_blocks(
    const unsigned *active,
    const unsigned *index,
    const float *hessian, unsigned arity,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned item) {
    fixed_push_blocks_at(index, hessian, arity, fixed_index, fixed_offset,
                         fixed_value, row_count, refused, witness,
                         active[item]);
}

// THE ELEMENT IS THE THREAD INDEX AND THE RUN IS DECIDED IN THE KERNEL, which
// is `energy.cu`'s shape: it dispatches over the full element count and gates
// on the element's own props rather than over a list the host compacted.
//
// THE GATE IS NOT AN OPTIMIZATION AND THE ZERO BLOCKS ARE NOT HARMLESS. An
// element outside the run has a Hessian the stiffness scale left at zero, and
// adding 0.0f to a running sum is exact, so pushing it looks free. It is not:
// `fixed_csr_atomic_push` COUNTS a refusal, and an element outside the run has
// no reason for `builder.rs` to have registered its stencil in the pattern, so
// the pass would raise a fatal over blocks nobody wanted pushed.
[[seam::device_fn]] inline void fixed_push_element_blocks_gated(
    const float *live,
    const unsigned *index,
    const float *hessian, unsigned arity,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned element) {
    if (live[element] <= 0.0f) {
        return;
    }
    fixed_push_blocks_at(index, hessian, arity, fixed_index, fixed_offset,
                         fixed_value, row_count, refused, witness, element);
}

// THE SAME GATED PUSH OVER AN UNSIGNED VERDICT, because a gate is not always a
// stiffness. `fixed_push_element_blocks_gated` reads a `const float *` and tests
// `<= 0.0f`, which suits every caller whose gate IS the scale it multiplies by;
// the rod strain limiter's verdict is a `unsigned` its own body scatters
// (`strainlimiting/rod_strain.kernel.cpp`), and a `Handle` carries no type, so
// handing that array to the float form would compile and REINTERPRET THE BITS.
// It would even appear to work, `1u` being a denormal above zero and `0u` being
// exactly it, which is the kind of accident that survives every test until the
// truthy value changes.
[[seam::entry(element)]]
[[seam::device_fn]] inline void fixed_push_element_blocks_live(
    const unsigned *live,
    const unsigned *index,
    const float *hessian, unsigned arity,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned element) {
    if (live[element] == 0u) {
        return;
    }
    fixed_push_blocks_at(index, hessian, arity, fixed_index, fixed_offset,
                         fixed_value, row_count, refused, witness, element);
}

[[seam::entry(count, element)]] void fixed_push_element_blocks_gated(
    const float *live,
    const unsigned *index,
    const float *hessian, unsigned arity,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value,
    unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned element,
    unsigned count);

[[seam::entry(count, item)]] void fixed_push_element_blocks(
    const unsigned *active,
    const unsigned *index,
    const float *hessian, unsigned arity,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value,
    unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned item,
    unsigned count);

// The push as an entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the range shim a host C++ compiler compiles, and the Rust
// `#[repr(C)]` twin the driver fills.
//
// THE VERDICT IS THE POINT OF THE SCATTER AND MUST NOT BE DROPPED. The body
// returns false for a block whose `(row, column)` is outside the build-time
// sparsity, and a caller that ignores that verdict drops a Hessian block and
// leaves an indefinite matrix, which is how the rod-bend `(j, k)` stencil bug
// shipped. `[[seam::scatter]]` carries the body's RETURN VALUE to this
// element's slot, so the verdict reaches the driver by construction rather than
// by a launcher remembering to store it.
//
// `stored` is `unsigned` rather than a byte because a record holds only 4-byte
// scalars and 16-byte handles, so it has no padding and its size is the sum of
// its fields; a narrower slot would move no size and is the one drift a sizeof
// assert cannot see. The driver reads it as a zero-or-one verdict either way.
//
// `row` and `column` are gathers rather than an index list: they are this
// block's own coordinates and address nothing, so there is no slot to bound.
// `index`, `offset` and `value` stay base pointers because the body walks the
// pattern itself, from `offset[i]` to `offset[i + 1]`, over a row whose width
// is data rather than a fixed run.
[[seam::entry(count)]] void fixed_csr_atomic_push(
    const unsigned *index,
    const unsigned *offset,
    compute::atomic_float_t *value,
    unsigned row_count, const unsigned *row,
    const unsigned *column,
    const Mat3x3f *block,
    [[seam::scatter]] unsigned *stored,
    unsigned count);

// THE BLOCK-JACOBI PRECONDITIONER'S DIAGONAL, in the masked-and-unmasked pair
// shape sections 35 and 36 use: whether a scene carries a DYNAMIC matrix at all
// is a driver configuration, and the shim these replace tested
// `dyn_offset != nullptr` per row to find it out. The shim said so itself in a
// `SEAM_MOVE` note.
//
// THE ACCUMULATION ORDER IS LOAD-BEARING AND IS PRESERVED EXACTLY: zero, then
// the dynamic row's matching blocks in slot order, then the fixed diagonal
// block, then the caller's own diagonal. fp32 addition is not associative, so a
// reordering that looks equivalent is a different number.
//
// IT ACCUMULATES IN A THREAD BLOCK RATHER THAN INTO THE DESTINATION, which the
// shim could not do and which changes no value: the sequence of additions is
// the same on the same operands, so only where the running sum lives differs.
// The shim folded through `vec_add_scaled` at unit scale, and a thread-space
// source cannot reach that body, whose parameters are both `[[seam::device]]`.
// Multiplying by one is exact, so `+= 1.0f * y` and `+= y` are the same float.
[[seam::device_fn]] inline Mat3x3f precond_diagonal_finish(
    const Mat3x3f &partial,
    const unsigned *index,
    const unsigned *offset,
    const float *value, unsigned row_count,
    const float *diagonal, unsigned row) {
    Mat3x3f block = partial;
    const Mat3x3f fixed_block =
        fixed_csr_read(index, offset, value, row_count, row, row);
    for (unsigned k = 0; k < 9; ++k) {
        block.m[k] += fixed_block.m[k];
    }
    for (unsigned k = 0; k < 9; ++k) {
        block.m[k] += diagonal[9 * row + k];
    }
    return block;
}

[[seam::entry(row, out)]]
[[seam::device_fn]] inline Mat3x3f precond_diagonal(
    const unsigned *index,
    const unsigned *offset,
    const float *value, unsigned row_count,
    const float *diagonal, unsigned row) {
    const Mat3x3f zero = Mat3x3f::Zero();
    return precond_diagonal_finish(zero, index, offset, value, row_count,
                                       diagonal, row);
}

// The dynamic term is folded FIRST, which is why this is not a wrapper around
// the form above: the two differ in the ORDER of the sum, not only in whether a
// term is present.
//
// A ROW'S DYNAMIC SLOTS ARE SCANNED FOR THE DIAGONAL ONE and the rest skipped,
// rather than indexed, because the dynamic pattern is built by insertion and
// carries no promise about where a row's diagonal block sits. The sum over
// matching slots is what makes the reading correct even before a row has been
// compacted.
[[seam::entry(row, out)]]
[[seam::device_fn]] inline Mat3x3f precond_diagonal_dynamic(
    const unsigned *index,
    const unsigned *offset,
    const float *value, unsigned row_count,
    const float *diagonal,
    const unsigned *dyn_index,
    const unsigned *dyn_offset,
    const float *dyn_value, unsigned row) {
    Mat3x3f block = Mat3x3f::Zero();
    for (unsigned slot = dyn_offset[row]; slot < dyn_offset[row + 1]; ++slot) {
        if (dyn_index[slot] != row) {
            continue;
        }
        for (unsigned k = 0; k < 9; ++k) {
            block.m[k] += dyn_value[9 * slot + k];
        }
    }
    return precond_diagonal_finish(block, index, offset, value, row_count,
                                       diagonal, row);
}
