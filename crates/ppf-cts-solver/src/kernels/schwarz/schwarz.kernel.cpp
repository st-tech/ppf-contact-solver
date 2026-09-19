// File: schwarz.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The arithmetic of one additive aggregate-Schwarz domain, written once and
// rendered by ppf-cts-compute/seam/kernelgen.py into the form each of nvcc, the Metal
// shader compiler and a host C++ compiler reads. Each routine is a
// straight-line read of a dense work array in the threadgroup address space, so
// the two backends share the numerics and each keeps its own launch geometry,
// its own barriers and its own diagnostic channel.
//
// A NEUTRAL KERNEL SOURCE, so it is plain C++: no preprocessor conditional, no
// macro of its own, and no spelling that only one of those compilers accepts.
// The two facts a backend cannot infer are written as C++ attributes.
// `[[seam::device_fn]]` is the execution space. `[[seam::threadgroup]]` is the
// address space of the work array every routine here reads: MSL requires an
// address space on every pointer and reference type, and threadgroup memory is
// a distinct one there, while CUDA and the host have a single address space and
// are handed the same declarations with it removed. That qualifier is the whole
// reason these bodies take pointers rather than values: the arrays are the
// group's shared staging buffer, sized by the aggregate, and owned by no single
// lane.
//
// WHAT AN AGGREGATE FACTOR IS. `A_loc` is the principal submatrix of the Newton
// operator M = A_contact + B_fixed + C_diag restricted to one aggregate's
// degrees of freedom, gathered dense and row-major at dimension d = 3 * members.
// It is SPD because M is, so it admits a Cholesky factor A_loc = L L^T, and the
// preconditioner stores the INVERSE factor G = L^-1 rather than A_loc^-1. The
// apply is then the Gram form z = G^T (G r), whose inner product with r is
// ||G r||^2, non-negative in float32 for any G at all. That is why the
// preconditioner cannot produce the r.z <= 0 breakdown that a directly
// inverted, marginally indefinite block would.
//
// WHY THE PIVOT IS CLAMPED. The pivot is floored at the same absolute value the
// diagonal was raised by, so a block that float32 cancellation drives to a
// non-positive pivot still yields a valid SPD factor: the result is then the
// exact inverse factor of a NEARBY SPD matrix rather than a garbage inverse of
// this one. A preconditioner only sets the convergence rate, so conditioning
// with a nearby matrix costs iterations and never correctness, while an
// unclamped 1/sqrt of a negative pivot would put a NaN into the search
// direction.
//
// WHAT IS NOT HERE. The gather that fills A_loc and the barriers between the
// Cholesky columns belong to the entry point, because their shape is the launch
// geometry: which thread owns which row, and how a backend spells "wait for the
// whole group". The routines below are what each of those threads computes.

#pragma once

// Packed lower-triangle layout for a d x d lower-triangular matrix: row i holds
// entries [i, 0..i], so its base is i(i+1)/2 and (i,j) with i >= j lives at
// i(i+1)/2 + j. Storing G packed halves the bytes the apply has to stage.
[[seam::device_fn]] inline unsigned schwarz_tri_size(unsigned d) {
    return d * (d + 1u) / 2u;
}

[[seam::device_fn]] inline unsigned schwarz_tri_index(unsigned i,
                                                         unsigned j) {
    // Requires i >= j; a caller that has the transpose entry swaps first.
    return i * (i + 1u) / 2u + j;
}

// The absolute diagonal floor added to A_loc before the factorization, from the
// mean diagonal entry. It is what makes a fully prescribed or otherwise dead
// block invertible instead of a division by zero, and it doubles as the pivot
// clamp below, so the factored matrix and the clamped one differ by the same
// scale. Relative to the block's own magnitude through the trace term, with an
// absolute term so an all-zero block still floors.
[[seam::device_fn]] inline float schwarz_diagonal_floor(float trace,
                                                           unsigned d) {
    return 1.0e-6f * fmath::div(trace, (float)d) + 1.0e-8f;
}

// The reciprocal a caller forms once per Cholesky column and then multiplies
// each of that column's rows by. It is here rather than spelled at the two entry
// points so both compile the one correctly rounded division: MSL's plain `/`
// is not required to be, which is why `fmath::div` is `precise::divide` there,
// and a fast reciprocal here would put the two backends a rounding apart on
// every sub-diagonal entry of every factor.
[[seam::device_fn]] inline float schwarz_reciprocal(float x) {
    return fmath::div(1.0f, x);
}

// Column j's diagonal entry of L, from the columns below j that are already
// written. Sequential in j: every caller runs this on one thread and then
// synchronizes the group.
[[seam::device_fn]] inline float schwarz_cholesky_diagonal(
    const float *a, unsigned d, unsigned j, float floor) {
    float s = a[j * d + j];
    for (unsigned k = 0; k < j; ++k) {
        const float ljk = a[j * d + k];
        s -= ljk * ljk;
    }
    return fmath::sqrt(s > floor ? s : floor);
}

// Sub-diagonal entry (i, j) of L, i > j. The rows of one column are independent
// of each other, so a group fills them in parallel. `inverse_pivot` is the
// reciprocal of the diagonal the routine above returned, formed once per column
// by the caller rather than divided per row.
[[seam::device_fn]] inline float schwarz_cholesky_column(
    const float *a, unsigned d, unsigned j, unsigned i,
    float inverse_pivot) {
    float s = a[i * d + j];
    for (unsigned k = 0; k < j; ++k) {
        s -= a[i * d + k] * a[j * d + k];
    }
    return s * inverse_pivot;
}

// One column of G = L^-1, lower triangular, written into `inverse` at the same
// dense d x d layout. Column `column` depends only on L, which is read-only by
// this point, and on earlier rows of the SAME column, so columns are
// independent and a group fills a stripe each with no barrier between them.
[[seam::device_fn]] inline void schwarz_inverse_column(
    const float *l,
    float *inverse, unsigned d, unsigned column) {
    inverse[column * d + column] = fmath::div(1.0f, l[column * d + column]);
    for (unsigned i = column + 1; i < d; ++i) {
        float s = 0.0f;
        for (unsigned k = column; k < i; ++k) {
            s += l[i * d + k] * inverse[k * d + column];
        }
        inverse[i * d + column] = fmath::div(-s, l[i * d + i]);
    }
}

// Row p of y = G r, with G lower triangular and packed: G[p][q] is
// packed[tri_index(p, q)] for q <= p.
[[seam::device_fn]] inline float schwarz_gram_lower(
    const float *packed,
    const float *r, unsigned p) {
    float acc = 0.0f;
    for (unsigned q = 0; q <= p; ++q) {
        acc += packed[schwarz_tri_index(p, q)] * r[q];
    }
    return acc;
}

// Row p of z = G^T y. (G^T)[p][q] is G[q][p], which is
// packed[tri_index(q, p)] for q >= p.
[[seam::device_fn]] inline float schwarz_gram_upper(
    const float *packed,
    const float *y, unsigned d, unsigned p) {
    float acc = 0.0f;
    for (unsigned q = p; q < d; ++q) {
        acc += packed[schwarz_tri_index(q, p)] * y[q];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// The DOMAIN construction: `build_domains` in the reference, transcribed.
//
// It turns a per-vertex aggregate assignment into the three arrays the factor
// and the apply both read: where each aggregate's members start, who they are,
// and where each aggregate's packed inverse begins.
//
// FOUR ROWS AND TWO SCANS, and the scans are not here: the reference calls
// `kernels::exclusive_scan` between them, and this driver already carries one
// as `primitives/scan_levels.kernel.cpp`, a multi-level block scan the driver's
// `scan::ScanScratch` runs. Reusing it rather than adding a second scan is the
// standing rule about checking whether the tree already carries the mechanism.

// Count the members of each aggregate. `offset` opens at zero and this is the
// counting half of the CSR-style build; the scan turns the counts into starts.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_count_members(const unsigned *aggregate,
                      compute::atomic_uint_t *offset,
                      unsigned element) {
    compute::atomic_add(offset + aggregate[element], 1u);
}

// Place each vertex in its aggregate's span.
//
// THE POSITION IS CLAIMED, NOT COMPUTED, so the order WITHIN an aggregate is
// whatever the claims happen to interleave. That is the reference's own
// behavior (`atomicAdd(&cur.data[g], 1u)` and write at `off[g] + pos`), and it
// is not a defect here: the factor gathers the aggregate's whole dense block
// and the apply solves it, so the block is permuted consistently on both sides
// and the solve is the same solve. Do NOT "fix" it into a sorted order without
// establishing that nothing downstream reads a member index as a rank.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_scatter_members(const unsigned *aggregate,
                        const unsigned *offset,
                        compute::atomic_uint_t *cursor,
                        unsigned *members, unsigned element) {
    const unsigned group = aggregate[element];
    const unsigned position = compute::atomic_add(cursor + group, 1u);
    members[offset[group] + position] = element;
}

// Each aggregate's packed lower-triangle size, in floats.
//
// The dimension is `3 * members` because a member is a VERTEX and a vertex
// carries three degrees of freedom, and the packing is triangular because the
// local matrix is symmetric. This is the counting half again: the scan that
// follows turns these into the offsets the factor writes at.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_domain_inverse_size(const unsigned *offset,
                            unsigned *inverse_offset,
                            unsigned element) {
    const unsigned dimension = 3u * (offset[element + 1u] - offset[element]);
    inverse_offset[element] = schwarz_tri_size(dimension);
}

// ---------------------------------------------------------------------------
// The FINE GRAPH: `build_fine_graph` in the reference, transcribed.
//
// The partition needs a weighted adjacency over the Newton operator's block
// rows, and the operator is stored in two matrices with four spans per row: the
// dynamic matrix's direct entries and its reference entries, and the fixed
// matrix's entries and its transpose pairs. The graph is the union of all four
// with one weight each.

// A 3x3 block's Frobenius norm, which is the edge weight.
//
// `frob9` in the reference. IT IS A MAGNITUDE, NOT A STIFFNESS IN ANY UNIT: the
// partition compares it against other blocks' norms and never against a
// threshold, so the scale is free. `precise::sqrt` is not required here for the
// reason the ACCD sites need it, since nothing downstream is a predicate on
// this value; it orders a sort.
[[seam::device_fn]] inline float
schwarz_block_weight(const float *block) {
    float sum = 0.0f;
    for (unsigned entry = 0; entry < 9u; ++entry) {
        sum += block[entry] * block[entry];
    }
    return fmath::sqrt(sum);
}

// One row's edge count, which is the counting half of the same CSR build the
// domains use.
[[seam::device_fn]] inline void
schwarz_fine_graph_count(const unsigned *dynamic_offset,
                         const unsigned *reference_offset,
                         const unsigned *fixed_offset,
                         const unsigned *transpose_offset,
                         unsigned *count, unsigned row) {
    count[row] = (dynamic_offset[row + 1u] - dynamic_offset[row]) +
                 (reference_offset[row + 1u] - reference_offset[row]) +
                 (fixed_offset[row + 1u] - fixed_offset[row]) +
                 (transpose_offset[row + 1u] - transpose_offset[row]);
}

// One row's edges, written at the offset the scan produced.
//
// THE FOUR LOOPS ARE THE REFERENCE'S FOUR LOOPS IN ITS ORDER, and the order is
// not decorative: the partition sorts by accumulated weight and a different
// concatenation would break ties differently. A reference entry addresses the
// dynamic matrix's global value buffer indirectly, which is why it reads
// `global_value + 9 * reference_value[slot]` rather than a value of its own; a
// transpose pair does the same into the fixed matrix's values, taking the slot
// from the pair's second component.
[[seam::device_fn]] inline void
schwarz_fine_graph_fill(const unsigned *dynamic_index,
                        const float *dynamic_value,
                        const unsigned *dynamic_offset,
                        const unsigned *reference_index,
                        const unsigned *reference_value,
                        const unsigned *reference_offset,
                        const float *global_value,
                        const unsigned *fixed_index,
                        const unsigned *fixed_offset,
                        const float *fixed_value,
                        const unsigned *transpose_pair,
                        const unsigned *transpose_offset,
                        const unsigned *graph_offset,
                        unsigned *column,
                        float *weight, unsigned row) {
    unsigned write = graph_offset[row];
    for (unsigned slot = dynamic_offset[row]; slot < dynamic_offset[row + 1u];
         ++slot) {
        column[write] = dynamic_index[slot];
        weight[write] = schwarz_block_weight(dynamic_value + 9u * slot);
        ++write;
    }
    for (unsigned slot = reference_offset[row];
         slot < reference_offset[row + 1u]; ++slot) {
        column[write] = reference_index[slot];
        weight[write] =
            schwarz_block_weight(global_value + 9u * reference_value[slot]);
        ++write;
    }
    for (unsigned slot = fixed_offset[row]; slot < fixed_offset[row + 1u];
         ++slot) {
        column[write] = fixed_index[slot];
        weight[write] = schwarz_block_weight(fixed_value + 9u * slot);
        ++write;
    }
    for (unsigned slot = transpose_offset[row];
         slot < transpose_offset[row + 1u]; ++slot) {
        column[write] = transpose_pair[2u * slot + 0u];
        weight[write] =
            schwarz_block_weight(fixed_value + 9u * transpose_pair[2u * slot + 1u]);
        ++write;
    }
}

[[seam::entry(count_of_rows, row)]] void schwarz_fine_graph_count(
    const unsigned *dynamic_offset,
    const unsigned *reference_offset,
    const unsigned *fixed_offset,
    const unsigned *transpose_offset,
    unsigned *count,
    unsigned row, unsigned count_of_rows);

[[seam::entry(count, row)]] void schwarz_fine_graph_fill(
    const unsigned *dynamic_index,
    const float *dynamic_value,
    const unsigned *dynamic_offset,
    const unsigned *reference_index,
    const unsigned *reference_value,
    const unsigned *reference_offset,
    const float *global_value,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const unsigned *graph_offset,
    unsigned *column,
    float *weight,
    unsigned row, unsigned count);

// ---------------------------------------------------------------------------
// The per-domain FACTORIZATION: `factor_kernel` in the reference, decomposed.
//
// THE REFERENCE IS ONE COOPERATIVE KERNEL AND THIS IS SIX ELEMENT-WISE ROWS.
// The arithmetic is identical, body for body: the rows below call the same
// shared bodies the reference's kernel calls, in the same order, over the same
// dense layout. What differs is where the phases are separated. The reference
// puts `__syncthreads()` between them; here the separation is KERNEL
// COMPLETION, which is the decomposition this port already uses for the PDRD
// cooperative kernels.
//
// IT IS DECOMPOSED BECAUSE OF THE CPU BACKEND, not for taste. `kernelgen.py`
// runs a group entry's lanes one after another, lane 0 through the whole body
// first, and leaves `compute::threadgroup_barrier` undefined so that a body
// which synchronizes is refused at the neutral file and line. A cooperative
// factorization therefore has no host rendering at all.
//
// THE DENSE BLOCK AND THE WORKING G ARE DEVICE ARRAYS, one `d * d` span per
// aggregate at `dense_offset`, which is the layout Metal's kernel already used
// for `dense` and now uses for both.
//
// WHAT IS STORED IS G = L^{-1}, NOT THE INVERSE OF THE BLOCK, and that is
// guarantee-class rather than a choice. The apply is `z = G^T (G r)`, so
// `r . z = ||G r||^2 >= 0` in float32 whatever the rounding, and the aggregate
// term can never produce the `rz <= 0` breakdown that latches the block-Jacobi
// fallback. Storing `A^{-1}` and applying it symmetrically does not have that
// property.

// A member's local slot, or the sentinel when the column is outside this
// aggregate. `find_local` in the reference: a linear scan, because an aggregate
// is bounded by the dense-block cap and a search structure would cost more than
// it saves.
[[seam::device_fn]] inline unsigned
schwarz_find_local(const unsigned *members, unsigned base,
                   unsigned count, unsigned vert) {
    for (unsigned slot = 0; slot < count; ++slot) {
        if (members[base + slot] == vert) {
            return slot;
        }
    }
    return 0xffffffffu;
}

// Accumulate one operator block into the dense local matrix.
//
// `transposed` selects the orientation, which is what lets the four operator
// spans share one helper: the dynamic matrix's reference entries and the fixed
// matrix's transpose pairs store the block for the OTHER triangle, so they
// contribute `blk(s, r)` where the direct spans contribute `blk(r, s)`.
// A 3x3 BLOCK IS COLUMN-MAJOR, which decides both spellings below and is the
// one thing here that cannot be read off the loop. `Mat3x3f` is
// `linalg::SMat<float, 3, 3>`, so element `(r, c)` lives at `block[3 * c + r]`,
// and `mat3_mul` says the same: its `result[r]` reads `matrix[r]`,
// `matrix[3 + r]` and `matrix[6 + r]`. Reading `block[3 * r + s]` for element
// `(r, s)` transposes every block, which is SYMMETRIC over a whole matrix and
// therefore invisible to a symmetry check, and wrong.
[[seam::device_fn]] inline void
schwarz_accumulate_block(float *dense, unsigned dimension,
                         unsigned local_row, unsigned local_column,
                         const float *block, bool transposed) {
    for (unsigned r = 0; r < 3u; ++r) {
        for (unsigned s = 0; s < 3u; ++s) {
            const float value =
                transposed ? block[3u * r + s] : block[3u * s + r];
            dense[(3u * local_row + r) * dimension + (3u * local_column + s)] +=
                value;
        }
    }
}

// ROW 1. One member's rows of the dense block.
//
// THREAD `local` OWNS MEMBER ROW `local`, which is the reference's assignment
// and is what makes the accumulation race-free without an atomic: every write
// lands in rows `3*local .. 3*local+2` and no other thread touches them. The
// four spans are walked in the reference's order. The dense span opens at zero,
// which the caller's fill guarantees.
// The six rows' entry points. Each is ELEMENT-shaped: the aggregate and the
// slot within it are recovered from one flat index by a `stride` the host
// picks, which is the dense-block cap for the member rows and the maximum
// dimension for the cell passes. A slot past its own aggregate's size returns,
// which is the same guard the reference's stride loops carry.
[[seam::entry(element)]]
[[seam::device_fn]] inline void schwarz_factor_gather(
    const unsigned *aggregate_offset,
    const unsigned *members,
    const unsigned *dense_offset,
    const unsigned *dynamic_index,
    const float *dynamic_value,
    const unsigned *dynamic_offset,
    const unsigned *reference_index,
    const unsigned *reference_value,
    const unsigned *reference_offset,
    const float *global_value,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal,
    float *dense, unsigned stride, unsigned element) {
    const unsigned group = element / stride;
    const unsigned local = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned count = aggregate_offset[group + 1u] - base;
    if (local >= count) {
        return;
    }
    const unsigned dimension = 3u * count;
    float *block = dense + dense_offset[group];
    const unsigned row = members[base + local];

    for (unsigned slot = dynamic_offset[row]; slot < dynamic_offset[row + 1u];
         ++slot) {
        const unsigned column =
            schwarz_find_local(members, base, count, dynamic_index[slot]);
        if (column != 0xffffffffu) {
            schwarz_accumulate_block(block, dimension, local, column,
                                         dynamic_value + 9u * slot, false);
        }
    }
    for (unsigned slot = reference_offset[row];
         slot < reference_offset[row + 1u]; ++slot) {
        const unsigned column =
            schwarz_find_local(members, base, count, reference_index[slot]);
        if (column != 0xffffffffu) {
            schwarz_accumulate_block(
                block, dimension, local, column,
                global_value + 9u * reference_value[slot], true);
        }
    }
    for (unsigned slot = fixed_offset[row]; slot < fixed_offset[row + 1u];
         ++slot) {
        const unsigned column =
            schwarz_find_local(members, base, count, fixed_index[slot]);
        if (column != 0xffffffffu) {
            schwarz_accumulate_block(block, dimension, local, column,
                                         fixed_value + 9u * slot, false);
        }
    }
    for (unsigned slot = transpose_offset[row];
         slot < transpose_offset[row + 1u]; ++slot) {
        const unsigned column = schwarz_find_local(
            members, base, count, transpose_pair[2u * slot + 0u]);
        if (column != 0xffffffffu) {
            schwarz_accumulate_block(
                block, dimension, local, column,
                fixed_value + 9u * transpose_pair[2u * slot + 1u], true);
        }
    }
    // THE OPERATOR'S DIAGONAL, which is neither matrix's. The Newton operator
    // is the two matrices PLUS a per-vertex 3x3 block, and that block is the
    // dominant term: it carries `mass / dt^2` and every energy's diagonal
    // contribution. Omitting it factors something that is not the operator, and
    // the preconditioner built from it makes the solve worse rather than
    // better. The reference adds it here, after the four spans, at `(a, a)`.
    schwarz_accumulate_block(block, dimension, local, local,
                                 diagonal + 9u * row, false);
}

// ROW 2. The diagonal floor, one thread per aggregate.
//
// A fraction of the mean diagonal plus an absolute term, ADDED to every
// diagonal entry rather than used as a clamp: the block is a principal
// submatrix of an SPD operator and so is SPD in exact arithmetic, and the floor
// is what keeps float32 cancellation from producing a pivot with no real root.
// It is kept because the Cholesky row below clamps against it per column.
[[seam::entry(group)]]
[[seam::device_fn]] inline void
schwarz_factor_floor(const unsigned *aggregate_offset,
                     const unsigned *dense_offset,
                     float *dense,
                     float *floor_out, unsigned group) {
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    float *block = dense + dense_offset[group];
    float trace = 0.0f;
    for (unsigned k = 0; k < dimension; ++k) {
        trace += block[k * dimension + k];
    }
    const float floor_value = schwarz_diagonal_floor(trace, dimension);
    for (unsigned k = 0; k < dimension; ++k) {
        block[k * dimension + k] += floor_value;
    }
    floor_out[group] = floor_value;
}

// ROW 3. Column `column`'s DIAGONAL entry, one thread per aggregate.
//
// The reference gives this to lane 0 and barriers, because every row of the
// same column divides by it. Here the dispatch boundary is the barrier. An
// aggregate whose dimension this column has already passed does nothing, which
// is what lets one dispatch serve every aggregate at column `column` however
// their sizes differ.
[[seam::entry(group)]]
[[seam::device_fn]] inline void schwarz_factor_cholesky_diagonal(
    const unsigned *aggregate_offset,
    const unsigned *dense_offset,
    const float *floor_in, float *dense,
    unsigned column, unsigned group) {
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    if (column >= dimension) {
        return;
    }
    float *block = dense + dense_offset[group];
    block[column * dimension + column] =
        schwarz_cholesky_diagonal(block, dimension, column, floor_in[group]);
}

// ROW 4. Column `column`'s SUB-DIAGONAL rows, one thread per (aggregate, row).
//
// Every row of the column is independent once the diagonal is known, which is
// the reference's own inner parallelism. Only the stripe assignment differs,
// and a stripe assignment cannot change a value: each row's sum is over the
// same terms in the same order.
[[seam::entry(element)]]
[[seam::device_fn]] inline void schwarz_factor_cholesky_column(
    const unsigned *aggregate_offset,
    const unsigned *dense_offset,
    float *dense, unsigned column, unsigned stride,
    unsigned element) {
    const unsigned group = element / stride;
    const unsigned offset = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    const unsigned row = column + 1u + offset;
    if (column >= dimension || row >= dimension) {
        return;
    }
    float *block = dense + dense_offset[group];
    const float inverse_pivot =
        schwarz_reciprocal(block[column * dimension + column]);
    block[row * dimension + column] = schwarz_cholesky_column(
        block, dimension, column, row, inverse_pivot);
}

// ROW 5. One column of `G = L^{-1}`, one thread per (aggregate, column).
//
// COLUMNS ARE INDEPENDENT and the reference says why: column `column` of G
// depends only on L, which is read-only by now, and on earlier rows of its OWN
// column. So this needs no barrier inside it and no ordering between columns.
[[seam::entry(element)]]
[[seam::device_fn]] inline void schwarz_factor_inverse_column(
    const unsigned *aggregate_offset,
    const unsigned *dense_offset,
    const float *dense, float *work,
    unsigned stride, unsigned element) {
    const unsigned group = element / stride;
    const unsigned column = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    if (column >= dimension) {
        return;
    }
    schwarz_inverse_column(dense + dense_offset[group],
                               work + dense_offset[group], dimension, column);
}

// ROW 6. Pack `G`'s lower triangle into the aggregate's slot.
//
// A TRIANGLE because G is lower-triangular: storing the zeros above the
// diagonal would cost the apply a wider read for no term.
// `schwarz_tri_index` is the mapping the sizing pass counted with, which is
// what makes the two agree.
[[seam::entry(element)]]
[[seam::device_fn]] inline void schwarz_factor_pack(
    const unsigned *aggregate_offset,
    const unsigned *dense_offset,
    const unsigned *inverse_offset,
    const float *work, float *packed,
    unsigned stride, unsigned element) {
    const unsigned group = element / stride;
    const unsigned cell = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    if (cell >= dimension * dimension) {
        return;
    }
    const unsigned row = cell / dimension;
    const unsigned column = cell % dimension;
    if (row < column) {
        return;
    }
    packed[inverse_offset[group] + schwarz_tri_index(row, column)] =
        work[dense_offset[group] + cell];
}

// ---------------------------------------------------------------------------
// The APPLY: `apply_kernel` in the reference, decomposed the same way.
//
// `z_domain = A_loc^{-1} r_domain`, applied as `z = G^T (G r)` with `G = L^{-1}`
// the packed factor. THIS IS SPD BY CONSTRUCTION and that is the reason for the
// two-pass form rather than one symmetric matvec: `r . z = ||G r||^2 >= 0` in
// float32 whatever the rounding, so an aggregate term can never produce the
// `rz <= 0` breakdown that latches the block-Jacobi fallback.
//
// Three rows over one flat index, `element = group * stride + p`, where `p`
// runs over the domain's `d = 3 * members` degrees of freedom. The reference's
// `rloc` and `yloc` are threadgroup arrays there and per-aggregate spans of a
// device scratch here, for the reason the factorization states.
//
// A VERTEX IS WRITTEN BY EXACTLY ONE AGGREGATE, because the aggregates
// PARTITION the vertices, so the final scatter assigns rather than accumulates
// and needs no atomic. That is the reference's own `result.data[...] = acc`.

// Row 1. Gather this domain's slice of the residual into a contiguous span.
//
// `p / 3` is the member and `p % 3` its component, which is what makes the
// domain's degrees of freedom contiguous for the two matvecs below.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_apply_gather(const unsigned *aggregate_offset,
                     const unsigned *members,
                     const float *x,
                     float *residual_local, unsigned stride,
                     unsigned element) {
    const unsigned group = element / stride;
    const unsigned p = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    if (p >= dimension) {
        return;
    }
    residual_local[group * stride + p] =
        x[3u * members[base + p / 3u] + (p % 3u)];
}

// Row 2. `y = G r`, the lower-triangular matvec.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_apply_lower(const unsigned *aggregate_offset,
                    const unsigned *inverse_offset,
                    const float *packed,
                    const float *residual_local,
                    float *y_local, unsigned stride,
                    unsigned element) {
    const unsigned group = element / stride;
    const unsigned p = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    if (p >= dimension) {
        return;
    }
    y_local[group * stride + p] =
        schwarz_gram_lower(packed + inverse_offset[group],
                               residual_local + group * stride, p);
}

// Row 3. `z = G^T y`, scattered back to the vertices this domain owns.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_apply_upper(const unsigned *aggregate_offset,
                    const unsigned *members,
                    const unsigned *inverse_offset,
                    const float *packed,
                    const float *y_local,
                    float *result, unsigned stride,
                    unsigned element) {
    const unsigned group = element / stride;
    const unsigned p = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned dimension = 3u * (aggregate_offset[group + 1u] - base);
    if (p >= dimension) {
        return;
    }
    result[3u * members[base + p / 3u] + (p % 3u)] =
        schwarz_gram_upper(packed + inverse_offset[group],
                               y_local + group * stride, dimension, p);
}

// ---------------------------------------------------------------------------
// The LEVEL TRANSFERS of the multilevel extension: `restrict_direct` and
// `prolong_add_direct` in the reference.
//
// The coarse space is PIECEWISE CONSTANT: level `l`'s node `g` owns every fine
// vertex whose `map_fine` entry is `g`, and the transfer between them is a plain
// sum one way and a broadcast the other. That is the `C_l` the Galerkin product
// coarsens with, so restriction and prolongation are transposes of each other
// by construction rather than by two definitions agreeing.

// `rc[g] += x[i]` for every fine vertex `i` the coarse node `g` owns.
//
// AN ATOMIC BECAUSE THE MAP IS MANY TO ONE, which is the reference's own
// `atomicAdd`: several fine vertices land on one coarse node and the order they
// arrive in is whatever the hardware gives. The sum is over floats, so the
// order is a reassociation and the result differs between runs in the last
// bits; that is the same reassociation every atomic scatter in this tree
// already carries, and the preconditioner is a heuristic whose exactness the
// solve does not depend on.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_restrict_row(const unsigned *map_fine,
                     const float *x,
                     compute::atomic_float_t *coarse,
                     unsigned element) {
    const unsigned group = map_fine[element];
    compute::atomic_add(coarse + 3u * group + 0u, x[3u * element + 0u]);
    compute::atomic_add(coarse + 3u * group + 1u, x[3u * element + 1u]);
    compute::atomic_add(coarse + 3u * group + 2u, x[3u * element + 2u]);
}

// `z[i] += weight * ec[g]`, the transpose of the restriction above.
//
// NO ATOMIC HERE, and the asymmetry is the map's: one fine vertex reads one
// coarse node, so every write lands in a slot no other thread touches. The
// weight is the additive level's own scale, which is what keeps a sum of
// several levels' corrections from over-correcting.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_prolong_row(const unsigned *map_fine,
                    const float *coarse,
                    float *z, float weight,
                    unsigned element) {
    const unsigned group = map_fine[element];
    for (unsigned k = 0; k < 3u; ++k) {
        z[3u * element + k] += weight * coarse[3u * group + k];
    }
}

// Level `l`'s map from a fine vertex, composed from the level below it.
//
// `map_fine[i]` is which node of level `l` owns fine vertex `i`. At level 1
// that is the fine aggregation itself; above it, it is the level below's map
// followed by that level's own aggregation, which is the composition the
// reference writes as `pa[pm[i]]`.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_compose_map_row(const unsigned *previous_map,
                        const unsigned *previous_aggregate,
                        unsigned *map_fine,
                        unsigned element) {
    map_fine[element] = previous_aggregate[previous_map[element]];
}

// ---------------------------------------------------------------------------
// LEVEL 0 MATERIALIZED: `materialize_level0` in the reference.
//
// The multilevel extension coarsens a BLOCK-CSR matrix, and the fine operator
// is not one: it is two matrices in one-triangle storage plus a per-vertex
// diagonal. This writes it out as a single row-major block CSR carrying BOTH
// triangles, which is the form the Galerkin product and the coarse factor read.
//
// THE SHAPE IS THE FINE GRAPH'S, one entry per stored block instead of one
// weight, PLUS the diagonal, which the graph did not need and the operator
// does: its `+ 1u` per row is the reference's own.
//
// NOTHING IS DEDUPLICATED HERE. A vertex pair coupled through both matrices
// appears twice, and every consumer sums a row's entries, so two entries at one
// column are the same as one carrying their sum. The Galerkin product dedups by
// sorting, which is where it matters.

// One row's entry count: the four spans plus its own diagonal.
[[seam::device_fn]] inline void
schwarz_level0_count(const unsigned *dynamic_offset,
                     const unsigned *reference_offset,
                     const unsigned *fixed_offset,
                     const unsigned *transpose_offset,
                     unsigned *count, unsigned row) {
    count[row] = (dynamic_offset[row + 1u] - dynamic_offset[row]) +
                 (reference_offset[row + 1u] - reference_offset[row]) +
                 (fixed_offset[row + 1u] - fixed_offset[row]) +
                 (transpose_offset[row + 1u] - transpose_offset[row]) + 1u;
}

// Copy one 3x3 block, transposing it or not.
//
// COLUMN-MAJOR, as everywhere: element `(r, c)` is at `[3 * c + r]`, so a
// transpose swaps the two indices and a straight copy is nine reads in order.
[[seam::device_fn]] inline void
schwarz_copy_block(const float *source,
                   float *destination, bool transposed) {
    for (unsigned r = 0; r < 3u; ++r) {
        for (unsigned c = 0; c < 3u; ++c) {
            destination[3u * c + r] =
                transposed ? source[3u * r + c] : source[3u * c + r];
        }
    }
}

// One row's blocks, written at the offset the scan produced.
[[seam::device_fn]] inline void
schwarz_level0_fill(const unsigned *dynamic_index,
                    const float *dynamic_value,
                    const unsigned *dynamic_offset,
                    const unsigned *reference_index,
                    const unsigned *reference_value,
                    const unsigned *reference_offset,
                    const float *global_value,
                    const unsigned *fixed_index,
                    const unsigned *fixed_offset,
                    const float *fixed_value,
                    const unsigned *transpose_pair,
                    const unsigned *transpose_offset,
                    const float *diagonal,
                    const unsigned *coarse_offset,
                    unsigned *column,
                    float *value, unsigned row) {
    unsigned write = coarse_offset[row];
    for (unsigned slot = dynamic_offset[row]; slot < dynamic_offset[row + 1u];
         ++slot) {
        column[write] = dynamic_index[slot];
        schwarz_copy_block(dynamic_value + 9u * slot, value + 9u * write, false);
        ++write;
    }
    for (unsigned slot = reference_offset[row];
         slot < reference_offset[row + 1u]; ++slot) {
        column[write] = reference_index[slot];
        schwarz_copy_block(global_value + 9u * reference_value[slot],
                               value + 9u * write, true);
        ++write;
    }
    for (unsigned slot = fixed_offset[row]; slot < fixed_offset[row + 1u];
         ++slot) {
        column[write] = fixed_index[slot];
        schwarz_copy_block(fixed_value + 9u * slot, value + 9u * write, false);
        ++write;
    }
    for (unsigned slot = transpose_offset[row];
         slot < transpose_offset[row + 1u]; ++slot) {
        column[write] = transpose_pair[2u * slot + 0u];
        schwarz_copy_block(fixed_value + 9u * transpose_pair[2u * slot + 1u],
                               value + 9u * write, true);
        ++write;
    }
    column[write] = row;
    schwarz_copy_block(diagonal + 9u * row, value + 9u * write, false);
}

[[seam::entry(count_of_rows, row)]] void schwarz_level0_count(
    const unsigned *dynamic_offset,
    const unsigned *reference_offset,
    const unsigned *fixed_offset,
    const unsigned *transpose_offset,
    unsigned *count,
    unsigned row, unsigned count_of_rows);

[[seam::entry(count, row)]] void schwarz_level0_fill(
    const unsigned *dynamic_index,
    const float *dynamic_value,
    const unsigned *dynamic_offset,
    const unsigned *reference_index,
    const unsigned *reference_value,
    const unsigned *reference_offset,
    const float *global_value,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal,
    const unsigned *coarse_offset,
    unsigned *column,
    float *value,
    unsigned row, unsigned count);

// ---------------------------------------------------------------------------
// The COARSE gather: `factor_kernel_coarse`'s first phase.
//
// A coarse level is one block-CSR matrix carrying both triangles, so this is a
// SINGLE span walk where the fine gather walks four and adds a diagonal. Every
// phase after it is shared: the floor, the Cholesky's two rows, the triangular
// inverse and the pack all read `dense` and know nothing about where it came
// from, which is what makes one factorization serve both levels.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
schwarz_coarse_gather(const unsigned *aggregate_offset,
                      const unsigned *members,
                      const unsigned *dense_offset,
                      const unsigned *matrix_offset,
                      const unsigned *matrix_column,
                      const float *matrix_value,
                      float *dense, unsigned stride,
                      unsigned element) {
    const unsigned group = element / stride;
    const unsigned local = element % stride;
    const unsigned base = aggregate_offset[group];
    const unsigned count = aggregate_offset[group + 1u] - base;
    if (local >= count) {
        return;
    }
    const unsigned dimension = 3u * count;
    float *block = dense + dense_offset[group];
    const unsigned row = members[base + local];
    for (unsigned slot = matrix_offset[row]; slot < matrix_offset[row + 1u];
         ++slot) {
        const unsigned column =
            schwarz_find_local(members, base, count, matrix_column[slot]);
        if (column != 0xffffffffu) {
            schwarz_accumulate_block(block, dimension, local, column,
                                         matrix_value + 9u * slot, false);
        }
    }
}

// ---------------------------------------------------------------------------
// The GALERKIN coarsening: `galerkin` in the reference, transcribed.
//
// `dst = C src C^T` for a piecewise-constant `C`, whose columns are indicator
// vectors, so every source entry `(i, j)` lands at the coarse entry
// `(agg[i], agg[j])` and the entries landing together are SUMMED. The reference
// assembles that COO product into CSR by packing each entry's coarse pair into
// one 32-bit key `g * n_agg + h`, sorting the keys, flagging the first entry of
// each equal-key run, scanning the flags into edge indices and segment-summing
// the nine floats of each run. The sort makes a coarse edge's source entries
// contiguous, so the sum needs no atomic.
//
// FOUR ROWS AND TWO SCANS, and the scans are `scan::ScanScratch` again, for the
// reason the domain construction reuses it: the reference calls
// `kernels::exclusive_scan` between these passes, and that is what this driver
// already carries.
//
// WHAT IS NOT HERE IS THE SORT. `primitives/radix.kernel.cpp` is cooperative and
// has no host rendering, so the driver sorts the KEYS through the bitonic
// network in `devsort::SortScratch` between the first and second rows. NOTHING
// CROSSES TO THE HOST: a host sort would bring the keys and their permutation
// down for a `par_stable_sort_by_key` and send them back, where the network
// sorts them in place.

// ROW 1. One source row's keys, and the identity permutation the sort carries
// alongside them.
//
// THE KEY PACKS BOTH COARSE INDICES INTO ONE UNSIGNED, which is what lets a
// single sort group by the pair: `g * coarse_rows + h` with `h < coarse_rows`
// is monotone in `(g, h)` lexicographically, so a sorted run of equal keys is
// exactly one coarse entry and the sorted edges come out in CSR order, row by
// row and column by column within a row. The product fits 32 bits at the sizes
// a coarse level reaches, which is the reference's own bound.
[[seam::entry(row)]]
[[seam::device_fn]] inline void
schwarz_galerkin_key(const unsigned *aggregate,
                     const unsigned *offset,
                     const unsigned *column,
                     unsigned *key,
                     unsigned *permutation,
                     unsigned coarse_rows, unsigned row) {
    const unsigned coarse_row = aggregate[row];
    for (unsigned slot = offset[row]; slot < offset[row + 1u]; ++slot) {
        key[slot] = coarse_row * coarse_rows + aggregate[column[slot]];
        permutation[slot] = slot;
    }
}

// ROW 2. The first entry of each equal-key run, as a zero-or-one flag.
//
// The scan that follows turns the flags into each run's edge index, and its
// total is the number of coarse entries. `key` is a BASE pointer rather than an
// element gather because the body reads its neighbor below as well as its own
// slot, which is what makes a run's first entry recognizable without a second
// pass over the runs.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void
schwarz_galerkin_edge_flag(const unsigned *key,
                           unsigned *edge, unsigned slot) {
    edge[slot] = (slot == 0u || key[slot] != key[slot - 1u]) ? 1u : 0u;
}

// ROW 3. Each run's head: the coarse column it lands in, where the run starts,
// and one count into its coarse row.
//
// THE ROW COUNT IS THE ONLY ATOMIC IN THE COARSENING, and it is a count rather
// than a claim: the edge INDEX is already decided by the scan, so nothing here
// depends on the order the increments interleave. The scan that follows turns
// the counts into the coarse matrix's own row offsets.
//
// `edge_start` is written at the run's edge index and read back by the
// segmented sum, which needs the run's end as well as its start; the host
// writes the final entry, the source entry count, so that the last run has an
// end like every other one.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void
schwarz_galerkin_edge_head(const unsigned *key,
                           const unsigned *edge,
                           unsigned *column,
                           unsigned *edge_start,
                           compute::atomic_uint_t *row_count,
                           unsigned coarse_rows, unsigned slot) {
    if (slot != 0u && key[slot] == key[slot - 1u]) {
        return;
    }
    const unsigned edge_index = edge[slot];
    column[edge_index] = key[slot] % coarse_rows;
    edge_start[edge_index] = slot;
    compute::atomic_add(row_count + key[slot] / coarse_rows, 1u);
}

// ROW 4. One coarse entry's nine floats, summed over its run.
//
// THE RUN IS CONTIGUOUS BECAUSE THE KEYS WERE SORTED, so this is a plain
// segmented sum with no atomic and no read-modify-write of a shared slot. The
// permutation is what carries each sorted slot back to the source entry it came
// from, and the accumulation walks the run in sorted order, which for a stable
// sort is the source entries' own index order.
[[seam::entry(edge)]]
[[seam::device_fn]] inline void
schwarz_galerkin_segment_sum(const unsigned *edge_start,
                             const unsigned *permutation,
                             const float *source_value,
                             float *value, unsigned edge) {
    float acc[9];
    for (unsigned entry = 0; entry < 9u; ++entry) {
        acc[entry] = 0.0f;
    }
    for (unsigned slot = edge_start[edge]; slot < edge_start[edge + 1u];
         ++slot) {
        const float *block =
            source_value + 9u * permutation[slot];
        for (unsigned entry = 0; entry < 9u; ++entry) {
            acc[entry] += block[entry];
        }
    }
    for (unsigned entry = 0; entry < 9u; ++entry) {
        value[9u * edge + entry] = acc[entry];
    }
}
