// File: spmv.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// THE PCG LATCH SLOTS, in the float array pcg.rs passes as `latch`: the freeze,
// the direction combine's one-iteration-late copy of it, the mode (0 far,
// 1 near) and the iterate the freeze fell on. Defined here because this file
// is the first one pcg.kernel.cpp includes and the products read the freeze.
enum : unsigned {
    PCG_LATCH_FROZEN = 0u,
    PCG_LATCH_COMBINE = 1u,
    PCG_LATCH_MODE = 2u,
    PCG_LATCH_AT = 3u
};

// WHERE A GROUP'S PARTIAL LIVES, for the PCG's two single-group folds
// (`pcg_fold_alpha`, `pcg_fold_beta` in pcg.kernel.cpp). Defined here, beside
// the latch slots, because the folded applies below write these partials and
// this file is the first one pcg.kernel.cpp includes.
//
// A fold of `groups` partials gives each of its PCG_FOLD_LANES lanes one
// contiguous run of `per = ceil(groups / lanes)` of them and adds the run in
// order. Stored in group order, the lanes' k-th reads sit `per` floats apart,
// so every warp load of the fold touches as many cache lines as it has lanes:
// measured on twist, 5.3 us for a fold of 7,800 pairs of floats on one group.
// Stored LANE-MAJOR, group `g` (lane `g / per`, step `g % per`) at
// `(g % per) * lanes + g / per`, the same reads are consecutive. Each lane
// still adds the same partials in the same order, so the sums are the same
// bits; only the addresses move.
//
enum : unsigned { PCG_FOLD_LANES = 64u };

[[seam::device_fn]] inline unsigned pcg_fold_slot(unsigned group,
                                                  unsigned groups) {
    const unsigned per = (groups + PCG_FOLD_LANES - 1u) / PCG_FOLD_LANES;
    return (group % per) * PCG_FOLD_LANES + group / per;
}

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::device]]` is the address space of a pointer parameter,
// which MSL requires on every reference and pointer type.
//
// No include of its own. `Vec3f` arrives from whatever declares it for the
// backend that is compiling, which is data.hpp under nvcc and on the host and
// the shader prologue's aliases under MSL.
//
// EVERY POINTER HERE IS `[[seam::device]]`, and that is a statement about where
// the memory is rather than a formality. The row's coefficients, the CSR index
// arrays and the vector being multiplied are all solver-owned buffers read by
// every thread in the dispatch, so under MSL they are `device` pointers; a
// `thread` pointer would name a per-thread copy that does not exist.

[[seam::device_fn]] inline Vec3f
mat3_mul(const float *matrix,
             const float *vector) {
    Vec3f result;
    result[0] =
        matrix[0] * vector[0] + matrix[3] * vector[1] + matrix[6] * vector[2];
    result[1] =
        matrix[1] * vector[0] + matrix[4] * vector[1] + matrix[7] * vector[2];
    result[2] =
        matrix[2] * vector[0] + matrix[5] * vector[1] + matrix[8] * vector[2];
    return result;
}

[[seam::device_fn]] inline Vec3f
mat3_transpose_mul(const float *matrix,
                       const float *vector) {
    Vec3f result;
    result[0] =
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
    result[1] =
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
    result[2] =
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
    return result;
}

// THE MAGNITUDE OF ONE BLOCK'S CONTRIBUTION, TAKEN OVER ITS ENTRY TERMS.
//
// `p^T A p` is a sum of signed fp32 terms, so when the true curvature is small
// beside the individual terms the fp32 result is dominated by round-off. A
// bound on that error is what lets a caller tell a canceled zero from a
// genuinely negative curvature.
//
// THE TERMS ARE THE ENTRY PRODUCTS `|M_ab| |x_b|`, NOT THE FINISHED BLOCK
// PRODUCT `|(M x)_a|`. The 3x3 product is itself an fp32 sum, and so is every
// stored entry, so both round at the scale of the entry terms. Taking the
// magnitude after the product spends the product's own cancellation first,
// and that cancellation is largest exactly where it matters: a vertex pressed
// between two surfaces carries a contact stiffness far above its other terms
// along a normal on no coordinate axis, so its block's entries are all of that
// stiffness and a direction across the normal makes them cancel. Measured on
// public issue #160's pinch, one such row's products cancel from 5.3e14 to
// 7.1e9; a scale built from the finished products gave a threshold of 2.3e5
// against a computed curvature of -3.55e6, and the run aborted as "not SPD" on
// a matrix whose only negative eigenvalue, -2.0e6, is rounding of entries of
// 3e13. The entry scale bounds what the stored entries' rounding can do to the
// curvature, and is never smaller than the finished-product scale, so no
// curvature that truncated before is fatal now. A genuine defect, a Rayleigh
// quotient of order one, still clears it by orders.
//
// THE CANCELLATION BETWEEN BLOCKS IS STILL CAPTURED, as in the reference
// (`solver.cu:826-890`): magnitudes accumulate componentwise in `abssum` and
// are contracted against `|x_row|` once, at the end.
//

// `matrix` as `mat3_mul` reads it, column-major: `(M x)_a = sum_b m[3b+a] x_b`.
[[seam::device_fn]] inline void spmv_accumulate_entry_magnitude(
    Vec3f &abssum, const float *matrix, const float *vector) {
    for (unsigned a = 0; a < 3; ++a) {
        abssum[a] += fmath::abs(matrix[a]) * fmath::abs(vector[0]) +
                     fmath::abs(matrix[3 + a]) * fmath::abs(vector[1]) +
                     fmath::abs(matrix[6 + a]) * fmath::abs(vector[2]);
    }
}

// `matrix` as `mat3_transpose_mul` reads it: `(M^T x)_a = sum_b m[3a+b] x_b`.
[[seam::device_fn]] inline void spmv_accumulate_entry_magnitude_transpose(
    Vec3f &abssum, const float *matrix, const float *vector) {
    for (unsigned a = 0; a < 3; ++a) {
        abssum[a] += fmath::abs(matrix[3 * a]) * fmath::abs(vector[0]) +
                     fmath::abs(matrix[3 * a + 1]) * fmath::abs(vector[1]) +
                     fmath::abs(matrix[3 * a + 2]) * fmath::abs(vector[2]);
    }
}

// The same two for a vector held by value (the difference form).
[[seam::device_fn]] inline void spmv_accumulate_entry_magnitude_vector(
    Vec3f &abssum, const float *matrix, const Vec3f &vector) {
    for (unsigned a = 0; a < 3; ++a) {
        abssum[a] += fmath::abs(matrix[a]) * fmath::abs(vector[0]) +
                     fmath::abs(matrix[3 + a]) * fmath::abs(vector[1]) +
                     fmath::abs(matrix[6 + a]) * fmath::abs(vector[2]);
    }
}

[[seam::device_fn]] inline void spmv_accumulate_entry_magnitude_transpose_vector(
    Vec3f &abssum, const float *matrix, const Vec3f &vector) {
    for (unsigned a = 0; a < 3; ++a) {
        abssum[a] += fmath::abs(matrix[3 * a]) * fmath::abs(vector[0]) +
                     fmath::abs(matrix[3 * a + 1]) * fmath::abs(vector[1]) +
                     fmath::abs(matrix[3 * a + 2]) * fmath::abs(vector[2]);
    }
}

// The scalar shift `D x_i` rides in the diagonal term; its terms are `|D| |x_k|`.
[[seam::device_fn]] inline void spmv_accumulate_shift_magnitude(
    Vec3f &abssum, float shift, const float *vector) {
    for (unsigned k = 0; k < 3; ++k) {
        abssum[k] += fmath::abs(shift) * fmath::abs(vector[k]);
    }
}

// THE SAME ENTRY SCALE FOR A BLOCK CONTRACTED ON BOTH SIDES, as the symmetric
// apply forms it: `sum_a |left_a| sum_b |M_ab| |right_b|`, `matrix` read as
// `mat3_mul` reads it. That apply takes one scalar `left . (M right)` per block,
// and its magnitude, like the finished product's, has already spent the
// cancellation inside the block.
[[seam::device_fn]] inline float spmv_entry_quadratic_magnitude(
    const float *matrix, const float *left, const float *right) {
    Vec3f abssum = Vec3f::Zero();
    spmv_accumulate_entry_magnitude(abssum, matrix, right);
    float magnitude = 0.0f;
    for (unsigned a = 0; a < 3; ++a) {
        magnitude += fmath::abs(left[a]) * abssum[a];
    }
    return magnitude;
}

// THE ROW'S CURVATURE TERM AND ITS ROUND-OFF BOUND, formed once from the
// finished row. `solver.cu` runs exactly this loop in its head lane.
[[seam::device_fn]] inline void spmv_row_bound(
    const Vec3f &sum, const Vec3f &abssum,
    const float *x, unsigned row,
    float &signed_sum,
    float &absolute) {
    for (unsigned k = 0; k < 3; ++k) {
        signed_sum += x[3 * row + k] * sum[k];
        absolute += fmath::abs(x[3 * row + k]) * abssum[k];
    }
}

// THE ROW'S FIXED-PATTERN PRODUCT, and the magnitude sum beside it.
//
// EACH OFF-DIAGONAL BLOCK IS VISITED TWICE HERE AND ONCE IN THE REFERENCE, and
// the two totals agree exactly. The reference walks the upper triangle and
// doubles: its term for `(i, j)` with `i != j` is `2 * x_i . (H x_j)`. This
// walks the direct run from row `i` and the transpose run from row `j`, whose
// scalars are `x_i . (H x_j)` and `x_j . (H^T x_i)`, and those are the same
// number because both are `x_i^T H x_j`. Two visits of magnitude `|t|` sum to
// `2|t|`, which is `|2t|`.
[[seam::device_fn]] inline Vec3f fixed_csr_apply_row(
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *x, unsigned row,
    Vec3f &abssum) {
    Vec3f result = Vec3f::Zero();
    // NO LOCAL POINTER FOR `x + 3 * row`. MSL requires an address space on
    // every pointer TYPE, and a local declaration has none to inherit; passing
    // the expression to a `[[seam::device]]` parameter carries it instead.
    for (unsigned slot = offset[row]; slot < offset[row + 1]; ++slot) {
        const Vec3f block = mat3_mul(value + 9 * slot, x + 3 * index[slot]);
        result += block;
        spmv_accumulate_entry_magnitude(abssum, value + 9 * slot, x + 3 * index[slot]);
    }
    for (unsigned entry = transpose_offset[row];
         entry < transpose_offset[row + 1]; ++entry) {
        const unsigned source_row = transpose_pair[2 * entry];
        const unsigned slot = transpose_pair[2 * entry + 1];
        const Vec3f block = mat3_transpose_mul(value + 9 * slot,
                                                   x + 3 * source_row);
        result += block;
        spmv_accumulate_entry_magnitude_transpose(abssum, value + 9 * slot,
                                                   x + 3 * source_row);
    }
    return result;
}

[[seam::device_fn]] inline Vec3f dynamic_csr_apply_row(
    const unsigned *direct_index,
    const float *direct_value, unsigned direct_count,
    const unsigned *reference_index,
    const unsigned *reference_value, unsigned reference_count,
    const float *global_value,
    const float *x, Vec3f &abssum) {
    Vec3f result = Vec3f::Zero();
    for (unsigned entry = 0; entry < direct_count; ++entry) {
        const Vec3f block = mat3_mul(direct_value + 9 * entry,
                                         x + 3 * direct_index[entry]);
        result += block;
        spmv_accumulate_entry_magnitude(abssum, direct_value + 9 * entry,
                                         x + 3 * direct_index[entry]);
    }
    for (unsigned entry = 0; entry < reference_count; ++entry) {
        const Vec3f block =
            mat3_transpose_mul(global_value + 9 * reference_value[entry],
                                   x + 3 * reference_index[entry]);
        result += block;
        spmv_accumulate_entry_magnitude_transpose(abssum, global_value + 9 * reference_value[entry],
                                   x + 3 * reference_index[entry]);
    }
    return result;
}

// One row of the Newton operator, `result_i = ((A + B + C) x)_i + D x_i`.
//
// THE ORDER IS THE POINT, and it is why this composition is a body rather than
// three calls in a driver. The dynamic (contact) row, the fixed-pattern row,
// the diagonal block and the scalar shift accumulate in that order. An fp32 sum
// is not associative, so composing the four in a driver would give a different
// last bit on every matvec, and PCG amplifies that over its iterations into a
// different search direction. There is no tolerance for that: the backends are
// supposed to be one implementation.
//
// ONE THING HERE IS NOT YET THE REFERENCE'S, AND SAYING SO IS THE POINT OF THIS
// PARAGRAPH. The reference (`csrmat.cu`) carries a SINGLE running sum across
// all four of its loops, giving `((((0 + d1) + d2) + f1) + f2) + C`. This
// composition folds the fixed half separately: `fixed_csr_apply_row` starts its
// own accumulator at zero and the finished sub-total is added, giving
// `((0 + d1) + d2) + (f1 + f2) + C`. Those agree exactly whenever the fixed run
// contributes fewer than two blocks, or the dynamic partial is zero, which is
// every row of a contact-free scene; they differ in the last bit on a row
// carrying BOTH a contact block and two or more fixed blocks.
//
// It is left as it is deliberately rather than overlooked. The movement is
// inside the run-to-run band the atomic assembly already imposes, and no
// cross-backend comparison can be a byte comparison anyway, since the frontend
// permutes edge ids per process. Closing it means a seeded
// `fixed_csr_accumulate_row(seed, ...)` with `fixed_csr_apply_row` restated as
// a one-line call passing a zero, which is a change to a neutral
// body and therefore owes a `--features cuda-abi` build and a
// displacement-judged scene sweep. Worth taking opportunistically when someone
// is already running those gates; not worth a commit of its own.
//
// THE DYNAMIC HALF IS CALLED EVEN WHEN THE SCENE CARRIES NO CONTACT MATRIX,
// with both counts zero. `dynamic_csr_apply_row` then returns the zero
// vector, adding it is exact, and what runs is literally this composition
// rather than a reduced form that would have to be re-derived when contact
// lands.
//
// `diagonal_block` is the row's OWN 3x3 block rather than the whole array, so a
// caller holding a bounds-checked container resolves the row itself and keeps
// its check; a caller holding a flat array passes `diagonal + 9 * row`.
//
// `diagonal_shift` is tested for truth, not compared against zero, exactly as
// the CUDA driver has always tested it: a shift of zero skips the tail rather
// than adding three exact zeros, and adding them would be exact anyway, so the
// branch is preserved for identity rather than for speed.
// THE B, C AND D TERMS ON A PARTIAL SUM, split out so they are STATED ONCE.
// `spmv_apply_row` below hands it the A term, and the generated entry point for
// a scene with no dynamic matrix hands it a zero. ON THE ZERO PATH the split
// changes no value, adding a zero vector being exact. On the other path it
// changes the last bit of a row carrying two or more fixed blocks, for the
// association reason stated above `spmv_apply_row`; the terms are in the
// reference's order either way, and it is only the bracketing that differs.
// THE SAME TWO WALKS, STRIDED ACROSS THE LANES THAT SHARE A ROW.
//
// `solver.cu`'s `cg_apply_dot_walk_kernel` splits a row over `CG_ROW_LANES`
// lanes and strides each list by that count, which is what these reproduce:
// `start` is the lane's index within its row group and `stride` the group's
// width. At `start = 0, stride = 1` they are the unstrided walks above, entry
// for entry and in the same order.
//
// THE ASSOCIATION CHANGES AND THAT IS THE POINT. A strided walk sums a
// different subset per lane and the lane reduction adds those subsets, so the
// float total is not bit-identical to the serial one. Rule (1-LANE) is explicit
// that a parity test between a cooperative body and its twin must not demand
// bit-identity, and a caller whose correctness depends on the association takes
// its bound from the body that computed the sum.
[[seam::device_fn]] inline Vec3f dynamic_csr_apply_row_strided(
    const unsigned *direct_index,
    const float *direct_value, unsigned direct_count,
    const unsigned *reference_index,
    const unsigned *reference_value, unsigned reference_count,
    const float *global_value,
    const float *x, unsigned start, unsigned stride, Vec3f &abssum) {
    Vec3f result = Vec3f::Zero();
    for (unsigned entry = start; entry < direct_count; entry += stride) {
        const Vec3f block = mat3_mul(direct_value + 9 * entry,
                                         x + 3 * direct_index[entry]);
        result += block;
        spmv_accumulate_entry_magnitude(abssum, direct_value + 9 * entry,
                                         x + 3 * direct_index[entry]);
    }
    for (unsigned entry = start; entry < reference_count; entry += stride) {
        const Vec3f block =
            mat3_transpose_mul(global_value + 9 * reference_value[entry],
                                   x + 3 * reference_index[entry]);
        result += block;
        spmv_accumulate_entry_magnitude_transpose(abssum, global_value + 9 * reference_value[entry],
                                   x + 3 * reference_index[entry]);
    }
    return result;
}

[[seam::device_fn]] inline Vec3f fixed_csr_apply_row_strided(
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *x, unsigned row, unsigned start, unsigned stride,
    Vec3f &abssum) {
    Vec3f result = Vec3f::Zero();
    for (unsigned slot = offset[row] + start; slot < offset[row + 1];
         slot += stride) {
        const Vec3f block = mat3_mul(value + 9 * slot, x + 3 * index[slot]);
        result += block;
        spmv_accumulate_entry_magnitude(abssum, value + 9 * slot, x + 3 * index[slot]);
    }
    for (unsigned entry = transpose_offset[row] + start;
         entry < transpose_offset[row + 1]; entry += stride) {
        const unsigned source_row = transpose_pair[2 * entry];
        const unsigned slot = transpose_pair[2 * entry + 1];
        const Vec3f block = mat3_transpose_mul(value + 9 * slot,
                                                   x + 3 * source_row);
        result += block;
        spmv_accumulate_entry_magnitude_transpose(abssum, value + 9 * slot,
                                                   x + 3 * source_row);
    }
    return result;
}

[[seam::device_fn]] inline Vec3f spmv_apply_row_finish(
    const Vec3f &partial,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal_block, float diagonal_shift,
    const float *x, unsigned row,
    Vec3f &abssum) {
    Vec3f sum = partial;
    sum += fixed_csr_apply_row(index, offset, value, transpose_pair,
                                   transpose_offset, x, row, abssum);
    // THE DIAGONAL IS ONE TERM, as it is in the reference, which forms
    // `v = C[i] * xi` and folds `|v[c]|` into `abssum` once
    // (`solver.cu:884-890`). The shift rides in the same vector rather than
    // becoming a second term: it is part of the same diagonal block, and it is
    // zero on every path that reads this bound.
    Vec3f diagonal = mat3_mul(diagonal_block, x + 3 * row);
    if (diagonal_shift) {
        for (unsigned k = 0; k < 3; ++k) {
            diagonal[k] += diagonal_shift * x[3 * row + k];
        }
    }
    sum += diagonal;
    spmv_accumulate_entry_magnitude(abssum, diagonal_block, x + 3 * row);
    spmv_accumulate_shift_magnitude(abssum, diagonal_shift, x + 3 * row);
    return sum;
}

[[seam::device_fn]] inline Vec3f spmv_apply_row(
    const unsigned *direct_index,
    const float *direct_value, unsigned direct_count,
    const unsigned *reference_index,
    const unsigned *reference_value, unsigned reference_count,
    const float *global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal_block, float diagonal_shift,
    const float *x, unsigned row,
    Vec3f &abssum) {
    const Vec3f partial = dynamic_csr_apply_row(
        direct_index, direct_value, direct_count, reference_index,
        reference_value, reference_count, global_value, x, abssum);
    return spmv_apply_row_finish(partial, index, offset, value,
                                     transpose_pair, transpose_offset,
                                     diagonal_block, diagonal_shift, x, row,
                                     abssum);
}

// THE NEWTON OPERATOR AS TWO ENTRY POINTS, and the pair exists for the reason
// sections 35 to 37 give: whether the scene carries a DYNAMIC (contact) matrix
// is a driver configuration, all seven of its buffers come from one `Option`,
// and a record field is a handle with no spelling for absent.
//
// NEITHER FORM RESTATES THE COMPOSITION. The dynamic one calls
// `spmv_apply_row` unchanged, which is what CUDA's own dispatch calls; the
// plain one calls the finish above with a zero partial, which is what the
// dynamic half returns when a scene has no contact matrix. The comment on that
// body warns against "a reduced form that would have to be re-derived when
// contact lands", and this is not one: there is a single statement of A and a
// single statement of B, C and D, and the pair only chooses whether to evaluate
// the first.
[[seam::entry(row)]]
[[seam::device_fn]] inline void operator_apply(
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal, float diagonal_shift,
    const float *x, float *result,
    float *absolute, float *curvature,
    unsigned row) {
    const Vec3f zero = Vec3f::Zero();
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    Vec3f abssum = Vec3f::Zero();
    const Vec3f row_sum = spmv_apply_row_finish(
        zero, index, offset, value, transpose_pair, transpose_offset,
        diagonal + 9 * row, diagonal_shift, x, row, abssum);
    map<Vec3f>(result + 3 * row) = row_sum;
    spmv_row_bound(row_sum, abssum, x, row, signed_sum, magnitude);
    absolute[row] = magnitude;
    curvature[row] = signed_sum;
}

[[seam::entry(row)]]
[[seam::device_fn]] inline void operator_apply_dynamic(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal, float diagonal_shift,
    const float *x, float *result,
    float *absolute, float *curvature,
    unsigned row) {
    const unsigned direct_begin = dyn_offset[row];
    const unsigned reference_begin = dyn_reference_offset[row];
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    Vec3f abssum = Vec3f::Zero();
    const Vec3f row_sum = spmv_apply_row(
        dyn_index + direct_begin, dyn_value + 9 * direct_begin,
        dyn_offset[row + 1] - direct_begin,
        dyn_reference_index + reference_begin,
        dyn_reference_value + reference_begin,
        dyn_reference_offset[row + 1] - reference_begin, dyn_global_value,
        index, offset, value, transpose_pair, transpose_offset,
        diagonal + 9 * row, diagonal_shift, x, row, abssum);
    map<Vec3f>(result + 3 * row) = row_sum;
    spmv_row_bound(row_sum, abssum, x, row, signed_sum, magnitude);
    absolute[row] = magnitude;
    curvature[row] = signed_sum;
}

// THE OPERATOR APPLY WITH ITS TWO SCALARS FOLDED IN, which is the shape
// `cg_apply_dot_walk_kernel` has in the reference: one pass forms the row
// product, the curvature that row contributes and the magnitude bounding it,
// and the GROUP reduces both before anything leaves the kernel.
//
// WHY THIS EXISTS BESIDE THE ELEMENT FORM. The element form writes one float
// per ROW into two full-length arrays, so the fold that follows starts from
// `nrow` and walks a fixed-width tree: measured on `bench_drape`, 81,920 rows
// fold 320, then 2, then 1, which is three dispatches per iteration for each of
// two chains. Folding inside the apply makes the fold's input one float per
// GROUP, and 320 floats are one dispatch. The reference issues five kernels per
// iteration where this tree issued 12.1, and this is the first of them.
//
// ONE ROW PER LANE, so the parallelism is exactly the element form's and the
// group index is the only thing added. A lane whose row is past the end still
// reaches both folds with a zero contribution, because `compute::block_sum` is
// a group-wide operation and a lane that returned early would leave the others
// waiting on a partial that never arrives.
//
// THE TWO FOLDS TAKE SEPARATE SCRATCH HALVES, for the reason
// `vec_block_sum_pair_cooperative` states: `compute::block_sum` stores into the
// scratch and reads it back, a second call over the same slots could store
// before every lane of the first has read, and a neutral body has no barrier to
// put between them.
[[seam::device_fn]] inline void operator_apply_folded(
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal, float diagonal_shift,
    const float *x, float *result,
    float *scratch,
    float *curvature_total,
    float *absolute_total, const float *latch, unsigned rows,
    unsigned lane, unsigned threads, unsigned block) {
    // A FROZEN PCG ITERATION (pcg.rs, `latch`) skips the product. Every lane
    // reads the same flag, so the group returns whole.
    if (latch[PCG_LATCH_FROZEN] != 0.0f) {
        return;
    }
    const unsigned row = block * threads + lane;
    // THE FOLD'S LANE-MAJOR SLOT (`pcg_fold_slot`) for this group's partials,
    // formed here, where its two divisions overlap the loads, rather than
    // in the writer lane after the reduction, where every group would wait
    // on them.
    const unsigned fold_slot = pcg_fold_slot(block, (rows + threads - 1u) / threads);
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    if (row < rows) {
        const Vec3f zero = Vec3f::Zero();
        Vec3f abssum = Vec3f::Zero();
        const Vec3f row_sum = spmv_apply_row_finish(
            zero, index, offset, value, transpose_pair, transpose_offset,
            diagonal + 9 * row, diagonal_shift, x, row, abssum);
        map<Vec3f>(result + 3 * row) = row_sum;
        spmv_row_bound(row_sum, abssum, x, row, signed_sum, magnitude);
    }
    const float curvature_folded =
        compute::block_sum(signed_sum, scratch, lane, threads);
    const float absolute_folded =
        compute::block_sum(magnitude, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        curvature_total[fold_slot] = curvature_folded;
        absolute_total[fold_slot] = absolute_folded;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void operator_apply_folded(
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal,
    float diagonal_shift,
    const float *x,
    float *result,
    [[seam::scratch(64)]] float *scratch,
    float *curvature_total,
    float *absolute_total,
    const float *latch,
    unsigned rows,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block,
    unsigned count);

// The dynamic half of the pair above, which evaluates the contact matrix as
// well. The two exist for the reason the element pair does: whether a scene
// carries a DYNAMIC matrix is a driver configuration and a record field has no
// spelling for absent.
// EIGHT LANES PER ROW, which is what `solver.cu` does and this tree did not.
//
// `cg_apply_dot_walk_kernel` sets `CG_ROW_LANES = 8` and walks a row with eight
// lanes; the body below walked it with ONE THREAD. Measured on `trapped`, this
// kernel was 28.87 s against the reference's 23.31 s over 2 percent fewer
// launches, which is 61 percent of the whole GPU gap between the two trees.
//
// AN `enum` RATHER THAN A `constexpr`, because MSL requires a program-scope
// constexpr variable to live in the constant address space and rejects it
// otherwise, which no other backend's compiler would have told us.
enum : unsigned { SPMV_ROW_LANES = 8 };

// THE COOPERATIVE HALF, under rule (1-LANE). The device arms run it; the host
// arm runs the serial twin below, which computes the same bits.
[[seam::device_fn]] [[seam::cooperative]] inline void operator_apply_dynamic_folded(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal, float diagonal_shift,
    const float *x, float *result,
    float *scratch,
    float *curvature_total,
    float *absolute_total, const float *latch, unsigned rows,
    unsigned lane, unsigned threads, unsigned block) {
    // A FROZEN PCG ITERATION (pcg.rs, `latch`) skips the product. Every lane
    // reads the same flag, so the group returns whole.
    if (latch[PCG_LATCH_FROZEN] != 0.0f) {
        return;
    }
    const unsigned row = (block * threads + lane) / SPMV_ROW_LANES;
    // THE FOLD'S LANE-MAJOR SLOT (`pcg_fold_slot`) for this group's partials,
    // formed here, where its two divisions overlap the loads, rather than
    // in the writer lane after the reduction, where every group would wait
    // on them.
    const unsigned fold_slot = pcg_fold_slot(block, (rows * SPMV_ROW_LANES + threads - 1u) / threads);
    const unsigned row_lane = lane % SPMV_ROW_LANES;
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    Vec3f row_sum = Vec3f::Zero();
    Vec3f abssum = Vec3f::Zero();
    if (row < rows) {
        const unsigned direct_begin = dyn_offset[row];
        const unsigned reference_begin = dyn_reference_offset[row];
        row_sum = dynamic_csr_apply_row_strided(
            dyn_index + direct_begin, dyn_value + 9 * direct_begin,
            dyn_offset[row + 1] - direct_begin,
            dyn_reference_index + reference_begin,
            dyn_reference_value + reference_begin,
            dyn_reference_offset[row + 1] - reference_begin, dyn_global_value,
            x, row_lane, SPMV_ROW_LANES, abssum);
        row_sum += fixed_csr_apply_row_strided(
            index, offset, value, transpose_pair, transpose_offset, x, row,
            row_lane, SPMV_ROW_LANES, abssum);
        // THE DIAGONAL IS ONE BLOCK, so it belongs to one lane rather than to
        // the stride. Added before the reduction so it reaches the total and
        // the magnitude bound exactly as `spmv_apply_row_finish` adds it.
        if (row_lane == 0) {
            Vec3f diagonal_term = mat3_mul(diagonal + 9 * row, x + 3 * row);
            if (diagonal_shift) {
                for (unsigned k = 0; k < 3; ++k) {
                    diagonal_term[k] += diagonal_shift * x[3 * row + k];
                }
            }
            row_sum += diagonal_term;
            spmv_accumulate_entry_magnitude(abssum, diagonal + 9 * row,
                                            x + 3 * row);
            spmv_accumulate_shift_magnitude(abssum, diagonal_shift,
                                            x + 3 * row);
        }
    }
    // THE ROW'S LANES FOLD INTO THEIR FIRST ONE. The offsets stay inside the
    // group without a width argument: lane 0 reads lane 4, then lane 2, then
    // lane 1, and every one of those is in the same group of eight.
    // THE ROW'S LANES FOLD, through the seam rather than a hand-written ladder.
    // `compute::lane_reduce_add` is the bounded fold `seam_host.h` documents
    // beside `block_sum`: each arm supplies the association its hardware gives,
    // and the two device arms agree bit-for-bit at the writer lane, which is
    // what keeps the curvature round-off bound calibrated across them.
    //
    // A SLOT PER VALUE, because one lane folds all six before the next lane
    // runs and they would otherwise share an accumulator on the host arm.
    for (unsigned k = 0; k < 3; ++k) {
        row_sum[k] = compute::lane_reduce_add(row_sum[k], scratch + 64 + k,
                                              lane, SPMV_ROW_LANES);
        abssum[k] = compute::lane_reduce_add(abssum[k], scratch + 67 + k, lane,
                                             SPMV_ROW_LANES);
    }
    // THE WRITER LANE IS A NAME, NOT A LITERAL. It is lane 0 of the run on both
    // device arms and the LAST lane on the host, because that arm folds in
    // ascending order. The singleton diagonal above stays on `row_lane == 0`
    // instead: it must be a lane whose value ENTERS the fold, which is the
    // first lane on every arm, and those two lanes are not the same one.
    if (row < rows && compute::is_lane_writer(row_lane, SPMV_ROW_LANES)) {
        map<Vec3f>(result + 3 * row) = row_sum;
        spmv_row_bound(row_sum, abssum, x, row, signed_sum, magnitude);
    }
    const float curvature_folded =
        compute::block_sum(signed_sum, scratch, lane, threads);
    const float absolute_folded =
        compute::block_sum(magnitude, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        curvature_total[fold_slot] = curvature_folded;
        absolute_total[fold_slot] = absolute_folded;
    }
}

// THE SERIAL TWIN, which the host arm runs. It evaluates the same sums as the
// cooperative body above does on the host arm, in the same association, with
// the eight lanes of a row walked by one call instead of eight.
//
// WHY IT EXISTS. The host shim runs a group's lanes one after another, so the
// body above costs eight calls per row, each setting up four strided walks
// that visit about one entry apiece and then folding six values through the
// group scratch. Measured on `bench_drape` (Apple M1, 8 threads) that made this
// kernel 45 percent of all host kernel time, at 2.5 times the cost per row of
// the element form `operator_apply_dynamic`, which walks the same entries.
//
// WHY NO SUM IS REASSOCIATED. On the host arm, lane `l` of a row sums the
// entries whose position in each run is `l` modulo `SPMV_ROW_LANES`, in
// ascending order, into its own accumulator, and `lane_reduce_add` then folds
// the lanes in ascending order. This walk keeps one accumulator per lane and
// visits every entry once, ascending, adding it into accumulator `position %
// SPMV_ROW_LANES`. Each accumulator therefore receives the same terms in the
// same order as that lane did, from the same starting zero, through the same
// helpers, and the lanes are then folded in the same ascending order. The
// group fold below is the cooperative body's call for call: every lane still
// hands `block_sum` its value, zero on the lanes that write no row.
//
// WHAT CAN STILL MOVE IS THE LAST BIT, AND ONLY THROUGH THE COMPILER'S FUSED
// MULTIPLY-ADD. The host build contracts (`-ffp-contract=fast`, matching the
// device compilers), and which product of a 3x3 row gets fused is decided per
// inlining context. Measured with both bodies on random rows: bit-identical
// with contraction off, last-bit differences with it on. Every such choice
// evaluates the same tree of rounded sums, so both are held to the one
// round-off bound the curvature guard takes its scale from.
//
[[seam::device_fn]] [[seam::serial]] inline void operator_apply_dynamic_folded(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal, float diagonal_shift,
    const float *x, float *result,
    float *scratch,
    float *curvature_total,
    float *absolute_total, const float *latch, unsigned rows,
    unsigned lane, unsigned threads, unsigned block) {
    if (latch[PCG_LATCH_FROZEN] != 0.0f) {
        return;
    }
    const unsigned row = (block * threads + lane) / SPMV_ROW_LANES;
    const unsigned row_lane = lane % SPMV_ROW_LANES;
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    // THE ROW IS FORMED BY ITS WRITER LANE, the one the cooperative body writes
    // from on this arm. The other lanes contribute zero to the group fold, as
    // they do there.
    //
    // EACH 3-VECTOR IS HELD IN FOUR FLOAT LANES (`compute::lane4`), component
    // `a` in lane `a`: a block's product and its entry magnitude are three
    // lanewise multiply-adds rather than nine scalar ones, and every lane
    // performs the scalar form's operations on the same operands in the same
    // order, so the values are the scalar twin's bit for bit. Measured on a
    // drape-shaped matrix (81,920 rows, 8 threads) against the scalar twin:
    // 3.32 -> 2.66 ms on an EPYC 7R13 and 2.18 -> 1.91 ms on an Apple M2.
    //
    if (row < rows && compute::is_lane_writer(row_lane, SPMV_ROW_LANES)) {
        // The dynamic and fixed partials of each lane, kept apart because the
        // cooperative body adds the fixed one to the dynamic one only after
        // both walks, and the magnitude accumulated in place across both.
        compute::lane4 dynamic_part[SPMV_ROW_LANES];
        compute::lane4 fixed_part[SPMV_ROW_LANES];
        compute::lane4 lane_abssum[SPMV_ROW_LANES];
        for (unsigned l = 0; l < SPMV_ROW_LANES; ++l) {
            dynamic_part[l] = compute::lane4_splat(0.0f);
            fixed_part[l] = compute::lane4_splat(0.0f);
            lane_abssum[l] = compute::lane4_splat(0.0f);
        }
        compute::lane4 c0, c1, c2;
        const unsigned direct_begin = dyn_offset[row];
        const unsigned direct_count = dyn_offset[row + 1] - direct_begin;
        const unsigned reference_begin = dyn_reference_offset[row];
        const unsigned reference_count =
            dyn_reference_offset[row + 1] - reference_begin;
        for (unsigned entry = 0; entry < direct_count; ++entry) {
            const unsigned l = entry % SPMV_ROW_LANES;
            const unsigned at = direct_begin + entry;
            const float *right = x + 3 * dyn_index[at];
            compute::lane4_block_columns(dyn_value + 9 * at, c0, c1, c2);
            dynamic_part[l] = dynamic_part[l] +
                              compute::lane4_block_product(c0, c1, c2, right);
            lane_abssum[l] = lane_abssum[l] +
                             compute::lane4_block_magnitude(c0, c1, c2, right);
        }
        for (unsigned entry = 0; entry < reference_count; ++entry) {
            const unsigned l = entry % SPMV_ROW_LANES;
            const unsigned at = reference_begin + entry;
            const float *right = x + 3 * dyn_reference_index[at];
            compute::lane4_block_columns_transposed(
                dyn_global_value + 9 * dyn_reference_value[at], c0, c1, c2);
            dynamic_part[l] = dynamic_part[l] +
                              compute::lane4_block_product(c0, c1, c2, right);
            lane_abssum[l] = lane_abssum[l] +
                             compute::lane4_block_magnitude(c0, c1, c2, right);
        }
        const unsigned fixed_begin = offset[row];
        const unsigned fixed_count = offset[row + 1] - fixed_begin;
        for (unsigned entry = 0; entry < fixed_count; ++entry) {
            const unsigned l = entry % SPMV_ROW_LANES;
            const unsigned slot = fixed_begin + entry;
            const float *right = x + 3 * index[slot];
            compute::lane4_block_columns(value + 9 * slot, c0, c1, c2);
            fixed_part[l] = fixed_part[l] +
                            compute::lane4_block_product(c0, c1, c2, right);
            lane_abssum[l] = lane_abssum[l] +
                             compute::lane4_block_magnitude(c0, c1, c2, right);
        }
        const unsigned transpose_begin = transpose_offset[row];
        const unsigned transpose_count =
            transpose_offset[row + 1] - transpose_begin;
        for (unsigned entry = 0; entry < transpose_count; ++entry) {
            const unsigned l = entry % SPMV_ROW_LANES;
            const unsigned at = transpose_begin + entry;
            const float *right = x + 3 * transpose_pair[2 * at];
            const unsigned slot = transpose_pair[2 * at + 1];
            compute::lane4_block_columns_transposed(value + 9 * slot, c0, c1,
                                                    c2);
            fixed_part[l] = fixed_part[l] +
                            compute::lane4_block_product(c0, c1, c2, right);
            lane_abssum[l] = lane_abssum[l] +
                             compute::lane4_block_magnitude(c0, c1, c2, right);
        }
        // Each lane's partial, finished as the cooperative body finishes it:
        // the fixed walk added to the dynamic one, and the diagonal on lane 0.
        compute::lane4 row_sum = compute::lane4_splat(0.0f);
        compute::lane4 abssum = compute::lane4_splat(0.0f);
        for (unsigned l = 0; l < SPMV_ROW_LANES; ++l) {
            compute::lane4 lane_sum = dynamic_part[l] + fixed_part[l];
            if (l == 0) {
                const float *own = x + 3 * row;
                compute::lane4_block_columns(diagonal + 9 * row, c0, c1, c2);
                compute::lane4 diagonal_term =
                    compute::lane4_block_product(c0, c1, c2, own);
                if (diagonal_shift) {
                    diagonal_term = diagonal_term +
                                    compute::lane4_splat(diagonal_shift) *
                                        compute::lane4_load3(own);
                }
                lane_sum = lane_sum + diagonal_term;
                lane_abssum[0] =
                    lane_abssum[0] +
                    compute::lane4_block_magnitude(c0, c1, c2, own);
                // `spmv_accumulate_shift_magnitude`: `|D| |x_k|` per lane.
                lane_abssum[0] =
                    lane_abssum[0] +
                    compute::lane4_abs(compute::lane4_splat(diagonal_shift)) *
                        compute::lane4_abs(compute::lane4_load3(own));
            }
            // THE LANE FOLD, as `lane_reduce_add` spells it on this arm: the
            // first lane stores and every later one adds, ascending.
            row_sum = l == 0 ? lane_sum : row_sum + lane_sum;
            abssum = l == 0 ? lane_abssum[l] : abssum + lane_abssum[l];
        }
        Vec3f row_total;
        for (unsigned k = 0; k < 3; ++k) {
            row_total[k] = row_sum[k];
        }
        map<Vec3f>(result + 3 * row) = row_total;
        // THE ROW'S CURVATURE TERMS, `spmv_row_bound`'s sums in its order, in
        // the form that keeps the shipped host build's bits (see
        // `compute::lane4_row_bound`).
        compute::lane4_row_bound(x + 3 * row, row_sum, abssum, signed_sum,
                                 magnitude);
    }
    const float curvature_folded =
        compute::block_sum(signed_sum, scratch, lane, threads);
    const float absolute_folded =
        compute::block_sum(magnitude, scratch + 32, lane, threads);
    // The cooperative body's lane-major slot (`pcg_fold_slot`), so the fold
    // that reads these partials finds them where it looks on every arm.
    if (compute::is_block_writer(lane, threads)) {
        const unsigned fold_slot = pcg_fold_slot(
            block, (rows * SPMV_ROW_LANES + threads - 1u) / threads);
        curvature_total[fold_slot] = curvature_folded;
        absolute_total[fold_slot] = absolute_folded;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void
operator_apply_dynamic_folded(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal,
    float diagonal_shift,
    const float *x,
    float *result,
    [[seam::scratch(96)]] float *scratch,
    float *curvature_total,
    float *absolute_total,
    const float *latch,
    unsigned rows,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block,
    unsigned count);

// THE SYMMETRIC APPLY, `solver.cu`'s `cg_apply_dot_sym_kernel`.
//
// The walk above reads every block TWICE, once from the row that stores it and
// once from the mirror list that names it for the other row. This reads each
// stored block ONCE and scatters both ways: `H p_j` into this row's sum and
// `H^T p_i` atomically into the other row's slot. `solver.cu` measures the
// crossover at `CG_SYM_SPMV_MIN_ROWS = 100000` rows, walk being cheaper below it
// where the mirror reads are L2-served and the atomic traffic dominates instead.
//
// **THE CALLER MUST ZERO `result` FIRST.** Every row's output is accumulated
// atomically, this row's own sum included, so nothing here overwrites a slot and
// a stale value would survive the pass.
//
// THE SUMMATION ORDER IS NOT FIXED, and that is the same tolerance this tree
// already accepts everywhere it deposits with `compute::atomic_add`: the contact
// assembly, the CSR push and the element force scatters are all atomic, so the
// solve is already order-independent rather than order-fixed. `solver.cu` says
// the same of this kernel in its own words.
[[seam::device_fn]] inline void
operator_apply_symmetric_folded(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const float *diagonal, float diagonal_shift,
    const float *x,
    compute::atomic_float_t *result,
    float *scratch,
    float *curvature_total,
    float *absolute_total, const float *latch, unsigned rows,
    unsigned lane, unsigned threads, unsigned block) {
    // A FROZEN PCG ITERATION (pcg.rs, `latch`) skips the product. Every lane
    // reads the same flag, so the group returns whole.
    if (latch[PCG_LATCH_FROZEN] != 0.0f) {
        return;
    }
    const unsigned row = (block * threads + lane) / SPMV_ROW_LANES;
    // THE FOLD'S LANE-MAJOR SLOT (`pcg_fold_slot`) for this group's partials,
    // formed here, where its two divisions overlap the loads, rather than
    // in the writer lane after the reduction, where every group would wait
    // on them.
    const unsigned fold_slot = pcg_fold_slot(block, (rows * SPMV_ROW_LANES + threads - 1u) / threads);
    const unsigned row_lane = lane % SPMV_ROW_LANES;
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    Vec3f row_sum = Vec3f::Zero();
    if (row < rows) {
        const float *own = x + 3 * row;
        // THE DYNAMIC MATRIX'S DIRECT LIST ONLY. The walk above reads this row's
        // own blocks AND the mirror list that names the blocks other rows store
        // for it; this reads each stored block ONCE and sends its transpose to
        // the other row instead.
        for (unsigned entry = dyn_offset[row] + row_lane;
             entry < dyn_offset[row + 1]; entry += SPMV_ROW_LANES) {
            const unsigned column = dyn_index[entry];
            const Vec3f product = mat3_mul(dyn_value + 9 * entry, x + 3 * column);
            row_sum += product;
            float term = 0.0f;
            for (unsigned k = 0; k < 3; ++k) {
                term += own[k] * product[k];
            }
            if (column != row) {
                const Vec3f mirror =
                    mat3_transpose_mul(dyn_value + 9 * entry, own);
                for (unsigned k = 0; k < 3; ++k) {
                    compute::atomic_add(result + 3 * column + k, mirror[k]);
                }
                // EACH OFF-DIAGONAL BLOCK CONTRIBUTES TWICE to the quadratic
                // form, once for (row, column) and once for its transpose, and
                // the second contribution is not otherwise visited.
                term += term;
            }
            signed_sum += term;
            magnitude += (column != row ? 2.0f : 1.0f) *
                         spmv_entry_quadratic_magnitude(dyn_value + 9 * entry,
                                                        own, x + 3 * column);
        }
        // THE FIXED MATRIX'S STORED LIST ONLY, for the same reason: its
        // transpose table is what the two-triangle walk reads and this scatters
        // instead.
        for (unsigned slot = offset[row] + row_lane; slot < offset[row + 1];
             slot += SPMV_ROW_LANES) {
            const unsigned column = index[slot];
            const Vec3f product = mat3_mul(value + 9 * slot, x + 3 * column);
            row_sum += product;
            float term = 0.0f;
            for (unsigned k = 0; k < 3; ++k) {
                term += own[k] * product[k];
            }
            if (column != row) {
                const Vec3f mirror = mat3_transpose_mul(value + 9 * slot, own);
                for (unsigned k = 0; k < 3; ++k) {
                    compute::atomic_add(result + 3 * column + k, mirror[k]);
                }
                term += term;
            }
            signed_sum += term;
            magnitude += (column != row ? 2.0f : 1.0f) *
                         spmv_entry_quadratic_magnitude(value + 9 * slot, own,
                                                        x + 3 * column);
        }
    }
    // The same bounded fold the walk uses; see the note there.
    for (unsigned k = 0; k < 3; ++k) {
        row_sum[k] = compute::lane_reduce_add(row_sum[k], scratch + 64 + k,
                                              lane, SPMV_ROW_LANES);
    }
    if (row < rows && compute::is_lane_writer(row_lane, SPMV_ROW_LANES)) {
        Vec3f diagonal_term = mat3_mul(diagonal + 9 * row, x + 3 * row);
        if (diagonal_shift) {
            for (unsigned k = 0; k < 3; ++k) {
                diagonal_term[k] += diagonal_shift * x[3 * row + k];
            }
        }
        row_sum += diagonal_term;
        float term = 0.0f;
        for (unsigned k = 0; k < 3; ++k) {
            term += x[3 * row + k] * diagonal_term[k];
        }
        signed_sum += term;
        magnitude += spmv_entry_quadratic_magnitude(diagonal + 9 * row,
                                                    x + 3 * row, x + 3 * row);
        for (unsigned k = 0; k < 3; ++k) {
            magnitude += fmath::abs(diagonal_shift) * x[3 * row + k] *
                         x[3 * row + k];
        }
        // THE ROW'S OWN ACCUMULATION IS ATOMIC TOO, because the transpose
        // scatters above target these same slots from other rows.
        for (unsigned k = 0; k < 3; ++k) {
            compute::atomic_add(result + 3 * row + k, row_sum[k]);
        }
    }
    const float curvature_folded =
        compute::block_sum(signed_sum, scratch, lane, threads);
    const float absolute_folded =
        compute::block_sum(magnitude, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        curvature_total[fold_slot] = curvature_folded;
        absolute_total[fold_slot] = absolute_folded;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void
operator_apply_symmetric_folded(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const float *diagonal,
    float diagonal_shift,
    const float *x,
    compute::atomic_float_t *result,
    [[seam::scratch(96)]] float *scratch,
    float *curvature_total,
    float *absolute_total,
    const float *latch,
    unsigned rows,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block,
    unsigned count);


// ---------------------------------------------------------------------------
// THE DIFFERENCE FORM: `(A x)_i = S_i x_i + sum_j A_ij (x_j - x_i)`, with
// `S_i = sum_j A_ij` the row's blocks summed once per Newton step.
//
// IT IS THE SAME PRODUCT, AND IT EXISTS FOR A VECTOR THE PLAIN FORM CANNOT
// RESOLVE. An elastic row sums to nearly nothing against a vector that varies
// slowly across it: every block multiplies a value of the vector's full size
// and the row's answer is what is left after they cancel, so in single precision
// the answer carries round-off of the size of the terms rather than of itself.
// A block-diagonal preconditioner never hands the conjugate gradient such a
// vector, each of its directions being rough at the scale of one row. A
// multigrid's coarse correction is exactly such a vector, a whole piece of mesh
// moving together, and measured on a solid beam the curvature `p . A p` of its
// first direction is 5e-6 of the magnitudes summed to form it, below the 3e-5 a
// curvature has to clear to be believed. Here each block multiplies the
// DIFFERENCE of two neighbors, which is small where the vector is smooth, and
// the same curvature is 2e-3 of its own magnitudes.
//
// THE MAGNITUDE SUM IS THIS FORM'S OWN, block by block as the plain form takes
// its own, so the bound the curvature is judged against is the round-off of the
// sum that was actually computed.
//
// `S_i` IS COMPUTED BY `operator_row_sum` AND NOT HERE, because it is the same
// for every vector the operator is applied to in a solve. Its own round-off is
// a fixed perturbation of the row's diagonal block of the order of single
// precision's epsilon times the row's blocks, which changes the system being
// solved by that much and no iteration by anything.

[[seam::device_fn]] inline Vec3f
mat3_mul_vector(const float *matrix, const Vec3f &vector) {
    Vec3f result;
    result[0] =
        matrix[0] * vector[0] + matrix[3] * vector[1] + matrix[6] * vector[2];
    result[1] =
        matrix[1] * vector[0] + matrix[4] * vector[1] + matrix[7] * vector[2];
    result[2] =
        matrix[2] * vector[0] + matrix[5] * vector[1] + matrix[8] * vector[2];
    return result;
}

[[seam::device_fn]] inline Vec3f
mat3_transpose_mul_vector(const float *matrix, const Vec3f &vector) {
    Vec3f result;
    result[0] =
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
    result[1] =
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
    result[2] =
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
    return result;
}

// `x_other - own`, the one vector every block of a row multiplies.
[[seam::device_fn]] inline Vec3f
spmv_difference(const float *x, unsigned other, const Vec3f &own) {
    Vec3f difference;
    difference[0] = x[3 * other + 0] - own[0];
    difference[1] = x[3 * other + 1] - own[1];
    difference[2] = x[3 * other + 2] - own[2];
    return difference;
}

// One row of the operator in the difference form, and its magnitude sum.
//
// THE FOUR SPANS IN THE PLAIN FORM'S ORDER, contact first, then the row's own
// term last where the plain form adds the block diagonal. A block whose column
// is the row itself multiplies a zero difference, so the diagonal blocks the
// two matrices store need no test: their whole contribution is inside `S_i`.
[[seam::device_fn]] inline Vec3f spmv_difference_row(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *row_sum,
    const float *x, unsigned row,
    Vec3f &abssum) {
    Vec3f own;
    own[0] = x[3 * row + 0];
    own[1] = x[3 * row + 1];
    own[2] = x[3 * row + 2];
    Vec3f result = Vec3f::Zero();
    for (unsigned slot = dyn_offset[row]; slot < dyn_offset[row + 1]; ++slot) {
        const Vec3f block = mat3_mul_vector(
            dyn_value + 9 * slot, spmv_difference(x, dyn_index[slot], own));
        result += block;
        spmv_accumulate_entry_magnitude_vector(abssum, 
            dyn_value + 9 * slot, spmv_difference(x, dyn_index[slot], own));
    }
    for (unsigned slot = dyn_reference_offset[row];
         slot < dyn_reference_offset[row + 1]; ++slot) {
        const Vec3f block = mat3_transpose_mul_vector(
            dyn_global_value + 9 * dyn_reference_value[slot],
            spmv_difference(x, dyn_reference_index[slot], own));
        result += block;
        spmv_accumulate_entry_magnitude_transpose_vector(abssum, 
            dyn_global_value + 9 * dyn_reference_value[slot],
            spmv_difference(x, dyn_reference_index[slot], own));
    }
    for (unsigned slot = offset[row]; slot < offset[row + 1]; ++slot) {
        const Vec3f block = mat3_mul_vector(
            value + 9 * slot, spmv_difference(x, index[slot], own));
        result += block;
        spmv_accumulate_entry_magnitude_vector(abssum, 
            value + 9 * slot, spmv_difference(x, index[slot], own));
    }
    for (unsigned entry = transpose_offset[row];
         entry < transpose_offset[row + 1]; ++entry) {
        const Vec3f block = mat3_transpose_mul_vector(
            value + 9 * transpose_pair[2 * entry + 1],
            spmv_difference(x, transpose_pair[2 * entry], own));
        result += block;
        spmv_accumulate_entry_magnitude_transpose_vector(abssum, 
            value + 9 * transpose_pair[2 * entry + 1],
            spmv_difference(x, transpose_pair[2 * entry], own));
    }
    const Vec3f diagonal = mat3_mul_vector(row_sum + 9 * row, own);
    result += diagonal;
    spmv_accumulate_entry_magnitude_vector(abssum, row_sum + 9 * row, own);
    return result;
}

// `S_i`: the block diagonal plus every block the row's four spans store, the
// transposed ones transposed.
[[seam::entry(row)]]
[[seam::device_fn]] inline void operator_row_sum(
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal,
    float *row_sum,
    unsigned row) {
    float sum[9];
    for (unsigned entry = 0; entry < 9; ++entry) {
        sum[entry] = diagonal[9 * row + entry];
    }
    for (unsigned slot = dyn_offset[row]; slot < dyn_offset[row + 1]; ++slot) {
        for (unsigned entry = 0; entry < 9; ++entry) {
            sum[entry] += dyn_value[9 * slot + entry];
        }
    }
    for (unsigned slot = dyn_reference_offset[row];
         slot < dyn_reference_offset[row + 1]; ++slot) {
        const unsigned source = dyn_reference_value[slot];
        for (unsigned r = 0; r < 3; ++r) {
            for (unsigned c = 0; c < 3; ++c) {
                sum[3 * c + r] += dyn_global_value[9 * source + 3 * r + c];
            }
        }
    }
    for (unsigned slot = offset[row]; slot < offset[row + 1]; ++slot) {
        for (unsigned entry = 0; entry < 9; ++entry) {
            sum[entry] += value[9 * slot + entry];
        }
    }
    for (unsigned entry = transpose_offset[row];
         entry < transpose_offset[row + 1]; ++entry) {
        const unsigned source = transpose_pair[2 * entry + 1];
        for (unsigned r = 0; r < 3; ++r) {
            for (unsigned c = 0; c < 3; ++c) {
                sum[3 * c + r] += value[9 * source + 3 * r + c];
            }
        }
    }
    for (unsigned entry = 0; entry < 9; ++entry) {
        row_sum[9 * row + entry] = sum[entry];
    }
}

// The difference form with one float per row for each of the two scalars, as
// `operator_apply` writes them.
[[seam::entry(row)]]
[[seam::device_fn]] inline void operator_apply_difference(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *row_sum,
    const float *x, float *result,
    float *absolute, float *curvature,
    unsigned row) {
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    Vec3f abssum = Vec3f::Zero();
    const Vec3f product = spmv_difference_row(
        dyn_index, dyn_value, dyn_offset, dyn_reference_index,
        dyn_reference_value, dyn_reference_offset, dyn_global_value, index,
        offset, value, transpose_pair, transpose_offset, row_sum, x, row,
        abssum);
    map<Vec3f>(result + 3 * row) = product;
    spmv_row_bound(product, abssum, x, row, signed_sum, magnitude);
    absolute[row] = magnitude;
    curvature[row] = signed_sum;
}

// The same with the two scalars folded per group, one row a lane, which is the
// shape `operator_apply_folded` has and for the reasons stated there.
[[seam::device_fn]] inline void operator_apply_difference_folded(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *row_sum,
    const float *x, float *result,
    float *scratch,
    float *curvature_total,
    float *absolute_total, const float *latch, unsigned rows,
    unsigned lane, unsigned threads, unsigned block) {
    // A FROZEN PCG ITERATION (pcg.rs, `latch`) skips the product. Every lane
    // reads the same flag, so the group returns whole.
    if (latch[PCG_LATCH_FROZEN] != 0.0f) {
        return;
    }
    const unsigned row = block * threads + lane;
    // THE FOLD'S LANE-MAJOR SLOT (`pcg_fold_slot`) for this group's partials,
    // formed here, where its two divisions overlap the loads, rather than
    // in the writer lane after the reduction, where every group would wait
    // on them.
    const unsigned fold_slot = pcg_fold_slot(block, (rows + threads - 1u) / threads);
    float magnitude = 0.0f;
    float signed_sum = 0.0f;
    if (row < rows) {
        Vec3f abssum = Vec3f::Zero();
        const Vec3f product = spmv_difference_row(
            dyn_index, dyn_value, dyn_offset, dyn_reference_index,
            dyn_reference_value, dyn_reference_offset, dyn_global_value, index,
            offset, value, transpose_pair, transpose_offset, row_sum, x, row,
            abssum);
        map<Vec3f>(result + 3 * row) = product;
        spmv_row_bound(product, abssum, x, row, signed_sum, magnitude);
    }
    const float curvature_folded =
        compute::block_sum(signed_sum, scratch, lane, threads);
    const float absolute_folded =
        compute::block_sum(magnitude, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        curvature_total[fold_slot] = curvature_folded;
        absolute_total[fold_slot] = absolute_folded;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void
operator_apply_difference_folded(
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *row_sum,
    const float *x,
    float *result,
    [[seam::scratch(64)]] float *scratch,
    float *curvature_total,
    float *absolute_total,
    const float *latch,
    unsigned rows,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block,
    unsigned count);

// The entry point for the block-diagonal apply, declared once and rendered for
// four targets: the `__global__` nvcc compiles, the `kernel void` the Metal
// shader compiler compiles, the `mat3_mul_entry` shim a host C++
// compiler compiles, and the Rust `#[repr(C)]` twin the driver fills. It is
// what a block-Jacobi preconditioner apply IS, one 3x3 block times one vector
// per row, so the entry names this body rather than the pass that dispatches
// it.
//
// THE TWO STRIDES ARE THE ROW'S OWN GEOMETRY, not a tuning choice: a row holds
// one 3x3 block, nine floats, against three components of the vector. Writing
// them here is what lets the entry hand the body `matrix + 9 * row` and
// `vector + 3 * row`, which is the arithmetic the launcher this replaces
// spelled, rather than the body doing its own addressing.
//
// The result is written through `[[seam::scatter]]` as one `Vec3f` element,
// whose 12 bytes every C++ rendering asserts. That is the same three floats at
// the same three addresses the launcher stored component by component: a
// `Vec3f` is exactly its elements, which is what the assertion pins.
[[seam::entry(count)]] void mat3_mul(
    [[seam::stride(9)]] const float *matrix,
    [[seam::stride(3)]] const float *vector,
    Vec3f *result,
    unsigned count);

// The whole-row apply's entry point, declared once and rendered for four
// targets. Six BASE pointers, the forwarded thread index and one element
// scatter: the body walks the row's own slot range and its transposed
// couplings, so the addressing is its own and only the result is per element.
//
// THE STORE IS ONE `Vec3f` AT THE ROW, which is the same twelve bytes at the
// same address the launcher wrote as three floats at `3 * row + k`, and the
// pod assertion is what pins that in every C++ rendering. Declaring the record
// here is also what removes a hand-written mirror PAIR: the shim used to take
// its own `FixedSpmvArgs` struct of raw pointers and the driver its own
// `FixedSpmvArgs` beside it, two declarations that could disagree with nothing
// linking them.
// THE ROW PRODUCT ALONE, for a caller that wants `A x` and not the magnitude
// sum beside it.
//
// A BODY OF ITS OWN RATHER THAN AN OVERLOAD. C++ would resolve two
// `fixed_csr_apply_row` declarations on arity and the build would be green, but
// `check-shared-wiring.py` keys a neutral body on its NAME and cannot tell two
// rows apart; the new function therefore takes the new name, the existing one
// keeping its callers.
[[seam::entry(row, result)]]
[[seam::device_fn]] inline Vec3f fixed_csr_product_row(
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *x, unsigned row) {
    Vec3f discarded_magnitude = Vec3f::Zero();
    return fixed_csr_apply_row(index, offset, value, transpose_pair,
                                   transpose_offset, x, row,
                                   discarded_magnitude);
}
