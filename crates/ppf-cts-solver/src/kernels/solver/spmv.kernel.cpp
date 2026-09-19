// File: spmv.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

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

// THE MAGNITUDE OF ONE BLOCK'S CONTRIBUTION, ACCUMULATED COMPONENTWISE.
//
// `p^T A p` is a sum of roughly `2 * nnz` SIGNED terms, so when the true
// curvature is small beside the individual terms the fp32 result is dominated
// by round-off. A bound on that error is what lets a caller tell a canceled
// zero from a genuinely negative curvature.
//
// THE BOUND IS THE REFERENCE'S, COMPONENT BY COMPONENT, and this is the shape
// rather than a scalar for a reason worth stating, because the scalar form is
// the one that looks right. `solver.cu:826-890` carries a `Vec3f abssum`,
// accumulates `abssum[c] += |v[c]|` as each block is multiplied, and forms
// `absdot = sum_k |p_k| * abssum_k` ONCE, at the end, beside
// `dot = sum_k p_k * sum_k`. The cancellation between blocks is still captured,
// because the magnitudes are still taken per block; what is deferred is only
// the contraction against the row vector.
//
// Accumulating `|p_i . v|` per block instead gives a STRICTLY TIGHTER number,
// since `|sum_k p_k v_k| <= sum_k |p_k| |v_k|`, and a tighter bound is not a
// better one here: it is a DIFFERENT bound, so the `pAp <= 0` guard classifies
// against a different threshold than the reference does and the two solvers
// disagree about which curvatures are resolvable. That is the second design
// rule (1a) exists to prevent, so the reference's form is what this computes.
[[seam::device_fn]] inline void spmv_accumulate_magnitude(
    Vec3f &abssum, const Vec3f &block) {
    for (unsigned k = 0; k < 3; ++k) {
        abssum[k] += fmath::abs(block[k]);
    }
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
        spmv_accumulate_magnitude(abssum, block);
    }
    for (unsigned entry = transpose_offset[row];
         entry < transpose_offset[row + 1]; ++entry) {
        const unsigned source_row = transpose_pair[2 * entry];
        const unsigned slot = transpose_pair[2 * entry + 1];
        const Vec3f block = mat3_transpose_mul(value + 9 * slot,
                                                   x + 3 * source_row);
        result += block;
        spmv_accumulate_magnitude(abssum, block);
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
        spmv_accumulate_magnitude(abssum, block);
    }
    for (unsigned entry = 0; entry < reference_count; ++entry) {
        const Vec3f block =
            mat3_transpose_mul(global_value + 9 * reference_value[entry],
                                   x + 3 * reference_index[entry]);
        result += block;
        spmv_accumulate_magnitude(abssum, block);
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
        spmv_accumulate_magnitude(abssum, block);
    }
    for (unsigned entry = start; entry < reference_count; entry += stride) {
        const Vec3f block =
            mat3_transpose_mul(global_value + 9 * reference_value[entry],
                                   x + 3 * reference_index[entry]);
        result += block;
        spmv_accumulate_magnitude(abssum, block);
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
        spmv_accumulate_magnitude(abssum, block);
    }
    for (unsigned entry = transpose_offset[row] + start;
         entry < transpose_offset[row + 1]; entry += stride) {
        const unsigned source_row = transpose_pair[2 * entry];
        const unsigned slot = transpose_pair[2 * entry + 1];
        const Vec3f block = mat3_transpose_mul(value + 9 * slot,
                                                   x + 3 * source_row);
        result += block;
        spmv_accumulate_magnitude(abssum, block);
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
    spmv_accumulate_magnitude(abssum, diagonal);
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
    float *absolute_total, unsigned rows, unsigned lane,
    unsigned threads, unsigned block) {
    const unsigned row = block * threads + lane;
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
        curvature_total[block] = curvature_folded;
        absolute_total[block] = absolute_folded;
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

// THE COOPERATIVE HALF, under rule (1-LANE). It is a twin rather than the only
// body because `compute::shuffle_down` has no single-thread meaning and
// `seam_host.h` deliberately leaves it undefined, so a serial target cannot
// compile this one and gets the walk below instead.
[[seam::device_fn]] inline void operator_apply_dynamic_folded(
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
    float *absolute_total, unsigned rows, unsigned lane,
    unsigned threads, unsigned block) {
    const unsigned row = (block * threads + lane) / SPMV_ROW_LANES;
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
            spmv_accumulate_magnitude(abssum, diagonal_term);
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
        curvature_total[block] = curvature_folded;
        absolute_total[block] = absolute_folded;
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
    float *absolute_total, unsigned rows, unsigned lane,
    unsigned threads, unsigned block) {
    const unsigned row = (block * threads + lane) / SPMV_ROW_LANES;
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
            magnitude += fmath::abs(term);
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
            magnitude += fmath::abs(term);
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
        magnitude += fmath::abs(term);
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
        curvature_total[block] = curvature_folded;
        absolute_total[block] = absolute_folded;
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
