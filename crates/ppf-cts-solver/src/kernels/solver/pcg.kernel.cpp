// File: pcg.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::device]]` is the address space of a pointer, which MSL
// requires on every reference and pointer type.
//
// The division goes through `fmath::div` rather than the operator, because MSL
// spells the correctly rounded quotient `precise::divide` and the plain
// operator is a different function there.

#include "spmv.kernel.cpp"

enum : int {
    PCG_BREAK_NONE = 0,
    PCG_BREAK_PAP = 1,
    PCG_BREAK_RZ = 2,
    PCG_BREAK_NOISE = 3
};

struct PcgScalar {
    float value;
    float noise;
    int cause;
};

struct PcgUpdate3 {
    float rz;
    float error;
};

// The three-way verdict on a curvature reading, as an enumerator rather than a
// program-scope constant: MSL rejects a variable outside the constant address
// space, and an enum with a fixed base carries an integer in a form all three
// compilers accept.
enum : int {
    // Resolvable positive curvature: alpha = rz / pAp.
    PCG_CURVATURE_OK = 0,
    // Within its own round-off: the sign is not resolvable in fp32, so the
    // direction carries no information and the solve stops here
    // (Steihaug-style) without aborting.
    PCG_CURVATURE_TRUNCATE = 1,
    // Negative BEYOND the bound, or NaN: a real assembly regression, not fp32
    // cancellation.
    PCG_CURVATURE_FATAL = 2
};

// Round-off floor of the reduction that produced a `p^T A p`, given the scale
// of that reduction.
//
// 256 is the slack and 1.19209290e-7f is float epsilon. The slack covers the
// GPU reduction's shape (a per-thread serial accumulation, a 256-wide shared
// tree and a strided fold, each contribution carrying its own epsilon), and at
// 256 * eps = 3.05e-5 relative it sits orders below the order-1 Rayleigh
// quotient a genuine assembly error produces. It is a round-off bound and NOT
// a tolerance on the guard: a curvature more negative than this still aborts.
//
// WHICH SCALE IS PASSED IS THE CALLER'S BUSINESS AND DIFFERS BY PATH. A device
// kernel that reduces `sum|contribution|` alongside the dot passes that sum. A
// host-synchronizing loop whose SpMV produces no such sum passes the
// Cauchy-Schwarz surrogate |p|_2 |Ap|_2 >= sum|p_i (Ap)_i| instead, which errs
// toward truncating rather than aborting. Only the scale is path-specific; the
// bound taken from it, and the verdict below, are not.
//
// FLOAT, NOT TEMPLATED, and deliberately so: the same rule runs on the GPU,
// where float64 is banned, and a host-and-device template instantiated with
// double by a host caller would emit exactly the float64 device instantiation
// that ban exists to prevent. Float is also right on the merits: every input is
// reduced from an fp32 operator, and the verdict is a sign test against a
// 256*eps slack rather than a quantity read to the last ulp. A host loop keeps
// double for its own arithmetic and narrows only at this call.
[[seam::host_device_fn]] inline float
pcg_curvature_noise(float absolute_dot) {
    return 256.0f * 1.19209290e-7f * absolute_dot;
}

// The verdict itself, in one place, so a curvature reading is classified
// identically wherever it is taken.
//
// A NaN fails every comparison, so it falls through the positive test and must
// be named here rather than being reported as a negative curvature or silently
// truncating. The test is `fmath::isnan` and never the `x != x` idiom: that
// idiom is deleted outright under Metal's fast math, where `isnan` still works.
[[seam::host_device_fn]] inline int pcg_curvature_verdict(float p_ap,
                                                             float bound) {
    if (p_ap > bound) {
        return PCG_CURVATURE_OK;
    }
    if (p_ap < -bound || fmath::isnan(p_ap)) {
        return PCG_CURVATURE_FATAL;
    }
    return PCG_CURVATURE_TRUNCATE;
}

[[seam::device_fn]] inline PcgScalar pcg_alpha(float rz, float p_ap,
                                                     float absolute_dot) {
    PcgScalar result;
    result.noise = pcg_curvature_noise(absolute_dot);
    const int verdict = pcg_curvature_verdict(p_ap, result.noise);
    if (verdict == PCG_CURVATURE_OK) {
        result.value = fmath::div(rz, p_ap);
        result.cause = PCG_BREAK_NONE;
    } else {
        result.value = 0.0f;
        result.cause = (verdict == PCG_CURVATURE_FATAL)
                           ? PCG_BREAK_PAP
                           : PCG_BREAK_NOISE;
    }
    return result;
}

[[seam::device_fn]] inline PcgScalar pcg_beta(float rz_next,
                                                    float rz_previous) {
    PcgScalar result;
    result.noise = 0.0f;
    if (rz_previous > 0.0f) {
        result.value = fmath::div(rz_next, rz_previous);
        result.cause = PCG_BREAK_NONE;
    } else {
        result.value = 0.0f;
        result.cause = PCG_BREAK_RZ;
    }

    return result;
}

[[seam::device_fn]] inline PcgUpdate3 pcg_update3(
    const float *p, const float *ap,
    float *x, float *r,
    float *z, const float *inverse_diagonal,
    unsigned row, float alpha) {
    float residual[3];
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        const unsigned index = 3 * row + dimension;
        x[index] += alpha * p[index];
        residual[dimension] = r[index] - alpha * ap[index];
        r[index] = residual[dimension];
    }
    // The attribute leads the declaration because C++ admits an attribute only
    // at the head of a declaration or at the end of the decl-specifier
    // sequence, never between `const` and the type it qualifies. MSL reads
    // `device const float *` and `const device float *` as the same type.
    const float *matrix = inverse_diagonal + 9 * row;
    Vec3f preconditioned;
    preconditioned[0] =
        matrix[0] * residual[0] + matrix[3] * residual[1] +
        matrix[6] * residual[2];
    preconditioned[1] =
        matrix[1] * residual[0] + matrix[4] * residual[1] +
        matrix[7] * residual[2];
    preconditioned[2] =
        matrix[2] * residual[0] + matrix[5] * residual[1] +
        matrix[8] * residual[2];
    PcgUpdate3 result{0.0f, 0.0f};
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        z[3 * row + dimension] = preconditioned[dimension];
        result.rz += residual[dimension] * preconditioned[dimension];
        result.error += residual[dimension] < 0.0f ? -residual[dimension]
                                                  : residual[dimension];
    }
    return result;
}

// One row's contribution to a vector inner product, and the magnitude of that
// same contribution.
//
// TWO NUMBERS FROM ONE EXPRESSION, WHICH IS THE POINT. The curvature verdict in
// `pcg_alpha` compares `p^T A p` against `256 * eps * absolute_dot`, and
// that comparison is only meaningful when the bound was measured over the SAME
// contributions the sum was taken over. Returning them separately from two
// passes would let a caller pair a sum with a bound from a different
// decomposition, and the bound would then be a number about some other fold.
//
// The three components are folded in ascending order, matching
// `pcg_update3`'s `rz` accumulation above, so a backend that computes `rz`
// through one and `p^T A p` through the other gets one association order rather
// than two.
struct PcgDotTerm {
    float product;
    float magnitude;
};

[[seam::device_fn]] inline PcgDotTerm
pcg_dot_term(const float *a, const float *b,
                 unsigned row) {
    PcgDotTerm term;
    term.product = 0.0f;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        const unsigned index = 3 * row + dimension;
        term.product += a[index] * b[index];
    }
    term.magnitude = fmath::abs(term.product);
    return term;
}

// THE SAME TWO NUMBERS, HANDED BACK THROUGH REFERENCES, so an entry point can
// reach them. A `[[seam::scatter]]` carries ONE return value and this pass
// produces two, which is exactly the shape a composition exists for: the pair
// above stays the definition and this only unpacks it, so the two numbers are
// still measured over the same fold and no caller can pair a sum with a bound
// from another decomposition.
//
// BOTH VECTORS STAY BASE POINTERS because the body addresses its own triple at
// `3 * row`, so the thread index is forwarded rather than spent on a gather;
// that is the same arrangement `dx_magnitude` takes over the same layout.
[[seam::entry(row)]]
[[seam::device_fn]] inline void
pcg_dot_terms(const float *a,
                  const float *b, unsigned row,
                  float &product,
                  float &absolute_product) {
    const PcgDotTerm term = pcg_dot_term(a, b, row);
    product = term.product;
    absolute_product = term.magnitude;
}

// THE ITERATE, THE RESIDUAL, THE PRECONDITIONED RESIDUAL AND THE `r . z` TERMS,
// in one pass over the row. It is `cg_fused_update_kernel`'s shape, which does
// the same four things over one pass and is what holds a PCG iteration to four
// dispatches rather than one per update.
//
// THE ORDER INSIDE THE ROW IS THE POINT AND IS NOT AN OPTIMIZATION. `z` is the
// preconditioner applied to the residual AFTER this step's update, and the
// terms are that same updated residual against that `z`, so the three cannot be
// reordered and a caller cannot pair an `r` from one iteration with a `z` from
// another. Splitting them into separate dispatches, which is what this
// replaces, made that ordering a property of the CALL SITE rather than of the
// arithmetic.
//
// EVERY VALUE IS THE ONE THE SPLIT PASSES PRODUCED, bit for bit: the same
// `fma` for the iterate and the residual, the same column-major `mat3_mul` for
// the preconditioner, and the same per-row accumulation and absolute value for
// the terms. Only the number of dispatches changes.
//
// A ROW IS THREE COMPONENTS, so this is dispatched over ROWS while the vectors
// it walks are `3 * rows` floats, and each output is a base pointer the body
// indexes rather than a `[[seam::scatter]]`, because a body returns one value
// and this produces five.
[[seam::entry(row)]]
[[seam::device_fn]] inline void pcg_update_row(
    const float *direction,
    const float *product_direction,
    const float *alpha,
    const float *inverse_diagonal,
    float *iterate, float *residual,
    float *preconditioned,
    float *term_product,
    float *term_magnitude, unsigned row) {
    const float scale = *alpha;
    float updated[3];
    for (unsigned c = 0; c < 3u; ++c) {
        const unsigned i = 3u * row + c;
        iterate[i] = fmath::fma(scale, direction[i], iterate[i]);
        updated[c] = fmath::fma(-scale, product_direction[i], residual[i]);
        residual[i] = updated[c];
    }
    // THE PRECONDITIONER, over the residual THIS ROW JUST WROTE. Reading it
    // back out of `residual` would be the same value; taking it from the local
    // says so, and costs no load.
    //
    // THE BLOCK IS INDEXED, NOT ALIASED BY A LOCAL POINTER. MSL requires an
    // explicit address space on every pointer type, so `const float *block =
    // inverse_diagonal + 9 * row` is "pointer type must have explicit address
    // space qualifier" there while nvcc and a host compiler take it without a
    // word. Indexing the parameter needs no address space of its own and is
    // the same three loads.
    const unsigned block = 9u * row;
    float applied[3];
    applied[0] = inverse_diagonal[block + 0] * updated[0] +
                 inverse_diagonal[block + 3] * updated[1] +
                 inverse_diagonal[block + 6] * updated[2];
    applied[1] = inverse_diagonal[block + 1] * updated[0] +
                 inverse_diagonal[block + 4] * updated[1] +
                 inverse_diagonal[block + 7] * updated[2];
    applied[2] = inverse_diagonal[block + 2] * updated[0] +
                 inverse_diagonal[block + 5] * updated[1] +
                 inverse_diagonal[block + 8] * updated[2];
    float dot = 0.0f;
    for (unsigned c = 0; c < 3u; ++c) {
        preconditioned[3u * row + c] = applied[c];
        dot += updated[c] * applied[c];
    }
    term_product[row] = dot;
    term_magnitude[row] = fmath::abs(dot);
}

// THE FUSED ROW WITH ITS TWO SCALARS FOLDED IN: the pass that writes the
// iterate, the residual and the preconditioned residual also reduces `r . z`
// and `||r||_1`, and only the two per-GROUP totals leave the kernel.
//
// WHY THE ELEMENT FORM ABOVE IS NOT ENOUGH. It writes `r . z` per ROW and
// leaves the residual norm to a separate magnitude level over `3 * rows`
// floats, so the chain after it is an abs level plus a three-level dual fold:
// five dispatches per iteration. Both sums here are per group, so the two
// arrays have ONE length and a single group folds them together.
//
// `term_magnitude` HAS NO COUNTERPART HERE. The element form writes it and
// nothing reads it; the number this body needs beside `r . z` is the residual
// L1 norm, which is a sum over the row's THREE components rather than over the
// row's dot.
//
// ONE ROW PER LANE, and a lane past the end contributes zero to both folds
// while still reaching them, because `compute::block_sum` is group-wide.
[[seam::device_fn]] inline void pcg_update_row_folded(
    const float *direction,
    const float *product_direction,
    const float *alpha,
    const float *inverse_diagonal,
    float *iterate, float *residual,
    float *preconditioned,
    float *scratch,
    float *product_total,
    float *residual_total, unsigned rows, unsigned lane,
    unsigned threads, unsigned block) {
    const unsigned row = block * threads + lane;
    float dot = 0.0f;
    float norm = 0.0f;
    if (row < rows) {
        const float scale = *alpha;
        float updated[3];
        for (unsigned c = 0; c < 3u; ++c) {
            const unsigned i = 3u * row + c;
            iterate[i] = fmath::fma(scale, direction[i], iterate[i]);
            updated[c] = fmath::fma(-scale, product_direction[i], residual[i]);
            residual[i] = updated[c];
            norm += fmath::abs(updated[c]);
        }
        const unsigned diagonal_block = 9u * row;
        float applied[3];
        applied[0] = inverse_diagonal[diagonal_block + 0] * updated[0] +
                     inverse_diagonal[diagonal_block + 3] * updated[1] +
                     inverse_diagonal[diagonal_block + 6] * updated[2];
        applied[1] = inverse_diagonal[diagonal_block + 1] * updated[0] +
                     inverse_diagonal[diagonal_block + 4] * updated[1] +
                     inverse_diagonal[diagonal_block + 7] * updated[2];
        applied[2] = inverse_diagonal[diagonal_block + 2] * updated[0] +
                     inverse_diagonal[diagonal_block + 5] * updated[1] +
                     inverse_diagonal[diagonal_block + 8] * updated[2];
        for (unsigned c = 0; c < 3u; ++c) {
            preconditioned[3u * row + c] = applied[c];
            dot += updated[c] * applied[c];
        }
    }
    const float product_folded =
        compute::block_sum(dot, scratch, lane, threads);
    const float residual_folded =
        compute::block_sum(norm, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        product_total[block] = product_folded;
        residual_total[block] = residual_folded;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void pcg_update_row_folded(
    const float *direction,
    const float *product_direction,
    const float *alpha,
    const float *inverse_diagonal,
    float *iterate, float *residual,
    float *preconditioned,
    [[seam::scratch(64)]] float *scratch,
    float *product_total,
    float *residual_total,
    unsigned rows,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block,
    unsigned count);

// THE TWO PCG COEFFICIENTS, each unpacked out of the struct its body returns
// into three out-references, exactly as `pcg_dot_terms` above unpacks a
// pair. A scatter carries ONE return value and each of these produces three, so
// all three are non-const gathers and neither entry has a scatter.
//
// A ONE-ELEMENT DISPATCH, AND THAT IS WHAT IT IS rather than a launcher shape
// borrowed for convenience: the verdict is a scalar the shared body decides, so
// that a solve aborts under ONE rule on every backend. CUDA evaluates the same
// arithmetic per thread inside its fused update kernel. The caller passes
// `count` 1 and reads the three out-parameters back.
//
// `cause` IS AN `int`, NOT AN `unsigned`, and it stays that way: it is the
// body's own classification, the caller maps it to an outcome, and `int` is one
// of the three scalar types a record field may hold.
[[seam::device_fn]] inline void pcg_alpha_terms(
    float rz, float p_ap, float absolute_dot,
    float &value, float &noise,
    int &cause) {
    const PcgScalar result = pcg_alpha(rz, p_ap, absolute_dot);
    value = result.value;
    noise = result.noise;
    cause = result.cause;
}

[[seam::entry(count)]] void pcg_alpha_terms(
    float rz, float p_ap, float absolute_dot,
    float *value_out,
    float *noise_out,
    int *cause_out,
    unsigned count);

[[seam::device_fn]] inline void pcg_beta_terms(
    float rz_next, float rz_previous, float &value,
    float &noise, int &cause) {
    const PcgScalar result = pcg_beta(rz_next, rz_previous);
    value = result.value;
    noise = result.noise;
    cause = result.cause;
}

[[seam::entry(count)]] void pcg_beta_terms(
    float rz_next, float rz_previous,
    float *value_out,
    float *noise_out,
    int *cause_out,
    unsigned count);

// ---------------------------------------------------------------------------
// THE SAME TWO COEFFICIENTS, TAKEN FROM SCALARS THAT NEVER LEAVE THE DEVICE.
// ---------------------------------------------------------------------------
//
// THE RESIDENT LOOP KEEPS EVERY PER-ITERATION SCALAR ON THE DEVICE for a whole
// solve: `rz0`, `rz1`, `pAp`, `alpha`, `beta`, `err`, the latched breakdown
// cause, value and iteration, the iteration counter, and the scale the
// curvature bound is taken from (`driver/pcg.rs`, `mod slot` and `mod
// cause_slot`). Each is produced by one dispatch and consumed by the next, and
// the host reads one small probe on a stride. The two bodies above take their
// inputs as `float` PARAMETERS, which is right for a loop that has the numbers
// on the host already, and wrong for a recurrence whose previous dispatch wrote
// them into a device buffer: filling a record's `float` field means a transfer
// and a stall per coefficient.
//
// So these two are the same verdicts over device scalars. THE VERDICT ITSELF IS
// NOT RESTATED: both call the same `pcg_alpha` and `pcg_beta` the parameter
// forms call, so a solve is classified into "proceed", "truncate" and "not
// positive definite" by ONE rule however the numbers arrive.

// THE SCALE THE CURVATURE BOUND IS TAKEN FROM.
//
// IT IS `sum |term|` OVER THE ROUGHLY `2 * nnz` SIGNED CONTRIBUTIONS that add
// up to `p^T A p`. The operator accumulates it per row while it multiplies and
// the caller folds it (`driver/pcg.rs`, `slot::ABSDOT`), so it arrives here
// already summed.
//
// IT IS NOT THE CAUCHY-SCHWARZ SURROGATE `|p|_2 |Ap|_2`, and the difference is
// the whole point of taking it this way. `|Ap|_2` has already spent the
// operator's cancellation, so on a strongly cancelling operator the surrogate
// is far SMALLER: for `A = [[1,-1],[-1,1]]` and `p = (1,1)` the operator gives
// `Ap = 0` and the surrogate is zero, while the signed terms are individually
// of order one. A smaller scale is a tighter bound, and a tighter bound
// classifies a round-off zero as a genuine negative curvature, which aborts the
// solve at `PCG_BREAK_PAP` rather than truncating it Steihaug-style and
// continuing. Only a path whose apply produces no such sum falls back to the
// surrogate, which is the case `pcg_curvature_noise` above describes.
[[seam::device_fn]] inline float pcg_curvature_scale(float absolute_sum) {
    return absolute_sum;
}

// THE BREAKDOWN LATCH, WRITTEN IN THE KERNEL THAT OBSERVES THE BREAKDOWN, and
// it is what makes a STRIDED residual read safe rather than merely tidy.
//
// The host samples the residual on a schedule (`driver/pcg.rs`), so a check can
// run up to `RESID_CHECK_STRIDE` iterations after the iterate that broke. By
// then `p_ap` and `p` have both been overwritten, and reporting those values
// describes a DIFFERENT iterate. Worse, `cause_out` alone is rewritten every
// iteration, so a breakdown followed by three healthy iterations would be
// erased before the host ever looked: the solve would continue on a direction
// the guard had already rejected. That is the `pAp <= 0` family, which must
// abort loudly and never be lost.
//
// So the three latch slots are STICKY, exactly as `cg_device` keeps its
// `breakdown` flag: memset once before the loop and never cleared, written only
// on the FIRST breakdown so the diagnostics name the offending iterate rather
// than the most recent one. `value` and `noise` stay per-iteration, because the
// recurrence consumes them immediately.
//
// THE ITERATION INDEX IS A DEVICE COUNTER, NOT A HOST SCALAR
// (`driver/pcg.rs`, `cause_slot::ITERATION`, advanced by the alpha pass). A
// by-value index is the one argument that would differ between two iterations
// of an otherwise identical body, and an iteration whose arguments differ
// CANNOT BE RECORDED ONCE AND REPLAYED, so holding the count on the device is
// what lets ONE recorded iteration be replayed for the whole solve. It is also
// the shape rule (1a) asks for on its own terms, a declaration answering to
// what each backend actually computes rather than to what one host finds cheap
// to supply.
// THE SAME CLASSIFICATION, TAKING THE SCALARS IT JUDGES BY VALUE.
//
// A CALLER THAT JUST FOLDED THESE NUMBERS HOLDS THEM IN REGISTERS, and reading
// them back through a device pointer it wrote one line earlier is a dependent
// global round trip on the single lane that does this work, at the tail of a
// one-group kernel where nothing hides the latency. That round trip, rather
// than any shortage of lanes, is what a pointer-taking fold pair costs: 3.23 us
// per launch.
//
// `pcg_alpha_resident` keeps its name and its entry point and now calls this,
// so the classification, the sticky latch and the announcement are still
// stated ONCE.
[[seam::device_fn]] inline void pcg_alpha_from_values(
    float rz_value, float curvature, float absolute_value,
    int *iteration_counter,
    float &value, float &noise,
    int &cause, int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired) {
    // ADVANCED HERE AND READ HERE, in the one dispatch of the iteration that
    // needs the number, so no other pass has to agree about when it moves. The
    // count is one-based, as the host index it replaces was.
    //
    // THE NEW VALUE IS FORMED BEFORE THE STORE rather than read back after it.
    // One lane owns this slot, so the two spellings carry the same number, and
    // this one does not wait for its own write to land.
    const int iteration = iteration_counter[0] + 1;
    iteration_counter[0] = iteration;
    const PcgScalar result =
        pcg_alpha(rz_value, curvature, pcg_curvature_scale(absolute_value));
    value = result.value;
    noise = result.noise;
    cause = result.cause;
    if (result.cause != PCG_BREAK_NONE && break_cause[0] == PCG_BREAK_NONE) {
        break_cause[0] = result.cause;
        break_value[0] = curvature;
        break_iteration[0] = iteration;
    }
    // THE SAME LATCH, ANNOUNCED IN THE BUFFER THE HOST ALREADY READS.
    // `break_cause` and `break_iteration` are ints and live in their own
    // allocation, so learning whether a breakdown fired from them alone costs
    // a second readback every batch. This is one float beside `break_value` in
    // the scalar buffer the probe downloads anyway, so the host reads the int
    // channel only when there is something in it. It is written on EVERY
    // latching iteration rather than only the first, because it answers "is
    // there anything to read" and not "which iteration was first", which is
    // what the sticky slots above are for.
    if (result.cause != PCG_BREAK_NONE) {
        break_fired[0] = 1.0f;
    }
}

// THE POINTER-TAKING FORM, which is what the standalone entry dispatches and
// what a caller holding no registers wants.
[[seam::device_fn]] inline void pcg_alpha_resident(
    const float *rz, const float *p_ap,
    const float *absolute_sum,
    int *iteration_counter,
    float &value, float &noise,
    int &cause, int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired) {
    pcg_alpha_from_values(*rz, *p_ap, *absolute_sum, iteration_counter, value,
                          noise, cause, break_cause, break_value,
                          break_iteration, break_fired);
}

[[seam::entry(count)]] void pcg_alpha_resident(
    const float *rz, const float *p_ap,
    const float *absolute_sum,
    int *iteration_counter,
    float *value_out,
    float *noise_out,
    int *cause_out,
    int *break_cause, float *break_value,
    int *break_iteration,
    float *break_fired, unsigned count);

// BETA, AND THE ROLL THAT GOES WITH IT, in one dispatch: the lane that computes
// `beta = rz1 / rz0` then assigns `rz0 = rz1`, so that scalar has a single
// reader and writer and the read-then-write order is safe. Splitting them would
// put a second dispatch between a read and a write of one scalar for no gain.
//
// THE NON-POSITIVE `rz_next` TEST IS HERE AND NOT AT THE CALLER, because the
// caller no longer has the number: `rz_next` is a device scalar a fold just
// wrote. `pcg_beta` classifies a non-positive PREVIOUS `rz` as
// `PCG_BREAK_RZ`, which is the same fault one iteration later, so the two share
// the cause: the block-Jacobi diagonal is inverted through a floored symmetric
// eigendecomposition and every per-vertex term is positive by construction, so
// a non-positive `r^T M^-1 r` means a block is NaN or infinite, or the residual
// itself is not finite.
//
// The test is spelled as a POSITIVE comparison rather than `<= 0.0f` so a NaN
// fails it and is reported, which the negated form would let through.
// BETA LATCHES INTO THE SAME THREE SLOTS AS ALPHA, and it must: the strided
// read cannot see a `PCG_BREAK_RZ` that a later iteration overwrote either. The
// two share one latch because the host reports whichever cause fired FIRST, and
// a solve stops at the first breakdown of either kind.
// THE FOLD THAT ALSO TAKES THE VERDICT, so no dispatch of its own stands
// between a fold and the coefficient it feeds: the pass that writes the two
// scalars forms alpha from them in-thread. Beta is the same shape one step
// later, riding the same single group with the roll.
//
// ONE GROUP, so the folded values ARE the scalars and the writer lane can take
// the verdict immediately. The partial arrays are one float per group of the
// pass that produced them, which is `rows / 256` floats, and a single group
// gives each lane a contiguous run of them.
//
// THE VERDICT BODIES ARE CALLED, NOT RESTATED. `pcg_alpha_resident` and
// `pcg_beta_resident` still hold the classification, the sticky latch and the
// announcement; this only moves WHERE they run. A second statement of when a
// solve breaks down is the one thing this must not become.
[[seam::device_fn]] inline void pcg_fold_run(unsigned length, unsigned threads,
                                             unsigned lane,
                                             unsigned &first,
                                             unsigned &last) {
    const unsigned per = (length + threads - 1u) / threads;
    unsigned mine_first = lane * per;
    unsigned mine_last = mine_first + per;
    if (mine_first > length) {
        mine_first = length;
    }
    if (mine_last > length) {
        mine_last = length;
    }
    first = mine_first;
    last = mine_last;
}

[[seam::device_fn]] inline void pcg_fold_alpha(
    const float *curvature_source,
    const float *absolute_source, unsigned length,
    float *scratch,
    float *curvature_out,
    float *absolute_out, const float *rz,
    int *iteration_counter,
    float *value_out, float *noise_out,
    int *cause_out, int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired, unsigned lane, unsigned threads,
    unsigned block) {
    // ONE GROUP, ALWAYS. The caller dispatches a single group whose window is
    // the whole partial array, so the folded values ARE the scalars and the
    // group index carries no information the body needs.
    (void)block;
    unsigned first = 0u;
    unsigned last = 0u;
    pcg_fold_run(length, threads, lane, first, last);
    float curvature_mine = 0.0f;
    float absolute_mine = 0.0f;
    for (unsigned k = first; k < last; ++k) {
        curvature_mine += curvature_source[k];
        absolute_mine += absolute_source[k];
    }
    const float curvature_folded =
        compute::block_sum(curvature_mine, scratch, lane, threads);
    const float absolute_folded =
        compute::block_sum(absolute_mine, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        curvature_out[0] = curvature_folded;
        absolute_out[0] = absolute_folded;
        float value = 0.0f;
        float noise = 0.0f;
        int cause = 0;
        pcg_alpha_from_values(*rz, curvature_folded, absolute_folded,
                              iteration_counter, value, noise, cause,
                              break_cause, break_value, break_iteration,
                              break_fired);
        value_out[0] = value;
        noise_out[0] = noise;
        cause_out[0] = cause;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void pcg_fold_alpha(
    const float *curvature_source,
    const float *absolute_source,
    unsigned length,
    [[seam::scratch(64)]] float *scratch,
    float *curvature_out,
    float *absolute_out,
    const float *rz,
    int *iteration_counter,
    float *value_out,
    float *noise_out,
    int *cause_out,
    int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block,
    unsigned count);

// BETA'S CLASSIFICATION, TAKING THE FOLDED `r . z` BY VALUE, for the same
// reason alpha has one: the fold that produced this number holds it in a
// register, and reading it back out of the slot just written is a dependent
// round trip on one lane. See `pcg_alpha_from_values`.
[[seam::device_fn]] inline void pcg_beta_from_values(
    float next,
    float *rz_previous,
    // READ, NEVER ADVANCED. The alpha pass of this same iteration moved the
    // counter, and both latches want the same number, so a second increment
    // here would make the reported iterate twice the real one.
    const int *iteration_counter,
    float &value, float &noise,
    int &cause, int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired) {
    if (next > 0.0f) {
        const PcgScalar result = pcg_beta(next, *rz_previous);
        value = result.value;
        noise = result.noise;
        cause = result.cause;
    } else {
        value = 0.0f;
        noise = 0.0f;
        cause = PCG_BREAK_RZ;
    }
    if (cause != PCG_BREAK_NONE && break_cause[0] == PCG_BREAK_NONE) {
        break_cause[0] = cause;
        // The offending quantity for this cause is `r . z`, not a curvature.
        break_value[0] = next;
        break_iteration[0] = iteration_counter[0];
    }
    // The same announcement alpha makes, in the scalar buffer the probe already
    // downloads, so the int latch is read only when it holds something.
    if (cause != PCG_BREAK_NONE) {
        break_fired[0] = 1.0f;
    }
    // The roll, unconditional. A solve that breaks here returns before the next
    // iteration reads it.
    *rz_previous = next;
}

// THE POINTER-TAKING FORM, which the standalone entry dispatches.
[[seam::device_fn]] inline void pcg_beta_resident(
    const float *rz_next,
    float *rz_previous,
    const int *iteration_counter,
    float &value, float &noise,
    int &cause, int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired) {
    pcg_beta_from_values(*rz_next, rz_previous, iteration_counter, value, noise,
                         cause, break_cause, break_value, break_iteration,
                         break_fired);
}

[[seam::entry(count)]] void pcg_beta_resident(
    const float *rz_next,
    float *rz_previous,
    const int *iteration_counter,
    float *value_out,
    float *noise_out,
    int *cause_out,
    int *break_cause, float *break_value,
    int *break_iteration,
    float *break_fired, unsigned count);

// BETA'S FOLD, the same shape as alpha's one step later: the group reduces
// `r . z` and `||r||_1` from the fused row's per-group partials, and the writer
// lane takes beta, the roll and the non-positive verdict from the scalars it
// just wrote, so all four happen in one dispatch.
[[seam::device_fn]] inline void pcg_fold_beta(
    const float *product_source,
    const float *residual_source, unsigned length,
    float *scratch,
    float *product_out,
    float *residual_out,
    float *rz_previous,
    const int *iteration_counter,
    float *value_out, float *noise_out,
    int *cause_out, int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired, unsigned lane, unsigned threads,
    unsigned block) {
    // ONE GROUP, ALWAYS. The caller dispatches a single group whose window is
    // the whole partial array, so the folded values ARE the scalars and the
    // group index carries no information the body needs.
    (void)block;
    unsigned first = 0u;
    unsigned last = 0u;
    pcg_fold_run(length, threads, lane, first, last);
    float product_mine = 0.0f;
    float residual_mine = 0.0f;
    for (unsigned k = first; k < last; ++k) {
        product_mine += product_source[k];
        residual_mine += residual_source[k];
    }
    const float product_folded =
        compute::block_sum(product_mine, scratch, lane, threads);
    const float residual_folded =
        compute::block_sum(residual_mine, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        product_out[0] = product_folded;
        residual_out[0] = residual_folded;
        float value = 0.0f;
        float noise = 0.0f;
        int cause = 0;
        // THE ROLL IS INSIDE, so `rz_previous` is read and then written by the
        // one lane that owns it: a single reader and writer, which is what
        // makes the read-then-write order safe.
        pcg_beta_from_values(product_folded, rz_previous, iteration_counter,
                             value, noise, cause, break_cause, break_value,
                             break_iteration, break_fired);
        value_out[0] = value;
        noise_out[0] = noise;
        cause_out[0] = cause;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void pcg_fold_beta(
    const float *product_source,
    const float *residual_source,
    unsigned length,
    [[seam::scratch(64)]] float *scratch,
    float *product_out,
    float *residual_out,
    float *rz_previous,
    const int *iteration_counter,
    float *value_out,
    float *noise_out,
    int *cause_out,
    int *break_cause,
    float *break_value,
    int *break_iteration,
    float *break_fired,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block,
    unsigned count);

// ---------------------------------------------------------------------------
// The REDUCED RIGID solve's per-DOF-group residual.
// ---------------------------------------------------------------------------
//
// The reduced vector mixes incommensurable rows: a cloth row is one vertex's
// force, while a PDRD body's six rows are a wrench summed over all that body's
// vertices. So a heavy or fast body owns `||r||_1` on its own, its 6x6 block is
// preconditioned by the exact reduced inertia and, absent contact, is coupled
// to nothing, so ONE CG step annihilates it. A tolerance measured over the
// whole reduced vector then crosses at iteration 1 while the cloth still
// carries full residual, and the cloth's Newton direction degenerates to a
// single Jacobi sweep. The tolerance is therefore held PER GROUP and the solve
// stops on the WORST group; these two bodies are what a group is measured with.
//
// The split must not collapse: the cloth block folds rows [0, body_base), and
// each body folds its own six.

// One PDRD body's six reduced wrench rows, folded into an L1 norm.
//
// The six are summed in index order, which is the association the fp32 result
// depends on, so the loop shape is part of the body rather than the caller's.
[[seam::device_fn]] inline float
pcg_rigid_body_l1(const float *reduced, unsigned body_base,
                      unsigned body) {
    const float *rows = reduced + body_base + 6u * body;
    float sum = 0.0f;
    for (unsigned k = 0; k < 6; ++k) {
        sum += fmath::abs(rows[k]);
    }
    return sum;
}

// ONE BODY'S L1 NORM, HANDED BACK THROUGH A REFERENCE so an entry point can
// reach it. The fold above stays the definition and this only unpacks it, which
// is the arrangement `pcg_dot_terms` takes over `pcg_dot_term`.
//
// THE OUTPUT IS INDEXED BY THE BODY, NOT BY `1 + body`. The caller's array
// carries the cloth block at slot 0 and the bodies after it, and the driver
// hands this entry a span that already starts at slot 1, so the kernel's own
// index space is the body list and nothing here knows about the cloth slot.
[[seam::entry(body)]]
[[seam::device_fn]] inline void
pcg_rigid_group_l1(const float *reduced, unsigned body_base,
                       unsigned body, float &norm) {
    norm = pcg_rigid_body_l1(reduced, body_base, body);
}

// One group's relative residual against its own seeded initial residual.
//
// A group whose seeded initial residual is exactly zero has no scale of its
// own (it starts solved and can only pick up residual through coupling to
// another group), so it is measured against the whole system's initial scale,
// which the caller has already established is positive. Every scale is
// therefore positive and a ratio can only go non-finite if the residual itself
// does.
//
// A NON-FINITE RATIO IS MAPPED TO INFINITY ON PURPOSE. The caller folds these
// with a max reduction, and a max DROPS a NaN (`max(NaN, x)` returns `x`), so a
// NaN residual would read as convergence. Infinity survives the reduction and
// trips the caller's non-finite check. The finiteness test is spelled as the
// negation of NaN-or-infinite because those two names are the ones every
// backend prologue carries.
[[seam::device_fn]] inline float
pcg_group_relative_residual(float current, float initial,
                                float fallback_scale) {
    const float scale = initial > 0.0f ? initial : fallback_scale;
    const float ratio = fmath::div(current, scale);
    if (fmath::isnan(ratio) || fmath::isinf(ratio)) {
        return fmath::infinity();
    }
    return ratio;
}
