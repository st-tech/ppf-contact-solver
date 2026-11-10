// File: multigrid.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The geometric multigrid preconditioner's device half: the coarse operators,
// the smoother's blocks and the passes of one V-cycle. The hierarchy these read
// is laid on the host once (`src/driver/multigrid_grid.rs`); everything that
// depends on the Newton operator's VALUES is here and runs once per solve.
//
// A NEUTRAL KERNEL SOURCE, so it is plain C++: no preprocessor conditional, no
// macro of its own, and no spelling that only one compiler accepts.
//
// THE THREE THINGS A LEVEL HOLDS, and how each is made.
//
// Its operator is the Galerkin product `P^T A P` of the level above, `P` the
// interpolation between the two, formed in two passes that each give one thread
// a short piece of work. The first writes `A P`, one FINER row per thread: each
// block of the row is spread over the nodes its column is interpolated from.
// The second writes `P^T (A P)`, one COARSE BLOCK per thread: it walks the finer
// rows its node gathers and picks its own column out of each. Both block
// patterns are known from the hierarchy, so nothing is written by two threads
// and no sum depends on scheduling. One pass with a coarse row per thread
// computes the same matrix and was measured several times slower on a GPU,
// where a kernel takes as long as its longest thread.
//
// Its smoother block is `M = D + sum X_ij`, the row's diagonal block plus, for
// each off-diagonal block, a bound `X_ij` that one block puts on both of its
// rows: `[[X, -B], [-B^T, X]]` is positive semidefinite, so `M - A` is
// positive semidefinite whatever `A` is, every eigenvalue of `M^-1 A` lies in
// `(0, 1]`, and a sweep `x + w M^-1 (b - A x)` contracts for any step `w`
// below two with no eigenvalue to estimate. `X` is the block's symmetric part
// taken in absolute value on its own eigenvectors plus its skew part's norm,
// lifted by its single precision round-off (`multigrid_block_bound`), so the
// claim holds for the numbers stored and not only for the real ones. The
// absolute value rather than the norm is what keeps the bound from swamping
// the direction a block does not act in: a membrane block stiffens the two
// in-plane directions of a vertex and its norm would stiffen the normal one
// as much, where only bending acts.
//
// Its cycle is one sweep before the coarse correction and one after, the same
// sweep both times, which makes the whole cycle symmetric as the conjugate
// gradient needs, and positive definite because each sweep contracts.
//
// THE FINEST PRODUCT IS TAKEN IN THE OPERATOR'S DIFFERENCE FORM. The
// conjugate gradient applies the operator as `S_i x_i + sum_j A_ij (x_j - x_i)`
// (`solver/spmv.kernel.cpp`), `S_i` the row's blocks summed, and the first
// coarse operator is the Galerkin product of exactly that: a row of `A P` is
// `S_i P_i + sum_j A_ij (P_j - P_i)`, each block multiplying the DIFFERENCE of
// two interpolations, formed node by node before it is scaled. It is the same
// row as the plain `A_ii P_i + sum_j A_ij P_j`, and it is the only one single
// precision resolves on a stiff pair: two vertices a contact gap apart share
// their coarse nodes with nearly equal weights, so the plain form adds `K w_i`
// and `-K w_j` to one coarse entry and leaves round-off of `eps K` against a
// true value of the inertia `mu`, while the difference form's term is
// `K (w_j - w_i)`, which is small with the weight difference and zero on a
// common motion. With the barrier's `K = m / gap^2` and `mu = m / dt^2`,
// `K / mu = dt^2 / gap^2` passes `1 / (2 eps)` at a gap near `dt sqrt(2 eps)`:
// a felt compressed to a 3.8e-6 m gap is past it.
//
// A CONTACT PARTNER THE PATTERN CANNOT HOLD KEEPS A DIAGONAL BOUND. The
// coarse pattern is laid once from the fixed matrix and the contact pattern
// changes with every solve, so a contact block whose partner interpolates from
// coarse nodes outside the row's pattern cannot be spread. That pair, and only
// that pair, is replaced for the product by its absolute value on the two
// diagonals, `[[|B|, -B], [-B^T, |B^T|]]` added to the operator, which is
// positive semidefinite: the operator the coarse levels see is then no softer
// than the true one in any direction, so a coarse correction never overshoots.
// Replacing the pair everywhere is what a coarse level used to do, and it
// stiffens the pair's common motion from `2 mu` to `2 mu + 4 K`, conditioning
// it no better than block-Jacobi. The decision is the pair's and not a row's:
// it is exact only when each row's pattern holds the other's nodes, so both
// rows decide alike and the coarse operator stays symmetric.
//
// The finest level's sweeps and the residual use the true operator.

#pragma once

#include "../solver/block_jacobi.kernel.cpp"

// One stored block, column-major, read into thread space.
[[seam::device_fn]] inline Mat3x3f
multigrid_read_block(const float *value, unsigned slot) {
    Mat3x3f block;
    for (unsigned entry = 0; entry < 9u; ++entry) {
        block.m[entry] = value[9u * slot + entry];
    }
    return block;
}

// SINGLE PRECISION'S ROUNDING BOUND `gamma_n = n u / (1 - n u)`, `u = 2^-24`:
// a sum or product chain of `n` rounded operations on nonnegative terms is off
// by at most `gamma_n` times the sum of the terms' magnitudes. Every outward
// lift below is this bound, never a tuned constant.
[[seam::device_fn]] inline float multigrid_gamma(float operations) {
    const float unit = 5.9604644775390625e-8f;
    return fmath::div(operations * unit, 1.0f - operations * unit);
}

// The Frobenius norm of a block, scaled by its largest entry so the squares
// stay in range.
[[seam::device_fn]] inline float multigrid_frobenius(const Mat3x3f &block) {
    float scale = 0.0f;
    for (int entry = 0; entry < 9; ++entry) {
        scale = fmath::max(scale, fmath::abs(block.m[entry]));
    }
    if (!(scale > 0.0f)) {
        return 0.0f;
    }
    const float inverse_scale = fmath::div(1.0f, scale);
    float sum = 0.0f;
    for (int entry = 0; entry < 9; ++entry) {
        const float x = block.m[entry] * inverse_scale;
        sum += x * x;
    }
    return scale * fmath::sqrt(sum);
}

// A symmetric block's eigendecomposition, and the bound on its residual.
//
// `residual_bound` majorizes the spectral norm of `Q Lambda Q^T - S` in exact
// arithmetic for the `Q` and `Lambda` returned: the computed residual's norm
// plus `gamma` of the magnitudes that formed it, lifted once more for the
// rounding of this sum itself. `Q` need not be orthogonal for any of this.
struct MultigridEigen {
    Vec3f lambda;
    Mat3x3f basis;
    float residual_bound;
    float magnitude;
};

[[seam::device_fn]] inline MultigridEigen
multigrid_eigen(const Mat3x3f &symmetric) {
    MultigridEigen result;
    linalg::eig::symm3x3(symmetric, result.lambda, result.basis);
    Mat3x3f residual = Mat3x3f::Zero();
    float magnitude = 0.0f;
    for (int k = 0; k < 3; ++k) {
        const float l = result.lambda[k];
        const float q0 = result.basis(0, k), q1 = result.basis(1, k),
                    q2 = result.basis(2, k);
        magnitude += fmath::abs(l) * (q0 * q0 + q1 * q1 + q2 * q2);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                residual(r, c) += l * result.basis(r, k) * result.basis(c, k);
            }
        }
    }
    residual -= symmetric;
    const float slack = multigrid_gamma(8.0f);
    result.magnitude = magnitude;
    result.residual_bound =
        (multigrid_frobenius(residual) +
         slack * (magnitude + multigrid_frobenius(symmetric))) *
        (1.0f + slack);
    return result;
}

// THE BOUND ONE OFF-DIAGONAL BLOCK `B` PUTS ON EACH OF ITS TWO ROWS: one `X`
// with `[[X, -B], [-B^T, X]]` positive semidefinite in exact arithmetic.
//
// `B` splits into its symmetric part `S` and its skew part, axial vector `k`.
// The symmetric part is bounded on its OWN eigenvectors, `Q |Lambda| Q^T`, so
// `X - S = Q (|Lambda| - Lambda) Q^T` and `X + S = Q (|Lambda| + Lambda) Q^T`
// are positive semidefinite for any `Q` up to the residual `Q Lambda Q^T - S`,
// which the lift by its bound covers. The skew part is bounded by `|k| I` on
// both rows. The rounding of forming `X` and of the lift's own
// addition is inside the lift, by `gamma`.
//
// NOT THROUGH THE GRAM PRODUCT. `(B B^T)^(1/2)` from an eigensolve of `B B^T`
// loses half the digits: a Gram eigenvalue carries round-off of
// `eps |B|^2`, and its square root brings a singular value near zero back as
// `sqrt(eps) |B|`. A felt at a contact gap of 3.8e-6 m
// measured that as a fine smoother short of its bound by 1.5e-4 of a row's
// stiffness, 2e4 against an inertia of 1, which made the cycle indefinite.
// Here the round-off is `eps |B|`, and it is lifted rather than hoped away.
//
// THE SAME BITS FROM EITHER ROW. The row that reads `B` transposed forms the
// same symmetric part (`(B_rc + B_cr) / 2` is commutative) and the same `|k|`,
// and evaluates this same function on them, so both rows of the pair carry
// the identical `X` the bound is stated for.
[[seam::device_fn]] inline Mat3x3f
multigrid_block_bound(const Mat3x3f &block) {
    Mat3x3f symmetric;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            symmetric(r, c) = 0.5f * (block(r, c) + block(c, r));
        }
    }
    const float a = 0.5f * (block(2, 1) - block(1, 2));
    const float b = 0.5f * (block(0, 2) - block(2, 0));
    const float c = 0.5f * (block(1, 0) - block(0, 1));
    Mat3x3f bound = Mat3x3f::Zero();
    const float scale = multigrid_frobenius(block);
    if (!(scale > 0.0f)) {
        return bound;
    }
    const MultigridEigen eigen = multigrid_eigen(symmetric);
    for (int k = 0; k < 3; ++k) {
        const float root = fmath::abs(eigen.lambda[k]);
        const float q0 = eigen.basis(0, k), q1 = eigen.basis(1, k),
                    q2 = eigen.basis(2, k);
        bound(0, 0) += root * q0 * q0;
        bound(0, 1) += root * q0 * q1;
        bound(0, 2) += root * q0 * q2;
        bound(1, 1) += root * q1 * q1;
        bound(1, 2) += root * q1 * q2;
        bound(2, 2) += root * q2 * q2;
    }
    bound(1, 0) = bound(0, 1);
    bound(2, 0) = bound(0, 2);
    bound(2, 1) = bound(1, 2);
    const float axial = fmath::abs(a) + fmath::abs(b) + fmath::abs(c);
    const float skew = axial > 0.0f
        ? axial * fmath::sqrt((a / axial) * (a / axial) + (b / axial) * (b / axial) +
                              (c / axial) * (c / axial))
        : 0.0f;
    const float slack = multigrid_gamma(8.0f);
    const float lift = (eigen.residual_bound + skew + slack * eigen.magnitude) *
                       (1.0f + slack);
    for (int d = 0; d < 3; ++d) {
        bound(d, d) += lift;
    }
    return bound;
}

// A smoother block accumulated term by term, with what its rounding needs: the
// sum of the terms' Frobenius norms and how many there were.
struct MultigridSum {
    Mat3x3f value;
    float magnitude;
    float terms;
};

[[seam::device_fn]] inline void multigrid_sum_add(MultigridSum &sum,
                                                  const Mat3x3f &term) {
    sum.value += term;
    sum.magnitude += multigrid_frobenius(term);
    sum.terms += 1.0f;
}

// The smoother block inverted and scaled by the row's step.
//
// THE STEP IS FOLDED INTO THE INVERSE, so a sweep is `x + inverse (b - A x)`
// with nothing else to multiply by. It is per row because rows differ: a row
// with a level below it is over-relaxed, which is what smooths, and a row with
// none takes the step one, which for a row coupled to nothing is the exact
// solve of its own block. The driver states which is which once, when it lays
// the hierarchy.
//
// OUTWARD, SO THE INVERSE NEVER UNDERSHOOTS THE BOUND. The cycle is positive
// semidefinite when `2 M / step - A` is, and `M - A` is, in exact arithmetic,
// for the block the terms sum to. The sum's own rounding is lifted by `gamma` of its terms'
// magnitudes, and the eigenvalues are lifted by the eigensolve's residual
// bound before they are inverted, so `Q (Lambda + tau) Q^T` majorizes the
// summed block. The floor `1e-6 lambda_max` that `block_jacobi_invert` uses is
// kept: it caps the inverted block's condition at 1e6, where forming
// `Q (Lambda + tau)^-1 Q^T` in single precision is off by at most a few times
// `1e6 eps`, 6e-2, inside the quarter of `M` that the
// step 1.6 leaves between `M / step` and `M / 2`.
//
// A ZERO BLOCK INVERTS TO ZERO, AND ONLY A ZERO ONE. A coarse node every one of
// whose vertices is held by a pin gathers nothing, so its row of the operator
// and its block here are exactly zero, and a zero inverse leaves that node out
// of every sweep, which is what it should be. Any other block that is not
// positive definite is an assembly defect upstream and is raised, as the
// block-Jacobi inversion raises it.
[[seam::device_fn]] inline void multigrid_store_inverse(
    const MultigridSum &sum,
    float scale,
    float *inverse,
    unsigned row,
    DiagHandle diag) {
    const Mat3x3f &bound = sum.value;
    bool empty = true;
    bool finite = true;
    float offending = 0.0f;
    for (unsigned entry = 0; entry < 9u; ++entry) {
        if (bound.m[entry] != 0.0f) {
            empty = false;
        }
        if (fmath::isnan(bound.m[entry]) || fmath::isinf(bound.m[entry])) {
            if (finite) {
                offending = bound.m[entry];
            }
            finite = false;
        }
    }
    if (empty) {
        for (unsigned entry = 0; entry < 9u; ++entry) {
            inverse[9u * row + entry] = 0.0f;
        }
        return;
    }
    DIAG_ASSERT4(diag, finite, offending, 0.0f, (float)row, 0.0f);
    Mat3x3f symmetric;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            symmetric(r, c) = 0.5f * (bound(r, c) + bound(c, r));
        }
    }
    const float slack = multigrid_gamma(sum.terms + 1.0f);
    const MultigridEigen eigen = multigrid_eigen(symmetric);
    const float lift = (eigen.residual_bound + slack * sum.magnitude) *
                       (1.0f + multigrid_gamma(8.0f));
    const float lmax = fmath::max(eigen.lambda[0],
                                  fmath::max(eigen.lambda[1], eigen.lambda[2]));
    const float lmin = fmath::min(eigen.lambda[0],
                                  fmath::min(eigen.lambda[1], eigen.lambda[2]));
    DIAG_ASSERT4(diag, lmax > 0.0f && !fmath::isinf(lmax), lmax, lmin,
                 (float)row, 1.0f);
    const float floor = lmax * 1.0e-6f;
    Mat3x3f result = Mat3x3f::Zero();
    for (int k = 0; k < 3; ++k) {
        const float lifted = fmath::max(eigen.lambda[k] + lift, floor);
        const float inv_lk = fmath::div(scale, lifted);
        const float q0 = eigen.basis(0, k), q1 = eigen.basis(1, k),
                    q2 = eigen.basis(2, k);
        result(0, 0) += inv_lk * q0 * q0;
        result(0, 1) += inv_lk * q0 * q1;
        result(0, 2) += inv_lk * q0 * q2;
        result(1, 1) += inv_lk * q1 * q1;
        result(1, 2) += inv_lk * q1 * q2;
        result(2, 2) += inv_lk * q2 * q2;
    }
    result(1, 0) = result(0, 1);
    result(2, 0) = result(0, 2);
    result(2, 1) = result(1, 2);
    for (unsigned entry = 0; entry < 9u; ++entry) {
        inverse[9u * row + entry] = result.m[entry];
    }
}

// ---------------------------------------------------------------------------
// THE FINEST LEVEL'S SMOOTHER.
//
// Its `M` is built from the four spans a row of the Newton operator is stored
// in: `diagonal` plus the absolute value of every off-diagonal block, contact
// included.
//
// `diagonal` is the operator's whole block diagonal, the one the block-Jacobi
// preconditioner inverts, so the contact matrix's own diagonal block is already
// in it and is not added a second time.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_fine_smoother_row(
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
    const float *step,
    float *inverse,
    unsigned row,
    DiagHandle diag) {
    MultigridSum bound{Mat3x3f::Zero(), 0.0f, 0.0f};
    multigrid_sum_add(bound, multigrid_read_block(diagonal, row));
    for (unsigned slot = dynamic_offset[row]; slot < dynamic_offset[row + 1u];
         ++slot) {
        if (dynamic_index[slot] != row) {
            multigrid_sum_add(bound, multigrid_block_bound(
                multigrid_read_block(dynamic_value, slot)));
        }
    }
    for (unsigned slot = reference_offset[row];
         slot < reference_offset[row + 1u]; ++slot) {
        if (reference_index[slot] != row) {
            multigrid_sum_add(bound, multigrid_block_bound(
                multigrid_read_block(global_value, reference_value[slot])));
        }
    }
    for (unsigned slot = fixed_offset[row]; slot < fixed_offset[row + 1u];
         ++slot) {
        if (fixed_index[slot] != row) {
            multigrid_sum_add(bound, multigrid_block_bound(
                multigrid_read_block(fixed_value, slot)));
        }
    }
    for (unsigned slot = transpose_offset[row];
         slot < transpose_offset[row + 1u]; ++slot) {
        if (transpose_pair[2u * slot + 0u] != row) {
            multigrid_sum_add(bound, multigrid_block_bound(multigrid_read_block(
                fixed_value, transpose_pair[2u * slot + 1u])));
        }
    }
    multigrid_store_inverse(bound, step[row], inverse, row, diag);
}

// Where a column sits in a sorted run of a block pattern. `0xFFFFFFFF` when the
// run has no such block.
[[seam::device_fn]] inline unsigned
multigrid_find_slot(const unsigned *column, unsigned begin, unsigned end,
                    unsigned wanted) {
    unsigned low = begin;
    unsigned high = end;
    while (low < high) {
        const unsigned middle = low + (high - low) / 2u;
        if (column[middle] < wanted) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    if (low < end && column[low] == wanted) {
        return low;
    }
    return 0xFFFFFFFFu;
}

// A coarse level's smoother, from the one block-CSR its operator is.
//
// A PAIR'S TWO BLOCKS ARE AVERAGED BEFORE THEY ARE BOUNDED. The Galerkin pass
// forms `(c, d)` and `(d, c)` each in its own order, so they are transposes of
// each other only to round-off, while the operator's quadratic form sees only
// their average `(B_cd + B_dc^T) / 2`. Each row bounds that average, read from
// both slots, and the row `d` forms the transpose of what the row `c` forms,
// so both carry the same bits into `multigrid_block_bound`. A pattern without
// the mirror slot is not the symmetric product the hierarchy laid, and is
// raised.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_smoother_row(
    const unsigned *offset,
    const unsigned *column,
    const float *value,
    const float *step,
    float *inverse,
    unsigned row,
    DiagHandle diag) {
    MultigridSum bound{Mat3x3f::Zero(), 0.0f, 0.0f};
    for (unsigned slot = offset[row]; slot < offset[row + 1u]; ++slot) {
        const Mat3x3f block = multigrid_read_block(value, slot);
        const unsigned other = column[slot];
        if (other == row) {
            multigrid_sum_add(bound, block);
            continue;
        }
        const unsigned mirror =
            multigrid_find_slot(column, offset[other], offset[other + 1u], row);
        DIAG_ASSERT4(diag, mirror != 0xFFFFFFFFu, (float)row, (float)other,
                     0.0f, 2.0f);
        if (mirror == 0xFFFFFFFFu) {
            continue;
        }
        const Mat3x3f partner = multigrid_read_block(value, mirror);
        Mat3x3f average;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                average(r, c) = 0.5f * (block(r, c) + partner(c, r));
            }
        }
        multigrid_sum_add(bound, multigrid_block_bound(average));
    }
    multigrid_store_inverse(bound, step[row], inverse, row, diag);
}

// ---------------------------------------------------------------------------
// THE GALERKIN PRODUCT, in its two passes.


// Spread one finer block over the nodes its column is interpolated from, into
// the run `begin .. end` of `A P`, which is the finer row's own.
//
// A BLOCK THE PATTERN CANNOT HOLD IS RAISED, NOT DROPPED. The pattern is the
// symbolic product of the same interpolation over the same finer pattern, so a
// missing slot means the two were built from different things, and a dropped
// block would leave a coarse operator that is still assembled and no longer the
// product.
[[seam::device_fn]] inline void multigrid_spread_block(
    const unsigned *node_of,
    const float *weight_of,
    const unsigned *column,
    float *value,
    unsigned begin,
    unsigned end,
    unsigned fine_column,
    const Mat3x3f &block,
    unsigned row,
    DiagHandle diag) {
    for (unsigned k = 0; k < 8u; ++k) {
        const float weight = weight_of[8u * fine_column + k];
        if (weight == 0.0f) {
            continue;
        }
        const unsigned target = node_of[8u * fine_column + k];
        const unsigned slot = multigrid_find_slot(column, begin, end, target);
        DIAG_ASSERT4(diag, slot != 0xFFFFFFFFu, (float)row, (float)target,
                     (float)fine_column, 0.0f);
        if (slot == 0xFFFFFFFFu) {
            continue;
        }
        for (unsigned entry = 0; entry < 9u; ++entry) {
            value[9u * slot + entry] += weight * block.m[entry];
        }
    }
}

// Whether a row is interpolated from the level below at all.
//
// A ROW THAT IS NOT HAS NO ROW OF `A P`, AND MUST NOT BE GIVEN ONE. A vertex in
// no element has no grid, and a piece's last node belongs to no level below it,
// so `P` has no row for either and the product does not involve it. But the
// operator still couples such a row to others, through a stitch between two
// pieces of different depth for one, and those others ARE interpolated, so a
// pass that spread the row's blocks anyway would look for slots the pattern was
// right not to have.
[[seam::device_fn]] inline bool
multigrid_is_interpolated(const float *weight_of, unsigned row) {
    for (unsigned k = 0; k < 8u; ++k) {
        if (weight_of[8u * row + k] != 0.0f) {
            return true;
        }
    }
    return false;
}

// One finer block in the difference form, `block (P_other - P_row)`, spread
// into the run `begin .. end` of `A P`, which is the finer row's own.
//
// EACH COARSE NODE'S WEIGHT DIFFERENCE IS FORMED BEFORE IT SCALES THE BLOCK.
// The two rows' nodes are merged into one list, the row's weights entering
// negated, so a node both interpolate from carries `w_other - w_row`, one
// rounding of a small number, rather than two products of the block's full
// size that cancel. A partner a pin removed takes no part in the product, so
// its interpolation is zero and only the row's own nodes are spread.
//
// A slot the pattern cannot hold is raised, as `multigrid_spread_block` raises
// it: a fixed block's partner is in the pattern by construction, and a contact
// block reaches here only after `multigrid_pair_exact` found both rows' nodes.
[[seam::device_fn]] inline void multigrid_spread_difference(
    const unsigned *node_of,
    const float *weight_of,
    const unsigned *column,
    float *value,
    unsigned begin,
    unsigned end,
    unsigned row,
    unsigned other,
    bool other_counts,
    const Mat3x3f &block,
    DiagHandle diag) {
    unsigned node[16];
    float coefficient[16];
    unsigned used = 0u;
    for (unsigned side = 0u; side < 2u; ++side) {
        if (side == 1u && !other_counts) {
            break;
        }
        const unsigned from = side == 0u ? row : other;
        for (unsigned k = 0; k < 8u; ++k) {
            const float weight = weight_of[8u * from + k];
            if (weight == 0.0f) {
                continue;
            }
            const unsigned target = node_of[8u * from + k];
            const float signed_weight = side == 0u ? -weight : weight;
            unsigned at = used;
            for (unsigned n = 0; n < used; ++n) {
                if (node[n] == target) {
                    at = n;
                }
            }
            if (at == used) {
                node[used] = target;
                coefficient[used] = signed_weight;
                ++used;
            } else {
                coefficient[at] += signed_weight;
            }
        }
    }
    for (unsigned n = 0; n < used; ++n) {
        if (coefficient[n] == 0.0f) {
            continue;
        }
        const unsigned slot = multigrid_find_slot(column, begin, end, node[n]);
        DIAG_ASSERT4(diag, slot != 0xFFFFFFFFu, (float)row, (float)node[n],
                     (float)other, 1.0f);
        if (slot == 0xFFFFFFFFu) {
            continue;
        }
        for (unsigned entry = 0; entry < 9u; ++entry) {
            value[9u * slot + entry] += coefficient[n] * block.m[entry];
        }
    }
}

// Whether `row`'s run of the reach pattern holds every coarse node `from`
// interpolates from.
[[seam::device_fn]] inline bool multigrid_reach_holds(
    const unsigned *node_of,
    const float *weight_of,
    const unsigned *reach_offset,
    const unsigned *reach_column,
    unsigned row,
    unsigned from) {
    for (unsigned k = 0; k < 8u; ++k) {
        if (weight_of[8u * from + k] == 0.0f) {
            continue;
        }
        if (multigrid_find_slot(reach_column, reach_offset[row],
                                reach_offset[row + 1u],
                                node_of[8u * from + k]) == 0xFFFFFFFFu) {
            return false;
        }
    }
    return true;
}

// Whether a contact pair enters the product exactly, or as its diagonal bound.
//
// THE PAIR'S DECISION, THE SAME FROM EITHER ROW. Exact when each row's pattern
// holds the other's nodes, a test symmetric in the two, so the two rows never
// disagree and `P^T A P` stays symmetric. A partner a pin removed, or one no
// level interpolates, has no row of `A P` and a zero interpolation, so the
// pair is exact with nothing to hold.
[[seam::device_fn]] inline bool multigrid_pair_exact(
    const unsigned *node_of,
    const float *weight_of,
    const unsigned *removed,
    const unsigned *reach_offset,
    const unsigned *reach_column,
    unsigned row,
    unsigned other) {
    if (removed[other] != 0u || !multigrid_is_interpolated(weight_of, other)) {
        return true;
    }
    return multigrid_reach_holds(node_of, weight_of, reach_offset, reach_column,
                                 row, other) &&
           multigrid_reach_holds(node_of, weight_of, reach_offset, reach_column,
                                 other, row);
}

// THE FIRST PASS AT THE FINEST LEVEL: one row of `A P`, from the Newton
// operator in its difference form.
//
// WHAT IS SPREAD: the row's sum `S_i` (`operator_row_sum`, every block of the
// row) at the row's own nodes, and every off-diagonal block of the four spans
// against the difference of the two interpolations. A block whose column is
// the row itself multiplies a zero difference and is inside `S_i` already. A
// contact pair the pattern cannot hold (`multigrid_pair_exact`) is not spread:
// its block `B` leaves the row and `|B| - B` joins the row's sum, which is the
// row of the operator with that pair's bound in place of the pair.
//
// A VERTEX A PIN REMOVED CONTRIBUTES NOTHING, as a row or as a column. Its row
// of the Newton system is the identity carrying its prescribed increment, which
// is not part of the operator the free vertices see, and a correction is never
// interpolated onto it. Its row of `A P` is written as zeros rather than left,
// because the second pass reads every row its node gathers.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_product_fine_row(
    const unsigned *node_of,
    const float *weight_of,
    const unsigned *removed,
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
    const float *row_sum,
    const unsigned *reach_offset,
    const unsigned *reach_column,
    float *reach_value,
    unsigned row,
    DiagHandle diag) {
    const unsigned begin = reach_offset[row];
    const unsigned end = reach_offset[row + 1u];
    for (unsigned entry = 9u * begin; entry < 9u * end; ++entry) {
        reach_value[entry] = 0.0f;
    }
    if (removed[row] != 0u || !multigrid_is_interpolated(weight_of, row)) {
        return;
    }
    Mat3x3f own = multigrid_read_block(row_sum, row);
    for (unsigned slot = dynamic_offset[row]; slot < dynamic_offset[row + 1u];
         ++slot) {
        const unsigned other = dynamic_index[slot];
        if (other == row) {
            continue;
        }
        const Mat3x3f block = multigrid_read_block(dynamic_value, slot);
        if (multigrid_pair_exact(node_of, weight_of, removed, reach_offset,
                                 reach_column, row, other)) {
            multigrid_spread_difference(node_of, weight_of, reach_column,
                                        reach_value, begin, end, row, other,
                                        removed[other] == 0u, block, diag);
        } else {
            Mat3x3f change = multigrid_block_bound(block);
            change -= block;
            own += change;
        }
    }
    for (unsigned slot = reference_offset[row];
         slot < reference_offset[row + 1u]; ++slot) {
        const unsigned other = reference_index[slot];
        if (other == row) {
            continue;
        }
        const Mat3x3f stored =
            multigrid_read_block(global_value, reference_value[slot]);
        if (multigrid_pair_exact(node_of, weight_of, removed, reach_offset,
                                 reach_column, row, other)) {
            multigrid_spread_difference(node_of, weight_of, reach_column,
                                        reach_value, begin, end, row, other,
                                        removed[other] == 0u,
                                        stored.transpose(), diag);
        } else {
            Mat3x3f change = multigrid_block_bound(stored);
            change -= stored.transpose();
            own += change;
        }
    }
    for (unsigned slot = fixed_offset[row]; slot < fixed_offset[row + 1u];
         ++slot) {
        const unsigned other = fixed_index[slot];
        if (other == row) {
            continue;
        }
        multigrid_spread_difference(node_of, weight_of, reach_column,
                                    reach_value, begin, end, row, other,
                                    removed[other] == 0u,
                                    multigrid_read_block(fixed_value, slot),
                                    diag);
    }
    for (unsigned slot = transpose_offset[row];
         slot < transpose_offset[row + 1u]; ++slot) {
        const unsigned other = transpose_pair[2u * slot + 0u];
        if (other == row) {
            continue;
        }
        const Mat3x3f block = multigrid_read_block(
            fixed_value, transpose_pair[2u * slot + 1u]);
        multigrid_spread_difference(node_of, weight_of, reach_column,
                                    reach_value, begin, end, row, other,
                                    removed[other] == 0u, block.transpose(),
                                    diag);
    }
    multigrid_spread_block(node_of, weight_of, reach_column, reach_value,
                           begin, end, row, own, row, diag);
}

// The first pass on a deeper level: one row of `A P`, from the block-CSR of the
// level above.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_product_row(
    const unsigned *node_of,
    const float *weight_of,
    const unsigned *fine_offset,
    const unsigned *fine_column,
    const float *fine_value,
    const unsigned *reach_offset,
    const unsigned *reach_column,
    float *reach_value,
    unsigned row,
    DiagHandle diag) {
    const unsigned begin = reach_offset[row];
    const unsigned end = reach_offset[row + 1u];
    for (unsigned entry = 9u * begin; entry < 9u * end; ++entry) {
        reach_value[entry] = 0.0f;
    }
    if (!multigrid_is_interpolated(weight_of, row)) {
        return;
    }
    for (unsigned slot = fine_offset[row]; slot < fine_offset[row + 1u];
         ++slot) {
        multigrid_spread_block(node_of, weight_of, reach_column, reach_value,
                               begin, end, fine_column[slot],
                               multigrid_read_block(fine_value, slot), row,
                               diag);
    }
}

// THE SECOND PASS, one coarse block per thread: `(P^T (A P))` at `(node,
// column)`, gathered from the rows of `A P` the node interpolates.
//
// A gathered row that does not reach this column contributes nothing, which is
// the ordinary case rather than a defect: a node's rows reach different sets of
// columns and the block's pattern is their union.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void multigrid_galerkin_slot(
    const unsigned *gather_offset,
    const unsigned *gather_row,
    const float *gather_weight,
    const unsigned *reach_offset,
    const unsigned *reach_column,
    const float *reach_value,
    const unsigned *slot_node,
    const unsigned *column,
    float *value,
    unsigned slot) {
    const unsigned node = slot_node[slot];
    const unsigned wanted = column[slot];
    Mat3x3f sum = Mat3x3f::Zero();
    for (unsigned g = gather_offset[node]; g < gather_offset[node + 1u]; ++g) {
        const unsigned row = gather_row[g];
        const unsigned found = multigrid_find_slot(
            reach_column, reach_offset[row], reach_offset[row + 1u], wanted);
        if (found == 0xFFFFFFFFu) {
            continue;
        }
        const float weight = gather_weight[g];
        for (unsigned entry = 0; entry < 9u; ++entry) {
            sum.m[entry] += weight * reach_value[9u * found + entry];
        }
    }
    for (unsigned entry = 0; entry < 9u; ++entry) {
        value[9u * slot + entry] = sum.m[entry];
    }
}

// ---------------------------------------------------------------------------
// THE CYCLE'S PASSES. Every one writes the row it was dispatched for and
// nothing else, so each is a plain range a backend may cut anywhere.

// `inverse * vector`, one block and three floats.
[[seam::device_fn]] inline Vec3f
multigrid_apply_block(const float *inverse, unsigned row, const Vec3f &vector) {
    Vec3f result;
    for (unsigned r = 0; r < 3u; ++r) {
        result[r] = inverse[9u * row + r] * vector[0] +
                    inverse[9u * row + 3u + r] * vector[1] +
                    inverse[9u * row + 6u + r] * vector[2];
    }
    return result;
}

// One row of `b - A x` for a block-CSR operator.
[[seam::device_fn]] inline Vec3f multigrid_residual_of(
    const unsigned *offset,
    const unsigned *column,
    const float *value,
    const float *right,
    const float *x,
    unsigned row) {
    Vec3f residual;
    residual[0] = right[3u * row + 0u];
    residual[1] = right[3u * row + 1u];
    residual[2] = right[3u * row + 2u];
    for (unsigned slot = offset[row]; slot < offset[row + 1u]; ++slot) {
        const unsigned other = column[slot];
        const float x0 = x[3u * other + 0u];
        const float x1 = x[3u * other + 1u];
        const float x2 = x[3u * other + 2u];
        for (unsigned r = 0; r < 3u; ++r) {
            residual[r] -= value[9u * slot + r] * x0 +
                           value[9u * slot + 3u + r] * x1 +
                           value[9u * slot + 6u + r] * x2;
        }
    }
    return residual;
}

// `x = inverse b`: the sweep from a zero iterate, which needs no operator.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_first_sweep_row(
    const float *inverse,
    const float *right,
    float *x,
    unsigned row) {
    Vec3f vector;
    vector[0] = right[3u * row + 0u];
    vector[1] = right[3u * row + 1u];
    vector[2] = right[3u * row + 2u];
    const Vec3f moved = multigrid_apply_block(inverse, row, vector);
    for (unsigned r = 0; r < 3u; ++r) {
        x[3u * row + r] = moved[r];
    }
}

// `r = b - A x` on a coarse level.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_residual_row(
    const unsigned *offset,
    const unsigned *column,
    const float *value,
    const float *right,
    const float *x,
    float *residual,
    unsigned row) {
    const Vec3f r = multigrid_residual_of(offset, column, value, right, x, row);
    for (unsigned k = 0; k < 3u; ++k) {
        residual[3u * row + k] = r[k];
    }
}

// `out = x + inverse (b - A x)` on a coarse level. `out` is another array than
// `x`: every row reads its neighbors' `x` as it was.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_sweep_row(
    const unsigned *offset,
    const unsigned *column,
    const float *value,
    const float *inverse,
    const float *right,
    const float *x,
    float *out,
    unsigned row) {
    const Vec3f r = multigrid_residual_of(offset, column, value, right, x, row);
    const Vec3f moved = multigrid_apply_block(inverse, row, r);
    for (unsigned k = 0; k < 3u; ++k) {
        out[3u * row + k] = x[3u * row + k] + moved[k];
    }
}

// The finest level's sweep, the operator already applied: `product = A x`, so
// `out = x + inverse (b - product)` is local to the row.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_fine_sweep_row(
    const float *inverse,
    const float *right,
    const float *product,
    const float *x,
    float *out,
    unsigned row) {
    Vec3f r;
    for (unsigned k = 0; k < 3u; ++k) {
        r[k] = right[3u * row + k] - product[3u * row + k];
    }
    const Vec3f moved = multigrid_apply_block(inverse, row, r);
    for (unsigned k = 0; k < 3u; ++k) {
        out[3u * row + k] = x[3u * row + k] + moved[k];
    }
}

// `coarse = P^T r` and that level's first sweep, one coarse node gathering the
// finer rows it interpolates.
//
// THE SWEEP FROM ZERO RIDES WITH THE RESTRICTION. It is `x = inverse coarse`,
// which reads nothing but the row's own right-hand side, so the thread that has
// just formed that side can finish it, and the level below needs no dispatch of
// its own to start.
[[seam::entry(node)]]
[[seam::device_fn]] inline void multigrid_restrict_row(
    const unsigned *gather_offset,
    const unsigned *gather_row,
    const float *gather_weight,
    const float *residual,
    const float *inverse,
    float *coarse,
    float *x,
    unsigned node) {
    Vec3f sum = Vec3f::Zero();
    for (unsigned g = gather_offset[node]; g < gather_offset[node + 1u]; ++g) {
        const unsigned row = gather_row[g];
        const float weight = gather_weight[g];
        sum[0] += weight * residual[3u * row + 0u];
        sum[1] += weight * residual[3u * row + 1u];
        sum[2] += weight * residual[3u * row + 2u];
    }
    const Vec3f moved = multigrid_apply_block(inverse, node, sum);
    for (unsigned k = 0; k < 3u; ++k) {
        coarse[3u * node + k] = sum[k];
        x[3u * node + k] = moved[k];
    }
}

// The same from the finest level, where the residual is `right - product` and a
// removed vertex is left out.
[[seam::entry(node)]]
[[seam::device_fn]] inline void multigrid_fine_restrict_row(
    const unsigned *gather_offset,
    const unsigned *gather_row,
    const float *gather_weight,
    const unsigned *removed,
    const float *right,
    const float *product,
    const float *inverse,
    float *coarse,
    float *x,
    unsigned node) {
    Vec3f sum = Vec3f::Zero();
    for (unsigned g = gather_offset[node]; g < gather_offset[node + 1u]; ++g) {
        const unsigned row = gather_row[g];
        if (removed[row] != 0u) {
            continue;
        }
        const float weight = gather_weight[g];
        sum[0] += weight * (right[3u * row + 0u] - product[3u * row + 0u]);
        sum[1] += weight * (right[3u * row + 1u] - product[3u * row + 1u]);
        sum[2] += weight * (right[3u * row + 2u] - product[3u * row + 2u]);
    }
    const Vec3f moved = multigrid_apply_block(inverse, node, sum);
    for (unsigned k = 0; k < 3u; ++k) {
        coarse[3u * node + k] = sum[k];
        x[3u * node + k] = moved[k];
    }
}

// `x += P coarse`, one finer row reading the nodes it is interpolated from.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_prolong_row(
    const unsigned *node_of,
    const float *weight_of,
    const float *coarse,
    float *x,
    unsigned row) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (unsigned k = 0; k < 8u; ++k) {
        const float weight = weight_of[8u * row + k];
        const unsigned node = node_of[8u * row + k];
        sum0 += weight * coarse[3u * node + 0u];
        sum1 += weight * coarse[3u * node + 1u];
        sum2 += weight * coarse[3u * node + 2u];
    }
    x[3u * row + 0u] += sum0;
    x[3u * row + 1u] += sum1;
    x[3u * row + 2u] += sum2;
}

// The same onto the finest level, where a removed vertex takes no correction.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_fine_prolong_row(
    const unsigned *node_of,
    const float *weight_of,
    const unsigned *removed,
    const float *coarse,
    float *x,
    unsigned row) {
    if (removed[row] != 0u) {
        return;
    }
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    for (unsigned k = 0; k < 8u; ++k) {
        const float weight = weight_of[8u * row + k];
        const unsigned node = node_of[8u * row + k];
        sum0 += weight * coarse[3u * node + 0u];
        sum1 += weight * coarse[3u * node + 1u];
        sum2 += weight * coarse[3u * node + 2u];
    }
    x[3u * row + 0u] += sum0;
    x[3u * row + 1u] += sum1;
    x[3u * row + 2u] += sum2;
}
