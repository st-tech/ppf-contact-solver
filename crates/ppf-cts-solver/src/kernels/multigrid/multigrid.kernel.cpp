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
// Its smoother block is `M = D + sum |A_ij|`, the row's diagonal block plus the
// ABSOLUTE VALUE `(A_ij A_ij^T)^(1/2)` of each off-diagonal block. For any block
// `B` the pair `(|B|_row, |B|_column)` bounds the coupling `(B, B^T)` from above,
// so `M - A` is positive semidefinite whatever `A` is, every eigenvalue of
// `M^-1 A` lies in `(0, 1]`, and a sweep `x + w M^-1 (b - A x)` contracts for any
// step `w` below two with no eigenvalue to estimate. The absolute value rather than
// the norm is what keeps the bound from swamping the direction a block does not
// act in: a membrane block stiffens the two in-plane directions of a vertex and
// its norm would stiffen the normal one as much, where only bending acts.
//
// Its cycle is one sweep before the coarse correction and one after, the same
// sweep both times, which makes the whole cycle symmetric as the conjugate
// gradient needs, and positive definite because each sweep contracts.
//
// THE CONTACT MATRIX REACHES THE COARSE LEVELS AS A DIAGONAL BOUND. Its pattern
// changes with every solve, and a coarse pattern that followed it would have to
// be rediscovered each time. So each off-diagonal contact block is replaced, for
// the purpose of the product only, by its absolute value on the diagonal of its
// own row. That is the same bound as above: the operator the coarse levels see
// is no softer than the true one in any direction, so a coarse correction never
// overshoots. The finest level's sweeps and the residual use the true operator.

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

// The absolute value of a block as its row sees it: `(B B^T)^(1/2)`, or
// `(B^T B)^(1/2)` when the stored block is read transposed.
//
// SCALED BEFORE IT IS SQUARED. A barrier block reaches 1e11 and beyond, and the
// Gram product of an unscaled one leaves single precision's range long before
// its square root would have brought it back.
[[seam::device_fn]] inline Mat3x3f
multigrid_block_bound(const Mat3x3f &block, bool transposed) {
    float scale = 0.0f;
    for (int entry = 0; entry < 9; ++entry) {
        scale = fmath::max(scale, fmath::abs(block.m[entry]));
    }
    Mat3x3f bound = Mat3x3f::Zero();
    if (!(scale > 0.0f)) {
        return bound;
    }
    const float inverse_scale = fmath::div(1.0f, scale);
    Mat3x3f gram;
    for (int r = 0; r < 3; ++r) {
        for (int c = r; c < 3; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                const float left = transposed ? block(k, r) : block(r, k);
                const float right = transposed ? block(k, c) : block(c, k);
                sum += (left * inverse_scale) * (right * inverse_scale);
            }
            gram(r, c) = sum;
            gram(c, r) = sum;
        }
    }
    Vec3f lambda;
    Mat3x3f basis;
    linalg::eig::symm3x3(gram, lambda, basis);
    for (int k = 0; k < 3; ++k) {
        const float root = scale * fmath::sqrt(fmath::max(lambda[k], 0.0f));
        const float q0 = basis(0, k), q1 = basis(1, k), q2 = basis(2, k);
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
    return bound;
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
// A ZERO BLOCK INVERTS TO ZERO, AND ONLY A ZERO ONE. A coarse node every one of
// whose vertices is held by a pin gathers nothing, so its row of the operator
// and its block here are exactly zero, and a zero inverse leaves that node out
// of every sweep, which is what it should be. Any other block that is not
// positive definite is an assembly defect upstream and is raised, as the
// block-Jacobi inversion raises it.
[[seam::device_fn]] inline void multigrid_store_inverse(
    const Mat3x3f &bound,
    float scale,
    float *inverse,
    unsigned row,
    DiagHandle diag) {
    bool empty = true;
    for (unsigned entry = 0; entry < 9u; ++entry) {
        if (bound.m[entry] != 0.0f) {
            empty = false;
        }
    }
    if (empty) {
        for (unsigned entry = 0; entry < 9u; ++entry) {
            inverse[9u * row + entry] = 0.0f;
        }
        return;
    }
    const BlockJacobiInverse result = block_jacobi_invert(bound);
    DIAG_ASSERT4(diag, result.finite_input, result.offending_entry,
                 result.lambda_max, (float)row, 0.0f);
    DIAG_ASSERT4(diag, result.valid, result.lambda_max, result.eigenvalue[0],
                 result.eigenvalue[2], (float)row);
    for (unsigned entry = 0; entry < 9u; ++entry) {
        inverse[9u * row + entry] = scale * result.inverse.m[entry];
    }
}

// ---------------------------------------------------------------------------
// THE FINEST LEVEL'S SMOOTHER, AND WHAT THE COARSE LEVELS ARE HANDED FOR
// CONTACT.
//
// The smoother's `M` is built from the four spans a row of the Newton operator
// is stored in. `contact` is what the coarse levels are handed in place of the
// contact matrix: its diagonal block where it has one, plus the absolute value
// of each of its off-diagonal blocks.
//
// `diagonal` is the operator's whole block diagonal, the one the block-Jacobi
// preconditioner inverts, so the contact matrix's own diagonal block is already
// in it and is not added to the bound a second time.
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
    float *contact,
    float *inverse,
    unsigned row,
    DiagHandle diag) {
    Mat3x3f contact_diagonal = Mat3x3f::Zero();
    Mat3x3f contact_bound = Mat3x3f::Zero();
    Mat3x3f bound = multigrid_read_block(diagonal, row);
    for (unsigned slot = dynamic_offset[row]; slot < dynamic_offset[row + 1u];
         ++slot) {
        const Mat3x3f block = multigrid_read_block(dynamic_value, slot);
        if (dynamic_index[slot] == row) {
            contact_diagonal += block;
        } else {
            contact_bound += multigrid_block_bound(block, false);
        }
    }
    for (unsigned slot = reference_offset[row];
         slot < reference_offset[row + 1u]; ++slot) {
        const Mat3x3f block =
            multigrid_read_block(global_value, reference_value[slot]);
        if (reference_index[slot] != row) {
            contact_bound += multigrid_block_bound(block, true);
        }
    }
    for (unsigned slot = fixed_offset[row]; slot < fixed_offset[row + 1u];
         ++slot) {
        if (fixed_index[slot] != row) {
            bound += multigrid_block_bound(
                multigrid_read_block(fixed_value, slot), false);
        }
    }
    for (unsigned slot = transpose_offset[row];
         slot < transpose_offset[row + 1u]; ++slot) {
        if (transpose_pair[2u * slot + 0u] != row) {
            bound += multigrid_block_bound(
                multigrid_read_block(fixed_value, transpose_pair[2u * slot + 1u]),
                true);
        }
    }
    bound += contact_bound;
    for (unsigned entry = 0; entry < 9u; ++entry) {
        contact[9u * row + entry] =
            contact_diagonal.m[entry] + contact_bound.m[entry];
    }
    multigrid_store_inverse(bound, step[row], inverse, row, diag);
}

// A coarse level's smoother, from the one block-CSR its operator is.
[[seam::entry(row)]]
[[seam::device_fn]] inline void multigrid_smoother_row(
    const unsigned *offset,
    const unsigned *column,
    const float *value,
    const float *step,
    float *inverse,
    unsigned row,
    DiagHandle diag) {
    Mat3x3f bound = Mat3x3f::Zero();
    for (unsigned slot = offset[row]; slot < offset[row + 1u]; ++slot) {
        const Mat3x3f block = multigrid_read_block(value, slot);
        if (column[slot] == row) {
            bound += block;
        } else {
            bound += multigrid_block_bound(block, false);
        }
    }
    multigrid_store_inverse(bound, step[row], inverse, row, diag);
}

// ---------------------------------------------------------------------------
// THE GALERKIN PRODUCT, in its two passes.

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

// THE FIRST PASS AT THE FINEST LEVEL: one row of `A P`, from the Newton
// operator.
//
// WHAT IS SPREAD: the fixed matrix's two spans, read as stored and transposed,
// and one diagonal block, which is the operator's block diagonal `C` plus
// `contact`, the contact matrix's bound from `multigrid_fine_smoother_row`.
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
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const float *diagonal,
    const float *contact,
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
    Mat3x3f own = multigrid_read_block(diagonal, row);
    own += multigrid_read_block(contact, row);
    multigrid_spread_block(node_of, weight_of, reach_column, reach_value,
                           begin, end, row, own, row, diag);
    for (unsigned slot = fixed_offset[row]; slot < fixed_offset[row + 1u];
         ++slot) {
        const unsigned other = fixed_index[slot];
        if (removed[other] != 0u) {
            continue;
        }
        multigrid_spread_block(node_of, weight_of, reach_column, reach_value,
                               begin, end, other,
                               multigrid_read_block(fixed_value, slot), row,
                               diag);
    }
    for (unsigned slot = transpose_offset[row];
         slot < transpose_offset[row + 1u]; ++slot) {
        const unsigned other = transpose_pair[2u * slot + 0u];
        if (removed[other] != 0u) {
            continue;
        }
        const Mat3x3f block = multigrid_read_block(
            fixed_value, transpose_pair[2u * slot + 1u]);
        multigrid_spread_block(node_of, weight_of, reach_column, reach_value,
                               begin, end, other, block.transpose(), row, diag);
    }
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
