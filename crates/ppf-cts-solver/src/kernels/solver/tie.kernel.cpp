// File: tie.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. `[[seam::device_fn]]` is the
// execution space; `Vec2u` and `Vec3f` arrive from whichever backend prologue
// is in scope.

// A TIE holds one vertex t on an affine combination of up to `tie_slots()`
// other vertices, its CONTROLLERS:
//
//     x_t = sum_i w_i x_{g_i},    sum_i w_i = 1.
//
// The held vertex owns no degree of freedom. Its Newton correction is the same
// combination of its controllers' corrections plus a consistency term, so the
// step is `dx = E dy` with
//
//     (E u)_v = u_v                              for a vertex that is not held,
//     (E u)_t = u_t + sum_i w_i u_{g_i}          for a held vertex t,
//
// and the system is solved on the subspace where every held row of `dy` is
// zero, by Galerkin projection with the FOLD
//
//     (F y)_g = y_g + sum_{(t, i): g_i = g} w_i y_t    for a free vertex g,
//     (F y)_g = y_g                                    for a prescribed one,
//     (F y)_t = 0                                      for a held vertex t.
//
// On that subspace F is E's transpose, so F K E is the Hessian restricted to
// the constraint and is SPD wherever K is. Nothing is assembled: the held
// vertex keeps its mass, its elasticity and its contact in K, and F carries
// them onto its controllers.
//
// THE HELD ROW'S OWN SLOT CARRIES THE CONSISTENCY TERM, and only the SEED has
// one. `tie_consistency_row` writes the difference from where the held vertex
// sits to where its tie puts it into that slot, so the seed's expansion is
// `E dy0 + c` and the residual the solve starts from carries it; every search
// direction afterwards has a zero held row, because F zeroes the residual
// there and the preconditioner maps a zero row to a zero row. This is the same
// arrangement as a prescribed row's seeded increment, and it is what keeps
// round-off from accumulating into a drift between a held vertex and its
// controllers over a run.
//
// A PRESCRIBED CONTROLLER IS SKIPPED BY THE FOLD, and that is the whole of its
// treatment. The Dirichlet pass made its row the identity with its increment
// on the right-hand side and seeded its correction; leaving that row untouched
// keeps its residual zero, so its search directions are zero and the seed's
// expansion carries its increment into the held row, which lifts it onto the
// free rows through K.
//
// EVERY PASS IS A GATHER and writes only its own row, so each is `Disjoint`:
// the transposed list `ctrl_offset` / `ctrl_entry` is what lets the fold
// gather onto a controller instead of scattering from a held vertex, and it
// fixes the summation order.
//
// A HELD VERTEX IS NEVER A CONTROLLER, which the driver refuses at
// `initialize()`. Every pass below that writes a controller's row while
// reading a held row, or the reverse, relies on that: the two sets are
// disjoint, so no thread reads a row another thread writes.

// The number of controller slots a tie carries. A shorter tie pads with
// weight zero, and a zero-weight slot is never read.
[[seam::device_fn]] inline unsigned tie_slots() { return 8u; }

// The marker `tie_of` holds for a vertex that is not held.
[[seam::device_fn]] inline unsigned tie_none() { return 0xffffffffu; }

// `output = E input`, one row per thread, on a vector of three floats per
// vertex. `input` and `output` must be different buffers.
[[seam::entry(row)]]
[[seam::device_fn]] inline void tie_expand_row(
    const unsigned *tie_of,
    const unsigned *ctrl,
    const float *weight,
    const float *input,
    float *output, unsigned row) {
    const unsigned tie = tie_of[row];
    float sum0 = input[3 * row + 0];
    float sum1 = input[3 * row + 1];
    float sum2 = input[3 * row + 2];
    if (tie != tie_none()) {
        for (unsigned slot = 0; slot < tie_slots(); ++slot) {
            const float w = weight[tie_slots() * tie + slot];
            if (w != 0.0f) {
                const unsigned g = ctrl[tie_slots() * tie + slot];
                sum0 += w * input[3 * g + 0];
                sum1 += w * input[3 * g + 1];
                sum2 += w * input[3 * g + 2];
            }
        }
    }
    output[3 * row + 0] = sum0;
    output[3 * row + 1] = sum1;
    output[3 * row + 2] = sum2;
}

// `output = F input`, one row per thread. `input` and `output` must be
// different buffers: a held row is read by its controllers' threads while its
// own thread zeroes it in the output.
//
// `ctrl_entry` holds `tie_slots() * tie + slot` for every nonzero slot naming
// this row, `ctrl_offset[row] .. ctrl_offset[row + 1]` of them, in ascending
// order, which fixes the order of the sum.
[[seam::entry(row)]]
[[seam::device_fn]] inline void tie_fold_row(
    const unsigned *tie_of,
    const unsigned *tie_vertex,
    const float *weight,
    const unsigned *ctrl_offset,
    const unsigned *ctrl_entry,
    const unsigned *dof_mask,
    const float *input,
    float *output, unsigned row) {
    if (tie_of[row] != tie_none()) {
        output[3 * row + 0] = 0.0f;
        output[3 * row + 1] = 0.0f;
        output[3 * row + 2] = 0.0f;
        return;
    }
    float sum0 = input[3 * row + 0];
    float sum1 = input[3 * row + 1];
    float sum2 = input[3 * row + 2];
    if (dof_mask[row] == 0u) {
        for (unsigned k = ctrl_offset[row]; k < ctrl_offset[row + 1]; ++k) {
            const unsigned entry = ctrl_entry[k];
            const unsigned t = tie_vertex[entry / tie_slots()];
            const float w = weight[entry];
            sum0 += w * input[3 * t + 0];
            sum1 += w * input[3 * t + 1];
            sum2 += w * input[3 * t + 2];
        }
    }
    output[3 * row + 0] = sum0;
    output[3 * row + 1] = sum1;
    output[3 * row + 2] = sum2;
}

// Zero every held row of a vector, in place, one row per thread. A
// preconditioner that mixes rows (the Cosserat edge blocks) can write a held
// row from its neighbors; this keeps its output on the subspace.
[[seam::entry(row)]]
[[seam::device_fn]] inline void tie_zero_row(
    const unsigned *tie_of,
    float *vector, unsigned row) {
    if (tie_of[row] != tie_none()) {
        vector[3 * row + 0] = 0.0f;
        vector[3 * row + 1] = 0.0f;
        vector[3 * row + 2] = 0.0f;
    }
}

// The consistency term, written into a held row of the seed `dx`.
//
// THE STEP IS SUBTRACTED, `x_new = x - dx` (`main/position_step.kernel.cpp`),
// so the correction that puts a held vertex back on its tie is
// `x_t - sum_i w_i x_{g_i} = sum_i w_i (x_t - x_{g_i})`. The second form,
// which the weights summing to one allows, is the one taken: each
// `x_t - x_{g_i}` is a difference of two nearby positions, which is exact or
// nearly so, and the weights then scale small numbers. The first form would
// subtract a weighted sum of large coordinates from a large coordinate and
// lose the leading digits of the answer.
[[seam::entry(row)]]
[[seam::device_fn]] inline void tie_consistency_row(
    const unsigned *tie_of,
    const unsigned *ctrl,
    const float *weight,
    const Vec3f *eval_x,
    float *dx, unsigned row) {
    const unsigned tie = tie_of[row];
    if (tie == tie_none()) {
        return;
    }
    const Vec3f held = eval_x[row];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (unsigned slot = 0; slot < tie_slots(); ++slot) {
        const float w = weight[tie_slots() * tie + slot];
        if (w != 0.0f) {
            // Read into a thread value first: Metal has no operator between a
            // thread-space and a device-space vector.
            const Vec3f controller = eval_x[ctrl[tie_slots() * tie + slot]];
            const Vec3f gap = held - controller;
            sum0 += w * gap[0];
            sum1 += w * gap[1];
            sum2 += w * gap[2];
        }
    }
    dx[3 * row + 0] = sum0;
    dx[3 * row + 1] = sum1;
    dx[3 * row + 2] = sum2;
}

// The solved correction's held rows: `dx_t += sum_i w_i dx_{g_i}`, in place.
// The held row holds the consistency term the seed put there and the solve
// never moved, so the sum is `(E dy)_t` with the seed's offset included.
// In place is safe because a held row reads only controller rows, which this
// pass does not write.
//
[[seam::entry(row)]]
[[seam::device_fn]] inline void tie_finish_row(
    const unsigned *tie_of,
    const unsigned *ctrl,
    const float *weight,
    float *dx, unsigned row) {
    const unsigned tie = tie_of[row];
    if (tie == tie_none()) {
        return;
    }
    float sum0 = dx[3 * row + 0];
    float sum1 = dx[3 * row + 1];
    float sum2 = dx[3 * row + 2];
    for (unsigned slot = 0; slot < tie_slots(); ++slot) {
        const float w = weight[tie_slots() * tie + slot];
        if (w != 0.0f) {
            const unsigned g = ctrl[tie_slots() * tie + slot];
            sum0 += w * dx[3 * g + 0];
            sum1 += w * dx[3 * g + 1];
            sum2 += w * dx[3 * g + 2];
        }
    }
    dx[3 * row + 0] = sum0;
    dx[3 * row + 1] = sum1;
    dx[3 * row + 2] = sum2;
}

// One 3x3 block `K_uv` (u != v) of the Newton operator, column-major, summed
// over both matrices and over both storage directions, times `scale`, added
// into `block`. A coupling is stored once, in the row with the smaller index,
// and the transpose run of the other row reaches it, so walking row `u`'s
// direct run for column `v` and its transpose run for source `v` finds it
// whichever row stored it.
[[seam::device_fn]] inline void tie_coupling(
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    unsigned u, unsigned v,
    float scale,
    float *block) {
    for (unsigned slot = offset[u]; slot < offset[u + 1]; ++slot) {
        if (index[slot] == v) {
            for (unsigned e = 0; e < 9u; ++e) {
                block[e] += scale * value[9 * slot + e];
            }
        }
    }
    for (unsigned entry = transpose_offset[u]; entry < transpose_offset[u + 1]; ++entry) {
        if (transpose_pair[2 * entry] == v) {
            const unsigned slot = transpose_pair[2 * entry + 1];
            for (unsigned r = 0; r < 3u; ++r) {
                for (unsigned c = 0; c < 3u; ++c) {
                    block[3 * c + r] += scale * value[9 * slot + 3 * r + c];
                }
            }
        }
    }
    for (unsigned entry = dyn_offset[u]; entry < dyn_offset[u + 1]; ++entry) {
        if (dyn_index[entry] == v) {
            for (unsigned e = 0; e < 9u; ++e) {
                block[e] += scale * dyn_value[9 * entry + e];
            }
        }
    }
    for (unsigned entry = dyn_reference_offset[u]; entry < dyn_reference_offset[u + 1];
         ++entry) {
        if (dyn_reference_index[entry] == v) {
            const unsigned slot = dyn_reference_value[entry];
            for (unsigned r = 0; r < 3u; ++r) {
                for (unsigned c = 0; c < 3u; ++c) {
                    block[3 * c + r] += scale * dyn_global_value[9 * slot + 3 * r + c];
                }
            }
        }
    }
}

// The EXACT diagonal block of the reduced operator `E^T K E` on a free
// controller `g`, added in place to the block the operator's own diagonal
// already holds there (`K_gg`):
//
//     (E^T K E)_gg = K_gg + sum_a w_a (K_{g t_a} + K_{t_a g})
//                         + sum_{a, b} w_a w_b K_{t_a t_b}
//
// over the held vertices `t_a` that `g` controls, with weights `w_a`. The
// pairs `a != b` are the couplings between two held vertices sharing a
// controller (consecutive vertices of one tied thread, whose stretch is stiff
// and of opposite sign), and `K_{g t_a}` is a held vertex's contact with its
// own controller; both are part of the block a block-Jacobi preconditioner
// should invert, and dropping them gives a diagonal that is not the reduced
// operator's. The block stays symmetric by construction, each pair entering
// with its transpose.
//
// A held row's own block is left as assembled: its residual is zero, so the
// preconditioner maps that row to zero whatever the block holds. A prescribed
// controller is skipped, its row being the identity. Held rows are read and
// never written, controllers written and never read by another thread, so the
// pass is in place.
[[seam::entry(row)]]
[[seam::device_fn]] inline void tie_precond_row(
    const unsigned *tie_vertex,
    const float *weight,
    const unsigned *ctrl_offset,
    const unsigned *ctrl_entry,
    const unsigned *dof_mask,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    float *diagonal, unsigned row) {
    if (dof_mask[row] != 0u || ctrl_offset[row] == ctrl_offset[row + 1]) {
        return;
    }
    float block[9];
    for (unsigned e = 0; e < 9u; ++e) {
        block[e] = 0.0f;
    }
    for (unsigned a = ctrl_offset[row]; a < ctrl_offset[row + 1]; ++a) {
        const unsigned entry_a = ctrl_entry[a];
        const unsigned t_a = tie_vertex[entry_a / tie_slots()];
        const float w_a = weight[entry_a];
        for (unsigned e = 0; e < 9u; ++e) {
            block[e] += w_a * w_a * diagonal[9 * t_a + e];
        }
        // K_{g t_a} + K_{t_a g}: the block and its transpose, which the
        // symmetrization below adds; and each pair (a, b > a) of held
        // vertices once, its transpose being the pair (b, a).
        float cross[9];
        for (unsigned e = 0; e < 9u; ++e) {
            cross[e] = 0.0f;
        }
        tie_coupling(index, offset, value, transpose_pair, transpose_offset,
                     dyn_index, dyn_value, dyn_offset, dyn_reference_index,
                     dyn_reference_value, dyn_reference_offset, dyn_global_value,
                     row, t_a, w_a, cross);
        for (unsigned b = a + 1; b < ctrl_offset[row + 1]; ++b) {
            const unsigned entry_b = ctrl_entry[b];
            const unsigned t_b = tie_vertex[entry_b / tie_slots()];
            if (t_b != t_a) {
                tie_coupling(index, offset, value, transpose_pair, transpose_offset,
                             dyn_index, dyn_value, dyn_offset, dyn_reference_index,
                             dyn_reference_value, dyn_reference_offset,
                             dyn_global_value, t_a, t_b, w_a * weight[entry_b], cross);
            }
        }
        for (unsigned r = 0; r < 3u; ++r) {
            for (unsigned c = 0; c < 3u; ++c) {
                block[3 * c + r] += cross[3 * c + r] + cross[3 * r + c];
            }
        }
    }
    for (unsigned e = 0; e < 9u; ++e) {
        diagonal[9 * row + e] += block[e];
    }
}

// The position half of one Cosserat edge's Schwarz block, for a scene with
// ties: the two vertices' diagonal blocks and the coupling between them,
// column-major, 27 floats per slot, read by `cosserat_schwarz_factor_row` in
// place of the vertex diagonal and the fixed coupling.
//
// AN EDGE WITH NO HELD END takes the REDUCED operator's blocks: the reduced
// diagonal the vertex blocks already hold (`tie_precond_row`) and the reduced
// coupling
//
//     R_uv = K_uv + sum_b w_b K_{u t_b} + sum_a w_a K_{t_a v}
//                 + sum_{a, b} w_a w_b K_{t_a t_b}
//
// over the held vertices `t_a` that `u` controls and `t_b` that `v` controls,
// `K_tt` being a held vertex's own diagonal block. The frame rows of the block
// are untouched, a frame coupling only to its own edge's two vertices, so the
// whole block is a principal submatrix of the reduced operator and is positive
// definite with it.
//
// AN EDGE WITH A HELD END takes the UNREDUCED blocks (`unreduced`, the
// operator's diagonal before the tie pass, and the fixed coupling the factor
// would have read), a principal submatrix of the full operator. Its held rows
// are zeroed after the solve, so the block only has to be positive definite.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void tie_edge_block_row(
    const Vec2u *edge_nodes,
    const unsigned *tie_of,
    const unsigned *tie_vertex,
    const float *weight,
    const unsigned *ctrl_offset,
    const unsigned *ctrl_entry,
    const unsigned *dof_mask,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    const unsigned *transpose_pair,
    const unsigned *transpose_offset,
    const unsigned *dyn_index,
    const float *dyn_value,
    const unsigned *dyn_offset,
    const unsigned *dyn_reference_index,
    const unsigned *dyn_reference_value,
    const unsigned *dyn_reference_offset,
    const float *dyn_global_value,
    const float *reduced,
    const float *unreduced,
    float *edge_block, unsigned slot) {
    const Vec2u nodes = edge_nodes[slot];
    const unsigned u = nodes[0];
    const unsigned v = nodes[1];
    float between[9];
    for (unsigned e = 0; e < 9u; ++e) {
        between[e] = 0.0f;
    }
    if (tie_of[u] != tie_none() || tie_of[v] != tie_none()) {
        for (unsigned k = offset[u]; k < offset[u + 1]; ++k) {
            if (index[k] == v) {
                for (unsigned e = 0; e < 9u; ++e) {
                    between[e] += value[9 * k + e];
                }
            }
        }
        for (unsigned entry = transpose_offset[u]; entry < transpose_offset[u + 1]; ++entry) {
            if (transpose_pair[2 * entry] == v) {
                const unsigned k = transpose_pair[2 * entry + 1];
                for (unsigned r = 0; r < 3u; ++r) {
                    for (unsigned c = 0; c < 3u; ++c) {
                        between[3 * c + r] += value[9 * k + 3 * r + c];
                    }
                }
            }
        }
        for (unsigned e = 0; e < 9u; ++e) {
            edge_block[27u * slot + e] = unreduced[9u * u + e];
            edge_block[27u * slot + 9u + e] = unreduced[9u * v + e];
            edge_block[27u * slot + 18u + e] = between[e];
        }
        return;
    }
    tie_coupling(index, offset, value, transpose_pair, transpose_offset,
                 dyn_index, dyn_value, dyn_offset, dyn_reference_index,
                 dyn_reference_value, dyn_reference_offset, dyn_global_value,
                 u, v, 1.0f, between);
    // A PRESCRIBED END keeps the coupling the Dirichlet pass left (cleared):
    // its row of the reduced operator is the identity, decoupled from every
    // other row, and the fold never reaches it.
    const bool prescribed = dof_mask[u] != 0u || dof_mask[v] != 0u;
    if (!prescribed) {
        for (unsigned b = ctrl_offset[v]; b < ctrl_offset[v + 1]; ++b) {
            const unsigned t_b = tie_vertex[ctrl_entry[b] / tie_slots()];
            tie_coupling(index, offset, value, transpose_pair, transpose_offset,
                         dyn_index, dyn_value, dyn_offset, dyn_reference_index,
                         dyn_reference_value, dyn_reference_offset, dyn_global_value,
                         u, t_b, weight[ctrl_entry[b]], between);
        }
        for (unsigned a = ctrl_offset[u]; a < ctrl_offset[u + 1]; ++a) {
            const unsigned t_a = tie_vertex[ctrl_entry[a] / tie_slots()];
            const float w_a = weight[ctrl_entry[a]];
            tie_coupling(index, offset, value, transpose_pair, transpose_offset,
                         dyn_index, dyn_value, dyn_offset, dyn_reference_index,
                         dyn_reference_value, dyn_reference_offset, dyn_global_value,
                         t_a, v, w_a, between);
            for (unsigned b = ctrl_offset[v]; b < ctrl_offset[v + 1]; ++b) {
                const unsigned t_b = tie_vertex[ctrl_entry[b] / tie_slots()];
                const float w_ab = w_a * weight[ctrl_entry[b]];
                if (t_a == t_b) {
                    for (unsigned e = 0; e < 9u; ++e) {
                        between[e] += w_ab * unreduced[9u * t_a + e];
                    }
                } else {
                    tie_coupling(index, offset, value, transpose_pair, transpose_offset,
                                 dyn_index, dyn_value, dyn_offset, dyn_reference_index,
                                 dyn_reference_value, dyn_reference_offset,
                                 dyn_global_value, t_a, t_b, w_ab, between);
                }
            }
        }
    }
    for (unsigned e = 0; e < 9u; ++e) {
        edge_block[27u * slot + e] = reduced[9u * u + e];
        edge_block[27u * slot + 9u + e] = reduced[9u * v + e];
        edge_block[27u * slot + 18u + e] = between[e];
    }
}
