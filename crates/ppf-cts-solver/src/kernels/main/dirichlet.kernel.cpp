// File: dirichlet.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

#include "dx_seed.kernel.cpp"

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. `[[seam::device_fn]]` is the execution
// space, `[[seam::thread]]` and `[[seam::device]]` the address spaces of a
// reference and of a pointer parameter; `Vec3f`, `Vec3f`, `Mat3x3f`, `map` and
// `compute::atomic_add` arrive from whichever backend prologue is in scope.
//
// ELIMINATING A PRESCRIBED DEGREE OF FREEDOM FROM THE NEWTON SYSTEM, AND THE
// LIFTING TERM THAT MAKES IT CORRECT.
//
// With `M dx = f` and `dx_i = p_i` prescribed on a set of removed vertices, the
// exact reduction of a FREE row `j` is
//
//     sum_{k free} M_jk dx_k  =  f_j  -  sum_{k removed} M_jk p_k
//                                       ^^^^^^^^^^^^^^^^^^^^^^^^^
//                                       the Dirichlet LIFTING term
//
// so the pass is three steps:
//
//   1. LIFT. Before dropping any coupling block M_jk to a removed vertex k,
//      move its known contribution to the right-hand side of the free row j.
//   2. ELIMINATE. Zero every stored block whose row OR column is removed.
//   3. PRESCRIBE. Set the removed row's diagonal to the identity and its
//      right-hand side to the prescribed increment.
//
// THE LIFT IS THE HALF THAT IS EASY TO LOSE AND EXPENSIVE TO LOSE. It is the
// only signal a free vertex gets that a prescribed one is advancing INTO it:
// the coupling Hessian block times the prescribed increment is precisely the
// "get out of the way by p_k" forcing. Omit it and the free side has nothing
// but the barrier gradient, whose Newton step is bounded by grad/curv = ghat/2
// however stiff the barrier is. A collider commanded further than that in one
// step then out-runs the free side every iteration: the gap closes onto the
// ACCD parking distance, the line search clamps toi to nearly zero to stop the
// penetration, that same clamp throttles the free side's escape, and the loop
// re-assembles a bit-identical system forever. The signature is `toi` and the
// residual bit-identical for a hundred thousand iterations with `toi` still far
// above FLT_EPSILON. A STATIONARY pin has p_k = 0, so the term vanishes and the
// defect is invisible on every scene whose colliders do not move.
//
// BOTH MATRICES STORE ONLY THE UPPER TRIANGLE, which is what the two directions
// below are for. A coupling appears once, in the row with the smaller index, so
// whichever of the two rows owns the stored block must lift the OTHER one, and
// it does so through the block's TRANSPOSE. The lift scatters across rows, so
// it accumulates atomically: a free row can be lifted by several removed
// columns and by several removed rows at once.

// One coupling: the free row's share of a removed vertex's prescribed motion.
//
// `transposed` selects which of the two rows the stored block is being read
// for. The stored block M_ij lifts row i by M_ij p_j, and lifts row j by
// (M_ij)^T p_i.
//
// Written out in column-major rather than routed through a matrix-vector
// operator: `Mat3x3f` is column major, so `block(r, c)` is element `c * 3 + r`,
// and spelling the nine products keeps the operand order explicit at the one
// site where getting the transpose backwards would be a silent sign error in
// the only term that couples the two sides.
[[seam::device_fn]] inline Vec3f
dirichlet_coupling(const Mat3x3f &block,
                       const Vec3f &prescribed,
                       bool transposed) {
    Vec3f contribution;
    if (transposed) {
        contribution[0] = block(0, 0) * prescribed[0] +
                          block(1, 0) * prescribed[1] +
                          block(2, 0) * prescribed[2];
        contribution[1] = block(0, 1) * prescribed[0] +
                          block(1, 1) * prescribed[1] +
                          block(2, 1) * prescribed[2];
        contribution[2] = block(0, 2) * prescribed[0] +
                          block(1, 2) * prescribed[1] +
                          block(2, 2) * prescribed[2];
    } else {
        contribution[0] = block(0, 0) * prescribed[0] +
                          block(0, 1) * prescribed[1] +
                          block(0, 2) * prescribed[2];
        contribution[1] = block(1, 0) * prescribed[0] +
                          block(1, 1) * prescribed[1] +
                          block(1, 2) * prescribed[2];
        contribution[2] = block(2, 0) * prescribed[0] +
                          block(2, 1) * prescribed[1] +
                          block(2, 2) * prescribed[2];
    }
    return contribution;
}

// Step 1: move one coupling's known contribution onto the free row's
// right-hand side, with the sign the reduction asks for.
//
// Atomic because the destination is another row: a free row can receive a lift
// from several removed columns in its own row walk AND from several removed
// rows lifting it by transpose in theirs.
//
// `compute::atomic_float_t` is the seam's own name for the destination's
// element type, `float` under nvcc and on the host and `atomic_float` under
// MSL, and on MSL the address space is part of that pointer type. A `float *`
// here compiles on two backends and fails on the third: every
// `compute::atomic_add` overload the shader prologue declares takes an
// atomic-typed pointer, and none of them accepts a plain `device float *`.
[[seam::device_fn]] inline void
dirichlet_lift(compute::atomic_float_t *force, unsigned row,
                   const Mat3x3f &block,
                   const Vec3f &prescribed, bool transposed) {
    const Vec3f contribution =
        dirichlet_coupling(block, prescribed, transposed);
    compute::atomic_add(force + 3 * row + 0, -contribution[0]);
    compute::atomic_add(force + 3 * row + 1, -contribution[1]);
    compute::atomic_add(force + 3 * row + 2, -contribution[2]);
}

// Step 2: drop a coupling whose row or column is a removed vertex.
//
// ONE WRITE CLEARS BOTH SCATTER DIRECTIONS. Each stored value aliases the
// canonical buffer slot the transpose mirror reads, so zeroing the upper
// triangle's block removes the coupling from row i AND from column i. Writing
// the zero rather than skipping the entry is what keeps the sparsity pattern,
// the preconditioner's view of it and the transpose index all consistent: the
// pattern is rebuilt from demand, not from which entries happen to be nonzero.
[[seam::device_fn]] inline void
dirichlet_clear(Mat3x3f &block) {
    block = Mat3x3f::Zero();
}

// Step 3: the removed row becomes the identity carrying its prescribed
// increment.
//
// This must run in a SEPARATE pass from the lift, not merely later in the same
// one: a removed row's right-hand side must carry the increment and NOTHING
// else, and the lifts above are atomic scatters that can still be in flight
// from any other row. Interleaving them would let a lift land on a row that has
// already been prescribed.
//
// The seed loaded into `dx` before the solve is the same value from the same
// body, so the initial residual on this row is exactly zero, and the eliminated
// column keeps `(A p)` zero on it for every PCG iteration afterwards. The
// preconditioner needs no special case either: it is rebuilt from the diagonal,
// and the inverse of the identity is the identity.
[[seam::device_fn]] inline void
dirichlet_prescribe(Mat3x3f &diagonal,
                        float *force, unsigned row,
                        const Vec3f &current,
                        const Vec3f &target) {
    diagonal = Mat3x3f::Identity();
    map<Vec3f>(force + 3 * row) = prescribed_increment(current, target);
}

// THE GATE THE LAUNCHER HELD. A row this pass does not remove is left exactly
// as the assembly wrote it, which is why the refusal is a bare return rather
// than a zero fill: unlike a diff table, there is nothing here whose stale
// bytes could be read.
//
// `dof_mask` ARRIVES AS AN ELEMENT rather than as a buffer, so the entry point
// holds the only subscript. It is the same mask `dirichlet_dof_removed`
// below decides, materialized once per step by the caller.
//
// `diagonal` IS A THREAD-SPACE BLOCK AND `force` IS NOT. The block is this
// row's own 3x3 and a generated entry gathers it, which on MSL means a copy in
// and a copy back; the force is a flat array the body addresses at `3 * row`,
// so it stays a base pointer with the thread index forwarded beside it. That
// asymmetry is the declaration's, not the body's.
[[seam::device_fn]] inline void dirichlet_prescribe_gated(
    unsigned dof_mask, Mat3x3f &diagonal,
    float *force, unsigned row,
    const Vec3f &current,
    const Vec3f &target) {
    if (dof_mask == 0u) {
        return;
    }
    dirichlet_prescribe(diagonal, force, row, current, target);
}

// The entry point. Three element gathers read, one non-const element gather
// written, the force as a base pointer and the thread index forwarded to it,
// and no scatter, because the body returns void and writes through what it is
// handed.
[[seam::entry(count, row)]] void dirichlet_prescribe_gated(
    const unsigned *dof_mask,
    Mat3x3f *diagonal,
    float *force,
    unsigned row,
    const Vec3f *eval_x,
    const Vec3f *target,
    unsigned count);

// Whether one vertex's degree of freedom is the one this pass removes.
//
// TWO FACTS, AND THE SECOND IS THE EXCEPTION THAT MAKES THE FIRST SAFE. A fix
// pin is an exact Dirichlet boundary condition, so its row is eliminated. A
// vertex INSIDE A RIGID BODY owns no per-vertex degree of freedom at all: the
// solve is reduced through the rigid Jacobian and the body is refitted
// afterwards, so a per-vertex Dirichlet row is not representable and the pin
// keeps its barrier instead. That anchor barrier is the only penalty pin left
// in the solver, and this predicate is what decides which pins get it.
//
// It is one function because the same test is asked in three places that must
// agree: the mask this pass runs over, the contact assembly deciding whether to
// put the pin barrier back, and the position update deciding whether a vertex
// is placed exactly rather than stepped. Two of the three failing to agree is
// silent: a pin would have neither a Dirichlet row nor a penalty, and would be
// held by nothing at all.
[[seam::device_fn]] inline bool
dirichlet_dof_removed(unsigned fix_index, unsigned pdrd_body_index) {
    return fix_index > 0 && pdrd_body_index == 0;
}

// B16, PASS ONE, AS A NEUTRAL BODY: walk one row of the FIXED matrix's stored
// upper triangle, lift a free row by every prescribed coupling it carries, and
// clear every coupling with a prescribed side.
//
// EVERY BUFFER IS A BASE POINTER AND THE ROW INDEX IS FORWARDED, which is what
// a walk over `offset[row] .. offset[row + 1]` needs: the slot range is data,
// so no index list can name it and no bound can be checked against it.
// A write at a data-driven slot is expressible this way, and the paragraph
// below is the reason being expressible is not the same as being safe to cut.
//
// SERIAL BY CONTRACT, AND THE DECLARATION CANNOT SAY SO. `dirichlet_lift`
// folds into ANOTHER row's force, which the host seam spells as a plain read,
// add and write back, so two threads walking two rows that couple to one free
// row would race. The rule therefore lives in the kernel table's
// `Scatter::Atomic` and the conversion carries it across unchanged.
//
// THE COUPLING AND THE TWO POSITIONS ARE COPIED INTO THREAD SPACE before they
// are handed on, because `dirichlet_lift` and `prescribed_increment`
// take `[[seam::thread]]` references and these are device elements. That is an
// address space, not a value: MSL is the one target with more than one, and on
// the other two the annotation is erased.
//
// THE LIFT AND THE CLEAR ARE ORDERED WITHIN A SLOT and must stay so: the lift
// reads the coupling and the clear zeroes it, so clearing first would lift by
// zero and lose the term.
//
// THE DYNAMIC (CONTACT) MATRIX HAS ITS OWN PASS AND IT IS NOT HERE. The two
// walks commute, a lift being an atomic add and a clear an idempotent store, so
// the row's final force is the same sum either way.
[[seam::entry(row)]]
[[seam::device_fn]] inline void dirichlet_lift_row(
    const unsigned *dof_mask,
    const unsigned *offset,
    const unsigned *column,
    Mat3x3f *value,
    compute::atomic_float_t *force,
    const Vec3f *eval_x,
    const Vec3f *target, unsigned row) {
    const bool mask_row = dof_mask[row] != 0u;
    for (unsigned k = offset[row]; k < offset[row + 1]; ++k) {
        const unsigned other = column[k];
        const bool mask_other = dof_mask[other] != 0u;
        if (mask_row != mask_other) {
            const Mat3x3f coupling = value[k];
            if (mask_other) {
                const Vec3f current = eval_x[other];
                const Vec3f goal = target[other];
                dirichlet_lift(force, row, coupling,
                                   prescribed_increment(current, goal),
                                   false);
            } else {
                const Vec3f current = eval_x[row];
                const Vec3f goal = target[row];
                dirichlet_lift(force, other, coupling,
                                   prescribed_increment(current, goal),
                                   true);
            }
        }
        if (mask_row || mask_other) {
            dirichlet_clear(value[k]);
        }
    }
}
