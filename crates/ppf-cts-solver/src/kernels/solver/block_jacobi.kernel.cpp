// File: block_jacobi.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::thread]]` is the address space of a reference parameter,
// which MSL requires on every reference and pointer type.
//
// The per-vertex block-Jacobi diagonal inverse. It operates on thread-space
// VALUES: no launch geometry, no reduction, no atomic, no address space other
// than `[[seam::thread]]`, so the backends share the arithmetic and each keeps
// its own entry point and its own diagnostic channel. The division goes through
// `fmath::div` rather than the operator, because MSL spells the correctly
// rounded quotient `precise::divide` and the plain operator is a different
// function there.
//
// WHY THE INVERSE GOES THROUGH THE EIGENDECOMPOSITION. The block is SPD by
// assembly (a strict mass/dt^2 inertia floor plus PSD-projected elastic,
// bending, contact and friction blocks), but it is not well conditioned: in a
// tight drape the elasticity-inclusive dynamic barrier stiffness ~mass/gap^2
// pushes the normal-direction stiffness to ~1e11 against a tangential floor
// ~1e2, a condition number ~1e9. A raw cofactor inverse then forms the
// tangential cofactors as differences of huge*floor products whose surviving
// digits fall below fp32 epsilon, so it can return a garbage or even
// sign-flipped, non-SPD block, which corrupts the PCG search direction.
// symm3x3 max-abs-scales the block, so no cross-magnitude cancellation occurs,
// and each 1/lambda_k is formed independently, which makes the result SPD and
// bounded at any conditioning. Eigenvalues are floored because symm3x3 cannot
// resolve one below about eps*lambda_max; the floor keeps the preconditioner SPD
// with a bounded condition number and only affects preconditioning quality, never
// correctness, in that unresolvable subspace. A well-conditioned block is
// unaffected: no eigenvalue is clamped and the reconstruction agrees with the
// cofactor inverse.
//
// WHY THE VERDICT IS RETURNED RATHER THAN ASSERTED HERE. The two conditions the
// caller must trap (a non-finite or non-positive largest eigenvalue) mean the
// assembly upstream is broken, and each backend reports that through its own
// channel: CUDA through a live release `DIAG_ASSERT4` on `diagnostics`, Metal
// through the 32-byte diagnostic record its dispatch reads back. Neither channel
// has a spelling the other compiles, so the shared body returns `lambda_max` and
// a `valid` flag and the entry point raises. A caller must NOT use the inverse
// when `valid` is false: below zero the floor is non-positive and the reciprocal
// is unbounded, which is why the flag exists rather than a clamp that would hand
// back a plausible block for a broken system.

#pragma once

#include "../linalg/eigsolve.hpp"
#include "../linalg/smat.hpp"

// The two names the bodies below use, declared here so this body stands on
// the header-only linalg alone. data.hpp declares the same aliases for the same
// underlying types, and the shader's alias segment declares them again, so any
// arrival order is fine: an alias redeclaration naming an identical type is
// legal, and linalg::SVec<T, N> IS linalg::SMat<T, N, 1>.
using Vec3f = linalg::SVec<float, 3>;
using Mat3x3f = linalg::SMat<float, 3, 3>;

// What one block inversion produced. `lambda_max` and the three eigenvalues are
// carried out so the caller's assert can report the numbers that decided the
// verdict rather than a bare failure.
struct BlockJacobiInverse {
    Mat3x3f inverse;
    Vec3f eigenvalue;
    float lambda_max;
    bool valid;
    // WHETHER THE INPUT BLOCK WAS FINITE, which `valid` cannot answer. Carried
    // separately so the caller reports the right fault; see the note on the
    // test below.
    bool finite_input;
    // The first non-finite entry, so the caller's report names a number the
    // reader can act on rather than a bare verdict.
    float offending_entry;
};

[[seam::device_fn]] inline BlockJacobiInverse
block_jacobi_invert(const Mat3x3f &m) {
    BlockJacobiInverse result;
    // THE FINITENESS TEST BELONGS ON THE INPUT, NOT ON THE EIGENVALUES, and the
    // eigensolver is why. `linalg::eig::symm3x3` reduces its scale with
    // `fmath::max` from 0.0f, and IEEE 754 maxNum returns the NON-NaN operand,
    // so an all-NaN block leaves that scale at exactly zero and takes the
    // `scale <= 0.0f` early return: eigenvalues (0, 0, 0) and an identity
    // basis. `lmax` below therefore comes back a finite zero and the finiteness
    // half of `valid` sees nothing wrong, leaving only the positivity half to
    // fire and report a NON-POSITIVE block when the real fault is a NON-FINITE
    // one. The same `fmath::max` property hides a NaN behind any real operand
    // in the reduction over the spectrum, so the blindness is there twice.
    // `tests/kernels/eigen_nan_input.cpp` is the gate on both halves.
    //
    // It is a SEPARATE verdict from `valid` rather than folded into it because
    // the two name different faults and the caller reports different numbers
    // for them: a non-finite input has no meaningful spectrum to quote.
    result.finite_input = true;
    result.offending_entry = 0.0f;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            if (fmath::isnan(m(a, b)) || fmath::isinf(m(a, b))) {
                if (result.finite_input) {
                    result.offending_entry = m(a, b);
                }
                result.finite_input = false;
            }
        }
    }
    // Enforce exact fp32 symmetry (the assembled block is symmetric only up to
    // atomic-accumulation order) before the eigensolve.
    Mat3x3f sym;
    for (int a = 0; a < 3; ++a) {
        sym(a, a) = m(a, a);
        for (int b = a + 1; b < 3; ++b) {
            float s = 0.5f * (m(a, b) + m(b, a));
            sym(a, b) = s;
            sym(b, a) = s;
        }
    }
    Vec3f lambda;
    Mat3x3f Q; // columns are the (ascending) eigenvectors
    linalg::eig::symm3x3(sym, lambda, Q);
    float lmax = fmath::max(lambda[0], fmath::max(lambda[1], lambda[2]));
    result.eigenvalue = lambda;
    result.lambda_max = lmax;
    // This second test still covers the EIGENSOLVE itself: `finite_input` above
    // clears the input, so a non-finite lmax from a finite input would be a
    // fault in `symm3x3` rather than upstream of it.
    result.valid = result.finite_input && !fmath::isnan(lmax) &&
                   !fmath::isinf(lmax) && lmax > 0.0f;
    Mat3x3f minv = Mat3x3f::Zero();
    if (!result.valid) {
        // A zero block, not a floored reciprocal of a non-positive eigenvalue.
        // The caller raises on `valid` and discards the dispatch, and a zero
        // here keeps the failing dispatch from also writing infinities into a
        // buffer a later pass would read.
        result.inverse = minv;
        return result;
    }
    const float lambda_floor = lmax * 1.0e-6f;
    for (int k = 0; k < 3; ++k) {
        float inv_lk = fmath::div(1.0f, fmath::max(lambda[k], lambda_floor));
        // Rank-1 q q^T accumulation with raw scalar arithmetic (a device Eigen
        // matrix-vector product or outer product is silently wrong on CUDA;
        // scalar fills are safe).
        float q0 = Q(0, k), q1 = Q(1, k), q2 = Q(2, k);
        minv(0, 0) += inv_lk * q0 * q0;
        minv(0, 1) += inv_lk * q0 * q1;
        minv(0, 2) += inv_lk * q0 * q2;
        minv(1, 1) += inv_lk * q1 * q1;
        minv(1, 2) += inv_lk * q1 * q2;
        minv(2, 2) += inv_lk * q2 * q2;
    }
    minv(1, 0) = minv(0, 1);
    minv(2, 0) = minv(0, 2);
    minv(2, 1) = minv(1, 2);
    result.inverse = minv;
    return result;
}

// ONE ROW'S INVERSION, dispatched over the whole diagonal.
//
// The composition between the entry point and `block_jacobi_invert` above: it
// reads the row's 3x3 out of the flat block array, hands it to the shared body,
// and writes the result back into the flat inverse array. Both arrays are
// column-major per block, nine floats each, which is the layout the operator's
// preconditioner pass writes and the PCG's preconditioner reads.
//
// THE VERDICT COMES BACK THROUGH THE DIAGNOSTIC CHANNEL rather than through a
// return value, because it is guarantee-class: a block that is not positive
// definite means the assembled Hessian is NaN, infinite or indefinite, which is
// an assembly defect upstream rather than a tolerance to widen. The payload
// carries the numbers that decided it, the largest eigenvalue and the two
// extremes of the spectrum, beside the row, so the host's message can name them
// as the per-row host loop did.
//
// A FAILING ROW STILL WRITES, and it writes the zero block the shared body
// returns rather than a floored reciprocal. The dispatch is discarded by the
// caller on any failure, and a zero keeps a later pass from reading infinities
// out of a buffer this one abandoned.
[[seam::entry(i)]]
[[seam::device_fn]] inline void
block_jacobi_invert_row(const float *diagonal,
                        float *inverse, unsigned i,
                        DiagHandle diag) {
    Mat3x3f m;
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            m(r, c) = diagonal[9 * i + 3 * c + r];
        }
    }
    const BlockJacobiInverse result = block_jacobi_invert(m);
    // THE INPUT FAULT IS REPORTED FIRST AND ON ITS OWN NUMBERS. A block that
    // arrived NaN from the elastic stencil has no spectrum worth quoting, and
    // reporting it through the positivity assert below would name a
    // non-positive block for a fault that is nothing of the kind.
    DIAG_ASSERT4(diag, result.finite_input, result.offending_entry,
                 result.lambda_max, (float)i, 0.0f);
    DIAG_ASSERT4(diag, result.valid, result.lambda_max, result.eigenvalue[0],
                 result.eigenvalue[2], (float)i);
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            inverse[9 * i + 3 * c + r] = result.inverse(r, c);
        }
    }
}
