// File: eigen_nan_input.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
// Gate for the premise the block-Jacobi guard in solver.cu rests on: a
// finiteness test on the EIGENVALUES cannot see a non-finite input matrix, so
// the test has to be on the matrix itself.
//
// `block_jacobi_invert_row` inverts one 3x3 diagonal block per vertex through
// `linalg::eig::symm3x3`. An all-NaN
// block reached it in issue #144 from the elastic stencil, and a guard built
// out of `isfinite(lmax)` and `lmax > 0` names neither fault: the first passes,
// and the second then reports a non-positive block when the block is NaN in
// every entry.
//
// One mechanism defeats it, at two call sites, and this asserts both:
//
//   1. `symm3x3` reduces its scale as `scale = fmaxf(scale, fabsf(A.m[i]))`
//      starting from 0.0f. `fmaxf` returns the non-NaN operand (IEEE 754
//      maxNum), so an ALL-NaN matrix leaves `scale` at exactly 0.0f and takes
//      the `scale <= 0.0f` early return, which yields eigenvalues (0, 0, 0) and
//      an identity basis. The Cardano path is never reached, so its acos clamp
//      is not what produces this. That early return is load-bearing and must
//      not be removed: it is what keeps an all-NaN block out of Cardano, whose
//      `fmath::cos_bounded` carries a live assert of its own.
//   2. The same `fmaxf` property applies again in the inverter's own reduction
//      over the three eigenvalues, so even a spectrum that DOES carry a NaN
//      reduces to a finite maximum and passes a finiteness test on it.
//
// The exact eigenvalues of a NaN matrix are not a contract worth pinning, so
// what is asserted is only that they are not a reliable NaN detector: case 1
// requires them to be finite, which is the property that defeats a
// spectrum-side guard. Should a future `symm3x3` propagate NaN instead, this
// test fails and says so, and the input-side test in the inverter stays correct
// either way.
//
// This covers the ALL-NaN block, which is what issue #144 produced. A PARTLY
// NaN block keeps a positive scale, reaches Cardano, and traps in
// `fmath::cos_bounded` rather than arriving here quietly.
//
// WHY IT NEEDS NO DEVICE. Everything it asserts is a property of fp32
// arithmetic that a host compiler reads out of the same neutral header: the
// `scale <= 0.0f` early return in `symm3x3`, and IEEE 754 maxNum in `fmaxf`.
// There is no warp or threadgroup operation anywhere in it. The one figure a
// device could move is which subnormals flush, and nothing here is asserted
// against a subnormal.
//
// Usage:
//   cargo test --test kernel_gates

#include "linalg/eigsolve.hpp"
#include <cmath>
#include <cstdio>

enum Result {
    NAN_LAMBDA_0,
    NAN_LAMBDA_1,
    NAN_LAMBDA_2,
    NAN_LMAX,
    NAN_SCALE,
    NAN_Q_TRACE,
    FMAXF_OVER_ONE_NAN,
    FMAXF_OVER_TWO_NAN,
    GOOD_LMAX,
    GOOD_LMIN,
    N_RESULT,
};

using M3 = linalg::eig::M3;
using V3 = linalg::eig::V3;

static void run(float *out) {
    // The block issue #144 delivered: every entry NaN, which is what an
    // elastic Hessian assembled from a rest matrix with no significant digits
    // left reduces to.
    M3 nan_block;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            nan_block(r, c) = nanf("");
        }
    }
    V3 lambda;
    M3 Q;
    linalg::eig::symm3x3(nan_block, lambda, Q);
    out[NAN_LAMBDA_0] = lambda[0];
    out[NAN_LAMBDA_1] = lambda[1];
    out[NAN_LAMBDA_2] = lambda[2];
    // The max-side reduction a spectrum-side guard would test.
    out[NAN_LMAX] = fmaxf(lambda[0], fmaxf(lambda[1], lambda[2]));
    // The MECHANISM, not just its symptom: symm3x3's own scale reduction, run
    // here over the same block. It staying at zero is what routes the block to
    // the `scale <= 0.0f` early return, and an identity basis (trace 3) is
    // that branch's tell.
    float scale = 0.0f;
    for (int i = 0; i < 9; ++i) {
        scale = fmaxf(scale, fabsf(nan_block.m[i]));
    }
    out[NAN_SCALE] = scale;
    out[NAN_Q_TRACE] = Q(0, 0) + Q(1, 1) + Q(2, 2);

    // The second blindness, independent of the eigensolver: NaN loses to a
    // real operand in fmaxf, so a partly-NaN triple still reduces to a finite
    // maximum.
    const float qnan = nanf("");
    out[FMAXF_OVER_ONE_NAN] = fmaxf(-5.0f, fmaxf(qnan, qnan));
    out[FMAXF_OVER_TWO_NAN] = fmaxf(qnan, fmaxf(qnan, -5.0f));

    // Control: a well-formed SPD block is unaffected by any of this, so the
    // input-side test costs a healthy vertex nothing. diag(2, 3, 7), which the
    // solver sorts ascending.
    M3 good = M3::Zero();
    good(0, 0) = 2.0f;
    good(1, 1) = 3.0f;
    good(2, 2) = 7.0f;
    linalg::eig::symm3x3(good, lambda, Q);
    out[GOOD_LMAX] = fmaxf(lambda[0], fmaxf(lambda[1], lambda[2]));
    out[GOOD_LMIN] = fminf(lambda[0], fminf(lambda[1], lambda[2]));
}

int main() {
    float r[N_RESULT] = {};
    run(r);

    int failures = 0;

    // 1. The eigenvalues of a NaN matrix are finite, so they cannot be used to
    //    detect one.
    for (int i = NAN_LAMBDA_0; i <= NAN_LAMBDA_2; ++i) {
        if (!std::isfinite(r[i])) {
            fprintf(stderr,
                    "symm3x3 now propagates NaN into eigenvalue %d (%g). The "
                    "comment in solver::invert that says it does not is stale; "
                    "the input-side test there stays correct regardless.\n",
                    i - NAN_LAMBDA_0, r[i]);
            ++failures;
        }
    }
    if (!std::isfinite(r[NAN_LMAX])) {
        fprintf(stderr, "lmax over a NaN block is %g, expected finite\n",
                r[NAN_LMAX]);
        ++failures;
    }

    // 2. The mechanism: the scale reduction stays at zero, so the early
    //    return fires and hands back an identity basis. If this changes, the
    //    block reaches Cardano and traps in fmath::cos_bounded instead, which
    //    is a different failure with a different remedy.
    if (r[NAN_SCALE] != 0.0f) {
        fprintf(stderr,
                "symm3x3's scale reduction over an all-NaN block is %g, not 0; "
                "the `scale <= 0.0f` early return no longer fires and the "
                "block now reaches the Cardano path\n",
                r[NAN_SCALE]);
        ++failures;
    }
    if (fabsf(r[NAN_Q_TRACE] - 3.0f) > 1e-5f) {
        fprintf(stderr,
                "the eigenvector basis for an all-NaN block has trace %g, not "
                "3; the `scale <= 0.0f` early return is not what produced "
                "these eigenvalues\n",
                r[NAN_Q_TRACE]);
        ++failures;
    }

    // 3. fmaxf hides a NaN behind any real operand, in either order.
    if (r[FMAXF_OVER_ONE_NAN] != -5.0f || r[FMAXF_OVER_TWO_NAN] != -5.0f) {
        fprintf(stderr,
                "fmaxf no longer returns the non-NaN operand (%g, %g); the "
                "second place a max-side finiteness test goes blind is gone\n",
                r[FMAXF_OVER_ONE_NAN], r[FMAXF_OVER_TWO_NAN]);
        ++failures;
    }

    // 4. Control.
    if (fabsf(r[GOOD_LMAX] - 7.0f) > 1e-5f || fabsf(r[GOOD_LMIN] - 2.0f) > 1e-5f) {
        fprintf(stderr, "diag(2,3,7) eigenvalues: max %g min %g, expected 7 and 2\n",
                r[GOOD_LMAX], r[GOOD_LMIN]);
        ++failures;
    }

    if (failures) {
        return 1;
    }
    printf("eigenvalues of a NaN block are finite (%g, %g, %g), lmax %g: a "
           "max-side finiteness test cannot see a NaN input\n",
           r[NAN_LAMBDA_0], r[NAN_LAMBDA_1], r[NAN_LAMBDA_2], r[NAN_LMAX]);
    return 0;
}
