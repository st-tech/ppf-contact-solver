// File: tet_hessian_agreement.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
// THE TWO TET HESSIANS MUST AGREE, AND ONE OF THEM HAS NEVER BEEN DISPATCHED.
//
// `eigenanalysis/tet_eigenanalysis.kernel.cpp` carries two forms of the same
// matrix. `tet_spectral_hessian` has an entry point and the staged assembly
// dispatches it, then `utility/tet_convert.kernel.cpp`'s `tet_convert_hessian`
// turns its 9x9 into the element's 12x12 and the caller scales by the mass.
// `tet_spectral_hessian_fused` beside it folds all three into one accumulate
// and has NO entry anywhere in the tree, so nothing has ever run it.
//
// The comment above the fused body asserts the two are byte-identical
// "expression for expression and in the same order". That assertion is what
// this gate turns into a measurement, and there is a reason to doubt it:
// `c4bcd850` reverted the driver at the fused path for losing SPD-by-assembly
// on `examples/cards`, which failed at frame 24 with the Newton Hessian not SPD
// at a relative residual of 11.69. The staged chain does not fail there. If the
// two Hessians differ, that difference is the defect, and it is the one part of
// the fused path no dispatch has ever exercised.
//
// The comparison is EXACT on the structure and tolerant on the last bits: the
// two forms reach the same value by different association, the fused one
// accumulating a 12x12 in place where the staged pair forms a 9x9 and converts
// it, so a difference of a few ulps is arithmetic and a difference in the
// symmetric part or in a whole block is not.

#include "data.hpp"
#include "linalg/eigsolve.hpp"
#include "eigenanalysis/tet_eigenanalysis.kernel.cpp"
#include "utility/tet_convert.kernel.cpp"

#include <cmath>
#include <cstdio>

namespace {

// A deterministic spread of inputs, so a failure names the same case twice.
unsigned seed = 12345u;
float next_float(float lo, float hi) {
    seed = seed * 1664525u + 1013904223u;
    const float unit = static_cast<float>((seed >> 8) & 0xFFFFFFu) / 16777215.0f;
    return lo + unit * (hi - lo);
}

Mat3x3f random_rotation() {
    // A rotation built from three Givens turns, which keeps it orthonormal to
    // round-off without needing a factorization here.
    Mat3x3f r = Mat3x3f::Identity();
    for (unsigned axis = 0; axis < 3; ++axis) {
        const float t = next_float(-3.0f, 3.0f);
        const float c = std::cos(t), s = std::sin(t);
        Mat3x3f g = Mat3x3f::Identity();
        const unsigned a = axis, b = (axis + 1u) % 3u;
        g(a, a) = c;  g(a, b) = -s;
        g(b, a) = s;  g(b, b) = c;
        r = r * g;
    }
    return r;
}

int failures = 0;

void check_case(unsigned which) {
    Vec3f sigma;
    for (unsigned k = 0; k < 3; ++k) sigma[k] = next_float(0.35f, 1.9f);
    const Mat3x3f u = random_rotation();
    const Mat3x3f vt = random_rotation();
    Vec3f gradient_sigma;
    for (unsigned k = 0; k < 3; ++k) gradient_sigma[k] = next_float(-2.0f, 2.0f);
    // The spectral Hessian is symmetric by construction where it comes from,
    // so the input this gate hands both forms is symmetrized too.
    Mat3x3f hessian_sigma;
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            hessian_sigma(i, j) = next_float(-1.5f, 2.5f);
    hessian_sigma = 0.5f * (hessian_sigma + hessian_sigma.transpose());
    Mat3x3f inverse_rest;
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            inverse_rest(i, j) = next_float(-1.2f, 1.2f);
    const float mass = next_float(0.2f, 3.0f);
    const float eps = 1.0e-6f;

    // THE STAGED PAIR, which the driver dispatches today.
    const Mat9x9f spectral = tet_spectral_hessian(gradient_sigma, hessian_sigma,
                                                  u, sigma, vt, eps);
    // The mass is applied INSIDE the converter now, which is where the fused
    // form applies it too, so the two sides stay the same expression.
    const Mat12x12f staged = tet_convert_hessian(spectral, inverse_rest, mass);

    // THE FUSED ONE, which nothing dispatches. It ACCUMULATES, so it is handed
    // a zero the way its caller hands it one.
    Mat12x12f fused = Mat12x12f::Zero();
    tet_spectral_hessian_fused(gradient_sigma, hessian_sigma, u, sigma, vt, eps,
                               inverse_rest, mass, fused);

    float worst = 0.0f, scale = 0.0f;
    for (unsigned i = 0; i < 12; ++i) {
        for (unsigned j = 0; j < 12; ++j) {
            worst = std::fmax(worst, std::fabs(staged(i, j) - fused(i, j)));
            scale = std::fmax(scale, std::fabs(staged(i, j)));
        }
    }
    const float tolerance = 1.0e-4f * std::fmax(scale, 1.0f);
    if (!(worst <= tolerance)) {
        ++failures;
        std::printf("case %u: the two tet Hessians differ by %.6g, tolerance "
                    "%.6g, |staged|max %.6g\n", which, worst, tolerance, scale);
    }
}

}  // namespace

int main() {
    for (unsigned which = 0; which < 64; ++which) {
        check_case(which);
    }
    if (failures == 0) {
        std::printf("the staged and fused tet Hessians agree on 64 cases\n");
        return 0;
    }
    std::printf("%d of 64 cases disagree\n", failures);
    return 1;
}
