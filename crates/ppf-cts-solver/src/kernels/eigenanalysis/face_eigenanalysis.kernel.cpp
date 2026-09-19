// File: face_eigenanalysis.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read, and the build hands
// each compiler its own form. The two facts a backend cannot infer are written
// as C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a reference parameter. MSL
// requires an address space on every reference; CUDA and the host have one
// address space and are handed the same declarations with it removed.
//
// No include of its own. The matrix and vector names, and linalg::eig::symm2x2,
// come from whatever declares them for the backend that is compiling, which is
// data.hpp under nvcc and on the host and the shader prologue's aliases under
// MSL. `fmath::abs`, `fmath::max` and `fmath::div` come from that backend's
// prologue in the same way: cpp/seam under nvcc and on the host, the prologue
// in metal/shader_compiler.mm under MSL. `fmath::div` is not decoration: it is
// `metal::precise::divide` on MSL, and the plain MSL operator is not correctly
// rounded.

// The two entry points, declared once each and rendered for four targets. Both
// are element gathers into one element scatter: the body takes the spectral
// terms of one face and returns its result, so the return value reaches memory
// the one way a return value can, at the thread index in the scatter buffer.
//
// `eps` sits where the body takes it, after `vt`, because a generated call
// forwards the declaration's order and the compiler reads the body's real
// signature: a parameter out of place does not compile.
[[seam::entry(force)]]
[[seam::device_fn]] inline Mat3x2f
face_spectral_force(const Vec2f &gradient_sigma,
                        const Mat3x2f &u,
                        const Mat2x2f &vt) {
    return u * gradient_sigma.asDiagonal() * vt;
}

[[seam::entry(hessian)]]
[[seam::device_fn]] inline Mat6x6f
face_spectral_hessian(const Vec2f &gradient_sigma,
                          const Mat2x2f &hessian_sigma,
                          const Mat3x2f &u2,
                          const Vec2f &sigma,
                          const Mat2x2f &vt, float eps) {
    Vec2f eigenvalues;
    Mat2x2f eigenvectors;
    linalg::eig::symm2x2(hessian_sigma, eigenvalues, eigenvectors);

    Mat3x3f u;
    u.col(0) = u2.col(0);
    u.col(1) = u2.col(1);
    u.col(2) = u2.col(0).cross(u2.col(1));

    constexpr float inverse_sqrt_two = 0.7071067811865475f;
    Mat3x2f modes[6];
    for (unsigned mode = 0; mode < 6; ++mode) {
        modes[mode] = Mat3x2f::Zero();
    }
    // Twist mode: ANTISYMMETRIC off-diagonal.
    modes[0](1, 0) = inverse_sqrt_two;
    modes[0](0, 1) = -inverse_sqrt_two;
    // Flip mode: SYMMETRIC off-diagonal, paired with the difference eigenvalue
    // (deda0 - deda1) / (a0 - a1) below, exactly like the 3x3 overload's
    // modes[3..5]. It must NOT be the antisymmetric twist matrix modes[0],
    // which would degenerate the mode basis (vec(u m0 vt) == vec(u m1 vt)): the
    // twist direction would receive lambda0 + lambda1 and the flip direction
    // zero, so the projected Hessian would be wrong in the flip subspace
    // whenever the singular values differ. Affects ARAP, StVK and SNHk shells
    // only, since BaraffWitkin does not take this path.
    modes[1](1, 0) = inverse_sqrt_two;
    modes[1](0, 1) = inverse_sqrt_two;
    modes[2](2, 0) = 1.0f;
    modes[3](2, 1) = 1.0f;
    modes[4](0, 0) = eigenvectors(0, 0);
    modes[4](1, 1) = eigenvectors(1, 0);
    modes[5](0, 0) = eigenvectors(0, 1);
    modes[5](1, 1) = eigenvectors(1, 1);

    const float difference = sigma[0] - sigma[1];
    float lambda[6];
    lambda[0] =
        fmath::max(0.0f, fmath::div(gradient_sigma[0] + gradient_sigma[1],
                                sigma[0] + sigma[1]));
    lambda[1] =
        fmath::max(0.0f,
                 fmath::abs(difference) > eps
                     ? fmath::div(gradient_sigma[0] - gradient_sigma[1],
                                difference)
                     : 0.5f * (hessian_sigma(0, 0) + hessian_sigma(1, 1)) -
                           0.5f *
                               (hessian_sigma(0, 1) + hessian_sigma(1, 0)));
    lambda[2] = fmath::max(0.0f, fmath::div(gradient_sigma[0], sigma[0]));
    lambda[3] = fmath::max(0.0f, fmath::div(gradient_sigma[1], sigma[1]));
    lambda[4] = fmath::max(0.0f, eigenvalues[0]);
    lambda[5] = fmath::max(0.0f, eigenvalues[1]);

    Mat6x6f result = Mat6x6f::Zero();
    for (unsigned mode = 0; mode < 6; ++mode) {
        if (lambda[mode] > 0.0f) {
            const Mat3x2f transformed = u * modes[mode] * vt;
            Vec6f vector;
            for (unsigned element = 0; element < 6; ++element) {
                vector[element] = transformed.m[element];
            }
            result += lambda[mode] * vector * vector.transpose();
        }
    }
    return result;
}
