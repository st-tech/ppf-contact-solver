// File: tet_eigenanalysis.kernel.cpp
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
// No include of its own. The matrix and vector names, and linalg::eig::symm3x3,
// come from whatever declares them for the backend that is compiling, which is
// data.hpp under nvcc and on the host and the shader prologue's aliases under
// MSL. `fmath::abs`, `fmath::max` and `fmath::div` come from that backend's
// prologue in the same way: cpp/seam under nvcc and on the host, the prologue
// in metal/shader_compiler.mm under MSL. `fmath::div` is not decoration: it is
// `metal::precise::divide` on MSL, and the plain MSL operator is not correctly
// rounded.

// The two entry points, declared once each and rendered for four targets. Both
// are element gathers into one element scatter, exactly as the face pair is:
// the body takes the spectral terms of one tet and returns its result, so the
// return value reaches memory at the thread index in the scatter buffer.
//
// `tet_spectral_hessian_fused` beside them has no entry of its own,
// because this driver does not dispatch it.
[[seam::entry(force)]]
[[seam::device_fn]] inline Mat3x3f
tet_spectral_force(const Vec3f &gradient_sigma,
                       const Mat3x3f &u,
                       const Mat3x3f &vt) {
    return u * gradient_sigma.asDiagonal() * vt;
}

[[seam::entry(hessian)]]
[[seam::device_fn]] inline Mat9x9f
tet_spectral_hessian(const Vec3f &gradient_sigma,
                         const Mat3x3f &hessian_sigma,
                         const Mat3x3f &u,
                         const Vec3f &sigma,
                         const Mat3x3f &vt, float eps) {
    Vec3f eigenvalues;
    Mat3x3f eigenvectors;
    linalg::eig::symm3x3(hessian_sigma, eigenvalues, eigenvectors);
    constexpr float inverse_sqrt_two = 0.7071067811865475f;
    Mat3x3f mode[9];
    for (unsigned i = 0; i < 9; ++i) {
        mode[i] = Mat3x3f::Zero();
    }
    mode[0](1, 0) = inverse_sqrt_two;
    mode[0](0, 1) = -inverse_sqrt_two;
    mode[1](2, 0) = inverse_sqrt_two;
    mode[1](0, 2) = -inverse_sqrt_two;
    mode[2](2, 1) = inverse_sqrt_two;
    mode[2](1, 2) = -inverse_sqrt_two;
    mode[3](1, 0) = inverse_sqrt_two;
    mode[3](0, 1) = inverse_sqrt_two;
    mode[4](2, 0) = inverse_sqrt_two;
    mode[4](0, 2) = inverse_sqrt_two;
    mode[5](2, 1) = inverse_sqrt_two;
    mode[5](1, 2) = inverse_sqrt_two;
    for (unsigned row = 0; row < 3; ++row) {
        mode[6](row, row) = eigenvectors(row, 0);
        mode[7](row, row) = eigenvectors(row, 1);
        mode[8](row, row) = eigenvectors(row, 2);
    }
    const float difference_ab = sigma[0] - sigma[1];
    const float difference_ac = sigma[0] - sigma[2];
    const float difference_bc = sigma[1] - sigma[2];
    float lambda[9];
    lambda[0] = fmath::max(
        0.0f, fmath::div(gradient_sigma[0] + gradient_sigma[1],
                       sigma[0] + sigma[1]));
    lambda[1] = fmath::max(
        0.0f, fmath::div(gradient_sigma[0] + gradient_sigma[2],
                       sigma[0] + sigma[2]));
    lambda[2] = fmath::max(
        0.0f, fmath::div(gradient_sigma[1] + gradient_sigma[2],
                       sigma[1] + sigma[2]));
    lambda[3] =
        fmath::max(0.0f,
                 fmath::abs(difference_ab) > eps
                     ? fmath::div(gradient_sigma[0] - gradient_sigma[1],
                                difference_ab)
                     : 0.5f * (hessian_sigma(0, 0) + hessian_sigma(1, 1)) -
                           0.5f *
                               (hessian_sigma(0, 1) + hessian_sigma(1, 0)));
    lambda[4] =
        fmath::max(0.0f,
                 fmath::abs(difference_ac) > eps
                     ? fmath::div(gradient_sigma[0] - gradient_sigma[2],
                                difference_ac)
                     : 0.5f * (hessian_sigma(0, 0) + hessian_sigma(2, 2)) -
                           0.5f *
                               (hessian_sigma(0, 2) + hessian_sigma(2, 0)));
    lambda[5] =
        fmath::max(0.0f,
                 fmath::abs(difference_bc) > eps
                     ? fmath::div(gradient_sigma[1] - gradient_sigma[2],
                                difference_bc)
                     : 0.5f * (hessian_sigma(1, 1) + hessian_sigma(2, 2)) -
                           0.5f *
                               (hessian_sigma(1, 2) + hessian_sigma(2, 1)));
    lambda[6] = fmath::max(0.0f, eigenvalues[0]);
    lambda[7] = fmath::max(0.0f, eigenvalues[1]);
    lambda[8] = fmath::max(0.0f, eigenvalues[2]);
    Mat9x9f result = Mat9x9f::Zero();
    for (unsigned i = 0; i < 9; ++i) {
        if (lambda[i] > 0.0f) {
            const Mat3x3f transformed = u * mode[i] * vt;
            Vec9f vector;
            for (unsigned element = 0; element < 9; ++element) {
                vector[element] = transformed.m[element];
            }
            result += lambda[i] * vector * vector.transpose();
        }
    }
    return result;
}

// The same PSD-projected Hessian, built straight into the 12x12 element block
// and never materializing the 9x9 dF-space intermediate.
//
// Algebra: the 9x9 is sum_i lambda_i q_i q_i^T with q_i = vec(M_i),
// M_i = U Q_i V^T; the chain-rule conversion contracts each dF index with the
// shape-gradient vectors g_a, so
//   d2edx2.block(a,b) = sum_i lambda_i (M_i g_a)(M_i g_b)^T
//                     = sum_i lambda_i vec(P_i) vec(P_i)^T,  P_i = M_i G
// with G's columns the four shape gradients. With W = V^T G precomputed, each
// twist and flip mode's P_i is TWO rank-1 outer products (its Q_i has two
// nonzeros) and each scaling mode's is three, so the 9x9 intermediate (324 B of
// thread-local traffic) and the 16x9-block conversion loop disappear. This
// kernel is latency-bound at low occupancy, so the shorter dependency chain and
// the smaller local footprint are the point.
//
// THE LAMBDA CLAMPING IS THE SPD PROJECTION AND IS BYTE-IDENTICAL TO
// tet_spectral_hessian ABOVE, expression for expression and in the same
// order. The two paths are the same Hessian, so a divergence between them is a
// defect in whichever one was edited alone.
//
// It ACCUMULATES into `out` rather than returning a matrix: a 12x12 returned by
// value blows the 1 KB default device stack, and accumulating in place keeps
// the tet chain's peak footprint at the caller's one 12x12.
//
// The shape gradients are formed here rather than through
// tet_shape_gradients (utility/tet_convert.kernel.cpp), which computes the
// identical expression. That is deliberate and is the one duplication in this
// file: the Metal shader is one concatenated string whose segment ORDER is the
// include mechanism, and this body sits ahead of tet_convert in it, so calling
// across would depend on an ordering this file cannot state. Reordering the
// segments and calling the shared body is a Metal-side change.
[[seam::device_fn]] inline void tet_spectral_hessian_fused(
    const Vec3f &gradient_sigma,
    const Mat3x3f &hessian_sigma,
    const Mat3x3f &u, const Vec3f &sigma,
    const Mat3x3f &vt, float eps,
    const Mat3x3f &inverse_rest, float mass,
    Mat12x12f &out) {
    Vec3f eigenvalues;
    Mat3x3f eigenvectors;
    linalg::eig::symm3x3(hessian_sigma, eigenvalues, eigenvectors);
    const float a = sigma[0];
    const float b = sigma[1];
    const float c = sigma[2];
    const float difference_ab = a - b;
    const float difference_ac = a - c;
    const float difference_bc = b - c;
    Vec9f lambda;
    lambda[0] = fmath::max(
        0.0f, fmath::div(gradient_sigma[0] + gradient_sigma[1], a + b));
    lambda[1] = fmath::max(
        0.0f, fmath::div(gradient_sigma[0] + gradient_sigma[2], a + c));
    lambda[2] = fmath::max(
        0.0f, fmath::div(gradient_sigma[1] + gradient_sigma[2], b + c));
    lambda[3] =
        fmath::max(0.0f,
                 fmath::abs(difference_ab) > eps
                     ? fmath::div(gradient_sigma[0] - gradient_sigma[1],
                                difference_ab)
                     : 0.5f * (hessian_sigma(0, 0) + hessian_sigma(1, 1)) -
                           0.5f *
                               (hessian_sigma(0, 1) + hessian_sigma(1, 0)));
    lambda[4] =
        fmath::max(0.0f,
                 fmath::abs(difference_ac) > eps
                     ? fmath::div(gradient_sigma[0] - gradient_sigma[2],
                                difference_ac)
                     : 0.5f * (hessian_sigma(0, 0) + hessian_sigma(2, 2)) -
                           0.5f *
                               (hessian_sigma(0, 2) + hessian_sigma(2, 0)));
    lambda[5] =
        fmath::max(0.0f,
                 fmath::abs(difference_bc) > eps
                     ? fmath::div(gradient_sigma[1] - gradient_sigma[2],
                                difference_bc)
                     : 0.5f * (hessian_sigma(1, 1) + hessian_sigma(2, 2)) -
                           0.5f *
                               (hessian_sigma(1, 2) + hessian_sigma(2, 1)));
    lambda[6] = fmath::max(0.0f, eigenvalues[0]);
    lambda[7] = fmath::max(0.0f, eigenvalues[1]);
    lambda[8] = fmath::max(0.0f, eigenvalues[2]);

    Vec3f shape_gradient =
        -inverse_rest.row(0) - inverse_rest.row(1) - inverse_rest.row(2);
    Mat3x4f shape;
    shape.col(0) = shape_gradient;
    shape.col(1) = inverse_rest.row(0);
    shape.col(2) = inverse_rest.row(1);
    shape.col(3) = inverse_rest.row(2);
    const Mat3x4f w = vt * shape;

    constexpr float inverse_sqrt_two = 0.7071067811865475f;
    // Twist (antisymmetric, sign -1) then flip (symmetric, sign +1) index
    // pairs, matching modes 0 through 5 of the non-fused form above.
    const unsigned mode_row[6] = {0u, 0u, 1u, 0u, 0u, 1u};
    const unsigned mode_column[6] = {1u, 2u, 2u, 1u, 2u, 2u};
    for (unsigned i = 0; i < 9; ++i) {
        if (lambda[i] == 0.0f) {
            continue;
        }
        Mat3x4f p;
        if (i < 6) {
            const float sign = (i < 3) ? -1.0f : 1.0f;
            const unsigned r = mode_row[i];
            const unsigned cc = mode_column[i];
            for (unsigned col = 0; col < 4; ++col) {
                p.col(col) = inverse_sqrt_two * (w(cc, col) * u.col(r) +
                                                 sign * w(r, col) * u.col(cc));
            }
        } else {
            const Vec3f v = eigenvectors.col(i - 6);
            for (unsigned col = 0; col < 4; ++col) {
                p.col(col) = v[0] * w(0, col) * u.col(0) +
                             v[1] * w(1, col) * u.col(1) +
                             v[2] * w(2, col) * u.col(2);
            }
        }
        const float scale = mass * lambda[i];
        const float *flat = p.data();
        for (unsigned r = 0; r < 12; ++r) {
            const float scaled_row = scale * flat[r];
            for (unsigned col = 0; col < 12; ++col) {
                out(r, col) += scaled_row * flat[col];
            }
        }
    }
}
