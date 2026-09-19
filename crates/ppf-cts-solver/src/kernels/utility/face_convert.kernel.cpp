// File: face_convert.kernel.cpp
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
// requires the second on every reference and pointer type; CUDA and the host
// have one address space and are handed the same declarations with it removed.
//
// No include of its own. The matrix and vector types arrive from the includer
// rather than from here, which is data.hpp under nvcc and on the host and the
// shader prologue's aliases under MSL.

// The two entry points, declared once each and rendered for four targets. Both
// are two element gathers into one element scatter: the body takes one face's
// material-frame quantity and its inverse rest shape and returns the
// world-frame result, which the scatter writes at the thread index.
[[seam::entry(force)]]
[[seam::device_fn]] inline Mat3x3f
face_convert_force(const Mat3x2f &gradient_f,
                       const Mat2x2f &inverse_rest) {
    Vec2f g0 = -inverse_rest.row(0) - inverse_rest.row(1);
    Vec2f g1 = inverse_rest.row(0);
    Vec2f g2 = inverse_rest.row(1);
    Mat3x3f result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        result(dimension, 0) = g0.dot(gradient_f.row(dimension));
        result(dimension, 1) = g1.dot(gradient_f.row(dimension));
        result(dimension, 2) = g2.dot(gradient_f.row(dimension));
    }
    return result;
}

[[seam::entry(hessian)]]
[[seam::device_fn]] inline Mat9x9f
face_convert_hessian(const Mat6x6f &hessian_f,
                         const Mat2x2f &inverse_rest) {
    Vec2f g[3];
    g[0] = -inverse_rest.row(0) - inverse_rest.row(1);
    g[1] = inverse_rest.row(0);
    g[2] = inverse_rest.row(1);
    Mat9x9f result;
    for (unsigned a = 0; a < 3; ++a) {
        for (unsigned b = 0; b < 3; ++b) {
            Mat3x3f block = Mat3x3f::Zero();
            for (unsigned d = 0; d < 2; ++d) {
                for (unsigned e = 0; e < 2; ++e) {
                    block += (g[a][d] * g[b][e]) *
                             hessian_f.block<3, 3>(3 * d, 3 * e);
                }
            }
            result.block<3, 3>(3 * a, 3 * b) = block;
        }
    }
    return result;
}
