// File: face_damping.kernel.cpp
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

[[seam::device_fn]] inline void face_add_stiffness_damping(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &curr0,
    const Vec3f &curr1, const Vec3f &curr2,
    float beta, float dt, Mat3x3f &gradient,
    Mat9x9f &hessian) {
    if (beta <= 0.0f || dt <= 0.0f) {
        return;
    }
    const float scale = beta / dt;
    float displacement[9];
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        displacement[dimension] =
            static_cast<float>(x0[dimension] - curr0[dimension]);
        displacement[3 + dimension] =
            static_cast<float>(x1[dimension] - curr1[dimension]);
        displacement[6 + dimension] =
            static_cast<float>(x2[dimension] - curr2[dimension]);
    }
    for (unsigned vertex_index = 0; vertex_index < 3; ++vertex_index) {
        float force[3] = {0.0f, 0.0f, 0.0f};
        for (unsigned column = 0; column < 9; ++column) {
            const float u = displacement[column];
            force[0] += hessian(3 * vertex_index, column) * u;
            force[1] += hessian(3 * vertex_index + 1, column) * u;
            force[2] += hessian(3 * vertex_index + 2, column) * u;
        }
        gradient(0, vertex_index) += scale * force[0];
        gradient(1, vertex_index) += scale * force[1];
        gradient(2, vertex_index) += scale * force[2];
    }
    hessian *= 1.0f + scale;
}

// THE ENTRY POINT'S FORM. It forwards unchanged: this body already spells its
// six positions one by one, which is what `[[seam::through]]` hands it, so the
// composition adds no work and exists for its NAME. That name is the one the
// driver's kernel id, its launcher and its dispatch label already carry, and
// all five members of this damping family are spelled the same way, so the set
// reads as one family rather than as five separate namings.
[[seam::device_fn]] inline void face_damping(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2, float beta, float dt,
    Mat3x3f &gradient, Mat9x9f &hessian) {
    face_add_stiffness_damping(x0, x1, x2, current0, current1, current2,
                                   beta, dt, gradient, hessian);
}

