// File: rod_damping.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++, no preprocessor conditional and no macro
// of its own. ppf-cts-compute/seam/kernelgen.py renders it into the three forms the three
// compilers read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space and `[[seam::thread]]` is the
// address space MSL requires on every reference and pointer.

[[seam::device_fn]] inline void rod_add_stiffness_damping(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &current0,
    const Vec3f &current1, float beta, float dt,
    Mat3x2f &gradient, Mat6x6f &hessian) {
    if (beta <= 0.0f || dt <= 0.0f) {
        return;
    }
    const float scale = beta / dt;
    float displacement[6];
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        displacement[dimension] =
            static_cast<float>(x0[dimension] - current0[dimension]);
        displacement[3 + dimension] =
            static_cast<float>(x1[dimension] - current1[dimension]);
    }
    for (unsigned row = 0; row < 6; ++row) {
        float force = 0.0f;
        for (unsigned column = 0; column < 6; ++column) {
            force += hessian(row, column) * displacement[column];
        }
        gradient(row % 3, row / 3) += scale * force;
    }
    hessian *= 1.0f + scale;
}

// THE ENTRY POINT'S FORM, forwarding unchanged for the reason
// `utility/face_damping.kernel.cpp` states: this body already spells its four
// positions one by one, and the composition carries the name the driver's kernel
// id, its launcher and its dispatch label share with the rest of this family.
[[seam::device_fn]] inline void rod_damping(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &current0,
    const Vec3f &current1, float beta, float dt,
    Mat3x2f &gradient, Mat6x6f &hessian) {
    rod_add_stiffness_damping(x0, x1, current0, current1, beta, dt, gradient,
                                  hessian);
}

// TWO POSITIONS AND TWO START-OF-STEP POSES READ THROUGH THE EDGE'S OWN INDEX
// LIST, with the bound each slot is checked against carried in the record. The
// gradient and the Hessian are non-const gathers and not scatters, and both are
// read before they are written; `utility/face_damping.kernel.cpp` states why for
// the whole family.
[[seam::entry(count)]] void rod_damping(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    const float *beta,
    float dt,
    Mat3x2f *gradient,
    Mat6x6f *hessian,
    unsigned count);
