// File: rod_bend_damping.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++, no preprocessor conditional and no macro
// of its own. ppf-cts-compute/seam/kernelgen.py renders it into the three forms the three
// compilers read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space and `[[seam::thread]]` is the
// address space MSL requires on every reference and pointer. Everything else is
// an ordinary call into the `ppf` namespace each backend prologue defines.

// Rayleigh bending damping for a rod's three-node bending stencil, with the
// damping operator LAGGED: `lagged` is the stencil Hessian evaluated at the
// START-OF-STEP positions `current`, not at the Newton iterate `x`. It is the
// arity-3 member of the set hinge_damping.kernel.cpp's arity-4 body belongs to.
//
// That makes the damping force (beta/dt) * K_lag * (x - x^n) exactly the
// gradient of the convex potential (beta/2dt) (x - x^n)^T K_lag (x - x^n), so
// it is guaranteed dissipative and the block it adds to the Hessian is the same
// PSD K_lag. The current-iterate form used for the membrane, the rod stretch
// and the solid drops the d(K v)/dx term, which is negligible for a smoothly
// varying Hessian but large for the turning-angle one (g g^T with a
// fast-changing angle gradient g), where it can inject energy instead of
// removing it.
//
// TWO DISTINCT POSES ARE REQUIRED, and passing one handle for both is a silent
// defect rather than a loud one: with x == current the displacement is
// identically zero, so the damping FORCE vanishes while the stiffness inflation
// survives, and a damped rod quietly becomes a stiffer undamped one.
//
// `x` and `current` are the stencil's three vertices in the SAME order the
// caller evaluated `lagged` and `gradient` in, which for the rod bend is the
// (j, i, k) order the turning-angle math wants: the two neighbors either side
// of the interior vertex, with the interior vertex in the middle.
//
// No-op when beta <= 0 or dt <= 0. Call AFTER the elastic gradient and Hessian
// are fully scaled to the true bending stiffness, and BEFORE the scatter.
[[seam::device_fn]] inline void rod_bend_add_stiffness_damping_lagged(
    const Vec3f *x, const Vec3f *current,
    float beta, float dt, Mat3x3f &gradient,
    Mat9x9f &hessian, const Mat9x9f &lagged) {
    if (beta <= 0.0f || dt <= 0.0f) {
        return;
    }
    const float scale = fmath::div(beta, dt);
    float displacement[9];
    for (unsigned vertex_index = 0; vertex_index < 3; ++vertex_index) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            displacement[3 * vertex_index + dimension] =
                static_cast<float>(x[vertex_index][dimension] -
                                   current[vertex_index][dimension]);
        }
    }
    for (unsigned vertex_index = 0; vertex_index < 3; ++vertex_index) {
        for (unsigned row_dimension = 0; row_dimension < 3; ++row_dimension) {
            float force = 0.0f;
            const unsigned row = 3 * vertex_index + row_dimension;
            for (unsigned column = 0; column < 9; ++column) {
                force += lagged(row, column) * displacement[column];
            }
            gradient(row_dimension, vertex_index) += scale * force;
        }
    }
    for (unsigned row = 0; row < 9; ++row) {
        for (unsigned column = 0; column < 9; ++column) {
            hessian(row, column) += scale * lagged(row, column);
        }
    }
}

// THE ENTRY POINT'S FORM, the arity-3 member of the pair
// `utility/hinge_damping.kernel.cpp` states in full: the three positions and
// three start-of-step poses arrive as six arguments and are rebuilt into the two
// arrays this body takes, and the lagged Hessian stays last as a separate input.
[[seam::device_fn]] inline void rod_bend_damping(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2, float beta, float dt,
    Mat3x3f &gradient, Mat9x9f &hessian,
    const Mat9x9f &lagged) {
    const Vec3f x[3] = {x0, x1, x2};
    const Vec3f current[3] = {current0, current1, current2};
    rod_bend_add_stiffness_damping_lagged(x, current, beta, dt, gradient,
                                              hessian, lagged);
}

// THREE POSITIONS AND THREE START-OF-STEP POSES READ THROUGH THE BENDING SITE'S
// OWN NODE LIST, with the bound each slot is checked against carried in the
// record. The two non-const gathers and the read-before-write rule are the
// family's, and `utility/face_damping.kernel.cpp` states them once.
[[seam::entry(count)]] void rod_bend_damping(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(3)]] const unsigned *node_index,
    [[seam::bound]] unsigned vertex_count,
    const float *beta,
    float dt,
    Mat3x3f *gradient,
    Mat9x9f *hessian,
    const Mat9x9f *lagged,
    unsigned count);
