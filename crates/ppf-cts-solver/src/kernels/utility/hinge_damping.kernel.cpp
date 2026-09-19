// File: hinge_damping.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. `[[seam::device_fn]]` is the
// execution space and `[[seam::thread]]` is the address space MSL requires on
// every reference and pointer type; CUDA and the host have one address space
// and are handed the same declarations with it removed. `fmath::div` is the
// backend prologue's division, which on MSL is the correctly rounded
// `precise::divide` rather than the default one.

// Rayleigh bending damping for a shell hinge, with the damping operator
// LAGGED: `lagged` is the hinge Hessian evaluated at the START-OF-STEP
// positions `current`, not at the Newton iterate `x`.
//
// That makes the damping force (beta/dt) * K_lag * (x - x^n) exactly the
// gradient of the convex potential (beta/2dt) (x - x^n)^T K_lag (x - x^n), so
// it is guaranteed dissipative and the block it adds to the Hessian is the same
// PSD K_lag. The current-iterate form used for the membrane and the solid drops
// the d(K v)/dx term, which is negligible for a smoothly varying Hessian but
// large for the dihedral one (g g^T with a fast-changing angle gradient g),
// where it can inject energy instead of removing it.
//
// TWO DISTINCT POSES ARE REQUIRED, and passing one handle for both is a silent
// defect rather than a loud one: with x == current the displacement is
// identically zero, so the damping FORCE vanishes while the stiffness inflation
// survives, and a damped material quietly becomes a stiffer undamped one.
//
// `x` and `current` are the hinge's four vertices in the SAME order the caller
// evaluated `lagged` and `gradient` in, which for the shell hinge is the
// (2,1,0,3) order the dihedral math wants, not the mesh order.
//
// No-op when beta <= 0 or dt <= 0. Call AFTER the elastic gradient and Hessian
// are fully scaled to the true bending stiffness, and BEFORE the scatter.
[[seam::device_fn]] inline void hinge_add_stiffness_damping_lagged(
    const Vec3f *x, const Vec3f *current,
    float beta, float dt, Mat3x4f &gradient,
    Mat12x12f &hessian,
    const Mat12x12f &lagged) {
    if (beta <= 0.0f || dt <= 0.0f) {
        return;
    }
    const float scale = fmath::div(beta, dt);
    float displacement[12];
    for (unsigned vertex_index = 0; vertex_index < 4; ++vertex_index) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            displacement[3 * vertex_index + dimension] =
                static_cast<float>(x[vertex_index][dimension] -
                                   current[vertex_index][dimension]);
        }
    }
    for (unsigned vertex_index = 0; vertex_index < 4; ++vertex_index) {
        for (unsigned row_dimension = 0; row_dimension < 3; ++row_dimension) {
            float force = 0.0f;
            const unsigned row = 3 * vertex_index + row_dimension;
            for (unsigned column = 0; column < 12; ++column) {
                force += lagged(row, column) * displacement[column];
            }
            gradient(row_dimension, vertex_index) += scale * force;
        }
    }
    for (unsigned row = 0; row < 12; ++row) {
        for (unsigned column = 0; column < 12; ++column) {
            hessian(row, column) += scale * lagged(row, column);
        }
    }
}

// THE ENTRY POINT'S FORM, whose four positions and four start-of-step poses
// arrive as eight arguments rather than as two arrays. `[[seam::through]]`
// expands into the N elements its index list names, in slot order, so a body
// reached that way spells them one by one; the body above takes two arrays of
// four, which is what its other call sites already hand it, so this rebuilds the
// arrays rather than changing a signature those call sites read.
//
// THE LAGGED HESSIAN STAYS LAST, as the body takes it. It is a separate input
// evaluated at the start-of-step pose, not a second view of `hessian`, and the
// two are the same type: exchanging them would compile and would damp against
// the iterate's own operator, which is the form this body exists to avoid.
[[seam::device_fn]] inline void hinge_damping(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const Vec3f &current3, float beta, float dt,
    Mat3x4f &gradient, Mat12x12f &hessian,
    const Mat12x12f &lagged) {
    const Vec3f x[4] = {x0, x1, x2, x3};
    const Vec3f current[4] = {current0, current1, current2, current3};
    hinge_add_stiffness_damping_lagged(x, current, beta, dt, gradient,
                                           hessian, lagged);
}

// FOUR POSITIONS AND FOUR START-OF-STEP POSES READ THROUGH THE HINGE'S OWN
// INDEX LIST, which for the shell hinge is the (2,1,0,3) order the dihedral math
// wants rather than the mesh order: the driver hands this entry the REMAPPED
// list, and `lagged` and `gradient` were evaluated in that same order.
//
// The bound, the two non-const gathers and the read-before-write rule are the
// family's, and `utility/face_damping.kernel.cpp` states them once.
[[seam::entry(count)]] void hinge_damping(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(4)]] const unsigned *hinge,
    [[seam::bound]] unsigned vertex_count,
    const float *beta,
    float dt,
    Mat3x4f *gradient,
    Mat12x12f *hessian,
    const Mat12x12f *lagged,
    unsigned count);
