// File: rod_bend_force.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. `[[seam::device_fn]]` is the
// execution space and `[[seam::thread]]` is the address space MSL requires on
// every reference and pointer.
//
// THE ROD'S BENDING TERM over one interior vertex and its two segments, scaled
// to the true bending stiffness and damped with the lagged form of that same
// stiffness.

#include "../utility/rod_bend_damping.kernel.cpp"
#include "model/rod_bend.kernel.cpp"

// The bending gradient and PSD-projected Hessian of one two-segment stencil.
//
// `x` and `current` are the three nodes in the stencil's own order, (j, i, k)
// with `i` the interior vertex, three entries each. THAT ORDER IS NOT
// NORMALIZED and must not be: `builder.rs` emits the rod-bend Hessian slot
// table in it, keyed by the interior vertex, so slot[ii * 3 + jj] targets the
// pair a push would. Which neighbor is `j` and which is `k` follows from the
// vertex-edge neighbor table, which is the caller's walk and stays there.
//
// THE DAMPING HESSIAN IS EVALUATED AT THE START-OF-STEP POSE, for the reason
// given in energy/hinge_force.kernel.cpp: it makes the damping force the exact
// gradient of a convex potential, so it is guaranteed dissipative, where the
// current-iterate form can inject energy for a fast-varying turning-angle
// Hessian. The lagged evaluation's force output is discarded; only its Hessian
// is used.
[[seam::device_fn]] inline void rod_bend_element_force_hessian(
    const Vec3f *x, const Vec3f *current,
    float rest_angle, float stiffness, float bend_damping, float dt,
    Mat3x3f &gradient, Mat9x9f &hessian) {
    rod_bend_force_hessian(x[0], x[1], x[2], rest_angle, gradient,
                               hessian);
    // Scale to the true bending stiffness first, then damp with that K.
    gradient *= stiffness;
    hessian *= stiffness;
    if (bend_damping > 0.0f) {
        Mat3x3f lagged_gradient;
        Mat9x9f lagged_hessian;
        rod_bend_force_hessian(current[0], current[1], current[2],
                                   rest_angle, lagged_gradient,
                                   lagged_hessian);
        lagged_hessian *= stiffness;
        rod_bend_add_stiffness_damping_lagged(x, current, bend_damping, dt,
                                                  gradient, hessian,
                                                  lagged_hessian);
    }
}
