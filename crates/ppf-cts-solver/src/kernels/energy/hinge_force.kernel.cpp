// File: hinge_force.kernel.cpp
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
// DIAG_ASSERT4 stays a macro for the reason given in
// contact/aabb_traversal.kernel.cpp: it records __FILE__ and __LINE__ of the
// failing check, which no function can read for its caller.
//
// THE SHELL HINGE'S BENDING TERM, scaled to the true bending stiffness and
// damped with the lagged form of that same stiffness.

#include "../utility/hinge_damping.kernel.cpp"
#include "model/shell_bend.kernel.cpp"

// The bending gradient and PSD-projected Hessian of one hinge.
//
// `x` and `current` are the hinge's four vertices in MESH order, four entries
// each. The (2, 1, 0, 3) permutation the dihedral math wants is applied here,
// to the VALUES; the caller applies the same permutation to the INDEX quadruple
// through shell_bend_remap before it scatters, and the Hessian slot table
// was built in that order too.
//
// THE STIFFNESS ARRIVES ALREADY FORMED, and that is deliberate. It is composed
// from the two bend directions, the hinge's UV orientation, its edge length,
// its area and its areal density in
// energy/model/shell_bend_stiffness.kernel.cpp, and the areal density is an
// fp32 running sum over the four vertices IN MESH ORDER, so the caller has to
// read the masses and areas before this permutation is applied. Folding that
// fold into this body would put it after the permutation and change its value.
//
// THE DAMPING HESSIAN IS EVALUATED AT THE START-OF-STEP POSE, not at the Newton
// iterate. That makes the damping force the exact gradient of a convex
// potential, so it is guaranteed dissipative; the current-iterate form used for
// the membrane drops a term that is negligible for a smoothly varying Hessian
// and large for the dihedral one, where it can inject energy instead of
// removing it. The lagged evaluation gets its OWN force output, which is
// discarded: only its Hessian is used.
//
// The three geometric quantities each evaluation returns are asserted positive
// through the diagnostic channel, once per evaluation, exactly as the driver
// did.
template <typename D>
[[seam::device_fn]] inline void hinge_bend_force_hessian(
    const Vec3f *x, const Vec3f *current,
    float rest_angle, float stiffness, float bend_damping, float dt,
    Mat3x4f &gradient, Mat12x12f &hessian,
    D diag) {
    Vec3f iterate[4];
    Vec3f step_start[4];
    const unsigned order[4] = {2u, 1u, 0u, 3u};
    for (unsigned node = 0; node < 4; ++node) {
        iterate[node] = x[order[node]];
        step_start[node] = current[order[node]];
    }
    float normal1_squared, normal2_squared, shared_edge_norm;
    shell_bend_force_hessian(iterate[0], iterate[1], iterate[2],
                                 iterate[3], rest_angle, gradient, hessian,
                                 normal1_squared, normal2_squared,
                                 shared_edge_norm);
    DIAG_ASSERT4(diag, normal1_squared > 0.0f, normal1_squared,
                normal2_squared, shared_edge_norm, 0.0f);
    DIAG_ASSERT4(diag, normal2_squared > 0.0f, normal1_squared,
                normal2_squared, shared_edge_norm, 1.0f);
    DIAG_ASSERT4(diag, shared_edge_norm > 0.0f, normal1_squared,
                normal2_squared, shared_edge_norm, 2.0f);
    // Scale to the true bending stiffness first, then damp with that K.
    gradient *= stiffness;
    hessian *= stiffness;
    if (bend_damping > 0.0f) {
        Mat3x4f lagged_gradient;
        Mat12x12f lagged_hessian;
        float lagged_normal1_squared, lagged_normal2_squared,
            lagged_shared_edge_norm;
        shell_bend_force_hessian(
            step_start[0], step_start[1], step_start[2], step_start[3],
            rest_angle, lagged_gradient, lagged_hessian,
            lagged_normal1_squared, lagged_normal2_squared,
            lagged_shared_edge_norm);
        DIAG_ASSERT4(diag, lagged_normal1_squared > 0.0f,
                    lagged_normal1_squared, lagged_normal2_squared,
                    lagged_shared_edge_norm, 0.0f);
        DIAG_ASSERT4(diag, lagged_normal2_squared > 0.0f,
                    lagged_normal1_squared, lagged_normal2_squared,
                    lagged_shared_edge_norm, 1.0f);
        DIAG_ASSERT4(diag, lagged_shared_edge_norm > 0.0f,
                    lagged_normal1_squared, lagged_normal2_squared,
                    lagged_shared_edge_norm, 2.0f);
        lagged_hessian *= stiffness;
        hinge_add_stiffness_damping_lagged(iterate, step_start,
                                               bend_damping, dt, gradient,
                                               hessian, lagged_hessian);
    }
}
