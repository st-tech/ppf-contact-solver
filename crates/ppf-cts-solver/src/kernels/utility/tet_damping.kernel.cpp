// File: tet_damping.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The two facts a backend cannot infer are written as
// C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a pointer or reference parameter.
// MSL requires the second on every pointer and reference type; CUDA and the
// host have one address space and are handed the same declarations with it
// removed.
//
// No include of its own. `Vec3f`, `Mat3x4f` and `Mat12x12f` arrive from
// whatever declares them for the backend that is compiling, which is data.hpp
// under nvcc and on the host and the shader prologue's aliases under MSL.

[[seam::device_fn]] inline void tet_add_stiffness_damping(
    const Vec3f *x, const Vec3f *current,
    float beta, float dt, Mat3x4f &gradient,
    Mat12x12f &hessian) {
    if (beta <= 0.0f || dt <= 0.0f) {
        return;
    }
    const float scale = beta / dt;
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
                force += hessian(row, column) * displacement[column];
            }
            gradient(row_dimension, vertex_index) += scale * force;
        }
    }
    hessian *= 1.0f + scale;
}

// THE ENTRY POINT'S FORM OF THE SAME DAMPING, and the rebuild above the call is
// the whole of what it adds. `[[seam::through]]` expands into the N elements its
// index list names, in slot order, in its own parameter position, so a body
// reached that way spells them one by one; the body above takes two arrays of
// four, which is what its other call sites already hand it, so this composition
// rebuilds the arrays rather than changing a signature six call sites read.
//
// IT TAKES ITS OWN NAME because the census keys a neutral body on its name and
// two bodies spelled alike are two rows it cannot tell apart.
[[seam::device_fn]] inline void tet_damping(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const Vec3f &current3, float beta, float dt,
    Mat3x4f &gradient, Mat12x12f &hessian) {
    const Vec3f x[4] = {x0, x1, x2, x3};
    const Vec3f current[4] = {current0, current1, current2, current3};
    tet_add_stiffness_damping(x, current, beta, dt, gradient, hessian);
}

// FOUR POSITIONS AND FOUR START-OF-STEP POSES READ THROUGH THE TET'S OWN INDEX
// LIST. Both buffers are reached at the SAME slots, which is what lets one index
// list serve them; exchanging the two makes the damping the reverse difference
// and returns a plausible number, so the parameter order here is load-bearing.
//
// THE `[[seam::bound]]` IS NOT BOOKKEEPING. The range shim this replaces
// subscripted the position arrays with an unchecked slot, and Metal answers an
// out-of-bounds read with 0.0 rather than a fault, so a corrupt tet table
// produced a plausible damping force there.
//
// THE GRADIENT AND THE HESSIAN ARE NON-CONST GATHERS AND NOT SCATTERS. A gather
// hands the body `buffer[index]`, which is an lvalue, so a non-const one IS the
// write, and there is no scatter because the body returns nothing. Both are
// READ before they are written: this pass ADDS the damping force to the
// gradient and scales the Hessian it was handed, so each carries what the
// elastic assembly left in it and a zero fill would be a different answer.
[[seam::entry(count)]] void tet_damping(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(4)]] const unsigned *tet,
    [[seam::bound]] unsigned vertex_count,
    const float *beta,
    float dt,
    Mat3x4f *gradient,
    Mat12x12f *hessian,
    unsigned count);
