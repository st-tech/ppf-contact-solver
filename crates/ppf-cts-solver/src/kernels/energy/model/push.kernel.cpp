// File: push.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::thread]]` is the address space of a reference parameter,
// which MSL requires on every reference and pointer type.
//
// No include of its own, matching friction.kernel.cpp beside it. `Vec3f` and
// `Mat3x3f` arrive from whatever declares them for the backend that is
// compiling, which is data.hpp under nvcc and on the host and the shader
// prologue's aliases under MSL.

// The one-sided push barrier: a cubic potential in the penetration depth, so
// the energy, its gradient and its curvature are all finite at the surface and
// all identically zero outside it.

[[seam::device_fn]] inline float push_energy(float signed_distance,
                                                float ghat) {
    if (signed_distance < 0.0f) {
        return -(signed_distance * signed_distance * signed_distance) /
               (3.0f * ghat);
    }
    return 0.0f;
}

[[seam::device_fn]] inline Vec3f
push_gradient(float signed_distance, const Vec3f &normal,
                  float ghat) {
    if (signed_distance < 0.0f) {
        return -(signed_distance * signed_distance) / ghat * normal;
    }
    return Vec3f::Zero();
}

[[seam::device_fn]] inline float push_curvature(float signed_distance,
                                                   float ghat) {
    if (signed_distance < 0.0f) {
        return -2.0f * signed_distance / ghat;
    }
    return 0.0f;
}

[[seam::device_fn]] inline Mat3x3f
push_hessian(float signed_distance, const Vec3f &normal,
                 float ghat) {
    return push_curvature(signed_distance, ghat) * normal *
           normal.transpose();
}

// The four entry points, each declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `push_*_entry` shim a host C++ compiler compiles, and
// the Rust `#[repr(C)]` twin the driver fills.
//
// EVERY ONE IS ELEMENT GATHERS INTO ONE ELEMENT SCATTER, which is what the
// launchers they replace already spelled: the signed distance and the normal
// are per contact, `ghat` is one value for the whole dispatch and so arrives in
// the record, and the result is written at the thread index. Nothing here
// changes an answer, because the generated statement is the launcher's
// argument for argument and store for store.
//
// The two shapes carry their own pod sizes, `Vec3f` at 12 bytes and `Mat3x3f`
// at 36, and every C++ rendering asserts them: a type one of the three
// compilers lays out differently fails to compile there rather than reading the
// wrong bytes on a backend that never faults.
[[seam::entry(count)]] void push_energy(
    const float *signed_distance, float ghat,
    float *energy,
    unsigned count);

[[seam::entry(count)]] void push_curvature(
    const float *signed_distance, float ghat,
    float *curvature,
    unsigned count);

[[seam::entry(count)]] void push_gradient(
    const float *signed_distance,
    const Vec3f *normal,
    float ghat,
    Vec3f *gradient,
    unsigned count);

[[seam::entry(count)]] void push_hessian(
    const float *signed_distance,
    const Vec3f *normal,
    float ghat,
    Mat3x3f *hessian,
    unsigned count);
