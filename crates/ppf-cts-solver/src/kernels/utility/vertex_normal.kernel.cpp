// File: vertex_normal.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The arithmetic of a per-vertex normal, shared by every backend.
//
// The GATHER is not here and cannot be: walking a vertex's incident faces means
// naming a device container, and the two backends address device memory
// differently (CUDA indexes a VecVec, Metal an arena handle plus an offset
// array). What the two must not each invent is the QUANTITY, so the per-face
// contribution and the finalize step live here and the gather is written once
// per backend around them.

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
// No include of its own. `Vec3f` arrives from whatever declares it for the
// backend that is compiling, which is data.hpp under nvcc and on the host and
// the shader prologue's aliases under MSL.

// One incident face's contribution: its area vector, whose magnitude is twice
// the triangle's area, so the accumulated sum is an area-weighted normal.
//
// THE TWO EDGE VECTORS ARE FORMED FIRST, and that is what keeps the direction
// usable far from the origin. `(z1 - z0) x (z2 - z0)` expands to
// `z1 x z2 - z0 x z2 - z1 x z0`, three products of ABSOLUTE positions whose
// leading digits then cancel, leaving the rounding of the large products as the
// answer. Differencing first makes both factors small, so the area vector keeps
// its own precision wherever the triangle sits.
[[seam::device_fn]] inline Vec3f
vertex_normal_face_term(const Vec3f &z0,
                            const Vec3f &z1,
                            const Vec3f &z2) {
    return (z1 - z0).cast<float>().cross((z2 - z0).cast<float>());
}

// Turns the accumulated area vectors into the unit normal. A vertex with no
// incident face, and one whose incident faces cancel exactly, both keep the
// zero vector: normalize() declines to act on a squared norm that is not
// positive, and a zero normal is how a caller reads "this vertex presents no
// surface".
// The finalize stage's entry point, declared once and rendered for four
// targets: the `__global__` nvcc compiles, the `kernel void` the Metal shader
// compiler compiles, the `vertex_normal_finalize_entry` shim a host C++
// compiler compiles, and the Rust `#[repr(C)]` twin the driver fills. One
// accumulated area vector in, one unit normal out, at the same element.
//
// ONLY THE FINALIZE STAGE. The per-face term reads its three positions through
// the face's own index triple, which is an indirect gather and none of the
// three shapes an entry declaration expresses, so its launcher stays
// hand-written until the body takes the addressing itself.
[[seam::entry(normal)]]
[[seam::device_fn]] inline Vec3f
vertex_normal_finalize(const Vec3f &accumulated) {
    Vec3f normal = accumulated;
    if (normal.squaredNorm()) {
        normal.normalize();
    }
    return normal;
}
