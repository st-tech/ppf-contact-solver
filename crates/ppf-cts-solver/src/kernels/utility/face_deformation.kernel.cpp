// File: face_deformation.kernel.cpp
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

[[seam::device_fn]] inline Mat3x2f face_deformation_gradient(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const Mat2x2f &inverse_rest) {
    Mat3x2f edges;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        edges(dimension, 0) = static_cast<float>(x1[dimension] - x0[dimension]);
        edges(dimension, 1) = static_cast<float>(x2[dimension] - x0[dimension]);
    }
    return edges * inverse_rest;
}

// The entry point, declared once and rendered for four targets. It reads the
// face's three positions THROUGH the element's own index list, which is the
// indirect gather: `x` is [[seam::through]] the [[seam::indices(3)]] slots, so
// the entry reads them, checks each against `vertex_count`, and hands the body
// the three positions they name as three arguments. The body is unchanged by
// that and does no index arithmetic, which is also how the fused CUDA and Metal
// kernels call it, so one body serves both arrangements.
//
// THE BOUND IS WHAT THE ENTRY FORM ADDS over passing `x` as a base pointer and
// letting the body subscript it. A slot is data rather than the thread index,
// so the [[seam::count]] guard says nothing about it; Metal returns 0.0 for an
// out-of-bounds read rather than faulting, which would turn a corrupt index
// list into a plausible deformation gradient instead of a stopped run.
//
// THE POSITIONS CROSS AS A LATTICE TYPE and never as floats. `Vec3f` is 12
// bytes of int32, and the body differences two of them before any cast, which
// is exact; a float image of either one is not.
[[seam::entry(count)]] void face_deformation_gradient(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    
    const Mat2x2f *inverse_rest,
    Mat3x2f *deformation,
    unsigned count);

// THE TWO PER-FACE SELECTIONS THE STRETCH INDICATOR MULTIPLIES, taken from the
// singular values of the gradient above and from the face's authored shrink
// factors.
//
// `max_sigma` is the largest principal stretch a face carries, and the solver
// reports the largest of those over the mesh; the smaller of the two shrink
// factors is what the same face was authored to shrink to. Their product is the
// stretch relative to the rest shape the material actually asks for, which is
// why the two are selected together rather than reduced separately.
//
// BOTH WERE SPELLED IN THE CPU BACKEND'S HAND-WRITTEN LAUNCHER, so a per-face
// value the reported indicator is built out of was written where only one of
// the three compilers reads it. There is no branch and no order to choose here,
// only two selections, which is exactly what a neutral body is for.
//
// TWO OUTPUTS AND THEREFORE NO SCATTER: a scatter carries one return value, so
// both destinations are reached through non-const gathers, which hand the body
// the element as an lvalue.
// THE SHRINK FACTORS COME OUT OF THE FACE'S OWN MATERIAL, which is where the
// reference reads them: `strainlimiting.cu` spells
// `fminf(fparam.shrink_x, fparam.shrink_y)` off the parameter record the kernel
// is already holding, and maintains no per-face array of either. This tree
// materialized both into device arrays filled by a HOST LOOP over every shell
// face on every step, out of inputs that are fixed for the life of the run: the
// topology, the collider and fixed flags, the material index, the material's
// own factors and a vertex's PDRD membership. So every step after the first
// recomputed the previous step's answer and paid two uploads to deliver it.
//
// THE GATE IS THE HOST LOOP'S, MOVED RATHER THAN CHANGED. A face on a PDRD
// body, a collider face and a fixed face all yielded zero there, and zero here,
// which drives the product this feeds to zero and excludes the face exactly as
// before.
[[seam::entry]]
[[seam::device_fn]] inline void shell_stretch_terms(
    const Vec2f &sigma, const Vec3u &face,
    const FaceProp &prop,
    const FaceParam *face_param,
    const VertexProp *vertex_prop,
    float &largest, float &shrink_min) {
    largest = fmath::max(sigma[0], sigma[1]);
    // THE THREAD-SPACE COPY A GENERATED ENTRY DID NOT MAKE: both of these are
    // reached by an index of the body's own rather than by the thread index, so
    // no gather brought them in and on Metal they are device-space lvalues.
    const VertexProp first = vertex_prop[face[0]];
    if (first.pdrd_body_index != 0u || prop.collider || prop.fixed) {
        shrink_min = 0.0f;
        return;
    }
    const FaceParam material = face_param[prop.param_index];
    shrink_min = fmath::min(material.shrink_x, material.shrink_y);
}
