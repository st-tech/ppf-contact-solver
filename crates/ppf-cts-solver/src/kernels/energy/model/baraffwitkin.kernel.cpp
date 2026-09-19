// File: baraffwitkin.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read.
// `[[seam::device_fn]]` is the execution space and `[[seam::thread]]` /
// `[[seam::device]]` are the address spaces MSL requires on every reference and
// pointer type.
//
// THE BARAFFWITKIN MEMBRANE AS A STAGED PASS. `energy/face_force.kernel.cpp`
// carries the same material for a caller that holds one face in registers and
// wants its element-space force and Hessian in one call. This file carries it
// for a caller that stages the membrane over a whole range, keeping the
// deformation-gradient quantities in device memory between passes.
//
// NEITHER SPELLS THE MATERIAL. Both call `BaraffWitkin::material`
// (`model/baraffwitkin.hpp`), which is where the four terms are composed, so
// the two arrangements cannot disagree about what a BaraffWitkin face is.
//
// BARAFFWITKIN IS NOT A DIFF-TABLE MODEL, which is why it has a pass of its
// own rather than an arm inside the material table. It goes from the
// deformation gradient to the gradient and the Hessian directly and never
// forms an SVD, and keeping it out of the spectral branch is deliberate: it is
// the default cloth path and the eigensolve's registers are not free.

#include "baraffwitkin.hpp"
#include "../elastic_model.kernel.cpp"

// One face's BaraffWitkin force and Hessian, written where the staged pipeline
// expects them.
//
// IT OVERWRITES RATHER THAN ACCUMULATES, which is what makes running it after
// the spectral stages correct: a BaraffWitkin face's diff table is zero, so
// those stages wrote it zeros, and this replaces them with its own material.
//
// A FACE NAMING ANY OTHER MODEL IS LEFT EXACTLY AS THEY LEFT IT, which is what
// the early return is. The model id reaches the body rather than the entry
// point for the reason every dispatch in this tree validates where it
// dispatches: a declaration cannot express a branch, and this branch decides
// whether the face's material is this one or the table's.
//
// The two destinations are pointers to this element's own slot rather than
// references, because that is the shape a generated entry point hands a body
// whose output lives in device memory: `[[seam::stride(1)]]` is the buffer
// advanced to the element, and the address space travels with it.
[[seam::device_fn]] inline void
face_baraffwitkin(unsigned model,
                      const Mat3x2f &deformation, float mu,
                      float lambda, Mat3x2f *gradient_f,
                      Mat6x6f *hessian_f) {
    if (model != ELASTIC_MODEL_BARAFF_WITKIN) {
        return;
    }
    Mat3x2f gradient;
    Mat6x6f hessian;
    BaraffWitkin::material(deformation, mu, lambda, gradient, hessian);
    gradient_f[0] = gradient;
    hessian_f[0] = hessian;
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `face_baraffwitkin_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// FOUR ELEMENT GATHERS AND TWO ELEMENT DESTINATIONS. The destinations are
// strides of one rather than a scatter because a scatter carries the body's
// RETURN VALUE, and one call returns one value while this body has two outputs
// and a case where it writes neither.
[[seam::entry(count)]] void face_baraffwitkin(
    const unsigned *model,
    const Mat3x2f *deformation,
    const float *mu,
    const float *lambda,
    [[seam::stride(1)]] Mat3x2f *gradient_f,
    [[seam::stride(1)]] Mat6x6f *hessian_f,
    unsigned count);
