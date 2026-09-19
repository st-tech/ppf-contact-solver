// File: position_step.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. `[[seam::device_fn]]` is the execution
// space, `[[seam::thread]]` and `[[seam::device]]` the address spaces of a
// reference and of a pointer parameter; `Vec3f`, `Vec3f`, `float` and `map`
// arrive from whichever backend prologue is in scope.

// THE POSITION UPDATE: apply a scaled Newton search direction to one position.
//
// This is the only place in the Newton loop where the iterate moves. The
// direction comes out of the linear solve as a flat float array of 3N
// components and `scale` is the step-size rescale the line search budget
// produced, so the arithmetic is a single fused multiply-and-subtract per
// coordinate. The subtraction is what the sign convention requires: the solve
// returns the correction to REMOVE from the iterate.
//
// THE DISPLACEMENT IS FORMED WHOLE AND APPLIED ONCE, WHICH IS THE WHOLE
// CONTENT OF THIS BODY. A scaled search direction is a DISPLACEMENT: small,
// translation-invariant, and carrying its own relative precision whatever
// coordinate it is about to land on. `scale` multiplies the DIRECTION and never
// the position, and the product meets the position in a single subtraction, so
// the iterate is rounded once per Newton step and the coordinate's magnitude
// enters no factor.
//
// The spelling is `position - delta`, which reads as "position minus
// displacement yields position". That is what the operation means, and it is
// the form that keeps the two roles apart.
[[seam::device_fn]] inline Vec3f
position_step(const Vec3f &position,
                  const float *direction, unsigned vert,
                  float scale) {
    return position - (scale * map<Vec3f>(direction + 3 * vert)).cast<float>();
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `position_step_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// POSITIONS MOVE AS WHOLE TRIPLES. `eval_x` is declared as a pointer to
// `Vec3f`, whose 12 bytes the generated `[[seam::pod(12)]]` assertion pins in
// every C++ rendering, so the gather reads one position and the scatter writes
// one back. The entry computes nothing: the body takes the SEARCH DIRECTION, a
// displacement, and subtracts it.
//
// `[[seam::gather]]` and `[[seam::scatter]]` on ONE parameter is the in-place
// element update the launcher this replaces spelled `x[i] = f(x[i], ...)`. The
// buffer is resolved once and the read happens inside the call, so the
// generated statement is the launcher's, character for character.
[[seam::entry(count, vert)]] void position_step(
    [[seam::gather]] [[seam::scatter]]
    Vec3f *eval_x,
    const float *direction, unsigned vert,
    float scale, unsigned count);
