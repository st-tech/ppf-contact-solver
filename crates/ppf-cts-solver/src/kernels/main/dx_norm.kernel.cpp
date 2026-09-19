// File: dx_norm.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. `[[seam::device_fn]]` is the execution
// space and `[[seam::device]]` is the address space of a pointer parameter;
// `Vec3f` and `map` arrive from whichever backend prologue is in scope.

// The magnitude of one vertex's Newton search direction.
//
// The linear solve returns `dx` as one flat float array of 3N components, so
// the per-vertex triple is a view rather than a copy. Reduced over the mesh
// with a maximum, the result is the step-size budget the line search is handed:
// the direction is rescaled by `max_dx / max|dx_i|` when that exceeds the
// configured per-vertex ceiling, so no single vertex is asked to move further
// in one Newton step than the contact machinery can follow.
//
// The reduction takes the maximum of NORMS rather than of squared norms, and
// the square root is therefore per vertex. That is the arithmetic the reduction
// consumes today and it is kept as it is: the maximum of the squares selects
// the same vertex, but the value is what divides into `max_dx`, so moving the
// root would change the rescale factor's rounding on every step.
// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `dx_magnitude_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// `direction` arrives as a BASE POINTER because the body does its own
// addressing: it reads the three components at `3 * vert` as one `Vec3f` view,
// which is why the thread index is forwarded rather than spent on a gather.
// The result reaches memory the one way a return value can, at the thread
// index in the [[seam::scatter]] buffer, so the generated statement is the
// launcher's `magnitude[i] = dx_magnitude(direction, i)` character for
// character.
[[seam::entry(vert, magnitude)]]
[[seam::device_fn]] inline float
dx_magnitude(const float *direction, unsigned vert) {
    return map<Vec3f>(direction + 3 * vert).norm();
}
