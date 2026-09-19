// File: position_accept.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. `[[seam::device_fn]]` is the execution
// space and `[[seam::thread]]` the address space of a reference parameter;
// `Vec3f` and `float` arrive from whichever backend prologue is in scope.

// THE ACCEPT LERP: commit the fraction of a proposed motion the line search
// certified as collision-free.
//
// The Newton loop proposes a whole trajectory, from `origin` to `proposed`, and
// the CCD line search answers with the largest `fraction` in [0, 1] along it at
// which nothing has yet collided or violated a strain limit. This body is what
// spends that answer: the vertex is placed at `fraction` of the way along its
// own segment, and the guarantee the line search computed transfers to the
// committed state because the interpolation is along the very segment that was
// swept.
//
// TWO CALLERS, ONE BODY, AND THE SECOND IS NOT AN AFTERTHOUGHT. The Newton
// iteration commits from the pre-step positions toward the solved iterate. The
// PDRD rigidify commit then treats the snap onto the nearest exactly-rigid
// configuration as a trajectory of its own and runs it through the same line
// search, so it commits from the current iterate toward the rigid target with
// its own fraction. Both are the same operation with the endpoints named
// differently, and a partial commit in either case leaves a residual the next
// iterations remove.
//
// THE ORDER OF OPERATIONS IS THE POINT. Both endpoints are POSITIONS that agree
// in their leading digits, so the segment is formed by subtracting them first,
// the fraction scales that difference, and only then is the offset added back
// to `origin`. Interpolating the endpoints directly, as
// `(1 - fraction) * origin + fraction * proposed`, is the same value
// algebraically and rounds each product at the magnitude of an absolute
// coordinate, so the committed point would be quantized far more coarsely than
// the segment it sits on, by an amount proportional to the vertex's distance
// from the origin.
[[seam::device_fn]] inline Vec3f
position_accept(const Vec3f &origin,
                    const Vec3f &proposed, float fraction) {
    const Vec3f offset = fraction * (proposed - origin).cast<float>();
    return origin + offset.cast<float>();
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `position_accept_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// POSITIONS MOVE AS WHOLE TRIPLES. Both buffers are declared as pointers to
// `Vec3f`, whose 12 bytes the generated `[[seam::pod(12)]]` assertion pins in
// every C++ rendering, so a gather reads one position and the scatter writes
// one back. The entry computes nothing: the segment is formed inside the body.
//
// `proposed` carries [[seam::gather]] and [[seam::scatter]] together, which is
// the in-place element update the launcher spelled `x[i] = f(y[i], x[i], ...)`:
// the buffer is resolved once and read inside the call, so the generated
// statement is that one character for character.
[[seam::entry(count)]] void position_accept(
    const Vec3f *origin,
    [[seam::gather]] [[seam::scatter]]
    Vec3f *proposed,
    float fraction, unsigned count);
