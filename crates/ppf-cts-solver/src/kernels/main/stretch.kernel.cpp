// File: stretch.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. `[[seam::device_fn]]` is the execution space
// and `[[seam::thread]]` is the address space MSL requires on every reference.
//
// THE STRETCH RATIO, per shell face and per rod segment.
//
// It is a DIAGNOSTIC and it is still a kernel body, because it is a value: the
// maximum over the mesh is the `max_sigma` channel a reader compares against
// the strain limit, and the whole point of that comparison is that two backends
// report the same number for the same pose. A per-backend spelling would make
// the channel incomparable, which is worse than not having it.

// One shell face's stretch, as the larger principal stretch scaled by the
// tighter of the two authored shrink factors.
//
// The shrink factors are the material's rest-shape scaling, so a face authored
// to sit at 0.9 of its drawn size reads 1.0 when it is at rest rather than 1.11.
// The TIGHTER of the two is used because the ratio is compared against one
// scalar limit and the tighter factor is the one that binds first.
[[seam::device_fn]] inline float shell_stretch_ratio(float sigma0,
                                                        float sigma1,
                                                        float shrink_x,
                                                        float shrink_y) {
    return fmath::max(sigma0, sigma1) * fmath::min(shrink_x, shrink_y);
}

// One rod segment's stretch: its current length over the length it was built
// with.
//
// The subtraction is what carries the signal, and it is taken directly on the
// two endpoints: an edge is small beside the coordinates it runs between, so
// differencing them cancels their leading digits and the remaining bits are
// what the ratio is formed from.
[[seam::device_fn]] inline float
rod_stretch_ratio(const Vec3f &x0,
                      const Vec3f &x1, float initial_length) {
    return fmath::div((x1 - x0).cast<float>().norm(), initial_length);
}

// THE GATE IS THE POINT OF THIS WRAPPER, exactly as it is for the two in
// `strainlimiting/strain_toi.kernel.cpp`. A segment with no initial length is
// one this indicator says nothing about, and the body above would divide by
// zero there. Reporting zero is what makes it invisible to the maximum the
// channel takes.
//
// A FIXED SEGMENT REACHES THIS WITH A ZERO INITIAL LENGTH, so one comparison
// serves both exclusions. That is the caller's encoding rather than a second
// meaning for zero: a segment the indicator skips has no rest length to report.
//
// THE COMPARISON IS NEGATED ON PURPOSE and must stay that way: `!(x > 0)` sends
// a NaN initial length to the zero branch, where `x <= 0` would divide by it and
// report a NaN into a maximum, which would then swallow the channel.
// The length-only gate, kept under its own name because the negated comparison
// below is load-bearing and has one home.
//
// THE COMPARISON IS NEGATED ON PURPOSE and must stay that way: `!(x > 0)` sends
// a NaN initial length to the zero branch, where `x <= 0` would divide by it and
// report a NaN into a maximum, which would then swallow the channel.
[[seam::device_fn]] inline float rod_stretch_ratio_gated_length(
    const Vec3f &x0, const Vec3f &x1,
    float initial_length) {
    if (!(initial_length > 0.0f)) {
        return 0.0f;
    }
    return rod_stretch_ratio(x0, x1, initial_length);
}

// THE REST LENGTH COMES OUT OF THE SEGMENT'S OWN RECORD, which is where the
// reference reads it, and the `fixed` gate with it. This tree materialized the
// length into a device array filled by a HOST LOOP over every rod on every
// step, out of `props[index].initial_length` and `props[index].fixed`, both
// fixed for the life of the run. So every step after the first recomputed the
// previous step's answer and paid an upload to deliver it.
[[seam::device_fn]] inline float rod_stretch_ratio_gated(
    const Vec3f &x0, const Vec3f &x1,
    const EdgeProp &prop) {
    // A FIXED SEGMENT REPORTS ZERO, which is the exclusion the host loop
    // spelled by leaving its slot at zero rather than a second meaning for it.
    if (prop.fixed) {
        return 0.0f;
    }
    return rod_stretch_ratio_gated_length(x0, x1, prop.initial_length);
}

// TWO POSITIONS READ THROUGH THE EDGE'S OWN INDEX PAIR, with the bound each slot
// is checked against carried in the record, the initial length as an element
// gather, and the ratio as the scatter that carries the body's return value.
//
// The CUDA orchestrator still holds its own copy of the gate in the launcher
// around its call to the ungated body, so the change that deletes that fork is
// rewriting that dispatch to call this one.
[[seam::entry(count)]] void rod_stretch_ratio_gated(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    const EdgeProp *prop,
    float *ratio,
    unsigned count);
