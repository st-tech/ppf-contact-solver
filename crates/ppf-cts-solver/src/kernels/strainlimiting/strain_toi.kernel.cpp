// File: strain_toi.kernel.cpp
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
// The arithmetic goes through `fmath::div`, `fmath::sqrt`, `fmath::min` and
// `fmath::max`, which each backend prologue defines with that backend's
// spelling. The first two are a correctness requirement rather than a naming
// convenience: under MSL they are `metal::precise::divide` and
// `metal::precise::sqrt`, because the default quotient and the default root are
// not correctly rounded.
#include "../contact/distance.hpp"
#include "../utility/face_deformation.kernel.cpp"
#include "../utility/svd3x2.kernel.cpp"

[[seam::device_fn]] inline float shell_effective_strain_limit(
    float strain_limit, float shrink_x, float shrink_y) {
    const float shrink_min = fmath::min(shrink_x, shrink_y);
    return fmath::div(strain_limit + 1.0f, shrink_min) - 1.0f;
}

// The two per-element strain readings, declared once each and rendered for four
// targets: the `__global__` nvcc compiles, the `kernel void` the Metal shader
// compiler compiles, the `ppf_<stem>_entry` shims a host C++ compiler compiles,
// and the Rust `#[repr(C)]` twins the driver fills.
//
// Each is element gathers into one element scatter, and neither is on the
// assembly path. The two bisections above call these bodies directly, once per
// probe, so what dispatches the entry points is the acceptance rig: it measures
// the value the line search is bisecting on through the shared body rather than
// through a second implementation of it.
//
// THE TWO TIMES OF IMPACT DECLARE THEIR ENTRIES BELOW, through the gated
// wrappers at the foot of this file. Each reads its element's positions through
// an index list, which `[[seam::indices]]` and `[[seam::through]]` express, and
// each gates on a non-positive limit, which is a branch on a physical quantity
// and so belongs in a body rather than in a launcher.
[[seam::entry(strain)]]
[[seam::device_fn]] inline float
shell_max_strain(const Mat3x2f &deformation) {
    Mat3x2f u;
    Vec2f sigma;
    Mat2x2f vt;
    svd3x2(deformation, u, sigma, vt);
    return fmath::max(sigma[0], sigma[1]) - 1.0f;
}

[[seam::device_fn]] inline float shell_strain_toi(
    const Vec3f &start0, const Vec3f &start1,
    const Vec3f &start2, const Vec3f &end0,
    const Vec3f &end1, const Vec3f &end2,
    const Mat2x2f &inverse_rest, float limit, float max_t) {
    const Mat3x2f f0 =
        face_deformation_gradient(start0, start1, start2, inverse_rest);
    const Mat3x2f f1 =
        face_deformation_gradient(end0, end1, end2, inverse_rest);
    const Mat3x2f df = f1 - f0;
    float t = max_t;
    if (shell_max_strain(f0 + t * df) >= limit) {
        float upper_t = t;
        float lower_t = 0.0f;
        float window = upper_t - lower_t;
        while (true) {
            t = 0.5f * (upper_t + lower_t);
            const float difference =
                shell_max_strain(f0 + t * df) - limit;
            if (difference < 0.0f) {
                lower_t = t;
            } else {
                upper_t = t;
            }
            const float new_window = upper_t - lower_t;
            if (new_window == window) {
                break;
            }
            window = new_window;
        }
        t = lower_t;
    }
    return t;
}

[[seam::device_fn]] inline float
rod_strain_value(const Vec3f &difference,
                     float rest_length) {
    return fmath::div(fmath::sqrt(difference.squaredNorm()), rest_length) -
           1.0f;
}

[[seam::device_fn]] inline float rod_strain_toi(
    const Vec3f &start0, const Vec3f &start1,
    const Vec3f &end0, const Vec3f &end1,
    float rest_length, float limit, float max_t) {
    const Vec3f d0 =
        proximity::difference<float, float>(start1, start0);
    const Vec3f d1 =
        proximity::difference<float, float>(end1, end0);
    const Vec3f dd = d1 - d0;
    float t = max_t;
    if (rod_strain_value(d0 + t * dd, rest_length) >= limit) {
        float upper_t = t;
        float lower_t = 0.0f;
        float window = upper_t - lower_t;
        while (true) {
            t = 0.5f * (upper_t + lower_t);
            const float difference =
                rod_strain_value(d0 + t * dd, rest_length) - limit;
            if (difference < 0.0f) {
                lower_t = t;
            } else {
                upper_t = t;
            }
            const float new_window = upper_t - lower_t;
            if (new_window == window) {
                break;
            }
            window = new_window;
        }
        t = lower_t;
    }
    return t;
}

[[seam::entry(count)]] void rod_strain_value(
    const Vec3f *difference,
    const float *rest_length,
    float *strain,
    unsigned count);

// THE TWO LINE-SEARCH TIMES OF IMPACT, each with its no-limit case decided here
// rather than by a launcher.
//
// THE GATE IS THE POINT OF THESE TWO WRAPPERS. An element whose limit is not
// positive has no strain limit, so it constrains the step not at all and its
// time of impact is the whole of `max_t`; the bisections above must not run
// there, because each opens by comparing a strain against a non-positive limit
// and would return a `t` at the bottom of the bracket, throttling a step no
// limit was asked to throttle. The rod also needs a positive rest length, which
// its strain is divided by.
//
// THE COMPARISONS ARE NEGATED ON PURPOSE and must stay that way: `!(limit > 0)`
// admits a NaN limit to the no-limit branch, where `limit <= 0` would send it
// into the bisection and return a NaN time of impact, which the line search
// would then take as its step.
//
// THE POSITIONS ARRIVE AS ELEMENTS, NOT AS BUFFERS. `[[seam::indices(N)]]`
// names the element's own slot list and `[[seam::through]]` passes another
// buffer's elements at those slots, so the entry point holds the only subscript
// and these bodies hold no index arithmetic. Both position buffers are read at
// the SAME slots, which is what lets one index list serve the start pose and
// the proposed one; exchanging the two measures the reverse sweep and returns a
// plausible number, so the parameter order here is load-bearing.
[[seam::device_fn]] inline float shell_strain_toi_gated(
    const Vec3f &start0, const Vec3f &start1,
    const Vec3f &start2, const Vec3f &end0,
    const Vec3f &end1, const Vec3f &end2,
    const Mat2x2f &inverse_rest, float effective_limit,
    float max_t) {
    if (!(effective_limit > 0.0f)) {
        return max_t;
    }
    return shell_strain_toi(start0, start1, start2, end0, end1, end2,
                                inverse_rest, effective_limit, max_t);
}

[[seam::device_fn]] inline float rod_strain_toi_gated(
    const Vec3f &start0, const Vec3f &start1,
    const Vec3f &end0, const Vec3f &end1,
    float rest_length, float limit, float max_t) {
    if (!(limit > 0.0f) || !(rest_length > 0.0f)) {
        return max_t;
    }
    return rod_strain_toi(start0, start1, end0, end1, rest_length, limit,
                              max_t);
}

// The same sweep with the shrink-corrected limit read from the face's own
// records.
//
// THE LINE SEARCH ASKED FOR THE LIMIT ARRAY AND SO OWED ITS UPLOAD. That is
// what this removes: the limit is `shell_effective_strain_limit` of three
// `FaceParam` fields reached through the face's own `param_index`, so a pass
// that holds the two records needs nothing staged for it. The gate is the
// limiter's own, `fixed` and `rest_excluded` and a positive authored limit,
// with NO `collider` test.
[[seam::device_fn]] inline float shell_strain_toi_from_records(
    const Vec3f &start0, const Vec3f &start1,
    const Vec3f &start2, const Vec3f &end0,
    const Vec3f &end1, const Vec3f &end2,
    const Mat2x2f &inverse_rest,
    const FaceProp &prop,
    const FaceParam *face_param, float max_t) {
    if (prop.fixed || prop.rest_excluded) {
        return max_t;
    }
    const FaceParam param = face_param[prop.param_index];
    if (!(param.strainlimit > 0.0f)) {
        return max_t;
    }
    const float limit = shell_effective_strain_limit(
        param.strainlimit, param.shrink_x, param.shrink_y);
    return shell_strain_toi_gated(start0, start1, start2, end0, end1, end2,
                                  inverse_rest, limit, max_t);
}

[[seam::entry(count)]] void shell_strain_toi_from_records(
    [[seam::through]] const Vec3f *start,
    [[seam::through]] const Vec3f *finish,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const Mat2x2f *inverse_rest,
    const FaceProp *prop,
    const FaceParam *face_param,
    float max_t,
    float *toi,
    unsigned count);

[[seam::entry(count)]] void shell_strain_toi_gated(
    [[seam::through]] const Vec3f *start,
    [[seam::through]] const Vec3f *finish,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const Mat2x2f *inverse_rest,
    const float *effective_limit,
    float max_t,
    float *toi,
    unsigned count);

[[seam::entry(count)]] void rod_strain_toi_gated(
    [[seam::through]] const Vec3f *start,
    [[seam::through]] const Vec3f *finish,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    const float *rest_length,
    const float *limit,
    float max_t,
    float *toi,
    unsigned count);
