// File: logarithm.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef LOGARITHM_HPP
#define LOGARITHM_HPP

#ifndef SM_MSL_CONCAT
#include "../data.hpp"
#include "../float_math.hpp"
#endif

#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define LOG_BARRIER_UNDEF_DIV
#endif
// The CUDA fallback routes through fmath rather than calling logf directly.
// The library logf reduces its argument with double arithmetic, so a kernel
// that calls it emits I2F.F64 and DMUL even though nothing in the source says
// double. fmath::log is the hardware special-function unit, float throughout.
// See float_math.hpp for the accuracy this trades and why the barrier can
// afford it. A backend that defines SM_LOG itself is responsible for keeping
// its own spelling in single precision.
#ifndef SM_LOG
#define SM_LOG fmath::log
#define LOG_BARRIER_UNDEF_LOG
#endif
#ifndef SM_INFINITY
#define SM_INFINITY (std::numeric_limits<float>::infinity())
#define LOG_BARRIER_UNDEF_INFINITY
#endif

namespace logarithm {

__device__ static float energy(float g, float ghat, float offset) {
    g -= offset;
    if (g <= 0.0f) {
        return SM_INFINITY;
    } else if (g >= ghat) {
        return 0.0f;
    }
    return -(g - ghat) * (g - ghat) * SM_LOG(SM_DIV(g, ghat));
}

__device__ static float gradient(float g, float ghat, float offset) {
    g -= offset;
    if (g <= 0.0f) {
        return -SM_INFINITY;
    } else if (g >= ghat) {
        return 0.0f;
    }
    return SM_DIV((ghat - g) *
                      (2.0f * g * SM_LOG(SM_DIV(g, ghat)) + g - ghat),
                  g);
}

__device__ static float curvature(float g, float ghat, float offset) {
    g -= offset;
    if (g <= 0.0f) {
        return SM_INFINITY;
    } else if (g >= ghat) {
        return 0.0f;
    }
    return -2.0f * SM_LOG(SM_DIV(g, ghat)) +
           SM_DIV(ghat * (ghat + 2.0f * g), g * g) - 3.0f;
}

} // namespace logarithm

#ifdef LOG_BARRIER_UNDEF_INFINITY
#undef SM_INFINITY
#undef LOG_BARRIER_UNDEF_INFINITY
#endif
#ifdef LOG_BARRIER_UNDEF_LOG
#undef SM_LOG
#undef LOG_BARRIER_UNDEF_LOG
#endif
#ifdef LOG_BARRIER_UNDEF_DIV
#undef SM_DIV
#undef LOG_BARRIER_UNDEF_DIV
#endif

#endif
