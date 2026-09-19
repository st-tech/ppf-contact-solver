// File: cubic.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CUBIC_HPP
#define CUBIC_HPP

#ifndef SM_MSL_CONCAT
#include "../data.hpp"
#endif

#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define CUBIC_UNDEF_DIV
#endif

namespace cubic {

__device__ static float energy(float g, float ghat, float offset) {
    g -= offset;
    float y = g - ghat;
    if (y < 0.0f) {
        return SM_DIV(-2.0f * (y * y * y), 3.0f * ghat);
    } else {
        return 0.0f;
    }
}

__device__ static float gradient(float g, float ghat, float offset) {
    g -= offset;
    float y = g - ghat;
    if (y < 0.0f) {
        return SM_DIV(-2.0f * y * y, ghat);
    } else {
        return 0.0f;
    }
}

__device__ static float curvature(float g, float ghat, float offset) {
    g -= offset;
    if (g - ghat < 0.0f) {
        return 4.0f * (1.0f - SM_DIV(g, ghat));
    } else {
        return 0.0f;
    }
}

} // namespace cubic

#ifdef CUBIC_UNDEF_DIV
#undef SM_DIV
#undef CUBIC_UNDEF_DIV
#endif

#endif
