// File: arap.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

// Corotated linear, not ARAP: the deviatoric term carries mu rather than
// 0.5*mu, and the volume term is 0.5*lmd*tr(strain)^2 in place of detsqr.
// This is the energy the Barrier-Free comparison runs against
// (https://arxiv.org/pdf/2512.12151, supplementary CubicBarrier/arap.hpp);
// the namespace keeps its ARAP name so every call site is unchanged.

#ifndef ARAP_HPP
#define ARAP_HPP

#ifndef SM_MSL_CONCAT
#include "../../common.hpp"
#include "../../data.hpp"
#endif

// SM_INLINE, NOT A BARE `__device__`, AND THAT IS LINKAGE RATHER THAN A HINT.
// Under nvcc's device LTO a non-inline `__device__` function has EXTERNAL
// linkage, so two translation units of one device link that both include this
// header define the same symbol and nvlink refuses the link by mangled name.
// That is not hypothetical here: a generated entry point compiles the body it
// wraps, and the library's kernel table compiles every argument record, which
// pulls each body in a second time. `model/baraffwitkin.hpp` carries the same
// guard for the same reason.
#ifndef SM_INLINE
#define SM_INLINE __device__ inline
#define ARAP_UNDEF_INLINE
#endif
#ifndef SM_THREAD
#define SM_THREAD
#define ARAP_UNDEF_THREAD
#endif

namespace ARAP {

SM_INLINE float energy(SM_THREAD const Vec2f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float trace_strain = a0 + a1;
    return mu * (a0 * a0 + a1 * a1) + 0.5f * lmd * (trace_strain * trace_strain);
}

SM_INLINE float energy(SM_THREAD const Vec3f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    float trace_strain = a0 + a1 + a2;
    return mu * (a0 * a0 + a1 * a1 + a2 * a2) +
           0.5f * lmd * (trace_strain * trace_strain);
}

SM_INLINE DiffTable2 make_diff_table2(SM_THREAD const Vec2f &a, float mu,
                                      float lmd) {
    DiffTable2 table;

    float trace_strain = (a[0] - 1.0f) + (a[1] - 1.0f);
    table.deda =
        2.0f * mu * (a - Vec2f::Ones()) + lmd * trace_strain * Vec2f::Ones();

    table.d2ed2a = 2.0f * mu * Mat2x2f::Identity();
    table.d2ed2a(0, 0) += lmd;
    table.d2ed2a(1, 1) += lmd;
    table.d2ed2a(0, 1) += lmd;
    table.d2ed2a(1, 0) += lmd;

    return table;
}

SM_INLINE DiffTable3 make_diff_table3(SM_THREAD const Vec3f &a, float mu,
                                      float lmd) {
    DiffTable3 table;

    float trace_strain = (a[0] - 1.0f) + (a[1] - 1.0f) + (a[2] - 1.0f);
    table.deda =
        2.0f * mu * (a - Vec3f::Ones()) + lmd * trace_strain * Vec3f::Ones();

    table.d2ed2a = 2.0f * mu * Mat3x3f::Identity();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            table.d2ed2a(i, j) += lmd;

    return table;
}

} // namespace ARAP

#ifdef ARAP_UNDEF_THREAD
#undef SM_THREAD
#undef ARAP_UNDEF_THREAD
#endif

#ifdef ARAP_UNDEF_INLINE
#undef SM_INLINE
#undef ARAP_UNDEF_INLINE
#endif

#endif
