// File: stvk.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef STVK_HPP
#define STVK_HPP

#ifndef SM_MSL_CONCAT
#include "../../common.hpp"
#include "../../data.hpp"
#include "detsqr.hpp"
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
#define STVK_UNDEF_INLINE
#endif
#ifndef SM_THREAD
#define SM_THREAD
#define STVK_UNDEF_THREAD
#endif

namespace StVK {

SM_INLINE float sqr(float x) { return x * x; }

SM_INLINE float energy(SM_THREAD const Vec2f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    return mu * (sqr(a0) * sqr(a0) + sqr(a1) * sqr(a1)) / 4.0f +
           detsqr::energy(a, lmd);
}

SM_INLINE float energy(SM_THREAD const Vec3f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    return mu * (sqr(a0) * sqr(a0) + sqr(a1) * sqr(a1) + sqr(a2) * sqr(a2)) /
               4.0f +
           detsqr::energy(a, lmd);
}

SM_INLINE Vec2f gradient(SM_THREAD const Vec2f &a) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    return Vec2f(a0 * a0 * a0, a1 * a1 * a1);
}

SM_INLINE Vec3f gradient(SM_THREAD const Vec3f &a) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    return Vec3f(a0 * a0 * a0, a1 * a1 * a1, a2 * a2 * a2);
}

SM_INLINE Mat2x2f hessian(SM_THREAD const Vec2f &a) {
    Mat2x2f result = Mat2x2f::Zero();
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    result(0, 0) = 3.0f * a0 * a0;
    result(1, 1) = 3.0f * a1 * a1;
    return result;
}

SM_INLINE Mat3x3f hessian(SM_THREAD const Vec3f &a) {
    Mat3x3f result = Mat3x3f::Zero();
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    result(0, 0) = 3.0f * a0 * a0;
    result(1, 1) = 3.0f * a1 * a1;
    result(2, 2) = 3.0f * a2 * a2;
    return result;
}

SM_INLINE DiffTable2 make_diff_table2(SM_THREAD const Vec2f &a, float mu,
                                      float lmd) {
    DiffTable2 table;
    DiffTable2 detsqr_table = detsqr::make_diff_table2(a, lmd);
    table.deda = mu * gradient(a) + detsqr_table.deda;
    table.d2ed2a = mu * hessian(a) + detsqr_table.d2ed2a;
    return table;
}

SM_INLINE DiffTable3 make_diff_table3(SM_THREAD const Vec3f &a, float mu,
                                      float lmd) {
    DiffTable3 table;
    DiffTable3 detsqr_table = detsqr::make_diff_table3(a, lmd);
    table.deda = mu * gradient(a) + detsqr_table.deda;
    table.d2ed2a = mu * hessian(a) + detsqr_table.d2ed2a;
    return table;
}

} // namespace StVK

#ifdef STVK_UNDEF_THREAD
#undef SM_THREAD
#undef STVK_UNDEF_THREAD
#endif

#ifdef STVK_UNDEF_INLINE
#undef SM_INLINE
#undef STVK_UNDEF_INLINE
#endif

#endif