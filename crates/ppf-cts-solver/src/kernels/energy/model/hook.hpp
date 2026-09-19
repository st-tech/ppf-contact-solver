// File: hook.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef HOOK_HPP
#define HOOK_HPP

// `proximity::difference`, unconditionally: both bodies below call it. Only an
// ANGLE include has to stay behind the role marker, because the run-time
// assembler neutralizes a quoted line as it splices, so naming this here costs
// the shipped shader nothing and lets an offline unit that includes this header
// get the namespace. Behind the marker that declaration is absent, `proximity::`
// resolves against MSL's builtin `distance` function, and the call then parses
// as a chain of comparisons, reporting four errors on a line that is correct.
// It sits ahead of the block below on purpose: `distance.hpp` names its own
// dependencies, so it compiles alone under nvcc from this position.
#include "../../contact/distance.hpp"

#ifndef SM_MSL_CONCAT
#include "../../common.hpp"
#include "../../data.hpp"
#endif

// SM_INLINE, NOT A BARE `__device__`, AND THAT IS LINKAGE RATHER THAN A HINT.
// Under nvcc's device LTO a non-inline `__device__` function has EXTERNAL
// linkage, so two translation units of one device link that both include this
// header define the same symbol and nvlink refuses the link by mangled name.
// That is what a generated entry point makes: `energy/rod_force.kernel.cpp`
// declares one over a body that calls `make_diff_table`, and the holding pen's
// `energy/energy.cu` includes the same body, so the two objects meet at the
// device link. `model/arap.hpp` carries the same guard for the same reason.
#ifndef SM_INLINE
#define SM_INLINE __device__ inline
#define HOOK_UNDEF_INLINE
#endif
#ifndef SM_THREAD
#define SM_THREAD
#define HOOK_UNDEF_THREAD
#endif
#ifndef SM_MAX
#define SM_MAX fmaxf
#define HOOK_UNDEF_MAX
#endif
#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define HOOK_UNDEF_DIV
#endif

namespace hook {

SM_INLINE float energy(SM_THREAD const Vec3f &x0,
                       SM_THREAD const Vec3f &x1, float l0) {
    Vec3f t = proximity::difference<float, float>(x1, x0);
    float r = SM_DIV(t.norm(), l0) - 1.0f;
    return 0.5f * r * r;
}

SM_INLINE void make_diff_table(SM_THREAD const Vec3f &x0,
                               SM_THREAD const Vec3f &x1, float l0,
                               float weight, SM_THREAD Mat3x2f &gradient,
                               SM_THREAD Mat6x6f &hessian) {
    Vec3f t = proximity::difference<float, float>(x1, x0);
    float l = t.norm();
    // Divided rather than scaled by a reciprocal, as `hook.hpp`'s `t / l` in
    // the reference is: one rounding per component instead of two.
    Vec3f n = t / l;
    Mat3x6f dtdx;
    dtdx << -Mat3x3f::Identity(), Mat3x3f::Identity();
    Vec3f dedt = (SM_DIV(l, l0) - 1.0f) * n;
    Vec6f g = dtdx.transpose() * n;
    float r = SM_DIV(l - l0, l);
    float c0 = SM_DIV(SM_MAX(0.0f, 1.0f - r), l0);
    float c1 = SM_DIV(SM_MAX(0.0f, r), l0);
    gradient.col(0) = -weight * dedt;
    gradient.col(1) = weight * dedt;
    hessian = weight * (c0 * g * g.transpose() + c1 * dtdx.transpose() * dtdx);
}

} // namespace hook

#ifdef HOOK_UNDEF_DIV
#undef SM_DIV
#undef HOOK_UNDEF_DIV
#endif
#ifdef HOOK_UNDEF_MAX
#undef SM_MAX
#undef HOOK_UNDEF_MAX
#endif
#ifdef HOOK_UNDEF_THREAD
#undef SM_THREAD
#undef HOOK_UNDEF_THREAD
#endif
#ifdef HOOK_UNDEF_INLINE
#undef SM_INLINE
#undef HOOK_UNDEF_INLINE
#endif

#endif
