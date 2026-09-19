// File: air_damper.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef air_DAMP_HPP
#define air_DAMP_HPP

// Shared device source: compiled by nvcc as CUDA, by clang as host C++ and by
// the Metal shader compiler as MSL, from these exact bytes. The Metal driver
// hands the shader compiler one concatenated string with no filesystem behind
// it, so an #include is a compile error there and the host splices this file in
// itself, in dependency order.
#ifndef SM_MSL_CONCAT
#include "../../data.hpp"
#include "../../float_math.hpp"
#endif

// SM_INLINE, NOT A BARE `__device__`, AND THAT IS LINKAGE RATHER THAN A HINT.
// Under nvcc's device LTO a non-inline `__device__` function has EXTERNAL
// linkage, so two translation units of one device link that both include this
// header define the same symbol and nvlink refuses the link by mangled name.
// That is what a generated entry point makes: `main/momentum.kernel.cpp`
// declares one over a body that calls `wind_weight`, and the holding pen
// includes this header too, so the two objects meet at the device link. On the
// HOST the same defect is a duplicate symbol at the ordinary link. `hook.hpp`
// and `model/arap.hpp` carry the same guard for the same reason.
#ifndef SM_INLINE
#define SM_INLINE __device__ inline
#define AIR_DAMPER_UNDEF_INLINE
#endif
#ifndef SM_THREAD
#define SM_THREAD
#define AIR_DAMPER_UNDEF_THREAD
#endif
#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define AIR_DAMPER_UNDEF_DIV
#endif
// Sine of an argument of any size. The wind ramp's argument grows with the
// simulation clock, so it leaves the range a special-function unit resolves and
// has to be reduced first; each backend meets that contract its own way, which
// is why the name is on the seam rather than a call to one library's sine.
#ifndef SM_SIN_PERIODIC
#define SM_SIN_PERIODIC(x) fmath::sin_periodic(x)
#define AIR_DAMPER_UNDEF_SIN_PERIODIC
#endif

namespace air_damper {

// The wind's strength over time: a quarter-amplitude gust riding on a constant
// three quarters, so the field never reverses and never stalls.
SM_INLINE float wind_weight(float time) {
    float angle = 30.0f * time;
    float t = 0.25f;
    return t * (0.5f * (1.0f + SM_SIN_PERIODIC(angle))) + (1.0f - t);
}

SM_INLINE Mat3x3f get_proj_op(SM_THREAD const Vec3f &normal) {
    return Mat3x3f::Identity() - normal * normal.transpose();
}

// THE ONE FIELD ALL THREE OF THESE READ IS `air_friction`, AND EACH IS SPELLED
// AGAINST THAT FLOAT BECAUSE THIS HEADER MAY NOT NAME `ParamSet` AT ALL.
// `ParamSet` opens with an `f64` that MSL has no type for, so the only device
// spelling of it is a hand-written mirror that lives with the backend
// assembling the shader, and a declaration here naming the type compiles only
// where that mirror is spliced ahead of this file. The offline entry check
// compiles a neutral body against the neutral vocabulary alone and splices no
// mirror, so a parameter of that type is `unknown type name 'ParamSet'` there
// however well the assembled shader builds. A caller holding one passes
// `param.air_friction`.
SM_INLINE float face_energy(float dt, SM_THREAD const Vec3f &x1,
                             SM_THREAD const Vec3f &x0,
                             SM_THREAD const Vec3f &normal,
                             SM_THREAD const Vec3f &wind, float air_friction) {
    Vec3f z = (x1 - x0).cast<float>() - dt * wind;
    Mat3x3f P = get_proj_op(normal);
    float f = normal.dot(z);
    Vec3f g = P * z;
    return SM_DIV(0.5f * (f * f + air_friction * g.squaredNorm()), dt * dt);
}

SM_INLINE Vec3f face_gradient(float dt, SM_THREAD const Vec3f &x1,
                               SM_THREAD const Vec3f &x0,
                               SM_THREAD const Vec3f &normal,
                               SM_THREAD const Vec3f &wind,
                               float air_friction) {
    Vec3f z = (x1 - x0).cast<float>() - dt * wind;
    Mat3x3f P = get_proj_op(normal);
    return normal * normal.dot(z) / (dt * dt) +
           air_friction * P * z / (dt * dt);
}

// PSD by construction at any air_friction >= 0: a positive multiple of the
// rank-1 n n^T plus a positive multiple of the complementary projection P,
// which share eigenvectors and have eigenvalues 1/dt^2 and air_friction/dt^2.
SM_INLINE Mat3x3f face_hessian(float dt, SM_THREAD const Vec3f &normal,
                                float air_friction) {
    Mat3x3f P = get_proj_op(normal);
    return normal * normal.transpose() / (dt * dt) +
           air_friction * P / (dt * dt);
}

} // namespace air_damper

#ifdef AIR_DAMPER_UNDEF_INLINE
#undef SM_INLINE
#undef AIR_DAMPER_UNDEF_INLINE
#endif
#ifdef AIR_DAMPER_UNDEF_SIN_PERIODIC
#undef SM_SIN_PERIODIC
#undef AIR_DAMPER_UNDEF_SIN_PERIODIC
#endif
#ifdef AIR_DAMPER_UNDEF_DIV
#undef SM_DIV
#undef AIR_DAMPER_UNDEF_DIV
#endif
#ifdef AIR_DAMPER_UNDEF_THREAD
#undef SM_THREAD
#undef AIR_DAMPER_UNDEF_THREAD
#endif

#endif
