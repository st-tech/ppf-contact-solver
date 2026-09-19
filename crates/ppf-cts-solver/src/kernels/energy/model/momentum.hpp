// File: momentum.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef MOMENTUM_HPP
#define MOMENTUM_HPP

// QUOTED AND UNCONDITIONAL, because this header names `proximity::` and the
// shader compiler has to see it. `common.hpp` and `data.hpp` stay behind the
// guard: `vec/vec.hpp` spells a raw `T *`, which has no address space, so MSL
// cannot read them in any position.
#include "../../contact/distance.hpp"
#ifndef SM_MSL_CONCAT
#include "../../common.hpp"
#include "../../data.hpp"
#endif

// SM_INLINE, NOT A BARE `__device__`, AND THAT IS LINKAGE RATHER THAN A HINT.
// Under nvcc's device LTO a non-inline `__device__` function has EXTERNAL
// linkage, so two translation units of one device link that both include this
// header define the same symbol and nvlink refuses the link by mangled name.
// That is what a generated entry point makes: `main/momentum.kernel.cpp`
// declares one over a body that calls `gradient`, and the holding pen includes
// this header too, so the two objects meet at the device link. On the HOST the
// same defect is a duplicate symbol at the ordinary link. `hook.hpp`,
// `model/arap.hpp` and `model/air_damper.hpp` carry the same guard for the same
// reason; `contact/distance.hpp` does not need it, its functions being TEMPLATES
// and so already weak.
#ifndef SM_INLINE
#define SM_INLINE __device__ inline
#define MOMENTUM_UNDEF_INLINE
#endif
#ifndef SM_THREAD
#define SM_THREAD
#define MOMENTUM_UNDEF_THREAD
#endif
#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define MOMENTUM_UNDEF_DIV
#endif
#ifndef SM_MIN
#define SM_MIN fminf
#define MOMENTUM_UNDEF_MIN
#endif

namespace momentum {

SM_INLINE float energy(float dt, SM_THREAD const Vec3f &x,
                        SM_THREAD const Vec3f &y) {
    const Vec3f difference =
        proximity::difference<float, float>(x, y);
    return SM_DIV(0.5f * difference.squaredNorm(), dt * dt);
}

SM_INLINE Vec3f gradient(float dt, SM_THREAD const Vec3f &x,
                          SM_THREAD const Vec3f &y) {
    // DIVIDED, not multiplied by a reciprocal. `momentum.hpp`'s `gradient` in
    // the reference is `(x - y).cast<float>() / (dt * dt)`, one IEEE division
    // per component; forming `1 / dt^2` first and multiplying rounds twice, and
    // this is the largest single term of a free vertex's right-hand side. The
    // two siblings below already divide, so this also makes the header
    // consistent with itself.
    return proximity::difference<float, float>(x, y) / (dt * dt);
}

SM_INLINE Mat3x3f hessian(float dt) {
    return SM_DIV(1.0f, dt * dt) * Mat3x3f::Identity();
}

// Isotropic air friction: a drag on the step a vertex takes, mass free and
// independent of any surface the vertex belongs to, so unlike the aerodynamic
// term in air_damper.hpp it reaches every free vertex including one that
// carries no area. `y` is the Newton iterate and `x` the start-of-step
// position, so the difference is the step the drag opposes.
SM_INLINE Vec3f isotropic_drag_gradient(float dt, SM_THREAD const Vec3f &y,
                                         SM_THREAD const Vec3f &x,
                                         float isotropic_air_friction) {
    return isotropic_air_friction * (y - x).cast<float>() / (dt * dt);
}

// PSD for any isotropic_air_friction >= 0: a non-negative multiple of the
// identity.
SM_INLINE Mat3x3f isotropic_drag_hessian(float dt,
                                          float isotropic_air_friction) {
    return SM_DIV(isotropic_air_friction, dt * dt) * Mat3x3f::Identity();
}

// The `fix-xz` drag holds a vertex's horizontal position still once the vertex
// rises above `threshold`, and it is TWO halves: the position-side half moves
// the accepted step's x and z back toward the previous pose, and the momentum
// side below is the force and Hessian the Newton system carries for the same
// threshold. A backend that applies one without the other solves a system that
// does not describe the step it then takes, so the two ship together.
//
// The predicate is separate from the two terms because the caller must skip
// both when it is false: below the threshold the term is absent, not zero.
SM_INLINE bool fix_xz_active(SM_THREAD const Vec3f &y, float threshold) {
    return threshold && y[1] > float(threshold);
}

// Ramps in over the first unit of height above the threshold and saturates.
SM_INLINE float fix_xz_ramp(SM_THREAD const Vec3f &y, float threshold) {
    return SM_MIN(1.0f, static_cast<float>(y[1] - float(threshold)));
}

SM_INLINE Vec3f fix_xz_gradient(float dt, SM_THREAD const Vec3f &y,
                                 SM_THREAD const Vec3f &x, float mass,
                                 float threshold) {
    float t = fix_xz_ramp(y, threshold);
    Vec3f n(0.0f, 1.0f, 0.0f);
    Mat3x3f P = Mat3x3f::Identity() - n * n.transpose();
    return P * t * mass * (y - x).cast<float>() / (dt * dt);
}

// PSD for any t >= 0 and mass >= 0: a non-negative multiple of the projection
// P, which drops the vertical component and leaves the horizontal one. The
// branch that reaches it gives t > 0.
//
// The dt factor here scales OPPOSITELY to the gradient's, which divides. Both
// backends carry this expression, byte for byte, from the one source above; if
// it is ever changed it has to be changed for both at once, with the CUDA
// reference re-baselined first, never on one side to make a parity check pass.
SM_INLINE Mat3x3f fix_xz_hessian(float dt, SM_THREAD const Vec3f &y,
                                  float mass, float threshold) {
    float t = fix_xz_ramp(y, threshold);
    Vec3f n(0.0f, 1.0f, 0.0f);
    Mat3x3f P = Mat3x3f::Identity() - n * n.transpose();
    return P * t * mass * (dt * dt);
}

} // namespace momentum

#ifdef MOMENTUM_UNDEF_INLINE
#undef SM_INLINE
#undef MOMENTUM_UNDEF_INLINE
#endif
#ifdef MOMENTUM_UNDEF_MIN
#undef SM_MIN
#undef MOMENTUM_UNDEF_MIN
#endif
#ifdef MOMENTUM_UNDEF_DIV
#undef SM_DIV
#undef MOMENTUM_UNDEF_DIV
#endif
#ifdef MOMENTUM_UNDEF_THREAD
#undef SM_THREAD
#undef MOMENTUM_UNDEF_THREAD
#endif

#endif
