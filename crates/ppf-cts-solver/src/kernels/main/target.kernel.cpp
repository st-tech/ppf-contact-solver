// File: target.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The two facts a backend cannot infer are written as
// C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` / `[[seam::device]]` are the address spaces of a reference
// and of a pointer parameter. MSL requires the second on every pointer and
// reference type; CUDA and the host have one address space and are handed the
// same declarations with it removed.
//
// The three positions and the gravity vector are the caller's own values, held
// in registers or on its stack, so they are `[[seam::thread]]`. The fix-pin
// array is GPU global memory, so it is `[[seam::device]]`. The distinction is
// what MSL needs and it is also what the code means: a reference into the
// caller's frame and a pointer into a buffer are different things.
//
// `Fix` is a template parameter because the fix-pin record differs by backend
// binding, and a template taking an address-space-qualified pointer is what MSL
// accepts here; the body reads one field of it.
//
// The seed a Newton solve starts from: where an unconstrained vertex would be
// at the end of the step under its own momentum and gravity, and exactly its
// prescribed position when the vertex is fix-pinned. The extrapolation is built
// as a DISPLACEMENT, `(current - previous) * ratio` plus the gravity term, and
// added to `current` once, so the coordinate's magnitude never multiplies
// anything.

template <class Fix>
[[seam::device_fn]] inline Vec3f compute_target(
    unsigned fix_index, const Vec3f &current,
    const Vec3f &previous, const Fix *fix,
    float dt, float previous_dt, const Vec3f &gravity,
    bool inactive_momentum) {
    if (fix_index > 0) {
        return fix[fix_index - 1].position;
    }
    if (inactive_momentum) {
        return current;
    }
    const float ratio = fmath::div(dt, previous_dt);
    const float dt_squared = dt * dt;
    Vec3f target;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        const float velocity_displacement =
            static_cast<float>(current[dimension] - previous[dimension]) *
            ratio;
        target[dimension] =
            current[dimension] +
            float(velocity_displacement + dt_squared * gravity[dimension]);
    }
    return target;
}

// The seed one vertex gets, with the fix-pin record named concretely.
//
// A COMPOSITION AND NOTHING ELSE. It calls the template above with the
// arguments that template has always been given, so what it adds is a
// declaration an entry point can be generated from rather than any arithmetic.
// The template keeps its own callers on the other backends untouched.
//
// WHY A SEPARATE FUNCTION RATHER THAN AN ENTRY ON THE TEMPLATE ITSELF. An entry
// declaration states one parameter list, and a template states a family of
// them; the concrete type has to be named somewhere, and naming it here keeps
// the template generic for the callers that still deduce it. `FixPair` is the
// one type this backend instantiates, and it is the same `repr(C)` record
// `data.rs` mirrors, whose 36 bytes the entry's `[[seam::pod(36)]]` pins.
//
// AND WHY IT CARRIES ITS OWN NAME RATHER THAN OVERLOADING THE TEMPLATE'S. The
// shared-wiring census keys a neutral body on its name and reports where each
// one is reached from, so two bodies spelled alike are two rows it cannot tell
// apart and it refuses the pair outright. C++ would have resolved the overload
// on arity; the census is the stricter reader, and the new function is the one
// that takes the new name because the template already has callers under its.
//
// GRAVITY ARRIVES AS THREE SCALARS, NOT AS A BUFFER, and that is the point of
// the shape rather than an accident of it. It is one scene-wide vector every
// thread reads identically, held on the host beside `dt`: routing it through a
// device allocation would cost a handle, an arena bounds check and a 12-byte
// allocation to deliver a value that is already a uniform. The body takes the
// `Vec3f` it has always taken; this assembles it in thread space.
[[seam::device_fn]] inline Vec3f
compute_target_seed(const Vec3f &current,
                        const Vec3f &previous,
                        unsigned fix_index,
                        const FixPair *fix, float dt,
                        float previous_dt, float gravity_x, float gravity_y,
                        float gravity_z, int inactive_momentum) {
    Vec3f gravity;
    gravity[0] = gravity_x;
    gravity[1] = gravity_y;
    gravity[2] = gravity_z;
    return compute_target<FixPair>(fix_index, current, previous, fix, dt,
                                       previous_dt, gravity,
                                       inactive_momentum != 0);
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `compute_target_entry` shim a host C++ compiler compiles,
// and the Rust `#[repr(C)]` twin the driver fills.
//
// POSITIONS CROSS AS WHOLE TRIPLES. `current`, `previous` and `target` are
// pointers to `Vec3f`, whose 12 bytes the generated `[[seam::pod(12)]]`
// assertion pins in every C++ rendering, so a gather reads one position and the
// scatter writes one back. The one quantity the template derives from a
// position is the velocity displacement, which is a DIFFERENCE of two of
// them.
//
// `fix` STAYS A BASE POINTER because the body does its own addressing: the pin
// it wants is at `fix_index - 1`, an index this declaration does not hold and
// which is meaningful only where `fix_index > 0`. A scene with no pins reaches
// the body with every index zero and the pointer is never dereferenced.
[[seam::entry(count)]] void compute_target_seed(
    const Vec3f *current,
    const Vec3f *previous,
    const unsigned *fix_index,
    const FixPair *fix, float dt,
    float previous_dt, float gravity_x, float gravity_y, float gravity_z,
    int inactive_momentum,
    Vec3f *target,
    unsigned count);
