// File: override_seed.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++, belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. The two facts a backend cannot infer
// are C++ attributes: `[[seam::device_fn]]` is the execution space and
// `[[seam::thread]]` is the address space of a reference parameter.
//
// No include of its own, matching grain_pair.kernel.cpp beside it. `Vec3f` and
// `float` arrive from whatever declares them for the backend that is
// compiling.

// THE KEYFRAME SEED: how a scripted linear or angular velocity enters an
// implicit step, and how a live position leaves it.
//
// The integrator reads the incoming velocity as `(curr - prev) / dt`, so there
// is no velocity buffer to write. A commanded velocity is injected by moving
// the PREVIOUS position: `prev = curr - v dt` makes the step start with exactly
// `v`. The three bodies here are that one relation, its rotational counterpart,
// and the readback the caller needs to place the rotation's pivot.
//
// WHY THIS IS A KERNEL BODY AND NOT HOST BOOKKEEPING. Every line below is
// arithmetic on a POSITION that already sits in device memory, so a host
// spelling would be a second implementation of it per backend, with no compiler
// checking that the three agree.
//
// TWO NUMERICAL RULES THE SEED FOLLOWS, both about where an absolute coordinate
// is allowed to enter an expression:
//
//   1. A position minus a DISPLACEMENT is a position, and that is the only
//      shape the seed is written in. `curr[k] - vx * dt` disturbs only the low
//      bits of the coordinate, because the travel over one substep is small
//      next to the coordinate it is subtracted from.
//   2. The rotation pivot is subtracted BEFORE the cross product, so only the
//      moment arm reaches a product. Expanding the same expression as
//      `omega x curr - omega x pivot` is algebraically identical and
//      numerically worse: each product carries the body's distance from the
//      origin, and the two then cancel, so the seeded velocity field would be
//      clean at the origin and progressively noisier away from it, in
//      proportion to that distance. Differencing first still loses the digits
//      the two coordinates share, but it loses them once, on the arm, rather
//      than in every product built from it.

// The linear seed: the previous position that makes this step start at the
// commanded velocity.
//
// `curr` is the start-of-step position and is not modified; the caller writes
// the result into `prev`. A vertex whose keyframe carries both a linear and an
// angular component takes this first and `override_angular_seed` second, so
// the two compose into a full rigid-velocity overwrite.
[[seam::device_fn]] inline Vec3f
override_velocity_seed(const Vec3f &curr, float vx,
                           float vy, float vz, float dt) {
    Vec3f seeded;
    seeded[0] = curr[0] - float(vx * dt);
    seeded[1] = curr[1] - float(vy * dt);
    seeded[2] = curr[2] - float(vz * dt);
    return seeded;
}

// The angular seed: a rigid spin field added on top of whatever `prev` already
// holds, which is `prev -= (omega x (curr - pivot)) dt`.
//
// It reads `prev` rather than `curr` for that reason: it ACCUMULATES onto the
// linear seed instead of replacing it. For a rigid body the field it adds is
// exactly rigid and survives the rigid refit; for a deformable it seeds a
// rotational velocity field.
//
// `omega.cross(arm)` expands to the same three fp32 expressions the CUDA
// launcher writes out by hand (`wy rz - wz ry`, `wz rx - wx rz`,
// `wx ry - wy rx`), in that operand order, so the shared spelling is not a
// re-derivation of them.
[[seam::device_fn]] inline Vec3f
override_angular_seed(const Vec3f &curr,
                          const Vec3f &prev, float wx,
                          float wy, float wz, float cx, float cy, float cz,
                          float dt) {
    const Vec3f pivot = Vec3f(float(cx), float(cy), float(cz));
    // The moment arm is translation-invariant: it only ever feeds
    // v = omega x r. So the pivot is subtracted here, ahead of the cross
    // product, and no absolute coordinate reaches a multiplication.
    const Vec3f arm = (curr - pivot).cast<float>();
    const Vec3f omega = Vec3f(wx, wy, wz);
    const Vec3f surface = omega.cross(arm);
    Vec3f seeded;
    seeded[0] = prev[0] - float(surface[0] * dt);
    seeded[1] = prev[1] - float(surface[1] * dt);
    seeded[2] = prev[2] - float(surface[2] * dt);
    return seeded;
}

// The live world position of one vertex, as the float triple the caller reads.
//
// ABSOLUTE BY CONTRACT, and this is one of the few places that is right. The
// consumer is the host's pivot solve for an angular keyframe: it takes a
// centroid, and for a principal-axis keyframe a covariance, over the listed
// vertices, and hands the resulting center straight back to
// `override_angular_seed`, which subtracts it from each vertex again. The
// pivot has to track the SIMULATED pose, so a build-time pose will not do, and
// a centroid is meaningless without the absolute magnitude. The precision that
// costs is bounded and is spent only on locating the pivot, never on the moment
// arm, which the body above forms as a difference.
[[seam::device_fn]] inline Vec3f
gather_position_absolute(const Vec3f &curr) {
    return Vec3f(curr[0], curr[1],
                 curr[2]);
}

// THE GATHER'S ENTRY POINT. One slot per element rather than a fixed run of
// them, which `[[seam::indices(1)]]` states and `[[seam::through]]` reads the
// position at; the body is handed that one element and returns the triple,
// which is the one shape a return value reaches memory in.
//
// THE BOUND IS THE POINT OF DECLARING IT THIS WAY. The range shim this replaces
// subscripted the position array with an unchecked index out of the caller's
// list, and Metal answers an out-of-bounds read with 0.0 rather than faulting,
// so a stale list produced a plausible position at the origin. `seed.rs` also
// checks the list on the host before dispatching, and the two are not
// redundant: that check runs where the list is built and this one runs where it
// is used, on every backend.
[[seam::entry(count)]] void gather_position_absolute(
    [[seam::through]] const Vec3f *curr,
    [[seam::indices(1)]] const unsigned *indices,
    [[seam::bound]] unsigned vertex_count,
    Vec3f *out,
    unsigned count);

// THE TWO SEEDS AT THEIR LISTED VERTICES, which is a WRITE AT A DATA-DRIVEN
// SLOT and has a lane: `prev` is a plain `[[seam::device]]` pointer with no
// access attribute, so the body is handed the array and writes `prev[vi]`.
// A write at a data-driven slot needs no attribute of its own: an unattributed
// device pointer is handed to the body whole and the body picks the slot.
//
// SERIAL BY CONTRACT, AND THE DECLARATION CANNOT SAY SO. The index list arrives
// from a keyframe and nothing proves it holds each vertex at most once. The
// linear seed is idempotent under a duplicate; the ANGULAR one reads `prev` and
// writes it back, so a duplicated index makes the result depend on which write
// lands last, and under a parallel partition it is a data race outright. A
// generated entry covers whatever range it is handed and says nothing about how
// that range may be cut, so the rule lives in the kernel table's
// `Scatter::Atomic` and BOTH ROWS MUST KEEP IT. The work is a keyframe-sized
// subset of the vertices, so serial costs nothing measurable.
//
// THE ORDER OF THE TWO IS LOAD-BEARING: the linear seed OVERWRITES and the
// angular one ACCUMULATES onto what it left, so a caller that ran them the
// other way would drop the linear component.
[[seam::entry(element)]]
[[seam::device_fn]] inline void override_velocity_seed_listed(
    const Vec3f *curr, Vec3f *prev,
    const unsigned *indices, float vx, float vy, float vz,
    float dt, unsigned element) {
    const unsigned vi = indices[element];
    // THREAD-SPACE COPY, and it is a Metal requirement rather than a style.
    // `curr` is a `[[seam::device]]` base pointer, so `curr[vi]` is a
    // DEVICE-space lvalue and MSL cannot bind one to the `[[seam::thread]]`
    // reference the seed body takes. The base pointer has to stay, because the
    // write is at a data-driven slot rather than at the thread index, so the
    // copy is the only option; it is what the generator emits for a
    // `[[seam::gather]]` parameter anyway, and it is the same value.
    const Vec3f curr_vi = curr[vi];
    prev[vi] = override_velocity_seed(curr_vi, vx, vy, vz, dt);
}

[[seam::entry(element)]]
[[seam::device_fn]] inline void override_angular_seed_listed(
    const Vec3f *curr, Vec3f *prev,
    const unsigned *indices, float wx, float wy, float wz,
    float cx, float cy, float cz, float dt, unsigned element) {
    const unsigned vi = indices[element];
    // Thread-space copies, for the reason the linear seed above states. Reading
    // `prev[vi]` into a local and writing the result back is the same
    // read-modify-write the accumulate already was, on the same value.
    const Vec3f curr_vi = curr[vi];
    const Vec3f prev_vi = prev[vi];
    prev[vi] = override_angular_seed(curr_vi, prev_vi, wx, wy, wz, cx, cy, cz,
                                     dt);
}
