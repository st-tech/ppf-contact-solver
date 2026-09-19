// File: velocity.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. The two facts a compiler cannot infer
// are C++ attributes: `[[seam::device_fn]]` is the execution space and
// `[[seam::thread]]` is the address space of a reference parameter. `Vec3f`,
// `float` and the `fmath::` names arrive from whichever backend prologue is in
// scope.
//
// THE INCOMING VELOCITY OF A STEP, AND THE TWO SCALARS MEASURED BESIDE IT.
//
// The integrator carries no velocity buffer. A step's incoming velocity is the
// divided difference of the two stored positions over the previous substep, so
// the first thing an advance does is form it once per vertex and reduce two
// scalars over the result: the largest speed, which sizes the substep, and how
// far the furthest vertex sits from the origin, which is the only per-step
// check that a position has not drifted out of the domain the scene is bounded
// to.
//
// WHY THIS IS A KERNEL BODY. It reads the step's live position arrays and folds
// two scalars out of them without either array leaving the device. Written in a
// backend's host language it would be a second implementation per backend, with
// no compiler checking that the three agree.

// The velocity a step starts with: the divided difference of the two stored
// positions over the substep that produced them.
//
// THE DIFFERENCE IS FORMED FIRST AND SCALED SECOND. The two stored positions
// agree in their leading digits, differing only by one substep of travel, so
// subtracting them cancels those digits and leaves the travel. Dividing each by
// `previous_dt` and subtracting the quotients is the same value algebraically
// and worse arithmetically: each quotient is rounded at the magnitude of a
// coordinate over a substep, and that rounding survives the cancellation. What
// it costs is zero at the origin and grows with distance from it, which is
// exactly the failure a test authored near 0 cannot see.
[[seam::device_fn]] inline Vec3f
vertex_velocity(const Vec3f &current,
                    const Vec3f &previous,
                    float previous_dt) {
    return (current - previous).cast<float>() / previous_dt;
}

// What one vertex contributes to the max-speed reduction, as a SQUARED speed:
// the reduction takes one square root at the end rather than one per vertex.
//
// A FIX-PINNED VERTEX CONTRIBUTES ZERO, and that is a statement about what the
// number is for. The maximum speed sizes the substep so that no free vertex
// travels further than the contact machinery can follow. A fix pin is an exact
// Dirichlet boundary condition: it is placed where its keyframe says, it does
// not integrate, and it can be commanded arbitrarily fast. Letting it into this
// maximum would shrink every substep in the scene to chase a vertex whose path
// is already known, so its speed is excluded here and its sweep is covered
// where a prescribed path is actually tested.
[[seam::device_fn]] inline float
vertex_speed_squared(const Vec3f &velocity,
                         unsigned fix_index) {
    return fix_index > 0 ? 0.0f : velocity.squaredNorm();
}

// How far this vertex sits from the origin, as the largest absolute coordinate.
// Reduced over the mesh, it is compared against the domain the scene is
// required to stay inside.
//
// ABSOLUTE BY CONTRACT, and this is one of the few places that is right. The
// magnitude relative to the ORIGIN is the entire measurement: what is being
// asked is whether a coordinate has left that domain, which is a property of
// the coordinate itself and not of any difference.
//
// The check exists because coordinate resolution falls off with magnitude: the
// gap between representable coordinates grows in proportion to the distance
// from the origin, so a position far enough out is quantized more coarsely than
// the clearances contact is measured at, and the run continues on numbers that
// no longer separate. Ingest is already bounded at scene build, so what this
// catches is a position that MOVED out during the run.
[[seam::device_fn]] inline float
coordinate_reach(const Vec3f &position) {
    return fmath::max(fmath::abs(position[0]),
                    fmath::max(fmath::abs(position[1]),
                             fmath::abs(position[2])));
}

// The three terms one vertex contributes, written where the step expects them.
//
// A COMPOSITION AND NOTHING ELSE. It calls the three bodies above in the order
// the step has always taken them, with the arguments they have always been
// given, so what it adds is a single declaration an entry point can be
// generated from rather than any arithmetic. The three above keep their own
// callers on the other backends untouched.
//
// THE ORDER IS PART OF THE ANSWER, not a style choice: `velocity` is read back
// as the step's velocity field while `speed_squared` and `reach` are folded to
// scalars, and both folds are fp32, so a term computed or written in a
// different order is a different number downstream.
//
// `prop` arrives as the whole `VertexProp` rather than as a bare `fix_index`
// because the driver holds one array of them and no flat index array; the body
// reads the one field, exactly as the range shim it replaces did.
//
// The three destinations are pointers to this element's own slot rather than
// references, which is the shape a generated entry hands a body whose output
// lives in device memory: `[[seam::stride(1)]]` is the buffer advanced to the
// element, and the address space travels with it. All three are written on
// every path, so none of them is the RETURN value a `[[seam::scatter]]` would
// carry; one call returns one value and this body has three outputs.
[[seam::device_fn]] inline void
velocity_terms(const Vec3f &current,
                   const Vec3f &previous,
                   const VertexProp &prop, float previous_dt,
                   Vec3f *velocity,
                   float *speed_squared,
                   float *reach) {
    const Vec3f u = vertex_velocity(current, previous, previous_dt);
    velocity[0] = u;
    speed_squared[0] = vertex_speed_squared(u, prop.fix_index);
    reach[0] = coordinate_reach(current);
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `velocity_terms_entry` shim a host C++ compiler compiles,
// and the Rust `#[repr(C)]` twin the driver fills.
//
// THREE ELEMENT GATHERS AND THREE ELEMENT DESTINATIONS. The two positions cross
// as `Vec3f`, whose 12 bytes the generated `[[seam::pod(12)]]` assertion pins
// in every C++ rendering, so a gather reads one position per element and the
// entry itself computes nothing: every value is formed inside the bodies above.
// `VertexProp` is 44 bytes with its own padding, which the same assertion
// pins.
[[seam::entry(count)]] void velocity_terms(
    const Vec3f *current,
    const Vec3f *previous,
    const VertexProp *prop,
    float previous_dt,
    [[seam::stride(1)]] Vec3f *velocity,
    [[seam::stride(1)]] float *speed_squared,
    [[seam::stride(1)]] float *reach,
    unsigned count);
