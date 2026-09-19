// File: grain_pair.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read, and the build hands
// each compiler its own form. The two facts a backend cannot infer are written
// as C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a reference parameter. MSL
// requires the second on every reference and pointer type; CUDA and the host
// have one address space and are handed the same declarations with it removed.
//
// No include of its own, matching sand_rigid.kernel.cpp and friction.kernel.cpp
// beside it. `Vec3f` arrives from whatever declares it for the backend that is
// compiling, which is data.hpp under nvcc and on the host and the shader
// prologue's aliases under MSL.

// The GRAIN-GRAIN half of the SAND rolling coupling: what a point-point contact
// between two surface vertices, at least one of them a grain, contributes to
// and takes from the grain angular degree of freedom.
//
// The other half is analytic (a grain against a floor or an analytic sphere)
// and lives in analytic_contact.kernel.cpp, where the angular DOF is
// Schur-condensed into the grain's own translation block and solved inside the
// Newton system. A two-body contact cannot be written that way, because the
// coupling it needs is off-diagonal and the per-vertex diagonal has nowhere to
// put it, so this half is staggered: the contact reads the LAGGED angular
// velocity when it
// forms its slip, and hands back the friction torque that
// `sand_grain_integrate` advances omega with once the solve has converged.
//
// The three bodies below are the whole of that exchange:
//
//   1. `grain_pair_spin` reads the lagged omega of BOTH endpoints and turns
//      it into the surface velocity the two contact points have relative to
//      their centers.
//   2. `grain_pair_slip` subtracts the resulting displacement from the
//      relative center step, which is the no-slip relation
//      u = dx_rel - dt (r_a omega_a x n + r_b omega_b x n). A pair rolling at
//      exactly the no-slip rate has u = 0 and friction does nothing, which is
//      what rolling IS; anything else slips and friction resists it.
//   3. `grain_pair_contribution` turns the resulting friction gradient back
//      into the torque, the angular stiffness and the contact normal that the
//      post-solve integrate consumes.
//
// THE SIGN THAT LOOKS WRONG AND IS NOT. Both endpoints receive `r (n x g)` with
// the SAME `n` and the SAME `g`, rather than one of them receiving the
// negation. `n` is the a-side outward normal, so b's own outward normal is
// `-n`, and the friction gradient b feels is `-g` for the same reason its
// barrier force
// is; the two negations cancel in the cross product, and what is left is the
// two grains counter-rotating, which is what rolling contact does. The normal
// accumulator is the one place the sides genuinely differ, and it differs
// because it records a DIRECTION rather than a moment: each grain records the
// direction it is being pushed in, so `sand_grain_tangential_step` can
// remove it from that grain's own step.
//
// Nothing here reaches a special-function unit, a division or a root, so the
// whole module is bit-exact comparable across backends and its parity gate
// admits no tolerance.

// The lagged surface velocity of the pair's contact point, as a displacement
// rate. `normal` is the a-side outward contact normal. A non-grain endpoint
// contributes nothing, which is what makes a grain against an ordinary surface
// vertex the same code path as a grain against a grain: the other side simply
// has no angular velocity to fold in.
[[seam::device_fn]] inline Vec3f
grain_pair_spin(bool first_is_grain, float first_radius,
                    const Vec3f &first_omega,
                    bool second_is_grain, float second_radius,
                    const Vec3f &second_omega,
                    const Vec3f &normal) {
    Vec3f spin = Vec3f::Zero();
    if (first_is_grain) {
        spin += first_radius * first_omega.cross(normal);
    }
    if (second_is_grain) {
        spin += second_radius * second_omega.cross(normal);
    }
    return spin;
}

// The bound the spin displacement must satisfy, as one point of control.
//
// The spin-induced contact displacement must never exceed the realized
// tangential center displacement, or the slip reverses past no-slip and kinetic
// friction flips into a propelling direction, which is a net energy source.
// That bound is real and is enforced, but NOT here: clamping the displacement
// in the embed dead-locks a grain at stick, because to roll it has to move and
// it cannot move until it rolls. It is applied to omega in the post-solve
// integrate instead (`sand_grain_integrate`'s under-roll cap), so this
// body is a pass-through and exists to keep the two halves of the argument in
// one place.
[[seam::device_fn]] inline Vec3f
grain_pair_clamp_spin(const Vec3f &spin_displacement,
                          const Vec3f &relative_step,
                          const Vec3f &normal) {
    (void)relative_step;
    (void)normal;
    return spin_displacement;
}

// The contact-point slip the friction term is built from, in place of the plain
// relative center step an ordinary contact uses.
[[seam::device_fn]] inline Vec3f
grain_pair_slip(const Vec3f &relative_step,
                    const Vec3f &spin,
                    const Vec3f &normal, float dt) {
    return relative_step -
           grain_pair_clamp_spin(dt * spin, relative_step, normal);
}

// One endpoint's share of a converged point-point friction force.
//
// `multiplicity` is the contact count the embed actually deposited its force
// with, so the torque tracks the force that was applied rather than the force
// of one nominal pair. `flip_normal` is set for the SECOND endpoint, whose
// outward normal is the negation of the one the pair was measured with.
//
// `angular_stiffness` is the friction Hessian's scale seen through the moment
// arm, `r^2 lambda`. It is the within-step damping of the torque's own omega
// dependence, and the integrate divides by it, which is what makes omega
// approach the rolling rate rather than overshoot it.
[[seam::device_fn]] inline void grain_pair_contribution(
    float radius, bool flip_normal, const Vec3f &normal,
    const Vec3f &friction_gradient, float stiffness,
    float multiplicity, Vec3f &torque,
    float &angular_stiffness,
    Vec3f &contact_normal) {
    const Vec3f force = multiplicity * friction_gradient;
    torque = radius * normal.cross(force);
    angular_stiffness = radius * radius * (multiplicity * stiffness);
    contact_normal = flip_normal ? -normal : normal;
}
