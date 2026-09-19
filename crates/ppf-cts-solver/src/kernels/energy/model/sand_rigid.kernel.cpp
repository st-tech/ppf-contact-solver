// File: sand_rigid.kernel.cpp
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
// No include of its own, matching push.kernel.cpp and friction.kernel.cpp
// beside it: `Vec3f` and `Mat3x3f` come from whatever declares them for the
// backend that is compiling, which is data.hpp under nvcc and on the host and
// the shader prologue's aliases under MSL.

// A SAND grain's angular degree of freedom, as one body for every backend.
//
// A grain is a sphere with isotropic inertia whose contact arm is the live
// contact normal, so its whole rotational state is the angular velocity omega;
// no orientation is stored, because a sphere has none that any later stage
// reads. Two schemes advance it, and which one a grain takes is decided by
// whether it had an ANALYTIC (floor or sphere) contact this Newton iteration:
//
//   * IMPLICIT, for a grain with an analytic contact. The contact assembly
//     accumulates the three Schur blocks below, `condense` folds them into the
//     grain's 3x3 translation block before the linear solve, and `recover`
//     back-substitutes omega out of the solved increment. The angular DOF is
//     then inside the Newton system rather than lagged behind it, which is what
//     makes the rolling rate bounded and free of per-scene tuning.
//
//   * STAGGERED, for a grain whose contacts are all grain-grain. A two-body
//     contact would need an off-diagonal coupling the per-vertex diagonal
//     cannot express, so `integrate` takes the converged friction torque after
//     the solve and advances omega by one semi-implicit step under a no-slip
//     cap. A grain with an analytic contact is excluded from it by the same
//     trace test `condense` admits one on, so exactly one scheme acts per
//     grain per step.
//
// EVERY DIVISION AND EVERY ROOT HERE GOES THROUGH `fmath::div` AND
// `fmath::sqrt`, which the backend prologue defines, and that is a correctness
// spelling rather than a style. MSL's plain `/` and `sqrt` are not correctly
// rounded even at mathMode Safe, while `precise::divide` and `precise::sqrt`
// are in every math mode and CUDA's are already; routing every one of them
// through the seam is what makes the two backends produce the same bits rather
// than nearly the same ones. Nothing here reaches a special-function unit, so
// with that spelling in place the whole module is bit-exact comparable across
// backends and its parity gate admits no tolerance.
//
// The sign chain, which the two Schur bodies depend on and which is easy to get
// backwards: `force` holds the GRADIENT, `diag_hess` holds the HESSIAN, the
// solver solves H dx = force, and Newton then applies eval_x -= dx. So the two
// corrections below carry a leading minus and need no further negation.

// The trace test that decides which scheme a grain takes. `angular` is the sum
// over the grain's analytic contacts of a positive semidefinite block, so its
// trace is positive exactly when at least one such contact contributed. The
// buffers are zeroed at the top of every Newton iteration, so a grain that had
// none reads zero here rather than the previous iteration's value.
[[seam::device_fn]] inline bool sand_grain_has_analytic_contact(
    const Mat3x3f &angular) {
    return angular(0, 0) + angular(1, 1) + angular(2, 2) > 0.0f;
}

// Condense: fold this grain's angular DOF into its own translation block.
//
//   A_full             = (I_center / dt^2) I3 + A
//   g_theta            = grad_rot - spin_couple (I_center / dt) omega_prev
//   hessian_correction = -B A_full^-1 B^T
//   force_correction   = -B A_full^-1 g_theta
//
// `I_center` is the BARE solid-sphere center inertia (2/5) m r^2, whose inverse
// the builder stores; the parallel-axis m r^2 that the rolling constraint adds
// re-emerges from condensing against the translational mass already in the
// global system, so using the effective inertia here would count it twice.
//
// The floor added alongside the inertia scale is a CONDITIONING term, not a
// safety one: A is positive semidefinite and the scale is strictly positive, so
// A_full is already SPD for any contact count, and the Hessian correction is
// then the Schur complement of an SPD 6x6 and leaves the assembled block SPD.
// Nothing here can introduce an indefinite direction, so no projection is
// applied on top of one. The floor keeps the inverse well conditioned when the
// scale itself is small, and it is proportional to that scale, so it moves with
// the system rather than fixing an absolute magnitude.
//
// `spin_couple` scales how much of the grain's existing angular momentum is fed
// back into its translation. Fully included it forms a positive-feedback
// runaway (spin drives translation drives more spin); at zero the rolling
// decays to about a third of the textbook rate. The spin itself always evolves
// with the full inertia in `sand_grain_recover`, which is why only this
// body scales it.
//
// THE FOUR STATEMENTS BUILDING A_full ARE REPEATED IN `recover` BELOW rather
// than factored into a helper the two share, and that is deliberate. A compiler
// free to contract a multiply-add contracts an expression tree, so the same
// arithmetic reached through a different call shape can land one unit in the
// last place away; measured here, hoisting these into a helper moved the
// condensed force by exactly that much under an aggressively contracting host
// build. Repeating them keeps the tree the CUDA site already had, which is what
// makes this refactor inert on the reference backend.
[[seam::device_fn]] inline void sand_grain_condense(
    const Mat3x3f &angular,
    const Mat3x3f &coupling,
    const Vec3f &rotational_gradient,
    const Vec3f &previous_omega, float inverse_center_inertia,
    float dt, float spin_couple,
    Mat3x3f &hessian_correction,
    Vec3f &force_correction) {
    Mat3x3f system = angular;
    const float center_inertia = fmath::div(1.0f, inverse_center_inertia);
    const float inertia_scale = fmath::div(center_inertia, dt * dt);
    const float conditioning_floor = 1.0e-6f * inertia_scale + 1.0e-12f;
    system(0, 0) += inertia_scale + conditioning_floor;
    system(1, 1) += inertia_scale + conditioning_floor;
    system(2, 2) += inertia_scale + conditioning_floor;
    const Mat3x3f inverse = system.inverse();
    hessian_correction = -(coupling * inverse * coupling.transpose());
    const Vec3f rhs = rotational_gradient -
                      (spin_couple * inertia_scale * dt) * previous_omega;
    force_correction = -(coupling * (inverse * rhs));
}

// Recover: read the angular velocity back out of the solved translation
// increment.
//
//   dtheta = -A_full^-1 (g_theta - B^T dx)
//   omega  = dtheta / dt
//
// A PINNED grain needs no special case. Its translation is an exact Dirichlet
// boundary condition, so the increment handed in here is the PRESCRIBED one and
// this is then the rotational equation of motion with the translation
// prescribed. The grain still spins, which is the physical answer: a nailed
// ball bearing rotates, and freezing it would be the fiction.
//
// The right-hand side here carries the FULL angular momentum term, unlike the
// condense above: what `spin_couple` limits is how much of the spin drives the
// translation, never how the spin itself evolves.
[[seam::device_fn]] inline Vec3f sand_grain_recover(
    const Mat3x3f &angular,
    const Mat3x3f &coupling,
    const Vec3f &rotational_gradient,
    const Vec3f &previous_omega,
    const Vec3f &increment, float inverse_center_inertia,
    float dt) {
    Mat3x3f system = angular;
    const float center_inertia = fmath::div(1.0f, inverse_center_inertia);
    const float inertia_scale = fmath::div(center_inertia, dt * dt);
    const float conditioning_floor = 1.0e-6f * inertia_scale + 1.0e-12f;
    system(0, 0) += inertia_scale + conditioning_floor;
    system(1, 1) += inertia_scale + conditioning_floor;
    system(2, 2) += inertia_scale + conditioning_floor;
    const Vec3f rhs =
        rotational_gradient - (inertia_scale * dt) * previous_omega;
    const Mat3x3f inverse = system.inverse();
    const Vec3f dtheta = -(inverse * (rhs - coupling.transpose() * increment));
    return dtheta * fmath::div(1.0f, dt);
}

// The dominant contact direction, and the center step with that direction
// removed.
//
// `normal_sum` is the SUM of the unit normals over all of this grain's contacts
// this step, so it is the direction the grain is constrained against under
// several simultaneous contacts (a corner, or the floor plus its neighbours in
// a pile) rather than whichever contact happened to be written last. A grain
// that touched nothing has a zero sum, and then the tangential step IS the
// step, which is the loose cap an airborne grain needs to keep its spin.
[[seam::device_fn]] inline Vec3f sand_grain_tangential_step(
    const Vec3f &center_step,
    const Vec3f &normal_sum) {
    const float squared = normal_sum.squaredNorm();
    if (squared > 0.0f) {
        const Vec3f direction =
            normal_sum * fmath::div(1.0f, fmath::sqrt(squared));
        return center_step - direction.dot(center_step) * direction;
    }
    return center_step;
}

// Integrate: one semi-implicit angular step for a grain whose contacts are all
// grain-grain, followed by the under-roll cap.
//
//   omega' = omega + dt Iinv tau / (1 + dt^2 Iinv kang + dt Iinv c_roll)
//   omega' = omega' min(1, cap / |omega'|)
//
// `Iinv` is the ROLLING (contact-point) inverse inertia 1/(I_center + m r^2),
// not the bare center inertia the two Schur bodies take: the parallel-axis term
// is the constrained rolling DOF's generalized inertia and is what shapes the
// torque response here. `kang` is the within-step friction-Hessian damping of
// the torque's own omega dependence, which is what makes omega approach the
// rolling rate instead of overshooting it.
//
// THE CAP IS WHAT KILLS THE ENERGY-PUMPING RATCHET, and it caps BELOW the
// no-slip rate rather than at it. Friction at a contact is computed from the
// LAGGED omega, so if the integrate leaves omega above the rolling rate the
// contact-point slip flips into the rolling-forward direction and kinetic
// friction points downslope and PROPELS the grain, doing net positive work. A
// grain rolling without slipping has surface speed radius*|omega| exactly equal
// to its tangential center speed, so capping at (1 - roll_resist) of
// |v_t|/radius leaves a small residual forward slip every step, friction always
// opposes the motion, and the travel stays bounded while the grain still
// visibly rolls. `roll_resist` in [0, 1): 0 is textbook rolling and pumps here,
// 1 is pure sliding.
[[seam::device_fn]] inline Vec3f sand_grain_integrate(
    const Vec3f &omega, const Vec3f &torque,
    const Vec3f &center_step,
    const Vec3f &normal_sum, float angular_stiffness,
    float inverse_rolling_inertia, float radius, float dt, float c_roll,
    float roll_resist) {
    const float scale =
        fmath::div(dt * inverse_rolling_inertia,
                 1.0f + dt * dt * inverse_rolling_inertia * angular_stiffness +
                     dt * inverse_rolling_inertia * c_roll);
    Vec3f updated = omega + scale * torque;
    const Vec3f tangential =
        sand_grain_tangential_step(center_step, normal_sum);
    const float cap =
        radius > 0.0f
            ? fmath::div(
                  (1.0f - roll_resist) * fmath::div(tangential.norm(), dt),
                  radius)
            : 0.0f;
    const float magnitude = updated.norm();
    if (magnitude > cap) {
        // magnitude > cap >= 0 implies magnitude > 0, so the divide is safe.
        updated *= fmath::div(cap, magnitude);
    }
    return updated;
}


// ===========================================================================
// THE THREE ENTRY POINTS, and why each needs a body of its own above the one it
// calls. The four bodies above take a grain's state as thread-space values, so
// something has to read that state out of the per-grain arrays and write the
// result back. That gather WAS the hand-written CUDA launch in
// `energy/model/sand_rigid.hpp`; below it is a neutral body per launch, which
// every backend renders from one declaration.
//
// EACH TAKES A NAME OF ITS OWN rather than overloading the body it calls. C++
// would accept the overload on arity and the build would be green, but the
// wiring census keys a neutral body on its NAME and cannot tell two rows apart.
// The `_row` suffix is what `solver/block_jacobi.kernel.cpp` already uses for
// exactly this.
//
// A BODY CARRIES `[[seam::device]]` AND NOTHING ELSE FROM THE ENTRY FAMILY.
// `[[seam::pod]]`, `[[seam::gather]]` and the rest are legal only inside a
// declaration carrying `[[seam::args]]`, so the widths appear once, below.
//
// `prop` AND `params` ARE BASE POINTERS, NOT GATHERS, because
// `params[prop[i].param_index]` is a two-level indirection and a record carries
// one index list, which is spent on neither of them. That is the spelling
// `contact/aabb.kernel.cpp` states for the same pair.

// The staggered (post-solve) rolling path, dispatched over the surface
// vertices: grains live in [0, surface_vert_count) and a non-grain has
// `inverse_rolling_inertia == 0`, which is the first guard. The second is the
// mixed-contact guard: a grain that had an analytic contact this step already
// had its spin solved implicitly by the two bodies below, so the staggered step
// must not also be applied to it.
[[seam::entry(i)]]
[[seam::device_fn]] inline void sand_grain_integrate_row(
    const float *inverse_rolling_inertia,
    const Mat3x3f *angular,
    const VertexProp *prop,
    const VertexParam *params,
    const Vec3f *curr, const Vec3f *prev,
    const Vec3f *torque,
    const Vec3f *normal_sum,
    const float *angular_stiffness,
    Vec3f *omega, float dt, float c_roll, float roll_resist,
    unsigned i) {
    const float iinv = inverse_rolling_inertia[i];
    if (iinv <= 0.0f) {
        return; // not a grain
    }
    // EVERY ELEMENT IS COPIED INTO THREAD SPACE BEFORE IT IS USED, and that is
    // a Metal requirement rather than a style: an array reached through a
    // `[[seam::device]]` base pointer yields a DEVICE-space lvalue, and MSL
    // cannot bind one to the thread-space reference these bodies take, nor
    // apply `operator-` to two of them. The copy is what the generator emits
    // for a `[[seam::gather]]` parameter anyway, so it costs nothing; these
    // stay base pointers because `params[prop[i].param_index]` is a two-level
    // indirection that a gather cannot express.
    const Mat3x3f angular_i = angular[i];
    if (sand_grain_has_analytic_contact(angular_i)) {
        return; // solved implicitly by the condense / recover pair
    }
    // The radius is the vertex's contact offset, and the realized travel is a
    // DIFFERENCE of two positions one step apart, so the rolling update is
    // built from a step-sized quantity rather than from either absolute
    // position.
    const float radius = params[prop[i].param_index].offset;
    const Vec3f curr_i = curr[i];
    const Vec3f prev_i = prev[i];
    const Vec3f step = (curr_i - prev_i).cast<float>();
    const Vec3f omega_i = omega[i];
    const Vec3f torque_i = torque[i];
    const Vec3f normal_sum_i = normal_sum[i];
    omega[i] = sand_grain_integrate(omega_i, torque_i, step, normal_sum_i,
                                    angular_stiffness[i], iinv, radius, dt,
                                    c_roll, roll_resist);
}

// The implicit path's first half: the Schur complement onto the grain's 3x3
// translation block and the matching reduction of the right-hand side.
//
// `diagonal` IS A NON-CONST GATHER AND NOT A SCATTER, which is the same
// distinction `utility/face_damping.kernel.cpp` carries: a gather hands the body
// `buffer[i]`, an lvalue, so a non-const one IS the write, and this pass ADDS
// into a block the elastic assembly already filled. `force` is a base pointer
// because a grain owns THREE consecutive floats in it rather than one element.
[[seam::entry(i)]]
[[seam::device_fn]] inline void sand_grain_condense_row(
    const float *inverse_center_inertia,
    const Mat3x3f *angular,
    const Mat3x3f *coupling,
    const Vec3f *rotational_gradient,
    const Vec3f *previous_omega,
    Mat3x3f *diagonal, float *force, float dt,
    float spin_couple, unsigned i) {
    const float iinv_c = inverse_center_inertia[i];
    if (iinv_c <= 0.0f) {
        return; // not a grain
    }
    // Thread-space copies, for the reason the integrate body above states.
    const Mat3x3f angular_i = angular[i];
    if (!sand_grain_has_analytic_contact(angular_i)) {
        return; // no analytic contact this iteration
    }
    const Mat3x3f coupling_i = coupling[i];
    const Vec3f rotational_gradient_i = rotational_gradient[i];
    const Vec3f previous_omega_i = previous_omega[i];
    Mat3x3f hessian_correction;
    Vec3f force_correction;
    sand_grain_condense(angular_i, coupling_i, rotational_gradient_i,
                        previous_omega_i, iinv_c, dt, spin_couple,
                        hessian_correction, force_correction);
    // READ, ADD, WRITE BACK rather than `+=` in place, and for the same
    // Metal reason as the copies above: `diagonal[i]` is a DEVICE-space
    // lvalue and the matrix type's compound assignment is declared for
    // thread space, so `+=` finds no viable overload there.
    Mat3x3f diagonal_i = diagonal[i];
    diagonal_i += hessian_correction;
    diagonal[i] = diagonal_i;
    for (int k = 0; k < 3; ++k) {
        force[3 * i + k] += force_correction[k];
    }
}

// The implicit path's second half, run after the Newton direction is known: the
// grain's angular increment recovered from its translation increment.
//
// A PINNED grain needs no special case. Its translation is an exact Dirichlet
// BC, so the row is prescribed and the condense pass's writes for it are
// overwritten; this body reads only `increment`, never the diagonal or the
// force, so substituting the prescribed increment IS the rotational equation of
// motion with the translation prescribed.
[[seam::entry(i)]]
[[seam::device_fn]] inline void sand_grain_recover_row(
    const float *inverse_center_inertia,
    const Mat3x3f *angular,
    const Mat3x3f *coupling,
    const Vec3f *rotational_gradient,
    const Vec3f *previous_omega,
    const float *increment, Vec3f *omega,
    float dt, unsigned i) {
    const float iinv_c = inverse_center_inertia[i];
    if (iinv_c <= 0.0f) {
        return; // not a grain
    }
    // Thread-space copies, for the reason the integrate body above states.
    const Mat3x3f angular_i = angular[i];
    if (!sand_grain_has_analytic_contact(angular_i)) {
        return; // the staggered integrate owns this grain's omega
    }
    const Mat3x3f coupling_i = coupling[i];
    const Vec3f rotational_gradient_i = rotational_gradient[i];
    const Vec3f previous_omega_i = previous_omega[i];
    const Vec3f increment_i(increment[3 * i], increment[3 * i + 1],
                            increment[3 * i + 2]);
    omega[i] = sand_grain_recover(angular_i, coupling_i,
                                  rotational_gradient_i, previous_omega_i,
                                  increment_i, iinv_c, dt);
}
