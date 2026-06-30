// File: sand_rigid.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// SAND grain spin (rolling). This module implements TWO coupling schemes:
//
//   * The implicit (Schur-condensed) rolling path (floor + dynamic sphere):
//     launch_condense_grains / launch_recover_grains below. The grain's angular
//     DOF is Schur-condensed into its 3x3 translation block INSIDE the Newton
//     solve, so the grain rolls at a near-textbook, bounded, non-pumping rate (no
//     per-scene tuning). This is the accurate path for single-grain analytic
//     contacts.
//
//   * The staggered (post-solve) rolling path (grain-grain):
//     launch_integrate_grains (this kernel). A two-body grain-grain contact would
//     need an off-diagonal coupling, so it uses the post-solve lagged-omega torque
//     integrate + no-slip / rolling-resistance clamp documented below. A grain
//     with an analytic contact is handled by the implicit rolling path and skipped
//     here (grain_A guard).
//
// Staggered integrator. After the translational contact solve converges, each
// grain's accumulated contact-friction torque grain_torque[i] (summed over its
// grain-grain contacts) spins its angular velocity grain_omega[i] by one
// semi-implicit step, then a no-slip clamp bounds it:
//
//   omega' = omega + (dt * Iinv * tau) / (1 + dt^2 * Iinv * kang + dt * Iinv * c_roll)
//   omega' = omega' * min(1, (|v_center| / radius) / |omega'|)        // no-slip clamp
//
// with Iinv = grain_inv_inertia the ROLLING (contact-point) inverse inertia
// 1/(I_center + m r^2) set in builder.rs, NOT the bare center inertia: the
// parallel-axis m r^2 is the constrained rolling DOF's generalized inertia and
// shapes the torque response. kang = sum lambda r^2 is the within-step
// friction-Hessian damping of tau's own omega-dependence; c_roll >= 0 is an
// optional linear rolling resistance.
//
// THE NO-SLIP CLAMP is what kills the energy-pumping ratchet. The friction at a
// contact is computed from the LAGGED omega (last step's value feeds fdx in
// contact.cu), so if the torque integrates omega ABOVE the rolling rate, the
// contact-point slip flips into the rolling-forward direction and kinetic
// friction (magnitude mu*N) points DOWN-slope and PROPELS the grain, doing net
// positive work (without the clamp a 20deg slope grain travels ~9x a frictionless
// slide). A grain rolling without slipping has surface spin speed radius*|omega|
// exactly equal to its center speed |v_center|; over-spin means radius*|omega| >
// |v_center|. So capping |omega| at |v_center|/radius forbids over-spin: the spin
// term radius*(omega x n)*dt in fdx can never exceed the realized center step, so
// the contact slip can never reverse and friction stays dissipative (travel
// bounded by a frictionless slide). The cap uses the realized center speed
// |v_center| = |curr - prev| / dt, which is CONTACT-AGNOSTIC: it holds no matter
// how many or what kind of surfaces the grain touches at once (floor, sphere,
// wall, or other grains in a pile), so a grain wedged between several contacts
// (small |v_center|) is correctly kept from spinning, while a freely rolling one
// spins up to the rolling rate. An airborne grain accumulates no friction torque
// (tau = 0), so it conserves its spin (the cap is loose: |v_center| is large).
//
// No orientation is tracked: a grain sphere is rotationally symmetric and the
// torque arm comes from the live contact normal (contact.cu), so omega alone
// closes the rolling loop, the lagged omega feeds the next step's contact-point
// friction. Shares PDRD's rotation primitives via rigid_core.hpp (none needed for
// the isotropic update, but the include keeps grains and PDRD on one module).

#ifndef SAND_RIGID_HPP
#define SAND_RIGID_HPP

#include "../../data.hpp"
#include "../../utility/dispatcher.hpp"
#include "rigid_core.hpp"

namespace SandRigid {

// One semi-implicit angular-velocity step per grain, dispatched over the
// surface vertices (grains live in [0, surface_vert_count)). Non-grain vertices
// have grain_inv_inertia == 0 and are skipped.
inline void launch_integrate_grains(const DataSet &data, float dt, float c_roll,
                                    float roll_resist) {
    unsigned n = data.surface_vert_count;
    DISPATCH_START(n)
    [data, dt, c_roll, roll_resist] __device__(unsigned i) mutable {
        float iinv = data.grain_inv_inertia[i];
        if (iinv <= 0.0f) {
            return; // not a grain
        }
        // Mixed-contact guard: a grain with an analytic (floor/sphere) contact
        // this step had its spin solved IMPLICITLY (the implicit rolling path,
        // condense/recover below); do NOT also apply the staggered integrate to
        // it. grain_A is nonzero only when an analytic contact contributed.
        Mat3x3f Ai = data.grain_A[i];
        if (Ai(0, 0) + Ai(1, 1) + Ai(2, 2) > 0.0f) {
            return;
        }
        Vec3f omega = data.grain_omega[i];
        Vec3f tau = data.grain_torque[i];
        float kang = data.grain_ang_stiff[i];
        // Semi-implicit: damp the friction torque's own omega-dependence
        // (stiffness kang = sum lambda*radius^2, with the dt^2 factor from
        // d tau / d omega) so omega converges to the rolling rate instead of
        // overshooting; c_roll adds optional linear rolling resistance.
        float scale =
            (dt * iinv) / (1.0f + dt * dt * iinv * kang + dt * iinv * c_roll);
        Vec3f omega_new = omega + scale * tau;

        // No-slip clamp (kills the energy-pumping ratchet; see header). Cap the
        // spin magnitude at the realized rolling rate |v_tangential|/radius so the
        // grain can never over-spin and let friction propel it. The tangential
        // (not full) center speed is used: |v_t| = |v - (v.n) n| with n the
        // dominant contact direction (the normalized SUM of all this grain's
        // contact normals, so it is correct under multiple simultaneous
        // contacts). Using |v_t| rather than |v| removes the normal-velocity
        // slack that would otherwise let a fast grain over-spin.
        const VertexProp &prop = data.prop.vertex[i];
        float radius = data.param_arrays.vertex[prop.param_index].offset;
        Vec3f step = (data.vertex.curr[i] - data.vertex.prev[i]).cast<float>();
        Vec3f nsum = data.grain_contact_normal[i];
        float nn = nsum.squaredNorm();
        Vec3f vt = step;
        if (nn > 0.0f) {
            Vec3f n = nsum * (1.0f / sqrtf(nn)); // normalized contact direction
            vt = step - n.dot(step) * n; // tangential center step
        }
        // nn == 0 means the grain touched nothing this step, so tau == 0 and
        // omega is unchanged; the loose |v|/dt cap then preserves airborne spin.
        //
        // Rolling-resistance UNDER-ROLL: cap omega at (1 - roll_resist) times the
        // no-slip rate |v_t|/radius, NOT the full no-slip rate. This is the key
        // anti-pump bound. Capping at exactly no-slip lets the lagged-omega
        // friction over-roll on the oscillating contact and PROPEL the grain (the
        // energy-pumping ratchet); capping slightly BELOW no-slip leaves a small
        // residual forward contact slip every step, so friction always opposes
        // motion (dissipative) and travel is bounded, while the grain still
        // visibly rolls. roll_resist in [0,1): 0 = textbook rolling (pumps here),
        // 1 = no spin (pure slide/stick). A modest value models the rolling
        // resistance real granular media have anyway (set via PPF_SAND_ROLL_RESIST).
        float cap = (radius > 0.0f)
                        ? (1.0f - roll_resist) * (vt.norm() / dt) / radius
                        : 0.0f;
        float wmag = omega_new.norm();
        if (wmag > cap) {
            // wmag > cap >= 0 implies wmag > 0, so the divide is safe.
            omega_new *= (cap / wmag);
        }
        data.grain_omega.data[i] = omega_new;
    } DISPATCH_END;
}

// ===========================================================================
// The implicit (Schur-condensed) rolling path for single-grain analytic
// contacts (floor + dynamic sphere). Grain-grain uses the staggered path above.
//
// A grain has translation DOF (the global Newton variable) and a local angular
// increment dtheta (omega = dtheta/dt). The contact-point slip is
//   u_c = P(dx - r (dtheta x n)) = P dx + r P [n]x dtheta,
// so the per-contact friction energy 1/2 lambda |u_c|^2 couples them. The
// contact.cu floor/sphere sites assemble, per grain, the angular block
//   A   = sum_c lambda_c r^2 (P_c [n_c]x)^T (P_c [n_c]x)      (grain_A; SPD part)
//   B   = sum_c lambda_c r   P_c [n_c]x                       (grain_B; coupling)
//   grad_rot = sum_c B_c^T (P_c dx)                           (grain_grot)
// Here we add the rotational inertia, condense dtheta out of the per-grain 6x6,
// and land the result on the grain's translation block:
//   A_full   = (I_center/dt^2) I3 + A          (SPD for any contact count)
//   g_theta  = grad_rot - (I_center/dt) omega_prev
//   diag_hess[i] += -B A_full^-1 B^T           (Schur correction, stays SPD)
//   force[i]     += -B A_full^-1 g_theta        (RHS reduction)
// Sign chain (verified): force = +gradient, diag_hess = +Hessian, solver solves
// H dx = force, Newton does eval_x -= dx; so the corrections take a leading minus
// and need no further negation. I_center = (2/5) m r^2 is the BARE center
// inertia; grain_inv_inertia_center holds its inverse 1/I_center. The
// parallel-axis m r^2 re-emerges from condensing against the translational mass
// already in the global system, so using I_eff here would double-count.
// omega_prev is the start-of-step snapshot (grain_omega_prev), held constant
// across Newton iterations.
// ===========================================================================

inline void launch_condense_grains(const DataSet &data, Vec<Mat3x3f> diag_hess,
                                   Vec<float> force, float dt,
                                   float spin_couple) {
    unsigned n = data.surface_vert_count;
    DISPATCH_START(n)
    [data, diag_hess, force, dt, spin_couple] __device__(unsigned i) mutable {
        float iinv_c = data.grain_inv_inertia_center[i];
        if (iinv_c <= 0.0f) {
            return; // not a grain
        }
        Mat3x3f A = data.grain_A[i];
        if (A(0, 0) + A(1, 1) + A(2, 2) <= 0.0f) {
            return; // no analytic (floor/sphere) contact this iteration
        }
        float i_center = 1.0f / iinv_c;
        float kin = i_center / (dt * dt);
        float fl = 1e-6f * kin + 1e-12f; // SPD conditioning floor
        A(0, 0) += kin + fl;
        A(1, 1) += kin + fl;
        A(2, 2) += kin + fl;
        Mat3x3f B = data.grain_B[i];
        Mat3x3f Ainv = A.inverse();
        diag_hess[i] += -(B * Ainv * B.transpose());
        // Force correction. grad_rot drives gravity-driven rolling; the
        // angular-momentum term -(I/dt) omega_prev lets the grain's existing spin
        // propel its translation (spin-to-translation), which sustains rolling
        // toward the textbook rate but, fully included (spin_couple=1), forms a
        // positive-feedback runaway (spin -> propulsion -> more spin). spin_couple
        // in [0,1] scales how much of that momentum is fed to the translation: 0 =
        // conservative (rolling decays, ~0.3x textbook), higher = closer to
        // textbook until it destabilizes. The spin itself always evolves with full
        // inertia in launch_recover_grains.
        Vec3f g_theta_force =
            data.grain_grot[i] -
            (spin_couple * kin * dt) * data.grain_omega_prev[i];
        Vec3f rhs = B * (Ainv * g_theta_force);
        Map<Vec3f>(force.data + 3 * i) += -rhs;
    } DISPATCH_END;
}

inline void launch_recover_grains(const DataSet &data, Vec<float> dx, float dt) {
    unsigned n = data.surface_vert_count;
    DISPATCH_START(n)
    [data, dx, dt] __device__(unsigned i) mutable {
        float iinv_c = data.grain_inv_inertia_center[i];
        if (iinv_c <= 0.0f) {
            return; // not a grain
        }
        Mat3x3f A = data.grain_A[i];
        if (A(0, 0) + A(1, 1) + A(2, 2) <= 0.0f) {
            return; // no analytic contact -> grain-grain integrate handles omega
        }
        float i_center = 1.0f / iinv_c;
        float kin = i_center / (dt * dt);
        float fl = 1e-6f * kin + 1e-12f;
        A(0, 0) += kin + fl;
        A(1, 1) += kin + fl;
        A(2, 2) += kin + fl;
        Mat3x3f B = data.grain_B[i];
        Vec3f g_theta = data.grain_grot[i] - (kin * dt) * data.grain_omega_prev[i];
        Mat3x3f Ainv = A.inverse();
        Vec3f dxi(dx.data[3 * i], dx.data[3 * i + 1], dx.data[3 * i + 2]);
        // Back-substitution: solve increment dtheta_s = A^-1 (g_theta - B^T dx);
        // the applied increment is its negation (Newton subtracts), so
        // omega = -A^-1 (g_theta - B^T dx) / dt.
        Vec3f dtheta = -(Ainv * (g_theta - B.transpose() * dxi));
        data.grain_omega.data[i] = dtheta * (1.0f / dt);
    } DISPATCH_END;
}

} // namespace SandRigid

#endif // SAND_RIGID_HPP
