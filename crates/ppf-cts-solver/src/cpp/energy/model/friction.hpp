// File: friction.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef FRICTION_HPP
#define FRICTION_HPP

#include "../../common.hpp"
#include "../../data.hpp"

// Regularized Coulomb friction for one contact. dx is the tangential slip of
// the pair over the step (relative to the start-of-step anchor), mu the
// combined friction coefficient, contact the normal force magnitude, and
// min_dx the static/kinetic transition slip (param.friction_eps). The force is
//   g(u) = mu * contact * u / max(min_dx, |u|),   u = P dx,
// a spring inside |u| <= min_dx (static branch) that saturates at mu * contact
// beyond it (kinetic branch), so |g| never exceeds the friction cone.
struct Friction {
    const Vec3f &dx;
    Mat3x3f P;
    Vec3f u;
    float lambda;
    float mu;
    float contact;
    __device__ Friction(const Vec3f &force_contact, const Vec3f &dx,
                        const Vec3f &normal, float mu, float min_dx)
        : dx(dx), mu(mu) {
        contact = -normal.dot(force_contact);
        P = get_projection(normal);
        // Project the slip in dot form rather than as a P * dx matvec: it is
        // the same quantity with one rounding instead of nine, and a small
        // dense matvec written as a device Eigen expression is a documented
        // silent-miscompile hazard.
        u = dx - normal * normal.dot(dx);
        if (mu > 0.0f) {
            // One expression covers both branches: a spring below min_dx,
            // saturating at mu * contact above it.
            lambda = mu * contact / fmaxf(min_dx, sqrtf(u.squaredNorm()));
        } else {
            lambda = 0.0f;
        }
    }
    __device__ Vec3f gradient() const { return lambda * u; }
    // The force above is the capped Coulomb force; the Hessian is the LAGGED
    // tangential stiffness lambda * P in BOTH branches, with lambda evaluated
    // at the current slip. This is deliberate, and it is NOT the exact Hessian
    // of the kinetic potential. Do not "correct" it to lambda * (P - s s^T).
    //
    // Along the slip direction s the kinetic potential mu * contact * |u| is
    // LINEAR, so its exact curvature there is zero and the exact Hessian
    // lambda * (P - s s^T) is SINGULAR in s. A Newton solve cannot use it: the
    // linear model has no minimizer along s, so the step is bounded only by
    // whatever inertia and contact stiffness happen to couple in, which in
    // practice means a converged solve commanding scene-scale slips (measured:
    // max_dx 9.6 on a 0.25-tall domino), the line search then truncating toi to
    // 1e-2..1e-3, and the solve degrading until the SPD guard trips.
    //
    // Retaining only a fraction kappa of the along-slip stiffness does not fix
    // this, it just moves the pole: with curvature kappa * lambda the Newton
    // update in s is u <- u (1 - 1/kappa) + T / (kappa * lambda), which
    // AMPLIFIES the current slip by |1 - 1/kappa| per iteration. Stability
    // needs |1 - 1/kappa| <= 1, i.e. kappa >= 1/2, and kappa = 1 (this code) is
    // the only choice that is both non-amplifying and free of a tuned constant.
    // kappa = 1e-3 amplifies 999x per iteration and reproducibly destroyed the
    // domino example.
    //
    // The cost of kappa = 1 is that a cone-saturated contact escapes the stick
    // geometrically rather than instantly: one iteration lands at u <- T / lambda
    // = |u| T / (mu * contact), a growth factor T / (mu * contact) > 1 whenever
    // the tangential load exceeds the cone, so it does escape, just not in a
    // single re-linearization. That is the standard semi-implicit friction
    // lagging and is the price of a nonsingular tangential system; the remedy
    // for a slow escape is more Newton iterations (the error reduction step),
    // never a singular or amplifying Hessian.
    //
    // lambda >= 0 and P = I - n n^T is a symmetric projector, so the block is
    // symmetric PSD (eigenvalues {0, lambda, lambda}) and SPD-by-assembly holds.
    __device__ Mat3x3f hessian() const { return lambda * P; }
    __device__ Mat3x3f get_projection(const Vec3f &normal) {
        return Mat3x3f::Identity() - normal * normal.transpose();
    }
};

#endif
