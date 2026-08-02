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
    Mat3x3f P;
    Vec3f u;
    float lambda;
    float mu;
    float contact;
    bool kinetic;
    Vec3f slip_dir;
    Vec3f n;
    __device__ Friction(const Vec3f &force_contact, const Vec3f &dx,
                        const Vec3f &normal, float mu, float min_dx)
        : mu(mu), n(normal) {
        contact = -normal.dot(force_contact);
        P = get_projection(normal);
        // Project the slip in dot form rather than as a P * dx matvec: it is
        // the same quantity with one rounding instead of nine, and a small
        // dense matvec written as a device Eigen expression is a documented
        // silent-miscompile hazard.
        u = dx - normal * normal.dot(dx);
        float u_norm = sqrtf(u.squaredNorm());
        kinetic = u_norm > min_dx;
        slip_dir = kinetic ? u / u_norm : Vec3f::Zero();
        if (mu > 0.0f) {
            // One expression covers both branches: a spring below min_dx,
            // saturating at mu * contact above it.
            lambda = mu * contact / fmaxf(min_dx, u_norm);
        } else {
            lambda = 0.0f;
        }
    }
    __device__ Vec3f gradient() const { return lambda * u; }
    // The static branch is a tangential spring, so its Hessian is lambda * P.
    // In the kinetic branch the capped potential mu * contact * |u| is linear
    // along the slip direction s and curved only across it. Its exact Hessian
    // is therefore lambda * w w^T, where w = n x s. The outer-product form is
    // algebraically equal to lambda * (P - s s^T), but is PSD by construction
    // in fp32 instead of relying on a subtraction that can leak a small
    // negative eigenvalue.
    //
    // Keeping lambda * P after the force saturates gives the sliding direction
    // a stiffness the force does not have. A contact loaded only slightly past
    // the Coulomb cone then increases its slip by only T/(mu*N) per Newton
    // iteration, a ratio arbitrarily close to one, and can remain numerically
    // stuck against another hard constraint. The branch-consistent zero
    // curvature releases that contact in one re-linearization. The resulting
    // raw Newton candidate can be large when no other tangential stiffness is
    // present; main.cu bounds it by max_dx before the contact and strain CCD
    // line searches choose the feasible committed step.
    //
    // Both branches are symmetric PSD, with eigenvalues
    // {0, lambda, lambda} and {0, 0, lambda}, respectively.
    __device__ Mat3x3f hessian() const {
        if (kinetic) {
            Vec3f w = n.cross(slip_dir);
            return lambda * (w * w.transpose());
        }
        return lambda * P;
    }
    __device__ Mat3x3f get_projection(const Vec3f &normal) {
        return Mat3x3f::Identity() - normal * normal.transpose();
    }
};

#endif
