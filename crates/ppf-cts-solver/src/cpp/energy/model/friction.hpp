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
//
// THE HESSIAN IS A MAJORIZER, NOT THE EXACT SECOND DERIVATIVE. Past the cone
// the potential mu * contact * |u| is linear along the slip, so its exact
// Hessian, lambda * (P - s s^T), is singular there, and this solver has no
// energy line search to contain a Newton step in a direction the model does
// not bound. What it uses instead is the quadratic surrogate
//   S_a(u) = mu * contact * (|u|^2 + a^2) / (2 a),
// which lies above mu * contact * |u| for every u and every anchor a > 0
// (arithmetic-geometric mean), touches it at |u| = a, and has the uniformly
// positive tangential Hessian (mu * contact / a) * P. Minimizing a majorizer
// is what makes the step safe without a line search: a structure held up by
// nothing but friction stands still under it, and does not under the singular
// form (measured on a house of cards, peak vertex motion over 20 frames:
// 4.4e-3 with this surrogate, 6.0e-1 with lambda * w w^T).
//
// THE ANCHOR IS WHAT DECIDES HOW FAST A SATURATED CONTACT RELEASES. With
// a = |u| (the classical lagged choice) the surrogate is tight at the current
// slip, and a Newton step from slip u under tangential load T lands at
// u * T / (mu * contact): the slip only ever grows by that ratio per
// iteration. Every step starts from u = 0, so a contact loaded past the cone
// at rest starts each step at stiffness mu * contact / min_dx and, with the
// two assemblies a default step performs, moves min_dx * T / (mu * contact)
// per step FOREVER, whatever the load. That is a body hanging on an edge it
// should slide off, at a creep speed set by a numerical epsilon.
//
// The anchor used here is the slip the contact is being driven to:
//   a = |u| + max(0, |P drive| - mu * contact) / drive_stiffness,
// where drive is the residual already assembled on the pair's slip coordinate
// and drive_stiffness the stiffness the pair-local model assigns to that
// coordinate (slip_prediction below). Inside the cone the excess is
// zero and a = |u|, the lagged surrogate exactly, so a sticking contact is
// untouched. Past it, a is the sliding slip of the pair-local model, and the
// surrogate is tight where the answer is: from u = 0 one Newton step lands on
// (|drive| - mu * contact) / drive_stiffness, the physical slip, instead of
// min_dx * T / (mu * contact). The surrogate stays a majorizer for any a, and
// a >= |u| means it is never stiffer than the lagged one, so no direction is
// ever singular and the step along the slip is bounded by the frictionless
// slide of the same local model.
//
// The gradient does not depend on the anchor. It is the true force at the
// current slip, lambda * u, so the residual the solve sees is exact and only
// the curvature is modeled.
struct Friction {
    Mat3x3f P;
    Vec3f u;
    // Secant stiffness of the force: gradient() = lambda * u.
    float lambda;
    // Stiffness of the surrogate: hessian() = stiffness * P. Equal to lambda
    // inside the cone, at most lambda past it.
    float stiffness;
    float mu;
    float contact;
    Vec3f n;
    __device__ Friction(const Vec3f &force_contact, const Vec3f &dx,
                        const Vec3f &normal, float mu, float min_dx,
                        const Vec3f &drive, float drive_stiffness)
        : mu(mu), n(normal) {
        contact = -normal.dot(force_contact);
        P = get_projection(normal);
        // Project the slip in dot form rather than as a P * dx matvec: it is
        // the same quantity with one rounding instead of nine, and a small
        // dense matvec written as a device Eigen expression is a documented
        // silent-miscompile hazard.
        u = dx - normal * normal.dot(dx);
        const float u_norm = sqrtf(u.squaredNorm());
        const float cone = mu * contact;
        if (cone > 0.0f) {
            // One expression covers both branches: a spring below min_dx,
            // saturating at cone above it.
            lambda = cone / fmaxf(min_dx, u_norm);
            float anchor = u_norm;
            if (drive_stiffness > 0.0f) {
                const Vec3f drive_t = drive - normal * normal.dot(drive);
                const float excess = sqrtf(drive_t.squaredNorm()) - cone;
                if (excess > 0.0f) {
                    anchor += excess / drive_stiffness;
                }
            }
            stiffness = cone / fmaxf(min_dx, anchor);
        } else {
            // No normal force, or no friction: nothing to resist with. A
            // negative cone would put a negative block into the SPD system.
            lambda = 0.0f;
            stiffness = 0.0f;
        }
    }
    __device__ Vec3f gradient() const { return lambda * u; }
    // Both branches are symmetric PSD with eigenvalues {0, stiffness,
    // stiffness}: the normal direction is free, the whole tangent plane is
    // held. Retaining only a fraction of the along-slip stiffness does not
    // speed the release, it moves the pole: the update along the slip becomes
    // u <- u (1 - 1/kappa) + T / (kappa lambda), which amplifies the slip by
    // |1 - 1/kappa| per iteration and is unstable below kappa = 1/2. The
    // anchor above is the release mechanism; the shape of the Hessian is not.
    __device__ Mat3x3f hessian() const { return stiffness * P; }
    __device__ Mat3x3f get_projection(const Vec3f &normal) {
        return Mat3x3f::Identity() - normal * normal.transpose();
    }
};

// A quadratic form written out rather than as t.dot(M * t): a device Eigen
// matvec is a documented silent-miscompile hazard.
__device__ inline float quadratic_form(const Mat3x3f &M, const Vec3f &t) {
    float q = 0.0f;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            q += t[a] * M(a, b) * t[b];
        }
    }
    return q;
}

// THE PAIR-LOCAL PREDICTION behind the anchor. The slip coordinate of a
// contact is u = sum_i w_i x_i over the pair's vertices (w sums to zero for a
// pair of dynamic primitives, and is the barycentric weights of the dynamic
// side against a static one), and a contact force f is embedded as w_i f on
// vertex i. Restricted to the tangential drive direction t, the pair-local
// model is the quadratic form of the pair's own elastic blocks plus its
// inertia m / dt^2 on the direction W = (w_i t):
//   stiffness = W^T K W / |W|^4,   drive = (sum_i w_i r_i) / |W|^2,
// both in the parametrization x = x0 + s W / |W|^2 that makes s the slip
// itself, so the frictionless slip is drive / stiffness and the sliding slip
// (|drive| - mu N) / stiffness. For one free vertex against a static surface
// this is r_t / (t^T K_ii t + m / dt^2); for two equal free masses with no
// elastic coupling it is the relative slip of the two bodies exactly.
//
// The elastic blocks are the pair's own, with everything outside the pair held
// fixed, which is the same pair-local model the barrier stiffness uses and is
// stiffer than the true response for every stiff body: a shell vertex on a
// floor predicts the slip of a vertex held by its neighbors, not of the whole
// shell sliding. That is the side to err on. A prediction that is too stiff
// leaves the contact at the lagged surrogate, which is what it has today; a
// prediction that is too soft lets the contact slide past what friction should
// allow for one iteration, and does so exactly at an impact, where the pushes
// of the other contacts closing in the same iteration are not in the residual
// yet. With the elastic blocks included the house of cards stands; on inertia
// alone it drifts.
//
// A vertex without a degree of freedom, an exact fix pin or a massless vertex
// of a static solid, must carry w_i = 0: its row is eliminated, it does not
// move however hard it is pushed. Returns false when nothing in the pair
// moves or the residual has no tangential part, and the anchor then falls
// back to the current slip. `hess` is any block lookup with
// hess(i, j) -> Mat3x3f that reads zero outside its pattern; the solver passes
// its elastic FixedCSRMat.
template <unsigned N, typename Hess>
__device__ inline bool
slip_prediction(const SVec<unsigned, N> &index, const SVecf<N> &w,
                const SVecf<N> &mass, const Hess &hess,
                const Vec<float> &residual, const Vec3f &normal, float dt,
                Vec3f &drive, float &stiffness) {
    float w2 = 0.0f;
    Vec3f r_sum = Vec3f::Zero();
    for (unsigned i = 0; i < N; ++i) {
        if (w[i] != 0.0f) {
            const float *r = residual.data + 3 * index[i];
            r_sum += w[i] * Vec3f(r[0], r[1], r[2]);
            w2 += w[i] * w[i];
        }
    }
    if (w2 == 0.0f) {
        return false;
    }
    const Vec3f d = r_sum - normal * normal.dot(r_sum);
    const float d2 = d.squaredNorm();
    if (d2 == 0.0f) {
        return false;
    }
    const Vec3f t = d / sqrtf(d2);
    float k = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        if (w[i] == 0.0f) {
            continue;
        }
        k += w[i] * w[i] * mass[i] / (dt * dt);
        for (unsigned j = 0; j < N; ++j) {
            if (w[j] != 0.0f) {
                k += w[i] * w[j] *
                     quadratic_form(hess(index[i], index[j]), t);
            }
        }
    }
    drive = d / w2;
    stiffness = k / (w2 * w2);
    return stiffness > 0.0f;
}

// The one-vertex case with its elastic block already in hand (a dynamic vertex
// against a static face, a wall or a sphere): w = 1, so the formulas above
// reduce to the residual and t^T K t + m / dt^2.
__device__ inline bool slip_prediction(const Mat3x3f &local_hess, float mass,
                                       const float *r, const Vec3f &normal,
                                       float dt, Vec3f &drive,
                                       float &stiffness) {
    const Vec3f res(r[0], r[1], r[2]);
    drive = res - normal * normal.dot(res);
    const float d2 = drive.squaredNorm();
    if (d2 == 0.0f || mass <= 0.0f) {
        return false;
    }
    const Vec3f t = drive / sqrtf(d2);
    stiffness = quadratic_form(local_hess, t) + mass / (dt * dt);
    return stiffness > 0.0f;
}

#endif
