// File: analytic_contact.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ belonging to no backend, rendered into the
// three forms the three compilers read by ppf-cts-compute/seam/kernelgen.py.
// The two facts a compiler cannot infer are C++ attributes,
// `[[seam::device_fn]]` for the execution space and `[[seam::thread]]` for the
// address space of a reference parameter. Every quantity here is one thread's
// own, so `thread` is the only address space that appears. Arithmetic is the
// `fmath::` table each backend prologue defines; `fmath::div` is the correctly
// rounded division on MSL, which is a correctness requirement rather than a
// preference.

#include "../energy/model/friction.kernel.cpp"
#include "../energy/model/push.kernel.cpp"

struct AnalyticContactResult {
    Vec3f force;
    Mat3x3f hessian;
    Mat3x3f friction_hessian;
    // Friction gradient alone, not folded into `force`. The SAND implicit
    // rolling Schur block contracts the skew against exactly this vector.
    Vec3f friction_gradient;
    Vec3f normal;
    float stiffness;
    bool valid_gap;
    // The normal force the friction was bounded by, which tells the caller to
    // record the pair in the friction's memory (`friction_memory.kernel.cpp`).
    // Zero where friction is off.
    float normal_force;
};

[[seam::device_fn]] inline float combine_friction_values(float a, float b,
                                                            unsigned mode) {
    if (mode == 1u) {
        return fmath::max(a, b);
    }
    if (mode == 2u) {
        return 0.5f * (a + b);
    }
    return fmath::min(a, b);
}

// THE RESIDUAL AN ANALYTIC CONTACT'S FRICTION READS, per vertex it names, so
// that each vertex's part is scaled by the contact's share of that vertex's
// cone (`friction_cone_fraction`) once the contact's normal force is known.
// `residual[k]` is the vertex's load (its residual plus its lagged non-friction
// contact load, `friction_load_settle`) already multiplied by its weight in the
// contact point, `weight[k]` the weight's magnitude, and `cone_sum[k]` and
// `cone_pairs[k]` the vertex's lagged cone sum and pair count. A vertex contact
// is one entry of weight one.
enum : unsigned { ANALYTIC_SHARE_VERTICES = 4u };
struct AnalyticShareInput {
    Vec3f residual[ANALYTIC_SHARE_VERTICES];
    float weight[ANALYTIC_SHARE_VERTICES];
    float cone_sum[ANALYTIC_SHARE_VERTICES];
    float cone_pairs[ANALYTIC_SHARE_VERTICES];
    unsigned count;
};

// THE ONE-VERTEX SLIP PREDICTION, which is `friction_slip_prediction<1>` with
// the weight fixed at one and the pair's elastic block already in hand: an
// analytic collider does not move, so the pair is the dynamic vertex alone and
// its own diagonal block plus inertia is the slip coordinate's stiffness. The
// caller supplies the vertex's residual by value, because it reads its own row
// once and a local pointer into a device array would need an address space.
//
// The enclosing test at every call site admits only a free vertex with mass, so
// there is no degree-of-freedom gate here. `mass <= 0` still returns false: a
// massless vertex belongs to a static solid and does not move.
[[seam::device_fn]] inline bool analytic_slip_prediction(
    const Mat3x3f &local_hessian, float mass,
    const Vec3f &residual, const Vec3f &normal, float dt,
    Vec3f &drive, float &stiffness) {
    drive = residual - normal * normal.dot(residual);
    const float drive2 = drive.squaredNorm();
    if (drive2 == 0.0f || mass <= 0.0f) {
        return false;
    }
    const Vec3f direction = drive.normalized();
    stiffness = friction_quadratic_form(local_hessian, direction) +
                fmath::div(mass, dt * dt);
    return stiffness > 0.0f;
}

[[seam::device_fn]] inline AnalyticContactResult
analytic_contact_evaluate(const Mat3x3f &local_hessian,
                              const Vec3f &slip,
                              // The slip this contact carried out of the last
                              // committed step, zero for none.
                              const Vec3f &carried,
                              // and how far it slid in that step.
                              float slide,
                              const Vec3f &normal,
                              float signed_distance, float physical_gap,
                              float ghat, float friction, float friction_eps,
                              float mass, const AnalyticShareInput &residual,
                              float dt,
                              bool kinematic,
                              bool include_friction) {
    AnalyticContactResult result;
    result.force = Vec3f::Zero();
    result.hessian = Mat3x3f::Zero();
    result.friction_hessian = Mat3x3f::Zero();
    result.friction_gradient = Vec3f::Zero();
    result.normal = normal;
    result.stiffness = 0.0f;
    result.valid_gap = physical_gap >= 0.0f;
    result.normal_force = 0.0f;
    if (signed_distance >= 0.0f) {
        return result;
    }

    if (kinematic) {
        result.stiffness = fmath::div(mass, ghat * ghat);
    } else if (result.valid_gap) {
        result.stiffness = normal.dot(local_hessian * normal) +
                           fmath::div(mass, physical_gap * physical_gap);
    } else {
        return result;
    }

    result.force =
        result.stiffness * push_gradient(signed_distance, normal, ghat);
    result.hessian =
        result.stiffness * push_hessian(signed_distance, normal, ghat);
    if (include_friction) {
        Vec3f tangent;
        Mat3x3f projection;
        float lambda;
        float stiffness;
        float contact;
        // THE ANCHOR THE FRICTION SURROGATE IS TIGHT AT, read off the residual
        // this vertex already carries. A prediction that does not resolve
        // leaves `drive_stiffness` at zero, which is the lagged surrogate.
        // EACH VERTEX'S SHARE, this contact's cone there over the vertex's
        // summed cone (`friction_cone_fraction`): the contact reads that part
        // of the vertex's residual and moves that part of its mass and
        // stiffness (`friction_slip_prediction` says why both). The caller has
        // already averaged the vertices' blocks and masses by weight, so the
        // stiffness takes the weight-averaged share.
        const float pair_cone =
            friction * fmath::max(0.0f, -normal.dot(result.force));
        Vec3f shared = Vec3f::Zero();
        float owned = 0.0f;
        float weight = 0.0f;
        for (unsigned k = 0; k < residual.count; ++k) {
            const float fraction = friction_cone_fraction(
                residual.weight[k] * pair_cone, residual.cone_sum[k],
                residual.cone_pairs[k]);
            shared = shared + fraction * residual.residual[k];
            owned = owned + residual.weight[k] * fraction;
            weight = weight + residual.weight[k];
        }
        Vec3f drive = Vec3f::Zero();
        float drive_stiffness = 0.0f;
        if (!analytic_slip_prediction(local_hessian, mass, shared, normal, dt,
                                          drive, drive_stiffness)) {
            drive_stiffness = 0.0f;
        } else if (owned != weight) {
            drive_stiffness = drive_stiffness * fmath::div(owned, weight);
        }
        const Vec3f held_slip =
            slip + (carried - normal * normal.dot(carried));
        friction_evaluate_anchored(
            result.force, held_slip, normal, friction, friction_eps, drive,
            drive_stiffness, slide, result.friction_gradient,
            result.friction_hessian,
            lambda, stiffness, tangent, projection, contact);
        if (friction > 0.0f) {
            result.normal_force = contact;
        }
        result.force += result.friction_gradient;
        result.hessian += result.friction_hessian;
    }
    return result;
}

[[seam::device_fn]] inline void
analytic_grain_schur(const Mat3x3f &friction_hessian,
                         const Vec3f &friction_gradient,
                         const Vec3f &normal, float radius,
                         Mat3x3f &angular,
                         Mat3x3f &coupling,
                         Vec3f &rotational_gradient) {
    const float trace = friction_hessian(0, 0) + friction_hessian(1, 1) +
                        friction_hessian(2, 2);
    if (trace <= 0.0f) {
        return;
    }
    Mat3x3f skew = Mat3x3f::Zero();
    skew(0, 1) = -normal[2];
    skew(0, 2) = normal[1];
    skew(1, 0) = normal[2];
    skew(1, 2) = -normal[0];
    skew(2, 0) = -normal[1];
    skew(2, 1) = normal[0];
    const Mat3x3f transpose_lambda = skew.transpose() * friction_hessian;
    angular += (radius * radius) * (transpose_lambda * skew);
    coupling += radius * (friction_hessian * skew);
    // Column-dot form instead of a Mat3x3f * Vec3f product: device Eigen
    // matrix-vector expressions can evaluate incorrectly, dot products are
    // safe. skew.col(k).dot(g) is component k of skew^T g.
    rotational_gradient +=
        radius * Vec3f(skew.col(0).dot(friction_gradient),
                       skew.col(1).dot(friction_gradient),
                       skew.col(2).dot(friction_gradient));
}

