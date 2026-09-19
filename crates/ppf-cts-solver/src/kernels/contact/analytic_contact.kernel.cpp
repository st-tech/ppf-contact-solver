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
                              const Vec3f &normal,
                              float signed_distance, float physical_gap,
                              float ghat, float friction, float friction_eps,
                              float mass, const Vec3f &residual, float dt,
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
        Vec3f drive = Vec3f::Zero();
        float drive_stiffness = 0.0f;
        if (!analytic_slip_prediction(local_hessian, mass, residual, normal, dt,
                                          drive, drive_stiffness)) {
            drive_stiffness = 0.0f;
        }
        friction_evaluate(
            result.force, slip, normal, friction, friction_eps, drive,
            drive_stiffness, result.friction_gradient, result.friction_hessian,
            lambda, stiffness, tangent, projection, contact);
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

// The friction-combining entry point, declared once and rendered for four
// targets. Two per-pair floats in, one out, with the rule selector in the
// record because one dispatch combines under one rule.
//
// IT NAMES THE BODY IT WRAPS, so the entry point is
// `combine_friction_values_entry`. An entry point that did not carry its
// body's name would be a second name for the same code, and a name covering
// two calling conventions is what a hand-written entry point costs.
//
// `mode` is `FrictionMode` as its underlying integer, 0 minimum, 1 maximum, 2
// mean, which is how the body already takes it: an enumeration is not one of
// the three scalars a record may carry, and widening it at the seam would be a
// second spelling of the same table.
[[seam::entry(count)]] void combine_friction_values(
    const float *a,
    const float *b, unsigned mode,
    float *out,
    unsigned count);
