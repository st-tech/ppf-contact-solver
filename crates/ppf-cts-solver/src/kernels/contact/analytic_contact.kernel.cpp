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

// THE FLOOR'S PUSH AS A FUNCTION OF THE GAP ALONE. With the stiffness the
// floor forms, `elastic + mass / gap^2`, the push's magnitude at gap `y` is
// `f(y) = (mass / y^2 + elastic) (ghat - y)^2 / ghat` inside the band and zero
// outside it. `elastic` is the lagged `up . H up`, held fixed over the step.
// The helpers below average it over a step's sweep in closed form.

// The logarithmic mean `L = (b - a) / ln(b / a)` of two positive gaps and its
// derivative in `b`. Near `a = b` the logarithm's ratio is close to one, where
// the device's single-precision log has an absolute error comparable to the
// answer, so the series in `t = (b - a) / (b + a)` takes over:
// `L = (a + b) / 2 * phi(t)`, `1 / phi = 1 + t^2/3 + t^4/5 + t^6/7 + ...`, cut
// where the next term is below single precision for `|t| <= 0.1`.
[[seam::device_fn]] inline float log_mean(float a, float b, float &slope) {
    const float sum = a + b;
    const float t = fmath::div(b - a, sum);
    if (fmath::abs(t) <= 0.1f) {
        const float t2 = t * t;
        const float phi = fmath::div(
            1.0f, 1.0f + t2 * (1.0f / 3.0f + t2 * (0.2f + t2 * (1.0f / 7.0f))));
        const float phi_t =
            -t * (2.0f / 3.0f + t2 * (0.8f + t2 * (6.0f / 7.0f))) * phi * phi;
        slope = 0.5f * phi + 0.5f * sum * phi_t * fmath::div(2.0f * a, sum * sum);
        return 0.5f * sum * phi;
    }
    const float log_ratio = fmath::log(fmath::div(b, a));
    slope = fmath::div(log_ratio - fmath::div(b - a, b), log_ratio * log_ratio);
    return fmath::div(b - a, log_ratio);
}

// The push's mean over the gaps `[a, b]`, both inside the band, and its
// derivative in `b`. The mass term's mean is
// `(mass / ghat) (ghat^2 / (a b) - 2 ghat / L + 1)`, written as
// `(ghat / G - 1)^2 + 2 ghat (1 / G - 1 / L)` with `G` the geometric mean so
// both parts are non-negative and neither cancels; its derivative in `b`
// simplifies to `mass (2 L_b / L^2 - ghat / (a b^2))`, which at `a = b` is half
// the push's own derivative there, as a mean's must be. The elastic term's
// mean is a cubic's difference quotient.
[[seam::device_fn]] inline float floor_push_mean(float a, float b, float ghat,
                                                  float mass, float elastic,
                                                  float &slope) {
    float log_slope;
    const float l = log_mean(a, b, log_slope);
    const float g = fmath::sqrt(a * b);
    const float ratio = fmath::div(ghat, g) - 1.0f;
    const float spread = fmath::max(fmath::div(1.0f, g) - fmath::div(1.0f, l), 0.0f);
    const float ra = ghat - a;
    const float rb = ghat - b;
    slope = mass * (fmath::div(2.0f * log_slope, l * l) - fmath::div(ghat, a * b * b)) -
            fmath::div(elastic * (ra + 2.0f * rb), 3.0f * ghat);
    return fmath::div(mass, ghat) * (ratio * ratio + 2.0f * ghat * spread) +
           fmath::div(elastic, 3.0f * ghat) * (ra * ra + ra * rb + rb * rb);
}

// THE SECANT PUSH: the push averaged over the gaps the step swept, from the
// committed gap `start` to the iterate's `end`, i.e. the barrier potential's
// secant slope over the step. A force evaluated this way does no more and no
// less work than the potential releases, so the contact cannot inject energy
// whatever the barrier's shape.
//
// The push is zero outside the band, so the part of the sweep past `ghat`
// contributes nothing and the mean is the in-band mean times the in-band share
// of the sweep. `curvature` is the derivative of the mean in `end`, negated,
// which is what the Newton step needs; it is exact in every case, so a step
// whose sweep is vanishingly short needs no special form: there the mean is
// the push itself and the curvature half its own.
[[seam::device_fn]] inline float floor_secant_push(float start, float end,
                                                    float ghat, float mass,
                                                    float elastic,
                                                    float &curvature) {
    curvature = 0.0f;
    if (start >= ghat && end >= ghat) {
        return 0.0f;
    }
    float slope;
    if (start < ghat && end < ghat) {
        const float mean = floor_push_mean(start, end, ghat, mass, elastic, slope);
        curvature = fmath::max(-slope, 0.0f);
        return mean;
    }
    const float sweep = end - start;
    if (end >= ghat) {
        // Leaving the band: the in-band part `[start, ghat]` is fixed, and the
        // mean falls as the sweep lengthens.
        const float inside = floor_push_mean(start, ghat, ghat, mass, elastic, slope);
        const float mean = fmath::div(ghat - start, sweep) * inside;
        curvature = fmath::max(fmath::div(mean, sweep), 0.0f);
        return mean;
    }
    // Entering the band: the in-band part is `[end, ghat]`, oriented from
    // `ghat` to `end`, so it is the mean over `[ghat, end]` in the order swept.
    const float inside = floor_push_mean(ghat, end, ghat, mass, elastic, slope);
    const float mean = fmath::div(end - ghat, sweep) * inside;
    curvature = fmath::max(
        -fmath::div(inside + (end - ghat) * slope - mean, sweep), 0.0f);
    return mean;
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

