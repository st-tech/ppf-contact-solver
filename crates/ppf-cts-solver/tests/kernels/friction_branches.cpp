// File: friction_branches.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`. It was `.cu` and ran a
// `__global__` at <<<1, 1>>>: one thread, no device work of any kind.
//
// IT CALLS THE NEUTRAL BODIES DIRECTLY. `analytic_contact.kernel.cpp` is
// included for the one-vertex slip prediction, which lives beside the analytic
// collider that is its only caller; it pulls in `friction.kernel.cpp` and
// `push.kernel.cpp`, both of which a host compiler reads.

#include "data.hpp"
#include "contact/analytic_contact.kernel.cpp"

#include <cmath>
#include <cstdio>

enum Result {
    STATIC_GRAD_X,
    STATIC_H_XX,
    STATIC_H_YY,
    STATIC_H_ZZ,
    KINETIC_GRAD_X,
    KINETIC_GRAD_Y,
    KINETIC_H_XX,
    KINETIC_H_YY,
    KINETIC_H_ZZ,
    KINETIC_H_XY,
    KINETIC_H_YX,
    INSIDE_CONE_H_XX,
    INSIDE_CONE_GRAD_X,
    NORMAL_DRIVE_H_XX,
    REST_OVER_CONE_GRAD_X,
    REST_OVER_CONE_H_XX,
    REST_OVER_CONE_H_YY,
    REST_OVER_CONE_H_ZZ,
    SLIDING_OVER_CONE_GRAD_X,
    SLIDING_OVER_CONE_H_XX,
    NO_PREDICTION_H_XX,
    SLIDE_OVER_PREDICTION_H_XX,
    SLIDE_UNDER_PREDICTION_H_XX,
    SLIDE_INSIDE_CONE_H_XX,
    PULLING_H_XX,
    VERTEX_PRED_STIFFNESS,
    VERTEX_PRED_DRIVE_X,
    VERTEX_PRED_DRIVE_Z,
    PAIR_PRED_STIFFNESS,
    PAIR_PRED_DRIVE_X,
    PAIR_PRED_PINNED,
    SHARE_LONE,
    SHARE_THIRD,
    SHARE_GROWN,
    SHARED_UNIT_DRIVE_X,
    SHARED_THREE_H_XX,
    UNSHARED_THREE_H_XX,
    SHARED_PAST_CONE_SLIP,
    N_RESULT,
};

// The two branch cases the numbers below pin: a slip inside `min_dx` (static,
// a tangential spring) and one past it (kinetic, saturated at mu * contact),
// then the anchor either side of the cone and both slip predictions.
void run(float *out, const float *residual) {
    const Vec3f normal(0.0f, 0.0f, 1.0f);
    const Vec3f contact_force(0.0f, 0.0f, -2.0f);
    const float mu = 0.4f;
    const float eps = 0.01f;
    const Vec3f no_drive = Vec3f::Zero();

    Vec3f gradient, tangent;
    Mat3x3f hessian, projection;
    float lambda = 0.0f, stiffness = 0.0f, contact = 0.0f;

    // No drive information: the classical lagged surrogate in both branches.
    friction_evaluate(contact_force, Vec3f(0.005f, 0.0f, 0.0f), normal, mu, eps,
                      no_drive, 0.0f, gradient, hessian, lambda, stiffness,
                      tangent, projection, contact);
    out[STATIC_GRAD_X] = gradient[0];
    out[STATIC_H_XX] = hessian(0, 0);
    out[STATIC_H_YY] = hessian(1, 1);
    out[STATIC_H_ZZ] = hessian(2, 2);

    friction_evaluate(contact_force, Vec3f(0.012f, 0.016f, 0.0f), normal, mu,
                      eps, no_drive, 0.0f, gradient, hessian, lambda, stiffness,
                      tangent, projection, contact);
    out[KINETIC_GRAD_X] = gradient[0];
    out[KINETIC_GRAD_Y] = gradient[1];
    out[KINETIC_H_XX] = hessian(0, 0);
    out[KINETIC_H_YY] = hessian(1, 1);
    out[KINETIC_H_ZZ] = hessian(2, 2);
    out[KINETIC_H_XY] = hessian(0, 1);
    out[KINETIC_H_YX] = hessian(1, 0);

    // A drive inside the cone (0.5 < 0.8) changes nothing.
    friction_evaluate(contact_force, Vec3f(0.005f, 0.0f, 0.0f), normal, mu, eps,
                      Vec3f(0.5f, 0.0f, 0.0f), 10.0f, gradient, hessian, lambda,
                      stiffness, tangent, projection, contact);
    out[INSIDE_CONE_H_XX] = hessian(0, 0);
    out[INSIDE_CONE_GRAD_X] = gradient[0];

    // A drive along the normal is not a tangential drive, however large.
    friction_evaluate(contact_force, Vec3f(0.005f, 0.0f, 0.0f), normal, mu, eps,
                      Vec3f(0.0f, 0.0f, 100.0f), 10.0f, gradient, hessian,
                      lambda, stiffness, tangent, projection, contact);
    out[NORMAL_DRIVE_H_XX] = hessian(0, 0);

    // At rest under a drive past the cone: the anchor is the sliding slip
    // (1.8 - 0.8) / 10 = 0.1, and the force is still zero.
    friction_evaluate(contact_force, Vec3f::Zero(), normal, mu, eps,
                      Vec3f(1.8f, 0.0f, 0.0f), 10.0f, gradient, hessian, lambda,
                      stiffness, tangent, projection, contact);
    out[REST_OVER_CONE_GRAD_X] = gradient[0];
    out[REST_OVER_CONE_H_XX] = hessian(0, 0);
    out[REST_OVER_CONE_H_YY] = hessian(1, 1);
    out[REST_OVER_CONE_H_ZZ] = hessian(2, 2);

    // Already sliding under the same drive: the anchor adds the current slip
    // (0.02 + 0.1), and the force stays saturated at the cone.
    friction_evaluate(contact_force, Vec3f(0.02f, 0.0f, 0.0f), normal, mu, eps,
                      Vec3f(1.8f, 0.0f, 0.0f), 10.0f, gradient, hessian, lambda,
                      stiffness, tangent, projection, contact);
    out[SLIDING_OVER_CONE_GRAD_X] = gradient[0];
    out[SLIDING_OVER_CONE_H_XX] = hessian(0, 0);

    // Nothing in the pair moves (stiffness 0 = no prediction): lagged form.
    friction_evaluate(contact_force, Vec3f::Zero(), normal, mu, eps,
                      Vec3f(1.8f, 0.0f, 0.0f), 0.0f, gradient, hessian, lambda,
                      stiffness, tangent, projection, contact);
    out[NO_PREDICTION_H_XX] = hessian(0, 0);

    // The last step's slide past the cone: the anchor adds the LARGER of it
    // and the predicted slip, so a slide of 0.5 over a prediction of 0.1 gives
    // 0.02 + 0.5, and a slide of 0.05 under it leaves 0.02 + 0.1.
    friction_evaluate_anchored(contact_force, Vec3f(0.02f, 0.0f, 0.0f), normal,
                               mu, eps, Vec3f(1.8f, 0.0f, 0.0f), 10.0f, 0.5f,
                               gradient, hessian, lambda, stiffness, tangent,
                               projection, contact);
    out[SLIDE_OVER_PREDICTION_H_XX] = hessian(0, 0);
    friction_evaluate_anchored(contact_force, Vec3f(0.02f, 0.0f, 0.0f), normal,
                               mu, eps, Vec3f(1.8f, 0.0f, 0.0f), 10.0f, 0.05f,
                               gradient, hessian, lambda, stiffness, tangent,
                               projection, contact);
    out[SLIDE_UNDER_PREDICTION_H_XX] = hessian(0, 0);
    // Inside the cone a slide changes nothing: the static row again.
    friction_evaluate_anchored(contact_force, Vec3f(0.005f, 0.0f, 0.0f), normal,
                               mu, eps, Vec3f(0.5f, 0.0f, 0.0f), 10.0f, 0.5f,
                               gradient, hessian, lambda, stiffness, tangent,
                               projection, contact);
    out[SLIDE_INSIDE_CONE_H_XX] = hessian(0, 0);

    // A contact force that pulls (a negative normal force) has no cone and must
    // not put a negative block into the system.
    friction_evaluate(Vec3f(0.0f, 0.0f, 2.0f), Vec3f(0.005f, 0.0f, 0.0f),
                      normal, mu, eps, no_drive, 0.0f, gradient, hessian, lambda,
                      stiffness, tangent, projection, contact);
    out[PULLING_H_XX] = hessian(0, 0);

    // The one-vertex prediction: elastic block diag(30, 0, 0), mass 2,
    // dt = 0.1, residual (3, 0, 5) against normal z. The drive is the
    // tangential residual (3, 0, 0), t = x, and the stiffness is
    // t^T K t + m / dt^2 = 30 + 200.
    Mat3x3f block = Mat3x3f::Zero();
    block(0, 0) = 30.0f;
    Vec3f vertex_drive = Vec3f::Zero();
    float vertex_stiffness = 0.0f;
    analytic_slip_prediction(block, 2.0f, Vec3f(3.0f, 0.0f, 5.0f), normal, 0.1f,
                             vertex_drive, vertex_stiffness);
    out[VERTEX_PRED_STIFFNESS] = vertex_stiffness;
    out[VERTEX_PRED_DRIVE_X] = vertex_drive[0];
    out[VERTEX_PRED_DRIVE_Z] = vertex_drive[2];

    // The pair prediction on a point-point pair (weights 1, -1) of equal masses
    // 2 with no elastic blocks, dt = 0.1, residuals x = +3 on the point and -1
    // on the other: the drive is (3 + 1) / |W|^2 = 2 and the stiffness
    // (200 + 200) / |W|^4 = 100, the reduced mass m / 2 over dt^2. The
    // frictionless slip 2 / 100 = 0.02 is the relative slip of the two bodies
    // moved by their own residuals, 3 dt^2 / m + 1 dt^2 / m.
    SVecf<6> pair_load;
    for (unsigned k = 0; k < 6u; ++k) {
        pair_load[k] = residual[k];
    }
    SVecf<2> weight;
    weight[0] = 1.0f;
    weight[1] = -1.0f;
    SVecf<2> mass;
    mass[0] = 2.0f;
    mass[1] = 2.0f;
    const SMatf<6, 6> no_elastic = SMatf<6, 6>::Zero();
    Vec3f pair_drive = Vec3f::Zero();
    float pair_stiffness = 0.0f;
    SVecf<2> whole;
    whole[0] = 1.0f;
    whole[1] = 1.0f;
    friction_slip_prediction<2>(weight, mass, no_elastic, pair_load,
                                whole, normal, 0.1f, pair_drive, pair_stiffness);
    out[PAIR_PRED_STIFFNESS] = pair_stiffness;
    out[PAIR_PRED_DRIVE_X] = pair_drive[0];
    // With the second vertex pinned (weight zero) only the first moves: drive
    // 3, stiffness 200.
    SVecf<2> pinned_weight;
    pinned_weight[0] = 1.0f;
    pinned_weight[1] = 0.0f;
    friction_slip_prediction<2>(pinned_weight, mass, no_elastic, pair_load,
                                whole, normal, 0.1f, pair_drive, pair_stiffness);
    out[PAIR_PRED_PINNED] = pair_stiffness * pair_drive[0];

    // THE SHARE OF A VERTEX'S RESIDUAL a pair reads. A vertex that carried one
    // pair gives it everything; one of three equal pairs reads a third; a pair
    // whose cone grew past the lagged sum is capped at the whole.
    out[SHARE_LONE] = friction_cone_fraction(0.8f, 0.7f, 1.0f);
    out[SHARE_THIRD] = friction_cone_fraction(0.8f, 2.4f, 3.0f);
    out[SHARE_GROWN] = friction_cone_fraction(3.0f, 2.4f, 3.0f);
    // A share of one is the prediction without it, bit for bit.
    {
        SVecf<2> unit;
        unit[0] = 1.0f;
        unit[1] = 1.0f;
        Vec3f a = Vec3f::Zero(), b = Vec3f::Zero();
        float ka = 0.0f, kb = 0.0f;
        friction_slip_prediction<2>(weight, mass, no_elastic, pair_load,
                                    unit, normal, 0.1f, a, ka);
        friction_slip_prediction<2>(weight, mass, no_elastic, pair_load,
                                    whole, normal, 0.1f, b, kb);
        out[SHARED_UNIT_DRIVE_X] = (a[0] == b[0] && ka == kb) ? 1.0f : 0.0f;
    }
    // THREE EQUAL PAIRS AT ONE VERTEX, each with cone 0.8, the vertex loaded
    // at 0.59 of the three cones together (x = 0.59 * 2.4 = 1.416): the
    // vertex holds. Each pair reading the whole residual sees 1.416 > 0.8,
    // releases, and takes the last step's slide 0.1 into its anchor; reading
    // its third, 0.472, it stays at the static stiffness 0.8 / 0.01.
    {
        SVecf<1> w1;
        w1[0] = 1.0f;
        SVecf<1> m1;
        m1[0] = 2.0f;
        const SMatf<3, 3> none = SMatf<3, 3>::Zero();
        SVecf<3> loaded;
        loaded[0] = 1.416f;
        loaded[1] = 0.0f;
        loaded[2] = 0.0f;
        SVecf<1> third;
        third[0] = friction_cone_fraction(0.8f, 2.4f, 3.0f);
        SVecf<1> all;
        all[0] = 1.0f;
        Vec3f drive = Vec3f::Zero();
        float k = 0.0f;
        friction_slip_prediction<1>(w1, m1, none, loaded, third, normal,
                                    0.1f, drive, k);
        friction_evaluate_anchored(contact_force, Vec3f(0.005f, 0.0f, 0.0f),
                                   normal, mu, eps, drive, k, 0.1f, gradient,
                                   hessian, lambda, stiffness, tangent,
                                   projection, contact);
        out[SHARED_THREE_H_XX] = hessian(0, 0);
        friction_slip_prediction<1>(w1, m1, none, loaded, all, normal,
                                    0.1f, drive, k);
        friction_evaluate_anchored(contact_force, Vec3f(0.005f, 0.0f, 0.0f),
                                   normal, mu, eps, drive, k, 0.1f, gradient,
                                   hessian, lambda, stiffness, tangent,
                                   projection, contact);
        out[UNSHARED_THREE_H_XX] = hessian(0, 0);
        // PAST THE SUMMED CONE each pair predicts the VERTEX's slip: load 3.0
        // over the three cones' 2.4 slides the vertex (3.0 - 2.4) / 200, and
        // a pair owning a third reads drive 1.0 against stiffness 200 / 3, so
        // its excess 0.2 predicts the same 0.003.
        SVecf<3> heavy;
        heavy[0] = 3.0f;
        heavy[1] = 0.0f;
        heavy[2] = 0.0f;
        friction_slip_prediction<1>(w1, m1, none, heavy, third, normal,
                                    0.1f, drive, k);
        out[SHARED_PAST_CONE_SLIP] = (drive[0] - 0.8f) / k;
    }
}

static bool near(float actual, float expected) {
    float scale = fmaxf(1.0f, fabsf(expected));
    return fabsf(actual - expected) <= 1e-5f * scale;
}

int main() {
    float result[N_RESULT] = {};
    // x components +3 and -1 on the two vertices; y and z zero.
    const float residual[6] = {3.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f};
    run(result, residual);

    // THE TWO BRANCHES DIFFER IN THE FORCE AND NOT IN THE SHAPE OF THE HESSIAN,
    // which is what the kinetic rows assert: both take the tangential stiffness
    // times P, with no off-diagonal term. The exact kinetic Hessian,
    // lambda * (P - s s^T), is singular along the slip and a Newton solve
    // without an energy line search cannot use it; friction.kernel.cpp carries
    // the argument and the measurements. If the five kinetic numbers are ever
    // "corrected" to 25.6, 14.4, 0, -19.2, -19.2, the singular form has
    // returned and a house of cards will not stand.
    //
    // WHAT MAKES A SATURATED CONTACT RELEASE IS THE ANCHOR, which the rows
    // after them pin: it departs from the lagged form only past the cone, by
    // exactly the sliding slip of the pair-local model, and never touches the
    // force.
    const float expected[N_RESULT] = {
        // contact = -n . f = 2, mu * contact = 0.8, min_dx = 0.01.
        // Static: |u| = 0.005, lambda = 0.8 / 0.01 = 80, gradient 80 * 0.005.
        0.4f, 80.0f, 80.0f, 0.0f,
        // Kinetic: |u| = 0.02, lambda = 0.8 / 0.02 = 40, so the force is
        // saturated at the cone while H = 40 * P = diag(40, 40, 0) with no
        // off-diagonal: the slip direction is special to the force, not to the
        // curvature.
        0.48f, 0.64f, 40.0f, 40.0f, 0.0f, 0.0f, 0.0f,
        // Inside the cone and along the normal: the static rows again.
        80.0f, 0.4f, 80.0f,
        // At rest past the cone: gradient 0, stiffness 0.8 / 0.1 = 8 on the
        // whole tangent plane, nothing along the normal.
        0.0f, 8.0f, 8.0f, 0.0f,
        // Sliding past the cone: force 0.8 (saturated), stiffness
        // 0.8 / (0.02 + 0.1).
        0.8f, 0.8f / 0.12f,
        // No prediction: 0.8 / min_dx.
        80.0f,
        // The last step's slide: 0.8 / (0.02 + 0.5), 0.8 / (0.02 + 0.1), and
        // inside the cone the static 80.
        0.8f / 0.52f, 0.8f / 0.12f, 80.0f,
        // Pulling: no cone, no block.
        0.0f,
        // The predictions.
        230.0f, 3.0f, 0.0f, 100.0f, 2.0f, 600.0f,
        // The shares, then the three pairs: 80 shared, 0.8 / (0.005 + 0.1)
        // unshared.
        1.0f, 1.0f / 3.0f, 1.0f, 1.0f, 80.0f, 0.8f / 0.105f, 0.003f,
    };
    const char *name[N_RESULT] = {
        "static gradient", "static Hxx", "static Hyy", "static Hzz",
        "kinetic gradient x", "kinetic gradient y", "kinetic Hxx",
        "kinetic Hyy", "kinetic Hzz", "kinetic Hxy", "kinetic Hyx",
        "inside-cone Hxx", "inside-cone gradient x", "normal-drive Hxx",
        "rest-over-cone gradient x", "rest-over-cone Hxx",
        "rest-over-cone Hyy", "rest-over-cone Hzz",
        "sliding-over-cone gradient x", "sliding-over-cone Hxx",
        "no-prediction Hxx", "slide-over-prediction Hxx",
        "slide-under-prediction Hxx", "slide-inside-cone Hxx", "pulling Hxx",
        "vertex prediction stiffness", "vertex prediction drive x",
        "vertex prediction drive z", "pair prediction stiffness",
        "pair prediction drive x", "pair prediction pinned",
        "share of a lone pair", "share of one of three", "share of a grown pair",
        "unit share is the unshared prediction", "three shared pairs Hxx",
        "three unshared pairs Hxx", "shared pair past the cone predicts the vertex slip",
    };
    for (int i = 0; i < N_RESULT; ++i) {
        if (!near(result[i], expected[i])) {
            fprintf(stderr, "%s: got %.9g, expected %.9g\n",
                    name[i], result[i], expected[i]);
            return 1;
        }
    }

    printf("friction surrogate and anchor passed\n");
    return 0;
}
