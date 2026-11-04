// File: friction.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::thread]]` is the address space of a reference parameter,
// which MSL requires on every reference and pointer type.
//
// No include of its own, matching push.kernel.cpp beside it. Two things arrive
// from the includer instead. `Vec3f` and `Mat3x3f` come from whatever declares
// them for the backend that is compiling, which is data.hpp under nvcc and on
// the host and the shader prologue's aliases under MSL. The `fmath::`
// arithmetic the body calls (fmath::sqrt, fmath::div, fmath::max) comes from
// the backend prologue, which is seam/seam.hpp under nvcc and on the host
// (data.hpp includes it first) and kMslMacroSeam in
// ppf-cts-compute/metal/shader_compiler.mm under MSL. On MSL those two names are the correctly rounded precise:: forms,
// which is a correctness requirement rather than caution.

// `slide` is how far the contact slid in the last committed step, from the
// friction's memory (`contact/friction_memory.kernel.cpp`), and zero for a
// contact that has none; `friction_evaluate` below is this with no slide.
[[seam::device_fn]] inline void friction_evaluate_anchored(
    const Vec3f &contact_force,
    const Vec3f &slip, const Vec3f &normal,
    float friction, float minimum_slip,
    const Vec3f &drive, float drive_stiffness, float slide,
    Vec3f &gradient,
    Mat3x3f &hessian, float &lambda, float &stiffness,
    Vec3f &tangent, Mat3x3f &projection,
    float &contact) {
    contact = -normal.dot(contact_force);
    projection =
        Mat3x3f::Identity() - normal * normal.transpose();
    tangent = slip - normal * normal.dot(slip);
    const float tangent_norm = fmath::sqrt(tangent.squaredNorm());
    // NO BRANCH ON THE SLIP MAGNITUDE. The static and kinetic cases differ in
    // the FORCE, and one expression already covers both: the fmax below is a
    // spring below minimum_slip and saturates at mu * contact above it. They do
    // not differ in the Hessian, for the reason the block below gives.
    //
    // THE TEST IS ON THE CONE, NOT ON `friction`. A cone is mu times the normal
    // force, and a normal force that has gone negative would put a negative
    // block into a matrix the whole solve assumes SPD.
    const float cone = friction * contact;
    if (cone > 0.0f) {
        lambda = fmath::div(cone, fmath::max(minimum_slip, tangent_norm));
        // THE ANCHOR. `friction_slip_prediction` below derives `drive` and
        // `drive_stiffness`; inside the cone the excess is zero, the anchor is
        // the current slip, and `stiffness` equals `lambda`, which is the
        // lagged surrogate exactly.
        //
        // PAST THE CONE THE ANCHOR ADDS THE LARGER OF TWO SLIPS: the one the
        // pair-local model predicts, and the one the contact actually slid in
        // the last committed step. The prediction holds every other vertex
        // still, so for a body that slides as a whole it reads the body's
        // elastic stiffness where a translation meets none, and predicts a
        // slip too small to release it: alone, a stiff sheet on a slope past
        // its friction angle creeps by about minimum_slip * T / (mu N) a step.
        // The last step's slide is the lagged anchor carried across the step
        // boundary, so a contact the prediction holds back grows its slide by
        // about T / (mu N) a step until it slides as loaded, where the
        // prediction alone would hold it there for good. A contact inside the
        // cone has no excess and is untouched either way.
        //
        // THE DRIVE IS THE PAIR'S SHARE OF ITS VERTICES' LOAD, NOT THE
        // VERTICES' WHOLE RESIDUAL. An anchor past |u| makes the surrogate lie
        // above mu N |u| at the current slip, so it is not a majorizer touching
        // there, and nothing then forces each iteration downhill. A contact
        // whose drive reads past its cone while the contacts sharing its vertex
        // together hold that vertex jumps out of the stick range, is pushed back
        // by the stick spring the next iteration, and jumps out again: measured
        // on a yarn strip hung on its pins, every Newton direction from the
        // third iteration on was the exact negative of the one before
        // (cosine -1.00, 2.6e-5 m), so 16 and 48 iterations ended at the same
        // phase of the cycle and neither solved the step. Two things make the
        // drive read past the cone falsely, and each alone keeps the cycle:
        // several pairs each reading the vertex's whole load, and the load
        // missing the normal forces of the other contacts at the vertex. The
        // caller removes both (`friction_slip_prediction`, `share` and `load`,
        // and `friction_memory.kernel.cpp`), and with both removed the same
        // strip's Newton directions fall to round-off within a few iterations.
        float anchor = tangent_norm;
        if (drive_stiffness > 0.0f) {
            const Vec3f drive_tangent = drive - normal * normal.dot(drive);
            const float excess =
                fmath::sqrt(drive_tangent.squaredNorm()) - cone;
            if (excess > 0.0f) {
                anchor = anchor + fmath::max(
                                      fmath::div(excess, drive_stiffness), slide);
            }
        }
        stiffness = fmath::div(cone, fmath::max(minimum_slip, anchor));
    } else {
        lambda = 0.0f;
        stiffness = 0.0f;
    }
    gradient = lambda * tangent;
    // THE HESSIAN IS A MAJORIZER, NOT THE EXACT SECOND DERIVATIVE. Past the
    // cone the potential mu * contact * |u| is linear along the slip, so its
    // exact Hessian lambda * (P - s s^T) is singular there, and this solver has
    // no energy line search to contain a Newton step in a direction the model
    // does not bound (the line search is CCD only). What it uses instead is the
    // quadratic surrogate
    //     S_a(u) = mu * contact * (|u|^2 + a^2) / (2 a),
    // which lies above mu * contact * |u| for every u and every anchor a > 0
    // (arithmetic-geometric mean), touches it at |u| = a, and has the uniformly
    // positive tangential Hessian (mu * contact / a) * P. Minimizing a
    // majorizer is what makes the step safe without a line search: a structure
    // held up by nothing but friction stands still under it, and does not under
    // the singular form. Measured on a house of cards, peak vertex motion over
    // twenty frames of a scene that is supposed to stand still: 4.4e-3 with
    // this surrogate, 6.0e-1 with lambda * w w^T, and 2.7e-1 with a smooth
    // sqrt(|u|^2 + eps^2) potential and its exact Hessian. A reader who sees a
    // static scene shimmer should suspect this line before anything else.
    //
    // THE ANCHOR IS WHAT DECIDES HOW FAST A SATURATED CONTACT RELEASES, and it
    // is the release mechanism, not the shape of the Hessian. With a = |u|, the
    // classical lagged choice, the surrogate is tight at the current slip and a
    // Newton step from slip u under tangential load T lands at
    // u * T / (mu * contact): the slip only ever grows by that ratio per
    // iteration. Every step starts from u = 0, so a contact loaded past the
    // cone at rest starts each step at stiffness mu * contact / minimum_slip
    // and, with the two assemblies a default step performs, moves
    // minimum_slip * T / (mu * contact) per step FOREVER, whatever the load.
    // That is a body hanging on an edge it should slide off, at a creep speed
    // set by a numerical epsilon. With the anchor above it lands on the
    // physical slip in one iteration instead.
    //
    // Retaining only a FRACTION kappa of the along-slip stiffness does not
    // rescue the singular form, it moves the pole: the Newton update in s
    // becomes u <- u (1 - 1/kappa) + T / (kappa lambda), which amplifies the
    // slip by |1 - 1/kappa| per iteration and is unstable below kappa = 1/2.
    //
    // The gradient above does not depend on the anchor. It is the true force at
    // the current slip, lambda * u, so the residual the solve sees is exact and
    // only the curvature is modeled. Both branches are symmetric PSD with
    // eigenvalues {0, stiffness, stiffness}: the normal direction is free, the
    // whole tangent plane is held, and a >= |u| means the surrogate is never
    // stiffer than the lagged one.
    hessian = stiffness * projection;
}

[[seam::device_fn]] inline void friction_evaluate(
    const Vec3f &contact_force,
    const Vec3f &slip, const Vec3f &normal,
    float friction, float minimum_slip,
    const Vec3f &drive, float drive_stiffness, Vec3f &gradient,
    Mat3x3f &hessian, float &lambda, float &stiffness,
    Vec3f &tangent, Mat3x3f &projection,
    float &contact) {
    friction_evaluate_anchored(contact_force, slip, normal, friction,
                               minimum_slip, drive, drive_stiffness, 0.0f,
                               gradient, hessian, lambda, stiffness, tangent,
                               projection, contact);
}

// THE FRACTION OF A VERTEX'S RESIDUAL ONE FRICTIONAL PAIR READS, from the
// pair's own cone at the vertex `own` (|w_i| mu N) and the vertex's cone `sum`
// over `pairs` pairs in the last Newton iteration
// (`friction_memory.kernel.cpp`, "THE VERTEX'S CONE IS SHARED"). A vertex that
// carried at most one pair gives the whole residual: its sum was that one
// pair's cone, so the fraction is one, and reading it as the ratio of this
// iteration's normal force to the last one's would make a lone contact's
// release depend on how much its own load changed between two iterations. The
// ratio is capped at one for the same reason, for a pair whose load grew past
// the lagged sum.
[[seam::device_fn]] inline float friction_cone_fraction(float own, float sum,
                                                        float pairs) {
    if (pairs <= 1.0f || !(own > 0.0f)) {
        return 1.0f;
    }
    return fmath::div(own, fmath::max(sum, own));
}

// A quadratic form t^T M t written out rather than as `t.dot(M * t)`: a small
// dense matrix-vector product written as an expression is a documented
// silent-miscompile hazard in device code, and a scalar accumulation is not.
[[seam::device_fn]] inline float friction_quadratic_form(
    const Mat3x3f &matrix, const Vec3f &direction) {
    float q = 0.0f;
    for (unsigned a = 0; a < 3; ++a) {
        for (unsigned b = 0; b < 3; ++b) {
            q = q + direction[a] * matrix(a, b) * direction[b];
        }
    }
    return q;
}

// THE PAIR-LOCAL PREDICTION BEHIND THE ANCHOR. The slip coordinate of a contact
// is u = sum_i w_i x_i over the pair's vertices (w sums to zero for a pair of
// dynamic primitives, and is the barycentric weights of the dynamic side
// against a static one), and a contact force f is embedded as w_i f on vertex
// i. Restricted to the tangential drive direction t, the pair-local model is
// the quadratic form of the pair's own elastic blocks plus its inertia
// m / dt^2 on the direction W = (w_i t):
//     stiffness = W^T K W / |W|^4,   drive = (sum_i w_i r_i) / |W|^2,
// both in the parametrization x = x0 + s W / |W|^2 that makes s the slip
// itself, so the frictionless slip is drive / stiffness and the sliding slip
// (|drive| - mu N) / stiffness. For one free vertex against a static surface
// this is r_t / (t^T K_ii t + m / dt^2); for two equal free masses with no
// elastic coupling it is the relative slip of the two bodies exactly. THE
// WEIGHTING IS NOT OPTIONAL: dividing by |W|^2 is what keeps a two-body pair
// from being handed twice its drive.
//
// The elastic blocks are the pair's own, with everything outside the pair held
// fixed, which is the same pair-local model the barrier stiffness uses and is
// stiffer than the true response for every stiff body: a shell vertex on a
// floor predicts the slip of a vertex held by its neighbors, not of the whole
// shell sliding. That is the side to err on. A prediction that is too stiff
// leaves the contact at the lagged surrogate, which is what it has without
// this; a prediction that is too soft lets the contact slide past what friction
// should allow for one iteration, and does so exactly at an impact, where the
// pushes of the other contacts closing in the same iteration are not in the
// residual yet. With the elastic blocks included a house of cards stands; on
// inertia alone it drifts.
//
// A vertex without a degree of freedom, an exact fix pin or a massless vertex
// of a static solid, must carry w_i = 0: its row is eliminated, so it does not
// move however hard it is pushed. That gating is the caller's, because only it
// knows which array its indices name. Returns false when nothing in the pair
// moves or the residual has no tangential part, and the anchor then falls back
// to the current slip.
//
// `load` IS EACH VERTEX'S LOAD, three floats a vertex: its residual before
// any contact force plus the barrier forces of the contacts at it
// (`friction_cone_gather`), gathered by the caller.
//
// `share` IS THE FRACTION OF EACH VERTEX THIS PAIR OWNS, the pair's cone over
// the vertex's summed cone (`friction_cone_fraction`): one pair of several at
// a vertex is driven by its part of the load, not by all of it, and moves its
// part of the vertex's mass and stiffness, not all of them. Both halves are
// needed. Scaling only the drive would predict, for a vertex past its summed
// cone S under load T, a slip s (T - S) / k per pair instead of the vertex's
// (T - S) / k, so a contact that should slide would release at a fraction of
// its speed; scaling both gives every pair the vertex's own slip exactly. The
// stiffness is scaled as D K D with D = diag(sqrt(s_i)), which keeps it a
// quadratic form of the same sign. A vertex with one pair has a share of
// one.
//
// `local_hessian` IS THE PAIR'S ELASTIC BLOCKS ALREADY GATHERED, the same
// 3N x 3N the elasticity-inclusive contact stiffness is read out of, so this
// re-reads no CSR row. `residual` is a device array indexed by vertex and is
// read by INDEXING the parameter: a local pointer into a device array needs an
// address space MSL alone demands, so there is none here.
template <unsigned N>
[[seam::device_fn]] inline bool friction_slip_prediction(
    const SVecf<N> &weight,
    const SVecf<N> &mass,
    const SMatf<3 * N, 3 * N> &local_hessian,
    const SVecf<3 * N> &load,
    const SVecf<N> &share,
    const Vec3f &normal, float dt, Vec3f &drive,
    float &stiffness) {
    float weight2 = 0.0f;
    Vec3f residual_sum = Vec3f::Zero();
    for (unsigned i = 0; i < N; ++i) {
        if (weight[i] != 0.0f) {
            residual_sum = residual_sum +
                           (weight[i] * share[i]) *
                               Vec3f(load[3u * i], load[3u * i + 1u],
                                     load[3u * i + 2u]);
            weight2 = weight2 + weight[i] * weight[i];
        }
    }
    if (weight2 == 0.0f) {
        return false;
    }
    const Vec3f tangential = residual_sum - normal * normal.dot(residual_sum);
    const float tangential2 = tangential.squaredNorm();
    if (tangential2 == 0.0f) {
        return false;
    }
    // `normalized()`, NOT `d / sqrt(d2)`. The two agree bit for bit wherever
    // the squaring is safe, and `normalize()` carries an underflow fallback the
    // explicit form does not: `squaredNorm()` underflows to zero well before
    // `norm()` would, and dividing by that square root is a division by zero
    // where this spelling still returns a unit vector.
    const Vec3f direction = tangential.normalized();
    float k = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        if (weight[i] == 0.0f) {
            continue;
        }
        k = k + weight[i] * weight[i] * share[i] *
                    fmath::div(mass[i], dt * dt);
        for (unsigned j = 0; j < N; ++j) {
            if (weight[j] == 0.0f) {
                continue;
            }
            const float owned = share[i] == share[j]
                                    ? share[i]
                                    : fmath::sqrt(share[i] * share[j]);
            Mat3x3f block;
            for (unsigned a = 0; a < 3; ++a) {
                for (unsigned b = 0; b < 3; ++b) {
                    block(a, b) = local_hessian(3u * i + a, 3u * j + b);
                }
            }
            k = k + weight[i] * weight[j] * owned *
                        friction_quadratic_form(block, direction);
        }
    }
    // COMPONENTWISE, not a reciprocal times the vector: three divisions by
    // `w2` and one reciprocal multiplied through are different roundings in
    // fp32.
    drive = Vec3f(fmath::div(tangential[0], weight2),
                  fmath::div(tangential[1], weight2),
                  fmath::div(tangential[2], weight2));
    stiffness = fmath::div(k, weight2 * weight2);
    return stiffness > 0.0f;
}

