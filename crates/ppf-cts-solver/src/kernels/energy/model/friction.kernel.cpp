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

[[seam::device_fn]] inline void friction_evaluate(
    const Vec3f &contact_force,
    const Vec3f &slip, const Vec3f &normal,
    float friction, float minimum_slip,
    const Vec3f &drive, float drive_stiffness, Vec3f &gradient,
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
        float anchor = tangent_norm;
        if (drive_stiffness > 0.0f) {
            const Vec3f drive_tangent = drive - normal * normal.dot(drive);
            const float excess =
                fmath::sqrt(drive_tangent.squaredNorm()) - cone;
            if (excess > 0.0f) {
                anchor = anchor + fmath::div(excess, drive_stiffness);
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
// `local_hessian` IS THE PAIR'S ELASTIC BLOCKS ALREADY GATHERED, the same
// 3N x 3N the elasticity-inclusive contact stiffness is read out of, so this
// re-reads no CSR row. `residual` is a device array indexed by vertex and is
// read by INDEXING the parameter: a local pointer into a device array needs an
// address space MSL alone demands, so there is none here.
template <unsigned N>
[[seam::device_fn]] inline bool friction_slip_prediction(
    const unsigned *index,
    const SVecf<N> &weight,
    const SVecf<N> &mass,
    const SMatf<3 * N, 3 * N> &local_hessian,
    const float *residual,
    const Vec3f &normal, float dt, Vec3f &drive,
    float &stiffness) {
    float weight2 = 0.0f;
    Vec3f residual_sum = Vec3f::Zero();
    for (unsigned i = 0; i < N; ++i) {
        if (weight[i] != 0.0f) {
            const unsigned row = 3u * index[i];
            residual_sum = residual_sum +
                           weight[i] * Vec3f(residual[row], residual[row + 1u],
                                             residual[row + 2u]);
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
        k = k + weight[i] * weight[i] * fmath::div(mass[i], dt * dt);
        for (unsigned j = 0; j < N; ++j) {
            if (weight[j] == 0.0f) {
                continue;
            }
            Mat3x3f block;
            for (unsigned a = 0; a < 3; ++a) {
                for (unsigned b = 0; b < 3; ++b) {
                    block(a, b) = local_hessian(3u * i + a, 3u * j + b);
                }
            }
            k = k + weight[i] * weight[j] *
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

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `friction_evaluate_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// EVERY BUFFER IS AN ELEMENT GATHER, INCLUDING THE SIX OUTPUTS. The body
// returns void and writes its results through references, so `buffer[index]`
// is the destination rather than the source: a gather hands the body one
// element, which is an lvalue, and a non-const gather is therefore the write.
// Every output therefore lands at the same element index the inputs were
// gathered from, one contact per thread, with no second indexing rule to keep
// in step.
//
// The seven are separate arrays rather than one packed record because they are
// separate outputs upstream: `lambda`, `stiffness`, `tangent`, `projection` and
// `contact` are what the SAND rolling Schur block and the analytic contact path
// read, and folding them into the gradient here would lose them. Declaring them
// here is also what avoids a hand-written mirror: a host shim written by hand
// would carry its own struct of ten raw pointers with nothing linking it to the
// other backends, where one declaration renders all four targets.
//
// `lambda` AND `stiffness` ARE BOTH DECLARED BECAUSE THEY ARE DIFFERENT
// QUANTITIES: `lambda` is the secant stiffness of the FORCE, so the gradient is
// `lambda * u` and the SAND torque is built from it, while `stiffness` is the
// curvature of the SURROGATE and is what the Hessian carries. They are equal
// inside the cone and `stiffness <= lambda` past it.
//
// `minimum_slip` is one value for the whole dispatch, the slip magnitude below
// which the contact is treated as static, so it arrives in the record. `drive`
// and `drive_stiffness` are per contact and come from
// `friction_slip_prediction`; a dispatch with no prediction to offer passes a
// zero stiffness, which is what makes the anchor fall back to the current slip.
[[seam::entry(count)]] void friction_evaluate(
    const Vec3f *contact_force,
    const Vec3f *slip,
    const Vec3f *normal,
    const float *friction, float minimum_slip,
    const Vec3f *drive,
    const float *drive_stiffness,
    Vec3f *gradient,
    Mat3x3f *hessian,
    float *lambda,
    float *stiffness,
    Vec3f *tangent,
    Mat3x3f *projection,
    float *contact,
    unsigned count);
