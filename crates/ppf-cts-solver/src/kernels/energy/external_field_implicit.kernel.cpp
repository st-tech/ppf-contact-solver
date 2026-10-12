// File: external_field_implicit.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the forms the compilers read.
//
// THE FIELD TERMS THAT CANNOT BE EVALUATED ONCE AT THE STEP'S START. A strong
// drag evaluated at the starting velocity overshoots past zero, and a stiff
// spring evaluated at the starting position diverges, so these laws
// are evaluated at the Newton ITERATE `y`, with
// the step's velocity `v = (y - x) / dt`, and enter the system as a force and
// a Hessian beside the momentum row:
//
//   * Drag: `a = -W (linear |v| + quadratic) v`.
//   * Harmonic: the spring `-s W (|p| - rest) p_hat` (on a Plane,
//     `-s W pz z_hat`) and the damping `-[W > 0] 2 zeta sqrt(|s|) v`.
//   * Vortex damping: `-inflow W v` on a Point and `-[W > 0] (1 + inflow W) v`
//     on a Plane, which damps the velocity with Inflow 0 too. The swirl itself
//     is explicit (`external_field`).
//   * Fluid Flow: `s W rho (U / playback - flow v)`, the fluid's velocity
//     `U` and density `rho` read from its volume at the iterate, at the
//     instant the host's blend names, and `U` brought
//     from the scene clock to physical time by the step's Playback Speed
//     (below); unless the field weighs by density, `rho` is the domain's
//     presence, 1 inside and 0 outside, so outside the domain the damping
//     goes too (measured). The push `s W rho U / playback` is the gradient
//     alone, its spatial derivative dropped as the falloff's is;
//     `s W rho flow` is a damping like the others.
//
// THE GRADIENT IS EXACT AND THE HESSIAN IS PSD BY CONSTRUCTION, term by term
// at its source. An applied acceleration `a` enters the
// assembly as `f -= m a`, with `m` the vertex's mass times its field weight,
// and each Hessian term is added ONLY WHERE THE PRODUCT OF `m` AND THE TERM'S
// COEFFICIENT IS POSITIVE: a negative weight, Strength, Inflow or Flow
// reverses the term (it pumps energy rather than damping it), and such a term
// contributes its gradient and no Hessian. The drag's terms are
// `(m / dt) W (L |v| + Q) I` and `(m / dt) W L v v^T / |v|`; a damping's is
// `(m / dt) c I`; the spring's is `(m / ws) s W [p_hat p_hat^T +
// max(0, 1 - rest / |p|) (I - p_hat p_hat^T)]`, which drops the derivative of
// the falloff, an approximation of the Hessian alone. The Newton loop
// converges on the residual and the line search is the CCD filter, so neither
// needs these forces to derive from a potential.
//
// LENGTHS ARE SCENE UNITS AND A VERTEX'S VELOCITY IS PHYSICAL: the spring's
// displacement is divided by World Scaling, as every field's
// position is, and the velocity is the solver's own, per PHYSICAL second.
//
// A FLUID'S `U` IS BROUGHT TO THOSE UNITS IN TWO PLACES. Its volume is scene
// geometry and its velocity scene units per second, so the driver multiplies
// the velocity by World Scaling when it copies a frame into its slot (solver
// units). Its frames are placed on the scene CLOCK, and `U` is how far the
// fluid moves per clock second, while a step of physical length `dt`
// advances the clock by `dt / playback` (`step.rs`), so `playback` physical
// seconds pass per clock second. `U / playback` is therefore the fluid's
// speed per physical second, and this kernel divides by the step's
// `playback`, a per-call scalar because a dynamic Playback Speed changes it
// from step to step. Only then do `U` and `v` compare in the same units: a
// vertex the fluid carries at Flow 1 rides with its frames at any Playback
// Speed.

#include "external_field.kernel.cpp"

// Whether a field type has a term here.
[[seam::device_fn]] inline bool field_is_implicit(unsigned type) {
    return type == FIELD_DRAG || type == FIELD_HARMONIC || type == FIELD_VORTEX ||
           type == FIELD_FLUID;
}

// Adds the implicit field terms of one vertex to its force and its diagonal
// Hessian block, after the momentum row was written. A fix-pinned vertex owns
// no momentum row, and a weight of zero opts the vertex out, as in
// `external_field`. `dt` is the step's physical length and `playback` its
// Playback Speed, physical seconds per second of the scene clock, which the
// host holds positive and finite whenever a fluid reads it.
[[seam::entry(i)]]
[[seam::device_fn]] inline void external_field_implicit(
    const Vec3f *eval_x, const Vec3f *current, const VertexProp *prop,
    const float *weight, unsigned has_weight,
    const unsigned *target_mask, unsigned has_mask,
    const unsigned *field_header, const float *field_timing,
    const float *field_origin, const float *field_track, unsigned field_count,
    const unsigned *fluid_words, const float *fluid_floats,
    const unsigned *fluid_table, const float *fluid_data,
    float time, float dt, float playback, float inv_world_scaling, float *force,
    Mat3x3f *diagonal, DiagHandle diag, unsigned i) {
    if (prop[i].fix_index > 0u) {
        return;
    }
    const float w = has_weight != 0u ? weight[i] : 1.0f;
    if (w == 0.0f) {
        return;
    }
    const unsigned mask = has_mask != 0u ? target_mask[i] : 0u;
    // The field acts per unit mass and the weight scales it, as it scales the
    // explicit acceleration. A negative weight reverses every term, so every
    // Hessian gate below reads the product with this.
    const float mass = prop[i].mass * w;
    const Vec3f y = eval_x[i];
    const Vec3f x = current[i];
    const Vec3f v = fmath::div(1.0f, dt) * (y - x).cast<float>();
    const float speed = v.norm();
    Vec3f f = Vec3f::Zero();
    Mat3x3f H = Mat3x3f::Zero();
    for (unsigned k = 0u; k < field_count; ++k) {
        const unsigned type = field_header[FIELD_HEADER * k];
        if (!field_is_implicit(type) ||
            !field_applies(field_header[FIELD_HEADER * k + 6u], mask)) {
            continue;
        }
        FieldPoint fp;
        field_point(field_header, field_timing, field_origin, field_track, k, y,
                    time, inv_world_scaling, fp);
        DIAG_ASSERT4(diag, fp.weight_defined != 0u, static_cast<float>(i),
                     static_cast<float>(k), fp.axial, 12.0f);
        const float W = fp.W;
        // A damping coefficient times v: the gradient always, the Hessian
        // `m c / dt I` only where `m c` is positive.
        float damping = 0.0f;
        if (type == FIELD_DRAG) {
            const float c = W * (fp.linear_drag * speed + fp.quadratic_drag);
            f += (mass * c) * v;
            if (mass * c > 0.0f) {
                H += fmath::div(mass * c, dt) * Mat3x3f::Identity();
            }
            const float linear = mass * W * fp.linear_drag;
            if (linear > 0.0f && speed > 0.0f) {
                H += fmath::div(linear, dt * speed) * (v * v.transpose());
            }
        } else if (type == FIELD_VORTEX) {
            damping = fp.shape == FIELD_PLANE ? (W > 0.0f ? 1.0f + fp.inflow * W : 0.0f)
                                              : fp.inflow * W;
        } else if (type == FIELD_FLUID) {
            const unsigned h = FIELD_HEADER * k;
            Vec3f U;
            float rho;
            float presence;
            field_fluid(fluid_words, fluid_floats, fluid_table, fluid_data,
                        field_header[h + FIELD_HEADER_FLUID_INDEX], y, inv_world_scaling,
                        U, rho, presence);
            if (field_header[h + FIELD_HEADER_FLUID_DENSITY] == 0u) {
                rho = presence;
            }
            const float swr = fp.strength * W * rho;
            // Per clock second to per physical second (the header above).
            f -= (mass * swr * fmath::div(1.0f, playback)) * U;
            damping = swr * fp.inflow;
        } else if (type == FIELD_HARMONIC) {
            const float sw = fp.strength * W;
            // The stiffness enters the Hessian only where `m s W` is
            // positive; a reversed spring, pushing away from its rest
            // length, contributes its gradient alone.
            const bool stiff = mass * sw > 0.0f;
            Vec3f g_local = Vec3f::Zero();
            Mat3x3f K = Mat3x3f::Zero();
            if (fp.shape == FIELD_PLANE) {
                g_local[2] = sw * fp.pz;
                if (stiff) {
                    K(2, 2) = sw;
                }
            } else if (fp.distance > 0.0f) {
                const Vec3f p_hat(fmath::div(fp.px, fp.distance),
                                  fmath::div(fp.py, fp.distance),
                                  fmath::div(fp.pz, fp.distance));
                g_local = (sw * (fp.distance - fp.rest_length)) * p_hat;
                if (stiff) {
                    const Mat3x3f radial = p_hat * p_hat.transpose();
                    const float tangential = fmath::max(
                        1.0f - fmath::div(fp.rest_length, fp.distance), 0.0f);
                    K = sw * (radial + tangential * (Mat3x3f::Identity() - radial));
                }
            }
            // Back to world axes: g = R g_local, K_world = R K R^T, R's
            // columns the field's local axes.
            Mat3x3f R;
            for (unsigned c = 0u; c < 3u; ++c) {
                R(c, 0) = fp.ax[c];
                R(c, 1) = fp.ay[c];
                R(c, 2) = fp.az[c];
            }
            f += mass * (R * g_local);
            H += (mass * inv_world_scaling) * (R * K * R.transpose());
            if (W > 0.0f) {
                damping = 2.0f * fp.harmonic_damping *
                          fmath::sqrt(fmath::abs(fp.strength));
            }
        }
        if (damping != 0.0f) {
            f += (mass * damping) * v;
            if (mass * damping > 0.0f) {
                H += fmath::div(mass * damping, dt) * Mat3x3f::Identity();
            }
        }
    }
    const bool finite = !fmath::isnan(f[0]) && !fmath::isinf(f[0]) &&
                        !fmath::isnan(f[1]) && !fmath::isinf(f[1]) &&
                        !fmath::isnan(f[2]) && !fmath::isinf(f[2]);
    DIAG_ASSERT4(diag, finite, static_cast<float>(i), f[0], f[1], f[2]);
    if (!finite) {
        return;
    }
    for (unsigned c = 0u; c < 3u; ++c) {
        force[3u * i + c] += f[c];
    }
    // Through a thread-space copy: `diagonal[i]` is a device-space lvalue,
    // and the matrix's compound assignment takes a thread-space reference,
    // which MSL will not bind to it (the momentum row does the same).
    Mat3x3f diagonal_i = diagonal[i];
    diagonal_i += H;
    diagonal[i] = diagonal_i;
}
