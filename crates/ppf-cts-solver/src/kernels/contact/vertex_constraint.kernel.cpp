// File: vertex_constraint.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: the per-vertex analytic-collider passes, the
// assembly and the swept line-search test beside it.
//
// THE ANALYTIC COLLIDERS ARE THE SPHERE AND THE FLOOR, and neither is a mesh:
// a vertex is tested against every one of them, so the two inner loops are over
// the collider arrays rather than over a candidate pair list. That is why these
// passes carry `sphere_count` and `floor_count` as record scalars and read the
// arrays as BASE pointers.
//
// THE DIAGNOSTIC IS PART OF THE PHYSICS HERE, as in `contact_narrow`: a
// non-negative gap is the penetration-free guarantee rather than
// instrumentation, so both entries declare a `[[seam::diag]]` lane.
//
// `distance.hpp` FIRST, for the reason `contact_narrow.kernel.cpp` states: a
// neutral body includes no header of its own, so the type vocabulary reaches it
// from whichever SHARED HEADER an includer pulls in first.
#include "distance.hpp"

#include "../energy/model/fix.hpp"

#include "park.hpp"
#include "../csrmat/fixed_csr.kernel.cpp"
#include "analytic_contact.kernel.cpp"
#include "contact_statistics.kernel.cpp"

// ONE VERTEX AGAINST EVERY ANALYTIC COLLIDER, plus its pin barrier.
//
// THE PIN BRANCH IS ALMOST ALWAYS DEAD, and deliberately so. A fix pin is an
// exact Dirichlet BC whose row the driver eliminates, so a barrier here would be
// work the elimination overwrites. The exception is an anchor INSIDE a PDRD
// rigid body, which owns no per-vertex DOF and so cannot be given a Dirichlet
// row; that anchor is the only penalty pin left in the codebase, and
// `disable_pin_dof_removal` is the A/B lever that puts every fix pin back on
// this path.
//
// THE DEPOSIT IS DIRECT, which is `energy.cu`'s own
// `Map<Vec3f>(force.data + 3 * i) += f; diag_hess[i] += H;`. One thread owns
// one vertex and writes only that vertex's three force components and its nine
// diagonal ones, so the accumulation needs no atomic and the row's `Scatter` is
// `Disjoint`.
//
// NO HOST COMPACTION STANDS BEHIND IT. Writing a force, a Hessian and an
// `active` flag into three scratch arrays, downloading all three, walking them
// in ascending vertex order, and uploading the surviving run would let a
// scatter kernel add the force and a push offer the block to the FIXED pattern.
// The block belongs on the DIAGONAL rather than in that pattern, which is where
// the reference puts it and what `Operator`'s `C` term already carries, so that
// walk would buy nothing but three device-to-host copies of the whole vertex
// set.
//
// `out_count` STAYS, and it is telemetry rather than a routing: a fix-pin
// anchor contributes force and Hessian without counting as a contact, so the
// two numbers were never the same question.
[[seam::entry(i)]]
[[seam::device_fn]] inline void vertex_constraint(
    const Vec3f *eval_x,
    const Vec3f *current,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const FixPair *fix_pair,
    const Sphere *sphere, unsigned sphere_count,
    const Floor *floor, unsigned floor_count,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned disable_pin_dof_removal, float constraint_tol,
    unsigned friction_mode, float friction_eps,
    // THE RESIDUAL EVERY FRICTION TERM BELOW ANCHORS ITSELF ON. It is a COPY
    // of `force` taken before this pass, not `force` itself: this entry writes
    // only its own vertex's row, but the three collision-mesh passes that read
    // the same array deposit through atomics across vertices, so one snapshot
    // serves them all and none of them reads a row another has already moved.
    // It holds the momentum, elastic and strain-limit terms.
    const float *residual, float dt,
    const float *grain_inv_inertia,
    float *force, float *diagonal,
    unsigned *out_count,
    float *out_grain_angular,
    float *out_grain_coupling,
    float *out_grain_rotational, unsigned i,
    // THE PER-OBJECT STATISTICS CHANNEL. An analytic primitive belongs to no
    // object, so only the dynamic side is charged and no static map is needed.
    compute::atomic_uint_t *statistics_contact_count,
    unsigned statistics_contact_count_size,
    const unsigned *statistics_object_index,
    unsigned statistics_object_index_size,
    DiagHandle diag) {

    const VertexProp prop = vertex_prop[i];
    const VertexParam vparam = vertex_param[prop.param_index];
    const float mass = prop.mass;
    // A SAND GRAIN, and the radius its friction torque acts at. Both are read
    // the way `contact.cu` reads them: a non-zero inverse inertia is what makes
    // a vertex a grain, and the radius is the vertex's own contact offset.
    const bool is_grain = grain_inv_inertia[i] > 0.0f;
    const float grain_radius = vparam.offset;
    Mat3x3f grain_angular = Mat3x3f::Zero();
    Mat3x3f grain_coupling = Mat3x3f::Zero();
    Vec3f grain_rotational = Vec3f::Zero();
    uint32_t num_contact = 0u;
    bool contributed = false;

    const Mat3x3f local_hess =
        fixed_csr_read(fixed_index, fixed_offset, fixed_value,
                           row_count, i, i);
    // READ ONCE, at the top. Every analytic contact this vertex makes anchors
    // its friction on the same row, and nothing below writes it: the deposit at
    // the end goes to `force`, which is a different array.
    const Vec3f residual_i(residual[3u * i], residual[3u * i + 1u],
                           residual[3u * i + 2u]);
    Mat3x3f H = Mat3x3f::Zero();
    Vec3f f = Vec3f::Zero();
    const Vec3f x = eval_x[i];
    // Both operands copied into thread space: `x` above already is, and the
    // subtraction needs the two in one address space.
    const Vec3f current_i = current[i];
    const Vec3f dx = (x - current_i).cast<float>();

    if (prop.fix_index > 0u) {
        // A fix pin is an exact Dirichlet BC: the driver eliminates its row,
        // so a barrier here would be dead work the elimination overwrites.
        // The exception is an anchor INSIDE a PDRD rigid body, which owns no
        // per-vertex DOF and so cannot be given a Dirichlet row; that anchor
        // is the only penalty pin left in the codebase. The A/B lever
        // `disable-pin-dof-removal` puts every fix pin back on this path.
        if (prop.pdrd_body_index > 0u || disable_pin_dof_removal != 0u) {
            const FixPair pin = fix_pair[prop.fix_index - 1u];
            const Vec3f y = pin.position;
            const Vec3f w = (x - y).cast<float>();
            const float distance = w.norm();
            float gap = pin.ghat - distance;
            if (pin.kinematic) {
                gap = fmath::max(gap, constraint_tol * pin.ghat);
            } else {
                DIAG_ASSERT4(diag, gap >= 0.0f, gap, pin.ghat, (float)i,
                            (float)prop.fix_index);
            }
            const float reference =
                w.squaredNorm() ? (local_hess * w).dot(w) / w.squaredNorm()
                                : 0.0f;
            const float stiff_k = reference + mass / (gap * gap);
            f += stiff_k * fix::gradient(x, y);
            H += stiff_k * fix::hessian();
            contributed = true;
        }
    } else if (mass > 0.0f) {
        // A zero-mass vertex is a static solid, and a sphere and a floor are
        // static themselves, so there is no contact between the two.
        for (uint32_t j = 0; j < sphere_count; ++j) {
            const Sphere collider = sphere[j];
            const float ghat = collider.ghat;
            const float friction = combine_friction_values(
                collider.friction, vparam.friction,
                friction_mode);
            const bool bowl = collider.bowl;
            bool reverse = collider.reverse;
            float radius = collider.radius;
            Vec3f center = collider.center;
            // A BOWL slides its center up to the query's own height, which
            // turns the lower hemisphere into an infinite trough.
            center = (bowl && (x[1] > center[1]))
                         ? Vec3f(center[0], x[1], center[2])
                         : center;
            const float d2 = (x - center).cast<float>().squaredNorm();
            if (collider.kinematic) {
                if (d2) {
                    const Vec3f normal = (x - center).cast<float>().normalized();
                    const float eff_radius =
                        reverse ? (radius - ghat) : (radius + ghat);
                    const Vec3f target = eff_radius * normal;
                    const Vec3f o = (x - center).cast<float>();
                    if (bowl) {
                        reverse = true;
                    }
                    const float r2 = eff_radius * eff_radius;
                    // Pass through once the penetration depth exceeds the
                    // authored thickness, which is what makes a thin shell
                    // a shell rather than a solid.
                    const float dist = fmath::sqrt(d2);
                    const float depth = reverse ? (dist - collider.radius)
                                                : (collider.radius - dist);
                    if (collider.thickness > 0.0f && depth > collider.thickness) {
                        continue;
                    }
                    if (reverse == true && d2 > r2) {
                        num_contact += 1u;
                                        statistics_record_analytic_contact(
                            statistics_contact_count, statistics_contact_count_size,
                            statistics_object_index, statistics_object_index_size, i, diag);
                        const AnalyticContactResult contact =
                            analytic_contact_evaluate(
                                local_hess, dx, -normal,
                                (o - target).dot(-normal), ghat, ghat, 0.0f,
                                friction_eps, mass, residual_i, dt, true,
                                false);
                        f += contact.force;
                        H += contact.hessian;
                        contributed = true;
                    } else if (reverse == false && d2 < r2) {
                        num_contact += 1u;
                        statistics_record_analytic_contact(
                            statistics_contact_count, statistics_contact_count_size,
                            statistics_object_index, statistics_object_index_size, i, diag);
                        const AnalyticContactResult contact =
                            analytic_contact_evaluate(
                                local_hess, dx, normal,
                                (o - target).dot(normal), ghat, ghat, 0.0f,
                                friction_eps, mass, residual_i, dt, true,
                                false);
                        f += contact.force;
                        H += contact.hessian;
                        contributed = true;
                    }
                }
            } else {
                if (reverse) {
                    radius -= ghat;
                } else {
                    radius += ghat;
                }
                const float r2 = radius * radius;
                const bool intersected = reverse ? d2 > r2 : d2 < r2;
                if (intersected) {
                    const float dist = fmath::sqrt(d2);
                    const float depth = reverse ? (dist - collider.radius)
                                                : (collider.radius - dist);
                    if (collider.thickness > 0.0f && depth > collider.thickness) {
                        continue;
                    }
                    num_contact += 1u;
                    statistics_record_analytic_contact(
                        statistics_contact_count, statistics_contact_count_size,
                        statistics_object_index, statistics_object_index_size, i, diag);
                    Vec3f normal = (x - center).cast<float>().normalized();
                    const Vec3f projected_x = radius * normal;
                    const Vec3f o = (x - center).cast<float>();
                    if (reverse) {
                        normal = -normal;
                    }
                    float gap;
                    if (reverse) {
                        gap = collider.radius - fmath::sqrt(d2);
                    } else {
                        gap = fmath::sqrt(d2) - collider.radius;
                    }
                    DIAG_ASSERT4(diag, gap >= 0.0f, gap, collider.radius,
                                (float)i, (float)j);
                    const AnalyticContactResult contact =
                        analytic_contact_evaluate(
                            local_hess, dx, normal,
                            (o - projected_x).dot(normal), gap, ghat,
                            friction, friction_eps, mass, residual_i, dt,
                            false, true);
                    f += contact.force;
                    H += contact.hessian;
                    // A SAND GRAIN'S SPIN IS CONDENSED OUT OF THIS CONTACT.
                    // `contact.cu` accumulates the same three blocks at the same
                    // point, from the FRICTION Hessian and gradient alone: the
                    // normal barrier passes through a grain's center and exerts
                    // no torque about it, so only friction turns it.
                    if (is_grain) {
                        analytic_grain_schur(
                            contact.friction_hessian, contact.friction_gradient,
                            contact.normal, grain_radius, grain_angular,
                            grain_coupling, grain_rotational);
                    }
                    contributed = true;
                }
            }
        }

        for (uint32_t j = 0; j < floor_count; ++j) {
            const Floor plane = floor[j];
            const float ghat = plane.ghat;
            const float friction = combine_friction_values(
                plane.friction, vparam.friction,
                friction_mode);
            const Vec3f up = plane.up;
            const Vec3f ground =
                plane.ground + float(ghat) * up;
            const Vec3f e = (x - ground).cast<float>();
            if (e.dot(up) < 0.0f) {
                const float depth = -e.dot(up);
                if (plane.thickness > 0.0f && depth > plane.thickness) {
                    continue;
                }
                num_contact += 1u;
                statistics_record_analytic_contact(
                    statistics_contact_count, statistics_contact_count_size,
                    statistics_object_index, statistics_object_index_size, i, diag);
                const Vec3f projected_x = -e.dot(up) * up;
                float gap = (x - plane.ground).cast<float>().dot(up);
                if (plane.kinematic) {
                    gap = fmath::max(gap, constraint_tol * ghat);
                }
                DIAG_ASSERT4(diag, gap >= 0.0f, gap, ghat, (float)i,
                            (float)j);
                const AnalyticContactResult contact =
                    analytic_contact_evaluate(
                        local_hess, dx, up, (-projected_x).dot(up), gap,
                        ghat, friction, friction_eps, mass, residual_i, dt,
                        // FALSE, NOT THE FLOOR'S OWN `kinematic` FLAG. In
                        // `contact.cu` the floor's stiffness is
                        // `up.dot(local_hess * up) + mass / gap^2`
                        // UNCONDITIONALLY, and `floor.kinematic` does exactly one
                        // thing there: it clamps the gap above, which is the
                        // clamp a few lines up. The `mass / ghat^2` arm inside
                        // `analytic_contact_evaluate` is the SPHERE's, and
                        // handing the floor to it drops the elasticity-inclusive
                        // term and softens the barrier everywhere inside the
                        // band, where `gap < ghat` makes `mass / gap^2` the
                        // larger of the two.
                        false, true);
                f += contact.force;
                H += contact.hessian;
                if (is_grain) {
                    analytic_grain_schur(
                        contact.friction_hessian, contact.friction_gradient,
                        contact.normal, grain_radius, grain_angular,
                        grain_coupling, grain_rotational);
                }
                contributed = true;
            }
        }
    }

    if (contributed) {
        for (unsigned k = 0; k < 3; ++k) {
            force[3 * i + k] += f[k];
        }
        for (unsigned k = 0; k < 9; ++k) {
            diagonal[9 * i + k] += H.m[k];
        }
    }
    out_count[i] = num_contact;
    // THE THREE GRAIN BLOCKS ARE WRITTEN FOR EVERY VERTEX, grain or not, so the
    // pass below reads this iteration's values rather than the last one's. A
    // non-grain leaves them at the zero they were initialized to, which is what
    // `sand_grain_condense` tests when it checks the angular block's trace.
    for (unsigned k = 0; k < 9; ++k) {
        out_grain_angular[9 * i + k] = grain_angular.m[k];
        out_grain_coupling[9 * i + k] = grain_coupling.m[k];
    }
    for (unsigned k = 0; k < 3; ++k) {
        out_grain_rotational[3 * i + k] = grain_rotational[k];
    }
}

// A prescribed vertex's swept path against the analytic colliders, and a
// barrier-held pin's confinement to its own ghat ball.
//
// TWO SEPARATE RESULTS COME OUT OF IT. `out_toi` is the usual time of impact,
// folded by the caller. `out_infeasible` flags a FIX-PINNED vertex whose
// prescribed path crosses a collider it cannot yield to: clamping the time of
// impact would not help, because a prescribed vertex has no freedom to give, so
// the clamp would only stall the whole solve. The step reports it and fails
// instead.
//
// The test is SWEPT rather than endpoint-wise: `max_u` excludes fix pins, so the
// step size is chosen blind to a pin's speed and a fast pin can cross a
// primitive entirely within one step. It fires only on a genuine
// outside-to-inside crossing, so a collider the user deliberately embedded a pin
// inside is not flagged.
//
// `out_toi` is READ and written, seeded by the caller at `line_search_max_t`
// and folded down with `fminf`, which is why it is a base pointer and not a
// scatter.
[[seam::entry(i)]]
[[seam::device_fn]] inline void vertex_constraint_sweep(
    const Vec3f *x0, const Vec3f *x1,
    const VertexProp *vertex_prop,
    const FixPair *fix_pair,
    const Sphere *sphere, unsigned sphere_count,
    const Floor *floor, unsigned floor_count,
    unsigned disable_pin_dof_removal, float ccd_eps, float line_search_max_t,
    float *out_toi,
    unsigned *out_infeasible, unsigned i,
    DiagHandle diag) {

    out_infeasible[i] = 0u;
    const VertexProp prop = vertex_prop[i];
    const Vec3f p0 = x0[i];
    // THE SWEPT SEGMENT IS THE EXTENDED ONE, `x0 + line_search_max_t * (x1 -
    // x0)`, not the plain Newton segment. Every fraction `t` solved below is a
    // fraction of THAT horizon, which is what makes the `line_search_max_t * t`
    // conversions here and the caller's final divide by the same factor
    // consistent. `contact.cu`'s `vertex_constraint_line_search` builds it the
    // same way and so does the Metal copy.
    //
    // READING `x1[i]` VERBATIM UNDER-SWEEPS BY THE WHOLE FACTOR, which at the
    // shipped default of 1.25 is 25 percent of the path: a sphere or floor
    // crossing that happens beyond the Newton endpoint but inside the horizon
    // the line search may actually take is then never tested, and the step is
    // granted. The analytic colliders have no other CCD behind them, so that is
    // a tunneling path rather than a tolerance.
    //
    // THE AFFINE FORM IS DELIBERATE: the endpoint is written as a scaled
    // DIFFERENCE added back to the start, so the factor never multiplies an
    // absolute coordinate and the coordinate enters once, additively.
    // The Newton endpoint is copied into thread space before it is
    // differenced: under MSL `x1[i]` is a `const device` lvalue and `p0` is a
    // thread one, and the subtraction has no overload across the two.
    const Vec3f newton_end = x1[i];
    const Vec3f p1 = float(line_search_max_t) * (newton_end - p0) + p0;
    if (prop.fix_index > 0u) {
        const FixPair pin = fix_pair[prop.fix_index - 1u];
        for (uint32_t j = 0; j < sphere_count; ++j) {
            const Sphere collider = sphere[j];
            if (collider.kinematic) {
                continue;
            }
            const Vec3f center = collider.center;
            const Vec3f center0 = (collider.bowl && (p0[1] > center[1]))
                                       ? Vec3f(center[0], p0[1], center[2])
                                       : center;
            const Vec3f center1 = (collider.bowl && (p1[1] > center[1]))
                                       ? Vec3f(center[0], p1[1], center[2])
                                       : center;
            const float r = collider.radius;
            const float r0 = (p0 - center0).cast<float>().norm();
            const float r1 = (p1 - center1).cast<float>().norm();
            // Signed feasible depth, matching the assembly above: outside
            // for a solid sphere, inside for a reversed one (a container).
            const float d0 = collider.reverse ? (r - r0) : (r0 - r);
            const float d1 = collider.reverse ? (r - r1) : (r1 - r);
            if (d0 >= 0.0f && d1 < 0.0f) {
                out_infeasible[i] = 1u;
            }
        }
        for (uint32_t j = 0; j < floor_count; ++j) {
            const Floor plane = floor[j];
            if (plane.kinematic) {
                continue;
            }
            const float h0 = plane.up.dot((p0 - plane.ground).cast<float>());
            const float h1 = plane.up.dot((p1 - plane.ground).cast<float>());
            if (h0 >= 0.0f && h1 < 0.0f) {
                out_infeasible[i] = 1u;
            }
        }
        // Confine a barrier-held pin to its ghat ball. Only a PDRD anchor is
        // still held by the barrier (or every fix pin, under the A/B lever);
        // an exact Dirichlet pin sits at its target by construction, so this
        // clamp would have nothing to enforce and could only throttle the
        // shared time of impact.
        if ((prop.pdrd_body_index > 0u || disable_pin_dof_removal != 0u) &&
            pin.kinematic == false) {
            const Vec3f position = pin.position;
            const float r0 = (p0 - position).cast<float>().norm();
            const float r1 = (p1 - position).cast<float>().norm();
            DIAG_ASSERT4(diag, r0 < pin.ghat, r0, pin.ghat, (float)i,
                        (float)prop.fix_index);
            // Clearance inside the ball, and the park short of its
            // surface. The assembly above divides by this same clearance
            // (`mass / (gap * gap)` on the PDRD anchor), so landing the
            // vertex exactly on the ball is the zero divisor
            // `accd::park_gap_analytic` exists to make unreachable.
            const float clearance0 = pin.ghat - r0;
            const float clearance1 = pin.ghat - r1;
            const float park = accd::park_gap_analytic(
                clearance0, pin.ghat, ccd_eps);
            if (clearance1 < park) {
                const float denominator = clearance0 - clearance1;
                if (denominator) {
                    const float t = accd::park_crossing_analytic(
                        clearance0, clearance1, park);
                    out_toi[i] =
                        fmath::min(out_toi[i], line_search_max_t * t);
                    DIAG_ASSERT4(diag, out_toi[i] > 0.0f, out_toi[i],
                                line_search_max_t, (float)i, 0.0f);
                }
            }
        }
    } else if (prop.mass > 0.0f) {
        for (uint32_t j = 0; j < sphere_count; ++j) {
            const Sphere collider = sphere[j];
            if (collider.kinematic) {
                continue;
            }
            const bool reverse = collider.reverse;
            const bool bowl = collider.bowl;
            const Vec3f center = collider.center;
            const Vec3f center0 = (bowl && (p0[1] > center[1]))
                                       ? Vec3f(center[0], p0[1], center[2])
                                       : center;
            const Vec3f center1 = (bowl && (p1[1] > center[1]))
                                       ? Vec3f(center[0], p1[1], center[2])
                                       : center;
            const float r = collider.radius;
            const float r0 = (p0 - center0).cast<float>().norm();
            const float r1 = (p1 - center1).cast<float>().norm();
            // A start already embedded past the thickness cutoff passes
            // through, which also keeps an ill-placed initial vertex from
            // failing the crossing test rather than the assembly's.
            const float depth0 = reverse ? (r0 - r) : (r - r0);
            if (collider.thickness > 0.0f && depth0 > collider.thickness) {
                continue;
            }
            // The signed clearance on the feasible side, the convention
            // the fix-pin branch above already uses: outside for a solid
            // sphere, inside for a reversed one. The start clearance is the
            // exact negation of the penetration depth the thickness cutoff
            // just took, so it is not spelled a second time.
            const float clearance0 = -depth0;
            const float clearance1 = reverse ? (r - r1) : (r1 - r);
            const float park = accd::park_gap_analytic(
                clearance0, collider.ghat, ccd_eps);
            // A sweep starting on the feasible side stops at the park.
            // `park_gap_analytic` is strictly positive for a positive start
            // clearance, so testing against the park also catches every
            // crossing of the surface itself. One starting already inside
            // keeps the symmetric test, whose only
            // remaining case is a vertex on its way back out; it parks at
            // the surface, so the arithmetic there is unchanged.
            const bool crossed = (clearance0 > 0.0f)
                                     ? (clearance1 < park)
                                     : ((r0 - r) * (r1 - r) <= 0.0f);
            if (crossed) {
                const float t = accd::park_crossing_analytic(
                    clearance0, clearance1, park);
                out_toi[i] =
                    fmath::min(out_toi[i], line_search_max_t * t);
                DIAG_ASSERT4(diag, out_toi[i] > 0.0f, out_toi[i],
                            line_search_max_t, (float)i, 1.0f);
            }
        }
        for (uint32_t j = 0; j < floor_count; ++j) {
            const Floor plane = floor[j];
            if (plane.kinematic) {
                continue;
            }
            const Vec3f up = plane.up;
            const Vec3f ground = plane.ground;
            const float h0 = up.dot((p0 - ground).cast<float>());
            const float h1 = up.dot((p1 - ground).cast<float>());
            if (plane.thickness > 0.0f && -h0 > plane.thickness) {
                continue;
            }
            // `h` IS the gap the assembly divides by, so the sweep parks
            // short of the plane rather than solving for it.
            const float park =
                accd::park_gap_analytic(h0, plane.ghat, ccd_eps);
            if (h1 < park) {
                const float t = accd::park_crossing_analytic(h0, h1, park);
                out_toi[i] =
                    fmath::min(out_toi[i], line_search_max_t * t);
                DIAG_ASSERT4(diag, out_toi[i] > 0.0f, out_toi[i],
                            line_search_max_t, (float)i, 2.0f);
            }
        }
    }
}
