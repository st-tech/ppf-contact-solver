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
#include "friction_memory.kernel.cpp"
#include "contact_statistics.kernel.cpp"

// SURFACE TRIANGLES AND EDGES AGAINST A STATIC SPHERE.
//
// A VERTEX BARRIER DOES NOT KEEP A TRIANGLE OUT OF A SPHERE. A flat triangle
// whose three vertices sit at their gap sags toward the sphere between them by
// about `edge^2 / 8R`, which for a 3.3 cm edge on a 5 cm sphere is 2.7 mm, more
// than twice the default 1 mm gap. A plane cannot have this defect, since a
// triangle with its vertices above a plane is above it; a curved solid collider
// can. So a triangle and an edge each get the three things a vertex has: a
// BARRIER where their closest point to the center is interior
// (`sphere_feature_constraint`), a LINE-SEARCH BOUND on their whole surface
// (`sphere_feature_sweep`), and a CONSERVATIVE CHECK on every committed pose
// (`sphere_feature_gate`).
//
// A CONTAINER SPHERE NEEDS NONE OF THIS: its feasible side is the inside of a
// ball, which is convex, so a triangle with its vertices inside is inside. A
// KINEMATIC sphere and a BOWL are NOT covered: both are held by the per-vertex
// passes alone, so a triangle can still sag into either between its vertices.

// The unit round-off of float32, `u = 2^-24`. A literal, not `<cfloat>`, for
// the reason `ACCD_FOUR_ULP` gives in `accd.hpp`.
[[seam::device_fn]] inline float sphere_unit_roundoff() {
    return 5.9604644775390625e-8f;
}

// `|x - c| - R` for a point, the gap the vertex barrier divides by
// (`vertex_constraint` forms it the same way), negated inside a container.
[[seam::device_fn]] inline float
sphere_gap(const Vec3f &x, const Vec3f &center, float radius, bool inside) {
    const float gap = (x - center).norm() - radius;
    return inside ? -gap : gap;
}

// THE FIRST FRACTION `t` OF THE SEGMENT `p0 + t d` AT WHICH THE SPHERE GAP
// FALLS TO `level`, or 2 when it never does within `t <= 1`.
//
// SOLVED ON THE SEGMENT, NOT READ OFF ITS ENDS. A straight segment between two
// points outside a sphere dips toward it by up to `|d|^2 / 8R` in between, and
// an accepted step is an interior point of its segment whenever another vertex
// set the shared fraction. A test of the two endpoints alone therefore admits
// positions inside the sphere, by about 3 um for a 2 mm slide on a 15 cm
// sphere, which is more than a small contact gap. This is the quadratic
// `|r0 + t d|^2 = (R +- level)^2`, whose constant term is formed from the gap
// rather than from `|r0|^2 - R^2`, so that it does not cancel.
[[seam::device_fn]] inline float
sphere_level_crossing(const Vec3f &r0, const Vec3f &d, float radius, bool inside,
                      float gap0, float level) {
    const float a = d.squaredNorm();
    if (a <= 0.0f) {
        return 2.0f;
    }
    const float b = 2.0f * r0.dot(d);
    // |r0|^2 - (R + level)^2 outside, |r0|^2 - (R - level)^2 inside, written
    // through the gaps: (g0 - level)(2R + g0 + level) and
    // (level - g0)(2R - g0 - level) respectively.
    const float c = inside ? (level - gap0) * (2.0f * radius - gap0 - level)
                           : (gap0 - level) * (2.0f * radius + gap0 + level);
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) {
        return 2.0f;
    }
    const float root = fmath::sqrt(disc);
    // The two roots in the cancellation-free form.
    const float q = b < 0.0f ? -0.5f * (b - root) : -0.5f * (b + root);
    float t = 2.0f;
    if (q != 0.0f) {
        const float t1 = fmath::div(q, a);
        const float t2 = fmath::div(c, q);
        if (t1 > 0.0f) {
            t = fmath::min(t, t1);
        }
        if (t2 > 0.0f) {
            t = fmath::min(t, t2);
        }
    }
    return t;
}


// The closest point of the triangle `0, e1, e2` to `p` (all relative to the
// triangle's first vertex), as barycentric weights over its three vertices, and
// which region it lies in. Ericson's region walk ("Real-Time Collision
// Detection", 5.1.5), in fp32 on edge vectors of a few centimeters, so the
// weights are good to about 1e-7; only which region, and where the point is
// put, depend on them.
[[seam::device_fn]] inline unsigned sphere_feature_region_interior() { return 0u; }
[[seam::device_fn]] inline unsigned sphere_feature_region_boundary() { return 1u; }
[[seam::device_fn]] inline unsigned sphere_feature_region_vertex() { return 2u; }

[[seam::device_fn]] inline unsigned
triangle_closest_weights(const Vec3f &e1, const Vec3f &e2, const Vec3f &p,
                         float *w) {
    const float d1 = e1.dot(p);
    const float d2 = e2.dot(p);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        w[0] = 1.0f; w[1] = 0.0f; w[2] = 0.0f;
        return sphere_feature_region_vertex();
    }
    const Vec3f bp = p - e1;
    const float d3 = e1.dot(bp);
    const float d4 = e2.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        w[0] = 0.0f; w[1] = 1.0f; w[2] = 0.0f;
        return sphere_feature_region_vertex();
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = fmath::div(d1, d1 - d3);
        w[0] = 1.0f - v; w[1] = v; w[2] = 0.0f;
        return sphere_feature_region_boundary();
    }
    const Vec3f cp = p - e2;
    const float d5 = e1.dot(cp);
    const float d6 = e2.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        w[0] = 0.0f; w[1] = 0.0f; w[2] = 1.0f;
        return sphere_feature_region_vertex();
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float v = fmath::div(d2, d2 - d6);
        w[0] = 1.0f - v; w[1] = 0.0f; w[2] = v;
        return sphere_feature_region_boundary();
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float v = fmath::div(d4 - d3, (d4 - d3) + (d5 - d6));
        w[0] = 0.0f; w[1] = 1.0f - v; w[2] = v;
        return sphere_feature_region_boundary();
    }
    const float inverse = fmath::div(1.0f, va + vb + vc);
    w[1] = vb * inverse;
    w[2] = vc * inverse;
    w[0] = 1.0f - w[1] - w[2];
    if (!(w[0] > 0.0f && w[1] > 0.0f && w[2] > 0.0f)) {
        // Rounding put the interior point on the rim: clamp it to the nearest
        // edge's interior so the weights stay a convex combination.
        for (unsigned k = 0; k < 3u; ++k) {
            w[k] = fmath::max(w[k], 0.0f);
        }
        const float total = w[0] + w[1] + w[2];
        for (unsigned k = 0; k < 3u; ++k) {
            w[k] = fmath::div(w[k], total);
        }
        return sphere_feature_region_boundary();
    }
    return sphere_feature_region_interior();
}

// The closest point of the segment `0, e1` to `p`, as weights over its two
// vertices, and its region.
[[seam::device_fn]] inline unsigned
segment_closest_weights(const Vec3f &e1, const Vec3f &p, float *w) {
    const float length2 = e1.squaredNorm();
    const float along = e1.dot(p);
    if (!(along > 0.0f) || !(length2 > 0.0f)) {
        w[0] = 1.0f; w[1] = 0.0f;
        return sphere_feature_region_vertex();
    }
    if (along >= length2) {
        w[0] = 0.0f; w[1] = 1.0f;
        return sphere_feature_region_vertex();
    }
    w[1] = fmath::div(along, length2);
    w[0] = 1.0f - w[1];
    return sphere_feature_region_interior();
}

// HOW FAR THE CLOSEST POINT `y` CAN SIT FROM THE FEATURE, from its float
// construction `P0 + sum_k w_k (P_k - P0)` (`sphere_feature_gap`). Each edge
// vector, each product with a weight and each sum rounds once, by at most `u`
// of its magnitude per axis, and the final addition of `P0` by `u |P0|`. With
// `n` rounded terms per axis, every component of `y` is within
// `u (|P0| + (n + 1) sum_k w_k |P_k - P0|)` (infinity norms, `u` the unit
// round-off) of the corresponding component of a point `z` that lies exactly on
// the feature, and `sqrt(3)` turns that into a Euclidean distance. Doubled, so
// the bound holds with room for the weights' own rounding and the norm's. A
// vertex region uses the stored vertex itself, which is on the feature exactly,
// so its error is zero.
[[seam::device_fn]] inline float
sphere_feature_point_error(const Vec3f *P, unsigned n, const float *w,
                           unsigned region) {
    if (region == sphere_feature_region_vertex()) {
        return 0.0f;
    }
    const Vec3f p0 = P[0];
    float spread = 0.0f;
    for (unsigned k = 1; k < n; ++k) {
        if (w[k] != 0.0f) {
            const Vec3f pk = P[k];
            spread += w[k] * (pk - p0).cwiseAbs().maxCoeff();
        }
    }
    const float scale =
        p0.cwiseAbs().maxCoeff() + static_cast<float>(n + 1u) * spread;
    return 2.0f * 1.7320508f * sphere_unit_roundoff() * scale;
}

// THE GAP OF A TRIANGLE (`n == 3`) OR AN EDGE (`n == 2`) TO A SOLID SPHERE, A
// LOWER BOUND ON ITS TRUE MINIMUM.
//
// The closest point is built as `P0 + sum_k w_k (P_k - P0)`, from the edge
// vectors, and its gap is `sphere_gap`, the vertex barrier's gap of a point. The
// point lies within `sphere_feature_point_error` of a point `z` of the feature,
// and a gap moves by at most the distance its point moves, so the returned gap
// is the point's gap less that error and never exceeds the true minimum, to the
// same float round-off a vertex's own gap carries. An error in the weights only
// slides `z` along the feature, which moves its distance to the center at second
// order (about `(1e-7 edge)^2 / R`). At a vertex the point IS the vertex and
// nothing is subtracted.
//
// `w` and `region` come back with the gap, and `delta` is the point relative to
// the center (for a direction and a rate, never a gap).
[[seam::device_fn]] inline float
sphere_feature_gap(const Vec3f *P, unsigned n, const Vec3f &center,
                   float radius, float *w, unsigned &region, Vec3f &delta) {
    const Vec3f p0 = P[0];
    const Vec3f to_center = center - p0;
    const Vec3f e1 = P[1] - p0;
    if (n == 3u) {
        const Vec3f e2 = P[2] - p0;
        region = triangle_closest_weights(e1, e2, to_center, w);
    } else {
        region = segment_closest_weights(e1, to_center, w);
        w[2] = 0.0f;
    }
    Vec3f y = p0;
    if (region == sphere_feature_region_vertex()) {
        for (unsigned k = 0; k < n; ++k) {
            if (w[k] == 1.0f) {
                y = P[k];
            }
        }
    } else {
        Vec3f offset = Vec3f::Zero();
        for (unsigned k = 1; k < n; ++k) {
            if (w[k] != 0.0f) {
                const Vec3f pk = P[k];
                offset += w[k] * (pk - p0);
            }
        }
        y = p0 + offset;
    }
    delta = y - center;
    return sphere_gap(y, center, radius, false) -
           sphere_feature_point_error(P, n, w, region);
}

// Whether a sphere takes part in the feature passes: a solid, static,
// full sphere.
[[seam::device_fn]] inline bool sphere_feature_applies(const Sphere &collider) {
    return !collider.kinematic && !collider.reverse && !collider.bowl;
}

// The vertices of feature `element`: a surface triangle below `face_count`,
// otherwise a surface edge. Returns how many (3 or 2).
[[seam::device_fn]] inline unsigned
sphere_feature_vertices(const Vec3u *face, unsigned face_count,
                        const Vec2u *edge, unsigned element, unsigned *index) {
    if (element < face_count) {
        const Vec3u f = face[element];
        index[0] = f[0];
        index[1] = f[1];
        index[2] = f[2];
        return 3u;
    }
    const Vec2u e = edge[element - face_count];
    index[0] = e[0];
    index[1] = e[1];
    index[2] = e[0];
    return 2u;
}

// Whether any of the feature's vertices owns a degree of freedom: a feature
// made only of fix pins and massless (static solid) vertices cannot move, and
// a sphere is static too, so there is nothing between them to hold.
[[seam::device_fn]] inline bool
sphere_feature_moves(const VertexProp *vertex_prop, const unsigned *index,
                     unsigned n) {
    for (unsigned k = 0; k < n; ++k) {
        const VertexProp prop = vertex_prop[index[k]];
        if (prop.fix_index == 0u && prop.mass > 0.0f) {
            return true;
        }
    }
    return false;
}


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
    // THE FRICTION'S MEMORY (`friction_memory.kernel.cpp`): the last committed
    // step's pairs, sorted by key, and this iteration's record of every pair.
    const unsigned *memory_word0, const unsigned *memory_word1,
    const float *memory_slip, unsigned memory_count,
    unsigned *record_word0, unsigned *record_word1, unsigned *record_vertex, float *record_float,
    compute::atomic_uint_t *record_count, unsigned record_capacity,
    // THE LOAD EVERY FRICTION TERM BELOW ANCHORS ITSELF ON: each vertex's
    // residual before any contact plus its lagged non-friction contact load,
    // formed once an iteration (`friction_load_prepare`). It is not `force`
    // itself: the contact passes deposit into that array through atomics across
    // vertices, so a pair reading it would read a row another had moved.
    const float *friction_load, const float *friction_cone, float dt,
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
    AnalyticShareInput residual_i;
    {
        friction_cone_lagged(friction_cone, i, residual_i.cone_sum[0],
                             residual_i.cone_pairs[0]);
        residual_i.residual[0] =
            Vec3f(friction_load[3u * i], friction_load[3u * i + 1u],
                  friction_load[3u * i + 2u]);
    }
    residual_i.weight[0] = 1.0f;
    residual_i.count = 1u;
    Mat3x3f H = Mat3x3f::Zero();
    Vec3f f = Vec3f::Zero();
    const Vec3f x = eval_x[i];
    // Both operands copied into thread space: `x` above already is, and the
    // subtraction needs the two in one address space.
    const Vec3f current_i = current[i];
    const Vec3f dx = x - current_i;

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
            const Vec3f w = x - y;
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
            const float d2 = (x - center).squaredNorm();
            if (collider.kinematic) {
                if (d2) {
                    const Vec3f normal = (x - center).normalized();
                    const float eff_radius =
                        reverse ? (radius - ghat) : (radius + ghat);
                    const Vec3f target = eff_radius * normal;
                    const Vec3f o = x - center;
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
                                local_hess, dx, Vec3f::Zero(), 0.0f, -normal,
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
                                local_hess, dx, Vec3f::Zero(), 0.0f, normal,
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
                    Vec3f normal = (x - center).normalized();
                    const Vec3f projected_x = radius * normal;
                    const Vec3f o = x - center;
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
                    const Vec2u memory_key = friction_memory_key(FRICTION_KIND_SPHERE, i, j);
                    float slide;
                    const Vec3f carried = friction_memory_lookup(
                        memory_key, memory_word0, memory_word1, memory_slip, memory_count, slide);
                    const AnalyticContactResult contact =
                        analytic_contact_evaluate(
                            local_hess, dx, carried, slide, normal,
                            (o - projected_x).dot(normal), gap, ghat,
                            friction, friction_eps, mass, residual_i, dt,
                            false, true);
                    if (contact.normal_force > 0.0f) {
                        const unsigned memory_index[1] = {i};
                        const float memory_weight[1] = {1.0f};
                        friction_memory_record(
                            memory_key,
                            friction_memory_tangential(carried, normal), normal,
                            contact.normal_force, friction, 1.0f,
                            memory_index, memory_weight, 1u, !is_grain,
                            record_word0, record_word1, record_vertex,
                            record_float, record_count, record_capacity);
                    }
                    // A GRAIN'S FRICTION IS APPLIED HERE, every other vertex's
                    // by the frozen pass from the step's first-iteration
                    // record (`friction_memory.kernel.cpp`, "THE RECORD IS
                    // ALSO THE STEP'S FRICTION").
                    if (is_grain) {
                        f += contact.force;
                        H += contact.hessian;
                    } else {
                        f += contact.force - contact.friction_gradient;
                        H += contact.hessian - contact.friction_hessian;
                    }
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
            const Vec3f e = x - ground;
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
                float gap = (x - plane.ground).dot(up);
                if (plane.kinematic) {
                    gap = fmath::max(gap, constraint_tol * ghat);
                }
                DIAG_ASSERT4(diag, gap >= 0.0f, gap, ghat, (float)i,
                            (float)j);
                const Vec2u memory_key = friction_memory_key(FRICTION_KIND_FLOOR, i, j);
                float slide;
                const Vec3f carried = friction_memory_lookup(
                    memory_key, memory_word0, memory_word1, memory_slip, memory_count, slide);
                const AnalyticContactResult contact =
                    analytic_contact_evaluate(
                        local_hess, dx, carried, slide, up, (-projected_x).dot(up), gap,
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
                if (contact.normal_force > 0.0f) {
                    const unsigned memory_index[1] = {i};
                    const float memory_weight[1] = {1.0f};
                    friction_memory_record(
                        memory_key, friction_memory_tangential(carried, up), up,
                        contact.normal_force, friction, 1.0f,
                        memory_index, memory_weight, 1u, !is_grain,
                        record_word0, record_word1, record_vertex, record_float,
                        record_count, record_capacity);
                }
                // As the sphere's: only a grain applies its friction here.
                if (is_grain) {
                    f += contact.force;
                    H += contact.hessian;
                } else {
                    f += contact.force - contact.friction_gradient;
                    H += contact.hessian - contact.friction_hessian;
                }
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

// The smaller of two times of impact, NaN-propagating: a NaN bound must reach
// the positive-time assert and the host's check rather than be dropped by a
// fminf, which returns the other operand.
[[seam::device_fn]] inline float analytic_toi_min(float toi, float t) {
    return (t < toi || t != t) ? t : toi;
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
            const float r0 = (p0 - center0).norm();
            const float r1 = (p1 - center1).norm();
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
            const float h0 = plane.up.dot((p0 - plane.ground));
            const float h1 = plane.up.dot((p1 - plane.ground));
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
            const float r0 = (p0 - position).norm();
            const float r1 = (p1 - position).norm();
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
                        analytic_toi_min(out_toi[i], line_search_max_t * t);
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
            const float r0 = (p0 - center0).norm();
            const float r1 = (p1 - center1).norm();
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
            //
            // A START EXACTLY ON THE SURFACE (clearance0 == 0) is crossed only
            // when it moves inward. The symmetric test alone calls it crossed
            // whatever it does, and the crossing fraction is then 0 / 0 (a NaN
            // the step would drop) when it slides along the surface, or 0 when
            // it moves outward; inward it is 0, which the assert below refuses.
            const bool crossed = (clearance0 > 0.0f)
                                     ? (clearance1 < park)
                                     : (clearance0 == 0.0f
                                            ? (clearance1 < 0.0f)
                                            : ((r0 - r) * (r1 - r) <= 0.0f));
            if (crossed) {
                const float t = accd::park_crossing_analytic(
                    clearance0, clearance1, park);
                out_toi[i] =
                    analytic_toi_min(out_toi[i], line_search_max_t * t);
                DIAG_ASSERT4(diag, out_toi[i] > 0.0f, out_toi[i],
                            line_search_max_t, (float)i, 1.0f);
            }
            // A SOLID SPHERE IS ALSO CROSSED BETWEEN TWO CLEAR ENDPOINTS: the
            // straight segment dips toward the center by up to `|d|^2 / 8R`
            // (`sphere_level_crossing`), so it is solved on the segment. A
            // container cannot be left this way, its inside being convex, and
            // a bowl's center follows the query, so neither has a fixed
            // quadratic.
            if (!reverse && !bowl && clearance0 > 0.0f) {
                const float gap0 = sphere_gap(p0, center, r, false);
                if (gap0 > 0.0f) {
                    const float level =
                        accd::park_gap_analytic(gap0, collider.ghat, ccd_eps);
                    const float t = sphere_level_crossing(
                        (p0 - center), (p1 - p0), r,
                        false, gap0, level);
                    if (t <= 1.0f || t != t) {
                        out_toi[i] =
                            analytic_toi_min(out_toi[i], line_search_max_t * t);
                        DIAG_ASSERT4(diag, out_toi[i] > 0.0f, out_toi[i],
                                    line_search_max_t, (float)i, 1.0f);
                    }
                }
            }
        }
        for (uint32_t j = 0; j < floor_count; ++j) {
            const Floor plane = floor[j];
            if (plane.kinematic) {
                continue;
            }
            const Vec3f up = plane.up;
            const Vec3f ground = plane.ground;
            const float h0 = up.dot((p0 - ground));
            const float h1 = up.dot((p1 - ground));
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
                    analytic_toi_min(out_toi[i], line_search_max_t * t);
                DIAG_ASSERT4(diag, out_toi[i] > 0.0f, out_toi[i],
                            line_search_max_t, (float)i, 2.0f);
            }
        }
    }
}

// THE BARRIER OF A SURFACE TRIANGLE OR EDGE AGAINST EVERY STATIC SOLID SPHERE
// (`element` runs over the triangles, then the edges).
//
// THE TERM IS THE VERTEX BARRIER'S, AT THE FEATURE'S CLOSEST POINT. Where the
// closest point `y = sum_k w_k x_k` of the feature to the center is interior,
// its gap `|y - c| - R` (`sphere_feature_gap`, a lower bound on the true one)
// gets the cubic barrier `analytic_contact_evaluate` gives a vertex, with the
// radial normal through `y`, friction included. The weights are the closest
// point's, so by the envelope theorem the force `-dE/dx_k = w_k f` is the exact
// gradient of the barrier on the feature's true distance, and the Hessian
// blocks `w_a w_b H` form `w w^T (x) H`, positive semidefinite because `H` is.
//
// THE POINT'S OWN QUANTITIES are the weighted means over the vertices that own
// a degree of freedom, as `friction_slip_prediction` weights a mesh pair: the
// elastic diagonal block and the mass the elasticity-inclusive stiffness reads,
// the friction coefficient, and the residual the friction anchors on (a weighted
// SUM, the generalized force at the point). The slip is the point's own motion.
//
// THE MULTIPLICITY IS THE MESH CONTACT'S (`contact_point_edge_at`), with the
// sphere's center in the place of the query point. A triangle carries the
// contact where its closest point is interior; an edge carries it where its own
// closest point is interior, once for each incident triangle whose closest point
// is NOT interior, or once when it has none (a rod segment). At the boundary
// between a triangle's interior and one of its edges the two gaps agree, so the
// triangle's share passes to the edge without a jump. Vertices keep their own
// barrier (`vertex_constraint`), as point-point does beside the other two.
//
// EVERY HESSIAN BLOCK GOES INTO THE FIXED MATRIX: a triangle's and an edge's
// vertex pairs are in its pattern by construction (`builder.rs` registers every
// surface triangle and edge). A block the pattern refuses is COUNTED in
// `out_missing[0]` and the driver stops on it; it is never dropped quietly.
// `out_missing[1]` counts the contacts, for the status channel's analytic tally.
[[seam::entry(element)]]
[[seam::device_fn]] inline void sphere_feature_constraint(
    const Vec3f *eval_x,
    const Vec3f *current,
    const Vec3u *face, unsigned face_count,
    const Vec2u *edge,
    const unsigned *edge_face_index,
    const unsigned *edge_face_offset, unsigned has_edge_face,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const Sphere *sphere, unsigned sphere_count,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value, unsigned row_count,
    unsigned friction_mode, float friction_eps,
    // THE FRICTION'S MEMORY (`friction_memory.kernel.cpp`): the last committed
    // step's pairs, sorted by key, and this iteration's record of every pair.
    const unsigned *memory_word0, const unsigned *memory_word1,
    const float *memory_slip, unsigned memory_count,
    unsigned *record_word0, unsigned *record_word1, unsigned *record_vertex, float *record_float,
    compute::atomic_uint_t *record_count, unsigned record_capacity,
    const float *friction_load, const float *friction_cone, float dt,
    compute::atomic_float_t *force,
    compute::atomic_float_t *out_fixed_value,
    compute::atomic_uint_t *out_missing,
    unsigned element,
    DiagHandle diag) {
    unsigned index[3];
    const unsigned n = sphere_feature_vertices(face, face_count, edge, element, index);
    if (!sphere_feature_moves(vertex_prop, index, n)) {
        return;
    }
    Vec3f P[3];
    for (unsigned k = 0; k < n; ++k) {
        P[k] = eval_x[index[k]];
    }
    for (unsigned j = 0; j < sphere_count; ++j) {
        const Sphere collider = sphere[j];
        if (!sphere_feature_applies(collider)) {
            continue;
        }
        const Vec3f center = collider.center;
        float w[3];
        unsigned region;
        Vec3f delta;
        const float gap =
            sphere_feature_gap(P, n, center, collider.radius, w, region, delta);
        if (region != sphere_feature_region_interior()) {
            continue;
        }
        // Past the authored thickness the feature has passed through a thin
        // shell by design, as the vertex barrier lets a vertex.
        if (collider.thickness > 0.0f && -gap > collider.thickness) {
            continue;
        }
        const float ghat = collider.ghat;
        if (!(gap < ghat)) {
            continue;
        }
        // The non-negative gap the barrier is defined on, which the sweep and
        // the step-end gate keep; the stiffness divides by it.
        DIAG_ASSERT4(diag, gap > 0.0f, gap, collider.radius, (float)element,
                     (float)j);
        const float dist = delta.norm();
        if (!(gap > 0.0f) || !(dist > 0.0f)) {
            continue;
        }
        float multiplicity = 1.0f;
        if (element >= face_count && has_edge_face != 0u) {
            const unsigned e = element - face_count;
            const unsigned begin = edge_face_offset[e];
            const unsigned end = edge_face_offset[e + 1u];
            if (end > begin) {
                unsigned outside = 0u;
                for (unsigned m = begin; m < end; ++m) {
                    const Vec3u g = face[edge_face_index[m]];
                    const Vec3f g0 = eval_x[g[0]];
                    const Vec3f g1 = eval_x[g[1]];
                    const Vec3f g2 = eval_x[g[2]];
                    // The triangle's own test (`sphere_feature_gap`), on the
                    // same operands in the same order, so the two passes agree
                    // on every region.
                    const Vec3f ge1 = g1 - g0;
                    const Vec3f ge2 = g2 - g0;
                    const Vec3f gp = center - g0;
                    float gw[3];
                    const unsigned face_region =
                        triangle_closest_weights(ge1, ge2, gp, gw);
                    if (face_region != sphere_feature_region_interior()) {
                        ++outside;
                    }
                }
                multiplicity = static_cast<float>(outside);
            }
        }
        if (!(multiplicity > 0.0f)) {
            continue;
        }
        const Vec3f normal = fmath::div(1.0f, dist) * delta;
        Mat3x3f local_hess = Mat3x3f::Zero();
        Vec3f slip = Vec3f::Zero();
        AnalyticShareInput point_residual;
        point_residual.count = 0u;
        float mass = 0.0f;
        float friction = 0.0f;
        float weight = 0.0f;
        for (unsigned k = 0; k < n; ++k) {
            const Vec3f xk = eval_x[index[k]];
            const Vec3f ck = current[index[k]];
            slip += w[k] * (xk - ck);
            const VertexProp prop = vertex_prop[index[k]];
            if (prop.fix_index == 0u && prop.mass > 0.0f) {
                local_hess += w[k] * fixed_csr_read(fixed_index, fixed_offset,
                                                    fixed_value, row_count,
                                                    index[k], index[k]);
                const unsigned row = 3u * index[k];
                const unsigned at = point_residual.count;
                friction_cone_lagged(friction_cone, index[k],
                                     point_residual.cone_sum[at],
                                     point_residual.cone_pairs[at]);
                point_residual.residual[at] =
                    w[k] * Vec3f(friction_load[row], friction_load[row + 1u],
                                 friction_load[row + 2u]);
                point_residual.weight[at] = fmath::abs(w[k]);
                point_residual.count = at + 1u;
                mass += w[k] * prop.mass;
                friction += w[k] * vertex_param[prop.param_index].friction;
                weight += w[k];
            }
        }
        if (!(weight > 0.0f)) {
            continue;
        }
        const float inverse = fmath::div(1.0f, weight);
        local_hess = inverse * local_hess;
        mass *= inverse;
        friction = combine_friction_values(collider.friction, friction * inverse,
                                           friction_mode);
        // The signed distance into the barrier's band, `gap - ghat`, the
        // vertex barrier's `(o - projected_x) . normal`.
        const Vec2u memory_key = friction_memory_key(FRICTION_KIND_SPHERE_FEATURE, element, j);
        float slide;
        const Vec3f carried = friction_memory_lookup(
            memory_key, memory_word0, memory_word1, memory_slip, memory_count, slide);
        const AnalyticContactResult contact = analytic_contact_evaluate(
            local_hess, slip, carried, slide, normal, gap - ghat, gap, ghat, friction,
            friction_eps, mass, point_residual, dt, false, true);
        if (contact.normal_force > 0.0f) {
            friction_memory_record(
                memory_key, friction_memory_tangential(carried, normal), normal,
                contact.normal_force, friction, multiplicity,
                index, w, n, true, record_word0, record_word1, record_vertex,
                record_float, record_count, record_capacity);
        }
        // The friction is applied by the frozen pass from the step's
        // first-iteration record (`friction_memory.kernel.cpp`, "THE RECORD IS
        // ALSO THE STEP'S FRICTION"); this pass deposits the barrier alone.
        const Vec3f contact_force = contact.force - contact.friction_gradient;
        const Mat3x3f contact_hessian = contact.hessian - contact.friction_hessian;
        compute::atomic_add(out_missing + 1u, 1u);
        for (unsigned k = 0; k < n; ++k) {
            const float scale = multiplicity * w[k];
            for (unsigned d = 0; d < 3u; ++d) {
                compute::atomic_add(force + 3u * index[k] + d,
                                    scale * contact_force[d]);
            }
        }
        // The upper triangle; the matrix reaches the lower one through its
        // transpose index, and `H` is symmetric.
        for (unsigned a = 0; a < n; ++a) {
            for (unsigned b = 0; b < n; ++b) {
                const unsigned r = index[a];
                const unsigned c = index[b];
                if (r > c || (r == c && a != b)) {
                    continue;
                }
                const Mat3x3f block = (multiplicity * w[a] * w[b]) * contact_hessian;
                if (!fixed_csr_atomic_push(fixed_index, fixed_offset, out_fixed_value,
                                           row_count, r, c, block)) {
                    compute::atomic_add(out_missing, 1u);
                }
            }
        }
    }
}

// THE LINE-SEARCH BOUND OF A TRIANGLE OR AN EDGE AGAINST A STATIC SPHERE: the
// first fraction `t` of the extended segment (as `vertex_constraint_sweep`
// sweeps it) at which the feature's gap may fall to its PARK
// (`accd::park_gap_analytic`, the vertex sweep's), or 2 when it cannot within
// `t <= 1`.
//
// CONSERVATIVE ADVANCEMENT WITH A SECOND-ORDER BOUND. The endpoints alone are
// not enough: a straight motion dips between them. And the plain Lipschitz
// bound (no point of the feature moves faster than its fastest vertex) is first
// order in a motion that only slides the feature over the sphere, which it would
// stop at a fraction of `gap / step`. The bound used instead, at a fraction
// `tau` where the feature's closest material point is `y` with `delta = y - c`,
// `V` its velocity, and the feature's velocity gradient `A` (an affine field
// over the feature, so `V_w = V + A s` for a point offset `s` from `y` in the
// feature's plane):
//
//     m(tau + h) >= m + h m' - h^2 Q,   m = |delta|^2 - R^2,   m' = 2 delta . V,
//     Q = (|c + V| + kappa |c|)^2 - |V|^2,   c = A^T delta (in plane),
//     kappa = 1.5 h K / (1 - h K),   K >= ||A||.
//
// It follows from three facts: `|y + s - c|^2 >= |delta|^2 + |s|^2` for every
// `s` the feature reaches (`y` is its closest point, so `delta . s >= 0`); the
// path of each point is a straight line, so the squared distance is a quadratic
// in `h` whose leading term is non-negative; and the remaining cross term is
// minimized over `s` in closed form, with the pseudo-inverse of `I + h A`
// perturbed by at most `kappa`. A feature turning about the center (sliding
// round the sphere) has `c = -V` and `Q <= 0`; a translation along its own
// plane has `c = 0` and `Q = 0`; only a feature turning about its own contact
// point, which really does cut into the sphere by `(h |A|)^2 R / 2`, pays. Each
// advancement step recomputes the closest point and its conservative gap at the
// new fraction, and every step it takes is positive while the gap is above
// the park, so the bound never stalls at zero on a feature that can move.
[[seam::device_fn]] inline float sphere_feature_crossing(
    const Vec3f *Q0, const Vec3f *Q1, unsigned n, const Vec3f &center,
    float radius, float ghat, float thickness, float ccd_eps) {
    Vec3f d[3];
    float dmax = 0.0f;
    for (unsigned k = 0; k < n; ++k) {
        const Vec3f a = Q0[k];
        const Vec3f b = Q1[k];
        d[k] = b - a;
        dmax = fmath::max(dmax, d[k].norm());
    }
    if (!(dmax > 0.0f)) {
        return 2.0f;
    }
    float w[3];
    unsigned region;
    Vec3f delta;
    const float g0 = sphere_feature_gap(Q0, n, center, radius, w, region, delta);
    if (thickness > 0.0f && -g0 > thickness) {
        return 2.0f;
    }
    // A feature that starts at or inside the surface cannot be given
    // clearance by a shorter step; the caller's assert on a positive time of
    // impact refuses it, as the vertex sweep's does.
    if (!(g0 > 0.0f)) {
        return 0.0f;
    }
    const float level = accd::park_gap_analytic(g0, ghat, ccd_eps);
    // Nothing on the feature moves faster than its fastest vertex.
    if (g0 - dmax > level) {
        return 2.0f;
    }
    float tau = 0.0f;
    float g = g0;
    Vec3f P[3];
    for (unsigned k = 0; k < n; ++k) {
        P[k] = Q0[k];
    }
    for (unsigned iteration = 0; iteration < 48u; ++iteration) {
        if (iteration > 0u) {
            for (unsigned k = 0; k < n; ++k) {
                const Vec3f a = Q0[k];
                const Vec3f b = Q1[k];
                P[k] = tau * (b - a) + a;
            }
            g = sphere_feature_gap(P, n, center, radius, w, region, delta);
            // Close enough to the park: the bound is conservative, so stopping
            // a little short of it costs a little step and nothing else.
            if (!(g - level > 0.0625f * level)) {
                return tau;
            }
        }
        const float remaining = 1.0f - tau;
        // The feature's motion over the rest of the segment, from the state at
        // `tau`: the closest point's velocity, and its velocity gradient.
        Vec3f V = Vec3f::Zero();
        for (unsigned k = 0; k < n; ++k) {
            V += w[k] * d[k];
        }
        const float vnorm = V.norm();
        const float dist = delta.norm();
        // `m'`, less an allowance for `delta`'s error: its own fp32 rounding,
        // and how far the constructed `y` can sit from the feature
        // (`sphere_feature_point_error`), which on float positions is absolute
        // rather than relative to `dist`.
        const float position_error = sphere_feature_point_error(P, n, w, region);
        const float slope = 2.0f * delta.dot(V) -
                            2.0f * vnorm * (dist * 4.0e-6f + position_error);
        float second = -1.0f;
        float K = 0.0f;
        const Vec3f e1 = P[1] - P[0];
        const Vec3f f1 = d[1] - d[0];
        Vec3f c = Vec3f::Zero();
        if (n == 3u) {
            const Vec3f e2 = P[2] - P[0];
            const Vec3f f2 = d[2] - d[0];
            const float g11 = e1.dot(e1);
            const float g12 = e1.dot(e2);
            const float g22 = e2.dot(e2);
            const float det = g11 * g22 - g12 * g12;
            if (det > 1.0e-6f * g11 * g22) {
                const float inv = fmath::div(1.0f, det);
                const float i11 = g22 * inv;
                const float i12 = -g12 * inv;
                const float i22 = g11 * inv;
                const float a1 = f1.dot(delta);
                const float a2 = f2.dot(delta);
                const float z1 = i11 * a1 + i12 * a2;
                const float z2 = i12 * a1 + i22 * a2;
                c = z1 * e1 + z2 * e2;
                // ||A|| <= ||F G^-1 M^T||_F = sqrt(tr(G^-1 F^T F)).
                const float trace = i11 * f1.dot(f1) + 2.0f * i12 * f1.dot(f2) +
                                    i22 * f2.dot(f2);
                K = fmath::sqrt(fmath::max(trace, 0.0f));
                second = 1.0f;
            }
        } else {
            const float g11 = e1.dot(e1);
            if (g11 > 0.0f) {
                c = fmath::div(f1.dot(delta), g11) * e1;
                K = fmath::sqrt(fmath::div(f1.dot(f1), g11));
                second = 1.0f;
            }
        }
        // The room the bound has to spend, in `m`'s units, formed through the
        // gaps so that it does not cancel.
        const float room = (g - level) * (2.0f * radius + g + level);
        float h;
        if (second < 0.0f) {
            // A degenerate feature: the plain Lipschitz bound, in gap units.
            h = fmath::div(g - level, dmax);
        } else {
            float hmax = remaining;
            if (K > 0.0f) {
                hmax = fmath::min(hmax, fmath::div(0.25f, K));
            }
            const float hk = hmax * K;
            const float kappa = fmath::div(1.5f * hk, 1.0f - hk);
            const float cnorm = c.norm();
            const float lead = (c + V).norm() + kappa * cnorm;
            const float Qc = fmath::max(lead * lead - vnorm * vnorm, 0.0f) * 1.001f;
            // The first root of `room + h slope - h^2 Qc` in `(0, hmax]`.
            if (Qc > 0.0f) {
                const float disc = fmath::sqrt(slope * slope + 4.0f * Qc * room);
                h = slope >= 0.0f ? fmath::div(slope + disc, 2.0f * Qc)
                                  : fmath::div(2.0f * room, disc - slope);
            } else if (slope < 0.0f) {
                h = fmath::div(room, -slope);
            } else {
                h = 2.0f * hmax + 1.0f;
            }
            if (h >= hmax) {
                tau += hmax;
                if (tau >= 1.0f) {
                    return 2.0f;
                }
                continue;
            }
        }
        if (!(h > 0.0f)) {
            return tau;
        }
        tau += h;
        if (tau >= 1.0f) {
            return 2.0f;
        }
        // A step this short means the feature is closing on its park as fast
        // as the bound allows; what has been advanced so far is safe.
        if (h < 1.0e-7f) {
            return tau;
        }
    }
    return tau;
}

// THE SWEEP over every surface triangle and edge (`element`, triangles first),
// writing the smallest bound over the static solid spheres into `out_toi` in
// the line search's units (`line_search_max_t` times the fraction). The driver
// folds it with the vertex sweep's.
[[seam::entry(element)]]
[[seam::device_fn]] inline void sphere_feature_sweep(
    const Vec3f *x0, const Vec3f *x1,
    const Vec3u *face, unsigned face_count,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const Sphere *sphere, unsigned sphere_count,
    float ccd_eps, float line_search_max_t,
    float *out_toi,
    unsigned element,
    DiagHandle diag) {
    out_toi[element] = line_search_max_t;
    unsigned index[3];
    const unsigned n = sphere_feature_vertices(face, face_count, edge, element, index);
    if (!sphere_feature_moves(vertex_prop, index, n)) {
        return;
    }
    Vec3f Q0[3];
    Vec3f Q1[3];
    for (unsigned k = 0; k < n; ++k) {
        const Vec3f a = x0[index[k]];
        const Vec3f b = x1[index[k]];
        Q0[k] = a;
        Q1[k] = line_search_max_t * (b - a) + a;
    }
    float t = 2.0f;
    for (unsigned j = 0; j < sphere_count; ++j) {
        const Sphere collider = sphere[j];
        if (!sphere_feature_applies(collider)) {
            continue;
        }
        t = analytic_toi_min(t, sphere_feature_crossing(Q0, Q1, n, collider.center,
                                                  collider.radius, collider.ghat,
                                                  collider.thickness, ccd_eps));
    }
    if (t <= 1.0f || t != t) {
        out_toi[element] = line_search_max_t * t;
        DIAG_ASSERT4(diag, out_toi[element] > 0.0f, out_toi[element],
                     line_search_max_t, (float)element, 3.0f);
    }
}

// THE CONSERVATIVE CHECK: counts every surface triangle, edge and free vertex
// that is inside a static solid sphere at the pose `x`. `element` runs over the
// triangles, then the edges, then the surface vertices. A vertex is tested by
// the sign of its gap (`sphere_gap`, the vertex barrier's); a triangle or an edge by
// `sphere_feature_gap`, which never exceeds its true minimum gap, so a pose this
// passes has nothing inside a sphere. A feature or vertex deeper than the
// sphere's authored thickness has passed through it by design and is not
// counted.
[[seam::entry(element)]]
[[seam::device_fn]] inline void sphere_feature_gate(
    const Vec3f *x,
    const Vec3u *face, unsigned face_count,
    const Vec2u *edge, unsigned edge_count,
    const VertexProp *vertex_prop,
    const Sphere *sphere, unsigned sphere_count,
    compute::atomic_uint_t *out_count,
    unsigned element) {
    if (element >= face_count + edge_count) {
        const unsigned v = element - face_count - edge_count;
        const VertexProp prop = vertex_prop[v];
        if (!(prop.fix_index == 0u && prop.mass > 0.0f)) {
            return;
        }
        const Vec3f p = x[v];
        for (unsigned j = 0; j < sphere_count; ++j) {
            const Sphere collider = sphere[j];
            if (!sphere_feature_applies(collider)) {
                continue;
            }
            const float gap = sphere_gap(p, collider.center, collider.radius, false);
            if (gap < 0.0f && !(collider.thickness > 0.0f && -gap > collider.thickness)) {
                compute::atomic_add(out_count, 1u);
            }
        }
        return;
    }
    unsigned index[3];
    const unsigned n = sphere_feature_vertices(face, face_count, edge, element, index);
    if (!sphere_feature_moves(vertex_prop, index, n)) {
        return;
    }
    Vec3f P[3];
    for (unsigned k = 0; k < n; ++k) {
        P[k] = x[index[k]];
    }
    for (unsigned j = 0; j < sphere_count; ++j) {
        const Sphere collider = sphere[j];
        if (!sphere_feature_applies(collider)) {
            continue;
        }
        float w[3];
        unsigned region;
        Vec3f delta;
        const float gap =
            sphere_feature_gap(P, n, collider.center, collider.radius, w, region, delta);
        if (gap < 0.0f && !(collider.thickness > 0.0f && -gap > collider.thickness)) {
            compute::atomic_add(out_count, 1u);
        }
    }
}
