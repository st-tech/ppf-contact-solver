// File: distance.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include "../common.hpp"
#include "../data.hpp"

namespace distance {

template <class T, class Y>
__device__ Vec2<Y> point_edge_distance_coeff(const Vec3<T> &p,
                                             const Vec3<T> &e0,
                                             const Vec3<T> &e1) {
    Vec3<Y> r = (e1 - e0).template cast<Y>();
    Y d = r.dot(r);
    if (d > Y(0.0)) {
        Y t = r.dot((p - e0).template cast<Y>()) / d;
        return Vec2<Y>(Y(1.0) - t, t);
    } else {
        return Vec2<Y>::Ones() / Y(2.0);
    }
}

// Barycentric coordinates of p projected perpendicularly onto the plane of the
// triangle. The result is unclamped, so a coordinate outside [0, 1] means the
// projection falls outside the triangle and names a point of its plane rather
// than of the triangle itself. Callers that need a point ON the triangle read
// the sign of the coordinates to decide, or use the closest-point routine
// below, which selects among the boundary as well.
//
// The coordinates are taken through the cross product rather than through the
// Gram system (r0.r0)(r1.r1) - (r0.r1)^2. The values are the same and the
// conditioning is not: that determinant subtracts two products which agree to
// within the SQUARED sine of the angle between the two edges, and both
// numerators cancel the same way, so on a triangle with one wide interior angle
// the quotient keeps only a few of single precision's digits. Each component of
// the cross product subtracts products agreeing only to within the FIRST power
// of that sine, a whole factor of the sine better. That matters because every
// caller tests the coordinates against 0 and 1 to decide whether the projection
// lies inside, and on a sliver the Gram form can put that verdict on the wrong
// side of the comparison.
template <class T, class Y>
__device__ Vec3<Y>
point_triangle_distance_coeff(const Vec3<T> &p, const Vec3<T> &t0,
                              const Vec3<T> &t1, const Vec3<T> &t2) {
    Vec3<Y> r0 = (t1 - t0).template cast<Y>();
    Vec3<Y> r1 = (t2 - t0).template cast<Y>();
    Vec3<Y> d = (p - t0).template cast<Y>();
    Vec3<Y> nrm = r0.cross(r1);
    Y nn = nrm.dot(nrm);
    if (nn > Y(0)) {
        Y inv_nn = Y(1) / nn;
        Y u = d.cross(r1).dot(nrm) * inv_nn;
        Y v = r0.cross(d).dot(nrm) * inv_nn;
        return Vec3<Y>(Y(1.0) - u - v, u, v);
    } else {
        // Zero area: the three points are collinear (or coincident), so there
        // is no plane to project onto. The longest edge spans the whole
        // degenerate triangle, which makes projecting onto it the closest
        // available answer rather than an arbitrary one.
        Vec3<Y> e1 = (t2 - t1).template cast<Y>();
        Vec3<Y> e2 = (t0 - t2).template cast<Y>();
        Y len0 = r0.dot(r0);
        Y len1 = e1.dot(e1);
        Y len2 = e2.dot(e2);
        if (len0 >= len1 && len0 >= len2) {
            Vec2<Y> w = point_edge_distance_coeff<T, Y>(p, t0, t1);
            return Vec3<Y>(w(0), w(1), Y(0.0));
        } else if (len1 >= len2) {
            Vec2<Y> w = point_edge_distance_coeff<T, Y>(p, t1, t2);
            return Vec3<Y>(Y(0.0), w(0), w(1));
        } else {
            Vec2<Y> w = point_edge_distance_coeff<T, Y>(p, t2, t0);
            return Vec3<Y>(w(1), Y(0.0), w(0));
        }
    }
}

// Clamp to the closed unit interval. The clamped values are the literals 0
// and 1, which is what lets a caller read a coefficient of exactly 0 or 1 as
// "the closest point is an endpoint". A value that is not a number leaves as
// zero, so a returned coefficient can never name a point off its segment.
template <class Y> __device__ Y clamp_unit(Y v) {
    return v > Y(0) ? (v < Y(1) ? v : Y(1)) : Y(0);
}

// Closest-point barycentric coefficients of the segments A(s) = ea0 + s r0 and
// B(t) = eb0 + t r1, returned as (1 - s, s, 1 - t, t) with s and t in [0, 1].
//
// F(s, t) = |d + s r0 - t r1|^2 is a convex quadratic on the unit square, so
// its minimum is attained either at an interior stationary point or on the
// boundary. The boundary is four segments, and on each of them F restricts to
// a one-dimensional convex quadratic whose exact minimizer is a clamped
// projection. The five candidates below therefore cover every minimizer, and
// the routine returns whichever of them realizes the smallest distance.
//
// Selecting by the distance actually realized is what leaves the routine with
// no tolerance to tune: every candidate names a point of the unit square, so
// the distance it realizes is an upper bound on the minimum, and a stationary
// point recovered from an ill-conditioned quotient can only fail to win. The
// routine is therefore never asked whether two edges are parallel enough that
// the interior solve should be distrusted, which is the question no threshold
// can answer: an interior optimum is genuinely the answer at angles far below
// any usable parallel epsilon, while for exactly parallel segments the
// minimum is always attained with at least one of the two points at an
// endpoint of its own segment, so the four boundary candidates are exact
// there.
template <class T, class Y>
__device__ Vec4<Y>
edge_edge_distance_coeff(const Vec3<T> &ea0, const Vec3<T> &ea1,
                         const Vec3<T> &eb0, const Vec3<T> &eb1) {
    // Every quantity below is built from pairwise endpoint differences taken
    // in T and cast to Y afterwards, never from a cast of an endpoint itself:
    // a closest-point pair is separated by far less than its distance from the
    // origin, so casting first truncates at that distance and the subtraction
    // then cancels down to the truncation, which can flip the selection below.
    Vec3<Y> r0 = (ea1 - ea0).template cast<Y>();
    Vec3<Y> r1 = (eb1 - eb0).template cast<Y>();
    Vec3<Y> d = (ea0 - eb0).template cast<Y>();
    Y a = r0.dot(r0);
    Y e = r1.dot(r1);
    Y b = r0.dot(r1);
    Y c = r0.dot(d);
    Y f = r1.dot(d);

    // A squared length is exactly zero precisely when its segment has
    // collapsed to a point, and every parameter then names that same point, so
    // the guarded zero is the exact answer here and not a fallback.
    Y inv_a = a > Y(0) ? Y(1) / a : Y(0);
    Y inv_e = e > Y(0) ? Y(1) / e : Y(0);

    // The stationary point, s = (b f - c e) / (a e - b^2), taken through the
    // cross product rather than through those two Gram expressions. The
    // denominator a e - b^2 is |r0 x r1|^2 and the numerator b f - c e is
    // (r0 x r1) . (r1 x d), so the values are the same and the conditioning is
    // not: a e - b^2 subtracts two products that agree to within the SQUARED
    // sine of the angle between the edges, which in single precision leaves no
    // significant digits at all below about 3e-4 rad, whereas each component
    // of the cross product subtracts two products that agree only to within
    // the first power of that sine. A settled contact between two bodies
    // produces near-parallel edge pairs constantly, so this is the operating
    // regime and not a corner.
    //
    // What conditioning remains is bounded independently of the angle. Since t
    // follows s by exact projection, an error ds in s displaces the connecting
    // vector only by ds times the part of r0 perpendicular to r1, of magnitude
    // ds |r0| sin(angle); and the relative error of the quotient is itself of
    // order (unit round-off) / sin(angle), so the product is of order
    // (unit round-off) |r0| at every angle. That is already the resolution at
    // which the difference vector below can be evaluated in Y, so a
    // compensated cross product, which would remove the surviving 1/sin factor
    // from the quotient, does not lower that bound.
    Vec3<Y> n = r0.cross(r1);
    Y nn = n.dot(n);
    Y inv_nn = nn > Y(0) ? Y(1) / nn : Y(0);
    Y s_stat = clamp_unit<Y>(n.dot(r1.cross(d)) * inv_nn);
    Y t_stat = clamp_unit<Y>((b * s_stat + f) * inv_e);

    // The four boundary segments of the unit square: s = 0 and s = 1 project
    // A's endpoints onto B, and t = 0 and t = 1 project B's endpoints onto A.
    Y t_a0 = clamp_unit<Y>(f * inv_e);
    Y t_a1 = clamp_unit<Y>((f + b) * inv_e);
    Y s_b0 = clamp_unit<Y>(-c * inv_a);
    Y s_b1 = clamp_unit<Y>((b - c) * inv_a);

    const Y cand_s[5] = {s_stat, Y(0), Y(1), s_b0, s_b1};
    const Y cand_t[5] = {t_stat, t_a0, t_a1, Y(0), Y(1)};
    Y best_s = Y(0);
    Y best_t = Y(0);
    Y best_dist = std::numeric_limits<Y>::max();
    // The scan runs in a fixed order and keeps the first of equal distances,
    // so the choice is reproducible where the minimum is attained on a whole
    // interval. Unrolling is load-bearing: the loop indexes the two candidate
    // arrays, and a dynamic index would place them in local memory.
#pragma unroll
    for (unsigned i = 0; i < 5; ++i) {
        // This is an exact identity of the barycentric combination being
        // scored: ((1-s) ea0 + s ea1) - ((1-t) eb0 + t eb1) = d + s r0 - t r1.
        Vec3<Y> v = d + cand_s[i] * r0 - cand_t[i] * r1;
        Y dist = v.dot(v);
        if (dist < best_dist) {
            best_dist = dist;
            best_s = cand_s[i];
            best_t = cand_t[i];
        }
    }
    return Vec4<Y>(Y(1) - best_s, best_s, Y(1) - best_t, best_t);
}

// Barycentric coefficients of the point of the triangle closest to p, returned
// with all three in [0, 1] and summing to one.
//
// The squared distance from p to a point of the triangle is a convex quadratic
// on the barycentric simplex, so its minimum is attained either at the interior
// stationary point (the perpendicular projection, admissible only when it lands
// inside) or on the boundary. The boundary is three segments, and on each of
// them the exact minimizer is a clamped projection. The four candidates below
// therefore cover every minimizer, and the routine returns whichever of them
// realizes the smallest distance.
//
// Selecting by the distance actually realized is what makes the result exact
// rather than merely plausible. Every candidate names a point of the triangle,
// so the distance it realizes is an upper bound on the minimum and a candidate
// that is not the closest can only fail to win. Classifying instead by the
// SIGNS of the barycentric coordinates and then trusting the chosen edge is not
// equivalent: a sign test splits the plane by the triangle's edge lines, which
// separates the three edge regions only for an acute triangle. On an obtuse one
// the lines cross outside, and a point in the region beyond the wide vertex
// falls on the far side of an edge line whose edge is not its nearest feature,
// so the classification commits to that edge and reports a point further away
// than the true closest one.
//
// Over-reporting is the unsafe direction, because ccd_helper sizes each
// conservative advance as (distance - parking clearance) / (Lipschitz bound):
// a distance reported too large buys a step too long, and the sweep can cross a
// contact it never saw. The error is proportional to the true distance, with a
// worst case governed by the widest interior angle, so a sliver triangle turns
// a small mistake into a step that skips the collision entirely.
template <class T, class Y>
__device__ Vec3<Y> point_triangle_distance_coeff_unclassified(
    const Vec3<T> &p, const Vec3<T> &t0, const Vec3<T> &t1, const Vec3<T> &t2) {

    // Differences are taken in T and cast to Y afterwards, never a cast of an
    // endpoint itself: a closest-point pair is separated by far less than its
    // distance from the origin, so casting first truncates at that distance and
    // the subtraction then cancels down to the truncation.
    Vec3<Y> r0 = (t1 - t0).template cast<Y>();
    Vec3<Y> r1 = (t2 - t0).template cast<Y>();
    Vec3<Y> d = (p - t0).template cast<Y>();

    Vec3<Y> cand[4];
    // The interior candidate is the only one that can be inadmissible, so it
    // carries a flag rather than shortening the array: the scan below then runs
    // over a compile-time count with compile-time indices, which is what keeps
    // the candidates in registers. A running count would make the index dynamic
    // and place the array in local memory, in a routine every contact pair of
    // every line search evaluates.
    bool admissible[4] = {false, true, true, true};

    // The perpendicular projection, admissible only where it names a point of
    // the triangle. Elsewhere it is not a candidate at all, so a projection
    // that falls outside, or a degenerate triangle with no plane to project
    // onto, simply contributes nothing and cannot mislead the scan below. The
    // routine above owns this solve, including the cross-product conditioning
    // a sliver needs, so there is one implementation of it rather than two.
    cand[0] = point_triangle_distance_coeff<T, Y>(p, t0, t1, t2);
    admissible[0] = cand[0].minCoeff() >= Y(0);

    // The three boundary segments. clamp_unit keeps each projection on its own
    // edge, which also covers the vertices, so no separate vertex candidate is
    // needed.
    Y s01 = clamp_unit<Y>(point_edge_distance_coeff<T, Y>(p, t0, t1)(1));
    Y s12 = clamp_unit<Y>(point_edge_distance_coeff<T, Y>(p, t1, t2)(1));
    Y s20 = clamp_unit<Y>(point_edge_distance_coeff<T, Y>(p, t2, t0)(1));
    cand[1] = Vec3<Y>(Y(1) - s01, s01, Y(0));
    cand[2] = Vec3<Y>(Y(0), Y(1) - s12, s12);
    cand[3] = Vec3<Y>(s20, Y(0), Y(1) - s20);

    Vec3<Y> best = cand[1];
    Y best_dist = std::numeric_limits<Y>::max();
    // The scan runs in a fixed order and keeps the first of equal distances, so
    // the choice is reproducible where the minimum is attained on a whole edge.
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        if (!admissible[i]) {
            continue;
        }
        // Exact identity of the combination being scored:
        // p - (w0 t0 + w1 t1 + w2 t2) = d - w1 r0 - w2 r1, since the weights
        // sum to one.
        Vec3<Y> v = d - cand[i][1] * r0 - cand[i][2] * r1;
        Y dist = v.dot(v);
        if (dist < best_dist) {
            best_dist = dist;
            best = cand[i];
        }
    }
    return best;
}

template <class T, class Y>
__device__ Vec2<Y> point_edge_distance_coeff_unclassified(const Vec3<T> &p,
                                                          const Vec3<T> &e0,
                                                          const Vec3<T> &e1) {
    Vec2<Y> c = point_edge_distance_coeff<T, Y>(p, e0, e1);
    if (c(0) >= Y(0.0) && c(0) <= Y(1.0)) {
        return c;
    } else {
        if (c(0) > Y(1.0)) {
            return Vec2<Y>(Y(1.0), Y(0.0));
        } else {
            return Vec2<Y>(Y(0.0), Y(1.0));
        }
    }
}

} // namespace distance

#endif
