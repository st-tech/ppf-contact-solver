// File: accd.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ACCD_HPP
#define ACCD_HPP

#include "distance.hpp"
#include <cassert>
#include <cfloat>

namespace accd {

using _coord_ = float;

// Set to 1u by ccd_helper when a contact pair begins the step inside the
// contact offset (d2 <= offset^2): two surfaces start out touching or
// overlapping, which the conservative advance cannot resolve. Recorded here
// instead of trapping the device (a bare assert) so the host can end the run
// with a clear, structured OverlappingStart crash. Defined in contact.cu (the
// only translation unit that includes this header); cleared host-side before
// each line search and read after it.
extern __device__ unsigned g_ccd_overlap;

// A representative overlapping pair (two vertex indices) and its kind
// (0 = vertex-face, 1 = edge-edge, 2 = point-point), recorded once by the first
// contact CCD that detects the overlap so the crash can name the elements the
// way an infeasible-pin crash names its vertex. The CCD structs (which hold the
// indices) record these; ccd_helper only flips g_ccd_overlap. UINT_MAX means
// "unset". Cleared host-side alongside g_ccd_overlap.
extern __device__ unsigned g_ccd_overlap_v0;
extern __device__ unsigned g_ccd_overlap_v1;
extern __device__ unsigned g_ccd_overlap_kind;

// The squared start distance and offset of a flagged pair, in ccd_helper's
// internally rescaled units, written at the flag sites (plain last-writer
// stores: any flagged pair's values answer the diagnostic question of HOW
// overlapping the start was, e.g. exactly 0.0 means the two primitives evaluate
// as coincident). A pair flagged for a collapsed sweep frame has no
// such scale, so it records a distance of exactly zero and leaves the
// offset unset. Cleared host-side alongside g_ccd_overlap.
extern __device__ float g_ccd_overlap_d2;
extern __device__ float g_ccd_overlap_offset;


// Flag a pair whose sweep frame has collapsed to a point. The two primitives
// evaluate as coincident, which is the state ccd_helper's entry check reports,
// so report it the same way: the host reads g_ccd_overlap after the line search
// and ends the run with a structured OverlappingStart crash naming the pair.
// Returning zero grants no advance and makes the caller in contact.cu record
// the pair indices, exactly as it does for an overlap found inside ccd_helper.
// The distance is exactly zero, which needs no units. The offset is left
// unset: ccd_helper's flag sites record it in the frame rescale gave them,
// and a collapsed frame has no such scale, so recording the raw value here
// would put two different meanings behind one reported number.
__device__ inline float report_coincident() {
    g_ccd_overlap = 1u;
    g_ccd_overlap_d2 = 0.0f;
    g_ccd_overlap_offset = -1.0f;
    return 0.0f;
}

template <class T, unsigned R, unsigned C>
__device__ void centerize(SMat<T, R, C> &x) {
    SVec<T, R> mov = SVec<T, R>::Zero();
    T scale(1.0 / C);
    for (int k = 0; k < C; k++) {
        mov += scale * x.col(k);
    }
    for (int k = 0; k < C; k++) {
        x.col(k) -= mov;
    }
}

template <class T, unsigned R, unsigned C>
__device__ float rescale(SMat<T, R, C> &x, SMat<T, R, C> &dx, float max_t) {
    T max_entry =
        max(x.cwiseAbs().maxCoeff(), (x + T(max_t) * dx).cwiseAbs().maxCoeff());
    // Normalize the sweep's frame: the largest coordinate magnitude reached
    // anywhere in the sweep lands just under one.
    float scale = 0.99f / max_entry;
    // The loop below exits only when `scale <= s`, so it makes progress only
    // for a positive finite scale: an infinite one satisfies `scale > s` on
    // every iteration and the loop never ends, hanging the device with no
    // assert and no output. Two frames produce that infinity, and guarding the
    // quotient rather than the divisor covers both. A frame with no extent
    // (every column on the centroid at both ends, so the two primitives are
    // coincident) divides by zero, and a frame whose extent underflows far
    // enough that 0.99f divided by it leaves the float range overflows to the
    // same value. A non-finite extent is caught here as well, before it can
    // propagate through x and dx. Zero is unambiguous as the reported value
    // because every other path returns a product of positive factors.
    // Both halves of the test are load-bearing and neither is redundant: a
    // zero or underflowed extent sends the quotient to positive infinity,
    // which passes `scale > 0.0f`, while an infinite extent sends it to zero,
    // which passes `isfinite`.
    // The flag is raised here rather than only at the callers so that a
    // caller which forgets to check still ends the run loudly instead of
    // advancing on an unscaled frame with a zero offset.
    if (!(scale > 0.0f) || !isfinite(scale)) {
        return report_coincident();
    }
    float scaled = 1.0f;
    float s = 8.0f;
    while (true) {
        if (scale > s) {
            x *= T(s);
            dx *= T(s);
            scaled *= s;
            scale /= s;
        } else {
            x *= T(scale);
            dx *= T(scale);
            scaled *= scale;
            break;
        }
    }
    return scaled;
}

// Direction-agnostic Lipschitz bound on the rate at which the distance between
// the two primitives can shrink, used by ccd_helper as the floor under every
// conservative advance step. The closest point on each primitive is a convex
// combination of that primitive's vertices, so the relative velocity of the
// closest-point pair is a convex combination of the INTER-primitive
// column-velocity differences (columns [0, SPLIT) belong to primitive A,
// [SPLIT, C) to primitive B). Its norm is therefore bounded by the max over
// inter-primitive pairs alone; intra-primitive pairs (both columns on the same
// primitive) cannot increase the closest-point relative velocity and only
// inflate the bound, wasting advance-loop iterations.
//
// That restriction is not merely tighter than the all-pairs max, it is the end
// of this family: the norm is convex on the product of the two barycentric
// simplices, so it attains its maximum at a vertex of that polytope, and the
// vertices are exactly the inter-primitive corner pairs. No bound reading the
// velocities ALONE can be smaller, and the value is attained (place the closest
// pair on the maximizing corner with its relative velocity along the normal).
// Narrowing further requires reading the POSITIONS as well, which is what
// directional_advance below does.
template <class T, unsigned R, unsigned C, unsigned SPLIT>
__device__ float max_relative_u(const SMat<T, R, C> &u) {
    float max_u = 0.0f;
    for (int i = 0; i < SPLIT; i++) {
        for (int j = SPLIT; j < C; j++) {
            SVec<float, R> du = (u.col(i) - u.col(j)).template cast<float>();
            max_u = std::max<float>(max_u, du.squaredNorm());
        }
    }
    return sqrtf(max_u);
}

// Largest advance certified by a single direction, and the exact quantity the
// bound above can only approximate.
//
// Fix a direction. Both primitives are convex hulls of their own columns, and a
// linear objective over a hull is attained at a column, so the separation the
// two vertex SETS present along that direction is a lower bound on the distance:
//
//   d(t) >= g(t) := min_j <n, x_j(t)> - max_i <n, x_i(t)>
//                 = min over inter-primitive pairs of <n, x_j(t) - x_i(t)>
//
// Every column moves at a constant velocity over the sweep, so g is not merely
// bounded but is EXACTLY a minimum of affine functions of the advance s:
//
//   g(t + s) = min over pairs of ( <n, x_j - x_i> - s * <n, u_i - u_j> )
//
// and the largest s that keeps g at or above a level is available in closed
// form: the crossing time of each pair whose projected separation is closing,
// minimized over pairs. Nothing is relaxed and no Lipschitz constant is taken,
// so within the chosen direction this is the exact answer rather than a bound.
//
// That is precisely where max_relative_u is loose. It answers "how fast could
// the distance shrink under the WORST direction", which for a contact sliding
// tangentially is the entire sliding speed while the two surfaces are barely
// approaching at all. The projected rate <n, u_i - u_j> is the component that
// actually closes the gap, so a tangential pair certifies a step limited only by
// its normal closing rate, and a pair with no closing corner certifies the whole
// remaining sweep in one probe. The gain is the ratio of the two rates: 1 for a
// head-on approach, 1/sin(angle) for a glancing one, unbounded for a pair that
// is sliding or separating.
//
// The certificate holds for ANY direction, which is what makes it safe to build
// one from the closest-point difference the distance functor already computed.
// An inexact closest point yields a valid direction, only a less useful one (g
// then sits strictly below d and the step is shorter), so unlike the parking and
// advance arithmetic around it this routine does not rest on distance.hpp being
// exact. With the true closest direction the supporting-hyperplane property of
// two convex sets gives <n, x_j - x_i> >= d for every pair while every projected
// rate is at most u_max, so the certified step is never shorter than the
// direction-agnostic one; the caller takes the larger of the two regardless, so
// float round-off cannot turn this into a regression.
//
// The direction arrives UNNORMALIZED as w, with w_norm an upper bound on its
// length. Dividing both projections by |w| scales numerator and denominator
// alike, so the length survives only in the level term, and the routine needs no
// reciprocal square root and no argument that its direction is exactly unit.
//
// err is the round-off allowance, and it has to be evaluated on the operands
// actually used rather than fixed once for the frame. A frame-wide constant
// sized to the worst case exceeds the parking clearance of a coarse collider
// (a contact against a ground plane built from two huge triangles parks at a
// clearance of order 1e-6 in this frame), which would switch the certificate off
// in exactly the geometry where the advance loop is slowest. Each term is a
// standard bound on the operation that produced it: 4u covers the difference of
// two columns plus a three-term dot product, the velocity term enters scaled by
// max_t because the certificate is only ever consumed for s <= max_t, and the
// last term is the absolute round-off that evaluating the sweep leaves in each
// coordinate. That last term is itself frame-wide, and at 6.2e-7 it is already a
// sizeable fraction of the coarse-collider clearance just quoted, so on the
// coarsest geometry the certificate can decline (projected - dip_w <= 0) and the
// caller falls back to the direction-agnostic step. That forfeits the speedup on
// exactly those pairs and nothing else: the fallback is the step this routine was
// added to improve on, and the caller takes the larger of the two either way.
// Charging the velocity error to the numerator instead of inflating
// the rate keeps a barely-positive rate from turning into an enormous quotient.
// Two levels are in play, and the split is what keeps the certificate both
// useful and behaviorally neutral.
//
// `dip` decides whether this direction resolves the pair AT ALL. It is the same
// floor the direction-agnostic step certifies, so asking for anything stricter
// here would void the certificate for a contact settled a hair above its parking
// distance, which is exactly the configuration it is worth the most (a settled
// contact slides, so its projected closing rate is what the direction-agnostic
// bound overstates worst).
//
// `park` decides where a BOUNDED step stops. Solving the crossing one level
// higher makes a finite step land with the pair still outside its parking
// distance, so it never ends the advance loop and the bisection bracket stays as
// narrow as the direction-agnostic loop would have made it. That matters because
// bisection only converges to SOME crossing inside its bracket, and a bracket
// widened by a long step can straddle a dip below the park and back out, landing
// the recovered toi on a different crossing than the one this pair actually
// reaches first. Nothing about penetration turns on which it picks (every state
// in the bracket is certified at `dip`), but the accepted step is the solver's
// time-advance rate, and it should not move for a reason unrelated to physics.
//
// A step that is UNBOUNDED needs neither: with no corner closing along the
// direction the projected gap cannot decrease, so it stays at its current value
// for the rest of the sweep, and that value is already above the park.
template <class T, unsigned R, unsigned C, unsigned SPLIT>
__device__ float directional_advance(const SMat<T, R, C> &x,
                                     const SMat<T, R, C> &dx,
                                     const SVec<float, R> &w, float w_norm,
                                     float dip, float park, float max_t) {
    // 4u, u = 2^-24 being the float32 unit round-off.
    const float round_slack = 2.0f * FLT_EPSILON;
    // 2 * sqrt(3) * 3u. rescale() leaves every coordinate of the sweep below
    // one and every displacement below two, so evaluating x0 + t * dx rounds a
    // coordinate by at most 3u in ABSOLUTE terms, and the certificate compares
    // two such evaluations, one at each end of the step it grants, over the
    // three coordinates.
    const float coord_slack = 6.2e-7f;
    SVec<float, R> w_abs = w.cwiseAbs();
    float dip_w = dip * w_norm;
    float park_w = park * w_norm;
    float step = std::numeric_limits<float>::max();
    for (int i = 0; i < SPLIT; i++) {
        for (int j = SPLIT; j < C; j++) {
            SVec<float, R> gap = (x.col(j) - x.col(i)).template cast<float>();
            SVec<float, R> rel = (dx.col(i) - dx.col(j)).template cast<float>();
            float err = round_slack * (w_abs.dot(gap.cwiseAbs()) +
                                       max_t * w_abs.dot(rel.cwiseAbs()) +
                                       park_w) +
                        coord_slack * w_norm;
            float projected = w.dot(gap) - err;
            if (!(projected - dip_w > 0.0f)) {
                // This direction cannot certify even the floor the caller
                // already holds, which happens when the closest-point difference
                // is a poor direction or the pair sits inside the round-off
                // allowance. Grant nothing and let the caller fall back.
                return 0.0f;
            }
            float rate = w.dot(rel);
            if (rate > 0.0f) {
                step = fminf(step, fmaxf(0.0f, projected - park_w) / rate);
            }
        }
    }
    return step;
}

// Floor on the gap at which the CCD parks a pair, as a fraction of that pair's
// contact gap. Expressed against ghat because the quantity it protects is the
// fp32 dynamic contact stiffness mass/gap^2, whose conditioning depends on the
// gap measured in units of the barrier's own activation distance, not on any
// absolute length. Callers rescale it into the sweep frame alongside offset and
// eps. See the parking-clearance derivation in ccd_helper.
__device__ float park_floor(float ghat) { return 1e-2f * ghat; }

template <typename F, typename T, unsigned R, unsigned C, unsigned SPLIT>
__device__ float ccd_helper(const SMat<T, R, C> &x0, const SMat<T, R, C> &dx,
                            float u_max, F square_dist_func, float offset,
                            float eps, float floor_gap, const ParamSet &param) {
    SMat<T, R, C> x = x0;
    SVec<float, R> w;
    float lower_t = 0.0f;
    float toi = 0.0f;
    float max_t = param.line_search_max_t;
    float d2 = square_dist_func(x0, w);
    float target_squared = offset * offset;
    // The CCD advances from a SEPARATED start: `offset` here is the RAW
    // contact offset (the callers do not inflate it), so this entry check
    // fires only when two surfaces genuinely begin the step touching or
    // overlapping, the one state the barrier and the conservative advance
    // cannot handle. Record it and return 0 (no advance) rather than trapping
    // the device: the host reads g_ccd_overlap after the line search and ends
    // the run with a structured OverlappingStart crash and an actionable
    // message.
    if (!(d2 > target_squared)) {
        g_ccd_overlap = 1u;
        g_ccd_overlap_d2 = d2;
        g_ccd_overlap_offset = offset;
        return 0.0f;
    }
    // Parking clearance with hysteresis. Three surfaces matter:
    //   fatal = offset                the entry check above
    //   park                          where the bisection stops
    //   dip   = park - overshoot      worst reachable mid-sweep state
    // The park must sit strictly above fatal, because the accepted state is
    // not always this pair's own parked point: the global line-search toi is
    // the MIN over all pairs, so another pair's clamp can commit THIS pair
    // anywhere along its sweep, including between two conservative-advance
    // probes, where the distance may dip below park by the probe overshoot.
    // The park clearance has an ABSOLUTE floor F = max(2*eps, floor_gap),
    // where floor_gap is park_floor(ghat) rescaled into this sweep's frame by
    // the caller, alongside offset and eps. The floor exists for the fp32
    // assembly downstream: the dynamic contact stiffness mass/gap^2 and the
    // preconditioner block inversion have only ever been exercised against
    // gaps a bounded ratio below the barrier's activation distance, and a
    // parked gap far under that drives mass/gap^2 orders of magnitude higher
    // and overflows the fp32 PCG into indefinite-looking garbage (pAp <= 0
    // trips with order-1e2..1e3 Rayleigh quotients, cg failures). ghat is the
    // right anchor because it is the length scale mass/gap^2 is measured
    // against, and it is authored per element, so the floor is both
    // scale-covariant and mesh-resolution independent.
    // The floor must NOT instead be a constant in the RESCALED frame, however
    // natural that looks from inside this function: rescale() normalizes every
    // pair by its own largest coordinate, so a literal in these units is a
    // floor proportional to the PRIMITIVE'S OWN SIZE. A collider built from a
    // few huge triangles then parks orders of magnitude further out than the
    // identical contact on a refined mesh, and once a pair sits at that
    // inflated park the advance degenerates to eps-scale steps and toi
    // collapses to zero: a rigid body landing on a two-triangle ground plane
    // failed ccd where the same scene on a subdivided plane completed.
    // Nor may the floor scale with the pair's own CLEARANCE (ghat is static,
    // so it does not): a park at any fixed FRACTION of the current clearance
    // lets a loaded contact ratchet
    // inward geometrically (halving per line search), which reaches the
    // fp32-garbage regime within one time step of ordinary settling. So:
    //   - a pair with clearance >= F parks AT F, never lower, and its
    //     overshoot F/2 bounds every mid-sweep state at F/2 above fatal, an
    //     absolute bound;
    //   - a pair already inside the band (clearance < F, reachable only
    //     through a mid-sweep commit or authored that close) parks at its own
    //     start clearance minus eps/2, an absolute per-step allowance that
    //     keeps its toi positive (tangential sliding stays possible) while a
    //     head-on approach costs eps/2 per line search, a rate that only a
    //     sustained crush strong enough to overcome the contact force can
    //     accumulate; its overshoot shrinks so its sweep never probes below
    //     the F/2 dip bound, reaching zero for a pair below F/2 (whose sweep
    //     then cannot dip below its own park at all).
    // In exact arithmetic every reachable state therefore keeps a clearance of
    // at least min(F/2, start clearance - eps/2), so no geometric decay
    // exists. That bound has a precondition, because an accepted step commits
    // as an absolute position: committing rounds each coordinate by up to half
    // an ulp of the largest coordinate in the scene, which costs a pair's
    // clearance up to 2*sqrt(3) times that, and the bound holds only while the
    // parking clearance stays above it. Anchoring F to ghat rather than to eps
    // is what keeps it there: at the default contact gap 1e-3 the dip bound F/2
    // is 5e-6, roughly three times the 1.7e-6 a scene of radius 10 can lose to
    // commit rounding, while eps/2 at the default ccd-eps of 1e-7 is 5e-8 and
    // is far below it. eps/2 is therefore a per-step allowance, not a clearance
    // guarantee. Geometry placed far enough from the origin narrows that
    // margin, so compare ghat against FLT_EPSILON times the scene radius
    // before authoring one. Parking earlier is conservative, so
    // penetration-free is unaffected.
    float clearance = sqrtf(d2) - offset;
    float floor_clear = fmaxf(2.0f * eps, floor_gap);
    float park = offset + fminf(floor_clear,
                                fmaxf(clearance - 0.5f * eps, 0.5f * clearance));
    float overshoot = fmaxf(0.0f, (park - offset) - 0.5f * floor_clear);
    // The worst state the sweep may reach between two probes, named once so
    // both certificates below are stated against the same level: the
    // direction-agnostic step and the directional one then guarantee exactly
    // the clearance this comment block argues for, and taking the larger of
    // the two leaves that guarantee unchanged.
    float dip = park - overshoot;
    // The park sits below the pair's start distance by construction, but for
    // an in-band pair the margin is only eps/2, which can fall below the
    // spacing of representable values near park^2 once eps is small relative
    // to the pair's separation (the callers hand eps in already divided by the
    // pair's own size, so a coarse primitive shrinks it); keep the squared park
    // strictly below the entry d2 so the sweep always starts outside it and
    // the bisection's overlap branch below stays unreachable for a legal
    // (separated) start.
    float park_squared = fminf(park * park, nextafterf(d2, 0.0f));
    float inv_u_max = 1.0f / u_max;

    // With a zero overshoot (a pair inside the parking band) the advance
    // steps can be eps-scale, so the sweep may not reach max_t or the park
    // crossing in a bounded number of probes; the same holds when a step
    // underflows at the crossing. Cap the probes and return the last verified
    // probe: it satisfied d2 > park^2, so truncating this pair's toi there is
    // conservative.
    const unsigned max_probes = 4096u;
    unsigned probes = 0;
    // The loop steps first and probes afterwards, because the evaluation above
    // IS the probe at toi = 0: T(0) * dx is exactly zero, so re-entering the
    // body there would recompute a bit-identical d2 and w. That first probe can
    // also never end the loop, since park_squared is clamped strictly below the
    // entry d2 just above.
    while (true) {
        if (!(d2 > park_squared)) {
            break;
        }
        lower_t = toi;
        float dist = sqrtf(d2);
        float step = (dist - dip) * inv_u_max;
        // Refine only while there is sweep left to certify. Once the
        // direction-agnostic step already covers the rest of it the directional
        // certificate cannot add anything, and this is the common case: a
        // broad-phase candidate that is nowhere near contact clears the whole
        // sweep on its first probe and pays nothing for what follows.
        if (toi + step <= max_t) {
            // sqrtf(fl(w.w)) underestimates |w| by at most 2.5 units of
            // round-off; inflating shrinks the certified step, so it is the
            // conservative direction to err in.
            step = fmaxf(step, directional_advance<T, R, C, SPLIT>(
                                   x, dx, w, dist * (1.0f + 2.0f * FLT_EPSILON),
                                   dip, park, max_t));
        }
        toi += step;
        if (toi > max_t) {
            return max_t;
        } else if (++probes > max_probes) {
            return lower_t;
        }
        x = x0 + T(toi) * dx;
        d2 = square_dist_func(x, w);
    }

    float upper_t = toi;
    float window = upper_t - lower_t;
    while (true) {
        toi = 0.5f * (upper_t + lower_t);
        x = x0 + T(toi) * dx;
        d2 = square_dist_func(x);
        if (d2 > park_squared) {
            lower_t = toi;
        } else {
            upper_t = toi;
        }
        float new_window = upper_t - lower_t;
        if (new_window == window) {
            break;
        } else {
            window = new_window;
        }
    }
    if (!(lower_t > 0.0f)) {
        // Same overlap condition surfacing from the bisection: the pair reaches
        // the contact offset at the very start of the step (surfaces
        // effectively touching). Record it and return 0, as above.
        g_ccd_overlap = 1u;
        g_ccd_overlap_d2 = d2;
        g_ccd_overlap_offset = offset;
        return 0.0f;
    }
    return lower_t;
}

// centerize() and rescale() have already put the sweep in a frame centered on
// the pair and normalized so the largest coordinate is just under one, so every
// subtraction below is between coordinates of comparable magnitude. Each
// functor is templated on the coordinate type T of that frame and on the
// accumulation type Y of the result: it reduces the two closest points to a
// DIFFERENCE in T and squares only that difference, so the squared distance is
// never assembled as a difference of two large squares.
//
// Each also hands that difference back through `dir`, which costs nothing since
// it is already in a register, and orients it the SAME way in all four: from the
// closest point on primitive A (columns [0, SPLIT)) toward the one on primitive
// B (columns [SPLIT, C)). ccd_helper's directional certificate reads the
// projected gap as <dir, x_j - x_i> with i in A and j in B, so a functor that
// handed back the opposite sign would report every gap negated and certify
// nothing. The one-argument form exists for the entry check and the bisection,
// which need only the distance.
template <typename T, typename Y> struct EdgeEdgeSquaredDist {
    __device__ float operator()(const Mat3x4<T> &x) {
        Vec3<Y> dir;
        return (*this)(x, dir);
    }
    __device__ float operator()(const Mat3x4<T> &x, Vec3<Y> &dir) {
        const Vec3<T> &p0 = x.col(0);
        const Vec3<T> &p1 = x.col(1);
        const Vec3<T> &q0 = x.col(2);
        const Vec3<T> &q1 = x.col(3);
        Vec4<T> c = distance::edge_edge_distance_coeff<T, Y>(
                        p0, p1, q0, q1)
                        .template cast<T>();
        Vec3<T> x0 = c[0] * p0 + c[1] * p1;
        Vec3<T> x1 = c[2] * q0 + c[3] * q1;
        dir = (x1 - x0).template cast<Y>();
        return dir.dot(dir);
    }
};

template <typename T, typename Y> struct PointEdgeSquaredDist {
    __device__ float operator()(const Mat3x3<T> &x) {
        Vec3<Y> dir;
        return (*this)(x, dir);
    }
    __device__ float operator()(const Mat3x3<T> &x, Vec3<Y> &dir) {
        const Vec3<T> &p = x.col(0);
        const Vec3<T> &q0 = x.col(1);
        const Vec3<T> &q1 = x.col(2);
        Vec2<T> c =
            distance::point_edge_distance_coeff_unclassified<T, Y>(p, q0, q1)
                .template cast<T>();
        Vec3<T> q = c[0] * q0 + c[1] * q1;
        // The edge is primitive B, so the difference runs toward it. Negating a
        // float is exact, so d2 is unchanged by the orientation.
        dir = (q - p).template cast<Y>();
        return dir.dot(dir);
    }
};

template <typename T, typename Y> struct PointPointSquaredDist {
    __device__ float operator()(const Mat3x2<T> &x) {
        Vec3<Y> dir;
        return (*this)(x, dir);
    }
    __device__ float operator()(const Mat3x2<T> &x, Vec3<Y> &dir) {
        const Vec3<T> &p = x.col(0);
        const Vec3<T> &q = x.col(1);
        dir = (q - p).template cast<Y>();
        return dir.dot(dir);
    }
};

template <typename T, typename Y> struct PointTriangleSquaredDist {
    __device__ float operator()(const Mat3x4<T> &x) {
        Vec3<Y> dir;
        return (*this)(x, dir);
    }
    __device__ float operator()(const Mat3x4<T> &x, Vec3<Y> &dir) {
        const Vec3<T> &p = x.col(0);
        const Vec3<T> &t0 = x.col(1);
        const Vec3<T> &t1 = x.col(2);
        const Vec3<T> &t2 = x.col(3);
        Vec3<T> c = distance::point_triangle_distance_coeff_unclassified<T, Y>(
                        p, t0, t1, t2)
                        .template cast<T>();
        // A weighted sum of endpoint differences from p, so the result depends
        // only on the triangle's shape relative to p. The weights sum to one,
        // so this is (closest point on the triangle) minus p, already running
        // from the point toward the triangle.
        auto y = c(0) * (t0 - p) + c(1) * (t1 - p) + c(2) * (t2 - p);
        dir = y.template cast<Y>();
        return dir.dot(dir);
    }
};

__device__ float point_triangle_ccd(const Vec3f &p0, const Vec3f &p1,
                                    const Vec3f &t00, const Vec3f &t01,
                                    const Vec3f &t02, const Vec3f &t10,
                                    const Vec3f &t11, const Vec3f &t12,
                                    float offset, float ghat,
                                    const ParamSet &param) {
    Vec3f dp = p1 - p0;
    Vec3f dt0 = t10 - t00;
    Vec3f dt1 = t11 - t01;
    Vec3f dt2 = t12 - t02;
    Mat3x4f x0;
    Mat3x4f dx;
    x0 << p0, t00, t01, t02;
    dx << dp, dt0, dt1, dt2;
    centerize<float, 3, 4>(x0);
    centerize<float, 3, 4>(dx);
    float scale = rescale<float, 3, 4>(x0, dx, param.line_search_max_t);
    // rescale has already reported a collapsed frame; stop here so the sweep
    // never proceeds on an unscaled frame whose lengths all scaled to zero.
    if (!(scale > 0.0f)) {
        return 0.0f;
    }
    // point = col 0, triangle = cols 1..3 -> SPLIT = 1
    float u_max = max_relative_u<float, 3, 4, 1>(dx);
    if (u_max) {
        PointTriangleSquaredDist<_coord_, float> dist_func;
        return ccd_helper<PointTriangleSquaredDist<_coord_, float>, _coord_, 3,
                          4, 1>(x0, dx, u_max, dist_func, scale * offset,
                                scale * param.ccd_eps,
                                scale * park_floor(ghat), param);
    } else {
        return param.line_search_max_t;
    }
}

__device__ float point_edge_ccd(const Vec3f &p0, const Vec3f &p1,
                                const Vec3f &e00, const Vec3f &e01,
                                const Vec3f &e10, const Vec3f &e11,
                                float offset, float ghat,
                                const ParamSet &param) {
    Vec3f dp = p1 - p0;
    Vec3f dt0 = e10 - e00;
    Vec3f dt1 = e11 - e01;
    Mat3x3f x0;
    Mat3x3f dx;
    x0 << p0, e00, e01;
    dx << dp, dt0, dt1;
    centerize<float, 3, 3>(x0);
    centerize<float, 3, 3>(dx);
    float scale = rescale<float, 3, 3>(x0, dx, param.line_search_max_t);
    // rescale has already reported a collapsed frame; stop here so the sweep
    // never proceeds on an unscaled frame whose lengths all scaled to zero.
    if (!(scale > 0.0f)) {
        return 0.0f;
    }
    // point = col 0, edge = cols 1..2 -> SPLIT = 1
    float u_max = max_relative_u<float, 3, 3, 1>(dx);
    if (u_max) {
        PointEdgeSquaredDist<_coord_, float> dist_func;
        return ccd_helper<PointEdgeSquaredDist<_coord_, float>, _coord_, 3, 3,
                          1>(x0, dx, u_max, dist_func, scale * offset,
                             scale * param.ccd_eps, scale * park_floor(ghat),
                             param);
    } else {
        return param.line_search_max_t;
    }
}

__device__ float point_point_ccd(const Vec3f &p0, const Vec3f &p1,
                                 const Vec3f &q0, const Vec3f &q1, float offset,
                                 float ghat, const ParamSet &param) {
    Vec3f dp = p1 - p0;
    Vec3f dq = q1 - q0;
    Mat3x2f x0;
    Mat3x2f dx;
    x0 << p0, q0;
    dx << dp, dq;
    centerize<float, 3, 2>(x0);
    centerize<float, 3, 2>(dx);
    float scale = rescale<float, 3, 2>(x0, dx, param.line_search_max_t);
    // rescale has already reported a collapsed frame; stop here so the sweep
    // never proceeds on an unscaled frame whose lengths all scaled to zero.
    if (!(scale > 0.0f)) {
        return 0.0f;
    }
    // point = col 0, point = col 1 -> SPLIT = 1 (only the single inter pair)
    float u_max = max_relative_u<float, 3, 2, 1>(dx);
    if (u_max) {
        PointPointSquaredDist<_coord_, float> dist_func;
        return ccd_helper<PointPointSquaredDist<_coord_, float>, _coord_, 3, 2,
                          1>(x0, dx, u_max, dist_func, scale * offset,
                             scale * param.ccd_eps, scale * park_floor(ghat),
                             param);
    } else {
        return param.line_search_max_t;
    }
}

__device__ float edge_edge_ccd(const Vec3f &ea00, const Vec3f &ea01,
                               const Vec3f &eb00, const Vec3f &eb01,
                               const Vec3f &ea10, const Vec3f &ea11,
                               const Vec3f &eb10, const Vec3f &eb11,
                               float offset, float ghat,
                               const ParamSet &param) {
    Vec3f dea0 = ea10 - ea00;
    Vec3f dea1 = ea11 - ea01;
    Vec3f deb0 = eb10 - eb00;
    Vec3f deb1 = eb11 - eb01;
    Mat3x4f x0;
    Mat3x4f dx;
    x0 << ea00, ea01, eb00, eb01;
    dx << dea0, dea1, deb0, deb1;
    centerize<float, 3, 4>(x0);
    centerize<float, 3, 4>(dx);
    float scale = rescale<float, 3, 4>(x0, dx, param.line_search_max_t);
    // rescale has already reported a collapsed frame; stop here so the sweep
    // never proceeds on an unscaled frame whose lengths all scaled to zero.
    if (!(scale > 0.0f)) {
        return 0.0f;
    }
    // edge A = cols 0..1, edge B = cols 2..3 -> SPLIT = 2
    float u_max = max_relative_u<float, 3, 4, 2>(dx);
    if (u_max) {
        EdgeEdgeSquaredDist<_coord_, float> dist_func;
        return ccd_helper<EdgeEdgeSquaredDist<_coord_, float>, _coord_, 3, 4,
                          2>(x0, dx, u_max, dist_func, scale * offset,
                             scale * param.ccd_eps, scale * park_floor(ghat),
                             param);
    } else {
        return param.line_search_max_t;
    }
}

} // namespace accd

#endif
