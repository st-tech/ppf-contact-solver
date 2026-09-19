// File: park.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef PARK_HPP
#define PARK_HPP

// WHERE A SWEEP STOPS SHORT OF THE SURFACE IT IS SWEEPING AGAINST, and why any
// sweep must. The assembly downstream forms the elasticity-inclusive dynamic
// contact stiffness mass/gap^2 from the clearance the line search leaves
// behind, so a sweep that stops exactly ON the surface hands that expression a
// zero divisor. Parking is what makes a zero clearance unreachable, which is
// the only fix available: the divisor cannot be capped without weakening the
// barrier, and the assembly's own `gap >= 0` guard cannot be tightened to
// `> 0`, because a body resting at the surface is a legal state under the
// non-penetration guarantee and a `> 0` guard would refuse a correct scene.
//
// This header is separate from `accd.hpp` because the parking rule is needed by
// the analytic sweeps as well as the mesh ones, and everything here is `inline`
// and may be included anywhere. `accd.hpp` is include-safe on the same terms:
// its four `*_ccd` entry points carry `inline` too, which is what lets a
// generated entry point include a body that reaches them. A bare `__device__`
// there would be EXTERNAL linkage under device LTO and an ordinary definition
// on the host, so the second including translation unit would be a duplicate
// symbol at link; do not take the `inline` back off either header.

#ifndef SM_MAX
#define SM_MAX fmaxf
#define PARK_UNDEF_MAX
#endif
#ifndef SM_MIN
#define SM_MIN fminf
#define PARK_UNDEF_MIN
#endif
#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define PARK_UNDEF_DIV
#endif

namespace accd {

// Floor on the gap at which the CCD parks a pair, as a fraction of that pair's
// contact gap. Expressed against ghat because the quantity it protects is the
// fp32 dynamic contact stiffness mass/gap^2, whose conditioning depends on the
// gap measured in units of the barrier's own activation distance, not on any
// absolute length. Callers rescale it into the sweep frame alongside offset and
// eps. See the parking-clearance derivation in ccd_helper.
__device__ inline float park_floor(float ghat) { return 1e-2f * ghat; }

// Where an ANALYTIC sweep parks, and how much of that sweep reaches it.
//
// `ccd_helper` above bisects a sweep against a mesh primitive. A vertex against
// a sphere, a plane or a pin's ghat ball needs no bisection at all, because the
// clearance is a known function of one parameter and the crossing is solved in
// closed form. What the two paths share is the REASON for parking: the assembly
// downstream forms the elasticity-inclusive dynamic stiffness mass/gap^2 from
// the same clearance the sweep leaves behind, so a sweep that stops a vertex
// exactly ON the surface hands that expression a zero divisor. Solving for the
// surface itself would meet that divisor, because `clearance == 0` is an
// attainable value rather than a measure-zero coincidence: against an
// axis-aligned plane the clearance is the difference of two coordinates along
// one axis, and the subtraction of two nearby floats is exact, so a vertex that
// reaches the plane's own coordinate reports exactly zero.
//
// The parked clearance is `ccd_helper`'s, with the two terms that belong to a
// bisection dropped: with no intermediate probes there is no overshoot to bound
// and so no dip below the park to allow for. The remaining two-regime rule
// carries the same guarantees it does there:
//   - a vertex with clearance at or above park_floor(ghat) parks AT
//     park_floor(ghat) and never lower, so no geometric ratchet toward zero
//     exists;
//   - a vertex already inside that band parks at its own start clearance less
//     eps/2, an absolute per-step allowance that keeps the returned fraction
//     strictly positive, so tangential sliding stays possible while a head-on
//     approach spends eps/2 per line search instead of the whole clearance at
//     once.
// Parking short of the surface is conservative, so non-penetration is
// unaffected: the committed clearance only grows.
//
// The anchor is `park_floor(ghat)` rather than the `constraint-tol` parameter,
// which the assembly already multiplies by ghat to floor the gap of a KINEMATIC
// collider. The two are different mechanisms and only one of them is available
// here. A kinematic collider's pose is prescribed, so nothing may be parked and
// clamping the reported VALUE is all that can be done; a non-kinematic one can
// be stopped short, which leaves the reported gap equal to the true gap. Making
// the floor a tunable would also let a scene set it to zero and put the divisor
// back, and what it protects is an fp32 conditioning invariant rather than a
// preference.
//
// A sweep that STARTS at or inside the surface parks at the surface, which
// reproduces the unparked arithmetic exactly, bit for bit. That state is not
// this function's to resolve: a vertex already touching cannot be given
// clearance by taking a smaller step, and the caller's own assert on a positive
// time of impact is what refuses it.
__device__ inline float park_gap_analytic(float clearance0, float ghat,
                                          float eps) {
    if (!(clearance0 > 0.0f)) {
        return 0.0f;
    }
    float floor_clear = SM_MAX(2.0f * eps, park_floor(ghat));
    // `clearance0 - min(a, clearance0/2)` is `max(clearance0 - a, clearance0/2)`
    // rearranged, and in float it is the same value bit for bit, because
    // `clearance0 - clearance0/2` is exact. It is written this way so the
    // allowance is visible as one quantity that can be floored.
    //
    // THE ALLOWANCE MUST BE POSITIVE OR THE PARK IS THE START CLEARANCE, which
    // returns a zero-length step and stalls the line search instead of
    // advancing it. `ccd_eps` is 1e-7 by default and every production sweep has
    // it, but a zeroed `ParamSet` does not and neither does a scene that
    // authors it to zero, so the halving is the fallback rather than a
    // parameter's good behavior. Halving alone is NOT the rule for a live eps:
    // a fixed fraction lets a loaded contact ratchet inward geometrically,
    // which is the failure `ccd_helper`'s own derivation rejects it for.
    float allowance = (eps > 0.0f) ? (0.5f * eps) : (0.5f * clearance0);
    return SM_MIN(floor_clear,
                  clearance0 - SM_MIN(allowance, 0.5f * clearance0));
}

// The fraction of a sweep whose clearance runs affinely from `clearance0` to
// `clearance1` at which it reaches `park`. Callers scale it by
// `line_search_max_t` exactly as they scaled the unparked surface crossing, and
// call it only once they have found `clearance1 < park`.
//
// At `park == 0` this reduces to the plain surface crossing, spelled the same
// way: `(c0 - 0) / (c0 - c1)` is `-h0 / (h1 - h0)`.
__device__ inline float park_crossing_analytic(float clearance0,
                                               float clearance1, float park) {
    return SM_DIV(clearance0 - park, clearance0 - clearance1);
}

} // namespace accd

#ifdef PARK_UNDEF_MAX
#undef SM_MAX
#undef PARK_UNDEF_MAX
#endif
#ifdef PARK_UNDEF_MIN
#undef SM_MIN
#undef PARK_UNDEF_MIN
#endif
#ifdef PARK_UNDEF_DIV
#undef SM_DIV
#undef PARK_UNDEF_DIV
#endif

#endif // PARK_HPP
