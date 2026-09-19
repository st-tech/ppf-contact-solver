// File: pair_filter.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::thread]]` is the address space MSL requires on every
// reference parameter.
//
// No include of its own. `Vec2u` and `isect::intersection_tolerated` arrive
// from whatever declares them for the backend that is compiling, which is
// data.hpp under nvcc and on the host (it includes contact/intersect_policy.hpp
// itself) and the spliced segments under MSL.
//
// WHICH PAIRS THE INTERSECTION SCAN REPORTS, stated once for every element
// pairing and every backend. The host scene-build check and the solver's own
// `check_intersection` must grant exactly the SAME set: building at one gate
// and aborting at the other is a broken feature, and the reverse is a silent
// tolerance. That is only checkable while the rule has one statement, which is
// what this file is.

// One side of a candidate intersecting pair, as the five facts the verdict
// reads. A plain aggregate, because the callers derive its fields from
// different structs: the edge-edge tester reads `EdgeProp`, the face-edge
// tester reads `FaceProp` on one side and `EdgeProp` on the other, and the
// point-point tester reads `VertexProp`, where "prescribed" is spelled
// `fix_index != 0` rather than `fixed`. The DERIVATION stays at the call site
// and the COMPOSITION lives here, which is the half that has to agree.
struct IntersectSide {
    // Per-vertex object identity and its two allowance bits, both read off the
    // element's FIRST vertex, since the allowance parameters are per object.
    unsigned object_index;
    unsigned char intersect_policy;
    // Nonzero for a vertex inside a PDRD rigid body; 0 for everything else.
    unsigned pdrd_body_index;
    float mass;
    // Prescribed: an element no solve can move.
    bool fixed;
    // A driven collider, whose shape is authored rather than solved.
    bool collider;
    // Every pin covering this element asked for its intersections to be
    // tolerated (the build-time unanimity latch, not a live pin state).
    bool pin_allow_intersection;
};

// Whether an intersection between these two elements is worth reporting.
//
// Five conditions, and each one suppresses a case the solver could not resolve
// even if it were told about it:
//
//   `either_dyn`      at least one side is free to move. An intersection
//                     between two fully PRESCRIBED elements cannot be relieved
//                     by either side yielding, so reporting it only aborts a
//                     run over geometry the user authored. The canonical case
//                     is a fully-pinned kinematic body whose own animation
//                     self-intersects, an armpit or a crotch:
//                     `examples/fitting` pins an entire dancing body mesh, and
//                     ungating this aborts it at initialize.
//   `either_nonzero`  at least one side has mass. Two zero-mass elements (two
//                     static solids, two pin-shell vertices) never intersect.
//   `same_pdrd_body`  a rigid body never deforms, so an element of one piercing
//                     another element of the SAME body is a fixed, physically
//                     meaningless self-intersection that must be tolerated even
//                     when the body starts self-tangled.
//   `both_collider`   a collider's shape is authored and driven, so a pair of
//                     collider elements cannot be relieved by either side
//                     yielding. Rigged colliders also ship self-tangled
//                     (layered eye and mouth geometry, an arm inside a torso).
//                     Excluded whether the two sides are one collider or two.
//   `tolerated`       the issue #138 allowances, which suppress REPORTING and
//                     nothing else: contact, CCD and the line search never
//                     consult them, so the penetration-free guarantee holds for
//                     every pair, named or not.
//
// None of this re-opens a silent penetration. A fix pin is an exact Dirichlet
// boundary condition, so contact cannot shove a pinned patch into a collider,
// and a pin PRESCRIBED into one fails loudly with zero penetration.
//
// The verdict is symmetric under exchanging the two sides. `same_pdrd_body`
// looks asymmetric and is not: when the two indices are equal, one is nonzero
// exactly when the other is, and when they differ the test is false either way.
[[seam::device_fn]] inline bool
intersect_pair_reported(const IntersectSide &a,
                            const IntersectSide &b) {
    const bool either_dyn = a.fixed == false || b.fixed == false;
    const bool either_nonzero = a.mass > 0.0f || b.mass > 0.0f;
    const bool same_pdrd_body =
        a.pdrd_body_index != 0 && a.pdrd_body_index == b.pdrd_body_index;
    const bool both_collider = a.collider && b.collider;
    const bool tolerated = isect::intersection_tolerated(
        a.object_index, a.intersect_policy, b.object_index, b.intersect_policy,
        a.pin_allow_intersection, b.pin_allow_intersection);
    return either_dyn && either_nonzero && !same_pdrd_body && !both_collider &&
           !tolerated;
}

// A vertex is free when it carries no fix pin. The three exclusions are the
// ones every CONTACT and CCD visitor applies, in this order: a pair with
// nothing free cannot be resolved by either side, two vertices of one rigid
// body do not deform against each other, and two colliders are excluded because
// a rigged collider ships self-tangled.
//
// IT SITS BESIDE THE INTERSECTION RULE BECAUSE IT IS THE SAME KIND OF
// STATEMENT: which pairs a pass may act on. The assembly and the CCD sweep must
// grant exactly the SAME set, and the asymmetry is what makes that a rule
// rather than tidiness: a pair the barrier assembles and the sweep does not
// filter is bounded by nothing, which is a penetration, while the reverse only
// costs step length. `contact_narrow.kernel.cpp`'s visitors and
// `ccd_sweep.kernel.cpp`'s both call this, so the two sets agree by
// construction rather than by review.
[[seam::device_fn]] inline bool contact_pair_admitted(bool either_dyn,
                                                          unsigned pdrd_a,
                                                          unsigned pdrd_b,
                                                          bool collider_a,
                                                          bool collider_b) {
    const bool same_pdrd_body = pdrd_a != 0u && pdrd_a == pdrd_b;
    const bool both_collider = collider_a && collider_b;
    return either_dyn && !same_pdrd_body && !both_collider;
}

// Whether two edges name a vertex in common. Four integer comparisons, so the
// answer cannot depend on rounding, on an address space or on a backend.
//
// The two contact paths it gates are not the same: the edge-edge CONTACT embed
// skips a sharing pair outright (its closest pair is identically zero and says
// nothing about proximity), while the intersection tester takes a different
// measurement for one, the two point-edge distances from the free endpoints.
[[seam::device_fn]] inline bool
edge_has_shared_vert(const Vec2u &e0,
                         const Vec2u &e1) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (e0[i] == e1[j])
                return true;
        }
    }
    return false;
}
