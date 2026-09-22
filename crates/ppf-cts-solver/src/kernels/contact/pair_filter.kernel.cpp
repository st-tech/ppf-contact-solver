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
// No include of its own. `Vec2u`, the three element property records and
// `isect::intersection_tolerated` arrive from whatever declares them for the
// backend that is compiling, which is data.hpp under nvcc and on the host (it
// includes contact/intersect_policy.hpp itself) and the spliced segments under
// MSL.
//
// WHICH PAIRS A PASS MAY ACT ON, stated once for every element pairing and
// every backend. Three passes read it, and each pair of them has to agree:
//
//   - the barrier ASSEMBLY and the CCD SWEEP must admit exactly the SAME set. A
//     pair the barrier assembles and the sweep does not filter is bounded by
//     nothing, which is a penetration; the reverse only costs step length.
//   - the INTERSECTION SCAN must report only pairs contact acts on. A pair
//     contact ignores can pass through its partner by design, so reporting it
//     would abort the run the user asked for.
//   - the host scene-build check and the solver's own `check_intersection` must
//     grant exactly the SAME set: building at one gate and aborting at the
//     other is a broken feature, and the reverse is a silent tolerance.
//
// That is only checkable while the rule has one statement, which is what this
// file is: `intersect_pair_reported` is `contact_pair_admitted` narrowed by one
// further condition, so the scan's set is a subset of contact's by
// construction rather than by review.

// One side of a candidate pair, as the facts the verdicts read. A plain
// aggregate, because the callers derive its fields from different structs: an
// edge reads `EdgeProp`, a face reads `FaceProp`, and a vertex reads
// `VertexProp`, where "prescribed" is spelled `fix_index != 0` rather than
// `fixed`. The three `pair_side_of_*` constructors below are the whole
// derivation, so every pass describes an element the same way.
struct PairSide {
    // Per-vertex object and group identity and the allowance bits, all read
    // off the element's FIRST vertex, since the allowance parameters are per
    // object.
    unsigned object_index;
    unsigned group_index;
    unsigned char intersect_policy;
    // Nonzero for a vertex inside a PDRD rigid body; 0 for everything else.
    unsigned pdrd_body_index;
    float mass;
    // Prescribed: an element no solve can move.
    bool fixed;
    // A driven collider, whose shape is authored rather than solved.
    bool collider;
    // Every pin covering this element asked for its intersections to be
    // allowed (the build-time unanimity latch, not a live pin state).
    bool pin_allow_intersection;
};

// A face or an edge is described by its FIRST vertex plus the element's own
// three facts. The anchor is a thread-space copy the caller made: the MSL
// rendering refuses to bind a `device` lvalue to a thread reference.
[[seam::device_fn]] inline PairSide pair_side_of_face(
    const VertexProp &anchor,
    const FaceProp &prop) {
    PairSide side;
    side.object_index = anchor.object_index;
    side.group_index = anchor.group_index;
    side.intersect_policy = anchor.intersect_policy;
    side.pdrd_body_index = anchor.pdrd_body_index;
    side.mass = prop.mass;
    side.fixed = prop.fixed;
    side.collider = anchor.collider;
    side.pin_allow_intersection = prop.pin_allow_intersection;
    return side;
}

[[seam::device_fn]] inline PairSide pair_side_of_edge(
    const VertexProp &anchor,
    const EdgeProp &prop) {
    PairSide side;
    side.object_index = anchor.object_index;
    side.group_index = anchor.group_index;
    side.intersect_policy = anchor.intersect_policy;
    side.pdrd_body_index = anchor.pdrd_body_index;
    side.mass = prop.mass;
    side.fixed = prop.fixed;
    side.collider = anchor.collider;
    side.pin_allow_intersection = prop.pin_allow_intersection;
    return side;
}

// A VERTEX IS ITS OWN ELEMENT, so "prescribed" is `fix_index != 0` rather than a
// `fixed` flag and the all-vertices-pinned bit is its own.
[[seam::device_fn]] inline PairSide pair_side_of_vertex(
    const VertexProp &prop) {
    PairSide side;
    side.object_index = prop.object_index;
    side.group_index = prop.group_index;
    side.intersect_policy = prop.intersect_policy;
    side.pdrd_body_index = prop.pdrd_body_index;
    side.mass = prop.mass;
    side.fixed = prop.fix_index != 0u;
    side.collider = prop.collider;
    side.pin_allow_intersection = prop.pin_allow_intersection;
    return side;
}

// Whether the user allowed these two elements to intersect: the issue #138
// allowances (`allow-self-intersection`, `allow-inter-object-intersection`,
// `allow-inter-group-intersection` and a pin's `allow_intersection`). AN
// ALLOWED PAIR IS NOT A CONTACT PAIR. No pass acts on it: the barrier
// assembles nothing for it, the CCD line search does not filter the step
// against it, and the intersection scan does not report it, so the two
// elements pass through each other freely. The penetration-free guarantee
// covers every pair the user did not allow.
[[seam::device_fn]] inline bool pair_intersection_allowed(const PairSide &a,
                                                          const PairSide &b) {
    return isect::intersection_tolerated(
        a.object_index, a.group_index, a.intersect_policy, b.object_index,
        b.group_index, b.intersect_policy, a.pin_allow_intersection,
        b.pin_allow_intersection);
}

// The same question for a dynamic element against the static collision mesh.
// The collider side carries no object or group identity and no policy of its
// own, so it is handed `NO_OBJECT_INDEX`, `NO_GROUP_INDEX` and an empty
// policy, under which "either side opts in" reduces to "the dynamic side opted
// in": its object allows inter-object or inter-group intersections (the
// collision mesh is always another group), or its pins allow them. The
// collision-mesh contact visitors, their CCD sweeps and the collision-mesh
// intersection scan all ask this one function.
[[seam::device_fn]] inline bool
collider_intersection_allowed(const PairSide &dynamic) {
    return isect::intersection_tolerated(
        dynamic.object_index, dynamic.group_index, dynamic.intersect_policy,
        NO_OBJECT_INDEX, NO_GROUP_INDEX, 0u, dynamic.pin_allow_intersection,
        false);
}

// Whether contact may act on these two elements. The four exclusions are the
// ones every CONTACT and CCD visitor applies:
//
//   `either_dyn`      at least one side is free to move. A pair with nothing
//                     free cannot be resolved by either side yielding.
//   `same_pdrd_body`  a rigid body never deforms, so two elements of one body
//                     do not move against each other.
//   `both_collider`   a collider's shape is authored and driven, so a pair of
//                     collider elements cannot be relieved by either side
//                     yielding. Rigged colliders also ship self-tangled
//                     (layered eye and mouth geometry, an arm inside a torso).
//                     Excluded whether the two sides are one collider or two.
//   `allowed`         the user allowed this pair to intersect, see
//                     `pair_intersection_allowed`.
//
// `contact_narrow.kernel.cpp`'s visitors and `ccd_sweep.kernel.cpp`'s both call
// this, so the assembly and the sweep agree by construction rather than by
// review.
//
// The verdict is symmetric under exchanging the two sides. `same_pdrd_body`
// looks asymmetric and is not: when the two indices are equal, one is nonzero
// exactly when the other is, and when they differ the test is false either way.
[[seam::device_fn]] inline bool contact_pair_admitted(const PairSide &a,
                                                      const PairSide &b) {
    const bool either_dyn = a.fixed == false || b.fixed == false;
    const bool same_pdrd_body =
        a.pdrd_body_index != 0 && a.pdrd_body_index == b.pdrd_body_index;
    const bool both_collider = a.collider && b.collider;
    const bool allowed = pair_intersection_allowed(a, b);
    return either_dyn && !same_pdrd_body && !both_collider && !allowed;
}

// Whether an intersection between these two elements is worth reporting: every
// pair contact acts on, less one case.
//
//   `either_nonzero`  at least one side has mass. Two zero-mass elements (two
//                     static solids, two pin-shell vertices) never intersect.
//
// The `either_dyn` half of `contact_pair_admitted` carries more weight here
// than in contact. An intersection between two fully PRESCRIBED elements
// cannot be relieved by either side yielding, so reporting it only aborts a run
// over geometry the user authored. The canonical case is a fully-pinned
// kinematic body whose own animation self-intersects, an armpit or a crotch:
// `examples/fitting` pins an entire dancing body mesh, and ungating this aborts
// it at initialize. The same holds for a pair of the same PDRD body, a fixed,
// physically meaningless self-intersection that must be tolerated even when the
// body starts self-tangled.
//
// None of this re-opens a silent penetration. A fix pin is an exact Dirichlet
// boundary condition, so contact cannot shove a pinned patch into a collider,
// and a pin PRESCRIBED into one fails loudly with zero penetration.
[[seam::device_fn]] inline bool
intersect_pair_reported(const PairSide &a,
                            const PairSide &b) {
    const bool either_nonzero = a.mass > 0.0f || b.mass > 0.0f;
    return either_nonzero && contact_pair_admitted(a, b);
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
