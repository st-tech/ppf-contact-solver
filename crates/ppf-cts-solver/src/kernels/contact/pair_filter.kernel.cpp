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
// No include of its own. `Vec2u`, `Vec3u`, the three element property records,
// `START_LINK_COLLISION_VERTEX` and `isect::intersection_tolerated` arrive from
// whatever declares them for the
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
//
// ALLOW EXISTING INTERSECTIONS JOINS THE RULE HERE AND NOWHERE ELSE. The
// scene-build check links, vertex by vertex, the pairs a scene STARTS tangled
// with in the groups that opted in (`ppf-cts-core/src/kernels/start_links.rs`),
// and `pair_linked_at_start` reads that table. It is part of
// `contact_pair_admitted` and of `collider_pair_admitted`, both of which take
// the table as parameters, so a pass that forgets it does not compile rather
// than silently admitting a linked pair on one pass and not another.

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
    // The element's own vertices, which is what the start links are keyed
    // on. A collision-mesh element carries its indices with
    // START_LINK_COLLISION_VERTEX set, which is how the link table names that
    // pool; a dynamic index never has that bit, which `builder.rs` asserts.
    unsigned vert[3];
    unsigned vert_count;
};

// A face or an edge is described by its FIRST vertex plus the element's own
// three facts. The anchor is a thread-space copy the caller made: the MSL
// rendering refuses to bind a `device` lvalue to a thread reference.
[[seam::device_fn]] inline PairSide pair_side_of_face(
    const VertexProp &anchor,
    const FaceProp &prop,
    const Vec3u &face) {
    PairSide side;
    side.vert[0] = face[0];
    side.vert[1] = face[1];
    side.vert[2] = face[2];
    side.vert_count = 3u;
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
    const EdgeProp &prop,
    const Vec2u &edge) {
    PairSide side;
    side.vert[0] = edge[0];
    side.vert[1] = edge[1];
    side.vert[2] = 0u;
    side.vert_count = 2u;
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
    const VertexProp &prop,
    unsigned index) {
    PairSide side;
    side.vert[0] = index;
    side.vert[1] = 0u;
    side.vert[2] = 0u;
    side.vert_count = 1u;
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

// A COLLISION-MESH element, for the one question its side answers: whether a
// start link reaches it. The collision mesh carries no object or group
// identity, no policy and no pin of its own (`collider_intersection_allowed`
// below is stated from the dynamic side alone), so only the vertices are
// meaningful and they carry START_LINK_COLLISION_VERTEX. The remaining fields
// describe what the collision mesh is: driven, massless and prescribed.
[[seam::device_fn]] inline PairSide pair_side_of_collision_element(
    unsigned v0, unsigned v1, unsigned v2, unsigned count) {
    PairSide side;
    side.object_index = NO_OBJECT_INDEX;
    side.group_index = NO_GROUP_INDEX;
    side.intersect_policy = 0u;
    side.pdrd_body_index = 0u;
    side.mass = 0.0f;
    side.fixed = true;
    side.collider = true;
    side.pin_allow_intersection = false;
    side.vert[0] = v0 | START_LINK_COLLISION_VERTEX;
    side.vert[1] = v1 | START_LINK_COLLISION_VERTEX;
    side.vert[2] = v2 | START_LINK_COLLISION_VERTEX;
    side.vert_count = count;
    return side;
}

[[seam::device_fn]] inline PairSide pair_side_of_collision_face(
    const Vec3u &face) {
    return pair_side_of_collision_element(face[0], face[1], face[2], 3u);
}

[[seam::device_fn]] inline PairSide pair_side_of_collision_edge(
    const Vec2u &edge) {
    return pair_side_of_collision_element(edge[0], edge[1], 0u, 2u);
}

[[seam::device_fn]] inline PairSide pair_side_of_collision_vertex(
    unsigned index) {
    return pair_side_of_collision_element(index, 0u, 0u, 1u);
}

// Whether a vertex of `from` is linked to a vertex of `to`, reading `from`'s
// rows. Every row is short (a vertex is linked only to the elements it started
// tangled with), and an element has at most three vertices, so a plain scan is
// the whole cost; the row is sorted, which a search could use and this does
// not need.
[[seam::device_fn]] inline bool side_linked_to(
    const PairSide &from,
    const PairSide &to,
    const unsigned *start_link_index,
    const unsigned *start_link_offset) {
    for (unsigned i = 0; i < from.vert_count; ++i) {
        const unsigned v = from.vert[i];
        const unsigned end = start_link_offset[v + 1u];
        for (unsigned j = start_link_offset[v]; j < end; ++j) {
            const unsigned linked = start_link_index[j];
            for (unsigned k = 0; k < to.vert_count; ++k) {
                if (linked == to.vert[k]) {
                    return true;
                }
            }
        }
    }
    return false;
}

// ALLOW EXISTING INTERSECTIONS: whether these two elements are linked at start,
// that is, whether ANY vertex of one is linked to ANY vertex of the other.
//
// A LINKED PAIR IS A NEIGHBOR, exactly as two elements sharing a vertex are,
// and no pass acts on it: no barrier, no CCD filter, no intersection report.
// The links are the pairs the scene-build check found intersecting, or closer
// than their contact offsets, in a pair an opted-in object belongs to; the
// vertex granularity extends the exemption one ring on each side, so a fold
// can slide a little before it meets a pair with full contact. The table is
// fixed for the run and the solver never adds to it.
//
// ONLY A DYNAMIC VERTEX HAS A ROW. A dynamic-dynamic link is stored in both
// rows, so either side answers and `a`'s rows are read; a collision-mesh
// element has no row, so it is looked up from its dynamic partner's. Every
// element's vertices are in one pool, so the first vertex says which.
//
// `has_start_link` zero is every scene that does not use the option, and it
// returns before either array is read: the table is then a zero-length
// allocation the generated entry resolves but nothing indexes.
[[seam::device_fn]] inline bool pair_linked_at_start(
    const PairSide &a,
    const PairSide &b,
    const unsigned *start_link_index,
    const unsigned *start_link_offset,
    unsigned has_start_link) {
    if (has_start_link == 0u) {
        return false;
    }
    if ((a.vert[0] & START_LINK_COLLISION_VERTEX) != 0u) {
        return side_linked_to(b, a, start_link_index, start_link_offset);
    }
    return side_linked_to(a, b, start_link_index, start_link_offset);
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
//   `linked`          the pair started tangled in a group that opted into
//                     Allow Existing Intersections, see
//                     `pair_linked_at_start`.
//
// `contact_narrow.kernel.cpp`'s visitors and `ccd_sweep.kernel.cpp`'s both call
// this, so the assembly and the sweep agree by construction rather than by
// review.
//
// The verdict is symmetric under exchanging the two sides. `same_pdrd_body`
// looks asymmetric and is not: when the two indices are equal, one is nonzero
// exactly when the other is, and when they differ the test is false either way.
[[seam::device_fn]] inline bool contact_pair_admitted(
    const PairSide &a,
    const PairSide &b,
    const unsigned *start_link_index,
    const unsigned *start_link_offset,
    unsigned has_start_link) {
    const bool either_dyn = a.fixed == false || b.fixed == false;
    const bool same_pdrd_body =
        a.pdrd_body_index != 0 && a.pdrd_body_index == b.pdrd_body_index;
    const bool both_collider = a.collider && b.collider;
    const bool allowed = pair_intersection_allowed(a, b);
    if (!either_dyn || same_pdrd_body || both_collider || allowed) {
        return false;
    }
    return !pair_linked_at_start(a, b, start_link_index, start_link_offset,
                                 has_start_link);
}

// The same question for a dynamic element against a collision-mesh element:
// the allowance (settled from the dynamic side alone) and the start link.
// Every collision-mesh contact visitor, its CCD sweeps and its scan ask this
// per pair; the allowance half is also asked once per dynamic element before
// a traversal, which only saves the walk.
[[seam::device_fn]] inline bool collider_pair_admitted(
    const PairSide &dynamic,
    const PairSide &collider,
    const unsigned *start_link_index,
    const unsigned *start_link_offset,
    unsigned has_start_link) {
    if (collider_intersection_allowed(dynamic)) {
        return false;
    }
    return !pair_linked_at_start(dynamic, collider, start_link_index,
                                 start_link_offset, has_start_link);
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
                        const PairSide &b,
                        const unsigned *start_link_index,
                        const unsigned *start_link_offset,
                        unsigned has_start_link) {
    const bool either_nonzero = a.mass > 0.0f || b.mass > 0.0f;
    return either_nonzero &&
           contact_pair_admitted(a, b, start_link_index, start_link_offset,
                                 has_start_link);
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
