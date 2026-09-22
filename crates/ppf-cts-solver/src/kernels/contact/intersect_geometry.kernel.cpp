// File: intersect_geometry.kernel.cpp
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
// THE INTERSECTION SCAN, which is the final penetration gate: the geometry
// predicates first, then the four per-hit visitors and the four entry points
// that walk a BVH with them.
//
// The GEOMETRY half comes first because it is the part that answers "given two
// elements already admitted by `intersect_pair_reported`, do they actually
// intersect?". The pair filter, these predicates and the record write are three
// separate bodies because the four testers combine them differently: the
// collision-mesh tester has its verdict precomputed by its caller and reaches
// only the pierce, and the two proximity forms have no pierce at all.
//
// EVERY POSITION ARRIVES AS A THREAD-SPACE VALUE, and that is a requirement
// rather than a style. A position must be loaded out of device memory before
// any operator touches it: the Metal compiler rejects subtracting two
// `device`-space positions outright, and the caller's own bounds-checked
// accessor is what should perform the load, so nothing here takes a pointer.
//
// EVERY FLOAT THESE BODIES FORM IS A DIFFERENCE OF TWO POSITIONS, taken before
// anything scales, crosses or dots it (`(x0 - x1)`), so an absolute coordinate
// never enters a product. Subtracting two nearby fp32 coordinates cancels their
// leading digits, and what survives is limited by the float spacing at the
// coordinates' own magnitude: about 1.2e-7 near a coordinate of 1, about 6.1e-5
// near 1000. Differencing first, and anchoring at one of the two points rather
// than at the origin, is what keeps a small separation resolvable; it is also
// why a test scene authored around 0 cannot show the error a production scene
// placed far out runs into.

// THE FIRST INCLUDE ESTABLISHES THE TYPE VOCABULARY, which is what lets this
// file be compiled ALONE as a generated `*.entry.cu`. `distance.hpp` reaches
// `data.hpp`, so `Vec3f`, `Vec3u`, `Vec2u`, `Vec4f`, `Vec2f`, `AABB`,
// `IntersectionRecord` and the property and parameter records are all declared
// by the time the bodies below name them. Under MSL every one of these includes
// is neutralized as the shader is spliced and the prologue supplies the same
// names.
#include "distance.hpp"
#include "intersect_core.hpp"
#include "pair_filter.kernel.cpp"
#include "intersect_record.kernel.cpp"
#include "aabb_traversal.kernel.cpp"

// Whether the segment `(a0, a1)` pierces the triangle `(b0, b1, b2)`.
//
// A thin differencing wrapper over `isect::edge_triangle_intersect`, which is
// dependency-free, templated on the scalar and shared with the host build-time
// check in ppf-cts-core. That routine is translation invariant, so handing it
// vectors already relative to `b0` (with v0 = 0) changes no verdict and keeps
// the scene's absolute magnitude out of the arithmetic.
//
// `signed_volume_first` and `signed_volume_second` are exported for the
// denormal instrumentation channel, which is a per-backend diagnostic and so
// cannot live in a neutral body. They are the two quantities whose PRODUCT
// underflowed in the defect this predicate's sign form now avoids: each is the
// triangle's area vector dotted with an endpoint offset, so for an edge lying
// nearly in the triangle's plane both are tiny while still ordinary normal
// floats, and `s1 * s2` flushes to a signed zero that reads as "no crossing".
// Measured on an Apple GPU, the product form missed 5 of 8 genuine crossings.
// The predicate itself compares the two SIGNS and never materializes the
// product; these outputs let an instrumented build watch the magnitudes without
// a second copy of the arithmetic.
[[seam::device_fn]] inline bool edge_triangle_pierce(
    const Vec3f &a0, const Vec3f &a1,
    const Vec3f &b0, const Vec3f &b1,
    const Vec3f &b2,
    float &signed_volume_first,
    float &signed_volume_second) {
    Vec3f d1 = (b1 - b0).cast<float>();
    Vec3f d2 = (b2 - b0).cast<float>();
    Vec3f e0 = (a0 - b0).cast<float>();
    Vec3f e1 = (a1 - b0).cast<float>();
    const float zero[3] = {0.0f, 0.0f, 0.0f};
    const float e0a[3] = {e0[0], e0[1], e0[2]};
    const float e1a[3] = {e1[0], e1[1], e1[2]};
    const float d1a[3] = {d1[0], d1[1], d1[2]};
    const float d2a[3] = {d2[0], d2[1], d2[2]};
    Vec3f normal = d1.cross(d2);
    signed_volume_first = e0.dot(normal);
    signed_volume_second = e1.dot(normal);
    return isect::edge_triangle_intersect<float>(e0a, e1a, zero, d1a, d2a);
}

// Whether two edges that share NO vertex come within `offset` of each other.
//
// One edge-edge closest pair, evaluated at the coefficients and compared as a
// squared distance so no root is taken.
[[seam::device_fn]] inline bool edge_edge_disjoint_proximity(
    const Vec3f &p0, const Vec3f &p1,
    const Vec3f &q0, const Vec3f &q1,
    float offset) {
    Vec4f c = proximity::edge_edge_distance_coeff<float, float>(p0, p1, q0,
                                                                     q1)
                   ;
    Vec3f x0 = c[0] * p0 + c[1] * p1;
    Vec3f x1 = c[2] * q0 + c[3] * q1;
    Vec3f e = (x0 - x1).cast<float>();
    return e.dot(e) < offset * offset;
}

// Whether two edges that SHARE a vertex come within `offset` of each other.
//
// The edge-edge closest pair is identically zero for such a pair and says
// nothing, so the measurement is the two point-edge distances from the free
// endpoints instead: how far the first edge's free end lies from the second
// edge, and the reverse. The smaller of the two decides.
//
// `shared` is the common vertex, `free_first` the other end of the first edge
// and `free_second` the other end of the second.
[[seam::device_fn]] inline bool edge_edge_shared_vertex_proximity(
    const Vec3f &shared,
    const Vec3f &free_first,
    const Vec3f &free_second, float offset) {
    Vec2f c_0 = proximity::point_edge_distance_coeff_unclassified<float,
                                                                  float>(
                     free_first, shared, free_second)
                     ;
    Vec2f c_1 = proximity::point_edge_distance_coeff_unclassified<float,
                                                                  float>(
                     free_second, shared, free_first)
                     ;
    Vec3f e_0 =
        ((c_0[0] * shared + c_0[1] * free_second) - free_first).cast<float>();
    Vec3f e_1 =
        ((c_1[0] * shared + c_1[1] * free_first) - free_second).cast<float>();
    float sqr_d0 = e_0.dot(e_0);
    float sqr_d1 = e_1.dot(e_1);
    // The smaller of the two, spelled as `std::min` expands rather than through
    // a `fmath::min`. The two differ on a NaN operand, and MSL has no `std`.
    float smaller = sqr_d1 < sqr_d0 ? sqr_d1 : sqr_d0;
    return smaller < offset * offset;
}

// Whether two edges intersect, for the intersection scan's purposes.
//
// The four endpoint positions arrive by value in the order the index pairs name
// them: `p0 = x[e0[0]]`, `p1 = x[e0[1]]`, `q0 = x[e1[0]]`, `q1 = x[e1[1]]`. The
// index pairs are still needed, because which endpoints coincide selects the
// measurement.
//
// The shared-vertex scan visits all four (i, j) combinations rather than
// stopping at the first match, so a pair of edges naming the SAME two vertices
// is measured twice, once from each end. That is the behavior of the code this
// was taken from and it is preserved deliberately: stopping early would change
// which of the two measurements decides.
[[seam::device_fn]] inline bool edge_edge_intersect_proximity(
    const Vec2u &e0, const Vec2u &e1,
    const Vec3f &p0, const Vec3f &p1,
    const Vec3f &q0, const Vec3f &q1,
    float offset) {
    if (!edge_has_shared_vert(e0, e1)) {
        return edge_edge_disjoint_proximity(p0, p1, q0, q1, offset);
    }
    const Vec2u ij[] = {Vec2u(0, 0), Vec2u(0, 1), Vec2u(1, 0), Vec2u(1, 1)};
    for (unsigned k = 0; k < 4; ++k) {
        unsigned i = ij[k][0];
        unsigned j = ij[k][1];
        if (e0[i] == e1[j]) {
            Vec3f shared = i == 0 ? p0 : p1;
            Vec3f free_first = i == 0 ? p1 : p0;
            Vec3f free_second = j == 0 ? q1 : q0;
            if (edge_edge_shared_vertex_proximity(shared, free_first,
                                                      free_second, offset)) {
                return true;
            }
        }
    }
    return false;
}

// Whether two grains begin the step already inside each other's contact radius.
//
// `offset` is the SUM of the two vertices' own offsets, so the test is on the
// centers. The smooth contact barrier is undefined inside the wall, so such a
// start state would otherwise abort mid-advance with an opaque kernel
// assertion; detecting it lets initialization fail by name instead.
[[seam::device_fn]] inline bool
point_point_intersect_proximity(const Vec3f &p,
                                    const Vec3f &q,
                                    float offset) {
    Vec3f e = (p - q).cast<float>();
    return e.dot(e) < offset * offset;
}

// ---------------------------------------------------------------------------
// THE SCAN ITSELF: one dispatch per QUERY ELEMENT, the tester invoked as a
// PER-HIT DEVICE FUNCTOR inside the traversal, and nothing materialized between
// the two.
//
// One thread per query element builds its testers on its own stack and hands
// them to `aabb_query`; a hit never leaves the device. THREE THINGS CROSS TO
// THE HOST and every one is a scalar or a bounded array: a 4-byte claim
// counter, at most `capacity` records, and two per-element flag arrays folded
// to a verdict.
//
// WHY THERE IS NO CANDIDATE PAIR LIST. Running the broad phase into a flat
// pair list, downloading it, and applying the filters and the geometry to it in
// a serial host loop would RELOCATE the computation rather than read a result
// back: the filtering and the predicates are device work, and a host loop is
// the same work in a place the solve does not scale. A hit stays in registers
// here.
//
// THE CLAIM IS WHY THESE ROWS ARE `Scatter::Claim` AND NOT `Disjoint`. Threads
// take numbered slots out of one counter, so a slot assignment is reproducible
// only in ascending order. The FLAG write is this thread's own slot and would
// be disjoint on its own; the record claim is not, and the stricter of the two
// decides.
//
// EVERY VISITOR BOUNDS-CHECKS THE LEAF PRIMITIVE INDEX. It is DATA rather than
// the thread index, so the entry's `[[seam::count]]` guard says nothing about
// it, and Metal returns 0.0 for an out-of-bounds read and faults on nothing.
// A leaf array is reached through a raw pointer that carries no bounds assert
// of its own on any backend, so `[[seam::diag]]` is the channel that reports
// one.
// ---------------------------------------------------------------------------

// The edge-versus-face visitor: the query element is an edge, each leaf a face.
struct IntersectFaceEdgeVisitor {
    const Vec3f *vert;
    const Vec3u *face;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const FaceProp *face_prop;
    const EdgeProp *edge_prop;
    IntersectionRecord *records;
    compute::atomic_uint_t *counter;
    unsigned capacity;
    unsigned edge_index;
    unsigned face_count;
    DiagHandle diag;

    // BOTH METHODS CARRY THE EXECUTION SPACE, and neither may go without it:
    // `test` calls `aabb_overlap`, which is `[[seam::device_fn]]`, and a member
    // function with no annotation is a HOST function to nvcc.
    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag, index < face_count, static_cast<float>(index),
                     static_cast<float>(face_count),
                     static_cast<float>(edge_index), 0.0f);
        if (index >= face_count) {
            return false;
        }
        const Vec3u f = face[index];
        const Vec2u e = edge[edge_index];
        const FaceProp fprop = face_prop[index];
        const EdgeProp eprop = edge_prop[edge_index];
        const VertexProp fanchor = vertex_prop[f[0]];
        const VertexProp eanchor = vertex_prop[e[0]];
        const PairSide a = pair_side_of_face(fanchor, fprop);
        const PairSide b = pair_side_of_edge(eanchor, eprop);
        if (!intersect_pair_reported(a, b)) {
            return false;
        }
        // A face and an edge that share a vertex meet by construction, and
        // reporting that would abort every mesh at its own seams.
        if (e[0] == f[0] || e[0] == f[1] || e[0] == f[2] || e[1] == f[0] ||
            e[1] == f[1] || e[1] == f[2]) {
            return false;
        }
        const Vec3f x0 = vert[f[0]];
        const Vec3f x1 = vert[f[1]];
        const Vec3f x2 = vert[f[2]];
        const Vec3f y0 = vert[e[0]];
        const Vec3f y1 = vert[e[1]];
        float first = 0.0f;
        float second = 0.0f;
        if (edge_triangle_pierce(y0, y1, x0, x1, x2, first, second)) {
            Vec3f fv[3] = {x0, x1, x2};
            Vec3f ev[2] = {y0, y1};
            intersection_record_claim(records, counter, capacity,
                                      INTERSECT_RECORD_FACE_EDGE, index,
                                      edge_index, fv, 3, ev, 2);
            return true;
        }
        return false;
    }
};

// The edge-versus-edge visitor: both sides are edges, so the pair is halved below.
struct IntersectEdgeEdgeVisitor {
    const Vec3f *vert;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const EdgeProp *edge_prop;
    const EdgeParam *edge_param;
    IntersectionRecord *records;
    compute::atomic_uint_t *counter;
    unsigned capacity;
    unsigned edge_index;
    unsigned edge_count;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag, index < edge_count, static_cast<float>(index),
                     static_cast<float>(edge_count),
                     static_cast<float>(edge_index), 0.0f);
        if (index >= edge_count) {
            return false;
        }
        // THE UPPER-TRIANGULAR HALVING, keeping only `index < edge_index`, so
        // each unordered pair is measured once and from the same end.
        if (!(index < edge_index)) {
            return false;
        }
        const Vec2u e0 = edge[edge_index];
        const Vec2u e1 = edge[index];
        const EdgeProp pa = edge_prop[edge_index];
        const EdgeProp pb = edge_prop[index];
        const VertexProp anchor_a = vertex_prop[e0[0]];
        const VertexProp anchor_b = vertex_prop[e1[0]];
        const PairSide a = pair_side_of_edge(anchor_a, pa);
        const PairSide b = pair_side_of_edge(anchor_b, pb);
        if (!intersect_pair_reported(a, b)) {
            return false;
        }
        const EdgeParam param_a = edge_param[pa.param_index];
        const EdgeParam param_b = edge_param[pb.param_index];
        const float offset = param_a.offset + param_b.offset;
        const Vec3f p0 = vert[e0[0]];
        const Vec3f p1 = vert[e0[1]];
        const Vec3f q0 = vert[e1[0]];
        const Vec3f q1 = vert[e1[1]];
        if (edge_edge_intersect_proximity(e0, e1, p0, p1, q0, q1, offset)) {
            Vec3f ev0[2] = {p0, p1};
            Vec3f ev1[2] = {q0, q1};
            intersection_record_claim(records, counter, capacity,
                                      INTERSECT_RECORD_EDGE_EDGE, edge_index,
                                      index, ev0, 2, ev1, 2);
            return true;
        }
        return false;
    }
};

// The vertex-versus-vertex visitor, the pass a faceless SAND cloud depends on:
// the edge scans never run for one, so without it an overlapping cloud passes
// the initial check and aborts mid-advance.
struct IntersectPointPointVisitor {
    const Vec3f *vert;
    const VertexProp *vertex_prop;
    const VertexParam *vertex_param;
    IntersectionRecord *records;
    compute::atomic_uint_t *counter;
    unsigned capacity;
    unsigned vertex_index;
    unsigned vertex_count;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag, index < vertex_count, static_cast<float>(index),
                     static_cast<float>(vertex_count),
                     static_cast<float>(vertex_index), 0.0f);
        if (index >= vertex_count) {
            return false;
        }
        if (!(index < vertex_index)) {
            return false;
        }
        const VertexProp pa = vertex_prop[vertex_index];
        const VertexProp pb = vertex_prop[index];
        const PairSide a = pair_side_of_vertex(pa);
        const PairSide b = pair_side_of_vertex(pb);
        if (!intersect_pair_reported(a, b)) {
            return false;
        }
        const VertexParam param_a = vertex_param[pa.param_index];
        const VertexParam param_b = vertex_param[pb.param_index];
        const float offset = param_a.offset + param_b.offset;
        const Vec3f p = vert[vertex_index];
        const Vec3f q = vert[index];
        if (point_point_intersect_proximity(p, q, offset)) {
            Vec3f ev0[1] = {p};
            Vec3f ev1[1] = {q};
            intersection_record_claim(records, counter, capacity,
                                      INTERSECT_RECORD_POINT_POINT,
                                      vertex_index, index, ev0, 1, ev1, 1);
            return true;
        }
        return false;
    }
};

// The dynamic-edge-versus-collider-face visitor.
//
// The static side is a disjoint contact-only pool outside the solved namespace,
// so the pair is inter-object by construction and carries no self-intersection
// case. It has no material and no pins of its own, so the whole verdict rests
// on the dynamic edge and the caller settles it ONCE per edge rather than per
// leaf.
struct IntersectCollisionMeshVisitor {
    const Vec3f *collider_vertex;
    const Vec3u *collider_face;
    IntersectionRecord *records;
    compute::atomic_uint_t *counter;
    unsigned capacity;
    unsigned edge_index;
    unsigned collider_face_count;
    Vec3f y0;
    Vec3f y1;
    unsigned tolerated;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        if (tolerated != 0u) {
            return false;
        }
        DIAG_ASSERT4(diag, index < collider_face_count,
                     static_cast<float>(index),
                     static_cast<float>(collider_face_count),
                     static_cast<float>(edge_index), 0.0f);
        if (index >= collider_face_count) {
            return false;
        }
        const Vec3u f = collider_face[index];
        const Vec3f x0 = collider_vertex[f[0]];
        const Vec3f x1 = collider_vertex[f[1]];
        const Vec3f x2 = collider_vertex[f[2]];
        float first = 0.0f;
        float second = 0.0f;
        if (edge_triangle_pierce(y0, y1, x0, x1, x2, first, second)) {
            Vec3f fv[3] = {x0, x1, x2};
            Vec3f ev[2] = {y0, y1};
            intersection_record_claim(records, counter, capacity,
                                      INTERSECT_RECORD_COLLISION_MESH, index,
                                      edge_index, fv, 3, ev, 2);
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// The four composition bodies.
//
// EACH IS ONE QUERY ELEMENT'S WHOLE WORK: read its prebuilt query box, build
// the tester on the stack, walk the tree, and set this element's flag if the
// walk reported a hit. The flag is written ONLY on a hit and never cleared,
// which is what lets the three edge scans fold into one array.
//
// THE QUERY BOX IS PREBUILT ONE DISPATCH EARLIER rather than formed here, a
// difference in WHERE the box is formed and not in what it contains: the four
// `aabb_*_scan_query` entries in `aabb.kernel.cpp` form it from the element's
// own verts and apply the collision-window mask, on the device. An inactive box
// prunes at the root either way, because `aabb_overlap` refuses a pair with
// either side inactive; returning early here is that same verdict reached
// without descending.
// ---------------------------------------------------------------------------

[[seam::entry(element)]]
[[seam::device_fn]] inline void intersect_scan_face_edge(
    const Vec3f *vert,
    const Vec3u *face, unsigned face_count,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const EdgeProp *edge_prop,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const AABB *query, unsigned *flag,
    IntersectionRecord *records,
    compute::atomic_uint_t *counter, unsigned capacity,
    DiagHandle diag, unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return;
    }
    IntersectFaceEdgeVisitor op;
    op.vert = vert;
    op.face = face;
    op.edge = edge;
    op.vertex_prop = vertex_prop;
    op.face_prop = face_prop;
    op.edge_prop = edge_prop;
    op.records = records;
    op.counter = counter;
    op.capacity = capacity;
    op.edge_index = element;
    op.face_count = face_count;
    op.diag = diag;
    if (aabb_query(node, node_count, aabb, root, op, box, diag) > 0u) {
        flag[element] = 1u;
    }
}

[[seam::entry(element)]]
[[seam::device_fn]] inline void intersect_scan_edge_edge(
    const Vec3f *vert,
    const Vec2u *edge, unsigned edge_count,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const AABB *query, unsigned *flag,
    IntersectionRecord *records,
    compute::atomic_uint_t *counter, unsigned capacity,
    DiagHandle diag, unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return;
    }
    IntersectEdgeEdgeVisitor op;
    op.vert = vert;
    op.edge = edge;
    op.vertex_prop = vertex_prop;
    op.edge_prop = edge_prop;
    op.edge_param = edge_param;
    op.records = records;
    op.counter = counter;
    op.capacity = capacity;
    op.edge_index = element;
    op.edge_count = edge_count;
    op.diag = diag;
    if (aabb_query(node, node_count, aabb, root, op, box, diag) > 0u) {
        flag[element] = 1u;
    }
}

[[seam::entry(element)]]
[[seam::device_fn]] inline void intersect_scan_point_point(
    const Vec3f *vert, unsigned vertex_count,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const AABB *query, unsigned *flag,
    IntersectionRecord *records,
    compute::atomic_uint_t *counter, unsigned capacity,
    DiagHandle diag, unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return;
    }
    IntersectPointPointVisitor op;
    op.vert = vert;
    op.vertex_prop = vertex_prop;
    op.vertex_param = vertex_param;
    op.records = records;
    op.counter = counter;
    op.capacity = capacity;
    op.vertex_index = element;
    op.vertex_count = vertex_count;
    op.diag = diag;
    if (aabb_query(node, node_count, aabb, root, op, box, diag) > 0u) {
        flag[element] = 1u;
    }
}

// THE TWO PER-EDGE VERDICTS THAT ARE SETTLED OUTSIDE THE TRAVERSAL, because
// neither depends on the leaf. A zero-mass edge is a static solid and the
// collision mesh is one too, so the pair could never resolve; and the allowance
// is asked of the dynamic edge alone, the static side being handed
// `NO_OBJECT_INDEX` and an empty policy, under which "either side opts in"
// reduces to "the dynamic side opted in".
[[seam::entry(element)]]
[[seam::device_fn]] inline void intersect_scan_collision_mesh(
    const Vec3f *vert,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const Vec3f *collider_vertex,
    const Vec3u *collider_face, unsigned collider_face_count,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const AABB *query, unsigned *flag,
    IntersectionRecord *records,
    compute::atomic_uint_t *counter, unsigned capacity,
    DiagHandle diag, unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return;
    }
    const EdgeProp eprop = edge_prop[element];
    if (!(eprop.mass > 0.0f)) {
        return;
    }
    const Vec2u e = edge[element];
    const VertexProp anchor = vertex_prop[e[0]];
    if (collider_intersection_allowed(pair_side_of_edge(anchor, eprop))) {
        return;
    }
    IntersectCollisionMeshVisitor op;
    op.collider_vertex = collider_vertex;
    op.collider_face = collider_face;
    op.records = records;
    op.counter = counter;
    op.capacity = capacity;
    op.edge_index = element;
    op.collider_face_count = collider_face_count;
    op.y0 = vert[e[0]];
    op.y1 = vert[e[1]];
    op.tolerated = 0u;
    op.diag = diag;
    if (aabb_query(node, node_count, aabb, root, op, box, diag) > 0u) {
        flag[element] = 1u;
    }
}

// ---------------------------------------------------------------------------
// The four entry points.
//
// `query`, `flag` and `records` are BASE POINTERS rather than a gather and a
// scatter, and the reason is the same for all three: `query[element]` is copied
// into thread space by the body before anything touches it, `flag[element]` is
// written only on a hit so a scatter's unconditional store would clear the
// other scans' verdicts, and `records` is claimed at an index no thread owns.
//
// `capacity` is a parameter rather than a constant so that no backend carries a
// copy of the record array's size: the caller allocated it and is the one that
// knows.
// ---------------------------------------------------------------------------
