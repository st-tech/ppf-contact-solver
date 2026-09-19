// File: entrypoints/shim_contact.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The CPU backend's CONTACT entry points: the broad phase and the intersection
// gate.
//
// Same contract as kernel_shim.cpp, and it is the whole point of this file: a
// loop, a gather and a scatter, with no arithmetic of its own. Every value
// below comes from a neutral kernel body or a shared header that nvcc and the
// Metal shader compiler compile from the same bytes. A float expression written
// here would be a second implementation of a numerical kernel, which is a fork
// and not a port.
//
// THREE PLACES BELOW LOOK LIKE ARITHMETIC AND ARE NOT PHYSICS, and each is
// named where it sits rather than left for a reader to wonder about:
//
//   1. `leaf_margin` composes a broad-phase inflation from two authored
//      lengths. CUDA spells the same composition in `lbvh/lbvh.cu`, in its
//      three `compute_leaf_aabbs_*_kernel` launchers, which is a `.cu` and not
//      a neutral body, so the composition lives in the LAUNCHER on that backend
//      too. It is one named function here for that reason, so the two copies
//      are comparable by eye and by test rather than inlined at three sites.
//   2. `edge_triangle_intersect_fp` forms four DIFFERENCES against `b0` and
//      hands them to the shared pierce predicate. The wrapper in `contact.cu`
//      does exactly this and for the same reason: the predicate is translation
//      invariant, so differencing first keeps its products well conditioned
//      instead of spending the mantissa on the distance from the origin. The
//      predicate itself is `isect::edge_triangle_intersect`, shared.
//   3. The two proximity testers compare a squared length against a squared
//      offset. The length comes from `proximity::` (shared) and the comparison
//      is the same one `contact.cu` makes at the same sites.
//
// Nothing here is under the kernel-body rule that a `.kernel.cpp` may name no
// backend and take no branch: this file is a backend's launcher, the same role
// `contact.cu` plays for CUDA and `face_math.mm` for Metal.

// data.hpp first, as in kernel_shim.cpp: it brings the shared scalar and vector
// types (`float`, `Vec3f`), the linalg pack, `AABB`, `IntersectionRecord` and
// the property and parameter structs, and it is the one header that reaches the
// backend seam.
#include "../src/kernels/data.hpp"

// The shared narrow-phase headers. These are NOT kernel bodies: they carry
// their own `SM_*` seam and compile unchanged as host C++, which is how
// `metal/face_math.mm` uses them and how the host arm of
// `kernels/tests/test_intersect_allow.cu` uses them. They are included from
// `src/kernels` directly, not from the rendered root, because the renderer has
// nothing to do to them.
// `accd.hpp` loops its fixed-size matrices with `int k` against an `unsigned`
// template parameter, in twelve places. That is a property of a header written
// for nvcc, which does not warn on it, and it produces about a hundred lines of
// `-Wsign-compare` here that would drown a real warning from this file. The
// suppression is scoped to the include rather than passed as a build flag for
// exactly that reason: a sign comparison written IN this shim must still be
// reported.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "../src/kernels/contact/accd.hpp"
#pragma GCC diagnostic pop

#include "../src/kernels/contact/distance.hpp"
#include "../src/kernels/contact/intersect_core.hpp"
#include "../src/kernels/contact/intersect_policy.hpp"

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// The diagnostic channel.
//
// `aabb_traversal.kernel.cpp` records four invariants through `DIAG_ASSERT4`,
// and every one of them is guarantee-class: a node index past the end, a leaf
// with no primitive, or a traversal stack that ran out of room. The last is the
// one that must never pass quietly, because the body's response is to `break`
// out of the walk, which SILENTLY TRUNCATES the traversal and drops whatever
// candidate pairs the abandoned subtrees held. A dropped candidate pair is a
// missed contact, and a missed contact is a penetration.
//
// So the CPU backend gives the macro a real channel rather than the `((void)0)`
// stub a pure host oracle uses. Each traversal chunk carries its own record,
// which Rust merges first-writer-wins in ascending chunk order, so the report
// does not depend on the thread count. The record and the macros live in
// `src/kernels/seam/seam_host.h`, which this file reaches through `data.hpp`:
// more than one shim expands them, Rust mirrors the record's layout, and a
// NEUTRAL kernel body reaching the same names is what lets a body with an
// invariant to report become a generated entry point.
// ---------------------------------------------------------------------------

#include "shim_diag.h"

// The rendered neutral bodies. Included AFTER the macro above, because the
// traversal body expands it.
#include "contact/aabb.kernel.cpp"
#include "contact/aabb_traversal.kernel.cpp"
#include "contact/pair_cache.kernel.cpp"
#include "lbvh/lbvh.kernel.cpp"

// ---------------------------------------------------------------------------
// Layout agreement.
//
// `AABB` is device-only and is not part of the Rust wire ABI, so Rust mirrors
// it by hand. A mirror that drifts is a silent wrong answer rather than a link
// error, so the size and the offsets are published for a Rust test to compare
// against rather than left to inspection.
// ---------------------------------------------------------------------------

extern "C" {

uint32_t aabb_sizeof_abi(void) { return (uint32_t)sizeof(AABB); }
uint32_t aabb_offset_min_abi(void) { return (uint32_t)offsetof(AABB, min); }
uint32_t aabb_offset_max_abi(void) { return (uint32_t)offsetof(AABB, max); }
uint32_t aabb_offset_active_abi(void) {
    return (uint32_t)offsetof(AABB, active);
}
uint32_t intersection_record_sizeof_abi(void) {
    return (uint32_t)sizeof(IntersectionRecord);
}
uint32_t diag_sizeof_abi(void) { return (uint32_t)sizeof(ChunkDiag); }
uint32_t aabb_max_query_abi(void) { return (uint32_t)AABB_MAX_QUERY; }
uint32_t max_intersection_records_abi(void) {
    return (uint32_t)MAX_INTERSECTION_RECORDS;
}

} // extern "C"

// ---------------------------------------------------------------------------
// The LBVH tree, as Karras builds it.
//
// The topology comes entirely from `lbvh/lbvh.kernel.cpp`: the leaf node, the
// internal node and its split search, and the depth walk. This file supplies
// the loop bounds and nothing else, which is what keeps the CPU tree the same
// tree CUDA and Metal build from the same Morton order.
// ---------------------------------------------------------------------------

extern "C" {

// THE DEPTH WALK IS NOT HERE. It declares its own entry point beside its body
// (`src/kernels/lbvh/lbvh.kernel.cpp`) and the range shim is rendered into
// `entrypoints/entries.cpp`.

} // extern "C"

// ---------------------------------------------------------------------------
// Leaf and internal AABBs.
// ---------------------------------------------------------------------------

// THE BROAD-PHASE INFLATION IS NOT HERE. This file stated
// `0.5f * ghat + offset` itself, beside the CUDA orchestrator's own three
// copies, and then forwarded to `aabb_leaf_margin` once that body existed.
// Every launcher that called it is now a generated entry, so the forwarder went
// with them and the statement lives once, in
// `src/kernels/contact/aabb.kernel.cpp`. It is a candidate-set decision rather
// than a physics value, so two statements would let two backends generate
// different candidate pairs from one scene;
// `the_leaf_margin_matches_cuda` in `src/driver/lbvh.rs` is what keeps the
// remaining CUDA copies equal until those launchers call the body too.

extern "C" {

// `aabb_overlap_abi` is defined once, in kernel_shim.cpp. Both shims wrap
// the same neutral body with the same signature, so a second definition here
// would be a duplicate symbol at link time rather than a second behavior.

// Clear the `active` flag of every leaf whose primitive is outside its
// collision window. Mirrors `invalidate_inactive_aabbs` in main.cu, including
// its position AFTER the bottom-up merge: the internal boxes then still bound
// the inactive leaves, which only widens the candidate set.
} // extern "C"

// ---------------------------------------------------------------------------
// Traversal.
//
// The visitor is the narrowest one that can exist: it tests the box and records
// the primitive. Every pair FILTER (the upper-triangular halving, `either_dyn`,
// the PDRD and collider exclusions, the allowance rule) is applied afterwards,
// by the caller that knows which of the four contact types it is walking. Doing
// it that way keeps the broad phase one piece of code rather than four, and
// keeps the filters where they can be read beside the rule they implement.
// ---------------------------------------------------------------------------

namespace {


} // namespace

extern "C" {

// Walks the tree once per query in [begin, end), appending (query, primitive)
// pairs. Returns the number of pairs FOUND, which exceeds `capacity` when the
// buffer was too small; in that case the first `capacity` pairs are valid and
// the caller must grow and re-walk.
//
// The order is deterministic: queries ascend, and within a query the node stack
// is walked in a fixed order by the shared body.

} // extern "C"

// ---------------------------------------------------------------------------
// The intersection gate.
//
// `check_intersection` is one half of what makes this solver penetration-free
// (the ACCD-filtered line search is the other), so every predicate below is the
// shared one and none of them is re-derived here.
// ---------------------------------------------------------------------------

namespace {

// The position gather in front of the shared pierce predicate.
//
// MIRRORS the `edge_triangle_intersect` wrapper in `contact.cu`. The four
// differences are formed FIRST and the predicate sees only those: two nearby
// coordinates cancel their leading digits when subtracted, so a predicate
// handed absolute positions would spend its mantissa on the distance from the
// origin rather than on the separation it has to resolve. The shared routine is
// translation invariant, so handing it vectors already relative to `b0` (with
// v0 = 0) asks it the same question.
inline bool edge_triangle_intersect_fp(const Vec3f &a0, const Vec3f &a1,
                                       const Vec3f &b0, const Vec3f &b1,
                                       const Vec3f &b2) {
    const Vec3f d1 = (b1 - b0).cast<float>();
    const Vec3f d2 = (b2 - b0).cast<float>();
    const Vec3f e0 = (a0 - b0).cast<float>();
    const Vec3f e1 = (a1 - b0).cast<float>();
    const float zero[3] = {0.0f, 0.0f, 0.0f};
    const float e0a[3] = {e0[0], e0[1], e0[2]};
    const float e1a[3] = {e1[0], e1[1], e1[2]};
    const float d1a[3] = {d1[0], d1[1], d1[2]};
    const float d2a[3] = {d2[0], d2[1], d2[2]};
    return isect::edge_triangle_intersect<float>(e0a, e1a, zero, d1a, d2a);
}

// MIRRORS `edge_has_shared_vert` in contact.cu.
inline bool edges_share_a_vertex(const Vec2u &a, const Vec2u &b) {
    return a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1];
}

} // namespace

extern "C" {

// The allowance rule, as `isect::intersection_tolerated` defines it. Given
// its own entry point so a Rust test can compare against the shared body rather
// than against a second statement of the rule.
int intersection_tolerated_abi(uint32_t a_object_index,
                                   unsigned char a_intersect_policy,
                                   uint32_t b_object_index,
                                   unsigned char b_intersect_policy,
                                   int a_pin_allows, int b_pin_allows) {
    return isect::intersection_tolerated(
               a_object_index, a_intersect_policy, b_object_index,
               b_intersect_policy, a_pin_allows != 0, b_pin_allows != 0)
               ? 1
               : 0;
}

uint32_t no_object_index_abi(void) { return (uint32_t)NO_OBJECT_INDEX; }

// The edge-triangle pierce, addressed by index into the position array.
int edge_triangle_intersect_abi(const Vec3f *vertex, uint32_t e0,
                                    uint32_t e1, uint32_t f0, uint32_t f1,
                                    uint32_t f2) {
    return edge_triangle_intersect_fp(vertex[e0], vertex[e1], vertex[f0],
                                      vertex[f1], vertex[f2])
               ? 1
               : 0;
}

// The same pierce against a SEPARATE position array for the triangle, which is
// what a dynamic edge against the rest-pose collision mesh needs: the two sides
// live in disjoint pools.
int edge_triangle_intersect_split_abi(const Vec3f *edge_vertex,
                                          uint32_t e0, uint32_t e1,
                                          const Vec3f *face_vertex,
                                          uint32_t f0, uint32_t f1,
                                          uint32_t f2) {
    return edge_triangle_intersect_fp(edge_vertex[e0], edge_vertex[e1],
                                      face_vertex[f0], face_vertex[f1],
                                      face_vertex[f2])
               ? 1
               : 0;
}

// Two edges closer than `offset`. MIRRORS `EdgeEdgeIntersectTester` in
// contact.cu, including its split between the disjoint case (one edge-edge
// closest pair) and the shared-vertex case (the two point-edge distances from
// the free endpoints), because a shared vertex makes the edge-edge closest pair
// identically zero and says nothing.
int edge_edge_proximity_abi(const Vec3f *vertex, uint32_t a0, uint32_t a1,
                                uint32_t b0, uint32_t b1, float offset) {
    const Vec2u e0(a0, a1);
    const Vec2u e1(b0, b1);
    if (!edges_share_a_vertex(e0, e1)) {
        const Vec3f p0 = vertex[a0];
        const Vec3f p1 = vertex[a1];
        const Vec3f q0 = vertex[b0];
        const Vec3f q1 = vertex[b1];
        const Vec4f c =
            proximity::edge_edge_distance_coeff<float, float>(p0, p1, q0,
                                                                  q1)
                ;
        const Vec3f x0 = c[0] * p0 + c[1] * p1;
        const Vec3f x1 = c[2] * q0 + c[3] * q1;
        const Vec3f e = (x0 - x1).cast<float>();
        return e.dot(e) < offset * offset ? 1 : 0;
    }
    const Vec2u ij[] = {Vec2u(0, 0), Vec2u(0, 1), Vec2u(1, 0), Vec2u(1, 1)};
    for (unsigned k = 0; k < 4; ++k) {
        const unsigned i = ij[k][0];
        const unsigned j = ij[k][1];
        if (e0[i] != e1[j]) {
            continue;
        }
        const Vec3f &q0 = vertex[e0[i]];
        const Vec3f &q1 = vertex[e0[1 - i]];
        const Vec3f &q2 = vertex[e1[1 - j]];
        const Vec2f c_0 =
            proximity::point_edge_distance_coeff_unclassified<float, float>(
                q1, q0, q2)
                ;
        const Vec2f c_1 =
            proximity::point_edge_distance_coeff_unclassified<float, float>(
                q2, q0, q1)
                ;
        const Vec3f e_0 = ((c_0[0] * q0 + c_0[1] * q2) - q1).cast<float>();
        const Vec3f e_1 = ((c_1[0] * q0 + c_1[1] * q1) - q2).cast<float>();
        const float sqr_d0 = e_0.dot(e_0);
        const float sqr_d1 = e_1.dot(e_1);
        const float smaller = sqr_d0 < sqr_d1 ? sqr_d0 : sqr_d1;
        if (smaller < offset * offset) {
            return 1;
        }
    }
    return 0;
}

// Two grains closer than `offset`. MIRRORS `PointPointIntersectTester`.
int point_point_proximity_abi(const Vec3f *vertex, uint32_t a, uint32_t b,
                                  float offset) {
    const Vec3f e = (vertex[a] - vertex[b]).cast<float>();
    return e.dot(e) < offset * offset ? 1 : 0;
}

// Fills one record. MIRRORS `record_intersection` in contact.cu, absolute by
// the same contract: the record reports world positions to the host for
// diagnostics, so the absolute magnitude is what belongs in it.
void fill_intersection_record_abi(IntersectionRecord *record, uint32_t type,
                                      uint32_t elem0, uint32_t elem1,
                                      const Vec3f *vertex0,
                                      const uint32_t *index0, uint32_t n0,
                                      const Vec3f *vertex1,
                                      const uint32_t *index1, uint32_t n1) {
    record->type = type;
    record->elem0 = elem0;
    record->elem1 = elem1;
    record->num_verts0 = n0;
    record->num_verts1 = n1;
    unsigned k = 0;
    for (uint32_t i = 0; i < n0; ++i) {
        for (unsigned d = 0; d < 3; ++d) {
            record->positions[k++] = vertex0[index0[i]][d];
        }
    }
    for (uint32_t i = 0; i < n1; ++i) {
        for (unsigned d = 0; d < 3; ++d) {
            record->positions[k++] = vertex1[index1[i]][d];
        }
    }
    for (; k < 15u; ++k) {
        record->positions[k] = 0.0f;
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// The DYNAMIC CSR matrix, which is CONTACT's matrix.
//
// Elastic and bending assemble into the FIXED matrix, whose pattern is known at
// build time. Contact cannot: which pairs touch is discovered per step, so its
// rows grow and shrink and the pattern is carried forward. That is why the
// dynamic matrix lives beside the contact entry points rather than beside the
// elastic ones.
//
// `kernel_shim.cpp` already exposes the four row primitives (sort, bisect,
// merge, compaction). These are the two remaining bodies a matrix needs, plus
// the block accumulate the scatter performs.
// ---------------------------------------------------------------------------

#include "csrmat/dynamic_csr.kernel.cpp"

extern "C" {

// One 3x3 block accumulated into another.
//
// Through `compute::atomic_add`, which is what the device assembly uses and
// what the host seam defines as a plain read, add and write back. Written here
// rather than as nine `+=` in Rust for the reason the whole seam exists: a
// float expression on the Rust side is a second implementation of arithmetic
// the other two backends get from one source.
//
// The destination is a `compute::atomic_float_t *` because that is what the
// operation takes: the type says the nine elements are reached by more than one
// thread, and a plain `float *` does not bind to it. The exported symbol is
// unchanged, a pointer being a pointer across the C ABI.
void block_add_abi(compute::atomic_float_t *destination, const float *source) {
    for (unsigned element = 0; element < 9; ++element) {
        compute::atomic_add(destination + element, source[element]);
    }
}

// True when a block is zero by the rule the compaction applies, which is
// Eigen's dummy precision and NOT exact equality. Exposed so a caller can ask
// the same question the compaction will, rather than restate the threshold.
int block_is_zero_abi(const float *value) {
    return dynamic_csr_block_is_zero(
               reinterpret_cast<const Mat3x3f *>(value), 0)
               ? 1
               : 0;
}

} // extern "C"

// ---------------------------------------------------------------------------
// ACCD: the conservative advance the line search filters every step with.
//
// This is the OTHER half of the penetration guarantee, beside
// `check_intersection` above. The barrier does not provide it: the cubic
// barrier is finite at the surface, so what actually stops a crossing is that
// the line search never takes a step past the time of impact this returns.
//
// Every one of the four entry points is `accd.hpp`, unmodified and shared. This
// file adds the gather (which primitive indices, out of which two position
// arrays) and hands back the two `OverlapInfo` fields as plain floats.
//
// TWO PROPERTIES OF THAT HEADER A CALLER MUST NOT UNDO, restated where a caller
// will read them:
//
//   * A returned time of exactly ZERO means the pair began the step already
//     inside the contact offset, and `overlap` was written. That is not "no
//     advance this step, carry on": the conservative advance cannot resolve a
//     start state that is already overlapping, and the run must end naming the
//     pair.
//   * `ccd_helper` works in a RESCALED frame, normalized by each pair's own
//     largest coordinate, which is why the entry points multiply `offset`,
//     `ccd_eps` and `park_floor(ghat)` by `scale` before passing them down. A
//     length handed in unrescaled is a size-dependent bug: its world value
//     would be `literal * max_entry / 0.99`, so it would behave differently on
//     a domino and against a six-metre ground triangle. Nothing here introduces
//     a length of its own, and nothing here may.
// ---------------------------------------------------------------------------

extern "C" {

// `accd::OverlapInfo` as two plain floats, so Rust need not mirror a C++ struct
// for a two-field out-parameter.
struct Overlap {
    float d2;
    float offset;
    // `accd::OverlapInfo::flagged`. `overlap_sizeof_abi` below is what catches
    // a half-applied change to this record.
    uint32_t flagged;
};

uint32_t overlap_sizeof_abi(void) {
    return (uint32_t)sizeof(Overlap);
}

// The parking floor, exposed so a Rust test can check that it is the shared
// one. It is a CONDITIONING parameter and not a safety one: `park > offset`
// holds for any positive floor and the entry check gives `clearance > 0`, so
// non-penetration does not depend on its magnitude. It reads a world `ghat` and
// returns a world length; the entry points rescale it.
float park_floor_abi(float ghat) { return accd::park_floor(ghat); }

// The ANALYTIC parking rule, exposed for the same reason and checked the same
// way. Unlike the mesh floor above these two are a SAFETY property rather than
// a conditioning one: they are what keeps a swept vertex off the surface, so a
// zero clearance never reaches the `mass / gap^2` the collider assembly forms.
// A Rust test asserts the three properties the sweeps rely on, which no call
// site can state for itself.
float park_gap_analytic_abi(float clearance0, float ghat, float eps) {
    return accd::park_gap_analytic(clearance0, ghat, eps);
}

float park_crossing_analytic_abi(float clearance0, float clearance1,
                                     float park) {
    return accd::park_crossing_analytic(clearance0, clearance1, park);
}

float point_triangle_ccd_abi(const Vec3f *x0, const Vec3f *x1,
                                 uint32_t point, uint32_t t0, uint32_t t1,
                                 uint32_t t2, float offset, float ghat,
                                 const ParamSet *param,
                                 Overlap *overlap) {
    accd::OverlapInfo info{0.0f, 0.0f, 0u};
    const float toi = accd::point_triangle_ccd(
        x0[point], x1[point], x0[t0], x0[t1], x0[t2], x1[t0], x1[t1], x1[t2],
        offset, ghat, param->line_search_max_t, param->ccd_eps, &info);
    overlap->d2 = info.d2;
    overlap->offset = info.offset;
    overlap->flagged = info.flagged;
    return toi;
}

float point_edge_ccd_abi(const Vec3f *x0, const Vec3f *x1, uint32_t point,
                             uint32_t e0, uint32_t e1, float offset, float ghat,
                             const ParamSet *param, Overlap *overlap) {
    accd::OverlapInfo info{0.0f, 0.0f, 0u};
    const float toi =
        accd::point_edge_ccd(x0[point], x1[point], x0[e0], x0[e1], x1[e0],
                             x1[e1], offset, ghat, param->line_search_max_t, param->ccd_eps, &info);
    overlap->d2 = info.d2;
    overlap->offset = info.offset;
    overlap->flagged = info.flagged;
    return toi;
}

// THE ARGUMENT ORDER IS (a's start, a's end, b's start, b's end), which is NOT
// the order the other three sweeps take and is the one this entry point had
// wrong. `accd::point_point_ccd` reads its four positions as two TRAJECTORIES,
// one per point, while `point_triangle_ccd` and `edge_edge_ccd` take every
// start before every end. Passing the start-major order here builds a frame out
// of one vertex at two times, whose start separation is that vertex's own step:
// smaller than any contact offset, so every pair is reported as beginning
// already inside its offset and every contact scene stops at its first step.
float point_point_ccd_abi(const Vec3f *x0, const Vec3f *x1, uint32_t a,
                              uint32_t b, float offset, float ghat,
                              const ParamSet *param, Overlap *overlap) {
    accd::OverlapInfo info{0.0f, 0.0f, 0u};
    const float toi = accd::point_point_ccd(x0[a], x1[a], x0[b], x1[b], offset,
                                            ghat, param->line_search_max_t, param->ccd_eps, &info);
    overlap->d2 = info.d2;
    overlap->offset = info.offset;
    overlap->flagged = info.flagged;
    return toi;
}

float edge_edge_ccd_abi(const Vec3f *x0, const Vec3f *x1, uint32_t a0,
                            uint32_t a1, uint32_t b0, uint32_t b1, float offset,
                            float ghat, const ParamSet *param,
                            Overlap *overlap) {
    accd::OverlapInfo info{0.0f, 0.0f, 0u};
    const float toi =
        accd::edge_edge_ccd(x0[a0], x0[a1], x0[b0], x0[b1], x1[a0], x1[a1],
                            x1[b0], x1[b1], offset, ghat, param->line_search_max_t, param->ccd_eps, &info);
    overlap->d2 = info.d2;
    overlap->offset = info.offset;
    overlap->flagged = info.flagged;
    return toi;
}

// THE THREE COLLISION-MESH SWEEPS. The static side is a rest-pose pool with one
// pose, so the same position is passed as both its start and its end and the
// swept frame carries the motion of the dynamic side alone. They sit in this
// file rather than in a collider shim of their own because `accd.hpp` declares
// its four entry points without `inline`, so a second translation unit
// including it would be a duplicate symbol at link.

// A moving dynamic vertex against a collider triangle.
float collision_point_triangle_ccd_abi(const Vec3f *x0, const Vec3f *x1,
                                           uint32_t point,
                                           const Vec3f *statics, uint32_t t0,
                                           uint32_t t1, uint32_t t2,
                                           float offset, float ghat,
                                           const ParamSet *param,
                                           Overlap *overlap) {
    accd::OverlapInfo info{0.0f, 0.0f, 0u};
    const float toi = accd::point_triangle_ccd(
        x0[point], x1[point], statics[t0], statics[t1], statics[t2],
        statics[t0], statics[t1], statics[t2], offset, ghat, param->line_search_max_t, param->ccd_eps, &info);
    overlap->d2 = info.d2;
    overlap->offset = info.offset;
    overlap->flagged = info.flagged;
    return toi;
}

// A collider vertex against a moving dynamic triangle.
float collision_point_triangle_ccd_static_point_abi(
    const Vec3f *statics, uint32_t point, const Vec3f *x0, const Vec3f *x1,
    uint32_t t0, uint32_t t1, uint32_t t2, float offset, float ghat,
    const ParamSet *param, Overlap *overlap) {
    accd::OverlapInfo info{0.0f, 0.0f, 0u};
    const float toi = accd::point_triangle_ccd(
        statics[point], statics[point], x0[t0], x0[t1], x0[t2], x1[t0], x1[t1],
        x1[t2], offset, ghat, param->line_search_max_t, param->ccd_eps, &info);
    overlap->d2 = info.d2;
    overlap->offset = info.offset;
    overlap->flagged = info.flagged;
    return toi;
}

// A moving dynamic edge against a collider edge.
float collision_edge_edge_ccd_abi(const Vec3f *x0, const Vec3f *x1,
                                      uint32_t a0, uint32_t a1,
                                      const Vec3f *statics, uint32_t b0,
                                      uint32_t b1, float offset, float ghat,
                                      const ParamSet *param,
                                      Overlap *overlap) {
    accd::OverlapInfo info{0.0f, 0.0f, 0u};
    const float toi = accd::edge_edge_ccd(
        x0[a0], x0[a1], statics[b0], statics[b1], x1[a0], x1[a1], statics[b0],
        statics[b1], offset, ghat, param->line_search_max_t, param->ccd_eps, &info);
    overlap->d2 = info.d2;
    overlap->offset = info.offset;
    overlap->flagged = info.flagged;
    return toi;
}

} // extern "C"
