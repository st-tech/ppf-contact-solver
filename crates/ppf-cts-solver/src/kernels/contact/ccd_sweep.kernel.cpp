// File: ccd_sweep.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts.
// ppf-cts-compute/seam/kernelgen.py renders it into the forms the four
// compilers read.
//
// THE SHAPE OF THE CCD LINE SEARCH: one dispatch per (query kind, tree) pair,
// one thread per QUERY primitive, the query box built inside the body, and the
// conservative advance invoked as a PER-HIT FUNCTOR inside the traversal.
// Nothing is materialized between the phases.
//
// WHY THIS FILE EXISTS. A BROAD phase that materialized a flat candidate pair
// list would cost `2 * queries * capacity_per_query` words, a capacity growable
// only by doubling, an overflow path for the scenes that outgrow it, and a
// download of the whole list before any sweep could run. There is no such
// buffer here, no such capacity and no overflow path, because a pair is never
// held at all: `aabb_query` (`aabb_traversal.kernel.cpp`) is a device template
// and a hit is consumed where it is found. The cost of the alternative is not
// marginal: on `examples/codim` such a list reaches 5.1 GiB and on
// `examples/twist` 6.9 GiB, past the 4 GiB an arena offset can address.
//
// WHAT LEAVES THE DEVICE. Two float arrays, one per primitive, min-folded into
// one number each; and two record arrays the host reads only when that number
// is exactly zero.

// THE FIRST INCLUDE ESTABLISHES THE TYPE VOCABULARY, which is what lets this
// file be compiled ALONE as a generated `*.entry.cu`. `accd.hpp` reaches
// `distance.hpp` and through it `data.hpp`, so `Vec3f`, `Vec3u`, `Vec2u`, the
// property and parameter records and `AABB` are all declared by the time the
// bodies below name them. Under MSL every one of these includes is neutralized
// as the shader is spliced and the prologue supplies the same names.
#include "accd.hpp"
#include "aabb_traversal.kernel.cpp"
#include "pair_filter.kernel.cpp"

// ---------------------------------------------------------------------------
// The overlap report.
// ---------------------------------------------------------------------------

// Record the first flagged pair this QUERY saw.
//
// FIRST WRITER WINS PER SLOT, and no atomic is needed because the slot belongs
// to exactly one thread. Latching one shared global under a compare-and-swap is
// not available in any case: `compute::` carries no compare-and-swap and Metal
// has no mutable device global, which is why `accd::OverlapInfo` is an
// out-parameter rather than a global here.
//
// THE FLAG IS THE RECORD'S OWN, NOT THE RETURNED TIME. A returned zero does not
// mean the record was written: ACCD's probe cap returns `lower_t`, which is
// zero when the very first advance underflows, and that path writes nothing.
// Reading the time instead would report a structured overlapping start for a
// pair that did not begin the step overlapping, naming a `d2` and an `offset`
// that are the initializer above rather than measurements.
[[seam::device_fn]] inline void ccd_record_overlap(
    CcdOverlapRecord &record, unsigned kind, unsigned elem0,
    unsigned elem1, const accd::OverlapInfo &info) {
    if (info.flagged != 0u && record.flagged == 0u) {
        record.flagged = 1u;
        record.kind = kind;
        record.elem0 = elem0;
        record.elem1 = elem1;
        record.d2 = info.d2;
        record.offset = info.offset;
    }
}

// An unwritten report, which is what every query starts from.
[[seam::device_fn]] inline CcdOverlapRecord ccd_no_overlap() {
    CcdOverlapRecord record;
    record.flagged = 0u;
    record.kind = 0u;
    record.elem0 = 0u;
    record.elem1 = 0u;
    record.d2 = 0.0f;
    record.offset = 0.0f;
    return record;
}

// The lowest-indexed flagged report in one block of a report array whose kind
// lies in `[kind_first, kind_last]`, as `base + slot`, or `0xFFFFFFFF` where the
// block holds none.
//
// ONE WORD CROSSES TO THE HOST PER ASSEMBLY. The smallest of these words over
// both arrays is both the flag and the slot's name, and the driver reads the
// pair's details out of that one record alone, only once the word says a pair
// was flagged.
//
// `base` IS WHAT KEEPS TWO ARRAYS IN ORDER. The vertex reports reduce at base
// zero and the edge reports past the last vertex slot, so the smallest word over
// both is the slot an ascending vertex-then-edge scan stops at, which is the
// order the driver reports in. An unsigned minimum is exact, so the association
// of the reduction cannot move it.
//
// THE KIND TEST IS THE DECODER'S. `ccd::decode_assembly_overlap` skips a flagged
// record whose kind it does not own, and so does this.
[[seam::device_fn]] inline void overlap_first_flagged_leaf(
    const CcdOverlapRecord *overlap, unsigned count, unsigned kind_first,
    unsigned kind_last, unsigned base, unsigned block_size, unsigned *out,
    unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    unsigned first = 0xFFFFFFFFu;
    for (unsigned i = 0; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        const unsigned flagged = overlap[at].flagged;
        const unsigned kind = overlap[at].kind;
        if (flagged != 0u && kind >= kind_first && kind <= kind_last) {
            first = base + at;
            break;
        }
    }
    out[block_index] = first;
}

[[seam::entry(blocks, block_index)]] void overlap_first_flagged_leaf(
    const CcdOverlapRecord *overlap, unsigned count, unsigned kind_first,
    unsigned kind_last, unsigned base, unsigned block_size, unsigned *out,
    unsigned block_index, unsigned blocks);

// The four sweep names, as `CcdOverlapRecord::kind` encodes them and as the
// driver's `Sweep` enum reads them back.
enum : unsigned {
    CCD_SWEEP_POINT_TRIANGLE = 0u,
    CCD_SWEEP_POINT_EDGE = 1u,
    CCD_SWEEP_POINT_POINT = 2u,
    CCD_SWEEP_EDGE_EDGE = 3u
};

// Fold one query's answer into its own slot of the per-primitive arrays.
//
// A MIN, so the order of the six dispatches cannot change the number. The slot
// is the query's own and the six sweeps are sequential dispatches, so nothing
// here races with anything.
[[seam::device_fn]] inline void ccd_commit(
    float *out_toi,
    CcdOverlapRecord *out_overlap, unsigned slot, float toi,
    const CcdOverlapRecord &record) {
    out_toi[slot] = fmath::min(out_toi[slot], toi);
    if (record.flagged != 0u && out_overlap[slot].flagged == 0u) {
        out_overlap[slot] = record;
    }
}

// ---------------------------------------------------------------------------
// The six per-hit visitors, one for each sweep below.
//
// BOTH METHODS CARRY THE EXECUTION SPACE, and neither may go without it:
// `test` calls `aabb_overlap`, which is `[[seam::device_fn]]`, and a member
// function with no annotation is a HOST function to nvcc. Only
// `--features cuda-abi` catches that.
//
// EVERY VISITOR BOUNDS-CHECKS THE LEAF PRIMITIVE INDEX. It is DATA rather than
// the thread index, so the entry's `[[seam::count]]` guard says nothing about
// it, and Metal returns 0.0 for an out-of-bounds read and faults on nothing.
// CUDA's live release asserts inside `Vec::operator[]` have no Metal
// equivalent, so `[[seam::diag]]` is the channel that reports it here.
// ---------------------------------------------------------------------------

// The dynamic point-face visitor: one dynamic vertex against a dynamic
// triangle.
struct CcdPointFaceVisitor {
    const Vec3f *x0;
    const Vec3f *x1;
    const Vec3u *face;
    const VertexProp *vertex_prop;
    const FaceProp *face_prop;
    const VertexParam *vertex_param;
    const FaceParam *face_param;
    unsigned vertex_index;
    unsigned face_count;
    float max_t;
    float ccd_eps;
    float toi;
    CcdOverlapRecord overlap;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag, index < face_count, static_cast<float>(index),
                     static_cast<float>(face_count),
                     static_cast<float>(vertex_index), 0.0f);
        if (index >= face_count) {
            return false;
        }
        const Vec3u f = face[index];
        const VertexProp vprop = vertex_prop[vertex_index];
        const FaceProp fprop = face_prop[index];
        const VertexProp anchor = vertex_prop[f[0]];
        if (!contact_pair_admitted(pair_side_of_vertex(vprop),
                                   pair_side_of_face(anchor, fprop))) {
            return false;
        }
        const VertexParam vparam = vertex_param[vprop.param_index];
        accd::OverlapInfo info;
        info.d2 = 0.0f;
        info.offset = 0.0f;
        info.flagged = 0u;
        float result = max_t;
        unsigned kind = CCD_SWEEP_POINT_TRIANGLE;
        // A VERTEX OF THE FACE ITSELF IS NOT SKIPPED, it is swept against the
        // OPPOSITE EDGE. The two share a triangle, so they can never collide
        // face-on, but the vertex can still be driven onto the far edge and
        // that crossing is a real one.
        unsigned dup = 3u;
        for (unsigned slot = 0u; slot < 3u; ++slot) {
            if (f[slot] == vertex_index) {
                dup = slot;
                break;
            }
        }
        if (dup == 3u) {
            const FaceParam fparam = face_param[fprop.param_index];
            const float offset = vparam.offset + fparam.offset;
            const float ghat = 0.5f * (vparam.ghat + fparam.ghat);
            const Vec3f p0 = x0[vertex_index];
            const Vec3f p1 = x1[vertex_index];
            const Vec3f t00 = x0[f[0]];
            const Vec3f t01 = x0[f[1]];
            const Vec3f t02 = x0[f[2]];
            const Vec3f t10 = x1[f[0]];
            const Vec3f t11 = x1[f[1]];
            const Vec3f t12 = x1[f[2]];
            result = accd::point_triangle_ccd(p0, p1, t00, t01, t02, t10, t11,
                                              t12, offset, ghat, max_t,
                                              ccd_eps, &info);
        } else {
            const float offset = 2.0f * vparam.offset;
            const float ghat = vparam.ghat;
            const unsigned j = (dup + 1u) % 3u;
            const unsigned k = (dup + 2u) % 3u;
            const Vec3f p0 = x0[f[dup]];
            const Vec3f p1 = x1[f[dup]];
            const Vec3f q00 = x0[f[j]];
            const Vec3f q10 = x1[f[j]];
            const Vec3f q01 = x0[f[k]];
            const Vec3f q11 = x1[f[k]];
            result = accd::point_edge_ccd(p0, p1, q00, q01, q10, q11, offset,
                                          ghat, max_t, ccd_eps, &info);
            kind = CCD_SWEEP_POINT_EDGE;
        }
        ccd_record_overlap(overlap, kind, vertex_index, f[0], info);
        if (result < max_t) {
            toi = fmath::min(toi, result);
            return true;
        }
        return false;
    }
};

// The dynamic point-point visitor, the sweep a faceless
// SAND cloud depends on: it has no point-face candidate among the mesh
// primitives, so without this nothing bounds a Newton step that drives two
// grains through each other.
struct CcdPointPointVisitor {
    const Vec3f *x0;
    const Vec3f *x1;
    const VertexProp *vertex_prop;
    const VertexParam *vertex_param;
    unsigned vertex_index;
    unsigned vertex_count;
    float max_t;
    float ccd_eps;
    float toi;
    CcdOverlapRecord overlap;
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
        // THE UPPER-TRIANGULAR HALVING: each unordered pair is swept once, by
        // the thread whose own index is the larger of the two.
        if (!(index < vertex_index)) {
            return false;
        }
        const VertexProp a = vertex_prop[vertex_index];
        const VertexProp b = vertex_prop[index];
        if (!contact_pair_admitted(pair_side_of_vertex(a),
                                   pair_side_of_vertex(b))) {
            return false;
        }
        const VertexParam pa = vertex_param[a.param_index];
        const VertexParam pb = vertex_param[b.param_index];
        const float offset = pa.offset + pb.offset;
        const float ghat = 0.5f * (pa.ghat + pb.ghat);
        accd::OverlapInfo info;
        info.d2 = 0.0f;
        info.offset = 0.0f;
        info.flagged = 0u;
        const Vec3f a0 = x0[vertex_index];
        const Vec3f a1 = x1[vertex_index];
        const Vec3f b0 = x0[index];
        const Vec3f b1 = x1[index];
        const float result = accd::point_point_ccd(a0, a1, b0, b1, offset, ghat,
                                                   max_t, ccd_eps, &info);
        ccd_record_overlap(overlap, CCD_SWEEP_POINT_POINT, vertex_index, index,
                           info);
        if (result < max_t) {
            toi = fmath::min(toi, result);
            return true;
        }
        return false;
    }
};

// The dynamic edge-edge visitor.
struct CcdEdgeEdgeVisitor {
    const Vec3f *x0;
    const Vec3f *x1;
    const Vec2u *edge;
    const VertexProp *vertex_prop;
    const EdgeProp *edge_prop;
    const EdgeParam *edge_param;
    unsigned edge_index;
    unsigned edge_count;
    float max_t;
    float ccd_eps;
    float toi;
    CcdOverlapRecord overlap;
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
        if (!(edge_index < index)) {
            return false;
        }
        const Vec2u e0 = edge[edge_index];
        const Vec2u e1 = edge[index];
        const EdgeProp pa = edge_prop[edge_index];
        const EdgeProp pb = edge_prop[index];
        const VertexProp anchor_a = vertex_prop[e0[0]];
        const VertexProp anchor_b = vertex_prop[e1[0]];
        if (!contact_pair_admitted(pair_side_of_edge(anchor_a, pa),
                                   pair_side_of_edge(anchor_b, pb))) {
            return false;
        }
        const EdgeParam ea = edge_param[pa.param_index];
        const EdgeParam eb = edge_param[pb.param_index];
        const float offset = ea.offset + eb.offset;
        const float ghat = 0.5f * (ea.ghat + eb.ghat);
        accd::OverlapInfo info;
        info.d2 = 0.0f;
        info.offset = 0.0f;
        info.flagged = 0u;
        float result = max_t;
        unsigned kind = CCD_SWEEP_EDGE_EDGE;
        if (!edge_has_shared_vert(e0, e1)) {
            const Vec3f p00 = x0[e0[0]];
            const Vec3f p01 = x0[e0[1]];
            const Vec3f q00 = x0[e1[0]];
            const Vec3f q01 = x0[e1[1]];
            const Vec3f p10 = x1[e0[0]];
            const Vec3f p11 = x1[e0[1]];
            const Vec3f q10 = x1[e1[0]];
            const Vec3f q11 = x1[e1[1]];
            result = accd::edge_edge_ccd(p00, p01, q00, q01, p10, p11, q10, q11,
                                         offset, ghat, max_t, ccd_eps, &info);
        } else {
            // TWO EDGES SHARING A VERTEX still have two free endpoints, and
            // either can be driven onto the other's segment, so both
            // point-edge pairs are swept and the smaller time kept.
            kind = CCD_SWEEP_POINT_EDGE;
            for (unsigned slot = 0u; slot < 4u; ++slot) {
                const unsigned i = slot / 2u;
                const unsigned j = slot % 2u;
                if (e0[i] != e1[j]) {
                    continue;
                }
                const unsigned idx0 = e0[i];
                const unsigned idx1 = e0[1u - i];
                const unsigned idx2 = e1[1u - j];
                const Vec3f q00 = x0[idx0];
                const Vec3f q10 = x1[idx0];
                const Vec3f q01 = x0[idx1];
                const Vec3f q11 = x1[idx1];
                const Vec3f q02 = x0[idx2];
                const Vec3f q12 = x1[idx2];
                const float toi_0 = accd::point_edge_ccd(
                    q01, q11, q00, q02, q10, q12, offset, ghat, max_t, ccd_eps,
                    &info);
                const float toi_1 = accd::point_edge_ccd(
                    q02, q12, q00, q01, q10, q11, offset, ghat, max_t, ccd_eps,
                    &info);
                result = fmath::min(toi_0, toi_1);
                break;
            }
        }
        ccd_record_overlap(overlap, kind, e0[0], e1[0], info);
        if (result < max_t) {
            toi = fmath::min(toi, result);
            return true;
        }
        return false;
    }
};

// The mesh-against-collider point-face visitor: a
// moving dynamic vertex against a collider triangle. The collider has ONE pose,
// so the same position is passed as both ends of its half of the swept frame.
struct CcdCollisionPointFaceM2cVisitor {
    const Vec3f *x0;
    const Vec3f *x1;
    const Vec3f *collider_vertex;
    const Vec3u *collider_face;
    const VertexProp *vertex_prop;
    const FaceProp *collider_face_prop;
    const VertexParam *vertex_param;
    const FaceParam *collider_face_param;
    unsigned vertex_index;
    unsigned collider_face_count;
    float max_t;
    float ccd_eps;
    float toi;
    CcdOverlapRecord overlap;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag, index < collider_face_count,
                     static_cast<float>(index),
                     static_cast<float>(collider_face_count),
                     static_cast<float>(vertex_index), 0.0f);
        if (index >= collider_face_count) {
            return false;
        }
        const VertexProp vprop = vertex_prop[vertex_index];
        // A zero-mass vertex is a static solid, including a moving-static
        // shell, and never collides with the collision mesh, which is itself a
        // static solid.
        if (vprop.mass == 0.0f) {
            return false;
        }
        const Vec3u f = collider_face[index];
        const Vec3f t0 = collider_vertex[f[0]];
        const Vec3f t1 = collider_vertex[f[1]];
        const Vec3f t2 = collider_vertex[f[2]];
        const Vec3f p0 = x0[vertex_index];
        const Vec3f p1 = x1[vertex_index];
        const VertexParam vparam = vertex_param[vprop.param_index];
        const FaceParam fparam =
            collider_face_param[collider_face_prop[index].param_index];
        const float offset = vparam.offset + fparam.offset;
        const float ghat = 0.5f * (vparam.ghat + fparam.ghat);
        accd::OverlapInfo info;
        info.d2 = 0.0f;
        info.offset = 0.0f;
        info.flagged = 0u;
        const float result =
            accd::point_triangle_ccd(p0, p1, t0, t1, t2, t0, t1, t2, offset,
                                     ghat, max_t, ccd_eps, &info);
        ccd_record_overlap(overlap, CCD_SWEEP_POINT_TRIANGLE, vertex_index,
                           f[0], info);
        if (result < max_t) {
            toi = fmath::min(toi, result);
            return true;
        }
        return false;
    }
};

// The collider-against-mesh point-face visitor: a
// collider vertex against a moving dynamic triangle. The query primitive is the
// collider's, so its box is UNSWEPT.
struct CcdCollisionPointFaceC2mVisitor {
    const Vec3f *x0;
    const Vec3f *x1;
    const Vec3f *collider_vertex;
    const Vec3u *face;
    const VertexProp *vertex_prop;
    const FaceProp *face_prop;
    const VertexProp *collider_vertex_prop;
    const FaceParam *face_param;
    const VertexParam *collider_vertex_param;
    unsigned vertex_index;
    unsigned face_count;
    float max_t;
    float ccd_eps;
    float toi;
    CcdOverlapRecord overlap;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag, index < face_count, static_cast<float>(index),
                     static_cast<float>(face_count),
                     static_cast<float>(vertex_index), 0.0f);
        if (index >= face_count) {
            return false;
        }
        const FaceProp fprop = face_prop[index];
        if (fprop.fixed || !(fprop.mass > 0.0f)) {
            return false;
        }
        const Vec3u f = face[index];
        const VertexProp anchor = vertex_prop[f[0]];
        if (collider_intersection_allowed(pair_side_of_face(anchor, fprop))) {
            return false;
        }
        const Vec3f t00 = x0[f[0]];
        const Vec3f t01 = x0[f[1]];
        const Vec3f t02 = x0[f[2]];
        const Vec3f t10 = x1[f[0]];
        const Vec3f t11 = x1[f[1]];
        const Vec3f t12 = x1[f[2]];
        const Vec3f p = collider_vertex[vertex_index];
        const FaceParam fparam = face_param[fprop.param_index];
        const VertexParam vparam =
            collider_vertex_param[collider_vertex_prop[vertex_index]
                                      .param_index];
        const float offset = fparam.offset + vparam.offset;
        const float ghat = 0.5f * (fparam.ghat + vparam.ghat);
        accd::OverlapInfo info;
        info.d2 = 0.0f;
        info.offset = 0.0f;
        info.flagged = 0u;
        const float result =
            accd::point_triangle_ccd(p, p, t00, t01, t02, t10, t11, t12, offset,
                                     ghat, max_t, ccd_eps, &info);
        ccd_record_overlap(overlap, CCD_SWEEP_POINT_TRIANGLE, f[0],
                           vertex_index, info);
        if (result < max_t) {
            toi = fmath::min(toi, result);
            return true;
        }
        return false;
    }
};

// The collider edge-edge visitor: a dynamic edge against a collider edge.
struct CcdCollisionEdgeEdgeVisitor {
    const Vec3f *x0;
    const Vec3f *x1;
    const Vec2u *edge;
    const Vec3f *collider_vertex;
    const Vec2u *collider_edge;
    const EdgeProp *edge_prop;
    const EdgeProp *collider_edge_prop;
    const EdgeParam *edge_param;
    const EdgeParam *collider_edge_param;
    unsigned edge_index;
    unsigned collider_edge_count;
    float max_t;
    float ccd_eps;
    float toi;
    CcdOverlapRecord overlap;
    DiagHandle diag;

    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned index) {
        DIAG_ASSERT4(diag, index < collider_edge_count,
                     static_cast<float>(index),
                     static_cast<float>(collider_edge_count),
                     static_cast<float>(edge_index), 0.0f);
        if (index >= collider_edge_count) {
            return false;
        }
        const EdgeProp dyn = edge_prop[edge_index];
        if (dyn.fixed || !(dyn.mass > 0.0f)) {
            return false;
        }
        const Vec2u e0 = edge[edge_index];
        const Vec2u e1 = collider_edge[index];
        const Vec3f p00 = x0[e0[0]];
        const Vec3f p01 = x0[e0[1]];
        const Vec3f p10 = x1[e0[0]];
        const Vec3f p11 = x1[e0[1]];
        const Vec3f q0 = collider_vertex[e1[0]];
        const Vec3f q1 = collider_vertex[e1[1]];
        const EdgeParam ea = edge_param[dyn.param_index];
        const EdgeParam eb =
            collider_edge_param[collider_edge_prop[index].param_index];
        const float offset = ea.offset + eb.offset;
        const float ghat = 0.5f * (ea.ghat + eb.ghat);
        accd::OverlapInfo info;
        info.d2 = 0.0f;
        info.offset = 0.0f;
        info.flagged = 0u;
        const float result =
            accd::edge_edge_ccd(p00, p01, q0, q1, p10, p11, q0, q1, offset,
                                ghat, max_t, ccd_eps, &info);
        ccd_record_overlap(overlap, CCD_SWEEP_EDGE_EDGE, e0[0], e1[0], info);
        if (result < max_t) {
            toi = fmath::min(toi, result);
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// The six bodies, one per (query kind, tree) pair.
//
// THE MORTON REMAP IS A PURE PERMUTATION OF THE QUERY-TO-THREAD ASSIGNMENT.
// Leaf `t` of a tree stores primitive `node[2 * t] - 1` in Morton order, so
// adjacent lanes get spatially adjacent queries and their traversals touch
// overlapping node sets. The per-query result and its output slot are
// unchanged. `aabb_leaf_active` already relies on the same leaf layout.
// It is applied on the point-face, point-point and edge-edge dispatches, each
// of which opens with `ccd_morton_remap`, and NOT on the three collision-mesh
// ones, which index by `element` directly.
// ---------------------------------------------------------------------------

// Read the primitive a leaf slot names, checked against the index space it may
// name. A remap that is out of range would query one primitive's box and write
// another's slot, silently on Metal.
[[seam::device_fn]] inline unsigned ccd_morton_remap(
    const unsigned *node, unsigned element, unsigned bound,
    DiagHandle diag) {
    const unsigned biased = node[2u * element];
    DIAG_ASSERT4(diag, biased > 0u && biased - 1u < bound,
                 static_cast<float>(biased), static_cast<float>(bound),
                 static_cast<float>(element), 0.0f);
    if (biased == 0u || biased - 1u >= bound) {
        return element;
    }
    return biased - 1u;
}

// A vertex's swept query box, inflated by that vertex's own contact margin and
// masked against the collision window.
[[seam::device_fn]] inline AABB ccd_point_box(
    const Vec3f *x0, const Vec3f *x1,
    const VertexParam &vparam, float extrapolate,
    const unsigned *active, unsigned has_active,
    unsigned element) {
    const Vec3f a = x0[element];
    const Vec3f b = x1[element];
    AABB box = aabb_make_swept_point(a, b, extrapolate,
                                     aabb_leaf_margin(vparam.ghat,
                                                      vparam.offset));
    if (has_active != 0u && active[element] == 0u) {
        box.active = false;
    }
    return box;
}

// An edge's swept query box, on the same terms.
[[seam::device_fn]] inline AABB ccd_edge_box(
    const Vec3f *x0, const Vec3f *x1,
    const Vec2u &e,
    const EdgeParam &eparam, float extrapolate,
    const unsigned *active, unsigned has_active,
    unsigned element) {
    const Vec3f a0 = x0[e[0]];
    const Vec3f a1 = x0[e[1]];
    const Vec3f b0 = x1[e[0]];
    const Vec3f b1 = x1[e[1]];
    AABB box = aabb_make_swept_edge(a0, a1, b0, b1, extrapolate,
                                    aabb_leaf_margin(eparam.ghat,
                                                     eparam.offset));
    if (has_active != 0u && active[element] == 0u) {
        box.active = false;
    }
    return box;
}

// The dynamic point-face sweep.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ccd_point_face(
    const Vec3f *x0, const Vec3f *x1,
    const Vec3u *face, unsigned face_count,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const VertexParam *vertex_param,
    const FaceParam *face_param,
    const unsigned *vertex_node,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const unsigned *active, unsigned has_active, float max_t,
    float ccd_eps, float *out_toi,
    CcdOverlapRecord *out_overlap, unsigned query_count,
    DiagHandle diag, unsigned element) {
    const unsigned i = ccd_morton_remap(vertex_node, element, query_count, diag);
    const VertexParam vparam = vertex_param[vertex_prop[i].param_index];
    CcdPointFaceVisitor op;
    op.x0 = x0;
    op.x1 = x1;
    op.face = face;
    op.vertex_prop = vertex_prop;
    op.face_prop = face_prop;
    op.vertex_param = vertex_param;
    op.face_param = face_param;
    op.vertex_index = i;
    op.face_count = face_count;
    op.max_t = max_t;
    op.ccd_eps = ccd_eps;
    op.toi = max_t;
    op.overlap = ccd_no_overlap();
    op.diag = diag;
    const AABB box =
        ccd_point_box(x0, x1, vparam, max_t, active, has_active, i);
    aabb_query(node, node_count, aabb, root, op, box, diag);
    ccd_commit(out_toi, out_overlap, i, op.toi, op.overlap);
}

// The dynamic point-point sweep.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ccd_point_point(
    const Vec3f *x0, const Vec3f *x1,
    const VertexProp *vertex_prop,
    const VertexParam *vertex_param,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const unsigned *active, unsigned has_active, float max_t,
    float ccd_eps, float *out_toi,
    CcdOverlapRecord *out_overlap, unsigned query_count,
    DiagHandle diag, unsigned element) {
    // The remap tree and the traversal tree are the SAME here, which is why one
    // node array serves both.
    const unsigned i = ccd_morton_remap(node, element, query_count, diag);
    const VertexParam vparam = vertex_param[vertex_prop[i].param_index];
    CcdPointPointVisitor op;
    op.x0 = x0;
    op.x1 = x1;
    op.vertex_prop = vertex_prop;
    op.vertex_param = vertex_param;
    op.vertex_index = i;
    op.vertex_count = query_count;
    op.max_t = max_t;
    op.ccd_eps = ccd_eps;
    op.toi = max_t;
    op.overlap = ccd_no_overlap();
    op.diag = diag;
    const AABB box =
        ccd_point_box(x0, x1, vparam, max_t, active, has_active, i);
    aabb_query(node, node_count, aabb, root, op, box, diag);
    ccd_commit(out_toi, out_overlap, i, op.toi, op.overlap);
}

// The dynamic edge-edge sweep.
//
// SEEDED FROM THE POINT-FACE MINIMUM, `t_vf_seed`, which the caller folds over
// the vertex-space times before dispatching either edge sweep. An edge
// collision later than `T_vf` cannot beat
// an already-found earlier point-face hit, so bounding the sweep to
// `[0, T_vf]` prunes far BVH nodes while leaving the final min bit-identical.
// The margin is untouched, so coverage of `[0, T_vf]` stays conservative.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ccd_edge_edge(
    const Vec3f *x0, const Vec3f *x1,
    const Vec2u *edge,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const unsigned *active, unsigned has_active,
    float t_vf_seed, float max_t, float ccd_eps,
    float *out_toi_ee,
    CcdOverlapRecord *out_overlap_ee, unsigned query_count,
    DiagHandle diag, unsigned element) {
    const unsigned i = ccd_morton_remap(node, element, query_count, diag);
    const Vec2u e = edge[i];
    const EdgeParam eparam = edge_param[edge_prop[i].param_index];
    CcdEdgeEdgeVisitor op;
    op.x0 = x0;
    op.x1 = x1;
    op.edge = edge;
    op.vertex_prop = vertex_prop;
    op.edge_prop = edge_prop;
    op.edge_param = edge_param;
    op.edge_index = i;
    op.edge_count = query_count;
    op.max_t = max_t;
    op.ccd_eps = ccd_eps;
    op.toi = t_vf_seed;
    op.overlap = ccd_no_overlap();
    op.diag = diag;
    const AABB box =
        ccd_edge_box(x0, x1, e, eparam, t_vf_seed, active, has_active, i);
    aabb_query(node, node_count, aabb, root, op, box, diag);
    ccd_commit(out_toi_ee, out_overlap_ee, i, op.toi, op.overlap);
}

// A dynamic vertex against the collider face tree.
//
// A FIX-PINNED VERTEX IS SKIPPED BEFORE THE TRAVERSAL, at the
// `vprop.fix_index != 0u` return below: a vertex no solve can move contributes
// no time of impact, so traversing the tree for it can only produce hits to
// discard.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ccd_collision_point_face_m2c(
    const Vec3f *x0, const Vec3f *x1,
    const Vec3f *collider_vertex,
    const Vec3u *collider_face, unsigned collider_face_count,
    const VertexProp *vertex_prop,
    const FaceProp *collider_face_prop,
    const VertexParam *vertex_param,
    const FaceParam *collider_face_param,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const unsigned *active, unsigned has_active, float max_t,
    float ccd_eps, float *out_toi,
    CcdOverlapRecord *out_overlap, DiagHandle diag,
    unsigned element) {
    const VertexProp vprop = vertex_prop[element];
    if (vprop.fix_index != 0u) {
        return;
    }
    if (collider_intersection_allowed(pair_side_of_vertex(vprop))) {
        return;
    }
    const VertexParam vparam = vertex_param[vprop.param_index];
    CcdCollisionPointFaceM2cVisitor op;
    op.x0 = x0;
    op.x1 = x1;
    op.collider_vertex = collider_vertex;
    op.collider_face = collider_face;
    op.vertex_prop = vertex_prop;
    op.collider_face_prop = collider_face_prop;
    op.vertex_param = vertex_param;
    op.collider_face_param = collider_face_param;
    op.vertex_index = element;
    op.collider_face_count = collider_face_count;
    op.max_t = max_t;
    op.ccd_eps = ccd_eps;
    op.toi = max_t;
    op.overlap = ccd_no_overlap();
    op.diag = diag;
    const AABB box =
        ccd_point_box(x0, x1, vparam, max_t, active, has_active, element);
    aabb_query(node, node_count, aabb, root, op, box, diag);
    ccd_commit(out_toi, out_overlap, element, op.toi, op.overlap);
}

// A collider vertex against the dynamic face tree.
//
// THE ONE PASS WHOSE QUERY PRIMITIVE IS THE COLLIDER'S, so its box is UNSWEPT:
// the collider has one pose and contributes no motion. It writes at the
// COLLIDER vertex index into the same array the dynamic sweeps write, which is
// why that array is sized `max(surface_vert_count, collider_vert_count)`.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ccd_collision_point_face_c2m(
    const Vec3f *x0, const Vec3f *x1,
    const Vec3f *collider_vertex,
    const Vec3u *face, unsigned face_count,
    const VertexProp *vertex_prop,
    const FaceProp *face_prop,
    const VertexProp *collider_vertex_prop,
    const FaceParam *face_param,
    const VertexParam *collider_vertex_param,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root, float max_t,
    float ccd_eps, float *out_toi,
    CcdOverlapRecord *out_overlap, DiagHandle diag,
    unsigned element) {
    CcdCollisionPointFaceC2mVisitor op;
    op.x0 = x0;
    op.x1 = x1;
    op.collider_vertex = collider_vertex;
    op.face = face;
    op.vertex_prop = vertex_prop;
    op.face_prop = face_prop;
    op.collider_vertex_prop = collider_vertex_prop;
    op.face_param = face_param;
    op.collider_vertex_param = collider_vertex_param;
    op.vertex_index = element;
    op.face_count = face_count;
    op.max_t = max_t;
    op.ccd_eps = ccd_eps;
    op.toi = max_t;
    op.overlap = ccd_no_overlap();
    op.diag = diag;
    const Vec3f q = collider_vertex[element];
    const VertexParam vparam =
        collider_vertex_param[collider_vertex_prop[element].param_index];
    const AABB box =
        aabb_make_point(q, aabb_leaf_margin(vparam.ghat, vparam.offset));
    aabb_query(node, node_count, aabb, root, op, box, diag);
    ccd_commit(out_toi, out_overlap, element, op.toi, op.overlap);
}

// A dynamic edge against the collider edge tree, seeded from `t_vf_seed` like
// the dynamic edge sweep.
[[seam::entry(element)]]
[[seam::device_fn]] inline void ccd_collision_edge_edge(
    const Vec3f *x0, const Vec3f *x1,
    const Vec2u *edge,
    const Vec3f *collider_vertex,
    const Vec2u *collider_edge, unsigned collider_edge_count,
    const VertexProp *vertex_prop,
    const EdgeProp *edge_prop,
    const EdgeProp *collider_edge_prop,
    const EdgeParam *edge_param,
    const EdgeParam *collider_edge_param,
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const unsigned *active, unsigned has_active,
    float t_vf_seed, float max_t, float ccd_eps,
    float *out_toi_ee,
    CcdOverlapRecord *out_overlap_ee, DiagHandle diag,
    unsigned element) {
    const Vec2u e = edge[element];
    const EdgeProp eprop = edge_prop[element];
    const VertexProp anchor = vertex_prop[e[0]];
    if (collider_intersection_allowed(pair_side_of_edge(anchor, eprop))) {
        return;
    }
    const EdgeParam eparam = edge_param[eprop.param_index];
    CcdCollisionEdgeEdgeVisitor op;
    op.x0 = x0;
    op.x1 = x1;
    op.edge = edge;
    op.collider_vertex = collider_vertex;
    op.collider_edge = collider_edge;
    op.edge_prop = edge_prop;
    op.collider_edge_prop = collider_edge_prop;
    op.edge_param = edge_param;
    op.collider_edge_param = collider_edge_param;
    op.edge_index = element;
    op.collider_edge_count = collider_edge_count;
    op.max_t = max_t;
    op.ccd_eps = ccd_eps;
    op.toi = t_vf_seed;
    op.overlap = ccd_no_overlap();
    op.diag = diag;
    const AABB box = ccd_edge_box(x0, x1, e, eparam, t_vf_seed, active,
                                  has_active, element);
    aabb_query(node, node_count, aabb, root, op, box, diag);
    ccd_commit(out_toi_ee, out_overlap_ee, element, op.toi, op.overlap);
}

// ---------------------------------------------------------------------------
// The six entry points.
//
// `out_toi` AND `out_overlap` ARE BASE POINTERS RATHER THAN SCATTERS, because
// the slot a body writes is the MORTON-REMAPPED primitive rather than the
// thread index, and because the write is a MIN-FOLD into a slot the caller
// seeded rather than a plain store.
//
// THE `ParamSet` IS NOT IN ANY RECORD, its two read fields are: `ParamSet` is
// declared in data.hpp, which the Metal shader compiler cannot read in any
// position, and `accd.hpp` reads exactly `line_search_max_t` and `ccd_eps`.
//
// `has_active` IS THE DRIVER'S DECISION, and it carries what a null test would
// otherwise answer: a scene that authored no collision window has no mask,
// which the driver knows, and a record field is a handle with no spelling for
// absent. The handle then names a REAL zero-length
// allocation, never `Handle::NONE`, because a generated entry resolves every
// buffer it is handed before the body runs.
// ---------------------------------------------------------------------------
