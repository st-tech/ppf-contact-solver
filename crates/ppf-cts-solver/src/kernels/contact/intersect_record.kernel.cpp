// File: intersect_record.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::device]]` / `[[seam::thread]]` are the address spaces MSL
// requires on every pointer and reference type.
//
// No include of its own. `IntersectionRecord`, `Vec3f` and the `compute::`
// atomic names arrive from whatever declares them for the backend that is
// compiling, which is data.hpp plus the seam prologue under nvcc and on the
// host.
//
// The intersection scan's REPORT: one record per detected intersection, claimed
// out of a shared array so the host can name the elements and show where.
//
// THE SLOT IS CLAIMED EVEN WHEN IT OVERFLOWS, deliberately. `counter` counts
// every intersection found, whether or not a record was stored for it, so the
// host learns the true demand rather than the array's capacity. Read it as a
// demand count and clamp before indexing.

// The four record kinds, as `IntersectionRecord::type` encodes them and as the
// driver's `Report` reads them back. They are stated here, beside the writer,
// because a number written at the claim and read at the report is one fact and
// two spellings of it can disagree with nothing to notice.
enum : unsigned {
    INTERSECT_RECORD_FACE_EDGE = 0u,
    INTERSECT_RECORD_EDGE_EDGE = 1u,
    INTERSECT_RECORD_COLLISION_MESH = 2u,
    INTERSECT_RECORD_POINT_POINT = 3u
};

// THE ONE PLACE IN THE INTERSECTION PATH THAT STORES AN ABSOLUTE POSITION, and
// the justification travels with the body.
//
// Everywhere else the intersection path works in DIFFERENCES of nearby
// coordinates and never in the coordinates themselves, because subtracting two
// large nearby fp32 values cancels the leading digits and leaves only the
// precision the operands had left over.
//
// Here the argument is that the ABSOLUTE MAGNITUDE IS THE RESULT. The record is
// not an input to any further arithmetic: it is a diagnostic handed to the
// host, which reports world positions to a user who has to find the geometry in
// their scene. A translation-invariant difference would not answer the question
// the record is asked.
//
// `positions` is packed as `elem0`'s vertices then `elem1`'s, three floats
// each, so the caller must supply `n0 + n1 <= 5` for the fixed 15-float field.
[[seam::device_fn]] inline void intersection_record_write(
    IntersectionRecord *record, unsigned type, unsigned elem0,
    unsigned elem1, const Vec3f *verts0, unsigned n0,
    const Vec3f *verts1, unsigned n1) {
    record->type = type;
    record->elem0 = elem0;
    record->elem1 = elem1;
    record->num_verts0 = n0;
    record->num_verts1 = n1;
    unsigned k = 0;
    for (unsigned i = 0; i < n0; i++)
        for (unsigned d = 0; d < 3; d++)
            record->positions[k++] = verts0[i][d];
    for (unsigned i = 0; i < n1; i++)
        for (unsigned d = 0; d < 3; d++)
            record->positions[k++] = verts1[i][d];
}

// Claim a slot and, if one was available, fill it.
//
// `capacity` is a parameter rather than a constant so that no backend has to
// carry a copy of the array's size: the caller allocated the array and is the
// one that knows.
[[seam::device_fn]] inline void intersection_record_claim(
    IntersectionRecord *records,
    compute::atomic_uint_t *counter, unsigned capacity,
    unsigned type, unsigned elem0, unsigned elem1,
    const Vec3f *verts0, unsigned n0,
    const Vec3f *verts1, unsigned n1) {
    unsigned slot = compute::atomic_add(counter, 1u);
    if (slot < capacity) {
        intersection_record_write(records + slot, type, elem0, elem1,
                                      verts0, n0, verts1, n1);
    }
}
