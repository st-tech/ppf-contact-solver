// File: contact_statistics.kernel.cpp
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
// No include of its own. `NO_OBJECT_INDEX` and the `compute::` atomic names
// arrive from whatever declares them for the backend that is compiling, which
// is data.hpp (through contact/intersect_policy.hpp) plus the seam prologue
// under nvcc and on the host.
//
// DIAG_ASSERT4 IS NOT A PLATFORM BRANCH AND STAYS A MACRO, exactly as in
// contact/aabb_traversal.kernel.cpp: it records __FILE__ and __LINE__ of the
// failing check, which no function can read for its caller, and each backend
// defines it over its own diagnostic channel.
//
// PER-OBJECT CONTACT STATISTICS, the one channel in the contact kernels where
// an index is neither bounds-checked by its container nor in the caller's own
// index space, and which has already shipped two defects because of it.
//
// The two rules that came out of those defects, both load-bearing:
//
//   1. THE ARRAYS ARE OPTIONAL, so a recorder may not assume they exist. They
//      are filled only when the session directory carries a statistics input,
//      which the frontend writes only for a scene that has statistics objects;
//      otherwise all three are size 0. Statistics are telemetry, so absent
//      configuration means record nothing, and must never abort. Indexing them
//      unconditionally aborted a run on its FIRST contact and made every
//      session directory predating the channel unrunnable.
//
//   2. EVERY SUBSCRIPT IS CHECKED, HERE, AGAINST THE SIZE THE CALLER PASSES.
//      The dynamic and the static vertex spaces are both `unsigned`, so nothing
//      in the type system stops one being handed to the other's array, and a
//      float narrows to `unsigned` silently: a shadowed name once made a
//      contact FORCE's x component the index, which read in range often enough
//      to charge contacts to unrelated objects with nothing in the output to
//      say so. A backend that never faults on an out-of-bounds write (Metal
//      drops it and reports success) has no other guard at all.
//
// Each check both records through the diagnostic channel and skips the write.
// The skip is what makes the guard real where the assert is compiled out or
// where the platform has no assert; where the assert traps, it is unreachable.

// Whether the statistics channel is configured at all. Everything else is gated
// on this, and the ONE fact it reads is the counter array's size, because that
// array is what `configure_dataset` sizes when a statistics input is present.
[[seam::device_fn]] inline bool
statistics_enabled(unsigned contact_count_size) {
    return contact_count_size > 0;
}

// Charge one contact to one or two objects.
//
// `NO_OBJECT_INDEX` means "no object to attribute to" and is skipped rather
// than counted, and the second side is skipped when it names the same object,
// so a contact within one object counts once.
template <typename D>
[[seam::device_fn]] inline void statistics_record_object_contact(
    compute::atomic_uint_t *contact_count,
    unsigned contact_count_size, unsigned object_a, unsigned object_b,
    D diag) {
    if (object_a != NO_OBJECT_INDEX) {
        DIAG_ASSERT4(diag, object_a < contact_count_size,
                    static_cast<float>(object_a),
                    static_cast<float>(contact_count_size), 0.0f, 0.0f);
        if (object_a < contact_count_size) {
            compute::atomic_add(contact_count + object_a, 1u);
        }
    }
    if (object_b != NO_OBJECT_INDEX && object_b != object_a) {
        DIAG_ASSERT4(diag, object_b < contact_count_size,
                    static_cast<float>(object_b),
                    static_cast<float>(contact_count_size), 1.0f, 0.0f);
        if (object_b < contact_count_size) {
            compute::atomic_add(contact_count + object_b, 1u);
        }
    }
}

// A contact between two DYNAMIC vertices: both sides resolve through the same
// per-vertex object map.
template <typename D>
[[seam::device_fn]] inline void statistics_record_dynamic_contact(
    compute::atomic_uint_t *contact_count,
    unsigned contact_count_size,
    const unsigned *object_index, unsigned object_index_size,
    unsigned vertex_a, unsigned vertex_b, D diag) {
    if (!statistics_enabled(contact_count_size)) {
        return;
    }
    DIAG_ASSERT4(diag, vertex_a < object_index_size,
                static_cast<float>(vertex_a),
                static_cast<float>(object_index_size), 0.0f, 0.0f);
    DIAG_ASSERT4(diag, vertex_b < object_index_size,
                static_cast<float>(vertex_b),
                static_cast<float>(object_index_size), 1.0f, 0.0f);
    if (vertex_a >= object_index_size || vertex_b >= object_index_size) {
        return;
    }
    statistics_record_object_contact(contact_count, contact_count_size,
                                         object_index[vertex_a],
                                         object_index[vertex_b], diag);
}

// A contact between a DYNAMIC vertex and a rest-pose STATIC collider vertex.
//
// The two sides live in DISJOINT index spaces and resolve through DIFFERENT
// maps, which is why they are separate parameters here rather than one array
// indexed twice. The dynamic side comes first, and every caller must keep that
// order: the collision-mesh embeds run in both directions (a dynamic vertex
// against a collider face, and a collider vertex against a dynamic face) and
// the two therefore pass their own indices in opposite argument positions.
template <typename D>
[[seam::device_fn]] inline void statistics_record_dynamic_static_contact(
    compute::atomic_uint_t *contact_count,
    unsigned contact_count_size,
    const unsigned *object_index, unsigned object_index_size,
    const unsigned *static_object_index,
    unsigned static_object_index_size, unsigned dynamic_vertex,
    unsigned static_vertex, D diag) {
    if (!statistics_enabled(contact_count_size)) {
        return;
    }
    DIAG_ASSERT4(diag, dynamic_vertex < object_index_size,
                static_cast<float>(dynamic_vertex),
                static_cast<float>(object_index_size), 0.0f, 0.0f);
    DIAG_ASSERT4(diag, static_vertex < static_object_index_size,
                static_cast<float>(static_vertex),
                static_cast<float>(static_object_index_size), 1.0f, 0.0f);
    if (dynamic_vertex >= object_index_size ||
        static_vertex >= static_object_index_size) {
        return;
    }
    statistics_record_object_contact(
        contact_count, contact_count_size, object_index[dynamic_vertex],
        static_object_index[static_vertex], diag);
}

// A contact between a DYNAMIC vertex and an analytic primitive (a floor, a
// sphere), which belongs to no object, so only one side is charged.
template <typename D>
[[seam::device_fn]] inline void statistics_record_analytic_contact(
    compute::atomic_uint_t *contact_count,
    unsigned contact_count_size,
    const unsigned *object_index, unsigned object_index_size,
    unsigned dynamic_vertex, D diag) {
    if (!statistics_enabled(contact_count_size)) {
        return;
    }
    DIAG_ASSERT4(diag, dynamic_vertex < object_index_size,
                static_cast<float>(dynamic_vertex),
                static_cast<float>(object_index_size), 0.0f, 0.0f);
    if (dynamic_vertex >= object_index_size) {
        return;
    }
    statistics_record_object_contact(contact_count, contact_count_size,
                                         object_index[dynamic_vertex],
                                         NO_OBJECT_INDEX, diag);
}
