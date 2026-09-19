// File: collision_window.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. `[[seam::device_fn]]` is the execution
// space and `[[seam::device]]` is the address space of a pointer into a buffer,
// which MSL requires.
//
// COLLISION WINDOWS: which primitives are eligible for contact at this instant.
//
// A vertex may carry a list of half-open time windows during which it collides.
// The three bodies here are the whole rule: whether one vertex is live now,
// how that propagates to the faces and edges it belongs to, and how a leaf of
// the bounding-volume hierarchy is told about it. What stays in a caller is the
// walk over the arrays and the re-propagation of the interior boxes.

#include "../lbvh/lbvh.kernel.cpp"

// Whether one vertex collides at `time`.
//
// `windows` points at THIS vertex's list, `count` pairs of (start, end) laid
// out consecutively; the caller owns the stride into the shared table, because
// the table's row width is a capacity the host chose rather than a property of
// the rule.
//
// NO WINDOWS MEANS ALWAYS ACTIVE, which is the whole reason the count is read
// before the loop rather than folded into it: an empty list is "unrestricted",
// not "never", and a scene that carries no windows at all must collide exactly
// as it did before the feature existed.
//
// The interval is HALF-OPEN, `start <= time < end`, so two abutting windows
// admit the vertex exactly once at their shared instant rather than twice or
// not at all.
[[seam::device_fn]] inline bool
collision_window_active(const float *windows,
                            unsigned count, float time) {
    bool active = (count == 0);
    for (unsigned w = 0; w < count; ++w) {
        const float start = windows[2 * w];
        const float end = windows[2 * w + 1];
        if (time >= start && time < end) {
            active = true;
            break;
        }
    }
    return active;
}

// A face or an edge collides when ANY of its vertices does.
//
// ANY rather than ALL, and the direction matters: an element half of whose
// vertices are live still has geometry in play, and excluding it would open a
// hole in the barrier for the live half. The cost of the choice is that an
// element stays eligible slightly longer than its least active vertex, which
// errs toward assembling a contact that is not needed rather than missing one
// that is.
[[seam::device_fn]] inline bool collision_face_active(bool a, bool b,
                                                         bool c) {
    return a || b || c;
}

[[seam::device_fn]] inline bool collision_edge_active(bool a, bool b) {
    return a || b;
}

// The eligibility of the primitive a hierarchy leaf stands for.
//
// The leaf's box carries the flag, so an inactive primitive drops out of every
// traversal that reaches it once the interior boxes are re-merged. The
// primitive index is read back through the shared node format rather than by
// subtracting one here.
[[seam::device_fn]] inline bool
collision_leaf_active(const unsigned char *active,
                          unsigned first_child) {
    return active[lbvh_leaf_primitive(first_child)] != 0u;
}
