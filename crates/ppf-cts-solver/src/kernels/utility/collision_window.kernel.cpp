// File: collision_window.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++, no preprocessor conditional and no macro
// of its own; ppf-cts-compute/seam/kernelgen.py renders it for the four targets.
//
// THE COLLISION WINDOW MASKS, which decide whether an element may collide at
// the current time. `main.cu:2400`'s `refresh_collision_active` runs exactly
// these three passes as three DISPATCHES; this driver ran them as a host loop
// over every vertex, face and edge and then uploaded three arrays, which is
// the rule (1a-0) shape the port is not allowed to have.
//
// THE WINDOW TABLE IS READ, NEVER COMPUTED ON. Both tests are comparisons
// against authored bounds, so nothing here does arithmetic that could move a
// boundary; a group with NO interval is always collidable, which is what the
// absence of a window means rather than a fallback.

// One vertex: active when its group authored no window, or when the clock is
// inside one of the group's `[start, end)` intervals.
//
// `stride` is `MAX_COLLISION_WINDOWS * 2`, the row width of `windows`, passed
// rather than spelled so the neutral body carries no build constant.
[[seam::entry(element)]]
[[seam::device_fn]] inline void collision_window_vertex(
    const unsigned *vertex_group,
    const float *windows,
    const unsigned *window_count, unsigned stride, float time,
    unsigned *vertex_active, unsigned element) {
    const unsigned group = vertex_group[element];
    const unsigned count = window_count[group];
    unsigned active = count == 0u ? 1u : 0u;
    for (unsigned w = 0; w < count; ++w) {
        const float start = windows[group * stride + w * 2u];
        const float end = windows[group * stride + w * 2u + 1u];
        if (time >= start && time < end) {
            active = 1u;
            break;
        }
    }
    vertex_active[element] = active;
}

// A face is collidable when ANY of its three vertices is, which is the
// reference's `d_va[f[0]] || d_va[f[1]] || d_va[f[2]]`.
[[seam::entry(element)]]
[[seam::device_fn]] inline void collision_window_face(
    const unsigned *vertex_active,
    const unsigned *face,
    unsigned *face_active, unsigned element) {
    unsigned active = 0u;
    for (unsigned k = 0; k < 3u; ++k) {
        if (vertex_active[face[3u * element + k]] != 0u) {
            active = 1u;
        }
    }
    face_active[element] = active;
}

// An edge is collidable when either of its two vertices is.
[[seam::entry(element)]]
[[seam::device_fn]] inline void collision_window_edge(
    const unsigned *vertex_active,
    const unsigned *edge,
    unsigned *edge_active, unsigned element) {
    unsigned active = 0u;
    for (unsigned k = 0; k < 2u; ++k) {
        if (vertex_active[edge[2u * element + k]] != 0u) {
            active = 1u;
        }
    }
    edge_active[element] = active;
}
