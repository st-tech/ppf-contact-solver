// File: aabb.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::thread]]` is the address space of a reference parameter,
// which MSL requires on every reference and pointer type.
//
// No include of its own. `AABB` and `Vec3f` arrive from whatever declares them
// for the backend that is compiling, which is data.hpp under nvcc and on the
// host and the shader prologue's aliases under MSL.
//
// Every bound here is a POSITION, and the margin is added to the coordinate
// directly. `aabb_min` and `aabb_max` are templates, so one definition orders
// whichever scalar a box carries.

template <class T>
[[seam::device_fn]] inline T aabb_min(T a, T b) {
    return a < b ? a : b;
}

template <class T>
[[seam::device_fn]] inline T aabb_max(T a, T b) {
    return a > b ? a : b;
}

[[seam::device_fn]] inline AABB aabb_join(const AABB &a,
                                             const AABB &b) {
    AABB result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        result.min[dimension] =
            aabb_min(a.min[dimension], b.min[dimension]);
        result.max[dimension] =
            aabb_max(a.max[dimension], b.max[dimension]);
    }
    result.active = a.active && b.active;
    return result;
}

[[seam::device_fn]] inline AABB aabb_merge_active(
    const AABB &a, const AABB &b) {
    if (a.active && b.active) {
        AABB result;
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            result.min[dimension] =
                aabb_min(a.min[dimension], b.min[dimension]);
            result.max[dimension] =
                aabb_max(a.max[dimension], b.max[dimension]);
        }
        result.active = true;
        return result;
    }
    if (a.active) {
        return a;
    }
    if (b.active) {
        return b;
    }
    AABB result = a;
    result.active = false;
    return result;
}

[[seam::device_fn]] inline AABB aabb_make_triangle(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, float margin_float) {
    AABB result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        result.min[dimension] =
            aabb_min(x0[dimension],
                         aabb_min(x1[dimension], x2[dimension]));
        result.max[dimension] =
            aabb_max(x0[dimension],
                         aabb_max(x1[dimension], x2[dimension]));
    }
    result.active = true;
    if (margin_float != 0.0f) {
        const float margin(margin_float);
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            result.min[dimension] -= margin;
            result.max[dimension] += margin;
        }
    }
    return result;
}

[[seam::device_fn]] inline AABB aabb_make_edge(
    const Vec3f &x0, const Vec3f &x1,
    float margin_float) {
    AABB result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        result.min[dimension] =
            aabb_min(x0[dimension], x1[dimension]);
        result.max[dimension] =
            aabb_max(x0[dimension], x1[dimension]);
    }
    result.active = true;
    if (margin_float != 0.0f) {
        const float margin(margin_float);
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            result.min[dimension] -= margin;
            result.max[dimension] += margin;
        }
    }
    return result;
}

[[seam::device_fn]] inline AABB aabb_make_point(
    const Vec3f &x, float margin_float) {
    const float margin(margin_float);
    AABB result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        result.min[dimension] = x[dimension] - margin;
        result.max[dimension] = x[dimension] + margin;
    }
    result.active = true;
    return result;
}

[[seam::device_fn]] inline AABB aabb_make_swept_triangle(
    const Vec3f &x00, const Vec3f &x01,
    const Vec3f &x02, const Vec3f &x10,
    const Vec3f &x11, const Vec3f &x12,
    float extrapolate, float margin_float) {
    const float scale(extrapolate);
    const float margin(margin_float);
    AABB result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        const float z10 =
            x00[dimension] + scale * (x10[dimension] - x00[dimension]);
        const float z11 =
            x01[dimension] + scale * (x11[dimension] - x01[dimension]);
        const float z12 =
            x02[dimension] + scale * (x12[dimension] - x02[dimension]);
        const float minimum0 =
            aabb_min(x00[dimension],
                         aabb_min(x01[dimension], x02[dimension]));
        const float maximum0 =
            aabb_max(x00[dimension],
                         aabb_max(x01[dimension], x02[dimension]));
        const float minimum1 =
            aabb_min(z10, aabb_min(z11, z12));
        const float maximum1 =
            aabb_max(z10, aabb_max(z11, z12));
        result.min[dimension] = aabb_min(minimum0, minimum1) - margin;
        result.max[dimension] = aabb_max(maximum0, maximum1) + margin;
    }
    result.active = true;
    return result;
}

[[seam::device_fn]] inline AABB aabb_make_swept_edge(
    const Vec3f &x00, const Vec3f &x01,
    const Vec3f &x10, const Vec3f &x11,
    float extrapolate, float margin_float) {
    const float scale(extrapolate);
    const float margin(margin_float);
    AABB result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        const float z10 =
            x00[dimension] + scale * (x10[dimension] - x00[dimension]);
        const float z11 =
            x01[dimension] + scale * (x11[dimension] - x01[dimension]);
        result.min[dimension] =
            aabb_min(aabb_min(x00[dimension], x01[dimension]),
                         aabb_min(z10, z11)) -
            margin;
        result.max[dimension] =
            aabb_max(aabb_max(x00[dimension], x01[dimension]),
                         aabb_max(z10, z11)) +
            margin;
    }
    result.active = true;
    return result;
}

[[seam::device_fn]] inline AABB aabb_make_swept_point(
    const Vec3f &x0, const Vec3f &x1,
    float extrapolate, float margin_float) {
    const float scale(extrapolate);
    const float margin(margin_float);
    AABB result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        const float z1 =
            x0[dimension] + scale * (x1[dimension] - x0[dimension]);
        result.min[dimension] = aabb_min(x0[dimension], z1) - margin;
        result.max[dimension] = aabb_max(x0[dimension], z1) + margin;
    }
    result.active = true;
    return result;
}

[[seam::device_fn]] inline bool aabb_overlap(
    const AABB &a, const AABB &b) {
    if (!a.active || !b.active) {
        return false;
    }
    return (a.min[0] <= b.max[0] && a.max[0] >= b.min[0]) &&
           (a.min[1] <= b.max[1] && a.max[1] >= b.min[1]) &&
           (a.min[2] <= b.max[2] && a.max[2] >= b.min[2]);
}

// THE BROAD-PHASE INFLATION, and moving it here CLOSES A FORK rather than
// merely converting a launcher. It composes a candidate-set decision from two
// authored lengths: the box must reach at least as far as the narrow phase can,
// and the backends must agree or one of them generates a different candidate
// set from the same scene. It was spelled in the CPU shim and again in the CUDA
// orchestrator's three leaf launchers, which is two statements of one rule.
[[seam::device_fn]] inline float aabb_leaf_margin(float ghat,
                                                      float offset) {
    return 0.5f * ghat + offset;
}

// ONE LEAF'S BOX, per primitive arity. Each reads the leaf's primitive out of
// the tree's own node array and inflates the swept box by that primitive's
// material margin.
//
// THE NODE ENTRY IS BIASED BY ONE and that is what keeps `nodes` a BASE POINTER
// rather than a `[[seam::indices]]` list: a slot list is read and checked
// against the bound as it stands, and `nodes[2 * leaf]` is the primitive index
// plus one, so the check would test the biased number. The three arrays it then
// addresses are base pointers for the same reason. The hand-written launcher
// checked none of them either, so this conversion neither adds nor loses a
// bound; what it removes is a mirror pair.
//
// THE BOX IS THE RETURN VALUE, so it reaches memory as a scatter at the thread
// index, which is exactly the `aabb[leaf]` the launcher wrote.
// The three entry points. `AABB` is `alignas(32)` and reaches a record as a
// `[[seam::pod(32)]]` pointee, which the generator permitted only from
// 2026-08-23: its alignment assert was an EQUALITY against four until the three
// arenas were read and found to serve more.
[[seam::entry(leaf, aabb)]]
[[seam::device_fn]] inline AABB aabb_leaf_face(
    const Vec3f *x0, const Vec3f *x1,
    float extrapolate, const Vec3u *face,
    const FaceProp *prop,
    const FaceParam *params,
    const unsigned *nodes, unsigned leaf) {
    const unsigned primitive = nodes[2 * leaf] - 1u;
    const Vec3u f = face[primitive];
    const FaceParam p = params[prop[primitive].param_index];
    // EVERY ELEMENT IS COPIED INTO THREAD SPACE BEFORE IT IS PASSED, and that
    // is a Metal requirement rather than a style: an array reached through a
    // `[[seam::device]]` base pointer yields a DEVICE-space lvalue, and MSL
    // cannot bind one to the thread-space reference the box builders above
    // take. The copy is what the generator already emits for a
    // `[[seam::gather]]` parameter, so it costs nothing, and these stay base
    // pointers for the reasons stated above: `nodes` is biased by one and
    // `params[prop[i].param_index]` is a two-level indirection.
    const Vec3f x0_a = x0[f[0]];
    const Vec3f x0_b = x0[f[1]];
    const Vec3f x0_c = x0[f[2]];
    const Vec3f x1_a = x1[f[0]];
    const Vec3f x1_b = x1[f[1]];
    const Vec3f x1_c = x1[f[2]];
    return aabb_make_swept_triangle(x0_a, x0_b, x0_c, x1_a, x1_b, x1_c,
                                    extrapolate,
                                    aabb_leaf_margin(p.ghat, p.offset));
}

[[seam::entry(leaf, aabb)]]
[[seam::device_fn]] inline AABB aabb_leaf_edge(
    const Vec3f *x0, const Vec3f *x1,
    float extrapolate, const Vec2u *edge,
    const EdgeProp *prop,
    const EdgeParam *params,
    const unsigned *nodes, unsigned leaf) {
    const unsigned primitive = nodes[2 * leaf] - 1u;
    const Vec2u e = edge[primitive];
    const EdgeParam p = params[prop[primitive].param_index];
    // Thread-space copies, for the reason `aabb_leaf_face` above states.
    const Vec3f x0_a = x0[e[0]];
    const Vec3f x0_b = x0[e[1]];
    const Vec3f x1_a = x1[e[0]];
    const Vec3f x1_b = x1[e[1]];
    return aabb_make_swept_edge(x0_a, x0_b, x1_a, x1_b,
                                    extrapolate,
                                    aabb_leaf_margin(p.ghat, p.offset));
}

[[seam::entry(leaf, aabb)]]
[[seam::device_fn]] inline AABB aabb_leaf_vertex(
    const Vec3f *x0, const Vec3f *x1,
    float extrapolate, const VertexProp *prop,
    const VertexParam *params,
    const unsigned *nodes, unsigned leaf) {
    const unsigned primitive = nodes[2 * leaf] - 1u;
    const VertexParam p = params[prop[primitive].param_index];
    // Thread-space copies, for the reason `aabb_leaf_face` above states.
    const Vec3f x0_i = x0[primitive];
    const Vec3f x1_i = x1[primitive];
    return aabb_make_swept_point(x0_i, x1_i, extrapolate,
                                     aabb_leaf_margin(p.ghat, p.offset));
}

// ONE LEAF'S ACTIVE FLAG, from the collision-window mask. The box itself is
// untouched: only the flag the broad phase tests is rewritten, which is why
// this is a non-const gather rather than a scatter.
//
// THE MASK IS `unsigned` AND NOT A BYTE, because a buffer pointee is one of
// float, int or unsigned. `nodes` is a base pointer for the reason the three
// leaf boxes above give: its entry is the primitive index PLUS ONE.
[[seam::device_fn]] inline void aabb_leaf_active(
    const unsigned *active,
    const unsigned *nodes, unsigned leaf,
    AABB &box) {
    box.active = active[nodes[2 * leaf] - 1u] != 0u;
}

[[seam::entry(count, leaf)]] void aabb_leaf_active(
    const unsigned *active,
    const unsigned *nodes,
    unsigned leaf,
    AABB *aabb,
    unsigned count);

// THE ASSEMBLY QUERY BOXES, and they come in PAIRS because the mask is a
// per-DISPATCH configuration rather than a per-element fact.
//
// A scene that authored no collision window has no mask at all, and testing
// `active != nullptr` per element would be one way for a body to find that out.
// That test is a configuration the DRIVER already knows: a dispatch decision
// belongs to the driver rather than to a kernel body, and a record field is a
// handle with no spelling for absent, so the answer is two entry points over
// one body and the driver picks. The masked form is a thin wrapper, so the box
// itself is stated once.
//
// The bodies below take `prop` and `params` as BASE POINTERS, because
// `params[prop[i].param_index]` is a two-level indirection: a record carries one
// index list and it is spent on neither of them.
[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_point_contact_query(
    const Vec3f *x,
    const VertexProp *prop,
    const VertexParam *params, unsigned element) {
    const VertexParam p = params[prop[element].param_index];
    // A thread-space copy, for the reason `aabb_leaf_face` above states.
    const Vec3f x_i = x[element];
    return aabb_make_point(x_i,
                               aabb_leaf_margin(p.ghat, p.offset));
}

[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_point_contact_query_masked(
    const Vec3f *x,
    const VertexProp *prop,
    const VertexParam *params,
    const unsigned *active, unsigned element) {
    AABB box = aabb_point_contact_query(x, prop, params, element);
    if (active[element] == 0u) {
        box.active = false;
    }
    return box;
}

[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_edge_contact_query(
    const Vec3f *x, const Vec2u *edge,
    const EdgeProp *prop,
    const EdgeParam *params, unsigned element) {
    const Vec2u e = edge[element];
    const EdgeParam p = params[prop[element].param_index];
    // Thread-space copies, for the reason `aabb_leaf_face` above states.
    const Vec3f x_a = x[e[0]];
    const Vec3f x_b = x[e[1]];
    return aabb_make_edge(x_a, x_b,
                              aabb_leaf_margin(p.ghat, p.offset));
}

[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_edge_contact_query_masked(
    const Vec3f *x, const Vec2u *edge,
    const EdgeProp *prop,
    const EdgeParam *params,
    const unsigned *active, unsigned element) {
    AABB box = aabb_edge_contact_query(x, edge, prop, params, element);
    if (active[element] == 0u) {
        box.active = false;
    }
    return box;
}

// THE INTERSECTION SCAN'S QUERY BOXES, in the same masked and unmasked pair the
// assembly query boxes above come in: the mask is a per-DISPATCH configuration,
// so the driver picks the entry and no body carries a null test.
//
// A SCAN box carries no margin at all for an edge, and a grain's own contact
// offset for a point: it is the box the final penetration gate walks, and
// inflating it would report a pair that is not touching.
//
// THE LINE SEARCH'S SWEPT BOXES ARE NOT HERE, by design. Each of its six sweeps
// builds the query box INSIDE the kernel that traverses with it, from the
// primitive's own two poses and its own contact margin. A pre-pass writing an
// array of them would only serve a broad phase that materializes candidate
// pairs, and this traversal materializes none.
[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_edge_scan_query(
    const Vec3f *vert, const Vec2u *edge,
    unsigned element) {
    const Vec2u e = edge[element];
    // Thread-space copies, for the reason `aabb_leaf_face` above states.
    const Vec3f vert_a = vert[e[0]];
    const Vec3f vert_b = vert[e[1]];
    return aabb_make_edge(vert_a, vert_b, 0.0f);
}

[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_edge_scan_query_masked(
    const Vec3f *vert, const Vec2u *edge,
    const unsigned *active, unsigned element) {
    AABB box = aabb_edge_scan_query(vert, edge, element);
    if (active[element] == 0u) {
        box.active = false;
    }
    return box;
}

[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_vertex_scan_query(
    const Vec3f *vert,
    const VertexProp *prop,
    const VertexParam *params, unsigned element) {
    // A thread-space copy, for the reason `aabb_leaf_face` above states.
    const Vec3f vert_i = vert[element];
    return aabb_make_point(vert_i,
                               params[prop[element].param_index].offset);
}

[[seam::entry(element, out)]]
[[seam::device_fn]] inline AABB aabb_vertex_scan_query_masked(
    const Vec3f *vert,
    const VertexProp *prop,
    const VertexParam *params,
    const unsigned *active, unsigned element) {
    AABB box = aabb_vertex_scan_query(vert, prop, params, element);
    if (active[element] == 0u) {
        box.active = false;
    }
    return box;
}

// ONE LEVEL OF THE BOTTOM-UP MERGE, and it is here to record that a WRITE AT A
// DATA-DRIVEN SLOT has a lane after all.
//
// Section 12b's blocker 1 says `buffer[list[i]] = ...` is not expressible,
// because `[[seam::through]]` must be const and `[[seam::scatter]]` writes at
// the THREAD index. Both are true and neither is the only way a body reaches
// memory: a plain `[[seam::device]]` pointer with NO access attribute is a BASE
// POINTER, the body is handed the array itself, and nothing constrains which
// element it writes. `main/dirichlet.kernel.cpp` already writes its force that
// way; the only difference here is that the index comes from another buffer.
//
// WHAT A BASE POINTER GIVES UP IS THE BOUND, and this pass never had one: the
// hand-written launcher indexed `aabb[node]` unchecked. What it does NOT give
// up is the race discipline, which lives in the kernel table's `Scatter` rather
// than in the declaration. This row is `Disjoint` by construction: a level's
// nodes are distinct and their children sit strictly deeper, so no two threads
// name one box and the range may be cut anywhere.
//
// A LEAF AMONG THE LEVEL IS LEFT ALONE, which is what the zero second child
// means, and the caller relies on it: the leaf boxes were written by the leaf
// pass and merging them again would bound them by themselves.
[[seam::entry(element)]]
[[seam::device_fn]] inline void aabb_merge_level(
    const unsigned *level,
    const unsigned *nodes, AABB *aabb,
    unsigned element) {
    const unsigned node = level[element];
    const unsigned second = nodes[2 * node + 1];
    if (second == 0u) {
        return;
    }
    const unsigned left = nodes[2 * node] - 1u;
    // BOTH CHILD BOXES ARE COPIED INTO THREAD SPACE, for the reason
    // `aabb_leaf_face` above states: `aabb` is a base pointer, so `aabb[left]`
    // is a DEVICE-space lvalue that MSL cannot bind to the thread-space
    // reference `aabb_merge_active` takes. Both reads still precede the write,
    // exactly as evaluating them as arguments did, and a level's children sit
    // strictly deeper than its own nodes, so neither is the box this thread
    // writes.
    const AABB left_box = aabb[left];
    const AABB right_box = aabb[second - 1u];
    aabb[node] = aabb_merge_active(left_box, right_box);
}
