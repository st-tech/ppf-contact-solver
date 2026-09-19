// File: aabb_traversal.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read. The two facts a
// backend cannot infer are written as C++ attributes: `[[seam::device_fn]]` is
// the execution space, and `[[seam::device]]` and `[[seam::thread]]` are the
// address spaces MSL requires on every pointer and reference type; CUDA and the
// host have one address space and are handed the same declarations with it
// removed.
//
// The two tree arrays are `[[seam::device]]` because they are the whole BVH,
// one allocation every thread reads. Everything else the query touches is this
// thread's own: the visitor, the query primitive, the diagnostic handle and the
// traversal stack.
//
// No include of its own. `AABB` arrives from whatever declares it for the
// backend that is compiling, which is data.hpp under nvcc and on the host and
// the shader prologue's aliases under MSL.
//
// DIAG_ASSERT4 IS NOT A PLATFORM BRANCH AND STAYS A MACRO. It records __FILE__
// and __LINE__ of the failing check, which no function can read for its caller,
// and it is defined outside this file by each backend's diagnostic channel:
// diagnostics/diagnostics.hpp under nvcc, metal/diagnostics.mm for the
// shader, and a `((void)0)` stub in the host oracle. Nothing about its spelling
// differs between backends here.

// THE PRUNE TEST LIVES NEXT DOOR. `aabb_overlap` is in `aabb.kernel.cpp`, and
// this file is rendered and compiled ON ITS OWN by `entry-check`, so the symbol
// has to arrive by a quoted include rather than by whatever the host
// translation unit happened to pull in first.
#include "aabb.kernel.cpp"

enum : unsigned { AABB_MAX_QUERY = 128u };

template <typename F, typename T, typename D>
[[seam::device_fn]] inline unsigned aabb_query(
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root, F &op,
    const T &query, D diag) {
    unsigned stack[AABB_MAX_QUERY];
    unsigned count = 0;
    unsigned head = 0;
    if (node_count == 0) {
        return 0;
    }
    DIAG_ASSERT4(diag, root < node_count, static_cast<float>(root),
                static_cast<float>(node_count), 0.0f, 0.0f);
    if (root >= node_count) {
        return 0;
    }
    stack[head++] = root;
    while (head) {
        const unsigned index = stack[--head];
        DIAG_ASSERT4(diag, index < node_count, static_cast<float>(index),
                    static_cast<float>(node_count), static_cast<float>(root),
                    static_cast<float>(head));
        if (index >= node_count) {
            break;
        }
        const AABB box = aabb[index];
        if (!op.test(box, query)) {
            continue;
        }
        const unsigned first = node[2 * index];
        const unsigned second = node[2 * index + 1];
        if (second == 0) {
            DIAG_ASSERT4(diag, first > 0, static_cast<float>(first),
                        static_cast<float>(index), 0.0f, 0.0f);
            if (first > 0 && op(first - 1)) {
                ++count;
            }
        } else {
            if (head + 2 >= AABB_MAX_QUERY) {
                DIAG_ASSERT4(diag, false, static_cast<float>(head),
                            static_cast<float>(AABB_MAX_QUERY),
                            static_cast<float>(index),
                            static_cast<float>(root));
                break;
            }
            DIAG_ASSERT4(diag, first > 0 && second > 0,
                        static_cast<float>(first), static_cast<float>(second),
                        static_cast<float>(index), 0.0f);
            if (first == 0 || second == 0) {
                break;
            }
            stack[head++] = first - 1;
            stack[head++] = second - 1;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// The broad phase as ONE DISPATCH, one thread per query box.
// ---------------------------------------------------------------------------

// The per-hit visitor, writing this thread's own slot.
//
// EACH QUERY OWNS `capacity` PAIRS AND WRITES NO OTHER THREAD'S. The shim this
// replaces packed sequentially against a count shared across a host chunk, so
// the pair list depended on where the chunk boundaries fell and the caller had
// to concatenate per-chunk buffers in ascending order to hide it. A per-query
// slot needs none of that: the layout is the same whatever the backend's
// partition is, which is what lets the partition BE the backend's.
//
// COUNTS PAST CAPACITY ON PURPOSE, the discipline `pair_cache_record` follows:
// the requirement is knowable even on the pass that could not meet it, so the
// caller sizes the retry instead of guessing.
struct AabbPairCollect {
    unsigned *out;
    unsigned base;
    unsigned capacity;
    unsigned count;
    unsigned query_index;

    // BOTH METHODS CARRY THE EXECUTION SPACE, and neither may go without it.
    // `test` calls `aabb_overlap`, which is `[[seam::device_fn]]`, and a member
    // function with no annotation is a HOST function to nvcc: the CUDA build
    // fails with "calling a __device__ function from a __host__ function".
    // Neither the Metal shader compiler nor the host oracle sees this, because
    // MSL has no host/device split and the host build compiles everything for
    // the host, so `--features cuda-abi` is the only gate that catches it.
    [[seam::device_fn]] bool test(const AABB &box,
                                  const AABB &q) const {
        return aabb_overlap(box, q);
    }

    [[seam::device_fn]] bool operator()(unsigned primitive) {
        if (count < capacity) {
            out[base + 2u * count] = query_index;
            out[base + 2u * count + 1u] = primitive;
        }
        ++count;
        return true;
    }
};

[[seam::device_fn]] inline unsigned aabb_query_pairs(
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const AABB *query, unsigned *out,
    unsigned capacity, DiagHandle diag, unsigned element) {
    const AABB box = query[element];
    if (!box.active) {
        return 0u;
    }
    AabbPairCollect collect{out, 2u * element * capacity, capacity, 0u,
                            element};
    aabb_query(node, node_count, aabb, root, collect, box, diag);
    // THE COUNT THIS QUERY WANTED, not the count it wrote. A value above
    // `capacity` is the retry size the caller needs and is not an error here.
    // RETURNED rather than written: the entry's one `[[seam::scatter]]` puts it
    // at this thread's index, which is the one shape a return value reaches
    // memory in.
    return collect.count;
}

// The entry.
//
// `out` IS A BASE POINTER AND `found` IS THE SCATTER, which is the seam's rule
// read correctly rather than worked around: a thread writes a RANGE of `out`,
// indexed by the `base` the body computes, and that is the base-pointer shape.
// What it RETURNS is one value at its own index, the count this query wanted,
// and that is the scatter. One call returns one value, so there is one scatter.
[[seam::entry(count, element)]] void aabb_query_pairs(
    const unsigned *node, unsigned node_count,
    const AABB *aabb, unsigned root,
    const AABB *query,
    unsigned *out,
    [[seam::scatter]] unsigned *found, unsigned capacity,
    DiagHandle diag, unsigned element,
    unsigned count);
