// File: reduce_bounds.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the four forms the four targets read.
//
// THE SCENE'S CENTROID BOUNDS, ON THE DEVICE. The three centroid arrays are
// folded where they already live, once per tree per step, rather than being
// downloaded and folded on the host.
//
// A BLOCK REDUCTION WITHOUT THE COOPERATIVE LAYER, the same shape and the same
// argument as `scan_levels.kernel.cpp`. Reducing within a block through shared
// memory and a barrier and then combining the blocks with a float atomic min
// and max has no host spelling for either the barrier or those atomics, so the
// cooperative layer comes out and one thread walks a block.
//
// THE RESULT IS UNAFFECTED, AND MORE STRONGLY THAN IT IS FOR A SUM. Minimum and
// maximum are associative AND exact on floats: no rounding happens at all, so
// every association order gives the same bits. This is not the float-sum case,
// where an order change is a different answer.
//
// NO SENTINEL, and that is deliberate. `common.hpp` redefines `FLT_MAX` to
// `1.0e8f` and `FLT_MIN` to a NEGATIVE `-1.0e8f`, so a body that seeded from
// either would be reading a tree-specific constant it does not include. Each
// block seeds from its own FIRST element instead, which needs no constant and
// is exact for any input; the driver never dispatches a block with no elements.

// One block's bounds, over the three centroid arrays.
[[seam::device_fn]] inline void bounds_leaf(
    const float *cx, const float *cy,
    const float *cz, unsigned count, unsigned block_size,
    float *out, unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    float lo[3];
    float hi[3];
    lo[0] = hi[0] = cx[begin];
    lo[1] = hi[1] = cy[begin];
    lo[2] = hi[2] = cz[begin];
    for (unsigned i = 1; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        lo[0] = fmath::min(lo[0], cx[at]);
        hi[0] = fmath::max(hi[0], cx[at]);
        lo[1] = fmath::min(lo[1], cy[at]);
        hi[1] = fmath::max(hi[1], cy[at]);
        lo[2] = fmath::min(lo[2], cz[at]);
        hi[2] = fmath::max(hi[2], cz[at]);
    }
    for (unsigned d = 0; d < 3; ++d) {
        out[6u * block_index + d] = lo[d];
        out[6u * block_index + 3u + d] = hi[d];
    }
}

// One block's bounds, over a run of the level below's six-float records.
//
// `source` AND `destination` ARE DIFFERENT SPANS, never the same one: a thread
// reads `block_size` records and writes one, so an in-place pass would have a
// thread reading a record another thread has already replaced.
[[seam::device_fn]] inline void bounds_merge(
    const float *source, unsigned count, unsigned block_size,
    float *destination, unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    float lo[3];
    float hi[3];
    for (unsigned d = 0; d < 3; ++d) {
        lo[d] = source[6u * begin + d];
        hi[d] = source[6u * begin + 3u + d];
    }
    for (unsigned i = 1; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        for (unsigned d = 0; d < 3; ++d) {
            lo[d] = fmath::min(lo[d], source[6u * at + d]);
            hi[d] = fmath::max(hi[d], source[6u * at + 3u + d]);
        }
    }
    for (unsigned d = 0; d < 3; ++d) {
        destination[6u * block_index + d] = lo[d];
        destination[6u * block_index + 3u + d] = hi[d];
    }
}

[[seam::entry(blocks, block_index)]] void bounds_leaf(
    const float *cx, const float *cy,
    const float *cz, unsigned count, unsigned block_size,
    float *out,
    unsigned block_index,
    unsigned blocks);

[[seam::entry(blocks, block_index)]] void bounds_merge(
    const float *source, unsigned count, unsigned block_size,
    float *destination,
    unsigned block_index,
    unsigned blocks);
