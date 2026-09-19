// File: reduce_scalar.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts.
//
// THE MINIMUM AND THE MAXIMUM OF A FLOAT ARRAY, ON THE DEVICE. Nine sites want
// one per Newton iteration: the line search's per-primitive time of impact, the
// step's maximum displacement and reach, and the strain limiter's ratios. Each
// folds with a device reduction and moves ONE FLOAT, which is the distinction
// that decides where a fold belongs: transporting a scalar is transport, and
// downloading an ARRAY to run arithmetic on it is a relocation.
//
// ONLY EXACT REDUCTIONS LIVE HERE, and the omission is deliberate. A float
// minimum and maximum are associative AND exact: no rounding happens, so every
// association order gives the same bits and moving the fold cannot move a
// result. A float SUM is not like that, and the driver's `reduce::sum` states
// the shape its callers depend on, so converting it is a separate change with
// its own argument. The unsigned reductions below are exact on the same terms.
//
// NO SENTINEL: each block seeds from its own first element, so the bodies need
// no constant, and `common.hpp`'s tree-specific `FLT_MAX` of `1.0e8f` and
// NEGATIVE `FLT_MIN` never come into it.

[[seam::device_fn]] inline void reduce_min_leaf(
    const float *values, unsigned count, unsigned block_size,
    float *out, unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    float best = values[begin];
    for (unsigned i = 1; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        best = fmath::min(best, values[at]);
    }
    out[block_index] = best;
}

[[seam::device_fn]] inline void reduce_max_leaf(
    const float *values, unsigned count, unsigned block_size,
    float *out, unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    float best = values[begin];
    for (unsigned i = 1; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        best = fmath::max(best, values[at]);
    }
    out[block_index] = best;
}

// THE SMALLEST WORD AND THE TOTAL OF AN UNSIGNED ARRAY. Both are exact for the
// reason the float minimum is: an unsigned minimum does no arithmetic, and the
// total below cannot wrap, so every association order gives the same answer.
// The driver uses them where the answer is one index or one count in an array
// it would otherwise download.
//
// THE TOTAL IS TWO WORDS, the low and high halves of a 64-bit count, because
// the seam carries no 64-bit buffer element and Metal has no 64-bit atomic to
// fall back on. A carry out of the low word is the unsigned wrap
// `low < previous`. The high word cannot wrap: the driver hands at most
// `2^32 - 1` words, each below `2^32`, so the total stays below `2^64`.

[[seam::device_fn]] inline void reduce_min_u32_leaf(
    const unsigned *values, unsigned count, unsigned block_size,
    unsigned *out, unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    unsigned best = values[begin];
    for (unsigned i = 1; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        const unsigned value = values[at];
        if (value < best) {
            best = value;
        }
    }
    out[block_index] = best;
}

// One block's total of single words, written as a (low, high) pair.
[[seam::device_fn]] inline void reduce_sum_u32_leaf(
    const unsigned *values, unsigned count, unsigned block_size,
    unsigned *out, unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    unsigned low = 0u;
    unsigned high = 0u;
    for (unsigned i = 0; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        const unsigned previous = low;
        low += values[at];
        if (low < previous) {
            high += 1u;
        }
    }
    out[2u * block_index] = low;
    out[2u * block_index + 1u] = high;
}

// One block's total of (low, high) pairs, written as a pair.
//
// `source` AND `destination` ARE DIFFERENT SPANS, as `bounds_merge` requires
// and for the same reason: a thread reads `block_size` pairs and writes one, so
// an in-place pass would read a pair another thread has already replaced.
[[seam::device_fn]] inline void reduce_sum_wide_merge(
    const unsigned *source, unsigned count, unsigned block_size,
    unsigned *destination, unsigned block_index) {
    const unsigned begin = block_index * block_size;
    if (begin >= count) {
        return;
    }
    unsigned low = 0u;
    unsigned high = 0u;
    for (unsigned i = 0; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        const unsigned previous = low;
        low += source[2u * at];
        if (low < previous) {
            high += 1u;
        }
        high += source[2u * at + 1u];
    }
    destination[2u * block_index] = low;
    destination[2u * block_index + 1u] = high;
}

[[seam::entry(blocks, block_index)]] void reduce_min_leaf(
    const float *values, unsigned count, unsigned block_size,
    float *out,
    unsigned block_index,
    unsigned blocks);

[[seam::entry(blocks, block_index)]] void reduce_max_leaf(
    const float *values, unsigned count, unsigned block_size,
    float *out,
    unsigned block_index,
    unsigned blocks);

[[seam::entry(blocks, block_index)]] void reduce_min_u32_leaf(
    const unsigned *values, unsigned count, unsigned block_size,
    unsigned *out,
    unsigned block_index,
    unsigned blocks);

[[seam::entry(blocks, block_index)]] void reduce_sum_u32_leaf(
    const unsigned *values, unsigned count, unsigned block_size,
    unsigned *out,
    unsigned block_index,
    unsigned blocks);

[[seam::entry(blocks, block_index)]] void reduce_sum_wide_merge(
    const unsigned *source, unsigned count, unsigned block_size,
    unsigned *destination,
    unsigned block_index,
    unsigned blocks);
