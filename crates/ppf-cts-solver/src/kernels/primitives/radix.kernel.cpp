// File: radix.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only one of the three
// compilers accepts. ppf-cts-compute/seam/kernelgen.py renders it into the .cu, the .metal
// and the .cpp that nvcc, the Metal shader compiler and a host C++ compiler
// read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space, and `[[seam::device]]` and
// `[[seam::threadgroup]]` are the address spaces of the pointer parameters. MSL
// requires an address space on every pointer, and it is the ONE distinction
// these bodies could not otherwise carry: the histogram lives in threadgroup
// memory and the keys live in device memory, and no other spelling in the file
// says so. CUDA and the host have a single address space and are handed the
// same declarations with it removed.
//
// No include of its own. The `compute::` lane operations and atomics arrive
// from whatever the backend compiles ahead of this body, which is data.hpp
// under nvcc and the shader prologue under MSL.
//
// THE COOPERATIVE BODIES HERE ARE DEVICE-ONLY, and the seam says so by
// omission. They name three operations with no single-thread meaning, a lane
// ballot (`compute::simd_ballot`), a lane-mask population count
// (`bits::popcount`) and a threadgroup barrier
// (`compute::threadgroup_barrier`). The nvcc prologue
// (ppf-cts-compute/cuda/seam_cuda.cuh) and the Metal prologue (kMslMacroSeam in
// ppf-cts-compute/metal/shader_compiler.mm) define all three; the host prologue
// (kernels/seam/seam_host.h) deliberately defines none of them, and states why.
// A `[[seam::cooperative]]` body renders only to the device targets, so no host
// translation unit reaches one; the host arm compiles the `[[seam::serial]]`
// twins below instead. A host build that did reach a cooperative body would
// stop on the undeclared name rather than compute something else.

[[seam::device_fn]] inline unsigned
radix_digit(unsigned key, unsigned shift, unsigned radix_mask) {
    return (key >> shift) & radix_mask;
}

[[seam::device_fn]] [[seam::cooperative]] inline void radix_histogram(
    const unsigned *keys, unsigned count, unsigned shift,
    unsigned radix_size, unsigned radix_mask,
    unsigned *block_histograms, unsigned block_count,
    compute::atomic_uint_t *local_histogram,
    unsigned thread_index, unsigned block_index, unsigned threads_per_group) {
    if (thread_index < radix_size) {
        compute::atomic_store(&local_histogram[thread_index], 0u);
    }
    compute::threadgroup_barrier();
    unsigned index = block_index * threads_per_group + thread_index;
    if (index < count) {
        unsigned digit = radix_digit(keys[index], shift, radix_mask);
        compute::atomic_add(&local_histogram[digit], 1u);
    }
    compute::threadgroup_barrier();
    if (thread_index < radix_size) {
        block_histograms[thread_index * block_count + block_index] =
            compute::atomic_load(&local_histogram[thread_index]);
    }
}

[[seam::device_fn]] [[seam::cooperative]] inline void radix_scatter(
    const unsigned *keys_in,
    const unsigned *values_in,
    unsigned *keys_out, unsigned *values_out,
    unsigned count, unsigned shift, unsigned radix_size, unsigned radix_mask,
    const unsigned *global_offsets, unsigned block_count,
    unsigned *local_offset,
    unsigned *warp_counts, unsigned thread_index,
    unsigned block_index, unsigned threads_per_group) {
    // THE WIDTH COMES FROM THE SEAM, NOT FROM A LITERAL HERE. A subgroup is 32
    // lanes on CUDA and on Metal and 64 on an AMD wave64 target, and this body
    // is the only one in the tree that ballots, so a literal here is the one
    // place a 64-lane group would be miscounted. `compute::simd_width` is a
    // compile-time constant on every target, so nothing is paid for reading it.
    const unsigned simd_width = compute::simd_width;
    const unsigned warp_count = threads_per_group / simd_width;
    const unsigned index =
        block_index * threads_per_group + thread_index;
    const unsigned lane = thread_index & (simd_width - 1u);
    const unsigned warp = thread_index / simd_width;
    if (thread_index < radix_size) {
        local_offset[thread_index] =
            global_offsets[thread_index * block_count + block_index];
    }
    unsigned key = 0, value = 0, digit = 0;
    bool valid = index < count;
    if (valid) {
        key = keys_in[index];
        value = values_in[index];
        digit = radix_digit(key, shift, radix_mask);
    }
    // THE MASK IS AS WIDE AS THE BALLOT, and that is why it is not `unsigned`.
    // `1u << lane` is undefined once `lane` reaches 32, so on a 64-lane target
    // the literal spelling loses the upper half of the group with no diagnostic.
    // The shifted one is typed from the ballot so the two always agree.
    const compute::ballot_t lane_mask =
        (compute::ballot_t(1) << lane) - compute::ballot_t(1);
    unsigned rank = 0;
    for (unsigned d = 0; d < radix_size; ++d) {
        const compute::ballot_t mask = compute::simd_ballot(valid && digit == d);
        if (valid && digit == d) {
            rank = bits::popcount(mask & lane_mask);
        }
        if (lane == 0) {
            warp_counts[warp * radix_size + d] = bits::popcount(mask);
        }
    }
    compute::threadgroup_barrier();
    if (thread_index < radix_size) {
        unsigned acc = 0;
        for (unsigned w = 0; w < warp_count; ++w) {
            unsigned at = w * radix_size + thread_index;
            unsigned c = warp_counts[at];
            warp_counts[at] = acc;
            acc += c;
        }
    }
    compute::threadgroup_barrier();
    if (valid) {
        unsigned destination = local_offset[digit] +
                               warp_counts[warp * radix_size + digit] + rank;
        keys_out[destination] = key;
        values_out[destination] = value;
    }
}

// ---------------------------------------------------------------------------
// THE SERIAL TWINS every cooperative body owes, which
// `ppf-cts-compute/seam/kernelgen.py` enforces by name.
//
// WHY THESE TWO NEED A TWIN WHERE THE FOLD DID NOT. A block SUM promises that
// every lane's value is folded in, which a shim running lanes in order over a
// persistent scratch keeps exactly, so `compute::block_sum` has an honest
// meaning on all three arms from one body. A BALLOT promises each lane a view of
// every OTHER lane's predicate AT THE SAME INSTANT, and lanes that run one after
// another never share an instant. There is no accumulate-in-order reading of it,
// which is why the fork is here and not there.
//
// LANE 0 DOES THE WHOLE BLOCK AND THE REST RETURN. That is the shape a serial
// twin takes on a shim whose lanes run 0 through width - 1: the work is the
// group's, so one lane does it and the others must not repeat it. It is a
// transcription of the cooperative body rather than a second algorithm, which
// is the limit a twin is held to: two algorithms under one name are a fork.

[[seam::device_fn]] [[seam::serial]] inline void radix_histogram(
    const unsigned *keys, unsigned count, unsigned shift,
    unsigned radix_size, unsigned radix_mask,
    unsigned *block_histograms, unsigned block_count,
    compute::atomic_uint_t *local_histogram,
    unsigned thread_index, unsigned block_index, unsigned threads_per_group) {
    if (thread_index != 0u) {
        return;
    }
    for (unsigned d = 0; d < radix_size; ++d) {
        compute::atomic_store(&local_histogram[d], 0u);
    }
    for (unsigned lane = 0; lane < threads_per_group; ++lane) {
        const unsigned index = block_index * threads_per_group + lane;
        if (index < count) {
            const unsigned digit = radix_digit(keys[index], shift, radix_mask);
            compute::atomic_add(&local_histogram[digit], 1u);
        }
    }
    for (unsigned d = 0; d < radix_size; ++d) {
        block_histograms[d * block_count + block_index] =
            compute::atomic_load(&local_histogram[d]);
    }
}

// THE STABLE COUNTING SORT THE BALLOT FORM COMPUTES. Walking the block once per
// digit in ascending lane order gives each element the rank the lane mask gives
// it there: the number of EARLIER lanes carrying the same digit. Stability is
// the property a radix pass is built on, so it is the one to preserve rather
// than the instruction sequence.
[[seam::device_fn]] [[seam::serial]] inline void radix_scatter(
    const unsigned *keys_in,
    const unsigned *values_in,
    unsigned *keys_out, unsigned *values_out,
    unsigned count, unsigned shift, unsigned radix_size, unsigned radix_mask,
    const unsigned *global_offsets, unsigned block_count,
    unsigned *local_offset,
    unsigned *warp_counts, unsigned thread_index,
    unsigned block_index, unsigned threads_per_group) {
    if (thread_index != 0u) {
        return;
    }
    for (unsigned d = 0; d < radix_size; ++d) {
        local_offset[d] = global_offsets[d * block_count + block_index];
        warp_counts[d] = 0u;
    }
    for (unsigned lane = 0; lane < threads_per_group; ++lane) {
        const unsigned index = block_index * threads_per_group + lane;
        if (index >= count) {
            continue;
        }
        const unsigned key = keys_in[index];
        const unsigned digit = radix_digit(key, shift, radix_mask);
        const unsigned destination = local_offset[digit] + warp_counts[digit];
        warp_counts[digit] += 1u;
        keys_out[destination] = key;
        values_out[destination] = values_in[index];
    }
}

// ---------------------------------------------------------------------------
// THE TWO ENTRY POINTS. Both are GROUP-shaped: a block's threads share the
// histogram and the ranking, which is what a group is for and what an element
// launch cannot express.
//
// `count` IS THE BLOCK COUNT, not the key count, because the guard is on the
// GROUP index. The keys are addressed by the body from the block index and its
// own lane, so the key count travels as an ordinary scalar.

[[seam::entry(groups, block_index)]] [[seam::group]] void radix_histogram(
    const unsigned *keys, unsigned count, unsigned shift,
    unsigned radix_size, unsigned radix_mask,
    unsigned *block_histograms, unsigned block_count,
    [[seam::scratch(256)]]
    compute::atomic_uint_t *local_histogram,
    [[seam::lane]] unsigned thread_index, unsigned block_index,
    [[seam::width]] unsigned threads_per_group,
    unsigned groups);

[[seam::entry(groups, block_index)]] [[seam::group]] void radix_scatter(
    const unsigned *keys_in,
    const unsigned *values_in,
    unsigned *keys_out, unsigned *values_out,
    unsigned count, unsigned shift, unsigned radix_size, unsigned radix_mask,
    const unsigned *global_offsets, unsigned block_count,
    [[seam::scratch(256)]] unsigned *local_offset,
    [[seam::scratch(256)]] unsigned *warp_counts,
    [[seam::lane]] unsigned thread_index, unsigned block_index,
    [[seam::width]] unsigned threads_per_group,
    unsigned groups);
