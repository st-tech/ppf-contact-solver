// File: scan.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The two facts a backend cannot infer are written as
// C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::device]]` and `[[seam::threadgroup]]` are the address spaces of the
// pointer parameters. MSL requires the second on every pointer and reference
// type; CUDA and the host have one address space and are handed the same
// declarations with it removed.
//
// No include of its own: every name below is either declared in this file or
// arrives from the backend prologue, which is `seam_cuda.cuh` under nvcc,
// `seam/seam_host.h` on the host, and the prologue in
// `metal/shader_compiler.mm` under MSL.
//
// THIS BODY IS DEVICE-ONLY, and it says so twice over. Every function carries
// `[[seam::device_fn]]` rather than `[[seam::host_device_fn]]`, and the SIMD
// geometry a scan is written against reaches it through three names the host
// prologue deliberately does not define: `compute::simd_width` is the lane
// count, `compute::shuffle_up` the lane shuffle, and
// `compute::threadgroup_barrier` the threadgroup barrier. The nvcc prologue and
// the Metal prologue define all three; seam_host.h defines none of them and
// states why. A host translation unit that reached this body would stop on the
// undefined name rather than compute something else.

[[seam::device_fn]] inline void block_exclusive_scan(
    unsigned *data, unsigned *block_sums,
    unsigned count, unsigned block_index, unsigned thread_index,
    unsigned threads_per_group,
    unsigned *warp_exclusive) {
    const unsigned num_warps = threads_per_group / compute::simd_width;
    const unsigned lane = thread_index & (compute::simd_width - 1);
    const unsigned warp = thread_index / compute::simd_width;
    const unsigned block_offset = block_index * threads_per_group;
    const unsigned index = block_offset + thread_index;
    const unsigned input = index < count ? data[index] : 0u;
    unsigned inclusive = input;
    for (unsigned offset = 1; offset < compute::simd_width; offset <<= 1) {
        unsigned other = compute::shuffle_up(inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    unsigned exclusive = inclusive - input;
    if (lane == compute::simd_width - 1) {
        warp_exclusive[warp] = inclusive;
    }
    compute::threadgroup_barrier();
    if (warp == 0) {
        unsigned warp_sum =
            lane < num_warps ? warp_exclusive[lane] : 0u;
        for (unsigned offset = 1; offset < compute::simd_width; offset <<= 1) {
            unsigned other = compute::shuffle_up(warp_sum, offset);
            if (lane >= offset) {
                warp_sum += other;
            }
        }
        if (lane < num_warps) {
            warp_exclusive[lane] = warp_sum - warp_exclusive[lane];
        }
    }
    compute::threadgroup_barrier();
    const unsigned scanned = exclusive + warp_exclusive[warp];
    if (index < count) {
        data[index] = scanned;
    }
    const unsigned block_count =
        count > block_offset
            ? (threads_per_group < count - block_offset
                   ? threads_per_group
                   : count - block_offset)
            : 0u;
    const unsigned last = block_count ? block_count - 1 : 0u;
    if (block_sums && thread_index == last) {
        block_sums[block_index] = scanned + input;
    }
}

[[seam::device_fn]] inline void add_group_offset(
    unsigned *data, unsigned count,
    const unsigned *parent, unsigned parent_count,
    unsigned index, unsigned threads_per_group) {
    if (index < count) {
        unsigned group = index / threads_per_group;
        data[index] += group < parent_count ? parent[group] : 0u;
    }
}

[[seam::device_fn]] inline void add_block_offset(
    unsigned *data,
    const unsigned *block_exclusive, unsigned count,
    unsigned index, unsigned block_index) {
    if (index < count) {
        data[index] += block_exclusive[block_index];
    }
}
