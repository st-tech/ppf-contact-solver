// File: reduce.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// THIS BODY IS DEVICE-ONLY, and the seam says so by omission. It names the
// SIMD geometry a reduction is written against: the lane count
// (compute::simd_width), a lane shuffle (compute::shuffle_down) and a
// threadgroup barrier (compute::threadgroup_barrier), on top of the ordinary
// [[seam::device_fn]] and [[seam::threadgroup]]. The nvcc prologue
// (ppf-cts-compute/cuda/seam_cuda.cuh) and the Metal prologue (kMslMacroSeam
// in ppf-cts-compute/metal/shader_compiler.mm) define all of them; the host
// prologue (kernels/seam/seam_host.h) deliberately defines none, and states
// why. So the host rendering of this file compiles nowhere, and that is the
// loud answer: a host translation unit reaching it stops on the undefined name
// rather than computing something else.
//
// compute::shuffle_down is GENERIC OVER THE SHUFFLED TYPE, and both device
// prologues say so at the definition. It has to be: warp_reduce is a template
// over the reduced type `T` and shuffles `T` itself, and block_reduce below
// calls it on its own `T`, so the shuffle has to accept whatever a caller
// reduces. Narrowing it to one type would CONVERT those values rather than
// refuse them, truncating a float on the way in and back.

template <class T, class Op>
[[seam::device_fn]] inline T warp_reduce(T value, Op op) {
    for (unsigned offset = compute::simd_width / 2; offset > 0; offset >>= 1) {
        value = op(value, compute::shuffle_down(value, offset));
    }
    return value;
}

template <class T, class Op>
[[seam::device_fn]] inline T block_reduce(T value, Op op, T identity,
                                             T *warp_results,
                                             unsigned thread_index,
                                             unsigned threads_per_group) {
    const unsigned warps =
        (threads_per_group + compute::simd_width - 1) / compute::simd_width;
    const unsigned lane = thread_index & (compute::simd_width - 1);
    const unsigned warp = thread_index / compute::simd_width;
    value = warp_reduce(value, op);
    if (lane == 0) {
        warp_results[warp] = value;
    }
    compute::threadgroup_barrier();
    value = thread_index < warps ? warp_results[thread_index] : identity;
    if (warp == 0) {
        value = warp_reduce(value, op);
    }
    return value;
}
