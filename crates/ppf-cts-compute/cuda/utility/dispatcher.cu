// File: dispatcher.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "common.hpp"
#include "diagnostics/diagnostics.hpp"
#include "../cuda_utils.hpp"

// Elementwise dispatch over [0, n): the body `fn(i)` is a device functor (an
// extended __device__ lambda at the call site). Implemented as a plain kernel
// launch rather than thrust::for_each on purpose: thrust/CUB have shown
// driver-version-dependent miscompiles, so the backend depends on neither. The
// grid-flat index guard matches the raw kernels in kernels/vec_ops.cu and
// kernels/reduce.cu.
template <typename F> __global__ void indexed_apply(unsigned n, F fn) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fn(idx);
    }
}

template <typename F>
__global__ void indexed_apply_diag(unsigned n,
                                   diagnostics::Channel channel, F fn) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fn(idx, diagnostics::bind(channel, idx));
    }
}

// Synchronizing dispatch on the legacy stream. The launch is asynchronous, so
// the trailing sync preserves the historical (thrust::device) behavior: a
// caller may read the result on the host immediately after DISPATCH_END.
#define DISPATCH_START(n)                                                      \
    {                                                                          \
        const unsigned n_threads(n);                                           \
        constexpr bool dispatch_diagnostics = false;                       \
        auto kernel =

#define DISPATCH_DIAG_START(n)                                                 \
    {                                                                          \
        const unsigned n_threads(n);                                           \
        constexpr bool dispatch_diagnostics = true;                        \
        auto kernel =

#define DISPATCH_END                                                           \
    ;                                                                          \
    if (n_threads > 0) {                                                       \
        const unsigned dispatch_block = 256;                                   \
        const unsigned dispatch_grid =                                         \
            (n_threads + dispatch_block - 1) / dispatch_block;                 \
        if constexpr (dispatch_diagnostics) {                              \
            diagnostics::reset_assert_global();                          \
            indexed_apply_diag<<<dispatch_grid, dispatch_block>>>(             \
                n_threads, diagnostics::global(), kernel);                \
        } else {                                                               \
            indexed_apply<<<dispatch_grid, dispatch_block>>>(n_threads,        \
                                                              kernel);         \
        }                                                                      \
        CUDA_HANDLE_ERROR(cudaGetLastError());                                 \
        CUDA_HANDLE_ERROR(cudaStreamSynchronize(0));                           \
        if constexpr (dispatch_diagnostics) {                              \
            diagnostics::check_assert_global();                          \
        }                                                                      \
    }                                                                          \
    }

// Non-synchronizing dispatch on a caller-owned stream. Work queues on `q` and
// the caller owns ordering and synchronization, which is what lets a caller
// chain a whole sequence of launches with no host round-trip.
#define DISPATCH_QUEUE_START(n, q)                                             \
    {                                                                          \
        const unsigned n_threads(n);                                           \
        const cudaStream_t queue_handle(q);                                    \
        auto kernel =

#define DISPATCH_QUEUE_END                                                      \
    ;                                                                          \
    if (n_threads > 0) {                                                       \
        const unsigned dispatch_block = 256;                                   \
        const unsigned dispatch_grid =                                         \
            (n_threads + dispatch_block - 1) / dispatch_block;                 \
        indexed_apply<<<dispatch_grid, dispatch_block, 0, queue_handle>>>(     \
            n_threads, kernel);                                                \
        CUDA_HANDLE_ERROR(cudaGetLastError());                                 \
    }                                                                          \
    }
