// File: dispatcher.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../common.hpp"
#include "../main/cuda_utils.hpp"

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

// Synchronizing dispatch on the legacy stream. The launch is asynchronous, so
// the trailing sync preserves the historical (thrust::device) behavior: a
// caller may read the result on the host immediately after DISPATCH_END.
#define DISPATCH_START(n)                                                      \
    {                                                                          \
        const unsigned n_threads(n);                                           \
        auto kernel =

#define DISPATCH_END                                                           \
    ;                                                                          \
    if (n_threads > 0) {                                                       \
        const unsigned dispatch_block = 256;                                   \
        const unsigned dispatch_grid =                                         \
            (n_threads + dispatch_block - 1) / dispatch_block;                 \
        indexed_apply<<<dispatch_grid, dispatch_block>>>(n_threads, kernel);   \
        CUDA_HANDLE_ERROR(cudaGetLastError());                                 \
        CUDA_HANDLE_ERROR(cudaStreamSynchronize(0));                           \
    }                                                                          \
    }

// Non-synchronizing dispatch on a caller-owned stream. Work queues on `q` and
// the caller owns ordering/synchronization (used by the PCG fast loop so a whole
// CG iteration chains with no host round-trip).
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
