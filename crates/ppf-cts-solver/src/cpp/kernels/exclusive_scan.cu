// File: exclusive_scan.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../utility/dispatcher.hpp"
#include "../vec/vec.hpp"
#include "../buffer/buffer.hpp"
#include "exclusive_scan.hpp"
#include <cassert>

namespace kernels {

// Block size for exclusive scan operations
static const unsigned SCAN_BLOCK_SIZE = 256;

__device__ __host__ inline unsigned umin(unsigned a, unsigned b) {
    return (a < b) ? a : b;
}

__device__ __host__ unsigned nextPowerOf2(unsigned n) {
    unsigned power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

__global__ void block_scan_kernel(unsigned *d_data, unsigned *d_block_sums,
                                  unsigned n) {
    extern __shared__ unsigned warp_excl[];

    const unsigned G = blockDim.x;
    const unsigned num_warps = G / WARP_SIZE;
    const unsigned bid = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (WARP_SIZE - 1);
    const unsigned warp_id = tid / WARP_SIZE;
    const unsigned block_offset = bid * G;
    const unsigned idx = block_offset + tid;

    unsigned x = (idx < n) ? d_data[idx] : 0u;
    unsigned v = x;
    for (unsigned ofs = 1; ofs < WARP_SIZE; ofs <<= 1) {
        unsigned y = __shfl_up_sync(0xffffffffu, v, ofs);
        if (lane >= ofs) {
            v += y;
        }
    }
    unsigned excl = v - x;
    if (lane == WARP_SIZE - 1) {
        warp_excl[warp_id] = v;
    }
    __syncthreads();

    if (warp_id == 0) {
        unsigned ws = (lane < num_warps) ? warp_excl[lane] : 0u;
        for (unsigned ofs = 1; ofs < WARP_SIZE; ofs <<= 1) {
            unsigned y = __shfl_up_sync(0xffffffffu, ws, ofs);
            if (lane >= ofs) {
                ws += y;
            }
        }
        if (lane < num_warps) {
            warp_excl[lane] = ws - warp_excl[lane];
        }
    }
    __syncthreads();

    unsigned warp_offset = warp_excl[warp_id];
    unsigned scanned_excl = excl + warp_offset;

    if (idx < n) {
        d_data[idx] = scanned_excl;
    }

    const unsigned block_n =
        (n > block_offset) ? ((G < (n - block_offset)) ? G : (n - block_offset))
                           : 0u;
    const unsigned last = block_n ? (block_n - 1) : 0u;
    if (d_block_sums && tid == last) {
        d_block_sums[bid] = scanned_excl + x;
    }
}

__global__ void add_block_offsets_kernel(unsigned *d_data,
                                         unsigned *d_block_sums, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < n) {
        d_data[idx] += d_block_sums[blockIdx.x - 1];
    }
}

__global__ void blelloch_single_block_kernel(unsigned *data, unsigned n,
                                             unsigned *total_sum) {
    extern __shared__ unsigned temp[];

    unsigned tid = threadIdx.x;
    unsigned offset = 1;

    temp[2 * tid] = (2 * tid < n) ? data[2 * tid] : 0;
    temp[2 * tid + 1] = (2 * tid + 1 < n) ? data[2 * tid + 1] : 0;

    unsigned d = nextPowerOf2(n);

    for (unsigned stride = d >> 1; stride > 0; stride >>= 1) {
        __syncthreads();

        if (tid < stride) {
            unsigned ai = offset * (2 * tid + 1) - 1;
            unsigned bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (tid == 0 && total_sum != nullptr) {
        *total_sum = temp[d - 1];
        temp[d - 1] = 0;
    }
    for (unsigned stride = 1; stride < d; stride *= 2) {
        offset >>= 1;
        __syncthreads();

        if (tid < stride) {
            unsigned ai = offset * (2 * tid + 1) - 1;
            unsigned bi = offset * (2 * tid + 2) - 1;

            unsigned t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    if (2 * tid < n) {
        data[2 * tid] = temp[2 * tid];
    }
    if (2 * tid + 1 < n) {
        data[2 * tid + 1] = temp[2 * tid + 1];
    }
}

__global__ void add_group_offsets_kernel(unsigned *data, unsigned n,
                                         const unsigned *parent,
                                         unsigned parent_n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    unsigned g = idx / blockDim.x;
    unsigned off = (g < parent_n) ? parent[g] : 0u;
    data[idx] += off;
}

__global__ void add_block_base_offsets_kernel(unsigned *d_data,
                                              const unsigned *d_block_excl,
                                              unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    d_data[idx] += d_block_excl[blockIdx.x];
}

unsigned exclusive_scan(unsigned *d_data, unsigned n) {
    if (n == 0) {
        return 0;
    }

    const unsigned G = SCAN_BLOCK_SIZE;
    const size_t shmem = G * sizeof(unsigned);
    const unsigned MAX_LEVELS = 32;

    // Calculate levels and sizes
    unsigned levels = 0;
    unsigned level_n[MAX_LEVELS] = {};
    unsigned curr_n = n;
    while (true) {
        unsigned m = (curr_n + G - 1) / G;
        level_n[levels] = m;
        ++levels;
        if (m == 1) {
            break;
        }
        curr_n = m;
        if (levels >= MAX_LEVELS) {
            assert(false && "MAX_LEVELS too small for this n/SCAN_BLOCK_SIZE");
        }
    }

    // Allocate buffers from pool
    buffer::MemoryPool &pool = buffer::get();

    // Calculate total buffer size needed
    size_t total_size = 0;
    for (unsigned i = 0; i < levels; ++i) {
        total_size += level_n[i];
    }
    total_size += 1; // for d_total

    // Allocate one large buffer and partition it
    auto combined_buffer = pool.get<unsigned>(total_size);

    // Partition the buffer into level pointers
    unsigned *level_ptr[MAX_LEVELS];
    size_t offset = 0;
    for (unsigned i = 0; i < levels; ++i) {
        level_ptr[i] = combined_buffer.data + offset;
        offset += level_n[i];
    }
    unsigned *d_total = combined_buffer.data + offset;

    unsigned num_blocks_lvl0 = (n + G - 1) / G;

    block_scan_kernel<<<level_n[0], G, shmem>>>(d_data, level_ptr[0], n);
    CUDA_HANDLE_ERROR(cudaGetLastError());

    for (unsigned li = 1; li < levels; ++li) {
        block_scan_kernel<<<level_n[li], G, shmem>>>(
            level_ptr[li - 1], level_ptr[li], level_n[li - 1]);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }

    block_scan_kernel<<<1, G, shmem>>>(level_ptr[levels - 1], d_total,
                                       level_n[levels - 1]);
    CUDA_HANDLE_ERROR(cudaGetLastError());

    for (unsigned li = levels - 1; li > 0; --li) {
        unsigned child_n = level_n[li - 1];
        unsigned parent_n = level_n[li];
        unsigned grid = (child_n + G - 1) / G;
        add_group_offsets_kernel<<<grid, G>>>(level_ptr[li - 1], child_n,
                                              level_ptr[li], parent_n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }

    add_block_base_offsets_kernel<<<num_blocks_lvl0, G>>>(d_data, level_ptr[0],
                                                          n);
    CUDA_HANDLE_ERROR(cudaGetLastError());

    unsigned total_sum = 0;
    CUDA_HANDLE_ERROR(cudaMemcpy(&total_sum, d_total, sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    return total_sum;
}

} // namespace kernels
