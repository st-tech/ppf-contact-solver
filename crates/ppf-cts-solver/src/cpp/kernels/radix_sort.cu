// File: radix_sort.cu
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "radix_sort.hpp"
#include "../main/cuda_utils.hpp"
#include "exclusive_scan.hpp"

namespace kernels {

constexpr unsigned RADIX_MASK = RADIX_SIZE - 1;

__global__ void radix_histogram_kernel(const unsigned *keys, unsigned n,
                                        unsigned shift, unsigned *block_histograms,
                                        unsigned num_blocks) {
    __shared__ unsigned local_hist[RADIX_SIZE];

    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;
    unsigned gid = bid * blockDim.x + tid;

    if (tid < RADIX_SIZE) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    if (gid < n) {
        unsigned digit = (keys[gid] >> shift) & RADIX_MASK;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    if (tid < RADIX_SIZE) {
        block_histograms[tid * num_blocks + bid] = local_hist[tid];
    }
}

__global__ void radix_scatter_kernel(const unsigned *keys_in,
                                      const unsigned *values_in,
                                      unsigned *keys_out, unsigned *values_out,
                                      unsigned n, unsigned shift,
                                      const unsigned *global_offsets,
                                      unsigned num_blocks) {
    __shared__ unsigned local_offset[RADIX_SIZE];
    __shared__ unsigned digit_counts[RADIX_SIZE];
    __shared__ unsigned thread_ranks[SORT_BLOCK_SIZE];

    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;
    unsigned gid = bid * blockDim.x + tid;

    if (tid < RADIX_SIZE) {
        local_offset[tid] = global_offsets[tid * num_blocks + bid];
        digit_counts[tid] = 0;
    }
    __syncthreads();

    unsigned my_key = 0, my_value = 0, my_digit = 0;
    bool valid = (gid < n);
    if (valid) {
        my_key = keys_in[gid];
        my_value = values_in[gid];
        my_digit = (my_key >> shift) & RADIX_MASK;
    }

    for (unsigned t = 0; t < blockDim.x; ++t) {
        if (t == tid && valid) {
            thread_ranks[tid] = digit_counts[my_digit];
            digit_counts[my_digit]++;
        }
        __syncthreads();
    }

    if (valid) {
        unsigned dest = local_offset[my_digit] + thread_ranks[tid];
        keys_out[dest] = my_key;
        values_out[dest] = my_value;
    }
}

void radix_sort_pairs(unsigned *keys, unsigned *values, unsigned n,
                      unsigned *temp_keys, unsigned *temp_values,
                      unsigned *histogram_buffer) {
    if (n <= 1) return;

    unsigned num_blocks = (n + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE;
    unsigned hist_size = RADIX_SIZE * num_blocks;

    unsigned *src_keys = keys;
    unsigned *src_values = values;
    unsigned *dst_keys = temp_keys;
    unsigned *dst_values = temp_values;

    for (unsigned shift = 0; shift < 32; shift += RADIX_BITS) {
        radix_histogram_kernel<<<num_blocks, SORT_BLOCK_SIZE>>>(
            src_keys, n, shift, histogram_buffer, num_blocks);
        CUDA_HANDLE_ERROR(cudaGetLastError());

        exclusive_scan(histogram_buffer, hist_size);

        radix_scatter_kernel<<<num_blocks, SORT_BLOCK_SIZE>>>(
            src_keys, src_values, dst_keys, dst_values, n, shift,
            histogram_buffer, num_blocks);
        CUDA_HANDLE_ERROR(cudaGetLastError());

        unsigned *tmp = src_keys;
        src_keys = dst_keys;
        dst_keys = tmp;

        tmp = src_values;
        src_values = dst_values;
        dst_values = tmp;
    }
}

void radix_sort_pairs(unsigned *keys, unsigned *values, unsigned n,
                      unsigned *temp_keys, unsigned *temp_values) {
    if (n <= 1) return;

    unsigned num_blocks = (n + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE;
    unsigned hist_size = RADIX_SIZE * num_blocks;

    unsigned *histogram_buffer;
    CUDA_HANDLE_ERROR(cudaMalloc(&histogram_buffer, hist_size * sizeof(unsigned)));

    radix_sort_pairs(keys, values, n, temp_keys, temp_values, histogram_buffer);

    CUDA_HANDLE_ERROR(cudaFree(histogram_buffer));
}

} // namespace kernels
