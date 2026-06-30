// File: radix_sort.cu
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "radix_sort.hpp"
#include "../buffer/buffer.hpp"
#include "../main/cuda_utils.hpp"
#include "exclusive_scan.hpp"

namespace kernels {

constexpr unsigned RADIX_MASK = RADIX_SIZE - 1;

// The histogram/scatter kernels init one shared bin per thread for tid <
// RADIX_SIZE, so a block must have at least RADIX_SIZE threads. This holds for
// RADIX_BITS=4 (16 bins) and the 8-bit case (256 bins == SORT_BLOCK_SIZE), but
// guards any future block-size reduction from silently leaving bins unset.
static_assert(SORT_BLOCK_SIZE >= RADIX_SIZE,
              "SORT_BLOCK_SIZE must cover every radix bin");

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
    // Intra-block stable rank via warp-ballot multisplit. This replaces a
    // thread-serial rank loop (exactly one active thread per __syncthreads,
    // blockDim.x barriers deep) with a per-warp ballot plus a small per-digit
    // prefix across warps. The global destination order is unchanged: it stays
    // a stable counting sort keyed by (digit, then ascending tid), which the
    // LBVH tree builder relies on for equal-Morton leaves.
    constexpr unsigned WARP = 32u;
    constexpr unsigned NUM_WARPS = SORT_BLOCK_SIZE / WARP;
    static_assert(SORT_BLOCK_SIZE % WARP == 0u,
                  "SORT_BLOCK_SIZE must be a whole number of warps");

    __shared__ unsigned local_offset[RADIX_SIZE];
    // wcount[w][d]: first the count of digit d in warp w, then (after the
    // prefix below) the exclusive prefix of those counts over ascending w.
    __shared__ unsigned wcount[NUM_WARPS][RADIX_SIZE];

    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;
    unsigned gid = bid * blockDim.x + tid;
    unsigned lane = tid & (WARP - 1u);
    unsigned warp_id = tid / WARP;

    if (tid < RADIX_SIZE) {
        local_offset[tid] = global_offsets[tid * num_blocks + bid];
    }

    unsigned my_key = 0, my_value = 0, my_digit = 0;
    bool valid = (gid < n);
    if (valid) {
        my_key = keys_in[gid];
        my_value = values_in[gid];
        my_digit = (my_key >> shift) & RADIX_MASK;
    }

    // Per-warp counting sort over the RADIX_SIZE digits. Every thread takes part
    // in every ballot (uniform control flow); invalid tail lanes (gid >= n) pass
    // the predicate as false and contribute nothing, so they cannot inflate a
    // count or steal a slot.
    unsigned lanemask_lt = (1u << lane) - 1u;
    unsigned my_rank_in_warp = 0;
    for (unsigned d = 0; d < RADIX_SIZE; ++d) {
        unsigned mask = __ballot_sync(0xffffffffu, valid && my_digit == d);
        if (valid && my_digit == d) {
            my_rank_in_warp = __popc(mask & lanemask_lt);
        }
        if (lane == 0) {
            wcount[warp_id][d] = __popc(mask);
        }
    }
    __syncthreads();

    // Exclusive prefix of each digit's counts across warps (ascending warp_id),
    // in place. RADIX_SIZE threads, one digit-column each: no barriers, no
    // shared-atomic contention.
    if (tid < RADIX_SIZE) {
        unsigned d = tid;
        unsigned acc = 0;
        for (unsigned w = 0; w < NUM_WARPS; ++w) {
            unsigned c = wcount[w][d];
            wcount[w][d] = acc;
            acc += c;
        }
    }
    __syncthreads();

    if (valid) {
        unsigned dest = local_offset[my_digit] + wcount[warp_id][my_digit] +
                        my_rank_in_warp;
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

        exclusive_scan_nocopy(histogram_buffer, hist_size);

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

    // Draw the histogram scratch from the pooled allocator (high-water reuse)
    // rather than a raw cudaMalloc/cudaFree per call, so repeated sorts (the
    // schwarz Galerkin path, whose `n` scales with contact count) perform no
    // dynamic GPU alloc/dealloc once the pool has warmed up. lbvh uses the
    // 6-arg overload with its own pre-allocated histogram and never reaches
    // here. The PooledVec auto-releases at function exit.
    auto histogram_buffer = buffer::get().get<unsigned>(hist_size);

    radix_sort_pairs(keys, values, n, temp_keys, temp_values,
                     histogram_buffer.data);
}

} // namespace kernels
