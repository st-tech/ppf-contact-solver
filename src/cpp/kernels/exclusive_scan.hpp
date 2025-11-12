// File: exclusive_scan.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef EXCLUSIVE_SCAN_HPP
#define EXCLUSIVE_SCAN_HPP

#include "../main/cuda_utils.hpp"

namespace kernels {

__device__ __host__ unsigned nextPowerOf2(unsigned n);

__global__ void block_scan_kernel(unsigned *d_data, unsigned *d_block_sums,
                                  unsigned n);

__global__ void add_block_offsets_kernel(unsigned *d_data,
                                         unsigned *d_block_sums, unsigned n);

__global__ void blelloch_single_block_kernel(unsigned *data, unsigned n,
                                             unsigned *total_sum);

__global__ void add_group_offsets_kernel(unsigned *data, unsigned n,
                                         const unsigned *parent,
                                         unsigned parent_n);

__global__ void add_block_base_offsets_kernel(unsigned *d_data,
                                              const unsigned *d_block_excl,
                                              unsigned n);

unsigned exclusive_scan(unsigned *d_data, unsigned n);

} // namespace kernels

#endif // EXCLUSIVE_SCAN_HPP
