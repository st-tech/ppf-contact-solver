// File: exclusive_scan.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef EXCLUSIVE_SCAN_HPP
#define EXCLUSIVE_SCAN_HPP

#include "../main/cuda_utils.hpp"

namespace kernels {

__global__ void block_scan_kernel(unsigned *d_data, unsigned *d_block_sums,
                                  unsigned n);

__global__ void add_group_offsets_kernel(unsigned *data, unsigned n,
                                         const unsigned *parent,
                                         unsigned parent_n);

__global__ void add_block_base_offsets_kernel(unsigned *d_data,
                                              const unsigned *d_block_excl,
                                              unsigned n);

// Exclusive prefix sum over d_data (in place). Returns the grand total, which
// requires a blocking device-to-host copy of the final element.
unsigned exclusive_scan(unsigned *d_data, unsigned n);

// Same scan, but does not read the grand total back to the host (no sync). Use
// on paths that only need the in-place offsets, e.g. radix_sort_pairs.
void exclusive_scan_nocopy(unsigned *d_data, unsigned n);

} // namespace kernels

#endif // EXCLUSIVE_SCAN_HPP
