// File: radix_sort.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef RADIX_SORT_HPP
#define RADIX_SORT_HPP

namespace kernels {

// Number of bits per radix sort pass
constexpr unsigned RADIX_BITS = 4;
constexpr unsigned RADIX_SIZE = 1 << RADIX_BITS; // 16 buckets

// Block size for sorting kernels
constexpr unsigned SORT_BLOCK_SIZE = 256;

// Sort key-value pairs by key using radix sort
void radix_sort_pairs(unsigned *keys, unsigned *values, unsigned n,
                      unsigned *temp_keys, unsigned *temp_values,
                      unsigned *histogram_buffer);

// Convenience function that allocates histogram buffer internally
void radix_sort_pairs(unsigned *keys, unsigned *values, unsigned n,
                      unsigned *temp_keys, unsigned *temp_values);

} // namespace kernels

#endif // RADIX_SORT_HPP
