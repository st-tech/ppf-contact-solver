// File: reduce.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef REDUCE_HPP
#define REDUCE_HPP

#include "../main/cuda_utils.hpp"

namespace kernels {

template <typename T, typename Op> __device__ T warp_reduce(T val, Op func);

template <typename T, typename Op>
__device__ T block_reduce(T val, Op func, T init_val);

template <class Y, typename Op>
__global__ void reduce_op_kernel_1(const Y *input, Y *output,
                                   Op func, Y init_val, unsigned n);

template <class Y, typename Op>
__global__ void reduce_op_kernel_2(const Y *input1, const Y *input2, Y *output,
                                   Op func, Y init_val, unsigned n);

template <typename Y, typename Op>
__global__ void final_reduce_kernel(Y *data, Y *output, Op func, Y init_val, unsigned n);

template <class Y, typename Op1, typename Op2>
Y reduce(const Y *d_input1, const Y *d_input2, Op1 func1, Op2 func2, Y init_val, unsigned n);

template <class Y, typename Op>
Y reduce_1(const Y *d_input, Op func, Y init_val, unsigned n);

template <class Y, typename Op>
Y reduce_2(const Y *d_input1, const Y *d_input2, Op func, Y init_val, unsigned n);

template <class T> T sum_array(const T *array, unsigned size);

template <class T> T min_array(const T *array, unsigned size, T init_val);

template <class T> T max_array(const T *array, unsigned size, T init_val);

template <class T> T inner_product(const T *array1, const T *array2, unsigned size);

// Device-resident reduction outputs for the sync-free PCG inner loop. Same
// float32 arithmetic as inner_product()/sum_array(), but the scalar is left at
// the caller-supplied device address `out` instead of copied to the host, and
// the multi-pass tail ping-pongs scratch buffers rather than issuing a
// host-synchronizing device-to-device copy. No device-to-host copy occurs, so
// the whole reduction queues on `queue` with no host round-trip.
void inner_product_into(const float *a, const float *b, float *out, unsigned n,
                        cudaStream_t queue = 0);
void sum_into(const float *in, float *out, unsigned n, cudaStream_t queue = 0);

} // namespace kernels

#endif // REDUCE_HPP
