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

} // namespace kernels

#endif // REDUCE_HPP
