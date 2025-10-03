// File: reduce.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef REDUCE_HPP
#define REDUCE_HPP

#include "../main/cuda_utils.hpp"

namespace kernels {

struct ReduceInfo {
    char *d_buffer = nullptr;
    size_t buffer_bytes = 0;

    void alloc(size_t required_bytes);
    ~ReduceInfo();

    template <typename T> T *get_block_sums(unsigned max_blocks) {
        return reinterpret_cast<T *>(d_buffer);
    }
    template <typename T> T *get_temp(unsigned max_blocks) {
        return reinterpret_cast<T *>(d_buffer + max_blocks * sizeof(T));
    }
    template <typename T> T *get_final_result(unsigned max_blocks) {
        return reinterpret_cast<T *>(d_buffer + 2 * max_blocks * sizeof(T));
    }
};

extern ReduceInfo reduce_info;

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
