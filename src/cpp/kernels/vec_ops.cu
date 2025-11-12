// File: vec_ops.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../common.hpp"
#include "../data.hpp"
#include "../main/cuda_utils.hpp"
#include "reduce.hpp"
#include "vec_ops.hpp"

namespace kernels {

template <typename T>
__global__ void fill_kernel(T *array, unsigned n, T value) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = value;
    }
}

template <typename T>
__global__ void copy_kernel(const T *src, T *dst, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template <typename T>
__global__ void add_scaled_kernel(const T *src, T *dst, T scale, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += scale * src[idx];
    }
}

template <typename T>
__global__ void combine_kernel(const T *src_A, const T *src_B, T *dst, T a, T b,
                               unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a * src_A[idx] + b * src_B[idx];
    }
}

template <typename T> void set(T *array, unsigned n, T value) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        fill_kernel<<<blocks, block_size>>>(array, n, value);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

template <typename T> void copy(const T *src, T *dst, unsigned n) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        copy_kernel<<<blocks, block_size>>>(src, dst, n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

template <typename T>
void add_scaled(const T *src, T *dst, T scale, unsigned n) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        add_scaled_kernel<<<blocks, block_size>>>(src, dst, scale, n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

template <typename T>
void combine(const T *src_A, const T *src_B, T *dst, T a, T b, unsigned n) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        combine_kernel<<<blocks, block_size>>>(src_A, src_B, dst, a, b, n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

template void set(float *array, unsigned n, float value);
template void copy(const float *src, float *dst, unsigned n);
template void add_scaled(const float *src, float *dst, float scale, unsigned n);
template void combine(const float *src_A, const float *src_B, float *dst,
                      float a, float b, unsigned n);

template void set(char *array, unsigned n, char value);
template void copy(const char *src, char *dst, unsigned n);

template void set(unsigned *array, unsigned n, unsigned value);
template void copy(const unsigned *src, unsigned *dst, unsigned n);

template void set(Mat3x3f *array, unsigned n, Mat3x3f value);
template void copy(const Mat3x3f *src, Mat3x3f *dst, unsigned n);

template void set(Vec3f *array, unsigned n, Vec3f value);
template void copy(const Vec3f *src, Vec3f *dst, unsigned n);

} // namespace kernels
