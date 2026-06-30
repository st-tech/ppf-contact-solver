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

// --- Indirect-coefficient kernels (coefficient read from a device pointer) ---

__global__ void add_scaled_indirect_kernel(const float *src, float *dst,
                                           const float *coeff, float sign,
                                           unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = sign * (*coeff);
        dst[idx] = fmaf(s, src[idx], dst[idx]);
    }
}

__global__ void combine_indirect_kernel(const float *src_A, const float *src_B,
                                        float *dst, float a,
                                        const float *coeff_B, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float b = *coeff_B;
        dst[idx] = a * src_A[idx] + b * src_B[idx];
    }
}

__global__ void scalar_div_kernel(float *out, const float *num,
                                  const float *den, int *breakdown) {
    float d = *den;
    if (d > 0.0f) {
        *out = (*num) / d;
    } else {
        *out = 0.0f;
        if (breakdown) {
            *breakdown = 1;
        }
    }
}

__global__ void scalar_assign_kernel(float *dst, const float *src) {
    *dst = *src;
}

__global__ void flag_if_nonpositive_kernel(const float *val, int *flag) {
    float v = *val;
    if (!(v > 0.0f) || !isfinite(v)) { // catches v <= 0, NaN, +/-Inf
        *flag = 1;
    }
}

// --- Launchers ---

template <typename T>
void set(T *array, unsigned n, T value, cudaStream_t queue) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        fill_kernel<<<blocks, block_size, 0, queue>>>(array, n, value);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

template <typename T>
void copy(const T *src, T *dst, unsigned n, cudaStream_t queue) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        copy_kernel<<<blocks, block_size, 0, queue>>>(src, dst, n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

template <typename T>
void add_scaled(const T *src, T *dst, T scale, unsigned n, cudaStream_t queue) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        add_scaled_kernel<<<blocks, block_size, 0, queue>>>(src, dst, scale, n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

template <typename T>
void combine(const T *src_A, const T *src_B, T *dst, T a, T b, unsigned n,
             cudaStream_t queue) {
    if (n == 0) {
        return;
    } else {
        unsigned block_size = choose_block_size(n);
        unsigned blocks = (n + block_size - 1) / block_size;
        combine_kernel<<<blocks, block_size, 0, queue>>>(src_A, src_B, dst, a, b,
                                                         n);
        CUDA_HANDLE_ERROR(cudaGetLastError());
    }
}

void add_scaled_indirect(const float *src, float *dst, const float *coeff,
                         float sign, unsigned n, cudaStream_t queue) {
    if (n == 0) {
        return;
    }
    unsigned block_size = choose_block_size(n);
    unsigned blocks = (n + block_size - 1) / block_size;
    add_scaled_indirect_kernel<<<blocks, block_size, 0, queue>>>(
        src, dst, coeff, sign, n);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

void combine_indirect(const float *src_A, const float *src_B, float *dst,
                      float a, const float *coeff_B, unsigned n,
                      cudaStream_t queue) {
    if (n == 0) {
        return;
    }
    unsigned block_size = choose_block_size(n);
    unsigned blocks = (n + block_size - 1) / block_size;
    combine_indirect_kernel<<<blocks, block_size, 0, queue>>>(
        src_A, src_B, dst, a, coeff_B, n);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

void scalar_div(float *out, const float *num, const float *den, int *breakdown,
                cudaStream_t queue) {
    scalar_div_kernel<<<1, 1, 0, queue>>>(out, num, den, breakdown);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

void scalar_assign(float *dst, const float *src, cudaStream_t queue) {
    scalar_assign_kernel<<<1, 1, 0, queue>>>(dst, src);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

void flag_if_nonpositive(const float *val, int *flag, cudaStream_t queue) {
    flag_if_nonpositive_kernel<<<1, 1, 0, queue>>>(val, flag);
    CUDA_HANDLE_ERROR(cudaGetLastError());
}

template void set(float *array, unsigned n, float value, cudaStream_t queue);
template void copy(const float *src, float *dst, unsigned n,
                   cudaStream_t queue);
template void add_scaled(const float *src, float *dst, float scale, unsigned n,
                         cudaStream_t queue);
template void combine(const float *src_A, const float *src_B, float *dst,
                      float a, float b, unsigned n, cudaStream_t queue);

template void set(char *array, unsigned n, char value, cudaStream_t queue);
template void copy(const char *src, char *dst, unsigned n, cudaStream_t queue);

template void set(unsigned *array, unsigned n, unsigned value,
                  cudaStream_t queue);
template void copy(const unsigned *src, unsigned *dst, unsigned n,
                   cudaStream_t queue);

template void set(Mat3x3f *array, unsigned n, Mat3x3f value,
                  cudaStream_t queue);
template void copy(const Mat3x3f *src, Mat3x3f *dst, unsigned n,
                   cudaStream_t queue);

template void set(Vec3f *array, unsigned n, Vec3f value, cudaStream_t queue);
template void copy(const Vec3f *src, Vec3f *dst, unsigned n, cudaStream_t queue);

} // namespace kernels
