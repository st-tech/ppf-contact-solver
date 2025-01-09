#include "../data.hpp"

#define BLOCK_SIZE 256

namespace VecDev {

template <class T> __global__ void set_kernel(T *array, unsigned n, T val) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = val;
    }
}

template <class T>
__global__ void copy_kernel(const T *src, T *dst, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template <class T>
__global__ void add_scaled_kernel(const T *src, T *dst, T scale, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += scale * src[idx];
    }
}

template <class T>
__global__ void combine_kernel(const T *src_A, const T *src_B, T *dst, T a, T b,
                               unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a * src_A[idx] + b * src_B[idx];
    }
}

template <class T>
__global__ void inner_product_kernel(const T *a, const T *b, T *c, unsigned n) {
    __shared__ unsigned char _shared_data[BLOCK_SIZE];
    T *shared_data = reinterpret_cast<T *>(_shared_data);
    unsigned tid = threadIdx.x;
    unsigned global_idx = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = (global_idx < n) ? a[global_idx] * b[global_idx] : T();
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        c[blockIdx.x] = shared_data[0];
    }
}

template <class T> T inner_product(const T *a, const T *b, unsigned n) {
    unsigned grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static unsigned max_grid_size = 0;
    static T *d_c = nullptr;
    static T *h_c = nullptr;
    if (d_c == nullptr) {
        cudaMalloc(&d_c, grid_size * sizeof(T));
        h_c = new T[grid_size];
        max_grid_size = grid_size;
    } else if (max_grid_size < grid_size) {
        cudaFree(d_c);
        delete[] h_c;
        cudaMalloc(&d_c, grid_size * sizeof(T));
        h_c = new T[grid_size];
        max_grid_size = grid_size;
    }
    inner_product_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(T)>>>(a, b, d_c, n);
    cudaMemcpy(h_c, d_c, grid_size * sizeof(T), cudaMemcpyDeviceToHost);
    T result = T();
    for (unsigned i = 0; i < grid_size; ++i) {
        result += h_c[i];
    }
    return result;
}

template <class T> void set(T *array, unsigned n, T val) {
    unsigned grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    set_kernel<<<grid_size, BLOCK_SIZE>>>(array, n, val);
}

template <class T> void copy(const T *src, T *dst, unsigned n) {
    unsigned grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copy_kernel<<<grid_size, BLOCK_SIZE>>>(src, dst, n);
}

template <class T> void add_scaled(const T *src, T *dst, T scale, unsigned n) {
    unsigned grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_scaled_kernel<<<grid_size, BLOCK_SIZE>>>(src, dst, scale, n);
}

template <class T>
void combine(const T *src_A, const T *src_B, T *dst, T a, T b, unsigned n) {
    unsigned grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    combine_kernel<<<grid_size, BLOCK_SIZE>>>(src_A, src_B, dst, a, b, n);
}

} // namespace VecDev

template void VecDev::set(float *array, unsigned size, float init_val);
template void VecDev::copy(const float *src, float *dst, unsigned size);
template void VecDev::add_scaled(const float *src, float *dst, float scale,
                                 unsigned size);
template void VecDev::combine(const float *src_A, const float *src_B,
                              float *dst, float a, float b, unsigned size);
template float VecDev::inner_product(const float *a, const float *b,
                                     unsigned size);

template void VecDev::set(char *array, unsigned size, char init_val);
template void VecDev::copy(const char *src, char *dst, unsigned size);

template void VecDev::set(unsigned *array, unsigned size, unsigned init_val);
template void VecDev::copy(const unsigned *src, unsigned *dst, unsigned size);

template void VecDev::set(Mat3x3f *array, unsigned size, Mat3x3f init_val);
template void VecDev::copy(const Mat3x3f *src, Mat3x3f *dst, unsigned size);

template void VecDev::set(Vec3f *array, unsigned size, Vec3f init_val);
template void VecDev::copy(const Vec3f *src, Vec3f *dst, unsigned size);
