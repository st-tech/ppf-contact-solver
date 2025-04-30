#include "../common.hpp"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

template <typename Lambda> __global__ void launch_kernel(Lambda func, int n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        func(idx);
    }
}

namespace main_helper {
extern bool use_thrust;
}

#define DISPATCH_START(n)                                                      \
    {                                                                          \
        const unsigned n_threads(n);                                           \
        auto kernel =

#define DISPATCH_END                                                           \
    ;                                                                          \
    if (n_threads > 0) {                                                       \
        if (main_helper::use_thrust) {                                         \
            thrust::for_each(thrust::counting_iterator<unsigned>(0),           \
                             thrust::counting_iterator<unsigned>(n_threads),   \
                             kernel);                                          \
        } else {                                                               \
            unsigned block_size;                                               \
            unsigned grid_size;                                                \
            if (n_threads < BLOCK_SIZE) {                                      \
                block_size = n_threads;                                        \
                grid_size = 1;                                                 \
            } else {                                                           \
                block_size = BLOCK_SIZE;                                       \
                grid_size = (n_threads + block_size - 1) / block_size;         \
            }                                                                  \
            launch_kernel<<<grid_size, block_size>>>(kernel, n_threads);       \
        }                                                                      \
        cudaError_t error = cudaGetLastError();                                \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "    \
                      << __LINE__ << ": " << cudaGetErrorString(error)         \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    }                                                                          \
    }
