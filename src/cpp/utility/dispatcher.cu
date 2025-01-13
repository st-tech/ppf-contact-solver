#include "../common.hpp"

template <typename Lambda> __global__ void launch_kernel(Lambda func, int n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        func(idx);
    }
}

#define DISPATCH_START(n)                                                      \
    {                                                                          \
        const unsigned n_threads(n);                                           \
        const unsigned block_size = BLOCK_SIZE;                                \
        const unsigned grid_size = (n_threads + block_size - 1) / block_size;  \
    launch_kernel<<<grid_size, block_size>>>(
#define DISPATCH_END , n_threads);                                             \
    }
