// File: dispatcher.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../common.hpp"
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#define DISPATCH_START(n)                                                      \
    {                                                                          \
        const unsigned n_threads(n);                                           \
        auto kernel =

#define DISPATCH_END                                                           \
    ;                                                                          \
    if (n_threads > 0) {                                                       \
        thrust::for_each(                                                      \
            thrust::device, thrust::counting_iterator<unsigned>(0),            \
            thrust::counting_iterator<unsigned>(n_threads), kernel);           \
        cudaError_t error = cudaGetLastError();                                \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "    \
                      << __LINE__ << ": " << cudaGetErrorString(error)         \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    }                                                                          \
    }
