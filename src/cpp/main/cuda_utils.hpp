// File: cuda_utils.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}

#define CUDA_HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#endif
