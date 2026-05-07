// File: cuda_utils.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

// The emulator (libsimbackend_cpu) compiles this header tree with a
// plain C++ host compiler, no nvcc, no cuda_runtime.h. Detect that
// case and stub the macro: the cudaMalloc / cudaFree calls inside the
// macro argument are passed as preprocessor tokens and discarded
// without being parsed, so vec/vec.hpp's allocation methods never
// reference the CUDA runtime as long as they aren't instantiated by
// the emulator's code paths.
#ifdef __CUDACC__
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
#else
#define CUDA_HANDLE_ERROR(err) ((void)0)
// nvcc defines these as keywords; the host-only emulator compile gets
// them as no-op macros so vec/vec.hpp parses identically.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

#endif
