// File: cuda_utils.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

// Fatal-exit reason code, read by the Rust host's atexit hook to label an
// exit(1) crash with a sub-kind instead of the coarse lock-based
// UnknownAbrupt. Values mirror ppf_cts_formats::status::error_code
// (2 = OOM, 3 = CUDA driver/runtime). Defined once in cpp/main/main.cu;
// the emulator never sets it (its CUDA_HANDLE_ERROR is a no-op).
extern "C" unsigned char g_ppf_fatal_code;

// Process-wide counters of device-memory alloc / free events (every
// Vec<T>::alloc/reserve and Vec<T>::free bumps these; see vec/vec.hpp).
// They exist so advance() can log the per-step delta and prove the solve
// loop reaches a steady state with ZERO dynamic GPU alloc/dealloc once the
// pre-allocated / high-water pools have warmed up. Plain host globals, so
// the emulator build links them too. Defined once in cpp/main/main.cu.
extern unsigned long long g_device_alloc_count;
extern unsigned long long g_device_free_count;

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
        // Label the imminent exit(1) so the host's atexit hook can write
        // Crashed{Oom} vs Crashed{CudaDriver} rather than UnknownAbrupt.
        g_ppf_fatal_code = (err == cudaErrorMemoryAllocation) ? 2 : 3;
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
// The emulator host-compile has no CUDA runtime, but it does parse the shared
// kernel-launcher declarations (kernels/vec_ops.hpp, kernels/reduce.hpp), which
// carry an optional stream handle. Provide a stand-in type so those headers
// parse; the emulator never issues stream work (the launchers are CUDA-only
// definitions it does not compile or link).
typedef void *cudaStream_t;
#endif

#endif
