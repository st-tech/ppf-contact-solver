// File: cuda_utils.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

// Brings in `g_ppf_fatal_code` / `g_ppf_fatal_detail` (the reason and message
// the Rust host's atexit hook reads to label an exit(1) crash) plus
// `ppf_fatal`. Included outside the __CUDACC__ guard below because the
// emulator compiles this header too and terminates on the same invariants.
#include "fatal.hpp"

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

// State of the operating system's kernel-execution watchdog on the device
// this process selected, as read from `cudaDeviceProp::kernelExecTimeoutEnabled`
// by the preflight in `initialize()`. Negative until that runs. A crash report
// naming a launch timeout is only readable next to this value, and the flag is
// not recoverable after the fact, so it is carried in memory rather than left
// to a log line. Defined once in cpp/main/main.cu.
extern int g_ppf_kernel_timeout_enabled;

// Which crash sub-kind a `cudaError_t` reports as. Three errors carry a cause
// the user can act on differently from "the CUDA runtime failed":
// an allocation failure means the scene does not fit, cudaErrorAssert means a
// device-side assert trapped (a violated solver invariant rather than a driver
// or hardware problem), and cudaErrorLaunchTimeout means the OS reset the
// device out from under a kernel, which is a property of the machine (a GPU
// that also drives a display) and is fixed by running on a GPU with no display
// attached. Everything else, from an unsupported architecture to a corrupted
// context, is reported as CudaDriver and distinguished by the detail, which
// carries the runtime's own name and message for the error.
static unsigned char fatal_code_for_cuda(cudaError_t err) {
    if (err == cudaErrorMemoryAllocation) {
        return PPF_FATAL_OOM;
    }
    if (err == cudaErrorAssert) {
        return PPF_FATAL_DEVICE_ASSERT;
    }
    if (err == cudaErrorLaunchTimeout) {
        return PPF_FATAL_WATCHDOG_TIMEOUT;
    }
    return PPF_FATAL_CUDA_DRIVER;
}

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        // Label the imminent exit(1) so the host's atexit hook can name the
        // cause instead of writing UnknownAbrupt.
        const unsigned char code = fatal_code_for_cuda(err);
        char detail[sizeof(g_ppf_fatal_detail)];
        int used = snprintf(detail, sizeof(detail),
                            "%s (cuda error %d %s) in %s at line %d",
                            cudaGetErrorString(err), (int)err,
                            cudaGetErrorName(err), file, line);
        if (code == PPF_FATAL_OOM && used > 0 &&
            (size_t)used < sizeof(detail)) {
            // Valid only on the allocation branch: a failed allocation
            // leaves the context usable, whereas a sticky error would make
            // this query fail in turn and report nonsense.
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                snprintf(detail + used, sizeof(detail) - (size_t)used,
                         "; %zu MB free of %zu MB on the device",
                         free_bytes >> 20, total_bytes >> 20);
            }
        }
        if (code == PPF_FATAL_WATCHDOG_TIMEOUT && used > 0 &&
            (size_t)used < sizeof(detail)) {
            // Report the flag as it was actually read, so a timeout on a
            // device with no watchdog armed says so rather than asserting a
            // cause the preflight contradicts.
            const char *state = g_ppf_kernel_timeout_enabled > 0
                                    ? "armed"
                                    : (g_ppf_kernel_timeout_enabled == 0
                                           ? "not armed"
                                           : "not known");
            snprintf(detail + used, sizeof(detail) - (size_t)used,
                     "; the operating system's kernel-execution watchdog is "
                     "%s on this device",
                     state);
        }
        ppf_fatal_set_detail(detail);
        g_ppf_fatal_code = code;
        exit(1);
    }
}

#define CUDA_HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// Persistent page-locked (pinned) host scratch for tiny device-to-host
// readbacks (reduction results, residual probes, counters). A D2H copy into
// PAGEABLE host memory degrades cudaMemcpy/cudaMemcpyAsync to a blocking
// staged transfer costing ~100 us of host time per call regardless of size;
// with a pinned destination the same copy is a direct DMA (~5 us API).
// Measured on the trapped bench: 144k residual-probe reads alone accounted
// for ~14.9 s of host API time in a ~35 s run. Single host solver thread; the
// buffer grows monotonically and is intentionally never freed (process-lived,
// a few hundred bytes). Callers must consume the staged value before the next
// pinned_scratch-based readback on the same thread.
inline void *pinned_scratch(size_t bytes) {
    static void *ptr = nullptr;
    static size_t cap = 0;
    if (cap < bytes) {
        if (ptr) {
            CUDA_HANDLE_ERROR(cudaFreeHost(ptr));
        }
        size_t want = bytes < 256 ? 256 : bytes;
        CUDA_HANDLE_ERROR(cudaMallocHost(&ptr, want));
        cap = want;
    }
    return ptr;
}

// Read one POD value from device memory through the pinned scratch (the fast
// path for the ubiquitous single-scalar reduction readback).
template <class Y> inline Y pinned_read(const Y *dev) {
    Y *stage = static_cast<Y *>(pinned_scratch(sizeof(Y)));
    CUDA_HANDLE_ERROR(
        cudaMemcpy(stage, dev, sizeof(Y), cudaMemcpyDeviceToHost));
    return *stage;
}
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
