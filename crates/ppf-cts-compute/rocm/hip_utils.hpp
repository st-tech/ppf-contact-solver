// File: hip_utils.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef HIP_UTILS_HPP
#define HIP_UTILS_HPP

// The HIP arm of what cuda_utils.hpp is for CUDA: the error-to-crash-kind
// mapping and the HANDLE_ERROR macro a generated entry point's launcher calls
// after its dispatch. It is a SEPARATE FILE from its CUDA sibling and not a
// second arm inside it, because rule (1b) admits no backend file that another
// backend also compiles, and because the two runtimes' error enumerations are
// different types that happen to spell some members alike.
//
// It carries `pinned_scratch` for the same reason its CUDA sibling does: the
// backend stages a host-side transfer through one pinned buffer and grows it
// rather than allocating per call, which is what keeps the steady-state step
// free of device allocation.
#include "main/fatal.hpp"

// Counted by the ROCm arena for the same reason the CUDA one counts: advance()
// logs the per-step delta to show the solve loop reaches a steady state with no
// dynamic device allocation. Plain host globals, which is what lets a backend
// link them; defined once in the ROCm backend's own translation unit.
extern unsigned long long g_device_alloc_count;
extern unsigned long long g_device_free_count;

// THE RUNTIME HEADER IS INCLUDED UNCONDITIONALLY, and that is not the shape its
// CUDA sibling has. `cuda_utils.hpp` guards on `__CUDACC__` because a host-only
// translation unit legitimately includes it; this header is reached only from a
// HIP rendering, which always has the HIP runtime available.
//
// Guarding it on `__HIPCC__` was measured WRONG: under `HIP_PLATFORM=nvidia`
// hipcc drives nvcc, which does not define `__HIPCC__`, so the guard took the
// stub arm, `hip/hip_runtime.h` was never read, and the generated launcher
// failed on an undefined `hipStream_t`. That is the one configuration this port
// can actually EXECUTE on, so the guard broke exactly the leg it needed.
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#if defined(__HIPCC__) || defined(__CUDACC__)

// Which crash sub-kind a `hipError_t` reports as.
//
// THE THREE ACTIONABLE CAUSES ARE THE SAME THREE the CUDA mapping names, and
// they are mapped to the same `FATAL_*` codes on purpose: the crash kind is a
// HOST CONTRACT that the addon's slug table and four i18n catalogs already
// read, so a backend inventing its own code for "the scene does not fit" would
// surface as an unknown kind rather than as an out-of-memory report.
//
// `hipErrorLaunchTimeOut` is kept even though the Windows watchdog and the
// Linux one are reached by different mechanisms: what the code says is that the
// operating system reset the device out from under a kernel, which is true of
// both and is fixed the same way.
static unsigned char fatal_code_for_hip(hipError_t err) {
    if (err == hipErrorOutOfMemory) {
        return FATAL_OOM;
    }
    if (err == hipErrorAssert) {
        return FATAL_DEVICE_ASSERT;
    }
    if (err == hipErrorLaunchTimeOut) {
        return FATAL_WATCHDOG_TIMEOUT;
    }
    return FATAL_CUDA_DRIVER;
}

static void HandleHipError(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        char detail[512];
        snprintf(detail, sizeof(detail), "%s (%s) at %s:%d",
                 hipGetErrorString(err), hipGetErrorName(err), file, line);
        fatal(fatal_code_for_hip(err), detail);
    }
}

#define HIP_HANDLE_ERROR(err) (HandleHipError((err), __FILE__, __LINE__))

// One process-wide pinned staging buffer, grown and never shrunk. Pinned memory
// is what lets a transfer overlap, and reallocating per call would put an
// allocation back in the per-step path that the counters above exist to show is
// empty.
inline void *pinned_scratch(size_t bytes) {
    static void *ptr = nullptr;
    static size_t cap = 0;
    if (cap < bytes) {
        if (ptr) {
            HIP_HANDLE_ERROR(hipHostFree(ptr));
        }
        size_t want = bytes < 256 ? 256 : bytes;
        HIP_HANDLE_ERROR(hipHostMalloc(&ptr, want));
        cap = want;
    }
    return ptr;
}

#else

// The stub arm. The replacement list names no parameter, so a caller's argument
// is discarded as preprocessor tokens rather than compiled, which is what lets a
// host compiler read a file that calls the HIP runtime.
#define HIP_HANDLE_ERROR(err) ((void)0)

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#endif  // __HIPCC__

#endif  // HIP_UTILS_HPP
