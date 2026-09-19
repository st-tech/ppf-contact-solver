// File: common.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef COMMON_HPP
#define COMMON_HPP

// HOST-ONLY, behind the role marker the Metal prologue defines. The shader
// compiler serves no standard library, so an angle include reaching it is a
// compile error. This file is not spliced into any shader, and it IS read by
// an offline translation unit built around a neutral body that names it, which
// is what the guard is for. What survives on that side is the scalar
// vocabulary below; `logging` does not, and is guarded with them.
#ifndef SM_MSL_CONCAT
#include <cstdio>
#include <cstdlib>
#include <stdarg.h>
#endif

#ifdef EPSILON
#undef EPSILON
#endif

#ifdef OVERFLOW
#undef OVERFLOW
#endif

#ifdef FLT_MAX
#undef FLT_MAX
#endif

#ifdef FLT_MIN
#undef FLT_MIN
#endif

#ifndef __device__
#define __device__
#define __host__
#endif

#define EPSILON 1.0e-8f
#define FLT_MAX 1.0e8f
#define FLT_MIN -1.0e8f
#define DT_MIN 1e-5f
#define PI 3.14159265358979323846f
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define WARP_SIZE 32

inline unsigned choose_block_size(unsigned n) {
    if (n <= 32) {
        return 32;
    } else if (n <= 64) {
        return 64;
    } else if (n <= 128) {
        return 128;
    } else {
        return 256;
    }
}

// THE LOG CHANNEL, HOST AND DEVICE-WITH-A-HOST-BEHIND-IT ONLY. It formats with
// `va_list` and `vsnprintf` and calls back into the Rust host, none of which a
// shader has. A shader's diagnostic channel is a device buffer the backend
// reads after the command buffer completes (metal/diagnostics.mm).
#ifndef SM_MSL_CONCAT
namespace logging {
extern "C" {
void print_rust(const char *message);
}

static void info(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list size_args;
    va_copy(size_args, args);
    const int length = vsnprintf(nullptr, 0, fmt, size_args);
    va_end(size_args);
    if (length < 0) {
        va_end(args);
        print_rust("logging::info formatting failed");
        return;
    }
    char *buffer = static_cast<char *>(std::malloc(static_cast<size_t>(length) + 1));
    if (!buffer) {
        va_end(args);
        print_rust("logging::info allocation failed");
        return;
    }
    vsnprintf(buffer, static_cast<size_t>(length) + 1, fmt, args);
    va_end(args);
    print_rust(buffer);
    std::free(buffer);
}
} // namespace logging
#endif  // SM_MSL_CONCAT

#endif
