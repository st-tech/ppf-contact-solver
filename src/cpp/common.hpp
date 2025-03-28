// File: common.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdio>
#include <stdarg.h>

#ifdef EPSILON
#undef EPSILON
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
#define BLOCK_SIZE 256
#define DEBUG_MODE 0

namespace logging {
static char buffer[2048];

extern "C" {
void print_rust(const char *message);
}

static void info(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsprintf(buffer, fmt, args);
    va_end(args);
    print_rust(buffer);
}
} // namespace logging

#endif
