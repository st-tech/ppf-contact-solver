// File: fatal.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef PPF_FATAL_HPP
#define PPF_FATAL_HPP

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Deliberate, unconditional termination on a failed invariant, with the
// reason preserved for the Rust host.
//
// The host registers a `libc::atexit` hook that reads `ppf_fatal_code()` and
// `ppf_fatal_detail()` and writes the terminal `Crashed{kind, detail}` record
// the server reads. `atexit` handlers run on `exit()` and NOT on `abort()`, so
// terminating through `exit(1)` here is what lets an invariant failure report
// itself; an `abort()` leaves the server with only "the process is gone".
//
// The two costs of `exit(1)` over `abort()`, both accepted: no core dump (the
// release profile sets `strip = "symbols"`, so a released core carries no
// symbols anyway), and static destructors run (which the CUDA error path
// already does).
//
// This changes nothing about WHEN the solver stops. Every caller terminates
// exactly as unconditionally as before, prints exactly the same text, and no
// check, assert or CCD guard is weakened by routing its exit through here.
//
// Header-only because the CUDA Makefile compiles one `.cu` per source
// directory plus two fixed `.cpp` files, so a separate translation unit for
// this would have to be threaded through both that Makefile and the Windows
// `build.bat`.

// Fatal-exit reason code. Mirrors `ppf_cts_formats::status::error_code`;
// defined in cpp/main/main.cu for the CUDA backend and in cpp_emul/main.cpp
// for the emulator.
extern "C" unsigned char g_ppf_fatal_code;

// First line of the fatal report, for the one-line crash detail the addon
// panel shows. The full multi-line report goes to stderr and reaches the user
// through the server's stderr tail. Empty when nothing was recorded, in which
// case the host falls back to a generic detail naming the code.
extern "C" char g_ppf_fatal_detail[512];

enum {
    PPF_FATAL_INIT_INTERSECTION = 1,
    PPF_FATAL_OOM = 2,
    PPF_FATAL_CUDA_DRIVER = 3,
    PPF_FATAL_SOLVER_INVARIANT = 4,
    PPF_FATAL_DEVICE_ASSERT = 5,
    PPF_FATAL_WATCHDOG_TIMEOUT = 6,
};

// Record `fmt`'s first line as the fatal detail. Split out so the CUDA error
// handler, which composes its detail from `cudaGetErrorString` rather than
// from its own format string, can reuse it.
inline void ppf_fatal_set_detail(const char *text) {
    std::snprintf(g_ppf_fatal_detail, sizeof(g_ppf_fatal_detail), "%s", text);
    if (char *nl = std::strchr(g_ppf_fatal_detail, '\n')) {
        *nl = '\0';
    }
}

// Print the report to stderr verbatim, keep its first line as the detail,
// stamp `code`, and exit(1).
[[noreturn]] inline void ppf_fatal(unsigned char code, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list detail_args;
    va_copy(detail_args, args);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    char first_line[sizeof(g_ppf_fatal_detail)];
    std::vsnprintf(first_line, sizeof(first_line), fmt, detail_args);
    va_end(detail_args);
    ppf_fatal_set_detail(first_line);

    // stdout is a redirected file and therefore block buffered. `exit()`
    // flushes it as well; flushing both here fixes the order the two logs
    // land in, which is what a reader comparing them depends on.
    std::fflush(stdout);
    std::fflush(stderr);
    g_ppf_fatal_code = code;
    std::exit(1);
}

#endif
