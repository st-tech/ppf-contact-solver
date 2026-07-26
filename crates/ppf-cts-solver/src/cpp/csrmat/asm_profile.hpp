// File: asm_profile.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ASM_PROFILE_HPP
#define ASM_PROFILE_HPP

#include "../simplelog/SimpleLog.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

// Opt-in attribution of the "asm contact" timer, enabled by setting
// PPF_ASM_PROFILE. The contact assembly is a sequence of DISPATCH_START /
// DISPATCH_END blocks and each of those synchronizes the stream, so plain
// host wall-clock around a phase measures that phase and nothing else; no
// extra synchronization is introduced for the measurement. Off by default,
// and the environment is read once, so a normal run pays one predicted
// branch per phase and nothing else.
namespace asm_profile {

inline bool enabled() {
    static const bool on = std::getenv("PPF_ASM_PROFILE") != nullptr;
    return on;
}

using Clock = std::chrono::steady_clock;

inline Clock::time_point tick() { return Clock::now(); }

inline double ms_since(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since)
        .count();
}

// Order statistics of a per-row width array, reported as a single line. The
// mean alone hides the shape that matters here: the assembly cost is driven by
// the WIDEST rows, not the average one, so the tail percentiles and the argmax
// row are the payload. `values` is consumed (partially sorted in place).
inline void report_widths(const char *tag, std::vector<unsigned> &values) {
    if (values.empty()) {
        SimpleLog::message("[asm-prof] %s: empty", tag);
        return;
    }
    const size_t n = values.size();
    double total = 0.0;
    unsigned argmax = 0;
    unsigned peak = 0;
    for (size_t i = 0; i < n; ++i) {
        total += values[i];
        if (values[i] > peak) {
            peak = values[i];
            argmax = (unsigned)i;
        }
    }
    auto pct = [&](double q) {
        size_t k = (size_t)(q * (double)(n - 1));
        std::nth_element(values.begin(), values.begin() + k, values.end());
        return values[k];
    };
    // Ascending order so each nth_element only has to sort the remaining tail.
    const unsigned p50 = pct(0.50);
    const unsigned p90 = pct(0.90);
    const unsigned p99 = pct(0.99);
    const unsigned p999 = pct(0.999);
    SimpleLog::message("[asm-prof] %s: n=%zu sum=%.0f mean=%.2f p50=%u p90=%u "
                       "p99=%u p99.9=%u max=%u argmax_row=%u",
                       tag, n, total, total / (double)n, p50, p90, p99, p999,
                       peak, argmax);
}

} // namespace asm_profile

#endif
