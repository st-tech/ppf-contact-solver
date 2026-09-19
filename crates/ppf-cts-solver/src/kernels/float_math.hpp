// File: float_math.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef FLOAT_MATH_HPP
#define FLOAT_MATH_HPP

// THIS FILE IS THE CUDA AND HOST ARM OF `namespace fmath`, AND THE GUARD IS HOW
// IT SAYS SO. Metal's arm of the same namespace is in the Metal prologue
// (kMslMacroSeam, ppf-cts-compute/metal/shader_compiler.mm), which declares the
// whole table itself.
//
// Everything below picks between a CUDA intrinsic and a libm call on
// `__CUDA_ARCH__`, and the Metal shader compiler is neither: it serves no
// standard library, so the two includes alone are a compile error there, and
// the bodies past them would name functions it does not have. What a neutral
// kernel body calls is the same five names either way, so the split is in which
// file supplies them and not in what a body may write.
//
// A NAME ADDED HERE HAS TO BE ADDED TO THE METAL PROLOGUE IN THE SAME CHANGE,
// or a body naming it compiles on two backends out of three and the shader
// compile fails at run time, after every build leg has gone green. Rule 6 of
// check-shared-wiring.py reads this file as part of the reference table for
// exactly that reason, so the omission fails the build rather than the run.
#ifndef SM_MSL_CONCAT

#include <cassert>
#include <cmath>

// Some of the headers that include this are also compiled without nvcc, where
// the execution-space qualifiers do not exist.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

// Transcendentals that stay in single precision on the device.
//
// The solver runs single precision on the GPU. The library sinf, cosf, expf and
// logf do not honor that: their accurate paths reduce or scale the argument
// with 64-bit integer and double arithmetic, so a kernel that merely calls one
// of them emits I2F.F64 and DMUL. Nothing in the source says `double`, which is
// why this is enforced by reading the compiled SASS at build time rather than
// by review.
//
// The replacements are the hardware special-function unit, which is float
// throughout. The cost is accuracy, roughly 2^-21 relative against the library
// versions' 2^-24. That is affordable here only because every caller feeds a
// quantity whose own uncertainty is larger, and each one says why at its call
// site. It would not be affordable for a quantity the solver differentiates or
// compares against a tight tolerance.
//
// The bounded and periodic forms are separate on purpose. The special-function
// unit loses its argument's low bits as the argument grows, so it is accurate
// only over a few periods. A caller that can guarantee a bounded argument
// should say so and pay nothing, and one that cannot must reduce first. Making
// that a choice at the call site keeps the reduction from being either silently
// skipped or silently paid for.
namespace fmath {

// Largest magnitude at which the special-function unit is used directly. Two
// full turns covers every bounded caller here with room to spare, and the
// assert traps a caller that assumed a bound it does not have. The production
// device build keeps asserts live, so this is a real check and not a comment.
//
// NOT PART OF THE SEAM TABLE, and the only name in this file that is not. The
// Metal arm takes the library's accurate paths, which need no such bound and
// declare no such constant, so a kernel body naming this compiles on two
// backends and fails the shader compile on the third. kernelgen.py refuses a
// body that names it, which makes that a diagnostic at the neutral file and
// line rather than a run-time failure.
enum : int { BOUNDED_LIMIT = 7 };

// cos and sin for an argument the caller knows is bounded.
__device__ __host__ inline float cos_bounded(float x) {
    assert(fabsf(x) <= float(BOUNDED_LIMIT) &&
           "fmath::cos_bounded called with an unbounded argument");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __cosf(x);
#else
    return std::cos(x);
#endif
}

__device__ __host__ inline float sin_bounded(float x) {
    assert(fabsf(x) <= float(BOUNDED_LIMIT) &&
           "fmath::sin_bounded called with an unbounded argument");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __sinf(x);
#else
    return std::sin(x);
#endif
}

// sin for an argument of any size, reduced into one turn first. The reduction
// is a float multiply, floor and subtract, so it costs the low bits of a large
// argument; a caller that needs those bits needs a different formulation, not a
// more accurate reduction here.
__device__ __host__ inline float sin_periodic(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const float kTwoPi = 6.28318530718f;
    float turns = x * (1.0f / kTwoPi);
    return __sinf(x - kTwoPi * floorf(turns));
#else
    return std::sin(x);
#endif
}

__device__ __host__ inline float exp(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __expf(x);
#else
    return std::exp(x);
#endif
}

__device__ __host__ inline float log(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __logf(x);
#else
    return std::log(x);
#endif
}

} // namespace fmath

#endif  // SM_MSL_CONCAT

#endif  // FLOAT_MATH_HPP
