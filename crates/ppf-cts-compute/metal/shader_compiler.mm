// File: shader_compiler.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Implementation of the concatenating MSL assembler declared in
// shader_compiler.hpp, plus the MSL half of the backend macro seam.
//
// No Metal type appears here: compilation goes through context_compile_library,
// which owns the one mathMode = Safe options helper. This file is Objective-C++
// only because it is part of the Metal backend's build.

#include "shader_compiler.hpp"

#include "diagnostics.hpp"
#include "metal_context.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace metal_backend {

namespace {

// ---------------------------------------------------------------------------
// Prologue part 1: the MSL standard library.
//
// Kept separate from the macro seam so a diagnostic inside either is attributed
// to the right one.
// ---------------------------------------------------------------------------
const char *const kMslStandardInclude = R"MSL(
#include <metal_stdlib>
#include <metal_math>
#include <metal_integer>
#include <metal_atomic>
#include <metal_simdgroup>

using namespace metal;
)MSL";

// ---------------------------------------------------------------------------
// Prologue part 2: the MSL half of the backend macro seam.
//
// Everything backend-specific about the dialect is confined to these macros.
// The shared device sources contain no backend conditional at all, and the
// count of macros here is the real measure of how thin the seam is. The CUDA
// half expands the same names to the nvcc spellings, and the host half to the
// std:: spellings.
//
// The inventory is the one the kernel-2 conformance suite proved on both
// backends (32 PASS, 0 FAIL with no backend-specific fixes), plus the three
// names the ACCD area added (SM_DIV, SM_NEXTAFTER, SM_ISINF). Names are not
// invented here: a rename would have to be carried through every kernel later.
// ---------------------------------------------------------------------------
const char *const kMslMacroSeam = R"MSL(
// Role marker. Metal has no filesystem for the shader compiler: the driver
// hands newLibraryWithSource: one concatenated string, so an #include in shared
// device code is a compile error. The host splices those files in itself, in
// dependency order.
//
// A QUOTED include does not need this marker: the assembler neutralizes the
// line as it splices the segment (neutralize_quoted_includes below), so a
// kernel body names its dependency unconditionally and carries no conditional.
// The marker is for the headers whose include block gates more than that, an
// angle <cmath> among them, which the shader compiler cannot serve either.
#define SM_MSL_CONCAT 1

// Default file id, in effect until the first segment overrides it. Zero means
// "not a registered source file", which is what the prologue itself is.
#define DIAG_FILE_ID 0

#define SM_INLINE inline

// The execution-space names, all of them plain `inline` here. MSL has no
// execution-space qualifiers at all: a shader source is device code and nothing
// else, so there is nothing for the distinction to select. It is nvcc that needs
// three names, because a body marked for both spaces is compiled for both, and
// widening a device-only body is one of the ways float64 reaches the GPU
// (cpp/seam/seam_cuda.cuh states the rule with the reason). A kernel therefore
// names the spelling it wants and this backend maps all of them onto `inline`.
//
// SM_INLINE above is the name a header defines for itself when it carries its
// own fallback block; the three below are the ones cpp/seam supplies centrally.
#define SM_INLINE_DEVICE inline
#define SM_INLINE_DEVICE_HOST inline

// Address spaces. MSL requires one on every pointer and reference; the host
// prologue expands all three to nothing.
#define SM_THREAD thread
#define SM_DEVICE device
#define SM_THREADGROUP threadgroup

// Fixed arena ABI: arenas occupy slots 0..28, inline handles use 29, and
// diagnostics use 30.
#define ARENA_ARGS                                                        \
    device uchar *arena_0 [[buffer(0)]],                                  \
    device uchar *arena_1 [[buffer(1)]],                                  \
    device uchar *arena_2 [[buffer(2)]],                                  \
    device uchar *arena_3 [[buffer(3)]],                                  \
    device uchar *arena_4 [[buffer(4)]],                                  \
    device uchar *arena_5 [[buffer(5)]],                                  \
    device uchar *arena_6 [[buffer(6)]],                                  \
    device uchar *arena_7 [[buffer(7)]],                                  \
    device uchar *arena_8 [[buffer(8)]],                                  \
    device uchar *arena_9 [[buffer(9)]],                                  \
    device uchar *arena_10 [[buffer(10)]],                                \
    device uchar *arena_11 [[buffer(11)]],                                \
    device uchar *arena_12 [[buffer(12)]],                                \
    device uchar *arena_13 [[buffer(13)]],                                \
    device uchar *arena_14 [[buffer(14)]],                                \
    device uchar *arena_15 [[buffer(15)]],                                \
    device uchar *arena_16 [[buffer(16)]],                                \
    device uchar *arena_17 [[buffer(17)]],                                \
    device uchar *arena_18 [[buffer(18)]],                                \
    device uchar *arena_19 [[buffer(19)]],                                \
    device uchar *arena_20 [[buffer(20)]],                                \
    device uchar *arena_21 [[buffer(21)]],                                \
    device uchar *arena_22 [[buffer(22)]],                                \
    device uchar *arena_23 [[buffer(23)]],                                \
    device uchar *arena_24 [[buffer(24)]],                                \
    device uchar *arena_25 [[buffer(25)]],                                \
    device uchar *arena_26 [[buffer(26)]],                                \
    device uchar *arena_27 [[buffer(27)]],                                \
    device uchar *arena_28 [[buffer(28)]]

#define ARENA_PTR(id, offset)                                             \
    ((id) == 0u    ? arena_0 + (offset)                                   \
     : (id) == 1u  ? arena_1 + (offset)                                   \
     : (id) == 2u  ? arena_2 + (offset)                                   \
     : (id) == 3u  ? arena_3 + (offset)                                   \
     : (id) == 4u  ? arena_4 + (offset)                                   \
     : (id) == 5u  ? arena_5 + (offset)                                   \
     : (id) == 6u  ? arena_6 + (offset)                                   \
     : (id) == 7u  ? arena_7 + (offset)                                   \
     : (id) == 8u  ? arena_8 + (offset)                                   \
     : (id) == 9u  ? arena_9 + (offset)                                   \
     : (id) == 10u ? arena_10 + (offset)                                  \
     : (id) == 11u ? arena_11 + (offset)                                  \
     : (id) == 12u ? arena_12 + (offset)                                  \
     : (id) == 13u ? arena_13 + (offset)                                  \
     : (id) == 14u ? arena_14 + (offset)                                  \
     : (id) == 15u ? arena_15 + (offset)                                  \
     : (id) == 16u ? arena_16 + (offset)                                  \
     : (id) == 17u ? arena_17 + (offset)                                  \
     : (id) == 18u ? arena_18 + (offset)                                  \
     : (id) == 19u ? arena_19 + (offset)                                  \
     : (id) == 20u ? arena_20 + (offset)                                  \
     : (id) == 21u ? arena_21 + (offset)                                  \
     : (id) == 22u ? arena_22 + (offset)                                  \
     : (id) == 23u ? arena_23 + (offset)                                  \
     : (id) == 24u ? arena_24 + (offset)                                  \
     : (id) == 25u ? arena_25 + (offset)                                  \
     : (id) == 26u ? arena_26 + (offset)                                  \
     : (id) == 27u ? arena_27 + (offset)                                  \
                    : arena_28 + (offset))

// Namespace-scope constants must sit in the `constant` address space in MSL. A
// bare `constexpr float` at file scope is rejected by the Metal compiler, and
// nvcc imposes the same restriction on namespace-scope constexpr in device
// code, so the shared spelling satisfies both. Never `__constant__` on the CUDA
// side: that is device memory the host reference cannot read, and it silently
// yields 0 on the host.
#define SM_CONSTANT constant

// Math. MSL puts these in namespace metal (pulled in above), the host in std.
// The two namespaces do not overlap, so every call site in shared code goes
// through a macro.
#define SM_ABS fabs
#define SM_MAX fmax
#define SM_MIN fmin
#define SM_FMA fma
#define SM_ISNAN isnan
#define SM_ISINF isinf
#define SM_NEXTAFTER(a, b) nextafter((a), (b))
#define SM_ACOS acos
#define SM_COS cos
// The bounded-argument sine and cosine. The distinction exists for CUDA's sake:
// there the library sinf and cosf reduce their argument in 64-bit integer and
// double arithmetic, so a kernel that merely calls one emits FP64 and trips the
// release build's SASS guard, and the CUDA seam therefore maps these to
// fmath::sin_bounded / fmath::cos_bounded, the float-throughout
// special-function unit. MSL has no double at all, so that hazard cannot exist
// here and the accurate library forms are both correct and cheap enough. They
// are also slightly more accurate than CUDA's, which is why the two call sites
// that use them (the Rodrigues map and the polar fit, both in
// pdrd_rigid.kernel.cpp) are documented there as the places the backends are
// not bit-identical.
#define SM_SIN_BOUNDED sin
#define SM_COS_BOUNDED cos
// The unbounded-argument sine, whose one caller is the wind ramp in
// air_damper.hpp: its argument is 30 times the simulation clock, so it leaves
// the range a special-function unit resolves within a fraction of a second.
// The CUDA seam meets the contract by reducing into one turn in float and then
// calling the special-function unit; MSL's library sine is accurate over the
// whole range, so it meets it directly. This backend evaluates the ramp on the
// HOST, once per dispatch, and hands the shader the finished wind vector, so
// the name is here to let the shared header compile rather than to be called.
#define SM_SIN_PERIODIC(x) sin(x)
#define SM_CLZ(value) clz(value)
#define SM_NUMERIC_MAX(T) T(3.402823466e38f)
#define SM_LOG(value) log(value)
#define SM_INFINITY as_type<float>(0x7f800000u)
#define SM_ATAN2(y, x) atan2((y), (x))
#define SM_EXP(value) exp(value)
#define FP_INT_ABS(value) ((value) < 0 ? -(value) : (value))
#define FP_INT_MAX(a, b) ((a) > (b) ? (a) : (b))
#define FP_INT_MIN(a, b) ((a) < (b) ? (a) : (b))

// Square root and division are the correctly rounded forms, and that is a
// correctness setting rather than caution.
//
// Measured: mathMode = Safe does NOT buy a correctly rounded sqrt on this
// device. Plain `sqrt` returned the correctly rounded root on only 53283 of
// 65536 threads, worst deviation 1 ulp, where nvcc for sm_89 returned it on
// 65536 of 65536 with worst deviation 0. `precise::sqrt` and `precise::divide`
// are correctly rounded in ALL math modes, which is also why they survive a
// lost compile flag, and in the ACCD area `metal_precise` was the only one of
// six backends bit-identical to the host reference on all 47 configurations.
//
// They are the whole-unit spelling, not a per-site one, because the backend
// compiles ONE concatenated translation unit: there is no point at which a
// different definition could be swapped in for a subset of the sources. Since
// CUDA's `/` and `sqrtf` are already correctly rounded, choosing the precise
// forms here is what makes the two backends agree by construction. If a future
// measurement shows the cost matters somewhere it is not load-bearing, the
// answer is a second macro name, never a quietly reduced default.
#define SM_SQRT precise::sqrt
#define SM_DIV(a, b) precise::divide((a), (b))

// High half of a signed 32x32 multiply. CUDA spells this __mulhi; the host has
// no intrinsic and widens to 64-bit instead.
#define SM_MULHI(a, b) mulhi((a), (b))

// Bit casts.
#define SM_AS_UINT(x) as_type<uint>((x))
#define SM_AS_FLOAT(x) as_type<float>((x))

#define SM_POPCOUNT(x) popcount((x))
// Measured: MSL clz(0u) returns 32, matching CUDA __clz(0), so no zero special
// case is needed on either backend.
#define SM_CLZ(x) clz((x))

// SIMD geometry. Measured 32 on Apple silicon (threadExecutionWidth and the
// threads_per_simdgroup attribute agree, including in a live 26-lane trailing
// simdgroup); entry points re-check it at run time and raise the diagnostic
// flag rather than trusting this constant.
#define SM_SIMD_WIDTH 32u

// Lane index. MSL has no expression form: the lane index arrives only as the
// [[thread_index_in_simdgroup]] kernel-parameter attribute, so shared code
// derives it from the thread-in-threadgroup index instead. The identity is
// measured (simdgroups are contiguous 32-thread runs) and entry points
// cross-check the derivation against the real attribute.
#define SM_LANE_ID(tid_in_tg) ((tid_in_tg) & (SM_SIMD_WIDTH - 1u))
#define SM_SHUFFLE_DOWN(value, offset) simd_shuffle_down((value), (offset))
#define SM_SHUFFLE_UP(value, offset) simd_shuffle_up((value), (offset))

#define SM_THREADGROUP_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)

// ---------------------------------------------------------------------------
// Atomics.
//
// memory_order_relaxed is not a shortcut, it is the only order this device
// accepts: memory_order_acquire / release / acq_rel are not declared, and
// atomic_fetch_add_explicit has no seq_cst overload (measured). Relaxed is also
// the right semantic here, where only the atomicity of the read-modify-write
// matters and never the ordering against other addresses; the host reads every
// buffer only after waitUntilCompleted.
//
// The helpers are FUNCTIONS, not macro bodies, because MSL makes the address
// space part of the pointer type and gives no way to template over it. Function
// overloading is the one mechanism that does work, so each operation has a
// `device` and a `threadgroup` overload and the macro dispatches by argument
// type.
//
// Every operation returns the PRE-IMAGE, which is CUDA's native convention and
// the one the shared contract adopts.
// ---------------------------------------------------------------------------

#define SM_ATOMIC_UINT atomic_uint
#define SM_ATOMIC_FLOAT atomic_float

inline uint sm_atomic_load_uint(device atomic_uint *p) {
    return atomic_load_explicit(p, memory_order_relaxed);
}
inline uint sm_atomic_load_uint(threadgroup atomic_uint *p) {
    return atomic_load_explicit(p, memory_order_relaxed);
}

inline void sm_atomic_store_uint(device atomic_uint *p, uint v) {
    atomic_store_explicit(p, v, memory_order_relaxed);
}
inline void sm_atomic_store_uint(threadgroup atomic_uint *p, uint v) {
    atomic_store_explicit(p, v, memory_order_relaxed);
}

inline uint sm_atomic_add_uint(device atomic_uint *p, uint v) {
    return atomic_fetch_add_explicit(p, v, memory_order_relaxed);
}
inline uint sm_atomic_add_uint(threadgroup atomic_uint *p, uint v) {
    return atomic_fetch_add_explicit(p, v, memory_order_relaxed);
}

inline uint sm_atomic_max_uint(device atomic_uint *p, uint v) {
    return atomic_fetch_max_explicit(p, v, memory_order_relaxed);
}
inline uint sm_atomic_max_uint(threadgroup atomic_uint *p, uint v) {
    return atomic_fetch_max_explicit(p, v, memory_order_relaxed);
}

// Returns the PRE-IMAGE, which is CUDA atomicCAS semantics, not MSL's own. MSL
// gives a weak compare-exchange returning bool and updating `expected` in
// place; a weak CAS may fail spuriously, so a bare wrapper would sometimes
// report "the value was what you expected" while having stored nothing. The
// retry loop turns it into the strong CAS the shared contract promises.
inline uint sm_atomic_cas_uint(device atomic_uint *p, uint expected,
                               uint desired) {
    uint observed = expected;
    while (!atomic_compare_exchange_weak_explicit(p, &observed, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
        if (observed != expected) {
            return observed;  // genuine failure: report what was there
        }
        observed = expected;  // spurious failure: retry
    }
    return expected;
}
inline uint sm_atomic_cas_uint(threadgroup atomic_uint *p, uint expected,
                               uint desired) {
    uint observed = expected;
    while (!atomic_compare_exchange_weak_explicit(p, &observed, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
        if (observed != expected) {
            return observed;
        }
        observed = expected;
    }
    return expected;
}

// DEVICE ONLY, and deliberately so. A threadgroup atomic_float DECLARES fine
// but has no usable operation on this device: atomic_load_explicit,
// atomic_store_explicit and atomic_fetch_add_explicit all fail to resolve for
// it (measured). There is therefore no threadgroup overload, and a kernel that
// tries to float-accumulate in threadgroup memory gets a compile error naming
// the missing overload instead of a silent wrong path. The sanctioned way to do
// it is SM_ATOMIC_ADD_FLOAT_VIA_CAS over a threadgroup atomic_uint.
inline float sm_atomic_add_float(device atomic_float *p, float v) {
    return atomic_fetch_add_explicit(p, v, memory_order_relaxed);
}

// Float accumulation carried by an integer CAS loop over the bit pattern. This
// is the portable fallback and the only threadgroup-memory float accumulator
// available here. Returns the PRE-IMAGE as a float.
inline float sm_atomic_add_float_via_cas(device atomic_uint *p, float v) {
    uint oldbits = sm_atomic_load_uint(p);
    for (;;) {
        float oldval = as_type<float>(oldbits);
        uint newbits = as_type<uint>(oldval + v);
        uint seen = sm_atomic_cas_uint(p, oldbits, newbits);
        if (seen == oldbits) {
            return oldval;
        }
        oldbits = seen;
    }
}
inline float sm_atomic_add_float_via_cas(threadgroup atomic_uint *p, float v) {
    uint oldbits = sm_atomic_load_uint(p);
    for (;;) {
        float oldval = as_type<float>(oldbits);
        uint newbits = as_type<uint>(oldval + v);
        uint seen = sm_atomic_cas_uint(p, oldbits, newbits);
        if (seen == oldbits) {
            return oldval;
        }
        oldbits = seen;
    }
}

#define SM_ATOMIC_LOAD_UINT(p) sm_atomic_load_uint((p))
#define SM_ATOMIC_STORE_UINT(p, v) sm_atomic_store_uint((p), (v))
#define SM_ATOMIC_ADD_UINT(p, v) sm_atomic_add_uint((p), (v))
#define SM_ATOMIC_ADD_FLOAT(p, v) sm_atomic_add_float((p), (v))
#define SM_ATOMIC_CAS_UINT(p, expected, desired)                               \
    sm_atomic_cas_uint((p), (expected), (desired))
#define SM_ATOMIC_MAX_UINT(p, v) sm_atomic_max_uint((p), (v))
#define SM_ATOMIC_ADD_FLOAT_VIA_CAS(p, v) sm_atomic_add_float_via_cas((p), (v))

// ---------------------------------------------------------------------------
// Width-8 segmented reduction.
//
// The XOR form is the PRIMARY one because an xor by 4, 2 or 1 only ever flips
// the low three lane bits, so it structurally CANNOT cross an 8-lane segment
// boundary on any backend. It needs no width argument, no clamp and no guarding
// of the segment tail, and it leaves the segment sum in EVERY lane.
//
// The accumulation order at the head lane is
//     ((v0+v4) + (v2+v6)) + ((v1+v5) + (v3+v7))
// which is bit-for-bit the order CUDA's width-8 tree produces, and that exact
// order is what keeps the solver's CG_CURVATURE_NOISE_SLACK bound calibrated
// across the port: the slack is measured against the round-off of THIS sum.
// ---------------------------------------------------------------------------

inline float sm_seg8_reduce_xor(float v) {
    v += simd_shuffle_xor(v, (ushort)4);
    v += simd_shuffle_xor(v, (ushort)2);
    v += simd_shuffle_xor(v, (ushort)1);
    return v;
}

// The shfl_down form, kept only so a conformance case can compare the two.
// Metal's simd_shuffle_down does NOT clamp at a segment boundary the way CUDA's
// width-8 __shfl_down_sync does: it is a sliding window over the whole
// simdgroup, so ONLY lane%8==0 holds the segment sum here, while on CUDA every
// lane holds something well defined. That asymmetry is why the xor form is
// primary.
inline float sm_seg8_reduce_down(float v) {
    v += simd_shuffle_down(v, (ushort)4);
    v += simd_shuffle_down(v, (ushort)2);
    v += simd_shuffle_down(v, (ushort)1);
    return v;
}

#define SM_SEG8_REDUCE(v) sm_seg8_reduce_xor((v))
#define SM_SEG8_REDUCE_DOWN(v) sm_seg8_reduce_down((v))

// Full-simdgroup sum. NOT bit-exact comparable across backends by contract:
// Metal's simd_sum is a hardware reduction whose association order is
// unspecified, and CUDA has no equivalent instruction at all (its prologue
// spells this as an explicit 5-step butterfly). Use SM_SEG8_REDUCE, or an
// explicit tree, whenever the comparison has to be bitwise.
#define SM_SIMD_SUM(v) simd_sum((v))

// Ballot, normalized to the low 32 bits as a uint. Metal's simd_vote::vote_t is
// 64 bits wide (measured) while CUDA's __ballot_sync returns 32, so the shared
// spelling names one width, and 32 is the one that matches the lane count.
// NOTE the measured Metal behavior: simd_ballot does NOT restrict the vote to
// ACTIVE lanes under most forms of divergence, so a ballot mask is not
// interchangeable across backends without folding the validity condition into
// an opaque predicate.
inline uint sm_simd_ballot(bool pred) {
    simd_vote v = simd_ballot(pred);
    return uint(static_cast<simd_vote::vote_t>(v) & 0xffffffffu);
}
#define SM_SIMD_BALLOT(pred) sm_simd_ballot((pred))

// ===========================================================================
// The `fmath` namespace: the same table as ordinary C++ names.
//
// A neutral kernel source (`*.kernel.cpp`, rendered by ppf-cts-compute/seam/kernelgen.py)
// has no preprocessor at all, so it cannot name a macro. It calls these
// instead, and each backend prologue defines them with that backend's
// spelling. The replacement text is the same as the macro directly above it,
// name for name, and cpp/seam/seam_cuda.cuh and cpp/seam/seam_host.h carry the
// other two columns.
//
// Every name below is QUALIFIED against `metal::`. The using-directive at the
// top of this unit puts the library names at global scope, so an unqualified
// `log` inside `namespace fmath` would find `fmath::log` first and recurse.
// ===========================================================================
namespace fmath {

// --- Arithmetic ------------------------------------------------------------

inline float abs(float x) { return metal::fabs(x); }
inline float max(float a, float b) { return metal::fmax(a, b); }
inline float min(float a, float b) { return metal::fmin(a, b); }
inline float fma(float a, float b, float c) { return metal::fma(a, b, c); }
inline float acos(float x) { return metal::acos(x); }
inline float atan2(float y, float x) { return metal::atan2(y, x); }
inline float pow(float x, float y) { return metal::pow(x, y); }
inline bool isnan(float x) { return metal::isnan(x); }
inline bool isinf(float x) { return metal::isinf(x); }
inline float nextafter(float a, float b) { return metal::nextafter(a, b); }

// Square root and division are the correctly rounded forms, and that is a
// correctness setting rather than caution: plain `sqrt` returned the correctly
// rounded root on only 53283 of 65536 threads on this device, where nvcc for
// sm_89 returned it on 65536 of 65536. The reasoning is written out beside
// SM_SQRT above; these two names must not be given the plain forms.
inline float sqrt(float x) { return metal::precise::sqrt(x); }
inline float div(float a, float b) { return metal::precise::divide(a, b); }

// The bounded-argument sine and cosine. The distinction exists for CUDA's
// sake, where the library forms emit FP64; MSL has no double at all, so the
// accurate library forms are both correct and cheap enough here.
inline float cos_bounded(float x) { return metal::cos(x); }
inline float sin_bounded(float x) { return metal::sin(x); }
inline float sin_periodic(float x) { return metal::sin(x); }
inline float log(float x) { return metal::log(x); }
inline float exp(float x) { return metal::exp(x); }

inline float infinity() { return as_type<float>(0x7f800000u); }
inline float float_max() { return 3.402823466e38f; }

}  // namespace fmath

// --- Bit intrinsics --------------------------------------------------------
//
// `bits::` rather than `compute::`: these are integer, so `fmath::` is the
// wrong side of the float line, and they compute the same value on every
// target, so `compute::` would overstate what varies.

namespace bits {

// Measured: MSL clz(0u) returns 32, matching CUDA __clz(0), so no zero special
// case is needed on either backend.
inline int clz(uint value) { return int(metal::clz(value)); }
inline uint popcount(uint value) { return uint(metal::popcount(value)); }
// Counts a 64-bit `compute::ballot_t`. Unreachable on this target, where that
// type is `uint`; it exists so a neutral body counting a ballot compiles here
// and on a target whose subgroup is 64 lanes.
inline uint popcount(ulong value) { return uint(metal::popcount(value)); }

}  // namespace bits

// --- SIMD geometry ---------------------------------------------------------
//
// `compute::`, because a lane is exactly what differs per backend: the width,
// the shuffle and the ballot are the hardware's, and a target with one thread
// per body has no meaning for any of them at all.

namespace compute {

// Measured 32 on Apple silicon (threadExecutionWidth and the
// threads_per_simdgroup attribute agree, including in a live 26-lane trailing
// simdgroup); entry points re-check it at run time and raise the diagnostic
// flag rather than trusting this constant. Namespace-scope constants must sit
// in the `constant` address space here.
constant constexpr uint simd_width = 32u;

// Generic over the shuffled type, matching cpp/seam/seam_cuda.cuh: a lane
// shuffle moves a value between lanes and does not care what the value means,
// and warp_reduce (primitives/reduce.kernel.cpp) is a template over the
// reduced type. metal::simd_shuffle_down is itself templated, so this is the
// same replacement text SM_SHUFFLE_DOWN carries, one name removed.
template <class T>
inline T shuffle_down(T value, uint offset) {
    return simd_shuffle_down(value, ushort(offset));
}
template <class T>
inline T shuffle_up(T value, uint offset) {
    return simd_shuffle_up(value, ushort(offset));
}
// THE BALLOT'S TYPE IS NAMED, for the reason seam_cuda.cuh states at its own
// copy: a ballot is one bit per lane, so its width follows `simd_width` rather
// than the target's word size. A Metal SIMD group is 32 lanes, so this is
// `uint` and the emitted shader is unchanged.
using ballot_t = uint;

inline ballot_t simd_ballot(bool predicate) { return sm_simd_ballot(predicate); }

inline void threadgroup_barrier() {
    metal::threadgroup_barrier(mem_flags::mem_threadgroup);
}

// --- Atomics ---------------------------------------------------------------
//
// `compute::` for the same reason the lane names above are: what an atomic
// read-modify-write costs and how the address is typed is the backend's, and
// a target running one thread through a body reaches the same answer with a
// plain read, add and write back.
//
// The address space is part of the pointer type here and there is no way to
// template over it, so each operation has a `device` and a `threadgroup`
// overload. Overload resolution is what a neutral kernel body relies on: it
// writes compute::atomic_add(p, v) and the address space of `p` picks the body.
//
// Every operation returns the PRE-IMAGE, which is CUDA's native convention and
// the one the shared contract adopts.

using atomic_uint_t = atomic_uint;
using atomic_float_t = atomic_float;

inline uint atomic_load(device atomic_uint_t *p) {
    return sm_atomic_load_uint(p);
}
inline uint atomic_load(threadgroup atomic_uint_t *p) {
    return sm_atomic_load_uint(p);
}
inline void atomic_store(device atomic_uint_t *p, uint v) {
    sm_atomic_store_uint(p, v);
}
inline void atomic_store(threadgroup atomic_uint_t *p, uint v) {
    sm_atomic_store_uint(p, v);
}
inline uint atomic_add(device atomic_uint_t *p, uint v) {
    return sm_atomic_add_uint(p, v);
}
inline uint atomic_add(threadgroup atomic_uint_t *p, uint v) {
    return sm_atomic_add_uint(p, v);
}

// DEVICE ONLY, and deliberately so. A threadgroup atomic_float declares fine
// but has no usable operation on this device (measured), so a kernel that
// tries to float-accumulate in threadgroup memory gets a compile error naming
// the missing overload instead of a silent wrong path.
inline float atomic_add(device atomic_float_t *p, float v) {
    return sm_atomic_add_float(p, v);
}

// A LARGEST-WINS ACCUMULATOR OVER UINT-TYPED STORAGE, FOR NON-NEGATIVE VALUES.
//
// THE STORAGE IS `atomic_uint_t` BECAUSE THIS TARGET IS THE REASON IT HAS TO
// BE. There is no float atomic max in MSL; there is an integer one, and the
// IEEE-754 bit pattern of a NON-NEGATIVE float orders the same way as the
// float, so the integer maximum of the bits is the maximum of the values. A
// negative argument compares inverted, which is why the callers pass a
// magnitude and the other two prologues state the same precondition.
//
// It returns nothing so that no neutral body needs a bitcast: the pattern this
// serves converts the bits back to float on the HOST after the readback.
inline void atomic_max(device atomic_uint_t *p, float v) {
    sm_atomic_max_uint(p, as_type<uint>(v));
}
inline void atomic_max(threadgroup atomic_uint_t *p, float v) {
    sm_atomic_max_uint(p, as_type<uint>(v));
}

// ---------------------------------------------------------------------------
// THE BLOCK FOLD, the same operation the CUDA prologue and the host seam carry,
// with the total left on lane 0 and `is_block_writer` naming that lane.
//
// THE SHUFFLE FORM IS THE FULL-SIMDGROUP ONE, which is the case Metal's
// non-clamping `simd_shuffle_down` handles the same way CUDA's does: a sliding
// window over the whole simdgroup leaves the total on lane 0 for offsets
// 16 down to 1. The asymmetry `sm_seg8_reduce_down` documents is a SEGMENT
// property and does not reach this.
//
// NOT `simd_sum`: its association order is unspecified, so it is not comparable
// against the CUDA arm run to run. The explicit tree fixes the order, and the
// cross-simdgroup step below is a deterministic ascending walk rather than an
// atomic for the same reason.
//
// `scratch` holds one element PER SIMDGROUP, stated by the entry's
// `[[seam::scratch(N)]]`, and is in the threadgroup address space because that
// is what a group-local array is here.
template <class T>
inline T block_sum(T value, threadgroup T *scratch, uint thread_index,
                   uint threads_per_group) {
    for (uint offset = simd_width / 2u; offset > 0u; offset >>= 1) {
        value += shuffle_down(value, offset);
    }
    const uint warps = (threads_per_group + simd_width - 1u) / simd_width;
    if (warps == 1u) {
        return value;
    }
    const uint lane = thread_index & (simd_width - 1u);
    const uint warp = thread_index / simd_width;
    if (lane == 0u) {
        scratch[warp] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T total = scratch[0];
    if (thread_index == 0u) {
        for (uint w = 1u; w < warps; ++w) {
            total += scratch[w];
        }
    }
    return total;
}

inline bool is_block_writer(uint thread_index, uint threads_per_group) {
    (void)threads_per_group;
    return thread_index == 0u;
}
// The bounded fold `seam_host.h` documents: `lanes` consecutive threads share
// one work item and this folds their values.
//
// THE XOR FORM, WHICH IS WHY `sm_seg8_reduce_xor` ABOVE IS PRIMARY. Metal's
// `simd_shuffle_down` does not clamp at a segment boundary the way CUDA's
// width-N `__shfl_down_sync` does, so a down ladder leaves the segment sum on
// lane%N==0 only. The xor butterfly leaves it on EVERY lane of the run and, at
// N = 8, produces `((v0+v4) + (v2+v6)) + ((v1+v5) + (v3+v7))`, which is
// bit-for-bit CUDA's width-8 tree. That exact order is what keeps the solver's
// curvature round-off bound calibrated across the two backends, so this arm
// must not be "simplified" to the down form.
template <class T>
inline T lane_reduce_add(T value, threadgroup T *scratch, uint lane,
                         uint lanes) {
    (void)scratch;
    (void)lane;
    for (uint reach = lanes / 2u; reach > 0u; reach >>= 1) {
        value += simd_shuffle_xor(value, (ushort)reach);
    }
    return value;
}
// Lane 0 of each run, matching CUDA. The xor fold leaves the total on every lane
// here, so this is a convention rather than a constraint on THIS arm; it is a
// name because the host arm's answer is the LAST lane.
inline bool is_lane_writer(uint lane, uint lanes) {
    return (lane % lanes) == 0u;
}

}  // namespace compute
)MSL";

// Synthetic segment names for the generated prologue. They are not files on
// disk, so they are never registered with the diagnostic channel; they exist so
// a compile error inside the prologue names which part of it it came from.
const char *const kSegStandardInclude = "<metal-backend/msl-stdlib>";
const char *const kSegMacroSeam = "<metal-backend/macro-seam>";
const char *const kSegDiagPrologue = "<metal-backend/diagnostics-prologue>";

const char *const kDumpEnvVar = "PPF_METAL_SHADER_DUMP";

// One appended segment, and where its body landed in the assembled text. The
// mapping is what lets a diagnostic reported against the INJECTED file name be
// quoted from the text the compiler actually saw.
struct Segment {
    std::string path;       // exactly the string in the injected #line
    size_t first_line = 0;  // 1-based physical line of the body in the unit
    size_t line_count = 0;
};

size_t count_lines(const std::string &text) {
    size_t n = 0;
    for (char c : text) {
        if (c == '\n') {
            ++n;
        }
    }
    return n;
}

// A path is spliced into a #line directive, so a quote or a newline in it would
// silently corrupt the directive and mis-attribute every following line.
bool path_is_directive_safe(const std::string &path, std::string *why) {
    if (path.empty()) {
        *why = "path is empty";
        return false;
    }
    for (char c : path) {
        if (c == '"' || c == '\\' || c == '\n' || c == '\r') {
            *why = "path contains a character that cannot appear in a #line "
                   "directive (quote, backslash or newline)";
            return false;
        }
    }
    return true;
}

// THE INCLUDE MECHANISM, AND WHY IT LIVES HERE.
//
// A quoted #include is how a shared kernel body names the header it depends on,
// and nvcc and the host compiler both resolve it against the filesystem. The
// Metal driver is handed one concatenated string with no filesystem behind it,
// so the same line reaching the shader compiler is a compile error. That is a
// property of this backend, so this backend answers for it: the assembler
// neutralizes each quoted include as it splices the segment, and the kernel
// body carries no conditional of its own.
//
// The host IS the include mechanism here. A dependency a shader actually needs
// is supplied by placing that file earlier in the segment list, which is what
// the shader_add_segment calls in the backend logic tree do; a
// dependency it does not need (float_math.hpp, common.hpp) simply never becomes
// a segment, and the seam macros stand in for its contents.
//
// The rewrite REPLACES the line rather than deleting it, so the segment keeps
// its line count. Segment attribution is a line-for-line mapping onto the file
// on disk (`#line 1 "<path>"` plus Segment::first_line), so dropping a line
// would shift every diagnostic and every device assert record below it by one.
//
// Angle-bracket includes are left alone. <metal_stdlib> and its siblings are
// the one form the shader compiler can serve, and any other angle include is a
// wiring mistake that check-shared-wiring.py reports by name.
std::string neutralize_quoted_includes(const std::string &body) {
    static const char kMarker[] = "// [metal] host-resolved include: ";
    std::string out;
    out.reserve(body.size() + 64);
    size_t start = 0;
    while (start <= body.size()) {
        size_t nl = body.find('\n', start);
        size_t end = (nl == std::string::npos) ? body.size() : nl;
        const char *p = body.data() + start;
        const char *stop = body.data() + end;
        auto skip_blank = [&]() {
            while (p < stop && (*p == ' ' || *p == '\t')) {
                ++p;
            }
        };
        skip_blank();
        bool quoted_include = false;
        if (p < stop && *p == '#') {
            ++p;
            skip_blank();
            const char *word = "include";
            const size_t word_len = 7;
            if (static_cast<size_t>(stop - p) >= word_len &&
                std::strncmp(p, word, word_len) == 0) {
                p += word_len;
                skip_blank();
                quoted_include = (p < stop && *p == '"');
            }
        }
        if (quoted_include) {
            out += kMarker;
        }
        out.append(body, start, end - start);
        if (nl == std::string::npos) {
            break;
        }
        out += '\n';
        start = nl + 1;
    }
    return out;
}

std::string line_of(const std::string &text, size_t line_no) {
    if (line_no == 0) {
        return std::string();
    }
    size_t start = 0;
    size_t current = 1;
    while (current < line_no) {
        size_t nl = text.find('\n', start);
        if (nl == std::string::npos) {
            return std::string();
        }
        start = nl + 1;
        ++current;
    }
    size_t nl = text.find('\n', start);
    size_t end = (nl == std::string::npos) ? text.size() : nl;
    while (end > start && text[end - 1] == '\r') {
        --end;
    }
    return text.substr(start, end - start);
}

bool read_whole_file(const char *path, std::string *out, std::string *err) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        *err = std::string("cannot open shader source '") + path +
               "': " + std::strerror(errno);
        return false;
    }
    std::string data;
    char buf[65536];
    for (;;) {
        size_t got = std::fread(buf, 1, sizeof(buf), f);
        data.append(buf, got);
        if (got != sizeof(buf)) {
            if (std::ferror(f)) {
                *err = std::string("read error on shader source '") + path + "'";
                std::fclose(f);
                return false;
            }
            break;
        }
    }
    std::fclose(f);
    *out = std::move(data);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// The assembler
// ---------------------------------------------------------------------------

struct ShaderSource {
    Context *ctx = nullptr;
    Diagnostics *diag = nullptr;
    std::string text;
    std::vector<Segment> segments;
    size_t line_count = 0;  // physical lines currently in `text`
    // First latched failure from a void-returning entry point. While set, the
    // assembled unit is known incomplete and shader_compile refuses to compile
    // it, because compiling a unit that is silently missing a segment produces
    // a "kernel not found" much further downstream.
    std::string fatal;
};

namespace {

void latch_fatal(ShaderSource *s, const std::string &msg) {
    std::fprintf(stderr, "[metal] shader assembly failed: %s\n", msg.c_str());
    std::fflush(stderr);
    if (s->fatal.empty()) {
        s->fatal = msg;
    }
}

// Appends one segment with its directives. 'registered' distinguishes a real
// source file, which gets a DIAG_FILE_ID from the diagnostic channel, from a
// synthetic prologue segment, which has no file to resolve back to.
void append_segment(ShaderSource *s, const std::string &path,
                    const std::string &raw_body, bool registered) {
    const std::string body = neutralize_quoted_includes(raw_body);
    std::string why;
    if (!path_is_directive_safe(path, &why)) {
        latch_fatal(s, "segment '" + path + "': " + why);
        return;
    }
    if (body.find('\0') != std::string::npos) {
        latch_fatal(s, "segment '" + path +
                           "' contains a NUL byte, so it is not MSL source");
        return;
    }

    Segment seg;
    seg.path = path;

    if (registered) {
        const unsigned file_id = diag_register_file(s->diag, path.c_str());
        s->text += "#undef  DIAG_FILE_ID\n";
        s->text += "#define DIAG_FILE_ID " + std::to_string(file_id) + "\n";
        s->line_count += 2;
    }

    // Last directive before the body: "#line N" numbers the NEXT line, so any
    // directive placed after it would consume line numbers belonging to the
    // file being named.
    s->text += "#line 1 \"" + path + "\"\n";
    s->line_count += 1;

    seg.first_line = s->line_count + 1;
    s->text += body;
    seg.line_count = count_lines(body);
    // A segment that does not end in a newline would glue its last line to the
    // next segment's directive, and a directive is only a directive when it
    // starts a line.
    if (!body.empty() && body.back() != '\n') {
        s->text += '\n';
        seg.line_count += 1;
    }
    s->line_count += seg.line_count;
    s->segments.push_back(seg);
}

// Maps a diagnostic's (file, line) back to a physical line of the assembled
// text. Returns 0 when the reference cannot be placed, which happens for
// diagnostics reported against Metal's own system headers.
size_t assembled_line_for(const ShaderSource *s, const std::string &file,
                          size_t logical_line) {
    for (const Segment &seg : s->segments) {
        if (seg.path == file && logical_line >= 1 &&
            logical_line <= seg.line_count) {
            return seg.first_line + logical_line - 1;
        }
    }
    // With no #line in effect the compiler numbers the whole source, which it
    // names "program_source". Nothing in the assembled unit is outside a
    // segment today, but a future prologue tweak could be.
    if (file == "program_source" && logical_line >= 1 &&
        logical_line <= s->line_count) {
        return logical_line;
    }
    return 0;
}

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// Collects every "<file>:<line>:" reference in one diagnostic line. This is the
// clang form the Metal front end emits, and it is the only form worth quoting
// against.
//
// The reference is NOT anchored to the start of the line: the context wraps the
// compiler's text in its own prefix, so "MSL compilation failed:
// contact/accd.hpp:4:10: error: ..." is the shape actually seen.
void collect_diagnostic_refs(
    const std::string &line,
    std::vector<std::pair<std::string, size_t>> *refs) {
    size_t i = 0;
    while (i < line.size()) {
        size_t colon = line.find(':', i);
        if (colon == std::string::npos || colon + 1 >= line.size()) {
            return;
        }
        size_t j = colon + 1;
        size_t value = 0;
        while (j < line.size() && is_digit(line[j])) {
            value = value * 10 + static_cast<size_t>(line[j] - '0');
            ++j;
        }
        // A reference is "<something>:<digits>:", the trailing colon being the
        // start of the column or of the severity.
        if (j == colon + 1 || j >= line.size() || line[j] != ':') {
            i = colon + 1;
            continue;
        }
        // Walk back over the file token. A path holds neither whitespace nor a
        // colon in anything this backend concatenates.
        size_t begin = colon;
        while (begin > 0) {
            const char c = line[begin - 1];
            if (c == ' ' || c == '\t' || c == ':' || c == '(' || c == ',') {
                break;
            }
            --begin;
        }
        const std::string file = line.substr(begin, colon - begin);
        bool all_digits = !file.empty();
        for (char c : file) {
            all_digits = all_digits && is_digit(c);
        }
        if (!file.empty() && !all_digits) {
            refs->emplace_back(file, value);
        }
        // Past the trailing colon, so the "<col>:" half of a
        // "<file>:<line>:<col>:" reference cannot rematch as its own reference.
        i = j + 1;
    }
}

// Quotes every source line the diagnostic points at, taken from the ASSEMBLED
// text rather than from disk. The injected file name says which file the
// compiler thinks it read; the assembled text is what it actually read, and a
// disagreement between the two is itself the bug.
std::string quote_referenced_lines(const ShaderSource *s,
                                   const std::string &diagnostic) {
    std::vector<std::pair<std::string, size_t>> refs;
    size_t pos = 0;
    while (pos <= diagnostic.size()) {
        size_t nl = diagnostic.find('\n', pos);
        const std::string line = diagnostic.substr(
            pos, (nl == std::string::npos ? diagnostic.size() : nl) - pos);
        pos = (nl == std::string::npos) ? diagnostic.size() + 1 : nl + 1;
        collect_diagnostic_refs(line, &refs);
    }

    std::vector<std::pair<std::string, size_t>> seen;
    std::string out;
    for (const auto &ref : refs) {
        const std::string &file = ref.first;
        const size_t line_no = ref.second;
        bool dup = false;
        for (const auto &e : seen) {
            if (e.first == file && e.second == line_no) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        seen.push_back(ref);

        size_t physical = assembled_line_for(s, file, line_no);
        char head[512];
        if (physical == 0) {
            std::snprintf(head, sizeof(head),
                          "  %s:%zu -> not a line of the assembled unit",
                          file.c_str(), line_no);
            out += head;
            out += '\n';
            continue;
        }
        std::snprintf(head, sizeof(head), "  %s:%zu (assembled line %zu): ",
                      file.c_str(), line_no, physical);
        out += head;
        out += line_of(s->text, physical);
        out += '\n';
    }
    if (out.empty()) {
        return std::string();
    }
    return "--- offending lines, quoted from the assembled translation unit "
           "---\n" +
           out;
}

}  // namespace

ShaderSource *shader_create(Context *ctx, Diagnostics *diag) {
    if (!ctx || !diag) {
        std::fprintf(stderr,
                     "[metal] shader_create: %s is null; the assembler "
                     "compiles through the context and takes its file ids from "
                     "the diagnostic channel\n",
                     ctx ? "diag" : "ctx");
        std::fflush(stderr);
        return nullptr;
    }
    ShaderSource *s = new ShaderSource();
    s->ctx = ctx;
    s->diag = diag;

    // Prologue first, in the order the shader needs it: the standard library,
    // then the macro seam that every shared header is written against, then the
    // diagnostic channel's device-side declarations, which are themselves
    // written in terms of the seam.
    append_segment(s, kSegStandardInclude, kMslStandardInclude, false);
    append_segment(s, kSegMacroSeam, kMslMacroSeam, false);
    const char *diag_prologue = diag_shader_prologue(diag);
    if (!diag_prologue) {
        latch_fatal(s, "the diagnostic channel returned no shader prologue, so "
                       "no assert or bounds check in this unit would report "
                       "anything");
        diag_prologue = "";
    }
    append_segment(s, kSegDiagPrologue, diag_prologue, false);
    return s;
}

void shader_destroy(ShaderSource *s) {
    delete s;
}

void shader_add_segment(ShaderSource *s, const char *path, const char *text) {
    if (!s) {
        std::fprintf(stderr, "[metal] shader_add_segment: null assembler\n");
        std::fflush(stderr);
        return;
    }
    if (!path || !text) {
        latch_fatal(s, "shader_add_segment called with a null path or text");
        return;
    }
    append_segment(s, path, text, true);
}

bool shader_add_file(ShaderSource *s, const char *path, std::string *err) {
    if (!s) {
        if (err) {
            *err = "shader_add_file: null assembler";
        }
        return false;
    }
    if (!path) {
        latch_fatal(s, "shader_add_file called with a null path");
        if (err) {
            *err = s->fatal;
        }
        return false;
    }
    std::string body;
    std::string read_err;
    if (!read_whole_file(path, &body, &read_err)) {
        latch_fatal(s, read_err);
        if (err) {
            *err = read_err;
        }
        return false;
    }
    const size_t before = s->segments.size();
    append_segment(s, path, body, true);
    if (s->segments.size() == before) {
        if (err) {
            *err = s->fatal.empty() ? std::string("segment was not appended")
                                    : s->fatal;
        }
        return false;
    }
    return true;
}

const std::string &shader_assembled(ShaderSource *s) {
    static const std::string empty;
    if (!s) {
        return empty;
    }
    return s->text;
}

bool shader_dump(ShaderSource *s, const char *path, std::string *err) {
    if (!s || !path) {
        if (err) {
            *err = "shader_dump: null assembler or path";
        }
        return false;
    }
    FILE *f = std::fopen(path, "wb");
    if (!f) {
        if (err) {
            *err = std::string("cannot open '") + path +
                   "' for the shader dump: " + std::strerror(errno);
        }
        return false;
    }
    const size_t n = s->text.size();
    const bool wrote = (n == 0) || (std::fwrite(s->text.data(), 1, n, f) == n);
    const bool closed = (std::fclose(f) == 0);
    if (!wrote || !closed) {
        if (err) {
            *err = std::string("failed to write the shader dump to '") + path +
                   "'";
        }
        return false;
    }
    return true;
}

std::vector<std::string> shader_segment_paths(ShaderSource *s) {
    std::vector<std::string> paths;
    if (!s) {
        return paths;
    }
    paths.reserve(s->segments.size());
    for (const Segment &seg : s->segments) {
        paths.push_back(seg.path);
    }
    return paths;
}

unsigned shader_compile(ShaderSource *s, std::string *err) {
    if (err) {
        err->clear();
    }
    if (!s) {
        if (err) {
            *err = "shader_compile: null assembler";
        }
        return 0;
    }
    if (!s->fatal.empty()) {
        if (err) {
            *err = "the assembled translation unit is incomplete: " + s->fatal;
        }
        return 0;
    }

    // An explicit opt-in, because a concatenated unit has no file on disk to
    // look at when a diagnostic does not explain itself.
    const char *dump_path = std::getenv(kDumpEnvVar);
    if (dump_path && dump_path[0] != '\0') {
        std::string dump_err;
        if (shader_dump(s, dump_path, &dump_err)) {
            std::fprintf(stderr, "[metal] assembled shader source dumped to %s "
                                 "(%zu bytes, %zu lines)\n",
                         dump_path, s->text.size(), s->line_count);
        } else {
            std::fprintf(stderr, "[metal] %s is set but the dump failed: %s\n",
                         kDumpEnvVar, dump_err.c_str());
        }
        std::fflush(stderr);
    }

    // Compilation runs through the context, which owns the single
    // mathMode = Safe options helper. This module never builds compile options
    // of its own.
    std::string diagnostic;
    const unsigned library =
        context_compile_library(s->ctx, s->text.c_str(), &diagnostic);
    if (library == 0) {
        if (err) {
            *err = diagnostic;
            if (!err->empty() && err->back() != '\n') {
                *err += '\n';
            }
            *err += quote_referenced_lines(s, diagnostic);
            if (!dump_path || dump_path[0] == '\0') {
                *err += std::string("(set ") + kDumpEnvVar +
                        "=<path> to dump the assembled translation unit)\n";
            }
        }
        return 0;
    }
    return library;
}

}  // namespace metal_backend
