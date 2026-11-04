// File: seam_host.h
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// The backend seam as a plain host C++ compiler spells it. Reached only through
// seam.hpp, which is the one file that decides between this prologue and the
// nvcc one; see the contract and the list of deliberate absences there.
//
// The host target is not hypothetical. `build.rs` hands five translation units
// to a host C++ compiler: `entrypoints/entries.cpp`, which names every
// generated entry point (`check_entry_coverage` fails the build when a
// declaration is missing from its include list) and so compiles the neutral
// bodies those entries call, and the hand-written launchers in
// `kernel_shim.cpp`, `shim_step.cpp`, `shim_contact.cpp` and
// `shim_override_seed.cpp`. The regression gates under `tests/kernels/`,
// compiled and run by `cargo test`, and the two host unit tests under
// `energy/model/tests` include the neutral bodies and shared headers they
// assert against. Those compilations are what makes a shared body's oracle the
// same bytes as the body.
//
// Every definition below is a flat `#define`. There is no conditional in this
// file: the target is already decided by the time it is read.

// ---------------------------------------------------------------------------
// Inline annotations.
//
// A host compiler has one execution space, so all three names are plain
// `inline`. The distinction between them exists for nvcc, where widening a
// device-only body to both spaces changes what is compiled for the GPU; see
// seam_cuda.cuh.
// ---------------------------------------------------------------------------

#define SM_INLINE_DEVICE inline
#define SM_INLINE_DEVICE_HOST inline

// ---------------------------------------------------------------------------
// Address spaces. MSL requires one on every pointer and reference; CUDA and the
// host have a single address space and expand all three to nothing.
// ---------------------------------------------------------------------------

#define SM_THREAD
#define SM_DEVICE
#define SM_THREADGROUP

// ---------------------------------------------------------------------------
// Math.
//
// These are the C float forms, not the `std::` overload set, and that is what
// makes the host arithmetic the same arithmetic the device runs: an unsuffixed
// name would promote a float argument to double and give the oracle more
// accuracy than the body under test has.
// ---------------------------------------------------------------------------

#define SM_ABS fabsf
#define SM_MAX fmaxf
#define SM_MIN fminf
#define SM_SQRT sqrtf
#define SM_DIV(a, b) ((a) / (b))
#define SM_FMA fmaf
#define SM_ACOS acosf
#define SM_ATAN2(y, x) atan2f((y), (x))
#define SM_ISNAN isnan
#define SM_ISINF isinf
#define SM_NEXTAFTER(a, b) nextafterf((a), (b))
#define SM_NUMERIC_MAX(T) std::numeric_limits<T>::max()
#define SM_INFINITY (std::numeric_limits<float>::infinity())

// Trigonometry, exponential and logarithm go through float_math.hpp on this
// target too, and for the same reason the device does: the oracle has to run
// the arithmetic the kernel runs, so it takes the same reduction and the same
// bounded and periodic split rather than the library's accurate paths. The
// `fmath` host arms are `std::cos`, `std::sin`, `std::log` and `std::exp`.
#define SM_COS fmath::cos_bounded
#define SM_COS_BOUNDED fmath::cos_bounded
#define SM_SIN_BOUNDED fmath::sin_bounded
#define SM_SIN_PERIODIC(x) fmath::sin_periodic(x)
#define SM_LOG fmath::log
#define SM_EXP fmath::exp

// ---------------------------------------------------------------------------
// Bit intrinsics.
//
// SM_CLZ goes through a helper defined below the includes, because
// backend/contract.md answers 32 for an input of 0 and neither host builtin
// does: `__builtin_clz(0)` is undefined and `_BitScanReverse` leaves its index
// unset. CUDA's `__clz` and MSL's `clz` both give 32.
// ---------------------------------------------------------------------------

#define SM_CLZ(value) seam_host_detail::clz_or_32(value)

// ---------------------------------------------------------------------------
// Atomics.
//
// A host translation unit runs one thread through a shared body, so the
// read-modify-write is a compound assignment and the load and the store are
// plain dereferences. The type names here are the plain scalars the buffers
// hold. The `compute::` table further down types the address as a SLOT instead,
// and that is the table a neutral kernel body reaches, since a body carries no
// preprocessor and cannot name a macro at all.
// ---------------------------------------------------------------------------

#define SM_ATOMIC_UINT unsigned
#define SM_ATOMIC_FLOAT float

#define SM_ATOMIC_LOAD_UINT(pointer) (*(pointer))
#define SM_ATOMIC_STORE_UINT(pointer, value) (*(pointer) = (value))
#define SM_ATOMIC_ADD_UINT(pointer, value) (*(pointer) += (value))
#define SM_ATOMIC_ADD_FLOAT(pointer, value) (*(pointer) += (value))

// ===========================================================================
// The `ppf` namespace: the same table as ordinary C++ names.
//
// A neutral kernel source (`*.kernel.cpp`, rendered by seam/kernelgen.py) has
// no preprocessor at all, so it cannot name a macro. It calls these instead,
// and each backend prologue defines them with that backend's spelling. The
// replacement text is the same as the macro directly above it, name for name.
//
// These are the C float forms, not the `std::` overload set, and that is what
// makes the host arithmetic the same arithmetic the device runs: an unsuffixed
// name would promote a float argument to double and give the oracle more
// accuracy than the body under test has. Each takes `float` rather than a
// template parameter, so a widened argument is narrowed at the call rather than
// silently computed in double.
//
// SEVEN NAMES ARE DELIBERATELY MISSING here, as they are from the macro table
// below: bits::popcount, compute::simd_ballot, compute::ballot_t,
// compute::threadgroup_barrier, compute::shuffle_down, compute::shuffle_up and
// compute::simd_width. Each names a warp or threadgroup operation, the lane
// count such an operation is written against, or the TYPE such an operation
// returns, and none of them has a single-thread meaning. Inventing bodies for
// them would decide what a lane shuffle means with one lane, here rather than
// where the kernel is, and would replace a compile error with an oracle that
// computes something else.
// ===========================================================================

// ---------------------------------------------------------------------------
// The device QUEUE a host launcher orders its work on.
//
// DELIBERATELY NOT A SEAM NAME; the reasoning sits in seam_cuda.cuh beside
// nvcc's spelling of it. A host compiler has no queue to order work on, so the
// handle is an opaque pointer and every caller leaves it at its default. The
// name exists here so a launcher declaration that names a queue can still be
// read by a host compiler, which is what makes those declarations parse
// outside a CUDA translation unit.
// ---------------------------------------------------------------------------

using DeviceQueue = void *;

#include "../float_math.hpp"

#include <cstring>
#include <limits>
#ifdef _MSC_VER
// The bit and atomic helpers below are GCC/Clang builtins on the Unix hosts and
// MSVC intrinsics on Windows (_BitScanReverse, _Interlocked*); both are declared
// here. There is no Unix counterpart to include, so this stays guarded.
#include <intrin.h>
#endif

// SM_CLZ's body. It is not a seam name: the macro table is what a shared header
// reaches, a neutral body reaches `bits::clz` instead, and so this sits outside
// the namespaces seam names are read from.
//
// NOT `__lzcnt`. That MSVC intrinsic exists only for x86 and x64, so it does not
// compile for ARM64, and on an x86 CPU without LZCNT the instruction decodes as
// BSR and returns the index of the highest set bit rather than the count.
namespace seam_host_detail {

inline int clz_or_32(unsigned value) {
    if (value == 0u) {
        return 32;
    }
#ifdef _MSC_VER
    unsigned long index;
    _BitScanReverse(&index, value);
    return 31 - static_cast<int>(index);
#else
    return __builtin_clz(value);
#endif
}

}  // namespace seam_host_detail

namespace fmath {

// --- Arithmetic ------------------------------------------------------------

inline float abs(float x) {
    return ::fabsf(x);
}
inline float max(float a, float b) {
    return ::fmaxf(a, b);
}
inline float min(float a, float b) {
    return ::fminf(a, b);
}
inline float sqrt(float x) {
    return ::sqrtf(x);
}
inline float div(float a, float b) {
    return a / b;
}
inline float fma(float a, float b, float c) {
    return ::fmaf(a, b, c);
}
inline float acos(float x) {
    return ::acosf(x);
}
inline float atan2(float y, float x) {
    return ::atan2f(y, x);
}
// A real exponent.
inline float pow(float x, float y) {
    return ::powf(x, y);
}
inline bool isnan(float x) {
    return std::isnan(x);
}
inline bool isinf(float x) {
    return std::isinf(x);
}
inline float nextafter(float a, float b) {
    return ::nextafterf(a, b);
}

// cos_bounded, sin_bounded, sin_periodic, log and exp are NOT here, and their
// absence is the point: float_math.hpp, included above, declares them in this
// same namespace on this target too. The oracle has to run the arithmetic the
// kernel runs, so it takes the same reduction and the same bounded and periodic
// split rather than the library's accurate paths, and taking them from the same
// header the device build takes them from is how that is guaranteed rather than
// asserted.

inline float infinity() {
    return std::numeric_limits<float>::infinity();
}
inline float float_max() {
    return std::numeric_limits<float>::max();
}

}  // namespace fmath

// --- Bit intrinsics --------------------------------------------------------
//
// `bits::` rather than `compute::`: these are integer, so `fmath::` is the
// wrong side of the float line, and they compute the same value on every
// target, so `compute::` would overstate what varies.

namespace bits {

inline int clz(unsigned value) {
#ifdef _MSC_VER
    // _BitScanReverse gives the index of the highest set bit; the leading-zero
    // count is 31 minus that. Like __builtin_clz(0), value == 0 is undefined,
    // and no caller passes it.
    unsigned long index;
    _BitScanReverse(&index, value);
    return 31 - static_cast<int>(index);
#else
    return __builtin_clz(value);
#endif
}

}  // namespace bits

// --- Atomics ---------------------------------------------------------------
//
// `compute::` for the same reason the lane names above are: what an atomic
// read-modify-write costs and how the address is typed is the backend's, and
// a target running one thread through a body reaches the same answer with a
// plain read, add and write back.

namespace compute {

//
// AN ACCUMULATOR SLOT IS ITS OWN TYPE, AND THE OPERATIONS BELOW ARE THE ONLY
// WAY TO REACH ONE.
//
// A slot holds the same 4 bytes an ordinary `float` or `unsigned` buffer holds,
// and the same bytes are reached plainly elsewhere: one kernel clears a buffer
// with an ordinary store, a second accumulates into it, and the host reads it
// back. Atomicity is a property of how a PARTICULAR kernel reaches the storage,
// so it is carried by the parameter type and not by the buffer. One entry
// declares `compute::atomic_float_t *` over an allocation and another declares
// `float *` over the same one, and both are correct.
//
// Three properties hold, and a reader can check each of them in this file. The
// storage member is private and the operations below are its only friends, so a
// plain read, a plain store, a compound assignment, a slot-to-slot copy and a
// conversion to `float *` are each a compile error. Only the default
// constructor is declared and it is defaulted, so the type is trivially default
// constructible and standard layout: an array of slots is valid, and the raw
// bytes an arena hands out are valid storage for one. And each class is 4 bytes
// at 4-byte alignment, asserted below, which is what the `[[seam::pod(4)]]` on
// every entry field naming one declares.
//
// A CAST STILL REACHES THE STORAGE, AND NOTHING REFUSES ONE.
// `*(float *) pointer += value` compiles here. What the type gives is that
// every ordinary spelling of a non-atomic touch is refused, so reaching the
// storage plainly costs a cast, which is visible at the call site and greppable
// across the tree.
//
// A host translation unit runs one thread through a shared body, so the
// read-modify-write is a plain read, add and write back, and the load and the
// store are a plain read and a plain write of the member.
//
// Every operation returns the PRE-IMAGE, which is the shared contract, spelled
// out in the Metal prologue and native to CUDA's atomicAdd. The macro form
// above is a compound assignment, whose value is the POST-image; no call site
// in this tree reads it, so the two agree everywhere they are used, and the
// function is the form that states the contract.

class AtomicFloatSlot;
class AtomicUintSlot;

// SPECULATIVE EXECUTION, the host backend's way to run a `Scatter::Atomic` or
// `Scatter::Claim` row on every core and still reach the serial pass's state.
//
// A row whose declaration opts in (`KernelDecl::speculate`) runs in two
// phases. Phase one runs every chunk at once with this flag set on the running
// thread, so the FIRST operation below that an element reaches throws
// `SeamDefer` instead of touching the slot: that element is set aside, with
// its diagnostic record put back as it was, and the chunk goes on with the next
// one. Phase two runs the set-aside elements one at a time, ascending, with the
// flag clear. An element that reaches no slot operation, which is most of a
// contact traversal and nearly all of an intersection scan, is finished in
// phase one; the slots then receive every operation in the order the serial
// pass gives them, because only phase two reaches a slot and it runs in that
// order. `ppf-cts-compute/cpu/host.rs` states what a body must satisfy for the
// two schedules to agree.
//
// A THREAD-LOCAL AND NOT A GLOBAL, because two devices may run in one process
// (the test harness runs tests on threads that share one pool), and a flag one
// of them raised must not reach the other's serial pass.
struct SeamDefer {};
inline thread_local bool seam_speculating = false;
inline void seam_defer_if_speculating() {
    if (seam_speculating) {
        throw SeamDefer{};
    }
}

// Declared ahead of the classes so that each friend declaration inside them
// names a function this namespace already has rather than introducing one.
inline float atomic_add(AtomicFloatSlot *pointer, float value);
inline unsigned atomic_add(AtomicUintSlot *pointer, unsigned value);
inline unsigned atomic_load(AtomicUintSlot *pointer);
inline void atomic_store(AtomicUintSlot *pointer, unsigned value);
inline void atomic_max(AtomicUintSlot *pointer, float value);

class AtomicFloatSlot {
  public:
    AtomicFloatSlot() = default;
    AtomicFloatSlot(const AtomicFloatSlot &) = delete;
    AtomicFloatSlot &operator=(const AtomicFloatSlot &) = delete;

  private:
    float storage;

    friend inline float atomic_add(AtomicFloatSlot *pointer, float value);
};

class AtomicUintSlot {
  public:
    AtomicUintSlot() = default;
    AtomicUintSlot(const AtomicUintSlot &) = delete;
    AtomicUintSlot &operator=(const AtomicUintSlot &) = delete;

  private:
    unsigned storage;

    friend inline unsigned atomic_add(AtomicUintSlot *pointer, unsigned value);
    friend inline unsigned atomic_load(AtomicUintSlot *pointer);
    friend inline void atomic_store(AtomicUintSlot *pointer, unsigned value);
    friend inline void atomic_max(AtomicUintSlot *pointer, float value);
};

// The seam names, which are what a neutral kernel body and an entry argument
// record spell. The class names above are deliberately not the seam names: a
// backend defines the seam name as whatever that backend's slot is, which on
// this target is the class and on Metal is `atomic_float`.
using atomic_uint_t = AtomicUintSlot;
using atomic_float_t = AtomicFloatSlot;

static_assert(sizeof(atomic_float_t) == 4 && alignof(atomic_float_t) == 4,
              "a slot must lay out as the 4 bytes its buffer holds");
static_assert(sizeof(atomic_uint_t) == 4 && alignof(atomic_uint_t) == 4,
              "a slot must lay out as the 4 bytes its buffer holds");

// THE UNSIGNED OPERATIONS ARE GENUINELY ATOMIC, and this is the one place in
// the three prologues where that had to be arranged rather than inherited.
// CUDA's `atomicAdd` and MSL's `atomic_fetch_add_explicit` are atomic against
// a fully concurrent grid; a plain `storage += value` here is not, and the
// host backend runs a `Scatter::Disjoint` row as concurrent rayon chunks
// (`ppf-cts-compute/cpu/host.rs`). So a body that reaches this from such a row
// would be a data race on THIS BACKEND ALONE, invisible wherever CUDA is the
// oracle. RELAXED is the right order and not a weakening: what these carry is
// a count or a claim, so only the operation's own indivisibility is needed,
// and no other memory is published through them.
//
// `check-atomic-scatter.py` is the gate that keeps the two halves in step: it
// refuses a Disjoint row reaching a FLOAT atomic, which is a plain `+=` below
// for the reason stated there.
// The MSVC arm spells each of these with an _Interlocked* intrinsic over the
// storage reinterpreted as `long` (same 4 bytes, asserted above). Those carry
// full-fence order, which is a safe superset of the RELAXED the Unix builtins
// take: these operations publish no other memory, so a stronger order changes
// only cost, never correctness.
inline unsigned atomic_load(atomic_uint_t *pointer) {
    seam_defer_if_speculating();
#ifdef _MSC_VER
    return static_cast<unsigned>(
        _InterlockedOr(reinterpret_cast<volatile long *>(&pointer->storage), 0));
#else
    return __atomic_load_n(&pointer->storage, __ATOMIC_RELAXED);
#endif
}
inline void atomic_store(atomic_uint_t *pointer, unsigned value) {
    seam_defer_if_speculating();
#ifdef _MSC_VER
    _InterlockedExchange(reinterpret_cast<volatile long *>(&pointer->storage),
                         static_cast<long>(value));
#else
    __atomic_store_n(&pointer->storage, value, __ATOMIC_RELAXED);
#endif
}
inline unsigned atomic_add(atomic_uint_t *pointer, unsigned value) {
    seam_defer_if_speculating();
#ifdef _MSC_VER
    return static_cast<unsigned>(_InterlockedExchangeAdd(
        reinterpret_cast<volatile long *>(&pointer->storage),
        static_cast<long>(value)));
#else
    return __atomic_fetch_add(&pointer->storage, value, __ATOMIC_RELAXED);
#endif
}
// THE FLOAT ADD IS DELIBERATELY NOT ATOMIC, because making it so would buy
// nothing and cost the property that matters. A float fold is only reproducible
// if its ORDER is fixed, and an atomic fixes indivisibility rather than order,
// so a concurrent float accumulation would be correct and non-deterministic:
// two runs of one scene would differ. Every caller therefore sits on a
// `Scatter::Atomic` or `Scatter::Claim` row, which the host backend runs as
// ONE serial ascending pass, and that pass is what makes both the order and
// the access safe. The gate above refuses the combination that would break it.
inline float atomic_add(atomic_float_t *pointer, float value) {
    seam_defer_if_speculating();
    const float previous = pointer->storage;
    pointer->storage += value;
    return previous;
}

// A LARGEST-WINS ACCUMULATOR OVER UINT-TYPED STORAGE, FOR NON-NEGATIVE VALUES.
//
// TWO THINGS ABOUT THE SIGNATURE ARE FORCED RATHER THAN CHOSEN, and both come
// from MSL. The storage is `atomic_uint_t` and not `atomic_float_t` because
// Metal has no float atomic max at all; what it has is an integer one, and the
// IEEE-754 bit pattern of a NON-NEGATIVE float orders the same way as the float
// itself, so the integer maximum of the bits IS the maximum of the values.
// A negative argument compares inverted and would silently return the wrong
// extremum, which is why the precondition is stated rather than assumed:
// callers pass a magnitude.
//
// AND IT RETURNS NOTHING, unlike every other operation here, so that no caller
// needs a bitcast. A neutral kernel body has no spelling for one; the pattern
// this serves accumulates on the device and converts the bits back to float on
// the HOST, after the readback, where a bitcast is ordinary code.
inline void atomic_max(atomic_uint_t *pointer, float value) {
    seam_defer_if_speculating();
    unsigned bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // A compare-and-swap loop, because there is no atomic maximum builtin and
    // the read-then-write it would otherwise be loses an update whenever two
    // threads raise the same slot. `expected` is reloaded on every failed
    // exchange, so the loop re-tests against what the winner actually left. The
    // order is RELAXED on the Unix builtins and full-fence on the MSVC
    // intrinsic, safe for the reason the load/store/add above give.
#ifdef _MSC_VER
    volatile long *slot = reinterpret_cast<volatile long *>(&pointer->storage);
    unsigned expected = static_cast<unsigned>(_InterlockedOr(slot, 0));
    while (bits > expected) {
        long seen = _InterlockedCompareExchange(slot, static_cast<long>(bits),
                                                static_cast<long>(expected));
        if (static_cast<unsigned>(seen) == expected) {
            return;
        }
        expected = static_cast<unsigned>(seen);
    }
#else
    unsigned expected = __atomic_load_n(&pointer->storage, __ATOMIC_RELAXED);
    while (bits > expected) {
        if (__atomic_compare_exchange_n(&pointer->storage, &expected, bits,
                                        true, __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED)) {
            return;
        }
    }
#endif
}


// ---------------------------------------------------------------------------
// THE BLOCK FOLD, WHICH IS NOT A BARRIER AND IS THE REASON IT HAS A HOST
// MEANING AT ALL.
//
// The six names above this file refuses are refused because each promises
// something serial lanes cannot deliver. `SM_THREADGROUP_BARRIER` promises that
// no lane proceeds until every lane arrives, and a rendering that runs lane 0 to
// completion before lane 1 starts breaks that promise however it is spelled.
//
// A BLOCK SUM PROMISES SOMETHING ELSE: that every lane's value is folded in.
// The generated group shim runs a group's lanes one after another, lane 0
// through lane width - 1, over a group-local scratch that persists across them,
// so a lane which adds into that scratch has ALREADY SEEN every earlier lane's
// contribution. The promise is kept exactly, and by construction rather than by
// assumption.
//
// SO THE TOTAL IS COMPLETE ON THE LAST LANE HERE AND ON LANE 0 ON A GPU, which
// is why the writer is a name of its own rather than a literal. A body says
// `if (compute::is_block_writer(...))` and both renderings are right; a body
// that hardcoded lane 0 would silently write a partial sum on this backend.
//
// `scratch` holds one element PER WARP on the device arms, which the entry
// states with `[[seam::scratch(N)]]`; this arm uses only the first. The first
// lane STORES rather than adds, so the caller owes no clear, and a group that
// runs twice cannot carry the first run's total into the second.
//
// THE ASSOCIATION DIFFERS FROM THE DEVICE ARMS and is meant to: this one is
// strictly ascending lane order, and a GPU folds within a warp first and then
// walks the warp partials. Both are deterministic and neither is the other's
// bits, which is the same standing every atomic-scatter kernel in this tree
// already has, and `disp_at_5` is how the backends are compared.
template <class T>
inline T block_sum(T value, T *scratch, unsigned thread_index,
                   unsigned threads_per_group) {
    (void)threads_per_group;
    scratch[0] = thread_index == 0u ? value : scratch[0] + value;
    return scratch[0];
}

// WHICH LANE HOLDS THE COMPLETE TOTAL. The LAST one here, because the fold above
// finishes when the last lane has added its value; lane 0 on a GPU, where the
// shuffle tree leaves the total there.
inline bool is_block_writer(unsigned thread_index, unsigned threads_per_group) {
    return thread_index + 1u == threads_per_group;
}
// A BOUNDED FOLD OVER THE `lanes` THREADS THAT SHARE ONE WORK ITEM, which is
// `block_sum` one level down: that folds a whole group, this folds each run of
// `lanes` consecutive threads within it. A kernel that gives eight threads to
// one matrix row wants this and not the group fold.
//
// IT IS NOT `compute::shuffle_down`, WHICH THIS FILE DELIBERATELY OMITS. A lane
// shuffle has no single-thread meaning, so inventing one would decide here what
// a shuffle means with one lane. A bounded FOLD does have one: it promises that
// every lane's value is added in, not that lanes rendezvous, and this arm keeps
// that promise by accumulating in ascending lane order exactly as `block_sum`
// does. That is why the fold is a seam name and the shuffle is not.
//
// ONE SLOT PER FOLD, NOT ONE PER LANE GROUP. The host shim runs a group's lanes
// ascending and consecutively, so lane run `g` finishes before run `g + 1`
// starts and they cannot interleave through the same slot. A caller folding
// several values passes a different slot for each, because those DO interleave:
// one lane folds all of them before the next lane runs.
template <class T>
inline T lane_reduce_add(T value, T *scratch, unsigned lane, unsigned lanes) {
    const unsigned within = lane % lanes;
    scratch[0] = within == 0u ? value : scratch[0] + value;
    return scratch[0];
}
// WHICH LANE OF THE RUN HOLDS THE COMPLETE FOLD. The LAST one here, because the
// fold above finishes when the run's last lane has added its value; lane 0 on
// both device arms, where the tree leaves the total there. A body that hardcoded
// either would be wrong on the other backend, which is the same reason
// `is_block_writer` exists rather than a literal.
inline bool is_lane_writer(unsigned lane, unsigned lanes) {
    return (lane % lanes) + 1u == lanes;
}

// ---------------------------------------------------------------------------
// FOUR FLOAT LANES FOR A SERIAL TWIN, holding one 3-vector per register.
//
// A host twin that walks 3x3 blocks evaluates each block product component by
// component. These let it evaluate the three components at once, component
// `a` in lane `a`, the fourth lane carried and ignored. Every operation is
// LANEWISE, so lane `a` performs exactly the scalar operation the scalar code
// performs on component `a`, on the same operands in the same order: no sum
// is reassociated and the result is the scalar result bit for bit. The
// compiler's freedom to fuse a product into a following add
// (`-ffp-contract=fast`) is the same freedom the scalar form has, on the same
// expression tree.
//
// HOST ONLY, and a twin is the only place that may spell it: the device arms
// already run a component per thread. GCC and Clang get their vector
// extension; any other compiler gets a struct of four floats with the same
// lanewise operators, which computes the same values.
#if defined(__GNUC__) || defined(__clang__)
typedef float lane4 __attribute__((vector_size(16)));
typedef unsigned lane4_bits __attribute__((vector_size(16)));
inline lane4 lane4_splat(float s) {
    lane4 v = {s, s, s, s};
    return v;
}
// `fabsf` is the sign-bit clear, so clearing it in every lane is `fabsf`
// lanewise, NaN payloads included.
inline lane4 lane4_abs(lane4 v) {
    return (lane4)((lane4_bits)v & 0x7fffffffu);
}
inline lane4 lane4_load(const float *p) {
    lane4 v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
#define SEAM_LANE4_SHUFFLE(a, b, i0, i1, i2, i3) \
    __builtin_shufflevector((a), (b), i0, i1, i2, i3)
#else
struct lane4 {
    float v[4];
    float &operator[](unsigned i) { return v[i]; }
    float operator[](unsigned i) const { return v[i]; }
};
inline lane4 operator+(const lane4 &a, const lane4 &b) {
    lane4 r;
    for (unsigned i = 0; i < 4; ++i) r.v[i] = a.v[i] + b.v[i];
    return r;
}
inline lane4 operator*(const lane4 &a, const lane4 &b) {
    lane4 r;
    for (unsigned i = 0; i < 4; ++i) r.v[i] = a.v[i] * b.v[i];
    return r;
}
inline lane4 lane4_splat(float s) {
    lane4 v = {{s, s, s, s}};
    return v;
}
inline lane4 lane4_abs(lane4 v) {
    for (unsigned i = 0; i < 4; ++i) v.v[i] = fabsf(v.v[i]);
    return v;
}
inline lane4 lane4_load(const float *p) {
    lane4 v;
    std::memcpy(v.v, p, sizeof v.v);
    return v;
}
inline lane4 seam_lane4_shuffle(const lane4 &a, const lane4 &b, unsigned i0,
                                unsigned i1, unsigned i2, unsigned i3) {
    const unsigned idx[4] = {i0, i1, i2, i3};
    lane4 r;
    for (unsigned i = 0; i < 4; ++i) {
        r.v[i] = idx[i] < 4 ? a.v[idx[i]] : b.v[idx[i] - 4];
    }
    return r;
}
#define SEAM_LANE4_SHUFFLE(a, b, i0, i1, i2, i3) \
    seam_lane4_shuffle((a), (b), i0, i1, i2, i3)
#endif

// A 3-vector into lanes 0..2, reading exactly three floats.
inline lane4 lane4_load3(const float *p) {
    lane4 v = lane4_splat(0.0f);
    v[0] = p[0];
    v[1] = p[1];
    v[2] = p[2];
    return v;
}

// THE COLUMNS OF A COLUMN-MAJOR 3x3 BLOCK, `(M x)_a = sum_b m[3b + a] x_b`:
// lane `a` of `c0`, `c1`, `c2` is `m[a]`, `m[3 + a]`, `m[6 + a]`. The loads sit at
// offsets 0, 3 and 5, so each reads four floats INSIDE the block's nine and the
// last block of an array is never read past its end; the third is shuffled
// down a lane.
//
inline void lane4_block_columns(const float *m, lane4 &c0, lane4 &c1,
                                lane4 &c2) {
    c0 = lane4_load(m);
    c1 = lane4_load(m + 3);
    const lane4 t = lane4_load(m + 5);
    c2 = SEAM_LANE4_SHUFFLE(t, t, 1, 2, 3, 3);
}

// THE COLUMNS OF ITS TRANSPOSE, `(M^T x)_a = sum_b m[3a + b] x_b`: lane `a` is
// `m[3a]`, `m[3a + 1]`, `m[3a + 2]`, gathered by shuffles from loads at 0, 4
// and 5, again inside the nine floats.
//
inline void lane4_block_columns_transposed(const float *m, lane4 &c0,
                                           lane4 &c1, lane4 &c2) {
    const lane4 a = lane4_load(m);
    const lane4 b = lane4_load(m + 4);
    const lane4 c = lane4_load(m + 5);
    c0 = SEAM_LANE4_SHUFFLE(a, b, 0, 3, 6, 6);
    c1 = SEAM_LANE4_SHUFFLE(a, b, 1, 4, 7, 7);
    c2 = SEAM_LANE4_SHUFFLE(a, c, 2, 4, 7, 7);
}

// `c0 x0 + c1 x1 + c2 x2`, the block product in the scalar form's order.
//
inline lane4 lane4_block_product(const lane4 &c0, const lane4 &c1,
                                 const lane4 &c2, const float *x) {
    return c0 * lane4_splat(x[0]) + c1 * lane4_splat(x[1]) +
           c2 * lane4_splat(x[2]);
}

// A ROW'S CURVATURE TERMS from its finished sum and entry magnitude:
// `signed_sum += x_k sum_k` and `magnitude += |x_k| abssum_k`, k ascending,
// the sums `spmv_row_bound` forms in its order.
//
// TWO SPELLINGS OF THE SAME SUMS, chosen so the host build's output stays the
// bits it had with the scalar twin, contraction on. The expression tree is
// the same in both; what differs is only which product each compiler fuses.
// GCC fused the scalar twin's products into the adds and fuses this loop the
// same way. Clang kept them as separate products (it vectorized them first)
// and fuses this loop when its operands come from lanes, so under Clang the
// products are one lane multiply, which has no add to fuse into, and the adds
// follow in order. Measured with both forms on random and drape-shaped rows:
// bit-identical to the scalar twin under each compiler.
//
inline void lane4_row_bound(const float *x, const lane4 &sum,
                            const lane4 &abssum, float &signed_sum,
                            float &magnitude) {
#if defined(__clang__)
    const lane4 own = lane4_load3(x);
    const lane4 signed_terms = own * sum;
    const lane4 magnitude_terms = lane4_abs(own) * abssum;
    // Two separate chains, written out: interleaved in one loop, Clang pairs
    // the two sums into one two-lane operation and fuses the products back in.
    signed_sum = ((signed_sum + signed_terms[0]) + signed_terms[1]) +
                 signed_terms[2];
    magnitude = ((magnitude + magnitude_terms[0]) + magnitude_terms[1]) +
                magnitude_terms[2];
#else
    for (unsigned k = 0; k < 3; ++k) {
        signed_sum += x[k] * sum[k];
        magnitude += fabsf(x[k]) * abssum[k];
    }
#endif
}

// `|c0| |x0| + |c1| |x1| + |c2| |x2|`, the block's entry magnitude term.
//
inline lane4 lane4_block_magnitude(const lane4 &c0, const lane4 &c1,
                                   const lane4 &c2, const float *x) {
    return lane4_abs(c0) * lane4_abs(lane4_splat(x[0])) +
           lane4_abs(c1) * lane4_abs(lane4_splat(x[1])) +
           lane4_abs(c2) * lane4_abs(lane4_splat(x[2]));
}

}  // namespace compute

// ---------------------------------------------------------------------------
// The DIAGNOSTIC CHANNEL, as a plain host C++ compiler spells it.
//
// A body that asserts an invariant has to report it somewhere, and the three
// backends disagree about what "somewhere" is: nvcc folds through device
// atomics into a claimed slot, MSL writes a header and a ring in device memory,
// and a host thread writes a plain struct. `DIAG_ASSERT4` is therefore a MACRO
// on every target rather than a function, because it records `__FILE__` and
// `__LINE__` of the failing check and no function can read those for its
// caller.
//
// `DiagHandle` IS THE ONE NAME A NEUTRAL BODY MAY SPELL. It is whatever
// `DIAG_BIND` yields on this target, which is a pointer here, a
// `diagnostics::Device` under nvcc and a `Diag` under MSL. A body takes
// it BY VALUE and passes it to the macros; it never dereferences it itself, so
// the three spellings need agree on nothing but the name.
//
// THIS LIVES IN THE SEAM RATHER THAN BESIDE A SHIM, and that is the whole point
// of the file: a neutral kernel body reaches the seam and reaches nothing under
// `entrypoints/`. It used to live in `entrypoints/shim_diag.h`, where only a
// hand-written launcher could see it, which is why a body with an invariant to
// report could not be converted at all.
//
// THE RECORD IS AN ABI. Rust mirrors it as `ppf_cts_compute::DiagRecord` and
// reads the payload, the file and the line out of it, so the field order and
// the widths below are not free. `the_diag_record_matches_cpp` compares this
// size against the compiler's own answer.
//
// The macro RECORDS AND CONTINUES, matching the other two channels. Stopping
// early would change which elements a pass visits, and the results of a pass
// that tripped an invariant are discarded by the caller anyway.
// ---------------------------------------------------------------------------

extern "C" {

struct ChunkDiag {
    // How many checks failed anywhere in this chunk. A count above one says the
    // recorded payload is the first of several, not the only one.
    unsigned fail_count;
    // 1 once `payload`, `file` and `line` describe a real failure.
    unsigned claimed;
    unsigned line;
    float payload[4];
    // The failing check's source file, borrowed from the string literal
    // `__FILE__` expands to, which has static storage duration.
    const char *file;
    // SPECULATION, read only by `seam_speculate_range` below and only when
    // `spec_active` is set (`compute::SeamDefer` states the scheme). The host
    // hands each chunk room for one entry per element in `spec_deferred`; the
    // range appends the elements it set aside, ascending, and names in
    // `spec_claim` the element whose check claimed this record, so the host can
    // merge the two phases in the serial pass's order.
    unsigned *spec_deferred;
    unsigned spec_active;
    unsigned spec_count;
    unsigned spec_claim;
};

}  // extern "C"

using DiagHandle = ChunkDiag *;

// Whether the host asked this call to speculate (`compute::SeamDefer`).
inline bool seam_speculation_requested(const ChunkDiag *diag) {
    return diag != nullptr && diag->spec_active != 0u;
}

// PHASE ONE OF A SPECULATIVE DISPATCH, over one chunk's elements in ascending
// order. Each element runs with the thread's flag raised; one that reaches a
// slot operation throws before touching the slot, and is set aside with the
// record restored to what it held before the element began, so its partial
// run leaves no count and no claim behind. Everything it did before the throw
// was a plain write to memory only that element touches, which the
// declaration's opt-in promises; phase two runs it again from the start.
template <class Call>
inline void seam_speculate_range(ChunkDiag *diag, unsigned begin, unsigned end,
                                 const Call &call) {
    for (unsigned element = begin; element < end; ++element) {
        const unsigned fail_count = diag->fail_count;
        const unsigned claimed = diag->claimed;
        const unsigned line = diag->line;
        const float p0 = diag->payload[0], p1 = diag->payload[1];
        const float p2 = diag->payload[2], p3 = diag->payload[3];
        const char *file = diag->file;
        compute::seam_speculating = true;
        try {
            call(element);
            compute::seam_speculating = false;
            if (claimed == 0u && diag->claimed != 0u) {
                diag->spec_claim = element;
            }
        } catch (const compute::SeamDefer &) {
            compute::seam_speculating = false;
            diag->fail_count = fail_count;
            diag->claimed = claimed;
            diag->line = line;
            diag->payload[0] = p0;
            diag->payload[1] = p1;
            diag->payload[2] = p2;
            diag->payload[3] = p3;
            diag->file = file;
            diag->spec_deferred[diag->spec_count++] = element;
        }
    }
}

#define DIAG_ASSERT4(diag, cond, p0, p1, p2, p3)                                \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ChunkDiag *_ppf_d = (diag);                                     \
            if (_ppf_d != nullptr) {                                           \
                _ppf_d->fail_count += 1u;                                      \
                if (_ppf_d->claimed == 0u) {                                   \
                    _ppf_d->payload[0] = (p0);                                 \
                    _ppf_d->payload[1] = (p1);                                 \
                    _ppf_d->payload[2] = (p2);                                 \
                    _ppf_d->payload[3] = (p3);                                 \
                    _ppf_d->file = __FILE__;                                   \
                    _ppf_d->line = (unsigned)__LINE__;                         \
                    _ppf_d->claimed = 1u;                                      \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

#define DIAG_ASSERT(diag, cond) DIAG_ASSERT4(diag, cond, 0.0f, 0.0f, 0.0f, 0.0f)

#define DIAG_BOUNDS_CHECK(diag, index, count)                                   \
    do {                                                                       \
        unsigned _ppf_i = (unsigned)(index);                                   \
        unsigned _ppf_n = (unsigned)(count);                                   \
        DIAG_ASSERT4(diag, _ppf_i < _ppf_n, (float)_ppf_i, (float)_ppf_n, 0.0f, \
                    0.0f);                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// SIX NAMES ARE DELIBERATELY MISSING, and adding them would be a change rather
// than a completion: SM_POPCOUNT, SM_SIMD_BALLOT, SM_THREADGROUP_BARRIER,
// SM_SHUFFLE_DOWN, SM_SHUFFLE_UP and SM_SIMD_WIDTH.
//
// Each of them names a warp or threadgroup operation, or the lane count such an
// operation is written against, that has no single-thread meaning, and no host
// spelling for any of them exists anywhere in this tree.
// Their only users are primitives/radix.kernel.cpp, primitives/reduce.kernel.cpp and
// primitives/scan.kernel.cpp, whose only includers are kernels/radix_sort.cu,
// kernels/reduce.cu and kernels/exclusive_scan.cu, all compiled by nvcc. So the
// host target is not universal across the shared surface, and a host
// translation unit that reaches one of those headers fails to compile on the
// undefined name, which is the loud answer.
//
// Inventing bodies for them would be a design decision about what a lane
// shuffle means with one lane, taken here rather than where the kernel is, and
// it would replace that compile error with an oracle that silently computes
// something else.
// ---------------------------------------------------------------------------
