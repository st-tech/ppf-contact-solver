// File: vec_ops.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Backend-neutral per-element vector operations. CUDA and Metal provide only
// the thread index and launch ABI; the arithmetic is written once here.

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The two facts a backend cannot infer are written as
// C++ attributes: `[[seam::host_device_fn]]` is the execution space, and
// `[[seam::device]]` is the address space of a pointer parameter. MSL requires
// the second on every pointer and reference type; CUDA and the host have one
// address space and are handed the same declarations with it removed.
//
// The bodies below are called from a kernel AND from the host: primitives/vec_ops
// launches them on the device, and metal/vec_ops.mm calls the same bodies
// directly to produce the oracle its fixtures compare against. So they take the
// host-and-device execution space, `[[seam::host_device_fn]]`, rather than the
// device-only one.
//
// `fmath::fma` is the fused multiply-add, defined by the backend prologue:
// ppf-cts-compute/cuda/seam_cuda.cuh under nvcc, kernels/seam/seam_host.h on
// the host, and the prologue in ppf-cts-compute/metal/shader_compiler.mm under
// MSL. It takes `float` rather than a template parameter, so a widened
// argument is narrowed at the call instead of emitting a float64 device
// instantiation.

template <class T>
[[seam::host_device_fn]] inline void
vec_fill(T *array, unsigned index, T value) {
    array[index] = value;
}

template <class T>
[[seam::host_device_fn]] inline void
vec_copy(const T *source, T *destination,
             unsigned index) {
    destination[index] = source[index];
}

template <class T>
[[seam::host_device_fn]] inline void
vec_add_scaled(const T *source,
                   T *destination, T scale, unsigned index) {
    destination[index] += scale * source[index];
}

template <class T>
[[seam::host_device_fn]] inline void
vec_combine(const T *source_a,
                const T *source_b,
                T *destination, T a, T b, unsigned index) {
    destination[index] = a * source_a[index] + b * source_b[index];
}

[[seam::host_device_fn]] inline void vec_add_scaled_indirect(
    const float *source, float *destination,
    const float *coefficient, float sign, unsigned index) {
    float scale = sign * (*coefficient);
    destination[index] =
        fmath::fma(scale, source[index], destination[index]);
}

// The entry point for `vec_combine_indirect`, declared once and rendered
// for four targets: the `__global__` nvcc compiles, the `kernel void` the
// Metal shader compiler compiles, the `vec_combine_indirect_entry` shim
// a host C++ compiler compiles, and the Rust `#[repr(C)]` twin the driver
// fills, all three under one name. A
// hand-written entry point is a mirror pair per backend with nothing linking
// the halves; this declaration is the one place the argument record exists.
//
// It is a DECLARATION, not a definition, and that is what makes an entry point
// unable to acquire logic: there is no body to put a branch, a phase order or
// a convergence test in. What the generator adds is the guard against
// `[[seam::count]]`, the resolution of each `[[seam::device]]` pointer from its
// (arena, offset) handle, and the call passing every other parameter through
// unchanged with `[[seam::index]]` in the position written here.
[[seam::entry(index)]]
[[seam::host_device_fn]] inline void vec_combine_indirect(
    const float *source_a,
    const float *source_b, float *destination,
    float a, const float *coefficient_b, unsigned index) {
    destination[index] =
        a * source_a[index] + (*coefficient_b) * source_b[index];
}

// THE SAME COEFFICIENT RULE FOR THE AXPY, and it is the other half of one
// recurrence rather than a second convenience. `x += alpha p` and
// `r -= alpha Ap` are the two updates a PCG iteration makes, and `alpha` is a
// scalar the previous dispatch wrote on the device: taking it back to the host
// to fill a record's `float` field is what makes a whole iteration wait on a
// transfer. `sign` carries the minus of the second update, so both take one
// declaration and neither needs the coefficient negated on the host.
[[seam::entry(count, index)]] void vec_add_scaled_indirect(
    const float *source, float *destination,
    const float *coefficient, float sign,
    unsigned count, unsigned index);

// The three the CPU driver dispatches, declared here for the same reason and
// rendered for the same four targets. Each wraps a TEMPLATE body and names
// `float`, which is the type every caller in this tree instantiates: an entry
// point is one launch shape, so a second element type is a second declaration
// rather than a parameter, and there is nowhere in a declaration to hide a
// branch that picks between them.
[[seam::entry(count, index)]] void vec_copy(
    const float *source, float *destination,
    unsigned count, unsigned index);

[[seam::entry(count, index)]] void vec_add_scaled(
    const float *source, float *destination,
    float scale, unsigned count,
    unsigned index);

[[seam::entry(count, index)]] void vec_combine(
    const float *source_a,
    const float *source_b, float *destination,
    float a, float b, unsigned count,
    unsigned index);

// The fill, declared here for the same reason and rendered for the same four
// targets. It names `float`, as the three above do.
//
// The CPU driver dispatches it, and the array it names is a device handle at
// every call site. The fill opens a buffer at a known value before anything
// reads it: zero for the accumulators a later pass adds into, and the line
// search ceiling for a time of impact, which a minimum narrows only in the
// slots a sweep finds a hit in.
[[seam::entry(count, index)]] void vec_fill(
    float *array, unsigned index, float value,
    unsigned count);

// The same fill over an UNSIGNED array, which is a separate body rather than a
// template because a `[[seam::entry]]` declaration names concrete types: the
// generator emits one record per declaration and a record field has one size.
//
// It exists because a counting pass needs its target at zero before the first
// atomic, and the counts a CSR-style build accumulates are unsigned.
[[seam::entry(index)]]
[[seam::device_fn]] inline void
vec_fill_u32(unsigned *array, unsigned index, unsigned value) {
    array[index] = value;
}

// ONE ELEMENT'S DENSE PACK, scaled by that element's own scalar and accumulated
// into a destination that opens at zero. This is `dedx += mass *
// convert_force(...)` and `d2edx2 += mass * convert_hessian(...)`, with the
// multiply taken from `vec_add_scaled` above rather than spelled again: a
// matrix times a scalar IS that scalar applied to each of its stored floats,
// and an `SMat`'s whole storage is one flat `float[R * C]`, so the elementwise
// form is the same expression on the same bytes.
//
// THE ACCUMULATE-INTO-ZERO SHAPE IS KEPT rather than reduced to a plain store,
// because that is what the call sites do and it is what a second contributor to
// the same element would need.
//
// `stride` IS A RUNTIME ARGUMENT AND THAT IS WHY THE TWO BUFFERS ARE BASE
// POINTERS. The packs dispatched here run from 1 float to 324, so no
// `[[seam::stride(N)]]` literal describes them and the element's own run has to
// be computed from a value the record carries. That leaves the base pointer,
// which is one of the five shapes, and the thread index, which
// `[[seam::index]]` names.
//
// **ONE THREAD PER FLOAT, NOT ONE PER ELEMENT.** The dispatch is flat over the
// whole array: the thread index IS the float's index, and the element it
// belongs to is a division. Dispatching one thread per element with an inner
// loop of `stride` instead would put 144 sequential floats in one thread for a
// tet Hessian, 144 times less parallelism than the array has. Measured on
// `trapped` at 25 frames, that element-wise form cost 227.9 ms of 1,583 ms of
// GPU time, the single largest kernel in the run.
[[seam::device_fn]] inline void element_add_scaled(
    const float *source, float *destination,
    const float *scale, unsigned stride, unsigned index) {
    vec_add_scaled(source, destination, scale[index / stride], index);
}

[[seam::entry(count, index)]] void element_add_scaled(
    const float *source, float *destination,
    const float *scale, unsigned stride,
    unsigned count, unsigned index);

// ---------------------------------------------------------------------------
// ONE LEVEL OF A FOLD, WHICH IS HOW A REDUCTION REACHES EVERY BACKEND.
// ---------------------------------------------------------------------------
//
// `primitives/reduce.kernel.cpp` holds the cooperative form, a warp shuffle
// tree under a threadgroup barrier, and it deliberately has no host rendering:
// `seam_host.h` leaves `compute::simd_width` and
// `compute::threadgroup_barrier` undefined, so a body written against them
// compiles on two targets and nowhere else. `crates/ppf-cts-compute/src/device.rs`
// states the consequence in as many words: a reduction reaches the CPU backend
// only as a body written WITHOUT a barrier. This is that body.
//
// One element sums one fixed block of the input, in ascending index order, and
// writes that block's total. A caller folds an array to a single scalar by
// dispatching this over `ceil(length / width)` elements and dispatching it
// again over the totals, until one element remains; KERNEL COMPLETION is the
// barrier between levels, and it is the decomposition a reduction takes when no
// backend-wide barrier is available. No level is a single lane wearing a
// dispatch: every level but the last covers as many elements as it has
// blocks.
//
// THE SHAPE IS A CONTRACT AND NOT A TUNING KNOB. The block width and the
// ascending order inside a block decide the last bits of every scalar the PCG
// recurrence carries, and the curvature bound in `solver/pcg.kernel.cpp` is
// stated against a fold of this depth: "a per-thread serial accumulation, a
// 256-wide shared tree and a strided fold, each contribution carrying its own
// epsilon". The width is a parameter rather than a constant here so that ONE
// place states it, `src/driver/reduce.rs`, which also states the shape of the
// host folds that remain.
//
// `length` IS THE INPUT'S LENGTH AND `count` IS THE BLOCK COUNT, and they are
// two numbers because they measure two different arrays: the guard bounds the
// thread against the blocks, and the tail block is short. A block whose first
// index is past the end sums nothing and writes a zero, which is the identity.
[[seam::entry(block, total)]]
[[seam::device_fn]] inline float
vec_block_sum(const float *source, unsigned length,
                  unsigned width, unsigned block) {
    const unsigned first = width * block;
    unsigned last = first + width;
    if (last > length) {
        last = length;
    }
    float total = 0.0f;
    for (unsigned k = first; k < last; ++k) {
        total += source[k];
    }
    return total;
}

// THE SAME BLOCK SUM OVER `unsigned`, which is what folds a per-vertex flag
// array down to a count.
//
// A SEPARATE BODY RATHER THAN A TEMPLATE, because a neutral kernel body is not
// generic: the four renderings each take a concrete element type. The float
// form above cannot serve, and not only on types. A count folded through
// `float` is exact only while it stays under 2^24, so a scene past sixteen
// million marked vertices would start rounding its own tally, and the failure
// would be a wrong number rather than a refusal.
//
// NO SATURATION AND NO ATOMIC: the fold is a tree of ordinary adds over
// disjoint blocks, so this needs neither, and an `unsigned` sum of per-vertex
// flags cannot overflow a scene the vertex count itself fits in.
[[seam::entry(block, total)]]
[[seam::device_fn]] inline unsigned
vec_block_sum_u32(const unsigned *source, unsigned length,
                      unsigned width, unsigned block) {
    const unsigned first = width * block;
    unsigned last = first + width;
    if (last > length) {
        last = length;
    }
    unsigned total = 0u;
    for (unsigned k = first; k < last; ++k) {
        total += source[k];
    }
    return total;
}

// THE SAME BLOCK SUM WITH THE BLOCK'S THREADS SHARING IT, which is the shape
// that gets both parallelism and a short chain.
//
// WHY THE ELEMENT FORM ABOVE CANNOT: its element IS a block, so one thread walks
// the block's `width` values and a level's thread count is `values / width`. A
// fold over a 78,000-vertex scene's `3 * n` at width 256 launches 915 threads,
// which does not fill a GPU, and the first level reads every value. Widening
// removes a level and starves it further; narrowing fills it and adds levels.
// Measured that way, the three fold entries take 53 percent of GPU time against
// the SpMV's 10.7.
//
// EACH LANE TAKES A CONTIGUOUS RUN, NOT A STRIDE, and that is the whole reason
// the host arm's bits do not move. `compute::block_sum` folds the lanes in
// ASCENDING order there, so contiguous runs combined in lane order reproduce the
// element form's left-to-right sum EXACTLY. A strided decomposition would not,
// and this fold's association is what the curvature bound is derived against.
// The device arms fold within a warp first and then walk the warp partials,
// which is deterministic and is not those bits, the standing every
// atomic-scatter kernel here already has.
// THE RUN ONE LANE OWNS, stated once because four bodies below need it and a
// second statement of it is a second decomposition that can disagree.
//
// THE DIVISION IS ROUNDED UP so the runs cover the block with no gap, and a lane
// whose run starts past the end gets an empty one, which contributes the
// identity. Contiguous rather than strided: `compute::block_sum` folds lanes in
// ascending order on the host, so contiguous runs keep the difference from the
// element form inside the fp32 summation bound over a cancelling input.
[[seam::device_fn]] inline void
vec_lane_run(unsigned length, unsigned width, unsigned lane, unsigned threads,
             unsigned block, unsigned &run_first,
             unsigned &run_last) {
    const unsigned first = width * block;
    unsigned last = first + width;
    if (last > length) {
        last = length;
    }
    const unsigned span = last > first ? last - first : 0u;
    const unsigned per = (span + threads - 1u) / threads;
    unsigned mine_first = first + lane * per;
    unsigned mine_last = mine_first + per;
    if (mine_first > last) {
        mine_first = last;
    }
    if (mine_last > last) {
        mine_last = last;
    }
    run_first = mine_first;
    run_last = mine_last;
}

[[seam::device_fn]] inline void
vec_block_sum_cooperative(const float *source, unsigned length,
                          unsigned width,
                          float *scratch,
                          float *total, unsigned lane,
                          unsigned threads, unsigned block) {
    unsigned mine_first = 0u;
    unsigned mine_last = 0u;
    vec_lane_run(length, width, lane, threads, block, mine_first, mine_last);
    float mine = 0.0f;
    for (unsigned k = mine_first; k < mine_last; ++k) {
        mine += source[k];
    }
    const float folded = compute::block_sum(mine, scratch, lane, threads);
    // THE WRITER LANE IS A NAME, NOT A LITERAL. `compute::block_sum` leaves the
    // complete total on lane 0 where a warp tree folded it and on the LAST lane
    // where the group's lanes ran one after another, so a body that wrote from
    // lane 0 unconditionally would store a partial sum on the CPU backend.
    if (compute::is_block_writer(lane, threads)) {
        total[block] = folded;
    }
}

// ONE GROUP PER BLOCK. The guard is on the GROUP index, which is uniform across
// the group, so a group returns whole and the fold inside is reached by every
// thread of every group that did not.
//
// THE SCRATCH IS ONE FLOAT PER WARP of the widest group this entry is launched
// with. 32 covers a 1024-thread group at a 32-wide simd, which is the widest
// either device arm allows.
[[seam::entry(count, block)]] [[seam::group]] void vec_block_sum_cooperative(
    const float *source, unsigned length, unsigned width,
    [[seam::scratch(32)]] float *scratch,
    float *total, [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block, unsigned count);

// THE MAGNITUDE LEVEL, cooperatively. The element form beside it was 11.6
// percent of GPU time.
[[seam::device_fn]] inline void vec_block_sum_abs_cooperative(
    const float *source, unsigned length, unsigned width,
    float *scratch, float *total,
    unsigned lane, unsigned threads, unsigned block) {
    unsigned mine_first = 0u;
    unsigned mine_last = 0u;
    vec_lane_run(length, width, lane, threads, block, mine_first, mine_last);
    float mine = 0.0f;
    for (unsigned k = mine_first; k < mine_last; ++k) {
        mine += fmath::abs(source[k]);
    }
    const float folded = compute::block_sum(mine, scratch, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        total[block] = folded;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void
vec_block_sum_abs_cooperative(
    const float *source, unsigned length, unsigned width,
    [[seam::scratch(32)]] float *scratch,
    float *total, [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block, unsigned count);

// TWO SOURCES OF ONE LENGTH, cooperatively. The element form was 26.2 percent of
// GPU time, the largest single kernel in the run.
//
// THE TWO FOLDS TAKE SEPARATE SCRATCH HALVES rather than reusing one, and that
// is not tidiness. `compute::block_sum` stores into the scratch and then reads
// it back; a second call over the same slots could store before every lane of
// the first has read, and the neutral body has no barrier to put between them
// because a barrier is the one thing the host arm cannot express. Two halves
// makes the question not arise.
[[seam::device_fn]] inline void vec_block_sum_pair_cooperative(
    const float *first_source,
    const float *second_source, unsigned length,
    unsigned width, float *scratch,
    float *first_total,
    float *second_total, unsigned lane, unsigned threads,
    unsigned block) {
    unsigned mine_first = 0u;
    unsigned mine_last = 0u;
    vec_lane_run(length, width, lane, threads, block, mine_first, mine_last);
    float first_mine = 0.0f;
    float second_mine = 0.0f;
    for (unsigned k = mine_first; k < mine_last; ++k) {
        first_mine += first_source[k];
        second_mine += second_source[k];
    }
    const float first_folded =
        compute::block_sum(first_mine, scratch, lane, threads);
    const float second_folded =
        compute::block_sum(second_mine, scratch + 32, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        first_total[block] = first_folded;
        second_total[block] = second_folded;
    }
}

[[seam::entry(count, block)]] [[seam::group]] void
vec_block_sum_pair_cooperative(
    const float *first_source,
    const float *second_source, unsigned length,
    unsigned width,
    [[seam::scratch(64)]] float *scratch,
    float *first_total,
    float *second_total, [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block, unsigned count);

// TWO SOURCES OF DIFFERENT LENGTHS, cooperatively. The element form was 15.2
// percent of GPU time. Each chain has its own block count, so a block past one
// chain's end folds only the other.
[[seam::device_fn]] inline void vec_block_sum_dual_cooperative(
    const float *first_source, unsigned first_length,
    const float *second_source, unsigned second_length,
    unsigned width, float *scratch,
    float *first_total,
    float *second_total, unsigned first_count,
    unsigned second_count, unsigned lane, unsigned threads, unsigned block) {
    if (block < first_count) {
        unsigned run_first = 0u;
        unsigned run_last = 0u;
        vec_lane_run(first_length, width, lane, threads, block, run_first,
                     run_last);
        float mine = 0.0f;
        for (unsigned k = run_first; k < run_last; ++k) {
            mine += first_source[k];
        }
        const float folded =
            compute::block_sum(mine, scratch, lane, threads);
        if (compute::is_block_writer(lane, threads)) {
            first_total[block] = folded;
        }
    }
    if (block < second_count) {
        unsigned run_first = 0u;
        unsigned run_last = 0u;
        vec_lane_run(second_length, width, lane, threads, block, run_first,
                     run_last);
        float mine = 0.0f;
        for (unsigned k = run_first; k < run_last; ++k) {
            mine += second_source[k];
        }
        const float folded =
            compute::block_sum(mine, scratch + 32, lane, threads);
        if (compute::is_block_writer(lane, threads)) {
            second_total[block] = folded;
        }
    }
}

[[seam::entry(count, block)]] [[seam::group]] void
vec_block_sum_dual_cooperative(
    const float *first_source, unsigned first_length,
    const float *second_source, unsigned second_length,
    unsigned width,
    [[seam::scratch(64)]] float *scratch,
    float *first_total,
    float *second_total, unsigned first_count,
    unsigned second_count, [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block, unsigned count);

// THE SAME LEVEL OVER MAGNITUDES, which is a second body rather than a flag on
// the first. A flag would be a branch on a record field taken by every element
// of every level, and the two are not interchangeable at the call site either:
// only the FIRST level of an L1 norm takes absolute values, because every level
// above it folds totals that are already non-negative. The pair is
// `src/driver/reduce.rs`'s `sum` and `sum_abs`, which are kept beside each
// other for the same reason: the curvature guard needs a signed sum and a
// bound reduced in the SAME shape, and that is a property of the pair.
[[seam::entry(block, total)]]
[[seam::device_fn]] inline float
vec_block_sum_abs(const float *source, unsigned length,
                      unsigned width, unsigned block) {
    const unsigned first = width * block;
    unsigned last = first + width;
    if (last > length) {
        last = length;
    }
    float total = 0.0f;
    for (unsigned k = first; k < last; ++k) {
        total += fmath::abs(source[k]);
    }
    return total;
}

// TWO INDEPENDENT FOLDS SHARING ONE DISPATCH, which is a launch-count change and
// NOT a shape change. Each output is the block sum of its OWN array, over the
// same width and in the same ascending order the single-array body uses, so
// every scalar is bit-identical to the pair of dispatches it replaces and the
// curvature bound stated against that shape still holds.
//
// THE OUTPUTS ARE BASE POINTERS, NOT `[[seam::scatter]]`, and that distinction
// is the whole reason this is expressible. A `[[seam::scatter]]` buffer is
// where a body's RETURN VALUE lands, so there can only be one of them and the
// generator refuses a second by name. A body that writes its outputs ITSELF
// takes them as plain `[[seam::device]]` pointers and indexes them, which is
// what `contact_narrow.kernel.cpp` does for its five.
//
// WHY THE PCG WANTS IT. The operator writes the curvature terms and their
// magnitudes as two arrays of the same length, and folding each separately is
// two dispatches per level where one will do.
//
// BOTH SOURCES MUST HAVE THE SAME LENGTH. They share `length`, `width` and the
// block index, which is what makes one dispatch legitimate; a caller with two
// lengths has two folds and dispatches two.
[[seam::entry(block)]]
[[seam::device_fn]] inline void vec_block_sum_pair(
    const float *first_source,
    const float *second_source, unsigned length,
    unsigned width, float *first_total,
    float *second_total, unsigned block) {
    first_total[block] = vec_block_sum(first_source, length, width, block);
    second_total[block] = vec_block_sum(second_source, length, width, block);
}

// TWO FOLDS OF DIFFERENT LENGTHS SHARING ONE DISPATCH, which the pair above
// cannot express. The pair shares one `length` and one block count, and its
// comment says why that is the honest restriction for it: two arrays folded in
// lockstep are one fold shape. THE CHAINS THIS SERVES ARE NOT IN LOCKSTEP. The
// PCG folds `||r||_1` over `3 * rows` floats and `r . z` over `rows` of them,
// one after the other and with nothing between, so they are independent in
// every way except that each level of one happens to sit beside a level of the
// other.
//
// EACH SOURCE CARRIES ITS OWN LENGTH AND ITS OWN BLOCK COUNT, and the dispatch
// covers the LARGER count. A thread past a source's own count writes nothing
// for that source, which is what makes the two chains independent rather than
// padded: the shorter chain reaches one element first, its final value is
// already in its output, and the levels above it pass a count of zero and leave
// that output alone.
//
// THE ARITHMETIC IS UNCHANGED, and that is the whole claim. Each source is
// summed over the same `width` in the same ascending order, at the same level,
// as the single-array chain it replaces, so both scalars are bit-identical to
// the two chains this merges and every bound stated against that shape still
// holds. It is a launch-count change and nothing else, the same argument the
// pair makes.
//
// THE FIRST LEVEL OF AN L1 NORM IS NOT THIS BODY. `||r||_1` takes magnitudes at
// its first level and no level above it does, so that one level dispatches
// `vec_block_sum_abs` alone and the merge begins at the level after it. A flag
// here would be a branch on a record field taken by every element of every
// level, which is the same reason `vec_block_sum_abs` is its own body.
[[seam::entry(block)]]
[[seam::device_fn]] inline void vec_block_sum_dual(
    const float *first_source, unsigned first_length,
    const float *second_source, unsigned second_length,
    unsigned width, float *first_total,
    float *second_total, unsigned first_count,
    unsigned second_count, unsigned block) {
    if (block < first_count) {
        first_total[block] =
            vec_block_sum(first_source, first_length, width, block);
    }
    if (block < second_count) {
        second_total[block] =
            vec_block_sum(second_source, second_length, width, block);
    }
}
