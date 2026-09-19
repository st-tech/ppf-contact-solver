// File: scan_levels.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the four forms the four targets read.
//
// A SEPARATE FILE FROM `scan.kernel.cpp`, AND THE SPLIT IS THE POINT. That file
// is DEVICE-ONLY by design and the build's `KERNELS` list says so: every body in
// it names `compute::simd_width`, `compute::shuffle_up` or
// `compute::threadgroup_barrier`, none of which `seam_host.h` defines, so a host
// translation unit that included it would stop on an undefined name. A
// `.kernel.cpp` is compiled whole, so a host-runnable body cannot live beside a
// device-only one; hence two files rather than one, and hence this one carries
// no warp or threadgroup name at all.

// ---------------------------------------------------------------------------
// THE ELEMENT-WISE MULTI-LEVEL SCAN, which is the one the driver dispatches.
//
// WHY IT IS NOT THE THREE BODIES IN `scan.kernel.cpp`. Those carry the
// cooperative intra-block scan, and they are device-only for the reason this
// file's header states: they name `compute::simd_width`, `compute::shuffle_up`
// and `compute::threadgroup_barrier`, and `seam_host.h` defines none of the
// three. A `[[seam::group]]` entry over `block_exclusive_scan` would therefore
// render on CUDA and on Metal and NOT on the host, and the driver is written
// once: a phase the CPU backend cannot run is a fork of the step loop, not a
// slower port. So the cooperative layer is what comes out, and nothing else
// does.
//
// THE STRUCTURE IS A MULTI-LEVEL BLOCK SCAN: sum each block, scan the block
// sums, add each block's base back down. That is exactly the three entries
// below, and the recursion in `driver::scan` is the level loop over them.
//
// HOW THE WORK INSIDE ONE BLOCK IS SPREAD OVER A GROUP'S LANES IS FREE. The
// values are `unsigned` and integer addition is exact and associative, so every
// distribution over the same elements produces bit-identical output, which is
// what makes the choice a performance one rather than a numerical one. What it
// decides is the cost: O(n) work in O(log n) dispatches against the O(n) SERIAL
// device steps a single-lane pass would take.

// One block's total, for the level above.
//
// `block_size` IS A PARAMETER RATHER THAN A CONSTANT because the recursion
// hands the same body arrays of very different lengths, and the level that
// scans the block sums wants the same width as the level below it: a width
// baked in here would be a second place to keep in step with the driver's
// `ceil(count / block_size)` extent.
// ONE GROUP PER BLOCK, THE LANES STRIDING ITS ELEMENTS, with the per-lane
// partial sums folded through `compute::block_sum` below.
//
// WHY THE LANES AND NOT ONE THREAD. Summing `block_size` elements in a single
// thread measures 359.5 ms on `drape`, 16.5 us per launch, against 39.3 ms and
// 1.2 us for a lane-cooperative fold of the same data: a factor of 13.7 per
// dispatch and 9.14 in total. A residency census cannot see that at all, since
// it counts host relocations and a serial pass over a device array is not one.
//
// THE BOUND IS TESTED PER ELEMENT, not by breaking out of the loop. A lane
// striding by the group width reaches its elements out of order relative to its
// neighbors, so an early `break` on the first out-of-range index would drop
// the in-range elements of every LATER stride.
// ONE BODY, NOT A TWIN. It names only `compute::block_sum` and
// `compute::is_block_writer`, and `seam_host.h` defines BOTH: the six names it
// leaves undefined are `bits::popcount`, `compute::simd_ballot`,
// `compute::threadgroup_barrier`, `compute::shuffle_down`, `compute::shuffle_up`
// and `compute::simd_width`, and this body uses none of them. The
// `[[seam::cooperative]]` attribute is what created the obligation to write a
// serial twin, through the generator's refusal; the seam never required one.
//
// AND THE TWIN COULD NOT EVEN DIFFER IN ITS ANSWER. The fold is over `unsigned`,
// whose addition is exactly associative, so the strided-then-folded sum and the
// twin's sequential one were bit-identical rather than merely close. Rule
// (1-LANE)'s license for a twin to associate differently had nothing to bite on.
[[seam::device_fn]] inline void scan_block_total(
    const unsigned *data, unsigned count, unsigned block_size,
    unsigned *scratch,
    unsigned *total, unsigned lane, unsigned threads,
    unsigned block_index) {
    const unsigned begin = block_index * block_size;
    unsigned mine = 0u;
    for (unsigned i = lane; i < block_size; i += threads) {
        const unsigned at = begin + i;
        if (at < count) {
            mine += data[at];
        }
    }
    const unsigned folded = compute::block_sum(mine, scratch, lane, threads);
    if (compute::is_block_writer(lane, threads)) {
        total[block_index] = folded;
    }
}

// The exclusive prefix within one block, seeded by that block's own base.
//
// IN PLACE, AND THE COUNT IS READ BEFORE THE OFFSET IS WRITTEN, which is the
// same ordering `dynamic_csr_build_offsets` states and for the same reason: a
// caller scans one array in place, so writing first would overwrite the count
// this thread has not read yet.
// THE IN-BLOCK EXCLUSIVE SCAN, SPLIT ACROSS THE LANES. Each lane owns a
// CONTIGUOUS run of the block, so the three passes below are: sum my run, scan
// those per-lane sums across the group, then walk my own run from the base that
// scan gives me. That is the standard two-level scan, and it is what makes the
// serial depth `block_size / threads` rather than `block_size`.
//
// WHY THE SPLIT, on the same `drape` profile that priced `scan_block_total`: a
// single-lane walk of the block measures 1,055.8 ms over 22,220 launches, 4.2
// percent of ALL GPU time, against 23.3 ms for a two-kernel lane-cooperative
// scan of the same data. A factor of 45, the largest ratio measured anywhere in
// this solver.
//
// THE RUNS ARE CONTIGUOUS, NOT STRIDED, and that is the opposite of
// `scan_block_total` above for a reason: a prefix sum is ordered, so a lane
// must own an unbroken span for its own serial walk to mean anything. It also
// makes an early exit correct here where the strided body could not have one,
// since every index after the first out-of-range one in a contiguous run is
// out of range too.
//
// NO LANE TOUCHES ANOTHER LANE'S ELEMENTS, so the read in the first pass and
// the write in the third need no barrier between them; the only shared state
// is `scratch`, and every access to it is fenced.
[[seam::device_fn]] [[seam::cooperative]] inline void scan_block_apply(
    unsigned *data, unsigned count, unsigned block_size,
    unsigned write_total, const unsigned *base,
    unsigned *scratch, unsigned lane, unsigned threads,
    unsigned block_index) {
    const unsigned begin = block_index * block_size;
    // THE RUN IS ROUNDED UP so the lanes cover the block even when the width
    // does not divide it, and the span is then clamped BOTH to the block and to
    // the array. Clamping to `count` alone would let a high lane of one block
    // walk into the next block's elements.
    const unsigned run = (block_size + threads - 1u) / threads;
    unsigned block_end = begin + block_size;
    if (block_end > count) {
        block_end = count;
    }
    const unsigned lo = begin + lane * run;
    unsigned hi = lo + run;
    if (hi > block_end) {
        hi = block_end;
    }
    if (hi < lo) {
        hi = lo;
    }

    unsigned mine = 0u;
    for (unsigned at = lo; at < hi; ++at) {
        mine += data[at];
    }
    scratch[lane] = mine;
    compute::threadgroup_barrier();

    // THE SCAN ACROSS LANES, inclusive, in place. The read is separated from
    // the write by a barrier on both sides because every lane is reading a slot
    // some other lane is about to write.
    for (unsigned offset = 1u; offset < threads; offset <<= 1u) {
        const unsigned addend = lane >= offset ? scratch[lane - offset] : 0u;
        compute::threadgroup_barrier();
        scratch[lane] += addend;
        compute::threadgroup_barrier();
    }

    // AN EXCLUSIVE BASE FROM AN INCLUSIVE SCAN is the lane below's total, and
    // lane 0 owns none.
    unsigned running =
        base[block_index] + (lane == 0u ? 0u : scratch[lane - 1u]);
    for (unsigned at = lo; at < hi; ++at) {
        const unsigned value = data[at];
        data[at] = running;
        running += value;
    }

    // THE GRAND TOTAL GOES IN SLOT `count`, which is what
    // `dynamic_csr_build_offsets` did with its `offset[row_count_size] = total`
    // and what every caller of it reads back as the array's width. The LAST
    // LANE of the last block is the one that knows it: its `running` has just
    // passed through every element at or below this block. It is the last lane
    // on BOTH cooperative arms rather than whichever `is_block_writer` names,
    // because the total here comes from the scan this body wrote into
    // `scratch`, not from a shuffle tree whose answer lands on lane 0.
    //
    // Dropping this is not a small difference: the offsets are still right and
    // the width the next pass allocates against is whatever the slot held from
    // the previous step, which reads as a corrupt matrix far from here.
    if (write_total != 0u && begin + block_size >= count &&
        lane + 1u == threads) {
        data[count] = running;
    }
}

// THE SERIAL TWIN every cooperative body owes, which
// `ppf-cts-compute/seam/kernelgen.py` enforces by name. One lane walks the
// whole block and the others return.
[[seam::device_fn]] [[seam::serial]] inline void scan_block_apply(
    unsigned *data, unsigned count, unsigned block_size,
    unsigned write_total, const unsigned *base,
    unsigned *scratch, unsigned lane, unsigned threads,
    unsigned block_index) {
    (void)scratch;
    if (!compute::is_block_writer(lane, threads)) {
        return;
    }
    const unsigned begin = block_index * block_size;
    unsigned running = base[block_index];
    for (unsigned i = 0; i < block_size; ++i) {
        const unsigned at = begin + i;
        if (at >= count) {
            break;
        }
        const unsigned value = data[at];
        data[at] = running;
        running += value;
    }
    if (write_total != 0u && begin + block_size >= count) {
        data[count] = running;
    }
}

[[seam::device_fn]] inline void scan_zero(
    unsigned *data, unsigned count, unsigned index) {
    if (index < count) {
        data[index] = 0u;
    }
}

[[seam::entry(blocks, block_index)]] [[seam::group]] void scan_block_total(
    const unsigned *data, unsigned count, unsigned block_size,
    [[seam::scratch(64)]] unsigned *scratch,
    unsigned *total,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block_index,
    unsigned blocks);

[[seam::entry(blocks, block_index)]] [[seam::group]] void scan_block_apply(
    unsigned *data, unsigned count, unsigned block_size,
    unsigned write_total, const unsigned *base,
    [[seam::scratch(64)]] unsigned *scratch,
    [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned threads,
    unsigned block_index,
    unsigned blocks);

[[seam::entry(elements, index)]] void scan_zero(
    unsigned *data, unsigned count,
    unsigned index,
    unsigned elements);
