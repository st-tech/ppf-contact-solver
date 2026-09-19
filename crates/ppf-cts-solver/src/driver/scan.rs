// File: scan.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The exclusive scan: a multi-level block scan dispatched over the whole
//! array.
//!
//! # Why it is a real parallel scan
//!
//! A kernel that opens `if (element != 0u) { return; }` and is dispatched at
//! extent 1 is not running on the device, it is a single lane behind a
//! dispatch, whatever answer it produces. Every caller that needs a row-offset
//! array scanned dispatches this instead: O(n) work in O(log n) dispatches,
//! against the O(n) serial device steps such a pass would take.
//!
//! # Why the intra-block scan is not cooperative
//!
//! A warp shuffle and a threadgroup barrier are DEVICE-ONLY.
//! `crates/ppf-cts-solver/src/kernels/primitives/scan.kernel.cpp` carries that
//! shape and names `compute::shuffle_up` and `compute::threadgroup_barrier`,
//! neither of which `seam_host.h` defines, so a `[[seam::group]]` entry over one
//! of its bodies would render on CUDA and on Metal and not on the host. Rule
//! (1a) is that the driver is written once: a phase the CPU backend cannot run
//! is a fork of the step loop rather than a slower port. The three entries this
//! module dispatches are in `scan_levels.kernel.cpp`, which names no warp or
//! threadgroup primitive at all.
//!
//! So the cooperative layer is what comes out, and nothing else does. The
//! structure is a multi-level block scan: sum each block, scan the block sums,
//! add each block's base back down. What is free is how one block's elements
//! are spread over a group's lanes.
//!
//! **That distribution does not change the result, and that is what makes it a
//! performance choice rather than a numerical one.** The values are `unsigned`
//! and integer addition is exact and associative, so every distribution over
//! the same elements produces bit-identical output. This is not the float case,
//! where an order change is a different answer.

use ppf_cts_compute::{AllocLabel, Buffer, Device, EncoderExt, Fault, Handle};

use super::kernels::{ScanBlockApplyArgs, ScanBlockTotalArgs, ScanZeroArgs};

/// The width one thread walks, and the branching factor of the recursion.
///
/// CHOSEN FOR PARALLELISM, WHICH IS WHAT THIS KERNEL IS SHORT OF. One thread
/// walks one block serially, so `BLOCK` sets BOTH the serial depth per thread
/// and the thread count, `ceil(n / BLOCK)`, and the two trade against each
/// other directly.
///
/// AT 4096 IT WAS MEASURED AS THE SECOND LARGEST COST IN THE SOLVE. `nsys` over
/// a 10-frame `plastic` put `scan_block_apply` at 253.8 us per call and 15.8
/// percent of all GPU time. At 4096 a million rows get 245 threads, which is
/// not a GPU's worth of work however much it beats a single-lane pass: judging
/// it against that pass sets the bar in the wrong place.
///
/// At 256 the same array gets 3907 threads and each walks 256 elements. The
/// recursion goes one level deeper, since two levels now cover 65,536 rows
/// rather than 16.7 million, and a third level is one dispatch over a few
/// hundred elements. That is a good trade: the extra level is negligible and
/// the parallelism is 16x.
///
/// MEASURED, on the same 10-frame `plastic` profile that found it
/// `[L40S, 2026-08-29]`: `scan_block_apply` went 253.8 us to 22.4 us
/// per call and `scan_block_total` 86.9 us to 10.0 us, taking the two together
/// from 21.2 percent of all GPU time to 2.4 percent. That is about 81 ms of
/// roughly 422 ms removed by one constant.
///
/// THE BLOCK IS SCANNED BY A GROUP RATHER THAN BY ONE THREAD, which is what
/// makes this a parallel scan. `BLOCK` sets only the serial DEPTH: a group of
/// `SCAN_LANES` lanes strides its
/// `BLOCK` elements, so each lane sums `BLOCK / SCAN_LANES` of them and
/// `compute::block_sum` folds the lanes.
const BLOCK: u32 = 256;

/// THE LANES PER BLOCK, matching `[[seam::scratch(64)]]` on the entry and the
/// fold width the PCG already uses. `BLOCK / SCAN_LANES` is each lane's serial
/// run, so the two constants are read together: raising `BLOCK` alone lengthens
/// that run, and raising both keeps it fixed while widening the group.
const SCAN_LANES: u32 = 64;

/// The per-level block sums.
///
/// ONE BUFFER FOR EVERY LEVEL, not one per level, because the levels are
/// disjoint in time: level `d + 1` reads what level `d` wrote and no level
/// reads its own input again. `SPAN` is the largest any level needs, and
/// `Buffer::size` grows only past capacity, so a per-step scan of a smaller
/// array reuses the same allocation rather than making a new one.
/// A buffer a per-step routine needs is allocated ONCE and reused, and there is
/// no `Drop`, so a per-call allocation would LEAK its arena span rather than
/// churn it.
#[derive(Debug, Default)]
pub struct ScanScratch {
    totals: Buffer<u32>,
}

impl ScanScratch {
    /// Size the scratch for one array of `longest` elements.
    ///
    /// Called by [`Self::exclusive`] itself; a caller that knows the widest
    /// array up front may call it early so the allocation lands with the rest
    /// of a step's sizing rather than inside the step.
    ///
    /// The sum over levels of `ceil(n / BLOCK^(d + 1))` is under
    /// `ceil(n / BLOCK) * BLOCK / (BLOCK - 1)`, so one more block than the
    /// first level needs covers every level below it.
    pub fn size<D: Device>(&mut self, device: &mut D, longest: u32) -> Result<(), Fault> {
        let first = longest.div_ceil(BLOCK);
        let span = (first + first.div_ceil(BLOCK) + 2) as usize;
        self.totals
            .size(device, span.max(2), AllocLabel("scan.totals"))
    }

    /// The exclusive scan of `count` elements at `array`, in place.
    ///
    /// # Safety
    /// `array` must name at least `count` elements and outlive the call.
    pub unsafe fn exclusive<D: Device>(
        &mut self,
        device: &mut D,
        region: &'static str,
        array: Handle,
        count: u32,
    ) -> Result<(), Fault> {
        if count == 0 {
            return Ok(());
        }
        // THE SCRATCH SIZES ITSELF, so a caller cannot forget. `Buffer::size`
        // grows only past CAPACITY, so this allocates on the first scan of a
        // new largest array and never again, which is the per-step reuse the
        // scratch exists for, without a second place that has to know the
        // width. What it
        // does do every call is zero the scratch, which is `ceil(count / BLOCK)`
        // elements and not the array's own length.
        self.size(device, count)?;
        // ONE SUBMIT FOR EVERY LEVEL, not one per level. The levels are
        // ordered among themselves and nothing between them asks the host
        // anything, so they are entries of ONE region; consecutive entries of a
        // region are barrier-ordered on every backend. Measured on ten frames of
        // `drape`, the three dynamic-CSR scans alone cost 1,302 submits as a
        // launch per level.
        //
        // `recurse` STAYS, unused by this path, as the second implementation the
        // differential test below needs: the two are only a check on each other
        // while they are two.
        device
            .run(region, |encoder| {
                // Safety: the caller's contract, forwarded.
                unsafe { self.encode(encoder, array, count, 0, 1) }
            })
            .map(|_| ())
    }

    /// One level: sum the blocks, scan the sums, add each block's base back.
    ///
    /// `offset` is where this level's block sums live inside the one scratch
    /// buffer. Level `d` writes above what level `d - 1` still holds, which is
    /// what lets the levels share an allocation: a level's sums are read by the
    /// level above it and then again on the way back down.
    /// The scan's levels, ENCODED into a caller's region rather than submitted.
    ///
    /// The same levels [`Self::recurse`] issues, in the same order, with each
    /// one an entry of the caller's region instead of its own submit.
    /// Consecutive entries of one region are barrier-ordered on every backend,
    /// which is what a level needs of the one before it.
    ///
    /// # Safety
    /// As [`Self::exclusive`], and the scratch must already be sized:
    /// allocation cannot happen inside a region.
    pub unsafe fn encode_exclusive(
        &mut self,
        encoder: &mut dyn ppf_cts_compute::Encoder,
        array: Handle,
        count: u32,
    ) -> Result<(), Fault> {
        if count == 0 {
            return Ok(());
        }
        self.encode(encoder, array, count, 0, 1)
    }

    /// # Safety
    /// As [`Self::encode_exclusive`].
    unsafe fn encode(
        &mut self,
        encoder: &mut dyn ppf_cts_compute::Encoder,
        array: Handle,
        count: u32,
        offset: u32,
        write_total: u32,
    ) -> Result<(), Fault> {
        // ONE GUARD, NOT TWO. `recurse` dispatches the zero kernel when the
        // level is trivial AND owes no total, and otherwise falls through to the
        // block path, which is what WRITES the total. Splitting this into
        // `count <= 1 && write_total == 0 -> return` and `count <= 1 ->
        // zero` inverts it: a one-element scan that owes a total then takes the
        // zero path and the total is never written.
        if count <= 1 && write_total == 0 {
            let zero = ScanZeroArgs {
                data: array,
                count,
                elements: count,
                seam_arena_count: 0,
            };
            return encoder.elements(&zero, count);
        }
        let blocks = count.div_ceil(BLOCK);
        let totals = self.totals.span(offset as usize, blocks as usize);
        let total = ScanBlockTotalArgs {
            data: array,
            count,
            block_size: BLOCK,
            total: totals,
            blocks,
            seam_arena_count: 0,
        };
        encoder.groups(&total, blocks, SCAN_LANES)?;
        self.encode(encoder, totals, blocks, offset + blocks, 0)?;
        let apply = ScanBlockApplyArgs {
            data: array,
            count,
            block_size: BLOCK,
            write_total,
            base: totals,
            blocks,
            seam_arena_count: 0,
        };
        encoder.groups(&apply, blocks, SCAN_LANES)
    }

    #[cfg(test)]
    unsafe fn recurse<D: Device>(
        &mut self,
        device: &mut D,
        region: &'static str,
        array: Handle,
        count: u32,
        offset: u32,
        write_total: u32,
    ) -> Result<(), Fault> {
        // THE BASE, AND ITS EXTENT IS THE ARRAY'S OWN LENGTH. The exclusive
        // scan of a one-element array is a zero, and writing it from the host
        // would be the relocation this module exists to remove.
        //
        // ONLY AN INNER LEVEL TAKES IT. A level that owes the grand total has
        // to reach `scan_block_apply` to write it, so an outer call over one
        // element goes the long way round: one block, whose own base is the
        // level below's zero. Guarding on the count alone would return before
        // the total was written, and the caller would read the previous step's
        // width out of that slot.
        if count <= 1 && write_total == 0 {
            let zero = ScanZeroArgs {
                data: array,
                count,
                elements: count,
                seam_arena_count: 0,
            };
            return device.launch(region, &zero, count).map(|_| ());
        }
        let blocks = count.div_ceil(BLOCK);
        let totals = self.totals.span(offset as usize, blocks as usize);
        let total = ScanBlockTotalArgs {
            data: array,
            count,
            block_size: BLOCK,
            total: totals,
            blocks,
            seam_arena_count: 0,
        };
        device.groups_launch(region, &total, blocks, SCAN_LANES)?;
        self.recurse(device, region, totals, blocks, offset + blocks, 0)?;
        let apply = ScanBlockApplyArgs {
            data: array,
            count,
            block_size: BLOCK,
            write_total,
            base: totals,
            blocks,
            seam_arena_count: 0,
        };
        device
            .groups_launch(region, &apply, blocks, SCAN_LANES)
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;
    use ppf_cts_compute::ReadbackBuffer;

    /// The scan against a host oracle, at the sizes that exercise every level.
    ///
    /// THE SIZES ARE CHOSEN AGAINST `BLOCK`, not arbitrarily: one under a block,
    /// exactly a block, one over (which is the first size to need two levels),
    /// and one that makes the SECOND level itself more than one block. A scan
    /// tested only below `BLOCK` never recurses at all and would pass with the
    /// recursion deleted.
    #[test]
    fn the_scan_matches_a_host_oracle_at_every_recursion_depth() {
        let mut device = host_device();
        for &count in &[1u32, 2, 7, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK + 5] {
            // `count + 1` slots: the scan writes the exclusive prefix into the
            // first `count` and the GRAND TOTAL into the last, which is what
            // every caller reads back as the array's width. A test that
            // checked only the prefix would
            // pass with that write deleted, and the width the next pass
            // allocates against would be the previous step's.
            let mut counts: Vec<u32> = (0..count).map(|i| (i % 13) + 1).collect();
            let mut expected = Vec::with_capacity(counts.len() + 1);
            let mut running = 0u32;
            for &c in &counts {
                expected.push(running);
                running += c;
            }
            expected.push(running);
            counts.push(0xdead_beef);
            let mut array = ReadbackBuffer::<u32>::default();
            array
                .size(&mut device, counts.len(), AllocLabel("test.scan"))
                .expect("sized");
            array.seed(&mut device, &counts).expect("seeded");
            let mut scratch = ScanScratch::default();
            scratch.size(&mut device, count).expect("scratch sized");
            // Safety: the buffer outlives the call and names `count` elements.
            unsafe {
                scratch
                    .exclusive(&mut device, "test.scan", array.handle(), count)
                    .expect("scanned");
            }
            array.download(&mut device).expect("downloaded");
            assert_eq!(
                array.host(),
                expected.as_slice(),
                "count {count} scanned wrong"
            );
        }
    }

    /// A zero-length scan touches nothing and does not fail.
    #[test]
    fn an_empty_scan_is_a_no_op() {
        let mut device = host_device();
        let mut scratch = ScanScratch::default();
        scratch.size(&mut device, 0).expect("scratch sized");
        let mut array = ReadbackBuffer::<u32>::default();
        array
            .size(&mut device, 1, AllocLabel("test.scan.empty"))
            .expect("sized");
        // Safety: as above.
        unsafe {
            scratch
                .exclusive(&mut device, "test.scan", array.handle(), 0)
                .expect("scanned");
        }
    }

    /// The ENCODED scan must agree with the SUBMITTING one, bit for bit.
    ///
    /// **THE FIRST VERSION OF THIS TEST WAS WORTHLESS AND PASSED.** It compared
    /// `exclusive` against `encode_exclusive` at a moment when `exclusive` had
    /// already been rewritten to wrap `encode`, so it compared the new path to
    /// itself and said nothing. `exclusive` keeps its own recursion for that
    /// reason: the two forms are only a check on each other while they are two
    /// implementations.
    #[test]
    fn the_encoded_scan_agrees_with_the_submitting_one() {
        let mut device = host_device();
        for &count in &[1u32, 2, 16, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK + 5] {
            let input: Vec<u32> = (0..count).map(|i| (i % 13) + 1).collect();
            let mut a = Buffer::<u32>::none();
            let mut b = Buffer::<u32>::none();
            for buf in [&mut a, &mut b] {
                buf.size(&mut device, count as usize + 1, AllocLabel("t.scan"))
                    .unwrap();
                buf.write(&mut device, 0, &input).unwrap();
            }
            let mut sa = ScanScratch::default();
            let mut sb = ScanScratch::default();
            sb.size(&mut device, count).unwrap();
            // Safety: both arrays outlive the calls.
            sa.size(&mut device, count).unwrap();
            // THE SUBMITTING RECURSION DIRECTLY, not through `exclusive`, which
            // now takes the encoded path: comparing `exclusive` against
            // `encode_exclusive` would compare one implementation to itself,
            // which an earlier version of this test did and passed.
            unsafe {
                sa.recurse(&mut device, "t", a.span(0, count as usize), count, 0, 1)
            }
            .unwrap();
            unsafe {
                device
                    .run("t", |e| sb.encode_exclusive(e, b.span(0, count as usize), count))
                    .unwrap()
            };
            let mut ga = vec![0u32; count as usize + 1];
            let mut gb = vec![0u32; count as usize + 1];
            a.read(&mut device, 0, &mut ga).unwrap();
            b.read(&mut device, 0, &mut gb).unwrap();
            assert_eq!(ga, gb, "the two scan forms disagree at count {count}");
        }
    }
}
