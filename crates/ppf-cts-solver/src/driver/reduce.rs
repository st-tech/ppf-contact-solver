// File: crates/ppf-cts-solver/src/driver/reduce.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Reductions, and the one place in this backend where there is deliberately no
//! shared body.
//!
//! Every other kernel here calls C++ that CUDA compiles from the same bytes. A
//! fold is the documented exception: CUDA reduces through a warp shuffle tree
//! and a block tree, and neither has a CPU spelling, so there is no per-element
//! arithmetic to extract. What is shared instead is a CONTRACT, and this module
//! is where that contract is written down.
//!
//! # The contract
//!
//! 1. **The shape is fixed and independent of the thread count.** The array is
//!    cut into fixed-size blocks, each block is summed in a fixed order, and the
//!    block totals are summed in index order. Rayon may run the blocks in any
//!    order and on any number of threads; it may not change which values land in
//!    which block, because that is what would move the result.
//! 2. **A partial sum is `f32`, matching the device.** Widening to `f64` here
//!    would be more accurate and would make this backend disagree with CUDA in a
//!    direction no tolerance was written for.
//! 3. **The block width is part of the contract, not a tuning knob.** Changing
//!    it changes the answer in the last bits, so it moves with a recorded
//!    measurement or not at all.
//!
//! # Why this matters beyond tidiness
//!
//! The PCG curvature verdict is read off a sum of signed contributions against a
//! bound scaled by `CG_CURVATURE_NOISE_SLACK`, and on CUDA that constant is
//! justified at `solver.cu:57-66` by the device's specific fold: "a per-thread
//! serial accumulation, a 256-wide shared tree and a strided fold". That
//! justification does not transfer to any other shape. So a CPU fold whose shape
//! drifts silently invalidates a bound that decides whether a solve aborts, and
//! the failure is quiet in the worst direction: a bound that is too loose
//! truncates a healthy solve at iteration 1 and returns a garbage direction with
//! nothing logged.
//!
//! Until that constant is either parameterized on the shape or moved into the
//! backend contract (an open item in the plan), this module's job is to make the
//! CPU shape a fixed, stated thing rather than whatever rayon chose today.

// The folds themselves are called from `step.rs`, `contact.rs`, `collider.rs`
// and `pcg.rs`. The allow covers `sum_with_bound`, which only this module's own
// tests reach, and is scoped per module so a genuinely unused item elsewhere
// still surfaces.
#![allow(dead_code)]

use rayon::prelude::*;

/// Values per block, and part of the contract above.
///
/// Chosen to match the device's per-block width so the two folds have the same
/// depth for the same input length, which is the property the curvature bound's
/// derivation actually depends on.
///
/// IT IS ALSO A PARALLELISM PARAMETER ON THE DEVICE, which an earlier note here
/// denied. Each level is an ELEMENT dispatch whose element is a block and whose
/// body walks the block's `BLOCK` values serially, so the level's thread count
/// is `values / BLOCK`: at 256 a fold over a 78,000-vertex scene's `3 * n`
/// launches 915 threads, which does not fill a GPU, and the first level reads
/// every value. Measured on ten frames of `drape`, the three fold entries were
/// 53 percent of GPU time against the SpMV's 10.7.
///
/// 64 WOULD COST NO LEVEL AND WAS TRIED: depth is `ceil(log_BLOCK(values))`, so
/// over the same input 256 and 64 both give three levels while 64 gives the
/// FIRST level four times the threads. **It is not takeable as a one-constant
/// change**, and the test that refuses it is right: this constant is SHARED with
/// the host fold, whose own shape is two levels whatever the length, so
/// `the_device_fold_agrees_with_the_host_fold_bit_for_bit` holds only while the
/// device also finishes in two, which is `BLOCK * BLOCK` values. Narrowing the
/// width narrows that agreement from 65,536 values to 4,096.
///
/// SO THE WIDTH IS NOT THE FIX. Parallelism and level count trade against each
/// other for as long as ONE THREAD OWNS A BLOCK, and no width wins both. The
/// shape that does is a threadgroup reducing cooperatively, which is
/// `block_reduce`, which is a barrier, so the entry would have to carry a
/// cooperative body beside a serial twin.
use ppf_cts_compute::{
    AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, Handle, ReadbackBuffer,
};
use super::kernels::{
    ReduceMaxLeafArgs, ReduceMinLeafArgs, ReduceMinU32LeafArgs, ReduceSumU32LeafArgs,
    ReduceSumWideMergeArgs,
};

pub const BLOCK: usize = 256;

/// Lanes per block in the fold's shape, mirroring `driver::pcg`'s `FOLD_LANES`.
///
/// IT IS PART OF THE SHAPE, NOT A DEVICE DETAIL. A block is folded by this many
/// lanes taking CONTIGUOUS runs, each run summed left to right and the run
/// totals combined in ascending lane order. That is what
/// `vec_block_sum_cooperative` does with `compute::block_sum` on the host arm,
/// and summing a block flat instead is a DIFFERENT expression: fp32 addition is
/// not associative, so run-then-combine is not left-to-right over the block.
const LANES: usize = 64;

/// One block, folded in the shape the cooperative kernel folds it in.
fn block_total(chunk: &[f32]) -> f32 {
    let per = chunk.len().div_ceil(LANES.max(1));
    let mut total = 0.0f32;
    for lane in 0..LANES {
        let first = (lane * per).min(chunk.len());
        let last = (first + per).min(chunk.len());
        let mut mine = 0.0f32;
        for value in &chunk[first..last] {
            mine += *value;
        }
        // ASCENDING LANE ORDER, which is `compute::block_sum`'s host arm: the
        // first lane stores and every later one adds.
        total = if lane == 0 { mine } else { total + mine };
    }
    total
}

/// The same block fold over MAGNITUDES, which is `vec_block_sum_abs`'s shape.
fn block_total_abs(chunk: &[f32]) -> f32 {
    let per = chunk.len().div_ceil(LANES.max(1));
    let mut total = 0.0f32;
    for lane in 0..LANES {
        let first = (lane * per).min(chunk.len());
        let last = (first + per).min(chunk.len());
        let mut mine = 0.0f32;
        for value in &chunk[first..last] {
            mine += value.abs();
        }
        total = if lane == 0 { mine } else { total + mine };
    }
    total
}


/// Sum `values` under the fixed shape.
///
/// Deterministic: the same input gives the same bits regardless of the thread
/// count, because the partition into blocks and the order of the block totals
/// are both fixed by the index rather than by the schedule.
pub fn sum(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    // Each block is folded in the shape the cooperative kernel folds it in.
    let mut level: Vec<f32> = values.par_chunks(BLOCK).map(block_total).collect();
    // THE LEVELS RECURSE IN THE SAME SHAPE, which is what the device fold does
    // and what a flat sum of the block totals here did NOT. A second level over
    // 157 block totals is a block like any other: 64 lanes over contiguous runs,
    // combined ascending. Folding it flat instead is a different expression, and
    // `the_device_fold_agrees_with_the_host_fold_bit_for_bit` is what says so.
    while level.len() > 1 {
        level = level.par_chunks(BLOCK).map(block_total).collect();
    }
    level[0]
}

/// Sum of absolute values, under the same shape.
///
/// Kept beside `sum` rather than expressed through it because the PCG curvature
/// guard needs the signed sum and this bound reduced in the SAME shape, which is
/// a property of the pair rather than of either one.
pub fn sum_abs(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    // THE MAGNITUDE IS TAKEN AT THE FIRST LEVEL ONLY, because every level above
    // it folds totals that are already non-negative, so `abs` there is the
    // identity. That is the device fold's shape too.
    let mut level: Vec<f32> =
        values.par_chunks(BLOCK).map(block_total_abs).collect();
    while level.len() > 1 {
        level = level.par_chunks(BLOCK).map(block_total).collect();
    }
    let total = level[0];
    total
}

/// Largest value under the same fixed shape, with `initial` as the identity.
///
/// A maximum is associative AND commutative in fp32 (no rounding is involved),
/// so unlike the sums above its answer cannot depend on the order. The block
/// shape is kept anyway: this backend's contract is that every reduction has
/// one stated shape, and an exception that happens to be safe today is one a
/// later reader has to re-derive. NaN propagates rather than being skipped,
/// because a NaN reaching a max-reduced quantity (`max_u`, `max_dx`, the reach
/// from the origin) means the step already produced one and the driver's own
/// checks must see it.
pub fn max(values: &[f32], initial: f32) -> f32 {
    if values.is_empty() {
        return initial;
    }
    let blocks: Vec<f32> = values
        .par_chunks(BLOCK)
        .map(|chunk| {
            let mut best = initial;
            for value in chunk {
                if value.is_nan() {
                    return f32::NAN;
                }
                if *value > best {
                    best = *value;
                }
            }
            best
        })
        .collect();
    let mut best = initial;
    for block in &blocks {
        if block.is_nan() {
            return f32::NAN;
        }
        if *block > best {
            best = *block;
        }
    }
    best
}

/// Smallest value under the same fixed shape, with `initial` as the identity.
///
/// The mirror of [`max`], and it exists for the line search: a time of impact
/// is min-reduced over the mesh, with `line_search_max_t` as the identity, and
/// the reduction has to be the one this backend states rather than whatever
/// `Iterator::fold` happens to do.
pub fn min(values: &[f32], initial: f32) -> f32 {
    if values.is_empty() {
        return initial;
    }
    let blocks: Vec<f32> = values
        .par_chunks(BLOCK)
        .map(|chunk| {
            let mut best = initial;
            for value in chunk {
                if value.is_nan() {
                    return f32::NAN;
                }
                if *value < best {
                    best = *value;
                }
            }
            best
        })
        .collect();
    let mut best = initial;
    for block in &blocks {
        if block.is_nan() {
            return f32::NAN;
        }
        if *block < best {
            best = *block;
        }
    }
    best
}

/// The signed sum and its round-off bound, reduced together.
///
/// The two must come from one traversal in the same shape: the bound is only
/// meaningful against the sum it was measured beside. Returning them separately
/// from two calls would let a caller pair a sum with a bound from a different
/// shape, which is exactly the mistake this returns a tuple to prevent.
pub fn sum_with_bound(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    // ONE TRAVERSAL, IN THE COOPERATIVE SHAPE, which is what makes the bound
    // meaningful against the sum: `vec_block_sum_pair_cooperative` folds the two
    // over the SAME lane runs, and a bound reduced in a different shape is a
    // bound on a different sum. The magnitude is taken at the first level only,
    // as it is there.
    let mut signed: Vec<f32> = values.par_chunks(BLOCK).map(block_total).collect();
    let mut magnitude: Vec<f32> =
        values.par_chunks(BLOCK).map(block_total_abs).collect();
    while signed.len() > 1 {
        signed = signed.par_chunks(BLOCK).map(block_total).collect();
        magnitude = magnitude.par_chunks(BLOCK).map(block_total).collect();
    }
    (signed[0], magnitude[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cancelling input, which is where a fold's shape shows.
    fn cancelling(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let magnitude = 1.0e6 * ((i % 17) as f32 + 1.0);
                if i % 2 == 0 {
                    magnitude
                } else {
                    -magnitude + 1.0
                }
            })
            .collect()
    }

    #[test]
    fn the_sum_does_not_depend_on_the_thread_count() {
        let values = cancelling(100_000);
        let reference = sum(&values);
        for threads in [1usize, 2, 3, 5, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| sum(&values));
            assert_eq!(
                reference.to_bits(),
                got.to_bits(),
                "the fold moved at {threads} threads: {reference} against {got}. \
                 The shape is a contract, not a scheduling outcome."
            );
        }
        assert!(reference != 0.0, "the fixture canceled to zero, so the test was vacuous");
    }

    #[test]
    fn the_bound_and_its_sum_come_from_one_traversal() {
        let values = cancelling(50_000);
        let (signed, magnitude) = sum_with_bound(&values);
        assert_eq!(signed.to_bits(), sum(&values).to_bits());
        assert_eq!(magnitude.to_bits(), sum_abs(&values).to_bits());
        assert!(
            magnitude > signed.abs(),
            "a cancelling input must have a magnitude sum far above its signed \
             sum, or it does not exercise what the bound is for"
        );
    }

    #[test]
    fn the_fold_is_f32_throughout() {
        // 2^24 + 1 is the first integer f32 cannot represent, so adding 1 to
        // 2^24 in f32 stalls while an f64 partial would not. One block is enough
        // to show it, and one block is the right fixture: a longer input would
        // demonstrate the opposite, because blocked summation is MORE accurate
        // than a single running total and the stall moves.
        let mut values = vec![0.0f32; 2];
        values[0] = (1u32 << 24) as f32;
        values[1] = 1.0;
        let total = sum(&values);
        assert_eq!(
            total.to_bits(),
            ((1u32 << 24) as f32).to_bits(),
            "an f32 partial must stall at 2^24; an f64 one would reach 2^24 + 1, \
             and would disagree with the device in a direction no tolerance was \
             written for"
        );

        // The companion property, so the test cannot pass merely because the
        // sum is broken: the same two values in f64 DO reach 2^24 + 1.
        let widened: f64 = values.iter().map(|v| f64::from(*v)).sum();
        assert_eq!(widened, ((1u64 << 24) + 1) as f64);
    }

    #[test]
    fn the_maximum_carries_its_identity_and_propagates_nan() {
        assert_eq!(max(&[], 3.0), 3.0);
        assert_eq!(max(&[1.0, -2.0, 0.5], 0.0), 1.0);
        assert_eq!(max(&[-1.0, -2.0], 0.0), 0.0, "the identity is a floor");
        assert!(max(&[1.0, f32::NAN, 2.0], 0.0).is_nan());
        // Long enough to cross the block boundary, so the serial fold over the
        // block totals is exercised rather than one block's own loop.
        let mut long = vec![0.25f32; 4 * BLOCK + 7];
        long[3 * BLOCK + 1] = 9.5;
        assert_eq!(max(&long, 0.0), 9.5);
    }

    #[test]
    fn the_minimum_mirrors_the_maximum() {
        assert_eq!(min(&[], 1.0), 1.0);
        assert_eq!(min(&[0.5, 0.25, 2.0], 1.0), 0.25);
        assert_eq!(min(&[2.0, 3.0], 1.0), 1.0, "the identity is a ceiling");
        assert!(min(&[1.0, f32::NAN], 1.0).is_nan());
    }

    #[test]
    fn an_empty_input_sums_to_zero() {
        assert_eq!(sum(&[]).to_bits(), 0.0f32.to_bits());
        assert_eq!(sum_abs(&[]).to_bits(), 0.0f32.to_bits());
    }
}

/// The device fold, which is what the per-step callers use.
///
/// # What this replaces
///
/// The functions above fold a HOST SLICE, so every caller had to download the
/// array first: nine sites did that once per Newton iteration, over the line
/// search's per-primitive time of impact, the step's maximum displacement and
/// reach, and the strain limiter's ratios. The line those crossed is this:
/// moving a SCALAR to the host is transport and the reference does it, while
/// downloading an ARRAY to run arithmetic on it relocates the computation
/// itself.
///
/// # Why only the minimum and the maximum
///
/// Both are associative AND exact on floats: no rounding happens, so every
/// association order gives the same bits and moving the fold cannot move a
/// result. [`sum`] is not like that. Its own comment states the shape its
/// callers depend on, a fixed partition into [`BLOCK`]-wide chunks summed in
/// index order, and the PCG's curvature bound is derived against that depth, so
/// converting it is a separate change with its own argument to make.
///
/// # The result is one float
///
/// Which is the point: the levels run on the device and four bytes come back.
#[derive(Debug, Default)]
pub struct DeviceFold {
    /// One float per block, ping-ponged between levels.
    level: ReadbackBuffer<f32>,
}

impl DeviceFold {
    /// The width one thread walks, and the branching factor of the recursion.
    ///
    /// [`BLOCK`], so a device fold and the host fold beside it have the same
    /// depth for the same length. Nothing depends on that for a minimum or a
    /// maximum, which are exact; it is here so the two stay comparable when a
    /// test checks one against the other.
    const WIDTH: u32 = BLOCK as u32;

    /// The smallest value in `values[..count]`, not below `floor`.
    ///
    /// # Safety
    /// `values` must name at least `count` floats and outlive the call.
    pub unsafe fn min<D: Device>(
        &mut self,
        device: &mut D,
        region: &'static str,
        values: ppf_cts_compute::Handle,
        count: u32,
        floor: f32,
    ) -> Result<f32, Fault> {
        Ok(self.fold(device, region, values, count, true)?.map_or(floor, |v| v.min(floor)))
    }

    /// The largest value in `values[..count]`, not below `floor`.
    ///
    /// # Safety
    /// As [`Self::min`].
    pub unsafe fn max<D: Device>(
        &mut self,
        device: &mut D,
        region: &'static str,
        values: ppf_cts_compute::Handle,
        count: u32,
        floor: f32,
    ) -> Result<f32, Fault> {
        Ok(self.fold(device, region, values, count, false)?.map_or(floor, |v| v.max(floor)))
    }

    /// The levels, then the one float.
    unsafe fn fold<D: Device>(
        &mut self,
        device: &mut D,
        region: &'static str,
        values: ppf_cts_compute::Handle,
        count: u32,
        smallest: bool,
    ) -> Result<Option<f32>, Fault> {
        if count == 0 {
            return Ok(None);
        }
        // TWO SPANS LIVE AT ONCE, the level being written and the one being
        // read, so the buffer holds the first level plus everything below it.
        let first = count.div_ceil(Self::WIDTH).max(1);
        let span = (first + first.div_ceil(Self::WIDTH) + 2) as usize;
        self.level.size(device, span, AllocLabel("reduce.level"))?;

        let mut blocks = first;
        let mut source = values;
        let mut length = count;
        let mut at = 0usize;
        loop {
            let out = self.level.span(at, blocks as usize);
            if smallest {
                let args = ReduceMinLeafArgs {
                    values: source,
                    count: length,
                    block_size: Self::WIDTH,
                    out,
                    blocks,
                    seam_arena_count: 0,
                };
                device.launch(region, &args, blocks)?;
            } else {
                let args = ReduceMaxLeafArgs {
                    values: source,
                    count: length,
                    block_size: Self::WIDTH,
                    out,
                    blocks,
                    seam_arena_count: 0,
                };
                device.launch(region, &args, blocks)?;
            }
            if blocks == 1 {
                self.level.download(device)?;
                return Ok(Some(self.level.host()[at]));
            }
            source = self.level.span(at, blocks as usize);
            length = blocks;
            at += blocks as usize;
            blocks = blocks.div_ceil(Self::WIDTH);
        }
    }
}

/// Exact unsigned reductions that finish on the device: the smallest word of an
/// array, and a total that cannot wrap.
///
/// # Why they encode rather than submit
///
/// Each level of [`DeviceFold`] is its own submit. These ENCODE into a region
/// the caller opens instead: the analytic collider's contact tally adds its
/// ladder to the submit that writes the tally, and the assembly overlap
/// selection puts its two leaf passes and the ladder in one submit. The answer
/// is left in the first words of the fold's own buffer, where one small read
/// collects it after the region ends. Both are exact, the minimum doing no
/// arithmetic and the total carrying into a second word, so the shape of the
/// ladder cannot move either answer.
///
/// # Layout
///
/// `level` holds the result first (one word for a minimum, a low and a high word
/// for a total), then the leaf level, then every merge level but the last, which
/// writes the result. A single-block input still takes one merge, over one
/// element, so the result is always where the read looks.
#[derive(Debug, Default)]
pub struct DeviceWordFold {
    level: Buffer<u32>,
}

impl DeviceWordFold {
    /// The width one thread walks, and the branching factor of the ladder.
    pub const WIDTH: u32 = BLOCK as u32;

    /// The words a ladder over `blocks` leaf outputs of `stride` words each
    /// occupies, the result included.
    fn words_for(blocks: u32, stride: usize) -> usize {
        let mut words = stride * (1 + blocks as usize);
        let mut level = blocks;
        while level > 1 {
            level = level.div_ceil(Self::WIDTH);
            if level > 1 {
                words += stride * level as usize;
            }
        }
        words
    }

    /// Size for a minimum over `blocks` leaf words. Allocation cannot happen
    /// inside a region, so a caller sizes before it opens one.
    pub fn size_min(&mut self, device: &mut impl Device, blocks: u32) -> Result<(), Fault> {
        self.level.size(device, Self::words_for(blocks, 1), AllocLabel("reduce.word_level"))
    }

    /// The `count` leaf words starting at leaf `first`, for the caller's own
    /// leaf pass to write before [`Self::encode_min`] reduces them.
    pub fn leaves(&self, first: u32, count: u32) -> Handle {
        self.level.span(1 + first as usize, count as usize)
    }

    /// Encode the ladder from `blocks` leaf words down to the result word.
    ///
    /// # Safety
    /// The leaf words must be written earlier in the same region, and the
    /// buffer sized by [`Self::size_min`] for `blocks`.
    pub unsafe fn encode_min(&self, encoder: &mut dyn Encoder, blocks: u32) -> Result<(), Fault> {
        assert!(blocks > 0, "solver driver: a word minimum over no leaf words has no answer");
        let mut source_at = 1usize;
        let mut length = blocks;
        let mut free_at = 1 + blocks as usize;
        loop {
            let next = length.div_ceil(Self::WIDTH);
            let destination_at = if next == 1 { 0 } else { free_at };
            let args = ReduceMinU32LeafArgs {
                values: self.level.span(source_at, length as usize),
                count: length,
                block_size: Self::WIDTH,
                out: self.level.span(destination_at, next as usize),
                blocks: next,
                seam_arena_count: 0,
            };
            encoder.elements(&args, next)?;
            if next == 1 {
                return Ok(());
            }
            source_at = destination_at;
            free_at = destination_at + next as usize;
            length = next;
        }
    }

    /// The word the last [`Self::encode_min`] settled on.
    pub fn read_min(&self, device: &mut impl Device) -> Result<u32, Fault> {
        self.level.read_one(device, 0)
    }

    /// Size for a total over `count` words.
    pub fn size_sum(&mut self, device: &mut impl Device, count: u32) -> Result<(), Fault> {
        let blocks = count.div_ceil(Self::WIDTH);
        self.level.size(device, Self::words_for(blocks, 2), AllocLabel("reduce.word_level"))
    }

    /// Encode the total of `values[..count]`, leaf and ladder.
    ///
    /// # Safety
    /// `values` must name at least `count` words and outlive the region, and
    /// the buffer must be sized by [`Self::size_sum`] for `count`.
    pub unsafe fn encode_sum(
        &self,
        encoder: &mut dyn Encoder,
        values: Handle,
        count: u32,
    ) -> Result<(), Fault> {
        assert!(count > 0, "solver driver: a word total over no words is encoded by nobody");
        let blocks = count.div_ceil(Self::WIDTH);
        let leaf = ReduceSumU32LeafArgs {
            values,
            count,
            block_size: Self::WIDTH,
            out: self.level.span(2, 2 * blocks as usize),
            blocks,
            seam_arena_count: 0,
        };
        encoder.elements(&leaf, blocks)?;
        let mut source_at = 2usize;
        let mut length = blocks;
        loop {
            let next = length.div_ceil(Self::WIDTH);
            let destination_at = if next == 1 { 0 } else { source_at + 2 * length as usize };
            let merge = ReduceSumWideMergeArgs {
                source: self.level.span(source_at, 2 * length as usize),
                count: length,
                block_size: Self::WIDTH,
                destination: self.level.span(destination_at, 2 * next as usize),
                blocks: next,
                seam_arena_count: 0,
            };
            encoder.elements(&merge, next)?;
            if next == 1 {
                return Ok(());
            }
            source_at = destination_at;
            length = next;
        }
    }

    /// The total the last [`Self::encode_sum`] settled on.
    pub fn read_sum(&self, device: &mut impl Device) -> Result<u64, Fault> {
        let mut pair = [0u32; 2];
        self.level.read(device, 0, &mut pair)?;
        Ok(u64::from(pair[0]) | (u64::from(pair[1]) << 32))
    }
}

#[cfg(test)]
mod word_fold_tests {
    use super::*;
    use crate::driver::launch::host_device;

    /// A reproducible spread of words, with no dependency.
    fn word(i: u32) -> u32 {
        i.wrapping_mul(2_654_435_761).rotate_left(7)
    }

    #[test]
    fn the_word_minimum_matches_the_host_minimum_at_every_ladder_depth() {
        let mut device = host_device();
        let mut fold = DeviceWordFold::default();
        // One leaf word takes the single one-element merge, 257 takes two
        // rungs, 65_537 takes three, and the sizes shrink between cases so a
        // reused level buffer is exercised too.
        for blocks in [1u32, 2, 256, 257, 65_537, 65_536, 3] {
            let words: Vec<u32> = (0..blocks).map(word).collect();
            fold.size_min(&mut device, blocks).expect("the level sizes");
            fold.level.write(&mut device, 1, &words).expect("the leaf words upload");
            let ladder = &fold;
            device
                .run("test.word_min", |encoder| {
                    // Safety: the level outlives the region.
                    unsafe { ladder.encode_min(encoder, blocks) }
                })
                .expect("the ladder runs");
            assert_eq!(
                fold.read_min(&mut device).expect("the result reads back"),
                *words.iter().min().expect("the case is not empty"),
                "blocks={blocks}"
            );
        }
    }

    #[test]
    fn the_word_total_carries_past_one_word_and_matches_a_u64_sum() {
        let mut device = host_device();
        let mut fold = DeviceWordFold::default();
        let cases: Vec<Vec<u32>> = vec![
            vec![0],
            // 2^33 exactly, from a carry inside one leaf block.
            vec![u32::MAX, u32::MAX, 2],
            // A carry in the leaf and again in the merge.
            vec![u32::MAX; 257],
            (0..70_000).map(word).collect(),
            vec![u32::MAX; 65_537],
            vec![1, 2, 3],
        ];
        for values in cases {
            let count = values.len() as u32;
            let mut source = Buffer::<u32>::none();
            source
                .size(&mut device, values.len(), AllocLabel("test.word_sum.source"))
                .expect("the source sizes");
            source.write(&mut device, 0, &values).expect("the source uploads");
            fold.size_sum(&mut device, count).expect("the level sizes");
            let ladder = &fold;
            let handle = source.handle();
            device
                .run("test.word_sum", |encoder| {
                    // Safety: the source and the level outlive the region.
                    unsafe { ladder.encode_sum(encoder, handle, count) }
                })
                .expect("the ladder runs");
            let want: u64 = values.iter().map(|&value| u64::from(value)).sum();
            assert_eq!(
                fold.read_sum(&mut device).expect("the total reads back"),
                want,
                "count={count}"
            );
        }
    }

    #[test]
    #[ignore = "release timing experiment"]
    fn benchmark_contact_count_total() {
        let mut device = host_device();
        let mut fold = DeviceWordFold::default();
        for count in [16_384usize, 262_144, 1_048_576] {
            let values: Vec<u32> = (0..count as u32).map(|i| word(i) % 3).collect();
            let want: u64 = values.iter().map(|&value| u64::from(value)).sum();
            let mut mirror = ReadbackBuffer::<u32>::default();
            mirror.size(&mut device, count, AllocLabel("test.count.mirror")).expect("sizes");
            mirror.seed(&mut device, &values).expect("seeds");
            fold.size_sum(&mut device, count as u32).expect("sizes");
            let mut samples = [Vec::new(), Vec::new()];
            for repetition in 0..10 {
                for mode in [repetition % 2, 1 - repetition % 2] {
                    let start = std::time::Instant::now();
                    let total = if mode == 0 {
                        let _ = mirror.handle();
                        mirror.download(&mut device).expect("downloads");
                        mirror.host().iter().map(|&value| u64::from(value)).sum()
                    } else {
                        let ladder = &fold;
                        let handle = mirror.handle();
                        device
                            .run("test.count.total", |encoder| {
                                // Safety: the mirror and the level outlive the region.
                                unsafe { ladder.encode_sum(encoder, handle, count as u32) }
                            })
                            .expect("runs");
                        fold.read_sum(&mut device).expect("reads")
                    };
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(total, want);
                    if repetition > 1 {
                        samples[mode].push(elapsed);
                    }
                }
            }
            for (mode, times) in samples.iter_mut().enumerate() {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "count-total count={count} path={} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
                    if mode == 0 { "host" } else { "device" },
                    times[4],
                    times[0],
                    times[7]
                );
            }
        }
    }
}
