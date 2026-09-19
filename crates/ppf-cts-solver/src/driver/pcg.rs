// File: crates/ppf-cts-solver/src/driver/pcg.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Preconditioned conjugate gradient: the iteration control.
//!
//! Every value comes from a shared body: the operator apply through
//! `spmv.kernel.cpp`, the preconditioner through `block_jacobi.kernel.cpp`, the
//! vector updates through `vec_ops.kernel.cpp`. What lives here is the loop, the
//! convergence test, the guards and the sequencing, none of which computes a
//! value that any other backend also computes.
//!
//! # Two rules inherited whole, because each has shipped as a defect once
//!
//! **The relative-residual denominator is the SEEDED initial residual
//! `||b - A x0||_1`, never `||b||_1`.** The solver never enters with a zero `x`:
//! the driver pre-loads every prescribed (Dirichlet) row of `dx` with its exact
//! correction before the solve. Those rows carry the pin-barrier stiffness, so
//! with any kinematically driven collider they dominate `||b||_1` by orders of
//! magnitude while saying nothing about the dynamic degrees of freedom. Switching
//! to `||b||_1` makes the tolerance exceed the whole dynamic residual, so PCG
//! "converges" in a few iterations and the stiff membrane stretch modes never
//! resolve. It shipped once and was reverted; it must not come back.
//!
//! **The curvature guard fires on a SIGN, and that sign is only real above the
//! round-off of its own sum.** `p^T A p` sums signed fp32 contributions whose
//! magnitudes can exceed the curvature by more than fp32 epsilon, so a small
//! negative value is noise rather than an indefinite operator. A value negative
//! BEYOND the bound aborts; within it, the curvature is unresolvable and the
//! iteration truncates and returns its iterate. It is a round-off bound, not a
//! loosened guard: a genuine assembly defect puts the Rayleigh quotient at order
//! one, six orders above it.
//!
//! # Where the bound comes from, and which scale it is measured over
//!
//! The verdict itself is `pcg_alpha` (`src/kernels/solver/pcg.kernel.cpp`), one
//! shared body every path reaches, so the classification into "proceed",
//! "truncate" and "not positive definite" is not spelled twice. What IS the
//! caller's business is the SCALE handed to `pcg_curvature_noise`, and there
//! are exactly two, chosen by whether the operator apply produced a
//! `sum|contribution|` beside its dot:
//!
//! - a FUSED apply, which accumulates `sum|contribution|` over the very same
//!   signed terms that form `p^T A p`, passes that sum. The device-resident
//!   loop below takes this one, through [`slot::ABSDOT`].
//! - an apply that produces no such sum passes the Cauchy-Schwarz surrogate
//!   `|p|_2 |Ap|_2 >= sum_i |p_i . (Ap)_i|`, formed from three plain inner
//!   products. The locked and PDRD-reduced solves take this one, through
//!   [`DotTerms::norm_product`].
//!
//! The surrogate is the LOOSER of the two, and that is why it is admissible
//! where the fused sum is unavailable: it errs toward truncating rather than
//! aborting.
//!
//! **THE PER-ROW MAGNITUDE BESIDE THE DOT IS NOT A THIRD OPTION.**
//! `pcg_dot_terms` sums a row's three components BEFORE taking the absolute
//! value, so `sum_i |p_i . (Ap)_i|` has already spent every cancellation
//! within a row and is never larger than either quantity above. Reducing it
//! and calling it `absolute_dot` makes the bound TIGHTER than the two the
//! shared verdict is written against, which moves marginal iterates from
//! truncating quietly to aborting loudly. That is the wrong direction and it is
//! not a conservative choice: the whole `pAp<=0` triage keys on the distinction
//! between an unresolvable curvature, which truncates Steihaug-style and is not
//! an error, and a genuine assembly defect, which clears the bound by six
//! orders and must abort. Narrowing the bound erases exactly the first
//! category. The magnitude array stays because `pcg_dot_terms` is a shared body
//! producing both numbers; nothing here judges a curvature by it.

// The solves are called from `step.rs`. The allow is for what no dispatch path
// reaches, including `build_block_jacobi` and `block_jacobi_failure`, which
// only this module's own tests call, and `pcg_beta`, `encode_combine_in_place`
// and `DotTerms::dot_with_bound`, which nothing calls.
#![allow(dead_code)]

use ppf_cts_compute::{AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, ReadbackBuffer};
use super::kernels::{
    Mat3MulArgs, PcgAlphaTermsArgs, PcgBetaResidentArgs,
    PcgBetaTermsArgs, PcgDotTermsArgs, VecAddScaledArgs, VecAddScaledIndirectArgs,
    PcgFoldAlphaArgs, PcgFoldBetaArgs, PcgRigidGroupL1Args,
    PcgUpdateRowFoldedArgs,
    VecBlockSumAbsCooperativeArgs, VecBlockSumCooperativeArgs,
    VecBlockSumDualCooperativeArgs, VecBlockSumPairCooperativeArgs,
    VecCombineArgs,
    VecCombineIndirectArgs,
};
// Named only by the test that drives the resident alpha step on its own.
#[cfg(test)]
use super::kernels::PcgAlphaResidentArgs;
use super::operator::Operator;
use super::reduce;

/// The `cause` codes `pcg_alpha` and `pcg_beta` classify their scalar
/// with, mirroring the anonymous enum in `src/kernels/solver/pcg.kernel.cpp`. Named
/// here rather than compared as bare integers so a reordering there is a
/// mismatch a reader can see.
const PCG_BREAK_NONE: i32 = 0;
const PCG_BREAK_PAP: i32 = 1;
const PCG_BREAK_RZ: i32 = 2;
const PCG_BREAK_NOISE: i32 = 3;

extern "C" {
    // THE ONE SHIM THIS MODULE STILL NAMES DIRECTLY, and it is named here so the
    // exception is visible rather than buried.
    //
    // `block_jacobi_invert` is a PER-BLOCK helper with a return value, called
    // in a host loop that stops at the FIRST refused row so the report can name
    // it. The seam dispatches over a range and returns no value, so putting this
    // behind it needs a `_range` shim plus a first-failure reduction over the
    // rows, which is a change to the C++ side rather than to the driver. Until
    // that lands this is a direct call, and it is the only one left in the linear
    // solve.
    fn block_jacobi_invert_abi(
        block: *const f32,
        inverse: *mut f32,
        lambda_max_out: *mut f32,
    ) -> i32;
}

/// `destination += scale * source`, elementwise, through the shared body.
///
/// The body is `vec_add_scaled` in `primitives/vec_ops.kernel.cpp`, which CUDA
/// and Metal compile from the same bytes. Writing `destination[i] += scale *
/// source[i]` here in Rust instead would be a second implementation of it, and
/// not a harmless one: the host renderings are compiled with
/// `-ffp-contract=fast` so their arithmetic matches a contracting device, and
/// Rust never contracts, so the two spellings can differ in the last bit.
///
/// # Safety
/// Both slices must outlive the dispatch this appends.
unsafe fn encode_add_scaled(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    scale: f32,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecAddScaledArgs {
        source,
        destination,
        scale,
        // The in-kernel guard bound, which a generated entry point carries in
        // its record because Metal never faults on an out-of-bounds access and
        // every backend rounds a launch up to whole threadgroups. It is the
        // same `n` the extent below takes, and `Encoder::elements` faults if
        // the two ever differ.
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `destination = a * source_a + b * source_b`, elementwise, through the shared
/// body `vec_combine`.
///
/// # Safety
/// The three references must each name `n` floats that outlive the dispatch.
/// `source_b` MAY alias `destination`, which is the `p = z + beta p` case: the
/// body reads element `i` before writing element `i` and every chunk is a
/// disjoint span of indices. `source_a` must not alias `destination`.
unsafe fn encode_combine_raw(
    encoder: &mut dyn Encoder,
    source_a: ppf_cts_compute::Handle,
    source_b: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    b: f32,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecCombineArgs {
        source_a,
        source_b,
        destination,
        a,
        b,
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `destination = a * source_a + b * source_b` for three distinct buffers.
fn combine<D: Device>(
    device: &mut D,
    source_a: ppf_cts_compute::Handle,
    source_b: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    b: f32,
    n: usize,
) -> Result<(), Fault> {
    let refs = (source_a, source_b, destination);
    device.run("pcg.combine", |encoder| {
        // Safety: three distinct slices, each `n` long, so nothing aliases, and
        // all three are borrowed for the whole call.
        unsafe { encode_combine_raw(encoder, refs.0, refs.1, refs.2, a, b, n) }
    })?;
    Ok(())
}

/// Values per block in the DEVICE fold, and the same contract
/// [`super::reduce`] states for the host folds that remain.
///
/// It is stated in ONE place and handed to the kernel as an argument, because
/// two statements of it are two fold shapes that can disagree, and the shape
/// decides the last bits of every scalar this recurrence carries.
const FOLD_WIDTH: usize = super::reduce::BLOCK;

/// Threads per group in the COOPERATIVE fold, and the reason it is not
/// `FOLD_WIDTH`.
///
/// `FOLD_WIDTH` is how many VALUES a block folds; this is how many THREADS share
/// that block. The two were one number while a block was one thread's serial
/// walk, which is exactly what made the first level of a fold over
/// `3 * vertices` launch 915 threads and cost 53 percent of GPU time.
/// Separating them is the fix: the block keeps its width, so the LEVEL COUNT
/// and the values each level folds are unchanged, and 64 lanes share it, so the
/// level's thread count is 64 times what it was.
const FOLD_LANES: u32 = 64;

/// Where each scalar of the recurrence lives in the workspace's scalar buffer.
///
/// ONE BUFFER RATHER THAN TEN. Every scalar of the recurrence is carved out of
/// a single allocation, so a scheduled check reads a window of them with one
/// copy. A separate allocation per scalar would be ten handles to bind and ten
/// transfers to read.
mod slot {
    /// `r . z` from the PREVIOUS iteration, which alpha divides and the beta
    /// step rolls forward.
    pub const RZ0: usize = 0;
    /// `r . z` at the current iterate.
    pub const RZ1: usize = 1;
    pub const P_AP: usize = 2;
    pub const ALPHA: usize = 3;
    pub const BETA: usize = 4;
    /// `||r||_1`, the only scalar the host reads every iteration.
    pub const ERR: usize = 5;
    /// The scale the curvature bound is taken from: the sum, over the whole
    /// vector, of the MAGNITUDES of the signed contributions that add up to
    /// `p^T A p`. The operator forms it per row while it multiplies and the
    /// caller folds it here, so the bound and the curvature it bounds come out
    /// of one pass over the same terms.
    pub const ABSDOT: usize = 6;
    /// The offending scalar LATCHED at the iterate that broke, written by the
    /// resident alpha and beta bodies and never overwritten afterwards. It is a
    /// curvature for `PCG_BREAK_PAP` and an `r . z` for `PCG_BREAK_RZ`.
    ///
    /// It exists because the host samples on a STRIDE: by the time a scheduled
    /// check runs, `P_AP` holds a later iterate's value, and reporting that
    /// describes a different iterate entirely.
    pub const BREAK_VALUE: usize = 7;
    /// The round-off floors the two verdict bodies report beside their value.
    /// Nothing reads them today; they are bound because the bodies yield them
    /// and a gathered output has to name an allocation.
    pub const ALPHA_NOISE: usize = 8;
    pub const BETA_NOISE: usize = 9;
    /// NON-ZERO ONCE EITHER GUARD HAS LATCHED, and nothing more than that.
    ///
    /// The cause and the iteration it fired at are INTS and live in their own
    /// allocation, so reading them cost a second device-to-host copy on every
    /// batch, healthy or not, and they were measured at 939 of 2,404 such
    /// copies on `drape`. This float rides the scalar buffer the probe already
    /// downloads, so the int channel is read only when it holds something.
    ///
    /// IT IS NOT THE CAUSE, AND MUST NOT BECOME IT. Storing the enum here would
    /// buy the same copy and put a typed channel into a float, which is what
    /// the separate allocation exists to avoid; this answers only whether there
    /// is anything to read.
    pub const BREAK_FIRED: usize = 10;
    /// Where a ONE-OFF device fold lands when the host reads its result
    /// immediately. `DotTerms::dot` folds the per-row products here rather
    /// than downloading the whole vector and summing it on the host, which
    /// would relocate the reduction rather than transport its result. It is
    /// reused by every such fold because each is read before the next is
    /// encoded.
    pub const FOLD_SCRATCH: usize = 11;
    pub const COUNT: usize = 12;
}

/// Where each verdict's cause lives, which is a SECOND buffer because a cause
/// is an `int`.
///
/// Packing the cause into the float block and reinterpreting the bytes would
/// save a read; this driver keeps the types honest and pays one more small read
/// per iteration for it.
mod cause_slot {
    pub const ALPHA: usize = 0;
    pub const BETA: usize = 1;
    /// The STICKY breakdown cause, written by whichever of the two bodies
    /// breaks FIRST and never cleared inside a solve. Without it a strided
    /// read would miss a breakdown that
    /// a later healthy iteration overwrote, and the solve would continue on a
    /// direction the guard had already rejected.
    pub const BREAK_CAUSE: usize = 2;
    /// The iteration index the latch fired at, so the report names the
    /// offending iterate rather than the check that noticed it.
    pub const BREAK_ITER: usize = 3;
    /// The RUNNING ITERATION COUNT, advanced on the device by the alpha pass.
    ///
    /// It lives on the device rather than in a host variable, and the reason is
    /// not bookkeeping: an
    /// iteration whose arguments differ between two turns of the loop cannot be
    /// RECORDED ONCE AND REPLAYED, and a by-value index was the only argument
    /// that differed. It lives in this int block rather than the float one
    /// beside `BREAK_ITER`, which reads it.
    pub const ITERATION: usize = 4;
    pub const COUNT: usize = 5;
}

/// Floats the fold needs for its intermediate levels, folding `count` values.
///
/// The LAST level writes into the caller's scalar slot rather than into the
/// scratch, so a fold short enough to finish in one level needs none of this.
fn fold_scratch_len(count: usize) -> usize {
    let mut total = 0usize;
    let mut remaining = count;
    loop {
        let blocks = remaining.div_ceil(FOLD_WIDTH);
        if blocks <= 1 {
            return total;
        }
        total += blocks;
        remaining = blocks;
    }
}

/// Fold `source[..count]` to the single float `out` names, entirely on the
/// device.
///
/// THE SHAPE, WHICH IS A CONTRACT. Each level cuts its input into blocks of
/// [`FOLD_WIDTH`], sums each block serially in ascending index order, and
/// writes the block totals; the next level folds those the same way, until one
/// element remains. Kernel completion is the barrier between levels, which is
/// what lets a reduction exist at all on a seam with no cooperative launch
/// shape: `crates/ppf-cts-compute/src/device.rs` records that a reduction
/// reaches the CPU backend only as a body written without a threadgroup
/// barrier.
///
/// IT AGREES WITH [`super::reduce::sum`] BIT FOR BIT UP TO
/// `FOLD_WIDTH * FOLD_WIDTH` VALUES: blocks summed in index order, then the
/// block totals summed serially in index order. Past that length this takes
/// another level where a single serial pass would keep accumulating, which is a
/// MORE accurate association rather than a different rule, and it is the
/// association a strided device fold takes.
///
/// `absolute` applies to the FIRST level only, and that is not a convenience:
/// every level above it folds block totals that are already non-negative, so
/// taking magnitudes again would be a second absolute value over the same
/// numbers.
///
/// # Safety
/// `source`, `out` and the scratch must outlive the dispatches this appends,
/// and `out` must name exactly one float.
unsafe fn encode_fold(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    count: usize,
    scratch: &Buffer<f32>,
    out: ppf_cts_compute::Handle,
    absolute: bool,
) -> Result<(), Fault> {
    debug_assert!(count > 0, "a fold of nothing has no level to dispatch");
    let mut input = source;
    let mut remaining = count;
    let mut absolute = absolute;
    let mut cursor = 0usize;
    loop {
        let blocks = remaining.div_ceil(FOLD_WIDTH);
        let destination = if blocks == 1 {
            out
        } else {
            scratch.span(cursor, blocks)
        };
        if absolute {
            let args = VecBlockSumAbsCooperativeArgs {
                source: input,
                length: remaining as u32,
                width: FOLD_WIDTH as u32,
                total: destination,
                count: blocks as u32,
                seam_arena_count: 0,
            };
            encoder.groups(
                &args,
                blocks as u32,
                FOLD_LANES,
            )?;
        } else {
            let args = VecBlockSumCooperativeArgs {
                source: input,
                length: remaining as u32,
                width: FOLD_WIDTH as u32,
                total: destination,
                count: blocks as u32,
                seam_arena_count: 0,
            };
            encoder.groups(
                &args,
                blocks as u32,
                FOLD_LANES,
            )?;
        }
        if blocks == 1 {
            return Ok(());
        }
        cursor += blocks;
        input = destination;
        remaining = blocks;
        absolute = false;
    }
}

/// Fold TWO arrays of the same length to two scalars, in one dispatch chain.
///
/// EVERY LEVEL IS ONE DISPATCH INSTEAD OF TWO, and nothing else changes: each
/// output is the block sum of its own array over the same [`FOLD_WIDTH`] and in
/// the same ascending order [`encode_fold`] uses, so both scalars are
/// bit-identical to what two separate chains would produce and the curvature
/// bound stated against that shape still holds. It changes the launch count and
/// nothing else.
///
/// THE TWO ARRAYS NEED SEPARATE SCRATCH, because a level reads both and writes
/// both, so one region cannot serve. `second_base` is where the second array's
/// levels begin, and [`State::size_for`] sizes the buffer for both.
///
/// # Safety
/// Every buffer must outlive the dispatches this appends, and both outputs must
/// name exactly one float.
/// Fold TWO arrays of one length to two scalars in ONE dispatch.
///
/// [`encode_fold_pair`] walks a fixed-width tree because it starts from a
/// FULL-LENGTH array and no single group should own `rows` elements. The
/// partials the folded apply writes are one per group, so the tree has nothing
/// left to do: a single group whose window is the whole array gives each lane a
/// contiguous run of it, and the two outputs are folded by that one group.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends, and both outputs must
/// name exactly one float.
unsafe fn encode_fold_pair_once(
    encoder: &mut dyn Encoder,
    first_source: ppf_cts_compute::Handle,
    second_source: ppf_cts_compute::Handle,
    count: usize,
    first_out: ppf_cts_compute::Handle,
    second_out: ppf_cts_compute::Handle,
) -> Result<(), Fault> {
    debug_assert!(count > 0, "a fold of nothing has no group to dispatch");
    let args = VecBlockSumPairCooperativeArgs {
        first_source,
        second_source,
        length: count as u32,
        // THE WINDOW IS THE WHOLE ARRAY, which is what makes this one group:
        // `vec_lane_run` splits `width` across the lanes, so a width equal to
        // the length leaves nothing for a second group to own.
        width: count as u32,
        first_total: first_out,
        second_total: second_out,
        count: 1,
        seam_arena_count: 0,
    };
    encoder.groups(&args, 1, FOLD_LANES)
}

unsafe fn encode_fold_pair(
    encoder: &mut dyn Encoder,
    first_source: ppf_cts_compute::Handle,
    second_source: ppf_cts_compute::Handle,
    count: usize,
    scratch: &Buffer<f32>,
    second_base: usize,
    first_out: ppf_cts_compute::Handle,
    second_out: ppf_cts_compute::Handle,
) -> Result<(), Fault> {
    debug_assert!(count > 0, "a fold of nothing has no level to dispatch");
    let mut first_in = first_source;
    let mut second_in = second_source;
    let mut remaining = count;
    let mut cursor = 0usize;
    loop {
        let blocks = remaining.div_ceil(FOLD_WIDTH);
        let (first_dst, second_dst) = if blocks == 1 {
            (first_out, second_out)
        } else {
            (
                scratch.span(cursor, blocks),
                scratch.span(second_base + cursor, blocks),
            )
        };
        let args = VecBlockSumPairCooperativeArgs {
            first_source: first_in,
            second_source: second_in,
            length: remaining as u32,
            width: FOLD_WIDTH as u32,
            first_total: first_dst,
            second_total: second_dst,
            count: blocks as u32,
            seam_arena_count: 0,
        };
        encoder.groups(
            &args,
            blocks as u32,
            FOLD_LANES,
        )?;
        if blocks == 1 {
            return Ok(());
        }
        cursor += blocks;
        first_in = first_dst;
        second_in = second_dst;
        remaining = blocks;
    }
}

/// Fold TWO arrays of DIFFERENT lengths to two scalars, in one dispatch chain.
///
/// [`encode_fold_pair`] shares one length and one block count between its two
/// arrays, which is the honest restriction for it: the curvature and its bound
/// are two views of one pass and are folded in lockstep. THE TWO CHAINS HERE
/// ARE NOT IN LOCKSTEP. `||r||_1` runs over `3 * rows` floats and `r . z` over
/// `rows` of them, so their levels have different extents and one reaches a
/// single element before the other does.
///
/// EACH CHAIN KEEPS ITS OWN CURSOR, EXTENT AND DEPTH, and the dispatch covers
/// the larger of the two block counts. A chain that has reached its output
/// passes a count of zero and is left alone by every level above it, so the
/// shorter chain is finished rather than padded and its scalar is written
/// exactly once. Every scalar is bit-identical to the two chains this merges:
/// each source is summed over the same [`FOLD_WIDTH`] in the same ascending
/// order at the same level.
///
/// THE MAGNITUDE LEVEL IS NOT THIS FUNCTION'S. Only the first level of an L1
/// norm takes absolute values, so the caller dispatches that one alone and
/// hands the result here as an ordinary array; see the call site.
///
/// # Safety
/// Every buffer must outlive the dispatches this appends, and both outputs must
/// name exactly one float. A finished chain names its own output as its source,
/// which is a real allocation and never [`ppf_cts_compute::Handle::NONE`]: a
/// generated entry resolves every handle it is given before the body runs.
unsafe fn encode_fold_dual(
    encoder: &mut dyn Encoder,
    first_source: ppf_cts_compute::Handle,
    first_count: usize,
    second_source: ppf_cts_compute::Handle,
    second_count: usize,
    scratch: &Buffer<f32>,
    first_base: usize,
    second_base: usize,
    first_out: ppf_cts_compute::Handle,
    second_out: ppf_cts_compute::Handle,
) -> Result<(), Fault> {
    debug_assert!(
        first_count > 0 && second_count > 0,
        "a fold of nothing has no level to dispatch"
    );
    let mut first_in = first_source;
    let mut second_in = second_source;
    let mut first_remaining = first_count;
    let mut second_remaining = second_count;
    // EACH CHAIN'S CURSOR STARTS PAST WHAT IS ALREADY WRITTEN, which is what
    // keeps a level from writing the scratch it is reading: a level reads
    // `[256 i, 256 i + 256)` and writes `i`, so an output overlapping its own
    // input is a race between threads rather than an aliasing question.
    let mut first_cursor = first_base;
    let mut second_cursor = second_base;
    let mut first_live = true;
    let mut second_live = true;
    loop {
        let first_blocks = if first_live {
            first_remaining.div_ceil(FOLD_WIDTH)
        } else {
            0
        };
        let second_blocks = if second_live {
            second_remaining.div_ceil(FOLD_WIDTH)
        } else {
            0
        };
        if first_blocks == 0 && second_blocks == 0 {
            return Ok(());
        }
        // A FINISHED CHAIN STILL NAMES REAL MEMORY. Its count is zero so no
        // thread writes for it, but the entry resolves both handles before the
        // body runs, so it points at its own output and reads one float it
        // already holds.
        let (first_src, first_dst, first_len) = if !first_live {
            (first_out, first_out, 1usize)
        } else if first_blocks == 1 {
            (first_in, first_out, first_remaining)
        } else {
            (
                first_in,
                scratch.span(first_cursor, first_blocks),
                first_remaining,
            )
        };
        let (second_src, second_dst, second_len) = if !second_live {
            (second_out, second_out, 1usize)
        } else if second_blocks == 1 {
            (second_in, second_out, second_remaining)
        } else {
            (
                second_in,
                scratch.span(second_cursor, second_blocks),
                second_remaining,
            )
        };
        let args = VecBlockSumDualCooperativeArgs {
            first_source: first_src,
            first_length: first_len as u32,
            second_source: second_src,
            second_length: second_len as u32,
            width: FOLD_WIDTH as u32,
            first_total: first_dst,
            second_total: second_dst,
            first_count: first_blocks as u32,
            second_count: second_blocks as u32,
            count: first_blocks.max(second_blocks) as u32,
            seam_arena_count: 0,
        };
        encoder.groups(
            &args,
            first_blocks.max(second_blocks) as u32,
            FOLD_LANES,
        )?;
        if first_blocks == 1 {
            first_live = false;
        } else if first_blocks > 1 {
            first_cursor += first_blocks;
            first_in = first_dst;
            first_remaining = first_blocks;
        }
        if second_blocks == 1 {
            second_live = false;
        } else if second_blocks > 1 {
            second_cursor += second_blocks;
            second_in = second_dst;
            second_remaining = second_blocks;
        }
    }
}

/// The FIRST level of an L1 norm, dispatched alone so the levels above it can
/// share a chain with an unrelated fold.
///
/// Returns the array the rest of that chain folds and its length, which is the
/// source itself when one level was the whole fold.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends.
unsafe fn encode_fold_abs_level(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    count: usize,
    scratch: &Buffer<f32>,
    out: ppf_cts_compute::Handle,
) -> Result<(ppf_cts_compute::Handle, usize), Fault> {
    let blocks = count.div_ceil(FOLD_WIDTH);
    let destination = if blocks == 1 { out } else { scratch.span(0, blocks) };
    let args = VecBlockSumAbsCooperativeArgs {
        source,
        length: count as u32,
        width: FOLD_WIDTH as u32,
        total: destination,
        count: blocks as u32,
        seam_arena_count: 0,
    };
    encoder.groups(
        &args,
        blocks as u32,
        FOLD_LANES,
    )?;
    Ok((destination, blocks))
}

/// `destination += sign * (*coefficient) * source`, with the coefficient read
/// from a device scalar rather than carried in the record.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends.
unsafe fn encode_add_scaled_indirect(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    coefficient: ppf_cts_compute::Handle,
    sign: f32,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecAddScaledIndirectArgs {
        source,
        destination,
        coefficient,
        sign,
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `p = z + beta p`, with `beta` read from a device scalar.
///
/// # Safety
/// Every buffer must outlive the dispatch this appends. `source_b` MAY alias
/// `destination`, which is the whole point of this shape: the body reads
/// element `i` before writing element `i` and every chunk is a disjoint span of
/// indices.
unsafe fn encode_combine_indirect(
    encoder: &mut dyn Encoder,
    source_a: ppf_cts_compute::Handle,
    source_b: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    coefficient_b: ppf_cts_compute::Handle,
    n: usize,
) -> Result<(), Fault> {
    if n == 0 {
        return Ok(());
    }
    let args = VecCombineIndirectArgs {
        source_a,
        source_b,
        destination,
        a,
        coefficient_b,
        count: n as u32,
        seam_arena_count: 0,
    };
    encoder.elements(&args, n as u32)
}

/// `destination = a * source + b * destination`, the aliasing form.
///
/// # Safety
/// Both slices must outlive the dispatch this appends.
unsafe fn encode_combine_in_place(
    encoder: &mut dyn Encoder,
    source: ppf_cts_compute::Handle,
    destination: ppf_cts_compute::Handle,
    a: f32,
    b: f32,
    n: usize,
) -> Result<(), Fault> {
    encode_combine_raw(encoder, source, destination, destination, a, b, n)
}

/// Why a solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The relative residual reached the tolerance.
    Converged,
    /// The iteration limit was reached first.
    MaxIterations,
    /// The curvature was unresolvable above the round-off of its own sum, so the
    /// iteration truncated and returned its iterate. Steihaug-style, and NOT an
    /// error: the direction so far is still a descent direction.
    CurvatureTruncated,
    /// The curvature was negative BEYOND the round-off bound. The assembled
    /// Newton Hessian is not SPD, which means a real assembly defect upstream:
    /// a per-element Hessian missing its PSD projection, a sign error, or a
    /// dropped off-diagonal block.
    ///
    /// SEPARATE FROM THE ONE BELOW BECAUSE THE REMEDIES DIFFER. This one sends
    /// a reader to the assembly; the other sends them to the preconditioner.
    /// One message for both would send every reader to the wrong half of the
    /// solver half the time.
    IndefiniteMatrix,
    /// `r^T M^-1 r` was non-positive. The PRECONDITIONER is not SPD.
    ///
    /// Every block-Jacobi diagonal block is inverted through a floored
    /// symmetric eigendecomposition, so each per-vertex term is positive by
    /// construction and their sum cannot be negative in exact arithmetic: a
    /// non-positive value here means a block is NaN or infinite, or the
    /// residual itself is not finite.
    NonSpdPreconditioner,
}

/// What a solve reports back.
#[derive(Debug, Clone, Copy)]
pub struct Report {
    pub outcome: Outcome,
    pub iterations: u32,
    /// Relative to the SEEDED initial residual, per the rule above.
    pub relative_residual: f32,
}

/// Build the block-Jacobi preconditioner into a caller-owned buffer.
///
/// Returns `false` when a block is unusable, which the caller must treat as a
/// fatal assembly defect rather than continuing: a zero block would not trip any
/// PCG guard and would silently freeze that vertex at its seed.
///
/// THE BUFFER IS THE CALLER'S BECAUSE THIS RUNS INSIDE A NEWTON ITERATION.
/// Allocating here would put a failure point in the middle of a solve, where
/// there is no good answer; the driver sizes it once at `initialize()`, where a
/// failure can be reported before a frame is written.
pub fn build_block_jacobi(diagonal: &[f32], inverse: &mut [f32], rows: u32) -> bool {
    block_jacobi_failure(diagonal, inverse, rows).is_none()
}

/// As [`build_block_jacobi`], naming the row it refused and what that row held.
///
/// THE ROW IS WHAT MAKES THE FAILURE ACTIONABLE. A refused block is an assembly
/// defect upstream rather than a tolerance, and the assembly that produced it is
/// identified by which vertex the row belongs to and what the nine floats are:
/// a NaN, an infinity and a negative eigenvalue are three different upstream
/// mistakes and a bare "not positive definite" separates none of them.
pub fn block_jacobi_failure(
    diagonal: &[f32],
    inverse: &mut [f32],
    rows: u32,
) -> Option<(usize, [f32; 9])> {
    assert_eq!(diagonal.len(), 9 * rows as usize, "one 3x3 block per row");
    assert_eq!(inverse.len(), 9 * rows as usize, "one 3x3 block per row");
    for row in 0..rows as usize {
        let mut lambda_max = 0.0f32;
        let ok = unsafe {
            block_jacobi_invert_abi(
                diagonal[9 * row..].as_ptr(),
                inverse[9 * row..].as_mut_ptr(),
                &mut lambda_max,
            )
        };
        if ok == 0 {
            let mut block = [0.0f32; 9];
            block.copy_from_slice(&diagonal[9 * row..9 * row + 9]);
            return Some((row, block));
        }
    }
    None
}

/// The vectors one PCG solve needs, owned by the caller for the same reason the
/// preconditioner's buffer is.
///
/// Sized once for the scene. `resize` here would be an allocation inside a
/// Newton iteration; the driver calls [`Workspace::size_for`] at
/// `initialize()`.
#[derive(Default)]
pub struct Workspace {
    ap: Buffer<f32>,
    /// THE RESIDUAL CARRIES A MIRROR, because the convergence test reads its
    /// L1 norm on the host once per iteration. That host dependency is not new:
    /// the residual was a host slice before and the same reduction ran on it
    /// every pass. What changed is that the transfer is now explicit.
    r: ReadbackBuffer<f32>,
    z: Buffer<f32>,
    p: Buffer<f32>,
    product: ReadbackBuffer<f32>,
    magnitude: ReadbackBuffer<f32>,
    /// Per row, the sum of the MAGNITUDES of that row's signed contributions to
    /// `p^T A p`, which the operator now forms while it multiplies.
    ///
    /// IT CANNOT SHARE `magnitude`. That one carries the per-row absolute
    /// product the dot terms yield, and both are live in the same region: the
    /// apply writes this one and the dot fill writes that one, an instruction
    /// apart.
    row_absolute: Buffer<f32>,
    /// Per row, the SIGNED sum of the same contributions `row_absolute` takes
    /// the magnitudes of, so `p^T A p` and its round-off bound come from ONE
    /// accumulation. Forming the curvature separately as `p . Ap` after the
    /// product would be a different rounding on the one scalar the step length
    /// divides by, and the bound would then be measuring a sum that is not the
    /// one being bounded.
    row_curvature: Buffer<f32>,
    /// The alpha / beta verdict: the scalar and the noise floor beside it.
    ///
    /// A WORKSPACE FIELD RATHER THAN A STACK SCALAR, because these run once per
    /// PCG iteration and a per-call device allocation would be paid every one of
    /// them. `Buffer::size` grows only past capacity, so the sizing below costs
    /// nothing after the first.
    verdict: ReadbackBuffer<f32>,
    /// The body's own classification of that scalar, which is an `i32` and so
    /// cannot share the buffer above.
    verdict_cause: ReadbackBuffer<i32>,
    /// EVERY SCALAR THE RECURRENCE CARRIES, in one allocation, laid out by
    /// [`slot`].
    ///
    /// It is a `ReadbackBuffer` for one field only: `||r||_1` is what the
    /// convergence test reads, and the loop cannot decide to stop without it.
    /// The other nine are written by one dispatch and read by the next without
    /// ever crossing the seam.
    scalar: ReadbackBuffer<f32>,
    /// The two verdicts' causes, in one allocation laid out by [`cause_slot`].
    cause: ReadbackBuffer<i32>,
    /// The fold's intermediate levels.
    ///
    /// A PLAIN `Buffer`: no level of a fold is ever read by the host, only the
    /// last one's destination, which is a slot of `scalar` above.
    fold: Buffer<f32>,
}

impl Workspace {
    /// Size every vector for a scene of `rows` vertices.
    pub fn size_for(&mut self, device: &mut impl Device, rows: u32) -> Result<(), Fault> {
        let n = 3 * rows as usize;
        self.ap.size(device, n, AllocLabel("pcg.ap"))?;
        self.r.size(device, n, AllocLabel("pcg.r"))?;
        self.z.size(device, n, AllocLabel("pcg.z"))?;
        self.p.size(device, n, AllocLabel("pcg.p"))?;
        // THE TWO FOLD OUTPUTS ARE DEVICE ALLOCATIONS, written by the per-row
        // pass and reduced on the host, so they carry a mirror and their sizing
        // needs a backend where the four vectors above do not.
        self.product
            .size(device, rows as usize, AllocLabel("pcg.product"))?;
        self.verdict
            .size(device, 2, AllocLabel("pcg.verdict"))?;
        self.verdict_cause
            .size(device, 1, AllocLabel("pcg.verdict_cause"))?;
        self.magnitude
            .size(device, rows as usize, AllocLabel("pcg.magnitude"))?;
        self.row_absolute
            .size(device, rows as usize, AllocLabel("pcg.row_absolute"))?;
        self.row_curvature
            .size(device, rows as usize, AllocLabel("pcg.row_curvature"))?;
        // THE RECURRENCE'S SCALARS AND THE FOLD THAT PRODUCES THEM. Sized once
        // for the scene like everything else here: a fold level's extent is
        // decided by the vector it reduces, so the deepest chain any call makes
        // is the one over `n` floats and the scratch is sized for that.
        self.scalar
            .size(device, slot::COUNT, AllocLabel("pcg.scalar"))?;
        self.cause
            .size(device, cause_slot::COUNT, AllocLabel("pcg.cause"))?;
        // TWICE THE CHAIN, because `encode_fold_pair` runs two arrays through
        // the levels at once and a level reads both while writing both, so one
        // region cannot serve. The single-array folds beside it use the first
        // half and are unaffected.
        self.fold.size(
            device,
            (2 * fold_scratch_len(n)).max(1),
            AllocLabel("pcg.fold"),
        )?;
        Ok(())
    }
}

/// `z = M^-1 r`, one 3x3 block per row.
///
/// # Safety
/// Every slice must outlive the dispatch this appends.
unsafe fn encode_preconditioner(
    encoder: &mut dyn Encoder,
    inverse: ppf_cts_compute::Handle,
    r: ppf_cts_compute::Handle,
    z: ppf_cts_compute::Handle,
    rows: u32,
) -> Result<(), Fault> {
    if rows == 0 {
        return Ok(());
    }
    // The record is the BODY's, not this pass's: `mat3_mul` takes a 3x3
    // block and a vector and returns the product, and one dispatch of it per
    // row IS the block-diagonal apply. The entry point reads the block at a
    // stride of nine floats and the vector at three, which is the addressing
    // a hand-written range shim would have to spell out.
    let args = Mat3MulArgs {
        matrix: inverse,
        vector: r,
        result: z,
        count: rows,
        seam_arena_count: 0,
    };
    encoder.elements(&args, rows)
}

/// The relative residual, refreshed from the device first.
///
/// `handle()` stales the mirror, so a report built after a dispatch has to read
/// the array again rather than answer out of the previous pass.
fn residual_of<D: Device>(
    device: &mut D,
    r: &mut ReadbackBuffer<f32>,
    err0: f32,
) -> Result<f32, Fault> {
    r.download(device)?;
    Ok(reduce::sum_abs(r.host()) / err0)
}

/// Solve `M x = b` by preconditioned conjugate gradient, with the recurrence
/// on the device.
///
/// `x` enters carrying the caller's initial guess and leaves carrying the
/// iterate. That is not an optimization: the driver seeds prescribed rows with
/// their exact correction, and the denominator rule above depends on the
/// residual being formed against that seed.
///
/// # What crosses the seam, and what does not
///
/// EVERY PER-ITERATION SCALAR STAYS ON THE DEVICE, and an iteration is issued
/// as one unbroken stream of kernels; the host reads a small probe and nothing
/// else. One region per iteration carries the matvec, the three folds the
/// curvature verdict needs, that verdict, the two updates, the residual norm,
/// the preconditioner, `r . z`, the beta verdict with its roll, and the
/// direction update; then the host reads the two scalar blocks, which are ten
/// floats and two ints.
///
/// THE ALTERNATIVE IS FOUR FULL-LENGTH DOWNLOADS AN ITERATION, one of the
/// residual and three of the per-row product array, each folded on the host. A
/// readback of a scalar is TRANSPORT; downloading an array and folding it is the
/// computation itself moved to the host, which rule (1a-0) forbids because it
/// makes this a different solver rather than a slower one.
///
/// # Two things this deliberately does NOT change
///
/// **The termination ORDER is fixed**: the alpha verdict is tested first, then
/// convergence, then the beta verdict. Reordering the three changes which
/// iterate a marginal solve returns, so it is a decision of its own and never a
/// side effect of a change about where arithmetic runs.
///
/// **The iteration is RECORDED ONCE AND REPLAYED**: the loop encodes one
/// iteration and replays it, sampling the residual one time in four while the
/// residual is far from the tolerance and every iteration once within
/// `NEAR_TOL_FACTOR` of it. The batch size and the sampling rate are ONE number,
/// because a backend synchronizes at the end of every submit, so a check the
/// batch does not end at buys nothing.
#[allow(clippy::too_many_arguments)]
pub fn solve<D: Device>(
    device: &mut D,
    op: &Operator,
    inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    work: &mut Workspace,
    // THE PRECONDITIONER, when it is not the block-diagonal one.
    //
    // `None` is block-Jacobi, the default, and stays the fast path: one
    // dispatch of `mat3_mul` per row IS the block-diagonal apply. `Some` is an
    // additive Schwarz sweep, three dispatches over the domains the caller
    // built. Both are pushed onto the same encoder, so an iteration still
    // costs one submission either way.
    mut schwarz: Option<&mut super::schwarz::State>,
    tolerance: f32,
    max_iterations: u32,
) -> Result<Report, Fault> {
    let rows = op.rows();
    let n = 3 * rows as usize;
    assert_eq!(work.ap.len(), n, "the workspace was sized for a different scene");

    let Workspace {
        ap,
        r,
        z,
        p,
        product,
        magnitude,
        row_absolute,
        row_curvature,
        scalar,
        cause,
        fold,
        ..
    } = work;
    let mut terms = DotTerms {
        product,
        magnitude,
    };
    // A SCENE WITH NO ROWS HAS NOTHING TO FOLD, and a fold of nothing has no
    // level to dispatch. It is already solved by inspection.
    if n == 0 {
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 0,
            relative_residual: 0.0,
        });
    }
    let term_rows = terms.rows();

    // r = b - A x0, the preconditioner, the first search direction and BOTH
    // opening scalars, in ONE boundary. Consecutive entries of a region are
    // ordered with a full barrier between them on every backend, so a phase
    // whose steps feed each other costs one host round trip rather than one per
    // dispatch.
    {
        let ap_h = ap.handle();
        let r_h = r.handle();
        let z_h = z.handle();
        let p_h = p.handle();
        let product_h = terms.product_handle();
        let err_out = scalar.span(slot::ERR, 1);
        let rz0_out = scalar.span(slot::RZ0, 1);
        // CLEAR THE SCALAR BUFFER BEFORE THE PHASE THAT FILLS IT, which the
        // announcement flag needs and nothing else here minds: every other slot
        // is written by a dispatch below or by the loop before anything reads
        // it. The flag is sticky within a solve and the workspace is reused by
        // every solve, so without this a breakdown in one solve would send the
        // next one to read a latch that the `cause` seed had already cleared,
        // and a Metal arena hands out allocations without zeroing besides.
        //
        // A WHOLE-BUFFER SEED RATHER THAN A ONE-SLOT FILL, because a fill takes
        // a live allocation and a slot is a span inside one.
        scalar.seed(device, &[0.0; slot::COUNT])?;
        device.run("pcg.seed", |encoder| {
            // Safety: every buffer is borrowed for the whole call, and a
            // sweep's handles outlive it for the same reason.
            unsafe {
                op.encode_apply(encoder, x, ap_h, row_absolute.handle(),
                                row_curvature.handle())?;
                encode_combine_raw(encoder, b, ap_h, r_h, 1.0, -1.0, n)?;
                // err0 = ||r||_1, the denominator every relative residual below
                // is measured against. It is the SEEDED residual and never
                // ||b||_1; see the module doc.
                encode_fold(encoder, r_h, n, fold, err_out, true)?;
                match schwarz.as_deref_mut() {
                    // THE WHOLE SWEEP, fine level and every coarse level,
                    // pushed onto this one encoder so an iteration still costs
                    // one submission whichever preconditioner runs.
                    Some(state) => super::schwarz::encode_whole_sweep(
                        encoder, state, r_h, z_h, rows)?,
                    None => encode_preconditioner(
                        encoder, inverse_diagonal, r_h, z_h, rows)?,
                }
                // p = z, which is the beta-free case of the combine the loop
                // uses below.
                encode_combine_raw(encoder, z_h, p_h, p_h, 1.0, 0.0, n)?;
                terms.encode_fill(encoder, r_h, z_h, n)?;
                encode_fold(encoder, product_h, term_rows, fold, rz0_out, false)
            }
        })?;
    }
    // THE ONE HOST READ AT SETUP. It answers two questions at once: the
    // trivially-already-converged case, and whether the preconditioner is SPD
    // at the seed.
    scalar.download(device)?;
    let err0 = scalar.host()[slot::ERR];
    if err0 == 0.0 {
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 0,
            relative_residual: 0.0,
        });
    }
    let rz0 = scalar.host()[slot::RZ0];
    // The negation is deliberate and clippy's suggestion would break it: NaN
    // fails `> 0.0`, so `!(rz0 > 0.0)` traps a NaN where `rz0 <= 0.0` would let
    // it through and let the solve run on garbage.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(rz0 > 0.0) {
        // The preconditioner is SPD by construction, so a non-positive r.z
        // means the residual or the preconditioner is broken upstream.
        return Ok(Report {
            outcome: Outcome::NonSpdPreconditioner,
            iterations: 0,
            relative_residual: 1.0,
        });
    }

    // The last residual the loop measured, which the iteration-cap exit reports.
    // Re-folding `r` at that exit would read the same residual out of the same
    // unchanged vector, so the loop's last measurement is carried instead.
    let mut residual = 1.0f32;
    // THE BATCH SIZE, WHICH IS ALSO THE RESIDUAL SAMPLING SCHEDULE.
    //
    // Every per-iteration scalar already stays on the device: the recurrence is
    // folds and indirect axpys, and alpha and beta are decided by one-element
    // resident bodies. The only thing crossing the seam is this probe, and
    // crossing on EVERY iteration is a full pipeline stall per iteration.
    // It is read on a stride of 4 while the residual is far from the tolerance
    // and every iteration once within 8x of it, so a sub-tol crossing is never
    // sampled past and convergence is detected at the same iterate an unstrided
    // loop would have found.
    //
    // WHAT MAKES IT SAFE IS THE LATCH, not the schedule. A breakdown between
    // two checks is held in the three sticky slots the resident bodies write,
    // so it is reported at the next check with the offending iterate's own
    // values. Without that latch this would silently drop a `pAp <= 0`.
    const RESID_CHECK_STRIDE: u32 = 4;
    const NEAR_TOL_FACTOR: f32 = 8.0;
    let mut stride = RESID_CHECK_STRIDE;
    // CLEAR THE LATCH BEFORE THE LOOP. The workspace is allocated once and
    // reused by every solve, and the latch is sticky BY DESIGN, so without this
    // a breakdown in one solve would abort the next one before it had taken a
    // single iteration. A Metal arena also hands out allocations without
    // zeroing and never faults on an uninitialized read, so the first solve
    // needs it as much as the second.
    cause.seed(device, &[PCG_BREAK_NONE; cause_slot::COUNT])?;
    // ONE SUBMIT PER BATCH, NOT PER ITERATION, and that is the whole point.
    //
    // A BACKEND SYNCHRONIZES AT THE END OF EVERY SUBMIT, which is the contract
    // `Device::run` is defined under, so a submit per iteration is a full
    // device stall per iteration whatever the probe does. Striding only the READ
    // leaves the stall in place and was measured at 1.24x; aligning the SUBMIT
    // with the check is the change that matters.
    //
    // Nothing on the host is needed between iterations: every per-iteration
    // scalar is device-resident and the breakdown is held in the sticky latch,
    // so `stride` iterations encode back to back into one batch. The stride is
    // therefore the BATCH SIZE and the sampling rate at once.
    // THE FIRST CHECK IS AT ITERATION ONE, so the
    // first iteration is sampled on its own and the residual has a chance to
    // tighten the stride before any coarse batch runs. Skipping that first
    // sample quantizes a short solve's reported iteration count to a multiple
    // of the stride, which is enough to hide a preconditioner that is not being
    // applied.
    // THE RECORDED ITERATION, captured once after a warm-up iteration and
    // replayed for every batch after the first, which collapses the iteration's
    // launches into one graph launch.
    //
    // IT IS RECORDABLE ONLY BECAUSE NOTHING IN THE BODY DEPENDS ON WHICH TURN
    // OF THE LOOP IT IS. Every per-iteration scalar is device-resident, and the
    // last host-supplied one, the iteration index, is a device counter the
    // alpha pass advances for itself.
    //
    // AND ONLY BECAUSE THE FAILURE MODE IS LATCH AND CONTINUE, which is
    // `Device::replay`'s stated precondition: a breakdown on repeat `k` is
    // reported after the whole batch has run, and the sticky `break_cause`
    // latch is exactly what makes that safe. The strided probe already accepted
    // that delay.
    let mut region: Option<D::Region> = None;
    // THE VERDICT, CARRIED OUT OF THE LOOP RATHER THAN RETURNED FROM INSIDE IT,
    // so the recorded region is released on every path that reaches an answer.
    // `Device::Region` has no `Drop`: `release` takes it by value and must be
    // called, and this solve runs thousands of times a frame.
    let mut finished: Option<Report> = None;
    let mut next_check = 1u32;
    let mut iteration = 0u32;
    while iteration < max_iterations {
        let target = next_check.min(max_iterations);
        let batch = target - iteration;
        let _first = iteration + 1;
        iteration = target;
        let ap_h = ap.handle();
        let r_h = r.handle();
        let z_h = z.handle();
        let p_h = p.handle();
        let product_h = terms.product_handle();
        let rz0_slot = scalar.span(slot::RZ0, 1);
        let rz1_slot = scalar.span(slot::RZ1, 1);
        let p_ap_slot = scalar.span(slot::P_AP, 1);
        let alpha_slot = scalar.span(slot::ALPHA, 1);
        let beta_slot = scalar.span(slot::BETA, 1);
        let err_slot = scalar.span(slot::ERR, 1);
        // ONE SLOT NOW, NOT TWO. The scale is the magnitude sum the apply
        // forms; the two Cauchy-Schwarz halves it replaced are gone.
        let absdot_slot = scalar.span(slot::ABSDOT, 1);
        // The three STICKY latch slots. They are written only on the first
        // breakdown and read at a scheduled check, which is what lets the host
        // sample on a stride without losing one.
        let break_value_slot = scalar.span(slot::BREAK_VALUE, 1);
        let break_cause_slot = cause.span(cause_slot::BREAK_CAUSE, 1);
        let break_iter_slot = cause.span(cause_slot::BREAK_ITER, 1);
        let iteration_slot = cause.span(cause_slot::ITERATION, 1);
        let break_fired_slot = scalar.span(slot::BREAK_FIRED, 1);
        let absolute_h = row_absolute.handle();
        // ONE GROUP PER `APPLY_GROUP` ROWS, the same split the folded apply
        // uses, so both per-group partial arrays are indexed the same way and
        // both are at most `rows / APPLY_GROUP` long.
        let update_groups = rows.div_ceil(super::operator::APPLY_GROUP);
        let curvature_h = row_curvature.handle();
        let alpha_noise_slot = scalar.span(slot::ALPHA_NOISE, 1);
        let beta_noise_slot = scalar.span(slot::BETA_NOISE, 1);
        let alpha_cause_slot = cause.span(cause_slot::ALPHA, 1);
        let beta_cause_slot = cause.span(cause_slot::BETA, 1);

        if region.is_none() {
            region = Some(device.record("pcg.iteration", |encoder| {
            // Safety: every buffer is borrowed for the whole call, consecutive
            // entries of a region are ordered, and the only deliberate aliasing
            // is `p` in the direction update, whose body reads element `i`
            // before writing element `i`.
            unsafe {
                // THE APPLY FOLDS ITS OWN TWO SCALARS, per group rather
                // than per row, so what follows reduces `groups` floats and
                // not `rows` of them. Measured before this: 81,920 rows folded
                // 320, then 2, then 1, three dispatches for each of two chains
                // every iteration.
                let groups =
                    op.encode_apply_folded(encoder, p_h, ap_h, curvature_h, absolute_h)?;
                // p^T A p, and the scale its round-off bound is taken from.
                //
                // THE SCALE COMES OUT OF THE APPLY ITSELF: the operator sums
                // the MAGNITUDES of the signed contributions it is already
                // computing, per row, and this folds them. That is one fold
                // rather than the two the
                // Cauchy-Schwarz surrogate needed, and two fewer full-length
                // fills, because `|p|` and `|Ap|` are not wanted at all.
                // BOTH SCALARS COME OUT OF THE APPLY, from one pass over the
                // same terms. A separate dot fill would walk the product a
                // second time and give the curvature a different rounding from
                // the bound measuring it.
                // ONE CHAIN FOR BOTH. The two arrays have the same length and
                // the same shape, so a level folds both in one dispatch.
                // THE FOLD TAKES THE VERDICT, so no dispatch of its own stands
                // between the sums and the alpha they decide. The
                // classification, the sticky latch and the announcement are all
                // `pcg_alpha_resident`'s, reached from inside the fold's last
                // level.
                let alpha_fold_args = PcgFoldAlphaArgs {
                    curvature_source: curvature_h,
                    absolute_source: absolute_h,
                    length: groups,
                    curvature_out: p_ap_slot,
                    absolute_out: absdot_slot,
                    rz: rz0_slot,
                    iteration_counter: iteration_slot,
                    value_out: alpha_slot,
                    noise_out: alpha_noise_slot,
                    cause_out: alpha_cause_slot,
                    break_cause: break_cause_slot,
                    break_value: break_value_slot,
                    break_iteration: break_iter_slot,
                    break_fired: break_fired_slot,
                    count: 1,
                    seam_arena_count: 0,
                };
                encoder.groups(&alpha_fold_args, 1, FOLD_LANES)?;
                // x += alpha p and r -= alpha Ap. A breakdown leaves `alpha` at
                // zero, so both updates are exact no-ops and the iterate the
                // report returns is the one the guard fired at.
                match schwarz.as_deref_mut() {
                    // THE SCHWARZ PATH STAYS SPLIT. Its sweep is a multi-level
                    // pass of its own shape, not a per-row apply, so there is
                    // nothing for the fused row to absorb and the four
                    // dispatches below are what it costs.
                    Some(state) => {
                        encode_add_scaled_indirect(encoder, p_h, x, alpha_slot, 1.0, n)?;
                        encode_add_scaled_indirect(
                            encoder, ap_h, r_h, alpha_slot, -1.0, n)?;
                        super::schwarz::encode_whole_sweep(
                            encoder, state, r_h, z_h, rows)?;
                        terms.encode_fill(encoder, r_h, z_h, n)?;
                    }
                    // ONE PASS OVER THE ROW for the iterate, the residual, the
                    // preconditioned residual and the `r . z` terms. A breakdown
                    // leaves `alpha` at zero, so the two updates are exact
                    // no-ops and the iterate the report returns is the one the
                    // guard fired at.
                    None => {
                        // THE ROW FOLDS ITS OWN TWO SCALARS, per group. The
                        // element form left `r . z` per ROW and the residual
                        // norm to a magnitude level over `3 * rows` floats, so
                        // what followed was an abs level plus a three-level
                        // dual chain. Both sums are per group here, so they
                        // share one length and one dispatch folds them.
                        let update_args = PcgUpdateRowFoldedArgs {
                            direction: p_h,
                            product_direction: ap_h,
                            alpha: alpha_slot,
                            inverse_diagonal,
                            iterate: x,
                            residual: r_h,
                            preconditioned: z_h,
                            product_total: terms.product_handle(),
                            residual_total: terms.magnitude_handle(),
                            rows,
                            count: update_groups,
                            seam_arena_count: 0,
                        };
                        encoder.groups(&update_args, update_groups, super::operator::APPLY_GROUP)?;
                    }
                }
                // ||r||_1 at the new iterate. Folding it BEFORE the
                // preconditioner would give the same number, since the
                // preconditioner only reads `r`. The fused row writes `r` and
                // `z` in one pass, so there is no point before it to fold at,
                // and the value is the same either way.
                // ONE CHAIN FOR BOTH, which is a launch-count change and
                // nothing else. The two folds are independent and adjacent
                // with nothing between them, so every level of one can share a
                // dispatch with a level of the other; they differ only in
                // length, which `encode_fold_dual` carries per source. Each
                // scalar is summed over the same width in the same ascending
                // order at the same level as the chain it replaces.
                //
                // THE MAGNITUDE LEVEL GOES FIRST AND ALONE. Only the first
                // level of an L1 norm takes absolute values, so it is its own
                // dispatch and the merge begins above it. When that one level
                // is the whole residual fold there is nothing left to merge
                // and `r . z` folds by itself.
                match schwarz.as_deref_mut() {
                    // THE SCHWARZ PATH KEEPS THE CHAIN, because its sweep is a
                    // pass of its own shape and its row is not the fused one:
                    // `terms.encode_fill` still writes per-ROW arrays there.
                    Some(_) => {
                        let (residual_source, residual_count) =
                            encode_fold_abs_level(encoder, r_h, n, fold, err_slot)?;
                        if residual_count == 1 {
                            encode_fold(
                                encoder, product_h, term_rows, fold, rz1_slot, false)?;
                        } else {
                            encode_fold_dual(
                                encoder,
                                residual_source,
                                residual_count,
                                product_h,
                                term_rows,
                                fold,
                                residual_count,
                                fold_scratch_len(n),
                                err_slot,
                                rz1_slot,
                            )?;
                        }
                        // BETA STAYS ITS OWN DISPATCH ON THIS PATH, because the
                        // chain above ends in a plain fold that takes no
                        // verdict. The block-Jacobi arm below folds and
                        // classifies in one group.
                        let beta_args = PcgBetaResidentArgs {
                            rz_next: rz1_slot,
                            rz_previous: rz0_slot,
                            iteration_counter: iteration_slot,
                            value_out: beta_slot,
                            noise_out: beta_noise_slot,
                            cause_out: beta_cause_slot,
                            break_cause: break_cause_slot,
                            break_value: break_value_slot,
                            break_iteration: break_iter_slot,
                            break_fired: break_fired_slot,
                            count: 1,
                            seam_arena_count: 0,
                        };
                        encoder.elements(&beta_args, 1)?;
                    }
                    // Both partials are one float per group and share a length,
                    // so this is the same single dispatch the apply's pair uses.
                    None => {
                        // The same fold-and-verdict as alpha's, one step later:
                        // beta, the roll of `r . z` and the non-positive test
                        // all happen inside the fold that finishes the sums they
                        // read.
                        let beta_fold_args = PcgFoldBetaArgs {
                            product_source: product_h,
                            residual_source: terms.magnitude_handle(),
                            length: update_groups,
                            product_out: rz1_slot,
                            residual_out: err_slot,
                            rz_previous: rz0_slot,
                            iteration_counter: iteration_slot,
                            value_out: beta_slot,
                            noise_out: beta_noise_slot,
                            cause_out: beta_cause_slot,
                            break_cause: break_cause_slot,
                            break_value: break_value_slot,
                            break_iteration: break_iter_slot,
                            break_fired: break_fired_slot,
                            count: 1,
                            seam_arena_count: 0,
                        };
                        encoder.groups(&beta_fold_args, 1, FOLD_LANES)?;
                    }
                }
                encode_combine_indirect(encoder, z_h, p_h, p_h, 1.0, beta_slot, n)?;
            }
            Ok(())
            })?);
        }
        // THE ONE SUBMIT THE BATCH PAYS, whatever its size. The region is
        // replayed back to back with no host round trip inside it, and the
        // diagnostic channel is collected once for the whole batch.
        device.replay(
            region.as_ref().expect("the region was recorded on this pass"),
            batch,
        )?;

        // THE PROBE, once per BATCH. The batch is sized so it ends on the
        // scheduled check, so the submit's own synchronize is the only one the
        // iteration pays and this readback rides it rather than adding a
        // second.
        scalar.download(device)?;
        residual = scalar.host()[slot::ERR] / err0;
        // THE INT LATCH IS READ ONLY WHEN IT HOLDS SOMETHING. Both guards
        // announce a breakdown by writing `BREAK_FIRED` into the scalar buffer
        // this probe already downloads, so a healthy batch costs ONE
        // device-to-host copy rather than two. Measured on `drape`,
        // `pcg.cause` was 939 of 2,404 such copies.
        //
        // THE FLAG IS NECESSARY AND NOT MERELY SUFFICIENT, which is what makes
        // skipping the read safe: it is written by the same branch that writes
        // `break_cause`, so the int slot cannot hold a cause while the flag
        // reads zero. It is deliberately NOT derived from `BREAK_VALUE`, which
        // a real `pAp <= 0` breakdown can legitimately latch as 0.0.
        let (alpha_cause, break_iteration) = if scalar.host()[slot::BREAK_FIRED] != 0.0 {
            cause.download(device)?;
            // THE LATCHED CAUSE, not this iteration's. Between two checks the
            // live `cause_out` slots have been rewritten up to three times; the
            // sticky slot holds whichever breakdown fired first and the iterate
            // it fired at.
            (
                cause.host()[cause_slot::BREAK_CAUSE],
                cause.host()[cause_slot::BREAK_ITER],
            )
        } else {
            (PCG_BREAK_NONE, 0)
        };
        // CONVERGENCE IS TESTED FIRST, and the latch makes that REQUIRED rather
        // than a preference. A sticky flag can have fired several iterations
        // ago while the solve went on to converge; a finite residual below tol
        // wins in that case, and a NaN or Inf residual fails this test and
        // falls through to the breakdown branches below.
        if residual < tolerance {
            finished = Some(Report {
                outcome: Outcome::Converged,
                iterations: iteration,
                relative_residual: residual,
            });
            break;
        }
        // THE LATCHED CAUSE IS MATCHED EXHAUSTIVELY, because one slot now
        // carries all three. Alpha latches `PCG_BREAK_PAP` and
        // `PCG_BREAK_NOISE`, beta latches `PCG_BREAK_RZ`, and reading anything
        // that is not `PAP` as a truncation, which is what a two-way test did
        // while the slots were separate, would report a non-SPD preconditioner
        // as a benign Steihaug stop.
        if alpha_cause != PCG_BREAK_NONE {
            // The iterate the latch fired at, not the check that noticed it.
            let broke_at = if break_iteration > 0 {
                break_iteration as u32
            } else {
                iteration
            };
            let outcome = match alpha_cause {
                // Negative beyond the bound, or NaN: the sign is real, so the
                // assembled Newton Hessian is not SPD.
                PCG_BREAK_PAP => Outcome::IndefiniteMatrix,
                // The previous iterate's `r^T M^-1 r` was non-positive, so beta
                // could not be formed.
                PCG_BREAK_RZ => Outcome::NonSpdPreconditioner,
                // Unresolvable above the round-off of its own sum. Steihaug:
                // the direction so far is still a descent direction.
                //
                // REPORTED HERE rather than by the caller. A truncated
                // solve returns SUCCESS and an iterate the Newton line search
                // then accepts, so without this line a step that stopped early
                // on an unresolvable curvature is indistinguishable in the log
                // from one that converged.
                _ => {
                    ::log::info!(
                        "* cg truncated: curvature {:.3e} within round-off of its own \
                         reduction at iter {} (reresid {:.3e})",
                        scalar.host()[slot::BREAK_VALUE],
                        broke_at,
                        residual
                    );
                    Outcome::CurvatureTruncated
                }
            };
            finished = Some(Report {
                outcome,
                iterations: broke_at,
                relative_residual: residual,
            });
            break;
        }
        // Within reach of the tolerance: check every iteration from here so the
        // crossing is detected at the same iterate the unstrided loop would
        // have found.
        if residual < NEAR_TOL_FACTOR * tolerance {
            stride = 1;
        }
        next_check = iteration + stride;
    }

    if let Some(region) = region {
        device.release(region);
    }
    Ok(finished.unwrap_or(Report {
        outcome: Outcome::MaxIterations,
        iterations: max_iterations,
        relative_residual: residual,
    }))
}

/// One scalar from `pcg_alpha` / `pcg_beta`, with the body's own
/// classification of it.
struct PcgScalar {
    value: f32,
    cause: i32,
}

fn pcg_alpha<D: Device>(
    device: &mut D,
    // THE TWO OUT-BUFFERS, not the whole workspace: the caller has already
    // destructured it into disjoint field borrows, so a second borrow of the
    // whole would conflict with the vectors the same iteration is using.
    verdict: &mut ReadbackBuffer<f32>,
    verdict_cause: &mut ReadbackBuffer<i32>,
    rz: f32,
    p_ap: f32,
    absolute_dot: f32,
) -> Result<PcgScalar, Fault> {
    let args = PcgAlphaTermsArgs {
        // THE THREE OUT-PARAMETERS ARE THE WORKSPACE'S, not stack scalars: a
        // handle names an allocation, and a local has none.
        value_out: verdict.span(0, 1),
        noise_out: verdict.span(1, 1),
        cause_out: verdict_cause.span(0, 1),
        rz,
        p_ap,
        absolute_dot,
        count: 1,
        seam_arena_count: 0,
    };
    // ONE ELEMENT, because that is what it is: a scalar verdict evaluated by the
    // shared body. The device-resident loop reaches the same body from inside
    // its fold instead, which is a dispatch shape rather than different
    // arithmetic.
    // Safety: the three out-parameters outlive the dispatch.
    unsafe { device.launch("pcg.alpha", &args, 1) }?;
    // THE MIRRORS, brought current before the verdict is read: taking a handle
    // above staled both by construction.
    verdict.download(device)?;
    verdict_cause.download(device)?;
    let value = verdict.host()[0];
    let noise = verdict.host()[1];
    let cause = verdict_cause.host()[0];
    let _ = noise;
    Ok(PcgScalar { value, cause })
}

fn pcg_beta<D: Device>(
    device: &mut D,
    // THE TWO OUT-BUFFERS, not the whole workspace: the caller has already
    // destructured it into disjoint field borrows, so a second borrow of the
    // whole would conflict with the vectors the same iteration is using.
    verdict: &mut ReadbackBuffer<f32>,
    verdict_cause: &mut ReadbackBuffer<i32>,
    rz_next: f32,
    rz_previous: f32,
) -> Result<PcgScalar, Fault> {
    let args = PcgBetaTermsArgs {
        // As `pcg_alpha`: the workspace's, because a handle names an allocation.
        value_out: verdict.span(0, 1),
        noise_out: verdict.span(1, 1),
        cause_out: verdict_cause.span(0, 1),
        rz_next,
        rz_previous,
        count: 1,
        seam_arena_count: 0,
    };
    // Safety: the three out-parameters outlive the dispatch.
    unsafe { device.launch("pcg.beta", &args, 1) }?;
    // THE MIRRORS, brought current before the verdict is read: taking a handle
    // above staled both by construction.
    verdict.download(device)?;
    verdict_cause.download(device)?;
    let value = verdict.host()[0];
    let noise = verdict.host()[1];
    let cause = verdict_cause.host()[0];
    let _ = noise;
    Ok(PcgScalar { value, cause })
}

/// Per-row inner-product terms, folded under `reduce`'s fixed shape.
///
/// TWO ARRAYS BECAUSE THE SHARED BODY YIELDS TWO NUMBERS. `pcg_dot_terms`
/// computes a row's product and the magnitude of that same product from one
/// expression and writes both through gathered outputs, so both have to be
/// bound. Only the product is folded here: the scale the curvature bound is
/// taken from is [`DotTerms::norm_product`], for the reason the module doc
/// gives. `reduce::sum` fixes the block shape, so the fold is independent of
/// how rayon split the work.
struct DotTerms<'a> {
    product: &'a mut ReadbackBuffer<f32>,
    magnitude: &'a mut ReadbackBuffer<f32>,
}

impl DotTerms<'_> {
    /// Refresh the PRODUCT mirror, which is the only one anything reads.
    ///
    /// THE MAGNITUDE ARRAY IS DELIBERATELY LEFT STALE. `pcg_dot_terms` is a
    /// shared body that yields both numbers from one expression, so the
    /// gathered output has to be bound and written; nothing on this side folds
    /// it, because the scale the curvature bound is taken from is
    /// [`norm_product`](Self::norm_product) rather than the per-row magnitude
    /// (see the module doc). Downloading it anyway would be `rows` floats a
    /// Newton iteration carried across the seam for no reader.
    ///
    /// Leaving it stale is loud rather than quiet: `ReadbackBuffer::host`
    /// panics on a mirror that a dispatch named and no download refreshed, so
    /// a future reader who folds it gets an assertion naming the buffer, not
    /// the contents from before the pass.
    fn download(&mut self, device: &mut impl Device) -> Result<(), Fault> {
        self.product.download(device)
    }

    /// The per-row product array, as a handle, for the DEVICE fold that
    /// reduces it.
    ///
    /// It stales the mirror, as every `handle()` does, and that is correct
    /// here: nothing on this side reads the array any more. The fold's last
    /// level writes one float into a slot of the workspace's scalar buffer,
    /// and that slot is what the host reads.
    fn product_handle(&mut self) -> ppf_cts_compute::Handle {
        self.product.handle()
    }

    /// The magnitude array's handle, for the fused row that writes it.
    ///
    /// IT STILL HAS NO READER on this side, for the reason the note above
    /// gives; the fused body produces both numbers from one expression exactly
    /// as `pcg_dot_terms` does, so the output has to be bound either way.
    fn magnitude_handle(&mut self) -> ppf_cts_compute::Handle {
        self.magnitude.handle()
    }

    /// Rows, which is the extent every fill dispatches over and the length
    /// every fold of the product array reduces.
    fn rows(&self) -> usize {
        self.product.len()
    }

    /// Fill both arrays from the shared body.
    ///
    /// # Safety
    /// Both inputs must outlive the dispatch this appends.
    unsafe fn encode_fill(
        &mut self,
        encoder: &mut dyn Encoder,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        _n: usize,
    ) -> Result<(), Fault> {
        let rows = self.product.len() as u32;
        if rows == 0 {
            return Ok(());
        }
        let args = PcgDotTermsArgs {
            a,
            b,
            product: self.product.handle(),
            absolute_product: self.magnitude.handle(),
            count: rows,
            seam_arena_count: 0,
        };
        encoder.elements(&args, rows)
    }

    fn fill<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        _n: usize,
    ) -> Result<(), Fault> {
        let product = self.product.handle();
        let magnitude = self.magnitude.handle();
        let rows = self.product.len() as u32;
        if rows == 0 {
            return Ok(());
        }
        let args = PcgDotTermsArgs {
            a,
            b,
            product,
            absolute_product: magnitude,
            count: rows,
            seam_arena_count: 0,
        };
        // Safety: all four buffers are borrowed for the whole call.
        unsafe { device.launch("pcg.dot_terms", &args, rows) }?;
        Ok(())
    }

    /// `a . b` under the fixed fold shape.
    /// `a . b`, REDUCED ON THE DEVICE.
    ///
    /// The fold lands in one scalar slot and only that slot is read back, so
    /// what crosses the seam is a result rather than the vector it was
    /// computed from.
    ///
    /// IT IS BIT-IDENTICAL TO THE EQUIVALENT HOST FOLD.
    /// `the_device_fold_agrees_with_the_host_fold_bit_for_bit` pins the two
    /// shapes together, so reducing here rather than on the host changes what is
    /// TRANSPORTED and not what is COMPUTED.
    fn dot<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        n: usize,
        fold: &Buffer<f32>,
        scalars: &mut ReadbackBuffer<f32>,
    ) -> Result<f32, Fault> {
        self.fill(device, a, b, n)?;
        let rows = self.product.len();
        if rows == 0 {
            return Ok(0.0);
        }
        let source = self.product.handle();
        let out = scalars.span(slot::FOLD_SCRATCH, 1);
        device.run("pcg.dot", |encoder| {
            // Safety: the product vector and the scratch outlive the call.
            unsafe { encode_fold(encoder, source, rows, fold, out, false) }
        })?;
        scalars.download(device)?;
        Ok(scalars.host()[slot::FOLD_SCRATCH])
    }

    /// `a . b` and the round-off scale the curvature bound is taken from,
    /// measured over the same two vectors.
    ///
    /// THE SCALE IS THE CAUCHY-SCHWARZ SURROGATE, not the fold beside the dot,
    /// for the reason [`norm_product`](Self::norm_product) records: this
    /// backend's per-row magnitude has already spent the row's own
    /// cancellation, so it is not the quantity the shared bound is written
    /// against.
    fn dot_with_bound<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        n: usize,
        fold: &Buffer<f32>,
        scalars: &mut ReadbackBuffer<f32>,
    ) -> Result<(f32, f32), Fault> {
        let dot = self.dot(device, a, b, n, fold, scalars)?;
        let scale = self.norm_product(device, a, b, n, fold, scalars)?;
        Ok((dot, scale))
    }

    /// `|a|_2 |b|_2`, the scale `pcg_curvature_noise` is given on a path whose
    /// operator apply produces no `sum|contribution|` beside its dot.
    ///
    /// IT IS A SANCTIONED SCALE, NOT AN APPROXIMATION OF THE FUSED ONE.
    /// `sqrt(p.p) * sqrt(Ap.Ap)` is formed from three plain inner products, and
    /// `pcg_curvature_noise` in `src/kernels/solver/pcg.kernel.cpp` names it in
    /// as many words: "A host-synchronizing loop whose SpMV produces no such sum
    /// passes the Cauchy-Schwarz surrogate |p|_2 |Ap|_2 >= sum|p_i (Ap)_i|
    /// instead, which errs toward truncating rather than aborting."
    ///
    /// The two square roots are taken in host double and narrowed only at the
    /// return. That is a host scalar and
    /// not device arithmetic, so the float64 ban does not reach it. Taking
    /// `sqrt(aa * bb)` in fp32 instead would be one multiplication rather than
    /// two roots and is not the same function: `aa * bb` can overflow fp32
    /// while neither factor nor the product of the roots does.
    fn norm_product<D: Device>(
        &mut self,
        device: &mut D,
        a: ppf_cts_compute::Handle,
        b: ppf_cts_compute::Handle,
        n: usize,
        fold: &Buffer<f32>,
        scalars: &mut ReadbackBuffer<f32>,
    ) -> Result<f32, Fault> {
        // Sequential on purpose: each fold is read back before the next is
        // encoded, so both can use the one scratch slot.
        let aa = self.dot(device, a, a, n, fold, scalars)?;
        let bb = self.dot(device, b, b, n, fold, scalars)?;
        Ok(((aa as f64).sqrt() * (bb as f64).sqrt()) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;
    use crate::driver::spmv;

    #[test]
    #[ignore = "release dispatch-overhead timing, run alone with --nocapture"]
    fn benchmark_pcg_dispatch_overhead() {
        use std::hash::{Hash, Hasher};

        // The production driver calls from outside Rayon. Installing this test
        // in a pool would omit the queue-entry overhead it measures.
        let threads = rayon::current_num_threads();
        assert!(rayon::current_thread_index().is_none());
        for rows in [1025, 8192, 81920, 600001] {
            let matrix = spd(rows);
            let n = 3 * rows as usize;
            let mut device = host_device();
            let resident = matrix.upload(&mut device);
            let diagonal = staged_block(&mut device, &matrix.zero_diagonal, "test.zero");
            let inverse = matrix.inverse(&mut device);
            let b = staged_block(
                &mut device,
                &(0..n).map(|i| (i % 17) as f32 / 17.0).collect::<Vec<_>>(),
                "test.b",
            );
            let zero = vec![0.0; n];
            let mut x = staged_block(&mut device, &zero, "test.x");
            let op = matrix.op(&resident, diagonal.handle());
            let mut work = matrix.work(&mut device);
            let mut expected = None;
            let mut samples = Vec::new();
            for trial in 0..9 {
                let mut elapsed = 0.0;
                for _ in 0..8 {
                    x.write(&mut device, 0, &zero).unwrap();
                    let start = std::time::Instant::now();
                    let report = solve(
                        &mut device, &op, inverse.handle(), b.handle(),
                        x.handle(), &mut work, None, 1.0e-5, 100,
                    ).unwrap();
                    elapsed += start.elapsed().as_secs_f64() * 1e3;
                    assert_eq!(report.outcome, Outcome::Converged);
                    let value: Vec<u32> = read_back(&mut device, &x, n)
                        .into_iter().map(f32::to_bits).collect();
                    let answer = (value, report.iterations, report.relative_residual.to_bits());
                    if let Some(ref expected) = expected {
                        assert_eq!(&answer, expected);
                    } else {
                        expected = Some(answer);
                    }
                }
                if trial > 0 {
                    samples.push(elapsed / 8.0);
                }
            }
            samples.sort_by(f64::total_cmp);
            let answer = expected.unwrap();
            let mut fingerprint = std::hash::DefaultHasher::new();
            answer.hash(&mut fingerprint);
            println!(
                "dispatch-pcg threads={threads} rows={rows} ms={:.6} min={:.6} max={:.6} \
                 iterations={} answer={:016x}",
                (samples[3] + samples[4]) / 2.0, samples[0], samples[7],
                answer.1, fingerprint.finish(),
            );
        }
    }

    #[test]
    #[ignore = "release solve timing, run alone with --nocapture"]
    fn benchmark_pcg_scheduling() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            for rows in [81920, 200001, 600001] {
                let matrix = spd(rows);
                let n = 3 * rows as usize;
                let mut samples = [Vec::new(), Vec::new()];
                let mut expected = None;
                for trial in 0..7 {
                    for mode in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let mut device = crate::driver::launch::scheduling_tests::device(mode);
                        let resident = matrix.upload(&mut device);
                        let diagonal = staged_block(&mut device, &matrix.zero_diagonal, "test.zero");
                        let inverse = matrix.inverse(&mut device);
                        let b = staged_block(&mut device,
                            &(0..n).map(|i| (i % 17) as f32 / 17.0).collect::<Vec<_>>(), "test.b");
                        let zero = vec![0.0; n];
                        let mut x = staged_block(&mut device, &zero, "test.x");
                        let op = matrix.op(&resident, diagonal.handle());
                        let mut work = matrix.work(&mut device);
                        for warmup in [true, false] {
                            x.write(&mut device, 0, &zero).unwrap();
                            let start = std::time::Instant::now();
                            let report = solve(&mut device, &op, inverse.handle(), b.handle(),
                                x.handle(), &mut work, None, 1.0e-5, 100).unwrap();
                            let elapsed = start.elapsed().as_secs_f64() * 1e3;
                            assert_eq!(report.outcome, Outcome::Converged);
                            let value: Vec<u32> = read_back(&mut device, &x, n)
                                .into_iter().map(f32::to_bits).collect();
                            let answer = (value, report.iterations, report.relative_residual.to_bits());
                            if let Some(ref expected) = expected { assert_eq!(&answer, expected); }
                            else { expected = Some(answer); }
                            if !warmup {
                                samples[mode].push(elapsed);
                                println!("pcg rows={rows} trial={trial} mode={mode} iterations={} ms={elapsed:.3}",
                                    report.iterations);
                            }
                        }
                    }
                }
                for (mode, values) in samples.iter_mut().enumerate() {
                    values.sort_by(f64::total_cmp);
                    println!("pcg-summary rows={rows} mode={mode} ms={:.3} min={:.3} max={:.3}",
                        values[3], values[0], values[6]);
                }
            }
        });
    }

    /// A diagonally dominant SPD matrix in the fixed-CSR form, upper triangle
    /// stored with the transpose table naming the mirrored blocks.
    struct Spd {
        index: Vec<u32>,
        offset: Vec<u32>,
        value: Vec<f32>,
        transpose_pair: Vec<u32>,
        transpose_offset: Vec<u32>,
        /// The fixed matrix's own diagonal blocks, kept separately as the
        /// preconditioner's source.
        diagonal: Vec<f32>,
        /// The operator's `C` term. ZERO for this fixture, so the composed
        /// operator is exactly the fixed matrix and the tests below measure PCG
        /// rather than the composition. Adding an exact zero block is exact, so
        /// the composition is still the one the driver runs.
        zero_diagonal: Vec<f32>,
        rows: u32,
    }

    fn spd(rows: u32) -> Spd {
        let n = rows as usize;
        let mut index = Vec::new();
        let mut offset = vec![0u32];
        let mut value = Vec::new();
        let mut diagonal = vec![0.0f32; 9 * n];
        for i in 0..n {
            // Diagonal block: strongly dominant so the system is well
            // conditioned and CG converges quickly.
            let d = 8.0 + (i % 5) as f32;
            index.push(i as u32);
            let block = [d, 0.3, 0.1, 0.3, d + 1.0, 0.2, 0.1, 0.2, d + 2.0];
            value.extend_from_slice(&block);
            diagonal[9 * i..9 * i + 9].copy_from_slice(&block);
            if i + 1 < n {
                index.push(i as u32 + 1);
                let o = 0.75;
                value.extend_from_slice(&[o, 0.0, 0.0, 0.0, o, 0.0, 0.0, 0.0, o]);
            }
            offset.push(index.len() as u32);
        }
        let mut transpose_pair = Vec::new();
        let mut transpose_offset = vec![0u32];
        for j in 0..n {
            if j > 0 {
                let source = j - 1;
                let slot = offset[source] as usize + 1;
                transpose_pair.push(source as u32);
                transpose_pair.push(slot as u32);
            }
            transpose_offset.push((transpose_pair.len() / 2) as u32);
        }
        Spd {
            index,
            offset,
            value,
            transpose_pair,
            transpose_offset,
            zero_diagonal: vec![0.0f32; 9 * n],
            diagonal,
            rows,
        }
    }

    /// A solved vector, read back off its device allocation.
    fn read_back(device: &mut impl Device, buffer: &Buffer<f32>, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        buffer
            .read(device, 0, &mut out)
            .expect("the solved vector reads back");
        out
    }

    /// A device copy of one block array, which the solve now takes by handle.
    fn staged_block(
        device: &mut impl Device,
        host: &[f32],
        label: &'static str,
    ) -> Buffer<f32> {
        let mut buffer = Buffer::<f32>::none();
        buffer
            .size(device, host.len(), AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, host)
            .expect("the test upload succeeds");
        buffer
    }

    /// This fixture's five arrays, on the device the solve runs on.
    ///
    /// THE VIEW IS ALL HANDLES NOW, so the fixture has to own allocations
    /// rather than lend slices. One of these is built per test and lives as
    /// long as the device does, which is what keeps a handle resolvable.
    struct SpdDevice {
        index: Buffer<u32>,
        offset: Buffer<u32>,
        value: Buffer<f32>,
        transpose_pair: Buffer<u32>,
        transpose_offset: Buffer<u32>,
        rows: u32,
    }

    /// As [`staged_block`], for an index array.
    fn staged_index(device: &mut impl Device, host: &[u32], label: &'static str) -> Buffer<u32> {
        let mut buffer = Buffer::<u32>::none();
        buffer
            .size(device, host.len(), AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, host)
            .expect("the test upload succeeds");
        buffer
    }

    impl SpdDevice {
        fn view(&self) -> spmv::FixedCsrView {
            spmv::FixedCsrView {
                index: self.index.span(0, self.index.len()),
                offset: self.offset.span(0, self.offset.len()),
                value: self.value.span(0, self.value.len()),
                transpose_pair: self.transpose_pair.span(0, self.transpose_pair.len()),
                transpose_offset: self.transpose_offset.span(0, self.transpose_offset.len()),
                rows: self.rows,
            }
        }
    }

    impl Spd {
        /// Upload the five arrays and hand back the owner.
        fn upload(&self, device: &mut impl Device) -> SpdDevice {
            SpdDevice {
                index: staged_index(device, &self.index, "test.spd.index"),
                offset: staged_index(device, &self.offset, "test.spd.offset"),
                value: staged_block(device, &self.value, "test.spd.value"),
                transpose_pair: staged_index(device, &self.transpose_pair, "test.spd.tpair"),
                transpose_offset: staged_index(
                    device,
                    &self.transpose_offset,
                    "test.spd.toffset",
                ),
                rows: self.rows,
            }
        }

        /// The block-Jacobi inverse of this fixture's diagonal.
        ///
        /// Wraps the caller-owned-buffer form so a test reads as one line; the
        /// buffer belongs to the caller because the real one is a driver
        /// allocation made once at `initialize()`.
        fn inverse(&self, device: &mut impl Device) -> Buffer<f32> {
            let mut host = vec![0.0f32; 9 * self.rows as usize];
            assert!(
                build_block_jacobi(&self.diagonal, &mut host, self.rows),
                "the fixture's diagonal blocks are SPD and must invert"
            );
            staged_block(device, &host, "test.inverse")
        }


        /// A workspace sized for this fixture.
        /// A workspace sized ON THE CALLER'S DEVICE. The fold outputs are device
        /// allocations now, and a handle is only meaningful to the backend that
        /// opened its arena, so sizing here and solving on another instance
        /// would read an arena that was never opened.
        fn work(&self, device: &mut impl Device) -> Workspace {
            let mut work = Workspace::default();
            work.size_for(device, self.rows)
                .expect("the fixture workspace sizes");
            work
        }

        /// The operator over this fixture's matrix, with a caller-supplied zero
        /// block diagonal: the operator carries a HANDLE now, and a handle has
        /// to name an allocation on the device the solve will run on.
        fn op(&self, resident: &SpdDevice, diagonal: ppf_cts_compute::Handle) -> Operator {
            Operator {
                dynamic: None,
                fixed: resident.view(),
                diagonal,
            }
        }
    }


    /// A BREAKDOWN NAMES ITS CAUSE, because the two causes send a reader to
    /// different code.
    ///
    /// These two matter beyond tidiness. `step.rs` turns either into a fatal
    /// abort, and the whole `pAp<=0` triage keys on which one it is: an
    /// indefinite MATRIX is a lost PSD projection or a dropped CSR block in the
    /// assembly, while a non-SPD PRECONDITIONER is a NaN or infinite diagonal
    /// block. Reporting one message for both also collapses them onto the
    /// `CrashKind::Cg` an ordinary iteration-cap exhaustion produces, and the
    /// natural response to THAT is to raise `cg-max-iter`.
    #[test]
    fn an_indefinite_matrix_is_reported_as_the_matrix_and_not_the_preconditioner() {
        // NEGATE THE MATRIX AND NOTHING ELSE. The preconditioner is built from
        // `diagonal`, which is untouched, so it stays SPD and `r.z` is positive
        // at the seed: the only thing broken is the operator, and `p^T A p` is
        // then negative by the full magnitude of the SPD form rather than by a
        // round-off sliver, which is what puts it past the bound.
        let mut a = spd(200);
        for v in &mut a.value {
            *v = -*v;
        }
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 7) as f32 - 3.0) * 0.25).collect();
        let mut device = host_device();
        let b_in = staged_block(&mut device, &b_host, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);
        let mut x = staged_block(&mut device, &vec![0.0f32; n], "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-5, 500).expect("the solve returns a report rather than an error");
        assert_eq!(
            report.outcome,
            Outcome::IndefiniteMatrix,
            "a negated SPD operator is indefinite by the whole magnitude of its \
             own quadratic form, so this must name the MATRIX; got {report:?}"
        );
    }

    /// THE LATCH IS WHAT MAKES THE STRIDED RESIDUAL READ SAFE, so it is tested
    /// directly rather than through a solve.
    ///
    /// The host samples the residual every `RESID_CHECK_STRIDE` iterations
    /// while it is far from the tolerance, so a breakdown can fire on an
    /// iteration nobody looks at. The live `cause_out` slot is rewritten every
    /// iteration, so without a sticky slot a `pAp <= 0` followed by three
    /// healthy iterations would be ERASED before the next check and the solve
    /// would continue on a direction the guard had already rejected. That is
    /// the `pAp <= 0` family, which must abort loudly.
    ///
    /// Every existing breakdown test breaks at iteration 1, which is always a
    /// check, so none of them covers this.
    #[test]
    fn a_breakdown_between_two_checks_survives_to_the_next_one() {
        let mut device = host_device();
        let mut floats = ReadbackBuffer::<f32>::default();
        floats
            .size(&mut device, 5, AllocLabel("test.pcg_latch.f32"))
            .expect("the test allocation succeeds");
        // THE LATCHED VALUE GETS ITS OWN ALLOCATION, because `seed` writes a
        // WHOLE buffer: keeping it beside the inputs meant re-seeding the
        // inputs for the next dispatch silently cleared the latch, which is the
        // very thing under test.
        let mut latched = ReadbackBuffer::<f32>::default();
        latched
            .size(&mut device, 1, AllocLabel("test.pcg_latch.value"))
            .expect("the test allocation succeeds");
        latched
            .seed(&mut device, &[0.0])
            .expect("the latched value starts clear");
        // THE ANNOUNCEMENT FLAG, likewise its own allocation and for the same
        // reason: the probe skips reading the int latch when this reads zero,
        // so a flag cleared by a re-seed would hide the very latch under test.
        let mut fired = ReadbackBuffer::<f32>::default();
        fired
            .size(&mut device, 1, AllocLabel("test.pcg_latch.fired"))
            .expect("the test allocation succeeds");
        fired
            .seed(&mut device, &[0.0])
            .expect("the flag starts clear");
        let mut ints = ReadbackBuffer::<i32>::default();
        ints.size(&mut device, 4, AllocLabel("test.pcg_latch.i32"))
            .expect("the test allocation succeeds");

        // slots: 0 rz, 1 p_ap, 2 absolute_sum, 3 value, 4 noise, 5 break_value
        // ints:  0 cause_out, 1 break_cause, 2 break_iteration, 3 counter
        ints.seed(&mut device, &[PCG_BREAK_NONE; 4])
            .expect("the latch starts clear");

        let mut fire = |device: &mut crate::driver::launch::HostDevice,
                        p_ap: f32| {
            floats
                .seed(device, &[1.0, p_ap, 1.0e-30, 0.0, 0.0])
                .expect("the inputs upload");
            // NOTHING RE-SEEDS THE INT BLOCK, which holds both the sticky latch
            // this test exists for and the iteration counter. The counter opens
            // at zero and the body advances it, so the three calls below ARE
            // iterations 1, 2 and 3, which is the same advance the solve makes.
            let args = PcgAlphaResidentArgs {
                rz: floats.span(0, 1),
                p_ap: floats.span(1, 1),
                absolute_sum: floats.span(2, 1),
                iteration_counter: ints.span(3, 1),
                value_out: floats.span(3, 1),
                noise_out: floats.span(4, 1),
                cause_out: ints.span(0, 1),
                break_cause: ints.span(1, 1),
                break_value: latched.span(0, 1),
                break_iteration: ints.span(2, 1),
                break_fired: fired.span(0, 1),
                count: 1,
                seam_arena_count: 0,
            };
            device
                .run("test.pcg_latch", |encoder| {
                    // Safety: one element, three disjoint out-slots, and the
                    // two buffers are borrowed for the whole call.
                    unsafe { encoder.elements(&args, 1) }
                })
                .expect("the dispatch succeeds");
        };

        // Iteration 1 breaks: a curvature negative far beyond any round-off
        // bound, which is the fatal classification rather than the truncation.
        fire(&mut device, -1.0);
        // Iterations 2 and 3 are healthy and REWRITE the live cause slot, which
        // is exactly what the latch has to survive.
        fire(&mut device, 1.0);
        fire(&mut device, 1.0);

        ints.download(&mut device).expect("the latch reads back");
        latched.download(&mut device).expect("the value reads back");
        assert_eq!(
            ints.host()[0],
            PCG_BREAK_NONE,
            "the LIVE cause slot must show the last iteration's healthy verdict, \
             or this test is not exercising the overwrite it exists for"
        );
        assert_eq!(
            ints.host()[1],
            PCG_BREAK_PAP,
            "the STICKY cause must survive two healthy iterations; if it does \
             not, a strided residual read drops a pAp <= 0 and the solve runs \
             on a direction the guard rejected"
        );
        assert_eq!(
            ints.host()[2], 1,
            "the latch must name the iterate that BROKE, not the check that \
             noticed it, and the number is the DEVICE counter's: the first call \
             below is iteration 1 because the body advances the counter itself"
        );
        assert_eq!(
            latched.host()[0], -1.0,
            "the latched value must be the offending curvature, not a later \
             iterate's"
        );
        // THE FLAG THE PROBE GATES ON. If this were ever zero while the sticky
        // cause held a breakdown, the batch probe would skip the int readback
        // and the solve would run on a direction the guard had rejected, which
        // is the whole failure the gating could introduce.
        fired.download(&mut device).expect("the flag reads back");
        assert_ne!(
            fired.host()[0], 0.0,
            "the announcement flag must be set whenever the sticky cause is, or \
             the probe skips the readback that would have stopped the solve"
        );
    }

    /// A HEALTHY SOLVE LEAVES THE ANNOUNCEMENT FLAG CLEAR, which is the other
    /// half of what makes gating the int readback sound. The flag being
    /// NECESSARY is asserted above; this asserts it is not simply always set,
    /// which would make the gate a no-op that still passes that test.
    #[test]
    fn a_healthy_alpha_leaves_the_announcement_flag_clear() {
        let mut device = host_device();
        let mut floats = ReadbackBuffer::<f32>::default();
        floats
            .size(&mut device, 5, AllocLabel("test.pcg_clear.f32"))
            .expect("the test allocation succeeds");
        let mut latched = ReadbackBuffer::<f32>::default();
        latched
            .size(&mut device, 1, AllocLabel("test.pcg_clear.value"))
            .expect("the test allocation succeeds");
        latched.seed(&mut device, &[0.0]).expect("the value starts clear");
        let mut fired = ReadbackBuffer::<f32>::default();
        fired
            .size(&mut device, 1, AllocLabel("test.pcg_clear.fired"))
            .expect("the test allocation succeeds");
        fired.seed(&mut device, &[0.0]).expect("the flag starts clear");
        let mut ints = ReadbackBuffer::<i32>::default();
        // FOUR SLOTS: the live cause, the sticky one, the latched iteration,
        // and the iteration counter the body advances for itself.
        ints.size(&mut device, 4, AllocLabel("test.pcg_clear.i32"))
            .expect("the test allocation succeeds");
        ints.seed(&mut device, &[PCG_BREAK_NONE; 4])
            .expect("the latch starts clear");
        // A positive curvature far above the round-off bound: healthy on every
        // branch alpha classifies.
        floats
            .seed(&mut device, &[1.0, 1.0, 1.0e-30, 0.0, 0.0])
            .expect("the inputs upload");
        let args = PcgAlphaResidentArgs {
            rz: floats.span(0, 1),
            p_ap: floats.span(1, 1),
            absolute_sum: floats.span(2, 1),
            iteration_counter: ints.span(3, 1),
            value_out: floats.span(3, 1),
            noise_out: floats.span(4, 1),
            cause_out: ints.span(0, 1),
            break_cause: ints.span(1, 1),
            break_value: latched.span(0, 1),
            break_iteration: ints.span(2, 1),
            break_fired: fired.span(0, 1),
            count: 1,
            seam_arena_count: 0,
        };
        device
            .run("test.pcg_clear", |encoder| {
                // Safety: one element and every buffer is borrowed for the call.
                unsafe { encoder.elements(&args, 1) }
            })
            .expect("the dispatch succeeds");
        fired.download(&mut device).expect("the flag reads back");
        ints.download(&mut device).expect("the latch reads back");
        assert_eq!(
            ints.host()[1],
            PCG_BREAK_NONE,
            "the fixture must be healthy, or it is not testing the clear case"
        );
        assert_eq!(
            fired.host()[0], 0.0,
            "a healthy iteration must leave the flag clear, or gating the int \
             readback on it saves nothing"
        );
    }

    #[test]
    fn a_non_spd_preconditioner_is_reported_as_the_preconditioner() {
        // NEGATE THE PRECONDITIONER AND NOTHING ELSE, so `r^T M^-1 r` is
        // negative at the seed and the loop breaks before it ever forms
        // `p^T A p`. The operator is the untouched SPD fixture.
        let a = spd(200);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 7) as f32 - 3.0) * 0.25).collect();
        let mut device = host_device();
        let b_in = staged_block(&mut device, &b_host, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let mut inverse_host = vec![0.0f32; 9 * a.rows as usize];
        assert!(
            build_block_jacobi(&a.diagonal, &mut inverse_host, a.rows),
            "the fixture's diagonal blocks are SPD and must invert"
        );
        for v in &mut inverse_host {
            *v = -*v;
        }
        let inverse = staged_block(&mut device, &inverse_host, "test.negated_inverse");
        let mut x = staged_block(&mut device, &vec![0.0f32; n], "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-5, 500).expect("the solve returns a report rather than an error");
        assert_eq!(
            report.outcome,
            Outcome::NonSpdPreconditioner,
            "a negated block-Jacobi inverse makes r^T M^-1 r negative at the \
             seed, so this must name the PRECONDITIONER; got {report:?}"
        );
    }

    /// THE DEVICE FOLD IS THE HOST FOLD, up to the length at which the host
    /// fold stops having a second level.
    ///
    /// `super::reduce`'s contract is that the array is cut into fixed blocks,
    /// each summed in index order, and the block totals summed in index order.
    /// That is exactly two levels, so for any input the device fold finishes in
    /// two levels the two must agree BIT FOR BIT, not merely closely: the
    /// curvature bound this backend judges a solve against is stated over a
    /// fold of a particular shape, and a fold whose shape drifts silently
    /// invalidates it.
    ///
    /// The COOPERATIVE fold agrees with the element fold inside the fp32
    /// summation bound, and CANNOT agree to the bit.
    ///
    /// **THE BIT-IDENTICAL VERSION OF THIS TEST WAS WRITTEN FIRST AND IS
    /// UNACHIEVABLE.** The lanes take CONTIGUOUS runs in lane order and
    /// `compute::block_sum` folds them in ASCENDING order on this backend, which
    /// looked like it reproduced the element form's left-to-right sum. It does
    /// not: summing each run and then combining the run totals is
    /// `(a0+a1+a2+a3) + (a4+a5+a6+a7) + ...`, and fp32 addition is not
    /// associative, so it is not `((((a0+a1)+a2)+a3)+a4)+...`. **No lane
    /// decomposition reproduces a single thread's sum**, whatever order the
    /// lanes are combined in; only one lane doing all the work would.
    ///
    /// SO THE ASSERTION IS A BOUND, and it is `n * eps * sum|x|`, the standard
    /// fp32 summation bound and the one this solver already classifies against:
    /// `pcg_alpha` decides a breakdown by comparing a curvature against the
    /// round-off of its OWN absolute sum, because a cancelling sum's error is
    /// set by the magnitudes that went in and not by what came out. A bound
    /// taken relative to the TOTAL was tried and is wrong: on this input one
    /// block cancels from 1e7 terms down to 60, where five absolute is four ULP
    /// of the input and 8 percent of the answer.
    ///
    /// WHAT THIS MEANS FOR A CALLER. Moving a fold from the element form to this
    /// one MOVES ITS LAST BITS, so it is a numerics change and not a drop-in:
    /// `the_device_fold_agrees_with_the_host_fold_bit_for_bit` pins the PCG's
    /// fold against `reduce::sum`, and switching that recurrence over needs the
    /// host fold to take the same shape in the same change.
    #[test]
    fn the_cooperative_fold_agrees_with_the_element_fold_within_the_summation_bound() {
        use crate::driver::kernels::{VecBlockSumArgs, VecBlockSumCooperativeArgs};
        let mut device = host_device();
        // A CANCELLING INPUT, for the reason the fold test beside this one uses
        // one: an association only shows on a sum that cancels.
        let values: Vec<f32> = (0..5_000)
            .map(|i| {
                let magnitude = 1.0e6 * ((i % 17) as f32 + 1.0);
                if i % 2 == 0 { magnitude } else { -magnitude + 1.0 }
            })
            .collect();
        let source = staged_block(&mut device, &values, "test.coop.source");
        let blocks = values.len().div_ceil(FOLD_WIDTH);
        let mut element_out = Buffer::<f32>::none();
        element_out
            .size(&mut device, blocks, AllocLabel("test.coop.element"))
            .expect("the element output sizes");
        let mut group_out = Buffer::<f32>::none();
        group_out
            .size(&mut device, blocks, AllocLabel("test.coop.group"))
            .expect("the group output sizes");
        let source_handle = source.span(0, values.len());
        let element = VecBlockSumArgs {
            source: source_handle,
            length: values.len() as u32,
            width: FOLD_WIDTH as u32,
            total: element_out.handle(),
            count: blocks as u32,
            seam_arena_count: 0,
        };
        let cooperative = VecBlockSumCooperativeArgs {
            source: source_handle,
            length: values.len() as u32,
            width: FOLD_WIDTH as u32,
            total: group_out.handle(),
            count: blocks as u32,
            seam_arena_count: 0,
        };
        device
            .run("test.coop", |encoder| {
                // Safety: every buffer outlives the call.
                unsafe {
                    encoder.elements(&element, blocks as u32)?;
                    // THE WIDTH AND THE SCRATCH COME FROM THE DECLARATION, which
                    // is what stops a call site inventing a group-local array the
                    // body did not ask for.
                    encoder.groups(
                        &cooperative,
                        blocks as u32,
                        64,
                    )
                }
            })
            .expect("both folds dispatch");
        let mut from_elements = vec![0.0f32; blocks];
        let mut from_groups = vec![0.0f32; blocks];
        element_out
            .read(&mut device, 0, &mut from_elements)
            .expect("the element fold reads back");
        group_out
            .read(&mut device, 0, &mut from_groups)
            .expect("the group fold reads back");
        for (block, (a, b)) in
            from_elements.iter().zip(from_groups.iter()).enumerate()
        {
            // THE BOUND IS `n * eps * sum|x|`, WHICH IS THE STANDARD FP32
            // SUMMATION BOUND AND THE ONE THIS SOLVER ALREADY CLASSIFIES
            // AGAINST. `pcg_alpha` decides a breakdown by comparing a curvature
            // against the round-off of its OWN absolute sum, for exactly this
            // reason: a cancelling sum's error is set by the magnitudes that
            // went in, not by what came out. A bound taken relative to the
            // TOTAL would be a bound on the cancellation instead, and on this
            // input one block cancels from 1e7 terms down to 60, where five
            // absolute is four ULP of the input and 8 percent of the answer.
            let first = block * FOLD_WIDTH;
            let last = (first + FOLD_WIDTH).min(values.len());
            let magnitude: f32 =
                values[first..last].iter().map(|v| v.abs()).sum();
            let bound = (last - first) as f32 * f32::EPSILON * magnitude;
            assert!(
                (a - b).abs() <= bound,
                "block {block}: the cooperative fold is {a} against the \
                 element fold's {b}, {} apart against a summation bound of \
                 {bound}. Within the bound is the association difference this \
                 shape is allowed; past it is a different sum",
                (a - b).abs()
            );
        }
    }

    /// The input cancels, because a fold's shape only shows on one that does.
    #[test]
    fn the_device_fold_agrees_with_the_host_fold_bit_for_bit() {
        let mut device = host_device();
        // 40_000 values is 157 blocks, which the device fold reduces in two
        // levels, and it crosses the block boundary many times over.
        let values: Vec<f32> = (0..40_000)
            .map(|i| {
                let magnitude = 1.0e6 * ((i % 17) as f32 + 1.0);
                if i % 2 == 0 {
                    magnitude
                } else {
                    -magnitude + 1.0
                }
            })
            .collect();
        assert!(
            values.len() <= FOLD_WIDTH * FOLD_WIDTH,
            "past this length the device fold takes a third level and the host \
             fold does not, which is a different association by design"
        );
        let source = staged_block(&mut device, &values, "test.fold.source");
        let mut out = Buffer::<f32>::none();
        out.size(&mut device, 2, AllocLabel("test.fold.out"))
            .expect("the fold output sizes");
        let mut scratch = Buffer::<f32>::none();
        scratch
            .size(
                &mut device,
                fold_scratch_len(values.len()).max(1),
                AllocLabel("test.fold.scratch"),
            )
            .expect("the fold scratch sizes");
        let signed_out = out.span(0, 1);
        let magnitude_out = out.span(1, 1);
        let source_handle = source.span(0, values.len());
        device
            .run("test.fold", |encoder| {
                // Safety: every buffer outlives the call.
                unsafe {
                    encode_fold(
                        encoder,
                        source_handle,
                        values.len(),
                        &scratch,
                        signed_out,
                        false,
                    )?;
                    encode_fold(
                        encoder,
                        source_handle,
                        values.len(),
                        &scratch,
                        magnitude_out,
                        true,
                    )
                }
            })
            .expect("the fold dispatches");
        let mut got = [0.0f32; 2];
        out.read(&mut device, 0, &mut got).expect("the fold reads back");
        assert_eq!(
            got[0].to_bits(),
            reduce::sum(&values).to_bits(),
            "the device fold moved off the host fold: {} against {}",
            got[0],
            reduce::sum(&values)
        );
        assert_eq!(
            got[1].to_bits(),
            reduce::sum_abs(&values).to_bits(),
            "the device magnitude fold moved off the host one: {} against {}",
            got[1],
            reduce::sum_abs(&values)
        );
        assert!(
            reduce::sum_abs(&values) > reduce::sum(&values).abs(),
            "the fixture canceled too little to exercise what the shape is for"
        );
    }

    /// THE DUAL FOLD IS THE TWO CHAINS IT MERGES, BIT FOR BIT.
    ///
    /// The only claim [`encode_fold_dual`] makes is that sharing a dispatch
    /// changes no arithmetic: each source is summed over the same
    /// [`FOLD_WIDTH`] in the same ascending order at the same level as the
    /// single-array chain it replaces. This asserts that directly against
    /// [`encode_fold`], which is the thing a scene statistic cannot settle: the
    /// solver is non-deterministic run to run, so a displacement inside its
    /// envelope is consistent with a fold that is bit-exact AND with one that
    /// is merely close.
    ///
    /// THE LENGTHS ARE CHOSEN SO THE CHAINS HAVE DIFFERENT DEPTHS, because the
    /// level where one chain has reached its output and the other has not is
    /// the case the per-source counts exist for. A fixture with two equal
    /// lengths would exercise the equal-length pair and call it a pass.
    #[test]
    fn the_dual_fold_agrees_with_two_separate_folds_bit_for_bit() {
        // 300 reduces in two levels (300 -> 2 -> 1) and 70_000 in three
        // (70_000 -> 274 -> 2 -> 1), so each ordering puts a finished chain
        // beside a live one for a level.
        for (first_len, second_len) in
            [(300usize, 70_000usize), (70_000, 300), (40_000, 40_000), (1, 5_000)]
        {
            let mut device = host_device();
            let first: Vec<f32> = (0..first_len)
                .map(|i| {
                    let m = 1.0e6 * ((i % 17) as f32 + 1.0);
                    if i % 2 == 0 { m } else { -m + 1.0 }
                })
                .collect();
            let second: Vec<f32> = (0..second_len)
                .map(|i| {
                    let m = 1.0e5 * ((i % 13) as f32 + 1.0);
                    if i % 3 == 0 { -m } else { m + 0.5 }
                })
                .collect();
            let a = staged_block(&mut device, &first, "test.dual.a");
            let b = staged_block(&mut device, &second, "test.dual.b");
            let mut out = Buffer::<f32>::none();
            out.size(&mut device, 4, AllocLabel("test.dual.out"))
                .expect("the fold output sizes");
            let mut scratch = Buffer::<f32>::none();
            let first_span = fold_scratch_len(first_len).max(1);
            scratch
                .size(
                    &mut device,
                    first_span + fold_scratch_len(second_len).max(1),
                    AllocLabel("test.dual.scratch"),
                )
                .expect("the fold scratch sizes");
            let a_h = a.span(0, first_len);
            let b_h = b.span(0, second_len);
            device
                .run("test.dual", |encoder| {
                    // Safety: every buffer outlives the call.
                    unsafe {
                        encode_fold(encoder, a_h, first_len, &scratch, out.span(0, 1), false)?;
                        encode_fold(encoder, b_h, second_len, &scratch, out.span(1, 1), false)?;
                        encode_fold_dual(
                            encoder,
                            a_h,
                            first_len,
                            b_h,
                            second_len,
                            &scratch,
                            0,
                            first_span,
                            out.span(2, 1),
                            out.span(3, 1),
                        )
                    }
                })
                .expect("the folds dispatch");
            let mut got = [0.0f32; 4];
            out.read(&mut device, 0, &mut got).expect("the folds read back");
            assert_eq!(
                got[2].to_bits(),
                got[0].to_bits(),
                "the merged first chain moved off the separate one at \
                 ({first_len}, {second_len}): {} against {}",
                got[2],
                got[0]
            );
            assert_eq!(
                got[3].to_bits(),
                got[1].to_bits(),
                "the merged second chain moved off the separate one at \
                 ({first_len}, {second_len}): {} against {}",
                got[3],
                got[1]
            );
        }
    }

    /// SPLITTING THE MAGNITUDE LEVEL OFF AN L1 FOLD CHANGES NOTHING EITHER.
    ///
    /// The PCG dispatches the first level of `||r||_1` alone so the levels
    /// above it can share a chain with `r . z`. That is only sound if one
    /// magnitude level followed by plain levels is the same number as
    /// [`encode_fold`] with `absolute` set, which is what this asserts.
    #[test]
    fn the_split_magnitude_level_agrees_with_the_absolute_fold_bit_for_bit() {
        for count in [1usize, FOLD_WIDTH, 40_000, 70_000] {
            let mut device = host_device();
            let values: Vec<f32> = (0..count)
                .map(|i| {
                    let m = 1.0e6 * ((i % 17) as f32 + 1.0);
                    if i % 2 == 0 { m } else { -m + 1.0 }
                })
                .collect();
            let source = staged_block(&mut device, &values, "test.absplit.source");
            let tail = staged_block(&mut device, &[7.5f32], "test.absplit.tail");
            let mut out = Buffer::<f32>::none();
            out.size(&mut device, 3, AllocLabel("test.absplit.out"))
                .expect("the fold output sizes");
            let mut scratch = Buffer::<f32>::none();
            scratch
                .size(
                    &mut device,
                    2 * fold_scratch_len(count).max(1),
                    AllocLabel("test.absplit.scratch"),
                )
                .expect("the fold scratch sizes");
            let source_h = source.span(0, count);
            let tail_h = tail.span(0, 1);
            let tail_out = out.span(2, 1);
            device
                .run("test.absplit", |encoder| {
                    // Safety: every buffer outlives the call.
                    unsafe {
                        encode_fold(encoder, source_h, count, &scratch, out.span(0, 1), true)?;
                        let (rest, rest_len) = encode_fold_abs_level(
                            encoder,
                            source_h,
                            count,
                            &scratch,
                            out.span(1, 1),
                        )?;
                        if rest_len > 1 {
                            // THE PRODUCTION SHAPE: the levels above the
                            // magnitude one run as the FIRST chain of a dual,
                            // based past what that level wrote. The second
                            // chain here is one element, so it reaches its
                            // output immediately and every level after that
                            // must leave it alone.
                            encode_fold_dual(
                                encoder,
                                rest,
                                rest_len,
                                tail_h,
                                1,
                                &scratch,
                                rest_len,
                                fold_scratch_len(count).max(1),
                                out.span(1, 1),
                                tail_out,
                            )?;
                        }
                        Ok(())
                    }
                })
                .expect("the folds dispatch");
            let mut got = [0.0f32; 3];
            out.read(&mut device, 0, &mut got).expect("the folds read back");
            assert_eq!(
                got[1].to_bits(),
                got[0].to_bits(),
                "splitting the magnitude level moved the L1 norm at {count}: {} against {}",
                got[1],
                got[0]
            );
        }
    }

    /// THE FOLD SCRATCH COVERS EVERY LEVEL IT DISPATCHES, including the third
    /// one a long input takes.
    ///
    /// `Buffer::span` panics past the allocation, so an undersized scratch is
    /// loud rather than a wild write; this asserts the sizing arithmetic
    /// agrees with the loop that consumes it, which is the pair that can drift.
    #[test]
    fn the_fold_scratch_covers_every_level() {
        for count in [1usize, 2, FOLD_WIDTH, FOLD_WIDTH + 1, 40_000, 200_000, 3_000_000] {
            let mut needed = 0usize;
            let mut remaining = count;
            loop {
                let blocks = remaining.div_ceil(FOLD_WIDTH);
                if blocks == 1 {
                    break;
                }
                needed += blocks;
                remaining = blocks;
            }
            assert_eq!(
                fold_scratch_len(count),
                needed,
                "the scratch sizing and the fold's own level walk disagree at \
                 {count} values"
            );
        }
    }

    /// ONE REGION PER ITERATION, WHICH IS WHAT DEVICE RESIDENCY MEANS HERE.
    ///
    /// `Counters::syncs` is one per `Device::run`, so it counts host round
    /// trips. The recurrence this replaced took eight of them an iteration:
    /// the matvec and its fold, two more folds for the curvature scale, the
    /// alpha verdict, the two updates, the preconditioner and its fold, the
    /// beta verdict and the direction update, each a separate boundary because
    /// a host fold sat between them. Every one of those scalars is on the
    /// device now, so an iteration is one boundary and the seed is one more.
    ///
    /// THIS IS THE GATE THAT NOTICES A HOST READ COMING BACK. A fold moved back
    /// to the host cannot stay inside the region, because an `Encoder` cannot
    /// read; it has to split the iteration in two, and that shows up here as a
    /// second sync per iteration.
    #[test]
    fn an_iteration_costs_one_host_round_trip() {
        let a = spd(600);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 11) as f32 - 5.0) * 0.5).collect();
        let mut device = host_device();
        let b_in = staged_block(&mut device, &b_host, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);
        let x = staged_block(&mut device, &vec![0.0f32; n], "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        device.counters_reset();
        let report = solve(
            &mut device,
            &a.op(&resident, zero_diag.handle()),
            inverse.handle(),
            b_in.handle(),
            x.handle(),
            &mut work,
            None,
            1.0e-5,
            500,
        )
        .expect("solve");
        assert_eq!(report.outcome, Outcome::Converged);
        // THE PROPERTY IS ONE SYNC PER BATCH, NOT PER ITERATION, and the batch
        // is the residual stride. `be_encoder_submit` synchronizes at the end
        // of every `device.run`, so a submit per iteration is a device stall
        // per iteration; the loop therefore encodes a whole batch and reads the
        // probe once, on the submit's own synchronize.
        //
        // The schedule is a check after iteration 1, then
        // every `RESID_CHECK_STRIDE` until the residual comes within
        // `NEAR_TOL_FACTOR` of the tolerance, after which every iteration is
        // its own batch. So the count is bounded BELOW by the tightened tail
        // and ABOVE by one sync per iteration, and it must be strictly under
        // the latter or the batching has stopped working.
        let syncs = device.counters().syncs;
        let per_iteration = 1 + u64::from(report.iterations);
        assert!(
            syncs < per_iteration,
            "the solve took {syncs} host round trips over {} iterations, which is \
             one per iteration: the submit is not batched, and every \
             iteration is paying a full device synchronize",
            report.iterations
        );
        assert!(
            syncs >= 2,
            "the solve took {syncs} round trips, which is fewer than the seed \
             plus one check; a solve that never samples its residual cannot \
             have detected convergence"
        );
    }

    #[test]
    fn it_solves_an_spd_system_and_the_residual_is_real() {
        let a = spd(600);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 11) as f32 - 5.0) * 0.5).collect();
        let b = &b_host;
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let x_host = vec![0.0f32; n];
        let mut x = staged_block(&mut device, &x_host, "test.x");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-5, 500).expect("solve");
        assert_eq!(
            report.outcome,
            Outcome::Converged,
            "a well-conditioned SPD system must converge; got {report:?}"
        );

        // Verify independently: recompute the residual from the returned x
        // rather than trusting the loop's own bookkeeping.
        let mut ax = vec![0.0f32; n];
        let mut ax_d = staged_block(&mut device, &ax, "test.ax");
        // Safety: every handle names a live allocation on this device.
        unsafe {
            spmv::fixed_csr_spmv(
                &mut device,
                &resident.view(),
                x.span(0, n),
                ax_d.span(0, n),
                n,
            )
        }
        .expect("the fixed-pattern matvec dispatches");
        ax_d.read(&mut device, 0, &mut ax).expect("ax reads back");
        let x = read_back(&mut device, &x, n);
        let residual: f32 = reduce::sum_abs(
            &b.iter().zip(ax.iter()).map(|(p, q)| p - q).collect::<Vec<f32>>(),
        );
        let scale = reduce::sum_abs(&b);
        assert!(
            residual / scale < 1.0e-4,
            "the returned x does not satisfy the system: relative residual {}",
            residual / scale
        );
    }

    #[test]
    fn the_preconditioner_reduces_the_iteration_count() {
        // The point of a preconditioner, checked rather than assumed: the same
        // system with an identity preconditioner must take more iterations.
        let a = spd(600);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 7) as f32 - 3.0) * 0.25).collect();
        let b = &b_host;
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let x1_host = vec![0.0f32; n];
        let mut x1 = staged_block(&mut device, &x1_host, "test.x1");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let with = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x1.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");

        let mut identity_host = vec![0.0f32; 9 * a.rows as usize];
        for i in 0..a.rows as usize {
            identity_host[9 * i] = 1.0;
            identity_host[9 * i + 4] = 1.0;
            identity_host[9 * i + 8] = 1.0;
        }
        let x2_host = vec![0.0f32; n];
        let mut x2 = staged_block(&mut device, &x2_host, "test.x2");
        // NO SECOND DEVICE. `b_in` and `x2` are handles into the arena the first
        // one opened, and a fresh backend resolves the same arena index against
        // its own empty table: the unpreconditioned solve saw a zero right-hand
        // side and reported convergence at iteration zero.
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let identity = staged_block(&mut device, &identity_host, "test.identity");
        let without = solve(&mut device, &a.op(&resident, zero_diag.handle()), identity.handle(), b_in.handle(), x2.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");

        assert_eq!(with.outcome, Outcome::Converged);
        assert_eq!(without.outcome, Outcome::Converged);
        assert!(
            with.iterations < without.iterations,
            "block-Jacobi took {} iterations against {} unpreconditioned, so it \
             is not preconditioning anything",
            with.iterations,
            without.iterations
        );
    }

    #[test]
    fn the_solve_is_deterministic_across_thread_counts() {
        let a = spd(900);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 13) as f32 - 6.0) * 0.3).collect();
        let b = &b_host;
        // ONE DEVICE for the whole test: the inverse below is a handle into its
        // arena, so a nested block with its own backend could not resolve it.
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let reference = {
            let x_host = vec![0.0f32; n];
            let mut x = staged_block(&mut device, &x_host, "test.x");
            let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
            let r = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-6, 500).expect("solve");
            (read_back(&mut device, &x, n), r.iterations)
        };
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| {
                let x_host = vec![0.0f32; n];
                let mut x = staged_block(&mut device, &x_host, "test.x");
                let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
                let r = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), x.handle(), &mut work, None, 1.0e-6, 500).expect("solve");
                (read_back(&mut device, &x, n), r.iterations)
            });
            assert_eq!(
                reference.1, got.1,
                "the iteration count moved at {threads} threads"
            );
            for (i, (p, q)) in reference.0.iter().zip(got.0.iter()).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "x[{i}] moved at {threads} threads; the solve must be \
                     deterministic, which is what the fixed fold shape buys"
                );
            }
        }
    }

    #[test]
    fn the_seed_is_used_rather_than_discarded() {
        // Cap the work at one iteration and compare where each start lands. If
        // the solve discarded the seed and began from zero, both would land in
        // the same place; because it does not, the seeded start is far closer to
        // the answer after the same single step.
        let a = spd(300);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 9) as f32 - 4.0) * 0.2).collect();
        let b = &b_host;
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let answer_host = vec![0.0f32; n];
        let mut answer = staged_block(&mut device, &answer_host, "test.answer");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let converged = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), answer.handle(), &mut work, None, 1.0e-7, 2000).expect("solve");
        assert_eq!(converged.outcome, Outcome::Converged);

        // The converged answer, on the host, so the two single-step solves below
        // can be measured against it.
        let answer_values = read_back(&mut device, &answer, n);
        let distance = |x: &[f32]| -> f32 {
            let d: Vec<f32> = x
                .iter()
                .zip(answer_values.iter())
                .map(|(p, q)| p - q)
                .collect();
            reduce::sum_abs(&d)
        };

        // ONE DEVICE throughout: every buffer here is a handle into its arena.
        let from_zero_host = vec![0.0f32; n];
        let mut from_zero = staged_block(&mut device, &from_zero_host, "test.from_zero");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), from_zero.handle(), &mut work, None, 1.0e-12, 1).expect("solve");

        // Warm-started from the converged answer rather than from zero.
        let mut from_answer = staged_block(&mut device, &answer_values, "test.from_answer");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), from_answer.handle(), &mut work, None, 1.0e-12, 1).expect("solve");

        assert!(
            distance(&read_back(&mut device, &from_answer, n))
                < distance(&read_back(&mut device, &from_zero, n)) * 1.0e-3,
            "after one iteration the seeded start is {} from the answer and the \
             zero start is {}; if the seed were discarded these would match",
            distance(&read_back(&mut device, &from_answer, n)),
            distance(&read_back(&mut device, &from_zero, n))
        );
    }

    #[test]
    fn a_better_seed_does_not_buy_fewer_iterations() {
        // The consequence of the denominator rule, asserted so nobody "fixes"
        // it later by switching to ||b||. The tolerance is relative to the
        // SEEDED residual, so starting closer tightens the target in proportion
        // and the iteration count does not collapse. The same property holds on
        // the CUDA side: a warm start is a wash, and that is correct rather
        // than a plumbing defect.
        let a = spd(400);
        let n = 3 * a.rows as usize;
        let b_host: Vec<f32> = (0..n).map(|k| ((k % 9) as f32 - 4.0) * 0.2).collect();
        let b = &b_host;
        // ONE DEVICE for the whole test: the inverse below is a handle into its
        // arena, so a nested block with its own backend could not resolve it.
        let mut device = host_device();
        let b_in = staged_block(&mut device, b, "test.b");
        let zero_diag = staged_block(
            &mut device,
            &vec![0.0f32; 9 * a.rows as usize],
            "test.zero_diagonal",
        );
        let inverse = a.inverse(&mut device);

        let answer_host = vec![0.0f32; n];
        let mut answer = staged_block(&mut device, &answer_host, "test.answer");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let cold = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), answer.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");
        assert_eq!(cold.outcome, Outcome::Converged);

        // The warm start is a COPY of the converged answer, staged as its own
        // allocation so the cold solve's result is not overwritten.
        let answer_values = read_back(&mut device, &answer, n);
        let mut warm = staged_block(&mut device, &answer_values, "test.warm");
        let resident = a.upload(&mut device);
        let mut work = a.work(&mut device);
        let warm_report = solve(&mut device, &a.op(&resident, zero_diag.handle()), inverse.handle(), b_in.handle(), warm.handle(), &mut work, None, 1.0e-6, 2000).expect("solve");
        assert!(
            warm_report.iterations > 0,
            "a warm start that converged in zero iterations would mean the \
             residual was formed against ||b|| rather than against the seed"
        );
    }

    /// A numerical gate on the SHARED body, runnable without a GPU.
    ///
    /// Routing CUDA's `invert()` through `block_jacobi.kernel.cpp` moved the
    /// device image by 88 instructions, because the shared body returns the
    /// eigenvalues and a validity flag the inline version did not materialize.
    /// The arithmetic path is identical by inspection; this is the check that it
    /// is identical in behavior.
    ///
    /// WHAT IS ASSERTED, and what deliberately is not. For a well-conditioned
    /// block the result is a real inverse. For an extremely anisotropic one it is
    /// NOT, and that is the design rather than a defect: eigenvalues below about
    /// `eps * lambda_max` are unresolvable by `symm3x3`, so they are floored, and
    /// the header states that this "only affects preconditioning quality, never
    /// correctness". A preconditioner that is SPD but inexact costs iterations;
    /// PCG still converges. So the invariants that hold for EVERY block are
    /// symmetry and positive definiteness, and exactness is asserted only where
    /// the design offers it.
    ///
    /// Measured while writing this, and pre-existing rather than introduced: on
    /// `diag(1e6, 1, 1)` the reciprocal of the smallest eigenvalue comes back as
    /// 0.0105 rather than 1.0, so `symm3x3` has already lost that direction at an
    /// anisotropy of 1e6. The same arithmetic ran inline before this refactor, so
    /// CUDA has always done this.
    #[test]
    fn the_inverse_is_exact_where_the_design_promises_it() {
        let well_conditioned: [[f32; 9]; 2] = [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 0.5, 0.25, 0.5, 5.0, 0.75, 0.25, 0.75, 6.0],
        ];
        for (case, block) in well_conditioned.iter().enumerate() {
            let mut arr = [0.0f32; 9];
            assert!(
                build_block_jacobi(block, &mut arr, 1),
                "case {case}: an SPD block must invert"
            );
            for col in 0..3 {
                let mut e = [0.0f32; 3];
                e[col] = 1.0;
                let back = spmv::mat3_mul(block, &spmv::mat3_mul(&arr, &e));
                for row in 0..3 {
                    let want = if row == col { 1.0 } else { 0.0 };
                    assert!(
                        (back[row] - want).abs() < 1.0e-4,
                        "case {case}: M M^-1 is not the identity at ({row}, {col}): \
                         got {} want {want}",
                        back[row]
                    );
                }
            }
        }
    }

    /// Symmetry and positive definiteness hold for EVERY block, including the
    /// ill-conditioned drape case the header's comment is about.
    ///
    /// These are the invariants the PCG guards depend on: a non-SPD
    /// preconditioner corrupts the search direction, which is the failure the
    /// eigendecomposition was chosen over a cofactor inverse to prevent.
    #[test]
    fn the_inverse_is_symmetric_and_positive_definite_at_any_conditioning() {
        let cases: [[f32; 9]; 4] = [
            [4.0, 0.5, 0.25, 0.5, 5.0, 0.75, 0.25, 0.75, 6.0],
            [1.0e6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            // The drape case: normal stiffness ~1e11 against a tangential floor
            // ~1e2, where a raw cofactor inverse returns a sign-flipped block.
            [1.0e11, 0.0, 0.0, 0.0, 1.0e2, 0.0, 0.0, 0.0, 1.0e2],
            [1.0e-3, 1.0e-4, 0.0, 1.0e-4, 1.0e-3, 0.0, 0.0, 0.0, 1.0e-3],
        ];
        for (case, block) in cases.iter().enumerate() {
            let mut arr = [0.0f32; 9];
            assert!(
                build_block_jacobi(block, &mut arr, 1),
                "case {case}: an SPD block must invert"
            );
            let scale = arr.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0e-30);

            for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
                assert!(
                    (arr[3 * b + a] - arr[3 * a + b]).abs() <= 1.0e-6 * scale,
                    "case {case}: the inverse is not symmetric at ({a}, {b})"
                );
            }

            // Positive definite: v^T M^-1 v > 0 for a spread of directions.
            for v in [
                [1.0f32, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 0.5],
            ] {
                let mv = spmv::mat3_mul(&arr, &v);
                let quadratic: f32 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    quadratic > 0.0,
                    "case {case}: v^T M^-1 v = {quadratic} for v = {v:?}; a \
                     non-SPD preconditioner corrupts the PCG search direction, \
                     which is what the eigendecomposition exists to prevent"
                );
            }
        }
    }

    #[test]
    fn a_non_spd_block_is_refused_rather_than_inverted() {
        // A zero diagonal block has lambda_max 0, which the shared body reports
        // as invalid. Returning a plausible inverse here would freeze that
        // vertex silently, because a zero preconditioner block trips no PCG
        // guard: it contributes exactly zero to r.z and pAp, never a negative.
        let rows = 4u32;
        let mut diagonal = vec![0.0f32; 9 * rows as usize];
        for i in 0..rows as usize - 1 {
            diagonal[9 * i] = 1.0;
            diagonal[9 * i + 4] = 1.0;
            diagonal[9 * i + 8] = 1.0;
        }
        let mut inverse = vec![0.0f32; 9 * rows as usize];
        assert!(
            !build_block_jacobi(&diagonal, &mut inverse, rows),
            "a zero diagonal block must be refused, not inverted"
        );
    }

    /// The refusal above, through the path the SOLVE actually takes.
    ///
    /// The test above exercises the shared body; this one exercises the
    /// dispatch, and the two are not the same walk: the driver does not invert
    /// row by row on the host, so a body that refuses correctly while the entry
    /// point drops the verdict would pass that test and ship a solve that
    /// silently froze a vertex. What is asserted here is the CHANNEL: a
    /// non-SPD block must come back as `Fault::Device`, counted, with the row
    /// in its payload.
    #[test]
    fn the_dispatched_inversion_reports_a_non_spd_block_through_the_diagnostic_lane() {
        use crate::driver::kernels::BlockJacobiInvertRowArgs;
        let rows = 4u32;
        let mut diagonal = vec![0.0f32; 9 * rows as usize];
        for i in 0..rows as usize - 1 {
            diagonal[9 * i] = 1.0;
            diagonal[9 * i + 4] = 1.0;
            diagonal[9 * i + 8] = 1.0;
        }
        let mut device = host_device();
        let diagonal_in = staged_block(&mut device, &diagonal, "test.bj_diagonal");
        let mut inverse_out = Buffer::<f32>::none();
        inverse_out
            .size(&mut device, 9 * rows as usize, AllocLabel("test.bj_inverse"))
            .expect("the test allocation succeeds");
        let args = BlockJacobiInvertRowArgs {
            diagonal: diagonal_in.handle(),
            inverse: inverse_out.handle(),
            count: rows,
            seam_arena_count: 0,
        };
        // Safety: both allocations outlive the dispatch and hold one block per row.
        let result = device.run("test.block_jacobi_invert", |encoder| unsafe {
            encoder.elements(&args, rows)
        });
        match result {
            Err(Fault::Device { diag, .. }) => {
                assert_eq!(diag.failures, 1, "exactly the one zero block is refused");
                let first = diag.first.expect("the refusal carries its payload");
                assert_eq!(
                    first.payload[3], (rows - 1) as f32,
                    "the payload names the row that was refused, got {first}"
                );
            }
            other => panic!("a zero diagonal block must be refused, got {other:?}"),
        }
    }
}

/// The extra vectors the locked solve needs beside [`Workspace`].
///
/// SEPARATE FROM `Workspace` so an unlocked scene allocates none of it. Every
/// buffer is sized once for the run and reused, as the rest of the driver's are.
#[derive(Default)]
pub struct LockedWorkspace {
    /// The affine feasible correction the projector builds, and the tangent
    /// increment the recurrence solves for on top of it. `x = q + zsol`.
    pub q: Buffer<f32>,
    pub zsol: Buffer<f32>,
}

impl LockedWorkspace {
    pub fn allocate<D: Device>(&mut self, device: &mut D, rows: u32) -> Result<(), Fault> {
        let n = 3 * rows as usize;
        self.q.size(device, n, ppf_cts_compute::AllocLabel("pcg.locked.q"))?;
        self.zsol
            .size(device, n, ppf_cts_compute::AllocLabel("pcg.locked.zsol"))?;
        Ok(())
    }
}

/// PCG under an aggregate lock.
///
/// THE SYSTEM IS NOT THE UNLOCKED ONE WITH EXTRA STEPS. Writing the Newton
/// system as `M dx = b` and splitting `dx = q + z` with `q` the affine feasible
/// correction the projector builds and `z` a tangent increment, the equation
/// this solves is `Q M Q z = Q (b - M q)`. So it seeds from `q`, measures its
/// residual against `||Q (b - M q)||`, and returns `q + zsol` rather than the
/// iterate directly. That is also what preserves the seeded-residual denominator
/// rule: `err0` is the CONSTRAINED residual after the exact known correction,
/// never `||b||`.
///
/// EVERY PROJECTION IS ITS OWN BOUNDARY, and that is why this is the
/// host-syncing solve rather than the device-resident graph-captured one: a
/// projection is three dispatches with a host readback between the frames and
/// the rows, so it cannot be folded into a neighboring encoder region.
///
/// # Safety
/// Every handle must name a live allocation sized for the scene, and the
/// projector must have been `prepare`d against the same positions this solve
/// runs at, with `locked.q` holding the correction that call produced.
#[allow(clippy::too_many_arguments)]
pub unsafe fn solve_locked<D: Device>(
    device: &mut D,
    op: &Operator,
    projector: &mut super::lock::Projector<'_>,
    inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    work: &mut Workspace,
    locked: &mut LockedWorkspace,
    // THE PRECONDITIONER, exactly as the unlocked `solve` takes it.
    //
    // The caller builds the hierarchy BEFORE it branches on whether the scene is
    // aggregate-locked and hands it to both arms, so a locked scene gets the
    // preconditioner it asked for. A
    // locked solve that took block-Jacobi regardless would answer
    // `precond = schwarz` with a label rather than a preconditioner, which is
    // the failure this seam is most exposed to: a preconditioner that is
    // compiled, correct and reached by nothing.
    mut schwarz: Option<&mut super::schwarz::State>,
    tolerance: f32,
    max_iterations: u32,
) -> super::scene::FatalResult<Report> {
    let rows = op.rows();
    let n = 3 * rows as usize;
    // `q` IS AN INPUT, already built by the caller's `Projector::prepare`. The
    // dispatcher prepares and then calls the solve, so the affine correction is
    // formed once against the positions the frames were built at rather than
    // rebuilt inside the loop.
    let q = locked.q.handle();
    let zsol = locked.zsol.handle();

    let Workspace {
        ap,
        r,
        z,
        p,
        product,
        magnitude,
        row_absolute,
        row_curvature,
        verdict,
        verdict_cause,
        // THE FOLD SCRATCH AND THE SCALAR SLOTS, so this path reduces on the
        // DEVICE as the unlocked one does. Downloading the whole vector and
        // folding it here would be a relocation of the computation rather than
        // transport of a result, which is what rule (1a-0) forbids.
        //
        // NAMED `scalars` BECAUSE `scalar` IS TAKEN further down, by the alpha
        // verdict's `let scalar = pcg_alpha(...)`. A second `scalar` here would
        // be shadowed from that line on and the slots would quietly stop being
        // reachable half way through the loop.
        fold,
        scalar: scalars,
        ..
    } = work;
    let mut terms = DotTerms { product, magnitude };
    // THE APPLY WRITES ITS MAGNITUDE SUM SOMEWHERE AND THIS PATH IGNORES IT.
    // The locked and reduced solves take their curvature bound from the
    // Cauchy-Schwarz surrogate instead, because the operator they call is
    // wrapped in projections and its per-row sum is not the sum that forms
    // their `p^T A p`.
    let discard_absolute = row_absolute.handle();
    let discard_curvature = row_curvature.handle();

    // r = Q (b - M q), zsol = 0.
    op.apply(device, q, ap.handle(), discard_absolute, discard_curvature)?;
    combine(device, b, ap.handle(), r.handle(), 1.0, -1.0, n)?;
    projector
        .project(device, r.handle(), 1)?;
    combine(device, zsol, zsol, zsol, 0.0, 0.0, n)?;

    let err_slot = scalars.span(slot::ERR, 1);
    device.run("pcg.locked.err0", |encoder| {
        // Safety: `r` and the fold scratch are borrowed for the whole call.
        unsafe { encode_fold(encoder, r.handle(), n, fold, err_slot, true) }
    })?;
    scalars.download(device)?;
    let err0 = scalars.host()[slot::ERR];
    if err0 == 0.0 {
        // Already feasible and already solved: the answer is q itself.
        combine(device, q, q, x, 1.0, 0.0, n)?;
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 0,
            relative_residual: 0.0,
        });
    }

    device.run("pcg.locked.precondition", |encoder| {
        // Safety: every buffer is borrowed for the whole call.
        unsafe {
            match schwarz.as_deref_mut() {
                Some(state) => super::schwarz::encode_whole_sweep(
                    encoder, state, r.handle(), z.handle(), rows),
                None => encode_preconditioner(
                    encoder, inverse_diagonal, r.handle(), z.handle(), rows),
            }
        }
    })?;
    projector
        .project(device, z.handle(), 1)?;
    let mut rz = terms.dot(device, r.handle(), z.handle(), n, fold, scalars)?;
    // The negation traps a NaN, which `<= 0.0` would let through; the idiom is
    // the unlocked solve's.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(rz > 0.0) {
        return Ok(Report {
            outcome: Outcome::NonSpdPreconditioner,
            iterations: 0,
            relative_residual: 1.0,
        });
    }
    combine(device, z.handle(), p.handle(), p.handle(), 1.0, 0.0, n)?;
    projector
        .project(device, p.handle(), 1)?;

    // `x = q + zsol`, which every exit below takes.
    let finish = |device: &mut D, zsol: ppf_cts_compute::Handle| -> Result<(), Fault> {
        combine(device, q, zsol, x, 1.0, 1.0, n)
    };


    for iteration in 1..=max_iterations {
        // Ap = Q M Q p. `p` is projected after every update, and projecting it
        // again here makes the identity explicit and limits accumulated fp32
        // drift.
        projector
            .project(device, p.handle(), 1)?;
        op.apply(device, p.handle(), ap.handle(), discard_absolute, discard_curvature)?;
        projector
            .project(device, ap.handle(), 1)?;

        device.run("pcg.locked.terms", |encoder| {
            // Safety: `p` and `ap` are borrowed for the whole call.
            unsafe { terms.encode_fill(encoder, p.handle(), ap.handle(), n) }
        })?;
        let pap_slot = scalars.span(slot::P_AP, 1);
        let product_h = terms.product.handle();
        device.run("pcg.locked.pap", |encoder| {
            // Safety: the product vector and the scratch are borrowed for the
            // whole call.
            unsafe {
                encode_fold(encoder, product_h, rows as usize, fold, pap_slot, false)
            }
        })?;
        scalars.download(device)?;
        let pap = scalars.host()[slot::P_AP];
        // As the unlocked solve: `|p|_2 |Ap|_2`, read after `pap` because it
        // reuses the same fold buffers.
        let absolute_dot =
            terms.norm_product(device, p.handle(), ap.handle(), n, fold, scalars)?;
        let scalar = pcg_alpha(device, verdict, verdict_cause, rz, pap, absolute_dot)?;
        if scalar.cause != PCG_BREAK_NONE {
            let outcome = if scalar.cause == PCG_BREAK_PAP {
                Outcome::IndefiniteMatrix
            } else {
                // Curvature below the round-off of its own reduction: truncate
                // Steihaug-style with the direction fp32 can still see, and
                // report it as a SUCCESS, since the direction so far is still a
                // descent direction.
                projector
                    .project(device, zsol, 3)?;
                finish(device, zsol)?;
                let reresid = residual_of(device, r, err0)?;
                // As on the unlocked path above, and for the same reason: a
                // truncation that returns success and is never mentioned makes
                // a step that stopped early read exactly like one that
                // converged.
                ::log::info!(
                    "* cg truncated: curvature within round-off of its own \
                     reduction at iter {} (reresid {:.3e})",
                    iteration,
                    reresid
                );
                return Ok(Report {
                    outcome: Outcome::CurvatureTruncated,
                    iterations: iteration,
                    relative_residual: reresid,
                });
            };
            return Ok(Report {
                outcome,
                iterations: iteration,
                relative_residual: 1.0,
            });
        }

        let alpha = scalar.value;
        device.run("pcg.locked.update", |encoder| {
            // Safety: `p`, `zsol`, `ap` and `r` are borrowed for the whole call
            // and the two updates touch disjoint destinations.
            unsafe {
                encode_add_scaled(encoder, p.handle(), zsol, alpha, n)?;
                encode_add_scaled(encoder, ap.handle(), r.handle(), -alpha, n)
            }
        })?;
        projector
            .project(device, r.handle(), 1)?;

        let resid_slot = scalars.span(slot::ERR, 1);
        device.run("pcg.locked.residual", |encoder| {
            // Safety: `r` and the fold scratch are borrowed for the whole call.
            unsafe { encode_fold(encoder, r.handle(), n, fold, resid_slot, true) }
        })?;
        scalars.download(device)?;
        let residual = scalars.host()[slot::ERR] / err0;
        if residual < tolerance {
            projector
                .project(device, zsol, 3)?;
            finish(device, zsol)?;
            return Ok(Report {
                outcome: Outcome::Converged,
                iterations: iteration,
                relative_residual: residual,
            });
        }
        if iteration >= max_iterations || !residual.is_finite() {
            projector
                .project(device, zsol, 3)?;
            finish(device, zsol)?;
            return Ok(Report {
                outcome: Outcome::MaxIterations,
                iterations: iteration,
                relative_residual: residual,
            });
        }

        device.run("pcg.locked.precondition", |encoder| {
            // Safety: as above.
            unsafe {
                match schwarz.as_deref_mut() {
                    Some(state) => super::schwarz::encode_whole_sweep(
                        encoder, state, r.handle(), z.handle(), rows),
                    None => encode_preconditioner(
                        encoder, inverse_diagonal, r.handle(), z.handle(), rows),
                }
            }
        })?;
        projector
            .project(device, z.handle(), 1)?;
        let rz_next = terms.dot(device, r.handle(), z.handle(), n, fold, scalars)?;
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(rz_next > 0.0) {
            return Ok(Report {
                outcome: Outcome::NonSpdPreconditioner,
                iterations: iteration,
                relative_residual: residual,
            });
        }
        let beta = rz_next / rz;
        combine(device, z.handle(), p.handle(), p.handle(), 1.0, beta, n)?;
        projector
            .project(device, p.handle(), 1)?;
        rz = rz_next;
    }

    projector
        .project(device, zsol, 3)?;
    finish(device, zsol)?;
    Ok(Report {
        outcome: Outcome::MaxIterations,
        iterations: max_iterations,
        relative_residual: 1.0,
    })
}

/// The vectors the PDRD reduced solve needs beside [`Workspace`].
///
/// SIZED ONCE FOR THE RUN and reused by every call, which is the rule the
/// driver is held to: `xv` and `mxv` in particular are shared by every
/// application of the reduced operator within one solve, so they are per-solve
/// state rather than per-call temporaries.
#[derive(Default)]
pub struct RigidWorkspace {
    /// Reduced, `dim` floats each.
    pub f: Buffer<f32>,
    pub xr: Buffer<f32>,
    pub r: ReadbackBuffer<f32>,
    pub z: Buffer<f32>,
    pub p: Buffer<f32>,
    pub rp: Buffer<f32>,
    /// Full space, `3 * nrow` floats each: the operator's own scratch.
    pub xv: Buffer<f32>,
    pub mxv: Buffer<f32>,
    /// One `vec![0.0; ...]` the reduction's clears write from, so no clear
    /// allocates.
    pub zero: Vec<f32>,
    /// The per-group L1 norms, seeded and current.
    /// The per-group L1 norms as the DEVICE writes them: slot 0 is the cloth
    /// block and slots 1.. are the bodies. Only these `1 + bodies` floats cross
    /// the seam; the reduced vector itself stays on the device.
    pub group_norm: ReadbackBuffer<f32>,
    pub group_initial: Vec<f32>,
    pub group_current: Vec<f32>,
    /// THE REDUCED SOLVE NEEDS ITS OWN DOT TERMS, sized to `dim` rather than to
    /// `3 * rows`. `DotTerms::fill` takes the extent from its own buffer's
    /// length and ignores the `n` it is passed, so the main workspace's terms
    /// would fold over the FULL vertex length and read past the end of every
    /// reduced vector. That is not a stale-tail question: the products
    /// themselves would be computed from whatever lies beyond `dim`.
    pub product: ReadbackBuffer<f32>,
    pub magnitude: ReadbackBuffer<f32>,
}

impl RigidWorkspace {
    pub fn allocate<D: Device>(
        &mut self,
        device: &mut D,
        dim: usize,
        rows: usize,
        bodies: usize,
    ) -> Result<(), Fault> {
        use ppf_cts_compute::AllocLabel;
        self.f.size(device, dim, AllocLabel("pdrd.cg.f"))?;
        self.xr.size(device, dim, AllocLabel("pdrd.cg.xr"))?;
        self.r.size(device, dim, AllocLabel("pdrd.cg.r"))?;
        self.z.size(device, dim, AllocLabel("pdrd.cg.z"))?;
        self.p.size(device, dim, AllocLabel("pdrd.cg.p"))?;
        self.rp.size(device, dim, AllocLabel("pdrd.cg.rp"))?;
        self.xv.size(device, 3 * rows, AllocLabel("pdrd.cg.xv"))?;
        self.mxv.size(device, 3 * rows, AllocLabel("pdrd.cg.mxv"))?;
        // SIZED IN ROWS OF THREE, NOT IN FLOATS. `DotTerms::fill` takes its
        // extent from this buffer's own length and each thread folds THREE
        // components, which is why the main workspace sizes its pair by the
        // VERTEX count. Sizing them by `dim` makes every fold read three times
        // past the end of a reduced vector, and the products come back too
        // large by a factor that grows as the solve proceeds. `dim` is
        // `3 * n_cloth + 6 * n_bodies` and so is always a multiple of three.
        debug_assert_eq!(dim % 3, 0, "the reduced dimension is a multiple of three");
        self.product
            .size(device, dim / 3, ppf_cts_compute::AllocLabel("pdrd.cg.product"))?;
        self.magnitude
            .size(device, dim / 3, ppf_cts_compute::AllocLabel("pdrd.cg.magnitude"))?;
        self.zero = vec![0.0; dim.max(3 * rows).max(36 * bodies)];
        self.group_norm.size(
            device,
            1 + bodies,
            ppf_cts_compute::AllocLabel("pdrd.cg.group_norm"),
        )?;
        self.group_initial = vec![0.0; 1 + bodies];
        self.group_current = vec![0.0; 1 + bodies];
        Ok(())
    }
}

/// PCG on the PDRD reduced system.
///
/// IT SOLVES `R u = P^T b` WITH `R = P^T M P`, and `R` is never formed: each
/// application prolongs the reduced vector to the full one, applies the
/// assembled per-vertex Newton operator, restricts the result back, and
/// projects each body's six DOFs onto its joint and lock subspace.
///
/// THE TOLERANCE IS MEASURED PER DOF GROUP, which is the property this solve
/// most easily loses and the reason `super::pdrd::worst_relative_residual`
/// exists. Collapsing it to one norm over the reduced vector lets a heavy body,
/// whose 6x6 is exactly preconditioned and dies in one step, carry the whole
/// residual across the tolerance while the cloth is untouched.
///
/// IT IS SEEDED FROM THE CALLER'S GUESS rather than zeroed, and through
/// `seed_restrict` rather than the force restriction. Every prescribed row of
/// `x` carries its exact Dirichlet correction, and the whole system routes
/// through here as soon as any PDRD body exists; a memset would throw that seed
/// away, leaving a prescribed row converging only to PCG tolerance and putting
/// its displacement into `err0` beside genuine force rows, which is the
/// pin-dominated denominator that must never be the tolerance's scale.
///
/// # Safety
/// Every handle must name a live allocation sized for the scene, the reduction
/// must have been given this iteration's rotated rest vectors, and the
/// preconditioner must have been built against the matrices `op` applies.
#[allow(clippy::too_many_arguments)]
/// The per-group L1 norms of `reduced`, FOLDED ON THE DEVICE.
///
/// Three dispatches form them: an `fabsf` map over the cloth rows, a device
/// reduction of that into slot 0, and one thread per body summing its own six
/// rows into slot `1 + b`. Only the `1 + bodies` results cross the seam.
/// Downloading the whole reduced vector and folding it on the host instead
/// would relocate the computation rather than transport its result, which is
/// what rule (1a-0) forbids.
///
/// THE CLOTH FOLD IS SKIPPED WHEN THERE IS NO CLOTH BLOCK and a zero is written
/// instead: `encode_fold` has no meaning over zero rows.
unsafe fn group_l1_device<D: Device>(
    device: &mut D,
    reduced: ppf_cts_compute::Handle,
    norms: &mut ReadbackBuffer<f32>,
    fold: &Buffer<f32>,
    body_base: usize,
    bodies: usize,
    out: &mut [f32],
) -> Result<(), Fault> {
    debug_assert_eq!(out.len(), 1 + bodies);
    let cloth_slot = norms.span(0, 1);
    let body_slot = norms.span(1, bodies.max(1));
    device.run("pcg.rigid.group_l1", |encoder| {
        if body_base > 0 {
            // Safety: `reduced` and the scratch outlive the call.
            unsafe { encode_fold(encoder, reduced, body_base, fold, cloth_slot, true) }?;
        }
        if bodies > 0 {
            let args = PcgRigidGroupL1Args {
                reduced,
                body_base: body_base as u32,
                norm: body_slot,
                count: bodies as u32,
                seam_arena_count: 0,
            };
            // Safety: both handles name live allocations for the whole call.
            unsafe { encoder.elements(&args, bodies as u32) }?;
        }
        Ok(())
    })?;
    norms.download(device)?;
    let host = norms.host();
    out[0] = if body_base > 0 { host[0] } else { 0.0 };
    for body in 0..bodies {
        out[1 + body] = host[1 + body];
    }
    Ok(())
}

pub unsafe fn solve_rigid<D: Device>(
    device: &mut D,
    op: &Operator,
    reduction: &mut super::pdrd::Reduction<'_>,
    factor: ppf_cts_compute::Handle,
    inverse_diagonal: ppf_cts_compute::Handle,
    b: ppf_cts_compute::Handle,
    x: ppf_cts_compute::Handle,
    work: &mut Workspace,
    rigid: &mut RigidWorkspace,
    rotation_out: Option<ppf_cts_compute::Handle>,
    // THE PROJECTED LOCK, when the scene carries one.
    //
    // A LOCKED REDUCED SOLVE IS THE SAME LOOP, not a second algorithm. What
    // differs is that the reduced operator projects the FULL-SPACE product
    // before restricting it, and that the seed carries the lock's particular
    // solution. Composing the two projectors in the other order would not be the
    // same operator, which is why the lock's projection sits inside the reduced
    // apply rather than around it.
    lock: Option<&mut super::lock::Projector<'_>>,
    // The scene's `TranslationLock` records, which the particular term reads
    // for each locked body's total mass.
    locks: ppf_cts_compute::Handle,
    tolerance: f32,
    max_iterations: u32,
) -> super::scene::FatalResult<Report> {
    let dim = reduction.map.dim;
    let bodies = reduction.map.n_bodies;
    let body_base = reduction.map.body_base;

    let mut lock = lock;

    // THE TANGENT PROJECTION. It is the projector every Krylov vector in a
    // LOCKED solve passes through, and it is not the same thing as
    // `project_bodies`.
    //
    // THE TWO HALVES LIVE IN DIFFERENT SPACES, which is the whole reason it has
    // five steps. A body's constraint rows are expressed in the REDUCED basis,
    // so `project_bodies` handles them; the aggregate lock's rows are built over
    // VERTICES, so they can only be applied in the full space. The vector is
    // therefore projected in the reduced basis, prolonged, projected again in
    // the full one, and only its CLOTH rows brought back, because the body rows
    // were already correct and the prolonged image would overwrite them.
    //
    // WITHOUT A LOCK IT DEGENERATES TO `project_bodies`, which is why the
    // unlocked arm is untouched by this: the middle three steps would prolong,
    // apply nothing, and copy the cloth rows back unchanged.
    macro_rules! project_tangent {
        ($vector:expr) => {{
            reduction.project_bodies(device, $vector)?;
            if let Some(projector) = lock.as_deref_mut() {
                reduction.prolong(device, $vector, rigid.xv.handle())?;
                projector.project(device, rigid.xv.handle(), 1)?;
                reduction.copy_projected_cloth(device, rigid.xv.handle(), $vector)?;
                reduction.project_bodies(device, $vector)?;
            }
        }};
    }

    // `R v` WITHOUT projecting the input. It has exactly one caller, the affine
    // particular solution, whose whole purpose is to carry the INHOMOGENEOUS
    // part that the tangent projector would remove. Every other apply in this
    // function is the tangent one below.
    macro_rules! apply_reduced_affine {
        ($input:expr, $output:expr) => {{
            reduction.prolong(device, $input, rigid.xv.handle())?;
            op.apply(device, rigid.xv.handle(), rigid.mxv.handle(),
                     work.row_absolute.handle(), work.row_curvature.handle())?;
            // THE LOCK PROJECTS THE FULL-SPACE PRODUCT, before the restriction
            // takes it back to the reduced vector. The order is not
            // interchangeable: the lock's rows are built over VERTICES, so
            // projecting the reduced vector instead would apply it in a basis it
            // was not built in.
            if let Some(projector) = lock.as_deref_mut() {
                projector.project(device, rigid.mxv.handle(), 1)?;
            }
            reduction.restrict(device, rigid.mxv.handle(), $output, &rigid.zero)?;
            reduction.project_bodies(device, $output)?;
        }};
    }

    // `R v` WITH its input projected first, and the only operator the CG loop
    // may use.
    //
    // PROJECTING BOTH SIDES IS WHAT MAKES IT SYMMETRIC. The loop's operator is
    // `Q M Q`; projecting only the output gives `Q M`, which is NOT symmetric
    // whenever `Q` is not the identity, and conjugate gradients on a
    // non-symmetric operator has no descent guarantee and no reason to converge.
    // Without a lock `project_tangent` is `project_bodies`, every Krylov vector
    // already carries it, and the two applies agree.
    macro_rules! apply_reduced_tangent {
        ($input:expr, $output:expr) => {{
            project_tangent!($input);
            apply_reduced_affine!($input, $output);
        }};
    }

    // f = Pi P^T b.
    reduction.restrict(device, b, rigid.f.handle(), &rigid.zero)?;
    reduction.project_bodies(device, rigid.f.handle())?;

    // xr = Pi S x, the seed, then r = f - R xr.
    reduction.seed_restrict(device, x, rigid.xr.handle(), &rigid.zero)?;
    // THE LOCK'S PARTICULAR SOLUTION, BEFORE THE BODY PROJECTION. A locked
    // body's three translation DOFs are not free: the lock's drift over its
    // total mass IS their value, so the seed carries it and the CG solves only
    // the homogeneous part. It is written between the seed's restriction and the
    // first residual, so the residual is measured against a seed that already
    // satisfies the lock.
    if let Some(projector) = lock.as_deref_mut() {
        reduction.translation_lock_particular(
            device,
            locks,
            projector.scratch.drift.handle(),
            rigid.xr.handle(),
        )?;
    }
    reduction.project_bodies(device, rigid.xr.handle())?;
    // THE AFFINE APPLY, and the one place it is used. The seed carries the
    // lock's inhomogeneous particular correction, which the tangent projector is
    // built to remove, so projecting the input here would discard exactly the
    // part the residual has to see.
    apply_reduced_affine!(rigid.xr.handle(), rigid.rp.handle());
    combine(device, rigid.f.handle(), rigid.rp.handle(), rigid.r.handle(), 1.0, -1.0, dim)?;
    // The seed residual is projected before `err0` is taken from it, for the
    // reason the per-iteration projection below states.
    project_tangent!(rigid.r.handle());

    // Safety: every handle names a live allocation for the whole call.
    unsafe {
        group_l1_device(
            device,
            rigid.r.handle(),
            &mut rigid.group_norm,
            &work.fold,
            body_base,
            bodies,
            &mut rigid.group_initial,
        )
    }?;
    let err0: f32 = rigid.group_initial.iter().sum();
    if err0 == 0.0 {
        // Already exact. The seed is prolonged back out rather than `x` being
        // zeroed, which would discard the prescribed rows the caller loaded.
        reduction.prolong(device, rigid.xr.handle(), x)?;
        return Ok(Report {
            outcome: Outcome::Converged,
            iterations: 1,
            relative_residual: 0.0,
        });
    }

    // THE REDUCED WORKSPACE'S OWN TERMS, not the main one's: the fold's extent
    // is its buffer's length, so folding with the vertex-sized pair would read
    // past the end of every reduced vector here.
    let RigidWorkspace { product, magnitude, .. } = rigid;
    let mut terms = DotTerms { product, magnitude };

    reduction.apply_precond(device, factor, inverse_diagonal, rigid.r.handle(), rigid.z.handle())?;
    // The preconditioner is not the projector's, so its output re-enters the
    // constrained subspace only by this call.
    project_tangent!(rigid.z.handle());
    combine(device, rigid.z.handle(), rigid.p.handle(), rigid.p.handle(), 1.0, 0.0, dim)?;
    // The first search direction is projected in its own right, not left to
    // inherit `z`'s projection through the combine.
    project_tangent!(rigid.p.handle());
    let mut rz =
        terms.dot(device, rigid.r.handle(), rigid.z.handle(), dim, &work.fold, &mut work.scalar)?;

    // `x = P u` and the rotation export, which every exit below takes.
    macro_rules! finish {
        ($outcome:expr, $iteration:expr, $residual:expr) => {{
            // EVERY EXIT PROJECTS THE SOLUTION, not only the converged one: a
            // truncated or capped solve returns an iterate the caller applies,
            // so it owes the same feasibility the converged one does.
            project_tangent!(rigid.xr.handle());
            reduction.prolong(device, rigid.xr.handle(), x)?;
            if let Some(rotation) = rotation_out {
                reduction.extract_body_rotation(device, rigid.xr.handle(), rotation)?;
            }
            // THE REDUCED SOLVE'S OWN REPORT. It
            // names the WORST DOF GROUP's relative residual rather than a norm
            // over the whole reduced vector, which is the whole point of the
            // per-group measure: one heavy body's exactly-preconditioned block
            // is annihilated in a single step, so a global norm crosses the
            // tolerance while the cloth still carries its full residual.
            ::log::info!(
                "reduced PDRD solve took {} iteration(s), worst DOF group at a \
                 relative residual of {:.9}{}",
                $iteration,
                $residual,
                if matches!($outcome, Outcome::CurvatureTruncated) {
                    " (curvature truncated)"
                } else {
                    ""
                }
            );
            return Ok(Report {
                outcome: $outcome,
                iterations: $iteration,
                relative_residual: $residual,
            });
        }};
    }

    for iteration in 1..=max_iterations {
        apply_reduced_tangent!(rigid.p.handle(), rigid.rp.handle());
        let prp = terms.dot(
            device, rigid.p.handle(), rigid.rp.handle(), dim, &work.fold, &mut work.scalar)?;
        // A ZERO CURVATURE DIVIDES BY 1 RATHER THAN BY ZERO, which keeps the
        // step finite and lets the residual test below decide. It is NOT treated
        // as a breakdown here: the reduced operator is projected on both sides,
        // so an exactly zero `p^T R p` is a direction the constraints have
        // removed rather than an indefinite matrix.
        let alpha = rz / if prp != 0.0 { prp } else { 1.0 };
        device.run("pdrd.cg.update", |encoder| {
            // Safety: the four buffers are borrowed for the whole call and the
            // two updates touch disjoint destinations.
            unsafe {
                encode_add_scaled(encoder, rigid.p.handle(), rigid.xr.handle(), alpha, dim)?;
                encode_add_scaled(encoder, rigid.rp.handle(), rigid.r.handle(), -alpha, dim)
            }
        })?;

        // Before the residual is measured: the norm has to be taken over the
        // PROJECTED residual, or a component the constraint removes is counted
        // against the tolerance and the solve never reaches it.
        project_tangent!(rigid.r.handle());
        // Safety: every handle names a live allocation for the whole call.
        unsafe {
            group_l1_device(
                device,
                rigid.r.handle(),
                &mut rigid.group_norm,
                &work.fold,
                body_base,
                bodies,
                &mut rigid.group_current,
            )
        }?;
        let residual = super::pdrd::worst_relative_residual(
            &rigid.group_current,
            &rigid.group_initial,
            err0,
        );
        if residual < tolerance {
            finish!(Outcome::Converged, iteration, residual);
        }
        if iteration >= max_iterations || !residual.is_finite() {
            finish!(Outcome::MaxIterations, iteration, residual);
        }

        reduction.apply_precond(
            device,
            factor,
            inverse_diagonal,
            rigid.r.handle(),
            rigid.z.handle(),
        )?;
        // As at the seed: the preconditioner's output re-enters the constrained
        // subspace only by this call.
        project_tangent!(rigid.z.handle());
        let rz_next = terms.dot(
            device, rigid.r.handle(), rigid.z.handle(), dim, &work.fold, &mut work.scalar)?;
        let beta = rz_next / rz;
        combine(
            device,
            rigid.z.handle(),
            rigid.p.handle(),
            rigid.p.handle(),
            1.0,
            beta,
            dim,
        )?;
        // The new search direction in its own right, as at the seed.
        project_tangent!(rigid.p.handle());
        rz = rz_next;
    }

    let _ = work;
    finish!(Outcome::MaxIterations, max_iterations, 1.0);
}
