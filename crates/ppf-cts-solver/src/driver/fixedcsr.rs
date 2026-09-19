// File: crates/ppf-cts-solver/src/driver/fixedcsr.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! `FixedCSRMat`: the topology-fixed half of the Newton matrix.
//!
//! Two of these exist per step (`super::step`): `fixed_hess` carries elastic,
//! stitch, strain-limit and fix-pin blocks, and `tmp_fixed` is the snapshot of
//! it the contact assembly reads as its stiffness reference.
//! ON THIS BACKEND `tmp_fixed` IS A REAL SECOND BUFFER. Host memory is cheap, so
//! it is allocated and copied the way CUDA does it; Metal's evaluate/scatter
//! split exists only because Metal could not afford the second matrix, and
//! copying that shape here would import a constraint this backend does not have.
//!
//! # This module owns no arithmetic, and that is the whole design
//!
//! Every value in the matrix arrives through a shared body:
//! `csrmat/fixed_csr.kernel.cpp` finds the slot and folds the block in, and
//! `solver/spmv.kernel.cpp` multiplies a row. What lives here is the pattern
//! bookkeeping, the allocation, and the validation below. There is no `f32`
//! operator anywhere outside `#[cfg(test)]`.
//!
//! # The pattern is built by `builder.rs` and is READ here, never derived
//!
//! `DataSet::fixed_index_table` (`builder.rs:1440`) is the sparsity: row `i`
//! holds the ascending column indices `j >= i` that any element stencil can
//! write. `DataSet::transpose_table` (`builder.rs:1455`) is its mirror: row `j`
//! holds `(i, slot)` for every stored off-diagonal block whose column is `j`.
//! Both are `repr(C)` `CVecVec`s the host already owns, so this module borrows
//! them and allocates only the values.
//!
//! Deriving the transpose here instead would be a second implementation of a
//! table the scene build already emits, and the two could disagree without any
//! test noticing: the matrix would still be symmetric-looking and every block
//! would still be present, only the SpMV's lower-triangle contribution would be
//! wrong. That is why [`FixedCsr::new`] CROSS-CHECKS the two tables against each
//! other instead.
//!
//! # Why the cross-check is not paranoia
//!
//! Only the upper triangle is stored (`i <= j`), so `M x` reads each stored
//! block twice: once as itself for row `i`, and once transposed for row `j`.
//! The second read is driven entirely by `transpose_table`. A missing entry
//! silently drops the lower-triangle coupling, which leaves the operator
//! non-symmetric while every block is still in the matrix. The PCG guards
//! cannot see that: `p^T A p` of a non-symmetric operator is the quadratic form
//! of its symmetric part, which stays positive. It surfaces only as a Newton
//! direction that is quietly wrong, which is precisely the failure class this
//! backend has no hardware fault to catch.

// The container is landed ahead of the assembly and driver that consume it; the
// allow comes off in the change that wires them.
#![allow(dead_code)]

use crate::cvecvec::CVecVec;
use crate::data::{DataSet, Mat3x3f, Vec2u};

use ppf_cts_compute::{AllocLabel, Device, ReadbackBuffer};
use super::kernels::FixedCsrAtomicPushArgs;
use super::scene::{Fatal, FatalResult};
use super::spmv::FixedCsrView;

// THE TWO ENTRY POINTS THIS MODULE STILL NAMES DIRECTLY, AND WHY EACH STAYS.
// The push, which is the one pass here that has a thread index, went behind the
// seam; what is left is two SINGLE-PAIR QUERIES, and neither is a dispatch.
//
// Both answer a question about one `(i, j)`: which slot holds it, and what that
// slot holds. There is no extent, no thread index and nothing indexed by one,
// which is the same shape as `block_jacobi_invert_abi` in `super::pcg` and
// `position_domain_abi` in `super::step`, both kept as direct calls for the
// same reason. `find` also RETURNS a value, and the seam's dispatch returns
// none, so putting it behind one would need an output buffer, which is a change
// to the C++ side rather than to this driver.
//
// A second reason applies to `read` and is worth stating separately, because it
// would survive a query surface on `Device`: it is the ORACLE this module's own
// tests check `push_blocks` with. A dispatch verified against a reader that
// went through the same dispatch machinery would be checking that machinery
// against itself. If `Device` ever grows a query, these two move with the two
// precedents named above and not one at a time, and the oracle question is
// settled then rather than assumed now.
extern "C" {
    fn fixed_csr_find_abi(
        index: *const u32,
        offset: *const u32,
        row_count: u32,
        i: u32,
        j: u32,
        transpose_out: *mut i32,
    ) -> u32;
    fn fixed_csr_read_abi(
        index: *const u32,
        offset: *const u32,
        value: *const f32,
        row_count: u32,
        i: u32,
        j: u32,
        out: *mut f32,
    );
}

/// The nine floats of one 3x3 block, column-major, as the shared bodies store
/// them.
///
/// Named rather than spelled `[f32; 9]` at each call site so the layout claim
/// has one home. `Mat3x3f` is `na::Matrix3<f32>`, whose storage is exactly this,
/// and [`FixedCsr::push_blocks`] asserts that rather than assuming it.
pub const BLOCK_FLOATS: usize = 9;

/// The sparsity pattern, borrowed from the scene's own tables.
///
/// Separate from the values because the two have different lifetimes and
/// different owners: the pattern is built once at scene build and never changes,
/// while a step allocates two independent value arrays over it.
#[derive(Clone, Copy)]
pub struct FixedPattern<'a> {
    /// Column indices, row-major, ascending within a row, all `>= ` their row.
    pub index: &'a [u32],
    /// `rows + 1` entries; row `i` spans `offset[i] .. offset[i + 1]`.
    pub offset: &'a [u32],
    /// `(source_row, slot)` pairs, flattened, one pair per stored off-diagonal
    /// block, grouped by the block's COLUMN.
    pub transpose_pair: &'a [u32],
    /// `rows + 1` entries into `transpose_pair`, counted in PAIRS.
    pub transpose_offset: &'a [u32],
    /// One row per vertex.
    pub rows: u32,
}

/// The pattern's four arrays as the handles the state already holds.
///
/// A `FixedCsr` owns its values and borrows its pattern, and the pattern's
/// device copies live in `SolverState`, which the matrix cannot borrow because
/// it is handed to code that holds the state mutably. So the handles arrive as
/// an argument, exactly as `MeshRefs` does for the contact passes.
#[derive(Clone, Copy)]
pub struct FixedPatternRefs {
    pub index: ppf_cts_compute::Handle,
    pub offset: ppf_cts_compute::Handle,
    pub transpose_pair: ppf_cts_compute::Handle,
    pub transpose_offset: ppf_cts_compute::Handle,
    pub rows: u32,
}

/// Identity of the pattern a [`validate`] call covered.
///
/// The four slices' base address and length. Two wraps over the same borrowed
/// `DataSet` tables produce the same value, and any other pattern is a
/// different allocation or a different length, so a cached verdict cannot be
/// read for a pattern it was not taken on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PatternFingerprint {
    index: (usize, usize),
    offset: (usize, usize),
    transpose_pair: (usize, usize),
    transpose_offset: (usize, usize),
    rows: u32,
}

impl PatternFingerprint {
    fn of(pattern: &FixedPattern<'_>) -> Self {
        fn span(slice: &[u32]) -> (usize, usize) {
            (slice.as_ptr() as usize, slice.len())
        }
        Self {
            index: span(pattern.index),
            offset: span(pattern.offset),
            transpose_pair: span(pattern.transpose_pair),
            transpose_offset: span(pattern.transpose_offset),
            rows: pattern.rows,
        }
    }
}

impl<'a> FixedPattern<'a> {
    /// Number of stored blocks.
    pub fn nnz(&self) -> usize {
        self.index.len()
    }
}

/// One fixed-pattern matrix: the borrowed pattern plus its own values.
pub struct FixedCsr<'a> {
    pattern: FixedPattern<'a>,
    /// The same pattern as device handles.
    ///
    /// OWNED RATHER THAN PASSED, and the difference is not cosmetic: a push
    /// looks a block up in the pattern, so the pattern it looks up in must be
    /// the one this matrix was BUILT over. Taking it as a parameter would let a
    /// caller hand in the state's pattern while the matrix held a different
    /// one, silently bypassing the dropped-block detection that exists to
    /// catch a lost Hessian coupling.
    refs: FixedPatternRefs,
    /// `9 * nnz` floats, column-major per block, in the pattern's slot order.
    ///
    /// KERNEL-WRITTEN AND HOST-READ, which is what picks the type: the assembly
    /// and `dirichlet::lift` write it from kernels, and `read` and `values`
    /// answer the host off the mirror. A read that has missed a download panics
    /// naming this buffer rather than answering out of the previous step.
    value: ReadbackBuffer<f32>,
}

// Shape only. A matrix over a real scene carries millions of floats, and a
// `{:?}` that printed them would turn a one-line assertion failure into an
// unreadable dump; the two numbers below are what identifies a matrix.
impl std::fmt::Debug for FixedCsr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FixedCsr {{ rows: {}, stored blocks: {} }}",
            self.pattern.rows,
            self.nnz()
        )
    }
}

impl<'a> FixedCsr<'a> {
    /// Borrow the scene's pattern, validate it, and allocate a zeroed matrix.
    ///
    /// The validation runs once per matrix rather than per step because the
    /// pattern is immutable after scene build. It is exhaustive on purpose: an
    /// index error here is not a crash, it is a wrong answer with the right
    /// shape, and every check below costs one pass over a table that is already
    /// in cache.
    ///
    /// # Safety
    /// `data` must point at a live `DataSet` that outlives `'a`.
    pub unsafe fn from_dataset<D: Device>(
        device: &mut D,
        refs: FixedPatternRefs,
        data: &'a DataSet,
    ) -> FatalResult<Self> {
        let pattern = borrow_pattern(&data.fixed_index_table, &data.transpose_table)?;
        Self::new(device, refs, pattern)
    }

    /// Adopt an existing value array over the scene's pattern.
    ///
    /// The Newton driver holds its two matrices' value arrays for the whole run
    /// and wraps them fresh each step, because the pattern is borrowed from the
    /// `DataSet` and a matrix that owned that borrow could not live in a static.
    /// So the ALLOCATION is reused and only the wrapper is rebuilt.
    ///
    /// The pattern is re-validated on every wrap. That is one pass over the two
    /// offset tables per matrix per step, against an assembly that walks every
    /// element: cheap, and the alternative is a validated-once flag that a
    /// future caller could set on a different pattern.
    ///
    /// # Safety
    /// `data` must point at a live `DataSet` that outlives `'a`.
    pub unsafe fn adopt_from_dataset<D: Device>(
        device: &mut D,
        refs: FixedPatternRefs,
        data: &'a DataSet,
        value: ReadbackBuffer<f32>,
    ) -> FatalResult<Self> {
        let pattern = borrow_pattern(&data.fixed_index_table, &data.transpose_table)?;
        validate(&pattern)?;
        Self::wrap(device, refs, pattern, value)
    }

    /// As [`Self::adopt_from_dataset`], skipping the pattern walk when this
    /// exact pattern has already been validated.
    ///
    /// [`validate`] walks every stored block, so it is one pass over the whole
    /// sparsity. The Newton driver wraps its two matrices once per ITERATION,
    /// which spent that pass thousands of times per run on a pattern that
    /// cannot have changed: measured at 19.0 s of a 171 s `trapped` run.
    ///
    /// It cannot have changed because `advance` takes `&DataSet`, so the tables
    /// the pattern borrows are immutable for the whole run. The fingerprint is
    /// belt and braces rather than the argument: a caller that wraps a
    /// DIFFERENT pattern gets a different base address or length and is
    /// validated in full, so a cached verdict is never read for a pattern it
    /// was not taken on.
    ///
    /// # Safety
    /// As [`Self::adopt_from_dataset`].
    pub unsafe fn adopt_validated<D: Device>(
        device: &mut D,
        refs: FixedPatternRefs,
        data: &'a DataSet,
        value: ReadbackBuffer<f32>,
        checked: &mut Option<PatternFingerprint>,
    ) -> FatalResult<Self> {
        let pattern = borrow_pattern(&data.fixed_index_table, &data.transpose_table)?;
        let fingerprint = PatternFingerprint::of(&pattern);
        if *checked != Some(fingerprint) {
            validate(&pattern)?;
            *checked = Some(fingerprint);
        }
        Self::wrap(device, refs, pattern, value)
    }

    /// Size the value array to the pattern and pair the two. The pattern is
    /// already validated by whichever constructor called this.
    fn wrap<D: Device>(
        device: &mut D,
        refs: FixedPatternRefs,
        pattern: FixedPattern<'a>,
        value: ReadbackBuffer<f32>,
    ) -> FatalResult<Self> {
        let floats = BLOCK_FLOATS * pattern.nnz();
        let mut value = value;
        if value.len() != floats {
            // A LABEL OF ITS OWN, because a tally keys on it and both matrices
            // were sized under one name: 12 downloads of 200 MB showed against
            // `csr.fixed_value` on a scene whose only reader of that mirror,
            // `FixedCsr::download`, is called from `mod tests` alone. One name
            // over two allocations cannot say which of them moved.
            value.size(device, floats, AllocLabel("csr.adopted_value")).map_err(|error| {
                Fatal::out_of_memory(format!(
                    "solver driver: cannot allocate {floats} floats ({} MiB) for a \
                     fixed-pattern Hessian over {} rows and {} stored blocks: {error:?}",
                    (floats * std::mem::size_of::<f32>()) >> 20,
                    pattern.rows,
                    pattern.nnz()
                ))
            })?;
        }
        Ok(Self { pattern, refs, value })
    }

    /// Hand the value array back so the next step can reuse the allocation.
    pub fn into_values(self) -> ReadbackBuffer<f32> {
        self.value
    }

    /// As [`Self::from_dataset`], for a pattern assembled by a caller.
    pub fn new<D: Device>(
        device: &mut D,
        refs: FixedPatternRefs,
        pattern: FixedPattern<'a>,
    ) -> FatalResult<Self> {
        validate(&pattern)?;
        let floats = BLOCK_FLOATS
            .checked_mul(pattern.nnz())
            .ok_or_else(|| Fatal::out_of_memory("solver driver: the fixed matrix's value array overflows a usize"))?;
        let mut value = ReadbackBuffer::default();
        value
            .size(device, floats, AllocLabel("csr.fixed_value"))
            .map_err(|error| {
                Fatal::out_of_memory(format!(
                    "solver driver: cannot allocate {floats} floats ({} MiB) for a fixed-pattern \
                     Hessian over {} rows and {} stored blocks: {error:?}",
                    (floats * std::mem::size_of::<f32>()) >> 20,
                    pattern.rows,
                    pattern.nnz()
                ))
            })?;
        Ok(Self { pattern, refs, value })
    }

    pub fn rows(&self) -> u32 {
        self.pattern.rows
    }

    pub fn nnz(&self) -> usize {
        self.pattern.nnz()
    }

    pub fn pattern(&self) -> FixedPattern<'a> {
        self.pattern
    }

    /// Zero every block, which is how a Newton iteration opens.
    ///
    /// Written as a fill rather than a subtraction because it IS a fill: the
    /// CUDA side spells it `value.clear(Mat3x3f::Zero())`, and the bytes are the
    /// same. A buffer whose consumers read further than its producers write is
    /// cleared by the pass that opens the assembly, never by a kernel inside it.
    ///
    /// It is a host `fill` rather than a dispatch because the value array is a
    /// driver-owned `Vec` rather than a [`ppf_cts_compute::Handle`], which is
    /// the one case `Encoder::fill` is named for and does not yet cover; the
    /// same applies to [`Self::copy_from`], which is a copy rather than a
    /// computation. Both are device operations now: this one is a `fill_zero`
    /// over the value allocation and that one a device-to-device copy.
    pub fn clear<D: Device>(&mut self, device: &mut D) -> FatalResult<()> {
        let bytes = self.value.len() * std::mem::size_of::<f32>();
        let handle = self.value.handle();
        device
            .fill_zero(handle, bytes)
            .map_err(|error| Fatal::invariant(format!(
                "solver driver: cannot zero the fixed matrix's values: {error:?}"
            )))?;
        Ok(())
    }

    /// Snapshot `other` into this matrix, which is `tmp_fixed.copy(fixed_hess)`.
    ///
    /// The two must share a pattern; they do by construction (both are built
    /// from the scene's one table), and the assertion is what makes that a
    /// checked claim rather than a remembered one.
    pub fn copy_from<D: Device>(&mut self, device: &mut D, other: &FixedCsr<'_>) -> FatalResult<()> {
        if self.value.len() != other.value.len() || self.pattern.rows != other.pattern.rows {
            return Err(Fatal::invariant(format!(
                "solver driver: tmp_fixed and fixed_hess were built over different patterns \
                 ({} rows / {} blocks against {} rows / {} blocks), so the snapshot the contact \
                 stiffness reads would be indexed by the wrong slots",
                self.pattern.rows,
                self.nnz(),
                other.pattern.rows,
                other.nnz()
            )));
        }
        self.value.copy_from(device, &other.value).map_err(|error| {
            Fatal::invariant(format!(
                "solver driver: cannot snapshot the fixed matrix: {error:?}"
            ))
        })?;
        Ok(())
    }

    /// The flat view the SpMV and the preconditioner read.
    pub fn view(&mut self) -> FixedCsrView {
        FixedCsrView {
            index: self.refs.index,
            offset: self.refs.offset,
            value: self.value.handle(),
            transpose_pair: self.refs.transpose_pair,
            transpose_offset: self.refs.transpose_offset,
            rows: self.refs.rows,
        }
    }

    /// The value array itself, for a device-to-device copy.
    pub fn value_readback(&self) -> &ReadbackBuffer<f32> {
        &self.value
    }

    /// The value array's handle, for a record that writes it.
    pub fn value_handle(&mut self) -> ppf_cts_compute::Handle {
        self.value.handle()
    }

    /// Bring the mirror current, which every host read below requires.
    pub fn download<D: Device>(&mut self, device: &mut D) -> FatalResult<()> {
        self.value.download(device).map_err(|error| {
            Fatal::invariant(format!(
                "solver driver: cannot read back the fixed matrix's values: {error:?}"
            ))
        })
    }

    /// The raw value array, for the Dirichlet pass that zeroes stored blocks.
    pub fn values_mut(&mut self) -> ppf_cts_compute::Handle {
        self.value.handle()
    }

    pub fn values(&self) -> &[f32] {
        self.value.host()
    }

    /// Block `(i, j)` as the shared body reads it: the zero block when the
    /// pattern carries no slot, and the stored transpose when `i > j`.
    pub fn read(&self, i: u32, j: u32) -> [f32; BLOCK_FLOATS] {
        let mut out = [0.0f32; BLOCK_FLOATS];
        // Safety: the pattern was validated at construction, the value array is
        // `9 * nnz` long, and the body reads no further than the row it is given.
        unsafe {
            fixed_csr_read_abi(
                self.pattern.index.as_ptr(),
                self.pattern.offset.as_ptr(),
                self.value.host().as_ptr(),
                self.pattern.rows,
                i,
                j,
                out.as_mut_ptr(),
            )
        };
        out
    }

    /// The slot holding `(i, j)`, and whether the stored block is its transpose.
    ///
    /// `None` when the pattern has no slot. Exposed because the assembly's
    /// slot-replay tables are validated against it and because a lost stencil is
    /// the defect this pattern exists to make findable.
    pub fn find(&self, i: u32, j: u32) -> Option<(u32, bool)> {
        let mut transposed = 0i32;
        // Safety: as `read`.
        let slot = unsafe {
            fixed_csr_find_abi(
                self.pattern.index.as_ptr(),
                self.pattern.offset.as_ptr(),
                self.pattern.rows,
                i,
                j,
                &mut transposed,
            )
        };
        if slot == u32::MAX {
            None
        } else {
            Some((slot, transposed != 0))
        }
    }

    /// Fold `blocks[k]` into `(row[k], column[k])`, in ascending `k`.
    ///
    /// SERIAL BY CONTRACT, AND THE CONTRACT IS NOW THE DECLARATION'S RATHER THAN
    /// THIS CALL SITE'S. The shared body's fold is `compute::atomic_add`, which
    /// `kernels/seam/seam_host.h` spells as a plain read-add-write on the
    /// stated premise that one thread runs a shared body. Two threads folding
    /// blocks that land in the same slot would be a data race, not merely a
    /// nondeterministic order, and the running sum's ORDER is part of the fp32
    /// answer besides. `super::kernels` therefore declares this kernel
    /// `Scatter::Atomic`, so the backend runs the whole batch as one ascending
    /// pass whoever dispatches it; a caller that wants parallelism owes the
    /// disjointness argument (evaluate in parallel into per-element scratch,
    /// then fold serially in ascending element index) and
    /// still cannot get it by cutting this range. That serial fold is also what
    /// makes this backend's answer independent of the thread count, which is the
    /// one property it has that CUDA does not.
    ///
    /// A `false` in the returned flags is a LOST HESSIAN BLOCK, not a rounding
    /// difference: the pattern had no slot for that pair, so the block was
    /// dropped and the Newton matrix is missing a coupling. `FixedCSRMat::push`
    /// returns the same verdict and its CUDA callers ignore it, which is exactly
    /// how the rod-bend `(j, k)` stencil bug shipped. Callers here must not, and
    /// crossing the seam changed nothing about that: the verdict is a per-block
    /// output buffer of the dispatch, so it survives the move and is still the
    /// caller's to raise on.
    /// The four things a kernel needs to push into this matrix itself.
    ///
    /// THE PATTERN AND THE VALUES TRAVEL TOGETHER, for the reason the struct's
    /// own note gives: a push looks a block up in the pattern the matrix was
    /// BUILT over, so handing a caller the pattern separately would let it pair
    /// one matrix's values with another's slots, and the failure is silent
    /// because production always agrees.
    pub fn device_push_refs(&mut self) -> (ppf_cts_compute::Handle, ppf_cts_compute::Handle, ppf_cts_compute::Handle, u32) {
        let rows = self.pattern.rows;
        (self.refs.index, self.refs.offset, self.value.handle(), rows)
    }

    /// Push every active element's Hessian blocks FROM A KERNEL.
    ///
    /// WHAT A HOST PASS WOULD COST, measured at 56 percent of this tree's
    /// host-to-device bytes: downloading the element Hessians, walking them on
    /// the host into `push_row`, `push_column` and `push_block`, uploading
    /// those three and pushing from them. The kernel offers each block in the
    /// element's own thread instead, and nothing crosses the seam.
    ///
    /// A REFUSAL IS A FATAL AND THE COUNT IS HOW IT TRAVELS. The counter opens
    /// at zero, the kernel adds one per block the pattern cannot hold, and this
    /// reads it once. A per-block verdict would be an array the host then
    /// walks, which is the staging being removed.
    ///
    /// # Safety
    /// Every handle must outlive the dispatch.
    /// The same push with the element as the THREAD INDEX and the run decided
    /// in the kernel.
    ///
    /// THERE IS NO ACTIVE LIST anywhere in the energy path: it dispatches over
    /// the full element count and gates on the element's own props. This is
    /// that shape, and what it saves is not the dispatch but the list, which
    /// the host could only build by downloading the array the gate reads.
    ///
    /// # Safety
    /// Every handle must outlive the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn push_element_blocks_gated<D: Device>(
        &mut self,
        device: &mut D,
        staging: &mut super::state::PushStaging,
        live: ppf_cts_compute::Handle,
        index: ppf_cts_compute::Handle,
        hessian: ppf_cts_compute::Handle,
        arity: u32,
        count: u32,
        what: &str,
    ) -> FatalResult<()> {
        if count == 0 {
            return Ok(());
        }
        staging.refused.size(device, 1, AllocLabel("csr.push_refused"))?;
        { let h = staging.refused.handle(); device.fill_zero(h, 4)?; }
        staging.witness.size(device, 2, AllocLabel("csr.push_witness"))?;
        { let h = staging.witness.handle(); device.fill_zero(h, 8)?; }
        let (fixed_index, fixed_offset, fixed_value, row_count) = self.device_push_refs();
        let args = crate::driver::kernels::FixedPushElementBlocksGatedArgs {
            live,
            index,
            hessian,
            arity,
            fixed_index,
            fixed_offset,
            fixed_value,
            row_count,
            refused: staging.refused.handle(),
            witness: staging.witness.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: the caller's handles outlive the call, and `Device::launch`
        // executes and waits before returning.
        unsafe { device.launch("assemble.fixed_push", &args, count) }?;
        self.report_refusals(device, staging, what)
    }

    /// The same push over an UNSIGNED verdict rather than a float scale.
    ///
    /// A `Handle` carries no type, so handing a `u32` verdict array to the
    /// float form above would compile and reinterpret the bits: `1u` is a
    /// denormal above zero and `0u` is exactly it, so it would even appear to
    /// work. The rod strain limiter's `ok` is scattered as `unsigned` by its
    /// own body, and this is the entry that reads it as one.
    ///
    /// # Safety
    /// As [`FixedCsr::push_element_blocks_gated`].
    pub unsafe fn push_element_blocks_live<D: Device>(
        &mut self,
        device: &mut D,
        staging: &mut super::state::PushStaging,
        live: ppf_cts_compute::Handle,
        index: ppf_cts_compute::Handle,
        hessian: ppf_cts_compute::Handle,
        arity: u32,
        count: u32,
        what: &str,
    ) -> FatalResult<()> {
        if count == 0 {
            return Ok(());
        }
        staging.refused.size(device, 1, AllocLabel("csr.push_refused"))?;
        { let h = staging.refused.handle(); device.fill_zero(h, 4)?; }
        staging.witness.size(device, 2, AllocLabel("csr.push_witness"))?;
        { let h = staging.witness.handle(); device.fill_zero(h, 8)?; }
        let (fixed_index, fixed_offset, fixed_value, row_count) = self.device_push_refs();
        let args = crate::driver::kernels::FixedPushElementBlocksLiveArgs {
            live,
            index,
            hessian,
            arity,
            fixed_index,
            fixed_offset,
            fixed_value,
            row_count,
            refused: staging.refused.handle(),
            witness: staging.witness.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: the caller's handles outlive the call, and `Device::launch`
        // executes and waits before returning.
        unsafe { device.launch("assemble.fixed_push_live", &args, count) }?;
        self.report_refusals(device, staging, what)
    }

    /// The same push, deposited at a precomputed slot table.
    ///
    /// THE SLOT-TABLE DEPOSIT. It reads the slot `builder.rs` computed for
    /// each block instead of searching the row for it, and there is no verdict
    /// to report: a slot came from the pattern the matrix was built over, so
    /// nothing can be refused and this takes no `staging` and calls no
    /// `report_refusals`.
    ///
    /// # Safety
    /// Every handle must name a live allocation that outlives the dispatch.
    /// The slot deposit with the run decided in the thread.
    ///
    /// # Safety
    /// Every handle must name a live allocation that outlives the dispatch.
    pub unsafe fn push_element_blocks_gated_at<D: Device>(
        &mut self,
        device: &mut D,
        live: ppf_cts_compute::Handle,
        slots: ppf_cts_compute::Handle,
        hessian: ppf_cts_compute::Handle,
        arity: u32,
        count: u32,
    ) -> FatalResult<()> {
        if count == 0 {
            return Ok(());
        }
        let (_, _, fixed_value, _) = self.device_push_refs();
        let args = crate::driver::kernels::FixedPushElementBlocksGatedAtArgs {
            live,
            slots,
            hessian,
            arity,
            fixed_value,
            count,
            seam_arena_count: 0,
        };
        // Safety: the caller's handles outlive the call.
        unsafe { device.launch("assemble.fixed_push_gated_at", &args, count) }?;
        Ok(())
    }

    pub unsafe fn push_element_blocks_at<D: Device>(
        &mut self,
        device: &mut D,
        active: ppf_cts_compute::Handle,
        slots: ppf_cts_compute::Handle,
        hessian: ppf_cts_compute::Handle,
        arity: u32,
        count: u32,
    ) -> FatalResult<()> {
        if count == 0 {
            return Ok(());
        }
        let (_, _, fixed_value, _) = self.device_push_refs();
        let args = crate::driver::kernels::FixedPushElementBlocksAtArgs {
            active,
            slots,
            hessian,
            arity,
            fixed_value,
            count,
            seam_arena_count: 0,
        };
        // Safety: the caller's handles outlive the call.
        unsafe { device.launch("assemble.fixed_push_at", &args, count) }?;
        Ok(())
    }

    pub unsafe fn push_element_blocks<D: Device>(
        &mut self,
        device: &mut D,
        staging: &mut super::state::PushStaging,
        active: ppf_cts_compute::Handle,
        index: ppf_cts_compute::Handle,
        hessian: ppf_cts_compute::Handle,
        arity: u32,
        count: u32,
        what: &str,
    ) -> FatalResult<()> {
        if count == 0 {
            return Ok(());
        }
        // THREE SLOTS: the count, and one refused block's row and column, so
        // the fatal below can name a block rather than only a number.
        staging.refused.size(device, 1, AllocLabel("csr.push_refused"))?;
        { let h = staging.refused.handle(); device.fill_zero(h, 4)?; }
        staging.witness.size(device, 2, AllocLabel("csr.push_witness"))?;
        { let h = staging.witness.handle(); device.fill_zero(h, 8)?; }
        let (fixed_index, fixed_offset, fixed_value, row_count) = self.device_push_refs();
        let args = crate::driver::kernels::FixedPushElementBlocksArgs {
            active,
            index,
            hessian,
            arity,
            fixed_index,
            fixed_offset,
            fixed_value,
            row_count,
            refused: staging.refused.handle(),
            witness: staging.witness.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: the caller's handles outlive the call, and `Device::launch`
        // executes and waits before returning.
        unsafe { device.launch("assemble.fixed_push", &args, count) }?;
        self.report_refusals(device, staging, what)
    }

    /// Raise a fatal naming one block the fixed pattern could not hold.
    ///
    /// A STENCIL THE PATTERN CANNOT HOLD IS A `builder.rs` DEFECT, not a
    /// routing: the coupling is dropped, the Newton matrix is missing it, and
    /// damping can mask the result, which is what the rod-bend `(j, k)` blocks
    /// cost once. The kernel counts refusals and records one pair rather than
    /// returning a per-block verdict, because a caller cannot act on one.
    /// Seed the refusal counter and its witness before a pass that pushes.
    ///
    /// Split out of the push launchers so a kernel that forms its own blocks
    /// and pushes them itself, `shell_bend_embed`, can take the same staging
    /// and report through the same [`FixedCsr::report_refusals`].
    pub fn seed_push_staging<D: Device>(
        device: &mut D,
        staging: &mut super::state::PushStaging,
    ) -> FatalResult<()> {
        staging.refused.size(device, 1, AllocLabel("csr.push_refused"))?;
        { let h = staging.refused.handle(); device.fill_zero(h, 4)?; }
        staging.witness.size(device, 2, AllocLabel("csr.push_witness"))?;
        { let h = staging.witness.handle(); device.fill_zero(h, 8)?; }
        Ok(())
    }

    pub fn report_refusals<D: Device>(
        &mut self,
        device: &mut D,
        staging: &mut super::state::PushStaging,
        what: &str,
    ) -> FatalResult<()> {
        staging.refused.download(device)?;
        staging.witness.download(device)?;
        let refused = staging.refused.host()[0];
        let (row, column) = (staging.witness.host()[0], staging.witness.host()[1]);
        if refused > 0 {
            return Err(Fatal::invariant(format!(
                "solver driver: the scene's fixed sparsity has no slot for Hessian block \
                 ({row}, {column}), one of {refused} a {what} contributes, so that coupling \
                 would be dropped and the Newton matrix would be missing it. The stencil is \
                 registered in ppf-cts-core's fixed_index_table"
            )));
        }
        Ok(())
    }

    pub fn push_blocks<D: Device>(
        &mut self,
        device: &mut D,
        // THE DRIVER'S ONE STAGING AREA. The caller still builds its triple in
        // a host `Vec`, because the scatter that fills it walks a chunk on the
        // host; this uploads that into buffers the record can name, sized once
        // and grown as a chunk gets wider.
        staging: &mut super::state::PushStaging,
        rows: &[u32],
        columns: &[u32],
        blocks: &[f32],
        stored: &mut [u32],
    ) -> FatalResult<()> {
        // `Mat3x3f` is `na::Matrix3<f32>`; the shim reinterprets the incoming
        // float array as an array of them, so the two layouts must agree. Checked
        // here rather than assumed because a silent disagreement would transpose
        // or shift every block.
        const _: () = assert!(std::mem::size_of::<Mat3x3f>() == BLOCK_FLOATS * 4);
        let count = rows.len();
        if columns.len() != count || stored.len() != count || blocks.len() != BLOCK_FLOATS * count {
            return Err(Fatal::invariant(format!(
                "solver driver: a fixed-matrix push was given {count} rows, {} columns, {} verdict \
                 slots and {} floats, which cannot describe one block per row",
                columns.len(),
                stored.len(),
                blocks.len()
            )));
        }
        if count == 0 {
            return Ok(());
        }
        for k in 0..count {
            if rows[k] >= self.pattern.rows || columns[k] >= self.pattern.rows {
                return Err(Fatal::device_assert(format!(
                    "solver driver: a fixed-matrix push named block ({}, {}) in a matrix with {} \
                     rows",
                    rows[k], columns[k], self.pattern.rows
                )));
            }
        }
        // The pattern is `Copy` and is taken before the value array is borrowed
        // mutably, so the record names both halves of the matrix at once.
        let pattern = self.pattern;
        // Upload the triple and size the verdict. `Buffer::size` grows only past
        // CAPACITY, so a chunk no wider than the last costs no allocation.
        staging
            .row
            .size(device, count, AllocLabel("csr.push_row"))
            .and_then(|()| staging.row.write(device, 0, rows))
            .and_then(|()| staging.column.size(device, count, AllocLabel("csr.push_column")))
            .and_then(|()| staging.column.write(device, 0, columns))
            .and_then(|()| staging.block.size(device, blocks.len(), AllocLabel("csr.push_block")))
            .and_then(|()| staging.block.write(device, 0, blocks))
            .and_then(|()| staging.stored.size(device, count, AllocLabel("csr.push_stored")))
            .map_err(|error| {
                Fatal::out_of_memory(format!("solver driver: cannot stage a fixed-matrix push: {error:?}"))
            })?;
        let args = FixedCsrAtomicPushArgs {
            index: self.refs.index,
            offset: self.refs.offset,
            value: self.value.handle(),
            row_count: pattern.rows,
            row: staging.row.span(0, count),
            column: staging.column.span(0, count),
            block: staging.block.span(0, blocks.len()),
            stored: staging.stored.span(0, count),
            count: count as u32,
            seam_arena_count: 0,
        };
        // Safety: every index was just bounds-checked, the lengths agree, and
        // every buffer the record names is borrowed by this call. `Device::run`
        // executes and waits before returning, so none of those borrows ends
        // before the dispatch has.
        unsafe { device.launch("assemble.fixed_csr.push", &args, count as u32) }?;
        // THE VERDICT COMES BACK. A `false` here is a LOST HESSIAN COUPLING and
        // every caller checks it, so the mirror the dispatch staled has to be
        // current before they do.
        staging.stored.download(device).map_err(|error| {
            Fatal::invariant(format!("solver driver: cannot read back the push verdict: {error:?}"))
        })?;
        stored.copy_from_slice(&staging.stored.host()[..count]);
        Ok(())
    }
}

/// Borrow the two scene tables as one pattern.
///
/// # Safety
/// Both `CVecVec`s must describe live allocations.
pub(crate) unsafe fn borrow_pattern<'a>(
    index_table: &'a CVecVec<u32>,
    transpose_table: &'a CVecVec<Vec2u>,
) -> FatalResult<FixedPattern<'a>> {
    // `Vec2u` is `na::Vector2<u32>`, whose storage is two contiguous `u32`s, so
    // the pair array and a flat `u32` array of twice the length are the same
    // bytes. The shim's `transpose_pair` parameter is the flat form, and this is
    // where the reinterpretation happens, once.
    const _: () = assert!(std::mem::size_of::<Vec2u>() == 2 * std::mem::size_of::<u32>());
    const _: () = assert!(std::mem::align_of::<Vec2u>() == std::mem::align_of::<u32>());

    if index_table.size != transpose_table.size {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene's fixed sparsity has {} rows and its transpose table {}. \
             They are two views of one matrix and are built together in builder.rs, so a \
             disagreement means the DataSet was assembled by something else",
            index_table.size, transpose_table.size
        )));
    }
    let rows = index_table.size;
    Ok(FixedPattern {
        index: raw_slice(index_table.data, index_table.nnz as usize),
        offset: raw_slice(index_table.offset, rows as usize + 1),
        transpose_pair: raw_slice(
            transpose_table.data as *const u32,
            2 * transpose_table.nnz as usize,
        ),
        transpose_offset: raw_slice(transpose_table.offset, rows as usize + 1),
        rows,
    })
}

/// # Safety
/// `ptr` must address `len` elements, or `len` must be zero.
unsafe fn raw_slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// Everything the SpMV assumes about the two tables, checked once.
///
/// The order is deliberate: the structural checks come first so a later check
/// cannot index out of range while reporting a different problem.
fn validate(pattern: &FixedPattern<'_>) -> FatalResult<()> {
    let rows = pattern.rows as usize;
    if rows == 0 {
        // A scene with no vertices has no matrix, and every array is empty. Not
        // an error: the refusal gate accepts such a scene, so the container must
        // too.
        if !pattern.index.is_empty() || !pattern.transpose_pair.is_empty() {
            return Err(Fatal::invariant(
                "solver driver: the fixed sparsity has no rows but carries stored blocks",
            ));
        }
        return Ok(());
    }
    if pattern.offset.len() != rows + 1 {
        return Err(Fatal::invariant(format!(
            "solver driver: the fixed sparsity has {rows} rows and {} row offsets; a CSR offset \
             array is one longer than its row count",
            pattern.offset.len()
        )));
    }
    if pattern.transpose_offset.len() != rows + 1 {
        return Err(Fatal::invariant(format!(
            "solver driver: the transpose table has {rows} rows and {} row offsets",
            pattern.transpose_offset.len()
        )));
    }
    if pattern.offset[0] != 0 || pattern.offset[rows] as usize != pattern.index.len() {
        return Err(Fatal::invariant(format!(
            "solver driver: the fixed sparsity's row offsets run {}..{} over {} column entries",
            pattern.offset[0],
            pattern.offset[rows],
            pattern.index.len()
        )));
    }
    if pattern.transpose_offset[0] != 0
        || 2 * pattern.transpose_offset[rows] as usize != pattern.transpose_pair.len()
    {
        return Err(Fatal::invariant(format!(
            "solver driver: the transpose table's row offsets run {}..{} over {} pairs",
            pattern.transpose_offset[0],
            pattern.transpose_offset[rows],
            pattern.transpose_pair.len() / 2
        )));
    }

    // Row contents. `fixed_csr_find` bisects, so ascending order is not a
    // convention here, it is what makes the lookup correct; an unsorted row
    // makes a present block report as absent and the assembly then drops it.
    let mut off_diagonal = 0usize;
    for row in 0..rows {
        let begin = pattern.offset[row] as usize;
        let end = pattern.offset[row + 1] as usize;
        if begin > end || end > pattern.index.len() {
            return Err(Fatal::invariant(format!(
                "solver driver: row {row} of the fixed sparsity spans {begin}..{end}"
            )));
        }
        for slot in begin..end {
            let column = pattern.index[slot];
            if column as usize >= rows {
                return Err(Fatal::device_assert(format!(
                    "solver driver: the fixed sparsity names column {column} in a matrix with \
                     {rows} rows"
                )));
            }
            if (column as usize) < row {
                return Err(Fatal::invariant(format!(
                    "solver driver: row {row} of the fixed sparsity stores column {column}. Only \
                     the upper triangle is stored, and a lower-triangle entry would be read \
                     twice by the SpMV: once directly and once through the transpose table"
                )));
            }
            if slot > begin && pattern.index[slot - 1] >= column {
                return Err(Fatal::invariant(format!(
                    "solver driver: row {row} of the fixed sparsity is not strictly ascending at \
                     slot {slot} ({} then {column}). The slot lookup bisects, so an unsorted \
                     row makes a stored block report as absent",
                    pattern.index[slot - 1]
                )));
            }
            if column as usize != row {
                off_diagonal += 1;
            }
        }
    }

    // The cross-check. Every stored off-diagonal block must appear exactly once
    // in the transpose table, in the row named by its COLUMN, pointing back at
    // its own slot with the row it came from.
    let pairs = pattern.transpose_offset[rows] as usize;
    if pairs != off_diagonal {
        return Err(Fatal::invariant(format!(
            "solver driver: the fixed sparsity stores {off_diagonal} off-diagonal blocks and the \
             transpose table carries {pairs} entries. Only the upper triangle is stored, so \
             every off-diagonal block is read once directly and once transposed; a mismatch \
             drops or duplicates a lower-triangle coupling, which leaves the operator \
             non-symmetric while every block is still present"
        )));
    }
    let mut seen = vec![false; pattern.index.len()];
    for column_row in 0..rows {
        let begin = pattern.transpose_offset[column_row] as usize;
        let end = pattern.transpose_offset[column_row + 1] as usize;
        if begin > end || end > pairs {
            return Err(Fatal::invariant(format!(
                "solver driver: row {column_row} of the transpose table spans {begin}..{end} of \
                 {pairs} pairs"
            )));
        }
        for entry in begin..end {
            let source_row = pattern.transpose_pair[2 * entry] as usize;
            let slot = pattern.transpose_pair[2 * entry + 1] as usize;
            if slot >= pattern.index.len() {
                return Err(Fatal::device_assert(format!(
                    "solver driver: the transpose table points at slot {slot} of {} stored blocks",
                    pattern.index.len()
                )));
            }
            if seen[slot] {
                return Err(Fatal::invariant(format!(
                    "solver driver: slot {slot} of the fixed sparsity appears twice in the \
                     transpose table, so its coupling is applied twice on the lower-triangle \
                     side"
                )));
            }
            seen[slot] = true;
            if source_row >= rows {
                return Err(Fatal::device_assert(format!(
                    "solver driver: the transpose table names source row {source_row} in a matrix \
                     with {rows} rows"
                )));
            }
            let owner_begin = pattern.offset[source_row] as usize;
            let owner_end = pattern.offset[source_row + 1] as usize;
            if slot < owner_begin || slot >= owner_end {
                return Err(Fatal::invariant(format!(
                    "solver driver: the transpose table says slot {slot} belongs to row \
                     {source_row}, whose slots are {owner_begin}..{owner_end}"
                )));
            }
            if pattern.index[slot] as usize != column_row {
                return Err(Fatal::invariant(format!(
                    "solver driver: the transpose table files slot {slot} under column \
                     {column_row}, but the fixed sparsity stores it at column {}",
                    pattern.index[slot]
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::state::PatternDevice;

    /// One host float array on the device, for a test that drives an entry
    /// taking handles.
    fn pattern_array_f32(
        device: &mut impl Device,
        host: &[f32],
        label: &'static str,
    ) -> ppf_cts_compute::Buffer<f32> {
        let mut buffer = ppf_cts_compute::Buffer::<f32>::none();
        buffer
            .size(device, host.len(), ppf_cts_compute::AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, host)
            .expect("the test upload succeeds");
        buffer
    }
    use super::*;
    use crate::driver::launch::host_device;
    use crate::driver::spmv;

    /// Build the two tables the way `builder.rs` does, from an upper-triangle
    /// pattern given as one ascending column list per row.
    ///
    /// A deliberate re-expression of `builder.rs:1440-1464` rather than a call
    /// into it: what is under test is whether this module reads that layout
    /// correctly, so the fixture has to produce the layout independently.
    fn tables(rows: &[Vec<u32>]) -> (CVecVec<u32>, CVecVec<Vec2u>) {
        let mut transpose: Vec<Vec<Vec2u>> = vec![Vec::new(); rows.len()];
        let mut slot = 0u32;
        for (i, row) in rows.iter().enumerate() {
            for &j in row {
                if i as u32 != j {
                    transpose[j as usize].push(Vec2u::new(i as u32, slot));
                }
                slot += 1;
            }
        }
        (CVecVec::from(rows), CVecVec::from(&transpose[..]))
    }

    /// The pattern of a single tet on four vertices: every pair, upper triangle.
    fn tet_rows() -> Vec<Vec<u32>> {
        vec![vec![0, 1, 2, 3], vec![1, 2, 3], vec![2, 3], vec![3]]
    }

    fn pattern_of<'a>(index: &'a CVecVec<u32>, transpose: &'a CVecVec<Vec2u>) -> FixedPattern<'a> {
        unsafe { borrow_pattern(index, transpose) }.expect("the fixture builds a valid pattern")
    }

    #[test]
    fn a_tet_pattern_validates_and_allocates_one_block_per_pair() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let (index, transpose) = tables(&tet_rows());
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");
        // Four vertices, all pairs, upper triangle: 4 + 3 + 2 + 1.
        assert_eq!(matrix.nnz(), 10);
        assert_eq!(matrix.rows(), 4);
        assert_eq!(matrix.values().len(), 9 * 10);
        assert!(matrix.values().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_pushed_block_reads_back_through_the_shared_body() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let (index, transpose) = tables(&tet_rows());
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let mut matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");

        // Column-major, deliberately non-symmetric so a transposed read is
        // distinguishable from a direct one.
        let block: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut stored = [0u32; 1];
        matrix
            .push_blocks(&mut device, &mut staging, &[1], &[3], &block, &mut stored)
            .expect("the pair is in the pattern");
            // The push is a DISPATCH, so the mirror every read below
            // answers from is stale until it is brought current.
            matrix.download(&mut device).expect("the values read back");
        assert_eq!(stored[0], 1);

        assert_eq!(matrix.read(1, 3), block);
        // (3, 1) is the same slot read transposed: element (r, c) of the answer
        // is element (c, r) of the stored block.
        let lower = matrix.read(3, 1);
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(lower[3 * c + r], block[3 * r + c]);
            }
        }
    }

    #[test]
    fn the_slot_deposit_lands_what_the_row_search_lands() {
        // THE SLOT DEPOSIT AGAINST THE ROW SEARCH, at the level of the CSR
        // operation rather than of a physics layer.
        // `push_element_blocks` searches each block's row for its slot;
        // `push_element_blocks_at` reads the slot `builder.rs` precomputed.
        // The two must fill the matrix identically, which is what makes the
        // table an optimization of the LOOKUP rather than a second opinion
        // about where a block belongs.
        //
        // THE HESSIAN IS DELIBERATELY ASYMMETRIC, every one of the 144 entries
        // distinct, so a transposed or permuted slot lands a different number
        // rather than the same one. A symmetric fixture cannot see that.
        let mut device = host_device();
        let (index, transpose) = tables(&tet_rows());

        // The element is the whole tet, so its sixteen blocks cover every pair
        // in the pattern, upper triangle and lower alike.
        let tet: [u32; 4] = [0, 1, 2, 3];
        let hessian: Vec<f32> = (0..144).map(|k| 1.0 + k as f32).collect();
        // The slot table `builder.rs` would ship: row `i` holds the columns
        // `tet_rows()[i]`, laid out flat in that order, and a lower-triangle
        // pair is the sentinel.
        const SENTINEL: u32 = 0xFFFF_FFFF;
        let rows = tet_rows();
        let mut row_offset = vec![0u32; rows.len() + 1];
        for (i, row) in rows.iter().enumerate() {
            row_offset[i + 1] = row_offset[i] + row.len() as u32;
        }
        let mut slots = Vec::with_capacity(16);
        for ii in 0..4usize {
            for jj in 0..4usize {
                let (r, c) = (tet[ii], tet[jj]);
                slots.push(if r > c {
                    SENTINEL
                } else {
                    let at = rows[r as usize].iter().position(|&x| x == c).expect("in pattern");
                    row_offset[r as usize] + at as u32
                });
            }
        }

        let mut fill = |use_slots: bool| -> Vec<f32> {
            let pattern = pattern_of(&index, &transpose);
            let pattern_dev = PatternDevice::of(&mut device, &pattern);
            let mut staging = super::super::state::PushStaging::default();
            let mut matrix =
                FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");
            let mut active = ppf_cts_compute::StagedBuffer::<u32>::default();
            active.size(&mut device, 1, AllocLabel("test.active")).expect("sized");
            active.at()[0] = 0;
            let active_h = active.upload_span(&mut device, 1).expect("uploaded");
            let mut hess = ppf_cts_compute::StagedBuffer::<f32>::default();
            hess.size(&mut device, 144, AllocLabel("test.hessian")).expect("sized");
            hess.at().copy_from_slice(&hessian);
            let hess_h = hess.upload_span(&mut device, 144).expect("uploaded");
            let mut nodes = ppf_cts_compute::StagedBuffer::<u32>::default();
            nodes.size(&mut device, 4, AllocLabel("test.tet")).expect("sized");
            nodes.at().copy_from_slice(&tet);
            let nodes_h = nodes.upload_span(&mut device, 4).expect("uploaded");
            if use_slots {
                let mut table = ppf_cts_compute::StagedBuffer::<u32>::default();
                table.size(&mut device, 16, AllocLabel("test.slots")).expect("sized");
                table.at().copy_from_slice(&slots);
                let table_h = table.upload_span(&mut device, 16).expect("uploaded");
                // Safety: every handle outlives the dispatch below.
                unsafe {
                    matrix
                        .push_element_blocks_at(&mut device, active_h, table_h, hess_h, 4, 1)
                        .expect("the slot deposit runs");
                }
            } else {
                // Safety: every handle outlives the dispatch below.
                unsafe {
                    matrix
                        .push_element_blocks(
                            &mut device, &mut staging, active_h, nodes_h, hess_h, 4, 1, "test",
                        )
                        .expect("the search deposit runs");
                }
            }
            matrix.download(&mut device).expect("the values read back");
            matrix.values().to_vec()
        };

        let by_search = fill(false);
        let by_slots = fill(true);
        assert_eq!(by_slots.len(), by_search.len());
        assert_eq!(
            by_slots, by_search,
            "the slot deposit and the row search fill the matrix differently"
        );
        // AND SOMETHING LANDED, or the comparison above is two zeroed matrices
        // agreeing and says nothing about either path.
        assert!(
            by_slots.iter().any(|&v| v != 0.0),
            "the slot deposit wrote no block at all"
        );
    }

    #[test]
    fn a_pair_outside_the_pattern_reports_a_lost_block_rather_than_landing() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        // A pattern with NO coupling between 0 and 2: exactly the shape a
        // missing element stencil leaves behind.
        let rows = vec![vec![0, 1], vec![1, 2], vec![2]];
        let (index, transpose) = tables(&rows);
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let mut matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");

        let block = [1.0f32; 9];
        let mut stored = [7u32; 1];
        matrix
            .push_blocks(&mut device, &mut staging, &[0], &[2], &block, &mut stored)
            .expect("the call itself is well formed");
            // The push is a DISPATCH, so the mirror every read below
            // answers from is stale until it is brought current.
            matrix.download(&mut device).expect("the values read back");
        assert_eq!(
            stored[0], 0,
            "a pair the pattern does not carry must report the block as dropped; \
             ignoring this verdict is how the rod-bend stencil bug shipped"
        );
        assert!(
            matrix.values().iter().all(|&v| v == 0.0),
            "a dropped block must not have landed anywhere"
        );
    }

    #[test]
    fn push_refuses_an_index_outside_the_matrix() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let (index, transpose) = tables(&tet_rows());
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let mut matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");
        let block = [1.0f32; 9];
        let mut stored = [0u32; 1];
        let fatal = matrix
            .push_blocks(&mut device, &mut staging, &[4], &[4], &block, &mut stored)
            .expect_err("row 4 does not exist in a four-row matrix");
            // The push is a DISPATCH, so the mirror every read below
            // answers from is stale until it is brought current.
            matrix.download(&mut device).expect("the values read back");
        assert_eq!(fatal.code, ppf_cts_formats::status::error_code::DEVICE_ASSERT);
        assert!(fatal.detail.contains('4'), "{}", fatal.detail);
    }

    #[test]
    fn the_snapshot_copies_values_and_refuses_a_foreign_pattern() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let (index, transpose) = tables(&tet_rows());
        let pattern = pattern_of(&index, &transpose);
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let mut source = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");
        let mut snapshot = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");

        let block: [f32; 9] = [3.0; 9];
        let mut stored = [0u32; 1];
        source
            .push_blocks(&mut device, &mut staging, &[0], &[0], &block, &mut stored)
            .unwrap();
            // The push is a DISPATCH, so the mirror every read below
            // answers from is stale until it is brought current.
            source.download(&mut device).expect("the values read back");
        snapshot.copy_from(&mut device, &source).expect("same pattern");
        // The snapshot is a DEVICE-TO-DEVICE copy, so the destination's mirror
        // is stale even though the source's was current.
        snapshot.download(&mut device).expect("the snapshot reads back");
        assert_eq!(snapshot.read(0, 0), block);

        // A clear on the source must not reach the snapshot: the contact
        // stiffness reads the snapshot AFTER the elastic matrix has moved on.
        source.clear(&mut device).expect("the clear dispatches");
        // The clear is a dispatch too. The snapshot is NOT re-read from the
        // device here on purpose: its mirror is current from the download
        // above, and that it still holds the block is the property under test.
        source.download(&mut device).expect("the cleared values read back");
        assert_eq!(source.read(0, 0), [0.0f32; 9]);
        assert_eq!(snapshot.read(0, 0), block);

        let (other_index, other_transpose) = tables(&[vec![0, 1], vec![1]]);
        let other = pattern_of(&other_index, &other_transpose);
        let other_dev = PatternDevice::of(&mut device, &other);
        let mut foreign =
            FixedCsr::new(&mut device, other_dev.refs(), other).expect("valid pattern");
        let fatal = foreign
            .copy_from(&mut device, &source)
            .expect_err("two different patterns must not be copied between");
        assert!(fatal.detail.contains("different patterns"), "{}", fatal.detail);
    }

    /// The property the whole cross-check exists for.
    ///
    /// Only the upper triangle is stored, so `M x` is correct only if the
    /// transpose table drives the lower-triangle half. This assembles a small
    /// symmetric system through the shared push body and compares the shared
    /// SpMV against a dense reference built here, in the test, where `f32`
    /// arithmetic is allowed.
    #[test]
    fn the_spmv_reproduces_the_full_symmetric_operator() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let rows = vec![vec![0, 1, 2], vec![1, 2], vec![2]];
        let (index, transpose) = tables(&rows);
        let pattern = pattern_of(&index, &transpose);
        // The view is all handles, so this fixture owns the pattern's device
        // image the way the driver's state owns the production one.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let mut matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");

        // One block per stored pair, all distinct and none symmetric in itself,
        // so a dropped transpose or a transposed-the-wrong-way read shows up.
        let mut push_rows = Vec::new();
        let mut push_cols = Vec::new();
        let mut blocks: Vec<f32> = Vec::new();
        let mut next = 1.0f32;
        for (i, row) in rows.iter().enumerate() {
            for &j in row {
                push_rows.push(i as u32);
                push_cols.push(j);
                for _ in 0..9 {
                    blocks.push(next);
                    next += 1.0;
                }
            }
        }
        let mut stored = vec![0u32; push_rows.len()];
        matrix
            .push_blocks(&mut device, &mut staging, &push_rows, &push_cols, &blocks, &mut stored)
            .expect("every pair is in the pattern");
            // The push is a DISPATCH, so the mirror every read below
            // answers from is stale until it is brought current.
            matrix.download(&mut device).expect("the values read back");
        assert!(stored.iter().all(|&s| s == 1), "no block may be dropped");

        // Dense 9x9 reference: the stored upper triangle plus its transpose.
        let n = 3usize;
        let mut dense = vec![0.0f32; (3 * n) * (3 * n)];
        for k in 0..push_rows.len() {
            let (i, j) = (push_rows[k] as usize, push_cols[k] as usize);
            for c in 0..3 {
                for r in 0..3 {
                    let v = blocks[9 * k + 3 * c + r];
                    dense[(3 * i + r) * (3 * n) + (3 * j + c)] += v;
                    if i != j {
                        dense[(3 * j + c) * (3 * n) + (3 * i + r)] += v;
                    }
                }
            }
        }

        let x: Vec<f32> = (0..3 * n).map(|k| (k as f32) - 4.0).collect();
        let mut expected = vec![0.0f32; 3 * n];
        for r in 0..3 * n {
            let mut acc = 0.0f32;
            for c in 0..3 * n {
                acc += dense[r * (3 * n) + c] * x[c];
            }
            expected[r] = acc;
        }

        let mut got = vec![0.0f32; 3 * n];
        let x_d = pattern_array_f32(&mut device, &x, "test.x");
        let mut got_d = pattern_array_f32(&mut device, &got, "test.got");
        // Safety: every handle names a live allocation on this device.
        unsafe {
            spmv::fixed_csr_spmv(
                &mut device,
                &matrix.view(),
                x_d.span(0, x.len()),
                got_d.span(0, got.len()),
                x.len(),
            )
        }
        .expect("the fixed-pattern matvec dispatches");
        got_d
            .read(&mut device, 0, &mut got)
            .expect("the result reads back");

        for r in 0..3 * n {
            assert!(
                (got[r] - expected[r]).abs() <= 1e-3 * expected[r].abs().max(1.0),
                "row {r}: shared SpMV gave {}, the dense symmetric reference {}",
                got[r],
                expected[r]
            );
        }
    }

    #[test]
    fn an_unsorted_row_is_refused_by_name() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        // The lookup bisects, so an unsorted row makes a stored block report as
        // absent and the assembly silently drops it.
        let (index, transpose) = tables(&[vec![0, 2, 1], vec![1, 2], vec![2]]);
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let fatal = FixedCsr::new(&mut device, pattern_dev.refs(), pattern)
            .expect_err("an unsorted row must be refused");
        assert!(fatal.detail.contains("ascending"), "{}", fatal.detail);
    }

    #[test]
    fn a_lower_triangle_entry_is_refused_by_name() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let (index, transpose) = tables(&[vec![0, 1], vec![0, 1]]);
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let fatal = FixedCsr::new(&mut device, pattern_dev.refs(), pattern)
            .expect_err("a lower-triangle entry must be refused");
        assert!(fatal.detail.contains("upper triangle"), "{}", fatal.detail);
    }

    /// THE CHECK THAT COSTS THE MOST TO GET WRONG.
    ///
    /// A transpose table missing one entry leaves every block present and the
    /// matrix looking assembled, while the SpMV drops that coupling's
    /// lower-triangle half. The operator is then non-symmetric, which no PCG
    /// guard can see: `p^T A p` measures the symmetric part and stays positive.
    #[test]
    fn a_transpose_table_missing_an_entry_is_refused() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let rows = tet_rows();
        let (index, transpose_full) = tables(&rows);
        // Drop the last pair of the last non-empty transpose row, the way a
        // one-off in the table build would.
        let mut lists: Vec<Vec<Vec2u>> = Vec::new();
        for r in 0..transpose_full.size as usize {
            let begin = unsafe { *transpose_full.offset.add(r) } as usize;
            let end = unsafe { *transpose_full.offset.add(r + 1) } as usize;
            let mut list: Vec<Vec2u> = (begin..end)
                .map(|k| unsafe { *transpose_full.data.add(k) })
                .collect();
            if r + 1 == transpose_full.size as usize && !list.is_empty() {
                list.pop();
            }
            lists.push(list);
        }
        let short = CVecVec::from(&lists[..]);
        // The refusal happens in `validate`, before the handles matter, so the
        // image here is only what the signature needs.
        let short_pattern = pattern_of(&index, &short);
        let short_dev = PatternDevice::of(&mut device, &short_pattern);
        let fatal = FixedCsr::new(&mut device, short_dev.refs(), short_pattern)
            .expect_err("a short transpose table must be refused");
        assert!(
            fatal.detail.contains("non-symmetric"),
            "the message must name what a missing entry costs, got: {}",
            fatal.detail
        );
        // `transpose_full` owns the pairs `lists` was copied out of, so it must
        // outlive the copy; naming it here states that rather than leaving it to
        // drop order.
        drop(transpose_full);
    }

    #[test]
    fn a_transpose_entry_filed_under_the_wrong_column_is_refused() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let rows = tet_rows();
        let (index, _correct) = tables(&rows);
        // Slot 1 is block (0, 1); file it under column 2 instead.
        let mut lists: Vec<Vec<Vec2u>> = vec![Vec::new(); rows.len()];
        let mut slot = 0u32;
        for (i, row) in rows.iter().enumerate() {
            for &j in row {
                if i as u32 != j {
                    let filed_under = if slot == 1 { 2 } else { j };
                    lists[filed_under as usize].push(Vec2u::new(i as u32, slot));
                }
                slot += 1;
            }
        }
        let wrong = CVecVec::from(&lists[..]);
        let pattern = pattern_of(&index, &wrong);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let fatal = FixedCsr::new(&mut device, pattern_dev.refs(), pattern)
            .expect_err("a misfiled transpose entry must be refused");
        assert!(fatal.detail.contains("files slot"), "{}", fatal.detail);
    }

    #[test]
    fn an_empty_scene_has_an_empty_matrix_rather_than_an_error() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let (index, transpose) = tables(&[]);
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("no rows is not an error");
        assert_eq!(matrix.rows(), 0);
        assert_eq!(matrix.nnz(), 0);
    }

    /// The verdict is PER BLOCK, and only a MIXED batch shows that.
    ///
    /// A batch where every block lands and a batch where none does are both
    /// passed by a record that wired `stored` to the wrong length, to the wrong
    /// buffer, or to a single slot: each would fill the array uniformly and
    /// agree with the expected answer. One dropped block in the middle of a
    /// batch that otherwise lands is what separates the wiring from the answer,
    /// and it is the shape a missing element stencil actually produces.
    #[test]
    fn a_mixed_batch_reports_each_blocks_own_verdict() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        // Row 0 couples to 0 and 1, so the pattern has no slot for (0, 2).
        let rows = vec![vec![0, 1], vec![1, 2], vec![2]];
        let (index, transpose) = tables(&rows);
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let mut matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");

        let push_rows = [0u32, 0, 1];
        let push_cols = [0u32, 2, 2];
        let blocks: Vec<f32> = (0..27).map(|k| 1.0 + k as f32).collect();
        // Seeded with a value neither verdict uses, so an untouched slot is
        // distinguishable from a written one.
        let mut stored = [7u32; 3];
        matrix
            .push_blocks(
                &mut device,
                &mut staging,
                &push_rows,
                &push_cols,
                &blocks,
                &mut stored,
            )
            .expect("the call itself is well formed");
        assert_eq!(
            stored,
            [1, 0, 1],
            "the middle block is the only one outside the pattern, so the \
             verdict must be per block rather than per batch"
        );
        // And the two that landed did land, so the drop cost only its own block.
        // The push is a DISPATCH: bring the mirror current before reading it.
        matrix.download(&mut device).expect("the values read back");
        assert_eq!(matrix.read(0, 0)[..], blocks[0..9]);
        assert_eq!(matrix.read(1, 2)[..], blocks[18..27]);
    }

    /// Two blocks of one batch that share a slot must SUM.
    ///
    /// The fold is `compute::atomic_add`, which the host seam spells as a plain
    /// read-add-write, so a batch that overwrote instead of accumulating would
    /// keep only the last contribution and still produce a plausible matrix. A
    /// vertex shared by two elements is the ordinary case, not a corner one.
    #[test]
    fn two_blocks_of_one_batch_that_share_a_slot_are_summed() {
        // ONE DEVICE PER FIXTURE, allocations and dispatches alike: a
        // handle from one allocator dispatched against another is a
        // segfault rather than a diagnostic.
        let mut device = host_device();
        let rows = vec![vec![0, 1], vec![1]];
        let (index, transpose) = tables(&rows);
        let pattern = pattern_of(&index, &transpose);
        // The push takes the pattern's two arrays as handles, so the
        // fixture owns their device image the way the state does.
        let pattern_dev = PatternDevice::of(&mut device, &pattern);
        let mut staging = super::super::state::PushStaging::default();
        let mut matrix = FixedCsr::new(&mut device, pattern_dev.refs(), pattern).expect("valid pattern");

        let push_rows = [0u32, 0];
        let push_cols = [1u32, 1];
        let mut blocks = vec![0.0f32; 18];
        for k in 0..9 {
            blocks[k] = 1.0 + k as f32;
            blocks[9 + k] = 100.0 * (1.0 + k as f32);
        }
        let mut stored = [0u32; 2];
        matrix
            .push_blocks(
                &mut device,
                &mut staging,
                &push_rows,
                &push_cols,
                &blocks,
                &mut stored,
            )
            .expect("both pairs are in the pattern");
        assert_eq!(stored, [1, 1]);
        // The push is a DISPATCH: bring the mirror current before reading it.
        matrix.download(&mut device).expect("the values read back");
        let landed = matrix.read(0, 1);
        for k in 0..9 {
            assert_eq!(landed[k], blocks[k] + blocks[9 + k], "coefficient {k}");
        }
    }

    /// THE DECLARATION IS THE SERIAL RULE NOW, so the declaration is what a test
    /// has to hold.
    ///
    /// Moving this row to `Scatter::Disjoint` would let the backend cut the
    /// batch across threads, and two chunks folding into one slot through a
    /// plain read-add-write is a data race rather than a different fold order.
    /// A race does not fail reliably, so no value test can stand in for this
    /// one: what is checkable is the declaration itself.
    #[test]
    fn the_push_is_declared_atomic_so_a_backend_cannot_cut_the_batch() {
        use ppf_cts_compute::Scatter;
        use crate::driver::kernels::{id, TABLE};
        assert_eq!(
            TABLE[id::FIXED_CSR_ATOMIC_PUSH.0 as usize].scatter,
            Scatter::Atomic,
            "the fixed-matrix push accumulates into slots two blocks of one \
             batch can share, and the running sum's order is part of the fp32 \
             answer; it may not be declared disjoint to gain parallelism"
        );
    }
}
