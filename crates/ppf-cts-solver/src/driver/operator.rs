// File: crates/ppf-cts-solver/src/driver/operator.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The Newton operator, `A + B + C`, as PCG sees it.
//!
//! `A` is the dynamic (contact) matrix, `B` the fixed-pattern matrix carrying
//! the elastic, stitch, strain-limit and pin-barrier blocks, and `C` the block
//! diagonal carrying inertia, the aerodynamic term and the pull springs.
//!
//! # Why this is a module and not three calls in the driver
//!
//! `spmv_apply_row` in `src/kernels/solver/spmv.kernel.cpp` accumulates the
//! three into ONE fp32 running sum, in that order, and this module only chooses
//! which rows a thread walks. Composing them in Rust instead would be a float
//! add outside a shared body, which rule 8 forbids, and it would also be a
//! DIFFERENT association order from the one every backend compiles. The second
//! half is the one that matters: an fp32 sum is not associative, so the two
//! spellings disagree in the last bits of every matvec, and PCG amplifies that
//! disagreement over its iterations into a different search direction. There is
//! no tolerance for that, because the three backends are supposed to be one
//! implementation.
//!
//! # The dynamic half is optional and its absence is spelled, not branched
//!
//! A contact-free scene hands the composition a `None`, which becomes a null
//! `dyn_offset` and a zero count on every row: `dynamic_csr_apply_row` then
//! returns the zero vector, adding it is exact, and what the driver runs is the
//! same three-term composition rather than a reduced form that would have to be
//! re-derived when contact appears. The same is true of the preconditioner's
//! diagonal, whose `A(i, i)` term is then zero.

use ppf_cts_compute::{Device, Encoder, EncoderExt, Fault};
use super::kernels::{
    OperatorApplyArgs, OperatorApplyDynamicArgs, OperatorApplyDynamicFoldedArgs,
    OperatorApplyFoldedArgs, OperatorApplySymmetricFoldedArgs, PrecondDiagonalArgs,
    PrecondDiagonalDynamicArgs, VecFillArgs,
};
use super::spmv::FixedCsrView;

/// Rows per group in the FOLDED apply, so one lane owns one row and the
/// parallelism is exactly the element form's.
///
/// 256 is `reduce::BLOCK`, so a scene's group count here is the same number the
/// element form's first fold level produces and the two forms hand the fold
/// that follows the same extent.
pub const APPLY_GROUP: u32 = 256;

/// Lanes that share one row in the dynamic apply.
///
/// A row's non-zeros are strided across eight lanes and folded into the first
/// of them. It is spelled here as well as in
/// `src/kernels/solver/spmv.kernel.cpp` because the DISPATCH has to agree with
/// the body about how many threads a row wants, and a kernel cannot tell its
/// launcher.
pub const SPMV_ROW_LANES: u32 = 8;

/// Rows at or above which the apply takes the SYMMETRIC form.
///
/// **MEASURED ON THIS TREE'S OWN TWO KERNELS.** At `trapped`'s 135,396 rows the
/// walk runs the scene in 101.94 s and the symmetric form in 108.93, while at
/// `twist`'s 203,965 rows the walk takes 345.05 and the symmetric 310.03. So the
/// crossover sits ABOVE 135,396 and at or below 203,965.
///
/// A threshold carried over from a different pair of kernels does not transfer.
/// Both forms here lane a row eight ways ([`SPMV_ROW_LANES`]), which speeds the
/// walk relative to the symmetric form, and a pair that laned only the walk
/// would cross over lower.
///
/// **200,000 IS THE LARGEST VALUE THE MEASUREMENT ADMITS, and the bracket is
/// two scenes wide rather than finely resolved.** Preferring the larger end is
/// deliberate: the walk is the form whose summation order is fixed, so where the
/// measurement is silent this keeps the reproducible one. A scene between those
/// two counts would narrow it, and none ships.
pub const SYM_MIN_ROWS: u32 = 200_000;

/// Every buffer one row of the whole operator is stored in.
///
/// The mapping to a kernel's parameter names is the one `apply` already makes:
/// the dynamic matrix's TRANSPOSE arrays are what a kernel calls its REFERENCE
/// arrays, and the global value buffer they index into is the dynamic matrix's
/// own values. Spelled once here so a second consumer does not have to rederive
/// it from a dispatch.
impl Operator {
    /// The operator's buffers, flat, for a pass that reads the whole matrix.
    ///
    /// A scene with no contact has no dynamic matrix, and the four handles it
    /// would supply become an empty offset array: a kernel then walks a span of
    /// length zero at every row, which is the same answer by a different route.
    pub fn rows_view(&self, empty: ppf_cts_compute::Handle) -> super::schwarz::OperatorRows {
        let dynamic = self.dynamic;
        super::schwarz::OperatorRows {
            dynamic_index: dynamic.map_or(empty, |d| d.index),
            dynamic_value: dynamic.map_or(empty, |d| d.value),
            dynamic_offset: dynamic.map_or(empty, |d| d.offset),
            reference_index: dynamic.map_or(empty, |d| d.transpose_index),
            reference_value: dynamic.map_or(empty, |d| d.transpose_value),
            reference_offset: dynamic.map_or(empty, |d| d.transpose_offset),
            global_value: dynamic.map_or(empty, |d| d.value),
            fixed_index: self.fixed.index,
            fixed_offset: self.fixed.offset,
            fixed_value: self.fixed.value,
            transpose_pair: self.fixed.transpose_pair,
            transpose_offset: self.fixed.transpose_offset,
            diagonal: self.diagonal,
        }
    }
}

/// `A`, the dynamic (contact) matrix, in the flat per-row form the apply reads.
///
/// The two halves are the row's own stored blocks and the transpose index that
/// reaches the lower triangle, which is how the upper-triangle storage serves a
/// symmetric matvec without a scatter.
#[derive(Clone, Copy)]
pub struct DynamicView {
    pub offset: ppf_cts_compute::Handle,
    pub index: ppf_cts_compute::Handle,
    pub value: ppf_cts_compute::Handle,
    pub transpose_offset: ppf_cts_compute::Handle,
    pub transpose_index: ppf_cts_compute::Handle,
    pub transpose_value: ppf_cts_compute::Handle,
}

/// The assembled Newton operator over one Newton iteration.
pub struct Operator {
    /// `A`, the dynamic (contact) matrix, absent on a contact-free scene.
    pub dynamic: Option<DynamicView>,
    /// `B`, the fixed-pattern matrix.
    pub fixed: FixedCsrView,
    /// `C`, one 3x3 block per row, column-major, `9 * rows` floats.
    pub diagonal: ppf_cts_compute::Handle,
}

impl Operator {
    /// Rows, which is the vertex count.
    pub fn rows(&self) -> u32 {
        self.fixed.rows
    }

    /// The argument record for `result = (A + B + C) x`.
    ///
    /// A null buffer reference is how the seam reads "no dynamic matrix", and
    /// every dependent reference is null with it so a partial wiring cannot be
    /// mistaken for a matrix. `super::launch` refuses a record whose halves
    /// disagree, which is the check that makes that spelling safe.
    ///
    /// # Safety
    /// Every slice must outlive the dispatch, and `result` must not alias any
    /// input.
    /// Dispatch `result = (A + B + C) x` through whichever of the two entry
    /// points this scene needs.
    ///
    /// WHICH ONE IS THIS DRIVER'S DECISION. Whether the scene carries a DYNAMIC
    /// (contact) matrix at all is a configuration, not an addressing choice,
    /// and all seven of its buffers come from one `Option` so a partial wiring
    /// cannot be mistaken for a matrix. The two entries share the finishing
    /// half of the sum; the dynamic one folds the A term in FIRST, which is
    /// where fp32 makes the order load-bearing.
    ///
    /// # Safety
    /// Every slice must outlive the dispatch, and `result` must not alias any
    /// input.
    unsafe fn dispatch_apply(
        &self,
        encoder: &mut dyn Encoder,
        rows: u32,
        x: ppf_cts_compute::Handle,
        result: ppf_cts_compute::Handle,
        absolute: ppf_cts_compute::Handle,
        curvature: ppf_cts_compute::Handle,
    ) -> Result<(), Fault> {
        let index = self.fixed.index;
        let offset = self.fixed.offset;
        let value = self.fixed.value;
        let transpose_pair = self.fixed.transpose_pair;
        let transpose_offset = self.fixed.transpose_offset;
        let diagonal = self.diagonal;
        let xr = x;
        let out = result;
        // `solver::apply` is called with `D = 0` from every PCG path; the
        // parameter exists because the operator carries it, not because this
        // caller uses it.
        let diagonal_shift = 0.0;
        match self.dynamic {
            Some(d) => {
                let args = OperatorApplyDynamicArgs {
                    dyn_index: d.index,
                    dyn_value: d.value,
                    dyn_offset: d.offset,
                    dyn_reference_index: d.transpose_index,
                    dyn_reference_value: d.transpose_value,
                    dyn_reference_offset: d.transpose_offset,
                    dyn_global_value: d.value,
                    index,
                    offset,
                    value,
                    transpose_pair,
                    transpose_offset,
                    diagonal,
                    diagonal_shift,
                    x: xr,
                    result: out,
                    absolute,
                    curvature,
                    count: rows,
                    seam_arena_count: 0,
                };
                encoder.elements(&args, rows)
            }
            None => {
                let args = OperatorApplyArgs {
                    index,
                    offset,
                    value,
                    transpose_pair,
                    transpose_offset,
                    diagonal,
                    diagonal_shift,
                    x: xr,
                    result: out,
                    absolute,
                    curvature,
                    count: rows,
                    seam_arena_count: 0,
                };
                encoder.elements(&args, rows)
            }
        }
    }

    /// Append `result = (A + B + C) x` to a region.
    ///
    /// Separate from [`Operator::apply`] so the same composition can be encoded
    /// into a recorded region: the seam's one deferred construct takes a list of
    /// dispatches, not a list of calls that each open their own boundary.
    ///
    /// # Panics
    /// If a length disagrees with the row count. A mismatch is a caller defect
    /// and this is the side that knows the lengths.
    ///
    /// # Safety
    /// `x` and `result` must outlive the dispatch this appends.
    /// `absolute` receives, per row, the sum of the MAGNITUDES of that row's
    /// signed contributions to `x^T A x`. The caller folds it into the scale
    /// the curvature bound is taken from; a caller that does not want it passes
    /// a buffer it ignores, which costs one store per row.
    pub unsafe fn encode_apply(
        &self,
        encoder: &mut dyn Encoder,
        x: ppf_cts_compute::Handle,
        result: ppf_cts_compute::Handle,
        absolute: ppf_cts_compute::Handle,
        curvature: ppf_cts_compute::Handle,
    ) -> Result<(), Fault> {
        let rows = self.fixed.rows;
        // THE OFFSET-LENGTH ASSERTS SURVIVED THE MIGRATION, against the handle
        // rather than a slice: a `Handle` carries `size` in ELEMENTS. Neither
        // `operator_apply` nor `operator_apply_dynamic` declares a
        // `[[seam::bound]]`, so these are the only place the invariant is
        // checked at all.
        assert_eq!(self.fixed.offset.size as usize, rows as usize + 1);
        assert_eq!(self.fixed.transpose_offset.size as usize, rows as usize + 1);
        if rows == 0 {
            return Ok(());
        }
        self.dispatch_apply(encoder, rows, x, result, absolute, curvature)
    }

    /// Append `result = (A + B + C) x` with the curvature and its bound folded
    /// PER GROUP, and answer how many groups were dispatched.
    ///
    /// The element form above writes one float per ROW into `absolute` and
    /// `curvature`, so the fold that follows starts from `rows` and walks a
    /// fixed-width tree. This writes one float per GROUP, which is `rows / 256`
    /// of them, and the fold that follows is a single dispatch: the block reduce
    /// lives inside `operator_apply_folded` itself, so the matvec and the first
    /// fold level are one pass over the rows rather than two.
    ///
    /// # Panics
    /// If a length disagrees with the row count, as [`Operator::encode_apply`].
    ///
    /// # Safety
    /// Every handle must outlive the dispatch this appends, and both partial
    /// arrays must name at least the returned group count.
    pub unsafe fn encode_apply_folded(
        &self,
        encoder: &mut dyn Encoder,
        x: ppf_cts_compute::Handle,
        result: ppf_cts_compute::Handle,
        curvature_total: ppf_cts_compute::Handle,
        absolute_total: ppf_cts_compute::Handle,
    ) -> Result<u32, Fault> {
        let rows = self.fixed.rows;
        assert_eq!(self.fixed.offset.size as usize, rows as usize + 1);
        assert_eq!(self.fixed.transpose_offset.size as usize, rows as usize + 1);
        if rows == 0 {
            return Ok(0);
        }
        let groups = rows.div_ceil(APPLY_GROUP);
        // THE DYNAMIC APPLY RUNS EIGHT LANES PER ROW, so its extent is eight
        // times the row count and it fills eight times as many groups. The
        // partial arrays are sized at one float per ROW and this is
        // `rows.div_ceil(32)`, so they still hold it; the assert below checks
        // the larger of the two rather than assuming.
        let dynamic_groups = (rows * SPMV_ROW_LANES).div_ceil(APPLY_GROUP);
        assert!(
            curvature_total.size >= dynamic_groups.max(groups)
                && absolute_total.size >= dynamic_groups.max(groups),
            "the partial arrays must hold one float per group: {groups} groups \
             against {} and {}",
            curvature_total.size,
            absolute_total.size
        );
        let index = self.fixed.index;
        let offset = self.fixed.offset;
        let value = self.fixed.value;
        let transpose_pair = self.fixed.transpose_pair;
        let transpose_offset = self.fixed.transpose_offset;
        let diagonal = self.diagonal;
        let diagonal_shift = 0.0;
        match self.dynamic {
            // THE SYMMETRIC APPLY ABOVE `SYM_MIN_ROWS`. Below that row count
            // the walk is cheaper, the mirror reads being L2-served while the
            // symmetric form's atomic traffic dominates instead; the threshold
            // is the measured crossover between the two, and [`SYM_MIN_ROWS`]
            // records the measurement it comes from.
            //
            // THE RESULT IS ZEROED FIRST because every row is accumulated
            // atomically, its own sum included, so no write overwrites a slot
            // and a stale value would survive the pass. The fill rides the same
            // submit as the apply, so the pair costs one boundary.
            Some(d) if rows >= SYM_MIN_ROWS => {
                let zero = VecFillArgs {
                    array: result,
                    value: 0.0,
                    count: 3 * rows,
                    seam_arena_count: 0,
                };
                encoder.elements(&zero, 3 * rows)?;
                let args = OperatorApplySymmetricFoldedArgs {
                    dyn_index: d.index,
                    dyn_value: d.value,
                    dyn_offset: d.offset,
                    index,
                    offset,
                    value,
                    diagonal,
                    diagonal_shift,
                    x,
                    result,
                    curvature_total,
                    absolute_total,
                    rows,
                    count: dynamic_groups,
                    seam_arena_count: 0,
                };
                encoder.groups(&args, dynamic_groups, APPLY_GROUP)?;
                return Ok(dynamic_groups);
            }
            Some(d) => {
                let args = OperatorApplyDynamicFoldedArgs {
                    dyn_index: d.index,
                    dyn_value: d.value,
                    dyn_offset: d.offset,
                    dyn_reference_index: d.transpose_index,
                    dyn_reference_value: d.transpose_value,
                    dyn_reference_offset: d.transpose_offset,
                    dyn_global_value: d.value,
                    index,
                    offset,
                    value,
                    transpose_pair,
                    transpose_offset,
                    diagonal,
                    diagonal_shift,
                    x,
                    result,
                    curvature_total,
                    absolute_total,
                    rows,
                    count: dynamic_groups,
                    seam_arena_count: 0,
                };
                encoder.groups(&args, dynamic_groups, APPLY_GROUP)?;
            }
            None => {
                let args = OperatorApplyFoldedArgs {
                    index,
                    offset,
                    value,
                    transpose_pair,
                    transpose_offset,
                    diagonal,
                    diagonal_shift,
                    x,
                    result,
                    curvature_total,
                    absolute_total,
                    rows,
                    count: groups,
                    seam_arena_count: 0,
                };
                encoder.groups(&args, groups, APPLY_GROUP)?;
                return Ok(groups);
            }
        }
        Ok(dynamic_groups)
    }

    /// `result = (A + B + C) x`, as one boundary of its own.
    pub fn apply<D: Device>(
        &self,
        device: &mut D,
        x: ppf_cts_compute::Handle,
        result: ppf_cts_compute::Handle,
        absolute: ppf_cts_compute::Handle,
        curvature: ppf_cts_compute::Handle,
    ) -> Result<(), Fault> {
        device.run("operator.apply", |encoder| {
            // Safety: both slices are borrowed for the whole call, and the
            // dispatch completes before `run` returns.
            unsafe { self.encode_apply(encoder, x, result, absolute, curvature) }
        })?;
        Ok(())
    }

    /// The block-Jacobi preconditioner's diagonal, `A(i, i) + B(i, i) + C[i]`.
    ///
    /// Built AFTER the Dirichlet pass, which is what makes a removed row's
    /// preconditioner fall out with no special case: that pass zeroes the row's
    /// stored blocks and sets its diagonal to the identity, so this returns the
    /// identity and its inverse is the identity.
    pub fn precond_diagonal<D: Device>(
        &self,
        device: &mut D,
        out: ppf_cts_compute::Handle,
    ) -> Result<(), Fault> {
        let rows = self.fixed.rows;
        if rows == 0 {
            return Ok(());
        }
        // WHICH OF THE TWO ENTRIES, and it is this driver's decision. Whether
        // the scene carries a dynamic matrix at all is a configuration, not an
        // addressing choice, so the pair exists and this picks; the two share
        // the finishing half of the sum and differ only in the term folded
        // FIRST, which is where fp32 makes the order load-bearing.
        let index = self.fixed.index;
        let offset = self.fixed.offset;
        let value = self.fixed.value;
        let diagonal = self.diagonal;

        match self.dynamic {
            Some(d) => {
                let args = PrecondDiagonalDynamicArgs {
                    index,
                    offset,
                    value,
                    row_count: rows,
                    diagonal,
                    dyn_index: d.index,
                    dyn_offset: d.offset,
                    dyn_value: d.value,
                    out,
                    count: rows,
                    seam_arena_count: 0,
                };
                // Safety: every slice is borrowed for the whole call.
                device.run("operator.precond_diagonal", |encoder| unsafe {
                    encoder.elements(&args, rows)
                })?;
            }
            None => {
                let args = PrecondDiagonalArgs {
                    index,
                    offset,
                    value,
                    row_count: rows,
                    diagonal,
                    out,
                    count: rows,
                    seam_arena_count: 0,
                };
                // Safety: every slice is borrowed for the whole call.
                device.run("operator.precond_diagonal", |encoder| unsafe {
                    encoder.elements(&args, rows)
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;

    /// A 2-row matrix whose fixed part holds one off-diagonal coupling, plus a
    /// block diagonal, so the composition below is exercised on all three terms
    /// rather than on `B` alone.
    fn fixture() -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>) {
        // Pattern: row 0 holds (0,0) and (0,1); row 1 holds (1,1).
        let index = vec![0u32, 1, 1];
        let offset = vec![0u32, 2, 3];
        let mut value = vec![0.0f32; 9 * 3];
        // (0,0) = 2I, (0,1) = 0.5I, (1,1) = 3I.
        for k in 0..3 {
            value[4 * k] = 2.0;
            value[9 + 4 * k] = 0.5;
            value[18 + 4 * k] = 3.0;
        }
        // Row 1 receives (0,1)^T from row 0: one pair (source_row 0, slot 1).
        let transpose_pair = vec![0u32, 1];
        let transpose_offset = vec![0u32, 0, 1];
        // C: 1I on row 0, 10I on row 1.
        let mut diagonal = vec![0.0f32; 9 * 2];
        for k in 0..3 {
            diagonal[4 * k] = 1.0;
            diagonal[9 + 4 * k] = 10.0;
        }
        (index, offset, value, transpose_pair, transpose_offset, diagonal)
    }

    /// A diagonal-only operator of `rows` rows, big enough that the seam cuts it
    /// into parallel chunks.
    #[allow(clippy::type_complexity)]
    fn wide_fixture(rows: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>) {
        let index: Vec<u32> = (0..rows as u32).collect();
        let offset: Vec<u32> = (0..=rows as u32).collect();
        let mut value = vec![0.0f32; 9 * rows];
        let mut diagonal = vec![0.0f32; 9 * rows];
        for row in 0..rows {
            for k in 0..3 {
                value[9 * row + 4 * k] = 1.0 + (row % 13) as f32 * 0.125;
                diagonal[9 * row + 4 * k] = 0.5 + (row % 7) as f32 * 0.25;
            }
        }
        let transpose_pair: Vec<u32> = Vec::new();
        let transpose_offset: Vec<u32> = vec![0u32; rows + 1];
        (index, offset, value, transpose_pair, transpose_offset, diagonal)
    }


    /// A device INPUT holding `values`, since the matvec takes handles.
    /// As [`device_in`], for an index array.
    fn index_in(
        device: &mut impl Device,
        values: &[u32],
        label: &'static str,
    ) -> ppf_cts_compute::Buffer<u32> {
        let mut buffer = ppf_cts_compute::Buffer::<u32>::none();
        buffer
            .size(device, values.len(), ppf_cts_compute::AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, values)
            .expect("the test upload succeeds");
        buffer
    }

    fn device_in(
        device: &mut impl Device,
        values: &[f32],
        label: &'static str,
    ) -> ppf_cts_compute::Buffer<f32> {
        let mut buffer = ppf_cts_compute::Buffer::<f32>::none();
        buffer
            .size(device, values.len(), ppf_cts_compute::AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
            .write(device, 0, values)
            .expect("the test upload succeeds");
        buffer
    }

    /// A device OUTPUT with a mirror the assertions read.
    fn device_out(
        device: &mut impl Device,
        count: usize,
        label: &'static str,
    ) -> ppf_cts_compute::ReadbackBuffer<f32> {
        let mut buffer = ppf_cts_compute::ReadbackBuffer::<f32>::default();
        buffer
            .size(device, count, ppf_cts_compute::AllocLabel(label))
            .expect("the test allocation succeeds");
        buffer
    }

    #[test]
    fn the_composition_carries_the_fixed_matrix_and_the_block_diagonal() {
        let (index, offset, value, tp, to, diagonal) = fixture();
        let mut device = host_device();
        let diagonal_in = device_in(&mut device, &diagonal, "test.diagonal");
        // The view is all handles, so the fixture's five host arrays are
        // uploaded once and owned for the length of the test.
        let index_dev = index_in(&mut device, &index, "test.index");
        let offset_dev = index_in(&mut device, &offset, "test.offset");
        let value_dev = device_in(&mut device, &value, "test.value");
        let tp_dev = index_in(&mut device, &tp, "test.tpair");
        let to_dev = index_in(&mut device, &to, "test.toffset");
        let op = Operator {
            dynamic: None,
            fixed: FixedCsrView {
                index: index_dev.span(0, index_dev.len()),
                offset: offset_dev.span(0, offset_dev.len()),
                value: value_dev.span(0, value_dev.len()),
                transpose_pair: tp_dev.span(0, tp_dev.len()),
                transpose_offset: to_dev.span(0, to_dev.len()),
                rows: 2,
            },
            diagonal: diagonal_in.handle(),
        };
        let x_host = vec![1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];
        let x = device_in(&mut device, &x_host, "test.x");
        let mut result = device_out(&mut device, 6, "test.result");
        let mut absolute: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut curvature: ppf_cts_compute::Buffer<f32> = Default::default();
        absolute.size(&mut device, op.rows() as usize, ppf_cts_compute::AllocLabel("test.absolute")).expect("size");
        curvature.size(&mut device, op.rows() as usize, ppf_cts_compute::AllocLabel("test.curvature")).expect("size");
        op.apply(&mut device, x.handle(), result.handle(), absolute.handle(), curvature.handle()).expect("apply");
        result.download(&mut device).expect("the result reads back");
        let result = result.host();
        // Row 0: 2*1 + 0.5*2 (fixed) + 1*1 (diagonal) = 4.
        // Row 1: 0.5*1 (transpose) + 3*2 (fixed) + 10*2 (diagonal) = 26.5.
        assert_eq!(result[0], 4.0);
        assert_eq!(result[3], 26.5);
    }

    /// THE ROUND-OFF BOUND IS ACCUMULATED COMPONENT BY COMPONENT.
    ///
    /// `spmv_apply_row` carries a `Vec3f abssum`, folds `|v[c]|` into it as each
    /// block is multiplied, and forms `absolute = sum_k |x_k| * abssum_k` once
    /// at the end. Accumulating the scalar `|x . v|` per block instead is a
    /// STRICTLY TIGHTER number and therefore a different bound, so the
    /// `pAp <= 0` guard would classify against a different threshold and the
    /// three backends would disagree about which curvatures are resolvable.
    ///
    /// THE FIXTURE IS CHOSEN SO THE TWO FORMS DISAGREE, which most do not: with
    /// one block and a vector parallel to it the two are equal, and that is why
    /// the difference survived unnoticed. Here the single block maps
    /// `x = (1, 1, 0)` to `v = (1, -1, 0)`, so the dot cancels to exactly zero
    /// while the componentwise magnitudes do not.
    ///
    ///   curvature = 1*1 + 1*(-1) = 0
    ///   absolute  = |1|*1 + |1|*1 = 2      (the scalar form would give 0)
    #[test]
    fn the_curvature_bound_is_accumulated_componentwise() {
        // One row, one block at (0, 0), equal to diag(1, -1, 0).
        let index = vec![0u32];
        let offset = vec![0u32, 1];
        let mut value = vec![0.0f32; 9];
        value[0] = 1.0;
        value[4] = -1.0;
        value[8] = 0.0;
        let transpose_pair: Vec<u32> = vec![];
        let transpose_offset = vec![0u32, 0];
        let diagonal = vec![0.0f32; 9];

        let mut device = host_device();
        let diagonal_in = device_in(&mut device, &diagonal, "b.diagonal");
        let index_dev = index_in(&mut device, &index, "b.index");
        let offset_dev = index_in(&mut device, &offset, "b.offset");
        let value_dev = device_in(&mut device, &value, "b.value");
        let tp_dev = index_in(&mut device, &transpose_pair, "b.tpair");
        let to_dev = index_in(&mut device, &transpose_offset, "b.toffset");
        let op = Operator {
            dynamic: None,
            fixed: FixedCsrView {
                index: index_dev.span(0, index_dev.len()),
                offset: offset_dev.span(0, offset_dev.len()),
                value: value_dev.span(0, value_dev.len()),
                transpose_pair: tp_dev.span(0, tp_dev.len()),
                transpose_offset: to_dev.span(0, to_dev.len()),
                rows: 1,
            },
            diagonal: diagonal_in.handle(),
        };

        let x_host = vec![1.0f32, 1.0, 0.0];
        let x = device_in(&mut device, &x_host, "b.x");
        let mut result = device_out(&mut device, 3, "b.result");
        let mut absolute = device_out(&mut device, 1, "b.absolute");
        let mut curvature = device_out(&mut device, 1, "b.curvature");
        op.apply(
            &mut device,
            x.handle(),
            result.handle(),
            absolute.handle(),
            curvature.handle(),
        )
        .expect("apply");

        absolute.download(&mut device).expect("absolute reads back");
        curvature.download(&mut device).expect("curvature reads back");
        assert_eq!(
            curvature.host()[0],
            0.0,
            "the signed curvature must cancel to zero on this row"
        );
        assert_eq!(
            absolute.host()[0],
            2.0,
            "the bound must be sum_k |x_k| * abssum_k, which is 2 here; a \
             value of 0 means the magnitudes were contracted against x inside \
             the block walk, which is the tighter scalar form this \
             bound is deliberately not"
        );
    }

    #[test]
    fn the_answer_does_not_depend_on_the_thread_count() {
        // The composition is a pure gather, so a row belongs to exactly one
        // chunk and the seam cuts by a fixed chunk width rather than by the
        // thread count. This is the operator's end of that guarantee; the
        // backend's own test covers the cut itself.
        let rows = 8192usize;
        let (index, offset, value, tp, to, diagonal) = wide_fixture(rows);
        let x: Vec<f32> = (0..3 * rows).map(|i| ((i % 31) as f32 - 15.0) * 0.125).collect();
        let run = || {
            let mut device = host_device();
            let diagonal_in = device_in(&mut device, &diagonal, "test.diagonal");
        // The view is all handles, so the fixture's five host arrays are
        // uploaded once and owned for the length of the test.
        let index_dev = index_in(&mut device, &index, "test.index");
        let offset_dev = index_in(&mut device, &offset, "test.offset");
        let value_dev = device_in(&mut device, &value, "test.value");
        let tp_dev = index_in(&mut device, &tp, "test.tpair");
        let to_dev = index_in(&mut device, &to, "test.toffset");
            let op = Operator {
                dynamic: None,
                fixed: FixedCsrView {
                    index: index_dev.span(0, index_dev.len()),
                    offset: offset_dev.span(0, offset_dev.len()),
                    value: value_dev.span(0, value_dev.len()),
                    transpose_pair: tp_dev.span(0, tp_dev.len()),
                    transpose_offset: to_dev.span(0, to_dev.len()),
                    rows: rows as u32,
                },
                diagonal: diagonal_in.handle(),
            };
            let x_in = device_in(&mut device, &x, "test.x");
            let mut result = device_out(&mut device, 3 * rows, "test.result");
            let mut absolute: ppf_cts_compute::Buffer<f32> = Default::default();
            let mut curvature: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut curvature: ppf_cts_compute::Buffer<f32> = Default::default();
            absolute.size(&mut device, op.rows() as usize, ppf_cts_compute::AllocLabel("test.absolute")).expect("size");
        curvature.size(&mut device, op.rows() as usize, ppf_cts_compute::AllocLabel("test.curvature")).expect("size");
            op.apply(&mut device, x_in.handle(), result.handle(), absolute.handle(), curvature.handle())
                .expect("apply");
            result
                .download(&mut device)
                .expect("the result reads back");
            result.host().to_vec()
        };
        let reference = run();
        for threads in [1usize, 2, 3, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(run);
            for (i, (a, b)) in reference.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "element {i} moved at {threads} threads; determinism is a \
                     stated guarantee, not an accident of scheduling"
                );
            }
        }
    }

    #[test]
    fn the_preconditioner_diagonal_is_the_fixed_diagonal_plus_the_block() {
        let (index, offset, value, tp, to, diagonal) = fixture();
        let mut device = host_device();
        let diagonal_in = device_in(&mut device, &diagonal, "test.diagonal");
        // The view is all handles, so the fixture's five host arrays are
        // uploaded once and owned for the length of the test.
        let index_dev = index_in(&mut device, &index, "test.index");
        let offset_dev = index_in(&mut device, &offset, "test.offset");
        let value_dev = device_in(&mut device, &value, "test.value");
        let tp_dev = index_in(&mut device, &tp, "test.tpair");
        let to_dev = index_in(&mut device, &to, "test.toffset");
        let op = Operator {
            dynamic: None,
            fixed: FixedCsrView {
                index: index_dev.span(0, index_dev.len()),
                offset: offset_dev.span(0, offset_dev.len()),
                value: value_dev.span(0, value_dev.len()),
                transpose_pair: tp_dev.span(0, tp_dev.len()),
                transpose_offset: to_dev.span(0, to_dev.len()),
                rows: 2,
            },
            diagonal: diagonal_in.handle(),
        };
        // A DEVICE ALLOCATION, because the pass writes through a handle now, and
        // a mirror to read the eighteen floats back out of.
        let mut out_buf = ppf_cts_compute::ReadbackBuffer::<f32>::default();
        out_buf
            .size(&mut device, 18, ppf_cts_compute::AllocLabel("test.precond_out"))
            .expect("the test allocation succeeds");
        op.precond_diagonal(&mut device, out_buf.handle())
            .expect("precond diagonal");
        out_buf
            .download(&mut device)
            .expect("the precond mirror refreshes");
        let out = out_buf.host();
        // Row 0: B(0,0) = 2I plus C[0] = 1I. Row 1: B(1,1) = 3I plus C[1] = 10I.
        assert_eq!(out[0], 3.0);
        assert_eq!(out[4], 3.0);
        assert_eq!(out[9], 13.0);
        assert_eq!(out[13], 13.0);
        // The off-diagonal entries of a diagonal-only fixture stay zero, which
        // is what says the read picked the (i, i) slot and not a neighbor.
        assert_eq!(out[1], 0.0);
        assert_eq!(out[10], 0.0);
    }
}
