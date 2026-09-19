// File: schwarz.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The Schwarz preconditioner's host half.
//!
//! The whole multilevel additive Schwarz preconditioner is here: the heavy-edge
//! aggregation ([`hem_partition`]), the domain construction
//! ([`build_domains`]), the per-domain factorization ([`factor_domains`]), the
//! Galerkin coarsening ([`galerkin`]), the level hierarchy
//! ([`build_hierarchy`]) and the sweep ([`encode_whole_sweep`]). The kernel
//! bodies it dispatches are in `src/kernels/schwarz/schwarz.kernel.cpp`, so
//! every backend runs the same hierarchy from the same sources.

use ppf_cts_compute::{Device, Encoder, EncoderExt, Fault, Handle};


use super::kernels::{
    SchwarzCountMembersArgs, SchwarzDomainInverseSizeArgs,
    SchwarzFactorCholeskyColumnArgs, SchwarzFactorCholeskyDiagonalArgs,
    SchwarzFactorFloorArgs, SchwarzFactorGatherArgs, SchwarzFactorInverseColumnArgs,
    SchwarzFactorPackArgs, SchwarzFineGraphCountArgs, SchwarzFineGraphFillArgs,
    SchwarzApplyGatherArgs, SchwarzApplyLowerArgs, SchwarzApplyUpperArgs,
    SchwarzCoarseGatherArgs, SchwarzComposeMapRowArgs,
    SchwarzGalerkinEdgeFlagArgs, SchwarzGalerkinEdgeHeadArgs,
    SchwarzGalerkinKeyArgs, SchwarzGalerkinSegmentSumArgs,
    SchwarzLevel0CountArgs, SchwarzLevel0FillArgs,
    SchwarzProlongRowArgs, SchwarzRestrictRowArgs,
    SchwarzScatterMembersArgs, VecFillU32Args,
};

/// Turn a per-vertex aggregate assignment into the three arrays the factor and
/// the apply read.
///
/// The shape is CSR: count the members of each aggregate, scan the counts into
/// starts, then claim a slot per vertex and write it. `inverse_offset` is the
/// same pattern once more over the packed lower triangle each domain's
/// factorization needs.
///
/// # Where the total is
///
/// NOT RETURNED, and deliberately. A dispatch produces no value, so returning
/// the packed-inverse total would mean a `download()` inside this function
/// whether or not the caller needs it yet. It is the LAST SLOT of
/// `inverse_offset`, written by the same scan that filled the rest, and the
/// caller reads it with its own readback at the point it sizes the inverse
/// array. `dyncsr` reads its own total the same way.
///
/// # Safety
///
/// Every handle must name a live allocation: `aggregate` holds `vertices`
/// entries, `offset` and `inverse_offset` hold `aggregates + 1`, `cursor` holds
/// `aggregates`, and `members` holds `vertices`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_domains<D: Device>(
    device: &mut D,
    aggregate: Handle,
    vertices: u32,
    aggregates: u32,
    offset: Handle,
    members: Handle,
    inverse_offset: Handle,
    cursor: Handle,
    scan_scratch: &mut super::scan::ScanScratch,
) -> Result<(), Fault> {
    if aggregates == 0 {
        return Ok(());
    }
    // THE COUNTS OPEN AT ZERO, and the fill covers `aggregates + 1` rather than
    // `aggregates`: the scan writes the total into the last slot, and a slot
    // the scan is about to overwrite still has to be a defined value first,
    // because Metal returns whatever bytes an unwritten allocation holds rather
    // than faulting.
    let zero = VecFillU32Args {
        array: offset,
        value: 0,
        count: aggregates + 1,
        seam_arena_count: 0,
    };
    device.launch("schwarz.zero_offset", &zero, aggregates + 1)?;

    let count = SchwarzCountMembersArgs {
        aggregate,
        offset,
        count: vertices,
        seam_arena_count: 0,
    };
    device.launch("schwarz.count_members", &count, vertices)?;

    // THE EXCLUSIVE SCAN, which reads a count array and writes exclusive starts
    // with the total in the final slot. It is multi-level rather than a single
    // pass at EXTENT ONE, which would be one device thread walking the array in
    // sequence; see [`super::scan`].
    scan_scratch.exclusive(device, "schwarz.scan_offset", offset, aggregates)?;

    let zero_cursor = VecFillU32Args {
        array: cursor,
        value: 0,
        count: aggregates,
        seam_arena_count: 0,
    };
    device.launch("schwarz.zero_cursor", &zero_cursor, aggregates)?;

    let scatter = SchwarzScatterMembersArgs {
        aggregate,
        offset,
        cursor,
        members,
        count: vertices,
        seam_arena_count: 0,
    };
    device.launch("schwarz.scatter_members", &scatter, vertices)?;

    let sizes = SchwarzDomainInverseSizeArgs {
        offset,
        inverse_offset,
        count: aggregates,
        seam_arena_count: 0,
    };
    device.launch("schwarz.domain_inverse_size", &sizes, aggregates)?;

    // The same scan again, over the per-domain inverse sizes.
    scan_scratch.exclusive(device, "schwarz.scan_inverse", inverse_offset, aggregates)?;
    Ok(())
}

/// Every buffer one row of the Newton operator is stored in.
///
/// A record rather than fifteen parameters, because the fine graph reads the
/// WHOLE operator and the two passes below read the same set. It carries only
/// handles: nothing here is host-addressable, which is why the census counts
/// zero for all of it.
#[derive(Clone, Copy)]
pub struct OperatorRows {
    pub dynamic_index: Handle,
    pub dynamic_value: Handle,
    pub dynamic_offset: Handle,
    pub reference_index: Handle,
    pub reference_value: Handle,
    pub reference_offset: Handle,
    pub global_value: Handle,
    pub fixed_index: Handle,
    pub fixed_offset: Handle,
    pub fixed_value: Handle,
    pub transpose_pair: Handle,
    pub transpose_offset: Handle,
    /// The operator's per-vertex 3x3 diagonal, which is NEITHER matrix's.
    /// The Newton operator is the two matrices plus this block, and this
    /// block is the dominant term: it carries `mass / dt^2` and every
    /// energy's diagonal contribution.
    pub diagonal: Handle,
}

/// Factor every domain: `dense = L L^T`, then `work = L^{-1}`, then packed.
///
/// SIX DISPATCHES, whose phases are separated by KERNEL COMPLETION rather than
/// by a threadgroup barrier, because kernel completion is the only separation
/// the CPU backend has: `kernelgen.py`
/// runs a group entry's lanes one after another and leaves
/// `compute::threadgroup_barrier` undefined, so a cooperative factorization has
/// no host rendering at all.
///
/// THE COLUMN LOOP IS SEQUENTIAL AND SO ARE ITS DISPATCHES. Column `j` of a
/// Cholesky depends on every column before it, so the two rows inside the loop
/// run once per column, over EVERY aggregate at that column: an aggregate whose
/// dimension the column has passed does nothing, which is what lets one
/// dispatch serve domains of different sizes.
///
/// # Safety
///
/// `dense` and `work` must each hold `dense_offset[aggregates]` floats,
/// `packed` must hold `inverse_offset[aggregates]`, and `floor` one float per
/// aggregate. `max_dimension` must be at least `3 * kmax` for the partition
/// that produced these domains, or a column past it is never taken and the
/// factor is silently incomplete.
#[allow(clippy::too_many_arguments)]
pub unsafe fn factor_domains<D: Device>(
    device: &mut D,
    rows: OperatorRows,
    aggregates: u32,
    max_members: u32,
    max_dimension: u32,
    aggregate_offset: Handle,
    members: Handle,
    dense_offset: Handle,
    inverse_offset: Handle,
    dense: Handle,
    work: Handle,
    floor: Handle,
    packed: Handle,
) -> Result<(), Fault> {
    if aggregates == 0 {
        return Ok(());
    }
    // THE DENSE BLOCK OPENS AT ZERO because the gather ACCUMULATES into it: the
    // four operator spans each add, and a slot no span reaches must be the zero
    // it would have been. Metal returns whatever bytes an unwritten allocation
    // holds rather than faulting, so this is not a formality.
    let cells = max_dimension * max_dimension;
    let zero = VecFillU32Args {
        array: dense,
        value: 0,
        count: aggregates * cells,
        seam_arena_count: 0,
    };
    device.launch("schwarz.zero_dense", &zero, aggregates * cells)?;

    let gather = SchwarzFactorGatherArgs {
        aggregate_offset,
        members,
        dense_offset,
        dynamic_index: rows.dynamic_index,
        dynamic_value: rows.dynamic_value,
        dynamic_offset: rows.dynamic_offset,
        reference_index: rows.reference_index,
        reference_value: rows.reference_value,
        reference_offset: rows.reference_offset,
        global_value: rows.global_value,
        fixed_index: rows.fixed_index,
        fixed_offset: rows.fixed_offset,
        fixed_value: rows.fixed_value,
        transpose_pair: rows.transpose_pair,
        transpose_offset: rows.transpose_offset,
        diagonal: rows.diagonal,
        dense,
        stride: max_members,
        count: aggregates * max_members,
        seam_arena_count: 0,
    };
    device.launch("schwarz.factor_gather", &gather, aggregates * max_members)?;
    factor_from_dense(
        device,
        aggregates,
        max_dimension,
        aggregate_offset,
        dense_offset,
        inverse_offset,
        dense,
        work,
        floor,
        packed,
    )
}

/// The factorization's phases AFTER the gather, which are the same phases
/// whatever filled `dense`.
///
/// SPLIT OUT BECAUSE A COARSE LEVEL FILLS IT DIFFERENTLY: the fine gather walks
/// the operator's four spans and adds a per-vertex diagonal, while a coarse one
/// walks a single block-CSR row. Everything from the diagonal floor onward reads
/// `dense` and knows nothing about where it came from, so one copy serves both.
///
/// # Safety
/// As [`factor_domains`], with `dense` already filled.
#[allow(clippy::too_many_arguments)]
pub unsafe fn factor_from_dense<D: Device>(
    device: &mut D,
    aggregates: u32,
    max_dimension: u32,
    aggregate_offset: Handle,
    dense_offset: Handle,
    inverse_offset: Handle,
    dense: Handle,
    work: Handle,
    floor: Handle,
    packed: Handle,
) -> Result<(), Fault> {
    if aggregates == 0 {
        return Ok(());
    }
    let cells = max_dimension * max_dimension;
    let floor_args = SchwarzFactorFloorArgs {
        aggregate_offset,
        dense_offset,
        dense,
        floor_out: floor,
        count: aggregates,
        seam_arena_count: 0,
    };
    device.launch("schwarz.factor_floor", &floor_args, aggregates)?;

    for column in 0..max_dimension {
        let diagonal = SchwarzFactorCholeskyDiagonalArgs {
            aggregate_offset,
            dense_offset,
            floor_in: floor,
            dense,
            column,
            count: aggregates,
            seam_arena_count: 0,
        };
        device.launch("schwarz.cholesky_diagonal", &diagonal, aggregates)?;
        // The rows BELOW the diagonal, which is why the extent shrinks as the
        // column advances: at column `c` there are at most
        // `max_dimension - c - 1` of them.
        let below = max_dimension - column - 1;
        if below == 0 {
            continue;
        }
        let sub = SchwarzFactorCholeskyColumnArgs {
            aggregate_offset,
            dense_offset,
            dense,
            column,
            stride: below,
            count: aggregates * below,
            seam_arena_count: 0,
        };
        device.launch("schwarz.cholesky_column", &sub, aggregates * below)?;
    }

    let inverse = SchwarzFactorInverseColumnArgs {
        aggregate_offset,
        dense_offset,
        dense,
        work,
        stride: max_dimension,
        count: aggregates * max_dimension,
        seam_arena_count: 0,
    };
    device.launch("schwarz.inverse_column", &inverse, aggregates * max_dimension)?;

    let pack = SchwarzFactorPackArgs {
        aggregate_offset,
        dense_offset,
        inverse_offset,
        work,
        packed,
        stride: cells,
        count: aggregates * cells,
        seam_arena_count: 0,
    };
    device.launch("schwarz.factor_pack", &pack, aggregates * cells)?;
    Ok(())
}

/// Every buffer one additive sweep needs, so a PCG iteration can name one thing.
///
/// A record because the sweep is three dispatches over eight handles and the
/// PCG calls it once per iteration: threading eight arguments through the
/// iteration would put the preconditioner's shape in the solve's signature.
#[derive(Clone, Copy)]
pub struct Sweep {
    pub aggregates: u32,
    pub max_dimension: u32,
    pub aggregate_offset: Handle,
    pub members: Handle,
    pub inverse_offset: Handle,
    pub packed: Handle,
    pub residual_local: Handle,
    pub y_local: Handle,
}

/// One additive sweep, `result = M^{-1} x`, pushed onto a caller's encoder.
///
/// THREE ROWS: each domain's slice of `x` is gathered contiguous, multiplied
/// by `G`, then by `G^T` and scattered back.
/// The PCG batches an iteration's work into one submission, so the
/// preconditioner has to be expressible as encoder operations or the batching
/// breaks around it. An encoder orders the three rows, which is the ordering
/// the two matvecs need since the second reads what the first wrote.
///
/// SPD BY CONSTRUCTION, which is why the two-pass form rather than one
/// symmetric matvec: `r . z = ||G r||^2 >= 0` in float32 whatever the rounding,
/// so an aggregate term can never produce the `rz <= 0` breakdown that latches
/// the block-Jacobi fallback.
///
/// A VERTEX IS WRITTEN BY EXACTLY ONE AGGREGATE, the aggregates partitioning
/// the vertices, so the scatter assigns rather than accumulates and `result`
/// needs no clearing.
///
/// # Safety
/// Every handle in `sweep` must outlive the submission, `x` and `result` must
/// name three floats per vertex, and the two locals `aggregates *
/// max_dimension` floats each.
pub unsafe fn encode_apply(
    encoder: &mut dyn Encoder,
    sweep: Sweep,
    x: Handle,
    result: Handle,
) -> Result<(), Fault> {
    if sweep.aggregates == 0 {
        return Ok(());
    }
    let extent = sweep.aggregates * sweep.max_dimension;
    let gather = SchwarzApplyGatherArgs {
        aggregate_offset: sweep.aggregate_offset,
        members: sweep.members,
        x,
        residual_local: sweep.residual_local,
        stride: sweep.max_dimension,
        count: extent,
        seam_arena_count: 0,
    };
    encoder.elements(&gather, extent)?;
    let lower = SchwarzApplyLowerArgs {
        aggregate_offset: sweep.aggregate_offset,
        inverse_offset: sweep.inverse_offset,
        packed: sweep.packed,
        residual_local: sweep.residual_local,
        y_local: sweep.y_local,
        stride: sweep.max_dimension,
        count: extent,
        seam_arena_count: 0,
    };
    encoder.elements(&lower, extent)?;
    let upper = SchwarzApplyUpperArgs {
        aggregate_offset: sweep.aggregate_offset,
        members: sweep.members,
        inverse_offset: sweep.inverse_offset,
        packed: sweep.packed,
        y_local: sweep.y_local,
        result,
        stride: sweep.max_dimension,
        count: extent,
        seam_arena_count: 0,
    };
    encoder.elements(&upper, extent)
}

/// Level `l`'s map from a fine vertex, composed from the level below it.
///
/// # Safety
/// All three arrays hold one entry per fine vertex, and `previous_aggregate`
/// one per node of the level below.
pub unsafe fn compose_map<D: Device>(
    device: &mut D,
    previous_map: Handle,
    previous_aggregate: Handle,
    map_fine: Handle,
    vertices: u32,
) -> Result<(), Fault> {
    if vertices == 0 {
        return Ok(());
    }
    let args = SchwarzComposeMapRowArgs {
        previous_map,
        previous_aggregate,
        map_fine,
        count: vertices,
        seam_arena_count: 0,
    };
    device.launch("schwarz.compose_map", &args, vertices)?;
    Ok(())
}

/// A block-CSR matrix carrying BOTH triangles, which is what the multilevel
/// extension coarsens.
///
/// Symmetry is owned by STORING both triangles,
/// so a row is a single walk with no transpose bookkeeping; the fine operator's
/// one-triangle-plus-reference form is what [`materialize_level0`] converts out
/// of, and every level above is a Galerkin product that comes out this way.
#[derive(Default)]
pub struct CoarseMat {
    /// Block rows.
    pub rows: u32,
    /// Stored blocks, both triangles.
    pub blocks: u32,
    pub offset: ReadbackBuffer<u32>,
    pub column: Buffer<u32>,
    /// Nine floats per stored block, column-major.
    pub value: Buffer<f32>,
}

/// Write the fine operator out as a block-CSR matrix.
///
/// The four operator spans plus the per-vertex diagonal, one entry per stored
/// block.
///
/// NOTHING IS DEDUPLICATED. A vertex pair coupled through both matrices appears
/// twice, and every consumer sums a row's entries, so two entries at one column
/// are the same as one carrying their sum. The Galerkin product dedups by
/// sorting, which is where it matters.
///
/// # Safety
/// Every handle in `rows` must name a live allocation of the operator's arrays,
/// and `vertices` must be its row count.
pub unsafe fn materialize_level0<D: Device>(
    device: &mut D,
    out: &mut CoarseMat,
    rows: OperatorRows,
    vertices: u32,
    scan_scratch: &mut super::scan::ScanScratch,
) -> Result<(), Fault> {
    out.rows = vertices;
    if vertices == 0 {
        out.blocks = 0;
        return Ok(());
    }
    out.offset.size(device, vertices as usize + 1, AllocLabel("schwarz.m0_off"))?;
    // CLEARED BEFORE THE COUNT, EXPLICITLY. `level0_count` ACCUMULATES into
    // these slots, so they must open at zero, and a rebuild reuses the buffer.
    // `Buffer::size` zeroes only the bytes it has just allocated, so without
    // this a second build would count on top of the first one's offsets and the
    // scan would then read past the columns they address.
    device.fill_zero(
        out.offset.handle(),
        (vertices as usize + 1) * std::mem::size_of::<u32>(),
    )?;
    let count = SchwarzLevel0CountArgs {
        dynamic_offset: rows.dynamic_offset,
        reference_offset: rows.reference_offset,
        fixed_offset: rows.fixed_offset,
        transpose_offset: rows.transpose_offset,
        count: out.offset.handle(),
        count_of_rows: vertices,
        seam_arena_count: 0,
    };
    device.launch("schwarz.level0_count", &count, vertices)?;
        // The multi-level exclusive scan rather than a single pass at EXTENT
        // ONE, which would be one device thread walking the array in
        // sequence; see [`super::scan`].
        scan_scratch.exclusive(device, "schwarz.level0_scan", out.offset.handle(), vertices)?;
    out.offset.download(device)?;
    let blocks = *out.offset.host().last().expect("offsets are rows + 1") as usize;
    out.blocks = blocks as u32;

    out.column.size(device, blocks.max(1), AllocLabel("schwarz.m0_col"))?;
    out.value.size(device, 9 * blocks.max(1), AllocLabel("schwarz.m0_val"))?;
    let fill = SchwarzLevel0FillArgs {
        dynamic_index: rows.dynamic_index,
        dynamic_value: rows.dynamic_value,
        dynamic_offset: rows.dynamic_offset,
        reference_index: rows.reference_index,
        reference_value: rows.reference_value,
        reference_offset: rows.reference_offset,
        global_value: rows.global_value,
        fixed_index: rows.fixed_index,
        fixed_offset: rows.fixed_offset,
        fixed_value: rows.fixed_value,
        transpose_pair: rows.transpose_pair,
        transpose_offset: rows.transpose_offset,
        diagonal: rows.diagonal,
        coarse_offset: out.offset.handle(),
        column: out.column.handle(),
        value: out.value.handle(),
        count: vertices,
        seam_arena_count: 0,
    };
    device.launch("schwarz.level0_fill", &fill, vertices)?;
    Ok(())
}

/// The scratch the Galerkin coarsening reuses across levels and rebuilds.
///
/// THE SCRATCH IS OWNED RATHER THAN TAKEN PER CALL. [`Buffer::size`] grows
/// only past capacity, so a slot carried on the state is allocation-free after
/// the first build that sized it. A per-call allocation would be worse than a
/// pooled one here: nothing in this driver implements `Drop`, so a span
/// allocated per call is stranded rather than churned.
#[derive(Default)]
pub struct GalerkinScratch {
    /// The device sort's own arrays; see [`super::devsort`].
    sort: super::devsort::SortScratch,
    /// One packed `(coarse row, coarse column)` key per source entry.
    key: ReadbackBuffer<u32>,
    /// Which source entry each sorted slot came from.
    permutation: ReadbackBuffer<u32>,
    /// Per source entry: the run flag, then the scan that turns it into the
    /// coarse entry index. `entries + 1` slots, the last carrying the total.
    ///
    /// A PLAIN `Buffer` THOUGH THE HOST READS IT, and the element it reads is
    /// the one the scan wrote last, which is the total. [`Buffer::read_one`]
    /// copies that one scalar; a [`ReadbackBuffer`] would answer the same
    /// question by downloading the whole array, which is the relocation this
    /// routine exists to remove.
    edge: Buffer<u32>,
    /// Per coarse entry: where its run of sorted slots starts, plus a final
    /// slot holding the source entry count so the last run has an end.
    edge_start: Buffer<u32>,
}

/// `dst = C src C^T`, the Galerkin coarsening.
///
/// Every source entry `(i, j)` lands at the coarse entry `(agg[i], agg[j])`,
/// and the entries that land together are SUMMED.
/// That is exactly the triple product for a piecewise-constant `C`, since a
/// column of `C` is an indicator vector.
///
/// FOUR DISPATCHES AND TWO SCANS: pack each source entry's coarse pair into
/// one 32-bit key, sort the keys,
/// flag the first entry of each equal-key run, scan the flags into coarse entry
/// indices, take each run's head (its coarse column, its start, and one count
/// into its coarse row), scan those counts into the coarse row offsets, and
/// segment-sum the nine floats of each run. Both scans are the exclusive scan
/// [`build_domains`] uses, and for the same reason: each turns a count array
/// into the starts the next dispatch writes at.
///
/// # THE SORT RUNS ON THE DEVICE, THROUGH [`super::devsort`]
///
/// Bringing the key and permutation arrays down, sorting the pair on the host
/// and sending them back would move the sort itself off the device. Nothing
/// here does that: [`super::devsort`] is a stable radix sort over
/// `(key, index)` pairs, dispatched like any other pass, so the nine floats of
/// a block never leave the device and no arithmetic on them runs on the host.
///
/// THE ORDER IS TOTAL, and the next reader should not have to re-derive that: a
/// radix pass is stable and the passes run least significant digit first, so
/// equal keys keep ascending source index. Each coarse entry's run therefore
/// holds the same slots in the same order, and the float sum is accumulated in
/// the same order.
///
/// # Safety
/// `src` must be a materialized level, and `aggregate` must name a live
/// allocation of one entry per its rows, each below `coarse_rows`.
pub unsafe fn galerkin<D: Device>(
    device: &mut D,
    scratch: &mut GalerkinScratch,
    src: &mut CoarseMat,
    aggregate: Handle,
    coarse_rows: u32,
    dst: &mut CoarseMat,
    scan_scratch: &mut super::scan::ScanScratch,
) -> Result<(), Fault> {
    dst.rows = coarse_rows;
    if coarse_rows == 0 || src.rows == 0 {
        dst.blocks = 0;
        return Ok(());
    }
    let entries = src.blocks as usize;
    // THE COARSE ROW OFFSETS OPEN AT ZERO, because the head pass ADDS into
    // them, and they are sized before anything is known about how many coarse
    // entries there will be: the array is one per coarse row plus the total,
    // whatever the source holds. `size` zeroes, which is the clear this pass
    // needs.
    dst.offset
        .size(device, coarse_rows as usize + 1, AllocLabel("schwarz.gal_off"))?;
    // CLEARED FOR THE SAME REASON as the level-0 offsets above: the Galerkin
    // count accumulates, and a rebuild reuses the buffer.
    device.fill_zero(
        dst.offset.handle(),
        (coarse_rows as usize + 1) * std::mem::size_of::<u32>(),
    )?;
    if entries == 0 {
        // A source level with no stored block coarsens to one with none, and
        // its offsets are the zeros just written.
        //
        // THE TWO EMPTY ARRAYS ARE STILL ALLOCATED, at one entry. A consumer
        // names them whatever they hold, and an unallocated buffer hands out
        // `Handle::NONE`, which a generated entry rejects as an arena out of
        // range rather than reading nothing.
        dst.blocks = 0;
        dst.column
            .size(device, 1, AllocLabel("schwarz.gal_col"))?;
        dst.value.size(device, 9, AllocLabel("schwarz.gal_val"))?;
        return Ok(());
    }

    scratch
        .key
        .size(device, entries, AllocLabel("schwarz.gal_key"))?;
    scratch
        .permutation
        .size(device, entries, AllocLabel("schwarz.gal_perm"))?;
    let keys = SchwarzGalerkinKeyArgs {
        aggregate,
        offset: src.offset.handle(),
        column: src.column.handle(),
        key: scratch.key.handle(),
        permutation: scratch.permutation.handle(),
        coarse_rows,
        count: src.rows,
        seam_arena_count: 0,
    };
    device.launch("schwarz.galerkin_key", &keys, src.rows)?;

    // THE SORT, ON THE DEVICE. Bringing both arrays down, sorting the pair on
    // the host and sending them back would move the sort itself off the device,
    // so the keys are never seen by the host at all. The order is total, so it
    // agrees with a stable sort by key. See [`super::devsort`].
    let (sorted_key, sorted_permutation) = scratch
        .sort
        .sort(device, "schwarz.galerkin_sort", scratch.key.handle(), entries)?;
    // The two arrays the rest of this function names are the SORTED ones, so
    // the network's own spans are copied back over them rather than every
    // reader being retargeted.
    device.copy(scratch.key.handle(), 0, sorted_key, 0,
                entries * std::mem::size_of::<u32>())?;
    device.copy(scratch.permutation.handle(), 0, sorted_permutation, 0,
                entries * std::mem::size_of::<u32>())?;

    // Each equal-key run's first slot, then the scan that numbers the runs.
    // The scan writes the run count into the slot past the flags, which is the
    // total the coarse entry count is read from.
    scratch
        .edge
        .size(device, entries + 1, AllocLabel("schwarz.gal_edge"))?;
    let flags = SchwarzGalerkinEdgeFlagArgs {
        key: scratch.key.handle(),
        edge: scratch.edge.handle(),
        count: entries as u32,
        seam_arena_count: 0,
    };
    device.launch("schwarz.galerkin_edge_flag", &flags, entries as u32)?;
        // The multi-level exclusive scan rather than a single pass at EXTENT
        // ONE, which would be one device thread walking the array in
        // sequence; see [`super::scan`].
        scan_scratch.exclusive(device, "schwarz.galerkin_edge_scan", scratch.edge.handle(), entries as u32)?;
    let coarse_entries = scratch.edge.read_one(device, entries)? as usize;

    dst.blocks = coarse_entries as u32;
    dst.column
        .size(device, coarse_entries.max(1), AllocLabel("schwarz.gal_col"))?;
    dst.value
        .size(device, 9 * coarse_entries.max(1), AllocLabel("schwarz.gal_val"))?;
    scratch
        .edge_start
        .size(device, coarse_entries + 1, AllocLabel("schwarz.gal_start"))?;

    let heads = SchwarzGalerkinEdgeHeadArgs {
        key: scratch.key.handle(),
        edge: scratch.edge.handle(),
        column: dst.column.handle(),
        edge_start: scratch.edge_start.handle(),
        row_count: dst.offset.handle(),
        coarse_rows,
        count: entries as u32,
        seam_arena_count: 0,
    };
    device.launch("schwarz.galerkin_edge_head", &heads, entries as u32)?;
    // The last run's END. Every other run ends where the next one starts, so
    // this one slot is written on its own, a single value copied from the
    // host.
    scratch
        .edge_start
        .write(device, coarse_entries, &[entries as u32])?;

        // The multi-level exclusive scan rather than a single pass at EXTENT
        // ONE, which would be one device thread walking the array in
        // sequence; see [`super::scan`].
        scan_scratch.exclusive(device, "schwarz.galerkin_row_scan", dst.offset.handle(), coarse_rows)?;

    let sums = SchwarzGalerkinSegmentSumArgs {
        edge_start: scratch.edge_start.handle(),
        permutation: scratch.permutation.handle(),
        source_value: src.value.handle(),
        value: dst.value.handle(),
        count: coarse_entries as u32,
        seam_arena_count: 0,
    };
    device.launch(
        "schwarz.galerkin_segment_sum",
        &sums,
        coarse_entries as u32,
    )?;
    Ok(())
}

/// One coarse level: its domains, its factorization and its map from the fine.
#[derive(Default)]
pub struct Level {
    /// Nodes at this level.
    pub nodes: u32,
    /// Domains this level's own aggregation produced.
    pub domains: u32,
    /// The level's operator, `C A C^T` of the level below.
    pub matrix: CoarseMat,
    /// Per node: its domain at this level, which the level ABOVE coarsens with.
    pub aggregate: Vec<u32>,
    aggregate_device: Buffer<u32>,
    aggregate_offset: ReadbackBuffer<u32>,
    members: Buffer<u32>,
    inverse_offset: ReadbackBuffer<u32>,
    cursor: Buffer<u32>,
    dense_offset: Buffer<u32>,
    dense: Buffer<f32>,
    work: Buffer<f32>,
    floor: Buffer<f32>,
    packed: Buffer<f32>,
    /// Per FINE vertex: which node of this level owns it.
    map_fine: Buffer<u32>,
    /// The restricted residual and the smoothed correction, three per node.
    residual: Buffer<f32>,
    correction: Buffer<f32>,
    residual_local: Buffer<f32>,
    y_local: Buffer<f32>,
}

/// Aggregate a coarse level's own graph, then build and factor its domains.
///
/// # THE AGGREGATION IS THE HEAVY-EDGE ONE AT EVERY LEVEL
///
/// [`hem_partition`] aggregates each coarse level's own graph, not just the
/// fine one, in place of a connectivity-blind independent-set sweep.
///
/// Which aggregation is used is a convergence-rate question and not a
/// correctness one: any aggregation yields a valid SPD preconditioner, because
/// each block is the inverse of an SPD principal submatrix applied as a Gram
/// form. The heavy-edge partition is the better coarsener of the two, since it
/// keeps the stiffest couplings inside each dense block, which is what makes
/// the coarsening effective.
///
/// The edge weight is the block's Frobenius norm, which is what the fine
/// graph [`partition`] builds already weighs a coupling by.
///
/// # Safety
/// `level.matrix` must be a populated `CoarseMat`.
pub unsafe fn build_level<D: Device>(
    device: &mut D,
    level: &mut Level,
    vertices: u32,
    scan_scratch: &mut super::scan::ScanScratch,
) -> Result<(), Fault> {
    let nodes = level.matrix.rows;
    level.nodes = nodes;
    if nodes == 0 {
        level.domains = 0;
        return Ok(());
    }
    // The level's own graph, read off its matrix: one edge per stored block,
    // weighted by the block's Frobenius norm.
    level.matrix.offset.download(device)?;
    let offset = level.matrix.offset.host().to_vec();
    let blocks = level.matrix.blocks as usize;
    let mut column = vec![0u32; blocks.max(1)];
    let mut value = vec![0.0f32; 9 * blocks.max(1)];
    if blocks > 0 {
        level.matrix.column.read(device, 0, &mut column)?;
        level.matrix.value.read(device, 0, &mut value)?;
    }
    let weight: Vec<f32> = (0..blocks)
        .map(|slot| {
            (0..9usize)
                .map(|e| value[9 * slot + e] * value[9 * slot + e])
                .sum::<f32>()
                .sqrt()
        })
        .collect();

    level.aggregate = vec![0u32; nodes as usize];
    let domains = hem_partition(
        &offset,
        &column[..blocks],
        &weight[..blocks],
        nodes as usize,
        MAX_MEMBERS,
        &mut level.aggregate,
    );
    level.domains = domains;

    level.aggregate_device.size(device, nodes as usize, AllocLabel("schwarz.lagg"))?;
    level.aggregate_device.write(device, 0, &level.aggregate)?;
    level.aggregate_offset.size(device, domains as usize + 1, AllocLabel("schwarz.laoff"))?;
    level.members.size(device, nodes as usize, AllocLabel("schwarz.lmem"))?;
    level.inverse_offset.size(device, domains as usize + 1, AllocLabel("schwarz.lioff"))?;
    level.cursor.size(device, domains as usize, AllocLabel("schwarz.lcur"))?;
    build_domains(
        device,
        level.aggregate_device.handle(),
        nodes,
        domains,
        level.aggregate_offset.handle(),
        level.members.handle(),
        level.inverse_offset.handle(),
        level.cursor.handle(),
        scan_scratch,
    )?;
    level.inverse_offset.download(device)?;
    let packed_total =
        *level.inverse_offset.host().last().expect("offsets are domains + 1") as usize;

    let max_dimension = 3 * MAX_MEMBERS;
    let cells = (max_dimension * max_dimension) as usize;
    let offsets: Vec<u32> = (0..=domains).map(|g| g * cells as u32).collect();
    level.dense_offset.size(device, offsets.len(), AllocLabel("schwarz.ldoff"))?;
    level.dense_offset.write(device, 0, &offsets)?;
    level.dense.size(device, domains as usize * cells, AllocLabel("schwarz.ldense"))?;
    level.work.size(device, domains as usize * cells, AllocLabel("schwarz.lwork"))?;
    level.floor.size(device, domains as usize, AllocLabel("schwarz.lfloor"))?;
    level.packed.size(device, packed_total.max(1), AllocLabel("schwarz.lpacked"))?;
    level.map_fine.size(device, vertices as usize, AllocLabel("schwarz.lmap"))?;
    level.residual.size(device, 3 * nodes as usize, AllocLabel("schwarz.lrc"))?;
    level.correction.size(device, 3 * nodes as usize, AllocLabel("schwarz.lec"))?;
    let local = domains as usize * max_dimension as usize;
    level.residual_local.size(device, local, AllocLabel("schwarz.lrl"))?;
    level.y_local.size(device, local, AllocLabel("schwarz.lyl"))?;

    // The gather is the coarse one; every phase after it is the fine
    // factorization's, unchanged, because they read `dense` and know nothing
    // about where it came from.
    let zero = VecFillU32Args {
        array: level.dense.handle(),
        value: 0,
        count: (domains as usize * cells) as u32,
        seam_arena_count: 0,
    };
    device.launch("schwarz.level_zero", &zero, (domains as usize * cells) as u32)?;
    let gather = SchwarzCoarseGatherArgs {
        aggregate_offset: level.aggregate_offset.handle(),
        members: level.members.handle(),
        dense_offset: level.dense_offset.handle(),
        matrix_offset: level.matrix.offset.handle(),
        matrix_column: level.matrix.column.handle(),
        matrix_value: level.matrix.value.handle(),
        dense: level.dense.handle(),
        stride: MAX_MEMBERS,
        count: domains * MAX_MEMBERS,
        seam_arena_count: 0,
    };
    device.launch("schwarz.coarse_gather", &gather, domains * MAX_MEMBERS)?;
    factor_from_dense(
        device,
        domains,
        max_dimension,
        level.aggregate_offset.handle(),
        level.dense_offset.handle(),
        level.inverse_offset.handle(),
        level.dense.handle(),
        level.work.handle(),
        level.floor.handle(),
        level.packed.handle(),
    )
}

/// One coarse level's contribution, as [`encode_apply`] takes the fine one.
#[derive(Clone, Copy)]
pub struct LevelSweep {
    pub nodes: u32,
    pub map_fine: Handle,
    pub sweep: Sweep,
    pub residual: Handle,
    pub correction: Handle,
}

impl Level {
    /// Retire the slot, KEEPING every allocation it names.
    ///
    /// Zeroing the two counts is all a retirement is, with no free around it:
    /// a level is rebuilt in place every solve, and a `Buffer` has no `Drop` at
    /// all, so a dropped slot is a stranded span rather than a freed one.
    ///
    /// The counts are what a consumer reads, and [`encode_whole_sweep`] skips
    /// a level carrying zero of either, so a retired slot is inert even before
    /// [`State::live_levels`] bounds the walk.
    fn retire(&mut self) {
        self.nodes = 0;
        self.domains = 0;
    }

    /// This level's contribution, for the multilevel apply.
    pub fn sweep(&mut self) -> LevelSweep {
        LevelSweep {
            nodes: self.nodes,
            map_fine: self.map_fine.handle(),
            sweep: Sweep {
                aggregates: self.domains,
                max_dimension: 3 * MAX_MEMBERS,
                aggregate_offset: self.aggregate_offset.handle(),
                members: self.members.handle(),
                inverse_offset: self.inverse_offset.handle(),
                packed: self.packed.handle(),
                residual_local: self.residual_local.handle(),
                y_local: self.y_local.handle(),
            },
            residual: self.residual.handle(),
            correction: self.correction.handle(),
        }
    }
}

/// The additive level's weight, which scales every coarse correction.
///
/// Nothing reads the environment for it, so every level's correction enters at
/// full weight. It is the knob that keeps a sum of several levels' corrections
/// from over-correcting.
pub const COARSE_WEIGHT: f32 = 1.0;

/// Build the coarse hierarchy above the fine level.
///
/// Level 0 is the fine operator materialized; each level above is the Galerkin
/// coarsening of the one below over that level's own aggregation, with its
/// domains built and factored, and its map from the fine composed through the
/// level below it.
///
/// IT STOPS FOR TWO REASONS, both of which are the coarsening having nothing
/// left to do: a level small enough to be ONE domain is already
/// an exact-ish local solve, and a level whose aggregation produced no fewer
/// nodes than it has has stalled.
///
/// # THE SLOTS ARE RETIRED AND REUSED, NEVER DROPPED
///
/// This runs once per NEWTON STEP, and a [`Level`] names EIGHTEEN device
/// allocations. A `Buffer` has no `Drop` (a drop cannot reach the device that
/// owns the span) and production frees nothing, so a dropped level STRANDS
/// every span it held: emptying the vector here would leak a whole hierarchy
/// per Newton step.
///
/// The slots are therefore kept for the run and indexed in place.
/// [`Buffer::size`] reallocates only when the held capacity is short, so a slot
/// reused this way costs nothing after the build that first sized it.
///
/// WHAT RETIRING MEANS, and why the depth is a field rather than the vector's
/// length: a rebuild can reach FEWER levels than the one before it, and the
/// slots past it still hold their allocations and their old counts. Every slot
/// is retired to zero here, this build fills the ones it reaches, and
/// [`State::live_levels`] bounds both consumers at the depth reached, so a
/// shorter hierarchy cannot apply a stale deeper level.
///
/// # Safety
/// [`partition`] must have run, and `rows` must name the operator's arrays.
pub unsafe fn build_hierarchy<D: Device>(
    device: &mut D,
    state: &mut State,
    rows: OperatorRows,
    vertices: u32,
    levels: u32,
) -> Result<(), Fault> {
    // RETIRED, NOT CLEARED: the allocations stay and the counts go to zero, so
    // a build that stops short leaves nothing a consumer would act on.
    for level in state.levels.iter_mut() {
        level.retire();
    }
    state.live_depth = 0;
    if levels <= 1 || state.aggregates == 0 || vertices == 0 {
        return Ok(());
    }
    materialize_level0(device, &mut state.level0, rows, vertices, &mut state.scan)?;

    for depth in 1..levels {
        // One slot per coarse level, grown once and indexed in place after
        // that. The push can only ever extend by one: the depths are walked in
        // order, so a slot exists for every level below this one.
        let slot = depth as usize - 1;
        if state.levels.len() == slot {
            state.levels.push(Level::default());
        }
        if depth == 1 {
            // The first coarsening reads level 0 and aggregates it with the
            // FINE partition, which is what makes level 1's nodes the fine
            // domains. THE AGGREGATION IS TAKEN AS A DEVICE HANDLE, the one
            // `partition` uploaded; the host copy beside it is what the level's
            // map from a fine vertex is written from, further down.
            let aggregate = state.assignment.handle();
            galerkin(
                device,
                &mut state.galerkin,
                &mut state.level0,
                aggregate,
                state.aggregates,
                &mut state.levels[slot].matrix,
                &mut state.scan,
            )?;
        } else {
            // The level below is the SOURCE, and both are slots of one vector,
            // so the split is what hands out the two borrows at once.
            let (below, here) = state.levels.split_at_mut(slot);
            let previous = &mut below[slot - 1];
            let nodes = previous.domains;
            let aggregate = previous.aggregate_device.handle();
            galerkin(
                device,
                &mut state.galerkin,
                &mut previous.matrix,
                aggregate,
                nodes,
                &mut here[0].matrix,
                &mut state.scan,
            )?;
        }
        build_level(device, &mut state.levels[slot], vertices, &mut state.scan)?;
        if state.levels[slot].domains == 0 {
            break;
        }
        // The map from a fine vertex to this level's node.
        if depth == 1 {
            let map: Vec<u32> = state.assignment_host.clone();
            state.levels[slot].map_fine.write(device, 0, &map)?;
        } else {
            let (below, here) = state.levels.split_at_mut(slot);
            let previous = &below[slot - 1];
            compose_map(
                device,
                previous.map_fine.handle(),
                previous.aggregate_device.handle(),
                here[0].map_fine.handle(),
                vertices,
            )?;
        }
        let nodes = state.levels[slot].nodes;
        let domains = state.levels[slot].domains;
        // THE DEPTH MOVES ONLY ONCE THE LEVEL IS WHOLE, after its map is
        // composed, so a build that fails part way through leaves the slot
        // retired rather than half live.
        state.live_depth = slot + 1;
        // A level that is one domain is already an exact-ish local solve, and a
        // level whose aggregation produced no fewer nodes has stalled.
        if nodes <= MAX_MEMBERS || domains >= nodes {
            break;
        }
    }
    Ok(())
}

/// Push a whole additive sweep, fine level and every coarse level, onto a
/// caller's encoder.
///
/// The fine sweep, then per level restrict the fine residual onto it, smooth it
/// with that level's own domains, and prolong the correction back scaled by the
/// additive weight. The PCG batches an
/// iteration's work into one submission, so the whole preconditioner has to be
/// expressible as encoder operations or the batching breaks around it.
///
/// THE LEVELS ARE ADDITIVE, which is what the name says: each is a separate
/// correction and they are summed, so the order between them changes nothing
/// but the float rounding.
///
/// THE COARSE SPACE IS PIECEWISE CONSTANT, so the restriction is a plain sum
/// and the prolongation is its transpose by construction rather than by two
/// definitions agreeing.
///
/// # Safety
/// As [`encode_apply`], and every level's buffers must be the ones
/// [`build_level`] sized.
pub unsafe fn encode_whole_sweep(
    encoder: &mut dyn Encoder,
    state: &mut State,
    x: Handle,
    result: Handle,
    vertices: u32,
) -> Result<(), Fault> {
    let fine = state.sweep();
    encode_apply(encoder, fine, x, result)?;
    // THE LIVE PREFIX, never the whole vector: a slot past the depth this
    // state's last build reached keeps its allocations for the next one and
    // holds an operator belonging to an earlier structure.
    for level in state.live_levels() {
        if level.domains == 0 || level.nodes == 0 {
            continue;
        }
        let sweep = level.sweep();
        // The level's restriction target opens at zero, the restriction being
        // an accumulation.
        let zero = VecFillU32Args {
            array: sweep.residual,
            value: 0,
            count: 3 * sweep.nodes,
            seam_arena_count: 0,
        };
        encoder.elements(&zero, 3 * sweep.nodes)?;
        let restrict = SchwarzRestrictRowArgs {
            map_fine: sweep.map_fine,
            x,
            coarse: sweep.residual,
            count: vertices,
            seam_arena_count: 0,
        };
        encoder.elements(&restrict, vertices)?;
        encode_apply(encoder, sweep.sweep, sweep.residual, sweep.correction)?;
        let prolong = SchwarzProlongRowArgs {
            map_fine: sweep.map_fine,
            coarse: sweep.correction,
            z: result,
            weight: COARSE_WEIGHT,
            count: vertices,
            seam_arena_count: 0,
        };
        encoder.elements(&prolong, vertices)?;
    }
    Ok(())
}

/// The sentinel for a vertex no aggregate has claimed.
///
/// A SENTINEL, not an index: it is compared against and never used to address
/// anything.
const UNSET: u32 = 0xffff_ffff;

/// The heavy-edge connectivity partition, and the DEFAULT aggregation.
///
/// HOST CODE, and that is the one place in this preconditioner where a whole
/// pass runs on the CPU: the graph is read down and the partition is decided
/// there, because a sequential greedy sweep fits no dispatch shape. Only the
/// per-vertex aggregate assignment goes back up.
///
/// The shape: order the vertices by how heavily they are coupled to anything but
/// themselves, then walk that order taking each unclaimed vertex as a seed and
/// growing its aggregate up to `kmax` by repeatedly absorbing whichever
/// unclaimed neighbor has accumulated the largest edge weight into it. Isolated
/// leftovers become singletons.
///
/// WHY IT KEEPS THE STIFFEST COUPLINGS INSIDE A BLOCK, which is the reason it is
/// the default rather than a variation: the aggregate is the domain the factor
/// inverts exactly, so an edge inside one is handled exactly and an edge across
/// one is not handled at all at this level. Sorting by incident weight puts the
/// stiff couplings inside.
///
/// # Ties
///
/// Ties in the sort key are broken by VERTEX INDEX, which makes the
/// partition reproducible; a run-to-run difference in it would otherwise show
/// up as a difference in iteration counts with nothing naming its cause. Any
/// tie-break would be admissible, because the aggregation affects the
/// convergence rate and never correctness: every block is the inverse of an SPD
/// principal submatrix whatever the partition.
///
/// # Panics
///
/// Never. A column index at or past `rows` is skipped, which is a real guard
/// rather than a formality: the fine graph is built from a contact pattern that
/// can name a vertex outside the block range.
pub fn hem_partition(
    offset: &[u32],
    column: &[u32],
    weight: &[f32],
    rows: usize,
    kmax: u32,
    aggregate: &mut [u32],
) -> u32 {
    assert_eq!(offset.len(), rows + 1, "the graph offsets are rows + 1 long");
    assert_eq!(column.len(), weight.len(), "every edge carries one weight");
    assert_eq!(aggregate.len(), rows, "one aggregate slot per vertex");

    // The incident weight, EXCLUDING the diagonal: a vertex's coupling to
    // itself says nothing about who it should share a block with. Accumulated
    // in double, which is a host sum and so outside the float32 rule.
    let mut incident = vec![0.0f64; rows];
    for row in 0..rows {
        for slot in offset[row] as usize..offset[row + 1] as usize {
            if column[slot] as usize != row {
                incident[row] += f64::from(weight[slot]);
            }
        }
    }
    let mut order: Vec<u32> = (0..rows as u32).collect();
    order.sort_by(|a, b| {
        incident[*b as usize]
            .partial_cmp(&incident[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });

    aggregate.fill(UNSET);
    let mut gain = vec![0.0f32; rows];
    let mut touched: Vec<u32> = Vec::with_capacity(8 * kmax as usize);
    let mut next = 0u32;

    for &seed in &order {
        if aggregate[seed as usize] != UNSET {
            continue;
        }
        aggregate[seed as usize] = next;
        let mut size = 1u32;
        touched.clear();
        absorb(seed, offset, column, weight, rows, aggregate, &mut gain, &mut touched);
        while size < kmax {
            // The best unclaimed neighbor by accumulated gain. A gain of
            // exactly zero never wins, because `best` opens at 0.0 and the
            // comparison is strict: a vertex reachable only by a zero-weight
            // edge is not worth pulling in.
            let mut best = 0.0f32;
            let mut chosen = UNSET;
            for &vertex in &touched {
                if aggregate[vertex as usize] == UNSET && gain[vertex as usize] > best {
                    best = gain[vertex as usize];
                    chosen = vertex;
                }
            }
            if chosen == UNSET {
                break;
            }
            aggregate[chosen as usize] = next;
            size += 1;
            absorb(chosen, offset, column, weight, rows, aggregate, &mut gain, &mut touched);
        }
        for &vertex in &touched {
            gain[vertex as usize] = 0.0;
        }
        next += 1;
    }
    // A vertex with no edges was never touched by any seed's absorb, so it is
    // still unclaimed here and gets an aggregate of its own.
    for slot in aggregate.iter_mut() {
        if *slot == UNSET {
            *slot = next;
            next += 1;
        }
    }
    next
}

/// Add `vertex`'s edges into the gains of its unclaimed neighbors.
///
/// `touched` records who has a nonzero gain so the sweep above can both scan
/// them and reset them, which is what keeps the cost proportional to the
/// aggregate's neighborhood rather than to the whole graph.
#[allow(clippy::too_many_arguments)]
fn absorb(
    vertex: u32,
    offset: &[u32],
    column: &[u32],
    weight: &[f32],
    rows: usize,
    aggregate: &[u32],
    gain: &mut [f32],
    touched: &mut Vec<u32>,
) {
    for slot in offset[vertex as usize] as usize..offset[vertex as usize + 1] as usize {
        let neighbor = column[slot];
        if neighbor == vertex || neighbor as usize >= rows || aggregate[neighbor as usize] != UNSET
        {
            continue;
        }
        if gain[neighbor as usize] == 0.0 {
            touched.push(neighbor);
        }
        gain[neighbor as usize] += weight[slot];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ppf_cts_compute::{AllocLabel, Buffer, ReadbackBuffer};

    /// The domains a known assignment produces, checked as a PARTITION rather
    /// than element by element.
    ///
    /// THE MEMBER ORDER WITHIN A DOMAIN IS NOT CHECKED, because the scatter
    /// claims its slot with an atomic, so the order is whatever the claims
    /// interleave. What must hold is that every vertex
    /// appears exactly once, in its own aggregate's span, which is the property
    /// the factor and the apply actually rely on.
    #[test]
    fn build_domains_partitions_every_vertex_into_its_aggregate() {
        let mut scan_scratch = super::super::scan::ScanScratch::default();
        let mut device = super::super::launch::host_device();
        // Six vertices over three aggregates, deliberately uneven and
        // deliberately not in order: aggregate 1 takes three, aggregate 0 two,
        // aggregate 2 one.
        let assignment: [u32; 6] = [0, 1, 2, 1, 0, 1];
        let vertices = assignment.len() as u32;
        let aggregates = 3u32;

        let mut aggregate = Buffer::<u32>::default();
        let mut offset = ReadbackBuffer::<u32>::default();
        let mut members = ReadbackBuffer::<u32>::default();
        let mut inverse_offset = ReadbackBuffer::<u32>::default();
        let mut cursor = Buffer::<u32>::default();
        aggregate
            .size(&mut device, vertices as usize, AllocLabel("test.agg"))
            .unwrap();
        aggregate.write(&mut device, 0, &assignment).unwrap();
        offset
            .size(&mut device, aggregates as usize + 1, AllocLabel("test.off"))
            .unwrap();
        members
            .size(&mut device, vertices as usize, AllocLabel("test.mem"))
            .unwrap();
        inverse_offset
            .size(&mut device, aggregates as usize + 1, AllocLabel("test.inv"))
            .unwrap();
        cursor
            .size(&mut device, aggregates as usize, AllocLabel("test.cur"))
            .unwrap();

        // Safety: every buffer is sized above exactly as the contract asks.
        unsafe {
            build_domains(
                &mut device,
                aggregate.handle(),
                vertices,
                aggregates,
                offset.handle(),
                members.handle(),
                inverse_offset.handle(),
                cursor.handle(),
            
            &mut scan_scratch,
        )
        }
        .unwrap();
        offset.download(&mut device).unwrap();
        members.download(&mut device).unwrap();
        inverse_offset.download(&mut device).unwrap();

        // Aggregate 0 has vertices 0 and 4, aggregate 1 has 1, 3 and 5,
        // aggregate 2 has 2. The starts are the running sum of those counts.
        assert_eq!(offset.host(), &[0, 2, 5, 6]);

        let members = members.host();
        for (group, expected) in [
            (0usize, vec![0u32, 4]),
            (1, vec![1, 3, 5]),
            (2, vec![2]),
        ] {
            let span = offset.host()[group] as usize..offset.host()[group + 1] as usize;
            let mut got = members[span].to_vec();
            got.sort_unstable();
            assert_eq!(got, expected, "aggregate {group}");
        }

        // The packed lower triangle of a `3 * m` square, per domain: m = 2 gives
        // d = 6 and 21 floats, m = 3 gives d = 9 and 45, m = 1 gives d = 3 and 6.
        // The last slot is the total, which is what the caller sizes with.
        assert_eq!(inverse_offset.host(), &[0, 21, 66, 72]);
    }

    /// No aggregates is not an error and must touch nothing.
    #[test]
    fn build_domains_with_no_aggregates_is_a_no_op() {
        let mut scan_scratch = super::super::scan::ScanScratch::default();
        let mut device = super::super::launch::host_device();
        let handle = Handle::NONE;
        // Safety: the early return happens before any handle is resolved, which
        // is the only reason `Handle::NONE` is admissible here and nowhere else.
        unsafe { build_domains(&mut device, handle, 0, 0, handle, handle, handle, handle, &mut scan_scratch) }
            .unwrap();
    }
}

#[cfg(test)]
mod partition_tests {
    use super::*;

    /// A symmetric both-triangle graph from an edge list, which is the shape
    /// the fine graph pass in [`partition`] produces.
    fn graph(rows: usize, edges: &[(u32, u32, f32)]) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let mut adjacency: Vec<Vec<(u32, f32)>> = vec![Vec::new(); rows];
        for &(a, b, w) in edges {
            adjacency[a as usize].push((b, w));
            adjacency[b as usize].push((a, w));
        }
        let mut offset = Vec::with_capacity(rows + 1);
        let mut column = Vec::new();
        let mut weight = Vec::new();
        offset.push(0u32);
        for row in adjacency {
            for (c, w) in row {
                column.push(c);
                weight.push(w);
            }
            offset.push(column.len() as u32);
        }
        (offset, column, weight)
    }

    /// Every vertex lands in exactly one aggregate and no aggregate is empty.
    /// This is the invariant the factor and the apply both depend on, and it is
    /// checked before any claim about WHICH aggregate.
    #[test]
    fn every_vertex_is_claimed_exactly_once() {
        let (offset, column, weight) = graph(
            8,
            &[
                (0, 1, 5.0), (1, 2, 4.0), (2, 3, 3.0), (3, 0, 2.0),
                (4, 5, 5.0), (5, 6, 1.0), (6, 7, 5.0),
            ],
        );
        let mut aggregate = vec![0u32; 8];
        let count = hem_partition(&offset, &column, &weight, 8, 4, &mut aggregate);
        assert!(count > 0 && count as usize <= 8, "{count}");
        assert!(aggregate.iter().all(|&g| g < count), "{aggregate:?}");
        let mut seen = vec![0usize; count as usize];
        for &g in &aggregate {
            seen[g as usize] += 1;
        }
        assert!(seen.iter().all(|&n| n > 0), "an aggregate is empty: {seen:?}");
        assert_eq!(seen.iter().sum::<usize>(), 8);
    }

    /// No aggregate exceeds `kmax`, which is what bounds the dense block the
    /// factor allocates and inverts. A violation is not a quality problem: the
    /// factor sizes its scratch from `kmax`.
    #[test]
    fn no_aggregate_exceeds_kmax() {
        // A dense clique of six, where an unbounded greedy sweep would happily
        // take all of them.
        let mut edges = Vec::new();
        for a in 0..6u32 {
            for b in (a + 1)..6u32 {
                edges.push((a, b, 1.0 + f32::from(a as u16)));
            }
        }
        let (offset, column, weight) = graph(6, &edges);
        for kmax in [1u32, 2, 3, 5] {
            let mut aggregate = vec![0u32; 6];
            let count = hem_partition(&offset, &column, &weight, 6, kmax, &mut aggregate);
            let mut seen = vec![0u32; count as usize];
            for &g in &aggregate {
                seen[g as usize] += 1;
            }
            assert!(
                seen.iter().all(|&n| n <= kmax),
                "kmax={kmax} produced {seen:?}"
            );
        }
    }

    /// THE STIFFEST EDGE ENDS UP INSIDE AN AGGREGATE, which is the whole reason
    /// this partition is the default rather than the connectivity-blind one: an
    /// edge inside a domain is inverted exactly and an edge across one is not
    /// handled at this level at all.
    #[test]
    fn the_stiffest_edge_lands_inside_one_aggregate() {
        // Four vertices in a line, with the middle edge far stiffer than the
        // two ends. With kmax = 2 the partition has to choose, and the middle
        // pair is the one worth keeping.
        let (offset, column, weight) =
            graph(4, &[(0, 1, 1.0), (1, 2, 100.0), (2, 3, 1.0)]);
        let mut aggregate = vec![0u32; 4];
        hem_partition(&offset, &column, &weight, 4, 2, &mut aggregate);
        assert_eq!(
            aggregate[1], aggregate[2],
            "the stiff pair was split: {aggregate:?}"
        );
    }

    /// A vertex with no edges is its own aggregate rather than being dropped.
    #[test]
    fn isolated_vertices_become_singletons() {
        let (offset, column, weight) = graph(5, &[(0, 1, 1.0)]);
        let mut aggregate = vec![0u32; 5];
        let count = hem_partition(&offset, &column, &weight, 5, 4, &mut aggregate);
        let mut seen = vec![0usize; count as usize];
        for &g in &aggregate {
            seen[g as usize] += 1;
        }
        assert!(seen.iter().all(|&n| n > 0));
        assert_eq!(seen.iter().sum::<usize>(), 5);
        // 2, 3 and 4 touch nothing, so each is alone.
        for isolated in [2usize, 3, 4] {
            assert_eq!(seen[aggregate[isolated] as usize], 1, "{aggregate:?}");
        }
    }

    /// The partition is REPRODUCIBLE even when every sort key ties. Two runs
    /// over the same graph give the same answer, so a change in iteration
    /// counts is never the partition moving underneath the measurement.
    #[test]
    fn the_partition_is_reproducible() {
        // Every incident weight equal, so every sort key ties.
        let mut edges = Vec::new();
        for a in 0..10u32 {
            edges.push((a, (a + 1) % 10, 1.0));
        }
        let (offset, column, weight) = graph(10, &edges);
        let mut first = vec![0u32; 10];
        let mut second = vec![0u32; 10];
        let a = hem_partition(&offset, &column, &weight, 10, 3, &mut first);
        let b = hem_partition(&offset, &column, &weight, 10, 3, &mut second);
        assert_eq!(a, b);
        assert_eq!(first, second);
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use ppf_cts_compute::{AllocLabel, Buffer, ReadbackBuffer};

    /// The fine graph over an operator whose only entries are in the FIXED
    /// matrix, which is the smallest shape that exercises the count, the scan
    /// and the fill together.
    ///
    /// Three block rows, with row 0 coupled to 1, row 1 coupled to 0 and 2, and
    /// row 2 coupled to 1. Every block is a scaled identity, so its Frobenius
    /// norm is `sqrt(3) * scale` and the weights are checkable by hand.
    #[test]
    fn the_fine_graph_carries_one_weighted_edge_per_operator_entry() {
        let mut scan_scratch = super::super::scan::ScanScratch::default();
        let mut device = super::super::launch::host_device();
        let rows_count = 3u32;
        let fixed_offset: [u32; 4] = [0, 1, 3, 4];
        let fixed_index: [u32; 4] = [1, 0, 2, 1];
        let scales: [f32; 4] = [2.0, 2.0, 5.0, 5.0];
        let mut fixed_value = vec![0.0f32; 9 * 4];
        for (entry, scale) in scales.iter().enumerate() {
            for d in 0..3 {
                fixed_value[9 * entry + 4 * d] = *scale;
            }
        }
        // No dynamic entries and no transpose pairs: every offset array is flat
        // zero, which is a real state (a scene before any contact) rather than a
        // convenience.
        let empty_offsets = [0u32; 4];

        let mut buffers: Vec<Buffer<u32>> = Vec::new();
        let mut make_u32 = |data: &[u32], label: &'static str, device: &mut _| {
            let mut b = Buffer::<u32>::default();
            b.size(device, data.len().max(1), AllocLabel(label)).unwrap();
            if !data.is_empty() {
                b.write(device, 0, data).unwrap();
            }
            let h = b.handle();
            buffers.push(b);
            h
        };
        let fixed_offset_h = make_u32(&fixed_offset, "t.foff", &mut device);
        let fixed_index_h = make_u32(&fixed_index, "t.fidx", &mut device);
        let dyn_off_h = make_u32(&empty_offsets, "t.doff", &mut device);
        let ref_off_h = make_u32(&empty_offsets, "t.roff", &mut device);
        let tra_off_h = make_u32(&empty_offsets, "t.toff", &mut device);
        let empty_u32 = make_u32(&[0u32], "t.empty", &mut device);

        let mut fixed_value_buf = Buffer::<f32>::default();
        fixed_value_buf
            .size(&mut device, fixed_value.len(), AllocLabel("t.fval"))
            .unwrap();
        fixed_value_buf.write(&mut device, 0, &fixed_value).unwrap();
        let mut empty_f32 = Buffer::<f32>::default();
        empty_f32.size(&mut device, 1, AllocLabel("t.ef")).unwrap();
        let mut zero_diagonal = Buffer::<f32>::default();
        zero_diagonal
            .size(&mut device, 9 * rows_count as usize, AllocLabel("t.diag"))
            .unwrap();

        let mut graph_offset = ReadbackBuffer::<u32>::default();
        let mut column = ReadbackBuffer::<u32>::default();
        let mut weight = ReadbackBuffer::<f32>::default();
        graph_offset
            .size(&mut device, rows_count as usize + 1, AllocLabel("t.goff"))
            .unwrap();
        column.size(&mut device, 4, AllocLabel("t.col")).unwrap();
        weight.size(&mut device, 4, AllocLabel("t.wgt")).unwrap();

        let operator = OperatorRows {
            dynamic_index: empty_u32,
            dynamic_value: empty_f32.handle(),
            dynamic_offset: dyn_off_h,
            reference_index: empty_u32,
            reference_value: empty_u32,
            reference_offset: ref_off_h,
            global_value: empty_f32.handle(),
            fixed_index: fixed_index_h,
            fixed_offset: fixed_offset_h,
            fixed_value: fixed_value_buf.handle(),
            transpose_pair: empty_u32,
            transpose_offset: tra_off_h,
            diagonal: zero_diagonal.handle(),
        };
        // THE THREE ROWS `partition` ISSUES: count each row's four spans, scan
        // the counts into starts, then walk the spans again writing one column
        // and one Frobenius norm per entry.
        let count = SchwarzFineGraphCountArgs {
            dynamic_offset: operator.dynamic_offset,
            reference_offset: operator.reference_offset,
            fixed_offset: operator.fixed_offset,
            transpose_offset: operator.transpose_offset,
            count: graph_offset.handle(),
            count_of_rows: rows_count,
            seam_arena_count: 0,
        };
        let fill = SchwarzFineGraphFillArgs {
            dynamic_index: operator.dynamic_index,
            dynamic_value: operator.dynamic_value,
            dynamic_offset: operator.dynamic_offset,
            reference_index: operator.reference_index,
            reference_value: operator.reference_value,
            reference_offset: operator.reference_offset,
            global_value: operator.global_value,
            fixed_index: operator.fixed_index,
            fixed_offset: operator.fixed_offset,
            fixed_value: operator.fixed_value,
            transpose_pair: operator.transpose_pair,
            transpose_offset: operator.transpose_offset,
            graph_offset: graph_offset.handle(),
            column: column.handle(),
            weight: weight.handle(),
            count: rows_count,
            seam_arena_count: 0,
        };
        // Safety: every buffer above is sized as the contract asks.
        unsafe {
            device.launch("t.graph_count", &count, rows_count).unwrap();
            scan_scratch
                .exclusive(&mut device, "t.graph_scan", graph_offset.handle(), rows_count)
                .unwrap();
            device.launch("t.graph_fill", &fill, rows_count).unwrap();
        }
        graph_offset.download(&mut device).unwrap();
        column.download(&mut device).unwrap();
        weight.download(&mut device).unwrap();

        assert_eq!(graph_offset.host(), &[0, 1, 3, 4]);
        assert_eq!(column.host(), &[1, 0, 2, 1]);
        let root3 = 3.0f32.sqrt();
        for (slot, scale) in scales.iter().enumerate() {
            let got = weight.host()[slot];
            assert!(
                (got - root3 * scale).abs() <= 1e-5 * root3 * scale,
                "slot {slot}: {got} against {}",
                root3 * scale
            );
        }
    }

    /// An operator with no rows is not an error and dispatches nothing.
    #[test]
    fn an_empty_operator_builds_an_empty_graph() {
        let mut device = super::super::launch::host_device();
        let h = Handle::NONE;
        let operator = OperatorRows {
            dynamic_index: h, dynamic_value: h, dynamic_offset: h,
            reference_index: h, reference_value: h, reference_offset: h,
            global_value: h, fixed_index: h, fixed_offset: h, fixed_value: h,
            transpose_pair: h, transpose_offset: h, diagonal: h,
        };
        let mut state = State::default();
        // Safety: the early return happens before any handle is resolved.
        unsafe { partition(&mut device, &mut state, operator, 0) }.unwrap();
        assert_eq!(state.aggregates, 0, "an empty operator produced an aggregate");
    }
}

#[cfg(test)]
mod factor_tests {
    use super::*;
    use ppf_cts_compute::{AllocLabel, Buffer, ReadbackBuffer};

    /// Factor one domain and check the property the apply relies on.
    ///
    /// THE ASSERTION IS `G A G^T ~ I`, not a comparison against a reference
    /// factorization. `G = L^{-1}` where `A = L L^T`, so `G A G^T` is the
    /// identity exactly, and that is the statement the preconditioner needs:
    /// the apply is `z = G^T (G r)`, whose quadratic form is `||G r||^2 >= 0`,
    /// and it is a useful preconditioner only if `G^T G` really is `A^{-1}`.
    /// Comparing against a stored `L` would instead assert a particular
    /// factorization, which is a weaker and more brittle thing to check.
    ///
    /// The floor the factorization adds makes it `G (A + fI) G^T = I`, so the
    /// tolerance below is against `A + fI`.
    #[test]
    fn the_packed_factor_inverts_its_own_block() {
        let mut device = super::super::launch::host_device();
        // One aggregate of two vertices: a 6x6 block. The operator is the fixed
        // matrix only, with a diagonal block per vertex and a coupling between
        // them, chosen so the result is SPD and not a multiple of the identity.
        let aggregates = 1u32;
        let members_host: [u32; 2] = [0, 1];
        let max_members = 2u32;
        let max_dimension = 6u32;

        let diag0 = [4.0f32, 0.5, 0.0, 0.5, 5.0, 0.25, 0.0, 0.25, 6.0];
        let diag1 = [7.0f32, -0.5, 0.2, -0.5, 8.0, 0.0, 0.2, 0.0, 9.0];
        let couple = [0.5f32, 0.1, 0.0, 0.1, 0.4, 0.0, 0.0, 0.0, 0.3];
        // Row 0 holds its own block then the coupling to 1; row 1 holds the
        // coupling's transpose then its own block. Storing both triangles is
        // what the fine graph and the gather both expect.
        let fixed_offset: [u32; 3] = [0, 2, 4];
        let fixed_index: [u32; 4] = [0, 1, 0, 1];
        let mut transposed = [0.0f32; 9];
        for r in 0..3 {
            for c in 0..3 {
                transposed[3 * r + c] = couple[3 * c + r];
            }
        }
        let mut fixed_value = Vec::new();
        fixed_value.extend_from_slice(&diag0);
        fixed_value.extend_from_slice(&couple);
        fixed_value.extend_from_slice(&transposed);
        fixed_value.extend_from_slice(&diag1);

        let mut keep: Vec<Buffer<u32>> = Vec::new();
        let mut u32buf = |data: &[u32], label: &'static str, device: &mut _| {
            let mut b = Buffer::<u32>::default();
            b.size(device, data.len().max(1), AllocLabel(label)).unwrap();
            b.write(device, 0, data).unwrap();
            let h = b.handle();
            keep.push(b);
            h
        };
        let aggregate_offset = u32buf(&[0u32, 2], "t.aoff", &mut device);
        let members = u32buf(&members_host, "t.mem", &mut device);
        let dense_offset = u32buf(&[0u32, 36], "t.doff", &mut device);
        let inverse_offset = u32buf(&[0u32, 21], "t.ioff", &mut device);
        let fixed_offset_h = u32buf(&fixed_offset, "t.foff", &mut device);
        let fixed_index_h = u32buf(&fixed_index, "t.fidx", &mut device);
        let empty_off = u32buf(&[0u32, 0, 0], "t.eoff", &mut device);
        let empty = u32buf(&[0u32], "t.e", &mut device);

        let mut fixed_value_buf = Buffer::<f32>::default();
        fixed_value_buf
            .size(&mut device, fixed_value.len(), AllocLabel("t.fval"))
            .unwrap();
        fixed_value_buf.write(&mut device, 0, &fixed_value).unwrap();
        let mut empty_f = Buffer::<f32>::default();
        empty_f.size(&mut device, 1, AllocLabel("t.ef")).unwrap();
        let mut zero_diagonal = Buffer::<f32>::default();
        zero_diagonal.size(&mut device, 9 * 2, AllocLabel("t.diag")).unwrap();

        let mut dense = ReadbackBuffer::<f32>::default();
        let mut work = Buffer::<f32>::default();
        let mut floor = ReadbackBuffer::<f32>::default();
        let mut packed = ReadbackBuffer::<f32>::default();
        dense.size(&mut device, 36, AllocLabel("t.dense")).unwrap();
        work.size(&mut device, 36, AllocLabel("t.work")).unwrap();
        floor.size(&mut device, 1, AllocLabel("t.floor")).unwrap();
        packed.size(&mut device, 21, AllocLabel("t.packed")).unwrap();

        let operator = OperatorRows {
            dynamic_index: empty,
            dynamic_value: empty_f.handle(),
            dynamic_offset: empty_off,
            reference_index: empty,
            reference_value: empty,
            reference_offset: empty_off,
            global_value: empty_f.handle(),
            fixed_index: fixed_index_h,
            fixed_offset: fixed_offset_h,
            fixed_value: fixed_value_buf.handle(),
            transpose_pair: empty,
            transpose_offset: empty_off,
            // ZERO, so the operator is exactly the blocks written above and the
            // expected matrix below needs no extra term. A real scene's
            // diagonal is the dominant part of the operator; a fixture that
            // wants to check the gather reads it should say so explicitly.
            diagonal: zero_diagonal.handle(),
        };
        // Safety: every buffer is sized as the contract asks.
        unsafe {
            factor_domains(
                &mut device, operator, aggregates, max_members, max_dimension,
                aggregate_offset, members, dense_offset, inverse_offset,
                dense.handle(), work.handle(), floor.handle(), packed.handle(),
            )
        }
        .unwrap();
        floor.download(&mut device).unwrap();
        packed.download(&mut device).unwrap();

        // Rebuild A + fI on the host, from the same blocks the gather read.
        let f = floor.host()[0];
        assert!(f > 0.0, "the floor must be positive: {f}");
        let mut a = [[0.0f64; 6]; 6];
        for (block, (br, bc)) in [
            (&diag0, (0usize, 0usize)),
            (&couple, (0, 1)),
            (&transposed, (1, 0)),
            (&diag1, (1, 1)),
        ] {
            for r in 0..3 {
                for c in 0..3 {
                    a[3 * br + r][3 * bc + c] += f64::from(block[3 * r + c]);
                }
            }
        }
        for k in 0..6 {
            a[k][k] += f64::from(f);
        }
        // Unpack G from its lower triangle.
        let mut g = [[0.0f64; 6]; 6];
        for row in 0..6usize {
            for column in 0..=row {
                g[row][column] =
                    f64::from(packed.host()[row * (row + 1) / 2 + column]);
            }
        }
        // G A G^T, which must be the identity.
        let mut ga = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                ga[i][j] = (0..6).map(|k| g[i][k] * a[k][j]).sum();
            }
        }
        let mut worst = 0.0f64;
        for i in 0..6 {
            for j in 0..6 {
                let value: f64 = (0..6).map(|k| ga[i][k] * g[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((value - expected).abs());
            }
        }
        assert!(worst < 2e-5, "G A G^T is not the identity, worst {worst:e}");

        // ---- and now the APPLY over the same factor, which is what the PCG
        // calls. `z = G^T (G r)` must satisfy `(A + fI) z = r`, and the two
        // properties below are the ones the solve depends on.
        let r_host: [f32; 6] = [1.0, -2.0, 0.5, 3.0, 0.25, -1.5];
        let mut x = Buffer::<f32>::default();
        let mut residual_local = Buffer::<f32>::default();
        let mut y_local = Buffer::<f32>::default();
        let mut z = ReadbackBuffer::<f32>::default();
        x.size(&mut device, 6, AllocLabel("t.x")).unwrap();
        x.write(&mut device, 0, &r_host).unwrap();
        residual_local.size(&mut device, 6, AllocLabel("t.rl")).unwrap();
        y_local.size(&mut device, 6, AllocLabel("t.yl")).unwrap();
        z.size(&mut device, 6, AllocLabel("t.z")).unwrap();
        let sweep = Sweep {
            aggregates,
            max_dimension,
            aggregate_offset,
            members,
            inverse_offset,
            packed: packed.handle(),
            residual_local: residual_local.handle(),
            y_local: y_local.handle(),
        };
        let x_handle = x.handle();
        let z_handle = z.handle();
        // Safety: every buffer is sized as the contract asks.
        device
            .run("t.apply", |encoder| unsafe {
                encode_apply(encoder, sweep, x_handle, z_handle)
            })
            .unwrap();
        z.download(&mut device).unwrap();

        // (1) IT SOLVES: (A + fI) z must reproduce r.
        let mut worst_solve = 0.0f64;
        for i in 0..6 {
            let got: f64 = (0..6).map(|k| a[i][k] * f64::from(z.host()[k])).sum();
            worst_solve = worst_solve.max((got - f64::from(r_host[i])).abs());
        }
        assert!(
            worst_solve < 2e-4,
            "(A + fI) z did not reproduce r, worst {worst_solve:e}"
        );

        // (2) IT IS POSITIVE: r . z = ||G r||^2 >= 0, which is the property that
        // keeps an aggregate term from producing the rz <= 0 breakdown. Checked
        // rather than assumed, because it is the reason the apply is two
        // triangular passes instead of one symmetric matvec.
        let dot: f64 = (0..6)
            .map(|k| f64::from(r_host[k]) * f64::from(z.host()[k]))
            .sum();
        assert!(dot > 0.0, "r . z must be positive, got {dot:e}");
    }
}

// ---------------------------------------------------------------------------
// The preconditioner's state and its build.

use ppf_cts_compute::{AllocLabel, Buffer, ReadbackBuffer};

/// Everything one Schwarz preconditioner owns.
///
/// REBUILT WHEN THE STRUCTURE CHANGES, not every step. The factorization is
/// quadratic in the dense-block cap and the partition is a sequential host
/// sweep, so paying either per Newton step would cost more than the
/// preconditioner saves.
#[derive(Default)]
pub struct State {
    /// The exclusive scan's per-level block sums; see [`super::scan`].
    ///
    /// ONE FOR THE WHOLE HIERARCHY, because the scans below are sequential in
    /// time and none reads another's scratch. Sized for the widest array any of
    /// them walks, which is the vertex count.
    pub scan: super::scan::ScanScratch,
    /// The vertex count this was built for; a change means a rebuild.
    pub rows: u32,
    pub aggregates: u32,
    pub max_dimension: u32,
    /// The weighted adjacency the partition sorts over.
    graph_offset: ReadbackBuffer<u32>,
    graph_column: ReadbackBuffer<u32>,
    graph_weight: ReadbackBuffer<f32>,
    /// Per vertex: its aggregate.
    assignment: Buffer<u32>,
    /// The domains.
    aggregate_offset: ReadbackBuffer<u32>,
    members: Buffer<u32>,
    inverse_offset: ReadbackBuffer<u32>,
    cursor: Buffer<u32>,
    /// The factorization: the dense block and the working G, back to back in
    /// one span per aggregate, then the packed lower triangle of G.
    dense_offset: Buffer<u32>,
    dense: Buffer<f32>,
    work: Buffer<f32>,
    floor: Buffer<f32>,
    packed: Buffer<f32>,
    /// The apply's per-domain scratch.
    residual_local: Buffer<f32>,
    y_local: Buffer<f32>,
    /// The fine aggregation, kept on the host: the first coarsening reads it
    /// and every level's map composes through it.
    pub assignment_host: Vec<u32>,
    /// The multilevel hierarchy above the fine level, empty at one level.
    ///
    /// THE SLOTS OUTLIVE A REBUILD, so the vector's length is a CAPACITY and
    /// `live_depth` is the count that means anything. It is private for that
    /// reason: a consumer reading the whole vector would apply a level a
    /// shorter rebuild retired. [`State::live_levels`] is the read.
    levels: Vec<Level>,
    /// How many of `levels` the last [`build_hierarchy`] filled.
    ///
    /// It counts the COARSE levels only, the fine level not being one of
    /// them, and every consumer walks it rather than the vector's own extent.
    live_depth: usize,
    /// Level 0 materialized, the source the first coarsening reads.
    level0: CoarseMat,
    /// The coarsening's scratch, held for the run rather than per call.
    galerkin: GalerkinScratch,
    /// A zero offset array of `rows + 1`, for a scene with NO dynamic matrix.
    ///
    /// AN ABSENT SPAN IS AN OFFSET ARRAY OF ZEROS, NOT A ONE-ELEMENT STAND-IN.
    /// A kernel walking `offset[row] .. offset[row + 1]` reads BOTH ends for
    /// every row, so an array shorter than `rows + 1` is read out of bounds at
    /// the first row and every row after. Handing it some other live buffer is
    /// worse than a short one: the bytes parse as enormous offsets and the walk
    /// reads values that are not values, which arrives as a NaN in the solve
    /// with nothing naming its cause.
    pub empty_offset: Buffer<u32>,
}

impl State {
    /// Size and zero the stand-in offset array, and hand back its handle.
    ///
    /// CALLED BEFORE THE OPERATOR VIEW IS TAKEN, because the view names this
    /// array for every span the scene does not have and a handle cannot be read
    /// out of a value that is being mutably borrowed to size it.
    pub fn empty_offsets<D: Device>(
        &mut self,
        device: &mut D,
        vertices: u32,
    ) -> Result<Handle, Fault> {
        if self.empty_offset.len() != vertices as usize + 1 {
            self.empty_offset
                .size(device, vertices as usize + 1, AllocLabel("schwarz.empty"))?;
            self.empty_offset
                .write(device, 0, &vec![0u32; vertices as usize + 1])?;
        }
        Ok(self.empty_offset.handle())
    }

    /// The coarse levels the last [`build_hierarchy`] built, and ONLY those.
    ///
    /// THE BOUND IS THE POINT. A rebuild can produce fewer levels than the one
    /// before it, and the slots past it keep their allocations and their old
    /// counts so the next build can reuse them; walking the whole vector would
    /// restrict onto a level whose operator belongs to an earlier structure.
    pub fn live_levels(&mut self) -> &mut [Level] {
        let live = self.live_depth;
        &mut self.levels[..live]
    }

    /// The sweep this state describes, for [`encode_apply`].
    pub fn sweep(&mut self) -> Sweep {
        Sweep {
            aggregates: self.aggregates,
            max_dimension: self.max_dimension,
            aggregate_offset: self.aggregate_offset.handle(),
            members: self.members.handle(),
            inverse_offset: self.inverse_offset.handle(),
            packed: self.packed.handle(),
            residual_local: self.residual_local.handle(),
            y_local: self.y_local.handle(),
        }
    }
}

/// The dense-block cap: vertices per aggregate.
///
/// SIXTEEN, AND THE SAME SIXTEEN ON EVERY BACKEND. A cap tied to a
/// threadgroup's shared memory would have to shrink when two `d x d` arrays no
/// longer fit; that does not apply here, the dense arrays being device
/// allocations this host sizes rather than a threadgroup budget.
pub const MAX_MEMBERS: u32 = 16;

/// Build the preconditioner: the graph, the partition, the domains, the factor.
///
/// THE PARTITION IS A HOST SWEEP AND SO IS THE READBACK IT NEEDS.
/// [`hem_partition`] is sequential and greedy, so the fine graph comes down to
/// run it. That is the one readback in this path, and it happens on a rebuild
/// rather than per step.
///
/// THE GRAPH IS NOT DEDUPLICATED AND NOT SYMMETRIZED, and that is deliberate
/// rather than an oversight. A vertex pair coupled through both the
/// dynamic and the fixed matrix appears twice, and the partition ADDS the two
/// weights when it accumulates a gain, which is the intended reading: two
/// couplings are stiffer than one. Deduplicating would change the partition.
///
/// # Safety
/// Every handle in `rows` must name a live allocation of the operator's arrays,
/// and `vertices` must be the operator's row count.
pub unsafe fn partition<D: Device>(
    device: &mut D,
    state: &mut State,
    rows: OperatorRows,
    vertices: u32,
) -> Result<(), Fault> {
    state.rows = vertices;
    if vertices == 0 {
        state.aggregates = 0;
        return Ok(());
    }
    // The graph's edge count is not known until the count pass has run, so the
    // offsets are sized first and the columns after, which is the same two-step
    // the dynamic matrix's flatten does.
    // The scan's scratch, sized once for the widest array the whole hierarchy
    // scans. Every level below the fine one is smaller, so the vertex count
    // covers all of them and no level allocates.
    state.scan.size(device, vertices)?;
    state.graph_offset.size(device, vertices as usize + 1, AllocLabel("schwarz.goff"))?;
    let count_only = SchwarzFineGraphCountArgs {
        dynamic_offset: rows.dynamic_offset,
        reference_offset: rows.reference_offset,
        fixed_offset: rows.fixed_offset,
        transpose_offset: rows.transpose_offset,
        count: state.graph_offset.handle(),
        count_of_rows: vertices,
        seam_arena_count: 0,
    };
    device.launch("schwarz.graph_count", &count_only, vertices)?;
        // The multi-level exclusive scan rather than a single pass at EXTENT
        // ONE, which would be one device thread walking the array in
        // sequence; see [`super::scan`].
    {
        let graph = state.graph_offset.handle();
        state
            .scan
            .exclusive(device, "schwarz.graph_scan", graph, vertices)?;
    }
    state.graph_offset.download(device)?;
    // THE OFFSETS ARE COPIED OUT BEFORE THE FILL, not read after it. The fill
    // dispatch NAMES this buffer, which stales its host mirror, and the seam
    // refuses a read of a stale mirror rather than answering out of the
    // contents from before the dispatch. That guard caught this exact mistake.
    let graph_offset_host: Vec<u32> = state.graph_offset.host().to_vec();
    let edges = *graph_offset_host.last().expect("offsets are rows + 1") as usize;

    state.graph_column.size(device, edges.max(1), AllocLabel("schwarz.gcol"))?;
    state.graph_weight.size(device, edges.max(1), AllocLabel("schwarz.gwgt"))?;
    let fill = SchwarzFineGraphFillArgs {
        dynamic_index: rows.dynamic_index,
        dynamic_value: rows.dynamic_value,
        dynamic_offset: rows.dynamic_offset,
        reference_index: rows.reference_index,
        reference_value: rows.reference_value,
        reference_offset: rows.reference_offset,
        global_value: rows.global_value,
        fixed_index: rows.fixed_index,
        fixed_offset: rows.fixed_offset,
        fixed_value: rows.fixed_value,
        transpose_pair: rows.transpose_pair,
        transpose_offset: rows.transpose_offset,
        graph_offset: state.graph_offset.handle(),
        column: state.graph_column.handle(),
        weight: state.graph_weight.handle(),
        count: vertices,
        seam_arena_count: 0,
    };
    device.launch("schwarz.graph_fill", &fill, vertices)?;
    state.graph_column.download(device)?;
    state.graph_weight.download(device)?;

    let mut assignment = vec![0u32; vertices as usize];
    let aggregates = hem_partition(
        &graph_offset_host,
        &state.graph_column.host()[..edges],
        &state.graph_weight.host()[..edges],
        vertices as usize,
        MAX_MEMBERS,
        &mut assignment,
    );
    state.aggregates = aggregates;
    state.max_dimension = 3 * MAX_MEMBERS;
    // THE PARTITION ANNOUNCES ITSELF, and it is the anti-silent-fallback line.
    // Schwarz sizes its footprint before allocating and DEGRADES rather than
    // aborting, so `precond = schwarz` can run as block-Jacobi with nothing to
    // read; check for this line before crediting an iteration count to the
    // preconditioner that was asked for. A run that prints this partitioned;
    // one that does not, did not.
    ::log::info!(
        "schwarz partitioned {vertices} vertices into {aggregates} aggregates"
    );

    state.assignment.size(device, vertices as usize, AllocLabel("schwarz.agg"))?;
    state.assignment.write(device, 0, &assignment)?;
    state.assignment_host = assignment;
    state.aggregate_offset.size(device, aggregates as usize + 1, AllocLabel("schwarz.aoff"))?;
    state.members.size(device, vertices as usize, AllocLabel("schwarz.mem"))?;
    state.inverse_offset.size(device, aggregates as usize + 1, AllocLabel("schwarz.ioff"))?;
    state.cursor.size(device, aggregates as usize, AllocLabel("schwarz.cur"))?;
    build_domains(
        device,
        state.assignment.handle(),
        vertices,
        aggregates,
        state.aggregate_offset.handle(),
        state.members.handle(),
        state.inverse_offset.handle(),
        state.cursor.handle(),
        &mut state.scan,
    )?;
    state.inverse_offset.download(device)?;
    let packed_total =
        *state.inverse_offset.host().last().expect("offsets are aggregates + 1") as usize;

    // ONE SPAN PER AGGREGATE AT A FIXED STRIDE, which is what lets the six
    // factorization rows recover a group from a flat index by division.
    // Packing the spans tightly instead would save the tails, and only a
    // per-launch shared-memory budget would make that worth it; here the
    // stride is the cap and the tail of a smaller domain is never read.
    let cells = (state.max_dimension * state.max_dimension) as usize;
    let offsets: Vec<u32> = (0..=aggregates).map(|g| g * cells as u32).collect();
    state.dense_offset.size(device, offsets.len(), AllocLabel("schwarz.doff"))?;
    state.dense_offset.write(device, 0, &offsets)?;
    state.dense.size(device, aggregates as usize * cells, AllocLabel("schwarz.dense"))?;
    state.work.size(device, aggregates as usize * cells, AllocLabel("schwarz.work"))?;
    state.floor.size(device, aggregates as usize, AllocLabel("schwarz.floor"))?;
    state.packed.size(device, packed_total.max(1), AllocLabel("schwarz.packed"))?;
    let local = aggregates as usize * state.max_dimension as usize;
    state.residual_local.size(device, local, AllocLabel("schwarz.rl"))?;
    state.y_local.size(device, local, AllocLabel("schwarz.yl"))?;

    Ok(())
}

/// Refactor every domain against the operator as it stands NOW.
///
/// SEPARATE FROM THE PARTITION BECAUSE THEY RUN ON DIFFERENT SCHEDULES, and
/// getting that wrong is not a small error. The partition depends only on the
/// operator's STRUCTURE, so it is reused while that structure holds. The
/// factorization depends on its VALUES, which change at every Newton step: the
/// diagonal carries the step's own inertia and elastic terms, and the contact
/// blocks change as pairs come and go, so the factorization runs
/// unconditionally on every build.
///
/// Measured on a pinned sheet: reusing one factorization across the step's
/// Newton iterations took the mean PCG count from 51 to 119 against
/// block-Jacobi's 51, hitting the iteration cap. A stale factorization is a
/// WORSE preconditioner than the block diagonal, not merely a weaker one.
///
/// # Safety
/// [`partition`] must have run for this operator's row count, and every handle
/// in `rows` must name a live allocation.
pub unsafe fn refactor<D: Device>(
    device: &mut D,
    state: &mut State,
    rows: OperatorRows,
) -> Result<(), Fault> {
    if state.aggregates == 0 {
        return Ok(());
    }
    factor_domains(
        device,
        rows,
        state.aggregates,
        MAX_MEMBERS,
        state.max_dimension,
        state.aggregate_offset.handle(),
        state.members.handle(),
        state.dense_offset.handle(),
        state.inverse_offset.handle(),
        state.dense.handle(),
        state.work.handle(),
        state.floor.handle(),
        state.packed.handle(),
    )?;
    Ok(())
}

#[cfg(test)]
mod build_tests {
    use super::*;

    /// Build the preconditioner over a real operator and check the property the
    /// PCG needs of it.
    ///
    /// THE OPERATOR IS A CHAIN, so the partition has a real choice to make: 12
    /// vertices coupled to their neighbors, which at a cap of 16 becomes one
    /// aggregate, and at a smaller cap several. Every block is diagonally
    /// dominant so the whole matrix is SPD, which is what the factorization
    /// assumes and what the Newton operator is.
    ///
    /// The assertion is that `M^{-1}` is SPD ON THE ACTUAL OPERATOR:
    /// `r . M^{-1} r > 0` for a spread of residuals, and the result is finite.
    /// That is exactly what keeps the PCG's `rz <= 0` guard from tripping, and
    /// it is a stronger statement than any single solve, since a preconditioner
    /// that inverted one domain wrongly would still solve its own block.
    #[test]
    fn the_built_preconditioner_is_spd_on_the_operator() {
        let mut device = super::super::launch::host_device();
        let vertices = 12u32;
        // A chain: row i couples to i-1 and i+1, both triangles stored.
        let mut offset = vec![0u32];
        let mut index = Vec::new();
        let mut value: Vec<f32> = Vec::new();
        for row in 0..vertices {
            let mut neighbours = vec![row];
            if row > 0 {
                neighbours.push(row - 1);
            }
            if row + 1 < vertices {
                neighbours.push(row + 1);
            }
            neighbours.sort_unstable();
            for n in &neighbours {
                index.push(*n);
                // The diagonal is heavy and the couplings light, which is what
                // makes the whole thing SPD.
                let scale = if *n == row { 10.0 } else { -1.0 };
                for r in 0..3 {
                    for c in 0..3 {
                        value.push(if r == c { scale } else { 0.0 });
                    }
                }
            }
            offset.push(index.len() as u32);
        }
        let mut keep: Vec<Buffer<u32>> = Vec::new();
        let mut u32buf = |data: &[u32], label: &'static str, device: &mut _| {
            let mut b = Buffer::<u32>::default();
            b.size(device, data.len().max(1), AllocLabel(label)).unwrap();
            b.write(device, 0, data).unwrap();
            let h = b.handle();
            keep.push(b);
            h
        };
        let fixed_offset = u32buf(&offset, "t.fo", &mut device);
        let fixed_index = u32buf(&index, "t.fi", &mut device);
        let empty_off = u32buf(&vec![0u32; vertices as usize + 1], "t.eo", &mut device);
        let empty = u32buf(&[0u32], "t.e", &mut device);
        let mut fixed_value = Buffer::<f32>::default();
        fixed_value.size(&mut device, value.len(), AllocLabel("t.fv")).unwrap();
        fixed_value.write(&mut device, 0, &value).unwrap();
        let mut empty_f = Buffer::<f32>::default();
        empty_f.size(&mut device, 1, AllocLabel("t.ef")).unwrap();
        let mut zero_diagonal = Buffer::<f32>::default();
        zero_diagonal
            .size(&mut device, 9 * vertices as usize, AllocLabel("t.diag"))
            .unwrap();

        let rows = OperatorRows {
            dynamic_index: empty,
            dynamic_value: empty_f.handle(),
            dynamic_offset: empty_off,
            reference_index: empty,
            reference_value: empty,
            reference_offset: empty_off,
            global_value: empty_f.handle(),
            fixed_index,
            fixed_offset,
            fixed_value: fixed_value.handle(),
            transpose_pair: empty,
            transpose_offset: empty_off,
            // ZERO, so the operator is exactly the blocks written above and the
            // expected matrix below needs no extra term. A real scene's
            // diagonal is the dominant part of the operator; a fixture that
            // wants to check the gather reads it should say so explicitly.
            diagonal: zero_diagonal.handle(),
        };
        let mut state = State::default();
        // Safety: every handle above names a live allocation of the right size.
        unsafe { partition(&mut device, &mut state, rows, vertices) }.unwrap();
        // Safety: the partition above sized everything this reads.
        unsafe { refactor(&mut device, &mut state, rows) }.unwrap();
        assert!(state.aggregates > 0, "the partition produced no aggregate");
        assert!(
            state.aggregates as usize * MAX_MEMBERS as usize >= vertices as usize,
            "{} aggregates cannot hold {vertices} vertices at a cap of {MAX_MEMBERS}",
            state.aggregates
        );

        let mut x = Buffer::<f32>::default();
        let mut z = ReadbackBuffer::<f32>::default();
        x.size(&mut device, 3 * vertices as usize, AllocLabel("t.x")).unwrap();
        z.size(&mut device, 3 * vertices as usize, AllocLabel("t.z")).unwrap();

        // Several residuals, including one that is zero everywhere but a single
        // component, which is the case a domain-local preconditioner is most
        // likely to mishandle.
        for trial in 0..4usize {
            let r: Vec<f32> = (0..3 * vertices as usize)
                .map(|k| match trial {
                    0 => 1.0,
                    1 => if k % 2 == 0 { 1.0 } else { -1.0 },
                    2 => (k as f32) * 0.1 - 1.0,
                    _ => if k == 7 { 1.0 } else { 0.0 },
                })
                .collect();
            x.write(&mut device, 0, &r).unwrap();
            let sweep = state.sweep();
            let x_handle = x.handle();
            let z_handle = z.handle();
            // Safety: the sweep's handles and both vectors are live and sized.
            device
                .run("t.apply", |encoder| unsafe {
                    encode_apply(encoder, sweep, x_handle, z_handle)
                })
                .unwrap();
            z.download(&mut device).unwrap();
            assert!(
                z.host().iter().all(|v| v.is_finite()),
                "trial {trial} produced a non-finite z"
            );
            let dot: f64 = r
                .iter()
                .zip(z.host())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            assert!(dot > 0.0, "trial {trial}: r . z must be positive, got {dot:e}");
        }
    }
}

#[cfg(test)]
mod gather_tests {
    use super::*;

    /// The gathered block, entry by entry, against the same operator built on
    /// the host, with EVERY span populated.
    ///
    /// THE OTHER TESTS HERE LEAVE THE TRANSPOSE AND REFERENCE SPANS EMPTY, and
    /// those are exactly the spans a live scene fills. Both matrices store ONE
    /// triangle: a row holds `(i, j)` for `j >= i` directly, and reaches
    /// `(j, i)` through a second span that names the stored slot and the source
    /// column, with the block TRANSPOSED at use. So a symmetric operator only
    /// comes out symmetric if the gather walks both, and only a comparison
    /// against an independently built operator says whether it does.
    ///
    /// Every block below is deliberately ASYMMETRIC, so a wrong orientation
    /// cannot pass: transposing one changes it.
    #[test]
    fn the_gathered_block_is_the_operator_restricted_to_the_domain() {
        let mut device = super::super::launch::host_device();
        let vertices = 3u32;
        // The fixed matrix's upper triangle, row by row:
        //   row 0: (0,0) (0,1) (0,2)  slots 0 1 2
        //   row 1: (1,1) (1,2)        slots 3 4
        //   row 2: (2,2)              slot  5
        let fixed_offset: [u32; 4] = [0, 3, 5, 6];
        let fixed_index: [u32; 6] = [0, 1, 2, 1, 2, 2];
        // Its lower triangle by reference: row 1 reaches (1,0) through slot 1,
        // row 2 reaches (2,0) through slot 2 and (2,1) through slot 4.
        let transpose_offset: [u32; 4] = [0, 0, 1, 3];
        let transpose_pair: [u32; 6] = [0, 1, 0, 2, 1, 4];
        let mut value = vec![0.0f32; 9 * 6];
        for slot in 0..6usize {
            for e in 0..9usize {
                value[9 * slot + e] = (slot * 9 + e) as f32 * 0.5 + 1.0;
            }
        }
        for slot in [0usize, 3, 5] {
            for d in 0..3 {
                value[9 * slot + 4 * d] += 100.0;
            }
        }
        // The dynamic matrix, stored the same way: row 0 holds (0,2) directly
        // and row 2 reaches (2,0) by reference to that slot.
        let dyn_offset: [u32; 4] = [0, 1, 1, 1];
        let dyn_index: [u32; 1] = [2];
        let ref_offset: [u32; 4] = [0, 0, 0, 1];
        let ref_index: [u32; 1] = [0];
        let ref_value: [u32; 1] = [0];
        let dyn_value: Vec<f32> = (0..9usize).map(|e| 2.0 + e as f32 * 0.25).collect();
        let diagonal: Vec<f32> =
            (0..9 * vertices as usize).map(|k| 1.0 + (k % 5) as f32).collect();

        // The operator, built on the host from the same five sources.
        let dim = 3 * vertices as usize;
        let mut expected = vec![vec![0.0f64; dim]; dim];
        for row in 0..vertices as usize {
            for slot in fixed_offset[row] as usize..fixed_offset[row + 1] as usize {
                let col = fixed_index[slot] as usize;
                for r in 0..3 {
                    for c in 0..3 {
                        expected[3 * row + r][3 * col + c] +=
                            f64::from(value[9 * slot + 3 * c + r]);
                    }
                }
            }
            for e in transpose_offset[row] as usize..transpose_offset[row + 1] as usize {
                let source = transpose_pair[2 * e] as usize;
                let slot = transpose_pair[2 * e + 1] as usize;
                for r in 0..3 {
                    for c in 0..3 {
                        expected[3 * row + r][3 * source + c] +=
                            f64::from(value[9 * slot + 3 * r + c]);
                    }
                }
            }
            for slot in dyn_offset[row] as usize..dyn_offset[row + 1] as usize {
                let col = dyn_index[slot] as usize;
                for r in 0..3 {
                    for c in 0..3 {
                        expected[3 * row + r][3 * col + c] +=
                            f64::from(dyn_value[9 * slot + 3 * c + r]);
                    }
                }
            }
            for e in ref_offset[row] as usize..ref_offset[row + 1] as usize {
                let col = ref_index[e] as usize;
                let slot = ref_value[e] as usize;
                for r in 0..3 {
                    for c in 0..3 {
                        expected[3 * row + r][3 * col + c] +=
                            f64::from(dyn_value[9 * slot + 3 * r + c]);
                    }
                }
            }
            for r in 0..3 {
                for c in 0..3 {
                    expected[3 * row + r][3 * row + c] +=
                        f64::from(diagonal[9 * row + 3 * c + r]);
                }
            }
        }

        let mut keep: Vec<Buffer<u32>> = Vec::new();
        let mut u32buf = |data: &[u32], label: &'static str, device: &mut _| {
            let mut b = Buffer::<u32>::default();
            b.size(device, data.len().max(1), AllocLabel(label)).unwrap();
            b.write(device, 0, data).unwrap();
            let h = b.handle();
            keep.push(b);
            h
        };
        let agg_off = u32buf(&[0u32, 3], "g.ao", &mut device);
        let members = u32buf(&[0u32, 1, 2], "g.m", &mut device);
        let dense_off = u32buf(&[0u32, 81], "g.do", &mut device);
        let fo = u32buf(&fixed_offset, "g.fo", &mut device);
        let fi = u32buf(&fixed_index, "g.fi", &mut device);
        let to = u32buf(&transpose_offset, "g.to", &mut device);
        let tp = u32buf(&transpose_pair, "g.tp", &mut device);
        let dyo = u32buf(&dyn_offset, "g.dyo", &mut device);
        let dyi = u32buf(&dyn_index, "g.dyi", &mut device);
        let rfo = u32buf(&ref_offset, "g.rfo", &mut device);
        let rfi = u32buf(&ref_index, "g.rfi", &mut device);
        let rfv = u32buf(&ref_value, "g.rfv", &mut device);

        let mut fv = Buffer::<f32>::default();
        fv.size(&mut device, value.len(), AllocLabel("g.fv")).unwrap();
        fv.write(&mut device, 0, &value).unwrap();
        let mut dv = Buffer::<f32>::default();
        dv.size(&mut device, dyn_value.len(), AllocLabel("g.dv")).unwrap();
        dv.write(&mut device, 0, &dyn_value).unwrap();
        let mut dg = Buffer::<f32>::default();
        dg.size(&mut device, diagonal.len(), AllocLabel("g.dg")).unwrap();
        dg.write(&mut device, 0, &diagonal).unwrap();
        let mut dense = ReadbackBuffer::<f32>::default();
        dense.size(&mut device, 81, AllocLabel("g.dense")).unwrap();

        // GATHER ONLY: `factor_domains` would go on to overwrite the block with
        // its Cholesky, and what is under test is what the gather produced.
        let zero = VecFillU32Args {
            array: dense.handle(),
            value: 0,
            count: 81,
            seam_arena_count: 0,
        };
        let gather = SchwarzFactorGatherArgs {
            aggregate_offset: agg_off,
            members,
            dense_offset: dense_off,
            dynamic_index: dyi,
            dynamic_value: dv.handle(),
            dynamic_offset: dyo,
            reference_index: rfi,
            reference_value: rfv,
            reference_offset: rfo,
            global_value: dv.handle(),
            fixed_index: fi,
            fixed_offset: fo,
            fixed_value: fv.handle(),
            transpose_pair: tp,
            transpose_offset: to,
            diagonal: dg.handle(),
            dense: dense.handle(),
            stride: 3,
            count: 3,
            seam_arena_count: 0,
        };
        // Safety: every buffer is sized as the contract asks.
        unsafe {
            device.launch("t.zero", &zero, 81).unwrap();
            device.launch("t.gather", &gather, 3).unwrap();
        }
        dense.download(&mut device).unwrap();

        let mut worst = 0.0f64;
        let mut worst_at = (0usize, 0usize);
        for i in 0..dim {
            for j in 0..dim {
                let got = f64::from(dense.host()[i * dim + j]);
                let want = expected[i][j];
                let e = (got - want).abs() / want.abs().max(1.0);
                if e > worst {
                    worst = e;
                    worst_at = (i, j);
                }
            }
        }
        let (i, j) = worst_at;
        assert!(
            worst < 1e-5,
            "the gathered block differs from the operator at ({i}, {j}): got {} want {} \
             (relative {worst:e})",
            dense.host()[i * dim + j],
            expected[i][j]
        );

        // ---- AND AGAINST THE OPERATOR THE PCG ACTUALLY APPLIES, which is the
        // check the comparison above cannot make. Both sides there read the
        // same handles, so a shared misreading of what those handles MEAN would
        // pass. `Operator::apply` is the definition of the operator, and for a
        // vector supported on the domain, `(A v)` restricted to the domain's
        // rows IS the domain block times the domain's slice.
        let mut xv = Buffer::<f32>::default();
        let mut av = ReadbackBuffer::<f32>::default();
        xv.size(&mut device, dim, AllocLabel("g.x")).unwrap();
        av.size(&mut device, dim, AllocLabel("g.av")).unwrap();
        let v: Vec<f32> = (0..dim).map(|k| 1.0 + (k % 4) as f32 * 0.5).collect();
        xv.write(&mut device, 0, &v).unwrap();
        let operator = super::super::operator::Operator {
            dynamic: Some(super::super::operator::DynamicView {
                offset: dyo,
                index: dyi,
                value: dv.handle(),
                transpose_offset: rfo,
                transpose_index: rfi,
                transpose_value: rfv,
            }),
            fixed: super::super::spmv::FixedCsrView {
                index: fi,
                offset: fo,
                value: fv.handle(),
                transpose_pair: tp,
                transpose_offset: to,
                rows: vertices,
            },
            diagonal: dg.handle(),
        };
        // The apply's magnitude sum, which this test does not read.
        let mut absolute: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut curvature: ppf_cts_compute::Buffer<f32> = Default::default();
        absolute.size(&mut device, operator.rows() as usize, ppf_cts_compute::AllocLabel("test.absolute")).unwrap();
        curvature.size(&mut device, operator.rows() as usize, ppf_cts_compute::AllocLabel("test.curvature")).unwrap();
        operator.apply(&mut device, xv.handle(), av.handle(), absolute.handle(), curvature.handle()).unwrap();
        av.download(&mut device).unwrap();
        let mut worst_op = 0.0f64;
        let mut worst_op_at = 0usize;
        for i in 0..dim {
            let block: f64 = (0..dim)
                .map(|j| f64::from(dense.host()[i * dim + j]) * f64::from(v[j]))
                .sum();
            let applied = f64::from(av.host()[i]);
            let e = (block - applied).abs() / applied.abs().max(1.0);
            if e > worst_op {
                worst_op = e;
                worst_op_at = i;
            }
        }
        assert!(
            worst_op < 1e-4,
            "the gathered block does not act like the operator at row {worst_op_at}: \
             block gives {} and Operator::apply gives {} (relative {worst_op:e})",
            (0..dim)
                .map(|j| f64::from(dense.host()[worst_op_at * dim + j]) * f64::from(v[j]))
                .sum::<f64>(),
            av.host()[worst_op_at]
        );
    }
}

#[cfg(test)]
mod transfer_tests {
    use super::*;

    /// Restriction and prolongation are TRANSPOSES of each other.
    ///
    /// THE GALERKIN COARSENING DEPENDS ON THIS AND CANNOT CHECK IT. `A_l = C A
    /// C^T` is symmetric only if the `C` the restriction applies really is the
    /// `C^T` the prolongation applies, and a coarse operator that is not
    /// symmetric is not a preconditioner: the PCG's `r . z` would stop being a
    /// quadratic form. The property is `<C x, y> == <x, C^T y>` for arbitrary
    /// vectors, which is what this asserts rather than checking either half
    /// against a hand-written expectation.
    #[test]
    fn restriction_and_prolongation_are_transposes() {
        let mut device = super::super::launch::host_device();
        // Seven fine vertices onto three coarse nodes, deliberately uneven so a
        // map that ignored multiplicity would show up.
        let map: [u32; 7] = [0, 2, 1, 0, 0, 2, 1];
        let vertices = map.len() as u32;
        let nodes = 3u32;
        let mut map_buf = Buffer::<u32>::default();
        map_buf.size(&mut device, map.len(), AllocLabel("x.map")).unwrap();
        map_buf.write(&mut device, 0, &map).unwrap();

        let x: Vec<f32> = (0..3 * vertices as usize)
            .map(|k| 1.0 + (k as f32) * 0.25)
            .collect();
        let y: Vec<f32> = (0..3 * nodes as usize)
            .map(|k| 2.0 - (k as f32) * 0.5)
            .collect();

        let mut xb = Buffer::<f32>::default();
        let mut cb = ReadbackBuffer::<f32>::default();
        xb.size(&mut device, x.len(), AllocLabel("x.x")).unwrap();
        xb.write(&mut device, 0, &x).unwrap();
        cb.size(&mut device, y.len(), AllocLabel("x.c")).unwrap();
        let map_handle = map_buf.handle();
        let x_handle = xb.handle();
        let coarse_handle = cb.handle();
        device
            .run("x.restrict", |encoder| {
                // The target opens at zero because the restriction accumulates
                // into it.
                let zero = VecFillU32Args {
                    array: coarse_handle,
                    value: 0,
                    count: 3 * nodes,
                    seam_arena_count: 0,
                };
                let args = SchwarzRestrictRowArgs {
                    map_fine: map_handle,
                    x: x_handle,
                    coarse: coarse_handle,
                    count: vertices,
                    seam_arena_count: 0,
                };
                // Safety: both buffers are sized as the contract asks.
                unsafe {
                    encoder.elements(&zero, 3 * nodes)?;
                    encoder.elements(&args, vertices)
                }
            })
            .unwrap();
        cb.download(&mut device).unwrap();
        // <C x, y>
        let left: f64 = cb
            .host()
            .iter()
            .zip(&y)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();

        // C^T y, by prolonging y into a zeroed destination at weight one.
        let mut yb = Buffer::<f32>::default();
        let mut zb = ReadbackBuffer::<f32>::default();
        yb.size(&mut device, y.len(), AllocLabel("x.y")).unwrap();
        yb.write(&mut device, 0, &y).unwrap();
        zb.size(&mut device, x.len(), AllocLabel("x.z")).unwrap();
        zb.seed(&mut device, &vec![0.0f32; x.len()]).unwrap();
        let y_handle = yb.handle();
        let z_handle = zb.handle();
        device
            .run("x.prolong", |encoder| {
                let args = SchwarzProlongRowArgs {
                    map_fine: map_handle,
                    coarse: y_handle,
                    z: z_handle,
                    weight: 1.0,
                    count: vertices,
                    seam_arena_count: 0,
                };
                // Safety: as above.
                unsafe { encoder.elements(&args, vertices) }
            })
            .unwrap();
        zb.download(&mut device).unwrap();
        // <x, C^T y>
        let right: f64 = zb
            .host()
            .iter()
            .zip(&x)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();

        assert!(
            (left - right).abs() <= 1e-5 * left.abs().max(1.0),
            "the transfers are not transposes: <C x, y> = {left} against \
             <x, C^T y> = {right}"
        );
        assert!(left.abs() > 1.0, "the fixture must not be trivially zero");
    }

    /// Composing a level's map from the one below it.
    #[test]
    fn the_level_map_composes_through_the_level_below() {
        let mut device = super::super::launch::host_device();
        // Six fine vertices onto four level-1 nodes, then onto two level-2.
        let previous_map: [u32; 6] = [0, 0, 1, 2, 3, 3];
        let previous_aggregate: [u32; 4] = [0, 0, 1, 1];
        let mut pm = Buffer::<u32>::default();
        let mut pa = Buffer::<u32>::default();
        let mut out = ReadbackBuffer::<u32>::default();
        pm.size(&mut device, 6, AllocLabel("x.pm")).unwrap();
        pm.write(&mut device, 0, &previous_map).unwrap();
        pa.size(&mut device, 4, AllocLabel("x.pa")).unwrap();
        pa.write(&mut device, 0, &previous_aggregate).unwrap();
        out.size(&mut device, 6, AllocLabel("x.out")).unwrap();
        // Safety: every buffer is sized as the contract asks.
        unsafe { compose_map(&mut device, pm.handle(), pa.handle(), out.handle(), 6) }
            .unwrap();
        out.download(&mut device).unwrap();
        assert_eq!(out.host(), &[0, 0, 0, 1, 1, 1]);
    }
}

#[cfg(test)]
mod level0_tests {
    use super::*;

    /// The materialized matrix ACTS like the operator.
    ///
    /// Not "has the entries I expect": that comparison reads the same handles
    /// twice and passed for a transposed gather until `Operator::apply` was
    /// brought in to settle it. This multiplies the materialized matrix by a
    /// vector on the host and diffs against the operator's own product, which
    /// is the definition rather than a second reading of its inputs.
    #[test]
    fn the_materialized_matrix_acts_like_the_operator() {
        let mut scan_scratch = super::super::scan::ScanScratch::default();
        let mut device = super::super::launch::host_device();
        let vertices = 4u32;
        // A chain with both triangles reachable: row i stores (i, i+1), and row
        // i+1 reaches (i+1, i) through the transpose table.
        let fixed_offset: [u32; 5] = [0, 1, 2, 3, 3];
        let fixed_index: [u32; 3] = [1, 2, 3];
        let transpose_offset: [u32; 5] = [0, 0, 1, 2, 3];
        let transpose_pair: [u32; 6] = [0, 0, 1, 1, 2, 2];
        let value: Vec<f32> = (0..9 * 3).map(|k| 1.0 + k as f32 * 0.3).collect();
        let diagonal: Vec<f32> = (0..9 * vertices as usize)
            .map(|k| if k % 4 == 0 { 50.0 } else { (k % 7) as f32 * 0.2 })
            .collect();

        let mut keep: Vec<Buffer<u32>> = Vec::new();
        let mut u32buf = |data: &[u32], label: &'static str, device: &mut _| {
            let mut b = Buffer::<u32>::default();
            b.size(device, data.len().max(1), AllocLabel(label)).unwrap();
            b.write(device, 0, data).unwrap();
            let h = b.handle();
            keep.push(b);
            h
        };
        let fo = u32buf(&fixed_offset, "l.fo", &mut device);
        let fi = u32buf(&fixed_index, "l.fi", &mut device);
        let to = u32buf(&transpose_offset, "l.to", &mut device);
        let tp = u32buf(&transpose_pair, "l.tp", &mut device);
        let eo = u32buf(&[0u32; 5], "l.eo", &mut device);
        let e = u32buf(&[0u32], "l.e", &mut device);
        let mut fv = Buffer::<f32>::default();
        fv.size(&mut device, value.len(), AllocLabel("l.fv")).unwrap();
        fv.write(&mut device, 0, &value).unwrap();
        let mut dg = Buffer::<f32>::default();
        dg.size(&mut device, diagonal.len(), AllocLabel("l.dg")).unwrap();
        dg.write(&mut device, 0, &diagonal).unwrap();
        let mut ef = Buffer::<f32>::default();
        ef.size(&mut device, 1, AllocLabel("l.ef")).unwrap();

        let rows = OperatorRows {
            dynamic_index: e,
            dynamic_value: ef.handle(),
            dynamic_offset: eo,
            reference_index: e,
            reference_value: e,
            reference_offset: eo,
            global_value: ef.handle(),
            fixed_index: fi,
            fixed_offset: fo,
            fixed_value: fv.handle(),
            transpose_pair: tp,
            transpose_offset: to,
            diagonal: dg.handle(),
        };
        let mut m0 = CoarseMat::default();
        // Safety: every handle names a live allocation of the right size.
        unsafe { materialize_level0(&mut device, &mut m0, rows, vertices, &mut scan_scratch) }.unwrap();
        // THE FILL NAMED THE OFFSETS, which stales their host mirror, so they
        // are read back again rather than out of the contents from before it.
        m0.offset.download(&mut device).unwrap();
        // Four rows: 1 + 0 + 1 = 2, 1 + 1 + 1 = 3, 1 + 1 + 1 = 3, 0 + 1 + 1 = 2.
        assert_eq!(m0.blocks, 10, "offsets {:?}", m0.offset.host());

        let dim = 3 * vertices as usize;
        let v: Vec<f32> = (0..dim).map(|k| 1.0 + (k % 5) as f32 * 0.4).collect();
        let mut column = vec![0u32; m0.blocks as usize];
        let mut blocks = vec![0.0f32; 9 * m0.blocks as usize];
        m0.column.read(&mut device, 0, &mut column).unwrap();
        m0.value.read(&mut device, 0, &mut blocks).unwrap();
        let offsets = m0.offset.host().to_vec();

        // The materialized product, on the host, column-major throughout.
        let mut product = vec![0.0f64; dim];
        for row in 0..vertices as usize {
            for slot in offsets[row] as usize..offsets[row + 1] as usize {
                let col = column[slot] as usize;
                for r in 0..3 {
                    for c in 0..3 {
                        product[3 * row + r] += f64::from(blocks[9 * slot + 3 * c + r])
                            * f64::from(v[3 * col + c]);
                    }
                }
            }
        }

        let mut xv = Buffer::<f32>::default();
        let mut av = ReadbackBuffer::<f32>::default();
        xv.size(&mut device, dim, AllocLabel("l.x")).unwrap();
        xv.write(&mut device, 0, &v).unwrap();
        av.size(&mut device, dim, AllocLabel("l.av")).unwrap();
        let operator = super::super::operator::Operator {
            dynamic: None,
            fixed: super::super::spmv::FixedCsrView {
                index: fi,
                offset: fo,
                value: fv.handle(),
                transpose_pair: tp,
                transpose_offset: to,
                rows: vertices,
            },
            diagonal: dg.handle(),
        };
        // The apply's magnitude sum, which this test does not read.
        let mut absolute: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut curvature: ppf_cts_compute::Buffer<f32> = Default::default();
        absolute.size(&mut device, operator.rows() as usize, ppf_cts_compute::AllocLabel("test.absolute")).unwrap();
        curvature.size(&mut device, operator.rows() as usize, ppf_cts_compute::AllocLabel("test.curvature")).unwrap();
        operator.apply(&mut device, xv.handle(), av.handle(), absolute.handle(), curvature.handle()).unwrap();
        av.download(&mut device).unwrap();
        let mut worst = 0.0f64;
        let mut worst_at = 0usize;
        for i in 0..dim {
            let e = (product[i] - f64::from(av.host()[i])).abs()
                / f64::from(av.host()[i]).abs().max(1.0);
            if e > worst {
                worst = e;
                worst_at = i;
            }
        }
        assert!(
            worst < 1e-5,
            "the materialized matrix does not act like the operator at row \
             {worst_at}: {} against {} (relative {worst:e})",
            product[worst_at],
            av.host()[worst_at]
        );
    }
}

#[cfg(test)]
mod galerkin_tests {
    use super::*;

    /// Multiply a `CoarseMat` by a vector, on the host.
    fn spmv(m: &CoarseMat, offset: &[u32], column: &[u32], value: &[f32], v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; 3 * m.rows as usize];
        for row in 0..m.rows as usize {
            for slot in offset[row] as usize..offset[row + 1] as usize {
                let col = column[slot] as usize;
                for r in 0..3 {
                    for c in 0..3 {
                        out[3 * row + r] +=
                            f64::from(value[9 * slot + 3 * c + r]) * v[3 * col + c];
                    }
                }
            }
        }
        out
    }

    /// THE GALERKIN IDENTITY: `<C A C^T y, y> == <A C^T y, C^T y>`.
    ///
    /// This is what makes the coarse operator the RIGHT one rather than merely
    /// a smaller one: the coarse quadratic form must equal the fine form
    /// evaluated on the prolonged vector. A coarsening that summed the wrong
    /// entries, or transposed a block, or dropped one, fails this while still
    /// producing a matrix of the right shape.
    ///
    /// It also gives the coarse operator's SPD-ness for free: if `A` is
    /// positive definite then so is `C A C^T` on any `y` whose prolongation is
    /// nonzero, and the preconditioner depends on that.
    #[test]
    fn the_coarse_operator_carries_the_fine_quadratic_form() {
        let mut scan_scratch = super::super::scan::ScanScratch::default();
        let mut device = super::super::launch::host_device();
        // A five-row source matrix, both triangles stored, deliberately
        // asymmetric per block so a transposed copy would show.
        let rows = 5u32;
        let src_offset: [u32; 6] = [0, 2, 4, 7, 9, 11];
        let src_column: [u32; 11] = [0, 1, 1, 0, 2, 1, 3, 3, 2, 4, 0];
        let src_value: Vec<f32> = (0..9 * 11).map(|k| 0.5 + k as f32 * 0.07).collect();

        let mut m = CoarseMat {
            rows,
            blocks: 11,
            ..CoarseMat::default()
        };
        m.offset.size(&mut device, 6, AllocLabel("g.off")).unwrap();
        m.offset.seed(&mut device, &src_offset).unwrap();
        m.column.size(&mut device, 11, AllocLabel("g.col")).unwrap();
        m.column.write(&mut device, 0, &src_column).unwrap();
        m.value.size(&mut device, 99, AllocLabel("g.val")).unwrap();
        m.value.write(&mut device, 0, &src_value).unwrap();

        // Five rows onto three coarse nodes, uneven.
        let aggregate: [u32; 5] = [0, 0, 1, 2, 1];
        let coarse_rows = 3u32;
        let mut aggregate_device: Buffer<u32> = Buffer::none();
        aggregate_device
            .size(&mut device, 5, AllocLabel("g.agg"))
            .unwrap();
        aggregate_device.write(&mut device, 0, &aggregate).unwrap();
        let mut coarse = CoarseMat::default();
        let mut scratch = GalerkinScratch::default();
        // Safety: the source is populated above and the aggregation covers it.
        unsafe {
            galerkin(
                &mut device,
                &mut scratch,
                &mut m,
                aggregate_device.handle(),
                coarse_rows,
                &mut coarse,
                &mut scan_scratch,
            )
        }
        .unwrap();
        assert!(coarse.blocks > 0, "the coarsening produced nothing");

        // y on the coarse space, and its prolongation C^T y on the fine.
        let y: Vec<f64> = (0..3 * coarse_rows as usize)
            .map(|k| 1.0 - (k as f64) * 0.3)
            .collect();
        let mut prolonged = vec![0.0f64; 3 * rows as usize];
        for row in 0..rows as usize {
            for k in 0..3 {
                prolonged[3 * row + k] = y[3 * aggregate[row] as usize + k];
            }
        }

        // <A C^T y, C^T y>, the fine form.
        let fine = spmv(&m, &src_offset, &src_column, &src_value, &prolonged);
        let fine_form: f64 = fine.iter().zip(&prolonged).map(|(a, b)| a * b).sum();

        // <C A C^T y, y>, the coarse form.
        coarse.offset.download(&mut device).unwrap();
        let c_offset = coarse.offset.host().to_vec();
        let mut c_column = vec![0u32; coarse.blocks as usize];
        let mut c_value = vec![0.0f32; 9 * coarse.blocks as usize];
        coarse.column.read(&mut device, 0, &mut c_column).unwrap();
        coarse.value.read(&mut device, 0, &mut c_value).unwrap();
        let coarse_product = spmv(&coarse, &c_offset, &c_column, &c_value, &y);
        let coarse_form: f64 = coarse_product.iter().zip(&y).map(|(a, b)| a * b).sum();

        assert!(
            (fine_form - coarse_form).abs() <= 1e-4 * fine_form.abs().max(1.0),
            "the Galerkin identity fails: fine form {fine_form} against coarse \
             form {coarse_form}"
        );
        assert!(
            fine_form.abs() > 1.0,
            "the fixture must not be trivially zero"
        );
    }

    /// The coarse CSR is SORTED BY COLUMN WITHIN A ROW AND CARRIES NO DUPLICATE.
    ///
    /// THAT PROPERTY IS THE SORT'S, and it is the one thing the Galerkin
    /// identity above cannot see: a matrix holding one coarse entry twice, with
    /// the two halves of its sum in different slots, evaluates the same
    /// quadratic form and is still wrong. Every consumer walks a row once, and
    /// `schwarz_find_local` resolves a column to ONE local slot, so a repeated
    /// column would accumulate one block and drop the other.
    ///
    /// The old host fold got this from a `BTreeMap`. It comes from the key sort
    /// now: `g * coarse_rows + h` is monotone in `(g, h)`, so a sorted run of
    /// equal keys is exactly one coarse entry and the runs come out in CSR
    /// order.
    ///
    /// THE SCRATCH IS REUSED, larger source first and smaller second, which is
    /// the shape a rebuild takes: a second call must not read the first call's
    /// tail. `Buffer::size` grows only past capacity, so the second call runs
    /// against an allocation wider than it needs.
    #[test]
    fn the_coarse_rows_are_sorted_and_deduplicated_across_a_reused_scratch() {
        let mut scan_scratch = super::super::scan::ScanScratch::default();
        let mut device = super::super::launch::host_device();
        let mut scratch = GalerkinScratch::default();

        // Deliberately DESCENDING columns within each source row and several
        // source rows per coarse row, so the coarsening has both an ordering
        // and a merge to do. Coarse row 0 takes source rows 0, 1 and 4.
        let cases: [(u32, Vec<u32>, Vec<u32>, Vec<u32>, u32); 2] = [
            (
                5,
                vec![0, 3, 6, 8, 10, 13],
                vec![4, 2, 0, 4, 3, 1, 3, 0, 2, 1, 4, 2, 0],
                vec![0, 0, 1, 2, 1],
                3,
            ),
            (3, vec![0, 2, 3, 5], vec![2, 0, 1, 2, 0], vec![0, 1, 0], 2),
        ];
        for (rows, offset, column, aggregate, coarse_rows) in cases {
            let blocks = column.len();
            let value: Vec<f32> = (0..9 * blocks).map(|k| 0.25 + k as f32 * 0.13).collect();
            let mut m = CoarseMat {
                rows,
                blocks: blocks as u32,
                ..CoarseMat::default()
            };
            m.offset
                .size(&mut device, offset.len(), AllocLabel("d.off"))
                .unwrap();
            m.offset.seed(&mut device, &offset).unwrap();
            m.column
                .size(&mut device, blocks, AllocLabel("d.col"))
                .unwrap();
            m.column.write(&mut device, 0, &column).unwrap();
            m.value
                .size(&mut device, 9 * blocks, AllocLabel("d.val"))
                .unwrap();
            m.value.write(&mut device, 0, &value).unwrap();

            let mut aggregate_device: Buffer<u32> = Buffer::none();
            aggregate_device
                .size(&mut device, rows as usize, AllocLabel("d.agg"))
                .unwrap();
            aggregate_device.write(&mut device, 0, &aggregate).unwrap();

            let mut coarse = CoarseMat::default();
            // Safety: the source is populated above and the aggregation covers
            // it, every entry below `coarse_rows`.
            unsafe {
                galerkin(
                    &mut device,
                    &mut scratch,
                    &mut m,
                    aggregate_device.handle(),
                    coarse_rows,
                    &mut coarse,
                    &mut scan_scratch,
                )
            }
            .unwrap();

            coarse.offset.download(&mut device).unwrap();
            let out_offset = coarse.offset.host().to_vec();
            let mut out_column = vec![0u32; coarse.blocks as usize];
            coarse.column.read(&mut device, 0, &mut out_column).unwrap();
            assert_eq!(
                out_offset.len(),
                coarse_rows as usize + 1,
                "the coarse offsets are one per coarse row plus the total"
            );
            assert_eq!(
                *out_offset.last().unwrap(),
                coarse.blocks,
                "the offsets\' last slot is the stored block count"
            );
            // What the coarsening SHOULD produce, taken straight off the
            // definition: the distinct coarse pairs the source entries land on.
            let mut expected: std::collections::BTreeSet<(u32, u32)> =
                std::collections::BTreeSet::new();
            for row in 0..rows as usize {
                for slot in offset[row] as usize..offset[row + 1] as usize {
                    expected.insert((aggregate[row], aggregate[column[slot] as usize]));
                }
            }
            assert_eq!(
                coarse.blocks as usize,
                expected.len(),
                "the coarsening produced {} entries against {} distinct coarse \
                 pairs, so a run was split or two were merged",
                coarse.blocks,
                expected.len()
            );
            let mut seen: Vec<(u32, u32)> = Vec::new();
            for coarse_row in 0..coarse_rows as usize {
                let span = out_offset[coarse_row] as usize..out_offset[coarse_row + 1] as usize;
                for pair in out_column[span.clone()].windows(2) {
                    assert!(
                        pair[0] < pair[1],
                        "coarse row {coarse_row} is not strictly ascending by \
                         column: {} then {}",
                        pair[0],
                        pair[1]
                    );
                }
                for &col in &out_column[span] {
                    seen.push((coarse_row as u32, col));
                }
            }
            assert_eq!(
                seen,
                expected.into_iter().collect::<Vec<_>>(),
                "the coarse entries are not the distinct coarse pairs in CSR order"
            );
        }
    }
}

#[cfg(test)]
mod hierarchy_tests {
    use super::*;

    /// The multilevel preconditioner is SPD on the operator it was built from.
    ///
    /// THE LEVELS ARE ADDITIVE, so the whole sweep is a sum of terms each of
    /// which is `C_l^T G_l^T G_l C_l`, and a sum of positive semidefinite terms
    /// with one positive definite among them is positive definite. That is what
    /// the PCG's `r . z > 0` guard needs, and what a coarsening with a
    /// transposed block or a lost entry would break while still producing
    /// correctly shaped matrices.
    ///
    /// It also asserts the hierarchy actually COARSENED, because a build that
    /// silently produced no levels would pass an SPD check on the fine sweep
    /// alone.
    #[test]
    fn the_multilevel_sweep_is_spd_and_actually_coarsens() {
        let mut device = super::super::launch::host_device();
        // Enough vertices that a cap of 16 needs several domains and those
        // domains need coarsening: a chain of 200.
        let vertices = 200u32;
        let mut offset = vec![0u32];
        let mut index = Vec::new();
        let mut value: Vec<f32> = Vec::new();
        for row in 0..vertices {
            let mut neighbours = vec![row];
            if row + 1 < vertices {
                neighbours.push(row + 1);
            }
            for n in &neighbours {
                index.push(*n);
                let scale = if *n == row { 20.0 } else { -1.0 };
                for r in 0..3 {
                    for c in 0..3 {
                        value.push(if r == c { scale } else { 0.0 });
                    }
                }
            }
            offset.push(index.len() as u32);
        }
        // The lower triangle by reference: row i reaches (i, i-1).
        let mut t_offset = vec![0u32];
        let mut t_pair: Vec<u32> = Vec::new();
        for row in 0..vertices as usize {
            if row > 0 {
                // (row - 1, row) is the second entry of row - 1.
                t_pair.push(row as u32 - 1);
                t_pair.push(offset[row - 1] + 1);
            }
            t_offset.push((t_pair.len() / 2) as u32);
        }

        let mut keep: Vec<Buffer<u32>> = Vec::new();
        let mut u32buf = |data: &[u32], label: &'static str, device: &mut _| {
            let mut b = Buffer::<u32>::default();
            b.size(device, data.len().max(1), AllocLabel(label)).unwrap();
            b.write(device, 0, data).unwrap();
            let h = b.handle();
            keep.push(b);
            h
        };
        let fo = u32buf(&offset, "h.fo", &mut device);
        let fi = u32buf(&index, "h.fi", &mut device);
        let to = u32buf(&t_offset, "h.to", &mut device);
        let tp = u32buf(&t_pair, "h.tp", &mut device);
        let eo = u32buf(&vec![0u32; vertices as usize + 1], "h.eo", &mut device);
        let e = u32buf(&[0u32], "h.e", &mut device);
        let mut fv = Buffer::<f32>::default();
        fv.size(&mut device, value.len(), AllocLabel("h.fv")).unwrap();
        fv.write(&mut device, 0, &value).unwrap();
        let mut dg = Buffer::<f32>::default();
        dg.size(&mut device, 9 * vertices as usize, AllocLabel("h.dg")).unwrap();
        let mut ef = Buffer::<f32>::default();
        ef.size(&mut device, 1, AllocLabel("h.ef")).unwrap();

        let rows = OperatorRows {
            dynamic_index: e,
            dynamic_value: ef.handle(),
            dynamic_offset: eo,
            reference_index: e,
            reference_value: e,
            reference_offset: eo,
            global_value: ef.handle(),
            fixed_index: fi,
            fixed_offset: fo,
            fixed_value: fv.handle(),
            transpose_pair: tp,
            transpose_offset: to,
            diagonal: dg.handle(),
        };
        let mut state = State::default();
        // Safety: every handle names a live allocation of the right size.
        unsafe {
            partition(&mut device, &mut state, rows, vertices).unwrap();
            refactor(&mut device, &mut state, rows).unwrap();
            build_hierarchy(&mut device, &mut state, rows, vertices, 3).unwrap();
        }
        assert!(state.aggregates > 1, "the fine partition made one domain");
        assert!(
            state.live_depth > 0,
            "the hierarchy did not coarsen: {} fine domains",
            state.aggregates
        );
        for (depth, level) in state.live_levels().iter().enumerate() {
            assert!(
                level.nodes > 0 && level.domains > 0,
                "level {depth} is empty: {} nodes, {} domains",
                level.nodes,
                level.domains
            );
            assert!(
                level.domains <= level.nodes,
                "level {depth} did not coarsen: {} domains over {} nodes",
                level.domains,
                level.nodes
            );
        }

        let mut xb = Buffer::<f32>::default();
        let mut zb = ReadbackBuffer::<f32>::default();
        xb.size(&mut device, 3 * vertices as usize, AllocLabel("h.x")).unwrap();
        zb.size(&mut device, 3 * vertices as usize, AllocLabel("h.z")).unwrap();

        for trial in 0..3usize {
            let r: Vec<f32> = (0..3 * vertices as usize)
                .map(|k| match trial {
                    0 => 1.0,
                    1 => if k % 3 == 0 { 1.0 } else { -0.5 },
                    _ => if k == 41 { 1.0 } else { 0.0 },
                })
                .collect();
            xb.write(&mut device, 0, &r).unwrap();
            zb.seed(&mut device, &vec![0.0f32; 3 * vertices as usize]).unwrap();
            let x_handle = xb.handle();
            let z_handle = zb.handle();
            // Safety: every handle is live and sized.
            device
                .run("h.sweep", |encoder| unsafe {
                    encode_whole_sweep(encoder, &mut state, x_handle, z_handle,
                                       vertices)
                })
                .unwrap();
            zb.download(&mut device).unwrap();
            assert!(
                zb.host().iter().all(|v| v.is_finite()),
                "trial {trial} produced a non-finite z"
            );
            let dot: f64 = r
                .iter()
                .zip(zb.host())
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            assert!(dot > 0.0, "trial {trial}: r . z must be positive, got {dot:e}");
        }
    }

    /// Everything a chain-operator fixture has to keep alive.
    ///
    /// The handles in `rows` name these allocations, so the fixture is one
    /// value a test holds for as long as it uses the view.
    struct Chain {
        rows: OperatorRows,
        _index: Vec<Buffer<u32>>,
        _value: Vec<Buffer<f32>>,
    }

    /// An SPD chain: 20 on the diagonal and -1 to the next vertex, stored as
    /// the fixed matrix's upper triangle plus the transpose references its
    /// lower triangle is reached through.
    fn chain_operator<D: Device>(device: &mut D, vertices: u32) -> Chain {
        let mut offset = vec![0u32];
        let mut index = Vec::new();
        let mut value: Vec<f32> = Vec::new();
        for row in 0..vertices {
            let mut neighbours = vec![row];
            if row + 1 < vertices {
                neighbours.push(row + 1);
            }
            for n in &neighbours {
                index.push(*n);
                let scale = if *n == row { 20.0 } else { -1.0 };
                for r in 0..3 {
                    for c in 0..3 {
                        value.push(if r == c { scale } else { 0.0 });
                    }
                }
            }
            offset.push(index.len() as u32);
        }
        let mut t_offset = vec![0u32];
        let mut t_pair: Vec<u32> = Vec::new();
        for row in 0..vertices as usize {
            if row > 0 {
                t_pair.push(row as u32 - 1);
                t_pair.push(offset[row - 1] + 1);
            }
            t_offset.push((t_pair.len() / 2) as u32);
        }

        let mut keep: Vec<Buffer<u32>> = Vec::new();
        let mut u32buf = |data: &[u32], label: &'static str, device: &mut D| {
            let mut b = Buffer::<u32>::default();
            b.size(device, data.len().max(1), AllocLabel(label)).unwrap();
            b.write(device, 0, data).unwrap();
            let h = b.handle();
            keep.push(b);
            h
        };
        let fo = u32buf(&offset, "c.fo", device);
        let fi = u32buf(&index, "c.fi", device);
        let to = u32buf(&t_offset, "c.to", device);
        let tp = u32buf(&t_pair, "c.tp", device);
        let eo = u32buf(&vec![0u32; vertices as usize + 1], "c.eo", device);
        let e = u32buf(&[0u32], "c.e", device);
        let mut fv = Buffer::<f32>::default();
        fv.size(device, value.len(), AllocLabel("c.fv")).unwrap();
        fv.write(device, 0, &value).unwrap();
        let mut dg = Buffer::<f32>::default();
        dg.size(device, 9 * vertices as usize, AllocLabel("c.dg")).unwrap();
        let mut ef = Buffer::<f32>::default();
        ef.size(device, 1, AllocLabel("c.ef")).unwrap();
        let rows = OperatorRows {
            dynamic_index: e,
            dynamic_value: ef.handle(),
            dynamic_offset: eo,
            reference_index: e,
            reference_value: e,
            reference_offset: eo,
            global_value: ef.handle(),
            fixed_index: fi,
            fixed_offset: fo,
            fixed_value: fv.handle(),
            transpose_pair: tp,
            transpose_offset: to,
            diagonal: dg.handle(),
        };
        Chain { rows, _index: keep, _value: vec![fv, dg, ef] }
    }

    /// A rebuild REUSES its slots and allocates nothing, and a shorter one
    /// retires the levels it did not reach without freeing them.
    ///
    /// WHAT THIS CATCHES, and why the allocator generation is the assertion:
    /// [`build_hierarchy`] runs once per Newton step, a [`Level`] names eighteen
    /// device allocations, and a `Buffer` has no `Drop`, so a build that
    /// dropped its levels would strand every span it had named. That leak is
    /// invisible to a correctness test, because each rebuilt level is right;
    /// what it moves is `allocator_generation`, which the seam bumps on every
    /// `alloc`, `grow` and `free`. Growing each slot in place and freeing it
    /// only on a structural change makes the steady state zero allocator
    /// traffic per solve, and that is what this pins.
    ///
    /// The second half is what makes retiring safe: after a build that reaches
    /// fewer levels, the deeper slot keeps its allocations (so the next full
    /// build costs nothing) while [`State::live_levels`] stops short of it, and
    /// its counts are zero so a consumer walking further would still skip it.
    #[test]
    fn a_rebuilt_hierarchy_reuses_its_slots_and_retires_the_rest() {
        let mut device = super::super::launch::host_device();
        // Enough vertices for TWO coarse levels at a cap of 16: 1200 fine
        // vertices make about 75 domains, which coarsen again to about five.
        let vertices = 1200u32;
        let chain = chain_operator(&mut device, vertices);
        let rows = chain.rows;
        let mut state = State::default();
        // Safety: every handle names a live allocation of the right size, held
        // by `chain` for the whole test.
        unsafe {
            partition(&mut device, &mut state, rows, vertices).unwrap();
            refactor(&mut device, &mut state, rows).unwrap();
        }
        // THE FIRST BUILD MUST MOVE THE COUNTER, which is what entitles the
        // later builds to assert it does not: a generation that never moved in
        // this path would make every assertion below pass by measuring nothing.
        let first = device.allocator_generation();
        // Safety: as above.
        unsafe {
            build_hierarchy(&mut device, &mut state, rows, vertices, 3).unwrap();
        }
        assert!(
            device.allocator_generation() > first,
            "the first build allocated nothing, so the counter measures nothing"
        );
        assert_eq!(
            state.live_depth, 2,
            "the fixture must coarsen twice for this to test anything"
        );
        let slots = state.levels.len();
        let named: Vec<Handle> = state
            .levels
            .iter()
            .map(|level| level.map_fine.handle())
            .collect();

        // A second build at the same depth: the slots are indexed in place, so
        // nothing is allocated, grown or freed.
        let before = device.allocator_generation();
        // Safety: as above.
        unsafe {
            refactor(&mut device, &mut state, rows).unwrap();
            build_hierarchy(&mut device, &mut state, rows, vertices, 3).unwrap();
        }
        assert_eq!(
            device.allocator_generation(),
            before,
            "a rebuild allocated: {} device blocks are reserved",
            device.bytes_reserved()
        );
        assert_eq!(state.live_depth, 2, "the rebuild lost a level");
        assert_eq!(state.levels.len(), slots, "the rebuild grew the vector");
        let again: Vec<Handle> = state
            .levels
            .iter()
            .map(|level| level.map_fine.handle())
            .collect();
        assert_eq!(named, again, "a level's allocation moved across a rebuild");

        // A SHORTER build: one coarse level, and the deeper slot is retired
        // rather than dropped.
        // Safety: as above.
        unsafe {
            refactor(&mut device, &mut state, rows).unwrap();
            build_hierarchy(&mut device, &mut state, rows, vertices, 2).unwrap();
        }
        assert_eq!(state.live_depth, 1, "a two-level build made a coarse level");
        assert_eq!(state.live_levels().len(), 1, "the walk reaches a retired level");
        assert_eq!(state.levels.len(), slots, "the retired slot was dropped");
        assert_eq!(state.levels[1].nodes, 0, "the retired slot kept its node count");
        assert_eq!(state.levels[1].domains, 0, "the retired slot kept its domains");
        assert_eq!(
            state.levels[1].map_fine.handle(),
            named[1],
            "retiring a slot released its allocation"
        );

        // And back to the full depth, still without allocating: the retired
        // slot's buffers are the ones this build fills.
        let before = device.allocator_generation();
        // Safety: as above.
        unsafe {
            refactor(&mut device, &mut state, rows).unwrap();
            build_hierarchy(&mut device, &mut state, rows, vertices, 3).unwrap();
        }
        assert_eq!(
            device.allocator_generation(),
            before,
            "restoring the depth allocated instead of reusing the retired slot"
        );
        assert_eq!(state.live_depth, 2, "the restored build lost a level");
    }
}
