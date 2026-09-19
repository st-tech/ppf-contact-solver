// File: crates/ppf-cts-solver/src/driver/dyncsr.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The dynamic CSR matrix: the one contact assembles into.
//!
//! Elastic and bending go into the FIXED matrix, whose pattern `builder.rs`
//! knows at build time. Contact cannot: which pairs touch is discovered per
//! step, so a row grows and shrinks and carries its pattern forward. Hence a
//! separate matrix with a separate lifecycle.
//!
//! # The storage is ONE device slab, and that is what this module is
//!
//! The matrix owns three device arrays and nothing else: `dyn_row_offsets`,
//! `dyn_index_buff` and `dyn_value_buff`. A row is an offset into them rather
//! than storage of its own, so every step of the lifecycle is an ordinary
//! dispatch over rows or over contributions, and the per-row algorithms
//! (bisect, heapsort, compact, merge) run one row per thread.
//!
//! Eight passes make up that lifecycle, declared beside their bodies in
//! `csrmat/dynamic_csr.kernel.cpp`:
//!
//! | pass | body |
//! | --- | --- |
//! | begin | `dyn_row_begin_pass` |
//! | dry push | `dyn_dry_push_pass` |
//! | scan | an exclusive scan over `dyn_row_offsets` |
//! | seed | `dyn_row_seed_pass` |
//! | push | `dyn_push_pass` |
//! | compact | `dyn_row_compact_pass` |
//! | scan | an exclusive scan over `fixed_row_offsets` |
//! | emit | `dyn_row_emit_pass` |
//!
//! **THE COUNTING PASS IS WHAT SIZES A ROW'S SLAB, and it is not optional.** A
//! row that took more entries than its slab holds would write into the NEXT
//! row's blocks, and the Hessian that came out would still look assembled. The
//! fill refuses that write and counts the refusal, and a non-zero count is
//! fatal.
//!
//! # A row is a VERTEX, and that is why the compaction rule matters
//!
//! A coarse collider against a fine deformable puts one collider vertex in
//! contact with every fine vertex it touches, so ONE row goes enormous while
//! the mean stays tiny: 33,672 columns measured against a mean of 4. Asking
//! "is this column already in this row?" per contribution, at the row's width,
//! in each of the three places that need the answer cost `asm_contact` 37.0 s
//! per Newton step, measured, 95% of it compaction.
//!
//! The fix is an invariant rather than a faster search:
//!
//! > `push` appends a column ONLY after failing to find it among the carried
//! > ones. So carried columns are distinct and disjoint from appended ones, and
//! > only appended-against-appended can collide.
//!
//! From that: the carried pattern is kept SORTED so both passes bisect it, the
//! appended run is SORTED rather than searched, and the compaction's output is
//! two ascending runs the emit pass MERGES instead of sorting a row.
//!
//! **The disjointness check must stay.** A broken invariant leaves the row
//! NUMERICALLY correct, because the duplicate's carried entry goes unpushed and
//! drops as zero, so the only symptom is a step that costs forty seconds
//! instead of a fraction of one. A divergence with no wrong answer attached is
//! the kind nobody finds. It rides the `[[seam::diag]]` lane of the compaction
//! pass, so a violation names the row, the column and the row's widths and
//! stops the step.
//!
//! # What is still on the host, and why it is here rather than in the kernel
//!
//! [`Self::push`] takes one contribution and STAGES it: a row, a column and
//! nine floats appended to three host arrays, with no search and no arithmetic.
//! The matrix is then built on the device in [`Self::to_flat_into`], which is
//! the first point in the step where a device is in scope.
//!
//! **THAT STAGING IS THE RESIDUAL DIVERGENCE AND IT IS NOT THIS MODULE'S TO
//! REMOVE.** Every contact block is offered to the FIXED pattern first, and
//! only what has no slot there reaches this matrix. That routing decision is
//! made by `driver/contact.rs`'s host loop rather than inside the contact
//! kernel, and while the ROUTING is on the host the triple is too. What this
//! module owes that change is the shape it now has: a counting pass, a slab,
//! and a fill that is a dispatch over contributions. When the routing moves
//! into the contact kernel, the fill's three staged arrays become that kernel's
//! own values and the staging goes with them.
//!
//! # The transpose, and the two scans
//!
//! [`Flat::transpose_into`] is a fill / count / scan / fill / scatter, each
//! pass one row per thread.
//!
//! **BOTH SCANS ARE STILL AT EXTENT ONE, WHICH IS A KNOWN DIVERGENCE AND NOT
//! THE SHAPE THE WORK ASKS FOR.** The shape they owe is a multi-level block
//! scan; at extent one the whole scan is a single lane, which is one core of N
//! on the CPU backend and one thread of a GPU on the other two. What each
//! launch site still waits on is written at the site.

// The per-row readers below have no PRODUCTION caller: the driver reads the
// flat form, while `read_row` and `read_pattern` are how a test inspects one
// row without restating the layout. `block_is_zero` is the compaction's own
// rule, exposed for the same reason.
#![allow(dead_code)]
use ppf_cts_compute::{AllocLabel, Buffer, Device, EncoderExt, Fault, Handle, ReadbackBuffer};
use super::kernels::{
    DynCountTransposePassArgs, DynDryPushPassArgs,
    DynPushPassArgs, DynRowBeginPassArgs, DynRowCompactPassArgs, DynRowEmitPassArgs,
    DynRowSeedPassArgs, DynScatterTransposePassArgs, VecFillU32Args,
};
use super::scene::{Fatal, FatalResult};

// THE ONE ENTRY POINT THIS MODULE STILL NAMES DIRECTLY, AND WHY IT STAYS.
//
// `block_is_zero` has no production caller at all. It is the compaction's own
// zero rule, exposed so a test can ask the same question the compaction will
// rather than restate the threshold, and it returns a verdict rather than
// writing a buffer, so a dispatch could not carry its answer back without a
// change to the C++ side. It takes no thread range, so it is a shim HELPER and
// not a shim launcher, which is the distinction `check-shared-wiring.py` rule
// 10 draws.
extern "C" {
    fn block_is_zero_abi(value: *const f32) -> i32;
}

/// One 3x3 block, column-major, as `Mat3x3f` stores it.
pub type Block = [f32; 9];

/// A block is zero by the rule the compaction applies.
///
/// Eigen's dummy precision, NOT exact equality. The two are different rules and
/// drop different numbers of blocks, which changes the pattern the next step
/// carries, so the question is asked of the shared body rather than restated.
pub fn block_is_zero(value: &Block) -> bool {
    unsafe { block_is_zero_abi(value.as_ptr()) != 0 }
}

/// Grow a buffer to `count` elements, KEEPING what it already holds.
///
/// [`Buffer::size`] zero-fills unconditionally, which is right for a buffer a
/// step rewrites and wrong for the carried pattern, whose whole purpose is to
/// survive from one step to the next. So a growth allocates beside the old one,
/// copies the live prefix across on the device, and releases the old.
fn grow_keeping<T: ppf_cts_compute::Pod, D: Device>(
    buffer: &mut Buffer<T>,
    device: &mut D,
    count: usize,
    keep: usize,
    label: AllocLabel,
) -> Result<(), Fault> {
    if buffer.len() >= count {
        return Ok(());
    }
    let mut next: Buffer<T> = Buffer::default();
    next.size(device, count, label)?;
    if keep > 0 {
        device.copy(
            next.handle(),
            0,
            buffer.handle(),
            0,
            keep * std::mem::size_of::<T>(),
        )?;
    }
    buffer.free(device)?;
    *buffer = next;
    Ok(())
}

/// Size a buffer to at least `count`, and to nothing at all when it already is.
///
/// A RESERVE RATHER THAN A LENGTH, because [`Buffer::size`] zero-fills on every
/// call and these arrays are rewritten each step by a pass that covers exactly
/// what the step needs. Calling it per step would put a device fill of the whole
/// slab in the hottest loop the backend has for no reader; growing only past
/// capacity is what makes per-step reuse allocation-free, so a run over an
/// unchanged scene allocates these arrays once and then never again.
fn reserve<T, D: Device>(
    buffer: &mut Buffer<T>,
    device: &mut D,
    count: usize,
    label: AllocLabel,
) -> Result<(), Fault>
where
    T: ppf_cts_compute::Pod + Default,
{
    if buffer.len() < count {
        buffer.size(device, count, label)?;
    }
    Ok(())
}

/// A symmetric block-sparse matrix whose pattern is discovered per step.
#[derive(Debug, Default)]
pub struct DynCsrMat {
    nrow: usize,

    /// The exclusive scan's per-level block sums; see [`super::scan`].
    ///
    /// A FIELD RATHER THAN A LOCAL, because both scans below run every step and
    /// there is no `Drop`: a per-call allocation would leak its arena span
    /// rather than churn it, which is the rule that a buffer a per-step routine
    /// needs is allocated ONCE and reused.
    scan_scratch: super::scan::ScanScratch,

    /// The carried pattern: `nrow + 1` offsets and the columns they index.
    ///
    /// KEPT ACROSS STEPS, which is the one thing in this structure that is not
    /// rewritten from scratch, and the reason [`grow_keeping`] exists. The emit
    /// pass writes the next step's pattern into `fixed_index`, and the offsets
    /// arrive by swapping [`Self::head`] into this field, so the array the scan
    /// produced is the array the next step bisects rather than a copy of it.
    fixed_offset: Buffer<u32>,
    fixed_index: Buffer<u32>,

    /// The step's slab: per-row reserve offsets, `nrow + 1` long with the total
    /// in the last slot, and the flat column and block arrays it indexes.
    dyn_offset: Buffer<u32>,
    dyn_index: Buffer<u32>,
    /// Nine floats per entry, parallel to `dyn_index`.
    dyn_value: Buffer<f32>,

    /// Per-row width during the step; the next step's pattern offsets after the
    /// second scan, which is what the swap above hands on.
    head: Buffer<u32>,
    /// Where each row's surviving carried run ends and its appended run begins.
    split: Buffer<u32>,
    /// One slot: fills the slab refused, which is a fatal rather than a number.
    refused: Buffer<u32>,

    /// This step's contributions, staged on the host by [`Self::push`].
    stage_row: Vec<u32>,
    stage_column: Vec<u32>,
    stage_block: Vec<f32>,
    /// Their device images. Fields rather than locals so a step that pushes no
    /// more than the last one allocates nothing.
    stage_row_device: Buffer<u32>,
    stage_column_device: Buffer<u32>,
    stage_block_device: Buffer<f32>,
    /// The slot cursor a DEVICE stager claims from, and the count it reached.
    ///
    /// A kernel that appends a block has no sequential counter to use, so it
    /// takes its slot from this and the host reads the total back once per
    /// chunk. That single word is what replaces downloading the whole staged
    /// Hessian and uploading the unpacked blocks, which was 9.4 MB each way per
    /// 16,384-pair chunk.
    stage_claim: ReadbackBuffer<u32>,
    /// The largest total a step has staged on the device, which sizes the next
    /// step's slab so it is allocated once while empty rather than grown under
    /// itself. It never shrinks.
    stage_high_water: usize,
    /// Blocks a device stager appended this step, read back from `stage_claim`.
    /// Added to the host-staged count, so the two paths can coexist while the
    /// remaining callers convert.
    staged_on_device: usize,

    /// Blocks the last step handed on, which with the staged count bounds the
    /// slab this step can possibly need.
    carried: usize,
    /// Blocks stored after the last build.
    stored: usize,
    /// True between `start_rebuild` and `finalize`, so a misuse is caught at
    /// the call rather than as a wrong matrix later.
    building: bool,
}

/// The four handles a device stager needs, taken together so a caller cannot
/// pair one step's cursor with another step's arrays.
pub struct DeviceStaging {
    pub claim: ppf_cts_compute::Handle,
    pub row: ppf_cts_compute::Handle,
    pub column: ppf_cts_compute::Handle,
    pub block: ppf_cts_compute::Handle,
    pub capacity: u32,
}

impl DynCsrMat {
    /// An empty matrix over `rows` vertices, with no carried pattern.
    ///
    /// NOTHING IS ALLOCATED HERE, because a device is not in scope: every
    /// buffer is sized on the first [`Self::to_flat_into`] and grown only past
    /// its capacity afterwards.
    pub fn new(rows: usize) -> Self {
        DynCsrMat {
            nrow: rows,
            ..Default::default()
        }
    }

    pub fn row_count(&self) -> usize {
        self.nrow
    }

    /// Contributions staged this step, before any compaction.
    ///
    /// BOTH PATHS COUNT. A block appended by a kernel is staged exactly as one
    /// appended by [`Self::push`] is, and the sizing passes downstream read
    /// this total rather than either half.
    pub fn staged(&self) -> usize {
        self.stage_row.len() + self.staged_on_device
    }

    /// Blocks the last build stored.
    pub fn stored(&self) -> usize {
        self.stored
    }

    /// Begin a step's assembly.
    pub fn start_rebuild(&mut self) {
        // THE VECTORS ARE CLEARED, NOT DROPPED, so the capacity a busy step
        // reached is still there for the next one and a steady state stages
        // without allocating.
        self.stage_row.clear();
        self.stage_column.clear();
        self.stage_block.clear();
        // THE DEVICE CURSOR IS RESET WITH THEM. It is zeroed on the device by
        // `device_staging`, which is the only place a device is in scope; this
        // clears the host's memory of the last step's total so a step that
        // stages nothing on the device does not inherit one.
        self.staged_on_device = 0;
        self.building = true;
    }

    /// The device staging a kernel appends into, sized for `capacity` blocks.
    ///
    /// THE APPEND STAYS ON THE DEVICE. Doing it on the host instead means
    /// downloading a chunk's whole staged Hessian, unpacking every 3x3 block
    /// and uploading the result, which measures 9.4 MB each way per 16,384-pair
    /// chunk and is the largest single term in a step's host-device traffic.
    ///
    /// THE CURSOR IS ZEROED HERE, which is the only place a device is in scope
    /// between `start_rebuild` and the append. A stale cursor would append this
    /// step's blocks past the last step's total and leave the slots between
    /// them holding whatever the previous chunk wrote, which is a wrong matrix
    /// rather than a loud failure.
    pub fn device_staging<D: Device>(
        &mut self,
        device: &mut D,
        capacity: usize,
    ) -> FatalResult<DeviceStaging> {
        // THE BASE IS BOTH STAGED COUNTS, and getting this wrong is what the
        // `pAp <= 0` guard caught: `deposit` runs once per CHUNK per contact
        // KIND, so a base of only the host count is zero every time and each
        // chunk overwrites the last one's blocks from slot zero. The dropped
        // couplings left the Newton matrix non-SPD and the solve aborted at
        // iteration 19 of a drape that had been clean.
        //
        // The claim below is reset per call on purpose: each chunk claims from
        // zero RELATIVE to this base, and `take_device_staged` adds what it
        // claimed to the running total.
        let base = self.stage_row.len() + self.staged_on_device;
        let total = base + capacity;
        // THE SLAB IS NOT SIZED HERE. `Buffer::size` ends in `fill_zero` over
        // the WHOLE allocation, so it zeroes on every call and not only when it
        // grows; resizing from a later chunk would wipe every block the earlier
        // chunks staged. That is the defect `plastic` found, a non-SPD matrix
        // at frame 22.
        //
        // `reserve_device_staging` does the sizing ONCE per step, before any
        // chunk runs, and this only hands out the next span. A span past the
        // end is refused rather than served.
        if total > self.stage_row_device.len() {
            return Err(Fatal::invariant(format!(
                "the dynamic staging slab holds {} blocks and this chunk needs {total}. It is \
                 sized once per step by reserve_device_staging, because resizing it zeroes it \
                 and would discard everything staged so far",
                self.stage_row_device.len()
            )));
        }
        if self.stage_claim.len() < 1 {
            self.stage_claim.size(device, 1, AllocLabel("csr.dyn_stage_claim"))?;
        }
        { let h = self.stage_claim.handle(); device.fill_zero(h, 4)?; }
        Ok(DeviceStaging {
            claim: self.stage_claim.handle(),
            row: self.stage_row_device.span(base, capacity),
            column: self.stage_column_device.span(base, capacity),
            block: self.stage_block_device.span(9 * base, 9 * capacity),
            capacity: capacity as u32,
        })
    }

    /// Size the device staging for a WHOLE step, before any chunk stages into
    /// it.
    ///
    /// SEPARATE FROM `device_staging` ON PURPOSE. `Buffer::size` zeroes the
    /// whole allocation on every call, so the slab can only be sized while it
    /// is empty; a resize from inside the chunk loop discards what the earlier
    /// chunks staged, which is silent and leaves a matrix missing Hessian
    /// couplings. Call this once, then hand out spans.
    pub fn reserve_device_staging<D: Device>(
        &mut self,
        device: &mut D,
        capacity: usize,
    ) -> FatalResult<()> {
        // BOTH HALVES, WHICH IS THE SAME BASE `device_staging` HANDS OUT FROM.
        // It spans at `stage_row.len() + staged_on_device`, so sizing against
        // the device half alone is short by exactly the host half, and the
        // shortfall is invisible until a step stages on BOTH: measured on
        // `plastic`, three host-staged blocks against a 262,144-slot chunk gave
        // "the slab holds 262144 blocks and this chunk needs 262147".
        let staged = self.staged();
        // AT LEAST ONE ELEMENT, ALWAYS. A path that stages nothing still hands
        // the kernel these handles, and a generated entry asserts
        // `args->f.arena < args->seam_arena_count` and RESOLVES the address
        // before the body runs, so a field must name a real arena even when the
        // body will not read it. An unallocated buffer spans to `Handle::NONE`,
        // whose arena is `u32::MAX`, and the assert fires inside a generated
        // file nobody wrote. That is the rule that an absent buffer takes a
        // real ZERO-LENGTH allocation rather than a null.
        let want = (staged + capacity).max(self.stage_high_water).max(1);
        if want <= self.stage_row_device.len() {
            return Ok(()); // already big enough, and a no-op does not zero
        }
        // Preserve deposited blocks on the device while growing the slab.
        // THE CARRY IS BOUNDED BY WHAT THE ALLOCATION ACTUALLY HOLDS, which is
        // NOT the same number as the one that sized it. `staged` counts the host
        // half too, and the host blocks are written into this slab only at
        // `finish_rebuild`; before that the device half sits ABOVE an
        // uninitialized host prefix and the buffer itself may be shorter than
        // `staged`, because on the first grow of a step it can still be the
        // one-element placeholder. Reading `staged` elements from it is then an
        // out-of-bounds readback, which the ABI refuses by name: `yarn` and
        // `cards` both died on "the window [0, 488) is outside the 4 bytes this
        // handle names". Carry the overlap and let the grow zero the rest.
        let carry_len = staged.min(self.stage_row_device.len());
        grow_keeping(&mut self.stage_row_device, device, want, carry_len, AllocLabel("csr.dyn_stage_row"))?;
        grow_keeping(&mut self.stage_column_device, device, want, carry_len, AllocLabel("csr.dyn_stage_column"))?;
        grow_keeping(&mut self.stage_block_device, device, 9 * want, 9 * carry_len, AllocLabel("csr.dyn_stage_block"))?;
        Ok(())
    }

    /// Read the cursor back and add what a kernel appended to the staged count.
    ///
    /// IT FAILS ON OVERFLOW RATHER THAN TRUNCATING. The kernel drops a block
    /// whose claim is past the capacity and counts it anyway, so a claim above
    /// what was offered means Hessian couplings were lost, and a lost coupling
    /// leaves an indefinite matrix that `pAp <= 0` reports only sometimes.
    pub fn take_device_staged<D: Device>(
        &mut self,
        device: &mut D,
        capacity: usize,
    ) -> FatalResult<()> {
        self.stage_claim.download(device)?;
        let claimed = self.stage_claim.host()[0] as usize;
        if claimed > capacity {
            // THE CLAIM IS IN THE MESSAGE, because a caller sizing a reserve
            // needs the number it fell short by and not only that it did. The
            // kernel counts past capacity on purpose for exactly this.
            return Err(Fatal::invariant(format!(
                "solver driver: the contact Hessian staged {claimed} blocks into a dynamic \
                 slab offered {capacity}. Every block past the offer was DROPPED, and a \
                 dropped block is a lost Hessian coupling: the assembled Newton matrix is \
                 missing a term and is no longer the SPD-by-assembly matrix the PCG guards \
                 assume"
            )));
        }
        self.staged_on_device += claimed;
        self.stage_high_water = self.stage_high_water.max(self.staged_on_device);
        Ok(())
    }

    /// Stage `block` at `(row, column)`.
    ///
    /// TRANSPORT AND NOTHING ELSE: no search, no fold, no ordering. Which slot
    /// the block lands in is the fill pass's decision and is made on the device
    /// against the pattern the row carries, which is the only place the carried
    /// and appended runs can be kept disjoint.
    ///
    /// Nothing is dropped: unlike the FIXED matrix, whose `push` silently
    /// returns false for a column outside its build-time pattern, this matrix
    /// has no pattern to be outside of. The only invalid call is an
    /// out-of-range row or column, which panics.
    pub fn push(&mut self, row: usize, column: u32, block: &Block) {
        assert!(
            self.building,
            "push() outside a rebuild: the contribution would be staged into a \
             step that has already been built"
        );
        let rows = self.nrow;
        assert!(row < rows, "row {row} is outside a matrix of {rows} rows");
        assert!(
            (column as usize) < rows,
            "column {column} is outside a matrix of {rows} rows"
        );
        self.stage_row.push(row as u32);
        self.stage_column.push(column);
        self.stage_block.extend_from_slice(block);
    }

    /// Close the step's staging.
    ///
    /// The matrix is BUILT in [`Self::to_flat_into`], for the reason the module
    /// documentation gives: a device is not in scope until then. This is the
    /// boundary that says no further contribution is coming, which is what the
    /// counting pass needs before it can size anything.
    pub fn finalize(&mut self) {
        assert!(self.building, "finalize() without a matching start_rebuild()");
        self.building = false;
    }

    /// Build the matrix on the device and leave it in `flat`.
    ///
    /// TAKES THE DESTINATION RATHER THAN RETURNING ONE, and that is the whole
    /// point of the signature: this runs once per Newton step, and a function
    /// that returned a `Flat` would allocate three device buffers per call. A
    /// caller-owned `Flat` holds its `offset`, `index` and `value` buffers for
    /// the run, and `Flat::size_for` grows them only past capacity, so a step
    /// over an unchanged scene allocates nothing.
    ///
    /// # The slab's capacity is DERIVED, not configured
    ///
    /// No maximum-nnz parameter decides it, and none is needed: a row reserves
    /// its carried pattern plus at most one slot per contribution it receives,
    /// so `carried + staged` is an EXACT upper bound on what the counting pass
    /// can produce, and it is known on the host before any pass runs. The bound
    /// is the capacity and the refusal counter is the guard, which is the half
    /// that actually protects the neighboring row's blocks.
    pub fn to_flat_into<D: Device>(&mut self, device: &mut D, flat: &mut Flat) -> FatalResult<()> {
        assert!(
            !self.building,
            "to_flat_into() inside a rebuild: the staged contributions are not \
             complete, so the counting pass would size the slab for a subset"
        );
        let nrow = self.nrow;
        flat.rows = nrow;
        if nrow == 0 {
            flat.nnz = 0;
            return Ok(());
        }
        // BOTH HALVES. `staged()`'s own doc says the sizing passes downstream
        // read this total rather than either half, and this is the pass it
        // meant: reading `stage_row.len()` here makes every block a KERNEL
        // staged invisible to the merge, so the dry pass reserves no slot for
        // it and the fill never walks it. The block is then silently dropped,
        // which is a lost Hessian coupling of exactly the kind
        // `FixedCSRMat::push` can produce, one level further out.
        //
        // It is dormant while the contact deposit runs on the host, because
        // `staged_on_device` is zero and the collider path offers no slots at
        // all. Under the fused deposit it bites the moment contact density puts
        // a block outside the fixed pattern: measured on `plastic`, the solve
        // ran normally for 0.35 s and then collapsed from 78 PCG iterations per
        // solve to 2, the matrix having lost its stiffest couplings.
        let staged = self.staged();
        // THE EXACT UPPER BOUND ON THE COUNTING PASS, computed rather than
        // assumed: every row opens its reserve at its carried width and the dry
        // pass adds at most one slot per contribution.
        let bound = self.carried + staged;
        let rows = nrow as u32;

        self.size_for(device, bound, staged)?;

        device.run("dyncsr.deposit", |encoder| {
        // 1. Order each row's carried pattern and open its reserve at that
        //    width. `dyn_row_begin_pass`.
        //
        // Safety: every buffer a record names is borrowed for the whole call and
        // `Device::run` executes and waits before returning. The same holds for
        // every dispatch below.
        unsafe {
            encoder.elements(
                &DynRowBeginPassArgs {
                    fixed_offset: self.fixed_offset.handle(),
                    fixed_index: self.fixed_index.handle(),
                    reserve: self.dyn_offset.handle(),
                    count: rows,
                    seam_arena_count: 0,
                },
                rows,
            )
        }?;

        // 2. One slot reserved per contribution whose column the row does not
        //    already carry. `dyn_dry_push_pass`.
        if staged > 0 {
            unsafe {
                encoder.elements(
                    &DynDryPushPassArgs {
                        fixed_offset: self.fixed_offset.handle(),
                        fixed_index: self.fixed_index.handle(),
                        push_row: self.stage_row_device.span(0, staged),
                        push_column: self.stage_column_device.span(0, staged),
                        reserve: self.dyn_offset.handle(),
                        count: staged as u32,
                        seam_arena_count: 0,
                    },
                    staged as u32,
                )
            }?;
        }

        // 3. The reserves become slab offsets, IN PLACE, which is what the
        //    shared body is written to allow: it reads a row's count before it
        //    writes that row's offset, so one buffer may be passed twice.
        {
            let array = self.dyn_offset.handle();
            unsafe { self.scan_scratch.encode_exclusive(encoder, array, rows) }?;
        }

        // 4. Lay the carried pattern into each row's slab and open its width at
        //    that pattern. `dyn_row_seed_pass`.
        unsafe {
            encoder.elements(
                &DynRowSeedPassArgs {
                    fixed_offset: self.fixed_offset.handle(),
                    fixed_index: self.fixed_index.handle(),
                    dyn_offset: self.dyn_offset.handle(),
                    dyn_index: self.dyn_index.handle(),
                    dyn_value: self.dyn_value.handle(),
                    head: self.head.handle(),
                    count: rows,
                    seam_arena_count: 0,
                },
                rows,
            )
        }?;

        // 5. The fill. `dyn_push_pass`.
        if staged > 0 {
            unsafe {
                encoder.elements(
                    &DynPushPassArgs {
                        fixed_offset: self.fixed_offset.handle(),
                        dyn_offset: self.dyn_offset.handle(),
                        dyn_index: self.dyn_index.handle(),
                        dyn_value: self.dyn_value.handle(),
                        push_row: self.stage_row_device.span(0, staged),
                        push_column: self.stage_column_device.span(0, staged),
                        push_block: self.stage_block_device.span(0, 9 * staged),
                        head: self.head.handle(),
                        refused: self.refused.handle(),
                        count: staged as u32,
                        seam_arena_count: 0,
                    },
                    staged as u32,
                )
            }?;
        }

        Ok(())
        })?;

        // 6. THE FILL'S VERDICT ON ITS OWN SLAB, READ BEFORE THE COMPACTION
        //    AND NOT AFTER IT, and that order is not a matter of taste. A
        //    refused write still CLAIMED its slot: the fill increments the
        //    row's width and only then finds the slot is past the slab, so an
        //    overflowed row's width names entries that are not its own, and
        //    compacting it would sort and fold the NEXT row's blocks. The count
        //    comes back as a single scalar, so the read costs one word.
        let refused = self.refused.read_one(device, 0).map_err(|error| {
            Fatal::invariant(format!(
                "solver driver: cannot read back the dynamic CSR overflow count: {error:?}"
            ))
        })?;
        if refused != 0 {
            return Err(Fatal::invariant(format!(
                "solver driver: the dynamic CSR fill refused {refused} of {staged} \
                 contributions. The counting pass reserved fewer entries for those rows \
                 than the fill then wrote, so the fill was refused to keep it from running \
                 into the neighboring row's blocks. The two passes must visit the same \
                 contributions: a predicate that differs between them, or a contact set \
                 that changed between counting and filling, will do this. The assembled \
                 matrix is incomplete, so the step is abandoned rather than solved against \
                 a silently wrong Hessian."
            )));
        }

        // THE SLAB'S FILL FRACTION, which is the number that answers whether the
        // contact Hessian needs more device memory. `dyn_offset` is the scanned
        // per-row reserve, so its last slot is the total the counting pass
        // reserved across every row, and `dyn_index` is the slab that total is
        // spent against. The ratio is published on the channel the server
        // declares for it, `advance.dyn_consumed.out`.
        //
        // IT IS READ HERE RATHER THAN AFTER THE SCAN THAT PRODUCES IT, so it
        // costs no synchronization of its own: the overflow check above already
        // reads a word back at this point, and nothing between the scan and here
        // writes `dyn_offset`. Every later pass reads it, which the fill relies
        // on when it takes a row's width as `dyn_offset[row + 1] - dyn_offset[row]`.
        //
        // A RATIO APPROACHING ONE MEANS THE NEXT STEP MAY NOT FIT. The fill
        // refuses a write past the slab and that refusal is fatal above, so this
        // is the warning that precedes it rather than a curiosity: a run whose
        // ratio climbs toward 1.0 is one contact set away from being abandoned.
        let reserved = self.dyn_offset.read_one(device, self.nrow).map_err(|error| {
            Fatal::invariant(format!(
                "solver driver: cannot read back the dynamic CSR reserve total: {error:?}"
            ))
        })?;
        let capacity = self.dyn_index.handle().size;
        if capacity != 0 {
            super::log::mark("advance", "dyn_consumed", reserved as f64 / capacity as f64);
        }

        // 7. Compaction, and the disjointness check that rides its diagnostic
        //    lane. `Row::finalize`.
        unsafe {
            device.launch(
                "dyncsr.row_compact",
                &DynRowCompactPassArgs {
                    fixed_offset: self.fixed_offset.handle(),
                    fixed_index: self.fixed_index.handle(),
                    dyn_offset: self.dyn_offset.handle(),
                    dyn_index: self.dyn_index.handle(),
                    dyn_value: self.dyn_value.handle(),
                    head: self.head.handle(),
                    split: self.split.handle(),
                    count: rows,
                    seam_arena_count: 0,
                },
                rows,
            )
        }
        .map_err(|fault| {
            Fatal::invariant(format!(
                "solver driver: the dynamic CSR compaction reported a column present in \
                 both a row's carried run and its appended run, so the fill appended a \
                 column it should have found. The row is still numerically correct, which \
                 is why nothing else reports this: the only symptom is that compaction \
                 goes quadratic in the row's width. {}",
                describe(&fault)
            ))
        })?;

        // 8. The surviving widths become the next step's pattern offsets.
        {
            let array = self.head.handle();
            Self::scan_into(&mut self.scan_scratch, device, "dyncsr.pattern_scan", array, rows)?;
        }

        // THE SECOND AND LAST SCALAR. The total is the last offset, which the
        // scan wrote from the same running sum it returned. A dispatch produces
        // no value, so the driver reads the slot the pass wrote rather than a
        // return code, and the whole build crosses to the host twice: this and
        // the refusal count above.
        let total = self.head.read_one(device, nrow).map_err(|error| {
            Fatal::invariant(format!(
                "solver driver: cannot read back the dynamic CSR block count: {error:?}"
            ))
        })? as usize;
        assert!(
            total <= bound,
            "the dynamic CSR stored {total} blocks against a slab bounded at {bound}, \
             so a row was handed more entries than any contribution could have produced"
        );

        // 9. The pattern the next step carries, and the contiguous copy the
        //    sparse matvec reads.
        flat.size_for(device, nrow, total)?;
        flat.offset.copy_from(device, &self.head).map_err(|error| {
            Fatal::out_of_memory(format!(
                "solver driver: cannot publish the dynamic CSR row offsets: {error:?}"
            ))
        })?;
        flat.nnz = total;
        unsafe {
            device.launch(
                "dyncsr.row_emit",
                &DynRowEmitPassArgs {
                    dyn_offset: self.dyn_offset.handle(),
                    dyn_index: self.dyn_index.handle(),
                    dyn_value: self.dyn_value.handle(),
                    head: self.head.handle(),
                    split: self.split.handle(),
                    pattern: self.fixed_index.handle(),
                    flat_index: flat.index.handle(),
                    flat_value: flat.value.handle(),
                    count: rows,
                    seam_arena_count: 0,
                },
                rows,
            )
        }?;

        // THE SCAN'S OUTPUT BECOMES THE CARRIED OFFSETS BY SWAP, not by copy.
        // `head` holds exactly what the next step's `fixed_offset` must be, and
        // the next step rewrites `head` end to end (the seed pass fills every
        // row and the scan fills the last slot), so the two arrays exchange
        // roles rather than one being copied onto the other.
        std::mem::swap(&mut self.fixed_offset, &mut self.head);
        self.carried = total;
        self.stored = total;
        Ok(())
    }

    /// The exclusive scan, in place over one array of `rows` counts.
    ///
    /// A MULTI-LEVEL SCAN over the whole array, not a single pass at EXTENT
    /// ONE: that would be one device thread walking every row in sequence. See
    /// [`super::scan`] for why the levels are element-wise rather than a
    /// cooperative block scan, and why that substitution cannot change the
    /// answer.
    fn scan_into<D: Device>(
        scratch: &mut super::scan::ScanScratch,
        device: &mut D,
        region: &'static str,
        array: Handle,
        rows: u32,
    ) -> FatalResult<()> {
        // Safety: the buffer is borrowed for the whole call and every dispatch
        // inside runs and waits before the next.
        unsafe { scratch.exclusive(device, region, array, rows) }?;
        Ok(())
    }

    /// Size every buffer the build reads or writes.
    fn size_for<D: Device>(
        &mut self,
        device: &mut D,
        bound: usize,
        staged: usize,
    ) -> FatalResult<()> {
        let nrow = self.nrow;
        let carried = self.carried;
        let sized = (|| -> Result<(), Fault> {
            // THE CARRIED PATTERN GROWS WITHOUT LOSING ITSELF. Its offsets never
            // change width, so they are sized once; its columns can outgrow
            // their allocation, and what is in them is this step's input.
            reserve(&mut self.fixed_offset, device, nrow + 1, AllocLabel("csr.dyn_pattern_offset"))?;
            grow_keeping(
                &mut self.fixed_index,
                device,
                bound.max(1),
                carried,
                AllocLabel("csr.dyn_pattern_index"),
            )?;
            reserve(&mut self.dyn_offset, device, nrow + 1, AllocLabel("csr.dyn_reserve"))?;
            // The scan's scratch, sized for the longest array either scan walks,
            // which is `nrow`. Sized here rather than at the call so a step
            // never allocates.
            self.scan_scratch.size(device, nrow as u32)?;
            reserve(&mut self.head, device, nrow + 1, AllocLabel("csr.dyn_head"))?;
            reserve(&mut self.split, device, nrow, AllocLabel("csr.dyn_split"))?;
            reserve(&mut self.dyn_index, device, bound.max(1), AllocLabel("csr.dyn_index"))?;
            reserve(&mut self.dyn_value, device, 9 * bound.max(1), AllocLabel("csr.dyn_value"))?;
            reserve(&mut self.refused, device, 1, AllocLabel("csr.dyn_refused"))?;
            // THE REFUSAL COUNTER OPENS AT ZERO EVERY STEP, and a four-byte
            // write is what does it: `Buffer::size` zero-fills only when it
            // allocates, which is once.
            self.refused.write(device, 0, &[0u32])?;
            if staged > 0 {
                reserve(&mut self.stage_row_device, device, staged, AllocLabel("csr.dyn_stage_row"))?;
                reserve(
                    &mut self.stage_column_device,
                    device,
                    staged,
                    AllocLabel("csr.dyn_stage_column"),
                )?;
                reserve(
                    &mut self.stage_block_device,
                    device,
                    9 * staged,
                    AllocLabel("csr.dyn_stage_block"),
                )?;
                self.stage_row_device.write(device, 0, &self.stage_row)?;
                self.stage_column_device.write(device, 0, &self.stage_column)?;
                self.stage_block_device.write(device, 0, &self.stage_block)?;
            }
            Ok(())
        })();
        sized.map_err(|error| {
            Fatal::out_of_memory(format!(
                "solver driver: cannot size the dynamic CSR for {nrow} rows and {staged} \
                 staged contributions: {error:?}"
            ))
        })
    }

    /// One row's stored columns and blocks, read back from a built matrix.
    ///
    /// A row comes out of compaction as the surviving CARRIED entries in arrival
    /// order followed by the surviving APPENDED ones in ascending order, which
    /// is not one ascending sequence. Every reader walks a whole row, so that
    /// costs nothing; a test comparing rows must compare BY COLUMN and not by
    /// position.
    pub fn read_row<D: Device>(
        &self,
        device: &mut D,
        flat: &Flat,
        row: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), Fault> {
        let mut bounds = [0u32; 2];
        flat.offset.read(device, row, &mut bounds)?;
        let (begin, end) = (bounds[0] as usize, bounds[1] as usize);
        let mut columns = vec![0u32; end - begin];
        let mut blocks = vec![0.0f32; 9 * (end - begin)];
        if end > begin {
            flat.index.read(device, begin, &mut columns)?;
            flat.value.read(device, 9 * begin, &mut blocks)?;
        }
        Ok((columns, blocks))
    }

    /// The pattern this row carries into the next step, ascending.
    pub fn read_pattern<D: Device>(&self, device: &mut D, row: usize) -> Result<Vec<u32>, Fault> {
        let mut bounds = [0u32; 2];
        self.fixed_offset.read(device, row, &mut bounds)?;
        let (begin, end) = (bounds[0] as usize, bounds[1] as usize);
        let mut columns = vec![0u32; end - begin];
        if end > begin {
            self.fixed_index.read(device, begin, &mut columns)?;
        }
        Ok(columns)
    }

    /// One block by column, or `None`.
    pub fn read_block<D: Device>(
        &self,
        device: &mut D,
        flat: &Flat,
        row: usize,
        column: u32,
    ) -> Result<Option<Block>, Fault> {
        let (columns, blocks) = self.read_row(device, flat, row)?;
        Ok(columns.iter().position(|c| *c == column).map(|slot| {
            let mut block = [0.0f32; 9];
            block.copy_from_slice(&blocks[9 * slot..9 * slot + 9]);
            block
        }))
    }
}

/// What a fault says, for a message that has to carry it into a `Fatal`.
fn describe(fault: &Fault) -> String {
    match fault {
        Fault::Device { diag, .. } => match diag.first.as_ref() {
            Some(first) => format!(
                "{} failing checks; the first names row {}, column {}, carried {}, width {}",
                diag.failures,
                first.payload[0],
                first.payload[1],
                first.payload[2],
                first.payload[3]
            ),
            None => format!(
                "{} failing checks, with no record; the diagnostic channel lost the first",
                diag.failures
            ),
        },
        other => format!("{other:?}"),
    }
}

/// The flat form: row offsets, column indices and blocks.
///
/// WHY THE CONTIGUOUS COPY EXISTS AT ALL. A matvec that walked a row's own
/// `index`, `value` and width would pay nothing for the gap between that width
/// and its reserve, and would need no flat form. `fixed_csr_apply_row` instead
/// walks `offset[row]` to `offset[row + 1]`, which admits no gap, so the emit
/// pass copies each row into these three arrays. That is a choice of STORAGE
/// rather than of arithmetic, and it is the shape the sparse matvec's entry
/// declaration asks for.
#[derive(Debug, Default)]
pub struct Flat {
    pub offset: Buffer<u32>,
    pub index: Buffer<u32>,
    pub value: Buffer<f32>,
    /// Rows, which is the matrix's row count and not a length of `offset`:
    /// `offset` is a reserve and can be wider.
    pub rows: usize,
    /// Stored blocks, for the same reason.
    pub nnz: usize,
}

impl Flat {
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    fn size_for<D: Device>(
        &mut self,
        device: &mut D,
        rows: usize,
        nnz: usize,
    ) -> FatalResult<()> {
        let sized = (|| -> Result<(), Fault> {
            reserve(&mut self.offset, device, rows + 1, AllocLabel("csr.flat_offset"))?;
            reserve(&mut self.index, device, nnz.max(1), AllocLabel("csr.flat_index"))?;
            reserve(&mut self.value, device, 9 * nnz.max(1), AllocLabel("csr.flat_value"))?;
            Ok(())
        })();
        sized.map_err(|error| {
            Fatal::out_of_memory(format!(
                "solver driver: cannot size the flat dynamic matrix for {rows} rows and \
                 {nnz} blocks: {error:?}"
            ))
        })
    }

    /// The transpose index a symmetric matvec reads to reach the lower
    /// triangle, filling `out` and REUSING its allocations.
    ///
    /// Takes the destination for the reason [`DynCsrMat::to_flat_into`] does: it
    /// runs once per Newton step, and returning a `Transpose` would allocate
    /// four device buffers per call.
    ///
    /// # The sequence: fill, count, scan, fill, scatter
    ///
    /// FILL, COUNT, SCAN, FILL, SCATTER, and every one of those but the scan is
    /// an ordinary element-wise dispatch over the rows:
    ///
    /// 1. **fill** the per-column arrival counters with zero, which is a fill
    ///    KERNEL rather than a host write. It covers
    ///    `rows + 1` rather than `rows` because the scan writes the total into
    ///    the last slot, and a slot the scan is about to overwrite still has to
    ///    hold a defined value first: Metal returns whatever bytes an unwritten
    ///    allocation holds rather than faulting.
    /// 2. **count** each column's arrivals, one row per thread, adding into the
    ///    counter of every off-diagonal column that row names.
    /// 3. **scan** the counts into exclusive starts, in place, with the total in
    ///    the last slot.
    /// 4. **fill** the claim counters with zero. A per-row form would set each
    ///    counter beside the span its arrivals land in; here the spans ARE
    ///    `transpose_offset`, because this `Transpose` is already flat, so the
    ///    clear is all that is left.
    /// 5. **scatter**, one row per thread again, each arrival taking the place
    ///    its column's claim counter hands it.
    ///
    /// # The bound the two row kernels run under is the HOST's
    ///
    /// Each walks `row_offset[row]` to `row_offset[row + 1]` and reads
    /// `index[slot]` across that run, so the slot a thread touches is DATA
    /// rather than a function of its own index, and no `[[seam::bound]]` on
    /// either entry could cover it. What covers it is the emit pass: it writes
    /// exactly `head[row + 1] - head[row]` entries at `head[row]`, and the last
    /// offset is the total the flat arrays are sized to, asserted below.
    ///
    /// WHICH PLACE INSIDE A COLUMN'S RUN AN ARRIVAL GETS IS NOT PART OF THE
    /// CONTRACT. Claiming with a grid-wide `atomicAdd` on each column's counter
    /// assigns those places in an order that differs run to run; the `Claim`
    /// kernel here runs as one ascending pass, so its assignment is the
    /// ascending one. What holds either way is the SET: every stored off-diagonal
    /// block appears exactly once in its column's run, naming the row it came
    /// from and the slot it occupies in the flat value array.
    pub fn transpose_into<D: Device>(
        &mut self,
        device: &mut D,
        out: &mut Transpose,
    ) -> FatalResult<()> {
        let rows = self.rows;
        let nnz = self.nnz;
        let Transpose {
            offset: transpose_offset,
            index: transpose_index,
            value: transpose_value,
            cursor,
            total,
            scan_scratch,
        } = out;
        let sized = (|| -> Result<(), Fault> {
            reserve(transpose_offset, device, rows + 1, AllocLabel("csr.transpose_row_offset"))?;
            reserve(transpose_index, device, nnz.max(1), AllocLabel("csr.transpose_row_index"))?;
            reserve(transpose_value, device, nnz.max(1), AllocLabel("csr.transpose_row_value"))?;
            reserve(cursor, device, rows + 1, AllocLabel("csr.transpose_cursor"))?;
            Ok(())
        })();
        sized.map_err(|error| {
            Fatal::out_of_memory(format!(
                "solver driver: cannot size the dynamic transpose: {error:?}"
            ))
        })?;
        if rows == 0 {
            *total = 0;
            return Ok(());
        }
        let count = rows as u32;
        scan_scratch.size(device, count)?;

        device.run("dyncsr.transpose", |encoder| {
        // 1. The counters open at zero.
        let zero_counts = VecFillU32Args {
            array: transpose_offset.handle(),
            value: 0,
            count: count + 1,
            seam_arena_count: 0,
        };
        // Safety: every buffer a record names is borrowed for the whole call,
        // and `Device::run` executes and waits before returning. The same holds
        // for the four dispatches below.
        unsafe { encoder.elements(&zero_counts, count + 1) }?;

        // 2. One row per thread, adding into the column it names.
        let counts = DynCountTransposePassArgs {
            row_offset: self.offset.handle(),
            index: self.index.handle(),
            transpose_count: transpose_offset.handle(),
            count,
            seam_arena_count: 0,
        };
        unsafe { encoder.elements(&counts, count) }?;

        // 3. Counts to exclusive starts, IN PLACE, through the multi-level
        //    scan rather than a single pass at EXTENT ONE, which would be one
        //    device thread walking the array in sequence. See [`super::scan`].
        {
            let array = transpose_offset.handle();
            // Safety: the buffer outlives the call and every dispatch inside
            // runs and waits before the next.
            unsafe { scan_scratch.encode_exclusive(encoder, array, count) }?;
        }

        // 4. The claim counters open at zero.
        let zero_cursor = VecFillU32Args {
            array: cursor.handle(),
            value: 0,
            count,
            seam_arena_count: 0,
        };
        unsafe { encoder.elements(&zero_cursor, count) }?;

        // 5. One row per thread again, each arrival claiming its place.
        let scatter = DynScatterTransposePassArgs {
            row_offset: self.offset.handle(),
            index: self.index.handle(),
            transpose_offset: transpose_offset.handle(),
            cursor: cursor.handle(),
            transpose_index: transpose_index.handle(),
            transpose_value: transpose_value.handle(),
            count,
            seam_arena_count: 0,
        };
        unsafe { encoder.elements(&scatter, count) }?;
        Ok(())
        })?;

        // ONE SCALAR COMES BACK: the total is the last offset, which the scan
        // wrote from the same running sum it returned. The scatter above only
        // READS this array, so the value is the one the scan left.
        *total = transpose_offset.read_one(device, rows).map_err(|error| {
            Fatal::invariant(format!(
                "solver driver: cannot read back the transpose total: {error:?}"
            ))
        })? as usize;
        assert!(
            *total <= nnz,
            "the transpose claims {total} slots against {nnz} stored blocks, so \
             an arrival was counted that no stored block produced"
        );
        Ok(())
    }
}

/// Which stored slots belong to each row's lower triangle.
#[derive(Debug, Default)]
pub struct Transpose {
    pub offset: Buffer<u32>,
    /// The row the stored slot came from.
    pub index: Buffer<u32>,
    /// The slot in [`Flat::value`].
    pub value: Buffer<u32>,
    /// The prefix-sum scratch the build pass writes through.
    ///
    /// A MEMBER RATHER THAN A LOCAL, for the same reason the three arrays above
    /// are: it is per-step scratch, and a local would allocate once per Newton
    /// step.
    pub cursor: Buffer<u32>,
    /// The exclusive scan's per-level block sums; see [`super::scan`]. A member
    /// for the same reason `cursor` is.
    pub scan_scratch: super::scan::ScanScratch,
    /// Stored slots, which is what the two arrays above are TRUNCATED to.
    ///
    /// THE TRUNCATION IS A SPAN, not a `Vec::truncate`: the pass sizes both
    /// arrays for the widest case and the total is only known once the kernel
    /// has run, so a consumer takes `span(0, total)` rather than the whole
    /// allocation. Handing on the whole one would compile and would WIDEN the
    /// bound the entry point checks.
    pub total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::launch::host_device;

    fn block(v: f32) -> Block {
        [v; 9]
    }

    #[test]
    fn growing_device_staging_preserves_all_deposited_fields() {
        let mut device = host_device();
        let mut matrix = DynCsrMat::new(4);
        for count in [1usize, 3, 17, 65, 129] {
            let previous = matrix.staged_on_device;
            matrix.reserve_device_staging(&mut device, count - previous).unwrap();
            if previous > 0 {
                let mut rows = vec![0; previous];
                let mut columns = vec![0; previous];
                let mut blocks = vec![0.0; 9 * previous];
                matrix.stage_row_device.read(&mut device, 0, &mut rows).unwrap();
                matrix.stage_column_device.read(&mut device, 0, &mut columns).unwrap();
                matrix.stage_block_device.read(&mut device, 0, &mut blocks).unwrap();
                assert_eq!(rows, (0..previous as u32).collect::<Vec<_>>());
                assert_eq!(columns, (0..previous as u32).map(|x| 2 * x).collect::<Vec<_>>());
                assert_eq!(blocks, (0..9 * previous).map(|x| x as f32).collect::<Vec<_>>());
            }
            matrix.stage_row_device.write(&mut device, 0, &(0..count as u32).collect::<Vec<_>>()).unwrap();
            matrix.stage_column_device.write(&mut device, 0, &(0..count as u32).map(|x| 2 * x).collect::<Vec<_>>()).unwrap();
            matrix.stage_block_device.write(&mut device, 0, &(0..9 * count).map(|x| x as f32).collect::<Vec<_>>()).unwrap();
            matrix.staged_on_device = count;
        }
    }

    #[test]
    #[ignore = "release timing experiment"]
    fn benchmark_contact_staging_growth() {
        for count in [65536usize, 262144] {
            let mut cold = Vec::new();
            let mut warm = Vec::new();
            for _ in 0..9 {
                let mut device = host_device();
                let mut matrix = DynCsrMat::new(4);
                matrix.reserve_device_staging(&mut device, count).unwrap();
                matrix.staged_on_device = count;
                let start = std::time::Instant::now();
                matrix.reserve_device_staging(&mut device, count).unwrap();
                cold.push(start.elapsed().as_secs_f64() * 1000.0);
                let start = std::time::Instant::now();
                for _ in 0..1000 {
                    matrix.reserve_device_staging(&mut device, count).unwrap();
                }
                warm.push(start.elapsed().as_secs_f64() * 1000.0 / 1000.0);
            }
            cold.sort_by(f64::total_cmp);
            warm.sort_by(f64::total_cmp);
            eprintln!("staging count={count} cold_ms={:.6} [{:.6},{:.6}] warm_ms={:.9}",
                      cold[4], cold[0], cold[8], warm[4]);
        }
    }

    #[test]
    #[ignore = "release timing experiment"]
    fn benchmark_contact_csr_rebuild() {
        for rows in [1024usize, 8192] {
            let mut device = host_device();
            let mut matrix = DynCsrMat::new(rows);
            let mut flat = Flat::default();
            let mut transpose = Transpose::default();
            let pushes: Vec<_> = (0..rows)
                .flat_map(|row| (row..(row + 3).min(rows)).map(move |column| (row, column as u32, 1.0)))
                .collect();
            let mut samples = Vec::new();
            for batch in 0..7 {
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    build(&mut matrix, &mut device, &mut flat, &pushes);
                    flat.transpose_into(&mut device, &mut transpose).unwrap();
                    assert_eq!(flat.nnz(), pushes.len());
                    assert_eq!(transpose.total, pushes.len() - rows);
                }
                if batch > 0 {
                    samples.push(start.elapsed().as_secs_f64() * 1000.0 / 10.0);
                }
            }
            samples.sort_by(f64::total_cmp);
            eprintln!("csr rows={rows} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
                      samples[3], samples[0], samples[5]);
        }
    }

    /// One step: stage the contributions and build the matrix.
    ///
    /// EVERY TEST GOES THROUGH THE DEVICE, because there is no host copy of the
    /// matrix left to inspect. That is the point of the conversion: the row
    /// algorithms run one row per thread out of flat storage, so what a test
    /// reads is what the sparse matvec reads.
    fn build<D: Device>(m: &mut DynCsrMat, device: &mut D, flat: &mut Flat, pushes: &[(usize, u32, f32)]) {
        m.start_rebuild();
        for (row, column, value) in pushes {
            m.push(*row, *column, &block(*value));
        }
        m.finalize();
        m.to_flat_into(device, flat).expect("the build dispatches");
    }

    fn columns_of<D: Device>(m: &DynCsrMat, device: &mut D, flat: &Flat, row: usize) -> Vec<u32> {
        m.read_row(device, flat, row).expect("the row reads back").0
    }

    #[test]
    fn a_push_to_a_new_column_appears_in_the_row() {
        let mut device = host_device();
        let mut m = DynCsrMat::new(4);
        let mut flat = Flat::default();
        build(&mut m, &mut device, &mut flat, &[(1, 2, 1.0), (1, 3, 2.0)]);
        assert_eq!(flat.nnz(), 2);
        assert_eq!(
            m.read_block(&mut device, &flat, 1, 2).unwrap().unwrap(),
            block(1.0)
        );
        assert_eq!(
            m.read_block(&mut device, &flat, 1, 3).unwrap().unwrap(),
            block(2.0)
        );
        assert!(m.read_block(&mut device, &flat, 0, 2).unwrap().is_none());
    }

    #[test]
    fn two_pushes_to_one_column_are_summed() {
        // Two contact pairs can land on one (row, column). The compaction folds
        // the appended run; the carried path adds in place. Both must sum.
        let mut device = host_device();
        let mut m = DynCsrMat::new(2);
        let mut flat = Flat::default();
        build(
            &mut m,
            &mut device,
            &mut flat,
            &[(0, 1, 1.0), (0, 1, 2.0), (0, 1, 4.0)],
        );
        assert_eq!(flat.nnz(), 1);
        assert_eq!(
            m.read_block(&mut device, &flat, 0, 1).unwrap().unwrap(),
            block(7.0)
        );

        // And again through the CARRIED path, which is a different branch of
        // the fill: the column is now in the pattern, so it lands in place.
        assert_eq!(m.read_pattern(&mut device, 0).unwrap(), vec![1]);
        build(&mut m, &mut device, &mut flat, &[(0, 1, 3.0), (0, 1, 5.0)]);
        assert_eq!(flat.nnz(), 1);
        assert_eq!(
            m.read_block(&mut device, &flat, 0, 1).unwrap().unwrap(),
            block(8.0)
        );
    }

    #[test]
    fn the_carried_pattern_comes_out_ascending() {
        // Both later passes bisect it, so a pattern out of order silently
        // misses columns, which reappears as duplicate entries and a quadratic
        // compaction.
        let mut device = host_device();
        let mut m = DynCsrMat::new(10);
        let mut flat = Flat::default();
        let pushes: Vec<(usize, u32, f32)> =
            [7u32, 0, 3, 9, 1, 5].iter().map(|c| (0usize, *c, 1.0f32)).collect();
        build(&mut m, &mut device, &mut flat, &pushes);
        assert_eq!(m.read_pattern(&mut device, 0).unwrap(), vec![0, 1, 3, 5, 7, 9]);
    }

    #[test]
    fn a_pattern_carried_across_steps_is_bisected_not_appended() {
        // The invariant the whole design rests on: a column already carried is
        // found, so it is never appended, so the two runs stay disjoint. A break
        // trips the compaction's diagnostic lane, which turns the build into an
        // error rather than a slow row.
        let mut device = host_device();
        let mut m = DynCsrMat::new(100);
        let mut flat = Flat::default();
        let first: Vec<(usize, u32, f32)> = (0..40u32).map(|c| (0usize, c, 1.0f32)).collect();
        build(&mut m, &mut device, &mut flat, &first);

        let mut second: Vec<(usize, u32, f32)> = (0..40u32).map(|c| (0usize, c, 2.0f32)).collect();
        // And a genuinely new one, which must append.
        second.push((0, 99, 3.0));
        build(&mut m, &mut device, &mut flat, &second);
        assert_eq!(flat.nnz(), 41);
        assert_eq!(
            m.read_block(&mut device, &flat, 0, 0).unwrap().unwrap(),
            block(2.0)
        );
        assert_eq!(
            m.read_block(&mut device, &flat, 0, 99).unwrap().unwrap(),
            block(3.0)
        );
    }

    #[test]
    fn a_block_that_stayed_zero_leaves_the_pattern() {
        // A pair that separated stops contributing, and its block must not be
        // carried forever: the row would grow monotonically over a run.
        let mut device = host_device();
        let mut m = DynCsrMat::new(3);
        let mut flat = Flat::default();
        build(&mut m, &mut device, &mut flat, &[(0, 1, 1.0), (0, 2, 1.0)]);
        assert_eq!(m.read_pattern(&mut device, 0).unwrap(), vec![1, 2]);

        // Column 2 gets nothing this step.
        build(&mut m, &mut device, &mut flat, &[(0, 1, 1.0)]);
        assert_eq!(flat.nnz(), 1);
        assert_eq!(m.read_pattern(&mut device, 0).unwrap(), vec![1]);
    }

    #[test]
    fn zero_is_the_compactions_rule_and_not_exact_equality() {
        // Eigen's dummy precision. The two rules drop different blocks, which
        // changes the pattern the next step carries, so the question is asked
        // of the shared body rather than restated here.
        assert!(block_is_zero(&[0.0; 9]));
        assert!(block_is_zero(&[1.0e-6; 9]));
        assert!(!block_is_zero(&[1.0e-3; 9]));
        // A NaN coefficient is NOT zero, which is what keeps a broken block in
        // the matrix where the solver's guards can see it.
        let mut nan = [0.0f32; 9];
        nan[4] = f32::NAN;
        assert!(!block_is_zero(&nan));
    }

    #[test]
    fn a_row_survives_a_coarse_collider_against_a_fine_deformable() {
        // The shape that cost 37 seconds a step: ONE row far wider than the
        // mean. The test is that it stays correct and finishes, which the
        // quadratic form also did; what it guards is that the invariant holds
        // at that width, since the diagnostic lane is the only thing that
        // reports a break.
        const WIDE: u32 = 4_000;
        let mut device = host_device();
        let mut m = DynCsrMat::new(WIDE as usize + 1);
        let mut flat = Flat::default();
        let mut pushes: Vec<(usize, u32, f32)> = Vec::new();
        for column in 0..WIDE {
            pushes.push((0, column, 1.0));
            // Every other row stays at the measured mean of about four.
            pushes.push((column as usize + 1, column + 1, 1.0));
        }
        build(&mut m, &mut device, &mut flat, &pushes);
        assert_eq!(flat.nnz(), WIDE as usize * 2);
        assert_eq!(columns_of(&m, &mut device, &flat, 0).len(), WIDE as usize);

        // A second step over the same pairs, where every column is carried.
        let repeat: Vec<(usize, u32, f32)> = (0..WIDE).map(|c| (0usize, c, 1.0f32)).collect();
        build(&mut m, &mut device, &mut flat, &repeat);
        let pattern = m.read_pattern(&mut device, 0).unwrap();
        assert_eq!(pattern.len(), WIDE as usize);
        assert!(
            pattern.windows(2).all(|w| w[0] < w[1]),
            "the wide row's pattern is not strictly ascending"
        );
    }

    #[test]
    #[should_panic(expected = "outside a rebuild")]
    fn pushing_outside_a_rebuild_is_refused() {
        let mut m = DynCsrMat::new(2);
        m.push(0, 1, &block(1.0));
    }

    #[test]
    #[should_panic(expected = "outside a matrix")]
    fn an_out_of_range_column_is_refused_rather_than_stored() {
        let mut m = DynCsrMat::new(2);
        m.start_rebuild();
        m.push(0, 5, &block(1.0));
    }

    #[test]
    fn the_flat_form_agrees_with_the_rows_it_came_from() {
        let mut device = host_device();
        let mut m = DynCsrMat::new(3);
        let mut flat = Flat::default();
        build(
            &mut m,
            &mut device,
            &mut flat,
            &[(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 2, 4.0)],
        );
        assert_eq!(flat.nnz(), 4);
        let mut offset = vec![0u32; 4];
        flat.offset
            .read(&mut device, 0, &mut offset)
            .expect("the offsets read back");
        assert_eq!(offset, vec![0, 2, 3, 4]);
        for row in 0..3usize {
            let begin = offset[row] as usize;
            let end = offset[row + 1] as usize;
            let mut index = vec![0u32; end - begin];
            flat.index
                .read(&mut device, begin, &mut index)
                .expect("the column indices read back");
            assert_eq!(index, columns_of(&m, &mut device, &flat, row));
        }
    }

    #[test]
    fn the_transpose_reaches_every_off_diagonal_from_the_other_side() {
        // Both CSR matrices store the upper triangle only, so a symmetric
        // matvec needs the transpose index to pick up the lower half. A missed
        // entry is a silently asymmetric operator, which the PCG curvature
        // guard would then trip on for a reason nobody would look for here.
        let mut device = host_device();
        let mut m = DynCsrMat::new(3);
        let mut flat = Flat::default();
        build(
            &mut m,
            &mut device,
            &mut flat,
            &[(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0), (1, 2, 4.0)],
        );
        let mut t = Transpose::default();
        flat.transpose_into(&mut device, &mut t)
            .expect("the transpose dispatches");

        // Row 1 receives from (0, 1); row 2 from (0, 2) and (1, 2). The
        // diagonal is not transposed.
        let mut t_offset = vec![0u32; 4];
        t.offset
            .read(&mut device, 0, &mut t_offset)
            .expect("the transpose offsets read back");
        let mut t_index = vec![0u32; t.total];
        if t.total > 0 {
            t.index
                .read(&mut device, 0, &mut t_index)
                .expect("the transpose rows read back");
        }
        let entries = |row: usize| -> Vec<u32> {
            let begin = t_offset[row] as usize;
            let end = t_offset[row + 1] as usize;
            let mut v: Vec<u32> = t_index[begin..end].to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(entries(0), Vec::<u32>::new());
        assert_eq!(entries(1), vec![0]);
        assert_eq!(entries(2), vec![0, 1]);

        // And every transposed slot names a real stored block.
        let mut t_value = vec![0u32; t.total];
        if t.total > 0 {
            t.value
                .read(&mut device, 0, &mut t_value)
                .expect("the transposed slots read back");
        }
        for slot in &t_value {
            assert!((*slot as usize) < flat.nnz());
        }
    }

    #[test]
    fn the_transpose_carries_every_off_diagonal_arrival_exactly_once() {
        // THE GATE THE ELEMENT-WISE CONVERSION OF THIS PASS HAS TO CLEAR, and
        // it is deliberately written to say nothing about the order.
        //
        // Claiming each arrival's slot with a per-column `atomicAdd` over the
        // whole grid leaves which slot inside a column's run a given arrival
        // takes to whatever the interleaving hands out, so it can differ run to
        // run. What may NOT change is the SET:
        // every stored off-diagonal block appears exactly once in its column's
        // run, naming the row it came from and the slot it occupies in the flat
        // value array, and the diagonal appears nowhere.
        //
        // WIDE ON PURPOSE. A three-row case cannot tell a claim counter that
        // works from one that hands two arrivals the same slot: with one arrival
        // per column, every wrong assignment is still the right one. Here one
        // column receives from almost every row.
        const ROWS: u32 = 200;
        let mut device = host_device();
        let mut m = DynCsrMat::new(ROWS as usize);
        let mut flat = Flat::default();
        // UPPER TRIANGLE ONLY, which is what the assembly pushes: both CSR
        // matrices store one half and the transpose is how a symmetric matvec
        // reaches the other.
        let mut expected: Vec<Vec<u32>> = vec![Vec::new(); ROWS as usize];
        let mut pushes: Vec<(usize, u32, f32)> = Vec::new();
        for row in 0..ROWS {
            let mut columns = vec![row];
            for candidate in [row + 1, row + 7, ROWS - 1] {
                if candidate > row && candidate < ROWS && !columns.contains(&candidate) {
                    columns.push(candidate);
                }
            }
            for column in &columns {
                pushes.push((row as usize, *column, 1.0));
                if *column != row {
                    expected[*column as usize].push(row);
                }
            }
        }
        build(&mut m, &mut device, &mut flat, &pushes);

        let mut flat_offset = vec![0u32; ROWS as usize + 1];
        flat.offset
            .read(&mut device, 0, &mut flat_offset)
            .expect("the flat offsets read back");
        let mut slot_of: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for row in 0..ROWS as usize {
            for (position, column) in columns_of(&m, &mut device, &flat, row).iter().enumerate() {
                slot_of.insert((row as u32, *column), flat_offset[row] + position as u32);
            }
        }

        let mut t = Transpose::default();
        flat.transpose_into(&mut device, &mut t)
            .expect("the transpose dispatches");
        let mut t_offset = vec![0u32; ROWS as usize + 1];
        t.offset
            .read(&mut device, 0, &mut t_offset)
            .expect("the transpose offsets read back");
        let mut t_index = vec![0u32; t.total];
        let mut t_value = vec![0u32; t.total];
        t.index
            .read(&mut device, 0, &mut t_index)
            .expect("the transpose rows read back");
        t.value
            .read(&mut device, 0, &mut t_value)
            .expect("the transposed slots read back");

        let off_diagonal: usize = expected.iter().map(Vec::len).sum();
        assert_eq!(
            t.total, off_diagonal,
            "the transpose total is the off-diagonal stored block count"
        );
        for column in 0..ROWS as usize {
            let begin = t_offset[column] as usize;
            let end = t_offset[column + 1] as usize;
            let mut got: Vec<(u32, u32)> =
                (begin..end).map(|k| (t_index[k], t_value[k])).collect();
            got.sort_unstable();
            let mut want: Vec<(u32, u32)> = expected[column]
                .iter()
                .map(|row| (*row, slot_of[&(*row, column as u32)]))
                .collect();
            want.sort_unstable();
            assert_eq!(
                got, want,
                "column {column} does not receive exactly its off-diagonal \
                 arrivals: a repeated slot means two arrivals were handed one \
                 place, and a missing one means an arrival was never claimed"
            );
        }
    }

    #[test]
    fn the_matrix_does_not_depend_on_the_order_pairs_were_pushed_in() {
        // The COLUMNS and their sums, not their positions: compaction leaves a
        // row as two runs rather than one ascending sequence, deliberately.
        let pushes = [
            (0usize, 3u32, 1.0f32),
            (0, 1, 1.0),
            (1, 1, 1.0),
            (0, 3, 1.0),
            (2, 2, 1.0),
            (0, 1, 1.0),
        ];
        let mut device = host_device();
        let sorted_row = |m: &DynCsrMat, device: &mut _, flat: &Flat, row: usize| -> Vec<(u32, f32)> {
            let (columns, blocks) = m.read_row(device, flat, row).unwrap();
            let mut v: Vec<(u32, f32)> = columns
                .iter()
                .enumerate()
                .map(|(slot, c)| (*c, blocks[9 * slot]))
                .collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };

        let mut forward = DynCsrMat::new(4);
        let mut forward_flat = Flat::default();
        build(&mut forward, &mut device, &mut forward_flat, &pushes);

        let mut backward = DynCsrMat::new(4);
        let mut backward_flat = Flat::default();
        let reversed: Vec<(usize, u32, f32)> = pushes.iter().rev().copied().collect();
        build(&mut backward, &mut device, &mut backward_flat, &reversed);

        for row in 0..3usize {
            assert_eq!(
                sorted_row(&forward, &mut device, &forward_flat, row),
                sorted_row(&backward, &mut device, &backward_flat, row),
                "row {row} differs between the two push orders"
            );
        }
    }

    /// The passes must keep the scatter each one earns.
    ///
    /// NEITHER ATOMIC ROW MAY BECOME `Disjoint`, and they are not the same case.
    /// The COUNT accumulates into a shared slot, the column's arrival counter,
    /// which two rows naming one column both reach; on the host seam an atomic
    /// add is a plain read, add and write back, so a cut range would be a data
    /// race rather than a different fold order. The SCATTER and the FILL take
    /// NUMBERED slots out of a claim counter, which no range partition can serve
    /// at all. The SCAN is one lane by its own guard and carries a counter
    /// forward across the whole row range.
    ///
    /// A value test on this backend cannot carry either property, because this
    /// backend runs every non-`Disjoint` row as one ascending pass whatever the
    /// declaration says.
    #[test]
    fn the_index_builders_keep_the_scatter_that_forbids_a_cut_range() {
        use ppf_cts_compute::Scatter;
        use crate::driver::kernels::{id, TABLE};
        for (kernel, want) in [
            (id::DYN_COUNT_TRANSPOSE_PASS, Scatter::Atomic),
            (id::DYN_SCATTER_TRANSPOSE_PASS, Scatter::Claim),
            (id::DYN_DRY_PUSH_PASS, Scatter::Atomic),
            (id::DYN_PUSH_PASS, Scatter::Claim),
        ] {
            let row = TABLE[kernel.0 as usize];
            assert_eq!(
                row.scatter, want,
                "{} reaches a shared slot through an index it reads rather than \
                 through the thread index, so it may not be declared disjoint",
                row.name
            );
        }
        // And the three per-row passes ARE disjoint: row r reads and writes only
        // row r's slab, which is what lets a backend cut the range.
        for kernel in [
            id::DYN_ROW_BEGIN_PASS,
            id::DYN_ROW_SEED_PASS,
            id::DYN_ROW_COMPACT_PASS,
            id::DYN_ROW_EMIT_PASS,
        ] {
            let row = TABLE[kernel.0 as usize];
            assert_eq!(
                row.scatter,
                Scatter::Disjoint,
                "{} touches only its own row's slab",
                row.name
            );
        }
    }
}
