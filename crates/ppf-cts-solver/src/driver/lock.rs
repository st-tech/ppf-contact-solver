// File: lock.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The aggregate lock's projector.
//!
//! A transcription of `FullProjector` in
//! `src/kernels/solver/translation_lock.hpp`, dispatch for dispatch. That class
//! is host orchestration around eleven generated entry points; the entry points
//! are shared, the orchestration is not, so the Rust driver needs its own copy
//! for the same reason it has its own Newton loop.
//!
//! WHAT THE PROJECTOR IS. A locked group constrains the mass-weighted centroid
//! of its members to a fixed line, or their mass-weighted best-fit INCREMENTAL
//! rotation to or away from one axis, or both. Those are up to four linear rows
//! `C q = h` over the whole vertex vector. `prepare` builds the affine feasible
//! correction `q` that satisfies them, and `project` removes the constraint-space
//! component from any vector, which is what makes the CG search directions stay
//! in the tangent space.
//!
//! IT IS AN EXACT CONSTRAINT ON THE NEWTON DIRECTION, never a penalty: no energy
//! term, no Hessian block, no stiffness. SPD-by-assembly and the `pAp <= 0`
//! guards are untouched by it, which is why a locked scene runs the host-syncing
//! CG rather than the device-resident graph-captured one.
//!
//! THE PDRD ROWS ARE DELIBERATELY LEFT OUT. A group whose `pdrd_body_index` is
//! non-zero lives in the reduced body vector and is projected by PDRD's own
//! exact six-DOF projector; the two have disjoint supports and `builder.rs`
//! rejects every grouping that would break that.

use ppf_cts_compute::{AllocLabel, Buffer, Device, ReadbackBuffer, StagedBuffer};

use super::kernels::{
    LockCenterOfMassAccumulateRowArgs, LockConstraintAssembleRowArgs,
    LockFrameCenterOfMassRowArgs, LockFrameClearRowArgs, LockInertiaAccumulateRowArgs,
    LockProjectOutRowsRowArgs, LockRefineTowardRhsRowArgs, LockRowSumsAccumulateRowArgs,
    LockSeedFreeSolutionRowArgs, LockTorqueAccumulateRowArgs, TranslationLockDriftRowArgs,
    VecFillArgs,
};
use super::lock_math::{
    matvec6, pseudoinverse_gram, rotation_lock_enabled, rotation_mode_valid,
    translation_lock_enabled, translation_mode_valid, Mat6x6f, Vec6f, LOCK_ROTATION_ROW0,
    LOCK_ROTATION_ROW1, LOCK_ROTATION_ROW2, LOCK_TRANSLATION_ROW0, LOCK_TRANSLATION_ROW1,
    LOCK_TRANSLATION_ROW2, MAX_ROWS, ROTATION_ROW_BASE, ROTATION_ROW_COUNT,
};
use super::scene::{Fatal, FatalResult};
use crate::data::{
    LockFrame, Mat3x3f, TranslationLock, Vec3f, ROTATION_LOCK_ALL,
    ROTATION_LOCK_PROHIBIT_AXIS, TRANSLATION_LOCK_ALL,
};
// Named only where the tests below build a single-axis translation lock.
#[cfg(test)]
use crate::data::TRANSLATION_LOCK_AXIS;

/// The scratch the projector owns.
///
/// ALLOCATED ONCE AND REUSED. Every buffer here is sized by the group count or
/// the vertex count, both fixed for a run, and the projector is called several
/// times per Newton iteration; a per-call allocation would churn the arena and,
/// with no `Drop` on these types, leak the span rather than churning it.
#[derive(Default)]
pub struct Scratch {
    /// Per group: the constraint rows, their reduction and their right-hand
    /// side. Written by the frame passes on the device and read back on the
    /// host where the pseudoinverse and the feasibility verdict are formed.
    pub frames: ReadbackBuffer<LockFrame>,
    /// Per group, sixteen floats: the Gram matrix the assemble pass builds.
    pub gram: ReadbackBuffer<f32>,
    /// Per group, four floats: `C v`, the constraint-space image of whatever
    /// vector is being projected. FLOAT-typed rather than `Vec6f`-typed because
    /// the accumulating row builds it with float atomics and the two correcting
    /// rows read the same four components back.
    pub sums: Buffer<f32>,
    /// Per group, three floats: the centroid drift the assemble pass measures.
    pub drift: ReadbackBuffer<f32>,
    /// Per group, four floats: the exactly-prescribed contribution, which is
    /// its own buffer because a record addresses one allocation per field.
    pub fixed: ReadbackBuffer<f32>,
    /// Per group, three floats: the mass-weighted centroid sum, and nine: the
    /// inertia about it. Both are built by float atomics, so both are float.
    pub center_of_mass: Buffer<f32>,
    pub inertia: ReadbackBuffer<f32>,
    /// Per group, three floats: the best-fit angular increment the read-only
    /// tangent check probes for.
    pub torque: ReadbackBuffer<f32>,
    /// Per vertex: which rows are removed, seeded from the pin set.
    pub dof_mask: StagedBuffer<u32>,
}

impl Scratch {
    /// Size every buffer for this scene.
    pub fn allocate<D: Device>(
        &mut self,
        device: &mut D,
        groups: usize,
        vertices: usize,
    ) -> FatalResult<()> {
        self.frames.size(device, groups, AllocLabel("lock.frames"))?;
        self.gram.size(device, MAX_ROWS * MAX_ROWS * groups, AllocLabel("lock.gram"))?;
        self.sums.size(device, MAX_ROWS * groups, AllocLabel("lock.sums"))?;
        self.drift.size(device, 3 * groups, AllocLabel("lock.drift"))?;
        self.fixed.size(device, MAX_ROWS * groups, AllocLabel("lock.fixed"))?;
        self.center_of_mass
            .size(device, 3 * groups, AllocLabel("lock.com"))?;
        self.inertia
            .size(device, 9 * groups, AllocLabel("lock.inertia"))?;
        self.torque
            .size(device, 3 * groups, AllocLabel("lock.torque"))?;
        self.dof_mask
            .size(device, vertices, AllocLabel("lock.dof_mask"))?;
        Ok(())
    }
}

/// A deterministic orthonormal tangent basis for a normalized axis.
///
/// The reference's `tangent_basis`, spelled the same way including the 0.9
/// threshold that picks the reference direction: the point of that branch is
/// that the cross product below never degenerates, and the exact threshold is
/// what makes the basis reproducible rather than merely valid.
fn tangent_basis(axis: &Vec3f) -> (Vec3f, Vec3f) {
    let reference = if axis[2].abs() < 0.9 {
        Vec3f::new(0.0, 0.0, 1.0)
    } else {
        Vec3f::new(1.0, 0.0, 0.0)
    };
    let mut b0 = axis.cross(&reference);
    let n0 = b0.norm();
    if n0 > 0.0 {
        b0 /= n0;
    }
    let mut b1 = axis.cross(&b0);
    let n1 = b1.norm();
    if n1 > 0.0 {
        b1 /= n1;
    }
    (b0, b1)
}



/// Open a device accumulator at zero, ON THE DEVICE.
///
/// `Vec<T>::clear` in the reference (`vec/vec.hpp:142`) dispatches
/// `kernels::set`, so every accumulator `FullProjector` opens at zero
/// (`sums_`, `gram_`, `drift_`, `torque_`, the centroid and the inertia) is
/// cleared by a kernel and never by a transfer. Uploading a host buffer of
/// zeros instead moves that pass to the host, and it costs a host allocation
/// and a host-to-device copy at each of the sites below, one of which runs
/// several times per Newton iteration.
///
/// `vec_fill` is the entry point the rest of this driver already clears its
/// atomic accumulators with (`assemble.rs`, `collider.rs`, `schwarz.rs`), so
/// this reaches the same body the reference's `set` kernel is.
///
/// # Safety
/// `array` must name a live allocation of at least `count` floats.
unsafe fn clear_accumulator<D: Device>(
    device: &mut D,
    region: &'static str,
    array: ppf_cts_compute::Handle,
    count: usize,
) -> FatalResult<()> {
    if count == 0 {
        return Ok(());
    }
    let args = VecFillArgs {
        array,
        value: 0.0,
        count: count as u32,
        seam_arena_count: 0,
    };
    device.launch(region, &args, count as u32)?;
    Ok(())
}

/// Build every group's constraint rows from its authored axes.
///
/// HOST WORK, as it is in the reference: this is per GROUP and reads nothing but
/// the group's own record, so it is a loop over a handful of elements rather
/// than a dispatch. The two passes that follow it, the centroid and the inertia,
/// are per MEMBER and are dispatches.
fn build_row_bases(locks: &[TranslationLock]) -> FatalResult<Vec<LockFrame>> {
    let mut frames = Vec::with_capacity(locks.len());
    for lock in locks {
        // EVERY MEMBER ZEROED, not just the ones a current reader happens to
        // touch. `LockFrame::default()` already does that here, which is what
        // makes "a frame is fully zeroed before any conditional assignment" a
        // property the next unconditional read can rely on; the reference has
        // to spell each member out because its `SMat` default constructor is
        // empty and leaves stack garbage. It is not hypothetical for the basis
        // vectors: the tangent check recomputes all three rotation
        // coefficients before consulting the mask.
        let mut frame = LockFrame::default();
        if !translation_mode_valid(lock.translation_mode) {
            return Err(Fatal::invariant(format!(
                "solver driver: translation lock displacement group {} has invalid mode {}",
                lock.dmap_index, lock.translation_mode
            )));
        }
        if !rotation_mode_valid(lock.rotation_mode) {
            return Err(Fatal::invariant(format!(
                "solver driver: rotation lock group {} has invalid mode {}",
                lock.dmap_index, lock.rotation_mode
            )));
        }
        if translation_lock_enabled(lock) {
            if lock.translation_mode == TRANSLATION_LOCK_ALL {
                // Three rows from the identity basis, so the center of mass is
                // confined to a POINT rather than to a line.
                frame.translation_basis0 = Vec3f::new(1.0, 0.0, 0.0);
                frame.translation_basis1 = Vec3f::new(0.0, 1.0, 0.0);
                frame.translation_basis2 = Vec3f::new(0.0, 0.0, 1.0);
                frame.row_mask |=
                    LOCK_TRANSLATION_ROW0 | LOCK_TRANSLATION_ROW1 | LOCK_TRANSLATION_ROW2;
            } else {
                // Two rows perpendicular to the axis, so the component ALONG it
                // stays free and the center of mass is confined to a line.
                let (b0, b1) = tangent_basis(&lock.axis);
                frame.translation_basis0 = b0;
                frame.translation_basis1 = b1;
                frame.row_mask |= LOCK_TRANSLATION_ROW0 | LOCK_TRANSLATION_ROW1;
            }
        }
        // A PDRD group's rotation rows live in the reduced body vector and are
        // PDRD's to project, so this projector builds none for it.
        if lock.pdrd_body_index == 0 && rotation_lock_enabled(lock) {
            if lock.rotation_mode == ROTATION_LOCK_ALL {
                // Three rows: no net rotation about any axis. Taken through the
                // same inverse-inertia form as the per-axis rows rather than the
                // cheaper `m (e_k x r)`, which spans the same row space and so
                // gives the same projector, while keeping ONE formula in the
                // device path for the tangent check to be a copy of.
                frame.rotation_basis0 = Vec3f::new(1.0, 0.0, 0.0);
                frame.rotation_basis1 = Vec3f::new(0.0, 1.0, 0.0);
                frame.rotation_basis2 = Vec3f::new(0.0, 0.0, 1.0);
                frame.row_mask |= LOCK_ROTATION_ROW0 | LOCK_ROTATION_ROW1 | LOCK_ROTATION_ROW2;
            } else if lock.rotation_mode == ROTATION_LOCK_PROHIBIT_AXIS {
                // One row: the forbidden direction itself.
                frame.rotation_basis0 = lock.rotation_axis;
                frame.row_mask |= LOCK_ROTATION_ROW0;
            } else {
                // Two rows: everything ORTHOGONAL to the allowed direction.
                let (b0, b1) = tangent_basis(&lock.rotation_axis);
                frame.rotation_basis0 = b0;
                frame.rotation_basis1 = b1;
                frame.row_mask |= LOCK_ROTATION_ROW0 | LOCK_ROTATION_ROW1;
            }
        }
        frames.push(frame);
    }
    Ok(frames)
}

/// Read a group's 3x3 inertia out of the flat nine-float readback.
///
/// Column-major, which is the layout a `Mat3x3f` already has, so this is a
/// reinterpretation rather than a transpose.
fn inertia_of(flat: &[f32], group: usize) -> Mat3x3f {
    let base = 9 * group;
    let mut out = Mat3x3f::zeros();
    for column in 0..3 {
        for row in 0..3 {
            out[(row, column)] = flat[base + 3 * column + row];
        }
    }
    out
}

fn gram_of(flat: &[f32], group: usize) -> Mat6x6f {
    let base = MAX_ROWS * MAX_ROWS * group;
    let mut out = Mat6x6f::ZERO;
    out.0.copy_from_slice(&flat[base..base + MAX_ROWS * MAX_ROWS]);
    out
}

/// Everything the eleven rows need that does not change between them.
///
/// A struct rather than eleven repeated field lists, for the reason the
/// reference's `fill_row_inputs` template exists: writing the six lines out per
/// dispatch is six chances to disagree.
#[derive(Clone, Copy)]
pub struct RowInputs {
    pub lock_index: ppf_cts_compute::Handle,
    pub locks: ppf_cts_compute::Handle,
    pub prop: ppf_cts_compute::Handle,
    pub positions: ppf_cts_compute::Handle,
    pub dof_mask: ppf_cts_compute::Handle,
    /// The initial pose each group's drift is measured against, which only the
    /// constraint assembly reads.
    pub initial: ppf_cts_compute::Handle,
}

/// The projector.
///
/// `prepare` must run before any `project` in the same Newton iteration: the
/// frames it builds are a function of the ITERATE, and every projection scales
/// by them.
pub struct Projector<'a> {
    pub scratch: &'a mut Scratch,
    pub groups: usize,
    pub vertices: usize,
    pub rows: RowInputs,
}

impl<'a> Projector<'a> {
    /// Build every group's frame from the current positions.
    ///
    /// FOUR STEPS, in the reference's order and for its reasons: the row bases
    /// come from the authored axes alone and are host work; the centroid needs
    /// one pass over the members; the inertia is about that centroid, so it
    /// cannot start until the centroid is finished; and the inverse inertia is
    /// a per-group 3x3 Cholesky that the reference also does on the host, in
    /// double, because it carries a feasibility verdict that aborts.
    /// # Safety
    /// Every handle in `rows` must name a live allocation sized for the scene.
    unsafe fn initialize_frames<D: Device>(
        &mut self,
        device: &mut D,
        locks: &[TranslationLock],
    ) -> FatalResult<()> {
        let groups = self.groups;
        let frames = build_row_bases(locks)?;
        self.scratch.frames.seed(device, &frames)?;

        // CLEARED BEFORE EACH ACCUMULATION, both being built by float atomics
        // over the members, and cleared ON THE DEVICE as `com.clear()` and
        // `inertia.clear()` are. The readback below reads this pass rather than
        // the previous iteration's because `download` runs between the
        // accumulation and the first `host()`, which `handle()` enforces.
        clear_accumulator(
            device,
            "lock.com.fill",
            self.scratch.center_of_mass.handle(),
            3 * groups,
        )?;
        clear_accumulator(
            device,
            "lock.inertia.fill",
            self.scratch.inertia.handle(),
            9 * groups,
        )?;

        let com_args = LockCenterOfMassAccumulateRowArgs {
            lock_index: self.rows.lock_index,
            locks: self.rows.locks,
            prop: self.rows.prop,
            positions: self.rows.positions,
            mass_weighted_sum: self.scratch.center_of_mass.handle(),
            count: self.vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.center_of_mass", &com_args, self.vertices as u32)?;

        let divide_args = LockFrameCenterOfMassRowArgs {
            locks: self.rows.locks,
            mass_weighted_sum: self.scratch.center_of_mass.handle(),
            frames: self.scratch.frames.handle(),
            count: groups as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.frame_center_of_mass", &divide_args, groups as u32)?;

        let inertia_args = LockInertiaAccumulateRowArgs {
            lock_index: self.rows.lock_index,
            locks: self.rows.locks,
            prop: self.rows.prop,
            positions: self.rows.positions,
            frames: self.scratch.frames.handle(),
            inertia: self.scratch.inertia.handle(),
            count: self.vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.inertia", &inertia_args, self.vertices as u32)?;

        // THE INVERSE INERTIA IS HOST WORK AND MUST BE. It is one 3x3 Cholesky
        // per group, in double, and it ABORTS on a group whose extent is
        // singular or below what float32 can resolve: a lock is an exact
        // constraint on the Newton direction, so a merely large inverse would
        // turn it into a stiff spring nobody asked for.
        self.scratch.frames.download(device)?;
        self.scratch.inertia.download(device)?;
        let inertia_host = self.scratch.inertia.host().to_vec();
        let mut updated = self.scratch.frames.host().to_vec();
        for (group, lock) in locks.iter().enumerate() {
            if lock.pdrd_body_index == 0 && rotation_lock_enabled(lock) {
                updated[group].inv_inertia = super::lock_math::invert_inertia(
                    &inertia_of(&inertia_host, group),
                    lock.dmap_index,
                )?;
            }
        }
        self.scratch.frames.seed(device, &updated)?;
        Ok(())
    }

    /// Assemble the constraint rows and the reduction that inverts them.
    /// # Safety
    /// As `initialize_frames`, and `seed` must hold `3 * vertices` floats.
    unsafe fn assemble_constraints<D: Device>(
        &mut self,
        device: &mut D,
        locks: &[TranslationLock],
        seed: ppf_cts_compute::Handle,
    ) -> FatalResult<()> {
        let groups = self.groups;
        // The three accumulators the assemble pass fills by atomics, opened at
        // zero on the device as `gram_.clear()` and `drift_.clear()` are.
        // `fixed` is a field of the frame record in the reference and is zeroed
        // by the frame-clear row below; it is a buffer of its own here because a
        // record addresses one allocation per field, so it takes its own clear.
        clear_accumulator(device, "lock.gram.fill", self.scratch.gram.handle(),
                          MAX_ROWS * MAX_ROWS * groups)?;
        clear_accumulator(device, "lock.drift.fill", self.scratch.drift.handle(), 3 * groups)?;
        clear_accumulator(device, "lock.fixed.fill", self.scratch.fixed.handle(),
                          MAX_ROWS * groups)?;

        let clear_args = LockFrameClearRowArgs {
            frames: self.scratch.frames.handle(),
            count: groups as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.frame_clear", &clear_args, groups as u32)?;

        let assemble_args = LockConstraintAssembleRowArgs {
            lock_index: self.rows.lock_index,
            locks: self.rows.locks,
            prop: self.rows.prop,
            positions: self.rows.positions,
            initial: self.rows.initial,
            dof_mask: self.rows.dof_mask,
            frames: self.scratch.frames.handle(),
            seed,
            drift: self.scratch.drift.handle(),
            fixed: self.scratch.fixed.handle(),
            gram: self.scratch.gram.handle(),
            count: self.vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.constraint_assemble", &assemble_args, self.vertices as u32)?;

        // THE PSEUDOINVERSE AND THE FEASIBILITY VERDICT ARE HOST WORK, one 4x4
        // per group. The verdict is what makes a group whose exact pins leave a
        // residual outside the free tangent space stop the run rather than be
        // solved approximately.
        self.scratch.frames.download(device)?;
        self.scratch.gram.download(device)?;
        self.scratch.drift.download(device)?;
        self.scratch.fixed.download(device)?;
        let gram_host = self.scratch.gram.host().to_vec();
        let drift_host = self.scratch.drift.host().to_vec();
        let fixed_host = self.scratch.fixed.host().to_vec();
        const EPS: f32 = 1.192_092_9e-7;
        let mut updated = self.scratch.frames.host().to_vec();
        {
            for (group, lock) in locks.iter().enumerate() {
                let frame = &mut updated[group];
                let mut rhs = Vec6f::ZERO;
                for row in 0..MAX_ROWS {
                    rhs.0[row] = -fixed_host[MAX_ROWS * group + row];
                }
                // KEYED ON THE ROW MASK, not on the mode. The drift term belongs
                // to whichever translation rows the frame actually built, and
                // reading the mask is the one spelling that stays correct for
                // both modes without naming either.
                {
                    let drift = Vec3f::new(
                        drift_host[3 * group],
                        drift_host[3 * group + 1],
                        drift_host[3 * group + 2],
                    );
                    if frame.row_mask & LOCK_TRANSLATION_ROW0 != 0 {
                        rhs.0[0] += frame.translation_basis0.dot(&drift);
                    }
                    if frame.row_mask & LOCK_TRANSLATION_ROW1 != 0 {
                        rhs.0[1] += frame.translation_basis1.dot(&drift);
                    }
                    if frame.row_mask & LOCK_TRANSLATION_ROW2 != 0 {
                        rhs.0[2] += frame.translation_basis2.dot(&drift);
                    }
                }
                frame.rhs = rhs.0;
                if lock.pdrd_body_index != 0 {
                    continue;
                }
                let gram = gram_of(&gram_host, group);
                let pinv = pseudoinverse_gram(&gram)?;
                frame.gram_pinv = pinv.0;
                let lambda = matvec6(&pinv, &rhs);
                let resolved = matvec6(&gram, &lambda);
                let mut difference = Vec6f::ZERO;
                for row in 0..MAX_ROWS {
                    difference.0[row] = rhs.0[row] - resolved.0[row];
                }
                let residual = difference.norm();
                let scale = rhs.norm().max(1.0);
                let bound = 4096.0 * EPS * scale;
                if !residual.is_finite() || residual > bound {
                    return Err(Fatal::invariant(format!(
                        "solver driver: aggregate lock group {} is infeasible: exact fix pins \
                         leave a constraint residual {residual:.6e} outside the free tangent \
                         space (bound {bound:.6e})",
                        lock.dmap_index
                    )));
                }
            }
        }
        self.scratch.frames.seed(device, &updated)?;
        Ok(())
    }
}

impl<'a> Projector<'a> {
    /// Build the affine feasible Newton correction `q`.
    ///
    /// Exact pin increments are retained on removed rows; the free portion is
    /// the minimum-norm solution of `C_free q_free = h - C_fixed p`.
    ///
    /// TWO REFINEMENTS, and they are not belt-and-braces. The first float32
    /// application of the pseudoinverse leaves a visible residual whenever
    /// translation and rotation rows carry different dimensional scales;
    /// repeating the constraint-space correction is what drives
    /// `C_free q_free = rhs` down to the same projected round-off `project`
    /// itself reaches, and the count is the reference's.
    ///
    /// # Safety
    /// As `initialize_frames`, and both handles must hold `3 * vertices` floats.
    pub unsafe fn prepare<D: Device>(
        &mut self,
        device: &mut D,
        locks: &[TranslationLock],
        seed: ppf_cts_compute::Handle,
        values: ppf_cts_compute::Handle,
    ) -> FatalResult<()> {
        if self.groups == 0 {
            return Ok(());
        }
        self.initialize_frames(device, locks)?;
        self.assemble_constraints(device, locks, seed)?;

        let seed_args = LockSeedFreeSolutionRowArgs {
            lock_index: self.rows.lock_index,
            locks: self.rows.locks,
            prop: self.rows.prop,
            positions: self.rows.positions,
            dof_mask: self.rows.dof_mask,
            frames: self.scratch.frames.handle(),
            seed,
            values,
            count: self.vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.seed_free_solution", &seed_args, self.vertices as u32)?;

        for _refinement in 0..2 {
            self.accumulate_row_sums(device, values)?;
            let refine_args = LockRefineTowardRhsRowArgs {
                lock_index: self.rows.lock_index,
                locks: self.rows.locks,
                prop: self.rows.prop,
                positions: self.rows.positions,
                dof_mask: self.rows.dof_mask,
                frames: self.scratch.frames.handle(),
                sums: self.scratch.sums.handle(),
                values,
                count: self.vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("lock.refine_toward_rhs", &refine_args, self.vertices as u32)?;
        }
        Ok(())
    }

    /// `C v`, the constraint-space image of the vector being projected.
    ///
    /// CLEARED FIRST, because the row that fills it accumulates by atomics over
    /// the members and this runs several times per projection.
    ///
    /// # Safety
    /// As `prepare`.
    unsafe fn accumulate_row_sums<D: Device>(
        &mut self,
        device: &mut D,
        values: ppf_cts_compute::Handle,
    ) -> FatalResult<()> {
        clear_accumulator(
            device,
            "lock.sums.fill",
            self.scratch.sums.handle(),
            MAX_ROWS * self.groups,
        )?;
        let args = LockRowSumsAccumulateRowArgs {
            lock_index: self.rows.lock_index,
            locks: self.rows.locks,
            prop: self.rows.prop,
            positions: self.rows.positions,
            dof_mask: self.rows.dof_mask,
            frames: self.scratch.frames.handle(),
            values,
            sums: self.scratch.sums.handle(),
            count: self.vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.row_sums", &args, self.vertices as u32)?;
        Ok(())
    }

    /// Q: remove the constraint-space component from a full vector.
    ///
    /// REPEATED `refinements` TIMES for the reason `prepare` repeats its own
    /// correction: the float32 pseudoinverse is only an idempotent projector to
    /// within its own round-off, and combined translation and rotation rows with
    /// different scales make that visible in one pass. The caller chooses the
    /// count, and the reference's CG passes 1 everywhere except the three sites
    /// where it passes 3.
    ///
    /// PDRD ROWS ARE NOT TOUCHED. They live in the reduced body vector and are
    /// `launch_project_bodies`'s to remove.
    ///
    /// # Safety
    /// As `prepare`, and `values` must hold `3 * vertices` floats.
    pub unsafe fn project<D: Device>(
        &mut self,
        device: &mut D,
        values: ppf_cts_compute::Handle,
        refinements: u32,
    ) -> FatalResult<()> {
        if self.groups == 0 {
            return Ok(());
        }
        for _refinement in 0..refinements {
            self.accumulate_row_sums(device, values)?;
            let args = LockProjectOutRowsRowArgs {
                lock_index: self.rows.lock_index,
                locks: self.rows.locks,
                prop: self.rows.prop,
                positions: self.rows.positions,
                dof_mask: self.rows.dof_mask,
                frames: self.scratch.frames.handle(),
                sums: self.scratch.sums.handle(),
                values,
                count: self.vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("lock.project_out_rows", &args, self.vertices as u32)?;
        }
        Ok(())
    }
}

impl<'a> Projector<'a> {
    /// Verify, without changing it, that a solved correction carries no
    /// forbidden best-fit angular increment.
    ///
    /// READ-ONLY, AND DELIBERATELY SO. A rotation lock constrains an
    /// INCREMENTAL best-fit angular component, not an absolute pose, so there is
    /// no meaningful post-step snap; and a snap after the CCD line search would
    /// reintroduce penetration, which is why the solver never performs one.
    ///
    /// THE DEVICE PROBE ALONE DOES NOT ABORT. It is an fp32 reduction over the
    /// group's members, so a violation it reports can be its own round-off. The
    /// reference re-verifies the same rows on the host in DOUBLE and accepts the
    /// group when the verified magnitude is inside the reduction's own error
    /// bound, and this does the same: aborting on the probe would stop runs the
    /// reference completes.
    ///
    /// # Safety
    /// As `prepare`, and `step` must hold `3 * vertices` floats.
    pub unsafe fn check_tangent<D: Device>(
        &mut self,
        device: &mut D,
        locks: &[TranslationLock],
        step: ppf_cts_compute::Handle,
        positions_host: &[f32],
        step_host: &[f32],
        lock_index_host: &[u32],
        mass_host: &[f32],
        where_: &str,
    ) -> FatalResult<()> {
        if self.groups == 0 {
            return Ok(());
        }
        clear_accumulator(
            device,
            "lock.torque.fill",
            self.scratch.torque.handle(),
            3 * self.groups,
        )?;
        let args = LockTorqueAccumulateRowArgs {
            lock_index: self.rows.lock_index,
            locks: self.rows.locks,
            prop: self.rows.prop,
            positions: self.rows.positions,
            frames: self.scratch.frames.handle(),
            step,
            torque: self.scratch.torque.handle(),
            count: self.vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("lock.torque_accumulate", &args, self.vertices as u32)?;

        self.scratch.torque.download(device)?;
        self.scratch.frames.download(device)?;
        let torque_host = self.scratch.torque.host().to_vec();
        let frames_host = self.scratch.frames.host().to_vec();
        const EPS: f32 = 1.192_092_9e-7;

        for (group, lock) in locks.iter().enumerate() {
            if lock.pdrd_body_index != 0 || !rotation_lock_enabled(lock) {
                continue;
            }
            let frame = &frames_host[group];
            let torque = Vec3f::new(
                torque_host[3 * group],
                torque_host[3 * group + 1],
                torque_host[3 * group + 2],
            );
            let omega = frame.inv_inertia * torque;
            // The angular component the mode forbids. ALL-AXES IS ITS OWN
            // BRANCH rather than left to fall through: with a zero axis the
            // allow-only branch would give the same number by accident, and
            // that accident would break the moment the axis were initialized
            // differently.
            let magnitude = if lock.rotation_mode == ROTATION_LOCK_ALL {
                omega.norm()
            } else if lock.rotation_mode == ROTATION_LOCK_PROHIBIT_AXIS {
                lock.rotation_axis.dot(&omega).abs()
            } else {
                let along = omega.dot(&lock.rotation_axis);
                (omega - lock.rotation_axis * along).norm()
            };
            let omega_scale = omega.norm();
            let torque_scale = torque.norm();
            let inverse_scale = frame.inv_inertia[(0, 0)]
                .abs()
                .max(frame.inv_inertia[(1, 1)].abs())
                .max(frame.inv_inertia[(2, 2)].abs());
            let bound =
                4096.0 * EPS * (1.0f32).max(omega_scale + torque_scale * inverse_scale);
            if magnitude.is_finite() && magnitude <= bound {
                continue;
            }

            // THE HOST RE-VERIFICATION, in double, over the same rows. Its
            // `continue` is the whole point: the device probe is an fp32
            // reduction and this decides whether what it saw is real.
            let mut row_sum = [0.0f64; ROTATION_ROW_COUNT];
            let mut row_abs = [0.0f64; ROTATION_ROW_COUNT];
            let active = frame.row_mask >> ROTATION_ROW_BASE;
            let mut contributions = 0.0f64;
            // ALL THREE ARE FORMED BEFORE THE MASK IS CONSULTED, which is why
            // `build_row_bases` zeroes every basis rather than only the ones a
            // mode writes: an unwritten basis would put an inf or a NaN into a
            // quantity that is merely discarded rather than one never computed.
            let u0 = frame.inv_inertia * frame.rotation_basis0;
            let u1 = frame.inv_inertia * frame.rotation_basis1;
            let u2 = frame.inv_inertia * frame.rotation_basis2;
            for vertex in 0..self.vertices {
                if lock_index_host[vertex] != group as u32 {
                    continue;
                }
                let relative = Vec3f::new(
                    relative_component(positions_host[3 * vertex], lock.anchor[0]),
                    relative_component(positions_host[3 * vertex + 1], lock.anchor[1]),
                    relative_component(positions_host[3 * vertex + 2], lock.anchor[2]),
                ) - frame.com_relative;
                let mass = mass_host[vertex];
                let coefficient = [
                    u0.cross(&relative) * mass,
                    u1.cross(&relative) * mass,
                    u2.cross(&relative) * mass,
                ];
                for row in 0..ROTATION_ROW_COUNT {
                    if active & (1 << row) == 0 {
                        continue;
                    }
                    let mut contribution = 0.0f64;
                    for component in 0..3 {
                        contribution += coefficient[row][component] as f64
                            * step_host[3 * vertex + component] as f64;
                    }
                    row_sum[row] += contribution;
                    row_abs[row] += contribution.abs();
                }
                contributions += 1.0;
            }
            // Each row reduction is three products and their additions per
            // vertex; the projector and this check each incur one, so four
            // gamma_n bounds cover both passes plus the row coefficients.
            let operations = 4.0 * contributions + 32.0;
            let unit_roundoff = EPS as f64;
            let gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff);
            let mut magnitude_square = 0.0f64;
            let mut bound_square = 0.0f64;
            for row in 0..ROTATION_ROW_COUNT {
                if active & (1 << row) == 0 {
                    continue;
                }
                let row_bound = 4.0 * gamma * row_abs[row].abs().max(1.0);
                magnitude_square += row_sum[row] * row_sum[row];
                bound_square += row_bound * row_bound;
            }
            let verified = magnitude_square.sqrt();
            let verified_bound = bound_square.sqrt();
            if verified.is_finite() && verified <= verified_bound {
                continue;
            }
            return Err(Fatal::invariant(format!(
                "solver driver: rotation lock group {} violated at {where_}: host-verified \
                 forbidden constraint-row magnitude {verified:.6e} exceeds the fp32 \
                 reduction bound {verified_bound:.6e} (device probe {magnitude:.6e}). This \
                 is an invariant check only; the solver does not snap the state",
                lock.dmap_index
            )));
        }
        Ok(())
    }
}

/// One position component, relative to an anchor component.
///
/// Differencing against the anchor first is what keeps the small relative
/// quantity away from the cancellation that subtracting two large absolute
/// coordinates would produce.
fn relative_component(value: f32, anchor: f32) -> f32 {
    value - anchor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cvec::CVec;
    use crate::driver::launch::host_device;
    use crate::driver::state::SolverState;
    use crate::driver::test_scene::{position, TestScene};

    /// Four vertices, all in one group whose centroid is locked to the y axis.
    struct Locked {
        scene: TestScene,
        state: SolverState,
        device: ppf_cts_compute::HostDevice,
    }

    impl Locked {
        fn along_y(vertices: usize) -> Self {
            let mut scene = TestScene::new(vertices);
            for i in 0..vertices {
                scene.place(i, i as f32 * 0.25, 0.5, 0.0);
                scene.vertex_props_mut()[i].mass = 1.0;
            }
            scene.data.translation_lock = CVec::from(
                &[TranslationLock {
                    // THE AXIS IS THE LINE THE CENTROID MAY MOVE ALONG, so the
                    // two constraint rows are its tangent basis and the
                    // projector removes everything PERPENDICULAR to it.
                    axis: Vec3f::new(0.0, 1.0, 0.0),
                    translation_mode: TRANSLATION_LOCK_AXIS,
                    total_mass: vertices as f32,
                    pdrd_body_index: 0,
                    dmap_index: 3,
                    rotation_axis: Vec3f::new(0.0, 0.0, 0.0),
                    rotation_mode: 0,
                    anchor: position(0.0, 0.0, 0.0),
                }][..],
            );
            scene.data.translation_lock_index = CVec::from(&vec![0u32; vertices][..]);
            let initial: Vec<crate::data::Vec3f> =
                (0..vertices).map(|i| position(i as f32 * 0.25, 0.5, 0.0)).collect();
            scene.data.translation_lock_initial = CVec::from(&initial[..]);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("a locked scene allocates");
            Self { scene, state, device }
        }
    }

    /// Set every vertex to a pose and run the invariant check on it.
    ///
    /// The two tests below differ only in that pose, which is the whole point:
    /// the check must pass one and fail the other, or it is not measuring
    /// anything.
    fn judge(fixture: &mut Locked, pose: &[f32]) -> FatalResult<()> {
        let vertices = pose.len() / 3;
        let Locked { scene, state, device } = fixture;
        crate::driver::state::reseed_committed(device, state, &scene.data);
        crate::driver::state::reseed_props(device, state, &scene.data);
        let seeded_pose: Vec<f32> = pose.to_vec();
        // Safety: `seeded_pose` holds `vertices` triples for the whole call.
        unsafe {
            state
                .eval_x
                .seed(device, &seeded_pose)
                .expect("the iterate seeds");
        }
        let locks: Vec<TranslationLock> =
            unsafe { super::super::scene::slice(&scene.data.translation_lock) }.to_vec();
        let rows = RowInputs {
            lock_index: state.translation_lock_index.handle(),
            locks: state.translation_lock.handle(),
            prop: state.prop_vertex.handle(),
            positions: state.eval_x.handle(),
            dof_mask: state.dof_mask.handle(),
            initial: state.translation_lock_initial.handle(),
        };
        let seeded = state.eval_x.handle();
        // Safety: every handle above names a live allocation of this scene.
        unsafe {
            check_invariant(
                device,
                rows,
                seeded,
                &locks,
                &mut state.lock_drift,
                &mut state.lock_max_displacement,
                vertices,
                "test",
            )
        }
    }

    /// A pose that moved only ALONG the locked axis satisfies the invariant.
    ///
    /// The displacement here is large, 4 units of y against a bound on the
    /// order of 1e-4, so a check that reported any motion at all would fail
    /// this. What it measures is the PERPENDICULAR component, which is exactly
    /// zero for this pose.
    #[test]
    fn motion_along_the_locked_axis_satisfies_the_invariant() {
        let vertices = 4;
        let mut fixture = Locked::along_y(vertices);
        let pose: Vec<f32> = (0..vertices)
            .flat_map(|i| [i as f32 * 0.25, 4.5, 0.0])
            .collect();
        assert!(
            judge(&mut fixture, &pose).is_ok(),
            "sliding the whole group along its own axis is what the lock permits"
        );
    }

    /// THE LOCKED SOLVE IS REACHED END TO END.
    ///
    /// `pcg::solve_locked` had exactly one caller (`step.rs`) and no test:
    /// the two lock tests here cover the PROJECTOR, Metal refuses projected
    /// locks by name, and no scene in the acceptance sweep carries one. Without
    /// this test the whole host-syncing CG is production-only code, which is how
    /// a defect in it reaches production unseen.
    ///
    /// It is a prerequisite rather than a coverage tick. `solve_locked`
    /// downloads a FULL VECTOR and folds it on the host three times per
    /// iteration where the reference uses a device `inner_product`, and that
    /// cannot be converted safely while nothing runs the function.
    ///
    /// THE SYSTEM IS THE IDENTITY ON PURPOSE. With no off-diagonal blocks and
    /// an identity block diagonal, `A` is the identity, the preconditioner is
    /// exact, and the answer is the projected right-hand side, so the test
    /// asserts the SOLVE rather than a tolerance on some physical scene. What
    /// it exercises is the loop, the three folds, the projector's `prepare`
    /// and every scalar the recurrence forms.
    #[test]
    fn the_locked_solve_reaches_its_answer() {
        let vertices = 4usize;
        let mut fixture = Locked::along_y(vertices);
        let Locked { scene, state, device } = &mut fixture;
        crate::driver::state::reseed_committed(device, state, &scene.data);
        crate::driver::state::reseed_props(device, state, &scene.data);

        // No off-diagonal blocks, so both patterns are empty and every row is
        // its own diagonal.
        let empty_u: Vec<u32> = Vec::new();
        let offset: Vec<u32> = vec![0; vertices + 1];
        let empty_f: Vec<f32> = Vec::new();
        let mut identity = vec![0.0f32; 9 * vertices];
        for i in 0..vertices {
            for k in 0..3 {
                identity[9 * i + 4 * k] = 1.0;
            }
        }

        let mut index_d: ppf_cts_compute::Buffer<u32> = Default::default();
        let mut offset_d: ppf_cts_compute::Buffer<u32> = Default::default();
        let mut value_d: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut tp_d: ppf_cts_compute::Buffer<u32> = Default::default();
        let mut to_d: ppf_cts_compute::Buffer<u32> = Default::default();
        let mut diag_d: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut inv_d: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut b_d: ppf_cts_compute::Buffer<f32> = Default::default();
        let mut x_d: ppf_cts_compute::ReadbackBuffer<f32> = Default::default();
        for (buf, data, label) in [
            (&mut index_d, &empty_u, "t.index"),
            (&mut offset_d, &offset, "t.offset"),
            (&mut tp_d, &empty_u, "t.tpair"),
            (&mut to_d, &offset, "t.toffset"),
        ] {
            buf.size(device, data.len().max(1), ppf_cts_compute::AllocLabel(label))
                .expect("size");
            if !data.is_empty() {
                buf.write(device, 0, data).expect("upload");
            }
        }
        value_d
            .size(device, 1, ppf_cts_compute::AllocLabel("t.value"))
            .expect("size");
        let _ = &empty_f;
        for (buf, label) in [
            (&mut diag_d, "t.diagonal"),
            (&mut inv_d, "t.inverse"),
        ] {
            buf.size(device, identity.len(), ppf_cts_compute::AllocLabel(label))
                .expect("size");
            buf.write(device, 0, &identity).expect("upload");
        }
        // A right-hand side that is NOT along the locked axis, so the projector
        // has something to remove and the answer is not the input.
        let force: Vec<f32> = (0..vertices)
            .flat_map(|i| [1.0 + i as f32 * 0.1, -0.5, 0.25])
            .collect();
        b_d.size(device, force.len(), ppf_cts_compute::AllocLabel("t.b"))
            .expect("size");
        b_d.write(device, 0, &force).expect("upload");
        x_d.size(device, force.len(), ppf_cts_compute::AllocLabel("t.x"))
            .expect("size");

        let op = super::super::operator::Operator {
            dynamic: None,
            fixed: super::super::spmv::FixedCsrView {
                index: index_d.handle(),
                offset: offset_d.span(0, vertices + 1),
                value: value_d.handle(),
                transpose_pair: tp_d.handle(),
                transpose_offset: to_d.span(0, vertices + 1),
                rows: vertices as u32,
            },
            diagonal: diag_d.handle(),
        };

        let rows = RowInputs {
            lock_index: state.translation_lock_index.handle(),
            locks: state.translation_lock.handle(),
            prop: state.prop_vertex.handle(),
            positions: state.eval_x.handle(),
            dof_mask: state.dof_mask.handle(),
            initial: state.translation_lock_initial.handle(),
        };
        let mut projector = Projector {
            scratch: &mut state.lock,
            groups: 1,
            vertices,
            rows,
        };

        // `prepare` MUST RUN FIRST, as `step.rs` runs it: the frames it builds
        // are a function of the iterate, and the affine feasible correction `q`
        // it seeds is what the recurrence solves on top of. Omitting it leaves
        // the projector with no frames, and the solve then returns the
        // right-hand side unchanged, which is what the assertion below caught.
        let locked_records: Vec<TranslationLock> =
            unsafe { super::super::scene::slice(&scene.data.translation_lock) }.to_vec();

        let mut work = super::super::pcg::Workspace::default();
        work.size_for(device, vertices as u32).expect("workspace");
        let mut locked = super::super::pcg::LockedWorkspace::default();
        locked
            .allocate(device, vertices as u32)
            .expect("locked workspace");

        // Safety: the records outlive the call and every handle names a live
        // allocation of this fixture.
        unsafe {
            projector.prepare(device, &locked_records, x_d.handle(), locked.q.handle())
        }
        .expect("the projector prepares");

        // Safety: every handle above names a live allocation of this fixture,
        // and all of them outlive the call.
        let report = unsafe {
            super::super::pcg::solve_locked(
                device,
                &op,
                &mut projector,
                inv_d.handle(),
                b_d.handle(),
                x_d.handle(),
                &mut work,
                &mut locked,
                None,
                1e-4,
                64,
            )
        }
        .expect("the locked solve runs");

        // THE ITERATION COUNT IS THE SHARP ASSERTION, and it is why this
        // system is the identity. With `A = I` and an exact block-Jacobi
        // preconditioner the preconditioned residual is already the answer, so
        // correct PCG converges in ONE iteration. Every scalar the recurrence
        // forms has to be right for that to happen: a wrong `pAp` gives a wrong
        // step length and the residual does not vanish, so the loop runs on.
        // A weaker check (finite, non-zero, not the input) passes with the
        // folds scaled by a constant, which was measured by injecting exactly
        // that.
        assert!(
            report.iterations >= 1,
            "the solve must take at least one iteration, not short-circuit"
        );
        assert!(
            report.iterations <= 2,
            "an exactly preconditioned identity system must converge in one \
             iteration; {} means a scalar the recurrence formed is wrong",
            report.iterations
        );
        assert!(
            report.relative_residual <= 1e-4,
            "the solve reported convergence it did not reach: {}",
            report.relative_residual
        );
        assert!(
            report.relative_residual.is_finite(),
            "a non-finite residual means a fold read garbage: {}",
            report.relative_residual
        );

        // THE ANSWER ITSELF, which is what makes this more than a smoke test.
        // `A` is the identity, so the correction is the projected right-hand
        // side: it must be non-zero, finite, and it must NOT be the raw force,
        // because the projector removes the component the lock forbids.
        x_d.download(device).expect("the answer reads back");
        let x = x_d.host();
        assert_eq!(x.len(), force.len());
        let mut moved = 0usize;
        for (k, value) in x.iter().enumerate() {
            assert!(
                value.is_finite(),
                "component {k} is not finite: a fold that read a stale or \
                 partial buffer reaches the recurrence as a NaN"
            );
            if *value != 0.0 {
                moved += 1;
            }
        }
        assert!(
            moved > 0,
            "the solve returned an all-zero correction, so nothing was solved"
        );
        // The perpendicular part of the force is what the lock removes, so the
        // answer cannot equal the input on every component.
        let identical = x
            .iter()
            .zip(force.iter())
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        assert!(
            identical < force.len(),
            "the answer equals the right-hand side on every component, so the \
             projector was not applied"
        );
    }

    /// A pose whose centroid moved ACROSS the locked axis is reported.
    ///
    /// THIS IS THE TEST THAT PROVES THE CHECK IS REACHED. A verification that is
    /// compiled, wired and dispatched from nowhere passes every scene, which is
    /// indistinguishable from a lock that holds; only a case that must FAIL
    /// tells the two apart. The drift below is 0.5 in x against a round-off
    /// bound of about 1e-4, so it clears the bound by three orders and does not
    /// depend on the accumulation order.
    #[test]
    fn perpendicular_drift_of_the_centroid_is_reported() {
        let vertices = 4;
        let mut fixture = Locked::along_y(vertices);
        let pose: Vec<f32> = (0..vertices)
            .flat_map(|i| [i as f32 * 0.25 + 0.5, 0.5, 0.0])
            .collect();
        let verdict = judge(&mut fixture, &pose);
        let fatal = verdict.expect_err("a centroid moved across its axis violates the lock");
        assert!(
            fatal.detail.contains("translation lock group 3 violated"),
            "the report names the group by its dmap index: {}",
            fatal.detail
        );
        assert!(
            fatal.detail.contains("never snaps"),
            "the report says the solver does not repair the pose: {}",
            fatal.detail
        );
    }

    /// After projection the group's mass-weighted centroid moves ONLY along its
    /// axis.
    ///
    /// THIS IS WHAT THE PROJECTOR IS FOR, and it is the property a half-wired
    /// lock fails: preparing the affine correction and never projecting the
    /// search directions leaves a solve that looks like it locks and does not.
    /// The vector below is deliberately dominated by its perpendicular
    /// components, so a projector that did nothing would fail by a wide margin.
    #[test]
    fn a_projected_vector_moves_the_centroid_only_along_the_axis() {
        let vertices = 4;
        let mut fixture = Locked::along_y(vertices);
        let Locked { scene, state, device } = &mut fixture;

        crate::driver::state::reseed_committed(device, state, &scene.data);
        crate::driver::state::reseed_props(device, state, &scene.data);
        // Safety: the scene is live and holds `vertices` triples.
        unsafe {
            let pose = crate::driver::state::slice_or_empty(
                scene.data.vertex.curr.data as *const f32,
                3 * vertices,
            );
            state.eval_x.seed(device, pose).expect("the iterate seeds");
        }
        // THE MASK IS KERNEL-WRITTEN NOW, so a fixture that wants it all-zero
        // seeds the device rather than filling a host mirror there is no
        // longer. `seed` is the both-written shape's host half.
        let zeros = vec![0u32; vertices];
        state
            .dof_mask
            .write(device, 0, &zeros)
            .expect("the mask seeds");

        // An arbitrary direction, mostly perpendicular to the locked axis.
        let mut values: ppf_cts_compute::Buffer<f32> = Default::default();
        values
            .size(device, 3 * vertices, AllocLabel("test.lock.values"))
            .expect("the vector allocates");
        let raw: Vec<f32> = (0..vertices)
            .flat_map(|i| [1.0 + i as f32, 0.125, -2.0 - i as f32])
            .collect();
        values.write(device, 0, &raw).expect("the vector uploads");

        let locks: Vec<TranslationLock> =
            unsafe { super::super::scene::slice(&scene.data.translation_lock) }.to_vec();
        let rows = RowInputs {
            lock_index: state.translation_lock_index.handle(),
            locks: state.translation_lock.handle(),
            prop: state.prop_vertex.handle(),
            positions: state.eval_x.handle(),
            dof_mask: state.dof_mask.handle(),
            initial: state.translation_lock_initial.handle(),
        };
        let mut projector = Projector {
            scratch: &mut state.lock,
            groups: 1,
            vertices,
            rows,
        };
        // PREPARE WRITES ITS OWN BUFFER, not `values`. Handing it the same
        // allocation for both would leave `values` holding the affine feasible
        // correction, which is already on the axis, and the projection below
        // would then be a no-op that the assertion could not tell from a working
        // one.
        let mut correction: ppf_cts_compute::Buffer<f32> = Default::default();
        correction
            .size(device, 3 * vertices, AllocLabel("test.lock.q"))
            .expect("the correction allocates");
        // Safety: every handle names a live allocation sized for the scene.
        unsafe {
            projector
                .prepare(device, &locks, values.handle(), correction.handle())
                .expect("the projector prepares");
            projector
                .project(device, values.handle(), 1)
                .expect("the projector runs");
        }

        let mut projected = vec![0.0f32; 3 * vertices];
        values
            .read(device, 0, &mut projected)
            .expect("the projected vector reads back");

        // The mass-weighted centroid displacement, which every mass being 1
        // makes the plain mean.
        let mut centroid = [0.0f32; 3];
        for vertex in 0..vertices {
            for k in 0..3 {
                centroid[k] += projected[3 * vertex + k];
            }
        }
        for value in centroid.iter_mut() {
            *value /= vertices as f32;
        }
        let scale = projected.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        assert!(
            centroid[0].abs() < 1.0e-4 * scale && centroid[2].abs() < 1.0e-4 * scale,
            "the centroid moved off its axis by ({}, {}), against a per-component \
             scale of {scale}: the locked axis is +y, so x and z must vanish",
            centroid[0],
            centroid[2]
        );
    }
}

// ---------------------------------------------------------------------------
// THE FEASIBILITY CHECK A HINGED, LOCKED BODY OWES.

/// Refuse a PDRD body that carries BOTH a hinge and a translation lock whose
/// requested correction is real.
///
/// A HINGE PERMITS NO TRANSLATION AT ALL. Its joint projector is
/// `blockdiag(0, a a^T)`, so the three translation DOFs are gone from the
/// reduced system. A translation lock on the same body asks for a perpendicular
/// centroid correction, and there is no DOF left to deliver it: the request is
/// genuinely infeasible rather than something to drop by composing the two
/// projectors, which is what silently zeroing it would amount to.
///
/// THIS IS THE HOST CHECK THE DEVICE SIDE ALREADY ASSUMES.
/// `pdrd_translation_lock_particular_row` returns early for a hinged body and
/// says in its own comment that the compatibility "is checked on the host
/// before this runs". That sentence is a claim about this function existing;
/// without it the kernel's early return is not a safe skip of an impossible
/// case but a silent discard of a correction the scene asked for.
/// `solver.cu:1731-1770` is the same check.
///
/// The bound is an absolute floor plus the fp32 accumulation floor, and it
/// carries NO displacement term, unlike `check_invariant`: this quantity is a
/// requested correction rather than a measured drift over a moved pose.
///
/// # Safety
/// `drift` must name a live allocation of three floats per group, and `jmode`
/// and `tlock` must be the reduction's own per-body arrays.
pub unsafe fn check_hinge_lock_feasible<D: Device>(
    device: &mut D,
    drift: &mut ReadbackBuffer<f32>,
    locks: &[TranslationLock],
    jmode: &[u32],
    tlock: &[u32],
) -> FatalResult<()> {
    if locks.is_empty() {
        return Ok(());
    }
    let hinged_and_locked = jmode
        .iter()
        .zip(tlock.iter())
        .any(|(mode, lock)| {
            *mode == super::rigid_map::PDRD_JOINT_HINGE
                && *lock != super::rigid_map::RIGID_UNSET
        });
    if !hinged_and_locked {
        return Ok(());
    }
    drift.download(device)?;
    let values = drift.host();
    const FP32_EPS: f32 = 1.192_092_9e-7;
    // An absolute floor of half an epsilon, so a group whose coordinates are
    // near zero still gets a bound the accumulated round-off can fit inside,
    // plus the relative term the sum's own width contributes.
    let bound = 0.5 * FP32_EPS + 256.0 * FP32_EPS;
    for (body, (mode, lock)) in jmode.iter().zip(tlock.iter()).enumerate() {
        if *mode != super::rigid_map::PDRD_JOINT_HINGE
            || *lock == super::rigid_map::RIGID_UNSET
        {
            continue;
        }
        let li = *lock as usize;
        if li >= locks.len() || 3 * li + 2 >= values.len() {
            continue;
        }
        let d = [values[3 * li], values[3 * li + 1], values[3 * li + 2]];
        let residual =
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / locks[li].total_mass;
        if residual > bound {
            return Err(Fatal::invariant(format!(
                "PDRD body {} has both a hinge and a translation lock, but its \
                 current COM requires a constrained translation of {:.6e}. The \
                 hinge removes that degree of freedom, so the requested lock \
                 correction is infeasible.",
                body + 1,
                residual
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// THE READ-ONLY VERIFICATION.

/// Verify the absolute translation invariant on a pose, without modifying it.
///
/// A transcription of `translation_lock::check_invariant`
/// (`src/kernels/solver/translation_lock.hpp:983`), which the reference runs at
/// two points: on the seeded pose at `main.cu:155`, and on the pose about to be
/// committed at `main.cu:1627`.
///
/// IT IS AN INVARIANT CHECK, NEVER A CORRECTIVE SNAP. Nothing here writes a
/// position. A snap applied after the CCD line search would move a vertex the
/// line search had already certified, which is how a penetration is
/// reintroduced, so a group that has drifted off its commanded line is
/// REPORTED rather than repaired.
///
/// THE BOUND IS ROUND-OFF, NOT A TOLERANCE. The first term is an absolute
/// floor, so a group sitting near the origin is not held to a purely relative
/// tolerance, and `256 * eps * max(1, max_displacement)` is the fp32
/// accumulation error over a group's reduction, which scales with the largest
/// displacement anywhere in that group. A genuine constraint failure clears
/// this by orders of magnitude; it is not a knob to widen when a scene trips
/// it.
///
/// Only the translation half has an absolute-pose invariant. Rotation has none,
/// so the solved tangent direction is what gets verified there, by
/// `check_tangent` before the line search applies it.
///
/// # Safety
/// `rows` must name live device allocations, `positions` a pose of `vertices`
/// coordinate triples, and `locks_host` the same group records the device copy
/// holds.
#[allow(clippy::too_many_arguments)]
pub unsafe fn check_invariant<D: Device>(
    device: &mut D,
    rows: RowInputs,
    positions: ppf_cts_compute::Handle,
    locks_host: &[TranslationLock],
    drift: &mut ReadbackBuffer<f32>,
    max_displacement: &mut ReadbackBuffer<u32>,
    vertices: usize,
    site: &str,
) -> FatalResult<()> {
    let groups = locks_host.len();
    if groups == 0 || vertices == 0 {
        return Ok(());
    }
    // `size` fills with zeros on every call, which is the clear both reductions
    // need: the sum starts at zero and the maximum over the bit patterns of
    // non-negative floats starts at zero too, that being the pattern of +0.0.
    drift.size(device, 3 * groups, AllocLabel("lock.drift"))?;
    max_displacement.size(device, groups, AllocLabel("lock.maxdisp"))?;

    let args = TranslationLockDriftRowArgs {
        lock_index: rows.lock_index,
        locks: rows.locks,
        positions,
        initial: rows.initial,
        prop: rows.prop,
        drift: drift.handle(),
        max_displacement: max_displacement.handle(),
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("lock.drift", &args, vertices as u32)?;

    drift.download(device)?;
    max_displacement.download(device)?;
    let sums = &drift.host()[..3 * groups];
    let maxima = &max_displacement.host()[..groups];

    // The absolute floor and the fp32 epsilon. Both bounds in this file spell
    // them the same way so the two are the same number.
    const FP32_EPS: f32 = 1.192_092_9e-7;
    for (li, lock) in locks_host.iter().enumerate() {
        if !translation_lock_enabled(lock) {
            continue;
        }
        let sum = [sums[3 * li], sums[3 * li + 1], sums[3 * li + 2]];
        let residual =
            (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt() / lock.total_mass;
        // The maximum arrives as the bit pattern of a non-negative float, which
        // is what `compute::atomic_max` stores because no backend has a float
        // atomic maximum. Converting back here is what keeps a bitcast out of
        // every neutral body.
        let largest = f32::from_bits(maxima[li]);
        let bound = 0.5 * FP32_EPS + 256.0 * FP32_EPS * largest.max(1.0);
        if !residual.is_finite() || residual > bound {
            return Err(Fatal::invariant(format!(
                "translation lock group {} violated at {}: constrained COM drift \
                 {:.6e} exceeds the fp32 round-off bound {:.6e}. The solver never \
                 snaps this state; inspect the constrained Newton direction.",
                lock.dmap_index, site, residual, bound
            )));
        }
    }
    Ok(())
}
