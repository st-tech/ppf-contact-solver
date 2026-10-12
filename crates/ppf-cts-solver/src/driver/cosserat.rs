// File: cosserat.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Cosserat rods: the frame state and the Newton step's dispatches over it.
//!
//! A Cosserat rod's edge carries a material frame beside its two node positions.
//! The frames are built on the host (`crate::cosserat`) and handed over in the
//! `DataSet`; this module owns their device copies for the run and launches the
//! rows of `kernels/energy/model/cosserat.kernel.cpp`. The augmented solve that
//! consumes the assembled blocks is `pcg::solve_cosserat`.
//!
//! WHERE EACH CALL SITS IN THE STEP is decided in `step.rs` and stated there;
//! the order this module needs is: [`State::restore`] at the start of a step,
//! [`State::assemble`] after the rod stretch, [`State::reduce_dirichlet`] after
//! the matrix's own Dirichlet pass, the solve, [`State::advance`] at the line
//! search's accepted fraction, and [`State::commit`] with the positions.
//!
//! THE FRAME HISTORY IS TWO DEEP, `prev` and `prev2`, because the rotational
//! inertia predicts a frame from the last step's incremental rotation, which is
//! the rotational twin of the position predictor reading `prev` and `curr`.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Handle, StagedBuffer};

use super::kernels::{
    CosseratBendRowArgs, CosseratDirichletRowArgs, CosseratDriveRowArgs,
    CosseratInertiaRowArgs, CosseratPrescribeRowArgs, CosseratShearRowArgs,
    CosseratStepRowArgs, DxMagnitudeArgs,
};
use super::scene::{Fatal, FatalResult};
use super::state::SolverState;
use crate::data::DataSet;

/// The largest turn one Newton iteration may give a frame, in radians.
///
/// THE ROTATIONAL TWIN OF `max-dx`, and it rescales the WHOLE step rather than
/// the frames alone, as `max-dx` does: positions and frames are one Newton
/// direction, and moving them by different fractions would take a step the
/// solve did not compute.
///
/// A QUARTER TURN IS NOT A TUNING. The frame step is solved in the tangent of a
/// left perturbation, `R <- exp(skew(theta)) R`, against a quadratic model built
/// at `theta = 0`. The rotational inertia, the one term every frame carries, is
/// convex in that chart exactly while the frame stays within a quarter turn of
/// its prediction (its Hessian `tr(M) I - sym(M)` loses definiteness there), so
/// past it the model is a projected one and its minimizer is not a step the
/// energy vouches for. The bound is a property of the parametrization; it binds
/// only a step that would otherwise turn a frame further than any physical
/// substep does.
pub const MAX_FRAME_TURN: f32 = std::f32::consts::FRAC_PI_2;

/// The device copies of the scene's Cosserat frames and everything the Newton
/// step assembles over them.
///
/// SIZED ONCE, at `initialize()`, and reused by every step: nothing here
/// allocates inside a Newton iteration.
#[derive(Default)]
pub struct State {
    /// Cosserat edges, which is the frame count. Zero on every scene without a
    /// Cosserat rod, and every call below returns at once on it.
    pub slots: usize,
    /// Bend and twist sites, one per pair of consecutive frames.
    pub sites: usize,
    /// Whether `cosserat-fix-ends` holds any frame, and whether any held frame
    /// is driven. Both are properties of the scene, so they are read once.
    held_static: bool,
    driven: bool,

    // THE TOPOLOGY, staged once. Flat words and floats rather than vector
    // records, which is how the mesh's own edge array is staged.
    /// Per slot, the rod edge it rides on.
    slot_edge: StagedBuffer<u32>,
    /// Per slot, the edge's two vertices in the rod's chain order.
    edge_nodes: StagedBuffer<u32>,
    /// Per slot, whether the frame is held by `cosserat-fix-ends`.
    fixed: StagedBuffer<u32>,
    /// Per slot, a held frame's turn rate about its own tangent, rad/s.
    end_omega: StagedBuffer<f32>,
    /// Per site, the previous and next slot.
    bend_slots: StagedBuffer<u32>,
    /// Per site, the signed rest Darboux vector in the material frame.
    rest_darboux: StagedBuffer<f32>,
    /// Per site, the inverse of the Voronoi length the curvature is taken over.
    inv_len: StagedBuffer<f32>,
    /// Per site, the rest metric's two factors (`crate::cosserat::rest_metric`).
    rest_metric: StagedBuffer<f32>,
    /// The chains, as the connected components of the bend sites: a chain's
    /// slots are `chain_slot[chain_offset[c]..chain_offset[c + 1]]`, ascending.
    /// They are the coarse space of the frame preconditioner (`pcg::solve_cosserat`).
    pub chains: usize,
    chain_offset: StagedBuffer<u32>,
    chain_slot: StagedBuffer<u32>,
    /// Three floats per slot: each free frame's own third axis, written at the
    /// start of every solve, the chain roll the coarse correction adds back.
    roll: Buffer<f32>,
    /// Per vertex, 1 when a Cosserat edge covers it: the preconditioner's edge
    /// blocks reach it, and its block-Jacobi row is not taken.
    covered: StagedBuffer<u32>,
    /// Per slot, the Cholesky factor of its edge block, 45 floats, formed once
    /// per solve.
    factor: Buffer<f32>,

    // THE FRAME STATE, four floats per slot each, a unit quaternion (x, y, z, w).
    /// The iterate the Newton loop moves.
    quat: Buffer<f32>,
    /// The last committed frame, and the one before it.
    quat_prev: Buffer<f32>,
    quat_prev2: Buffer<f32>,

    // WHAT ONE NEWTON ITERATION ASSEMBLES, cleared before the shear pass.
    /// Three floats per slot: the energy's gradient in the frame's tangent.
    gradient: Buffer<f32>,
    /// Nine floats per slot, column major: the frame's own 3x3 block.
    self_block: Buffer<f32>,
    /// Nine floats per slot each: the position rows of the slot's first and
    /// second vertex against the frame's columns.
    coupling_from: Buffer<f32>,
    coupling_to: Buffer<f32>,
    /// Nine floats per site: the next frame's rows against the previous frame's
    /// columns.
    site_block: Buffer<f32>,

    /// Per slot, this step's held mask: `fixed`, or both of the edge's
    /// vertices prescribed (a clamp). Written by [`State::prescribe`].
    held: Buffer<u32>,
    /// Three floats per slot: the augmented solve's seed on the frame rows,
    /// the turn a clamped edge's frame takes with it and zero elsewhere.
    prescribed: Buffer<f32>,
    /// Three floats per slot: the frame half of the solved Newton direction,
    /// copied out of the augmented solve for [`State::advance`].
    pub step: Buffer<f32>,
    /// One float per slot: the turn `step` asks of each frame, for the cap.
    turn: Buffer<f32>,
}

/// Stage a flat array of the scene's, `per` words per element, checking its
/// length against the element count it is declared over.
///
/// # Safety
/// `source` must name `count * per` live elements.
unsafe fn stage<D: Device, T: ppf_cts_compute::Pod + Default>(
    device: &mut D,
    buffer: &mut StagedBuffer<T>,
    source: *const T,
    count: usize,
    label: &'static str,
) -> FatalResult<()> {
    // ONE ELEMENT AT LEAST, so an empty array still names a real allocation:
    // a generated entry resolves every buffer before its body runs, and an
    // absent one takes a real zero-length handle rather than `Handle::NONE`.
    buffer.size(device, count.max(1), AllocLabel(label))?;
    if count > 0 {
        buffer.at()[..count].copy_from_slice(std::slice::from_raw_parts(source, count));
    }
    buffer.upload(device)?;
    Ok(())
}

impl State {
    /// Stage the scene's frames and size every per-iteration buffer.
    ///
    /// THE SCENE'S ARRAYS ARE CHECKED AGAINST ONE ANOTHER BEFORE ANY IS STAGED.
    /// They come from the host frame builder and agree by construction; a
    /// disagreement means a scene that did not come from it, and indexing a
    /// device array by a slot it does not hold is a silent read on Metal, which
    /// never faults.
    ///
    /// # Safety
    /// The scene's arrays must be live for the call.
    pub unsafe fn allocate<D: Device>(
        &mut self,
        device: &mut D,
        data: &DataSet,
        vertices: usize,
        rods: usize,
    ) -> FatalResult<()> {
        let slots = data.cosserat_edge.size as usize;
        let sites = data.cosserat_bend_slots.size as usize;
        self.slots = slots;
        self.sites = sites;
        if slots == 0 {
            if sites != 0 {
                return Err(Fatal::invariant(format!(
                    "solver driver: the scene carries {sites} Cosserat bend sites and no \
                     Cosserat edge, so the sites name frames that do not exist"
                )));
            }
            return Ok(());
        }
        let expect = |name: &str, got: usize, want: usize| -> FatalResult<()> {
            if got == want {
                Ok(())
            } else {
                Err(Fatal::invariant(format!(
                    "solver driver: the scene's Cosserat array `{name}` holds {got} entries \
                     where its {slots} frames and {sites} sites need {want}"
                )))
            }
        };
        expect("cosserat_edge_nodes", data.cosserat_edge_nodes.size as usize, slots)?;
        expect("cosserat_quat", data.cosserat_quat.size as usize, 4 * slots)?;
        expect("cosserat_fixed", data.cosserat_fixed.size as usize, slots)?;
        expect("cosserat_end_omega", data.cosserat_end_omega.size as usize, slots)?;
        expect("cosserat_rest_darboux", data.cosserat_rest_darboux.size as usize, sites)?;
        expect("cosserat_bend_invlen", data.cosserat_bend_invlen.size as usize, sites)?;
        expect("cosserat_rest_metric", data.cosserat_rest_metric.size as usize, 2 * sites)?;

        let edge: &[u32] = super::scene::slice(&data.cosserat_edge);
        let nodes: &[crate::data::Vec2u] = super::scene::slice(&data.cosserat_edge_nodes);
        let bend: &[crate::data::Vec2u] = super::scene::slice(&data.cosserat_bend_slots);
        let inv_len: &[f32] = super::scene::slice(&data.cosserat_bend_invlen);
        let mesh: &[crate::data::Vec2u] = super::scene::slice(&data.mesh.mesh.edge);
        for slot in 0..slots {
            let e = edge[slot] as usize;
            if e >= rods {
                return Err(Fatal::invariant(format!(
                    "solver driver: Cosserat frame {slot} rides on edge {e}, outside the \
                     {rods} rod edges"
                )));
            }
            let (a, b) = (nodes[slot][0], nodes[slot][1]);
            let (m0, m1) = (mesh[e][0], mesh[e][1]);
            if !((a == m0 && b == m1) || (a == m1 && b == m0)) {
                return Err(Fatal::invariant(format!(
                    "solver driver: Cosserat frame {slot} names vertices ({a}, {b}) but rides \
                     on edge {e}, whose vertices are ({m0}, {m1})"
                )));
            }
            if a as usize >= vertices || b as usize >= vertices {
                return Err(Fatal::invariant(format!(
                    "solver driver: Cosserat frame {slot} names vertex {} of {vertices}",
                    a.max(b)
                )));
            }
        }
        for site in 0..sites {
            let (p, n) = (bend[site][0] as usize, bend[site][1] as usize);
            if p >= slots || n >= slots || p == n {
                return Err(Fatal::invariant(format!(
                    "solver driver: Cosserat bend site {site} joins frames {p} and {n} of {slots}"
                )));
            }
            if !(inv_len[site] > 0.0 && inv_len[site].is_finite()) {
                return Err(Fatal::invariant(format!(
                    "solver driver: Cosserat bend site {site} has inverse length {}; the \
                     curvature is a rotation per unit length and needs a positive one",
                    inv_len[site]
                )));
            }
        }
        let fixed: &[u32] = super::scene::slice(&data.cosserat_fixed);
        let omega: &[f32] = super::scene::slice(&data.cosserat_end_omega);
        self.held_static = fixed.iter().any(|&f| f != 0);
        self.driven = omega.iter().any(|&w| w != 0.0);

        stage(device, &mut self.slot_edge, data.cosserat_edge.data, slots, "cosserat.slot_edge")?;
        stage(
            device,
            &mut self.edge_nodes,
            data.cosserat_edge_nodes.data as *const u32,
            2 * slots,
            "cosserat.edge_nodes",
        )?;
        stage(device, &mut self.fixed, data.cosserat_fixed.data, slots, "cosserat.fixed")?;
        stage(device, &mut self.end_omega, data.cosserat_end_omega.data, slots, "cosserat.end_omega")?;
        stage(
            device,
            &mut self.bend_slots,
            data.cosserat_bend_slots.data as *const u32,
            2 * sites,
            "cosserat.bend_slots",
        )?;
        stage(
            device,
            &mut self.rest_darboux,
            data.cosserat_rest_darboux.data as *const f32,
            3 * sites,
            "cosserat.rest_darboux",
        )?;
        stage(device, &mut self.inv_len, data.cosserat_bend_invlen.data, sites, "cosserat.inv_len")?;
        stage(
            device,
            &mut self.rest_metric,
            data.cosserat_rest_metric.data,
            2 * sites,
            "cosserat.rest_metric",
        )?;

        // THE CHAINS, the connected components of the bend sites. The frame
        // builder splits a rod at every junction and leaves no site across one,
        // so a component is one chain or one closed loop.
        let mut parent: Vec<usize> = (0..slots).collect();
        fn root(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        for pair in bend {
            let (a, b) = (root(&mut parent, pair[0] as usize), root(&mut parent, pair[1] as usize));
            if a != b {
                parent[a.max(b)] = a.min(b);
            }
        }
        let mut members: std::collections::BTreeMap<usize, Vec<u32>> = std::collections::BTreeMap::new();
        for slot in 0..slots {
            let r = root(&mut parent, slot);
            members.entry(r).or_default().push(slot as u32);
        }
        let mut offsets = vec![0u32];
        let mut order = Vec::with_capacity(slots);
        for list in members.values() {
            order.extend_from_slice(list);
            offsets.push(order.len() as u32);
        }
        self.chains = members.len();
        stage(device, &mut self.chain_offset, offsets.as_ptr(), offsets.len(), "cosserat.chain_offset")?;
        stage(device, &mut self.chain_slot, order.as_ptr(), slots, "cosserat.chain_slot")?;
        self.roll.size(device, 3 * slots, AllocLabel("cosserat.roll"))?;
        let mut covered = vec![0u32; vertices];
        for pair in nodes {
            covered[pair[0] as usize] = 1;
            covered[pair[1] as usize] = 1;
        }
        stage(device, &mut self.covered, covered.as_ptr(), vertices, "cosserat.covered")?;
        self.factor.size(device, 45 * slots, AllocLabel("cosserat.factor"))?;

        // THE HISTORY STARTS AT REST: all three copies hold the built frames,
        // so the first step's predicted rotation is the identity.
        let quat: &[f32] = super::scene::slice(&data.cosserat_quat);
        for (buffer, label) in [
            (&mut self.quat, "cosserat.quat"),
            (&mut self.quat_prev, "cosserat.quat_prev"),
            (&mut self.quat_prev2, "cosserat.quat_prev2"),
        ] {
            buffer.size(device, 4 * slots, AllocLabel(label))?;
            buffer.write(device, 0, quat)?;
        }
        self.gradient.size(device, 3 * slots, AllocLabel("cosserat.gradient"))?;
        self.self_block.size(device, 9 * slots, AllocLabel("cosserat.self_block"))?;
        self.coupling_from.size(device, 9 * slots, AllocLabel("cosserat.coupling_from"))?;
        self.coupling_to.size(device, 9 * slots, AllocLabel("cosserat.coupling_to"))?;
        self.site_block.size(device, 9 * sites.max(1), AllocLabel("cosserat.site_block"))?;
        self.held.size(device, slots, AllocLabel("cosserat.held"))?;
        self.prescribed.size(device, 3 * slots, AllocLabel("cosserat.prescribed"))?;
        self.step.size(device, 3 * slots, AllocLabel("cosserat.step"))?;
        self.turn.size(device, slots, AllocLabel("cosserat.turn"))?;
        ::log::info!(
            "Cosserat rods carry {slots} frames in {} chains and {sites} bend and twist \
             sites; the augmented solve has {} orientation DOFs",
            self.chains,
            3 * slots
        );
        Ok(())
    }

    /// The frames' handles the augmented solve reads, taken together so a
    /// caller cannot pair one iteration's blocks with another's. `prescribed`
    /// says whether any vertex is prescribed this step, which is when a clamp
    /// can hold a frame beside the ones `cosserat-fix-ends` holds.
    pub fn blocks(&self, prescribed: bool) -> Blocks {
        Blocks {
            slots: self.slots as u32,
            sites: self.sites as u32,
            project: self.held_static || prescribed,
            edge_nodes: self.edge_nodes.handle(),
            slot_edge: self.slot_edge.handle(),
            held: self.held.handle(),
            prescribed: self.prescribed.handle(),
            quat: self.quat.handle(),
            chains: self.chains as u32,
            chain_offset: self.chain_offset.handle(),
            chain_slot: self.chain_slot.handle(),
            roll: self.roll.handle(),
            covered: self.covered.handle(),
            factor: self.factor.handle(),
            bend_slots: self.bend_slots.handle(),
            gradient: self.gradient.handle(),
            self_block: self.self_block.handle(),
            coupling_from: self.coupling_from.handle(),
            coupling_to: self.coupling_to.handle(),
            site_block: self.site_block.span(0, 9 * self.sites),
            step: self.step.handle(),
        }
    }

    /// Return the frames to the last committed state.
    ///
    /// A no-op after a committed step, since the commit leaves the two equal,
    /// and the rollback of a step that was abandoned after its Newton loop had
    /// already turned the frames: the positions restart from the committed
    /// pose, and the frames must restart with them.
    pub fn restore<D: Device>(&mut self, device: &mut D) -> FatalResult<()> {
        if self.slots == 0 {
            return Ok(());
        }
        self.quat.copy_from(device, &self.quat_prev)?;
        Ok(())
    }

    /// The three energies of one Newton iteration, from cleared buffers.
    ///
    /// THE SHEAR PASS writes the position half where the rod stretch writes its
    /// own, the force and the fixed matrix through the rod edge's CSR slots, so
    /// it runs where the rod stretch does: after it and before anything that
    /// snapshots the matrix. Its frame half, and the inertia and bend passes,
    /// go to this module's buffers.
    ///
    /// `tr` is `dt / prev_dt`, the ratio the position predictor scales its
    /// velocity by, so the frame predictor extrapolates over the same step.
    ///
    /// # Safety
    /// The scene's arrays must be live for the call.
    pub unsafe fn assemble<D: Device>(
        state: &mut SolverState,
        device: &mut D,
        fixed: &mut super::fixedcsr::FixedCsr<'_>,
        dt: f32,
        tr: f32,
    ) -> FatalResult<()> {
        let slots = state.cosserat.slots;
        if slots == 0 {
            return Ok(());
        }
        let sites = state.cosserat.sites;
        {
            let c = &state.cosserat;
            for buffer in [&c.gradient, &c.self_block, &c.coupling_from, &c.coupling_to] {
                device.fill_zero(buffer.handle(), buffer.len() * size_of::<f32>())?;
            }
            device.fill_zero(c.site_block.handle(), c.site_block.len() * size_of::<f32>())?;
        }

        let (fixed_index, fixed_offset, fixed_value, row_count) = fixed.device_push_refs();
        super::fixedcsr::FixedCsr::seed_push_staging(device, &mut state.push)?;
        let c = &state.cosserat;
        let shear = CosseratShearRowArgs {
            x: state.eval_x.handle(),
            edge_nodes: c.edge_nodes.handle(),
            slot_edge: c.slot_edge.handle(),
            rod_edge: state.mesh_edge.handle(),
            edge_prop: state.prop_edge.handle(),
            edge_param: state.param_edge.handle(),
            quat: c.quat.handle(),
            hess_slots: state.edge_hess_slots.handle(),
            has_hess_slots: u32::from(state.edge_hess_slots.len() > 0),
            force: state.force.handle(),
            fixed_index,
            fixed_offset,
            fixed_value,
            row_count,
            refused: state.push.refused.handle(),
            witness: state.push.witness.handle(),
            gradient: c.gradient.handle(),
            self_block: c.self_block.handle(),
            coupling_from: c.coupling_from.handle(),
            coupling_to: c.coupling_to.handle(),
            count: slots as u32,
            seam_arena_count: 0,
        };
        device.launch("cosserat.shear", &shear, slots as u32)?;

        let inertia = CosseratInertiaRowArgs {
            slot_edge: c.slot_edge.handle(),
            edge_prop: state.prop_edge.handle(),
            quat: c.quat.handle(),
            quat_prev: c.quat_prev.handle(),
            quat_prev2: c.quat_prev2.handle(),
            dt,
            tr,
            gradient: c.gradient.handle(),
            self_block: c.self_block.handle(),
            count: slots as u32,
            seam_arena_count: 0,
        };
        device.launch("cosserat.inertia", &inertia, slots as u32)?;

        if sites > 0 {
            let bend = CosseratBendRowArgs {
                bend_slots: c.bend_slots.handle(),
                slot_edge: c.slot_edge.handle(),
                edge_prop: state.prop_edge.handle(),
                edge_param: state.param_edge.handle(),
                inv_len: c.inv_len.handle(),
                rest_metric: c.rest_metric.handle(),
                rest_darboux: c.rest_darboux.handle(),
                quat: c.quat.handle(),
                gradient: c.gradient.handle(),
                self_block: c.self_block.handle(),
                site_block: c.site_block.handle(),
                count: sites as u32,
                seam_arena_count: 0,
            };
            device.launch("cosserat.bend", &bend, sites as u32)?;
        }
        fixed.report_refusals(device, &mut state.push, "Cosserat shear")
    }

    /// Lift each removed vertex's known increment onto the frames it couples
    /// to, and zero those couplings.
    ///
    /// AFTER the fixed matrix's own Dirichlet pass and with the targets that
    /// pass read, before anything overwrites `target`: the matrix pass cannot
    /// see the couplings, which live here, and an identity row with a live
    /// coupling makes the augmented matrix indefinite.
    pub fn reduce_dirichlet<D: Device>(state: &mut SolverState, device: &mut D) -> FatalResult<()> {
        let slots = state.cosserat.slots;
        if slots == 0 {
            return Ok(());
        }
        let c = &state.cosserat;
        let args = CosseratDirichletRowArgs {
            edge_nodes: c.edge_nodes.handle(),
            dof_mask: state.dof_mask.handle(),
            eval_x: state.eval_x.handle(),
            target: state.target.handle(),
            gradient: c.gradient.handle(),
            coupling_from: c.coupling_from.handle(),
            coupling_to: c.coupling_to.handle(),
            count: slots as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer is the state's and outlives the dispatch.
        unsafe { device.launch("cosserat.dirichlet", &args, slots as u32) }?;
        Ok(())
    }

    /// This step's held mask and the solve's seed on the frame rows, from the
    /// vertices the Dirichlet pass prescribes and the targets it read.
    ///
    /// EVERY ITERATION, not only when a vertex is prescribed: with none, the
    /// mask is `cosserat-fix-ends`' own and the seed is zero, and computing that
    /// on the device is one dispatch where keeping a second copy current would
    /// be a branch in every caller.
    pub fn prescribe<D: Device>(state: &mut SolverState, device: &mut D) -> FatalResult<()> {
        let slots = state.cosserat.slots;
        if slots == 0 {
            return Ok(());
        }
        let c = &state.cosserat;
        let args = CosseratPrescribeRowArgs {
            edge_nodes: c.edge_nodes.handle(),
            fixed: c.fixed.handle(),
            dof_mask: state.dof_mask.handle(),
            eval_x: state.eval_x.handle(),
            target: state.target.handle(),
            held: c.held.handle(),
            prescribed: c.prescribed.handle(),
            count: slots as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer is the state's and outlives the dispatch.
        unsafe { device.launch("cosserat.prescribe", &args, slots as u32) }?;
        Ok(())
    }

    /// The largest turn the solved direction asks of any frame, in radians.
    pub fn largest_turn<D: Device>(
        state: &mut SolverState,
        device: &mut D,
    ) -> FatalResult<f32> {
        let slots = state.cosserat.slots;
        if slots == 0 {
            return Ok(0.0);
        }
        let args = DxMagnitudeArgs {
            direction: state.cosserat.step.handle(),
            magnitude: state.cosserat.turn.handle(),
            count: slots as u32,
            seam_arena_count: 0,
        };
        // Safety: both buffers are the state's and outlive the dispatches.
        unsafe {
            device.launch("cosserat.turn", &args, slots as u32)?;
            let turn = state.cosserat.turn.handle();
            Ok(state.fold.max(device, "cosserat.max_turn", turn, slots as u32, 0.0)?)
        }
    }

    /// Turn every frame by the fraction of the solved step the line search
    /// accepted: `R <- exp(-scale dtheta) R`, the twin of `x -= scale dx`, and
    /// carry a clamped edge's frame with its edge exactly.
    ///
    /// `before` is the iterate's positions before the step and `after` the
    /// accepted ones, which at the call site are `target` and `eval_x`.
    pub fn advance<D: Device>(
        state: &mut SolverState,
        device: &mut D,
        scale: f32,
        before: Handle,
        after: Handle,
    ) -> FatalResult<()> {
        let slots = state.cosserat.slots;
        if slots == 0 {
            return Ok(());
        }
        let c = &state.cosserat;
        let args = CosseratStepRowArgs {
            step: c.step.handle(),
            scale,
            edge_nodes: c.edge_nodes.handle(),
            dof_mask: state.dof_mask.handle(),
            before,
            after,
            quat: c.quat.handle(),
            count: slots as u32,
            seam_arena_count: 0,
        };
        // Safety: every buffer is the state's and outlives the dispatch.
        unsafe { device.launch("cosserat.step", &args, slots as u32) }?;
        Ok(())
    }

    /// Drive the held ends over the committed step's `dt`, then move the
    /// history one step: `prev2` takes the OLD `prev` before `prev` takes the
    /// current frame.
    pub fn commit<D: Device>(&mut self, device: &mut D, dt: f32) -> FatalResult<()> {
        if self.slots == 0 {
            return Ok(());
        }
        if self.driven {
            let args = CosseratDriveRowArgs {
                end_omega: self.end_omega.handle(),
                dt,
                quat: self.quat.handle(),
                count: self.slots as u32,
                seam_arena_count: 0,
            };
            // Safety: both buffers are this state's and outlive the dispatch.
            unsafe { device.launch("cosserat.drive", &args, self.slots as u32) }?;
        }
        self.quat_prev2.copy_from(device, &self.quat_prev)?;
        self.quat_prev.copy_from(device, &self.quat)?;
        Ok(())
    }

    /// The current frames, read back, four floats per slot.
    #[allow(dead_code)]
    pub fn frames<D: Device>(&self, device: &mut D) -> FatalResult<Vec<f32>> {
        let mut out = vec![0.0f32; 4 * self.slots];
        self.quat.read(device, 0, &mut out)?;
        Ok(out)
    }

    /// Overwrite the frame state with a stated pose and history.
    #[cfg(test)]
    pub fn set_frames<D: Device>(
        &mut self,
        device: &mut D,
        quat: &[f32],
        prev: &[f32],
        prev2: &[f32],
    ) -> FatalResult<()> {
        self.quat.write(device, 0, quat)?;
        self.quat_prev.write(device, 0, prev)?;
        self.quat_prev2.write(device, 0, prev2)?;
        Ok(())
    }

    /// What the last assembly wrote to this module's buffers, read back:
    /// the gradient, the frame blocks, the two couplings and the site blocks.
    #[cfg(test)]
    pub fn assembled<D: Device>(&self, device: &mut D) -> FatalResult<[Vec<f32>; 5]> {
        let read = |device: &mut D, buffer: &Buffer<f32>, n: usize| -> FatalResult<Vec<f32>> {
            let mut out = vec![0.0f32; n];
            buffer.read(device, 0, &mut out)?;
            Ok(out)
        };
        Ok([
            read(device, &self.gradient, 3 * self.slots)?,
            read(device, &self.self_block, 9 * self.slots)?,
            read(device, &self.coupling_from, 9 * self.slots)?,
            read(device, &self.coupling_to, 9 * self.slots)?,
            read(device, &self.site_block, 9 * self.sites)?,
        ])
    }
}

/// What the augmented solve reads of the frames, by handle.
#[derive(Clone, Copy)]
pub struct Blocks {
    pub slots: u32,
    pub sites: u32,
    /// Whether any frame can be held this step, so the projection can be
    /// skipped when none can.
    pub project: bool,
    pub edge_nodes: Handle,
    /// Per slot, the rod edge it rides on.
    pub slot_edge: Handle,
    /// This step's held mask, one word per slot.
    pub held: Handle,
    /// The solve's seed on the frame rows, three floats per slot.
    pub prescribed: Handle,
    /// The frames, which the coarse space's roll vectors are read off.
    pub quat: Handle,
    /// The chains and their roll vectors, the coarse space.
    pub chains: u32,
    pub chain_offset: Handle,
    pub chain_slot: Handle,
    pub roll: Handle,
    /// The preconditioner's per-vertex coverage mask and per-slot factors.
    pub covered: Handle,
    pub factor: Handle,
    pub bend_slots: Handle,
    pub gradient: Handle,
    pub self_block: Handle,
    pub coupling_from: Handle,
    pub coupling_to: Handle,
    pub site_block: Handle,
    pub step: Handle,
}

#[cfg(test)]
mod tests {
    //! The three frame energies against a finite-difference oracle.
    //!
    //! THE ORACLE IS THE ENERGY, written out here in f64 from the formulas the
    //! kernel file states, and differenced: the gradient by central differences
    //! and the Hessian by second differences of the energy as a function of the
    //! perturbation vector, the frame perturbed on the left,
    //! `R <- exp(skew(theta)) R`, as the kernels define it. The kernel's Hessian
    //! is PSD-projected per element, so each test isolates ONE element and
    //! compares against the projection of the differenced Hessian, which makes
    //! the projection part of what is checked.

    use super::*;
    use crate::cvec::CVec;
    use crate::data::{EdgeParam, Vec2u, Vec3f};
    use crate::driver::assemble::tests::{
        clear_force, dense_of, force_of, install_pattern, install_rod_adjacency, positions_of,
    };
    use crate::driver::fixedcsr::FixedCsr;
    use crate::driver::launch::host_device;
    use crate::driver::test_scene::TestScene;
    use ppf_cts_compute::HostDevice;

    type M3 = [[f64; 3]; 3];

    const EDGE_MASS: f32 = 0.05;
    const VERTEX_MASS: f32 = 0.025;

    fn identity() -> M3 {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn mul(a: &M3, b: &M3) -> M3 {
        let mut c = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    fn transpose(a: &M3) -> M3 {
        let mut t = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                t[i][j] = a[j][i];
            }
        }
        t
    }

    fn skew(v: [f64; 3]) -> M3 {
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    }

    fn exp(v: [f64; 3]) -> M3 {
        let a = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let k = skew(v);
        let k2 = mul(&k, &k);
        let (s, c) = if a < 1e-12 { (1.0, 0.5) } else { (a.sin() / a, (1.0 - a.cos()) / (a * a)) };
        let mut r = identity();
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] += s * k[i][j] + c * k2[i][j];
            }
        }
        r
    }

    fn axial_diff(m: &M3) -> [f64; 3] {
        [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]]
    }

    /// The discrete Darboux vector `(2 / l) vec(q(g0))`, written as the
    /// quaternion's vector part rather than through the kernel's quotient, so
    /// the oracle is an independent spelling of the same quantity.
    fn darboux(g0: &M3, inv_len: f64) -> [f64; 3] {
        let w = 0.5 * (1.0 + g0[0][0] + g0[1][1] + g0[2][2]).sqrt();
        let v = [
            (g0[2][1] - g0[1][2]) / (4.0 * w),
            (g0[0][2] - g0[2][0]) / (4.0 * w),
            (g0[1][0] - g0[0][1]) / (4.0 * w),
        ];
        v.map(|x| 2.0 * inv_len * x)
    }

    /// Shepperd's quaternion `(x, y, z, w)` of a rotation.
    fn quat_of(r: &M3) -> [f32; 4] {
        let tr = r[0][0] + r[1][1] + r[2][2];
        let q = if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, 0.25 * s]
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
            [0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s]
        } else if r[1][1] > r[2][2] {
            let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
            [(r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s]
        } else {
            let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
            [(r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s, (r[1][0] - r[0][1]) / s]
        };
        [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
    }

    /// The frame a stored quaternion stands for, as `pdrd_quat_to_mat` reads it.
    fn frame_of(q: [f32; 4]) -> M3 {
        let [x, y, z, w] = q.map(f64::from);
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    }

    /// The nearest PSD matrix to a symmetric one, by Jacobi rotations.
    fn psd_project(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut m: Vec<Vec<f64>> =
            (0..n).map(|i| (0..n).map(|j| 0.5 * (a[i][j] + a[j][i])).collect()).collect();
        let mut v: Vec<Vec<f64>> =
            (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
        for _ in 0..200 {
            let off: f64 = (0..n)
                .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
                .map(|(i, j)| m[i][j] * m[i][j])
                .sum();
            if off < 1e-40 {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    if m[p][q].abs() < 1e-300 {
                        continue;
                    }
                    let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let (kp, kq) = (m[k][p], m[k][q]);
                        m[k][p] = c * kp - s * kq;
                        m[k][q] = s * kp + c * kq;
                    }
                    for k in 0..n {
                        let (pk, qk) = (m[p][k], m[q][k]);
                        m[p][k] = c * pk - s * qk;
                        m[q][k] = s * pk + c * qk;
                    }
                    for k in 0..n {
                        let (kp, kq) = (v[k][p], v[k][q]);
                        v[k][p] = c * kp - s * kq;
                        v[k][q] = s * kp + c * kq;
                    }
                }
            }
        }
        let mut out = vec![vec![0.0; n]; n];
        for k in 0..n {
            let lambda = m[k][k].max(0.0);
            for i in 0..n {
                for j in 0..n {
                    out[i][j] += lambda * v[i][k] * v[j][k];
                }
            }
        }
        out
    }

    /// A chain of Cosserat edges `0-1-2-...` and what it is made of.
    struct Chain {
        shear: f32,
        bend: f32,
        twist: f32,
        /// Per site: the two slots, the rest Darboux vector, the inverse length.
        sites: Vec<([u32; 2], [f32; 3], f32)>,
    }

    struct Fixture {
        scene: TestScene,
        state: SolverState,
        device: HostDevice,
        chain: Chain,
        vertices: usize,
        slots: usize,
    }

    /// What one assembly produced, over `u = [x | theta]`.
    struct Assembled {
        gradient: Vec<f64>,
        hessian: Vec<Vec<f64>>,
    }

    impl Fixture {
        fn new(positions: &[[f32; 3]], chain: Chain) -> Self {
            let vertices = positions.len();
            let slots = vertices - 1;
            let edges: Vec<Vec2u> = (0..slots as u32).map(|i| Vec2u::new(i, i + 1)).collect();
            let mut scene = TestScene::new(vertices).with_edges(&edges);
            for (i, x) in positions.iter().enumerate() {
                scene.place(i, x[0], x[1], x[2]);
            }
            scene.data.rod_count = slots as u32;
            scene.data.surface_vert_count = vertices as u32;
            scene.data.param_arrays.edge = CVec::from(
                &[EdgeParam {
                    cosserat_shear: chain.shear,
                    cosserat_bend: chain.bend,
                    cosserat_twist: chain.twist,
                    ..EdgeParam::default()
                }][..],
            );
            for (i, prop) in scene.data.prop.edge.as_mut_slice().iter_mut().enumerate() {
                let (a, b) = (positions[i], positions[i + 1]);
                let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2))
                    .sqrt();
                prop.length = length;
                prop.initial_length = length;
                prop.mass = EDGE_MASS;
            }
            for vertex in scene.data.prop.vertex.as_mut_slice() {
                vertex.mass = VERTEX_MASS;
            }
            let identity = [0.0f32, 0.0, 0.0, 1.0];
            scene.data.cosserat_edge = CVec::from(&(0..slots as u32).collect::<Vec<_>>()[..]);
            scene.data.cosserat_edge_nodes = CVec::from(&edges[..]);
            scene.data.cosserat_quat = CVec::from(&identity.repeat(slots)[..]);
            scene.data.cosserat_fixed = CVec::from(&vec![0u32; slots][..]);
            scene.data.cosserat_end_omega = CVec::from(&vec![0.0f32; slots][..]);
            let bend: Vec<Vec2u> =
                chain.sites.iter().map(|(s, _, _)| Vec2u::new(s[0], s[1])).collect();
            let rest: Vec<Vec3f> =
                chain.sites.iter().map(|(_, r, _)| Vec3f::new(r[0], r[1], r[2])).collect();
            let inv_len: Vec<f32> = chain.sites.iter().map(|(_, _, l)| *l).collect();
            scene.data.cosserat_bend_slots = CVec::from(&bend[..]);
            scene.data.cosserat_rest_darboux = CVec::from(&rest[..]);
            scene.data.cosserat_bend_invlen = CVec::from(&inv_len[..]);
            let metric: Vec<f32> = chain
                .sites
                .iter()
                .flat_map(|(_, r, l)| crate::cosserat::rest_metric(*r, *l))
                .collect();
            scene.data.cosserat_rest_metric = CVec::from(&metric[..]);
            install_rod_adjacency(&mut scene.data, vertices, &edges);
            install_pattern(&mut scene.data, vertices);
            let mut state = SolverState::default();
            let mut device = host_device();
            // Safety: the scene lives in its box for the whole test.
            unsafe { state.allocate(&mut device, &scene.data) }
                .expect("the fixture scene allocates");
            assert_eq!(state.cosserat.slots, slots);
            Self { scene, state, device, chain, vertices, slots }
        }

        /// The positions the kernels see, in f64.
        fn positions(&self) -> Vec<[f64; 3]> {
            let flat = positions_of(&self.scene);
            (0..self.vertices)
                .map(|i| [0, 1, 2].map(|k| f64::from(flat[3 * i + k])))
                .collect()
        }

        fn assemble(&mut self, frames: &[M3], prev: &[M3], prev2: &[M3], dt: f32, tr: f32) -> Assembled {
            let flat = |m: &[M3]| m.iter().flat_map(|r| quat_of(r)).collect::<Vec<f32>>();
            let Fixture { scene, state, device, .. } = self;
            state
                .cosserat
                .set_frames(device, &flat(frames), &flat(prev), &flat(prev2))
                .expect("the frames upload");
            clear_force(device, state);
            crate::driver::state::reseed_committed(device, state, &scene.data);
            crate::driver::state::reseed_props(device, state, &scene.data);
            crate::driver::state::reseed_pattern(device, state, &scene.data);
            state.eval_x.seed(device, &positions_of(scene)).expect("the iterate seeds");
            // Safety: the scene is live and the state was allocated for it.
            let (force, dense) = unsafe {
                let mut fixed = FixedCsr::adopt_from_dataset(
                    device,
                    state.fixed_pattern_refs(),
                    &scene.data,
                    Default::default(),
                )
                .expect("the matrix adopts the pattern");
                State::assemble(state, device, &mut fixed, dt, tr).expect("the frames assemble");
                fixed.download(device).expect("the matrix reads back");
                (force_of(device, state), dense_of(&fixed, self.vertices))
            };
            let [gradient, own, from, to, site] =
                state.cosserat.assembled(device).expect("the frame blocks read back");
            let (nx, n) = (3 * self.vertices, 3 * (self.vertices + self.slots));
            let mut g = vec![0.0; n];
            let mut h = vec![vec![0.0; n]; n];
            for i in 0..nx {
                g[i] = f64::from(force[i]);
                for j in 0..nx {
                    h[i][j] = f64::from(dense[nx * i + j]);
                }
            }
            for s in 0..self.slots {
                let t = nx + 3 * s;
                for i in 0..3 {
                    g[t + i] = f64::from(gradient[3 * s + i]);
                    for j in 0..3 {
                        h[t + i][t + j] += f64::from(own[9 * s + i + 3 * j]);
                        // The slot's first vertex is `s` and its second `s + 1`.
                        for (vertex, block) in [(s, &from), (s + 1, &to)] {
                            let value = f64::from(block[9 * s + i + 3 * j]);
                            h[3 * vertex + i][t + j] += value;
                            h[t + j][3 * vertex + i] += value;
                        }
                    }
                }
            }
            for (k, (pair, _, _)) in self.chain.sites.iter().enumerate() {
                let (prev_t, next_t) = (nx + 3 * pair[0] as usize, nx + 3 * pair[1] as usize);
                for i in 0..3 {
                    for j in 0..3 {
                        let value = f64::from(site[9 * k + i + 3 * j]);
                        h[next_t + i][prev_t + j] += value;
                        h[prev_t + j][next_t + i] += value;
                    }
                }
            }
            Assembled { gradient: g, hessian: h }
        }

        /// The energy at a perturbation `u = [x | theta]` of the stated pose.
        #[allow(clippy::too_many_arguments)]
        fn energy(
            &self,
            u: &[f64],
            x0: &[[f64; 3]],
            frames: &[M3],
            prev: &[M3],
            prev2: &[M3],
            dt: f64,
            tr: f64,
        ) -> f64 {
            let nx = 3 * self.vertices;
            let x: Vec<[f64; 3]> =
                (0..self.vertices).map(|i| [0, 1, 2].map(|k| x0[i][k] + u[3 * i + k])).collect();
            let r: Vec<M3> = (0..self.slots)
                .map(|s| mul(&exp([u[nx + 3 * s], u[nx + 3 * s + 1], u[nx + 3 * s + 2]]), &frames[s]))
                .collect();
            let mass = f64::from(EDGE_MASS);
            let props = self.scene.edge_props();
            let mut e = 0.0;
            for s in 0..self.slots {
                let shear = f64::from(self.chain.shear);
                if shear > 0.0 {
                    let d: Vec<f64> = (0..3).map(|k| x[s + 1][k] - x[s][k]).collect();
                    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let dot: f64 = (0..3).map(|k| d[k] / l * r[s][k][2]).sum();
                    e += 0.5 * shear * mass * (1.0 - dot);
                }
                if dt > 0.0 {
                    let length = f64::from(props[s].length);
                    let j = mass * length * length / 12.0;
                    let w = axial_diff(&mul(&prev[s], &transpose(&prev2[s]))).map(|v| 0.5 * tr * v);
                    let mut step = skew(w);
                    for i in 0..3 {
                        step[i][i] += 1.0;
                    }
                    let predicted = mul(&step, &prev[s]);
                    let m0 = mul(&r[s], &transpose(&predicted));
                    e -= j / (dt * dt) * (m0[0][0] + m0[1][1] + m0[2][2]);
                }
            }
            for (pair, rest, inv_len) in &self.chain.sites {
                let g0 = mul(&transpose(&r[pair[0] as usize]), &r[pair[1] as usize]);
                let inv_len = f64::from(*inv_len);
                let w = darboux(&g0, inv_len);
                // The rigidity is the parameter times the linear density times
                // (1 cm)^2, the node's mass over its Voronoi length being the
                // linear density.
                let voronoi = 0.5
                    * (f64::from(props[pair[0] as usize].length)
                        + f64::from(props[pair[1] as usize].length));
                let density = mass / voronoi;
                let k = [
                    f64::from(self.chain.bend) * density * 1e-4,
                    f64::from(self.chain.bend) * density * 1e-4,
                    f64::from(self.chain.twist) * density * 1e-4,
                ];
                // The rest metric M along and across the rest axis, and the
                // energy 1/2 l d^T M K M d, d = w - w0.
                let [along, across] = crate::cosserat::rest_metric(*rest, inv_len as f32)
                    .map(f64::from);
                let r: Vec<f64> = rest.iter().map(|v| f64::from(*v)).collect();
                let norm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
                let axis: Vec<f64> = if norm > 0.0 { r.iter().map(|v| v / norm).collect() } else { vec![0.0; 3] };
                let d: Vec<f64> = (0..3).map(|i| w[i] - r[i]).collect();
                let along_d: f64 = (0..3).map(|i| axis[i] * d[i]).sum();
                let md: Vec<f64> = (0..3).map(|i| across * d[i] + (along - across) * axis[i] * along_d).collect();
                for i in 0..3 {
                    e += 0.5 / inv_len * k[i] * md[i] * md[i];
                }
            }
            e
        }

        /// The differenced gradient and the projected differenced Hessian.
        fn oracle(&self, frames: &[M3], prev: &[M3], prev2: &[M3], dt: f64, tr: f64) -> Assembled {
            let x0 = self.positions();
            let n = 3 * (self.vertices + self.slots);
            let f = |u: &[f64]| self.energy(u, &x0, frames, prev, prev2, dt, tr);
            let at = |pairs: &[(usize, f64)]| {
                let mut u = vec![0.0; n];
                for &(i, h) in pairs {
                    u[i] += h;
                }
                f(&u)
            };
            let (hg, hh) = (1e-6, 1e-4);
            let gradient = (0..n).map(|i| (at(&[(i, hg)]) - at(&[(i, -hg)])) / (2.0 * hg)).collect();
            let mut hessian = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    hessian[i][j] = (at(&[(i, hh), (j, hh)]) - at(&[(i, hh), (j, -hh)])
                        - at(&[(i, -hh), (j, hh)])
                        + at(&[(i, -hh), (j, -hh)]))
                        / (4.0 * hh * hh);
                }
            }
            Assembled { gradient, hessian: psd_project(&hessian) }
        }
    }

    /// The largest entry of `a - b` against the largest of `b`.
    fn relative(a: &[f64], b: &[f64]) -> f64 {
        let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let error = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        error / scale
    }

    fn check(got: &Assembled, want: &Assembled, what: &str) {
        let flat = |h: &[Vec<f64>]| h.iter().flatten().copied().collect::<Vec<_>>();
        let gradient = relative(&got.gradient, &want.gradient);
        let hessian = relative(&flat(&got.hessian), &flat(&want.hessian));
        assert!(
            gradient < 1e-4,
            "{what}: the gradient is {gradient:.2e} off the differenced energy\n got {:?}\nwant {:?}",
            got.gradient,
            want.gradient
        );
        assert!(
            hessian < 2e-4,
            "{what}: the Hessian is {hessian:.2e} off the projected differenced Hessian"
        );
    }

    #[test]
    fn the_shear_energy_matches_its_differenced_energy() {
        let mut fixture = Fixture::new(
            &[[0.125, 0.25, -0.0625], [0.5, 0.375, 0.125]],
            Chain { shear: 1.0e3, bend: 0.0, twist: 0.0, sites: vec![] },
        );
        // Misaligned with the edge by much more than round-off, so the exact
        // Hessian is indefinite and the projection is part of what is checked.
        let frames = [exp([0.3, -0.2, 0.5])];
        let got = fixture.assemble(&frames, &frames, &frames, 0.0, 1.0);
        let want = fixture.oracle(&frames, &frames, &frames, 0.0, 1.0);
        check(&got, &want, "shear");
    }

    #[test]
    fn the_shear_of_a_frame_on_its_edge_is_at_rest() {
        // A frame whose third axis is the edge's direction has no shear: no
        // force, no torque.
        let mut fixture = Fixture::new(
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
            Chain { shear: 1.0e3, bend: 0.0, twist: 0.0, sites: vec![] },
        );
        let frames = [exp([0.0, 0.0, 0.7])];
        let got = fixture.assemble(&frames, &frames, &frames, 0.0, 1.0);
        assert!(got.gradient.iter().all(|g| g.abs() < 1e-6), "{:?}", got.gradient);
    }

    #[test]
    fn the_bend_and_twist_energy_matches_its_differenced_energy() {
        let mut fixture = Fixture::new(
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.125, 0.5]],
            Chain {
                shear: 0.0,
                bend: 5.0e4,
                twist: 3.0e4,
                sites: vec![([0, 1], [2.0, -1.0, 0.5], 4.0)],
            },
        );
        let first = exp([0.1, 0.2, 0.3]);
        let frames = [first, mul(&exp([-0.2, 0.4, 0.1]), &first)];
        let got = fixture.assemble(&frames, &frames, &frames, 0.0, 1.0);
        let want = fixture.oracle(&frames, &frames, &frames, 0.0, 1.0);
        check(&got, &want, "bend and twist");
    }

    #[test]
    fn the_rotational_inertia_matches_its_differenced_energy() {
        let mut fixture = Fixture::new(
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]],
            Chain { shear: 0.0, bend: 0.0, twist: 0.0, sites: vec![] },
        );
        let base = exp([0.2, -0.1, 0.4]);
        let prev2 = [mul(&exp([0.05, 0.0, 0.02]), &base)];
        let prev = [base];
        let frames = [mul(&exp([0.1, -0.2, 0.15]), &base)];
        let got = fixture.assemble(&frames, &prev, &prev2, 0.01, 0.8);
        let want = fixture.oracle(&frames, &prev, &prev2, 0.01, 0.8);
        check(&got, &want, "rotational inertia");
    }

    #[test]
    fn a_site_turned_past_a_right_angle_has_no_second_rest_state() {
        // THE PROPERTY THE QUATERNION MEASURE BUYS. The axial vector of G0's
        // antisymmetric part, sin(phi) a / l, peaks at a quarter turn, so a site
        // resting at phi0 = 1 rad would rest again at pi - 1 rad; a crimp of a
        // radian reached it under load and folded. With w = (2 / l) sin(phi / 2) a
        // the energy keeps rising, and the torque keeps pointing back to rest.
        let inv_len = 4.0f32;
        let phi0 = 1.0;
        let rest_w = darboux(&exp([phi0, 0.0, 0.0]), f64::from(inv_len)).map(|v| v as f32);
        let mut fixture = Fixture::new(
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.5]],
            Chain { shear: 0.0, bend: 5.0e4, twist: 3.0e4, sites: vec![([0, 1], rest_w, inv_len)] },
        );
        let mirror = [identity(), exp([std::f64::consts::PI - phi0, 0.0, 0.0])];
        let got = fixture.assemble(&mirror, &mirror, &mirror, 0.0, 1.0);
        // Turning the next frame further (positive x) must cost energy, so the
        // gradient's x component on the next frame is positive. The rows are the
        // three vertices' nine, then slot 0 (the previous frame), then slot 1.
        assert!(got.gradient[12] > 1.0, "no restoring torque at pi - phi0: {:?}", &got.gradient[9..]);
        let want = fixture.oracle(&mirror, &mirror, &mirror, 0.0, 1.0);
        check(&got, &want, "bend and twist past a right angle");
    }

    /// The rotation vector of a rotation matrix (angle times axis, below a half
    /// turn), in f64.
    fn log(r: &M3) -> [f64; 3] {
        let c = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
        let angle = c.acos();
        let u = axial_diff(r);
        if angle < 1e-12 {
            return u.map(|v| 0.5 * v);
        }
        u.map(|v| v * angle / (2.0 * angle.sin()))
    }

    #[test]
    fn the_bend_energy_at_rest_has_the_exact_rotation_vectors_second_variation() {
        // A site resting at a turn of 0.96 rad, a dense crimp's: the kernel's
        // Hessian there must be the exact energy's, 1/2 l |log(G0) / l - w0|^2_K
        // in the rotation vector, which the plain rod's turning angle also has
        // in its own plane. Without the rest metric the stiffness along the
        // turn's own axis falls to cos^2(0.48) = 79 % of it.
        let inv_len = 4.0f32;
        let rest_turn = [0.96, 0.0, 0.0];
        let bent = exp(rest_turn);
        let rest_w = darboux(&bent, f64::from(inv_len)).map(|v| v as f32);
        let mut fixture = Fixture::new(
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.5]],
            Chain { shear: 0.0, bend: 5.0e4, twist: 3.0e4, sites: vec![([0, 1], rest_w, inv_len)] },
        );
        let frames = [identity(), bent];
        let got = fixture.assemble(&frames, &frames, &frames, 0.0, 1.0);
        // The exact energy, differenced in the next frame's left perturbation.
        let density = f64::from(EDGE_MASS) / 0.25;
        let k = [5.0e4 * density * 1e-4, 5.0e4 * density * 1e-4, 3.0e4 * density * 1e-4];
        let l = 1.0 / f64::from(inv_len);
        let w0 = log(&bent).map(|v| v / l);
        let energy = |theta: [f64; 3]| -> f64 {
            let g = mul(&transpose(&frames[0]), &mul(&exp(theta), &frames[1]));
            let w = log(&g).map(|v| v / l);
            (0..3).map(|i| 0.5 * l * k[i] * (w[i] - w0[i]).powi(2)).sum()
        };
        let h = 1e-4;
        for a in 0..3 {
            for b in 0..3 {
                let at = |sa: f64, sb: f64| {
                    let mut t = [0.0; 3];
                    t[a] += sa * h;
                    t[b] += sb * h;
                    energy(t)
                };
                let exact = (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * h * h);
                // The next frame is slot 1, its rows after the nine vertex rows
                // and slot 0's three.
                let kernel = got.hessian[12 + a][12 + b];
                assert!(
                    (kernel - exact).abs() < 2e-3 * k[0] / l,
                    "H[{a}][{b}]: the kernel's {kernel} against the exact {exact}"
                );
            }
        }
    }

    #[test]
    fn a_frame_pair_at_the_mirror_of_its_rest_curvature_is_not_at_rest() {
        // THE PROPERTY A SIGNED REST CURVATURE BUYS. A plain rod's rest bend is
        // a turning angle without a sign, so a joint bent the other way by the
        // same angle is at rest too; a Cosserat site's rest Darboux vector is
        // signed, so its mirror carries the full energy and a restoring torque.
        let inv_len = 4.0f32;
        let bent = exp([0.3, 0.0, 0.0]);
        let rest_w = darboux(&bent, f64::from(inv_len)).map(|v| v as f32);
        let chain = || Chain {
            shear: 0.0,
            bend: 5.0e4,
            twist: 3.0e4,
            sites: vec![([0, 1], rest_w, inv_len)],
        };
        let positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.5]];
        let mut at_rest = Fixture::new(&positions, chain());
        let frames = [identity(), bent];
        let rest = at_rest.assemble(&frames, &frames, &frames, 0.0, 1.0);
        let mut mirrored = Fixture::new(&positions, chain());
        let mirror = [identity(), exp([-0.3, 0.0, 0.0])];
        let flipped = mirrored.assemble(&mirror, &mirror, &mirror, 0.0, 1.0);
        let torque = |a: &Assembled| a.gradient[9..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(torque(&rest) < 1e-5, "the rest pose carries a torque {}", torque(&rest));
        assert!(
            torque(&flipped) > 1.0,
            "the mirrored pose carries no restoring torque ({})",
            torque(&flipped)
        );
    }
}
