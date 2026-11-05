// File: crates/ppf-cts-solver/src/driver/tie.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Tie constraints: a vertex held on an affine combination of other vertices.
//!
//! A tie holds its vertex `t` at `x_t = sum_i w_i x_{g_i}` with
//! `sum_i w_i = 1`, so the held vertex owns no degree of freedom and moves
//! exactly with its controllers. `solver/tie.kernel.cpp` states the algebra:
//! the Newton system is solved on the subspace where every held row is zero,
//! through the expand `E` and the fold `F = E^T` wrapped around every operator
//! application (`super::operator::Operator::tie`), and the held rows are filled
//! in from their controllers after the solve. This module owns the topology,
//! its validation, and the five places a Newton iteration touches it:
//!
//! 1. the reduced block-Jacobi diagonal, after the operator's own
//!    ([`State::precond`]);
//! 2. the consistency term in each held row of the seed ([`State::seed`]);
//! 3. the folded right-hand side ([`State::fold_force`]);
//! 4. the wrapped operator ([`View`]);
//! 5. the held rows of the solved correction ([`State::finish`]).
//!
//! NOTHING IS ASSEMBLED AND NOTHING IS SNAPPED. The held vertex keeps its
//! mass, its elasticity and its contact in the full-space matrices, and the
//! fold carries them onto its controllers, so the reduced operator is SPD
//! wherever the full one is. The full correction leaves through the CCD line
//! search like any other, and because the tie is linear it holds along the
//! whole segment the line search tests.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, Handle, StagedBuffer};

use super::kernels::{
    TieConsistencyRowArgs, TieEdgeBlockRowArgs, TieExpandRowArgs, TieFinishRowArgs, TieFoldRowArgs,
    TiePrecondRowArgs, TieZeroRowArgs,
};
use super::scene::{Fatal, FatalResult};
use crate::data::{DataSet, VertexProp};

/// Controller slots per tie, the kernel's `tie_slots()`.
pub const TIE_SLOTS: usize = 8;

/// The `tie_of` marker of a vertex no tie holds, the kernel's `tie_none()`.
pub const TIE_NONE: u32 = u32::MAX;

/// How far a tie's stored weights may sum from one. They are authored in
/// double precision summing to one and stored as float32, so the sum misses one
/// by the float32 rounding of at most eight weights of order one, and a sum off
/// by more than this is a different constraint than the one authored.
pub const WEIGHT_SUM_TOLERANCE: f32 = 1.0e-4;

/// A scene array as a slice, for the host readers outside the driver (the
/// frame writer).
///
/// # Safety
/// The array must be live for the returned lifetime.
pub unsafe fn scene_slice<'a, T>(cvec: &crate::cvec::CVec<T>) -> &'a [T] {
    super::scene::slice(cvec)
}

/// The ties' host topology: per vertex the tie holding it, and per vertex the
/// slots that name it as a controller.
#[derive(Debug, Default, PartialEq)]
pub struct Topology {
    /// Per vertex, the index of the tie that holds it, or [`TIE_NONE`].
    pub tie_of: Vec<u32>,
    /// `vertices + 1` offsets into `ctrl_entry`.
    pub ctrl_offset: Vec<u32>,
    /// `TIE_SLOTS * tie + slot` for every nonzero slot naming a vertex, grouped
    /// by that vertex in ascending order. Ascending is what fixes the order of
    /// the fold's sum.
    pub ctrl_entry: Vec<u32>,
}

/// Validate the scene's ties and build their topology.
///
/// Every refusal names the tie and the vertex, since each one is an authoring
/// error a run cannot recover from: a tie the solve cannot honor exactly would
/// otherwise be honored approximately, or not at all, with nothing to say so.
pub fn build_topology(
    vertex: &[u32],
    ctrl: &[u32],
    weight: &[f32],
    props: &[VertexProp],
) -> Result<Topology, String> {
    let vertices = props.len();
    let ties = vertex.len();
    if ctrl.len() != TIE_SLOTS * ties || weight.len() != TIE_SLOTS * ties {
        return Err(format!(
            "the scene carries {ties} ties with {} controller indices and {} weights; each \
             tie carries {TIE_SLOTS} of each",
            ctrl.len(),
            weight.len()
        ));
    }
    let mut tie_of = vec![TIE_NONE; vertices];
    for (tie, &held) in vertex.iter().enumerate() {
        let t = held as usize;
        if t >= vertices {
            return Err(format!(
                "tie {tie} holds vertex {held}, outside the {vertices} vertices"
            ));
        }
        if tie_of[t] != TIE_NONE {
            return Err(format!(
                "vertex {held} is held by two ties, {} and {tie}; a vertex is held by one tie",
                tie_of[t]
            ));
        }
        tie_of[t] = tie as u32;
        let prop = &props[t];
        if prop.fix_index > 0 || prop.pull_index > 0 {
            return Err(format!(
                "tie {tie} holds vertex {held}, which a pin also holds; a held vertex owns no \
                 degree of freedom for a pin to prescribe, so pin its controllers instead"
            ));
        }
        if prop.pdrd_body_index != 0 || prop.collider {
            return Err(format!(
                "tie {tie} holds vertex {held}, which belongs to a rigid (PDRD) body or a \
                 static collider; a tie holds a vertex of a deformable object"
            ));
        }
    }
    let mut count = vec![0u32; vertices];
    for tie in 0..ties {
        let held = vertex[tie];
        let row = &ctrl[TIE_SLOTS * tie..TIE_SLOTS * (tie + 1)];
        let w = &weight[TIE_SLOTS * tie..TIE_SLOTS * (tie + 1)];
        let mut sum = 0.0f32;
        let mut used = 0usize;
        for slot in 0..TIE_SLOTS {
            if !w[slot].is_finite() {
                return Err(format!("tie {tie} (vertex {held}) carries the weight {}", w[slot]));
            }
            if w[slot] == 0.0 {
                continue;
            }
            used += 1;
            sum += w[slot];
            let g = row[slot] as usize;
            if g >= vertices {
                return Err(format!(
                    "tie {tie} (vertex {held}) names controller {g}, outside the {vertices} \
                     vertices"
                ));
            }
            if g == held as usize {
                return Err(format!("tie {tie} names its own vertex {held} as a controller"));
            }
            if (0..slot).any(|other| w[other] != 0.0 && row[other] == row[slot]) {
                return Err(format!(
                    "tie {tie} (vertex {held}) names controller {g} twice; merge the two \
                     weights into one slot"
                ));
            }
            if tie_of[g] != TIE_NONE {
                return Err(format!(
                    "tie {tie} (vertex {held}) is controlled by vertex {g}, which tie {} holds; \
                     a held vertex cannot control another tie",
                    tie_of[g]
                ));
            }
            let prop = &props[g];
            if prop.pdrd_body_index != 0 || prop.collider {
                return Err(format!(
                    "tie {tie} (vertex {held}) is controlled by vertex {g}, which belongs to a \
                     rigid (PDRD) body or a static collider"
                ));
            }
            count[g] += 1;
        }
        if used == 0 {
            return Err(format!("tie {tie} (vertex {held}) has no controller of nonzero weight"));
        }
        if (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(format!(
                "the weights of tie {tie} (vertex {held}) sum to {sum}; an affine combination \
                 sums to one"
            ));
        }
    }
    let mut ctrl_offset = Vec::with_capacity(vertices + 1);
    ctrl_offset.push(0u32);
    for v in 0..vertices {
        ctrl_offset.push(ctrl_offset[v] + count[v]);
    }
    let mut fill = ctrl_offset[..vertices].to_vec();
    let mut ctrl_entry = vec![0u32; *ctrl_offset.last().unwrap() as usize];
    // Ascending in `TIE_SLOTS * tie + slot`, because ties are walked in order.
    for tie in 0..ties {
        for slot in 0..TIE_SLOTS {
            let entry = TIE_SLOTS * tie + slot;
            if weight[entry] != 0.0 {
                let g = ctrl[entry] as usize;
                ctrl_entry[fill[g] as usize] = entry as u32;
                fill[g] += 1;
            }
        }
    }
    Ok(Topology { tie_of, ctrl_offset, ctrl_entry })
}

/// The ties' device state. `ties` is zero on every scene without one, and
/// [`State::view`] is then `None`, which is what leaves every other scene's
/// solve exactly as it was.
#[derive(Default)]
pub struct State {
    pub ties: usize,
    rows: usize,
    tie_of: StagedBuffer<u32>,
    tie_vertex: StagedBuffer<u32>,
    ctrl: StagedBuffer<u32>,
    weight: StagedBuffer<f32>,
    ctrl_offset: StagedBuffer<u32>,
    ctrl_entry: StagedBuffer<u32>,
    /// The expanded input and the unfolded output of one operator application,
    /// over the vertex rows; and their augmented twins for the Cosserat solve,
    /// whose operator also carries frame rows.
    expanded: Buffer<f32>,
    unfolded: Buffer<f32>,
    aug_expanded: Buffer<f32>,
    aug_unfolded: Buffer<f32>,
    /// A contact matrix with no entries, `rows + 1` zero offsets and one-element
    /// index and value arrays, for the reduced diagonal on an iteration that
    /// carries no contact: an absent buffer takes a real allocation.
    empty_offset: StagedBuffer<u32>,
    empty_index: StagedBuffer<u32>,
    empty_value: StagedBuffer<f32>,
    /// The operator's diagonal before the tie pass reduces it, and the
    /// position half of each Cosserat edge's Schwarz block (27 floats per
    /// slot); the second is empty without Cosserat frames.
    unreduced: Buffer<f32>,
    edge_block: Buffer<f32>,
    slots: usize,
}

/// Every handle the expand and the fold read, for one Newton iteration.
#[derive(Clone, Copy)]
pub struct View {
    pub rows: u32,
    tie_of: Handle,
    tie_vertex: Handle,
    ctrl: Handle,
    weight: Handle,
    ctrl_offset: Handle,
    ctrl_entry: Handle,
    dof_mask: Handle,
    empty_offset: Handle,
    empty_index: Handle,
    empty_value: Handle,
    unreduced: Handle,
    pub edge_block: Handle,
    /// Scratch for [`View::encode_apply_wrapped`]'s caller: the expanded input
    /// and the unfolded output over the vertex rows.
    pub expanded: Handle,
    pub unfolded: Handle,
}

unsafe fn stage<D: Device, T: ppf_cts_compute::Pod + Default>(
    device: &mut D,
    buffer: &mut StagedBuffer<T>,
    source: &[T],
    label: &'static str,
) -> FatalResult<()> {
    // ONE ELEMENT AT LEAST, so an empty array still names a real allocation.
    buffer.size(device, source.len().max(1), AllocLabel(label))?;
    buffer.at()[..source.len()].copy_from_slice(source);
    buffer.upload(device)?;
    Ok(())
}

impl State {
    /// Validate the scene's ties, stage their topology and size the scratch.
    /// `augmented` is the Cosserat solve's vector length, zero without one.
    ///
    /// # Safety
    /// `data` must be live and carry `vertices` vertex props.
    pub unsafe fn allocate<D: Device>(
        &mut self,
        device: &mut D,
        data: &DataSet,
        vertices: usize,
        augmented: usize,
    ) -> FatalResult<()> {
        let vertex: &[u32] = super::scene::slice(&data.tie_vertex);
        self.ties = vertex.len();
        self.rows = vertices;
        if self.ties == 0 {
            return Ok(());
        }
        let ctrl: &[u32] = super::scene::slice(&data.tie_ctrl);
        let weight: &[f32] = super::scene::slice(&data.tie_weight);
        let props: &[VertexProp] = super::scene::slice(&data.prop.vertex);
        let topology = build_topology(vertex, ctrl, weight, &props[..vertices.min(props.len())])
            .map_err(|why| Fatal::invariant(format!("solver driver: {why}")))?;
        stage(device, &mut self.tie_of, &topology.tie_of, "tie.tie_of")?;
        stage(device, &mut self.tie_vertex, vertex, "tie.vertex")?;
        stage(device, &mut self.ctrl, ctrl, "tie.ctrl")?;
        stage(device, &mut self.weight, weight, "tie.weight")?;
        stage(device, &mut self.ctrl_offset, &topology.ctrl_offset, "tie.ctrl_offset")?;
        stage(device, &mut self.ctrl_entry, &topology.ctrl_entry, "tie.ctrl_entry")?;
        stage(device, &mut self.empty_offset, &vec![0u32; vertices + 1], "tie.empty_offset")?;
        stage(device, &mut self.empty_index, &[0u32], "tie.empty_index")?;
        stage(device, &mut self.empty_value, &[0.0f32; 9], "tie.empty_value")?;
        self.expanded.size(device, 3 * vertices, AllocLabel("tie.expanded"))?;
        self.unreduced.size(device, 9 * vertices, AllocLabel("tie.unreduced"))?;
        self.slots = data.cosserat_edge.size as usize;
        self.edge_block.size(device, 27 * self.slots.max(1), AllocLabel("tie.edge_block"))?;
        self.unfolded.size(device, 3 * vertices, AllocLabel("tie.unfolded"))?;
        if augmented > 0 {
            self.aug_expanded.size(device, augmented, AllocLabel("tie.aug_expanded"))?;
            self.aug_unfolded.size(device, augmented, AllocLabel("tie.aug_unfolded"))?;
        }
        ::log::info!(
            "Tie constraints hold {} vertices on {} controller slots; the reduced solve \
             has {} fewer vertex rows",
            self.ties,
            topology.ctrl_entry.len(),
            self.ties
        );
        Ok(())
    }

    /// The Cosserat solve's augmented scratch, `None` without a tie or
    /// without frames.
    pub fn augmented_scratch(&self) -> Option<(&Buffer<f32>, &Buffer<f32>)> {
        if self.ties == 0 || self.aug_expanded.len() == 0 {
            return None;
        }
        Some((&self.aug_expanded, &self.aug_unfolded))
    }

    /// The handles one Newton iteration reads, or `None` without a tie.
    pub fn view(&mut self, dof_mask: Handle) -> Option<View> {
        if self.ties == 0 {
            return None;
        }
        Some(View {
            rows: self.rows as u32,
            tie_of: self.tie_of.handle(),
            tie_vertex: self.tie_vertex.handle(),
            ctrl: self.ctrl.handle(),
            weight: self.weight.handle(),
            ctrl_offset: self.ctrl_offset.handle(),
            ctrl_entry: self.ctrl_entry.handle(),
            dof_mask,
            empty_offset: self.empty_offset.handle(),
            empty_index: self.empty_index.handle(),
            empty_value: self.empty_value.handle(),
            unreduced: self.unreduced.handle(),
            edge_block: self.edge_block.handle(),
            expanded: self.expanded.handle(),
            unfolded: self.unfolded.handle(),
        })
    }
}

impl View {
    /// `output = E input` over the vertex rows. The two must not alias.
    ///
    /// # Safety
    /// Both handles must hold at least `3 * rows` floats and outlive the
    /// dispatch.
    pub unsafe fn encode_expand(
        &self,
        encoder: &mut dyn Encoder,
        input: Handle,
        output: Handle,
    ) -> Result<(), Fault> {
        let args = TieExpandRowArgs {
            tie_of: self.tie_of,
            ctrl: self.ctrl,
            weight: self.weight,
            input,
            output,
            count: self.rows,
            seam_arena_count: 0,
        };
        encoder.elements(&args, self.rows)
    }

    /// `output = F input` over the vertex rows. The two must not alias.
    ///
    /// # Safety
    /// As [`View::encode_expand`].
    pub unsafe fn encode_fold(
        &self,
        encoder: &mut dyn Encoder,
        input: Handle,
        output: Handle,
    ) -> Result<(), Fault> {
        let args = TieFoldRowArgs {
            tie_of: self.tie_of,
            tie_vertex: self.tie_vertex,
            weight: self.weight,
            ctrl_offset: self.ctrl_offset,
            ctrl_entry: self.ctrl_entry,
            dof_mask: self.dof_mask,
            input,
            output,
            count: self.rows,
            seam_arena_count: 0,
        };
        encoder.elements(&args, self.rows)
    }

    /// Zero every held row of `vector`, in place.
    ///
    /// # Safety
    /// As [`View::encode_expand`].
    pub unsafe fn encode_zero(&self, encoder: &mut dyn Encoder, vector: Handle) -> Result<(), Fault> {
        let args = TieZeroRowArgs {
            tie_of: self.tie_of,
            vector,
            count: self.rows,
            seam_arena_count: 0,
        };
        encoder.elements(&args, self.rows)
    }

    /// The reduced block-Jacobi diagonal: each free controller's block
    /// becomes the exact diagonal block of `E^T K E`, read from the operator's
    /// two matrices and its diagonal (`solver/tie.kernel.cpp`). Run after the
    /// operator's diagonal is formed and before it is inverted.
    ///
    /// # Safety
    /// `diagonal` must hold `9 * rows` floats and `op` name live matrices.
    pub unsafe fn precond<D: Device>(
        &self,
        device: &mut D,
        op: &super::operator::Operator,
        diagonal: Handle,
    ) -> Result<(), Fault> {
        let (dyn_index, dyn_value, dyn_offset, ref_index, ref_value, ref_offset, global) =
            self.dynamic(op);
        let args = TiePrecondRowArgs {
            tie_vertex: self.tie_vertex,
            weight: self.weight,
            ctrl_offset: self.ctrl_offset,
            ctrl_entry: self.ctrl_entry,
            dof_mask: self.dof_mask,
            index: op.fixed.index,
            offset: op.fixed.offset,
            value: op.fixed.value,
            transpose_pair: op.fixed.transpose_pair,
            transpose_offset: op.fixed.transpose_offset,
            dyn_index,
            dyn_value,
            dyn_offset,
            dyn_reference_index: ref_index,
            dyn_reference_value: ref_value,
            dyn_reference_offset: ref_offset,
            dyn_global_value: global,
            diagonal,
            count: self.rows,
            seam_arena_count: 0,
        };
        device.copy(self.unreduced, 0, diagonal, 0, 9 * self.rows as usize * size_of::<f32>())?;
        device.launch("tie.precond", &args, self.rows)?;
        Ok(())
    }

    /// The position half of every Cosserat edge's Schwarz block, read by
    /// `cosserat_schwarz_factor_row` in place of the vertex diagonal and the
    /// fixed coupling. Run after [`View::precond`].
    ///
    /// # Safety
    /// `edge_nodes` must hold two vertices per slot and `diagonal` be the
    /// reduced diagonal [`View::precond`] wrote.
    pub unsafe fn edge_blocks<D: Device>(
        &self,
        device: &mut D,
        op: &super::operator::Operator,
        edge_nodes: Handle,
        slots: u32,
        diagonal: Handle,
    ) -> Result<(), Fault> {
        if slots == 0 {
            return Ok(());
        }
        let (dyn_index, dyn_value, dyn_offset, ref_index, ref_value, ref_offset, global) =
            self.dynamic(op);
        let args = TieEdgeBlockRowArgs {
            edge_nodes,
            tie_of: self.tie_of,
            tie_vertex: self.tie_vertex,
            weight: self.weight,
            ctrl_offset: self.ctrl_offset,
            ctrl_entry: self.ctrl_entry,
            dof_mask: self.dof_mask,
            index: op.fixed.index,
            offset: op.fixed.offset,
            value: op.fixed.value,
            transpose_pair: op.fixed.transpose_pair,
            transpose_offset: op.fixed.transpose_offset,
            dyn_index,
            dyn_value,
            dyn_offset,
            dyn_reference_index: ref_index,
            dyn_reference_value: ref_value,
            dyn_reference_offset: ref_offset,
            dyn_global_value: global,
            reduced: diagonal,
            unreduced: self.unreduced,
            edge_block: self.edge_block,
            count: slots,
            seam_arena_count: 0,
        };
        device.launch("tie.edge_blocks", &args, slots)?;
        Ok(())
    }

    /// The contact matrix's seven handles, or the empty matrix without one.
    #[allow(clippy::type_complexity)]
    fn dynamic(
        &self,
        op: &super::operator::Operator,
    ) -> (Handle, Handle, Handle, Handle, Handle, Handle, Handle) {
        match op.dynamic {
            Some(d) => (d.index, d.value, d.offset, d.transpose_index, d.transpose_value,
                        d.transpose_offset, d.value),
            None => (self.empty_index, self.empty_value, self.empty_offset, self.empty_index,
                     self.empty_index, self.empty_offset, self.empty_value),
        }
    }

    /// Write each held row of the seed `dx` with its consistency term.
    ///
    /// # Safety
    /// `eval_x` must hold `rows` positions and `dx` `3 * rows` floats.
    pub unsafe fn seed<D: Device>(&self, device: &mut D, eval_x: Handle, dx: Handle) -> Result<(), Fault> {
        let args = TieConsistencyRowArgs {
            tie_of: self.tie_of,
            ctrl: self.ctrl,
            weight: self.weight,
            eval_x,
            dx,
            count: self.rows,
            seam_arena_count: 0,
        };
        device.launch("tie.seed", &args, self.rows)?;
        Ok(())
    }

    /// Fold the right-hand side in place, through the scratch: each held row's
    /// force joins its free controllers' rows and the held row becomes zero.
    ///
    /// # Safety
    /// `force` must hold `3 * rows` floats.
    pub unsafe fn fold_force<D: Device>(&self, device: &mut D, force: Handle) -> Result<(), Fault> {
        let unfolded = self.unfolded;
        let rows = self.rows as usize;
        device.run("tie.fold_force", |encoder| {
            // Safety: both buffers hold `3 * rows` floats and outlive the call.
            unsafe { self.encode_fold(encoder, force, unfolded) }
        })?;
        device.copy(force, 0, unfolded, 0, 3 * rows * size_of::<f32>())?;
        Ok(())
    }

    /// Fill each held row of the solved correction from its controllers.
    ///
    /// # Safety
    /// `dx` must hold `3 * rows` floats.
    pub unsafe fn finish<D: Device>(&self, device: &mut D, dx: Handle) -> Result<(), Fault> {
        let args = TieFinishRowArgs {
            tie_of: self.tie_of,
            ctrl: self.ctrl,
            weight: self.weight,
            dx,
            count: self.rows,
            seam_arena_count: 0,
        };
        device.launch("tie.finish", &args, self.rows)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(n: usize) -> Vec<VertexProp> {
        (0..n).map(|_| VertexProp::default()).collect()
    }

    fn row(entries: &[(u32, f32)]) -> (Vec<u32>, Vec<f32>) {
        let mut ctrl = vec![entries[0].0; TIE_SLOTS];
        let mut weight = vec![0.0; TIE_SLOTS];
        for (slot, &(g, w)) in entries.iter().enumerate() {
            ctrl[slot] = g;
            weight[slot] = w;
        }
        (ctrl, weight)
    }

    #[test]
    fn the_transposed_list_names_every_nonzero_slot_in_order() {
        let (c0, w0) = row(&[(0, 0.5), (1, 0.5)]);
        let (c1, w1) = row(&[(1, 2.0), (2, -1.0)]);
        let ctrl = [c0, c1].concat();
        let weight = [w0, w1].concat();
        let top = build_topology(&[3, 4], &ctrl, &weight, &props(5)).unwrap();
        assert_eq!(top.tie_of, vec![TIE_NONE, TIE_NONE, TIE_NONE, 0, 1]);
        assert_eq!(top.ctrl_offset, vec![0, 1, 3, 4, 4, 4]);
        assert_eq!(top.ctrl_entry, vec![0, 1, 8, 9]);
    }

    #[test]
    fn a_chain_a_pinned_held_vertex_and_a_bad_sum_are_refused() {
        let (c0, w0) = row(&[(0, 1.0)]);
        let (c1, w1) = row(&[(1, 1.0)]);
        let chain = build_topology(&[1, 2], &[c0.clone(), c1].concat(), &[w0.clone(), w1].concat(), &props(3));
        assert!(chain.unwrap_err().contains("cannot control another tie"));

        let mut pinned = props(2);
        pinned[1].fix_index = 1;
        assert!(build_topology(&[1], &c0, &w0, &pinned).unwrap_err().contains("pin"));

        let (c, w) = row(&[(0, 0.5), (1, 0.4)]);
        assert!(build_topology(&[2], &c, &w, &props(3)).unwrap_err().contains("sum to"));

        let (c, w) = row(&[(0, 0.5), (0, 0.5)]);
        assert!(build_topology(&[2], &c, &w, &props(3)).unwrap_err().contains("twice"));

        let (c, w) = row(&[(2, 1.0)]);
        assert!(build_topology(&[2], &c, &w, &props(3)).unwrap_err().contains("own vertex"));
    }

    /// A small scene's ties staged on the host backend: vertex 5 held by
    /// (0, 1, 2, 3), vertex 6 by (1, 4), vertex 4 prescribed.
    fn staged(device: &mut ppf_cts_compute::HostDevice) -> (State, Buffer<u32>, usize) {
        let n = 7usize;
        let (c0, w0) = row(&[(0, 0.4), (1, 0.3), (2, -0.2), (3, 0.5)]);
        let (c1, w1) = row(&[(1, 1.5), (4, -0.5)]);
        let (ctrl, weight) = ([c0, c1].concat(), [w0, w1].concat());
        let vertex = [5u32, 6];
        let mut p = props(n);
        p[4].fix_index = 1;
        let top = build_topology(&vertex, &ctrl, &weight, &p).unwrap();
        let mut state = State { ties: 2, rows: n, ..State::default() };
        unsafe {
            stage(device, &mut state.tie_of, &top.tie_of, "t.tie_of").unwrap();
            stage(device, &mut state.tie_vertex, &vertex, "t.vertex").unwrap();
            stage(device, &mut state.ctrl, &ctrl, "t.ctrl").unwrap();
            stage(device, &mut state.weight, &weight, "t.weight").unwrap();
            stage(device, &mut state.ctrl_offset, &top.ctrl_offset, "t.offset").unwrap();
            stage(device, &mut state.ctrl_entry, &top.ctrl_entry, "t.entry").unwrap();
        }
        state.expanded.size(device, 3 * n, AllocLabel("t.expanded")).unwrap();
        state.unfolded.size(device, 3 * n, AllocLabel("t.unfolded")).unwrap();
        let mut mask: Buffer<u32> = Buffer::default();
        mask.size(device, n, AllocLabel("t.mask")).unwrap();
        mask.write(device, 0, &[0, 0, 0, 0, 1, 0, 0]).unwrap();
        (state, mask, n)
    }

    fn pseudo(seed: u32, n: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2654435761).max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// THE FOLD IS THE EXPAND'S TRANSPOSE ON THE SUBSPACE THE SOLVE STAYS IN,
    /// which is what makes `F K E` symmetric: `<E u, y> = <u, F y>` for every
    /// `u` whose held and prescribed rows are zero, and the fold leaves a
    /// prescribed row exactly as it was and zeroes a held one.
    ///
    #[test]
    fn the_fold_is_the_transpose_of_the_expand() {
        let mut device = crate::driver::launch::host_device();
        let (mut state, mask, n) = staged(&mut device);
        let view = state.view(mask.handle()).unwrap();
        let mut input: Buffer<f32> = Buffer::default();
        let mut output: Buffer<f32> = Buffer::default();
        input.size(&mut device, 3 * n, AllocLabel("t.in")).unwrap();
        output.size(&mut device, 3 * n, AllocLabel("t.out")).unwrap();
        for trial in 0..8u32 {
            let mut u = pseudo(2 * trial + 1, 3 * n);
            for v in [4usize, 5, 6] {
                u[3 * v..3 * v + 3].fill(0.0);
            }
            let y = pseudo(2 * trial + 2, 3 * n);
            let (i, o) = (input.handle(), output.handle());
            input.write(&mut device, 0, &u).unwrap();
            device
                .run("t.expand", |e| unsafe { view.encode_expand(e, i, o) })
                .unwrap();
            let mut eu = vec![0.0f32; 3 * n];
            output.read(&mut device, 0, &mut eu).unwrap();
            input.write(&mut device, 0, &y).unwrap();
            device.run("t.fold", |e| unsafe { view.encode_fold(e, i, o) }).unwrap();
            let mut fy = vec![0.0f32; 3 * n];
            output.read(&mut device, 0, &mut fy).unwrap();
            let lhs: f64 = eu.iter().zip(&y).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
            let rhs: f64 = u.iter().zip(&fy).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
            assert!((lhs - rhs).abs() < 1.0e-5 * (1.0 + lhs.abs()), "trial {trial}: {lhs} vs {rhs}");
            // The held rows of E u are their controllers' combinations.
            for k in 0..3 {
                let want = 0.4 * u[k] + 0.3 * u[3 + k] - 0.2 * u[6 + k] + 0.5 * u[9 + k];
                assert!((eu[15 + k] - want).abs() < 1.0e-6);
            }
            assert_eq!(&fy[12..15], &y[12..15], "a prescribed row is not folded into");
            assert!(fy[15..21].iter().all(|v| *v == 0.0), "a held row folds to zero");
            for k in 0..3 {
                let want = y[3 + k] + 0.3 * y[15 + k] + 1.5 * y[18 + k];
                assert!((fy[3 + k] - want).abs() < 1.0e-6);
            }
        }
    }

    /// The finish fills a held row from its controllers on top of the offset
    /// the seed left there.
    #[test]
    fn the_finish_fills_a_held_row() {
        let mut device = crate::driver::launch::host_device();
        let (mut state, mask, n) = staged(&mut device);
        let view = state.view(mask.handle()).unwrap();
        let mut dx: Buffer<f32> = Buffer::default();
        dx.size(&mut device, 3 * n, AllocLabel("t.dx")).unwrap();
        let mut d = pseudo(9, 3 * n);
        d[15..18].copy_from_slice(&[0.01, 0.02, 0.03]);
        dx.write(&mut device, 0, &d).unwrap();
        unsafe { view.finish(&mut device, dx.handle()) }.unwrap();
        let mut out = vec![0.0f32; 3 * n];
        dx.read(&mut device, 0, &mut out).unwrap();
        for k in 0..3 {
            let want = 0.01 * (k as f32 + 1.0) + 0.4 * d[k] + 0.3 * d[3 + k] - 0.2 * d[6 + k] + 0.5 * d[9 + k];
            assert!((out[15 + k] - want).abs() < 1.0e-6);
            let want = 1.5 * d[3 + k] - 0.5 * d[12 + k];
            assert!((out[18 + k] - (d[18 + k] + want)).abs() < 1.0e-6);
        }
        assert_eq!(&out[..15], &d[..15]);

    }

    /// THE REDUCED DIAGONAL IS THE EXACT DIAGONAL BLOCK OF `E^T K E`, checked
    /// against a dense product over a small symmetric operator stored as the
    /// solver stores it: an upper-triangle CSR with its transpose run and a
    /// diagonal array. The couplings that matter are the ones a held pair
    /// shares (vertices 5 and 6 share controller 1) and a held vertex's
    /// coupling to its own controller (5 and 0).
    #[test]
    fn the_reduced_diagonal_is_the_exact_block_of_the_reduced_operator() {
        let mut device = crate::driver::launch::host_device();
        let (mut state, mask, n) = staged(&mut device);
        let r = pseudo(77, 9 * 64);
        let mut k = vec![[[0.0f64; 3]; 3]; n * n];
        let mut pairs: Vec<(usize, usize)> = vec![(0, 5), (5, 6), (1, 2), (1, 6), (3, 5), (2, 6)];
        pairs.sort();
        let mut seed = 0usize;
        let mut next = || {
            seed += 1;
            f64::from(r[seed % r.len()])
        };
        for &(i, j) in &pairs {
            for a in 0..3 {
                for b in 0..3 {
                    let v = next();
                    k[i * n + j][a][b] = v;
                    k[j * n + i][b][a] = v;
                }
            }
        }
        let mut diag_blocks = vec![0.0f32; 9 * n];
        for i in 0..n {
            for a in 0..3 {
                for b in 0..3 {
                    let v = if a == b { 10.0 } else { 0.5 * (next() + 0.0) };
                    k[i * n + i][a][b] += v;
                }
            }
            // Symmetrize the diagonal block.
            for a in 0..3 {
                for b in 0..a {
                    let m = 0.5 * (k[i * n + i][a][b] + k[i * n + i][b][a]);
                    k[i * n + i][a][b] = m;
                    k[i * n + i][b][a] = m;
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    diag_blocks[9 * i + 3 * b + a] = k[i * n + i][a][b] as f32;
                }
            }
        }
        // Upper triangle CSR, column-major blocks, and the transpose run.
        let (mut index, mut offset, mut value) = (Vec::new(), vec![0u32], Vec::new());
        let mut lower: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for i in 0..n {
            for &(a, b) in &pairs {
                if a == i {
                    let slot = index.len() as u32;
                    index.push(b as u32);
                    for c in 0..3 {
                        for rr in 0..3 {
                            value.push(k[a * n + b][rr][c] as f32);
                        }
                    }
                    lower[b].push((a as u32, slot));
                }
            }
            offset.push(index.len() as u32);
        }
        let mut tpair = Vec::new();
        let mut toff = vec![0u32];
        for row in lower.iter() {
            for &(src, slot) in row {
                tpair.push(src);
                tpair.push(slot);
            }
            toff.push((tpair.len() / 2) as u32);
        }
        let mut bufs_u: Vec<Buffer<u32>> = Vec::new();
        for data in [&index, &offset, &tpair, &toff] {
            let mut b: Buffer<u32> = Buffer::default();
            b.size(&mut device, data.len().max(1), AllocLabel("t.csr")).unwrap();
            b.write(&mut device, 0, data).unwrap();
            bufs_u.push(b);
        }
        let mut val: Buffer<f32> = Buffer::default();
        val.size(&mut device, value.len(), AllocLabel("t.val")).unwrap();
        val.write(&mut device, 0, &value).unwrap();
        state.unreduced.size(&mut device, 9 * n, AllocLabel("t.unreduced")).unwrap();
        let mut zero_diag: Buffer<f32> = Buffer::default();
        zero_diag.size(&mut device, 9 * n, AllocLabel("t.zd")).unwrap();
        let op = crate::driver::operator::Operator {
            dynamic: None,
            fixed: crate::driver::spmv::FixedCsrView {
                index: bufs_u[0].handle(),
                offset: bufs_u[1].span(0, n + 1),
                value: val.handle(),
                transpose_pair: bufs_u[2].handle(),
                transpose_offset: bufs_u[3].span(0, n + 1),
                rows: n as u32,
            },
            diagonal: zero_diag.handle(),
            difference: None,
            tie: None,
        };
        // Allocate the empty contact matrix the precond pass reads.
        unsafe {
            stage(&mut device, &mut state.empty_offset, &vec![0u32; n + 1], "t.eo").unwrap();
            stage(&mut device, &mut state.empty_index, &[0u32], "t.ei").unwrap();
            stage(&mut device, &mut state.empty_value, &[0.0f32; 9], "t.ev").unwrap();
        }
        let view = state.view(mask.handle()).unwrap();
        let mut diag: Buffer<f32> = Buffer::default();
        diag.size(&mut device, 9 * n, AllocLabel("t.diag")).unwrap();
        diag.write(&mut device, 0, &diag_blocks).unwrap();
        unsafe { view.precond(&mut device, &op, diag.handle()) }.unwrap();
        let mut got = vec![0.0f32; 9 * n];
        diag.read(&mut device, 0, &mut got).unwrap();
        // Dense E: vertex 5 = .4 x0 + .3 x1 - .2 x2 + .5 x3, vertex 6 = 1.5 x1 - .5 x4.
        let mut e = vec![0.0f64; n * n];
        for v in 0..5 {
            e[v * n + v] = 1.0;
        }
        for (g, w) in [(0, 0.4), (1, 0.3), (2, -0.2), (3, 0.5)] {
            e[5 * n + g] = w;
        }
        for (g, w) in [(1, 1.5), (4, -0.5)] {
            e[6 * n + g] = w;
        }
        for g in [0usize, 1, 2, 3] {
            for a in 0..3 {
                for b in 0..3 {
                    let mut want = 0.0f64;
                    for u in 0..n {
                        for v in 0..n {
                            want += e[u * n + g] * e[v * n + g] * k[u * n + v][a][b];
                        }
                    }
                    let have = f64::from(got[9 * g + 3 * b + a]);
                    assert!((have - want).abs() < 1.0e-4 * (1.0 + want.abs()),
                            "controller {g} ({a},{b}): {have} vs {want}");
                }
            }
        }
        // Vertex 4 is prescribed and 5, 6 are held: untouched.
        assert_eq!(&got[36..63], &diag_blocks[36..63]);

        // The Cosserat edge blocks: (0, 1) and (1, 2) take the reduced
        // coupling, (5, 6) has held ends and takes the unreduced blocks.
        let nodes: Vec<u32> = vec![0, 1, 1, 2, 5, 6, 1, 4];
        let mut node_buf: Buffer<u32> = Buffer::default();
        node_buf.size(&mut device, 8, AllocLabel("t.nodes")).unwrap();
        node_buf.write(&mut device, 0, &nodes).unwrap();
        state.edge_block.size(&mut device, 27 * 4, AllocLabel("t.eb")).unwrap();
        let view = state.view(mask.handle()).unwrap();
        unsafe { view.edge_blocks(&mut device, &op, node_buf.handle(), 4, diag.handle()) }.unwrap();
        let mut eb = vec![0.0f32; 108];
        state.edge_block.read(&mut device, 0, &mut eb).unwrap();
        for (slot, (u, v)) in [(0usize, 1usize), (1, 2)].into_iter().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    let mut want = 0.0f64;
                    for x in 0..n {
                        for y in 0..n {
                            want += e[x * n + u] * e[y * n + v] * k[x * n + y][a][b];
                        }
                    }
                    let have = f64::from(eb[27 * slot + 18 + 3 * b + a]);
                    assert!((have - want).abs() < 1.0e-4 * (1.0 + want.abs()),
                            "edge ({u},{v}) ({a},{b}): {have} vs {want}");
                    assert_eq!(eb[27 * slot + 3 * b + a], got[9 * u + 3 * b + a]);
                }
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(eb[54 + 3 * b + a], diag_blocks[45 + 3 * b + a]);
                assert!((f64::from(eb[54 + 18 + 3 * b + a]) - k[5 * n + 6][a][b]).abs() < 1.0e-6);
                // (1, 4): vertex 4 is prescribed, so no reduced coupling is
                // added to the (absent, here zero) fixed one.
                assert_eq!(eb[81 + 18 + 3 * b + a], 0.0);
            }
        }
    }

    #[test]
    fn a_pinned_controller_is_admitted() {
        let mut pinned = props(3);
        pinned[0].fix_index = 1;
        let (c, w) = row(&[(0, 0.25), (1, 0.75)]);
        assert!(build_topology(&[2], &c, &w, &pinned).is_ok());
    }
}
