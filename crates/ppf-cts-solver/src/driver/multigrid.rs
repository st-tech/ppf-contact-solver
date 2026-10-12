// File: crates/ppf-cts-solver/src/driver/multigrid.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The geometric multigrid preconditioner's driver half.
//!
//! [`super::multigrid_grid`] lays the hierarchy over the scene's authored
//! positions, on the host, once; this module carries it to the device, brings
//! every level's operator and smoother up to date with the Newton operator, and
//! encodes one V-cycle into the conjugate gradient's own region.
//!
//! # Three schedules
//!
//! - **Once a run**, [`install`], straight after `initialize()` accepts the
//!   scene: the grids, the interpolations and the coarse operators' block
//!   patterns, which depend on the mesh and on its authored positions and on
//!   nothing a step or a resume changes. Every device allocation this
//!   preconditioner makes is made here, so a solve allocates nothing and a
//!   scene too large for its levels is reported before a frame is written.
//! - **Once a solve**, [`refresh`]: the coarse operators' values and every
//!   level's smoother blocks, which depend on the Newton operator's values and
//!   so change at every Newton step. One submission, no host round trip, and
//!   three dispatches a level: `A P` a finer row at a time, `P^T (A P)` a
//!   coarse block at a time, and the smoother a node at a time.
//! - **Once an iteration**, [`encode_cycle`]: the cycle itself, a fixed list of
//!   dispatches that the conjugate gradient records with the rest of its
//!   iteration and replays.
//!
//! # The cycle
//!
//! On each level, from the finest: one sweep from zero, the residual, its
//! restriction to the level below, that level's cycle, the correction
//! interpolated back and one more sweep. The deepest level is one sweep. The
//! sweep from zero of every coarse level is taken by the restriction that
//! fills its right-hand side, so a coarse level costs four dispatches and the
//! deepest one none of its own. A
//! sweep is `x + step M^-1 (b - A x)` with `M` the block bound
//! `multigrid.kernel.cpp` describes, which makes it a contraction for any step
//! below two, and the same sweep runs before and after the correction, so the
//! cycle is a symmetric positive definite operator whatever the Newton operator
//! holds. The step is [`STEP`] on a row that has a level below it and one on a
//! row that has none.
//!
//! The finest level's operator is the Newton operator itself, applied by the
//! same entry the conjugate gradient applies it with. Only the coarse levels
//! hold a matrix of their own.

use ppf_cts_compute::{AllocLabel, Buffer, Device, Encoder, EncoderExt, Fault, Handle};

use super::kernels::{
    MultigridFineProlongRowArgs, MultigridFineRestrictRowArgs, MultigridFineSmootherRowArgs,
    MultigridFineSweepRowArgs, MultigridFirstSweepRowArgs, MultigridGalerkinSlotArgs,
    MultigridProductFineRowArgs, MultigridProductRowArgs, MultigridProlongRowArgs,
    MultigridResidualRowArgs, MultigridRestrictRowArgs, MultigridSmootherRowArgs,
    MultigridSweepRowArgs,
};
use super::multigrid_grid::{self, Hierarchy, Mesh, Pattern, WIDTH};
use super::operator::Operator;
use super::scene::{Fatal, FatalResult};
use crate::data::DataSet;
use super::schwarz::OperatorRows;

/// The step of a sweep on a row that has a level below it.
///
/// A sweep is `x + step M^-1 (b - A x)` and the eigenvalues of `M^-1 A` lie in
/// `(0, 1]`. An elastic operator's eigenvalue grows with the square of a
/// component's frequency, so a grid of twice the spacing resolves the
/// components below about a quarter of the largest and the sweep's job is the
/// rest, `[1/4, 1]`. The one step that damps both ends of that interval equally
/// is `2 / (1/4 + 1) = 1.6`, the first-degree Chebyshev step, which leaves at
/// most `0.6` of any component in it. It is below two, so the sweep contracts
/// on the whole range, whatever the operator.
pub const STEP: f32 = 1.6;

/// One coarse level on the device.
#[derive(Default)]
struct Level {
    /// Nodes of this level.
    nodes: u32,
    /// The interpolation: [`WIDTH`] nodes and weights per finer row.
    node_of: Buffer<u32>,
    weight_of: Buffer<f32>,
    /// The same weights per node.
    gather_offset: Buffer<u32>,
    gather_row: Buffer<u32>,
    gather_weight: Buffer<f32>,
    /// `A P`, the product's first pass: a fixed block pattern over the finer
    /// rows and the values of this solve.
    reach_offset: Buffer<u32>,
    reach_column: Buffer<u32>,
    reach_value: Buffer<f32>,
    /// The operator: a fixed block pattern, each block's own row, and the
    /// values of this solve.
    offset: Buffer<u32>,
    column: Buffer<u32>,
    slot_node: Buffer<u32>,
    value: Buffer<f32>,
    blocks: u32,
    /// The smoother: each row's step, and the inverse scaled by it.
    step: Buffer<f32>,
    inverse: Buffer<f32>,
    /// The cycle's vectors, three floats a node.
    right: Buffer<f32>,
    x: Buffer<f32>,
    y: Buffer<f32>,
    residual: Buffer<f32>,
}

/// Everything the preconditioner owns.
#[derive(Default)]
pub struct State {
    /// The vertex count this was built for; zero before the first build.
    rows: u32,
    built: bool,
    levels: Vec<Level>,
    /// The finest level's smoother, and the contact matrix's bound the first
    /// coarse operator is handed.
    contact: Buffer<f32>,
    step: Buffer<f32>,
    inverse: Buffer<f32>,
    /// The finest level's iterate and the operator applied to it.
    x: Buffer<f32>,
    product: Buffer<f32>,
    /// The operator's entry folds two scalars this preconditioner has no use
    /// for; these receive them.
    fold_a: Buffer<f32>,
    fold_b: Buffer<f32>,
    /// The operator's rows summed, which its difference form reads.
    row_sum: Buffer<f32>,
    /// A zero offset array of `rows + 1`, for a scene with no contact matrix.
    empty_offset: Buffer<u32>,
    /// The mask of removed vertices the last [`refresh`] was given, which the
    /// cycle's two transfers at the finest level read.
    removed: Option<Handle>,
}

impl State {
    pub fn is_built_for(&self, vertices: u32) -> bool {
        self.built && self.rows == vertices
    }

    /// Refuse a solve that reached the multigrid with no hierarchy laid for it.
    ///
    /// The host lays the hierarchy once `initialize()` has accepted a scene
    /// that names this preconditioner, and a solve never does: laying it is a
    /// host pass over the whole mesh plus every allocation the preconditioner
    /// makes, neither of which belongs inside a Newton step. So a solve that
    /// finds none was handed a preconditioner the scene did not name when it
    /// was accepted, and it says so instead of building one late.
    pub fn require_built(&self, vertices: u32) -> FatalResult<()> {
        if self.is_built_for(vertices) {
            return Ok(());
        }
        Err(Fatal::invariant(format!(
            "solver driver: this solve is preconditioned by the multigrid and no hierarchy was \
             laid for its {vertices} vertices ({}). The host lays it once, after initialize(), \
             for a scene that names the multigrid, so either that call was skipped or the \
             preconditioner changed after the scene was accepted",
            if self.built {
                format!("the one in hand covers {}", self.rows)
            } else {
                String::from("none was laid at all")
            }
        )))
    }

    /// Coarse levels below the finest.
    #[cfg(test)]
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// The two buffers the operator's difference form reads: each row's blocks
    /// summed, which [`Operator::sum_rows`] fills every solve, and the stand-in
    /// offsets. [`build`] sized both.
    ///
    /// THE MULTIGRID IS WHY THE OPERATOR IS APPLIED IN THAT FORM. Its coarse
    /// correction moves whole pieces of mesh together, and the plain product
    /// of the operator with such a vector cancels down to its own round-off,
    /// so the conjugate gradient cannot tell the curvature of the direction it
    /// was just handed from zero. `solver/spmv.kernel.cpp` carries the numbers.
    pub fn difference(&self) -> super::operator::DifferenceView {
        super::operator::DifferenceView {
            row_sum: self.row_sum.handle(),
            empty_offset: self.empty_offset.handle(),
        }
    }

    /// The stand-in offset array, which the operator's view names for every
    /// span this scene does not have: `rows + 1` zeros, written by [`build`].
    pub fn empty_offsets(&self) -> Handle {
        self.empty_offset.handle()
    }
}

/// Upload a slice into a buffer sized for it. An empty slice still names one
/// element, because a record never carries a handle that names nothing.
fn upload<D: Device, T: ppf_cts_compute::Pod + Default + Clone>(
    device: &mut D,
    buffer: &mut Buffer<T>,
    data: &[T],
    label: &'static str,
) -> Result<(), Fault> {
    buffer.size(device, data.len().max(1), AllocLabel(label))?;
    if data.is_empty() {
        buffer.write(device, 0, &[T::default()])
    } else {
        buffer.write(device, 0, data)
    }
}

/// Per row, [`STEP`] where the row interpolates from a level below and one
/// where it does not.
fn steps(rows: usize, below: Option<&multigrid_grid::Transfer>) -> Vec<f32> {
    (0..rows)
        .map(|row| match below {
            Some(t) if t.weight[WIDTH * row..WIDTH * (row + 1)].iter().any(|w| *w > 0.0) => STEP,
            _ => 1.0,
        })
        .collect()
}

/// Carry a hierarchy to the device and size everything a cycle needs.
///
/// ONCE A RUN. Every allocation the preconditioner makes is made here, and a
/// buffer has no destructor, so calling this again for a scene already built
/// would strand the levels it replaces; [`State::is_built_for`] is what a
/// caller asks first.
///
/// # Safety
/// `vertices` must be the Newton operator's row count and the hierarchy must
/// have been laid over a pattern of that many rows.
pub unsafe fn build<D: Device>(
    device: &mut D,
    state: &mut State,
    hierarchy: &Hierarchy,
    vertices: u32,
) -> Result<(), Fault> {
    let n = vertices as usize;
    state.rows = vertices;
    state.built = true;
    state.levels.clear();
    state.contact.size(device, (9 * n).max(1), AllocLabel("multigrid.contact"))?;
    state.inverse.size(device, (9 * n).max(1), AllocLabel("multigrid.inverse"))?;
    state.x.size(device, (3 * n).max(1), AllocLabel("multigrid.x"))?;
    state.product.size(device, (3 * n).max(1), AllocLabel("multigrid.product"))?;
    state.fold_a.size(device, n.max(1), AllocLabel("multigrid.fold_a"))?;
    state.fold_b.size(device, n.max(1), AllocLabel("multigrid.fold_b"))?;
    state.row_sum.size(device, (9 * n).max(1), AllocLabel("multigrid.row_sum"))?;
    upload(device, &mut state.empty_offset, &vec![0u32; n + 1], "multigrid.empty")?;
    let below = hierarchy.levels.first().map(|l| &l.transfer);
    upload(device, &mut state.step, &steps(n, below), "multigrid.step")?;
    for (depth, plan) in hierarchy.levels.iter().enumerate() {
        let t = &plan.transfer;
        let nodes = t.nodes as usize;
        let above = if depth == 0 { n } else { hierarchy.levels[depth - 1].transfer.nodes as usize };
        assert_eq!(t.fine as usize, above, "level {depth} does not follow the one above it");
        let mut level =
            Level { nodes: t.nodes, blocks: plan.pattern.blocks() as u32, ..Level::default() };
        upload(device, &mut level.node_of, &t.node, "multigrid.node_of")?;
        upload(device, &mut level.weight_of, &t.weight, "multigrid.weight_of")?;
        upload(device, &mut level.gather_offset, &t.gather_offset, "multigrid.gather_offset")?;
        upload(device, &mut level.gather_row, &t.gather_row, "multigrid.gather_row")?;
        upload(device, &mut level.gather_weight, &t.gather_weight, "multigrid.gather_weight")?;
        upload(device, &mut level.reach_offset, &plan.reach.offset, "multigrid.reach_offset")?;
        upload(device, &mut level.reach_column, &plan.reach.column, "multigrid.reach_column")?;
        level.reach_value.size(
            device,
            (9 * plan.reach.blocks()).max(1),
            AllocLabel("multigrid.reach_value"),
        )?;
        upload(device, &mut level.offset, &plan.pattern.offset, "multigrid.offset")?;
        upload(device, &mut level.column, &plan.pattern.column, "multigrid.column")?;
        let slot_node: Vec<u32> = (0..nodes as u32)
            .flat_map(|node| std::iter::repeat(node).take(plan.pattern.row(node as usize).len()))
            .collect();
        upload(device, &mut level.slot_node, &slot_node, "multigrid.slot_node")?;
        let blocks = plan.pattern.blocks();
        level.value.size(device, (9 * blocks).max(1), AllocLabel("multigrid.value"))?;
        level.inverse.size(device, (9 * nodes).max(1), AllocLabel("multigrid.level_inverse"))?;
        let below = hierarchy.levels.get(depth + 1).map(|l| &l.transfer);
        upload(device, &mut level.step, &steps(nodes, below), "multigrid.level_step")?;
        level.right.size(device, (3 * nodes).max(1), AllocLabel("multigrid.right"))?;
        level.x.size(device, (3 * nodes).max(1), AllocLabel("multigrid.level_x"))?;
        level.y.size(device, (3 * nodes).max(1), AllocLabel("multigrid.level_y"))?;
        level.residual.size(device, (3 * nodes).max(1), AllocLabel("multigrid.residual"))?;
        state.levels.push(level);
    }
    Ok(())
}

/// Lay the hierarchy over a scene: its positions, its elements, and the fixed
/// matrix's block pattern.
///
/// `positions` is the scene's positions, three coordinates a vertex, and
/// `upper_offset` / `upper_index` the fixed matrix's upper triangle.
pub fn plan(
    positions: &[f32],
    faces: &[[u32; 3]],
    edges: &[[u32; 2]],
    tets: &[[u32; 4]],
    upper_offset: &[u32],
    upper_index: &[u32],
) -> Hierarchy {
    let started = std::time::Instant::now();
    let positions: Vec<[f64; 3]> = positions
        .chunks_exact(3)
        .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        .collect();
    let fine = Pattern::from_upper_csr(upper_offset, upper_index);
    let mesh = Mesh { positions: &positions, faces, edges, tets };
    let hierarchy = multigrid_grid::build(&mesh, &fine);
    // THE HIERARCHY ANNOUNCES ITSELF. A run that prints this line was
    // preconditioned by these levels, and one that does not was not, which is
    // what an iteration count has to be read against.
    let nodes: Vec<u32> = hierarchy.levels.iter().map(|l| l.transfer.nodes).collect();
    let blocks: Vec<usize> = hierarchy.levels.iter().map(|l| l.pattern.blocks()).collect();
    ::log::info!(
        "multigrid laid {} levels over {} pieces of mesh ({} vertices in no element) in {:.0} ms: \
         {} vertices and {} blocks at the finest, nodes per level {:?}, blocks per level {:?}",
        hierarchy.levels.len(),
        hierarchy.pieces,
        hierarchy.loose,
        started.elapsed().as_secs_f64() * 1.0e3,
        positions.len(),
        fine.blocks(),
        nodes,
        blocks
    );
    hierarchy
}

/// Lay the hierarchy over the scene `initialize()` accepted and carry it to the
/// device.
///
/// ONCE A RUN, AND BEFORE ITS FIRST STEP. The grids are a host pass over the
/// whole mesh and the levels are every allocation this preconditioner makes.
/// Neither belongs inside a Newton step: an allocation that fails there has no
/// frame to be reported before, and the pass would be charged to the first
/// linear solve, where it reads as that solve being slow. So the host calls
/// this through `install_multigrid` once the scene is accepted, and a solve
/// only asks [`State::require_built`].
///
/// `positions` ARE THE SCENE'S AUTHORED POSITIONS, three coordinates a
/// vertex, AND NOT `data.vertex.curr`. That array is the pose the run starts
/// from, which on a resumed run is the saved one. Any pose gives a symmetric
/// positive definite preconditioner, because the grids only choose which
/// vertices a node averages; what the pose decides is how well a level's nodes
/// follow the mesh, and so what a cycle costs. A sheet authored flat gets a
/// planar grid from its authored pose and a three-dimensional one from a
/// draped pose, with more nodes to a level and a coarser first spacing, so a
/// hierarchy laid over the saved pose makes the resumed half of a run slower
/// than the same frames of the run it continues.
///
/// # Safety
/// `data` must be the live scene the solver state was sized for, and `vertices`
/// that state's vertex count.
pub unsafe fn install<D: Device>(
    device: &mut D,
    state: &mut State,
    data: &DataSet,
    positions: &[f32],
    vertices: usize,
) -> FatalResult<()> {
    let pattern =
        super::fixedcsr::borrow_pattern(&data.fixed_index_table, &data.transpose_table)?;
    // An element's index triple and the rest are each their component type
    // laid end to end, which is what lets the scene's arrays be read as flat
    // slices without a copy.
    const _: () = assert!(
        std::mem::size_of::<crate::data::Vec3u>() == 3 * std::mem::size_of::<u32>()
            && std::mem::size_of::<crate::data::Vec2u>() == 2 * std::mem::size_of::<u32>()
            && std::mem::size_of::<crate::data::Vec4u>() == 4 * std::mem::size_of::<u32>()
    );
    fn flat<'a, T, U>(data: *const T, count: usize) -> &'a [U] {
        if data.is_null() || count == 0 {
            &[]
        } else {
            // SAFETY: the caller's array holds `count` elements of `T`, and
            // `U` is `T`'s own components as an array, asserted above.
            unsafe { std::slice::from_raw_parts(data as *const U, count) }
        }
    }
    if positions.len() != 3 * vertices || pattern.rows as usize != vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the multigrid preconditioner was asked to cover {vertices} \
             vertices, and was handed {} authored coordinates and a fixed pattern of {} rows",
            positions.len(),
            pattern.rows
        )));
    }
    let mesh = &data.mesh.mesh;
    let faces: &[[u32; 3]] = flat(mesh.face.data, mesh.face.size as usize);
    let edges: &[[u32; 2]] = flat(mesh.edge.data, mesh.edge.size as usize);
    let tets: &[[u32; 4]] = flat(mesh.tet.data, mesh.tet.size as usize);
    let hierarchy = plan(positions, faces, edges, tets, pattern.offset, pattern.index);
    build(device, state, &hierarchy, vertices as u32)?;
    Ok(())
}

/// Bring every level up to date with the Newton operator as it stands now.
///
/// `block_diagonal` is the operator's whole block diagonal, the one the
/// block-Jacobi preconditioner inverts, and `removed` the per-vertex mask of
/// the degrees of freedom a pin removed.
///
/// # Safety
/// [`build`] must have run for this operator's row count, which
/// [`State::require_built`] answers, and every handle must name a live
/// allocation of the operator's arrays.
pub unsafe fn refresh<D: Device>(
    device: &mut D,
    state: &mut State,
    rows: OperatorRows,
    block_diagonal: Handle,
    removed: Handle,
) -> FatalResult<()> {
    let vertices = state.rows;
    state.removed = Some(removed);
    if vertices == 0 {
        return Ok(());
    }
    let contact = state.contact.handle();
    let step = state.step.handle();
    let inverse = state.inverse.handle();
    let levels = &state.levels;
    let result = device.run("multigrid.refresh", |encoder| {
        let fine = MultigridFineSmootherRowArgs {
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
            diagonal: block_diagonal,
            step,
            contact,
            inverse,
            count: vertices,
            seam_arena_count: 0,
        };
        encoder.elements(&fine, vertices)?;
        for (depth, level) in levels.iter().enumerate() {
            if level.nodes == 0 {
                continue;
            }
            // `A P`, one finer row a thread
            if depth == 0 {
                let product = MultigridProductFineRowArgs {
                    node_of: level.node_of.handle(),
                    weight_of: level.weight_of.handle(),
                    removed,
                    fixed_index: rows.fixed_index,
                    fixed_offset: rows.fixed_offset,
                    fixed_value: rows.fixed_value,
                    transpose_pair: rows.transpose_pair,
                    transpose_offset: rows.transpose_offset,
                    diagonal: rows.diagonal,
                    contact,
                    reach_offset: level.reach_offset.handle(),
                    reach_column: level.reach_column.handle(),
                    reach_value: level.reach_value.handle(),
                    count: vertices,
                    seam_arena_count: 0,
                };
                encoder.elements(&product, vertices)?;
            } else {
                let above = &levels[depth - 1];
                let product = MultigridProductRowArgs {
                    node_of: level.node_of.handle(),
                    weight_of: level.weight_of.handle(),
                    fine_offset: above.offset.handle(),
                    fine_column: above.column.handle(),
                    fine_value: above.value.handle(),
                    reach_offset: level.reach_offset.handle(),
                    reach_column: level.reach_column.handle(),
                    reach_value: level.reach_value.handle(),
                    count: above.nodes,
                    seam_arena_count: 0,
                };
                encoder.elements(&product, above.nodes)?;
            }
            // `P^T (A P)`, one coarse block a thread
            if level.blocks > 0 {
                let galerkin = MultigridGalerkinSlotArgs {
                    gather_offset: level.gather_offset.handle(),
                    gather_row: level.gather_row.handle(),
                    gather_weight: level.gather_weight.handle(),
                    reach_offset: level.reach_offset.handle(),
                    reach_column: level.reach_column.handle(),
                    reach_value: level.reach_value.handle(),
                    slot_node: level.slot_node.handle(),
                    column: level.column.handle(),
                    value: level.value.handle(),
                    count: level.blocks,
                    seam_arena_count: 0,
                };
                encoder.elements(&galerkin, level.blocks)?;
            }
            let smoother = MultigridSmootherRowArgs {
                offset: level.offset.handle(),
                column: level.column.handle(),
                value: level.value.handle(),
                step: level.step.handle(),
                inverse: level.inverse.handle(),
                count: level.nodes,
                seam_arena_count: 0,
            };
            encoder.elements(&smoother, level.nodes)?;
        }
        Ok(())
    });
    match result {
        Ok(_) => Ok(()),
        Err(Fault::Device { diag, .. }) => Err(Fatal::invariant(format!(
            "solver driver: the multigrid preconditioner could not be built from this Newton \
             operator: {} check(s) failed. Either a smoother block is not positive definite, \
             which means the assembled operator is NaN, infinite or indefinite and is an \
             assembly defect upstream rather than a tolerance, or a coarse operator was \
             handed a block its pattern has no slot for, which means the hierarchy was laid \
             over a different matrix than the one being solved. The first, by row: {}",
            diag.failures,
            diag.first.as_ref().map_or(String::new(), |first| format!("{first}"))
        ))),
        Err(other) => Err(Fatal::from(other)),
    }
}

/// Append one V-cycle, `result = M^-1 residual`, to a region.
///
/// # Safety
/// [`refresh`] must have run against `operator`, and `residual` and `result`
/// must each hold three floats a row and outlive the dispatches this appends.
///
/// # Panics
/// If no [`refresh`] has run, which is a caller defect: a cycle read against
/// levels that were never filled would be a preconditioner made of whatever
/// the allocations held.
pub unsafe fn encode_cycle(
    encoder: &mut dyn Encoder,
    state: &mut State,
    operator: &Operator,
    residual: Handle,
    result: Handle,
) -> Result<(), Fault> {
    let vertices = state.rows;
    if vertices == 0 {
        return Ok(());
    }
    let removed = state
        .removed
        .expect("the multigrid cycle was encoded before its levels were refreshed");
    assert_eq!(operator.rows(), vertices, "the cycle was built for another operator");
    let inverse = state.inverse.handle();
    let x = state.x.handle();
    let product = state.product.handle();
    let (fold_a, fold_b) = (state.fold_a.handle(), state.fold_b.handle());
    let first = MultigridFirstSweepRowArgs {
        inverse,
        right: residual,
        x,
        count: vertices,
        seam_arena_count: 0,
    };
    encoder.elements(&first, vertices)?;
    operator.encode_apply_folded(encoder, x, product, fold_a, fold_b)?;
    if let Some(level) = state.levels.first().filter(|l| l.nodes > 0) {
        let restrict = MultigridFineRestrictRowArgs {
            gather_offset: level.gather_offset.handle(),
            gather_row: level.gather_row.handle(),
            gather_weight: level.gather_weight.handle(),
            removed,
            right: residual,
            product,
            inverse: level.inverse.handle(),
            coarse: level.right.handle(),
            x: level.x.handle(),
            count: level.nodes,
            seam_arena_count: 0,
        };
        encoder.elements(&restrict, level.nodes)?;
        let correction = encode_level(encoder, &state.levels, 0)?;
        let prolong = MultigridFineProlongRowArgs {
            node_of: level.node_of.handle(),
            weight_of: level.weight_of.handle(),
            removed,
            coarse: correction,
            x,
            count: vertices,
            seam_arena_count: 0,
        };
        encoder.elements(&prolong, vertices)?;
        operator.encode_apply_folded(encoder, x, product, fold_a, fold_b)?;
    }
    let last = MultigridFineSweepRowArgs {
        inverse,
        right: residual,
        product,
        x,
        out: result,
        count: vertices,
        seam_arena_count: 0,
    };
    encoder.elements(&last, vertices)
}

/// The cycle from coarse level `depth` down. The restriction that filled the
/// level's `right` has already taken its first sweep into `x`. Answers the
/// buffer the level's correction is left in.
unsafe fn encode_level(
    encoder: &mut dyn Encoder,
    levels: &[Level],
    depth: usize,
) -> Result<Handle, Fault> {
    let level = &levels[depth];
    let below = match levels.get(depth + 1).filter(|l| l.nodes > 0) {
        Some(below) => below,
        // the deepest level is its one sweep
        None => return Ok(level.x.handle()),
    };
    let residual = MultigridResidualRowArgs {
        offset: level.offset.handle(),
        column: level.column.handle(),
        value: level.value.handle(),
        right: level.right.handle(),
        x: level.x.handle(),
        residual: level.residual.handle(),
        count: level.nodes,
        seam_arena_count: 0,
    };
    encoder.elements(&residual, level.nodes)?;
    let restrict = MultigridRestrictRowArgs {
        gather_offset: below.gather_offset.handle(),
        gather_row: below.gather_row.handle(),
        gather_weight: below.gather_weight.handle(),
        residual: level.residual.handle(),
        inverse: below.inverse.handle(),
        coarse: below.right.handle(),
        x: below.x.handle(),
        count: below.nodes,
        seam_arena_count: 0,
    };
    encoder.elements(&restrict, below.nodes)?;
    let correction = encode_level(encoder, levels, depth + 1)?;
    let prolong = MultigridProlongRowArgs {
        node_of: below.node_of.handle(),
        weight_of: below.weight_of.handle(),
        coarse: correction,
        x: level.x.handle(),
        count: level.nodes,
        seam_arena_count: 0,
    };
    encoder.elements(&prolong, level.nodes)?;
    let sweep = MultigridSweepRowArgs {
        offset: level.offset.handle(),
        column: level.column.handle(),
        value: level.value.handle(),
        inverse: level.inverse.handle(),
        right: level.right.handle(),
        x: level.x.handle(),
        out: level.y.handle(),
        count: level.nodes,
        seam_arena_count: 0,
    };
    encoder.elements(&sweep, level.nodes)?;
    Ok(level.y.handle())
}

#[cfg(test)]
mod tests {
    use super::super::launch::host_device;
    use super::super::operator::DynamicView;
    use super::super::pcg;
    use super::super::spmv::FixedCsrView;
    use super::*;
    use nalgebra::DMatrix;
    use ppf_cts_compute::ReadbackBuffer;
    use std::collections::BTreeMap;

    type Block = [[f64; 3]; 3];

    /// A sheet of `n x n` vertices as a network of springs along its mesh
    /// edges, in the storage the Newton operator has: a fixed matrix as an
    /// upper triangle with its transpose references, a block diagonal, and
    /// optionally a contact matrix and pinned vertices.
    ///
    /// A spring stiffens its own direction a thousand times more than the two
    /// across it, which is the shape a membrane over a soft bending term gives
    /// a real block and the one an isotropic bound would get wrong.
    struct Sheet {
        vertices: usize,
        positions: Vec<[f64; 3]>,
        faces: Vec<[u32; 3]>,
        /// The whole operator, dense, for the host's side of each check.
        dense: DMatrix<f64>,
        offset: Vec<u32>,
        index: Vec<u32>,
        value: Vec<f32>,
        transpose_offset: Vec<u32>,
        transpose_pair: Vec<u32>,
        diagonal: Vec<f32>,
        /// Contact couplings `(i, j, block)` with `i < j`.
        contact: Vec<(u32, u32, Block)>,
        removed: Vec<u32>,
    }

    fn outer(d: [f64; 3], k: f64) -> Block {
        let mut b = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                b[r][c] = k * d[r] * d[c];
            }
        }
        b
    }

    fn sheet(n: usize, stiffness: f64, mass: f64, contacts: usize, pins: &[u32]) -> Sheet {
        pieces(&[(n, 1.0, [0.0; 3])], &[], stiffness, mass, contacts, pins)
    }

    /// Several sheets, each `(side, size, offset)`, and springs between vertices
    /// of different ones, which is what a stitch is to the operator: a block of
    /// the fixed matrix coupling two pieces of mesh that share no element.
    fn pieces(
        sheets: &[(usize, f64, [f64; 3])],
        stitches: &[(u32, u32)],
        stiffness: f64,
        mass: f64,
        contacts: usize,
        pins: &[u32],
    ) -> Sheet {
        let mut positions = Vec::new();
        let mut faces = Vec::new();
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for &(n, size, offset) in sheets {
            let first = positions.len() as u32;
            for j in 0..n {
                for i in 0..n {
                    let (x, y) = (i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64);
                    positions.push([
                        offset[0] + size * x,
                        offset[1] + size * 0.04 * (5.0 * x).sin() * (3.0 * y).cos(),
                        offset[2] + size * y,
                    ]);
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let a = first + (j * n + i) as u32;
                    let (b, c, d) = (a + 1, a + n as u32, a + n as u32 + 1);
                    faces.push([a, b, c]);
                    faces.push([b, d, c]);
                    edges.extend([(a, b), (a, c), (b, c), (b, d), (c, d)]);
                }
            }
        }
        let vertices = positions.len();
        edges.extend(stitches.iter().map(|&(a, b)| (a.min(b), a.max(b))));
        edges.sort_unstable();
        edges.dedup();
        let removed: Vec<u32> = (0..vertices as u32).map(|v| u32::from(pins.contains(&v))).collect();
        let free = |v: u32| removed[v as usize] == 0;
        // the fixed matrix's blocks, both triangles
        let mut fixed: BTreeMap<(u32, u32), Block> = BTreeMap::new();
        for v in 0..vertices as u32 {
            fixed.insert((v, v), [[0.0; 3]; 3]);
        }
        for &(a, b) in &edges {
            let (p, q) = (positions[a as usize], positions[b as usize]);
            let mut d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for c in &mut d {
                *c /= length;
            }
            let mut k = outer(d, stiffness);
            for r in 0..3 {
                k[r][r] += 1.0e-3 * stiffness;
            }
            // a pinned vertex's row and column are cleared, as the Dirichlet
            // pass clears them
            if !free(a) || !free(b) {
                fixed.insert((a, b), [[0.0; 3]; 3]);
                fixed.insert((b, a), [[0.0; 3]; 3]);
                for (v, other) in [(a, b), (b, a)] {
                    if free(v) && !free(other) {
                        let own = fixed.get_mut(&(v, v)).unwrap();
                        for r in 0..3 {
                            for c in 0..3 {
                                own[r][c] += k[r][c];
                            }
                        }
                    }
                }
                continue;
            }
            for (v, w, sign) in [(a, a, 1.0), (b, b, 1.0), (a, b, -1.0), (b, a, -1.0)] {
                let slot = fixed.entry((v, w)).or_insert([[0.0; 3]; 3]);
                for r in 0..3 {
                    for c in 0..3 {
                        slot[r][c] += sign * k[r][c];
                    }
                }
            }
        }
        // the block diagonal: the mass, or the identity on a pinned vertex
        let mut c_blocks: Vec<Block> = (0..vertices as u32)
            .map(|v| {
                let m = if free(v) { mass } else { 1.0 };
                [[m, 0.0, 0.0], [0.0, m, 0.0], [0.0, 0.0, m]]
            })
            .collect();
        // contact: a stiff coupling along one direction between vertices far
        // apart in the mesh, its diagonal halves on the block diagonal
        let mut contact = Vec::new();
        for k in 0..contacts {
            let a = ((7 * k + 3) % (vertices / 2)) as u32;
            let b = (vertices - 1 - (5 * k + 1) % (vertices / 2)) as u32;
            if a >= b || !free(a) || !free(b) || contact.iter().any(|c: &(u32, u32, Block)| (c.0, c.1) == (a, b)) {
                continue;
            }
            let normal = [0.6, 0.64, 0.48];
            let block = outer(normal, 4.0 * stiffness);
            for v in [a, b] {
                for r in 0..3 {
                    for c in 0..3 {
                        c_blocks[v as usize][r][c] += block[r][c];
                    }
                }
            }
            let mut off = block;
            for row in &mut off {
                for e in row {
                    *e = -*e;
                }
            }
            contact.push((a, b, off));
        }
        contact.sort_by_key(|c| (c.0, c.1));
        // the dense whole
        let mut dense = DMatrix::<f64>::zeros(3 * vertices, 3 * vertices);
        let mut add = |i: u32, j: u32, block: &Block, transpose: bool| {
            for r in 0..3 {
                for c in 0..3 {
                    dense[(3 * i as usize + r, 3 * j as usize + c)] +=
                        if transpose { block[c][r] } else { block[r][c] };
                }
            }
        };
        for (&(i, j), block) in &fixed {
            add(i, j, block, false);
        }
        for (v, block) in c_blocks.iter().enumerate() {
            add(v as u32, v as u32, block, false);
        }
        for (i, j, block) in &contact {
            add(*i, *j, block, false);
            add(*j, *i, block, true);
        }
        // the fixed matrix's storage: the upper triangle, then its transposes
        let flat = |block: &Block| -> [f32; 9] {
            let mut out = [0.0f32; 9];
            for r in 0..3 {
                for c in 0..3 {
                    out[3 * c + r] = block[r][c] as f32;
                }
            }
            out
        };
        let (mut offset, mut index, mut value) = (vec![0u32], Vec::new(), Vec::new());
        let mut pairs: Vec<Vec<u32>> = vec![Vec::new(); vertices];
        for i in 0..vertices as u32 {
            for (&(_, j), block) in fixed.range((i, i)..(i + 1, 0)) {
                if j != i {
                    pairs[j as usize].extend([i, index.len() as u32]);
                }
                index.push(j);
                value.extend(flat(block));
            }
            offset.push(index.len() as u32);
        }
        let (mut transpose_offset, mut transpose_pair) = (vec![0u32], Vec::new());
        for row in &pairs {
            transpose_pair.extend_from_slice(row);
            transpose_offset.push((transpose_pair.len() / 2) as u32);
        }
        let diagonal = c_blocks.iter().flat_map(flat).collect();
        Sheet {
            vertices,
            positions,
            faces,
            dense,
            offset,
            index,
            value,
            transpose_offset,
            transpose_pair,
            diagonal,
            contact,
            removed,
        }
    }

    /// The sheet's operator on a device, and the buffers its handles name.
    struct Resident {
        u32s: Vec<Buffer<u32>>,
        f32s: Vec<Buffer<f32>>,
        operator: Operator,
        removed: Handle,
        block_diagonal: Handle,
    }

    fn on_device<D: Device>(device: &mut D, sheet: &Sheet) -> Resident {
        let mut u32s: Vec<Buffer<u32>> = Vec::new();
        let mut f32s: Vec<Buffer<f32>> = Vec::new();
        let mut put_u32 = |device: &mut D, data: &[u32]| {
            let mut b = Buffer::<u32>::default();
            upload(device, &mut b, data, "t.u32").unwrap();
            let h = b.handle();
            u32s.push(b);
            h
        };
        let index = put_u32(device, &sheet.index);
        let offset = put_u32(device, &sheet.offset);
        let transpose_pair = put_u32(device, &sheet.transpose_pair);
        let transpose_offset = put_u32(device, &sheet.transpose_offset);
        let removed = put_u32(device, &sheet.removed);
        // the contact matrix: the upper rows, then each row's transposes
        let n = sheet.vertices;
        let (mut d_offset, mut d_index, mut d_value) = (vec![0u32; n + 1], Vec::new(), Vec::new());
        let mut refs: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for (slot, (i, j, block)) in sheet.contact.iter().enumerate() {
            d_offset[*i as usize + 1] += 1;
            d_index.push(*j);
            for c in 0..3 {
                for r in 0..3 {
                    d_value.push(block[r][c] as f32);
                }
            }
            refs[*j as usize].push((*i, slot as u32));
        }
        for row in 0..n {
            d_offset[row + 1] += d_offset[row];
        }
        let (mut t_offset, mut t_index, mut t_value) = (vec![0u32], Vec::new(), Vec::new());
        for row in &refs {
            for (other, slot) in row {
                t_index.push(*other);
                t_value.push(*slot);
            }
            t_offset.push(t_index.len() as u32);
        }
        let dynamic = if sheet.contact.is_empty() {
            None
        } else {
            let offset = put_u32(device, &d_offset);
            let index = put_u32(device, &d_index);
            let transpose_offset = put_u32(device, &t_offset);
            let transpose_index = put_u32(device, &t_index);
            let transpose_value = put_u32(device, &t_value);
            let mut value = Buffer::<f32>::default();
            upload(device, &mut value, &d_value, "t.dv").unwrap();
            let view = DynamicView {
                offset,
                index,
                value: value.handle(),
                transpose_offset,
                transpose_index,
                transpose_value,
            };
            f32s.push(value);
            Some(view)
        };
        let mut value = Buffer::<f32>::default();
        upload(device, &mut value, &sheet.value, "t.fv").unwrap();
        let mut diagonal = Buffer::<f32>::default();
        upload(device, &mut diagonal, &sheet.diagonal, "t.diag").unwrap();
        let mut block_diagonal = Buffer::<f32>::default();
        block_diagonal.size(device, 9 * n, AllocLabel("t.bd")).unwrap();
        let operator = Operator {
            dynamic,
            fixed: FixedCsrView {
                index,
                offset,
                value: value.handle(),
                transpose_pair,
                transpose_offset,
                rows: n as u32,
            },
            diagonal: diagonal.handle(),
            difference: None,
        };
        operator.precond_diagonal(device, block_diagonal.handle()).unwrap();
        let block_diagonal_handle = block_diagonal.handle();
        f32s.extend([value, diagonal, block_diagonal]);
        Resident { u32s, f32s, operator, removed, block_diagonal: block_diagonal_handle }
    }

    fn hierarchy_of(sheet: &Sheet) -> Hierarchy {
        let mesh = Mesh { positions: &sheet.positions, faces: &sheet.faces, edges: &[], tets: &[] };
        multigrid_grid::build(&mesh, &Pattern::from_upper_csr(&sheet.offset, &sheet.index))
    }

    /// Build and refresh the preconditioner over a sheet, and give the sheet's
    /// operator the difference form, which is what the step driver does.
    fn ready<D: Device>(device: &mut D, sheet: &Sheet, resident: &mut Resident) -> (State, Hierarchy) {
        let hierarchy = hierarchy_of(sheet);
        let mut state = State::default();
        // Safety: the hierarchy was laid over this sheet's own pattern.
        unsafe { build(device, &mut state, &hierarchy, sheet.vertices as u32) }.unwrap();
        resident.operator.difference = Some(state.difference());
        resident.operator.sum_rows(device).unwrap();
        let empty = state.empty_offsets();
        // Safety: every handle names a live allocation of the sheet's operator.
        unsafe {
            refresh(
                device,
                &mut state,
                resident.operator.rows_view(empty),
                resident.block_diagonal,
                resident.removed,
            )
        }
        .unwrap_or_else(|fatal| panic!("the refresh failed: {}", fatal.detail));
        (state, hierarchy)
    }

    /// The interpolation of one level as a dense matrix acting on the three
    /// unknowns of a row alike.
    fn dense_transfer(t: &multigrid_grid::Transfer, skip: &[u32]) -> DMatrix<f64> {
        let mut p = DMatrix::<f64>::zeros(3 * t.fine as usize, 3 * t.nodes as usize);
        for row in 0..t.fine as usize {
            if skip.get(row).is_some_and(|s| *s != 0) {
                continue;
            }
            for k in 0..WIDTH {
                let w = t.weight[WIDTH * row + k] as f64;
                if w > 0.0 {
                    for a in 0..3 {
                        p[(3 * row + a, 3 * t.node[WIDTH * row + k] as usize + a)] = w;
                    }
                }
            }
        }
        p
    }

    fn read_f32<D: Device>(device: &mut D, buffer: &Buffer<f32>, len: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        buffer.read(device, 0, &mut out).unwrap();
        out
    }

    /// Every level's operator is the Galerkin product of the one above it, on
    /// exactly the pattern the hierarchy laid.
    #[test]
    fn every_coarse_operator_is_the_galerkin_product_of_the_one_above() {
        let sheet = sheet(13, 2.0e3, 1.0, 0, &[40, 41]);
        let mut device = host_device();
        let mut resident = on_device(&mut device, &sheet);
        let (state, hierarchy) = ready(&mut device, &sheet, &mut resident);
        assert!(state.depth() >= 2, "a 13 x 13 sheet has {} coarse levels", state.depth());
        let mut above = sheet.dense.clone();
        // a pinned vertex is no part of what the free vertices see
        for (v, r) in sheet.removed.iter().enumerate() {
            if *r != 0 {
                for a in 0..3 {
                    above[(3 * v + a, 3 * v + a)] = 0.0;
                }
            }
        }
        for (depth, plan) in hierarchy.levels.iter().enumerate() {
            let skip: &[u32] = if depth == 0 { &sheet.removed } else { &[] };
            let p = dense_transfer(&plan.transfer, skip);
            let want = p.transpose() * &above * &p;
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let level = &state.levels[depth];
            let value = read_f32(&mut device, &level.value, 9 * plan.pattern.blocks());
            let mut covered = DMatrix::<f64>::zeros(want.nrows(), want.ncols());
            for node in 0..plan.pattern.rows() {
                for slot in plan.pattern.offset[node] as usize..plan.pattern.offset[node + 1] as usize {
                    let column = plan.pattern.column[slot] as usize;
                    for r in 0..3 {
                        for c in 0..3 {
                            let got = value[9 * slot + 3 * c + r] as f64;
                            let expect = want[(3 * node + r, 3 * column + c)];
                            assert!(
                                (got - expect).abs() <= 2.0e-5 * scale,
                                "level {depth} block ({node}, {column}) entry ({r}, {c}): {got} against {expect}"
                            );
                            covered[(3 * node + r, 3 * column + c)] = 1.0;
                        }
                    }
                }
            }
            for i in 0..want.nrows() {
                for j in 0..want.ncols() {
                    assert!(
                        covered[(i, j)] == 1.0 || want[(i, j)].abs() <= 1.0e-12 * scale,
                        "level {depth}: the product has an entry at ({i}, {j}) the pattern does not hold"
                    );
                }
            }
            above = want;
        }
    }

    /// Pieces of different depth, stitched: the small one's grid ends levels
    /// above the large one's, and the operator still couples its last node to
    /// nodes that go on.
    ///
    /// WHAT THIS CATCHES. A row with no interpolation has no row of `A P`, and
    /// the product pass has to leave it out rather than spread its blocks into
    /// a pattern that rightly has no slot for them. A scene of one piece, or of
    /// pieces that all end at one level, never has such a row; a dress of
    /// twenty-two stitched panels does, and it stopped at its first solve.
    #[test]
    fn a_piece_that_ends_early_stays_out_of_the_levels_below_it() {
        // the second sheet is a quarter the size at a third of the resolution,
        // and a vertex in no element is stitched on as well
        let mut fixture = pieces(
            &[(17, 1.0, [0.0; 3]), (5, 0.25, [1.5, 0.0, 0.0])],
            &[(16, 289), (16 + 17, 289 + 5), (16 + 34, 289 + 10)],
            2.0e3,
            1.0,
            0,
            &[],
        );
        // the loose vertex: appended by hand, coupled to one vertex of each sheet
        let loose = fixture.vertices as u32;
        {
            let n = fixture.vertices + 1;
            let mut dense = DMatrix::<f64>::zeros(3 * n, 3 * n);
            dense.view_mut((0, 0), (3 * (n - 1), 3 * (n - 1))).copy_from(&fixture.dense);
            let k = 50.0;
            for other in [3u32, 300] {
                for a in 0..3 {
                    dense[(3 * loose as usize + a, 3 * other as usize + a)] -= k;
                    dense[(3 * other as usize + a, 3 * loose as usize + a)] -= k;
                    dense[(3 * other as usize + a, 3 * other as usize + a)] += k;
                    dense[(3 * loose as usize + a, 3 * loose as usize + a)] += k;
                }
            }
            for a in 0..3 {
                dense[(3 * loose as usize + a, 3 * loose as usize + a)] += 1.0;
            }
            fixture.positions.push([9.0, 9.0, 9.0]);
            fixture.removed.push(0);
            fixture.vertices = n;
            // rebuild the stored form from the dense one: upper triangle rows,
            // their transposes, and the mass on the block diagonal
            let block = |i: usize, j: usize| -> [f32; 9] {
                let mut out = [0.0f32; 9];
                for r in 0..3 {
                    for c in 0..3 {
                        out[3 * c + r] = dense[(3 * i + r, 3 * j + c)] as f32;
                    }
                }
                out
            };
            let (mut offset, mut index, mut value) = (vec![0u32], Vec::new(), Vec::new());
            let mut pairs: Vec<Vec<u32>> = vec![Vec::new(); n];
            for i in 0..n {
                for j in i..n {
                    let b = block(i, j);
                    if j != i && b.iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    if j != i {
                        pairs[j].extend([i as u32, index.len() as u32]);
                    }
                    index.push(j as u32);
                    value.extend(b);
                }
                offset.push(index.len() as u32);
            }
            let (mut transpose_offset, mut transpose_pair) = (vec![0u32], Vec::new());
            for row in &pairs {
                transpose_pair.extend_from_slice(row);
                transpose_offset.push((transpose_pair.len() / 2) as u32);
            }
            fixture.offset = offset;
            fixture.index = index;
            fixture.value = value;
            fixture.transpose_offset = transpose_offset;
            fixture.transpose_pair = transpose_pair;
            fixture.diagonal = vec![0.0f32; 9 * n];
            fixture.dense = dense;
        }
        let mut device = host_device();
        let mut resident = on_device(&mut device, &fixture);
        let (mut state, hierarchy) = ready(&mut device, &fixture, &mut resident);
        assert_eq!((hierarchy.pieces, hierarchy.loose), (2, 1));
        // the small sheet is one node levels before the large one is
        let ends: Vec<usize> = hierarchy
            .levels
            .iter()
            .map(|l| {
                let t = &l.transfer;
                (0..t.fine as usize)
                    .filter(|r| t.weight[WIDTH * r..WIDTH * (r + 1)].iter().all(|w| *w == 0.0))
                    .count()
            })
            .collect();
        assert!(ends[0] == 1, "only the loose vertex is uninterpolated at the finest level: {ends:?}");
        assert!(
            ends[1..].iter().any(|e| *e > 0),
            "no piece ended above the last level, so the fixture tests nothing: {ends:?}"
        );
        // every level is still the Galerkin product of the one above
        let mut above = fixture.dense.clone();
        for (depth, plan) in hierarchy.levels.iter().enumerate() {
            let p = dense_transfer(&plan.transfer, &[]);
            let want = p.transpose() * &above * &p;
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let value = read_f32(&mut device, &state.levels[depth].value, 9 * plan.pattern.blocks());
            for node in 0..plan.pattern.rows() {
                for slot in plan.pattern.offset[node] as usize..plan.pattern.offset[node + 1] as usize {
                    let column = plan.pattern.column[slot] as usize;
                    for r in 0..3 {
                        for c in 0..3 {
                            let got = value[9 * slot + 3 * c + r] as f64;
                            let expect = want[(3 * node + r, 3 * column + c)];
                            assert!(
                                (got - expect).abs() <= 2.0e-5 * scale,
                                "level {depth} block ({node}, {column}) entry ({r}, {c}): {got} against {expect}"
                            );
                        }
                    }
                }
            }
            above = want;
        }
        // and the cycle over it is still positive on a residual
        let n = 3 * fixture.vertices;
        let mut r = Buffer::<f32>::default();
        let mut z = ReadbackBuffer::<f32>::default();
        upload(&mut device, &mut r, &vec![1.0f32; n], "t.r").unwrap();
        z.size(&mut device, n, AllocLabel("t.z")).unwrap();
        let (r_h, z_h) = (r.handle(), z.handle());
        device
            .run("t.cycle", |encoder| unsafe {
                encode_cycle(encoder, &mut state, &resident.operator, r_h, z_h)
            })
            .unwrap();
        z.download(&mut device).unwrap();
        let dot: f64 = z.host().iter().map(|v| *v as f64).sum();
        assert!(z.host().iter().all(|v| v.is_finite()) && dot > 0.0, "r . M^-1 r is {dot:e}");
    }

    /// The smoother's bound majorizes the operator, contact matrix included,
    /// and the cycle built on it is symmetric and positive definite.
    #[test]
    fn the_bound_majorizes_the_operator_and_the_cycle_is_symmetric_positive_definite() {
        let sheet = sheet(11, 5.0e3, 0.5, 12, &[0, 10, 60]);
        assert!(sheet.contact.len() >= 6, "the fixture carries {} contacts", sheet.contact.len());
        let n = 3 * sheet.vertices;
        let mut device = host_device();
        let mut resident = on_device(&mut device, &sheet);
        let (mut state, _) = ready(&mut device, &sheet, &mut resident);

        // M - A is positive semidefinite, M the block-diagonal bound. The
        // device keeps `step M^-1`, so M is read back out of it.
        let inverse = read_f32(&mut device, &state.inverse, 9 * sheet.vertices);
        let step = read_f32(&mut device, &state.step, sheet.vertices);
        let mut gap = -sheet.dense.clone();
        for v in 0..sheet.vertices {
            let scaled = nalgebra::Matrix3::<f64>::from_fn(|r, c| inverse[9 * v + 3 * c + r] as f64);
            let bound = scaled.try_inverse().expect("a smoother block is invertible") * step[v] as f64;
            for r in 0..3 {
                for c in 0..3 {
                    gap[(3 * v + r, 3 * v + c)] += bound[(r, c)];
                }
            }
        }
        let scale = sheet.dense.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let lowest = nalgebra::SymmetricEigen::new(0.5 * (&gap + gap.transpose()))
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(lowest >= -1.0e-3 * scale, "M - A has an eigenvalue of {lowest} at a scale of {scale}");

        // the cycle on a spread of residuals, zero on the pinned vertices as
        // the conjugate gradient's residual is
        let mut r = Buffer::<f32>::default();
        let mut z = ReadbackBuffer::<f32>::default();
        r.size(&mut device, n, AllocLabel("t.r")).unwrap();
        z.size(&mut device, n, AllocLabel("t.z")).unwrap();
        let mut seed = 0x2545_f491u32;
        let mut residuals: Vec<Vec<f32>> = Vec::new();
        let mut results: Vec<Vec<f32>> = Vec::new();
        for trial in 0..6usize {
            let host: Vec<f32> = (0..n)
                .map(|k| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    if sheet.removed[k / 3] != 0 {
                        0.0
                    } else if trial == 0 {
                        1.0
                    } else if trial == 1 {
                        if k == 3 * 37 + 1 { 1.0 } else { 0.0 }
                    } else {
                        (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                    }
                })
                .collect();
            r.write(&mut device, 0, &host).unwrap();
            let (r_h, z_h) = (r.handle(), z.handle());
            device
                .run("t.cycle", |encoder| unsafe {
                    encode_cycle(encoder, &mut state, &resident.operator, r_h, z_h)
                })
                .unwrap();
            z.download(&mut device).unwrap();
            assert!(z.host().iter().all(|v| v.is_finite()), "trial {trial} is not finite");
            for (k, v) in z.host().iter().enumerate() {
                assert!(
                    sheet.removed[k / 3] == 0 || *v == 0.0,
                    "trial {trial}: pinned vertex {} was given a correction of {v}",
                    k / 3
                );
            }
            residuals.push(host);
            results.push(z.host().to_vec());
        }
        let dot = |a: &[f32], b: &[f32]| -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum() };
        for a in 0..residuals.len() {
            let own = dot(&residuals[a], &results[a]);
            assert!(own > 0.0, "trial {a}: r . M^-1 r is {own:e}");
            for b in 0..a {
                let (ab, ba) = (dot(&residuals[a], &results[b]), dot(&residuals[b], &results[a]));
                let size = (own * dot(&residuals[b], &results[b])).sqrt();
                assert!(
                    (ab - ba).abs() <= 2.0e-3 * size,
                    "trials {a} and {b}: {ab:e} against {ba:e} at a size of {size:e}"
                );
            }
        }
    }

    /// A solve asks for the hierarchy and never lays one: a state nothing was
    /// built into is refused by name, and so is one built for another scene.
    ///
    /// This is what keeps the host pass and the allocations out of a Newton
    /// step. A solve that built on demand instead would pass every other test
    /// here and charge the build to the first linear solve.
    #[test]
    fn a_solve_refuses_a_hierarchy_that_was_never_laid() {
        let sheet = sheet(9, 1.0e3, 1.0, 8, &[]);
        let vertices = sheet.vertices as u32;
        let mut device = host_device();
        let mut state = State::default();
        let unbuilt = state.require_built(vertices).expect_err("nothing was laid");
        assert!(unbuilt.detail.contains("none was laid at all"), "{}", unbuilt.detail);
        // Safety: the hierarchy was laid over this sheet's own pattern.
        unsafe { build(&mut device, &mut state, &hierarchy_of(&sheet), vertices) }.unwrap();
        state.require_built(vertices).expect("the hierarchy in hand covers this scene");
        let other = state.require_built(vertices + 1).expect_err("laid for another row count");
        assert!(
            other.detail.contains(&format!("the one in hand covers {vertices}")),
            "{}",
            other.detail
        );
    }

    /// The difference form is the same product as the plain one, and it
    /// resolves the curvature of a vector the plain one cannot.
    ///
    /// The vector is a whole sheet moving together, which is what a coarse
    /// correction is. Its curvature is the sheet's mass alone, every spring
    /// being unstretched, and the plain product forms it as what is left of
    /// spring terms ten thousand times larger.
    ///
    /// THE STIFFNESS IS NOT TAKEN FURTHER, and the limit is the form's own. The
    /// row sums are formed in single precision from the same blocks, so a mass
    /// below the round-off of the springs around it is not in them either. That
    /// is a property of the matrix as stored, not of how it is applied.
    #[test]
    fn the_difference_form_is_the_same_product_and_resolves_a_smooth_vector() {
        let sheet = sheet(21, 1.0e4, 1.0, 8, &[]);
        let n = 3 * sheet.vertices;
        let mut device = host_device();
        let mut resident = on_device(&mut device, &sheet);
        let mut state = State::default();
        // Safety: the hierarchy was laid over this sheet's own pattern.
        unsafe { build(&mut device, &mut state, &hierarchy_of(&sheet), sheet.vertices as u32) }.unwrap();
        let view = state.difference();
        let mut x = Buffer::<f32>::default();
        x.size(&mut device, n, AllocLabel("t.x")).unwrap();
        let mut outputs: Vec<ReadbackBuffer<f32>> = (0..3).map(|_| ReadbackBuffer::default()).collect();
        outputs[0].size(&mut device, n, AllocLabel("t.y")).unwrap();
        outputs[1].size(&mut device, sheet.vertices, AllocLabel("t.abs")).unwrap();
        outputs[2].size(&mut device, sheet.vertices, AllocLabel("t.curv")).unwrap();
        let mut apply = |device: &mut _, resident: &Resident, host: &[f32]| {
            x.write(device, 0, host).unwrap();
            let (y, a, c) = (outputs[0].handle(), outputs[1].handle(), outputs[2].handle());
            resident.operator.apply(device, x.handle(), y, a, c).unwrap();
            for o in outputs.iter_mut() {
                o.download(device).unwrap();
            }
            let curvature: f64 = outputs[2].host().iter().map(|v| *v as f64).sum();
            let absolute: f64 = outputs[1].host().iter().map(|v| *v as f64).sum();
            (outputs[0].host().to_vec(), curvature, absolute)
        };
        // a rough vector: the two forms agree to single precision
        let mut seed = 0x9e37_79b9u32;
        let rough: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        let (plain, _, _) = apply(&mut device, &resident, &rough);
        resident.operator.difference = Some(view);
        resident.operator.sum_rows(&mut device).unwrap();
        let (different, _, _) = apply(&mut device, &resident, &rough);
        let scale = plain.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (k, (a, b)) in plain.iter().zip(&different).enumerate() {
            assert!((a - b).abs() <= 2.0e-4 * scale, "component {k}: {a} against {b}");
        }
        // a smooth one: the whole sheet translating
        let smooth: Vec<f32> = (0..n).map(|k| [0.3f32, -0.7, 0.5][k % 3]).collect();
        let exact: f64 = {
            let v = nalgebra::DVector::<f64>::from_iterator(n, smooth.iter().map(|v| *v as f64));
            (v.transpose() * &sheet.dense * &v)[(0, 0)]
        };
        let (_, curvature, absolute) = apply(&mut device, &resident, &smooth);
        resident.operator.difference = None;
        let (_, plain_curvature, plain_absolute) = apply(&mut device, &resident, &smooth);
        assert!(
            (curvature - exact).abs() <= 2.0e-2 * exact,
            "the difference form gives a curvature of {curvature} against {exact}"
        );
        assert!(
            curvature / absolute > 100.0 * (plain_curvature.abs() / plain_absolute),
            "the difference form resolves {:e} of its magnitudes and the plain form {:e}",
            curvature / absolute,
            plain_curvature.abs() / plain_absolute
        );
    }

    /// The point of the preconditioner, measured rather than assumed: the same
    /// solve takes several times fewer iterations than under block-Jacobi.
    #[test]
    fn the_cycle_cuts_the_conjugate_gradient_iterations() {
        let sheet = sheet(41, 2.0e5, 1.0, 0, &[0, 40]);
        let n = 3 * sheet.vertices;
        let mut device = host_device();
        let mut resident = on_device(&mut device, &sheet);
        let (mut state, _) = ready(&mut device, &sheet, &mut resident);

        let mut seed = 0x1234_5678u32;
        let b_host: Vec<f32> = (0..n)
            .map(|k| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if sheet.removed[k / 3] != 0 { 0.0 } else { (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5 }
            })
            .collect();
        let mut b = Buffer::<f32>::default();
        upload(&mut device, &mut b, &b_host, "t.b").unwrap();
        let mut inverse = Buffer::<f32>::default();
        {
            let diagonal = read_f32(&mut device, resident.f32s.last().unwrap(), 9 * sheet.vertices);
            let mut host = vec![0.0f32; 9 * sheet.vertices];
            assert!(pcg::build_block_jacobi(&diagonal, &mut host, sheet.vertices as u32));
            upload(&mut device, &mut inverse, &host, "t.inverse").unwrap();
        }
        let solve = |device: &mut _, sweep: Option<pcg::Sweep<'_>>| {
            let mut x = Buffer::<f32>::default();
            upload(device, &mut x, &vec![0.0f32; n], "t.x").unwrap();
            let mut work = pcg::Workspace::default();
            work.size_for(device, sheet.vertices as u32).unwrap();
            pcg::solve(device, &resident.operator, inverse.handle(), b.handle(), x.handle(), &mut work, sweep, 1.0e-4, 4000)
                .expect("the solve runs")
        };
        let plain = solve(&mut device, None);
        let cycled = solve(&mut device, Some(pcg::Sweep::Multigrid(&mut state)));
        assert_eq!(plain.outcome, pcg::Outcome::Converged, "block-Jacobi: {plain:?}");
        assert_eq!(cycled.outcome, pcg::Outcome::Converged, "multigrid: {cycled:?}");
        assert!(
            4 * cycled.iterations < plain.iterations,
            "the multigrid took {} iterations against block-Jacobi's {}",
            cycled.iterations,
            plain.iterations
        );
        let _ = &resident.u32s;
    }
}
