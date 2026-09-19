// File: crates/ppf-cts-solver/src/driver/dirichlet.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Dirichlet DOF removal: the last writer of the Newton system before the solve.
//!
//! A fix pin is an exact boundary condition, not a stiff spring, so its rows and
//! columns leave the system and the free rows they coupled to receive the known
//! contribution on their right-hand side. `main/dirichlet.kernel.cpp` carries
//! the reduction, the sign of the lifting term and why each of the three steps
//! is what it is; this module owns only the walk and the split into two passes.
//!
//! # The lifting term is the whole point, and dropping it is a HANG
//!
//! With `M dx = f` and `dx_i = p_i` prescribed on the removed vertices, the
//! exact reduction of a free row `j` is
//! `sum_{k free} M_jk dx_k = f_j - sum_{k removed} M_jk p_k`. That last sum is
//! the only signal a free vertex has that a prescribed one is advancing into it.
//! Zero the coupling without moving it to the right-hand side and a moving
//! collider out-runs the cloth every iteration: the gap closes onto the parking
//! distance, the line search clamps `toi` to nearly zero, and the loop
//! re-assembles a bit-identical system forever. That failure is a hang, not a
//! wrong number, which is why it is stated here rather than left to the body.
//!
//! # Two passes, and they may not be merged
//!
//! A removed row's right-hand side must carry the increment and NOTHING else.
//! The lifts are scatters into other rows, so a lift issued while pass two is
//! running could land on a row that has already been prescribed. main.cu splits
//! them into two dispatches for that reason and so does this.

use ppf_cts_compute::Device;
use super::kernels::{DirichletLiftRowArgs, DirichletPrescribeGatedArgs};
use super::scene::FatalResult;
use super::state::SolverState;

/// Reduce one matrix: lift every coupling between a free row and a removed one
/// onto the right-hand side, then clear it.
///
/// `offset` and `column` are the matrix's per-row pattern and `values` its value
/// array, which this modifies in place. It is called once per matrix, because
/// the Newton system has two: the contact matrix, whose pattern is discovered
/// per step, and the fixed-pattern one carrying the elastic, stitch,
/// strain-limit and pin-barrier blocks. `main.cu` walks the two inside one
/// dispatch; here they are two calls, which changes only the order in which the
/// lifts accumulate into `force`, and that order is an atomic scatter there.
///
/// SERIAL, BY CONTRACT. The lift is `compute::atomic_add` into another row's
/// force, which the host seam spells as a plain read, add and write back, and a
/// free row can receive lifts from several removed rows at once. The whole pass
/// is one call for that reason.
///
/// # Safety
/// The pattern arrays must describe the value array, and every buffer must be
/// sized for the scene.
pub unsafe fn lift<D: Device>(
    device: &mut D,
    state: &mut SolverState,
    offset: ppf_cts_compute::Handle,
    column: ppf_cts_compute::Handle,
    values: ppf_cts_compute::Handle,
) -> FatalResult<()> {
    let vertices = state.sizes.vertices;
    if vertices == 0 {
        return Ok(());
    }
    // THE `offset.len() == vertices + 1` CHECK SURVIVED THE MIGRATION, against
    // the handle's own `size`, which is in ELEMENTS. `dirichlet_lift_row`
    // declares no `[[seam::bound]]` and CANNOT: its own body says so, because a
    // walk over `offset[row] .. offset[row + 1]` reads a range that is data, so
    // no index list can name it and no device-side bound can be checked against
    // it. This assert is therefore the only check there is.
    debug_assert_eq!(offset.size as usize, vertices + 1);
    // THE SERIAL PASS IS THE DECLARATION'S, NOT THIS CALL SITE'S. The kernel is
    // declared `Scatter::Atomic` in `super::kernels`, so the backend runs it as
    // one ascending pass whoever dispatches it; before the seam that rule lived
    // as a comment here and as a call that happened to pass the whole range.
    let args = DirichletLiftRowArgs {
        dof_mask: state.dof_mask.handle(),
        offset,
        column,
        value: values,
        force: state.force.handle(),
        eval_x: state.eval_x.handle(),
        target: state.target.handle(),
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("dirichlet.lift", &args, vertices as u32)?;
    Ok(())
}

/// Make every removed row the identity with the prescribed right-hand side.
///
/// RUN AFTER EVERY MATRIX HAS BEEN LIFTED, never interleaved with one. A removed
/// row's right-hand side must carry its increment and nothing else, and a lift
/// issued afterwards would land on a row already prescribed.
///
/// # Safety
/// Every buffer must be sized for the scene.
pub unsafe fn prescribe<D: Device>(device: &mut D, state: &mut SolverState) -> FatalResult<()> {
    let vertices = state.sizes.vertices;
    if vertices == 0 {
        return Ok(());
    }
    let args = DirichletPrescribeGatedArgs {
        dof_mask: state.dof_mask.handle(),
        diagonal: state.diagonal.handle(),
        force: state.force.handle(),
        eval_x: state.eval_x.handle(),
        target: state.target.handle(),
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("dirichlet.prescribe", &args, vertices as u32)?;
    Ok(())
}
