// File: crates/ppf-cts-solver/src/driver/step.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The Newton driver: one `advance()`.
//!
//! One substep of the implicit integrator, expressed as a Rust loop over the
//! shared neutral kernel bodies. The `// STEP` markers below number the stages
//! in the order they run, and that order is load-bearing rather than cosmetic:
//! moving a stage past another changes an fp32 fold order, or moves the
//! `tmp_fixed` snapshot boundary, and neither shows up as a failure.
//!
//! # What this driver covers, and what it refuses
//!
//! Tets, the shell membrane and its bending, the two rod terms, both strain
//! limiters, the cross-stitch, and self-contact with its barrier, its friction,
//! its ACCD line search and its intersection gate. Everything else is refused
//! BY NAME at `initialize()` (`super::refusal`), and this function asserts the
//! refusals held rather than assuming them: a driver that accepted a scene it
//! cannot step would write a full run of plausible frames, and this backend has
//! no hardware fault to catch that.
//!
//! # Three traps kept deliberately
//!
//! 1. `target` is CLOBBERED at B20 to hold the iterate's pre-step positions,
//!    and every CONTINUING iteration rebuilds it with `compute_target` at the
//!    bottom of the loop. The line search reads the old positions out of it, so
//!    the reuse is load-bearing rather than a saved allocation.
//! 2. `tmp_fixed` is a REAL second buffer here. Metal splits the assembly into
//!    an evaluate and a scatter because it has no second matrix to spare;
//!    memory is cheap on the host, so this backend keeps the whole snapshot and
//!    assembles in one pass.
//! 3. The bounded `max-newton-steps` escape ships WITH the loop. It is the only
//!    exit from an over-constrained configuration: `toi <= FLT_EPSILON` does
//!    not cover a prescribed vertex driven into geometry that cannot yield,
//!    because the line search clamps the SHARED toi just above zero and the
//!    loop then re-assembles a bit-identical system forever. Without the bound
//!    such a scene HANGS instead of reporting `NewtonStall`.
//!
//! # The host scalars, and why they are not kernels moved to the host
//!
//! Eleven float expressions in this module and one in `pcg` are not routed
//! through a shared body. EVERY ONE OPERATES ON A SCALAR THAT HAS ALREADY BEEN
//! REDUCED OUT OF THE MESH, so it is host arithmetic by nature rather than a
//! kernel rewritten on the host. The list is exhaustive on purpose: it is
//! what a rule-8 allowlist would have to name, and a twelfth site appearing
//! without an entry here is the thing to catch.
//!
//! - `max_u = max(speed^2).sqrt()`, the square root taken after the fold.
//! - `dt = param.dt * playback`.
//! - `back = 1 - toi_advanced`, seeding the kinematic rewind.
//! - `dt = (double(dt) * toi_advanced) as f32`. `dt` is fp32 and `toi_advanced`
//!   is double, so the product is formed in double and rounded ONCE; forming it
//!   in fp32 rounds twice and gives a different step size, which is the clock
//!   the whole trailing iteration integrates against.
//! - `toi_recale = min(1, max_dx_param / max_dx)`.
//! - `toi = min(1, smallest / line_search_max_t)`, `smallest` being the folded
//!   per-primitive contact time of impact.
//! - `shell_toi = smallest / line_search_max_t`, then `toi = min(toi,
//!   shell_toi)`, and the same for the rod.
//! - `sl_toi = min(1, shell_toi, rod_toi)`.
//! - `toi_advanced += max(0, 1 - toi_advanced) * (toi_recale * toi)`, in double.
//! - `time += dt / playback`, in double.
//! - `reresid = sum|r| / err0` (`pcg`), the ratio taken on the host once both
//!   terms exist.
//!
//! Every PER-ELEMENT value comes from a body, including the ones that look like
//! they could not: the per-tet mass scale and the preconditioner's diagonal sum
//! both go through `vec_add_scaled`, because a matrix times a scalar is that
//! scalar applied to each of its stored floats.
//!
//! # THE FOLDS THAT PRODUCE THOSE SCALARS RUN ON THE HOST, AND THAT IS A
//! RECORDED DIVERGENCE
//!
//! Read the list above for what it says and not for what it looks like it
//! says. Each entry is true of the ARITHMETIC once the scalar exists; none of
//! them is a claim about how the scalar was reduced out of the mesh, and the
//! reduction is where this file diverges.
//!
//! Each of those scalars should fall out of a DEVICE reduction, a warp-shuffle
//! tree into a block tree into a strided fold, ending in a read of four bytes.
//! This module instead calls `ReadbackBuffer::download` on the whole array and
//! folds it in `super::reduce` over rayon, everywhere it needs one: `max_u` and
//! `reach` at the start of the step, `max_sigma` over the shell faces and the
//! rods, the count of removed Dirichlet rows, `max_dx` inside the Newton loop,
//! and both strain-limit times of impact. Four bytes should cross the seam
//! where `vertices`, `faces` or `rods` floats cross today, and the arithmetic
//! on top of the scalar is the same either way, so nothing in the list above
//! shows the difference.
//!
//! **THIS IS A RECORDED DIVERGENCE, NOT A TRANSPORT PREFERENCE**, and it takes
//! the same shape the PCG scalars do: a pass whose work is proportional to the
//! mesh belongs on the device, and running it on the host is a different solver
//! rather than a slower one. It is not closable from this file:
//! `kernels/primitives/reduce.kernel.cpp` declares `warp_reduce` and
//! `block_reduce` as `[[seam::device_fn]]` and declares NO `[[seam::entry]]`, so
//! there is no reduction for `driver/launch.rs` to bind and no verb on `Device`
//! to dispatch. Adding one is what closes every one of those folds at once, and
//! converting any of them any other way would be inventing a second design.

use crate::data::{DataSet, ParamSet, StepResult};

use super::contact::Windows;
use super::fixedcsr::FixedCsr;
use super::operator::{DynamicView, Operator};
use super::scene::{Fatal, FatalResult};
use super::state::SolverState;
use super::{assemble, dirichlet, log, pcg};

use ppf_cts_compute::{Device, EncoderExt, Fault, Handle};
use super::kernels::{
    SandGrainCondenseRowArgs, SandGrainIntegrateRowArgs, SandGrainRecoverRowArgs,
    BlockJacobiInvertRowArgs, ComputeTargetSeedArgs, VecFillArgs, DxMagnitudeArgs, DxSeedArgs, FixXzDragArgs, PositionAcceptArgs,
    PositionStepArgs, RewindFixArgs, VelocityTermsArgs,
};

extern "C" {
    // NOT A DISPATCH, and the only shim this module still names. It reports the
    // largest absolute coordinate the position representation covers, a
    // compile-time constant of the shared header rather than work over a range,
    // so it has no extent and no argument record and the seam has nothing to
    // carry it.
    fn position_domain_abi() -> f32;
}

/// Whether the DOF-removal count has been reported for this process.
///
/// Reported once: the mask is rebuilt every step but its SIZE is a
/// property of the scene, so repeating the line every step would bury the log
/// under a constant.
static DOF_REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The parameters one step reads, taken once at its start.
///
/// `ParamSet` is neither `Copy` nor `Clone` (it is a `repr(C)` mirror of a C++
/// record and copying it wholesale would copy container handles too), so the
/// snapshot names its fields. That is the better shape here anyway: it lists,
/// in one place, exactly what this driver's answer depends on.
struct StepParams {
    time: f64,
    dt: f32,
    playback: f32,
    prev_dt: f32,
    gravity: [f32; 3],
    inactive_momentum: bool,
    cg_tol: f32,
    cg_max_iter: u32,
    max_dx: f32,
    max_newton_steps: u32,
    min_newton_steps: u32,
    target_toi: f32,
    line_search_max_t: f32,
    eiganalysis_eps: f32,
    /// The factor the cross-stitch length cap scales its rest length by. One
    /// scene-wide scalar, read by the stitch layer and by nothing else.
    stitch_length_factor: f32,
    /// The barrier shape, as the `repr(C)` discriminant the shared bodies
    /// dispatch over. Carried as an integer because that is what crosses the
    /// FFI boundary; `Barrier` itself is the Rust mirror of the C++ enum.
    barrier: u32,
    disable_contact: bool,
    disable_pin_dof_removal: bool,
    /// The `fix-xz` drag's threshold, and the switch that decides whether the
    /// drag runs at all: at zero the step dispatches neither half. Both halves
    /// ship, so a scene setting it is accepted rather than refused, which
    /// `refusal.rs` asserts as a difference against the same scene with the
    /// threshold off.
    fix_xz: f32,
}

impl StepParams {
    /// # Safety
    /// `param` must address a live `ParamSet`.
    unsafe fn snapshot(param: *const ParamSet) -> Self {
        let p = &*param;
        Self {
            time: p.time,
            dt: p.dt,
            playback: p.playback,
            prev_dt: p.prev_dt,
            gravity: [p.gravity[0], p.gravity[1], p.gravity[2]],
            inactive_momentum: p.inactive_momentum,
            cg_tol: p.cg_tol,
            cg_max_iter: p.cg_max_iter,
            max_dx: p.max_dx,
            max_newton_steps: p.max_newton_steps,
            min_newton_steps: p.min_newton_steps,
            target_toi: p.target_toi,
            line_search_max_t: p.line_search_max_t,
            eiganalysis_eps: p.eiganalysis_eps,
            stitch_length_factor: p.stitch_length_factor,
            barrier: p.barrier as u32,
            disable_contact: p.disable_contact,
            disable_pin_dof_removal: p.disable_pin_dof_removal,
            fix_xz: p.fix_xz,
        }
    }
}

/// `fix_index` for every vertex, which `compute_target` and the seed both read.
///
/// Gathered into its own array because the shared body takes the index list
/// rather than the whole `VertexProp`: the same body serves a backend whose
/// props live in a different layout. Written into a buffer the driver already
/// owns, so a step allocates nothing.
///
/// # Safety
/// `data` must address a live `DataSet` whose prop array is `out.len()` long.
/// The fraction of a grain's spin angular momentum fed back into its
/// translation, read from `PPF_SAND_SPIN_COUPLE` and defaulting to 0.5.
///
/// 0.5 IS NEAR-TEXTBOOK ROLLING WHILE STAYING BOUNDED. Fully included, the spin
/// propels the translation which feeds the spin, which is a runaway rather than
/// a stiff case; zero leaves rolling to decay to about a third of the textbook
/// rate.
fn sand_spin_couple() -> f32 {
    std::env::var("PPF_SAND_SPIN_COUPLE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5)
}

/// The under-roll fraction the grain spin integrate caps omega at, read from
/// `PPF_SAND_ROLL_RESIST` and defaulting to 0.05.
///
/// ZERO IS NOT THE NEUTRAL VALUE HERE, WHICH IS WHY THIS IS NOT A LITERAL.
/// The cap is `(1 - roll_resist)` times the no-slip rate `|v_t| / (dt * r)`.
/// Capping at EXACTLY no-slip lets the lagged omega over-roll on an oscillating
/// contact and PROPEL the grain, which is the energy-pumping ratchet the bound
/// exists to prevent; capping slightly below it leaves a small residual forward
/// slip every step, so friction always opposes motion and travel stays bounded
/// while the grain still visibly rolls. `sand_rigid.kernel.cpp` carries the same
/// reasoning at the clamp itself.
///
/// The spin integrate takes `c_roll` as a literal `0.0` and `roll_resist` as
/// THIS, so the two trailing arguments of that dispatch do not agree and must
/// not be written as if they did.
fn sand_roll_resist() -> f32 {
    std::env::var("PPF_SAND_ROLL_RESIST")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.05)
}

/// Whether `PPF_SAND_NO_ROLL` pins every grain's omega at zero so grains slide
/// instead of roll, which is the lever for separating a rolling artifact from a
/// sliding one. Presence is the signal; the value is not read.
fn sand_no_roll() -> bool {
    std::env::var_os("PPF_SAND_NO_ROLL").is_some()
}

/// A per-step delay in milliseconds, for a caller that needs to OBSERVE a run
/// while it is in progress. Zero, and off, unless `PPF_STEP_DELAY_MS` says
/// otherwise.
///
/// WHY IT EXISTS. Seven rig scenarios watch a solve as it runs: a progress bar
/// advancing, a statistics panel filling, a frame counter tracking, a transfer
/// refused mid-run. A real solve on a scenario-sized scene finishes faster than
/// the observer can sample it, so a run that is correct is also unobservable,
/// and the scenario cannot tell "the state never appeared" from "the state was
/// never sampled".
///
/// IT DEFAULTS TO ZERO, AND IS A DELAY RATHER THAN A FREEZE. Defaulting to
/// anything else would slow every run to serve a handful of tests. Freezing
/// would let a scenario pass while the solver never reached the state it claims
/// to observe, which is the failure a pacing hook is most able to hide.
///
/// IT IS NOT REACHABLE FROM A SCENE. Read from the process environment and
/// never from `ParamSet`: a scene parameter that slows the solver is a way to
/// make a slow solver look correct, and a build default would do the same to
/// every user. It is read ONCE, so a run cannot change pace halfway and leave
/// two halves that are not comparable.
fn step_delay() -> std::time::Duration {
    static DELAY: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *DELAY.get_or_init(|| parse_step_delay(std::env::var("PPF_STEP_DELAY_MS").ok().as_deref()))
}

/// The parse, split out so it can be tested without the process environment.
///
/// UNPARSABLE READS AS ZERO RATHER THAN FAILING, which is the one place this
/// knob differs from the project's fail-loud default and is deliberate. A
/// pacing hook has no effect on what is computed, so a typo that silently
/// leaves the run at full speed costs an observer a retry, while a panic would
/// take down a solve over a debugging aid.
fn parse_step_delay(value: Option<&str>) -> std::time::Duration {
    std::time::Duration::from_millis(
        value
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(0),
    )
}

/// One `advance()`.
///
/// # Safety
/// `data` must address a live `DataSet`, `param` a live `ParamSet` the host
/// keeps for the whole run, and `state` must have been allocated for this scene.
/// Bring the Schwarz preconditioner up to date for this Newton step.
///
/// EXTRACTED SO BOTH SOLVE PATHS CAN INSTALL IT. The hierarchy is set on the
/// operator BEFORE the caller branches on `aggregate_locked`, so the locked
/// solve gets the preconditioner the scene asked for. A locked scene that took
/// block-Jacobi regardless would answer `precond = schwarz` with a label rather
/// than a preconditioner.
///
/// THE PARTITION IS REUSED AND THE FACTORIZATION IS NOT, and the split follows
/// from what each depends on: the partition depends on the operator's STRUCTURE
/// and the factorization on its VALUES, and the values change at every Newton
/// step.
/// The hierarchy is rebuilt with the factorization rather than with the
/// partition, every level's operator being a Galerkin product of the fine one.
///
/// # Safety
/// The operator's handles must be live for this step and `vertices` must be its
/// row count.
unsafe fn install_schwarz<D: Device>(
    device: &mut D,
    schwarz: &mut super::schwarz::State,
    operator: &Operator,
    vertices: usize,
    param: *const ParamSet,
) -> FatalResult<()> {
    // THE STAND-IN OFFSETS COME FIRST, because the operator view names them for
    // every span this scene does not have.
    let empty = schwarz.empty_offsets(device, vertices as u32)?;
    let rows_view = operator.rows_view(empty);
    if schwarz.rows != vertices as u32 {
        super::schwarz::partition(device, schwarz, rows_view, vertices as u32)?;
    }
    super::schwarz::refactor(device, schwarz, rows_view)?;
    let levels = if (*param).schwarz_levels > 0 {
        (*param).schwarz_levels
    } else {
        2
    };
    super::schwarz::build_hierarchy(device, schwarz, rows_view, vertices as u32, levels)?;
    Ok(())
}

pub unsafe fn advance<D: Device>(
    device: &mut D,
    data: &DataSet,
    param: *mut ParamSet,
    state: &mut SolverState,
) -> FatalResult<StepResult> {
    // Name: Time Per Simulation Step
    // Format: list[(time, ms)]
    // Map: time_per_step
    // Description:
    // Wall-clock time in milliseconds spent inside a single advance call
    // (one simulation step). Note that a step does not advance by a fixed
    // dt: the actual step size is reduced by the accumulated time of
    // impact found during the inner Newton loop, so these values also
    // reflect how hard the solver had to work to progress the step.
    let _section = log::Section::new("advance");
    // THE DOCSTRING ABOVE IS LOAD-BEARING AND ITS PLACEMENT IS TOO.
    // `parsers.rs` registers a channel only when a `Name:` field precedes the
    // call, with everything between `Description:` and the call becoming the
    // text the addon shows. So no note of ours may sit inside that run, which
    // is why this one is below the call rather than above it.
    //
    // The section is the step's own log scope: opening it prints
    // `====== advance ======`, dropping it prints
    // `===== advance: N msec =====` and writes the step's total to
    // `advance.out`.
    //
    // DECLARED FIRST SO IT DROPS LAST. Rust drops locals in reverse
    // declaration order, and every early return below drops it on the way out,
    // so the footer closes a failed step as well as a clean one.
    // PACING, BEFORE ANY WORK. A zero delay is the default and costs a branch
    // on a `Duration` that is already resolved; see `step_delay`. It sits here
    // rather than at the end so a caller sampling between steps observes a run
    // IN PROGRESS rather than one between steps, which is the state the seven
    // observation scenarios assert on.
    let delay = step_delay();
    if !delay.is_zero() {
        std::thread::sleep(delay);
    }

    // STEP A0. Step entry: the result opens optimistic and every failure path
    // below clears exactly the flag it failed on, which is what lets
    // `backend.rs` derive a crash sub-kind without parsing a log.
    let mut result = StepResult {
        time: 0.0,
        ccd_success: true,
        pcg_success: true,
        intersection_free: true,
        newton_progress: true,
        pin_feasible: true,
        contact_separated: true,
    };

    // `time_f32` is written from the double clock BEFORE the record is
    // snapshotted, and the wind ramp reads it, so the order matters.
    (*param).time_f32 = (*param).time as f32;
    // THE SNAPSHOT. It matters
    // that this is a copy and not a borrow: STEP C3 writes `prev_dt` and `time`
    // back through the same pointer, and every field read below must be the one
    // the step OPENED with. Only the fields the driver reads are taken, so the
    // record need not be `Clone` and a field that arrives later cannot be read
    // by accident from a stale copy.
    let prm = StepParams::snapshot(param);
    log::set_time(prm.time);

    let vertices = state.sizes.vertices;
    // THE AGGREGATE LOCK'S GROUPS, read once per step. Zero is the ordinary
    // case and every projector method returns early on it, so an unlocked scene
    // pays one comparison.
    let locked_groups = data.translation_lock.size as usize;
    // How many PDRD rigid bodies. A body's vertices own no per-vertex degrees
    // of freedom, so one body routes the WHOLE system through the reduction.
    let pdrd_bodies = state.rigid.n_bodies;
    // How many SAND grains the scene carries, taken from the SCENE and not from
    // the contact layer.
    //
    // THE DISTINCTION IS NOT ACADEMIC: `disable-contact` leaves
    // `state.contact` at `None`, so reading the count off it reports zero for a
    // scene that plainly has grains, and the three sites below would then
    // condense, recover and integrate nothing. The analytic colliders are
    // assembled OUTSIDE that gate, so a grain resting on a floor still needs
    // every one of them. The count is derived at setup, scanning the rolling
    // inertia once.
    let grains = state.sizes.grains;
    // Safety: the scene is live and holds `locked_groups` records.
    let locked_records: &[crate::data::TranslationLock] =
        super::scene::slice(&data.translation_lock);
    if vertices == 0 {
        result.time = prm.time;
        return Ok(result);
    }
    if data.vertex.curr.size as usize != vertices || data.vertex.prev.size as usize != vertices {
        return Err(Fatal::invariant(format!(
            "solver driver: the scene was allocated for {vertices} vertices and now reports \
             {} current and {} previous positions",
            data.vertex.curr.size, data.vertex.prev.size
        )));
    }

    // THE REFUSALS ARE ASSERTED, NOT ASSUMED. Each of these is refused by name
    // at `initialize()`; reaching one here means the gate and the driver
    // disagree, which is the one failure this backend must never answer with a
    // plausible frame.
    assert_unsupported_absent(data, param)?;

    let curr = state.positions.handle();
    // THE MESH TOPOLOGY, taken once. `mesh_face` and `mesh_edge` are staged
    // buffers the state seeds at allocate and the mesh does not move after the
    // build, so the contact subsystem takes the two handles rather than its own
    // copies. `StagedBuffer::handle` asserts the device copy is current, so a
    // missed upload traps here rather than dispatching stale topology.
    let mesh_refs = state.refs();
    let prev = state.positions_prev.handle();

    // The collision-window masks, taken ONCE and held for the whole step. They
    // are read by every broad-phase pass and by the intersection scan, and the
    // host may install a new table only between steps, so one borrow is both
    // cheaper and the only reading that cannot see a table change mid-step.
    let mut window_guard = super::lock_collision_windows();
    let windows = Windows {
        vertex: window_guard.as_mut().and_then(|table| table.vertex_active()),
        edge: window_guard.as_mut().and_then(|table| table.edge_active()),
        face: window_guard.as_mut().and_then(|table| table.face_active()),
    };

    let _pre_newton = std::time::Instant::now();
    // STEP A1. The broad phase. The three trees are REBUILT from the current
    // pose, not merely refreshed: the Morton order is recomputed, so the tree's
    // topology tracks the geometry rather than the pose the scene was authored
    // in. Skipped whole when the scene set `disable-contact`.
    if !prm.disable_contact {
        let Some(contact) = state.contact.as_mut() else {
            return Err(Fatal::invariant(
                "solver driver: this scene wants contact and the contact subsystem was never \
                 allocated, so the step would run with no broad phase, no barrier, no CCD \
                 filter and no intersection gate. That is a run without the penetration \
                 guarantee, which this backend does not perform",
            ));
        };
        let _phase = super::phase::start("contact.rebuild_trees");
        let _lbvh = std::time::Instant::now();
        contact.rebuild_trees(device, data, mesh_refs, curr, windows)?;
        drop(_phase);
        // Name: LBVH Build Time
        // Format: list[(time, ms)]
        // Map: lbvh_build
        // Description:
        // Wall-clock time in milliseconds to rebuild the LBVH (Linear
        // Bounding Volume Hierarchy) over faces, edges, and vertices at
        // the start of each simulation step. This BVH underpins broad-phase
        // contact detection, so this cost tracks mesh size and how often
        // primitives are deactivated by collision windows.
        log::mark("advance", "lbvh_build", _lbvh.elapsed());
    }

    // STEP A3. The start-of-step velocity, the largest free-vertex speed, and
    // the domain check. One pass over the two position arrays for all three.
    let velocity_args = VelocityTermsArgs {
        current: curr,
        previous: prev,
        prop: state.prop_vertex.handle(),
        previous_dt: prm.prev_dt,
        velocity: state.velocity.handle(),
        speed_squared: state.scalar.handle(),
        reach: state.scalar_b.handle(),
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("step.velocity", &velocity_args, vertices as u32)?;
    // THE TWO FOLDS READ THE MIRRORS, so both come back across the seam first;
    // each went stale at its own `handle()` above.
    //
    // THE VERDICTS ARE SCALARS AND ONLY THE SCALARS MOVE. `max_u` and `reach`
    // are one float each, so folding them on the host would move `vertices`
    // floats twice across the seam. Both folds run on the device and only the
    // four bytes of each verdict come back.
    let (max_u, reach) = {
        let a = state.scalar.handle();
        let b = state.scalar_b.handle();
        // Safety: both buffers outlive the calls and name `vertices` floats.
        let u = unsafe { state.fold.max(device, "step.max_u", a, vertices as u32, 0.0) }?;
        let r = unsafe { state.fold.max(device, "step.reach", b, vertices as u32, 0.0) }?;
        (u.sqrt(), r)
    };
    let domain = position_domain_abi();
    if !(reach < domain) {
        // LEAVING THE DOMAIN IS NOT CAUGHT ANYWHERE ELSE: the overflow guards
        // in the shared coordinate header are behind a debug switch that is
        // never defined, so a position past the end of the representable range
        // is carried on silently and the run continues on nonsense. Ingest is
        // bounded at scene build, so what this catches is a position that MOVED
        // out during the run.
        return Err(Fatal::invariant(format!(
            "solver driver: a vertex has moved to {reach:.6} from the origin, outside the \
             +/-{domain:.6} domain the position representation covers. Nothing past that \
             end is a meaningful position, so the run cannot continue. The scene either drifts \
             or has blown up; check max_u ({max_u:.3e} here) and the frames before this one"
        )));
    }
    // Name: Max Vertex Velocity
    // Format: list[(time, m/s)]
    // Map: max_velocity
    // Description:
    // Maximum speed (in meters per second) among all non-pinned vertices,
    // measured from the previous to the current positions at the start of
    // the step. Pinned (fixed) vertices are excluded. Useful for spotting
    // explosions or abrupt motion in the simulation.
    log::mark("advance", "max_u", max_u as f64);

    // STEP A5. The step size this advance aims at. Host arithmetic on two
    // scalars the scene already carries.
    let mut dt = prm.dt * prm.playback;
    // Name: Target Step Size
    // Format: list[(time, seconds)]
    // Description:
    // Target integration step size in seconds at the start of this
    // simulation step, computed as the configured dt scaled by the current
    // playback rate. The actually advanced step size can be smaller (see
    // the Final Step Size channel) if the line search reduces it.
    log::mark("advance", "dt", dt as f64);
    // Name: Playback Speed
    // Format: list[(time, ratio)]
    // Description:
    // Playback rate applied this step, as a multiplier on the configured
    // dt. A value of 1.0 means real-time playback, below 1.0 slows motion
    // down, and above 1.0 speeds it up. The value can change between
    // steps when the scene scripts playback over time.
    log::mark("advance", "playback", prm.playback as f64);

    // STEP A6. `max_sigma`, the largest shell / rod stretch ratio, measured at
    // the start-of-step pose. Telemetry rather than a term of the solve: nothing
    // below reads it and a scene runs the same trajectory with and without it.
    // It is emitted because it is one of the three indicator streams a lost PCG
    // residual denominator is read off, and a collapsing `SL_toi` is only
    // diagnostic beside it.
    //
    // GATED ON THE TWO ELEMENT COUNTS, so a tet-only scene pays nothing and
    // emits nothing. The reduction is here rather than in the walk because this
    // backend's determinism rests on every fold having one stated shape;
    // `reduce::max`'s floor argument is the identity the fold opens with, which
    // is how the shell and rod folds chain into one verdict.
    if state.sizes.shell_faces > 0 || state.sizes.rods > 0 {
        assemble::stretch_indicator(device, data, state)?;
        let mut max_sigma = 0.0f32;
        {
            let faces = state.sizes.shell_faces as u32;
            let rods = state.sizes.rods as u32;
            let shell = state.stretch.ratio.handle();
            let rod = state.stretch.rod_ratio.handle();
            // Safety: both buffers outlive the calls and name their counts.
            max_sigma =
                unsafe { state.fold.max(device, "step.max_sigma", shell, faces, max_sigma) }?;
            max_sigma =
                unsafe { state.fold.max(device, "step.max_sigma_rod", rod, rods, max_sigma) }?;
        }
        // Name: Max Stretch Ratio
        // Format: list[(time, ratio)]
        // Description:
        // Maximum stretch ratio among all shell faces and rod edges in the
        // scene, measured at the start of the step before the Newton loop.
        // For shells this is the largest singular value of the deformation
        // gradient (scaled by the shrink factor), for rods it is the current
        // edge length divided by its rest length. A value of 1.02 means a
        // 2 percent stretch. Useful for diagnosing strain-limit tightness.
        log::mark("advance", "max_sigma", max_sigma as f64);
    }

    // STEP A7. The implicit target, and the iterate seeded at the current pose.
    // THE PIN INDEX, GATHERED ON THE DEVICE off the record that already holds
    // it. A host pass would be O(vertices) plus one upload per advance for a
    // field `prop_vertex` already carries.
    {
        let count = state.sizes.vertices as u32;
        let args = crate::driver::kernels::VertexFixIndexFromRecordsArgs {
            prop: state.prop_vertex.handle(),
            fix_index: state.fix_index.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("step.fix_index", &args, count) }?;
    }
    // TAKES THE THREE ARRAYS RATHER THAN `&mut SolverState`, so it can be
    // called while other fields of the same state are borrowed, and so no
    // buffer has to be moved out and put back. The put-back version had a real
    // defect: an early return from a failed step never reached the restore,
    // and the NEXT step then gathered the pin indices into an empty slice and
    // handed the shared body a pointer to nothing.
    compute_target(
        device,
        &prm,
        curr,
        prev,
        vertices,
        state.target.handle(),
        state.fix.handle(),
        dt,
        state.fix_index.handle(),
    )?;
    // The iterate starts at the committed pose. DEVICE WORK now that both are
    // device buffers, rather than a round trip through the host.
    state.eval_x.copy_from(device, &state.positions)?;

    // STEP A8. Which vertices leave the system as exact Dirichlet rows. A fix
    // pin inside a PDRD body keeps its barrier instead (it owns no per-vertex
    // DOF), and the A/B lever reverts every pin to the barrier; PDRD is refused
    // here, so the mask is `fix_index > 0` unless the lever is set.
    //
    let disable_dof_removal = prm.disable_pin_dof_removal;
    // ON THE DEVICE, BOTH HALVES. A dispatch over the vertices writes the mask
    // off the record (`vertex_dof_removal_mask`), and the fold below reduces it
    // to one word. Walking every vertex on the host, uploading the mask and
    // summing the host copy would put a per-vertex pass and a reduction on the
    // host, which is the divergence this shape exists to avoid.
    //
    // THE FOLD IS A `u32` ONE, NOT THE FLOAT FORM. `driver::reduce` is the CPU
    // backend's fold arm over host slices, and `vec_block_sum_u32` sits beside
    // the float form for this count: a count folded through `float` is exact
    // only under 2^24, and a scene past sixteen million marked vertices would
    // round its own tally rather than refuse.
    {
        let count = vertices as u32;
        let mask_args = crate::driver::kernels::VertexDofRemovalMaskArgs {
            prop: state.prop_vertex.handle(),
            disable: u32::from(disable_dof_removal),
            mask: state.dof_mask.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("step.dof_mask", &mask_args, count) }?;
    }
    // LEVELS UNTIL ONE WORD, alternating into the scratch and ending in
    // `dof_total`, which is the only thing read back.
    let removed: u32 = {
        let width = crate::driver::state::DOF_FOLD_WIDTH;
        let mut input = state.dof_mask.handle();
        let mut remaining = vertices;
        let mut cursor = 0usize;
        loop {
            let blocks = remaining.div_ceil(width);
            let destination = if blocks == 1 {
                state.dof_total.handle()
            } else {
                state.dof_fold.span(cursor, blocks)
            };
            let args = crate::driver::kernels::VecBlockSumU32Args {
                source: input,
                length: remaining as u32,
                width: width as u32,
                total: destination,
                count: blocks as u32,
                seam_arena_count: 0,
            };
            // Safety: every handle names a live allocation for the call.
            unsafe { device.launch("step.dof_fold", &args, blocks as u32) }?;
            if blocks == 1 {
                break;
            }
            input = state.dof_fold.span(cursor, blocks);
            cursor += blocks;
            remaining = blocks;
        }
        state.dof_total.download(device)?;
        state.dof_total.host()[0]
    };
    // One-time visibility, and silent in a scene with no pins at all. It is
    // not decoration: a driver that skipped the elimination would still hold
    // the pins approximately, so this line and the tracking error are the two
    // things that separate an exact Dirichlet pin from a penalty.
    //
    // A LINE THIS DRIVER EMITS ON ITS OWN ACCOUNT, AND SO IT KEEPS ITS
    // `::log::` PREFIX. The DOF-removal count is not part of the solver's own
    // transcript, so it takes the timestamped Rust convention; only a line
    // belonging to that transcript goes out bare (see `super::log`).
    if !DOF_REPORTED.swap(true, std::sync::atomic::Ordering::SeqCst)
        && (removed > 0 || disable_dof_removal)
    {
        ::log::info!(
            "dof-removal: {removed} pinned vertices eliminated as Dirichlet BCs ({})",
            if disable_dof_removal {
                "DISABLED via env"
            } else {
                "enabled"
            }
        );
    }
    if disable_dof_removal && removed == 0 && !state.fix.is_empty() {
        // The lever leaves every fix pin on the barrier path, which this
        // driver does not assemble. Rather than run a scene whose pins are held
        // by nothing, say so.
        return Err(Fatal::invariant(
            "solver driver: PPF_DISABLE_PIN_DOF_REMOVAL reverts every fix pin to the contact \
             barrier, which this backend does not assemble, so the pins would be held by \
             nothing at all. Unset it to run this scene here",
        ));
    }

    let mut toi_advanced: f64 = 0.0;
    let mut step: u32 = 1;
    let mut final_step = false;
    let mut last_toi: f32 = 1.0;

    // RECORDED AND NOT PRINTED. The head of the step is not one of the
    // solver's own printed scopes, so a printed row here would read, in a
    // transcript diff, as this driver's own noise. The stream file still
    // carries it.
    log::mark(
        "advance",
        "pre_newton",
        log::Marked::quiet(_pre_newton.elapsed().as_secs_f64() * 1000.0),
    );
    loop {
        let _nt_head = std::time::Instant::now();
        // STEP B0. The bounded escape. A bound, not a heuristic: no progress
        // window and no tuned epsilon. See the module docs for why the other
        // escape does not cover this.
        if !final_step && prm.max_newton_steps > 0 && step >= prm.max_newton_steps {
            // BOTH LINES BELONG TO THE SOLVER'S OWN TRANSCRIPT, so both are
            // bare. The two exponents are rebuilt because C's `%.2e` writes
            // `5.96e-09` where Rust's `{:.2e}` writes `5.96e-9`.
            log::message!(
                "### newton stalled: no acceptable step after {step} iterations \
                 (last toi: {}, toi_advanced: {})",
                log::c_exponential(f64::from(last_toi), 2),
                log::c_exponential(toi_advanced, 2)
            );
            log::message!(
                "### an over-constrained configuration cannot be advanced: a prescribed pin \
                 driven into geometry that cannot yield has no way to resolve. Re-author the \
                 pin's path, or make it a soft pull pin."
            );
            result.newton_progress = false;
            return Ok(result);
        }

        // STEP B1. Open the iteration: every accumulator is cleared by the pass
        // that opens the assembly, never by a kernel inside it.
        // THE OPENING CLEAR, on the device. Every element model scatters into
        // this array and the seam refuses a stale mirror, so the zero it opens
        // at is written by a dispatch rather than by the host.
        let force_seed = VecFillArgs {
            array: state.force.handle(),
            value: 0.0,
            count: (3 * vertices) as u32,
            seam_arena_count: 0,
        };
        device.launch("step.force.fill", &force_seed, (3 * vertices) as u32)?;
        // THE SEARCH DIRECTION opens at zero on the device, as the right-hand
        // side above does.
        let dx_seed = VecFillArgs {
            array: state.dx.handle(),
            value: 0.0,
            count: (3 * vertices) as u32,
            seam_arena_count: 0,
        };
        device.launch("step.dx.fill", &dx_seed, (3 * vertices) as u32)?;
        // THE BLOCK DIAGONAL opens at zero on the device, as the two arrays
        // above do; momentum and the Dirichlet rows accumulate into it.
        let diagonal_seed = VecFillArgs {
            array: state.diagonal.handle(),
            value: 0.0,
            count: (9 * vertices) as u32,
            seam_arena_count: 0,
        };
        device.launch("step.diagonal.fill", &diagonal_seed, (9 * vertices) as u32)?;
        {
            // The per-step clear, which is now a device fill rather than a host
            // one. `Buffer::size` zeroes at allocation and this is the per-step
            // reset the doc there says it is not a substitute for.
            let bytes = state.fixed_values.len() * std::mem::size_of::<f32>();
            let handle = state.fixed_values.handle();
            device.fill_zero(handle, bytes)?;
        }

        // THE STEP ANNOUNCES ITSELF, and the two wordings are a HOST CONTRACT.
        // `------ newton step N ------` opens an ordinary iteration and
        // `------ error reduction step ------` the trailing one. A reader
        // following a run, and a gate counting iterations, have nothing to read
        // without these two lines: a structurally marked step that prints
        // nothing shows a scene finishing with no evidence that any Newton
        // iteration happened.
        if final_step {
            log::message!("------ error reduction step ------");
        } else {
            log::message!("------ newton step {step} ------");
        }

        // STEP B3. The trailing error-reduction iteration: shrink the clock to
        // what was actually integrated and bring the kinematic pins back in
        // step with it BEFORE the targets are rebuilt off them.
        if final_step {
            if toi_advanced < 1.0 && !state.fix.is_empty() {
                let back = 1.0 - toi_advanced as f32;
                let pins = state.fix.len() as u32;
                let rewind_args = RewindFixArgs {
                    fix: state.fix.handle(),
                    back,
                    count: pins,
                    seam_arena_count: 0,
                };
                device.launch("step.rewind_fix", &rewind_args, pins)?;
            }
            // IN DOUBLE, THEN NARROWED, WHICH IS NOT THE OBVIOUS SPELLING.
            // `dt` is fp32 and `toi_advanced` is double, so the product is
            // formed in double and rounded ONCE on the way back. Multiplying in
            // f32 rounds twice and gives a different final step size, which is
            // the clock the whole trailing iteration integrates against.
            dt = (f64::from(dt) * toi_advanced) as f32;
            compute_target(
                device,
                &prm,
                curr,
                prev,
                vertices,
                state.target.handle(),
                state.fix.handle(),
                dt,
                state.fix_index.handle(),
            )?;
        }

        // The pattern's handles, taken once: both matrices below are built
        // over the SAME pattern and each keeps its own copy, which is what
        // makes a push look its block up in the pattern that matrix was built
        // over rather than in whichever one a caller happened to pass.
        let pattern_refs = state.fixed_pattern_refs();
        // The two matrices, wrapped fresh over the scene's pattern. The value
        // arrays are the state's, so nothing here allocates.
        // THE PATTERN IS WALKED ONCE PER RUN, NOT ONCE PER ITERATION. The
        // verdict is cached on the pattern's identity, which is what makes the
        // skip safe: a different pattern re-validates in full.
        let mut checked = state.fixed_pattern_checked;
        let mut fixed = FixedCsr::adopt_validated(
            device,
            pattern_refs,
            data,
            std::mem::take(&mut state.fixed_values),
            &mut checked,
        )?;
        state.fixed_pattern_checked = checked;

        // THE ASSEMBLY SCOPE OPENS HERE, NOT AT THE MOMENTUM LAYER. It opens
        // immediately above this seed and closes after the `num_contact` mark,
        // so it encloses the seed, every elastic layer, the snapshot, both
        // strain limiters, the whole contact assembly and the three marks. It
        // must not open at the momentum layer and close after the stitch: that
        // measures a strict subset and reports it under the `matrix_assembly`
        // name, and the printed order is the tell, since `> matrix_assembly`
        // belongs AFTER `* num_contact`.
        let _assembly = std::time::Instant::now();
        // STEP B4. Seed the search direction on every prescribed row.
        let seed_args = DxSeedArgs {
            eval_x: state.eval_x.handle(),
            target: state.target.handle(),
            prop: state.prop_vertex.handle(),
            dx: state.dx.handle(),
            count: vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("step.dx_seed", &seed_args, vertices as u32)?;

        // RECORDED AND NOT PRINTED: the head of a Newton iteration is not one
        // of the solver's own printed scopes. Its span overlaps the assembly's,
        // which is fine, the two being independent clocks.
        log::mark(
            "advance",
            "nt_head",
            log::Marked::quiet(_nt_head.elapsed().as_secs_f64() * 1000.0),
        );
        // STEP B6. The momentum layer. INSIDE the assembly scope opened at
        // STEP B4 and OUTSIDE the `asm_elastic` sub-timer below, so the
        // `matrix assembly` total counts it and `asm_elastic` does not.
        let _phase = super::phase::start("assemble.momentum");
        assemble::momentum(device, data, param, state, dt)?;
        drop(_phase);

        // THE ELASTIC SUB-TIMER spans the five elastic layers alone: the
        // momentum layer above and the stitch below are outside it, so the
        // channel measures elasticity and nothing else.
        let _asm_elastic = std::time::Instant::now();
        // STEP B7a. The rod bending layer, FIRST of the five elastic
        // dispatches, so it is assembled before every other elastic term. The
        // order is load-bearing because the assembled sums are fp32.
        assemble::rod_bend(device, data, state, &mut fixed, dt)?;

        // STEP B7b. The rod stretch layer, SECOND of the five.
        assemble::rod_stretch(device, data, state, &mut fixed, dt)?;

        // STEP B7c. The shell membrane layer. BEFORE the tets: the five
        // elastic dispatches run as rod bend, rod, shell face, tet and hinge,
        // and the assembled sums are fp32, so the order decides the last bits.
        let _phase = super::phase::start("assemble.shell_membrane");
        assemble::shell_membrane(device, data, prm.eiganalysis_eps, state, &mut fixed, dt)?;
        drop(_phase);

        // STEP B7d. The tet elastic layer.
        assemble::tet_elastic(device, data, prm.eiganalysis_eps, state, &mut fixed, dt)?;

        // STEP B7e. The shell hinge bending layer, LAST of the five elastic
        // dispatches, so it is assembled after the membrane and after the
        // tets.
        let _phase = super::phase::start("assemble.shell_bending");
        assemble::shell_bending(device, data, state, &mut fixed, dt)?;
        drop(_phase);
        // Name: Assembly: Elastic
        // Format: list[(time, ms)]
        // Description:
        // Diagnostic sub-timer of "matrix assembly": wall-clock ms spent
        // assembling the elastic (membrane / bending / solid) Hessian and
        // force into the fixed matrix.
        log::mark("advance", "asm_elastic", _asm_elastic.elapsed());

        // STEP B10. The cross-stitch layer. IT RUNS AHEAD OF B9 IN THIS FILE,
        // because the stitch blocks must land in the live matrix BEFORE that
        // matrix is snapshotted, and the snapshot is what the contact stiffness
        // contracts. Placing it after the snapshot would leave every contact on
        // a stitched scene measuring itself against a matrix the stitch blocks
        // are missing from.
        let _phase = super::phase::start("assemble.stitch");
        assemble::stitch(device, data, prm.stitch_length_factor, state, &mut fixed)?;
        drop(_phase);

        // STEP B9. Snapshot the elastic matrix. The contact stiffness reads
        // this reference rather than the matrix it is about to add to, so the
        // snapshot boundary is where it is for a reason: moving it would change
        // what a later phase reads.
        {
            // THE SNAPSHOT IS A DEVICE-TO-DEVICE COPY, not a host round trip:
            // the seam's copy verb is what keeps the matrix on the device
            // instead of reading it back and writing it out again.
            let _asm_copy = std::time::Instant::now();
            state.tmp_fixed_values.copy_from(device, fixed.value_readback())?;
            // Name: Assembly: Fixed Copy
            // Format: list[(time, ms)]
            // Description:
            // Diagnostic sub-timer of "matrix assembly": wall-clock ms to snapshot
            // the elastic fixed matrix into tmp_fixed (the contact-stiffness
            // reference read by the contact assembly).
            log::mark("advance", "asm_copy", _asm_copy.elapsed());
        }

        // STEP B12 and B13. The two strain limiters, shell then rod, reading
        // the snapshot above and writing the live
        // matrix. Both are dispatched on an element count alone, so a scene
        // reaches them the moment it carries a shell face or a rod whose
        // material sets `strain-limit`.
        //
        // THE SNAPSHOT IS WRAPPED RATHER THAN BORROWED, so the two matrices are
        // separate objects the borrow checker can hold at once, and the values
        // go back to the state afterwards. Nothing allocates: `adopt_from_dataset`
        // reuses the array it is handed.
        {
            let mut checked = state.fixed_pattern_checked;
            let mut reference = FixedCsr::adopt_validated(
                device,
                pattern_refs,
                data,
                std::mem::take(&mut state.tmp_fixed_values),
                &mut checked,
            )?;
            state.fixed_pattern_checked = checked;
            // THE STRAIN-LIMIT SUB-TIMER spans both limiters and stops BEFORE
            // the contact assembly opens its own scope. The `shell` and `rod`
            // results are inspected at the bottom of this block rather than
            // here, so what the channel measures is the two dispatches
            // themselves.
            let _asm_strainlimit = std::time::Instant::now();
            let shell = assemble::shell_strain(
                device,
                data,
                prm.eiganalysis_eps,
                prm.barrier,
                state,
                &mut reference,
                &mut fixed,
            );
            let rod = if shell.is_ok() {
                assemble::rod_strain(device, data, prm.barrier, state, &mut reference, &mut fixed)
            } else {
                Ok(())
            };
            // Name: Assembly: Strain Limit
            // Format: list[(time, ms)]
            // Description:
            // Diagnostic sub-timer of "matrix assembly": wall-clock ms for the
            // strain-limiting Hessian / force contributions.
            log::mark("advance", "asm_strainlimit", _asm_strainlimit.elapsed());
            // STEP B15. Contact assembly, INSIDE this block because it reads the
            // same elastic snapshot: `barrier::compute_stiffness` measures the
            // contact's stiffness against the elastic Hessian the contact is
            // about to be added to, and reading the live matrix instead would
            // let one contact's stiffness depend on another's already being
            // assembled.
            // THE STATISTICS COUNTER IS CLEARED ONCE PER NEWTON ITERATION: it
            // counts the contacts THIS iteration's
            // assembly deposits, so a counter carried across iterations would
            // report the running total of every iteration in the step. The
            // clear runs whether or not the assembly is taken, so a step that
            // skips contact reports zero rather than the previous iteration's
            // count.
            // THE PER-OBJECT STATISTICS CHANNEL, built once for the three
            // assemblies below because all three record into ONE counter, and
            // rebuilt each iteration because taking the counter's handle stales
            // its mirror, which is what makes `fetch` download before reading.
            let statistics = crate::driver::contact::StatisticsRefs {
                contact_count_size: state.statistics_contact_count.len() as u32,
                contact_count: state.statistics_contact_count.handle(),
                object_index_size: state.statistics_object_index.len() as u32,
                object_index: state.statistics_object_index.handle(),
                static_object_index_size: state.statistics_static_object_index.len() as u32,
                static_object_index: state.statistics_static_object_index.handle(),
            };
            {
                let objects = state.statistics_contact_count.len();
                if objects > 0 {
                    let counter = state.statistics_contact_count.handle();
                    device.fill_zero(counter, objects * size_of::<u32>())?;
                }
            }

            // THE COLLAPSED-SEPARATION SLOTS ARE CLEARED ONCE, HERE, BEFORE
            // THE FIRST OF THE TWO ASSEMBLY PASSES. Every pass writes its
            // query's slot first-writer-wins, so a clear BETWEEN them drops the
            // earlier pass's report before `collect_assembly_overlap` below
            // reads it. Clearing at the top of `Contact::assemble` would be
            // correct only if that pass ran first, and it does not, so the
            // collision-mesh kinds 7, 8 and 9 would be written and then
            // erased. The line search clears them again
            // before the sweep, which reads them under its own decoder.
            if let Some(contact) = state.contact.as_mut() {
                contact.clear_overlap_slots(device)?;
                if shell.is_ok() && rod.is_ok() && !prm.disable_contact {
                    contact.refresh_contact_queries(
                        device, data, mesh_refs, state.eval_x.handle(), windows,
                    )?;
                }
            }
            // THE RESIDUAL SNAPSHOT THIS PASS'S FRICTION ANCHORS ON. Taken
            // here rather than read out of `force` directly because the
            // contacts below deposit into `force` through atomics, so a pair
            // reading that vector would read a row another pair had already
            // moved. TWO SNAPSHOTS, NOT ONE, because this driver deposits both
            // halves straight into `force` rather than accumulating the
            // self-contact half into a vector of its own and folding it in at
            // the end. The second copy therefore carries the analytic and
            // collision-mesh forces, which is what the self-contact pass must
            // anchor on.
            {
                let bytes = 3 * state.sizes.vertices * size_of::<f32>();
                let (residual, force) = (state.residual.handle(), state.force.handle());
                device.copy(residual, 0, force, 0, bytes)?;
            }
            // STEP B15a. The constraint layer: geometry outside the solved
            // namespace. The analytic sphere and floor barriers run whether or
            // not `disable-contact` is set; the collision-mesh half is inside
            // that gate.
            //
            // FIRST, before the self-contact assembly, and reading the SAME
            // elastic snapshot. The self-contact block below says why the order
            // is load-bearing.
            let constraint_result = if shell.is_ok() && rod.is_ok() {
                let SolverState {
                    analytic: analytic_state,
                    contact: contact_state,
                    force,
                    diagonal,
                    eval_x,
                    fix,
                    sphere,
                    floor,
                    push,
                    // THE THREE GRAIN SCHUR BLOCKS, per vertex and owned here.
                    // They are sized with the scene whenever it carries grains,
                    // so the analytic pass writes into an array that exists
                    // whether or not this scene has a collider.
                    grain_angular: grain_angular_buffer,
                    grain_coupling: grain_coupling_buffer,
                    grain_rotational: grain_rotational_buffer,
                    residual: residual_buffer,
                    ..
                } = &mut *state;
                let residual = residual_buffer.handle();
                let grain_angular = grain_angular_buffer.handle();
                let grain_coupling = grain_coupling_buffer.handle();
                let grain_rotational = grain_rotational_buffer.handle();
                let iterate = eval_x.handle();
                let Some(analytic) = analytic_state.as_mut() else {
                    return Err(Fatal::invariant(
                        "solver driver: the analytic collider assembly was reached with \
                         no collider layer allocated, so a scene's sphere or \
                         floor would hold nothing at all",
                    ));
                };
                let mut outcome = analytic.assemble(
                    device,
                    data,
                    mesh_refs,
                    param,
                    curr,
                    iterate,
                    fix.handle(),
                    sphere,
                    floor,
                    &mut reference,
                    &mut fixed,
                    push,
                    force.handle(),
                    diagonal.handle(),
                    grain_angular,
                    grain_coupling,
                    grain_rotational,
                    residual,
                    statistics,
                );
                if outcome.is_ok() && !prm.disable_contact {
                    if let Some(contact) = contact_state.as_mut() {
                        outcome = contact.assemble_collision_mesh_prepared(
                            device,
                            data,
                            mesh_refs,
                            param,
                            curr,
                            iterate,
                            &mut reference,
                            &mut fixed,
                            force.handle(),
                            push,
                            residual,
                            statistics,
                        );
                    }
                }
                outcome
            } else {
                Ok(())
            };
            // THE RESIDUAL SNAPSHOT THIS PASS'S FRICTION ANCHORS ON. Taken
            // here rather than read out of `force` directly because the
            // contacts below deposit into `force` through atomics, so a pair
            // reading that vector would read a row another pair had already
            // moved. TWO SNAPSHOTS, NOT ONE, because this driver deposits both
            // halves straight into `force` rather than accumulating the
            // self-contact half into a vector of its own and folding it in at
            // the end. The second copy therefore carries the analytic and
            // collision-mesh forces, which is what the self-contact pass must
            // anchor on.
            {
                let bytes = 3 * state.sizes.vertices * size_of::<f32>();
                let (residual, force) = (state.residual.handle(), state.force.handle());
                device.copy(residual, 0, force, 0, bytes)?;
            }
            // THE SELF-CONTACT ASSEMBLY, timed as `asm_contact`.
            //
            // AFTER the analytic and collision-mesh contacts, and the order is
            // load-bearing rather than
            // incidental: every friction term anchors itself on the
            // residual assembled so far, and the contacts above are the ones
            // carrying the normal load in most scenes, so a mesh contact built
            // after them reads a tangential drive with that load already
            // removed. The reverse order would hand a vertex resting on a floor
            // its full weight as the sideways drive of a contact against a
            // neighbor. Both are fp32 running sums either way, so the order is
            // not free to change back.
            let _asm_contact = std::time::Instant::now();
            let contact_result = if shell.is_ok() && rod.is_ok() && !prm.disable_contact {
                let SolverState {
                    contact: contact_state,
                    force,
                    eval_x,
                    push,
                    grain_omega: omega,
                    grain_torque: torque,
                    grain_stiffness: stiffness,
                    grain_normal: normal,
                    residual: residual_buffer,
                    ..
                } = &mut *state;
                let residual = residual_buffer.handle();
                let iterate = eval_x.handle();
                let grain_omega = omega.handle();
                let (grain_torque, grain_stiffness, grain_normal) =
                    (torque.handle(), stiffness.handle(), normal.handle());
                match contact_state.as_mut() {
                    Some(contact) => contact.assemble_prepared(
                        device,
                        data,
                        mesh_refs,
                        param,
                        curr,
                        iterate,
                        &mut reference,
                        &mut fixed,
                        force.handle(),
                        push,
                        grain_omega,
                        grain_torque,
                        grain_stiffness,
                        grain_normal,
                        residual,
                        statistics,
                    ),
                    None => Err(Fatal::invariant(
                        "solver driver: the contact assembly was reached with no contact \
                         subsystem allocated",
                    )),
                }
            } else {
                Ok(())
            };
            // Given back before any failure is reported, so a step that stops
            // here leaves the state holding its buffer rather than an empty one
            // the next step would size again.
            state.tmp_fixed_values = reference.into_values();
            shell?;
            rod?;
            // Name: Assembly: Contact
            // Format: list[(time, ms)]
            // Description:
            // Diagnostic sub-timer of "matrix assembly": wall-clock ms for the
            // self-contact / collision-mesh Hessian + force assembly (the CSR fill
            // path, including the dynamic-matrix rebuild).
            log::mark("advance", "asm_contact", _asm_contact.elapsed());
            contact_result?;
            constraint_result?;
            // A PAIR THE ASSEMBLY FOUND ALREADY COLLAPSED TO ITS OFFSET ends
            // the advance here, and the report is a HOST CONTRACT: the kind
            // string, the pair, the world-space lengths, and for a
            // collision-mesh kind the note saying which space the second index
            // is in. `contact_separated = false` is what turns it into a
            // structured OverlappingStart on the host.
            if let Some(contact) = state.contact.as_mut() {
                if let Some(overlap) = contact.collect_assembly_overlap(device)? {
                    log::message!("### {}", overlap.describe());
                    if overlap.collision_mesh {
                        log::message!(
                            "### the second index is in the static collision-mesh \
                             vertex space; the first is a dynamic vertex."
                        );
                    }
                    log::message!(
                        "### an intersection allowance suppresses the BUILD-TIME report \
                         of an overlap; it does not make one solvable, and contact never \
                         consults it. A scene admitted that way still has to start with \
                         its surfaces apart."
                    );
                    log::message!(
                        "### give the initial geometry a small clearance so nothing starts \
                         in contact, or check whether a stitch or pin is pulling elements \
                         together faster than contact can resolve."
                    );
                    result.contact_separated = false;
                    return Ok(result);
                }
            }
        }
        // How many contact pairs the assembly deposited. Telemetry rather than
        // a term of the solve, and the one
        // stream that separates a scene whose geometry never touches from a
        // backend whose broad phase found nothing.
        {
            let self_contact = state
                .contact
                .as_ref()
                .map_or(0, |contact| contact.assembled);
            let constraint = state
                .analytic
                .as_ref()
                .map_or(0, |analytic| analytic.assembled);
            // Name: Total Contact Count
            // Format: list[(time, count)]
            // Description:
            // Total number of active contact and constraint pairs assembled
            // into the system matrix for this Newton iteration, summed across
            // self-contact, collision-mesh contact, and analytic constraints
            // (sphere, floor). A useful proxy for how crowded the collision
            // scene is at this iteration.
            log::mark("advance", "num_contact", (self_contact + constraint) as f64);
        }
        // Name: Matrix Assembly Time
        // Format: list[(time, ms)]
        // Description:
        // Wall-clock time in milliseconds spent assembling the global
        // system matrix and right-hand side for the Newton linear solve,
        // including inertia, elastic, stitch, strain-limiting, and contact
        // contributions. One entry per Newton iteration.
        //
        // CLOSED AFTER `num_contact` AND NOT BEFORE IT, which is what puts
        // `> matrix_assembly` below `* num_contact` in the stream.
        log::mark("advance", "matrix_assembly", _assembly.elapsed());

        // STEP B15b. THE SAND GRAIN'S SPIN, CONDENSED OUT OF THE NEWTON SYSTEM.
        // After the whole system is assembled and BEFORE the Dirichlet removal
        // and the solve. The grain's
        // angular degree of freedom is eliminated into the translation block
        // it couples to, so the solve below stays a translation-only system.
        if grains > 0 {
            // THE STATE'S OWN BLOCKS, present whenever the scene has grains.
            // Asking the collider and substituting `Handle::NONE` when there is
            // none left this pass with an unresolvable buffer in exactly the
            // scenes that carry grains and no floor or sphere.
            let (angular, coupling, rotational) = (
                state.grain_angular.handle(),
                state.grain_coupling.handle(),
                state.grain_rotational.handle(),
            );
            let condense = SandGrainCondenseRowArgs {
                inverse_center_inertia: state.grain_inv_inertia_center.handle(),
                angular,
                coupling,
                rotational_gradient: rotational,
                previous_omega: state.grain_omega_prev.handle(),
                diagonal: state.diagonal.handle(),
                force: state.force.handle(),
                dt,
                // The fraction of a grain's spin angular momentum fed back into
                // its translation, read from the environment and defaulting to
                // 0.5, which is near-textbook rolling while staying bounded:
                // fully included it is a positive-feedback runaway.
                spin_couple: sand_spin_couple(),
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("sand.condense", &condense, vertices as u32)?;
        }

        let _dirichlet = std::time::Instant::now();
        // STEP B16. Dirichlet DOF removal, both passes, WITH the lifting term,
        // over BOTH matrices. The contact matrix carries a moving collider's
        // coupling to the geometry it is advancing into, so a reduction that
        // skipped it would drop exactly the signal the lifting term exists to
        // deliver, and the step would deadlock rather than compute a wrong
        // number.
        //
        // The flat contact arrays are TAKEN and given back, for the reason
        // `tmp_fixed_values` is: the lift needs the driver's force, iterate and
        // target at the same time as the matrix, and they live in one struct.
        if removed > 0 {
            let mut flat = state
                .contact
                .as_mut()
                .map(|contact| std::mem::take(&mut contact.flat));
            let dynamic = if let Some(flat) = flat.as_mut() {
                dirichlet::lift(device, state, flat.offset.handle(), flat.index.handle(), flat.value.handle())
            } else {
                Ok(())
            };
            let (offset, column) =
                (state.fixed_offset.handle(), state.fixed_index.handle());
            let fixed_lift = if dynamic.is_ok() {
                dirichlet::lift(device, state, offset, column, fixed.values_mut())
            } else {
                Ok(())
            };
            if let (Some(flat), Some(contact)) = (flat, state.contact.as_mut()) {
                contact.flat = flat;
            }
            dynamic?;
            fixed_lift?;
            dirichlet::prescribe(device, state)?;
            // THE CHANNEL CARRIES NO DOCSTRING ABOVE IT, so it is deliberately
            // absent from `session.get.log.names()`, and `asm_dirichlet` is the
            // name it is read under.
            //
            // INSIDE THE `removed > 0` BRANCH: a scene with no prescribed row
            // does no Dirichlet work and so prints no line, and marking it
            // unconditionally here would put a row in the stream for work that
            // did not happen.
            log::mark("advance", "asm_dirichlet", _dirichlet.elapsed());
        }
        // THE LINEAR SOLVE'S TIMER OPENS HERE: after the Dirichlet scope
        // closes and before anything the solve needs is prepared. That places
        // the block-Jacobi build inside it, the preconditioner being part of
        // the solve, and it covers all THREE of this driver's solve branches.
        // A timer around the unreduced branch alone would leave every PDRD and
        // every locked scene with no `> linsolve` line at all.
        let _linsolve = std::time::Instant::now();
        // STEP B17. The linear solve.
        // The pattern handles are taken BEFORE the destructure below, which
        // holds the state's fields disjointly and so cannot lend the whole of
        // it to a method.
        let pattern_refs = state.fixed_pattern_refs();
        // The disjoint field borrows the solve needs, taken once: the operator
        // reads the block diagonal while the preconditioner, the right-hand
        // side, the direction and the PCG workspace are written.
        let SolverState {
            diagonal,
            precond_diagonal,
            precond_inverse: inverse,
            force,
            eval_x,
            dx,
            pcg: work,
            contact: contact_state,
            ..
        } = state;
        let operator = Operator {
            dynamic: contact_state.as_mut().map(|contact| DynamicView {
                offset: contact.flat.offset.handle(),
                index: contact.flat.index.handle(),
                value: contact.flat.value.handle(),
                // THE TRANSPOSE ARRAYS ARE SPANNED TO THE STORED TOTAL, which
                // is what a host truncate would do: the pass sizes them for the
                // widest case and only the kernel knows the real width.
                transpose_offset: contact.transpose.offset.handle(),
                transpose_index: contact.transpose.index.span(0, contact.transpose.total),
                transpose_value: contact.transpose.value.span(0, contact.transpose.total),
            }),
            fixed: fixed.view(),
            diagonal: diagonal.span(0, 9 * vertices),
        };
        operator.precond_diagonal(device, precond_diagonal.span(0, 9 * vertices))?;
        // THE INVERSION IS A DISPATCH, not a host walk. The loop over the rows
        // is the kernel's, and so is the verdict, which comes back through the
        // diagnostic lane rather than as a returned row. The payload the entry
        // raises is the largest eigenvalue, the two extremes of the spectrum
        // and the row, which is everything a per-row message would otherwise
        // read out of the block itself.
        let invert_args = BlockJacobiInvertRowArgs {
            diagonal: precond_diagonal.span(0, 9 * vertices),
            inverse: inverse.span(0, 9 * vertices),
            count: vertices as u32,
            seam_arena_count: 0,
        };
        let inverted = device.run("step.block_jacobi_invert", |encoder| {
            // Safety: both arrays are borrowed for the whole call and hold one
            // 3x3 block per vertex.
            unsafe { encoder.elements(&invert_args, vertices as u32) }
        });
        if let Err(fault) = inverted {
            return Err(match fault {
                Fault::Device { diag, .. } => Fatal::invariant(format!(
                    "solver driver: {} block-Jacobi diagonal block(s) are not positive \
                     definite. Every block is inverted through a floored symmetric \
                     eigendecomposition, so this means the assembled block is NaN, \
                     infinite, or has a non-positive largest eigenvalue, which is an \
                     assembly defect upstream rather than a tolerance. The first, by row, \
                     reports [lambda_max, lambda_min, lambda_max_again, row] as {}",
                    diag.failures,
                    diag.first
                        .as_ref()
                        .map_or(String::new(), |first| format!("{first}"))
                )),
                other => Fatal::from(other),
            });
        }
        // THE LOCKED SOLVE IS A DIFFERENT SYSTEM, not the same one with extra
        // steps, so it is a branch of its own:
        // the dispatcher prepares the affine feasible correction against the
        // positions the frames were built at, runs `Q M Q z = Q (b - M q)`, and
        // verifies the solved correction carries no forbidden angular
        // increment.
        let report = if pdrd_bodies > 0 {
            // THE REDUCED SOLVE, taken whenever
            // the scene carries a body. A PDRD body's vertices own no per-vertex
            // degrees of freedom, so the whole system, cloth rows included,
            // routes through the reduction as soon as one exists.
            //
            // THE FIT AND THE SCATTER RUN FIRST AND EVERY ITERATION. The body's
            // rotation moves under the Newton loop, and both the prolongation
            // and the force restriction read each vertex's rotated rest vector
            // as the rigid Jacobian's moment arm; fitting once per step would
            // leave the reduction working in a frame the body has left.
            let SolverState {
                rigid,
                rigid_staged,
                rigid_state,
                rigid_blocks,
                rigid_factor,
                rigid_fit_scratch,
                rigid_rotation_step,
                pcg_rigid,
                eval_x,
                prop_vertex,
                // THE LOCK STATE, for a scene that carries both. The reduced
                // solve projects the FULL-SPACE product inside its own operator
                // rather than being wrapped by the lock, which is the order
                // `cg_rigid_translation_locked` uses and the only order that is
                // the same operator.
                lock,
                pcg_locked,
                translation_lock,
                translation_lock_index,
                translation_lock_initial,
                dof_mask,
                rigid_running_rotation,
                rigid_rotation_seeded,
                ..
            } = state;
            let body_vertices = data.pdrd_vert_list.size;
            let vert_list = rigid_staged.vert_list.handle();
            let body_prop = rigid_staged.body_prop.handle();
            let rest_centered = rigid_staged.rest_centered.handle();
            let mut reduction = super::pdrd::Reduction {
                map: rigid,
                staged: rigid_staged,
            };
            reduction.fit(
                device,
                vert_list,
                prop_vertex.handle(),
                body_prop,
                rest_centered,
                eval_x.handle(),
                rigid_fit_scratch.handle(),
                rigid_state.handle(),
                body_vertices,
                &pcg_rigid.zero,
            )?;
            // SEED THE RUNNING ROTATION FROM THE POSE THIS RUN WAS HANDED,
            // once, behind the `pdrd_rprev_seeded` latch. The fit above has
            // just written each body's absolute
            // best-fit rotation into `rigid_state`, so the seed is a copy.
            //
            // THE IDENTITY IS ONLY RIGHT ON A FRESH SCENE. An unrotated body
            // fits to the identity and the two agree, which is why this is
            // invisible on every run that starts from the authored pose. A
            // `--load` resume starts from a pose whose bodies have already
            // turned, and seeding the identity there aims the anchored rigidify
            // at the authored orientation rather than the saved one.
            //
            // AFTER THE FIRST STEP IT MUST NOT RUN AGAIN: the running rotation
            // is then the composition of the increments the line search
            // accepted, and re-fitting it to a contact-sheared pose is the
            // drift the anchored rigidify exists to avoid.
            if !*rigid_rotation_seeded {
                reduction.copy_state_rotation(
                    device,
                    rigid_state.handle(),
                    rigid_running_rotation.handle(),
                )?;
                *rigid_rotation_seeded = true;
            }
            reduction.scatter_rotated_rest(
                device,
                vert_list,
                prop_vertex.handle(),
                rigid_state.handle(),
                rest_centered,
                body_vertices,
            )?;
            let empty_dyn = (
                reduction.staged.empty_dyn_offset.handle(),
                reduction.staged.empty_dyn_index.handle(),
                reduction.staged.empty_dyn_value.handle(),
            );
            let precond_inputs = super::pdrd::PrecondInputs {
                vert_list,
                vertex_prop: prop_vertex.handle(),
                state: rigid_state.handle(),
                rest_centered,
                // AN EMPTY MATRIX WHEN THERE IS NO CONTACT, never
                // `Handle::NONE`. The sandwich row resolves all three buffers
                // before its body runs and then indexes `dyn_offset[v]` and
                // `dyn_offset[v + 1]` unconditionally, so a scene that
                // configures no contact needs a real `rows + 1` run of zeros
                // here rather than an absent buffer. `Handle::NONE` carries
                // `u32::MAX` as its arena and trapped the entry's own assert.
                dyn_index: operator
                    .dynamic
                    .as_ref()
                    .map_or_else(|| empty_dyn.1, |d| d.index),
                dyn_offset: operator
                    .dynamic
                    .as_ref()
                    .map_or_else(|| empty_dyn.0, |d| d.offset),
                dyn_value: operator
                    .dynamic
                    .as_ref()
                    .map_or_else(|| empty_dyn.2, |d| d.value),
                fixed_index: pattern_refs.index,
                fixed_offset: pattern_refs.offset,
                fixed_value: fixed.value_handle(),
                rows: pattern_refs.rows,
                body_vertices,
            };
            reduction.build_precond(
                device,
                &precond_inputs,
                rigid_blocks,
                rigid_factor,
                dt,
            )?;
            // The projector, built exactly as the locked branch builds it, and
            // `None` when the scene carries no lock so the reduced apply keeps
            // its unlocked shape.
            let mut lock_projector = if locked_groups > 0 {
                let lock_rows = super::lock::RowInputs {
                    lock_index: translation_lock_index.handle(),
                    locks: translation_lock.handle(),
                    prop: prop_vertex.handle(),
                    positions: eval_x.handle(),
                    dof_mask: dof_mask.handle(),
                    initial: translation_lock_initial.handle(),
                };
                let mut projector = super::lock::Projector {
                    scratch: lock,
                    groups: locked_groups,
                    vertices,
                    rows: lock_rows,
                };
                projector.prepare(
                    device,
                    locked_records,
                    dx.handle(),
                    pcg_locked.q.handle(),
                )?;
                // THE HINGE-AND-LOCK FEASIBILITY CHECK, which the device side
                // already assumes has run:
                // `pdrd_translation_lock_particular_row` returns early for a
                // hinged body on the stated grounds that the host has checked
                // it. Without this the early return silently discards a
                // correction the scene asked for rather than refusing an
                // impossible one. `prepare` has just written the drift the
                // verdict reads.
                super::lock::check_hinge_lock_feasible(
                    device,
                    &mut projector.scratch.drift,
                    locked_records,
                    &reduction.map.jmode,
                    &reduction.map.tlock,
                )?;
                Some(projector)
            } else {
                None
            };
            let solved = pcg::solve_rigid(
                device,
                &operator,
                &mut reduction,
                rigid_factor.handle(),
                inverse.span(0, 9 * vertices),
                force.handle(),
                dx.handle(),
                work,
                pcg_rigid,
                // THE ROTATION EXPORT, and it is not optional on this path.
                // The rigidify commit rebuilds each body from the running
                // rotation, so a rotation the solve took but never exported is
                // one the commit then OVERWRITES: the body would translate
                // correctly and never turn, which is worse than not rigidifying
                // at all because the motion is silently wrong rather than
                // absent.
                Some(rigid_rotation_step.handle()),
                lock_projector.as_mut(),
                translation_lock.handle(),
                prm.cg_tol,
                prm.cg_max_iter,
            )?;
            // THE TANGENT CHECK ON THIS ARM TOO, easy to read as belonging
            // only to the deformable
            // path. A PDRD body's own six DOFs are constrained exactly by the
            // reduced projector, but a locked GROUP in a scene that also
            // carries a rigid body still has deformable and SAND members, and
            // the aggregate projector is all that constrains their rows.
            //
            // GATED ON A CONVERGED SOLVE, as it is on the arm below: an
            // unconverged correction has not been projected to completion, so a
            // violation read off it would raise a fatal where an ordinary
            // failed step is the right answer.
            if let Some(projector) = lock_projector.as_mut() {
                if matches!(
                    solved.outcome,
                    pcg::Outcome::Converged | pcg::Outcome::CurvatureTruncated
                ) {
                    eval_x.download(device)?;
                    let mut step_host = vec![0.0f32; 3 * vertices];
                    dx.read(device, 0, &mut step_host)?;
                    let mass_host: Vec<f32> = prop_vertex.host()[..vertices]
                        .iter()
                        .map(|prop| prop.mass)
                        .collect();
                    projector.check_tangent(
                        device,
                        locked_records,
                        dx.handle(),
                        eval_x.host(),
                        &step_host,
                        translation_lock_index.host(),
                        &mass_host,
                        "constrained Newton correction",
                    )?;
                }
            }
            solved
        } else if locked_groups > 0 {
            let SolverState {
                lock,
                pcg_locked,
                translation_lock,
                translation_lock_index,
                translation_lock_initial,
                prop_vertex,
                dof_mask,
                eval_x,
                schwarz,
                ..
            } = state;
            let rows = super::lock::RowInputs {
                lock_index: translation_lock_index.handle(),
                locks: translation_lock.handle(),
                prop: prop_vertex.handle(),
                positions: eval_x.handle(),
                dof_mask: dof_mask.handle(),
                initial: translation_lock_initial.handle(),
            };
            let mut projector = super::lock::Projector {
                scratch: lock,
                groups: locked_groups,
                vertices,
                rows,
            };
            // `prepare` seeds from the caller's initial guess, which is `dx`:
            // the pinned rows already carry their exact prescribed correction.
            projector.prepare(
                device,
                locked_records,
                dx.handle(),
                pcg_locked.q.handle(),
            )?;
            // THE PRECONDITIONER THE SCENE ASKED FOR, ON THIS PATH TOO. The
            // hierarchy is installed before the locked branch is taken, so a
            // locked scene that set `precond = schwarz` gets Schwarz rather
            // than block-Jacobi under a Schwarz label.
            let sweep = if (*param).precond == crate::data::PrecondMode::Schwarz {
                install_schwarz(device, schwarz, &operator, vertices, param)?;
                Some(&mut *schwarz)
            } else {
                None
            };
            super::dump_linsys::maybe_dump(device, &operator, force.handle(), eval_x.handle(), vertices)?;
            let solved = pcg::solve_locked(
                device,
                &operator,
                &mut projector,
                inverse.span(0, 9 * vertices),
                force.handle(),
                dx.handle(),
                work,
                pcg_locked,
                sweep,
                prm.cg_tol,
                prm.cg_max_iter,
            )?;
            // ON SUCCESS ONLY: a failed solve has no correction worth
            // verifying, and the check is read-only either way. It aborts on a
            // violation rather than snapping, because a snap after the CCD line
            // search would reintroduce penetration.
            //
            // SUCCESS IS CONVERGENCE OR STEIHAUG TRUNCATION, AND AN
            // ITERATION-CAP EXHAUSTION IS NEITHER. The locked solve reports
            // `Converged` when the residual falls, `CurvatureTruncated` when an
            // unresolvable curvature stops it, and a failure at the cap or on a
            // non-finite residual. A gate spelled as "not a breakdown" would let
            // the cap through instead, and that costs twice: the whole pose and
            // the whole search direction would cross the seam to verify a
            // direction the step abandons at `### cg failed` a few lines below,
            // and the verdict would be taken on an UNCONVERGED correction, so a
            // violation it reported would raise a Fatal where an ordinary failed
            // step is the right answer.
            if matches!(
                solved.outcome,
                pcg::Outcome::Converged | pcg::Outcome::CurvatureTruncated
            ) {
                eval_x.download(device)?;
                // `dx` IS A PLAIN DEVICE BUFFER, so the check reads it back
                // explicitly. The host re-verification below is the only thing
                // that wants it, and it runs once per solve rather than per
                // iteration.
                let mut step_host = vec![0.0f32; 3 * vertices];
                dx.read(device, 0, &mut step_host)?;
                let mass_host: Vec<f32> =
                    prop_vertex.host()[..vertices].iter().map(|p| p.mass).collect();
                // THE TWO MIRRORS ARE LENT, NOT COPIED. `host()` already answers
                // out of a host array the buffer owns for the whole run, so a
                // `to_vec()` on either argument would be a second copy of
                // `vertices` position triples and `vertices` indices per Newton
                // iteration, held only for the duration of one call.
                projector.check_tangent(
                    device,
                    locked_records,
                    dx.handle(),
                    eval_x.host(),
                    &step_host,
                    translation_lock_index.host(),
                    &mass_host,
                    "constrained Newton correction",
                )?;
            }
            solved
        } else {
            // THE PRECONDITIONER THE SCENE ASKED FOR. Schwarz is REBUILT when
            // the row count changes and reused otherwise: the factorization is
            // quadratic in the dense-block cap and the partition is a
            // sequential host sweep, so paying either per Newton step would
            // cost more than the preconditioner saves.
            let sweep = if (*param).precond == crate::data::PrecondMode::Schwarz {
                install_schwarz(device, &mut state.schwarz, &operator, vertices, param)?;
                Some(&mut state.schwarz)
            } else {
                None
            };
            super::dump_linsys::maybe_dump(device, &operator, force.handle(), eval_x.handle(), vertices)?;
            pcg::solve(
                device,
                &operator,
                inverse.span(0, 9 * vertices),
                force.handle(),
                dx.handle(),
                work,
                sweep,
                prm.cg_tol,
                prm.cg_max_iter,
            )?
        };
        drop(operator);
        // Give the value array back so the next iteration reuses it.
        state.fixed_values = fixed.into_values();

        // Name: Linear Solve Time
        // Format: list[(time, ms)]
        // Map: pcg_linsolve
        // Description:
        // Wall-clock time in milliseconds spent in the preconditioned
        // conjugate gradient (PCG) linear solve for the Newton step
        // direction. One entry per Newton iteration. Typically the
        // dominant per-iteration cost.
        //
        // RECORDED BEFORE THE VERDICT IS READ, because the scope closes before
        // the outcome is tested: a solve that ran to the iteration
        // cap still spent the time it spent, and dropping the row there would
        // leave the stream shortest exactly where it is most worth reading. A
        // `?` inside a branch above escapes without this row, and that path is
        // a `Fatal` that ends the run rather than a failed step.
        log::mark("advance", "linsolve", _linsolve.elapsed());

        // STEP B17a. The solve's verdict.
        // Name: Linear Solve Iteration Count
        // Format: list[(time, iterations)]
        // Map: pcg_iter
        // Description:
        // Number of preconditioned conjugate gradient (PCG) iterations
        // consumed during the linear solve for this Newton iteration.
        // High values indicate an ill-conditioned system or a tight
        // tolerance and often correlate with long linear-solve times.
        log::mark("advance", "iter", report.iterations as f64);
        // Name: Linear Solve Relative Residual
        // Format: list[(time, ratio)]
        // Map: pcg_resid
        // Description:
        // Final relative residual reached by the PCG linear solve for this
        // Newton iteration. When this stays well below the configured
        // tolerance, the solve converged cleanly, values close to the
        // tolerance indicate the iteration cap was hit. In a scene with rigid
        // (PDRD) bodies the reduced solve measures each degree-of-freedom group
        // against its own initial residual, and this reports the worst group,
        // so a body and the cloth cannot mask each other.
        log::mark("advance", "reresid", report.relative_residual as f64);
        // Name: Schwarz Block-Jacobi Fallback
        // Format: list[(time, count)]
        // Description:
        // 1 if the solver fell back from the Schwarz preconditioner to the
        // SPD-safe block-Jacobi base for this Newton iteration's PCG solve, else
        // 0. Always 0 under the block-jacobi preconditioner; a nonzero entry is
        // worth reviewing. Recorded every iteration but only printed when
        // nonzero so the common 0 case does not clutter the log.
        //
        // A STRUCTURAL ZERO HERE, NOT AN UNMEASURED ONE, which is why it is
        // `quiet` rather than a number whose print is decided per iteration.
        // Two things could raise it and this driver has neither: there is no
        // memory guard that degrades Schwarz to block-Jacobi, and `schwarz.rs`
        // builds its aggregate term so the `rz <= 0` breakdown cannot arise.
        // The channel is recorded so a reader asking for it by name gets an
        // answer, and the day a degrade path is added this becomes a `Number`
        // and starts printing.
        log::mark("advance", "schwarz_fallback", log::Marked::quiet(0.0));
        match report.outcome {
            // BOTH ARE SUCCESSES, and the truncation reports itself at the
            // site where it happens (`pcg.rs`). Nothing is added here.
            pcg::Outcome::Converged | pcg::Outcome::CurvatureTruncated => {}
            pcg::Outcome::MaxIterations => {
                // A FAILURE: the cap is reported as `### cg failed` with
                // `pcg_success = false` and an abandoned step.
                //
                // THE ITERATE IS STILL RETURNED, AND THAT IS NOT PERMISSION TO
                // CONTINUE. A direction that never met `cg-tol` is not a Newton
                // direction, and taking it produces a run that exits 0 with
                // every frame written and no statement anywhere that the linear
                // solves never converged.
                //
                // `CurvatureTruncated` above stays a SUCCESS on the opposite
                // grounds: Steihaug truncation on an unresolvable curvature
                // stops at the best direction fp32 can see, rather than at an
                // unconverged one.
                //
                // THE LINE BELONGS TO THE SOLVER'S OWN TRANSCRIPT, so it is
                // bare. The iteration count and residual are this driver's
                // addition and the exponent is rebuilt to C's `%.3e` for the
                // same reason every other mark is.
                log::message!(
                    "### cg failed (iteration cap, {} iterations, relative residual {})",
                    report.iterations,
                    log::c_exponential(f64::from(report.relative_residual), 3)
                );
                result.pcg_success = false;
                return Ok(result);
            }
            // A BREAKDOWN IS FATAL, NOT A FAILED STEP, and the distinction is
            // the whole of the `pAp<=0` triage. The Newton
            // matrix and the block-Jacobi preconditioner are SPD BY
            // CONSTRUCTION, every per-element Hessian block that can be
            // indefinite being PSD-projected at the source, so a sign that
            // survives the round-off bound is a real regression upstream: a
            // lost projection, a sign or assembly error, a dropped CSR block.
            // Reporting it as an ordinary unconverged step collapses it onto
            // the same `CrashKind::Cg` an iteration-cap exhaustion produces,
            // and the natural response to THAT is to raise `cg-max-iter` or
            // loosen `cg-tol`, which buries the defect instead of surfacing it.
            // Every one of these sites aborts the run.
            pcg::Outcome::IndefiniteMatrix => {
                return Err(Fatal::invariant(format!(
                    "PCG breakdown: p^T A p is negative beyond the round-off \
                     bound of its own sum at iteration {}, relative residual \
                     {:.3e}. The assembled Newton Hessian is not SPD. Beyond \
                     that bound the sign is real, so this is a per-element \
                     Hessian missing its PSD projection, a sign or assembly \
                     error, or a dropped off-diagonal block. It is NOT a \
                     tolerance to loosen: a genuine defect puts the Rayleigh \
                     quotient at order one, six orders above the bound.",
                    report.iterations, report.relative_residual
                )));
            }
            pcg::Outcome::NonSpdPreconditioner => {
                return Err(Fatal::invariant(format!(
                    "PCG breakdown: r^T M^-1 r is non-positive at iteration \
                     {}, relative residual {:.3e}. The preconditioner is not \
                     SPD. Every block-Jacobi diagonal block is inverted \
                     through a floored symmetric eigendecomposition, so each \
                     per-vertex term is positive by construction and their sum \
                     cannot be negative in exact arithmetic: a non-positive \
                     value here means a block is NaN or infinite, or the \
                     residual itself is not finite.",
                    report.iterations, report.relative_residual
                )));
            }
        }

        // STEP B18a. THE SPIN RECOVERED from the solved translation increment,
        // by Schur back-substitution, immediately after the solve that produced
        // the increment. It uses the RAW solve direction: the line search's toi
        // rescale is not applied to the recovered spin, which is a small-toi
        // approximation stated here rather than corrected.
        if grains > 0 {
            // THE STATE'S OWN BLOCKS, present whenever the scene has grains.
            // Asking the collider and substituting `Handle::NONE` when there is
            // none left this pass with an unresolvable buffer in exactly the
            // scenes that carry grains and no floor or sphere.
            let (angular, coupling, rotational) = (
                state.grain_angular.handle(),
                state.grain_coupling.handle(),
                state.grain_rotational.handle(),
            );
            let recover = SandGrainRecoverRowArgs {
                inverse_center_inertia: state.grain_inv_inertia_center.handle(),
                angular,
                coupling,
                rotational_gradient: rotational,
                previous_omega: state.grain_omega_prev.handle(),
                increment: dx.handle(),
                omega: state.grain_omega.handle(),
                dt,
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("sand.recover", &recover, vertices as u32)?;
        }

        let _nt_mid = std::time::Instant::now();
        // STEP B19. The search direction's largest per-vertex magnitude, and
        // the rescale that keeps any one vertex inside `max_dx`.
        let magnitude_args = DxMagnitudeArgs {
            direction: state.dx.handle(),
            magnitude: state.scalar.handle(),
            count: vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("step.dx_magnitude", &magnitude_args, vertices as u32)?;
        // AND THIS ONE FOLDS ON THE DEVICE TOO; see the start-of-step pair.
        let max_dx = {
            let a = state.scalar.handle();
            // Safety: the buffer outlives the call and names `vertices` floats.
            unsafe { state.fold.max(device, "step.max_dx", a, vertices as u32, 0.0) }?
        };
        // Name: Max Search Direction Magnitude
        // Format: list[(time, meters)]
        // Map: max_search_dir
        // Description:
        // Maximum per-vertex magnitude (L2 norm) of the Newton search
        // direction returned by the linear solve for this Newton
        // iteration, in meters. Compared against the max_dx parameter to
        // decide whether the search direction must be rescaled before
        // the line search.
        log::mark("advance", "max_dx", max_dx as f64);
        let toi_recale = (prm.max_dx / max_dx).min(1.0);
        // Name: Search Direction Rescale Factor
        // Format: list[(time, ratio)]
        // Description:
        // Scalar in (0, 1] applied to the Newton search direction before
        // the line search, so that no per-vertex displacement exceeds the
        // configured max_dx. A value of 1.0 means the direction was
        // already within budget, smaller values clamp an over-eager step.
        log::mark("advance", "toi_recale", toi_recale as f64);

        // STEP B20. `target` is reused to hold the iterate's pre-step
        // positions, then the rescaled step is applied to the iterate.
        state.target.copy_from(device, &state.eval_x)?;
        let step_args = PositionStepArgs {
            eval_x: state.eval_x.handle(),
            direction: state.dx.handle(),
            scale: toi_recale,
            count: vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("step.position_step", &step_args, vertices as u32)?;

        // STEP B21 and B22. The `fix-xz` position drag, the half that moves the
        // accepted step. Its momentum half is assembled above, and the two ship
        // together: the momentum term is what the linear system is built from
        // and this clamp is what the step does, so a backend carrying one alone
        // would solve a system that does not describe the step it then takes.
        //
        // IT DRAGS TOWARD `vertex.prev`, THE PREVIOUS STEP, not toward the
        // pre-step iterate `target` now holds. Those differ once the Newton
        // loop has taken more than one step, and the previous step is the one
        // the clamp is defined against.
        if prm.fix_xz != 0.0 {
            let drag_args = FixXzDragArgs {
                eval_x: state.eval_x.handle(),
                previous: prev,
                dof_removed: state.dof_mask.handle(),
                threshold: prm.fix_xz,
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("step.fix_xz_drag", &drag_args, vertices as u32)?;
        }

        // RECORDED AND NOT PRINTED: the rescale and the position step are not
        // one of the solver's own printed scopes.
        log::mark(
            "advance",
            "nt_mid",
            log::Marked::quiet(_nt_mid.elapsed().as_secs_f64() * 1000.0),
        );
        // STEP B28. The AABB refresh the line search queries against. The
        // leaves are re-bounded over the WHOLE candidate step, from the
        // pre-step positions `target` now holds to the proposed iterate, so a
        // pair that ends the step apart but crossed inside it is still a
        // candidate.
        //
        // The refresh comes BEFORE the sweep and after the step was proposed:
        // bounding the old pose alone would miss every crossing the step
        // introduces.
        if !prm.disable_contact {
            let _aabb = std::time::Instant::now();
            let SolverState {
                contact: contact_state,
                target,
                eval_x,
                ..
            } = &mut *state;
            let (start, finish) = (target.handle(), eval_x.handle());
            if let Some(contact) = contact_state.as_mut() {
                contact.refresh_leaves(device, data, mesh_refs, start, finish, prm.line_search_max_t, windows)?;
            }
            // THIS SCOPE CLOSES BEFORE THE LINE SEARCH OPENS, which is what
            // puts `> aabb_update` above `> line_search` in the stream. It
            // carries no docstring, so the channel is absent from `names()` and
            // only the printed line exists.
            log::mark("advance", "aabb_update", _aabb.elapsed());
        }

        // STEP B29. The line search. The ACCD filter is one of the two things
        // that make this solver penetration-free: it returns the largest
        // fraction of the proposed step that crosses nothing, and no step past
        // it is ever taken. With `disable-contact` set there are no pairs to
        // sweep and the filter returns its whole span, which the ratio below
        // turns into 1.
        // THE LINE SEARCH, timed as `line_search`.
        let _line_search = std::time::Instant::now();
        let mut toi = {
            let filter = if prm.disable_contact {
                None
            } else {
                let SolverState {
                    contact: contact_state,
                    target,
                    eval_x,
                    ..
                } = &mut *state;
                let (start, finish) = (target.handle(), eval_x.handle());
                // NO POSE CROSSES TO THE HOST HERE. The sweep is six device
                // dispatches over the two handles above, each walking a BVH
                // with the ACCD advance as a per-hit functor. What comes back
                // is two per-primitive float arrays reduced to one number
                // each.
                match contact_state.as_mut() {
                    Some(contact) => Some(contact.line_search(
                        device,
                        data,
                        mesh_refs,
                        &*param,
                        start,
                        finish,
                        windows,
                    )?),
                    None => {
                        return Err(Fatal::invariant(
                            "solver driver: the line search was reached with no contact \
                             subsystem allocated, so nothing would bound the step",
                        ))
                    }
                }
            };
            // A PAIR THAT BEGAN THE STEP INSIDE ITS OFFSET RETURNS EXACTLY
            // ZERO, and reading that as "no progress, try a shorter step" spins
            // forever on a state the conservative advance cannot resolve. It is
            // asked about BEFORE the time is used, and the step ends naming the
            // pair.
            // THE ANALYTIC SWEEP, which runs OUTSIDE the `disable-contact`
            // gate: a prescribed vertex driven through a floor tunnels it
            // whether or not mesh contact is live, and no barrier can stop
            // it.
            let sweep = {
                let SolverState {
                    analytic: analytic_state,
                    target,
                    eval_x,
                    fix,
                    sphere,
                    floor,
                    ..
                } = &mut *state;
                let (start, finish) = (target.handle(), eval_x.handle());
                let Some(analytic) = analytic_state.as_mut() else {
                    return Err(Fatal::invariant(
                        "solver driver: the analytic collider sweep was reached with no \
                         collider layer allocated, so nothing would bound a \
                         vertex against a sphere or a floor",
                    ));
                };
                analytic.line_search(device, data, mesh_refs, param, start, finish, fix.handle(), sphere, floor)?
            };
            // A FIX-PINNED VERTEX DRIVEN THROUGH A COLLIDER IS REPORTED, NOT
            // CLAMPED. It has no freedom to yield, so a clamped time of impact
            // would stall the whole solve without preventing the crossing. The
            // step fails naming the vertex.
            if let Some(bad) = sweep.infeasible_pin {
                // THE WORDING IS A HOST CONTRACT, character for character. A
                // Rust string literal keeps the indentation of its continuation
                // lines, so the naive spelling carries a run of spaces
                // mid-sentence and matches nothing grepping for the message.
                // Concatenated adjacent literals are what avoid that.
                log::message!(
                    "### infeasible pin: prescribed vertex {bad} is driven \
                     through an analytic collider (floor/sphere) it cannot \
                     yield to"
                );
                log::message!(
                    "### re-author the pin's path so it stays outside the \
                     collider, or make it a soft pull pin so it can yield."
                );
                // THE ROW IS RECORDED AFTER THE REPORT: both messages, then
                // the scope's own row. The row has to be written at all on an
                // early return, because an unbalanced scope is an error in the
                // log, but it belongs after the two lines that explain the
                // failure and not before them.
                log::mark("advance", "line_search", _line_search.elapsed());
                result.pin_feasible = false;
                return Ok(result);
            }
            if let Some(overlap) = filter.as_ref().and_then(super::ccd::Filter::overlapping_start)
            {
                log::message!("### contact starts overlapping: {}", overlap.describe());
                log::message!(
                    "### give the initial geometry a small clearance so nothing starts in \
                     contact, or check whether a pin is pulling elements together faster than \
                     contact can resolve."
                );
                // Recorded after the report, for the reason above.
                log::mark("advance", "line_search", _line_search.elapsed());
                result.contact_separated = false;
                return Ok(result);
            }
            // THE CONTACT FILTER'S OWN FRACTION, recorded before the analytic
            // sweep and the strain limiters narrow it. It is a different
            // question from the accepted `toi` below: this one says whether
            // MESH CONTACT bit, which is the only reading that distinguishes a
            // step contact bounded from one a limiter did.
            //
            // RECORDED AND NOT PRINTED. The value keeps its stream file, which
            // `examples/metal_pdrd_*_fixture.py` read, and stays out of the
            // printed transcript.
            log::mark(
                "advance",
                "contact_toi",
                log::Marked::quiet(f64::from(
                    filter
                        .as_ref()
                        .map_or(prm.line_search_max_t, super::ccd::Filter::time_of_impact)
                        / prm.line_search_max_t,
                )),
            );
            let smallest = filter
                .as_ref()
                .map_or(prm.line_search_max_t, super::ccd::Filter::time_of_impact)
                .min(sweep.time_of_impact);
            (smallest / prm.line_search_max_t).min(1.0)
        };
        // THE STRAIN LIMITER'S OWN HALF, folded in after the contact CCD.
        // This is the half that makes the limit
        // a limit: the barrier above raises the cost of stretching and this
        // REFUSES the fraction of the step that would cross it, so an authored
        // limit bounds the accepted pose rather than merely discouraging it.
        //
        // `SL_toi` OPENS AT 1.0 AND IS THE MINIMUM OVER BOTH HALVES, which is
        // what the failure message below reports; `toi` takes each half
        // separately. Both reductions fold the per-element
        // array the walk wrote, so the answer does not depend on how the range
        // was cut.
        let mut sl_toi = 1.0f32;
        if state.sizes.shell_faces > 0 {
            assemble::shell_strain_toi(device, data, state, prm.line_search_max_t)?;
            let faces = state.sizes.shell_faces;
            let toi_array = state.face_strain.toi.handle();
            // Safety: the buffer outlives the call and names `faces` floats.
            let smallest = unsafe {
                state.fold.min(
                    device,
                    "step.shell_strain_toi",
                    toi_array,
                    faces as u32,
                    prm.line_search_max_t,
                )
            }?;
            let shell_toi = smallest / prm.line_search_max_t;
            sl_toi = sl_toi.min(shell_toi);
            toi = toi.min(shell_toi);
            // Name: Strain-Limit Time of Impact
            // Format: list[(time, ratio)]
            // Description:
            // Fraction in (0, 1] of the rescaled search direction that can
            // be taken without violating the configured shell or rod
            // strain limits, as returned by the strain-limiting line
            // search. A value of 1.0 means strain limits never bound the
            // step, smaller values mean the strain limiter clamped it.
            log::mark("advance", "SL_toi", shell_toi as f64);
        }
        if state.sizes.rods > 0 {
            assemble::rod_strain_toi(device, data, state, prm.line_search_max_t)?;
            let rods = state.sizes.rods;
            let toi_array = state.rod_strain.toi.handle();
            // Safety: the buffer outlives the call and names `rods` floats.
            let smallest = unsafe {
                state.fold.min(
                    device,
                    "step.rod_strain_toi",
                    toi_array,
                    rods as u32,
                    prm.line_search_max_t,
                )
            }?;
            let rod_toi = smallest / prm.line_search_max_t;
            sl_toi = sl_toi.min(rod_toi);
            toi = toi.min(rod_toi);
            // Name: SL-rod-toi
            // Format: list[(time, ratio)]
            // Map: SL-rod-toi
            // Description:
            // Fraction in (0, 1] of the rescaled search direction the ROD
            // strain limiter allows, reported separately from the shell
            // limiter's because a rod-only scene never marks that one. The
            // docstring above is what makes the channel askable by name: a
            // value that only reaches stdout ties every caller to the exact
            // phrasing of a printed line.
            log::mark("advance", "SL_rod_toi", rod_toi as f64);
        }
        // Name: Line Search Time
        // Format: list[(time, ms)]
        // Description:
        // Wall-clock time in milliseconds spent in the per-iteration
        // line search, which runs continuous collision detection (CCD)
        // plus strain-limit CCD to find the largest feasible substep
        // along the rescaled search direction. One entry per Newton
        // iteration.
        //
        // CLOSED AFTER BOTH STRAIN LIMITERS, NOT BEFORE THEM. The scope opens
        // above the contact sweep and closes after `SL_toi` and `SL_rod_toi`
        // are marked, so the limiters are inside the measured span and their
        // two lines print ABOVE `> line_search`. Recording it where the contact
        // sweep ends would report a shorter time under the `line_search` name
        // and put the lines in the wrong order.
        log::mark("advance", "line_search", _line_search.elapsed());
        let toi = toi;
        // Name: Line Search Time of Impact
        // Format: list[(time, ratio)]
        // Description:
        // Fraction in (0, 1] of the rescaled Newton search direction that
        // can be taken without causing a collision or violating strain
        // limits, as the minimum of the contact CCD result and the
        // strain-limit TOI. A value of 1.0 means the full Newton step was
        // accepted, smaller values mean the line search cut it short.
        log::mark("advance", "toi", toi as f64);
        last_toi = toi;
        if toi <= f32::EPSILON {
            // BOTH LINES BELONG TO THE SOLVER'S OWN TRANSCRIPT, both at `%.2e`.
            log::message!("### ccd failed (toi: {})", log::c_exponential(f64::from(toi), 2));
            if sl_toi < 1.0 {
                log::message!(
                    "strain limiting toi: {}",
                    log::c_exponential(f64::from(sl_toi), 2)
                );
            }
            result.ccd_success = false;
            return Ok(result);
        }

        let _nt_tail = std::time::Instant::now();
        // STEP B30. How far through its span the step has advanced. Kept in
        // double, because it accumulates across every Newton iteration.
        if !final_step {
            toi_advanced += (1.0 - toi_advanced).max(0.0) * f64::from(toi_recale * toi);
        }
        // PRINTED AT EVERY NEWTON ITERATION, so it is a bare transcript line
        // rather than a debug one. It
        // reads like a mark and is not: the recorded channel is written once
        // per step at STEP C1, and this is the running total the reader watches
        // climb toward `target_toi`.
        log::message!("* toi_advanced: {}", log::c_exponential(toi_advanced, 2));

        // STEP B31. Accept the line search's fraction, from the pre-step
        // positions `target` is holding.
        let accept_args = PositionAcceptArgs {
            origin: state.target.handle(),
            proposed: state.eval_x.handle(),
            fraction: toi,
            count: vertices as u32,
            seam_arena_count: 0,
        };
        device.launch("step.position_accept", &accept_args, vertices as u32)?;

        // STEP B32. THE PDRD RIGIDIFY COMMIT, and it is the reason a body
        // stays a body across frames rather than only within a step.
        //
        // The reduced solve moves each body rigidly, but the iterate it lands
        // on has also been through contact, the strain limiters and the line
        // search, all of which act PER VERTEX and none of which knows the body
        // is rigid. So the accepted pose is very slightly sheared, and refitting
        // to it next frame would fold that shear into the fit. Across a long run
        // it accumulates as a non-rigid shrink.
        //
        // The fix is to rebuild each body from a rotation that is ANCHORED
        // rather than refitted: `running_rotation` carries the rotation the
        // solve has actually applied, composed across every iteration and every
        // frame, so the rigid image owes nothing to the sheared iterate but its
        // CENTROID.
        //
        // AND THE COMMIT IS CCD-FILTERED LIKE ANY OTHER STEP. Snapping to the
        // rigid image would move vertices, and a move that crosses geometry
        // penetrates whatever it crossed no matter how principled the pose it
        // moves toward. So the same sweep that bounds the Newton step bounds
        // this one, and the same two failures end the step naming what they hit.
        if pdrd_bodies > 0 {
            {
                let SolverState {
                    rigid,
                    rigid_staged,
                    rigid_running_rotation,
                    rigid_rotation_step,
                    rigid_centroid,
                    eval_x,
                    rigid_target,
                    prop_vertex,
                    pcg_rigid,
                    ..
                } = &mut *state;
                let body_vertices = data.pdrd_vert_list.size;
                let vert_list = rigid_staged.vert_list.handle();
                let body_prop = rigid_staged.body_prop.handle();
                let rest_centered = rigid_staged.rest_centered.handle();
                let mut reduction = super::pdrd::Reduction {
                    map: rigid,
                    staged: rigid_staged,
                };
                reduction.compose_running_rotation(
                    device,
                    rigid_running_rotation.handle(),
                    rigid_rotation_step.handle(),
                    -(toi_recale * toi),
                )?;
                // The target starts as the iterate, so a NON-PDRD vertex is
                // carried through the commit untouched and its lerp below is a
                // no-op. Only the bodies' own vertices are overwritten.
                rigid_target.copy_from(device, eval_x)?;
                reduction.rigidify(
                    device,
                    vert_list,
                    prop_vertex.handle(),
                    body_prop,
                    rest_centered,
                    eval_x.handle(),
                    rigid_running_rotation.handle(),
                    rigid_centroid.handle(),
                    rigid_target.handle(),
                    body_vertices,
                    &pcg_rigid.zero,
                )?;
            }

            // The sweep, from the iterate to the rigid image.
            let toi_rig = {
                // THE `rigidify_ccd` SCOPE OPENS HERE and closes after the
                // sweep and its readback and BEFORE the infeasibility report,
                // so the span covers the AABB refresh and both halves of the
                // sweep. The mesh and analytic halves are separate calls here,
                // so the mark below closes after the second.
                let _rigidify_ccd = std::time::Instant::now();
                if !prm.disable_contact {
                    let SolverState {
                        contact: contact_state,
                        eval_x,
                        rigid_target,
                        ..
                    } = &mut *state;
                    let (start, finish) = (eval_x.handle(), rigid_target.handle());
                    if let Some(contact) = contact_state.as_mut() {
                        contact.refresh_leaves(
                            device, data, mesh_refs, start, finish,
                            prm.line_search_max_t, windows,
                        )?;
                    }
                }
                let filter = if prm.disable_contact {
                    None
                } else {
                    let SolverState {
                        contact: contact_state,
                        eval_x,
                        rigid_target,
                        ..
                    } = &mut *state;
                    let (start, finish) = (eval_x.handle(), rigid_target.handle());
                    // THE SAME DEVICE NARROW PHASE AS B29, over the two handles
                    // above: six dispatches, each walking a BVH with the ACCD
                    // advance as a per-hit functor, and no pose on the host.
                    match contact_state.as_mut() {
                        Some(contact) => Some(contact.line_search(
                            device, data, mesh_refs, &*param, start, finish,
                            windows,
                        )?),
                        None => {
                            return Err(Fatal::invariant(
                                "solver driver: the rigidify commit was reached with no contact \
                                 subsystem allocated, so nothing would bound it",
                            ))
                        }
                    }
                };
                // THE ANALYTIC SWEEP IS INSIDE THE GATE HERE, AND THAT IS NOT
                // WHAT B29 DOES. The two sites gate at different levels and the
                // difference is easy to read backwards.
                //
                // At B29 only the MESH sweep is behind `disable-contact`: the
                // analytic and pin sweep runs regardless, because a prescribed
                // vertex driven through a floor tunnels it whether or not mesh
                // contact is live.
                //
                // At the RIGIDIFY COMMIT the whole sweep is behind the gate,
                // the AABB refresh included, so with contact disabled neither
                // the mesh nor the analytic half runs. Hoisting the analytic
                // half out of the gate here, on the grounds that B29 does so,
                // reads the inner gate as if it were the outer one.
                let sweep = if prm.disable_contact {
                    super::collider::Sweep::unobstructed(prm.line_search_max_t)
                } else {
                    let SolverState {
                        analytic: analytic_state,
                        eval_x,
                        rigid_target,
                        fix,
                        sphere,
                        floor,
                        ..
                    } = &mut *state;
                    let (start, finish) = (eval_x.handle(), rigid_target.handle());
                    let Some(analytic) = analytic_state.as_mut() else {
                        return Err(Fatal::invariant(
                            "solver driver: the rigidify commit's analytic collider sweep was \
                             reached with no collider layer allocated",
                        ));
                    };
                    analytic.line_search(
                        device, data, mesh_refs, param, start, finish,
                        fix.handle(), sphere, floor,
                    )?
                };
                // GATED, because the scope covers work that does not happen
                // with contact off: neither half of the sweep runs, so a row
                // here would time nothing.
                if !prm.disable_contact {
                    log::mark("advance", "rigidify_ccd", _rigidify_ccd.elapsed());
                }
                if let Some(bad) = sweep.infeasible_pin {
                    log::message!(
                        "### infeasible pin: prescribed vertex {bad} is driven through an \
                         analytic collider during the rigidify commit"
                    );
                    result.pin_feasible = false;
                    return Ok(result);
                }
                if let Some(overlap) =
                    filter.as_ref().and_then(super::ccd::Filter::overlapping_start)
                {
                    log::message!(
                        "### contact starts overlapping during the rigidify commit: {}",
                        overlap.describe()
                    );
                    result.contact_separated = false;
                    return Ok(result);
                }
                let smallest = filter
                    .as_ref()
                    .map_or(prm.line_search_max_t, super::ccd::Filter::time_of_impact)
                    .min(sweep.time_of_impact);
                (smallest / prm.line_search_max_t).min(1.0)
            };
            log::mark("advance", "rigidify_toi", f64::from(toi_rig));

            // The lerp toward the rigid image. `position_accept` writes into
            // its `proposed` argument, so the accepted pose lands in the target
            // and is copied back: the arithmetic is the same interpolation
            // every other accept uses, so the commit advances along its segment
            // by the same rule as every other accepted step.
            let accept = PositionAcceptArgs {
                origin: state.eval_x.handle(),
                proposed: state.rigid_target.handle(),
                fraction: toi_rig,
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("step.rigidify_accept", &accept, vertices as u32)?;
            let SolverState { eval_x, rigid_target, .. } = &mut *state;
            eval_x.copy_from(device, rigid_target)?;
        }

        // RECORDED AND NOT PRINTED: the accept and the rigidify commit are not
        // one of the solver's own printed scopes.
        log::mark(
            "advance",
            "nt_tail",
            log::Marked::quiet(_nt_tail.elapsed().as_secs_f64() * 1000.0),
        );
        // STEP B33. Loop control. A continuing iteration REBUILDS the target,
        // which B20 clobbered; without it the next iteration would integrate
        // against the previous iterate's positions.
        if !result.success() || final_step {
            break;
        }
        if toi_advanced >= f64::from(prm.target_toi) && step >= prm.min_newton_steps {
            final_step = true;
        } else {
            step += 1;
            compute_target(
                device,
                &prm,
                curr,
                prev,
                vertices,
                state.target.handle(),
                state.fix.handle(),
                dt,
                state.fix_index.handle(),
            )?;
        }
    }

    if result.success() {
        // STEP C1 and C2. The final AABB refresh and `check_intersection`: the
        // SECOND of the two things that make this solver penetration-free. The
        // ACCD filter refuses a step that would cross; this reports a crossing
        // that got through anyway, at the pose the step is about to commit.
        //
        // Reporting is GATED ON "DYNAMIC" and stays gated: an intersection
        // between two fully prescribed elements cannot be resolved by either
        // side yielding, so reporting it would only abort a run over geometry
        // the solver was never going to fix. The three intersection allowances
        // reach `intersection.rs`'s one predicate and suppress REPORTING only.
        if !prm.disable_contact {
            let SolverState {
                contact: contact_state,
                eval_x,
                prop_vertex,
                ..
            } = &mut *state;
            let pose = eval_x.handle();
            let Some(contact) = contact_state.as_mut() else {
                return Err(Fatal::invariant(
                    "solver driver: the intersection gate was reached with no contact subsystem \
                     allocated, so a penetrating pose would be committed unreported",
                ));
            };
            let _phase = super::phase::start("contact.refresh_leaves");
            let _aabb = std::time::Instant::now();
            contact.refresh_leaves(device, data, mesh_refs, pose, pose, 1.0, windows)?;
            drop(_phase);
            // THE SECOND SITE OF THE SAME CHANNEL. This scope closes before
            // the intersection gate opens, so `> aabb_update` prints above
            // `> check_intersection`. One channel, two sites per step.
            log::mark("advance", "aabb_update", _aabb.elapsed());
            // The scan walks the committed pose on the host, so it downloads
            // here. `pose` was taken above and staled the mirror, which is why
            // this is not optional.
            //
            // AND THE SCAN IS WHERE THE DIVERGENCE IS. The gate should run
            // three traversals per edge in ONE dispatch and move a 4-byte
            // counter, at most `capacity` records and two flag arrays, which is
            // the shape `contact/intersect_geometry.entry.cpp` already
            // declares. `intersection.rs` instead walks a materialized pair
            // list serially on the host, and that is a recorded divergence: a
            // pass whose work is proportional to the mesh belongs on the
            // device. The CCD half of the same row closed with
            // `contact/ccd_sweep.kernel.cpp`, which is the shape this one takes
            // next; nothing blocks it. This download is the transport that host
            // walk needs and comes out with it.
            // THE FINAL PENETRATION GATE, timed as `check_intersection`. The
            // download above is inside the scope because it is part of what the
            // gate costs today.
            let _gate = std::time::Instant::now();
            eval_x.download(device)?;
            let report =
                contact.check_intersection(device, data, mesh_refs, pose, eval_x.host(), prop_vertex.host(), windows)?;
            if report.is_clean() {
                // THE CLEAN PATH IS `debug!`, SO THE TRANSCRIPT ONLY SPEAKS
                // WHEN THE GATE FINDS SOMETHING. A line per committed pose puts
                // one sentence per step into a stream that otherwise carries
                // marks and timings, and on a real scene that is most of what a
                // reader sees go by.
                //
                // IT IS AT `debug!` RATHER THAN ABSENT, because the argument for
                // having it is still good: this is the final penetration gate,
                // a gate nothing can observe is not a gate, and a run that died
                // in the first line search would otherwise report zero
                // intersections while proving nothing.
                // `examples/metal_pdrd_cloth_contact_fixture.py` asserts the
                // gate RAN and sets `RUST_LOG=debug` to see it.
                ::log::debug!("check_intersection: the committed pose carries no \
                              intersecting pair");
            } else {
                // THE LINE BELONGS TO THE SOLVER'S OWN TRANSCRIPT, so it is
                // bare. The count is this driver's addition.
                log::message!(
                    "### intersection detected: {} intersecting pair(s) at the committed pose",
                    report.found
                );
                result.intersection_free = false;
            }
            // THE ROW IS RECORDED AFTER THE REPORT: `### intersection
            // detected` prints inside the scope and the timing line follows the
            // finding rather than preceding it. No docstring here, so the
            // channel is absent from `names()` and only the printed line
            // exists; the INITIALIZE-time scan is the one that carries a
            // docstring, under `initial_check_intersection`.
            log::mark("advance", "check_intersection", _gate.elapsed());
            super::publish_intersection_records(&report);
        }

        // Name: Advanced Fractional Step Size
        // Format: list[(time, ratio)]
        // Description:
        // Fraction in (0, 1] of the target step size that the Newton loop
        // actually advanced, accumulated across all its iterations. The
        // final Final Step Size equals this fraction times the target dt.
        // A value of 1.0 means the full target step completed, smaller
        // values mean contacts or strain limits forced a partial step.
        log::mark("advance", "toi_advanced", toi_advanced);
        // Name: Newton Iteration Count
        // Format: list[(time, iterations)]
        // Description:
        // Number of Newton iterations consumed in this simulation step
        // (before the trailing error-reduction iteration). Values above
        // the configured min_newton_steps indicate the solver needed
        // extra iterations to reach the target advanced step size.
        log::mark("advance", "newton_steps", step as f64);
        // Name: Final Step Size
        // Format: list[(time, seconds)]
        // Description:
        // Step size in seconds that was actually integrated this
        // simulation step. In easy cases this matches the target dt, but
        // it is reduced by the advanced TOI fraction when contacts or
        // strain limits shorten the step, and can also be reduced when
        // enable_retry is on and the PCG solve fails.
        log::mark("advance", "final_dt", dt as f64);

        let _commit = std::time::Instant::now();
        // STEP C3. Commit the clock. `prev_dt` is what the NEXT step's velocity
        // reconstruction divides by, and `backend.rs` reads it back into the
        // checkpoint, so both fields are written through to the host's record.
        (*param).prev_dt = dt;
        (*param).time += f64::from(dt / prm.playback);
        (*param).time_f32 = (*param).time as f32;

        // STEP C4. The translation-lock invariant check, on the pose about to
        // be committed.
        //
        // READ-ONLY BY DESIGN. It accumulates each group's mass-weighted
        // perpendicular center-of-mass drift and the largest per-axis
        // displacement in that group, then reaches a host verdict against a
        // round-off bound. It never snaps a position: a correction applied
        // after the CCD line search would move a vertex the line search had
        // already certified, which is how a penetration gets reintroduced. A
        // group that has drifted off its commanded line is reported instead.
        if locked_groups > 0 {
            let rows = super::lock::RowInputs {
                lock_index: state.translation_lock_index.handle(),
                locks: state.translation_lock.handle(),
                prop: state.prop_vertex.handle(),
                positions: state.eval_x.handle(),
                dof_mask: state.dof_mask.handle(),
                initial: state.translation_lock_initial.handle(),
            };
            let iterate = state.eval_x.handle();
            super::lock::check_invariant(
                device,
                rows,
                iterate,
                locked_records,
                &mut state.lock_drift,
                &mut state.lock_max_displacement,
                vertices,
                "completed Newton step",
            )?;
        }

        // STEP C5. Commit the positions: the current pose becomes the previous
        // one and the iterate becomes current.
        // ON THE DEVICE, in that order. `fetch()` is what puts these in the
        // arrays `backend.rs` reads.
        state.positions_prev.copy_from(device, &state.positions)?;
        state.positions.copy_from(device, &state.eval_x)?;

        // STEP C6. THE SAND SPIN INTEGRATE, on the pose just committed, which
        // is where the committed pose makes it meaningful. It consumes the CONVERGED friction
        // torque the contact fold summed over the final Newton iteration's
        // simultaneous contacts, and the angular stiffness beside it is what
        // makes omega approach the rolling rate rather than overshoot it.
        //
        // A GRAIN-GRAIN PAIR IS A STAGGERED, POST-SOLVE UPDATE, unlike the
        // analytic contacts, whose spin is condensed into the solve itself.
        //
        // THE TWO TRAILING ARGUMENTS DO NOT AGREE, THOUGH BOTH LOOK LIKE
        // TUNING. `c_roll` is a literal zero below and is the optional linear
        // resistance in the implicit denominator; `roll_resist` is the
        // anti-pump under-roll cap and defaults to 0.05. Passing zero for both
        // leaves friction able to do net positive work on every grain-grain
        // contact in every SAND scene.
        if grains > 0 && !sand_no_roll() {
            // THE BLOCKS ARE THE STATE'S, so this reads a real per-vertex array
            // whether or not the scene has an analytic collider. Asking the
            // collider and substituting `Handle::NONE` when there is none does
            // not work: a generated entry resolves the handle BEFORE the body
            // runs, and the body reads `angular[i]` unconditionally, so a grain
            // scene with no floor and no sphere would have nothing valid to
            // read.
            let angular = state.grain_angular.handle();
            // THE STATE'S OWN ACCUMULATORS, NOT THE CONTACT LAYER'S. Taking
            // them from that layer would couple this to something
            // `disable-contact` does not build, while the analytic colliders
            // are assembled outside it, so the grain would have spin to
            // integrate and nowhere to accumulate it.
            let torque = state.grain_torque.handle();
            let stiffness = state.grain_stiffness.handle();
            let normal = state.grain_normal.handle();
            let integrate = SandGrainIntegrateRowArgs {
                inverse_rolling_inertia: state.grain_inv_inertia.handle(),
                angular,
                prop: state.prop_vertex.handle(),
                params: state.param_vertex.handle(),
                curr: state.positions.handle(),
                prev: state.positions_prev.handle(),
                torque,
                normal_sum: normal,
                angular_stiffness: stiffness,
                omega: state.grain_omega.handle(),
                dt,
                c_roll: 0.0,
                roll_resist: sand_roll_resist(),
                count: vertices as u32,
                seam_arena_count: 0,
            };
            device.launch("sand.integrate", &integrate, vertices as u32)?;
            // THE NEXT STEP'S CONDENSE READS THIS STEP'S SPIN, so the snapshot
            // is taken here rather than at the top of the next step: `omega` is
            // the one piece of SAND state that carries across steps.
            state.grain_omega_prev.copy_from(device, &state.grain_omega)?;
        }

        // STEP C7 and C8. The four plastic creeps, on the pose just committed
        // and in a fixed order: shell faces, tets, hinges, then interior rod
        // vertices.
        //
        // THE RAW SUBSTEP, NOT THE LOCAL `dt`. The plasticity kernels take the
        // step size the scene configured, while every other phase of the step
        // takes the local `dt`, which by here has been scaled by `playback` and
        // shortened by the line search. The creep rate depends on which one it
        // gets, so `prm.dt` is passed rather than the local.
        super::plasticity::creep(device, data, state, prm.dt)?;

        result.time = (*param).time;
        // RECORDED AND NOT PRINTED: the commit is not one of the solver's own
        // printed scopes.
        log::mark(
            "advance",
            "commit",
            log::Marked::quiet(_commit.elapsed().as_secs_f64() * 1000.0),
        );
    }
    Ok(result)
}

/// Every scene class the driver does not implement, checked before it steps.
///
/// A SECOND CALL TO THE SAME GATE, DELIBERATELY, and it is one function rather
/// than two lists on purpose. `super::refusal::scene_refusals` runs once at
/// `initialize()` against the scene as built; this runs it again every step
/// against the scene as it now is, which is where a per-step upload can
/// introduce something the first call never saw: `make_constraint` is rebuilt
/// from the schedule every step, so a collider that switches on at t = 2 is
/// invisible to a gate that only ran at t = 0.
///
/// TWO LISTS WOULD BE THE DEFECT THIS GUARDS AGAINST. If the gate and the
/// driver could disagree, one of the two orders is a scene refused that the
/// driver could step (loud and merely annoying) and the other is a scene
/// stepped that the gate meant to refuse, which writes plausible frames with
/// nothing in the output to say so. Sharing the function makes the second
/// impossible rather than unlikely.
///
/// # Safety
/// `param` must address a live `ParamSet`.
unsafe fn assert_unsupported_absent(data: &DataSet, param: *const ParamSet) -> FatalResult<()> {
    let refusals = super::refusal::scene_refusals(data, &*param);
    if refusals.is_empty() {
        return Ok(());
    }
    let named: Vec<String> = refusals.iter().map(super::refusal::Refusal::describe).collect();
    Err(Fatal::invariant(format!(
        "solver driver: advance() reached a scene the capability gate refuses: {}. Either \
         initialize() accepted it, which would mean the driver steps a scene it cannot solve, \
         or the scene gained the feature partway through the run. The step stops either way",
        named.join("; ")
    )))
}
/// The implicit target every vertex is solved toward, for one step size.
///
/// A free function rather than a closure over the step's locals, because it now
/// takes the device and a closure holding a mutable borrow of it could not be
/// called from inside the Newton loop where the device is also used.
///
/// # Safety
/// `current` and `previous` must each address `3 * vertices` position
/// components that outlive the dispatch.
#[allow(clippy::too_many_arguments)]
unsafe fn compute_target<D: Device>(
    device: &mut D,
    prm: &StepParams,
    current: Handle,
    previous: Handle,
    vertices: usize,
    target: Handle,
    fix: Handle,
    dt: f32,
    fix_index: Handle,
) -> Result<(), Fault> {
    let args = ComputeTargetSeedArgs {
        current,
        previous,
        fix_index,
        fix,
        dt,
        previous_dt: prm.prev_dt,
        // Gravity rides the record as three scalars rather than as a sixth
        // buffer. It is one scene-wide vector every thread reads identically,
        // and it lives on this stack frame beside `dt`, so there is no
        // allocation for a handle to name and never will be.
        gravity_x: prm.gravity[0],
        gravity_y: prm.gravity[1],
        gravity_z: prm.gravity[2],
        inactive_momentum: i32::from(prm.inactive_momentum),
        target,
        count: vertices as u32,
        seam_arena_count: 0,
    };
    device.launch("step.compute_target", &args, vertices as u32)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_step_delay_defaults_to_zero_and_parses_milliseconds() {
        use std::time::Duration;
        // ZERO IS THE DEFAULT AND THE ABSENT CASE, which is what keeps the hook
        // off for every run that did not ask for it. The deleted stub defaulted
        // to 1000 ms and made seven scenarios observable by accident; this one
        // makes nobody slower until asked.
        assert_eq!(super::parse_step_delay(None), Duration::ZERO);
        assert_eq!(super::parse_step_delay(Some("0")), Duration::ZERO);
        assert_eq!(
            super::parse_step_delay(Some("200")),
            Duration::from_millis(200)
        );
        // Whitespace survives a shell that quotes awkwardly.
        assert_eq!(
            super::parse_step_delay(Some(" 600 ")),
            Duration::from_millis(600)
        );
        // Unparsable is zero, not a panic: see `parse_step_delay`.
        assert_eq!(super::parse_step_delay(Some("fast")), Duration::ZERO);
        assert_eq!(super::parse_step_delay(Some("-5")), Duration::ZERO);
        assert_eq!(super::parse_step_delay(Some("")), Duration::ZERO);
    }

    use super::*;
    use crate::driver::launch::host_device;
    use crate::driver::state::SolverState;
    use crate::driver::test_scene::TestScene;

    /// THE PIN INDEX GATHER.
    ///
    /// This module exists because nothing else covers the dispatch that fills
    /// `state.fix_index`: zeroing that kernel's only write leaves every one of
    /// the 454 tests outside this module passing. A per-element write that no
    /// assertion reads is the same gap the rod bending slot key slipped
    /// through.
    /// THE DOF MASK AND ITS FOLD, both on the device.
    ///
    /// The mask alone would be half a fix: a build-on-device that downloaded
    /// the whole mask to count it would keep the reduction on the host while
    /// moving the walk.
    #[test]
    fn the_dof_mask_and_its_sum_are_both_computed_on_the_device() {
        let mut scene = TestScene::new(5);
        {
            let props = scene.data.prop.vertex.as_mut_slice();
            // THREE SHAPES THE MASK MUST TELL APART: unpinned, pinned and free,
            // and pinned INSIDE a PDRD body. The last owns no per-vertex degree
            // of freedom, so it is not eliminated and its anchor keeps the
            // barrier; a mask that ignored `pdrd_body_index` would count it.
            props[0].fix_index = 0;
            props[1].fix_index = 2;
            props[2].fix_index = 5;
            props[3].fix_index = 9;
            props[3].pdrd_body_index = 1;
            props[4].fix_index = 0;
        }
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("the fixture scene allocates");
        crate::driver::state::reseed_props(&mut device, &mut state, &scene.data);

        let count = state.sizes.vertices as u32;
        let args = crate::driver::kernels::VertexDofRemovalMaskArgs {
            prop: state.prop_vertex.handle(),
            disable: 0,
            mask: state.dof_mask.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("test.dof_mask", &args, count) }
            .expect("the mask dispatches");

        // ONE LEVEL IS ENOUGH AT THIS SIZE, which is the fold's own tail case.
        let fold = crate::driver::kernels::VecBlockSumU32Args {
            source: state.dof_mask.handle(),
            length: count,
            width: crate::driver::state::DOF_FOLD_WIDTH as u32,
            total: state.dof_total.handle(),
            count: 1,
            seam_arena_count: 0,
        };
        // Safety: as above.
        unsafe { device.launch("test.dof_fold", &fold, 1) }.expect("the fold dispatches");
        state.dof_total.download(&mut device).expect("the total reads back");
        assert_eq!(
            state.dof_total.host()[0],
            2,
            "two vertices are pinned and free of a PDRD body; the third pin is \
             inside one and owns no degree of freedom to remove"
        );
    }

    #[test]
    fn the_pin_index_gather_copies_every_vertex_record() {
        let mut scene = TestScene::new(4);
        {
            let props = scene.data.prop.vertex.as_mut_slice();
            // A MIX RATHER THAN ALL-PINNED: a gather that wrote a constant, or
            // that wrote the right value at the wrong slot, has to be told
            // apart from one that copies each record.
            props[0].fix_index = 0;
            props[1].fix_index = 3;
            props[2].fix_index = 0;
            props[3].fix_index = 7;
        }
        let mut state = SolverState::default();
        let mut device = host_device();
        // Safety: the scene lives in its box for the whole test.
        unsafe { state.allocate(&mut device, &scene.data) }
            .expect("the fixture scene allocates");
        crate::driver::state::reseed_props(&mut device, &mut state, &scene.data);

        let count = state.sizes.vertices as u32;
        let args = crate::driver::kernels::VertexFixIndexFromRecordsArgs {
            prop: state.prop_vertex.handle(),
            fix_index: state.fix_index.handle(),
            count,
            seam_arena_count: 0,
        };
        // Safety: both arrays are borrowed for the whole call.
        unsafe { device.launch("test.fix_index", &args, count) }
            .expect("the gather dispatches");
        state
            .fix_index
            .download(&mut device)
            .expect("the gather reads back");
        assert_eq!(
            &state.fix_index.host()[..4],
            &[0u32, 3, 0, 7][..],
            "the gather copies each vertex's own pin index"
        );
    }
}
