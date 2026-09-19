// File: crates/ppf-cts-solver/src/driver/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The CPU backend: a Rust implementation of the 19-symbol backend surface.
//!
//! This module is selected by the `cpu` cargo feature, which makes `build.rs`
//! compile no C++ library at all. It supplies the same `extern "C"` symbols the
//! CUDA library exports, with `#[no_mangle]`, so the unconditional `extern "C"`
//! blocks in `backend.rs`, `main.rs` and `status_writer.rs` resolve here instead
//! and not one of their call sites moves. That property is the reason the
//! backend seam is worth keeping exactly where it is.
//!
//! # What this backend currently does
//!
//! It steps solids (tets), the SHELL MEMBRANE and SHELL BENDING, the ROD
//! STRETCH and ROD BENDING, STRAIN LIMITING on both shell faces and rods, and
//! SELF-CONTACT between meshes, under implicit Euler with projected Newton and
//! a block-Jacobi PCG, with exact Dirichlet pins, soft pull pins, the air
//! damper and the isotropic drag. All four shell material models are
//! assembled; the shell hinge carries its directional stiffness and both
//! bending terms carry their lagged Rayleigh damping; both strain limiters
//! carry their barrier AND the time of impact that truncates a step, which is
//! what makes an authored limit a bound rather than a discouragement. Every
//! other scene class is refused BY NAME AND BY COUNT at `initialize()`, and an
//! element class is several capabilities rather than one: a shell face's
//! inflate pressure keeps an entry of its own.
//!
//! The refusal is not a placeholder that will be forgotten: plan Decision 3
//! makes a named refusal the second of the two conditions that define
//! completeness, and the reason it is written first is that a backend which
//! quietly ran a scene it cannot solve would write a full run of plausible
//! frames with nothing in the output to say so. Each subsystem deletes exactly
//! the refusal it implements, in the commit that implements it.
//!
//! CONTACT IS LIVE, AND IT LANDED AS THREE PIECES AT ONCE. Non-penetration is
//! enforced structurally by the ACCD CCD-filtered line search plus
//! `check_intersection`, NOT by the barrier, which is a cubic energy finite at
//! the surface. A build carrying the barrier and only one of the other two
//! would COMPLETE and exit 0 while surfaces passed through each other, which is
//! the one outcome worse than a blanket refusal, so the refusal came off only
//! when all three were live together. `super::contact` holds them.
//!
//! Geometry that is not part of the solved namespace has the same shape and the
//! same rule. `super::collider` holds the analytic sphere and floor, which need
//! no hierarchy and so are assembled outside the `disable-contact` gate;
//! `super::contact` holds the rest-pose static collision mesh, whose three pair
//! types are assembled, swept and scanned beside the self-contact ones.
//!
//! `super::assemble::stitch` holds the cross-stitch, whose six-slot
//! barycentric spring lands in the fixed matrix between the elastic layers and
//! the `tmp_fixed` snapshot.
//!
//! What contact still refuses is refused BY ITS OWN NAME rather than by one
//! blanket entry: the SAND grain angular degree of freedom keeps a refusal.
//!
//! # Two gates, one function
//!
//! `refusal::scene_refusals` is called at `initialize()` AND again at the top of
//! every `advance()`. The second call is not redundancy for its own sake: the
//! host rebuilds the pin constraint from the schedule every step, so a scene can
//! gain a torque group at t = 2 that a gate running only at t = 0 never saw.
//! Sharing one function is what makes it impossible for the gate and the driver
//! to disagree about which scenes are steppable.
//!
//! `refusal::material_defects` is a THIRD gate and runs only at `initialize()`,
//! which is not an oversight: it reads the scene's material table, and a
//! material is fixed when the scene is built, so a second call would ask a
//! question whose answer cannot have changed. It is reported apart from the
//! refusals because it names a defect in the scene rather than a capability
//! this backend lacks.
//!
//! # The surface is honest even where the solver is absent
//!
//! A refusal covers the SOLVE. It does not cover the entry points the host
//! calls around a solve, and those are the dangerous ones, because an empty
//! body there returns success: a pin that never moves, a keyframe that is
//! dropped, a position gather that hands back zeros, a collision window that
//! never closes. Each of those produces a full run of plausible frames with
//! nothing in the output to say so, which is the one failure mode this backend
//! exists to make impossible. They are therefore implemented here whether or
//! not the driver that would consume them exists yet.
//!
//! # The fatal vocabulary this backend uses
//!
//! `crates/ppf-cts-formats/src/status/mod.rs` defines the codes; a run that
//! stamps the same one for everything tells a reader nothing. What each means
//! here, and where it can come from:
//!
//! | code | meaning here | live today |
//! |---|---|---|
//! | 1 `INIT_INTERSECTION` | an intersection found at `initialize()` | yes: the scan over the previous and the current positions, and no other path may borrow this code |
//! | 2 `OOM` | an allocation this backend could not make | yes: the collision-window table, and every driver buffer, which is sized at `initialize()` through `try_reserve` so a scene too large for the machine is a named refusal rather than a process abort |
//! | 3 `CUDA_DRIVER` | never applies: there is no driver | no, and never |
//! | 4 `SOLVER_INVARIANT` | a host-side contract violated before any kernel ran | yes |
//! | 5 `DEVICE_ASSERT` | an index a kernel range was about to dereference | yes: a constraint's vertex index, a tet's vertex or material index, and the vertex-face adjacency, all checked before the range that would read them |
//! | 6 `WATCHDOG_TIMEOUT`, 7 `ARCH_UNSUPPORTED` | GPU-only conditions | no |
//!
//! Code 1 having no site is stated rather than filled: taking it for a nearby
//! failure would make the one code that means "this scene starts already
//! penetrating" mean something else, and that is the code the penetration gate
//! reports through.

use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;

use crate::data::{
    Constraint, DataSet, IntersectionRecord, MaterialParamUpdate, ParamSet, RestShapeUpdate,
    StepResult,
};

mod assemble;
pub(crate) mod phase;
// EVERY `rust` ENTRY RENDERING, COMPILED. `kernels` includes the ones this
// driver dispatches; this module includes all of them, so a rendering nothing
// dispatches is still read by rustc. Nothing here is dispatchable and nothing
// constructs a record; read its own comment for what that buys and what it
// deliberately does not.
mod generated_entries;
mod collider;
mod collision_window;
pub mod contact;
mod constraint;
mod dirichlet;
mod dump_linsys;
/// The aggregate lock's projector: the dispatches that build the feasible
/// correction for a locked group and remove the constraint-space component from
/// a search direction.
mod lock;
/// The aggregate lock's host arithmetic: the per-group 3x3 and 4x4 reductions,
/// computed on the host in double precision.
mod lock_math;
mod plasticity;
mod refusal;
/// The PDRD reduced six-DOF solve's host orchestration.
mod pdrd;
mod rest_shape;
/// The PDRD reduction's topology: which vertices a rigid body carries, and
/// where every free vertex sits in the reduced vector.
mod rigid_map;
mod devsort;
mod scan;
mod schwarz;
mod scene;
mod seed;
mod state;
mod step;
#[cfg(test)]
mod test_scene;
pub mod bvh;
pub mod csr;
// The backend surface is `ppf-cts-compute`, and a driver module names that
// crate directly rather than reaching a local alias for it. Nothing about a
// device is declared here any more; what stayed behind is the one item on that
// seam which decides a value rather than a mechanism.
pub mod cadence;
// The launch table: one compiled entry point per kernel id. The device that
// consumes it is the compute crate's.
mod launch;
pub mod kernels;
pub mod fixedcsr;
pub mod log;
pub mod ccd;
pub mod dyncsr;
pub mod intersection;
pub mod lbvh;
pub mod pair_cache;
pub mod operator;
pub mod pcg;
pub mod reduce;
pub mod sort;
pub mod spmv;

pub use refusal::scene_refusals;

/// Borrow the collision-window table for one step.
///
/// The driver holds this for the whole step rather than asking per pass: the
/// masks are read by every broad-phase query and by the intersection scan, and
/// a table installed between two of those reads would give one step two
/// different answers about which geometry collides.
pub(crate) fn lock_collision_windows(
) -> Option<std::sync::MutexGuard<'static, CollisionWindows>> {
    COLLISION_WINDOWS.lock().ok()
}

use collision_window::CollisionWindows;
use scene::{Fatal, FatalResult, SceneView};

/// Fatal reason for the current process, mirroring `g_ppf_fatal_code` in
/// `src/kernels/main/fatal.hpp`. Read through `fatal_code()` by
/// `status_writer::atexit_fatal_hook`.
static FATAL_CODE: AtomicU8 = AtomicU8::new(0);

/// First-writer-wins latch for the fatal detail.
///
/// Several rayon workers can trip the same invariant in the same step, and the
/// first one to arrive is the one that describes the failure; a later writer
/// would overwrite a precise report with a consequence of it. The C++ backend
/// gets this property from being single-threaded on the host side, so it is one
/// of the few places the Rust backend needs a mechanism where CUDA needed none.
static FATAL_LATCHED: AtomicBool = AtomicBool::new(false);

/// Process-lived storage for the detail string.
///
/// `fatal_detail()` hands out a `*const c_char` that the caller reads after
/// the run, so the buffer must outlive every borrow of it; a `String` returned
/// by value could not satisfy that through a C signature.
static FATAL_DETAIL: Mutex<Option<std::ffi::CString>> = Mutex::new(None);

/// The live scene, set by `initialize()`.
///
/// The pointers inside `DataSet` are owned by the Rust caller in `backend.rs`
/// and outlive the run, so this stores the pointer rather than the struct.
static SCENE: Mutex<Option<Scene>> = Mutex::new(None);

/// The scene's collision-window table, installed once at `initialize()`.
static COLLISION_WINDOWS: Mutex<CollisionWindows> = Mutex::new(CollisionWindows::new());

/// The Newton driver's buffers and this step's pin copy.
///
/// One per process, as the scene is: `initialize()` sizes it and every step
/// reuses it, so a Newton iteration allocates nothing. See `state::SolverState`
/// for why the pins are COPIED here rather than borrowed.
static SOLVER: Mutex<Option<state::SolverState>> = Mutex::new(None);

/// The backend, and the only way the driver reaches one.
///
/// Held beside the solver state rather than inside it, because the two are the
/// two halves this seam separates: the state is the DRIVER's (the Newton
/// iterate, the matrices, the trees), and this is the BACKEND's (allocation, the
/// ranged launch, the diagnostic transport). A driver written against
/// `Device` is what the CUDA and Metal backends will be handed unchanged; see
/// `ppf_cts_compute`.
///
/// `Option` because `launch::host_device` is not a `const fn`, not because a run can
/// proceed without one.
static DEVICE: Mutex<Option<launch::Backend>> = Mutex::new(None);

/// Take the backend for the duration of one host entry point.
///
/// # Panics
/// Never: a poisoned lock is reported through the caller's own fatal path
/// rather than unwrapped here.
fn with_device<T>(
    body: impl FnOnce(&mut launch::Backend) -> FatalResult<T>,
) -> FatalResult<T> {
    let mut guard = DEVICE.lock().map_err(|_| {
        Fatal::invariant(
            "solver driver: the backend lock is poisoned, so an earlier failure went unreported",
        )
    })?;
    let device = guard.get_or_insert_with(launch::backend);
    body(device)
}

/// Open this build's device the way a run does and say what it is called.
///
/// For `ppf-contact-solver --probe`. The device is opened afresh and dropped at
/// the end of this call, never stored in [`DEVICE`], because a probe runs no
/// step and the process exits right after.
pub(crate) fn probe_device() -> Result<String, String> {
    use ppf_cts_compute::Device as _;
    launch::try_backend().map(|device| device.info().device_name.clone())
}

struct Scene {
    dataset: *const DataSet,
    /// MUTABLE, because the step writes back through it.
    ///
    /// `advance()` commits `prev_dt` and advances `time` on the host's own
    /// record, which is what the CUDA backend does through its `ParamSet *`
    /// and what the next step's velocity reconstruction and `backend.rs`'s
    /// checkpoint both read. `backend.rs` declares the entry point as
    /// `*const ParamSet` for every backend, so the write side of that contract
    /// is recovered here, once, rather than at each site.
    param: *mut ParamSet,
}

impl Scene {
    /// True when both records are still addressable.
    ///
    /// The check exists so `advance()` can tell "no scene was initialized" from
    /// "a scene was initialized and this backend cannot step it", which are
    /// different defects and would otherwise produce the same message.
    fn is_live(&self) -> bool {
        !self.dataset.is_null() && !self.param.is_null()
    }
}

// The scene record is reachable from the FFI entry points, which the host may
// call from any thread. Both pointers address memory owned by `backend.rs` for
// the whole run and neither is written through here.
unsafe impl Send for Scene {}

/// Stamp a fatal reason, first writer wins.
pub(crate) fn set_fatal(code: u8, detail: impl Into<String>) {
    if FATAL_LATCHED.swap(true, Ordering::SeqCst) {
        return;
    }
    FATAL_CODE.store(code, Ordering::SeqCst);
    let text = detail.into();
    // A NUL inside the message would truncate it at the C boundary, so it is
    // replaced rather than allowed to silently shorten the report.
    let text = text.replace('\0', "?");
    if let Ok(mut slot) = FATAL_DETAIL.lock() {
        *slot = std::ffi::CString::new(text).ok();
    }
}

/// Stop the run where the invariant broke, the way the C++ backends do.
///
/// `exit`, not `abort` or a panic: the host's terminal crash record is written
/// by a `libc::atexit` hook (`crate::status_writer`), which `abort()` does not
/// run and which an unwind out of an `extern "C"` function would not reach
/// either. Both channels carry the message, because `initialize()` can fail
/// before the frontend has a logger attached and stderr is what a terminal
/// sees.
fn fatal_exit(fatal: Fatal) -> ! {
    // NO BACKEND NAME. This driver is shared by all three, so naming one here
    // is wrong on two of them: a CUDA run and a Metal run both reported
    // "### cpu: FATAL" until this was corrected, which sends whoever reads the
    // failure to the wrong backend. Every other `###` line in this driver names
    // the CONDITION and not the machine.
    ::log::error!("### FATAL: {}", fatal.detail);
    eprintln!("### FATAL: {}", fatal.detail);
    set_fatal(fatal.code, fatal.detail);
    std::process::exit(1);
}

/// The live scene, or a fatal naming the entry point that arrived too early.
///
/// The host's step loop calls `initialize()` first and stops the run when it
/// returns false, so reaching one of these entry points before that is a defect
/// in the caller rather than a mode this backend supports. Without the check it
/// would be a null dereference, or worse a quiet return that looks like a
/// keyframe with nothing in it.
fn require_scene(who: &str) -> SceneView {
    let live = SCENE
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|scene| (scene.is_live(), scene.dataset)));
    match live {
        Some((true, dataset)) => {
            // Safety: `backend.rs` owns the `DataSet` for the whole run and it
            // was non-null when `initialize()` stored it.
            unsafe { SceneView::new(dataset) }
        }
        _ => fatal_exit(Fatal::invariant(format!(
            "solver driver: {who} was called before initialize() succeeded"
        ))),
    }
}

// ---------------------------------------------------------------------------
// The 19-symbol surface.
//
// Every one is present from the first commit, including the three that are easy
// to miss because they are declared outside `backend.rs`: `set_log_path` lives
// in `main.rs`, and `fatal_code` / `fatal_detail` in `status_writer.rs`.
// A missing symbol is a link failure rather than a silent gap, which is why the
// stubs are written before any physics.
// ---------------------------------------------------------------------------

/// # Safety
/// `data_dir` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn set_log_path(data_dir: *const c_char) {
    if data_dir.is_null() {
        return;
    }
    // KEEP THE PATH. Creating the directory and then dropping it would leave no
    // per-step indicator stream writable: `advance.iter.out`,
    // `advance.SL_toi.out` and `advance.max_sigma.out` are how a lost PCG
    // residual denominator is diagnosed, and their absence reads as a run that
    // produced no indicators rather than as a backend with no channel.
    //
    // THE LINE BELOW IS `debug!` AND NOT `info!`, because a default run already
    // announces the directory: `log::set_path` prints
    // `* data_directory_path path = ...` bare through `message`, in the same
    // mark-shaped vocabulary as the rest of the streamed transcript. A second
    // line saying the same thing in this driver's own words would put two
    // spellings of one fact in the default stream.
    match CStr::from_ptr(data_dir).to_str() {
        Ok(path) => {
            if log::set_path(std::path::Path::new(path)) {
                ::log::debug!("solver driver: log path {path}");
            }
        }
        Err(error) => {
            // Reported rather than swallowed: a path this backend cannot read
            // is a path whose streams silently go nowhere.
            ::log::warn!("solver driver: the log path is not valid UTF-8 ({error})");
        }
    }
}

#[no_mangle]
pub extern "C" fn fatal_code() -> u8 {
    FATAL_CODE.load(Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn fatal_detail() -> *const c_char {
    // The guard keeps the pointer valid for the caller: the CString stays owned
    // by the static, and only its interior pointer is handed out.
    static EMPTY: &[u8] = b"\0";
    match FATAL_DETAIL.lock() {
        Ok(slot) => match slot.as_ref() {
            Some(text) => text.as_ptr(),
            None => EMPTY.as_ptr() as *const c_char,
        },
        Err(_) => EMPTY.as_ptr() as *const c_char,
    }
}

/// Refuse a CPU that cannot run the instructions the kernels were built for.
///
/// `build.rs` compiles the shared bodies at an ISA baseline carrying FMA
/// (`x86-64-v3` by default), because a contraction POLICY alone cannot fuse into
/// an instruction the target lacks and the resulting non-contracting oracle
/// disagrees with CUDA and Metal by a wrong tolerance rather than a wrong answer.
/// The cost of that baseline is that an older CPU meets an unknown opcode.
///
/// Without this check that arrives as `SIGILL` from inside a shared body, which
/// names neither the cause nor the fix. With it, the run stops at `initialize()`
/// saying which feature is missing and which environment variable rebuilds for
/// this machine.
///
/// The detection itself is safe to run here: `is_x86_feature_detected!` expands
/// to a CPUID query in the RUST crate, which is not built at the raised
/// baseline, so nothing in this function can be the illegal instruction it is
/// looking for.
fn check_host_baseline() -> Result<(), String> {
    let built = env!("PPF_HOST_BASELINE_BUILT");
    #[cfg(target_arch = "x86_64")]
    {
        // Only the default baseline is decomposed into features. A caller who
        // overrode PPF_HOST_BASELINE chose the target and owns the consequence.
        if built == "x86-64-v3" {
            let missing: Vec<&str> = [
                ("avx", is_x86_feature_detected!("avx")),
                ("avx2", is_x86_feature_detected!("avx2")),
                ("fma", is_x86_feature_detected!("fma")),
                ("bmi1", is_x86_feature_detected!("bmi1")),
                ("bmi2", is_x86_feature_detected!("bmi2")),
            ]
            .iter()
            .filter(|(_, present)| !present)
            .map(|(name, _)| *name)
            .collect();
            if !missing.is_empty() {
                return Err(format!(
                    "solver driver: the shared kernel bodies were compiled for the \
                     {built} baseline, and this CPU is missing {}. Running them \
                     would trap on an unknown instruction inside a kernel. \
                     Rebuild with PPF_HOST_BASELINE set to a baseline this machine \
                     carries (for example x86-64-v2), accepting that a baseline \
                     without FMA makes this backend a non-contracting oracle that \
                     disagrees with CUDA and Metal.",
                    missing.join(", ")
                ));
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // THE AARCH64 BASELINE IS A TUNING MODEL, NOT A FEATURE SET, WHICH IS
        // WHY THIS BRANCH VERIFIES LESS THAN THE x86 ONE ABOVE AND SAYS SO.
        //
        // On macOS, `-mcpu=apple-m1` asks clang to SCHEDULE for that core. It
        // does not require an instruction a later Apple part lacks, because every
        // Apple Silicon Mac is an M1 or newer, so the floor cannot be missed by a
        // machine capable of running this binary at all. On every other aarch64
        // target no core is requested and the recorded baseline is "none": the
        // bodies are built for the ARMv8 baseline, which every such machine
        // carries and which already includes FMA. Either way there is no CPUID
        // decomposition to do and nothing that would trap.
        //
        // What WOULD trap is an override: a caller who set PPF_HOST_BASELINE to
        // a core newer than the running machine gets instructions it may not
        // have. `std::arch::is_aarch64_feature_detected!` cannot answer "is this
        // an M2", so that case is not detectable here and is the caller's own,
        // exactly as an x86 override is. Named rather than silently skipped.
        let _ = built;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No baseline is requested on any other architecture, so there is
        // nothing to verify. Named rather than silently skipped.
        let _ = built;
    }
    Ok(())
}

/// # Safety
/// `data` and `param` must point at live `DataSet` / `ParamSet` records that
/// outlive the run, as `backend.rs` guarantees.
#[no_mangle]
pub unsafe extern "C" fn initialize(data: *const DataSet, param: *const ParamSet) -> bool {
    // Name: Initialization Time
    // Format: list[(time, ms)]
    // Description:
    // Total wall-clock time in milliseconds spent inside the one-time
    // solver initialization (buffer allocation, contact setup, initial
    // LBVH build, initial intersection check). Only a single record is
    // expected, written when the initialize scope exits. The leading
    // time column is the simulation time at the moment of recording,
    // which is 0 for initialization.
    let _section = log::Section::new("initialize");
    // The block above DECLARES the channel, and it sits immediately above the
    // call because `parsers.rs` registers a channel only when a `Name:` field
    // precedes it and treats every comment line after `Description:` as the
    // text. A note of ours inside that run would become part of what the addon
    // shows, so this one is below.
    //
    // DECLARED FIRST SO IT DROPS LAST, which matters more here than in a step:
    // this function has a dozen early `return false` paths and every one of
    // them is a setup failure a reader wants the footer for.
    // The host-phase tally's exit hook, installed here so a run that ends
    // without unwinding still prints it. No-op unless PPF_PHASE_STATS is set.
    phase::register();
    if data.is_null() || param.is_null() {
        set_fatal(
            ppf_cts_formats::status::error_code::SOLVER_INVARIANT,
            "solver driver: initialize() received a null DataSet or ParamSet",
        );
        return false;
    }

    if let Err(message) = check_host_baseline() {
        ::log::error!("{message}");
        set_fatal(
            ppf_cts_formats::status::error_code::SOLVER_INVARIANT,
            message,
        );
        return false;
    }

    if let Ok(mut slot) = SCENE.lock() {
        *slot = Some(Scene {
            dataset: data,
            param: param as *mut ParamSet,
        });
    }

    // THE MATERIAL GATE RUNS FIRST, and it is a different question from the
    // capability gate below. An element naming a model its own element kind has
    // no form of is a defect in the SCENE, which the CUDA backend answers with a
    // live device assert inside the assembly, so reporting it before the
    // refusals keeps a user from being sent to a host that fails the same way.
    // The elastic assembly's own dispatch returns the same verdict per element;
    // see `refusal::material_defects` for why the answer is knowable here and
    // what asking the device for it instead would cost.
    let defects = refusal::material_defects(&*data);
    if !defects.is_empty() {
        let mut message = String::from(
            "this scene names an elastic model on an element kind that has no form of it, which \
             no backend can assemble:",
        );
        for defect in &defects {
            message.push_str("\n  - ");
            message.push_str(&defect.describe());
        }
        message.push_str(
            "\nCorrect the material on those elements. A CUDA host answers the same scene with a \
             live device assert inside the elastic assembly.",
        );
        ::log::error!("{message}");
        set_fatal(
            ppf_cts_formats::status::error_code::SOLVER_INVARIANT,
            message,
        );
        return false;
    }

    // The capability gate. Every scene class this backend cannot solve is named
    // here with the count that was found, before a frame is written.
    let refusals = scene_refusals(&*data, &*param);
    if !refusals.is_empty() {
        let mut message = String::from(
            "the CPU backend cannot solve this scene yet, and refuses it rather \
             than writing frames computed for a different problem:",
        );
        for refusal in &refusals {
            message.push_str("\n  - ");
            message.push_str(&refusal.describe());
        }
        message.push_str(
            "\nRun this scene on a CUDA host. Each entry is removed in the change \
             that implements it.",
        );
        ::log::error!("{message}");
        set_fatal(
            ppf_cts_formats::status::error_code::SOLVER_INVARIANT,
            message,
        );
        return false;
    }

    // The scene is accepted, so size the driver's buffers now: an allocation
    // inside a Newton iteration has no good answer, while one here can be
    // reported before a frame is written.
    let mut allocated = state::SolverState::default();
    if let Err(fatal) = with_device(|device| allocated.allocate(device, &*data)) {
        ::log::error!("{}", fatal.detail);
        set_fatal(fatal.code, fatal.detail);
        return false;
    }
    // THE TRANSLATION-LOCK INVARIANT, ON THE SEEDED POSE. A scene whose
    // authored pose already violates its own lock is reported here rather than
    // after a step has moved it, so the verdict names the scene instead of the
    // solve.
    {
        let groups = (*data).translation_lock.size as usize;
        if groups > 0 {
            // Safety: the scene is live and holds `groups` lock records.
            let locks: &[crate::data::TranslationLock] =
                std::slice::from_raw_parts((*data).translation_lock.data, groups);
            let rows = lock::RowInputs {
                lock_index: allocated.translation_lock_index.handle(),
                locks: allocated.translation_lock.handle(),
                prop: allocated.prop_vertex.handle(),
                positions: allocated.positions.handle(),
                dof_mask: allocated.dof_mask.handle(),
                initial: allocated.translation_lock_initial.handle(),
            };
            let seeded = allocated.positions.handle();
            let vertices = allocated.sizes.vertices;
            let state::SolverState {
                lock_drift,
                lock_max_displacement,
                ..
            } = &mut allocated;
            if let Err(fatal) = with_device(|device| {
                lock::check_invariant(
                    device,
                    rows,
                    seeded,
                    locks,
                    lock_drift,
                    lock_max_displacement,
                    vertices,
                    "initial state",
                )
            }) {
                ::log::error!("{}", fatal.detail);
                set_fatal(fatal.code, fatal.detail);
                return false;
            }
        }
    }

    // THE ANALYTIC COLLIDER LAYER, allocated on every scene: a sphere and a
    // floor are assembled outside the `disable-contact` gate, so a scene that
    // set it still needs this.
    match with_device(|device| collider::Analytic::allocate(device, &*data)) {
        Ok(analytic) => allocated.analytic = Some(analytic),
        Err(fatal) => {
            ::log::error!("{}", fatal.detail);
            set_fatal(fatal.code, fatal.detail);
            return false;
        }
    }
    // THE CONTACT SUBSYSTEM AND THE INITIAL PENETRATION CHECK. A scene that
    // starts already tangled cannot be fixed by any step, so it is reported
    // here, before a frame is written, under the one error code that means
    // exactly that.
    if !(*param).disable_contact {
        let mut live = match with_device(|device| contact::Contact::allocate(device, &*data)) {
            Ok(live) => live,
            Err(fatal) => {
                ::log::error!("{}", fatal.detail);
                set_fatal(fatal.code, fatal.detail);
                return false;
            }
        };
        // The masks are not applied here: the collision window table may not
        // have been installed yet, and a scene that starts intersecting is
        // intersecting whether or not the pair is scheduled to collide later.
        let windows = contact::Windows::default();
        // The seeded device copies, which `allocate` above filled from the
        // scene. The build-time pose is the same on both sides here.
        let prev = allocated.positions_prev.handle();
        let curr = allocated.positions.handle();
        // Safety: both `CVec`s hold `vertices` position triples, and the device
        // copies above were seeded from exactly these arrays, so the mirror and
        // the host array agree here without a download.
        let (prev_host, curr_host) = unsafe {
            (
                std::slice::from_raw_parts((*data).vertex.prev.data as *const f32, 3 * allocated.sizes.vertices),
                std::slice::from_raw_parts((*data).vertex.curr.data as *const f32, 3 * allocated.sizes.vertices),
            )
        };
        let mesh_refs = allocated.refs();
        // THE TREES ARE BUILT ONCE, OVER `curr`, AND BOTH POSES ARE SCANNED
        // AGAINST THEM: `rebuild_trees` below fits every BVH to `vertex.curr`,
        // and the loop after it runs `check_intersection` on `prev` and then on
        // `curr` with no rebuild between them.
        //
        // REBUILDING PER POSE IS NOT THE SAFER READING OF THE SAME CHECK, it is
        // a different one. Boxes fitted to `prev` bound that pose tightly, so a
        // per-pose rebuild finds pairs the `curr`-fitted boxes do not, and this
        // gate REFUSES a scene: tightening it here declines scenes that are
        // legal, which costs a user a run rather than costing a guarantee. At
        // `initialize()` the two poses are the authored one on a fresh scene
        // and one step apart on a resume, so the boxes bound both.
        let initial_lbvh = std::time::Instant::now();
        if let Err(fatal) =
            with_device(|device| live.rebuild_trees(device, &*data, mesh_refs, curr, windows))
        {
            ::log::error!("{}", fatal.detail);
            set_fatal(fatal.code, fatal.detail);
            return false;
        }
        // Name: Initial LBVH Build Time
        // Format: list[(time, ms)]
        // Map: initial_lbvh_build
        // Description:
        // Wall-clock time in milliseconds to build the initial LBVH (Linear
        // Bounding Volume Hierarchy) over faces, edges, and vertices at the
        // start of the simulation, including the collision-mesh BVH. Only a
        // single record is expected.
        log::mark("initialize", "lbvh_build", initial_lbvh.elapsed());
        // THE SCAN'S OWN TIMER, one scope spanning BOTH poses. It is marked on
        // the failure path as well as the clean one (see the `log::mark` beside
        // the `return false` below), so a scene refused for starting tangled
        // still reports what the scan cost.
        let initial_scan = std::time::Instant::now();
        for (pose, pose_host, which) in [
            (prev, prev_host, "previous"),
            (curr, curr_host, "current"),
        ] {
            let scan = with_device(|device| {
                live.check_intersection(device, &*data, mesh_refs, pose, pose_host, allocated.prop_vertex.host(), windows)
            });
            let report = match scan {
                Ok(report) => report,
                Err(fatal) => {
                    ::log::error!("{}", fatal.detail);
                    set_fatal(fatal.code, fatal.detail);
                    return false;
                }
            };
            publish_intersection_records(&report);
            // THE CLEAN PATH SAYS SO, and that is not noise. This scan is the
            // gate that refuses a scene which starts already tangled, and a
            // gate nothing can observe is not a gate: a fixture asserting it
            // ran, and a reader asking whether it did, both have nothing to
            // read when it passes silently. The same work is timed as the
            // `check_intersection` channel below, published as
            // `initial_check_intersection`.
            //
            // INSIDE THE BRANCH, not before it. Logged unconditionally this
            // line asserts the scan found nothing and then the next one says
            // what it found, which is a log contradicting itself one line
            // later.
            if report.is_clean() {
                ::log::info!(
                    "check intersection: the scene's {which} positions carry \
                     no intersecting pair"
                );
            } else {
                let message = format!(
                    "### intersection detected: the scene's {which} positions carry {} \
                     intersecting pair(s). A run cannot start from a tangled state: no step \
                     can separate geometry that already crosses, so the solver would either \
                     stall or report the same pairs every frame. Separate the geometry, or \
                     mark the pairs with an intersection allowance if the tangle is authored",
                    report.found
                );
                ::log::error!("{message}");
                set_fatal(
                    ppf_cts_formats::status::error_code::INIT_INTERSECTION,
                    message,
                );
                log::mark("initialize", "check_intersection", initial_scan.elapsed());
                return false;
            }
        }
        // Name: Initial Intersection Check Time
        // Format: list[(time, ms)]
        // Map: initial_check_intersection
        // Description:
        // Wall-clock time in milliseconds spent scanning the previous and
        // current vertex positions for self-intersections at the start of
        // the simulation. Only a single record is expected. Useful for
        // diagnosing geometry that begins the simulation already tangled.
        log::mark("initialize", "check_intersection", initial_scan.elapsed());
        allocated.contact = Some(live);
    }

    if let Ok(mut slot) = SOLVER.lock() {
        *slot = Some(allocated);
    }

    true
}

/// # Safety
/// `result` must point at a writable `StepResult`.
#[no_mangle]
pub unsafe extern "C" fn advance(result: *mut StepResult) {
    if result.is_null() {
        return;
    }
    // Opened pessimistic: every field is overwritten by a successful step, and
    // a failure before the driver runs leaves a record that says no progress
    // was made rather than one that says a clean step was taken.
    (*result).time = 0.0;
    (*result).ccd_success = true;
    (*result).pcg_success = true;
    (*result).intersection_free = true;
    (*result).newton_progress = false;
    (*result).pin_feasible = true;
    (*result).contact_separated = true;

    let live = SCENE
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|scene| (scene.is_live(), scene.dataset, scene.param)));
    let Some((true, dataset, param)) = live else {
        fatal_exit(Fatal::invariant(
            "solver driver: advance() was called before initialize() succeeded",
        ));
    };
    let Ok(mut guard) = SOLVER.lock() else {
        fatal_exit(Fatal::invariant(
            "solver driver: the solver state is poisoned, so an earlier failure went unreported",
        ));
    };
    let Some(state) = guard.as_mut() else {
        fatal_exit(Fatal::invariant(
            "solver driver: initialize() accepted this scene without sizing the solver state",
        ));
    };
    // Safety: `backend.rs` owns both records for the whole run, and the host's
    // step loop calls one backend entry point at a time, so no other reference
    // to the scene's buffers is alive here.
    match with_device(|device| step::advance(device, &*dataset, param, state)) {
        Ok(outcome) => *result = outcome,
        Err(fatal) => {
            drop(guard);
            fatal_exit(fatal);
        }
    }
}

/// Copy the committed positions into the arrays `backend.rs` reads.
///
/// `fetch()` copies `vertex.curr` and `vertex.prev` device-to-host. The solve
/// runs over device allocations (`SolverState::positions`), so the host `CVec`s
/// inside `DataSet` still hold the pose the step began from until this function
/// writes them.
///
/// AN EMPTY OR SILENTLY-FAILING `fetch()` IS THE FROZEN-ANIMATION DEFECT, and it
/// is a measured one: every frame writes the build-time pose, the run completes,
/// and nothing in the output says the solve was discarded. So every path out of
/// this function either copies or is fatal, and none returns quietly.
#[no_mangle]
pub extern "C" fn fetch() {
    let Some(dataset) = SCENE
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|scene| scene.dataset))
    else {
        fatal_exit(Fatal::invariant(
            "solver driver: fetch() was called with no live scene, so the frame would carry the \
             build-time pose with nothing in the output saying so",
        ));
    };
    let Ok(mut guard) = SOLVER.lock() else {
        fatal_exit(Fatal::invariant(
            "solver driver: the solver state is poisoned, so an earlier failure went unreported",
        ));
    };
    let Some(state) = guard.as_mut() else {
        fatal_exit(Fatal::invariant(
            "solver driver: fetch() was called before initialize() sized the solver state",
        ));
    };
    let vertices = state.sizes.vertices;
    if vertices == 0 {
        return;
    }
    let downloaded = with_device(|device| {
        state
            .positions
            .download(device)
            .and_then(|()| state.positions_prev.download(device))
            // THE STATISTICS COUNTER RIDES THE SAME READBACK: it is per-frame
            // telemetry, so it is read once at the frame boundary rather than
            // per Newton iteration, and it carries the count the LAST
            // iteration's assembly deposited.
            .and_then(|()| state.statistics_contact_count.download(device))
            .map_err(|fault| {
                Fatal::invariant(&format!(
                    "solver driver: the committed positions could not be read back ({fault:?}), \
                     so this frame would carry the previous pose with nothing in the output \
                     saying so"
                ))
            })
    });
    if let Err(fatal) = downloaded {
        drop(guard);
        fatal_exit(fatal);
    }
    // Safety: `backend.rs` owns the dataset for the whole run and calls one
    // backend entry point at a time, and both `CVec`s hold `vertices` position
    // triples, which is what `allocate` sized the device buffers to.
    unsafe {
        std::ptr::copy_nonoverlapping(
            state.positions.host().as_ptr(),
            (*dataset).vertex.curr.data as *mut f32,
            3 * vertices,
        );
        std::ptr::copy_nonoverlapping(
            state.positions_prev.host().as_ptr(),
            (*dataset).vertex.prev.data as *mut f32,
            3 * vertices,
        );
        let objects = state.statistics_contact_count.len();
        if objects > 0 {
            std::ptr::copy_nonoverlapping(
                state.statistics_contact_count.host().as_ptr(),
                (*dataset).statistics_contact_count.data,
                objects,
            );
        }
    }
}

/// Copy the crept inverse rest matrices into the arrays `save_state` serializes.
///
/// NOT EMPTY ANY MORE, and its old comment is why: it read "`inv_rest2x2` and
/// `inv_rest3x3` are the host's own arrays, and the plasticity creep, when it
/// lands, writes them in place", which stopped being true when they became
/// staged buffers. That is the THIRD comment in this file to describe a live
/// defect the moment the model under it changed; see [`fetch`] and
/// [`fetch_rest_angles`].
///
/// A DOWNLOAD IS NEEDED, as in [`fetch`], and this is the FOURTH comment in
/// this file to have described a live defect the moment the model under it
/// changed. It read "NO DOWNLOAD IS NEEDED ... these are `StagedBuffer`s and
/// every writer is a host writer, so the mirror is the authority", which stopped
/// being true when the plastic commit became a kernel: the device is the
/// authority now and the mirror is stale from the commit until this reads it.
#[no_mangle]
pub extern "C" fn fetch_inv_rest() {
    let Some(dataset) = SCENE
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|scene| scene.dataset))
    else {
        return;
    };
    let Ok(mut guard) = SOLVER.lock() else {
        fatal_exit(Fatal::invariant(
            "solver driver: the solver state is poisoned, so an earlier failure went unreported",
        ));
    };
    let Some(state) = guard.as_mut() else {
        return;
    };
    // Safety: `backend.rs` owns the dataset for the whole run, and each staged
    // buffer was sized from its own array at `allocate`.
    let read = with_device(|device| {
        state
            .inv_rest2x2
            .download(device)
            .and_then(|()| state.inv_rest3x3.download(device))
            .map_err(|fault| {
                Fatal::invariant(&format!(
                    "solver driver: the crept rest matrices could not be read back \
                     ({fault:?}), so a checkpoint would carry the build-time rest shape \
                     with nothing in the output saying so"
                ))
            })
    });
    if let Err(fatal) = read {
        drop(guard);
        fatal_exit(fatal);
    }
    unsafe {
        let n2 = state.inv_rest2x2.len();
        if n2 > 0 {
            std::ptr::copy_nonoverlapping(
                state.inv_rest2x2.host().as_ptr(),
                (*dataset).inv_rest2x2.data as *mut f32,
                n2,
            );
        }
        let n3 = state.inv_rest3x3.len();
        if n3 > 0 {
            std::ptr::copy_nonoverlapping(
                state.inv_rest3x3.host().as_ptr(),
                (*dataset).inv_rest3x3.data as *mut f32,
                n3,
            );
        }
    }
}

/// Copy the crept vertex properties back into the array `save_state` serializes.
///
/// THIS IS NOT EMPTY, and residency is why. `VertexProp::rest_bend_angle`
/// creeps in `SolverState::prop_vertex`, which is device-resident, so without
/// this copy a checkpoint holds the BUILD-TIME rest angles and the run
/// completes with nothing saying so.
///
/// `HingeProp::rest_angle` is NOT here: hinge props are host-resident, so this
/// backend mutates them in place and a crept rest angle is already in the array
/// `save_state` serializes.
#[no_mangle]
pub extern "C" fn fetch_rest_angles() {
    let Some(dataset) = SCENE
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|scene| scene.dataset))
    else {
        return;
    };
    let Ok(mut guard) = SOLVER.lock() else {
        fatal_exit(Fatal::invariant(
            "solver driver: the solver state is poisoned, so an earlier failure went unreported",
        ));
    };
    let Some(state) = guard.as_mut() else {
        return;
    };
    let count = state.prop_vertex.len();
    if count == 0 {
        return;
    }
    // Safety: `backend.rs` owns the dataset for the whole run and calls one
    // backend entry point at a time, and the staged buffer was sized from this
    // very array at `allocate`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            state.prop_vertex.host().as_ptr(),
            (*dataset).prop.vertex.data,
            count.min((*dataset).prop.vertex.size as usize),
        );
    }
}

/// The empty dynamic-CSR contract: a backend with no assembled contact matrix
/// reports no values and no rows, which is what the checkpoint writer expects.
///
/// # Safety
/// Both out-parameters must be writable.
#[no_mangle]
pub unsafe extern "C" fn fetch_dyn_counts(n_value: *mut u32, n_offset: *mut u32) {
    if !n_value.is_null() {
        *n_value = 0;
    }
    if !n_offset.is_null() {
        *n_offset = 0;
    }
}

/// # Safety
/// Pointers are unused while the dynamic matrix is empty.
#[no_mangle]
pub unsafe extern "C" fn fetch_dyn(_index: *mut u32, _value: *mut f32, _offset: *mut u32) {}

/// # Safety
/// Pointers are unused while the dynamic matrix is empty.
#[no_mangle]
pub unsafe extern "C" fn update_dyn(_index: *const u32, _offset: *const u32) {}

/// Rebuild the pin indices and the per-element `fixed` flags for this step.
///
/// See `constraint::rebuild` for what this owes and what it deliberately does
/// not do (it must not write a pin's position into `vertex.curr`).
///
/// # Safety
/// `constraint` must point at a live `Constraint`.
#[no_mangle]
pub unsafe extern "C" fn update_constraint(constraint: *const Constraint) {
    if constraint.is_null() {
        fatal_exit(Fatal::invariant(
            "solver driver: update_constraint received a null Constraint",
        ));
    }
    let view = require_scene("update_constraint");
    if let Err(fatal) = constraint::rebuild(&view, &*constraint) {
        fatal_exit(fatal);
    }
    // COPY THE PINS, do not keep the handles. `backend.rs` drops the previous
    // `Constraint` and frees its buffers as it assigns the new one, so a
    // pointer kept from here to `advance()` would address freed memory. The
    // CUDA backend does not have the problem because its own `update_constraint`
    // uploads the arrays to the device, which is this copy.
    //
    // `DataSet::constraint` is NOT a substitute: `backend.rs` never assigns it,
    // so it holds whatever scene build left there, and a step reading it would
    // honor last week's pins with nothing in the output to say so.
    if let Ok(mut guard) = SOLVER.lock() {
        if let Some(state) = guard.as_mut() {
            let fix = scene::slice(&(*constraint).fix);
            let pull = scene::slice(&(*constraint).pull);
            if let Err(fatal) = with_device(|device| state.stash_pins(device, fix, pull)) {
                drop(guard);
                fatal_exit(fatal);
            }
            // THE PER-VERTEX PIN INDICES MUST REACH THE DEVICE TOO, and this is
            // the only place that does it. `constraint::rebuild` has just
            // rewritten `fix_index` and `pull_index` in the host `DataSet` from
            // the new pin set, and nothing else carries those two fields
            // across. Without the restage below the staged buffer keeps what
            // `allocate()` put there, so the momentum row's
            // `prop[i].fix_index > 0` gate and the Dirichlet `dof_mask` both
            // read the BUILD-TIME pin set: a pin reaching its `unpin_time` is
            // released on the host and still prescribed on the device, and the
            // vertex assembles no row at all while `compute_target` has already
            // stopped driving it. A scene with a static pin set is unaffected,
            // which is why this is invisible to the sweep.
            //
            // ONLY THE TWO INDICES ARE COPIED. A wholesale copy from the
            // `DataSet` would be wrong in this direction: the staged buffer is
            // the AUTHORITY for `rest_bend_angle`, which `plasticity.rs` creeps
            // through `at()` every step and pushes back to the `DataSet` only at
            // checkpoint time, so copying the whole record would reset every
            // crept rod rest angle to its build-time value. The mirror image of
            // that hazard, a host write clobbering the device's crept values, is
            // guarded in `plasticity.rs`, which downloads them before it writes
            // the host array.
            if let Err(fatal) = with_device(|device| {
                state.restage_pin_indices(device, view.vertex_props_mut())?;
                // THE PER-ELEMENT `fixed` FLAGS TOO, which `constraint::rebuild`
                // has just recomputed as the AND over each element's vertices
                // and which the CONTACT kernels read through `MeshRefs`. Same
                // defect, different consumer.
                state.restage_element_fixed(device, view.face_props_mut(), view.edge_props_mut())
            }) {
                drop(guard);
                fatal_exit(fatal);
            }
            let sphere = scene::slice(&(*constraint).sphere);
            let floor = scene::slice(&(*constraint).floor);
            if let Err(fatal) =
                with_device(|device| state.stash_colliders(device, sphere, floor))
            {
                drop(guard);
                fatal_exit(fatal);
            }
            let torque_groups = scene::slice(&(*constraint).torque_groups);
            let torque_vertices = scene::slice(&(*constraint).torque_vertices);
            if let Err(fatal) = with_device(|device| {
                state.stash_torque(device, torque_groups, torque_vertices)
            }) {
                drop(guard);
                fatal_exit(fatal);
            }
            let stitch = scene::slice(&(*constraint).stitch);
            // SOLVER THEN DEVICE, the order `advance` already takes, so the
            // added lock introduces no new pair to order.
            if let Err(fatal) = with_device(|device| state.stash_stitches(device, stitch)) {
                drop(guard);
                fatal_exit(fatal);
            }
        }
    }
    // NOTHING IN THIS RECORD IS REFUSED PER STEP ANY MORE. The torque groups
    // were the last entry on that list and are stashed and assembled above, as
    // the analytic colliders and the cross-stitches already were. The list
    // existed because a scene can gain a constraint between steps,
    // `make_constraint` being rebuilt from the schedule every step, so a
    // constraint that switches on at t = 2 is invisible to a gate that only ran
    // at t = 0. That hazard has not gone away: a constraint kind added here
    // that the assembly does not carry owes its check back.
}

/// Replace this frame's inverse rest matrices and per-element exclusion mask.
///
/// # Safety
/// `update` must point at a live `RestShapeUpdate`.
#[no_mangle]
pub unsafe extern "C" fn update_rest_shape(update: *const RestShapeUpdate) {
    if update.is_null() {
        fatal_exit(Fatal::invariant(
            "solver driver: update_rest_shape received a null RestShapeUpdate",
        ));
    }
    let view = require_scene("update_rest_shape");
    if let Err(fatal) = rest_shape::apply(&view, &*update) {
        fatal_exit(fatal);
    }
    // AND THE STREAMED MATRICES MUST REACH THE DEVICE. `rest_shape::apply` has
    // written the live `DataSet`, but every elastic and strain-limit dispatch
    // reads a STAGED device buffer, seeded once at `allocate()`. Without the
    // restage below a scene carrying rest-shape keyframes simulates against the
    // BUILD-TIME rest pose for the whole run, silently.
    //
    // SOLVER THEN DEVICE, the order `advance` and `update_constraint` already
    // take, so this adds no new pair to order.
    if let Ok(mut guard) = SOLVER.lock() {
        if let Some(state) = guard.as_mut() {
            if let Err(fatal) =
                with_device(|device| {
                    state.restage_rest_shape(
                        device,
                        view.inv_rest2x2_mut(),
                        view.inv_rest3x3_mut(),
                    )?;
                    // AND THE `rest_excluded` MASK THIS PATH JUST WROTE.
                    // `rest_shape::apply` sets it on the HOST face props and
                    // the membrane's gate reads it off the DEVICE record
                    // (`kernels/energy/face_force.kernel.cpp:342`), and the
                    // only other uploader of those props is
                    // `update_constraint`, which owns `fixed` and runs on its
                    // own schedule. Without this the mask reaches the device
                    // one frame late: on the frame a face is newly flagged
                    // near-singular the membrane assembles it anyway, against
                    // the freshly restaged near-singular `inv_rest2x2` above,
                    // which is exactly the pairing the exclusion exists to
                    // prevent.
                    //
                    // COPYING `face_prop` INSIDE THIS ENTRY POINT IS WHAT
                    // CLOSES THAT WINDOW: it leaves no stale device state and
                    // no ordering dependence on `update_constraint`, which
                    // touches only `fixed`.
                    //
                    // THE TET SIDE NEEDS NOTHING. `rest_shape::apply` writes
                    // the tet mask too, and `tet_elastic` reads it on the HOST
                    // out of the live `DataSet` (`assemble.rs:311`), so it is
                    // current already; no tet prop buffer is staged at all.
                    state.restage_element_fixed(
                        device,
                        view.face_props_mut(),
                        view.edge_props_mut(),
                    )
                })
            {
                drop(guard);
                fatal_exit(fatal);
            }
        }
    }
}

/// Replace this frame's material tables with the streamed ones.
///
/// The animated-parameter path calls this once per step, right after the pin
/// constraint and the streamed rest shape and for the same reason: the energy
/// kernels re-read the material every Newton iteration, so overwriting the
/// tables here drives the material through the step.
///
/// UNLIKE `update_rest_shape` THIS TOUCHES NO HOST STATE. The rest shape has a
/// host reader (the tet layer reads `rest_excluded` off the live `DataSet`),
/// so that path writes the scene and then re-stages. Nothing here reads a
/// material on the host after `allocate()`: every consumer is a kernel reading
/// a staged buffer, and the collider's own tables are staged separately from a
/// different mesh. So the staged write IS the update, and adding a write to
/// the live `DataSet` beside it would create a second authority for the same
/// values with nothing keeping them in step.
///
/// # Safety
/// `update` must point at a live `MaterialParamUpdate` whose four `CVec`s
/// address their stated lengths.
#[no_mangle]
pub unsafe extern "C" fn update_material_params(update: *const MaterialParamUpdate) {
    if update.is_null() {
        fatal_exit(Fatal::invariant(
            "solver driver: update_material_params received a null MaterialParamUpdate",
        ));
    }
    let update = &*update;
    // The scene must be live for the same reason the other per-step entry
    // points require it: a table streamed before `initialize()` has no staged
    // buffer to land in and would be dropped without a word.
    let _ = require_scene("update_material_params");
    let faces = scene::slice(&update.face);
    let vertices = scene::slice(&update.vertex);
    let edges = scene::slice(&update.edge);
    let hinges = scene::slice(&update.hinge);
    if let Ok(mut guard) = SOLVER.lock() {
        if let Some(state) = guard.as_mut() {
            if let Err(fatal) = with_device(|device| {
                state.restage_material_params(device, faces, vertices, edges, hinges)
            }) {
                drop(guard);
                fatal_exit(fatal);
            }
        }
    }
}

/// Seed the implicit predictor with a commanded linear velocity.
///
/// # Safety
/// `indices` must address `count` elements.
#[no_mangle]
pub unsafe extern "C" fn override_velocity(
    indices: *const u32,
    count: u32,
    vx: f32,
    vy: f32,
    vz: f32,
    dt: f32,
) {
    let Some(indices) = borrow_indices("override_velocity", indices, count) else {
        return;
    };
    let view = require_scene("override_velocity");
    let Ok(mut guard) = SOLVER.lock() else {
        fatal_exit(Fatal::invariant(
            "solver driver: the solver state is poisoned, so an earlier failure went unreported",
        ));
    };
    let Some(state) = guard.as_mut() else {
        fatal_exit(Fatal::invariant(
            "solver driver: override_velocity was called before initialize() sized the state",
        ));
    };
    let state::SolverState { positions, positions_prev, seed_indices, .. } = state;
    if let Err(fatal) = with_device(|device| {
        seed::override_velocity(
            device, &view, positions, positions_prev, seed_indices, indices, vx, vy, vz, dt,
        )
    }) {
        drop(guard);
        fatal_exit(fatal);
    }
}

/// Pack the live world position of each listed vertex into `out`.
///
/// # Safety
/// `indices` must address `count` elements and `out` `3 * count` floats.
#[no_mangle]
pub unsafe extern "C" fn gather_current_positions(indices: *const u32, count: u32, out: *mut f32) {
    if out.is_null() {
        return;
    }
    let Some(indices) = borrow_indices("gather_current_positions", indices, count) else {
        return;
    };
    let view = require_scene("gather_current_positions");
    let out = std::slice::from_raw_parts_mut(out, 3 * count as usize);
    let Ok(mut guard) = SOLVER.lock() else {
        fatal_exit(Fatal::invariant(
            "solver driver: the solver state is poisoned, so an earlier failure went unreported",
        ));
    };
    let Some(state) = guard.as_mut() else {
        fatal_exit(Fatal::invariant(
            "solver driver: gather_current_positions was called before initialize() sized the state",
        ));
    };
    let state::SolverState { positions, seed_indices, seed_out, .. } = state;
    if let Err(fatal) = with_device(|device| {
        seed::gather_current_positions(
            device, &view, positions, seed_indices, seed_out, indices, out,
        )
    }) {
        drop(guard);
        fatal_exit(fatal);
    }
}

/// Add a rigid spin field on top of whatever the linear seed already wrote.
///
/// # Safety
/// `indices` must address `count` elements.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn override_angular_velocity(
    indices: *const u32,
    count: u32,
    wx: f32,
    wy: f32,
    wz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    dt: f32,
) {
    let Some(indices) = borrow_indices("override_angular_velocity", indices, count) else {
        return;
    };
    let view = require_scene("override_angular_velocity");
    let Ok(mut guard) = SOLVER.lock() else {
        fatal_exit(Fatal::invariant(
            "solver driver: the solver state is poisoned, so an earlier failure went unreported",
        ));
    };
    let Some(state) = guard.as_mut() else {
        fatal_exit(Fatal::invariant(
            "solver driver: override_angular_velocity was called before initialize() sized the state",
        ));
    };
    let state::SolverState { positions, positions_prev, seed_indices, .. } = state;
    if let Err(fatal) = with_device(|device| {
        seed::override_angular_velocity(
            device,
            &view,
            positions,
            positions_prev,
            seed_indices,
            indices,
            wx,
            wy,
            wz,
            cx,
            cy,
            cz,
            dt,
        )
    }) {
        drop(guard);
        fatal_exit(fatal);
    }
}

/// Install the scene's collision-window table.
///
/// # Safety
/// The window tables must address the counts they declare:
/// `vert_dmap` holds `vert_count` entries, `windows` holds
/// `n_groups * MAX_COLLISION_WINDOWS * 2` floats, and `window_counts` holds
/// `n_groups` entries.
#[no_mangle]
pub unsafe extern "C" fn init_collision_windows(
    vert_dmap: *const u32,
    vert_count: u32,
    windows: *const f32,
    window_counts: *const u32,
    n_groups: u32,
) {
    use ppf_cts_core::datamodel::object::MAX_COLLISION_WINDOWS;

    let view = require_scene("init_collision_windows");
    // The interval table and its counts are always required; the per-vertex
    // group list is required only when it has entries, which is Metal's rule and
    // is what lets a scene with no vertices through to the length check rather
    // than stopping on a pointer that has nothing to point at. The distinction
    // is load-bearing in Rust even though it is not in C++: `from_raw_parts` is
    // undefined on a null pointer at any length, zero included.
    if windows.is_null() || window_counts.is_null() || (vert_count > 0 && vert_dmap.is_null()) {
        fatal_exit(Fatal::invariant(
            "solver driver: init_collision_windows received a null table pointer",
        ));
    }
    let vertex_group = if vert_count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(vert_dmap, vert_count as usize)
    };
    let interval_bounds =
        std::slice::from_raw_parts(windows, n_groups as usize * MAX_COLLISION_WINDOWS * 2);
    let counts = std::slice::from_raw_parts(window_counts, n_groups as usize);

    // DEVICE FIRST, THEN THE TABLE, which is the order `advance` already takes:
    // `step::advance` runs inside `with_device` and takes the window lock from
    // there, so acquiring them the other way round here would be an inversion
    // rather than a new pair to order.
    let outcome = with_device(|device| {
        let Ok(mut table) = COLLISION_WINDOWS.lock() else {
            return Err(Fatal::invariant(
                "solver driver: the collision-window table is poisoned, so an earlier \
                 failure went unreported",
            ));
        };
        table.initialize(device, &view, vertex_group, interval_bounds, counts, n_groups)
    });
    if let Err(fatal) = outcome {
        fatal_exit(fatal);
    }
}

/// Recompute which vertices, faces and edges are collidable at `time`.
///
/// Called every step whether or not the scene authored any window; with none
/// installed it does nothing, because there is nothing to evaluate and every
/// element is collidable.
#[no_mangle]
pub extern "C" fn refresh_collision_active(time: f32) {
    let installed = COLLISION_WINDOWS
        .lock()
        .map(|table| table.is_initialized())
        .unwrap_or(false);
    if !installed {
        return;
    }
    let view = require_scene("refresh_collision_active");
    // DEVICE FIRST, THEN THE TABLE, which is the order `advance` already takes:
    // `step::advance` runs inside `with_device` and takes the window lock from
    // there, so acquiring them the other way round here would be an inversion
    // rather than a new pair to order.
    let outcome = with_device(|device| {
        let Ok(mut table) = COLLISION_WINDOWS.lock() else {
            return Err(Fatal::invariant(
                "solver driver: the collision-window table is poisoned, so an earlier \
                 failure went unreported",
            ));
        };
        // Safety: `require_scene` returned a view onto the live `DataSet`, and
        // no other reference to its mesh arrays is alive here.
        unsafe { table.refresh(device, &view, time) }
    });
    if let Err(fatal) = outcome {
        fatal_exit(fatal);
    }
}

/// Borrow an FFI index list, or `None` when there is nothing to do.
///
/// A null pointer with a non-zero count is a caller defect and stops the run;
/// a zero count is an empty keyframe and is not.
///
/// # Safety
/// `indices` must address `count` elements when both are non-trivial.
unsafe fn borrow_indices<'a>(who: &str, indices: *const u32, count: u32) -> Option<&'a [u32]> {
    if count == 0 {
        return None;
    }
    if indices.is_null() {
        fatal_exit(Fatal::invariant(format!(
            "solver driver: {who} was given {count} vertices through a null index list"
        )));
    }
    Some(std::slice::from_raw_parts(indices, count as usize))
}

/// The records the last intersection scan produced.
///
/// The host reads them through `fetch_intersection_records` after a step that
/// reported `intersection_free = false`, so they outlive the scan that found
/// them and are replaced whole by the next.
static INTERSECTION_RECORDS: Mutex<Vec<IntersectionRecord>> = Mutex::new(Vec::new());

/// Hand one scan's records to the host-facing buffer.
///
/// Called on EVERY scan, including a clean one: leaving the previous scan's
/// records in place would let a later failure report an earlier pose's pairs.
pub(crate) fn publish_intersection_records(report: &intersection::Report) {
    if let Ok(mut slot) = INTERSECTION_RECORDS.lock() {
        slot.clear();
        slot.extend_from_slice(&report.records);
    }
}

/// # Safety
/// `out` must address `max_count` records.
#[no_mangle]
pub unsafe extern "C" fn fetch_intersection_records(
    out: *mut IntersectionRecord,
    max_count: u32,
) -> u32 {
    if out.is_null() || max_count == 0 {
        return 0;
    }
    let Ok(records) = INTERSECTION_RECORDS.lock() else {
        return 0;
    };
    let count = records.len().min(max_count as usize);
    std::ptr::copy_nonoverlapping(records.as_ptr(), out, count);
    count as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatal_latch_keeps_the_first_writer() {
        // A fresh process would be cleaner, but the latch is process-global by
        // design, so the test asserts the property that matters: the second
        // writer does not overwrite the first.
        set_fatal(4, "first");
        set_fatal(5, "second");
        assert_eq!(fatal_code(), 4);
        let detail = unsafe { CStr::from_ptr(fatal_detail()) }
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(detail, "first");
    }

    #[test]
    fn empty_dyn_counts_are_reported_as_zero() {
        let mut n_value = 7u32;
        let mut n_offset = 9u32;
        unsafe { fetch_dyn_counts(&mut n_value, &mut n_offset) };
        assert_eq!((n_value, n_offset), (0, 0));
    }

    /// The shared bodies must be compiled for an ISA that HAS an FMA instruction.
    ///
    /// This guards a defect that was live and invisible: `build.rs` set
    /// `-ffp-contract=fast` and no `-march`, and baseline `x86-64` carries no FMA,
    /// so the contraction policy was honored and fused nothing. The bodies
    /// compiled as a non-contracting oracle while nvcc (`--fmad=true`) and the
    /// Metal shader compiler both contract, which shows up as a wrong TOLERANCE
    /// rather than a wrong answer and so survives every correctness test.
    ///
    /// Asserting the recorded baseline is the cheap half. The expensive half, and
    /// the one that actually proves fusion happened, is reading the built archive:
    /// `objdump -d $(find target/release/build -name libppf_kernels.a) | grep -c vfmadd`
    /// which measures 0 without an FMA-carrying baseline and thousands with one.
    /// Read it as zero against non-zero: the count scales with the body set, so
    /// a fixed number would go stale the next time a body is wired.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_shared_bodies_are_built_for_an_fma_baseline() {
        let built = env!("PPF_HOST_BASELINE_BUILT");
        assert_ne!(
            built, "none",
            "no ISA baseline was passed to the shared bodies on x86_64, so \
             -ffp-contract=fast has no FMA instruction to fuse into and this \
             backend is silently a non-contracting oracle"
        );
        // x86-64-v3 and later carry FMA; v1 and v2 do not. An explicit override
        // may legitimately name something else, so only the DEFAULT is asserted
        // to be FMA-carrying, and an override is allowed through on the operator's
        // judgement (build.rs documents what it costs).
        if std::env::var("PPF_HOST_BASELINE").is_err() {
            assert_eq!(
                built, "x86-64-v3",
                "the default baseline moved; if that was deliberate, confirm the \
                 new one carries FMA before changing this test"
            );
        }
    }
}
