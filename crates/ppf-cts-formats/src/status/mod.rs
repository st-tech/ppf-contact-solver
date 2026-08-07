// File: crates/ppf-cts-formats/src/status/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Structured solver run-status record: the single source of truth for
//! solver lifecycle and outcome, replacing free-form `### ...` stdout
//! strings parsed by substring tables and the overlapping
//! `finished.txt` / `crashed.txt` sentinel files.
//!
//! # Positive classification
//!
//! A healthy exit ALWAYS writes a terminal [`Outcome`]. Its ABSENCE
//! (confirmed against a freed liveness [`lock`] and a dead owning PID)
//! is therefore a crash by construction, with no log scraping. This is
//! the inversion that fixes the recurring "abrupt death misread as a
//! clean Resumable" bug: the old code asked "did I find proof of a
//! crash?"; this asks "did the run declare a clean outcome?".
//!
//! # Writers and readers
//!
//! The solver's Rust host is the sole writer of the record while its run
//! is alive, and the holder of the lock. The server's monitor is the
//! reader, and it writes exactly one thing: when a run dies without
//! reaching a terminal [`Outcome`], the monitor seals its own verdict into
//! the record after it has established that the owning process is gone
//! (lock free AND pid dead). [`write_terminal`] is first-writer-wins, so a
//! terminal outcome the solver did manage to write is never overwritten,
//! and sealing is what lets a later reconnect read back the same cause the
//! live report named. The C++ layer never touches the schema (it only
//! widens `StepResult` so crash sub-kinds reach the host). The Blender
//! addon learns outcomes over the socket, never by reading this file.
//!
//! # Versioning
//!
//! The record carries its own [`STATUS_VERSION`] via the envelope, kept
//! independent of the cross-language [`crate::SCHEMA_VERSION`] so the
//! status layout can evolve without invalidating `data.pickle` /
//! `param.pickle`. Forward compatibility within a version: unknown
//! [`CrashKind`] sub-kinds fold to [`CrashKind::UnknownAbrupt`] (a serde
//! catch-all), and an unrecognized terminal [`Outcome`] tag folds to
//! [`Outcome::Unknown`] via a custom `Deserialize` (internally-tagged
//! enums cannot use `#[serde(other)]` at the enum level).

pub mod lock;

use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize};

use crate::envelope::{from_cbor_with_version, to_cbor_with_version, FormatError};
use crate::files;

pub use lock::{pid_alive, Lock};

/// Envelope `kind` tag for the status record.
pub const KIND_RUN_STATUS: &str = "RunStatus";

/// Version of the [`RunStatus`] layout, independent of the shared
/// [`crate::SCHEMA_VERSION`]. Bump only when the record layout changes
/// incompatibly; a bump here never invalidates Scene / Param files.
pub const STATUS_VERSION: u32 = 1;

/// Lifecycle phase of a run. `Ended` is the only phase that carries a
/// terminal [`Outcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Starting,
    Initialized,
    Running,
    Saving,
    Ended,
}

/// Who asked an intentional terminate to happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TermSource {
    /// The addon / server requested the stop (`DoKillSolver`).
    AddonTerminate,
    /// An externally-launched run (e.g. Ctrl-C in a JupyterLab session)
    /// stopped cooperatively.
    External,
}

/// Cause of a crash. The first three come directly from the
/// `StepResult` booleans the host already has (no string parsing); the
/// rest come from a dedicated init/fatal error code or are synthesized
/// by the server.
///
/// New sub-kinds are additive: an older reader folds any unrecognized
/// tag to [`CrashKind::UnknownAbrupt`] via the serde catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrashKind {
    /// `!intersection_free` from a `StepResult`.
    Intersection,
    /// `!ccd_success` from a `StepResult`.
    Ccd,
    /// `!pcg_success` from a `StepResult`.
    Cg,
    /// `!newton_progress` from a `StepResult`: the Newton loop hit
    /// `max-newton-steps` without reaching an acceptable step. The usual
    /// cause is an over-constrained configuration, e.g. a prescribed pin
    /// driven into geometry that cannot yield: the contact line search
    /// clamps the shared time of impact toward zero to prevent the
    /// penetration, and that same clamp throttles everything else, so no
    /// iteration makes progress. Without this bound the loop spins
    /// forever (the time of impact stays above `FLT_EPSILON`, so the CCD
    /// trap never fires).
    NewtonStall,
    /// `!pin_feasible` from a `StepResult`: a prescribed (fix-pinned)
    /// vertex's swept path crosses an analytic collider (floor, sphere,
    /// wall). Such a vertex has no degrees of freedom to yield with, so
    /// the prescription itself is infeasible and the scene must be
    /// re-authored (or the pin made a soft pull pin).
    PinInfeasible,
    /// `!contact_separated` from a `StepResult`: a contact pair begins the
    /// timestep already inside the contact offset, i.e. two surfaces start
    /// out touching or overlapping. The conservative CCD advances from a
    /// separated start and cannot proceed from an overlapping one, so the
    /// scene must start with a small clearance. Usual causes are geometry
    /// authored in contact, a self-overlapping mesh, or a stitch / pin
    /// pulling elements together faster than contact can resolve.
    OverlappingStart,
    /// Intersection at t=0 detected inside `initialize()`.
    InitIntersection,
    /// GPU / host out of memory, from the fatal-exit hook.
    Oom,
    /// CUDA runtime / driver abort, from the fatal-exit hook.
    CudaDriver,
    /// `cudaErrorLaunchTimeout`: the operating system's kernel-execution
    /// watchdog reset the device out from under a running kernel. The
    /// runtime raises this only where such a timeout is configured, which
    /// is any GPU that also drives a display (WDDM on Windows, an X server
    /// on Linux). It names a property of the machine, not of the driver, so
    /// it is separated from [`CrashKind::CudaDriver`]: the action is to run
    /// on a GPU with no display attached (or raise the OS timeout), not to
    /// reinstall a driver. The detail reports whether the watchdog was
    /// actually armed on the device, as read from `cudaDeviceProp` before
    /// the run started.
    WatchdogTimeout,
    /// Rust host panic, from the panic hook.
    Panic,
    /// A solver internal invariant failed and the run was stopped at the
    /// detection site (a `PPF FATAL:` report). These are deliberate
    /// unconditional checks over assembled state (a non-SPD Newton matrix,
    /// an infeasible lock, a position off the representable domain), so
    /// the detail carries the check's own report and the run cannot be
    /// resumed past it without changing the scene or fixing the defect.
    SolverInvariant,
    /// A device-side `assert` trapped, surfacing as `cudaErrorAssert` on
    /// the next checked CUDA call. The asserts are live in the release
    /// build on purpose (the penetration-free family among them), so this
    /// names a violated GPU invariant, not a build misconfiguration. The
    /// device `printf` that would name the assert is lost on the trap, so
    /// the stdout tail is the only place its neighborhood shows.
    DeviceAssert,
    /// The process was killed by a fatal signal (or, on Windows, a
    /// structured exception) before it could write a terminal record. The
    /// detail names the signal when it is known; a `SIGKILL` leaves no
    /// signal record at all and is inferred from the launcher's exit
    /// status.
    KilledBySignal,
    /// The dynamic loader could not resolve a library the solver needs, so
    /// the image it was asked to start (or to extend, for a delay-loaded
    /// module) never ran. Named separately from [`CrashKind::LaunchFailed`]
    /// because it says WHICH part failed and what to check: the library
    /// search path, not the run script or the scene.
    LibraryLoadFailed,
    /// The process exited before `status_writer::init` ran, so no record was
    /// ever written. That is all the absence of a record establishes; what
    /// stopped it is named from the launcher's exit status when that status
    /// names anything, and left unnamed when it does not. The stderr tail is
    /// the evidence either way.
    LaunchFailed,
    /// Synthesized by the SERVER ONLY: lock free + owning PID dead + no
    /// terminal outcome, with no evidence that names a cause. Also the
    /// forward-compat catch-all for a sub-kind an older build does not
    /// recognize. Reported WITH the raw facts rather than a guess.
    #[serde(other)]
    UnknownAbrupt,
}

impl CrashKind {
    /// One human-readable summary per kind. This is the single
    /// replacement for the two divergent `ERROR_PATTERNS` message tables
    /// (`monitor.rs` and `core/datamodel/session/log.rs`).
    pub fn summary(self) -> &'static str {
        match self {
            CrashKind::Intersection => "Intersection detected",
            CrashKind::Ccd => "Continuous collision detection failed",
            CrashKind::Cg => "Linear solver failed to converge",
            CrashKind::NewtonStall => {
                "Newton solve made no progress (over-constrained configuration)"
            }
            CrashKind::PinInfeasible => "A pinned vertex is driven into a collider it cannot yield to",
            CrashKind::OverlappingStart => {
                "Two surfaces start the step already touching or overlapping"
            }
            CrashKind::InitIntersection => "Intersection in the initial configuration",
            CrashKind::Oom => "Out of GPU memory",
            CrashKind::CudaDriver => "Unrecoverable CUDA runtime or driver error",
            CrashKind::WatchdogTimeout => {
                "A GPU kernel ran past the operating system's watchdog timeout"
            }
            CrashKind::Panic => "Solver host panicked",
            CrashKind::SolverInvariant => "Solver stopped on a failed internal check",
            CrashKind::DeviceAssert => "A solver invariant failed on the GPU",
            CrashKind::KilledBySignal => "The solver process was killed before it could report",
            CrashKind::LibraryLoadFailed => "A required library could not be loaded",
            CrashKind::LaunchFailed => "The solver exited before it started",
            CrashKind::UnknownAbrupt => "Solver exited abnormally without reporting a cause",
        }
    }

    /// The exact `snake_case` spelling serde writes for this variant.
    ///
    /// One spelling with three consumers: the CBOR record's `sub_kind`, the
    /// `crash_kind` field on the status response, and the addon's
    /// translation-key selector. `crash_kind_tag_matches_serde` compares
    /// this against what serde actually emits, so the three cannot drift.
    pub fn tag(self) -> &'static str {
        match self {
            CrashKind::Intersection => "intersection",
            CrashKind::Ccd => "ccd",
            CrashKind::Cg => "cg",
            CrashKind::NewtonStall => "newton_stall",
            CrashKind::PinInfeasible => "pin_infeasible",
            CrashKind::OverlappingStart => "overlapping_start",
            CrashKind::InitIntersection => "init_intersection",
            CrashKind::Oom => "oom",
            CrashKind::CudaDriver => "cuda_driver",
            CrashKind::WatchdogTimeout => "watchdog_timeout",
            CrashKind::Panic => "panic",
            CrashKind::SolverInvariant => "solver_invariant",
            CrashKind::DeviceAssert => "device_assert",
            CrashKind::KilledBySignal => "killed_by_signal",
            CrashKind::LibraryLoadFailed => "library_load_failed",
            CrashKind::LaunchFailed => "launch_failed",
            CrashKind::UnknownAbrupt => "unknown_abrupt",
        }
    }

    /// Every variant, in declaration order. Lets a test (and the addon's
    /// i18n gate, through the rig) enumerate the tag set exhaustively.
    pub const ALL: &'static [CrashKind] = &[
        CrashKind::Intersection,
        CrashKind::Ccd,
        CrashKind::Cg,
        CrashKind::NewtonStall,
        CrashKind::PinInfeasible,
        CrashKind::OverlappingStart,
        CrashKind::InitIntersection,
        CrashKind::Oom,
        CrashKind::CudaDriver,
        CrashKind::WatchdogTimeout,
        CrashKind::Panic,
        CrashKind::SolverInvariant,
        CrashKind::DeviceAssert,
        CrashKind::KilledBySignal,
        CrashKind::LibraryLoadFailed,
        CrashKind::LaunchFailed,
        CrashKind::UnknownAbrupt,
    ];
}

/// Map the `StepResult` success booleans to a [`CrashKind`]. The solver
/// host calls this on a failed advance, so the sub-kind is never derived
/// by parsing a log line.
///
/// Priority runs from the most specific diagnosis to the least. An
/// infeasible pin and an overlapping start are root causes, so they outrank
/// the symptoms they would otherwise surface as (a collapsed time of impact
/// reads as a CCD failure or a Newton stall). A Newton stall ranks last: it
/// is the generic "no iteration made progress" outcome, and any more precise
/// boolean explains it better.
pub fn crash_kind_from_step(
    ccd_ok: bool,
    pcg_ok: bool,
    isect_free: bool,
    newton_progress: bool,
    pin_feasible: bool,
    contact_separated: bool,
) -> CrashKind {
    if !pin_feasible {
        CrashKind::PinInfeasible
    } else if !contact_separated {
        CrashKind::OverlappingStart
    } else if !isect_free {
        CrashKind::Intersection
    } else if !ccd_ok {
        CrashKind::Ccd
    } else if !pcg_ok {
        CrashKind::Cg
    } else if !newton_progress {
        CrashKind::NewtonStall
    } else {
        // success()==true yet the host chose to fail: defensive, should
        // not happen, reported coarsely rather than silently dropped.
        CrashKind::UnknownAbrupt
    }
}

/// Fatal error codes set on the non-`StepResult` paths (init failure and
/// the C++ `exit(1)` fatal-exit hook), mapped to a [`CrashKind`]. The
/// numeric values are the contract between the C++ fatal hook
/// (`cpp/main/fatal.hpp`, whose `PPF_FATAL_*` enumerators carry the same
/// numbers) and the Rust host; the two must stay in sync.
pub mod error_code {
    /// No fatal code set (the StepResult booleans are authoritative).
    pub const NONE: u8 = 0;
    /// Intersection detected inside `initialize()`.
    pub const INIT_INTERSECTION: u8 = 1;
    /// `cudaErrorMemoryAllocation` / `cudaErrorOutOfMemory`.
    pub const OOM: u8 = 2;
    /// Any other `cudaError_t` from `CUDA_HANDLE_ERROR`.
    pub const CUDA_DRIVER: u8 = 3;
    /// A `PPF FATAL:` invariant check stopped the run at its detection site.
    pub const SOLVER_INVARIANT: u8 = 4;
    /// `cudaErrorAssert`: a device-side `assert` trapped.
    pub const DEVICE_ASSERT: u8 = 5;
    /// `cudaErrorLaunchTimeout`: the OS kernel-execution watchdog fired.
    pub const WATCHDOG_TIMEOUT: u8 = 6;
}

/// Map a fatal `error_code` (see [`error_code`]) to a [`CrashKind`].
/// `NONE` (and any unrecognized code) yields `None` so the caller falls
/// back to [`crash_kind_from_step`].
pub fn crash_kind_from_error_code(code: u8) -> Option<CrashKind> {
    match code {
        error_code::INIT_INTERSECTION => Some(CrashKind::InitIntersection),
        error_code::OOM => Some(CrashKind::Oom),
        error_code::CUDA_DRIVER => Some(CrashKind::CudaDriver),
        error_code::SOLVER_INVARIANT => Some(CrashKind::SolverInvariant),
        error_code::DEVICE_ASSERT => Some(CrashKind::DeviceAssert),
        error_code::WATCHDOG_TIMEOUT => Some(CrashKind::WatchdogTimeout),
        _ => None,
    }
}

/// The fatal signals the solver installs a handler for, as
/// `(number, name)`. The solver writes the name into the
/// [`signal_sidecar`] from inside the handler and re-raises under the
/// previous disposition, so the process still dies with the right status.
///
/// `SIGKILL` is deliberately ABSENT: it cannot be caught, so no handler
/// can record it. That asymmetry against [`signal_name`] is what makes the
/// supervisor's fallback an elimination rather than a guess. When a
/// launcher reports `128 + N` for a signal that IS in this table, the
/// sidecar would have named it, so its absence means the handler never
/// ran; when `N` is one of the uncatchable signals, no handler could have
/// run at all. Either way the exit status is the only witness left, and
/// the report says exactly that instead of naming a cause.
///
/// Every number comes from the TARGET's own `libc`, never from a literal.
/// The six do not agree across platforms: `SIGBUS` is 7 on Linux and 10 on
/// Darwin, and 10 on Linux is `SIGUSR1`, so a literal table written for
/// either one installs the handler on one signal and labels it another on
/// the other.
#[cfg(unix)]
pub const HANDLED_SIGNALS: &[(i32, &str)] = &[
    (libc::SIGILL, "SIGILL"),
    (libc::SIGABRT, "SIGABRT"),
    (libc::SIGFPE, "SIGFPE"),
    (libc::SIGBUS, "SIGBUS"),
    (libc::SIGSEGV, "SIGSEGV"),
    (libc::SIGTERM, "SIGTERM"),
];

/// Empty off unix: no handler is installed there and no sidecar is ever
/// written, so there is no number to name.
#[cfg(not(unix))]
pub const HANDLED_SIGNALS: &[(i32, &str)] = &[];

/// The names a sidecar can carry, independent of any platform's numbering.
///
/// A number means something only on the platform whose `libc` defines it,
/// which is why [`HANDLED_SIGNALS`] is built from the target's own values
/// and is empty off unix. A NAME is just the text the handler wrote, so
/// recognizing one is not a per-platform question: the sidecar reader
/// validates the string it read, and a reader that could not do that off
/// unix would reject a record that is perfectly well formed.
pub const HANDLED_SIGNAL_NAMES: &[&str] = &[
    "SIGILL", "SIGABRT", "SIGFPE", "SIGBUS", "SIGSEGV", "SIGTERM",
];

/// The two tables name the same six signals. Checked here rather than
/// trusted, so adding a signal to one and not the other does not compile.
#[cfg(unix)]
const _: () = {
    assert!(HANDLED_SIGNALS.len() == HANDLED_SIGNAL_NAMES.len());
};

/// Name a fatal signal number, covering [`HANDLED_SIGNALS`] plus the
/// uncatchable `SIGKILL`. Returns `None` for anything else, which the
/// supervisor reports as unknown with the raw number rather than guessing.
///
/// The numbers are the target's own `libc` values, so a name is only ever
/// returned for the signal that actually carries it on the platform this
/// build runs on. Off unix the table is empty and this returns `None` for
/// everything, which is what stops the supervisor's `128 + N` decode (a
/// POSIX shell convention) from reading a Windows exit code as a signal.
pub fn signal_name(n: i32) -> Option<&'static str> {
    #[cfg(unix)]
    if n == libc::SIGKILL {
        return Some("SIGKILL");
    }
    HANDLED_SIGNALS
        .iter()
        .find(|(num, _)| *num == n)
        .map(|(_, name)| *name)
}

/// The one-line signal sidecar (`<output>/crash_signal`).
///
/// A signal handler cannot serialize CBOR, allocate, or lock, so a signal
/// death cannot write the terminal status record. It writes this instead:
/// a single pre-formatted `"<SIGNAME> <launch_id>\n"` through one
/// `write(2)`. The WRITER lives in the solver crate, beside the handler
/// whose async-signal-safety audit constrains it; only the reader and the
/// scrub live here, where the supervisor can reach them.
pub mod signal_sidecar {
    use std::path::Path;

    use crate::files;

    /// Read the recorded signal name, but only when the record belongs to
    /// `launch_id`.
    ///
    /// The launch id is what makes the read safe. The file is also scrubbed
    /// at `status_writer::init`, but a resume skips the output-directory
    /// wipe, so a prior run's sidecar can outlive its run in the window
    /// before the scrub; matching the id is the check that does not depend
    /// on when the reader looks.
    ///
    /// Returns the name only when it is one of [`super::HANDLED_SIGNALS`],
    /// so a corrupt or foreign file cannot inject arbitrary text into a
    /// user-visible message.
    pub fn read(output_dir: &Path, launch_id: &str) -> Option<&'static str> {
        let body = std::fs::read_to_string(output_dir.join(files::CRASH_SIGNAL)).ok()?;
        let mut parts = body.split_whitespace();
        let name = parts.next()?;
        if parts.next()? != launch_id {
            return None;
        }
        // Validated against the NAMES rather than the number table: the
        // sidecar carries text, and the number table is empty off unix,
        // which would otherwise make every well-formed record unreadable
        // there.
        super::HANDLED_SIGNAL_NAMES
            .iter()
            .find(|known| **known == name)
            .copied()
    }

    /// Remove a stale sidecar. Called once at solver startup, for the same
    /// reason the status record is removed there.
    pub fn scrub(output_dir: &Path) {
        let _ = std::fs::remove_file(output_dir.join(files::CRASH_SIGNAL));
    }
}

/// Terminal outcome of a run, present iff `phase == Ended`.
///
/// Internally tagged on `kind`. Serialization is derived; deserialization
/// is hand-written so an unrecognized `kind` (a newer terminal variant
/// read by an older build, within the same [`STATUS_VERSION`]) folds to
/// [`Outcome::Unknown`] instead of erroring, which serde cannot express
/// with `#[serde(other)]` on an internally-tagged enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome {
    /// Frames-done clean completion.
    Finished,
    /// Intentional checkpoint exit, distinct from `Finished`.
    SavedAndQuit,
    /// Cooperative / intentional stop.
    Terminated { source: TermSource },
    /// Genuine failure.
    Crashed { sub_kind: CrashKind, detail: String },
    /// Forward-compat sink: a terminal record whose `kind` this build
    /// does not recognize. Never WRITTEN by us (only produced on read);
    /// treated as a clean-but-opaque terminal stop, never a live run.
    Unknown { raw_kind: String },
}

impl<'de> Deserialize<'de> for Outcome {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Capture the tag plus every possible field with defaults, then
        // route. Unknown fields are ignored; an unknown `kind` routes to
        // `Unknown` rather than failing the whole record.
        #[derive(Deserialize)]
        struct Raw {
            kind: String,
            #[serde(default)]
            source: Option<TermSource>,
            #[serde(default)]
            sub_kind: Option<CrashKind>,
            #[serde(default)]
            detail: Option<String>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Ok(match raw.kind.as_str() {
            "finished" => Outcome::Finished,
            "saved_and_quit" => Outcome::SavedAndQuit,
            "terminated" => Outcome::Terminated {
                source: raw.source.unwrap_or(TermSource::External),
            },
            "crashed" => Outcome::Crashed {
                sub_kind: raw.sub_kind.unwrap_or(CrashKind::UnknownAbrupt),
                detail: raw.detail.unwrap_or_default(),
            },
            other => Outcome::Unknown {
                raw_kind: other.to_string(),
            },
        })
    }
}

/// The full run-status record. One ~120-byte CBOR blob under
/// `<output>/status.cbor`, wrapped in `Envelope<RunStatus>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunStatus {
    pub phase: Phase,
    /// Last emitted frame (mirrors `vert_<N>`); the progress source.
    pub frame: i32,
    pub sim_time: f64,
    /// A `state_<N>.bin.gz` checkpoint exists / was just written.
    pub resumable: bool,
    /// `Some` iff `phase == Ended`.
    #[serde(default)]
    pub outcome: Option<Outcome>,
    /// Monotonic write counter; a reader prefers the highest `seq` and
    /// can reject a stale or torn record.
    pub seq: u64,
    /// Owning host PID: the ONLY liveness cross-check (never the global
    /// process-name scan, which a second unrelated solver would trip).
    pub pid: u32,
    /// 12-hex identity stamped at launch; lets a reader reject a stale
    /// record left by a prior run in the same directory.
    pub launch_id: String,
    pub emulated: bool,
}

impl RunStatus {
    /// True once the run has reached a terminal outcome.
    pub fn is_terminal(&self) -> bool {
        self.outcome.is_some()
    }
}

/// Write a non-terminal progress record IN PLACE (truncate + write, no
/// fsync). Cheap enough to call per emitted frame. A crash mid-write
/// leaves a torn file, which [`read`] surfaces as [`FormatError::CborDe`]
/// and the server treats as "no terminal outcome" (i.e. the crux), so
/// the lack of durability here is safe by design.
pub fn write_progress(output_dir: &Path, status: &RunStatus) -> Result<(), FormatError> {
    let bytes = to_cbor_with_version(STATUS_VERSION, KIND_RUN_STATUS, status)?;
    let path = output_dir.join(files::STATUS_RECORD);
    std::fs::write(&path, &bytes).map_err(|e| FormatError::CborSer(format!("status write: {e}")))
}

/// Write a TERMINAL outcome durably (tmp + fsync + atomic rename) and
/// idempotently: if a terminal record already exists on disk this is a
/// no-op, so the per-path terminal write plus the panic / SIGTERM /
/// atexit hooks can all call it and the first writer wins.
///
/// `status.outcome` must be `Some`; a debug assertion guards misuse.
pub fn write_terminal(output_dir: &Path, status: &RunStatus) -> Result<(), FormatError> {
    debug_assert!(
        status.outcome.is_some(),
        "write_terminal called with a non-terminal RunStatus"
    );
    // First writer wins: never clobber an existing terminal outcome.
    if let Ok(Some(existing)) = read(output_dir) {
        if existing.is_terminal() {
            return Ok(());
        }
    }
    let bytes = to_cbor_with_version(STATUS_VERSION, KIND_RUN_STATUS, status)?;
    let path = output_dir.join(files::STATUS_RECORD);
    let tmp = output_dir.join(format!("{}.tmp", files::STATUS_RECORD));
    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| FormatError::CborSer(format!("status tmp create: {e}")))?;
        f.write_all(&bytes)
            .map_err(|e| FormatError::CborSer(format!("status tmp write: {e}")))?;
        f.sync_all()
            .map_err(|e| FormatError::CborSer(format!("status tmp fsync: {e}")))?;
    }
    std::fs::rename(&tmp, &path).map_err(|e| FormatError::CborSer(format!("status rename: {e}")))
}

/// Read the current record.
///
/// - `Ok(None)`            : the record is absent (no run yet / scrubbed).
/// - `Err(VersionMismatch)`: a newer [`STATUS_VERSION`] than this build.
/// - `Err(CborDe)`         : a zero-length or torn file (an interrupted
///   in-place write). The server routes this, on a confirmed-dead owning
///   PID, to the same `UnknownAbrupt` crux verdict as a non-terminal
///   record, never to a silent clean state.
pub fn read(output_dir: &Path) -> Result<Option<RunStatus>, FormatError> {
    let path = output_dir.join(files::STATUS_RECORD);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(FormatError::CborDe(format!("status read: {e}"))),
    };
    if bytes.is_empty() {
        return Err(FormatError::CborDe("status.cbor is empty (torn write)".into()));
    }
    let status: RunStatus = from_cbor_with_version(STATUS_VERSION, KIND_RUN_STATUS, &bytes)?;
    Ok(Some(status))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> RunStatus {
        RunStatus {
            phase: Phase::Running,
            frame: 42,
            sim_time: 0.7,
            resumable: true,
            outcome: None,
            seq: 43,
            pid: 81231,
            launch_id: "a1b2c3d4e5f6".into(),
            emulated: false,
        }
    }

    #[test]
    fn progress_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let s = base();
        write_progress(dir.path(), &s).unwrap();
        let back = read(dir.path()).unwrap().unwrap();
        assert_eq!(s, back);
        assert!(!back.is_terminal());
    }

    #[test]
    fn read_absent_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read(dir.path()).unwrap().is_none());
    }

    #[test]
    fn terminal_roundtrip_each_outcome() {
        for outcome in [
            Outcome::Finished,
            Outcome::SavedAndQuit,
            Outcome::Terminated {
                source: TermSource::AddonTerminate,
            },
            Outcome::Crashed {
                sub_kind: CrashKind::Ccd,
                detail: "ccd failed (toi: 4.20e-09) at frame 17".into(),
            },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut s = base();
            s.phase = Phase::Ended;
            s.outcome = Some(outcome.clone());
            write_terminal(dir.path(), &s).unwrap();
            let back = read(dir.path()).unwrap().unwrap();
            assert_eq!(back.outcome.as_ref(), Some(&outcome));
            assert!(back.is_terminal());
        }
    }

    #[test]
    fn terminal_write_is_idempotent_first_writer_wins() {
        let dir = tempfile::tempdir().unwrap();
        let mut first = base();
        first.phase = Phase::Ended;
        first.outcome = Some(Outcome::Crashed {
            sub_kind: CrashKind::Cg,
            detail: "first".into(),
        });
        write_terminal(dir.path(), &first).unwrap();
        // A later hook (panic / SIGTERM / atexit) must not clobber it.
        let mut second = base();
        second.phase = Phase::Ended;
        second.outcome = Some(Outcome::Terminated {
            source: TermSource::AddonTerminate,
        });
        write_terminal(dir.path(), &second).unwrap();
        assert_eq!(read(dir.path()).unwrap().unwrap().outcome, first.outcome);
    }

    #[test]
    fn empty_file_is_cbor_error_not_silent_none() {
        // The crux: a torn / zero-length record must not read as "clean".
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(files::STATUS_RECORD), b"").unwrap();
        assert!(matches!(read(dir.path()), Err(FormatError::CborDe(_))));
    }

    #[test]
    fn torn_file_is_cbor_error() {
        let dir = tempfile::tempdir().unwrap();
        // Truncate a valid record to simulate an interrupted write.
        write_progress(dir.path(), &base()).unwrap();
        let p = dir.path().join(files::STATUS_RECORD);
        let full = std::fs::read(&p).unwrap();
        std::fs::write(&p, &full[..full.len() / 2]).unwrap();
        assert!(matches!(read(dir.path()), Err(FormatError::CborDe(_))));
    }

    #[test]
    fn newer_status_version_is_rejected() {
        // A record written under a higher STATUS_VERSION must be refused,
        // not silently mis-parsed, and must NOT touch the shared
        // SCHEMA_VERSION used by Scene / Param.
        let dir = tempfile::tempdir().unwrap();
        let bytes = to_cbor_with_version(STATUS_VERSION + 1, KIND_RUN_STATUS, &base()).unwrap();
        std::fs::write(dir.path().join(files::STATUS_RECORD), &bytes).unwrap();
        assert!(matches!(
            read(dir.path()),
            Err(FormatError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn unknown_outcome_kind_folds_to_unknown_sink() {
        // Forward compat: a future terminal variant read by this build
        // becomes Outcome::Unknown, treated as a terminal stop, never a
        // live run, never an error.
        #[derive(Serialize)]
        struct FutureStatus {
            phase: Phase,
            frame: i32,
            sim_time: f64,
            resumable: bool,
            outcome: FutureOutcome,
            seq: u64,
            pid: u32,
            launch_id: String,
            emulated: bool,
        }
        #[derive(Serialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum FutureOutcome {
            Suspended { reason: String },
        }
        let dir = tempfile::tempdir().unwrap();
        let future = FutureStatus {
            phase: Phase::Ended,
            frame: 9,
            sim_time: 0.1,
            resumable: true,
            outcome: FutureOutcome::Suspended {
                reason: "hibernate".into(),
            },
            seq: 10,
            pid: 1,
            launch_id: "ffffffffffff".into(),
            emulated: true,
        };
        let bytes = to_cbor_with_version(STATUS_VERSION, KIND_RUN_STATUS, &future).unwrap();
        std::fs::write(dir.path().join(files::STATUS_RECORD), &bytes).unwrap();
        let back = read(dir.path()).unwrap().unwrap();
        assert_eq!(
            back.outcome,
            Some(Outcome::Unknown {
                raw_kind: "suspended".into()
            })
        );
        assert!(back.is_terminal());
    }

    #[test]
    fn unknown_crash_subkind_folds_to_unknown_abrupt() {
        // A future CrashKind sub-kind an older build does not know folds
        // to UnknownAbrupt (the serde catch-all), still a crash.
        let dir = tempfile::tempdir().unwrap();
        #[derive(Serialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum FutureOutcome {
            Crashed { sub_kind: String, detail: String },
        }
        #[derive(Serialize)]
        struct S {
            phase: Phase,
            frame: i32,
            sim_time: f64,
            resumable: bool,
            outcome: FutureOutcome,
            seq: u64,
            pid: u32,
            launch_id: String,
            emulated: bool,
        }
        let s = S {
            phase: Phase::Ended,
            frame: 3,
            sim_time: 0.0,
            resumable: false,
            outcome: FutureOutcome::Crashed {
                sub_kind: "thermal_throttle".into(),
                detail: "GPU too hot".into(),
            },
            seq: 4,
            pid: 1,
            launch_id: "ffffffffffff".into(),
            emulated: true,
        };
        let bytes = to_cbor_with_version(STATUS_VERSION, KIND_RUN_STATUS, &s).unwrap();
        std::fs::write(dir.path().join(files::STATUS_RECORD), &bytes).unwrap();
        let back = read(dir.path()).unwrap().unwrap();
        assert_eq!(
            back.outcome,
            Some(Outcome::Crashed {
                sub_kind: CrashKind::UnknownAbrupt,
                detail: "GPU too hot".into()
            })
        );
    }

    #[test]
    fn crash_kind_from_step_priority() {
        // Most specific diagnosis first: an infeasible pin is a root cause, so it
        // outranks the symptoms it would otherwise surface as (a collapsed time of
        // impact reads as a CCD failure or a Newton stall). A Newton stall ranks
        // last: any more precise boolean explains it better.
        // pin_infeasible > overlapping_start > intersection > ccd > cg > newton_stall.
        let ok = |ccd, pcg, isect, newton, pin, sep| {
            crash_kind_from_step(ccd, pcg, isect, newton, pin, sep)
        };
        // Everything failed at once: the pin diagnosis wins.
        assert_eq!(ok(false, false, false, false, false, false), CrashKind::PinInfeasible);
        // Pin feasible, contacts overlapping: the overlap outranks the rest.
        assert_eq!(ok(false, false, false, false, true, false), CrashKind::OverlappingStart);
        // Pin feasible, contacts separated, everything else failed: intersection wins.
        assert_eq!(ok(false, false, false, false, true, true), CrashKind::Intersection);
        assert_eq!(ok(false, true, true, true, true, true), CrashKind::Ccd);
        assert_eq!(ok(true, false, true, true, true, true), CrashKind::Cg);
        // Only the Newton bound tripped: nothing more precise to report.
        assert_eq!(ok(true, true, true, false, true, true), CrashKind::NewtonStall);
        // Nothing failed, yet the host chose to fail: reported coarsely.
        assert_eq!(ok(true, true, true, true, true, true), CrashKind::UnknownAbrupt);
    }

    #[test]
    fn crash_kind_from_error_code_mapping() {
        assert_eq!(
            crash_kind_from_error_code(error_code::INIT_INTERSECTION),
            Some(CrashKind::InitIntersection)
        );
        assert_eq!(
            crash_kind_from_error_code(error_code::OOM),
            Some(CrashKind::Oom)
        );
        assert_eq!(
            crash_kind_from_error_code(error_code::CUDA_DRIVER),
            Some(CrashKind::CudaDriver)
        );
        assert_eq!(
            crash_kind_from_error_code(error_code::SOLVER_INVARIANT),
            Some(CrashKind::SolverInvariant)
        );
        assert_eq!(
            crash_kind_from_error_code(error_code::DEVICE_ASSERT),
            Some(CrashKind::DeviceAssert)
        );
        assert_eq!(
            crash_kind_from_error_code(error_code::WATCHDOG_TIMEOUT),
            Some(CrashKind::WatchdogTimeout)
        );
        assert_eq!(crash_kind_from_error_code(error_code::NONE), None);
        assert_eq!(crash_kind_from_error_code(200), None);
    }

    #[test]
    fn new_crash_kinds_roundtrip() {
        for kind in [
            CrashKind::SolverInvariant,
            CrashKind::DeviceAssert,
            CrashKind::KilledBySignal,
            CrashKind::LibraryLoadFailed,
            CrashKind::LaunchFailed,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut s = base();
            s.phase = Phase::Ended;
            s.outcome = Some(Outcome::Crashed {
                sub_kind: kind,
                detail: "detail".into(),
            });
            write_terminal(dir.path(), &s).unwrap();
            let back = read(dir.path()).unwrap().unwrap();
            assert_eq!(
                back.outcome,
                Some(Outcome::Crashed {
                    sub_kind: kind,
                    detail: "detail".into()
                })
            );
        }
    }

    #[test]
    fn crash_kind_tag_matches_serde() {
        // tag() is the wire spelling AND the addon's translation-key
        // selector, so it must be exactly what serde writes. Compare
        // against the serializer rather than a second hand-written list.
        for kind in CrashKind::ALL {
            let serialized = serde_json::to_value(kind).unwrap();
            assert_eq!(serialized, serde_json::Value::String(kind.tag().into()));
        }
    }

    #[cfg(unix)]
    #[test]
    fn signal_table_excludes_sigkill() {
        // The elimination argument the supervisor's fallback rests on: a
        // handler exists for every signal it CAN catch, and SIGKILL is not
        // one of them, so a bare `128 + 9` launcher code with no sidecar is
        // an uncatchable kill rather than a missing handler.
        assert_eq!(signal_name(libc::SIGKILL), Some("SIGKILL"));
        assert!(!HANDLED_SIGNALS
            .iter()
            .any(|(n, _)| *n == libc::SIGKILL));
        for (n, name) in HANDLED_SIGNALS {
            assert_eq!(signal_name(*n), Some(*name));
        }
    }

    #[cfg(unix)]
    #[test]
    fn signal_numbers_are_the_target_libc_values() {
        // Comparing names against `libc` is the only form of this check that
        // cannot pass on one OS while lying on the other. A literal table is
        // right for exactly one platform: `SIGBUS` is 7 on Linux and 10 on
        // Darwin, and 10 on Linux is `SIGUSR1`, so a Darwin-shaped literal
        // arms the handler on `SIGUSR1`, labels it `SIGBUS`, and leaves the
        // real `SIGBUS` uncovered.
        assert_eq!(signal_name(libc::SIGILL), Some("SIGILL"));
        assert_eq!(signal_name(libc::SIGABRT), Some("SIGABRT"));
        assert_eq!(signal_name(libc::SIGFPE), Some("SIGFPE"));
        assert_eq!(signal_name(libc::SIGBUS), Some("SIGBUS"));
        assert_eq!(signal_name(libc::SIGSEGV), Some("SIGSEGV"));
        assert_eq!(signal_name(libc::SIGTERM), Some("SIGTERM"));
        // Signals the solver never handles must stay unnamed, or the
        // supervisor reports a hardware cause for a routine notification.
        assert_eq!(signal_name(libc::SIGUSR1), None);
        assert_eq!(signal_name(libc::SIGUSR2), None);
        assert_eq!(signal_name(libc::SIGINT), None);
    }

    #[test]
    fn sidecar_rejects_foreign_launch_id() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(files::CRASH_SIGNAL),
            "SIGSEGV a1b2c3d4e5f6\n",
        )
        .unwrap();
        assert_eq!(
            signal_sidecar::read(dir.path(), "a1b2c3d4e5f6"),
            Some("SIGSEGV")
        );
        // A sidecar left by a different launch is not this run's evidence.
        assert_eq!(signal_sidecar::read(dir.path(), "ffffffffffff"), None);
        // An unrecognized name never reaches a user-visible message.
        std::fs::write(
            dir.path().join(files::CRASH_SIGNAL),
            "NOT_A_SIGNAL a1b2c3d4e5f6\n",
        )
        .unwrap();
        assert_eq!(signal_sidecar::read(dir.path(), "a1b2c3d4e5f6"), None);
        signal_sidecar::scrub(dir.path());
        assert_eq!(signal_sidecar::read(dir.path(), "a1b2c3d4e5f6"), None);
    }
}
