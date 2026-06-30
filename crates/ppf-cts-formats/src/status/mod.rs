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
//! The solver's Rust host is the SOLE writer of the record and the
//! holder of the lock. The server's monitor is the reader. The C++
//! layer never touches the schema (it only widens `StepResult` so crash
//! sub-kinds reach the host). The Blender addon learns outcomes over the
//! socket, never by reading this file.
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
    /// Intersection at t=0 detected inside `initialize()`.
    InitIntersection,
    /// GPU / host out of memory, from the fatal-exit hook.
    Oom,
    /// CUDA runtime / driver abort, from the fatal-exit hook.
    CudaDriver,
    /// Rust host panic, from the panic hook.
    Panic,
    /// Synthesized by the SERVER ONLY: lock free + owning PID dead + no
    /// terminal outcome. Covers SIGKILL / OOM-kill and any fatal path
    /// the host hook could not catch. Also the forward-compat catch-all
    /// for a sub-kind an older build does not recognize.
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
            CrashKind::InitIntersection => "Intersection in the initial configuration",
            CrashKind::Oom => "Out of GPU or host memory",
            CrashKind::CudaDriver => "Unrecoverable CUDA runtime or driver error",
            CrashKind::Panic => "Solver host panicked",
            CrashKind::UnknownAbrupt => "Solver exited abnormally without reporting a cause",
        }
    }
}

/// Map the three `StepResult` success booleans to a [`CrashKind`]. The
/// solver host calls this on a failed advance, so the sub-kind is never
/// derived by parsing a log line. Priority matches the solver's own
/// reporting order (intersection, then CCD, then CG).
pub fn crash_kind_from_step(ccd_ok: bool, pcg_ok: bool, isect_free: bool) -> CrashKind {
    if !isect_free {
        CrashKind::Intersection
    } else if !ccd_ok {
        CrashKind::Ccd
    } else if !pcg_ok {
        CrashKind::Cg
    } else {
        // success()==true yet the host chose to fail: defensive, should
        // not happen, reported coarsely rather than silently dropped.
        CrashKind::UnknownAbrupt
    }
}

/// Fatal error codes set on the non-`StepResult` paths (init failure and
/// the C++ `exit(1)` fatal-exit hook), mapped to a [`CrashKind`]. The
/// numeric values are the contract between the C++ fatal hook and the
/// Rust host; keep them in sync with the C++ side.
pub mod error_code {
    /// No fatal code set (the StepResult booleans are authoritative).
    pub const NONE: u8 = 0;
    /// Intersection detected inside `initialize()`.
    pub const INIT_INTERSECTION: u8 = 1;
    /// `cudaErrorMemoryAllocation` / `cudaErrorOutOfMemory`.
    pub const OOM: u8 = 2;
    /// Any other `cudaError_t` from `CUDA_HANDLE_ERROR`.
    pub const CUDA_DRIVER: u8 = 3;
}

/// Map a fatal `error_code` (see [`error_code`]) to a [`CrashKind`].
/// `NONE` (and any unrecognized code) yields `None` so the caller falls
/// back to [`crash_kind_from_step`].
pub fn crash_kind_from_error_code(code: u8) -> Option<CrashKind> {
    match code {
        error_code::INIT_INTERSECTION => Some(CrashKind::InitIntersection),
        error_code::OOM => Some(CrashKind::Oom),
        error_code::CUDA_DRIVER => Some(CrashKind::CudaDriver),
        _ => None,
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
        // intersection > ccd > cg, matching the solver's report order.
        assert_eq!(
            crash_kind_from_step(false, false, false),
            CrashKind::Intersection
        );
        assert_eq!(crash_kind_from_step(false, true, true), CrashKind::Ccd);
        assert_eq!(crash_kind_from_step(true, false, true), CrashKind::Cg);
        assert_eq!(
            crash_kind_from_step(true, true, true),
            CrashKind::UnknownAbrupt
        );
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
        assert_eq!(crash_kind_from_error_code(error_code::NONE), None);
        assert_eq!(crash_kind_from_error_code(200), None);
    }
}
