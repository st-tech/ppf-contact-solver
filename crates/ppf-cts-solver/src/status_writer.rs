// File: status_writer.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Process-global writer for the structured solver status record
//! (`ppf_cts_formats::status`). The solver host is the sole writer of
//! `status.cbor` and the holder of the liveness lock.
//!
//! A healthy run always stamps a terminal [`Outcome`]; its absence
//! (with a freed lock and a dead owning PID) is the server's
//! crash-by-absence verdict. So the contract here is: write a terminal
//! record at every clean or detectable-failure exit, and hold the lock for
//! the whole process.
//!
//! Four writers cover the exits, in decreasing order of how much of the
//! process still works when they run:
//!
//! * the lifecycle calls (Starting / Initialized / Running / Saving /
//!   Finished / SavedAndQuit) and `Crashed` from the `StepResult` booleans,
//! * the Rust panic hook,
//! * the `libc::atexit` hook, which covers every C++ `exit(1)` fatal path
//!   (`ppf_fatal` and the CUDA error handler) since those never unwind Rust,
//! * the fatal-signal handler in [`crate::signal_sidecar`], which cannot
//!   write a record at all and leaves a one-token sidecar for the server.
//!
//! `SIGKILL` escapes all four by construction, so an OOM-kill still reaches
//! the server as an exit with no record; the server names it from the
//! launcher's status rather than guessing a cause.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use ppf_cts_formats::files;
use ppf_cts_formats::status::{
    self, crash_kind_from_error_code, lock::Lock, CrashKind, Outcome, Phase, RunStatus,
};

extern "C" {
    // Defined by the linked backend (libsimbackend_cuda / libsimbackend_cpu):
    // the fatal-exit reason a C++ exit(1) path stamped before dying (see
    // `ppf_cts_formats::status::error_code`), or 0 for a clean run / panic.
    fn ppf_fatal_code() -> u8;
    // First line of that path's report, or an empty string when the path set
    // a code without a message.
    fn ppf_fatal_detail() -> *const std::os::raw::c_char;
}

/// The backend's one-line fatal detail, or `None` when it is empty.
fn fatal_detail() -> Option<String> {
    // SAFETY: the backend returns a pointer to a process-lived static buffer
    // that is NUL-terminated by construction (`snprintf` into a fixed array).
    let raw = unsafe { ppf_fatal_detail() };
    if raw.is_null() {
        return None;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

struct Inner {
    output_dir: PathBuf,
    pid: u32,
    launch_id: String,
    emulated: bool,
    frame: i32,
    sim_time: f64,
    resumable: bool,
    seq: u64,
    terminal_written: bool,
    // Held (never explicitly dropped) for the whole process; the OS
    // releases the advisory lock on any death, which is the point.
    _lock: Option<Lock>,
}

static WRITER: Mutex<Option<Inner>> = Mutex::new(None);

/// Poison-tolerant lock: the writer must keep working even if some other
/// thread panicked while holding it (we never panic while holding it, but
/// the panic hook calls in here, so recover defensively).
fn writer_lock() -> MutexGuard<'static, Option<Inner>> {
    WRITER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Initialize the process-global writer. Call once in `main()`, after
/// `setup()`'s output-dir wipe and before the backend runs:
///   * scrub any stale `status.cbor` / `terminate_request` / `crash_signal`
///     (a resume skips the wipe, so a prior run's files can survive); never
///     scrub `status.lock`, which this process is about to own,
///   * acquire the liveness lock (held for the whole process),
///   * stamp `pid` + `launch_id`, write the initial `Starting` record,
///   * install the panic hook that stamps `Crashed{Panic}`, the fatal-signal
///     handler that records the signal, and the `atexit` hook that reads the
///     backend's fatal code.
pub fn init(output_dir: &str, launch_id: String, resumable_initial: bool) {
    let dir = PathBuf::from(output_dir);
    let _ = std::fs::remove_file(dir.join(files::STATUS_RECORD));
    let _ = std::fs::remove_file(dir.join(files::TERMINATE_REQUEST));
    status::signal_sidecar::scrub(&dir);
    let lock = match status::lock::acquire(&dir) {
        Ok(l) => Some(l),
        Err(e) => {
            log::warn!(
                "status: could not acquire liveness lock ({e}); \
                 crash-by-absence detection is degraded for this run"
            );
            None
        }
    };
    // Stamped with the same launch id as the record, so a sidecar left by a
    // prior run in a resumed output directory is not read as this run's.
    #[cfg(unix)]
    crate::signal_sidecar::install(&dir, &launch_id);
    let inner = Inner {
        output_dir: dir,
        pid: std::process::id(),
        launch_id,
        emulated: cfg!(feature = "emulated"),
        frame: 0,
        sim_time: 0.0,
        resumable: resumable_initial,
        seq: 0,
        terminal_written: false,
        _lock: lock,
    };
    *writer_lock() = Some(inner);
    write(Phase::Starting, None);
    install_panic_hook();
    // The backend's exit(1) paths (`ppf_fatal` and the CUDA error handler)
    // bypass Rust unwinding and the panic hook, so catch them via a C atexit
    // hook that reads the backend's fatal code and detail.
    unsafe { libc::atexit(atexit_fatal_hook) };
}

/// Registered with `libc::atexit`; runs on every process exit, including
/// the C++ `exit(1)` fatal paths that never unwind Rust. If the backend
/// stamped a fatal code, write the matching terminal `Crashed{kind}`
/// (idempotent: a clean run leaves the code 0 and a terminal record
/// already present, so this is a no-op then).
extern "C" fn atexit_fatal_hook() {
    let code = unsafe { ppf_fatal_code() };
    if code != 0 {
        let kind = crash_kind_from_error_code(code).unwrap_or(CrashKind::UnknownAbrupt);
        // The fallback covers a path that stamped a code without a message.
        // It names the code rather than inventing a cause, so a future path
        // that forgets its detail reports honestly instead of borrowing the
        // wrong text.
        let detail = fatal_detail()
            .unwrap_or_else(|| format!("solver exited via fatal hook (code {code}); see solver log"));
        terminal_crash(kind, detail);
    }
}

/// Stamp the terminal record for a backend `initialize()` that returned
/// false.
///
/// `initialize()` reports failure through its return value alone, so the
/// cause is whatever fatal code and detail it stamped on the way out. A code
/// of zero means it failed without naming a cause, which is reported as
/// exactly that: the solver knows more here than the server can (the server
/// would only see a process that exited), and an honest unknown outranks
/// attributing the failure to the one path that does set a code.
pub fn terminal_init_failure() {
    let code = unsafe { ppf_fatal_code() };
    let (kind, detail) = init_failure_outcome(code, fatal_detail());
    terminal_crash(kind, detail);
}

/// The decision `terminal_init_failure` makes, separated from the FFI read so
/// it can be exercised without a backend.
fn init_failure_outcome(code: u8, detail: Option<String>) -> (CrashKind, String) {
    match crash_kind_from_error_code(code) {
        Some(kind) => (
            kind,
            detail.unwrap_or_else(|| {
                format!("initialize() failed with fatal code {code}; see solver log")
            }),
        ),
        None => (
            CrashKind::UnknownAbrupt,
            "initialize() failed without setting a cause; see solver log".to_string(),
        ),
    }
}

/// Update the live progress record in place (no fsync; cheap per frame).
/// No-op once a terminal outcome has been written.
pub fn progress(phase: Phase, frame: i32, sim_time: f64) {
    {
        let mut g = writer_lock();
        if let Some(inner) = g.as_mut() {
            inner.frame = frame;
            inner.sim_time = sim_time;
        }
    }
    write(phase, None);
}

/// Record that a resumable checkpoint now exists on disk (call wherever a
/// `state_<N>.bin.gz` is written).
pub fn note_saved() {
    if let Some(inner) = writer_lock().as_mut() {
        inner.resumable = true;
    }
}

/// Stamp a terminal outcome durably (idempotent, first-writer-wins).
pub fn terminal(outcome: Outcome) {
    write(Phase::Ended, Some(outcome));
}

/// Convenience for the crash terminal.
pub fn terminal_crash(kind: CrashKind, detail: String) {
    terminal(Outcome::Crashed {
        sub_kind: kind,
        detail,
    });
}

/// Build the record under the lock, release it, then do the file I/O, so
/// the panic hook (which also calls here) can never deadlock on a write
/// that is in flight. A status-write failure never breaks the solver.
fn write(phase: Phase, outcome: Option<Outcome>) {
    let (dir, record, is_terminal) = {
        let mut g = writer_lock();
        let inner = match g.as_mut() {
            Some(i) => i,
            None => return,
        };
        if inner.terminal_written {
            return;
        }
        inner.seq += 1;
        let is_terminal = outcome.is_some();
        let record = RunStatus {
            phase,
            frame: inner.frame,
            sim_time: inner.sim_time,
            resumable: inner.resumable,
            outcome,
            seq: inner.seq,
            pid: inner.pid,
            launch_id: inner.launch_id.clone(),
            emulated: inner.emulated,
        };
        (inner.output_dir.clone(), record, is_terminal)
    };
    let res = if is_terminal {
        status::write_terminal(&dir, &record)
    } else {
        status::write_progress(&dir, &record)
    };
    match res {
        Ok(()) if is_terminal => {
            if let Some(inner) = writer_lock().as_mut() {
                inner.terminal_written = true;
            }
        }
        Ok(()) => {}
        Err(e) => log::warn!("status: write failed: {e}"),
    }
}

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Catch-all: a detectable failure (advance / init) stamps its
        // specific sub-kind first, and write_terminal is first-writer-
        // wins, so this only fires for panics with no prior terminal
        // record (e.g. an unexpected host panic).
        terminal_crash(CrashKind::Panic, info.to_string());
        prev(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_failure_names_the_cause_the_backend_stamped() {
        let (kind, detail) = init_failure_outcome(
            ppf_cts_formats::status::error_code::INIT_INTERSECTION,
            Some("the scene is already self-intersecting at t=0".into()),
        );
        assert_eq!(kind, CrashKind::InitIntersection);
        assert_eq!(detail, "the scene is already self-intersecting at t=0");
    }

    #[test]
    fn init_failure_without_a_code_reports_unknown_not_intersection() {
        // The point of the branch: `initialize()` returns a bare false, and
        // exactly one path inside it stamps a code. A second failure mode
        // added later must report honestly rather than inheriting the label
        // of the path that does stamp one.
        let (kind, detail) = init_failure_outcome(0, None);
        assert_eq!(kind, CrashKind::UnknownAbrupt);
        assert!(detail.contains("without setting a cause"), "{detail}");
    }

    #[test]
    fn init_failure_with_a_code_but_no_detail_names_the_code() {
        let (kind, detail) =
            init_failure_outcome(ppf_cts_formats::status::error_code::OOM, None);
        assert_eq!(kind, CrashKind::Oom);
        assert!(detail.contains("fatal code 2"), "{detail}");
    }
}
