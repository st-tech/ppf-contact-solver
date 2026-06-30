// File: status_writer.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Process-global writer for the structured solver status record
//! (`ppf_cts_formats::status`): the solver host is the sole
//! writer of `status.cbor` and the holder of the liveness lock.
//!
//! A healthy run always stamps a terminal [`Outcome`]; its absence
//! (with a freed lock and a dead owning PID) is the server's
//! crash-by-absence verdict. So the contract here is: write a terminal
//! record at every clean or detectable-failure exit, hold the lock for
//! the whole process, and let abrupt deaths (segfault / OOM-kill /
//! unrecoverable CUDA abort) fall through to the lock + non-terminal
//! record the server reads.
//!
//! The lifecycle writers (Starting / Initialized / Running / Saving /
//! Finished / SavedAndQuit) plus Crashed from the `StepResult` booleans
//! and the Rust panic hook stamp the terminal record on the paths they
//! own. Abrupt `exit(1)` OOM / CUDA-driver paths that bypass them
//! surface as the lock-based `UnknownAbrupt` crux, never as a silent
//! clean exit.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use ppf_cts_formats::files;
use ppf_cts_formats::status::{
    self, crash_kind_from_error_code, lock::Lock, CrashKind, Outcome, Phase, RunStatus,
};

extern "C" {
    // Defined by the linked backend (libsimbackend_cuda / libsimbackend_cpu):
    // the fatal-exit reason a CUDA exit(1) path stamped before dying
    // (2 = OOM, 3 = CUDA driver), or 0 for a clean run / panic.
    fn ppf_fatal_code() -> u8;
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
///   * scrub any stale `status.cbor` / `terminate_request` (a resume
///     skips the wipe, so a prior run's files can survive); never scrub
///     `status.lock`, which this process is about to own,
///   * acquire the liveness lock (held for the whole process),
///   * stamp `pid` + `launch_id`, write the initial `Starting` record,
///   * install the panic hook that stamps `Crashed{Panic}`.
pub fn init(output_dir: &str, launch_id: String, resumable_initial: bool) {
    let dir = PathBuf::from(output_dir);
    let _ = std::fs::remove_file(dir.join(files::STATUS_RECORD));
    let _ = std::fs::remove_file(dir.join(files::TERMINATE_REQUEST));
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
    // The CUDA backend's exit(1) paths (OOM / driver error) bypass Rust
    // unwinding and the panic hook, so catch them via a C atexit hook that
    // reads the backend's fatal code.
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
        terminal_crash(
            kind,
            format!("solver exited via fatal hook (code {code}); see solver log"),
        );
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
