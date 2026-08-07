// File: crates/ppf-cts-server/tests/monitor_integration.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// End-to-end test for the engine + monitor pipeline.
// Sets up a real filesystem (tempdir), drives the state machine via
// the public `dispatch` API to put the engine into Solver::Running,
// then writes solver output files (vert_*.bin, the terminal status.cbor
// record, intersection_records.json) and asserts the monitor task picks
// them up and dispatches the corresponding events.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ppf_cts_core::events::Event;
use ppf_cts_core::state::{Build, Data, Solver};
use ppf_cts_formats::status::{self, CrashKind, Outcome, Phase, RunStatus};
use ppf_cts_server::config::EngineConfig;
use ppf_cts_server::monitor::spawn_monitor;
use ppf_cts_server::{DefaultExecutor, EffectExecutor, ServerEngine};

mod common;
use common::wait_until;

fn out_dir(root: &Path) -> std::path::PathBuf {
    let out = root.join("session/output");
    std::fs::create_dir_all(&out).unwrap();
    out
}

/// A live-run record. `pid` is this test process, so the monitor's
/// PID-scoped liveness check sees the run as alive and never synthesizes
/// a crash from a non-terminal record.
fn live_status(frame: i32, phase: Phase) -> RunStatus {
    RunStatus {
        phase,
        frame,
        sim_time: 0.0,
        resumable: false,
        outcome: None,
        seq: frame as u64 + 1,
        pid: std::process::id(),
        launch_id: "testlaunch00".into(),
        emulated: true,
    }
}

/// Solver-authored progress record (the source of truth the monitor reads).
fn stage_running(root: &Path, frame: i32) {
    status::write_progress(&out_dir(root), &live_status(frame, Phase::Running)).unwrap();
}

fn stage_saving(root: &Path, frame: i32) {
    status::write_progress(&out_dir(root), &live_status(frame, Phase::Saving)).unwrap();
}

/// Solver-authored terminal record.
fn stage_terminal(root: &Path, frame: i32, outcome: Outcome) {
    let mut rec = live_status(frame, Phase::Ended);
    rec.outcome = Some(outcome);
    status::write_terminal(&out_dir(root), &rec).unwrap();
}

/// A non-terminal record whose owning process is confirmed dead: pid 0 is
/// never alive and no lock file exists, which is exactly the crash-by-absence
/// shape the classifier is for.
fn stage_dead_without_outcome(root: &Path, frame: i32) {
    let mut rec = live_status(frame, Phase::Running);
    rec.pid = 0;
    status::write_progress(&out_dir(root), &rec).unwrap();
}

/// The one-line signal record a fatal-signal handler leaves behind.
fn stage_signal_sidecar(root: &Path, name: &str) {
    std::fs::write(
        out_dir(root).join(ppf_cts_formats::files::CRASH_SIGNAL),
        format!("{name} testlaunch00\n"),
    )
    .unwrap();
}

/// Solver stderr, which the crash report quotes as evidence.
fn stage_stderr(root: &Path, body: &str) {
    let session = root.join("session");
    std::fs::create_dir_all(&session).unwrap();
    std::fs::write(session.join(ppf_cts_formats::files::ERROR_LOG), body).unwrap();
}

/// An engine plus executor wired for a fast poll and a short startup grace.
fn quick_engine() -> (ServerEngine, Arc<dyn EffectExecutor>) {
    let cfg = EngineConfig {
        monitor_interval_ms: 10,
        solver_startup_grace_ms: 50,
        ..Default::default()
    };
    (
        ServerEngine::new(cfg),
        Arc::new(DefaultExecutor::new()) as Arc<dyn EffectExecutor>,
    )
}

fn write_state_checkpoint(root: &Path, n: i32) {
    std::fs::write(out_dir(root).join(format!("state_{n}.bin.gz")), b"ckpt").unwrap();
}

/// Drive the engine into `solver = Running` for `name=p, root=<dir>`
/// without going through DoSpawnBuild (which is a frontend stub).
/// Must mirror the state-machine path:
/// upload landed → built (bypass) → start.
fn drive_to_running(engine: &ServerEngine, name: &str, root: &Path) {
    engine.set_project_context(name, root.to_str().unwrap());
    // UploadLanded gets data into Uploaded.
    engine.dispatch(Event::upload_landed("uid"));
    assert_eq!(engine.state().data, Data::Uploaded);

    // Bypass DoSpawnBuild by faking BuildCompleted directly. The
    // engine's transition layer accepts BuildCompleted unconditionally.
    engine.dispatch(Event::BuildCompleted);
    assert_eq!(engine.state().build, Build::Built);

    engine.dispatch(Event::StartRequested);
    assert_eq!(engine.state().solver, Solver::Running);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_detects_frame_updates() {
    let dir = tempfile::tempdir().unwrap();
    // Tight poll so the test doesn't sleep long.
    let cfg = EngineConfig {
        monitor_interval_ms: 25,
        ..Default::default()
    };
    let engine = ServerEngine::new(cfg);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());

    drive_to_running(&engine, "p", dir.path());
    let _h = spawn_monitor(engine.clone(), executor);

    // Solver advances; the monitor picks up the latest frame from the
    // status record it rewrites in place each frame.
    for n in [1, 3, 7] {
        stage_running(dir.path(), n);
    }

    // Wait up to ~500 ms for the monitor to observe.
    let _ = wait_until(|| engine.state().frame == 7, Duration::from_millis(500)).await;
    assert_eq!(engine.state().frame, 7, "monitor should pick up the latest frame");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_dispatches_finished_on_finished_file() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        monitor_interval_ms: 25,
        ..Default::default()
    };
    let engine = ServerEngine::new(cfg);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());

    drive_to_running(&engine, "p", dir.path());
    let _h = spawn_monitor(engine.clone(), executor);

    // Solver finishes: it writes a terminal Finished record at frame 10.
    stage_terminal(dir.path(), 10, Outcome::Finished);

    let _ = wait_until(
        || engine.state().solver == Solver::Idle,
        Duration::from_millis(500),
    )
    .await;
    let s = engine.state();
    assert_eq!(s.solver, Solver::Idle);
    // Final-frame catch-up ran before the SolverFinished transition.
    assert_eq!(s.frame, 10);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_reports_crash_after_grace_period() {
    let dir = tempfile::tempdir().unwrap();
    // Production grace is 3000 ms; tests use a small grace so wall
    // time stays short while still exercising the
    // pre-grace-suppression behavior.
    let cfg = EngineConfig {
        monitor_interval_ms: 10,
        solver_startup_grace_ms: 50,
        ..Default::default()
    };
    let engine = ServerEngine::new(cfg);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());

    drive_to_running(&engine, "p", dir.path());
    // The solver wrote a terminal Crashed record. A terminal outcome is
    // authoritative (no grace / liveness needed): the monitor reports
    // SolverCrashed straight from the record.
    stage_terminal(
        dir.path(),
        0,
        Outcome::Crashed {
            sub_kind: CrashKind::Intersection,
            detail: "near tri 42".into(),
        },
    );
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver == Solver::Failed,
        Duration::from_millis(500),
    )
    .await;

    let s = engine.state();
    assert_eq!(s.solver, Solver::Failed);
    assert!(
        s.error.starts_with("Intersection detected"),
        "unexpected error: {}",
        s.error
    );
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_marks_resumable_when_state_files_exist_at_finish() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        monitor_interval_ms: 25,
        ..Default::default()
    };
    let engine = ServerEngine::new(cfg);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());

    drive_to_running(&engine, "p", dir.path());
    let _h = spawn_monitor(engine.clone(), executor);

    // Solver wrote a checkpoint then finished cleanly. The monitor
    // should fire SolverFinished{resumable=true} (finish_solver scans
    // state_*.bin.gz), which the state machine surfaces on
    // `state.resumable`.
    write_state_checkpoint(dir.path(), 5);
    stage_terminal(dir.path(), 5, Outcome::Finished);

    let _ = wait_until(
        || engine.state().solver == Solver::Idle,
        Duration::from_millis(500),
    )
    .await;
    let s = engine.state();
    assert_eq!(s.solver, Solver::Idle);
    assert!(s.resumable, "resumable should reflect on-disk state files");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_dispatches_saving_when_sentinel_appears() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        monitor_interval_ms: 25,
        // Long grace so solver_busy = false on a test machine doesn't
        // flip the engine to Idle before we can observe the Saving
        // transition.
        solver_startup_grace_ms: 60_000,
        ..Default::default()
    };
    let engine = ServerEngine::new(cfg);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());

    drive_to_running(&engine, "p", dir.path());
    let _h = spawn_monitor(engine.clone(), executor);

    // Solver enters the saving phase (in response to a save_and_quit
    // request); the monitor reads phase=Saving from the live record and
    // dispatches SolverSaving.
    stage_saving(dir.path(), 0);

    let _ = wait_until(
        || engine.state().solver == Solver::Saving,
        Duration::from_millis(500),
    )
    .await;
    assert_eq!(engine.state().solver, Solver::Saving);
}

// ---------------------------------------------------------------------------
// Abrupt exits: a dead owning process with no terminal outcome. This is the
// branch the "segfault / OOM-kill / unrecoverable abort" report was about, and
// nothing covered it before.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_abrupt_exit_names_signal_from_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, executor) = quick_engine();
    drive_to_running(&engine, "p", dir.path());
    stage_dead_without_outcome(dir.path(), 12);
    stage_signal_sidecar(dir.path(), "SIGSEGV");
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver == Solver::Failed,
        Duration::from_millis(1000),
    )
    .await;
    let s = engine.state();
    assert_eq!(s.solver, Solver::Failed);
    assert_eq!(s.crash_kind, "killed_by_signal");
    assert!(s.error.contains("SIGSEGV"), "unexpected error: {}", s.error);
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_abrupt_exit_without_evidence_reports_unknown_with_raw_facts() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, executor) = quick_engine();
    drive_to_running(&engine, "p", dir.path());
    // No sidecar and no reaped launcher status: nothing names a cause.
    stage_dead_without_outcome(dir.path(), 9);
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver == Solver::Failed,
        Duration::from_millis(1000),
    )
    .await;
    let s = engine.state();
    assert_eq!(s.crash_kind, "unknown_abrupt");
    assert!(s.error.contains("frame 9"), "unexpected error: {}", s.error);
    assert!(
        s.error.contains("No signal record was written"),
        "unexpected error: {}",
        s.error
    );
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_classification_ignores_log_text() {
    // The never-guess gate. Two runs differ only in what `error.log` says;
    // the classification must be identical, because the classifier cannot see
    // the logs at all (they are not fields of AbruptEvidence).
    let mut kinds = Vec::new();
    for body in ["", "CUDA error: out of memory\nfatal: out of memory\n"] {
        let dir = tempfile::tempdir().unwrap();
        let (engine, executor) = quick_engine();
        drive_to_running(&engine, "p", dir.path());
        stage_dead_without_outcome(dir.path(), 4);
        stage_stderr(dir.path(), body);
        let monitor = spawn_monitor(engine.clone(), executor);
        let _ = wait_until(
            || engine.state().solver == Solver::Failed,
            Duration::from_millis(1000),
        )
        .await;
        kinds.push(engine.state().crash_kind);
        monitor.abort();
    }
    assert_eq!(
        kinds[0], kinds[1],
        "log text changed the classification: {kinds:?}"
    );
    assert_eq!(kinds[0], "unknown_abrupt");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_attaches_stderr_tail_on_abrupt_exit() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, executor) = quick_engine();
    drive_to_running(&engine, "p", dir.path());
    stage_dead_without_outcome(dir.path(), 3);
    stage_stderr(
        dir.path(),
        "PPF FATAL: aggregate lock Gram eigensolve failed.\n",
    );
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver == Solver::Failed,
        Duration::from_millis(1000),
    )
    .await;
    let s = engine.state();
    assert!(
        s.error
            .contains("PPF FATAL: aggregate lock Gram eigensolve failed."),
        "the stderr tail must reach the report: {}",
        s.error
    );
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_sigterm_without_a_request_is_not_a_finish() {
    // SIGTERM is the default kill of every supervisor, admin script and batch
    // scheduler, so it is not evidence that anyone MEANT to stop this run.
    // Reporting one as a finish would tell a user that a 300-frame run
    // completed at frame 6, and they would export that as the result.
    let dir = tempfile::tempdir().unwrap();
    let (engine, executor) = quick_engine();
    drive_to_running(&engine, "p", dir.path());
    stage_dead_without_outcome(dir.path(), 6);
    stage_signal_sidecar(dir.path(), "SIGTERM");
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver != Solver::Running,
        Duration::from_millis(1000),
    )
    .await;
    let s = engine.state();
    assert_eq!(
        s.solver,
        Solver::Failed,
        "an outside SIGTERM must not read as a completed run"
    );
    assert_eq!(s.crash_kind, "killed_by_signal");
    assert!(s.error.contains("SIGTERM"), "unexpected error: {}", s.error);
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_terminate_request_is_a_clean_stop() {
    // The server's own `terminate_request` IS evidence of intent: it writes
    // that file before killing, so a run that stops with it present is a
    // cooperative stop and never a crash, sidecar or not.
    let dir = tempfile::tempdir().unwrap();
    let (engine, executor) = quick_engine();
    drive_to_running(&engine, "p", dir.path());
    stage_dead_without_outcome(dir.path(), 6);
    stage_signal_sidecar(dir.path(), "SIGTERM");
    std::fs::write(
        out_dir(dir.path()).join(ppf_cts_formats::files::TERMINATE_REQUEST),
        b"",
    )
    .unwrap();
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver != Solver::Running,
        Duration::from_millis(1000),
    )
    .await;
    let s = engine.state();
    assert_eq!(s.solver, Solver::Idle, "a requested stop is not a crash");
    assert!(s.crash_kind.is_empty());
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_seals_its_verdict_into_the_status_record() {
    // The launcher's exit status lives only in the server's memory, so a
    // reconnect that re-derives the cause from disk alone can name strictly
    // less than the live report did. Sealing the verdict at the moment it is
    // made is what keeps the two reports the same.
    let dir = tempfile::tempdir().unwrap();
    let (engine, executor) = quick_engine();
    drive_to_running(&engine, "p", dir.path());
    stage_dead_without_outcome(dir.path(), 11);
    stage_signal_sidecar(dir.path(), "SIGSEGV");
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver == Solver::Failed,
        Duration::from_millis(1000),
    )
    .await;
    let rec = status::read(&out_dir(dir.path())).unwrap().unwrap();
    assert_eq!(rec.phase, Phase::Ended);
    match rec.outcome {
        Some(Outcome::Crashed { sub_kind, detail }) => {
            assert_eq!(sub_kind, CrashKind::KilledBySignal);
            assert!(detail.contains("SIGSEGV"), "sealed detail: {detail}");
        }
        other => panic!("expected a sealed Crashed outcome, got {other:?}"),
    }
    // Copied from the record the solver left, never invented.
    assert_eq!(rec.frame, 11);
    assert_eq!(rec.launch_id, "testlaunch00");
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_attaches_tails_on_launch_failure() {
    // No status record at all: the run script died before the solver wrote
    // one, so the only diagnosis available is what it printed.
    let dir = tempfile::tempdir().unwrap();
    let (engine, executor) = quick_engine();
    drive_to_running(&engine, "p", dir.path());
    out_dir(dir.path());
    stage_stderr(
        dir.path(),
        "error while loading shared libraries: libsimbackend_cuda.so\n",
    );
    let monitor = spawn_monitor(engine.clone(), executor);

    let _ = wait_until(
        || engine.state().solver == Solver::Failed,
        Duration::from_millis(1000),
    )
    .await;
    let s = engine.state();
    assert_eq!(s.crash_kind, "launch_failed");
    assert!(
        s.error.contains("libsimbackend_cuda.so"),
        "the stderr tail must reach the report: {}",
        s.error
    );
    monitor.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitor_torn_record_while_saving_reports_crash() {
    // A torn record is written by dying DURING write_progress, and Saving is
    // the phase most likely to be mid-write, so an engine state of Saving is
    // not evidence of a clean checkpoint here.
    let dir = tempfile::tempdir().unwrap();
    let cfg = EngineConfig {
        monitor_interval_ms: 10,
        solver_startup_grace_ms: 50,
        ..Default::default()
    };
    let engine = ServerEngine::new(cfg);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());
    drive_to_running(&engine, "p", dir.path());
    stage_saving(dir.path(), 8);
    let monitor = spawn_monitor(engine.clone(), executor);
    let _ = wait_until(
        || engine.state().solver == Solver::Saving,
        Duration::from_millis(1000),
    )
    .await;
    assert_eq!(engine.state().solver, Solver::Saving);

    // Truncate the record, as an interrupted in-place write leaves it.
    let path = out_dir(dir.path()).join(ppf_cts_formats::files::STATUS_RECORD);
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, &full[..full.len() / 2]).unwrap();

    let _ = wait_until(
        || engine.state().solver != Solver::Saving,
        Duration::from_millis(1000),
    )
    .await;
    let s = engine.state();
    assert_eq!(
        s.solver,
        Solver::Failed,
        "a torn record while saving is a crash during the checkpoint, not a clean save"
    );
    assert_eq!(s.crash_kind, "unknown_abrupt");
    monitor.abort();
}
