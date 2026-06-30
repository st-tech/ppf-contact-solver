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
