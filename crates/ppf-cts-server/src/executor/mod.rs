// File: crates/ppf-cts-server/src/executor/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `EffectExecutor`: handles the side-effects emitted by the engine.
// Direct port of server/engine.py's `EffectExecutor`.
//
// The build pipeline runs in a Python subprocess
// (`frontend/build_worker.py`). The Rust side does not link libpython:
// it spawns the worker, parses line-oriented progress and error
// markers from its stdout, and forwards a SIGTERM on cooperative
// cancel. The remaining frontend-dependent effects (sentinel writes,
// project-dir creation, solver launch) stay native.
//
// The trait is intentionally minimal so wire-protocol handlers,
// debug runners, and tests can compose their own executor
// (e.g. an emulator path that fakes solver IO).

use async_trait::async_trait;
use ppf_cts_core::effects::Effect;
use ppf_cts_core::events::Event;

use crate::engine::ServerEngine;

mod build;
mod session;
pub(crate) mod solver;

// Pick the right solver-busy + terminate variants per build. The
// emulated build runs under the test rig where many workers share the
// same host; using the global scan would let one worker's solver kill
// another's (the historical `Utils.busy` patch from server/emulator.py
// addressed the same race on the python side). The check and the kill
// must be selected as a matched pair: a descendant-only busy check
// paired with a host-global kill would still SIGTERM every peer
// worker's solver the moment our own descendant is detected, so we
// also narrow the terminator to descendants under `emulated`.
#[cfg(feature = "emulated")]
use ppf_cts_core::utils::{
    solver_busy_descendants_only as solver_busy_for_check,
    terminate_solver_descendants_only as terminate_solver_for_kill,
};
#[cfg(not(feature = "emulated"))]
use ppf_cts_core::utils::{
    solver_busy as solver_busy_for_check, terminate_solver as terminate_solver_for_kill,
};

/// Trait for processing one effect. Stateless from the trait's
/// perspective; impls hold whatever state they need internally.
///
/// `execute` is `async fn` so implementations can `.await` directly
/// (e.g. `tokio::process::Command::status().await`) without
/// fabricating a runtime handle. The `async-trait` desugaring keeps
/// the trait object-safe so we can keep using `Arc<dyn EffectExecutor>`.
#[async_trait]
pub trait EffectExecutor: Send + Sync {
    async fn execute(&self, effect: Effect, engine: &ServerEngine);
}

/// Default Rust-native implementation. Frontend-dependent effects
/// dispatch through tokio tasks; the build effect spawns the Python
/// build worker and forwards its stdout protocol as engine events
/// (see `executor::build`).
pub struct DefaultExecutor;

impl DefaultExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EffectExecutor for DefaultExecutor {
    async fn execute(&self, effect: Effect, engine: &ServerEngine) {
        match effect {
            Effect::DoLog { message } => {
                log::info!(target: "ppf::executor", "{message}");
            }
            Effect::DoDeleteProjectData { root } => {
                if root.is_empty() {
                    return;
                }
                match std::fs::remove_dir_all(std::path::Path::new(&root)) {
                    Ok(()) => log::info!(target: "ppf::executor", "Deleted project data at {root}"),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        log::debug!(target: "ppf::executor", "Project dir already absent: {root}");
                    }
                    Err(e) => log::error!(target: "ppf::executor", "Failed to delete {root}: {e}"),
                }
            }
            Effect::DoCancelBuild => {
                engine.cancel_active_build();
            }
            Effect::DoKillSolver => {
                // Stamp a durable record of intent before the kill: a
                // reconnecting server has no in-memory Idle, so it reads
                // terminate_request to report a clean Terminated instead of
                // synthesizing a crash from the lock-free + dead-pid crux.
                // The next launch scrubs it. Best-effort; the in-memory Idle
                // remains the primary classifier for the live monitor.
                let root = engine.state().root;
                if !root.is_empty() {
                    let p = ppf_cts_formats::files::session_output_dir(std::path::Path::new(&root))
                        .join(ppf_cts_formats::files::TERMINATE_REQUEST);
                    let _ = std::fs::write(&p, b"");
                }
                if solver_busy_for_check() {
                    log::info!(target: "ppf::solver", "DoKillSolver: terminating active solver");
                    terminate_solver_for_kill();
                } else {
                    log::debug!(target: "ppf::solver", "DoKillSolver: no solver running");
                }
            }
            Effect::DoSpawnBuild { preserve_output } => {
                build::spawn_build_task(engine, preserve_output);
            }
            Effect::DoLaunchSolver { resume_from } => {
                solver::launch_solver(engine, resume_from).await;
            }
            Effect::DoRequestSaveAndQuit => {
                session::request_save_and_quit(engine).await;
            }
            Effect::DoLoadApp { name, root } => {
                session::load_app(engine, &name, &root).await;
            }
        }
    }
}

/// Convenience: dispatch an event through `engine.dispatch` then
/// drain the resulting effects through `executor`.
pub(crate) async fn dispatch_with_executor(
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    event: Event,
) {
    let effects = engine.dispatch(event);
    for fx in effects {
        executor.execute(fx, engine).await;
    }
}

/// Dispatch an event using whichever executor the engine has been
/// attached to via `ServerEngine::attach_executor`, falling back to
/// a fresh `DefaultExecutor` if no executor is bound (test paths
/// that drive the engine in isolation). Used by re-entrant effect
/// handlers (build pipeline, `launch_solver`,
/// `request_save_and_quit`, `load_app`) that need to surface
/// follow-up events without owning a reference to the outer
/// executor; the attached executor is required so test harnesses
/// (e.g. counting wrappers) observe the re-dispatched effects.
pub(crate) async fn dispatch_re_entrant(engine: &ServerEngine, event: Event) {
    if let Some(exec) = engine.executor() {
        dispatch_with_executor(engine, exec.as_ref(), event).await;
    } else {
        let exec = DefaultExecutor::new();
        dispatch_with_executor(engine, &exec, event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use ppf_cts_core::events::Event;
    use ppf_cts_core::state::Build;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    /// Build an engine with the data-uploaded prerequisite met so
    /// `BuildRequested` actually advances state instead of being
    /// rejected by the transition guard. `ServerEngine` is already
    /// `Clone` (Arc inside), so callers don't need to wrap it.
    fn engine_ready_to_build() -> ServerEngine {
        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("p", "/tmp/p");
        engine.dispatch(Event::upload_landed("uid"));
        engine
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_build_via_executor_trips_token() {
        let engine = ServerEngine::new(EngineConfig::default());
        let exec = DefaultExecutor::new();
        let h = engine.install_cancel_handle();
        exec.execute(Effect::DoCancelBuild, &engine).await;
        assert!(h.is_cancelled());
    }

    /// Re-entrant dispatch must route through the attached executor
    /// instead of fabricating a fresh `DefaultExecutor`. We attach a
    /// counting executor, trigger a re-entrant event (the no-session
    /// save/quit path), and check the count went up.
    #[tokio::test]
    async fn dispatch_re_entrant_uses_attached_executor() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingExecutor {
            inner: DefaultExecutor,
            calls: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl EffectExecutor for CountingExecutor {
            async fn execute(&self, effect: Effect, engine: &ServerEngine) {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.execute(effect, engine).await;
            }
        }

        let engine = ServerEngine::new(EngineConfig::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let exec: Arc<dyn EffectExecutor> = Arc::new(CountingExecutor {
            inner: DefaultExecutor::new(),
            calls: calls.clone(),
        });
        engine.attach_executor(&exec);

        // No project root => DoRequestSaveAndQuit re-dispatches
        // ErrorOccurred via dispatch_re_entrant. Both the original
        // DoRequestSaveAndQuit and the follow-up DoLog from the
        // ErrorOccurred transition pass through the counting executor.
        exec.execute(Effect::DoRequestSaveAndQuit, &engine).await;
        let n = calls.load(Ordering::SeqCst);
        assert!(
            n >= 2,
            "expected at least 2 routed effects (original + re-dispatch), got {n}"
        );
    }

    #[tokio::test]
    async fn delete_project_data_handles_missing_dir() {
        let engine = ServerEngine::new(EngineConfig::default());
        let exec = DefaultExecutor::new();
        exec.execute(
            Effect::DoDeleteProjectData {
                root: "/tmp/ppf-cts-test-does-not-exist-{abc}".into(),
            },
            &engine,
        )
        .await;
    }

    #[tokio::test]
    async fn delete_project_data_removes_real_dir() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("project");
        std::fs::create_dir_all(p.join("session/output")).unwrap();
        std::fs::write(p.join("session/output/finished.txt"), "ok").unwrap();
        assert!(p.exists());

        let engine = ServerEngine::new(EngineConfig::default());
        let exec = DefaultExecutor::new();
        exec.execute(
            Effect::DoDeleteProjectData {
                root: p.to_string_lossy().to_string(),
            },
            &engine,
        )
        .await;
        assert!(!p.exists(), "project dir should be gone");
    }

    /// 4.2: with `execute` async, this test now drives the full
    /// effect set including `DoSpawnBuild` (which still requires
    /// a tokio runtime, but tests get one for free under
    /// `#[tokio::test]`). The pre-4.2 version of this test had to
    /// skip `DoSpawnBuild`; the comment-out is now gone.
    #[tokio::test]
    async fn dispatch_with_executor_runs_emitted_effects() {
        let engine = ServerEngine::new(EngineConfig::default());
        let exec = DefaultExecutor::new();

        engine.set_project_context("p", "/tmp/p-no-such-dir-{abc}");
        engine.dispatch(Event::upload_landed("uid"));
        let effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
        for fx in effects {
            exec.execute(fx, &engine).await;
        }
        assert_eq!(engine.state().build, Build::Building);
    }

    // ----- DoLoadApp -----

    #[tokio::test]
    async fn load_app_creates_missing_project_dir() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("demo");
        assert!(!project.exists());

        let engine = ServerEngine::new(EngineConfig::default());
        let exec = DefaultExecutor::new();
        exec.execute(
            Effect::DoLoadApp {
                name: "demo".into(),
                root: project.to_string_lossy().to_string(),
            },
            &engine,
        )
        .await;

        assert!(project.exists(), "project dir should be created");
        let s = engine.state();
        assert_eq!(s.name, "demo");
        assert_eq!(s.root, project.to_string_lossy());
    }

    #[tokio::test]
    async fn load_app_existing_dir_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("preexisting");
        std::fs::create_dir_all(&project).unwrap();

        let engine = ServerEngine::new(EngineConfig::default());
        let exec = DefaultExecutor::new();
        exec.execute(
            Effect::DoLoadApp {
                name: "preexisting".into(),
                root: project.to_string_lossy().to_string(),
            },
            &engine,
        )
        .await;
        assert!(project.exists());
        assert_eq!(engine.state().name, "preexisting");
    }

    // ----- DoRequestSaveAndQuit -----

    #[tokio::test]
    async fn request_save_and_quit_writes_sentinel_file() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path();

        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("p", &project.to_string_lossy());
        let exec = DefaultExecutor::new();
        exec.execute(Effect::DoRequestSaveAndQuit, &engine).await;

        let sentinel = project.join("session").join("output").join("save_and_quit");
        assert!(sentinel.exists(), "save_and_quit sentinel should be written");
    }

    #[tokio::test]
    async fn request_save_and_quit_dispatches_error_when_no_root() {
        let engine = ServerEngine::new(EngineConfig::default());
        let exec = DefaultExecutor::new();
        exec.execute(Effect::DoRequestSaveAndQuit, &engine).await;
        // No project context -> ErrorOccurred fires; state.error
        // gets populated by the transition.
        let err = engine.state().error;
        assert!(
            err.contains("no session"),
            "expected 'no session' error, got {:?}",
            err
        );
    }

    // ----- DoLaunchSolver -----

    #[tokio::test]
    async fn launch_solver_no_run_script_dispatches_error() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path();
        // No command.sh / command.bat exists.

        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("p", &project.to_string_lossy());
        let exec = DefaultExecutor::new();
        exec.execute(Effect::DoLaunchSolver { resume_from: None }, &engine).await;

        let err = engine.state().error;
        assert!(
            err.contains("run script not found") || err.contains("not found"),
            "expected missing-run-script error, got {:?}",
            err
        );
    }

    #[tokio::test]
    #[cfg(not(target_os = "windows"))]
    async fn launch_solver_spawns_real_subprocess() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path();
        let session = project.join("session");
        std::fs::create_dir_all(&session).unwrap();
        let cmd_path = session.join("command.sh");
        // A trivial script that exits cleanly so the test doesn't
        // leave a process around. We still get to verify the spawn
        // plumbing: argv, cwd, log redirection.
        std::fs::write(
            &cmd_path,
            "#!/usr/bin/env bash\n\
             echo \"args=$*\" >&1\n\
             echo \"pwd=$(pwd)\" >&1\n\
             echo \"err sample\" >&2\n\
             exit 0\n",
        )
        .unwrap();

        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("p", &project.to_string_lossy());
        let exec = DefaultExecutor::new();
        exec.execute(
            Effect::DoLaunchSolver {
                resume_from: Some(-1),
            },
            &engine,
        )
        .await;

        // Give the subprocess time to run + flush.
        tokio::time::sleep(Duration::from_millis(400)).await;

        let log = std::fs::read_to_string(session.join("stdout.log")).unwrap_or_default();
        let err = std::fs::read_to_string(session.join("error.log")).unwrap_or_default();
        assert!(
            log.contains("--load=-1"),
            "expected --load=-1 in stdout log, got {:?}",
            log
        );
        assert!(
            log.contains(&project.to_string_lossy().to_string())
                || log.contains("pwd="),
            "expected cwd hint in stdout, got {:?}",
            log
        );
        assert!(err.contains("err sample"), "expected stderr capture, got {:?}", err);
        // ErrorOccurred should NOT have fired on success.
        assert_eq!(engine.state().error, "");
    }

    // ----- DoSpawnBuild -----

    #[tokio::test]
    async fn spawn_build_completes_or_fails_gracefully() {
        let engine = engine_ready_to_build();
        // BuildRequested -> DoSpawnBuild. The test runtime's
        // spawned task drives the build pipeline; we wait for the
        // engine state to settle.
        let exec = DefaultExecutor::new();
        let effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
        for fx in effects {
            exec.execute(fx, &engine).await;
        }
        assert_eq!(engine.state().build, Build::Building);

        // Wait long enough for the worker spawn plus build, or for
        // the no-GPU short-circuit to GpuCheckFailed. On hosts
        // without a GPU the pipeline short-circuits to Failed; either
        // Built or Failed is an acceptable terminal state for this
        // plumbing test.
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let s = engine.state();
            if s.build != Build::Building {
                assert!(
                    matches!(s.build, Build::Built | Build::Failed),
                    "unexpected build state: {:?}",
                    s.build
                );
                return;
            }
        }
        panic!(
            "build pipeline never left Building state (final state: {:?})",
            engine.state().build
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_build_observes_cancellation() {
        let engine = engine_ready_to_build();
        let exec = DefaultExecutor::new();
        let effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
        for fx in effects {
            exec.execute(fx, &engine).await;
        }
        // Cancel almost immediately.
        tokio::time::sleep(Duration::from_millis(20)).await;
        engine.cancel_active_build();

        // `drive_build_worker` selects on the cancel token while
        // draining worker stdout and forwards SIGTERM on cancel;
        // cancellation should land within a tick or two. On a host
        // without a GPU the build may instead exit via
        // GpuCheckFailed before the cancel is observed, which is
        // also a valid terminal state.
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let s = engine.state();
            if s.build != Build::Building {
                assert!(
                    matches!(s.build, Build::None | Build::Failed | Build::Built),
                    "unexpected post-cancel state: {:?}",
                    s.build
                );
                return;
            }
        }
        panic!(
            "build pipeline ignored cancel (final state: {:?})",
            engine.state().build
        );
    }

    // ----- drive_build_worker (mock script) -----

    /// Drive `drive_build_worker` against a tiny inline shell script
    /// (chmod +x) that emits the wire protocol. We use `/bin/sh` as
    /// the "interpreter" and pass the script as the worker path so
    /// the test doesn't require Python on the runner. The protocol
    /// is deliberately interpreter-agnostic, so this exercises the
    /// Rust line-parser and lifecycle without coupling to Python.
    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_emits_progress_then_completes() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("mock_worker.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             echo 'PROGRESS percent=0.10 info=Loading scene data'\n\
             echo 'PROGRESS percent=0.50 info=Decoding scene'\n\
             echo 'PROGRESS percent=0.95 info=Building fixed scene'\n\
             exit 0\n",
        )
        .unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();

        let engine = engine_ready_to_build();
        let cancel = engine.install_cancel_handle();
        // Use /bin/sh as the "python" so the script runs directly.
        let outcome = build::drive_build_worker(
            &engine,
            cancel,
            Path::new("/bin/sh"),
            &script,
            "demo",
            "/tmp/demo",
            false,
        )
        .await;
        assert!(
            matches!(outcome, build::BuildOutcome::Completed),
            "expected Completed, got {}",
            other_disc(&outcome),
        );

        // Engine should have absorbed the last progress beat (0.95
        // here; the orchestrator that owns BuildCompleted dispatch is
        // `run_build_pipeline`'s caller, so we check progress only).
        let s = engine.state();
        assert!(s.build_progress > 0.9, "got progress {}", s.build_progress);
        assert!(
            s.build_info.contains("fixed scene") || s.build_info.contains("Building"),
            "unexpected build_info {:?}",
            s.build_info
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_propagates_error_line() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("mock_err.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             echo 'PROGRESS percent=0.10 info=Loading'\n\
             echo 'ERROR tetwild segfaulted'\n\
             exit 1\n",
        )
        .unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();

        let engine = engine_ready_to_build();
        let cancel = engine.install_cancel_handle();
        let outcome = build::drive_build_worker(
            &engine,
            cancel,
            Path::new("/bin/sh"),
            &script,
            "demo",
            "/tmp/demo",
            false,
        )
        .await;
        match outcome {
            build::BuildOutcome::Failed(msg) => assert!(
                msg.contains("tetwild"),
                "expected ERROR text in failure reason, got {:?}",
                msg
            ),
            other => panic!("expected Failed, got {:?}", other_disc(&other)),
        }
    }

    /// A descendant that inherits the worker's stdout must not hold the
    /// drain open after the worker itself has exited.
    ///
    /// EOF on that pipe reports that every holder of its write end has
    /// closed it, which is a different question from whether the worker
    /// finished. A build whose worker leaves a background process behind
    /// would otherwise sit in BUILDING for as long as that process lives,
    /// with the cancel select already exited so cancel could not reach it.
    ///
    /// Both halves are asserted: the drain ends promptly, AND the worker's
    /// ERROR line still arrives. A bound that truncated the report would
    /// satisfy the first on its own.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drive_build_worker_ends_when_a_descendant_holds_stdout() {
        let dir = tempfile::tempdir().unwrap();
        // `sleep` inherits stdout and outlives the worker by far more than
        // the post-exit grace, so EOF cannot be what ends the drain.
        let script = write_mock_worker(
            dir.path(),
            "mock_lingering_child.sh",
            "sleep 30 &
             echo 'PROGRESS percent=0.10 info=Loading'
             echo 'ERROR tetwild segfaulted'
             exit 1
",
        );

        let engine = engine_ready_to_build();
        let cancel = engine.install_cancel_handle();
        let started = std::time::Instant::now();
        let outcome = build::drive_build_worker(
            &engine,
            cancel,
            Path::new("/bin/sh"),
            &script,
            "demo",
            "/tmp/demo",
            false,
        )
        .await;
        let elapsed = started.elapsed();

        match outcome {
            build::BuildOutcome::Failed(msg) => assert!(
                msg.contains("tetwild"),
                "the bound cost the worker its report: {msg:?}"
            ),
            other => panic!("expected Failed, got {:?}", other_disc(&other)),
        }
        assert!(
            elapsed < Duration::from_secs(10),
            "drive_build_worker waited on the descendant, not the worker: {elapsed:?}"
        );
    }

    /// Write an executable `/bin/sh` mock worker and return its path.
    /// The marker tests below each need several protocol lines, so they
    /// share the chmod boilerplate rather than repeating it six times.
    #[cfg(unix)]
    fn write_mock_worker(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join(name);
        std::fs::write(&script, format!("#!/bin/sh\n{body}")).unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();
        script
    }

    /// Drive a mock worker to exit and return the failure reason it
    /// produced. Panics with the outcome tag if the worker did not fail.
    #[cfg(unix)]
    async fn failure_reason(script: &Path) -> String {
        let engine = engine_ready_to_build();
        let cancel = engine.install_cancel_handle();
        let outcome = build::drive_build_worker(
            &engine,
            cancel,
            Path::new("/bin/sh"),
            script,
            "demo",
            "/tmp/demo",
            false,
        )
        .await;
        match outcome {
            build::BuildOutcome::Failed(msg) => msg,
            other => panic!("expected Failed, got {}", other_disc(&other)),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_attaches_error_detail_lines() {
        let dir = tempfile::tempdir().unwrap();
        let script = write_mock_worker(
            dir.path(),
            "mock_detail.sh",
            "echo 'ERROR OSError: [Errno 22] Invalid argument'\n\
             echo 'ERRORDETAIL Traceback (most recent call last):'\n\
             echo 'ERRORDETAIL   File \"frontend/_mesh_.py\", line 899'\n\
             exit 1\n",
        );
        let reason = failure_reason(&script).await;
        assert_eq!(
            reason.split('\n').collect::<Vec<_>>(),
            vec![
                "OSError: [Errno 22] Invalid argument",
                "Traceback (most recent call last):",
                // Indentation survives; a traceback is unreadable without it.
                "  File \"frontend/_mesh_.py\", line 899",
            ],
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_detail_belongs_to_last_error() {
        let dir = tempfile::tempdir().unwrap();
        let script = write_mock_worker(
            dir.path(),
            "mock_two_errors.sh",
            "echo 'ERROR first failure'\n\
             echo 'ERRORDETAIL first detail'\n\
             echo 'ERROR second failure'\n\
             echo 'ERRORDETAIL second detail'\n\
             exit 1\n",
        );
        let reason = failure_reason(&script).await;
        // Last ERROR wins, so the detail under the discarded headline goes
        // with it instead of being read as the survivor's traceback.
        assert_eq!(reason, "second failure\nsecond detail");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_names_last_stage_on_bare_exit() {
        let dir = tempfile::tempdir().unwrap();
        let script = write_mock_worker(
            dir.path(),
            "mock_bare_exit.sh",
            "echo 'PROGRESS percent=0.17 info=Tetrahedralizing Rock (1/1, new)...'\n\
             exit 3\n",
        );
        let reason = failure_reason(&script).await;
        assert!(reason.contains("code 3"), "lost the exit code: {reason:?}");
        assert!(
            reason.contains("Tetrahedralizing Rock (1/1, new)..."),
            "lost the last stage: {reason:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_forwards_stderr_tail_when_no_error_line() {
        let dir = tempfile::tempdir().unwrap();
        let script = write_mock_worker(
            dir.path(),
            "mock_native_crash.sh",
            "echo 'Fatal Python error: Segmentation fault MARKER_XYZ' >&2\n\
             exit 1\n",
        );
        let reason = failure_reason(&script).await;
        assert!(
            reason.contains("Build Worker stderr"),
            "expected the stderr tail delimiter: {reason:?}"
        );
        assert!(
            reason.contains("MARKER_XYZ"),
            "expected the stderr line itself: {reason:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_omits_stderr_tail_when_error_line_present() {
        let dir = tempfile::tempdir().unwrap();
        let script = write_mock_worker(
            dir.path(),
            "mock_chatty.sh",
            "echo 'tetrahedralizer chatter MARKER_XYZ' >&2\n\
             echo 'ERROR ValueError: Plane: no enclosed volume'\n\
             exit 1\n",
        );
        let reason = failure_reason(&script).await;
        // The tetrahedralizer writes to stderr unredirected on Windows, so
        // a message the worker did report must not be buried under it.
        assert_eq!(reason, "ValueError: Plane: no enclosed volume");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_caps_error_detail_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let script = write_mock_worker(
            dir.path(),
            "mock_flood.sh",
            "echo 'ERROR RuntimeError: deep recursion'\n\
             i=0\n\
             while [ $i -lt 5000 ]; do\n\
             echo \"ERRORDETAIL   File x.py, line $i, in frame$i\"\n\
             i=$((i+1))\n\
             done\n\
             exit 1\n",
        );
        let reason = failure_reason(&script).await;
        assert!(
            reason.len() < 2 * build::MAX_ERROR_DETAIL_BYTES,
            "detail block outgrew its cap: {} bytes",
            reason.len()
        );
        assert!(
            reason.ends_with("full traceback in server.log>"),
            "expected the truncation notice last, got tail {:?}",
            &reason[reason.len().saturating_sub(200)..]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drive_build_worker_survives_non_utf8_stdout() {
        let dir = tempfile::tempdir().unwrap();
        // A native library that writes straight to fd 1 follows the console
        // code page, so on a Japanese Windows install its chatter reaches
        // this stream as cp932 bytes. Those bytes must cost one garbled
        // line, never the report that follows them.
        let script = write_mock_worker(
            dir.path(),
            "mock_bad_bytes.sh",
            "printf 'chatter \\377\\376 bad\\n'\n\
             echo 'ERROR OSError: [Errno 22] Invalid argument | while: Tetrahedralizing Rock'\n\
             echo 'ERRORDETAIL   File \"frontend/_mesh_.py\", line 899'\n\
             exit 1\n",
        );
        let reason = failure_reason(&script).await;
        assert_eq!(
            reason.split('\n').collect::<Vec<_>>(),
            vec![
                "OSError: [Errno 22] Invalid argument | while: Tetrahedralizing Rock",
                "  File \"frontend/_mesh_.py\", line 899",
            ],
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drive_build_worker_observes_sigterm_cancel() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("mock_sleep.sh");
        // Trap SIGTERM and exit 130 so the cancel path looks just
        // like the real worker's KeyboardInterrupt branch. The
        // outer sleep is 30s; we cancel within ~50ms so the test
        // budget stays small.
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             trap 'exit 130' TERM\n\
             echo 'PROGRESS percent=0.05 info=Starting'\n\
             # Background sleep + wait so trap fires immediately.\n\
             sleep 30 &\n\
             wait $!\n\
             exit 0\n",
        )
        .unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();

        let engine = engine_ready_to_build();
        let cancel = engine.install_cancel_handle();
        let cancel_for_trip = cancel.clone();
        // Trip cancel after a short delay so the worker has a chance
        // to install its trap and emit at least one progress line.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            cancel_for_trip.cancel();
        });

        let outcome = build::drive_build_worker(
            &engine,
            cancel,
            Path::new("/bin/sh"),
            &script,
            "demo",
            "/tmp/demo",
            false,
        )
        .await;
        assert!(
            matches!(outcome, build::BuildOutcome::Cancelled),
            "expected Cancelled, got {:?}",
            other_disc(&outcome)
        );
    }

    /// Helper for assertion messages: BuildOutcome doesn't derive
    /// Debug (Failed carries arbitrary text), so we render the
    /// variant tag only when a test panics.
    fn other_disc(o: &build::BuildOutcome) -> &'static str {
        match o {
            build::BuildOutcome::Completed => "Completed",
            build::BuildOutcome::Cancelled => "Cancelled",
            build::BuildOutcome::Failed(_) => "Failed",
            build::BuildOutcome::AlreadyDispatched => "AlreadyDispatched",
        }
    }

    // ----- non-UTF-8 on a worker stream -----
    //
    // Both drains read their pipe as UTF-8 lines. A byte sequence that is
    // not valid UTF-8 makes the read return `ErrorKind::InvalidData`, which
    // is a decode verdict about one line, not a signal that the pipe is
    // finished. A build worker's streams carry whatever the tools it shells
    // out to emit, so a stray byte in some tool's locale is ordinary input.
    //
    // Unix-only, like the three mock-driven tests above: the harness is a
    // `/bin/sh` script, which a Windows runner does not have. `build.rs`'s
    // drains therefore have no Windows coverage at all, here or elsewhere.
    //
    // Both gates are ARMED against defects that are open. Each carries
    // `#[should_panic(expected = ...)]` naming the wrong outcome the defect
    // produces, which is the Rust counterpart of the Python gates'
    // `pytest.mark.xfail(strict=True)`: the test runs in the blocking unit
    // job, passes while the defect is present, and fails with "test did not
    // panic as expected" the moment it is fixed. Answer that failure by
    // deleting the attribute, which turns the test into the permanent
    // regression gate its name describes.

    /// Write *body* as an executable `/bin/sh` script under a fresh temp
    /// directory and return the directory (which must outlive the run) and
    /// the script path.
    #[cfg(unix)]
    fn write_utf8_probe_worker(name: &str, body: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join(name);
        std::fs::write(&script, body).unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();
        (dir, script)
    }

    /// A failing worker's ERROR line must reach the caller even when an
    /// earlier stdout line was not valid UTF-8.
    ///
    /// The reason string is the only thing the user is shown, so losing it
    /// turns a named failure into the exit code that produced it.
    ///
    /// The drain decodes lossily and splits on newline bytes, so a byte no
    /// valid UTF-8 sequence contains costs its own line and nothing after it.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drive_build_worker_reports_the_error_line_after_invalid_utf8() {
        // \377 is 0xFF, which no valid UTF-8 sequence contains.
        let (_dir, script) = write_utf8_probe_worker(
            "mock_bad_utf8_stdout.sh",
            "#!/bin/sh\n\
             echo 'PROGRESS percent=0.10 info=Loading'\n\
             printf '\\377\\n'\n\
             echo 'ERROR tetwild segfaulted'\n\
             exit 1\n",
        );

        let engine = engine_ready_to_build();
        let cancel = engine.install_cancel_handle();
        let outcome = build::drive_build_worker(
            &engine,
            cancel,
            Path::new("/bin/sh"),
            &script,
            "demo",
            "/tmp/demo",
            false,
        )
        .await;
        match outcome {
            build::BuildOutcome::Failed(msg) => assert!(
                msg.contains("tetwild"),
                "expected the worker's ERROR text in the failure reason, got {msg:?}"
            ),
            other => panic!("expected Failed, got {}", other_disc(&other)),
        }
    }

    /// A worker that emits one non-UTF-8 byte on stderr and then keeps
    /// writing must still complete.
    ///
    /// The observable is the EXIT STATUS, not a timeout. Ending the stderr
    /// drain drops its reader, which closes the parent's read end, so the
    /// worker takes SIGPIPE on its next write and dies in microseconds:
    /// there is no deadlock to time out on, and a timeout-based assertion
    /// would pass against exactly this defect. A worker killed by a signal
    /// has no exit code, which `drive_build_worker` reports as
    /// "terminated by signal".
    ///
    /// The stderr drain keeps reading past a byte no valid UTF-8 sequence
    /// contains, so its read end stays open and the worker is never
    /// SIGPIPEd by a drain that gave up early.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drive_build_worker_survives_invalid_utf8_on_stderr() {
        // ~2 MB of stderr after the bad byte, comfortably past any pipe
        // buffer, so a closed read end is certain to be observed.
        let (_dir, script) = write_utf8_probe_worker(
            "mock_bad_utf8_stderr.sh",
            "#!/bin/sh\n\
             echo 'warning: locale not set' >&2\n\
             printf '\\377\\n' >&2\n\
             s=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\
             s=$s$s$s$s\n\
             s=$s$s$s$s\n\
             s=$s$s$s$s\n\
             i=0\n\
             while [ $i -lt 1000 ]; do printf '%s\\n' \"$s\" >&2; i=$((i+1)); done\n\
             echo 'PROGRESS percent=0.95 info=Finalizing'\n\
             exit 0\n",
        );

        let engine = engine_ready_to_build();
        let cancel = engine.install_cancel_handle();
        let outcome = build::drive_build_worker(
            &engine,
            cancel,
            Path::new("/bin/sh"),
            &script,
            "demo",
            "/tmp/demo",
            false,
        )
        .await;
        match outcome {
            build::BuildOutcome::Completed => {}
            build::BuildOutcome::Failed(msg) => {
                panic!("expected Completed, got Failed({msg:?})")
            }
            other => panic!("expected Completed, got {}", other_disc(&other)),
        }
    }
}
