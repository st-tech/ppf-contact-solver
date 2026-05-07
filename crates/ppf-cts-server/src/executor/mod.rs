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
mod solver;

// Pick the right solver-busy variant per build. The emulated build
// runs under the test rig where many workers share the same host;
// using the global scan would let one worker's solver kill another's
// (the historical `Utils.busy` patch from server/emulator.py addressed
// the same race on the python side).
#[cfg(feature = "emulated")]
use ppf_cts_core::utils::solver_busy_descendants_only as solver_busy_for_check;
#[cfg(not(feature = "emulated"))]
use ppf_cts_core::utils::solver_busy as solver_busy_for_check;

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
/// dispatch through tokio tasks; the real build body is being
/// ported in parallel and currently runs a placeholder loop.
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
                if solver_busy_for_check() {
                    log::info!(target: "ppf::solver", "DoKillSolver: terminating active solver");
                    ppf_cts_core::utils::terminate_solver();
                } else {
                    log::debug!(target: "ppf::solver", "DoKillSolver: no solver running");
                }
            }
            Effect::DoSpawnBuild => {
                build::spawn_build_task(engine);
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
        let effects = engine.dispatch(Event::BuildRequested);
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
        let effects = engine.dispatch(Event::BuildRequested);
        for fx in effects {
            exec.execute(fx, &engine).await;
        }
        assert_eq!(engine.state().build, Build::Building);

        // Wait long enough for the placeholder pipeline (4 stages
        // x ~100ms) plus a margin. On systems without a GPU the
        // pipeline short-circuits to GpuCheckFailed and lands in
        // Failed; either is an acceptable terminal state for the
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
        let effects = engine.dispatch(Event::BuildRequested);
        for fx in effects {
            exec.execute(fx, &engine).await;
        }
        // Cancel almost immediately.
        tokio::time::sleep(Duration::from_millis(20)).await;
        engine.cancel_active_build();

        // The placeholder pipeline polls the cancel token between
        // stages; cancellation should land within a tick or two.
        // On a system without a GPU the build may instead exit via
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
}
