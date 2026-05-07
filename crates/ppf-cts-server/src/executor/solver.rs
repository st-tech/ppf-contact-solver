// File: crates/ppf-cts-server/src/executor/solver.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Solver-launch plumbing for `DoLaunchSolver`. Mirrors
// `frontend/_session_.py` session.start: build a shell command from
// the per-project run script, run it detached with cwd pinned to the
// project root, redirect stdout/stderr to log files inside the
// session dir.

use std::path::PathBuf;

use ppf_cts_core::events::Event;
use ppf_cts_formats::files::{FINISHED, SAVE_AND_QUIT};

use super::{dispatch_re_entrant, solver_busy_for_check};
use crate::engine::ServerEngine;

/// Spawn the solver subprocess. Build a shell command from the
/// per-project `command.sh` (or `command.bat` on Windows), run it
/// detached with cwd pinned to the project root, redirect
/// stdout/stderr to log files inside the session dir.
///
/// `resume_from` maps to the `--load N` argument:
///   * None        -> --load 0       (fresh start, scrub export dir)
///   * Some(-1)    -> --load -1      (resume from latest checkpoint)
///   * Some(n)     -> --load n       (resume from specific frame)
pub(super) async fn launch_solver(engine: &ServerEngine, resume_from: Option<i32>) {
    let state = engine.state();
    let root = state.root.clone();
    if root.is_empty() {
        log::error!(target: "ppf::solver", "DoLaunchSolver: no project root set");
        dispatch_re_entrant(
            engine,
            Event::ErrorOccurred {
                error: "no project root set".into(),
            },
        )
        .await;
        return;
    }

    // Make sure no leftover sentinel from a previous run trips the
    // monitor before the new solver has a chance to write a frame.
    // The chain scenarios (run → fetch → run) hit exactly this:
    // ``finished.txt`` from run #1 is still on disk when run #2's
    // first monitor tick fires, and we'd dispatch SolverFinished a
    // few ms after spawn.
    let session_dir = PathBuf::from(&root).join("session");
    let output_dir = session_dir.join("output");
    for sentinel in [SAVE_AND_QUIT, FINISHED] {
        let p = output_dir.join(sentinel);
        if p.exists() {
            let _ = std::fs::remove_file(&p);
        }
    }

    // Kill any stragglers before spawning a fresh solver.
    if solver_busy_for_check() {
        log::info!(target: "ppf::solver", "DoLaunchSolver: terminating prior solver before relaunch");
        ppf_cts_core::utils::terminate_solver();
    }

    let load = resume_from.unwrap_or(0);
    let log_path = session_dir.join("stdout.log");
    let err_path = session_dir.join("error.log");
    if let Err(e) = std::fs::create_dir_all(&session_dir) {
        log::error!(target: "ppf::solver", "DoLaunchSolver: failed to mkdir {session_dir:?}: {e}");
        dispatch_re_entrant(
            engine,
            Event::ErrorOccurred {
                error: format!("session dir create failed: {e}"),
            },
        )
        .await;
        return;
    }

    // Build the command. On Windows the run script is a .bat file
    // invoked directly; on Unix we hand it to bash so the script
    // doesn't need its execute bit set.
    #[cfg(target_os = "windows")]
    let (program, args, cmd_path): (&str, Vec<String>, PathBuf) = {
        let cmd_path = session_dir.join("command.bat");
        (
            "cmd",
            vec![
                "/C".into(),
                cmd_path.to_string_lossy().to_string(),
                // Use ``--load=N`` (single token) instead of two
                // separate args: the solver's clap parser treats a
                // bare ``-1`` as a flag and rejects it ("Found
                // argument '-1' which wasn't expected").
                format!("--load={}", load),
            ],
            cmd_path,
        )
    };
    #[cfg(not(target_os = "windows"))]
    let (program, args, cmd_path): (&str, Vec<String>, PathBuf) = {
        let cmd_path = session_dir.join("command.sh");
        (
            "bash",
            vec![
                cmd_path.to_string_lossy().to_string(),
                // Use ``--load=N`` (single token) instead of two
                // separate args: the solver's clap parser treats a
                // bare ``-1`` as a flag and rejects it ("Found
                // argument '-1' which wasn't expected").
                format!("--load={}", load),
            ],
            cmd_path,
        )
    };

    if !cmd_path.exists() {
        let msg = format!("solver run script not found: {}", cmd_path.display());
        log::error!(target: "ppf::solver", "DoLaunchSolver: {msg}");
        dispatch_re_entrant(engine, Event::ErrorOccurred { error: msg }).await;
        return;
    }

    let stdout_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => {
            log::error!(target: "ppf::solver", "DoLaunchSolver: stdout log open failed: {e}");
            dispatch_re_entrant(
                engine,
                Event::ErrorOccurred {
                    error: format!("stdout open: {e}"),
                },
            )
            .await;
            return;
        }
    };
    let stderr_file = match std::fs::File::create(&err_path) {
        Ok(f) => f,
        Err(e) => {
            log::error!(target: "ppf::solver", "DoLaunchSolver: stderr log open failed: {e}");
            dispatch_re_entrant(
                engine,
                Event::ErrorOccurred {
                    error: format!("stderr open: {e}"),
                },
            )
            .await;
            return;
        }
    };

    let mut cmd = tokio::process::Command::new(program);
    cmd.args(&args)
        .current_dir(&root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(stdout_file))
        .stderr(std::process::Stdio::from(stderr_file));

    // Detach so a server shutdown doesn't tear down the long-running
    // solver. On Unix, start_new_session(true) puts the child in its
    // own session group so a Ctrl-C on the server doesn't propagate.
    #[cfg(unix)]
    {
        // tokio::process::Command re-exports the std builder; this
        // is the same flag used by frontend/_session_.py's
        // start_new_session=True branch.
        cmd.process_group(0);
    }

    log::info!(
        target: "ppf::solver",
        "DoLaunchSolver: spawning {program} {args:?} (cwd={root}, load={load})"
    );
    match cmd.spawn() {
        Ok(child) => {
            log::info!(target: "ppf::solver", "DoLaunchSolver: solver pid={:?}", child.id());
            // Detach: drop the Child without awaiting. The monitor
            // task is the authoritative source of solver lifecycle
            // events (it watches finished.txt + error.log).
            std::mem::drop(child);
        }
        Err(e) => {
            log::error!(target: "ppf::solver", "DoLaunchSolver: spawn failed: {e}");
            dispatch_re_entrant(
                engine,
                Event::ErrorOccurred {
                    error: format!("spawn: {e}"),
                },
            )
            .await;
        }
    }
}
