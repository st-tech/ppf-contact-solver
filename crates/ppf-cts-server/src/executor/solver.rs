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
use ppf_cts_formats::files::{
    session_dir as session_dir_for, session_output_dir, CRASH_SIGNAL, ERROR_LOG, FINISHED,
    SAVE_AND_QUIT, STATUS_RECORD, STDOUT_LOG, TERMINATE_REQUEST,
};

use super::{dispatch_re_entrant, solver_busy_for_check, terminate_solver_for_kill};
use crate::engine::ServerEngine;

/// The reaped exit status of the launcher process, kept for the monitor.
///
/// The direct child is `bash` (or `cmd`), not the solver, so this is
/// corroborating evidence and never the lifecycle source: the monitor's
/// reading of `status.cbor` stays authoritative. What it adds is the ONLY
/// witness to an uncatchable kill. A `SIGKILL`ed solver writes nothing, and
/// its launcher reports `128 + 9`, which is the whole reason the status is
/// retained rather than dropped with the `Child`.
pub(crate) mod exit_watch {
    use std::process::ExitStatus;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::time::SystemTime;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct SolverExit {
        pub code: Option<i32>,
        pub signal: Option<i32>,
        /// When the launcher's status was observed.
        ///
        /// `SystemTime` rather than a monotonic `Instant` because the only
        /// thing this is ever compared against is the modification time of
        /// the solver's own `status.cbor`, which is a `SystemTime` from the
        /// same host clock. A backward clock step between the two makes a
        /// contemporaneous status read as older than the record, and the
        /// monitor then declines to quote it, which is the safe direction.
        pub reaped_at: SystemTime,
    }

    /// Identity of ONE launch: the project root plus a monotonic generation.
    ///
    /// The root alone cannot identify a launch, because every launch of a
    /// project carries the same one and the server holds one project at a
    /// time. Terminate-then-Run produces two launches under that same root
    /// with their waits overlapping, so a root-keyed write lets the first
    /// launcher's status land in the second launch's slot. The generation is
    /// what [`record`] and [`release`] check, so a stale launcher can only
    /// write to a slot that is still its own.
    ///
    /// Named for the launcher process, not for the solver's own `launch_id`
    /// (the 12-hex stamp in `RunStatus`), which is a different identity with
    /// a different lifetime.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct LaunchToken {
        pub root: String,
        generation: u64,
    }

    struct Slot {
        launch: LaunchToken,
        exit: Option<SolverExit>,
    }

    /// Holds at most one entry: [`arm`] claims the slot for the launch about
    /// to be spawned, which clears whatever the previous launch left, and
    /// [`disarm`] / [`release`] free it when there is no launcher of ours
    /// left to wait for. Without that second half the slot outlives its
    /// launch, and either a later externally-started run reads the previous
    /// run's exit status as its own, or the slot stays armed with nothing to
    /// record and pins the monitor's liveness witness at "still starting".
    static SLOT: Mutex<Option<Slot>> = Mutex::new(None);

    /// Never reused within a process, so a token from a launch that has been
    /// superseded can never match the slot again.
    static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

    fn slot() -> std::sync::MutexGuard<'static, Option<Slot>> {
        SLOT.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Claim the slot immediately before spawning, so a stale status from an
    /// earlier launch is never attributed to this one. The returned token is
    /// this launch's identity; the task that reaps its launcher must carry it.
    pub(crate) fn arm(root: &str) -> LaunchToken {
        let launch = LaunchToken {
            root: root.to_string(),
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
        };
        *slot() = Some(Slot {
            launch: launch.clone(),
            exit: None,
        });
        launch
    }

    /// Release the slot for `root`, whichever launch holds it. The monitor
    /// calls this when it adopts a run this server did not spawn: whatever
    /// status the slot holds belongs to an earlier launch of that same
    /// project, and it must not be offered as evidence about the run now in
    /// progress.
    pub(crate) fn disarm(root: &str) {
        let mut g = slot();
        if g.as_ref().is_some_and(|s| s.launch.root == root) {
            *g = None;
        }
    }

    /// Release the slot held by `launch`, and only by `launch`.
    ///
    /// Called when a launch will never produce a status: its spawn failed, or
    /// its launcher could not be reaped. Leaving it armed keeps [`snapshot`]
    /// answering "armed, nothing recorded", which the monitor reads as a
    /// solver that is still starting, for the rest of the process's life.
    /// Generation-scoped, so a launch that has already been superseded cannot
    /// release its successor's slot.
    pub(crate) fn release(launch: &LaunchToken) {
        let mut g = slot();
        if g.as_ref().is_some_and(|s| s.launch == *launch) {
            *g = None;
        }
    }

    /// Record the launcher's reaped status, if the slot still belongs to this
    /// exact launch. A relaunch during the wait re-armed it for a later one.
    pub(crate) fn record(launch: &LaunchToken, status: ExitStatus) {
        let mut g = slot();
        if let Some(s) = g.as_mut() {
            if s.launch == *launch {
                #[cfg(unix)]
                let signal = {
                    use std::os::unix::process::ExitStatusExt as _;
                    status.signal()
                };
                #[cfg(not(unix))]
                let signal = None;
                s.exit = Some(SolverExit {
                    code: status.code(),
                    signal,
                    reaped_at: SystemTime::now(),
                });
            }
        }
    }

    /// The slot's whole state for `root` from ONE read: `(status, armed)`.
    ///
    /// Reading without consuming, so a monitor tick that runs twice before the
    /// state settles sees the same evidence.
    ///
    /// A caller that needs both must take them together. Read separately, a
    /// status landing between the two calls reports a launcher that has been
    /// reaped alongside the `None` that says it has not, and the monitor then
    /// opens a verdict on a witness it cannot quote.
    ///
    /// `armed` is what separates the two readings of a `None` status. Armed
    /// means this server spawned a launcher for that project and it has not
    /// been reaped yet, so the run is still going; unarmed means there is no
    /// launcher of ours to wait for, and liveness has to come from somewhere
    /// else.
    pub(crate) fn snapshot(root: &str) -> (Option<SolverExit>, bool) {
        match slot().as_ref() {
            Some(s) if s.launch.root == root => (s.exit.clone(), true),
            _ => (None, false),
        }
    }
}

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
    // The chain scenarios (run → fetch → run) hit exactly this: a
    // terminal ``status.cbor`` from run #1 is still on disk when run #2's
    // first monitor tick fires, and we'd dispatch SolverFinished a
    // few ms after spawn.
    let root_path = PathBuf::from(&root);
    let session_dir = session_dir_for(&root_path);
    let output_dir = session_output_dir(&root_path);
    // STATUS_RECORD / TERMINATE_REQUEST / CRASH_SIGNAL are scrubbed for the
    // same reason: a stale status.cbor, terminate_request or crash_signal from
    // a prior run must not be read by the monitor before the fresh solver
    // writes its own. STATUS_LOCK is deliberately NOT scrubbed: the
    // about-to-spawn solver owns it, and an advisory lock left by a dead prior
    // run is already released by the OS.
    for sentinel in [
        SAVE_AND_QUIT,
        FINISHED,
        STATUS_RECORD,
        TERMINATE_REQUEST,
        CRASH_SIGNAL,
    ] {
        let p = output_dir.join(sentinel);
        if p.exists() {
            let _ = std::fs::remove_file(&p);
        }
    }

    // Kill any stragglers before spawning a fresh solver.
    if solver_busy_for_check() {
        log::info!(target: "ppf::solver", "DoLaunchSolver: terminating prior solver before relaunch");
        terminate_solver_for_kill();
    }

    let load = resume_from.unwrap_or(0);
    let log_path = session_dir.join(STDOUT_LOG);
    let err_path = session_dir.join(ERROR_LOG);
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
        let cmd_path = session_dir.join(ppf_cts_formats::files::COMMAND_BAT);
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
        let cmd_path = session_dir.join(ppf_cts_formats::files::COMMAND_SH);
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
    let launch = exit_watch::arm(&root);
    match cmd.spawn() {
        Ok(mut child) => {
            log::info!(target: "ppf::solver", "DoLaunchSolver: solver pid={:?}", child.id());
            // Reap the launcher in the background and keep its exit status.
            // The monitor stays the authoritative lifecycle source (it reads
            // the solver-authored status.cbor); this status is corroborating
            // evidence, and the only witness left when the solver is killed by
            // a signal it cannot catch. Awaiting does not kill the child
            // (`kill_on_drop` is false) and `process_group(0)` above already
            // provides the signal isolation the detach is for, so the solver
            // still outlives a server shutdown.
            tokio::spawn(async move {
                let root = &launch.root;
                match child.wait().await {
                    Ok(status) => {
                        log::info!(
                            target: "ppf::solver",
                            "solver launcher for {root} exited: {status}"
                        );
                        exit_watch::record(&launch, status);
                    }
                    Err(e) => {
                        log::warn!(
                            target: "ppf::solver",
                            "could not reap the solver launcher for {root}: {e}"
                        );
                        // No status will ever arrive for this launch. Hand the
                        // slot back so the monitor's liveness witness falls
                        // through to the process scan; an armed slot with
                        // nothing to record reads as "still starting" forever.
                        exit_watch::release(&launch);
                    }
                }
            });
        }
        Err(e) => {
            // Same reason as the unreapable-launcher arm above: nothing was
            // spawned, so nothing will ever be recorded against this slot.
            exit_watch::release(&launch);
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
