// File: crates/ppf-cts-server/src/monitor.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Background tokio task that polls the project and dispatches
// solver-monitor events into the engine. Classification reads the
// solver-authored `status.cbor` record (`ppf_cts_formats::status`) as the
// single source of truth: phase + terminal Outcome drive
// frame/initialized/saving/finished/crashed, and a missing terminal
// Outcome with the owning process confirmed dead (the liveness lock is
// free AND the owning pid is gone, via `status::lock`) is an abrupt crash
// by construction. Liveness is judged from this project's own `status.lock`
// and the owning pid, never from the global `solver_busy` scan, because that
// scan matches any host process named "ppf-contact" and a neighboring
// checkout's solver would otherwise suppress this project's verdict. The one
// window those two cannot cover is the stretch between exec and
// `status_writer::init`, where neither a record nor a lock file exists yet;
// there the witness is `exit_watch`, the reaped status of the launcher this
// server spawned. The scan is left as the fallback for a run this server did
// not spawn, which has no launcher of ours to wait for.
// `project_resumable` reports whether a `state_<N>.bin.gz` checkpoint exists.
//
// When a run dies without a terminal outcome, `classify_abrupt` names the
// cause from a closed set of witnesses (the solver's own signal record and
// the launcher's exit status) and reports "unknown, here are the raw facts"
// for anything else. It deliberately cannot read the logs.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ppf_cts_core::datamodel::project_resumable;
use ppf_cts_core::events::Event;
use ppf_cts_core::state::{Build, Data, Solver};
use ppf_cts_formats::files::{
    session_dir as session_dir_for, session_output_dir, DATA_PICKLE, ERROR_LOG,
    INTERSECTION_RECORDS_JSON, PARAM_PICKLE, STDOUT_LOG, TERMINATE_REQUEST,
};
use ppf_cts_formats::status::{self, lock, signal_name, signal_sidecar, CrashKind, Outcome, Phase};
use ppf_cts_formats::FormatError;
// Test rig spawns peer workers as sibling processes; the
// emulated-feature build narrows the busy check to descendants only
// so a foreign worker's solver doesn't trip our liveness watchdog.
#[cfg(feature = "emulated")]
use ppf_cts_core::utils::solver_busy_descendants_only as solver_busy;
#[cfg(not(feature = "emulated"))]
use ppf_cts_core::utils::solver_busy;
use serde::Deserialize;

use crate::engine::ServerEngine;
use crate::executor::{dispatch_with_executor, EffectExecutor};

/// Spawn the monitor task. Returns a `JoinHandle` so the caller can
/// await it on shutdown. The task runs forever; drop the handle
/// (or call `abort`) to tear it down.
pub fn spawn_monitor(
    engine: ServerEngine,
    executor: Arc<dyn EffectExecutor>,
) -> tokio::task::JoinHandle<()> {
    let interval_ms = engine.config().monitor_interval_ms;
    let grace_ms = engine.config().solver_startup_grace_ms;
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut ctx = MonitorContext::default();
        loop {
            ticker.tick().await;
            if let Err(e) = tick(&engine, executor.as_ref(), &mut ctx, grace_ms).await {
                log::error!(target: "ppf::monitor", "monitor tick failed: {e}");
            }
        }
    })
}

#[derive(Default)]
struct MonitorContext {
    last_solver_state: Solver,
    /// `tokio::time::Instant` is monotonic and avoids wall-clock
    /// jumps. None until the solver enters Running for the first
    /// time.
    solver_started_at: Option<tokio::time::Instant>,
}

async fn tick(
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    ctx: &mut MonitorContext,
    grace_ms: u64,
) -> Result<(), MonitorError> {
    let mut s = engine.state();

    // External-solver adoption. If the engine thinks the solver is
    // idle but a `ppf-contact` process is live under our project
    // root, JupyterLab (or any other notebook) launched a run that
    // didn't go through our effect pipeline. Promote state to
    // `Running` so the rest of this tick drives frame / finish /
    // crash transitions normally; otherwise the addon's status panel
    // would stay frozen at "Ready to Run" for the entire run. The
    // build-pipeline guard skips adoption while we're already running
    // a build (the build subprocess can briefly look like a busy
    // ppf-contact descendant on some platforms).
    if s.solver == Solver::Idle
        && s.build != Build::Building
        && !s.root.is_empty()
        && solver_busy()
    {
        let root = PathBuf::from(&s.root);
        let has_data_on_disk = root.join(DATA_PICKLE).exists()
            && root.join(PARAM_PICKLE).exists();
        // The run in progress was not spawned here, so any launcher status
        // still in the slot belongs to an earlier launch of this same
        // project. Release it: an absent witness reports "no exit status was
        // captured", which is the honest answer, while a stale one would be
        // quoted as this run's exit code.
        crate::executor::solver::exit_watch::disarm(&s.root);
        dispatch_with_executor(
            engine,
            executor,
            Event::ExternalSolverAdopted { has_data_on_disk },
        )
        .await;
        s = engine.state();
        // Past-date the start instant so the grace check below treats
        // the long-running external solver as already past its window.
        ctx.solver_started_at = Some(
            tokio::time::Instant::now()
                .checked_sub(Duration::from_millis(grace_ms.saturating_add(1000)))
                .unwrap_or_else(tokio::time::Instant::now),
        );
        ctx.last_solver_state = s.solver;
    }

    // Post-mortem external adoption. Mirror of the live-adoption block
    // above for the case where an external run (typically command.sh /
    // a JupyterLab notebook) has already reached a terminal outcome by
    // the time the engine first sees the root: the process is gone so
    // solver_busy() is false, but its terminal status.cbor is on disk.
    // Reading the record (instead of finished.txt) also reports a
    // crashed external run as Failed instead of silently finished.
    // Guard with state.frame == 0 so we fire exactly once per fresh
    // adoption.
    if s.solver == Solver::Idle
        && s.build != Build::Building
        && !s.root.is_empty()
        && s.data == Data::Uploaded
        && s.frame == 0
    {
        let root = PathBuf::from(&s.root);
        if let Ok(Some(rec)) = status::read(&output_dir(&root)) {
            if let Some(outcome) = rec.outcome {
                if rec.frame > 0 {
                    dispatch_with_executor(
                        engine,
                        executor,
                        Event::SolverFrameUpdated { frame: rec.frame },
                    )
                    .await;
                }
                match outcome {
                    Outcome::Crashed { sub_kind, detail } => {
                        report_crash_kind(engine, executor, ctx, &root, sub_kind, &detail).await?;
                    }
                    _ => {
                        let resumable = rec.resumable || project_resumable(&root);
                        dispatch_with_executor(
                            engine,
                            executor,
                            Event::SolverFinished { resumable },
                        )
                        .await;
                    }
                }
                s = engine.state();
                ctx.last_solver_state = s.solver;
            }
        }
    }

    // Track the running edge so we can apply the startup grace
    // period before declaring a solver dead.
    if matches!(s.solver, Solver::Running | Solver::Saving)
        && ctx.last_solver_state == Solver::Idle
    {
        ctx.solver_started_at = Some(tokio::time::Instant::now());
    }
    ctx.last_solver_state = s.solver;

    // Only poll when the engine thinks the solver is active.
    if !matches!(s.solver, Solver::Running | Solver::Saving) {
        // Clear the grace timer so the NEXT launch (e.g. the second
        // run in a chain scenario) gets a fresh window. Without this,
        // a Running → Idle → Running cycle that completes between
        // two monitor ticks loses the Idle observation, and the new
        // Running keeps the stale solver_started_at from the prior
        // run -- which is already past the grace window, so the busy
        // check below trips on the still-spawning solver and
        // dispatches SolverFinished instantly.
        ctx.solver_started_at = None;
        return Ok(());
    }
    if s.root.is_empty() {
        return Ok(());
    }
    let root = PathBuf::from(&s.root);
    let out_dir = output_dir(&root);
    let elapsed_ok = ctx
        .solver_started_at
        .map(|t| t.elapsed() >= Duration::from_millis(grace_ms))
        .unwrap_or(false);

    // Everything this server knows about the launcher it spawned for this
    // project, if it spawned one, plus whether the launched process is
    // provably gone. Evaluated lazily at each use, and never before the
    // cheaper conditions beside it: the fallback inside [`launcher_witness`]
    // walks every process on the host.
    let read_launcher = || launcher_witness(&s.root);

    // The solver-authored status record (status.cbor) is the single source
    // of truth. A clean exit ALWAYS writes a terminal Outcome, so a missing
    // terminal plus the owning process confirmed dead (the liveness lock is
    // free AND the owning pid is gone) is an abrupt crash by construction.
    // No log scraping, no sentinel files, no substring tables.
    match status::read(&out_dir) {
        Ok(Some(rec)) => {
            // Progress + the in-process initialize() finish come straight
            // from the record the solver updates as it runs.
            if rec.frame != s.frame {
                dispatch_with_executor(
                    engine,
                    executor,
                    Event::SolverFrameUpdated { frame: rec.frame },
                )
                .await;
            }
            if !s.initialized && rec.phase != Phase::Starting {
                dispatch_with_executor(engine, executor, Event::SolverInitialized).await;
            }
            match rec.outcome {
                Some(Outcome::Finished) => {
                    finish_solver(engine, executor, ctx, &root, false).await;
                }
                Some(Outcome::SavedAndQuit) => {
                    finish_solver(engine, executor, ctx, &root, true).await;
                }
                // Intentional terminate, or an opaque-but-clean terminal
                // stop written by a newer solver this build does not
                // recognize: never a crash.
                Some(Outcome::Terminated { .. }) => {
                    finish_solver(engine, executor, ctx, &root, false).await;
                }
                Some(Outcome::Unknown { ref raw_kind }) => {
                    // A terminal outcome a newer solver wrote and this build
                    // does not recognize. Treated as a clean stop, but named
                    // rather than accepted in silence: an unrecognized
                    // terminal outcome means the two halves are out of sync.
                    log::warn!(
                        target: "ppf::monitor",
                        "status.cbor carries terminal outcome '{raw_kind}', which this \
                         build does not recognize; treating it as a clean stop"
                    );
                    finish_solver(engine, executor, ctx, &root, false).await;
                }
                Some(Outcome::Crashed { sub_kind, detail }) => {
                    report_crash_kind(engine, executor, ctx, &root, sub_kind, &detail).await?;
                }
                None => {
                    // No terminal outcome yet: live, or died abruptly. The
                    // lock and the owning PID are the crux, and the check is
                    // PID-scoped so a second unrelated solver (e.g. another
                    // run sharing the same host) cannot suppress it.
                    //
                    // The pid is the one witness here that goes stale on its
                    // own: pids recycle (32768 by default on Linux) and a run
                    // here is routinely longer than the wrap, so a record can
                    // name a pid that now belongs to something else entirely.
                    // Left as a bare disjunct it outvotes both fresh witnesses
                    // and the run never reaches any verdict. So a reaped
                    // launcher of ours overrides it.
                    //
                    // Only that witness, never the `gone` half of
                    // `launcher_witness`: its process-scan fallback is a weak
                    // NEGATIVE that misses a live solver this server did not
                    // spawn, and overriding a live pid with it would report a
                    // crash on a running solver, which is far worse than the
                    // stale reading it fixes. A run with no launcher of ours
                    // keeps the pid as its witness.
                    let alive = lock::is_held_by_other(&out_dir)
                        || (lock::pid_alive(rec.pid) && !our_launcher_was_reaped(&s.root));
                    if alive {
                        if rec.phase == Phase::Saving && s.solver == Solver::Running {
                            dispatch_with_executor(engine, executor, Event::SolverSaving)
                                .await;
                        }
                    } else if elapsed_ok {
                        let (launcher, launcher_gone) = read_launcher();
                        if !launcher_gone {
                            // The launcher this server spawned has not been
                            // reaped, and that is the window in which an
                            // uncatchable kill has no witness at all: the
                            // solver releases its lock and vacates its pid the
                            // instant it dies, while the launcher's `137` still
                            // has to travel through bash, tokio's reaper and
                            // `exit_watch::record`. Sealing here would write
                            // "the cause is not recorded" as the run's terminal
                            // outcome milliseconds before the cause arrives,
                            // and the verdict is never revisited.
                        } else if terminate_intended(&out_dir, engine) {
                            // `terminate_request` is the only evidence of
                            // intent there is, and this server writes it itself
                            // before killing. A recorded SIGTERM is NOT
                            // evidence of intent: it is the default kill of
                            // every supervisor, admin script and batch
                            // scheduler, so reporting one as a finish would
                            // tell the user a run completed at whatever frame
                            // an outside kill stopped it at.
                            //
                            // With the request, an intentional stop whose host
                            // left no terminal record: a hard kill, a Windows
                            // uncatchable terminate, or the mid-tick race the
                            // in-memory Idle covers. Clean, never a crash.
                            finish_solver(engine, executor, ctx, &root, false).await;
                        } else {
                            // Without the request the stop is classified like
                            // any other abrupt death, which names SIGTERM and
                            // the frame it reached.
                            let witness = launcher
                                .as_ref()
                                .filter(|e| launcher_witnessed_this_death(&out_dir, e));
                            let evidence = AbruptEvidence {
                                signal: signal_sidecar::read(&out_dir, &rec.launch_id),
                                launcher_code: witness.and_then(|e| e.code),
                                launcher_signal: witness.and_then(|e| e.signal),
                                pid: Some(rec.pid),
                                frame: Some(rec.frame),
                            };
                            let (kind, detail) = classify_abrupt(&evidence);
                            seal_abrupt_crash(&out_dir, &rec, kind, &detail);
                            report_crash_kind(engine, executor, ctx, &root, kind, &detail)
                                .await?;
                        }
                    }
                    // else: still live, or the grace has not elapsed.
                }
            }
        }
        Ok(None) => {
            // No record yet: the solver process is spawned but has not
            // reached status_writer::init. Claiming a launch failure needs a
            // witness that the launched process is GONE, and this project's
            // own lock is not one before init: the lock file is created
            // inside init, and `is_held_by_other` reads a missing file as
            // free, so the whole exec-to-init window would read as dead. That
            // window is the entire subject of this branch, and it is not
            // short: it covers the shell launcher, the dynamic load of the
            // CUDA runtime and libsimbackend_cuda, and `setup()`'s wipe of a
            // prior run's output directory. `launcher_witness` supplies the
            // missing witness, and it is read last because it is the only
            // condition here that can walk the host's process table.
            //
            // The lock stays in the condition as the project-scoped half: the
            // process scan matches any host process named "ppf-contact", so a
            // neighboring checkout's solver would otherwise suppress this
            // project's launch failure.
            if elapsed_ok
                && !lock::is_held_by_other(&out_dir)
                && !terminate_intended(&out_dir, engine)
            {
                let (launcher, launcher_gone) = read_launcher();
                if launcher_gone {
                    let (kind, detail) = launch_failure_verdict(launcher.as_ref());
                    report_crash_kind(engine, executor, ctx, &root, kind, &detail).await?;
                }
            }
        }
        Err(FormatError::VersionMismatch { found, expected }) => {
            // Single-version fleet: should not happen. Surface it rather
            // than silently misreading a newer record.
            log::error!(
                target: "ppf::monitor",
                "status.cbor schema mismatch (found {found}, expected {expected}); \
                 solver and server are out of sync"
            );
        }
        Err(_) => {
            // Torn / zero-length record: a non-terminal record whose write
            // was interrupted. No pid is available from a failed read, so
            // the liveness verdict comes from this project's own lock rather
            // than the global process scan, which a neighboring checkout's
            // solver would trip.
            //
            // A torn record is produced by dying DURING `write_progress`, and
            // the phase most likely to be mid-write is Saving, so an engine
            // state of Saving is not evidence of a clean checkpoint here; it
            // is the most likely shape of a crash during one. Only an
            // explicit terminate intent makes this a clean stop.
            //
            // The lock is a real witness here (a torn record means init ran,
            // so the lock file exists), but a failed read yields no pid, so
            // it is the ONLY status-side witness. `launcher_witness` is the
            // second one, and it is the same one the branches above use.
            if elapsed_ok && !lock::is_held_by_other(&out_dir) {
                let (launcher, launcher_gone) = read_launcher();
                if !launcher_gone {
                    // Same window as the branch above: wait for the witness.
                } else if terminate_intended(&out_dir, engine) {
                    finish_solver(engine, executor, ctx, &root, false).await;
                } else {
                    // The record names nothing, but the launcher's status is
                    // the SAME witness the branch above reads, and it names
                    // the cause of the very death that tore the record. The
                    // sidecar is genuinely out of reach: matching it needs the
                    // launch id, which only the record carries.
                    let witness = launcher
                        .as_ref()
                        .filter(|e| launcher_witnessed_this_death(&out_dir, e));
                    let evidence = AbruptEvidence {
                        signal: None,
                        launcher_code: witness.and_then(|e| e.code),
                        launcher_signal: witness.and_then(|e| e.signal),
                        pid: None,
                        frame: None,
                    };
                    let (kind, cause) = classify_abrupt(&evidence);
                    // The addon draws this on one clipped label, so whichever
                    // fact discriminates has to lead. When the launcher named
                    // the cause that is the cause; when nothing did, the
                    // truncation is the only fact there is.
                    let detail = if kind == CrashKind::UnknownAbrupt {
                        format!(
                            "the status record is truncated, so it names neither \
                             the frame the run reached nor a cause. {cause}"
                        )
                    } else {
                        format!(
                            "{cause} The status record was torn mid-write, so it \
                             names neither the frame the run reached nor a cause."
                        )
                    };
                    seal_torn_record_crash(&out_dir, kind, &detail);
                    report_crash_kind(engine, executor, ctx, &root, kind, &detail).await?;
                }
            }
        }
    }

    Ok(())
}

/// Finish the active solver run: decide resumability (the saving path
/// always produces a resumable checkpoint by construction; otherwise
/// consult the on-disk `state_<N>.bin.gz` files via `project_resumable`),
/// dispatch `SolverFinished`, then snap the local state mirror back to
/// Idle so the next Idle → Running edge (e.g. the second run in a chain
/// scenario) fires the grace-timer reset. Both the terminal-outcome branch
/// (from the status.cbor record) and the liveness-exit branch share this
/// rule, so it lives in one
/// place. The post-mortem external-adoption block intentionally does
/// not call this: it always treats the run as resumable and skips the
/// edge-state reset.
async fn finish_solver(
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    ctx: &mut MonitorContext,
    root: &Path,
    saving: bool,
) {
    let resumable = if saving {
        true
    } else {
        project_resumable(root)
    };
    dispatch_with_executor(engine, executor, Event::SolverFinished { resumable }).await;
    ctx.last_solver_state = Solver::Idle;
    ctx.solver_started_at = None;
}

/// Report an abnormal solver exit: read any intersection violations,
/// dispatch `SolverCrashed`, then snap the local edge-state mirror back to
/// Idle (like `finish_solver`) so the next Idle -> Running edge re-arms the
/// grace timer. The durable crash record is the terminal `Crashed` in
/// `status.cbor`, written by the solver when it can and by
/// [`seal_abrupt_crash`] when it died before it could; reconnect reads that,
/// so there is no separate crash marker to write here.
async fn report_crash(
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    ctx: &mut MonitorContext,
    root: &Path,
    kind_tag: String,
    error: String,
) -> Result<(), MonitorError> {
    let violations = read_intersection_violations(root)?;
    dispatch_with_executor(
        engine,
        executor,
        Event::SolverCrashed {
            error,
            kind_tag,
            violations,
        },
    )
    .await;
    ctx.last_solver_state = Solver::Idle;
    ctx.solver_started_at = None;
    Ok(())
}

/// Render and report a crash from a structured kind. Every crash branch goes
/// through here so the addon always receives both the stable tag and the same
/// rendered report, whatever produced the verdict.
async fn report_crash_kind(
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    ctx: &mut MonitorContext,
    root: &Path,
    kind: CrashKind,
    detail: &str,
) -> Result<(), MonitorError> {
    let error = render_crash(kind, detail, root);
    // Mirror the solver's stderr into the server log so an operator reading
    // server.log sees what the addon sees. It belongs here rather than in
    // `render_crash`: this is the live-crash path and it runs once per crash,
    // whereas rendering also happens on the reconnect path, which a connected
    // addon drives several times a second.
    for line in read_log_tail(root, ERROR_LOG).lines() {
        log::warn!(target: "ppf::monitor", "[SOLVER stderr] {line}");
    }
    report_crash(engine, executor, ctx, root, kind.tag().to_string(), error).await
}

/// Write the server's own verdict into `status.cbor` as the run's terminal
/// outcome.
///
/// The solver owns the record while it lives, and this runs only once the
/// monitor has established that it does not: the liveness lock is free and
/// the owning pid is gone. [`status::write_terminal`] is first-writer-wins,
/// so a terminal outcome the solver did manage to write is never
/// overwritten.
///
/// What it buys is that a reconnect reads back the SAME cause. The launcher's
/// exit status lives only in this process's memory, so a server that has to
/// re-derive the verdict from disk alone can name strictly less than the live
/// report did: a `SIGKILL`, whose only witness is that exit status, would
/// come back as "no cause recorded". Every field but the outcome is copied
/// from the record the solver left, so the frame, pid and launch id are its
/// own and nothing here is invented.
fn seal_abrupt_crash(
    out_dir: &Path,
    rec: &status::RunStatus,
    kind: CrashKind,
    detail: &str,
) {
    let sealed = status::RunStatus {
        phase: Phase::Ended,
        outcome: Some(Outcome::Crashed {
            sub_kind: kind,
            detail: detail.to_string(),
        }),
        seq: rec.seq.saturating_add(1),
        ..rec.clone()
    };
    if let Err(e) = status::write_terminal(out_dir, &sealed) {
        log::warn!(
            target: "ppf::monitor",
            "could not seal the crash verdict into status.cbor: {e}; a \
             reconnect will re-derive the cause from what is left on disk"
        );
    }
}

/// Seal a verdict over a TORN record, which leaves nothing to copy from.
///
/// The reconnect path routes a failed read to "no crash" (there is no record
/// to reconstruct one from), so without this a run that died mid-checkpoint
/// reads as Failed until the server restarts and as a healthy project
/// afterwards, with the truncated run free to be treated as complete.
///
/// Every field except the outcome states what the torn record established:
/// the frame, pid and launch id are unknown, so they are written as the zero
/// that means unknown, which is exactly what the detail already says. The
/// emulated flag is this build's own, and it is the only field that is not a
/// property of the dead run.
fn seal_torn_record_crash(out_dir: &Path, kind: CrashKind, detail: &str) {
    let sealed = status::RunStatus {
        phase: Phase::Ended,
        frame: 0,
        sim_time: 0.0,
        resumable: false,
        outcome: Some(Outcome::Crashed {
            sub_kind: kind,
            detail: detail.to_string(),
        }),
        seq: 0,
        pid: 0,
        launch_id: String::new(),
        emulated: cfg!(feature = "emulated"),
    };
    if let Err(e) = status::write_terminal(out_dir, &sealed) {
        log::warn!(
            target: "ppf::monitor",
            "could not seal the torn-record crash verdict into status.cbor: {e}; \
             a reconnect will read the run as never having crashed"
        );
    }
}

/// Kind and detail for a run whose solver wrote no status record.
///
/// The absence of a record establishes exactly one thing: the process never
/// reached `status_writer::init`. It does not name WHAT stopped it, and the
/// launcher's exit status, which the server already holds, is the only witness
/// that can. So the status is decoded first, through the same ladder
/// [`classify_abrupt`] runs, and the report falls back to naming the absence
/// itself only for a status that names nothing.
///
/// The kind therefore varies with the evidence: a `128 + N` code or a signalled
/// launcher is a kill, a Windows loader status is a loader failure, and only a
/// status this build cannot decode is reported as a bare
/// [`CrashKind::LaunchFailed`]. An externally-adopted run has no launcher of
/// ours, and the report says so rather than naming a status it does not have.
fn launch_failure_verdict(
    launcher: Option<&crate::executor::solver::exit_watch::SolverExit>,
) -> (CrashKind, String) {
    /// What the branch's own precondition establishes, and nothing more. It
    /// does NOT name the run script, the interpreter or the loader: any of
    /// them can be the culprit, and so can the solver itself exiting before
    /// `status_writer::init` (a bad argument, a `--load` with no checkpoint).
    const NO_RECORD: &str =
        "the solver wrote no status record, so it stopped before it reached its own \
         reporting";
    let code = launcher.and_then(|e| e.code);
    let signal = launcher.and_then(|e| e.signal);

    if let Some(name) = launcher_signal_name(code, signal) {
        let how = match code {
            Some(c) => format!("the run script exited {c}"),
            None => "the run script itself carried the signal".to_string(),
        };
        return (
            CrashKind::KilledBySignal,
            format!("killed by {name} ({how}); {NO_RECORD}"),
        );
    }

    if let Some(code) = code {
        if let Some((class, name, status)) = windows_exception(code) {
            return (
                class.crash_kind(),
                format!("{status} ({name}); {NO_RECORD}"),
            );
        }
        // POSIX shell conventions for "I could not run it at all". `cmd.exe`
        // has no equivalent (it reports 9009 for a missing command), so this
        // decode is unix-only, like the signal table backing `signal_name`.
        #[cfg(unix)]
        {
            let shell = match code {
                127 => Some(
                    "the shell's code for a command, or a shared library it needs, \
                     not being found",
                ),
                126 => Some(
                    "the shell's code for a command that was found but could not be \
                     executed",
                ),
                _ => None,
            };
            if let Some(convention) = shell {
                return (
                    CrashKind::LaunchFailed,
                    format!("the run script exited {code}, {convention}"),
                );
            }
        }
        return (
            CrashKind::LaunchFailed,
            format!(
                "the run script exited 0x{code:08X} ({code}), which names no cause \
                 this build recognizes; {NO_RECORD}"
            ),
        );
    }

    (
        CrashKind::LaunchFailed,
        format!("no exit status was captured for the run script; {NO_RECORD}"),
    )
}

/// The launcher's status and whether the process launched for `root` is
/// provably gone, from ONE read of the slot: `(status, gone)`.
///
/// The two belong together. Every branch that acts on `gone` then quotes that
/// status as the cause, and the verdict it seals is never revisited, so a
/// status landing between two separate reads would open the branch on a
/// witness the report goes on to say it does not have.
///
/// For a run this server spawned, `gone` is exactly "the launcher has been
/// reaped": `exit_watch` is armed at spawn and carries a status only once
/// `wait` returned. It is precise and project-scoped. A run adopted from
/// outside never armed the slot, so there the global process scan is the only
/// liveness signal available, with the caveat that it matches any host
/// process named "ppf-contact".
///
/// This witness exists because the two the rest of the monitor relies on, the
/// status record and the liveness lock, do not exist during the stretch
/// between exec and `status_writer::init`: the lock file is created inside
/// init, and `lock::is_held_by_other` reads a missing file as free. Without
/// this, that whole window reads as a solver that already died.
fn launcher_witness(
    root: &str,
) -> (Option<crate::executor::solver::exit_watch::SolverExit>, bool) {
    let (exit, armed) = crate::executor::solver::exit_watch::snapshot(root);
    let gone = if armed { exit.is_some() } else { !solver_busy() };
    (exit, gone)
}

/// Whether the launcher this server spawned for `root` has been reaped.
///
/// A positive, project-scoped witness with NO fallback: false both while the
/// launcher is still running and for a run this server never spawned. That is
/// what separates it from the `gone` half of [`launcher_witness`], whose
/// process-scan fallback is a weak negative (it matches on a process NAME, so
/// it misses a live solver whenever the scan is narrowed or the name does not
/// match).
///
/// Used where a false positive would report a crash over a running solver, so
/// only a witness that cannot produce one is admissible.
fn our_launcher_was_reaped(root: &str) -> bool {
    let (exit, armed) = crate::executor::solver::exit_watch::snapshot(root);
    armed && exit.is_some()
}

/// True iff the run was intentionally stopped, so a non-terminal record
/// must not be reclassified as a crash. Two signals: the server wrote a
/// `terminate_request` before killing (durable, survives a reconnect), or
/// a Terminate already moved the engine to Idle. `dispatch` commits the
/// Running -> Idle transition before its DoKillSolver effect runs, so the
/// engine state is already Idle here even for a terminate that landed
/// mid-tick (the mid-tick race the interim fix handled).
fn terminate_intended(out_dir: &Path, engine: &ServerEngine) -> bool {
    out_dir.join(TERMINATE_REQUEST).exists() || engine.state().solver == Solver::Idle
}

/// Everything the supervisor can witness about a solver that died without
/// writing a terminal outcome.
///
/// The log tails are deliberately NOT fields here. `classify_abrupt` takes
/// only this struct, so it CANNOT reach the logs, which is what structurally
/// prevents a substring table from ever creeping back into the
/// classification. The caller appends the tails after the kind is chosen, as
/// evidence for a human rather than as input to a verdict.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct AbruptEvidence {
    /// Signal name the solver's own handler recorded, when it got that far.
    pub signal: Option<&'static str>,
    /// Exit code of the launcher process (`bash` / `cmd`), when reaped AND
    /// established to describe the same death (see
    /// [`launcher_witnessed_this_death`]).
    pub launcher_code: Option<i32>,
    /// Signal that killed the LAUNCHER itself, under the same condition.
    pub launcher_signal: Option<i32>,
    /// Owning pid and last emitted frame, when the record named them. A torn
    /// record names neither, and the report then omits the clause instead of
    /// printing the zeroes that stand in for "not known".
    pub pid: Option<u32>,
    pub frame: Option<i32>,
}

/// Name the crash from the evidence, or say it cannot be named.
///
/// The one rule: never guess. Every arm either has a witness that names the
/// cause or falls through to [`CrashKind::UnknownAbrupt`] with the raw facts
/// quoted verbatim. In particular a `128 + 9` launcher code is reported as
/// "killed by SIGKILL", never as "out of memory": an OOM-kill and a
/// `kill -9` are indistinguishable from here, and naming the wrong one is
/// worse than naming none.
///
/// Every arm leads with the fact that DISCRIMINATES (a signal name, an
/// `NTSTATUS`, an exit code) and follows with where that fact came from. The
/// addon panel draws this on one label, which renders a single clipped line,
/// so a detail that opens with shared boilerplate spends the whole visible
/// budget saying nothing; the full report reaches the Console either way.
pub(crate) fn classify_abrupt(e: &AbruptEvidence) -> (CrashKind, String) {
    // Empty when the record named neither, which is what a torn record leaves.
    // Both halves come from the same record, so they are known together.
    let stopped = match (e.pid, e.frame) {
        (Some(pid), Some(frame)) => format!("; pid {pid} stopped after frame {frame}"),
        _ => String::new(),
    };

    if let Some(name) = e.signal {
        return (
            CrashKind::KilledBySignal,
            format!(
                "killed by {name}{stopped}. The solver recorded the signal on \
                 its way out but could not write a terminal outcome."
            ),
        );
    }

    // No signal record. The solver installs a handler for every fatal signal
    // it CAN catch, so an uncaught one is uncatchable, and the launcher's
    // status is the only remaining witness. This is an elimination, not an
    // inference about WHY the signal was sent.
    if let Some(name) = launcher_signal_name(e.launcher_code, e.launcher_signal) {
        // `wait` reports a signalled child as a signal with no code, and a
        // POSIX shell reports a signalled child of its own as `128 + N`. The
        // two encodings carry the same fact, so the report names which
        // encoding it read and stops there: WHO the signal reached besides the
        // launcher is not observable from a single `wait` status (a group kill
        // and a targeted `pkill` on the run script produce the identical one).
        let how = match e.launcher_code {
            Some(code) => format!("the run script exited {code}"),
            None => "the run script itself carried the signal".to_string(),
        };
        return (
            CrashKind::KilledBySignal,
            format!(
                "killed by {name} ({how}){stopped}. No signal record was \
                 written; who sent it is not recorded anywhere the server can \
                 read."
            ),
        );
    }

    if let Some(code) = e.launcher_code {
        if let Some((class, name, status)) = windows_exception(code) {
            return (
                class.crash_kind(),
                format!(
                    "{status} ({name}){stopped}. No signal record was written \
                     and the run script exited with that status."
                ),
            );
        }
        return (
            CrashKind::UnknownAbrupt,
            format!(
                "the run script exited 0x{code:08X} ({code}), which names no \
                 cause this build recognizes{stopped}. No terminal outcome and \
                 no signal record were written."
            ),
        );
    }

    if let Some(n) = e.launcher_signal {
        return (
            CrashKind::UnknownAbrupt,
            format!(
                "the run script itself was killed by signal {n}, which names no \
                 cause this build recognizes{stopped}. No terminal outcome and \
                 no signal record were written."
            ),
        );
    }

    (
        CrashKind::UnknownAbrupt,
        format!(
            "No signal record was written and no exit status was captured, so \
             the cause is not recorded{stopped}."
        ),
    )
}

/// The signal a launcher status names, in either of the two encodings a
/// launcher can carry it in: `128 + N` in an exit code, the POSIX shell's
/// convention for a child it saw die of a signal, or a signalled `wait` status
/// with no code at all, which is the launcher itself dying of one.
///
/// One function so the two branches that read a launcher status cannot decode
/// it differently, which is exactly what let the same `137` read as `SIGKILL`
/// in one report and as a broken dynamic loader in another.
///
/// `signal_name`'s table is empty off unix, so a Windows exit code can never
/// be read as `128 + N`.
fn launcher_signal_name(code: Option<i32>, signal: Option<i32>) -> Option<&'static str> {
    if let Some(code) = code {
        if (128..192).contains(&code) {
            return signal_name(code - 128);
        }
        return None;
    }
    signal.and_then(signal_name)
}

/// Whether a Windows `NTSTATUS` names a fault in an image that was already
/// running, or the loader failing to bring one in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowsStatusClass {
    Fault,
    Loader,
}

impl WindowsStatusClass {
    /// The kind each class implies. Both are valid under either caller's
    /// precondition, which is why the class carries no assumption about
    /// whether the solver had started: a fault ends a running image, and a
    /// loader status names a library that could not be resolved, whether that
    /// happened at start-up or at a delay-loaded module's first use.
    fn crash_kind(self) -> CrashKind {
        match self {
            WindowsStatusClass::Fault => CrashKind::KilledBySignal,
            WindowsStatusClass::Loader => CrashKind::LibraryLoadFailed,
        }
    }
}

/// The Windows `NTSTATUS` values this build names, with what each one is.
///
/// This is a chosen SUBSET of the status space, not an enumeration of it. An
/// unlisted status falls through to the caller's unknown arm carrying its raw
/// number, and that fallback is what makes an omission safe rather than a
/// misattribution; the entries here are the ones a CUDA application on
/// Windows actually reaches.
fn windows_exception(code: i32) -> Option<(WindowsStatusClass, &'static str, &'static str)> {
    use WindowsStatusClass::{Fault, Loader};
    let named = match code as u32 {
        0xC000_0005 => (Fault, "access violation", "0xC0000005"),
        0xC000_0006 => (Fault, "in-page error", "0xC0000006"),
        0xC000_001D => (Fault, "illegal instruction", "0xC000001D"),
        0xC000_0094 => (Fault, "integer divide by zero", "0xC0000094"),
        0xC000_00FD => (Fault, "stack overflow", "0xC00000FD"),
        0xC000_0135 => (Loader, "a required DLL was not found", "0xC0000135"),
        0xC000_0139 => (Loader, "a DLL entry point was not found", "0xC0000139"),
        0xC000_0142 => (Loader, "DLL initialization failed", "0xC0000142"),
        0xC000_0374 => (Fault, "heap corruption", "0xC0000374"),
        0xC000_0409 => (Fault, "stack buffer overrun", "0xC0000409"),
        _ => return None,
    };
    Some(named)
}

/// Whether the launcher's exit status describes the same death as the solver's.
///
/// The launcher is `bash` (or `cmd`) running the solver as its child, and its
/// status is the only witness to an uncatchable kill. It is evidence about the
/// SOLVER only when the solver stopped writing first: a launcher that died
/// while the solver kept running (a `kill` aimed at the shell, or a script
/// that backgrounded the solver) names a signal that never touched the solver.
///
/// The comparison is the record's modification time against the moment the
/// status was reaped. A solver run as a foreground child cannot write after
/// its own launcher has been reaped, so the ordinary case is never refused;
/// a solver still writing afterwards proves the two are separate events.
///
/// A record whose modification time cannot be read is accepted: this filters a
/// status that is provably stale, and provides no proof either way for one it
/// cannot time.
fn launcher_witnessed_this_death(
    out_dir: &Path,
    exit: &crate::executor::solver::exit_watch::SolverExit,
) -> bool {
    let written = std::fs::metadata(out_dir.join(ppf_cts_formats::files::STATUS_RECORD))
        .and_then(|m| m.modified());
    match written {
        Ok(t) => t <= exit.reaped_at,
        Err(_) => true,
    }
}

/// Human-readable crash string for a structured `Crashed{kind, detail}`:
/// the kind summary (the single message table, replacing the two drifted
/// substring tables), the solver's own detail, and excerpts of `stdout.log`
/// and `error.log` as supplementary context (never as the classifier).
///
/// Pure: it reads files and returns a string, with no side effect a caller
/// pays for. Both the live-crash path and the reconnect path call it, and the
/// reconnect path runs on a request an addon repeats several times a second.
///
/// The stderr tail needs no anchoring: whatever a dying solver reported is by
/// construction the last thing it wrote there. `stdout.log` needs both ends,
/// see [`read_log_head_and_tail`].
pub(crate) fn render_crash(kind: CrashKind, detail: &str, root: &Path) -> String {
    let mut out = format!("{}: {detail}", kind.summary());
    let (stdout_head, stdout_tail) = read_log_head_and_tail(root, STDOUT_LOG);
    if !stdout_head.is_empty() {
        out.push_str(&format!(
            "\n--- Solver Log (first {CRASH_HEAD_LINES} lines) ---\n{stdout_head}"
        ));
    }
    if !stdout_tail.is_empty() {
        out.push_str(&format!(
            "\n--- Solver Log (last {CRASH_TAIL_LINES} lines) ---\n{stdout_tail}"
        ));
    }
    let stderr_tail = read_log_tail(root, ERROR_LOG);
    if !stderr_tail.is_empty() {
        out.push_str(&format!(
            "\n--- Solver Errors (last {CRASH_TAIL_LINES} lines) ---\n{stderr_tail}"
        ));
    }
    out
}

/// Last `CRASH_TAIL_LINES` lines of a session log, or empty if absent.
fn read_log_tail(root: &Path, filename: &str) -> String {
    let path = session_dir(root).join(filename);
    let lines = read_lines_if_exists(&path).unwrap_or_default();
    let start = lines.len().saturating_sub(CRASH_TAIL_LINES);
    lines[start..].join("\n")
}

/// Leading and trailing excerpts of a session log, as `(head, tail)`.
///
/// The solver states the environment it is about to run in before it runs
/// anything: the device it selected, its compute capability, whether it is
/// driving a display, and whether the operating system's kernel-execution
/// watchdog is armed on it. Those lines are at the HEAD of `stdout.log` by
/// construction, so a tail-only excerpt structurally cannot carry them, and
/// on every run long enough to matter it does not. The tail carries the
/// failure neighborhood. Both are needed to read a crash report.
///
/// The two slices never overlap, so a log shorter than the two budgets is
/// quoted once rather than twice.
fn read_log_head_and_tail(root: &Path, filename: &str) -> (String, String) {
    let path = session_dir(root).join(filename);
    let lines = read_lines_if_exists(&path).unwrap_or_default();
    let head_end = lines.len().min(CRASH_HEAD_LINES);
    let tail_start = lines
        .len()
        .saturating_sub(CRASH_TAIL_LINES)
        .max(head_end);
    (
        lines[..head_end].join("\n"),
        lines[tail_start..].join("\n"),
    )
}

#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// Filesystem helpers.

/// `<root>/session`, the per-project session directory.
fn session_dir(root: &Path) -> PathBuf {
    session_dir_for(root)
}

/// `<root>/session/output`, the canonical location this module keys off
/// (frame files, finished.txt, error markers, intersection records).
fn output_dir(root: &Path) -> PathBuf {
    session_output_dir(root)
}

/// How many trailing `stdout.log` lines are folded into a crash report as
/// supplementary context (never as the classifier). The crash string
/// interpolates this same const so the prose can't desync from the slice.
const CRASH_TAIL_LINES: usize = 32;

/// How many LEADING `stdout.log` lines are folded in as well. Sized to cover
/// the solver's start-of-run environment record (the CUDA device, its compute
/// capability, and the state of the operating system's kernel-execution
/// watchdog), which `initialize()` prints before any solve and which no tail
/// can reach. See [`read_log_head_and_tail`].
const CRASH_HEAD_LINES: usize = 24;

/// Read intersection records from `intersection_records.json`.
pub(crate) fn read_intersection_violations(root: &Path) -> Result<Vec<String>, MonitorError> {
    let path = output_dir(root).join(INTERSECTION_RECORDS_JSON);
    if !path.exists() {
        return Ok(vec![]);
    }
    let body = std::fs::read_to_string(&path)?;
    let parsed: IntersectionFile = serde_json::from_str(&body)?;
    if parsed.records.is_empty() {
        return Ok(vec![]);
    }
    // The Python source returns a list-of-dicts shape that the
    // response builder hands straight to JSON. The Rust state machine
    // models violations as `Vec<String>` (opaque payload), so we
    // shove the JSON-encoded record string per entry. The wire-format
    // response builder re-parses or passes through.
    let mut out: Vec<String> = Vec::with_capacity(parsed.records.len());
    for rec in parsed.records {
        out.push(serde_json::to_string(&rec)?);
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
struct IntersectionFile {
    // The on-disk JSON also carries a redundant `count`; serde ignores
    // unknown keys by default, so we only model what we read.
    #[serde(default)]
    records: Vec<serde_json::Value>,
}

fn read_lines_if_exists(path: &Path) -> std::io::Result<Vec<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(s.lines().map(str::to_owned).collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> AbruptEvidence {
        AbruptEvidence {
            pid: Some(4242),
            frame: Some(17),
            ..Default::default()
        }
    }

    #[test]
    fn abrupt_sidecar_names_the_signal() {
        let e = AbruptEvidence {
            signal: Some("SIGSEGV"),
            ..evidence()
        };
        let (kind, detail) = classify_abrupt(&e);
        assert_eq!(kind, CrashKind::KilledBySignal);
        assert!(detail.contains("SIGSEGV"), "detail was: {detail}");
        assert!(detail.contains("pid 4242"), "detail was: {detail}");
    }

    // POSIX semantics: `launcher_signal` comes from `ExitStatus::signal()`,
    // which does not exist off unix, and the `128 + N` code is a shell
    // convention that is deliberately inert on Windows so an exit of 137
    // there is not read as a signal. Both inputs are unreachable on
    // Windows, where the same call correctly answers UnknownAbrupt.
    #[cfg(unix)]
    #[test]
    fn abrupt_launcher_signal_code_is_decoded_without_naming_a_reason() {
        // 137 = 128 + 9 = SIGKILL. It is named, but WHY it was sent is not
        // recorded anywhere the server can read: an OOM-kill and a `kill -9`
        // produce exactly this, so claiming either would be a guess.
        let e = AbruptEvidence {
            launcher_code: Some(137),
            ..evidence()
        };
        let (kind, detail) = classify_abrupt(&e);
        assert_eq!(kind, CrashKind::KilledBySignal);
        assert!(detail.contains("SIGKILL"), "detail was: {detail}");
        assert!(
            !detail.to_lowercase().contains("memory"),
            "must not name a reason it cannot know: {detail}"
        );
    }

    /// The mirror of the unix test above, asserting the behavior that
    /// makes gating it correct rather than merely convenient.
    ///
    /// `128 + N` is a POSIX shell convention. A Windows process can exit
    /// 137 for its own reasons, so decoding it as SIGKILL there would name
    /// a cause that did not happen. The classifier reports unknown and
    /// carries the raw code, which is the rule the whole classifier is
    /// held to.
    #[cfg(not(unix))]
    #[test]
    fn abrupt_shell_signal_code_is_not_decoded_off_unix() {
        let e = AbruptEvidence {
            launcher_code: Some(137),
            ..evidence()
        };
        let (kind, detail) = classify_abrupt(&e);
        assert_eq!(kind, CrashKind::UnknownAbrupt);
        assert!(
            !detail.contains("SIGKILL"),
            "named a signal that has no meaning here: {detail}"
        );
        assert!(detail.contains("137"), "lost the raw code: {detail}");
    }

    #[test]
    fn abrupt_windows_exception_code_is_named() {
        let e = AbruptEvidence {
            launcher_code: Some(0xC000_0005u32 as i32),
            ..evidence()
        };
        let (kind, detail) = classify_abrupt(&e);
        assert_eq!(kind, CrashKind::KilledBySignal);
        assert!(detail.contains("0xC0000005"), "detail was: {detail}");
        assert!(detail.contains("access violation"), "detail was: {detail}");
    }

    #[test]
    fn abrupt_windows_loader_status_names_the_library_not_the_launch() {
        // This branch runs only where a status record exists, which means
        // `status_writer::init` ran and the solver DID start. Reporting a
        // launch failure there contradicts the frame the same line quotes:
        // a run cannot both have failed to start and have stopped after
        // frame 17. A delay-loaded module failing mid-run has this shape.
        let e = AbruptEvidence {
            launcher_code: Some(0xC000_0135u32 as i32),
            ..evidence()
        };
        let (kind, detail) = classify_abrupt(&e);
        assert_eq!(kind, CrashKind::LibraryLoadFailed);
        assert!(detail.contains("0xC0000135"), "detail was: {detail}");
        assert!(detail.contains("DLL"), "detail was: {detail}");
        assert!(
            !kind.summary().contains("before it started"),
            "the summary must not contradict the frame the detail quotes: {detail}"
        );
    }

    // POSIX semantics: `launcher_signal` comes from `ExitStatus::signal()`,
    // which does not exist off unix, and the `128 + N` code is a shell
    // convention that is deliberately inert on Windows so an exit of 137
    // there is not read as a signal. Both inputs are unreachable on
    // Windows, where the same call correctly answers UnknownAbrupt.
    #[cfg(unix)]
    #[test]
    fn abrupt_launcher_signal_is_named_like_a_launcher_code() {
        // `wait` reports a signalled child as a signal with no code, which is
        // the shape a kill that reached the launcher itself takes. That is the
        // same fact as `128 + N` in the other encoding, and it must not read
        // as "no exit status was captured".
        let e = AbruptEvidence {
            launcher_signal: Some(9),
            ..evidence()
        };
        let (kind, detail) = classify_abrupt(&e);
        assert_eq!(kind, CrashKind::KilledBySignal);
        assert!(detail.contains("SIGKILL"), "detail was: {detail}");
        assert!(
            !detail.contains("the cause is not recorded"),
            "must not claim ignorance of the fact it just named: {detail}"
        );
    }

    // POSIX semantics: `launcher_signal` comes from `ExitStatus::signal()`,
    // which does not exist off unix, and the `128 + N` code is a shell
    // convention that is deliberately inert on Windows so an exit of 137
    // there is not read as a signal. Both inputs are unreachable on
    // Windows, where the same call correctly answers UnknownAbrupt.
    #[cfg(unix)]
    #[test]
    fn abrupt_details_lead_with_what_discriminates() {
        // The addon panel draws this on one clipped label, so the fact that
        // names the cause has to survive the cut. 96 characters is the panel's
        // budget; assert against a stricter prefix so a rewording that buries
        // the discriminator fails here rather than in a screenshot.
        for e in [
            AbruptEvidence {
                signal: Some("SIGSEGV"),
                ..evidence()
            },
            AbruptEvidence {
                launcher_code: Some(137),
                ..evidence()
            },
            AbruptEvidence {
                launcher_signal: Some(9),
                ..evidence()
            },
        ] {
            let (_, detail) = classify_abrupt(&e);
            let head: String = detail.chars().take(40).collect();
            assert!(
                head.contains("SIGSEGV") || head.contains("SIGKILL"),
                "the signal name must be in the first 40 characters: {detail}"
            );
        }
        let (_, detail) = classify_abrupt(&AbruptEvidence {
            launcher_code: Some(0xC000_0005u32 as i32),
            ..evidence()
        });
        assert!(
            detail.starts_with("0xC0000005"),
            "the NTSTATUS must lead: {detail}"
        );
    }

    fn exit(code: Option<i32>, signal: Option<i32>)
        -> crate::executor::solver::exit_watch::SolverExit {
        crate::executor::solver::exit_watch::SolverExit {
            code,
            signal,
            reaped_at: std::time::SystemTime::now(),
        }
    }

    #[test]
    fn launch_failure_decodes_the_launcher_status_before_naming_a_culprit() {
        // The absence of a record proves only that the solver never reached
        // `status_writer::init`. What stopped it is in the launcher's status,
        // and each shape gets the kind its own evidence supports.
        #[cfg(unix)]
        {
            let (kind, d) = launch_failure_verdict(Some(&exit(Some(127), None)));
            assert_eq!(kind, CrashKind::LaunchFailed);
            assert!(d.contains("127"), "detail was: {d}");
            let (kind, d) = launch_failure_verdict(Some(&exit(Some(126), None)));
            assert_eq!(kind, CrashKind::LaunchFailed);
            assert!(d.contains("126"), "detail was: {d}");
        }
        // A loader status names the library, not the run script.
        let (kind, d) = launch_failure_verdict(Some(&exit(Some(0xC000_0135u32 as i32), None)));
        assert_eq!(kind, CrashKind::LibraryLoadFailed);
        assert!(d.contains("0xC0000135"), "detail was: {d}");
        // A status that names nothing keeps the bare launch failure, and says
        // only what the missing record establishes.
        let (kind, d) = launch_failure_verdict(Some(&exit(Some(42), None)));
        assert_eq!(kind, CrashKind::LaunchFailed);
        assert!(d.contains("names no cause"), "detail was: {d}");
        // An adopted run has no launcher of ours, and the report says that
        // rather than naming a status it does not have.
        let (kind, d) = launch_failure_verdict(None);
        assert_eq!(kind, CrashKind::LaunchFailed);
        assert!(d.contains("no exit status was captured"), "detail was: {d}");
    }

    #[cfg(unix)]
    #[test]
    fn launch_failure_names_a_kill_instead_of_blaming_the_loader() {
        // 137 is 128 + 9. The same byte is decoded to SIGKILL on the abrupt
        // branch, and a cgroup OOM kill during the multi-second window before
        // `status_writer::init` (the output wipe, the dynamic load of the CUDA
        // runtime) lands here instead. Naming the loader there sends the user
        // to check an installation that is fine.
        for e in [exit(Some(137), None), exit(None, Some(9))] {
            let (kind, d) = launch_failure_verdict(Some(&e));
            assert_eq!(kind, CrashKind::KilledBySignal, "detail was: {d}");
            assert!(d.contains("SIGKILL"), "detail was: {d}");
            assert!(
                !d.contains("loader") && !d.contains("interpreter"),
                "a kill must not be reported as a loader failure: {d}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_launcher_status_and_the_abrupt_branch_decode_the_same_byte_alike() {
        // The two branches differ only in whether a record exists, which
        // changes what the FALLBACK may claim but never what the evidence
        // names. A byte that names a signal must name the same signal in both,
        // or one death gets two different causes depending on how far the
        // solver got before it died.
        for (code, name) in [
            (128 + libc::SIGABRT, "SIGABRT"),
            (128 + libc::SIGSEGV, "SIGSEGV"),
            (128 + libc::SIGKILL, "SIGKILL"),
            (128 + libc::SIGTERM, "SIGTERM"),
        ] {
            let (a, da) = classify_abrupt(&AbruptEvidence {
                launcher_code: Some(code),
                ..evidence()
            });
            let (b, db) = launch_failure_verdict(Some(&exit(Some(code), None)));
            assert_eq!(a, CrashKind::KilledBySignal, "detail was: {da}");
            assert_eq!(b, CrashKind::KilledBySignal, "detail was: {db}");
            assert!(da.contains(name), "detail was: {da}");
            assert!(db.contains(name), "detail was: {db}");
        }
    }

    // POSIX semantics: `launcher_signal` comes from `ExitStatus::signal()`,
    // which does not exist off unix, and the `128 + N` code is a shell
    // convention that is deliberately inert on Windows so an exit of 137
    // there is not read as a signal. Both inputs are unreachable on
    // Windows, where the same call correctly answers UnknownAbrupt.
    #[cfg(unix)]
    #[test]
    fn a_torn_record_report_states_no_pid_and_no_frame() {
        // A torn record names neither, and 0 is not a pid or a frame the run
        // reached: printing them would state as fact what the same sentence
        // says is unknown.
        let (kind, detail) = classify_abrupt(&AbruptEvidence {
            launcher_code: Some(137),
            ..Default::default()
        });
        assert_eq!(kind, CrashKind::KilledBySignal);
        assert!(detail.contains("SIGKILL"), "detail was: {detail}");
        assert!(!detail.contains("pid 0"), "detail was: {detail}");
        assert!(!detail.contains("frame 0"), "detail was: {detail}");
    }

    #[test]
    fn abrupt_unrecognized_code_reports_unknown_with_the_raw_number() {
        let e = AbruptEvidence {
            launcher_code: Some(42),
            ..evidence()
        };
        let (kind, detail) = classify_abrupt(&e);
        assert_eq!(kind, CrashKind::UnknownAbrupt);
        assert!(detail.contains("42"), "detail was: {detail}");
        assert!(
            detail.contains("names no cause"),
            "an unrecognized code must say so: {detail}"
        );
    }

    #[test]
    fn abrupt_without_any_evidence_reports_unknown_with_raw_facts() {
        let (kind, detail) = classify_abrupt(&evidence());
        assert_eq!(kind, CrashKind::UnknownAbrupt);
        assert!(detail.contains("pid 4242"), "detail was: {detail}");
        assert!(detail.contains("frame 17"), "detail was: {detail}");
        assert!(
            detail.contains("No signal record was written"),
            "detail was: {detail}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_launched_solver_is_not_gone_until_its_launcher_is_reaped() {
        use crate::executor::solver::exit_watch;
        use std::os::unix::process::ExitStatusExt as _;
        // A distinct root, because the slot is process-wide and holds one
        // entry.
        let _g = EXIT_WATCH_SLOT.blocking_lock();
        let root = "/nonexistent/ppf-launched-process-is-gone";
        let launch = exit_watch::arm(root);
        // Armed with no status yet: the launcher this server spawned has not
        // been reaped, so the solver is still starting. This is the entire
        // exec-to-init window, which on a cold host covers the dynamic load
        // of the CUDA runtime and the wipe of a prior run's output directory.
        // Reading it as gone reports a launch failure over a live solver, and
        // the monitor never revisits that verdict.
        assert!(!launcher_witness(root).1);
        exit_watch::record(&launch, std::process::ExitStatus::from_raw(0));
        assert!(launcher_witness(root).1);
        // Adopting an external run releases the slot, so a previous launch's
        // status is never offered as evidence about the run now in progress.
        exit_watch::disarm(root);
        assert!(!exit_watch::snapshot(root).1);
        assert_eq!(exit_watch::snapshot(root).0, None);
    }

    #[test]
    fn a_crash_report_quotes_both_ends_of_the_solver_log() {
        // The solver states the device it is about to run on, and whether the
        // operating system's kernel-execution watchdog is armed on it, before
        // any solve. Those lines sit at the head of the log by construction,
        // so a tail-only excerpt structurally cannot carry them.
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("session");
        std::fs::create_dir_all(&session).unwrap();
        let mut body = String::from(
            "cuda: device 0 is NVIDIA GeForce RTX 5090, kernelExecTimeoutEnabled 1\n",
        );
        for n in 0..400 {
            body.push_str(&format!("step {n}\n"));
        }
        std::fs::write(session.join(STDOUT_LOG), &body).unwrap();
        let report = render_crash(CrashKind::UnknownAbrupt, "detail", dir.path());
        assert!(
            report.contains("kernelExecTimeoutEnabled 1"),
            "the environment record must survive: {report}"
        );
        assert!(
            report.contains("step 399"),
            "the failure neighborhood must survive: {report}"
        );
    }

    #[test]
    fn a_short_solver_log_is_quoted_once() {
        // Head and tail must not overlap, or a short log is printed twice.
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("session");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(session.join(STDOUT_LOG), "only line\n").unwrap();
        let report = render_crash(CrashKind::UnknownAbrupt, "detail", dir.path());
        assert_eq!(
            report.matches("only line").count(),
            1,
            "log quoted more than once: {report}"
        );
    }

    // ---- Whole-tick tests over the `exit_watch` slot -----------------------
    //
    // The slot is process-wide and holds one entry, so these run one at a
    // time. Two of them concurrently would each observe the other's arm.
    // A tokio mutex because half of these tests await `tick` while holding it;
    // a `std` guard across an await is what `clippy::await_holding_lock` is for.
    static EXIT_WATCH_SLOT: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// An engine already in `Solver::Running` for `root`, with no grace
    /// period, so one `tick` reaches a verdict with no sleeping.
    fn running_engine(root: &Path) -> (crate::engine::ServerEngine, Arc<dyn EffectExecutor>) {
        use crate::config::EngineConfig;
        let cfg = EngineConfig {
            monitor_interval_ms: 10,
            solver_startup_grace_ms: 0,
            ..Default::default()
        };
        let engine = crate::engine::ServerEngine::new(cfg);
        engine.set_project_context("p", root.to_str().unwrap());
        engine.dispatch(Event::upload_landed("uid"));
        engine.dispatch(Event::BuildCompleted);
        engine.dispatch(Event::StartRequested);
        assert_eq!(engine.state().solver, Solver::Running);
        (
            engine,
            Arc::new(crate::executor::DefaultExecutor::new()) as Arc<dyn EffectExecutor>,
        )
    }

    fn make_out_dir(root: &Path) -> PathBuf {
        let out = output_dir(root);
        std::fs::create_dir_all(&out).unwrap();
        out
    }

    fn progress_record(frame: i32, pid: u32) -> status::RunStatus {
        status::RunStatus {
            phase: Phase::Running,
            frame,
            sim_time: 0.0,
            resumable: false,
            outcome: None,
            seq: 3,
            pid,
            launch_id: "testlaunch00".into(),
            emulated: true,
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_relaunch_does_not_inherit_the_previous_launcher_status() {
        use crate::executor::solver::exit_watch;
        use std::os::unix::process::ExitStatusExt as _;
        // Terminate then Run: both launches carry the same project root, and
        // launcher #1 is still being awaited when launch #2 arms the slot.
        // Keyed by root alone, #1's status lands in #2's slot and the monitor
        // reads a solver that is still loading its libraries as already gone.
        let _g = EXIT_WATCH_SLOT.blocking_lock();
        let root = "/nonexistent/ppf-relaunch-generation";
        let first = exit_watch::arm(root);
        let second = exit_watch::arm(root);
        exit_watch::record(&first, std::process::ExitStatus::from_raw(15));
        assert_eq!(exit_watch::snapshot(root).0, None);
        assert!(
            !launcher_witness(root).1,
            "launch #2 has not been reaped, so it is still starting"
        );
        // The launch that owns the slot still records normally.
        exit_watch::record(&second, std::process::ExitStatus::from_raw(15));
        assert!(exit_watch::snapshot(root).0.is_some());
        assert!(launcher_witness(root).1);
        exit_watch::disarm(root);
    }

    #[test]
    fn a_launch_that_will_never_report_hands_the_slot_back() {
        use crate::executor::solver::exit_watch;
        // A failed spawn, or a launcher that cannot be reaped, produces no
        // status ever. Left armed, the slot answers "still starting" for the
        // rest of the process's life and no launch failure is ever declared.
        let _g = EXIT_WATCH_SLOT.blocking_lock();
        let root = "/nonexistent/ppf-released-launch";
        let launch = exit_watch::arm(root);
        assert!(exit_watch::snapshot(root).1);
        exit_watch::release(&launch);
        assert!(!exit_watch::snapshot(root).1);
        // A superseded launch cannot release its successor's slot.
        let stale = exit_watch::arm(root);
        let current = exit_watch::arm(root);
        exit_watch::release(&stale);
        assert!(exit_watch::snapshot(root).1);
        exit_watch::release(&current);
        assert!(!exit_watch::snapshot(root).1);
    }

    #[tokio::test]
    async fn a_released_slot_lets_the_launch_failure_branch_fire_again() {
        use crate::executor::solver::exit_watch;
        // The consequence of the test above, end to end: with the slot handed
        // back, the liveness witness falls through to the process scan and the
        // launch failure is declared, instead of the panel sitting at
        // "Running" with the Run button disabled forever.
        let _g = EXIT_WATCH_SLOT.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let (engine, ex) = running_engine(dir.path());
        let root = engine.state().root;
        make_out_dir(dir.path());
        let launch = exit_watch::arm(&root);
        let mut ctx = MonitorContext::default();
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        assert_eq!(
            engine.state().solver,
            Solver::Running,
            "while the launcher is armed and unreaped the solver is starting"
        );
        exit_watch::release(&launch);
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        let s = engine.state();
        exit_watch::disarm(&root);
        assert_eq!(s.solver, Solver::Failed);
        assert_eq!(s.crash_kind, "launch_failed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn no_verdict_is_sealed_before_the_launcher_is_reaped() {
        use crate::executor::solver::exit_watch;
        use std::os::unix::process::ExitStatusExt as _;
        // A SIGKILLed solver writes no sidecar by construction, so the
        // launcher's `137` is the only witness. The solver releases its lock
        // and vacates its pid at the instant of death, while that `137` still
        // has to travel through bash, tokio's reaper and `exit_watch::record`.
        // A tick landing in between must wait, not seal "cause not recorded"
        // as the run's terminal outcome.
        let _g = EXIT_WATCH_SLOT.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let (engine, ex) = running_engine(dir.path());
        let root = engine.state().root;
        let out = make_out_dir(dir.path());
        status::write_progress(&out, &progress_record(7, 0)).unwrap();
        let launch = exit_watch::arm(&root);
        let mut ctx = MonitorContext::default();
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        assert_eq!(
            engine.state().solver,
            Solver::Running,
            "the only witness that can name an uncatchable kill is still in flight"
        );

        exit_watch::record(&launch, std::process::ExitStatus::from_raw(137 << 8));
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        let s = engine.state();
        exit_watch::disarm(&root);
        assert_eq!(s.crash_kind, "killed_by_signal");
        assert!(s.error.contains("SIGKILL"), "unexpected error: {}", s.error);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_launcher_that_died_first_is_not_quoted_as_the_solver_cause() {
        use crate::executor::solver::exit_watch;
        use std::os::unix::process::ExitStatusExt as _;
        // The shell was killed with SIGTERM and the solver outlived it, kept
        // writing, and died of something else. The launcher's status names a
        // signal that never reached the solver, and quoting it sends the user
        // looking for whoever sent a SIGTERM.
        let _g = EXIT_WATCH_SLOT.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let (engine, ex) = running_engine(dir.path());
        let root = engine.state().root;
        let launch = exit_watch::arm(&root);
        exit_watch::record(&launch, std::process::ExitStatus::from_raw(15));
        std::thread::sleep(Duration::from_millis(20));
        status::write_progress(&make_out_dir(dir.path()), &progress_record(7, 0)).unwrap();
        let mut ctx = MonitorContext::default();
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        let s = engine.state();
        exit_watch::disarm(&root);
        assert_eq!(s.solver, Solver::Failed);
        assert_eq!(
            s.crash_kind, "unknown_abrupt",
            "a launcher status older than the solver's own last write is not \
             evidence about the solver's death: {}",
            s.error
        );
        assert!(
            !s.error.contains("SIGTERM"),
            "the wrong signal must not be named: {}",
            s.error
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_torn_record_reads_the_launcher_status_and_is_sealed() {
        use crate::executor::solver::exit_watch;
        use std::os::unix::process::ExitStatusExt as _;
        // Dying mid-write tears the record, and a torn record names nothing.
        // The launcher's status names the same death, and it is the same
        // binding the branch above reads on the same tick. Without it a run
        // that died a millisecond into a checkpoint gets strictly less
        // diagnosis than one that died a millisecond earlier.
        let _g = EXIT_WATCH_SLOT.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let (engine, ex) = running_engine(dir.path());
        let root = engine.state().root;
        let out = make_out_dir(dir.path());
        status::write_progress(&out, &progress_record(7, 0)).unwrap();
        let path = out.join(ppf_cts_formats::files::STATUS_RECORD);
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() / 2]).unwrap();
        let launch = exit_watch::arm(&root);
        exit_watch::record(&launch, std::process::ExitStatus::from_raw(137 << 8));
        let mut ctx = MonitorContext::default();
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        let s = engine.state();
        exit_watch::disarm(&root);
        assert_eq!(s.crash_kind, "killed_by_signal");
        assert!(s.error.contains("SIGKILL"), "unexpected error: {}", s.error);
        assert!(
            s.error.contains("torn mid-write"),
            "the truncation is still stated: {}",
            s.error
        );
        // The panel draws one clipped line, so the fact that discriminates
        // leads and the truncation follows it.
        assert!(
            s.error.find("SIGKILL") < s.error.find("torn mid-write"),
            "the cause must lead: {}",
            s.error
        );
        // Sealed, or the crash exists only in this process's memory: the
        // reconnect path routes a failed read to "no crash", so a restart
        // would present the truncated run as a healthy project.
        match status::read(&out) {
            Ok(Some(rec)) => match rec.outcome {
                Some(Outcome::Crashed { sub_kind, .. }) => {
                    assert_eq!(sub_kind, CrashKind::KilledBySignal);
                }
                other => panic!("expected a sealed Crashed outcome, got {other:?}"),
            },
            other => panic!("the torn record was never sealed: {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_recycled_pid_does_not_outvote_two_fresh_witnesses() {
        use crate::executor::solver::exit_watch;
        use std::os::unix::process::ExitStatusExt as _;
        // Pids recycle and a run here is routinely longer than the wrap, so a
        // record can name a pid that now belongs to something else. Read as a
        // bare disjunct it suppresses every verdict and the panel shows a run
        // in progress that ended minutes ago; there is no other watchdog.
        let _g = EXIT_WATCH_SLOT.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let (engine, ex) = running_engine(dir.path());
        let root = engine.state().root;
        // This process stands in for the unrelated process that took the pid.
        status::write_progress(
            &make_out_dir(dir.path()),
            &progress_record(7, std::process::id()),
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        let launch = exit_watch::arm(&root);
        exit_watch::record(&launch, std::process::ExitStatus::from_raw(0));
        let mut ctx = MonitorContext::default();
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        let s = engine.state();
        exit_watch::disarm(&root);
        assert_eq!(
            s.solver,
            Solver::Failed,
            "a free lock plus a reaped launcher outrank a recycled pid"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_live_run_whose_launcher_is_armed_keeps_its_recycled_pid_reading() {
        use crate::executor::solver::exit_watch;
        // The corroboration above must not fire while the launcher is still
        // running: that is the ordinary case, and reading a live solver as
        // dead would end every run at its first tick.
        let _g = EXIT_WATCH_SLOT.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let (engine, ex) = running_engine(dir.path());
        let root = engine.state().root;
        status::write_progress(
            &make_out_dir(dir.path()),
            &progress_record(7, std::process::id()),
        )
        .unwrap();
        let launch = exit_watch::arm(&root);
        let mut ctx = MonitorContext::default();
        tick(&engine, ex.as_ref(), &mut ctx, 0).await.unwrap();
        let s = engine.state();
        exit_watch::release(&launch);
        assert_eq!(s.solver, Solver::Running);
        assert_eq!(s.frame, 7, "progress is still reported");
    }




    #[test]
    fn read_intersection_violations_parses() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("session/output")).unwrap();
        let body = serde_json::json!({
            "records": [
                {"type": "tri-tri", "elem0": 1, "elem1": 2}
            ],
            "count": 1
        })
        .to_string();
        std::fs::write(
            dir.path()
                .join("session/output/intersection_records.json"),
            body,
        )
        .unwrap();
        let v = read_intersection_violations(dir.path()).unwrap();
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("tri-tri"));
    }
}
