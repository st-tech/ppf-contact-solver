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
// by construction. The global `solver_busy` scan is used only to adopt an
// externally-launched run and as the liveness gate when a record cannot
// be read (torn / not yet written); `project_resumable` reports whether a
// `state_<N>.bin.gz` checkpoint exists.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ppf_cts_core::datamodel::project_resumable;
use ppf_cts_core::events::Event;
use ppf_cts_core::state::{Build, Data, Solver};
use ppf_cts_formats::files::{
    session_dir as session_dir_for, session_output_dir, DATA_PICKLE, INTERSECTION_RECORDS_JSON,
    PARAM_PICKLE, STDOUT_LOG, TERMINATE_REQUEST,
};
use ppf_cts_formats::status::{self, lock, CrashKind, Outcome, Phase};
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
                        let error = render_crash(sub_kind, &detail, &root);
                        report_crash(engine, executor, ctx, &root, error).await?;
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
                Some(Outcome::Terminated { .. }) | Some(Outcome::Unknown { .. }) => {
                    finish_solver(engine, executor, ctx, &root, false).await;
                }
                Some(Outcome::Crashed { sub_kind, detail }) => {
                    let error = render_crash(sub_kind, &detail, &root);
                    report_crash(engine, executor, ctx, &root, error).await?;
                }
                None => {
                    // No terminal outcome yet: live, or died abruptly. The
                    // lock and the owning PID are the crux, and the check is
                    // PID-scoped so a second unrelated solver (e.g. another
                    // run sharing the same host) cannot suppress it.
                    let alive =
                        lock::is_held_by_other(&out_dir) || lock::pid_alive(rec.pid);
                    if alive {
                        if rec.phase == Phase::Saving && s.solver == Solver::Running {
                            dispatch_with_executor(engine, executor, Event::SolverSaving)
                                .await;
                        }
                    } else if elapsed_ok {
                        if terminate_intended(&out_dir, engine) {
                            // Intentional stop whose host left no terminal
                            // record: a hard kill, a Windows uncatchable
                            // terminate, or the mid-tick race the in-memory
                            // Idle covers. Clean, never a crash.
                            finish_solver(engine, executor, ctx, &root, false).await;
                        } else {
                            let error = format!(
                                "Solver exited abnormally: pid {} stopped after frame {} \
                                 without writing a terminal outcome (segfault / OOM-kill / \
                                 unrecoverable abort)",
                                rec.pid, rec.frame
                            );
                            report_crash(engine, executor, ctx, &root, error).await?;
                        }
                    }
                    // else: grace not elapsed or signals disagree -> wait.
                }
            }
        }
        Ok(None) => {
            // No record yet: the solver process is spawned but has not
            // reached status_writer::init. If the grace window elapsed and
            // the process is already gone with no terminate intent, it died
            // before initializing (a launch / exec failure).
            if elapsed_ok && !solver_busy() && !terminate_intended(&out_dir, engine) {
                report_crash(
                    engine,
                    executor,
                    ctx,
                    &root,
                    "Solver exited before initializing (no status record written)".into(),
                )
                .await?;
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
            // the process scan gates the liveness verdict; it still
            // distinguishes a deliberate stop from a crash.
            if elapsed_ok && !solver_busy() {
                if terminate_intended(&out_dir, engine) {
                    finish_solver(engine, executor, ctx, &root, false).await;
                } else if engine.state().solver == Solver::Saving {
                    finish_solver(engine, executor, ctx, &root, true).await;
                } else {
                    report_crash(
                        engine,
                        executor,
                        ctx,
                        &root,
                        "Solver exited abnormally with a truncated status record".into(),
                    )
                    .await?;
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
/// grace timer. The durable crash record is the solver-authored terminal
/// `Crashed` in `status.cbor`; reconnect reads that, so there is no
/// separate crash marker to write here.
async fn report_crash(
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    ctx: &mut MonitorContext,
    root: &Path,
    error: String,
) -> Result<(), MonitorError> {
    let violations = read_intersection_violations(root)?;
    dispatch_with_executor(engine, executor, Event::SolverCrashed { error, violations }).await;
    ctx.last_solver_state = Solver::Idle;
    ctx.solver_started_at = None;
    Ok(())
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

/// Human-readable crash string for a structured `Crashed{kind, detail}`:
/// the kind summary (the single message table, replacing the two drifted
/// substring tables), the solver's own detail, and a tail of stdout.log
/// as supplementary context (never as the classifier).
fn render_crash(kind: CrashKind, detail: &str, root: &Path) -> String {
    let tail = read_log_tail(root);
    if tail.is_empty() {
        format!("{}: {detail}", kind.summary())
    } else {
        format!(
            "{}: {detail}\n--- Solver Log (last {CRASH_TAIL_LINES} lines) ---\n{tail}",
            kind.summary()
        )
    }
}

/// Last `CRASH_TAIL_LINES` lines of `stdout.log`, or empty if absent.
fn read_log_tail(root: &Path) -> String {
    let path = session_dir(root).join(STDOUT_LOG);
    let lines = read_lines_if_exists(&path).unwrap_or_default();
    let start = lines.len().saturating_sub(CRASH_TAIL_LINES);
    lines[start..].join("\n")
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
