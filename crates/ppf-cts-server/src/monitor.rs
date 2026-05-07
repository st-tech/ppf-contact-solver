// File: crates/ppf-cts-server/src/monitor.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Background tokio task that polls the project's session/output dir
// and dispatches solver-monitor events into the engine. The three
// presence checks delegate to core helpers:
//   * `Utils.busy()` external-solver detection routes to
//     `core::utils::solver_busy`, which scans for live `ppf-contact`
//     processes via sysinfo.
//   * `app.is_saving_in_progress()` save-flag detection routes to
//     `core::datamodel::is_saving_in_progress`, which checks for the
//     `save_and_quit` sentinel under `<root>/session/output/`.
//   * `app.resumable()` checkpoint check routes to
//     `core::datamodel::project_resumable`, which checks for any
//     `state_<N>.bin.gz` checkpoint under the same directory.
//
// This module covers the file-watching half (frame counting,
// finished.txt, error.log, intersection_records.json).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ppf_cts_core::datamodel::{is_saving_in_progress, project_resumable};
use ppf_cts_core::events::Event;
use ppf_cts_core::state::{Build, Solver};
use ppf_cts_formats::files::{
    DATA_PICKLE, FINISHED, INITIALIZE_FINISH, PARAM_PICKLE,
};
// Test rig spawns peer workers as sibling processes; the
// emulated-feature build narrows the busy check to descendants only
// so a foreign worker's solver doesn't trip our liveness watchdog.
#[cfg(feature = "emulated")]
use ppf_cts_core::utils::solver_busy_descendants_only as solver_busy;
#[cfg(not(feature = "emulated"))]
use ppf_cts_core::utils::solver_busy;
use serde::Deserialize;

use crate::engine::ServerEngine;
use crate::executor::EffectExecutor;

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
        let effects = engine.dispatch(Event::ExternalSolverAdopted { has_data_on_disk });
        for fx in effects {
            executor.execute(fx, engine).await;
        }
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

    let frame = count_frames(&root);
    if frame != s.frame {
        let effects = engine.dispatch(Event::SolverFrameUpdated { frame });
        for fx in effects {
            executor.execute(fx, engine).await;
        }
    }

    // Detect the in-process `initialize()` finish. The solver writes
    // `<output>/initialize_finish.txt` immediately after `initialize()`
    // returns true and before the per-frame loop starts. Flipping
    // `initialized` here lets the addon promote the local solver state
    // from STARTING to RUNNING without waiting for the first frame's
    // advance to complete (which can be several seconds under heavy
    // contact load). The flag is reset to false on Start/Resume/
    // Terminate/SolverFinished/SolverCrashed, so we only dispatch when
    // currently false. The simulator removes the file at the start of
    // each `run()`, so a second run in a chain scenario gets a fresh
    // observation window.
    if !s.initialized
        && root.join("session").join("output").join(INITIALIZE_FINISH).exists()
    {
        let effects = engine.dispatch(Event::SolverInitialized);
        for fx in effects {
            executor.execute(fx, engine).await;
        }
    }

    // `finished.txt` is the solver's clean-exit marker. Once it's
    // there we transition to Idle/Failed depending on the saving
    // path.
    let finished = root
        .join("session")
        .join("output")
        .join(FINISHED)
        .exists();

    if finished {
        // Edge: catch the last frame the solver wrote between ticks.
        let final_frame = count_frames(&root);
        if final_frame != engine.state().frame {
            let effects = engine.dispatch(Event::SolverFrameUpdated { frame: final_frame });
            for fx in effects {
                executor.execute(fx, engine).await;
            }
        }

        // Saving path always produces a resumable checkpoint by
        // construction. Otherwise consult the on-disk state files
        // (`state_<N>.bin.gz`) via `project_resumable`.
        let resumable = if s.solver == Solver::Saving {
            true
        } else {
            project_resumable(&root)
        };
        let effects = engine.dispatch(Event::SolverFinished { resumable });
        for fx in effects {
            executor.execute(fx, engine).await;
        }
        // Snap the local state mirror back to Idle so the next
        // Idle → Running edge (e.g. the second run in a chain
        // scenario) fires the grace-timer reset below. Without this
        // the engine briefly returns to Idle and back to Running
        // between ticks, the monitor never observes the Idle pass,
        // and the new run inherits stale solver_started_at from the
        // first.
        ctx.last_solver_state = Solver::Idle;
        ctx.solver_started_at = None;
        return Ok(());
    }

    let elapsed_ok = ctx
        .solver_started_at
        .map(|t| t.elapsed() >= Duration::from_millis(grace_ms))
        .unwrap_or(false);

    // Crash detection from error.log when the solver should be live
    // but evidence of failure exists. Apply the startup grace period.
    if elapsed_ok {
        if let Some(error) = check_solver_error(&root)? {
            let violations = read_intersection_violations(&root)?;
            let effects = engine.dispatch(Event::SolverCrashed { error, violations });
            for fx in effects {
                executor.execute(fx, engine).await;
            }
            ctx.last_solver_state = Solver::Idle;
            ctx.solver_started_at = None;
            return Ok(());
        }
    }

    // External-solver liveness via `solver_busy` (mirrors
    // `Utils.busy()`). After the grace period, if no `ppf-contact`
    // process is alive the solver has exited without writing
    // `finished.txt` and without leaving a crash signature in
    // error.log. Treat the saving path as a clean checkpoint, the
    // running path as a clean exit whose resumability comes from
    // the on-disk state files.
    if elapsed_ok && !solver_busy() {
        let resumable = if s.solver == Solver::Saving {
            true
        } else {
            project_resumable(&root)
        };
        let effects = engine.dispatch(Event::SolverFinished { resumable });
        for fx in effects {
            executor.execute(fx, engine).await;
        }
        ctx.last_solver_state = Solver::Idle;
        ctx.solver_started_at = None;
        return Ok(());
    }

    // Save-in-progress detection via the `save_and_quit` sentinel
    // (`is_saving_in_progress`). The transition from Running to
    // Saving is one-way; we only fire the event on the first tick
    // that observes the file.
    if s.solver == Solver::Running && is_saving_in_progress(&root) {
        let effects = engine.dispatch(Event::SolverSaving);
        for fx in effects {
            executor.execute(fx, engine).await;
        }
    }

    Ok(())
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

/// Count `vert_*.bin` files in `<root>/session/output/`. Returns the
/// max frame number seen, or 0.
pub(crate) fn count_frames(root: &Path) -> i32 {
    let dir = root.join("session").join("output");
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut max_num: i32 = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        if let Some(rest) = name.strip_prefix("vert_") {
            if let Some(num_str) = rest.strip_suffix(".bin") {
                if let Ok(n) = num_str.parse::<i32>() {
                    if n > max_num {
                        max_num = n;
                    }
                }
            }
        }
    }
    max_num
}

/// Detect a crash by analyzing `error.log` and `stdout.log`. Returns
/// `Some(error_string)` when a crash is detected, `None` otherwise.
pub(crate) fn check_solver_error(root: &Path) -> Result<Option<String>, MonitorError> {
    let session = root.join("session");
    let output_dir = session.join("output");
    if output_dir.join(FINISHED).exists() {
        return Ok(None); // normal completion
    }
    let err_path = session.join("error.log");
    let log_path = session.join("stdout.log");

    let err_lines: Vec<String> = read_lines_if_exists(&err_path)?;
    let mut log_lines: Vec<String> = read_lines_if_exists(&log_path)?;
    if log_lines.len() > 200 {
        let drop = log_lines.len() - 200;
        log_lines.drain(..drop);
    }

    let err_content = err_lines.join("").trim().to_string();
    if err_content.is_empty() || err_content.eq_ignore_ascii_case("terminated") {
        return Ok(None); // user-initiated terminate, not a crash
    }

    let mut reason = "Solver crashed unexpectedly";
    let patterns: &[(&str, &str)] = &[
        ("### intersection detected", "Intersection detected"),
        ("### ccd failed", "CCD failed"),
        ("### cg failed", "Linear solver failed"),
        ("failed to advance", "Solver crashed"),
        ("panic", "Solver panic"),
        ("assert", "Assertion failed"),
    ];
    for line in log_lines.iter().chain(err_lines.iter()) {
        let low = line.to_lowercase();
        for (pat, msg) in patterns {
            if low.contains(pat) {
                reason = msg;
                break;
            }
        }
    }

    let tail_start = log_lines.len().saturating_sub(32);
    let tail = log_lines[tail_start..].join("");
    Ok(Some(format!(
        "{reason}\n--- Solver Log (last 32 lines) ---\n{tail}--- Solver Error Log ---\n{err_content}"
    )))
}

/// Read intersection records from `intersection_records.json`.
pub(crate) fn read_intersection_violations(root: &Path) -> Result<Vec<String>, MonitorError> {
    let path = root
        .join("session")
        .join("output")
        .join("intersection_records.json");
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
    #[serde(default)]
    records: Vec<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    count: Option<i64>,
}

fn read_lines_if_exists(path: &Path) -> std::io::Result<Vec<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(s.lines().map(|l| format!("{l}\n")).collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::File::create(path).unwrap();
    }

    #[test]
    fn count_frames_picks_max_index() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("session/output");
        std::fs::create_dir_all(&out).unwrap();
        for n in [1, 5, 12, 7] {
            touch(&out.join(format!("vert_{n}.bin")));
        }
        // Distractor files should be ignored.
        touch(&out.join("vert_garbage.bin"));
        touch(&out.join("foo.bin"));
        assert_eq!(count_frames(dir.path()), 12);
    }

    #[test]
    fn count_frames_empty_or_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(count_frames(dir.path()), 0);
        std::fs::create_dir_all(dir.path().join("session/output")).unwrap();
        assert_eq!(count_frames(dir.path()), 0);
    }

    #[test]
    fn check_solver_error_quiet_when_finished_present() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("session/output");
        std::fs::create_dir_all(&out).unwrap();
        touch(&out.join("finished.txt"));
        // Error log content should be ignored when finished.txt exists.
        std::fs::write(dir.path().join("session/error.log"), "Segfault\n").unwrap();
        assert!(check_solver_error(dir.path()).unwrap().is_none());
    }

    #[test]
    fn check_solver_error_reports_intersection_pattern() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("session/output")).unwrap();
        std::fs::write(
            dir.path().join("session/error.log"),
            "### intersection detected near tri 42\n",
        )
        .unwrap();
        let err = check_solver_error(dir.path()).unwrap().unwrap();
        assert!(err.starts_with("Intersection detected"), "got: {err}");
    }

    #[test]
    fn check_solver_error_silent_on_terminated() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("session/output")).unwrap();
        std::fs::write(dir.path().join("session/error.log"), "Terminated\n").unwrap();
        assert!(check_solver_error(dir.path()).unwrap().is_none());
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
