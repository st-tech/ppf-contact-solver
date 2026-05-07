// File: crates/ppf-cts-server/src/executor/build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Build pipeline plumbing for `DoSpawnBuild`. The build runs in a
// Python subprocess (`frontend/build_worker.py`). The Rust side
// spawns the worker, parses line-oriented progress + error markers
// from its stdout, and forwards a SIGTERM on cooperative cancel.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use ppf_cts_core::cancel::CancelHandle;
use ppf_cts_core::events::Event;

use super::dispatch_re_entrant;
use crate::engine::ServerEngine;

/// Spawn the build task and install a cancel handle on the engine.
/// The body is a placeholder loop that polls the cancel token at
/// regular intervals and emits `BuildProgress` events. The real
/// build pipeline (decoder, tetrahedralize, FixedScene assembly)
/// is being ported in parallel; the plumbing here already matches
/// the contract expected by the engine + monitor.
pub(super) fn spawn_build_task(engine: &ServerEngine) {
    let cancel = engine.install_cancel_handle();
    let engine = engine.clone();

    tokio::spawn(async move {
        let result = run_build_pipeline(&engine, cancel.clone()).await;

        // Re-entrant dispatch routes through whichever executor the
        // engine was attached to (`ServerEngine::attach_executor`),
        // falling back to a fresh `DefaultExecutor` for tests that
        // drive the build pipeline in isolation.
        match result {
            BuildOutcome::Completed => {
                dispatch_re_entrant(&engine, Event::BuildCompleted).await;
            }
            BuildOutcome::Cancelled => {
                log::info!(target: "ppf::build", "[BUILD] cancelled by user");
                dispatch_re_entrant(&engine, Event::BuildCancelledEvent).await;
            }
            BuildOutcome::Failed(error) => {
                dispatch_re_entrant(
                    &engine,
                    Event::BuildFailed {
                        error,
                        violations: vec![],
                    },
                )
                .await;
            }
            BuildOutcome::AlreadyDispatched => {
                // Pipeline body owns the terminal dispatch (e.g.
                // GpuCheckFailed). No further event from us.
            }
        }

        engine.clear_cancel_handle();
    });
}

pub(super) enum BuildOutcome {
    Completed,
    Cancelled,
    /// Decoder / tetrahedralize / FixedScene errors surface here
    /// after the worker prints `ERROR <msg>` and exits non-zero, or
    /// when we cannot launch the worker at all.
    Failed(String),
    /// The pipeline already dispatched a terminal event (e.g.
    /// `GpuCheckFailed`) and the caller must NOT dispatch another
    /// `BuildCompleted` / `BuildCancelledEvent` / `BuildFailed`.
    /// Only constructed in non-emulated builds (the GPU-check path),
    /// but the test harness in `executor/mod.rs::other_disc` matches
    /// every variant so the enum needs to keep it under `emulated`.
    #[cfg_attr(feature = "emulated", allow(dead_code))]
    AlreadyDispatched,
}

/// Run the build pipeline in a Python subprocess.
///
/// Cancel signaling: on Unix the worker installs a SIGTERM handler
/// that translates the signal into `KeyboardInterrupt`, so SIGTERM is
/// the cooperative path. On Windows there is no SIGTERM; we route
/// through `Child::start_kill` (TerminateProcess), which the worker
/// cannot intercept but does cause it to exit (sufficient for
/// `BuildOutcome::Cancelled`). We never escalate to SIGKILL
/// automatically on Unix: a stuck worker is better surfaced as a hung
/// build than as silent data corruption from a half-released GPU
/// buffer. Operators who need to force-kill can do so manually with
/// the worker pid logged at spawn time.
async fn run_build_pipeline(engine: &ServerEngine, cancel: CancelHandle) -> BuildOutcome {
    // Cached GPU check, mirroring the EffectExecutor._gpu_checked
    // class-level guard. We don't have a process-global cache yet,
    // so we just call into utils::check_gpu directly. A future
    // phase can introduce a OnceCell if the cost ever shows up.
    //
    // The emulated build skips the check entirely (mirrors
    // server/emulator.py's `Utils.check_gpu = no-op` patch); the
    // emulated solver doesn't touch CUDA so nvidia-smi being absent
    // is irrelevant.
    #[cfg(not(feature = "emulated"))]
    if let Err(e) = ppf_cts_core::utils::check_gpu() {
        // GPU check failure is its own event in transitions.
        dispatch_re_entrant(
            engine,
            Event::GpuCheckFailed {
                error: e.to_string(),
            },
        )
        .await;
        // GpuCheckFailed already moved state to Failed; we must NOT
        // dispatch a follow-up BuildCancelledEvent / BuildFailed,
        // both of which would clobber the Failed terminal state.
        return BuildOutcome::AlreadyDispatched;
    }

    let (name, root) = {
        let s = engine.state();
        (s.name, s.root)
    };
    if name.is_empty() || root.is_empty() {
        return BuildOutcome::Failed(
            "no project context: name/root must be set before BuildRequested".into(),
        );
    }

    let worker = match locate_build_worker() {
        Some(p) => p,
        None => {
            return BuildOutcome::Failed(
                "build worker script not found (set PPF_CTS_BUILD_WORKER or install frontend/build_worker.py)"
                    .into(),
            );
        }
    };
    let python = python_executable();
    drive_build_worker(engine, cancel, &python, &worker, &name, &root).await
}

/// Spawn the worker process and translate its stdout protocol into
/// engine events. Factored out of `run_build_pipeline` so unit tests
/// can drive a mock worker without going through the GPU check.
pub(super) async fn drive_build_worker(
    engine: &ServerEngine,
    cancel: CancelHandle,
    python: &Path,
    worker: &Path,
    name: &str,
    root: &str,
) -> BuildOutcome {
    let mut cmd = Command::new(python);
    cmd.arg(worker)
        .arg(name)
        .arg(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Line-buffer Python so PROGRESS lines arrive promptly.
        .env("PYTHONUNBUFFERED", "1");
    // ``frontend`` lives one directory above ``build_worker.py``;
    // exposing that on PYTHONPATH lets the worker import it
    // regardless of the server's cwd (the test rig runs the server
    // from a per-worker temp dir, where ``import frontend`` would
    // otherwise raise ``ModuleNotFoundError``).
    if let Some(repo_root) = worker.parent().and_then(|p| p.parent()) {
        let existing = std::env::var_os("PYTHONPATH");
        let new_path = match existing {
            Some(prev) if !prev.is_empty() => {
                let sep = if cfg!(target_os = "windows") { ";" } else { ":" };
                let mut combined = repo_root.as_os_str().to_owned();
                combined.push(sep);
                combined.push(&prev);
                combined
            }
            _ => repo_root.as_os_str().to_owned(),
        };
        cmd.env("PYTHONPATH", new_path);
    }
    // Detach from the server's process group so a Ctrl-C on the
    // server doesn't blow away an in-flight build (the server's own
    // shutdown path issues SIGTERM through `cancel`).
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }

    log::info!(
        target: "ppf::build",
        "[BUILD] {name}: spawning {} {} {} {}",
        python.display(),
        worker.display(),
        name,
        root,
    );
    let mut child: Child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return BuildOutcome::Failed(format!(
                "failed to spawn build worker ({}): {}",
                python.display(),
                e
            ));
        }
    };

    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => return BuildOutcome::Failed("worker stdout not captured".into()),
    };
    let stderr = child.stderr.take();

    // Drain stderr in a side task so a chatty worker can't deadlock
    // by filling the kernel pipe buffer. We forward each line to the
    // server log under [BUILD] so operators can correlate failures.
    if let Some(err) = stderr {
        let name = name.to_string();
        tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                log::warn!(target: "ppf::build", "[BUILD stderr] {name}: {line}");
            }
        });
    }

    let mut reader = BufReader::new(stdout).lines();
    let mut error_reason: Option<String> = None;
    let mut sigterm_sent = false;

    loop {
        tokio::select! {
            biased;
            _ = cancel.token().cancelled(), if !sigterm_sent => {
                let pid = child.id();
                log::info!(
                    target: "ppf::build",
                    "[BUILD] {name}: cancel observed; signaling worker pid={pid:?}",
                );
                send_cancel_signal(&mut child).await;
                sigterm_sent = true;
                // Don't return yet: keep draining stdout so the
                // worker has a chance to flush a final ERROR line
                // and exit code 130 cleanly.
            }
            line = reader.next_line() => {
                match line {
                    Ok(Some(text)) => {
                        if let Some((progress, info)) = parse_progress_line(&text) {
                            dispatch_re_entrant(
                                engine,
                                Event::BuildProgress { progress, info },
                            )
                            .await;
                        } else if let Some(total_frames) = parse_meta_frames_line(&text) {
                            dispatch_re_entrant(
                                engine,
                                Event::BuildMetadata { total_frames },
                            )
                            .await;
                        } else if let Some(msg) = parse_error_line(&text) {
                            // Last ERROR wins; the worker may print
                            // several before exiting. Real builds
                            // print a single line then exit non-zero.
                            error_reason = Some(msg);
                        } else if !text.is_empty() {
                            log::debug!(target: "ppf::build", "[BUILD stdout] {name}: {text}");
                        }
                    }
                    Ok(None) => break, // worker closed stdout
                    Err(e) => {
                        log::warn!(target: "ppf::build", "[BUILD] {name}: stdout read error: {e}");
                        break;
                    }
                }
            }
        }
    }

    // Stdout has closed; the worker is exiting. Wait for the exit
    // status so we can distinguish cancel (130) from crash (non-zero
    // without our SIGTERM) from clean completion (0).
    let status = match child.wait().await {
        Ok(s) => s,
        Err(e) => return BuildOutcome::Failed(format!("worker wait failed: {e}")),
    };

    if sigterm_sent || cancel.is_cancelled() {
        return BuildOutcome::Cancelled;
    }
    if status.success() {
        return BuildOutcome::Completed;
    }
    let reason = error_reason.unwrap_or_else(|| match status.code() {
        Some(code) => format!("build worker exited with code {code}"),
        None => "build worker terminated by signal".into(),
    });
    BuildOutcome::Failed(reason)
}

/// Parse `PROGRESS percent=<float> info=<text>`. Returns `None` if
/// the line isn't a progress marker. Tolerates info strings that
/// contain `=` (we only split once on the percent boundary) and any
/// run of whitespace between fields.
fn parse_progress_line(line: &str) -> Option<(f64, String)> {
    let rest = line.strip_prefix("PROGRESS ")?.trim_start();
    let pct_rest = rest.strip_prefix("percent=")?;
    // Stop at the first whitespace; everything after is the info
    // payload (possibly empty, possibly with `info=` prefix).
    let split = pct_rest.find(char::is_whitespace);
    let (pct_str, after) = match split {
        Some(idx) => (&pct_rest[..idx], pct_rest[idx..].trim_start()),
        None => (pct_rest, ""),
    };
    let pct: f64 = pct_str.parse().ok()?;
    let info = after.strip_prefix("info=").unwrap_or(after).to_string();
    Some((pct.clamp(0.0, 1.0), info))
}

fn parse_error_line(line: &str) -> Option<String> {
    line.strip_prefix("ERROR ").map(|m| m.trim().to_string())
}

/// Parse `META frames=<int>`. Returns the parsed count or `None` for
/// malformed / non-META lines. Negative values are rejected; the
/// response builder treats `total_frames <= 0` as "unknown".
fn parse_meta_frames_line(line: &str) -> Option<i32> {
    let rest = line.strip_prefix("META ")?.trim_start();
    let val_str = rest.strip_prefix("frames=")?.trim();
    let n: i32 = val_str.parse().ok()?;
    if n > 0 { Some(n) } else { None }
}

/// Resolve the python interpreter the worker should run under. The
/// launcher script in `effect_runner.py` activates the project venv
/// before exec'ing the Rust binary, so honoring `VIRTUAL_ENV` first
/// keeps the worker on the same interpreter the addon expects.
fn python_executable() -> PathBuf {
    if let Ok(p) = std::env::var("PPF_CTS_BUILD_PYTHON") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        if !venv.is_empty() {
            #[cfg(target_os = "windows")]
            let candidate = PathBuf::from(&venv).join("Scripts").join("python.exe");
            #[cfg(not(target_os = "windows"))]
            let candidate = PathBuf::from(&venv).join("bin").join("python");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    // Fall back to the bare command; the OS resolves it through PATH.
    #[cfg(target_os = "windows")]
    {
        PathBuf::from("python.exe")
    }
    #[cfg(not(target_os = "windows"))]
    {
        PathBuf::from("python3")
    }
}

/// Locate `frontend/build_worker.py`. Order:
///   1. `PPF_CTS_BUILD_WORKER` env var (deployment override).
///   2. `<cwd>/frontend/build_worker.py` (the launcher in
///      `effect_runner.py` cd's into the repo root before exec).
///   3. Walk up from `current_exe()` looking for
///      `frontend/build_worker.py`.
///
/// We intentionally prefer the cwd path over `current_exe()`: the
/// release binary lives at `<repo>/target/release/ppf-cts-server`
/// where the ancestor walk also succeeds, but a developer running
/// from a worktree (the .claude agent worktrees) may have a different
/// frontend they want to test, and they'll be in that worktree's
/// cwd, not the install root.
fn locate_build_worker() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PPF_CTS_BUILD_WORKER") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let p = cwd.join("frontend").join("build_worker.py");
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors() {
            let p = ancestor.join("frontend").join("build_worker.py");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Deliver a cancel signal to the build worker.
///
/// On Unix this sends SIGTERM, which the worker's signal handler
/// translates into `KeyboardInterrupt` so Python frames unwind
/// cleanly (finally blocks run, partially-written outputs get flushed,
/// exit code lands at 130). The cooperative path is the contract the
/// worker relies on for save-on-cancel semantics.
///
/// On Windows there is no SIGTERM. We route to `Child::start_kill`,
/// which calls `TerminateProcess`. The worker cannot catch it: Python
/// frames do not unwind, no `KeyboardInterrupt`, no finally blocks. The
/// process does exit, which is the contract `BuildOutcome::Cancelled`
/// requires from the caller's perspective. Operators on win_native
/// should expect that a cancel mid-build leaves no half-written
/// artifacts for the next launch to scrub. Manual smoke path: launch
/// a build on a Windows host, click Cancel, verify the addon transitions
/// out of Building (engine emits `BuildCancelledEvent`).
#[cfg(unix)]
async fn send_cancel_signal(child: &mut Child) {
    // SAFETY: kill(2) with SIGTERM is harmless on a non-existent pid
    // (returns ESRCH); we only care about the side-effect on our
    // child. We don't read errno because the loss of the signal is
    // covered by the subsequent wait().
    if let Some(pid) = child.id() {
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
    }
}

#[cfg(not(unix))]
async fn send_cancel_signal(child: &mut Child) {
    // `start_kill` issues TerminateProcess on Windows. The worker
    // cannot intercept it, so Python frames do not unwind, but the
    // process exits and `child.wait()` returns. That is enough for
    // the caller to surface `BuildOutcome::Cancelled`.
    if let Err(e) = child.start_kill() {
        log::warn!(target: "ppf::build", "[BUILD] start_kill failed on cancel: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_progress_line_extracts_percent_and_info() {
        let (p, info) = parse_progress_line("PROGRESS percent=0.42 info=Decoding scene")
            .expect("parses");
        assert!((p - 0.42).abs() < 1e-9);
        assert_eq!(info, "Decoding scene");
    }

    #[test]
    fn parse_progress_line_clamps_out_of_range_values() {
        let (p, _) = parse_progress_line("PROGRESS percent=1.5 info=overshoot").unwrap();
        assert_eq!(p, 1.0);
        let (p, _) = parse_progress_line("PROGRESS percent=-0.1 info=undershoot").unwrap();
        assert_eq!(p, 0.0);
    }

    #[test]
    fn parse_progress_line_accepts_empty_info() {
        let (p, info) = parse_progress_line("PROGRESS percent=0.5").unwrap();
        assert_eq!(p, 0.5);
        assert_eq!(info, "");
    }

    #[test]
    fn parse_progress_line_rejects_non_progress_lines() {
        assert!(parse_progress_line("ERROR boom").is_none());
        assert!(parse_progress_line("random log").is_none());
        assert!(parse_progress_line("PROGRESS bogus").is_none());
    }

    #[test]
    fn parse_meta_frames_line_extracts_count() {
        assert_eq!(parse_meta_frames_line("META frames=180"), Some(180));
        assert_eq!(parse_meta_frames_line("META frames=1"), Some(1));
    }

    #[test]
    fn parse_meta_frames_line_rejects_non_meta_or_zero() {
        assert!(parse_meta_frames_line("META frames=0").is_none());
        assert!(parse_meta_frames_line("META frames=-5").is_none());
        assert!(parse_meta_frames_line("META frames=abc").is_none());
        assert!(parse_meta_frames_line("PROGRESS percent=1.0").is_none());
        assert!(parse_meta_frames_line("META other=180").is_none());
    }

    #[test]
    fn parse_error_line_strips_prefix_and_trims() {
        assert_eq!(parse_error_line("ERROR boom").unwrap(), "boom");
        assert_eq!(
            parse_error_line("ERROR   tetwild crashed  ").unwrap(),
            "tetwild crashed"
        );
        assert!(parse_error_line("PROGRESS percent=1.0").is_none());
    }

    #[test]
    fn locate_build_worker_honors_env_override() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("custom_worker.py");
        std::fs::write(&p, "# stub").unwrap();

        // Save + restore so we don't leak into other tests.
        let prior = std::env::var_os("PPF_CTS_BUILD_WORKER");
        std::env::set_var("PPF_CTS_BUILD_WORKER", &p);
        let found = locate_build_worker();
        match prior {
            Some(v) => std::env::set_var("PPF_CTS_BUILD_WORKER", v),
            None => std::env::remove_var("PPF_CTS_BUILD_WORKER"),
        }
        assert_eq!(found.as_deref(), Some(p.as_path()));
    }

    /// Windows-only smoke for the TerminateProcess path. Spawns a
    /// long-running native command and verifies `send_cancel_signal`
    /// causes it to exit. Cannot share the `/bin/sh` mock harness
    /// in mod.rs's tests because Windows runners don't ship with it.
    /// The full `drive_build_worker` cancel flow on Windows is
    /// exercised manually on a Windows host (see `send_cancel_signal`
    /// doc comment).
    #[cfg(windows)]
    #[tokio::test]
    async fn send_cancel_signal_terminates_child_on_windows() {
        use std::time::Duration;
        // `ping -n 30 127.0.0.1` runs ~30s if left alone; we cancel
        // within ms to keep the test budget tight.
        let mut child = Command::new("cmd")
            .args(["/C", "ping", "-n", "30", "127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("cmd ping must spawn on Windows");

        send_cancel_signal(&mut child).await;
        // wait() must return; if TerminateProcess didn't take, the
        // test would hang and CI would flag it.
        let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .expect("child did not exit within 5s of TerminateProcess")
            .expect("wait failed");
        assert!(!status.success(), "TerminateProcess'd child reported success");
    }
}
