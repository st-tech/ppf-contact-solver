// File: crates/ppf-cts-server/src/executor/build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Build pipeline plumbing for `DoSpawnBuild`. The build runs in a
// Python subprocess (`frontend/build_worker.py`). The Rust side
// spawns the worker, parses line-oriented progress + error markers
// from its stdout, and forwards a SIGTERM on cooperative cancel.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use ppf_cts_core::cancel::CancelHandle;
use ppf_cts_core::events::Event;

use super::dispatch_re_entrant;
use crate::engine::ServerEngine;

/// Worker stderr lines retained for the failure message. Only used when
/// the worker produced no `ERROR` line of its own (a native crash, where
/// `faulthandler`'s dump on stderr is the whole diagnostic).
const STDERR_TAIL_LINES: usize = 32;

/// Lines retained from a native-crash dump, counted from its banner. A
/// fixed-size tail is the wrong shape for this payload: the dump carries
/// one section per thread, so a worker with a thread pool overruns
/// `STDERR_TAIL_LINES` and the banner and first frames, the part that
/// names the fault, are exactly what gets evicted. Latching at the banner
/// keeps the head of the dump however it ends, including when it ends
/// mid-line because the interpreter faulted again on its way out.
const CRASH_DUMP_MAX_LINES: usize = 256;

/// Banners a hosted interpreter writes when it dies without raising:
/// `faulthandler` emits the first on POSIX and the second on Windows.
const CRASH_BANNERS: [&str; 2] = ["Fatal Python error", "Windows fatal exception"];

/// How long the stderr drain is joined for once the worker has exited.
/// The join orders the drain's last lines before the tail is read; the
/// bound covers a descendant that inherited the pipe and outlives the
/// worker, which leaves the drain with no EOF to wait for.
const STDERR_DRAIN_JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// How often the stdout drain checks whether the worker has exited.
const WORKER_EXIT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// How long the stdout drain keeps reading after the worker has exited
/// and the stream has gone quiet.
///
/// Everything the worker wrote is in the pipe by the time it exits, so
/// this bounds how long a descendant holding the same write end can keep
/// the drain waiting. It is measured from the last byte read, not from
/// the exit, so a worker still flushing is never cut off mid-report.
const POST_EXIT_DRAIN_GRACE: Duration = Duration::from_millis(500);

/// Cap on the `ERRORDETAIL` traceback appended to a failure reason. The
/// reason is re-serialized into every status response, so the server
/// bounds it in bytes; the worker separately bounds it in lines for
/// legibility. Different quantities, so neither constant crosses the
/// language boundary.
pub(super) const MAX_ERROR_DETAIL_BYTES: usize = 8192;

/// Spawn the build task and install a cancel handle on the engine.
/// The body runs the build in a Python subprocess
/// (`frontend/build_worker.py`) via `run_build_pipeline` ->
/// `drive_build_worker`, translating the worker's stdout
/// PROGRESS/META/ERROR markers into `BuildProgress`/`BuildMetadata`/
/// `BuildFailed` events and forwarding a SIGTERM on cooperative cancel.
pub(super) fn spawn_build_task(engine: &ServerEngine, preserve_output: bool) {
    let cancel = engine.install_cancel_handle();
    let engine = engine.clone();

    tokio::spawn(async move {
        // Clear any sidecar left by a previous build so a later failure
        // that produces no structured violations (e.g. a tetwild crash)
        // can't inherit stale geometry. `root` is the project dir the
        // worker also writes `build_violations.json` to. We snapshot
        // name/root once here and thread the same values into the
        // pipeline so the sidecar clear/read and the worker's write
        // cannot disagree if the project context flips mid-build.
        let (name, root) = {
            let s = engine.state();
            (s.name, s.root)
        };
        clear_build_violations(&root);
        // Drop a prior run's status record so the post-rebuild status reads
        // READY/RESUMABLE, not the stale "Failed" a reconnect would
        // otherwise reconstruct from a previous run's status.cbor between
        // build-done and the next run. The next launch scrubs it again;
        // clearing here closes the build-done .. run window.
        clear_stale_status(&root);

        let result =
            run_build_pipeline(&engine, cancel.clone(), &name, &root, preserve_output).await;

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
                // The worker persists ValidationError's structured
                // geometry to a sidecar; pull it in so the add-on can
                // highlight the offending faces in the viewport.
                let violations = read_build_violations(&root);
                dispatch_re_entrant(&engine, Event::BuildFailed { error, violations }).await;
            }
            BuildOutcome::AlreadyDispatched => {
                // Pipeline body owns the terminal dispatch (e.g.
                // GpuCheckFailed). No further event from us.
            }
        }

        engine.clear_cancel_handle();
    });
}

/// Remove a stale `<root>/build_violations.json` before a build runs so
/// a later failure that produces no structured violations can't inherit
/// the geometry from a previous self-intersection failure.
fn clear_build_violations(root: &str) {
    if root.is_empty() {
        return;
    }
    let path = Path::new(root).join("build_violations.json");
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => log::warn!(
            target: "ppf::build",
            "[BUILD] could not clear build_violations.json: {e}",
        ),
    }
}

/// Remove stale status markers (`status.cbor`, `terminate_request`) before a
/// build runs so a clean
/// rebuild doesn't leave a reconnect reconstructing `Solver::Failed` from
/// a previous run's status record. Best-effort, mirrors
/// `clear_build_violations`.
fn clear_stale_status(root: &str) {
    if root.is_empty() {
        return;
    }
    let out = ppf_cts_formats::files::session_output_dir(Path::new(root));
    for f in [
        ppf_cts_formats::files::STATUS_RECORD,
        ppf_cts_formats::files::TERMINATE_REQUEST,
    ] {
        let _ = std::fs::remove_file(out.join(f));
    }
}

/// Read the structured violation payload the worker wrote to
/// `<root>/build_violations.json` when scene validation failed. Mirrors
/// `monitor::read_intersection_violations`: the state machine carries
/// violations as `Vec<String>` (opaque payload), so each violation dict
/// is JSON-encoded into one entry; the response builder
/// (`response::shape::violations_to_json`) re-parses them into nested
/// JSON the add-on's overlay consumes. Best-effort: any I/O or parse
/// error yields an empty list so the build still fails cleanly with its
/// error message, just without the viewport highlight.
fn read_build_violations(root: &str) -> Vec<String> {
    if root.is_empty() {
        return vec![];
    }
    let path = Path::new(root).join("build_violations.json");
    let body = match std::fs::read_to_string(&path) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    let parsed: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            log::warn!(target: "ppf::build", "[BUILD] malformed build_violations.json: {e}");
            return vec![];
        }
    };
    parsed
        .get("violations")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|rec| serde_json::to_string(rec).ok())
                .collect()
        })
        .unwrap_or_default()
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
async fn run_build_pipeline(
    engine: &ServerEngine,
    cancel: CancelHandle,
    name: &str,
    root: &str,
    preserve_output: bool,
) -> BuildOutcome {
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
    let (python, source) = python_executable();
    let outcome =
        drive_build_worker(engine, cancel, &python, &worker, name, root, preserve_output).await;
    // Rewrite a bare missing-dependency failure into an actionable
    // message about the interpreter / venv before it surfaces to the user.
    if let BuildOutcome::Failed(reason) = outcome {
        return BuildOutcome::Failed(enrich_build_failure(reason, &python, source));
    }
    outcome
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
    preserve_output: bool,
) -> BuildOutcome {
    let mut cmd = Command::new(python);
    cmd.arg(worker)
        .arg(name)
        .arg(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Line-buffer Python so PROGRESS lines arrive promptly.
        .env("PYTHONUNBUFFERED", "1")
        // Pin the worker's stdio to UTF-8. The worker reconfigures its
        // streams at import as well; this applies before the interpreter
        // starts, so it also covers an import-time failure that raises
        // before that runs. Without it, Windows follows the console code
        // page and a non-ASCII object name or path makes the report raise
        // `UnicodeEncodeError` inside its own handler, costing us the
        // whole error line.
        .env("PYTHONIOENCODING", "utf-8");
    // Append `--preserve-output` after the positional `<name> <root>`
    // so it lands at worker argv[3] (parsed from argv[3:]) without
    // shifting the name/root positions the worker reads as argv[1]/[2].
    // This keeps `session/output/` checkpoints in place for a resume.
    if preserve_output {
        cmd.arg("--preserve-output");
    }
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
    // server log under [BUILD] so operators can correlate failures, and
    // keep the last `STDERR_TAIL_LINES` for a failure that reaches us
    // with no `ERROR` line to explain it.
    let stderr_tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
    // Separate from the tail on purpose: the tail answers "what was the
    // worker saying when it died", the dump answers "where did it die",
    // and only the second survives being truncated from the front.
    let crash_dump = Arc::new(Mutex::new(Vec::<String>::new()));
    let stderr_task = stderr.map(|err| {
        let name = name.to_string();
        let tail = stderr_tail.clone();
        let dump = crash_dump.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(err);
            let mut buf: Vec<u8> = Vec::new();
            loop {
                buf.clear();
                // `read_until` rather than `lines()`: `lines()` yields an
                // error on the first non-UTF-8 byte and the drain would
                // end there, dropping every later line including the
                // faulthandler dump a native crash writes on its way out.
                match reader.read_until(b'\n', &mut buf).await {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => {
                        log::warn!(
                            target: "ppf::build",
                            "[BUILD] {name}: stderr read error: {e}",
                        );
                        break;
                    }
                }
                let line = String::from_utf8_lossy(&buf)
                    .trim_end_matches('\n')
                    .trim_end_matches('\r')
                    .to_string();
                log::warn!(target: "ppf::build", "[BUILD stderr] {name}: {line}");
                {
                    let mut held = dump.lock().unwrap_or_else(|e| e.into_inner());
                    if held.is_empty() {
                        if CRASH_BANNERS.iter().any(|b| line.contains(b)) {
                            held.push(line.clone());
                        }
                    } else if held.len() < CRASH_DUMP_MAX_LINES {
                        held.push(line.clone());
                    }
                }
                let mut kept = tail.lock().unwrap_or_else(|e| e.into_inner());
                if kept.len() == STDERR_TAIL_LINES {
                    kept.pop_front();
                }
                kept.push_back(line);
            }
        })
    });

    let mut reader = BufReader::new(stdout);
    // `read_until` rather than `lines()`, for the same reason the stderr
    // drain gives, and the stakes are higher on this stream: `lines()`
    // yields an error on the first non-UTF-8 byte and the drain ends
    // there, so a library writing straight to fd 1 in the console code
    // page (cp932 on a Japanese Windows install) costs the worker its
    // whole report and not just the bytes it wrote. `from_utf8_lossy`
    // spends one garbled line instead.
    //
    // The buffer outlives an iteration on purpose: the cancel branch can
    // drop a half-finished `read_until`, which leaves the bytes it already
    // took in the buffer, and the next call then appends the rest of that
    // same line rather than losing its head.
    let mut line_buf: Vec<u8> = Vec::new();
    let mut error_reason: Option<String> = None;
    let mut error_detail: Vec<String> = Vec::new();
    let mut error_detail_bytes: usize = 0;
    let mut last_stage: Option<String> = None;
    let mut sigterm_sent = false;
    // EOF on this pipe means every process holding its write end has
    // closed it, which is not the same question as whether the worker
    // has exited: a descendant that inherited stdout holds it open, and
    // the drain then waits on a process nobody is supervising. So the
    // worker's own exit is observed separately, and the read that
    // follows it is bounded.
    let mut worker_exit: Option<std::process::ExitStatus> = None;
    let mut quiet_polls_after_exit: u32 = 0;
    let post_exit_quiet_polls =
        (POST_EXIT_DRAIN_GRACE.as_millis() / WORKER_EXIT_POLL_INTERVAL.as_millis()) as u32;

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
            read = reader.read_until(b'\n', &mut line_buf) => {
                if let Err(e) = read {
                    log::warn!(target: "ppf::build", "[BUILD] {name}: stdout read error: {e}");
                    break;
                }
                // `read_until` hands back whatever it holds when the stream
                // ends without a delimiter, so an unterminated final line
                // still arrives. An empty buffer is therefore EOF with
                // nothing left to deliver, and the byte count is not the
                // test: a call that reads zero bytes still owes us a line
                // the cancel branch interrupted.
                if line_buf.is_empty() {
                    break; // every holder of the write end closed it
                }
                quiet_polls_after_exit = 0;
                let text = String::from_utf8_lossy(&line_buf)
                    .trim_end_matches('\n')
                    .trim_end_matches('\r')
                    .to_string();
                line_buf.clear();
                if let Some((progress, info)) = parse_progress_line(&text) {
                    // Remember what the worker was doing, so an exit
                    // that carries no ERROR line can still name the
                    // stage it died in.
                    if !info.is_empty() {
                        last_stage = Some(info.clone());
                    }
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
                    // Detail lines follow their own headline, so a
                    // new headline discards the previous one's.
                    error_reason = Some(msg);
                    error_detail.clear();
                    error_detail_bytes = 0;
                } else if let Some(detail) = parse_error_detail_line(&text) {
                    if error_detail_bytes < MAX_ERROR_DETAIL_BYTES {
                        error_detail_bytes += detail.len();
                        error_detail.push(detail);
                        if error_detail_bytes >= MAX_ERROR_DETAIL_BYTES {
                            error_detail.push(format!(
                                "<error detail truncated at \
                                 {MAX_ERROR_DETAIL_BYTES} bytes; full \
                                 traceback in server.log>"
                            ));
                        }
                    }
                } else if !text.is_empty() {
                    log::debug!(target: "ppf::build", "[BUILD stdout] {name}: {text}");
                }
            }
            _ = tokio::time::sleep(WORKER_EXIT_POLL_INTERVAL) => {
                // Reached only while the stream is quiet, since the read
                // arm above is polled first and this loop is biased.
                if worker_exit.is_none() {
                    match child.try_wait() {
                        Ok(Some(status)) => worker_exit = Some(status),
                        Ok(None) => {}
                        Err(e) => {
                            log::warn!(
                                target: "ppf::build",
                                "[BUILD] {name}: could not poll worker exit: {e}",
                            );
                            break;
                        }
                    }
                } else {
                    quiet_polls_after_exit += 1;
                    if quiet_polls_after_exit >= post_exit_quiet_polls {
                        log::warn!(
                            target: "ppf::build",
                            "[BUILD] {name}: worker exited but its stdout is still \
                             held open after {}ms of silence, so a descendant \
                             inherited it; ending the drain",
                            POST_EXIT_DRAIN_GRACE.as_millis(),
                        );
                        break;
                    }
                }
            }
        }
    }

    // Wait for the exit
    // status so we can distinguish cancel (130) from crash (non-zero
    // without our SIGTERM) from clean completion (0).
    let status = match worker_exit {
        Some(s) => s,
        None => match child.wait().await {
            Ok(s) => s,
            Err(e) => return BuildOutcome::Failed(format!("worker wait failed: {e}")),
        },
    };
    // Join the drain so the tail read below cannot race the last lines
    // the worker wrote on its way out. Bounded, because the write end of
    // that pipe belongs to every process that inherited it and not to the
    // worker alone: a descendant the worker leaves behind holds it open,
    // the drain's `read_until` then has no EOF to reach, and an unbounded
    // join would strand the build in BUILDING with the cancel select loop
    // already exited, so cancel could not reach it either. Cancel on
    // Windows terminates the worker alone, which is one way to be left
    // with such a descendant. A timeout drops the handle, which detaches
    // the drain rather than aborting it, so its lines still reach the
    // server log; what the bound spends is their place in this failure's
    // stderr tail.
    if let Some(task) = stderr_task {
        let _ = tokio::time::timeout(STDERR_DRAIN_JOIN_TIMEOUT, task).await;
    }

    if sigterm_sent || cancel.is_cancelled() {
        return BuildOutcome::Cancelled;
    }
    if status.success() {
        return BuildOutcome::Completed;
    }
    let had_error_line = error_reason.is_some();
    let mut reason = error_reason.unwrap_or_else(|| {
        let exit = match status.code() {
            Some(code) => format!("build worker exited with code {code}"),
            None => "build worker terminated by signal".to_string(),
        };
        match last_stage.as_deref().filter(|s| !s.is_empty()) {
            Some(stage) => format!("{exit} (last stage: {stage})"),
            None => exit,
        }
    });
    if !error_detail.is_empty() {
        reason.push('\n');
        reason.push_str(&error_detail.join("\n"));
    } else if !had_error_line {
        // No structured report at all, which is what a native crash looks
        // like: the interpreter dies without raising, so `faulthandler`'s
        // stderr dump is the only diagnostic. Gated on the no-ERROR case
        // deliberately: the tetrahedralizer's C++ chatter reaches stderr
        // unredirected on Windows, so an always-on tail would bury a real
        // message under it.
        // Prefer the latched dump when the worker left one: it starts at
        // the banner, so it names the fault even when the tail has rolled
        // past it. Fall back to the tail for a death that printed no
        // banner at all (a signal from outside, an abort in a library
        // that never reached the interpreter's handler).
        let dump = {
            let held = crash_dump.lock().unwrap_or_else(|e| e.into_inner());
            held.join("\n")
        };
        if !dump.is_empty() {
            reason.push_str(&format!(
                "\n--- Build Worker native crash (from the {} banner, up to {CRASH_DUMP_MAX_LINES} lines) ---\n{dump}",
                "faulthandler",
            ));
        }
        let tail = {
            let kept = stderr_tail.lock().unwrap_or_else(|e| e.into_inner());
            kept.iter().cloned().collect::<Vec<_>>().join("\n")
        };
        if !tail.is_empty() {
            reason.push_str(&format!(
                "\n--- Build Worker stderr (last {STDERR_TAIL_LINES} lines) ---\n{tail}"
            ));
        }
    }
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

/// Parse `ERRORDETAIL <text>`: one traceback line belonging to the most
/// recent `ERROR`. Unlike `parse_error_line` this does NOT trim, because
/// the leading indentation is what makes a traceback readable and the
/// marker already delimits the payload. The two prefixes cannot collide:
/// `"ERROR "` requires a space where `ERRORDETAIL` has a `D`.
fn parse_error_detail_line(line: &str) -> Option<String> {
    line.strip_prefix("ERRORDETAIL ").map(|m| m.to_string())
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

/// How `python_executable` picked the interpreter, kept so a build
/// failure can explain which Python ran and why it was chosen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PythonSource {
    /// `PPF_CTS_BUILD_PYTHON` env override.
    Explicit,
    /// `VIRTUAL_ENV`'s interpreter (the addon launcher activates it).
    Venv,
    /// Bare `python3` / `python.exe` from PATH: neither the override nor
    /// a usable `VIRTUAL_ENV` was set. The likely cause of a missing
    /// frontend dependency, since the project deps live in the venv.
    PathFallback,
}

impl PythonSource {
    fn describe(self) -> &'static str {
        match self {
            PythonSource::Explicit => "PPF_CTS_BUILD_PYTHON",
            PythonSource::Venv => "the active VIRTUAL_ENV",
            PythonSource::PathFallback => {
                "PATH (neither PPF_CTS_BUILD_PYTHON nor a usable VIRTUAL_ENV was set)"
            }
        }
    }
}

/// Resolve the python interpreter the worker should run under, plus how
/// it was resolved (for diagnostics). Resolution order:
///   1. `PPF_CTS_BUILD_PYTHON` env var (explicit deployment/override;
///      honored first when non-empty).
///   2. `VIRTUAL_ENV`'s `bin/python` (Unix) or `Scripts/python.exe`
///      (Windows) when that file exists. The launcher script in
///      `blender_addon/core/effect_runner.py` activates the project
///      venv before exec'ing the Rust binary, so this keeps the worker
///      on the same interpreter the addon expects.
///   3. A bare `python3` / `python.exe` resolved through PATH.
fn python_executable() -> (PathBuf, PythonSource) {
    if let Ok(p) = std::env::var("PPF_CTS_BUILD_PYTHON") {
        if !p.is_empty() {
            return (PathBuf::from(p), PythonSource::Explicit);
        }
    }
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        if !venv.is_empty() {
            #[cfg(target_os = "windows")]
            let candidate = PathBuf::from(&venv).join("Scripts").join("python.exe");
            #[cfg(not(target_os = "windows"))]
            let candidate = PathBuf::from(&venv).join("bin").join("python");
            if candidate.exists() {
                return (candidate, PythonSource::Venv);
            }
            // VIRTUAL_ENV is set but its interpreter is missing: a
            // misconfiguration worth surfacing rather than silently
            // dropping to PATH.
            log::warn!(
                target: "ppf::build",
                "[BUILD] VIRTUAL_ENV={venv} set but {} does not exist; falling back to PATH",
                candidate.display(),
            );
        }
    }
    // Fall back to the bare command; the OS resolves it through PATH.
    #[cfg(target_os = "windows")]
    let bare = PathBuf::from("python.exe");
    #[cfg(not(target_os = "windows"))]
    let bare = PathBuf::from("python3");
    (bare, PythonSource::PathFallback)
}

/// Detect a Python import failure in a worker's error text.
fn is_missing_dependency_error(reason: &str) -> bool {
    reason.contains("ModuleNotFoundError")
        || reason.contains("No module named")
        || reason.contains("ImportError")
}

/// The part of a worker headline that identifies the failure: everything
/// before the first ` | `.
///
/// `build_worker.py` writes its headline as `<Type>: <message>` followed
/// by ` | `-separated context fields (`caused by ...`, the OS errno set,
/// the frontend frame, the stage, the interpreter). Only that first
/// segment names the exception the worker raised. `caused by` names a
/// SECOND exception, the root of the cause chain, and a
/// `ModuleNotFoundError` swallowed by an optional-dependency probe
/// (`try: import scipy` / `except ImportError:`) lands there whenever the
/// real failure is raised inside the handler.
///
/// A headline carrying no ` | ` comes from another producer (the usage
/// line, a `SystemExit` payload, a reason this file synthesized) and
/// identifies itself in full.
fn failure_identity(headline: &str) -> &str {
    match headline.split_once(" | ") {
        Some((identity, _)) => identity,
        None => headline,
    }
}

/// Turn a cryptic `ModuleNotFoundError: No module named 'pythreejs'`
/// into an actionable message that names the interpreter, how it was
/// chosen, and how to point the build at the project venv. Non-import
/// failures pass through unchanged.
///
/// Classification reads the first line's identity segment only
/// (`failure_identity`), which is the exception the worker raised.
/// Everything around it names some OTHER exception: the lines below the
/// headline are the traceback, which routinely mentions an unrelated
/// `ImportError` through the "During handling of the above exception"
/// chain, and the fields after the first ` | ` carry the root of that
/// chain by name. Matching over either would rewrite an unrelated failure
/// into venv boilerplate and state a confidently wrong diagnosis. The
/// whole headline and the traceback are both re-attached unchanged, so
/// narrowing the match costs the reader no context.
fn enrich_build_failure(reason: String, python: &Path, source: PythonSource) -> String {
    let (headline, detail) = match reason.split_once('\n') {
        Some((head, rest)) => (head, rest),
        None => (reason.as_str(), ""),
    };
    if !is_missing_dependency_error(failure_identity(headline)) {
        return reason;
    }
    let mut enriched = format!(
        "build worker's Python ({}, resolved from {}) is missing a required frontend \
         dependency: {headline}. The frontend deps (numpy, scipy, pythreejs, ...) live in the \
         ppf-cts venv. Point PPF_CTS_BUILD_PYTHON at that venv's interpreter (e.g. \
         <data-root>/venv/bin/python) or launch the server with the venv activated so \
         VIRTUAL_ENV is set, then rebuild.",
        python.display(),
        source.describe(),
    );
    if !detail.is_empty() {
        enriched.push('\n');
        enriched.push_str(detail);
    }
    enriched
}

/// Locate `frontend/build_worker.py`. Order:
///   1. `PPF_CTS_BUILD_WORKER` env var (deployment override).
///   2. `<cwd>/frontend/build_worker.py` (the launcher in
///      `blender_addon/core/effect_runner.py` cd's into the repo root
///      before exec).
///   3. Walk up from `current_exe()` looking for
///      `frontend/build_worker.py`.
///
/// We intentionally prefer the cwd path over `current_exe()`: the
/// release binary lives at `<repo>/target/release/ppf-cts-server`
/// where the ancestor walk also succeeds, but a developer running
/// from a separate worktree may have a different frontend they want
/// to test, and they'll be in that worktree's cwd, not the install
/// root.
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
/// a build on a Windows host, click Cancel, verify the addon
/// transitions out of Building (engine emits `BuildCancelledEvent`).
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
    fn enrich_build_failure_rewrites_missing_module() {
        let msg = enrich_build_failure(
            "ModuleNotFoundError: No module named 'pythreejs'".to_string(),
            Path::new("/usr/bin/python3"),
            PythonSource::PathFallback,
        );
        // Names the interpreter, the resolution source, the missing dep,
        // and the venv remedy.
        assert!(msg.contains("/usr/bin/python3"));
        assert!(msg.contains("PPF_CTS_BUILD_PYTHON"));
        assert!(msg.contains("pythreejs"));
        assert!(msg.contains("venv"));
    }

    #[test]
    fn enrich_build_failure_passes_through_non_import_errors() {
        let original = "tetwild failed: self-intersecting input".to_string();
        let msg = enrich_build_failure(
            original.clone(),
            Path::new("/x/venv/bin/python"),
            PythonSource::Venv,
        );
        assert_eq!(msg, original);
    }

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
    fn parse_error_detail_line_keeps_indentation() {
        // Traceback indentation is the structure of the dump, so unlike
        // the headline this payload must survive byte for byte.
        assert_eq!(
            parse_error_detail_line("ERRORDETAIL   File \"x.py\", line 3").unwrap(),
            "  File \"x.py\", line 3"
        );
        assert_eq!(parse_error_detail_line("ERRORDETAIL ").unwrap(), "");
        assert!(parse_error_detail_line("PROGRESS percent=1.0").is_none());
    }

    #[test]
    fn error_and_error_detail_prefixes_are_disjoint() {
        // Neither marker may consume the other: a detail line read as the
        // headline would replace the verdict with a traceback fragment.
        assert!(parse_error_line("ERRORDETAIL   File \"x.py\"").is_none());
        assert!(parse_error_detail_line("ERROR boom").is_none());
    }

    #[test]
    fn enrich_build_failure_classifies_on_headline_only() {
        // A traceback that merely mentions ImportError must not turn an
        // unrelated OSError into missing-dependency guidance.
        let original = concat!(
            "OSError: [Errno 22] Invalid argument | errno=22 winerror=None\n",
            "Traceback (most recent call last):\n",
            "  File \"frontend/_mesh_.py\", line 899, in _run_ftetwild_subprocess\n",
            "During handling of the above exception, another exception occurred:\n",
            "ImportError: cannot import name 'x'",
        )
        .to_string();
        let msg = enrich_build_failure(
            original.clone(),
            Path::new("/x/venv/bin/python"),
            PythonSource::Venv,
        );
        assert_eq!(msg, original);
    }

    #[test]
    fn enrich_build_failure_classifies_on_identity_segment_only() {
        // Verbatim worker headline for an `OSError` raised inside an
        // `except ImportError:` handler, so the swallowed probe becomes the
        // root of the cause chain and its type name lands in a context
        // field. The failure has nothing to do with a missing dependency.
        let original = concat!(
            "OSError: [Errno 22] Invalid argument ",
            "| caused by ModuleNotFoundError: No module named 'pytetwild' ",
            "| errno=22 winerror=None strerror='Invalid argument' ",
            "filename=None filename2=None ",
            "| at frontend/_mesh_.py:899 in _run_ftetwild_subprocess ",
            "| while: Tetrahedralizing Rock (1/1, new)... ",
            "| python 3.11.9 win32 at C:\\dev\\python.exe",
        )
        .to_string();
        let msg = enrich_build_failure(
            original.clone(),
            Path::new("/x/venv/bin/python"),
            PythonSource::Venv,
        );
        assert_eq!(msg, original);
    }

    #[test]
    fn enrich_build_failure_still_rewrites_a_headline_carrying_context() {
        // The narrowing must not cost a genuine import failure its
        // guidance: the identity segment still carries the type name when
        // the context fields follow it.
        let msg = enrich_build_failure(
            concat!(
                "ModuleNotFoundError: No module named 'pytetwild' ",
                "| at frontend/_mesh_.py:820 in tetrahedralize ",
                "| python 3.11.9 win32 at C:\\dev\\python.exe",
            )
            .to_string(),
            Path::new("/usr/bin/python3"),
            PythonSource::PathFallback,
        );
        assert!(msg.contains("pytetwild"), "lost the module: {msg:?}");
        assert!(msg.contains("venv"), "lost the remedy: {msg:?}");
        // The context fields survive; only the classifier ignores them.
        assert!(
            msg.contains("at frontend/_mesh_.py:820 in tetrahedralize"),
            "lost the context fields: {msg:?}"
        );
    }

    #[test]
    fn enrich_build_failure_preserves_detail_block() {
        let original = concat!(
            "ModuleNotFoundError: No module named 'pytetwild'\n",
            "Traceback (most recent call last):\n",
            "  File \"frontend/_mesh_.py\", line 850, in _run_ftetwild_subprocess",
        )
        .to_string();
        let msg = enrich_build_failure(
            original,
            Path::new("/usr/bin/python3"),
            PythonSource::PathFallback,
        );
        let mut lines = msg.split('\n');
        let head = lines.next().unwrap();
        assert!(head.contains("pytetwild"), "head lost the module: {head:?}");
        assert!(head.contains("venv"), "head lost the remedy: {head:?}");
        assert_eq!(
            lines.collect::<Vec<_>>(),
            vec![
                "Traceback (most recent call last):",
                "  File \"frontend/_mesh_.py\", line 850, in _run_ftetwild_subprocess",
            ],
        );
    }

    #[test]
    fn read_build_violations_parses_sidecar_into_json_strings() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        std::fs::write(
            dir.path().join("build_violations.json"),
            r#"{"violations":[{"type":"self_intersection","count":2,"tris":[[[0,0,0],[1,0,0],[0,1,0]],[[0,0,1],[1,0,1],[0,1,1]]]}]}"#,
        )
        .unwrap();

        let out = read_build_violations(&root);
        assert_eq!(out.len(), 1);
        // Each entry round-trips back into a violation dict the response
        // builder can re-parse and the add-on overlay can draw.
        let v: serde_json::Value = serde_json::from_str(&out[0]).unwrap();
        assert_eq!(v["type"], "self_intersection");
        assert_eq!(v["count"], 2);
        assert!(v["tris"].as_array().is_some());
    }

    #[test]
    fn read_build_violations_absent_or_malformed_yields_empty() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        // No file yet.
        assert!(read_build_violations(&root).is_empty());
        // Malformed JSON degrades to empty rather than erroring.
        std::fs::write(dir.path().join("build_violations.json"), "{not json").unwrap();
        assert!(read_build_violations(&root).is_empty());
        // Empty root short-circuits.
        assert!(read_build_violations("").is_empty());
    }

    #[test]
    fn clear_build_violations_removes_stale_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        let path = dir.path().join("build_violations.json");
        std::fs::write(&path, r#"{"violations":[]}"#).unwrap();
        assert!(path.exists());
        clear_build_violations(&root);
        assert!(!path.exists());
        // Idempotent: clearing an already-absent file is a no-op.
        clear_build_violations(&root);
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

    /// A descendant that inherited the worker's stderr holds the write
    /// end of that pipe open after the worker itself exits, so the drain
    /// never reaches EOF. The join over it is bounded, so the build still
    /// reports its outcome; an unbounded one returns only when the
    /// descendant does, and the engine sits in BUILDING until then with
    /// the cancel select loop already exited.
    #[cfg(unix)]
    #[tokio::test]
    async fn stderr_drain_join_survives_a_lingering_descendant() {
        use crate::config::EngineConfig;
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("mock_lingering_child.sh");
        // The descendant keeps stderr and gives up stdout, so the stdout
        // loop still reaches EOF and the run gets as far as the join. It
        // outlives the bound below by enough that the join cannot be the
        // thing that ends it.
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             sleep 20 >/dev/null &\n\
             echo 'ERROR ValueError: Plane: no enclosed volume'\n\
             exit 1\n",
        )
        .unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();

        let engine = ServerEngine::new(EngineConfig::default());
        let cancel = engine.install_cancel_handle();
        let outcome = tokio::time::timeout(
            STDERR_DRAIN_JOIN_TIMEOUT * 3,
            drive_build_worker(
                &engine,
                cancel,
                Path::new("/bin/sh"),
                &script,
                "demo",
                "/tmp/demo",
                false,
            ),
        )
        .await
        .expect("drive_build_worker never returned: the stderr drain join is unbounded");
        match outcome {
            // The worker's own report survives the bounded join.
            BuildOutcome::Failed(msg) => {
                assert_eq!(msg, "ValueError: Plane: no enclosed volume");
            }
            _ => panic!("expected Failed"),
        }
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
