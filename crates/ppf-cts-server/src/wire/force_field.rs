// File: crates/ppf-cts-server/src/wire/force_field.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `force_field_check`: compile a force-field script with the frontend this
// server builds scenes with, and answer whether it compiles, without a build.
//
// THE ANSWER COMES FROM THE SAME COMPILER A TRANSFER WOULD RUN, which is the
// point of asking the server rather than the add-on: the add-on ships no
// frontend, and a check that ran a different compiler than the build could
// pass a script the build then refuses. The worker is `python -m
// frontend._force_field_ check`, under the interpreter and the frontend the
// build worker uses, reading one JSON request on stdin and writing one JSON
// answer on stdout.

use std::collections::HashMap;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use super::write_response;
use crate::error::ServerError;
use crate::executor::build::{locate_build_worker, python_executable};

/// A script longer than this is not a force-field formula.
const MAX_SOURCE_BYTES: usize = 256 * 1024;
/// The compile and its cross-check take milliseconds; this bounds a worker
/// that hangs so the add-on's request cannot.
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) async fn handle_force_field_check<W>(
    writer: &mut W,
    req: &HashMap<String, Value>,
) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let Some(source) = req.get("source").and_then(Value::as_str) else {
        let resp = ServerError::BadRequest("Missing source".into()).into_response();
        return write_response(writer, &resp).await;
    };
    if source.len() > MAX_SOURCE_BYTES {
        let resp = ServerError::BadRequest(format!(
            "the script is {} bytes, past the {MAX_SOURCE_BYTES}-byte limit",
            source.len()
        ))
        .into_response();
        return write_response(writer, &resp).await;
    }
    let request = json!({
        "source": source,
        "z_up": req.get("z_up").and_then(Value::as_bool).unwrap_or(false),
    });
    let resp = match run_worker(&request).await {
        Ok(answer) => json!({"status": "ok", "result": answer}),
        Err(message) => ServerError::BadRequest(message).into_response(),
    };
    write_response(writer, &resp).await
}

async fn run_worker(request: &Value) -> Result<Value, String> {
    let worker = locate_build_worker().ok_or_else(|| {
        "the frontend was not found (set PPF_CTS_BUILD_WORKER or install \
         frontend/build_worker.py)"
            .to_string()
    })?;
    let repo_root = worker
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| format!("{} has no repository root", worker.display()))?
        .to_path_buf();
    let (python, _) = python_executable();
    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg("frontend._force_field_")
        .arg("check")
        .current_dir(&repo_root)
        .env("PYTHONPATH", &repo_root)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot run {}: {e}", python.display()))?;
    let mut stdin = child.stdin.take().expect("stdin was piped");
    stdin
        .write_all(request.to_string().as_bytes())
        .await
        .map_err(|e| format!("cannot write the request to the worker: {e}"))?;
    drop(stdin);
    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut stderr = child.stderr.take().expect("stderr was piped");
    let collect = async {
        let mut out = String::new();
        let mut err = String::new();
        let _ = stdout.read_to_string(&mut out).await;
        let _ = stderr.read_to_string(&mut err).await;
        let status = child.wait().await;
        (out, err, status)
    };
    let (out, err, status) = tokio::time::timeout(WORKER_TIMEOUT, collect)
        .await
        .map_err(|_| format!("the compile worker did not answer in {WORKER_TIMEOUT:?}"))?;
    let status = status.map_err(|e| format!("the compile worker failed: {e}"))?;
    if !status.success() {
        let tail: String = err.lines().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        return Err(format!("the compile worker exited with {status}: {tail}"));
    }
    serde_json::from_str(out.trim())
        .map_err(|e| format!("the compile worker answered {out:?}, which is not JSON: {e}"))
}
