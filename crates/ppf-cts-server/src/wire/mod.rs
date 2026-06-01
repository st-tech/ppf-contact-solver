// File: crates/ppf-cts-server/src/wire/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Per-connection wire-protocol handler on top of `tokio::net::TcpStream`.
//
// Three paths after the 4-byte header:
//   * TCMD: text command line, dispatch to engine, send JSON status.
//   * JSON: JSON request line; sub-handler keyed by `request` field
//     (upload_atomic, data_send, data_receive, notebook_send,
//     notebook_delete).
//   * BDAT: drain bytes, ack with `BINARY_OK`.
//
// The TCMD response shape is the protocol contract: every status
// poll the addon makes goes through this code.

use std::collections::HashMap;

use ppf_cts_core::events::Event;
use ppf_cts_formats::files::{DATA_PICKLE, PARAM_PICKLE, SCENE_INFO_JSON};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::easy_parse::{self, get as get_arg};
use crate::engine::ServerEngine;
use crate::error::ServerError;
use crate::executor::{dispatch_with_executor, EffectExecutor};
use crate::protocol::{
    read_header, read_line, write_json_line, HEADER_BDAT, HEADER_JSON, HEADER_TCMD,
};
use crate::response::build_response;

mod data;
mod notebook;
mod upload;

const MAX_TCMD_BYTES: usize = 64 * 1024;
const MAX_JSON_LINE_BYTES: usize = 64 * 1024;

/// Run the per-connection handler to completion. Errors during
/// reading/writing log and return; the caller drops the socket. Owns
/// the socket halves so we can split read/write across awaits.
pub async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer: std::net::SocketAddr,
    engine: ServerEngine,
    executor: std::sync::Arc<dyn EffectExecutor>,
) {
    let (mut reader, mut writer) = stream.into_split();
    let header = match read_header(&mut reader).await {
        Ok(Some(h)) => h,
        Ok(None) => {
            log::debug!(target: "ppf::wire", "{peer}: clean disconnect before header");
            return;
        }
        Err(e) => {
            log::warn!(target: "ppf::wire", "{peer}: header read: {e}");
            return;
        }
    };

    let result = if &header == HEADER_TCMD {
        handle_tcmd(&mut reader, &mut writer, &engine, executor.as_ref(), peer).await
    } else if &header == HEADER_JSON {
        handle_json(&mut reader, &mut writer, &engine, executor.as_ref(), peer).await
    } else if &header == HEADER_BDAT {
        data::handle_bdat(&mut reader, &mut writer, peer).await
    } else {
        log::warn!(target: "ppf::wire", "{peer}: unknown header {:?}", String::from_utf8_lossy(&header));
        let payload = ServerError::UnknownHeader(hex_string(&header)).into_response();
        let _ = write_json_line(&mut writer, &payload).await;
        Ok(())
    };

    if let Err(e) = result {
        log::warn!(target: "ppf::wire", "{peer}: handler error: {e}");
    }
}

fn hex_string(buf: &[u8]) -> String {
    let mut s = String::with_capacity(buf.len() * 2);
    for b in buf {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// TCMD path

async fn handle_tcmd<R, W>(
    reader: &mut R,
    writer: &mut W,
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    peer: std::net::SocketAddr,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    // TCMD wire: the 4-byte header is followed by a big-endian u32
    // length prefix, then exactly that many payload bytes (the
    // `--key value` argument string). Earlier versions of this
    // handler read until EOF instead, relying on the client calling
    // shutdown(SHUT_WR) to signal end-of-input: that worked on POSIX
    // but on Windows tokio doesn't reliably surface the half-close
    // as Ok(0) to AsyncRead, so the read loop hung forever and
    // connections piled up in FIN_WAIT_2 state until the server
    // stopped accepting new requests entirely.
    let mut len_buf = [0u8; 4];
    if let Err(e) = reader.read_exact(&mut len_buf).await {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            log::debug!(target: "ppf::wire", "{peer}: empty TCMD (no length prefix)");
            return Ok(());
        }
        return Err(e);
    }
    let payload_len = u32::from_be_bytes(len_buf) as usize;
    if payload_len > MAX_TCMD_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "TCMD payload length {payload_len} exceeds max {MAX_TCMD_BYTES}"
            ),
        ));
    }
    let buf = if payload_len == 0 {
        Vec::new()
    } else {
        crate::protocol::read_exact_n_chunked(reader, payload_len).await?
    };

    if buf.is_empty() {
        log::debug!(target: "ppf::wire", "{peer}: empty TCMD");
        return Ok(());
    }

    let line = match std::str::from_utf8(&buf) {
        Ok(s) => s,
        Err(e) => {
            let resp = ServerError::TextDecode(e.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    let args = easy_parse::parse(line);

    let name = match get_arg(&args, "name") {
        Some(n) => n.to_string(),
        None => {
            let resp = ServerError::NoId.into_response();
            return write_response(writer, &resp).await;
        }
    };

    // The server (not the addon) decides where project data lives:
    // `make_root` pins every project under
    // `~/.local/share/ppf-cts/git-<branch>/<name>` (POSIX) or the
    // Windows / `PPF_CTS_DATA_ROOT` equivalent. The canonical root
    // flows back to the addon via this status response's `root`
    // field, which it stores in `state.remote_root` and uses for
    // subsequent uploads.
    let root = crate::upload::make_root(&name, engine.config())
        .to_string_lossy()
        .into_owned();
    // Hydrate engine state from disk before dispatching: read the
    // upload identity, hashes, and on-disk artifact presence so the
    // engine's `data` and `build` fields reflect what the previous
    // process (or an external builder like JupyterLab) left behind.
    // Without this, a server restart against an existing project
    // would report `data="NO_DATA"` even with `data.pickle` and
    // `app_state.pickle` on disk.
    let project = reconcile_project_from_disk(&name, &root);
    dispatch_with_executor(engine, executor, project).await;

    if let Some(req) = get_arg(&args, "request") {
        log::debug!(target: "ppf::wire", "{peer}: TCMD request name={name} request={req}");
        if let Some(event) = tcmd_request_to_event(req) {
            dispatch_with_executor(engine, executor, event).await;
        } else {
            log::warn!(target: "ppf::wire", "{peer}: unknown TCMD request {req}");
        }
    } else {
        log::debug!(target: "ppf::wire", "{peer}: TCMD ping name={name}");
    }

    let mut response = build_response(&engine.state(), engine.config());
    if let Value::Object(ref mut obj) = response {
        obj.entry("stdout".to_string()).or_insert_with(|| Value::String(String::new()));
        obj.entry("stderr".to_string()).or_insert_with(|| Value::String(String::new()));
    }
    write_response(writer, &response).await
}

/// Build a `ProjectSelected` event by reading on-disk artifacts:
/// `data.pickle` and `param.pickle` presence, the `upload_id.txt`
/// identity stamp, the data and param hash files, and
/// `app_state.pickle` (the build_worker's "make() succeeded"
/// marker). `is_resumable` falls out of `project_resumable`, which
/// scans for the solver's `state_<N>.bin.gz` checkpoints. The
/// transition layer then advances `state.data` / `state.build` so
/// the addon's status string reflects what survived the restart.
fn reconcile_project_from_disk(name: &str, root: &str) -> Event {
    use std::path::PathBuf;

    let root_path = PathBuf::from(root);
    let has_data = root_path.join(DATA_PICKLE).exists();
    let has_param = root_path.join(PARAM_PICKLE).exists();
    // build_worker writes app_state.pickle at the tail end of make();
    // its presence is the "build is fresh" signal.
    let has_app = root_path.join("app_state.pickle").exists();
    let is_resumable =
        ppf_cts_core::datamodel::project_resumable(&root_path);

    let upload_id = if has_data && has_param {
        crate::upload::read_upload_id(&root_path).unwrap_or_default()
    } else {
        String::new()
    };
    let data_hash = crate::upload::read_data_hash(&root_path);
    let param_hash = crate::upload::read_param_hash(&root_path);
    // BuildMetadata is the only event that normally sets total_frames,
    // and it only fires during a fresh build. So a re-select of a
    // previously-built project (typically the connect probe routing
    // through __probe__ and back to the real project) leaves
    // total_frames at 0 -- which makes the response shape suppress
    // the progress field entirely (shape::insert_progress requires
    // total_frames > 0), and the addon's progress bar sits at 0.
    // Read it back from scene_info.json's "Total Frames" so the
    // transition layer can rehydrate the field without forcing a
    // rebuild.
    let total_frames = read_total_frames_from_scene_info(&root_path);

    Event::ProjectSelected {
        name: name.to_string(),
        root: root.to_string(),
        has_data,
        has_param,
        has_app,
        is_resumable,
        upload_id,
        data_hash,
        param_hash,
        total_frames,
    }
}

/// Parse `<root>/scene_info.json`'s `"Total Frames"` string into an
/// `i32`. Returns 0 when the file is missing, malformed, or the field
/// is absent / unparseable -- same fallback the transition uses for a
/// just-uploaded project that has never been built.
fn read_total_frames_from_scene_info(root: &std::path::Path) -> i32 {
    let path = root.join(SCENE_INFO_JSON);
    let text = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return 0,
    };
    v.get("Total Frames")
        .and_then(|x| x.as_str())
        .and_then(|s| s.replace(',', "").parse::<i32>().ok())
        .unwrap_or(0)
}

/// Map a TCMD `request` string to an engine `Event`.
fn tcmd_request_to_event(req: &str) -> Option<Event> {
    Some(match req {
        "build" => Event::BuildRequested,
        "cancel_build" => Event::CancelBuildRequested,
        "start" => Event::StartRequested,
        "resume" => Event::ResumeRequested,
        "terminate" => Event::TerminateRequested,
        "save_and_quit" => Event::SaveAndQuitRequested,
        "delete" => Event::DeleteRequested,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// JSON path

async fn handle_json<R, W>(
    reader: &mut R,
    writer: &mut W,
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    peer: std::net::SocketAddr,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let line_bytes = match read_line(reader, MAX_JSON_LINE_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            let resp = ServerError::JsonHeaderRead(e.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    let trimmed: &[u8] = line_bytes
        .strip_suffix(b"\n")
        .unwrap_or(&line_bytes);
    let req: HashMap<String, Value> = match serde_json::from_slice(trimmed) {
        Ok(v) => v,
        Err(e) => {
            let resp = ServerError::JsonParse(e.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };

    let req_type = req.get("request").and_then(Value::as_str).unwrap_or("");
    log::debug!(target: "ppf::wire", "{peer}: JSON request type={req_type}");

    match req_type {
        "upload_atomic" => upload::handle_upload_atomic(reader, writer, engine, executor, &req).await,
        "upload_notify" => upload::handle_upload_notify(writer, engine, executor, &req).await,
        "data_send" => data::handle_data_send(reader, writer, &req).await,
        "data_receive" => data::handle_data_receive(writer, &req).await,
        "notebook_send" => notebook::handle_notebook_send(reader, writer, &req).await,
        "notebook_delete" => notebook::handle_notebook_delete(writer, &req).await,
        other => {
            let resp = ServerError::UnknownRequest(other.to_string()).into_response();
            write_response(writer, &resp).await
        }
    }
}

// ---------------------------------------------------------------------------
// shared helpers used across submodules

pub(super) fn json_u64(v: Option<&Value>) -> u64 {
    match v {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0),
        Some(Value::String(s)) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

pub(super) async fn write_response<W>(writer: &mut W, response: &Value) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    write_json_line(writer, response).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcmd_event_map_covers_all_known_requests() {
        for r in [
            "build", "cancel_build", "start", "resume",
            "terminate", "save_and_quit", "delete",
        ] {
            assert!(tcmd_request_to_event(r).is_some(), "missing {r}");
        }
        assert!(tcmd_request_to_event("not_a_real_request").is_none());
    }
}
