// File: crates/ppf-cts-server/src/wire/upload.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Atomic upload sub-handler. Handles the `upload_atomic` JSON
// request: receive optional `data.pickle` + `param.pickle` payloads,
// stage them under tempfile names, rename into place, then dispatch
// `UploadLanded` so the engine can advance.

use std::collections::HashMap;

use ppf_cts_core::events::Event;
use ppf_cts_formats::files::{DATA_PICKLE, PARAM_PICKLE};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{json_u64, write_response, MAX_PAYLOAD_BYTES};
use crate::engine::ServerEngine;
use crate::error::ServerError;
use crate::executor::{dispatch_with_executor, EffectExecutor};
use crate::protocol::read_exact_n_chunked;
use crate::upload;

pub(super) async fn handle_upload_atomic<R, W>(
    reader: &mut R,
    writer: &mut W,
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    req: &HashMap<String, Value>,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    // The client's `path` field is accepted for protocol compatibility
    // but ignored for the actual write location: the server resolves
    // the project root via `make_root(name)` so every upload lands at
    // the canonical `~/.local/share/ppf-cts/git-<branch>/<name>` (or
    // `PPF_CTS_DATA_ROOT` equivalent), regardless of what
    // `current_directory` the addon connected with. The addon mirrors
    // the canonical path back via `state.remote_root` after the next
    // status round-trip.
    let _path = req.get("path").and_then(Value::as_str).unwrap_or("");
    let name = req.get("name").and_then(Value::as_str).unwrap_or("");
    let data_size = json_u64(req.get("data_size"));
    let param_size = json_u64(req.get("param_size"));
    let data_hash = req
        .get("data_hash")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let param_hash = req
        .get("param_hash")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    if name.is_empty() {
        let resp = ServerError::BadRequest("Missing name".into()).into_response();
        return write_response(writer, &resp).await;
    }
    if data_size == 0 && param_size == 0 {
        let resp = ServerError::BadRequest(
            "upload_atomic requires at least one of data_size or param_size to be positive".into(),
        )
        .into_response();
        return write_response(writer, &resp).await;
    }
    // Bound each declared size, and their sum, on the u64 before the
    // `as usize` casts feed read_exact_n_chunked. The sum check guards
    // against two individually-legal sizes that together still ask for
    // an outsized total transfer; `saturating_add` keeps the
    // comparison sound at the u64 ceiling.
    if data_size > MAX_PAYLOAD_BYTES
        || param_size > MAX_PAYLOAD_BYTES
        || data_size.saturating_add(param_size) > MAX_PAYLOAD_BYTES
    {
        let resp = ServerError::BadRequest(format!(
            "declared payload (data_size {data_size}, param_size {param_size}) exceeds max payload {MAX_PAYLOAD_BYTES}"
        ))
        .into_response();
        return write_response(writer, &resp).await;
    }
    if engine.state().build == ppf_cts_core::state::Build::Building {
        let resp = ServerError::Conflict(
            "Cannot upload while a build is in progress. Abort the build first, then retry the upload.".into(),
        )
        .into_response();
        return write_response(writer, &resp).await;
    }

    let project_root = crate::upload::make_root(name, engine.config());
    if let Err(e) = tokio::fs::create_dir_all(&project_root).await {
        let resp = ServerError::Internal(format!(
            "create_dir_all({}): {}",
            project_root.display(),
            e
        ))
        .into_response();
        return write_response(writer, &resp).await;
    }

    let stamp = upload::temp_suffix();
    let data_final = project_root.join(DATA_PICKLE);
    let param_final = project_root.join(PARAM_PICKLE);
    let data_tmp = project_root.join(format!("{DATA_PICKLE}.tmp.{stamp}"));
    let param_tmp = project_root.join(format!("{PARAM_PICKLE}.tmp.{stamp}"));

    let mut data_tmp_exists = false;
    let mut param_tmp_exists = false;

    let result: std::io::Result<()> = async {
        if data_size > 0 {
            let payload = read_exact_n_chunked(reader, data_size as usize).await?;
            tokio::fs::write(&data_tmp, &payload).await?;
            data_tmp_exists = true;
        }
        if param_size > 0 {
            let payload = read_exact_n_chunked(reader, param_size as usize).await?;
            tokio::fs::write(&param_tmp, &payload).await?;
            param_tmp_exists = true;
        }
        if data_tmp_exists {
            tokio::fs::rename(&data_tmp, &data_final).await?;
            data_tmp_exists = false;
        }
        if param_tmp_exists {
            tokio::fs::rename(&param_tmp, &param_final).await?;
            param_tmp_exists = false;
        }
        Ok(())
    }
    .await;

    // Cleanup any remaining tempfile on the failure path.
    if data_tmp_exists {
        let _ = tokio::fs::remove_file(&data_tmp).await;
    }
    if param_tmp_exists {
        let _ = tokio::fs::remove_file(&param_tmp).await;
    }

    if let Err(e) = result {
        let resp = ServerError::Internal(format!("upload_atomic transfer: {e}")).into_response();
        return write_response(writer, &resp).await;
    }

    // Mint the upload id + write hashes (only the side that uploaded)
    let upload_id = upload::new_upload_id();
    if let Err(e) = upload::write_upload_id(&project_root, &upload_id) {
        let resp = ServerError::Internal(format!("write_upload_id: {e}")).into_response();
        return write_response(writer, &resp).await;
    }
    if data_size > 0 {
        if let Err(e) = upload::write_data_hash(&project_root, &data_hash) {
            log::warn!(target: "ppf::wire", "{upload_id}: write_data_hash: {e}");
        }
    }
    if param_size > 0 {
        if let Err(e) = upload::write_param_hash(&project_root, &param_hash) {
            log::warn!(target: "ppf::wire", "{upload_id}: write_param_hash: {e}");
        }
    }

    let has_data = data_final.exists();
    let has_param = param_final.exists();

    // Refresh project context so the engine knows about this upload.
    engine.set_project_context(name, project_root.to_str().unwrap_or(""));

    dispatch_with_executor(
        engine,
        executor,
        Event::UploadLanded {
            upload_id,
            data_hash,
            param_hash,
            has_data,
            has_param,
        },
    )
    .await;

    writer.write_all(b"OK\n").await?;
    Ok(())
}

/// Handle the `upload_notify` JSON request: the co-located addon
/// (`local` / `win_native`) has already written `data.pickle` /
/// `param.pickle` to the canonical project root on disk, so there is
/// no payload to stream. This handler does what the tail of
/// `handle_upload_atomic` does: stamp the addon-supplied `upload_id`
/// and hash files, then dispatch the same `UploadLanded` event so the
/// engine advances to `Data::Uploaded` and invalidates any stale
/// build, identical to the streamed path.
///
/// Unlike `upload_atomic`, the `upload_id` is minted by the addon and
/// carried on the wire; the server trusts it rather than generating
/// its own, so the addon can assert end-to-end that the direct-disk
/// path ran (the streamed path would surface a server-minted id).
pub(super) async fn handle_upload_notify<W>(
    writer: &mut W,
    engine: &ServerEngine,
    executor: &dyn EffectExecutor,
    req: &HashMap<String, Value>,
) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let name = req.get("name").and_then(Value::as_str).unwrap_or("");
    let upload_id = req
        .get("upload_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let data_hash = req
        .get("data_hash")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let param_hash = req
        .get("param_hash")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    if name.is_empty() {
        let resp = ServerError::BadRequest("Missing name".into()).into_response();
        return write_response(writer, &resp).await;
    }
    if upload_id.is_empty() {
        let resp =
            ServerError::BadRequest("upload_notify requires a non-empty upload_id".into())
                .into_response();
        return write_response(writer, &resp).await;
    }
    if engine.state().build == ppf_cts_core::state::Build::Building {
        let resp = ServerError::Conflict(
            "Cannot upload while a build is in progress. Abort the build first, then retry the upload.".into(),
        )
        .into_response();
        return write_response(writer, &resp).await;
    }

    // The addon wrote to the canonical root; resolve the same path the
    // streamed handler uses and confirm the pickles actually landed
    // before we advance engine state.
    let project_root = crate::upload::make_root(name, engine.config());
    let data_final = project_root.join(DATA_PICKLE);
    let param_final = project_root.join(PARAM_PICKLE);
    let has_data = data_final.exists();
    let has_param = param_final.exists();

    if !has_data && !has_param {
        let resp = ServerError::BadRequest(format!(
            "upload_notify: neither {DATA_PICKLE} nor {PARAM_PICKLE} found under {}",
            project_root.display()
        ))
        .into_response();
        return write_response(writer, &resp).await;
    }

    // Stamp the addon-supplied id + hashes next to the pickles so a
    // later disk reconcile (server restart, status round-trip) reads
    // back the same identity the streamed path would have written.
    if let Err(e) = upload::write_upload_id(&project_root, &upload_id) {
        let resp = ServerError::Internal(format!("write_upload_id: {e}")).into_response();
        return write_response(writer, &resp).await;
    }
    if has_data {
        if let Err(e) = upload::write_data_hash(&project_root, &data_hash) {
            log::warn!(target: "ppf::wire", "{upload_id}: write_data_hash: {e}");
        }
    }
    if has_param {
        if let Err(e) = upload::write_param_hash(&project_root, &param_hash) {
            log::warn!(target: "ppf::wire", "{upload_id}: write_param_hash: {e}");
        }
    }

    engine.set_project_context(name, project_root.to_str().unwrap_or(""));

    dispatch_with_executor(
        engine,
        executor,
        Event::UploadLanded {
            upload_id,
            data_hash,
            param_hash,
            has_data,
            has_param,
        },
    )
    .await;

    writer.write_all(b"OK\n").await?;
    Ok(())
}
