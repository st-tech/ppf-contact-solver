// File: crates/ppf-cts-server/src/wire/data.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Generic file transports + the BDAT stub.
//
// `data_send` and `data_receive` move arbitrary files between the
// addon and the server. They explicitly refuse `data.pickle` /
// `param.pickle` so the only path for scene uploads is
// `upload_atomic` (atomic + hashed). BDAT is a binary-frame header
// kept for compatibility with older smoke tools: it drains bytes
// and acks `BINARY_OK`.

use std::collections::HashMap;
use std::path::Path;

use ppf_cts_formats::files::{DATA_PICKLE, PARAM_PICKLE};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{json_u64, write_response};
use crate::error::ServerError;
use crate::protocol::read_exact_n_chunked;

pub(super) async fn handle_data_send<R, W>(
    reader: &mut R,
    writer: &mut W,
    req: &HashMap<String, Value>,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let path = req.get("path").and_then(Value::as_str).unwrap_or("");
    let name = req.get("name").and_then(Value::as_str).unwrap_or("");
    let size = json_u64(req.get("size"));
    if path.is_empty() || name.is_empty() {
        let resp = ServerError::BadRequest("Missing path or name".into()).into_response();
        return write_response(writer, &resp).await;
    }
    let basename = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if basename == DATA_PICKLE || basename == PARAM_PICKLE {
        let resp = ServerError::BadRequest(format!(
            "data_send no longer accepts {basename}; use upload_atomic for scene uploads."
        ))
        .into_response();
        return write_response(writer, &resp).await;
    }
    if size == 0 {
        let resp = ServerError::BadRequest("Invalid size".into()).into_response();
        return write_response(writer, &resp).await;
    }
    if let Some(parent) = Path::new(path).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let payload = read_exact_n_chunked(reader, size as usize).await?;
    let tmp = format!(
        "{path}.tmp.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    tokio::fs::write(&tmp, &payload).await?;
    if let Err(e) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    writer.write_all(b"OK\n").await?;
    Ok(())
}

pub(super) async fn handle_data_receive<W>(
    writer: &mut W,
    req: &HashMap<String, Value>,
) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let path = req.get("path").and_then(Value::as_str).unwrap_or("");
    let name = req.get("name").and_then(Value::as_str).unwrap_or("");
    if path.is_empty() || name.is_empty() {
        let resp = ServerError::BadRequest("Missing path or name".into()).into_response();
        return write_response(writer, &resp).await;
    }
    let path = Path::new(path);
    // `tokio::fs::metadata` returns NotFound rather than panicking,
    // and getting size first lets us emit the metadata header (a
    // single `{"size": N}` JSON line) before touching the file body.
    let meta = match tokio::fs::metadata(path).await {
        Ok(m) => m,
        Err(_) => {
            let resp = ServerError::NotFound("File not found".into()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    let size = meta.len();
    let header = serde_json::json!({"size": size});
    write_response(writer, &header).await?;

    // Stream the body in 32 KB chunks. Reading the whole file into a
    // `Vec<u8>` before writing OOM-trips the server on large
    // `app_state.pickle` / session artifacts (hundreds of MB).
    use tokio::io::AsyncReadExt as _;
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; 32 * 1024];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n]).await?;
    }
    Ok(())
}

pub(super) async fn handle_bdat<R, W>(
    reader: &mut R,
    writer: &mut W,
    peer: std::net::SocketAddr,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let mut total: u64 = 0;
    let mut buf = [0u8; 4096];
    loop {
        let n = match reader.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };
        total += n as u64;
    }
    log::debug!(target: "ppf::wire", "{peer}: BDAT received {total} bytes");
    writer.write_all(b"BINARY_OK").await?;
    Ok(())
}
