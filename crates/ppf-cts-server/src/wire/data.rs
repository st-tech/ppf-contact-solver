// File: crates/ppf-cts-server/src/wire/data.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Generic file transports + the BDAT stub.
//
// `data_send` and `data_receive` move files between the addon and the
// server, anchored under the per-project root (`make_root(name)`, the
// same canonical directory `upload_atomic` writes to). The client
// `path` is resolved against that root and any attempt to escape it
// (absolute path outside the root, or a `..` component) is rejected
// with `SandboxEscape`. They additionally refuse `data.pickle` /
// `param.pickle` so the only path for scene uploads is `upload_atomic`
// (atomic + hashed). BDAT is a binary-frame header kept for
// compatibility with older smoke tools: it drains bytes and acks
// `BINARY_OK`.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use ppf_cts_formats::files::{DATA_PICKLE, PARAM_PICKLE};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{json_u64, write_response, MAX_PAYLOAD_BYTES};
use crate::engine::ServerEngine;
use crate::error::ServerError;
use crate::protocol::read_exact_n_chunked;

pub(super) async fn handle_data_send<R, W>(
    reader: &mut R,
    writer: &mut W,
    engine: &ServerEngine,
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
    // Bound the client-declared size on the u64 before the `as usize`
    // cast so an absurd declaration can't drive a giant allocation in
    // read_exact_n_chunked.
    if size > MAX_PAYLOAD_BYTES {
        let resp = ServerError::BadRequest(format!(
            "size {size} exceeds max payload {MAX_PAYLOAD_BYTES}"
        ))
        .into_response();
        return write_response(writer, &resp).await;
    }
    let project_root = crate::upload::make_root(name, engine.config());
    let target = match resolve_under_root(&project_root, path) {
        Some(p) => p,
        None => {
            let resp = ServerError::SandboxEscape(path.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let payload = read_exact_n_chunked(reader, size as usize).await?;
    let tmp = PathBuf::from(format!("{}.tmp.{}", target.display(), crate::upload::temp_suffix()));
    crate::upload::stage_and_rename(&tmp, &target, &payload).await?;
    writer.write_all(b"OK\n").await?;
    Ok(())
}

pub(super) async fn handle_data_receive<W>(
    writer: &mut W,
    engine: &ServerEngine,
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
    let project_root = crate::upload::make_root(name, engine.config());
    let path = match resolve_under_root(&project_root, path) {
        Some(p) => p,
        None => {
            let resp = ServerError::SandboxEscape(path.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    let path = path.as_path();
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

/// Resolve a client-supplied `data_send` / `data_receive` path against
/// the per-project root and reject anything that escapes it.
///
/// The legitimate caller (`debug.data_send`) builds an absolute path of
/// the form `{remote_root}/dummy_data.pickle`, where `remote_root` is
/// exactly the server's `make_root(name)` directory mirrored back over a
/// status round-trip. We therefore accept an absolute path iff it lies
/// inside `root` after the same `..`-rejecting component walk used by
/// `notebook::resolve_sandbox`; a relative path is normalized under
/// `root` with that same walk. Returns `None` for any path that would
/// land outside `root` so the handler can answer `SandboxEscape`.
fn resolve_under_root(root: &Path, path: &str) -> Option<PathBuf> {
    let candidate = Path::new(path);
    let mut normalized = if candidate.is_absolute() {
        // An absolute request must already point inside the project
        // root. Reject any `..` (and any RootDir component beyond the
        // leading one) so the contained check can't be defeated by
        // climbing back out after entering the root prefix.
        if !candidate.starts_with(root) {
            return None;
        }
        PathBuf::new()
    } else {
        root.to_path_buf()
    };
    for comp in candidate.components() {
        match comp {
            Component::Prefix(_) | Component::RootDir => normalized.push(comp.as_os_str()),
            Component::Normal(s) => normalized.push(s),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() || !normalized.starts_with(root) {
                    return None;
                }
            }
        }
    }
    if !normalized.starts_with(root) {
        return None;
    }
    Some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_under_root_is_allowed() {
        let root = PathBuf::from("/data/proj");
        let p = resolve_under_root(&root, "/data/proj/dummy_data.pickle").unwrap();
        assert_eq!(p, PathBuf::from("/data/proj/dummy_data.pickle"));
    }

    #[test]
    fn relative_is_anchored_under_root() {
        let root = PathBuf::from("/data/proj");
        let p = resolve_under_root(&root, "sub/dummy_data.pickle").unwrap();
        assert_eq!(p, PathBuf::from("/data/proj/sub/dummy_data.pickle"));
    }

    #[test]
    fn absolute_outside_root_is_rejected() {
        let root = PathBuf::from("/data/proj");
        assert!(resolve_under_root(&root, "/etc/cron.d/x").is_none());
        assert!(resolve_under_root(&root, "/home/user/.ssh/authorized_keys").is_none());
    }

    #[test]
    fn parent_dir_escape_is_rejected() {
        let root = PathBuf::from("/data/proj");
        assert!(resolve_under_root(&root, "../escape").is_none());
        assert!(resolve_under_root(&root, "sub/../../escape").is_none());
        assert!(resolve_under_root(&root, "/data/proj/../escape").is_none());
    }
}
