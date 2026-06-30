// File: crates/ppf-cts-server/src/wire/notebook.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `notebook_send` / `notebook_delete` JSON sub-handlers, sandboxed
// under `<bin>/../examples`. The relative-path argument is normalized
// with a path-traversal guard (`resolve_sandbox`) that mirrors the
// Python `os.path.commonpath` check.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{json_u64, write_response, MAX_PAYLOAD_BYTES};
use crate::error::ServerError;
use crate::protocol::read_exact_n_chunked;

fn examples_root() -> PathBuf {
    // Resolve relative to the repo root: <repo_root>/examples. The
    // server binary lives at <repo_root>/target/release/ppf-cts-server,
    // so walk three parents up (file -> release -> target -> root) to
    // get there. JupyterLab is started from `<repo>/examples`;
    // resolving against `<bin>/../examples` instead would land
    // notebooks under `target/release/examples`, where JupyterLab
    // never sees them.
    //
    // `PPF_CTS_EXAMPLES_DIR` overrides the path so the test rig can
    // sandbox writes when the binary's parent chain is unrelated to
    // any examples tree (the orchestrator spawns workers under a temp
    // dir).
    if let Ok(over) = std::env::var("PPF_CTS_EXAMPLES_DIR") {
        if !over.is_empty() {
            return PathBuf::from(over);
        }
    }
    if let Some(root) = crate::upload::repo_root_from_exe_opt() {
        return root.join("examples");
    }
    PathBuf::from(".").join("examples")
}

pub(super) async fn handle_notebook_send<R, W>(
    reader: &mut R,
    writer: &mut W,
    req: &HashMap<String, Value>,
) -> std::io::Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let name = req.get("name").and_then(Value::as_str).unwrap_or("");
    let rel = req.get("relative_path").and_then(Value::as_str).unwrap_or("");
    let size = json_u64(req.get("size"));
    if name.is_empty() || rel.is_empty() {
        let resp =
            ServerError::BadRequest("Missing name or relative_path".into()).into_response();
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
    let target = match resolve_sandbox(&examples_root(), rel) {
        Some(p) => p,
        None => {
            let resp = ServerError::SandboxEscape(rel.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let payload = read_exact_n_chunked(reader, size as usize).await?;
    tokio::fs::write(&target, &payload).await?;
    writer.write_all(b"OK\n").await?;
    Ok(())
}

pub(super) async fn handle_notebook_delete<W>(
    writer: &mut W,
    req: &HashMap<String, Value>,
) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let name = req.get("name").and_then(Value::as_str).unwrap_or("");
    let rel = req.get("relative_path").and_then(Value::as_str).unwrap_or("");
    if name.is_empty() || rel.is_empty() {
        let resp =
            ServerError::BadRequest("Missing name or relative_path".into()).into_response();
        return write_response(writer, &resp).await;
    }
    let target = match resolve_sandbox(&examples_root(), rel) {
        Some(p) => p,
        None => {
            let resp = ServerError::SandboxEscape(rel.to_string()).into_response();
            return write_response(writer, &resp).await;
        }
    };
    if target.exists() {
        if let Err(e) = tokio::fs::remove_file(&target).await {
            let resp = ServerError::Internal(format!("Delete failed: {e}")).into_response();
            return write_response(writer, &resp).await;
        }
    }
    writer.write_all(b"OK\n").await?;
    Ok(())
}

/// Path-traversal guard: resolves `rel` under `root` and returns
/// `None` when the result escapes the sandbox.
pub(super) fn resolve_sandbox(root: &Path, rel: &str) -> Option<PathBuf> {
    let trimmed = rel.trim_start_matches(['/', '\\']);
    // Build a normalized form by walking components; we don't
    // require the path to exist (notebook_send creates it).
    let mut normalized = root.to_path_buf();
    for comp in std::path::Path::new(trimmed).components() {
        match comp {
            std::path::Component::Normal(s) => normalized.push(s),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() || !normalized.starts_with(root) {
                    return None;
                }
            }
            _ => return None, // RootDir / Prefix not allowed in `rel`
        }
    }
    // Sanity: the normalized path must be *under* root.
    if !normalized.starts_with(root) {
        return None;
    }
    Some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_strips_leading_slash() {
        let root = PathBuf::from("/tmp/examples");
        let p = resolve_sandbox(&root, "/foo/bar.ipynb").unwrap();
        assert_eq!(p, PathBuf::from("/tmp/examples/foo/bar.ipynb"));
    }

    #[test]
    fn sandbox_blocks_parent_escape() {
        let root = PathBuf::from("/tmp/examples");
        assert!(resolve_sandbox(&root, "../escape").is_none());
        assert!(resolve_sandbox(&root, "deep/../../escape").is_none());
    }
}
