// File: crates/ppf-cts-server/src/upload.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Upload-id and content-hash file helpers. All writes are atomic:
// temp file in the same directory + `fs::rename` so concurrent
// readers never see a half-written upload-id. Empty hash strings
// delete the file rather than write blanks.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use ppf_cts_core::datamodel::compose_data_dir;
use ppf_cts_formats::files::{DATA_HASH_FILE, PARAM_HASH_FILE, UPLOAD_ID_FILE};

/// Resolve the on-disk project root for `name`. The server (not the
/// client) decides where uploads land. The addon's
/// `current_directory` is advisory for SSH-and-friends bootstrapping
/// (where `server/restart.sh` lives, where the venv is) but does NOT
/// pick the data location.
///
/// Resolution order:
///   1. `config.data_root` (set in tests via `EngineConfig`):
///      `<override>/<name>` (flat -- no git-<branch> segment).
///   2. `PPF_CTS_DATA_ROOT` env var (set by the test rig orchestrator):
///      `<env>/<name>` (flat -- no git-<branch> segment).
///   3. POSIX: `~/.local/share/ppf-cts/git-<branch>/<name>`.
///   4. Windows: `<repo_root>/local/share/ppf-cts/git-<branch>/<name>`.
///
/// Branch resolution mirrors `data_dirpath_for`: read
/// `<repo_root>/.git/branch_name.txt` first (release packaging stamps
/// it), fall back to `git branch --show-current`, then `"unknown"`.
pub fn make_root(name: &str, config: &crate::EngineConfig) -> PathBuf {
    let repo_root = repo_root_from_exe();
    let home_dir = std::env::var_os("HOME").map(PathBuf::from);

    // Override roots (config.data_root, PPF_CTS_DATA_ROOT) layout:
    // `<override>/<name>` -- flat. The test rig orchestrator allocates
    // a per-worker temp dir as PPF_CTS_DATA_ROOT and expects the upload
    // to land directly at `<root>/<name>` (see
    // blender_addon/debug/orchestrator.py). Production layout still
    // uses `<base>/git-<branch>/<name>` so multiple checkouts don't
    // collide on the canonical home-relative path.
    let data_dir = if let Some(over) = config.data_root.as_deref() {
        over.to_path_buf()
    } else if let Some(over) = std::env::var_os("PPF_CTS_DATA_ROOT") {
        let over = PathBuf::from(over);
        if !over.as_os_str().is_empty() {
            over
        } else {
            let branch = resolve_branch(&repo_root);
            compose_data_dir(&repo_root, home_dir.as_deref(), &branch)
        }
    } else {
        let branch = resolve_branch(&repo_root);
        compose_data_dir(&repo_root, home_dir.as_deref(), &branch)
    };
    data_dir.join(name)
}

/// Resolve the repo root from the running binary, or `None` when it
/// can't be recovered. The binary lives at
/// `<repo>/target/release/ppf-cts-server`, so walk three parents up
/// (file -> release -> target -> root). Returns `None` inside
/// `cargo test`, where the test binary lives somewhere unrelated to the
/// repo root. Callers keep their own fallback so the single definition
/// of the parent-walk depth is shared without changing each site's
/// fallback behavior.
pub(crate) fn repo_root_from_exe_opt() -> Option<PathBuf> {
    std::env::current_exe().ok().and_then(|exe| {
        exe.parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
    })
}

fn repo_root_from_exe() -> PathBuf {
    // Falls back to cwd when current_exe can't be resolved (e.g. inside
    // `cargo test`, where the test binary lives somewhere unrelated to
    // the repo root).
    repo_root_from_exe_opt()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn resolve_branch(repo_root: &Path) -> String {
    // Match `app::data_dirpath_for`: branch_name.txt first, then
    // `git branch --show-current`, then "unknown". Inlined because the
    // helpers are private to ppf_cts_core::datamodel::app, but the
    // logic is short enough to duplicate.
    let branch_file = repo_root.join(".git").join("branch_name.txt");
    if let Ok(raw) = fs::read_to_string(&branch_file) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if !repo_root.join(".git").is_dir() {
        return "unknown".to_string();
    }
    if let Ok(out) = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(repo_root)
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    "unknown".to_string()
}

/// Mint a 12-character hex upload id (`uuid.uuid4().hex[:12]`).
pub(crate) fn new_upload_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
}

fn upload_id_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(UPLOAD_ID_FILE)
}
fn data_hash_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(DATA_HASH_FILE)
}
fn param_hash_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(PARAM_HASH_FILE)
}

/// Per-write temp-file suffix: process id plus nanoseconds since the
/// UNIX epoch. Shared by the sync `atomic_write` here and the async
/// wire-layer stagers (`wire::data`, `wire::upload`) so all three agree
/// on one `tmp.{pid}.{nanos}` naming scheme. The suffix only needs to
/// avoid collisions between concurrent writers in the same directory;
/// the `unwrap_or(0)` on a clock that predates the epoch is harmless
/// because the pid still disambiguates and the rename is atomic.
pub(crate) fn temp_suffix() -> String {
    format!(
        "{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    )
}

fn atomic_write(path: &Path, content: &str) -> io::Result<()> {
    let tmp = path.with_extension(format!("tmp.{}", temp_suffix()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
    }
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Cleanup on rename failure.
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Async single-file atomic stage: write `bytes` to `tmp`, then rename
/// onto `final_`, removing `tmp` if the rename fails. Unlike the sync
/// `atomic_write` above this does NOT fsync: the wire payloads it backs
/// can be hundreds of MB and the cross-restart durability that the tiny
/// id/hash metadata needs is not worth the latency here. Callers own
/// the `tmp` name (build it with `temp_suffix`); this helper only
/// stages and commits the one file, so it is unsuitable for the
/// multi-file two-phase commit in `wire::upload`.
pub(crate) async fn stage_and_rename(tmp: &Path, final_: &Path, bytes: &[u8]) -> io::Result<()> {
    tokio::fs::write(tmp, bytes).await?;
    if let Err(e) = tokio::fs::rename(tmp, final_).await {
        let _ = tokio::fs::remove_file(tmp).await;
        return Err(e);
    }
    Ok(())
}

// `read_text_file` and the `read_*` helpers below are only exercised
// by the unit tests in this file: production code only writes these
// scratch files (the engine reads them via Python through other entry
// points). They stay gated on `cfg(test)` so the test roundtrips keep
// documenting the format without leaking dead-code warnings into
// release builds. `UploadError` itself is un-gated so the wire-layer
// `ServerError` can `From`-convert it.
fn read_text_file(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_default().trim().to_string()
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum UploadError {
    /// `read_upload_id` returns this when the file is missing or
    /// empty. The wire-layer reconcile path treats it as "no upload
    /// id yet"; only `data + param + missing id` is the invariant
    /// violation worth flagging (we log rather than panic).
    #[error("upload_id.txt is missing or empty in {0}")]
    MissingId(PathBuf),
    #[error("io: {0}")]
    Io(#[from] io::Error),
}

/// Read the persisted upload id. Returns `MissingId` if the file
/// doesn't exist or is empty (invariant violation: callers should
/// raise).
pub(crate) fn read_upload_id(root: impl AsRef<Path>) -> Result<String, UploadError> {
    let p = upload_id_path(&root);
    let s = read_text_file(&p);
    if s.is_empty() {
        return Err(UploadError::MissingId(p));
    }
    Ok(s)
}

/// Atomically write the upload id.
pub(crate) fn write_upload_id(root: impl AsRef<Path>, id: &str) -> io::Result<()> {
    atomic_write(&upload_id_path(&root), id)
}

/// Read the data hash. Returns empty string if the file is absent
/// (older projects without a hash file should not crash).
pub(crate) fn read_data_hash(root: impl AsRef<Path>) -> String {
    read_text_file(&data_hash_path(&root))
}

/// Atomically write the data hash. Empty string deletes the file.
pub(crate) fn write_data_hash(root: impl AsRef<Path>, hash: &str) -> io::Result<()> {
    write_hash(&data_hash_path(&root), hash)
}

/// Read the param hash. Returns empty string if absent.
pub(crate) fn read_param_hash(root: impl AsRef<Path>) -> String {
    read_text_file(&param_hash_path(&root))
}

/// Atomically write the param hash. Empty string deletes the file.
pub(crate) fn write_param_hash(root: impl AsRef<Path>, hash: &str) -> io::Result<()> {
    write_hash(&param_hash_path(&root), hash)
}

fn write_hash(path: &Path, hash: &str) -> io::Result<()> {
    if hash.is_empty() {
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    } else {
        atomic_write(path, hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_id_is_12_hex() {
        let id = new_upload_id();
        assert_eq!(id.len(), 12);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn upload_id_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        write_upload_id(dir.path(), "abc123").unwrap();
        assert_eq!(read_upload_id(dir.path()).unwrap(), "abc123");
    }

    #[test]
    fn read_upload_id_missing_errors() {
        let dir = tempfile::tempdir().unwrap();
        match read_upload_id(dir.path()) {
            Err(UploadError::MissingId(_)) => {}
            other => panic!("expected MissingId, got {other:?}"),
        }
    }

    #[test]
    fn data_hash_roundtrip_and_empty_deletes() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_data_hash(dir.path()), "");
        write_data_hash(dir.path(), "deadbeef").unwrap();
        assert_eq!(read_data_hash(dir.path()), "deadbeef");
        // Empty hash should delete the file.
        write_data_hash(dir.path(), "").unwrap();
        assert!(!dir.path().join("data_hash.txt").exists());
        assert_eq!(read_data_hash(dir.path()), "");
    }

    #[test]
    fn param_hash_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        write_param_hash(dir.path(), "feedface").unwrap();
        assert_eq!(read_param_hash(dir.path()), "feedface");
    }
}
