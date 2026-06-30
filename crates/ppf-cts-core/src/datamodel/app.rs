// File: crates/ppf-cts-core/src/datamodel/app.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Direct port of the path / filesystem helpers used by the user-facing
// `App` class in `frontend/_app_.py`. The high-level `App` itself stays
// in Python because it owns Python-only manager objects (PlotManager,
// SessionManager, etc.) and serializes them via pickle through the
// CBOR envelope. We only port the side-effect-free or filesystem-only
// pieces that benefit from a single source of truth:
//
//   * `data_dirpath_for(base_dir)`: branch-name resolution + platform
//     branch on the data directory (`local/share/ppf-cts/git-{branch}`
//     on Windows; `~/.local/share/ppf-cts/git-{branch}` elsewhere).
//   * `default_cache_dir(base_dir)`: project-relative cache on Windows,
//     `~/.cache/ppf-cts` on Linux/macOS. Mirrors `App.__init__` defaults.
//   * `app_pickle_path(name, base_dir, ci_dir)`: where `App.save`
//     writes (and `App.__init__` reloads from).
//   * `recover_session_path(name, base_dir)`: resolves the symlink (or
//     `.txt` fallback on Windows) under `{data_dir}/symlinks/{name}`,
//     plus the `{data_dir}/{name}/session` fallback. Returns the path
//     to the `fixed_session.pickle` file the caller will then load and
//     unpickle in Python.
//   * `clear_cache_dir(cache_dir)`: equivalent of `App.clear_cache`'s
//     directory-walk-and-delete loop.
//
// What stays in Python:
//   * Constructing the manager graph (PlotManager / SessionManager /
//     ...). Those are Python class instances by construction.
//   * Pickle (de)serialization. We only return the path; the caller
//     reads bytes and calls `pickle.loads` (or the CBOR byte-sniff
//     wrapper around it).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Constant carried over from `frontend/_app_.py`. Used by both the
/// Python loader and any future Rust consumer that resolves a
/// recoverable session by path.
pub const RECOVERABLE_FIXED_SESSION_NAME: &str = "fixed_session.pickle";

/// Result of resolving a symlink-or-fallback path for a recoverable
/// session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoverablePath {
    /// Absolute path to the `fixed_session.pickle` file on disk.
    pub pickle_path: PathBuf,
    /// True iff the resolution went through the named location, i.e. the
    /// symlink (or its Windows `.txt` twin, which is a path-bearing text
    /// file rather than an actual symlink), False if the fallback
    /// `{data_dir}/{name}/session` directory was used. Pure diagnostic,
    /// read only by the `#[cfg(test)]` assertions below; the PyO3 wrapper
    /// `recover_session_path` returns only `pickle_path` and never
    /// surfaces this flag to Python. Branch error differentiation is done
    /// by the `AppPathError` variants (`NamedLocationMissing` vs
    /// `NoSessionWithName`), not by this flag.
    pub via_symlink: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum AppPathError {
    #[error("No recoverable fixed session found at named location: {name}")]
    NamedLocationMissing { name: String },
    #[error("No session found with name: {name}")]
    NoSessionWithName { name: String },
}

/// Read `{base_dir}/.git/branch_name.txt` if present and non-empty.
/// This is the file the addon writes during release packaging; in dev
/// the file does not exist and we fall through to `git branch
/// --show-current`. Returns the trimmed branch name on success.
fn read_branch_file(base_dir: &Path) -> Option<String> {
    let p = base_dir.join(".git").join("branch_name.txt");
    let raw = fs::read_to_string(&p).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `git branch --show-current` in `base_dir`, falling back to
/// `"unknown"` on either subprocess error or an empty branch name.
fn git_current_branch(base_dir: &Path) -> String {
    let out = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(base_dir)
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "unknown".to_string()
            } else {
                s
            }
        }
        _ => "unknown".to_string(),
    }
}

/// Compose the platform-aware data directory. `home_dir` is
/// `std::env::var_os("HOME")` (or `USERPROFILE` on Windows when
/// callers prefer); we accept it explicitly so unit tests stay
/// hermetic. The branch is resolved from `{base_dir}/.git/
/// branch_name.txt` first, then `git branch --show-current`, then
/// `"unknown"`.
pub fn data_dirpath_for(base_dir: &Path, home_dir: Option<&Path>) -> PathBuf {
    let branch = read_branch_file(base_dir).unwrap_or_else(|| git_current_branch(base_dir));
    compose_data_dir(base_dir, home_dir, &branch)
}

/// Build the platform-specific data dir from an already-resolved
/// branch name. Split out so tests can target the platform branch
/// without invoking git.
pub fn compose_data_dir(base_dir: &Path, home_dir: Option<&Path>, branch: &str) -> PathBuf {
    let segment = format!("git-{branch}");
    if cfg!(target_os = "windows") {
        base_dir
            .join("local")
            .join("share")
            .join("ppf-cts")
            .join(segment)
    } else {
        // The `/tmp` fallback is a test-only hermetic default; production
        // bindings always pass a resolved home (HOME, or the pwd entry) and
        // must never reach it.
        let home = home_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        home.join(".local").join("share").join("ppf-cts").join(segment)
    }
}

/// Default cache directory: project-relative on Windows,
/// `~/.cache/ppf-cts` elsewhere. Does not create the directory;
/// callers call `fs::create_dir_all` if they need it.
pub fn default_cache_dir(base_dir: &Path, home_dir: Option<&Path>) -> PathBuf {
    if cfg!(target_os = "windows") {
        base_dir.join("cache").join("ppf-cts")
    } else {
        // The `/tmp` fallback is a test-only hermetic default; production
        // bindings always pass a resolved home (HOME, or the pwd entry) and
        // must never reach it.
        let home = home_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        home.join(".cache").join("ppf-cts")
    }
}

/// Resolve `{data_dir}/{name}/app.pickle` (or `{ci_dir}/app.pickle`
/// when running in CI).
pub fn app_pickle_path(name: &str, data_dir: &Path, ci_dir: Option<&Path>) -> PathBuf {
    match ci_dir {
        Some(p) => p.join("app.pickle"),
        None => data_dir.join(name).join("app.pickle"),
    }
}

/// Compose the parent directory of an `App`'s frontend file. The
/// Python equivalents repeatedly call
/// `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`; this
/// returns the same `frontend_base_dir` we feed everywhere else.
pub fn frontend_base_dir_from_file(frontend_file: &Path) -> PathBuf {
    let abs = if frontend_file.is_absolute() {
        frontend_file.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(frontend_file))
            .unwrap_or_else(|_| frontend_file.to_path_buf())
    };
    abs.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Resolve a recoverable session pickle. Tries:
///   1. `{data_dir}/symlinks/{name}` as a symlink (Linux/macOS).
///   2. `{data_dir}/symlinks/{name}.txt` as a path-bearing text file
///      (Windows fallback).
///   3. `{data_dir}/{name}/session/fixed_session.pickle` (the
///      no-symlink default).
///
/// Returns `Ok(RecoverablePath)` only when the resolved
/// `fixed_session.pickle` file actually exists.
pub fn recover_session_path(name: &str, data_dir: &Path) -> Result<RecoverablePath, AppPathError> {
    let symlink_path = data_dir.join("symlinks").join(name);
    let mut session_dir: Option<PathBuf> = None;

    // 1. Symlink (POSIX). `read_link` returns the raw target; a relative
    // target is resolved against the symlink's own directory, matching how
    // the OS resolves the link itself.
    if let Ok(meta) = fs::symlink_metadata(&symlink_path) {
        if meta.file_type().is_symlink() {
            if let Ok(target) = fs::read_link(&symlink_path) {
                let resolved = if target.is_relative() {
                    symlink_path
                        .parent()
                        .map(|p| p.join(&target))
                        .unwrap_or(target)
                } else {
                    target
                };
                session_dir = Some(resolved);
            }
        }
    }

    // 2. `.txt` fallback (Windows). A relative path is resolved against the
    // `.txt` file's own directory, mirroring the symlink branch above.
    if session_dir.is_none() {
        let txt_path = symlink_path.with_extension("txt");
        if let Ok(raw) = fs::read_to_string(&txt_path) {
            let target = PathBuf::from(raw.trim().to_string());
            let resolved = if target.is_relative() {
                txt_path
                    .parent()
                    .map(|p| p.join(&target))
                    .unwrap_or(target)
            } else {
                target
            };
            session_dir = Some(resolved);
        }
    }

    if let Some(dir) = session_dir {
        let pickle_path = dir.join(RECOVERABLE_FIXED_SESSION_NAME);
        if pickle_path.exists() {
            return Ok(RecoverablePath {
                pickle_path,
                via_symlink: true,
            });
        }
        return Err(AppPathError::NamedLocationMissing {
            name: name.to_string(),
        });
    }

    // 3. Fallback: `{data_dir}/{name}/session/fixed_session.pickle`.
    let fallback = data_dir
        .join(name)
        .join("session")
        .join(RECOVERABLE_FIXED_SESSION_NAME);
    if fallback.exists() {
        return Ok(RecoverablePath {
            pickle_path: fallback,
            via_symlink: false,
        });
    }
    Err(AppPathError::NoSessionWithName {
        name: name.to_string(),
    })
}

/// Wipe everything under `cache_dir` (subdirs recursively, files
/// individually) without removing the directory itself. Errors on
/// individual entries are surfaced as `Err`.
pub fn clear_cache_dir(cache_dir: &Path) -> std::io::Result<()> {
    if !cache_dir.exists() || !cache_dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            fs::remove_dir_all(&path)?;
        } else {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn compose_data_dir_unix_layout() {
        if cfg!(target_os = "windows") {
            return;
        }
        let dir = compose_data_dir(
            Path::new("/repo"),
            Some(Path::new("/home/u")),
            "feature-x",
        );
        assert_eq!(
            dir,
            PathBuf::from("/home/u/.local/share/ppf-cts/git-feature-x")
        );
    }

    #[test]
    fn compose_data_dir_unknown_branch() {
        if cfg!(target_os = "windows") {
            return;
        }
        let dir =
            compose_data_dir(Path::new("/repo"), Some(Path::new("/home/u")), "unknown");
        assert!(dir.ends_with("git-unknown"));
    }

    #[test]
    fn default_cache_dir_unix_uses_home() {
        if cfg!(target_os = "windows") {
            return;
        }
        let dir = default_cache_dir(Path::new("/repo"), Some(Path::new("/home/u")));
        assert_eq!(dir, PathBuf::from("/home/u/.cache/ppf-cts"));
    }

    #[test]
    fn read_branch_file_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let git = tmp.path().join(".git");
        fs::create_dir_all(&git).unwrap();
        fs::write(git.join("branch_name.txt"), "release/2026-05\n").unwrap();
        assert_eq!(
            read_branch_file(tmp.path()),
            Some("release/2026-05".to_string())
        );
    }

    #[test]
    fn read_branch_file_empty_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let git = tmp.path().join(".git");
        fs::create_dir_all(&git).unwrap();
        fs::write(git.join("branch_name.txt"), "   \n").unwrap();
        assert!(read_branch_file(tmp.path()).is_none());
    }

    #[test]
    fn app_pickle_path_uses_name_when_no_ci() {
        let p = app_pickle_path("drape", Path::new("/data"), None);
        assert_eq!(p, PathBuf::from("/data/drape/app.pickle"));
    }

    #[test]
    fn app_pickle_path_uses_ci_dir_when_given() {
        let p = app_pickle_path(
            "drape",
            Path::new("/data"),
            Some(Path::new("/ci/workspace")),
        );
        assert_eq!(p, PathBuf::from("/ci/workspace/app.pickle"));
    }

    #[test]
    fn recover_no_session_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = recover_session_path("missing", tmp.path()).unwrap_err();
        assert!(matches!(err, AppPathError::NoSessionWithName { .. }));
    }

    #[test]
    fn recover_via_fallback_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let session_dir = tmp.path().join("hello").join("session");
        fs::create_dir_all(&session_dir).unwrap();
        let pickle = session_dir.join(RECOVERABLE_FIXED_SESSION_NAME);
        fs::write(&pickle, b"\x80fake").unwrap();

        let resolved = recover_session_path("hello", tmp.path()).unwrap();
        assert_eq!(resolved.pickle_path, pickle);
        assert!(!resolved.via_symlink);
    }

    #[test]
    fn recover_via_txt_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let actual_session = tmp.path().join("elsewhere");
        fs::create_dir_all(&actual_session).unwrap();
        fs::write(
            actual_session.join(RECOVERABLE_FIXED_SESSION_NAME),
            b"\x80data",
        )
        .unwrap();

        let symlink_dir = tmp.path().join("symlinks");
        fs::create_dir_all(&symlink_dir).unwrap();
        fs::write(
            symlink_dir.join("namedrun.txt"),
            actual_session.to_string_lossy().as_bytes(),
        )
        .unwrap();

        let resolved = recover_session_path("namedrun", tmp.path()).unwrap();
        assert!(resolved.pickle_path.ends_with(RECOVERABLE_FIXED_SESSION_NAME));
        assert!(resolved.via_symlink);
    }

    #[test]
    fn recover_named_location_missing_pickle() {
        let tmp = tempfile::tempdir().unwrap();
        let actual_session = tmp.path().join("elsewhere");
        fs::create_dir_all(&actual_session).unwrap();
        // Note: deliberately no fixed_session.pickle written.

        let symlink_dir = tmp.path().join("symlinks");
        fs::create_dir_all(&symlink_dir).unwrap();
        fs::write(
            symlink_dir.join("namedrun.txt"),
            actual_session.to_string_lossy().as_bytes(),
        )
        .unwrap();

        let err = recover_session_path("namedrun", tmp.path()).unwrap_err();
        assert!(matches!(err, AppPathError::NamedLocationMissing { .. }));
    }

    #[test]
    fn clear_cache_dir_handles_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope");
        assert!(clear_cache_dir(&missing).is_ok());
    }

    #[test]
    fn frontend_base_dir_from_file_strips_two_levels() {
        let p = frontend_base_dir_from_file(Path::new("/repo/frontend/_app_.py"));
        assert_eq!(p, PathBuf::from("/repo"));
    }

    #[test]
    fn clear_cache_dir_removes_files_and_subdirs() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        fs::create_dir_all(&cache).unwrap();
        fs::write(cache.join("a.bin"), b"abc").unwrap();
        let sub = cache.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("b.bin"), b"def").unwrap();

        clear_cache_dir(&cache).unwrap();

        assert!(cache.exists(), "cache dir itself stays put");
        let mut leftover: Vec<_> = fs::read_dir(&cache)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        leftover.sort();
        assert!(leftover.is_empty(), "cache should be empty: {:?}", leftover);
    }
}
