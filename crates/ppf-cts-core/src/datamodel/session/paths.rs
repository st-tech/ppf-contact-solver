// File: crates/ppf-cts-core/src/datamodel/session/paths.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Path arithmetic, sentinel files, and project-root helpers used by
// the Python `Session*` wrappers. Each function below replaces one
// branch / arithmetic block on the Python side.

use std::path::{Path, PathBuf};

use super::frames::list_saved_states;
use super::types::Platform;
use crate::datamodel::param_manager::ParamManager;

/// Read `param_summary.txt` under a session directory, return newline-
/// stripped lines. Empty when the file is missing.
pub fn param_summary_lines(session_path: &Path) -> Vec<String> {
    let p = session_path.join("param_summary.txt");
    let raw = match std::fs::read_to_string(&p) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    raw.lines().map(|l| l.to_string()).collect()
}

/// Format the `nvidia-smi` text dump the same way `SessionGet.nvidia_smi`
/// does: read both files and concatenate. Missing files surface as
/// "<file> not found" placeholders.
pub fn nvidia_smi_text(session_path: &Path) -> String {
    let dir = session_path.join("nvidia-smi");
    let main = dir.join("nvidia-smi.txt");
    let q = dir.join("nvidia-smi-q.txt");
    let mut out = String::new();
    match std::fs::read_to_string(&main) {
        Ok(body) => {
            out.push_str(&body);
            out.push('\n');
            out.push_str(&"=".repeat(80));
            out.push_str("\n\n");
        }
        Err(_) => {
            out.push_str("nvidia-smi.txt not found\n\n");
        }
    }
    match std::fs::read_to_string(&q) {
        Ok(body) => out.push_str(&body),
        Err(_) => out.push_str("nvidia-smi-q.txt not found\n"),
    }
    out
}

/// Produce the launcher-script path under a session directory if the
/// per-platform file exists.
pub fn command_path(session_path: &Path, platform: Platform) -> Option<PathBuf> {
    let p = session_path.join(platform.script_filename());
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// Convenience boolean for `<output>/<marker>` sentinels that
/// `FixedSession.finished` and `FixedSession.initialize_finished` poll.
/// `marker` is the full filename including extension (e.g. `finished.txt`,
/// `initialize_finish.txt`); no suffix is appended.
pub fn marker_exists(output_path: &Path, marker: &str) -> bool {
    output_path.join(marker).exists()
}

/// `save_and_quit` sentinel path.
pub fn save_and_quit_file_path(session_path: &Path) -> PathBuf {
    session_path.join("output").join("save_and_quit")
}

/// Touch the `save_and_quit` sentinel. Returns the path written.
pub fn touch_save_and_quit(session_path: &Path) -> std::io::Result<PathBuf> {
    let p = save_and_quit_file_path(session_path);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&p, b"")?;
    Ok(p)
}

/// Recursively delete the on-disk session directory if it exists.
pub fn delete_session_dir(session_path: &Path) -> std::io::Result<()> {
    if session_path.exists() {
        std::fs::remove_dir_all(session_path)?;
    }
    Ok(())
}

/// Clear every direct child of the session directory except the solver
/// `output/` subtree. This preserves the saved checkpoints
/// (`state_<N>.bin.gz`, `dataset.bin.gz`, `meshset.bin.gz`) and the
/// `save_and_quit` sentinel (all of which live under `output/`) while
/// removing the scene input so it can be re-decoded for a resume.
///
/// No-op when the directory is missing. Degenerates to a full clear
/// (the session directory itself is kept) when there is no `output/`
/// child yet.
pub fn delete_session_dir_keep_output(session_path: &Path) -> std::io::Result<()> {
    if !session_path.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(session_path)? {
        let entry = entry?;
        if entry.file_name() == "output" {
            continue;
        }
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(entry.path())?;
        } else {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// Project-root convenience for `save_and_quit_file_path`. Returns
/// true when the `save_and_quit` sentinel exists under
/// `<root>/session/output/`.
pub fn is_saving_in_progress(project_root: &Path) -> bool {
    let session_path = project_root.join("session");
    save_and_quit_file_path(&session_path).exists()
}

/// Project-root convenience for `list_saved_states`. Returns true
/// when the solver has written at least one `state_<N>.bin.gz`
/// checkpoint.
pub fn project_resumable(project_root: &Path) -> bool {
    let output_dir = project_root.join("session").join("output");
    !list_saved_states(&output_dir).is_empty()
}

/// Replace the `name == ""` branch in `SessionManager.create`. Given
/// the names of existing sessions and a base ("session"), find the
/// next free `<base>-<counter>` name. Returns the chosen name and the
/// counter (0 means "no suffix").
pub fn autogenerate_session_name(
    existing: &[String],
    base: &str,
) -> (String, u32) {
    let mut counter: u32 = 0;
    let mut name = base.to_string();
    let existing_set: std::collections::HashSet<&str> =
        existing.iter().map(String::as_str).collect();
    while existing_set.contains(name.as_str()) {
        counter += 1;
        name = format!("{base}-{counter}");
    }
    (name, counter)
}

/// Resolve the symlink name written under `data_dir/symlinks/...`.
/// When `autogenerated` is `None` the bare session name is used;
/// otherwise `app_name` is used for counter 0 and `app_name-{counter}`
/// for everything else.
pub fn build_symlink_name(
    app_name: &str,
    session_name: &str,
    autogenerated: Option<u32>,
) -> String {
    match autogenerated {
        None => session_name.to_string(),
        Some(0) => app_name.to_string(),
        Some(n) => format!("{app_name}-{n}"),
    }
}

/// Path layout used by `Session._save_fixed_session`. `session_dir`
/// is the per-session directory under the app root; `recoverable_pickle`
/// is where the CBOR-wrapped pickle gets written.
#[derive(Debug, Clone)]
pub struct FixedSessionDirLayout {
    pub session_dir: PathBuf,
    pub recoverable_pickle: PathBuf,
}

pub fn fixed_session_dir_layout(app_root: &str, name: &str) -> FixedSessionDirLayout {
    let session_dir = Path::new(app_root).join(name);
    let recoverable_pickle = session_dir.join(crate::datamodel::app::RECOVERABLE_FIXED_SESSION_NAME);
    FixedSessionDirLayout {
        session_dir,
        recoverable_pickle,
    }
}

/// `<data_dir>/symlinks/<name>` (no extension; the `.txt` fallback is
/// added by the caller).
pub fn symlink_target_path(data_dir: &str, name: &str) -> PathBuf {
    Path::new(data_dir).join("symlinks").join(name)
}

/// Write `param.toml` (and `dyn_param.txt` when there are dynamic
/// overrides) under `path`. The `fast_check` flag forces `frames=1`;
/// the caller must thread that bit through (we don't read globals
/// here).
pub fn param_export_to_disk(
    path: &Path,
    param: &ParamManager,
    fast_check: bool,
) -> Result<(), crate::datamodel::param_manager::ParamManagerError> {
    let toml_body = param.to_toml_string(fast_check);
    if !toml_body.is_empty() {
        std::fs::create_dir_all(path)?;
        std::fs::write(path.join("param.toml"), toml_body)?;
    }
    let dyn_body = param.to_dyn_param_string()?;
    if !dyn_body.is_empty() {
        std::fs::create_dir_all(path)?;
        std::fs::write(path.join("dyn_param.txt"), dyn_body)?;
    }
    Ok(())
}

/// `(stdout.log, error.log)` under a session info path.
pub fn stdout_error_log_paths(info_path: &Path) -> (PathBuf, PathBuf) {
    (info_path.join("stdout.log"), info_path.join("error.log"))
}

/// Validate the GPU driver version. Returns the user-facing error
/// message when the version is missing or below the cutoff; `Ok(())`
/// otherwise. The caller is expected to raise the message as
/// `ValueError`.
pub fn validate_driver_version(driver_version: Option<u32>, minimum: u32) -> Result<(), String> {
    match driver_version {
        Some(v) if v < minimum => Err(format!(
            "Driver version is {v}. It must be {minimum} or newer"
        )),
        Some(_) => Ok(()),
        None => Err("Driver version could not be detected.".to_string()),
    }
}

/// Resolve the directory under the export base where animation frames
/// are written: `<export_base>/<app_name>/<info_name>`.
pub fn export_base_path_for(export_base: &Path, app_name: &str, info_name: &str) -> PathBuf {
    export_base.join(app_name).join(info_name)
}

/// `max(x for x in vals if isinstance(x, float))` semantics from the
/// `start()` strain-limit lookup. None inputs are dropped; if no
/// values survive the result is `default_zero` which the Python source
/// falls back to (initial `max_strain_limit = 0.0`).
pub fn max_strain_limit_default_zero(vals: &[f64]) -> f64 {
    vals.iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max)
}

/// Pre-flight a "zip target path": existing zip is removed; final
/// `<dirpath>.zip` is returned.
pub fn zip_target_path(dirpath: &Path) -> PathBuf {
    let mut s = dirpath.as_os_str().to_os_string();
    s.push(".zip");
    PathBuf::from(s)
}

/// Remove an existing zip output if present. Returns the path to be
/// written next.
pub fn prepare_zip_target(dirpath: &Path) -> std::io::Result<PathBuf> {
    let target = zip_target_path(dirpath);
    if target.exists() {
        std::fs::remove_file(&target)?;
    }
    Ok(target)
}

/// Project-root resolution Mirror for the
/// `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
/// idiom in `SessionExport.animation`. Delegates to
/// `app::frontend_base_dir_from_file` so the absolutization and
/// fallback policy stay identical across the crate.
pub fn project_root_from_frontend_file(frontend_file: &Path) -> PathBuf {
    // `<root>/frontend/_session_.py` -> `<root>` (two parents).
    crate::datamodel::app::frontend_base_dir_from_file(frontend_file)
}
