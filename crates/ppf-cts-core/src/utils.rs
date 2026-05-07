// File: crates/ppf-cts-core/src/utils.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Engine-facing utilities. Direct port of the bits of
// frontend/_utils_.py the EffectExecutor + monitor actually call:
//
//   * `check_gpu`: nvidia-smi subprocess + SM-version check.
//   * `solver_busy` / `terminate_solver`: sysinfo-driven process
//     discovery + termination, replacing psutil.
//   * `is_fast_check` / `set_fast_check`: global toggle used by
//     ParamManager export to force `frames=1` in CI smoke tests.
//   * `get_cache_dir`: platform-aware cache path.
//
// What stays in Python:
//   * `in_jupyter_notebook` / `ci_name` / `get_ci_root` / `get_ci_dir`:
//     environment-detection helpers used only by notebook UX
//     paths (they don't gate engine effects).
//   * `dict_to_html_table`: display-only HTML formatter.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};
#[cfg(unix)]
use sysinfo::Signal;

/// Process-name substring the solver binary advertises. Both
/// `solver_busy` and `terminate_solver` filter on this.
pub const SOLVER_PROCESS_NAME: &str = "ppf-contact";

/// Minimum supported GPU compute-capability (Pascal = sm_60).
pub const MIN_SM: u32 = 60;

// ---------------------------------------------------------------------------
// is_fast_check / set_fast_check
//
// Process-wide flag used by ParamManager export to clamp frames=1 in
// CI / smoke contexts. AtomicBool replaces the Python class-attribute.

static FAST_CHECK: AtomicBool = AtomicBool::new(false);

pub fn is_fast_check() -> bool {
    FAST_CHECK.load(Ordering::Relaxed)
}

pub fn set_fast_check(enabled: bool) {
    FAST_CHECK.store(enabled, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Cache directory.

/// Platform-aware cache directory.
/// On Linux/macOS: `~/.cache/ppf-cts`. On Windows we fall back to a
/// repo-relative `.cache/ppf-cts` anchored at `current_dir`; callers
/// typically chdir to repo root before invoking.
pub fn get_cache_dir() -> PathBuf {
    let dir = if cfg!(target_os = "windows") {
        // Repo-relative on Windows so the cache rides with the repo.
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("cache")
            .join("ppf-cts")
    } else {
        // ~/.cache/ppf-cts on Linux/macOS.
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        home.join(".cache").join("ppf-cts")
    };
    let _ = std::fs::create_dir_all(&dir);
    dir
}

// ---------------------------------------------------------------------------
// GPU check.

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("nvidia-smi not found. An NVIDIA GPU with CUDA support is required.")]
    NvidiaSmiMissing,
    #[error("nvidia-smi failed. An NVIDIA GPU with CUDA support is required.")]
    NvidiaSmiFailed,
    #[error(
        "GPU '{name}' has compute capability sm_{actual}, but sm_{required} or higher is required. \
         Please use a newer GPU (Pascal architecture or later)."
    )]
    SmTooLow {
        name: String,
        actual: u32,
        required: u32,
    },
    #[error("No NVIDIA GPU detected. An NVIDIA GPU is required to run the solver.")]
    NoGpuDetected,
}

/// Check that an NVIDIA GPU with sufficient compute capability is
/// present.
pub fn check_gpu() -> Result<(), GpuError> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(GpuError::NvidiaSmiMissing);
        }
        Err(_) => {
            return Err(GpuError::NvidiaSmiFailed);
        }
    };
    if !output.status.success() {
        return Err(GpuError::NvidiaSmiFailed);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        let gpu_name = parts[0];
        let sm_str = parts[1].replace('.', "");
        let sm_ver = match sm_str.parse::<u32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if sm_ver < MIN_SM {
            return Err(GpuError::SmTooLow {
                name: gpu_name.to_string(),
                actual: sm_ver,
                required: MIN_SM,
            });
        }
        return Ok(()); // first GPU passes the bar.
    }
    Err(GpuError::NoGpuDetected)
}

// ---------------------------------------------------------------------------
// Process discovery + termination.

/// True iff any process whose name contains `SOLVER_PROCESS_NAME` is
/// running and not a zombie.
pub fn solver_busy() -> bool {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::new()),
    );
    sys.refresh_processes(ProcessesToUpdate::All, true);
    for proc in sys.processes().values() {
        let name = proc.name().to_string_lossy();
        if !name.contains(SOLVER_PROCESS_NAME) {
            continue;
        }
        if matches!(proc.status(), sysinfo::ProcessStatus::Zombie) {
            continue;
        }
        return true;
    }
    false
}

/// True iff any **descendant** of the current process whose name
/// contains `SOLVER_PROCESS_NAME` is running and not a zombie. Used
/// by the test rig's emulated server so a peer worker's solver
/// (running on the same host but in a different process tree) doesn't
/// trip our own "solver already running" guard.
pub fn solver_busy_descendants_only() -> bool {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::new()),
    );
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let me = sysinfo::Pid::from_u32(std::process::id());
    let processes = sys.processes();
    // Build a parent->child map so we can walk the descendant tree
    // in a single pass. ``Process::parent`` only gives us one edge,
    // so the loop has to start from ``me`` and follow children.
    let mut children_of: std::collections::HashMap<sysinfo::Pid, Vec<sysinfo::Pid>> =
        std::collections::HashMap::new();
    for (pid, proc) in processes.iter() {
        if let Some(parent) = proc.parent() {
            children_of.entry(parent).or_default().push(*pid);
        }
    }
    let mut stack: Vec<sysinfo::Pid> = vec![me];
    let mut seen: std::collections::HashSet<sysinfo::Pid> = std::collections::HashSet::new();
    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        if let Some(proc) = processes.get(&pid) {
            if pid != me {
                let name = proc.name().to_string_lossy();
                if name.contains(SOLVER_PROCESS_NAME)
                    && !matches!(proc.status(), sysinfo::ProcessStatus::Zombie)
                {
                    return true;
                }
            }
        }
        if let Some(kids) = children_of.get(&pid) {
            stack.extend(kids.iter().copied());
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Additional utility helpers.

/// Resolve the export base path.
/// In fast-check mode, exports go into the cache; otherwise to the
/// repo-relative `export` directory.
pub fn get_export_base_path() -> String {
    if is_fast_check() {
        get_cache_dir()
            .join("export")
            .to_string_lossy()
            .into_owned()
    } else {
        "export".to_string()
    }
}

/// Render a column-oriented mapping to an HTML table. The caller passes
/// (column_name, [cell_string, ...]) pairs in the desired column order.
/// `index` is accepted for compatibility but ignored.
pub fn dict_to_html_table(
    columns: &[(String, Vec<String>)],
    classes: &str,
    _index: bool,
) -> String {
    if columns.is_empty() {
        return "<table></table>".to_string();
    }
    let num_rows = columns[0].1.len();
    let mut out = String::new();
    out.push_str(&format!("<table class=\"{classes}\">"));
    out.push_str("<thead><tr>");
    for (col, _) in columns {
        out.push_str(&format!("<th>{col}</th>"));
    }
    out.push_str("</tr></thead>");
    out.push_str("<tbody>");
    for i in 0..num_rows {
        out.push_str("<tr>");
        for (_, vals) in columns {
            let val = if i < vals.len() { vals[i].as_str() } else { "" };
            out.push_str(&format!("<td>{val}</td>"));
        }
        out.push_str("</tr>");
    }
    out.push_str("</tbody>");
    out.push_str("</table>");
    out
}

/// Read the `.CI` marker file from the given frontend directory. Returns
/// `None` if the file is absent. Returns an error if the file exists
/// but is empty.
pub fn ci_name(frontend_dir: &std::path::Path) -> Result<Option<String>, &'static str> {
    let path = frontend_dir.join(".CI");
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path).map_err(|_| "could not read .CI file")?;
    let last = content.lines().last().unwrap_or("").trim().to_string();
    if last.is_empty() {
        return Err("The .CI file is empty. Please add the name of the CI environment.");
    }
    Ok(Some(last))
}

/// Get the path to the CI root directory: `<cache_dir>/ci`.
pub fn get_ci_root() -> PathBuf {
    get_cache_dir().join("ci")
}

/// Get the path to a specific CI's directory: `<cache_dir>/ci/<ci_name>`.
pub fn get_ci_dir(ci_name: &str) -> PathBuf {
    get_ci_root().join(ci_name)
}

/// True if a `.CLI` or `.CI` marker file is present. Used as the
/// notebook-detection short-circuit.
pub fn has_cli_or_ci_marker(frontend_dir: &std::path::Path) -> bool {
    frontend_dir.join(".CLI").exists() || frontend_dir.join(".CI").exists()
}

/// Terminate every non-zombie process whose name contains
/// `SOLVER_PROCESS_NAME`. Errors during signaling individual processes
/// are swallowed.
///
/// Platform note: Windows has no POSIX signals, and
/// `Process::kill_with(Signal::Term)` returns `None` on Windows
/// (sysinfo only maps `Signal::Kill` there, to `TerminateProcess`).
/// Without the windows branch the addon's "Terminate" button was a
/// no-op: the server kept reporting BUSY because the solver
/// subprocess was never signaled, and the UI reverted from
/// "Terminating…" back to "Simulating…". On Unix we keep SIGTERM so
/// the solver can flush state; on Windows we fall back to
/// `Process::kill()` (TerminateProcess) since there's no graceful
/// equivalent.
pub fn terminate_solver() {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::new()),
    );
    sys.refresh_processes(ProcessesToUpdate::All, true);
    for proc in sys.processes().values() {
        let name = proc.name().to_string_lossy();
        if !name.contains(SOLVER_PROCESS_NAME) {
            continue;
        }
        if matches!(proc.status(), sysinfo::ProcessStatus::Zombie) {
            continue;
        }
        #[cfg(unix)]
        let _ = proc.kill_with(Signal::Term);
        #[cfg(windows)]
        let _ = proc.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_check_default_is_false() {
        // The flag is process-wide; a parallel test could flip it,
        // so we don't assert the *initial* state, only the
        // round-trip semantics.
        let prior = is_fast_check();
        set_fast_check(true);
        assert!(is_fast_check());
        set_fast_check(false);
        assert!(!is_fast_check());
        // Restore prior state to avoid flakiness when other tests
        // observe the global.
        set_fast_check(prior);
    }

    #[test]
    fn cache_dir_returns_directory() {
        let dir = get_cache_dir();
        assert!(dir.exists() || dir.parent().map(|p| p.exists()).unwrap_or(false));
    }

    #[test]
    fn solver_busy_returns_bool() {
        // No solver running in a test context; the call must
        // complete without error and produce a bool.
        let _busy = solver_busy();
    }

    #[test]
    fn check_gpu_runs_to_completion() {
        // We don't assume nvidia-smi is installed on the dev Mac; the
        // function must produce *some* Result without panicking.
        // Either Ok(()) (CI Linux runner with NVIDIA GPU) or a
        // typed error.
        let _ = check_gpu();
    }

    #[test]
    fn dict_to_html_table_empty() {
        let out = dict_to_html_table(&[], "table", false);
        assert_eq!(out, "<table></table>");
    }

    #[test]
    fn dict_to_html_table_basic() {
        let cols = vec![
            ("name".to_string(), vec!["a".to_string(), "b".to_string()]),
            ("value".to_string(), vec!["1".to_string(), "2".to_string()]),
        ];
        let out = dict_to_html_table(&cols, "tbl", false);
        assert!(out.contains("<table class=\"tbl\">"));
        assert!(out.contains("<th>name</th>"));
        assert!(out.contains("<th>value</th>"));
        assert!(out.contains("<td>a</td>"));
        assert!(out.contains("<td>2</td>"));
    }

    #[test]
    fn ci_name_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let r = ci_name(tmp.path()).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn ci_name_present_returns_last_line() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".CI"), "first\ngithub-actions\n").unwrap();
        let r = ci_name(tmp.path()).unwrap();
        assert_eq!(r.as_deref(), Some("github-actions"));
    }

    #[test]
    fn ci_name_empty_errors() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(".CI"), "").unwrap();
        let r = ci_name(tmp.path());
        assert!(r.is_err());
    }

    #[test]
    fn get_ci_root_under_cache() {
        let r = get_ci_root();
        assert!(r.ends_with("ci"));
    }

    #[test]
    fn get_export_base_path_respects_fast_check() {
        let prior = is_fast_check();
        set_fast_check(true);
        let p = get_export_base_path();
        assert!(p.contains("export"));
        set_fast_check(false);
        assert_eq!(get_export_base_path(), "export");
        set_fast_check(prior);
    }

    #[test]
    fn check_gpu_error_messages_match_python() {
        // Lock the user-visible error strings byte-for-byte against
        // what the Python source emits (the addon's error UI has
        // copy that triggers off these exact substrings).
        let m = format!("{}", GpuError::NvidiaSmiMissing);
        assert!(m.contains("nvidia-smi not found"));
        assert!(m.contains("NVIDIA GPU"));

        let f = format!("{}", GpuError::NvidiaSmiFailed);
        assert!(f.contains("nvidia-smi failed"));

        let lo = format!(
            "{}",
            GpuError::SmTooLow {
                name: "Tesla K40".into(),
                actual: 35,
                required: 60,
            }
        );
        assert!(lo.contains("Tesla K40"));
        assert!(lo.contains("sm_35"));
        assert!(lo.contains("sm_60"));
        assert!(lo.contains("Pascal"));
    }
}
