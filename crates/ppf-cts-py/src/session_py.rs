// File: crates/ppf-cts-py/src/session_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for the session-side I/O helpers in
// `core::datamodel::session`. The frontend `Session` /
// `FixedSession` Python classes call into here for the polled hot
// paths (output-dir scans, log-file tails, vertex binary reads,
// solver-error pattern match).
//
// Notably absent: subprocess launch / monitor wiring. Solver
// lifecycle (DoLaunchSolver, DoSpawnBuild) lives in
// `crates/ppf-cts-server/src/executor.rs`, not here. This file only
// holds frontend-side filesystem + parsing primitives.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ppf_cts_core::datamodel::session as core;
use ppf_cts_core::parsers as parsers_core;

#[pyfunction]
#[pyo3(signature = (path, n_lines=None))]
pub fn read_log_tail(path: &str, n_lines: Option<usize>) -> Vec<String> {
    core::read_log_tail(std::path::Path::new(path), n_lines)
}

#[pyfunction]
pub fn latest_vertex_frame(output_dir: &str) -> u64 {
    core::latest_vertex_frame(std::path::Path::new(output_dir))
}

#[pyfunction]
pub fn list_saved_states(output_dir: &str) -> Vec<u64> {
    core::list_saved_states(std::path::Path::new(output_dir))
}

/// Read `vert_<frame>.bin` and return a `(N, 3)` float32 ndarray.
/// Returns None when the file is missing or its byte count isn't a
/// multiple of 12. Mirrors the inner branch of `SessionGet.vertex`.
#[pyfunction]
pub fn read_vertex_bin<'py>(
    py: Python<'py>,
    output_dir: &str,
    frame: u64,
) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
    // Disk I/O + parse don't touch Python state. Release the GIL so a
    // notebook polling for live frames doesn't block the rest of the
    // wheel from progress.
    let raw = match py.allow_threads(|| {
        core::read_vertex_bin(std::path::Path::new(output_dir), frame)
    }) {
        Some(v) => v,
        None => return Ok(None),
    };
    let n = raw.len() / 3;
    let arr = PyArray2::<f32>::zeros(py, (n, 3), false);
    {
        let mut view = unsafe { arr.as_array_mut() };
        for (i, chunk) in raw.chunks_exact(3).enumerate() {
            view[(i, 0)] = chunk[0];
            view[(i, 1)] = chunk[1];
            view[(i, 2)] = chunk[2];
        }
    }
    Ok(Some(arr))
}

/// Read the latest vertex frame and return `(ndarray, frame)` or None.
/// Mirrors the no-argument branch of `SessionGet.vertex`.
#[pyfunction]
pub fn read_latest_vertex<'py>(
    py: Python<'py>,
    output_dir: &str,
) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, u64)>> {
    let dir = std::path::Path::new(output_dir);
    let frames = core::list_vertex_frames(dir);
    let last = match frames.into_iter().max() {
        Some(n) => n,
        None => return Ok(None),
    };
    let arr = read_vertex_bin(py, output_dir, last)?;
    Ok(arr.map(|a| (a, last)))
}

/// Scan combined log + err lines for a known error pattern. Mirrors
/// `FixedSession._analyze_solver_error`. Returns None when nothing
/// matched.
#[pyfunction]
pub fn analyze_solver_error(
    log_lines: Vec<String>,
    err_lines: Vec<String>,
) -> Option<String> {
    core::analyze_solver_error(&log_lines, &err_lines)
}

#[pyfunction]
pub fn convert_time(ms: f64) -> String {
    core::convert_time(ms)
}

/// Build the "latest values" log summary as a Python dict. `max_sigma`
/// of None or <= 0 omits the `"stretch"` entry.
#[pyfunction]
#[pyo3(signature = (
    time_per_frame_ms,
    time_per_step_ms,
    num_contact,
    newton_steps,
    pcg_iter,
    max_sigma=None,
))]
pub fn log_summary<'py>(
    py: Python<'py>,
    time_per_frame_ms: f64,
    time_per_step_ms: f64,
    num_contact: f64,
    newton_steps: f64,
    pcg_iter: f64,
    max_sigma: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let pairs = core::format_log_summary(
        time_per_frame_ms,
        time_per_step_ms,
        num_contact,
        newton_steps,
        pcg_iter,
        max_sigma,
    );
    let d = pyo3::types::PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Build the "averaged" log summary as a Python dict. Each metric is
/// optional; omitted (`None`) values are skipped.
#[pyfunction]
#[pyo3(signature = (
    time_per_frame_ms_avg=None,
    time_per_step_ms_avg=None,
    num_contact_max=None,
    newton_steps_avg=None,
    pcg_iter_avg=None,
    max_sigma_avg=None,
    matrix_assembly_ms_avg=None,
    pcg_linsolve_ms_avg=None,
    toi_advanced_avg=None,
    dyn_consumed_max=None,
    line_search_ms_avg=None,
    toi_avg=None,
))]
pub fn log_average_summary<'py>(
    py: Python<'py>,
    time_per_frame_ms_avg: Option<f64>,
    time_per_step_ms_avg: Option<f64>,
    num_contact_max: Option<f64>,
    newton_steps_avg: Option<f64>,
    pcg_iter_avg: Option<f64>,
    max_sigma_avg: Option<f64>,
    matrix_assembly_ms_avg: Option<f64>,
    pcg_linsolve_ms_avg: Option<f64>,
    toi_advanced_avg: Option<f64>,
    dyn_consumed_max: Option<f64>,
    line_search_ms_avg: Option<f64>,
    toi_avg: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let pairs = core::format_log_average_summary(
        time_per_frame_ms_avg,
        time_per_step_ms_avg,
        num_contact_max,
        newton_steps_avg,
        pcg_iter_avg,
        max_sigma_avg,
        matrix_assembly_ms_avg,
        pcg_linsolve_ms_avg,
        toi_advanced_avg,
        dyn_consumed_max,
        line_search_ms_avg,
        toi_avg,
    );
    let d = pyo3::types::PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Read disk + format the average-summary dict in one call. `log_filenames`
/// is a list of `(name, filename)` pairs as discovered by the docstring
/// parser. Mirrors `read_average_summary_from_disk`.
#[pyfunction]
pub fn average_summary_from_disk<'py>(
    py: Python<'py>,
    data_dir: &str,
    log_filenames: Vec<(String, String)>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let pairs_ref: Vec<(&str, &str)> = log_filenames
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let out = core::average_summary_from_disk(std::path::Path::new(data_dir), &pairs_ref);
    let d = pyo3::types::PyDict::new(py);
    for (k, v) in out {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Read `<data_dir>/<filename>` log pairs and return the squashed
/// list-of-list mirroring `SessionLog.numbers`. Each `[x, y]` carries
/// Python `int` for components whose value is integer-valued; all
/// others stay `float`. Returns `None` when the file is missing.
#[pyfunction]
pub fn read_log_numbers_squashed<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<Option<Bound<'py, pyo3::types::PyList>>> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return Ok(None);
    }
    let pairs = core::read_log_numbers(p);
    let out = pyo3::types::PyList::empty(py);
    for (x, y) in pairs {
        let (x, y, xi, yi) = core::float_or_int_pair(x, y);
        let item = pyo3::types::PyList::empty(py);
        if xi {
            item.append(x as i64)?;
        } else {
            item.append(x)?;
        }
        if yi {
            item.append(y as i64)?;
        } else {
            item.append(y)?;
        }
        out.append(item)?;
    }
    Ok(Some(out))
}

/// Read `<session_path>/param_summary.txt` and return the newline-stripped
/// lines. Mirrors `SessionGet.param_summary`.
#[pyfunction]
pub fn param_summary_lines(session_path: &str) -> Vec<String> {
    core::param_summary_lines(std::path::Path::new(session_path))
}

/// Build the concatenated nvidia-smi text dump. Mirrors
/// `SessionGet.nvidia_smi`'s string construction (the caller `print()`s
/// the result).
#[pyfunction]
pub fn nvidia_smi_text(session_path: &str) -> String {
    core::nvidia_smi_text(std::path::Path::new(session_path))
}

/// Resolve the per-platform launcher script path under a session
/// directory; returns None when missing. Mirrors `SessionGet.command`.
/// `which`: "windows" or "unix".
#[pyfunction]
pub fn command_path(session_path: &str, which: &str) -> PyResult<Option<String>> {
    let plat = parse_platform(which)?;
    Ok(core::command_path(std::path::Path::new(session_path), plat)
        .map(|p| p.to_string_lossy().into_owned()))
}

/// Whether `<output_path>/<marker>` exists. Mirrors the body of
/// `FixedSession.finished` / `FixedSession.initialize_finished`.
#[pyfunction]
pub fn marker_exists(output_path: &str, marker: &str) -> bool {
    core::marker_exists(std::path::Path::new(output_path), marker)
}

/// `<session_path>/output/save_and_quit`. Mirrors
/// `FixedSession.save_and_quit_file_path`.
#[pyfunction]
pub fn save_and_quit_file_path(session_path: &str) -> String {
    core::save_and_quit_file_path(std::path::Path::new(session_path))
        .to_string_lossy()
        .into_owned()
}

/// Touch the `save_and_quit` sentinel under a session directory.
/// Mirrors `FixedSession.save_and_quit`.
#[pyfunction]
pub fn touch_save_and_quit(session_path: &str) -> PyResult<String> {
    core::touch_save_and_quit(std::path::Path::new(session_path))
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|e| PyValueError::new_err(format!("touch save_and_quit: {e}")))
}

/// Recursively delete the session directory. Mirrors `FixedSession.delete`.
/// No-op when missing.
#[pyfunction]
pub fn delete_session_dir(session_path: &str) -> PyResult<()> {
    core::delete_session_dir(std::path::Path::new(session_path))
        .map_err(|e| PyValueError::new_err(format!("delete session dir: {e}")))
}

/// Clear the session directory while preserving the solver `output/`
/// subtree (checkpoints + `save_and_quit` sentinel). Mirrors
/// `FixedSession.delete(preserve_output=True)`. No-op when missing.
#[pyfunction]
pub fn delete_session_dir_keep_output(session_path: &str) -> PyResult<()> {
    core::delete_session_dir_keep_output(std::path::Path::new(session_path))
        .map_err(|e| PyValueError::new_err(format!("delete session dir (keep output): {e}")))
}

/// Write `param.toml` (and `dyn_param.txt` when there are dynamic
/// overrides) under `path`. Mirrors `ParamManager.export`.
///
/// `param_holder` is the underlying `_ppf_cts_py.ParamHolder`. We
/// reach into it through its public python surface to build a
/// short-lived `core::ParamManager`. (No solver-side state is needed:
/// `ParamManager.export` only cares about the holder + dynamic
/// overrides.)
#[pyfunction]
#[pyo3(signature = (path, holder, dyn_param=None, fast_check=false))]
pub fn param_export_to_disk(
    path: &str,
    holder: &Bound<'_, pyo3::types::PyAny>,
    dyn_param: Option<&Bound<'_, pyo3::types::PyDict>>,
    fast_check: bool,
) -> PyResult<()> {
    use ppf_cts_core::datamodel::param_manager::ParamManager as CoreParam;

    // Pull `(key, value)` pairs out of the holder so we can repopulate
    // a transient ParamManager. We mutate from a fresh
    // ParamManager::new() because the holder's items may shadow the
    // app_param() defaults.
    let items_obj = holder.call_method0("items")?;
    let items: Vec<(String, Bound<'_, pyo3::types::PyAny>)> = items_obj.extract()?;
    let mut pm = CoreParam::new();
    for (k, v) in items {
        let pv = crate::param_py::pyany_to_param_value(&v)?;
        // Use bool-only set for the defaults that don't already exist
        // is unnecessary here: every key from `holder.items()` already
        // lives in the underlying app_param() default set.
        let _ = pm.set(&k, Some(pv));
    }

    if let Some(dyn_dict) = dyn_param {
        // The frontend Python ParamManager owns the dyn/time/change/hold
        // time cursor and already assembled each per-key list (starting
        // with the `(0.0, initial)` seed). We hand those lists straight
        // to `inject_dyn_entries`, which stores them for
        // `to_dyn_param_string` to serialize.
        for (key, vals) in dyn_dict.iter() {
            let key_s: String = key.extract()?;
            let entries: Vec<Bound<'_, pyo3::types::PyAny>> = vals.extract()?;
            let mut converted: Vec<(f64, ppf_cts_core::datamodel::params::ParamValue)> =
                Vec::with_capacity(entries.len());
            for entry_bound in entries {
                let tup: (f64, Bound<'_, pyo3::types::PyAny>) = entry_bound.extract()?;
                converted.push((tup.0, crate::param_py::pyany_to_param_value(&tup.1)?));
            }
            pm.inject_dyn_entries(&key_s, converted)
                .map_err(|e| PyValueError::new_err(format!("inject_dyn: {e}")))?;
        }
    }

    core::param_export_to_disk(std::path::Path::new(path), &pm, fast_check)
        .map_err(|e| PyValueError::new_err(format!("param export: {e}")))
}

// ---------------------------------------------------------------------------
// Cold-tail port helpers added for the second wave. Each one is a pure
// helper (no I/O) used by the Python side of `_session_.py`.

/// `SessionManager.create` autogenerated-name resolver. Given the names
/// of existing sessions and a base name, return the next free name and
/// the counter (0 = bare base name, no suffix). Mirrors the inline
/// `while name in self._sessions:` loop.
#[pyfunction]
#[pyo3(signature = (existing, base))]
pub fn session_autogenerate_name(existing: Vec<String>, base: &str) -> (String, u32) {
    core::autogenerate_session_name(&existing, base)
}

/// `Session.build` symlink-name resolver. App name on counter 0,
/// `app_name-{counter}` for further counters, `session_name` when not
/// autogenerated.
#[pyfunction]
#[pyo3(signature = (app_name, session_name, autogenerated=None))]
pub fn session_build_symlink_name(
    app_name: &str,
    session_name: &str,
    autogenerated: Option<u32>,
) -> String {
    core::build_symlink_name(app_name, session_name, autogenerated)
}

/// `SessionLog.numbers` path lookup. Returns the absolute path to the
/// SimpleLog `.out` file, or `None` when `name` is not in the log map.
#[pyfunction]
#[pyo3(signature = (info_path, name, log_filenames))]
pub fn session_log_filename_path(
    info_path: &str,
    name: &str,
    log_filenames: Vec<(String, String)>,
) -> Option<String> {
    let pairs: Vec<(&str, &str)> = log_filenames
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    core::log_filename_path(std::path::Path::new(info_path), name, &pairs)
        .map(|p| p.to_string_lossy().into_owned())
}

/// `SessionLog.stdout` / `.stderr` path picker. `stream` is "stdout"
/// or "stderr". Returns the absolute file path under `info_path`.
#[pyfunction]
#[pyo3(signature = (info_path, stream))]
pub fn session_log_tail_path(info_path: &str, stream: &str) -> PyResult<String> {
    let s = match stream {
        "stdout" => core::LogStream::Stdout,
        "stderr" => core::LogStream::Stderr,
        other => {
            return Err(PyValueError::new_err(format!(
                "stream must be 'stdout' or 'stderr', got {other}"
            )))
        }
    };
    Ok(core::log_tail_path(std::path::Path::new(info_path), s)
        .to_string_lossy()
        .into_owned())
}

/// `Session._save_fixed_session` layout helper. Returns the resolved
/// session directory and the path to the recoverable pickle. Symlink
/// creation/cleanup stays in Python because it touches `os.symlink` on
/// a per-platform basis.
#[pyfunction]
#[pyo3(signature = (app_root, name))]
pub fn session_fixed_dir_layout(app_root: &str, name: &str) -> (String, String) {
    let layout = core::fixed_session_dir_layout(app_root, name);
    (
        layout.session_dir.to_string_lossy().into_owned(),
        layout.recoverable_pickle.to_string_lossy().into_owned(),
    )
}

/// Compose `<data_dir>/symlinks/<name>` for `Session._save_fixed_session`.
#[pyfunction]
#[pyo3(signature = (data_dir, name))]
pub fn session_symlink_target_path(data_dir: &str, name: &str) -> String {
    core::symlink_target_path(data_dir, name)
        .to_string_lossy()
        .into_owned()
}

/// Walk `root` recursively for `.cu` / `.rs` files and return a
/// `name -> {Label: value, ..., Description: str, filename: str}`
/// dict harvested from logging docstrings. Wraps
/// `core::parsers::get_logging_docstrings`.
#[pyfunction]
pub fn get_logging_docstrings<'py>(
    py: Python<'py>,
    root: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let entries = parsers_core::get_logging_docstrings(root);
    let out = pyo3::types::PyDict::new(py);
    for (name, entry) in entries {
        let inner = pyo3::types::PyDict::new(py);
        for (label, value) in &entry.fields {
            inner.set_item(label, value)?;
        }
        if !entry.description.is_empty() {
            inner.set_item("Description", &entry.description)?;
        }
        inner.set_item("filename", &entry.filename)?;
        out.set_item(name, inner)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Frontend port: third-wave bindings. One pyfunction per branch the
// Python `_session_.py` was still computing in Python (path joins,
// validation guards, error-message templating, file persistence with
// platform-aware chmod). Each binding is one line of Rust dispatch.

fn parse_platform(which: &str) -> PyResult<core::Platform> {
    match which {
        "windows" | "win" | "Windows" => Ok(core::Platform::Windows),
        "unix" | "linux" | "macos" | "darwin" => Ok(core::Platform::Unix),
        other => Err(PyValueError::new_err(format!(
            "unknown platform tag: {other}"
        ))),
    }
}

/// Persist the solver-launcher script under `<session_path>/<filename>`,
/// chmod 0o755 on Unix, return the absolute path. Mirrors the body of
/// `SessionExport.shell_command` minus the `param.export(...)` call
/// (which the caller still drives via `param_export_to_disk`).
#[pyfunction]
#[pyo3(signature = (session_path, output_path, proj_root, which))]
pub fn write_shell_command_script(
    session_path: &str,
    output_path: &str,
    proj_root: &str,
    which: &str,
) -> PyResult<String> {
    let plat = parse_platform(which)?;
    core::write_shell_command_script(
        std::path::Path::new(session_path),
        std::path::Path::new(output_path),
        std::path::Path::new(proj_root),
        plat,
    )
    .map(|p| p.to_string_lossy().into_owned())
    .map_err(|e| PyValueError::new_err(format!("write shell command: {e}")))
}

/// Build the `subprocess.Popen` command body for a given launcher and
/// `--load` value. Mirrors the inline ternary in `FixedSession.start`.
#[pyfunction]
#[pyo3(signature = (cmd_path, load, which))]
pub fn solver_subprocess_command(
    cmd_path: &str,
    load: i64,
    which: &str,
) -> PyResult<String> {
    let plat = parse_platform(which)?;
    Ok(core::solver_subprocess_command(
        std::path::Path::new(cmd_path),
        load,
        plat,
    ))
}

/// `(stdout.log, error.log)` under a session info path. Mirrors
/// the repeated `os.path.join(self.info.path, ...)` pairs in
/// `FixedSession.start` / `stream`.
#[pyfunction]
#[pyo3(signature = (info_path))]
pub fn stdout_error_log_paths(info_path: &str) -> (String, String) {
    let (a, b) = core::stdout_error_log_paths(std::path::Path::new(info_path));
    (
        a.to_string_lossy().into_owned(),
        b.to_string_lossy().into_owned(),
    )
}

/// Validate the GPU driver version. Returns `None` when the version
/// passes; the user-facing error message string otherwise. Mirrors the
/// early validation block in `FixedSession.start`.
#[pyfunction]
#[pyo3(signature = (driver_version=None, minimum=520))]
pub fn validate_driver_version(
    driver_version: Option<u32>,
    minimum: u32,
) -> Option<String> {
    core::validate_driver_version(driver_version, minimum).err()
}

/// `os.path.join(get_export_base_path(), app_name, info_name)`
/// equivalent. Mirrors the `export_path` arithmetic used in
/// `SessionExport.animation` and `FixedSession.start`.
#[pyfunction]
#[pyo3(signature = (export_base, app_name, info_name))]
pub fn export_base_path_for(export_base: &str, app_name: &str, info_name: &str) -> String {
    core::export_base_path_for(std::path::Path::new(export_base), app_name, info_name)
        .to_string_lossy()
        .into_owned()
}

/// Resume-frame chooser for `FixedSession.resume`. `requested = -1`
/// means "pick the latest"; positive values are honored. Returns
/// `None` when no valid resume target exists (caller logs "no saved
/// state found").
#[pyfunction]
#[pyo3(signature = (saved, requested))]
pub fn select_resume_frame(saved: Vec<u64>, requested: i64) -> Option<u64> {
    core::select_resume_frame(&saved, requested)
}

/// `max(x for x in vals if isinstance(x, float))` with a `0.0`
/// fallback. Mirrors the strain-limit lookup at the tail of
/// `FixedSession.start` (the ``max_strain_limit + 1.0`` is added by
/// the caller because it's a single Python op).
#[pyfunction]
#[pyo3(signature = (vals))]
pub fn max_strain_limit_default_zero(vals: Vec<f64>) -> f64 {
    core::max_strain_limit_default_zero(&vals)
}

/// Tail of an existing log file as a single string (Python's
/// `"".join(lines[-n:]).strip()`). Used by `FixedSession.stream`'s
/// per-poll widget refresh.
#[pyfunction]
#[pyo3(signature = (path, n_lines))]
pub fn read_log_tail_joined(path: &str, n_lines: usize) -> String {
    core::read_log_tail_joined(std::path::Path::new(path), n_lines)
}

/// Pre-flight a "zip target path": existing zip is removed; final
/// `<dirpath>.zip` is returned. Mirrors `if os.path.exists(...):
/// os.remove(...)` at the top of `Zippable.zip`.
#[pyfunction]
#[pyo3(signature = (dirpath))]
pub fn prepare_zip_target(dirpath: &str) -> PyResult<String> {
    core::prepare_zip_target(std::path::Path::new(dirpath))
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|e| PyValueError::new_err(format!("prepare zip target: {e}")))
}

/// Format the user-facing violation message
/// `SessionManager.create` raises. Mirrors `f"Cannot create
/// session: {'; '.join(messages)}. "`.
#[pyfunction]
#[pyo3(signature = (messages))]
pub fn session_violations_message(messages: Vec<String>) -> String {
    core::session_violations_message(&messages)
}

/// `[line.rstrip("\n") for line in lines]` from `display_log`.
#[pyfunction]
#[pyo3(signature = (lines))]
pub fn rstrip_newlines(lines: Vec<String>) -> Vec<String> {
    core::rstrip_newlines(&lines)
}

/// Find a bundled `ffmpeg` binary under `project_root`. Mirrors the
/// `ffmpeg_candidates` list in `SessionExport.animation`. The Python
/// caller still does `shutil.which("ffmpeg")` when this returns
/// `None`, since `which` walks `$PATH` (a Python concern).
#[pyfunction]
#[pyo3(signature = (project_root))]
pub fn locate_bundled_ffmpeg(project_root: &str) -> Option<String> {
    core::locate_bundled_ffmpeg(std::path::Path::new(project_root))
        .map(|p| p.to_string_lossy().into_owned())
}

/// Build the ffmpeg command line `SessionExport.animation` runs via
/// `subprocess.run(..., shell=True)`. Mirrors the inline format
/// string.
#[pyfunction]
#[pyo3(signature = (ffmpeg_path, ext, vid_name))]
pub fn ffmpeg_video_command(ffmpeg_path: &str, ext: &str, vid_name: &str) -> String {
    core::ffmpeg_video_command(std::path::Path::new(ffmpeg_path), ext, vid_name)
}

/// `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
/// equivalent. Mirrors the project-root resolution in
/// `SessionExport.animation`.
#[pyfunction]
#[pyo3(signature = (frontend_file))]
pub fn project_root_from_frontend_file(frontend_file: &str) -> String {
    core::project_root_from_frontend_file(std::path::Path::new(frontend_file))
        .to_string_lossy()
        .into_owned()
}

/// Read a log file as a list of `f.readlines()`-style lines (each line
/// keeps its trailing `\n`). Missing file returns an empty list.
/// Mirrors the repeated `if os.path.exists(p): with open(p) as f:
/// f.readlines() else: []` block in `FixedSession.start`.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn read_lines_with_newlines(path: &str) -> Vec<String> {
    core::read_lines_with_newlines(std::path::Path::new(path))
}

/// `f"Solver failed: {''.join(err_lines[:5])}"` from `FixedSession.start`.
#[pyfunction]
#[pyo3(signature = (err_lines))]
pub fn solver_failed_short_message(err_lines: Vec<String>) -> String {
    core::solver_failed_short_message(&err_lines)
}

/// `f"Solver failed to start (rc={rc})"` from `FixedSession.start`.
#[pyfunction]
#[pyo3(signature = (rc=None))]
pub fn solver_failed_to_start_message(rc: Option<i32>) -> String {
    core::solver_failed_to_start_message(rc)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_logging_docstrings, m)?)?;
    m.add_function(wrap_pyfunction!(read_log_tail, m)?)?;
    m.add_function(wrap_pyfunction!(latest_vertex_frame, m)?)?;
    m.add_function(wrap_pyfunction!(list_saved_states, m)?)?;
    m.add_function(wrap_pyfunction!(read_vertex_bin, m)?)?;
    m.add_function(wrap_pyfunction!(read_latest_vertex, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_solver_error, m)?)?;
    m.add_function(wrap_pyfunction!(convert_time, m)?)?;
    m.add_function(wrap_pyfunction!(log_summary, m)?)?;
    m.add_function(wrap_pyfunction!(log_average_summary, m)?)?;
    m.add_function(wrap_pyfunction!(average_summary_from_disk, m)?)?;
    m.add_function(wrap_pyfunction!(read_log_numbers_squashed, m)?)?;
    m.add_function(wrap_pyfunction!(param_summary_lines, m)?)?;
    m.add_function(wrap_pyfunction!(nvidia_smi_text, m)?)?;
    m.add_function(wrap_pyfunction!(command_path, m)?)?;
    m.add_function(wrap_pyfunction!(marker_exists, m)?)?;
    m.add_function(wrap_pyfunction!(save_and_quit_file_path, m)?)?;
    m.add_function(wrap_pyfunction!(touch_save_and_quit, m)?)?;
    m.add_function(wrap_pyfunction!(delete_session_dir, m)?)?;
    m.add_function(wrap_pyfunction!(delete_session_dir_keep_output, m)?)?;
    m.add_function(wrap_pyfunction!(param_export_to_disk, m)?)?;
    m.add_function(wrap_pyfunction!(session_autogenerate_name, m)?)?;
    m.add_function(wrap_pyfunction!(session_build_symlink_name, m)?)?;
    m.add_function(wrap_pyfunction!(session_log_filename_path, m)?)?;
    m.add_function(wrap_pyfunction!(session_log_tail_path, m)?)?;
    m.add_function(wrap_pyfunction!(session_fixed_dir_layout, m)?)?;
    m.add_function(wrap_pyfunction!(session_symlink_target_path, m)?)?;
    m.add_function(wrap_pyfunction!(write_shell_command_script, m)?)?;
    m.add_function(wrap_pyfunction!(solver_subprocess_command, m)?)?;
    m.add_function(wrap_pyfunction!(stdout_error_log_paths, m)?)?;
    m.add_function(wrap_pyfunction!(validate_driver_version, m)?)?;
    m.add_function(wrap_pyfunction!(export_base_path_for, m)?)?;
    m.add_function(wrap_pyfunction!(select_resume_frame, m)?)?;
    m.add_function(wrap_pyfunction!(max_strain_limit_default_zero, m)?)?;
    m.add_function(wrap_pyfunction!(read_log_tail_joined, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_zip_target, m)?)?;
    m.add_function(wrap_pyfunction!(session_violations_message, m)?)?;
    m.add_function(wrap_pyfunction!(rstrip_newlines, m)?)?;
    m.add_function(wrap_pyfunction!(locate_bundled_ffmpeg, m)?)?;
    m.add_function(wrap_pyfunction!(ffmpeg_video_command, m)?)?;
    m.add_function(wrap_pyfunction!(project_root_from_frontend_file, m)?)?;
    m.add_function(wrap_pyfunction!(read_lines_with_newlines, m)?)?;
    m.add_function(wrap_pyfunction!(solver_failed_short_message, m)?)?;
    m.add_function(wrap_pyfunction!(solver_failed_to_start_message, m)?)?;
    Ok(())
}
