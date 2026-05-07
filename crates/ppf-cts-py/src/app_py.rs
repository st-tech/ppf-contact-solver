// File: crates/ppf-cts-py/src/app_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for `core::datamodel::app`. The frontend `App` class in
// `frontend/_app_.py` dispatches its filesystem-only helpers (data
// directory resolution, default cache dir, app-pickle path, cache wipe)
// through here.
//
// What we do *not* port: constructing the manager graph (PlotManager,
// SessionManager, AssetManager, SceneManager, MeshManager); those are
// Python class instances assembled in `App.__init__`. Pickle envelope
// (de)serialization also stays in Python because pickle is a Python-
// specific format and the byte-sniff CBOR adapter just landed.

use std::path::{Path, PathBuf};

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use ppf_cts_core::datamodel::app as core_app;

use crate::errors::into_py_err;

fn home_path_buf() -> Option<PathBuf> {
    // On Linux/macOS the Python source uses `os.path.expanduser('~')`
    // which reads `HOME`. On Windows the cache + data branches both
    // use a base_dir-relative path, so HOME is not consulted.
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Mirror `App.get_data_dirpath`. Reads `{base_dir}/.git/branch_name.txt`
/// first, then `git branch --show-current`, then `"unknown"`.
#[pyfunction]
#[pyo3(signature = (base_dir))]
pub fn get_data_dirpath(base_dir: &str) -> String {
    let home = home_path_buf();
    core_app::data_dirpath_for(Path::new(base_dir), home.as_deref())
        .to_string_lossy()
        .into_owned()
}

/// Resolve a default cache directory when the caller passes an empty
/// `cache_dir`. Does *not* create the directory; the Python caller
/// runs `os.makedirs` afterwards.
#[pyfunction]
#[pyo3(signature = (base_dir))]
pub fn default_cache_dir(base_dir: &str) -> String {
    let home = home_path_buf();
    core_app::default_cache_dir(Path::new(base_dir), home.as_deref())
        .to_string_lossy()
        .into_owned()
}

/// Compose `{data_dir}/{name}/app.pickle`, or `{ci_dir}/app.pickle`
/// when `ci_dir` is provided. Mirrors the assignment to `self._path`.
#[pyfunction]
#[pyo3(signature = (name, data_dir, ci_dir=None))]
pub fn app_pickle_path(name: &str, data_dir: &str, ci_dir: Option<&str>) -> String {
    let ci = ci_dir.map(Path::new);
    core_app::app_pickle_path(name, Path::new(data_dir), ci)
        .to_string_lossy()
        .into_owned()
}

/// Resolve the `fixed_session.pickle` path for a recoverable session.
/// Returns the absolute path string. Raises:
///   * generic `Exception` with the same message Python uses on
///     "named-location missing" or "no session found" so existing
///     callers asserting on the message keep working.
#[pyfunction]
#[pyo3(signature = (name, data_dir))]
pub fn recover_session_path(name: &str, data_dir: &str) -> PyResult<String> {
    let rp = core_app::recover_session_path(name, Path::new(data_dir)).map_err(into_py_err)?;
    Ok(rp.pickle_path.to_string_lossy().into_owned())
}

/// Wipe the contents of `cache_dir` (subdirs recursively, files
/// individually). Mirrors `App.clear_cache`. Errors are surfaced as
/// `OSError` so the Python caller's `try` block can still catch them.
#[pyfunction]
#[pyo3(signature = (cache_dir))]
pub fn clear_cache_dir(cache_dir: &str) -> PyResult<()> {
    core_app::clear_cache_dir(Path::new(cache_dir))
        .map_err(|e| PyOSError::new_err(e.to_string()))
}

/// Build the native CBOR map payload that `App.save` writes.
/// Mirrors `App._to_cbor_dict`: top-level inspectable metadata
/// (`name`, `root`, `asset_names`) plus the opaque `pickle_blob`.
/// Pickle bytes are produced by the Python caller and forwarded
/// here without copy.
#[pyfunction]
#[pyo3(signature = (name, root, asset_names, pickle_blob))]
pub fn app_to_cbor_dict<'py>(
    py: Python<'py>,
    name: &str,
    root: &str,
    asset_names: &Bound<'_, PyList>,
    pickle_blob: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", name)?;
    d.set_item("root", root)?;
    d.set_item("asset_names", asset_names)?;
    d.set_item("pickle_blob", pickle_blob)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_data_dirpath, m)?)?;
    m.add_function(wrap_pyfunction!(default_cache_dir, m)?)?;
    m.add_function(wrap_pyfunction!(app_pickle_path, m)?)?;
    m.add_function(wrap_pyfunction!(recover_session_path, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache_dir, m)?)?;
    m.add_function(wrap_pyfunction!(app_to_cbor_dict, m)?)?;
    Ok(())
}
