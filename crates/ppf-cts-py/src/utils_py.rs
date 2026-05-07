// File: crates/ppf-cts-py/src/utils_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for `core::utils`. Exposes `check_gpu`, `solver_busy`,
// `terminate_solver`, `is_fast_check`, `set_fast_check`,
// `get_cache_dir`, `process_name` to the Blender addon and JupyterLab
// notebooks.

use std::path::PathBuf;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use ppf_cts_core::utils;

// ---------------------------------------------------------------------------
// Shared array-shape and index-decode helpers used by `decoder_py`,
// `kernels`, and `scene_py`. Centralizing keeps the dtype-acceptance
// rules and error-message format consistent across bindings.

/// Verify that `arr` is a 2-D ndarray of shape `(N, K)` and return `N`.
/// Emits `PyValueError` on shape mismatch.
pub(crate) fn require_n_by_k<T: numpy::Element, const K: usize>(
    arr: &PyReadonlyArray2<'_, T>,
    name: &str,
) -> PyResult<usize> {
    let s = arr.shape();
    if s.len() != 2 || s[1] != K {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {K}), got {s:?}"
        )));
    }
    Ok(s[0])
}

/// Output integer types accepted by [`read_index_array`]. Implemented
/// for `u32` and `i64`; conversion is the same `as`-cast each call site
/// used pre-refactor.
pub(crate) trait IndexOut: Copy {
    fn from_u32(v: u32) -> Self;
    fn from_i32(v: i32) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_u64(v: u64) -> Self;
}

impl IndexOut for u32 {
    #[inline]
    fn from_u32(v: u32) -> Self { v }
    #[inline]
    fn from_i32(v: i32) -> Self { v as u32 }
    #[inline]
    fn from_i64(v: i64) -> Self { v as u32 }
    #[inline]
    fn from_u64(v: u64) -> Self { v as u32 }
}

impl IndexOut for i64 {
    #[inline]
    fn from_u32(v: u32) -> Self { v as i64 }
    #[inline]
    fn from_i32(v: i32) -> Self { v as i64 }
    #[inline]
    fn from_i64(v: i64) -> Self { v }
    #[inline]
    fn from_u64(v: u64) -> Self { v as i64 }
}

/// Read an index ndarray of shape `(N, COLS)` and return a flat
/// `Vec<T>` of length `N * COLS`. Accepts dtypes `u32`, `i32`, `i64`,
/// `u64`. `name` is the label used in error messages, e.g. `"faces"`,
/// and is also used in the trailing dtype hint. `COLS == 1` is
/// allowed (use `read_index_array_1d_u32` for true 1-D inputs).
pub(crate) fn read_index_array<T: IndexOut, const COLS: usize>(
    arr: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<T>> {
    fn shape_err<const COLS: usize>(name: &str, s: &[usize]) -> PyErr {
        PyValueError::new_err(format!("{name} must be (N, {COLS}), got {s:?}"))
    }
    fn cont_err(name: &str) -> PyErr {
        PyTypeError::new_err(format!("{name} must be C-contiguous"))
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, u32>>() {
        let s = view.shape();
        if s.len() != 2 || s[1] != COLS {
            return Err(shape_err::<COLS>(name, s));
        }
        let slice = view.as_slice().map_err(|_| cont_err(name))?;
        return Ok(slice.iter().map(|&v| T::from_u32(v)).collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i32>>() {
        let s = view.shape();
        if s.len() != 2 || s[1] != COLS {
            return Err(shape_err::<COLS>(name, s));
        }
        let slice = view.as_slice().map_err(|_| cont_err(name))?;
        return Ok(slice.iter().map(|&v| T::from_i32(v)).collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i64>>() {
        let s = view.shape();
        if s.len() != 2 || s[1] != COLS {
            return Err(shape_err::<COLS>(name, s));
        }
        let slice = view.as_slice().map_err(|_| cont_err(name))?;
        return Ok(slice.iter().map(|&v| T::from_i64(v)).collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, u64>>() {
        let s = view.shape();
        if s.len() != 2 || s[1] != COLS {
            return Err(shape_err::<COLS>(name, s));
        }
        let slice = view.as_slice().map_err(|_| cont_err(name))?;
        return Ok(slice.iter().map(|&v| T::from_u64(v)).collect());
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be (N, {COLS}) ndarray of int32/int64/uint32/uint64",
    )))
}

/// Read a `(N, COLS)` index ndarray (any int dtype) as a `Vec<[T;
/// COLS]>`. Thin chunked wrapper over [`read_index_array`] for kernels
/// that consume `&[[T; COLS]]` directly.
pub(crate) fn read_index_array_chunked<T: IndexOut + Default, const COLS: usize>(
    arr: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<[T; COLS]>> {
    let flat = read_index_array::<T, COLS>(arr, name)?;
    let n = flat.len() / COLS;
    let mut out: Vec<[T; COLS]> = Vec::with_capacity(n);
    for chunk in flat.chunks_exact(COLS) {
        let mut row: [T; COLS] = [T::default(); COLS];
        for (i, v) in chunk.iter().enumerate() {
            row[i] = *v;
        }
        out.push(row);
    }
    Ok(out)
}

/// Read a 1-D index ndarray of any int dtype (u32 / i32 / i64 / u64)
/// and return a `Vec<u32>`. Requires C-contiguous input.
pub(crate) fn read_index_array_1d_u32(
    arr: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<u32>> {
    use numpy::PyReadonlyArray1;
    if let Ok(view) = arr.extract::<PyReadonlyArray1<'_, u32>>() {
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .to_vec());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray1<'_, i64>>() {
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .iter()
            .map(|&v| v as u32)
            .collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray1<'_, i32>>() {
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .iter()
            .map(|&v| v as u32)
            .collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray1<'_, u64>>() {
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .iter()
            .map(|&v| v as u32)
            .collect());
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a 1-D numpy array of int/uint dtype"
    )))
}

#[pyfunction]
pub fn check_gpu() -> PyResult<()> {
    utils::check_gpu().map_err(crate::errors::into_py_err)
}

#[pyfunction]
pub fn solver_busy() -> bool {
    utils::solver_busy()
}

#[pyfunction]
pub fn terminate_solver() {
    utils::terminate_solver();
}

#[pyfunction]
pub fn is_fast_check() -> bool {
    utils::is_fast_check()
}

#[pyfunction]
#[pyo3(signature = (enabled=true))]
pub fn set_fast_check(enabled: bool) {
    utils::set_fast_check(enabled);
}

#[pyfunction]
pub fn get_cache_dir() -> String {
    utils::get_cache_dir().to_string_lossy().into_owned()
}

#[pyfunction]
pub fn process_name() -> &'static str {
    utils::SOLVER_PROCESS_NAME
}

// ---------------------------------------------------------------------------
// Additional `_utils_` helpers.

/// Resolve the export base path, honoring fast-check mode.
#[pyfunction]
pub fn get_export_base_path() -> String {
    utils::get_export_base_path()
}

/// Render a column-oriented mapping to an HTML table. Mirrors
/// `_utils_.py:dict_to_html_table`. Cell values are passed pre-
/// stringified so the Rust side stays agnostic of Python types.
#[pyfunction]
#[pyo3(signature = (columns, classes = "table".to_string(), index = false))]
pub fn dict_to_html_table(
    columns: Vec<(String, Vec<String>)>,
    classes: String,
    index: bool,
) -> String {
    utils::dict_to_html_table(&columns, &classes, index)
}

/// Read the `.CI` marker file under the given frontend directory.
/// Returns `None` if absent. Raises `ValueError` if present but empty.
#[pyfunction]
pub fn ci_name(frontend_dir: &str) -> PyResult<Option<String>> {
    utils::ci_name(std::path::Path::new(frontend_dir))
        .map_err(|msg| PyValueError::new_err(msg.to_string()))
}

/// Path to the CI root directory: `<cache_dir>/ci`.
#[pyfunction]
pub fn get_ci_root() -> String {
    utils::get_ci_root().to_string_lossy().into_owned()
}

/// Path to a specific CI's directory: `<cache_dir>/ci/<ci_name>`.
#[pyfunction]
pub fn get_ci_dir(ci: &str) -> String {
    utils::get_ci_dir(ci).to_string_lossy().into_owned()
}

/// True if `.CLI` or `.CI` marker file is present in the given
/// frontend directory.
#[pyfunction]
pub fn has_cli_or_ci_marker(frontend_dir: &str) -> bool {
    utils::has_cli_or_ci_marker(std::path::Path::new(frontend_dir))
}

/// Create the directory at `path` if it doesn't exist (parents
/// included). Mirrors `os.makedirs(..., exist_ok=True)`.
#[pyfunction]
pub fn make_dir(path: &str) -> PyResult<()> {
    std::fs::create_dir_all(PathBuf::from(path))
        .map_err(|e| PyRuntimeError::new_err(format!("create_dir_all({path}) failed: {e}")))
}

/// Register the additional helpers. The callers in `lib.rs` already
/// register the original `Utils.*` shims by name; this adds the
/// remaining `_utils_` helpers.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_export_base_path, m)?)?;
    m.add_function(wrap_pyfunction!(dict_to_html_table, m)?)?;
    m.add_function(wrap_pyfunction!(ci_name, m)?)?;
    m.add_function(wrap_pyfunction!(get_ci_root, m)?)?;
    m.add_function(wrap_pyfunction!(get_ci_dir, m)?)?;
    m.add_function(wrap_pyfunction!(has_cli_or_ci_marker, m)?)?;
    m.add_function(wrap_pyfunction!(make_dir, m)?)?;
    Ok(())
}
