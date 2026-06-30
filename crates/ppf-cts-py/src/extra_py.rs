// File: crates/ppf-cts-py/src/extra_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for `core::extra`. The Python frontend's `Extra`
// class re-exports these so notebooks calling `app.extra.*` keep
// working.

use ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::path::PathBuf;

use ppf_cts_core::extra;

/// Returns `(vertices(N,3) f64, faces(M,3) i32, (stitch_index(K,4)
/// i32, stitch_weight(K,4) f64))`. Same layout as the Python source.
#[pyfunction]
pub fn load_cipc_stitch_mesh<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let mesh = extra::load_cipc_stitch_mesh(path).map_err(|e| match e {
        extra::ExtraError::OpenFailed { .. } => PyFileNotFoundError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    })?;

    let v_arr = Array2::from_shape_vec((mesh.n_vertices, 3), mesh.vertices)
        .map_err(|e| PyRuntimeError::new_err(format!("vertex reshape failed: {e}")))?;
    let f_arr = Array2::from_shape_vec((mesh.n_faces, 3), mesh.faces)
        .map_err(|e| PyRuntimeError::new_err(format!("face reshape failed: {e}")))?;
    let s_idx_arr = Array2::from_shape_vec((mesh.n_stitch, 4), mesh.stitch_index)
        .map_err(|e| PyRuntimeError::new_err(format!("stitch index reshape failed: {e}")))?;
    let s_w_arr = Array2::from_shape_vec((mesh.n_stitch, 4), mesh.stitch_weight)
        .map_err(|e| PyRuntimeError::new_err(format!("stitch weight reshape failed: {e}")))?;

    let v_py = v_arr.into_pyarray(py).into_any();
    let f_py = f_arr.into_pyarray(py).into_any();
    let s_idx_py = s_idx_arr.into_pyarray(py).into_any();
    let s_w_py = s_w_arr.into_pyarray(py).into_any();

    let stitch_tuple = PyTuple::new(py, [s_idx_py, s_w_py])?.into_any();
    let outer = PyTuple::new(py, [v_py, f_py, stitch_tuple])?.into_any();
    Ok(outer)
}

/// Wrap `extra::sparse_clone`. Accepts a Python list of strings for
/// `paths`.
#[pyfunction]
#[pyo3(signature = (url, dest, paths, delete_exist=false))]
pub fn sparse_clone(
    url: &str,
    dest: &str,
    paths: &Bound<'_, PyList>,
    delete_exist: bool,
) -> PyResult<()> {
    let paths_owned: Vec<String> = paths
        .iter()
        .map(|p| p.extract::<String>())
        .collect::<PyResult<_>>()?;
    let paths_ref: Vec<&str> = paths_owned.iter().map(|s| s.as_str()).collect();
    let dest_path = PathBuf::from(dest);
    extra::sparse_clone(url, &dest_path, &paths_ref, delete_exist).map_err(|e| match e {
        extra::ExtraError::GitNotFound => PyFileNotFoundError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_cipc_stitch_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_clone, m)?)?;
    Ok(())
}
