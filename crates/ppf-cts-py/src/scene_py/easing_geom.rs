// File: crates/ppf-cts-py/src/scene_py/easing_geom.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Easing curves and small geometric helpers (triangle areas,
// face-to-vertex weights, direction / cylinder color, all-vertices-pinned
// scan, per-face UV from two orthogonal directions).

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use ppf_cts_core::kernels::scene_build as sb;

use super::helpers::read_faces_u32;
use crate::errors::into_py_err;

#[pyfunction]
#[pyo3(signature = (t, handles))]
pub(super) fn scene_bezier_progress(
    t: f64,
    handles: ((f64, f64), (f64, f64)),
) -> f64 {
    let (h_r, h_l) = handles;
    sb::bezier_progress(t, ([h_r.0, h_r.1], [h_l.0, h_l.1]))
}

#[pyfunction]
#[pyo3(signature = (time, t_start, t_end, transition, bezier_handles=None))]
pub(super) fn scene_eased_progress(
    time: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    bezier_handles: Option<((f64, f64), (f64, f64))>,
) -> f64 {
    let h = bezier_handles.map(|(hr, hl)| ([hr.0, hr.1], [hl.0, hl.1]));
    sb::eased_progress(time, t_start, t_end, transition, h)
}

/// Per-triangle area `0.5 * |edge1 x edge2|`. Mirrors
/// `_compute_triangle_areas_vectorized`.
#[pyfunction]
#[pyo3(signature = (verts, tris))]
pub(super) fn scene_triangle_areas<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    tris: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let s = verts.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "verts must be (N, 3), got {s:?}"
        )));
    }
    let v = verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("verts must be C-contiguous"))?;
    let tris_owned = read_faces_u32(tris)?;
    let out = py.allow_threads(|| sb::triangle_areas(v, &tris_owned));
    Ok(out.into_pyarray(py))
}

/// Per-vertex `1.0 / (face_count + epsilon)`. Mirrors the numba pair
/// `compute_face_to_vertex_counts + compute_face_to_vertex_weights`.
#[pyfunction]
#[pyo3(signature = (n_verts, tris, epsilon=1e-4))]
pub(super) fn scene_face_to_vert_weights<'py>(
    py: Python<'py>,
    n_verts: usize,
    tris: &Bound<'py, PyAny>,
    epsilon: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let tris_owned = read_faces_u32(tris)?;
    let out = py.allow_threads(|| sb::face_to_vert_weights(n_verts, &tris_owned, epsilon));
    Ok(out.into_pyarray(py))
}

/// Average of a 1D area buffer, or 0 for empty. Mirrors
/// `FixedScene._average_tri_area`.
#[pyfunction]
#[pyo3(signature = (area))]
pub(super) fn scene_average_tri_area<'py>(
    area: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let a = area
        .as_slice()
        .map_err(|_| PyTypeError::new_err("area must be C-contiguous"))?;
    Ok(sb::average_tri_area(a))
}

#[pyfunction]
#[pyo3(signature = (verts, direction))]
pub(super) fn scene_direction_color<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    direction: [f64; 3],
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let s = verts.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "verts must be (N, 3), got {s:?}"
        )));
    }
    let v = verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("verts must be C-contiguous"))?;
    let n = s[0];
    let out = py.allow_threads(|| sb::direction_color(v, direction));
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (verts, center, direction, up))]
pub(super) fn scene_cylinder_color<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    center: [f64; 3],
    direction: [f64; 3],
    up: [f64; 3],
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let s = verts.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "verts must be (N, 3), got {s:?}"
        )));
    }
    let v = verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("verts must be C-contiguous"))?;
    let n = s[0];
    let out = py.allow_threads(|| sb::cylinder_color(v, center, direction, up));
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// `Object.update_static`: are all `[0, n_verts)` covered by some pin
/// list? Accepts a Python list of int sequences (one per pin holder).
///
/// Implemented in PyO3 here because the Python caller already owns
/// `list[int]` pin indices and converting them to ndarrays is wasteful
/// for this O(N) flag check.
#[pyfunction]
#[pyo3(signature = (n_verts, pin_indices))]
pub(super) fn scene_all_vertices_pinned(
    n_verts: usize,
    pin_indices: &Bound<'_, PyList>,
) -> PyResult<bool> {
    let mut owned: Vec<Vec<i64>> = Vec::with_capacity(pin_indices.len());
    for item in pin_indices.iter() {
        let v: Vec<i64> = item.extract()?;
        owned.push(v);
    }
    let refs: Vec<&[i64]> = owned.iter().map(|v| v.as_slice()).collect();
    Ok(sb::all_vertices_pinned(n_verts, &refs))
}

/// `Object.direction`: per-face UVs from two orthogonal directions.
#[pyfunction]
#[pyo3(signature = (verts, tris, ex, ey, eps=1e-3))]
pub(super) fn scene_uv_from_directions<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    tris: &Bound<'py, PyAny>,
    ex: [f64; 3],
    ey: [f64; 3],
    eps: f64,
) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
    let s = verts.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "verts must be (N, 3), got {s:?}"
        )));
    }
    let v = verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("verts must be C-contiguous"))?;
    let tris_owned = read_faces_u32(tris)?;
    let n_tri = tris_owned.len();
    let out = py
        .allow_threads(|| sb::uv_from_directions(v, &tris_owned, ex, ey, eps))
        .map_err(into_py_err)?;
    let arr = ndarray::Array3::from_shape_vec((n_tri, 3, 2), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}
