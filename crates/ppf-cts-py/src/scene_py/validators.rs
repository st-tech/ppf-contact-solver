// File: crates/ppf-cts-py/src/scene_py/validators.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Cold-tail validators / arithmetic helpers for `_scene_.py`. Each one
// is intentionally pure: the Python wrapper supplies the inputs, we
// raise the same exception class with a matching message, or return a
// scalar/tuple the Python source can drop into `self`. Also includes
// the small mesh-literal builders (line / box / tet_box / cone) and the
// `Object.report` / `FixedScene.report` formatting helpers.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ppf_cts_core::datamodel::elastic_model as elastic_model_core;
use ppf_cts_core::datamodel::mesh as mesh_core;
use ppf_cts_core::datamodel::scene as scene_helpers;
use ppf_cts_core::datamodel::session as session_core;
use ppf_cts_core::datamodel::validators as validators_core;

use crate::errors::into_py_err;

/// Validate a non-empty surface-map key. Raises ValueError matching
/// `Scene.set_surface_map`'s message.
#[pyfunction]
#[pyo3(signature = (name))]
pub(super) fn scene_validate_surface_map_key(name: &str) -> PyResult<()> {
    validators_core::validate_surface_map_key(name).map_err(into_py_err)
}

/// Validate a parameter key contains no underscore. Mirrors
/// `ParamManager.set` / `WallParam.set` / `SphereParam.set`.
#[pyfunction]
#[pyo3(signature = (key))]
pub(super) fn scene_validate_param_key_no_underscore(key: &str) -> PyResult<()> {
    validators_core::validate_param_key_no_underscore(key).map_err(into_py_err)
}

/// Decode `"x"`/`"y"`/`"z"` (case-insensitive) into 0/1/2.
#[pyfunction]
#[pyo3(signature = (axis))]
pub(super) fn scene_axis_letter_to_index(axis: &str) -> PyResult<usize> {
    scene_helpers::axis_letter_to_index(axis).map_err(PyValueError::new_err)
}

/// Map an elastic model name (`"arap"`, `"stvk"`, `"baraff-witkin"`,
/// `"snhk"`) to the u8 id the solver reads back. Authoritative table is
/// in `ppf_cts_core::datamodel::elastic_model`, so the Python exporter
/// and the solver share one encoding. Raises ValueError on an unknown
/// name, matching the old inline `assert` in `FixedScene` (`_scene_.py`).
#[pyfunction]
#[pyo3(signature = (name))]
pub(super) fn scene_model_name_to_id(name: &str) -> PyResult<u8> {
    elastic_model_core::model_name_to_id(name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown elastic model name: {name}")))
}

/// Reduce a per-object min-or-max stream into a scalar bound. `kind`
/// must be `"min"` or `"max"`. Empty `values` returns the appropriate
/// infinity. Mirrors `Scene.min` / `Scene.max`.
#[pyfunction]
#[pyo3(signature = (values, kind))]
pub(super) fn scene_reduce_axis_bound(values: Vec<f64>, kind: &str) -> PyResult<f64> {
    let mm = match kind {
        "min" => scene_helpers::MinMax::Min,
        "max" => scene_helpers::MinMax::Max,
        other => {
            return Err(PyValueError::new_err(format!(
                "kind must be 'min' or 'max', got {other}"
            )))
        }
    };
    Ok(scene_helpers::reduce_axis_bound(values, mm))
}

/// Raises ValueError when `t_end <= t_start`. Mirrors the guard in
/// `Object.move`, `Object.animate_rotate`, and several PinHolder
/// methods.
#[pyfunction]
#[pyo3(signature = (t_start, t_end))]
pub(super) fn scene_validate_time_window(t_start: f64, t_end: f64) -> PyResult<()> {
    validators_core::validate_time_window(t_start, t_end).map_err(into_py_err)
}

/// True iff the dynamic-color name is supported. Today this is just
/// `"area"`.
#[pyfunction]
#[pyo3(signature = (name))]
pub(super) fn scene_is_supported_dyn_color(name: &str) -> bool {
    scene_helpers::is_supported_dyn_color(name)
}

/// Wall.move_by arithmetic: previous position + delta. Returns the new
/// absolute position.
#[pyfunction]
#[pyo3(signature = (prev_position, delta))]
pub(super) fn scene_wall_move_by_position(
    prev_position: [f64; 3],
    delta: [f64; 3],
) -> [f64; 3] {
    scene_helpers::wall_move_by_position(prev_position, delta)
}

/// Sphere.move_by arithmetic: previous position + delta, radius reused.
#[pyfunction]
#[pyo3(signature = (prev_position, prev_radius, delta))]
pub(super) fn scene_sphere_move_by_entry(
    prev_position: [f64; 3],
    prev_radius: f64,
    delta: [f64; 3],
) -> ([f64; 3], f64) {
    scene_helpers::sphere_move_by_entry(prev_position, prev_radius, delta)
}

/// Strict-increasing keyframe-time check shared by `Wall._check_time`
/// and `Sphere._check_time`. Raises a `ValueError` whose message
/// includes the previous time.
#[pyfunction]
#[pyo3(signature = (prev_time, time))]
pub(super) fn scene_validate_collider_time(prev_time: f64, time: f64) -> PyResult<()> {
    validators_core::validate_collider_time(prev_time, time)
        .map_err(into_py_err)
}

/// `WallParam.set` / `SphereParam.set` "unknown parameter" guard.
/// Raises Exception (matching the Python source) with the same wording
/// when the key is not in `allowed`.
#[pyfunction]
#[pyo3(signature = (name, allowed))]
pub(super) fn scene_validate_known_param_name(name: &str, allowed: Vec<String>) -> PyResult<()> {
    let allowed_refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
    validators_core::validate_known_param_name(name, &allowed_refs)
        .map_err(into_py_err)
}

/// `Wall.add` / `Sphere.add` "already exists" guard.
#[pyfunction]
#[pyo3(signature = (already, kind))]
pub(super) fn scene_validate_collider_not_already_added(already: bool, kind: &str) -> PyResult<()> {
    validators_core::validate_collider_not_already_added(already, kind)
        .map_err(into_py_err)
}

/// `Object.rotate` axis validator. Returns the lowercased axis char
/// on success.
#[pyfunction]
#[pyo3(signature = (axis))]
pub(super) fn scene_validate_object_rotate_axis(axis: &str) -> PyResult<String> {
    validators_core::validate_object_rotate_axis(axis)
        .map(|c| c.to_string())
        .map_err(into_py_err)
}

/// `Object.stitch` chained validator: static -> Ind -> W. Raises with
/// the matching Python message on failure.
#[pyfunction]
#[pyo3(signature = (is_static, has_ind, has_w))]
pub(super) fn scene_validate_stitch_attach(
    is_static: bool,
    has_ind: bool,
    has_w: bool,
) -> PyResult<()> {
    validators_core::validate_stitch_attach(is_static, has_ind, has_w)
        .map_err(into_py_err)
}

/// `Object.set_uv` "must be tri" guard.
#[pyfunction]
#[pyo3(signature = (obj_type))]
pub(super) fn scene_validate_set_uv_obj_type(obj_type: &str) -> PyResult<()> {
    validators_core::validate_set_uv_obj_type(obj_type)
        .map_err(into_py_err)
}

/// `Object.normalize` "already normalized" guard.
#[pyfunction]
#[pyo3(signature = (already_normalized))]
pub(super) fn scene_validate_object_normalize(already_normalized: bool) -> PyResult<()> {
    validators_core::validate_object_normalize(already_normalized)
        .map_err(into_py_err)
}

/// `Object.velocity` "object is static" guard.
#[pyfunction]
#[pyo3(signature = (is_static))]
pub(super) fn scene_validate_object_not_static(is_static: bool) -> PyResult<()> {
    validators_core::validate_object_not_static(is_static)
        .map_err(into_py_err)
}

/// `Object.velocity` schedule classification. Returns
/// `(replace_initial, [u, v, w])` so the Python wrapper updates either
/// `self._velocity` (when the bool is True) or appends to
/// `self._velocity_schedule`.
#[pyfunction]
#[pyo3(signature = (u, v, w, t))]
pub(super) fn scene_classify_velocity_entry(u: f64, v: f64, w: f64, t: f64) -> (bool, [f64; 3]) {
    scene_helpers::classify_velocity_entry(u, v, w, t)
}

/// `Scene.select` lookup.
#[pyfunction]
#[pyo3(signature = (exists, name))]
pub(super) fn scene_validate_scene_select(exists: bool, name: &str) -> PyResult<()> {
    validators_core::validate_scene_select(exists, name)
        .map_err(into_py_err)
}

/// `MeshManager.line` arithmetic. Returns `(verts, edges)` numpy arrays.
#[pyfunction]
#[pyo3(signature = (p0, p1, n))]
pub(super) fn scene_line_mesh<'py>(
    py: Python<'py>,
    p0: [f64; 3],
    p1: [f64; 3],
    n: usize,
) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, numpy::PyArray2<u32>>)> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be > 0"));
    }
    let (verts, edges) = py.allow_threads(|| mesh_core::line_mesh(p0, p1, n));
    let v_arr = ndarray::Array2::from_shape_vec((n + 1, 3), verts)
        .map_err(|e| PyValueError::new_err(format!("vert reshape failed: {e}")))?;
    let e_arr = ndarray::Array2::from_shape_vec((n, 2), edges)
        .map_err(|e| PyValueError::new_err(format!("edge reshape failed: {e}")))?;
    Ok((v_arr.into_pyarray(py), e_arr.into_pyarray(py)))
}

/// `MeshManager.box` literal.
#[pyfunction]
#[pyo3(signature = (width, height, depth))]
pub(super) fn scene_box_mesh<'py>(
    py: Python<'py>,
    width: f64,
    height: f64,
    depth: f64,
) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, numpy::PyArray2<u32>>)> {
    let (verts, faces) = mesh_core::box_mesh(width, height, depth);
    let v_arr = ndarray::Array2::from_shape_vec((8, 3), verts)
        .map_err(|e| PyValueError::new_err(format!("vert reshape failed: {e}")))?;
    let f_arr = ndarray::Array2::from_shape_vec((12, 3), faces)
        .map_err(|e| PyValueError::new_err(format!("face reshape failed: {e}")))?;
    Ok((v_arr.into_pyarray(py), f_arr.into_pyarray(py)))
}

/// `MeshManager.tet_box` literal: returns (verts, faces, tets).
#[pyfunction]
#[pyo3(signature = (width, height, depth))]
pub(super) fn scene_tet_box_mesh<'py>(
    py: Python<'py>,
    width: f64,
    height: f64,
    depth: f64,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<u32>>,
    Bound<'py, numpy::PyArray2<u32>>,
)> {
    let (verts, faces, tets) = mesh_core::tet_box_mesh(width, height, depth);
    let v_arr = ndarray::Array2::from_shape_vec((8, 3), verts)
        .map_err(|e| PyValueError::new_err(format!("vert reshape failed: {e}")))?;
    let f_arr = ndarray::Array2::from_shape_vec((12, 3), faces)
        .map_err(|e| PyValueError::new_err(format!("face reshape failed: {e}")))?;
    let t_arr = ndarray::Array2::from_shape_vec((5, 4), tets)
        .map_err(|e| PyValueError::new_err(format!("tet reshape failed: {e}")))?;
    Ok((
        v_arr.into_pyarray(py),
        f_arr.into_pyarray(py),
        t_arr.into_pyarray(py),
    ))
}

/// `MeshManager.cone` generator.
#[pyfunction]
#[pyo3(signature = (nr, ny, nb, radius, height, sharpen))]
pub(super) fn scene_cone_mesh<'py>(
    py: Python<'py>,
    nr: usize,
    ny: usize,
    nb: usize,
    radius: f64,
    height: f64,
    sharpen: f64,
) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, numpy::PyArray2<u32>>)> {
    if nr < 3 || ny < 2 || nb < 2 {
        return Err(PyValueError::new_err(format!(
            "cone requires nr >= 3, ny >= 2, nb >= 2 (got nr={nr}, ny={ny}, nb={nb})"
        )));
    }
    let (verts, faces) = py
        .allow_threads(|| mesh_core::cone_mesh(nr, ny, nb, radius, height, sharpen));
    let n_v = verts.len() / 3;
    let n_f = faces.len() / 3;
    let v_arr = ndarray::Array2::from_shape_vec((n_v, 3), verts)
        .map_err(|e| PyValueError::new_err(format!("vert reshape failed: {e}")))?;
    let f_arr = ndarray::Array2::from_shape_vec((n_f, 3), faces)
        .map_err(|e| PyValueError::new_err(format!("face reshape failed: {e}")))?;
    Ok((v_arr.into_pyarray(py), f_arr.into_pyarray(py)))
}

/// `App.get_proj_root` path math.
#[pyfunction]
#[pyo3(signature = (frontend_file))]
pub(super) fn scene_project_root_from_frontend_file(frontend_file: &str) -> String {
    session_core::project_root_from_frontend_file(std::path::Path::new(frontend_file))
        .to_string_lossy()
        .into_owned()
}

/// `ParamManager.dyn` lookup. Mirrors the Python `if key not in
/// self._param.key_list(): raise ValueError(...)`.
#[pyfunction]
#[pyo3(signature = (exists, key))]
pub(super) fn scene_validate_param_key_exists(exists: bool, key: &str) -> PyResult<()> {
    validators_core::validate_param_key_exists(exists, key).map_err(into_py_err)
}

/// `ParamManager.time` strictly-increasing check.
#[pyfunction]
#[pyo3(signature = (prev, next_time))]
pub(super) fn scene_validate_param_time_strictly_increasing(prev: f64, next_time: f64) -> PyResult<()> {
    validators_core::validate_param_time_strictly_increasing(prev, next_time).map_err(into_py_err)
}

/// `SessionManager.select` lookup.
#[pyfunction]
#[pyo3(signature = (exists, name))]
pub(super) fn scene_validate_session_exists(exists: bool, name: &str) -> PyResult<()> {
    validators_core::validate_session_exists(exists, name).map_err(into_py_err)
}

/// FixedScene.report data formatter. Returns a list of `(label, str)`
/// pairs the caller can drop into `dict_to_html_table`.
#[pyfunction]
#[pyo3(signature = (
    n_vert,
    n_rod,
    n_tri,
    n_tet,
    n_pin,
    n_static_vert,
    n_static_tri,
    n_stitch_ind,
    n_stitch_w,
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn scene_fixed_scene_report_entries(
    n_vert: usize,
    n_rod: usize,
    n_tri: usize,
    n_tet: usize,
    n_pin: usize,
    n_static_vert: usize,
    n_static_tri: usize,
    n_stitch_ind: usize,
    n_stitch_w: usize,
) -> Vec<(String, String)> {
    scene_helpers::fixed_scene_report_entries(
        n_vert,
        n_rod,
        n_tri,
        n_tet,
        n_pin,
        n_static_vert,
        n_static_tri,
        n_stitch_ind,
        n_stitch_w,
    )
}
