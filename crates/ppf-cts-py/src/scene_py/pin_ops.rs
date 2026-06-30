// File: crates/ppf-cts-py/src/scene_py/pin_ops.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pin-operation kernels: `MoveByOperation` / `MoveToOperation` /
// `SpinOperation` / `ScaleOperation` / `TransformKeyframeOperation` /
// `TransformAnimation` apply bodies. Inputs are `(N, 3)` f64 ndarrays
// and primitives; outputs are fresh `(N, 3)` ndarrays. The Python
// wrappers in `_scene_.py` are one-liners. Also includes the alias
// regrouping helper.

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ppf_cts_core::kernels::scene_build as sb;

fn read_n3_f64(arr: &PyReadonlyArray2<'_, f64>, label: &str) -> PyResult<(Vec<f64>, usize)> {
    // Wraps `utils_py::require_n_by_k` and slurps the buffer. Kept as a
    // 1-line shim because the kernels need the (Vec, N) tuple and we
    // don't want the helper to allocate when callers only need N.
    let n = crate::utils_py::require_n_by_k::<_, 3>(arr, label)?;
    let slice = arr
        .as_slice()
        .map_err(|_| PyTypeError::new_err(format!("{label} must be C-contiguous")))?;
    Ok((slice.to_vec(), n))
}

#[pyfunction]
#[pyo3(signature = (
    vertex, delta, time, t_start, t_end, transition, bezier_handles=None,
))]
pub(super) fn scene_move_by_apply<'py>(
    py: Python<'py>,
    vertex: PyReadonlyArray2<'py, f64>,
    delta: PyReadonlyArray2<'py, f64>,
    time: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    bezier_handles: Option<((f64, f64), (f64, f64))>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (v_buf, n) = read_n3_f64(&vertex, "vertex")?;
    let (d_buf, dn) = read_n3_f64(&delta, "delta")?;
    if dn != n {
        return Err(PyValueError::new_err(format!(
            "delta length {dn} mismatch with vertex {n}"
        )));
    }
    let h = bezier_handles.map(|(hr, hl)| ([hr.0, hr.1], [hl.0, hl.1]));
    let out = py.allow_threads(|| {
        sb::move_by_apply(&v_buf, &d_buf, time, t_start, t_end, transition, h)
    });
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    vertex, target, time, t_start, t_end, transition, bezier_handles=None,
))]
pub(super) fn scene_move_to_apply<'py>(
    py: Python<'py>,
    vertex: PyReadonlyArray2<'py, f64>,
    target: PyReadonlyArray2<'py, f64>,
    time: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    bezier_handles: Option<((f64, f64), (f64, f64))>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (v_buf, n) = read_n3_f64(&vertex, "vertex")?;
    let (t_buf, tn) = read_n3_f64(&target, "target")?;
    if tn != n {
        return Err(PyValueError::new_err(format!(
            "target length {tn} mismatch with vertex {n}"
        )));
    }
    let h = bezier_handles.map(|(hr, hl)| ([hr.0, hr.1], [hl.0, hl.1]));
    let out = py.allow_threads(|| {
        sb::move_to_apply(&v_buf, &t_buf, time, t_start, t_end, transition, h)
    });
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    vertex, center, axis, angular_velocity, t_start, t_end, time,
))]
pub(super) fn scene_spin_apply<'py>(
    py: Python<'py>,
    vertex: PyReadonlyArray2<'py, f64>,
    center: [f64; 3],
    axis: [f64; 3],
    angular_velocity: f64,
    t_start: f64,
    t_end: f64,
    time: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (v_buf, n) = read_n3_f64(&vertex, "vertex")?;
    let out = py.allow_threads(|| {
        sb::spin_apply(&v_buf, center, axis, angular_velocity, t_start, t_end, time)
    });
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    vertex, center, factor, t_start, t_end, transition, time,
))]
pub(super) fn scene_scale_apply<'py>(
    py: Python<'py>,
    vertex: PyReadonlyArray2<'py, f64>,
    center: [f64; 3],
    factor: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    time: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (v_buf, n) = read_n3_f64(&vertex, "vertex")?;
    let out = py.allow_threads(|| {
        sb::scale_apply(&v_buf, center, factor, t_start, t_end, transition, time)
    });
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// `TransformKeyframeOperation.apply`. `segments` is a list of dicts
/// each with `interpolation` ("LINEAR"/"BEZIER"/"CONSTANT") and for
/// "BEZIER" entries the `handle_right` and `handle_left` 2-vec floats.
/// Validation of unknown interpolation values raises ValueError, in
/// alignment with the Python source.
#[pyfunction]
#[pyo3(signature = (
    vertex, local_vert, times, translations, quaternions, scales,
    segments, rest_translation, time,
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn scene_transform_keyframe_apply<'py>(
    py: Python<'py>,
    vertex: PyReadonlyArray2<'py, f64>,
    local_vert: PyReadonlyArray2<'py, f64>,
    times: Vec<f64>,
    translations: Vec<[f64; 3]>,
    quaternions: Vec<[f64; 4]>,
    scales: Vec<[f64; 3]>,
    segments: &Bound<'py, PyList>,
    rest_translation: [f64; 3],
    time: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (v_buf, n) = read_n3_f64(&vertex, "vertex")?;
    let (lv_buf, _) = read_n3_f64(&local_vert, "local_vert")?;
    let n_times = times.len();
    if translations.len() != n_times
        || quaternions.len() != n_times
        || scales.len() != n_times
    {
        return Err(PyValueError::new_err(
            "translations/quaternions/scales length must equal times length",
        ));
    }
    let expected_segs = if n_times == 0 { 0 } else { n_times - 1 };
    if segments.len() != expected_segs {
        return Err(PyValueError::new_err(format!(
            "expected {} segments, got {}",
            expected_segs,
            segments.len()
        )));
    }
    let mut segs = Vec::with_capacity(expected_segs);
    for (i, item) in segments.iter().enumerate() {
        let d = item.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("segment {i} must be a dict"))
        })?;
        let interp: String = match d.get_item("interpolation")? {
            Some(v) => v.extract()?,
            None => "LINEAR".to_string(),
        };
        let seg = match interp.as_str() {
            "LINEAR" => sb::SegInterp::Linear,
            "CONSTANT" => sb::SegInterp::Constant,
            "BEZIER" => {
                let hr: [f64; 2] = match d.get_item("handle_right")? {
                    Some(v) => v.extract()?,
                    None => [1.0 / 3.0, 0.0],
                };
                let hl: [f64; 2] = match d.get_item("handle_left")? {
                    Some(v) => v.extract()?,
                    None => [2.0 / 3.0, 1.0],
                };
                sb::SegInterp::Bezier(hr, hl)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "transform_keyframes segment {i}: unsupported \
                     interpolation '{other}'. Supported: \
                     ['BEZIER', 'CONSTANT', 'LINEAR']"
                )));
            }
        };
        segs.push(seg);
    }
    let out = py.allow_threads(|| {
        sb::transform_keyframe_apply(
            &v_buf,
            &lv_buf,
            &times,
            &translations,
            &quaternions,
            &scales,
            &segs,
            rest_translation,
            time,
        )
    });
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// `TransformAnimation.evaluate`. Returns world-space vertex positions
/// at `time` with linear translation/scale and slerp on rotation.
#[pyfunction]
#[pyo3(signature = (
    local_vert, times, translations, quaternions, scales, time,
))]
pub(super) fn scene_transform_animation_evaluate<'py>(
    py: Python<'py>,
    local_vert: PyReadonlyArray2<'py, f64>,
    times: Vec<f64>,
    translations: Vec<[f64; 3]>,
    quaternions: Vec<[f64; 4]>,
    scales: Vec<[f64; 3]>,
    time: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (lv_buf, n) = read_n3_f64(&local_vert, "local_vert")?;
    let n_times = times.len();
    if translations.len() != n_times
        || quaternions.len() != n_times
        || scales.len() != n_times
    {
        return Err(PyValueError::new_err(
            "translations/quaternions/scales length must equal times length",
        ));
    }
    let out = py.allow_threads(|| {
        sb::transform_animation_evaluate(
            &lv_buf,
            &times,
            &translations,
            &quaternions,
            &scales,
            time,
        )
    });
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}
