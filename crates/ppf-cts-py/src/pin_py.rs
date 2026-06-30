// File: crates/ppf-cts-py/src/pin_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for `core::datamodel::pin`. The Python frontend's
// `PinHolder` (frontend/_scene_.py:1110) routes its builder calls
// through this class.
//
// Design: this is a *parallel validator*, not a full lockstep mirror.
// The Python `PinHolder` keeps owning a Python `PinData` dataclass
// (because the rest of the codebase, including `_decoder_.py` and the
// scene builder, reads and mutates that dataclass directly, and it is
// the sole source for the FixedScene export). The Rust holder runs the
// same argument validation and stores its own `PinData` for the fields
// it carries: `index`, `transition`, `unpin_time`, `pull_strength`,
// `pin_group_id`, and `operations`. The Python-only fields that the
// Rust `PinData` has no counterpart for (`pin_stiffness`,
// `pull_weights`, `rest_shape_track`) are validated and stored on the
// Python side alone, so they are NOT mirrored here. The parity test exercises the public surface and
// confirms both stores agree on the mirrored field set after a builder
// chain.
//
// The wire-format distinction:
//   * `PinData.transition` is a string (`"linear"`, `"smooth"`,
//     `"bezier"`). Mirrors Python field-for-field.
//   * `MoveBy.delta` is either a uniform `Vec3` or a per-vertex
//     `(N, 3)` array (`MoveByDelta` enum). Python's
//     `PinHolder.move_by` reshapes both inputs into `(N, 3)`, so on the
//     Rust side we always store `MoveByDelta::PerVertex` after the
//     reshape; the `as_uniform` helper is available for callers that
//     want the compact form.

use ndarray::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use ppf_cts_core::datamodel::pin::{
    CenterMode, InterpMode, KeyframeSegment, MoveByDelta, PinData, PinOperation,
};
use ppf_cts_core::datamodel::quat::{Quat, Vec3};

/// Allowed transition strings. Mirrors Python's `interp` setter, which
/// silently accepts any string but only `"linear"`, `"smooth"`, and
/// `"bezier"` produce non-linear easing in `_eased_progress`.
fn validate_transition(transition: &str) -> PyResult<()> {
    match transition {
        "linear" | "smooth" | "bezier" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "transition must be one of 'linear', 'smooth', 'bezier' (got '{other}')"
        ))),
    }
}

/// Coerce a Python `bezier_handles` argument into the Rust schema.
/// Accepts `None`, or a 2-tuple of 2-tuples of floats:
/// `((p1x, p1y), (p2x, p2y))`. Mirrors `_bezier_progress`'s handle
/// unpacking.
fn extract_bezier_handles(
    handles: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<[[f64; 2]; 2]>> {
    let Some(h) = handles else { return Ok(None) };
    if h.is_none() {
        return Ok(None);
    }
    let pair: ((f64, f64), (f64, f64)) = h.extract().map_err(|_| {
        PyValueError::new_err(
            "bezier_handles must be ((p1x, p1y), (p2x, p2y)) or None",
        )
    })?;
    Ok(Some([[pair.0 .0, pair.0 .1], [pair.1 .0, pair.1 .1]]))
}

fn extract_vec3(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec3> {
    if let Ok(v) = arr.extract::<Vec<f64>>() {
        if v.len() == 3 {
            return Ok([v[0], v[1], v[2]]);
        }
    }
    if let Ok(a) = arr.extract::<PyReadonlyArray1<f64>>() {
        let s = a.as_slice().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a contiguous length-3 array"))
        })?;
        if s.len() == 3 {
            return Ok([s[0], s[1], s[2]]);
        }
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a length-3 sequence",
    )))
}

/// Coerce a `delta_pos` argument to `MoveByDelta`. Replicates Python's
/// `np.array(delta_pos).reshape((-1, 3))` plus the broadcast/match
/// guard against the pin's vertex count.
fn coerce_move_by_delta(
    delta: &Bound<'_, PyAny>,
    n_pin_verts: usize,
) -> PyResult<MoveByDelta> {
    // Try a single 3-vec first. Python tiles a single delta to (N, 3),
    // but the math is identical to a uniform offset, so keep the compact
    // Uniform form regardless of n_pin_verts.
    if let Ok(v) = extract_vec3(delta, "delta_pos") {
        return Ok(MoveByDelta::Uniform(v));
    }
    // Per-vertex (N, 3) array.
    if let Ok(arr2) = delta.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr2.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyException::new_err(
                "delta_pos must reshape into (N, 3)",
            ));
        }
        let nrows = shape[0];
        if nrows == 1 {
            let s = arr2.as_array();
            return Ok(MoveByDelta::Uniform([s[[0, 0]], s[[0, 1]], s[[0, 2]]]));
        }
        if nrows != n_pin_verts {
            return Err(PyException::new_err(
                "delta_pos must have the same length as pin",
            ));
        }
        let owned = arr2.as_array().to_owned();
        return Ok(MoveByDelta::PerVertex(owned));
    }
    // Fallback: a python list-of-lists.
    if let Ok(list) = delta.downcast::<PyList>() {
        let mut rows: Vec<[f64; 3]> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let row: Vec<f64> = item.extract().map_err(|_| {
                PyException::new_err("delta_pos rows must be length-3 sequences")
            })?;
            if row.len() != 3 {
                return Err(PyException::new_err(
                    "delta_pos rows must be length-3 sequences",
                ));
            }
            rows.push([row[0], row[1], row[2]]);
        }
        if rows.is_empty() {
            return Err(PyException::new_err("delta_pos cannot be empty"));
        }
        if rows.len() == 1 {
            return Ok(MoveByDelta::Uniform(rows[0]));
        }
        if rows.len() != n_pin_verts {
            return Err(PyException::new_err(
                "delta_pos must have the same length as pin",
            ));
        }
        let mut arr = Array2::<f64>::zeros((rows.len(), 3));
        for (i, r) in rows.iter().enumerate() {
            arr[[i, 0]] = r[0];
            arr[[i, 1]] = r[1];
            arr[[i, 2]] = r[2];
        }
        return Ok(MoveByDelta::PerVertex(arr));
    }
    Err(PyException::new_err(
        "delta_pos must be a length-3 sequence or an (N, 3) array",
    ))
}

/// Coerce a `target_pos` argument (already resolved by Python to
/// `(N_pin, 3)` because the singleton-target case needs `obj.position`).
fn coerce_move_to_target(
    target: &Bound<'_, PyAny>,
    n_pin_verts: usize,
) -> PyResult<Array2<f64>> {
    if let Ok(arr2) = target.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr2.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyException::new_err(
                "target_pos must reshape into (N, 3)",
            ));
        }
        if shape[0] != n_pin_verts {
            return Err(PyException::new_err(
                "target_pos must have the same length as pin",
            ));
        }
        return Ok(arr2.as_array().to_owned());
    }
    if let Ok(list) = target.downcast::<PyList>() {
        let mut rows: Vec<[f64; 3]> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let row: Vec<f64> = item.extract().map_err(|_| {
                PyException::new_err("target_pos rows must be length-3 sequences")
            })?;
            if row.len() != 3 {
                return Err(PyException::new_err(
                    "target_pos rows must be length-3 sequences",
                ));
            }
            rows.push([row[0], row[1], row[2]]);
        }
        if rows.len() != n_pin_verts {
            return Err(PyException::new_err(
                "target_pos must have the same length as pin",
            ));
        }
        let mut arr = Array2::<f64>::zeros((rows.len(), 3));
        for (i, r) in rows.iter().enumerate() {
            arr[[i, 0]] = r[0];
            arr[[i, 1]] = r[1];
            arr[[i, 2]] = r[2];
        }
        return Ok(arr);
    }
    Err(PyException::new_err(
        "target_pos must be a list or numpy array of (N, 3)",
    ))
}

fn parse_center_mode(s: &str) -> PyResult<CenterMode> {
    match s {
        "absolute" => Ok(CenterMode::Absolute),
        "centroid" | "relative" => Ok(CenterMode::Relative),
        other => Err(PyValueError::new_err(format!(
            "center_mode must be 'absolute' or 'centroid' (got '{other}')"
        ))),
    }
}

/// PyO3 wrapper around `core::datamodel::pin::PinData`. Exposes the
/// builder surface from `frontend/_scene_.py::PinHolder` so the Python
/// dispatcher is a drop-in.
#[pyclass(name = "PinHolder", module = "_ppf_cts_py")]
pub struct PinHolder {
    inner: PinData,
}

#[pymethods]
impl PinHolder {
    /// Build a fresh holder. `indices` is the list of pinned local
    /// vertex indices; `pin_group_id` is the unique label assigned by
    /// the Python `PinHolder` constructor. We accept it as a
    /// constructor arg so the Rust mirror doesn't try to maintain a
    /// counter parallel to the Python class.
    #[new]
    #[pyo3(signature = (indices, pin_group_id))]
    fn new(indices: Vec<i32>, pin_group_id: String) -> Self {
        Self {
            inner: PinData {
                index: indices,
                operations: Vec::new(),
                unpin_time: None,
                transition: "linear".to_string(),
                pull_strength: 0.0,
                pin_group_id,
                hide_in_preview: false,
            },
        }
    }

    /// Set the default transition for subsequently added operations.
    fn interp<'py>(
        mut slf: PyRefMut<'py, Self>,
        transition: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        validate_transition(transition)?;
        slf.inner.transition = transition.to_string();
        Ok(slf)
    }

    /// Set the unpin time (must be non-negative).
    fn unpin<'py>(
        mut slf: PyRefMut<'py, Self>,
        time: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if time < 0.0 {
            return Err(PyException::new_err("unpin time must be non-negative"));
        }
        slf.inner.unpin_time = Some(time);
        Ok(slf)
    }

    fn pull<'py>(
        mut slf: PyRefMut<'py, Self>,
        strength: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.pull_strength = strength;
        Ok(slf)
    }

    /// Append a `MoveBy` operation. `delta_pos` may be a length-3
    /// sequence or an (N, 3) array, where N matches the pin's vertex
    /// count.
    #[pyo3(signature = (delta_pos, t_start=0.0, t_end=1.0, transition=None, bezier_handles=None))]
    fn move_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        delta_pos: &Bound<'_, PyAny>,
        t_start: f64,
        t_end: f64,
        transition: Option<&str>,
        bezier_handles: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if t_end <= t_start {
            return Err(PyException::new_err("t_end must be greater than t_start"));
        }
        let n = slf.inner.index.len();
        let delta = coerce_move_by_delta(delta_pos, n)?;
        let trans_str = match transition {
            Some(t) => {
                validate_transition(t)?;
                t.to_string()
            }
            None => slf.inner.transition.clone(),
        };
        let handles = extract_bezier_handles(bezier_handles)?;
        slf.inner.operations.push(PinOperation::MoveBy {
            delta,
            t_start,
            t_end,
            transition: trans_str,
            bezier_handles: handles,
        });
        Ok(slf)
    }

    /// Append a `MoveTo` operation. `target_pos` must already be
    /// resolved to `(N_pin, 3)` by the Python caller (the singleton
    /// case needs `obj.position`, which lives on the Python side).
    #[pyo3(signature = (target_pos, t_start=0.0, t_end=1.0, transition=None, bezier_handles=None))]
    fn move_to<'py>(
        mut slf: PyRefMut<'py, Self>,
        target_pos: &Bound<'_, PyAny>,
        t_start: f64,
        t_end: f64,
        transition: Option<&str>,
        bezier_handles: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if t_end <= t_start {
            return Err(PyException::new_err("t_end must be greater than t_start"));
        }
        let n = slf.inner.index.len();
        let target = coerce_move_to_target(target_pos, n)?;
        let trans_str = match transition {
            Some(t) => {
                validate_transition(t)?;
                t.to_string()
            }
            None => slf.inner.transition.clone(),
        };
        let handles = extract_bezier_handles(bezier_handles)?;
        slf.inner.operations.push(PinOperation::MoveTo {
            target,
            t_start,
            t_end,
            transition: trans_str,
            bezier_handles: handles,
        });
        Ok(slf)
    }

    /// Append a `Spin` operation.
    #[pyo3(signature = (
        center=None, axis=None, angular_velocity=360.0,
        t_start=0.0, t_end=f64::INFINITY, center_mode="absolute"
    ))]
    fn spin<'py>(
        mut slf: PyRefMut<'py, Self>,
        center: Option<&Bound<'_, PyAny>>,
        axis: Option<&Bound<'_, PyAny>>,
        angular_velocity: f64,
        t_start: f64,
        t_end: f64,
        center_mode: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let center_v = match center {
            Some(c) => extract_vec3(c, "center")?,
            None => [0.0, 0.0, 0.0],
        };
        let axis_v = match axis {
            Some(a) => extract_vec3(a, "axis")?,
            None => [0.0, 1.0, 0.0],
        };
        let mode = parse_center_mode(center_mode)?;
        slf.inner.operations.push(PinOperation::Spin {
            center: center_v,
            axis: axis_v,
            angular_velocity,
            t_start,
            t_end,
            center_mode: mode,
        });
        Ok(slf)
    }

    /// Append a `Scale` operation. The transition is taken from the
    /// holder's current default (Python's behavior).
    #[pyo3(signature = (
        scale, t_start=0.0, t_end=1.0,
        center=None, center_mode="absolute"
    ))]
    fn scale<'py>(
        mut slf: PyRefMut<'py, Self>,
        scale: f64,
        t_start: f64,
        t_end: f64,
        center: Option<&Bound<'_, PyAny>>,
        center_mode: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let center_v = match center {
            Some(c) => extract_vec3(c, "center")?,
            None => [0.0, 0.0, 0.0],
        };
        let mode = parse_center_mode(center_mode)?;
        let trans = slf.inner.transition.clone();
        slf.inner.operations.push(PinOperation::Scale {
            center: center_v,
            factor: scale,
            t_start,
            t_end,
            transition: trans,
            // Python's `ScaleOperation` has no `bezier_handles` field;
            // mirror that by leaving handles unset.
            bezier_handles: None,
            center_mode: mode,
        });
        Ok(slf)
    }

    /// Append a `Torque` operation.
    #[pyo3(signature = (
        magnitude=1.0, axis_component=2, hint_vertex=-1,
        t_start=0.0, t_end=f64::INFINITY
    ))]
    fn torque<'py>(
        mut slf: PyRefMut<'py, Self>,
        magnitude: f64,
        axis_component: i32,
        hint_vertex: i32,
        t_start: f64,
        t_end: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.operations.push(PinOperation::Torque {
            axis_component,
            magnitude,
            hint_vertex,
            t_start,
            t_end,
        });
        Ok(slf)
    }

    /// Append a `TransformKeyframe` operation. Inputs match the Python
    /// signature; segments are dicts with `interpolation` ∈
    /// `{"LINEAR", "BEZIER", "CONSTANT"}` plus `handle_right` /
    /// `handle_left` for the bezier case.
    #[pyo3(signature = (
        local_vert, times, translations, quaternions, scales, segments, rest_translation
    ))]
    fn transform_keyframes<'py>(
        mut slf: PyRefMut<'py, Self>,
        local_vert: PyReadonlyArray2<'_, f64>,
        times: Vec<f64>,
        translations: &Bound<'_, PyList>,
        quaternions: &Bound<'_, PyList>,
        scales: &Bound<'_, PyList>,
        segments: &Bound<'_, PyList>,
        rest_translation: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let n = times.len();
        if translations.len() != n || quaternions.len() != n || scales.len() != n {
            return Err(PyValueError::new_err(
                "transform_keyframes: T/Q/S arrays must match times length",
            ));
        }
        if n > 0 && segments.len() != n - 1 {
            return Err(PyValueError::new_err(format!(
                "transform_keyframes: expected {} segments, got {}",
                n - 1,
                segments.len()
            )));
        }
        let local = local_vert.as_array().to_owned();
        let mut translations_v: Vec<Vec3> = Vec::with_capacity(n);
        let mut scales_v: Vec<Vec3> = Vec::with_capacity(n);
        let mut quaternions_v: Vec<Quat> = Vec::with_capacity(n);
        for i in 0..n {
            translations_v.push(extract_vec3(&translations.get_item(i)?, "translations")?);
            scales_v.push(extract_vec3(&scales.get_item(i)?, "scales")?);
            let q: Vec<f64> = quaternions.get_item(i)?.extract().map_err(|_| {
                PyValueError::new_err("quaternions entries must be length-4 sequences")
            })?;
            if q.len() != 4 {
                return Err(PyValueError::new_err(
                    "quaternions entries must be length-4 sequences",
                ));
            }
            quaternions_v.push([q[0], q[1], q[2], q[3]]);
        }
        let mut segs: Vec<KeyframeSegment> = Vec::with_capacity(segments.len());
        let allowed = ["LINEAR", "BEZIER", "CONSTANT"];
        for i in 0..segments.len() {
            let item = segments.get_item(i)?;
            let dict = item.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("transform_keyframes segment must be a dict")
            })?;
            let interp_str: String = match dict.get_item("interpolation")? {
                Some(v) => v.extract()?,
                None => "LINEAR".to_string(),
            };
            if !allowed.contains(&interp_str.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "transform_keyframes segment {i}: unsupported interpolation '{interp_str}'. \
                     Supported: [\"BEZIER\", \"CONSTANT\", \"LINEAR\"]"
                )));
            }
            let interp = match interp_str.as_str() {
                "LINEAR" => InterpMode::Linear,
                "CONSTANT" => InterpMode::Constant,
                "BEZIER" => {
                    let hr = dict.get_item("handle_right")?.ok_or_else(|| {
                        PyValueError::new_err(
                            "BEZIER segment missing 'handle_right'",
                        )
                    })?;
                    let hl = dict.get_item("handle_left")?.ok_or_else(|| {
                        PyValueError::new_err(
                            "BEZIER segment missing 'handle_left'",
                        )
                    })?;
                    let hr_v: Vec<f64> = hr.extract()?;
                    let hl_v: Vec<f64> = hl.extract()?;
                    if hr_v.len() != 2 || hl_v.len() != 2 {
                        return Err(PyValueError::new_err(
                            "handle_right / handle_left must be length-2 sequences",
                        ));
                    }
                    InterpMode::Bezier {
                        right_handle: [hr_v[0], hr_v[1]],
                        left_handle: [hl_v[0], hl_v[1]],
                    }
                }
                _ => unreachable!(),
            };
            segs.push(KeyframeSegment { interp });
        }
        let rest = extract_vec3(rest_translation, "rest_translation")?;
        slf.inner
            .operations
            .push(PinOperation::TransformKeyframe {
                local_vert: local,
                times,
                translations: translations_v,
                quaternions: quaternions_v,
                scales: scales_v,
                segments: segs,
                rest_translation: rest,
            });
        Ok(slf)
    }

    // ----- Read-side accessors. Mirrors Python `PinData` fields so the
    // Python wrapper can shadow them onto its dataclass.

    #[getter]
    fn index(&self) -> Vec<i32> {
        self.inner.index.clone()
    }

    #[getter]
    fn transition(&self) -> &str {
        &self.inner.transition
    }

    #[getter]
    fn unpin_time(&self) -> Option<f64> {
        self.inner.unpin_time
    }

    #[getter]
    fn pull_strength(&self) -> f64 {
        self.inner.pull_strength
    }

    #[getter]
    fn pin_group_id(&self) -> &str {
        &self.inner.pin_group_id
    }

    #[getter]
    fn hide_in_preview(&self) -> bool {
        self.inner.hide_in_preview
    }

    /// Number of operations in the chain. Lets the parity test confirm
    /// the Rust mirror appended in lockstep with Python.
    fn op_count(&self) -> usize {
        self.inner.operations.len()
    }

    /// Return the type tag of the i-th operation
    /// (`"move_by" | "move_to" | "spin" | "scale" | "torque" | "transform_keyframe"`).
    /// Used by parity tests.
    fn op_kind(&self, i: usize) -> PyResult<&'static str> {
        let op = self.inner.operations.get(i).ok_or_else(|| {
            PyException::new_err(format!("op index {i} out of range"))
        })?;
        Ok(match op {
            PinOperation::MoveBy { .. } => "move_by",
            PinOperation::MoveTo { .. } => "move_to",
            PinOperation::Spin { .. } => "spin",
            PinOperation::Scale { .. } => "scale",
            PinOperation::Torque { .. } => "torque",
            PinOperation::TransformKeyframe { .. } => "transform_keyframe",
        })
    }

    /// Return the (t_start, t_end) tuple of the i-th operation.
    fn op_time_range(&self, i: usize) -> PyResult<(f64, f64)> {
        let op = self.inner.operations.get(i).ok_or_else(|| {
            PyException::new_err(format!("op index {i} out of range"))
        })?;
        Ok(op.time_range())
    }

    /// For `MoveBy` ops: report whether the stored delta is uniform
    /// (and the value) or per-vertex (with shape). Lets parity tests
    /// confirm the schema fork is exercised.
    fn op_move_by_delta_kind(&self, i: usize) -> PyResult<String> {
        let op = self.inner.operations.get(i).ok_or_else(|| {
            PyException::new_err(format!("op index {i} out of range"))
        })?;
        match op {
            PinOperation::MoveBy { delta, .. } => Ok(match delta {
                MoveByDelta::Uniform(_) => "uniform".to_string(),
                MoveByDelta::PerVertex(_) => "per_vertex".to_string(),
            }),
            _ => Err(PyException::new_err(format!(
                "op {i} is not a MoveBy",
            ))),
        }
    }

    /// Echo back the transition + bezier_handles of an op that supports
    /// easing (`MoveBy`, `MoveTo`, `Scale`). Returns
    /// `(transition_str, bezier_handles_or_none)`. Used by parity tests
    /// to confirm we round-trip the two fields independently.
    fn op_transition<'py>(
        &self,
        py: Python<'py>,
        i: usize,
    ) -> PyResult<(String, PyObject)> {
        let op = self.inner.operations.get(i).ok_or_else(|| {
            PyException::new_err(format!("op index {i} out of range"))
        })?;
        let (t, h): (String, Option<[[f64; 2]; 2]>) = match op {
            PinOperation::MoveBy {
                transition,
                bezier_handles,
                ..
            }
            | PinOperation::MoveTo {
                transition,
                bezier_handles,
                ..
            }
            | PinOperation::Scale {
                transition,
                bezier_handles,
                ..
            } => (transition.clone(), *bezier_handles),
            _ => {
                return Err(PyException::new_err(format!(
                    "op {i} does not carry a transition",
                )))
            }
        };
        use pyo3::IntoPyObjectExt;
        let h_obj = match h {
            Some([r, l]) => ((r[0], r[1]), (l[0], l[1])).into_py_any(py)?,
            None => py.None(),
        };
        Ok((t, h_obj))
    }

    fn __len__(&self) -> usize {
        self.inner.operations.len()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PinHolder>()?;
    Ok(())
}
