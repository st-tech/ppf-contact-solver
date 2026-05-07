// File: crates/ppf-cts-py/src/scene_py/loops.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// scene_loops bindings: kernels that replace Python `for ...:
// list.append(...)` hotspots. Each binding mirrors one Python loop in
// frontend/_scene_.py.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ppf_cts_core::kernels::scene_loops as sl;

use crate::errors::into_py_err;

/// Stitch preview build (frontend/_scene_.py ~2568-2579).
#[pyfunction]
#[pyo3(signature = (vert, stitch_ind, stitch_w))]
pub(super) fn scene_stitch_preview_lines<'py>(
    py: Python<'py>,
    vert: PyReadonlyArray2<'py, f64>,
    stitch_ind: PyReadonlyArray2<'py, i64>,
    stitch_w: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, numpy::PyArray2<u32>>)> {
    let n_vert = crate::utils_py::require_n_by_k::<_, 3>(&vert, "vert")?;
    let n_stitch = crate::utils_py::require_n_by_k::<_, 4>(&stitch_ind, "stitch_ind")?;
    let ws = stitch_w.shape();
    if ws.len() != 2 || ws[1] != 4 || ws[0] != n_stitch {
        return Err(PyValueError::new_err(format!(
            "stitch_w must be ({n_stitch}, 4), got {ws:?}"
        )));
    }
    let v_slice = vert
        .as_slice()
        .map_err(|_| PyTypeError::new_err("vert must be C-contiguous"))?;
    let i_slice = stitch_ind
        .as_slice()
        .map_err(|_| PyTypeError::new_err("stitch_ind must be C-contiguous"))?;
    let w_slice = stitch_w
        .as_slice()
        .map_err(|_| PyTypeError::new_err("stitch_w must be C-contiguous"))?;
    let (sv, se) = py.allow_threads(|| {
        sl::stitch_preview_lines(v_slice, n_vert, i_slice, w_slice, n_stitch)
    });
    let v_arr = ndarray::Array2::from_shape_vec((2 * n_stitch, 3), sv)
        .map_err(|e| PyValueError::new_err(format!("stitch vert reshape failed: {e}")))?;
    let e_arr = ndarray::Array2::from_shape_vec((n_stitch, 2), se)
        .map_err(|e| PyValueError::new_err(format!("stitch edge reshape failed: {e}")))?;
    Ok((v_arr.into_pyarray(py), e_arr.into_pyarray(py)))
}

/// Per-face UV expansion (frontend/_scene_.py ~2773-2782). Returns a
/// (n_face, 3, 2) f64 array. Caller can reshape if a flat (n_face, 6)
/// view is preferred.
#[pyfunction]
#[pyo3(signature = (vertex_uv, faces))]
pub(super) fn scene_face_uv_expand<'py>(
    py: Python<'py>,
    vertex_uv: PyReadonlyArray2<'py, f64>,
    faces: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
    let vs = vertex_uv.shape();
    if vs.len() != 2 || vs[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "vertex_uv must be (N, 2), got {vs:?}"
        )));
    }
    let n_vert = vs[0];
    let face_vec = read_face_indices_i64(faces)?;
    let n_face = face_vec.len() / 3;
    let uv_slice = vertex_uv
        .as_slice()
        .map_err(|_| PyTypeError::new_err("vertex_uv must be C-contiguous"))?;
    let buf = py
        .allow_threads(|| sl::face_uv_expand(uv_slice, n_vert, &face_vec, n_face))
        .map_err(into_py_err)?;
    let arr = ndarray::Array3::from_shape_vec((n_face, 3, 2), buf)
        .map_err(|e| PyValueError::new_err(format!("face_uv reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

// Accepts only int32 and int64 (no u32/u64), so it stays separate from
// `crate::utils_py::read_index_array::<i64, 3>` which is broader. See
// utils_py.rs for the general helper.
fn read_face_indices_i64(arr: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i64>>() {
        let s = view.shape();
        if s.len() != 2 || s[1] != 3 {
            return Err(PyValueError::new_err(format!("faces must be (N, 3), got {s:?}")));
        }
        let slice = view
            .as_slice()
            .map_err(|_| PyTypeError::new_err("faces must be C-contiguous"))?;
        return Ok(slice.to_vec());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i32>>() {
        let s = view.shape();
        if s.len() != 2 || s[1] != 3 {
            return Err(PyValueError::new_err(format!("faces must be (N, 3), got {s:?}")));
        }
        let slice = view
            .as_slice()
            .map_err(|_| PyTypeError::new_err("faces must be C-contiguous"))?;
        return Ok(slice.iter().map(|&x| x as i64).collect());
    }
    Err(PyTypeError::new_err(
        "faces must be (N, 3) ndarray of int32 or int64",
    ))
}

/// Pin-marker index collection (frontend/_scene_.py ~2597-2604).
#[pyfunction]
#[pyo3(signature = (hide_flags, indices))]
pub(super) fn scene_collect_pin_marker_indices<'py>(
    py: Python<'py>,
    hide_flags: Vec<bool>,
    indices: Vec<Vec<i64>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if hide_flags.len() != indices.len() {
        return Err(PyValueError::new_err(format!(
            "hide_flags len {} != indices len {}",
            hide_flags.len(),
            indices.len()
        )));
    }
    let mut offsets = Vec::with_capacity(indices.len() + 1);
    offsets.push(0_usize);
    let mut data: Vec<i64> = Vec::new();
    for v in &indices {
        data.extend_from_slice(v);
        offsets.push(data.len());
    }
    let out = sl::collect_pin_marker_indices(&hide_flags, &offsets, &data);
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// Static transform-animation keyframe concat
/// (frontend/_scene_.py ~2207-2215). Caller pre-flattens each anim's
/// numpy arrays in iteration order; this kernel just validates shapes
/// and returns the four buffers as numpy arrays.
#[pyfunction]
#[pyo3(signature = (kf_counts, times, trans, quats, scales))]
pub(super) fn scene_concat_static_transform_anims<'py>(
    py: Python<'py>,
    kf_counts: Vec<usize>,
    times: PyReadonlyArray1<'py, f64>,
    trans: PyReadonlyArray2<'py, f64>,
    quats: PyReadonlyArray2<'py, f64>,
    scales: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let total_kf: usize = kf_counts.iter().sum();
    let times_slice = times
        .as_slice()
        .map_err(|_| PyTypeError::new_err("times must be C-contiguous"))?;
    if times_slice.len() != total_kf {
        return Err(PyValueError::new_err(format!(
            "times len {} != total kf {}",
            times_slice.len(),
            total_kf
        )));
    }
    let trans_slice = trans
        .as_slice()
        .map_err(|_| PyTypeError::new_err("trans must be C-contiguous"))?;
    let quats_slice = quats
        .as_slice()
        .map_err(|_| PyTypeError::new_err("quats must be C-contiguous"))?;
    let scales_slice = scales
        .as_slice()
        .map_err(|_| PyTypeError::new_err("scales must be C-contiguous"))?;
    sl::concat_static_transform_anims(
        &kf_counts,
        times_slice,
        trans_slice,
        quats_slice,
        scales_slice,
    )
    .map_err(into_py_err)?;
    let d = PyDict::new(py);
    d.set_item("total_keyframes", total_kf)?;
    Ok(d)
}

/// Transform-keyframe segment packing
/// (frontend/_scene_.py ~2277-2293).
#[pyfunction]
#[pyo3(signature = (interp_strs, handles_right, handles_left))]
pub(super) fn scene_pack_transform_keyframe_segments<'py>(
    py: Python<'py>,
    interp_strs: Vec<String>,
    handles_right: PyReadonlyArray2<'py, f64>,
    handles_left: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, numpy::PyArray2<f64>>)> {
    let n_seg = interp_strs.len();
    let hr_shape = handles_right.shape();
    if hr_shape.len() != 2 || hr_shape[1] != 2 || hr_shape[0] != n_seg {
        return Err(PyValueError::new_err(format!(
            "handles_right must be ({n_seg}, 2), got {hr_shape:?}"
        )));
    }
    let hl_shape = handles_left.shape();
    if hl_shape.len() != 2 || hl_shape[1] != 2 || hl_shape[0] != n_seg {
        return Err(PyValueError::new_err(format!(
            "handles_left must be ({n_seg}, 2), got {hl_shape:?}"
        )));
    }
    let hr_slice = handles_right
        .as_slice()
        .map_err(|_| PyTypeError::new_err("handles_right must be C-contiguous"))?;
    let hl_slice = handles_left
        .as_slice()
        .map_err(|_| PyTypeError::new_err("handles_left must be C-contiguous"))?;
    let strs: Vec<&str> = interp_strs.iter().map(|s| s.as_str()).collect();
    let (codes, handles) = sl::pack_transform_keyframe_segments(&strs, hr_slice, hl_slice)
        .map_err(into_py_err)?;
    let codes_arr = numpy::PyArray1::from_vec(py, codes);
    let handles_arr = ndarray::Array2::from_shape_vec((n_seg, 4), handles)
        .map_err(|e| PyValueError::new_err(format!("handles reshape failed: {e}")))?;
    Ok((codes_arr, handles_arr.into_pyarray(py)))
}

/// Pin TOML formatting (frontend/_scene_.py ~2014-2076). Input ops are
/// flat dicts (one per pin op) with a `type` discriminator; pin headers
/// are flat dicts. Returns a single string.
#[pyfunction]
#[pyo3(signature = (pin_blocks))]
pub(super) fn scene_format_pin_toml<'py>(
    pin_blocks: &Bound<'py, PyList>,
) -> PyResult<String> {
    let mut headers: Vec<sl::PinHeader> = Vec::with_capacity(pin_blocks.len());
    let mut ops_offsets: Vec<usize> = Vec::with_capacity(pin_blocks.len() + 1);
    ops_offsets.push(0);
    let mut ops_flat: Vec<sl::PinOpToml> = Vec::new();
    for block in pin_blocks.iter() {
        let d = block
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each pin_blocks entry must be a dict"))?;
        let header = decode_pin_header(d)?;
        let ops_item = d
            .get_item("ops")?
            .ok_or_else(|| PyValueError::new_err("pin block missing 'ops'"))?;
        let ops_list = ops_item
            .downcast::<PyList>()
            .map_err(|_| PyValueError::new_err("'ops' must be a list of dicts"))?;
        for op in ops_list.iter() {
            let od = op
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("each op must be a dict"))?;
            ops_flat.push(decode_pin_op(od)?);
        }
        ops_offsets.push(ops_flat.len());
        headers.push(header);
    }
    sl::format_all_pin_sections(&headers, &ops_offsets, &ops_flat)
        .map_err(into_py_err)
}

fn decode_pin_header(d: &Bound<'_, PyDict>) -> PyResult<sl::PinHeader> {
    let operation_count: usize = d
        .get_item("operation_count")?
        .ok_or_else(|| PyValueError::new_err("missing operation_count"))?
        .extract()?;
    let pin_count: usize = d
        .get_item("pin_count")?
        .ok_or_else(|| PyValueError::new_err("missing pin_count"))?
        .extract()?;
    let pull_strength: f64 = d
        .get_item("pull_strength")?
        .ok_or_else(|| PyValueError::new_err("missing pull_strength"))?
        .extract()?;
    let unpin_time: Option<f64> = d
        .get_item("unpin_time")?
        .map(|v| v.extract::<Option<f64>>())
        .transpose()?
        .flatten();
    let pin_group_id: Option<String> = d
        .get_item("pin_group_id")?
        .map(|v| v.extract::<Option<String>>())
        .transpose()?
        .flatten();
    Ok(sl::PinHeader {
        operation_count,
        pin_count,
        pull_strength,
        unpin_time,
        pin_group_id,
    })
}

fn decode_pin_op(d: &Bound<'_, PyDict>) -> PyResult<sl::PinOpToml> {
    let ty: String = d
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("op missing 'type'"))?
        .extract()?;
    match ty.as_str() {
        "move_by" => Ok(sl::PinOpToml::MoveBy {
            t_start: get_f64(d, "t_start")?,
            t_end: get_f64(d, "t_end")?,
            transition: get_string(d, "transition")?,
            bezier_handles: get_optional_handles(d)?,
        }),
        "move_to" => Ok(sl::PinOpToml::MoveTo {
            t_start: get_f64(d, "t_start")?,
            t_end: get_f64(d, "t_end")?,
            transition: get_string(d, "transition")?,
            bezier_handles: get_optional_handles(d)?,
        }),
        "spin" => Ok(sl::PinOpToml::Spin {
            center_mode: get_string(d, "center_mode")?,
            center: get_arr3(d, "center")?,
            axis: get_arr3(d, "axis")?,
            angular_velocity: get_f64(d, "angular_velocity")?,
            t_start: get_f64(d, "t_start")?,
            t_end: get_f64(d, "t_end")?,
        }),
        "scale" => Ok(sl::PinOpToml::Scale {
            center_mode: get_string(d, "center_mode")?,
            center: get_arr3(d, "center")?,
            factor: get_f64(d, "factor")?,
            t_start: get_f64(d, "t_start")?,
            t_end: get_f64(d, "t_end")?,
            transition: get_string(d, "transition")?,
            bezier_handles: get_optional_handles(d)?,
        }),
        "torque" => Ok(sl::PinOpToml::Torque {
            axis_component: get_i64(d, "axis_component")?,
            magnitude: get_f64(d, "magnitude")?,
            hint_vertex: get_i64(d, "hint_vertex")?,
            t_start: get_f64(d, "t_start")?,
            t_end: get_f64(d, "t_end")?,
        }),
        "transform_keyframes" => Ok(sl::PinOpToml::TransformKeyframes {
            keyframe_count: d
                .get_item("keyframe_count")?
                .ok_or_else(|| PyValueError::new_err("missing keyframe_count"))?
                .extract()?,
            t_start: get_f64(d, "t_start")?,
            t_end: get_f64(d, "t_end")?,
            rest_translation: get_arr3(d, "rest_translation")?,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown pin op type {other:?}"
        ))),
    }
}

fn get_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

fn get_i64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

fn get_string(d: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

fn get_arr3(d: &Bound<'_, PyDict>, key: &str) -> PyResult<[f64; 3]> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

/// Read an optional `bezier_handles = [hr_x, hr_y, hl_x, hl_y]` array.
/// Absent or `None` -> `None` (linear fallback). Mirrors the
/// `Option<[f64; 4]>` shape the centralized `pin_apply::progress_at`
/// expects.
fn get_optional_handles(d: &Bound<'_, PyDict>) -> PyResult<Option<[f64; 4]>> {
    let item = match d.get_item("bezier_handles")? {
        Some(v) => v,
        None => return Ok(None),
    };
    if item.is_none() {
        return Ok(None);
    }
    let parsed: [f64; 4] = item.extract().map_err(|_| {
        PyValueError::new_err("bezier_handles must be [hr_x, hr_y, hl_x, hl_y]")
    })?;
    Ok(Some(parsed))
}

/// Wall TOML formatting (frontend/_scene_.py ~2078-2088).
#[pyfunction]
#[pyo3(signature = (walls))]
pub(super) fn scene_format_wall_toml<'py>(walls: &Bound<'py, PyList>) -> PyResult<String> {
    let mut decoded: Vec<sl::WallToml> = Vec::with_capacity(walls.len());
    for w in walls.iter() {
        let d = w
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each wall must be a dict"))?;
        let keyframe: usize = d
            .get_item("keyframe")?
            .ok_or_else(|| PyValueError::new_err("missing keyframe"))?
            .extract()?;
        let normal: [f64; 3] = get_arr3(d, "normal")?;
        let transition: String = get_string(d, "transition")?;
        let params = decode_param_pairs(d)?;
        decoded.push(sl::WallToml {
            keyframe,
            normal,
            transition,
            params,
        });
    }
    Ok(sl::format_wall_sections(&decoded))
}

/// Sphere TOML formatting (frontend/_scene_.py ~2090-2098).
#[pyfunction]
#[pyo3(signature = (spheres))]
pub(super) fn scene_format_sphere_toml<'py>(spheres: &Bound<'py, PyList>) -> PyResult<String> {
    let mut decoded: Vec<sl::SphereToml> = Vec::with_capacity(spheres.len());
    for s in spheres.iter() {
        let d = s
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("each sphere must be a dict"))?;
        let keyframe: usize = d
            .get_item("keyframe")?
            .ok_or_else(|| PyValueError::new_err("missing keyframe"))?
            .extract()?;
        let is_hemisphere: bool = d
            .get_item("is_hemisphere")?
            .ok_or_else(|| PyValueError::new_err("missing is_hemisphere"))?
            .extract()?;
        let is_inverted: bool = d
            .get_item("is_inverted")?
            .ok_or_else(|| PyValueError::new_err("missing is_inverted"))?
            .extract()?;
        let transition: String = get_string(d, "transition")?;
        let params = decode_param_pairs(d)?;
        decoded.push(sl::SphereToml {
            keyframe,
            is_hemisphere,
            is_inverted,
            transition,
            params,
        });
    }
    Ok(sl::format_sphere_sections(&decoded))
}

fn decode_param_pairs(d: &Bound<'_, PyDict>) -> PyResult<Vec<(String, String)>> {
    let item = d
        .get_item("params")?
        .ok_or_else(|| PyValueError::new_err("missing 'params'"))?;
    let list = item
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("'params' must be a list of (key, value) tuples"))?;
    let mut out: Vec<(String, String)> = Vec::with_capacity(list.len());
    for entry in list.iter() {
        let pair: (String, String) = entry.extract()?;
        out.push(pair);
    }
    Ok(out)
}

/// dyn_param.txt formatting (frontend/_scene_.py ~2152-2163). Input is
/// a list of `(key, entries)` where each entry is either
/// `(time, [vx, vy, vz])` (velocity) or `(t_start, t_end)` (collision
/// window). The discriminator is the inner length: a 4-tuple is
/// velocity, a 2-tuple is collision.
#[pyfunction]
#[pyo3(signature = (blocks))]
pub(super) fn scene_format_dyn_param_toml<'py>(blocks: &Bound<'py, PyList>) -> PyResult<String> {
    let mut decoded: Vec<(String, Vec<sl::DynParamEntry>)> = Vec::with_capacity(blocks.len());
    for blk in blocks.iter() {
        let tup: (String, Bound<'_, PyList>) = blk.extract()?;
        let key = tup.0;
        let entries_list = tup.1;
        let mut entries: Vec<sl::DynParamEntry> = Vec::with_capacity(entries_list.len());
        for entry in entries_list.iter() {
            // Try (f64, [f64; 3]) first (velocity).
            if let Ok((t, v)) = entry.extract::<(f64, [f64; 3])>() {
                entries.push(sl::DynParamEntry::Velocity(t, v));
                continue;
            }
            // Then (f64, f64) (collision window).
            if let Ok((s, e)) = entry.extract::<(f64, f64)>() {
                entries.push(sl::DynParamEntry::CollisionWindow(s, e));
                continue;
            }
            return Err(PyValueError::new_err(
                "each dyn_param entry must be (t, [vx, vy, vz]) or (t_start, t_end)",
            ));
        }
        decoded.push((key, entries));
    }
    Ok(sl::format_dyn_param_sections(&decoded))
}

/// static_transform header section (frontend/_scene_.py ~2001-2012).
#[pyfunction]
#[pyo3(signature = (object_count, total_keyframes, keyframe_counts, vert_counts, vert_offsets))]
pub(super) fn scene_format_static_transform_header(
    object_count: usize,
    total_keyframes: usize,
    keyframe_counts: Vec<usize>,
    vert_counts: Vec<usize>,
    vert_offsets: Vec<usize>,
) -> PyResult<String> {
    Ok(sl::format_static_transform_header(
        object_count,
        total_keyframes,
        &keyframe_counts,
        &vert_counts,
        &vert_offsets,
    ))
}

/// Per-object axis bound (frontend/_scene_.py ~2913-2920 and
/// ~2937-2945). Caller passes a flat (sum_n, 3) vertex buffer plus
/// per-object cumulative offsets.
#[pyfunction]
#[pyo3(signature = (flat_verts, object_offsets, axis, is_min))]
pub(super) fn scene_per_object_axis_bound<'py>(
    flat_verts: PyReadonlyArray2<'py, f64>,
    object_offsets: Vec<usize>,
    axis: usize,
    is_min: bool,
) -> PyResult<f64> {
    let s = flat_verts.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "flat_verts must be (N, 3), got {s:?}"
        )));
    }
    let slice = flat_verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("flat_verts must be C-contiguous"))?;
    sl::per_object_axis_bound(slice, &object_offsets, axis, is_min)
        .map_err(into_py_err)
}

/// Concat per-object pin index lists into a single i64 array.
/// Replaces (frontend/_scene_.py ~3122-3125).
#[pyfunction]
#[pyo3(signature = (parts))]
pub(super) fn scene_concat_i64_lists<'py>(
    py: Python<'py>,
    parts: Vec<Vec<i64>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in &parts {
        out.extend_from_slice(p);
    }
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// Validate cross-stitches: every entry's source/target name must be in
/// `known_names`. Replaces (frontend/_scene_.py ~3218-3230) inner check.
#[pyfunction]
#[pyo3(signature = (pairs, known_names))]
pub(super) fn scene_validate_cross_stitch_names(
    pairs: Vec<(String, String)>,
    known_names: Vec<String>,
) -> PyResult<()> {
    let known: std::collections::HashSet<&str> =
        known_names.iter().map(|s| s.as_str()).collect();
    for (s, t) in pairs {
        if !known.contains(s.as_str()) || !known.contains(t.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Cross-stitch references unknown object: source={s:?} target={t:?}"
            )));
        }
    }
    Ok(())
}
