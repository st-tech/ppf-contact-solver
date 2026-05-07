// File: crates/ppf-cts-py/src/scene_py/transform.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Vertex / matrix / quaternion math bindings used by `Object` and
// `FixedScene`. Includes:
//   * apply_transform_batch / area_weighted_center / dynamic_color
//   * bbox + axis-min/max + grab + mat4 scale/rotate
//   * violation messages
//   * quaternion helpers (multiply / to_mat3 / from_axis_angle / from_mat3
//     / slerp) and TRS apply / decompose

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use ppf_cts_core::kernels::scene_build as sb;

use super::helpers::read_faces_u32;
use crate::errors::into_py_err;

/// Apply a 4x4 row-major transform's 3x3 + translation column to an
/// `(N, 3)` f64 vertex buffer. Allocates a new array; the caller
/// passes in the input by reference (no in-place mutation).
///
/// Optional `bbox` and `center` arrays carry the
/// `Object.normalize()` pre-step. When both are supplied, vertices
/// are first replaced with `(v - center) / max(bbox)`. `bbox` is the
/// per-axis bbox extent (length 3); `center` is the bbox midpoint
/// (length 3). The Python frontend always passes them through; the
/// kernel ignores them when `bbox` is `None` or `center` is `None`.
#[pyfunction]
#[pyo3(signature = (vertices, matrix, translate, bbox=None, center=None))]
pub(super) fn scene_apply_transform_batch<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    matrix: PyReadonlyArray2<'py, f64>,
    translate: bool,
    bbox: Option<PyReadonlyArray1<'py, f64>>,
    center: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must be (N, 3), got {v_shape:?}"
        )));
    }
    let m_shape = matrix.shape();
    if m_shape.len() != 2 || m_shape[0] != 4 || m_shape[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "matrix must be (4, 4), got {m_shape:?}"
        )));
    }
    let v_slice = vertices.as_slice().map_err(|_| {
        PyTypeError::new_err("vertices must be C-contiguous")
    })?;
    let m_slice = matrix.as_slice().map_err(|_| {
        PyTypeError::new_err("matrix must be C-contiguous")
    })?;
    let mut m = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = m_slice[4 * r + c];
        }
    }

    // Optional normalize pre-step. Both `bbox` and `center` must be
    // supplied; otherwise we fall through to the plain transform.
    let normalize = match (bbox, center) {
        (Some(b), Some(c)) => {
            let b_slice = b.as_slice().map_err(|_| {
                PyTypeError::new_err("bbox must be C-contiguous")
            })?;
            let c_slice = c.as_slice().map_err(|_| {
                PyTypeError::new_err("center must be C-contiguous")
            })?;
            if b_slice.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "bbox must be length 3, got {}",
                    b_slice.len()
                )));
            }
            if c_slice.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "center must be length 3, got {}",
                    c_slice.len()
                )));
            }
            let bbox_max = b_slice
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            Some(sb::Normalize {
                bbox_max,
                center: [c_slice[0], c_slice[1], c_slice[2]],
            })
        }
        _ => None,
    };

    let n = v_shape[0];
    let mut buf = vec![0.0f64; v_slice.len()];
    buf.copy_from_slice(v_slice);

    py.allow_threads(|| sb::apply_transform_batch(&m, &mut buf, translate, normalize));

    let arr = ndarray::Array2::from_shape_vec((n, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Area-weighted centroid of a triangle mesh. Returns `(cx, cy, cz)`
/// or raises `ValueError` when the total area is zero.
#[pyfunction]
#[pyo3(signature = (vertices, tris))]
pub(super) fn scene_area_weighted_center<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    tris: &Bound<'py, PyAny>,
) -> PyResult<(f64, f64, f64)> {
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must be (N, 3), got {v_shape:?}"
        )));
    }
    let v_slice = vertices.as_slice().map_err(|_| {
        PyTypeError::new_err("vertices must be C-contiguous")
    })?;
    let tris_owned = read_faces_u32(tris)?;

    let result = py
        .allow_threads(|| sb::area_weighted_center(v_slice, &tris_owned));
    let (c, total) = result;
    if total <= 0.0 {
        return Err(PyValueError::new_err("no area"));
    }
    Ok((c[0], c[1], c[2]))
}

/// Compute the per-vertex blended color for the dyn-color hot path.
///
/// Args:
///   * `vert`: `(n_verts, 3)` f64.
///   * `tris`: `(n_tris, 3)` int.
///   * `init_area`: `(n_tris,)` f64.
///   * `face_to_vert_weights`: `(n_verts,)` f64.
///   * `dyn_face_color`: `(n_tris,)` u8 (0=NONE, !=0=AREA).
///   * `dyn_face_intensity`: `(n_tris,)` f64.
///   * `base_color`: `(n_verts, 3)` f64.
///   * `max_area`: f64.
///
/// Returns `(n_verts, 3)` f64 ndarray.
#[pyfunction]
#[pyo3(signature = (
    vert, tris, init_area, face_to_vert_weights, dyn_face_color,
    dyn_face_intensity, base_color, max_area,
))]
pub(super) fn scene_dynamic_color<'py>(
    py: Python<'py>,
    vert: PyReadonlyArray2<'py, f64>,
    tris: &Bound<'py, PyAny>,
    init_area: PyReadonlyArray1<'py, f64>,
    face_to_vert_weights: PyReadonlyArray1<'py, f64>,
    dyn_face_color: PyReadonlyArray1<'py, u8>,
    dyn_face_intensity: PyReadonlyArray1<'py, f64>,
    base_color: PyReadonlyArray2<'py, f64>,
    max_area: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let v_shape = vert.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vert must be (N, 3), got {v_shape:?}"
        )));
    }
    let n_verts = v_shape[0];
    let bc_shape = base_color.shape();
    if bc_shape.len() != 2 || bc_shape[1] != 3 || bc_shape[0] != n_verts {
        return Err(PyValueError::new_err(format!(
            "base_color must be ({n_verts}, 3), got {bc_shape:?}"
        )));
    }
    let v_slice = vert.as_slice().map_err(|_| {
        PyTypeError::new_err("vert must be C-contiguous")
    })?;
    let bc_slice = base_color.as_slice().map_err(|_| {
        PyTypeError::new_err("base_color must be C-contiguous")
    })?;
    let init_slice = init_area.as_slice().map_err(|_| {
        PyTypeError::new_err("init_area must be C-contiguous")
    })?;
    let w_slice = face_to_vert_weights.as_slice().map_err(|_| {
        PyTypeError::new_err("face_to_vert_weights must be C-contiguous")
    })?;
    let dfc_slice = dyn_face_color.as_slice().map_err(|_| {
        PyTypeError::new_err("dyn_face_color must be C-contiguous")
    })?;
    let dfi_slice = dyn_face_intensity.as_slice().map_err(|_| {
        PyTypeError::new_err("dyn_face_intensity must be C-contiguous")
    })?;
    if w_slice.len() != n_verts {
        return Err(PyValueError::new_err(format!(
            "face_to_vert_weights length {} doesn't match vert count {}",
            w_slice.len(),
            n_verts
        )));
    }
    let tris_owned = read_faces_u32(tris)?;
    let n_tris = tris_owned.len();
    if init_slice.len() != n_tris
        || dfc_slice.len() != n_tris
        || dfi_slice.len() != n_tris
    {
        return Err(PyValueError::new_err(format!(
            "init_area / dyn_face_color / dyn_face_intensity must all be length {n_tris}"
        )));
    }

    let out = py.allow_threads(|| {
        sb::dynamic_color(
            v_slice,
            &tris_owned,
            init_slice,
            w_slice,
            dfc_slice,
            dfi_slice,
            bc_slice,
            max_area,
        )
    });

    let arr = ndarray::Array2::from_shape_vec((n_verts, 3), out)
        .map_err(|e| PyValueError::new_err(format!("output reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Returns `(hi, lo)` as two `(3,)` numpy arrays for the displaced
/// vertex buffer. Mirrors `FixedScene.bbox`.
#[pyfunction]
#[pyo3(signature = (local_vert, vert_idx, displacement))]
pub(super) fn scene_bbox_displaced<'py>(
    py: Python<'py>,
    local_vert: PyReadonlyArray2<'py, f64>,
    vert_idx: PyReadonlyArray1<'py, i64>,
    displacement: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
    let lv_shape = local_vert.shape();
    if lv_shape.len() != 2 || lv_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "local_vert must be (N, 3), got {lv_shape:?}"
        )));
    }
    let disp_shape = displacement.shape();
    if disp_shape.len() != 2 || disp_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "displacement must be (M, 3), got {disp_shape:?}"
        )));
    }
    let lv = local_vert.as_slice().map_err(|_| {
        PyTypeError::new_err("local_vert must be C-contiguous")
    })?;
    let idx = vert_idx.as_slice().map_err(|_| {
        PyTypeError::new_err("vert_idx must be C-contiguous")
    })?;
    let disp = displacement.as_slice().map_err(|_| {
        PyTypeError::new_err("displacement must be C-contiguous")
    })?;
    if idx.len() != lv_shape[0] {
        return Err(PyValueError::new_err(format!(
            "vert_idx length {} mismatch with local_vert {}",
            idx.len(),
            lv_shape[0]
        )));
    }
    let (hi, lo) = py.allow_threads(|| sb::bbox_displaced(lv, idx, disp));
    Ok((
        PyArray1::<f64>::from_slice(py, &hi),
        PyArray1::<f64>::from_slice(py, &lo),
    ))
}

/// World-space axis min/max as `(min, max)` for a single axis (0/1/2).
/// Mirrors the inner branch of `Scene.min` / `Scene.max` /
/// `Object.min` / `Object.max`. Returns `(+inf, -inf)` for empty input.
#[pyfunction]
#[pyo3(signature = (verts, axis))]
pub(super) fn scene_axis_min_max<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    axis: usize,
) -> PyResult<(f64, f64)> {
    if axis > 2 {
        return Err(PyValueError::new_err(format!("axis must be 0/1/2, got {axis}")));
    }
    let s = verts.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "verts must be (N, 3), got {s:?}"
        )));
    }
    let v = verts.as_slice().map_err(|_| {
        PyTypeError::new_err("verts must be C-contiguous")
    })?;
    Ok(py.allow_threads(|| sb::axis_min_max(v, axis)))
}

/// Apply a 4x4 matrix's rotation/scale block (no translation) and
/// return the resulting bbox as `(size, center)` numpy arrays.
/// Mirrors `Object.bbox` (non-normalize branch).
#[pyfunction]
#[pyo3(signature = (local_vert, matrix))]
pub(super) fn scene_object_bbox<'py>(
    py: Python<'py>,
    local_vert: PyReadonlyArray2<'py, f64>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
    let v_shape = local_vert.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "local_vert must be (N, 3), got {v_shape:?}"
        )));
    }
    let m_shape = matrix.shape();
    if m_shape.len() != 2 || m_shape[0] != 4 || m_shape[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "matrix must be (4, 4), got {m_shape:?}"
        )));
    }
    let lv = local_vert.as_slice().map_err(|_| {
        PyTypeError::new_err("local_vert must be C-contiguous")
    })?;
    let m_slice = matrix.as_slice().map_err(|_| {
        PyTypeError::new_err("matrix must be C-contiguous")
    })?;
    let mut m = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = m_slice[4 * r + c];
        }
    }
    let (size, center) = py.allow_threads(|| sb::object_bbox_no_translate(lv, &m));
    Ok((
        PyArray1::<f64>::from_slice(py, &size),
        PyArray1::<f64>::from_slice(py, &center),
    ))
}

/// Indices of vertices whose `dot(v, dir) > max_dot - eps`. Mirrors
/// `Object.grab`. Returns a `list[int]` (Python's signature).
#[pyfunction]
#[pyo3(signature = (verts, direction, eps=1e-3))]
pub(super) fn scene_grab_indices<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    direction: [f64; 3],
    eps: f64,
) -> PyResult<Vec<i64>> {
    let s = verts.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "verts must be (N, 3), got {s:?}"
        )));
    }
    let v = verts.as_slice().map_err(|_| {
        PyTypeError::new_err("verts must be C-contiguous")
    })?;
    Ok(py.allow_threads(|| sb::grab_indices(v, direction, eps)))
}

/// Compose `M @ diag(s, s, s, 1)`. Mirrors `Object.scale`.
#[pyfunction]
#[pyo3(signature = (matrix, scale))]
pub(super) fn scene_mat4_apply_scale<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    scale: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let m_shape = matrix.shape();
    if m_shape.len() != 2 || m_shape[0] != 4 || m_shape[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "matrix must be (4, 4), got {m_shape:?}"
        )));
    }
    let s_slice = matrix.as_slice().map_err(|_| {
        PyTypeError::new_err("matrix must be C-contiguous")
    })?;
    let mut m = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = s_slice[4 * r + c];
        }
    }
    let r = py.allow_threads(|| sb::mat4_apply_uniform_scale(&m, scale));
    let mut buf = vec![0.0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            buf[4 * i + j] = r[i][j];
        }
    }
    let arr = ndarray::Array2::from_shape_vec((4, 4), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Compose `R(axis, angle_deg) @ M` and restore `M`'s translation
/// column. Mirrors `Object.rotate`.
#[pyfunction]
#[pyo3(signature = (matrix, angle_deg, axis))]
pub(super) fn scene_mat4_apply_rotate<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    angle_deg: f64,
    axis: &str,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let m_shape = matrix.shape();
    if m_shape.len() != 2 || m_shape[0] != 4 || m_shape[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "matrix must be (4, 4), got {m_shape:?}"
        )));
    }
    let axis_char = axis.chars().next().ok_or_else(|| {
        PyValueError::new_err("invalid axis (empty)")
    })?;
    let s_slice = matrix.as_slice().map_err(|_| {
        PyTypeError::new_err("matrix must be C-contiguous")
    })?;
    let mut m = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = s_slice[4 * r + c];
        }
    }
    let r = py
        .allow_threads(|| sb::mat4_apply_axis_rotation_keep_translation(&m, axis_char, angle_deg))
        .map_err(into_py_err)?;
    let mut buf = vec![0.0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            buf[4 * i + j] = r[i][j];
        }
    }
    let arr = ndarray::Array2::from_shape_vec((4, 4), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Return the user-facing violation messages corresponding to the
/// given boolean flags. Mirrors `FixedScene.get_violation_messages`.
#[pyfunction]
#[pyo3(signature = (
    has_self_intersection,
    has_contact_offset_violation,
    has_wall_violation,
    has_sphere_violation,
))]
pub(super) fn scene_violation_messages(
    has_self_intersection: bool,
    has_contact_offset_violation: bool,
    has_wall_violation: bool,
    has_sphere_violation: bool,
) -> Vec<String> {
    sb::violation_messages(
        has_self_intersection,
        has_contact_offset_violation,
        has_wall_violation,
        has_sphere_violation,
    )
}

/// `q1 * q2` (Hamilton product) on `(w, x, y, z)` quaternions. Returns
/// a `(4,)` numpy array. Mirrors `_quat_multiply`.
#[pyfunction]
#[pyo3(signature = (q1, q2))]
pub(super) fn scene_quat_multiply<'py>(
    py: Python<'py>,
    q1: [f64; 4],
    q2: [f64; 4],
) -> Bound<'py, numpy::PyArray1<f64>> {
    let r = sb::quat_multiply(q1, q2);
    PyArray1::<f64>::from_slice(py, &r)
}

/// Convert quaternion `(w, x, y, z)` to a row-major 3x3 numpy array.
/// Mirrors `_quat_to_mat3`.
#[pyfunction]
#[pyo3(signature = (q))]
pub(super) fn scene_quat_to_mat3<'py>(
    py: Python<'py>,
    q: [f64; 4],
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let m = sb::quat_to_mat3(q);
    let mut buf = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            buf[3 * i + j] = m[i][j];
        }
    }
    let arr = ndarray::Array2::from_shape_vec((3, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Build a unit quaternion from `axis` and `angle_deg`. Mirrors
/// `_axis_angle_to_quat`. Axis is normalized internally.
#[pyfunction]
#[pyo3(signature = (axis, angle_deg))]
pub(super) fn scene_axis_angle_to_quat<'py>(
    py: Python<'py>,
    axis: [f64; 3],
    angle_deg: f64,
) -> Bound<'py, numpy::PyArray1<f64>> {
    let q = sb::axis_angle_to_quat(axis, angle_deg);
    PyArray1::<f64>::from_slice(py, &q)
}

/// Convert a 3x3 rotation matrix to a unit quaternion `(w, x, y, z)`.
/// Mirrors `_mat3_to_quat`.
#[pyfunction]
#[pyo3(signature = (matrix))]
pub(super) fn scene_mat3_to_quat<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let s = matrix.shape();
    if s.len() != 2 || s[0] != 3 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "matrix must be (3, 3), got {s:?}"
        )));
    }
    let slice = matrix
        .as_slice()
        .map_err(|_| PyTypeError::new_err("matrix must be C-contiguous"))?;
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = slice[3 * i + j];
        }
    }
    let q = sb::mat3_to_quat(m);
    Ok(PyArray1::<f64>::from_slice(py, &q))
}

/// SLERP from `q0` to `q1` at parameter `t`. Mirrors `_quat_slerp`.
#[pyfunction]
#[pyo3(signature = (q0, q1, t))]
pub(super) fn scene_quat_slerp<'py>(
    py: Python<'py>,
    q0: [f64; 4],
    q1: [f64; 4],
    t: f64,
) -> Bound<'py, numpy::PyArray1<f64>> {
    let r = sb::quat_slerp(q0, q1, t);
    PyArray1::<f64>::from_slice(py, &r)
}

/// Apply a `T * R(quat) * S(scale)` transform to local vertices.
/// Returns a fresh `(N, 3)` numpy array. Mirrors
/// `_apply_transform_to_verts`.
#[pyfunction]
#[pyo3(signature = (local_vert, translation, quaternion, scale))]
pub(super) fn scene_apply_trs_to_verts<'py>(
    py: Python<'py>,
    local_vert: PyReadonlyArray2<'py, f64>,
    translation: [f64; 3],
    quaternion: [f64; 4],
    scale: [f64; 3],
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let s = local_vert.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "local_vert must be (N, 3), got {s:?}"
        )));
    }
    let v = local_vert
        .as_slice()
        .map_err(|_| PyTypeError::new_err("local_vert must be C-contiguous"))?;
    let n = s[0];
    let out = py.allow_threads(|| sb::apply_trs_to_verts(v, translation, quaternion, scale));
    let arr = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Decompose a row-major 4x4 transform `M = T * R * diag(scale)` into
/// `(translation, quaternion, scale)`, all as `(3,)` / `(4,)` numpy
/// arrays. Mirrors the matrix-decompose chunk inside
/// `Object._ensure_transform_animation`.
#[pyfunction]
#[pyo3(signature = (matrix))]
pub(super) fn scene_decompose_trs<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    let s = matrix.shape();
    if s.len() != 2 || s[0] != 4 || s[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "matrix must be (4, 4), got {s:?}"
        )));
    }
    let slice = matrix
        .as_slice()
        .map_err(|_| PyTypeError::new_err("matrix must be C-contiguous"))?;
    let mut m = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = slice[4 * i + j];
        }
    }
    let (t, q, sc) = sb::decompose_trs(&m);
    Ok((
        PyArray1::<f64>::from_slice(py, &t),
        PyArray1::<f64>::from_slice(py, &q),
        PyArray1::<f64>::from_slice(py, &sc),
    ))
}
