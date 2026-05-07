// File: crates/ppf-cts-py/src/mesh_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for the pure-math mesh helpers in
// `ppf-cts-core::datamodel::mesh`. These wrap the inner-loop kernels
// the Python frontend dispatches into from `frontend/_mesh_.py`.
//
// Function index:
//   * mesh_bbox: per-axis (width, height, depth) extents.
//   * mesh_normalize_verts: center on origin and scale to fit
//     `[-0.5, 0.5]` along the longest axis.
//   * mesh_scale_per_axis: in-place per-axis scale around the centroid.
//   * mesh_generate_rect_faces: alternating-diagonal rect grid faces.
//   * mesh_generate_grid_faces: wrapped grid faces (Mobius).
//   * mesh_generate_cylinder_verts: cylinder vertices around the x-axis.
//   * mesh_generate_cylinder_faces: cylinder faces around the x-axis.
//   * mesh_transform_verts_2d: apply 3D basis to 2D grid vertices.
//   * mesh_mobius: full Mobius strip generator.
//   * mesh_icosphere: icosphere with subdivision.
//   * mesh_fix_skinny_triangles: union-find merge of skinny edges.
//   * mesh_polygon_area_2d: shoelace area.
//   * mesh_tet_extract_surface: post-fTetWild filter, surface
//     extraction, winding fix, and reindexing.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use ppf_cts_core::datamodel::mesh as mesh_core;

// ---------------------------------------------------------------------------
// Helpers.

fn read_verts_array2_f64(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<ndarray::Array2<f64>> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "verts must have shape (N, 3), got {shape:?}"
        )));
    }
    let s = arr.as_slice().map_err(|_| {
        PyTypeError::new_err("verts must be C-contiguous; call np.ascontiguousarray(...)")
    })?;
    ndarray::Array2::from_shape_vec((shape[0], 3), s.to_vec())
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))
}

/// Read `(N, 2)` f64 polygon points.
fn read_pts_array2_f64(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<ndarray::Array2<f64>> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "points must have shape (N, 2), got {shape:?}"
        )));
    }
    let s = arr.as_slice().map_err(|_| {
        PyTypeError::new_err("points must be C-contiguous; call np.ascontiguousarray(...)")
    })?;
    ndarray::Array2::from_shape_vec((shape[0], 2), s.to_vec())
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))
}

/// Try to read a `(N, M)` indices array. Accepts dtype int32, int64,
/// uint32, or uint64. Validates `ncols == m`.
fn read_indices_u32(arr: &Bound<'_, PyAny>, m: usize, name: &str) -> PyResult<ndarray::Array2<u32>> {
    fn check_shape(shape: &[usize], m: usize, name: &str) -> PyResult<()> {
        if shape.len() != 2 || shape[1] != m {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (N, {m}), got {shape:?}"
            )));
        }
        Ok(())
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, u32>>() {
        let shape = view.shape();
        check_shape(shape, m, name)?;
        let slice = view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?;
        return ndarray::Array2::from_shape_vec((shape[0], m), slice.to_vec())
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")));
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i32>>() {
        let shape = view.shape();
        check_shape(shape, m, name)?;
        let slice = view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?;
        let buf: Vec<u32> = slice.iter().map(|&v| v as u32).collect();
        return ndarray::Array2::from_shape_vec((shape[0], m), buf)
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")));
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i64>>() {
        let shape = view.shape();
        check_shape(shape, m, name)?;
        let slice = view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?;
        let buf: Vec<u32> = slice.iter().map(|&v| v as u32).collect();
        return ndarray::Array2::from_shape_vec((shape[0], m), buf)
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")));
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, u64>>() {
        let shape = view.shape();
        check_shape(shape, m, name)?;
        let slice = view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?;
        let buf: Vec<u32> = slice.iter().map(|&v| v as u32).collect();
        return ndarray::Array2::from_shape_vec((shape[0], m), buf)
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")));
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be int32/int64/uint32/uint64 ndarray of shape (N, {m})"
    )))
}

fn vec3_arg(arr: numpy::PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<[f64; 3]> {
    let s = arr
        .as_slice()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?;
    if s.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have length 3, got {}",
            s.len()
        )));
    }
    Ok([s[0], s[1], s[2]])
}

// ---------------------------------------------------------------------------
// Bindings.

/// Per-axis bbox extents (width, height, depth) of a vertex array.
#[pyfunction]
#[pyo3(signature = (verts))]
pub fn mesh_bbox<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v = read_verts_array2_f64(&verts)?;
    let bb = py.allow_threads(|| mesh_core::bbox(&v));
    let extents = if v.nrows() == 0 {
        vec![0.0_f64; 3]
    } else {
        vec![
            bb[[1, 0]] - bb[[0, 0]],
            bb[[1, 1]] - bb[[0, 1]],
            bb[[1, 2]] - bb[[0, 2]],
        ]
    };
    Ok(numpy::PyArray1::from_vec(py, extents))
}

/// Centered + uniformly scaled copy of `verts` so the longest axis fits
/// in `[-0.5, 0.5]`. Returns a new ndarray; does not mutate input.
#[pyfunction]
#[pyo3(signature = (verts))]
pub fn mesh_normalize_verts<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut v = read_verts_array2_f64(&verts)?;
    py.allow_threads(|| {
        // The Python helper divides by `max(bbox(vert))` after centering;
        // mesh_core::normalize matches that exactly (longest axis -> 1.0).
        if v.nrows() > 0 {
            let mean = {
                let mut m = [0.0_f64; 3];
                for r in 0..v.nrows() {
                    m[0] += v[[r, 0]];
                    m[1] += v[[r, 1]];
                    m[2] += v[[r, 2]];
                }
                let inv = 1.0 / v.nrows() as f64;
                [m[0] * inv, m[1] * inv, m[2] * inv]
            };
            for r in 0..v.nrows() {
                v[[r, 0]] -= mean[0];
                v[[r, 1]] -= mean[1];
                v[[r, 2]] -= mean[2];
            }
            let bb = mesh_core::bbox(&v);
            let extents = [
                bb[[1, 0]] - bb[[0, 0]],
                bb[[1, 1]] - bb[[0, 1]],
                bb[[1, 2]] - bb[[0, 2]],
            ];
            let max_extent = extents[0].max(extents[1]).max(extents[2]);
            if max_extent > 0.0 {
                let inv = 1.0 / max_extent;
                for r in 0..v.nrows() {
                    v[[r, 0]] *= inv;
                    v[[r, 1]] *= inv;
                    v[[r, 2]] *= inv;
                }
            }
        }
    });
    Ok(v.into_pyarray(py))
}

/// Per-axis scale around centroid. Returns a new ndarray; does not
/// mutate the input.
#[pyfunction]
#[pyo3(signature = (verts, sx, sy, sz))]
pub fn mesh_scale_per_axis<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    sx: f64,
    sy: f64,
    sz: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut v = read_verts_array2_f64(&verts)?;
    py.allow_threads(|| mesh_core::scale_per_axis(&mut v, sx, sy, sz));
    Ok(v.into_pyarray(py))
}

/// Alternating-diagonal grid faces, returned as int32 (N, 3).
#[pyfunction]
#[pyo3(signature = (res_x, res_y))]
pub fn mesh_generate_rect_faces<'py>(
    py: Python<'py>,
    res_x: usize,
    res_y: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    if res_x < 2 || res_y < 2 {
        return Err(PyValueError::new_err(format!(
            "res_x and res_y must be >= 2, got ({res_x}, {res_y})"
        )));
    }
    let f = py.allow_threads(|| mesh_core::generate_rect_faces(res_x, res_y));
    let n = f.nrows();
    let buf: Vec<i32> = f.iter().map(|&v| v as i32).collect();
    let arr = ndarray::Array2::from_shape_vec((n, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Wrapped grid faces (Mobius), returned as int32 (N, 3).
#[pyfunction]
#[pyo3(signature = (length_split, width_split))]
pub fn mesh_generate_grid_faces<'py>(
    py: Python<'py>,
    length_split: usize,
    width_split: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    if length_split < 2 || width_split < 2 {
        return Err(PyValueError::new_err(format!(
            "length_split and width_split must be >= 2, got ({length_split}, {width_split})"
        )));
    }
    let f = py.allow_threads(|| mesh_core::generate_grid_faces(length_split, width_split));
    let n = f.nrows();
    let buf: Vec<i32> = f.iter().map(|&v| v as i32).collect();
    let arr = ndarray::Array2::from_shape_vec((n, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Cylinder vertex array, shape `((n + 1) * ny, 3)`.
#[pyfunction]
#[pyo3(signature = (n, ny, min_x, dx, dy, r))]
pub fn mesh_generate_cylinder_verts<'py>(
    py: Python<'py>,
    n: usize,
    ny: usize,
    min_x: f64,
    dx: f64,
    dy: f64,
    r: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let v = py.allow_threads(|| mesh_core::generate_cylinder_verts(n, ny, min_x, dx, dy, r));
    Ok(v.into_pyarray(py))
}

/// Cylinder face array, shape `(2 * n * ny, 3)`, dtype int32.
#[pyfunction]
#[pyo3(signature = (n, ny))]
pub fn mesh_generate_cylinder_faces<'py>(
    py: Python<'py>,
    n: usize,
    ny: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let f = py.allow_threads(|| mesh_core::generate_cylinder_faces(n, ny));
    let nrows = f.nrows();
    let buf: Vec<i32> = f.iter().map(|&v| v as i32).collect();
    let arr = ndarray::Array2::from_shape_vec((nrows, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Apply 3D basis vectors `ex`, `ey` to 2D grid vertices. Output
/// shape `(N, 3)`.
#[pyfunction]
#[pyo3(signature = (verts, ex, ey))]
pub fn mesh_transform_verts_2d<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    ex: numpy::PyReadonlyArray1<'py, f64>,
    ey: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Accept (N, 2) or (N, 3) input; both forms appear in callers.
    let v_shape = verts.shape();
    if v_shape.len() != 2 || (v_shape[1] != 2 && v_shape[1] != 3) {
        return Err(PyValueError::new_err(format!(
            "verts must have shape (N, 2) or (N, 3), got {v_shape:?}"
        )));
    }
    let s = verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("verts must be C-contiguous"))?;
    // Reshape into a `(N, 3)` array zero-padding the third column when
    // the input is 2D.
    let n = v_shape[0];
    let mut v = ndarray::Array2::<f64>::zeros((n, 3));
    if v_shape[1] == 2 {
        for i in 0..n {
            v[[i, 0]] = s[2 * i];
            v[[i, 1]] = s[2 * i + 1];
        }
    } else {
        for i in 0..n {
            v[[i, 0]] = s[3 * i];
            v[[i, 1]] = s[3 * i + 1];
            v[[i, 2]] = s[3 * i + 2];
        }
    }
    let ex_v = vec3_arg(ex, "ex")?;
    let ey_v = vec3_arg(ey, "ey")?;
    let out = py.allow_threads(|| mesh_core::transform_verts_2d(&v, ex_v, ey_v));
    Ok(out.into_pyarray(py))
}

/// Mobius strip generator. Returns `(verts (N, 3), faces (M, 3) int32)`.
#[pyfunction]
#[pyo3(signature = (length_split, width_split, twists, r, flatness, width, scale))]
pub fn mesh_mobius<'py>(
    py: Python<'py>,
    length_split: usize,
    width_split: usize,
    twists: i32,
    r: f64,
    flatness: f64,
    width: f64,
    scale: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    if length_split < 2 || width_split < 2 {
        return Err(PyValueError::new_err(format!(
            "length_split and width_split must be >= 2, got ({length_split}, {width_split})"
        )));
    }
    let (v, f) = py.allow_threads(|| {
        mesh_core::mobius(length_split, width_split, twists, r, flatness, width, scale)
    });
    let n_faces = f.nrows();
    let buf: Vec<i32> = f.iter().map(|&v| v as i32).collect();
    let f_arr = ndarray::Array2::from_shape_vec((n_faces, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    let v_py = v.into_pyarray(py);
    let f_py = f_arr.into_pyarray(py);
    Ok(PyTuple::new(py, &[v_py.into_any(), f_py.into_any()])?)
}

/// Icosphere with `subdiv_count` Loop-style midpoint subdivisions of
/// the icosahedron, projected onto a sphere of radius `r`.
#[pyfunction]
#[pyo3(signature = (r, subdiv_count))]
pub fn mesh_icosphere<'py>(
    py: Python<'py>,
    r: f64,
    subdiv_count: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let (v, f) = py.allow_threads(|| mesh_core::icosphere(r, subdiv_count));
    let n_faces = f.nrows();
    let buf: Vec<i32> = f.iter().map(|&v| v as i32).collect();
    let f_arr = ndarray::Array2::from_shape_vec((n_faces, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    let v_py = v.into_pyarray(py);
    let f_py = f_arr.into_pyarray(py);
    Ok(PyTuple::new(py, &[v_py.into_any(), f_py.into_any()])?)
}

/// Iteratively merge vertices opposite very small triangle angles.
/// Returns `(verts (N, 3) f64, faces (M, 3) int32)`.
#[pyfunction]
#[pyo3(signature = (verts, faces, min_angle_deg = 1.0))]
pub fn mesh_fix_skinny_triangles<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    faces: &Bound<'py, PyAny>,
    min_angle_deg: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let v = read_verts_array2_f64(&verts)?;
    let f = read_indices_u32(faces, 3, "faces")?;
    let (v_out, f_out) = py.allow_threads(|| mesh_core::fix_skinny_triangles(&v, &f, min_angle_deg));
    let n_faces = f_out.nrows();
    let buf: Vec<i32> = f_out.iter().map(|&v| v as i32).collect();
    let f_arr = ndarray::Array2::from_shape_vec((n_faces, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    let v_py = v_out.into_pyarray(py);
    let f_py = f_arr.into_pyarray(py);
    Ok(PyTuple::new(py, &[v_py.into_any(), f_py.into_any()])?)
}

/// Shoelace area of a closed 2D polygon `(N, 2)`.
#[pyfunction]
#[pyo3(signature = (pts))]
pub fn mesh_polygon_area_2d<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    let p = read_pts_array2_f64(&pts)?;
    Ok(py.allow_threads(|| mesh_core::polygon_area_2d(&p)))
}

/// Post-fTetWild tet-mesh cleanup. Filters tets with `|V| <
/// min_volume`, extracts boundary surface, fixes winding, and reindexes
/// vertices. Returns `(verts (N, 3) f64, surface_tri (M, 3) int32,
/// tet (K, 4) int32)`.
#[pyfunction]
#[pyo3(signature = (verts, tet, min_volume = 1e-15))]
pub fn mesh_tet_extract_surface<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    tet: &Bound<'py, PyAny>,
    min_volume: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let v = read_verts_array2_f64(&verts)?;
    let t = read_indices_u32(tet, 4, "tet")?;
    let out = py.allow_threads(|| mesh_core::tet_extract_surface(&v, &t, min_volume));
    let n_tri = out.tri.nrows();
    let n_tet = out.tet.nrows();
    let tri_buf: Vec<i32> = out.tri.iter().map(|&v| v as i32).collect();
    let tet_buf: Vec<i32> = out.tet.iter().map(|&v| v as i32).collect();
    let tri_arr = ndarray::Array2::from_shape_vec((n_tri, 3), tri_buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    let tet_arr = ndarray::Array2::from_shape_vec((n_tet, 4), tet_buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    let v_py = out.verts.into_pyarray(py);
    let tri_py = tri_arr.into_pyarray(py);
    let tet_py = tet_arr.into_pyarray(py);
    Ok(PyTuple::new(
        py,
        &[v_py.into_any(), tri_py.into_any(), tet_py.into_any()],
    )?)
}

// ---------------------------------------------------------------------------
// Additional `_mesh_` helpers.

/// Closed 2D polygon points around the origin, shape `(n, 2)`.
#[pyfunction]
#[pyo3(signature = (n, r))]
pub fn mesh_circle_points_2d<'py>(
    py: Python<'py>,
    n: usize,
    r: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pts = py.allow_threads(|| mesh_core::circle_points_2d(n, r));
    Ok(pts.into_pyarray(py))
}

/// Closed line-loop edges over `n` vertices, shape `(n, 2)` int32.
#[pyfunction]
#[pyo3(signature = (n))]
pub fn mesh_tri_edge_loop<'py>(
    py: Python<'py>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let e = py.allow_threads(|| mesh_core::tri_edge_loop(n));
    let buf: Vec<i32> = e.iter().map(|&v| v as i32).collect();
    let arr = ndarray::Array2::from_shape_vec((n, 2), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Cylinder generator parameters: `(dx, ny, dy)`.
#[pyfunction]
#[pyo3(signature = (min_x, max_x, n, r))]
pub fn mesh_cylinder_dx_ny_dy(
    min_x: f64,
    max_x: f64,
    n: usize,
    r: f64,
) -> (f64, usize, f64) {
    mesh_core::cylinder_dx_ny_dy(min_x, max_x, n, r)
}

/// Build a rectangle mesh in the plane spanned by `ex`, `ey`. Returns
/// `(verts, faces)` where verts is `(N, 5)` (with UVs) or `(N, 3)`.
#[pyfunction]
#[pyo3(signature = (res_x, width, height, ex, ey, gen_uv = true))]
pub fn mesh_rectangle_with_uv<'py>(
    py: Python<'py>,
    res_x: usize,
    width: f64,
    height: f64,
    ex: numpy::PyReadonlyArray1<'py, f64>,
    ey: numpy::PyReadonlyArray1<'py, f64>,
    gen_uv: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    if res_x < 2 {
        return Err(PyValueError::new_err(format!(
            "res_x must be >= 2, got {res_x}"
        )));
    }
    let ex_v = vec3_arg(ex, "ex")?;
    let ey_v = vec3_arg(ey, "ey")?;
    let (verts, faces) = py.allow_threads(|| {
        mesh_core::rectangle_with_uv(res_x, width, height, ex_v, ey_v, gen_uv)
    });
    let n_faces = faces.nrows();
    let buf: Vec<i32> = faces.iter().map(|&v| v as i32).collect();
    let f_arr = ndarray::Array2::from_shape_vec((n_faces, 3), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    let v_py = verts.into_pyarray(py);
    let f_py = f_arr.into_pyarray(py);
    Ok(PyTuple::new(py, &[v_py.into_any(), f_py.into_any()])?)
}

/// Resolve a preset's `(filename_stem, url)` pair, or raise
/// `ValueError` for an unknown name.
#[pyfunction]
#[pyo3(signature = (name))]
pub fn mesh_preset_lookup(name: &str) -> PyResult<(String, String)> {
    if let Some(stem) = mesh_core::preset_filename_stem(name) {
        Ok((stem.to_string(), mesh_core::preset_url(stem)))
    } else {
        let names = mesh_core::preset_names();
        Err(PyValueError::new_err(format!(
            "Unknown preset: {name}. Available: {names:?}"
        )))
    }
}

/// Compute the `.npz` cache path for a given mesh hash + tag.
#[pyfunction]
#[pyo3(signature = (cache_dir, hash, name))]
pub fn mesh_cache_path(cache_dir: &str, hash: &str, name: &str) -> String {
    mesh_core::cache_path(cache_dir, hash, name)
        .to_string_lossy()
        .into_owned()
}

/// Format the triangulate `pa{area}q{min_angle}` argument string.
#[pyfunction]
#[pyo3(signature = (area, min_angle))]
pub fn mesh_format_triangulate_args(area: f64, min_angle: f64) -> String {
    mesh_core::format_triangulate_args(area, min_angle)
}

/// Build the `tetrahedralize` cache filename component from string-
/// stringified positional + keyword args.
#[pyfunction]
#[pyo3(signature = (args, kwargs))]
pub fn mesh_tetrahedralize_arg_str(
    args: Vec<String>,
    kwargs: Vec<(String, String)>,
) -> String {
    mesh_core::tetrahedralize_arg_str(&args, &kwargs)
}

/// Filter & default fTetWild kwargs. Input is a list of
/// `(key, value_repr_str)`. Output preserves input order and appends
/// defaults at the end if absent.
#[pyfunction]
#[pyo3(signature = (kwargs))]
pub fn mesh_ftetwild_kwargs(
    kwargs: Vec<(String, String)>,
) -> Vec<(String, String)> {
    mesh_core::ftetwild_kwargs(&kwargs)
}

/// Register all mesh helpers with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mesh_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_normalize_verts, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_scale_per_axis, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_generate_rect_faces, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_generate_grid_faces, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_generate_cylinder_verts, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_generate_cylinder_faces, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_transform_verts_2d, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_mobius, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_icosphere, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_fix_skinny_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_polygon_area_2d, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_tet_extract_surface, m)?)?;
    // Additional helpers.
    m.add_function(wrap_pyfunction!(mesh_circle_points_2d, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_tri_edge_loop, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_cylinder_dx_ny_dy, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_rectangle_with_uv, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_preset_lookup, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_cache_path, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_format_triangulate_args, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_tetrahedralize_arg_str, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_ftetwild_kwargs, m)?)?;
    Ok(())
}
