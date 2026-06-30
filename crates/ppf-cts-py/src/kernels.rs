// File: crates/ppf-cts-py/src/kernels.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 wrappers for the numeric kernels in ppf-cts-core. Each binding:
//   1. accepts numpy arrays from the Python caller,
//   2. extracts contiguous f64 / bool slices (zero-copy when possible),
//   3. releases the GIL via `py.allow_threads`,
//   4. dispatches to the pure-Rust kernel.
//
// The Python wrapper at frontend/_invisible_collider_.py owns the
// signature contract (Wall/Sphere object unpacking, normal
// normalization, pinned-set construction). These bindings are the raw
// numeric chokepoints; they do not parse Wall / Sphere objects.

use ppf_cts_core::kernels::bvh as bvh_k;
use ppf_cts_core::kernels::intersection as isect;
use ppf_cts_core::kernels::invisible_collider as ic;
use ppf_cts_core::kernels::proximity as prox;
use ppf_cts_core::kernels::rasterizer as raster_k;
use ppf_cts_core::kernels::sdf as sdf_k;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (vertices, is_pinned, wall_pos, wall_normal_unit))]
pub fn check_wall_violations_single(
    py: Python<'_>,
    vertices: PyReadonlyArray2<'_, f64>,
    is_pinned: PyReadonlyArray1<'_, bool>,
    wall_pos: PyReadonlyArray1<'_, f64>,
    wall_normal_unit: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<(usize, f64)>> {
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must have shape (N, 3), got {v_shape:?}"
        )));
    }
    let n_verts = v_shape[0];
    let v_slice = vertices.as_slice().map_err(|_| {
        PyTypeError::new_err(
            "vertices must be C-contiguous; pass np.ascontiguousarray(...)",
        )
    })?;
    let p_slice = is_pinned.as_slice().map_err(|_| {
        PyTypeError::new_err("is_pinned must be C-contiguous")
    })?;
    if p_slice.len() != n_verts {
        return Err(PyValueError::new_err(format!(
            "is_pinned length {} doesn't match vertex count {}",
            p_slice.len(),
            n_verts
        )));
    }
    let wp = crate::utils_py::vec3(&wall_pos, "wall_pos")?;
    let wn = crate::utils_py::vec3(&wall_normal_unit, "wall_normal_unit")?;

    let out = py.allow_threads(|| ic::check_wall_violations(v_slice, p_slice, wp, wn));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Rasterizer bindings

use numpy::PyReadwriteArray3 as PyRW3;

#[pyfunction]
#[pyo3(signature = (framebuffer, depth, screen_verts, colors, normals, faces, light_dir, ambient))]
pub fn rasterize_triangles(
    py: Python<'_>,
    mut framebuffer: PyRW3<'_, u8>,
    mut depth: numpy::PyReadwriteArray2<'_, f32>,
    screen_verts: PyReadonlyArray2<'_, f32>,
    colors: PyReadonlyArray2<'_, f32>,
    normals: PyReadonlyArray2<'_, f32>,
    faces: PyReadonlyArray2<'_, i32>,
    light_dir: PyReadonlyArray1<'_, f32>,
    ambient: f32,
) -> PyResult<()> {
    let fb_shape = framebuffer.shape();
    if fb_shape.len() != 3 || fb_shape[2] != 4 {
        return Err(PyValueError::new_err(format!(
            "framebuffer must be (H, W, 4), got {fb_shape:?}"
        )));
    }
    let height = fb_shape[0];
    let width = fb_shape[1];
    let dp_shape = depth.shape();
    if dp_shape != [height, width] {
        return Err(PyValueError::new_err(format!(
            "depth shape {dp_shape:?} mismatches framebuffer {height}x{width}"
        )));
    }
    let ld_slice = light_dir
        .as_slice()
        .map_err(|_| PyTypeError::new_err("light_dir must be C-contiguous"))?;
    if ld_slice.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "light_dir must have length 3, got {}",
            ld_slice.len()
        )));
    }
    let lf = [ld_slice[0], ld_slice[1], ld_slice[2]];

    let fb_slice = framebuffer.as_slice_mut().map_err(|_| {
        PyTypeError::new_err("framebuffer must be C-contiguous")
    })?;
    let dp_slice = depth
        .as_slice_mut()
        .map_err(|_| PyTypeError::new_err("depth must be C-contiguous"))?;
    let sv = screen_verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("screen_verts must be C-contiguous"))?;
    let col = colors
        .as_slice()
        .map_err(|_| PyTypeError::new_err("colors must be C-contiguous"))?;
    let nor = normals
        .as_slice()
        .map_err(|_| PyTypeError::new_err("normals must be C-contiguous"))?;
    let fa = faces
        .as_slice()
        .map_err(|_| PyTypeError::new_err("faces must be C-contiguous"))?;

    py.allow_threads(|| {
        raster_k::rasterize_triangles(
            fb_slice, dp_slice, width, height, sv, col, nor, fa, lf, ambient,
        );
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (framebuffer, depth, screen_verts, colors, segments, line_width))]
pub fn rasterize_lines(
    py: Python<'_>,
    mut framebuffer: PyRW3<'_, u8>,
    mut depth: numpy::PyReadwriteArray2<'_, f32>,
    screen_verts: PyReadonlyArray2<'_, f32>,
    colors: PyReadonlyArray2<'_, f32>,
    segments: PyReadonlyArray2<'_, i32>,
    line_width: i32,
) -> PyResult<()> {
    let fb_shape = framebuffer.shape();
    if fb_shape.len() != 3 || fb_shape[2] != 4 {
        return Err(PyValueError::new_err(format!(
            "framebuffer must be (H, W, 4), got {fb_shape:?}"
        )));
    }
    let height = fb_shape[0];
    let width = fb_shape[1];
    let dp_shape = depth.shape();
    if dp_shape != [height, width] {
        return Err(PyValueError::new_err(format!(
            "depth shape {dp_shape:?} mismatches framebuffer {height}x{width}"
        )));
    }

    let fb_slice = framebuffer
        .as_slice_mut()
        .map_err(|_| PyTypeError::new_err("framebuffer must be C-contiguous"))?;
    let dp_slice = depth
        .as_slice_mut()
        .map_err(|_| PyTypeError::new_err("depth must be C-contiguous"))?;
    let sv = screen_verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("screen_verts must be C-contiguous"))?;
    let col = colors
        .as_slice()
        .map_err(|_| PyTypeError::new_err("colors must be C-contiguous"))?;
    let seg = segments
        .as_slice()
        .map_err(|_| PyTypeError::new_err("segments must be C-contiguous"))?;

    py.allow_threads(|| {
        raster_k::rasterize_lines(
            fb_slice, dp_slice, width, height, sv, col, seg, line_width,
        );
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (vertices, faces))]
pub fn normals<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f32>,
    faces: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must be (N, 3), got {v_shape:?}"
        )));
    }
    let n_verts = v_shape[0];
    let v_slice = vertices
        .as_slice()
        .map_err(|_| PyTypeError::new_err("vertices must be C-contiguous"))?;
    let f_slice = faces
        .as_slice()
        .map_err(|_| PyTypeError::new_err("faces must be C-contiguous"))?;

    let normals = py.allow_threads(|| raster_k::normals(v_slice, f_slice));
    let arr = ndarray::Array2::from_shape_vec((n_verts, 3), normals)
        .map_err(|e| PyValueError::new_err(format!("normals reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// SDF bindings

fn sdf_inputs_check(
    sdf_types: &PyReadonlyArray1<'_, i32>,
    sdf_params: &PyReadonlyArray2<'_, f64>,
) -> PyResult<usize> {
    let n_sdfs = sdf_types.shape()[0];
    let p_shape = sdf_params.shape();
    if p_shape.len() != 2 || p_shape[1] != 8 {
        return Err(PyValueError::new_err(format!(
            "sdf_params must have shape (N, 8), got {p_shape:?}"
        )));
    }
    if p_shape[0] != n_sdfs {
        return Err(PyValueError::new_err(format!(
            "sdf_types length {} doesn't match sdf_params rows {}",
            n_sdfs, p_shape[0]
        )));
    }
    Ok(n_sdfs)
}

#[pyfunction]
#[pyo3(signature = (xs, ys, zs, sdf_types, sdf_params))]
pub fn eval_sdf_grid<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    zs: PyReadonlyArray1<'py, f64>,
    sdf_types: PyReadonlyArray1<'py, i32>,
    sdf_params: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
    sdf_inputs_check(&sdf_types, &sdf_params)?;
    let xs_s = xs.as_slice().map_err(|_| PyTypeError::new_err("xs must be C-contiguous"))?;
    let ys_s = ys.as_slice().map_err(|_| PyTypeError::new_err("ys must be C-contiguous"))?;
    let zs_s = zs.as_slice().map_err(|_| PyTypeError::new_err("zs must be C-contiguous"))?;
    let t_s = sdf_types
        .as_slice()
        .map_err(|_| PyTypeError::new_err("sdf_types must be C-contiguous"))?;
    let p_s = sdf_params
        .as_slice()
        .map_err(|_| PyTypeError::new_err("sdf_params must be C-contiguous"))?;
    let nx = xs_s.len();
    let ny = ys_s.len();
    let nz = zs_s.len();

    let flat = py.allow_threads(|| sdf_k::eval_sdf_grid(xs_s, ys_s, zs_s, t_s, p_s));
    let arr = ndarray::Array3::from_shape_vec((nx, ny, nz), flat)
        .map_err(|e| PyValueError::new_err(format!("grid reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (xs, ys, zs, step, sdf_types, sdf_params))]
pub fn marching_cubes<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    zs: PyReadonlyArray1<'py, f64>,
    step: f64,
    sdf_types: PyReadonlyArray1<'py, i32>,
    sdf_params: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i32>>)> {
    sdf_inputs_check(&sdf_types, &sdf_params)?;
    let xs_s = xs.as_slice().map_err(|_| PyTypeError::new_err("xs must be C-contiguous"))?;
    let ys_s = ys.as_slice().map_err(|_| PyTypeError::new_err("ys must be C-contiguous"))?;
    let zs_s = zs.as_slice().map_err(|_| PyTypeError::new_err("zs must be C-contiguous"))?;
    let t_s = sdf_types
        .as_slice()
        .map_err(|_| PyTypeError::new_err("sdf_types must be C-contiguous"))?;
    let p_s = sdf_params
        .as_slice()
        .map_err(|_| PyTypeError::new_err("sdf_params must be C-contiguous"))?;

    let (verts_flat, faces_flat) =
        py.allow_threads(|| sdf_k::marching_cubes(xs_s, ys_s, zs_s, step, t_s, p_s));
    let n_verts = verts_flat.len() / 3;
    let n_faces = faces_flat.len() / 3;
    let verts_arr = ndarray::Array2::from_shape_vec((n_verts, 3), verts_flat)
        .map_err(|e| PyValueError::new_err(format!("vert reshape failed: {e}")))?;
    let faces_arr = ndarray::Array2::from_shape_vec((n_faces, 3), faces_flat)
        .map_err(|e| PyValueError::new_err(format!("face reshape failed: {e}")))?;
    Ok((verts_arr.into_pyarray(py), faces_arr.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Self-intersection binding

#[pyfunction]
#[pyo3(signature = (vertices, tris, is_collider=None, rod_edges=None))]
pub fn check_self_intersection(
    py: Python<'_>,
    vertices: PyReadonlyArray2<'_, f64>,
    tris: PyReadonlyArray2<'_, i32>,
    is_collider: Option<PyReadonlyArray1<'_, bool>>,
    rod_edges: Option<PyReadonlyArray2<'_, i32>>,
) -> PyResult<Vec<(i32, i32)>> {
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must have shape (N, 3), got {v_shape:?}"
        )));
    }
    let v_slice = vertices
        .as_slice()
        .map_err(|_| PyTypeError::new_err("vertices must be C-contiguous"))?;

    let t_shape = tris.shape();
    if t_shape.len() != 2 || t_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "tris must have shape (M, 3), got {t_shape:?}"
        )));
    }
    let t_slice = tris
        .as_slice()
        .map_err(|_| PyTypeError::new_err("tris must be C-contiguous"))?;
    let n_tris = t_shape[0];

    let coll_borrow;
    let coll: Option<&[bool]> = if let Some(c) = is_collider.as_ref() {
        let s = c
            .as_slice()
            .map_err(|_| PyTypeError::new_err("is_collider must be C-contiguous"))?;
        if s.len() != n_tris {
            return Err(PyValueError::new_err(format!(
                "is_collider length {} doesn't match n_tris = {n_tris}",
                s.len()
            )));
        }
        coll_borrow = s;
        Some(coll_borrow)
    } else {
        None
    };

    let rod_borrow;
    let rod: Option<&[i32]> = if let Some(r) = rod_edges.as_ref() {
        let s = r.shape();
        if s.len() != 2 || s[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "rod_edges must have shape (K, 2), got {s:?}"
            )));
        }
        rod_borrow = r
            .as_slice()
            .map_err(|_| PyTypeError::new_err("rod_edges must be C-contiguous"))?;
        Some(rod_borrow)
    } else {
        None
    };

    let pairs = py.allow_threads(|| {
        isect::check_self_intersection(isect::IntersectionInput {
            verts: v_slice,
            tris: t_slice,
            is_collider: coll,
            rod_edges: rod,
            // Standalone utility binding (tests / _intersection_.py); the PDRD
            // body filter is applied only in the scene-build assemble path.
            tri_body_id: None,
        })
    });
    Ok(pairs)
}

// ---------------------------------------------------------------------------
// Proximity binding

#[pyfunction]
#[pyo3(signature = (vertices, tris=None, edges=None, is_collider=None, contact_offset=None))]
pub fn check_contact_offset_violation(
    py: Python<'_>,
    vertices: PyReadonlyArray2<'_, f64>,
    tris: Option<PyReadonlyArray2<'_, i32>>,
    edges: Option<PyReadonlyArray2<'_, i32>>,
    is_collider: Option<PyReadonlyArray1<'_, bool>>,
    contact_offset: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Vec<(i32, i32)>> {
    // Vertex shape check.
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must have shape (N, 3), got {v_shape:?}"
        )));
    }
    let v_slice = vertices
        .as_slice()
        .map_err(|_| PyTypeError::new_err("vertices must be C-contiguous"))?;

    // Optional tris (M, 3).
    let mut tris_owned: Option<&[i32]> = None;
    let tri_borrow;
    if let Some(t) = tris.as_ref() {
        let s = t.shape();
        if s.len() != 2 || s[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "tris must have shape (M, 3), got {s:?}"
            )));
        }
        tri_borrow = t
            .as_slice()
            .map_err(|_| PyTypeError::new_err("tris must be C-contiguous"))?;
        tris_owned = Some(tri_borrow);
    }

    // Optional edges (K, 2).
    let mut edges_owned: Option<&[i32]> = None;
    let edge_borrow;
    if let Some(e) = edges.as_ref() {
        let s = e.shape();
        if s.len() != 2 || s[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "edges must have shape (K, 2), got {s:?}"
            )));
        }
        edge_borrow = e
            .as_slice()
            .map_err(|_| PyTypeError::new_err("edges must be C-contiguous"))?;
        edges_owned = Some(edge_borrow);
    }

    // Optional is_collider / contact_offset, length n_tris + n_edges.
    let n_tris = tris_owned.map(|t| t.len() / 3).unwrap_or(0);
    let n_edges = edges_owned.map(|e| e.len() / 2).unwrap_or(0);
    let n_elems = n_tris + n_edges;

    let coll_borrow;
    let coll: Option<&[bool]> = if let Some(c) = is_collider.as_ref() {
        let s = c
            .as_slice()
            .map_err(|_| PyTypeError::new_err("is_collider must be C-contiguous"))?;
        if s.len() != n_elems {
            return Err(PyValueError::new_err(format!(
                "is_collider length {} doesn't match n_tris + n_edges = {n_elems}",
                s.len()
            )));
        }
        coll_borrow = s;
        Some(coll_borrow)
    } else {
        None
    };

    let off_borrow;
    let off: Option<&[f64]> = if let Some(o) = contact_offset.as_ref() {
        let s = o
            .as_slice()
            .map_err(|_| PyTypeError::new_err("contact_offset must be C-contiguous"))?;
        if s.len() != n_elems {
            return Err(PyValueError::new_err(format!(
                "contact_offset length {} doesn't match n_tris + n_edges = {n_elems}",
                s.len()
            )));
        }
        off_borrow = s;
        Some(off_borrow)
    } else {
        None
    };

    let pairs = py.allow_threads(|| {
        prox::check_contact_offset_violation(prox::ProximityInput {
            verts: v_slice,
            tris: tris_owned,
            edges: edges_owned,
            is_collider: coll,
            contact_offset: off,
        })
    });
    Ok(pairs)
}

// ---------------------------------------------------------------------------
// BVH frame-mapping bindings

use crate::utils_py::require_n_by_k;

#[pyfunction]
#[pyo3(signature = (orig_vert, new_vert, new_tri))]
pub fn frame_mapping<'py>(
    py: Python<'py>,
    orig_vert: PyReadonlyArray2<'py, f64>,
    new_vert: PyReadonlyArray2<'py, f64>,
    new_tri: PyReadonlyArray2<'py, i32>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray2<f64>>)> {
    let n_orig = require_n_by_k::<_, 3>(&orig_vert, "orig_vert")?;
    require_n_by_k::<_, 3>(&new_vert, "new_vert")?;
    require_n_by_k::<_, 3>(&new_tri, "new_tri")?;

    let orig = orig_vert
        .as_slice()
        .map_err(|_| PyTypeError::new_err("orig_vert must be C-contiguous"))?;
    let nv = new_vert
        .as_slice()
        .map_err(|_| PyTypeError::new_err("new_vert must be C-contiguous"))?;
    let nt = new_tri
        .as_slice()
        .map_err(|_| PyTypeError::new_err("new_tri must be C-contiguous"))?;

    let (tri_indices, coefs_flat) =
        py.allow_threads(|| bvh_k::frame_mapping(orig, nv, nt));

    debug_assert_eq!(tri_indices.len(), n_orig);
    debug_assert_eq!(coefs_flat.len(), n_orig * 3);

    let tri_idx_py = tri_indices.into_pyarray(py);
    // Reshape flat (n_orig * 3) into (n_orig, 3) for the caller.
    let coefs_arr = ndarray::Array2::from_shape_vec((n_orig, 3), coefs_flat)
        .map_err(|e| PyValueError::new_err(format!("coef reshape failed: {e}")))?;
    let coefs_py = coefs_arr.into_pyarray(py);
    Ok((tri_idx_py, coefs_py))
}

#[pyfunction]
#[pyo3(signature = (deformed_vert, surf_tri, tri_indices, coefs))]
pub fn interpolate_surface<'py>(
    py: Python<'py>,
    deformed_vert: PyReadonlyArray2<'py, f64>,
    surf_tri: PyReadonlyArray2<'py, i32>,
    tri_indices: PyReadonlyArray1<'py, i32>,
    coefs: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_n_by_k::<_, 3>(&deformed_vert, "deformed_vert")?;
    require_n_by_k::<_, 3>(&surf_tri, "surf_tri")?;
    let n_pts = require_n_by_k::<_, 3>(&coefs, "coefs")?;
    if tri_indices.shape() != [n_pts] {
        return Err(PyValueError::new_err(format!(
            "tri_indices length {} doesn't match coefs row count {n_pts}",
            tri_indices.shape()[0]
        )));
    }

    let dv = deformed_vert
        .as_slice()
        .map_err(|_| PyTypeError::new_err("deformed_vert must be C-contiguous"))?;
    let st = surf_tri
        .as_slice()
        .map_err(|_| PyTypeError::new_err("surf_tri must be C-contiguous"))?;
    let ti = tri_indices
        .as_slice()
        .map_err(|_| PyTypeError::new_err("tri_indices must be C-contiguous"))?;
    let cf = coefs
        .as_slice()
        .map_err(|_| PyTypeError::new_err("coefs must be C-contiguous"))?;

    let out_flat = py.allow_threads(|| bvh_k::interpolate_surface(dv, st, ti, cf));
    let out_arr = ndarray::Array2::from_shape_vec((n_pts, 3), out_flat)
        .map_err(|e| PyValueError::new_err(format!("output reshape failed: {e}")))?;
    Ok(out_arr.into_pyarray(py))
}

