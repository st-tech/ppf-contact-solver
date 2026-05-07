// File: crates/ppf-cts-py/src/render_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for the pure-compute pieces of `frontend/_render_.py`
// and `frontend/_rasterizer_.py` (the `SoftwareRenderer.render`
// transform pipeline + `MitsubaRenderer._export_ply` PLY writer).
// Mitsuba scene-dict construction stays in Python because it has to
// import `mitsuba`; only the math + binary I/O ride into Rust.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ppf_cts_core::kernels::invisible_collider as ic;
use ppf_cts_core::kernels::rasterizer as raster;

/// Build a 4x4 orthographic projection matrix as a numpy `(4, 4)`
/// float32 array. Mirrors `_ortho_matrix` in `frontend/_rasterizer_.py`.
#[pyfunction]
pub fn ortho_matrix<'py>(
    py: Python<'py>,
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Bound<'py, PyArray2<f32>> {
    let m = raster::ortho_matrix(left, right, bottom, top, near, far);
    let arr = PyArray2::<f32>::zeros(py, (4, 4), false);
    {
        let mut view = unsafe { arr.as_array_mut() };
        for r in 0..4 {
            for c in 0..4 {
                view[(r, c)] = m[r * 4 + c];
            }
        }
    }
    arr
}

/// Run the centering + 10-degree-tilt + scale-to-fit + orthographic
/// projection + screen-space pipeline. Returns `(rotated_verts,
/// screen_verts)`:
///   * `rotated_verts` is `(N, 3)` float32 (the post-tilt mesh-space
///     coordinates that `normals` consumes).
///   * `screen_verts` is `(N, 4)` float32 (`[sx, sy, z, 1]`).
/// Mirrors lines 84-126 of `_rasterizer_.py`.
#[pyfunction]
pub fn render_transform<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f32>,
    width: u32,
    height: u32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let v = verts.as_array();
    let n = v.shape()[0];
    if v.shape()[1] != 3 {
        return Err(PyValueError::new_err("verts must have shape (N, 3)"));
    }
    let mut flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat.push(v[(i, 0)]);
        flat.push(v[(i, 1)]);
        flat.push(v[(i, 2)]);
    }
    let (rotated, screen) = raster::render_transform(&flat, width, height);

    let rot_arr = PyArray2::<f32>::zeros(py, (n, 3), false);
    {
        let mut view = unsafe { rot_arr.as_array_mut() };
        for i in 0..n {
            view[(i, 0)] = rotated[3 * i];
            view[(i, 1)] = rotated[3 * i + 1];
            view[(i, 2)] = rotated[3 * i + 2];
        }
    }
    let scr_arr = PyArray2::<f32>::zeros(py, (n, 4), false);
    {
        let mut view = unsafe { scr_arr.as_array_mut() };
        for i in 0..n {
            view[(i, 0)] = screen[4 * i];
            view[(i, 1)] = screen[4 * i + 1];
            view[(i, 2)] = screen[4 * i + 2];
            view[(i, 3)] = screen[4 * i + 3];
        }
    }
    Ok((rot_arr, scr_arr))
}

/// Write a binary little-endian PLY to disk with per-vertex RGB
/// colors. Mirrors `MitsubaRenderer._export_ply`.
#[pyfunction]
pub fn write_ply_binary(
    path: &str,
    verts: PyReadonlyArray2<'_, f32>,
    colors: PyReadonlyArray2<'_, f32>,
    faces: PyReadonlyArray2<'_, i32>,
) -> PyResult<()> {
    let v = verts.as_array();
    let c = colors.as_array();
    let f = faces.as_array();
    if v.shape()[1] != 3 || c.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "verts and colors must each have shape (N, 3)",
        ));
    }
    if f.shape()[1] != 3 {
        return Err(PyValueError::new_err("faces must have shape (F, 3)"));
    }
    if v.shape()[0] != c.shape()[0] {
        return Err(PyValueError::new_err(
            "verts and colors must have the same length",
        ));
    }
    let n = v.shape()[0];
    let nf = f.shape()[0];
    let mut vflat = Vec::with_capacity(n * 3);
    for i in 0..n {
        vflat.push(v[(i, 0)]);
        vflat.push(v[(i, 1)]);
        vflat.push(v[(i, 2)]);
    }
    let mut cflat = Vec::with_capacity(n * 3);
    for i in 0..n {
        cflat.push(c[(i, 0)]);
        cflat.push(c[(i, 1)]);
        cflat.push(c[(i, 2)]);
    }
    let mut fflat = Vec::with_capacity(nf * 3);
    for i in 0..nf {
        fflat.push(f[(i, 0)]);
        fflat.push(f[(i, 1)]);
        fflat.push(f[(i, 2)]);
    }
    raster::write_ply_to_file(std::path::Path::new(path), &vflat, &cflat, &fflat)
        .map_err(|e| PyValueError::new_err(format!("write_ply: {e}")))
}

/// Compute the Mitsuba auto-camera placement: returns `(origin,
/// target)` as `[f32; 3]` triples. Mirrors the bounds + offset block
/// in `MitsubaRenderer.render` when `camera` is None.
#[pyfunction]
pub fn mitsuba_auto_camera(
    verts: PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
) -> PyResult<([f32; 3], [f32; 3])> {
    let v = verts.as_array();
    let n = v.shape()[0];
    if v.shape()[1] != 3 || n == 0 {
        return Err(PyValueError::new_err(
            "verts must have shape (N, 3) with N >= 1",
        ));
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        for c in 0..3 {
            let x = v[(i, c)];
            if x < mn[c] {
                mn[c] = x;
            }
            if x > mx[c] {
                mx[c] = x;
            }
        }
    }
    let bounds = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    let max_b = bounds[0].max(bounds[1]).max(bounds[2]);
    let rat = (width as f32) / (height as f32);
    let target = [
        bounds[0] / 2.0 + mn[0],
        bounds[1] / 2.0 + mn[1],
        bounds[2] / 2.0 + mn[2],
    ];
    let offset = max_b * 0.5 * rat;
    let origin = [
        target[0],
        target[1] + offset,
        target[2] + 5.0 * offset,
    ];
    Ok((origin, target))
}

/// Verify the per-input mesh data + return the flattened f32 light
/// direction vector normalized to length 1. Mirrors the
/// `SoftwareRenderer.__init__` snippet that normalizes
/// `light_dir = [0.3, 0.5, 0.8]`. Trivial but pure compute.
#[pyfunction]
pub fn normalize_light_dir(dir: PyReadonlyArray1<'_, f32>) -> PyResult<[f32; 3]> {
    let v = dir.as_array();
    if v.len() != 3 {
        return Err(PyValueError::new_err("dir must have length 3"));
    }
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 0.0 {
        return Err(PyValueError::new_err("dir must have positive length"));
    }
    Ok([v[0] / len, v[1] / len, v[2] / len])
}

/// Build a `(n_verts,)` bool array marking pinned vertex indices.
/// Out-of-range indices are silently dropped, matching the
/// `0 <= vi < n_verts` guard in
/// `frontend/_invisible_collider_.py:check_wall_violations`.
#[pyfunction]
#[pyo3(signature = (n_verts, pinned=None))]
pub fn build_pinned_mask<'py>(
    py: Python<'py>,
    n_verts: usize,
    pinned: Option<Vec<i64>>,
) -> Bound<'py, PyArray1<bool>> {
    let mut mask = vec![false; n_verts];
    if let Some(idxs) = pinned {
        for vi in idxs {
            if vi >= 0 {
                let u = vi as usize;
                if u < n_verts {
                    mask[u] = true;
                }
            }
        }
    }
    PyArray1::from_vec(py, mask)
}

/// Merge a partial caller-supplied dict with the renderer's default
/// args (the `default_args` literal in `frontend/_render_.py`). Every
/// missing key is filled in. The defaults are constructed at call
/// time so the cache-dir-derived `tmp_path` reflects the current
/// environment.
///
/// Mirrors the body of `update_default_args` plus the `default_args`
/// table, moved into Rust so the Python call site is just a single
/// function call.
#[pyfunction]
#[pyo3(signature = (args, tmp_path))]
pub fn render_default_args<'py>(
    py: Python<'py>,
    args: Option<Bound<'py, PyDict>>,
    tmp_path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let merged = match args {
        Some(d) => d,
        None => PyDict::new(py),
    };
    use pyo3::IntoPyObjectExt;
    let pairs: [(&str, PyObject); 9] = [
        ("variant", "cuda_ad_rgb".into_py_any(py)?),
        ("max_depth", 12i64.into_py_any(py)?),
        ("width", 640i64.into_py_any(py)?),
        ("height", 480i64.into_py_any(py)?),
        ("fov", 20i64.into_py_any(py)?),
        ("camera", py.None()),
        (
            "up",
            pyo3::types::PyList::new(py, [0i64, 1i64, 0i64])?.into_py_any(py)?,
        ),
        ("sample_count", 64i64.into_py_any(py)?),
        ("tmp_path", tmp_path.into_py_any(py)?),
    ];
    for (k, v) in pairs.iter() {
        if !merged.contains(*k)? {
            merged.set_item(*k, v)?;
        }
    }
    Ok(merged)
}

/// Evaluate a single sphere SDF at a point. Mirrors
/// `frontend/_sdf_.py:SphereSDF.__call__`.
#[pyfunction]
pub fn sphere_sdf_eval(x: f64, y: f64, z: f64, cx: f64, cy: f64, cz: f64, r: f64) -> f64 {
    let dx = x - cx;
    let dy = y - cy;
    let dz = z - cz;
    (dx * dx + dy * dy + dz * dz).sqrt() - r
}

/// Evaluate a single capsule SDF at a point. Mirrors
/// `frontend/_sdf_.py:CapsuleSDF.__call__`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn capsule_sdf_eval(
    x: f64,
    y: f64,
    z: f64,
    p0x: f64,
    p0y: f64,
    p0z: f64,
    bax: f64,
    bay: f64,
    baz: f64,
    ba_dot_ba: f64,
    radius: f64,
) -> f64 {
    let pax = x - p0x;
    let pay = y - p0y;
    let paz = z - p0z;
    let pa_dot_ba = pax * bax + pay * bay + paz * baz;
    let h = (pa_dot_ba / ba_dot_ba).clamp(0.0, 1.0);
    let dx = pax - bax * h;
    let dy = pay - bay * h;
    let dz = paz - baz * h;
    (dx * dx + dy * dy + dz * dz).sqrt() - radius
}

/// Compute the axis-aligned bounding box of a sphere padded by 0.1
/// in each direction. Returns `((minx, miny, minz), (maxx, maxy,
/// maxz))` as a pair of f64 triples.
#[pyfunction]
pub fn sphere_bounds(cx: f64, cy: f64, cz: f64, r: f64) -> ((f64, f64, f64), (f64, f64, f64)) {
    (
        (cx - r - 0.1, cy - r - 0.1, cz - r - 0.1),
        (cx + r + 0.1, cy + r + 0.1, cz + r + 0.1),
    )
}

/// Compute the axis-aligned bounding box of a capsule padded by 0.1.
/// `p0` and `p1` are the segment endpoints; `r` is the radius.
#[pyfunction]
pub fn capsule_bounds(
    p0x: f64,
    p0y: f64,
    p0z: f64,
    p1x: f64,
    p1y: f64,
    p1z: f64,
    r: f64,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let mins = (
        p0x.min(p1x) - r - 0.1,
        p0y.min(p1y) - r - 0.1,
        p0z.min(p1z) - r - 0.1,
    );
    let maxs = (
        p0x.max(p1x) + r + 0.1,
        p0y.max(p1y) + r + 0.1,
        p0z.max(p1z) + r + 0.1,
    );
    (mins, maxs)
}

/// Pack the canonical `(types, params)` pair the SDF kernel evaluator
/// expects: `types` is `(N,) int32`, `params` is `(N, 8) f64`. `kind`
/// is 0 for sphere, 1 for capsule (matches `SDF_SPHERE`/`SDF_CAPSULE`
/// in the Python source).
#[pyfunction]
pub fn pack_sdf_primitive<'py>(
    py: Python<'py>,
    kind: i32,
    params: [f64; 8],
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray2<f64>>) {
    let t = PyArray1::from_vec(py, vec![kind]);
    let p_arr = PyArray2::<f64>::zeros(py, (1, 8), false);
    {
        let mut view = unsafe { p_arr.as_array_mut() };
        for (i, &v) in params.iter().enumerate() {
            view[(0, i)] = v;
        }
    }
    (t, p_arr)
}

fn extract_vec3_seq(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<[f64; 3]> {
    let seq = obj
        .extract::<Vec<f64>>()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be a length-3 sequence of floats")))?;
    if seq.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have length 3, got {}",
            seq.len()
        )));
    }
    Ok([seq[0], seq[1], seq[2]])
}

/// Drive the wall-violation collector end to end: extract each
/// wall's `get_entry()` first keyframe + `wall.normal`, drop
/// kinematic / empty walls, normalize the normal, and emit
/// `(vertex_index, wall_index, signed_distance)` rows. Mirrors the
/// per-wall Python `for wall_idx, wall in enumerate(walls)` loop in
/// `frontend/_invisible_collider_.py:check_wall_violations`,
/// including the static-wall filter, while keeping the inner work
/// in Rust instead of growing Python lists.
///
/// Returns `(violations, static_indices)`: `violations` is the flat
/// row list, `static_indices` is the original list position of every
/// wall that was actually scanned (kinematic / empty walls dropped),
/// so the verbose summary can iterate in the same order the Python
/// loop used.
#[pyfunction]
#[pyo3(signature = (vertices, is_pinned, walls))]
pub fn check_walls_violations_for_objs(
    py: Python<'_>,
    vertices: PyReadonlyArray2<'_, f64>,
    is_pinned: PyReadonlyArray1<'_, bool>,
    walls: &Bound<'_, PyList>,
) -> PyResult<(Vec<(usize, usize, f64)>, Vec<usize>)> {
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must have shape (N, 3), got {v_shape:?}"
        )));
    }
    let n_verts = v_shape[0];
    let v_slice = vertices.as_slice().map_err(|_| {
        PyTypeError::new_err("vertices must be C-contiguous; pass np.ascontiguousarray(...)")
    })?;
    let p_slice = is_pinned
        .as_slice()
        .map_err(|_| PyTypeError::new_err("is_pinned must be C-contiguous"))?;
    if p_slice.len() != n_verts {
        return Err(PyValueError::new_err(format!(
            "is_pinned length {} doesn't match vertex count {}",
            p_slice.len(),
            n_verts
        )));
    }
    let mut descs: Vec<ic::WallDesc> = Vec::with_capacity(walls.len());
    let mut idxs: Vec<usize> = Vec::with_capacity(walls.len());
    for (wall_idx, wall) in walls.iter().enumerate() {
        let entry_obj = wall.call_method0("get_entry")?;
        // Empty entry list: skip.
        let entry_len = entry_obj.len()?;
        if entry_len == 0 {
            continue;
        }
        // Kinematic wall (multiple keyframes): handled elsewhere.
        if entry_len > 1 {
            continue;
        }
        // entry[0][0] is the keyframe-0 position.
        let kf0 = entry_obj.get_item(0)?;
        let pos_obj = kf0.get_item(0)?;
        let pos = extract_vec3_seq(&pos_obj, "wall keyframe position")?;
        let normal_attr = wall.getattr("normal")?;
        let raw_normal = extract_vec3_seq(&normal_attr, "wall.normal")?;
        let nlen = (raw_normal[0] * raw_normal[0]
            + raw_normal[1] * raw_normal[1]
            + raw_normal[2] * raw_normal[2])
            .sqrt();
        let normal_unit = if nlen > 0.0 {
            [raw_normal[0] / nlen, raw_normal[1] / nlen, raw_normal[2] / nlen]
        } else {
            raw_normal
        };
        descs.push(ic::WallDesc {
            pos,
            normal_unit,
        });
        idxs.push(wall_idx);
    }
    if descs.is_empty() {
        return Ok((Vec::new(), idxs));
    }
    let out = py.allow_threads(|| ic::check_walls_violations_batch(v_slice, p_slice, &descs, &idxs));
    Ok((out, idxs))
}

/// Drive the sphere-violation collector end to end: extract each
/// sphere's `get_entry()`, `is_inverted`, `is_hemisphere`, drop
/// kinematic / empty spheres, and emit
/// `(vertex_index, sphere_index, distance_to_surface)` rows. Returns
/// `(violations, static_indices, mode_tags)` where `mode_tags[i]`
/// is the verbose-mode tag for `static_indices[i]`. Replaces the
/// per-sphere Python `for sphere_idx, sphere in enumerate(spheres)`
/// loop and the small `mode` list builder it embedded.
#[pyfunction]
#[pyo3(signature = (vertices, is_pinned, spheres))]
pub fn check_spheres_violations_for_objs(
    py: Python<'_>,
    vertices: PyReadonlyArray2<'_, f64>,
    is_pinned: PyReadonlyArray1<'_, bool>,
    spheres: &Bound<'_, PyList>,
) -> PyResult<(Vec<(usize, usize, f64)>, Vec<usize>, Vec<String>)> {
    let v_shape = vertices.shape();
    if v_shape.len() != 2 || v_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vertices must have shape (N, 3), got {v_shape:?}"
        )));
    }
    let n_verts = v_shape[0];
    let v_slice = vertices.as_slice().map_err(|_| {
        PyTypeError::new_err("vertices must be C-contiguous; pass np.ascontiguousarray(...)")
    })?;
    let p_slice = is_pinned
        .as_slice()
        .map_err(|_| PyTypeError::new_err("is_pinned must be C-contiguous"))?;
    if p_slice.len() != n_verts {
        return Err(PyValueError::new_err(format!(
            "is_pinned length {} doesn't match vertex count {}",
            p_slice.len(),
            n_verts
        )));
    }
    let mut descs: Vec<ic::SphereDesc> = Vec::with_capacity(spheres.len());
    let mut idxs: Vec<usize> = Vec::with_capacity(spheres.len());
    let mut tags: Vec<String> = Vec::with_capacity(spheres.len());
    for (sphere_idx, sphere) in spheres.iter().enumerate() {
        let entry_obj = sphere.call_method0("get_entry")?;
        let entry_len = entry_obj.len()?;
        if entry_len == 0 {
            continue;
        }
        if entry_len > 1 {
            continue;
        }
        // entry[0] = (pos, radius, _)
        let kf0 = entry_obj.get_item(0)?;
        let pos_obj = kf0.get_item(0)?;
        let radius_obj = kf0.get_item(1)?;
        let pos = extract_vec3_seq(&pos_obj, "sphere keyframe position")?;
        let radius: f64 = radius_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("sphere keyframe radius must be float"))?;
        let is_inverted: bool = sphere.getattr("is_inverted")?.extract()?;
        let is_hemisphere: bool = sphere.getattr("is_hemisphere")?.extract()?;
        descs.push(ic::SphereDesc {
            center: pos,
            radius,
            is_inverted,
            is_hemisphere,
        });
        idxs.push(sphere_idx);
        tags.push(ic::format_sphere_mode_tag(is_inverted, is_hemisphere));
    }
    if descs.is_empty() {
        return Ok((Vec::new(), idxs, tags));
    }
    let out = py.allow_threads(|| {
        ic::check_spheres_violations_batch(v_slice, p_slice, &descs, &idxs)
    });
    Ok((out, idxs, tags))
}

/// Reduce per-child bounding boxes via `union` or `intersect`. Walks
/// a list of SDF child objects, calls `bounds()` on each, and folds
/// the boxes in Rust. The per-child box list never materializes in
/// Python; used by `UnionSDF.bounds` / `IntersectionSDF.bounds`
/// (frontend/_sdf_.py).
#[pyfunction]
#[pyo3(signature = (children, op))]
pub fn reduce_bounds_from_children(
    children: &Bound<'_, PyList>,
    op: &str,
) -> PyResult<((f64, f64, f64), (f64, f64, f64))> {
    let mut iter = children.iter();
    let first = iter.next().ok_or_else(|| {
        PyValueError::new_err("reduce_bounds_from_children: empty children list")
    })?;
    let (mut min0, mut max0): ((f64, f64, f64), (f64, f64, f64)) =
        first.call_method0("bounds")?.extract()?;
    match op {
        "union" => {
            for child in iter {
                let (lo, hi): ((f64, f64, f64), (f64, f64, f64)) =
                    child.call_method0("bounds")?.extract()?;
                min0.0 = min0.0.min(lo.0);
                min0.1 = min0.1.min(lo.1);
                min0.2 = min0.2.min(lo.2);
                max0.0 = max0.0.max(hi.0);
                max0.1 = max0.1.max(hi.1);
                max0.2 = max0.2.max(hi.2);
            }
        }
        "intersect" => {
            for child in iter {
                let (lo, hi): ((f64, f64, f64), (f64, f64, f64)) =
                    child.call_method0("bounds")?.extract()?;
                min0.0 = min0.0.max(lo.0);
                min0.1 = min0.1.max(lo.1);
                min0.2 = min0.2.max(lo.2);
                max0.0 = max0.0.min(hi.0);
                max0.1 = max0.1.min(hi.1);
                max0.2 = max0.2.min(hi.2);
            }
        }
        other => {
            return Err(PyValueError::new_err(format!("unknown op: {other}")));
        }
    }
    Ok((min0, max0))
}

/// Walk a list of SDF child objects, call `get_kernel_primitives()` on
/// each, and concatenate the resulting `(types, params)` arrays in a
/// single pass. Replaces the Python list-builder + `np.concatenate`
/// pair in `UnionSDF.get_kernel_primitives` (frontend/_sdf_.py), so no
/// intermediate Python list grows on the way to the kernel evaluator.
#[pyfunction]
#[pyo3(signature = (children))]
pub fn concat_sdf_primitives_from_children<'py>(
    py: Python<'py>,
    children: &Bound<'_, PyList>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray2<f64>>)> {
    let mut types_out: Vec<i32> = Vec::new();
    let mut params_out: Vec<f64> = Vec::new();
    for child in children.iter() {
        let pair = child.call_method0("get_kernel_primitives")?;
        let tup = pair.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            PyTypeError::new_err("get_kernel_primitives() must return a (types, params) tuple")
        })?;
        if tup.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "get_kernel_primitives() tuple must have length 2, got {}",
                tup.len()
            )));
        }
        let t_arr = tup
            .get_item(0)?
            .downcast_into::<PyArray1<i32>>()
            .map_err(|_| PyTypeError::new_err("types must be a (N,) int32 ndarray"))?;
        let p_arr = tup
            .get_item(1)?
            .downcast_into::<PyArray2<f64>>()
            .map_err(|_| PyTypeError::new_err("params must be a (N, 8) f64 ndarray"))?;
        let t_view = t_arr.readonly();
        let p_view = p_arr.readonly();
        let t_slice = t_view
            .as_slice()
            .map_err(|_| PyTypeError::new_err("types must be C-contiguous"))?;
        let p_shape = p_view.shape();
        if p_shape.len() != 2 || p_shape[1] != 8 {
            return Err(PyValueError::new_err(format!(
                "params must have shape (N, 8), got {p_shape:?}"
            )));
        }
        if p_shape[0] != t_slice.len() {
            return Err(PyValueError::new_err(format!(
                "types length {} doesn't match params row count {}",
                t_slice.len(),
                p_shape[0]
            )));
        }
        let p_slice = p_view
            .as_slice()
            .map_err(|_| PyTypeError::new_err("params must be C-contiguous"))?;
        types_out.extend_from_slice(t_slice);
        params_out.extend_from_slice(p_slice);
    }
    let n = types_out.len();
    let t_out = PyArray1::from_vec(py, types_out);
    let p_out = ndarray::Array2::<f64>::from_shape_vec((n, 8), params_out)
        .map_err(|e| PyValueError::new_err(format!("params reshape failed: {e}")))?
        .into_pyarray(py);
    Ok((t_out, p_out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ortho_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(render_transform, m)?)?;
    m.add_function(wrap_pyfunction!(write_ply_binary, m)?)?;
    m.add_function(wrap_pyfunction!(mitsuba_auto_camera, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_light_dir, m)?)?;
    m.add_function(wrap_pyfunction!(build_pinned_mask, m)?)?;
    m.add_function(wrap_pyfunction!(render_default_args, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_sdf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(capsule_sdf_eval, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(capsule_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(pack_sdf_primitive, m)?)?;
    m.add_function(wrap_pyfunction!(check_walls_violations_for_objs, m)?)?;
    m.add_function(wrap_pyfunction!(check_spheres_violations_for_objs, m)?)?;
    m.add_function(wrap_pyfunction!(concat_sdf_primitives_from_children, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_bounds_from_children, m)?)?;
    Ok(())
}
