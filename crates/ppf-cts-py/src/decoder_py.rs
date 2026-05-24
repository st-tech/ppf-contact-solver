// File: crates/ppf-cts-py/src/decoder_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 wrappers for the pure-compute kernels backing
// `frontend/_decoder_.py`. The Python decoder still owns pickle / CBOR
// dispatch and the dataclass / scene / pin assembly: only the
// numpy-heavy inner loops live here.

use std::path::{Path, PathBuf};

use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice};

use ppf_cts_core::datamodel::decoder as dec;

use crate::utils_py::require_n_by_k;

/// Apply a 4x4 transform to local-space vertices and return
/// world-space `(N, 3)` `float32` vertices.
///
/// Wraps `dec::apply_transform_4x4`. Mirrors
/// `SceneDecoder._apply_transform`.
#[pyfunction]
#[pyo3(signature = (local_vert, transform))]
pub fn apply_transform_4x4<'py>(
    py: Python<'py>,
    local_vert: PyReadonlyArray2<'py, f64>,
    transform: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n = require_n_by_k::<_, 3>(&local_vert, "local_vert")?;
    let ts = transform.shape();
    if ts.len() != 2 || ts[0] != 4 || ts[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "transform must have shape (4, 4), got {ts:?}"
        )));
    }
    let v = local_vert
        .as_slice()
        .map_err(|_| PyTypeError::new_err("local_vert must be C-contiguous"))?;
    let m = transform
        .as_slice()
        .map_err(|_| PyTypeError::new_err("transform must be C-contiguous"))?;

    let out_flat = py.allow_threads(|| dec::apply_transform_4x4(v, m));
    debug_assert_eq!(out_flat.len(), n * 3);
    let arr = Array2::<f32>::from_shape_vec((n, 3), out_flat)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Format a build-plan job list into the human-readable status string.
///
/// `jobs` is a list of `dict` each containing at least `"name": str`
/// and `"cached": bool`. Wraps `dec::summarize_tetra_jobs`.
#[pyfunction]
#[pyo3(signature = (jobs))]
pub fn summarize_tetra_jobs(jobs: &Bound<'_, PyList>) -> PyResult<String> {
    let mut names: Vec<String> = Vec::with_capacity(jobs.len());
    let mut cached: Vec<bool> = Vec::with_capacity(jobs.len());
    for job in jobs.iter() {
        let d = job
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each job must be a dict"))?;
        let name_val = d
            .get_item("name")?
            .ok_or_else(|| PyValueError::new_err("job missing 'name' key"))?;
        let cached_val = d
            .get_item("cached")?
            .ok_or_else(|| PyValueError::new_err("job missing 'cached' key"))?;
        names.push(name_val.extract::<String>()?);
        cached.push(cached_val.extract::<bool>()?);
    }
    let job_views: Vec<dec::TetraJob<'_>> = names
        .iter()
        .zip(cached.iter())
        .map(|(n, c)| dec::TetraJob {
            name: n.as_str(),
            cached: *c,
        })
        .collect();
    Ok(dec::summarize_tetra_jobs(&job_views))
}

/// Project anchor points onto the closest target triangle and emit
/// the per-anchor stitch row (`ind`, `w`).
///
/// Inputs:
/// - `target_face`: `(F, 3)` int64 triangle indices.
/// - `target_pos`: `(P, 3)` float64 vertex positions (world space).
/// - `anchors`: `(K, 3)` float64 anchor positions.
/// - `src_indices`: `(K,)` int64 source-side vertex ids.
///
/// Returns `(ind, w)`: `(M, 4)` int64 and `(M, 4)` float32, where
/// `M <= K` (anchors that fall on a fully degenerate mesh are dropped).
#[pyfunction]
#[pyo3(signature = (target_face, target_pos, anchors, src_indices))]
pub fn barycentric_project_anchors<'py>(
    py: Python<'py>,
    target_face: PyReadonlyArray2<'py, i64>,
    target_pos: PyReadonlyArray2<'py, f64>,
    anchors: PyReadonlyArray2<'py, f64>,
    src_indices: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
    let _f = require_n_by_k::<_, 3>(&target_face, "target_face")?;
    let _p = require_n_by_k::<_, 3>(&target_pos, "target_pos")?;
    let n_a = require_n_by_k::<_, 3>(&anchors, "anchors")?;
    let n_s = src_indices.shape()[0];
    if n_s != n_a {
        return Err(PyValueError::new_err(format!(
            "src_indices length {n_s} doesn't match anchors rows {n_a}"
        )));
    }

    let face = target_face
        .as_slice()
        .map_err(|_| PyTypeError::new_err("target_face must be C-contiguous"))?;
    let pos = target_pos
        .as_slice()
        .map_err(|_| PyTypeError::new_err("target_pos must be C-contiguous"))?;
    let an = anchors
        .as_slice()
        .map_err(|_| PyTypeError::new_err("anchors must be C-contiguous"))?;
    let si = src_indices
        .as_slice()
        .map_err(|_| PyTypeError::new_err("src_indices must be C-contiguous"))?;

    let rows = py.allow_threads(|| dec::barycentric_project_anchors(face, pos, an, si));
    let m = rows.ind.len();
    let mut ind_flat = Vec::with_capacity(m * 4);
    let mut w_flat = Vec::with_capacity(m * 4);
    for r in &rows.ind {
        ind_flat.extend_from_slice(r);
    }
    for r in &rows.w {
        w_flat.extend_from_slice(r);
    }
    let ind_arr = Array2::<i64>::from_shape_vec((m, 4), ind_flat)
        .map_err(|e| PyValueError::new_err(format!("ind reshape failed: {e}")))?;
    let w_arr = Array2::<f32>::from_shape_vec((m, 4), w_flat)
        .map_err(|e| PyValueError::new_err(format!("w reshape failed: {e}")))?;
    Ok((ind_arr.into_pyarray(py), w_arr.into_pyarray(py)))
}

/// Compute the per-Blender-vertex closest simulation surface vertex
/// from a frame mapping. Mirrors the loop at lines 1349-1372 of
/// `_decoder_.py`.
///
/// Inputs:
/// - `tri_indices`: `(M,)` int64 triangle id per Blender vertex.
/// - `coefs`: `(M, 3)` float64 frame coefficients.
/// - `faces`: `(NF, 3)` int64 simulation surface triangles.
/// - `verts`: `(NV, 3)` float64 simulation surface positions.
///
/// Returns a Python list of int (length `M`).
#[pyfunction]
#[pyo3(signature = (tri_indices, coefs, faces, verts))]
pub fn solid_orig_to_sim<'py>(
    py: Python<'py>,
    tri_indices: PyReadonlyArray1<'py, i64>,
    coefs: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, i64>,
    verts: PyReadonlyArray2<'py, f64>,
) -> PyResult<Vec<i64>> {
    let m = tri_indices.shape()[0];
    let cs = coefs.shape();
    if cs.len() != 2 || cs[0] != m || cs[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "coefs must have shape (M, 3) matching tri_indices length {m}, got {cs:?}"
        )));
    }
    let _ = require_n_by_k::<_, 3>(&faces, "faces")?;
    let _ = require_n_by_k::<_, 3>(&verts, "verts")?;

    let ti = tri_indices
        .as_slice()
        .map_err(|_| PyTypeError::new_err("tri_indices must be C-contiguous"))?;
    let cf = coefs
        .as_slice()
        .map_err(|_| PyTypeError::new_err("coefs must be C-contiguous"))?;
    let fc = faces
        .as_slice()
        .map_err(|_| PyTypeError::new_err("faces must be C-contiguous"))?;
    let vt = verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("verts must be C-contiguous"))?;

    Ok(py.allow_threads(|| dec::solid_orig_to_sim(ti, cf, fc, vt)))
}

// ---------------------------------------------------------------------
// `BlenderApp` path resolution + validation wrappers.
// ---------------------------------------------------------------------

fn home_path_buf() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Resolve `BlenderApp.__init__`'s on-disk layout. Returns a dict with
/// `"data_dirpath"`, `"root"`, `"cache_root"`, all string paths.
/// Mirrors the path math at the top of `BlenderApp.__init__`.
#[pyfunction]
#[pyo3(signature = (frontend_file, name))]
pub fn blender_app_paths<'py>(
    py: Python<'py>,
    frontend_file: &str,
    name: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let home = home_path_buf();
    let p = dec::blender_app_paths(Path::new(frontend_file), name, home.as_deref());
    let d = PyDict::new(py);
    d.set_item("data_dirpath", p.data_dirpath.to_string_lossy().into_owned())?;
    d.set_item("root", p.root.to_string_lossy().into_owned())?;
    d.set_item("cache_root", p.cache_root.to_string_lossy().into_owned())?;
    Ok(d)
}

/// Filename used by `SceneDecoder._tetra_cache_path`. The on-disk path
/// is built by prepending the cache directory; this function only
/// produces the basename.
#[pyfunction]
#[pyo3(signature = (tri_mesh_hash))]
pub fn tetra_cache_filename(tri_mesh_hash: &str) -> String {
    dec::tetra_cache_filename(tri_mesh_hash)
}

/// Resolve the `app_state.pickle` final + tmp path pair used in
/// `BlenderApp._persist_app_state`.
#[pyfunction]
#[pyo3(signature = (root))]
pub fn app_state_persist_paths<'py>(py: Python<'py>, root: &str) -> PyResult<Bound<'py, PyDict>> {
    let p = dec::app_state_persist_paths(Path::new(root));
    let d = PyDict::new(py);
    d.set_item("final_path", p.final_path.to_string_lossy().into_owned())?;
    d.set_item("tmp_path", p.tmp_path.to_string_lossy().into_owned())?;
    Ok(d)
}

/// Assert a file path ends in `.pickle`. Raises `ValueError` with
/// the same string the Python source's `assert` raised.
#[pyfunction]
#[pyo3(signature = (filepath))]
pub fn validate_pickle_extension(filepath: &str) -> PyResult<()> {
    dec::validate_pickle_extension(filepath).map_err(crate::errors::into_py_err)
}

/// Confirm the loaded `param.pickle` carries the two required top-level
/// keys (`group`, `scene`). Raises `ValueError` on missing keys.
#[pyfunction]
#[pyo3(signature = (has_group, has_scene))]
pub fn validate_param_top_keys(has_group: bool, has_scene: bool) -> PyResult<()> {
    dec::validate_param_top_keys(has_group, has_scene)
        .map_err(crate::errors::into_py_err)
}

/// Reject pin op-type combinations the simulator can't represent.
/// Raises `ValueError` if `op_types` contains both `"torque"` and any
/// of `{"spin", "scale", "move_by"}`.
#[pyfunction]
#[pyo3(signature = (op_types, pin_name))]
pub fn validate_pin_op_types(op_types: Vec<String>, pin_name: &str) -> PyResult<()> {
    let refs: Vec<&str> = op_types.iter().map(|s| s.as_str()).collect();
    dec::validate_pin_op_types(&refs, pin_name)
        .map_err(crate::errors::into_py_err)
}

/// Mirror the assertion in `apply_invisible_colliders`: returns the
/// thickness back unchanged if positive, raises `AssertionError` (via
/// `ValueError`) if not.
#[pyfunction]
#[pyo3(signature = (kind, value))]
pub fn validate_invisible_collider_thickness(kind: &str, value: f64) -> PyResult<f64> {
    dec::validate_invisible_collider_thickness(kind, value)
        .map_err(crate::errors::into_py_err)
}

/// Mirror the legacy-pickle guard in `ParamDecoder.apply_to_objects`:
/// the third tuple slot must be present.
#[pyfunction]
#[pyo3(signature = (length))]
pub fn validate_param_group_has_uuids(length: usize) -> PyResult<()> {
    dec::validate_param_group_has_uuids(length)
        .map_err(crate::errors::into_py_err)
}

/// Reject empty UUIDs on the per-object pass of
/// `ParamDecoder.apply_to_objects`.
#[pyfunction]
#[pyo3(signature = (obj_name, uuid))]
pub fn validate_param_object_uuid(obj_name: &str, uuid: &str) -> PyResult<()> {
    dec::validate_param_object_uuid(obj_name, uuid)
        .map_err(crate::errors::into_py_err)
}

/// Reject objects without a name/uuid in `SceneDecoder.populate_objects`.
#[pyfunction]
#[pyo3(signature = (name, uuid))]
pub fn validate_scene_object_identity(name: &str, uuid: &str) -> PyResult<()> {
    dec::validate_scene_object_identity(name, uuid)
        .map_err(crate::errors::into_py_err)
}

/// Reject the encoder bug where a STATIC object carries more than one
/// motion source (`transform_animation`, `static_ops`, or
/// `static_deform_animation`). At most one may be present.
#[pyfunction]
#[pyo3(signature = (name, has_anim, has_ops, has_deform))]
pub fn validate_static_anim_xor_ops(
    name: &str,
    has_anim: bool,
    has_ops: bool,
    has_deform: bool,
) -> PyResult<()> {
    dec::validate_static_anim_xor_ops(name, has_anim, has_ops, has_deform)
        .map_err(crate::errors::into_py_err)
}

/// Validate a STATIC op type tag. Mirrors the `else` arm of the
/// dispatch in `populate_objects`'s static-ops loop.
#[pyfunction]
#[pyo3(signature = (op_type))]
pub fn validate_static_op_type(op_type: &str) -> PyResult<()> {
    dec::validate_static_op_type(op_type)
        .map_err(crate::errors::into_py_err)
}

/// Reject ROD objects without an edge buffer.
#[pyfunction]
#[pyo3(signature = (name, has_edge))]
pub fn validate_rod_has_edges(name: &str, has_edge: bool) -> PyResult<()> {
    dec::validate_rod_has_edges(name, has_edge)
        .map_err(crate::errors::into_py_err)
}

/// Reject groups with an unknown discriminator string.
#[pyfunction]
#[pyo3(signature = (group_type))]
pub fn validate_group_type(group_type: &str) -> PyResult<()> {
    dec::validate_group_type(group_type)
        .map_err(crate::errors::into_py_err)
}

/// Reject `mesh_ref` values that do not match any canonical mesh UUID.
#[pyfunction]
#[pyo3(signature = (name, mesh_ref, found))]
pub fn validate_mesh_ref_known(name: &str, mesh_ref: &str, found: bool) -> PyResult<()> {
    dec::validate_mesh_ref_known(name, mesh_ref, found)
        .map_err(crate::errors::into_py_err)
}

/// Reject objects that have neither inline `vert` nor a valid `mesh_ref`.
#[pyfunction]
#[pyo3(signature = (name, uuid, has_mesh))]
pub fn validate_object_has_mesh(name: &str, uuid: &str, has_mesh: bool) -> PyResult<()> {
    dec::validate_object_has_mesh(name, uuid, has_mesh)
        .map_err(crate::errors::into_py_err)
}

/// Find the row in a `(P, 3)` `float64` buffer closest (squared L2)
/// to `target`. Wraps `dec::closest_vertex_index`.
#[pyfunction]
#[pyo3(signature = (verts, target))]
pub fn closest_vertex_index<'py>(
    py: Python<'py>,
    verts: PyReadonlyArray2<'py, f64>,
    target: PyReadonlyArray1<'py, f64>,
) -> PyResult<usize> {
    let _n = require_n_by_k::<_, 3>(&verts, "verts")?;
    if target.shape()[0] != 3 {
        return Err(PyValueError::new_err(format!(
            "target must have shape (3,), got {:?}",
            target.shape()
        )));
    }
    let v = verts
        .as_slice()
        .map_err(|_| PyTypeError::new_err("verts must be C-contiguous"))?;
    let t = target
        .as_slice()
        .map_err(|_| PyTypeError::new_err("target must be C-contiguous"))?;
    Ok(py.allow_threads(|| dec::closest_vertex_index(v, t)))
}

/// Compute per-segment `(t_start, t_end, dx, dy, dz)` triples for a
/// keyframe-driven pin animation. Mirrors the embedded-move
/// derivation in `apply_pin_config`. `positions` is `(K, 3)`.
///
/// Returns a Python list of `(float, float, [float, float, float])`
/// tuples, one per consecutive pair; an input shorter than two
/// keyframes yields an empty list.
#[pyfunction]
#[pyo3(signature = (times, positions))]
pub fn keyframe_translation_segments<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyList>> {
    let n = times.shape()[0];
    let pshape = positions.shape();
    if pshape.len() != 2 || pshape[0] != n || pshape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "positions must have shape (K, 3) matching times length {n}, got {pshape:?}"
        )));
    }
    let t = times
        .as_slice()
        .map_err(|_| PyTypeError::new_err("times must be C-contiguous"))?;
    let p = positions
        .as_slice()
        .map_err(|_| PyTypeError::new_err("positions must be C-contiguous"))?;
    let segs = py.allow_threads(|| dec::keyframe_translation_segments(t, p));
    let out = PyList::empty(py);
    for (t_start, t_end, d) in segs {
        out.append((t_start, t_end, (d[0], d[1], d[2])))?;
    }
    Ok(out)
}

/// Walk a `param.pickle` group list and build the `uuid -> ftetwild
/// kwargs` map the populate phase consumes. Mirrors the Python loop
/// in `BlenderApp.populate`.
///
/// Each `group_entry` is expected to be a tuple/list of length >=3:
/// `(params_dict, objects, uuids, ...)`. Entries that are too short
/// or have empty UUIDs are skipped (matching the Python guard).
#[pyfunction]
#[pyo3(signature = (group_entries))]
pub fn extract_ftetwild_by_uuid<'py>(
    py: Python<'py>,
    group_entries: &Bound<'_, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for entry in group_entries.iter() {
        // `entry[0]` -> params, `entry[2]` -> uuids.
        let len_obj = entry.len()?;
        if len_obj < 3 {
            continue;
        }
        let params = entry.get_item(0)?;
        let uuids = entry.get_item(2)?;
        let kw_obj = if params.is_none() {
            None
        } else {
            // params.get("ftetwild") may itself be None.
            let params_dict = params
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("group entry params must be a dict"))?;
            params_dict.get_item("ftetwild")?
        };
        let kw = match kw_obj {
            Some(kw) if !kw.is_none() => {
                let dct = kw
                    .downcast::<PyDict>()
                    .map_err(|_| {
                        PyTypeError::new_err("ftetwild value must be a dict")
                    })?
                    .clone();
                dct
            }
            _ => continue,
        };
        if kw.len() == 0 {
            continue;
        }
        let uuid_list = uuids
            .downcast::<PyList>()
            .map_err(|_| PyTypeError::new_err("uuids must be a list"))?;
        for uuid_item in uuid_list.iter() {
            let uuid: String = uuid_item.extract()?;
            if uuid.is_empty() {
                continue;
            }
            out.set_item(uuid, &kw)?;
        }
    }
    Ok(out)
}

/// Run the dedup pass over object plan entries and rebuild the
/// post-dedup `tetra_jobs` list. Replaces both the per-entry
/// `tet_hash_seen` dict-builder loop and the
/// `tetra_jobs.append({"name", "cached"})` rebuild loop in
/// `SceneDecoder.populate_objects` (frontend/_decoder_.py).
///
/// Mutations applied to each entry:
///   - `tetra_reuse_from`: sets to the earlier entry index (int) when
///     `tri_mesh.hash` matches an earlier SOLID entry, else `None`.
///   - `tetra_weight`: zeroed for reuse rows.
///   - `tetra_index`: 1-indexed integer for fresh tet jobs, `None`
///     for reuse rows.
///
/// Returns `(tetra_jobs, work_delta)` where `work_delta` is the total
/// `tetra_weight` reclaimed (negative or zero) which the caller adds
/// to its running `object_work` accumulator.
#[pyfunction]
#[pyo3(signature = (object_entries))]
pub fn dedup_and_rebuild_tetra_jobs<'py>(
    py: Python<'py>,
    object_entries: &Bound<'_, PyList>,
) -> PyResult<(Bound<'py, PyList>, f64)> {
    use std::collections::HashMap;
    let mut tet_hash_seen: HashMap<String, usize> = HashMap::new();
    let mut work_delta: f64 = 0.0;
    // First pass: dedup decisions.
    for (i, entry_obj) in object_entries.iter().enumerate() {
        let entry = entry_obj
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each object entry must be a dict"))?;
        let group_type: String = match entry.get_item("group_type")? {
            Some(v) => v.extract().unwrap_or_default(),
            None => String::new(),
        };
        let tri_mesh_obj = match entry.get_item("tri_mesh")? {
            Some(v) if !v.is_none() => v,
            _ => continue,
        };
        if group_type != "SOLID" {
            continue;
        }
        let h: String = tri_mesh_obj.getattr("hash")?.extract()?;
        if let Some(&earlier) = tet_hash_seen.get(&h) {
            entry.set_item("tetra_reuse_from", earlier)?;
            let weight: f64 = entry
                .get_item("tetra_weight")?
                .map(|v| v.extract().unwrap_or(0.0f64))
                .unwrap_or(0.0);
            work_delta -= weight;
            entry.set_item("tetra_weight", 0.0f64)?;
        } else {
            tet_hash_seen.insert(h, i);
            entry.set_item("tetra_reuse_from", py.None())?;
        }
    }
    // Second pass: rebuild the (deduped) jobs list and reassign tetra_index.
    let jobs = PyList::empty(py);
    let mut tetra_idx: i64 = 0;
    for entry_obj in object_entries.iter() {
        let entry = entry_obj
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each object entry must be a dict"))?;
        let group_type: String = match entry.get_item("group_type")? {
            Some(v) => v.extract().unwrap_or_default(),
            None => String::new(),
        };
        let tri_mesh_present = match entry.get_item("tri_mesh")? {
            Some(v) => !v.is_none(),
            None => false,
        };
        if group_type != "SOLID" || !tri_mesh_present {
            continue;
        }
        let is_reuse = match entry.get_item("tetra_reuse_from")? {
            Some(v) => !v.is_none(),
            None => false,
        };
        if is_reuse {
            entry.set_item("tetra_index", py.None())?;
        } else {
            tetra_idx += 1;
            entry.set_item("tetra_index", tetra_idx)?;
            let name: String = match entry.get_item("name")? {
                Some(v) => v.extract().unwrap_or_default(),
                None => String::new(),
            };
            let cached: bool = match entry.get_item("tetra_cached")? {
                Some(v) => v.extract().unwrap_or(false),
                None => false,
            };
            let row = PyDict::new(py);
            row.set_item("name", name)?;
            row.set_item("cached", cached)?;
            jobs.append(row)?;
        }
    }
    Ok((jobs, work_delta))
}

/// Apply a batch of explicit cross-stitch entries to a scene's
/// `_cross_stitch` list. Replaces the per-entry Python `for entry in
/// param_decoder.cross_stitch: ...` loop in `BlenderApp.make()`
/// (frontend/_decoder_.py:229) plus the per-call `.append` in
/// `BlenderApp._apply_explicit_cross_stitch`.
///
/// For each entry: validate that source/target are known, drop
/// entries with empty `ind` / `w`, run `barycentric_project_anchors`
/// for SOLID targets, build the canonical dict, and append it to
/// `scene_cross_stitch`. The Python caller never grows a list in its
/// own loop body.
///
/// Returns the count of entries that were actually appended.
#[pyfunction]
#[pyo3(signature = (entries, obj_info, scene_cross_stitch, verbose))]
pub fn cross_stitch_apply_batch<'py>(
    py: Python<'py>,
    entries: &Bound<'py, PyList>,
    obj_info: &Bound<'py, PyDict>,
    scene_cross_stitch: &Bound<'py, PyList>,
    verbose: bool,
) -> PyResult<usize> {
    let np = py.import("numpy")?;
    let mut appended = 0usize;
    for entry_obj in entries.iter() {
        let entry = entry_obj
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("each cross_stitch entry must be a dict"))?;
        let source_name: String = match entry.get_item("source_uuid")? {
            Some(v) => v.extract().unwrap_or_default(),
            None => String::new(),
        };
        let target_name: String = match entry.get_item("target_uuid")? {
            Some(v) => v.extract().unwrap_or_default(),
            None => String::new(),
        };
        let source_info = obj_info.get_item(&source_name)?;
        let target_info = obj_info.get_item(&target_name)?;
        // Same guard as the rest of the cross-stitch decoder path.
        dec::validate_cross_stitch_endpoints(
            &source_name,
            &target_name,
            source_info.is_some(),
            target_info.is_some(),
        )
        .map_err(crate::errors::into_py_err)?;

        let ind_raw = entry.get_item("ind")?.unwrap_or_else(|| py.None().into_bound(py));
        let w_raw = entry.get_item("w")?.unwrap_or_else(|| py.None().into_bound(py));
        let mut ind_arr: Bound<'py, PyAny> = np
            .call_method1("asarray", (ind_raw,))?
            .call_method1("astype", ("int64",))?;
        let mut w_arr: Bound<'py, PyAny> = np
            .call_method1("asarray", (w_raw,))?
            .call_method1("astype", ("float32",))?;
        let ind_len: usize = ind_arr.len()?;
        let w_len: usize = w_arr.len()?;
        if ind_len == 0 || w_len == 0 {
            continue;
        }

        let target_info_dict = target_info
            .expect("validated above")
            .downcast_into::<PyDict>()
            .map_err(|_| PyTypeError::new_err("target obj_info entry must be a dict"))?;
        let is_solid = match target_info_dict.get_item("type")? {
            Some(t) => t.extract::<String>().unwrap_or_default() == "SOLID",
            None => false,
        };

        if is_solid {
            let target_points = match entry.get_item("target_points")? {
                Some(v) => v,
                None => continue,
            };
            let tp_len: usize = target_points.len().unwrap_or(0);
            if tp_len != ind_len {
                continue;
            }
            let target_pos_v = target_info_dict
                .get_item("V")?
                .ok_or_else(|| PyValueError::new_err("target_info missing V"))?;
            let target_face_f = target_info_dict
                .get_item("F")?
                .ok_or_else(|| PyValueError::new_err("target_info missing F"))?;
            let target_pos: Bound<'py, PyArray2<f64>> = np
                .call_method1("ascontiguousarray", (
                    np.call_method1("asarray", (target_pos_v,))?
                        .call_method1("astype", ("float64",))?,
                ))?
                .downcast_into::<PyArray2<f64>>()
                .map_err(|_| PyTypeError::new_err("target_pos must be (N, 3) f64"))?;
            let target_face: Bound<'py, PyArray2<i64>> = np
                .call_method1("ascontiguousarray", (
                    np.call_method1("asarray", (target_face_f,))?
                        .call_method1("astype", ("int64",))?,
                ))?
                .downcast_into::<PyArray2<i64>>()
                .map_err(|_| PyTypeError::new_err("target_face must be (M, 3) i64"))?;
            let anchors: Bound<'py, PyArray2<f64>> = np
                .call_method1("ascontiguousarray", (
                    np.call_method1("asarray", (target_points,))?
                        .call_method1("astype", ("float64",))?,
                ))?
                .downcast_into::<PyArray2<f64>>()
                .map_err(|_| PyTypeError::new_err("target_points must be (K, 3) f64"))?;
            // src_indices = ind[:, 0], contiguous int64. The first
            // tuple element must be a real slice; ``py.None()`` is
            // interpreted by numpy as ``np.newaxis`` (adding a leading
            // axis) instead of ``:``, leaving src_col 2D and tripping
            // the (K,) downcast below.
            let full_slice = PySlice::full(py);
            let src_col = ind_arr
                .call_method1("__getitem__", ((&full_slice, 0usize),))
                .or_else(|_| {
                    // Fallback: use numpy slicing via np.asarray + indexing
                    np.call_method1("asarray", (&ind_arr,))?
                        .call_method1("__getitem__", ((&full_slice, 0usize),))
                })?;
            let src_indices: Bound<'py, PyArray1<i64>> = np
                .call_method1("ascontiguousarray", (
                    np.call_method1("asarray", (src_col,))?
                        .call_method1("astype", ("int64",))?,
                ))?
                .downcast_into::<PyArray1<i64>>()
                .map_err(|_| PyTypeError::new_err("src_indices must be (K,) i64"))?;

            let target_face_view = target_face.readonly();
            let target_pos_view = target_pos.readonly();
            let anchors_view = anchors.readonly();
            let src_indices_view = src_indices.readonly();
            let tf = target_face_view
                .as_slice()
                .map_err(|_| PyTypeError::new_err("target_face must be C-contiguous"))?;
            let tp = target_pos_view
                .as_slice()
                .map_err(|_| PyTypeError::new_err("target_pos must be C-contiguous"))?;
            let an = anchors_view
                .as_slice()
                .map_err(|_| PyTypeError::new_err("anchors must be C-contiguous"))?;
            let si = src_indices_view
                .as_slice()
                .map_err(|_| PyTypeError::new_err("src_indices must be C-contiguous"))?;
            let stitch =
                py.allow_threads(|| dec::barycentric_project_anchors(tf, tp, an, si));
            if stitch.ind.is_empty() {
                continue;
            }
            // Reshape rust result into (K, 4) numpy arrays.
            let n_rows = stitch.ind.len();
            let mut ind_flat: Vec<i64> = Vec::with_capacity(n_rows * 4);
            for row in &stitch.ind {
                ind_flat.extend_from_slice(row);
            }
            let mut w_flat: Vec<f32> = Vec::with_capacity(n_rows * 4);
            for row in &stitch.w {
                w_flat.extend_from_slice(row);
            }
            let ind_new = ndarray::Array2::<i64>::from_shape_vec((n_rows, 4), ind_flat)
                .map_err(|e| PyValueError::new_err(format!("ind reshape: {e}")))?
                .into_pyarray(py);
            let w_new = ndarray::Array2::<f32>::from_shape_vec((n_rows, 4), w_flat)
                .map_err(|e| PyValueError::new_err(format!("w reshape: {e}")))?
                .into_pyarray(py);
            ind_arr = ind_new.into_any();
            w_arr = w_new.into_any();
        }

        let stitch_stiffness: f64 = match entry.get_item("stitch_stiffness")? {
            Some(v) => v.extract().unwrap_or(1.0f64),
            None => 1.0f64,
        };
        let row = PyDict::new(py);
        row.set_item("source_name", &source_name)?;
        row.set_item("target_name", &target_name)?;
        row.set_item("ind", &ind_arr)?;
        row.set_item("w", &w_arr)?;
        row.set_item("stitch_stiffness", stitch_stiffness)?;
        scene_cross_stitch.append(row)?;
        appended += 1;
        if verbose {
            let n_ind = ind_arr.len()?;
            println!(
                "  Stitch {source_name} -> {target_name}: {n_ind} edges (explicit)"
            );
        }
    }
    Ok(appended)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_transform_4x4, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_tetra_jobs, m)?)?;
    m.add_function(wrap_pyfunction!(barycentric_project_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(solid_orig_to_sim, m)?)?;
    m.add_function(wrap_pyfunction!(blender_app_paths, m)?)?;
    m.add_function(wrap_pyfunction!(tetra_cache_filename, m)?)?;
    m.add_function(wrap_pyfunction!(app_state_persist_paths, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pickle_extension, m)?)?;
    m.add_function(wrap_pyfunction!(validate_param_top_keys, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pin_op_types, m)?)?;
    m.add_function(wrap_pyfunction!(validate_invisible_collider_thickness, m)?)?;
    m.add_function(wrap_pyfunction!(validate_param_group_has_uuids, m)?)?;
    m.add_function(wrap_pyfunction!(validate_param_object_uuid, m)?)?;
    m.add_function(wrap_pyfunction!(validate_scene_object_identity, m)?)?;
    m.add_function(wrap_pyfunction!(extract_ftetwild_by_uuid, m)?)?;
    m.add_function(wrap_pyfunction!(closest_vertex_index, m)?)?;
    m.add_function(wrap_pyfunction!(keyframe_translation_segments, m)?)?;
    m.add_function(wrap_pyfunction!(validate_static_anim_xor_ops, m)?)?;
    m.add_function(wrap_pyfunction!(validate_static_op_type, m)?)?;
    m.add_function(wrap_pyfunction!(validate_rod_has_edges, m)?)?;
    m.add_function(wrap_pyfunction!(validate_group_type, m)?)?;
    m.add_function(wrap_pyfunction!(validate_mesh_ref_known, m)?)?;
    m.add_function(wrap_pyfunction!(validate_object_has_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(cross_stitch_apply_batch, m)?)?;
    m.add_function(wrap_pyfunction!(dedup_and_rebuild_tetra_jobs, m)?)?;
    Ok(())
}
