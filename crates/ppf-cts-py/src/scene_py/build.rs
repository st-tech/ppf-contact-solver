// File: crates/ppf-cts-py/src/scene_py/build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Scene.build full-body assembly. Walks every dyn / static object and
// produces the global numeric arrays the Python FixedScene constructor
// consumes. Pin / TransformAnimation construction stays Python-side
// (Python dataclasses); everything else flows through Rust. Also hosts
// the shell shrink/strain-limit conflict check used during build.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use ppf_cts_core::kernels::scene_build as sb;

use super::helpers::{read_edges_u32, read_faces_u32, read_tets_u32};
use crate::errors::into_py_err;

fn read_n3_f64_to_vec<'py>(
    arr: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<(Vec<f64>, usize)> {
    let view = arr.extract::<PyReadonlyArray2<'py, f64>>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be (N, 3) ndarray of float64"))
    })?;
    let s = view.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must be (N, 3), got {s:?}"
        )));
    }
    let slice = view
        .as_slice()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?;
    Ok((slice.to_vec(), s[0]))
}

fn read_uv_buf<'py>(arr: &Bound<'py, PyAny>) -> PyResult<(Vec<f64>, usize)> {
    // UV is a list[np.ndarray] in Python; each entry has 6 elements
    // (either shape (3,2) or fallback (2,3)). We accept a single
    // (n_faces, 6) f64 numpy array as the kernel input form.
    let view = arr.extract::<PyReadonlyArray2<'py, f64>>().map_err(|_| {
        PyTypeError::new_err("uv must be (n_faces, 6) ndarray of float64")
    })?;
    let s = view.shape();
    if s.len() != 2 || s[1] != 6 {
        return Err(PyValueError::new_err(format!(
            "uv must be (n_faces, 6), got {s:?}"
        )));
    }
    let slice = view
        .as_slice()
        .map_err(|_| PyTypeError::new_err("uv must be C-contiguous"))?;
    Ok((slice.to_vec(), s[0]))
}

/// Walk every dyn / static object and produce all the heavy global
/// arrays Scene.build assembles. Returns a Python dict containing
/// every concatenated buffer keyed by the same names as the original
/// Python locals (`concat_vert`, `concat_color`, etc.). Pin /
/// TransformAnimation construction stays Python; this kernel only
/// returns numeric arrays + per-object stats so the wrapper can do
/// param replication and Python-object construction with the right
/// counts.
///
/// Args (Python-side):
///   * `dyn_objects`: list of dicts. Required keys: `name: str`,
///     `obj_type: str` ("rod"|"tri"|"tet"), `vertex` (n_verts, 3) f64,
///     `color` (n_verts, 3) f64, `velocity` [3] f64, `dynamic_color`
///     (int 0/1), `dynamic_intensity` (float), `position` [3] f64,
///     `pinned_indices` list[int]. Optional: `edges` (n,2) int,
///     `faces` (n,3) int, `tets` (n,4) int, `uv` (n_faces,6) f64,
///     `stitch_ind` (n, c) int, `stitch_w` (n, k) f64.
///   * `static_objects`: list of dicts. Required: `name`, `vertex_world`
///     (n_verts, 3) f64, `color` [3] f64, `position` [3] f64. Optional:
///     `faces` (n, 3) int.
///   * `dmap_order`: list of (name, [x, y, z]) covering every object
///     (dyn + static) in original insertion order.
///   * `map_by_name`: dict[str, ndarray(int64)]
///   * `concat_count`: int
///   * `has_merge`: bool
///   * `cross_stitches`: list of dicts {source_name, target_name,
///     ind: (k, 4) int, w: (k, 4) f64}
#[pyfunction]
#[pyo3(signature = (
    dyn_objects, static_objects, dmap_order, map_by_name, concat_count,
    has_merge, cross_stitches,
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn scene_build_fixed<'py>(
    py: Python<'py>,
    dyn_objects: &Bound<'py, PyList>,
    static_objects: &Bound<'py, PyList>,
    dmap_order: &Bound<'py, PyList>,
    map_by_name: &Bound<'py, PyDict>,
    concat_count: usize,
    has_merge: bool,
    cross_stitches: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    // ----- decode dyn objects -----
    struct OwnedDyn {
        name: String,
        obj_type: String,
        vertex: Vec<f64>,
        color: Vec<f64>,
        velocity: [f64; 3],
        edges: Option<Vec<[u32; 2]>>,
        faces: Option<Vec<[u32; 3]>>,
        tets: Option<Vec<[u32; 4]>>,
        uv: Option<Vec<f64>>,
        dynamic_color: u8,
        dynamic_intensity: f64,
        pinned_indices: Vec<i64>,
        stitch_ind: Option<Vec<i64>>,
        stitch_ind_cols: usize,
        stitch_w: Option<Vec<f64>>,
        stitch_w_cols: usize,
        position: [f64; 3],
    }
    let mut dyn_owned: Vec<OwnedDyn> = Vec::with_capacity(dyn_objects.len());
    for obj in dyn_objects.iter() {
        let d = obj.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("each dyn_objects entry must be a dict")
        })?;
        let name: String = d
            .get_item("name")?
            .ok_or_else(|| PyValueError::new_err("missing 'name'"))?
            .extract()?;
        let obj_type: String = d
            .get_item("obj_type")?
            .ok_or_else(|| PyValueError::new_err("missing 'obj_type'"))?
            .extract()?;
        let v_item = d
            .get_item("vertex")?
            .ok_or_else(|| PyValueError::new_err("missing 'vertex'"))?;
        let (vertex, _) = read_n3_f64_to_vec(&v_item, "vertex")?;
        let c_item = d
            .get_item("color")?
            .ok_or_else(|| PyValueError::new_err("missing 'color'"))?;
        let (color, _) = read_n3_f64_to_vec(&c_item, "color")?;
        let velocity: [f64; 3] = d
            .get_item("velocity")?
            .ok_or_else(|| PyValueError::new_err("missing 'velocity'"))?
            .extract()?;
        let position: [f64; 3] = d
            .get_item("position")?
            .ok_or_else(|| PyValueError::new_err("missing 'position'"))?
            .extract()?;
        let dynamic_color: u8 = d
            .get_item("dynamic_color")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0u8);
        let dynamic_intensity: f64 = d
            .get_item("dynamic_intensity")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1.0);
        let pinned_indices: Vec<i64> = d
            .get_item("pinned_indices")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let edges = match d.get_item("edges")? {
            Some(item) if !item.is_none() => Some(read_edges_u32(&item)?),
            _ => None,
        };
        let faces = match d.get_item("faces")? {
            Some(item) if !item.is_none() => Some(read_faces_u32(&item)?),
            _ => None,
        };
        let tets = match d.get_item("tets")? {
            Some(item) if !item.is_none() => Some(read_tets_u32(&item)?),
            _ => None,
        };
        let uv = match d.get_item("uv")? {
            Some(item) if !item.is_none() => {
                let (buf, _) = read_uv_buf(&item)?;
                Some(buf)
            }
            _ => None,
        };
        let (stitch_ind, stitch_ind_cols) = match d.get_item("stitch_ind")? {
            Some(item) if !item.is_none() => {
                let view = item
                    .extract::<PyReadonlyArray2<'_, i64>>()
                    .map_err(|_| {
                        PyTypeError::new_err("stitch_ind must be (M, 3) or (M, 4) int64")
                    })?;
                let s = view.shape();
                if s.len() != 2 || (s[1] != 3 && s[1] != 4) {
                    return Err(PyValueError::new_err(format!(
                        "stitch_ind must be (M, 3) or (M, 4), got {s:?}"
                    )));
                }
                let slice = view
                    .as_slice()
                    .map_err(|_| PyTypeError::new_err("stitch_ind must be C-contiguous"))?;
                (Some(slice.to_vec()), s[1])
            }
            _ => (None, 0),
        };
        let (stitch_w, stitch_w_cols) = match d.get_item("stitch_w")? {
            Some(item) if !item.is_none() => {
                let view = item
                    .extract::<PyReadonlyArray2<'_, f64>>()
                    .map_err(|_| {
                        PyTypeError::new_err("stitch_w must be (M, 2) or (M, 4) float64")
                    })?;
                let s = view.shape();
                if s.len() != 2 || (s[1] != 2 && s[1] != 4) {
                    return Err(PyValueError::new_err(format!(
                        "stitch_w must be (M, 2) or (M, 4), got {s:?}"
                    )));
                }
                let slice = view
                    .as_slice()
                    .map_err(|_| PyTypeError::new_err("stitch_w must be C-contiguous"))?;
                (Some(slice.to_vec()), s[1])
            }
            _ => (None, 0),
        };
        dyn_owned.push(OwnedDyn {
            name,
            obj_type,
            vertex,
            color,
            velocity,
            edges,
            faces,
            tets,
            uv,
            dynamic_color,
            dynamic_intensity,
            pinned_indices,
            stitch_ind,
            stitch_ind_cols,
            stitch_w,
            stitch_w_cols,
            position,
        });
    }

    // ----- decode static objects -----
    struct OwnedStatic {
        name: String,
        vertex_world: Vec<f64>,
        color: [f64; 3],
        faces: Option<Vec<[u32; 3]>>,
        position: [f64; 3],
    }
    let mut static_owned: Vec<OwnedStatic> = Vec::with_capacity(static_objects.len());
    for obj in static_objects.iter() {
        let d = obj.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("each static_objects entry must be a dict")
        })?;
        let name: String = d
            .get_item("name")?
            .ok_or_else(|| PyValueError::new_err("missing 'name'"))?
            .extract()?;
        let v_item = d
            .get_item("vertex_world")?
            .ok_or_else(|| PyValueError::new_err("missing 'vertex_world'"))?;
        let (vertex_world, _) = read_n3_f64_to_vec(&v_item, "vertex_world")?;
        let color: [f64; 3] = d
            .get_item("color")?
            .ok_or_else(|| PyValueError::new_err("missing 'color'"))?
            .extract()?;
        let position: [f64; 3] = d
            .get_item("position")?
            .ok_or_else(|| PyValueError::new_err("missing 'position'"))?
            .extract()?;
        let faces = match d.get_item("faces")? {
            Some(item) if !item.is_none() => Some(read_faces_u32(&item)?),
            _ => None,
        };
        static_owned.push(OwnedStatic {
            name,
            vertex_world,
            color,
            faces,
            position,
        });
    }

    // ----- decode dmap order -----
    let mut dmap_owned: Vec<(String, [f64; 3])> = Vec::with_capacity(dmap_order.len());
    for entry in dmap_order.iter() {
        let tup = entry.downcast::<PyTuple>().map_err(|_| {
            PyValueError::new_err("each dmap_order entry must be a (name, [x,y,z]) tuple")
        })?;
        if tup.len() != 2 {
            return Err(PyValueError::new_err(
                "each dmap_order entry must be a 2-tuple",
            ));
        }
        let name: String = tup.get_item(0)?.extract()?;
        let pos: [f64; 3] = tup.get_item(1)?.extract()?;
        dmap_owned.push((name, pos));
    }

    // ----- decode map_by_name into HashMap<String, Vec<i64>> -----
    let mut map_owned: std::collections::HashMap<String, Vec<i64>> =
        std::collections::HashMap::with_capacity(map_by_name.len());
    for (k, v) in map_by_name.iter() {
        let key: String = k.extract()?;
        let view = v
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyTypeError::new_err("map_by_name values must be int64 ndarray"))?;
        let slice = view
            .as_slice()
            .map_err(|_| PyTypeError::new_err("map_by_name values must be C-contiguous"))?;
        map_owned.insert(key, slice.to_vec());
    }

    // ----- decode cross stitches -----
    struct OwnedCross {
        source_name: String,
        target_name: String,
        ind: Vec<i64>,
        weights: Vec<f64>,
        k: usize,
    }
    let mut cs_owned: Vec<OwnedCross> = Vec::with_capacity(cross_stitches.len());
    for cs in cross_stitches.iter() {
        let d = cs.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("each cross_stitches entry must be a dict")
        })?;
        let source_name: String = d
            .get_item("source_name")?
            .ok_or_else(|| PyValueError::new_err("missing 'source_name'"))?
            .extract()?;
        let target_name: String = d
            .get_item("target_name")?
            .ok_or_else(|| PyValueError::new_err("missing 'target_name'"))?
            .extract()?;
        let ind_item = d
            .get_item("ind")?
            .ok_or_else(|| PyValueError::new_err("missing 'ind'"))?;
        let view_i = ind_item
            .extract::<PyReadonlyArray2<'_, i64>>()
            .map_err(|_| PyTypeError::new_err("cross stitch ind must be (K, 4) int64"))?;
        let s = view_i.shape();
        if s.len() != 2 || s[1] != 4 {
            return Err(PyValueError::new_err(format!(
                "cross stitch ind must be (K, 4), got {s:?}"
            )));
        }
        let ind = view_i
            .as_slice()
            .map_err(|_| PyTypeError::new_err("cross stitch ind must be C-contiguous"))?
            .to_vec();
        let w_item = d
            .get_item("w")?
            .ok_or_else(|| PyValueError::new_err("missing 'w'"))?;
        let view_w = w_item
            .extract::<PyReadonlyArray2<'_, f64>>()
            .map_err(|_| PyTypeError::new_err("cross stitch w must be (K, 4) float64"))?;
        let s2 = view_w.shape();
        if s2.len() != 2 || s2[1] != 4 {
            return Err(PyValueError::new_err(format!(
                "cross stitch w must be (K, 4), got {s2:?}"
            )));
        }
        let weights = view_w
            .as_slice()
            .map_err(|_| PyTypeError::new_err("cross stitch w must be C-contiguous"))?
            .to_vec();
        cs_owned.push(OwnedCross {
            source_name,
            target_name,
            ind,
            weights,
            k: s[0],
        });
    }

    // ----- build view structs and call kernel -----
    let dyn_view: Vec<sb::AssembleObject> = dyn_owned
        .iter()
        .map(|o| sb::AssembleObject {
            name: o.name.as_str(),
            obj_type: o.obj_type.as_str(),
            vertex: o.vertex.as_slice(),
            color: o.color.as_slice(),
            velocity: o.velocity,
            edges: o.edges.as_deref(),
            faces: o.faces.as_deref(),
            tets: o.tets.as_deref(),
            uv: o.uv.as_deref(),
            dynamic_color: o.dynamic_color,
            dynamic_intensity: o.dynamic_intensity,
            pinned_indices: o.pinned_indices.as_slice(),
            stitch_ind: o.stitch_ind.as_deref(),
            stitch_ind_cols: o.stitch_ind_cols,
            stitch_w: o.stitch_w.as_deref(),
            stitch_w_cols: o.stitch_w_cols,
            position: o.position,
        })
        .collect();
    let cs_view: Vec<sb::CrossStitch> = cs_owned
        .iter()
        .map(|c| sb::CrossStitch {
            source_name: c.source_name.as_str(),
            target_name: c.target_name.as_str(),
            ind: c.ind.as_slice(),
            weights: c.weights.as_slice(),
            k: c.k,
        })
        .collect();
    let static_view: Vec<sb::AssembleStaticObject> = static_owned
        .iter()
        .map(|so| sb::AssembleStaticObject {
            name: so.name.as_str(),
            vertex_world: so.vertex_world.as_slice(),
            color: so.color,
            faces: so.faces.as_deref(),
            n_faces: so.faces.as_deref().map(|f| f.len()).unwrap_or(0),
            position: so.position,
        })
        .collect();

    let dyn_result = py
        .allow_threads(|| {
            sb::assemble_dyn_scene(
                &dyn_view,
                &map_owned,
                concat_count,
                has_merge,
                &dmap_owned,
                &cs_view,
            )
        })
        .map_err(into_py_err)?;
    let static_result = py
        .allow_threads(|| sb::assemble_static_scene(&static_view, &dmap_owned))
        .map_err(into_py_err)?;

    // ----- pack output dict -----
    let out = PyDict::new(py);

    // Vertex-side (N, 3) arrays.
    let n_global = dyn_result.concat_count;
    let to_n3 = |buf: Vec<f64>, n: usize| -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        ndarray::Array2::from_shape_vec((n, 3), buf)
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))
            .map(|a| a.into_pyarray(py))
    };
    out.set_item("concat_vert", to_n3(dyn_result.concat_vert, n_global)?)?;
    out.set_item("concat_color", to_n3(dyn_result.concat_color, n_global)?)?;
    out.set_item("concat_vel", to_n3(dyn_result.concat_vel, n_global)?)?;
    out.set_item(
        "concat_vert_dmap",
        PyArray1::<u32>::from_vec(py, dyn_result.concat_vert_dmap),
    )?;
    let n_dmap = dyn_result.concat_displacement.len() / 3;
    out.set_item(
        "concat_displacement",
        to_n3(dyn_result.concat_displacement, n_dmap)?,
    )?;

    // Topology arrays.
    let n_rod = dyn_result.concat_rod.len() / 2;
    let rod_arr = ndarray::Array2::from_shape_vec((n_rod, 2), dyn_result.concat_rod)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("concat_rod", rod_arr.into_pyarray(py))?;

    let n_tri = dyn_result.concat_tri.len() / 3;
    let tri_arr = ndarray::Array2::from_shape_vec((n_tri, 3), dyn_result.concat_tri)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("concat_tri", tri_arr.into_pyarray(py))?;

    let n_tet = dyn_result.concat_tet.len() / 4;
    let tet_arr = ndarray::Array2::from_shape_vec((n_tet, 4), dyn_result.concat_tet)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("concat_tet", tet_arr.into_pyarray(py))?;

    // Per-shell-tri auxiliary arrays.
    let n_uv = dyn_result.concat_uv.len() / 6;
    let uv_arr = ndarray::Array2::from_shape_vec((n_uv, 6), dyn_result.concat_uv)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("concat_uv", uv_arr.into_pyarray(py))?;
    out.set_item(
        "concat_dyn_tri_color",
        PyArray1::<u8>::from_vec(py, dyn_result.concat_dyn_tri_color),
    )?;
    out.set_item(
        "concat_dyn_tri_intensity",
        PyArray1::<f64>::from_vec(py, dyn_result.concat_dyn_tri_intensity),
    )?;
    out.set_item(
        "concat_rod_is_collider",
        PyArray1::<u8>::from_vec(py, dyn_result.concat_rod_is_collider),
    )?;
    out.set_item(
        "concat_tri_is_collider",
        PyArray1::<u8>::from_vec(py, dyn_result.concat_tri_is_collider),
    )?;

    // Stitches.
    let n_st = dyn_result.concat_stitch_ind.len() / 4;
    let st_ind_arr =
        ndarray::Array2::from_shape_vec((n_st, 4), dyn_result.concat_stitch_ind)
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("concat_stitch_ind", st_ind_arr.into_pyarray(py))?;
    let st_w_arr = ndarray::Array2::from_shape_vec((n_st, 4), dyn_result.concat_stitch_w)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("concat_stitch_w", st_w_arr.into_pyarray(py))?;

    out.set_item("rod_count", dyn_result.rod_count)?;
    out.set_item("shell_count", dyn_result.shell_count)?;

    // Per-object stats: name -> {rod_added, tri_added, tet_added,
    // is_pure_shell, rod_keep_mask (list[bool]), tri_keep_mask}.
    let stats = PyDict::new(py);
    // Consume `per_object` so we can move keep_masks out without
    // copying. The previous `.to_vec()` clones allocated 2x list<bool>
    // per object on every Scene.build.
    for (name, st) in dyn_result.per_object {
        let entry = PyDict::new(py);
        entry.set_item("rod_added", st.rod_added)?;
        entry.set_item("tri_added", st.tri_added)?;
        entry.set_item("tet_added", st.tet_added)?;
        entry.set_item("is_pure_shell", st.is_pure_shell)?;
        entry.set_item("rod_keep_mask", st.rod_keep_mask)?;
        entry.set_item("tri_keep_mask", st.tri_keep_mask)?;
        stats.set_item(name, entry)?;
    }
    out.set_item("stats_by_name", stats)?;

    // Static.
    let n_sv = static_result.static_vert.len() / 3;
    let n_st_tri = static_result.static_tri.len() / 3;
    let sv_arr = ndarray::Array2::from_shape_vec((n_sv, 3), static_result.static_vert)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("static_vert", sv_arr.into_pyarray(py))?;
    let sc_arr =
        ndarray::Array2::from_shape_vec((n_sv, 3), static_result.static_color)
            .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("static_color", sc_arr.into_pyarray(py))?;
    let st_arr = ndarray::Array2::from_shape_vec((n_st_tri, 3), static_result.static_tri)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    out.set_item("static_tri", st_arr.into_pyarray(py))?;
    out.set_item(
        "static_vert_dmap",
        PyArray1::<u32>::from_vec(py, static_result.static_vert_dmap),
    )?;
    let sper = PyDict::new(py);
    for (name, offset, n_face) in &static_result.per_object {
        let e = PyDict::new(py);
        e.set_item("offset", offset)?;
        e.set_item("n_face", n_face)?;
        sper.set_item(name, e)?;
    }
    out.set_item("static_per_object", sper)?;

    Ok(out)
}

/// Mirrors `Scene.build`'s shell shrink/strain-limit conflict check.
/// Returns `(face_idx, shrink_x, shrink_y, strain_limit)` when a
/// conflict is detected; returns `None` when there is no conflict.
#[pyfunction]
#[pyo3(signature = (shrink_x, shrink_y, strain_limit))]
pub(super) fn scene_shell_shrink_strain_limit_conflict(
    shrink_x: Vec<f64>,
    shrink_y: Vec<f64>,
    strain_limit: Vec<f64>,
) -> Option<(usize, f64, f64, f64)> {
    sb::check_shell_shrink_strain_limit_conflict(&shrink_x, &shrink_y, &strain_limit)
}
