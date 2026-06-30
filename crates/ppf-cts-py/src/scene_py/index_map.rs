// File: crates/ppf-cts-py/src/scene_py/index_map.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 binding for `scene_build::build_index_map` (Scene.build core
// loop). Assigns rod/shell/finalize global indices for every object in
// the scene.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ppf_cts_core::kernels::scene_build as sb;

use super::helpers::{read_edges_u32, read_faces_u32, read_tets_u32};
use crate::errors::into_py_err;

/// Build the per-object `local -> global` vertex index map.
///
/// Args:
///   * `objects`: list of dicts with keys `name: str`, `n_verts: int`,
///     and optional `edges`, `faces`, `tets` numpy arrays.
///
/// Returns a dict:
///   * `map_by_name`: `{name: ndarray int64 (n_verts,)}`
///   * `rod_vert_range`: `(int, int)`
///   * `shell_vert_range`: `(int, int)`
///   * `concat_count`: int
#[pyfunction]
#[pyo3(signature = (objects))]
pub(super) fn scene_build_index_map<'py>(
    py: Python<'py>,
    objects: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    // Decode objects. We hold owned Vec<[u32; N]> per object so the
    // borrows we hand to the kernel outlive the call.
    struct Owned {
        name: String,
        n_verts: usize,
        edges: Option<Vec<[u32; 2]>>,
        faces: Option<Vec<[u32; 3]>>,
        tets: Option<Vec<[u32; 4]>>,
    }
    let mut owned: Vec<Owned> = Vec::with_capacity(objects.len());
    for obj in objects.iter() {
        let d = obj.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("each object must be a dict with name/n_verts/...")
        })?;
        let name: String = d
            .get_item("name")?
            .ok_or_else(|| PyValueError::new_err("missing 'name'"))?
            .extract()?;
        let n_verts: usize = d
            .get_item("n_verts")?
            .ok_or_else(|| PyValueError::new_err("missing 'n_verts'"))?
            .extract()?;
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
        owned.push(Owned { name, n_verts, edges, faces, tets });
    }

    // Build the borrow-view structs the kernel wants.
    let view: Vec<sb::IndexMapObject> = owned
        .iter()
        .map(|o| sb::IndexMapObject {
            name: o.name.as_str(),
            n_verts: o.n_verts,
            edges: o.edges.as_deref(),
            faces: o.faces.as_deref(),
            tets: o.tets.as_deref(),
        })
        .collect();

    let result = py
        .allow_threads(|| sb::build_index_map(&view))
        .map_err(into_py_err)?;

    // Pack output dict.
    let out = PyDict::new(py);
    let map_dict = PyDict::new(py);
    for (name, m) in &result.map_by_name {
        let arr = PyArray1::<i64>::from_slice(py, m);
        map_dict.set_item(name, arr)?;
    }
    out.set_item("map_by_name", map_dict)?;
    out.set_item(
        "rod_vert_range",
        (result.rod_vert_range.0, result.rod_vert_range.1),
    )?;
    out.set_item(
        "shell_vert_range",
        (result.shell_vert_range.0, result.shell_vert_range.1),
    )?;
    out.set_item("concat_count", result.concat_count)?;
    Ok(out)
}
