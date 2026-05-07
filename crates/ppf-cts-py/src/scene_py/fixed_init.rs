// File: crates/ppf-cts-py/src/scene_py/fixed_init.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// FixedScene.__init__ assembly: drives self-intersection / contact-offset /
// wall / sphere validation and the post-success derived data
// (`_area`, `_face_to_vert_weights`). Mirrors
// `frontend/_scene_.py:FixedScene.__init__` lines 1594-1819.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ppf_cts_core::kernels::fixed_scene_assemble as fsa;

#[inline]
fn fsa_read_u32_1d(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u32>> {
    crate::utils_py::read_index_array_1d_u32(arr, name)
}

fn fsa_read_i32_2d_flat(
    arr: &Bound<'_, PyAny>,
    cols: usize,
    name: &str,
) -> PyResult<Vec<i32>> {
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i32>>() {
        let s = view.shape();
        if s.len() != 2 || (s[0] > 0 && s[1] != cols) {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (M, {cols}), got {s:?}"
            )));
        }
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .to_vec());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, i64>>() {
        let s = view.shape();
        if s.len() != 2 || (s[0] > 0 && s[1] != cols) {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (M, {cols}), got {s:?}"
            )));
        }
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .iter()
            .map(|&v| v as i32)
            .collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, u32>>() {
        let s = view.shape();
        if s.len() != 2 || (s[0] > 0 && s[1] != cols) {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (M, {cols}), got {s:?}"
            )));
        }
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .iter()
            .map(|&v| v as i32)
            .collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, u64>>() {
        let s = view.shape();
        if s.len() != 2 || (s[0] > 0 && s[1] != cols) {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (M, {cols}), got {s:?}"
            )));
        }
        return Ok(view
            .as_slice()
            .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
            .iter()
            .map(|&v| v as i32)
            .collect());
    }
    if let Ok(view) = arr.extract::<PyReadonlyArray2<'_, f64>>() {
        let s = view.shape();
        if s[0] == 0 {
            return Ok(Vec::new());
        }
        return Err(PyValueError::new_err(format!(
            "{name} float dtype only allowed when rows == 0; got shape {s:?}"
        )));
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a 2-D numpy array of int dtype"
    )))
}

fn fsa_read_bool_1d(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    let view = arr
        .extract::<PyReadonlyArray1<'_, bool>>()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be a 1-D bool numpy array")))?;
    Ok(view
        .as_slice()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
        .to_vec())
}

fn fsa_read_f64_2d_flat(
    arr: &Bound<'_, PyAny>,
    cols: usize,
    name: &str,
) -> PyResult<Vec<f64>> {
    let view = arr.extract::<PyReadonlyArray2<'_, f64>>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a 2-D float64 numpy array"))
    })?;
    let s = view.shape();
    if s.len() != 2 || (s[0] > 0 && s[1] != cols) {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (M, {cols}), got {s:?}"
        )));
    }
    Ok(view
        .as_slice()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be C-contiguous")))?
        .to_vec())
}

/// FixedScene.__init__ body, ported. Takes the constructor inputs as
/// flat numpy arrays + small Python lists, runs every check, and
/// returns a Python dict the caller assigns to `self.<attr>`.
///
/// Returned dict keys:
///   * `has_self_intersection`, `has_contact_offset_violation`,
///     `has_wall_violation`, `has_sphere_violation`: bool flags.
///   * `combined_message`: joined human-readable violation text.
///   * `violations`: list[dict] mirroring
///     `ValidationError.violations` (truncated to first 100 per kind).
///   * `area`: `(M,)` float64 ndarray (empty when no triangles or
///     when validation failed).
///   * `face_to_vert_weights`: `(N,)` float64 ndarray, or `None` when
///     `has_dyn_color == False`.
#[pyfunction]
#[pyo3(signature = (
    vert_dmap,
    vert_local,
    displacement,
    tri,
    rod,
    tri_is_collider,
    rod_is_collider,
    tri_offset,
    rod_offset,
    static_verts,
    static_tris,
    pinned_vertices,
    walls,
    spheres,
    has_dyn_color,
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn scene_fixed_scene_assemble<'py>(
    py: Python<'py>,
    vert_dmap: &Bound<'py, PyAny>,
    vert_local: PyReadonlyArray2<'py, f64>,
    displacement: PyReadonlyArray2<'py, f64>,
    tri: &Bound<'py, PyAny>,
    rod: &Bound<'py, PyAny>,
    tri_is_collider: &Bound<'py, PyAny>,
    rod_is_collider: &Bound<'py, PyAny>,
    tri_offset: Vec<f64>,
    rod_offset: Vec<f64>,
    static_verts: Option<&Bound<'py, PyAny>>,
    static_tris: Option<&Bound<'py, PyAny>>,
    pinned_vertices: Vec<usize>,
    walls: Vec<([f64; 3], [f64; 3])>,
    spheres: Vec<([f64; 3], f64, bool, bool)>,
    has_dyn_color: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let vl_shape = vert_local.shape();
    if vl_shape.len() != 2 || vl_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "vert_local must be (N, 3), got {vl_shape:?}"
        )));
    }
    let vl = vert_local
        .as_slice()
        .map_err(|_| PyTypeError::new_err("vert_local must be C-contiguous"))?
        .to_vec();
    let disp_shape = displacement.shape();
    if disp_shape.len() != 2 || disp_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "displacement must be (D, 3), got {disp_shape:?}"
        )));
    }
    let disp = displacement
        .as_slice()
        .map_err(|_| PyTypeError::new_err("displacement must be C-contiguous"))?
        .to_vec();
    let dmap = fsa_read_u32_1d(vert_dmap, "vert_dmap")?;
    let tri_flat = fsa_read_i32_2d_flat(tri, 3, "tri")?;
    let rod_flat = fsa_read_i32_2d_flat(rod, 2, "rod")?;
    let tri_coll = fsa_read_bool_1d(tri_is_collider, "tri_is_collider")?;
    let rod_coll = fsa_read_bool_1d(rod_is_collider, "rod_is_collider")?;
    let static_verts_vec: Option<Vec<f64>> = match static_verts {
        Some(arr) => Some(fsa_read_f64_2d_flat(arr, 3, "static_verts")?),
        None => None,
    };
    let static_tris_vec: Option<Vec<i32>> = match static_tris {
        Some(arr) => Some(fsa_read_i32_2d_flat(arr, 3, "static_tris")?),
        None => None,
    };

    let walls_rs: Vec<fsa::WallEntry> = walls
        .into_iter()
        .map(|(pos, normal)| fsa::WallEntry { pos, normal })
        .collect();
    let spheres_rs: Vec<fsa::SphereEntry> = spheres
        .into_iter()
        .map(|(pos, radius, is_inverted, is_hemisphere)| fsa::SphereEntry {
            pos,
            radius,
            is_inverted,
            is_hemisphere,
        })
        .collect();

    let result = py.allow_threads(|| {
        fsa::fixed_scene_assemble(fsa::AssembleInput {
            vert_dmap: &dmap,
            vert_local: &vl,
            displacement: &disp,
            tri: &tri_flat,
            rod: &rod_flat,
            tri_is_collider: &tri_coll,
            rod_is_collider: &rod_coll,
            tri_offset: &tri_offset,
            rod_offset: &rod_offset,
            static_verts: static_verts_vec.as_deref(),
            static_tris: static_tris_vec.as_deref(),
            pinned_vertices: &pinned_vertices,
            walls: &walls_rs,
            spheres: &spheres_rs,
            has_dyn_color,
        })
    });

    let out = match result {
        Ok(o) => o,
        Err(fsa::AssembleError::RodTriOffset(v)) => {
            use ppf_cts_core::kernels::scene_build::RodTriOffsetViolation as V;
            return Err(match v {
                V::EdgeShorterThanOffset { offset, edge_len } => PyValueError::new_err(format!(
                    "Contact offset ({offset:.4}) exceeds rod edge length ({edge_len:.4}). \
                     Reduce the offset or increase mesh resolution."
                )),
                V::VertexInsideOffset { dist, required, .. } => PyValueError::new_err(format!(
                    "Rod vertex within contact offset of triangle (dist={dist:.4} < \
                     offset={required:.4}). Reduce contact-offset or move the curve away."
                )),
            });
        }
    };

    let violations = PyList::empty(py);
    if out.has_self_intersection {
        let d = PyDict::new(py);
        d.set_item("type", "self_intersection")?;
        let tris_list = PyList::empty(py);
        for entry in &out.self_intersections {
            let tri_positions = PyList::empty(py);
            for chunk in entry.tri_positions.chunks(3) {
                let chunk_list = PyList::empty(py);
                for v in chunk {
                    chunk_list.append(PyList::new(py, v)?)?;
                }
                tri_positions.append(chunk_list)?;
            }
            tris_list.append(tri_positions)?;
        }
        d.set_item("tris", tris_list)?;
        d.set_item("count", out.n_self_intersections_total)?;
        violations.append(d)?;
    }
    if out.has_contact_offset_violation {
        let d = PyDict::new(py);
        d.set_item("type", "contact_offset")?;
        let pairs_list = PyList::empty(py);
        for entry in &out.contact_offset_pairs {
            let p = PyDict::new(py);
            p.set_item(
                "ei_type",
                if entry.ei_is_triangle { "triangle" } else { "edge" },
            )?;
            p.set_item(
                "ej_type",
                if entry.ej_is_triangle { "triangle" } else { "edge" },
            )?;
            let ei_pos = PyList::empty(py);
            for v in &entry.ei_pos {
                ei_pos.append(PyList::new(py, v)?)?;
            }
            let ej_pos = PyList::empty(py);
            for v in &entry.ej_pos {
                ej_pos.append(PyList::new(py, v)?)?;
            }
            p.set_item("ei_pos", ei_pos)?;
            p.set_item("ej_pos", ej_pos)?;
            pairs_list.append(p)?;
        }
        d.set_item("pairs", pairs_list)?;
        d.set_item("count", out.n_contact_offset_total)?;
        violations.append(d)?;
    }
    if out.has_wall_violation {
        let d = PyDict::new(py);
        d.set_item("type", "wall")?;
        let verts_list = PyList::empty(py);
        for entry in &out.wall_vertices {
            let p = PyDict::new(py);
            p.set_item("pos", PyList::new(py, entry.pos)?)?;
            p.set_item("wall", entry.wall_idx)?;
            p.set_item("dist", entry.signed_dist)?;
            verts_list.append(p)?;
        }
        d.set_item("vertices", verts_list)?;
        d.set_item("count", out.n_wall_total)?;
        violations.append(d)?;
    }
    if out.has_sphere_violation {
        let d = PyDict::new(py);
        d.set_item("type", "sphere")?;
        let verts_list = PyList::empty(py);
        for entry in &out.sphere_vertices {
            let p = PyDict::new(py);
            p.set_item("pos", PyList::new(py, entry.pos)?)?;
            p.set_item("sphere", entry.sphere_idx)?;
            p.set_item("dist", entry.dist)?;
            verts_list.append(p)?;
        }
        d.set_item("vertices", verts_list)?;
        d.set_item("count", out.n_sphere_total)?;
        violations.append(d)?;
    }

    let result_dict = PyDict::new(py);
    result_dict.set_item("has_self_intersection", out.has_self_intersection)?;
    result_dict.set_item(
        "has_contact_offset_violation",
        out.has_contact_offset_violation,
    )?;
    result_dict.set_item("has_wall_violation", out.has_wall_violation)?;
    result_dict.set_item("has_sphere_violation", out.has_sphere_violation)?;
    result_dict.set_item("combined_message", out.combined_message)?;
    result_dict.set_item("violations", violations)?;
    let area_arr = PyArray1::<f64>::from_slice(py, &out.area);
    result_dict.set_item("area", area_arr)?;
    match out.face_to_vert_weights {
        Some(w) => {
            let arr = PyArray1::<f64>::from_slice(py, &w);
            result_dict.set_item("face_to_vert_weights", arr)?;
        }
        None => {
            result_dict.set_item("face_to_vert_weights", py.None())?;
        }
    }

    Ok(result_dict)
}
