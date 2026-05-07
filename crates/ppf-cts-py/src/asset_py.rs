// File: crates/ppf-cts-py/src/asset_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for the asset registry. Two layers live here:
//
//   1. Free functions `check_bounds`, `check_cols`, `check_tri_v_cols`
//      that mirror the Python-side validators in
//      `frontend/_asset_.py:AssetUploader.check_*`.
//
//   2. A `AssetRegistry` class that owns the `name -> mesh tuple`
//      storage. Storage is `HashMap<String, AssetEntry>` where
//      `AssetEntry` keeps the user-supplied numpy arrays as
//      `Py<PyAny>`. This
//      preserves the original dtype byte-for-byte: an `np.float64`
//      vertex buffer reads back as `np.float64`, an `np.int32` face
//      buffer reads back as `np.int32`. The core `AssetRegistry` in
//      `ppf_cts_core::datamodel::asset` enforces a canonical
//      `Array2<f32>/Array2<u32>` layout, which would silently re-type
//      user input; we deliberately don't route through it for storage
//      and use it only for its tested validation helpers when needed.

use std::collections::HashMap;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// Find the maximum value in a 2D numpy index array of integer dtype,
/// then return `Err` if it is `>= n_verts`. Mirrors
/// `frontend/_asset_.py::AssetUploader.check_bounds`.
///
/// Accepts any integer-typed numpy array (uint8/16/32/64,
/// int8/16/32/64) by trying contiguous extraction in priority order.
#[pyfunction]
#[pyo3(signature = (e, n_verts))]
pub fn check_bounds(e: &Bound<'_, PyAny>, n_verts: usize) -> PyResult<()> {
    let max_idx = max_index_of_array(e)?;
    if max_idx >= n_verts as i128 {
        return Err(PyValueError::new_err(format!(
            "E contains index {max_idx} out of bounds ({n_verts})"
        )));
    }
    Ok(())
}

fn max_index_of_array(e: &Bound<'_, PyAny>) -> PyResult<i128> {
    if let Ok(a) = e.extract::<PyReadonlyArray2<u32>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<PyReadonlyArray2<i32>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<PyReadonlyArray2<i64>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<PyReadonlyArray2<u64>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<PyReadonlyArray2<i16>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<PyReadonlyArray2<u16>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<PyReadonlyArray2<i8>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<PyReadonlyArray2<u8>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    // Fallback: 1D arrays in any of the above types (Python `np.max`
    // doesn't care about ndim).
    if let Ok(a) = e.extract::<numpy::PyReadonlyArray1<u32>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<numpy::PyReadonlyArray1<i32>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    if let Ok(a) = e.extract::<numpy::PyReadonlyArray1<i64>>() {
        return Ok(a.as_array().iter().copied().max().unwrap_or(0) as i128);
    }
    Err(PyTypeError::new_err(
        "check_bounds: unsupported array dtype (expected integer)",
    ))
}

/// Validate the column shape of a 2D numpy array. Used by
/// `AssetUploader.tri/tet/rod/stitch` which raise plain `Exception` on
/// mismatch in the Python source; we mirror the message so callers
/// asserting on it keep working.
#[pyfunction]
#[pyo3(signature = (arr, name, expected_cols))]
pub fn check_cols(arr: &Bound<'_, PyAny>, name: &str, expected_cols: usize) -> PyResult<()> {
    let cols = ncols_of_array(arr)?;
    if cols != expected_cols {
        return Err(PyValueError::new_err(format!(
            "{name} must have {expected_cols} columns"
        )));
    }
    Ok(())
}

fn ncols_of_array(arr: &Bound<'_, PyAny>) -> PyResult<usize> {
    // Use the untyped numpy interface so we don't have to enumerate
    // every dtype here; we just need shape[1].
    let untyped = arr.downcast::<numpy::PyUntypedArray>().map_err(|_| {
        PyTypeError::new_err("expected a numpy.ndarray")
    })?;
    let shape = untyped.shape();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "array must be 2-dimensional (got 1D)",
        ));
    }
    Ok(shape[1])
}


/// Validate `V` for `AssetUploader.tri`: must have 3 or 5 columns.
/// Returns the column count so the Python wrapper can branch on the
/// 3-vs-5 case.
#[pyfunction]
pub fn check_tri_v_cols(v: &Bound<'_, PyAny>) -> PyResult<usize> {
    let cols = ncols_of_array(v)?;
    if cols != 3 && cols != 5 {
        return Err(PyValueError::new_err("V must have 3 or 5 columns"));
    }
    Ok(cols)
}

// ---------------------------------------------------------------------------
// Rust-backed AssetRegistry. Replaces the Python `AssetManager._mesh`
// dict. Stores user numpy arrays as `Py<PyAny>` so dtypes round-trip
// exactly.

#[derive(Debug)]
enum AssetEntry {
    Tri {
        v: Py<PyAny>,
        f: Py<PyAny>,
        uv: Option<Py<PyAny>>,
    },
    Tet {
        v: Py<PyAny>,
        f: Py<PyAny>,
        t: Py<PyAny>,
    },
    Rod {
        v: Py<PyAny>,
        e: Py<PyAny>,
    },
    Stitch {
        ind: Py<PyAny>,
        w: Py<PyAny>,
    },
}

impl AssetEntry {
    fn kind(&self) -> &'static str {
        match self {
            AssetEntry::Tri { .. } => "tri",
            AssetEntry::Tet { .. } => "tet",
            AssetEntry::Rod { .. } => "rod",
            AssetEntry::Stitch { .. } => "stitch",
        }
    }
}

/// Rust-side asset store. Keys are asset names, values are
/// `AssetEntry` variants holding owned `Py<PyAny>` references to the
/// caller's numpy arrays. We never re-type; the caller's dtype is
/// what comes back out. Validation (column counts, bounds) is the
/// caller's responsibility, performed via the free `check_*`
/// functions before `add_*`.
#[pyclass(name = "AssetRegistry", module = "_ppf_cts_py")]
pub struct AssetRegistry {
    assets: HashMap<String, AssetEntry>,
}

#[pymethods]
impl AssetRegistry {
    #[new]
    fn new() -> Self {
        Self { assets: HashMap::new() }
    }

    fn list(&self) -> Vec<String> {
        self.assets.keys().cloned().collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.assets.contains_key(name)
    }

    fn remove(&mut self, name: &str) -> bool {
        self.assets.remove(name).is_some()
    }

    fn clear(&mut self) {
        self.assets.clear();
    }

    fn get_type(&self, name: &str) -> PyResult<&'static str> {
        match self.assets.get(name) {
            Some(e) => Ok(e.kind()),
            None => Err(PyKeyError::new_err(format!("Asset {name} does not exist"))),
        }
    }

    /// Generic dict accessor mirroring `AssetFetcher.get`.
    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let entry = self.assets.get(name).ok_or_else(|| {
            PyKeyError::new_err(format!("Asset {name} does not exist"))
        })?;
        let d = PyDict::new(py);
        match entry {
            AssetEntry::Tri { v, f, uv } => {
                d.set_item("V", v.clone_ref(py))?;
                d.set_item("F", f.clone_ref(py))?;
                if let Some(uv) = uv {
                    d.set_item("UV", uv.clone_ref(py))?;
                }
            }
            AssetEntry::Tet { v, f, t } => {
                d.set_item("V", v.clone_ref(py))?;
                d.set_item("F", f.clone_ref(py))?;
                d.set_item("T", t.clone_ref(py))?;
            }
            AssetEntry::Rod { v, e } => {
                d.set_item("V", v.clone_ref(py))?;
                d.set_item("E", e.clone_ref(py))?;
            }
            AssetEntry::Stitch { ind, w } => {
                d.set_item("Ind", ind.clone_ref(py))?;
                d.set_item("W", w.clone_ref(py))?;
            }
        }
        Ok(d)
    }

    #[pyo3(signature = (name, v, f, uv=None))]
    fn add_tri(
        &mut self,
        name: &str,
        v: Py<PyAny>,
        f: Py<PyAny>,
        uv: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if self.assets.contains_key(name) {
            return Err(PyValueError::new_err(format!("name '{name}' already exists")));
        }
        self.assets.insert(name.to_string(), AssetEntry::Tri { v, f, uv });
        Ok(())
    }


    fn add_tet(
        &mut self,
        name: &str,
        v: Py<PyAny>,
        f: Py<PyAny>,
        t: Py<PyAny>,
    ) -> PyResult<()> {
        if self.assets.contains_key(name) {
            return Err(PyValueError::new_err(format!("name '{name}' already exists")));
        }
        self.assets.insert(name.to_string(), AssetEntry::Tet { v, f, t });
        Ok(())
    }

    fn add_rod(&mut self, name: &str, v: Py<PyAny>, e: Py<PyAny>) -> PyResult<()> {
        if self.assets.contains_key(name) {
            return Err(PyValueError::new_err(format!("name '{name}' already exists")));
        }
        self.assets.insert(name.to_string(), AssetEntry::Rod { v, e });
        Ok(())
    }

    fn add_stitch(&mut self, name: &str, ind: Py<PyAny>, w: Py<PyAny>) -> PyResult<()> {
        if self.assets.contains_key(name) {
            return Err(PyValueError::new_err(format!("name '{name}' already exists")));
        }
        self.assets.insert(name.to_string(), AssetEntry::Stitch { ind, w });
        Ok(())
    }

    /// Snapshot the registry to a `dict[name, {"kind": str, "arrays":
    /// dict[str, ndarray]}]` shape suitable for pickle. Mirrors the
    /// loop body of `frontend/_asset_.py:AssetManager.__getstate__`.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, entry) in self.assets.iter() {
            let body = PyDict::new(py);
            body.set_item("kind", entry.kind())?;
            let arrays = PyDict::new(py);
            match entry {
                AssetEntry::Tri { v, f, uv } => {
                    arrays.set_item("V", v.clone_ref(py))?;
                    arrays.set_item("F", f.clone_ref(py))?;
                    if let Some(uv) = uv {
                        arrays.set_item("UV", uv.clone_ref(py))?;
                    }
                }
                AssetEntry::Tet { v, f, t } => {
                    arrays.set_item("V", v.clone_ref(py))?;
                    arrays.set_item("F", f.clone_ref(py))?;
                    arrays.set_item("T", t.clone_ref(py))?;
                }
                AssetEntry::Rod { v, e } => {
                    arrays.set_item("V", v.clone_ref(py))?;
                    arrays.set_item("E", e.clone_ref(py))?;
                }
                AssetEntry::Stitch { ind, w } => {
                    arrays.set_item("Ind", ind.clone_ref(py))?;
                    arrays.set_item("W", w.clone_ref(py))?;
                }
            }
            body.set_item("arrays", arrays)?;
            out.set_item(name, body)?;
        }
        Ok(out)
    }

    /// Restore the registry from a `snapshot()` dict. Replays each
    /// entry through the typed adders so the same bounds-checking
    /// path runs as on a fresh upload. Mirrors the dispatch loop of
    /// `frontend/_asset_.py:AssetManager.__setstate__`.
    fn restore<'py>(
        &mut self,
        _py: Python<'py>,
        snapshot: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        for (name_obj, body_obj) in snapshot.iter() {
            let name: String = name_obj.extract()?;
            let body = body_obj.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("snapshot entry must be a dict")
            })?;
            let kind: String = body
                .get_item("kind")?
                .ok_or_else(|| PyKeyError::new_err("snapshot entry missing 'kind'"))?
                .extract()?;
            let arrays_any = body
                .get_item("arrays")?
                .ok_or_else(|| PyKeyError::new_err("snapshot entry missing 'arrays'"))?;
            let arrays = arrays_any.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("snapshot 'arrays' must be a dict")
            })?;
            match kind.as_str() {
                "tri" => {
                    let v = arrays
                        .get_item("V")?
                        .ok_or_else(|| PyKeyError::new_err("tri missing 'V'"))?
                        .unbind();
                    let f = arrays
                        .get_item("F")?
                        .ok_or_else(|| PyKeyError::new_err("tri missing 'F'"))?
                        .unbind();
                    let uv = arrays.get_item("UV")?.map(|x| x.unbind());
                    self.add_tri(&name, v, f, uv)?;
                }
                "tet" => {
                    let v = arrays
                        .get_item("V")?
                        .ok_or_else(|| PyKeyError::new_err("tet missing 'V'"))?
                        .unbind();
                    let f = arrays
                        .get_item("F")?
                        .ok_or_else(|| PyKeyError::new_err("tet missing 'F'"))?
                        .unbind();
                    let t = arrays
                        .get_item("T")?
                        .ok_or_else(|| PyKeyError::new_err("tet missing 'T'"))?
                        .unbind();
                    self.add_tet(&name, v, f, t)?;
                }
                "rod" => {
                    let v = arrays
                        .get_item("V")?
                        .ok_or_else(|| PyKeyError::new_err("rod missing 'V'"))?
                        .unbind();
                    let e = arrays
                        .get_item("E")?
                        .ok_or_else(|| PyKeyError::new_err("rod missing 'E'"))?
                        .unbind();
                    self.add_rod(&name, v, e)?;
                }
                "stitch" => {
                    let ind = arrays
                        .get_item("Ind")?
                        .ok_or_else(|| PyKeyError::new_err("stitch missing 'Ind'"))?
                        .unbind();
                    let w = arrays
                        .get_item("W")?
                        .ok_or_else(|| PyKeyError::new_err("stitch missing 'W'"))?
                        .unbind();
                    self.add_stitch(&name, ind, w)?;
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown asset kind: {other}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Return `(V, F)` for a tri asset; raises if missing or wrong kind.
    fn get_tri(&self, py: Python<'_>, name: &str) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match self.assets.get(name) {
            None => Err(PyKeyError::new_err(format!("Tri {name} does not exist"))),
            Some(AssetEntry::Tri { v, f, .. }) => Ok((v.clone_ref(py), f.clone_ref(py))),
            Some(_) => Err(PyValueError::new_err(format!("Tri {name} is not a valid"))),
        }
    }

    fn get_tet(
        &self,
        py: Python<'_>,
        name: &str,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        match self.assets.get(name) {
            None => Err(PyKeyError::new_err(format!("Tet {name} does not exist"))),
            Some(AssetEntry::Tet { v, f, t }) => Ok((
                v.clone_ref(py),
                f.clone_ref(py),
                t.clone_ref(py),
            )),
            Some(_) => Err(PyValueError::new_err(format!("Tet {name} is not a valid"))),
        }
    }

    fn get_rod(&self, py: Python<'_>, name: &str) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match self.assets.get(name) {
            None => Err(PyKeyError::new_err(format!("Rod {name} does not exist"))),
            Some(AssetEntry::Rod { v, e }) => Ok((v.clone_ref(py), e.clone_ref(py))),
            Some(_) => Err(PyValueError::new_err(format!("Rod {name} is not a valid"))),
        }
    }

    fn get_stitch(&self, py: Python<'_>, name: &str) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match self.assets.get(name) {
            None => Err(PyKeyError::new_err(format!("Stitch {name} does not exist"))),
            Some(AssetEntry::Stitch { ind, w }) => Ok((ind.clone_ref(py), w.clone_ref(py))),
            Some(_) => Err(PyValueError::new_err(format!("Stitch {name} is not a valid"))),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(check_cols, m)?)?;
    m.add_function(wrap_pyfunction!(check_tri_v_cols, m)?)?;
    m.add_class::<AssetRegistry>()?;
    Ok(())
}
