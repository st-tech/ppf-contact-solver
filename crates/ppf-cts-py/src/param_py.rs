// File: crates/ppf-cts-py/src/param_py.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 bindings for `core::datamodel::params`. The frontend `ParamHolder`
// dispatches `set/get/get_desc/clear_all/key_list/items/copy` through
// here.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

use ppf_cts_core::datamodel::params::{
    app_param as core_app_param, object_param as core_object_param, ObjectKind, ParamEntry,
    ParamHolder as CoreParamHolder, ParamValue,
};

/// Coerce a Python value into the heterogeneous `ParamValue`. Mirrors
/// the union of types `frontend/_param_.py` actually emits: bool, int,
/// float, list-of-3-floats (gravity / wind), and str.
pub(crate) fn pyany_to_param_value(value: &Bound<'_, PyAny>) -> PyResult<ParamValue> {
    // Bool must come before int (PyBool is a subclass of PyInt).
    if let Ok(b) = value.extract::<bool>() {
        // Reject the case where Python passed an actual `int` whose
        // value is 0 or 1; `extract::<bool>` would also succeed there.
        // We disambiguate via the Python type.
        let ty = value.get_type();
        if ty.name()? == "bool" {
            return Ok(ParamValue::Bool(b));
        }
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(ParamValue::Int(i));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(ParamValue::Float(f));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(ParamValue::String(s));
    }
    // Sequence of three floats (gravity / wind).
    if let Ok(seq) = value.extract::<Vec<f64>>() {
        if seq.len() == 3 {
            return Ok(ParamValue::Vec3([seq[0], seq[1], seq[2]]));
        }
    }
    Err(PyValueError::new_err(
        "unsupported parameter value type (expected bool, int, float, str, or [f64; 3])",
    ))
}

fn param_value_to_pyobject(py: Python<'_>, v: &ParamValue) -> PyResult<PyObject> {
    use pyo3::IntoPyObjectExt;
    match v {
        ParamValue::Bool(b) => b.into_py_any(py),
        ParamValue::Int(i) => i.into_py_any(py),
        ParamValue::Float(f) => f.into_py_any(py),
        ParamValue::String(s) => s.into_py_any(py),
        // Mirror the Python source: `gravity` and `wind` are stored as
        // `list[float]` (frontend/_param_.py:156-157).
        ParamValue::Vec3(v) => PyList::new(py, v)?.into_py_any(py),
    }
}

/// PyO3 wrapper around `core::datamodel::params::ParamHolder`. Stores
/// the (default, name, desc) entries the frontend dispatches against.
#[pyclass(name = "ParamHolder", module = "_ppf_cts_py")]
pub struct ParamHolder {
    inner: CoreParamHolder,
}

#[pymethods]
impl ParamHolder {
    /// Build from a Python dict in the same `(default, name, desc)`
    /// shape `app_param()` produces. This lets callers seed a holder
    /// with custom keys (the test suite spelled out a fake holder).
    #[new]
    fn new(param: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut entries: BTreeMap<String, ParamEntry> = BTreeMap::new();
        for (key, value) in param.iter() {
            let k: String = key.extract()?;
            let tup = value.downcast::<PyTuple>().map_err(|_| {
                PyValueError::new_err(
                    "ParamHolder: each value must be a (default, display_name, description) tuple",
                )
            })?;
            if tup.len() != 3 {
                return Err(PyValueError::new_err(
                    "ParamHolder: tuple must have exactly 3 elements",
                ));
            }
            let default = pyany_to_param_value(&tup.get_item(0)?)?;
            let display_name: String = tup.get_item(1)?.extract()?;
            let description: String = tup.get_item(2)?.extract()?;
            entries.insert(
                k,
                ParamEntry {
                    default: default.clone(),
                    value: default,
                    display_name,
                    description,
                },
            );
        }
        Ok(Self {
            inner: CoreParamHolder::new(entries),
        })
    }

    fn clear_all(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.clear_all();
        slf
    }

    fn set<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let pv = pyany_to_param_value(value)?;
        slf.inner
            .set(key, pv)
            .map_err(crate::errors::into_py_err)?;
        Ok(slf)
    }

    fn get(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        let v = self
            .inner
            .get(key)
            .map_err(crate::errors::into_py_err)?;
        param_value_to_pyobject(py, v)
    }

    fn get_desc(&self, key: &str) -> PyResult<(String, String)> {
        let (n, d) = self
            .inner
            .get_desc(key)
            .map_err(crate::errors::into_py_err)?;
        Ok((n.to_string(), d.to_string()))
    }

    fn key_list(&self) -> Vec<String> {
        self.inner.key_list().into_iter().map(String::from).collect()
    }

    fn items(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
        self.inner
            .items()
            .into_iter()
            .map(|(k, v)| param_value_to_pyobject(py, v).map(|po| (k.to_string(), po)))
            .collect()
    }

    /// Mirror Python's `ParamHolder.copy()`. Returns a fresh holder
    /// whose entries are independent of `self`.
    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Module-level factory: `_ppf_cts_py.app_param() -> ParamHolder` with
/// every default identical to `frontend/_param_.py:app_param()`.
#[pyfunction]
fn app_param() -> ParamHolder {
    ParamHolder {
        inner: core_app_param(),
    }
}

/// Build the schema dict consumed by `frontend/_param_.py:app_param()`:
/// `{key: (default_value, display_name, description)}`. Mirrors the
/// Python dict literal byte-for-byte. The Python `ParamHolder.__init__`
/// consumes this directly.
#[pyfunction]
fn app_param_dict(py: Python<'_>) -> PyResult<Py<PyDict>> {
    holder_to_schema_dict(py, &core_app_param())
}

/// Same as `app_param_dict` but for the per-object schema. `kind`
/// must be one of `"tri"`, `"tet"`, `"rod"`.
#[pyfunction]
fn object_param_dict(py: Python<'_>, kind: &str) -> PyResult<Py<PyDict>> {
    let k = ObjectKind::from_str(kind).map_err(crate::errors::into_py_err)?;
    holder_to_schema_dict(py, &core_object_param(k))
}

fn holder_to_schema_dict(py: Python<'_>, holder: &CoreParamHolder) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    // `holder.entries` is private; rebuild via the public iterator on
    // `key_list()` plus `get_desc` and `get` (which returns the value;
    // value == default at construction time).
    for key in holder.key_list() {
        // We need the default value, not the (possibly mutated) value;
        // newly-constructed holders have value == default, so reading
        // via `get` is safe here.
        let v = holder.get(key).map_err(crate::errors::into_py_err)?;
        let (display_name, description) = holder
            .get_desc(key)
            .map_err(crate::errors::into_py_err)?;
        use pyo3::IntoPyObjectExt;
        let py_value = param_value_to_pyobject(py, v)?;
        let tup = PyTuple::new(
            py,
            &[
                py_value,
                display_name.into_py_any(py)?,
                description.into_py_any(py)?,
            ],
        )?;
        d.set_item(key, tup)?;
    }
    Ok(d.unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParamHolder>()?;
    m.add_function(wrap_pyfunction!(app_param, m)?)?;
    m.add_function(wrap_pyfunction!(app_param_dict, m)?)?;
    m.add_function(wrap_pyfunction!(object_param_dict, m)?)?;
    Ok(())
}
