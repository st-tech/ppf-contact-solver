// File: crates/ppf-cts-py/src/scene_py/helpers.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Shared sub-module helpers for `scene_py`. These are thin wrappers over
// `crate::utils_py::read_index_array_chunked` that fix the column count
// for the three common topology shapes (rod edges, triangle faces, and
// tetrahedra). They are reused by every sibling sub-module that has to
// decode topology arrays from Python.

use pyo3::prelude::*;

/// Read a `(N, 3)` u32-like ndarray into a `Vec<[u32; 3]>`. Accepts
/// numpy arrays of dtype int32, int64, uint32, or uint64. Conversion
/// happens eagerly; the kernel runs after the GIL is released.
#[inline]
pub(super) fn read_faces_u32(arr: &Bound<'_, PyAny>) -> PyResult<Vec<[u32; 3]>> {
    crate::utils_py::read_index_array_chunked::<u32, 3>(arr, "faces")
}

#[inline]
pub(super) fn read_edges_u32(arr: &Bound<'_, PyAny>) -> PyResult<Vec<[u32; 2]>> {
    crate::utils_py::read_index_array_chunked::<u32, 2>(arr, "edges")
}

#[inline]
pub(super) fn read_tets_u32(arr: &Bound<'_, PyAny>) -> PyResult<Vec<[u32; 4]>> {
    crate::utils_py::read_index_array_chunked::<u32, 4>(arr, "tets")
}
