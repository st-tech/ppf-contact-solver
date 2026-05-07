// File: crates/ppf-cts-py/src/lib.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// PyO3 entry point. The module exposes `version()` and
// `schema_version()` plus all native kernel bindings
// (`_invisible_collider`, `_distance`, ...) consumed by the Python
// frontend.

//! PyO3 bindings for the ppf-contact-solver Rust core.
//!
//! Compiled by maturin into the Python extension module
//! `_ppf_cts_py`. The pure-Python `frontend` package re-exports from
//! here, so JupyterLab notebooks keep using the same names while the
//! implementation lives in Rust.
//!
//! # Public surface
//!
//! The `#[pymodule]` `_ppf_cts_py` registers `version()`,
//! `schema_version()`, all numeric kernels (rasterizer, SDF,
//! marching cubes, self-intersection, frame mapping, normals),
//! `_utils_` helpers, and the per-domain holders (param, asset,
//! session, app, scene, pin, mesh, render, decoder).
//!
//! Algorithmic implementations live in [`ppf_cts_core`]; cross-language
//! wire types come from [`ppf_cts_formats`]. This crate is consumed
//! by `frontend/` Python via the `_ppf_cts_py` extension module.

use pyo3::prelude::*;

mod app_py;
mod asset_py;
mod decoder_py;
mod errors;
mod extra_py;
mod kernels;
mod mesh_py;
mod param_py;
mod pin_py;
mod render_py;
mod scene_py;
mod session_py;
mod utils_py;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn schema_version() -> u32 {
    ppf_cts_formats::SCHEMA_VERSION
}

#[pymodule]
fn _ppf_cts_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    // Numeric kernels
    m.add_function(wrap_pyfunction!(kernels::check_wall_violations, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::frame_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::interpolate_surface, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::check_contact_offset_violation, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::check_self_intersection, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::eval_sdf_grid, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::marching_cubes, m)?)?;
    // Rasterizer
    m.add_function(wrap_pyfunction!(kernels::rasterize_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::rasterize_lines, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::normals, m)?)?;
    // Utils (check_gpu / busy / terminate / fast_check)
    m.add_function(wrap_pyfunction!(utils_py::check_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(utils_py::solver_busy, m)?)?;
    m.add_function(wrap_pyfunction!(utils_py::terminate_solver, m)?)?;
    m.add_function(wrap_pyfunction!(utils_py::is_fast_check, m)?)?;
    m.add_function(wrap_pyfunction!(utils_py::set_fast_check, m)?)?;
    m.add_function(wrap_pyfunction!(utils_py::get_cache_dir, m)?)?;
    m.add_function(wrap_pyfunction!(utils_py::process_name, m)?)?;
    // Additional `_utils_` helpers.
    utils_py::register(m)?;
    // Extra (Extra.load_CIPC_stitch_mesh / sparse_clone)
    m.add_function(wrap_pyfunction!(extra_py::load_cipc_stitch_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(extra_py::sparse_clone, m)?)?;
    // Param holder and asset validation.
    param_py::register(m)?;
    asset_py::register(m)?;
    session_py::register(m)?;
    // App, scene hot-path kernels, pin holder.
    app_py::register(m)?;
    pin_py::register(m)?;
    scene_py::register(m)?;
    // Mesh primitive generators and post-processing kernels.
    mesh_py::register(m)?;
    render_py::register(m)?;
    decoder_py::register(m)?;
    Ok(())
}
