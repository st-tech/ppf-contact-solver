// File: crates/ppf-cts-core/src/kernels/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Numeric kernels. Each module is a self-contained replacement for a
// previous Numba-jitted hot path; PyO3 bindings live in ppf-cts-py.
// Kernels never own Python state. They take primitive slices and
// return primitive results, so they're testable in pure Rust.

pub mod bvh;
mod constants;
pub mod decoder;
pub mod fixed_scene_assemble;
mod geom_util;
pub mod intersection;
pub mod invisible_collider;
pub mod proximity;
pub mod rasterizer;
pub mod scene_build;
pub mod scene_loops;
pub mod sdf;
