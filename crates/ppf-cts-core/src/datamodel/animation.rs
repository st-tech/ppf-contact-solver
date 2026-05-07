// File: crates/ppf-cts-core/src/datamodel/animation.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Sparse rigid-body transform animation. Direct port of
// `TransformAnimation` (frontend/_scene_.py:772-809). Used for
// keyframed STATIC objects: every entry has a time stamp and a
// (translation, rotation, scale) triple. The world-space evaluation
// at a given time is performed by the kernel
// `kernels::scene_build::transform_animation_evaluate`, which is
// called from both Rust and Python via PyO3; this struct is a passive
// data container.

use ndarray::Array2;

use super::quat::{Quat, Vec3};

#[derive(Debug, Clone)]
pub struct TransformAnimation {
    /// Local-space vertex positions (N, 3). Constant across the
    /// animation; only the global transform varies with time.
    pub local_vert: Array2<f64>,

    /// Sorted ascending. Empty == no animation.
    pub times: Vec<f64>,
    pub translations: Vec<Vec3>,
    pub quaternions: Vec<Quat>,
    pub scales: Vec<Vec3>,
}
