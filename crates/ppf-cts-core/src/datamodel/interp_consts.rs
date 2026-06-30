// File: crates/ppf-cts-core/src/datamodel/interp_consts.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Shared numeric constants for the slerp / Bezier interpolation paths.
//
// These values are intentionally identical across every preview (core)
// and simulation (solver) copy of the slerp and Bezier math so the two
// produce bit-identical positions. Centralizing them here means a tuning
// change happens in one place instead of being re-edited in lockstep at
// each call site (quat.rs, easing.rs, and the scene_build kernels).

/// Slerp short-circuit threshold. When `dot > SLERP_LINEAR_BLEND_DOT_THRESHOLD`
/// the two quaternions are close enough that a normalized linear blend is
/// used instead of the trigonometric slerp.
pub const SLERP_LINEAR_BLEND_DOT_THRESHOLD: f64 = 0.9995;

/// Newton-Raphson iteration count for the Bezier `B_x(u) = t` solve.
pub const BEZIER_NEWTON_ITERS: usize = 8;

/// Newton-Raphson derivative floor for the Bezier solve. When `|dB_x/du|`
/// drops below this the iteration stops to avoid dividing by ~0.
pub const BEZIER_NEWTON_DERIV_EPS: f64 = 1e-10;
