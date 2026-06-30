// File: crates/ppf-cts-core/src/kernels/scene_build/pin_kernel.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pin operation kernels.
//
// Each `*_apply` mirrors a Python `Operation.apply` body. Inputs are
// flat `(N, 3)` f64 buffers; outputs are fresh `Vec<f64>` of the same
// length. Time-range gating is also handled here so the Python wrapper
// is a one-liner. Per-vertex math goes through `datamodel::pin_apply`
// so this module and the solver-side `src/scene.rs::apply_op` produce
// identical positions.
//
// The easing helpers (`bezier_progress` / `eased_progress`) are thin
// adapters over `datamodel::easing`: their only callers are pin
// operations and the Python easing wrapper, and they preserve the
// tuple-/string-based public API while the actual math lives in the
// centralized module.

use super::quaternion::{apply_trs_to_verts, quat_slerp};
use crate::datamodel::easing::{self, TransitionKind};

// ---------------------------------------------------------------------------
// Bezier / smooth easing.

/// Cubic Bezier easing. `handles = (p1, p2)` where each is `(x, y)`.
/// `P0 = (0, 0)`, `P3 = (1, 1)`. Mirrors `_bezier_progress`. Thin
/// adapter forwarding to `datamodel::easing::bezier_progress`
/// (`right = handles.0`, `left = handles.1`).
pub fn bezier_progress(t: f64, handles: ([f64; 2], [f64; 2])) -> f64 {
    easing::bezier_progress(t, handles.0, handles.1)
}

/// Linear progress in `[0, 1]` over `[t_start, t_end]`. `transition` is
/// one of `"linear"`, `"smooth"`, `"bezier"`. Mirrors `_eased_progress`.
/// When `transition` isn't recognized (or is `"bezier"` without
/// handles), falls through to the linear branch (same as Python). Thin
/// adapter forwarding to `datamodel::easing::eased_progress`, which also
/// guards the zero-duration window (`t_end == t_start` returns `0.0`
/// instead of a nan/inf).
pub fn eased_progress(
    time: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    bezier_handles: Option<([f64; 2], [f64; 2])>,
) -> f64 {
    let kind = match transition {
        "smooth" => TransitionKind::Smooth,
        "bezier" => match bezier_handles {
            Some((right, left)) => TransitionKind::Bezier {
                right_handle: right,
                left_handle: left,
            },
            None => TransitionKind::Linear,
        },
        _ => TransitionKind::Linear,
    };
    easing::eased_progress(time, t_start, t_end, &kind)
}

// ---------------------------------------------------------------------------
// Pin operations.

/// Mirrors `MoveByOperation.apply`. `delta` is `(N, 3)` matching
/// `vertex`. When `time < t_start` returns the input unchanged; when
/// `time >= t_end` returns `vertex + delta`; otherwise blends with the
/// eased progress. Per-vertex math goes through
/// `datamodel::pin_apply::move_by_step` so this kernel and the
/// solver-side `src/scene.rs::apply_op` produce identical positions.
pub fn move_by_apply(
    vertex: &[f64],
    delta: &[f64],
    time: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    bezier_handles: Option<([f64; 2], [f64; 2])>,
) -> Vec<f64> {
    use crate::datamodel::pin_apply;
    let n = vertex.len();
    let count = n / 3;
    let handles_flat = bezier_handles.map(|(r, l)| [r[0], r[1], l[0], l[1]]);
    let progress = if time < t_start {
        return vertex.to_vec();
    } else if time >= t_end {
        1.0
    } else {
        pin_apply::progress_at(time, t_start, t_end, transition, handles_flat)
    };
    let mut out = vec![0.0f64; n];
    for i in 0..count {
        let pos = [vertex[3 * i], vertex[3 * i + 1], vertex[3 * i + 2]];
        let d = [delta[3 * i], delta[3 * i + 1], delta[3 * i + 2]];
        let r = pin_apply::move_by_step(pos, d, progress);
        out[3 * i] = r[0];
        out[3 * i + 1] = r[1];
        out[3 * i + 2] = r[2];
    }
    out
}

/// Mirrors `MoveToOperation.apply`. `target` is `(N, 3)` matching
/// `vertex`. When `time < t_start` returns the input unchanged; when
/// `time >= t_end` returns a copy of `target`; otherwise lerp.
/// Per-vertex math goes through `datamodel::pin_apply::move_to_step`.
pub fn move_to_apply(
    vertex: &[f64],
    target: &[f64],
    time: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    bezier_handles: Option<([f64; 2], [f64; 2])>,
) -> Vec<f64> {
    use crate::datamodel::pin_apply;
    let n = vertex.len();
    let count = n / 3;
    let handles_flat = bezier_handles.map(|(r, l)| [r[0], r[1], l[0], l[1]]);
    if time < t_start {
        return vertex.to_vec();
    }
    if time >= t_end {
        return target.to_vec();
    }
    let progress = pin_apply::progress_at(time, t_start, t_end, transition, handles_flat);
    let mut out = vec![0.0f64; n];
    for i in 0..count {
        let pos = [vertex[3 * i], vertex[3 * i + 1], vertex[3 * i + 2]];
        let tgt = [target[3 * i], target[3 * i + 1], target[3 * i + 2]];
        let r = pin_apply::move_to_step(pos, tgt, progress);
        out[3 * i] = r[0];
        out[3 * i + 1] = r[1];
        out[3 * i + 2] = r[2];
    }
    out
}

/// Mirrors `SpinOperation.apply`. Rodrigues rotation around `axis`
/// passing through `center`. `angular_velocity_deg_per_s` is in
/// degrees/second. `t_eff = min(time, t_end) - t_start`. When
/// `t_eff <= 0` returns the input unchanged.
pub fn spin_apply(
    vertex: &[f64],
    center: [f64; 3],
    axis: [f64; 3],
    angular_velocity_deg_per_s: f64,
    t_start: f64,
    t_end: f64,
    time: f64,
) -> Vec<f64> {
    use crate::datamodel::pin_apply;
    let n = vertex.len();
    let count = n / 3;
    let angle = pin_apply::spin_angle_rad(angular_velocity_deg_per_s, t_start, t_end, time);
    if angle <= 0.0 {
        return vertex.to_vec();
    }
    let mut out = vec![0.0f64; n];
    for i in 0..count {
        let p = [vertex[3 * i], vertex[3 * i + 1], vertex[3 * i + 2]];
        let r = pin_apply::spin_step(p, center, axis, angle);
        out[3 * i] = r[0];
        out[3 * i + 1] = r[1];
        out[3 * i + 2] = r[2];
    }
    out
}

/// Mirrors `ScaleOperation.apply`. Scales around `center` from 1.0 to
/// `factor` over `[t_start, t_end]`. `transition` is `"linear"` or
/// `"smooth"`. When `time < t_start` returns the input unchanged.
pub fn scale_apply(
    vertex: &[f64],
    center: [f64; 3],
    factor: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    time: f64,
) -> Vec<f64> {
    use crate::datamodel::pin_apply;
    let n = vertex.len();
    let count = n / 3;
    if time < t_start {
        return vertex.to_vec();
    }
    let cur_factor =
        pin_apply::scale_factor_at(time, t_start, t_end, factor, transition, None);
    let mut out = vec![0.0f64; n];
    for i in 0..count {
        let p = [vertex[3 * i], vertex[3 * i + 1], vertex[3 * i + 2]];
        let r = pin_apply::scale_step(p, center, cur_factor);
        out[3 * i] = r[0];
        out[3 * i + 1] = r[1];
        out[3 * i + 2] = r[2];
    }
    out
}

/// Per-segment interpolation kind used by transform_keyframe ops.
#[derive(Debug, Clone, Copy)]
pub enum SegInterp {
    Linear,
    /// `(handle_right, handle_left)` cubic Bezier handles in `[0, 1]`.
    Bezier([f64; 2], [f64; 2]),
    Constant,
}

/// Mirrors `TransformKeyframeOperation.apply`. Returns
/// `R(t)*S(t)*local + T(t) - rest_translation` per vertex with slerp
/// on rotation. `times`, `translations`, `quaternions`, `scales` are
/// equal-length keyframe lists; `segments` has length `times.len() - 1`.
/// When `times` is empty, returns the input unchanged.
///
/// The per-vertex TRS evaluation routes through
/// `datamodel::quat::transform_keyframes_step` so this preview kernel
/// and the solver-side `src/scene.rs::apply_op` share one
/// implementation (same segment search, interp dispatch, T/S lerp, Q
/// slerp, and rest-translation offset). `segments` is translated into
/// the helper's `(interp code, handle pair)` form: `0` = LINEAR, `1`
/// = BEZIER (`[hr0, hr1, hl0, hl1]`), `2` = CONSTANT.
pub fn transform_keyframe_apply(
    vertex: &[f64],
    local_vert: &[f64],
    times: &[f64],
    translations: &[[f64; 3]],
    quaternions: &[[f64; 4]],
    scales: &[[f64; 3]],
    segments: &[SegInterp],
    rest_translation: [f64; 3],
    time: f64,
) -> Vec<f64> {
    use crate::datamodel::quat::transform_keyframes_step;
    if times.is_empty() {
        return vertex.to_vec();
    }
    let mut interps = Vec::with_capacity(segments.len());
    let mut handles = Vec::with_capacity(segments.len());
    for seg in segments {
        match *seg {
            SegInterp::Linear => {
                interps.push(0u8);
                handles.push([0.0; 4]);
            }
            SegInterp::Bezier(h_r, h_l) => {
                interps.push(1u8);
                handles.push([h_r[0], h_r[1], h_l[0], h_l[1]]);
            }
            SegInterp::Constant => {
                interps.push(2u8);
                handles.push([0.0; 4]);
            }
        }
    }
    let count = local_vert.len() / 3;
    let mut out = vec![0.0f64; local_vert.len()];
    for i in 0..count {
        let local = [
            local_vert[3 * i],
            local_vert[3 * i + 1],
            local_vert[3 * i + 2],
        ];
        let r = transform_keyframes_step(
            local,
            times,
            translations,
            quaternions,
            scales,
            &interps,
            &handles,
            rest_translation,
            time,
        );
        out[3 * i] = r[0];
        out[3 * i + 1] = r[1];
        out[3 * i + 2] = r[2];
    }
    out
}

/// Mirrors `TransformAnimation.evaluate`. Returns world-space vertex
/// positions at `time`, lerping translation/scale and slerping rotation.
/// For empty timelines returns a copy of `local_vert`.
pub fn transform_animation_evaluate(
    local_vert: &[f64],
    times: &[f64],
    translations: &[[f64; 3]],
    quaternions: &[[f64; 4]],
    scales: &[[f64; 3]],
    time: f64,
) -> Vec<f64> {
    if times.is_empty() {
        return local_vert.to_vec();
    }
    if time <= times[0] {
        return apply_trs_to_verts(local_vert, translations[0], quaternions[0], scales[0]);
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return apply_trs_to_verts(local_vert, translations[last], quaternions[last], scales[last]);
    }
    for i in 0..last {
        let t0 = times[i];
        let t1 = times[i + 1];
        if t0 <= time && time < t1 {
            let progress = (time - t0) / (t1 - t0);
            let inv = 1.0 - progress;
            let trans = [
                translations[i][0] * inv + translations[i + 1][0] * progress,
                translations[i][1] * inv + translations[i + 1][1] * progress,
                translations[i][2] * inv + translations[i + 1][2] * progress,
            ];
            let quat = quat_slerp(quaternions[i], quaternions[i + 1], progress);
            let scl = [
                scales[i][0] * inv + scales[i + 1][0] * progress,
                scales[i][1] * inv + scales[i + 1][1] * progress,
                scales[i][2] * inv + scales[i + 1][2] * progress,
            ];
            return apply_trs_to_verts(local_vert, trans, quat, scl);
        }
    }
    apply_trs_to_verts(local_vert, translations[last], quaternions[last], scales[last])
}
