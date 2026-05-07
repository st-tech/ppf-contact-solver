// File: crates/ppf-cts-core/src/datamodel/pin.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pin operations + pin data. Direct port of the Operation hierarchy
// (frontend/_scene_.py:830-1108) and the small dataclasses
// `PinKeyframe` / `SpinData` / `PinData`.
//
// Schema notes (kept aligned with the Python dataclasses for
// round-trip fidelity, not Rust elegance):
//
// * `MoveBy.delta` is either a single `Vec3` (uniform per-pin offset)
//   or a per-vertex `(N, 3)` array. Python builds a per-vertex array
//   by tiling a uniform input, but downstream consumers (the CBOR
//   wire format and the Blender decoder) pass a `(N, 3)` array
//   directly. We preserve both shapes so a Python-built `PinHolder`
//   survives a round trip into Rust and back.
//
// * `transition` is stored as a free-form string (`"linear"`,
//   `"smooth"`, `"bezier"`) matching `PinData.transition` in Python.
//   Bezier control points live in a sibling `bezier_handles` field
//   on each operation that supports easing (`MoveBy`, `MoveTo`,
//   `Scale`). The pair is folded into the local `TransitionKind`
//   (defined in `easing.rs`) only at apply time, so the on-disk +
//   PyO3 mirror keeps the same flat field set as Python.
//
// Each operation has an `apply(vertex, time)` method that takes the
// current vertex positions (as an `(N, 3)` ndarray) and returns the
// transformed positions. The chain of operations on a pin is applied
// in sequence; the Python reference iterates `for op in pin.operations:
// vertex = op.apply(vertex, time)`.

use ndarray::Array2;

use super::easing::{eased_progress, TransitionKind};
use super::pin_apply;
use super::quat::{apply_transform_to_verts, slerp, Quat, Vec3};

#[derive(Debug, Clone)]
pub struct SpinData {
    pub center: Vec3,
    pub axis: Vec3,
    pub angular_velocity: f64,
    pub t_start: f64,
    pub t_end: f64,
}

#[derive(Debug, Clone)]
pub struct PinKeyframe {
    pub position: Array2<f64>,
    pub time: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub enum CenterMode {
    #[default]
    Absolute,
    Relative,
}


#[derive(Debug, Clone)]
pub enum InterpMode {
    Linear,
    Constant,
    Bezier {
        right_handle: [f64; 2],
        left_handle: [f64; 2],
    },
}

#[derive(Debug, Clone)]
pub struct KeyframeSegment {
    pub interp: InterpMode,
}

/// `MoveBy` delta payload. Python's `MoveByOperation.delta` is always
/// an `(N, 3)` ndarray after the `PinHolder.move_by` reshape, but the
/// builder accepts both a single `[x, y, z]` and a `(N, 3)` array, and
/// downstream code (decoder, CBOR) sometimes hands us either shape.
/// Keeping both representations means the same operation survives
/// round-tripping through the wire format without an array allocation
/// in the uniform case.
#[derive(Debug, Clone)]
pub enum MoveByDelta {
    /// Single offset applied to every pinned vertex.
    Uniform(Vec3),
    /// Per-vertex `(N, 3)` offsets. `N` must equal the pin's vertex count.
    PerVertex(Array2<f64>),
}

impl MoveByDelta {
    /// Get a uniform offset if this is a uniform delta. Returns `None`
    /// for per-vertex deltas. Used by callers that already special-case
    /// uniform offsets (the CBOR encoder writes them as `[f64; 3]`).
    pub fn as_uniform(&self) -> Option<Vec3> {
        match self {
            MoveByDelta::Uniform(v) => Some(*v),
            MoveByDelta::PerVertex(_) => None,
        }
    }
}

/// Resolve a `(transition_str, bezier_handles)` pair into the
/// `TransitionKind` used by `eased_progress`. Mirrors the Python
/// `_eased_progress(time, ..., transition, bezier_handles)` dispatch.
fn resolve_transition(
    transition: &str,
    bezier_handles: Option<[[f64; 2]; 2]>,
) -> TransitionKind {
    match transition {
        "smooth" => TransitionKind::Smooth,
        "bezier" => match bezier_handles {
            Some([right, left]) => TransitionKind::Bezier {
                right_handle: right,
                left_handle: left,
            },
            // Match Python: `transition == "bezier"` with no handles
            // falls through to the linear branch.
            None => TransitionKind::Linear,
        },
        // Anything else (including "linear" and unknown values) is
        // treated as linear, matching Python's fall-through.
        _ => TransitionKind::Linear,
    }
}

/// Pin operations. Apply order is left-to-right; each receives the
/// vertex positions output by the previous and the current sim time.
#[derive(Debug, Clone)]
pub enum PinOperation {
    /// Move by a 3D delta (uniform or per-vertex) over [t_start, t_end].
    MoveBy {
        delta: MoveByDelta,
        t_start: f64,
        t_end: f64,
        transition: String,
        bezier_handles: Option<[[f64; 2]; 2]>,
    },
    /// Move from the current pose to an absolute (N, 3) target over
    /// [t_start, t_end]. Per-vertex linear interpolation in space.
    MoveTo {
        target: Array2<f64>,
        t_start: f64,
        t_end: f64,
        transition: String,
        bezier_handles: Option<[[f64; 2]; 2]>,
    },
    /// Spin about an arbitrary axis (Rodrigues rotation) at constant
    /// angular velocity (in degrees/s).
    Spin {
        center: Vec3,
        axis: Vec3,
        angular_velocity: f64,
        t_start: f64,
        t_end: f64,
        center_mode: CenterMode,
    },
    /// Scale about a center, easing from 1.0 to `factor` over the
    /// time window.
    Scale {
        center: Vec3,
        factor: f64,
        t_start: f64,
        t_end: f64,
        transition: String,
        bezier_handles: Option<[[f64; 2]; 2]>,
        center_mode: CenterMode,
    },
    /// Torque is a *force*, not a kinematic operation. Apply is a
    /// pass-through; the solver handles the dynamics. Mirrors
    /// `TorqueOperation.apply` returning `vertex` unchanged.
    Torque {
        axis_component: i32,
        magnitude: f64,
        hint_vertex: i32,
        t_start: f64,
        t_end: f64,
    },
    /// Sparse TRS-keyframe operation (used by Blender object
    /// animations exported as fcurves). Same per-segment slerp as
    /// `TransformAnimation` but with a per-segment interpolation mode.
    TransformKeyframe {
        local_vert: Array2<f64>,
        times: Vec<f64>,
        translations: Vec<Vec3>,
        quaternions: Vec<Quat>,
        scales: Vec<Vec3>,
        segments: Vec<KeyframeSegment>,
        rest_translation: Vec3,
    },
}

impl PinOperation {
    /// Time window the op is "alive" in. Mirrors `get_time_range`.
    pub fn time_range(&self) -> (f64, f64) {
        match self {
            PinOperation::MoveBy { t_start, t_end, .. }
            | PinOperation::MoveTo { t_start, t_end, .. }
            | PinOperation::Spin { t_start, t_end, .. }
            | PinOperation::Scale { t_start, t_end, .. }
            | PinOperation::Torque { t_start, t_end, .. } => (*t_start, *t_end),
            PinOperation::TransformKeyframe { times, .. } => {
                if times.is_empty() {
                    (0.0, 0.0)
                } else {
                    (times[0], *times.last().unwrap())
                }
            }
        }
    }

    /// Apply the op to a copy of `vertex` at `time`.
    pub fn apply(&self, vertex: &Array2<f64>, time: f64) -> Array2<f64> {
        match self {
            PinOperation::MoveBy {
                delta,
                t_start,
                t_end,
                transition,
                bezier_handles,
            } => {
                if time < *t_start {
                    return vertex.clone();
                }
                let kind = resolve_transition(transition, *bezier_handles);
                let progress = if time >= *t_end {
                    1.0
                } else {
                    eased_progress(time, *t_start, *t_end, &kind)
                };
                let mut out = vertex.clone();
                match delta {
                    MoveByDelta::Uniform(v) => {
                        for mut row in out.rows_mut() {
                            let p = pin_apply::move_by_step(
                                [row[0], row[1], row[2]],
                                *v,
                                progress,
                            );
                            row[0] = p[0];
                            row[1] = p[1];
                            row[2] = p[2];
                        }
                    }
                    MoveByDelta::PerVertex(arr) => {
                        debug_assert_eq!(
                            arr.dim(),
                            vertex.dim(),
                            "MoveBy per-vertex delta must match vertex shape"
                        );
                        for (mut row_o, row_d) in
                            out.rows_mut().into_iter().zip(arr.rows())
                        {
                            let p = pin_apply::move_by_step(
                                [row_o[0], row_o[1], row_o[2]],
                                [row_d[0], row_d[1], row_d[2]],
                                progress,
                            );
                            row_o[0] = p[0];
                            row_o[1] = p[1];
                            row_o[2] = p[2];
                        }
                    }
                }
                out
            }

            PinOperation::MoveTo {
                target,
                t_start,
                t_end,
                transition,
                bezier_handles,
            } => {
                debug_assert_eq!(target.dim(), vertex.dim(), "MoveTo target must match vertex shape");
                if time < *t_start {
                    return vertex.clone();
                }
                if time >= *t_end {
                    return target.clone();
                }
                let kind = resolve_transition(transition, *bezier_handles);
                let p = eased_progress(time, *t_start, *t_end, &kind);
                let mut out = vertex.clone();
                for ((row_v, row_t), mut row_o) in
                    vertex.rows().into_iter().zip(target.rows()).zip(out.rows_mut())
                {
                    for k in 0..3 {
                        row_o[k] = (1.0 - p) * row_v[k] + p * row_t[k];
                    }
                }
                out
            }

            PinOperation::Spin {
                center,
                axis,
                angular_velocity,
                t_start,
                t_end,
                center_mode: _,
            } => {
                let t_eff = time.min(*t_end) - *t_start;
                if t_eff <= 0.0 {
                    return vertex.clone();
                }
                let omega = angular_velocity / 180.0 * std::f64::consts::PI;
                let angle = omega * t_eff;
                let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
                let ax = if n > 0.0 {
                    [axis[0] / n, axis[1] / n, axis[2] / n]
                } else {
                    *axis
                };
                let mut out = vertex.clone();
                for mut row in out.rows_mut() {
                    let p = pin_apply::spin_step(
                        [row[0], row[1], row[2]],
                        *center,
                        ax,
                        angle,
                    );
                    row[0] = p[0];
                    row[1] = p[1];
                    row[2] = p[2];
                }
                out
            }

            PinOperation::Scale {
                center,
                factor,
                t_start,
                t_end,
                transition,
                bezier_handles,
                center_mode: _,
            } => {
                if time < *t_start {
                    return vertex.clone();
                }
                let factor = if time >= *t_end {
                    *factor
                } else {
                    let kind = resolve_transition(transition, *bezier_handles);
                    let p = eased_progress(time, *t_start, *t_end, &kind);
                    1.0 + (factor - 1.0) * p
                };
                let mut out = vertex.clone();
                for mut row in out.rows_mut() {
                    let p = pin_apply::scale_step(
                        [row[0], row[1], row[2]],
                        *center,
                        factor,
                    );
                    row[0] = p[0];
                    row[1] = p[1];
                    row[2] = p[2];
                }
                out
            }

            PinOperation::Torque { .. } => vertex.clone(),

            PinOperation::TransformKeyframe {
                local_vert,
                times,
                translations,
                quaternions,
                scales,
                segments,
                rest_translation,
            } => {
                if times.is_empty() {
                    return vertex.clone();
                }
                let last = times.len() - 1;
                let eval = |t: Vec3, q: Quat, s: Vec3| -> Array2<f64> {
                    let mut applied = apply_transform_to_verts(local_vert, t, q, s);
                    for mut row in applied.rows_mut() {
                        row[0] -= rest_translation[0];
                        row[1] -= rest_translation[1];
                        row[2] -= rest_translation[2];
                    }
                    applied
                };
                if time <= times[0] {
                    return eval(translations[0], quaternions[0], scales[0]);
                }
                if time >= times[last] {
                    return eval(translations[last], quaternions[last], scales[last]);
                }
                for i in 0..last {
                    let (t0, t1) = (times[i], times[i + 1]);
                    if t0 <= time && time < t1 {
                        let mut p = (time - t0) / (t1 - t0);
                        match &segments[i].interp {
                            InterpMode::Bezier {
                                right_handle,
                                left_handle,
                            } => {
                                p = super::easing::bezier_progress(
                                    p,
                                    *right_handle,
                                    *left_handle,
                                );
                            }
                            InterpMode::Constant => {
                                p = 0.0;
                            }
                            InterpMode::Linear => {}
                        }
                        let trans = lerp_vec3(translations[i], translations[i + 1], p);
                        let quat = slerp(quaternions[i], quaternions[i + 1], p);
                        let scl = lerp_vec3(scales[i], scales[i + 1], p);
                        return eval(trans, quat, scl);
                    }
                }
                // unreachable given the bounds above; mirror Python fallback.
                eval(translations[last], quaternions[last], scales[last])
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PinData {
    /// Vertex indices being pinned.
    pub index: Vec<i32>,
    /// Operations chain. Applied left-to-right.
    pub operations: Vec<PinOperation>,
    /// Optional time after which the pin releases. None == hold forever.
    pub unpin_time: Option<f64>,
    /// Default transition for newly added operations. Mirrors
    /// `PinData.transition` in Python (str, not enum).
    pub transition: String,
    pub pull_strength: f64,
    pub pin_group_id: String,
    /// Static moving objects use the pin-shell as an implementation
    /// detail; the user never asked for these pins, so the preview
    /// hides them.
    pub hide_in_preview: bool,
}

impl Default for PinData {
    fn default() -> Self {
        Self {
            index: Vec::new(),
            operations: Vec::new(),
            unpin_time: None,
            transition: "linear".to_string(),
            pull_strength: 0.0,
            pin_group_id: String::new(),
            hide_in_preview: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers.

#[inline]
fn lerp_vec3(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    [
        (1.0 - t) * a[0] + t * b[0],
        (1.0 - t) * a[1] + t * b[1],
        (1.0 - t) * a[2] + t * b[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn moveby_before_window_no_op() {
        let v = array![[1.0, 2.0, 3.0]];
        let op = PinOperation::MoveBy {
            delta: MoveByDelta::Uniform([10.0, 0.0, 0.0]),
            t_start: 1.0,
            t_end: 2.0,
            transition: "linear".to_string(),
            bezier_handles: None,
        };
        assert_eq!(op.apply(&v, 0.5), v);
    }

    #[test]
    fn moveby_after_window_full_offset() {
        let v = array![[1.0, 2.0, 3.0]];
        let op = PinOperation::MoveBy {
            delta: MoveByDelta::Uniform([10.0, 0.0, 0.0]),
            t_start: 1.0,
            t_end: 2.0,
            transition: "linear".to_string(),
            bezier_handles: None,
        };
        assert_eq!(op.apply(&v, 5.0), array![[11.0, 2.0, 3.0]]);
    }

    #[test]
    fn moveby_midpoint_linear() {
        let v = array![[0.0, 0.0, 0.0]];
        let op = PinOperation::MoveBy {
            delta: MoveByDelta::Uniform([10.0, 0.0, 0.0]),
            t_start: 0.0,
            t_end: 1.0,
            transition: "linear".to_string(),
            bezier_handles: None,
        };
        let out = op.apply(&v, 0.5);
        assert!(approx(out[[0, 0]], 5.0, 1e-15));
    }

    #[test]
    fn moveby_per_vertex_delta() {
        // Two pinned vertices, each with its own delta.
        let v = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let d = array![[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]];
        let op = PinOperation::MoveBy {
            delta: MoveByDelta::PerVertex(d),
            t_start: 0.0,
            t_end: 1.0,
            transition: "linear".to_string(),
            bezier_handles: None,
        };
        let out = op.apply(&v, 1.5);
        assert_eq!(out, array![[10.0, 0.0, 0.0], [1.0, 21.0, 1.0]]);
        // Midpoint: half of each per-vertex delta.
        let mid = op.apply(&v, 0.5);
        assert!(approx(mid[[0, 0]], 5.0, 1e-15));
        assert!(approx(mid[[1, 1]], 11.0, 1e-15));
    }

    #[test]
    fn moveby_smooth_easing() {
        // smoothstep(0.5) = 0.5, but it differs from linear at 0.25.
        let v = array![[0.0, 0.0, 0.0]];
        let op = PinOperation::MoveBy {
            delta: MoveByDelta::Uniform([10.0, 0.0, 0.0]),
            t_start: 0.0,
            t_end: 1.0,
            transition: "smooth".to_string(),
            bezier_handles: None,
        };
        // smoothstep(0.25) = 0.15625 → x = 1.5625
        let out = op.apply(&v, 0.25);
        assert!(approx(out[[0, 0]], 1.5625, 1e-12));
    }

    #[test]
    fn moveby_bezier_with_handles_round_trips() {
        // Bezier with diagonal handles == linear.
        let v = array![[0.0, 0.0, 0.0]];
        let op = PinOperation::MoveBy {
            delta: MoveByDelta::Uniform([10.0, 0.0, 0.0]),
            t_start: 0.0,
            t_end: 1.0,
            transition: "bezier".to_string(),
            bezier_handles: Some([[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]),
        };
        let out = op.apply(&v, 0.5);
        assert!(approx(out[[0, 0]], 5.0, 1e-6));
    }

    #[test]
    fn moveby_bezier_no_handles_falls_back_to_linear() {
        // Python's `_eased_progress`: `transition == "bezier"` with
        // `bezier_handles is None` falls through to the linear branch.
        let v = array![[0.0, 0.0, 0.0]];
        let op = PinOperation::MoveBy {
            delta: MoveByDelta::Uniform([10.0, 0.0, 0.0]),
            t_start: 0.0,
            t_end: 1.0,
            transition: "bezier".to_string(),
            bezier_handles: None,
        };
        let out = op.apply(&v, 0.5);
        assert!(approx(out[[0, 0]], 5.0, 1e-15));
    }

    #[test]
    fn moveto_endpoints_lock_to_target() {
        let v = array![[0.0, 0.0, 0.0]];
        let target = array![[10.0, 0.0, 0.0]];
        let op = PinOperation::MoveTo {
            target: target.clone(),
            t_start: 0.0,
            t_end: 1.0,
            transition: "linear".to_string(),
            bezier_handles: None,
        };
        assert_eq!(op.apply(&v, 1.5), target);
    }

    #[test]
    fn spin_zero_time_window_no_op() {
        let v = array![[1.0, 0.0, 0.0]];
        let op = PinOperation::Spin {
            center: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            angular_velocity: 360.0,
            t_start: 5.0,
            t_end: 6.0,
            center_mode: CenterMode::Absolute,
        };
        assert_eq!(op.apply(&v, 1.0), v);
    }

    #[test]
    fn spin_quarter_turn_about_y() {
        // 90 deg/s for 1s about +Y → x-axis rotates to -z.
        let v = array![[1.0, 0.0, 0.0]];
        let op = PinOperation::Spin {
            center: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            angular_velocity: 90.0,
            t_start: 0.0,
            t_end: 10.0,
            center_mode: CenterMode::Absolute,
        };
        let out = op.apply(&v, 1.0);
        assert!(approx(out[[0, 0]], 0.0, 1e-12));
        assert!(approx(out[[0, 1]], 0.0, 1e-12));
        assert!(approx(out[[0, 2]], -1.0, 1e-12));
    }

    #[test]
    fn scale_endpoints_full_factor() {
        let v = array![[2.0, 0.0, 0.0]];
        let op = PinOperation::Scale {
            center: [0.0, 0.0, 0.0],
            factor: 3.0,
            t_start: 0.0,
            t_end: 1.0,
            transition: "linear".to_string(),
            bezier_handles: None,
            center_mode: CenterMode::Absolute,
        };
        let out = op.apply(&v, 5.0);
        assert!(approx(out[[0, 0]], 6.0, 1e-15));
    }

    #[test]
    fn scale_midpoint_eases_factor() {
        let v = array![[2.0, 0.0, 0.0]];
        let op = PinOperation::Scale {
            center: [0.0, 0.0, 0.0],
            factor: 3.0,
            t_start: 0.0,
            t_end: 1.0,
            transition: "linear".to_string(),
            bezier_handles: None,
            center_mode: CenterMode::Absolute,
        };
        // Midpoint factor = 1 + (3 - 1) * 0.5 = 2.0; vertex 2 → 4.
        let out = op.apply(&v, 0.5);
        assert!(approx(out[[0, 0]], 4.0, 1e-15));
    }

    #[test]
    fn torque_is_passthrough() {
        let v = array![[1.0, 2.0, 3.0]];
        let op = PinOperation::Torque {
            axis_component: 0,
            magnitude: 10.0,
            hint_vertex: -1,
            t_start: 0.0,
            t_end: f64::INFINITY,
        };
        assert_eq!(op.apply(&v, 5.0), v);
    }

    #[test]
    fn time_range_correct_per_variant() {
        let mb = PinOperation::MoveBy {
            delta: MoveByDelta::Uniform([0.0; 3]),
            t_start: 1.0,
            t_end: 2.0,
            transition: "linear".to_string(),
            bezier_handles: None,
        };
        assert_eq!(mb.time_range(), (1.0, 2.0));

        let kf = PinOperation::TransformKeyframe {
            local_vert: array![[0.0, 0.0, 0.0]],
            times: vec![0.0, 1.5, 3.0],
            translations: vec![[0.0; 3]; 3],
            quaternions: vec![[1.0, 0.0, 0.0, 0.0]; 3],
            scales: vec![[1.0; 3]; 3],
            segments: vec![
                KeyframeSegment { interp: InterpMode::Linear },
                KeyframeSegment { interp: InterpMode::Linear },
            ],
            rest_translation: [0.0; 3],
        };
        assert_eq!(kf.time_range(), (0.0, 3.0));
    }

    #[test]
    fn moveby_delta_as_uniform_helper() {
        let u = MoveByDelta::Uniform([1.0, 2.0, 3.0]);
        assert_eq!(u.as_uniform(), Some([1.0, 2.0, 3.0]));
        let p = MoveByDelta::PerVertex(array![[1.0, 2.0, 3.0]]);
        assert_eq!(p.as_uniform(), None);
    }
}
