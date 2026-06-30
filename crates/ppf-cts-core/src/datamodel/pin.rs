// File: crates/ppf-cts-core/src/datamodel/pin.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pin operations + pin data. Direct port of the Operation hierarchy
// (frontend/_scene_.py:830-1108) and the small dataclass `PinData`.
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
//   `Scale`). The pair is folded into the `TransitionKind`
//   (defined in `easing.rs`) by the consuming kernels/solver when
//   they evaluate the easing, so the on-disk + PyO3 mirror keeps the
//   same flat field set as Python.
//
// This module is a pure data/schema layer: it defines the operation
// enum, the small dataclasses, and `time_range()`. The per-vertex
// transform math lives in `datamodel::pin_apply` (the single source of
// truth), driven by the preview kernels in
// `kernels::scene_build::pin_kernel` and by the solver's own apply path.

use ndarray::Array2;

use super::quat::{Quat, Vec3};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
