// File: crates/ppf-cts-formats/src/kinds/param.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Param wire format: the CBOR replacement for param.pickle.
//
// Producer: blender_addon/core/encoder/params.py:_build_param_dict()
// (around line 416). Consumer: frontend/_decoder_.py ParamDecoder
// (around line 449).
//
// Numeric width policy: every float on the wire is f64. The addon
// stamps many params with np.float32 today, but `cbor2.dumps` re-emits
// Python `float` (= double) so the wire never carries CBOR-32 floats.
// Internal solver code can still narrow to f32 where it wants; the
// schema just declares what crosses the language boundary.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParamPayload {
    pub scene: SceneParams,

    /// Per-group params. Each entry is a 3-tuple
    /// `(params, object_names, object_uuids)`. Encoded as a CBOR
    /// 3-array.
    pub group: Vec<GroupTriple>,

    /// Pin keyframes. Outer key is object UUID; inner key is vertex
    /// index. cbor2 emits Python int dict-keys as CBOR integer keys,
    /// which serde reads here as `i32`.
    #[serde(default)]
    pub pin_config: BTreeMap<String, BTreeMap<i32, PinData>>,

    /// Non-cross-stitch alias merges. 5-tuple
    /// `(name_a, name_b, stitch_stiffness, uuid_a, uuid_b)` where the
    /// trailing UUID slots are optional.
    #[serde(default)]
    pub merge_pairs: Vec<MergePair>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explicit_merge_pairs: Option<Vec<ExplicitMergePair>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_stitch: Option<Vec<CrossStitch>>,

    /// Dynamic params (cloth-only). Heterogeneous structure where
    /// `value[0]` is initial value and `value[1:]` are `(time, value,
    /// is_hold)` triples. Kept as opaque CBOR until we have a concrete
    /// consumer in Rust.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dyn_param: Option<BTreeMap<String, ciborium::Value>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invisible_colliders: Option<InvisibleColliders>,
}

pub type GroupTriple = (GroupParams, Vec<String>, Vec<String>);
pub type MergePair = (String, String, f64, Option<String>, Option<String>);

/// Global sim params. Field names mirror the addon's kebab-case keys.
/// Optional fields here are ones the producer skips conditionally
/// (e.g. `inactive-momentum` only when there's a SHELL group).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", default)]
pub struct SceneParams {
    pub dt: f64,
    pub min_newton_steps: i32,
    pub air_density: f64,
    pub air_friction: f64,
    /// "min" | "max" | "avg", case-folded by producer (params.py:74).
    pub friction_mode: String,
    /// Solver-frame (Y-up) gravity. Producer applies `_swap_axes`
    /// from Blender's Z-up before emitting.
    pub gravity: [f64; 3],
    /// Solver-frame wind (also `_swap_axes`'d).
    pub wind: [f64; 3],
    /// Frame count in solver convention (Blender N → solver N-1).
    pub frames: i32,
    pub fps: i32,
    pub csrmat_max_nnz: i64,
    pub isotropic_air_friction: f64,
    pub auto_save: i32,
    pub line_search_max_t: f64,
    pub constraint_ghat: f64,
    pub cg_max_iter: i32,
    pub cg_tol: f64,
    pub include_face_mass: bool,
    pub disable_contact: bool,
    pub stitch_stiffness: f64,

    /// `inactive_momentum_frames / fps` when there's a SHELL group;
    /// absent otherwise (params.py:90-91).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inactive_momentum: Option<f64>,
}

/// Per-group material + dynamics. The producer (params.py:240-292)
/// builds a flat dict, then strips keys not in `active_entries[type]`
/// so any single GroupParams may have only a subset of these
/// populated. We declare every union field as Option so the same
/// struct decodes any group type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", default)]
pub struct GroupParams {
    /// Deformation model: "arap" | "snhk" | "baraff-witkin" | "unknown".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub density: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub young_mod: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poiss_rat: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub friction: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact_gap: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact_offset: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub bend: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strain_limit: Option<f64>,

    /// Plasticity (TRI/TET).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plasticity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plasticity_threshold: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub bend_plasticity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bend_plasticity_threshold: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub bend_rest_from_geometry: Option<f64>,

    /// SHELL only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shrink_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shrink_y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pressure: Option<f64>,

    /// SOLID only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shrink: Option<f64>,

    /// ROD only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length_factor: Option<f64>,

    /// Initial velocity per UUID at frame 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity: Option<BTreeMap<String, [f64; 3]>>,

    /// Velocity keyframes after frame 1: `(time_seconds, velocity)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub velocity_schedule: Option<BTreeMap<String, Vec<(f64, [f64; 3])>>>,

    /// Per-object collision active windows: `(t_start, t_end)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collision_windows: Option<BTreeMap<String, Vec<(f64, f64)>>>,

    /// fTetWild kwargs (SOLID only). Opaque to Rust until the solver
    /// owns its own tetrahedralizer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ftetwild: Option<ciborium::Value>,
}

/// One pinned vertex's full keyframe spec. Decoder reads at
/// frontend/_decoder_.py:642-794.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PinData {
    pub unpin_time: Option<f64>,
    pub pull_strength: Option<f64>,
    pub pin_group_id: Option<i32>,

    /// Per-vertex animation keyframes. Outer key is the *target*
    /// vertex index this pin animates against.
    pub pin_anim: Option<BTreeMap<i32, PinAnim>>,

    /// Spin / scale / move_by / torque ops applied to this pin.
    pub operations: Vec<PinOperation>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PinAnim {
    pub time: Vec<f64>,
    pub position: Vec<[f64; 3]>,
}

/// Tagged operation; CBOR encodes via the `type` discriminant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PinOperation {
    Spin(SpinOp),
    Scale(ScaleOp),
    MoveBy(MoveByOp),
    Torque(TorqueOp),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CommonOpFields {
    pub t_start: f64,
    pub t_end: f64,
    pub transition: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SpinOp {
    pub t_start: f64,
    pub t_end: f64,
    pub transition: String,
    pub center: [f64; 3],
    pub center_mode: String,
    pub axis: [f64; 3],
    pub angular_velocity: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ScaleOp {
    pub t_start: f64,
    pub t_end: f64,
    pub transition: String,
    pub center: [f64; 3],
    pub center_mode: String,
    pub factor: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MoveByOp {
    pub t_start: f64,
    pub t_end: f64,
    pub transition: String,
    pub delta: [f64; 3],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TorqueOp {
    pub t_start: f64,
    pub t_end: f64,
    pub transition: String,
    pub magnitude: f64,
    pub axis_component: i32,
    pub hint_vertex: i32,
}

/// SHELL+ROD cross-merge. Producer: params.py:340-377.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplicitMergePair {
    pub source_uuid: String,
    pub target_uuid: String,
    /// `[source_vertex, target_vertex]` index pairs (best-bary chosen
    /// by the producer).
    pub pairs: Vec<[i32; 2]>,
}

/// Soft cross-stitch (anything not SHELL+SHELL or ROD+ROD). Producer:
/// params.py:380-413. Producer emits arbitrary keys from
/// `cross_stitch_json` plus `target_points` and `stitch_stiffness`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CrossStitch {
    pub source_uuid: String,
    pub target_uuid: String,

    /// Source vertex / face indices. Inner shape is producer-dependent
    /// (e.g. `(N, 4)` for shell-onto-tet); kept opaque here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ind: Option<ciborium::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub w: Option<ciborium::Value>,

    /// World-space anchor points (SOLID target only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_points: Option<Vec<[f64; 3]>>,

    pub stitch_stiffness: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InvisibleColliders {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub walls: Vec<Wall>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spheres: Vec<Sphere>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Wall {
    pub position: [f64; 3],
    pub normal: [f64; 3],
    pub contact_gap: f64,
    pub friction: f64,
    pub active_duration: f64,
    pub thickness: f64,
    pub keyframes: Vec<WallKeyframe>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WallKeyframe {
    pub position: [f64; 3],
    pub time: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Sphere {
    pub position: [f64; 3],
    pub radius: f64,
    pub hemisphere: bool,
    pub invert: bool,
    pub contact_gap: f64,
    pub friction: f64,
    pub active_duration: f64,
    pub thickness: f64,
    pub keyframes: Vec<SphereKeyframe>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SphereKeyframe {
    pub position: [f64; 3],
    pub radius: f64,
    pub time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{from_cbor, to_cbor};
    use crate::kinds::KIND_PARAM;

    #[test]
    fn param_minimal_roundtrip() {
        let mut payload = ParamPayload {
            scene: SceneParams {
                dt: 1e-3,
                gravity: [0.0, -9.8, 0.0],
                frames: 60,
                fps: 60,
                friction_mode: "min".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        payload.group.push((
            GroupParams {
                model: Some("baraff-witkin".into()),
                density: Some(1.0),
                young_mod: Some(1000.0),
                ..Default::default()
            },
            vec!["cloth".into()],
            vec!["uuid-1".into()],
        ));

        let bytes = to_cbor(KIND_PARAM, &payload).unwrap();
        let back: ParamPayload = from_cbor(KIND_PARAM, &bytes).unwrap();
        assert_eq!(back.scene.dt, 1e-3);
        assert_eq!(back.scene.frames, 60);
        assert_eq!(back.group.len(), 1);
        assert_eq!(back.group[0].0.model.as_deref(), Some("baraff-witkin"));
        assert_eq!(back.group[0].2, vec!["uuid-1"]);
    }

    #[test]
    fn pin_operation_tagging() {
        let op = PinOperation::Spin(SpinOp {
            t_start: 0.0,
            t_end: 1.0,
            transition: "linear".into(),
            center: [0.0, 0.0, 0.0],
            center_mode: "absolute".into(),
            axis: [0.0, 1.0, 0.0],
            angular_velocity: 360.0,
        });
        let mut bytes = Vec::new();
        ciborium::into_writer(&op, &mut bytes).unwrap();
        let back: PinOperation = ciborium::from_reader(bytes.as_slice()).unwrap();
        match back {
            PinOperation::Spin(s) => assert_eq!(s.angular_velocity, 360.0),
            other => panic!("expected Spin, got {other:?}"),
        }
    }

    #[test]
    fn pin_config_int_keys() {
        let mut payload = ParamPayload::default();
        let mut by_vert = BTreeMap::new();
        by_vert.insert(
            42,
            PinData {
                unpin_time: Some(1.5),
                ..Default::default()
            },
        );
        payload.pin_config.insert("uuid-x".into(), by_vert);

        let bytes = to_cbor(KIND_PARAM, &payload).unwrap();
        let back: ParamPayload = from_cbor(KIND_PARAM, &bytes).unwrap();
        let pd = back
            .pin_config
            .get("uuid-x")
            .and_then(|m| m.get(&42))
            .expect("vertex 42 entry should round-trip");
        assert_eq!(pd.unpin_time, Some(1.5));
    }

    #[test]
    fn merge_pair_optional_uuids() {
        let payload = ParamPayload {
            merge_pairs: vec![
                ("a".into(), "b".into(), 1.0, Some("ua".into()), Some("ub".into())),
                // name strings only, uuids absent
                ("c".into(), "d".into(), 0.5, None, None),
            ],
            ..Default::default()
        };
        let bytes = to_cbor(KIND_PARAM, &payload).unwrap();
        let back: ParamPayload = from_cbor(KIND_PARAM, &bytes).unwrap();
        assert_eq!(back.merge_pairs.len(), 2);
        assert!(back.merge_pairs[1].3.is_none());
    }
}
