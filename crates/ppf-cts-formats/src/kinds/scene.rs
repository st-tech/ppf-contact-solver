// File: crates/ppf-cts-formats/src/kinds/scene.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Scene wire format encoded as CBOR.
//
// The Python producer at blender_addon/core/encoder/mesh.py:280-462
// emits a `list[dict]` of object groups. Each group has a `type` tag
// and an `object` array, where each object carries either a canonical
// mesh (vert + face/edge + optional uv/stitch) OR a `mesh_ref` UUID
// pointing at an earlier object in the same payload that holds the
// canonical mesh, plus a 4x4 world transform. Object types: STATIC |
// SOLID | SHELL | ROD.
//
// The Python consumer at frontend/_decoder_.py:851-1430 reads the
// same structure. The addon emits CBOR via `cbor2.dumps`, and we
// serialize/deserialize through serde with the envelope from
// `crate::envelope`.

use serde::{Deserialize, Serialize};

/// Top-level: list of object groups.
pub type ScenePayload = Vec<ObjectGroup>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectGroup {
    /// One of "STATIC", "SOLID", "SHELL", "ROD".
    /// Kept as a string (not enum) so unknown future types don't fail
    /// decode hard; the consumer logic in _decoder_.py branches on
    /// the string value at lines 986/1112, so we mirror that surface.
    #[serde(rename = "type")]
    pub group_type: String,

    #[serde(rename = "object")]
    pub objects: Vec<ObjectInfo>,
}

/// Object slot. `vert`+`face`/`edge` and `mesh_ref` are mutually
/// exclusive: producers fill exactly one branch (canonical mesh vs.
/// reference to an earlier object in the same payload that holds
/// the canonical mesh). The decoder picks at
/// frontend/_decoder_.py:1002-1030.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObjectInfo {
    pub name: String,
    pub uuid: String,

    // -- Canonical mesh (emitted iff `mesh_ref` is None) --
    /// Local-space vertex positions, shape (N, 3). float32 in pickle;
    /// addon emits via `arr.astype(np.float32).tolist()` for CBOR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vert: Option<Vec<[f32; 3]>>,

    /// SHELL only, optional: local-space reference vertices for the
    /// bending rest angle, shape (N, 3), same vertex order as `vert`.
    /// Present when the object opted into a per-object reference rest
    /// angle (see mesh.py `_encode_bend_reference_verts`); the solver
    /// computes this object's hinge rest angles from these positions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bend_rest_vert: Option<Vec<[f32; 3]>>,

    /// Triangulated faces, shape (M, 3). uint32 in pickle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face: Option<Vec<[u32; 3]>>,

    /// Rod edges, shape (K, 2). uint32 in pickle. Required when group
    /// type is ROD; absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge: Option<Vec<[u32; 2]>>,

    /// Per-face UV data (SHELL only). Length matches `face` row count.
    /// Each entry is the per-triangle UV layout the addon's
    /// `loop_triangulate_mesh` produces (one (3, 2) corner-UV block per
    /// loop triangle); variable inner shape, kept as a CBOR-flexible
    /// value to avoid coupling to UV layout choices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv: Option<ciborium::Value>,

    /// Detected stitch edges + weights. Producer emits the
    /// 2-tuple `(list[(int,int)], list[float])` that
    /// `detect_stitch_edges` returns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stitch: Option<(Vec<[i32; 2]>, Vec<f32>)>,

    // -- Mesh reference (emitted iff vert/face/edge are None) --
    /// UUID of the canonical mesh to reuse, set on duplicate
    /// instances. See mesh.py:331-339.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_ref: Option<String>,

    // -- Transform (always present when there's geometry) --
    /// 4x4 world matrix, row-major. float64 in pickle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<[[f64; 4]; 4]>,

    // -- STATIC group only: animation. Mutually exclusive; the
    // encoder picks fcurve-based `transform_animation` over
    // UI-assigned `static_ops`. --
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform_animation: Option<TransformAnimation>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_ops: Option<Vec<StaticOp>>,

    // -- Non-STATIC groups only: pin spec. Vertex indices to pin to
    // their world-transformed rest pose. --
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin: Option<Vec<i32>>,
}

/// Keyframe-driven STATIC animation. Decoder reads at
/// frontend/_decoder_.py:1156, 1202-1206.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformAnimation {
    /// Keyframe times in seconds.
    pub time: Vec<f32>,
    pub translation: Vec<[f32; 3]>,
    /// (x, y, z, w) quaternion components.
    pub quaternion: Vec<[f32; 4]>,
    pub scale: Vec<[f32; 3]>,
    /// Optional segment markers; producers may omit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<ciborium::Value>,
}

/// UI-assigned static operation. The discriminant `op_type` tells which
/// payload fields are set; we keep all of them optional and let the
/// consumer branch on `op_type`, mirroring frontend/_decoder_.py:
/// 1219-1239.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticOp {
    /// "MOVE_BY" | "SPIN" | "SCALE".
    pub op_type: String,
    pub t_start: f32,
    pub t_end: f32,
    /// Easing curve name, e.g. "linear". Producer always emits via
    /// `str(...).lower()` (mesh.py:385).
    pub transition: String,

    // MOVE_BY
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<[f32; 3]>,

    // SPIN
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular_velocity: Option<f32>,

    // SCALE
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factor: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{from_cbor, to_cbor};
    use crate::kinds::KIND_SCENE;

    #[test]
    fn scene_canonical_roundtrip() {
        let payload: ScenePayload = vec![ObjectGroup {
            group_type: "SHELL".into(),
            objects: vec![ObjectInfo {
                name: "cloth".into(),
                uuid: "abc-123".into(),
                vert: Some(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                face: Some(vec![[0, 1, 2]]),
                transform: Some([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
                ..Default::default()
            }],
        }];
        let bytes = to_cbor(KIND_SCENE, &payload).unwrap();
        let back: ScenePayload = from_cbor(KIND_SCENE, &bytes).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].group_type, "SHELL");
        assert_eq!(back[0].objects[0].uuid, "abc-123");
        assert_eq!(back[0].objects[0].face.as_deref().unwrap()[0], [0, 1, 2]);
    }

    #[test]
    fn scene_mesh_ref_roundtrip() {
        let payload: ScenePayload = vec![ObjectGroup {
            group_type: "SOLID".into(),
            objects: vec![ObjectInfo {
                name: "instance".into(),
                uuid: "def-456".into(),
                mesh_ref: Some("abc-123".into()),
                transform: Some([
                    [2.0, 0.0, 0.0, 1.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
                ..Default::default()
            }],
        }];
        let bytes = to_cbor(KIND_SCENE, &payload).unwrap();
        let back: ScenePayload = from_cbor(KIND_SCENE, &bytes).unwrap();
        let obj = &back[0].objects[0];
        assert_eq!(obj.mesh_ref.as_deref(), Some("abc-123"));
        assert!(obj.vert.is_none());
    }

    #[test]
    fn scene_static_op_spin() {
        let op = StaticOp {
            op_type: "SPIN".into(),
            t_start: 0.0,
            t_end: 5.0,
            transition: "linear".into(),
            axis: Some([0.0, 1.0, 0.0]),
            angular_velocity: Some(90.0),
            ..no_optionals()
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&op, &mut bytes).unwrap();
        let back: StaticOp = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(back.op_type, "SPIN");
        assert_eq!(back.angular_velocity, Some(90.0));
        assert!(back.delta.is_none());
        assert!(back.factor.is_none());
    }

    fn no_optionals() -> StaticOp {
        StaticOp {
            op_type: String::new(),
            t_start: 0.0,
            t_end: 0.0,
            transition: String::new(),
            delta: None,
            axis: None,
            angular_velocity: None,
            factor: None,
        }
    }
}
