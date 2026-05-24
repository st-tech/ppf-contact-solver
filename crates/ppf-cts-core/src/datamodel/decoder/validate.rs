// File: crates/ppf-cts-core/src/datamodel/decoder/validate.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Validation helpers the Python decoder calls before mutating its
// caller's scene/session. Each function takes raw inputs and returns
// `Result<_, DecoderValidationError>`, so the PyO3 wrappers in
// `crates/ppf-cts-py/src/decoder_py.rs` can map them to `PyValueError`
// without holding the GIL.
//
// These were originally inline in `datamodel/decoder.rs`. They share
// `DecoderValidationError` rather than the `&'static str` /
// `String` style used by `datamodel/validators.rs`, so they live here
// next to the rest of the decoder support code.

#[derive(Debug, thiserror::Error)]
pub enum DecoderValidationError {
    #[error("File must be a pickle file.")]
    NotPickleFile,
    #[error("Group parameters not found in the data.")]
    MissingGroupKey,
    #[error("Scene parameters not found in the data.")]
    MissingSceneKey,
    #[error("Pin on '{name}': torque cannot be mixed with move/spin/scale operations")]
    TorqueMixedWithKinematic { name: String },
    #[error("invisible {kind} thickness must be > 0 (got {value})")]
    InvisibleColliderThickness { kind: String, value: f64 },
    #[error("Cross-stitch entry missing source_uuid/target_uuid")]
    CrossStitchMissingUuids,
    #[error("Cross-stitch references unknown object: source='{src}' target='{tgt}'")]
    CrossStitchUnknownObject { src: String, tgt: String },
    #[error("Parameter group missing UUID list (third element); re-export from Blender with current addon")]
    ParamGroupMissingUuids,
    #[error("Object '{name}' has empty UUID in parameter data")]
    ParamObjectEmptyUuid { name: String },
    #[error("Object missing required 'name' field")]
    ObjectMissingName,
    #[error("Object '{name}' missing required 'uuid' field")]
    ObjectMissingUuid { name: String },
    #[error("Parameter data not set. Call set_path() first.")]
    ParamDataNotSet,
    #[error("Object data not found in the group.")]
    ObjectDataMissing,
    #[error("Scene must be populated before making the app")]
    SceneNotPopulated,
    #[error("STATIC object '{name}' has more than one motion source ({sources}); the encoder should have rejected this.")]
    StaticBothAnimAndOps { name: String, sources: String },
    #[error("Unknown static op type {op_type}")]
    UnknownStaticOpType { op_type: String },
    #[error("Edge data not found for rod object {name}")]
    RodMissingEdge { name: String },
    #[error("Unknown group type: {group_type}")]
    UnknownGroupType { group_type: String },
    #[error("Object '{name}' has mesh_ref='{mesh_ref}' that does not match any canonical mesh UUID")]
    UnknownMeshRef { name: String, mesh_ref: String },
    #[error("Object '{name}' (uuid={uuid}) has no mesh data and no valid mesh_ref")]
    ObjectNoMeshData { name: String, uuid: String },
}

/// Assert that a file path ends in `.pickle` (the on-disk extension
/// retained for backward compatibility with the CBOR envelope).
pub fn validate_pickle_extension(path: &str) -> Result<(), DecoderValidationError> {
    if path.ends_with(".pickle") {
        Ok(())
    } else {
        Err(DecoderValidationError::NotPickleFile)
    }
}

/// Confirm the `param.pickle` payload carries the two top-level keys
/// the rest of `ParamDecoder` blindly subscripts.
pub fn validate_param_top_keys(
    has_group: bool,
    has_scene: bool,
) -> Result<(), DecoderValidationError> {
    if !has_group {
        return Err(DecoderValidationError::MissingGroupKey);
    }
    if !has_scene {
        return Err(DecoderValidationError::MissingSceneKey);
    }
    Ok(())
}

/// Reject pin op-type combinations the simulator can't represent.
pub fn validate_pin_op_types(
    op_types: &[&str],
    pin_name: &str,
) -> Result<(), DecoderValidationError> {
    let has_torque = op_types.contains(&"torque");
    let has_kin = op_types
        .iter()
        .any(|t| matches!(*t, "spin" | "scale" | "move_by"));
    if has_torque && has_kin {
        Err(DecoderValidationError::TorqueMixedWithKinematic {
            name: pin_name.to_string(),
        })
    } else {
        Ok(())
    }
}

/// Mirror the assertion in `apply_invisible_colliders`: thickness must
/// be strictly positive, otherwise the simulator can't seed contact.
pub fn validate_invisible_collider_thickness(
    kind: &str,
    value: f64,
) -> Result<f64, DecoderValidationError> {
    if value > 0.0 {
        Ok(value)
    } else {
        Err(DecoderValidationError::InvisibleColliderThickness {
            kind: kind.to_string(),
            value,
        })
    }
}

/// Mirror the cross-stitch entry validation in
/// `BlenderApp._apply_explicit_cross_stitch`.
pub fn validate_cross_stitch_endpoints(
    source: &str,
    target: &str,
    has_source: bool,
    has_target: bool,
) -> Result<(), DecoderValidationError> {
    if source.is_empty() || target.is_empty() {
        return Err(DecoderValidationError::CrossStitchMissingUuids);
    }
    if !has_source || !has_target {
        return Err(DecoderValidationError::CrossStitchUnknownObject {
            src: source.to_string(),
            tgt: target.to_string(),
        });
    }
    Ok(())
}

/// Mirror the per-group validation in `ParamDecoder.apply_to_objects`:
/// the encoder writes `(params, objects, uuids)` triples; legacy
/// pickles without the third slot get a hard error directing the user
/// to re-export.
pub fn validate_param_group_has_uuids(len: usize) -> Result<(), DecoderValidationError> {
    if len <= 2 {
        Err(DecoderValidationError::ParamGroupMissingUuids)
    } else {
        Ok(())
    }
}

/// Reject empty UUIDs on the per-object pass of
/// `ParamDecoder.apply_to_objects`.
pub fn validate_param_object_uuid(
    obj_name: &str,
    uuid: &str,
) -> Result<(), DecoderValidationError> {
    if uuid.is_empty() {
        Err(DecoderValidationError::ParamObjectEmptyUuid {
            name: obj_name.to_string(),
        })
    } else {
        Ok(())
    }
}

/// Mirror the missing-name/uuid checks at the start of
/// `SceneDecoder.populate_objects`'s per-object loop.
pub fn validate_scene_object_identity(
    name: &str,
    uuid: &str,
) -> Result<(), DecoderValidationError> {
    if name.is_empty() {
        return Err(DecoderValidationError::ObjectMissingName);
    }
    if uuid.is_empty() {
        return Err(DecoderValidationError::ObjectMissingUuid {
            name: name.to_string(),
        });
    }
    Ok(())
}

/// Reject the encoder bug where a STATIC object carries more than one
/// motion source. The three sources are:
///   * `transform_animation` (sparse T/R/S keyframes from object/parent
///     fcurves);
///   * `static_ops` (UI-authored move/spin/scale ops);
///   * `static_deform_animation` (per-frame depsgraph-baked vertex
///     buffer captured by the Capture Deformation operator).
///
/// At most one may be present per object. The deform cache always wins
/// over the other two when present, because the depsgraph already
/// composed those into its evaluation before the cache was captured.
pub fn validate_static_anim_xor_ops(
    name: &str,
    has_anim: bool,
    has_ops: bool,
    has_deform: bool,
) -> Result<(), DecoderValidationError> {
    let n = (has_anim as u32) + (has_ops as u32) + (has_deform as u32);
    if n <= 1 {
        return Ok(());
    }
    let mut active: Vec<&str> = Vec::with_capacity(3);
    if has_anim {
        active.push("transform_animation");
    }
    if has_ops {
        active.push("static_ops");
    }
    if has_deform {
        active.push("static_deform_animation");
    }
    Err(DecoderValidationError::StaticBothAnimAndOps {
        name: name.to_string(),
        sources: active.join(" + "),
    })
}

/// Validate a STATIC op type tag.
pub fn validate_static_op_type(op_type: &str) -> Result<(), DecoderValidationError> {
    match op_type {
        "MOVE_BY" | "SPIN" | "SCALE" => Ok(()),
        other => Err(DecoderValidationError::UnknownStaticOpType {
            op_type: format!("'{other}'"),
        }),
    }
}

/// Reject ROD objects without an edge buffer.
pub fn validate_rod_has_edges(name: &str, has_edge: bool) -> Result<(), DecoderValidationError> {
    if has_edge {
        Ok(())
    } else {
        Err(DecoderValidationError::RodMissingEdge {
            name: name.to_string(),
        })
    }
}

/// Reject groups with an unknown discriminator string.
pub fn validate_group_type(group_type: &str) -> Result<(), DecoderValidationError> {
    match group_type {
        "STATIC" | "SOLID" | "SHELL" | "ROD" => Ok(()),
        other => Err(DecoderValidationError::UnknownGroupType {
            group_type: other.to_string(),
        }),
    }
}

/// Reject `mesh_ref` values that do not match any canonical mesh
/// UUID in the build plan.
pub fn validate_mesh_ref_known(
    name: &str,
    mesh_ref: &str,
    found: bool,
) -> Result<(), DecoderValidationError> {
    if found {
        Ok(())
    } else {
        Err(DecoderValidationError::UnknownMeshRef {
            name: name.to_string(),
            mesh_ref: mesh_ref.to_string(),
        })
    }
}

/// Reject objects that have neither inline `vert` nor a valid
/// `mesh_ref`.
pub fn validate_object_has_mesh(
    name: &str,
    uuid: &str,
    has_mesh: bool,
) -> Result<(), DecoderValidationError> {
    if has_mesh {
        Ok(())
    } else {
        Err(DecoderValidationError::ObjectNoMeshData {
            name: name.to_string(),
            uuid: uuid.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_pickle_extension_accepts_pickle() {
        assert!(validate_pickle_extension("/tmp/data.pickle").is_ok());
        assert!(validate_pickle_extension("data.pickle").is_ok());
    }

    #[test]
    fn validate_pickle_extension_rejects_others() {
        assert!(matches!(
            validate_pickle_extension("data.cbor"),
            Err(DecoderValidationError::NotPickleFile)
        ));
        assert!(matches!(
            validate_pickle_extension("data.pickle.tmp"),
            Err(DecoderValidationError::NotPickleFile)
        ));
    }

    #[test]
    fn validate_param_top_keys_requires_both() {
        assert!(validate_param_top_keys(true, true).is_ok());
        assert!(matches!(
            validate_param_top_keys(false, true),
            Err(DecoderValidationError::MissingGroupKey)
        ));
        assert!(matches!(
            validate_param_top_keys(true, false),
            Err(DecoderValidationError::MissingSceneKey)
        ));
    }

    #[test]
    fn validate_pin_op_types_blocks_torque_with_kinematic() {
        assert!(validate_pin_op_types(&["spin", "scale"], "p").is_ok());
        assert!(validate_pin_op_types(&["torque"], "p").is_ok());
        assert!(matches!(
            validate_pin_op_types(&["torque", "move_by"], "p"),
            Err(DecoderValidationError::TorqueMixedWithKinematic { .. })
        ));
    }

    #[test]
    fn validate_invisible_thickness_positive() {
        assert_eq!(
            validate_invisible_collider_thickness("wall", 0.5).unwrap(),
            0.5
        );
        assert!(matches!(
            validate_invisible_collider_thickness("wall", 0.0),
            Err(DecoderValidationError::InvisibleColliderThickness { .. })
        ));
        assert!(matches!(
            validate_invisible_collider_thickness("sphere", -0.1),
            Err(DecoderValidationError::InvisibleColliderThickness { .. })
        ));
    }

    #[test]
    fn validate_cross_stitch_endpoints_checks_uuids() {
        assert!(validate_cross_stitch_endpoints("a", "b", true, true).is_ok());
        assert!(matches!(
            validate_cross_stitch_endpoints("", "b", true, true),
            Err(DecoderValidationError::CrossStitchMissingUuids)
        ));
        assert!(matches!(
            validate_cross_stitch_endpoints("a", "b", false, true),
            Err(DecoderValidationError::CrossStitchUnknownObject { .. })
        ));
    }

    #[test]
    fn validate_param_group_has_uuids_checks_third_slot() {
        assert!(validate_param_group_has_uuids(3).is_ok());
        assert!(validate_param_group_has_uuids(5).is_ok());
        assert!(matches!(
            validate_param_group_has_uuids(2),
            Err(DecoderValidationError::ParamGroupMissingUuids)
        ));
    }

    #[test]
    fn validate_param_object_uuid_rejects_empty() {
        assert!(validate_param_object_uuid("obj", "abcd").is_ok());
        assert!(matches!(
            validate_param_object_uuid("obj", ""),
            Err(DecoderValidationError::ParamObjectEmptyUuid { .. })
        ));
    }

    #[test]
    fn validate_static_anim_xor_ops_rejects_overlap() {
        // Zero or one source: always OK.
        assert!(validate_static_anim_xor_ops("a", false, false, false).is_ok());
        assert!(validate_static_anim_xor_ops("a", true, false, false).is_ok());
        assert!(validate_static_anim_xor_ops("a", false, true, false).is_ok());
        assert!(validate_static_anim_xor_ops("a", false, false, true).is_ok());
        // Any pair: rejected.
        assert!(matches!(
            validate_static_anim_xor_ops("a", true, true, false),
            Err(DecoderValidationError::StaticBothAnimAndOps { .. })
        ));
        assert!(matches!(
            validate_static_anim_xor_ops("a", true, false, true),
            Err(DecoderValidationError::StaticBothAnimAndOps { .. })
        ));
        assert!(matches!(
            validate_static_anim_xor_ops("a", false, true, true),
            Err(DecoderValidationError::StaticBothAnimAndOps { .. })
        ));
        // All three: rejected, error names every conflicting source.
        match validate_static_anim_xor_ops("a", true, true, true) {
            Err(DecoderValidationError::StaticBothAnimAndOps { sources, .. }) => {
                assert!(sources.contains("transform_animation"));
                assert!(sources.contains("static_ops"));
                assert!(sources.contains("static_deform_animation"));
            }
            other => panic!("expected StaticBothAnimAndOps, got {other:?}"),
        }
    }

    #[test]
    fn validate_static_op_type_accepts_known() {
        for k in ["MOVE_BY", "SPIN", "SCALE"] {
            assert!(validate_static_op_type(k).is_ok(), "{k}");
        }
        assert!(matches!(
            validate_static_op_type("WIGGLE"),
            Err(DecoderValidationError::UnknownStaticOpType { .. })
        ));
    }

    #[test]
    fn validate_rod_has_edges_required() {
        assert!(validate_rod_has_edges("rod", true).is_ok());
        assert!(matches!(
            validate_rod_has_edges("rod", false),
            Err(DecoderValidationError::RodMissingEdge { .. })
        ));
    }

    #[test]
    fn validate_group_type_accepts_known() {
        for k in ["STATIC", "SOLID", "SHELL", "ROD"] {
            assert!(validate_group_type(k).is_ok(), "{k}");
        }
        assert!(matches!(
            validate_group_type("FLUID"),
            Err(DecoderValidationError::UnknownGroupType { .. })
        ));
    }

    #[test]
    fn validate_mesh_ref_and_object_mesh() {
        assert!(validate_mesh_ref_known("o", "u", true).is_ok());
        assert!(matches!(
            validate_mesh_ref_known("o", "u", false),
            Err(DecoderValidationError::UnknownMeshRef { .. })
        ));
        assert!(validate_object_has_mesh("o", "u", true).is_ok());
        assert!(matches!(
            validate_object_has_mesh("o", "u", false),
            Err(DecoderValidationError::ObjectNoMeshData { .. })
        ));
    }

    #[test]
    fn validate_scene_object_identity_requires_name_and_uuid() {
        assert!(validate_scene_object_identity("obj", "abcd").is_ok());
        assert!(matches!(
            validate_scene_object_identity("", "abcd"),
            Err(DecoderValidationError::ObjectMissingName)
        ));
        assert!(matches!(
            validate_scene_object_identity("obj", ""),
            Err(DecoderValidationError::ObjectMissingUuid { .. })
        ));
    }
}
