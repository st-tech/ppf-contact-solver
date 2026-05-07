// File: crates/ppf-cts-core/src/datamodel/validators.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Cold-tail validation helpers used by the Python frontend's `Scene`,
// `Object`, `Wall`, `Sphere`, `ParamManager`, and `SessionManager`
// builders. Each function takes raw inputs and returns a validated
// outcome, so the Python wrappers can replace their loop bodies with
// a single Rust call.
//
// These were originally a set of ~17 free functions inline in
// `datamodel::scene`. They have no scene-shape dependency; grouping
// them here keeps `scene.rs` focused on the `Scene` data model.
//
// All functions return `Result<_, ValidatorError>` (or, in the merge
// pair case, a tuple variant carrying the bad index + uuids). The
// `errors.rs` module in `ppf-cts-py` maps each variant onto the
// appropriate `PyErr` class so the wrapper layer can `?`-propagate.

/// Typed errors emitted by every validator in this module. Mirrors
/// the style used by `datamodel::decoder::validate::DecoderValidationError`.
#[derive(Debug, thiserror::Error)]
pub enum ValidatorError {
    /// Raised by `validate_merge_pair_uuids` for the first invalid
    /// `(source_uuid, target_uuid)` entry.
    #[error("merge pair [{i}] missing required source/target uuid: source_uuid='{src}' target_uuid='{tgt}'")]
    MergePairUuid { i: usize, src: String, tgt: String },
    #[error("set_surface_map requires a non-empty object UUID key")]
    SurfaceMapKeyEmpty,
    #[error("Key cannot contain underscore. Use '-' instead.")]
    ParamKeyHasUnderscore,
    #[error("t_end must be greater than t_start")]
    TimeWindowNotPositive,
    #[error("time must be greater than the last time")]
    ColliderTimeNotIncreasing,
    #[error("time must be greater than the last time. last time is {prev:.6}")]
    SphereTimeNotIncreasing { prev: f64 },
    #[error("unknown parameter {name}")]
    UnknownParamName { name: String },
    #[error("{kind} already exists")]
    ColliderAlreadyAdded { kind: String },
    #[error("invalid axis")]
    InvalidObjectRotateAxis,
    #[error("object is static")]
    ObjectIsStatic,
    #[error("Ind not found in stitch")]
    StitchMissingInd,
    #[error("W not found in stitch")]
    StitchMissingW,
    #[error("UV coordinates are only applicable to triangular meshes")]
    SetUvNonTriangular,
    #[error("already normalized")]
    AlreadyNormalized,
    #[error("object {name} does not exist")]
    SceneSelectMissing { name: String },
    #[error("Key {key} does not exist")]
    ParamKeyMissing { key: String },
    #[error("Time must be increasing")]
    ParamTimeNotIncreasing,
    #[error("Session {name} does not exist")]
    SessionMissing { name: String },
}

/// Validate a list of `(source_uuid, target_uuid)` pairs the same way
/// `Scene.set_explicit_merge_pairs` does. Returns the index of the
/// first invalid entry (and its uuids), or `Ok(())` when all are good.
/// Mirrors the validation portion only; the caller still owns the pair
/// list itself.
pub fn validate_merge_pair_uuids(pairs: &[(String, String)]) -> Result<(), ValidatorError> {
    for (i, (src, tgt)) in pairs.iter().enumerate() {
        if src.is_empty() || tgt.is_empty() {
            return Err(ValidatorError::MergePairUuid {
                i,
                src: src.clone(),
                tgt: tgt.clone(),
            });
        }
    }
    Ok(())
}

/// Validate a non-empty surface-map key. Mirrors the `if not name:`
/// guard in `Scene.set_surface_map`.
pub fn validate_surface_map_key(name: &str) -> Result<(), ValidatorError> {
    if name.is_empty() {
        Err(ValidatorError::SurfaceMapKeyEmpty)
    } else {
        Ok(())
    }
}

/// Validate a "may not contain underscore" key. Mirrors the guard in
/// `ParamManager.set` and is reused by the WallParam/SphereParam
/// validators.
pub fn validate_param_key_no_underscore(key: &str) -> Result<(), ValidatorError> {
    if key.contains('_') {
        Err(ValidatorError::ParamKeyHasUnderscore)
    } else {
        Ok(())
    }
}

/// Validate a transform-animation time window: `t_end > t_start`.
/// Mirrors the guard in `Object.move`, `Object.animate_rotate`, and
/// `PinHolder.move_by`/`move_to`/`scale`.
pub fn validate_time_window(t_start: f64, t_end: f64) -> Result<(), ValidatorError> {
    if t_end <= t_start {
        Err(ValidatorError::TimeWindowNotPositive)
    } else {
        Ok(())
    }
}

/// Validate that `time` is strictly greater than `prev_time`.
/// Used by both Wall and Sphere; the format differs for Sphere because
/// the message includes the previous time.
pub fn validate_collider_time(prev_time: f64, time: f64) -> Result<(), ValidatorError> {
    if time <= prev_time {
        Err(ValidatorError::ColliderTimeNotIncreasing)
    } else {
        Ok(())
    }
}

/// Strict-monotonic time check that includes the prior time in the
/// message. Mirrors the `Sphere._check_time` formatter exactly.
pub fn validate_sphere_time(prev_time: f64, time: f64) -> Result<(), ValidatorError> {
    if time <= prev_time {
        Err(ValidatorError::SphereTimeNotIncreasing { prev: prev_time })
    } else {
        Ok(())
    }
}

/// Validate that `name` is in `allowed`. Mirrors the guard in
/// `WallParam.set` and `SphereParam.set` ("unknown parameter X").
/// Returns `Ok(())` when the key already exists.
pub fn validate_known_param_name(name: &str, allowed: &[&str]) -> Result<(), ValidatorError> {
    if allowed.contains(&name) {
        Ok(())
    } else {
        Err(ValidatorError::UnknownParamName {
            name: name.to_string(),
        })
    }
}

/// Guard against re-adding to a non-empty entry list. Mirrors the
/// "wall already exists" / "sphere already exists" branches in
/// `Wall.add` and `Sphere.add`. `kind` is the noun used in the message.
pub fn validate_collider_not_already_added(already: bool, kind: &str) -> Result<(), ValidatorError> {
    if already {
        Err(ValidatorError::ColliderAlreadyAdded {
            kind: kind.to_string(),
        })
    } else {
        Ok(())
    }
}

/// Validate the axis string the Python `Object.rotate` accepts. Mirrors
/// `if a not in ("x", "y", "z"): raise Exception("invalid axis")`.
/// Returns the lowercased single character on success.
pub fn validate_object_rotate_axis(axis: &str) -> Result<char, ValidatorError> {
    let lower = axis.to_ascii_lowercase();
    match lower.as_str() {
        "x" => Ok('x'),
        "y" => Ok('y'),
        "z" => Ok('z'),
        _ => Err(ValidatorError::InvalidObjectRotateAxis),
    }
}

/// Per-axis stitch validator. Mirrors the chain in `Object.stitch`:
///   * raises "object is static" first,
///   * then "Ind not found in stitch" if the asset lacks an `Ind` key,
///   * then "W not found in stitch" if the asset lacks a `W` key.
pub fn validate_stitch_attach(
    is_static: bool,
    has_ind: bool,
    has_w: bool,
) -> Result<(), ValidatorError> {
    if is_static {
        Err(ValidatorError::ObjectIsStatic)
    } else if !has_ind {
        Err(ValidatorError::StitchMissingInd)
    } else if !has_w {
        Err(ValidatorError::StitchMissingW)
    } else {
        Ok(())
    }
}

/// Validate `Object.set_uv`: the obj_type must be `"tri"`. Mirrors the
/// raise in the Python source.
pub fn validate_set_uv_obj_type(obj_type: &str) -> Result<(), ValidatorError> {
    if obj_type == "tri" {
        Ok(())
    } else {
        Err(ValidatorError::SetUvNonTriangular)
    }
}

/// Validate `Object.normalize`: refuses to normalize twice. Mirrors
/// `if self._normalize: raise Exception("already normalized")`.
pub fn validate_object_normalize(already_normalized: bool) -> Result<(), ValidatorError> {
    if already_normalized {
        Err(ValidatorError::AlreadyNormalized)
    } else {
        Ok(())
    }
}

/// Validate `Object.velocity`: refuses to set on a static object.
/// Mirrors the corresponding raise.
pub fn validate_object_not_static(is_static: bool) -> Result<(), ValidatorError> {
    if is_static {
        Err(ValidatorError::ObjectIsStatic)
    } else {
        Ok(())
    }
}

/// `Scene.select` lookup. Mirrors `if name not in self._object: raise
/// Exception(f"object {name} does not exist")`.
pub fn validate_scene_select(exists: bool, name: &str) -> Result<(), ValidatorError> {
    if exists {
        Ok(())
    } else {
        Err(ValidatorError::SceneSelectMissing {
            name: name.to_string(),
        })
    }
}

/// `ParamManager.dyn` lookup. Mirrors `if key not in self._param.key_list():
/// raise ValueError(f"Key {key} does not exist")`.
pub fn validate_param_key_exists(exists: bool, key: &str) -> Result<(), ValidatorError> {
    if exists {
        Ok(())
    } else {
        Err(ValidatorError::ParamKeyMissing {
            key: key.to_string(),
        })
    }
}

/// `ParamManager.time` strictly increasing check. Mirrors `if time <=
/// self._time: raise ValueError("Time must be increasing")`.
pub fn validate_param_time_strictly_increasing(prev: f64, next: f64) -> Result<(), ValidatorError> {
    if next <= prev {
        Err(ValidatorError::ParamTimeNotIncreasing)
    } else {
        Ok(())
    }
}

/// `SessionManager.select` lookup. Mirrors `if name not in self._sessions:
/// raise ValueError(f"Session {name} does not exist")`.
pub fn validate_session_exists(exists: bool, name: &str) -> Result<(), ValidatorError> {
    if exists {
        Ok(())
    } else {
        Err(ValidatorError::SessionMissing {
            name: name.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_pair_validation_ok() {
        let pairs = vec![("a".to_string(), "b".to_string()), ("c".into(), "d".into())];
        assert!(validate_merge_pair_uuids(&pairs).is_ok());
    }

    #[test]
    fn merge_pair_validation_reports_first_bad_index() {
        let pairs = vec![
            ("a".to_string(), "b".to_string()),
            ("".to_string(), "x".to_string()),
            ("y".to_string(), "".to_string()),
        ];
        let err = validate_merge_pair_uuids(&pairs).unwrap_err();
        assert!(matches!(err, ValidatorError::MergePairUuid { i: 1, .. }));
    }

    #[test]
    fn surface_map_key_validation() {
        assert!(validate_surface_map_key("name").is_ok());
        assert!(matches!(
            validate_surface_map_key(""),
            Err(ValidatorError::SurfaceMapKeyEmpty)
        ));
    }

    #[test]
    fn param_key_underscore_check() {
        assert!(validate_param_key_no_underscore("frames").is_ok());
        assert!(validate_param_key_no_underscore("min-newton").is_ok());
        assert!(matches!(
            validate_param_key_no_underscore("min_newton"),
            Err(ValidatorError::ParamKeyHasUnderscore)
        ));
    }

    #[test]
    fn time_window_check() {
        assert!(validate_time_window(0.0, 1.0).is_ok());
        assert!(validate_time_window(1.0, 1.0).is_err());
        assert!(validate_time_window(2.0, 1.0).is_err());
    }

    #[test]
    fn collider_time_check_basic() {
        assert!(validate_collider_time(1.0, 2.0).is_ok());
        let err = validate_collider_time(1.0, 1.0).unwrap_err();
        assert!(err.to_string().contains("greater than the last time"));
    }

    #[test]
    fn sphere_time_check_includes_prev() {
        let err = validate_sphere_time(1.5, 1.0).unwrap_err();
        assert!(err.to_string().contains("1.500000"));
    }

    #[test]
    fn known_param_name_validation() {
        let allowed = ["contact-gap", "friction", "active-duration", "thickness"];
        assert!(validate_known_param_name("friction", &allowed).is_ok());
        let err = validate_known_param_name("bogus", &allowed).unwrap_err();
        assert!(err.to_string().contains("unknown parameter bogus"));
    }

    #[test]
    fn collider_already_added_check() {
        assert!(validate_collider_not_already_added(false, "wall").is_ok());
        let err = validate_collider_not_already_added(true, "sphere").unwrap_err();
        assert!(err.to_string().contains("sphere already exists"));
    }

    #[test]
    fn object_rotate_axis_validation() {
        assert_eq!(validate_object_rotate_axis("X").unwrap(), 'x');
        assert_eq!(validate_object_rotate_axis("y").unwrap(), 'y');
        assert!(matches!(
            validate_object_rotate_axis("w"),
            Err(ValidatorError::InvalidObjectRotateAxis)
        ));
    }

    #[test]
    fn stitch_attach_validation_orders() {
        // Static beats both missing keys.
        assert!(matches!(
            validate_stitch_attach(true, false, false),
            Err(ValidatorError::ObjectIsStatic)
        ));
        // Missing Ind beats missing W.
        assert!(matches!(
            validate_stitch_attach(false, false, false),
            Err(ValidatorError::StitchMissingInd)
        ));
        assert!(matches!(
            validate_stitch_attach(false, true, false),
            Err(ValidatorError::StitchMissingW)
        ));
        assert!(validate_stitch_attach(false, true, true).is_ok());
    }

    #[test]
    fn set_uv_obj_type_check() {
        assert!(validate_set_uv_obj_type("tri").is_ok());
        assert!(validate_set_uv_obj_type("tet").is_err());
        assert!(validate_set_uv_obj_type("rod").is_err());
    }

    #[test]
    fn object_normalize_check() {
        assert!(validate_object_normalize(false).is_ok());
        assert!(matches!(
            validate_object_normalize(true),
            Err(ValidatorError::AlreadyNormalized)
        ));
    }

    #[test]
    fn object_static_check() {
        assert!(validate_object_not_static(false).is_ok());
        assert!(matches!(
            validate_object_not_static(true),
            Err(ValidatorError::ObjectIsStatic)
        ));
    }

    #[test]
    fn scene_select_validator() {
        assert!(validate_scene_select(true, "sheet").is_ok());
        let err = validate_scene_select(false, "missing").unwrap_err();
        assert!(err.to_string().contains("object missing does not exist"));
    }

    #[test]
    fn param_key_exists_validator() {
        assert!(validate_param_key_exists(true, "frames").is_ok());
        let err = validate_param_key_exists(false, "bogus").unwrap_err();
        assert_eq!(err.to_string(), "Key bogus does not exist");
    }

    #[test]
    fn param_time_strictly_increasing() {
        assert!(validate_param_time_strictly_increasing(0.0, 1.0).is_ok());
        assert!(validate_param_time_strictly_increasing(1.0, 1.0).is_err());
        assert!(validate_param_time_strictly_increasing(2.0, 1.0).is_err());
    }

    #[test]
    fn session_select_validator() {
        assert!(validate_session_exists(true, "run-A").is_ok());
        let err = validate_session_exists(false, "missing").unwrap_err();
        assert_eq!(err.to_string(), "Session missing does not exist");
    }
}
