// File: crates/ppf-cts-core/src/datamodel/object.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `Object`: placed mesh instance. Mirrors the data layer of
// frontend/_scene_.py's `Object` class. Methods that depend on the
// asset registry (grab/stitch/normalize) or the PinHolder chain-API
// (pin) live alongside their owning types, not here.
//
// Carries:
//   * The asset reference (name + kind) so consumers know what
//     geometry to fetch when building a FixedScene.
//   * 4x4 world transform (T·R·S layout assembled by the builders).
//   * Per-object color spec (single RGB or per-vertex array).
//   * Initial velocity + keyframed schedule + collision-active windows.
//   * The list of pins applied to this object.
//   * Optional UV layers and a transform-animation keyframe stream.
//
// Builder methods are `&mut self -> &mut Self` rather than `self ->
// Self` so existing call sites that hold an `&mut Object` reference
// can still chain.

use ndarray::Array2;

use super::animation::TransformAnimation;
use super::asset::AssetKind;
use super::params::{object_param, ObjectKind, ParamHolder};
use super::pin::PinData;
use super::quat::{axis_angle_to_quat, quat_to_mat3, Vec3};

/// Maximum number of collision-active windows the solver can store per
/// dynamics group. This is the single source of truth for the cap.
///
/// MUST stay equal to the GPU-side `#define MAX_COLLISION_WINDOWS` in
/// `crates/ppf-cts-solver/src/cpp/main/main.cu`: the Rust side writes a
/// flat window table with stride `MAX_COLLISION_WINDOWS * 2` floats per
/// group and the kernel reads back with the same stride, so a mismatch
/// silently corrupts the simulation. The solver's collision-window table
/// builder (`scene.rs`) and the PyO3 binding (`ppf-cts-py`) reference this
/// constant; the frontend imports it through the PyO3 module rather than
/// re-declaring the literal.
pub const MAX_COLLISION_WINDOWS: usize = 8;

/// Per-object color spec: either uniform RGB or per-vertex.
/// Default is `None`; render uses the default tint.
#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub enum ObjectColor {
    #[default]
    None,
    Uniform([f64; 3]),
    PerVertex(Array2<f64>),
}


/// Dynamic color modes. Only the variants the addon-side decoder
/// constructs are listed; extend this enum if a new mode needs to
/// flow through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum DynamicColor {
    #[default]
    None,
    Strain,
}


#[derive(Debug, Clone)]
pub struct Object {
    /// Asset name in the registry (the key used at `scene.add(name)`).
    pub asset_name: String,
    /// Asset kind, copied from the registry at construction time so
    /// downstream code doesn't need to re-fetch it for a type check.
    pub asset_kind: AssetKind,
    /// True for static collider objects. Pinning every vertex flips
    /// this on (the Python source flips it inside `pin(None)`).
    pub is_static: bool,
    /// True when the object is an Painless Differentiable Rotation Dynamics (PDRD) body:
    /// its surface mesh moves rigidly via a single best-fit rigid
    /// transform shared by all vertices. Only valid for Tri assets;
    /// mutually exclusive with `is_static`.
    pub is_pdrd: bool,
    /// Per-object material parameters; defaulted from
    /// `object_param(kind)` at construction.
    pub param: ParamHolder,

    // --- Spatial transform ---
    /// 4x4 world transform, row-major. Identity at construction; the
    /// `at`/`translate`/`scale`/`rotate` builders mutate this.
    pub transform: [[f64; 4]; 4],

    // --- Visualization ---
    pub color: ObjectColor,
    pub dyn_color: DynamicColor,
    pub dyn_intensity: f64,
    pub static_color: [f64; 3],
    pub default_color: [f64; 3],

    // --- Dynamics ---
    /// Initial velocity (3D).
    pub velocity: Vec3,
    /// Keyframed velocity schedule: `(time_seconds, velocity)`.
    pub velocity_schedule: Vec<(f64, Vec3)>,
    /// Collision-active windows: `(t_start, t_end)`.
    pub collision_windows: Vec<(f64, f64)>,

    // --- Topology overlays ---
    pub pins: Vec<PinData>,
    /// Stitch asset name (if any).
    pub stitch: Option<String>,
    /// Per-face UV data (`tri` shells with UVs only).
    pub uv: Option<Array2<f64>>,
    /// Optional keyframe animation for STATIC objects.
    pub transform_animation: Option<TransformAnimation>,

    // --- Flags ---
    /// Hide this object's vertices from output dumps.
    pub exclude_from_output: bool,
    /// True when the object's pins drive a rigid-body motion of a
    /// static moving mesh; used to suppress pin markers in the
    /// preview viewer.
    pub is_static_moving: bool,
    /// Triggered when `normalize()` was applied; preserves the choice
    /// for downstream consumers that want to know.
    pub normalized: bool,
}

impl Object {
    /// Construct a fresh object referencing `asset_name` of kind
    /// `kind`.
    pub fn new(asset_name: impl Into<String>, kind: AssetKind) -> Self {
        let object_kind = match kind {
            AssetKind::Tri => Some(ObjectKind::Tri),
            AssetKind::Tet => Some(ObjectKind::Tet),
            AssetKind::Rod => Some(ObjectKind::Rod),
            AssetKind::Stitch => None,
        };
        let param = match object_kind {
            Some(k) => object_param(k),
            None => ParamHolder::default(),
        };
        Self {
            asset_name: asset_name.into(),
            asset_kind: kind,
            is_static: false,
            is_pdrd: false,
            param,
            transform: identity4(),
            color: ObjectColor::None,
            dyn_color: DynamicColor::None,
            dyn_intensity: 1.0,
            static_color: [0.75, 0.75, 0.75],
            default_color: [1.0, 0.85, 0.0],
            velocity: [0.0, 0.0, 0.0],
            velocity_schedule: vec![],
            collision_windows: vec![],
            pins: vec![],
            stitch: None,
            uv: None,
            transform_animation: None,
            exclude_from_output: false,
            is_static_moving: false,
            normalized: false,
        }
    }

    /// Switch this object to Painless Differentiable Rotation Dynamics. The body must
    /// reference a Tri asset (PDRD is implemented as an exact per-body
    /// rigid motion over a surface mesh; no tetrahedralization is
    /// needed or used). Swaps the parameter set to the PDRD defaults
    /// (volumetric `density`, contact/friction only; `young-mod`,
    /// `bend`, etc. are kept as zero placeholders so the existing
    /// per-face param expansion continues to work).
    ///
    /// Errors if the object is already flagged static, since PDRD and
    /// static colliders are mutually exclusive.
    pub fn as_pdrd(&mut self) -> Result<&mut Self, ObjectError> {
        if self.asset_kind != AssetKind::Tri {
            return Err(ObjectError::PdrdRequiresTriAsset);
        }
        if self.is_static {
            return Err(ObjectError::PdrdConflictsStatic);
        }
        self.is_pdrd = true;
        self.param = object_param(ObjectKind::Pdrd);
        Ok(self)
    }

    /// Reset every field except `asset_name`/`asset_kind`/`param`.
    pub fn clear(&mut self) {
        self.transform = identity4();
        self.color = ObjectColor::None;
        self.dyn_color = DynamicColor::None;
        self.dyn_intensity = 1.0;
        self.static_color = [0.75, 0.75, 0.75];
        self.default_color = [1.0, 0.85, 0.0];
        self.velocity = [0.0, 0.0, 0.0];
        self.velocity_schedule.clear();
        self.collision_windows.clear();
        self.pins.clear();
        self.stitch = None;
        self.uv = None;
        self.transform_animation = None;
        self.exclude_from_output = false;
        self.is_static_moving = false;
        self.normalized = false;
    }

    // -----------------------------------------------------------------
    // Transform builders. Each mutates `self.transform` and returns
    // `&mut self` so callers chain `.at(...).rotate(...).scale(...)`.

    /// Set the translation component to `(x, y, z)`. Replaces (does
    /// not add to) the prior translation.
    pub fn at(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.transform[0][3] = x;
        self.transform[1][3] = y;
        self.transform[2][3] = z;
        self
    }

    /// Add `(x, y, z)` onto the current translation. Convenience over
    /// `at`; not in the Python source but trivial to add.
    pub fn translate(&mut self, x: f64, y: f64, z: f64) -> &mut Self {
        self.transform[0][3] += x;
        self.transform[1][3] += y;
        self.transform[2][3] += z;
        self
    }

    /// Apply uniform scale `s`. Equivalent to
    /// `T = T @ diag(s, s, s, 1)`: multiplies the first three columns
    /// by `s`. Translation column is untouched (column 3).
    pub fn scale(&mut self, s: f64) -> &mut Self {
        for r in 0..4 {
            for c in 0..3 {
                self.transform[r][c] *= s;
            }
        }
        self
    }

    /// Apply rotation about a principal axis.
    ///
    ///   * Computes the axis-aligned rotation matrix `R`.
    ///   * Saves the current translation, applies `T = R @ T`, then
    ///     restores the translation. Net effect: rotation about the
    ///     origin, with the object's position pinned in place.
    ///
    /// `axis` accepts `'x' | 'y' | 'z'` (case-insensitive).
    pub fn rotate(&mut self, angle_deg: f64, axis: char) -> &mut Self {
        // Route the principal-axis rotation through the shared quat
        // helpers so the datamodel layer has one rotation-sign
        // convention. The quat helpers do not validate the axis, so the
        // invalid-axis panic is preserved here.
        let axis_vec: Vec3 = match axis.to_ascii_lowercase() {
            'x' => [1.0, 0.0, 0.0],
            'y' => [0.0, 1.0, 0.0],
            'z' => [0.0, 0.0, 1.0],
            other => panic!("invalid axis: {other:?} (expected 'x', 'y', or 'z')"),
        };
        let m = quat_to_mat3(axis_angle_to_quat(axis_vec, angle_deg));
        // Embed the 3x3 rotation into the top-left of an identity 4x4.
        let mut r = identity4();
        for row in 0..3 {
            for col in 0..3 {
                r[row][col] = m[row][col];
            }
        }
        let pos = [
            self.transform[0][3],
            self.transform[1][3],
            self.transform[2][3],
        ];
        let new = mat4_mul(&r, &self.transform);
        self.transform = new;
        // Restore translation post-rotation.
        self.transform[0][3] = pos[0];
        self.transform[1][3] = pos[1];
        self.transform[2][3] = pos[2];
        self
    }

    // -----------------------------------------------------------------
    // Color builders.

    /// Uniform RGB color.
    pub fn color(&mut self, red: f64, green: f64, blue: f64) -> &mut Self {
        self.color = ObjectColor::Uniform([red, green, blue]);
        self
    }

    /// Per-vertex RGB array, shape `(N, 3)`.
    pub fn vert_color(&mut self, color: Array2<f64>) -> &mut Self {
        debug_assert_eq!(color.ncols(), 3, "vert_color must be (N, 3)");
        self.color = ObjectColor::PerVertex(color);
        self
    }

    /// Set the default color, used when no explicit color is set.
    pub fn default_color(&mut self, red: f64, green: f64, blue: f64) -> &mut Self {
        self.default_color = [red, green, blue];
        self
    }

    /// Dynamic color mode + blend intensity.
    pub fn set_dyn_color(&mut self, kind: DynamicColor, intensity: f64) -> &mut Self {
        self.dyn_color = kind;
        self.dyn_intensity = intensity;
        self
    }

    // -----------------------------------------------------------------
    // Dynamics builders.

    /// Set initial velocity (when `t == 0.0`) or append a timed
    /// override (when `t > 0.0`). Errors if the object is flagged
    /// static.
    pub fn velocity(
        &mut self,
        u: f64,
        v: f64,
        w: f64,
        t: f64,
    ) -> Result<&mut Self, ObjectError> {
        if self.is_static {
            return Err(ObjectError::Static);
        }
        if t <= 0.0 {
            self.velocity = [u, v, w];
        } else {
            self.velocity_schedule.push((t, [u, v, w]));
        }
        Ok(self)
    }

    /// Replace the velocity schedule.
    pub fn set_velocity_schedule(&mut self, schedule: Vec<(f64, Vec3)>) -> &mut Self {
        self.velocity_schedule = schedule;
        self
    }

    /// Replace the collision-window list.
    pub fn set_collision_windows(&mut self, windows: Vec<(f64, f64)>) -> &mut Self {
        self.collision_windows = windows;
        self
    }

    /// Mark the object excluded from output dumps.
    pub fn exclude_from_output(&mut self, excluded: bool) -> &mut Self {
        self.exclude_from_output = excluded;
        self
    }

    /// World-space translation read from the transform's right column.
    pub fn position(&self) -> Vec3 {
        [
            self.transform[0][3],
            self.transform[1][3],
            self.transform[2][3],
        ]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ObjectError {
    #[error("object is static")]
    Static,
    #[error("PDRD requires a Tri (surface mesh) asset")]
    PdrdRequiresTriAsset,
    #[error("PDRD cannot be applied to a static object")]
    PdrdConflictsStatic,
}

// ---------------------------------------------------------------------------
// 4x4 helpers.

#[inline]
pub(super) fn identity4() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[inline]
fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[r][k] * b[k][c];
            }
            out[r][c] = s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn new_object_has_identity_transform() {
        let o = Object::new("sheet", AssetKind::Tri);
        assert_eq!(o.transform, identity4());
        assert_eq!(o.position(), [0.0, 0.0, 0.0]);
        assert!(!o.is_static);
        assert_eq!(o.asset_kind, AssetKind::Tri);
    }

    #[test]
    fn at_replaces_translation() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.translate(5.0, 0.0, 0.0).at(1.0, 2.0, 3.0);
        assert_eq!(o.position(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn translate_accumulates() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.translate(1.0, 0.0, 0.0).translate(2.0, 5.0, 0.0);
        assert_eq!(o.position(), [3.0, 5.0, 0.0]);
    }

    #[test]
    fn scale_applies_to_columns_0_to_2_only() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.translate(7.0, 0.0, 0.0); // translation column is index 3
        o.scale(2.0);
        // diag becomes 2,2,2,1; translation untouched per `T = T @ S`.
        assert!(approx(o.transform[0][0], 2.0, 1e-15));
        assert!(approx(o.transform[1][1], 2.0, 1e-15));
        assert!(approx(o.transform[2][2], 2.0, 1e-15));
        assert!(approx(o.transform[3][3], 1.0, 1e-15));
        assert_eq!(o.position(), [7.0, 0.0, 0.0]);
    }

    #[test]
    fn rotate_z90_swaps_x_into_y_in_first_column() {
        // After 90° about z, column 0 of T becomes (0, 1, 0, 0) (the
        // original x-axis rotated to +y).
        let mut o = Object::new("x", AssetKind::Tri);
        o.rotate(90.0, 'z');
        assert!(approx(o.transform[0][0], 0.0, 1e-12));
        assert!(approx(o.transform[1][0], 1.0, 1e-12));
        assert!(approx(o.transform[2][0], 0.0, 1e-12));
    }

    #[test]
    fn rotate_preserves_position() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.at(1.0, 2.0, 3.0).rotate(45.0, 'y').rotate(30.0, 'x');
        // Position must survive both rotations untouched.
        assert_eq!(o.position(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn rotate_invalid_axis_panics() {
        let mut o = Object::new("x", AssetKind::Tri);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            o.rotate(10.0, 'q');
        }));
        assert!(result.is_err());
    }

    #[test]
    fn clear_resets_transient_fields() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.at(1.0, 2.0, 3.0);
        o.scale(2.0);
        o.color = ObjectColor::Uniform([1.0, 0.0, 0.0]);
        o.velocity = [10.0, 0.0, 0.0];
        o.exclude_from_output = true;

        o.clear();
        assert_eq!(o.transform, identity4());
        assert_eq!(o.color, ObjectColor::None);
        assert_eq!(o.velocity, [0.0, 0.0, 0.0]);
        assert!(!o.exclude_from_output);
    }

    #[test]
    fn rod_object_uses_rod_param_defaults() {
        let o = Object::new("rope", AssetKind::Rod);
        // Rod material defaults: model="arap", density=1.0, young-mod=1e4.
        assert_eq!(
            o.param.get("model").unwrap(),
            &super::super::params::ParamValue::String("arap".into())
        );
        assert_eq!(
            o.param.get("young-mod").unwrap(),
            &super::super::params::ParamValue::Float(1e4)
        );
    }

    #[test]
    fn as_pdrd_switches_kind_and_params_on_tri_asset() {
        let mut o = Object::new("cube", AssetKind::Tri);
        assert!(!o.is_pdrd);
        o.as_pdrd().unwrap();
        assert!(o.is_pdrd);
        // Now the param holder is the PDRD set: model="pdrd", young-mod zeroed.
        assert_eq!(
            o.param.get("model").unwrap(),
            &super::super::params::ParamValue::String("pdrd".into())
        );
        assert_eq!(
            o.param.get("young-mod").unwrap(),
            &super::super::params::ParamValue::Float(0.0)
        );
    }

    #[test]
    fn as_pdrd_rejects_non_tri_assets() {
        let mut o = Object::new("brick", AssetKind::Tet);
        let err = o.as_pdrd().unwrap_err();
        assert!(matches!(err, ObjectError::PdrdRequiresTriAsset));
        // Object is unchanged.
        assert!(!o.is_pdrd);

        let mut o = Object::new("rope", AssetKind::Rod);
        let err = o.as_pdrd().unwrap_err();
        assert!(matches!(err, ObjectError::PdrdRequiresTriAsset));
        assert!(!o.is_pdrd);
    }

    #[test]
    fn as_pdrd_rejects_static_object() {
        let mut o = Object::new("wall", AssetKind::Tri);
        o.is_static = true;
        let err = o.as_pdrd().unwrap_err();
        assert!(matches!(err, ObjectError::PdrdConflictsStatic));
        assert!(!o.is_pdrd);
    }

    #[test]
    fn color_builders_match_python() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.color(0.5, 0.6, 0.7);
        assert_eq!(o.color, ObjectColor::Uniform([0.5, 0.6, 0.7]));

        let arr = ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        o.vert_color(arr.clone());
        assert_eq!(o.color, ObjectColor::PerVertex(arr));

        o.default_color(0.1, 0.2, 0.3);
        assert_eq!(o.default_color, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn velocity_t0_sets_initial() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.velocity(1.0, 2.0, 3.0, 0.0).unwrap();
        assert_eq!(o.velocity, [1.0, 2.0, 3.0]);
        assert!(o.velocity_schedule.is_empty());
    }

    #[test]
    fn velocity_t_positive_appends_schedule() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.velocity(0.0, -5.0, 0.0, 0.0).unwrap();
        o.velocity(0.0, 0.0, 0.0, 1.0).unwrap();
        assert_eq!(o.velocity, [0.0, -5.0, 0.0]);
        assert_eq!(o.velocity_schedule, vec![(1.0, [0.0, 0.0, 0.0])]);
    }

    #[test]
    fn velocity_on_static_errors() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.is_static = true;
        let err = o.velocity(1.0, 0.0, 0.0, 0.0).unwrap_err();
        assert!(matches!(err, ObjectError::Static));
    }

    #[test]
    fn collision_windows_replaces() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.set_collision_windows(vec![(0.2, 1.0), (2.0, 3.0)]);
        assert_eq!(o.collision_windows, vec![(0.2, 1.0), (2.0, 3.0)]);
    }

    #[test]
    fn dyn_color_builder_sets_kind_and_intensity() {
        let mut o = Object::new("x", AssetKind::Tri);
        o.set_dyn_color(DynamicColor::Strain, 0.75);
        assert_eq!(o.dyn_color, DynamicColor::Strain);
        assert_eq!(o.dyn_intensity, 0.75);
    }

    #[test]
    fn chain_compiles_in_typical_order() {
        // The classic notebook usage: scene.add("sheet").at(...).rotate(...)
        let mut o = Object::new("sheet", AssetKind::Tri);
        o.at(0.0, 0.6, 0.0).rotate(90.0, 'x').scale(0.5);
        // Position pinned at (0, 0.6, 0); first three columns are
        // rotated+scaled.
        assert_eq!(o.position(), [0.0, 0.6, 0.0]);
    }
}
