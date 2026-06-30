// File: crates/ppf-cts-core/src/datamodel/scene.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `Scene`: mutable container for objects, invisible colliders, and
// stitch / merge / surface-map metadata. Holds the non-builder data
// layer; the heavier computational pieces live elsewhere:
//
//   * `Scene.build()` → `FixedScene`: vertex remap + connectivity
//     construction in `kernels::scene_build`.
//   * `Scene.time(time)` evaluator: pin operations + transform
//     animation; snapshot at `time`.
//   * Min/max bounds query: depends on Asset registry lookup.
//   * `select(name)`: trivial wrapper around `objects.get(name)`.
//
// Free-function helpers ported alongside `Scene`:
//
//   * Validators ported in earlier waves now live in
//     `datamodel::validators` (shared with `Object`, `Wall`, `Sphere`,
//     `ParamManager`, `SessionManager`).
//   * Mesh primitive generators (`line_mesh`, `box_mesh`,
//     `tet_box_mesh`, `cone_mesh`) live in `datamodel::mesh`.

use std::collections::BTreeMap;

use ndarray::Array2;

use super::asset::{AssetKind, AssetRegistry};
use super::collider::{Sphere, Wall};
use super::object::Object;

#[derive(Debug, Clone, Default)]
pub struct CrossStitch {
    pub source_uuid: String,
    pub target_uuid: String,
    /// Source vertex indices (shape varies by source/target type;
    /// kept as opaque 2D array).
    pub ind: Array2<i32>,
    pub w: Array2<f64>,
    /// SOLID-target only. Empty Vec for SHELL/ROD targets.
    pub target_points: Vec<[f64; 3]>,
    pub stitch_stiffness: f64,
}

#[derive(Debug, Clone)]
pub struct SurfaceMap {
    pub tri_indices: Vec<i32>,
    /// (N, 3) frame coefficients (c1, c2, c3) per original vertex.
    pub coefs: Array2<f64>,
    /// (Q, 3) surface triangles of the tet mesh.
    pub surf_tri: Array2<i32>,
}

#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    #[error("object {0} does not exist")]
    ObjectNotFound(String),
    #[error("object {0} already exists")]
    ObjectAlreadyExists(String),
    #[error("set_surface_map requires a non-empty object UUID key")]
    SurfaceMapEmptyKey,
}

#[derive(Debug, Clone)]
pub struct Scene {
    pub name: String,
    /// Insertion-ordered? Python uses dict (insertion order); BTreeMap
    /// here is alphabetical. Downstream FixedScene.build() walks them
    /// in some order; alphabetical is stable across runs and the
    /// natural choice for diff-friendly outputs.
    pub objects: BTreeMap<String, Object>,
    pub walls: Vec<Wall>,
    pub spheres: Vec<Sphere>,
    pub cross_stitch: Vec<CrossStitch>,
    /// Frame-embedding surface mappings keyed by object UUID.
    pub surface_map_by_name: BTreeMap<String, SurfaceMap>,
}

impl Scene {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            objects: BTreeMap::new(),
            walls: vec![],
            spheres: vec![],
            cross_stitch: vec![],
            surface_map_by_name: BTreeMap::new(),
        }
    }

    /// Wipe all objects but keep colliders, stitches, and surface
    /// maps.
    pub fn clear(&mut self) {
        self.objects.clear();
    }

    /// Add a fresh object referencing an asset of the given kind. The
    /// kind is passed explicitly by the caller; this method does not
    /// consult the asset registry. On a duplicate name it overwrites
    /// (last-write-wins, see the inline note); use `add_unique` for
    /// strict semantics or `add_from_registry` to derive the kind from
    /// the registry. Returns a mutable reference so the caller can chain
    /// transform builders.
    pub fn add(&mut self, asset_name: impl Into<String>, kind: AssetKind) -> &mut Object {
        let name: String = asset_name.into();
        let obj = Object::new(name.clone(), kind);
        // Last-write wins to mirror Python dict semantics. Python's
        // `Scene.add` chains call into `ObjectAdder.<kind>` which
        // raises if the name already exists; we mirror via
        // `add_unique` for callers that want strict semantics.
        self.objects.insert(name.clone(), obj);
        self.objects.get_mut(&name).unwrap()
    }

    /// Stricter sibling: error on duplicate names.
    pub fn add_unique(
        &mut self,
        asset_name: impl Into<String>,
        kind: AssetKind,
    ) -> Result<&mut Object, SceneError> {
        let name: String = asset_name.into();
        if self.objects.contains_key(&name) {
            return Err(SceneError::ObjectAlreadyExists(name));
        }
        let obj = Object::new(name.clone(), kind);
        self.objects.insert(name.clone(), obj);
        Ok(self.objects.get_mut(&name).unwrap())
    }

    /// Convenience: look up the asset's kind in the registry, then
    /// `add_unique`. Errors if the asset doesn't exist (so callers
    /// don't accidentally upload a placeholder kind).
    pub fn add_from_registry(
        &mut self,
        registry: &AssetRegistry,
        asset_name: impl Into<String>,
    ) -> Result<&mut Object, SceneError> {
        let name: String = asset_name.into();
        let kind = registry
            .get(&name)
            .ok_or_else(|| SceneError::ObjectNotFound(name.clone()))?
            .kind();
        self.add_unique(name, kind)
    }

    pub fn select(&self, name: &str) -> Result<&Object, SceneError> {
        self.objects
            .get(name)
            .ok_or_else(|| SceneError::ObjectNotFound(name.to_string()))
    }

    pub fn select_mut(&mut self, name: &str) -> Result<&mut Object, SceneError> {
        self.objects
            .get_mut(name)
            .ok_or_else(|| SceneError::ObjectNotFound(name.to_string()))
    }

    pub fn add_wall(&mut self, wall: Wall) {
        self.walls.push(wall);
    }

    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }

    /// Stamp a frame-embedding surface map.
    pub fn set_surface_map(
        &mut self,
        name: impl Into<String>,
        map: SurfaceMap,
    ) -> Result<(), SceneError> {
        let name: String = name.into();
        if name.is_empty() {
            return Err(SceneError::SurfaceMapEmptyKey);
        }
        self.surface_map_by_name.insert(name, map);
        Ok(())
    }

    pub fn object_names(&self) -> Vec<&str> {
        self.objects.keys().map(String::as_str).collect()
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
}

// ---------------------------------------------------------------------------
// Builder helpers used by the Python frontend's `Scene` and `Object`
// builder methods. Each function takes raw data and returns a
// validated outcome, so the Python wrappers can replace their loop
// bodies with a single Rust call.
//
// (The validators that started life inline here moved to
// `datamodel::validators`; keep the small math / formatting helpers
// next to the `Scene` type they orbit.)

/// Decode `"x"`/`"y"`/`"z"` (case-insensitive) into 0/1/2.
pub fn axis_letter_to_index(axis: &str) -> Result<usize, String> {
    match axis.to_ascii_lowercase().as_str() {
        "x" => Ok(0),
        "y" => Ok(1),
        "z" => Ok(2),
        other => Err(format!("invalid axis: {other}")),
    }
}

/// Reduce a per-object min/max stream into a scalar bound.
/// `kind = MinMax::Min` keeps the smaller of `(running, candidate)`;
/// `MinMax::Max` keeps the larger. Empty input returns the appropriate
/// infinity.
#[derive(Debug, Clone, Copy)]
pub enum MinMax {
    Min,
    Max,
}

pub fn reduce_axis_bound(stream: impl IntoIterator<Item = f64>, kind: MinMax) -> f64 {
    let init = match kind {
        MinMax::Min => f64::INFINITY,
        MinMax::Max => f64::NEG_INFINITY,
    };
    stream.into_iter().fold(init, |acc, v| match kind {
        MinMax::Min => acc.min(v),
        MinMax::Max => acc.max(v),
    })
}

/// Decide whether the dynamic-color string is supported (only `"area"`
/// is accepted today).
pub fn is_supported_dyn_color(name: &str) -> bool {
    matches!(name, "area")
}

/// Resolve a `move_by` keyframe arithmetic for a wall: take the
/// previous position, add the delta. Returns the new absolute position.
pub fn wall_move_by_position(prev_position: [f64; 3], delta: [f64; 3]) -> [f64; 3] {
    [
        prev_position[0] + delta[0],
        prev_position[1] + delta[1],
        prev_position[2] + delta[2],
    ]
}

/// Resolve a `move_by` keyframe for a sphere: position is `prev + delta`,
/// radius is reused from the previous entry.
pub fn sphere_move_by_entry(
    prev_position: [f64; 3],
    prev_radius: f64,
    delta: [f64; 3],
) -> ([f64; 3], f64) {
    (wall_move_by_position(prev_position, delta), prev_radius)
}

/// Format an integer with thousand separators (`12345 -> "12,345"`).
pub fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Build the structured count map that `FixedScene.report` displays.
/// Each entry is `(label, count)`; entries with `count == 0` are
/// omitted to match the Python behavior. Counts are returned as
/// thousand-separator strings (`"1,234"`).
///
/// Inputs are the various structural lengths the Python `report`
/// reads off of `FixedScene` instance state.
pub fn fixed_scene_report_entries(
    n_vert: usize,
    n_rod: usize,
    n_tri: usize,
    n_tet: usize,
    n_pin: usize,
    n_static_vert: usize,
    n_static_tri: usize,
    n_stitch_ind: usize,
    n_stitch_w: usize,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::with_capacity(7);
    // Always emit #vert (Python source emits unconditionally).
    out.push(("#vert".to_string(), fmt_thousands(n_vert)));
    if n_rod > 0 {
        out.push(("#rod".to_string(), fmt_thousands(n_rod)));
    }
    if n_tri > 0 {
        out.push(("#tri".to_string(), fmt_thousands(n_tri)));
    }
    if n_tet > 0 {
        out.push(("#tet".to_string(), fmt_thousands(n_tet)));
    }
    if n_pin > 0 {
        out.push(("#pin".to_string(), fmt_thousands(n_pin)));
    }
    if n_static_vert > 0 && n_static_tri > 0 {
        out.push(("#static_vert".to_string(), fmt_thousands(n_static_vert)));
        out.push(("#static_tri".to_string(), fmt_thousands(n_static_tri)));
    }
    if n_stitch_ind > 0 && n_stitch_w > 0 {
        out.push(("#stitch_ind".to_string(), fmt_thousands(n_stitch_ind)));
    }
    out
}

/// Velocity schedule arithmetic: choose between setting the
/// instantaneous initial velocity (when `t <= 0.0`) or appending a
/// timed override. Returns `(replace_initial, [u, v, w])` where the
/// caller updates `self._velocity` in-place when `replace_initial` is
/// true and otherwise appends `(t, [u, v, w])` to the schedule.
pub fn classify_velocity_entry(
    u: f64,
    v: f64,
    w: f64,
    t: f64,
) -> (bool, [f64; 3]) {
    let replace = t <= 0.0;
    (replace, [u, v, w])
}

/// Format the lines printed by `Object.report`. Returns the lines in
/// emit order, ready for callers to `print()` each. The `pin_status`
/// is `"static"` (when the object is static), the count of pinned
/// vertices as a string (`"42"`), or `"none"` when there are no pins.
pub fn object_report_lines(
    transform: &[[f64; 4]; 4],
    color_repr: &str,
    velocity_repr: &str,
    normalized: bool,
    pin_status: &str,
) -> Vec<String> {
    // Match numpy's default row-aligned scientific-ish formatter
    // closely enough for round-trip parity. The Python source `print`s
    // the matrix as numpy renders it; replicating numpy's full
    // formatter is not worth it. The Python wrapper still owns
    // `print()` for the matrix; this helper only emits the textual
    // labels so the non-matrix lines stay parity-checked here.
    //
    // Returned lines (in order):
    //   * "transform:" header
    //   * one line per matrix row (renderable as `[ a b c d ]`)
    //   * "color: <color_repr>"
    //   * "velocity: <velocity_repr>"
    //   * "normalize: <True|False>"
    //   * "pin: <pin_status>"
    let mut out: Vec<String> = Vec::with_capacity(8);
    out.push("transform:".to_string());
    for row in transform.iter() {
        out.push(format!(
            "[{:>10.4} {:>10.4} {:>10.4} {:>10.4}]",
            row[0], row[1], row[2], row[3]
        ));
    }
    out.push(format!("color: {color_repr}"));
    out.push(format!("velocity: {velocity_repr}"));
    out.push(format!("normalize: {}", if normalized { "True" } else { "False" }));
    out.push(format!("pin: {pin_status}"));
    out
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn axis_letter_decode() {
        assert_eq!(axis_letter_to_index("x").unwrap(), 0);
        assert_eq!(axis_letter_to_index("Y").unwrap(), 1);
        assert_eq!(axis_letter_to_index("z").unwrap(), 2);
        assert!(axis_letter_to_index("w").is_err());
    }

    #[test]
    fn reduce_axis_bound_min_max() {
        let xs = [3.0, 1.5, 4.2, -2.0];
        assert_eq!(reduce_axis_bound(xs, MinMax::Min), -2.0);
        assert_eq!(reduce_axis_bound(xs, MinMax::Max), 4.2);
        let empty: [f64; 0] = [];
        assert!(reduce_axis_bound(empty, MinMax::Min).is_infinite());
    }

    #[test]
    fn dyn_color_supported_set() {
        assert!(is_supported_dyn_color("area"));
        assert!(!is_supported_dyn_color("strain"));
    }

    #[test]
    fn wall_move_by_arith() {
        let p = wall_move_by_position([1.0, 2.0, 3.0], [0.5, -0.5, 0.0]);
        assert_eq!(p, [1.5, 1.5, 3.0]);
    }

    #[test]
    fn sphere_move_by_keeps_radius() {
        let (pos, r) = sphere_move_by_entry([0.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0]);
        assert_eq!(pos, [1.0, 0.0, 0.0]);
        assert_eq!(r, 0.5);
    }

    #[test]
    fn report_entries_omit_zero() {
        let r = fixed_scene_report_entries(1500, 0, 200, 0, 0, 0, 0, 0, 0);
        let map: std::collections::HashMap<_, _> = r.iter().cloned().collect();
        assert_eq!(map["#vert"], "1,500");
        assert_eq!(map["#tri"], "200");
        assert!(!map.contains_key("#rod"));
        assert!(!map.contains_key("#tet"));
        assert!(!map.contains_key("#pin"));
        assert!(!map.contains_key("#static_vert"));
        assert!(!map.contains_key("#stitch_ind"));
    }

    #[test]
    fn report_entries_with_static_and_stitch() {
        let r = fixed_scene_report_entries(10, 0, 0, 0, 0, 5, 4, 3, 3);
        let map: std::collections::HashMap<_, _> = r.iter().cloned().collect();
        assert_eq!(map["#vert"], "10");
        assert_eq!(map["#static_vert"], "5");
        assert_eq!(map["#static_tri"], "4");
        assert_eq!(map["#stitch_ind"], "3");
    }

    #[test]
    fn report_entries_thousands_format() {
        let r = fixed_scene_report_entries(1_234_567, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r[0].1, "1,234,567");
    }

    #[test]
    fn classify_velocity_entry_branches() {
        let (replace, vel) = classify_velocity_entry(0.0, -5.0, 0.0, 0.0);
        assert!(replace);
        assert_eq!(vel, [0.0, -5.0, 0.0]);

        let (replace, vel) = classify_velocity_entry(1.0, 2.0, 3.0, 0.5);
        assert!(!replace);
        assert_eq!(vel, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn object_report_lines_format() {
        let m = [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let lines = object_report_lines(&m, "None", "[0.0 0.0 0.0]", false, "static");
        assert_eq!(lines[0], "transform:");
        assert_eq!(lines.len(), 1 + 4 + 4); // header + 4 rows + color/velocity/normalize/pin
        assert_eq!(lines[5], "color: None");
        assert_eq!(lines[6], "velocity: [0.0 0.0 0.0]");
        assert_eq!(lines[7], "normalize: False");
        assert_eq!(lines[8], "pin: static");
    }
}

#[cfg(test)]
mod tests {
    use super::super::asset::AssetRegistry;
    use super::*;
    use ndarray::array;

    fn unit_tri_registry() -> AssetRegistry {
        let mut r = AssetRegistry::new();
        let v = array![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let f = array![[0u32, 1, 2]];
        r.add_tri("sheet", v, f, None).unwrap();
        r
    }

    #[test]
    fn new_scene_is_empty() {
        let s = Scene::new("demo");
        assert_eq!(s.name, "demo");
        assert_eq!(s.object_count(), 0);
        assert!(s.walls.is_empty());
        assert!(s.spheres.is_empty());
    }

    #[test]
    fn add_returns_chainable_object() {
        let mut s = Scene::new("demo");
        s.add("sheet", AssetKind::Tri).at(0.0, 0.6, 0.0);
        let obj = s.select("sheet").unwrap();
        assert_eq!(obj.position(), [0.0, 0.6, 0.0]);
    }

    #[test]
    fn add_unique_rejects_duplicate() {
        let mut s = Scene::new("demo");
        s.add_unique("sheet", AssetKind::Tri).unwrap();
        let err = s.add_unique("sheet", AssetKind::Tri).unwrap_err();
        assert!(matches!(err, SceneError::ObjectAlreadyExists(_)));
    }

    #[test]
    fn add_from_registry_picks_correct_kind() {
        let r = unit_tri_registry();
        let mut s = Scene::new("demo");
        let obj = s.add_from_registry(&r, "sheet").unwrap();
        assert_eq!(obj.asset_kind, AssetKind::Tri);
    }

    #[test]
    fn add_from_registry_rejects_missing_asset() {
        let r = AssetRegistry::new();
        let mut s = Scene::new("demo");
        let err = s.add_from_registry(&r, "missing").unwrap_err();
        assert!(matches!(err, SceneError::ObjectNotFound(_)));
    }

    #[test]
    fn clear_drops_objects_but_keeps_colliders() {
        let mut s = Scene::new("demo");
        s.add("sheet", AssetKind::Tri);
        let mut w = Wall::new();
        w.add([0.0; 3], [0.0, 1.0, 0.0]).unwrap();
        s.add_wall(w);

        s.clear();
        assert_eq!(s.object_count(), 0);
        assert_eq!(s.walls.len(), 1);
    }

    #[test]
    fn set_surface_map_rejects_empty_key() {
        let mut s = Scene::new("demo");
        let map = SurfaceMap {
            tri_indices: vec![0],
            coefs: array![[0.0, 0.0, 0.0]],
            surf_tri: array![[0i32, 1, 2]],
        };
        let err = s.set_surface_map("", map).unwrap_err();
        assert!(matches!(err, SceneError::SurfaceMapEmptyKey));
    }

    #[test]
    fn select_after_add_returns_same_object() {
        let mut s = Scene::new("demo");
        s.add("sheet", AssetKind::Tri).at(1.0, 2.0, 3.0);
        s.add("ball", AssetKind::Tet).at(5.0, 6.0, 7.0);

        assert_eq!(s.select("sheet").unwrap().position(), [1.0, 2.0, 3.0]);
        assert_eq!(s.select("ball").unwrap().position(), [5.0, 6.0, 7.0]);

        let names = s.object_names();
        // BTreeMap → alphabetical.
        assert_eq!(names, vec!["ball", "sheet"]);
    }

    #[test]
    fn add_wall_and_sphere_grow_collections() {
        let mut s = Scene::new("demo");
        let mut w = Wall::new();
        w.add([0.0; 3], [0.0, 1.0, 0.0]).unwrap();
        s.add_wall(w);

        let mut sp = Sphere::new();
        sp.add([0.0, 1.0, 0.0], 0.5).unwrap();
        s.add_sphere(sp);

        assert_eq!(s.walls.len(), 1);
        assert_eq!(s.spheres.len(), 1);
    }
}
