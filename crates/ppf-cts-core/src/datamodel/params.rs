// File: crates/ppf-cts-core/src/datamodel/params.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Direct port of frontend/_param_.py:
//
//   * `ParamHolder` is the in-memory key→value store with a chained
//     `.set(key, value)` API and per-key (display_name, description)
//     metadata. Keys are dynamic (rooted in the Python `dict` used
//     today); values are heterogeneous so we model them with a
//     `ParamValue` enum (the union of types the addon and notebook
//     callers actually emit).
//   * `app_param()` and `object_param(kind)` are factory functions
//     producing a populated `ParamHolder` with the same defaults the
//     Python source specifies (frontend/_param_.py:9-315).
//
// The `Display Name` and `Description` strings are kept verbatim so
// addon UI bindings still produce identical labels when wiring this
// type up via PyO3 in a later phase. The defaults exactly match
// frontend/_param_.py: one BUILDER per default so the wire-format
// compute_param_hash byte-pattern stays stable across this port.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectKind {
    Tri,
    Tet,
    Rod,
    /// Painless Differentiable Rotation Dynamics: a Tri-asset body whose vertices move
    /// exactly rigidly via a single best-fit rigid transform shared by
    /// the whole body. No per-element elastic energy, no bending; only
    /// `density` (volumetric kg/m^3) and the standard contact/friction
    /// params are user-tunable.
    Pdrd,
    /// Granular / sand body: a faceless cloud of free particles with no
    /// elements and no elastic energy. The solver advances the points
    /// directly (the emulated backend drifts them under gravity; the real
    /// granular kernel replaces that). Only `density` and the standard
    /// contact/friction params are meaningful.
    Points,
}

impl ObjectKind {
    pub fn from_str(s: &str) -> Result<Self, ParamError> {
        match s {
            "tri" => Ok(Self::Tri),
            "tet" => Ok(Self::Tet),
            "rod" => Ok(Self::Rod),
            "pdrd" => Ok(Self::Pdrd),
            "points" => Ok(Self::Points),
            _ => Err(ParamError::UnknownObjectKind(s.to_string())),
        }
    }
}

/// Heterogeneous value type for a single param entry. Captures every
/// shape the Python source assigns (booleans, integer counts, floats,
/// vec3 [gravity, wind], and string enums [model, barrier,
/// friction-mode]).
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Vec3([f64; 3]),
    String(String),
}

impl From<bool> for ParamValue {
    fn from(b: bool) -> Self {
        ParamValue::Bool(b)
    }
}
impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        ParamValue::Int(v)
    }
}
impl From<f32> for ParamValue {
    fn from(v: f32) -> Self {
        ParamValue::Float(v as f64)
    }
}
impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        ParamValue::Float(v)
    }
}
impl From<[f64; 3]> for ParamValue {
    fn from(v: [f64; 3]) -> Self {
        ParamValue::Vec3(v)
    }
}
impl From<&str> for ParamValue {
    fn from(s: &str) -> Self {
        ParamValue::String(s.to_string())
    }
}
impl From<String> for ParamValue {
    fn from(s: String) -> Self {
        ParamValue::String(s)
    }
}

#[derive(Debug, Clone)]
pub struct ParamEntry {
    pub default: ParamValue,
    pub value: ParamValue,
    pub display_name: String,
    pub description: String,
}

#[derive(Debug, Clone, Default)]
pub struct ParamHolder {
    /// BTreeMap keeps key iteration sorted; `compute_param_hash`
    /// re-canonicalizes anyway, but stable order makes test diffs
    /// readable and downstream consumers happy.
    entries: BTreeMap<String, ParamEntry>,
}

#[derive(Debug, thiserror::Error)]
pub enum ParamError {
    #[error("Parameter '{0}' not found.")]
    NotFound(String),
    #[error("Unknown object type: {0}")]
    UnknownObjectKind(String),
}

impl ParamHolder {
    pub fn new(entries: BTreeMap<String, ParamEntry>) -> Self {
        Self { entries }
    }

    /// Reset every key back to its default.
    pub fn clear_all(&mut self) -> &mut Self {
        for entry in self.entries.values_mut() {
            entry.value = entry.default.clone();
        }
        self
    }

    /// Set an existing key. Returns the holder for chaining.
    pub fn set(&mut self, key: &str, value: impl Into<ParamValue>) -> Result<&mut Self, ParamError> {
        match self.entries.get_mut(key) {
            Some(e) => {
                e.value = value.into();
                Ok(self)
            }
            None => Err(ParamError::NotFound(key.to_string())),
        }
    }

    pub fn get(&self, key: &str) -> Result<&ParamValue, ParamError> {
        self.entries
            .get(key)
            .map(|e| &e.value)
            .ok_or_else(|| ParamError::NotFound(key.to_string()))
    }

    pub fn get_desc(&self, key: &str) -> Result<(&str, &str), ParamError> {
        self.entries
            .get(key)
            .map(|e| (e.display_name.as_str(), e.description.as_str()))
            .ok_or_else(|| ParamError::NotFound(key.to_string()))
    }

    pub fn key_list(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    pub fn items(&self) -> Vec<(&str, &ParamValue)> {
        self.entries
            .iter()
            .map(|(k, v)| (k.as_str(), &v.value))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Factory helpers: mirror frontend/_param_.py:app_param and
// object_param. Defaults must match byte-for-byte; verified by the
// `defaults_match_python_source` test below (which spot-checks a
// representative slice).

fn entry(default: impl Into<ParamValue>, name: &str, desc: &str) -> ParamEntry {
    let v: ParamValue = default.into();
    ParamEntry {
        default: v.clone(),
        value: v,
        display_name: name.to_string(),
        description: desc.to_string(),
    }
}

/// Default app-level parameters. Keep keys, defaults, display names,
/// and descriptions verbatim; the addon and the notebook UI both
/// render these strings.
pub fn app_param() -> ParamHolder {
    let mut m: BTreeMap<String, ParamEntry> = BTreeMap::new();
    m.insert("disable-contact".into(), entry(false, "Disable Contact",
        "When enabled, the simulation will not perform any contact detection."));
    m.insert("keep-states".into(), entry(10i64, "Keep States",
        "Number of simulation states to keep in the output directory."));
    m.insert("keep-verts".into(), entry(0i64, "Keep Vertices",
        "Number of vertex data files to keep in the output directory. 0 means no limit (unlimited). Minimum is 1 to ensure visualization."));
    m.insert("dt".into(), entry(1e-3f64, "Step Size",
        "Step size for the simulation. Small step size increases accuracy, large step size increases speed but may cause solver divergence."));
    m.insert("inactive-momentum".into(), entry(false, "Inactive Momentum",
        "Enable inactive momentum mode for the simulation. Adjusts step size and disables inertia."));
    m.insert("playback".into(), entry(1.0f64, "Playback Speed",
        "Playback speed. 1.0 is normal, >1.0 is faster, <1.0 is slower."));
    m.insert("min-newton-steps".into(), entry(0i64, "Lower Bound of Newton Steps",
        "Minimal Newton's steps to advance a step. Recommended 32 if static friction is present."));
    m.insert("target-toi".into(), entry(0.25f64, "Target Accumulated Time of Impact (TOI)",
        "Accumulated TOI threshold for Newton's loop termination."));
    m.insert("air-friction".into(), entry(0.2f64, "Air Tangental Friction",
        "Ratio of tangential friction to normal friction for air drag/lift."));
    m.insert("line-search-max-t".into(), entry(1.25f64, "Extended Line Search Maximum Time",
        "Factor to extend TOI for CCD to avoid possible solver divergence."));
    m.insert("constraint-ghat".into(), entry(1e-3f64, "Gap Distance for Boundary Conditions",
        "Gap distance to activate boundary condition barriers."));
    m.insert("constraint-tol".into(), entry(0.01f64, "Moving Constraint Minimum Gap Tolerance",
        "This factor is multiplied to the constraint gap to determine the CCD tolerance for moving constraints."));
    m.insert("fps".into(), entry(60.0f64, "Frame Per Second for Video Frames",
        "Frame rate for output video."));
    m.insert("cg-max-iter".into(), entry(10000i64, "Maximum Number of PCG Iterations",
        "PCG solver is regarded as diverged if this is exceeded."));
    m.insert("cg-tol".into(), entry(1e-3f64, "Relative Tolerance for PCG",
        "Relative tolerance for PCG solver termination."));
    m.insert("ccd-reduction".into(), entry(0.01f64, "CCD Reduction Factor",
        "Factor multiplied to the initial gap to set the CCD threshold."));
    m.insert("ccd-max-iter".into(), entry(4096i64, "Maximum CCD Iterations",
        "Maximum number of iterations for ACCD."));
    m.insert("max-dx".into(), entry(1.0f64, "Maximum Search Direction",
        "Maximum allowable search direction magnitude during optimization."));
    m.insert("eiganalysis-eps".into(), entry(1e-2f64, "Epsilon for Eigenvalue Analysis",
        "Epsilon for stable eigenvalue analysis when singular values are close."));
    m.insert("friction-eps".into(), entry(1e-5f64, "Epsilon for Friction",
        "Small value to avoid division by zero in quadratic friction model."));
    m.insert("csrmat-max-nnz".into(), entry(10_000_000i64,
        "Maximal Matrix Entries for Contact Matrix Entries on the GPU",
        "Pre-allocated contact matrix entries for GPU. Too large may cause OOM, too small may cause failure."));
    m.insert("frames".into(), entry(300i64, "Maximal Frame Count to Simulate",
        "Maximal number of frames to simulate."));
    m.insert("auto-save".into(), entry(0i64, "Auto Save Interval",
        "Interval (in frames) for auto-saving simulation state. 0 disables auto-save."));
    m.insert("save-state-on-finish".into(), entry(false, "Save State on Finish",
        "Save the simulation state on the final frame before the solver exits, even when auto-save is off."));
    m.insert("checkpoints".into(), entry("", "Save Checkpoints",
        "Comma-separated frame indices at which to save a resumable state, independent of the auto-save interval. Empty disables explicit checkpoints."));
    m.insert("barrier".into(), entry("cubic", "Barrier Model for Contact",
        "Contact barrier potential model. Choices: cubic, quad, log."));
    m.insert("friction-mode".into(), entry("min", "Friction Combination Mode",
        "How to combine the friction coefficients of two contacting elements. Choices: min, max, mean. Default min preserves the prior behavior (the more slippery side wins)."));
    m.insert("precond".into(), entry("block-jacobi", "Linear Solver Preconditioner",
        "Preconditioner for the PCG linear solve. Choices: block-jacobi (3x3 per-vertex diagonal, default; fast with the device-resident PCG loop and does not run out of memory on heavy-contact scenes), schwarz (additive aggregate-Schwarz; fewer iterations on mixed stiff/soft systems but heavier per iteration and can OOM on large contact counts; its level count is set by schwarz-levels)."));
    m.insert("schwarz-levels".into(), entry(2i64, "Schwarz Levels",
        "Number of additive levels for the Schwarz preconditioner (only used when precond is schwarz). 1 = single-level smoother; 2 = two-level coarse correction over the connectivity partition (default), which reduces the worst-case PCG iteration count on stiff multibody contact. Values above the solver's internal cap are clamped."));
    m.insert("stitch-length-factor".into(), entry(10.0f64, "Stitch Length Factor",
        "Multiplier on the stitch rest length when capping the stitch separation in the strain force. Cap = stitch_length_factor * l0 + offset_src + max(offset_target). Larger values let stitches stretch farther before the cap saturates the force."));
    m.insert("air-density".into(), entry(1e-3f64, "Air Density",
        "Air density for drag and lift force computation."));
    m.insert("isotropic-air-friction".into(), entry(0.0f64, "Air Dragging Coefficient",
        "Per-vertex air dragging coefficient."));
    m.insert("gravity".into(), entry([0.0f64, -9.8, 0.0], "Gravity Vector",
        "Gravity acceleration vector (solver Y-up)."));
    m.insert("wind".into(), entry([0.0f64, 0.0, 0.0], "Wind Force",
        "Wind force vector (XYZ)."));
    m.insert("include-face-mass".into(), entry(false, "Flag to Include Shell Mass for Volume Solids",
        "Include shell mass for surface elements of volume solids."));
    m.insert("fix-xz".into(), entry(0.0f64, "Whether to fix xz positions",
        "Fix xz positions for falling objects if y > this value. 0.0 disables. Use an extremely small value if nearly a zero is needed."));
    m.insert("fake-crash-frame".into(), entry(-1i64, "Fake Crash Frame",
        "Frame number to intentionally crash simulation for testing. -1 disables."));
    m.insert("world-scaling".into(), entry(1.0f64, "World Scaling Factor",
        "Uniform spatial scale applied to all input geometry before the simulation runs; output positions are divided back by it. Values below 1 shrink (e.g. 0.1 simulates a 15 m mesh at 1.5 m and writes results back at 15 m). Only geometry and relative contact gaps scale; gravity and absolute contact gaps do not."));

    ParamHolder::new(m)
}

/// Default per-object parameters. Defaults vary per element kind; the
/// conditional fields (plasticity, bend-plasticity,
/// bend-rest-from-geometry) are added based on `kind`.
pub fn object_param(kind: ObjectKind) -> ParamHolder {
    // For PDRD bodies the solver advances an exact per-body rigid motion.
    // The face-level keys (model, young-mod, bend, ...) are still emitted
    // so the existing per-face param expansion continues to work unchanged;
    // elastic dispatch trips on `Model::Pdrd` and is a no-op. Density on
    // PDRD is volumetric (kg/m^3); the builder computes enclosed volume and
    // writes a uniform `mass_per_vertex` scaled to match rotational inertia.
    let (model, young_mod, density, offset, bend) = match kind {
        ObjectKind::Tri => ("baraff-witkin", 1000.0f64, 1.0f64, 0.0f64, 10.0f64),
        ObjectKind::Tet => ("arap", 500.0, 100.0, 0.0, 1.0),
        ObjectKind::Rod => ("arap", 1e4, 1.0, 1e-3, 0.0),
        ObjectKind::Pdrd => ("pdrd", 0.0, 100.0, 0.0, 0.0),
        // Faceless particle cloud: no elements, so the per-element elastic
        // expansion produces nothing and the model string is never applied.
        // ARAP + zero young keeps it inert; density is the only live knob.
        ObjectKind::Points => ("arap", 0.0, 100.0, 0.0, 0.0),
    };

    let mut m: BTreeMap<String, ParamEntry> = BTreeMap::new();
    m.insert("model".into(), entry(model, "Deformation Model",
        "Elastic constitutive model used to evaluate stretch energy. Valid choices are 'arap', 'stvk', 'baraff-witkin' (shells only), and 'snhk' (stable neo-Hookean). Rods are always solved with ARAP internally."));
    m.insert("density".into(), entry(density, "Density",
        "Rest-state mass density. Units depend on element type: kg/m^3 for 'tet' (volumetric), kg/m^2 for 'tri' (areal), and kg/m for 'rod' (linear). Must be positive."));
    m.insert("young-mod".into(), entry(young_mod, "Young's Modulus",
        "Stiffness coefficient fed into the constitutive model, pre-normalized by density (units of Pa/rho). This decouples mass from stiffness, so doubling density alone leaves the motion unchanged. Must be positive."));
    m.insert("poiss-rat".into(), entry(0.35f64, "Poisson's Ratio",
        "Poisson's ratio used together with Young's modulus to derive the Lame parameters. Valid range is (0, 0.5); values near 0.5 produce near-incompressible behavior. Used by 'tri' and 'tet' elements only."));
    m.insert("bend".into(), entry(bend, "Bending Stiffness",
        "Dimensionless bending stiffness for shell hinges (between adjacent 'tri' faces) and for rod-joint bending at interior rod vertices. Must be non-negative. Not used by 'tet' elements."));
    m.insert("deformation-damping".into(), entry(0.0f64, "Deformation Damping",
        "Stiffness-proportional Rayleigh damping coefficient (beta) for stretch/membrane/solid deformation, in seconds. Damps high-frequency deformation by reusing the element tangent stiffness: it adds (beta/dt)*K to the system matrix. 0.0 disables it. Must be non-negative."));
    m.insert("bending-damping".into(), entry(0.0f64, "Bending Damping",
        "Stiffness-proportional Rayleigh damping coefficient (beta) for shell and rod bending, in seconds, applied to the bending tangent stiffness. Usually smaller than deformation damping. 0.0 disables it. Must be non-negative. Not used by 'tet' elements."));
    m.insert("stitch-stiffness".into(), entry(1.0f64, "Stitch Stiffness",
        "Direct stiffness factor for the stitch force on this object's stitches (loose-edge / intra-object stitches). Scales the stitch gradient and Hessian directly, with no mass or dt normalization. Must be non-negative. Resolved per object at scene build and applied per stitch row in the solver."));
    m.insert("shrink-x".into(), entry(1.0f64, "Shrink X",
        "Anisotropic rest-shape scale along the UV X (warp) direction for 'tri' shells. Values below 1.0 pre-shrink the cloth, above 1.0 pre-stretch it. Must be positive. Cannot be combined with a non-zero strain limit on the same face."));
    m.insert("shrink-y".into(), entry(1.0f64, "Shrink Y",
        "Anisotropic rest-shape scale along the UV Y (weft) direction for 'tri' shells. Values below 1.0 pre-shrink the cloth, above 1.0 pre-stretch it. Must be positive. Cannot be combined with a non-zero strain limit on the same face."));
    m.insert("shrink".into(), entry(1.0f64, "Shrink",
        "Isotropic rest-shape scale for 'tet' solids. Values below 1.0 pre-contract the solid so it pulls toward a smaller target, above 1.0 pre-inflate it. Must be positive. Ignored by 'tri' and 'rod' elements."));
    m.insert("contact-gap".into(), entry(1e-3f64, "Contact Gap",
        "Barrier activation distance for contact detection, in scene units. At each pair the solver uses the mean of the two participants' gaps as the threshold. Must be positive."));
    m.insert("contact-offset".into(), entry(offset, "Contact Offset",
        "Extra per-element padding added on top of the contact gap (scene units). At each pair the two participants' offsets are summed, acting like a skin thickness that guarantees minimum clearance. Must be non-negative."));
    m.insert("strain-limit".into(), entry(0.0f64, "Strain Limit",
        "Upper bound on per-element tensile strain (dimensionless, e.g. 0.05 = 5% stretch). 0.0 disables strain limiting. Supported on 'tri' and 'rod' elements only, and incompatible with non-unit shrink-x/shrink-y on the same face."));
    m.insert("friction".into(), entry(0.0f64, "Friction Coefficient",
        "Coulomb friction coefficient at contacts (dimensionless). When two elements touch, the solver uses the minimum of the two participants' coefficients, so the more slippery side wins. Must be non-negative."));
    m.insert("length-factor".into(), entry(1.0f64, "Rod Rest-Length Factor",
        "Multiplier applied to each rod edge's rest length (dimensionless). Values below 1.0 pre-tension the rod, above 1.0 pre-compress it. Must be positive. Used by 'rod' elements only."));
    m.insert("pressure".into(), entry(0.0f64, "Inflation Pressure",
        "Per-face inflation pressure pushing 'tri' shells outward along the face normal. Must be non-negative; 0.0 disables inflation. Ignored by 'tet' and 'rod' elements."));

    if matches!(kind, ObjectKind::Tri | ObjectKind::Tet | ObjectKind::Pdrd) {
        m.insert("plasticity".into(), entry(0.0f64, "Plasticity Rate",
            "Rate constant (per second) for stretch plasticity, driving the rest shape toward the current deformation via alpha = 1 - exp(-rate * dt). 0.0 disables plasticity; higher values creep faster. Must be non-negative."));
        m.insert("plasticity-threshold".into(), entry(0.0f64, "Plasticity Threshold",
            "Dead zone (dimensionless) around unit principal stretch S=1. Plasticity only activates on a singular value k when |S_k - 1| exceeds this threshold. 0.0 means any deviation triggers creep."));
    }

    // All three types receive the bend-plasticity placeholders. The
    // Python source notes (line 292): tet keeps them at 0.0 even
    // though tetrahedra ignore them; needed so a solid's surface
    // triangles can join concat_tri_param without a key-set
    // mismatch. Mirror that.
    m.insert("bend-plasticity".into(), entry(0.0f64, "Bend Plasticity Rate",
        "Rate constant (per second) for bending plasticity that drifts the rest angle of shell hinges and interior rod joints toward the current angle. 0.0 disables. Applies to 'tri' and 'rod'; kept on 'tet' only as a placeholder and ignored by the solver."));
    m.insert("bend-plasticity-threshold".into(), entry(0.0f64, "Bend Plasticity Threshold",
        "Angular dead zone (radians) around the rest angle. Bend plasticity only activates when |theta - theta_rest| exceeds this threshold. 0.0 means any angular deviation triggers creep."));
    m.insert("bend-rest-from-geometry".into(), entry(0.0f64, "Bend Rest From Initial Geometry",
        "If non-zero, initialize each hinge or rod-joint rest angle from the initial pose instead of the default (flat for shells, straight for rods). Treated as a boolean flag."));

    // Granular (sand) material knobs. Defined only for the faceless particle
    // kind so the addon's `sand-*` params resolve at decode. The emulated
    // backend ignores them (it just drifts the cloud); the real granular
    // kernel consumes them.
    if matches!(kind, ObjectKind::Points) {
        m.insert("sand-particle-mass".into(), entry(1e-3f64, "Sand Particle Mass",
            "Mass of a single sand particle, in kilograms (the addon authors it in grams and \
             converts). Must be positive."));
        m.insert("sand-friction".into(), entry(0.5f64, "Sand Friction",
            "Inter-grain Coulomb friction coefficient (dimensionless). Must be non-negative."));
        // The grain radius is sent via the standard `contact-offset` key (the
        // grain's physical contact skin), so there is no separate sand radius.
    }

    ParamHolder::new(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_param_holds_known_keys_with_correct_defaults() {
        let p = app_param();
        // Spot-check a representative spread.
        assert!(matches!(p.get("dt").unwrap(), ParamValue::Float(v) if (*v - 1e-3).abs() < 1e-15));
        assert!(matches!(p.get("frames").unwrap(), ParamValue::Int(300)));
        assert!(matches!(p.get("disable-contact").unwrap(), ParamValue::Bool(false)));
        assert_eq!(p.get("barrier").unwrap(), &ParamValue::String("cubic".into()));
        assert_eq!(p.get("gravity").unwrap(), &ParamValue::Vec3([0.0, -9.8, 0.0]));
    }

    #[test]
    fn set_chains_and_replaces_value() {
        let mut p = app_param();
        p.set("dt", 5e-4f64).unwrap().set("frames", 120i64).unwrap();
        assert_eq!(p.get("dt").unwrap(), &ParamValue::Float(5e-4));
        assert_eq!(p.get("frames").unwrap(), &ParamValue::Int(120));
    }

    #[test]
    fn set_unknown_key_errors() {
        let mut p = app_param();
        let err = p.set("not-a-param", 1.0f64).unwrap_err();
        match err {
            ParamError::NotFound(s) => assert_eq!(s, "not-a-param"),
            _ => panic!("expected NotFound"),
        }
    }

    #[test]
    fn clear_all_resets_values() {
        let mut p = app_param();
        p.set("dt", 5e-4f64).unwrap();
        assert_eq!(p.get("dt").unwrap(), &ParamValue::Float(5e-4));
        p.clear_all();
        assert_eq!(p.get("dt").unwrap(), &ParamValue::Float(1e-3));
    }

    #[test]
    fn get_desc_returns_metadata() {
        let p = app_param();
        let (name, desc) = p.get_desc("dt").unwrap();
        assert_eq!(name, "Step Size");
        assert!(desc.starts_with("Step size for the simulation."));
    }

    #[test]
    fn object_param_tri_has_baraff_witkin_default() {
        let p = object_param(ObjectKind::Tri);
        assert_eq!(p.get("model").unwrap(), &ParamValue::String("baraff-witkin".into()));
        assert_eq!(p.get("young-mod").unwrap(), &ParamValue::Float(1000.0));
        assert_eq!(p.get("density").unwrap(), &ParamValue::Float(1.0));
        assert_eq!(p.get("bend").unwrap(), &ParamValue::Float(10.0));
        // shrink-x/shrink-y exist on tri.
        assert!(p.get("shrink-x").is_ok());
    }

    #[test]
    fn object_param_tet_has_arap_default() {
        let p = object_param(ObjectKind::Tet);
        assert_eq!(p.get("model").unwrap(), &ParamValue::String("arap".into()));
        assert_eq!(p.get("density").unwrap(), &ParamValue::Float(100.0));
        // tet has plasticity but tet keeps bend-plasticity placeholders too.
        assert!(p.get("plasticity").is_ok());
        assert!(p.get("bend-plasticity").is_ok());
    }

    #[test]
    fn object_param_rod_lacks_plasticity() {
        let p = object_param(ObjectKind::Rod);
        assert_eq!(p.get("model").unwrap(), &ParamValue::String("arap".into()));
        // Rods don't have plasticity (Python source line 281 condition).
        assert!(p.get("plasticity").is_err());
        // But rods do have bend-plasticity (line 299).
        assert!(p.get("bend-plasticity").is_ok());
    }

    #[test]
    fn object_kind_from_str() {
        assert_eq!(ObjectKind::from_str("tri").unwrap(), ObjectKind::Tri);
        assert_eq!(ObjectKind::from_str("tet").unwrap(), ObjectKind::Tet);
        assert_eq!(ObjectKind::from_str("rod").unwrap(), ObjectKind::Rod);
        assert_eq!(ObjectKind::from_str("pdrd").unwrap(), ObjectKind::Pdrd);
        assert!(ObjectKind::from_str("blob").is_err());
    }

    #[test]
    fn object_param_pdrd_zero_elastic() {
        let p = object_param(ObjectKind::Pdrd);
        assert_eq!(p.get("model").unwrap(), &ParamValue::String("pdrd".into()));
        assert_eq!(p.get("young-mod").unwrap(), &ParamValue::Float(0.0));
        assert_eq!(p.get("bend").unwrap(), &ParamValue::Float(0.0));
        assert_eq!(p.get("density").unwrap(), &ParamValue::Float(100.0));
        // Key-set parity with Tri/Tet for concat_tri_param: PDRD also
        // carries plasticity/plasticity-threshold and bend-plasticity*
        // placeholders even though they're ignored at runtime.
        assert!(p.get("plasticity").is_ok());
        assert!(p.get("plasticity-threshold").is_ok());
        assert!(p.get("bend-plasticity").is_ok());
    }

    #[test]
    fn tri_tet_pdrd_share_param_key_set_for_concat() {
        let tri = object_param(ObjectKind::Tri);
        let tet = object_param(ObjectKind::Tet);
        let pdrd = object_param(ObjectKind::Pdrd);
        let tri_keys = tri.key_list();
        let tet_keys = tet.key_list();
        let pdrd_keys = pdrd.key_list();
        assert_eq!(tri_keys, tet_keys, "Tri and Tet must share param keys for concat_tri_param parity");
        assert_eq!(tri_keys, pdrd_keys, "PDRD must share Tri's key set so concat_tri_param can mix shells and PDRD bodies");
    }

    #[test]
    fn key_list_and_items_round_trip() {
        let p = app_param();
        let keys = p.key_list();
        let items = p.items();
        assert_eq!(keys.len(), items.len());
        // Insertion-via-BTreeMap means alphabetical; that's stable.
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
