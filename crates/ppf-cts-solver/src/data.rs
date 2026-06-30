// File: data.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use serde::{Deserialize, Serialize};

use super::cvec::*;
use super::cvecvec::*;

pub type Vec3f = na::Vector3<f32>;
pub type Vec2u = na::Vector2<u32>;
pub type Vec3u = na::Vector3<u32>;
pub type Vec4u = na::Vector4<u32>;
pub type Vec6u = na::SVector<u32, 6>;
pub type Vec6f = na::SVector<f32, 6>;
pub type Mat2x2f = na::Matrix2<f32>;
pub type Mat3x3f = na::Matrix3<f32>;

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct FfiVertexNeighbor {
    pub face: CVecVec<u32>,
    pub hinge: CVecVec<u32>,
    pub edge: CVecVec<u32>,
    pub rod: CVecVec<u32>,
}

impl FfiVertexNeighbor {
    pub fn new() -> Self {
        Self {
            face: CVecVec::new(),
            hinge: CVecVec::new(),
            edge: CVecVec::new(),
            rod: CVecVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct FfiHingeNeighbor {
    pub face: CVecVec<u32>,
}

impl FfiHingeNeighbor {
    pub fn new() -> Self {
        Self {
            face: CVecVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct FfiEdgeNeighbor {
    pub face: CVecVec<u32>,
}

impl FfiEdgeNeighbor {
    pub fn new() -> Self {
        Self {
            face: CVecVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct FfiMeshInfo {
    pub mesh: FfiMesh,
    pub neighbor: FfiNeighbor,
    pub ttype: Type,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct FfiMesh {
    pub face: CVec<Vec3u>,
    pub hinge: CVec<Vec4u>,
    pub edge: CVec<Vec2u>,
    pub tet: CVec<Vec4u>,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct FfiNeighbor {
    pub vertex: FfiVertexNeighbor,
    pub hinge: FfiHingeNeighbor,
    pub edge: FfiEdgeNeighbor,
}

impl FfiNeighbor {
    pub fn new() -> Self {
        Self {
            vertex: FfiVertexNeighbor::new(),
            hinge: FfiHingeNeighbor::new(),
            edge: FfiEdgeNeighbor::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Type {
    pub face: CVec<u8>,
    pub vertex: CVec<u8>,
    pub hinge: CVec<u8>,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct VertexParam {
    pub ghat: f32,
    pub offset: f32,
    pub friction: f32,
}

impl std::hash::Hash for VertexParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ghat.to_bits().hash(state);
        self.offset.to_bits().hash(state);
        self.friction.to_bits().hash(state);
    }
}

impl PartialEq for VertexParam {
    fn eq(&self, other: &Self) -> bool {
        self.ghat.to_bits() == other.ghat.to_bits()
            && self.offset.to_bits() == other.offset.to_bits()
            && self.friction.to_bits() == other.friction.to_bits()
    }
}

impl Eq for VertexParam {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct EdgeParam {
    pub stiffness: f32,
    pub bend: f32,
    pub ghat: f32,
    pub offset: f32,
    pub friction: f32,
    pub strainlimit: f32,
    pub plasticity: f32,
    pub plasticity_threshold: f32,
    pub bend_rest_from_geometry: bool,
    // Rayleigh damping coefficients (per object, replicated per element).
    // MUST mirror the C++ EdgeParam in cpp/data.hpp field-for-field (repr(C)).
    pub deform_damping: f32,
    pub bend_damping: f32,
}

impl std::hash::Hash for EdgeParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.stiffness.to_bits().hash(state);
        self.bend.to_bits().hash(state);
        self.ghat.to_bits().hash(state);
        self.offset.to_bits().hash(state);
        self.friction.to_bits().hash(state);
        self.strainlimit.to_bits().hash(state);
        self.plasticity.to_bits().hash(state);
        self.plasticity_threshold.to_bits().hash(state);
        self.bend_rest_from_geometry.hash(state);
        self.deform_damping.to_bits().hash(state);
        self.bend_damping.to_bits().hash(state);
    }
}

impl PartialEq for EdgeParam {
    fn eq(&self, other: &Self) -> bool {
        self.stiffness.to_bits() == other.stiffness.to_bits()
            && self.bend.to_bits() == other.bend.to_bits()
            && self.ghat.to_bits() == other.ghat.to_bits()
            && self.offset.to_bits() == other.offset.to_bits()
            && self.friction.to_bits() == other.friction.to_bits()
            && self.strainlimit.to_bits() == other.strainlimit.to_bits()
            && self.plasticity.to_bits() == other.plasticity.to_bits()
            && self.plasticity_threshold.to_bits() == other.plasticity_threshold.to_bits()
            && self.bend_rest_from_geometry == other.bend_rest_from_geometry
            && self.deform_damping.to_bits() == other.deform_damping.to_bits()
            && self.bend_damping.to_bits() == other.bend_damping.to_bits()
    }
}

impl Eq for EdgeParam {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FaceParam {
    pub model: Model,
    pub mu: f32,
    pub lambda: f32,
    pub friction: f32,
    pub ghat: f32,
    pub offset: f32,
    pub bend: f32,
    pub strainlimit: f32,
    pub shrink_x: f32,
    pub shrink_y: f32,
    pub pressure: f32,
    pub plasticity: f32,
    pub plasticity_threshold: f32,
    pub bend_plasticity: f32,
    pub bend_plasticity_threshold: f32,
    pub bend_rest_from_geometry: bool,
    // Rayleigh damping coefficients (per object, replicated per face).
    // MUST mirror the C++ FaceParam in cpp/data.hpp field-for-field (repr(C)).
    pub deform_damping: f32,
    pub bend_damping: f32,
}

impl std::hash::Hash for FaceParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        self.mu.to_bits().hash(state);
        self.lambda.to_bits().hash(state);
        self.friction.to_bits().hash(state);
        self.ghat.to_bits().hash(state);
        self.offset.to_bits().hash(state);
        self.bend.to_bits().hash(state);
        self.strainlimit.to_bits().hash(state);
        self.shrink_x.to_bits().hash(state);
        self.shrink_y.to_bits().hash(state);
        self.pressure.to_bits().hash(state);
        self.plasticity.to_bits().hash(state);
        self.plasticity_threshold.to_bits().hash(state);
        self.bend_plasticity.to_bits().hash(state);
        self.bend_plasticity_threshold.to_bits().hash(state);
        self.bend_rest_from_geometry.hash(state);
        self.deform_damping.to_bits().hash(state);
        self.bend_damping.to_bits().hash(state);
    }
}

impl PartialEq for FaceParam {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.mu.to_bits() == other.mu.to_bits()
            && self.lambda.to_bits() == other.lambda.to_bits()
            && self.friction.to_bits() == other.friction.to_bits()
            && self.ghat.to_bits() == other.ghat.to_bits()
            && self.offset.to_bits() == other.offset.to_bits()
            && self.bend.to_bits() == other.bend.to_bits()
            && self.strainlimit.to_bits() == other.strainlimit.to_bits()
            && self.shrink_x.to_bits() == other.shrink_x.to_bits()
            && self.shrink_y.to_bits() == other.shrink_y.to_bits()
            && self.pressure.to_bits() == other.pressure.to_bits()
            && self.plasticity.to_bits() == other.plasticity.to_bits()
            && self.plasticity_threshold.to_bits() == other.plasticity_threshold.to_bits()
            && self.bend_plasticity.to_bits() == other.bend_plasticity.to_bits()
            && self.bend_plasticity_threshold.to_bits() == other.bend_plasticity_threshold.to_bits()
            && self.bend_rest_from_geometry == other.bend_rest_from_geometry
            && self.deform_damping.to_bits() == other.deform_damping.to_bits()
            && self.bend_damping.to_bits() == other.bend_damping.to_bits()
    }
}

impl Eq for FaceParam {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct HingeParam {
    pub bend: f32,
    pub ghat: f32,
    pub offset: f32,
    pub plasticity: f32,
    pub plasticity_threshold: f32,
    // Rayleigh bending damping, area-averaged from adjacent faces in builder.rs.
    // MUST mirror the C++ HingeParam in cpp/data.hpp (repr(C)).
    pub bend_damping: f32,
}

impl std::hash::Hash for HingeParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bend.to_bits().hash(state);
        self.ghat.to_bits().hash(state);
        self.offset.to_bits().hash(state);
        self.plasticity.to_bits().hash(state);
        self.plasticity_threshold.to_bits().hash(state);
        self.bend_damping.to_bits().hash(state);
    }
}

impl PartialEq for HingeParam {
    fn eq(&self, other: &Self) -> bool {
        self.bend.to_bits() == other.bend.to_bits()
            && self.ghat.to_bits() == other.ghat.to_bits()
            && self.offset.to_bits() == other.offset.to_bits()
            && self.plasticity.to_bits() == other.plasticity.to_bits()
            && self.plasticity_threshold.to_bits() == other.plasticity_threshold.to_bits()
            && self.bend_damping.to_bits() == other.bend_damping.to_bits()
    }
}

impl Eq for HingeParam {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct TetParam {
    pub model: Model,
    pub mu: f32,
    pub lambda: f32,
    pub shrink: f32,
    pub plasticity: f32,
    pub plasticity_threshold: f32,
    // Rayleigh deformation damping, scales the solid tangent stiffness.
    // MUST mirror the C++ TetParam in cpp/data.hpp (repr(C)).
    pub deform_damping: f32,
}

impl std::hash::Hash for TetParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        self.mu.to_bits().hash(state);
        self.lambda.to_bits().hash(state);
        self.shrink.to_bits().hash(state);
        self.plasticity.to_bits().hash(state);
        self.plasticity_threshold.to_bits().hash(state);
        self.deform_damping.to_bits().hash(state);
    }
}

impl PartialEq for TetParam {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.mu.to_bits() == other.mu.to_bits()
            && self.lambda.to_bits() == other.lambda.to_bits()
            && self.shrink.to_bits() == other.shrink.to_bits()
            && self.plasticity.to_bits() == other.plasticity.to_bits()
            && self.plasticity_threshold.to_bits() == other.plasticity_threshold.to_bits()
            && self.deform_damping.to_bits() == other.deform_damping.to_bits()
    }
}

impl Eq for TetParam {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct VertexProp {
    pub area: f32,
    pub volume: f32,
    pub mass: f32,
    pub rest_bend_angle: f32,
    pub fix_index: u32,
    pub pull_index: u32,
    pub param_index: u32,
    /// 1-based PDRD body id (0 = not a member of any PDRD body).
    /// Used by the PDRD rigid-fit kernel to find the body this vertex
    /// belongs to, and by contact culling to skip same-body
    /// vertex/edge/face pairs (intra-body collisions are excluded
    /// because PDRD bodies move as exactly rigid bodies).
    pub pdrd_body_index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct EdgeProp {
    pub length: f32,
    pub initial_length: f32,
    pub mass: f32,
    pub fixed: bool,
    pub param_index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FaceProp {
    pub area: f32,
    pub mass: f32,
    pub fixed: bool,
    /// Excluded from the elastic/strain energy this frame because its streamed
    /// rest shape is near-singular (set per frame by `update_rest_shape`). Kept
    /// separate from `fixed` (kinematic pinning) so the two never alias. Field
    /// order must mirror `FaceProp` in `cpp/data.hpp` (repr(C) ABI).
    pub rest_excluded: bool,
    pub param_index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct HingeProp {
    pub length: f32,
    // Combined rest area of the two triangles incident to the hinge edge
    // (A1 + A2). With `length` it gives the resolution-independent bending
    // coefficient |e|^2 / area (see energy.cu embed_hinge_force_hessian).
    pub area: f32,
    pub rest_angle: f32,
    pub fixed: bool,
    pub param_index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct TetProp {
    pub mass: f32,
    pub volume: f32,
    pub fixed: bool,
    /// See `FaceProp::rest_excluded`. Field order must mirror `TetProp` in
    /// `cpp/data.hpp` (repr(C) ABI).
    pub rest_excluded: bool,
    pub param_index: u32,
}

/// One row per Painless Differentiable Rotation Dynamics body in the scene. Bodies own a
/// contiguous slice of `pdrd_vert_list` (NOT of the global vertex
/// array, see the `DataSet::pdrd_vert_list` doc). Rest-shape moments
/// (centroid and inverse Gram matrix of centered rest positions) are
/// precomputed at build time so the kernel can fit the body's best-fit
/// rigid transform (rotation + translation) each iteration.
#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct PdrdBodyProp {
    pub rest_centroid: Vec3f,
    pub rest_gram_inv: Mat3x3f,
    pub volume: f32,
    pub vertex_start: u32,
    pub vertex_count: u32,
    /// Per-vertex mass for this body: density × volume × scale / N,
    /// with `scale = trace(I_solid) / trace(I_uniform)` so the body's
    /// effective rotational inertia matches the volumetric tensor.
    /// Used only by the builder to override the per-vertex mass; not
    /// read by the kernel.
    pub mass_per_vertex: f32,
    /// Joint / DOF-filtering mode. 0 = free (full 6-DOF rigid body, the
    /// default), 1 = hinge: the body's reduced rigid DOF are filtered so
    /// translation is locked and rotation is restricted to the single
    /// world axle `joint_axis` through the body centroid (a pin joint).
    /// The filtering is applied in the reduced linear solve (see
    /// `project_body_dofs_kernel` in `pdrd_rigid.hpp`). Inert when 0, so
    /// non-jointed PDRD scenes are byte-for-byte unaffected. Field order
    /// must mirror `PdrdBodyProp` in `cpp/data.hpp` (repr(C) ABI).
    pub joint_mode: u32,
    /// World-frame unit rotation axle for a hinge: the chosen principal
    /// axis of the rest shape, evaluated at t=0 (the body starts at its
    /// rest orientation in world, so this is a fixed world direction).
    /// Unused when `joint_mode == 0`.
    pub joint_axis: Vec3f,
    /// World-frame anchor of the hinge axle (the body's initial world
    /// centroid). Recorded for tooling / visualization and a future
    /// off-centroid pivot; the live solve uses the body centroid as the
    /// pivot. Unused when `joint_mode == 0`.
    pub joint_pin: Vec3f,
}

/// Number of `f32` per PDRD body row in `pdrd_body.bin` and
/// `builder::PdrdSceneData::body_rows`. Layout: `vertex_start`,
/// `vertex_count`, `volume`, `centroid[3]`, `rest_gram_inv[9 row-major]`,
/// `mass_per_vertex`, `joint_mode`, `joint_axis[3]`, `joint_pin[3]`
/// (16 base + 7 joint = 23 floats).
pub const PDRD_BODY_ROW_LEN: usize = 23;

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct PropSet {
    pub vertex: CVec<VertexProp>,
    pub edge: CVec<EdgeProp>,
    pub face: CVec<FaceProp>,
    pub hinge: CVec<HingeProp>,
    pub tet: CVec<TetProp>,
    pub pdrd_body: CVec<PdrdBodyProp>,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct ParamArrays {
    pub vertex: CVec<VertexParam>,
    pub edge: CVec<EdgeParam>,
    pub face: CVec<FaceParam>,
    pub hinge: CVec<HingeParam>,
    pub tet: CVec<TetParam>,
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Copy, Clone, Default, Hash, PartialEq, Eq)]
pub enum Model {
    #[default]
    Arap,
    StVK,
    BaraffWitkin,
    SNHk,
    /// Painless Differentiable Rotation Dynamics: the face/tet that declares this model has
    /// no per-element elastic energy. The body is handled by the
    /// per-body rigid-fit path; the standard elastic dispatch must
    /// branch on `Model::Pdrd` and skip these elements.
    Pdrd,
}

impl Model {
    /// Decode a per-element u8 id (as written by the Python exporter)
    /// into a `Model`. The authoritative id<->name table lives in
    /// `ppf_cts_core::datamodel::elastic_model`; this maps the resolved
    /// name onto the matching `repr(C)` variant. The variant order must
    /// stay in step with that table (and the C++ `Model` enum), so this
    /// match is the single place the solver binds a name to a variant.
    pub fn from_id(id: u8) -> Option<Self> {
        match ppf_cts_core::datamodel::elastic_model::model_id_to_name(id)? {
            "arap" => Some(Model::Arap),
            "stvk" => Some(Model::StVK),
            "baraff-witkin" => Some(Model::BaraffWitkin),
            "snhk" => Some(Model::SNHk),
            "pdrd" => Some(Model::Pdrd),
            _ => None,
        }
    }
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub enum Barrier {
    Cubic,
    Quad,
    Log,
}

impl std::str::FromStr for Barrier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cubic" => Ok(Barrier::Cubic),
            "quad" => Ok(Barrier::Quad),
            "log" => Ok(Barrier::Log),
            _ => Err(format!(
                "Invalid barrier: {s} (valid choices: cubic, quad, log)"
            )),
        }
    }
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub enum FrictionMode {
    Min,
    Max,
    Mean,
}

impl std::str::FromStr for FrictionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "min" => Ok(FrictionMode::Min),
            "max" => Ok(FrictionMode::Max),
            "mean" => Ok(FrictionMode::Mean),
            _ => Err(format!(
                "Invalid friction-mode: {s} (valid choices: min, max, mean)"
            )),
        }
    }
}

/// Linear-solve preconditioner selector. Mirrors the C++ `enum class PrecondMode`
/// in `cpp/data.hpp` (repr(C) ABI; 4-byte int). The variant order MUST stay in
/// step with the C++ enum.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrecondMode {
    BlockJacobi,
    Schwarz,
}

impl std::str::FromStr for PrecondMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "block-jacobi" | "blockjacobi" | "jacobi" => Ok(PrecondMode::BlockJacobi),
            "schwarz" => Ok(PrecondMode::Schwarz),
            _ => Err(format!(
                "Invalid precond: {s} (valid choices: block-jacobi, schwarz)"
            )),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct FixPair {
    pub position: Vec3f,
    pub ghat: f32,
    pub index: u32,
    pub kinematic: bool,
    /// Per-pin scale on the constraint force, applied device-side only
    /// when `kinematic` is true. 1.0 leaves the force unchanged. Field
    /// order must mirror `FixPair` in `cpp/data.hpp` (repr(C) ABI).
    pub stiffness: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct PullPair {
    pub position: Vec3f,
    pub weight: f32,
    pub index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TorqueGroup {
    pub axis_component: u32,
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub hint_vertex: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TorqueVertex {
    pub magnitude: f32,
    pub index: u32,
    pub group_id: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Stitch {
    pub index: Vec6u,
    pub weight: Vec6f,
    pub stiffness: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3f,
    pub ghat: f32,
    pub friction: f32,
    pub radius: f32,
    pub thickness: f32,
    pub bowl: bool,
    pub reverse: bool,
    pub kinematic: bool,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Floor {
    pub ground: Vec3f,
    pub ghat: f32,
    pub friction: f32,
    pub thickness: f32,
    pub up: Vec3f,
    pub kinematic: bool,
}


#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct CollisionMeshPropSet {
    pub vertex: CVec<VertexProp>,
    pub face: CVec<FaceProp>,
    pub edge: CVec<EdgeProp>,
}

impl CollisionMeshPropSet {
    pub fn new() -> Self {
        Self {
            vertex: CVec::new(),
            face: CVec::new(),
            edge: CVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct CollisionMeshParamArrays {
    pub vertex: CVec<VertexParam>,
    pub face: CVec<FaceParam>,
    pub edge: CVec<EdgeParam>,
}

impl CollisionMeshParamArrays {
    pub fn new() -> Self {
        Self {
            vertex: CVec::new(),
            face: CVec::new(),
            edge: CVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct CollisionMesh {
    pub vertex: CVec<Vec3f>,
    pub face: CVec<Vec3u>,
    pub edge: CVec<Vec2u>,
    pub prop: CollisionMeshPropSet,
    pub param_arrays: CollisionMeshParamArrays,
    pub neighbor: FfiNeighbor,
}

impl CollisionMesh {
    pub fn new() -> Self {
        Self {
            vertex: CVec::new(),
            face: CVec::new(),
            edge: CVec::new(),
            prop: CollisionMeshPropSet::new(),
            param_arrays: CollisionMeshParamArrays::new(),
            neighbor: FfiNeighbor::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Constraint {
    pub fix: CVec<FixPair>,
    pub pull: CVec<PullPair>,
    pub torque_groups: CVec<TorqueGroup>,
    pub torque_vertices: CVec<TorqueVertex>,
    pub sphere: CVec<Sphere>,
    pub floor: CVec<Floor>,
    pub stitch: CVec<Stitch>,
    pub mesh: CollisionMesh,
}

/// Per-frame replacement of the inverse rest matrices, uploaded to the device
/// next to `update_constraint` when a time-varying rest shape is streamed (the
/// `rest_vert_schedule` path). The C++ side copies these straight into
/// `DataSet::inv_rest2x2` / `inv_rest3x3`, which the elastic kernels re-read
/// each Newton iteration. Field order must mirror `RestShapeUpdate` in
/// `cpp/data.hpp` (repr(C) ABI).
#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct RestShapeUpdate {
    pub inv_rest2x2: CVec<Mat2x2f>,
    pub inv_rest3x3: CVec<Mat3x3f>,
    /// Per shell-face / per-tet flag (1 = exclude this element from the elastic
    /// and strain energy this frame, by OR-ing into its `fixed` prop). Set when
    /// the streamed rest element is near-singular; see `builder::compute_inv_rest`.
    pub exclude_face: CVec<u8>,
    pub exclude_tet: CVec<u8>,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct ParamSet {
    pub time: f64,
    pub air_friction: f32,
    pub air_density: f32,
    pub constraint_tol: f32,
    pub prev_dt: f32,
    pub dt: f32,
    pub playback: f32,
    pub min_newton_steps: u32,
    pub target_toi: f32,
    pub stitch_length_factor: f32,
    pub cg_max_iter: u32,
    pub cg_tol: f32,
    pub line_search_max_t: f32,
    pub ccd_reduction: f32,
    pub ccd_max_iter: u32,
    pub max_dx: f32,
    pub eiganalysis_eps: f32,
    pub friction_eps: f32,
    pub isotropic_air_friction: f32,
    pub gravity: Vec3f,
    pub wind: Vec3f,
    pub barrier: Barrier,
    pub friction_mode: FrictionMode,
    pub csrmat_max_nnz: u32,
    pub fix_xz: f32,
    pub disable_contact: bool,
    pub inactive_momentum: bool,
    // Linear-solve preconditioner. Appended at the tail; field order and byte
    // layout MUST mirror the C++ ParamSet in cpp/data.hpp (repr(C) ABI).
    pub precond: PrecondMode,
    // Number of additive Schwarz levels (1 = single-level smoother, 2 =
    // two-level coarse correction). Only consulted when precond == Schwarz.
    // Appended at the tail; field order and byte layout MUST mirror the C++
    // ParamSet in cpp/data.hpp (repr(C) ABI).
    pub schwarz_levels: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Default)]
pub struct StepResult {
    pub time: f64,
    pub ccd_success: bool,
    pub pcg_success: bool,
    pub intersection_free: bool,
}

impl StepResult {
    pub fn success(&self) -> bool {
        self.ccd_success && self.pcg_success && self.intersection_free
    }
}

// Rust mirror of the C++ ABI capacity. The canonical definition lives in
// cpp/data.hpp (`#define MAX_INTERSECTION_RECORDS`), which bounds every
// C++-side write; this const MUST stay equal to it. The FFI buffer in
// backend.rs is sized with this value and passed as max_count, so a divergence
// silently under-reports intersection records. Update both files together.
pub const MAX_INTERSECTION_RECORDS: usize = 256;

/// One intersection record copied out of the C++ backend for the diagnostics
/// dump. Field order and array sizes (including `positions: [f32; 15]`) must
/// mirror `IntersectionRecord` in `cpp/data.hpp` (repr(C) ABI).
#[repr(C)]
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct IntersectionRecord {
    pub itype: u32,
    pub elem0: u32,
    pub elem1: u32,
    pub num_verts0: u32,
    pub num_verts1: u32,
    pub positions: [f32; 15],
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct VertexSet {
    pub prev: CVec<Vec3f>,
    pub curr: CVec<Vec3f>,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct DataSet {
    pub vertex: VertexSet,
    pub mesh: FfiMeshInfo,
    pub prop: PropSet,
    pub param_arrays: ParamArrays,
    pub inv_rest2x2: CVec<Mat2x2f>,
    pub inv_rest3x3: CVec<Mat3x3f>,
    pub constraint: Constraint,
    pub fixed_index_table: CVecVec<u32>,
    pub transpose_table: CVecVec<Vec2u>,
    pub rod_count: u32,
    pub shell_face_count: u32,
    pub surface_vert_count: u32,
    /// Flat list of global vertex indices that participate in an
    /// PDRD body. Bodies' slices are contiguous in this list (each
    /// `PdrdBodyProp` carries an offset+count into it). The list is
    /// independent of the global vertex layout, so PDRD vertices need
    /// not be contiguous in the global array.
    pub pdrd_vert_list: CVec<u32>,
    /// Centered rest position `ȳₖ = x̄ₖ − c̄_body` for each entry
    /// in `pdrd_vert_list`, in the same order. The kernel reads
    /// `eval_x[pdrd_vert_list[start+k]]` for the current position
    /// and `pdrd_rest_centered[start+k]` for the rest.
    pub pdrd_rest_centered: CVec<Vec3f>,
    /// SAND grain spin (rolling). Per-vertex angular velocity (rad/s), zero
    /// for non-grain vertices. The contact-friction torque integrates it
    /// post-solve and the lagged value feeds `v_contact = v + omega x r` in the
    /// next step's friction. A grain has isotropic sphere inertia and the
    /// torque arm is the contact normal, so no orientation is stored.
    pub grain_omega: CVec<Vec3f>,
    /// Per-grain inverse moment of inertia `1/((2/5) m r^2)` for a solid
    /// sphere, zero for non-grain vertices so the integrate skips them.
    pub grain_inv_inertia: CVec<f32>,
    /// Per-grain contact-friction torque `tau = sum r x F_t`, a transient
    /// working buffer the contact embed writes and the post-solve integrate
    /// consumes (overwritten each step). Built as zeros; zero for non-grains.
    pub grain_torque: CVec<Vec3f>,
    /// Per-grain angular friction stiffness `K = sum lambda*radius^2`, written
    /// alongside `grain_torque` and used for the semi-implicit omega step that
    /// stops the friction torque from overshooting the rolling rate. Transient,
    /// built as zeros.
    pub grain_ang_stiff: CVec<f32>,
    /// Per-grain SUM of unit contact normals over all the grain's contacts this
    /// step (zero when airborne). Written by the contact embed and consumed by
    /// the post-solve integrate, which normalizes it to the dominant contact
    /// direction and caps the spin at the tangential no-slip rate. Summing (not
    /// last-wins) keeps the clamp correct under multiple simultaneous contacts.
    /// Transient, built as zeros.
    pub grain_contact_normal: CVec<Vec3f>,
    /// Implicit (Schur-condensed) rolling: `1/I_center = 1/((2/5) m r^2)`, the
    /// bare center inertia for the Schur condense/recover (grain_inv_inertia
    /// stays `1/I_eff` for the staggered grain-grain path). Zero for non-grains.
    pub grain_inv_inertia_center: CVec<f32>,
    /// Start-of-step angular-velocity snapshot, held across Newton iterations as
    /// the inertia reference for the implicit rolling solve. Built as zeros.
    pub grain_omega_prev: CVec<Vec3f>,
    /// Transient per-grain Schur blocks (zeroed each Newton iteration, summed over
    /// the grain's floor/sphere contacts): `grain_A` SPD angular block, `grain_B`
    /// translation<->rotation coupling, `grain_grot` rotational gradient.
    pub grain_A: CVec<Mat3x3f>,
    pub grain_B: CVec<Mat3x3f>,
    pub grain_grot: CVec<Vec3f>,
}
