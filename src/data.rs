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
pub type Vec4f = na::Vector4<f32>;
pub type Mat2x2f = na::Matrix2<f32>;
pub type Mat3x3f = na::Matrix3<f32>;

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct VertexNeighbor {
    pub face: CVecVec<u32>,
    pub hinge: CVecVec<u32>,
    pub edge: CVecVec<u32>,
    pub rod: CVecVec<u32>,
}

impl VertexNeighbor {
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
pub struct HingeNeighbor {
    pub face: CVecVec<u32>,
}

impl HingeNeighbor {
    pub fn new() -> Self {
        Self {
            face: CVecVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct EdgeNeighbor {
    pub face: CVecVec<u32>,
}

impl EdgeNeighbor {
    pub fn new() -> Self {
        Self {
            face: CVecVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct MeshInfo {
    pub mesh: Mesh,
    pub neighbor: Neighbor,
    pub ttype: Type,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Mesh {
    pub face: CVec<Vec3u>,
    pub hinge: CVec<Vec4u>,
    pub edge: CVec<Vec2u>,
    pub tet: CVec<Vec4u>,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Neighbor {
    pub vertex: VertexNeighbor,
    pub hinge: HingeNeighbor,
    pub edge: EdgeNeighbor,
}

impl Neighbor {
    pub fn new() -> Self {
        Self {
            vertex: VertexNeighbor::new(),
            hinge: HingeNeighbor::new(),
            edge: EdgeNeighbor::new(),
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
}

impl std::hash::Hash for HingeParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bend.to_bits().hash(state);
        self.ghat.to_bits().hash(state);
        self.offset.to_bits().hash(state);
        self.plasticity.to_bits().hash(state);
        self.plasticity_threshold.to_bits().hash(state);
    }
}

impl PartialEq for HingeParam {
    fn eq(&self, other: &Self) -> bool {
        self.bend.to_bits() == other.bend.to_bits()
            && self.ghat.to_bits() == other.ghat.to_bits()
            && self.offset.to_bits() == other.offset.to_bits()
            && self.plasticity.to_bits() == other.plasticity.to_bits()
            && self.plasticity_threshold.to_bits() == other.plasticity_threshold.to_bits()
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
}

impl std::hash::Hash for TetParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.model.hash(state);
        self.mu.to_bits().hash(state);
        self.lambda.to_bits().hash(state);
        self.shrink.to_bits().hash(state);
        self.plasticity.to_bits().hash(state);
        self.plasticity_threshold.to_bits().hash(state);
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
    pub param_index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct HingeProp {
    pub length: f32,
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
    pub param_index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct PropSet {
    pub vertex: CVec<VertexProp>,
    pub edge: CVec<EdgeProp>,
    pub face: CVec<FaceProp>,
    pub hinge: CVec<HingeProp>,
    pub tet: CVec<TetProp>,
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
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub enum Barrier {
    Cubic,
    Quad,
    Log,
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub enum FrictionMode {
    Min,
    Max,
    Mean,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct FixPair {
    pub position: Vec3f,
    pub ghat: f32,
    pub index: u32,
    pub kinematic: bool,
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
    pub index: Vec4u,
    pub weight: Vec4f,
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
    pub neighbor: Neighbor,
}

impl CollisionMesh {
    pub fn new() -> Self {
        Self {
            vertex: CVec::new(),
            face: CVec::new(),
            edge: CVec::new(),
            prop: CollisionMeshPropSet::new(),
            param_arrays: CollisionMeshParamArrays::new(),
            neighbor: Neighbor::new(),
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
    pub stitch_stiffness: f32,
    pub cg_max_iter: u32,
    pub cg_tol: f32,
    pub line_search_max_t: f32,
    pub ccd_eps: f32,
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

pub const MAX_INTERSECTION_RECORDS: usize = 256;

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
    pub mesh: MeshInfo,
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
}
