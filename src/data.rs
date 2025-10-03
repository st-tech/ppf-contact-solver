// File: data.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use serde::{Deserialize, Serialize};

use super::cvec::*;
use super::cvecvec::*;


pub type Vec3f = na::Vector3<f32>;
pub type Vec3fp = na::Vector3<f32>;
pub type Vec2u = na::Vector2<u32>;
pub type Vec3u = na::Vector3<u32>;
pub type Vec4u = na::Vector4<u32>;
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
pub struct VertexProp {
    pub area: f32,
    pub volume: f32,
    pub mass: f32,
    pub ghat: f32,
    pub offset: f32,
    pub friction: f32,
    pub fix_index: u32,
    pub pull_index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct EdgeProp {
    pub fixed: bool,
    pub length: f32,
    pub mass: f32,
    pub stiffness: f32,
    pub bend: f32,
    pub ghat: f32,
    pub offset: f32,
    pub friction: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FaceProp {
    pub fixed: bool,
    pub area: f32,
    pub mass: f32,
    pub model: Model,
    pub mu: f32,
    pub lambda: f32,
    pub friction: f32,
    pub ghat: f32,
    pub offset: f32,
    pub strainlimit: f32,
    pub bend: f32,
    pub shrink: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct HingeProp {
    pub fixed: bool,
    pub length: f32,
    pub bend: f32,
    pub ghat: f32,
    pub offset: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct TetProp {
    pub fixed: bool,
    pub mass: f32,
    pub model: Model,
    pub volume: f32,
    pub mu: f32,
    pub lambda: f32,
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
pub struct Bvh {
    pub node: CVec<Vec2u>,
    pub level: CVecVec<u32>,
}

impl Bvh {
    pub fn new() -> Self {
        Self {
            node: CVec::new(),
            level: CVecVec::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct BvhSet {
    pub face: Bvh,
    pub edge: Bvh,
    pub vertex: Bvh,
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Copy, Clone, Default)]
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
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct FixPair {
    pub position: Vec3fp,
    pub ghat: f32,
    pub index: u32,
    pub kinematic: bool,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct PullPair {
    pub position: Vec3fp,
    pub weight: f32,
    pub index: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Stitch {
    pub index: Vec3u,
    pub weight: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3fp,
    pub ghat: f32,
    pub friction: f32,
    pub radius: f32,
    pub bowl: bool,
    pub reverse: bool,
    pub kinematic: bool,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Floor {
    pub ground: Vec3fp,
    pub ghat: f32,
    pub friction: f32,
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
pub struct CollisionMesh {
    pub vertex: CVec<Vec3fp>,
    pub face: CVec<Vec3u>,
    pub edge: CVec<Vec2u>,
    pub face_bvh: Bvh,
    pub edge_bvh: Bvh,
    pub prop: CollisionMeshPropSet,
    pub neighbor: Neighbor,
}

impl CollisionMesh {
    pub fn new() -> Self {
        Self {
            vertex: CVec::new(),
            face: CVec::new(),
            edge: CVec::new(),
            face_bvh: Bvh::new(),
            edge_bvh: Bvh::new(),
            prop: CollisionMeshPropSet::new(),
            neighbor: Neighbor::new(),
        }
    }
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Constraint {
    pub fix: CVec<FixPair>,
    pub pull: CVec<PullPair>,
    pub sphere: CVec<Sphere>,
    pub floor: CVec<Floor>,
    pub stitch: CVec<Stitch>,
    pub mesh: CollisionMesh,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct ParamSet {
    pub time: f64,
    pub disable_contact: bool,
    pub fitting: bool,
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
    pub csrmat_max_nnz: u32,
    pub bvh_alloc_factor: u32,
    pub fix_xz: f32,
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

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct VertexSet {
    pub prev: CVec<Vec3fp>,
    pub curr: CVec<Vec3fp>,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct DataSet {
    pub vertex: VertexSet,
    pub mesh: MeshInfo,
    pub prop: PropSet,
    pub inv_rest2x2: CVec<Mat2x2f>,
    pub inv_rest3x3: CVec<Mat3x3f>,
    pub constraint: Constraint,
    pub bvh: BvhSet,
    pub fixed_index_table: CVecVec<u32>,
    pub transpose_table: CVecVec<Vec2u>,
    pub rod_count: u32,
    pub shell_face_count: u32,
    pub surface_vert_count: u32,
}
