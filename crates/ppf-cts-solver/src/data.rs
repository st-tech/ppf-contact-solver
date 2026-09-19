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

// A device buffer is made of this, so `Pod` is stated where the layout is.
unsafe impl ppf_cts_compute::Pod for VertexParam {}

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
    // MUST mirror the C++ EdgeParam in data_records.hpp field-for-field (repr(C)).
    pub deform_damping: f32,
    pub bend_damping: f32,
}

// A device buffer is made of this, so `Pod` is stated where the layout is.
unsafe impl ppf_cts_compute::Pod for EdgeParam {}

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
    // MUST mirror the C++ FaceParam in data_records.hpp field-for-field (repr(C)).
    pub deform_damping: f32,
    pub bend_damping: f32,
    // Directional shell bending stiffnesses, ADDED to the isotropic `bend` at
    // an orientation-dependent weight; carried to the hinges (area-averaged in
    // builder.rs) exactly as `bend` and `bend_damping` are. Both 0.0 means
    // isotropic. Tail-appended; MUST mirror the C++ FaceParam in data_records.hpp
    // (repr(C) ABI).
    pub bend_warp: f32,
    pub bend_weft: f32,
}

// A device buffer is made of this, so `Pod` is stated where the layout is.
unsafe impl ppf_cts_compute::Pod for FaceParam {}

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
        self.bend_warp.to_bits().hash(state);
        self.bend_weft.to_bits().hash(state);
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
            && self.bend_warp.to_bits() == other.bend_warp.to_bits()
            && self.bend_weft.to_bits() == other.bend_weft.to_bits()
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
    // MUST mirror the C++ HingeParam in data_records.hpp (repr(C)).
    pub bend_damping: f32,
    // Directional bending stiffnesses added to `bend`, area-averaged from the
    // adjacent faces alongside it. These stay in the param (deduped, a handful
    // of distinct values per scene) rather than the prop so that retuning them
    // re-sends the param payload without rebuilding the mesh dataset; the
    // per-hinge geometry they weight lives in HingeProp::uv_edge_sin2.
    // Tail-appended; MUST mirror the C++ HingeParam (repr(C)).
    pub bend_warp: f32,
    pub bend_weft: f32,
}

unsafe impl ppf_cts_compute::Pod for HingeParam {}

impl std::hash::Hash for HingeParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bend.to_bits().hash(state);
        self.ghat.to_bits().hash(state);
        self.offset.to_bits().hash(state);
        self.plasticity.to_bits().hash(state);
        self.plasticity_threshold.to_bits().hash(state);
        self.bend_damping.to_bits().hash(state);
        self.bend_warp.to_bits().hash(state);
        self.bend_weft.to_bits().hash(state);
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
            && self.bend_warp.to_bits() == other.bend_warp.to_bits()
            && self.bend_weft.to_bits() == other.bend_weft.to_bits()
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
    // MUST mirror the C++ TetParam in data_records.hpp (repr(C)).
    pub deform_damping: f32,
}

// Safety: `repr(C)`, every field a scalar or a `repr(C)` scalar enum, no
// padding the device reads and no pointer.
unsafe impl ppf_cts_compute::Pod for TetParam {}

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
    /// Vertex belongs to a STATIC collider. Contact and intersection
    /// reporting skip a pair whose two sides are both collider vertices,
    /// whether that is one collider against itself or two different ones: a
    /// collider's shape is authored and driven, so neither side can yield to
    /// relieve the contact. An exactly pinned collider was already excluded
    /// by `either_dyn`; this preserves the exclusion once the collider is
    /// held by springs and its vertices become free. Field order must mirror
    /// `VertexProp` in `data_records.hpp` (repr(C) ABI).
    pub collider: bool,
    /// Source-object identity, the only thing that separates a SELF-
    /// intersection from an INTER-OBJECT one. `param_index` cannot serve:
    /// identical materials deduplicate to one entry, so two objects share it.
    /// Read off an element's FIRST vertex, the convention `pdrd_body_index`
    /// and `collider` already use, which holds because a vertex belongs to
    /// exactly one object and therefore so does every element built on it.
    /// `NO_OBJECT_INDEX` when the session directory carries no
    /// `object_vert.bin`. Field order must mirror `VertexProp` in
    /// `data_records.hpp` (repr(C) ABI).
    pub object_index: u32,
    /// This vertex's object's intersection tolerances, as
    /// `INTERSECT_ALLOW_SELF | INTERSECT_ALLOW_INTER_OBJECT`. Resolved per
    /// OBJECT by the frontend rather than per element, which is both the
    /// granularity the material param actually has and the only one defined
    /// for a faceless SAND grain (no incident element to read a material
    /// from).
    pub intersect_policy: u8,
    /// Every pin covering this vertex asked for its intersections to be
    /// tolerated. False for an unpinned vertex, so an element earns the
    /// exemption only when ALL of its vertices carry this. Latched at scene
    /// build from the initial constraint set, exactly like `FaceProp::fixed`;
    /// a pin that later reaches its unpin time does not take it back.
    pub pin_allow_intersection: bool,
}

// A device buffer is made of this, so the layout is asserted where it is
// stated. `#[repr(C)]` plus `Copy` above is exactly what `Pod` requires.
unsafe impl ppf_cts_compute::Pod for VertexProp {}

/// `VertexProp::object_index` for a vertex whose source object is unknown,
/// which is what a session directory written before `object_vert.bin` gives.
/// Deliberately not 0: two unknown indices must not compare equal, or every
/// such pair would read as a self-intersection and take that allowance.
pub const NO_OBJECT_INDEX: u32 = u32::MAX;

/// `VertexProp::intersect_policy` bits. Mirrored in `data_records.hpp`.
pub const INTERSECT_ALLOW_SELF: u8 = 1 << 0;
pub const INTERSECT_ALLOW_INTER_OBJECT: u8 = 1 << 1;

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct EdgeProp {
    pub length: f32,
    pub initial_length: f32,
    pub mass: f32,
    pub fixed: bool,
    pub param_index: u32,
    /// All of this edge's vertices are pinned by pins that asked for their
    /// intersections to be tolerated. Precomputed from
    /// `VertexProp::pin_allow_intersection` the same way `fixed` is
    /// precomputed from `fix_index`, and carrying the same build-time-snapshot
    /// caveat. Field order must mirror `EdgeProp` in `data_records.hpp`.
    pub pin_allow_intersection: bool,
}

// A device buffer is made of this, so `Pod` is stated where the layout is.
unsafe impl ppf_cts_compute::Pod for EdgeProp {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FaceProp {
    pub area: f32,
    pub mass: f32,
    pub fixed: bool,
    /// Excluded from the elastic/strain energy this frame because its streamed
    /// rest shape is near-singular (set per frame by `update_rest_shape`). Kept
    /// separate from `fixed` (kinematic pinning) so the two never alias. Field
    /// order must mirror `FaceProp` in `data_records.hpp` (repr(C) ABI).
    pub rest_excluded: bool,
    /// Belongs to a STATIC collider, so it carries no elastic energy. A
    /// collider's shape is held by its pins, not by stiffness of its own:
    /// while the pins were exact its DOF were removed and `fixed` already
    /// suppressed this term, but a spring-held collider's vertices are free
    /// and the term would otherwise come alive with whatever material the
    /// defaults supplied. Kept separate from both `fixed` (kinematic pinning)
    /// and `rest_excluded` (owned per frame by `update_rest_shape`) so none of
    /// the three alias. Field order must mirror `FaceProp` in `data_records.hpp`.
    pub collider: bool,
    pub param_index: u32,
    /// All of this face's vertices are pinned by pins that asked for their
    /// intersections to be tolerated. See `EdgeProp::pin_allow_intersection`.
    pub pin_allow_intersection: bool,
}

// A device buffer is made of this, so `Pod` is stated where the layout is.
unsafe impl ppf_cts_compute::Pod for FaceProp {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct HingeProp {
    pub length: f32,
    // Combined rest area of the two triangles incident to the hinge edge
    // (A1 + A2). With `length` it gives the resolution-independent bending
    // coefficient |e|^2 / area that `shell_bend_stiffness` forms in
    // energy/model/shell_bend_stiffness.kernel.cpp.
    pub area: f32,
    pub rest_angle: f32,
    /// `sin^2(psi)` for `psi` the angle between this hinge's shared edge and
    /// the UV X (warp) axis, in [0, 1], or -1.0 when the mesh carries no UV
    /// and the hinge is therefore isotropic. A hinge bends about its shared
    /// edge, so the surface curves ACROSS that edge: an edge along warp
    /// (`psi = 0`, `sin^2 = 0`) resists with the weft stiffness, which is why
    /// `shell_bend_directional` (energy/model/shell_bend_stiffness.kernel.cpp)
    /// pairs `sin^2` with warp and `cos^2` with weft. Squared components are
    /// stored rather than the angle so the kernel needs no trigonometry.
    /// Rest-topology constant, so the streamed rest-shape path does not
    /// recompute it.
    pub uv_edge_sin2: f32,
    pub fixed: bool,
    /// Belongs to a STATIC collider, so it carries no bending energy. See
    /// `FaceProp::collider`. Field order must mirror `HingeProp` in
    /// `data_records.hpp` (repr(C) ABI).
    pub collider: bool,
    pub param_index: u32,
}

// PLAIN DATA, like every other record staged to a device: `repr(C)`, no
// padding a reader could observe, and nothing that owns memory.
unsafe impl ppf_cts_compute::Pod for HingeProp {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct TetProp {
    pub mass: f32,
    pub volume: f32,
    pub fixed: bool,
    /// See `FaceProp::rest_excluded`. Field order must mirror `TetProp` in
    /// `data_records.hpp` (repr(C) ABI).
    pub rest_excluded: bool,
    pub param_index: u32,
}

// Safety: as `TetParam` above; the two `bool`s are the same one-byte shape
// the shared `TetProp` record declares.
unsafe impl ppf_cts_compute::Pod for TetProp {}

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
    /// `pdrd_project_body_dofs_row` in
    /// `energy/model/pdrd_lock_projector.kernel.cpp`). Inert when 0, so
    /// non-jointed PDRD scenes are byte-for-byte unaffected. Field order
    /// must mirror `PdrdBodyProp` in `data_records.hpp` (repr(C) ABI).
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

// SAFETY: `#[repr(C)]` with the field order the C++ `PdrdBodyProp` in
// data_records.hpp mirrors, floats, a 3x3, a float triple and unsigneds.
unsafe impl ppf_cts_compute::Pod for PdrdBodyProp {}

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
// `Copy` so a backend can read the selector out of a borrowed `ParamSet`: it is
// a fieldless `repr(C)` enum, so the copy is the four bytes the C++ side reads
// and the trait says what was already true of the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
// `Copy` for the reason `Barrier` above carries it: a backend reads the
// selector out of a borrowed `ParamSet`, and a fieldless `repr(C)` enum copies
// as the four bytes the C++ side reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
/// in `data_records.hpp` (repr(C) ABI; 4-byte int). The variant order MUST stay in
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
// `Default` for the same reason `PullPair` carries one: a device allocation of
// these is zeroed at `size`.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FixPair {
    /// Where the pin's prescribed path puts this vertex at the END of the
    /// step (the time the host aimed the step at).
    pub position: Vec3f,
    /// Displacement the pin travels over this step: `position` minus the same
    /// path evaluated at the step's START time. The CCD line search can
    /// truncate a step to a fraction of its span, and a kinematic pin must
    /// then stop at that same fraction of the path it was scheduled to travel,
    /// or it outruns the clock. `rewind_fix` (`main/rewind_fix.kernel.cpp`)
    /// applies that truncation, subtracting `(1 - toi) * step_delta` from
    /// `position`. Zero for a static pin, which never moves.
    pub step_delta: Vec3f,
    pub ghat: f32,
    pub index: u32,
    pub kinematic: bool,
    /// The pin that placed this vertex asked for its intersections to be
    /// tolerated. Consumed once, at scene build, to latch
    /// `VertexProp::pin_allow_intersection`; the solver's per-step constraint
    /// rebuild carries it along so the two constructions cannot disagree.
    pub allow_intersection: bool,
    // NOTE: every fix pin is an exact Dirichlet BC (the solver eliminates its
    // DOF), so there is no per-pin stiffness to scale: there is no penalty force
    // left. `ghat` and `kinematic` survive only for the PDRD anchor, the one pin
    // still held by the barrier (a vertex inside a rigid body owns no per-vertex
    // DOF). Field order must mirror `FixPair` in `data_records.hpp` (repr(C) ABI).
}

// SAFETY: as `PullPair` below. `#[repr(C)]` in the field order `data_records.hpp`
// mirrors, and every field is a float triple, a scalar or a `bool` the shared
// bodies write as 0 or 1.
unsafe impl ppf_cts_compute::Pod for FixPair {}

#[repr(C)]
// `Default` because a device allocation of these is zeroed at `size`, and an
// unused slot in one is the all-zero pin: index 0, no weight, at the origin.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct PullPair {
    pub position: Vec3f,
    pub weight: f32,
    pub index: u32,
    /// See `FixPair::allow_intersection`. A pull pin holds its vertex only to
    /// the extent of its own force, so an intersection under one is often not
    /// something the pin is responsible for; that is the case issue #138
    /// singles out.
    pub allow_intersection: bool,
}

// SAFETY: `#[repr(C)]` with the field order the C++ `PullPair` in data_records.hpp
// mirrors, and every field is a float triple, a float, an integer or a `bool`
// the shared bodies write as 0 or 1. A device allocation of these and the
// bodies' view of them are the same bytes.
unsafe impl ppf_cts_compute::Pod for PullPair {}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct TorqueGroup {
    pub axis_component: u32,
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub hint_vertex: u32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct TorqueVertex {
    pub magnitude: f32,
    pub index: u32,
    pub group_id: u32,
}

// SAFETY: `#[repr(C)]` with the field order the C++ `TorqueGroup` in
// data_records.hpp mirrors, four `unsigned`s and nothing else.
unsafe impl ppf_cts_compute::Pod for TorqueGroup {}

// SAFETY: `#[repr(C)]` with the field order the C++ `TorqueVertex` in
// data_records.hpp mirrors, a float and two `unsigned`s.
unsafe impl ppf_cts_compute::Pod for TorqueVertex {}

/// One torque group's frame: the mass-weighted centroid, the principal axis the
/// commanded torque turns about, and the reciprocal of the members' summed
/// squared perpendicular radius.
///
/// DEVICE-ONLY AND WRITTEN ONLY BY `torque_group_frame`, so it carries no serde
/// derives: it is an intermediate between two dispatches of one step, never a
/// scene input and never checkpointed. `TorqueGroupResult` in
/// `data_records.hpp` gives it the same three fields under the same name.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TorqueGroupResult {
    pub center: Vec3f,
    pub axis: Vec3f,
    pub inv_r_perp_sq_sum: f32,
}

// SAFETY: `#[repr(C)]` over two float triples and a float, which is what the
// C++ `TorqueGroupResult` in data_records.hpp holds in that order.
unsafe impl ppf_cts_compute::Pod for TorqueGroupResult {}

/// One PDRD body's fitted rigid state.
///
/// DEVICE-ONLY, written by the polar fit and read by everything downstream, so
/// it carries no serde derives. `PdrdRigidState` in
/// `energy/model/pdrd_rigid.kernel.cpp` holds the same fields in the same
/// order.
///
/// THE CENTROID IS STORED RELATIVE TO AN ANCHOR, NEVER AS AN ABSOLUTE
/// POSITION. A body's centroid is a position, so carrying it absolutely would
/// spend the coordinate's precision on its distance from the origin before any
/// of the algebra below it runs. The fit accumulates `(x - anchor)`
/// differences and the anchor is added back only where a world position is
/// wanted.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PdrdRigidState {
    pub anchor: [f32; 3],
    pub x: [f32; 3],
    /// Column-major, a `Mat3x3f`'s own layout.
    pub rotation: [f32; 9],
    pub reference_inertia: [f32; 9],
    pub mass_total: f32,
    pub count: u32,
}

// SAFETY: `#[repr(C)]` over six floats, two nines and two more
// scalars, which is the C++ record's own layout.
unsafe impl ppf_cts_compute::Pod for PdrdRigidState {}

/// One locked group's frame: the constraint rows, the reduction that inverts
/// them, and the right-hand side they are driven to.
///
/// DEVICE-ONLY, written by the frame passes and read by every row that projects
/// or refines, so it carries no serde derives. The C++ `LockFrame` in
/// `solver/translation_lock_math.hpp` holds the same fields in the same order
/// and the entry declarations spell its width as `[[seam::pod(292)]]`, which is
/// what the assertion beside this type checks.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LockFrame {
    pub translation_basis0: Vec3f,
    pub translation_basis1: Vec3f,
    pub translation_basis2: Vec3f,
    pub rotation_basis0: Vec3f,
    pub rotation_basis1: Vec3f,
    pub rotation_basis2: Vec3f,
    pub com_relative: Vec3f,
    pub inv_inertia: Mat3x3f,
    /// COLUMN-MAJOR thirty-six floats, the layout `Mat6x6f` has on the C++
    /// side. Floats rather than a matrix type because the host arithmetic that
    /// fills it lives in `driver::lock_math`, which spells its own.
    pub gram_pinv: [f32; 36],
    pub rhs: [f32; 6],
    pub row_mask: u32,
}

// BY HAND, because `derive(Default)` stops at a 32-element array and `gram_pinv`
// holds 36. The value is the all-zero frame the derive would have produced, and
// EVERY member is zeroed rather than only the ones a mode writes: `build_row_bases`
// assigns the bases conditionally, and the host tangent check forms all three
// rotation coefficients before consulting the mask, so an unwritten basis would put
// an inf or a NaN into a quantity that is merely discarded rather than one never
// computed.
impl Default for LockFrame {
    fn default() -> Self {
        Self {
            translation_basis0: Vec3f::zeros(),
            translation_basis1: Vec3f::zeros(),
            translation_basis2: Vec3f::zeros(),
            rotation_basis0: Vec3f::zeros(),
            rotation_basis1: Vec3f::zeros(),
            rotation_basis2: Vec3f::zeros(),
            com_relative: Vec3f::zeros(),
            inv_inertia: Mat3x3f::zeros(),
            gram_pinv: [0.0; 36],
            rhs: [0.0; 6],
            row_mask: 0,
        }
    }
}

// SAFETY: `#[repr(C)]` over seven float triples (84), a 3x3 (36), thirty-six
// floats (144), six floats (24) and an unsigned (4), which is the C++
// `LockFrame`'s own layout and totals 292.
unsafe impl ppf_cts_compute::Pod for LockFrame {}

const _: () = assert!(
    core::mem::size_of::<LockFrame>() == 292,
    "LockFrame must stay 292 bytes: the lock entries declare it [[seam::pod(292)]]"
);

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Stitch {
    pub index: Vec6u,
    pub weight: Vec6f,
    pub stiffness: f32,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
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
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct Floor {
    pub ground: Vec3f,
    pub ghat: f32,
    pub friction: f32,
    pub thickness: f32,
    pub up: Vec3f,
    pub kinematic: bool,
}

// SAFETY: both analytic colliders are `#[repr(C)]` in the field order
// `data_records.hpp` states, and every field is a float triple, a scalar or a
// `bool` the shared bodies read as 0 or 1. The constraint entry points read
// these exact bytes as `Sphere` and `Floor`, so the two spellings have to
// agree field for field; this marker records that they do and widens nothing
// else.
unsafe impl ppf_cts_compute::Pod for Sphere {}
unsafe impl ppf_cts_compute::Pod for Floor {}

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

/// Per-frame material tables, uploaded before `advance` exactly as
/// `RestShapeUpdate` is. Only the face table varies today; the hinge, edge and
/// vertex tables are derived from the faces at build and animating them needs
/// that derivation re-run, which `FACE_ONLY_ANIM_KEYS` refuses until it exists.
///
/// `#[repr(C)]` because it crosses the `extern "C"` `update_material_params`
/// entry point as a raw pointer: the caller in `backend.rs` and the definition
/// in `driver::update_material_params` have to read the same bytes.
#[repr(C)]
pub struct MaterialParamUpdate {
    pub face: CVec<FaceParam>,
    /// Derived from the faces: a vertex or edge averages its neighbors and a
    /// hinge averages its two faces, so a frame that moves the faces moves
    /// these too. Uploading the faces alone would animate the membrane while
    /// the hinges held their build-time stiffness.
    pub vertex: CVec<VertexParam>,
    pub edge: CVec<EdgeParam>,
    pub hinge: CVec<HingeParam>,
}

/// Per-frame replacement of the inverse rest matrices, uploaded to the device
/// next to `update_constraint` when a time-varying rest shape is streamed (the
/// `rest_vert_schedule` path). `driver::rest_shape::apply` copies them into
/// `DataSet::inv_rest2x2` / `inv_rest3x3`, which the elastic kernels re-read
/// each Newton iteration. Field order must mirror `RestShapeUpdate` in
/// `data.hpp` (repr(C) ABI).
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
    /// Float image of `time` for device code. MSL has no `double`; the Metal
    /// mirror retains the f64 slot as padding and reads this field instead.
    pub time_f32: f32,
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
    pub ccd_eps: f32,
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
    // layout MUST mirror the C++ ParamSet in data.hpp (repr(C) ABI).
    pub precond: PrecondMode,
    // Number of additive Schwarz levels (1 = single-level smoother, 2 =
    // two-level coarse correction). Only consulted when precond == Schwarz.
    // Appended at the tail; field order and byte layout MUST mirror the C++
    // ParamSet in data.hpp (repr(C) ABI).
    pub schwarz_levels: u32,
    // Upper bound on Newton iterations per substep. The loop is otherwise
    // unbounded, so an over-constrained configuration (a prescribed pin driven
    // into geometry that cannot yield) spins forever: the line search clamps
    // the shared toi toward zero to prevent the penetration, that same clamp
    // throttles every other vertex, and the toi never falls below FLT_EPSILON,
    // so the CCD trap never fires. This bound turns that hang into a loud
    // CrashKind::NewtonStall. 0 disables the bound (research only).
    // Appended at the tail; field order and byte layout MUST mirror the C++
    // ParamSet in data.hpp (repr(C) ABI).
    pub max_newton_steps: u32,
    // Diagnostic A/B lever (PPF_DISABLE_PIN_DOF_REMOVAL=1, set host-side in
    // advance()): revert every fix pin from an exact Dirichlet BC back to a
    // barrier penalty. It must reach the device, because BOTH halves have to
    // flip together: `driver::step` stops marking the pin's rows for
    // elimination AND `contact/vertex_constraint.kernel.cpp` must assemble the
    // barrier for it, or the pin would have neither and simply vanish.
    // Appended at the tail; field order and byte layout MUST mirror the C++
    // ParamSet in data.hpp (repr(C) ABI).
    pub disable_pin_dof_removal: bool,
}

#[repr(C)]
#[derive(Serialize, Deserialize, Default)]
pub struct StepResult {
    pub time: f64,
    pub ccd_success: bool,
    pub pcg_success: bool,
    pub intersection_free: bool,
    /// False when the Newton loop hit `max_newton_steps` without reaching an
    /// acceptable step. Field order must mirror `StepResult` in
    /// `data.hpp` (repr(C) ABI).
    pub newton_progress: bool,
    /// False when a prescribed (fix-pinned) vertex's swept path crosses an
    /// analytic collider it cannot yield to.
    pub pin_feasible: bool,
    /// False when a contact pair begins the step already inside the contact
    /// offset (two surfaces start out touching or overlapping), so the
    /// conservative CCD cannot advance from a separated start. Appended at the
    /// tail; field order must mirror `StepResult` in `data.hpp`.
    pub contact_separated: bool,
}

impl StepResult {
    pub fn success(&self) -> bool {
        self.ccd_success
            && self.pcg_success
            && self.intersection_free
            && self.newton_progress
            && self.pin_feasible
            && self.contact_separated
    }
}

// Rust mirror of the C++ ABI capacity. The canonical definition lives in
// data_records.hpp (`#define MAX_INTERSECTION_RECORDS`), which bounds every
// C++-side write; this const MUST stay equal to it. The FFI buffer in
// backend.rs is sized with this value and passed as max_count, so a divergence
// silently under-reports intersection records. Update both files together.
pub const MAX_INTERSECTION_RECORDS: usize = 256;

/// One intersection record copied out through `fetch_intersection_records` for
/// the diagnostics dump. Field order and array sizes (including
/// `positions: [f32; 15]`) must
/// mirror `IntersectionRecord` in `data_records.hpp` (repr(C) ABI).
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

/// One enabled aggregate rigid-mode lock. Translation constrains the physical
/// mass-weighted center of mass, and rotation constrains each Newton
/// correction's best-fit infinitesimal angular increment. The two are
/// independent: either, both, or neither may be enabled on one group.
///
/// Each half is described by a mode beside an axis, and THE MODE CARRIES THE
/// ENABLE BIT. In an axis mode the axis is a unit solver-space direction and a
/// zero axis means that half is off; in an all-axes mode the axis is
/// meaningless and is required to be exactly zero, so a record has one
/// canonical spelling. `builder.rs` asserts that biconditional. Reading
/// enablement off the axis alone is therefore wrong for the all-axes modes,
/// which is why both backends test through their own
/// `translation_lock_enabled` / `rotation_lock_enabled` helpers.
///
/// | mode | rows | meaning |
/// | ---- | ---- | ------- |
/// | `TRANSLATION_LOCK_AXIS` | 2 | the center of mass stays on the line through its initial value along `axis` |
/// | `TRANSLATION_LOCK_ALL` | 3 | the center of mass stays at its initial point |
/// | `ROTATION_LOCK_ALLOW_ONLY` | 2 | rotation about `rotation_axis` is the only angular freedom |
/// | `ROTATION_LOCK_PROHIBIT_AXIS` | 1 | rotation about `rotation_axis` is forbidden, the perpendicular plane stays free |
/// | `ROTATION_LOCK_ALL` | 3 | there is no net rotation about any axis |
///
/// Zero is the off value for both modes, so a session written without the lock
/// bins decodes to an unlocked scene rather than a malformed one.
///
/// `pdrd_body_index` is 0 for a deformable/SAND group and otherwise the
/// 1-based PDRD body that owns this lock. `anchor` is the locked initial
/// position: every current relative coordinate is formed as a difference
/// against it, so the projector works on small local offsets instead of
/// differencing two absolute coordinates whose leading digits cancel. Field
/// order must mirror `TranslationLock` in `data_records.hpp` (repr(C) ABI).
pub const TRANSLATION_LOCK_AXIS: u32 = 0;
pub const TRANSLATION_LOCK_ALL: u32 = 1;
pub const ROTATION_LOCK_ALLOW_ONLY: u32 = 0;
pub const ROTATION_LOCK_PROHIBIT_AXIS: u32 = 1;
pub const ROTATION_LOCK_ALL: u32 = 2;

#[repr(C)]
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct TranslationLock {
    pub axis: Vec3f,
    pub translation_mode: u32,
    pub total_mass: f32,
    pub pdrd_body_index: u32,
    pub dmap_index: u32,
    pub rotation_axis: Vec3f,
    pub rotation_mode: u32,
    pub anchor: Vec3f,
}

// SAFETY: `#[repr(C)]` with the field order the C++ `TranslationLock` in
// data_records.hpp mirrors: two float triples, two floats' worth of scalars,
// three unsigneds and a third float triple, no padding beyond what both sides
// share.
unsafe impl ppf_cts_compute::Pod for TranslationLock {}

// The record is memcpy'd raw into device memory, and nothing else compares this
// layout against its C++ mirror. A field added on one side only would
// reinterpret the fields after it and still compile clean on both sides.
// `data_records.hpp` asserts the same number.
const _: () = assert!(
    std::mem::size_of::<TranslationLock>() == 56,
    "TranslationLock must stay layout-identical to its mirror in data_records.hpp"
);

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
    /// the grain's floor/sphere contacts): `grain_a` is the SPD angular block `A`,
    /// `grain_b` the translation<->rotation coupling `B`, `grain_grot` the
    /// rotational gradient. (The C++ DataSet spells `A`/`B` uppercase per the math
    /// convention for matrices; here they are the same fields in the same order.)
    pub grain_a: CVec<Mat3x3f>,
    pub grain_b: CVec<Mat3x3f>,
    pub grain_grot: CVec<Vec3f>,
    /// Slot-replay assembly tables (item 3). Each holds the flat FixedCSRMat
    /// value-slot index of every 3x3 block the corresponding element writes,
    /// row-major (ii*N + jj), with 0xFFFFFFFF sentinels for lower-triangle
    /// (push() no-op) blocks. Empty when PPF_SLOT_REPLAY=0 (kernels fall back to
    /// push()). Field order and byte layout MUST mirror the Vec<unsigned> tail
    /// of the C++ DataSet in data.hpp (repr(C) ABI); tail-append only.
    pub tet_hess_slots: CVec<u32>,
    pub face_hess_slots: CVec<u32>,
    pub edge_hess_slots: CVec<u32>,
    pub hinge_hess_slots: CVec<u32>,
    pub rod_bend_hess_slots: CVec<u32>,
    pub stitch_hess_slots: CVec<u32>,
    /// Compact enabled lock records. Empty means no translation lock is active.
    /// `translation_lock_index` maps global vertices to these entries, using
    /// `u32::MAX` for a vertex outside every locked physical mass group.
    pub translation_lock: CVec<TranslationLock>,
    pub translation_lock_index: CVec<u32>,
    /// Initial positions for the locked-vertex COM reference. This is stored
    /// separately from `vertex.curr`, which advances during simulation, so the
    /// solver always forms `(current - initial)` against the step-zero pose
    /// rather than against a reference that moves with it.
    /// Empty when `translation_lock` is empty.
    pub translation_lock_initial: CVec<Vec3f>,
    /// Statistics object index for each dynamic and static collision-mesh
    /// vertex. `u32::MAX` marks a vertex outside the statistics manifest.
    /// The contact kernels use these tables to attribute every accepted
    /// constraint to both participating objects.
    pub statistics_object_index: CVec<u32>,
    pub statistics_static_object_index: CVec<u32>,
    /// One accepted-contact count per statistics object. Cleared before each
    /// Newton assembly, atomically accumulated by contact kernels, and fetched
    /// with the output pose.
    pub statistics_contact_count: CVec<u32>,
}
