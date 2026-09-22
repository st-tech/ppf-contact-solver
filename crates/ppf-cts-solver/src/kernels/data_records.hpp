// File: data_records.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE FIXED-SIZE RECORDS THE DATASET IS MADE OF, AND NOTHING THAT OWNS STORAGE.
//
// Every declaration below is a plain aggregate or an enum: a per-element
// parameter, a per-element property, a constraint pair, a bounding box, a
// lock. Not one of them holds a pointer, a length, or a view, so not one of
// them needs an allocator, a container, or a standard library to be declared.
//
// WHY IT IS A FILE RATHER THAN PART OF data.hpp.
// data.hpp is what a CUDA or host translation unit includes to get the whole
// shared vocabulary, and it reaches `vec/vec.hpp` for the array views the
// dataset is built out of. Those views spell a raw `T *`, and MSL has no
// default address space, so every pointer and every reference in a declaration
// must carry one: the Metal shader compiler refuses that header outright, and
// with it everything declared beside it. `linalg/type_aliases.hpp` was
// split out of the same file for the same reason and states the same argument;
// this is the second half of that boundary, the records rather than the
// spellings.
//
// WHO NEEDS IT SEPARATELY. Two consumers, and they must read the same bytes.
// The shipped Metal shader splices this file as a segment, so a kernel body
// that names `Barrier` or `FixPair` finds it declared. A GENERATED ENTRY POINT
// compiled on its own (ppf-cts-compute/metal/Makefile's `entry-check`) is one
// neutral body plus the argument record around it, and both may name any type
// here. A type declared only in data.hpp cannot appear in a generated entry
// point at all: the rendering compiles on CUDA and on the host, and the shader
// compiler has never been given a declaration for it, so the MSL rendering
// fails with `unknown type name` pointing at a line that is correct.
//
// data.hpp includes this file, so a CUDA or host translation unit reaches these
// records by including data.hpp alone and the two cannot drift: there is one
// declaration of each record and every backend reads it.

#ifndef DATA_RECORDS_HPP
#define DATA_RECORDS_HPP

// The `linalg::SMat` spellings these records are built out of (`Vec3f`,
// `Vec3f`, `Mat3x3f`, `SVecu`), and the intersection-policy constants
// `VertexProp` documents its own fields against. Both quoted, so the run-time
// assembler neutralizes them as it splices and the shipped shader keeps taking
// them from its segment order.
#include "contact/intersect_policy.hpp"
#include "linalg/type_aliases.hpp"

enum class Model { ARAP, StVK, BaraffWitkin, SNHk, Pdrd };
enum class Barrier { Cubic, Quad, Log };
enum class FrictionMode { Min, Max, Mean };
// Linear-solve preconditioner selector. Scoped enum so it is a 4-byte int,
// matching the Rust #[repr(C)] PrecondMode mirror in data.rs. Variant order MUST
// stay in step with the Rust enum.
enum class PrecondMode { BlockJacobi, Schwarz };

struct VertexParam {
    float ghat;
    float offset;
    float friction;
};

struct EdgeParam {
    float stiffness;
    float bend;
    float ghat;
    float offset;
    float friction;
    float strainlimit;
    float plasticity;
    float plasticity_threshold;
    bool bend_rest_from_geometry;
    // Rayleigh damping coefficients (per object, replicated per element).
    // deform_damping scales the stretch tangent stiffness; bend_damping the
    // rod-bending tangent stiffness. Appended at the tail; field order and byte
    // layout MUST mirror the Rust EdgeParam in data.rs (repr(C) ABI).
    float deform_damping;
    float bend_damping;
};

struct FaceParam {
    Model model;
    float mu;
    float lambda;
    float friction;
    float ghat;
    float offset;
    float bend;
    float strainlimit;
    float shrink_x;
    float shrink_y;
    float pressure;
    float plasticity;
    float plasticity_threshold;
    float bend_plasticity;
    float bend_plasticity_threshold;
    bool bend_rest_from_geometry;
    // Rayleigh damping coefficients (per object, replicated per face).
    // deform_damping scales the membrane tangent stiffness; bend_damping is
    // carried to the shell hinges (area-averaged in builder.rs). Tail-appended;
    // MUST mirror the Rust FaceParam in data.rs (repr(C) ABI).
    float deform_damping;
    float bend_damping;
    // Directional shell bending stiffnesses, ADDED to the isotropic `bend` at
    // an orientation-dependent weight (shell_bend_directional). Carried to
    // the hinges the same way bend_damping is. Both 0.0 means isotropic. MUST
    // mirror the Rust FaceParam (repr(C) ABI).
    float bend_warp;
    float bend_weft;
};

struct HingeParam {
    float bend;
    float ghat;
    float offset;
    float plasticity;
    float plasticity_threshold;
    // Rayleigh bending damping, area-averaged from the adjacent faces'
    // bend_damping. Tail-appended; MUST mirror the Rust HingeParam (repr(C)).
    float bend_damping;
    // Directional bending stiffnesses added to `bend`, area-averaged from the
    // adjacent faces alongside it. Paired with HingeProp::uv_edge_sin2 in
    // shell_bend_directional. Both 0.0 leaves the stiffness at `bend`. MUST
    // mirror the Rust HingeParam (repr(C)).
    float bend_warp;
    float bend_weft;
};

struct TetParam {
    Model model;
    float mu;
    float lambda;
    float shrink;
    float plasticity;
    float plasticity_threshold;
    // Rayleigh deformation damping, scales the solid tangent stiffness.
    // Tail-appended; MUST mirror the Rust TetParam in data.rs (repr(C) ABI).
    float deform_damping;
};

struct VertexProp {
    float area;
    float volume;
    float mass;
    float rest_bend_angle;
    unsigned fix_index;
    unsigned pull_index;
    unsigned param_index;
    // 1-based PDRD body id (0 = not a member of any PDRD body).
    unsigned pdrd_body_index;
    // Vertex belongs to a STATIC collider. A collider's shape is authored and
    // driven, so a contact between two collider elements is not the solver's to
    // resolve: neither side can yield to relieve it, and rigged colliders
    // routinely ship self-tangled (layered eyelash / eyeball / mouth geometry,
    // an arm resting inside a torso). Pairs with this set on BOTH sides are
    // skipped for contact and for intersection reporting, whether the two sides
    // belong to the same collider or to two different ones. An exactly pinned
    // collider was already excluded by `either_dyn`; this keeps that exclusion
    // when the collider is held by springs instead and its vertices become
    // free. Field order must mirror `VertexProp` in data.rs (repr(C) ABI).
    bool collider;
    // Source-object identity, the only thing that separates a SELF-
    // intersection from an INTER-OBJECT one. `param_index` cannot serve:
    // identical materials deduplicate to one entry, so two objects share it.
    // Read off an element's FIRST vertex, the convention `pdrd_body_index` and
    // `collider` already use, which holds because a vertex belongs to exactly
    // one object and therefore so does every element built on it.
    // NO_OBJECT_INDEX when the session directory carries no object_vert.bin.
    unsigned object_index;
    // Source-GROUP identity, which separates an INTER-GROUP pair from two
    // objects of one group. A group is whatever the frontend was told
    // (`Object.group`; the add-on's decoder names each object's group), and
    // every object told nothing shares one default group. NO_GROUP_INDEX for
    // the static collision mesh, which therefore counts as another group from
    // every object.
    unsigned group_index;
    // This vertex's object's intersection allowances, as
    // INTERSECT_ALLOW_SELF | INTERSECT_ALLOW_INTER_OBJECT |
    // INTERSECT_ALLOW_INTER_GROUP. Resolved per OBJECT
    // by the frontend rather than per element, which is both the granularity
    // the material param has and the only one defined for a faceless SAND
    // grain (no incident element to read a material from).
    unsigned char intersect_policy;
    // Every pin covering this vertex asked for its intersections to be
    // tolerated. False for an unpinned vertex, so an element earns the
    // exemption only when ALL of its vertices carry this. Latched at scene
    // build, exactly like FaceProp::fixed.
    bool pin_allow_intersection;
};

// NO_OBJECT_INDEX and the INTERSECT_ALLOW_* bits live in
// contact/intersect_policy.hpp, included above, beside the
// rule that reads them.

struct EdgeProp {
    float length;
    float initial_length;
    float mass;
    bool fixed;
    unsigned param_index;
    // All of this edge's vertices are pinned by pins that asked for their
    // intersections to be tolerated. Precomputed from
    // VertexProp::pin_allow_intersection the way `fixed` is precomputed from
    // fix_index. Field order must mirror Rust EdgeProp in data.rs.
    bool pin_allow_intersection;
};

struct FaceProp {
    float area;
    float mass;
    bool fixed;
    // Excluded from the elastic/strain energy this frame because its streamed
    // rest shape is near-singular (set per frame by update_rest_shape). Kept
    // separate from `fixed` (kinematic pinning) so the two never alias. Field
    // order must mirror Rust FaceProp in data.rs (repr(C) ABI).
    bool rest_excluded;
    // Belongs to a STATIC collider, so it carries no elastic energy. A
    // collider's shape is held by its pins, not by stiffness of its own: while
    // the pins were exact its DOF were removed and `fixed` already suppressed
    // this term, but a spring-held collider's vertices are free and the term
    // would otherwise come alive with whatever material the defaults supplied.
    // Kept separate from both `fixed` (kinematic pinning) and `rest_excluded`
    // (owned per frame by update_rest_shape) so none of the three alias. Field
    // order must mirror Rust FaceProp in data.rs (repr(C) ABI).
    bool collider;
    unsigned param_index;
    // All of this face's vertices are pinned by pins that asked for their
    // intersections to be tolerated. See EdgeProp::pin_allow_intersection.
    bool pin_allow_intersection;
};

struct HingeProp {
    float length;
    // Combined rest area of the two triangles incident to the hinge edge
    // (A1 + A2); paired with `length` for the resolution-independent bending
    // coefficient |e|^2 / area (see embed_hinge_force_hessian).
    float area;
    float rest_angle;
    // sin^2(psi) for psi the angle between this hinge's shared edge and the
    // UV X (warp) axis, in [0, 1], or -1.0 when the mesh carries no UV and the
    // hinge is therefore isotropic. A hinge bends about its shared edge, so
    // the surface curves ACROSS it: an edge along warp (psi = 0, sin^2 = 0)
    // resists with the weft stiffness, which is why
    // shell_bend_directional pairs sin^2 with warp and cos^2 with weft.
    // Stored squared so the kernel needs no trigonometry (sinf/cosf would also
    // drag FP64 argument reduction into the device binary). Field order must
    // mirror Rust HingeProp in data.rs.
    float uv_edge_sin2;
    bool fixed;
    // Belongs to a STATIC collider, so it carries no bending energy. See
    // FaceProp::collider. Field order must mirror Rust HingeProp in data.rs.
    bool collider;
    unsigned param_index;
};

// The per-hinge bending stiffness, and the orientation-dependent weight that
// mixes `bend` with `bend_warp` and `bend_weft`, live in
// energy/model/shell_bend_stiffness.kernel.cpp as shell_bend_stiffness and
// shell_bend_directional. They sit on the CUDA / MSL seam because both
// backends have to form the same scalar before either evaluates a hinge.

struct TetProp {
    float mass;
    float volume;
    bool fixed;
    // See FaceProp::rest_excluded. Field order must mirror Rust TetProp.
    bool rest_excluded;
    unsigned param_index;
};

// Mirror of Rust `PdrdBodyProp`. Per Painless Differentiable Rotation Dynamics body: a
// contiguous vertex range plus precomputed rest-shape moments
// (centroid, inverse rest Gram matrix) and the body's volume. The per-iterate
// best-fit rigid transform (rotation + translation) is solved from these.
struct PdrdBodyProp {
    Vec3f rest_centroid;
    Mat3x3f rest_gram_inv;
    float volume;
    unsigned vertex_start;
    unsigned vertex_count;
    // Per-vertex mass for the body's vertices; build-time override.
    // Not read by the kernel.
    float mass_per_vertex;
    // Joint / DOF-filtering: 0 = free (full 6-DOF rigid body), 1 = hinge
    // (reduced DOF filtered to lock translation and restrict rotation to
    // the single world axle `joint_axis` through the body centroid; see
    // project_body_dofs_kernel in pdrd_lock_projector.hpp). `joint_pin`
    // records the axle anchor (initial centroid). Field order must mirror Rust
    // `PdrdBodyProp` in data.rs (repr(C) ABI).
    unsigned joint_mode;
    Vec3f joint_axis;
    Vec3f joint_pin;
};

template <unsigned R, unsigned C> struct Svd {
    SMatf<R, C> U;
    SVecf<C> S;
    SMatf<C, C> Vt;
};

using Svd3x2 = Svd<3, 2>;
using Svd3x3 = Svd<3, 3>;

struct FixPair {
    // Where the pin's prescribed path puts this vertex at the END of the
    // step (the time the host aimed the step at).
    Vec3f position;
    // Displacement the pin travels over this step: `position` minus the same
    // path evaluated at the step's START time. The CCD line search can
    // truncate a step to a fraction `toi` of its span, and a kinematic pin
    // must then stop at that same fraction of the path it was scheduled to
    // travel, or it outruns the clock (see `main/rewind_fix.kernel.cpp`).
    // Held as the DELTA rather than as a second absolute position: the two
    // endpoints agree in their leading digits, so the difference is what fp32
    // resolves well, and scaling it by `toi` keeps the pin's own coordinate out
    // of the product. Zero for a static pin, which never moves.
    Vec3f step_delta;
    float ghat;
    unsigned index;
    bool kinematic;
    // The pin that placed this vertex asked for its intersections to be
    // tolerated. Consumed once, at scene build, to latch
    // VertexProp::pin_allow_intersection; the per-step constraint rebuild
    // carries it along so the two constructions cannot disagree.
    bool allow_intersection;
    // NOTE: every fix pin is an exact Dirichlet BC (the Dirichlet pass
    // eliminates its DOF), so there is no per-pin stiffness to scale: there is
    // no penalty force left.
    // `ghat` and `kinematic` survive only for the PDRD anchor, the one pin still
    // held by the barrier (a vertex inside a rigid body owns no per-vertex DOF).
    // Field order must mirror Rust FixPair in data.rs (repr(C) ABI).
};

struct PullPair {
    Vec3f position;
    float weight;
    unsigned index;
    // See FixPair::allow_intersection. A pull pin holds its vertex only to the
    // extent of its own force, so an intersection under one is often not
    // something the pin is responsible for; that is the case issue #138
    // singles out. Field order must mirror Rust PullPair in data.rs.
    bool allow_intersection;
};

struct TorqueGroup {
    unsigned axis_component;
    unsigned vertex_start;
    unsigned vertex_count;
    unsigned hint_vertex;
};

struct TorqueVertex {
    float magnitude;
    unsigned index;
    unsigned group_id;
};

struct TorqueGroupResult {
    Vec3f center;
    Vec3f axis;
    float inv_r_perp_sq_sum;
};

struct Stitch {
    Vec6u index;
    Vec6f weight;
    float stiffness;
};

struct Sphere {
    Vec3f center;
    float ghat;
    float friction;
    float radius;
    float thickness;
    bool bowl;
    bool reverse;
    bool kinematic;
};

struct Floor {
    Vec3f ground;
    float ghat;
    float friction;
    float thickness;
    Vec3f up;
    bool kinematic;
};

enum TranslationLockMode : unsigned {
    // The center of mass stays on the line through its initial value along
    // `axis` (two rows).
    TRANSLATION_LOCK_AXIS = 0u,
    // The center of mass stays at its initial point (three rows).
    TRANSLATION_LOCK_ALL = 1u,
};

enum RotationLockMode : unsigned {
    // Rotation about `rotation_axis` is the only angular freedom (two rows).
    ROTATION_LOCK_ALLOW_ONLY = 0u,
    // Rotation about `rotation_axis` is forbidden, the perpendicular plane
    // stays free (one row).
    ROTATION_LOCK_PROHIBIT_AXIS = 1u,
    // There is no net rotation about any axis (three rows).
    ROTATION_LOCK_ALL = 2u,
};

// Mirror of Rust TranslationLock. Each half is a mode beside an axis, and THE
// MODE CARRIES THE ENABLE BIT. In an axis mode the axis is a normalized
// solver-space direction and a zero axis means that half is off; in an
// all-axes mode the axis is meaningless and is exactly zero, asserted on the
// Rust side so a record has one canonical spelling. Testing `axis != 0`
// directly is therefore wrong for the all-axes modes: ask through
// `translation_lock_enabled()` / `rotation_lock_enabled()` in
// `solver/translation_lock_math.hpp`, which is where the lock algebra lives
// and where a function may carry the address-space spelling this file's plain
// aggregates do not need. Zero is
// the off value for both modes, so a session written without the lock bins is
// an unlocked scene. `anchor` lets the rotation projector form every relative
// coordinate as a difference of two positions before it enters a product, so
// the group's distance from the origin scales nothing. Field order must mirror
// Rust data.rs (repr(C) ABI).
struct TranslationLock {
    Vec3f axis;
    unsigned translation_mode;
    float total_mass;
    unsigned pdrd_body_index;
    unsigned dmap_index;
    Vec3f rotation_axis;
    unsigned rotation_mode;
    Vec3f anchor;
};

// The record is handed to the device as raw bytes and nothing else compares the
// two layouts. A field added on one side only would reinterpret total_mass,
// pdrd_body_index or anchor as garbage and still compile clean on both sides.
// `data.rs` asserts the same number.
static_assert(sizeof(TranslationLock) == 56,
              "TranslationLock must stay layout-identical to its repr(C) Rust "
              "mirror in data.rs; update both or neither");

/********** CUSTOM TYPES **********/

struct alignas(32) AABB {
    // alignas(32) pads the 28-byte payload to a 32-byte stride so every
    // random aabb[index] load in BVH traversal is exactly one 32-byte sector
    // (at 28 bytes, ~75% of loads straddled two sectors). Device-only struct
    // (not part of the Rust wire ABI).
    Vec3f min;
    Vec3f max;
    bool active;
};

template <unsigned N> struct Proximity {
    SVecu<N> index;
    SVecf<N> value;
};

// WHAT A CCD SWEEP LEAVES BEHIND WHEN A PAIR BEGAN THE STEP ALREADY INSIDE ITS
// CONTACT OFFSET, one record per QUERY primitive.
//
// A FIRST-THREAD-WINS LATCH IS NOT AVAILABLE HERE, and would be the weaker
// property if it were. Latching one global report under a compare-and-swap
// needs both a `compute::` compare-and-swap, which the seam does not carry, and
// a mutable device global, which Metal has not; that is also why
// `accd::OverlapInfo` is an out-parameter rather than a global. So the report is
// PER QUERY, written by the one thread that owns the slot and needing no atomic
// at all, and the host takes the LOWEST-INDEXED flagged slot, which names the
// same pair on every run, where a first-thread-wins latch would name whichever
// thread arrived first.
//
// It lives here rather than in data.hpp because the Metal shader compiler
// cannot read that header in any position, and a generated entry point resolves
// this pointee's size at compile time in every rendering.
struct CcdOverlapRecord {
    // Whether this slot was written at all. NOT inferable from a returned time
    // of zero: ACCD's probe cap returns `lower_t`, which is zero when the very
    // first advance underflows, and that path writes no report. Reading the
    // time instead reports a pair that did not begin the step overlapping, with
    // a `d2` and an `offset` that are the initializer rather than measurements.
    unsigned flagged;
    // Which of the four sweeps spoke, for the message: 0 point-triangle,
    // 1 point-edge, 2 point-point, 3 edge-edge.
    unsigned kind;
    // The two elements, in the index spaces `ccd_record_overlap`
    // (`contact/ccd_sweep.kernel.cpp`) is handed them in; `kind` above says
    // which sweep, and with it which space each index belongs to.
    unsigned elem0;
    unsigned elem1;
    // The squared start distance and the offset, both in the RESCALED units of
    // that pair's own sweep frame. A pair flagged for a COLLAPSED frame records
    // a distance of exactly zero and leaves the offset at the -1.0f sentinel.
    float d2;
    float offset;
};

#define MAX_INTERSECTION_RECORDS 256

struct IntersectionRecord {
    unsigned type;       // 0=face-edge, 1=edge-edge, 2=collision-mesh, 3=point-point
    unsigned elem0;      // first element index (face, edge, or vertex)
    unsigned elem1;      // second element index (edge or vertex)
    unsigned num_verts0; // vertex count for first element (1, 2, or 3)
    unsigned num_verts1; // vertex count for second element (1, 2, or 3)
    float positions[15]; // up to 5 vertices x 3 (x,y,z), packed: elem0 then elem1
};

#endif // DATA_RECORDS_HPP
