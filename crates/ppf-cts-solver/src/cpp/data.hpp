// File: data.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DATA_HPP
#define DATA_HPP

#include "linalg/smat.hpp"
#include "vec/vec.hpp"

using linalg::Map;
template <class T, unsigned N> using SVec = linalg::SVec<T, N>;
template <unsigned N> using SVecf = SVec<float, N>;
template <unsigned N> using SVecu = SVec<unsigned, N>;

template <class T> using Vec1 = SVec<T, 1>;
template <class T> using Vec2 = SVec<T, 2>;
template <class T> using Vec3 = SVec<T, 3>;
template <class T> using Vec4 = SVec<T, 4>;
template <class T> using Vec6 = SVec<T, 6>;
template <class T> using Vec9 = SVec<T, 9>;
template <class T> using Vec12 = SVec<T, 12>;

using Vec1f = Vec1<float>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Vec6f = Vec6<float>;
using Vec9f = Vec9<float>;
using Vec12f = Vec12<float>;

using Vec1u = Vec1<unsigned>;
using Vec2u = Vec2<unsigned>;
using Vec3u = Vec3<unsigned>;
using Vec4u = Vec4<unsigned>;
using Vec6u = Vec6<unsigned>;


template <class T, unsigned R, unsigned C>
using SMat = linalg::SMat<T, R, C>;
template <unsigned R, unsigned C> using SMatf = linalg::SMat<float, R, C>;

template <class T> using Mat3x2 = SMat<T, 3, 2>;
template <class T> using Mat2x2 = SMat<T, 2, 2>;
template <class T> using Mat2x3 = SMat<T, 2, 3>;
template <class T> using Mat3x3 = SMat<T, 3, 3>;
template <class T> using Mat3x4 = SMat<T, 3, 4>;
template <class T> using Mat3x5 = SMat<T, 3, 5>;
template <class T> using Mat3x6 = SMat<T, 3, 6>;
template <class T> using Mat4x3 = SMat<T, 4, 3>;
template <class T> using Mat4x4 = SMat<T, 4, 4>;
template <class T> using Mat3x9 = SMat<T, 3, 9>;
template <class T> using Mat6x6 = SMat<T, 6, 6>;
template <class T> using Mat6x9 = SMat<T, 6, 9>;
template <class T> using Mat9x9 = SMat<T, 9, 9>;
template <class T> using Mat9x12 = SMat<T, 9, 12>;
template <class T> using Mat12x12 = SMat<T, 12, 12>;

using Mat2x3f = Mat2x3<float>;
using Mat3x2f = Mat3x2<float>;
using Mat2x2f = Mat2x2<float>;
using Mat3x3f = Mat3x3<float>;
using Mat3x4f = Mat3x4<float>;
using Mat3x5f = Mat3x5<float>;
using Mat3x6f = Mat3x6<float>;
using Mat4x3f = Mat4x3<float>;
using Mat4x4f = Mat4x4<float>;
using Mat3x9f = Mat3x9<float>;
using Mat6x6f = Mat6x6<float>;
using Mat6x9f = Mat6x9<float>;
using Mat9x9f = Mat9x9<float>;
using Mat9x12f = Mat9x12<float>;
using Mat12x12f = Mat12x12<float>;


enum class Model { ARAP, StVK, BaraffWitkin, SNHk, Pdrd };
enum class Barrier { Cubic, Quad, Log };
enum class FrictionMode { Min, Max, Mean };
// Linear-solve preconditioner selector. Scoped enum so it is a 4-byte int,
// matching the Rust #[repr(C)] PrecondMode mirror in data.rs. Variant order MUST
// stay in step with the Rust enum.
enum class PrecondMode { BlockJacobi, Schwarz };

struct VertexNeighbor {
    VecVec<unsigned> face;
    VecVec<unsigned> hinge;
    VecVec<unsigned> edge;
    VecVec<unsigned> rod;
};

struct HingeNeighbor {
    VecVec<unsigned> face;
};

struct EdgeNeighbor {
    VecVec<unsigned> face;
};

struct MeshInfo {
    struct {
        Vec<Vec3u> face;
        Vec<Vec4u> hinge;
        Vec<Vec2u> edge;
        Vec<Vec4u> tet;
    } mesh;
    struct {
        VertexNeighbor vertex;
        HingeNeighbor hinge;
        EdgeNeighbor edge;
    } neighbor;
    struct {
        Vec<char> face;
        Vec<char> vertex;
        Vec<char> hinge;
    } type;
};

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
};

struct EdgeProp {
    float length;
    float initial_length;
    float mass;
    bool fixed;
    unsigned param_index;
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
    unsigned param_index;
};

struct HingeProp {
    float length;
    // Combined rest area of the two triangles incident to the hinge edge
    // (A1 + A2); paired with `length` for the resolution-independent bending
    // coefficient |e|^2 / area (see embed_hinge_force_hessian).
    float area;
    float rest_angle;
    bool fixed;
    unsigned param_index;
};

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
    // project_body_dofs_kernel in pdrd_rigid.hpp). `joint_pin` records the
    // axle anchor (initial centroid). Field order must mirror Rust
    // `PdrdBodyProp` in data.rs (repr(C) ABI).
    unsigned joint_mode;
    Vec3f joint_axis;
    Vec3f joint_pin;
};

struct PropSet {
    Vec<VertexProp> vertex;
    Vec<EdgeProp> edge;
    Vec<FaceProp> face;
    Vec<HingeProp> hinge;
    Vec<TetProp> tet;
    Vec<PdrdBodyProp> pdrd_body;
};

struct ParamArrays {
    Vec<VertexParam> vertex;
    Vec<EdgeParam> edge;
    Vec<FaceParam> face;
    Vec<HingeParam> hinge;
    Vec<TetParam> tet;
};

struct BVH {
    Vec<Vec2u> node;
    VecVec<unsigned> level; // Nodes grouped by depth (level[0] = root, level[last] = leaves)
    unsigned root{0};       // Root node index in `node`; query traversal seeds here.
};

struct BVHSet {
    BVH face;
    BVH edge;
    BVH vertex;
};

template <unsigned R, unsigned C> struct Svd {
    SMatf<R, C> U;
    SVecf<C> S;
    SMatf<C, C> Vt;
};

using Svd3x2 = Svd<3, 2>;
using Svd3x3 = Svd<3, 3>;

template <unsigned N> struct DiffTable {
    SVecf<N> deda;
    SMatf<N, N> d2ed2a;
};

using DiffTable2 = DiffTable<2>;
using DiffTable3 = DiffTable<3>;

struct FixPair {
    Vec3f position;
    float ghat;
    unsigned index;
    bool kinematic;
    // Per-pin force scale, applied only for kinematic pins. 1.0 is the
    // default (no change). Field order must mirror Rust FixPair in
    // data.rs (repr(C) ABI).
    float stiffness;
};

struct PullPair {
    Vec3f position;
    float weight;
    unsigned index;
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

struct CollisionMesh {
    Vec<Vec3f> vertex;
    Vec<Vec3u> face;
    Vec<Vec2u> edge;
    struct {
        Vec<VertexProp> vertex;
        Vec<FaceProp> face;
        Vec<EdgeProp> edge;
    } prop;
    struct {
        Vec<VertexParam> vertex;
        Vec<FaceParam> face;
        Vec<EdgeParam> edge;
    } param_arrays;
    struct {
        VertexNeighbor vertex;
        HingeNeighbor hinge;
        EdgeNeighbor edge;
    } neighbor;
};

struct Constraint {
    Vec<FixPair> fix;
    Vec<PullPair> pull;
    Vec<TorqueGroup> torque_groups;
    Vec<TorqueVertex> torque_vertices;
    Vec<Sphere> sphere;
    Vec<Floor> floor;
    Vec<Stitch> stitch;
    CollisionMesh mesh;
};

// Per-frame replacement of the inverse rest matrices for a streamed
// time-varying rest shape. Copied straight into DataSet.inv_rest2x2 /
// inv_rest3x3, which the elastic kernels re-read each Newton iteration.
// Field order must mirror Rust RestShapeUpdate in data.rs (repr(C) ABI).
struct RestShapeUpdate {
    Vec<Mat2x2f> inv_rest2x2;
    Vec<Mat3x3f> inv_rest3x3;
    // Per shell-face / per-tet flag (1 = exclude this element from the elastic
    // and strain energy this frame, OR-ed into its prop `fixed`). Field order
    // must mirror Rust RestShapeUpdate in data.rs (repr(C) ABI).
    Vec<unsigned char> exclude_face;
    Vec<unsigned char> exclude_tet;
};

struct ParamSet {
    double time;
    float air_friction;
    float air_density;
    float constraint_tol;
    float prev_dt;
    float dt;
    float playback;
    unsigned min_newton_steps;
    float target_toi;
    float stitch_length_factor;
    unsigned cg_max_iter;
    float cg_tol;
    float line_search_max_t;
    float ccd_reduction;
    unsigned ccd_max_iter;
    float max_dx;
    float eiganalysis_eps;
    float friction_eps;
    float isotropic_air_friction;
    Vec3f gravity;
    Vec3f wind;
    Barrier barrier;
    FrictionMode friction_mode;
    unsigned csrmat_max_nnz;
    float fix_xz;
    bool disable_contact;
    bool inactive_momentum;
    // Linear-solve preconditioner. Appended at the tail; field order and byte
    // layout MUST mirror the Rust ParamSet in data.rs (repr(C) ABI).
    PrecondMode precond;
    // Number of additive Schwarz levels (1 = single-level smoother, 2 =
    // two-level coarse correction). Only consulted when precond == Schwarz.
    // Appended at the tail; field order and byte layout MUST mirror the Rust
    // ParamSet in data.rs (repr(C) ABI).
    unsigned schwarz_levels;
};

struct StepResult {
    double time;
    bool ccd_success;
    bool pcg_success;
    bool intersection_free;
    bool success() const {
        return ccd_success && pcg_success && intersection_free;
    }
};

struct VertexSet {
    Vec<Vec3f> prev;
    Vec<Vec3f> curr;
};

struct DataSet {
    VertexSet vertex;
    MeshInfo mesh;
    PropSet prop;
    ParamArrays param_arrays;
    Vec<Mat2x2f> inv_rest2x2;
    Vec<Mat3x3f> inv_rest3x3;
    Constraint constraint;
    VecVec<unsigned> fixed_index_table;
    VecVec<Vec2u> transpose_table;
    unsigned rod_count;
    unsigned shell_face_count;
    unsigned surface_vert_count;
    // Flat list of global vertex indices participating in PDRD
    // bodies (parallel to pdrd_rest_centered).
    Vec<unsigned> pdrd_vert_list;
    // Centered rest position ȳₖ per PDRD vertex, same order as
    // pdrd_vert_list.
    Vec<Vec3f> pdrd_rest_centered;
    // SAND grain spin (rolling). Per-vertex angular velocity omega (rad/s),
    // zero for non-grain vertices. The contact-friction torque integrates it
    // post-solve (sand_rigid.hpp) and the lagged value feeds the contact-point
    // velocity v_contact = v_center + omega x r in the next step's friction.
    // A grain sphere has isotropic inertia and the torque arm is the contact
    // normal, so no orientation is stored (rolling is a motion, not a render).
    Vec<Vec3f> grain_omega;
    // Per-grain inverse moment of inertia 1/((2/5) m r^2) for a solid sphere;
    // zero for non-grain vertices, so the integrate skips them.
    Vec<float> grain_inv_inertia;
    // Per-grain contact-friction torque tau = sum over ALL of the grain's
    // contacts of r*(n x g) (g the friction gradient, n the outward contact
    // normal). ZEROED at the top of each Newton iteration (main.cu) and
    // ACCUMULATED with atomicAdd from BOTH the grain-grain point-point embed and
    // the per-vertex floor/sphere embed, so after convergence it holds the summed
    // converged torque consumed once by the post-solve integrate. Transient
    // working buffer, zero for non-grain vertices.
    Vec<Vec3f> grain_torque;
    // Per-grain angular friction stiffness K = sum lambda*radius^2 (lambda the
    // friction stiffness), accumulated alongside grain_torque. The integrate
    // uses it for a semi-implicit omega step omega += dt*Iinv*tau/(1 +
    // dt^2*Iinv*K), which damps the friction's own omega-dependence so omega
    // converges to the rolling rate instead of overshooting. Transient.
    Vec<float> grain_ang_stiff;
    // Per-grain SUM of the unit contact normals over ALL of the grain's contacts
    // this step (zero if airborne). Written by the contact embed and consumed by
    // the post-solve integrate, which normalizes it to the dominant contact
    // direction n and caps the spin at the TANGENTIAL no-slip rate
    // |v - (v.n) n| / radius. Summing (not last-wins) is what makes the clamp
    // correct when a grain touches several surfaces at once (a corner, or the
    // floor plus neighbor grains in a pile): the net normal is the direction the
    // grain is constrained against, so the tangential velocity it is capped to is
    // the direction it can actually roll. Using the tangential (not the full)
    // speed removes the normal-velocity slack that otherwise lets a fast grain
    // over-spin and pump. Transient, zero for non-grain / non-contacting verts.
    Vec<Vec3f> grain_contact_normal;
    // Implicit (Schur-condensed) rolling per-grain state. grain_inv_inertia_center
    // is 1/I_center = 1/((2/5) m r^2), the BARE solid-sphere center inertia used by
    // the Schur condense/recover (NOT grain_inv_inertia = 1/I_eff, which stays for
    // the grain-grain staggered (post-solve) integrate). grain_omega_prev is the start-of-step
    // angular-velocity snapshot (held constant across Newton iterations as the
    // inertia reference). grain_A (SPD angular block), grain_B (translation<->
    // rotation coupling), grain_grot (rotational gradient) are TRANSIENT, zeroed
    // each Newton iteration and summed over the grain's floor/sphere contacts;
    // sand_rigid.hpp condenses them into the grain's 3x3 translation block. All
    // zero for non-grain verts.
    Vec<float> grain_inv_inertia_center;
    Vec<Vec3f> grain_omega_prev;
    Vec<Mat3x3f> grain_A;
    Vec<Mat3x3f> grain_B;
    Vec<Vec3f> grain_grot;
};

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

#define MAX_INTERSECTION_RECORDS 256

struct IntersectionRecord {
    unsigned type;       // 0=face-edge, 1=edge-edge, 2=collision-mesh, 3=point-point
    unsigned elem0;      // first element index (face, edge, or vertex)
    unsigned elem1;      // second element index (edge or vertex)
    unsigned num_verts0; // vertex count for first element (1, 2, or 3)
    unsigned num_verts1; // vertex count for second element (1, 2, or 3)
    float positions[15]; // up to 5 vertices x 3 (x,y,z), packed: elem0 then elem1
};

#endif
