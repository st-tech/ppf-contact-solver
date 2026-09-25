// File: data.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DATA_HPP
#define DATA_HPP

// The backend seam, first, so every shared kernel body reached from this
// translation unit finds the SM_ spellings already defined and needs no
// preprocessor conditional of its own. This is the one place the CUDA and host
// prologues are pulled in; the Metal shader compiler reads neither of them and
// gets the same table from metal/shader_compiler.mm instead. Read
// seam/seam.hpp for the contract.
#include "seam/seam.hpp"

#include "linalg/smat.hpp"
#include "vec/vec.hpp"

// NO_OBJECT_INDEX and the INTERSECT_ALLOW_* bits are declared there, beside the
// rule that reads them, because the Metal shader compiler needs both and can
// only be handed a header it embeds.
#include "contact/intersect_policy.hpp"

#include "linalg/type_aliases.hpp"

// The fixed-size records every array below is made of: the enums, the
// per-element params and props, the constraint pairs, the bounding box, the
// locks. They are a separate file because the Metal shader compiler can read
// them and cannot read this one (`vec/vec.hpp` above spells a raw `T *`, which
// MSL refuses for want of an address space), and the shipped shader and the
// offline entry-point check both splice that file directly. Read the argument
// at the top of it.
#include "data_records.hpp"

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
    // Device-visible float image of time. MSL has no double, so its ParamSet
    // mirror keeps the slot above as padding and reads this field.
    float time_f32;
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
    float ccd_eps;
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
    // Upper bound on Newton iterations per substep. The loop is otherwise
    // unbounded, so an over-constrained configuration (a prescribed pin driven
    // into geometry that cannot yield) spins forever: the line search clamps
    // the shared toi toward zero to prevent the penetration, that same clamp
    // throttles every other vertex, and the toi never falls below FLT_EPSILON,
    // so the CCD trap never fires. This bound turns that hang into a loud
    // CrashKind::NewtonStall. 0 disables the bound (research only).
    // Appended at the tail; field order and byte layout MUST mirror the Rust
    // ParamSet in data.rs (repr(C) ABI).
    unsigned max_newton_steps;
    // Diagnostic A/B lever (PPF_DISABLE_PIN_DOF_REMOVAL=1, set host-side in
    // advance()): revert every fix pin from an exact Dirichlet BC back to the
    // old barrier penalty. Both halves must flip together -- main.cu stops
    // eliminating the rows AND contact.cu puts the barrier back -- or the pin
    // would have neither and simply vanish.
    // Appended at the tail; field order and byte layout MUST mirror the Rust
    // ParamSet in data.rs (repr(C) ABI).
    bool disable_pin_dof_removal;
};

struct StepResult {
    double time;
    bool ccd_success;
    bool pcg_success;
    bool intersection_free;
    // False when the Newton loop hit max_newton_steps without reaching an
    // acceptable step (an over-constrained configuration: the line search
    // clamps the shared toi toward zero to stop a penetration, and that same
    // clamp throttles every other vertex, so no iteration progresses). Field
    // order must mirror Rust StepResult in data.rs (repr(C) ABI).
    bool newton_progress;
    // False when a prescribed (fix-pinned) vertex's swept path crosses an
    // analytic collider (floor / sphere / wall). Such a vertex has no DOF to
    // yield with, so the prescription itself is infeasible.
    bool pin_feasible;
    // False when a contact pair begins the step already inside the contact
    // offset (two surfaces start out touching or overlapping), so the
    // conservative CCD cannot advance from a separated start. Appended at the
    // tail; field order and byte layout MUST mirror the Rust StepResult in
    // data.rs (repr(C) ABI).
    bool contact_separated;
    bool success() const {
        return ccd_success && pcg_success && intersection_free &&
               newton_progress && pin_feasible && contact_separated;
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
    // Per-grain angular friction scale K = sum lambda*radius^2, accumulated
    // alongside grain_torque. Lambda is the scalar force regularization scale;
    // using it here keeps the staggered spin update conservatively damped when
    // the kinetic translational Hessian has zero curvature along slip. The
    // integrate uses K for omega += dt*Iinv*tau/(1 + dt^2*Iinv*K), which
    // prevents overshoot past the rolling rate. Transient.
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
    // Slot-replay assembly tables. Flat FixedCSRMat value-slot index of
    // every 3x3 block each topology-fixed element writes, row-major (ii*N + jj),
    // with 0xFFFFFFFF sentinels for lower-triangle (push() no-op) blocks. Empty
    // (size 0) when PPF_SLOT_REPLAY=0, in which case the assembly kernels fall
    // back to push(). Field order/layout MUST mirror the CVec<u32> tail of the
    // Rust DataSet in data.rs (repr(C) ABI); tail-append only.
    Vec<unsigned> tet_hess_slots;      // 16 per tet
    Vec<unsigned> face_hess_slots;     // 9 per face (membrane/inflate/strain)
    Vec<unsigned> edge_hess_slots;     // 4 per edge (rod stretch/strain)
    Vec<unsigned> hinge_hess_slots;    // 16 per hinge, REMAPPED (2,1,0,3) order
    Vec<unsigned> rod_bend_hess_slots; // 9 per surface vertex (j,i,k stencil)
    Vec<unsigned> stitch_hess_slots;   // 36 per stitch seam
    // Center-of-mass translation lock data. `translation_lock_index` is
    // per-global-vertex and uses 0xffffffff for no lock. Initial positions are
    // immutable, so current-minus-initial is one difference against a fixed
    // reference rather than an accumulation of per-step deltas.
    Vec<TranslationLock> translation_lock;
    Vec<unsigned> translation_lock_index;
    Vec<Vec3f> translation_lock_initial;
    // Statistics object ownership and per-object accepted-contact counts.
    // u32::MAX marks a vertex outside the manifest. Counts are cleared before
    // each Newton assembly and fetched with the output pose.
    Vec<unsigned> statistics_object_index;
    Vec<unsigned> statistics_static_object_index;
    Vec<unsigned> statistics_contact_count;
    // Allow Existing Intersections' vertex links: one row per dynamic vertex
    // listing the vertices it is linked to, a collision-mesh vertex carrying
    // START_LINK_COLLISION_VERTEX. Empty (size 0) when nothing was linked.
    // Field order must mirror the Rust DataSet in data.rs (repr(C) ABI);
    // tail-append only.
    VecVec<unsigned> start_link;
};

#endif
