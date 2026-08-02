// File: main.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

// Windows DLL export macro
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#include "../buffer/buffer.hpp"
#include "../contact/contact.hpp"
#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../energy/energy.hpp"
#include "../energy/model/pdrd_rigid.hpp"
#include "../energy/model/sand_rigid.hpp"
#include "../kernels/exclusive_scan.hpp"
#include "../kernels/reduce.hpp"
#include "../kernels/vec_ops.hpp"
#include "../lbvh/bvh_storage.hpp"
#include "../lbvh/lbvh.hpp"
#include "../main/cuda_utils.hpp"
#include "../simplelog/SimpleLog.h"
#include "../solver/solver.hpp"
#include "../plasticity/plasticity.hpp"
#include "../strainlimiting/strainlimiting.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "cuda_utils.hpp"
#include "dump_linsys.hpp"
#include "mem.hpp"
#include <cassert>
#include <cstdlib>
#include <limits>

namespace tmp {
FixedCSRMat fixed_hessian;
FixedCSRMat tmp_fixed;
DynCSRMat dyn_hess;
} // namespace tmp

// Per-vertex collision window data on GPU, propagated to face/edge
#define MAX_COLLISION_WINDOWS 8
struct CollisionWindowData {
    unsigned *d_vert_dmap;
    float *d_windows;
    unsigned *d_window_counts;
    bool *d_vert_active;
    bool *d_face_active;
    bool *d_edge_active;
    unsigned vert_count;
    unsigned face_count;
    unsigned edge_count;
    unsigned n_groups;
    bool initialized;
};
static CollisionWindowData cw_data = {};

// Not `static`: NVCC on Windows rejects extended __device__ lambdas
// enclosed by a function with internal or no linkage. The function is
// still private to this translation unit; no header declares it.
void invalidate_inactive_aabbs() {
    if (!cw_data.initialized || !cw_data.d_face_active) return;

    const BVHSet &bvhset = bvh_storage::get_bvh();
    Vec<AABB> &face_aabb = contact::get_face_aabb();
    Vec<AABB> &edge_aabb = contact::get_edge_aabb();
    Vec<AABB> &vert_aabb = contact::get_vertex_aabb();
    auto d_fa = cw_data.d_face_active;
    auto d_ea = cw_data.d_edge_active;
    auto d_va = cw_data.d_vert_active;
    unsigned fc = face_aabb.size;
    unsigned ec = edge_aabb.size;
    unsigned svc = vert_aabb.size;
    auto fa_ptr = face_aabb.data;
    auto ea_ptr = edge_aabb.data;
    auto va_ptr = vert_aabb.data;
    auto face_nodes = bvhset.face.node.data;
    auto edge_nodes = bvhset.edge.node.data;
    auto vert_nodes = bvhset.vertex.node.data;

    if (fc > 0) {
        DISPATCH_START(fc)
        [fa_ptr, d_fa, face_nodes] __device__(unsigned i) mutable {
            unsigned prim = face_nodes[i][0] - 1;
            fa_ptr[i].active = d_fa[prim];
        }
        DISPATCH_END
    }
    if (ec > 0) {
        DISPATCH_START(ec)
        [ea_ptr, d_ea, edge_nodes] __device__(unsigned i) mutable {
            unsigned prim = edge_nodes[i][0] - 1;
            ea_ptr[i].active = d_ea[prim];
        }
        DISPATCH_END
    }
    if (svc > 0) {
        DISPATCH_START(svc)
        [va_ptr, d_va, vert_nodes] __device__(unsigned i) mutable {
            unsigned prim = vert_nodes[i][0] - 1;
            va_ptr[i].active = d_va[prim];
        }
        DISPATCH_END
    }

    // Re-propagate internal node AABBs with active-aware merge
    lbvh::propagate_aabbs(face_aabb, bvhset.face.node, bvhset.face.level);
    lbvh::propagate_aabbs(edge_aabb, bvhset.edge.node, bvhset.edge.level);
    lbvh::propagate_aabbs(vert_aabb, bvhset.vertex.node, bvhset.vertex.level);
}

namespace main_helper {
DataSet host_dataset, dev_dataset;
ParamSet *param;
// True iff the scene contains at least one SAND grain (grain_inv_inertia > 0).
// Computed once at initialize() by a host scan; gates the per-Newton-iteration
// grain-buffer clears so non-SAND scenes launch zero extra kernels. The grain_*
// buffers are sized n_vert in EVERY scene, so a size check would be useless.
bool has_grains = false;

bool initialize(DataSet _host_dataset, DataSet _dev_dataset, ParamSet *_param) {

    // Name: Initialization Time
    // Format: list[(time, ms)]
    // Description:
    // Total wall-clock time in milliseconds spent inside the one-time
    // solver initialization (buffer allocation, contact setup, initial
    // LBVH build, initial intersection check). Only a single record is
    // expected, written when the initialize scope exits. The leading
    // time column is the simulation time at the moment of recording,
    // which is 0 for initialization.
    SimpleLog logging("initialize");

    bool result = true;
    host_dataset = _host_dataset;
    dev_dataset = _dev_dataset;
    param = _param;

    // Detect SAND grains once (host scan of the host-side inverse-inertia mirror;
    // grain_inv_inertia > 0 only for grains, and 0 for every non-SAND scene).
    has_grains = false;
    for (unsigned j = 0; j < host_dataset.grain_inv_inertia.size; ++j) {
        if (host_dataset.grain_inv_inertia.data[j] > 0.0f) {
            has_grains = true;
            break;
        }
    }

    unsigned vert_count = host_dataset.vertex.curr.size;
    unsigned edge_count = host_dataset.mesh.mesh.edge.size;
    unsigned face_count = host_dataset.mesh.mesh.face.size;
    unsigned hinge_count = host_dataset.mesh.mesh.hinge.size;
    unsigned tet_count = host_dataset.mesh.mesh.tet.size;
    unsigned collision_mesh_vert_count =
        host_dataset.constraint.mesh.vertex.size;
    unsigned collision_mesh_edge_count = host_dataset.constraint.mesh.edge.size;

    unsigned shell_face_count = host_dataset.shell_face_count;
    unsigned surface_vert_count = host_dataset.surface_vert_count;

    // Buffer system now allocates on-demand, no initialization needed

    // Allocate matrix buffers
    tmp::dyn_hess = DynCSRMat::alloc(vert_count, _param->csrmat_max_nnz);
    tmp::fixed_hessian = FixedCSRMat::alloc(dev_dataset.fixed_index_table,
                                            dev_dataset.transpose_table);
    tmp::tmp_fixed = FixedCSRMat::alloc(dev_dataset.fixed_index_table,
                                        dev_dataset.transpose_table);

    contact::initialize(host_dataset, *param);

    // Re-seed the persistent PDRD per-body rotation on (re)initialize / scene
    // load, so the anchored rigidify starts from the absolute fit of the loaded
    // pose rather than a stale rotation.
    PDRD::pdrd_reset_rprev();

    // Initialize GPU LBVH construction buffers
    // Use max of main mesh and collision mesh sizes
    unsigned collision_mesh_face_count = host_dataset.constraint.mesh.face.size;
    unsigned max_faces = face_count > collision_mesh_face_count ? face_count : collision_mesh_face_count;
    unsigned max_edges = edge_count > collision_mesh_edge_count ? edge_count : collision_mesh_edge_count;
    unsigned max_verts = surface_vert_count > collision_mesh_vert_count ? surface_vert_count : collision_mesh_vert_count;
    lbvh::initialize(max_faces, max_edges, max_verts);

    // Pre-allocate the scratch pool for the mesh/body-bounded buffers the solve
    // loop reuses, so the hot path performs no dynamic GPU alloc/dealloc once
    // warmed up. schwarz's contact-driven scratch is left to grow the same pool
    // to its high-water mark at runtime (its sizes are not known here). The
    // vertex driver covers both the full-mesh DOF count and the surface/
    // collision-mesh contact counts.
    {
        unsigned pool_verts =
            vert_count > max_verts ? vert_count : max_verts;
        unsigned n_bodies = host_dataset.prop.pdrd_body.size;
        buffer::reserve_for_mesh(pool_verts, max_edges, max_faces, n_bodies);
    }

    if (!param->disable_contact) {
        // Name: Initial LBVH Build Time
        // Format: list[(time, ms)]
        // Map: initial_lbvh_build
        // Description:
        // Wall-clock time in milliseconds to build the initial LBVH (Linear
        // Bounding Volume Hierarchy) over faces, edges, and vertices at the
        // start of the simulation, including the collision-mesh BVH. Only a
        // single record is expected.
        logging.push("lbvh build");
        lbvh::build_face_bvh(dev_dataset.vertex.curr, dev_dataset.vertex.curr,
                             1.0f, dev_dataset.mesh.mesh.face,
                             bvh_storage::get_bvh().face, contact::get_face_aabb(),
                             dev_dataset.prop.face,
                             dev_dataset.param_arrays.face);
        lbvh::build_edge_bvh(dev_dataset.vertex.curr, dev_dataset.vertex.curr,
                             1.0f, dev_dataset.mesh.mesh.edge,
                             bvh_storage::get_bvh().edge, contact::get_edge_aabb(),
                             dev_dataset.prop.edge,
                             dev_dataset.param_arrays.edge);
        lbvh::build_vertex_bvh(dev_dataset.vertex.curr, dev_dataset.vertex.curr,
                               1.0f, bvh_storage::get_bvh().vertex,
                               contact::get_vertex_aabb(), surface_vert_count,
                               dev_dataset.prop.vertex,
                               dev_dataset.param_arrays.vertex);
        lbvh::build_collision_mesh_bvh(dev_dataset, *param);
        logging.pop();
        // Name: Initial Intersection Check Time
        // Format: list[(time, ms)]
        // Map: initial_check_intersection
        // Description:
        // Wall-clock time in milliseconds spent scanning the previous and
        // current vertex positions for self-intersections at the start of
        // the simulation. Only a single record is expected. Useful for
        // diagnosing geometry that begins the simulation already tangled.
        logging.push("check intersection");
        if (!contact::check_intersection(dev_dataset, dev_dataset.vertex.prev,
                                         *param) ||
            !contact::check_intersection(dev_dataset, dev_dataset.vertex.curr,
                                         *param)) {

            logging.message("### intersection detected");
            result = false;
        }
        logging.pop();
    }
    return result;
}

StepResult advance() {

    // Name: Time Per Simulation Step
    // Format: list[(time, ms)]
    // Map: time_per_step
    // Description:
    // Wall-clock time in milliseconds spent inside a single advance call
    // (one simulation step). Note that a step does not advance by a fixed
    // dt: the actual step size is reduced by the accumulated time of
    // impact found during the inner Newton loop, so these values also
    // reflect how hard the solver had to work to progress the step.
    SimpleLog logging("advance");

    // Device alloc/free counts at step entry. The per-step delta (logged at
    // exit) verifies the steady-state goal: once the scene reaches peak contact
    // and the pre-allocated / high-water pool has warmed up, the solve loop
    // performs ZERO dynamic GPU alloc/dealloc. A nonzero steady-state delta
    // flags a buffer that still escapes the pool.
    const unsigned long long dev_alloc_at_entry = g_device_alloc_count;
    const unsigned long long dev_free_at_entry = g_device_free_count;

    StepResult result;
    result.pcg_success = true;
    result.ccd_success = true;
    result.intersection_free = true;
    result.newton_progress = true;
    result.pin_feasible = true;
    result.contact_separated = true;
    // Clear the CCD "starts overlapping" flag once for this advance; every
    // contact line search below sets it if a pair begins the step inside the
    // contact offset, and the checks after each line search read it.
    contact::clear_ccd_overlap();

    DataSet &host_data = host_dataset;
    DataSet data = dev_dataset;
    ParamSet prm = *param;

    const unsigned vertex_count = host_data.vertex.curr.size;
    const unsigned shell_face_count = host_dataset.shell_face_count;
    const unsigned rod_count = host_dataset.rod_count;
    const unsigned tet_count = host_data.mesh.mesh.tet.size;

    // PDRD: PDRD bodies are exactly rigid. After each collision-free Newton
    // position update, the surface is snapped onto the nearest exactly-rigid
    // configuration (fit (x_b,R_b), reconstruct x_v = x_b + R_b ybar). That snap
    // is itself a trajectory, so its delta is run through the contact CCD line
    // search; if rigidifying would penetrate, we take a partial step (toi<1) and
    // the remaining non-rigid residual is removed over subsequent iterations
    // (at toi=0 the snap is identity, so a feasible step always exists).
    const unsigned n_pdrd_bodies = host_data.prop.pdrd_body.size;
    const bool rigid_pdrd = n_pdrd_bodies > 0;

    // Get buffers from buffer pool (auto-deduce PooledVec type)
    buffer::MemoryPool &pool = buffer::get();
    auto eval_x = pool.get<Vec3f>(vertex_count);
    auto target = pool.get<Vec3f>(vertex_count);

    // Get matrix buffers from tmp namespace
    DynCSRMat &dyn_hess = tmp::dyn_hess;
    FixedCSRMat &tmp_fixed = tmp::tmp_fixed;
    FixedCSRMat &fixed_hess = tmp::fixed_hessian;

    SimpleLog::set(prm.time);

    // Build BVH on GPU
    if (!prm.disable_contact) {
        // Name: LBVH Build Time
        // Format: list[(time, ms)]
        // Map: lbvh_build
        // Description:
        // Wall-clock time in milliseconds to rebuild the LBVH (Linear
        // Bounding Volume Hierarchy) over faces, edges, and vertices at
        // the start of each simulation step. This BVH underpins broad-phase
        // contact detection, so this cost tracks mesh size and how often
        // primitives are deactivated by collision windows.
        logging.push("lbvh build");
        lbvh::build_face_bvh(data.vertex.curr, data.vertex.curr, 1.0f,
                             data.mesh.mesh.face, bvh_storage::get_bvh().face,
                             contact::get_face_aabb(), data.prop.face,
                             data.param_arrays.face);
        lbvh::build_edge_bvh(data.vertex.curr, data.vertex.curr, 1.0f,
                             data.mesh.mesh.edge, bvh_storage::get_bvh().edge,
                             contact::get_edge_aabb(), data.prop.edge,
                             data.param_arrays.edge);
        lbvh::build_vertex_bvh(data.vertex.curr, data.vertex.curr, 1.0f,
                               bvh_storage::get_bvh().vertex, contact::get_vertex_aabb(),
                               host_data.surface_vert_count, data.prop.vertex,
                               data.param_arrays.vertex);
        logging.pop();
        invalidate_inactive_aabbs();
    }

    // Define data array pointers for reuse
    auto vertex_curr = data.vertex.curr.data;
    auto vertex_prev = data.vertex.prev.data;
    auto prop_vertex = data.prop.vertex.data;
    auto prop_face = data.prop.face.data;
    auto prop_edge = data.prop.edge.data;
    auto prop_tet = data.prop.tet.data;
    auto param_face = data.param_arrays.face.data;
    auto constraint_fix = data.constraint.fix.data;
    auto mesh_face = data.mesh.mesh.face.data;
    auto mesh_edge = data.mesh.mesh.edge.data;
    float prev_dt = prm.prev_dt;
    Vec3f gravity = prm.gravity;
    bool inactive_momentum = prm.inactive_momentum;
    float fix_xz_val = prm.fix_xz;

    // Compute max velocity and store velocities for later use
    auto velocity = pool.get<Vec3f>(vertex_count);
    float max_u;
    {
        auto tmp_scalar = pool.get<float>(vertex_count);
        tmp_scalar.clear();
        Vec<float> tmp_scalar_vec = tmp_scalar.as_vec();
        Vec<Vec3f> velocity_vec = velocity.as_vec();
        DISPATCH_START(vertex_count)
        [vertex_curr, vertex_prev, prop_vertex, tmp_scalar_vec, velocity_vec,
         prev_dt] __device__(unsigned i) mutable {
            Vec3f u = (vertex_curr[i] - vertex_prev[i]) / prev_dt;
            velocity_vec[i] = u;
            tmp_scalar_vec[i] =
                prop_vertex[i].fix_index > 0 ? 0.0f : u.squaredNorm();
        } DISPATCH_END;
        max_u = sqrtf(kernels::max_array(tmp_scalar.data, vertex_count, 0.0f));
    }

    // Name: Max Vertex Velocity
    // Format: list[(time, m/s)]
    // Map: max_velocity
    // Description:
    // Maximum speed (in meters per second) among all non-pinned vertices,
    // measured from the previous to the current positions at the start of
    // the step. Pinned (fixed) vertices are excluded. Useful for spotting
    // explosions or abrupt motion in the simulation.
    logging.mark("max_u", max_u);

    float dt = param->dt * param->playback;

    // Name: Target Step Size
    // Format: list[(time, seconds)]
    // Description:
    // Target integration step size in seconds at the start of this
    // simulation step, computed as the configured dt scaled by the current
    // playback rate. The actually advanced step size can be smaller (see
    // the Final Step Size channel) if the line search reduces it.
    logging.mark("dt", dt);

    // Name: Playback Speed
    // Format: list[(time, ratio)]
    // Description:
    // Playback rate applied this step, as a multiplier on the configured
    // dt. A value of 1.0 means real-time playback, below 1.0 slows motion
    // down, and above 1.0 speeds it up. The value can change between
    // steps when the scene scripts playback over time.
    logging.mark("playback", param->playback);

    if (shell_face_count || rod_count) {
        float max_sigma = 0.0f;
        if (shell_face_count) {
            auto svd = pool.get<Svd3x2>(shell_face_count);
            auto tmp_scalar = pool.get<float>(shell_face_count);
            utility::compute_svd(data, data.vertex.curr, svd, prm);
            tmp_scalar.clear();
            Vec<Svd3x2> svd_vec = svd.as_vec();
            Vec<float> tmp_scalar_vec = tmp_scalar.as_vec();
            DISPATCH_START(shell_face_count)
            [prop_face, param_face, mesh_face, prop_vertex, svd_vec,
             tmp_scalar_vec] __device__(unsigned i) mutable {
                const FaceProp &prop = prop_face[i];
                // PDRD bodies have no per-element elastic stretch; their
                // rigid fit always carries a small singular-value residual
                // that would dominate this metric and mislead diagnostics.
                // Exclude them entirely.
                const Vec3u &face = mesh_face[i];
                if (prop_vertex[face[0]].pdrd_body_index != 0) {
                    return;
                }
                if (!prop.fixed) {
                    const FaceParam &fparam = param_face[prop.param_index];
                    tmp_scalar_vec[i] =
                        fmaxf(svd_vec[i].S[0], svd_vec[i].S[1]) * fminf(fparam.shrink_x, fparam.shrink_y);
                }
            } DISPATCH_END;
            max_sigma = fmaxf(
                max_sigma,
                kernels::max_array(tmp_scalar.data, shell_face_count, 0.0f));
        }
        if (rod_count) {
            auto tmp_scalar = pool.get<float>(rod_count);
            tmp_scalar.clear();
            Vec<float> tmp_scalar_vec = tmp_scalar.as_vec();
            DISPATCH_START(rod_count)
            [prop_edge, mesh_edge, vertex_curr,
             tmp_scalar_vec] __device__(unsigned i) mutable {
                const EdgeProp &prop = prop_edge[i];
                if (!prop.fixed && prop.initial_length > 0.0f) {
                    const Vec2u &edge = mesh_edge[i];
                    Vec3f d = vertex_curr[edge[1]] - vertex_curr[edge[0]];
                    tmp_scalar_vec[i] = d.norm() / prop.initial_length;
                }
            } DISPATCH_END;
            max_sigma = fmaxf(
                max_sigma,
                kernels::max_array(tmp_scalar.data, rod_count, 0.0f));
        }
        // Name: Max Stretch Ratio
        // Format: list[(time, ratio)]
        // Description:
        // Maximum stretch ratio among all shell faces and rod edges in the
        // scene, measured at the start of the step before the Newton loop.
        // For shells this is the largest singular value of the deformation
        // gradient (scaled by the shrink factor), for rods it is the current
        // edge length divided by its rest length. A value of 1.02 means a
        // 2 percent stretch. Useful for diagnosing strain-limit tightness.
        logging.mark("max_sigma", max_sigma);
    }

    auto compute_target = [&](float dx) {
        Vec<Vec3f> target_vec = target.as_vec();
        DISPATCH_START(vertex_count)
        [prop_vertex, constraint_fix, vertex_curr, vertex_prev, target_vec, dx,
         dt, prev_dt, gravity, inactive_momentum] __device__(unsigned i) mutable {
            if (prop_vertex[i].fix_index > 0) {
                unsigned index = prop_vertex[i].fix_index - 1;
                target_vec[i] = constraint_fix[index].position;
            } else {
                Vec3f &x1 = vertex_curr[i];
                Vec3f &x0 = vertex_prev[i];
                float tr(dt / prev_dt), h2(dt * dt);
                Vec3f y = (x1 - x0) * tr + h2 * gravity;
                if (inactive_momentum) {
                    target_vec[i] = x1;
                } else {
                    target_vec[i] = x1 + y;
                }
            }
        } DISPATCH_END;
    };

    // Walk every kinematic fix pin back to the fraction of its scheduled step
    // it is actually going to travel. The host aims a step at some end time and
    // writes each pin's pose there into FixPair::position, but the CCD line
    // search may only get `toi` of the way through the step, and the clock is
    // then advanced by that same fraction (`dt *= toi_advanced` below). A pin
    // left at the full step's pose would therefore sit where the animation puts
    // it at a time the simulation never reached: the collider outruns its own
    // animation for the rest of the step and snaps back on the next one. That
    // shows up as a jitter on the collider and, worse, as a spurious contact
    // impulse handed to whatever it is touching. (An exact-Dirichlet fix pin
    // makes this sharper than the old penalty, not softer: its DOF are
    // eliminated, so it lands on the full-step pose exactly.)
    //
    // A pin's path bends only where its keyframes sit, which is far apart
    // relative to a step, so over one step it is straight to within O(dt^2) and
    // the pose at fraction `toi` is `position - (1 - toi) * step_delta`. Static
    // pins hold a zero delta and do not move. Rewrites the constraint in place:
    // the host uploads a fresh one every step, and this runs once per step.
    auto rewind_kinematic_fix = [&](double toi) {
        unsigned fix_count = data.constraint.fix.size;
        if (toi >= 1.0 || fix_count == 0) {
            return;
        }
        float back = 1.0f - static_cast<float>(toi);
        Vec<FixPair> fix_vec = data.constraint.fix;
        DISPATCH_START(fix_count)
        [fix_vec, back] __device__(unsigned i) mutable {
            FixPair &fix = fix_vec[i];
            if (fix.kinematic) {
                fix.position = fix.position - back * fix.step_delta;
            }
        } DISPATCH_END;
    };

    compute_target(dt);

    kernels::copy(data.vertex.curr.data, eval_x.data, eval_x.size);

    double toi_advanced = 0.0f;
    unsigned step(1);
    bool final_step(false);
    // Last line-search fraction, hoisted out of the loop purely so the
    // Newton-stall diagnostic can report it (`toi` itself is loop-scoped).
    float last_toi = 1.0f;

    // Allocate buffers for Newton loop (auto-release when function exits)
    auto force = pool.get<float>(3 * vertex_count);
    auto dx = pool.get<float>(3 * vertex_count);
    auto diag_hess = pool.get<Mat3x3f>(vertex_count);

    // THE PIN MODEL. A pin is one of exactly two things, and which one is decided
    // by its pull weight, not by a stiffness scalar in between:
    //
    //   pull weight > 0  ->  PULL pin: a soft spring (f = w(y-x), H = wI). It
    //                        yields to contact and elasticity. This is the only
    //                        way to hold a vertex compliantly.
    //   pull weight == 0 ->  FIX pin: an exact DIRICHLET BOUNDARY CONDITION. Its
    //                        DOF are eliminated from the Newton system below, so
    //                        it tracks its prescribed position to round-off and
    //                        never yields.
    //
    // A fix pin used to be a barrier PENALTY (stiff_k = tmp + mass/gap^2, scaled
    // by a per-pin stiffness), i.e. a stiff spring. That was wrong twice over.
    // It let cloth contact shove a moving collider off its keyframe (a visible
    // jiggle), and, worse, a fully-pinned element is excluded from collision CCD
    // (`either_dyn` is false when neither side is free), so a pinned patch driven
    // into a static collider TUNNELED IT SILENTLY. A prescribed vertex is not a
    // compromise between contact and animation: the user gave it its entire
    // trajectory. Imposing it exactly is both more faithful and strictly safer
    // (the same scene now reports a loud NewtonStall instead of penetrating).
    //
    // THE ONE EXCEPTION: a vertex inside a PDRD rigid body owns no per-vertex
    // DOF. The solve is reduced through the rigid Jacobian and `launch_rigidify`
    // refits the body afterward, so a per-vertex Dirichlet row is not
    // representable there. Such an anchor keeps the barrier, and it is the only
    // penalty pin left in the codebase. (Partial-pinned SOLIDs are excluded for
    // free: the decoder reroutes their interior pins to PULL, so they are not
    // fix pins at all.)
    //
    // fix_index and pdrd_body_index are constant across this advance(), so the
    // mask is computed once here.
    //
    // Escape hatch / A-B switch: PPF_DISABLE_PIN_DOF_REMOVAL=1 leaves the mask
    // all-zero, reverting every fix pin to the barrier penalty. Diagnostic only.
    static const bool disable_dof_removal = [] {
        const char *e = std::getenv("PPF_DISABLE_PIN_DOF_REMOVAL");
        return e && e[0] == '1';
    }();
    // The device needs this too: contact.cu must put the fix barrier back for
    // exactly the pins whose rows we stop eliminating, or they would have
    // neither a Dirichlet row nor a penalty and would not be held at all.
    prm.disable_pin_dof_removal = disable_dof_removal;
    auto dof_removed_mask = pool.get<unsigned>(vertex_count);
    dof_removed_mask.clear();
    if (!disable_dof_removal) {
        Vec<unsigned> mask = dof_removed_mask.as_vec();
        DISPATCH_START(vertex_count)
        [prop_vertex, mask] __device__(unsigned i) mutable {
            const VertexProp &prop = prop_vertex[i];
            mask[i] = (prop.fix_index > 0 && prop.pdrd_body_index == 0) ? 1u : 0u;
        } DISPATCH_END;
    }
    const unsigned n_dof_removed =
        kernels::sum_array<unsigned>(dof_removed_mask.data, vertex_count);
    // One-time visibility (silent in a scene with no pins at all).
    static bool dof_reported = false;
    if (!dof_reported && (n_dof_removed > 0 || disable_dof_removal)) {
        dof_reported = true;
        ::logging::info(
            "dof-removal: %u pinned vertices eliminated as Dirichlet BCs (%s)",
            n_dof_removed,
            disable_dof_removal ? "DISABLED via env" : "enabled");
    }

    // Experimental CG warm-start (PPF_CG_WARMSTART=1): seed the first Newton
    // solve of each frame with the previous frame's converged search direction
    // instead of zero. PCG converges to the same solution within tol regardless
    // of the initial guess, so this only changes the iteration count, not the
    // result. `dx_warm` persists across advance() calls (one process per
    // session, like PDRD::pdrd_rprev); reseeded if vertex_count changes.
    // (A 2-frame linear-extrapolation variant was measured as a wash and dropped.)
    static bool cg_warmstart = [] {
        const char *e = std::getenv("PPF_CG_WARMSTART");
        return e && e[0] == '1';
    }();
    static Vec<float> dx_warm;
    static unsigned dx_warm_n = 0;
    static bool dx_warm_seeded = false;
    if (cg_warmstart && dx_warm_n != 3u * vertex_count) {
        if (dx_warm.size) {
            dx_warm.free();
        }
        dx_warm = Vec<float>::alloc(3u * vertex_count);
        dx_warm_n = 3u * vertex_count;
        dx_warm_seeded = false;
    }
    auto rigid_tgt = pool.get<Vec3f>(rigid_pdrd ? vertex_count : 1);

    // Anchored-rigidify state (PDRD): R_run integrates the applied rotation
    // increment onto the persistent committed rotation R_prev, so the rigidify
    // target rotation never re-fits the contact-sheared eval_x (which is what
    // accumulated the non-rigid shrink). pdrd_dtheta carries the per-body reduced
    // rotation step out of the solve each Newton iteration.
    unsigned n_pdrd = data.prop.pdrd_body.size;
    auto R_run = pool.get<float>(rigid_pdrd ? 9 * n_pdrd : 1);
    auto pdrd_dtheta = pool.get<float>(rigid_pdrd ? 3 * n_pdrd : 1);
    if (rigid_pdrd) {
        Vec<float> &R_prev = PDRD::pdrd_rprev();
        if (R_prev.size < 9u * n_pdrd) {
            R_prev.free();
            R_prev = Vec<float>::alloc(9u * n_pdrd);
            PDRD::pdrd_rprev_seeded() = false;
        }
        if (!PDRD::pdrd_rprev_seeded()) {
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            PDRD::launch_seed_rprev(data, eval_x_vec, R_prev);
            PDRD::pdrd_rprev_seeded() = true;
        }
        kernels::copy(R_prev.data, R_run.data, 9u * n_pdrd); // R_run <- R_prev
    }

    // Implicit (Schur-condensed) rolling: snapshot each grain's start-of-step
    // angular velocity. The condense/recover use it as the (constant) inertia
    // reference across Newton iterations, since recover overwrites grain_omega
    // in-loop.
    if (has_grains) {
        kernels::copy(data.grain_omega.data, data.grain_omega_prev.data,
                      data.grain_omega.size);
    }

    while (true) {
        // The Newton loop is otherwise unbounded, and its only escape is a
        // collapsed time of impact (`toi <= FLT_EPSILON`, below). That escape
        // does NOT cover an over-constrained configuration: when a prescribed
        // vertex is driven into geometry that cannot yield, the line search
        // clamps the SHARED toi to just above zero to stop the penetration, and
        // that same clamp throttles every other vertex, so the next iteration
        // re-assembles a bit-identical system and the loop spins forever with
        // toi still above FLT_EPSILON. Bound it, and report the stall instead of
        // hanging. (A bound, not a heuristic: no progress window, no tuned
        // epsilon. 0 disables it.)
        if (!final_step && prm.max_newton_steps > 0 &&
            step >= prm.max_newton_steps) {
            logging.message(
                "### newton stalled: no acceptable step after %u iterations "
                "(last toi: %.2e, toi_advanced: %.2e)",
                step, last_toi, toi_advanced);
            logging.message("### an over-constrained configuration cannot be "
                            "advanced: a prescribed pin driven into geometry "
                            "that cannot yield has no way to resolve. Re-author "
                            "the pin's path, or make it a soft pull pin.");
            result.newton_progress = false;
            return result;
        }
        if (final_step) {
            logging.message("------ error reduction step ------");
        } else {
            logging.message("------ newton step %u ------", step);
        }

        dyn_hess.clear();
        diag_hess.clear(Mat3x3f::Zero());
        fixed_hess.clear();
        force.clear();
        if (cg_warmstart && !final_step && step == 1 && dx_warm_seeded) {
            kernels::copy(dx_warm.data, dx.data, 3u * vertex_count);
        } else {
            dx.clear();
        }

        // SAND grains: zero the transient spin accumulators each iteration so the
        // contact embeds (grain-grain point-point + floor/sphere) ACCUMULATE
        // (atomicAdd) the friction torque / angular stiffness / contact-normal
        // sum over ALL of a grain's simultaneous contacts within this iteration,
        // rather than overwrite. The post-solve integrate consumes the converged
        // sums once. Gated so non-SAND scenes launch no extra kernels.
        if (has_grains) {
            data.grain_torque.clear(Vec3f::Zero());
            data.grain_ang_stiff.clear(0.0f);
            data.grain_contact_normal.clear(Vec3f::Zero());
            // Implicit (Schur-condensed) rolling Schur blocks (floor/sphere).
            data.grain_A.clear(Mat3x3f::Zero());
            data.grain_B.clear(Mat3x3f::Zero());
            data.grain_grot.clear(Vec3f::Zero());
        }

        if (final_step) {
            // The step only got `toi_advanced` of the way through its span, so
            // shrink it to what was actually integrated and bring the kinematic
            // fix pins back in step with it before the targets are rebuilt off
            // them.
            rewind_kinematic_fix(toi_advanced);
            dt *= toi_advanced;
            compute_target(dt);
        }

        // Name: Matrix Assembly Time
        // Format: list[(time, ms)]
        // Description:
        // Wall-clock time in milliseconds spent assembling the global
        // system matrix and right-hand side for the Newton linear solve,
        // including inertia, elastic, stitch, strain-limiting, and contact
        // contributions. One entry per Newton iteration.
        logging.push("matrix assembly");

        {
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            Vec<Vec3f> target_vec = target.as_vec();
            Vec<float> dx_vec = dx.as_vec();
            DISPATCH_START(vertex_count)
            [prop_vertex, eval_x_vec, target_vec,
             dx_vec] __device__(unsigned i) mutable {
                if (prop_vertex[i].fix_index > 0) {
                    Map<Vec3f>(dx_vec.data + 3 * i) =
                        (eval_x_vec[i] - target_vec[i]);
                }
            } DISPATCH_END;
        }

        // Pre-pass: compute torque group centroids and PCA axes
        auto torque_result = pool.get<TorqueGroupResult>(
            main_helper::host_dataset.constraint.torque_groups.size > 0
                ? main_helper::host_dataset.constraint.torque_groups.size : 1);
        Vec<TorqueGroupResult> torque_result_vec = torque_result.as_vec();
        energy::compute_torque_groups(data, eval_x, torque_result_vec);

        energy::embed_momentum_force_hessian(data, eval_x, velocity, dt, target,
                                             force, diag_hess, prm,
                                             torque_result_vec);

        // Name: Assembly: Elastic
        // Format: list[(time, ms)]
        // Description:
        // Diagnostic sub-timer of "matrix assembly": wall-clock ms spent
        // assembling the elastic (membrane / bending / solid) Hessian and
        // force into the fixed matrix.
        logging.push("asm elastic");
        energy::embed_elastic_force_hessian(data, eval_x, force, fixed_hess, dt,
                                            prm);
        logging.pop();

        if (host_data.constraint.stitch.size) {
            energy::embed_stitch_force_hessian(data, eval_x, force, fixed_hess,
                                               prm);
        }

        // Name: Assembly: Fixed Copy
        // Format: list[(time, ms)]
        // Description:
        // Diagnostic sub-timer of "matrix assembly": wall-clock ms to snapshot
        // the elastic fixed matrix into tmp_fixed (the contact-stiffness
        // reference read by the contact assembly).
        logging.push("asm copy");
        tmp_fixed.copy(fixed_hess);
        logging.pop();

        // Name: Assembly: Strain Limit
        // Format: list[(time, ms)]
        // Description:
        // Diagnostic sub-timer of "matrix assembly": wall-clock ms for the
        // strain-limiting Hessian / force contributions.
        logging.push("asm strainlimit");
        if (data.shell_face_count > 0) {
            strainlimiting::embed_strainlimiting_force_hessian(
                data, eval_x, force, tmp_fixed, fixed_hess, prm);
        }
        if (data.rod_count > 0) {
            strainlimiting::embed_rod_strainlimiting_force_hessian(
                data, eval_x, force, tmp_fixed, fixed_hess, prm);
        }
        logging.pop();
        unsigned num_contact = 0;
        float dyn_consumed = 0.0f;
        unsigned max_nnz_row = 0;
        // Name: Assembly: Contact
        // Format: list[(time, ms)]
        // Description:
        // Diagnostic sub-timer of "matrix assembly": wall-clock ms for the
        // self-contact / collision-mesh Hessian + force assembly (the CSR fill
        // path, including the dynamic-matrix rebuild).
        logging.push("asm contact");
        if (!param->disable_contact) {
            num_contact += contact::embed_contact_force_hessian(
                data, eval_x, force, tmp_fixed, fixed_hess, dyn_hess,
                max_nnz_row, dyn_consumed, dt, prm);
        }
        logging.pop();

        // Name: Dynamic Hessian Memory Usage Ratio
        // Format: list[(time, ratio)]
        // Description:
        // Fraction of the pre-allocated dynamic contact-Hessian GPU buffer
        // actually used during matrix assembly, as a value in [0, 1]. If
        // this ratio reaches 1.0, the solver is out of space for new
        // contact entries and the simulation will fail. Monitor this to
        // size csrmat_max_nnz correctly. Only contact contributions count
        // against this budget, not elastic or inertia terms.
        logging.mark("dyn_consumed", dyn_consumed);

        // Name: Max Non-Zero Entries Per Contact Matrix Row
        // Format: list[(time, count)]
        // Description:
        // Widest single row of the dynamic contact Hessian reserved for this
        // Newton iteration: the columns that row carried over from the
        // previous iteration, plus one slot for every block the counting pass
        // did not find among them. Rows grow wider when a vertex is in
        // contact with many primitives at once, and widest of all where a
        // coarse collider meets a finely sampled deformable, since each
        // collider vertex then couples to a large number of fine ones.
        //
        // This is a reservation, so it is an upper bound on the row's final
        // non-zero count rather than that count itself: two contacts that
        // contribute the same new column are counted separately here, and
        // fold together only when the row is compressed at the end of
        // assembly. That makes it the right number for sizing
        // csrmat-max-nnz, which is what the same reservation is charged
        // against (see "dyn_consumed" above).
        logging.mark("max_nnz_row", max_nnz_row);

        num_contact += contact::embed_constraint_force_hessian(
            data, eval_x, force, tmp_fixed, fixed_hess, dt, prm);

        // Name: Total Contact Count
        // Format: list[(time, count)]
        // Description:
        // Total number of active contact and constraint pairs assembled
        // into the system matrix for this Newton iteration, summed across
        // self-contact, collision-mesh contact, and analytic constraints
        // (sphere, floor). A useful proxy for how crowded the collision
        // scene is at this iteration.
        logging.mark("num_contact", num_contact);
        logging.pop();

        // Implicit (Schur-condensed) rolling: condense each grain's angular DOF
        // (from its floor/sphere friction, accumulated into grain_A/B/grot) into
        // its 3x3 translation block (diag_hess) and RHS (force) BEFORE the solve,
        // so the linear solve sees the rotation-translation coupling.
        if (has_grains) {
            // Fraction of the grain's spin angular-momentum fed back into its
            // translation (spin-to-translation rolling drive). 0.5 gives near-
            // textbook rolling while staying bounded (no energy-pump runaway);
            // higher approaches textbook but loses stability margin on steep
            // slopes. Tunable via PPF_SAND_SPIN_COUPLE.
            static const float sand_spin_couple = []() {
                const char *s = std::getenv("PPF_SAND_SPIN_COUPLE");
                return s ? std::strtof(s, nullptr) : 0.5f;
            }();
            SandRigid::launch_condense_grains(data, diag_hess.as_vec(),
                                              force.as_vec(), dt,
                                              sand_spin_couple);
        }

        // Dirichlet DOF removal for every masked (fix-pinned) vertex. After the
        // whole Newton system is assembled (this is the last writer, before the
        // solve), turn each such vertex into an exact boundary condition rather
        // than the barrier penalty it used to be. Writing the
        // system as M dx = f with M = A + B + C, and prescribing dx_i = p_i =
        // eval_x[i] - target[i] on the removed vertices, the exact reduction of
        // a FREE row j is
        //
        //     sum_{k free} M_jk dx_k  =  f_j  -  sum_{k removed} M_jk p_k
        //                                        ^^^^^^^^^^^^^^^^^^^^^^^^
        //                                        the Dirichlet LIFTING term
        //
        // so the pass is:
        //   (1) LIFT: before dropping any coupling block M_jk to a removed
        //       vertex k, move its known contribution to the right-hand side of
        //       the free row j. This is what tells a free vertex that the
        //       obstacle is advancing INTO it: the contact Hessian block times
        //       the prescribed vertex's increment is precisely the "get out of
        //       the way by p_k" forcing. Omitting it leaves the cloth with only
        //       the barrier gradient, whose Newton step is bounded by
        //       grad/curv = ghat/2 regardless of stiffness, so the cloth can
        //       never keep pace with a prescribed vertex that moves further than
        //       that in one step: the gap closes monotonically onto the ACCD parking
        //       distance, the line search then clamps toi to ~0 to stop the
        //       penetration, and that same toi also throttles the cloth's escape,
        //       so both sides freeze. (A STATIONARY pin has p_k = 0, which
        //       is why it never showed the bug.)
        //   (2) zero every stored (upper-triangle) block of both Hessians whose
        //       row OR column is a removed vertex. Each stored value aliases the
        //       canonical buffer slot the transpose mirror reads, so one write
        //       clears both SpMV scatter directions -> row i AND column i.
        //   (3) set the removed vertex's diagonal C[i] = I and its RHS
        //       force[i] = p_i, the same increment the dx seed wrote at the top
        //       of the loop.
        // Both matrices store only the upper triangle, so a coupling appears
        // once, in the row with the smaller index; whichever of the two rows
        // owns it lifts the OTHER one, hence the two symmetric branches (the
        // stored block M_ij lifts row j via its transpose). The lift scatters
        // across rows, so it accumulates atomically.
        // The block-Jacobi / Schwarz preconditioner is rebuilt from
        // A(i,i)+B(i,i)+C[i] inside solver::solve, so P[i] = invert(I) = I falls
        // out automatically and no PCG kernel or CUDA-graph edit is needed. With
        // seed dx[i] = force[i] the initial residual r[i] = 0, and the zeroed
        // column keeps (A p)[i] = 0 every iteration, so dx[i] is held exactly
        // regardless of cloth forces or the shared PCG scalars. A prescribed vertex
        // still reaches its keyframe through the ordinary line-search CCD path
        // (dx[i] = p_i is a genuine swept direction), so a fast prescribed vertex
        // cannot tunnel free geometry. Skipped when nothing is masked.
        if (n_dof_removed > 0) {
            logging.push("asm dirichlet");
            const unsigned *dof_mask = dof_removed_mask.data;
            Row *elim_rows = dyn_hess.rows.data;
            const unsigned *elim_fx_off = fixed_hess.index.offset;
            const unsigned *elim_fx_col = fixed_hess.index.data;
            Mat3x3f *elim_fx_val = fixed_hess.value.data;
            Mat3x3f *elim_diag = diag_hess.as_vec().data;
            float *elim_force = force.as_vec().data;
            const Vec3f *elim_eval_x = eval_x.as_vec().data;
            const Vec3f *elim_target = target.as_vec().data;
            DISPATCH_START(vertex_count)
            [dof_mask, elim_rows, elim_fx_off, elim_fx_col, elim_fx_val,
             elim_diag, elim_force, elim_eval_x,
             elim_target] __device__(unsigned i) mutable {
                const bool mask_i = dof_mask[i] != 0u;
                // Prescribed increment of a removed vertex (zero for a free one,
                // so the lift below vanishes on free-free couplings).
                auto prescribed = [&](unsigned v) -> Vec3f {
                    return (elim_eval_x[v] - elim_target[v]).cast<float>();
                };
                // f_row -= blk * p   (row is free), scattered atomically because
                // the lift for a free row can come from several removed columns
                // AND from removed rows lifting it by transpose.
                auto lift_sub = [&](unsigned row_v, const Mat3x3f &blk,
                                    const Vec3f &p, bool transposed) {
                    Vec3f c;
                    if (transposed) {
                        // blk^T * p, written out (device matvec on the in-house
                        // Mat3x3f is column-major; spell it to stay explicit).
                        c[0] = blk(0, 0) * p[0] + blk(1, 0) * p[1] +
                               blk(2, 0) * p[2];
                        c[1] = blk(0, 1) * p[0] + blk(1, 1) * p[1] +
                               blk(2, 1) * p[2];
                        c[2] = blk(0, 2) * p[0] + blk(1, 2) * p[1] +
                               blk(2, 2) * p[2];
                    } else {
                        c[0] = blk(0, 0) * p[0] + blk(0, 1) * p[1] +
                               blk(0, 2) * p[2];
                        c[1] = blk(1, 0) * p[0] + blk(1, 1) * p[1] +
                               blk(1, 2) * p[2];
                        c[2] = blk(2, 0) * p[0] + blk(2, 1) * p[1] +
                               blk(2, 2) * p[2];
                    }
                    atomicAdd(elim_force + 3 * row_v + 0, -c[0]);
                    atomicAdd(elim_force + 3 * row_v + 1, -c[1]);
                    atomicAdd(elim_force + 3 * row_v + 2, -c[2]);
                };
                // (1a)+(2a) dynamic (contact) matrix row i.
                Row &row = elim_rows[i];
                for (unsigned k = 0; k < row.head; ++k) {
                    const unsigned j = row.index[k];
                    const bool mask_j = dof_mask[j] != 0u;
                    if (mask_i != mask_j) {
                        // Exactly one side is prescribed: lift the free row.
                        if (mask_j) {
                            lift_sub(i, row.value[k], prescribed(j), false);
                        } else {
                            lift_sub(j, row.value[k], prescribed(i), true);
                        }
                    }
                    if (mask_i || mask_j) {
                        row.value[k] = Mat3x3f::Zero();
                    }
                }
                // (1b)+(2b) fixed (elastic/stitch/strain-limit/fix-pin) row i.
                const unsigned b0 = elim_fx_off[i], b1 = elim_fx_off[i + 1];
                for (unsigned k = b0; k < b1; ++k) {
                    const unsigned j = elim_fx_col[k];
                    const bool mask_j = dof_mask[j] != 0u;
                    if (mask_i != mask_j) {
                        if (mask_j) {
                            lift_sub(i, elim_fx_val[k], prescribed(j), false);
                        } else {
                            lift_sub(j, elim_fx_val[k], prescribed(i), true);
                        }
                    }
                    if (mask_i || mask_j) {
                        elim_fx_val[k] = Mat3x3f::Zero();
                    }
                }
            } DISPATCH_END;
            // (3) The removed rows become the identity with the prescribed RHS.
            // Done in a second pass so it cannot race the atomic lifts above
            // (a removed row's force must NOT accumulate any lift).
            DISPATCH_START(vertex_count)
            [dof_mask, elim_diag, elim_force, elim_eval_x,
             elim_target] __device__(unsigned i) mutable {
                if (dof_mask[i] != 0u) {
                    elim_diag[i] = Mat3x3f::Identity();
                    Map<Vec3f>(elim_force + 3 * i) =
                        (elim_eval_x[i] - elim_target[i]).cast<float>();
                }
            } DISPATCH_END;
            logging.pop();
        }

        unsigned iter;
        float reresid;
        unsigned schwarz_fallback = 0;

        // Name: Linear Solve Time
        // Format: list[(time, ms)]
        // Map: pcg_linsolve
        // Description:
        // Wall-clock time in milliseconds spent in the preconditioned
        // conjugate gradient (PCG) linear solve for the Newton step
        // direction. One entry per Newton iteration. Typically the
        // dominant per-iteration cost.
        logging.push("linsolve");

        Vec<Vec3f> eval_x_positions = eval_x.as_vec();
        Vec<float> pdrd_dtheta_vec =
            rigid_pdrd ? pdrd_dtheta.as_vec() : Vec<float>{};
        // Env-gated (PPF_DUMP_LINSYS=<k>) one-shot dump of the assembled Newton
        // system for offline analysis. No-op unless the env var is set. Placed
        // after assembly, before the solve.
        if (!rigid_pdrd) {
            dump_linsys::maybe_dump(dyn_hess, fixed_hess, diag_hess, force);
        }

        bool success =
            solver::solve(dyn_hess, fixed_hess, diag_hess, force, prm.cg_tol,
                          prm.cg_max_iter, dx, eval_x_positions, prm, iter,
                          reresid, schwarz_fallback, data, dt, pdrd_dtheta_vec);
        logging.pop();

        // Save the converged first-Newton search direction for next frame's
        // warm-start. dx is only read (never mutated) after the solve, so this
        // snapshot is the full Newton direction before the toi rescale/apply.
        if (cg_warmstart && success && !final_step && step == 1) {
            kernels::copy(dx.data, dx_warm.data, 3u * vertex_count);
            dx_warm_seeded = true;
        }

        // Name: Linear Solve Iteration Count
        // Format: list[(time, iterations)]
        // Map: pcg_iter
        // Description:
        // Number of preconditioned conjugate gradient (PCG) iterations
        // consumed during the linear solve for this Newton iteration.
        // High values indicate an ill-conditioned system or a tight
        // tolerance and often correlate with long linear-solve times.
        logging.mark("iter", iter);

        // Name: Linear Solve Relative Residual
        // Format: list[(time, ratio)]
        // Map: pcg_resid
        // Description:
        // Final relative residual reached by the PCG linear solve for this
        // Newton iteration. When this stays well below the configured
        // tolerance, the solve converged cleanly, values close to the
        // tolerance indicate the iteration cap was hit. In a scene with rigid
        // (PDRD) bodies the reduced solve measures each degree-of-freedom group
        // against its own initial residual, and this reports the worst group,
        // so a body and the cloth cannot mask each other.
        logging.mark("reresid", reresid);

        // Name: Schwarz Block-Jacobi Fallback
        // Format: list[(time, count)]
        // Description:
        // 1 if the solver fell back from the Schwarz preconditioner to the
        // SPD-safe block-Jacobi base for this Newton iteration's PCG solve, else
        // 0. Two causes both latch it: a non-SPD Schwarz residual (rz <= 0), or
        // the memory guard degrading to block-Jacobi because the Schwarz build
        // would not fit PPF_SCHWARZ_MEM_FRAC of free VRAM. Always 0 under the
        // block-jacobi preconditioner; a nonzero entry is worth reviewing.
        // Recorded every iteration but only printed when nonzero so the common 0
        // case does not clutter the log.
        logging.mark("schwarz_fallback", schwarz_fallback, schwarz_fallback != 0);

        if (!success) {
            logging.message("### cg failed");
            result.pcg_success = false;
            // PooledVec buffers will auto-release when returning
            return result;
        }

        // Implicit (Schur-condensed) rolling: recover each grain's angular
        // velocity from the solved translation increment via Schur back-substitution
        // (omega = -A^-1 (g_theta - B^T dx) / dt). Uses the raw solve direction dx
        // (the line-search toi-rescale is not applied to the recovered spin; a
        // documented small-toi approximation).
        if (has_grains) {
            SandRigid::launch_recover_grains(data, dx.as_vec(), dt);
        }

        float max_dx;
        {
            auto tmp_scalar = pool.get<float>(vertex_count);
            tmp_scalar.clear();
            Vec<float> dx_vec = dx.as_vec();
            Vec<float> tmp_scalar_vec = tmp_scalar.as_vec();
            DISPATCH_START(vertex_count)
            [dx_vec, tmp_scalar_vec] __device__(unsigned i) mutable {
                tmp_scalar_vec[i] = Map<Vec3f>(dx_vec.data + 3 * i).norm();
            } DISPATCH_END;
            max_dx = kernels::max_array(tmp_scalar.data, vertex_count, 0.0f);
        }

        // Name: Max Search Direction Magnitude
        // Format: list[(time, meters)]
        // Map: max_search_dir
        // Description:
        // Maximum per-vertex magnitude (L2 norm) of the Newton search
        // direction returned by the linear solve for this Newton
        // iteration, in meters. Compared against the max_dx parameter to
        // decide whether the search direction must be rescaled before
        // the line search.
        logging.mark("max_dx", max_dx);
        float toi_recale = fminf(1.0f, prm.max_dx / max_dx);

        // Name: Search Direction Rescale Factor
        // Format: list[(time, ratio)]
        // Description:
        // Scalar in (0, 1] applied to the Newton search direction before
        // the line search, so that no per-vertex displacement exceeds the
        // configured max_dx. A value of 1.0 means the direction was
        // already within budget, smaller values clamp an over-eager step.
        logging.mark("toi_recale", toi_recale);

        // Reuse the target buffer as scratch for the old eval_x.
        // This is safe because target won't be needed again until the
        // next iteration, where it will be recomputed if necessary.
        kernels::copy(eval_x.data, target.data, target.size);
        {
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            Vec<float> dx_vec = dx.as_vec();
            DISPATCH_START(vertex_count)
            [eval_x_vec, toi_recale, dx_vec] __device__(unsigned i) mutable {
                eval_x_vec[i] -= (toi_recale * Map<Vec3f>(dx_vec.data + 3 * i))
                                     ;
            } DISPATCH_END;
        }

        if (param->fix_xz) {
            {
                Vec<Vec3f> eval_x_vec = eval_x.as_vec();
                const unsigned *dof_mask = dof_removed_mask.data;
                DISPATCH_START(vertex_count)
                [eval_x_vec, vertex_prev, dof_mask,
                 fix_xz_val] __device__(unsigned i) mutable {
                    // A DOF-removed vertex is prescribed exactly; the
                    // fix_xz drag must not nudge its x/z off the keyframe.
                    if (dof_mask[i] != 0u) {
                        return;
                    }
                    if (eval_x_vec[i][1] > float(fix_xz_val)) {
                        float y = fminf(1.0f, eval_x_vec[i][1] -
                                                      float(fix_xz_val));
                        Vec3f z = vertex_prev[i];
                        eval_x_vec[i][0] -= y * (eval_x_vec[i][0] - z[0]);
                        eval_x_vec[i][2] -= y * (eval_x_vec[i][2] - z[2]);
                    }
                } DISPATCH_END;
            }
        }

        if (!param->disable_contact) {
            logging.push("aabb update");
            Vec<Vec3f> target_vec = target.as_vec();
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            lbvh::update_face_aabb(target_vec, eval_x_vec,
                                   prm.line_search_max_t, data.mesh.mesh.face,
                                   bvh_storage::get_bvh().face, contact::get_face_aabb(),
                                   data.prop.face, data.param_arrays.face);
            lbvh::update_edge_aabb(target_vec, eval_x_vec,
                                   prm.line_search_max_t, data.mesh.mesh.edge,
                                   bvh_storage::get_bvh().edge, contact::get_edge_aabb(),
                                   data.prop.edge, data.param_arrays.edge);
            lbvh::update_vertex_aabb(
                target_vec, eval_x_vec, prm.line_search_max_t, bvh_storage::get_bvh().vertex,
                contact::get_vertex_aabb(), host_data.surface_vert_count,
                data.prop.vertex, data.param_arrays.vertex);
                invalidate_inactive_aabbs();
                logging.pop();
        }
        // Name: Line Search Time
        // Format: list[(time, ms)]
        // Description:
        // Wall-clock time in milliseconds spent in the per-iteration
        // line search, which runs continuous collision detection (CCD)
        // plus strain-limit CCD to find the largest feasible substep
        // along the rescaled search direction. One entry per Newton
        // iteration.
        logging.push("line search");
        float SL_toi = 1.0f;
        float toi = 1.0f;
        // A prescribed (fix-pinned) vertex cannot yield to an analytic collider:
        // it has no DOF to give. The line search flags the smallest such vertex
        // index here rather than clamping the shared toi (which would only stall
        // the solve without preventing anything). UINT_MAX means "feasible".
        auto pin_infeasible = pool.get<unsigned>(1u);
        pin_infeasible.clear(0xFFFFFFFFu);
        toi = fminf(toi, contact::line_search(data, target, eval_x, prm,
                                              pin_infeasible.data));
        {
            unsigned bad_vert = 0xFFFFFFFFu;
            CUDA_HANDLE_ERROR(cudaMemcpy(&bad_vert, pin_infeasible.data,
                                         sizeof(unsigned),
                                         cudaMemcpyDeviceToHost));
            if (bad_vert != 0xFFFFFFFFu) {
                logging.message(
                    "### infeasible pin: prescribed vertex %u is driven through "
                    "an analytic collider (floor/sphere) it cannot yield to",
                    bad_vert);
                logging.message("### re-author the pin's path so it stays "
                                "outside the collider, or make it a soft pull "
                                "pin so it can yield.");
                // Balance the "line search" scope pushed above before this
                // early return, or ~SimpleLog trips check_empty (a host abort)
                // and the clean crash record is never written.
                logging.pop();
                result.pin_feasible = false;
                return result;
            }
        }
        if (contact::ccd_overlap_detected()) {
            unsigned ov0 = 0xFFFFFFFFu, ov1 = 0xFFFFFFFFu, okind = 0xFFFFFFFFu;
            float od2 = -1.0f, ooffset = -1.0f;
            contact::ccd_overlap_info(ov0, ov1, okind, od2, ooffset);
            const char *okind_str =
                okind == 0u   ? "vertex-face"
                : okind == 1u ? "edge-edge"
                : okind == 2u ? "point-point"
                : okind == 3u ? "vertex-face (collision mesh)"
                : okind == 4u ? "face-vertex (collision mesh)"
                : okind == 5u ? "edge-edge (collision mesh)"
                              : "unknown";
            // A negative offset is the unset value: the flag came from a
            // collapsed sweep frame, which has no scale to report lengths in.
            // The distance and offset belong to a flagged pair, which need not
            // be the named one, since the indices are first-writer-wins and
            // these two are last-writer.
            if (ooffset < 0.0f) {
                logging.message(
                    "### contact starts overlapping: the sweep frame of a %s "
                    "pair collapsed, so two primitives are coincident to the "
                    "resolution of the coordinates. offending pair: vertices "
                    "%u and %u.",
                    okind_str, ov0, ov1);
            } else {
                logging.message(
                    "### contact starts overlapping: two surfaces begin the "
                    "step already touching or overlapping (a contact pair is "
                    "inside the contact offset at the start of the step). "
                    "offending %s pair: vertices %u and %u. A flagged pair "
                    "has squared start distance %.6e against offset %.6e, in "
                    "the CCD's rescaled units.",
                    okind_str, ov0, ov1, od2, ooffset);
            }
            if (okind >= 3u && okind <= 5u) {
                logging.message(
                    "### the second index is in the static collision-mesh "
                    "vertex space; the first is a dynamic vertex.");
            }
            logging.message("### give the initial geometry a small clearance so "
                            "nothing starts in contact, or check whether a "
                            "stitch or pin is pulling elements together faster "
                            "than contact can resolve.");
            // Balance the "line search" scope (see the pin path above) before
            // returning, so ~SimpleLog's check_empty does not abort the run.
            logging.pop();
            result.contact_separated = false;
            return result;
        }
        if (shell_face_count > 0) {
            auto tmp_scalar = pool.get<float>(shell_face_count);
            SL_toi = strainlimiting::line_search(data, eval_x, target,
                                                 tmp_scalar, prm);
            toi = fminf(toi, SL_toi);
            // Name: Strain-Limit Time of Impact
            // Format: list[(time, ratio)]
            // Description:
            // Fraction in (0, 1] of the rescaled search direction that can
            // be taken without violating the configured shell or rod
            // strain limits, as returned by the strain-limiting line
            // search. A value of 1.0 means strain limits never bound the
            // step, smaller values mean the strain limiter clamped it.
            logging.mark("SL_toi", SL_toi);
        }
        if (rod_count > 0) {
            auto tmp_scalar = pool.get<float>(rod_count);
            float SL_rod_toi = strainlimiting::rod_line_search(
                data, eval_x, target, tmp_scalar, prm);
            SL_toi = fminf(SL_toi, SL_rod_toi);
            toi = fminf(toi, SL_rod_toi);
            logging.mark("SL_rod_toi", SL_rod_toi);
        }
        logging.pop();

        // Name: Line Search Time of Impact
        // Format: list[(time, ratio)]
        // Description:
        // Fraction in (0, 1] of the rescaled Newton search direction that
        // can be taken without causing a collision or violating strain
        // limits, as the minimum of the contact CCD result and the
        // strain-limit TOI. A value of 1.0 means the full Newton step was
        // accepted, smaller values mean the line search cut it short.
        logging.mark("toi", toi);
        last_toi = toi;
        if (toi <= std::numeric_limits<float>::epsilon()) {
            logging.message("### ccd failed (toi: %.2e)", toi);
            if (SL_toi < 1.0f) {
                logging.message("strain limiting toi: %.2e", SL_toi);
            }
            result.ccd_success = false;
            // PooledVec buffers will auto-release when returning
            return result;
        }

        if (!final_step) {
            toi_advanced += std::max(0.0, 1.0 - toi_advanced) *
                            static_cast<double>(toi_recale * toi);
        }
        logging.message("* toi_advanced: %.2e", toi_advanced);

        {
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            Vec<Vec3f> target_vec = target.as_vec();
            DISPATCH_START(vertex_count)
            [eval_x_vec, target_vec, toi] __device__(unsigned i) mutable {
                Vec3f d = toi * (eval_x_vec[i] - target_vec[i]);
                eval_x_vec[i] = target_vec[i] + d;
            } DISPATCH_END;
        }

        // PDRD rigidify: snap the PDRD surface onto the nearest exactly-rigid
        // configuration. eval_x is collision-free here; the snap (eval_x ->
        // rigid_tgt) is treated as a trajectory and run through the contact CCD
        // line search, so it can never introduce a penetration. A partial step
        // (toi_rig < 1) leaves a small non-rigid residual that the next Newton
        // iterations remove; toi_rig = 0 is always feasible (identity snap).
        if (rigid_pdrd) {
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            Vec<Vec3f> rigid_tgt_vec = rigid_tgt.as_vec();
            // Integrate this iteration's actually-applied rotation
            // (toi_recale*toi * reduced dtheta) onto the persistent R_run, then
            // build the rigidify target from R_run (anchored rotation + eval_x
            // centroid) instead of re-fitting the contact-sheared eval_x. This
            // breaks the cross-frame accumulation that drove the non-rigid
            // shrink; the lerp + CCD below are unchanged.
            Vec<float> R_run_vec = R_run.as_vec();
            // The applied per-vertex step eval_x -= toi_recale*toi*(dx_b - p x dth)
            // rotates the body by -(toi_recale*toi)*dth (note the sign from the
            // prolong's -p x dth and the -= update), so R_run integrates the
            // NEGATED scaled reduced rotation.
            PDRD::launch_compose_rrun(n_pdrd, R_run_vec, pdrd_dtheta.as_vec(),
                                      -(toi_recale * toi));
            // rigid_tgt = eval_x, then overwrite PDRD verts with the anchored
            // rigid image (centroid(eval_x) + R_run * ybar).
            kernels::copy(eval_x.data, rigid_tgt.data, eval_x.size);
            PDRD::launch_rigidify_from_rot(data, eval_x_vec, R_run_vec,
                                           rigid_tgt_vec);

            float toi_rig = 1.0f;
            if (!param->disable_contact) {
                logging.push("rigidify ccd");
                lbvh::update_face_aabb(eval_x_vec, rigid_tgt_vec,
                                       prm.line_search_max_t, data.mesh.mesh.face,
                                       bvh_storage::get_bvh().face,
                                       contact::get_face_aabb(), data.prop.face,
                                       data.param_arrays.face);
                lbvh::update_edge_aabb(eval_x_vec, rigid_tgt_vec,
                                       prm.line_search_max_t, data.mesh.mesh.edge,
                                       bvh_storage::get_bvh().edge,
                                       contact::get_edge_aabb(), data.prop.edge,
                                       data.param_arrays.edge);
                lbvh::update_vertex_aabb(
                    eval_x_vec, rigid_tgt_vec, prm.line_search_max_t,
                    bvh_storage::get_bvh().vertex, contact::get_vertex_aabb(),
                    host_data.surface_vert_count, data.prop.vertex,
                    data.param_arrays.vertex);
                invalidate_inactive_aabbs();
                // Same infeasibility flag as the Newton line search above: a
                // prescribed vertex swept through an analytic collider cannot
                // yield, here either. It was consumed (and cleared) above, so
                // re-clear before reusing it.
                pin_infeasible.clear(0xFFFFFFFFu);
                toi_rig = contact::line_search(data, eval_x_vec, rigid_tgt_vec,
                                               prm, pin_infeasible.data);
                unsigned bad_rig = 0xFFFFFFFFu;
                CUDA_HANDLE_ERROR(cudaMemcpy(&bad_rig, pin_infeasible.data,
                                             sizeof(unsigned),
                                             cudaMemcpyDeviceToHost));
                logging.pop();
                if (bad_rig != 0xFFFFFFFFu) {
                    logging.message(
                        "### infeasible pin: prescribed vertex %u is driven "
                        "through an analytic collider during the rigidify "
                        "commit",
                        bad_rig);
                    result.pin_feasible = false;
                    return result;
                }
                if (contact::ccd_overlap_detected()) {
                    unsigned ov0 = 0xFFFFFFFFu, ov1 = 0xFFFFFFFFu,
                             okind = 0xFFFFFFFFu;
                    float od2 = -1.0f, ooffset = -1.0f;
                    contact::ccd_overlap_info(ov0, ov1, okind, od2, ooffset);
                    logging.message(
                        "### contact starts overlapping during the rigidify "
                        "commit: two surfaces begin the step already touching "
                        "or overlapping (kind %u, vertices %u and %u, squared "
                        "start distance %.6e, offset %.6e). Give the "
                        "initial geometry a small clearance, or check a "
                        "stitch/pin pulling elements together faster than "
                        "contact can resolve.",
                        okind, ov0, ov1, od2, ooffset);
                    result.contact_separated = false;
                    return result;
                }
            }
            logging.mark("rigidify_toi", toi_rig);
            {
                DISPATCH_START(vertex_count)
                [eval_x_vec, rigid_tgt_vec, toi_rig] __device__(unsigned i) mutable {
                    Vec3f d = toi_rig *
                              (rigid_tgt_vec[i] - eval_x_vec[i]);
                    eval_x_vec[i] = eval_x_vec[i] + d;
                } DISPATCH_END;
            }
        }

        if (!result.success()) {
            // Early exit - buffers already released in error handling above
            break;
        }

        if (final_step) {
            break;
        } else if (toi_advanced >= param->target_toi &&
                   step >= param->min_newton_steps) {
            final_step = true;
            // target will be recomputed in next iteration, no need to restore
        } else {
            ++step;
            // Restore target for next iteration (since we reused it as
            // tmp_eval_x) target is recomputed from vertex_curr and vertex_prev
            // which are unchanged
            compute_target(dt);
        }
    }

    if (result.success()) {
        if (!param->disable_contact) {
            // Update AABBs for final positions before intersection check
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            logging.push("aabb update");
            lbvh::update_face_aabb(eval_x_vec, eval_x_vec, 1.0f,
                                   data.mesh.mesh.face, bvh_storage::get_bvh().face,
                                   contact::get_face_aabb(), data.prop.face,
                                   data.param_arrays.face);
            lbvh::update_edge_aabb(eval_x_vec, eval_x_vec, 1.0f,
                                   data.mesh.mesh.edge, bvh_storage::get_bvh().edge,
                                   contact::get_edge_aabb(), data.prop.edge,
                                   data.param_arrays.edge);
            lbvh::update_vertex_aabb(
                eval_x_vec, eval_x_vec, 1.0f, bvh_storage::get_bvh().vertex,
                contact::get_vertex_aabb(), host_data.surface_vert_count,
                data.prop.vertex, data.param_arrays.vertex);
                invalidate_inactive_aabbs();
                logging.pop();
            logging.push("check intersection");
            if (!contact::check_intersection(data, eval_x_vec, prm)) {
                logging.message("### intersection detected");
                result.intersection_free = false;
            }
            logging.pop();
        }

        // Name: Advanced Fractional Step Size
        // Format: list[(time, ratio)]
        // Description:
        // Fraction in (0, 1] of the target step size that the Newton loop
        // actually advanced, accumulated across all its iterations. The
        // final Final Step Size equals this fraction times the target dt.
        // A value of 1.0 means the full target step completed, smaller
        // values mean contacts or strain limits forced a partial step.
        logging.mark("toi_advanced", toi_advanced);

        // Name: Newton Iteration Count
        // Format: list[(time, iterations)]
        // Description:
        // Number of Newton iterations consumed in this simulation step
        // (before the trailing error-reduction iteration). Values above
        // the configured min_newton_steps indicate the solver needed
        // extra iterations to reach the target advanced step size.
        logging.mark("newton_steps", step);

        // Name: Final Step Size
        // Format: list[(time, seconds)]
        // Description:
        // Step size in seconds that was actually integrated this
        // simulation step. In easy cases this matches the target dt, but
        // it is reduced by the advanced TOI fraction when contacts or
        // strain limits shorten the step, and can also be reduced when
        // enable_retry is on and the PCG solve fails.
        logging.mark("final_dt", dt);

        param->prev_dt = dt;
        param->time += static_cast<double>(param->prev_dt / param->playback);

        kernels::copy(dev_dataset.vertex.curr.data,
                      dev_dataset.vertex.prev.data,
                      dev_dataset.vertex.prev.size);
        kernels::copy(eval_x.data, dev_dataset.vertex.curr.data,
                      dev_dataset.vertex.curr.size);

        // Carry this frame's integrated rotation to the next frame so the
        // anchored rigidify target stays exact (breaks the non-rigid drift
        // accumulation across frames).
        if (rigid_pdrd) {
            Vec<float> &R_prev = PDRD::pdrd_rprev();
            kernels::copy(R_run.data, R_prev.data, 9u * n_pdrd);
        }

        // SAND grain spin (staggered / post-solve rolling): the converged
        // contact-friction torque (grain_torque, written by the final
        // force/Hessian embed) spins each grain's angular velocity for the next
        // step, which feeds the next step's contact-point friction (contact.cu)
        // and closes the rolling loop. PPF_SAND_NO_ROLL pins omega at 0 so grains
        // slide instead of roll.
        static const bool sand_no_roll = std::getenv("PPF_SAND_NO_ROLL") != nullptr;
        // Rolling-resistance under-roll fraction (anti-pump; see sand_rigid.hpp).
        static const float sand_roll_resist = []() {
            const char *s = std::getenv("PPF_SAND_ROLL_RESIST");
            return s ? std::strtof(s, nullptr) : 0.05f;
        }();
        if (!sand_no_roll && has_grains) {
            SandRigid::launch_integrate_grains(data, dt, /*c_roll=*/0.0f,
                                               sand_roll_resist);
        }

        // Update plasticity (permanent deformation) on B matrices
        if (shell_face_count > 0) {
            plasticity::update_face_plasticity(data, prm);
        }
        if (host_data.mesh.mesh.tet.size > 0) {
            plasticity::update_tet_plasticity(data, prm);
        }
        if (host_data.mesh.mesh.hinge.size > 0) {
            plasticity::update_hinge_plasticity(data, prm);
        }
        if (rod_count > 0) {
            plasticity::update_rod_bend_plasticity(data, prm);
        }

        result.time = param->time;
    }

    // PooledVec buffers auto-release here when exiting function scope
    // No manual release() calls needed!

    // Dynamic GPU alloc/dealloc performed during this step. Settles to 0/0 once
    // the pools warm up (see entry capture above).
    {
        const unsigned long long dev_alloc =
            g_device_alloc_count - dev_alloc_at_entry;
        const unsigned long long dev_free =
            g_device_free_count - dev_free_at_entry;
        logging.mark("device-alloc", static_cast<double>(dev_alloc));
        logging.mark("device-free", static_cast<double>(dev_free));
        logging.message("* device alloc/free this step: %llu / %llu", dev_alloc,
                        dev_free);
    }

    return result;
}

} // namespace main_helper

// Fatal-exit reason set by the exit(1) paths (HandleError in
// cuda_utils.hpp, the no-device check below); the Rust host reads it in an
// atexit hook to write a terminal Crashed{Oom|CudaDriver} record. 0 means
// no fatal exit (a clean run or a panic, which the host handles
// separately).
extern "C" unsigned char g_ppf_fatal_code = 0;

// Device alloc/free instrumentation (declared in main/cuda_utils.hpp). Bumped
// by Vec<T>::alloc/reserve/free so advance() can log the per-step delta and we
// can verify the solve loop performs no dynamic GPU alloc/dealloc in steady
// state.
unsigned long long g_device_alloc_count = 0;
unsigned long long g_device_free_count = 0;

extern "C" DLL_EXPORT unsigned char ppf_fatal_code() {
    return g_ppf_fatal_code;
}

extern "C" DLL_EXPORT void set_log_path(const char *data_dir) {
    SimpleLog::setPath(data_dir);
}

DataSet malloc_dataset(DataSet dataset, ParamSet param) {

    VertexNeighbor dev_vertex_neighbor = {
        mem::malloc_device(dataset.mesh.neighbor.vertex.face),
        mem::malloc_device(dataset.mesh.neighbor.vertex.hinge),
        mem::malloc_device(dataset.mesh.neighbor.vertex.edge),
        mem::malloc_device(dataset.mesh.neighbor.vertex.rod),
    };

    HingeNeighbor dev_hinge_neighbor = {
        mem::malloc_device(dataset.mesh.neighbor.hinge.face)};

    EdgeNeighbor dev_edge_neighbor = {
        mem::malloc_device(dataset.mesh.neighbor.edge.face)};

    MeshInfo dev_mesh_info = //
        {{
             mem::malloc_device(dataset.mesh.mesh.face),
             mem::malloc_device(dataset.mesh.mesh.hinge),
             mem::malloc_device(dataset.mesh.mesh.edge),
             mem::malloc_device(dataset.mesh.mesh.tet),
         },
         {
             dev_vertex_neighbor,
             dev_hinge_neighbor,
             dev_edge_neighbor,
         },
         {
             mem::malloc_device(dataset.mesh.type.face),
             mem::malloc_device(dataset.mesh.type.vertex),
             mem::malloc_device(dataset.mesh.type.hinge),
         }};

    PropSet dev_prop_info = {mem::malloc_device(dataset.prop.vertex),
                             mem::malloc_device(dataset.prop.edge),
                             mem::malloc_device(dataset.prop.face),
                             mem::malloc_device(dataset.prop.hinge),
                             mem::malloc_device(dataset.prop.tet),
                             mem::malloc_device(dataset.prop.pdrd_body)};

    CollisionMesh tmp_collision_mesh = dataset.constraint.mesh;
    {
        tmp_collision_mesh.vertex =
            mem::malloc_device(dataset.constraint.mesh.vertex);
        tmp_collision_mesh.face =
            mem::malloc_device(dataset.constraint.mesh.face);
        tmp_collision_mesh.edge =
            mem::malloc_device(dataset.constraint.mesh.edge);

        tmp_collision_mesh.prop.vertex =
            mem::malloc_device(dataset.constraint.mesh.prop.vertex);
        tmp_collision_mesh.prop.face =
            mem::malloc_device(dataset.constraint.mesh.prop.face);
        tmp_collision_mesh.prop.edge =
            mem::malloc_device(dataset.constraint.mesh.prop.edge);

        tmp_collision_mesh.param_arrays.vertex =
            mem::malloc_device(dataset.constraint.mesh.param_arrays.vertex);
        tmp_collision_mesh.param_arrays.face =
            mem::malloc_device(dataset.constraint.mesh.param_arrays.face);
        tmp_collision_mesh.param_arrays.edge =
            mem::malloc_device(dataset.constraint.mesh.param_arrays.edge);

        tmp_collision_mesh.neighbor.vertex.face =
            mem::malloc_device(dataset.constraint.mesh.neighbor.vertex.face);
        tmp_collision_mesh.neighbor.vertex.hinge =
            mem::malloc_device(dataset.constraint.mesh.neighbor.vertex.hinge);
        tmp_collision_mesh.neighbor.vertex.edge =
            mem::malloc_device(dataset.constraint.mesh.neighbor.vertex.edge);
        tmp_collision_mesh.neighbor.vertex.rod =
            mem::malloc_device(dataset.constraint.mesh.neighbor.vertex.rod);
        tmp_collision_mesh.neighbor.hinge.face =
            mem::malloc_device(dataset.constraint.mesh.neighbor.hinge.face);
        tmp_collision_mesh.neighbor.edge.face =
            mem::malloc_device(dataset.constraint.mesh.neighbor.edge.face);
    }

    Constraint dev_constraint = {
        mem::malloc_device(dataset.constraint.fix),
        mem::malloc_device(dataset.constraint.pull),
        mem::malloc_device(dataset.constraint.torque_groups),
        mem::malloc_device(dataset.constraint.torque_vertices),
        mem::malloc_device(dataset.constraint.sphere),
        mem::malloc_device(dataset.constraint.floor),
        mem::malloc_device(dataset.constraint.stitch),
        tmp_collision_mesh,
    };

    Vec<Mat2x2f> dev_inv_rest2x2 = mem::malloc_device(dataset.inv_rest2x2);
    Vec<Mat3x3f> dev_inv_rest3x3 = mem::malloc_device(dataset.inv_rest3x3);

    VertexSet dev_vertex = {
        mem::malloc_device(dataset.vertex.prev),
        mem::malloc_device(dataset.vertex.curr),
    };

    VecVec<unsigned> dev_fixed_index_table =
        mem::malloc_device(dataset.fixed_index_table);
    VecVec<Vec2u> dev_transpose_table =
        mem::malloc_device(dataset.transpose_table);

    ParamArrays dev_param_arrays = {
        mem::malloc_device(dataset.param_arrays.vertex),
        mem::malloc_device(dataset.param_arrays.edge),
        mem::malloc_device(dataset.param_arrays.face),
        mem::malloc_device(dataset.param_arrays.hinge),
        mem::malloc_device(dataset.param_arrays.tet),
    };

    Vec<unsigned> dev_pdrd_vert_list = mem::malloc_device(dataset.pdrd_vert_list);
    Vec<Vec3f> dev_pdrd_rest_centered =
        mem::malloc_device(dataset.pdrd_rest_centered);
    Vec<Vec3f> dev_grain_omega = mem::malloc_device(dataset.grain_omega);
    Vec<float> dev_grain_inv_inertia =
        mem::malloc_device(dataset.grain_inv_inertia);
    Vec<Vec3f> dev_grain_torque = mem::malloc_device(dataset.grain_torque);
    Vec<float> dev_grain_ang_stiff =
        mem::malloc_device(dataset.grain_ang_stiff);
    Vec<Vec3f> dev_grain_contact_normal =
        mem::malloc_device(dataset.grain_contact_normal);
    Vec<float> dev_grain_inv_inertia_center =
        mem::malloc_device(dataset.grain_inv_inertia_center);
    Vec<Vec3f> dev_grain_omega_prev =
        mem::malloc_device(dataset.grain_omega_prev);
    Vec<Mat3x3f> dev_grain_A = mem::malloc_device(dataset.grain_A);
    Vec<Mat3x3f> dev_grain_B = mem::malloc_device(dataset.grain_B);
    Vec<Vec3f> dev_grain_grot = mem::malloc_device(dataset.grain_grot);
    Vec<unsigned> dev_tet_hess_slots =
        mem::malloc_device(dataset.tet_hess_slots);
    Vec<unsigned> dev_face_hess_slots =
        mem::malloc_device(dataset.face_hess_slots);
    Vec<unsigned> dev_edge_hess_slots =
        mem::malloc_device(dataset.edge_hess_slots);
    Vec<unsigned> dev_hinge_hess_slots =
        mem::malloc_device(dataset.hinge_hess_slots);
    Vec<unsigned> dev_rod_bend_hess_slots =
        mem::malloc_device(dataset.rod_bend_hess_slots);
    Vec<unsigned> dev_stitch_hess_slots =
        mem::malloc_device(dataset.stitch_hess_slots);

    DataSet dev_dataset = {dev_vertex,
                           dev_mesh_info,
                           dev_prop_info,
                           dev_param_arrays,
                           dev_inv_rest2x2,
                           dev_inv_rest3x3,
                           dev_constraint,
                           dev_fixed_index_table,
                           dev_transpose_table,
                           dataset.rod_count,
                           dataset.shell_face_count,
                           dataset.surface_vert_count,
                           dev_pdrd_vert_list,
                           dev_pdrd_rest_centered,
                           dev_grain_omega,
                           dev_grain_inv_inertia,
                           dev_grain_torque,
                           dev_grain_ang_stiff,
                           dev_grain_contact_normal,
                           dev_grain_inv_inertia_center,
                           dev_grain_omega_prev,
                           dev_grain_A,
                           dev_grain_B,
                           dev_grain_grot,
                           dev_tet_hess_slots,
                           dev_face_hess_slots,
                           dev_edge_hess_slots,
                           dev_hinge_hess_slots,
                           dev_rod_bend_hess_slots,
                           dev_stitch_hess_slots};

    return dev_dataset;
}

extern "C" DLL_EXPORT bool initialize(DataSet *dataset, ParamSet *param) {

    int num_device;
    CUDA_HANDLE_ERROR(cudaGetDeviceCount(&num_device));
    logging::info("cuda: detected %d devices...", num_device);
    if (num_device == 0) {
        logging::info("cuda: no device found...");
        g_ppf_fatal_code = 3; // CudaDriver: no usable CUDA device
        exit(1);
    }

    logging::info("cuda: allocating memory...");
    DataSet dev_dataset = malloc_dataset(*dataset, *param);

    return main_helper::initialize(*dataset, dev_dataset, param);
}

extern "C" DLL_EXPORT void advance(StepResult *result) {
    *result = main_helper::advance();
}

extern "C" DLL_EXPORT void fetch() {
    mem::copy_from_device_to_host(main_helper::dev_dataset.vertex.curr.data,
                                  main_helper::host_dataset.vertex.curr.data,
                                  main_helper::host_dataset.vertex.curr.size);
    mem::copy_from_device_to_host(main_helper::dev_dataset.vertex.prev.data,
                                  main_helper::host_dataset.vertex.prev.data,
                                  main_helper::host_dataset.vertex.prev.size);
}

extern "C" DLL_EXPORT void fetch_inv_rest() {
    mem::copy_from_device_to_host(
        main_helper::dev_dataset.inv_rest2x2.data,
        main_helper::host_dataset.inv_rest2x2.data,
        main_helper::host_dataset.inv_rest2x2.size);
    mem::copy_from_device_to_host(
        main_helper::dev_dataset.inv_rest3x3.data,
        main_helper::host_dataset.inv_rest3x3.data,
        main_helper::host_dataset.inv_rest3x3.size);
}

extern "C" DLL_EXPORT void fetch_rest_angles() {
    // Bend plasticity state lives as HingeProp.rest_angle and
    // VertexProp.rest_bend_angle, mutated in-place on GPU by
    // update_hinge_plasticity / update_rod_bend_plasticity. Mirrors
    // fetch_inv_rest: pull the full prop arrays back so save_state
    // serialises the plastic rest angles.
    if (main_helper::host_dataset.prop.hinge.size > 0) {
        mem::copy_from_device_to_host(
            main_helper::dev_dataset.prop.hinge.data,
            main_helper::host_dataset.prop.hinge.data,
            main_helper::host_dataset.prop.hinge.size);
    }
    if (main_helper::host_dataset.prop.vertex.size > 0) {
        mem::copy_from_device_to_host(
            main_helper::dev_dataset.prop.vertex.data,
            main_helper::host_dataset.prop.vertex.data,
            main_helper::host_dataset.prop.vertex.size);
    }
}

extern "C" DLL_EXPORT void fetch_dyn_counts(unsigned *n_value,
                                            unsigned *n_offset) {
    unsigned nrow = tmp::dyn_hess.nrow;
    *n_offset = nrow + 1;
    CUDA_HANDLE_ERROR(cudaMemcpy(n_value,
                                 tmp::dyn_hess.fixed_row_offsets.data + nrow,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
}

extern "C" DLL_EXPORT void fetch_dyn(unsigned *index, Mat3x3f *value,
                                     unsigned *offset) {
    tmp::dyn_hess.fetch(index, value, offset);
}

extern "C" DLL_EXPORT void update_dyn(unsigned *index, unsigned *offset) {
    tmp::dyn_hess.update(index, offset);
}

extern "C" DLL_EXPORT void update_constraint(const Constraint *constraint) {
    main_helper::host_dataset.constraint = *constraint;
    mem::copy_to_device(constraint->fix,
                        main_helper::dev_dataset.constraint.fix);
    mem::copy_to_device(constraint->pull,
                        main_helper::dev_dataset.constraint.pull);
    mem::copy_to_device(constraint->torque_groups,
                        main_helper::dev_dataset.constraint.torque_groups);
    mem::copy_to_device(constraint->torque_vertices,
                        main_helper::dev_dataset.constraint.torque_vertices);
    mem::copy_to_device(constraint->stitch,
                        main_helper::dev_dataset.constraint.stitch);
    mem::copy_to_device(constraint->sphere,
                        main_helper::dev_dataset.constraint.sphere);
    mem::copy_to_device(constraint->floor,
                        main_helper::dev_dataset.constraint.floor);

    // Before overwriting device-side prop arrays below, pull back any fields
    // that the simulation mutates in-place on the GPU (plasticity drift of
    // VertexProp.rest_bend_angle and HingeProp.rest_angle) so those mutations
    // survive the round-trip. Without this, update_constraint (called every
    // frame when pins move) clobbers the plastic rest angles back to their
    // build-time values.
    unsigned vertex_count = main_helper::host_dataset.prop.vertex.size;
    auto &host_vprop = main_helper::host_dataset.prop.vertex;
    if (vertex_count > 0) {
        mem::copy_from_device_to_host(
            main_helper::dev_dataset.prop.vertex.data, host_vprop.data,
            vertex_count);
    }
    unsigned hinge_count_pre = main_helper::host_dataset.prop.hinge.size;
    auto &host_hinge_pre = main_helper::host_dataset.prop.hinge;
    if (hinge_count_pre > 0) {
        mem::copy_from_device_to_host(
            main_helper::dev_dataset.prop.hinge.data, host_hinge_pre.data,
            hinge_count_pre);
    }

    // Rebuild vertex fix_index and pull_index to match the new constraint vectors
    for (unsigned i = 0; i < vertex_count; ++i) {
        host_vprop[i].fix_index = 0;
        host_vprop[i].pull_index = 0;
    }
    for (unsigned i = 0; i < constraint->fix.size; ++i) {
        host_vprop[constraint->fix[i].index].fix_index = i + 1;
    }
    for (unsigned i = 0; i < constraint->pull.size; ++i) {
        host_vprop[constraint->pull[i].index].pull_index = i + 1;
    }
    mem::copy_to_device(host_vprop, main_helper::dev_dataset.prop.vertex);

    // Rebuild element fixed flags based on current pin set.
    // At build time, elements with all vertices pinned get fixed=true
    // to skip energy computation. When pins expire (unpin_time), the
    // element must become unfixed so elastic forces apply again.
    auto &mesh = main_helper::host_dataset.mesh;
    auto is_fixed = [&](unsigned vi) -> bool {
        return host_vprop[vi].fix_index > 0;
    };

    auto &face_prop = main_helper::host_dataset.prop.face;
    for (unsigned i = 0; i < face_prop.size; ++i) {
        auto f = mesh.mesh.face[i];
        face_prop[i].fixed = is_fixed(f[0]) && is_fixed(f[1]) && is_fixed(f[2]);
    }
    mem::copy_to_device(face_prop, main_helper::dev_dataset.prop.face);

    auto &edge_prop = main_helper::host_dataset.prop.edge;
    for (unsigned i = 0; i < edge_prop.size; ++i) {
        auto e = mesh.mesh.edge[i];
        edge_prop[i].fixed = is_fixed(e[0]) && is_fixed(e[1]);
    }
    mem::copy_to_device(edge_prop, main_helper::dev_dataset.prop.edge);

    auto &tet_prop = main_helper::host_dataset.prop.tet;
    for (unsigned i = 0; i < tet_prop.size; ++i) {
        auto t = mesh.mesh.tet[i];
        tet_prop[i].fixed = is_fixed(t[0]) && is_fixed(t[1])
                         && is_fixed(t[2]) && is_fixed(t[3]);
    }
    mem::copy_to_device(tet_prop, main_helper::dev_dataset.prop.tet);

    auto &hinge_prop = main_helper::host_dataset.prop.hinge;
    for (unsigned i = 0; i < hinge_prop.size; ++i) {
        auto h = mesh.mesh.hinge[i];
        hinge_prop[i].fixed = is_fixed(h[0]) && is_fixed(h[1])
                           && is_fixed(h[2]) && is_fixed(h[3]);
    }
    mem::copy_to_device(hinge_prop, main_helper::dev_dataset.prop.hinge);
}

extern "C" DLL_EXPORT void update_rest_shape(const RestShapeUpdate *update) {
    // Replace the device-side inverse rest matrices with the streamed
    // time-varying rest shape for this frame. The elastic force/Hessian
    // kernels read inv_rest2x2/inv_rest3x3 fresh each Newton iteration, so
    // overwriting them here (right after update_constraint, before the next
    // advance) drives each element's rest pose per frame. Empty arrays (a
    // tet-only solid has no faces, a shell-only cloth has no tets) are
    // skipped. Plasticity does not coexist with a streamed rest shape (the
    // frontend refuses to ship both, leaving plasticity == 0 for these
    // elements), so there is no in-place mutation to preserve here.
    if (update->inv_rest2x2.size) {
        mem::copy_to_device(update->inv_rest2x2,
                            main_helper::dev_dataset.inv_rest2x2);
    }
    if (update->inv_rest3x3.size) {
        mem::copy_to_device(update->inv_rest3x3,
                            main_helper::dev_dataset.inv_rest3x3);
    }
    // Exclude near-singular rest elements from the elastic/strain energy via
    // the dedicated `rest_excluded` flag (energy.cu and strainlimiting.cu gate
    // on it independently of the pin-driven `fixed` flag, so the two never
    // alias). This path owns `rest_excluded`: it assigns the full per-element
    // mask every frame, so there is no stale state and no ordering dependence
    // on update_constraint (which only touches `fixed`).
    auto &face_prop = main_helper::host_dataset.prop.face;
    if (update->exclude_face.size) {
        for (unsigned i = 0; i < update->exclude_face.size && i < face_prop.size; ++i) {
            face_prop[i].rest_excluded = update->exclude_face[i] != 0;
        }
        mem::copy_to_device(face_prop, main_helper::dev_dataset.prop.face);
    }
    auto &tet_prop = main_helper::host_dataset.prop.tet;
    if (update->exclude_tet.size) {
        for (unsigned i = 0; i < update->exclude_tet.size && i < tet_prop.size; ++i) {
            tet_prop[i].rest_excluded = update->exclude_tet[i] != 0;
        }
        mem::copy_to_device(tet_prop, main_helper::dev_dataset.prop.tet);
    }
}

extern "C" DLL_EXPORT void override_velocity(
    const unsigned *indices, unsigned count,
    float vx, float vy, float vz, float dt
) {
    if (count == 0 || dt <= 0.0f) return;

    // Upload index array to device
    unsigned *d_indices;
    cudaMalloc(&d_indices, count * sizeof(unsigned));
    cudaMemcpy(d_indices, indices, count * sizeof(unsigned),
               cudaMemcpyHostToDevice);

    auto dev_curr = main_helper::dev_dataset.vertex.curr.data;
    auto dev_prev = main_helper::dev_dataset.vertex.prev.data;
    float fp_vx(vx * dt), fp_vy(vy * dt), fp_vz(vz * dt);

    DISPATCH_START(count)
    [d_indices, dev_curr, dev_prev, fp_vx, fp_vy, fp_vz] __device__(unsigned i) mutable {
        unsigned vi = d_indices[i];
        dev_prev[vi] = Vec3f(
            dev_curr[vi][0] - fp_vx,
            dev_curr[vi][1] - fp_vy,
            dev_curr[vi][2] - fp_vz
        );
    }
    DISPATCH_END

    cudaFree(d_indices);
}

// Gather the CURRENT positions of `indices` (start-of-step `vertex.curr`)
// into a contiguous host buffer `out` packed (x, y, z) per index. Used by
// the angular velocity overwrite: the Rust side computes the body's
// principal axes from these live positions each time a keyframe fires, so
// the spin axis tracks the simulated (rotated / deformed) geometry rather
// than a pose frozen at t=0.
extern "C" DLL_EXPORT void gather_current_positions(
    const unsigned *indices, unsigned count, float *out
) {
    if (count == 0) return;

    unsigned *d_indices;
    cudaMalloc(&d_indices, count * sizeof(unsigned));
    cudaMemcpy(d_indices, indices, count * sizeof(unsigned),
               cudaMemcpyHostToDevice);

    float *d_out;
    cudaMalloc(&d_out, count * 3 * sizeof(float));

    auto dev_curr = main_helper::dev_dataset.vertex.curr.data;

    DISPATCH_START(count)
    [d_indices, d_out, dev_curr] __device__(unsigned i) mutable {
        unsigned vi = d_indices[i];
        d_out[i * 3 + 0] = static_cast<float>(dev_curr[vi][0]);
        d_out[i * 3 + 1] = static_cast<float>(dev_curr[vi][1]);
        d_out[i * 3 + 2] = static_cast<float>(dev_curr[vi][2]);
    }
    DISPATCH_END

    cudaMemcpy(out, d_out, count * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_out);
}

// Inject an angular velocity ω (rad/s) about world-space center `c` into the
// listed vertices: the implicit integrator reads the incoming velocity as
// (curr - prev)/dt, so subtracting (ω × (curr - c)) · dt from `prev` adds a
// rigid spin field. Applied AFTER `override_velocity` in the same step, so a
// keyframe carrying both linear and angular components yields a full
// rigid-velocity overwrite prev = curr - (v_lin + ω × (x - c)) · dt. For a
// PDRD body the field is exactly rigid and survives the rigidify projection;
// for a deformable solid/shell it seeds a rotational velocity field.
extern "C" DLL_EXPORT void override_angular_velocity(
    const unsigned *indices, unsigned count,
    float wx, float wy, float wz,
    float cx, float cy, float cz, float dt
) {
    if (count == 0 || dt <= 0.0f) return;

    unsigned *d_indices;
    cudaMalloc(&d_indices, count * sizeof(unsigned));
    cudaMemcpy(d_indices, indices, count * sizeof(unsigned),
               cudaMemcpyHostToDevice);

    auto dev_curr = main_helper::dev_dataset.vertex.curr.data;
    auto dev_prev = main_helper::dev_dataset.vertex.prev.data;

    DISPATCH_START(count)
    [d_indices, dev_curr, dev_prev, wx, wy, wz, cx, cy, cz, dt]
    __device__(unsigned i) mutable {
        unsigned vi = d_indices[i];
        float rx = static_cast<float>(dev_curr[vi][0]) - cx;
        float ry = static_cast<float>(dev_curr[vi][1]) - cy;
        float rz = static_cast<float>(dev_curr[vi][2]) - cz;
        // v = ω × r
        float vwx = wy * rz - wz * ry;
        float vwy = wz * rx - wx * rz;
        float vwz = wx * ry - wy * rx;
        dev_prev[vi] = Vec3f(
            dev_prev[vi][0] - (vwx * dt),
            dev_prev[vi][1] - (vwy * dt),
            dev_prev[vi][2] - (vwz * dt)
        );
    }
    DISPATCH_END

    cudaFree(d_indices);
}

extern "C" DLL_EXPORT void init_collision_windows(
    const unsigned *vert_dmap, unsigned vert_count,
    const float *windows, const unsigned *window_counts,
    unsigned n_groups
) {
    if (cw_data.d_vert_dmap) cudaFree(cw_data.d_vert_dmap);
    if (cw_data.d_windows) cudaFree(cw_data.d_windows);
    if (cw_data.d_window_counts) cudaFree(cw_data.d_window_counts);
    if (cw_data.d_vert_active) cudaFree(cw_data.d_vert_active);
    if (cw_data.d_face_active) cudaFree(cw_data.d_face_active);
    if (cw_data.d_edge_active) cudaFree(cw_data.d_edge_active);

    cw_data.vert_count = vert_count;
    cw_data.face_count = 0;
    cw_data.edge_count = 0;
    cw_data.n_groups = n_groups;

    cudaMalloc(&cw_data.d_vert_dmap, vert_count * sizeof(unsigned));
    cudaMemcpy(cw_data.d_vert_dmap, vert_dmap, vert_count * sizeof(unsigned), cudaMemcpyHostToDevice);

    unsigned win_size = n_groups * MAX_COLLISION_WINDOWS * 2;
    cudaMalloc(&cw_data.d_windows, win_size * sizeof(float));
    cudaMemcpy(cw_data.d_windows, windows, win_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&cw_data.d_window_counts, n_groups * sizeof(unsigned));
    cudaMemcpy(cw_data.d_window_counts, window_counts, n_groups * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc(&cw_data.d_vert_active, vert_count * sizeof(bool));
    cudaMemset(cw_data.d_vert_active, 1, vert_count * sizeof(bool));
    // face/edge active arrays allocated lazily in refresh_collision_active
    cw_data.d_face_active = nullptr;
    cw_data.d_edge_active = nullptr;

    cw_data.initialized = true;
}

extern "C" DLL_EXPORT void refresh_collision_active(float time) {
    if (!cw_data.initialized) return;

    // Lazy alloc face/edge active arrays (sizes known only after first BVH build)
    unsigned fc = main_helper::dev_dataset.mesh.mesh.face.size;
    unsigned ec = main_helper::dev_dataset.mesh.mesh.edge.size;
    if (cw_data.face_count != fc) {
        if (cw_data.d_face_active) cudaFree(cw_data.d_face_active);
        cw_data.face_count = fc;
        cudaMalloc(&cw_data.d_face_active, fc * sizeof(bool));
        cudaMemset(cw_data.d_face_active, 1, fc * sizeof(bool));
    }
    if (cw_data.edge_count != ec) {
        if (cw_data.d_edge_active) cudaFree(cw_data.d_edge_active);
        cw_data.edge_count = ec;
        cudaMalloc(&cw_data.d_edge_active, ec * sizeof(bool));
        cudaMemset(cw_data.d_edge_active, 1, ec * sizeof(bool));
    }

    // Step 1: flag vertices from windows
    auto d_va = cw_data.d_vert_active;
    auto d_dm = cw_data.d_vert_dmap;
    auto d_w = cw_data.d_windows;
    auto d_wc = cw_data.d_window_counts;
    unsigned vc = cw_data.vert_count;

    DISPATCH_START(vc)
    [d_va, d_dm, d_w, d_wc, time] __device__(unsigned i) mutable {
        unsigned dm = d_dm[i];
        unsigned cnt = d_wc[dm];
        bool active = (cnt == 0);
        for (unsigned w = 0; w < cnt; ++w) {
            float ts = d_w[dm * MAX_COLLISION_WINDOWS * 2 + w * 2];
            float te = d_w[dm * MAX_COLLISION_WINDOWS * 2 + w * 2 + 1];
            if (time >= ts && time < te) { active = true; break; }
        }
        d_va[i] = active;
    }
    DISPATCH_END

    // Step 2: propagate to faces, active if ANY vertex is active
    auto d_fa = cw_data.d_face_active;
    auto faces = main_helper::dev_dataset.mesh.mesh.face.data;

    DISPATCH_START(fc)
    [d_fa, d_va, faces] __device__(unsigned i) mutable {
        Vec3u f = faces[i];
        d_fa[i] = d_va[f[0]] || d_va[f[1]] || d_va[f[2]];
    }
    DISPATCH_END

    // Step 3: propagate to edges, active if ANY vertex is active
    auto d_ea = cw_data.d_edge_active;
    auto edges = main_helper::dev_dataset.mesh.mesh.edge.data;

    DISPATCH_START(ec)
    [d_ea, d_va, edges] __device__(unsigned i) mutable {
        Vec2u e = edges[i];
        d_ea[i] = d_va[e[0]] || d_va[e[1]];
    }
    DISPATCH_END
}

const bool *contact::get_vert_collision_active() {
    return cw_data.initialized ? cw_data.d_vert_active : nullptr;
}

const bool *contact::get_edge_collision_active() {
    return cw_data.initialized ? cw_data.d_edge_active : nullptr;
}

const bool *contact::get_face_collision_active() {
    return cw_data.initialized ? cw_data.d_face_active : nullptr;
}

extern "C" DLL_EXPORT unsigned fetch_intersection_records(
    IntersectionRecord *out, unsigned max_count
) {
    unsigned count = std::min(contact::get_intersection_count(), max_count);
    if (count > 0) {
        memcpy(out, contact::get_intersection_records(),
               count * sizeof(IntersectionRecord));
    }
    return count;
}
