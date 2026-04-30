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
#include "mem.hpp"
#include <cassert>
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
// still private to this translation unit, no header declares it.
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

    // Initialize GPU LBVH construction buffers
    // Use max of main mesh and collision mesh sizes
    unsigned collision_mesh_face_count = host_dataset.constraint.mesh.face.size;
    unsigned max_faces = face_count > collision_mesh_face_count ? face_count : collision_mesh_face_count;
    unsigned max_edges = edge_count > collision_mesh_edge_count ? edge_count : collision_mesh_edge_count;
    unsigned max_verts = surface_vert_count > collision_mesh_vert_count ? surface_vert_count : collision_mesh_vert_count;
    lbvh::initialize(max_faces, max_edges, max_verts);

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

    StepResult result;
    result.pcg_success = true;
    result.ccd_success = true;
    result.intersection_free = true;

    DataSet &host_data = host_dataset;
    DataSet data = dev_dataset;
    ParamSet prm = *param;

    const unsigned vertex_count = host_data.vertex.curr.size;
    const unsigned shell_face_count = host_dataset.shell_face_count;
    const unsigned rod_count = host_dataset.rod_count;
    const unsigned tet_count = host_data.mesh.mesh.tet.size;

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
            [prop_face, param_face, svd_vec,
             tmp_scalar_vec] __device__(unsigned i) mutable {
                const FaceProp &prop = prop_face[i];
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
                    Vec3f d = (vertex_curr[edge[1]] - vertex_curr[edge[0]]);
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

    compute_target(dt);

    kernels::copy(data.vertex.curr.data, eval_x.data, eval_x.size);

    double toi_advanced = 0.0f;
    unsigned step(1);
    bool final_step(false);

    // Allocate buffers for Newton loop (auto-release when function exits)
    auto force = pool.get<float>(3 * vertex_count);
    auto dx = pool.get<float>(3 * vertex_count);
    auto diag_hess = pool.get<Mat3x3f>(vertex_count);

    while (true) {
        if (final_step) {
            logging.message("------ error reduction step ------");
        } else {
            logging.message("------ newton step %u ------", step);
        }

        dyn_hess.clear();
        diag_hess.clear(Mat3x3f::Zero());
        fixed_hess.clear();
        force.clear();
        dx.clear();

        if (final_step) {
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

        energy::embed_elastic_force_hessian(data, eval_x, force, fixed_hess, dt,
                                            prm);

        if (host_data.constraint.stitch.size) {
            energy::embed_stitch_force_hessian(data, eval_x, force, fixed_hess,
                                               prm);
        }

        tmp_fixed.copy(fixed_hess);

        if (data.shell_face_count > 0) {
            strainlimiting::embed_strainlimiting_force_hessian(
                data, eval_x, force, tmp_fixed, fixed_hess, prm);
        }
        if (data.rod_count > 0) {
            strainlimiting::embed_rod_strainlimiting_force_hessian(
                data, eval_x, force, tmp_fixed, fixed_hess, prm);
        }
        unsigned num_contact = 0;
        float dyn_consumed = 0.0f;
        unsigned max_nnz_row = 0;
        if (!param->disable_contact) {
            num_contact += contact::embed_contact_force_hessian(
                data, eval_x, force, tmp_fixed, fixed_hess, dyn_hess,
                max_nnz_row, dyn_consumed, dt, prm);
        }

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
        // Largest number of non-zero block entries found in any single row
        // of the dynamic contact Hessian for this Newton iteration. Rows
        // grow wider when a vertex is in contact with many primitives at
        // once. Useful as a diagnostic for crowded contact regions and
        // for sizing the per-row capacity of the dynamic CSR buffer.
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

        unsigned iter;
        float reresid;

        // Name: Linear Solve Time
        // Format: list[(time, ms)]
        // Map: pcg_linsolve
        // Description:
        // Wall-clock time in milliseconds spent in the preconditioned
        // conjugate gradient (PCG) linear solve for the Newton step
        // direction. One entry per Newton iteration. Typically the
        // dominant per-iteration cost.
        logging.push("linsolve");

        bool success =
            solver::solve(dyn_hess, fixed_hess, diag_hess, force, prm.cg_tol,
                          prm.cg_max_iter, dx, iter, reresid);
        logging.pop();

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
        // tolerance indicate the iteration cap was hit.
        logging.mark("reresid", reresid);

        if (!success) {
            logging.message("### cg failed");
            result.pcg_success = false;
            // PooledVec buffers will auto-release when returning
            return result;
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
        float toi_recale = fmin(1.0f, prm.max_dx / max_dx);

        // Name: Search Direction Rescale Factor
        // Format: list[(time, ratio)]
        // Description:
        // Scalar in (0, 1] applied to the Newton search direction before
        // the line search, so that no per-vertex displacement exceeds the
        // configured max_dx. A value of 1.0 means the direction was
        // already within budget, smaller values clamp an over-eager step.
        logging.mark("toi_recale", toi_recale);

        // Reuse target buffer to store old eval_x (previously tmp_eval_x)
        // This is safe because target won't be needed again until next
        // iteration, where it will be recomputed if necessary
        kernels::copy(eval_x.data, target.data, target.size);
        {
            Vec<Vec3f> eval_x_vec = eval_x.as_vec();
            Vec<float> dx_vec = dx.as_vec();
            DISPATCH_START(vertex_count)
            [eval_x_vec, toi_recale, dx_vec] __device__(unsigned i) mutable {
                eval_x_vec[i] -= toi_recale * Map<Vec3f>(dx_vec.data + 3 * i);
            } DISPATCH_END;
        }

        if (param->fix_xz) {
            {
                Vec<Vec3f> eval_x_vec = eval_x.as_vec();
                DISPATCH_START(vertex_count)
                [eval_x_vec, vertex_prev,
                 fix_xz_val] __device__(unsigned i) mutable {
                    if (eval_x_vec[i][1] > fix_xz_val) {
                        float y = fmin(1.0f, eval_x_vec[i][1] -
                                                      fix_xz_val);
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
        toi = fmin(toi, contact::line_search(data, target, eval_x, prm));
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

    return result;
}

} // namespace main_helper

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
                             mem::malloc_device(dataset.prop.tet)};

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
                           dataset.surface_vert_count};

    return dev_dataset;
}

extern "C" DLL_EXPORT bool initialize(DataSet *dataset, ParamSet *param) {

    int num_device;
    CUDA_HANDLE_ERROR(cudaGetDeviceCount(&num_device));
    logging::info("cuda: detected %d devices...", num_device);
    if (num_device == 0) {
        logging::info("cuda: no device found...");
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
    float dx = vx * dt, dy = vy * dt, dz = vz * dt;

    DISPATCH_START(count)
    [d_indices, dev_curr, dev_prev, dx, dy, dz] __device__(unsigned i) mutable {
        unsigned vi = d_indices[i];
        dev_prev[vi] = Vec3f(
            dev_curr[vi][0] - dx,
            dev_curr[vi][1] - dy,
            dev_curr[vi][2] - dz
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
