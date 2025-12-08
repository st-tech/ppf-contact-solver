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
#include "../main/cuda_utils.hpp"
#include "../simplelog/SimpleLog.h"
#include "../solver/solver.hpp"
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

namespace main_helper {
DataSet host_dataset, dev_dataset;
ParamSet *param;

bool initialize(DataSet _host_dataset, DataSet _dev_dataset, ParamSet *_param) {

    // Name: Initialization Time
    // Format: list[(int,ms)]
    // Description:
    // Time consumed for the initialization of the simulation.
    // Only a single record is expected.
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

    tmp::dyn_hess = DynCSRMat::alloc(vert_count, _param->csrmat_max_nnz);
    tmp::fixed_hessian = FixedCSRMat::alloc(dev_dataset.fixed_index_table,
                                            dev_dataset.transpose_table);
    tmp::tmp_fixed = FixedCSRMat::alloc(dev_dataset.fixed_index_table,
                                        dev_dataset.transpose_table);

    contact::initialize(host_dataset, *param);

    if (!param->disable_contact) {
        // Name: Initial Check Intersection Time
        // Format: list[(int,ms)]
        // Map: initial_check_intersection
        // Description:
        // Time consumed to check if any intersection is detected at the
        // beginning of the simulation.
        // Only a single record is expected.
        logging.push("check intersection");
        contact::update_aabb(host_dataset, dev_dataset, dev_dataset.vertex.prev,
                             dev_dataset.vertex.curr, *param);
        contact::update_collision_mesh_aabb(host_dataset, dev_dataset, *param);
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

    // Name: Consumued Time Per Step
    // Format: list[(vid_time,ms)]
    // Map: time_per_step
    // Description:
    // Time per step in milliseconds. Note that our time step does not
    // advance by a fixed time step, but a reduced one by the accumulated
    // time of impact during the inner Newton loop.
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

    auto vertex_curr = data.vertex.curr.data;
    auto vertex_prev = data.vertex.prev.data;
    auto prop_vertex = data.prop.vertex.data;
    auto prop_face = data.prop.face.data;
    auto prop_tet = data.prop.tet.data;
    auto param_face = data.param_arrays.face.data;
    auto constraint_fix = data.constraint.fix.data;
    auto mesh_face = data.mesh.mesh.face.data;
    float prev_dt = prm.prev_dt;
    Vec3f gravity = prm.gravity;
    bool fitting = prm.fitting;
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

    // Name: Max Velocity
    // Format: list[(vid_time,m/s)]
    // Map: max_velocity
    // Description:
    // Maximum velocity of all the vertices in the mesh.
    logging.mark("max_u", max_u);

    float dt = param->dt * param->playback;

    // Name: Step Size
    // Format: list[(vid_time,float)]
    // Description:
    // Target step size.
    logging.mark("dt", dt);

    // Name: playback
    // Format: list[(vid_time,float)]
    // Description:
    // Playback speed.
    logging.mark("playback", param->playback);

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
                    fmaxf(svd_vec[i].S[0], svd_vec[i].S[1]) * fparam.shrink;
            }
        } DISPATCH_END;
        float max_sigma =
            kernels::max_array(tmp_scalar.data, shell_face_count, 0.0f);
        // Name: Max Stretch
        // Format: list[(vid_time,float)]
        // Description:
        // Maximum stretch among all the shell elements in the scene.
        // If the maximal stretch is 2%, the recorded value is 1.02.
        logging.mark("max_sigma", max_sigma);
    }

    auto compute_target = [&](float dx) {
        Vec<Vec3f> target_vec = target.as_vec();
        DISPATCH_START(vertex_count)
        [prop_vertex, constraint_fix, vertex_curr, vertex_prev, target_vec, dx,
         dt, prev_dt, gravity, fitting] __device__(unsigned i) mutable {
            if (prop_vertex[i].fix_index > 0) {
                unsigned index = prop_vertex[i].fix_index - 1;
                target_vec[i] = constraint_fix[index].position;
            } else {
                Vec3f &x1 = vertex_curr[i];
                Vec3f &x0 = vertex_prev[i];
                float tr(dt / prev_dt), h2(dt * dt);
                Vec3f y = (x1 - x0) * tr + h2 * gravity;
                if (fitting) {
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
        // Format: list[(vid_time,ms)]
        // Description:
        // Time consumed for assembling the global matrix
        // for the linear system solver per Newton's step.
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
                        eval_x_vec[i] - target_vec[i];
                }
            } DISPATCH_END;
        }

        energy::embed_momentum_force_hessian(data, eval_x, velocity, dt, target,
                                             force, diag_hess, prm);

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
        unsigned num_contact = 0;
        float dyn_consumed = 0.0f;
        unsigned max_nnz_row = 0;
        if (!param->disable_contact) {
            num_contact += contact::embed_contact_force_hessian(
                data, eval_x, force, tmp_fixed, fixed_hess, dyn_hess,
                max_nnz_row, dyn_consumed, dt, prm);
        }

        // Name: Consumption Ratio of Dynamic Matrix Assembly Memory
        // Format: list[(vid_time,float)]
        // Description:
        // The GPU memory for the dynamic matrix assembly for contact is
        // pre-allocated.
        // This consumed ratio is the ratio of the memory actually used
        // for the dynamic matrix assembly. If the ratio exceeds 1.0,
        // simulation runs out of memory.
        // One may carefully monitor this value to determine how much
        // memory is required for the simulation.
        // This consumption is only related to contacts and does not
        // affect elastic or inertia terms.
        logging.mark("dyn_consumed", dyn_consumed);

        // Name: Max Row Count for the Contact Matrix
        // Format: list[(vid_time,int)]
        // Description:
        // Records the maximum row count for the contact matrix.
        logging.mark("max_nnz_row", max_nnz_row);

        num_contact += contact::embed_constraint_force_hessian(
            data, eval_x, force, tmp_fixed, fixed_hess, dt, prm);

        // Name: Total Contact Count
        // Format: list[(vid_time,int)]
        // Description:
        // Maximal contact count at a Newton's step.
        logging.mark("num_contact", num_contact);
        logging.pop();

        unsigned iter;
        float reresid;

        // Name: Linear Solve Time
        // Format: list[(vid_time,ms)]
        // Map: pcg_linsolve
        // Description:
        // Total PCG linear solve time per Newton's step.
        logging.push("linsolve");

        bool success =
            solver::solve(dyn_hess, fixed_hess, diag_hess, force, prm.cg_tol,
                          prm.cg_max_iter, dx, iter, reresid);
        logging.pop();

        // Name: Linear Solve Iteration Count
        // Format: list[(vid_time,int)]
        // Map: pcg_iter
        // Description:
        // Count of the PCG linear solve iterations per Newton's step.
        logging.mark("iter", iter);

        // Name: Linear Solve Relative Residual
        // Format: list[(vid_time,float)]
        // Map: pcg_resid
        // Description:
        // Relative Residual of the PCG linear solve iterations per Newton's
        // step.
        logging.mark("reresid", reresid);

        if (!success) {
            logging.message("### cg failed");
            result.pcg_success = false;
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

        // Name: Maximal Magnitude of Search Direction
        // Format: list[(vid_time,float)]
        // Map: max_search_dir
        // Description:
        // Maximum magnitude of the search direction in the Newton's step.
        logging.mark("max_dx", max_dx);
        float toi_recale = fmin(1.0f, prm.max_dx / max_dx);

        // Name: Time of Impact Recalibration
        // Format: list[(vid_time,float)]
        // Description:
        // Recalibration factor for the time of impact (TOI) to ensure
        // the search direction does not exceed the maximum allowed
        // magnitude.
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
                        float y = fmin(1.0f, eval_x_vec[i][1] - fix_xz_val);
                        Vec3f z = vertex_prev[i];
                        eval_x_vec[i][0] -= y * (eval_x_vec[i][0] - z[0]);
                        eval_x_vec[i][2] -= y * (eval_x_vec[i][2] - z[2]);
                    }
                } DISPATCH_END;
            }
        }

        // Name: Line Search Time
        // Format: list[(vid_time,ms)]
        // Description:
        // Line search time per Newton's step.
        // CCD is performed to find the maximal feasible substep without
        // collision.
        logging.push("line search");
        if (!param->disable_contact) {
            contact::update_aabb(host_data, data, target, eval_x, prm);
        }
        float SL_toi = 1.0f;
        float toi = 1.0f;
        toi = fmin(toi, contact::line_search(data, target, eval_x, prm));
        if (shell_face_count > 0) {
            auto tmp_scalar = pool.get<float>(shell_face_count);
            SL_toi = strainlimiting::line_search(data, eval_x, target,
                                                 tmp_scalar, prm);
            toi = fminf(toi, SL_toi);
            // Name: Strain Limiting Time of Impact
            // Format: list[(vid_time,float)]
            // Description:
            // Time of impact (TOI) per Newton's step, encoding the
            // maximal feasible step size without exceeding
            // strain limits.
            logging.mark("SL_toi", SL_toi);
        }
        logging.pop();

        // Name: Time of Impact
        // Format: list[(vid_time,float)]
        // Description:
        // Time of impact (TOI) per Newton's step, encoding the
        // maximal feasible step size without collision or exceeding strain
        // limits.
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
            break;
        }

        if (final_step) {
            break;
        } else if (toi_advanced >= param->target_toi &&
                   step >= param->min_newton_steps) {
            final_step = true;
        } else {
            logging.message("* toi_advanced: %.2e", toi_advanced);
            ++step;
            compute_target(dt);
        }
    }

    if (result.success()) {
        // Name: Time to Check Intersection
        // Format: list[(vid_time,ms)]
        // Map: runtime_intersection_check
        // Description:
        // At the end of step, an explicit intersection check is
        // performed. This number records the consumed time in
        // milliseconds.
        if (!param->disable_contact) {
            logging.push("check intersection");
            if (!contact::check_intersection(data, eval_x, prm)) {
                logging.message("### intersection detected");
                result.intersection_free = false;
            }
            logging.pop();
        }

        // Name: Advanced Fractional Step Size
        // Format: list[(vid_time,float)]
        // Description:
        // This is an accumulated TOI of all the Newton's steps.
        // This number is multiplied by the time step to yield the
        // actual step size advanced in the simulation.
        logging.mark("toi_advanced", toi_advanced);

        // Name: Total Count of Consumed Newton's Steps
        // Format: list[(vid_time,int)]
        // Description:
        // Total count of Newton's steps consumed in the single step.
        logging.mark("newton_steps", step);

        // Name: Final Step Size
        // Format: list[(vid_time,float)]
        // Description:
        // Actual step size advanced in the simulation.
        // For most of the cases, this value is the same as the step
        // size specified in the parameter. However, the actual step
        // size is reduced by `toi_advanced` and may be also reduced
        // when the option `enable_retry` is set to
        // true and the PCG fails.
        logging.mark("final_dt", dt);

        param->prev_dt = dt;
        param->time += static_cast<double>(param->prev_dt / param->playback);

        kernels::copy(dev_dataset.vertex.curr.data,
                      dev_dataset.vertex.prev.data,
                      dev_dataset.vertex.prev.size);
        kernels::copy(eval_x.data, dev_dataset.vertex.curr.data,
                      dev_dataset.vertex.curr.size);

        result.time = param->time;
    }

    return result;
}

void update_bvh(BVHSet bvh) {
    contact::resize_aabb(bvh);
    contact::update_aabb(host_dataset, dev_dataset, dev_dataset.vertex.curr,
                         dev_dataset.vertex.prev, *param);
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
        tmp_collision_mesh.face_bvh = {
            mem::malloc_device(dataset.constraint.mesh.face_bvh.node),
            mem::malloc_device(dataset.constraint.mesh.face_bvh.level)};
        tmp_collision_mesh.edge_bvh = {
            mem::malloc_device(dataset.constraint.mesh.edge_bvh.node),
            mem::malloc_device(dataset.constraint.mesh.edge_bvh.level)};

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
        mem::malloc_device(dataset.constraint.sphere),
        mem::malloc_device(dataset.constraint.floor),
        mem::malloc_device(dataset.constraint.stitch),
        tmp_collision_mesh,
    };

    BVH face_bvh = {
        mem::malloc_device(dataset.bvh.face.node, param.bvh_alloc_factor),
        mem::malloc_device(dataset.bvh.face.level, param.bvh_alloc_factor),
    };
    BVH edge_bvh = {
        mem::malloc_device(dataset.bvh.edge.node, param.bvh_alloc_factor),
        mem::malloc_device(dataset.bvh.edge.level, param.bvh_alloc_factor),
    };
    BVH vertex_bvh = {
        mem::malloc_device(dataset.bvh.vertex.node, param.bvh_alloc_factor),
        mem::malloc_device(dataset.bvh.vertex.level, param.bvh_alloc_factor),
    };
    BVHSet dev_bvhset = {face_bvh, edge_bvh, vertex_bvh};

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
                           dev_bvhset,
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

extern "C" DLL_EXPORT void update_bvh(const BVHSet *bvh) {
    main_helper::host_dataset.bvh = *bvh;
    if (bvh->face.node.size) {
        mem::copy_to_device(bvh->face.node,
                            main_helper::dev_dataset.bvh.face.node);
        mem::copy_to_device(bvh->face.level,
                            main_helper::dev_dataset.bvh.face.level);
    }
    if (bvh->edge.node.size) {
        mem::copy_to_device(bvh->edge.node,
                            main_helper::dev_dataset.bvh.edge.node);
        mem::copy_to_device(bvh->edge.level,
                            main_helper::dev_dataset.bvh.edge.level);
    }
    if (bvh->vertex.node.size) {
        mem::copy_to_device(bvh->vertex.node,
                            main_helper::dev_dataset.bvh.vertex.node);
        mem::copy_to_device(bvh->vertex.level,
                            main_helper::dev_dataset.bvh.vertex.level);
    }
    main_helper::update_bvh(main_helper::host_dataset.bvh);
}

extern "C" DLL_EXPORT void fetch_dyn_counts(unsigned *n_value, unsigned *n_offset) {
    unsigned nrow = tmp::dyn_hess.nrow;
    *n_offset = nrow + 1;
    CUDA_HANDLE_ERROR(cudaMemcpy(n_value,
                                 tmp::dyn_hess.fixed_row_offsets.data + nrow,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
}

extern "C" DLL_EXPORT void fetch_dyn(unsigned *index, Mat3x3f *value, unsigned *offset) {
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
    mem::copy_to_device(constraint->stitch,
                        main_helper::dev_dataset.constraint.stitch);
    mem::copy_to_device(constraint->sphere,
                        main_helper::dev_dataset.constraint.sphere);
    mem::copy_to_device(constraint->floor,
                        main_helper::dev_dataset.constraint.floor);
}
