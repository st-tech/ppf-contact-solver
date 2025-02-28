// File: main.cu
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../contact/contact.hpp"
#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../energy/energy.hpp"
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
Vec<Vec3f> eval_x;
Vec<Vec3f> tmp_eval_x;
Vec<Vec3f> target;
Vec<Vec3f> velocity;
Kinematic kinematic;
Vec<float> tmp_scalar;
Vec<Svd3x2> svd;
Vec<float> force;
Vec<float> dx;
Vec<Mat3x3f> diag_hessian;
FixedCSRMat fixed_hessian;
FixedCSRMat tmp_fixed;
DynCSRMat dyn_hess;
unsigned active_shell_count;
} // namespace tmp

namespace main_helper {
DataSet host_dataset, dev_dataset;
ParamSet *param;

void build_kinematic(const DataSet &host_dataset, const DataSet &dev_dataset,
                     const ParamSet &param) {

    unsigned vert_count = host_dataset.vertex.curr.size;
    unsigned edge_count = host_dataset.mesh.mesh.edge.size;
    unsigned face_count = host_dataset.mesh.mesh.face.size;
    unsigned hinge_count = host_dataset.mesh.mesh.hinge.size;
    unsigned tet_count = host_dataset.mesh.mesh.tet.size;
    const DataSet &data = dev_dataset;

    Vec<VertexKinematic> kinematic_vertex = tmp::kinematic.vertex;
    DISPATCH_START(vert_count)
    [data, kinematic_vertex] __device__(unsigned i) mutable {
        kinematic_vertex[i].active = false;
        for (unsigned j = 0; j < data.constraint.fix.size; ++j) {
            if (i == data.constraint.fix[j].index) {
                kinematic_vertex[i].active = true;
                kinematic_vertex[i].kinematic =
                    data.constraint.fix[j].kinematic;
                kinematic_vertex[i].position = data.constraint.fix[j].position;
            }
        }
        kinematic_vertex[i].rod = data.mesh.neighbor.vertex.rod.count(i) > 0;
    } DISPATCH_END;

    Vec<bool> kinematic_face = tmp::kinematic.face;
    DISPATCH_START(face_count)
    [data, kinematic_vertex, kinematic_face] __device__(unsigned i) mutable {
        Vec3u face = data.mesh.mesh.face[i];
        SVec<bool, 3> flag;
        for (int j = 0; j < 3; ++j) {
            flag[j] = kinematic_vertex[face[j]].active;
        }
        kinematic_face[i] = flag.all();
    } DISPATCH_END;

    Vec<float> tmp_face = tmp::tmp_scalar;
    Vec<unsigned> fake_tmp_face;
    fake_tmp_face.data = (unsigned *)tmp_face.data;
    fake_tmp_face.size = tmp_face.size;
    fake_tmp_face.allocated = tmp_face.allocated;
    DISPATCH_START(face_count)
    [data, fake_tmp_face, kinematic_face] __device__(unsigned i) mutable {
        fake_tmp_face[i] = kinematic_face[i] ? 0 : 1;
    } DISPATCH_END;
    tmp::active_shell_count =
        utility::sum_integer_array(fake_tmp_face, face_count);

    Vec<bool> kinematic_tet = tmp::kinematic.tet;
    DISPATCH_START(tet_count)
    [data, kinematic_vertex, kinematic_tet] __device__(unsigned i) mutable {
        Vec4u tet = data.mesh.mesh.tet[i];
        SVec<bool, 3> flag;
        for (int j = 0; j < 4; ++j) {
            flag[j] = kinematic_vertex[tet[j]].active;
        }
        kinematic_tet[i] = flag.all();
    } DISPATCH_END;

    Vec<bool> kinematic_edge = tmp::kinematic.edge;
    DISPATCH_START(edge_count)
    [data, kinematic_vertex, kinematic_edge] __device__(unsigned i) mutable {
        Vec2u edge = data.mesh.mesh.edge[i];
        SVec<bool, 2> flag;
        flag[0] = kinematic_vertex[edge[0]].active;
        flag[1] = kinematic_vertex[edge[1]].active;
        kinematic_edge[i] = flag.all();
    } DISPATCH_END;

    Vec<bool> kinematic_hinge = tmp::kinematic.hinge;
    DISPATCH_START(hinge_count)
    [data, kinematic_vertex, kinematic_hinge] __device__(unsigned i) mutable {
        Vec4u hinge = data.mesh.mesh.hinge[i];
        SVec<bool, 4> flag;
        for (int j = 0; j < 4; ++j) {
            flag[j] = kinematic_vertex[hinge[j]].active;
        }
        kinematic_hinge[i] = flag.all();
    } DISPATCH_END;
}

void initialize(DataSet _host_dataset, DataSet _dev_dataset, ParamSet *_param) {

    // Name: Initialization Time
    // Format: list[(int,ms)]
    // Description:
    // Time consumed for the initialization of the simulation.
    // Only a single record is expected.
    SimpleLog logging("initialize");

    host_dataset = _host_dataset;
    dev_dataset = _dev_dataset;
    param = _param;

    unsigned vert_count = host_dataset.vertex.curr.size;
    unsigned edge_count = host_dataset.mesh.mesh.edge.size;
    unsigned face_count = host_dataset.mesh.mesh.face.size;
    unsigned hinge_count = host_dataset.mesh.mesh.hinge.size;
    unsigned tet_count = host_dataset.mesh.mesh.tet.size;

    const unsigned max_reduce_count = std::max(
        std::max(face_count, edge_count), std::max(tet_count, 3 * vert_count));
    utility::set_max_reduce_count(max_reduce_count);

    unsigned collision_mesh_vert_count =
        host_dataset.constraint.mesh.active
            ? host_dataset.constraint.mesh.vertex.size
            : 0;
    unsigned collision_mesh_edge_count =
        host_dataset.constraint.mesh.active
            ? host_dataset.constraint.mesh.edge.size
            : 0;
    unsigned shell_face_count = host_dataset.shell_face_count;
    unsigned max_n = 0;
    max_n = std::max(max_n, vert_count);
    max_n = std::max(max_n, edge_count);
    max_n = std::max(max_n, face_count);
    max_n = std::max(max_n, tet_count);
    max_n = std::max(max_n, hinge_count);
    max_n = std::max(max_n, collision_mesh_vert_count);
    max_n = std::max(max_n, collision_mesh_edge_count);
    max_n = std::max(max_n, shell_face_count);
    tmp::tmp_scalar = Vec<float>::alloc(max_n);
    tmp::dx = Vec<float>::alloc(3 * vert_count);
    tmp::eval_x = Vec<Vec3f>::alloc(vert_count);
    tmp::tmp_eval_x = Vec<Vec3f>::alloc(vert_count);
    tmp::kinematic.vertex = Vec<VertexKinematic>::alloc(vert_count);
    tmp::kinematic.face = Vec<bool>::alloc(face_count);
    tmp::kinematic.edge = Vec<bool>::alloc(edge_count);
    tmp::kinematic.hinge = Vec<bool>::alloc(hinge_count);
    tmp::kinematic.tet = Vec<bool>::alloc(tet_count);
    tmp::target = Vec<Vec3f>::alloc(vert_count);
    tmp::velocity = Vec<Vec3f>::alloc(vert_count);
    tmp::svd = Vec<Svd3x2>::alloc(shell_face_count);
    tmp::force = Vec<float>::alloc(3 * vert_count);
    tmp::dyn_hess = DynCSRMat::alloc(vert_count, _param->csrmat_max_nnz);
    tmp::diag_hessian = Vec<Mat3x3f>::alloc(vert_count);
    tmp::fixed_hessian = FixedCSRMat::alloc(dev_dataset.fixed_index_table,
                                            dev_dataset.transpose_table);
    tmp::tmp_fixed = FixedCSRMat::alloc(dev_dataset.fixed_index_table,
                                        dev_dataset.transpose_table);

    contact::initialize(host_dataset, *param);

    logging.push("build_kinematic");
    build_kinematic(host_dataset, dev_dataset, *param);
    logging.pop();

    // Name: Initial Check Intersection Time
    // Format: list[(int,ms)]
    // Map: initial_check_intersection
    // Description:
    // Time consumed to check if any intersection is detected at the
    // beginning of the simulation.
    // Only a single record is expected.
    logging.push("check intersection");
    contact::update_aabb(host_dataset, dev_dataset, dev_dataset.vertex.prev,
                         dev_dataset.vertex.curr, tmp::kinematic.vertex,
                         *param);
    contact::update_collision_mesh_aabb(host_dataset, dev_dataset, *param);
    assert(contact::check_intersection(dev_dataset, tmp::kinematic,
                                       dev_dataset.vertex.prev));
    assert(contact::check_intersection(dev_dataset, tmp::kinematic,
                                       dev_dataset.vertex.curr));
    logging.pop();
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
    DataSet &data = dev_dataset;
    Kinematic &kinematic = tmp::kinematic;
    ParamSet &prm = *param;
    Vec<float> tmp_scalar = tmp::tmp_scalar;
    Vec<Vec3f> &eval_x = tmp::eval_x;
    Vec<Vec3f> &tmp_eval_x = tmp::tmp_eval_x;
    Vec<Vec3f> &target = tmp::target;
    Vec<Vec3f> &velocity = tmp::velocity;
    Vec<float> &force = tmp::force;
    Vec<float> &dx = tmp::dx;
    DynCSRMat &dyn_hess = tmp::dyn_hess;
    Vec<Svd3x2> &svd = tmp::svd;
    Vec<Mat3x3f> &diag_hess = tmp::diag_hessian;
    FixedCSRMat &tmp_fixed = tmp::tmp_fixed;
    FixedCSRMat &fixed_hess = tmp::fixed_hessian;
    const unsigned vertex_count = host_data.vertex.curr.size;
    const unsigned shell_face_count = host_dataset.shell_face_count;
    const unsigned tet_count = host_data.mesh.mesh.tet.size;

    SimpleLog::set(prm.time);

    logging.push("build_kinematic");
    build_kinematic(host_dataset, dev_dataset, *param);
    logging.pop();

    tmp_scalar.clear();
    DISPATCH_START(vertex_count)
    [data, tmp_scalar, velocity, kinematic,
     prm] __device__(unsigned i) mutable {
        Vec3f u = (data.vertex.curr[i] - data.vertex.prev[i]) / prm.prev_dt;
        velocity[i] = u;
        tmp_scalar[i] = kinematic.vertex[i].active ? 0.0f : u.squaredNorm();
    } DISPATCH_END;
    float max_u =
        sqrtf(utility::max_array(tmp_scalar.data, vertex_count, 0.0f));

    // Name: Max Velocity
    // Format: list[(vid_time,m/s)]
    // Map: max_velocity
    // Description:
    // Maximum velocity of all the vertices in the mesh.
    logging.mark("max_u", max_u);

    if (shell_face_count) {
        tmp_scalar.clear();
        DISPATCH_START(shell_face_count)
        [data, tmp_scalar, kinematic] __device__(unsigned i) mutable {
            if (!kinematic.face[i]) {
                tmp_scalar[i] = data.prop.face[i].mass;
            }
        } DISPATCH_END;
    }

    if (tet_count) {
        tmp_scalar.clear();
        DISPATCH_START(tet_count)
        [data, tmp_scalar, kinematic] __device__(unsigned i) mutable {
            if (!kinematic.tet[i]) {
                tmp_scalar[i] = data.prop.tet[i].mass;
            }
        } DISPATCH_END;
    }

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
        utility::compute_svd(data, data.vertex.curr, svd, prm);
        tmp_scalar.clear();
        DISPATCH_START(shell_face_count)
        [svd, tmp_scalar, kinematic] __device__(unsigned i) mutable {
            if (!kinematic.face[i]) {
                tmp_scalar[i] = fmaxf(svd[i].S[0], svd[i].S[1]);
            }
        } DISPATCH_END;
        float max_sigma =
            utility::max_array(tmp_scalar.data, shell_face_count, 0.0f);
        float avg_sigma = utility::sum_array(tmp_scalar, shell_face_count) /
                          tmp::active_shell_count;
        // Name: Max Stretch
        // Format: list[(vid_time,float)]
        // Description:
        // Maximum stretch among all the shell elements in the scene.
        // If the maximal stretch is 2%, the recorded value is 1.02.
        logging.mark("max_sigma", max_sigma);

        // Name: Average Stretch
        // Format: list[(vid_time,float)]
        // Description:
        // Average stretch of all the shell elements in the scene.
        // If the average stretch is 2%, the recorded value is 1.02.
        logging.mark("avg_sigma", avg_sigma);
    }

    auto compute_target = [&](float dx) {
        DISPATCH_START(vertex_count)
        [data, kinematic, dx, target, prm, dt] __device__(unsigned i) mutable {
            if (kinematic.vertex[i].active) {
                target[i] = kinematic.vertex[i].position;
            } else {
                Vec3f &x1 = data.vertex.curr[i];
                Vec3f &x0 = data.vertex.prev[i];
                float tr(dt / prm.prev_dt), h2(dt * dt);
                Vec3f y = (x1 - x0) * tr + h2 * prm.gravity;
                if (prm.fitting) {
                    target[i] = x1;
                } else {
                    target[i] = x1 + y;
                }
            }
        } DISPATCH_END;
    };

    compute_target(dt);

    eval_x.copy(data.vertex.curr);

    double toi_advanced = 0.0f;
    unsigned step(1);
    bool final_step(false);

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

        DISPATCH_START(vertex_count)
        [data, eval_x, kinematic, dx, target, prm,
         dt] __device__(unsigned i) mutable {
            if (kinematic.vertex[i].active) {
                Map<Vec3f>(dx.data + 3 * i) = eval_x[i] - target[i];
            }
        } DISPATCH_END;

        energy::embed_momentum_force_hessian(data, eval_x, kinematic, velocity,
                                             dt, target, force, diag_hess, prm);

        energy::embed_elastic_force_hessian(data, eval_x, kinematic, force,
                                            fixed_hess, dt, prm);

        if (host_data.constraint.stitch.size) {
            energy::embed_stitch_force_hessian(data, eval_x, force, fixed_hess,
                                               prm);
        }

        tmp_fixed.copy(fixed_hess);

        if (prm.strain_limit_eps && data.shell_face_count > 0) {
            strainlimiting::embed_strainlimiting_force_hessian(
                data, eval_x, kinematic, force, tmp_fixed, fixed_hess, prm);
        }

        unsigned num_contact = 0;
        float dyn_consumed = 0.0f;
        unsigned max_nnz_row = 0;
        contact::update_aabb(host_data, data, eval_x, eval_x,
                             tmp::kinematic.vertex, prm);
        num_contact += contact::embed_contact_force_hessian(
            data, eval_x, kinematic, force, tmp_fixed, fixed_hess, dyn_hess,
            max_nnz_row, dyn_consumed, dt, prm);

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
            data, eval_x, kinematic, force, tmp_fixed, fixed_hess, dt, prm);

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
            break;
        }

        tmp_scalar.clear();
        DISPATCH_START(vertex_count)
        [dx, tmp_scalar] __device__(unsigned i) mutable {
            tmp_scalar[i] = Map<Vec3f>(dx.data + 3 * i).norm();
        } DISPATCH_END;

        float max_dx = utility::max_array(tmp_scalar.data, vertex_count, 0.0f);

        // Name: Maximal Magnitude of Search Direction
        // Format: list[(vid_time,float)]
        // Map: max_search_dir
        // Description:
        // Maximum magnitude of the search direction in the Newton's step.
        logging.mark("max_dx", max_dx);
        float toi_recale = fmin(1.0f, dt * prm.max_search_dir_vel / max_dx);

        // Name: Time of Impact Recalibration
        // Format: list[(vid_time,float)]
        // Description:
        // Recalibration factor for the time of impact (TOI) to ensure
        // the search direction does not exceed the maximum allowed
        // magnitude.
        logging.mark("toi_recale", toi_recale);

        tmp_eval_x.copy(eval_x);
        DISPATCH_START(vertex_count)
        [eval_x, data, toi_recale, dx] __device__(unsigned i) mutable {
            eval_x[i] -= toi_recale * Map<Vec3f>(dx.data + 3 * i);
        } DISPATCH_END;

        if (param->fix_xz) {
            DISPATCH_START(vertex_count)
            [eval_x, data, prm] __device__(unsigned i) mutable {
                if (eval_x[i][1] > prm.fix_xz) {
                    float y = fmin(1.0f, eval_x[i][1] - prm.fix_xz);
                    Vec3f z = data.vertex.prev[i];
                    eval_x[i][0] -= y * (eval_x[i][0] - z[0]);
                    eval_x[i][2] -= y * (eval_x[i][2] - z[2]);
                }
            } DISPATCH_END;
        }

        // Name: Line Search Time
        // Format: list[(vid_time,ms)]
        // Description:
        // Line search time per Newton's step.
        // CCD is performed to find the maximal feasible substep without
        // collision.
        logging.push("line search");
        contact::update_aabb(host_data, data, tmp_eval_x, eval_x,
                             tmp::kinematic.vertex, prm);
        float SL_toi = 1.0f;
        float toi = 1.0f;
        toi = fmin(toi, contact::line_search(data, kinematic, tmp_eval_x,
                                             eval_x, prm));
        if (prm.strain_limit_eps && shell_face_count > 0) {
            SL_toi = strainlimiting::line_search(data, kinematic, eval_x,
                                                 tmp_eval_x, tmp_scalar, prm);
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
            break;
        }

        if (!final_step) {
            toi_advanced += std::max(0.0, 1.0 - toi_advanced) *
                            static_cast<double>(toi_recale * toi);
        }

        DISPATCH_START(vertex_count)
        [eval_x, tmp_eval_x, data, toi] __device__(unsigned i) mutable {
            Vec3f d = toi * (eval_x[i] - tmp_eval_x[i]);
            eval_x[i] = tmp_eval_x[i] + d;
        } DISPATCH_END;

        if (final_step) {
            break;
        } else if (toi_advanced >= param->target_toi &&
                   step >= param->min_newton_steps) {
            final_step = true;
        } else {
            logging.message("* toi_advanced: %.2e", toi_advanced);
            ++step;
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
        logging.push("check intersection");
        if (!contact::check_intersection(data, kinematic, eval_x)) {
            logging.message("### intersection detected");
            result.intersection_free = false;
        }
        logging.pop();
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
        // size specified in the parameter.
        logging.mark("final_dt", dt);

        param->prev_dt = dt;
        param->time += static_cast<double>(param->prev_dt / param->playback);

        dev_dataset.vertex.prev.copy(dev_dataset.vertex.curr);
        dev_dataset.vertex.curr.copy(eval_x);

        result.time = param->time;
    }
    return result;
}

void update_bvh(BVHSet bvh) {
    contact::resize_aabb(bvh);
    contact::update_aabb(host_dataset, dev_dataset, dev_dataset.vertex.curr,
                         dev_dataset.vertex.prev, tmp::kinematic.vertex,
                         *param);
}

} // namespace main_helper

extern "C" void set_log_path(const char *data_dir) {
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
                             mem::malloc_device(dataset.prop.rod),
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

        VertexNeighbor dev_vertex_neighbor = {
            mem::malloc_device(dataset.constraint.mesh.neighbor.vertex.face),
            mem::malloc_device(dataset.constraint.mesh.neighbor.vertex.hinge),
            mem::malloc_device(dataset.constraint.mesh.neighbor.vertex.edge),
        };

        HingeNeighbor dev_hinge_neighbor = {
            mem::malloc_device(dataset.constraint.mesh.neighbor.hinge.face)};

        EdgeNeighbor dev_edge_neighbor = {
            mem::malloc_device(dataset.constraint.mesh.neighbor.edge.face)};

        tmp_collision_mesh.neighbor = {
            dev_vertex_neighbor,
            dev_hinge_neighbor,
            dev_edge_neighbor,
        };
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

    DataSet dev_dataset = {dev_vertex,
                           dev_mesh_info,
                           dev_prop_info,
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

extern "C" void initialize(DataSet *dataset, ParamSet *param) {

    int num_device;
    CUDA_HANDLE_ERROR(cudaGetDeviceCount(&num_device));
    logging::info("cuda: detected %d devices...", num_device);
    if (num_device == 0) {
        logging::info("cuda: no device found...");
        exit(1);
    }

    logging::info("cuda: allocating memory...");
    DataSet dev_dataset = malloc_dataset(*dataset, *param);

    main_helper::initialize(*dataset, dev_dataset, param);
}

extern "C" void advance(StepResult *result) {
    *result = main_helper::advance();
}

extern "C" void fetch() {
    mem::copy_from_device_to_host(main_helper::dev_dataset.vertex.curr.data,
                                  main_helper::host_dataset.vertex.curr.data,
                                  main_helper::host_dataset.vertex.curr.size);
    mem::copy_from_device_to_host(main_helper::dev_dataset.vertex.prev.data,
                                  main_helper::host_dataset.vertex.prev.data,
                                  main_helper::host_dataset.vertex.prev.size);
}

extern "C" void update_bvh(const BVHSet *bvh) {
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

extern "C" void fetch_dyn_counts(unsigned *n_value, unsigned *n_offset) {
    unsigned nrow = tmp::dyn_hess.nrow;
    *n_offset = nrow + 1;
    CUDA_HANDLE_ERROR(cudaMemcpy(n_value,
                                 tmp::dyn_hess.fixed_row_offsets.data + nrow,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
}

extern "C" void fetch_dyn(unsigned *index, Mat3x3f *value, unsigned *offset) {
    tmp::dyn_hess.fetch(index, value, offset);
}

extern "C" void update_dyn(unsigned *index, unsigned *offset) {
    tmp::dyn_hess.update(index, offset);
}

extern "C" void update_constraint(const Constraint *constraint) {
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
