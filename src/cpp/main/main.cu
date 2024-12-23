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

    SimpleLog logging("initialize");

    host_dataset = _host_dataset;
    dev_dataset = _dev_dataset;
    param = _param;

    unsigned vert_count = host_dataset.vertex.curr.size;
    unsigned edge_count = host_dataset.mesh.mesh.edge.size;
    unsigned face_count = host_dataset.mesh.mesh.face.size;
    unsigned hinge_count = host_dataset.mesh.mesh.hinge.size;
    unsigned tet_count = host_dataset.mesh.mesh.tet.size;
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

    SimpleLog logging("advance");

    StepResult result;
    result.retry_count = 0;
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
    const float strain_limit_sum = prm.strain_limit_tau + prm.strain_limit_eps;
    SimpleLog::set(prm.time);
    logging.mark("vertex count", vertex_count);
    logging.mark("rod count", host_data.rod_count);
    logging.mark("shell count", host_data.shell_face_count);
    logging.mark("face count", host_data.mesh.mesh.face.size);
    logging.mark("tet count", host_data.mesh.mesh.tet.size);

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
    logging.mark("max_u", max_u);

    if (shell_face_count) {
        tmp_scalar.clear();
        DISPATCH_START(shell_face_count)
        [data, tmp_scalar, kinematic] __device__(unsigned i) mutable {
            if (!kinematic.face[i]) {
                tmp_scalar[i] = data.prop.face[i].mass;
            }
        } DISPATCH_END;
        float total_mass = utility::sum_array(tmp_scalar, shell_face_count);
        if (total_mass) {
            logging.mark("total shell mass", total_mass);
        }
    }
    if (tet_count) {
        tmp_scalar.clear();
        DISPATCH_START(tet_count)
        [data, tmp_scalar, kinematic] __device__(unsigned i) mutable {
            if (!kinematic.tet[i]) {
                tmp_scalar[i] = data.prop.tet[i].mass;
            }
        } DISPATCH_END;
        float total_mass = utility::sum_array(tmp_scalar, tet_count);
        if (total_mass) {
            logging.mark("total solid mass", total_mass);
        }
    }

    float dt = param->dt;
    logging.mark("dt", dt);

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
        logging.mark("max_sigma", max_sigma);
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

    unsigned retry_count(0);
    while (true) {
        compute_target(dt);

        eval_x.copy(data.vertex.curr);

        float toi_advanced = 0.0f;
        unsigned step(1);
        bool final_step(false);
        bool retry(false);

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

            logging.push("matrix assembly");

            DISPATCH_START(vertex_count)
            [data, eval_x, kinematic, dx, target, prm,
             dt] __device__(unsigned i) mutable {
                if (kinematic.vertex[i].active) {
                    Map<Vec3f>(dx.data + 3 * i) = eval_x[i] - target[i];
                }
            } DISPATCH_END;

            energy::embed_momentum_force_hessian(data, eval_x, kinematic,
                                                 velocity, dt, target, force,
                                                 diag_hess, prm);

            energy::embed_elastic_force_hessian(data, eval_x, kinematic, force,
                                                fixed_hess, dt, prm);

            if (host_data.constraint.stitch.size) {
                energy::embed_stitch_force_hessian(data, eval_x, force,
                                                   fixed_hess, prm);
            }

            tmp_fixed.copy(fixed_hess);

            if (strain_limit_sum && data.shell_face_count > 0) {
                strainlimiting::embed_strainlimiting_force_hessian(
                    data, eval_x, kinematic, force, tmp_fixed, fixed_hess, prm);
            }

            unsigned num_contact = 0;
            float dyn_consumed;
            unsigned max_nnz_row;
            contact::update_aabb(host_data, data, eval_x, eval_x,
                                 tmp::kinematic.vertex, prm);
            num_contact += contact::embed_contact_force_hessian(
                data, eval_x, kinematic, force, tmp_fixed, fixed_hess, dyn_hess,
                max_nnz_row, dyn_consumed, dt, prm);
            logging.mark("dyn_consumed", dyn_consumed);
            logging.mark("max_nnz_row", max_nnz_row);

            num_contact += contact::embed_constraint_force_hessian(
                data, eval_x, kinematic, force, tmp_fixed, fixed_hess, dt, prm);

            logging.mark("num_contact", num_contact);
            logging.pop();

            unsigned iter;
            float reresid;
            logging.push("linsolve");

            bool success =
                solver::solve(dyn_hess, fixed_hess, diag_hess, force,
                              prm.cg_tol, prm.cg_max_iter, dx, iter, reresid);
            logging.pop();

            if (!success) {
                logging.message("### cg failed");
                if (param->enable_retry && dt > DT_MIN) {
                    retry = true;
                } else {
                    result.pcg_success = false;
                }
                break;
            }
            logging.mark("iter", iter);
            logging.mark("reresid", reresid);

            tmp_scalar.clear();
            DISPATCH_START(vertex_count)
            [dx, tmp_scalar] __device__(unsigned i) mutable {
                tmp_scalar[i] = Map<Vec3f>(dx.data + 3 * i).norm();
            } DISPATCH_END;

            float max_dx =
                utility::max_array(tmp_scalar.data, vertex_count, 0.0f);
            logging.mark("max_dx", max_dx);

            tmp_eval_x.copy(eval_x);
            DISPATCH_START(vertex_count)
            [eval_x, data, dx] __device__(unsigned i) mutable {
                eval_x[i] -= Map<Vec3f>(dx.data + 3 * i);
            } DISPATCH_END;

            if (param->fix_xz) {
                DISPATCH_START(vertex_count)
                [eval_x, data, prm] __device__(unsigned i) mutable {
                    if (eval_x[i][1] > prm.fix_xz) {
                        float y = fmin(1.0f, eval_x[i][1] - prm.fix_xz);
                        Vec3f z = data.vertex.prev[i];
                        eval_x[i][0] = (1.0f - y) * eval_x[i][0] + y * z[0];
                        eval_x[i][2] = (1.0f - y) * eval_x[i][2] + y * z[2];
                    }
                } DISPATCH_END;
            }

            logging.push("line search");
            contact::update_aabb(host_data, data, tmp_eval_x, eval_x,
                                 tmp::kinematic.vertex, prm);
            float toi =
                contact::line_search(data, kinematic, tmp_eval_x, eval_x, prm);
            if (strain_limit_sum && shell_face_count > 0) {
                toi = fminf(toi, strainlimiting::line_search(data, kinematic,
                                                             eval_x, tmp_eval_x,
                                                             tmp_scalar, prm));
            }
            logging.pop();

            if (toi <= 0.0f) {
                logging.message("### ccd failed (toi: %.2e)", toi);
                result.ccd_success = false;
                break;
            }

            if (!final_step) {
                toi_advanced += fmaxf(0.0f, 1.0f - toi_advanced) * toi;
            }
            logging.mark("toi", toi);

            DISPATCH_START(vertex_count)
            [eval_x, tmp_eval_x, data, toi] __device__(unsigned i) mutable {
                eval_x[i] = (1.0f - toi) * tmp_eval_x[i] + toi * eval_x[i];
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
        if (retry) {
            dt *= param->dt_decrease_factor;
            logging.message("**** retry requested. dt: %.2e", dt);
            retry_count++;
        } else {
            if (!contact::check_intersection(data, kinematic, eval_x)) {
                logging.message("### intersection detected");
                result.intersection_free = false;
            }
            if (result.success()) {
                logging.mark("toi_advanced", toi_advanced);
                logging.mark("newton_steps", step);

                param->prev_dt = dt;
                param->time += param->prev_dt;

                dev_dataset.vertex.prev.copy(dev_dataset.vertex.curr);
                dev_dataset.vertex.curr.copy(eval_x);

                result.time = param->time;
            }
            break;
        }
    }
    if (param->enable_retry) {
        logging.mark("retry_count", retry_count);
    }
    result.retry_count = retry_count;
    return result;
}

void update_bvh(BVHSet bvh) {
    contact::resize_aabb(bvh);
    contact::update_aabb(host_dataset, dev_dataset, dev_dataset.vertex.curr,
                         dev_dataset.vertex.prev, tmp::kinematic.vertex,
                         *param);
}

} // namespace main_helper

extern "C" void set_log_path(const char *log_path, const char *data_dir) {
    SimpleLog::setPath(log_path, data_dir);
}

DataSet malloc_dataset(DataSet dataset, ParamSet param) {

    logging::info("cuda: dev_neighbor...");
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

    logging::info("cuda: dev_mesh_info...");
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

    logging::info("cuda: dev_prop_info...");
    PropSet dev_prop_info = {mem::malloc_device(dataset.prop.vertex),
                             mem::malloc_device(dataset.prop.rod),
                             mem::malloc_device(dataset.prop.face),
                             mem::malloc_device(dataset.prop.hinge),
                             mem::malloc_device(dataset.prop.tet)};

    logging::info("cuda: tmp_collision_mesh...");
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

    logging::info("cuda: dev_constraint...");
    Constraint dev_constraint = {
        mem::malloc_device(dataset.constraint.fix),
        mem::malloc_device(dataset.constraint.pull),
        mem::malloc_device(dataset.constraint.sphere),
        mem::malloc_device(dataset.constraint.floor),
        mem::malloc_device(dataset.constraint.stitch),
        tmp_collision_mesh,
    };

    logging::info("cuda: bvh...");
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

    logging::info("cuda: inv_rest...");
    Vec<Mat2x2f> dev_inv_rest2x2 = mem::malloc_device(dataset.inv_rest2x2);
    Vec<Mat3x3f> dev_inv_rest3x3 = mem::malloc_device(dataset.inv_rest3x3);

    logging::info("cuda: dev_vertex...");
    VertexSet dev_vertex = {
        mem::malloc_device(dataset.vertex.prev),
        mem::malloc_device(dataset.vertex.curr),
    };

    logging::info("cuda: dev_index_table...");
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
    logging::info("GPU::initialize");

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

extern "C" StepResult advance() {
    logging::info("GPU::advance");
    return main_helper::advance();
}

extern "C" void fetch() {
    logging::info("GPU::fetch");
    mem::copy_from_device_to_host(main_helper::dev_dataset.vertex.curr.data,
                                  main_helper::host_dataset.vertex.curr.data,
                                  main_helper::host_dataset.vertex.curr.size);
    mem::copy_from_device_to_host(main_helper::dev_dataset.vertex.prev.data,
                                  main_helper::host_dataset.vertex.prev.data,
                                  main_helper::host_dataset.vertex.prev.size);
}

extern "C" void update_bvh(const BVHSet *bvh) {
    logging::info("GPU::update_bvh");
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
    logging::info("GPU::fetch_counts");
    unsigned nrow = tmp::dyn_hess.nrow;
    *n_offset = nrow + 1;
    CUDA_HANDLE_ERROR(cudaMemcpy(n_value,
                                 tmp::dyn_hess.fixed_row_offsets.data + nrow,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
}

extern "C" void fetch_dyn(unsigned *index, Mat3x3f *value, unsigned *offset) {
    logging::info("GPU::fetch_dyn");
    tmp::dyn_hess.fetch(index, value, offset);
}

extern "C" void update_dyn(unsigned *index, unsigned *offset) {
    logging::info("GPU::update_dyn");
    tmp::dyn_hess.update(index, offset);
}

extern "C" void update_constraint(const Constraint *constraint) {
    logging::info("GPU::update_constraint");
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
