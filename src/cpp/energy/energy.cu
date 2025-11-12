// File: energy.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../eigenanalysis/eigenanalysis.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "model/air_damper.hpp"
#include "model/arap.hpp"
#include "model/baraffwitkin.hpp"
#include "model/dihedral_angle.hpp"
#include "model/hook.hpp"
#include "model/momentum.hpp"
#include "model/snhk.hpp"
#include "model/stvk.hpp"

namespace energy {

__device__ void
embed_vertex_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                           const Vec<Vec3f> &velocity, const Vec<Vec3f> &target,
                           Vec<float> &force, Vec<Mat3x3f> &diag_hess, float dt,
                           const ParamSet &param, unsigned i) {

    float mass = data.prop.vertex[i].mass;
    float area = data.prop.vertex[i].area;

    const Vec3f &x = data.vertex.curr[i];
    const Vec3f &y = eval_x[i];
    const Vec3f normal = utility::compute_vertex_normal(data, eval_x, i);

    Vec3f wind = Vec3f::Zero();
    if (!param.fitting) {
        wind = utility::get_wind_weight(param.time) * param.wind;
    }

    Vec3f f = Vec3f::Zero();
    Mat3x3f H = Mat3x3f::Zero();
    if (normal.isZero() == false && param.air_density) {
        f += area * param.air_density *
             air_damper::face_gradient(dt, y, x, normal, wind, param);
        H += area * param.air_density *
             air_damper::face_hessian(dt, normal, param);
    }

    bool pulled(false);
    for (unsigned j = 0; j < data.constraint.pull.size; ++j) {
        if (i == data.constraint.pull[j].index) {
            Vec3f position = data.constraint.pull[j].position;
            float weight = data.constraint.pull[j].weight;
            f += weight * (y - position);
            H += weight * Mat3x3f::Identity();
            pulled = true;
            break;
        }
    }

    if (!pulled) {
        f += mass * momentum::gradient(dt, y, target[i]);
        H += mass * momentum::hessian(dt);
    }

    if (param.isotropic_air_friction) {
        f += param.isotropic_air_friction * (y - x) / (dt * dt);
        H += (param.isotropic_air_friction / (dt * dt)) * Mat3x3f::Identity();
    }

    if (param.fix_xz && y[1] > float(param.fix_xz)) {
        float t = fmin(1.0f, y[1] - float(param.fix_xz));
        Vec3f n(0.0f, 1.0f, 0.0f);
        Mat3x3f P = Mat3x3f::Identity() - n * n.transpose();
        f += P * t * mass * (y - x) / (dt * dt);
        H += P * t * mass * (dt * dt);
    }

    Map<Vec3f>(force.data + 3 * i) += f;
    diag_hess[i] += H;
}

__device__ void embed_rod_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> &force, FixedCSRMat &hess,
                                        float dt, const ParamSet &param,
                                        unsigned i) {
    const Vec2u &edge = data.mesh.mesh.edge[i];
    const Vec3f &x0 = eval_x[edge[0]];
    const Vec3f &x1 = eval_x[edge[1]];

    const EdgeProp &prop = data.prop.edge[i];
    const EdgeParam &edge_param = data.param_arrays.edge[prop.param_index];
    float l0 = prop.length;
    Vec3f t = x1 - x0;
    float l = t.norm();
    float mass = prop.mass;
    float stiffness = edge_param.stiffness;
    if (stiffness > 0.0f) {
        Mat3x2f dedx;
        Mat6x6f d2edx2;
        hook::make_diff_table(x0, x1, l0, stiffness * mass, dedx, d2edx2);
        utility::atomic_embed_force<2>(edge, dedx, force);
        utility::atomic_embed_hessian<2>(edge, d2edx2, hess);
    }
}

__device__ void embed_face_force_hessian(const DataSet &data,
                                         const Vec<Vec3f> &eval_x,
                                         Vec<float> &force, FixedCSRMat &hess,
                                         float dt, const ParamSet &param,
                                         unsigned i) {
    const Vec3u &face = data.mesh.mesh.face[i];
    const FaceProp &prop = data.prop.face[i];
    const FaceParam &face_param = data.param_arrays.face[prop.param_index];
    const Vec3f &x0 = eval_x[face[0]];
    const Vec3f &x1 = eval_x[face[1]];
    const Vec3f &x2 = eval_x[face[2]];
    Mat3x3f dedx = Mat3x3f::Zero();
    Mat9x9f d2edx2 = Mat9x9f::Zero();
    float mass = prop.mass;
    float mu = face_param.mu;
    if (mu > 0.0f) {
        Mat3x2f F;
        Mat3x3f X;
        X << x0, x1, x2;
        F = utility::compute_deformation_grad(X, data.inv_rest2x2[i]);
        const Svd3x2 svd = utility::svd3x2(F);
        if (face_param.model == Model::BaraffWitkin) {
            Mat3x2f de0dF = BaraffWitkin::stretch_gradient(F, mu);
            Mat3x2f de1dF = BaraffWitkin::shear_gradient(F, face_param.lambda);
            Mat6x6f d2e0dF2 = BaraffWitkin::stretch_hessian(F, mu);
            Mat6x6f d2e1dF2 = BaraffWitkin::shear_hessian(F, face_param.lambda);
            Mat3x2f dedF = de0dF + de1dF;
            Mat6x6f d2edF2 = d2e0dF2 + d2e1dF2;
            dedx += mass * utility::convert_force(dedF, data.inv_rest2x2[i]);
            d2edx2 +=
                mass * utility::convert_hessian(d2edF2, data.inv_rest2x2[i]);
        } else {
            DiffTable2 table;
            Mat3x2f dedF;
            Mat6x6f d2edF2;
            if (face_param.model == Model::ARAP) {
                table = ARAP::make_diff_table2(svd.S, mu, face_param.lambda);
            } else if (face_param.model == Model::StVK) {
                table = StVK::make_diff_table2(svd.S, mu, face_param.lambda);
            } else if (face_param.model == Model::SNHk) {
                table = SNHk::make_diff_table2(svd.S, mu, face_param.lambda);
            } else {
                assert(false);
            }
            dedF = eigenanalysis::compute_force(table, svd);
            d2edF2 = eigenanalysis::compute_hessian(table, svd,
                                                    param.eiganalysis_eps);
            dedx += mass * utility::convert_force(dedF, data.inv_rest2x2[i]);
            d2edx2 +=
                mass * utility::convert_hessian(d2edF2, data.inv_rest2x2[i]);
        }
        utility::atomic_embed_force<3>(face, dedx, force);
        utility::atomic_embed_hessian<3>(face, d2edx2, hess);
    }
}

__device__ void embed_tet_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> &force, FixedCSRMat &hess,
                                        float dt, const ParamSet &param,
                                        unsigned i) {
    const Vec4u &tet = data.mesh.mesh.tet[i];
    const TetProp &prop = data.prop.tet[i];
    const TetParam &tet_param = data.param_arrays.tet[prop.param_index];
    const float mass = prop.mass;
    const float mu = tet_param.mu;
    if (mu > 0.0f) {
        const float lambda = tet_param.lambda;
        const Vec3f &x0 = eval_x[tet[0]];
        const Vec3f &x1 = eval_x[tet[1]];
        const Vec3f &x2 = eval_x[tet[2]];
        const Vec3f &x3 = eval_x[tet[3]];
        Mat3x4f X;
        X << x0, x1, x2, x3;
        Mat3x3f F = utility::compute_deformation_grad(X, data.inv_rest3x3[i]);
        Svd3x3 svd = utility::svd3x3_rv(F);
        DiffTable3 table;
        Mat3x3f dedF;
        Mat9x9f d2edF2;
        Mat3x4f dedx = Mat3x4f::Zero();
        Mat12x12f d2edx2 = Mat12x12f::Zero();
        if (tet_param.model == Model::ARAP) {
            table = ARAP::make_diff_table3(svd.S, mu, lambda);
        } else if (tet_param.model == Model::StVK) {
            table = StVK::make_diff_table3(svd.S, mu, lambda);
        } else if (tet_param.model == Model::SNHk) {
            table = SNHk::make_diff_table3(svd.S, mu, lambda);
        } else {
            assert(false);
        }
        dedF = eigenanalysis::compute_force(table, svd);
        d2edF2 =
            eigenanalysis::compute_hessian(table, svd, param.eiganalysis_eps);
        dedx += mass * utility::convert_force(dedF, data.inv_rest3x3[i]);
        d2edx2 += mass * utility::convert_hessian(d2edF2, data.inv_rest3x3[i]);
        utility::atomic_embed_force<4>(tet, dedx, force);
        utility::atomic_embed_hessian<4>(tet, d2edx2, hess);
    }
}

__device__ void embed_hinge_force_hessian(const DataSet &data,
                                          const Vec<Vec3f> &eval_x,
                                          Vec<float> &force, FixedCSRMat &hess,
                                          const ParamSet &param, unsigned i) {
    const HingeProp &prop = data.prop.hinge[i];
    const HingeParam &hinge_param = data.param_arrays.hinge[prop.param_index];
    float length = prop.length;
    float bend = hinge_param.bend;
    float ghat = hinge_param.ghat;
    float stiff_k = 2.0f * bend * length * ghat;
    if (stiff_k > 0.0f) {
        Vec4u hinge = data.mesh.mesh.hinge[i];
        Mat3x4f dedx;
        Mat12x12f d2edx2;
        dihedral_angle::face_compute_force_hessian(eval_x, hinge, dedx, d2edx2);
        utility::atomic_embed_force<4>(hinge, stiff_k * dedx, force);
        utility::atomic_embed_hessian<4>(hinge, stiff_k * d2edx2, hess);
    }
}

__device__ void
embed_rod_bend_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                             Vec<float> &force, FixedCSRMat &hess,
                             const ParamSet &param, unsigned i) {
    if (data.mesh.neighbor.vertex.edge.count(i) == 2 &&
        data.mesh.neighbor.vertex.face.count(i) == 0) {
        unsigned edge_idx_0 = data.mesh.neighbor.vertex.edge(i, 0);
        unsigned edge_idx_1 = data.mesh.neighbor.vertex.edge(i, 1);
        const EdgeParam &edge_param_0 =
            data.param_arrays.edge[data.prop.edge[edge_idx_0].param_index];
        const EdgeParam &edge_param_1 =
            data.param_arrays.edge[data.prop.edge[edge_idx_1].param_index];
        float bend_0 = edge_param_0.bend;
        float bend_1 = edge_param_1.bend;
        float bend = 0.5f * (bend_0 + bend_1);
        float mass = data.prop.vertex[i].mass;
        float stiff_k = bend * mass;
        if (mass > 0.0f && stiff_k > 0.0f) {
            Vec2u edge_0 = data.mesh.mesh.edge[edge_idx_0];
            Vec2u edge_1 = data.mesh.mesh.edge[edge_idx_1];
            unsigned j = edge_0[0] == i ? edge_0[1] : edge_0[0];
            unsigned k = edge_1[0] == i ? edge_1[1] : edge_1[0];
            Vec3u element = Vec3u(j, i, k);
            Vec3f x0 = eval_x[j];
            Vec3f x1 = eval_x[i];
            Vec3f x2 = eval_x[k];
            Mat3x3f dedx = dihedral_angle::strand_gradient(x0, x1, x2);
            Vec9f vec_dedx = Map<Vec9f>(dedx.data());
            Mat9x9f d2edx2 = vec_dedx * vec_dedx.transpose();
            utility::atomic_embed_force<3>(element, stiff_k * dedx, force);
            utility::atomic_embed_hessian<3>(element, stiff_k * d2edx2, hess);
        }
    }
}

void embed_momentum_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                                  const Vec<Vec3f> &velocity, float dt,
                                  const Vec<Vec3f> &target, Vec<float> &force,
                                  Vec<Mat3x3f> &diag_hess,
                                  const ParamSet &param) {
    DISPATCH_START(data.vertex.curr.size)
    [data, eval_x, velocity, dt, target, force, diag_hess,
     param] __device__(unsigned i) mutable {
        if (data.prop.vertex[i].fix_index == 0) {
            energy::embed_vertex_force_hessian(data, eval_x, velocity, target,
                                               force, diag_hess, dt, param, i);
        }
    } DISPATCH_END;
}

void embed_elastic_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                                 Vec<float> &force, FixedCSRMat &fixed_hess,
                                 float dt, const ParamSet &param) {
    unsigned surface_vert_count = data.surface_vert_count;
    unsigned hinge_count = data.mesh.mesh.hinge.size;
    unsigned shell_face_count = data.shell_face_count;
    unsigned rod_count = data.rod_count;
    unsigned tet_count = data.mesh.mesh.tet.size;

    DISPATCH_START(surface_vert_count)
    [data, eval_x, force, fixed_hess, dt,
     param] __device__(unsigned i) mutable {
        if (data.prop.vertex[i].fix_index == 0) {
            energy::embed_rod_bend_force_hessian(data, eval_x, force,
                                                 fixed_hess, param, i);
        }
    } DISPATCH_END;

    if (rod_count > 0) {
        DISPATCH_START(rod_count)
        [data, eval_x, force, fixed_hess, dt,
         param] __device__(unsigned i) mutable {
            if (!data.prop.edge[i].fixed) {
                energy::embed_rod_force_hessian(data, eval_x, force, fixed_hess,
                                                dt, param, i);
            }
        } DISPATCH_END;
    }

    if (shell_face_count > 0) {
        DISPATCH_START(shell_face_count)
        [data, eval_x, force, fixed_hess, dt,
         param] __device__(unsigned i) mutable {
            if (!data.prop.face[i].fixed) {
                energy::embed_face_force_hessian(data, eval_x, force,
                                                 fixed_hess, dt, param, i);
            }
        } DISPATCH_END;
    }

    if (tet_count > 0) {
        DISPATCH_START(tet_count)
        [data, eval_x, force, fixed_hess, dt,
         param] __device__(unsigned i) mutable {
            if (!data.prop.tet[i].fixed) {
                energy::embed_tet_force_hessian(data, eval_x, force, fixed_hess,
                                                dt, param, i);
            }
        } DISPATCH_END;
    }

    if (hinge_count > 0) {
        DISPATCH_START(hinge_count)
        [data, eval_x, force, fixed_hess,
         param] __device__(unsigned i) mutable {
            if (data.prop.hinge[i].fixed == false &&
                (data.mesh.type.hinge[i] & 1) == 0) {
                energy::embed_hinge_force_hessian(data, eval_x, force,
                                                  fixed_hess, param, i);
            }
        } DISPATCH_END;
    }
}

void embed_stitch_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                                Vec<float> &force, FixedCSRMat &fixed_out,
                                const ParamSet &param) {
    unsigned seam_count = data.constraint.stitch.size;
    if (seam_count) {
        DISPATCH_START(seam_count)
        [data, eval_x, force, fixed_out, param] __device__(unsigned i) mutable {
            const Stitch &stitch = data.constraint.stitch[i];
            Vec3u index(stitch.index[0], stitch.index[1], stitch.index[2]);
            const Vec3f &x0 = eval_x[index[0]];
            const Vec3f &x1 = eval_x[index[1]];
            const Vec3f &x2 = eval_x[index[2]];
            float ghat_0 =
                data.param_arrays.vertex[data.prop.vertex[index[0]].param_index]
                    .ghat;
            float ghat_1 =
                data.param_arrays.vertex[data.prop.vertex[index[1]].param_index]
                    .ghat;
            float ghat_2 =
                data.param_arrays.vertex[data.prop.vertex[index[2]].param_index]
                    .ghat;
            float w[] = {1.0f, 1.0f - stitch.weight, stitch.weight};
            float l0 = (w[0] * ghat_0 + w[1] * ghat_1 + w[2] * ghat_2) / 2.0f;
            float s(1.0f / 3.0f);
            const Vec3f cog = s * x0 + s * x1 + s * x2;
            Vec3f z0 = w[0] * (x0 - cog);
            Vec3f z1 = w[1] * (x1 - cog) + w[2] * (x2 - cog);
            Vec3f t = z0 - z1;
            float l = fmin(0.01f, t.norm());
            Vec3f n = t / l;
            Mat3x9f dtdx;
            dtdx << w[0] * Mat3x3f::Identity(), -w[1] * Mat3x3f::Identity(),
                -w[2] * Mat3x3f::Identity();
            Vec3f dedt = (l / l0 - 1.0f) * n;
            Vec9f g = dtdx.transpose() * n;
            float r = (l - l0) / l;
            float c0 = fmaxf(0.0f, 1.0f - r) / l0;
            float c1 = fmaxf(0.0f, r / l0);
            Mat3x3f gradient;
            gradient.col(0) = w[0] * dedt;
            gradient.col(1) = -w[1] * dedt;
            gradient.col(2) = -w[2] * dedt;
            Mat9x9f hessian =
                c0 * g * g.transpose() + c1 * dtdx.transpose() * dtdx;
            utility::atomic_embed_force<3>(
                index, param.stitch_stiffness * gradient, force);
            utility::atomic_embed_hessian<3>(
                index, param.stitch_stiffness * hessian, fixed_out);
        } DISPATCH_END;
    }
}

} // namespace energy
