// File: energy.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../eigenanalysis/eigenanalysis.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "model/air_damper.hpp"
#include "model/arap.hpp"
#include "model/inflate.hpp"
#include "model/baraffwitkin.hpp"
#include "model/dihedral_angle.hpp"
#include "model/hook.hpp"
#include "model/momentum.hpp"
#include "model/snhk.hpp"
#include "model/stvk.hpp"

// 3x3 symmetric eigendecomposition for PCA on device.
// Returns eigenvectors sorted by descending eigenvalue.
__device__ void eigen_sym3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    Vec3f evals, Mat3x3f &evecs) {

    // Eigenvalues via Cardano's formula for symmetric 3x3
    float q = (a00 + a11 + a22) / 3.0f;
    float p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) +
               (a22 - q) * (a22 - q) + 2.0f * (a01 * a01 + a02 * a02 + a12 * a12);
    float p = sqrtf(fmaxf(p2 / 6.0f, 0.0f));

    float lam0, lam1, lam2;
    if (p < 1e-12f) {
        // Matrix is (near) scalar multiple of identity
        lam0 = a00; lam1 = a11; lam2 = a22;
    } else {
        float inv_p = 1.0f / p;
        // B = (A - qI) / p
        float b00 = (a00 - q) * inv_p, b01 = a01 * inv_p, b02 = a02 * inv_p;
        float b11 = (a11 - q) * inv_p, b12 = a12 * inv_p;
        float b22 = (a22 - q) * inv_p;
        float det_b = b00 * (b11 * b22 - b12 * b12)
                    - b01 * (b01 * b22 - b12 * b02)
                    + b02 * (b01 * b12 - b11 * b02);
        float r = fminf(1.0f, fmaxf(-1.0f, det_b * 0.5f));
        float phi = acosf(r) / 3.0f;
        lam0 = q + 2.0f * p * cosf(phi);
        lam2 = q + 2.0f * p * cosf(phi + 2.0943951f); // 2π/3
        lam1 = 3.0f * q - lam0 - lam2;
    }

    // Sort descending
    if (lam1 > lam0) { float t = lam0; lam0 = lam1; lam1 = t; }
    if (lam2 > lam0) { float t = lam0; lam0 = lam2; lam2 = t; }
    if (lam2 > lam1) { float t = lam1; lam1 = lam2; lam2 = t; }
    evals = Vec3f(lam0, lam1, lam2);

    // Eigenvectors via cross product of rows of (A - λI)
    float vals[3] = {lam0, lam1, lam2};
    for (int c = 0; c < 3; ++c) {
        float l = vals[c];
        Vec3f r0(a00 - l, a01, a02);
        Vec3f r1(a01, a11 - l, a12);
        Vec3f r2(a02, a12, a22 - l);
        Vec3f v = r0.cross(r1);
        float vn = v.norm();
        if (vn < 1e-8f) { v = r0.cross(r2); vn = v.norm(); }
        if (vn < 1e-8f) { v = r1.cross(r2); vn = v.norm(); }
        if (vn > 1e-8f) {
            evecs.col(c) = v / vn;
        } else {
            evecs.col(c) = Vec3f::Unit(c);
        }
    }

    // Gram-Schmidt to ensure orthogonality
    Vec3f e0 = evecs.col(0).normalized();
    Vec3f e1 = evecs.col(1) - e0 * e0.dot(evecs.col(1));
    e1.normalize();
    Vec3f e2 = e0.cross(e1);
    evecs.col(0) = e0;
    evecs.col(1) = e1;
    evecs.col(2) = e2;
}

namespace energy {

__device__ void embed_vertex_force_hessian(
    const DataSet &data, const Vec<Vec3f> &eval_x, const Vec<Vec3f> &velocity,
    const Vec<Vec3f> &target, Vec<float> &force, Vec<Mat3x3f> &diag_hess,
    float dt, const ParamSet &param,
    const Vec<TorqueGroupResult> &torque_result, unsigned i) {

    float mass = data.prop.vertex[i].mass;
    float area = data.prop.vertex[i].area;

    const Vec3f &x = data.vertex.curr[i];
    const Vec3f &y = eval_x[i];
    const Vec3f normal = utility::compute_vertex_normal(data, eval_x, i);

    Vec3f wind = Vec3f::Zero();
    if (!param.inactive_momentum) {
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

    for (unsigned j = 0; j < data.constraint.torque_vertices.size; ++j) {
        if (i == data.constraint.torque_vertices[j].index) {
            float mag = data.constraint.torque_vertices[j].magnitude;
            unsigned gid = data.constraint.torque_vertices[j].group_id;

            // Read pre-computed centroid and PCA axis from pre-pass
            Vec3f center = torque_result[gid].center;
            Vec3f axis = torque_result[gid].axis;

            Vec3f r = y - center;
            Vec3f r_perp = r - axis * axis.dot(r);
            // F_i = axis × r_perp_i * magnitude / Σ|r_perp_j|²
            // ensures total torque = magnitude and zero net linear force
            float scale = mag * torque_result[gid].inv_r_perp_sq_sum;
            Vec3f torque_force = axis.cross(r_perp) * scale;
            f -= torque_force;
            // Symmetric Hessian approximation so Newton converges for
            // the torque term regardless of iteration count.
            // ∂(axis×r_perp)/∂y has a skew part (unusable) and a
            // symmetric magnitude part scale*(I - axis⊗axis).
            Mat3x3f P = Mat3x3f::Identity() - axis * axis.transpose();
            H += scale * P;
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

    if (param.fix_xz && y[1] > param.fix_xz) {
        float t = fmin(1.0f, y[1] - param.fix_xz);
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

    if (face_param.pressure > 0.0f) {
        utility::atomic_embed_force<3>(
            face, inflate::face_gradient(face_param.pressure, x0, x1, x2), force);
        utility::atomic_embed_hessian<3>(
            face, inflate::face_hessian(face_param.pressure, x0, x1, x2), hess);
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
        dihedral_angle::face_compute_force_hessian(eval_x, hinge,
                                                   prop.rest_angle, dedx,
                                                   d2edx2);
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
        const EdgeParam &edge_param_0 = data.param_arrays.edge[data.prop.edge[edge_idx_0].param_index];
        const EdgeParam &edge_param_1 = data.param_arrays.edge[data.prop.edge[edge_idx_1].param_index];
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
            float rest_angle = data.prop.vertex[i].rest_bend_angle;
            Mat3x3f dedx;
            Mat9x9f d2edx2;
            dihedral_angle::strand_compute_force_hessian(
                x0, x1, x2, rest_angle, dedx, d2edx2);
            utility::atomic_embed_force<3>(element, stiff_k * dedx, force);
            utility::atomic_embed_hessian<3>(element, stiff_k * d2edx2, hess);
        }
    }
}

void compute_torque_groups(const DataSet &data,
                           const Vec<Vec3f> &eval_x,
                           Vec<TorqueGroupResult> &result) {
    DISPATCH_START(data.constraint.torque_groups.size)
    [data, eval_x, result] __device__(unsigned g) mutable {
        const auto &grp = data.constraint.torque_groups[g];
        unsigned vs = grp.vertex_start;
        unsigned vn = grp.vertex_count;

        // Mass-weighted centroid (center of gravity)
        Vec3f center = Vec3f::Zero();
        float total_mass = 0.0f;
        for (unsigned k = vs; k < vs + vn; ++k) {
            unsigned idx = data.constraint.torque_vertices[k].index;
            float m = data.prop.vertex[idx].mass;
            center += eval_x[idx] * m;
            total_mass += m;
        }
        if (total_mass > 1e-12f) {
            center *= 1.0f / total_mass;
        }

        // 3x3 covariance matrix
        float c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
        for (unsigned k = vs; k < vs + vn; ++k) {
            Vec3f d = eval_x[data.constraint.torque_vertices[k].index] - center;
            c00 += d[0] * d[0]; c01 += d[0] * d[1]; c02 += d[0] * d[2];
            c11 += d[1] * d[1]; c12 += d[1] * d[2]; c22 += d[2] * d[2];
        }
        float inv_n = 1.0f / (float)vn;
        c00 *= inv_n; c01 *= inv_n; c02 *= inv_n;
        c11 *= inv_n; c12 *= inv_n; c22 *= inv_n;

        // Eigendecompose
        Vec3f evals;
        Mat3x3f evecs;
        eigen_sym3x3(c00, c01, c02, c11, c12, c22, evals, evecs);

        result[g].center = center;
        Vec3f ax = evecs.col(grp.axis_component).normalized();
        // Resolve sign ambiguity using hint vertex: orient axis so
        // dot(axis, hint_vertex - centroid) > 0
        Vec3f hint_dir = eval_x[grp.hint_vertex] - center;
        if (ax.dot(hint_dir) < 0.0f) ax = -ax;
        result[g].axis = ax;

        // Compute Σ|r_perp_i|² for torque normalization so total torque = magnitude
        float r_perp_sq_sum = 0.0f;
        for (unsigned k = vs; k < vs + vn; ++k) {
            Vec3f r = eval_x[data.constraint.torque_vertices[k].index] - center;
            Vec3f r_perp = r - ax * ax.dot(r);
            r_perp_sq_sum += r_perp.squaredNorm();
        }
        result[g].inv_r_perp_sq_sum = (r_perp_sq_sum > 1e-12f) ? 1.0f / r_perp_sq_sum : 0.0f;
    } DISPATCH_END;
}

void embed_momentum_force_hessian(const DataSet &data,
                                  const Vec<Vec3f> &eval_x,
                                  const Vec<Vec3f> &velocity, float dt,
                                  const Vec<Vec3f> &target, Vec<float> &force,
                                  Vec<Mat3x3f> &diag_hess,
                                  const ParamSet &param,
                                  const Vec<TorqueGroupResult> &torque_result) {
    DISPATCH_START(data.vertex.curr.size)
    [data, eval_x, velocity, dt, target, force, diag_hess,
     param, torque_result] __device__(unsigned i) mutable {
        if (data.prop.vertex[i].fix_index == 0) {
            energy::embed_vertex_force_hessian(data, eval_x, velocity, target,
                                               force, diag_hess, dt, param,
                                               torque_result, i);
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
            Vec4u index(stitch.index[0], stitch.index[1], stitch.index[2], stitch.index[3]);
            float ws = stitch.weight[0]; // source weight (always 1.0)
            float w1 = stitch.weight[1], w2 = stitch.weight[2], w3 = stitch.weight[3];

            const Vec3f &xs = eval_x[index[0]]; // source vertex
            // Barycentric target: w1*t0 + w2*t1 + w3*t2
            Vec3f target = eval_x[index[1]] * w1
                          + eval_x[index[2]] * w2
                          + eval_x[index[3]] * w3;

            // Rest length from contact gaps
            float ghat_s = data.param_arrays.vertex[data.prop.vertex[index[0]].param_index].ghat;
            float ghat_t = w1 * data.param_arrays.vertex[data.prop.vertex[index[1]].param_index].ghat
                         + w2 * data.param_arrays.vertex[data.prop.vertex[index[2]].param_index].ghat
                         + w3 * data.param_arrays.vertex[data.prop.vertex[index[3]].param_index].ghat;
            float l0 = (ghat_s + ghat_t) / 2.0f;

            Vec3f t = ws * (xs - target);
            float l = t.norm();
            if (l < 1e-10f) return;
            Vec3f n = t / l;

            // Gradient: dt/dx for 4 vertices (3x12 matrix)
            // dt/dx_s = ws * I, dt/dx_t0 = -ws*w1*I, dt/dx_t1 = -ws*w2*I, dt/dx_t2 = -ws*w3*I
            using Mat3x12f = Eigen::Matrix<float, 3, 12>;
            using Vec12f = Eigen::Vector<float, 12>;
            using Mat12x12f = Eigen::Matrix<float, 12, 12>;
            Mat3x12f dtdx;
            dtdx << ws * Mat3x3f::Identity(),
                    -ws * w1 * Mat3x3f::Identity(),
                    -ws * w2 * Mat3x3f::Identity(),
                    -ws * w3 * Mat3x3f::Identity();

            Vec3f dedt = (l / l0 - 1.0f) * n;
            Vec12f g = dtdx.transpose() * n;
            float r = (l - l0) / l;
            float c0 = fmaxf(0.0f, 1.0f - r) / l0;
            float c1 = fmaxf(0.0f, r / l0);

            Eigen::Matrix<float, 3, 4> gradient;
            gradient.col(0) = ws * dedt;
            gradient.col(1) = -ws * w1 * dedt;
            gradient.col(2) = -ws * w2 * dedt;
            gradient.col(3) = -ws * w3 * dedt;
            Mat12x12f hessian =
                c0 * g * g.transpose() + c1 * dtdx.transpose() * dtdx;

            utility::atomic_embed_force<4>(
                index, param.stitch_stiffness * gradient, force);
            utility::atomic_embed_hessian<4>(
                index, param.stitch_stiffness * hessian, fixed_out);
        } DISPATCH_END;
    }
}

} // namespace energy
