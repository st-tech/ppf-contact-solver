// File: strainlimiting.cu
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef STRAINLIMITING_HPP
#define STRAINLIMITING_HPP

#include "../barrier/barrier.hpp"
#include "../eigenanalysis/eigenanalysis.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "strainlimiting.hpp"
#include <limits>

namespace strainlimiting {

Vec<float> tmp_face;

__device__ float minimal_stretch_gap(const Svd3x2 &svd, float tau, float eps) {
    float max_sigma = 1.0f + tau + eps;
    return max_sigma - svd.S.maxCoeff();
}

__device__ float compute_stiffness(const Vec<Vec3f> &eval_x, const Vec3u &face,
                                   const FixedCSRMat &fixed_hess_in, float mass,
                                   const Svd3x2 &svd, float tau, float eps) {
    const Vec3f &x0 = eval_x[face[0]];
    const Vec3f &x1 = eval_x[face[1]];
    const Vec3f &x2 = eval_x[face[2]];
    Vec3f center = (x0 + x1 + x2) / 3.0f;
    Mat3x3f R;
    R << x0 - center, x1 - center, x2 - center;
    Vec9f w = Map<Vec9f>(R.data());
    Mat9x9f local_hess = Mat9x9f::Zero();
    for (unsigned ii = 0; ii < 3; ++ii) {
        for (unsigned jj = 0; jj < 3; ++jj) {
            local_hess.block<3, 3>(3 * ii, 3 * jj) =
                fixed_hess_in(face[ii], face[jj]);
        }
    }
    float d = minimal_stretch_gap(svd, tau, eps);
    return w.dot(local_hess * w) + mass / (d * d);
}

void embed_strainlimiting_force_hessian(
    const DataSet &data, const Vec<Vec3f> &eval_x, const Kinematic &kinematic,
    Vec<float> &force, const FixedCSRMat &fixed_hess_in,
    FixedCSRMat &fixed_hess_out, const ParamSet &param) {

    const Vec<Vec3f> &curr = data.vertex.curr;
    unsigned shell_face_count = data.shell_face_count;
    float tau(param.strain_limit_tau), eps(param.strain_limit_eps);

    DISPATCH_START(shell_face_count)
    [data, eval_x, kinematic, curr, force, fixed_hess_in, fixed_hess_out, tau,
     eps, param] __device__(unsigned i) mutable {
        if (!kinematic.face[i]) {
            const Vec3u &face = data.mesh.mesh.face[i];
            Mat3x3f X;
            X << eval_x[face[0]], eval_x[face[1]], eval_x[face[2]];
            Mat3x2f F =
                utility::compute_deformation_grad(X, data.inv_rest2x2[i]);
            Svd3x2 svd = utility::svd3x2(F);
            if (svd.S.maxCoeff() > 1.0f + tau) {
                const Vec3u &face = data.mesh.mesh.face[i];
                Mat3x3f dedx = Mat3x3f::Zero();
                Mat9x9f d2edx2 = Mat9x9f::Zero();
                DiffTable2 table = barrier::compute_strainlimiting_diff_table(
                    svd.S, tau, eps, param.barrier);
                Mat3x2f dedF = eigenanalysis::compute_force(table, svd);
                Mat6x6f d2edF2 = eigenanalysis::compute_hessian(
                    table, svd, param.eiganalysis_eps);
                dedx = utility::convert_force(dedF, data.inv_rest2x2[i]);
                d2edx2 = utility::convert_hessian(d2edF2, data.inv_rest2x2[i]);
                float mass = data.prop.face[i].mass;
                float stiffness = compute_stiffness(
                    data.vertex.curr, face, fixed_hess_in, mass, svd, tau, eps);
                utility::atomic_embed_force<3>(face, stiffness * dedx, force);
                utility::atomic_embed_hessian<3>(face, stiffness * d2edx2,
                                                 fixed_hess_out);
            }
        }
    } DISPATCH_END;
}

float line_search(const DataSet &data, const Kinematic &kinematic,
                  const Vec<Vec3f> &eval_x, const Vec<Vec3f> &prev,
                  Vec<float> &tmp_face, const ParamSet &param) {

    const MeshInfo &mesh = data.mesh;
    const Vec<Mat2x2f> &inv_rest2x2 = data.inv_rest2x2;
    unsigned shell_face_count = data.shell_face_count;
    Vec<float> toi = tmp_face;
    toi.size = shell_face_count;
    float tau(param.strain_limit_tau), eps(param.strain_limit_eps);
    float max_sigma = 1.0f + tau + eps;

    DISPATCH_START(shell_face_count)
    [inv_rest2x2, kinematic, mesh, eval_x, prev, param, toi,
     max_sigma] __device__(unsigned i) mutable {
        float t = param.line_search_max_t;
        if (!kinematic.face[i]) {
            Vec3u face = mesh.mesh.face[i];
            Mat3x3f x0, x1;
            x0 << prev[face[0]], prev[face[1]], prev[face[2]];
            x1 << eval_x[face[0]], eval_x[face[1]], eval_x[face[2]];
            const Mat3x2f F0 =
                utility::compute_deformation_grad(x0, inv_rest2x2[i]);
            const Mat3x2f F1 =
                utility::compute_deformation_grad(x1, inv_rest2x2[i]);
            const Mat3x2f dF = F1 - F0;
            float gap = max_sigma - utility::svd3x2(F0).S.maxCoeff();
            float target = max_sigma - param.ccd_reduction * gap;
            if (utility::svd3x2(F0 + t * dF).S.maxCoeff() > target) {
                float upper_t = t;
                float lower_t = 0.0f;
                unsigned iter(0);
                while (true) {
                    t = 0.5f * (upper_t + lower_t);
                    Svd3x2 svd = utility::svd3x2(F0 + t * dF);
                    float diff = svd.S.maxCoeff() - target;
                    if (diff < 0.0f) {
                        lower_t = t;
                    } else {
                        upper_t = t;
                    }
                    if (lower_t > 0.0f) {
                        if (upper_t - lower_t < param.ccd_reduction * lower_t) {
                            break;
                        }
                    }
                    assert(t > std::numeric_limits<float>::epsilon());
                }
                t = lower_t;
            }
            assert(utility::svd3x2(F0 + t * dF).S.maxCoeff() <= target);
        }
        toi[i] = t;
    } DISPATCH_END;

    float result =
        utility::min_array(toi.data, shell_face_count, param.line_search_max_t);
    return result / param.line_search_max_t;
}

} // namespace strainlimiting

#endif
