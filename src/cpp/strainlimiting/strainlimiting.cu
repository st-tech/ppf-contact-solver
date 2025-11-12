// File: strainlimiting.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../barrier/barrier.hpp"
#include "../eigenanalysis/eigenanalysis.hpp"
#include "../kernels/reduce.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "strainlimiting.hpp"

namespace strainlimiting {

Vec<float> tmp_face;

__device__ float minimal_stretch_gap(const Svd3x2 &svd, float eps) {
    return eps - svd.S.maxCoeff();
}

__device__ float compute_stiffness(const Vec<Vec3f> &eval_x, const Vec3u &face,
                                   const FixedCSRMat &fixed_hess_in, float mass,
                                   const Svd3x2 &svd, float eps) {
    const Vec3f &x0 = eval_x[face[0]];
    const Vec3f &x1 = eval_x[face[1]];
    const Vec3f &x2 = eval_x[face[2]];
    const float s(1.0f / 3.0f);
    Vec3f center = s * x0 + s * x1 + s * x2;
    Mat3x3f R;
    R << (x0 - center), //
        (x1 - center),  //
        (x2 - center);
    Vec9f w = Map<Vec9f>(R.data());
    Mat9x9f local_hess = Mat9x9f::Zero();
    for (unsigned ii = 0; ii < 3; ++ii) {
        for (unsigned jj = 0; jj < 3; ++jj) {
            local_hess.block<3, 3>(3 * ii, 3 * jj) =
                fixed_hess_in(face[ii], face[jj]);
        }
    }
    float d = minimal_stretch_gap(svd, eps);
    return w.dot(local_hess * w) + mass / (d * d);
}

void embed_strainlimiting_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> &force,
                                        const FixedCSRMat &fixed_hess_in,
                                        FixedCSRMat &fixed_hess_out,
                                        const ParamSet &param) {

    const Vec<Vec3f> &curr = data.vertex.curr;
    unsigned shell_face_count = data.shell_face_count;
    Vec<FaceParam> face_params = data.param_arrays.face;

    DISPATCH_START(shell_face_count)
    [data, eval_x, curr, force, fixed_hess_in, fixed_hess_out, face_params,
     param] __device__(unsigned i) mutable {
        const FaceProp &prop = data.prop.face[i];
        const FaceParam &fparam = face_params[prop.param_index];
        if (!prop.fixed && fparam.strainlimit > 0.0f) {
            float limit = (fparam.strainlimit + 1.0f) / fparam.shrink - 1.0f;
            const Vec3u &face = data.mesh.mesh.face[i];
            Mat3x3f X;
            X << eval_x[face[0]], eval_x[face[1]], eval_x[face[2]];
            Mat3x2f F =
                utility::compute_deformation_grad(X, data.inv_rest2x2[i]);
            Svd3x2 svd = utility::svd3x2_shifted(F);
            if (svd.S.maxCoeff() > 0.0f) {
                const Vec3u &face = data.mesh.mesh.face[i];
                Mat3x3f dedx = Mat3x3f::Zero();
                Mat9x9f d2edx2 = Mat9x9f::Zero();
                DiffTable2 table = barrier::compute_strainlimiting_diff_table(
                    svd.S, fparam.strainlimit, param.barrier);
                float mass = data.prop.face[i].mass;
                float stiffness = compute_stiffness(
                    data.vertex.curr, face, fixed_hess_in, mass, svd, limit);
                svd.S += Vec2f::Ones();
                Mat3x2f dedF = eigenanalysis::compute_force(table, svd);
                Mat6x6f d2edF2 = eigenanalysis::compute_hessian(
                    table, svd, param.eiganalysis_eps);
                dedx = utility::convert_force(dedF, data.inv_rest2x2[i]);
                d2edx2 = utility::convert_hessian(d2edF2, data.inv_rest2x2[i]);
                utility::atomic_embed_force<3>(face, stiffness * dedx, force);
                utility::atomic_embed_hessian<3>(face, stiffness * d2edx2,
                                                 fixed_hess_out);
            }
        }
    } DISPATCH_END;
}

float line_search(const DataSet &data, const Vec<Vec3f> &eval_x,
                  const Vec<Vec3f> &prev, Vec<float> &tmp_face,
                  const ParamSet &param) {

    const MeshInfo &mesh = data.mesh;
    const Vec<Mat2x2f> &inv_rest2x2 = data.inv_rest2x2;
    unsigned shell_face_count = data.shell_face_count;
    Vec<float> toi = tmp_face;
    toi.size = shell_face_count;
    Vec<FaceParam> face_params = data.param_arrays.face;

    DISPATCH_START(shell_face_count)
    [data, inv_rest2x2, mesh, eval_x, prev, param, toi,
     face_params] __device__(unsigned i) mutable {
        float t = param.line_search_max_t;
        const FaceProp &prop = data.prop.face[i];
        const FaceParam &fparam = face_params[prop.param_index];
        if (!prop.fixed && fparam.strainlimit > 0.0f) {
            float limit = (fparam.strainlimit + 1.0f) / fparam.shrink - 1.0f;
            Vec3u face = mesh.mesh.face[i];
            Mat3x3f x0, x1;
            x0 << prev[face[0]], prev[face[1]], prev[face[2]];
            x1 << eval_x[face[0]], eval_x[face[1]], eval_x[face[2]];
            const Mat3x2f F0 =
                utility::compute_deformation_grad(x0, inv_rest2x2[i]);
            const Mat3x2f F1 =
                utility::compute_deformation_grad(x1, inv_rest2x2[i]);
            const Mat3x2f dF = F1 - F0;
            if (utility::singular_vals_minus_one(F0 + t * dF).maxCoeff() >=
                limit) {
                float upper_t = t;
                float lower_t = 0.0f;
                float window = upper_t - lower_t;
                while (true) {
                    t = 0.5f * (upper_t + lower_t);
                    Vec2f singular_vals =
                        utility::singular_vals_minus_one(F0 + t * dF);
                    float diff = singular_vals.maxCoeff() - limit;
                    if (diff < 0.0f) {
                        lower_t = t;
                    } else {
                        upper_t = t;
                    }
                    float new_window = upper_t - lower_t;
                    if (new_window == window) {
                        break;
                    } else {
                        window = new_window;
                    }
                }
                t = lower_t;
            }
        }
        toi[i] = t;
    } DISPATCH_END;

    float result =
        kernels::min_array(toi.data, shell_face_count, param.line_search_max_t);
    return result / param.line_search_max_t;
}

} // namespace strainlimiting
