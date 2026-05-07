// File: utility.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef UTIL_DEF_HPP
#define UTIL_DEF_HPP

#include "../common.hpp"
#include "../csrmat/csrmat.hpp"
#include "../data.hpp"

namespace utility {

__device__ Vec3f compute_vertex_normal(const DataSet &data,
                                       const Vec<Vec3f> &vertex, unsigned i);

__device__ void solve_symm_eigen2x2(const Mat2x2f &matrix, Vec2f &eigenvalues,
                                    Mat2x2f &eigenvectors);
__device__ void solve_symm_eigen3x3(const Mat3x3f &matrix, Vec3f &eigenvalues,
                                    Mat3x3f &eigenvectors);
__device__ Vec2f singular_vals_minus_one(const Mat3x2f &F);
__device__ Svd3x2 svd3x2(const Mat3x2f &F);
__device__ Svd3x2 svd3x2_shifted(const Mat3x2f &F);
__device__ Svd3x3 svd3x3_rv(const Mat3x3f &F);

template <unsigned N, class MatType>
__device__ static void
atomic_embed_hessian(const Eigen::Vector<unsigned, N> &index,
                     const Eigen::Matrix<float, N * 3, N * 3> &H,
                     MatType &mat) {
    for (unsigned ii = 0; ii < N; ++ii) {
        for (unsigned jj = 0; jj < N; ++jj) {
            Mat3x3f val = H.template block<3, 3>(ii * 3, jj * 3);
            mat.push(index[ii], index[jj], val);
        }
    }
}

template <unsigned N>
__device__ static void
atomic_embed_force(const Eigen::Vector<unsigned, N> &index,
                   const Eigen::Matrix<float, 3, N> &f, Vec<float> &force) {
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned ii = 0; ii < 3; ++ii) {
            const unsigned row = index[i] * 3 + ii;
            const float val = f(ii, i);
            force.atomic_add(row, val);
        }
    }
}

__device__ Mat3x3f convert_force(const Mat3x2f &dedF,
                                 const Mat2x2f &inv_rest2x2);
__device__ Mat3x4f convert_force(const Mat3x3f &dedF,
                                 const Mat3x3f &inv_rest3x3);
__device__ Mat9x9f convert_hessian(const Mat6x6f &d2ed2f,
                                   const Mat2x2f &inv_rest2x2);
__device__ Mat12x12f convert_hessian(const Mat9x9f &d2ed2f,
                                     const Mat3x3f &inv_rest3x3);
__device__ Mat3x2f compute_deformation_grad(const Mat3x3f &x,
                                            const Mat2x2f &inv_rest2x2);
__device__ Mat3x3f compute_deformation_grad(const Mat3x4f &x,
                                            const Mat3x3f &inv_rest3x3);
__device__ float compute_face_area(const Mat3x3f &vertex);

void compute_svd(DataSet data, Vec<Vec3f> curr, Vec<Svd3x2> svd,
                 ParamSet param);
__device__ float get_wind_weight(float time);

} // namespace utility

#endif
