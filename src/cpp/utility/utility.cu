// File: utility.cu
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef UTIL_HPP
#define UTIL_HPP

#include "dispatcher.hpp"
#include "utility.hpp"
#include <limits>

#define _real_ float
#define USE_EIGEN_SYMM_EIGSOLVE
#include "eig-hpp/eigsolve2x2.hpp"
#include "eig-hpp/eigsolve3x3.hpp"

#define BLOCK_SIZE 256

namespace utility {

__device__ Vec3f compute_vertex_normal(const DataSet &data,
                                       const Vec<Vec3f> &vertex, unsigned i) {
    Vec3f normal = Vec3f::Zero();
    if (data.mesh.neighbor.vertex.face.size) {
        for (unsigned j = 0; j < data.mesh.neighbor.vertex.face.count(i); ++j) {
            const Vec3u &face =
                data.mesh.mesh.face[data.mesh.neighbor.vertex.face(i, j)];
            const Vec3f &z0 = vertex[face[0]];
            const Vec3f &z1 = vertex[face[1]];
            const Vec3f &z2 = vertex[face[2]];
            normal += (z1 - z0).cross(z2 - z0);
        }
        if (normal.squaredNorm()) {
            normal.normalize();
        }
    }
    return normal;
}

__device__ void solve_symm_eigen2x2(const Mat2x2f &matrix, Vec2f &eigenvalues,
                                    Mat2x2f &eigenvectors) {
    eig_tuple_2x2 result = sym_eigsolve_2x2(matrix);
    eigenvalues = result.lambda;
    eigenvectors = result.eigvecs;
}

__device__ void solve_symm_eigen3x3(const Mat3x3f &matrix, Vec3f &eigenvalues,
                                    Mat3x3f &eigenvectors) {
    eig_tuple_3x3 result = sym_eigsolve_3x3(matrix);
    eigenvalues = result.lambda;
    eigenvectors = result.eigvecs;
}

__device__ Svd3x2 svd3x2(const Mat3x2f &F) {
    eig_tuple_2x2 result = sym_eigsolve_2x2(F.transpose() * F);
    Vec2f sigma = result.lambda;
    Mat2x2f V = result.eigvecs;
    for (int i = 0; i < 2; ++i) {
        sigma[i] = sqrtf(fmax(0.0f, sigma[i]));
    }
    Mat3x2f U = F * V;
    for (int i = 0; i < U.cols(); i++) {
        U.col(i).normalize();
    }
    return {U, sigma, V.transpose()};
}

__device__ Svd3x3 svd3x3(const Mat3x3f &F) {
    eig_tuple_3x3 result = sym_eigsolve_3x3(F.transpose() * F);
    Vec3f sigma = result.lambda;
    Mat3x3f V = result.eigvecs;
    for (int i = 0; i < 3; ++i) {
        sigma[i] = sqrtf(fmax(0.0f, sigma[i]));
    }
    Mat3x3f U = F * V;
    for (int i = 0; i < U.cols(); i++) {
        U.col(i).normalize();
    }
    return {U, sigma, V.transpose()};
}

__device__ Svd3x3 svd3x3_rv(const Mat3x3f &F) {
    Svd3x3 svd = svd3x3(F);
    float det_u = svd.U.determinant();
    float det_vt = svd.Vt.determinant();
    Mat3x3f L = Mat3x3f::Identity();
    unsigned min_index;
    svd.S.minCoeff(&min_index);
    L(min_index, min_index) = -1.0f;
    if (det_u < 0.0f && det_vt > 0.0f) {
        svd.U = svd.U * L;
        svd.S[min_index] *= -1.0f;
    } else if (det_u > 0.0f && det_vt < 0.0f) {
        svd.Vt = L * svd.Vt;
        svd.S[min_index] *= -1.0f;
    }
    return svd;
}

__device__ Mat3x2f make_diff_mat3x2() {
    Mat3x2f result = Mat3x2f::Zero();
    result(0, 0) = -1.0f;
    result(0, 1) = -1.0f;
    result(1, 0) = 1.0f;
    result(2, 1) = 1.0f;
    return result;
}

__device__ Mat4x3f make_diff_mat4x3() {
    Mat4x3f result = Mat4x3f::Zero();
    result(0, 0) = -1.0f;
    result(0, 1) = -1.0f;
    result(0, 2) = -1.0f;
    result(1, 0) = 1.0f;
    result(2, 1) = 1.0f;
    result(3, 2) = 1.0f;
    return result;
}

__device__ Mat3x3f convert_force(const Mat3x2f &dedF,
                                 const Mat2x2f &inv_rest2x2) {
    const Mat3x2f g = make_diff_mat3x2() * inv_rest2x2;
    Mat3x3f result;
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned dim = 0; dim < 3; ++dim) {
            result(dim, i) = g.row(i).dot(dedF.row(dim));
        }
    }
    return result;
}

__device__ Mat3x4f convert_force(const Mat3x3f &dedF,
                                 const Mat3x3f &inv_rest3x3) {
    const Mat4x3f g = make_diff_mat4x3() * inv_rest3x3;
    Mat3x4f result;
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned dim = 0; dim < 3; ++dim) {
            result(dim, i) = g.row(i).dot(dedF.row(dim));
        }
    }
    return result;
}

__device__ Mat9x9f convert_hessian(const Mat6x6f &d2ed2f,
                                   const Mat2x2f &inv_rest2x2) {
    const Mat3x2f g = make_diff_mat3x2() * inv_rest2x2;
    Mat6x9f dfdx;
    for (unsigned j = 0; j < 9; ++j) {
        Mat3x3f dx_mat = Mat3x3f::Zero();
        Map<Vec9f>(dx_mat.data())[j] = 1.0f;
        Mat3x2f tmp = dx_mat * g;
        dfdx.col(j) = Map<Vec6f>(tmp.data());
    }
    Mat9x9f result = Mat9x9f::Zero();
    for (unsigned i = 0; i < 6; ++i) {
        for (unsigned j = 0; j < 6; ++j) {
            result += d2ed2f(i, j) * dfdx.row(i).transpose() * dfdx.row(j);
        }
    }
    return result; // dfdx.transpose() * d2ed2f * dfdx;
}

__device__ Mat12x12f convert_hessian(const Mat9x9f &d2ed2f,
                                     const Mat3x3f &inv_rest3x3) {
    const Mat4x3f g = make_diff_mat4x3() * inv_rest3x3;
    Mat9x12f dfdx;
    for (unsigned j = 0; j < 12; ++j) {
        Mat3x4f dx_mat = Mat3x4f::Zero();
        Map<Vec12f>(dx_mat.data())[j] = 1.0f;
        Mat3x3f tmp = dx_mat * g;
        dfdx.col(j) = Map<Vec9f>(tmp.data());
    }
    Mat12x12f result = Mat12x12f::Zero();
    for (unsigned i = 0; i < 9; ++i) {
        for (unsigned j = 0; j < 9; ++j) {
            result += d2ed2f(i, j) * dfdx.row(i).transpose() * dfdx.row(j);
        }
    }
    return result; // dfdx.transpose() * d2ed2f * dfdx;
}

__device__ Mat3x2f compute_deformation_grad(const Mat3x3f &x,
                                            const Mat2x2f &inv_rest2x2) {
    return x * make_diff_mat3x2() * inv_rest2x2;
}

__device__ Mat3x3f compute_deformation_grad(const Mat3x4f &x,
                                            const Mat3x3f &inv_rest3x3) {
    return x * make_diff_mat4x3() * inv_rest3x3;
}

__device__ float compute_face_area(const Mat3x3f &vertex) {
    const Vec3f v0 = vertex.col(0);
    const Vec3f v1 = vertex.col(1);
    const Vec3f v2 = vertex.col(2);
    return 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

template <class T, class Y, typename Op>
__global__ void reduce_op_kernel(const T *input, Y *output, Op func, Y init_val,
                                 unsigned n) {
    __shared__ unsigned char _shared_data[BLOCK_SIZE];
    Y *shared_data = reinterpret_cast<Y *>(_shared_data);
    unsigned tid = threadIdx.x;
    unsigned global_idx = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = (global_idx < n) ? input[global_idx] : init_val;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] =
                func(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

template <class T, class Y, typename Op>
Y reduce(const T *d_input, Op func, Y init_val, unsigned n) {
    unsigned grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static Y *d_output = nullptr;
    static Y *h_results = nullptr;
    static unsigned max_grid_size = 0;
    if (d_output == nullptr) {
        cudaMalloc(&d_output, grid_size * sizeof(Y));
        h_results = new Y[grid_size];
        max_grid_size = grid_size;
    } else if (grid_size > max_grid_size) {
        max_grid_size = grid_size;
        cudaFree(d_output);
        delete[] h_results;
        cudaMalloc(&d_output, grid_size * sizeof(Y));
        h_results = new Y[grid_size];
    }
    size_t shared_mem_size = BLOCK_SIZE * sizeof(Y);
    reduce_op_kernel<T, Y><<<grid_size, BLOCK_SIZE, shared_mem_size>>>(
        d_input, d_output, func, init_val, n);
    cudaMemcpy(h_results, d_output, grid_size * sizeof(Y),
               cudaMemcpyDeviceToHost);
    Y result = init_val;
    for (unsigned i = 0; i < grid_size; i++) {
        result = func(result, h_results[i]);
    }
    return result;
}

template <class T> T sum_array(Vec<T> array, unsigned size) {
    return reduce<T, T>(
        array.data, [] __host__ __device__(T a, T b) { return a + b; }, T(),
        size);
}

template <class T> unsigned sum_integer_array(Vec<T> array, unsigned size) {
    return reduce<T, unsigned>(
        array.data, [] __host__ __device__(T a, T b) { return a + b; }, 0u,
        size);
}

template <class T> T min_array(const T *array, unsigned size, T init_val) {
    return reduce<T, T>(
        array, [] __host__ __device__(T a, T b) { return a < b ? a : b; },
        init_val, size);
}

template <class T> T max_array(const T *array, unsigned size, T init_val) {
    return reduce<T, T>(
        array, [] __host__ __device__(T a, T b) { return a > b ? a : b; },
        init_val, size);
}

void compute_svd(DataSet data, Vec<Vec3f> curr, Vec<Svd3x2> svd,
                 ParamSet param) {
    unsigned shell_face_count = data.shell_face_count;
    DISPATCH_START(shell_face_count)
    [data, curr, svd, param] __device__(unsigned i) mutable {
        Vec3u face = data.mesh.mesh.face[i];
        Mat3x3f x;
        x << curr[face[0]], curr[face[1]], curr[face[2]];
        const Mat3x2f F =
            utility::compute_deformation_grad(x, data.inv_rest2x2[i]);
        svd[i] = utility::svd3x2(F);
    } DISPATCH_END;
}

__device__ float get_wind_weight(float time) {
    float angle = 30.0f * time;
    float t = 0.25f;
    return t * (0.5f * (1.0f + sinf(angle))) + (1.0f - t);
}

} // namespace utility

template float utility::sum_array(Vec<float> array, unsigned size);
template unsigned utility::sum_integer_array(Vec<unsigned> array,
                                             unsigned size);
template unsigned utility::sum_integer_array(Vec<char> array, unsigned size);
template float utility::min_array(const float *array, unsigned size,
                                  float init_val);
template float utility::max_array(const float *array, unsigned size,
                                  float init_val);
template char utility::min_array(const char *array, unsigned size,
                                 char init_val);
template char utility::max_array(const char *array, unsigned size,
                                 char init_val);
template unsigned utility::min_array(const unsigned *array, unsigned size,
                                     unsigned init_val);
template unsigned utility::max_array(const unsigned *array, unsigned size,
                                     unsigned init_val);

#endif
