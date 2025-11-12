// File: utility.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "dispatcher.hpp"
#include "utility.hpp"
#include <limits>

#include <Eigen/Eigenvalues>

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
    Eigen::SelfAdjointEigenSolver<Mat2x2f> eigensolver;
    eigensolver.computeDirect(matrix);
    eigenvalues = eigensolver.eigenvalues();
    eigenvectors = eigensolver.eigenvectors();
}

__device__ void solve_symm_eigen3x3(const Mat3x3f &matrix, Vec3f &eigenvalues,
                                    Mat3x3f &eigenvectors) {
    Eigen::SelfAdjointEigenSolver<Mat3x3f> eigensolver;
    eigensolver.computeDirect(matrix);
    eigenvalues = eigensolver.eigenvalues();
    eigenvectors = eigensolver.eigenvectors();
}

__device__ Vec2f singular_vals_minus_one(const Mat3x2f &F) {
    Mat2x2f A = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<Mat2x2f> eigensolver(A, 0);
    Vec2f lmd = eigensolver.eigenvalues();
    for (int i = 0; i < 2; ++i) {
        lmd[i] = sqrtf(lmd[i]) - 1.0f;
    }
    return lmd;
}

__device__ Svd3x2 svd3x2_shifted(const Mat3x2f &F) {
    Mat2x2f A = F.transpose() * F - Mat2x2f::Identity();
    Eigen::SelfAdjointEigenSolver<Mat2x2f> eigensolver;
    eigensolver.computeDirect(A);
    Mat2x2f V = eigensolver.eigenvectors();
    Mat3x2f U = F * V;
    for (unsigned i = 0; i < U.cols(); i++) {
        U.col(i).normalize();
    }
    return {U, singular_vals_minus_one(F), V.transpose()};
}

__device__ Svd3x2 svd3x2(const Mat3x2f &F) {
    Eigen::SelfAdjointEigenSolver<Mat2x2f> eigensolver;
    eigensolver.computeDirect(F.transpose() * F);
    Vec2f sigma = eigensolver.eigenvalues();
    Mat2x2f V = eigensolver.eigenvectors();
    for (unsigned i = 0; i < 2; ++i) {
        sigma[i] = sqrtf(fmax(0.0f, sigma[i]));
    }
    Mat3x2f U = F * V;
    for (unsigned i = 0; i < U.cols(); i++) {
        U.col(i).normalize();
    }
    return {U, sigma, V.transpose()};
}

__device__ Svd3x3 svd3x3(const Mat3x3f &F) {
    Eigen::SelfAdjointEigenSolver<Mat3x3f> eigensolver;
    eigensolver.computeDirect(F.transpose() * F);
    Vec3f sigma = eigensolver.eigenvalues();
    Mat3x3f V = eigensolver.eigenvectors();
    for (unsigned i = 0; i < 3; ++i) {
        sigma[i] = sqrtf(fmax(0.0f, sigma[i]));
    }
    Mat3x3f U = F * V;
    for (unsigned i = 0; i < U.cols(); i++) {
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

__device__ Mat3x3f convert_force(const Mat3x2f &dedF,
                                 const Mat2x2f &inv_rest2x2) {
    Vec2f g0 = -inv_rest2x2.row(0) - inv_rest2x2.row(1);
    Vec2f g1 = inv_rest2x2.row(0);
    Vec2f g2 = inv_rest2x2.row(1);

    Mat3x3f result;
    for (unsigned dim = 0; dim < 3; ++dim) {
        result(dim, 0) = g0.dot(dedF.row(dim));
        result(dim, 1) = g1.dot(dedF.row(dim));
        result(dim, 2) = g2.dot(dedF.row(dim));
    }
    return result;
}

__device__ Mat3x4f convert_force(const Mat3x3f &dedF,
                                 const Mat3x3f &inv_rest3x3) {
    Vec3f g0 = -inv_rest3x3.row(0) - inv_rest3x3.row(1) - inv_rest3x3.row(2);
    Vec3f g1 = inv_rest3x3.row(0);
    Vec3f g2 = inv_rest3x3.row(1);
    Vec3f g3 = inv_rest3x3.row(2);

    Mat3x4f result;
    for (unsigned dim = 0; dim < 3; ++dim) {
        result(dim, 0) = g0.dot(dedF.row(dim));
        result(dim, 1) = g1.dot(dedF.row(dim));
        result(dim, 2) = g2.dot(dedF.row(dim));
        result(dim, 3) = g3.dot(dedF.row(dim));
    }
    return result;
}

__device__ Mat9x9f convert_hessian(const Mat6x6f &d2ed2f,
                                   const Mat2x2f &inv_rest2x2) {
    Vec2f g0 = -inv_rest2x2.row(0) - inv_rest2x2.row(1);
    Vec2f g1 = inv_rest2x2.row(0);
    Vec2f g2 = inv_rest2x2.row(1);
    Mat6x9f dfdx = Mat6x9f::Zero();
    for (unsigned j = 0; j < 9; ++j) {
        unsigned col = j / 3;
        unsigned row = j % 3;
        Vec2f g_row = (col == 0) ? g0 : (col == 1) ? g1 : g2;
        dfdx(0 * 3 + row, j) = g_row[0];
        dfdx(1 * 3 + row, j) = g_row[1];
    }

    Mat9x9f result = Mat9x9f::Zero();
    for (unsigned i = 0; i < 6; ++i) {
        for (unsigned j = 0; j < 6; ++j) {
            if (fabs(d2ed2f(i, j)) > EPSILON) {
                result += d2ed2f(i, j) * dfdx.row(i).transpose() * dfdx.row(j);
            }
        }
    }
    return result;
}

__device__ Mat12x12f convert_hessian(const Mat9x9f &d2ed2f,
                                     const Mat3x3f &inv_rest3x3) {
    Vec3f g0 = -inv_rest3x3.row(0) - inv_rest3x3.row(1) - inv_rest3x3.row(2);
    Vec3f g1 = inv_rest3x3.row(0);
    Vec3f g2 = inv_rest3x3.row(1);
    Vec3f g3 = inv_rest3x3.row(2);

    Mat9x12f dfdx = Mat9x12f::Zero();
    for (unsigned j = 0; j < 12; ++j) {
        unsigned col = j / 3;
        unsigned row = j % 3;
        Vec3f g_row = (col == 0) ? g0 : (col == 1) ? g1 : (col == 2) ? g2 : g3;
        dfdx(0 * 3 + row, j) = g_row[0];
        dfdx(1 * 3 + row, j) = g_row[1];
        dfdx(2 * 3 + row, j) = g_row[2];
    }

    Mat12x12f result = Mat12x12f::Zero();
    for (unsigned i = 0; i < 9; ++i) {
        for (unsigned j = 0; j < 9; ++j) {
            if (fabs(d2ed2f(i, j)) > EPSILON) {
                result += d2ed2f(i, j) * dfdx.row(i).transpose() * dfdx.row(j);
            }
        }
    }
    return result;
}

__device__ Mat3x2f compute_deformation_grad(const Mat3x3f &x,
                                            const Mat2x2f &inv_rest2x2) {
    Mat3x2f dx;
    dx.col(0) = x.col(1) - x.col(0);
    dx.col(1) = x.col(2) - x.col(0);
    return dx * inv_rest2x2;
}

__device__ Mat3x3f compute_deformation_grad(const Mat3x4f &x,
                                            const Mat3x3f &inv_rest3x3) {
    Mat3x3f dx;
    dx.col(0) = x.col(1) - x.col(0);
    dx.col(1) = x.col(2) - x.col(0);
    dx.col(2) = x.col(3) - x.col(0);
    return dx * inv_rest3x3;
}

__device__ float compute_face_area(const Mat3x3f &vertex) {
    const Vec3f v0 = vertex.col(0);
    const Vec3f v1 = vertex.col(1);
    const Vec3f v2 = vertex.col(2);
    return 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

void compute_svd(DataSet data, Vec<Vec3f> curr, Vec<Svd3x2> svd,
                 ParamSet param) {
    unsigned shell_face_count = data.shell_face_count;
    auto mesh_face = data.mesh.mesh.face.data;
    auto curr_data = curr.data;
    auto svd_data = svd.data;
    auto inv_rest2x2 = data.inv_rest2x2.data;
    DISPATCH_START(shell_face_count)
    [mesh_face, curr_data, svd_data,
     inv_rest2x2] __device__(unsigned i) mutable {
        Vec3u face = mesh_face[i];
        Mat3x3f x;
        x << curr_data[face[0]], curr_data[face[1]], curr_data[face[2]];
        const Mat3x2f F = utility::compute_deformation_grad(x, inv_rest2x2[i]);
        svd_data[i] = utility::svd3x2(F);
    } DISPATCH_END;
}

__device__ float get_wind_weight(float time) {
    float angle = 30.0f * time;
    float t = 0.25f;
    return t * (0.5f * (1.0f + sinf(angle))) + (1.0f - t);
}

} // namespace utility
