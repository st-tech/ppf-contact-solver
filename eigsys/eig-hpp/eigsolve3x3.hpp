// File: eigsolve3x3.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef EIGEN_SOLVER_3X3_H
#define EIGEN_SOLVER_3X3_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#ifndef _real_
#define _real_ double
#endif

#ifndef __device__
#define __device__
#endif

using Mat3r = Eigen::Matrix<_real_, 3, 3>;
using Vec3r = Eigen::Vector<_real_, 3>;

// https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
static __device__ Vec3r eigvalues3(const Mat3r &A) {
    _real_ p1 = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    _real_ q = A.trace() / 3.0;
    _real_ p2 = (A(0, 0) - q) * (A(0, 0) - q) + (A(1, 1) - q) * (A(1, 1) - q) +
                (A(2, 2) - q) * (A(2, 2) - q) + 2.0 * p1;
    _real_ p = sqrt(p2 / 6.0);
    if (fabs(p) < 1e-8) {
        return Vec3r::Zero();
    } else {
        Mat3r B = (1.0 / p) * (A - q * Mat3r::Identity());
        _real_ r = B.determinant() / 2.0;
        _real_ phi;
        if (r <= -1.0) {
            phi = M_PI / 3.0;
        } else if (r >= 1.0) {
            phi = 0.0;
        } else {
            phi = acos(r) / 3.0;
        }
        _real_ eig1 = q + 2.0 * p * cos(phi);
        _real_ eig3 = q + 2.0 * p * cos(phi + 2.0 * M_PI / 3.0);
        _real_ eig2 = 3.0 * q - eig1 - eig3;
        return Vec3r(eig1, eig2, eig3);
    }
}

static __device__ Vec3r pick_largest(const Vec3r &a, const Vec3r &b,
                                     const Vec3r &c) {
    _real_ a_norm = a.squaredNorm();
    _real_ b_norm = b.squaredNorm();
    _real_ c_norm = c.squaredNorm();
    if (a_norm > b_norm) {
        if (a_norm > c_norm) {
            return a;
        } else {
            return c;
        }
    } else {
        if (b_norm > c_norm) {
            return b;
        } else {
            return c;
        }
    }
}

struct vec3r_vec3r {
    Vec3r v1, v2;
};
static __device__ vec3r_vec3r find_ortho3x3(const Mat3r &A) {
    _real_ eps = 1e-8;
    Vec3r u(A.col(0)), v(A.col(1)), w(A.col(2));
    Vec3r uv(u.cross(v)), vw(v.cross(w)), wu(w.cross(u));
    Vec3r q = pick_largest(uv, vw, wu);
    if (q.squaredNorm() < eps) {
        Vec3r p = pick_largest(u, v, w);
        Vec3r x = p.cross(Vec3r(1.0, 0.0, 0.0));
        if (x.squaredNorm() < eps) {
            x = p.cross(Vec3r(0.0, 1.0, 0.0));
        }
        Vec3r y = p.cross(x);
        return {x.normalized(), y.normalized()};
    } else {
        return {q.normalized(), Vec3r::Zero()};
    }
}

static __device__ Mat3r eigvectors3x3(const Mat3r &A, const Vec3r &lmd) {
    vec3r_vec3r uv = find_ortho3x3(A - lmd[0] * Mat3r::Identity());
    if (uv.v2.squaredNorm() == 0.0) {
        vec3r_vec3r tmp = find_ortho3x3(A - lmd[1] * Mat3r::Identity());
        uv.v2 = tmp.v1;
    }
    Vec3r w = uv.v1.cross(uv.v2);
    Mat3r result;
    result << uv.v1, uv.v2, w;
    return result;
}

struct eig_tuple_3x3 {
    Vec3r lambda;
    Mat3r eigvecs;
};

static __device__ eig_tuple_3x3 sym_eigsolve_3x3(const Mat3r &A) {
#ifdef USE_EIGEN_SYMM_EIGSOLVE
    Eigen::SelfAdjointEigenSolver<Mat3r> eigensolver;
    eigensolver.computeDirect(A);
    return {eigensolver.eigenvalues(), eigensolver.eigenvectors()};
#else
    _real_ scale = A.norm();
    Mat3r B = A / scale;
    Vec3r lmd = eigvalues3(B);
    if (lmd.squaredNorm()) {
        Mat3r eigvecs = eigvectors3x3(B, lmd);
        return {scale * lmd, eigvecs};
    } else {
        Mat3r eigvecs = Mat3r::Identity();
        _real_ val =
            (B.col(0).norm() + B.col(1).norm() + B.col(2).norm()) / 3.0;
        lmd << val, val, val;
        return {scale * lmd, eigvecs};
    }
#endif
}

struct svd_tuple_3x3 {
    Mat3r U;
    Vec3r lambda;
    Mat3r Vt;
};

static __device__ svd_tuple_3x3 run_svd_3x3(const Mat3r &F) {
    Mat3r FtF = F.transpose() * F;
    auto [lambda, V] = sym_eigsolve_3x3(FtF);
    for (int i = 0; i < 3; ++i) {
        lambda[i] = std::sqrt(std::fmax(0.0, lambda[i]));
    }
    Mat3r U = F * V;
    for (int i = 0; i < U.cols(); ++i) {
        U.col(i).normalize();
    }
    return {U, lambda, V.transpose()};
}

#endif // EIGEN_SOLVER_H
