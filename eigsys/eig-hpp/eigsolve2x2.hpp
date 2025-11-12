// File: eigsolve2x2.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef EIGEN_SOLVER_2X2_H
#define EIGEN_SOLVER_2X2_H

#include <Eigen/Dense>
#include <array>
#include <cmath>

#ifndef _real_
#define _real_ double
#endif

#ifndef __device__
#define __device__
#endif

using Mat2r = Eigen::Matrix<_real_, 2, 2>;
using Mat3x2r = Eigen::Matrix<_real_, 3, 2>;
using Vec2r = Eigen::Vector<_real_, 2>;

static __device__ Vec2r eigvalues(const Mat2r &A) {
    _real_ a00(A(0, 0)), a01(A(0, 1)), a11(A(1, 1));
    _real_ tmp = a00 - a11;
    _real_ D = 0.5 * std::sqrt(tmp * tmp + 4.0 * a01 * a01);
    _real_ mid = 0.5 * (a00 + a11);
    return {mid - D, mid + D};
}

static __device__ Vec2r rot90(const Vec2r &x) { return Vec2r(x[1], -x[0]); }

static __device__ Vec2r find_ortho2(const Mat2r &A, const Vec2r &x,
                                    _real_ sqr_eps) {
    Vec2r u(rot90(A.col(0))), v(rot90(A.col(1)));
    if (u.squaredNorm() > sqr_eps) {
        return u.normalized();
    } else if (v.squaredNorm() > sqr_eps) {
        return v.normalized();
    } else {
        return rot90(x);
    }
}

static __device__ Mat2r eigvectors2(const Mat2r &A, const Vec2r &lmd) {
    _real_ eps = 1e-8;
    _real_ sqr_eps = eps * eps;
    Vec2r u =
        find_ortho2(A - lmd[0] * Mat2r::Identity(), Vec2r(0.0, 1.0), sqr_eps);
    Vec2r v = find_ortho2(A - lmd[1] * Mat2r::Identity(), -u, sqr_eps);
    Mat2r result;
    result << u, v;
    return result;
}

struct eig_tuple_2x2 {
    Vec2r lambda;
    Mat2r eigvecs;
};

static __device__ eig_tuple_2x2 sym_eigsolve_2x2(const Mat2r &A) {
#ifdef USE_EIGEN_SYMM_EIGSOLVE
    Eigen::SelfAdjointEigenSolver<Mat2r> eigensolver;
    eigensolver.computeDirect(A);
    return {eigensolver.eigenvalues(), eigensolver.eigenvectors()};
#else
    _real_ scale = A.norm();
    Mat2r B = A / scale;
    Vec2r lmd = eigvalues(B);
    Mat2r eigvecs = eigvectors2(B, lmd);
    return {scale * lmd, eigvecs};
#endif
}

struct svd_tuple_3x2 {
    Mat3x2r U;
    Vec2r lambda;
    Mat2r Vt;
};

static __device__ svd_tuple_3x2 run_svd_3x2(const Mat3x2r &F) {
    Mat2r FtF = F.transpose() * F;
    auto [lambda, V] = sym_eigsolve_2x2(FtF);
    for (int i = 0; i < 2; ++i) {
        lambda[i] = std::sqrt(std::fmax(0.0, lambda[i]));
    }
    Mat3x2r U = F * V;
    for (int i = 0; i < U.cols(); ++i) {
        U.col(i).normalize();
    }
    return {U, lambda, V.transpose()};
}

static __device__ Vec2r singular_vals_minus_one(const Mat3x2r &F) {
    Mat2r A(F.transpose() * F);
#ifdef USE_EIGEN_SYMM_EIGSOLVE
    Eigen::SelfAdjointEigenSolver<Mat2r> eigensolver(A, 0);
    Vec2r lmd = eigensolver.eigenvalues();
#else
    Vec2r lmd = eigvalues(A);
#endif
    for (int i = 0; i < 2; ++i) {
        lmd[i] = sqrt(lmd[i]) - _real_(1.0);
    }
    return lmd;
}

#endif // EIGEN_SOLVER_2X2_H
