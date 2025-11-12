// File: eigsys_3.cpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "eig-hpp/eigsolve3x3.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace Eigen;
static double h = 1e-4;
inline double sqr(double x) { return x * x; }
using Vector9d = Vector<double, 9>;
using Matrix9d = Matrix<double, 9, 9>;

double energy_s(double a, double b, double c, const std::string &m) {
    if (m == "ARAP") {
        return sqr(a - 1) + sqr(b - 1) + sqr(c - 1);
    } else if (m == "SymDirichlet") {
        return sqr(a) + 1 / sqr(a) + sqr(b) + 1 / sqr(b) + sqr(c) + 1 / sqr(c);
    } else if (m == "MIPS") {
        return (sqr(a) + sqr(b) + sqr(c)) / (a * b * c);
    } else if (m == "Ogden") {
        double sum = 0.0;
        for (int k = 0; k < 5; ++k) {
            sum += std::pow(a, std::pow(0.5, k)) +
                   std::pow(b, std::pow(0.5, k)) +
                   std::pow(c, std::pow(0.5, k)) - 3;
        }
        return sum;
    } else if (m == "Yeoh") {
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) {
            sum += std::pow(sqr(a) + sqr(b) + sqr(c) - 3, k + 1);
        }
        return sum;
    } else {
        return 0.0;
    }
}

double energy_F(const Matrix3d &F, const std::string &m) {
    auto [_U, lambda, _Vt] = run_svd_3x3(F);
    return energy_s(lambda[0], lambda[1], lambda[2], m);
}

Vector3d approx_grad_s(double a, double b, double c, const std::string &m) {
    return (Vector3d() << (energy_s(a + h, b, c, m) -
                           energy_s(a - h, b, c, m)) /
                              (2 * h),
            (energy_s(a, b + h, c, m) - energy_s(a, b - h, c, m)) / (2 * h),
            (energy_s(a, b, c + h, m) - energy_s(a, b, c - h, m)) / (2 * h))
        .finished();
}

Matrix3d approx_hess_s(double a, double b, double c, const std::string &m) {
    Matrix3d H;
    H.col(0) = (approx_grad_s(a + h, b, c, m) - approx_grad_s(a - h, b, c, m)) /
               (2 * h);
    H.col(1) = (approx_grad_s(a, b + h, c, m) - approx_grad_s(a, b - h, c, m)) /
               (2 * h);
    H.col(2) = (approx_grad_s(a, b, c + h, m) - approx_grad_s(a, b, c - h, m)) /
               (2 * h);
    return H;
}

double approx_grad_F(const Matrix3d &F, const Matrix3d &dF,
                     const std::string &m) {
    return (energy_F(F + h * dF, m) - energy_F(F - h * dF, m)) / (2 * h);
}

Matrix9d approx_hess_F(const Matrix3d &F, const std::array<Matrix3d, 9> &dF,
                       const std::string &m) {
    int len = dF.size();
    Matrix9d H = Matrix9d::Zero(len, len);
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            H(i, j) = (approx_grad_F(F + h * dF[i], dF[j], m) -
                       approx_grad_F(F - h * dF[i], dF[j], m)) /
                      (2 * h);
        }
    }
    return H;
}

Matrix3d gen_dF(int i, int j) {
    Matrix3d dF = Matrix3d::Zero(3, 3);
    dF(i, j) = 1;
    return dF;
}

Vector9d mat2vec(const Matrix3d &A) {
    Vector9d x;
    int index = 0;
    for (int j = 0; j < A.cols(); ++j) {
        for (int i = 0; i < A.rows(); ++i) {
            x(index++) = A(i, j);
        }
    }
    return x;
}

int main() {
    std::srand(std::time(0));
    std::cout << std::scientific;
    std::cout << std::setprecision(3);

    Matrix3d F = Matrix3d::Random(3, 3);
    bool verbose = false;
    std::array<std::string, 5> models = {"ARAP", "SymDirichlet", "MIPS",
                                         "Ogden", "Yeoh"};

    std::array<Matrix3d, 9> dF;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            dF[i + 3 * j] = (gen_dF(i, j));
        }
    }

    auto [U, sigma, Vt] = run_svd_3x3(F);
    std::map<std::string, Vector2d> errors;
    for (const auto &m : models) {
        Vector9d g_F = Vector9d::Zero();
        for (int i = 0; i < 9; ++i) {
            g_F[i] = approx_grad_F(F, dF[i], m);
        }
        if (verbose) {
            std::cout << "---- (" << m << ") numerical gradient ----\n";
            std::cout << g_F << std::endl;
        }

        Matrix9d H_F = approx_hess_F(F, dF, m);
        if (verbose) {
            std::cout << "---- (" << m << ") numerical hessian ----\n";
            std::cout << H_F << std::endl;
        }

        Matrix3d H_s = approx_hess_s(sigma[0], sigma[1], sigma[2], m);
        Vector3d g_s = approx_grad_s(sigma[0], sigma[1], sigma[2], m);

        auto [S_s, U_s] = sym_eigsolve_3x3(H_s);
        Vector9d g_F_rebuilt = Vector9d::Zero();
        for (int i = 0; i < 9; ++i) {
            for (int k = 0; k < 3; ++k) {
                g_F_rebuilt[i] +=
                    g_s[k] * U.col(k).dot(dF[i] * Vt.row(k).transpose());
            }
        }
        if (verbose) {
            std::cout << "--- (" << m << ") analytical gradient ---\n";
            std::cout << g_F_rebuilt << std::endl;
        }

        double inv_sqrt2 = 1.0 / std::sqrt(2);
        std::array<Matrix3d, 9> Q = {
            {inv_sqrt2 * U *
                 (Matrix3d() << 0, 1, 0, -1, 0, 0, 0, 0, 0).finished() * Vt,
             inv_sqrt2 * U *
                 (Matrix3d() << 0, 0, 1, 0, 0, 0, -1, 0, 0).finished() * Vt,
             inv_sqrt2 * U *
                 (Matrix3d() << 0, 0, 0, 0, 0, 1, 0, -1, 0).finished() * Vt,
             inv_sqrt2 * U *
                 (Matrix3d() << 0, 1, 0, 1, 0, 0, 0, 0, 0).finished() * Vt,
             inv_sqrt2 * U *
                 (Matrix3d() << 0, 0, 1, 0, 0, 0, 1, 0, 0).finished() * Vt,
             inv_sqrt2 * U *
                 (Matrix3d() << 0, 0, 0, 0, 0, 1, 0, 1, 0).finished() * Vt,
             U * U_s.col(0).asDiagonal() * Vt, U * U_s.col(1).asDiagonal() * Vt,
             U * U_s.col(2).asDiagonal() * Vt}};

        Vector<double, 9> lmds(9);
        lmds << (g_s[0] + g_s[1]) / (sigma[0] + sigma[1]),
            (g_s[0] + g_s[2]) / (sigma[0] + sigma[2]),
            (g_s[1] + g_s[2]) / (sigma[1] + sigma[2]),
            (std::abs(sigma[0] - sigma[1]) > h
                 ? (g_s[0] - g_s[1]) / (sigma[0] - sigma[1])
                 : H_s(0, 0) - H_s(0, 1)),
            (std::abs(sigma[0] - sigma[2]) > h
                 ? (g_s[0] - g_s[2]) / (sigma[0] - sigma[2])
                 : H_s(0, 0) - H_s(0, 2)),
            (std::abs(sigma[1] - sigma[2]) > h
                 ? (g_s[1] - g_s[2]) / (sigma[1] - sigma[2])
                 : H_s(1, 1) - H_s(1, 2)),
            S_s[0], S_s[1], S_s[2];

        Matrix9d H_rebuilt = Matrix9d::Zero();
        for (int i = 0; i < 9; ++i) {
            if (lmds[i]) {
                Map<Vector9d> q(Q[i].data());
                H_rebuilt += lmds[i] * q * q.transpose();
            }
        }

        if (verbose) {
            std::cout << "--- (" << m << ") analytical hessian ---\n";
            std::cout << H_rebuilt << std::endl;
        }

        errors[m] = Vector2d((g_F - g_F_rebuilt).norm() / g_F.norm(),
                             (H_F - H_rebuilt).norm() / H_F.norm());
    }

    std::cout << "===== error summary =====\n";
    for (const auto &[name, err] : errors) {
        std::cout << name << ": " << std::endl << err << std::endl;
    }

    return 0;
}
