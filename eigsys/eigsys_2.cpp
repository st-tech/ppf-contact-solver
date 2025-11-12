// File: eigsys_2.cpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "eig-hpp/eigsolve2x2.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Eigen;
static double h = 1e-4;

double energy_s(double a, double b) {
    return std::log(a * a) + std::log(b * b) + a * b + a * a * b * b;
}

double energy_F(const Matrix<double, 3, 2> &F) {
    auto [_U, lambda, _Vt] = run_svd_3x2(F);
    return energy_s(lambda(0), lambda(1));
}

Vector2d approx_grad_s(double a, double b) {
    return Vector2d((energy_s(a + h, b) - energy_s(a - h, b)) / (2 * h),
                    (energy_s(a, b + h) - energy_s(a, b - h)) / (2 * h));
}

Matrix2d approx_hess_s(double a, double b) {
    Matrix2d H;
    H.col(0) = (approx_grad_s(a + h, b) - approx_grad_s(a - h, b)) / (2 * h);
    H.col(1) = (approx_grad_s(a, b + h) - approx_grad_s(a, b - h)) / (2 * h);
    return H;
}

double approx_grad_F(const Matrix<double, 3, 2> &F,
                     const Matrix<double, 3, 2> &dF) {
    return (energy_F(F + h * dF) - energy_F(F - h * dF)) / (2 * h);
}

Matrix<double, 6, 6>
approx_hess_F(const Matrix<double, 3, 2> &F,
              const std::array<Matrix<double, 3, 2>, 6> &dF) {
    int len = dF.size();
    Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero(len, len);
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            H(i, j) = (approx_grad_F(F + h * dF[i], dF[j]) -
                       approx_grad_F(F - h * dF[i], dF[j])) /
                      (2 * h);
        }
    }
    return H;
}

Matrix<double, 3, 2> gen_dF(int i, int j) {
    Matrix<double, 3, 2> dF = Matrix<double, 3, 2>::Zero(3, 2);
    dF(i, j) = 1.0;
    return dF;
}

Vector<double, 6> mat2vec(const Matrix<double, 3, 2> &A) {
    VectorXd x(A.size());
    int index = 0;
    for (int j = 0; j < A.cols(); ++j) {
        for (int i = 0; i < A.rows(); ++i) {
            x(index++) = A(i, j);
        }
    }
    return x;
}

Matrix3d expand_U(const Matrix<double, 3, 2> &U) {
    Vector3d cross = U.col(0).cross(U.col(1));
    Matrix3d result;
    result << U.col(0), U.col(1), cross;
    return result;
}

int main() {
    std::srand(std::time(0));
    std::cout << std::scientific;
    std::cout << std::setprecision(3);

    Matrix<double, 3, 2> F = Matrix<double, 3, 2>::Random(3, 2);
    std::cout << "---- F ----" << std::endl;
    std::cout << F << std::endl;

    std::array<Matrix<double, 3, 2>, 6> dF;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 3; ++i) {
            dF[i + 3 * j] = gen_dF(i, j);
        }
    }

    Vector<double, 6> g_F = Vector<double, 6>::Zero();
    for (int i = 0; i < 6; ++i) {
        g_F[i] = approx_grad_F(F, dF[i]);
    }
    std::cout << "---- numerical gradient ----" << std::endl;
    std::cout << g_F << std::endl;

    Matrix<double, 6, 6> H_F = approx_hess_F(F, dF);
    std::cout << "---- numerical hessian ----" << std::endl;
    std::cout << H_F << std::endl;

    auto [_U, lambda, Vt] = run_svd_3x2(F);
    Matrix2d H_s = approx_hess_s(lambda(0), lambda(1));
    Vector2d g_s = approx_grad_s(lambda(0), lambda(1));
    auto [S_s, U_s] = sym_eigsolve_2x2(H_s);
    Matrix3d U = expand_U(_U);

    printf("--- analytical gradient ---\n");
    Vector<double, 6> g_F_rebuilt = Vector<double, 6>::Zero();
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 2; ++k) {
            g_F_rebuilt[i] +=
                g_s[k] * U.col(k).dot(dF[i] * Vt.row(k).transpose());
        }
    }
    std::cout << g_F_rebuilt << std::endl;

    // clang-format off
    std::array<Matrix<double, 3, 2>, 6> Q = {
        U * (MatrixXd(3, 2) << 0, 1, -1, 0, 0, 0).finished() * Vt / std::sqrt(2),
        U * (MatrixXd(3, 2) << 0, 1, 1, 0, 0, 0).finished() * Vt / std::sqrt(2),
        U * (MatrixXd(3, 2) << 0, 0, 0, 0, 1, 0).finished() * Vt,
        U * (MatrixXd(3, 2) << 0, 0, 0, 0, 0, 1).finished() * Vt,
        U * (MatrixXd(3, 2) << U_s(0, 0), 0, 0, U_s(1, 0), 0, 0).finished() * Vt,
        U * (MatrixXd(3, 2) << U_s(0, 1), 0, 0, U_s(1, 1), 0, 0).finished() * Vt};
    // clang-format on

    Vector<double, 6> lmds(6);
    lmds << (g_s(0) + g_s(1)) / (lambda(0) + lambda(1)),
        (std::abs(lambda(0) - lambda(1)) > h
             ? (g_s(0) - g_s(1)) / (lambda(0) - lambda(1))
             : H_s(0, 0) - H_s(0, 1)),
        g_s(0) / lambda(0), g_s(1) / lambda(1), S_s(0), S_s(1);

    std::cout << "--- analytical hessian ---" << std::endl;
    Matrix<double, 6, 6> H_rebuilt = Matrix<double, 6, 6>::Zero();
    for (int i = 0; i < 6; ++i) {
        if (lmds[i]) {
            Map<Vector<double, 6>> q(Q[i].data());
            H_rebuilt += lmds[i] * q * q.transpose();
        }
    }
    std::cout << H_rebuilt << std::endl;
    std::cout << "--- error ---" << std::endl;
    std::cout << (g_F - g_F_rebuilt).norm() / g_F.norm() << std::endl;
    std::cout << (H_F - H_rebuilt).norm() / H_F.norm() << std::endl;

    return 0;
}
