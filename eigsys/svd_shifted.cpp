// File: svd_shifted.cpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <vector>

using namespace Eigen;

template <typename T> using Vec2 = Vector<T, 2>;
template <typename T> using Mat2x2 = Matrix<T, 2, 2>;
template <typename T> using Mat3x2 = Matrix<T, 3, 2>;
template <typename T> using Mat3x3 = Matrix<T, 3, 3>;

template <typename T> Mat3x2<T> make_nearly_isotropic_F(const Vec2<T> &eps) {
    Mat3x3<T> A = Mat3x3<T>::Random().householderQr().householderQ();
    Mat3x2<T> U = A.template block<3, 2>(0, 0);
    Mat2x2<T> V = Matrix<T, 2, 2>::Random().householderQr().householderQ();
    Mat2x2<T> S = Mat2x2<T>::Zero();
    for (int i = 0; i < 2; i++) {
        S(i, i) = T(1.0) + eps[i];
    }
    return U * S * V.transpose();
}

template <typename T> Vec2<T> compute_eigvals(const Mat2x2<T> A) {
    SelfAdjointEigenSolver<Mat2x2<T>> eigensolver(A, 0);
    Vec2<T> result = eigensolver.eigenvalues();
    return result;
}

template <typename T>
Vec2<T> singular_vals_minus_one_naive(const Mat3x2<T> &F) {
    Mat2x2<T> A(F.transpose() * F);
    Vec2<T> lmd = compute_eigvals(A);
    for (int i = 0; i < 2; ++i) {
        lmd[i] = sqrt(lmd[i]) - T(1.0);
    }
    std::sort(lmd.data(), lmd.data() + lmd.size());
    return lmd;
}

Vec2<float> singular_vals_minus_one_approx(const Mat3x2<float> &F) {
    Mat2x2<float> A(F.transpose() * F - Mat2x2<float>::Identity());
    Vec2<float> lmd = compute_eigvals(A);
    for (int i = 0; i < 2; ++i) {
        /*
        import sympy as sp
        lambda_ = sp.symbols('\lambda')
        f = sp.sqrt(1 + lambda_)
        display(sp.series(f, lambda_, 0, 11))
        */
        float lmd_1 = lmd[i];
        float lmd_2 = lmd_1 * lmd_1;
        float lmd_3 = lmd_2 * lmd_1;
        float lmd_4 = lmd_2 * lmd_2;
        float lmd_5 = lmd_3 * lmd_2;
        float lmd_6 = lmd_3 * lmd_3;
        float lmd_7 = lmd_4 * lmd_3;
        float lmd_8 = lmd_4 * lmd_4;
        float lmd_9 = lmd_5 * lmd_4;
        float lmd_10 = lmd_5 * lmd_5;
        float terms[] = {0.5f * lmd_1,                 //
                         -(1.0f / 8.0f) * lmd_2,       //
                         +(1.0f / 16.0f) * lmd_3,      //
                         -(5.0f / 128.0f) * lmd_4,     //
                         +(7.0f / 256.0f) * lmd_5,     //
                         -(21.0f / 1024.0f) * lmd_6,   //
                         +(33.0f / 2048.0f) * lmd_7,   //
                         -(429.0f / 32768.0f) * lmd_8, //
                         +(715.0f / 65536.0f) * lmd_9, //
                         -(2431.0f / 262144.0f) * lmd_10};
        float sum(0.0f);
        for (int k = sizeof(terms) / sizeof(float) - 1; k >= 0; --k) {
            sum += terms[k];
        }
        // sum = sqrt(lmd[i] + 1.0f) - 1.0f;
        lmd[i] = sum;
    }
    std::sort(lmd.data(), lmd.data() + lmd.size());
    return lmd;
}

double run(float _eps) {

    Vec2<float> eps = _eps * Vec2<float>::Random();
    std::sort(eps.data(), eps.data() + eps.size());

    Mat3x2<float> F = make_nearly_isotropic_F<float>(eps);
    Vec2<float> lmd_naive = singular_vals_minus_one_naive(F);
    Vec2<float> lmd_approx = singular_vals_minus_one_approx(F);
    float error_naive = 0.0f;
    float error_approx = 0.0f;
    for (int i = 0; i < 2; ++i) {
        error_naive =
            std::max(error_naive, std::abs((eps[i] - lmd_naive[i]) / eps[i]));
        error_approx =
            std::max(error_approx, std::abs((eps[i] - lmd_approx[i]) / eps[i]));
    }
    return error_approx / error_naive;
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(0)));
    float eps[] = {0.001, 0.01, 0.025, 0.05, 0.1};
    for (int k = 0; k < sizeof(eps) / sizeof(float); k++) {
        const int n_iter = 1000;
        double sum = 0.0;
        for (int i = 0; i < n_iter; ++i) {
            sum += run(eps[k]);
        }
        double average_result = sum / n_iter;
        printf("Average improvement (eps=%f): %f\n", eps[k], average_result);
    }

    return 0;
}
