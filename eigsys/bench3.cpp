// File: bench3.cpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "eig-hpp/eigsolve3x3.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

using namespace Eigen;

void eigen_decompose(const std::vector<Matrix3d> &F) {
    for (auto &f : F) {
        SelfAdjointEigenSolver<Matrix3d> eigensolver;
        eigensolver.computeDirect(f);
    }
}

void our_compute(const std::vector<Matrix3d> &F) {
    for (auto &f : F) {
        auto [_U, lambda, _Vt] = run_svd_3x3(f);
    }
}

template <typename Func>
double measure_execution_time(Func func, const std::vector<Matrix3d> &F) {
    auto start = std::chrono::high_resolution_clock::now();
    func(F);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

int main() {
    unsigned mat_count = 10000000;
    std::vector<Matrix3d> F(mat_count);
    for (unsigned i = 0; i < mat_count; ++i) {
        Matrix3d a = Matrix3d::Random();
        F[i] = a + a.transpose();
    }
    double time1 = measure_execution_time(eigen_decompose, F);
    std::cout << "Eigen: " << time1 << " seconds." << std::endl;
    double time2 = measure_execution_time(our_compute, F);
    std::cout << "Ours: " << time2 << " seconds." << std::endl;
    std::cout << "Speedup factor: " << time1 / time2 << std::endl;
    return 0;
}
