// File: stvk.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef STVK_HPP
#define STVK_HPP

#include "../../common.hpp"
#include "../../data.hpp"
#include "detsqr.hpp"

namespace StVK {

__device__ float sqr(float x) { return x * x; }

__device__ float energy(const Vec2f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    return mu * (sqr(a0) * sqr(a0) + sqr(a1) * sqr(a1)) / 4.0f +
           detsqr::energy(a, lmd);
}

__device__ float energy(const Vec3f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    return mu * (sqr(a0) * sqr(a0) + sqr(a1) * sqr(a1) + sqr(a2) * sqr(a2)) /
               4.0f +
           detsqr::energy(a, lmd);
}

__device__ Vec2f gradient(const Vec2f &a) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    return Vec2f(a0 * a0 * a0, a1 * a1 * a1);
}

__device__ Vec3f gradient(const Vec3f &a) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    return Vec3f(a0 * a0 * a0, a1 * a1 * a1, a2 * a2 * a2);
}

__device__ Mat2x2f hessian(const Vec2f &a) {
    Mat2x2f result = Mat2x2f::Zero();
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    result(0, 0) = 3.0f * a0 * a0;
    result(1, 1) = 3.0f * a1 * a1;
    return result;
}

__device__ Mat3x3f hessian(const Vec3f &a) {
    Mat3x3f result = Mat3x3f::Zero();
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    result(0, 0) = 3.0f * a0 * a0;
    result(1, 1) = 3.0f * a1 * a1;
    result(2, 2) = 3.0f * a2 * a2;
    return result;
}

__device__ DiffTable2 make_diff_table2(const Vec2f &a, float mu, float lmd) {
    DiffTable2 table;
    DiffTable2 detsqr_table = detsqr::make_diff_table2(a, lmd);
    table.deda = mu * gradient(a) + detsqr_table.deda;
    table.d2ed2a = mu * hessian(a) + detsqr_table.d2ed2a;
    return table;
}

__device__ DiffTable3 make_diff_table3(const Vec3f &a, float mu, float lmd) {
    DiffTable3 table;
    DiffTable3 detsqr_table = detsqr::make_diff_table3(a, lmd);
    table.deda = mu * gradient(a) + detsqr_table.deda;
    table.d2ed2a = mu * hessian(a) + detsqr_table.d2ed2a;
    return table;
}

} // namespace StVK

#endif