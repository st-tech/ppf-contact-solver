// File: arap.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ARAP_HPP
#define ARAP_HPP

#include "../../common.hpp"
#include "../../data.hpp"
#include "detsqr.hpp"

namespace ARAP {

__device__ float energy(const Vec2f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    return 0.5f * mu * (a0 * a0 + a1 * a1) + detsqr::energy(a, lmd);
}

__device__ float energy(const Vec3f &a, float mu, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    return 0.5f * mu * (a0 * a0 + a1 * a1 + a2 * a2) + detsqr::energy(a, lmd);
}

__device__ DiffTable2 make_diff_table2(const Vec2f &a, float mu, float lmd) {
    DiffTable2 table;
    DiffTable2 detsqr_table = detsqr::make_diff_table2(a, lmd);
    table.deda = mu * (a - Vec2f::Ones()) + detsqr_table.deda;
    table.d2ed2a = mu * (Mat2x2f::Identity()) + detsqr_table.d2ed2a;
    return table;
}

__device__ DiffTable3 make_diff_table3(const Vec3f &a, float mu, float lmd) {
    DiffTable3 table;
    DiffTable3 detsqr_table = detsqr::make_diff_table3(a, lmd);
    table.deda = mu * (a - Vec3f::Ones()) + detsqr_table.deda;
    table.d2ed2a = mu * (Mat3x3f::Identity()) + detsqr_table.d2ed2a;
    return table;
}

} // namespace ARAP

#endif