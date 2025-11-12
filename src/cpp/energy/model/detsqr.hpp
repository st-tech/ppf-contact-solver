// File: detsqr.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DETSQR_HPP
#define DETSQR_HPP

#include "../../common.hpp"
#include "../../data.hpp"

namespace detsqr {

__device__ float sqr(float x) { return x * x; }

__device__ float energy(const Vec2f &a, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    return lmd * 0.5f * sqr(a0 * a1 - 1.0f);
}

__device__ float energy(const Vec3f &a, float lmd) {
    float a0 = a[0] - 1.0f;
    float a1 = a[1] - 1.0f;
    float a2 = a[2] - 1.0f;
    return lmd * 0.5f * sqr(a0 * a1 * a2 - 1.0f);
}

__device__ DiffTable2 make_diff_table2(const Vec2f &a, float lmd) {
    DiffTable2 table;
    float J = a[0] * a[1];
    table.deda[0] = lmd * a[1] * (J - 1.0f);
    table.deda[1] = lmd * a[0] * (J - 1.0f);
    table.d2ed2a(0, 0) = lmd * sqr(a[1]);
    table.d2ed2a(1, 1) = lmd * sqr(a[0]);
    table.d2ed2a(0, 1) = lmd * (2.0f * a[0] * a[1] - 1.0f);
    table.d2ed2a(1, 0) = table.d2ed2a(0, 1);
    return table;
}

__device__ DiffTable3 make_diff_table3(const Vec3f &a, float lmd) {
    DiffTable3 table;
    float J = a[0] * a[1] * a[2];
    table.deda[0] = lmd * a[1] * a[2] * (J - 1.0f);
    table.deda[1] = lmd * a[0] * a[2] * (J - 1.0f);
    table.deda[2] = lmd * a[0] * a[1] * (J - 1.0f);
    table.d2ed2a(0, 0) = lmd * sqr(a[1] * a[2]);
    table.d2ed2a(1, 1) = lmd * sqr(a[0] * a[2]);
    table.d2ed2a(2, 2) = lmd * sqr(a[0] * a[1]);
    table.d2ed2a(0, 1) = lmd * a[2] * (2.0f * J - 1.0f);
    table.d2ed2a(0, 2) = lmd * a[1] * (2.0f * J - 1.0f);
    table.d2ed2a(1, 2) = lmd * a[0] * (2.0f * J - 1.0f);
    table.d2ed2a(1, 0) = table.d2ed2a(0, 1);
    table.d2ed2a(2, 0) = table.d2ed2a(0, 2);
    table.d2ed2a(2, 1) = table.d2ed2a(1, 2);
    return table;
}

} // namespace detsqr

#endif