// File: snhk.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef SNHK_HPP
#define SNHK_HPP

#include "../../common.hpp"
#include "../../data.hpp"

namespace SNHk {

__device__ float energy(const Vec3f &S, float mu, float lmd) {
    float a = S[0];
    float b = S[1];
    float c = S[2];
    float I2 = a * a + b * b + c * c;
    float I3 = a * b * c;
    return 0.5f * mu * (I2 - 3.0f) - mu * (I3 - 1.0f) +
           0.5f * lmd * (I3 - 1.0f) * (I3 - 1.0f);
}

__device__ DiffTable3 make_diff_table3(const Vec3f &S, float mu, float lmd) {
    DiffTable3 table;
    float a = S[0];
    float b = S[1];
    float c = S[2];
    float I3 = a * b * c;
    table.deda[0] = lmd * b * c * (I3 - 1.0f) + mu * (a - b * c);
    table.deda[1] = lmd * a * c * (I3 - 1.0f) + mu * (b - a * c);
    table.deda[2] = lmd * a * b * (I3 - 1.0f) + mu * (c - a * b);
    table.d2ed2a(0, 0) = lmd * b * b * c * c + mu;
    table.d2ed2a(1, 1) = lmd * a * a * c * c + mu;
    table.d2ed2a(2, 2) = lmd * a * a * b * b + mu;
    table.d2ed2a(0, 1) = lmd * I3 * c + lmd * c * (I3 - 1.0f) - mu * c;
    table.d2ed2a(0, 2) = lmd * I3 * b + lmd * b * (I3 - 1.0f) - mu * b;
    table.d2ed2a(1, 2) = lmd * I3 * a + lmd * a * (I3 - 1.0f) - mu * a;
    table.d2ed2a(1, 0) = table.d2ed2a(0, 1);
    table.d2ed2a(2, 0) = table.d2ed2a(0, 2);
    table.d2ed2a(2, 1) = table.d2ed2a(1, 2);
    return table;
}

__device__ DiffTable2 make_diff_table2(const Vec2f &S, float mu, float lmd) {
    DiffTable2 table;
    float a = S[0];
    float b = S[1];
    table.deda[0] = lmd * b * (a * b - 1.0f) + mu * (a - b);
    table.deda[1] = lmd * a * (a * b - 1.0f) + mu * (b - a);
    table.d2ed2a(0, 0) = lmd * b * b + mu;
    table.d2ed2a(1, 1) = lmd * a * a + mu;
    table.d2ed2a(0, 1) = 2.0f * lmd * a * b - lmd - mu;
    table.d2ed2a(1, 0) = table.d2ed2a(0, 1);
    return table;
}

} // namespace SNHk

#endif