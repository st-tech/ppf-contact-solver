// File: hook.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef HOOK_HPP
#define HOOK_HPP

#include "../../common.hpp"
#include "../../data.hpp"

namespace hook {

__device__ float energy(const Vec3f &x0, const Vec3f &x1, float l0) {
    Vec3f t = x1 - x0;
    float r = t.norm() / l0 - 1.0f;
    return 0.5f * r * r;
}

__device__ void make_diff_table(const Vec3f &x0, const Vec3f &x1, float l0,
                                float weight, Mat3x2f &gradient,
                                Mat6x6f &hessian) {
    Vec3f t = x1 - x0;
    float l = t.norm();
    Vec3f n = t / l;
    Mat3x6f dtdx;
    dtdx << -Mat3x3f::Identity(), Mat3x3f::Identity();
    Vec3f dedt = (l / l0 - 1.0f) * n;
    Vec6f g = dtdx.transpose() * n;
    float r = (l - l0) / l;
    float c0 = fmaxf(0.0f, 1.0f - r) / l0;
    float c1 = fmaxf(0.0f, r / l0);
    gradient.col(0) = -weight * dedt;
    gradient.col(1) = weight * dedt;
    hessian = weight * (c0 * g * g.transpose() + c1 * dtdx.transpose() * dtdx);
}

} // namespace hook

#endif
