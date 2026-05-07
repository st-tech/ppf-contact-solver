// File: air_damper.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef air_DAMP_HPP
#define air_DAMP_HPP

#include "../../data.hpp"

namespace air_damper {

__device__ Mat3x3f get_proj_op(const Vec3f &normal) {
    return Mat3x3f::Identity() - normal * normal.transpose();
}

__device__ float face_energy(float dt, const Vec3f &x1, const Vec3f &x0,
                             const Vec3f &normal, const Vec3f &wind,
                             const ParamSet &param) {
    Vec3f z = (x1 - x0) - dt * wind;
    Mat3x3f P = get_proj_op(normal);
    float f = normal.dot(z);
    Vec3f g = P * z;
    return 0.5f * (f * f + param.air_friction * g.squaredNorm()) / (dt * dt);
}

__device__ Vec3f face_gradient(float dt, const Vec3f &x1, const Vec3f &x0,
                               const Vec3f &normal, const Vec3f &wind,
                               const ParamSet &param) {
    Vec3f z = (x1 - x0) - dt * wind;
    Mat3x3f P = get_proj_op(normal);
    return normal * normal.dot(z) / (dt * dt) +
           param.air_friction * P * z / (dt * dt);
}

__device__ Mat3x3f face_hessian(float dt, const Vec3f &normal,
                                const ParamSet &param) {
    Mat3x3f P = get_proj_op(normal);
    return normal * normal.transpose() / (dt * dt) +
           param.air_friction * P / (dt * dt);
}

} // namespace air_damper

#endif
