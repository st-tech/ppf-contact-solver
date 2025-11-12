// File: fix.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef FIX_HPP
#define FIX_HPP

#include "../../common.hpp"
#include "../../data.hpp"

namespace fix {

__device__ float energy(const Vec3f &x, const Vec3f &y) {
    return 0.5f * (x - y).squaredNorm();
}

__device__ Vec3f gradient(const Vec3f &x, const Vec3f &y) {
    return (x - y);
}

__device__ Mat3x3f hessian() { return Mat3x3f::Identity(); }

} // namespace fix

#endif
