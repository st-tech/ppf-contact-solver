// File: aabb.cu
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef AABB_HPP
#define AABB_HPP

#include "../data.hpp"

namespace aabb {

__device__ AABB join(const AABB &a, const AABB &b) {
    return {Vec3f(fminf(a.min[0], b.min[0]), fminf(a.min[1], b.min[1]),
                  fminf(a.min[2], b.min[2])),
            Vec3f(fmaxf(a.max[0], b.max[0]), fmaxf(a.max[1], b.max[1]),
                  fmaxf(a.max[2], b.max[2]))};
}

__device__ AABB make(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2,
                     float margin) {
    AABB result = {Vec3f(fminf(x0[0], fminf(x1[0], x2[0])),
                         fminf(x0[1], fminf(x1[1], x2[1])),
                         fminf(x0[2], fminf(x1[2], x2[2]))),
                   Vec3f(fmaxf(x0[0], fmaxf(x1[0], x2[0])),
                         fmaxf(x0[1], fmaxf(x1[1], x2[1])),
                         fmaxf(x0[2], fmaxf(x1[2], x2[2])))};
    if (margin) {
        for (unsigned i = 0; i < 3; ++i) {
            result.min[i] -= margin;
            result.max[i] += margin;
        }
    }
    return result;
}

__device__ AABB make(const Vec3f &x0, const Vec3f &x1, float margin) {
    AABB result = {
        Vec3f(fminf(x0[0], x1[0]), fminf(x0[1], x1[1]), fminf(x0[2], x1[2])),
        Vec3f(fmaxf(x0[0], x1[0]), fmaxf(x0[1], x1[1]), fmaxf(x0[2], x1[2]))};
    if (margin) {
        for (unsigned i = 0; i < 3; ++i) {
            result.min[i] -= margin;
            result.max[i] += margin;
        }
    }
    return result;
}

__device__ AABB make(const Vec3f &x, float margin) {
    return {
        x - Vec3f(margin, margin, margin),
        x + Vec3f(margin, margin, margin),
    };
}

__device__ bool overlap(const AABB &a, const AABB &b) {
    return (a.min[0] <= b.max[0] && a.max[0] >= b.min[0]) &&
           (a.min[1] <= b.max[1] && a.max[1] >= b.min[1]) &&
           (a.min[2] <= b.max[2] && a.max[2] >= b.min[2]);
}

} // namespace aabb

#endif
