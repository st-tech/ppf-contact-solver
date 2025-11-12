// File: aabb.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef AABB_HPP
#define AABB_HPP

#include "../data.hpp"

#define AABB_MAX_QUERY 128

namespace aabb {

__device__ AABB join(const AABB &a, const AABB &b) {
    return {Vec3f(std::min(a.min[0], b.min[0]), std::min(a.min[1], b.min[1]),
                  std::min(a.min[2], b.min[2])),
            Vec3f(std::max(a.max[0], b.max[0]), std::max(a.max[1], b.max[1]),
                  std::max(a.max[2], b.max[2]))};
}

__device__ AABB make(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2,
                     float _margin) {
    AABB result = {Vec3f(std::min(x0[0], std::min(x1[0], x2[0])),
                         std::min(x0[1], std::min(x1[1], x2[1])),
                         std::min(x0[2], std::min(x1[2], x2[2]))),
                   Vec3f(std::max(x0[0], std::max(x1[0], x2[0])),
                         std::max(x0[1], std::max(x1[1], x2[1])),
                         std::max(x0[2], std::max(x1[2], x2[2])))};
    if (_margin) {
        float margin = _margin;
        for (unsigned i = 0; i < 3; ++i) {
            result.min[i] -= margin;
            result.max[i] += margin;
        }
    }
    return result;
}

__device__ AABB make(const Vec3f &x0, const Vec3f &x1, float _margin) {
    AABB result = {Vec3f(std::min(x0[0], x1[0]), std::min(x0[1], x1[1]),
                         std::min(x0[2], x1[2])),
                   Vec3f(std::max(x0[0], x1[0]), std::max(x0[1], x1[1]),
                         std::max(x0[2], x1[2]))};
    if (_margin) {
        float margin = _margin;
        for (unsigned i = 0; i < 3; ++i) {
            result.min[i] -= margin;
            result.max[i] += margin;
        }
    }
    return result;
}

__device__ AABB make(const Vec3f &x, float _margin) {
    float margin = _margin;
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

template <typename F, typename T>
__device__ unsigned query(const BVH &bvh, Vec<AABB> aabb, F op, T query) {
    unsigned stack[AABB_MAX_QUERY];
    unsigned count = 0;
    unsigned head = 0;
    if (bvh.node.size) {
        stack[head++] = bvh.node.size - 1;
        while (head) {
            unsigned index = stack[--head];
            if (op.test(aabb[index], query)) {
                if (bvh.node[index][1] == 0) {
                    unsigned leaf_index = bvh.node[index][0] - 1;
                    if (op.test(aabb[index], query)) {
                        if (op(leaf_index)) {
                            count++;
                        }
                    }
                } else {
                    if (head + 2 >= AABB_MAX_QUERY) {
                        printf("stack overflow!\n");
                        assert(false);
                        break;
                    } else {
                        stack[head++] = bvh.node[index][0] - 1;
                        stack[head++] = bvh.node[index][1] - 1;
                    }
                }
            }
        }
    }
    return count;
}

} // namespace aabb

#endif
