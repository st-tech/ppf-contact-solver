// File: aabb.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef AABB_DEF_HPP
#define AABB_DEF_HPP

#include "../data.hpp"

#define AABB_MAX_QUERY 128

namespace aabb {

__device__ AABB join(const AABB &a, const AABB &b);
__device__ AABB make(const Vec3f &x0, float margin);
__device__ AABB make(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2,
                     float margin);
__device__ AABB make(const Vec3f &x0, const Vec3f &x1, float margin);
__device__ bool overlap(const AABB &a, const AABB &b);

template <typename F, typename T>
__device__ static unsigned query(const BVH &bvh, Vec<AABB> aabb, F op,
                                 T query) {
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
