// File: translation_lock_math.hpp
// License: Apache v2.0

#ifndef TRANSLATION_LOCK_MATH_HPP
#define TRANSLATION_LOCK_MATH_HPP

#include "../data.hpp"

namespace translation_lock {

// B = I - a a^T for a normalized lock axis a. This small shared primitive is
// used by the CUDA projector and its host/device regression test.
__host__ __device__ inline Vec3f perpendicular(const Vec3f &v,
                                                const Vec3f &axis) {
    const float along = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    return Vec3f(v[0] - along * axis[0], v[1] - along * axis[1],
                 v[2] - along * axis[2]);
}

} // namespace translation_lock

#endif
