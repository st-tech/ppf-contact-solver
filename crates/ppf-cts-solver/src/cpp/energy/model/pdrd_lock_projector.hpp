// License: Apache v2.0

#ifndef PDRD_LOCK_PROJECTOR_HPP
#define PDRD_LOCK_PROJECTOR_HPP

#include "../../data.hpp"

#include <cassert>
#include <cmath>

namespace PDRD {

// PDRD joint / DOF-filtering modes (mirror PdrdBodyProp::joint_mode).
constexpr unsigned PDRD_JOINT_FREE = 0u;
constexpr unsigned PDRD_JOINT_HINGE = 1u;
constexpr unsigned RIGID_UNSET = 0xffffffffu;

// Form two unit vectors spanning the plane perpendicular to a normalized axis.
// The caller validates all lock axes before they reach CUDA; the fallback assert
// makes a malformed joint axis fail at the first use rather than silently
// weakening a rigid constraint.
__device__ inline void pdrd_tangent_basis(const Vec3f &axis, Vec3f &b0,
                                          Vec3f &b1) {
    const float n2 = axis.dot(axis);
    assert(isfinite(n2) && n2 > 0.0f);
    const Vec3f a = axis * (1.0f / sqrtf(n2));
    const float ax = fabsf(a[0]), ay = fabsf(a[1]), az = fabsf(a[2]);
    const Vec3f ref =
        ax <= ay && ax <= az ? Vec3f(1.0f, 0.0f, 0.0f)
        : ay <= az            ? Vec3f(0.0f, 1.0f, 0.0f)
                              : Vec3f(0.0f, 0.0f, 1.0f);
    b0 = a.cross(ref);
    b0 *= 1.0f / sqrtf(b0.dot(b0));
    b1 = a.cross(b0);
}

// Append one row after modified Gram-Schmidt. All source rows have unit scale,
// so this threshold only drops algebraically duplicate constraints. It does not
// relax a lock: removing a duplicate leaves the same null space.
__device__ inline void pdrd_append_constraint(float basis[6][6],
                                              unsigned &count,
                                              const float source[6]) {
    float row[6];
    for (unsigned i = 0; i < 6; ++i) row[i] = source[i];
    for (unsigned r = 0; r < count; ++r) {
        float dot = 0.0f;
        for (unsigned i = 0; i < 6; ++i) dot += basis[r][i] * row[i];
        for (unsigned i = 0; i < 6; ++i) row[i] -= dot * basis[r][i];
    }
    float norm2 = 0.0f;
    for (unsigned i = 0; i < 6; ++i) norm2 += row[i] * row[i];
    if (norm2 <= 4096.0f * 1.19209290e-7f) return;
    assert(count < 6u);
    const float inv_norm = 1.0f / sqrtf(norm2);
    for (unsigned i = 0; i < 6; ++i) basis[count][i] = row[i] * inv_norm;
    ++count;
}

// Add the two forbidden components in either the translation (offset 0) or
// rotation (offset 3) block of a reduced body vector.
__device__ inline void pdrd_append_axis_lock(float basis[6][6], unsigned &count,
                                             const Vec3f &axis,
                                             unsigned offset) {
    Vec3f b0, b1;
    pdrd_tangent_basis(axis, b0, b1);
    float row0[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float row1[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned i = 0; i < 3; ++i) {
        row0[offset + i] = b0[i];
        row1[offset + i] = b1[i];
    }
    pdrd_append_constraint(basis, count, row0);
    pdrd_append_constraint(basis, count, row1);
}

// Rotation Lock has two distinct null spaces. Allow-only removes the two
// tangent directions, while prohibit-axis removes only the selected axis.
__device__ inline void pdrd_append_rotation_lock(float basis[6][6],
                                                  unsigned &count,
                                                  const Vec3f &axis,
                                                  unsigned mode) {
    assert(mode == ROTATION_LOCK_ALLOW_ONLY ||
           mode == ROTATION_LOCK_PROHIBIT_AXIS);
    if (mode == ROTATION_LOCK_ALLOW_ONLY) {
        pdrd_append_axis_lock(basis, count, axis, 3u);
        return;
    }
    const float n2 = axis.dot(axis);
    assert(isfinite(n2) && n2 > 0.0f);
    const Vec3f normalized = axis * (1.0f / sqrtf(n2));
    float row[6] = {0.0f, 0.0f, 0.0f, normalized[0], normalized[1],
                    normalized[2]};
    pdrd_append_constraint(basis, count, row);
}

// Apply one combined orthogonal projector to every body's six reduced DOFs.
// Hinge, translation-lock, and rotation-lock rows are orthonormalized together,
// so overlapping rotation restrictions are intersected exactly rather than
// applying noncommuting projectors in sequence.
static __global__ void project_body_dofs_kernel(unsigned nb, unsigned body_base,
                                                Vec<unsigned> jmode,
                                                Vec<Vec3f> jaxis,
                                                Vec<unsigned> tlock,
                                                Vec<Vec3f> tlock_axis,
                                                Vec<unsigned> rlock,
                                                Vec<Vec3f> rlock_axis,
                                                Vec<unsigned> rlock_mode,
                                                Vec<float> u) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    float *q = u.data + body_base + 6u * b;
    float basis[6][6];
    unsigned count = 0;
    if (jmode.data[b] == PDRD_JOINT_HINGE) {
        for (unsigned i = 0; i < 3; ++i) {
            float row[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            row[i] = 1.0f;
            pdrd_append_constraint(basis, count, row);
        }
        pdrd_append_axis_lock(basis, count, jaxis.data[b], 3u);
    }
    if (tlock.data[b] != RIGID_UNSET) {
        pdrd_append_axis_lock(basis, count, tlock_axis.data[b], 0u);
    }
    if (rlock.data[b] != RIGID_UNSET) {
        pdrd_append_rotation_lock(basis, count, rlock_axis.data[b],
                                  rlock_mode.data[b]);
    }
    for (unsigned r = 0; r < count; ++r) {
        float dot = 0.0f;
        for (unsigned i = 0; i < 6; ++i) dot += basis[r][i] * q[i];
        for (unsigned i = 0; i < 6; ++i) q[i] -= dot * basis[r][i];
    }
}

} // namespace PDRD

#endif
