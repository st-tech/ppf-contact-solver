// File: translation_lock_frames.kernel.cpp
// License: Apache v2.0

// THE PER-GROUP HALF OF THE AGGREGATE LOCK. Both rows below are dispatched over
// LOCK GROUPS rather than over vertices, which is what separates them from the
// per-vertex passes: a group's frame is built once and then read by every
// vertex under it.

#include "translation_lock_math.hpp"

// Clear the right-hand side before a frame is reassembled. One of the frame's
// members; the rest are carried.
[[seam::entry(li)]]
[[seam::device_fn]] inline void
lock_frame_clear_row(LockFrame *frames, unsigned li) {
    frames[li].rhs = Vec6f::Zero();
}

// The group's center of mass, relative to its anchor: the accumulated
// mass-weighted sum divided by the group's total mass. A group that locks no
// rotation carries no rotation rows, so its frame needs no centroid, and the
// MODE is what says so: an all-axes rotation lock ships a zero axis, so an axis
// test would skip exactly the groups that need a centroid most.
[[seam::entry(li)]]
[[seam::device_fn]] inline void lock_frame_center_of_mass_row(
    const TranslationLock *locks,
    const float *mass_weighted_sum,
    LockFrame *frames, unsigned li) {
    // The element is copied into thread space before its members are read: a
    // `[[seam::device]]` base pointer yields a device-space lvalue that one of
    // the three backends cannot bind where a body names a member.
    const TranslationLock lock = locks[li];
    if (!translation_lock::rotation_lock_enabled(lock)) {
        return;
    }
    const Vec3f sum(mass_weighted_sum[3u * li + 0u],
                    mass_weighted_sum[3u * li + 1u],
                    mass_weighted_sum[3u * li + 2u]);
    frames[li].com_relative = sum / lock.total_mass;
}

// One vertex's mass-weighted contribution to its group's center of mass,
// measured from the group's ANCHOR so the sum stays at body-extent scale rather
// than at domain scale. Only a group that locks rotation needs a
// centroid at all, which is what the mode test skips; a PDRD body carries its
// own rigid frame and is excluded for the same reason.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_center_of_mass_accumulate_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    compute::atomic_float_t *mass_weighted_sum, unsigned i) {
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        return;
    }
    const TranslationLock lock = locks[li];
    if (lock.pdrd_body_index != 0 ||
        !translation_lock::rotation_lock_enabled(lock)) {
        return;
    }
    // A DIFFERENCE of two positions, so the anchor's own magnitude leaves the
    // sum before the mass scales it.
    const Vec3f position = positions[i];
    const Vec3f weighted = prop[i].mass * (position - lock.anchor).cast<float>();
    compute::atomic_add(mass_weighted_sum + 3u * li + 0u, weighted[0]);
    compute::atomic_add(mass_weighted_sum + 3u * li + 1u, weighted[1]);
    compute::atomic_add(mass_weighted_sum + 3u * li + 2u, weighted[2]);
}

// One vertex's contribution to its group's inertia tensor about the group
// centroid, `m (r.r I - r r^T)`. Nine float atomics, column-major, matching the
// storage `lock_matvec3` reads it back through.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_inertia_accumulate_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    const LockFrame *frames,
    compute::atomic_float_t *inertia, unsigned i) {
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        return;
    }
    const TranslationLock lock = locks[li];
    if (lock.pdrd_body_index != 0 ||
        !translation_lock::rotation_lock_enabled(lock)) {
        return;
    }
    const LockFrame frame = frames[li];
    const Vec3f position = positions[i];
    const Vec3f r = (position - lock.anchor).cast<float>() - frame.com_relative;
    const float mass = prop[i].mass;
    const float r2 = r.dot(r);
    for (unsigned row = 0; row < 3u; ++row) {
        for (unsigned col = 0; col < 3u; ++col) {
            const float value =
                mass * ((row == col ? r2 : 0.0f) - r[row] * r[col]);
            compute::atomic_add(inertia + 9u * li + 3u * col + row, value);
        }
    }
}
