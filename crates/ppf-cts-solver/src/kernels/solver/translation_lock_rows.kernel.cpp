// File: translation_lock_rows.kernel.cpp
// License: Apache v2.0

// THE PER-VERTEX ROWS OF THE AGGREGATE LOCK. Each is dispatched over VERTICES
// and needs the row algebra, which is what separates them from the per-group
// rows in `translation_lock_frames.kernel.cpp`.
//
// A VERTEX WITH A REMOVED DOF IS SKIPPED, and so is one whose group is a PDRD
// body: that body carries its own rigid frame and is projected by
// `pdrd_project_body_dofs_row` instead.

#include "translation_lock_math.hpp"

// The group's accumulated constraint-space image of a per-vertex field,
// `C v` summed over the group's free vertices. Four float atomics because the
// aggregate carries up to four rows.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_row_sums_accumulate_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    const unsigned *dof_mask,
    const LockFrame *frames,
    const float *values,
    compute::atomic_float_t *sums, unsigned i) {
    if (dof_mask[i] != 0u) {
        return;
    }
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        return;
    }
    // Copied into thread space because the row builder takes thread-space
    // references and a `[[seam::device]]` base pointer yields a device lvalue.
    const TranslationLock lock = locks[li];
    if (lock.pdrd_body_index != 0) {
        return;
    }
    const LockFrame frame = frames[li];
    const Vec3f position = positions[i];
    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, prop[i].mass);
    const Vec3f value(values[3u * i + 0u], values[3u * i + 1u],
                      values[3u * i + 2u]);
    const Vec6f contribution = lock_rows_times_vector(c, value);
    // A LOOP RATHER THAN ONE LINE PER COMPONENT: this accumulates a whole
    // constraint-space vector, so a hand-written component list silently drops
    // whichever rows it does not mention, and a dropped row is an
    // unconstrained direction that nothing downstream reports.
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        compute::atomic_add(sums + LOCK_MAX_ROWS * li + row, contribution[row]);
    }
}

// One residual correction toward the group's right-hand side: the free part of
// `q` is moved by `C^T (C C^T)^+ (rhs - C q)`. Repeating it is what makes the
// float32 pseudoinverse act as the exact projector the constraint asks for.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_refine_toward_rhs_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    const unsigned *dof_mask,
    const LockFrame *frames,
    const float *sums,
    float *values, unsigned i) {
    if (dof_mask[i] != 0u) {
        return;
    }
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        return;
    }
    const TranslationLock lock = locks[li];
    if (lock.pdrd_body_index != 0) {
        return;
    }
    const LockFrame frame = frames[li];
    const Vec3f position = positions[i];
    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, prop[i].mass);
    Vec6f accumulated = Vec6f::Zero();
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        accumulated[row] = sums[LOCK_MAX_ROWS * li + row];
    }
    const Vec6f residual = frame.rhs - accumulated;
    const Vec6f lambda = lock_matvec6(frame.gram_pinv, residual);
    const Vec3f correction = lock_rows_transpose_times(c, lambda);
    values[3u * i + 0u] += correction[0];
    values[3u * i + 1u] += correction[1];
    values[3u * i + 2u] += correction[2];
}

// Project the constrained rows OUT of a per-vertex field: `v -= C^T (C C^T)^+ C
// v`. A vertex whose DOF was removed is ZEROED rather than skipped, because an
// exactly prescribed row contributes nothing to a tangent direction and leaving
// its old value in would.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_project_out_rows_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    const unsigned *dof_mask,
    const LockFrame *frames,
    const float *sums,
    float *values, unsigned i) {
    if (dof_mask[i] != 0u) {
        values[3u * i + 0u] = 0.0f;
        values[3u * i + 1u] = 0.0f;
        values[3u * i + 2u] = 0.0f;
        return;
    }
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        return;
    }
    const TranslationLock lock = locks[li];
    if (lock.pdrd_body_index != 0) {
        return;
    }
    const LockFrame frame = frames[li];
    const Vec3f position = positions[i];
    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, prop[i].mass);
    Vec6f accumulated = Vec6f::Zero();
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        accumulated[row] = sums[LOCK_MAX_ROWS * li + row];
    }
    const Vec6f lambda = lock_matvec6(frame.gram_pinv, accumulated);
    const Vec3f correction = lock_rows_transpose_times(c, lambda);
    values[3u * i + 0u] -= correction[0];
    values[3u * i + 1u] -= correction[1];
    values[3u * i + 2u] -= correction[2];
}

// The affine free solution, before the residual refinements above sharpen it:
// `q = C^T (C C^T)^+ rhs` on a free vertex, the seed value copied through on an
// exactly prescribed one, and zero where no aggregate row reaches.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_seed_free_solution_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    const unsigned *dof_mask,
    const LockFrame *frames,
    const float *seed,
    float *values, unsigned i) {
    if (dof_mask[i] != 0u) {
        values[3u * i + 0u] = seed[3u * i + 0u];
        values[3u * i + 1u] = seed[3u * i + 1u];
        values[3u * i + 2u] = seed[3u * i + 2u];
        return;
    }
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        values[3u * i + 0u] = 0.0f;
        values[3u * i + 1u] = 0.0f;
        values[3u * i + 2u] = 0.0f;
        return;
    }
    const TranslationLock lock = locks[li];
    if (lock.pdrd_body_index != 0) {
        values[3u * i + 0u] = 0.0f;
        values[3u * i + 1u] = 0.0f;
        values[3u * i + 2u] = 0.0f;
        return;
    }
    const LockFrame frame = frames[li];
    const Vec3f position = positions[i];
    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, prop[i].mass);
    const Vec6f lambda = lock_matvec6(frame.gram_pinv, frame.rhs);
    const Vec3f value = lock_rows_transpose_times(c, lambda);
    values[3u * i + 0u] = value[0];
    values[3u * i + 1u] = value[1];
    values[3u * i + 2u] = value[2];
}

// The group's mass-weighted angular momentum of a step, `sum m r x dx`, taken
// about the group centroid. Only a group that locks rotation has anything to
// check, which is what the axis test skips.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_torque_accumulate_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    const LockFrame *frames,
    const float *step,
    compute::atomic_float_t *torque, unsigned i) {
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
    const Vec3f dx(step[3u * i + 0u], step[3u * i + 1u], step[3u * i + 2u]);
    const Vec3f contribution = prop[i].mass * r.cross(dx);
    compute::atomic_add(torque + 3u * li + 0u, contribution[0]);
    compute::atomic_add(torque + 3u * li + 1u, contribution[1]);
    compute::atomic_add(torque + 3u * li + 2u, contribution[2]);
}

// THE CONSTRAINT ASSEMBLY, which is the one row that writes THREE destinations:
// the group's constrained drift, the contribution of its exactly prescribed
// vertices, and its Gram matrix. A free vertex contributes to the Gram, a
// prescribed one to the fixed part, and every vertex under a translation lock
// to the drift.
//
// THE FIXED PART IS ITS OWN BUFFER RATHER THAN A MEMBER OF THE FRAME, because a
// record addresses one allocation per FIELD and cannot name a member at an
// offset inside a struct it also reads. It is consumed by the host fold in the
// same function, so nothing else had to learn about it.
[[seam::entry(i)]]
[[seam::device_fn]] inline void lock_constraint_assemble_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const VertexProp *prop,
    const Vec3f *positions,
    const Vec3f *initial,
    const unsigned *dof_mask,
    const LockFrame *frames,
    const float *seed,
    compute::atomic_float_t *drift,
    compute::atomic_float_t *fixed,
    compute::atomic_float_t *gram, unsigned i) {
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        return;
    }
    const TranslationLock lock = locks[li];
    const LockFrame frame = frames[li];
    const Vec3f position = positions[i];
    const float mass = prop[i].mass;
    if (translation_lock::translation_lock_enabled(lock)) {
        const Vec3f anchor = initial[i];
        const Vec3f delta = (position - anchor).cast<float>();
        // THROUGH `constrained_translation`, NOT `perpendicular`. An axis mode
        // leaves the component along its own axis free and an all-axes mode
        // constrains the whole vector, and the end-of-step invariant reads the
        // displacement through the same helper, so the two cannot disagree
        // about what is locked.
        const Vec3f moved =
            mass * translation_lock::constrained_translation(lock, delta);
        compute::atomic_add(drift + 3u * li + 0u, moved[0]);
        compute::atomic_add(drift + 3u * li + 1u, moved[1]);
        compute::atomic_add(drift + 3u * li + 2u, moved[2]);
    }
    // A PDRD group's aggregate DOFs are projected in the reduced six-vector, so
    // it contributes no rows here.
    if (lock.pdrd_body_index != 0) {
        return;
    }
    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, mass);
    if (dof_mask[i] != 0u) {
        const Vec3f prescribed(seed[3u * i + 0u], seed[3u * i + 1u],
                               seed[3u * i + 2u]);
        const Vec6f contribution = lock_rows_times_vector(c, prescribed);
        for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
            compute::atomic_add(fixed + LOCK_MAX_ROWS * li + row,
                                contribution[row]);
        }
        return;
    }
    // Column-major, matching `SMat::operator()(r, c) == base[r + c * ld]`.
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        for (unsigned col = 0; col < LOCK_MAX_ROWS; ++col) {
            compute::atomic_add(gram + LOCK_MAX_ROWS * LOCK_MAX_ROWS * li +
                                    LOCK_MAX_ROWS * col + row,
                                c.row[row].dot(c.row[col]));
        }
    }
}
