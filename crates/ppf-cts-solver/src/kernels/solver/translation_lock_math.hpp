// File: translation_lock_math.hpp
// License: Apache v2.0

#ifndef TRANSLATION_LOCK_MATH_HPP
#define TRANSLATION_LOCK_MATH_HPP

// THE SPELLINGS THIS FILE NAMES, QUOTED AND UNCONDITIONALLY. The run-time
// shader assembler neutralizes a quoted include as it splices, so this line
// costs the shipped shader nothing while an offline translation unit, which is
// what compiles a generated entry rendering, follows it and resolves `Vec3f`.
// It is deliberately NOT `data.hpp`: that header reaches `vec/vec.hpp`, whose
// raw `T *` has no address space, and the shader compiler cannot read it in any
// position.
#include "../linalg/type_aliases.hpp"
// `TranslationLock`, which the row builder reads.
#include "../data_records.hpp"

// MSL REQUIRES AN ADDRESS SPACE ON EVERY REFERENCE TYPE, and a shared header is
// not a neutral kernel body, so it spells one itself. CUDA and the host have a
// single address space and are handed the name defined empty. The pattern is
// `accd::park_floor`'s and `fix::gradient`'s.
#ifndef SM_THREAD
#define SM_THREAD
#define TRANSLATION_LOCK_MATH_UNDEF_THREAD
#endif

// One lock group's frame: the orthonormal row bases, the group's inertia and
// center of mass, the pseudoinverse of its Gram matrix and its right-hand side.
// It lives here rather than beside the projector because both the host that
// builds it and the neutral kernels that read it need the layout, and a second
// declaration of it is the mirror pair the seam exists to remove.
struct LockFrame {
    Vec3f translation_basis0;
    Vec3f translation_basis1;
    Vec3f translation_basis2;
    Vec3f rotation_basis0;
    Vec3f rotation_basis1;
    Vec3f rotation_basis2;
    Vec3f com_relative;
    Mat3x3f inv_inertia;
    Mat6x6f gram_pinv;
    Vec6f rhs;
    unsigned row_mask;
};
// THE AGGREGATE ROWS' ALGEBRA, AT GLOBAL SCOPE AND PREFIXED, because every
// record and function a neutral kernel body names has to be reachable from one
// and no `.kernel.cpp` in this tree opens a namespace.

// Which of the up-to-six aggregate rows a group's frame carries. An
// ENUMERATOR rather than a `constexpr` variable: MSL refuses a program-scope
// variable outside the constant address space, and an enumerator is no variable.
//
// BIT INDEX EQUALS ROW INDEX in LockRowCoefficients::row[], and
// lock_row_coefficients, lock_rows_times_vector, lock_rows_transpose_times, the
// Gram outer product and the host's tangent check all rely on it. The
// translation and rotation blocks are contiguous for the same reason, matching
// the PDRD reduced layout: the translation block at offset 0, the rotation
// block at offset 3.
enum : unsigned {
    // The widest constraint set: three translation rows plus three angular ones.
    LOCK_MAX_ROWS = 6u,
    LOCK_TRANSLATION_ROW_COUNT = 3u,
    // Where the rotation block starts. The host shifts `row_mask` down by this
    // to read the rotation rows as a contiguous mask.
    LOCK_ROTATION_ROW_BASE = LOCK_TRANSLATION_ROW_COUNT,
    LOCK_ROTATION_ROW_COUNT = 3u,

    LOCK_TRANSLATION_ROW0 = 1u << 0,
    LOCK_TRANSLATION_ROW1 = 1u << 1,
    LOCK_TRANSLATION_ROW2 = 1u << 2,
    LOCK_ROTATION_ROW0 = 1u << (LOCK_ROTATION_ROW_BASE + 0),
    LOCK_ROTATION_ROW1 = 1u << (LOCK_ROTATION_ROW_BASE + 1),
    LOCK_ROTATION_ROW2 = 1u << (LOCK_ROTATION_ROW_BASE + 2),
    LOCK_ROTATION_ROW_MASK =
        LOCK_ROTATION_ROW0 | LOCK_ROTATION_ROW1 | LOCK_ROTATION_ROW2,
};

// ONE definition each, compiled for host and device alike. These replace a
// `__device__` and a `__host__` copy that were the same arithmetic written
// twice, which is the mirror pair the seam exists to remove.
SM_INLINE_DEVICE_HOST Vec3f lock_matvec3(SM_THREAD const Mat3x3f &m,
                                         SM_THREAD const Vec3f &v) {
    return Vec3f(m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2],
                 m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2],
                 m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2]);
}

SM_INLINE_DEVICE_HOST Vec6f lock_matvec6(SM_THREAD const Mat6x6f &m,
                                         SM_THREAD const Vec6f &v) {
    Vec6f out = Vec6f::Zero();
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        for (unsigned col = 0; col < LOCK_MAX_ROWS; ++col) {
            out[row] += m(row, col) * v[col];
        }
    }
    return out;
}

// Scalar coefficients of the up-to-six aggregate rows for one vertex. A
// separate Vec3f per row because each row acts on a vertex's xyz block.
struct LockRowCoefficients {
    Vec3f row[LOCK_MAX_ROWS];
};

SM_INLINE_DEVICE_HOST LockRowCoefficients
lock_row_coefficients(SM_THREAD const TranslationLock &lock,
                      SM_THREAD const LockFrame &frame,
                      SM_THREAD const Vec3f &position, float mass) {
    LockRowCoefficients out{};
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        out.row[row] = Vec3f::Zero();
    }
    if (frame.row_mask & LOCK_TRANSLATION_ROW0) {
        out.row[0] = mass * frame.translation_basis0;
    }
    if (frame.row_mask & LOCK_TRANSLATION_ROW1) {
        out.row[1] = mass * frame.translation_basis1;
    }
    if (frame.row_mask & LOCK_TRANSLATION_ROW2) {
        out.row[2] = mass * frame.translation_basis2;
    }
    if (frame.row_mask & LOCK_ROTATION_ROW_MASK) {
        // r is a difference throughout: subtracting the anchor first takes the
        // absolute magnitude out before the cross products below scale it.
        const Vec3f r =
            (position - lock.anchor).cast<float>() - frame.com_relative;
        if (frame.row_mask & LOCK_ROTATION_ROW0) {
            const Vec3f u = lock_matvec3(frame.inv_inertia, frame.rotation_basis0);
            out.row[LOCK_ROTATION_ROW_BASE + 0] = mass * u.cross(r);
        }
        if (frame.row_mask & LOCK_ROTATION_ROW1) {
            const Vec3f u = lock_matvec3(frame.inv_inertia, frame.rotation_basis1);
            out.row[LOCK_ROTATION_ROW_BASE + 1] = mass * u.cross(r);
        }
        if (frame.row_mask & LOCK_ROTATION_ROW2) {
            const Vec3f u = lock_matvec3(frame.inv_inertia, frame.rotation_basis2);
            out.row[LOCK_ROTATION_ROW_BASE + 2] = mass * u.cross(r);
        }
    }
    return out;
}

SM_INLINE_DEVICE_HOST Vec3f
lock_rows_transpose_times(SM_THREAD const LockRowCoefficients &c,
                          SM_THREAD const Vec6f &lambda) {
    Vec3f out = Vec3f::Zero();
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        out += lambda[row] * c.row[row];
    }
    return out;
}

SM_INLINE_DEVICE_HOST Vec6f
lock_rows_times_vector(SM_THREAD const LockRowCoefficients &c,
                       SM_THREAD const Vec3f &v) {
    Vec6f out = Vec6f::Zero();
    for (unsigned row = 0; row < LOCK_MAX_ROWS; ++row) {
        out[row] = c.row[row].dot(v);
    }
    return out;
}

namespace translation_lock {

// How a lock spells "this group has no vertex under it". A per-vertex slot
// carrying it is skipped rather than treated as group zero.
enum : unsigned { UNSET = 0xffffffffu };


// B = I - a a^T for a normalized lock axis a. Shared by the projector, by its
// host and device regression test, and by the neutral drift kernel.
SM_INLINE_DEVICE_HOST Vec3f perpendicular(SM_THREAD const Vec3f &v,
                                          SM_THREAD const Vec3f &axis) {
    const float along = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
    return Vec3f(v[0] - along * axis[0], v[1] - along * axis[1],
                 v[2] - along * axis[2]);
}

// ONE definition of "this lock component is on", shared by the deformable
// projector, the PDRD reduced projector and the host that builds the frames.
//
// THE MODE CARRIES THE ENABLE BIT, NOT THE AXIS. An axis-mode component is on
// exactly when its axis is nonzero; an all-axes component has no axis at all
// (it is required to be exactly zero) and is on by its mode alone. Testing the
// axis directly therefore reads every all-axes lock as disabled while the UI
// still reports it as set, which is silent and produces no wrong number to
// notice. There is deliberately no `axis_enabled` helper left here for a caller
// to reach for by mistake.
SM_INLINE_DEVICE_HOST bool
translation_lock_enabled(SM_THREAD const TranslationLock &lock) {
    return lock.translation_mode == TRANSLATION_LOCK_ALL ||
           lock.axis[0] != 0.0f || lock.axis[1] != 0.0f || lock.axis[2] != 0.0f;
}

SM_INLINE_DEVICE_HOST bool
rotation_lock_enabled(SM_THREAD const TranslationLock &lock) {
    return lock.rotation_mode == ROTATION_LOCK_ALL ||
           lock.rotation_axis[0] != 0.0f || lock.rotation_axis[1] != 0.0f ||
           lock.rotation_axis[2] != 0.0f;
}

// The part of a center-of-mass displacement the translation lock forbids. An
// axis mode leaves the component ALONG its axis free, so only the perpendicular
// part is constrained; an all-axes mode constrains the whole vector. Both the
// drift accumulation and the end-of-step invariant read the displacement
// through this, so the two cannot disagree about what is locked.
SM_INLINE_DEVICE_HOST Vec3f
constrained_translation(SM_THREAD const TranslationLock &lock,
                        SM_THREAD const Vec3f &delta) {
    return lock.translation_mode == TRANSLATION_LOCK_ALL
               ? delta
               : perpendicular(delta, lock.axis);
}

SM_INLINE_DEVICE_HOST bool translation_mode_valid(unsigned mode) {
    return mode == TRANSLATION_LOCK_AXIS || mode == TRANSLATION_LOCK_ALL;
}

SM_INLINE_DEVICE_HOST bool rotation_mode_valid(unsigned mode) {
    return mode == ROTATION_LOCK_ALLOW_ONLY ||
           mode == ROTATION_LOCK_PROHIBIT_AXIS || mode == ROTATION_LOCK_ALL;
}

} // namespace translation_lock

#ifdef TRANSLATION_LOCK_MATH_UNDEF_THREAD
#undef SM_THREAD
#undef TRANSLATION_LOCK_MATH_UNDEF_THREAD
#endif

#endif
