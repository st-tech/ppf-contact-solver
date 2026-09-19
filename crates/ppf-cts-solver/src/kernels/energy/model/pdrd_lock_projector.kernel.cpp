// File: pdrd_lock_projector.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A NEUTRAL KERNEL SOURCE holding the per-body reduced-DOF projector. This file
// is plain C++ and belongs to no backend: no preprocessor conditional, no macro
// of its own, and no spelling that only nvcc or only the Metal shader compiler
// accepts. ppf-cts-compute/seam/kernelgen.py renders it into the three forms the three
// compilers read. The two facts a backend cannot infer are written as C++
// attributes: `[[seam::host_device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a pointer or reference parameter,
// which MSL requires on every one of them.
//
// A PDRD body owns six reduced coordinates (dx_b, dtheta_b); a joint or a lock
// forbids a subspace of them, and this file builds one orthonormal basis of the
// FORBIDDEN subspace and removes it.
//
// ONE COMBINED BASIS, NOT A SEQUENCE OF PROJECTORS. A hinge, a Lock Translation
// and a Lock Rotation can all bind the same body, and two rotational
// restrictions about non-parallel axes do not commute: applying their
// projectors one after another can leave a component both of them forbid. The
// rows are therefore orthonormalized together by modified Gram-Schmidt and
// removed in one pass, which intersects the constraints exactly.
// tests/kernels/translation_lock.cpp measures precisely that
// (test_pdrd_hinge_and_rotation_lock_intersection).
//
// The Gram-Schmidt drop threshold is a DUPLICATE-ROW test, not a compliance
// tolerance. Every source row is a unit vector in a 6-dimensional space, so a
// residual norm below it means the row is algebraically a combination of the
// rows already held; dropping it leaves the same null space. The threshold sits
// at 4096 * fp32 eps on the SQUARED norm, i.e. a residual length of about
// 0.022, four decades above the round-off of a unit-scale projection, so a
// genuinely independent row can never fall through it.
//
// STATUS RATHER THAN assert. CUDA runs its device asserts live in release and
// Metal has no device assert at all, so a shared body cannot spell its own trap.
// The build returns a status code and each backend routes it: the CUDA kernel
// asserts on it, and the Metal kernel reports it through the diagnostic ring.

#pragma once

// The `fmath::` arithmetic the bodies below call (fmath::abs, fmath::sqrt,
// fmath::div, fmath::isnan, fmath::isinf) comes from the backend prologue,
// which is seam/seam.hpp under nvcc and on the host (data.hpp includes it
// first) and kMslMacroSeam in metal/shader_compiler.mm under MSL. The bodies
// are reached from the host as well as from a kernel, the host caller being
// tests/kernels/translation_lock.cpp, so they take the host-and-device
// execution space rather than the device-only one.

#include "../../linalg/smat.hpp"
// PDRD_JOINT_HINGE, the joint mode this projector dispatches on, so the
// value is read from the one place that defines it. The Metal backend resolves
// this include itself and places the pdrd_rigid.kernel.cpp segment ahead of
// this one, which is where the enumerator comes from there.
#include "pdrd_rigid.kernel.cpp"
// The one vector name the bodies below use. data.hpp declares the same alias
// for the same underlying type, and the shader's alias segment declares it
// again, so any arrival order is fine: an alias redeclaration naming an
// identical type is legal.
using Vec3f = linalg::SVec<float, 3>;

// The two Lock Rotation semantics, and the projector's own sizes and status
// codes. Enumerators rather than `constexpr` variables, for the reason
// pdrd_rigid.kernel.cpp states: MSL rejects a program-scope constexpr variable
// outright, and an enumerator carries the same value in both languages.
//
// The mode values MUST equal `RotationLockMode` and `TranslationLockMode` in
// data_records.hpp, which is what the Rust side writes into
// `TranslationLock::rotation_mode` and `::translation_mode`.
enum : unsigned {
    LOCK_MODE_ALLOW_ONLY = 0u,
    LOCK_MODE_PROHIBIT_AXIS = 1u,
    // No net rotation about any axis: the whole rotation block goes.
    LOCK_MODE_ALL = 2u,
    // `TranslationLockMode`. An axis mode holds the center of mass on a line
    // along its axis; all-axes holds it at a point.
    LOCK_TRANSLATION_MODE_AXIS = 0u,
    LOCK_TRANSLATION_MODE_ALL = 1u,
    // A reduced body block is six coordinates, so at most six independent rows
    // can be removed, and the basis is that many rows of six floats.
    PDRD_LOCK_MAX_ROWS = 6u,
    PDRD_LOCK_BASIS_FLOATS = 36u,
    // Status. OK is zero so a caller can test the code directly.
    PDRD_LOCK_OK = 0u,
    PDRD_LOCK_BAD_AXIS = 1u,      // a zero or non-finite constraint axis
    PDRD_LOCK_BAD_MODE = 2u,      // a mode outside its admissible set
    PDRD_LOCK_OVERFLOW = 3u,      // more than six independent rows
};

// A deterministic orthonormal pair spanning the plane perpendicular to `axis`.
// The reference vector is chosen as the axis's SMALLEST component, so the cross
// product is never taken against a nearly parallel vector and b0 is
// well conditioned for any axis. Returns false when the axis is degenerate,
// leaving b0 and b1 untouched.
[[seam::host_device_fn]] inline bool
pdrd_tangent_basis(const Vec3f &axis,
                       Vec3f &b0,
                       Vec3f &b1) {
    const float n2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if (fmath::isnan(n2) || fmath::isinf(n2) || !(n2 > 0.0f)) {
        return false;
    }
    const float inv = fmath::div(1.0f, fmath::sqrt(n2));
    const Vec3f a(axis[0] * inv, axis[1] * inv, axis[2] * inv);
    const float ax = fmath::abs(a[0]), ay = fmath::abs(a[1]),
                az = fmath::abs(a[2]);
    const Vec3f ref = ax <= ay && ax <= az ? Vec3f(1.0f, 0.0f, 0.0f)
                      : ay <= az           ? Vec3f(0.0f, 1.0f, 0.0f)
                                           : Vec3f(0.0f, 0.0f, 1.0f);
    b0 = a.cross(ref);
    b0 = b0 * fmath::div(1.0f, fmath::sqrt(b0.dot(b0)));
    // a and b0 are orthonormal, so their cross product is already unit length.
    b1 = a.cross(b0);
    return true;
}

// Append one row after modified Gram-Schmidt against the rows already held.
// `basis` is PDRD_LOCK_BASIS_FLOATS floats, row major: row r starts at
// basis[6 * r]. Returns the new row count, which is `count` unchanged when the
// row is algebraically a duplicate. `status` receives PDRD_LOCK_OVERFLOW if
// a seventh independent row is ever produced, which the six-dimensional space
// makes impossible and which is therefore a defect rather than an input.
[[seam::host_device_fn]] inline unsigned pdrd_append_constraint(
    float *basis, unsigned count,
    const float *source, unsigned *status) {
    float row[6];
    for (unsigned i = 0; i < 6u; ++i) {
        row[i] = source[i];
    }
    for (unsigned r = 0; r < count; ++r) {
        float dot = 0.0f;
        for (unsigned i = 0; i < 6u; ++i) {
            dot += basis[6u * r + i] * row[i];
        }
        for (unsigned i = 0; i < 6u; ++i) {
            row[i] -= dot * basis[6u * r + i];
        }
    }
    float norm2 = 0.0f;
    for (unsigned i = 0; i < 6u; ++i) {
        norm2 += row[i] * row[i];
    }
    if (norm2 <= 4096.0f * 1.19209290e-7f) {
        return count;
    }
    if (count >= PDRD_LOCK_MAX_ROWS) {
        *status = PDRD_LOCK_OVERFLOW;
        return count;
    }
    const float inv_norm = fmath::div(1.0f, fmath::sqrt(norm2));
    for (unsigned i = 0; i < 6u; ++i) {
        basis[6u * count + i] = row[i] * inv_norm;
    }
    return count + 1u;
}

// Forbid the two directions PERPENDICULAR to `axis`, in either the translation
// block (offset 0) or the rotation block (offset 3) of a reduced body vector.
// What survives is motion along the axis alone.
[[seam::host_device_fn]] inline unsigned pdrd_append_axis_lock(
    float *basis, unsigned count,
    const Vec3f &axis, unsigned offset,
    unsigned *status) {
    Vec3f b0, b1;
    if (!pdrd_tangent_basis(axis, b0, b1)) {
        *status = PDRD_LOCK_BAD_AXIS;
        return count;
    }
    float row0[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float row1[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned i = 0; i < 3u; ++i) {
        row0[offset + i] = b0[i];
        row1[offset + i] = b1[i];
    }
    count = pdrd_append_constraint(basis, count, row0, status);
    return pdrd_append_constraint(basis, count, row1, status);
}

// Remove ALL THREE components of one block of a reduced body vector: the
// translation block (offset 0) for a center of mass pinned to a point, or the
// rotation block (offset 3) for a body with no angular freedom at all.
//
// It needs no axis, which is exactly why an all-axes lock ships a zero one and
// why enablement is asked of the MODE everywhere upstream.
[[seam::host_device_fn]] inline unsigned pdrd_append_all_lock(
    float *basis, unsigned count, unsigned offset,
    unsigned *status) {
    for (unsigned i = 0; i < 3u; ++i) {
        float row[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        row[offset + i] = 1.0f;
        count = pdrd_append_constraint(basis, count, row, status);
    }
    return count;
}

// Lock Rotation has three distinct null spaces. Allow-only forbids the two
// directions perpendicular to the axis, leaving spin about it; prohibit-axis
// forbids the axis alone, leaving the perpendicular plane; all-axes forbids the
// whole rotation block.
[[seam::host_device_fn]] inline unsigned pdrd_append_rotation_lock(
    float *basis, unsigned count,
    const Vec3f &axis, unsigned mode,
    unsigned *status) {
    if (mode == LOCK_MODE_ALL) {
        return pdrd_append_all_lock(basis, count, 3u, status);
    }
    if (mode == LOCK_MODE_ALLOW_ONLY) {
        return pdrd_append_axis_lock(basis, count, axis, 3u, status);
    }
    if (mode != LOCK_MODE_PROHIBIT_AXIS) {
        *status = PDRD_LOCK_BAD_MODE;
        return count;
    }
    const float n2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if (fmath::isnan(n2) || fmath::isinf(n2) || !(n2 > 0.0f)) {
        *status = PDRD_LOCK_BAD_AXIS;
        return count;
    }
    const float inv = fmath::div(1.0f, fmath::sqrt(n2));
    float row[6] = {0.0f,           0.0f,           0.0f,
                    axis[0] * inv, axis[1] * inv, axis[2] * inv};
    return pdrd_append_constraint(basis, count, row, status);
}

// Build the combined forbidden basis for one body. `basis` receives
// PDRD_LOCK_BASIS_FLOATS floats and the return value is how many of its
// rows are live. `status` is set to the FIRST fault seen and is left alone
// otherwise, so the caller initializes it to PDRD_LOCK_OK.
//
// A hinge forbids all three translations and the two rotations off the axle,
// which is the four-row form of "pinned at a point, free to spin about one
// axis". The three translation rows are the coordinate axes rather than a
// tangent pair, so a hinge needs no axis of its own for that half.
//
// THE APPEND ORDER IS LOAD-BEARING: modified Gram-Schmidt is order-dependent,
// so hinge rows first, then translation, then rotation is what keeps a scene
// that uses only the per-axis modes producing the same basis it would without
// the all-axes branches.
//
// SIX ROWS SATURATE THE SPACE, which is what a body with both all-axes locks
// asks for, and that is allowed. It needs no special case:
// pdrd_append_constraint drops an algebraically dependent row on its norm test
// before the row budget is reached, and a seventh INDEPENDENT row cannot exist
// in R^6, so PDRD_LOCK_OVERFLOW stays a genuine impossibility rather than a
// bound to raise.
[[seam::host_device_fn]] inline unsigned pdrd_build_body_basis(
    unsigned joint_mode, const Vec3f &joint_axis,
    bool has_translation_lock, const Vec3f &translation_axis,
    unsigned translation_mode,
    bool has_rotation_lock, const Vec3f &rotation_axis,
    unsigned rotation_mode, float *basis,
    unsigned *status) {
    unsigned count = 0u;
    if (joint_mode == PDRD_JOINT_HINGE) {
        for (unsigned i = 0; i < 3u; ++i) {
            float row[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            row[i] = 1.0f;
            count = pdrd_append_constraint(basis, count, row, status);
        }
        count = pdrd_append_axis_lock(basis, count, joint_axis, 3u, status);
    }
    if (has_translation_lock) {
        if (translation_mode == LOCK_TRANSLATION_MODE_ALL) {
            count = pdrd_append_all_lock(basis, count, 0u, status);
        } else if (translation_mode == LOCK_TRANSLATION_MODE_AXIS) {
            count = pdrd_append_axis_lock(basis, count, translation_axis, 0u,
                                              status);
        } else {
            *status = PDRD_LOCK_BAD_MODE;
        }
    }
    if (has_rotation_lock) {
        count = pdrd_append_rotation_lock(basis, count, rotation_axis,
                                              rotation_mode, status);
    }
    return count;
}

// Remove the span of the first `count` rows of `basis` from one body's six
// reduced coordinates, in place. The rows are orthonormal, so this is the
// exact Euclidean projector I - sum_r b_r b_r^T and repeating it is a no-op to
// round-off.
[[seam::host_device_fn]] inline void pdrd_apply_body_basis(
    const float *basis, unsigned count,
    float *reduced) {
    for (unsigned r = 0; r < count; ++r) {
        float dot = 0.0f;
        for (unsigned i = 0; i < 6u; ++i) {
            dot += basis[6u * r + i] * reduced[i];
        }
        for (unsigned i = 0; i < 6u; ++i) {
            reduced[i] -= dot * basis[6u * r + i];
        }
    }
}


// --------------------------------------------------------------- entry point

// One combined orthogonal projector applied to one body's six reduced DOFs.
// Hinge, translation-lock and rotation-lock rows are orthonormalized together
// by the basis builder above, so overlapping rotation restrictions are
// intersected exactly rather than applied as noncommuting projectors in
// sequence.
//
// THE STATUS IS A GUARANTEE-CLASS CHECK, NOT INSTRUMENTATION, which is why it
// travels on the diagnostic lane rather than being returned. A malformed axis
// or mode does not fail loudly on its own: it yields a basis with fewer rows
// than the constraint asks for, which SILENTLY WEAKENS a rigid constraint. An
// assert traps only on a backend whose asserts are live, so the status travels
// on the lane instead and the same trap reaches a backend that has no assert.
[[seam::entry(b)]]
[[seam::device_fn]] inline void pdrd_project_body_dofs_row(
    const unsigned *joint_mode,
    const Vec3f *joint_axis,
    const unsigned *translation_lock,
    const Vec3f *translation_axis,
    const unsigned *translation_mode,
    const unsigned *rotation_lock,
    const Vec3f *rotation_axis,
    const unsigned *rotation_mode,
    float *reduced, unsigned body_base, unsigned b,
    DiagHandle diag) {
    // Every element is copied into thread space before it is used: the two
    // helpers take thread-space references, and a `[[seam::device]]` base
    // pointer yields a device-space lvalue that one of the three backends
    // cannot bind to one.
    const Vec3f axle = joint_axis[b];
    const Vec3f lock_axis = translation_axis[b];
    const Vec3f spin_axis = rotation_axis[b];
    float basis[PDRD_LOCK_BASIS_FLOATS];
    unsigned status = PDRD_LOCK_OK;
    const unsigned count = pdrd_build_body_basis(
        joint_mode[b], axle, translation_lock[b] != PDRD_CLOTH_MARKER,
        lock_axis, translation_mode[b], rotation_lock[b] != PDRD_CLOTH_MARKER,
        spin_axis, rotation_mode[b], basis, &status);
    DIAG_ASSERT4(diag, status == PDRD_LOCK_OK, (float)status, (float)b,
                 (float)joint_mode[b], (float)count);
    // The six reduced coordinates travel through thread space for the same
    // address-space reason. Six floats is what that costs.
    const unsigned base = body_base + 6u * b;
    float q[6];
    for (unsigned i = 0; i < 6u; ++i) {
        q[i] = reduced[base + i];
    }
    pdrd_apply_body_basis(basis, count, q);
    for (unsigned i = 0; i < 6u; ++i) {
        reduced[base + i] = q[i];
    }
}
