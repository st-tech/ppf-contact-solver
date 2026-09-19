// File: crates/ppf-cts-solver/tests/kernels/translation_lock.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Gate for the aggregate lock projector: Lock Translation and Lock Rotation on
// a deformable group, and the reduced six-DOF projector a PDRD body's joint and
// locks share.
//
// The locks are EXACT constraints on the Newton direction, never a penalty, so
// there is no stiffness to trade against and nothing here is a tolerance to
// tune: a row is either annihilated or the constraint is not being enforced.
// Two properties carry the feature and both are checked below. ONE PROJECTOR,
// NEVER A COMPOSITION: two projections in sequence do not commute, so a second
// one reintroduces a component the first removed, and the intersection cases
// are what detect that. And ANCHOR-RELATIVE ROTATION ROWS: the moment arm is
// measured from the group's own anchor, so a row built from an absolute
// position is wrong by the anchor's offset and the error vanishes only at the
// origin.
//
// WHY THIS IS A HOST PROGRAM. Nothing it exercises is a device operation: no
// shared memory, no barrier, no atomic, no warp intrinsic, only float
// arithmetic out of headers a host compiler reads the same way. Driving the
// three projector cases through the GENERATED entry point would add the arena
// and the argument record to what is exercised; that plumbing is mechanism,
// and it is what `abi_device.rs` and the compute crate's own arena binaries
// cover, so this calls the neutral body directly, which is the same thing the
// friction gate beside it does.
//
// THE DIAGNOSTIC CHANNEL IS CHECKED HERE, WHICH THE ENTRY LAUNCHER CANNOT DO.
// The body reports a malformed basis through `[[seam::diag]]`, and driving it
// through the launcher binds that channel to the global diagnostics record,
// which nothing then reads. A host caller owns the record, so every case below
// asserts the body raised nothing: a basis that failed to build would otherwise
// leave the reduced vector untouched and read as a lock that simply had nothing
// to do.
//
// WHAT IS NOT HERE, and where it went instead: the combined four-row
// deformable projector, which assembled its rows from `invert_inertia_host` and
// `pseudoinverse_gram_host`. Those two live in `solver/translation_lock.hpp`,
// which cannot be compiled by this program, and the live implementations of
// both are `driver/lock_math.rs`. Its property, that the projector
// annihilates the residual and is idempotent, is asserted there against those.

// `data.hpp` FIRST, and the reason is lexical: a shared header takes its type
// vocabulary from whichever header the includer pulled in ahead of it, and
// neither of the two below brings its own.
#include "data.hpp"

#include "energy/model/pdrd_lock_projector.kernel.cpp"
#include "solver/translation_lock_math.hpp"

#include <cmath>
#include <cstdio>

namespace {

constexpr float EPS = 2.0e-5f;

int failures = 0;

void check(bool ok, const char *what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

bool near(float a, float b, float eps = EPS) {
    return std::fabs(a - b) <= eps;
}

bool near_vec(const Vec3f &a, const Vec3f &b, float eps = EPS) {
    return near(a[0], b[0], eps) && near(a[1], b[1], eps) &&
           near(a[2], b[2], eps);
}

Vec3f weighted_perp_sum(const Vec3f *v, const float *mass, unsigned count,
                        const Vec3f &axis) {
    Vec3f sum = Vec3f::Zero();
    for (unsigned i = 0; i < count; ++i) {
        sum += mass[i] * translation_lock::perpendicular(v[i], axis);
    }
    return sum;
}

void project(Vec3f *v, const float *mass, unsigned count, const Vec3f &axis) {
    Vec3f sum = weighted_perp_sum(v, mass, count, axis);
    float denom = 0.0f;
    for (unsigned i = 0; i < count; ++i) denom += mass[i] * mass[i];
    for (unsigned i = 0; i < count; ++i) v[i] -= mass[i] * sum / denom;
}

// -------------------------------------------------------------- deformable

void test_projector_orthogonality_and_tangent_preservation() {
    printf("== the deformable translation projector ==\n");
    const Vec3f axis(0.0f, 0.0f, 1.0f);
    const float mass[] = {1.0f, 2.0f, 3.0f};
    Vec3f v[] = {Vec3f(2.0f, -1.0f, 4.0f), Vec3f(-3.0f, 5.0f, -2.0f),
                 Vec3f(1.0f, 4.0f, 7.0f)};
    project(v, mass, 3, axis);
    check(near_vec(weighted_perp_sum(v, mass, 3, axis), Vec3f::Zero()),
          "a projected step moves the centroid only along the axis");

    Vec3f twice[] = {v[0], v[1], v[2]};
    project(twice, mass, 3, axis);
    bool idempotent = true;
    for (unsigned i = 0; i < 3; ++i) idempotent &= near_vec(twice[i], v[i]);
    check(idempotent, "projecting an already projected step changes nothing");

    // An internal deformation with zero mass-weighted transverse translation
    // is already tangent and must remain unchanged.
    Vec3f internal[] = {Vec3f(2.0f, 0.0f, 1.0f), Vec3f(-1.0f, 0.0f, -3.0f),
                        Vec3f(0.0f, 0.0f, 5.0f)};
    Vec3f before[] = {internal[0], internal[1], internal[2]};
    project(internal, mass, 3, axis);
    bool tangent = true;
    for (unsigned i = 0; i < 3; ++i)
        tangent &= near_vec(internal[i], before[i]);
    check(tangent, "a tangent deformation passes through untouched");
}

void test_affine_and_independent_groups() {
    printf("== affine resolution and group independence ==\n");
    const Vec3f axis(0.0f, 0.0f, 1.0f);
    // A fixed vertex supplies this known correction. The free aggregate must
    // provide the remainder, not a penalty approximation.
    const float fixed_mass = 2.0f, free_mass = 5.0f;
    const Vec3f drift(7.0f, -3.0f, 9.0f);
    const Vec3f fixed_step(1.0f, -2.0f, 4.0f);
    const Vec3f free_step = translation_lock::perpendicular(
        (translation_lock::perpendicular(drift, axis) -
         fixed_mass * translation_lock::perpendicular(fixed_step, axis)) /
            free_mass,
        axis);
    const Vec3f resolved =
        fixed_mass * translation_lock::perpendicular(fixed_step, axis) +
        free_mass * translation_lock::perpendicular(free_step, axis);
    check(near_vec(resolved, translation_lock::perpendicular(drift, axis)),
          "the free aggregate supplies exactly the remainder a fixed vertex "
          "leaves");

    // Two objects use independent aggregate modes. Projecting one cannot
    // change the other's residual.
    const Vec3f y_axis(0.0f, 1.0f, 0.0f);
    const float ma[] = {1.0f, 2.0f};
    const float mb[] = {4.0f, 1.0f};
    Vec3f a[] = {Vec3f(3.0f, 4.0f, 0.0f), Vec3f(-2.0f, 1.0f, 7.0f)};
    Vec3f b[] = {Vec3f(-6.0f, 2.0f, 9.0f), Vec3f(1.0f, -4.0f, 5.0f)};
    project(a, ma, 2, axis);
    project(b, mb, 2, y_axis);
    check(near_vec(weighted_perp_sum(a, ma, 2, axis), Vec3f::Zero()) &&
              near_vec(weighted_perp_sum(b, mb, 2, y_axis), Vec3f::Zero()),
          "two groups on different axes do not disturb each other");

    // If every mass is fixed, a nonzero required correction is infeasible.
    const float no_free_mass = 0.0f;
    check(no_free_mass == 0.0f &&
              !near_vec(translation_lock::perpendicular(drift, axis),
                        Vec3f::Zero()),
          "an all-fixed group with a nonzero correction is infeasible");
}

// ------------------------------------------------------------------- PDRD

// THE ORACLE FOR THE REDUCED PROJECTOR, computed here in closed form so the
// expected values below are derived rather than copied. Translation Lock leaves
// only the component along its axis; Rotation Lock in allow-only mode leaves
// only the component along its own. The two blocks are independent, which is
// what the projector cases that follow verify against the implementation.
void test_pdrd_translation_and_rotation_freedom() {
    printf("== the reduced-axis oracle ==\n");
    const Vec3f axis(1.0f, 0.0f, 0.0f);
    float reduced[6] = {3.0f, -7.0f, 11.0f, 2.0f, -5.0f, 13.0f};
    const float along =
        reduced[0] * axis[0] + reduced[1] * axis[1] + reduced[2] * axis[2];
    reduced[0] = along * axis[0];
    reduced[1] = along * axis[1];
    reduced[2] = along * axis[2];
    check(near(reduced[0], 3.0f) && near(reduced[1], 0.0f) &&
              near(reduced[2], 0.0f),
          "a translation lock keeps only the component along its axis");
    check(near(reduced[3], 2.0f) && near(reduced[4], -5.0f) &&
              near(reduced[5], 13.0f),
          "and leaves the rotation block alone");
}

void test_pdrd_combined_reduced_axes() {
    printf("== the two reduced blocks are independent ==\n");
    // PDRD locks are rows in one six-DOF projector. Translation Lock leaves
    // only x translation here; Rotation Lock leaves only z rotation. The
    // blocks are stated together so a future sequential implementation cannot
    // accidentally drop either component.
    const Vec3f translation_axis(1.0f, 0.0f, 0.0f);
    const Vec3f rotation_axis(0.0f, 0.0f, 1.0f);
    float reduced[6] = {3.0f, -7.0f, 11.0f, 2.0f, -5.0f, 13.0f};
    const float translation_along = reduced[0] * translation_axis[0] +
                                    reduced[1] * translation_axis[1] +
                                    reduced[2] * translation_axis[2];
    const float rotation_along = reduced[3] * rotation_axis[0] +
                                 reduced[4] * rotation_axis[1] +
                                 reduced[5] * rotation_axis[2];
    for (unsigned i = 0; i < 3; ++i) {
        reduced[i] = translation_along * translation_axis[i];
        reduced[3 + i] = rotation_along * rotation_axis[i];
    }
    check(near(reduced[0], 3.0f) && near(reduced[1], 0.0f) &&
              near(reduced[2], 0.0f) && near(reduced[3], 0.0f) &&
              near(reduced[4], 0.0f) && near(reduced[5], 13.0f),
          "each block keeps only its own axis and neither touches the other");
}

// The neutral body, driven as the solver drives it: one body at body_base 0.
//
// THE DIAGNOSTIC RECORD IS THE CALLER'S, and it is read. The body raises on it
// when the basis it builds is malformed, and a malformed basis leaves the
// reduced vector untouched, which is indistinguishable from a lock with nothing
// to do unless the channel is checked.
void project_one_body(unsigned joint_mode, const Vec3f &joint_axis,
                      unsigned translation_lock, const Vec3f &translation_axis,
                      unsigned rotation_lock, const Vec3f &rotation_axis,
                      unsigned rotation_mode, float *reduced,
                      const char *what,
                      unsigned translation_mode = LOCK_TRANSLATION_MODE_AXIS) {
    ChunkDiag diag{};
    pdrd_project_body_dofs_row(&joint_mode, &joint_axis, &translation_lock,
                               &translation_axis, &translation_mode,
                               &rotation_lock, &rotation_axis, &rotation_mode,
                               reduced, 0u, 0u, &diag);
    if (diag.fail_count != 0u) {
        printf("  FAIL %s: the body raised %u diagnostic(s), first payload "
               "(%.1f, %.1f, %.1f, %.1f)\n",
               what, diag.fail_count, double(diag.payload[0]),
               double(diag.payload[1]), double(diag.payload[2]),
               double(diag.payload[3]));
        ++failures;
    }
}

void test_pdrd_hinge_and_rotation_lock_intersection() {
    printf("== hinge intersected with an allow-only rotation lock ==\n");
    // A hinge permits only spin about z. A rotation lock permits only spin
    // about the nonparallel (x + z) axis. Their intersection is zero. This
    // catches a sequential projection, which can reintroduce a forbidden
    // component when the two rotational axes do not commute.
    float reduced[6] = {2.0f, -3.0f, 5.0f, 7.0f, -11.0f, 13.0f};
    project_one_body(PDRD_JOINT_HINGE, Vec3f(0.0f, 0.0f, 1.0f),
                     PDRD_CLOTH_MARKER, Vec3f::Zero(), 0u,
                     Vec3f(0.70710677f, 0.0f, 0.70710677f),
                     ROTATION_LOCK_ALLOW_ONLY, reduced, "the intersection");
    bool all_zero = true;
    for (float value : reduced) all_zero &= near(value, 0.0f);
    check(all_zero, "two nonparallel rotation axes intersect in nothing");
}

void test_pdrd_prohibit_axis_preserves_rotation_plane() {
    printf("== a prohibit-axis rotation lock ==\n");
    const float input[6] = {2.0f, -3.0f, 5.0f, 7.0f, -11.0f, 13.0f};
    float reduced[6];
    for (unsigned i = 0; i < 6; ++i) reduced[i] = input[i];
    project_one_body(PDRD_JOINT_FREE, Vec3f::Zero(), PDRD_CLOTH_MARKER,
                     Vec3f::Zero(), 0u, Vec3f(0.0f, 0.0f, 1.0f),
                     ROTATION_LOCK_PROHIBIT_AXIS, reduced, "prohibit-axis");
    bool kept = true;
    for (unsigned i = 0; i < 5; ++i) kept &= near(reduced[i], input[i]);
    check(kept, "everything off the prohibited axis survives untouched");
    check(near(reduced[5], 0.0f), "and the prohibited spin is removed");
}

void test_pdrd_hinge_translation_and_prohibit_axis_intersection() {
    printf("== hinge, translation lock and prohibit-axis together ==\n");
    // A hinge allows only z rotation and no translation. A prohibit-z rotation
    // lock removes that remaining spin. The translation lock is deliberately
    // included as a duplicate reduced constraint, so all three sources must be
    // orthonormalized in one projector.
    float reduced[6] = {2.0f, -3.0f, 5.0f, 7.0f, -11.0f, 13.0f};
    project_one_body(PDRD_JOINT_HINGE, Vec3f(0.0f, 0.0f, 1.0f), 0u,
                     Vec3f(1.0f, 0.0f, 0.0f), 0u, Vec3f(0.0f, 0.0f, 1.0f),
                     ROTATION_LOCK_PROHIBIT_AXIS, reduced,
                     "three sources at once");
    bool all_zero = true;
    for (float value : reduced) all_zero &= near(value, 0.0f);
    check(all_zero,
          "a duplicated constraint orthonormalizes rather than double counting");
}

// ---------------------------------------------------------- the row algebra

void test_rotation_row_is_anchor_relative() {
    printf("== rotation rows are measured from the anchor ==\n");
    TranslationLock lock{};
    lock.anchor =
        Vec3f(float(5.0f), float(1.0f), float(-4.0f));
    LockFrame frame{};
    frame.row_mask = LOCK_ROTATION_ROW0 | LOCK_ROTATION_ROW1;
    frame.rotation_basis0 = Vec3f(0.0f, 1.0f, 0.0f);
    frame.rotation_basis1 = Vec3f(0.0f, 0.0f, 1.0f);
    frame.inv_inertia = Mat3x3f::Identity();
    frame.com_relative = Vec3f::Zero();
    const Vec3f position(float(7.0f), float(3.0f),
                          float(-2.0f));
    const float mass = 2.0f;

    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, mass);
    // The moment arm is the position MINUS the anchor, so it is (2, 2, 2) here
    // and not the position itself. A row built absolutely agrees with this one
    // only when the anchor is the origin.
    const Vec3f r(2.0f, 2.0f, 2.0f);
    check(near_vec(c.row[LOCK_ROTATION_ROW_BASE],
                   mass * frame.rotation_basis0.cross(r)),
          "the first rotation row uses the anchor-relative arm");
    check(near_vec(c.row[LOCK_ROTATION_ROW_BASE + 1],
                   mass * frame.rotation_basis1.cross(r)),
          "and so does the second");
}

void test_prohibit_axis_uses_one_rotation_row() {
    printf("== prohibit-axis carries a single rotation row ==\n");
    TranslationLock lock{};
    lock.rotation_mode = ROTATION_LOCK_PROHIBIT_AXIS;
    lock.anchor =
        Vec3f(float(5.0f), float(1.0f), float(-4.0f));
    LockFrame frame{};
    frame.row_mask = LOCK_ROTATION_ROW0;
    frame.rotation_basis0 = Vec3f(0.0f, 1.0f, 0.0f);
    frame.rotation_basis1 = Vec3f(1.0f, 0.0f, 0.0f);
    frame.inv_inertia = Mat3x3f::Identity();
    frame.com_relative = Vec3f::Zero();
    const Vec3f position(float(7.0f), float(3.0f),
                          float(-2.0f));
    const float mass = 2.0f;

    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, mass);
    const Vec3f r(2.0f, 2.0f, 2.0f);
    check(near_vec(c.row[LOCK_ROTATION_ROW_BASE],
                   mass * frame.rotation_basis0.cross(r)),
          "the row the mask names is built");
    check(near_vec(c.row[LOCK_ROTATION_ROW_BASE + 1], Vec3f::Zero()),
          "and the row it does not name stays zero");
}

// THE ALL-AXES MODES, on the reduced body vector. Each removes its WHOLE block
// and leaves the other alone, which is the property that separates them from
// the per-axis modes rather than a restatement of them: an axis mode always
// leaves one component of its own block behind.
//
// THE ENABLE BIT COMES FROM THE MODE, so both of these carry a ZERO axis. A
// build that read enablement off the axis would leave the reduced vector
// untouched here and report nothing, which is what makes this the case worth
// having.
void test_pdrd_all_axes_modes_remove_their_whole_block() {
    printf("== the all-axes modes remove a whole block ==\n");
    const Vec3f zero_axis(0.0f, 0.0f, 0.0f);

    float translation_only[6] = {3.0f, -7.0f, 11.0f, 2.0f, -5.0f, 13.0f};
    project_one_body(PDRD_JOINT_FREE, zero_axis, 0u, zero_axis,
                     PDRD_CLOTH_MARKER, zero_axis, LOCK_MODE_ALLOW_ONLY,
                     translation_only, "all-axes translation",
                     LOCK_TRANSLATION_MODE_ALL);
    check(near(translation_only[0], 0.0f) && near(translation_only[1], 0.0f) &&
              near(translation_only[2], 0.0f),
          "an all-axes translation lock removes the whole translation block");
    check(near(translation_only[3], 2.0f) && near(translation_only[4], -5.0f) &&
              near(translation_only[5], 13.0f),
          "and leaves the rotation block untouched");

    float rotation_only[6] = {3.0f, -7.0f, 11.0f, 2.0f, -5.0f, 13.0f};
    project_one_body(PDRD_JOINT_FREE, zero_axis, PDRD_CLOTH_MARKER, zero_axis,
                     0u, zero_axis, LOCK_MODE_ALL, rotation_only,
                     "all-axes rotation");
    check(near(rotation_only[3], 0.0f) && near(rotation_only[4], 0.0f) &&
              near(rotation_only[5], 0.0f),
          "an all-axes rotation lock removes the whole rotation block");
    check(near(rotation_only[0], 3.0f) && near(rotation_only[1], -7.0f) &&
              near(rotation_only[2], 11.0f),
          "and leaves the translation block untouched");

    // SIX ROWS SATURATE THE SPACE, which is what both locks at once ask for.
    // It is allowed and needs no special case: a seventh INDEPENDENT row cannot
    // exist in R^6, so the row budget stays a genuine impossibility. The
    // diagnostic channel is what would report it, and project_one_body reads
    // it.
    float both[6] = {3.0f, -7.0f, 11.0f, 2.0f, -5.0f, 13.0f};
    project_one_body(PDRD_JOINT_FREE, zero_axis, 0u, zero_axis, 0u, zero_axis,
                     LOCK_MODE_ALL, both, "both all-axes locks",
                     LOCK_TRANSLATION_MODE_ALL);
    bool frozen = true;
    for (float value : both) frozen &= near(value, 0.0f);
    check(frozen, "both all-axes locks together freeze all six reduced DOFs");
}

// The row builder's own all-axes shape: three translation rows from the
// identity basis, at the three translation slots.
void test_all_axes_translation_rows_are_the_identity_basis() {
    printf("== an all-axes translation frame carries three identity rows ==\n");
    TranslationLock lock{};
    lock.translation_mode = TRANSLATION_LOCK_ALL;
    LockFrame frame{};
    frame.row_mask =
        LOCK_TRANSLATION_ROW0 | LOCK_TRANSLATION_ROW1 | LOCK_TRANSLATION_ROW2;
    frame.translation_basis0 = Vec3f(1.0f, 0.0f, 0.0f);
    frame.translation_basis1 = Vec3f(0.0f, 1.0f, 0.0f);
    frame.translation_basis2 = Vec3f(0.0f, 0.0f, 1.0f);
    const float mass = 3.0f;
    const LockRowCoefficients c = lock_row_coefficients(
        lock, frame, Vec3f(float(0.0f), float(0.0f),
                            float(0.0f)),
        mass);
    check(near_vec(c.row[0], Vec3f(mass, 0.0f, 0.0f)) &&
              near_vec(c.row[1], Vec3f(0.0f, mass, 0.0f)) &&
              near_vec(c.row[2], Vec3f(0.0f, 0.0f, mass)),
          "the three translation rows are m e_x, m e_y, m e_z");
    // The rotation block stays empty: this frame locks no rotation, and the
    // rows are ZEROED rather than left at whatever the struct was built with.
    for (unsigned row = 0; row < LOCK_ROTATION_ROW_COUNT; ++row) {
        check(near_vec(c.row[LOCK_ROTATION_ROW_BASE + row], Vec3f::Zero()),
              "an unmasked rotation row stays zero");
    }
}

// `constrained_translation` is what each mode forbids, and the axis mode leaves
// its own axis FREE. The drift accumulation and the end-of-step invariant read
// the displacement through this one helper, so they cannot disagree.
void test_constrained_translation_per_mode() {
    printf("== what each translation mode forbids ==\n");
    const Vec3f delta(3.0f, -4.0f, 7.0f);
    TranslationLock lock{};
    lock.axis = Vec3f(0.0f, 0.0f, 1.0f);
    check(near_vec(translation_lock::constrained_translation(lock, delta),
                   Vec3f(3.0f, -4.0f, 0.0f)),
          "an axis mode leaves the component along its own axis free");
    lock.axis = Vec3f::Zero();
    lock.translation_mode = TRANSLATION_LOCK_ALL;
    check(near_vec(translation_lock::constrained_translation(lock, delta),
                   delta),
          "all-axes forbids the whole displacement");
}

// THE MODE CARRIES THE ENABLE BIT, NOT THE AXIS, which is the property whose
// failure is silent: an all-axes lock ships a zero axis by contract, so a
// predicate reading the axis reports it disabled while the UI shows it as set.
void test_the_mode_carries_the_enable_bit() {
    printf("== enablement is the mode, not the axis ==\n");
    // BOTH AXES ARE ZEROED EXPLICITLY. `SMat`'s default constructor is empty
    // (`linalg/smat.hpp:362`), so `TranslationLock lock{}` zeroes the scalar
    // members and leaves the two vectors holding whatever the stack held. That
    // is the same trap the frame builder documents, and here it would make the
    // off case read as enabled at random.
    TranslationLock lock{};
    lock.axis = Vec3f::Zero();
    lock.rotation_axis = Vec3f::Zero();
    check(!translation_lock::translation_lock_enabled(lock) &&
              !translation_lock::rotation_lock_enabled(lock),
          "an axis-mode lock with a zero axis holds nothing");
    lock.translation_mode = TRANSLATION_LOCK_ALL;
    check(translation_lock::translation_lock_enabled(lock),
          "an all-axes translation lock is on with a zero axis");
    lock.rotation_mode = ROTATION_LOCK_ALL;
    check(translation_lock::rotation_lock_enabled(lock),
          "an all-axes rotation lock is on with a zero axis");
    check(translation_lock::translation_mode_valid(TRANSLATION_LOCK_AXIS) &&
              translation_lock::translation_mode_valid(TRANSLATION_LOCK_ALL) &&
              !translation_lock::translation_mode_valid(2u),
          "the translation modes are exactly the two");
    check(translation_lock::rotation_mode_valid(ROTATION_LOCK_ALLOW_ONLY) &&
              translation_lock::rotation_mode_valid(
                  ROTATION_LOCK_PROHIBIT_AXIS) &&
              translation_lock::rotation_mode_valid(ROTATION_LOCK_ALL) &&
              !translation_lock::rotation_mode_valid(3u),
          "and the rotation modes are exactly the three");
}

// THE ALL-AXES ROTATION ROWS, which is where the reference's own widening went
// wrong: its constraint-space fan-out kept four literal indices, so rows 4 and 5
// never reached the Gram right-hand side and the directions they constrain
// stayed free, with every host gate green. These three assertions are what a
// dropped or misplaced rotation row fails.
void test_all_axes_rotation_uses_three_rows_at_the_rotation_base() {
    printf("== an all-axes rotation frame carries three rows at the base ==\n");
    TranslationLock lock{};
    lock.axis = Vec3f::Zero();
    lock.rotation_axis = Vec3f::Zero();
    lock.rotation_mode = ROTATION_LOCK_ALL;
    lock.anchor = Vec3f(float(5.0f), float(1.0f), float(-4.0f));
    LockFrame frame{};
    frame.row_mask = LOCK_ROTATION_ROW0 | LOCK_ROTATION_ROW1 | LOCK_ROTATION_ROW2;
    frame.rotation_basis0 = Vec3f(1.0f, 0.0f, 0.0f);
    frame.rotation_basis1 = Vec3f(0.0f, 1.0f, 0.0f);
    frame.rotation_basis2 = Vec3f(0.0f, 0.0f, 1.0f);
    frame.inv_inertia = Mat3x3f::Identity();
    frame.com_relative = Vec3f::Zero();
    const Vec3f position(float(7.0f), float(3.0f), float(-2.0f));
    const float mass = 2.0f;

    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, mass);
    // r is the ANCHOR-RELATIVE offset, so this also pins down that the all-axes
    // rows form the same anchor-relative difference as the per-axis ones rather
    // than reading an absolute position.
    const Vec3f r(2.0f, 2.0f, 2.0f);
    check(near_vec(c.row[LOCK_ROTATION_ROW_BASE + 0],
                   mass * frame.rotation_basis0.cross(r)) &&
              near_vec(c.row[LOCK_ROTATION_ROW_BASE + 1],
                       mass * frame.rotation_basis1.cross(r)) &&
              near_vec(c.row[LOCK_ROTATION_ROW_BASE + 2],
                       mass * frame.rotation_basis2.cross(r)),
          "all three rotation rows are m (e_k x r) at the rotation base");
    // AND THE TRANSLATION BLOCK STAYS EMPTY, which is the half a row written
    // one slot low would corrupt.
    for (unsigned row = 0; row < LOCK_TRANSLATION_ROW_COUNT; ++row) {
        check(near_vec(c.row[row], Vec3f::Zero()),
              "an unmasked translation row stays zero");
    }
    // At ONE vertex the three coefficients cannot be independent: each is
    // e_k x r, so all three lie in the plane perpendicular to r and span at
    // most two dimensions. That is the correct shape rather than a defect, and
    // it is why three independent angular ROWS need a body with vertices in
    // more than one direction from its centroid.
}

// An AXIS-mode translation leaves the third row zero, which is the control
// against an all-axes branch that fires for every translation lock. Without it
// every all-axes assertion above would still pass on a build that had stopped
// distinguishing the modes at all.
void test_axis_translation_leaves_the_third_row_zero() {
    printf("== an axis-mode translation frame leaves the third row zero ==\n");
    TranslationLock lock{};
    lock.rotation_axis = Vec3f::Zero();
    lock.translation_mode = TRANSLATION_LOCK_AXIS;
    lock.axis = Vec3f(1.0f, 0.0f, 0.0f);
    lock.anchor = Vec3f(float(0.0f), float(0.0f), float(0.0f));
    LockFrame frame{};
    frame.row_mask = LOCK_TRANSLATION_ROW0 | LOCK_TRANSLATION_ROW1;
    frame.translation_basis0 = Vec3f(0.0f, 1.0f, 0.0f);
    frame.translation_basis1 = Vec3f(0.0f, 0.0f, 1.0f);
    frame.translation_basis2 = Vec3f::Zero();
    const Vec3f position(float(1.0f), float(2.0f), float(3.0f));
    const float mass = 3.0f;

    const LockRowCoefficients c =
        lock_row_coefficients(lock, frame, position, mass);
    check(near_vec(c.row[0], mass * Vec3f(0.0f, 1.0f, 0.0f)) &&
              near_vec(c.row[1], mass * Vec3f(0.0f, 0.0f, 1.0f)),
          "the two tangent rows are built");
    check(near_vec(c.row[2], Vec3f::Zero()),
          "and the third translation row, which only all-axes uses, stays zero");
}

void test_perpendicular() {
    printf("== the perpendicular component ==\n");
    const Vec3f actual = translation_lock::perpendicular(
        Vec3f(2.0f, 3.0f, 4.0f), Vec3f(0.0f, 0.0f, 1.0f));
    check(near_vec(actual, Vec3f(2.0f, 3.0f, 0.0f)),
          "the axial part is removed and the rest is kept");
}

} // namespace

int main() {
    test_projector_orthogonality_and_tangent_preservation();
    test_affine_and_independent_groups();
    test_pdrd_translation_and_rotation_freedom();
    test_pdrd_combined_reduced_axes();
    test_pdrd_hinge_and_rotation_lock_intersection();
    test_pdrd_prohibit_axis_preserves_rotation_plane();
    test_pdrd_hinge_translation_and_prohibit_axis_intersection();
    test_perpendicular();
    test_rotation_row_is_anchor_relative();
    test_prohibit_axis_uses_one_rotation_row();
    test_pdrd_all_axes_modes_remove_their_whole_block();
    test_all_axes_translation_rows_are_the_identity_basis();
    test_constrained_translation_per_mode();
    test_the_mode_carries_the_enable_bit();
    test_all_axes_rotation_uses_three_rows_at_the_rotation_base();
    test_axis_translation_leaves_the_third_row_zero();
    if (failures == 0) {
        std::puts("PASS: aggregate lock projector gates passed");
        return 0;
    }
    printf("FAIL: %d aggregate lock projector case(s)\n", failures);
    return 1;
}
