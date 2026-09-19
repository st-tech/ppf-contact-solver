// File: lock_math.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The aggregate lock's HOST arithmetic.
//!
//! `invert_inertia` and `pseudoinverse_gram` have no neutral counterpart in
//! this tree: `src/kernels/solver/translation_lock_math.hpp` declares the
//! `LockFrame` fields `inv_inertia` and `gram_pinv`, and the kernel bodies
//! only read them. Both belong on the host because each runs once per locked
//! GROUP rather than per vertex, on a 3x3 or a 4x4, and each carries a
//! feasibility verdict that aborts the run. Both are computed in DOUBLE
//! precision for that reason: float64 is banned in `src/kernels`, which is
//! device code, while a host-side scalar is not. The driver is the host.
//!
//! THE ORDER OF OPERATIONS IS LOAD-BEARING, constants included, because a
//! factorization decides a rank verdict from magnitudes it accumulates. Where
//! a line here looks like it could be shortened, the long form is what fixes
//! the rounding, so shortening it is a numerical change and has to be argued
//! as one rather than taken as tidying.

use crate::data::{
    Mat3x3f, TranslationLock, Vec3f, ROTATION_LOCK_ALL, ROTATION_LOCK_ALLOW_ONLY,
    ROTATION_LOCK_PROHIBIT_AXIS, TRANSLATION_LOCK_ALL, TRANSLATION_LOCK_AXIS,
};
use super::scene::{Fatal, FatalResult};

/// `f32::EPSILON` written out as an `f64` literal, exactly 2^-23. It is spelled
/// out rather than widened from `f32::EPSILON` because every comparison it
/// takes part in below runs in double precision, so the value is visible in the
/// precision it is used at.
const FLOAT_EPS: f64 = 1.1920928955078125e-7;

/// The widest constraint set a group can carry: three translation rows plus
/// three angular ones. `LOCK_MAX_ROWS` in
/// `solver/translation_lock_math.hpp`.
pub const MAX_ROWS: usize = 6;

/// Where the rotation block starts, which is also how many translation rows
/// there are. `LOCK_ROTATION_ROW_BASE` in the same header.
///
/// BIT INDEX EQUALS ROW INDEX in the frame's `row_mask`, and the two blocks are
/// contiguous, matching the PDRD reduced layout: translation at offset 0,
/// rotation at offset 3. Every consumer here and on the device relies on it, so
/// it is spelled once and imported rather than written out per file.
pub const ROTATION_ROW_BASE: usize = 3;
pub const ROTATION_ROW_COUNT: usize = 3;

pub const LOCK_TRANSLATION_ROW0: u32 = 1 << 0;
pub const LOCK_TRANSLATION_ROW1: u32 = 1 << 1;
pub const LOCK_TRANSLATION_ROW2: u32 = 1 << 2;
pub const LOCK_ROTATION_ROW0: u32 = 1 << (ROTATION_ROW_BASE as u32);
pub const LOCK_ROTATION_ROW1: u32 = 1 << (ROTATION_ROW_BASE as u32 + 1);
pub const LOCK_ROTATION_ROW2: u32 = 1 << (ROTATION_ROW_BASE as u32 + 2);

/// A 6x6 of floats in the layout `Mat6x6f` has: COLUMN-major, thirty-six
/// floats.
///
/// A local type rather than an `na::Matrix6<f32>` because it crosses to the
/// device as thirty-six floats and is read back the same way, and because every
/// operation it needs is written out below. `get`, `set` and `add_assign_at`
/// are the only places the column-major offset `MAX_ROWS * col + row` is
/// spelled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat6x6f(pub [f32; MAX_ROWS * MAX_ROWS]);

// BY HAND, because `derive(Default)` stops at a 32-element array and this one
// holds 36. The value is the same zero the derive would have produced.
impl Default for Mat6x6f {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Mat6x6f {
    pub const ZERO: Self = Self([0.0; MAX_ROWS * MAX_ROWS]);

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.0[MAX_ROWS * col + row]
    }

    /// No PRODUCTION caller: a live step takes a group's Gram from the device
    /// as thirty-six flat floats through `gram_of` in `driver/lock.rs`, so
    /// nothing there writes an entry at a time. What reads this are the tests
    /// in this file that build a Gram by hand and hand it to
    /// `pseudoinverse_gram`: `a_full_rank_gram_inverts_exactly`,
    /// `a_rank_deficient_gram_projects_onto_the_rows_it_has`,
    /// `a_rotation_row_survives_beside_a_much_larger_translation_row`, and the
    /// two projector tests
    /// `the_combined_projector_annihilates_its_residual_and_is_idempotent` and
    /// `six_saturating_rows_still_annihilate_and_stay_idempotent`, which form
    /// the Gram of their own row set entry by entry.
    #[allow(dead_code)]
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.0[MAX_ROWS * col + row] = value;
    }

    #[inline]
    pub fn add_assign_at(&mut self, row: usize, col: usize, value: f32) {
        self.0[MAX_ROWS * col + row] += value;
    }
}

/// Six floats, the layout `Vec6f` has.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec6f(pub [f32; MAX_ROWS]);

impl Vec6f {
    pub const ZERO: Self = Self([0.0; MAX_ROWS]);

    #[inline]
    pub fn norm(&self) -> f32 {
        self.0.iter().fold(0.0f32, |acc, v| acc + v * v).sqrt()
    }
}

/// `m * v` for the 6x6 rows, which is `lock_matvec6` in
/// `solver/translation_lock_math.hpp`.
#[inline]
pub fn matvec6(m: &Mat6x6f, v: &Vec6f) -> Vec6f {
    let mut out = [0.0f32; MAX_ROWS];
    for row in 0..MAX_ROWS {
        let mut sum = 0.0f32;
        for col in 0..MAX_ROWS {
            sum += m.get(row, col) * v.0[col];
        }
        out[row] = sum;
    }
    Vec6f(out)
}

/// Whether a group locks translation at all, which is
/// `translation_lock_enabled` in `solver/translation_lock_math.hpp`.
///
/// THE MODE CARRIES THE ENABLE BIT, NOT THE AXIS. An all-axes lock ships a zero
/// axis by contract, so an axis test would read it as disabled while the UI
/// still reported it as set, and nothing downstream would produce a wrong
/// number to notice. Within an axis mode the axis test is exact rather than a
/// tolerance, because the value is authored and a group that locks a direction
/// writes a unit vector there.
#[inline]
pub fn translation_lock_enabled(lock: &TranslationLock) -> bool {
    lock.translation_mode == TRANSLATION_LOCK_ALL
        || lock.axis[0] != 0.0
        || lock.axis[1] != 0.0
        || lock.axis[2] != 0.0
}

/// Whether a group locks rotation at all. Same rule as above.
#[inline]
pub fn rotation_lock_enabled(lock: &TranslationLock) -> bool {
    lock.rotation_mode == ROTATION_LOCK_ALL
        || lock.rotation_axis[0] != 0.0
        || lock.rotation_axis[1] != 0.0
        || lock.rotation_axis[2] != 0.0
}

/// The part of a center-of-mass displacement the translation lock forbids.
///
/// An axis mode leaves the component ALONG its axis free, so only the
/// perpendicular part is constrained; an all-axes mode constrains the whole
/// vector. The drift accumulation and the end-of-step invariant both read the
/// displacement through this, on the device as well, so the two cannot disagree
/// about what is locked.
///
/// No PRODUCTION caller for this host copy: both dispatches reach the shared
/// body `translation_lock::constrained_translation` in
/// `kernels/solver/translation_lock_math.hpp`, the drift accumulation in
/// `translation_lock_rows.kernel.cpp` and the end-of-step invariant in
/// `translation_lock_check.kernel.cpp`. What reads this is the test
/// `an_axis_mode_leaves_its_own_axis_free_and_all_axes_leaves_nothing` in this
/// file, which checks that the axis mode leaves the component along its own
/// axis free and that the all-axes mode forbids the whole vector.
#[allow(dead_code)]
#[inline]
pub fn constrained_translation(lock: &TranslationLock, delta: &Vec3f) -> Vec3f {
    if lock.translation_mode == TRANSLATION_LOCK_ALL {
        *delta
    } else {
        let axis = &lock.axis;
        let along = delta[0] * axis[0] + delta[1] * axis[1] + delta[2] * axis[2];
        Vec3f::new(
            delta[0] - along * axis[0],
            delta[1] - along * axis[1],
            delta[2] - along * axis[2],
        )
    }
}

#[inline]
pub fn translation_mode_valid(mode: u32) -> bool {
    mode == TRANSLATION_LOCK_AXIS || mode == TRANSLATION_LOCK_ALL
}

#[inline]
pub fn rotation_mode_valid(mode: u32) -> bool {
    mode == ROTATION_LOCK_ALLOW_ONLY
        || mode == ROTATION_LOCK_PROHIBIT_AXIS
        || mode == ROTATION_LOCK_ALL
}

/// The inverse of a group's inertia about its own centroid.
///
/// Cholesky in double precision, with the rank floor and the three abort
/// conditions below. A group whose inertia is singular, or whose
/// smallest direction is below what float32 can resolve, ABORTS: a lock is an
/// exact constraint on the Newton direction, so an inverse that is merely large
/// would silently turn it into a very stiff spring.
///
/// `dmap_index` names the group in the message, which is the number an author
/// can act on.
pub fn invert_inertia(input: &Mat3x3f, dmap_index: u32) -> FatalResult<Mat3x3f> {
    let mut a = [[0.0f64; 3]; 3];
    let mut scale = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            // SYMMETRIZED ON THE WAY IN. The accumulation is by atomics over the
            // group's members, so the two off-diagonal halves are summed
            // independently and differ in their last bits; averaging them is
            // what makes the Cholesky below see an exactly symmetric matrix.
            let value = 0.5 * (input[(i, j)] as f64 + input[(j, i)] as f64);
            if !value.is_finite() {
                return Err(geometry_fault(dmap_index, "the inertia reduction is non-finite"));
            }
            a[i][j] = value;
            scale = scale.max(value.abs());
        }
    }
    if !(scale > 0.0) || !scale.is_finite() {
        return Err(geometry_fault(dmap_index, "the inertia is singular"));
    }

    let rank_floor = 128.0 * FLOAT_EPS * scale;
    let mut l = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(sum > rank_floor) || !sum.is_finite() {
                    return Err(geometry_fault(
                        dmap_index,
                        "the inertia is singular or below float32 rank resolution",
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let mut inverse = Mat3x3f::zeros();
    for col in 0..3 {
        let mut y = [0.0f64; 3];
        for i in 0..3 {
            let mut sum = if i == col { 1.0 } else { 0.0 };
            for k in 0..i {
                sum -= l[i][k] * y[k];
            }
            y[i] = sum / l[i][i];
        }
        let mut x = [0.0f64; 3];
        for i in (0..3).rev() {
            let mut sum = y[i];
            for k in (i + 1)..3 {
                sum -= l[k][i] * x[k];
            }
            x[i] = sum / l[i][i];
        }
        for row in 0..3 {
            if !x[row].is_finite() || x[row].abs() > f32::MAX as f64 {
                return Err(geometry_fault(dmap_index, "the inverse inertia is non-finite"));
            }
            inverse[(row, col)] = x[row] as f32;
        }
    }
    Ok(inverse)
}

/// The Jacobi pseudoinverse of a group's 4x4 symmetric positive-semidefinite
/// Gram matrix.
///
/// It returns the exact orthogonal projector for the RESOLVED row rank, the
/// compatible all-pinned case (rank zero) included, which is why it is a
/// pseudoinverse rather than an inverse: a group whose rows are linearly
/// dependent, or whose every member is pinned, has fewer independent
/// constraints than rows and the missing directions must contribute nothing
/// rather than blow up.
pub fn pseudoinverse_gram(input: &Mat6x6f) -> FatalResult<Mat6x6f> {
    let mut raw = [[0.0f64; MAX_ROWS]; MAX_ROWS];
    let mut a = [[0.0f64; MAX_ROWS]; MAX_ROWS];
    let mut v = [[0.0f64; MAX_ROWS]; MAX_ROWS];
    let mut row_scale = [0.0f64; MAX_ROWS];
    let mut scale = 0.0f64;
    for i in 0..MAX_ROWS {
        v[i][i] = 1.0;
        for j in 0..MAX_ROWS {
            let value = 0.5 * (input.get(i, j) as f64 + input.get(j, i) as f64);
            if !value.is_finite() {
                return Err(Fatal::invariant(
                    "solver driver: aggregate lock Gram reduction is non-finite",
                ));
            }
            raw[i][j] = value;
        }
        row_scale[i] = if raw[i][i] > 0.0 { 1.0 / raw[i][i].sqrt() } else { 0.0 };
    }
    // SYMMETRIC ROW EQUILIBRATION prevents a physically valid rotation row from
    // being classified as rank zero merely because a translation row uses a
    // different dimensional scale. With `D = diag(row_scale)`, the eigensystem
    // of `D G D` is solved and its pseudoinverse mapped back as
    // `G^+ = D (D G D)^+ D`.
    for i in 0..MAX_ROWS {
        for j in 0..MAX_ROWS {
            a[i][j] = row_scale[i] * raw[i][j] * row_scale[j];
            scale = scale.max(a[i][j].abs());
        }
    }
    if scale == 0.0 {
        return Ok(Mat6x6f::ZERO);
    }

    let off_floor = 1.0e-14 * scale;
    // ONE JACOBI ROTATION PER ITERATION, always on the largest remaining
    // off-diagonal. A 6x6 has fifteen off-diagonal pairs, so this budget is
    // eight full cycles. `converged` is checked below rather than assumed:
    // exhausting the budget would otherwise return a partially diagonalized
    // matrix as though it were an eigendecomposition.
    let mut converged = false;
    for _sweep in 0..120 {
        let (mut p, mut q) = (0usize, 1usize);
        let mut largest = 0.0f64;
        for i in 0..MAX_ROWS {
            for j in (i + 1)..MAX_ROWS {
                let value = a[i][j].abs();
                if value > largest {
                    largest = value;
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= off_floor {
            converged = true;
            break;
        }
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        let tau = (aqq - app) / (2.0 * apq);
        let t = (if tau >= 0.0 { 1.0 } else { -1.0 }) / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..MAX_ROWS {
            if k == p || k == q {
                continue;
            }
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - s * akq;
            a[p][k] = a[k][p];
            a[k][q] = s * akp + c * akq;
            a[q][k] = a[k][q];
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..MAX_ROWS {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }

    if !converged {
        return Err(Fatal::invariant(
            "solver driver: aggregate lock Gram eigensolve did not converge within its \
             Jacobi budget",
        ));
    }
    let mut largest = 0.0f64;
    for i in 0..MAX_ROWS {
        if !a[i][i].is_finite() {
            return Err(Fatal::invariant(
                "solver driver: aggregate lock Gram eigensolve failed",
            ));
        }
        largest = largest.max(a[i][i].abs());
    }
    let rank_floor = 256.0 * FLOAT_EPS * largest;
    let mut inverse = Mat6x6f::ZERO;
    for k in 0..MAX_ROWS {
        if !(a[k][k] > rank_floor) {
            continue;
        }
        let inv = 1.0 / a[k][k];
        for i in 0..MAX_ROWS {
            for j in 0..MAX_ROWS {
                inverse.add_assign_at(
                    i,
                    j,
                    (row_scale[i] * v[i][k] * inv * v[j][k] * row_scale[j]) as f32,
                );
            }
        }
    }
    Ok(inverse)
}

fn geometry_fault(dmap_index: u32, what: &str) -> Fatal {
    Fatal::invariant(format!(
        "solver driver: locked group {dmap_index} has invalid geometry: {what}. A lock is an \
         exact constraint on the Newton direction, so this cannot be softened into a stiff \
         spring; the scene has to be authored so the group has a resolvable extent"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number of MEMBERS these tests build a group out of, which is not
    /// the number of ROWS. Four unequal masses is enough to make a Gram of
    /// full rank; the row count is `MAX_ROWS` and the two were spelled with
    /// the same literal before the all-axes modes widened one of them.
    const MEMBERS: usize = 4;

    /// THE COMBINED PROJECTOR ANNIHILATES ITS RESIDUAL AND IS IDEMPOTENT, over
    /// a real group rather than a synthetic Gram matrix.
    ///
    /// The tests above each check one routine against a closed form: a diagonal
    /// inertia, an identity Gram, a rank-one Gram. This one assembles what the
    /// solver assembles. Four members of unequal mass, two translation rows from
    /// a tangent basis and two rotation rows built through the inverted inertia,
    /// then the Gram of those rows, its pseudoinverse, and the correction that
    /// pseudoinverse drives. THE ROTATION ROWS SIT AT `ROTATION_ROW_BASE`, not
    /// at 2 and 3, so this also pins the row layout the device masks index
    /// into: the two blocks are contiguous and the rotation block starts at 3. The two properties are what make it a PROJECTOR
    /// rather than a correction that merely reduces the residual: `R x` is zero
    /// afterwards, and projecting twice changes nothing.
    ///
    /// IT IS HERE RATHER THAN IN C++ BECAUSE THE IMPLEMENTATION IS. It was the
    /// combined-projector case of a CUDA regression binary, which assembled its
    /// rows from `invert_inertia_host` and `pseudoinverse_gram_host` in
    /// `src/kernels/solver/translation_lock.hpp`. That header was the deleted
    /// orchestrator's, and the live transcriptions of both routines are the two
    /// functions this module exports.
    #[test]
    fn the_combined_projector_annihilates_its_residual_and_is_idempotent() {
        let position = [
            Vec3f::new(0.0, 0.0, 0.0),
            Vec3f::new(2.0, 0.0, 0.0),
            Vec3f::new(0.0, 1.0, 1.0),
            Vec3f::new(-1.0, 2.0, 0.5),
        ];
        let mass = [1.0f32, 2.0, 3.0, 5.0];

        let total: f32 = mass.iter().sum();
        let mut com = Vec3f::zeros();
        for i in 0..MEMBERS {
            com += mass[i] * position[i];
        }
        com /= total;

        let mut inertia = Mat3x3f::zeros();
        for i in 0..MEMBERS {
            let r = position[i] - com;
            let r2 = r.dot(&r);
            for row in 0..3 {
                for col in 0..3 {
                    inertia[(row, col)] += mass[i]
                        * ((if row == col { r2 } else { 0.0 }) - r[row] * r[col]);
                }
            }
        }
        let inverse = invert_inertia(&inertia, 17).expect("the inertia inverts");

        // The same tangent basis `lock::tangent_basis` builds, including the
        // 0.9 threshold, for a translation lock on z and a rotation lock on x.
        let tangent = |axis: Vec3f| -> (Vec3f, Vec3f) {
            let reference = if axis[2].abs() < 0.9 {
                Vec3f::new(0.0, 0.0, 1.0)
            } else {
                Vec3f::new(1.0, 0.0, 0.0)
            };
            let b0 = axis.cross(&reference).normalize();
            let b1 = axis.cross(&b0).normalize();
            (b0, b1)
        };
        let (tb0, tb1) = tangent(Vec3f::new(0.0, 0.0, 1.0));
        let (rb0, rb1) = tangent(Vec3f::new(1.0, 0.0, 0.0));

        // `MAX_ROWS` rows, each a Vec3f per member. An axis-mode group leaves
        // the third translation row and the third angular row empty, which is
        // exactly what an unset bit in `row_mask` means on the device.
        let mut row = [[Vec3f::zeros(); MEMBERS]; MAX_ROWS];
        for i in 0..MEMBERS {
            let r = position[i] - com;
            row[0][i] = mass[i] * tb0;
            row[1][i] = mass[i] * tb1;
            row[ROTATION_ROW_BASE][i] = mass[i] * (inverse * rb0).cross(&r);
            row[ROTATION_ROW_BASE + 1][i] = mass[i] * (inverse * rb1).cross(&r);
        }

        let rows_times = |value: &[Vec3f; MEMBERS]| -> Vec6f {
            let mut out = Vec6f([0.0; MAX_ROWS]);
            for k in 0..MAX_ROWS {
                for i in 0..MEMBERS {
                    out.0[k] += row[k][i].dot(&value[i]);
                }
            }
            out
        };

        let mut gram = Mat6x6f::ZERO;
        for a in 0..MAX_ROWS {
            for b in 0..MAX_ROWS {
                let mut sum = 0.0;
                for i in 0..MEMBERS {
                    sum += row[a][i].dot(&row[b][i]);
                }
                gram.set(a, b, sum);
            }
        }
        let pinv = pseudoinverse_gram(&gram).expect("the Gram of four rows reduces");

        let project = |value: &mut [Vec3f; MEMBERS]| {
            let lambda = matvec6(&pinv, &rows_times(value));
            for i in 0..MEMBERS {
                for k in 0..MAX_ROWS {
                    value[i] -= lambda.0[k] * row[k][i];
                }
            }
        };

        let mut value = [
            Vec3f::new(2.0, -1.0, 4.0),
            Vec3f::new(-3.0, 5.0, -2.0),
            Vec3f::new(1.0, 4.0, 7.0),
            Vec3f::new(-2.0, 1.0, 3.0),
        ];
        project(&mut value);
        let residual = rows_times(&value);
        for k in 0..MAX_ROWS {
            assert!(
                residual.0[k].abs() < 1.0e-4,
                "row {k} is not annihilated: {}",
                residual.0[k]
            );
        }

        let mut twice = value;
        project(&mut twice);
        for i in 0..MEMBERS {
            for c in 0..3 {
                assert!(
                    (twice[i][c] - value[i][c]).abs() < 1.0e-4,
                    "member {i} component {c} moved on the second projection: \
                     {} then {}",
                    value[i][c],
                    twice[i][c]
                );
            }
        }
    }

    /// The inverse of a diagonal inertia is the reciprocal diagonal.
    ///
    /// A case whose answer is known in closed form, which is what makes it a
    /// check on the Cholesky rather than a restatement of it.
    #[test]
    fn a_diagonal_inertia_inverts_to_its_reciprocals() {
        let mut inertia = Mat3x3f::zeros();
        inertia[(0, 0)] = 4.0;
        inertia[(1, 1)] = 2.0;
        inertia[(2, 2)] = 0.5;
        let inverse = invert_inertia(&inertia, 7).expect("a diagonal inertia inverts");
        for (k, expected) in [0.25f32, 0.5, 2.0].iter().enumerate() {
            assert!(
                (inverse[(k, k)] - expected).abs() < 1.0e-6,
                "diagonal {k}: expected {expected}, got {}",
                inverse[(k, k)]
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(inverse[(i, j)].abs() < 1.0e-6, "off-diagonal ({i},{j}) is not zero");
                }
            }
        }
    }

    /// A SINGULAR inertia aborts and names the group.
    ///
    /// This is the behavior the feature depends on: a lock cannot be applied to
    /// a group with no resolvable extent, and returning a huge inverse instead
    /// would turn the exact constraint into a spring nobody asked for.
    #[test]
    fn a_singular_inertia_aborts_naming_the_group() {
        // Two of the three directions have extent; the third is exactly zero,
        // which is a group whose members are coplanar.
        let mut inertia = Mat3x3f::zeros();
        inertia[(0, 0)] = 1.0;
        inertia[(1, 1)] = 1.0;
        let fault = invert_inertia(&inertia, 42).expect_err("a singular inertia must abort");
        let text = format!("{fault:?}");
        assert!(text.contains("42"), "the message must name the group: {text}");
        assert!(
            text.contains("singular") || text.contains("rank"),
            "the message must say what is wrong: {text}"
        );
    }

    /// The pseudoinverse of an identity Gram is the identity.
    #[test]
    fn a_full_rank_gram_inverts_exactly() {
        let mut gram = Mat6x6f::ZERO;
        for k in 0..MAX_ROWS {
            gram.set(k, k, 1.0);
        }
        let pinv = pseudoinverse_gram(&gram).expect("an identity Gram inverts");
        for i in 0..MAX_ROWS {
            for j in 0..MAX_ROWS {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (pinv.get(i, j) - expected).abs() < 1.0e-5,
                    "({i},{j}): expected {expected}, got {}",
                    pinv.get(i, j)
                );
            }
        }
    }

    /// A RANK-DEFICIENT Gram gives the projector onto the rows it does have,
    /// and contributes nothing in the directions it does not.
    ///
    /// THE ALL-PINNED CASE IS THIS ONE AT RANK ZERO, and it is compatible rather
    /// than an error: a group whose every member is exactly pinned has no free
    /// direction to project, so the pseudoinverse is the zero matrix and the
    /// correction it drives is zero.
    #[test]
    fn a_rank_deficient_gram_projects_onto_the_rows_it_has() {
        let mut gram = Mat6x6f::ZERO;
        gram.set(0, 0, 2.0);
        let pinv = pseudoinverse_gram(&gram).expect("a rank-one Gram reduces");
        assert!(
            (pinv.get(0, 0) - 0.5).abs() < 1.0e-5,
            "the resolved row inverts to its reciprocal, got {}",
            pinv.get(0, 0)
        );
        for i in 0..MAX_ROWS {
            for j in 0..MAX_ROWS {
                if i == 0 && j == 0 {
                    continue;
                }
                assert!(
                    pinv.get(i, j).abs() < 1.0e-6,
                    "unresolved direction ({i},{j}) must contribute nothing, got {}",
                    pinv.get(i, j)
                );
            }
        }

        // Rank zero: every row absent, which is the all-pinned group.
        let empty = pseudoinverse_gram(&Mat6x6f::ZERO).expect("a zero Gram is compatible");
        assert_eq!(empty, Mat6x6f::ZERO, "a rank-zero Gram gives the zero projector");
    }

    /// Equilibration is what lets a rotation row survive beside a translation
    /// row whose scale is orders larger.
    ///
    /// Without the `D G D` step the small row is below the rank floor of the
    /// large one and is dropped, which silently stops locking the rotation while
    /// the translation still holds.
    #[test]
    fn a_rotation_row_survives_beside_a_much_larger_translation_row() {
        let mut gram = Mat6x6f::ZERO;
        gram.set(0, 0, 1.0e6);
        gram.set(1, 1, 1.0e-3);
        let pinv = pseudoinverse_gram(&gram).expect("a mixed-scale Gram reduces");
        assert!(
            (pinv.get(0, 0) - 1.0e-6).abs() < 1.0e-9,
            "the large row inverts to its reciprocal, got {}",
            pinv.get(0, 0)
        );
        assert!(
            (pinv.get(1, 1) - 1.0e3).abs() < 1.0,
            "the small row must NOT be dropped as rank-deficient, got {}",
            pinv.get(1, 1)
        );
    }

    /// The identity block factors to the identity, to within the shift.
    ///
    /// The closed-form case: with `factor^T factor` the block's inverse, an
    /// identity block must give back something whose square is the identity.
    /// The `1e-8` relative shift the routine adds is why this is a tolerance
    /// rather than an equality, and it is that shift which makes a
    /// semi-definite block usable rather than a NaN.
    #[test]
    fn an_identity_block_factors_to_the_identity() {
        let mut block = [0.0f32; 36];
        for k in 0..6 {
            block[k * 6 + k] = 1.0;
        }
        let factor = factor_reduced_block(&block);
        for row in 0..6 {
            for column in 0..6 {
                let mut sum = 0.0f32;
                for k in 0..6 {
                    sum += factor[k * 6 + row] * factor[k * 6 + column];
                }
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1.0e-4,
                    "({row},{column}) of factor^T factor is {sum}, not {expected}"
                );
            }
        }
    }

    /// A DIAGONAL block inverts to its reciprocals, which is what the
    /// equilibration has to preserve.
    ///
    /// The scaling is applied symmetrically and undone in the last loop, so a
    /// block whose entries span orders of magnitude must still inverse-square
    /// to the right diagonal. Dropping the `scale[c]` factor on the way out
    /// leaves a factor for the EQUILIBRATED block rather than the real one,
    /// which is a preconditioner wrong by exactly the scaling: it changes the
    /// iteration count and never the answer, so nothing else would report it.
    #[test]
    fn a_diagonal_block_inverts_to_its_reciprocals() {
        let diagonal = [4.0f32, 1.0, 0.25, 100.0, 0.01, 9.0];
        let mut block = [0.0f32; 36];
        for k in 0..6 {
            block[k * 6 + k] = diagonal[k];
        }
        let factor = factor_reduced_block(&block);
        for k in 0..6 {
            let mut sum = 0.0f32;
            for r in 0..6 {
                sum += factor[r * 6 + k] * factor[r * 6 + k];
            }
            let expected = 1.0 / diagonal[k];
            assert!(
                (sum - expected).abs() < 1.0e-3 * expected.max(1.0e-3),
                "diagonal {k}: factor^T factor gives {sum}, not {expected}"
            );
        }
    }

    /// A SEMI-DEFINITE block yields a finite factor rather than a NaN.
    ///
    /// THIS IS THE GUARD THE ROUTINE EXISTS FOR. A body whose reduced block has
    /// a direction with no stiffness, an empty body among them, would otherwise
    /// divide by a zero pivot. The diagonal floor and the relative shift keep
    /// the factor finite, so the preconditioner stays usable rather than
    /// poisoning the solve with a NaN that spreads on its first application.
    #[test]
    fn a_semi_definite_block_stays_finite() {
        let mut block = [0.0f32; 36];
        for k in 0..3 {
            block[k * 6 + k] = 1.0;
        }
        let factor = factor_reduced_block(&block);
        for (k, value) in factor.iter().enumerate() {
            assert!(
                value.is_finite(),
                "entry {k} of a semi-definite block's factor is {value}"
            );
        }
    }

    /// THE MODE CARRIES THE ENABLE BIT, NOT THE AXIS.
    ///
    /// This is the property the all-axes modes turn on, and getting it wrong is
    /// SILENT: an all-axes lock ships a zero axis by contract, so a predicate
    /// that tested the axis would report it disabled while the UI still showed
    /// it as set, and no wrong number would appear anywhere to notice. Within
    /// an axis mode the axis test is still exact rather than a tolerance,
    /// because the value is authored rather than computed.
    #[test]
    fn the_mode_carries_the_enable_bit_not_the_axis() {
        let mut lock = TranslationLock::default();
        assert!(!translation_lock_enabled(&lock), "a default lock holds nothing");
        assert!(!rotation_lock_enabled(&lock));

        // Axis mode: the axis is what says on or off.
        lock.axis = Vec3f::new(0.0, 1.0, 0.0);
        assert!(translation_lock_enabled(&lock));
        // A tiny but non-zero axis IS enabled: nothing here rounds an authored
        // direction away.
        lock.axis = Vec3f::new(0.0, 0.0, 1.0e-30);
        assert!(translation_lock_enabled(&lock));

        // All-axes mode: enabled with an axis of exactly zero, which is the
        // case an axis test reads backwards.
        lock.axis = Vec3f::zeros();
        lock.translation_mode = TRANSLATION_LOCK_ALL;
        assert!(
            translation_lock_enabled(&lock),
            "an all-axes translation lock is on with a zero axis"
        );
        lock.rotation_axis = Vec3f::zeros();
        lock.rotation_mode = ROTATION_LOCK_ALL;
        assert!(
            rotation_lock_enabled(&lock),
            "an all-axes rotation lock is on with a zero axis"
        );

        assert!(translation_mode_valid(TRANSLATION_LOCK_AXIS));
        assert!(translation_mode_valid(TRANSLATION_LOCK_ALL));
        assert!(!translation_mode_valid(2));
        assert!(rotation_mode_valid(ROTATION_LOCK_ALLOW_ONLY));
        assert!(rotation_mode_valid(ROTATION_LOCK_PROHIBIT_AXIS));
        assert!(rotation_mode_valid(ROTATION_LOCK_ALL));
        assert!(!rotation_mode_valid(3));
    }

    /// `constrained_translation` is what each mode forbids, and the axis mode
    /// leaves its own axis FREE.
    ///
    /// Both the drift accumulation and the end-of-step invariant read the
    /// displacement through this one helper, so a disagreement between them is
    /// not expressible. Getting the axis mode backwards would constrain the one
    /// direction the group was supposed to keep.
    #[test]
    fn an_axis_mode_leaves_its_own_axis_free_and_all_axes_leaves_nothing() {
        let mut lock = TranslationLock::default();
        lock.axis = Vec3f::new(0.0, 0.0, 1.0);
        let delta = Vec3f::new(3.0, -4.0, 7.0);

        let held = constrained_translation(&lock, &delta);
        assert!(
            held[2].abs() < 1.0e-6,
            "the component ALONG the axis is free, got {}",
            held[2]
        );
        assert!((held[0] - 3.0).abs() < 1.0e-6 && (held[1] + 4.0).abs() < 1.0e-6);

        lock.axis = Vec3f::zeros();
        lock.translation_mode = TRANSLATION_LOCK_ALL;
        let held = constrained_translation(&lock, &delta);
        for c in 0..3 {
            assert!(
                (held[c] - delta[c]).abs() < 1.0e-6,
                "all-axes forbids the whole vector; component {c} was {} not {}",
                held[c],
                delta[c]
            );
        }
    }

    /// SIX ROWS SATURATE THE SPACE AND THE PROJECTOR STILL RESOLVES, which is
    /// what a group asking for both all-axes locks at once builds.
    ///
    /// The widening from four rows to six is only useful if the Gram of six
    /// independent rows still reduces: the Jacobi budget has to be enough for a
    /// 6x6 (fifteen off-diagonal pairs rather than six), and the sweep now
    /// reports non-convergence rather than returning a half-diagonalized matrix
    /// as though it were an eigendecomposition. Three translation rows from the
    /// identity basis, three angular rows through the inverted inertia.
    #[test]
    fn six_saturating_rows_still_annihilate_and_stay_idempotent() {
        let position = [
            Vec3f::new(0.0, 0.0, 0.0),
            Vec3f::new(2.0, 0.0, 0.0),
            Vec3f::new(0.0, 1.0, 1.0),
            Vec3f::new(-1.0, 2.0, 0.5),
        ];
        let mass = [1.0f32, 2.0, 3.0, 5.0];
        let total: f32 = mass.iter().sum();
        let mut com = Vec3f::zeros();
        for i in 0..MEMBERS {
            com += mass[i] * position[i];
        }
        com /= total;

        let mut inertia = Mat3x3f::zeros();
        for i in 0..MEMBERS {
            let r = position[i] - com;
            let r2 = r.dot(&r);
            for row in 0..3 {
                for col in 0..3 {
                    inertia[(row, col)] +=
                        mass[i] * ((if row == col { r2 } else { 0.0 }) - r[row] * r[col]);
                }
            }
        }
        let inverse = invert_inertia(&inertia, 23).expect("the inertia inverts");

        // The identity basis both all-axes modes use.
        let basis = [
            Vec3f::new(1.0, 0.0, 0.0),
            Vec3f::new(0.0, 1.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
        ];
        let mut row = [[Vec3f::zeros(); MEMBERS]; MAX_ROWS];
        for i in 0..MEMBERS {
            let r = position[i] - com;
            for k in 0..3 {
                row[k][i] = mass[i] * basis[k];
                row[ROTATION_ROW_BASE + k][i] = mass[i] * (inverse * basis[k]).cross(&r);
            }
        }

        let rows_times = |value: &[Vec3f; MEMBERS]| -> Vec6f {
            let mut out = Vec6f([0.0; MAX_ROWS]);
            for k in 0..MAX_ROWS {
                for i in 0..MEMBERS {
                    out.0[k] += row[k][i].dot(&value[i]);
                }
            }
            out
        };

        let mut gram = Mat6x6f::ZERO;
        for a in 0..MAX_ROWS {
            for b in 0..MAX_ROWS {
                let mut sum = 0.0;
                for i in 0..MEMBERS {
                    sum += row[a][i].dot(&row[b][i]);
                }
                gram.set(a, b, sum);
            }
        }
        let pinv = pseudoinverse_gram(&gram).expect("the Gram of six rows reduces");

        let project = |value: &mut [Vec3f; MEMBERS]| {
            let lambda = matvec6(&pinv, &rows_times(value));
            for i in 0..MEMBERS {
                for k in 0..MAX_ROWS {
                    value[i] -= lambda.0[k] * row[k][i];
                }
            }
        };

        let mut value = [
            Vec3f::new(2.0, -1.0, 4.0),
            Vec3f::new(-3.0, 5.0, -2.0),
            Vec3f::new(1.0, 4.0, 7.0),
            Vec3f::new(-2.0, 1.0, 3.0),
        ];
        project(&mut value);
        let residual = rows_times(&value);
        for k in 0..MAX_ROWS {
            assert!(
                residual.0[k].abs() < 1.0e-3,
                "row {k} of six is not annihilated: {}",
                residual.0[k]
            );
        }

        let mut twice = value;
        project(&mut twice);
        for i in 0..MEMBERS {
            for c in 0..3 {
                assert!(
                    (twice[i][c] - value[i][c]).abs() < 1.0e-4,
                    "member {i} component {c} moved on the second projection: {} then {}",
                    value[i][c],
                    twice[i][c]
                );
            }
        }
    }
}

/// Equilibrated Cholesky of a PDRD body's 6x6 reduced block.
///
/// Returns the lower-triangular inverse factor such that the block's inverse is
/// `factor^T factor`. Both are 36 floats, ROW major, which is the layout the
/// preconditioner's rows read.
///
/// A transcription of `pdrd_factor_reduced_block` in
/// `src/kernels/energy/model/pdrd_precond_factor.hpp`, and that header's own
/// reason for existing applies here twice over: a preconditioner difference
/// changes iteration counts and never the answer, so two copies of this
/// arithmetic would drift apart with nothing to report it. This one is a THIRD
/// copy and is written line for line against that file for exactly that reason.
///
/// EVERY GUARD IS LOAD-BEARING AND NONE IS A TOLERANCE. The input is
/// symmetrized, its diagonal floored relative to the trace, the scaling applied
/// symmetrically and a small relative shift added, so a semi-definite or badly
/// scaled block yields a usable positive factor rather than a NaN. An empty
/// body's block is the identity and runs through unchanged.
pub fn factor_reduced_block(block: &[f32; 36]) -> [f32; 36] {
    const N: usize = 6;
    let mut m = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            m[i][j] = block[i * N + j] as f64;
        }
    }
    for i in 0..N {
        for j in (i + 1)..N {
            let mean = 0.5 * (m[i][j] + m[j][i]);
            m[i][j] = mean;
            m[j][i] = mean;
        }
    }
    let mut trace = 0.0f64;
    for i in 0..N {
        trace += m[i][i];
    }
    let diagonal_floor = 1.0e-12 * (trace / N as f64) + 1.0e-30;
    let mut scale = [0.0f64; N];
    for i in 0..N {
        let diagonal = m[i][i];
        scale[i] = 1.0 / (if diagonal > diagonal_floor { diagonal } else { diagonal_floor }).sqrt();
    }
    for i in 0..N {
        for j in 0..N {
            m[i][j] *= scale[i] * scale[j];
        }
    }
    const EPS: f64 = 1.0e-8;
    for i in 0..N {
        m[i][i] += EPS;
    }
    let mut lower = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = m[i][j];
            for k in 0..j {
                sum -= lower[i][k] * lower[j][k];
            }
            if i == j {
                lower[i][i] = (if sum > 1.0e-20 { sum } else { 1.0e-20 }).sqrt();
            } else {
                lower[i][j] = sum / lower[j][j];
            }
        }
    }
    let mut factor = [0.0f32; 36];
    let mut column = [0.0f64; N];
    for c in 0..N {
        column[c] = 1.0 / lower[c][c];
        for i in (c + 1)..N {
            let mut sum = 0.0f64;
            for k in c..i {
                sum += lower[i][k] * column[k];
            }
            column[i] = -sum / lower[i][i];
        }
        for i in c..N {
            factor[i * N + c] = (column[i] * scale[c]) as f32;
        }
    }
    factor
}
