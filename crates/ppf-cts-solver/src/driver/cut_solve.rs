// File: crates/ppf-cts-solver/src/driver/cut_solve.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Linear solves for Newton directions the line search is about to cut.
//!
//! # What this is for
//!
//! When the line search stops a Newton step at a contact, the step takes only
//! a part of its direction and the next iteration solves again from where it
//! stopped. In a scene where bodies meet many new contacts in one step this
//! happens tens of times a step, and each of those directions was solved to
//! `cg-tol` only to have most of it thrown away. Measured on a pile of four
//! hundred thousand vertices, the conjugate gradient iterate holds 99% of the
//! direction's energy after about ninety iterations and runs to four hundred
//! and fifty to meet the residual rule, and 97% of those directions are then
//! cut to a few percent of their length.
//!
//! So a solve the driver EXPECTS to be cut may stop early, on a test that is a
//! bound rather than an estimate, and it starts from the part of the last
//! direction the line search left untaken.
//!
//! # The test
//!
//! The Newton matrix is the inertia term, `mass / dt^2` on the diagonal, plus
//! terms that are each positive semidefinite (elasticity after its
//! projection, the contact barrier, friction, the strain limiter). For an
//! iterate with residual `r` the part of the model's decrease it has not
//! captured is `r . A^-1 r`, and with the inertia term alone standing in for
//! `A` that is at most `dt^2 r . M^-1 r`: one scaled dot product of the
//! residual. The solve stops when that bound is at most
//! [`MISSED_ENERGY_RATIO`] times the decrease the iterate holds, which is to
//! say when it provably holds ten elevenths of what the direction can give.
//! The bound is tight exactly for the modes conjugate gradients resolves last
//! (a body's rigid motion, where inertia is the whole matrix), so it cannot be
//! met while those are unresolved. An estimate from how fast the energy is
//! still growing can: it stops on a plateau, and a pile's collective modes are
//! such a plateau.
//!
//! A row with no inertia term of its own needs its own account, and each is
//! read off what the assembly writes there. An exact pin's row IS the identity
//! (the Dirichlet pass sets it), so the identity bounds it exactly. A pull pin
//! replaces the inertia term by its own weight, which bounds that row the same
//! way. A free vertex with no mass has no bound at all, and a scene with one
//! is never stopped early ([`Staging::bounded`]).
//!
//! # What keeps the result what it was
//!
//! A direction stopped early is never taken whole: the step takes at most
//! [`HELD_BACK_FRACTION`] of it and the Newton loop does not end on it. Every
//! direction a step applies in full, and the trailing iteration, is solved to
//! `cg-tol` from the seed exactly as before. For the cut steps themselves the
//! loss is bounded by the test alone, whatever the solve started from: a step
//! of `t` along a direction that misses at most `ratio` times the energy it
//! holds shrinks the model's error by at least `1 - t (1 - s)`, with
//! `s = sqrt(ratio / (1 + ratio))`, where the exact direction shrinks it by
//! `1 - t`. At a ratio of 0.1 that is `1 - 0.70 t` in the worst case. For a
//! solve from zero the loss is exact rather than bounded: the squared error
//! falls by `t (2 - t)` times the energy the iterate holds.
//!
//! # When it is armed
//!
//! Two conditions, both about cost and neither about correctness:
//!
//! - the last [`CUTS_TO_EXPECT`] line searches were cut (for a step's first
//!   solve, the first line searches of that many steps before it);
//! - the last solve to `cg-tol` took [`SHORTEST_FULL_SOLVE`] iterations or
//!   more. Where a full solve is a few tens of iterations there is nothing to
//!   save, and those are the scenes (stacked cloth, a hanging sheet) whose line
//!   searches are cut by contacts that already have barrier rows, where a
//!   looser direction is cut sooner and costs a Newton iteration. Measured
//!   without this condition: stack 106.5 s to 179.0 s, hang 67.3 s to 83.3 s.
//!

use crate::data::{PullPair, VertexProp};

/// The most energy an early-stopped solve may still be missing, as a fraction
/// of the energy it holds. At 0.1 the iterate provably holds ten elevenths of
/// the direction's energy; measured, it holds 99.8% or more when the bound
/// first passes.
pub const MISSED_ENERGY_RATIO: f32 = 0.1;

/// The most of an early-stopped direction a step takes. The same fraction of
/// the way the line search parks a step that is cut at a contact.
pub const HELD_BACK_FRACTION: f32 = 0.8;

/// How many cut line searches in a row make the next solve expected to be cut.
const CUTS_TO_EXPECT: u32 = 2;

/// The least iteration count of the last solve to `cg-tol` at which an early
/// stop is worth arming. The first two looks of an armed solve fall at
/// iterations 1 and 9, so below about a hundred iterations it could save at
/// most a few tens of iterations a solve.
const SHORTEST_FULL_SOLVE: u32 = 128;

/// `PPF_SOLVE_TO_TOL=1`: every solve runs to `cg-tol`, as if this module were
/// not there. For measuring one build against itself.
pub fn disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("PPF_SOLVE_TO_TOL").is_some())
}

/// `PPF_VERIFY_EARLY_STOP=1`: a solve that meets the bound runs on to `cg-tol`
/// and the energy it gains after that point is compared with the bound, which
/// is the bound's whole claim. The run ends on the first solve that gains more.
pub fn verify() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("PPF_VERIFY_EARLY_STOP").is_some())
}

/// Whether the next line search is expected to be cut, from the ones before.
#[derive(Debug, Default)]
pub struct Expectation {
    /// Cut line searches in a row within the current step, seeded at a step's
    /// start from `first_cuts_in_a_row`.
    cuts_in_a_row: u32,
    /// Steps in a row whose FIRST line search was cut.
    first_cuts_in_a_row: u32,
    /// The current step has not recorded a line search yet.
    first_search_pending: bool,
    /// The iteration count of the last solve that ran to `cg-tol`.
    last_full_solve: u32,
}

impl Expectation {
    /// A time step begins. Its first solve is expected to be cut when the
    /// first line searches of the last steps were.
    pub fn begin_step(&mut self) {
        self.cuts_in_a_row = if self.first_cuts_in_a_row >= CUTS_TO_EXPECT {
            CUTS_TO_EXPECT
        } else {
            0
        };
        self.first_search_pending = true;
    }

    /// Whether the solve about to run may stop early.
    pub fn expects_cut(&self) -> bool {
        self.cuts_in_a_row >= CUTS_TO_EXPECT && self.last_full_solve >= SHORTEST_FULL_SOLVE
    }

    /// A solve ran to `cg-tol` in this many iterations.
    pub fn record_full_solve(&mut self, iterations: u32) {
        self.last_full_solve = iterations;
    }

    /// A line search of a regular (not trailing) iteration ended, cut or not.
    pub fn record_search(&mut self, cut: bool) {
        self.cuts_in_a_row = if cut { self.cuts_in_a_row.saturating_add(1) } else { 0 };
        if self.first_search_pending {
            self.first_search_pending = false;
            self.first_cuts_in_a_row = if cut {
                self.first_cuts_in_a_row.saturating_add(1)
            } else {
                0
            };
        }
    }
}

/// What the staged bound was built from. A change in any of them restages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StagedFor {
    rows: usize,
    dt_bits: u32,
    pin_generation: u64,
    pull_weights: u64,
}

/// The per-row inverse of the bound's diagonal, as staged on the device.
#[derive(Debug, Default)]
pub struct Staging {
    staged: Option<StagedFor>,
    /// Free rows with no inertia term and no pull weight, as last staged.
    unbounded_rows: usize,
}

impl Staging {
    /// Whether the staged blocks are the ones these inputs give.
    pub fn is_current(&self, rows: usize, dt: f32, pin_generation: u64, pulls: &[PullPair]) -> bool {
        self.staged == Some(staged_for(rows, dt, pin_generation, pulls))
    }

    /// Record that the blocks for these inputs are on the device.
    pub fn set_current(
        &mut self,
        rows: usize,
        dt: f32,
        pin_generation: u64,
        pulls: &[PullPair],
        unbounded_rows: usize,
    ) {
        self.staged = Some(staged_for(rows, dt, pin_generation, pulls));
        self.unbounded_rows = unbounded_rows;
    }

    /// Whether every free row has a bound, so the test is a bound at all.
    pub fn bounded(&self) -> bool {
        self.staged.is_some() && self.unbounded_rows == 0
    }
}

fn staged_for(rows: usize, dt: f32, pin_generation: u64, pulls: &[PullPair]) -> StagedFor {
    // FNV-1a over each pull pin's vertex and weight: a pull weight is scene
    // data that may be keyed over time without the pin set changing.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for pin in pulls {
        for word in [pin.index, pin.weight.to_bits()] {
            hash ^= u64::from(word);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    StagedFor { rows, dt_bits: dt.to_bits(), pin_generation, pull_weights: hash }
}

/// Fill `blocks`, nine floats a row, with the inverse of the part of each
/// row's diagonal block the bound may rely on, and return how many free rows
/// have none. Each case is what the assembly writes on that row:
///
/// - an exact pin (`fix_index > 0`): the identity. The Dirichlet pass makes
///   the row the identity and drops its couplings, so the identity is the
///   whole row.
/// - a pull pin: `1 / weight`. The pin's weight stands where the inertia term
///   would, and the FIRST pin in the list that names the vertex is the one
///   that applies, which is the momentum assembly's own rule.
/// - any other vertex: `dt^2 / mass`. A vertex of a body with a restitution
///   weighs its inertia against a shorter step, which only makes the true
///   term larger than this one.
pub fn inertia_inverse_blocks(
    props: &[VertexProp],
    pulls: &[PullPair],
    dt: f32,
    blocks: &mut [f32],
) -> usize {
    assert_eq!(blocks.len(), 9 * props.len(), "nine floats a row");
    blocks.fill(0.0);
    // First match wins, so later pins naming the same vertex are not inserted.
    let mut pull_weight = std::collections::HashMap::with_capacity(pulls.len());
    for pin in pulls {
        pull_weight.entry(pin.index).or_insert(pin.weight);
    }
    let mut unbounded = 0usize;
    for (row, prop) in props.iter().enumerate() {
        let diagonal = if prop.fix_index > 0 {
            1.0
        } else if let Some(weight) = pull_weight.get(&(row as u32)) {
            *weight
        } else {
            prop.mass / (dt * dt)
        };
        // NaN fails `> 0.0`, so a broken weight or mass counts as no bound.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(diagonal > 0.0) || !diagonal.is_finite() {
            unbounded += 1;
            continue;
        }
        let inverse = 1.0 / diagonal;
        blocks[9 * row] = inverse;
        blocks[9 * row + 4] = inverse;
        blocks[9 * row + 8] = inverse;
    }
    unbounded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prop(mass: f32, fix_index: u32, pull_index: u32) -> VertexProp {
        VertexProp { mass, fix_index, pull_index, ..VertexProp::default() }
    }

    fn pull(index: u32, weight: f32) -> PullPair {
        PullPair { index, weight, ..PullPair::default() }
    }

    #[test]
    fn nothing_is_expected_before_two_cuts_in_a_row_and_a_long_full_solve() {
        let mut e = Expectation::default();
        e.begin_step();
        assert!(!e.expects_cut());
        e.record_full_solve(400);
        e.record_search(true);
        assert!(!e.expects_cut(), "one cut is not two");
        e.record_search(true);
        assert!(e.expects_cut());
        e.record_search(false);
        assert!(!e.expects_cut(), "an uncut search ends the run");
        e.record_search(true);
        e.record_search(true);
        assert!(e.expects_cut());
        e.record_full_solve(40);
        assert!(!e.expects_cut(), "a short full solve leaves nothing to save");
    }

    #[test]
    fn a_steps_first_solve_is_expected_to_be_cut_after_two_steps_that_opened_on_a_cut() {
        let mut e = Expectation::default();
        e.record_full_solve(400);
        e.begin_step();
        e.record_search(true);
        e.record_search(false);
        e.begin_step();
        assert!(!e.expects_cut(), "one step opened on a cut");
        e.record_search(true);
        e.begin_step();
        assert!(e.expects_cut(), "two steps in a row opened on a cut");
        e.record_search(false);
        e.begin_step();
        assert!(!e.expects_cut(), "the last step opened uncut");
    }

    #[test]
    fn only_a_steps_first_search_counts_toward_the_next_steps_opening() {
        let mut e = Expectation::default();
        e.record_full_solve(400);
        for _ in 0..3 {
            e.begin_step();
            e.record_search(false);
            e.record_search(true);
            e.record_search(true);
        }
        e.begin_step();
        assert!(!e.expects_cut());
    }

    #[test]
    fn each_row_gets_the_inverse_of_what_the_assembly_puts_on_its_diagonal() {
        let props = [prop(2.0, 0, 0), prop(1.0, 1, 0), prop(1.0, 0, 0), prop(4.0, 0, 0)];
        let pulls = [pull(2, 8.0)];
        let mut blocks = vec![f32::NAN; 36];
        let unbounded = inertia_inverse_blocks(&props, &pulls, 0.5, &mut blocks);
        assert_eq!(unbounded, 0);
        let diagonal = |row: usize| [blocks[9 * row], blocks[9 * row + 4], blocks[9 * row + 8]];
        assert_eq!(diagonal(0), [0.125; 3], "dt^2 / mass");
        assert_eq!(diagonal(1), [1.0; 3], "an exact pin's row is the identity");
        assert_eq!(diagonal(2), [0.125; 3], "1 / weight");
        assert_eq!(diagonal(3), [0.0625; 3]);
        let off: f32 = (0..36).filter(|i| i % 9 != 0 && i % 9 != 4 && i % 9 != 8).map(|i| blocks[i]).sum();
        assert_eq!(off, 0.0, "the blocks are diagonal");
    }

    #[test]
    fn the_first_pull_pin_that_names_a_vertex_is_the_one_that_bounds_it() {
        // The vertex property's own pin index is not consulted: it names the
        // second pin here, and the momentum assembly applies the first.
        let props = [prop(1.0, 0, 2)];
        let pulls = [pull(0, 4.0), pull(0, 100.0)];
        let mut blocks = vec![0.0f32; 9];
        assert_eq!(inertia_inverse_blocks(&props, &pulls, 0.01, &mut blocks), 0);
        assert_eq!(blocks[0], 0.25);
    }

    #[test]
    fn a_free_row_with_nothing_on_its_diagonal_is_counted() {
        let props = [prop(0.0, 0, 0), prop(1.0, 0, 0), prop(f32::NAN, 0, 0), prop(0.0, 3, 0)];
        let pulls = [pull(1, 0.0)];
        let mut blocks = vec![1.0f32; 36];
        let unbounded = inertia_inverse_blocks(&props, &pulls, 0.01, &mut blocks);
        // No mass, a pull of no weight (which also takes the inertia term
        // away), a mass that is not a number. The massless exact pin is bound
        // by its identity row.
        assert_eq!(unbounded, 3);
        assert!(blocks[..27].iter().all(|v| *v == 0.0));
        assert_eq!([blocks[27], blocks[31], blocks[35]], [1.0; 3]);
    }

    #[test]
    fn the_staging_goes_stale_when_anything_it_was_built_from_changes() {
        let pulls = [pull(3, 2.0)];
        let mut staging = Staging::default();
        assert!(!staging.bounded(), "nothing staged yet");
        staging.set_current(10, 0.01, 7, &pulls, 0);
        assert!(staging.is_current(10, 0.01, 7, &pulls));
        assert!(staging.bounded());
        assert!(!staging.is_current(11, 0.01, 7, &pulls), "rows");
        assert!(!staging.is_current(10, 0.02, 7, &pulls), "dt");
        assert!(!staging.is_current(10, 0.01, 8, &pulls), "the pin set");
        assert!(!staging.is_current(10, 0.01, 7, &[pull(3, 2.5)]), "a pull weight");
        staging.set_current(10, 0.01, 7, &pulls, 2);
        assert!(!staging.bounded(), "two free rows have no bound");
    }
}
