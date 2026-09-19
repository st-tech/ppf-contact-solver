// File: crates/ppf-cts-solver/src/driver/ccd.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The ACCD filter: the time of impact the line search may not step past.
//!
//! # This is what makes the solver penetration-free, and the barrier is not
//!
//! The contact barrier is a CUBIC energy, `-d^3 / (3 * ghat)`, finite at the
//! surface with a finite gradient there. It has no pole, so it bounds nothing:
//! a large enough step walks straight through it. What actually prevents a
//! crossing is that the line search takes no step past the time of impact this
//! module computes, plus `check_intersection` (`intersection.rs`) as the final
//! check that nothing got through anyway.
//!
//! Both are required. A contact scene run with the barrier and only one of them
//! COMPLETES and exits 0 with surfaces passed through each other, which is
//! strictly worse than refusing the scene, and it is why contact was refused
//! whole until the sweep, the scan and the assembly arrived together.
//!
//! # The split, and WHERE THE SWEEP NOW RUNS
//!
//! **The narrow phase is on the DEVICE.** `contact/ccd_sweep.kernel.cpp`
//! declares six entry points, one per (query kind, tree) pair, each walking a
//! BVH with the conservative advance invoked as a per-hit device functor and
//! min-folding one time of impact into the slot its own query owns. That is
//! what `Contact::line_search` dispatches, and no traversal runs on the host.
//!
//! What is left in this module is the HOST half of that:
//!
//! | on the device | here, in Rust |
//! |---|---|
//! | the traversal, the pair filters and all four sweeps (`contact/ccd_sweep.kernel.cpp` over `contact/accd.hpp`) | the reduce over the two per-primitive arrays, and the value types the result is folded into |
//!
//! The fold is a MINIMUM, which is associative and commutative over finite
//! floats, so moving it from a host chunk merge to a device register plus one
//! array reduce cannot move the answer.
//!
//! **The seven per-pair wrappers below are a TEST ORACLE, not the production
//! path.** They call the same `contact/accd.hpp` entry points the device
//! functors call, one pair at a time, and the unit tests in this file are what
//! gate the conservative advance's own behavior: the parking floor, the
//! overlapping-start report, a comoving pair, a head-on approach. Deleting them
//! with the host walk would have retired that battery, which is the failure
//! mode a count-based test already taught this tree (a gate can be left passing
//! while checking nothing). They have no production caller and must not acquire
//! one: a second CCD path is exactly what this change removed.
//!
//! # Two properties of the shared header a caller must not undo
//!
//! **A returned time of exactly zero is not "no advance".** It means the pair
//! began the step already inside the contact offset, which the conservative
//! advance cannot resolve, and the run must end naming the pair. [`Filter`]
//! records which pair it was and what it measured, so the caller can raise
//! rather than quietly take a zero-length step forever.
//!
//! **`ccd_helper` runs in a RESCALED frame**, normalized by each pair's own
//! largest coordinate, so a bare length inside it is a size-dependent bug: its
//! world value is `literal * max_entry / 0.99`. That is why the parking floor
//! is `park_floor(ghat) = 1e-2 * ghat`, anchored to a per-element authored
//! length, and why the four entry points rescale it at the call. Nothing in
//! this module introduces a length of its own, and nothing in it may.


// Two entry points here have no PRODUCTION caller and are kept anyway.
// `park_floor` and `Overlap`'s size are what a test compares against the shared
// header, which is the only way this side can notice the parking floor or the
// record layout moving; `pair_count` is what a caller asks when it wants to know
// the filter saw anything at all.
#![allow(dead_code)]
use super::intersection::PositionTriple;
use crate::data::ParamSet;

/// `accd::OverlapInfo`, the two fields a flagged pair reports.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Overlap {
    /// Squared start distance, in the sweep frame's RESCALED units. Exactly
    /// zero means the pair evaluates as touching to round-off.
    pub d2: f32,
    /// The contact offset in those same units, or `-1.0` when the sweep frame
    /// itself collapsed to a point and there is no such scale.
    pub offset: f32,
    /// Whether the two fields above were written. Nonzero means a real report;
    /// zero means the slot was never claimed, so the two fields above are the
    /// initializer rather than measurements.
    pub flagged: u32,
}

impl Overlap {
    /// The sweep frame collapsed, rather than the pair merely starting inside
    /// the offset. The two are reported differently on purpose: a collapsed
    /// frame has no scale, so its distance carries no units.
    pub fn frame_collapsed(&self) -> bool {
        self.offset < 0.0
    }
}

// NOTHING IN THIS MODULE IS A DISPATCH, WHICH IS WHY NOTHING IN IT REACHES THE
// `Device` TRAIT.
//
// The seam is for a kernel over a thread range: an extent, a guard count and
// buffers the entry point indexes by the thread. Every entry point below takes
// ONE PAIR's primitive indices and returns ONE float, with no thread index, no
// extent and nothing indexed by one, which is the same category as
// `super::pcg`'s per-block inverse and `super::step`'s domain query. Forcing
// one into a dispatch shape would mean inventing an extent to satisfy a symbol
// count, and the answer would not change.
//
// The LOOP over pairs is real and it lives in `super::contact` and
// `super::collider`, which own the pair lists and the filters that decide which
// pairs are swept at all. What would move this module behind the seam is an
// entry point over a PAIR LIST, declared beside the shared sweep with
// `[[seam::args]] [[seam::entry]]` and rendered by
// `ppf-cts-compute/seam/kernelgen.py`. Writing one by hand instead would add a
// hand-written launcher to the stock rule 9 of
// `.github/workflows/scripts/check-shared-wiring.py` exists to ratchet down.
//
// Two of the seven are not even that: `overlap_sizeof_abi` is what a test
// compares this module's `Overlap` mirror against, and `park_floor_abi`
// returns `1e-2 * ghat` for a test to compare against the shared header. Both
// are layout and constant queries.
extern "C" {
    fn overlap_sizeof_abi() -> u32;
    fn park_floor_abi(ghat: f32) -> f32;
    fn park_gap_analytic_abi(clearance0: f32, ghat: f32, eps: f32) -> f32;
    fn park_crossing_analytic_abi(clearance0: f32, clearance1: f32, park: f32) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn point_triangle_ccd_abi(
        x0: *const f32,
        x1: *const f32,
        point: u32,
        t0: u32,
        t1: u32,
        t2: u32,
        offset: f32,
        ghat: f32,
        param: *const ParamSet,
        overlap: *mut Overlap,
    ) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn point_edge_ccd_abi(
        x0: *const f32,
        x1: *const f32,
        point: u32,
        e0: u32,
        e1: u32,
        offset: f32,
        ghat: f32,
        param: *const ParamSet,
        overlap: *mut Overlap,
    ) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn point_point_ccd_abi(
        x0: *const f32,
        x1: *const f32,
        a: u32,
        b: u32,
        offset: f32,
        ghat: f32,
        param: *const ParamSet,
        overlap: *mut Overlap,
    ) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn edge_edge_ccd_abi(
        x0: *const f32,
        x1: *const f32,
        a0: u32,
        a1: u32,
        b0: u32,
        b1: u32,
        offset: f32,
        ghat: f32,
        param: *const ParamSet,
        overlap: *mut Overlap,
    ) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn collision_point_triangle_ccd_abi(
        x0: *const f32,
        x1: *const f32,
        point: u32,
        statics: *const f32,
        t0: u32,
        t1: u32,
        t2: u32,
        offset: f32,
        ghat: f32,
        param: *const ParamSet,
        overlap: *mut Overlap,
    ) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn collision_point_triangle_ccd_static_point_abi(
        statics: *const f32,
        point: u32,
        x0: *const f32,
        x1: *const f32,
        t0: u32,
        t1: u32,
        t2: u32,
        offset: f32,
        ghat: f32,
        param: *const ParamSet,
        overlap: *mut Overlap,
    ) -> f32;
    #[allow(clippy::too_many_arguments)]
    fn collision_edge_edge_ccd_abi(
        x0: *const f32,
        x1: *const f32,
        a0: u32,
        a1: u32,
        statics: *const f32,
        b0: u32,
        b1: u32,
        offset: f32,
        ghat: f32,
        param: *const ParamSet,
        overlap: *mut Overlap,
    ) -> f32;
}

/// The parking floor, `1e-2 * ghat`, through the shared body.
///
/// A world length. The four sweeps rescale it into their own frame; a caller
/// that passes it anywhere itself has almost certainly made the size-dependent
/// mistake the module docs describe.
pub fn park_floor(ghat: f32) -> f32 {
    unsafe { park_floor_abi(ghat) }
}

/// Where an ANALYTIC sweep parks, through the shared body.
///
/// The floor, sphere and pin-ball sweeps each solve for a clearance in closed
/// form and each hand that clearance to a `mass / gap^2`, so this is what keeps
/// the divisor away from zero. Unlike [`park_floor`] it is a world length that
/// callers pass down unrescaled, because those sweeps have no rescaled frame.
pub fn park_gap_analytic(clearance0: f32, ghat: f32, eps: f32) -> f32 {
    unsafe { park_gap_analytic_abi(clearance0, ghat, eps) }
}

/// The fraction of an analytic sweep at which its clearance reaches `park`.
pub fn park_crossing_analytic(clearance0: f32, clearance1: f32, park: f32) -> f32 {
    unsafe { park_crossing_analytic_abi(clearance0, clearance1, park) }
}

/// A pair that began the step already inside the contact offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlappingStart {
    /// Which of the four sweeps reported it, for the message.
    pub kind: &'static str,
    /// The two element indices, in the caller's own index space.
    pub elements: (u32, u32),
    pub overlap: Overlap,
}

impl OverlappingStart {
    pub fn describe(&self) -> String {
        if self.overlap.frame_collapsed() {
            format!(
                "the {} pair ({}, {}) has a sweep frame that collapsed to a \
                 point, so its two primitives are coincident to round-off",
                self.kind, self.elements.0, self.elements.1
            )
        } else {
            format!(
                "the {} pair ({}, {}) began the step already inside its contact \
                 offset (squared distance {} against offset {} in the sweep \
                 frame's units), which the conservative advance cannot resolve",
                self.kind,
                self.elements.0,
                self.elements.1,
                self.overlap.d2,
                self.overlap.offset
            )
        }
    }
}

/// A pair the contact ASSEMBLY found already collapsed to its contact offset.
///
/// Kinds 6 to 9, read out of the same `CcdOverlapRecord` slots the sweep uses
/// but written by `embed_contact` and `embed_collision`
/// (`src/kernels/contact/contact_narrow.kernel.cpp` and
/// `src/kernels/contact/collision_narrow.kernel.cpp`) during the assembly
/// rather than by the sweep, with WORLD-space lengths where the sweep records
/// its rescaled ones.
///
/// THE EXACT WORDING OF THE REPORT IS A CONTRACT rather than a convenience:
/// `rig_coincident_contact_pair` and `rig_collider_coincident_pair` parse the
/// log for the `contact starts overlapping` phrase, for the kind string, for
/// `vertices <a> and <b>`, and for the index-space note, so rewording
/// [`AssemblyOverlap::describe`] or the caller's note turns those scenarios
/// red.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssemblyOverlap {
    /// The kind string [`decode_assembly_overlap`] maps kinds 6 to 9 to.
    pub kind: &'static str,
    /// True for kinds 7 to 9, whose SECOND index is a collision-mesh vertex.
    pub collision_mesh: bool,
    /// The two vertex ids. `collision_mesh` says which space the SECOND is in,
    /// and the caller's note names that space in the report.
    pub elements: (u32, u32),
    pub d2: f32,
    pub offset: f32,
}

impl AssemblyOverlap {
    /// The first report line for an assembly-side overlap, minus the leading
    /// `### ` the caller adds.
    pub fn describe(&self) -> String {
        format!(
            "contact starts overlapping: a {} pair's separation has already \
             collapsed to the contact offset, so the barrier has no direction to act \
             along and the pair carries no force. offending pair: vertices {} and {}, \
             squared separation {} against offset {}, in world units.",
            self.kind,
            self.elements.0,
            self.elements.1,
            // C's `%.6e`, NOT Rust's `{:.6e}`. Rust writes `1.5e-9` where C
            // writes `1.500000e-09`, so the two spellings of the same number do
            // not compare, and the rest of the solver's reports are C-formatted.
            super::log::c_exponential(self.d2 as f64, 6),
            super::log::c_exponential(self.offset as f64, 6)
        )
    }
}

/// Decode one slot as an ASSEMBLY report, or `None` if unwritten or written by
/// the sweep. Kinds 0 to 5 belong to `decode_overlap`; the two decoders are
/// kept apart because the units differ: the sweep's lengths are in its
/// rescaled frame and these are in world units, so one message carrying both
/// would print two incomparable numbers under one name.
pub fn decode_assembly_overlap(words: &[u32]) -> Option<AssemblyOverlap> {
    assert_eq!(words.len(), OVERLAP_WORDS);
    if words[0] == 0 {
        return None;
    }
    let (kind, collision_mesh) = match words[1] {
        6 => ("contact assembly", false),
        7 => ("vertex-face (collision mesh, assembly)", true),
        8 => ("face-vertex (collision mesh, assembly)", true),
        9 => ("edge-edge (collision mesh, assembly)", true),
        _ => return None,
    };
    Some(AssemblyOverlap {
        kind,
        collision_mesh,
        elements: (words[2], words[3]),
        d2: f32::from_bits(words[4]),
        offset: f32::from_bits(words[5]),
    })
}

/// How many 32-bit words one `CcdOverlapRecord` occupies.
///
/// That record is `[[seam::pod(24)]]` in `contact/ccd_sweep.kernel.cpp`: the
/// flag, the sweep name, the two element indices, and the squared distance and
/// offset as floats. The driver allocates the array as `u32` rather than as a
/// Rust mirror of the struct so ONE buffer serves both the record field and the
/// `vec_fill_u32` that clears it before every line search: a fill counts `u32`s,
/// and a handle over a record-typed allocation would carry its length in
/// records.
pub const OVERLAP_WORDS: usize = 6;

/// Decode one slot of the per-query overlap array, or `None` if unwritten.
///
/// A SLOT IS WRITTEN BY EXACTLY ONE THREAD, the query that owns it, so no
/// atomic is involved on either side of this. `flagged` is the record's own
/// field and NOT the returned time: ACCD's probe cap returns `lower_t`, which
/// is zero when the very first advance underflows, and that path writes no
/// report, so a caller reading a zero time as the signal reports a pair that
/// did not begin the step overlapping.
pub fn decode_overlap(words: &[u32]) -> Option<OverlappingStart> {
    assert_eq!(
        words.len(),
        OVERLAP_WORDS,
        "an overlap slot is {OVERLAP_WORDS} words; {} is a different record",
        words.len()
    );
    if words[0] == 0 {
        return None;
    }
    let kind = match words[1] {
        0 => Sweep::PointTriangle,
        1 => Sweep::PointEdge,
        2 => Sweep::PointPoint,
        3 => Sweep::EdgeEdge,
        other => panic!(
            "a CCD sweep recorded an overlapping start under sweep id {other}, \
             which names none of the four sweeps. The kernel and this reader \
             disagree about the record"
        ),
    };
    Some(OverlappingStart {
        kind: kind.name(),
        elements: (words[2], words[3]),
        overlap: Overlap {
            d2: f32::from_bits(words[4]),
            offset: f32::from_bits(words[5]),
            flagged: words[0],
        },
    })
}

/// The four sweeps, named so a report can say which one spoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sweep {
    PointTriangle,
    PointEdge,
    PointPoint,
    EdgeEdge,
}

impl Sweep {
    fn name(self) -> &'static str {
        match self {
            Sweep::PointTriangle => "point-triangle",
            Sweep::PointEdge => "point-edge",
            Sweep::PointPoint => "point-point",
            Sweep::EdgeEdge => "edge-edge",
        }
    }
}

/// One step's CCD filter: the smallest time of impact over every pair, and the
/// first pair that reported an overlapping start.
///
/// # Determinism
///
/// The time is a MIN-fold and needs no ordering. The overlap report does: two
/// pairs can be flagged in the same step, so the one that is REPORTED is
/// defined to be the one offered first, which for an ascending pair walk is the
/// lowest-indexed. The device latches whichever thread arrived first and does
/// not have that property.
#[derive(Debug, Clone)]
pub struct Filter {
    toi: f32,
    overlap: Option<OverlappingStart>,
    pairs: u64,
}

impl Filter {
    /// A filter over a step whose unfiltered length is `max_t`, which is
    /// `ParamSet::line_search_max_t`.
    pub fn new(max_t: f32) -> Self {
        assert!(
            max_t > 0.0 && max_t.is_finite(),
            "the line search's maximum step must be a positive finite number, \
             not {max_t}"
        );
        Filter {
            toi: max_t,
            overlap: None,
            pairs: 0,
        }
    }

    /// The filtered time of impact: the step the line search may take.
    ///
    /// Ask [`Self::overlapping_start`] BEFORE using it. A pair that began
    /// overlapping returns exactly zero, and a caller that reads this as "no
    /// progress, try again" spins forever on a state that cannot resolve.
    pub fn time_of_impact(&self) -> f32 {
        self.toi
    }

    /// The first pair that began the step already inside its contact offset.
    pub fn overlapping_start(&self) -> Option<&OverlappingStart> {
        self.overlap.as_ref()
    }

    /// How many pairs were swept.
    pub fn pair_count(&self) -> u64 {
        self.pairs
    }

    /// Fold in one pair's result.
    fn accept(&mut self, kind: Sweep, elements: (u32, u32), toi: f32, overlap: Overlap) {
        self.pairs += 1;
        // THE RECORD'S OWN FLAG, not the returned time. A zero `toi` does not
        // mean this record was written: ACCD's probe cap returns `lower_t`,
        // which is zero when the first advance underflows, and that path writes
        // nothing here. Reading the time instead reported a structured
        // `OverlappingStart` for a pair that did not begin the step
        // overlapping, naming a d2 and an offset that were the initializer
        // rather than measurements, on a run that ends with `### ccd failed`
        // instead. Checked BEFORE the min-fold, because folding first would
        // leave a zero `toi` with nothing to explain it.
        if overlap.flagged != 0 && self.overlap.is_none() {
            self.overlap = Some(OverlappingStart {
                kind: kind.name(),
                elements,
                overlap,
            });
        }
        // NaN is not a smaller time, it is a broken one, and `f32::min` would
        // discard it and let the step proceed on the other operand.
        assert!(
            !toi.is_nan(),
            "the {} sweep of pair ({}, {}) returned NaN, so no step length can \
             be trusted from this filter",
            kind.name(),
            elements.0,
            elements.1
        );
        if toi < self.toi {
            self.toi = toi;
        }
    }

    /// Fold in a time a DEVICE sweep already reduced over its own primitives.
    ///
    /// The per-pair fold lives in the kernel now: each query min-folds its own
    /// register into its own slot of a per-primitive array, and the host reduces
    /// that array once. `min` is associative and commutative over finite floats,
    /// so moving the fold from a host chunk merge to a device register plus one
    /// array reduce cannot move the answer.
    pub fn fold_time(&mut self, toi: f32) {
        // NaN is not a smaller time, it is a broken one, and `f32::min` would
        // discard it and let the step proceed on the other operand.
        assert!(
            !toi.is_nan(),
            "a device CCD sweep reduced to NaN, so no step length can be \
             trusted from this filter"
        );
        if toi < self.toi {
            self.toi = toi;
        }
    }

    /// Record a pair that began the step inside its offset, if none is recorded.
    ///
    /// THE CALLER OFFERS THEM IN ASCENDING SLOT ORDER, which is what makes two
    /// runs of one scene name the same pair. Latching whichever thread arrived
    /// first would not: the slots are filled concurrently, so the winner would
    /// vary run to run and a rig scenario could not pin the pair.
    pub fn record_overlap(&mut self, start: OverlappingStart) {
        if self.overlap.is_none() {
            self.overlap = Some(start);
        }
    }

}

/// The sweep inputs common to every pair: the two position arrays and the
/// solver parameters.
///
/// `x0` and `x1` are the start and end of the step, as raw position triples.
/// Nothing here converts or rescales a coordinate: the shared body differences
/// the two poses and normalizes the frame itself, so the absolute magnitude of
/// a coordinate never enters the sweep's arithmetic.
pub struct SweepFrame<'a> {
    pub x0: &'a [PositionTriple],
    pub x1: &'a [PositionTriple],
    pub param: &'a ParamSet,
}

impl SweepFrame<'_> {
    fn pointers(&self) -> (*const f32, *const f32, *const ParamSet) {
        (
            self.x0.as_ptr() as *const f32,
            self.x1.as_ptr() as *const f32,
            self.param as *const ParamSet,
        )
    }
}

/// A vertex against a triangle.
pub fn point_triangle(
    frame: &SweepFrame,
    filter: &mut Filter,
    point: u32,
    triangle: [u32; 3],
    element_ids: (u32, u32),
    offset: f32,
    ghat: f32,
) {
    let (x0, x1, param) = frame.pointers();
    let mut overlap = Overlap::default();
    let toi = unsafe {
        point_triangle_ccd_abi(
            x0,
            x1,
            point,
            triangle[0],
            triangle[1],
            triangle[2],
            offset,
            ghat,
            param,
            &mut overlap,
        )
    };
    filter.accept(Sweep::PointTriangle, element_ids, toi, overlap);
}

/// A vertex against an edge.
pub fn point_edge(
    frame: &SweepFrame,
    filter: &mut Filter,
    point: u32,
    edge: [u32; 2],
    element_ids: (u32, u32),
    offset: f32,
    ghat: f32,
) {
    let (x0, x1, param) = frame.pointers();
    let mut overlap = Overlap::default();
    let toi = unsafe {
        point_edge_ccd_abi(
            x0,
            x1,
            point,
            edge[0],
            edge[1],
            offset,
            ghat,
            param,
            &mut overlap,
        )
    };
    filter.accept(Sweep::PointEdge, element_ids, toi, overlap);
}

/// A vertex against a vertex, which is what a SAND grain pair is.
pub fn point_point(
    frame: &SweepFrame,
    filter: &mut Filter,
    a: u32,
    b: u32,
    offset: f32,
    ghat: f32,
) {
    let (x0, x1, param) = frame.pointers();
    let mut overlap = Overlap::default();
    let toi =
        unsafe { point_point_ccd_abi(x0, x1, a, b, offset, ghat, param, &mut overlap) };
    filter.accept(Sweep::PointPoint, (a, b), toi, overlap);
}

/// An edge against an edge.
pub fn edge_edge(
    frame: &SweepFrame,
    filter: &mut Filter,
    a: [u32; 2],
    b: [u32; 2],
    element_ids: (u32, u32),
    offset: f32,
    ghat: f32,
) {
    let (x0, x1, param) = frame.pointers();
    let mut overlap = Overlap::default();
    let toi = unsafe {
        edge_edge_ccd_abi(
            x0, x1, a[0], a[1], b[0], b[1], offset, ghat, param, &mut overlap,
        )
    };
    filter.accept(Sweep::EdgeEdge, element_ids, toi, overlap);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A position triple, in the world units a position component carries.
    fn point(x: f32, y: f32, z: f32) -> PositionTriple {
        [x, y, z]
    }

    /// A `ParamSet` with only the fields the sweeps read set.
    ///
    /// Safety: `ParamSet` is a `repr(C)` aggregate of scalars with no `Drop`,
    /// no references and no niche-optimized fields, so an all-zero bit pattern
    /// is a valid inhabitant.
    fn param(max_t: f32, ccd_eps: f32) -> ParamSet {
        let mut p: ParamSet = unsafe { std::mem::zeroed() };
        p.line_search_max_t = max_t;
        p.ccd_eps = ccd_eps;
        p
    }

    #[test]
    fn the_overlap_mirror_matches_cpp() {
        assert_eq!(
            std::mem::size_of::<Overlap>(),
            unsafe { overlap_sizeof_abi() } as usize
        );
    }

    #[test]
    fn the_parking_floor_is_the_shared_one() {
        // `park_floor(ghat) = 1e-2 * ghat`, anchored to a per-element authored
        // length. It shipped once as a bare literal, which made it a size
        // dependent bug inside the rescaled frame: it read about 4e-6 m on a
        // domino and 2.6e-4 m against a six-metre ground triangle, and the
        // advance collapsed at a particular MESH COARSENESS with the topology
        // and contact count unchanged.
        for ghat in [1e-3f32, 1e-2, 0.25, 1.0] {
            assert_eq!(
                park_floor(ghat),
                1e-2 * ghat,
                "the parking floor is no longer proportional to ghat, which \
                 makes it a length whose world value depends on the primitive's \
                 own size"
            );
        }
    }

    #[test]
    fn the_analytic_park_keeps_the_divisor_away_from_zero() {
        // THE THREE PROPERTIES THE THREE ANALYTIC SWEEPS RELY ON, and none of
        // them can be stated at a call site. The floor, the sphere and the
        // pin's ghat ball each solve for a clearance and each hand it to a
        // `mass / gap^2`, so a sweep landing on the surface is a zero divisor;
        // that is the defect this rule exists to make unreachable, and its
        // symptom was a block-Jacobi diagonal of nine NaN on `cards`.
        //
        // ZERO IS IN THE EPS GRID DELIBERATELY. A zeroed `ParamSet` carries
        // `ccd_eps == 0`, and with no absolute allowance the two-regime rule
        // would otherwise park AT the start clearance and hand the line search
        // a zero-length step.
        for eps in [1e-7f32, 1e-5, 0.0] {
            for ghat in [1e-3f32, 1e-2, 0.25, 1.0] {
                let floor = 1e-2 * ghat;
                let bound = (2.0 * eps).max(floor);
                for clearance0 in [
                    1e-9f32,
                    1e-6,
                    0.5 * floor,
                    floor,
                    2.0 * floor,
                    0.5 * ghat,
                    ghat,
                    10.0 * ghat,
                ] {
                    let park = park_gap_analytic(clearance0, ghat, eps);
                    // 1. Strictly positive, or the divisor can still be zero.
                    assert!(park > 0.0, "park {park} is not positive at start {clearance0}, ghat {ghat}, eps {eps}");
                    // 2. Strictly below the start, or the crossing sits at
                    //    t = 0 and the line search stalls instead of advancing.
                    assert!(park < clearance0, "park {park} leaves no room to advance from {clearance0}, ghat {ghat}, eps {eps}");
                    // 3. Never above the parking clearance, so a vertex
                    //    arriving from far away parks at a bounded distance
                    //    rather than stopping short of the contact entirely.
                    //    The bound is the header's `floor_clear`, not
                    //    `park_floor` alone: a `ccd_eps` large relative to ghat
                    //    legitimately raises it.
                    assert!(park <= bound, "park {park} exceeds the parking clearance {bound} at start {clearance0}");
                }
                // A vertex at or inside the surface parks AT it, which
                // reproduces the unparked arithmetic exactly. That state is
                // refused by the caller's own assert on a positive time of
                // impact, not here.
                for clearance0 in [0.0f32, -1e-9, -floor, -ghat] {
                    assert_eq!(park_gap_analytic(clearance0, ghat, eps), 0.0);
                }
            }
        }
    }

    #[test]
    fn the_analytic_crossing_reproduces_the_unparked_one_at_a_zero_park() {
        // With `park == 0` the shared crossing must be bit-for-bit the surface
        // crossing the three sweeps solved before this rule existed, which is
        // what makes the change a no-op on a start that is already touching.
        for (clearance0, clearance1) in [
            (1.0f32, -1.0f32),
            (0.25, -0.75),
            (1e-3, -1e-6),
            (-1e-9, -1.0),
        ] {
            assert_eq!(
                park_crossing_analytic(clearance0, clearance1, 0.0),
                clearance0 / (clearance0 - clearance1),
                "the parked crossing no longer degenerates to the surface crossing at park == 0"
            );
        }
        // And with a park it lands ON the park, to round-off.
        let park = 1e-5f32;
        let t = park_crossing_analytic(1.0, -1.0, park);
        let landed = (1.0 - t) * 1.0 + t * -1.0;
        assert!((landed - park).abs() < 1e-6, "the crossing landed at {landed}, not at the park {park}");
    }

    #[test]
    fn a_pair_that_cannot_meet_takes_the_whole_step() {
        // A point moving parallel to a triangle, far from it, must not shorten
        // the step at all.
        let x0 = vec![
            point(0.0, 0.0, 5.0),
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let mut x1 = x0.clone();
        x1[0] = point(0.5, 0.0, 5.0);
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let mut filter = Filter::new(p.line_search_max_t);
        point_triangle(&frame, &mut filter, 0, [1, 2, 3], (0, 0), 0.0, 1e-3);
        assert_eq!(
            filter.time_of_impact(),
            1.0,
            "a sweep that cannot reach the triangle shortened the step anyway"
        );
        assert!(filter.overlapping_start().is_none());
    }

    #[test]
    fn a_point_driven_through_a_triangle_is_stopped_short_of_it() {
        // THE PROPERTY THE GUARANTEE RESTS ON. The point is commanded from one
        // side of the triangle to the other in a single step. The filter must
        // return a time strictly below 1, and following it must leave the point
        // on its own side.
        let x0 = vec![
            point(0.0, 0.0, 1.0),
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let mut x1 = x0.clone();
        x1[0] = point(0.0, 0.0, -1.0);
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let mut filter = Filter::new(p.line_search_max_t);
        point_triangle(&frame, &mut filter, 0, [1, 2, 3], (0, 0), 0.0, 1e-3);
        let toi = filter.time_of_impact();
        assert!(
            toi < 1.0,
            "a point commanded from z = +1 to z = -1 through a triangle in the \
             z = 0 plane was granted the whole step; with no CCD filter the \
             solver would complete and exit 0 having passed a surface through \
             another"
        );
        assert!(toi > 0.0, "the pair did not start overlapping, so the advance \
             must grant something; got {toi}");
        assert!(
            filter.overlapping_start().is_none(),
            "a pair starting 1 m apart was reported as starting overlapped"
        );
        // Following the filtered step must leave the point ABOVE the plane.
        let z = x0[0][2] as f64 + toi as f64 * (x1[0][2] as f64 - x0[0][2] as f64);
        assert!(
            z > 0.0,
            "advancing by the filtered time put the point at z = {z} m, on \
             the far side of the triangle"
        );
    }

    #[test]
    fn an_overlapping_start_is_reported_rather_than_returned_as_a_zero_step() {
        // A returned zero is not "no advance, carry on". It means the pair was
        // already inside the offset, which the conservative advance cannot
        // resolve, and a caller that reads it as a short step spins forever.
        let x0 = vec![
            point(0.0, 0.0, 0.0),
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let mut x1 = x0.clone();
        x1[0] = point(0.0, 0.0, -1.0);
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let mut filter = Filter::new(p.line_search_max_t);
        // The point starts ON the triangle, with a positive contact offset, so
        // it begins the step inside it.
        point_triangle(&frame, &mut filter, 0, [1, 2, 3], (7, 9), 0.01, 1e-3);
        assert_eq!(filter.time_of_impact(), 0.0);
        let report = filter
            .overlapping_start()
            .expect("a zero time must come with a report naming the pair");
        assert_eq!(report.elements, (7, 9));
        assert_eq!(report.kind, "point-triangle");
        assert!(report.describe().contains("(7, 9)"));
    }

    #[test]
    fn two_edges_swept_across_each_other_are_stopped_short() {
        let x0 = vec![
            point(-1.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            point(0.0, -1.0, 1.0),
            point(0.0, 1.0, 1.0),
        ];
        let mut x1 = x0.clone();
        x1[2] = point(0.0, -1.0, -1.0);
        x1[3] = point(0.0, 1.0, -1.0);
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let mut filter = Filter::new(p.line_search_max_t);
        edge_edge(&frame, &mut filter, [0, 1], [2, 3], (0, 1), 0.0, 1e-3);
        let toi = filter.time_of_impact();
        assert!(toi > 0.0 && toi < 1.0, "expected a partial advance, got {toi}");
        let z = x0[2][2] as f64 + toi as f64 * (x1[2][2] as f64 - x0[2][2] as f64);
        assert!(z > 0.0, "the second edge crossed the first at z = {z}");
    }

    #[test]
    fn two_grains_driven_together_are_stopped_at_their_offsets() {
        // ASYMMETRIC ON PURPOSE, AND THAT IS THE WHOLE TEST. The four positions
        // reach `accd::point_point_ccd` as two TRAJECTORIES, one per point,
        // which is not the start-major order the point-triangle and edge-edge
        // sweeps take. A fixture whose two grains swap places is invariant under
        // that mistake, because one grain's end IS the other's start, so it
        // reports a clean partial advance either way.
        //
        // Here the first grain moves LESS than the contact offset while the
        // second closes on it. Read start-major, the frame becomes the first
        // grain at two times, whose separation is its own 0.05 step against an
        // offset of 0.1: an overlapping start, reported as a zero time of
        // impact, which is the assertion below.
        const OFFSET: f32 = 0.1;
        let x0 = vec![point(0.0, 0.0, 0.0), point(0.3, 0.0, 0.0)];
        let x1 = vec![point(0.05, 0.0, 0.0), point(0.1, 0.0, 0.0)];
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let mut filter = Filter::new(p.line_search_max_t);
        point_point(&frame, &mut filter, 0, 1, OFFSET, 1e-3);
        assert!(
            filter.overlapping_start().is_none(),
            "the pair starts 0.3 apart against an offset of {OFFSET}, so a \
             report of an overlapping start means the sweep was handed a frame \
             built from one grain at two times: {}",
            filter.overlapping_start().unwrap().describe()
        );
        let toi = filter.time_of_impact();
        assert!(toi > 0.0 && toi < 1.0, "expected a partial advance, got {toi}");
        // Separation after the filtered step must exceed the offset, measured on
        // the trajectories the caller actually asked about.
        let ax = x0[0][0] as f64 + toi as f64 * (x1[0][0] - x0[0][0]) as f64;
        let bx = x0[1][0] as f64 + toi as f64 * (x1[1][0] - x0[1][0]) as f64;
        let gap = (bx - ax).abs();
        assert!(
            gap > f64::from(OFFSET),
            "the two grains ended {gap} apart with a contact offset of {OFFSET}"
        );
    }

    #[test]
    fn a_point_swept_against_an_edge_is_stopped_short() {
        let x0 = vec![point(0.0, 0.0, 1.0), point(-1.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
        let mut x1 = x0.clone();
        x1[0] = point(0.0, 0.0, -1.0);
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let mut filter = Filter::new(p.line_search_max_t);
        point_edge(&frame, &mut filter, 0, [1, 2], (0, 0), 0.05, 1e-3);
        let toi = filter.time_of_impact();
        assert!(toi > 0.0 && toi < 1.0, "expected a partial advance, got {toi}");
        let z = x0[0][2] as f64 + toi as f64 * (x1[0][2] - x0[0][2]) as f64;
        assert!(z > 0.0, "the point crossed the edge's line at z = {z}");
    }

    #[test]
    fn the_filter_keeps_the_smallest_time_over_many_pairs() {
        // The fold that makes a step safe for EVERY pair rather than the last
        // one considered.
        let x0 = vec![
            point(0.0, 0.0, 1.0),
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
            // A second, nearer triangle.
            point(-1.0, -1.0, 0.5),
            point(1.0, -1.0, 0.5),
            point(0.0, 1.0, 0.5),
        ];
        let mut x1 = x0.clone();
        x1[0] = point(0.0, 0.0, -1.0);
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let mut far = Filter::new(p.line_search_max_t);
        point_triangle(&frame, &mut far, 0, [1, 2, 3], (0, 0), 0.0, 1e-3);
        let mut near = Filter::new(p.line_search_max_t);
        point_triangle(&frame, &mut near, 0, [4, 5, 6], (1, 0), 0.0, 1e-3);
        assert!(
            near.time_of_impact() < far.time_of_impact(),
            "the nearer triangle must give the shorter time, or the fixture \
             does not discriminate"
        );

        let mut both = Filter::new(p.line_search_max_t);
        point_triangle(&frame, &mut both, 0, [1, 2, 3], (0, 0), 0.0, 1e-3);
        point_triangle(&frame, &mut both, 0, [4, 5, 6], (1, 0), 0.0, 1e-3);
        assert_eq!(both.time_of_impact(), near.time_of_impact());
        assert_eq!(both.pair_count(), 2);

        // And the same two in the other order, because a min-fold must not
        // depend on the order the pairs arrived in.
        let mut reversed = Filter::new(p.line_search_max_t);
        point_triangle(&frame, &mut reversed, 0, [4, 5, 6], (1, 0), 0.0, 1e-3);
        point_triangle(&frame, &mut reversed, 0, [1, 2, 3], (0, 0), 0.0, 1e-3);
        assert_eq!(reversed.time_of_impact(), both.time_of_impact());
    }

    #[test]
    fn a_device_fold_takes_the_smallest_time_and_the_first_report() {
        // THE TWO OPERATIONS THE DEVICE PATH FOLDS WITH, and they are what the
        // per-chunk `merge` amounts to. `Contact::line_search` reduces each
        // per-primitive array to one number and offers it here; it then walks
        // the overlap slots in ASCENDING order and offers the first flagged
        // one, which is what makes two runs of one scene name the same pair.
        let mut a = Filter::new(1.0);
        a.fold_time(0.5);
        a.fold_time(0.25);
        a.fold_time(0.75);
        assert_eq!(a.time_of_impact(), 0.25);

        a.record_overlap(OverlappingStart {
            kind: "edge-edge",
            elements: (4, 5),
            overlap: Overlap {
                d2: 0.0,
                offset: 1.0,
                flagged: 1,
            },
        });
        assert_eq!(a.overlapping_start().unwrap().elements, (4, 5));
        // A LATER SLOT DOES NOT DISPLACE AN EARLIER ONE.
        a.record_overlap(OverlappingStart {
            kind: "point-point",
            elements: (9, 9),
            overlap: Overlap::default(),
        });
        assert_eq!(a.overlapping_start().unwrap().elements, (4, 5));
    }

    #[test]
    fn an_overlap_slot_decodes_what_the_kernel_wrote() {
        // THE ONE PLACE THE `[[seam::pod(24)]]` RECORD IS READ BACK, and the
        // two float fields cross as raw bits because the array is allocated as
        // `u32` so one buffer serves both the record field and the fill that
        // clears it.
        let unwritten = [0u32; OVERLAP_WORDS];
        assert!(decode_overlap(&unwritten).is_none());

        let mut words = [0u32; OVERLAP_WORDS];
        words[0] = 1;
        words[1] = 3;
        words[2] = 11;
        words[3] = 12;
        words[4] = 0.25f32.to_bits();
        words[5] = 0.5f32.to_bits();
        let start = decode_overlap(&words).expect("a flagged slot decodes");
        assert_eq!(start.kind, "edge-edge");
        assert_eq!(start.elements, (11, 12));
        assert_eq!(start.overlap.d2, 0.25);
        assert_eq!(start.overlap.offset, 0.5);
        assert!(start.overlap.flagged != 0);
    }

    #[test]
    #[should_panic(expected = "positive finite")]
    fn a_non_positive_step_bound_is_refused() {
        Filter::new(0.0);
    }

    #[test]
    fn a_collider_triangle_stops_a_vertex_driven_through_it() {
        // THE SAME PROPERTY AS THE SELF-CONTACT SWEEP, against a side that does
        // not move. The dynamic side and the collider live in SEPARATE arrays,
        // so a wrong argument order here would build a frame out of the wrong
        // pair of poses; the previous stage's point-point defect was exactly
        // that, and it stopped every rod scene at its first step.
        let x0 = vec![point(0.0, 0.0, 1.0)];
        let x1 = vec![point(0.0, 0.0, -1.0)];
        let collider = vec![
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let statics = collision::StaticSide { vertex: &collider };
        let mut filter = Filter::new(p.line_search_max_t);
        collision::point_triangle(&frame, &statics, &mut filter, 0, [0, 1, 2], (0, 0), 0.0, 1e-3);
        let toi = filter.time_of_impact();
        assert!(
            toi < 1.0,
            "a vertex commanded from z = +1 to z = -1 through a collider triangle in the \
             z = 0 plane was granted the whole step, so nothing would stop it passing through"
        );
        assert!(toi > 0.0, "the pair did not start overlapping, so the advance must \
             grant something; got {toi}");
        let z = x0[0][2] as f64 + toi as f64 * (x1[0][2] as f64 - x0[0][2] as f64);
        assert!(z > 0.0, "advancing by the filtered time put the vertex at z = {z} m, \
             on the far side of the collider");
    }

    #[test]
    fn a_collider_vertex_stops_a_triangle_driven_onto_it() {
        // The other direction: the collider vertex is still, and the DYNAMIC
        // triangle sweeps onto it. The static point is passed as both ends of
        // its own trajectory, so a swapped argument would give it the dynamic
        // side's motion.
        let collider = vec![point(0.0, 0.0, 0.0)];
        let x0 = vec![
            point(-1.0, -1.0, 1.0),
            point(1.0, -1.0, 1.0),
            point(0.0, 1.0, 1.0),
        ];
        let x1 = vec![
            point(-1.0, -1.0, -1.0),
            point(1.0, -1.0, -1.0),
            point(0.0, 1.0, -1.0),
        ];
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let statics = collision::StaticSide { vertex: &collider };
        let mut filter = Filter::new(p.line_search_max_t);
        collision::static_point_triangle(
            &frame, &statics, &mut filter, 0, [0, 1, 2], (0, 0), 0.0, 1e-3,
        );
        let toi = filter.time_of_impact();
        assert!(
            toi < 1.0,
            "a triangle swept from z = +1 to z = -1 across a collider vertex at the origin \
             was granted the whole step"
        );
        assert!(toi > 0.0, "the pair did not start overlapping; got {toi}");
    }

    #[test]
    fn a_collider_edge_stops_a_dynamic_edge_crossing_it() {
        let collider = vec![point(-1.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
        let x0 = vec![point(0.0, -1.0, 1.0), point(0.0, 1.0, 1.0)];
        let x1 = vec![point(0.0, -1.0, -1.0), point(0.0, 1.0, -1.0)];
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let statics = collision::StaticSide { vertex: &collider };
        let mut filter = Filter::new(p.line_search_max_t);
        collision::edge_edge(&frame, &statics, &mut filter, [0, 1], [0, 1], (0, 0), 0.0, 1e-3);
        let toi = filter.time_of_impact();
        assert!(
            toi < 1.0,
            "a dynamic edge swept from z = +1 to z = -1 across a collider edge in the \
             z = 0 plane was granted the whole step"
        );
        assert!(toi > 0.0, "the pair did not start overlapping; got {toi}");
    }

    #[test]
    fn a_collider_the_motion_cannot_reach_takes_the_whole_step() {
        // The negative case the three above need beside them: without it, a
        // sweep that returned a short time for EVERY pair would pass them all.
        let x0 = vec![point(0.0, 0.0, 5.0)];
        let x1 = vec![point(0.5, 0.0, 5.0)];
        let collider = vec![
            point(-1.0, -1.0, 0.0),
            point(1.0, -1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let p = param(1.0, 1e-6);
        let frame = SweepFrame {
            x0: &x0,
            x1: &x1,
            param: &p,
        };
        let statics = collision::StaticSide { vertex: &collider };
        let mut filter = Filter::new(p.line_search_max_t);
        collision::point_triangle(&frame, &statics, &mut filter, 0, [0, 1, 2], (0, 0), 0.0, 1e-3);
        assert_eq!(
            filter.time_of_impact(),
            1.0,
            "a sweep five metres from the collider shortened the step anyway"
        );
        assert!(filter.overlapping_start().is_none());
    }
}

/// The three sweeps against a rest-pose STATIC collision mesh.
///
/// The static side has one pose, so the frame carries the motion of the dynamic
/// side alone. Two position arrays are involved rather than one, which is the
/// only thing that separates these from the four above: the collider lives
/// outside the solved namespace and is indexed in its own space.
///
/// Both indices in `element_ids` are reported as the shared body's caller named
/// them, dynamic first, so an overlap message reads in the same order the
/// assembly's does.
pub mod collision {
    use super::{Filter, PositionTriple, Overlap, Sweep, SweepFrame};

    /// The position triples of the static side, borrowed as raw components.
    pub struct StaticSide<'a> {
        pub vertex: &'a [PositionTriple],
    }

    impl StaticSide<'_> {
        fn pointer(&self) -> *const f32 {
            self.vertex.as_ptr() as *const f32
        }
    }

    /// A moving dynamic vertex against a collider triangle.
    #[allow(clippy::too_many_arguments)]
    pub fn point_triangle(
        frame: &SweepFrame,
        statics: &StaticSide,
        filter: &mut Filter,
        point: u32,
        triangle: [u32; 3],
        element_ids: (u32, u32),
        offset: f32,
        ghat: f32,
    ) {
        let (x0, x1, param) = frame.pointers();
        let mut overlap = Overlap::default();
        let toi = unsafe {
            super::collision_point_triangle_ccd_abi(
                x0,
                x1,
                point,
                statics.pointer(),
                triangle[0],
                triangle[1],
                triangle[2],
                offset,
                ghat,
                param,
                &mut overlap,
            )
        };
        filter.accept(Sweep::PointTriangle, element_ids, toi, overlap);
    }

    /// A collider vertex against a moving dynamic triangle.
    #[allow(clippy::too_many_arguments)]
    pub fn static_point_triangle(
        frame: &SweepFrame,
        statics: &StaticSide,
        filter: &mut Filter,
        point: u32,
        triangle: [u32; 3],
        element_ids: (u32, u32),
        offset: f32,
        ghat: f32,
    ) {
        let (x0, x1, param) = frame.pointers();
        let mut overlap = Overlap::default();
        let toi = unsafe {
            super::collision_point_triangle_ccd_static_point_abi(
                statics.pointer(),
                point,
                x0,
                x1,
                triangle[0],
                triangle[1],
                triangle[2],
                offset,
                ghat,
                param,
                &mut overlap,
            )
        };
        filter.accept(Sweep::PointTriangle, element_ids, toi, overlap);
    }

    /// A moving dynamic edge against a collider edge.
    #[allow(clippy::too_many_arguments)]
    pub fn edge_edge(
        frame: &SweepFrame,
        statics: &StaticSide,
        filter: &mut Filter,
        dynamic: [u32; 2],
        collider: [u32; 2],
        element_ids: (u32, u32),
        offset: f32,
        ghat: f32,
    ) {
        let (x0, x1, param) = frame.pointers();
        let mut overlap = Overlap::default();
        let toi = unsafe {
            super::collision_edge_edge_ccd_abi(
                x0,
                x1,
                dynamic[0],
                dynamic[1],
                statics.pointer(),
                collider[0],
                collider[1],
                offset,
                ghat,
                param,
                &mut overlap,
            )
        };
        filter.accept(Sweep::EdgeEdge, element_ids, toi, overlap);
    }
}
