// File: crates/ppf-cts-core/src/kernels/start_links.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Allow Existing Intersections: the vertex links a scene starts with.
//
// Two elements that share a vertex are never a contact pair. This option
// extends that neighbor relation beyond mesh topology: two elements the scene
// STARTS intersecting with (or closer than their contact offsets, which the
// build refuses in the same way) are linked, vertex by vertex, and the solver
// treats a linked pair exactly like a topological neighbor for the whole run.
// No barrier, no CCD filter and no intersection report.
//
// THE LINK IS A VERTEX PAIR, NOT AN ELEMENT PAIR. For an intersecting pair of
// elements A and B, every vertex of A is linked with every vertex of B. The
// solver then exempts two elements when ANY vertex of one is linked to ANY
// vertex of the other, so an element sharing a vertex with a tangled one is
// exempt too: the tangled region plus one ring on each side. That ring is what
// lets a fold slide a little before it meets a pair with full contact.
//
// THIS FILE IS THE SINGLE SOURCE OF THE SET. The solver never adds a link of
// its own: a pair its initialize scan finds that no link covers stops the run
// by name. And a link is permanent for the run, as a topological neighbor is.
//
// The vertex namespace is the combined one the build check scans: the dynamic
// vertices first, then the rest-pose STATIC collision vertices appended after
// them. The solver splits it back into its two pools.

use std::collections::BTreeSet;

/// Collects the vertex links for the pairs the opt-in covers.
///
/// `allow` is per vertex of the combined namespace: true where the vertex's
/// object opted into Allow Existing Intersections. A STATIC collision vertex
/// carries false; a collider is always the OTHER object, so "either side opts
/// in" makes the dynamic side the whole decision.
pub struct StartLinks<'a> {
    allow: &'a [bool],
    links: BTreeSet<(u32, u32)>,
    element_pairs: usize,
    /// The first [`MAX_RECORDED_PAIRS`] linked element pairs, each as its two
    /// vertex lists, so the add-on can draw what was exempted.
    recorded: Vec<(Vec<u32>, Vec<u32>)>,
}

/// How many linked element pairs are kept for display. The links themselves
/// are never capped; this bounds only the overlay payload, which crosses the
/// server protocol on every status poll.
pub const MAX_RECORDED_PAIRS: usize = 4096;

impl<'a> StartLinks<'a> {
    pub fn new(allow: &'a [bool]) -> Self {
        StartLinks {
            allow,
            links: BTreeSet::new(),
            element_pairs: 0,
            recorded: Vec::new(),
        }
    }

    /// Whether the opt-in covers this pair: EITHER side's object opted in.
    ///
    /// Read off each element's FIRST vertex, which is the convention every
    /// per-object allowance uses: a vertex belongs to exactly one object, so
    /// every element built on it does too.
    pub fn covers(&self, a: &[u32], b: &[u32]) -> bool {
        let opted = |element: &[u32]| {
            element
                .first()
                .is_some_and(|&v| self.allow.get(v as usize).copied().unwrap_or(false))
        };
        opted(a) || opted(b)
    }

    /// Link this pair if the opt-in covers it, and say whether it did.
    ///
    /// A pair the opt-in does not cover is left to the caller, which reports
    /// it exactly as it would without the option.
    pub fn link_if_covered(&mut self, a: &[u32], b: &[u32]) -> bool {
        if !self.covers(a, b) {
            return false;
        }
        for &u in a {
            for &v in b {
                // Two elements the scan found intersecting share no vertex, so
                // `u == v` does not arise; a self-link would say nothing.
                if u != v {
                    self.links.insert((u.min(v), u.max(v)));
                }
            }
        }
        self.element_pairs += 1;
        if self.recorded.len() < MAX_RECORDED_PAIRS {
            self.recorded.push((a.to_vec(), b.to_vec()));
        }
        true
    }

    /// The linked element pairs kept for display, in the order they were
    /// linked; at most [`MAX_RECORDED_PAIRS`] of [`Self::element_pairs`].
    pub fn recorded_pairs(&self) -> &[(Vec<u32>, Vec<u32>)] {
        &self.recorded
    }

    /// How many element pairs were linked.
    pub fn element_pairs(&self) -> usize {
        self.element_pairs
    }

    /// The links as a flat `[u0, v0, u1, v1, ...]` list, each pair with
    /// `u < v`, sorted and deduplicated.
    pub fn into_flat(self) -> Vec<u32> {
        let mut flat = Vec::with_capacity(2 * self.links.len());
        for (u, v) in self.links {
            flat.push(u);
            flat.push(v);
        }
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_covered_pair_links_every_vertex_of_one_side_to_every_vertex_of_the_other() {
        let allow = vec![true; 6];
        let mut links = StartLinks::new(&allow);
        assert!(links.link_if_covered(&[0, 1, 2], &[3, 4, 5]));
        assert_eq!(links.element_pairs(), 1);
        let flat = links.into_flat();
        assert_eq!(flat.len(), 2 * 9);
        for u in 0..3u32 {
            for v in 3..6u32 {
                assert!(flat.chunks(2).any(|p| p == [u, v]), "missing link {u}-{v}");
            }
        }
    }

    #[test]
    fn links_are_ordered_and_deduplicated() {
        let allow = vec![true; 6];
        let mut links = StartLinks::new(&allow);
        links.link_if_covered(&[3, 4, 5], &[0, 1, 2]);
        links.link_if_covered(&[0, 1, 2], &[3, 4, 5]);
        assert_eq!(links.element_pairs(), 2);
        let flat = links.into_flat();
        assert_eq!(flat.len(), 2 * 9);
        for pair in flat.chunks(2) {
            assert!(pair[0] < pair[1]);
        }
        let mut sorted = flat.chunks(2).map(|p| (p[0], p[1])).collect::<Vec<_>>();
        sorted.sort();
        assert_eq!(sorted, flat.chunks(2).map(|p| (p[0], p[1])).collect::<Vec<_>>());
    }

    #[test]
    fn either_side_opting_in_is_enough_and_neither_is_refused() {
        // Vertices 0..3 opted in, 3..6 did not.
        let allow = [true, true, true, false, false, false];
        let links = StartLinks::new(&allow);
        assert!(links.covers(&[0, 1, 2], &[3, 4, 5]));
        assert!(links.covers(&[3, 4, 5], &[0, 1, 2]));
        assert!(!links.covers(&[3, 4], &[5, 4]));
        let mut links = StartLinks::new(&allow);
        assert!(!links.link_if_covered(&[3, 4], &[5]));
        assert_eq!(links.element_pairs(), 0);
        assert!(links.into_flat().is_empty());
    }

    #[test]
    fn a_rod_edge_against_a_triangle_links_two_by_three() {
        let allow = [true, true, false, false, false];
        let mut links = StartLinks::new(&allow);
        assert!(links.link_if_covered(&[0, 1], &[2, 3, 4]));
        assert_eq!(links.into_flat().len(), 2 * 6);
    }

    #[test]
    fn the_opt_in_is_read_off_the_first_vertex() {
        // Only vertex 1 opted in; the element [0, 1] is described by vertex 0.
        let allow = [false, true, false, false];
        let links = StartLinks::new(&allow);
        assert!(!links.covers(&[0, 1], &[2, 3]));
        assert!(links.covers(&[1, 0], &[2, 3]));
    }

    #[test]
    fn linked_pairs_are_recorded_for_display_and_uncovered_ones_are_not() {
        let allow = [true, true, false, false, false];
        let mut links = StartLinks::new(&allow);
        links.link_if_covered(&[0, 1], &[2, 3, 4]);
        links.link_if_covered(&[2, 3], &[4]);
        assert_eq!(links.recorded_pairs(), &[(vec![0, 1], vec![2, 3, 4])]);
    }

    #[test]
    fn a_vertex_past_the_opt_in_array_did_not_opt_in() {
        // The STATIC collision vertices sit past the dynamic ones, and a
        // caller may hand an array covering only the dynamic pool.
        let allow = [false, false];
        let links = StartLinks::new(&allow);
        assert!(!links.covers(&[5, 6, 7], &[0, 1]));
    }
}
