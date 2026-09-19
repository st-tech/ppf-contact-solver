// File: crates/ppf-cts-solver/src/driver/sort.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Sorting, and a contract that has no shared body to carry it.
//!
//! CUDA sorts Morton codes with an 8-pass 4-bit LSD radix and a ballot-ranked
//! scatter. A CPU sorts by comparison. There is no per-element arithmetic in
//! common, so this is the audit's second documented exception: a different
//! skeleton with NO shared body, where what must be shared is the CONTRACT.
//!
//! # The contract
//!
//! **The sort is STABLE, ascending by key, with equal keys left in their
//! original index order.**
//!
//! That is not a preference. `radix_sort.cu:56-58` records that the LBVH tree
//! builder depends on it: two primitives with the same Morton code must come out
//! in ascending primitive order, because Karras's internal-node construction
//! resolves a duplicate-key run by index and produces a different tree if the
//! run is permuted.
//!
//! # Why this module exists rather than a call to the standard library
//!
//! `sort_unstable_by_key` is the idiomatic Rust spelling, is faster, and is
//! WRONG here in a way nothing reports. It produces a valid tree, of a different
//! shape, whose traversal finds the same contacts in a different order, whose
//! fp32 contact assembly therefore sums in a different order. Nothing asserts,
//! nothing panics, and a parity run against CUDA drifts for a reason no one
//! would look for in a sort. Naming the requirement in a module, with a test
//! that fails when it is violated, is what turns that into something a reviewer
//! can see.

// `par_stable_sort_by_key` is called from `bvh.rs` and `schwarz.rs`. The
// allow covers `stable_sort_by_key`, which only this module's own tests reach.
#![allow(dead_code)]

use rayon::prelude::*;

/// Sort `indices` by `key`, stably and ascending.
///
/// `indices` carries the permutation; `key[i]` is the key of element `i`. Equal
/// keys keep their relative order, which for a freshly-initialized index array
/// means ascending index, exactly as the device's radix leaves them.
pub fn stable_sort_by_key(indices: &mut [u32], key: &[u32]) {
    assert!(
        indices.iter().all(|i| (*i as usize) < key.len()),
        "every index must address a key"
    );
    // `sort_by_key` is the STABLE one. The unstable sibling is faster and is a
    // silent defect here; see the module docs.
    indices.sort_by_key(|i| key[*i as usize]);
}

/// The parallel form, which must produce the identical permutation.
///
/// Rayon's `par_sort_by_key` is stable, so the parallel and serial forms agree.
/// The test below is what keeps that true rather than assumed, because it is a
/// property of the library rather than of this code.
pub fn par_stable_sort_by_key(indices: &mut [u32], key: &[u32]) {
    assert!(
        indices.iter().all(|i| (*i as usize) < key.len()),
        "every index must address a key"
    );
    indices.par_sort_by_key(|i| key[*i as usize]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heavily duplicated keys, which is the case the contract is about and the
    /// case a real Morton array produces wherever primitives share a cell.
    fn duplicated(n: usize, distinct: u32) -> Vec<u32> {
        // Knuth's multiplicative hash, wrapping deliberately: the point is a
        // well-mixed key with many duplicates, and an overflow panic in a
        // fixture is noise rather than a finding.
        (0..n)
            .map(|i| (i as u32).wrapping_mul(2654435761) % distinct)
            .collect()
    }

    #[test]
    fn equal_keys_keep_ascending_index_order() {
        let key = duplicated(10_000, 32);
        let mut indices: Vec<u32> = (0..key.len() as u32).collect();
        stable_sort_by_key(&mut indices, &key);

        // Ascending by key.
        for w in indices.windows(2) {
            assert!(
                key[w[0] as usize] <= key[w[1] as usize],
                "the sort is not ascending by key"
            );
        }
        // And within a run of equal keys, ascending by index. This is the half
        // an unstable sort silently breaks.
        for w in indices.windows(2) {
            if key[w[0] as usize] == key[w[1] as usize] {
                assert!(
                    w[0] < w[1],
                    "equal keys came out permuted: {} before {}. The LBVH tree \
                     builder resolves duplicate Morton codes by index, so this \
                     produces a different tree with nothing to report it.",
                    w[0],
                    w[1]
                );
            }
        }
    }

    #[test]
    fn the_parallel_form_produces_the_identical_permutation() {
        let key = duplicated(50_000, 64);
        let mut serial: Vec<u32> = (0..key.len() as u32).collect();
        let mut parallel = serial.clone();
        stable_sort_by_key(&mut serial, &key);
        par_stable_sort_by_key(&mut parallel, &key);
        assert_eq!(
            serial, parallel,
            "the parallel sort must be stable too; if rayon ever changes that, \
             this is where it surfaces rather than in a drifting parity run"
        );
    }

    #[test]
    fn the_permutation_does_not_depend_on_the_thread_count() {
        let key = duplicated(50_000, 64);
        let reference = {
            let mut v: Vec<u32> = (0..key.len() as u32).collect();
            par_stable_sort_by_key(&mut v, &key);
            v
        };
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| {
                let mut v: Vec<u32> = (0..key.len() as u32).collect();
                par_stable_sort_by_key(&mut v, &key);
                v
            });
            assert_eq!(reference, got, "the permutation moved at {threads} threads");
        }
    }

    /// The negative control: an unstable sort must actually break the property,
    /// or the tests above prove nothing about the choice of sort.
    #[test]
    fn an_unstable_sort_really_does_break_the_contract() {
        let key = duplicated(10_000, 8);
        let mut indices: Vec<u32> = (0..key.len() as u32).collect();
        indices.sort_unstable_by_key(|i| key[*i as usize]);
        let permuted = indices
            .windows(2)
            .any(|w| key[w[0] as usize] == key[w[1] as usize] && w[0] > w[1]);
        assert!(
            permuted,
            "the unstable sort happened to be stable on this fixture, so the \
             stability tests above are not discriminating; strengthen the \
             fixture rather than trusting them"
        );
    }

    #[test]
    fn an_empty_or_single_input_sorts() {
        let key: Vec<u32> = vec![7];
        let mut one = vec![0u32];
        stable_sort_by_key(&mut one, &key);
        assert_eq!(one, vec![0]);

        let mut none: Vec<u32> = Vec::new();
        stable_sort_by_key(&mut none, &key);
        assert!(none.is_empty());
    }
}
