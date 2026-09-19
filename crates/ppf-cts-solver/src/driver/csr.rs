// File: crates/ppf-cts-solver/src/driver/csr.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The CSR row algorithms, called on the HOST against the same shared bodies a
//! kernel calls.
//!
//! `csrmat/dynamic_csr.kernel.cpp` holds heapsort, bisect, merge and compaction
//! for one row. They are not physics, and they are kernels by the rule that
//! decides: they compute values, so a second copy is a second set of behavior in
//! the structure the whole linear solve reads.
//!
//! **THE DRIVER DOES NOT COME THROUGH HERE.** `super::dyncsr` holds the matrix
//! in flat device storage and dispatches every one of those algorithms as an
//! element-wise pass, one row per thread. What this module holds is the HOST
//! call into the same bodies, and its tests, which is the only place they run
//! one row at a time where a failure names a line rather than a dispatch.
//!
//! **This module deliberately contains no sort, no bisect and no merge.** It
//! carries the buffers and calls the shared body. Anything here that started
//! ordering entries would be the second implementation the plan forbids, and the
//! test at the bottom of the file checks that mechanically.
//!
//! # Why reimplementing these would be worse than it looks
//!
//! The obvious argument for rewriting them is that Rust has `sort` and
//! `binary_search` already. The reason not to is what happens when the two
//! copies disagree. `dynamic_csr_finalize` rests on an invariant the fill
//! asserts: carried entries are distinct and disjoint from appended ones, so
//! only appended-versus-appended can collide. When that invariant breaks the row
//! stays NUMERICALLY correct, because the duplicate's carried entry goes
//! unpushed and drops as zero, and the only symptom is that a Newton step takes
//! 40 seconds instead of a fraction of one. A divergence with no wrong answer
//! attached is exactly the kind nobody finds.

// Every function here is reached from this module's own tests and from nowhere
// else, which is what the module documentation above says it is for.
#![allow(dead_code)]

// THE FOUR SHIM HELPERS THIS MODULE NAMES, AND WHY EACH IS A HELPER.
//
// None of them takes a thread range, so none of them is a dispatch: each is one
// row's algorithm over a slice the caller owns, which is the distinction
// `check-shared-wiring.py` rule 10 draws between a shim HELPER and a shim
// LAUNCHER. The launchers over these bodies exist and are generated:
// `dyn_row_begin_pass`, `dyn_row_compact_pass` and `dyn_row_emit_pass` are the
// dispatches the driver runs, declared beside the bodies they call.
//
// What these four add is a host caller. A generated entry resolves an arena
// handle and runs a range, so a failure inside one arrives as a dispatch that
// wrote the wrong bytes; a direct call arrives as an assertion on a slice, with
// the row in front of the reader. That is worth keeping for four algorithms
// whose failure mode is a matrix that is still numerically correct.
extern "C" {
    fn sort_pattern_abi(pattern: *mut u32, count: u32);
    fn find_sorted_abi(pattern: *const u32, count: u32, key: u32) -> u32;
    fn merge_runs_abi(a: *const u32, na: u32, b: *const u32, nb: u32, out: *mut u32);
    fn row_dedupe_abi(
        index: *mut u32,
        value: *mut f32,
        count: u32,
        carried: u32,
        appended_begin_out: *mut u32,
    ) -> u32;
}

/// Sort a row's carried pattern ascending, in place.
///
/// Called where a step starts USING the pattern rather than where it finishes
/// building one: the CUDA side restores a saved pattern on resume, so sorting at
/// build time alone would leave a resumed run bisecting an unordered array.
pub fn sort_pattern(pattern: &mut [u32]) {
    unsafe { sort_pattern_abi(pattern.as_mut_ptr(), pattern.len() as u32) };
}

/// Position of `key` in an ascending pattern, or the count when absent.
pub fn find_sorted(pattern: &[u32], key: u32) -> u32 {
    unsafe { find_sorted_abi(pattern.as_ptr(), pattern.len() as u32, key) }
}

/// Merge two ascending runs into `out`, which must hold `a.len() + b.len()`.
pub fn merge_runs(a: &[u32], b: &[u32], out: &mut [u32]) {
    assert_eq!(out.len(), a.len() + b.len(), "out must hold both runs");
    unsafe {
        merge_runs_abi(
            a.as_ptr(),
            a.len() as u32,
            b.as_ptr(),
            b.len() as u32,
            out.as_mut_ptr(),
        )
    };
}

/// What a compaction left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Compacted {
    /// Surviving entries.
    pub count: u32,
    /// Where the surviving carried run ends and the appended run begins.
    pub appended_begin: u32,
}

/// Compact a row in place, dropping zero blocks and folding duplicates.
///
/// `carried` is the width of the pattern the row searched before appending, and
/// the caller owns that invariant: a smaller value is safe and wasteful, a larger
/// one skips real duplicate checks.
pub fn row_dedupe(index: &mut [u32], value: &mut [f32], carried: u32) -> Compacted {
    let count = index.len() as u32;
    assert_eq!(
        value.len(),
        9 * index.len(),
        "one 3x3 block per index entry"
    );
    let mut appended_begin = 0u32;
    let surviving = unsafe {
        row_dedupe_abi(
            index.as_mut_ptr(),
            value.as_mut_ptr(),
            count,
            carried,
            &mut appended_begin,
        )
    };
    Compacted {
        count: surviving,
        appended_begin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pattern_sorts_ascending() {
        let mut pattern = vec![9u32, 3, 7, 1, 8, 2, 6, 0, 5, 4];
        sort_pattern(&mut pattern);
        assert_eq!(pattern, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn sorting_an_already_sorted_pattern_is_a_no_op() {
        // The CUDA side relies on this: `sort_pattern` runs on every step's
        // start, and on an already-ordered pattern it must return after one
        // pass rather than reordering anything.
        let ordered: Vec<u32> = (0..64).collect();
        let mut pattern = ordered.clone();
        sort_pattern(&mut pattern);
        assert_eq!(pattern, ordered);
    }

    #[test]
    fn find_sorted_locates_a_key_and_reports_absence() {
        let pattern: Vec<u32> = (0..100).map(|i| i * 3).collect();
        for (position, key) in pattern.iter().enumerate() {
            assert_eq!(
                find_sorted(&pattern, *key),
                position as u32,
                "key {key} should be at {position}"
            );
        }
        // An absent key reports the count, which is how the caller tells "not
        // present" from "at index 0".
        assert_eq!(find_sorted(&pattern, 1), pattern.len() as u32);
        assert_eq!(find_sorted(&pattern, 100_000), pattern.len() as u32);
    }

    #[test]
    fn merge_runs_produces_one_ascending_run() {
        let a: Vec<u32> = (0..50).map(|i| i * 2).collect();
        let b: Vec<u32> = (0..50).map(|i| i * 2 + 1).collect();
        let mut out = vec![0u32; a.len() + b.len()];
        merge_runs(&a, &b, &mut out);
        assert_eq!(out, (0..100).collect::<Vec<u32>>());
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "the merge is not ascending");
        }
    }

    #[test]
    fn merge_runs_handles_an_empty_side() {
        let a: Vec<u32> = vec![1, 4, 9];
        let mut out = vec![0u32; 3];
        merge_runs(&a, &[], &mut out);
        assert_eq!(out, a);
        let mut out2 = vec![0u32; 3];
        merge_runs(&[], &a, &mut out2);
        assert_eq!(out2, a);
    }

    /// Build a row: `count` entries whose blocks are all the given scalar times
    /// the identity, so a zero scalar is a zero block.
    fn row(indices: &[u32], scalars: &[f32]) -> (Vec<u32>, Vec<f32>) {
        assert_eq!(indices.len(), scalars.len());
        let mut value = vec![0.0f32; 9 * indices.len()];
        for (i, s) in scalars.iter().enumerate() {
            value[9 * i] = *s;
            value[9 * i + 4] = *s;
            value[9 * i + 8] = *s;
        }
        (indices.to_vec(), value)
    }

    #[test]
    fn dedupe_drops_zero_blocks() {
        // A zero block is a contribution that cancelled, and it must not survive
        // into the pattern the next step carries.
        let (mut index, mut value) = row(&[2, 5, 7, 9], &[1.0, 0.0, 3.0, 0.0]);
        let result = row_dedupe(&mut index, &mut value, 4);
        assert_eq!(result.count, 2, "two zero blocks should have been dropped");
        assert_eq!(&index[..2], &[2, 7]);
    }

    #[test]
    fn dedupe_leaves_a_clean_row_untouched() {
        let (mut index, mut value) = row(&[1, 3, 5, 8], &[1.0, 2.0, 3.0, 4.0]);
        let before = index.clone();
        let result = row_dedupe(&mut index, &mut value, 4);
        assert_eq!(result.count, 4);
        assert_eq!(index, before);
    }

    #[test]
    fn dedupe_reports_where_the_appended_run_begins() {
        // Two carried entries and two appended, with one carried zero dropped:
        // the appended run must start at 1, not at 2, or the caller walks the
        // wrong span and both runs stop being ascending.
        let (mut index, mut value) = row(&[2, 5, 1, 9], &[0.0, 4.0, 1.0, 2.0]);
        let result = row_dedupe(&mut index, &mut value, 2);
        assert_eq!(result.appended_begin, 1);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn an_empty_row_compacts_to_nothing() {
        let mut index: Vec<u32> = Vec::new();
        let mut value: Vec<f32> = Vec::new();
        let result = row_dedupe(&mut index, &mut value, 0);
        assert_eq!(result.count, 0);
    }

    /// The module's own rule, checked mechanically: nothing here orders entries.
    ///
    /// A grep is a weak test in general and the right one here, because the
    /// failure it guards against is somebody adding a convenient `sort` rather
    /// than a subtle numerical drift.
    #[test]
    fn this_module_contains_no_ordering_of_its_own() {
        // Only the implementation half is scanned. The test half necessarily
        // NAMES the banned spellings in order to ban them, and a scan that
        // included itself would fail on its own list, which is what the first
        // version of this test did.
        let source = include_str!("csr.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("the file has an implementation half");
        for banned in ["sort_unstable", "binary_search", "sort_by_key"] {
            let uses: Vec<&str> = implementation
                .lines()
                .filter(|l| l.contains(banned) && !l.trim_start().starts_with("//"))
                .collect();
            assert!(
                uses.is_empty(),
                "this module must call the shared index algorithms rather than \
                 ordering entries itself; found {banned} in {uses:?}"
            );
        }
        // And the scan must be looking at something, or it certifies nothing.
        assert!(
            implementation.contains("sort_pattern_abi"),
            "the implementation half was not found, so this check passed over \
             an empty string"
        );
    }
}
