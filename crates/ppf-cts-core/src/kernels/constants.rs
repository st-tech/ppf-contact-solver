// File: crates/ppf-cts-core/src/kernels/constants.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Numerical constants shared across BVH, intersection, and proximity
// kernels. Per-file thresholds with distinct semantic intent (e.g.,
// `COPLANAR_EPS` vs. `SEGMENT_2D_EPS_CROSS` in intersection.rs, even
// when they share a numeric value) stay where they are; only constants
// that are genuinely the same thing in every file live here.

/// Maximum BVH traversal stack depth. log2(N/leaf_size) for N up to
/// 2^32 with leaf=1 is 32, so 64 is comfortably above any achievable
/// real-world depth and lets us stack-allocate the traversal stack
/// instead of heap-allocating one Vec per query (which dominated at
/// 1M-query benchmarks). Used by closest-point queries in bvh.rs and
/// by the proximity / intersection traversal scaffolds.
pub(super) const BVH_STACK_CAP: usize = 64;
