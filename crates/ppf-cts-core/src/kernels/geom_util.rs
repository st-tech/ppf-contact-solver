// File: crates/ppf-cts-core/src/kernels/geom_util.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Shared geometric helpers used across BVH, intersection, and proximity
// kernels. Single source so signatures, inlining, and numerics stay
// identical at every call site.

#[inline]
pub(super) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(super) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(super) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Axis-aligned bbox overlap test. Inclusive on the boundary, matching
/// the original duplicated implementations in intersection / proximity.
#[inline]
pub(super) fn bbox_overlap(
    a_min: &[f64; 3],
    a_max: &[f64; 3],
    b_min: &[f64; 3],
    b_max: &[f64; 3],
) -> bool {
    for d in 0..3 {
        if a_min[d] > b_max[d] || b_min[d] > a_max[d] {
            return false;
        }
    }
    true
}

/// Element bbox expanded by per-element contact offset:
/// `bbox_min - offset` / `bbox_max + offset`.
#[inline]
pub(super) fn expand_bbox(
    bb_min: &[f64; 3],
    bb_max: &[f64; 3],
    offset: f64,
) -> ([f64; 3], [f64; 3]) {
    (
        [bb_min[0] - offset, bb_min[1] - offset, bb_min[2] - offset],
        [bb_max[0] + offset, bb_max[1] + offset, bb_max[2] + offset],
    )
}

/// Squared distance from a point to an axis-aligned bbox. Zero if the
/// point lies inside or on the bbox.
#[inline]
pub(super) fn point_to_bbox_dist_sq(
    point: [f64; 3],
    bmin: &[f64; 3],
    bmax: &[f64; 3],
) -> f64 {
    let mut d_sq = 0.0;
    for i in 0..3 {
        if point[i] < bmin[i] {
            let d = bmin[i] - point[i];
            d_sq += d * d;
        } else if point[i] > bmax[i] {
            let d = point[i] - bmax[i];
            d_sq += d * d;
        }
    }
    d_sq
}

#[inline]
pub(super) fn elements_share_vertex_2_3(elem2: [i32; 2], elem3: [i32; 3]) -> bool {
    elem2[0] == elem3[0]
        || elem2[0] == elem3[1]
        || elem2[0] == elem3[2]
        || elem2[1] == elem3[0]
        || elem2[1] == elem3[1]
        || elem2[1] == elem3[2]
}

#[inline]
pub(super) fn elements_share_vertex_3_3(a: [i32; 3], b: [i32; 3]) -> bool {
    a[0] == b[0] || a[0] == b[1] || a[0] == b[2]
        || a[1] == b[0] || a[1] == b[1] || a[1] == b[2]
        || a[2] == b[0] || a[2] == b[1] || a[2] == b[2]
}
