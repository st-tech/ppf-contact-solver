// File: crates/ppf-cts-core/src/kernels/geom_util.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Shared geometric helpers used across BVH, intersection, and proximity
// kernels. Single source so signatures, inlining, and numerics stay
// identical at every call site.

/// Fetch vertex `idx` from a flat `[x, y, z, x, y, z, ...]` buffer.
/// Checked index, matching the original per-function closures.
#[inline]
pub(super) fn vert3(verts: &[f64], idx: i32) -> [f64; 3] {
    let k = idx as usize;
    [verts[3 * k], verts[3 * k + 1], verts[3 * k + 2]]
}

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

/// Clamp raw planar barycentric weights to the triangle and project the
/// query point onto the resulting convex combination.
///
/// Callers solve the planar barycentric coordinates themselves (each
/// keeps its own Gram-matrix degeneracy gate, with epsilons that differ
/// per site) and pass the raw `(alpha, beta, gamma)` in. This helper
/// runs only the shared tail: clamp each weight to `[0, 1]`, renormalize
/// by their sum (guarding a zero sum), project onto the triangle, and
/// report the distance from `p` to the projected point. The clamp-before-
/// renormalize order is preserved exactly, so this is a valid convex
/// projection but does not match a Voronoi-region closest-point search
/// for points outside the triangle.
///
/// Returns `(bary, proj, dist)` where `bary` are the clamped, renormalized
/// weights, `proj = bary[0]*v0 + bary[1]*v1 + bary[2]*v2`, and `dist` is
/// the Euclidean distance from `p` to `proj`.
#[inline]
pub(super) fn barycentric_clamp_project(
    alpha: f64,
    beta: f64,
    gamma: f64,
    p: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> ([f64; 3], [f64; 3], f64) {
    let mut a = alpha.clamp(0.0, 1.0);
    let mut b = beta.clamp(0.0, 1.0);
    let mut c = gamma.clamp(0.0, 1.0);
    let s = a + b + c;
    let safe_s = if s > 0.0 { s } else { 1.0 };
    a /= safe_s;
    b /= safe_s;
    c /= safe_s;
    let proj = [
        a * v0[0] + b * v1[0] + c * v2[0],
        a * v0[1] + b * v1[1] + c * v2[1],
        a * v0[2] + b * v1[2] + c * v2[2],
    ];
    let dx = p[0] - proj[0];
    let dy = p[1] - proj[1];
    let dz = p[2] - proj[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    ([a, b, c], proj, dist)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barycentric_clamp_project_interior_is_exact() {
        // Centroid of the unit right triangle projects to itself.
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let p = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        // Planar solve for an in-plane interior point gives the same weights.
        let (bary, proj, dist) =
            barycentric_clamp_project(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, p, v0, v1, v2);
        for k in 0..3 {
            assert!((bary[k] - 1.0 / 3.0).abs() < 1e-12);
            assert!((proj[k] - p[k]).abs() < 1e-12);
        }
        assert!(dist < 1e-12);
    }

    #[test]
    fn barycentric_clamp_project_exterior_matches_clamp_renormalize() {
        // Locks the clamp-then-renormalize numerics (NOT a Voronoi-region
        // closest point). Point p = (-1, 0.5, 0) over the unit right
        // triangle solves to raw bary (alpha, beta, gamma) = (1.5, -1, 0.5);
        // clamp-renormalize yields (2/3, 0, 1/3), proj (0, 1/3, 0), and a
        // distance of ~1.0138, well above the true closest distance 1.0.
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let p = [-1.0, 0.5, 0.0];
        let (bary, proj, dist) = barycentric_clamp_project(1.5, -1.0, 0.5, p, v0, v1, v2);
        assert!((bary[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!(bary[1].abs() < 1e-12);
        assert!((bary[2] - 1.0 / 3.0).abs() < 1e-12);
        assert!(proj[0].abs() < 1e-12);
        assert!((proj[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!(proj[2].abs() < 1e-12);
        let expected = (1.0_f64 + (0.5 - 1.0 / 3.0) * (0.5 - 1.0 / 3.0)).sqrt();
        assert!((dist - expected).abs() < 1e-12);
        // The Voronoi-region closest point would give exactly 1.0, so this
        // confirms the dedup did not silently swap in that tighter method.
        assert!(dist > 1.0);
    }
}
