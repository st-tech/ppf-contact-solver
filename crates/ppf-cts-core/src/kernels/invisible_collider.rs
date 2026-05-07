// File: crates/ppf-cts-core/src/kernels/invisible_collider.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Wall + sphere violation scans. 1:1 port of
// frontend/_invisible_collider_.py
// (`_check_wall_violations_parallel`, `_check_sphere_violations_parallel`).
//
// Numerics policy: f64 throughout, matching the Python path which
// `np.ascontiguousarray(..., dtype=np.float64)`s its inputs. Output
// vector is in vertex-index order, same as the Python loop's two-pass
// (mark-then-collect) shape.
//
// Parallelism: rayon `par_iter`. Each iteration is read-only on the
// inputs and produces an Option that the par_iter `filter_map`
// collects. Order is preserved by `collect()` on parallel iterators.

use rayon::prelude::*;

/// Scan vertices against a wall plane.
///
/// `vertices_flat` is the row-major (N, 3) buffer; `is_pinned` is a
/// length-N bool slice. `wall_normal_unit` must already be unit-length;
/// the caller normalizes.
///
/// Returns `(vertex_index, signed_distance)` for every vertex on the
/// wrong side of the plane (signed dist < 0), in vertex-index order.
pub fn check_wall_violations(
    vertices_flat: &[f64],
    is_pinned: &[bool],
    wall_pos: [f64; 3],
    wall_normal_unit: [f64; 3],
) -> Vec<(usize, f64)> {
    let n_verts = vertices_flat.len() / 3;
    debug_assert_eq!(vertices_flat.len(), n_verts * 3, "vertices_flat must be (N, 3)");
    debug_assert_eq!(is_pinned.len(), n_verts, "is_pinned must be length N");

    (0..n_verts)
        .into_par_iter()
        .filter_map(|i| {
            if is_pinned[i] {
                return None;
            }
            let dx = vertices_flat[3 * i] - wall_pos[0];
            let dy = vertices_flat[3 * i + 1] - wall_pos[1];
            let dz = vertices_flat[3 * i + 2] - wall_pos[2];
            let signed = dx * wall_normal_unit[0]
                + dy * wall_normal_unit[1]
                + dz * wall_normal_unit[2];
            if signed < 0.0 {
                Some((i, signed))
            } else {
                None
            }
        })
        .collect()
}

/// Scan vertices against a sphere.
///
/// * `is_inverted = false` (default): vertex must stay outside the
///   sphere; violation when distance < radius.
/// * `is_inverted = true`: vertex must stay inside; violation when
///   distance > radius.
/// * `is_hemisphere = true`: when the vertex is above center.y, the
///   effective center is lifted to the vertex's y-level so the top of
///   the hemisphere acts as an open cylinder.
///
/// Returns `(vertex_index, distance_to_surface)` where
/// `distance_to_surface = radius - distance` (positive for vertices
/// inside the sphere).
pub fn check_sphere_violations(
    vertices_flat: &[f64],
    is_pinned: &[bool],
    sphere_center: [f64; 3],
    sphere_radius: f64,
    is_inverted: bool,
    is_hemisphere: bool,
) -> Vec<(usize, f64)> {
    let n_verts = vertices_flat.len() / 3;
    debug_assert_eq!(vertices_flat.len(), n_verts * 3, "vertices_flat must be (N, 3)");
    debug_assert_eq!(is_pinned.len(), n_verts, "is_pinned must be length N");

    let radius_sq = sphere_radius * sphere_radius;

    (0..n_verts)
        .into_par_iter()
        .filter_map(|i| {
            if is_pinned[i] {
                return None;
            }
            let vx = vertices_flat[3 * i];
            let vy = vertices_flat[3 * i + 1];
            let vz = vertices_flat[3 * i + 2];

            // Hemisphere lifts the center to the vertex's y when
            // vertex is above the original center.
            let cy = if is_hemisphere && vy > sphere_center[1] {
                vy
            } else {
                sphere_center[1]
            };
            let dx = vx - sphere_center[0];
            let dy = vy - cy;
            let dz = vz - sphere_center[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            let violates = if is_inverted {
                dist_sq > radius_sq
            } else {
                dist_sq < radius_sq
            };
            if !violates {
                return None;
            }
            // Distance computed once here, only on violating vertices.
            let dist_to_surface = sphere_radius - dist_sq.sqrt();
            Some((i, dist_to_surface))
        })
        .collect()
}

/// Static wall descriptor used by the batch violation collector.
/// `wall_pos` is the keyframe-0 position; `wall_normal_unit` must
/// already be unit length (callers normalize before constructing).
#[derive(Clone, Copy, Debug)]
pub struct WallDesc {
    pub pos: [f64; 3],
    pub normal_unit: [f64; 3],
}

/// Static sphere descriptor used by the batch violation collector.
#[derive(Clone, Copy, Debug)]
pub struct SphereDesc {
    pub center: [f64; 3],
    pub radius: f64,
    pub is_inverted: bool,
    pub is_hemisphere: bool,
}

/// Batch wall scan: for each static wall, run `check_wall_violations`
/// and emit `(vertex_index, wall_index, signed_distance)` rows in
/// (wall_index ascending, vertex_index ascending) order.
///
/// `wall_indices_in` lets callers tag the results with the original
/// list position even when kinematic walls have been filtered out
/// upstream.
pub fn check_walls_violations_batch(
    vertices_flat: &[f64],
    is_pinned: &[bool],
    walls: &[WallDesc],
    wall_indices_in: &[usize],
) -> Vec<(usize, usize, f64)> {
    debug_assert_eq!(walls.len(), wall_indices_in.len());
    let mut out: Vec<(usize, usize, f64)> = Vec::new();
    for (w, &wall_idx) in walls.iter().zip(wall_indices_in.iter()) {
        let rows = check_wall_violations(vertices_flat, is_pinned, w.pos, w.normal_unit);
        out.reserve(rows.len());
        for (vi, signed) in rows {
            out.push((vi, wall_idx, signed));
        }
    }
    out
}

/// Batch sphere scan: for each static sphere, run
/// `check_sphere_violations` and emit
/// `(vertex_index, sphere_index, distance_to_surface)` rows.
pub fn check_spheres_violations_batch(
    vertices_flat: &[f64],
    is_pinned: &[bool],
    spheres: &[SphereDesc],
    sphere_indices_in: &[usize],
) -> Vec<(usize, usize, f64)> {
    debug_assert_eq!(spheres.len(), sphere_indices_in.len());
    let mut out: Vec<(usize, usize, f64)> = Vec::new();
    for (s, &sphere_idx) in spheres.iter().zip(sphere_indices_in.iter()) {
        let rows = check_sphere_violations(
            vertices_flat,
            is_pinned,
            s.center,
            s.radius,
            s.is_inverted,
            s.is_hemisphere,
        );
        out.reserve(rows.len());
        for (vi, dist) in rows {
            out.push((vi, sphere_idx, dist));
        }
    }
    out
}

/// Format the verbose-mode tag for a sphere descriptor:
/// `""`, `" (inverted)"`, `" (hemisphere)"`, or `" (inverted, hemisphere)"`.
pub fn format_sphere_mode_tag(is_inverted: bool, is_hemisphere: bool) -> String {
    match (is_inverted, is_hemisphere) {
        (false, false) => String::new(),
        (true, false) => " (inverted)".to_string(),
        (false, true) => " (hemisphere)".to_string(),
        (true, true) => " (inverted, hemisphere)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: &[[f64; 3]]) -> Vec<f64> {
        v.iter().flat_map(|p| p.iter().copied()).collect()
    }

    #[test]
    fn wall_floor_violations() {
        // Floor plane at y=0 with normal +Y. Two of three are below.
        let v = flat(&[
            [0.0, 0.5, 0.0],   // above (ok)
            [1.0, -0.1, 0.0],  // below (violates, signed = -0.1)
            [-1.0, -0.5, 0.5], // below (violates, signed = -0.5)
        ]);
        let pinned = vec![false; 3];
        let out =
            check_wall_violations(&v, &pinned, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 1);
        assert!((out[0].1 - (-0.1)).abs() < 1e-12);
        assert_eq!(out[1].0, 2);
        assert!((out[1].1 - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn wall_skips_pinned() {
        let v = flat(&[[0.0, -1.0, 0.0]]);
        let pinned = vec![true];
        let out =
            check_wall_violations(&v, &pinned, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(out.is_empty(), "pinned vertex must be skipped");
    }

    #[test]
    fn sphere_default_outside_required() {
        // Sphere at origin, r=1. Vertex inside violates.
        let v = flat(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let pinned = vec![false; 2];
        let out = check_sphere_violations(&v, &pinned, [0.0, 0.0, 0.0], 1.0, false, false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        // dist_to_surface = radius - dist = 1 - 0 = 1 (inside, positive)
        assert!((out[0].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sphere_inverted_inside_required() {
        // Inverted sphere at origin, r=1. Vertex outside violates.
        let v = flat(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let pinned = vec![false; 2];
        let out = check_sphere_violations(&v, &pinned, [0.0, 0.0, 0.0], 1.0, true, false);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 1);
        // dist_to_surface = 1 - 2 = -1 (outside, negative)
        assert!((out[0].1 - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn hemisphere_above_uses_cylinder_distance() {
        // Hemisphere at origin, r=1. Vertex (0.5, 5, 0) is far above
        // center but only 0.5 from the central axis: must violate
        // (inside the cylinder), not pass (outside if treated as a
        // full sphere).
        let v = flat(&[[0.5, 5.0, 0.0]]);
        let pinned = vec![false];
        let out = check_sphere_violations(&v, &pinned, [0.0, 0.0, 0.0], 1.0, false, true);
        assert_eq!(out.len(), 1, "above-center hemisphere check uses xz-only distance");
        // y is lifted to vertex.y, so dist = 0.5; dist_to_surface = 0.5
        assert!((out[0].1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn order_preserved() {
        // Verify rayon collect() preserves source order for our
        // filter_map shape, since the Python loop's two-pass
        // (mark-then-collect) emits in vertex-index order.
        let mut points = vec![];
        for i in 0..1000 {
            let y = if i % 3 == 0 { -1.0 } else { 1.0 };
            points.push([i as f64, y, 0.0]);
        }
        let v = flat(&points);
        let pinned = vec![false; points.len()];
        let out =
            check_wall_violations(&v, &pinned, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        for w in out.windows(2) {
            assert!(w[0].0 < w[1].0, "indices must be monotonic");
        }
    }

    #[test]
    fn walls_batch_emits_per_wall_grouped_rows() {
        let v = flat(&[[0.0, -1.0, 0.0], [0.0, 0.5, 0.0]]);
        let pinned = vec![false; 2];
        let walls = vec![
            WallDesc {
                pos: [0.0, 0.0, 0.0],
                normal_unit: [0.0, 1.0, 0.0],
            },
            WallDesc {
                pos: [0.0, 0.0, 0.0],
                normal_unit: [1.0, 0.0, 0.0],
            },
        ];
        let idxs = vec![3usize, 7usize];
        let out = check_walls_violations_batch(&v, &pinned, &walls, &idxs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[0].1, 3);
    }

    #[test]
    fn spheres_batch_uses_sphere_indices() {
        let v = flat(&[[0.0, 0.0, 0.0]]);
        let pinned = vec![false];
        let spheres = vec![SphereDesc {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            is_inverted: false,
            is_hemisphere: false,
        }];
        let out = check_spheres_violations_batch(&v, &pinned, &spheres, &[42]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, 42);
    }

    #[test]
    fn sphere_mode_tag_formats() {
        assert_eq!(format_sphere_mode_tag(false, false), "");
        assert_eq!(format_sphere_mode_tag(true, false), " (inverted)");
        assert_eq!(format_sphere_mode_tag(false, true), " (hemisphere)");
        assert_eq!(
            format_sphere_mode_tag(true, true),
            " (inverted, hemisphere)"
        );
    }

    #[test]
    fn wall_on_plane_is_not_violation() {
        // Vertex exactly on the wall plane has signed distance 0.
        // The kernel reports violations strictly below (signed < 0).
        let v = flat(&[[0.0, 0.0, 0.0]]);
        let pinned = vec![false];
        let out =
            check_wall_violations(&v, &pinned, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(out.is_empty(), "vertex on the plane is not a violation");
    }

    #[test]
    fn hemisphere_below_center_acts_as_full_sphere() {
        // Hemisphere lift only kicks in when vy > center.y. For a
        // vertex below center, the kernel falls back to the full
        // sphere distance, so a point at (0.5, -0.1, 0) is inside
        // (dist ~ 0.51 < 1) and must violate the default-outside rule.
        let v = flat(&[[0.5, -0.1, 0.0]]);
        let pinned = vec![false];
        let out = check_sphere_violations(&v, &pinned, [0.0, 0.0, 0.0], 1.0, false, true);
        assert_eq!(out.len(), 1, "below-center hemisphere falls through to sphere");
        assert_eq!(out[0].0, 0);
    }
}
