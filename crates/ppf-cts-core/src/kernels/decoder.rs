// File: crates/ppf-cts-core/src/kernels/decoder.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pure-compute kernels backing `frontend/_decoder_.py`. Each function
// is a self-contained slice-in / slice-out routine so the PyO3
// wrappers in `crates/ppf-cts-py/src/decoder_py.rs` can drive them
// without holding the GIL.
//
// Covers the math the decoder runs while it walks the in-memory
// scene/session structures: 4x4 vertex transforms, the barycentric
// anchor projection used by explicit cross-stitch, the surface-frame
// "orig_to_sim" lookup, the build-plan summary line, the
// closest-vertex search used by the torque-hint translator, and the
// per-segment translation deltas for keyframed pin playback.
//
// The decoder itself stays in Python because the on-disk envelope is
// pickle / CBOR dispatch and the consumers are Python objects (Scene,
// Session, PinHolder, ParamHolder); see `datamodel::decoder` for the
// path math and the validation helpers the Python decoder calls
// before mutating its caller's scene/session.

use super::geom_util::barycentric_clamp_project;

/// Apply a 4x4 transform to local-space vertices and return
/// world-space vertices in `f32`.
///
/// Computes `v @ mat[:3, :3].T + mat[:3, 3]` per row.
///
/// `local` is the flat `(N, 3)` row-major buffer; `transform` is the
/// flat `(4, 4)` row-major buffer. Returns a flat `(N, 3)` row-major
/// `Vec<f32>` with `3 * N` elements.
pub fn apply_transform_4x4(local: &[f64], transform: &[f64]) -> Vec<f32> {
    debug_assert_eq!(local.len() % 3, 0, "local must be (N, 3)");
    debug_assert_eq!(transform.len(), 16, "transform must be 4x4 (16 elements)");
    let n = local.len() / 3;
    // Row-major 4x4: index (r, c) = r*4 + c. We need the upper-left 3x3
    // and the translation column at indices (0,3), (1,3), (2,3).
    let r00 = transform[0];
    let r01 = transform[1];
    let r02 = transform[2];
    let tx = transform[3];
    let r10 = transform[4];
    let r11 = transform[5];
    let r12 = transform[6];
    let ty = transform[7];
    let r20 = transform[8];
    let r21 = transform[9];
    let r22 = transform[10];
    let tz = transform[11];

    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let x = local[3 * i];
        let y = local[3 * i + 1];
        let z = local[3 * i + 2];
        // world = local @ R^T + t (R is 3x3 upper-left), per row.
        let wx = r00 * x + r01 * y + r02 * z + tx;
        let wy = r10 * x + r11 * y + r12 * z + ty;
        let wz = r20 * x + r21 * y + r22 * z + tz;
        out.push(wx as f32);
        out.push(wy as f32);
        out.push(wz as f32);
    }
    out
}

/// One tetrahedralization job descriptor used by the build-plan
/// summary.
pub struct TetraJob<'a> {
    pub name: &'a str,
    pub cached: bool,
}

/// Format a list of tetrahedralization jobs into the human-readable
/// status line shown during build planning.
pub fn summarize_tetra_jobs(jobs: &[TetraJob<'_>]) -> String {
    if jobs.is_empty() {
        return "no solid tetrahedralizations".to_string();
    }
    let labels: Vec<String> = jobs
        .iter()
        .map(|j| {
            format!(
                "{} ({})",
                j.name,
                if j.cached { "cached" } else { "new" }
            )
        })
        .collect();
    if labels.len() > 5 {
        let head = labels[..5].join(", ");
        let extra = labels.len() - 5;
        return format!(
            "{} tetrahedralizations: {}, +{} more",
            labels.len(),
            head,
            extra
        );
    }
    format!(
        "{} tetrahedralizations: {}",
        labels.len(),
        labels.join(", ")
    )
}

/// Result of [`barycentric_project_anchors`]: each anchor maps to the
/// closest triangle `(v0, v1, v2)` in `tri` and its barycentric weights
/// `(alpha, beta, gamma)` in `bary`. This is a pure single-mesh
/// projection: the 6-slot stitch row is assembled by the caller, which
/// places one side's `(tri, bary)` into slots `[0..2]` (source) and the
/// other into `[3..5]` (target). Anchors on a mesh with no valid faces
/// are dropped (all-or-nothing: a mesh with >= 1 valid face keeps every
/// anchor in input order, so source and target projections stay aligned
/// per row).
pub struct StitchRows {
    pub tri: Vec<[i64; 3]>,
    pub bary: Vec<[f32; 3]>,
}

/// Project each anchor onto the closest triangle of one mesh and emit
/// its triangle + barycentric weights.
///
/// `target_face` is row-major `(F, 3)`. `target_pos` is row-major
/// `(P, 3)` (world space already). `anchors` is row-major `(K, 3)`.
/// Output rows are in input order; rows where every face is degenerate
/// are dropped (which only happens if the whole mesh has no valid face,
/// dropping all anchors at once).
pub fn barycentric_project_anchors(
    target_face: &[i64],
    target_pos: &[f64],
    anchors: &[f64],
) -> StitchRows {
    debug_assert_eq!(target_face.len() % 3, 0, "target_face must be (F, 3)");
    debug_assert_eq!(target_pos.len() % 3, 0, "target_pos must be (P, 3)");
    debug_assert_eq!(anchors.len() % 3, 0, "anchors must be (K, 3)");
    let n_face = target_face.len() / 3;
    let n_anchor = anchors.len() / 3;

    // Pre-compute per-face geometry.
    let mut p0 = vec![[0.0f64; 3]; n_face];
    let mut p1 = vec![[0.0f64; 3]; n_face];
    let mut p2 = vec![[0.0f64; 3]; n_face];
    let mut e1 = vec![[0.0f64; 3]; n_face];
    let mut e2 = vec![[0.0f64; 3]; n_face];
    let mut d00 = vec![0.0f64; n_face];
    let mut d01 = vec![0.0f64; n_face];
    let mut d11 = vec![0.0f64; n_face];
    let mut denom = vec![0.0f64; n_face];
    let mut valid = vec![false; n_face];
    let mut safe_denom = vec![1.0f64; n_face];

    for f in 0..n_face {
        let i0 = target_face[3 * f] as usize;
        let i1 = target_face[3 * f + 1] as usize;
        let i2 = target_face[3 * f + 2] as usize;
        for k in 0..3 {
            p0[f][k] = target_pos[3 * i0 + k];
            p1[f][k] = target_pos[3 * i1 + k];
            p2[f][k] = target_pos[3 * i2 + k];
            e1[f][k] = p1[f][k] - p0[f][k];
            e2[f][k] = p2[f][k] - p0[f][k];
        }
        d00[f] = e1[f][0] * e1[f][0] + e1[f][1] * e1[f][1] + e1[f][2] * e1[f][2];
        d01[f] = e1[f][0] * e2[f][0] + e1[f][1] * e2[f][1] + e1[f][2] * e2[f][2];
        d11[f] = e2[f][0] * e2[f][0] + e2[f][1] * e2[f][1] + e2[f][2] * e2[f][2];
        denom[f] = d00[f] * d11[f] - d01[f] * d01[f];
        valid[f] = denom[f].abs() > 1e-20;
        safe_denom[f] = if valid[f] { denom[f] } else { 1.0 };
    }

    let mut tri_out: Vec<[i64; 3]> = Vec::new();
    let mut bary_out: Vec<[f32; 3]> = Vec::new();

    for a in 0..n_anchor {
        let sv = [
            anchors[3 * a],
            anchors[3 * a + 1],
            anchors[3 * a + 2],
        ];
        let mut best_fi: usize = 0;
        let mut best_dist = f64::INFINITY;
        let mut best_alpha = 0.0f64;
        let mut best_beta = 0.0f64;
        let mut best_gamma = 0.0f64;
        for f in 0..n_face {
            let v0x = sv[0] - p0[f][0];
            let v0y = sv[1] - p0[f][1];
            let v0z = sv[2] - p0[f][2];
            let d20 = v0x * e1[f][0] + v0y * e1[f][1] + v0z * e1[f][2];
            let d21 = v0x * e2[f][0] + v0y * e2[f][1] + v0z * e2[f][2];
            let beta = if valid[f] {
                (d11[f] * d20 - d01[f] * d21) / safe_denom[f]
            } else {
                -1.0
            };
            let gamma = if valid[f] {
                (d00[f] * d21 - d01[f] * d20) / safe_denom[f]
            } else {
                -1.0
            };
            let alpha = 1.0 - beta - gamma;
            // Clamp to the triangle and project; the per-face 1e-20
            // degeneracy gate (`valid[f]`) stays out here so the shared
            // helper only runs the convex-projection tail.
            let (bary, _proj, dist) =
                barycentric_clamp_project(alpha, beta, gamma, sv, p0[f], p1[f], p2[f]);
            let dist = if valid[f] { dist } else { f64::INFINITY };
            if dist < best_dist {
                best_dist = dist;
                best_fi = f;
                best_alpha = bary[0];
                best_beta = bary[1];
                best_gamma = bary[2];
            }
        }
        if best_dist == f64::INFINITY {
            continue;
        }
        let fv0 = target_face[3 * best_fi];
        let fv1 = target_face[3 * best_fi + 1];
        let fv2 = target_face[3 * best_fi + 2];
        tri_out.push([fv0, fv1, fv2]);
        bary_out.push([best_alpha as f32, best_beta as f32, best_gamma as f32]);
    }
    StitchRows { tri: tri_out, bary: bary_out }
}

/// Compute the per-Blender-vertex "best simulation vertex" mapping
/// from a surface frame map.
///
/// Inputs:
/// - `tri_indices`: `(M,)` triangle id per Blender surface vertex.
/// - `coefs`: row-major `(M, 3)` frame coefficients (c1, c2, c3).
/// - `faces`: row-major `(NF, 3)` triangle vertex indices.
/// - `verts`: row-major `(NV, 3)` tet surface positions in the same
///   coordinate space as `coefs` (i.e. local-space, since the surface
///   map is computed on the local tet). Passing world-space verts under
///   a non-uniform object scale produces a `bp` that disagrees with the
///   actual world position by tens of centimeters and can pick the
///   wrong corner.
///
/// Output: `Vec<i64>` of length `M`, the closest simulation surface
/// vertex (one of the triangle's three) for each Blender vertex.
pub fn solid_orig_to_sim(
    tri_indices: &[i64],
    coefs: &[f64],
    faces: &[i64],
    verts: &[f64],
) -> Vec<i64> {
    debug_assert_eq!(coefs.len(), tri_indices.len() * 3);
    debug_assert_eq!(faces.len() % 3, 0);
    debug_assert_eq!(verts.len() % 3, 0);
    let m = tri_indices.len();
    let mut out = Vec::with_capacity(m);
    for bi in 0..m {
        let ti = tri_indices[bi] as usize;
        let i0 = faces[3 * ti] as usize;
        let i1 = faces[3 * ti + 1] as usize;
        let i2 = faces[3 * ti + 2] as usize;
        let c0 = coefs[3 * bi];
        let c1 = coefs[3 * bi + 1];
        let c2 = coefs[3 * bi + 2];
        let x0 = [verts[3 * i0], verts[3 * i0 + 1], verts[3 * i0 + 2]];
        let p1 = [verts[3 * i1], verts[3 * i1 + 1], verts[3 * i1 + 2]];
        let p2v = [verts[3 * i2], verts[3 * i2 + 1], verts[3 * i2 + 2]];
        let b1 = [p1[0] - x0[0], p1[1] - x0[1], p1[2] - x0[2]];
        let b2 = [p2v[0] - x0[0], p2v[1] - x0[1], p2v[2] - x0[2]];
        // n = b1 x b2
        let n = [
            b1[1] * b2[2] - b1[2] * b2[1],
            b1[2] * b2[0] - b1[0] * b2[2],
            b1[0] * b2[1] - b1[1] * b2[0],
        ];
        let nlen = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let n_hat = if nlen > 1e-10 {
            [n[0] / nlen, n[1] / nlen, n[2] / nlen]
        } else {
            [0.0, 0.0, 0.0]
        };
        let bp = [
            x0[0] + c0 * b1[0] + c1 * b2[0] + c2 * n_hat[0],
            x0[1] + c0 * b1[1] + c1 * b2[1] + c2 * n_hat[1],
            x0[2] + c0 * b1[2] + c1 * b2[2] + c2 * n_hat[2],
        ];
        let cand = [(i0, x0), (i1, p1), (i2, p2v)];
        let mut best_dist = f64::INFINITY;
        let mut best = i0 as i64;
        for (idx, p) in cand.iter() {
            let dx = p[0] - bp[0];
            let dy = p[1] - bp[1];
            let dz = p[2] - bp[2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < best_dist {
                best_dist = d;
                best = *idx as i64;
            }
        }
        out.push(best);
    }
    out
}

/// Walk a `(P, 3)` row-major buffer and return the index of the row
/// closest (squared L2) to `target`.
pub fn closest_vertex_index(verts: &[f64], target: &[f64]) -> usize {
    debug_assert_eq!(verts.len() % 3, 0);
    debug_assert_eq!(target.len(), 3);
    let n = verts.len() / 3;
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    let tx = target[0];
    let ty = target[1];
    let tz = target[2];
    for i in 0..n {
        let dx = verts[3 * i] - tx;
        let dy = verts[3 * i + 1] - ty;
        let dz = verts[3 * i + 2] - tz;
        let d = dx * dx + dy * dy + dz * dz;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Compute per-segment translation deltas for a pin's
/// transform-keyframe playback.
///
/// `times` length is `K`; `positions` is row-major `(K, 3)`. Returns
/// a vector of `(t_start, t_end, [dx, dy, dz])` for each consecutive
/// pair, with the start anchored to the previous keyframe time.
pub fn keyframe_translation_segments(
    times: &[f64],
    positions: &[f64],
) -> Vec<(f64, f64, [f64; 3])> {
    debug_assert_eq!(positions.len(), times.len() * 3);
    if times.len() < 2 {
        return Vec::new();
    }
    let mut out: Vec<(f64, f64, [f64; 3])> = Vec::with_capacity(times.len() - 1);
    for i in 1..times.len() {
        let prev_t = times[i - 1];
        let cur_t = times[i];
        let dx = positions[3 * i] - positions[3 * (i - 1)];
        let dy = positions[3 * i + 1] - positions[3 * (i - 1) + 1];
        let dz = positions[3 * i + 2] - positions[3 * (i - 1) + 2];
        out.push((prev_t, cur_t, [dx, dy, dz]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_transform_identity() {
        let local = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        #[rustfmt::skip]
        let m = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let out = apply_transform_4x4(&local, &m);
        assert_eq!(out, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn apply_transform_translation_and_scale() {
        // Diagonal scale (2, 3, 4) plus translation (10, 20, 30).
        let local = vec![1.0, 1.0, 1.0];
        #[rustfmt::skip]
        let m = vec![
            2.0, 0.0, 0.0, 10.0,
            0.0, 3.0, 0.0, 20.0,
            0.0, 0.0, 4.0, 30.0,
            0.0, 0.0, 0.0,  1.0,
        ];
        let out = apply_transform_4x4(&local, &m);
        assert_eq!(out, vec![12.0_f32, 23.0, 34.0]);
    }

    #[test]
    fn apply_transform_off_diagonal() {
        // Rotate 90deg around +Z so x -> y, y -> -x.
        let local = vec![1.0, 0.0, 0.0];
        #[rustfmt::skip]
        let m = vec![
            0.0, -1.0, 0.0, 0.0,
            1.0,  0.0, 0.0, 0.0,
            0.0,  0.0, 1.0, 0.0,
            0.0,  0.0, 0.0, 1.0,
        ];
        let out = apply_transform_4x4(&local, &m);
        // (1, 0, 0) -> (0, 1, 0)
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn summary_empty() {
        assert_eq!(summarize_tetra_jobs(&[]), "no solid tetrahedralizations");
    }

    #[test]
    fn summary_short_list() {
        let jobs = vec![
            TetraJob { name: "a", cached: true },
            TetraJob { name: "b", cached: false },
        ];
        let s = summarize_tetra_jobs(&jobs);
        assert_eq!(s, "2 tetrahedralizations: a (cached), b (new)");
    }

    #[test]
    fn summary_truncates_after_five() {
        let jobs: Vec<TetraJob> = (0..7)
            .map(|i| TetraJob {
                name: ["a", "b", "c", "d", "e", "f", "g"][i],
                cached: i % 2 == 0,
            })
            .collect();
        let s = summarize_tetra_jobs(&jobs);
        // Five labels then "+2 more"
        assert!(s.starts_with("7 tetrahedralizations: "));
        assert!(s.ends_with(", +2 more"));
        // Should mention exactly the first five names, in order.
        assert!(s.contains("a (cached)"));
        assert!(s.contains("e (cached)"));
        assert!(!s.contains("g ("));
    }

    #[test]
    fn barycentric_project_single_triangle() {
        // Single triangle in the XY plane.
        let face = vec![0_i64, 1, 2];
        let pos = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        // Anchor exactly at the first vertex => alpha=1.
        let anchors = vec![0.0, 0.0, 0.0];
        let r = barycentric_project_anchors(&face, &pos, &anchors);
        assert_eq!(r.tri.len(), 1);
        assert_eq!(r.bary.len(), 1);
        assert_eq!(r.tri[0], [0, 1, 2]);
        // alpha=1, beta=0, gamma=0.
        assert!((r.bary[0][0] - 1.0).abs() < 1e-5);
        assert!((r.bary[0][1] - 0.0).abs() < 1e-5);
        assert!((r.bary[0][2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn barycentric_project_off_plane_anchor() {
        // Triangle in z=0; anchor at (0.5, 0.25, 0.5) projects to
        // (0.5, 0.25, 0) and lies inside the triangle.
        let face = vec![0_i64, 1, 2];
        let pos = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let anchors = vec![0.5, 0.25, 0.5];
        let r = barycentric_project_anchors(&face, &pos, &anchors);
        assert_eq!(r.tri.len(), 1);
        assert_eq!(r.tri[0], [0, 1, 2]);
        // alpha=0.25, beta=0.5, gamma=0.25 (from the projection,
        // which clamps then renormalizes; here all in-range).
        let b = r.bary[0];
        assert!((b[0] - 0.25).abs() < 1e-5);
        assert!((b[1] - 0.5).abs() < 1e-5);
        assert!((b[2] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn barycentric_project_drops_degenerate() {
        // Two degenerate faces (collinear) -> all faces invalid -> drop.
        let face = vec![0_i64, 1, 2, 0, 1, 2];
        let pos = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0, // collinear
        ];
        let anchors = vec![0.5, 0.0, 0.0];
        let r = barycentric_project_anchors(&face, &pos, &anchors);
        assert_eq!(r.tri.len(), 0);
    }

    #[test]
    fn solid_orig_to_sim_picks_closest() {
        // One Blender vertex anchored exactly to triangle vertex 1.
        let tri_indices = vec![0_i64];
        // c1=1 means blender_pos = x0 + 1*b1 = vertex 1.
        let coefs = vec![1.0, 0.0, 0.0];
        let faces = vec![0_i64, 1, 2];
        let verts = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let out = solid_orig_to_sim(&tri_indices, &coefs, &faces, &verts);
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn solid_orig_to_sim_centroid_near_first() {
        // Blender pos near vertex 0 (alpha ~ 0).
        let tri_indices = vec![0_i64];
        let coefs = vec![0.05, 0.05, 0.0];
        let faces = vec![0_i64, 1, 2];
        let verts = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];
        let out = solid_orig_to_sim(&tri_indices, &coefs, &faces, &verts);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn closest_vertex_index_picks_nearest() {
        let verts = vec![
            0.0, 0.0, 0.0,
            10.0, 0.0, 0.0,
            5.0, 1.0, 0.0,
        ];
        assert_eq!(closest_vertex_index(&verts, &[5.5, 1.0, 0.0]), 2);
        assert_eq!(closest_vertex_index(&verts, &[-1.0, -1.0, 0.0]), 0);
    }

    #[test]
    fn keyframe_translation_segments_basic() {
        let times = vec![0.0, 1.0, 2.5];
        let positions = vec![
            1.0, 2.0, 3.0,
            1.0, 4.0, 3.0,
            -1.0, 4.0, 3.0,
        ];
        let segs = keyframe_translation_segments(&times, &positions);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], (0.0, 1.0, [0.0, 2.0, 0.0]));
        assert_eq!(segs[1], (1.0, 2.5, [-2.0, 0.0, 0.0]));
    }

    #[test]
    fn keyframe_translation_segments_short_returns_empty() {
        assert!(keyframe_translation_segments(&[0.0], &[1.0, 2.0, 3.0]).is_empty());
        assert!(keyframe_translation_segments(&[], &[]).is_empty());
    }
}
