// File: crates/ppf-cts-core/src/kernels/intersection.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Host-side self-intersection detection for the scene-build check.
//
// The edge-triangle pierce predicate is NOT reimplemented here: it is a
// single source of truth shared with the GPU contact kernels. The same
// templated routine in the solver's C++ tree
// (crates/ppf-cts-solver/src/cpp/contact/intersect_core.hpp) is
// instantiated for float on the device and for double here, reached via
// the `extern "C"` shim in crates/ppf-cts-core/cpp/intersect_ffi.cpp
// (compiled by build.rs). This module owns only the broad phase, the
// topological skip rules, and the coplanar-overlap fallback:
//
//   * `edge_triangle_intersect` (FFI to the shared device predicate).
//   * `triangles_coplanar_overlap` (2D-projection fallback; host only).
//   * `_segments_intersect_2d`, `_point_in_triangle_2d`.
//   * `_find_edge_tri_intersections` (BVH traversal with skip rules).
//
// Pipeline:
//   1. Extract unique mesh edges + edge_to_tri (parents per edge).
//   2. If rod_edges given, append them with parent = (-1, -1).
//   3. Build tri-BVH; compute per-edge bboxes.
//   4. rayon-parallel scan over edges → (edge_idx, tri_idx) hits.
//   5. Convert (edge, tri) hits to deduplicated, sorted (parent, tri)
//      pairs. Rod hits emerge as (-1, tri_idx) and append after.
//
// Skip rules (preserved from Python source):
//   * triangle == parent of edge: skipped (sharing topology).
//   * pairs sharing any vertex: skipped.
//   * collider × collider: skipped (where edge is "collider" iff
//     either of its parent triangles is flagged).

use std::collections::BTreeSet;

use rayon::prelude::*;

use super::bvh::{
    build_tri_bvh, compute_edge_bboxes, extract_edges_with_tri_map, traverse_overlap, Bvh,
};
use super::geom_util::{
    bbox_overlap, cross3, dot3, elements_share_vertex_2_3, elements_share_vertex_3_3, sub3, vert3,
};

const COPLANAR_EPS: f64 = 1e-10;
const SEGMENT_2D_EPS_PARAMETER: f64 = 1e-10;
const SEGMENT_2D_EPS_CROSS: f64 = 1e-14;

// ---------------------------------------------------------------------------
// Edge-triangle intersection primitive (shared with the GPU).

extern "C" {
    /// Single source of truth for the edge-triangle pierce test, defined
    /// in crates/ppf-cts-solver/src/cpp/contact/intersect_core.hpp and
    /// linked via the host shim compiled by build.rs. The device contact
    /// kernels call the same routine (instantiated for float); this is the
    /// double instantiation, so the host build-check runs it in f64. Each
    /// pointer addresses three contiguous `f64` coordinates.
    fn ppf_isect_edge_triangle_intersect(
        e0: *const f64,
        e1: *const f64,
        v0: *const f64,
        v1: *const f64,
        v2: *const f64,
    ) -> bool;
}

/// True iff segment `(e0, e1)` strictly crosses triangle `(v0, v1, v2)`.
/// Coplanar / touching cases return false; those are handled by
/// `triangles_coplanar_overlap` separately. Thin wrapper over the shared
/// device predicate (see the `extern "C"` block above).
#[inline]
fn edge_triangle_intersect(
    e0: [f64; 3],
    e1: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> bool {
    // Safety: every pointer addresses a live `[f64; 3]` on this stack
    // frame; the C routine only reads three doubles from each and does not
    // retain them.
    unsafe {
        ppf_isect_edge_triangle_intersect(
            e0.as_ptr(),
            e1.as_ptr(),
            v0.as_ptr(),
            v1.as_ptr(),
            v2.as_ptr(),
        )
    }
}

/// 2D segment intersection in the (open-interval) interior. Used
/// only by the coplanar fallback.
#[inline]
fn segments_intersect_2d(
    p1x: f64,
    p1y: f64,
    p2x: f64,
    p2y: f64,
    p3x: f64,
    p3y: f64,
    p4x: f64,
    p4y: f64,
) -> bool {
    let d1x = p2x - p1x;
    let d1y = p2y - p1y;
    let d2x = p4x - p3x;
    let d2y = p4y - p3y;
    let cross = d1x * d2y - d1y * d2x;
    if cross.abs() < SEGMENT_2D_EPS_CROSS {
        return false;
    }
    let t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / cross;
    let u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / cross;
    let eps = SEGMENT_2D_EPS_PARAMETER;
    eps < t && t < 1.0 - eps && eps < u && u < 1.0 - eps
}

#[inline]
fn point_in_triangle_2d(
    px: f64,
    py: f64,
    v0x: f64,
    v0y: f64,
    v1x: f64,
    v1y: f64,
    v2x: f64,
    v2y: f64,
) -> bool {
    #[inline(always)]
    fn sign(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
        (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    }
    let d1 = sign(px, py, v0x, v0y, v1x, v1y);
    let d2 = sign(px, py, v1x, v1y, v2x, v2y);
    let d3 = sign(px, py, v2x, v2y, v0x, v0y);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Coplanar-overlap fallback.
#[inline]
fn triangles_coplanar_overlap(
    t0_v: [[f64; 3]; 3],
    t1_v: [[f64; 3]; 3],
) -> bool {
    let e1 = sub3(t0_v[1], t0_v[0]);
    let e2 = sub3(t0_v[2], t0_v[0]);
    let n = cross3(e1, e2);
    let n_len_sq = dot3(n, n);
    let n_len = n_len_sq.sqrt();
    if n_len < COPLANAR_EPS {
        return false;
    }
    // Coplanarity check, scale-aware.
    let d0 = dot3(n, sub3(t1_v[0], t0_v[0])).abs();
    let d1 = dot3(n, sub3(t1_v[1], t0_v[0])).abs();
    let d2 = dot3(n, sub3(t1_v[2], t0_v[0])).abs();
    let thresh = n_len * COPLANAR_EPS * 100.0;
    if d0 > thresh || d1 > thresh || d2 > thresh {
        return false;
    }

    // Project to 2D using the largest normal component as the dropped axis.
    let an = [n[0].abs(), n[1].abs(), n[2].abs()];
    let (ax, ay) = if an[0] > an[1] && an[0] > an[2] {
        (1usize, 2usize)
    } else if an[1] > an[2] {
        (0, 2)
    } else {
        (0, 1)
    };
    let u = [
        (t0_v[0][ax], t0_v[0][ay]),
        (t0_v[1][ax], t0_v[1][ay]),
        (t0_v[2][ax], t0_v[2][ay]),
    ];
    let v = [
        (t1_v[0][ax], t1_v[0][ay]),
        (t1_v[1][ax], t1_v[1][ay]),
        (t1_v[2][ax], t1_v[2][ay]),
    ];

    // 9 edge × edge intersection tests.
    let t0_edges = [(u[0], u[1]), (u[1], u[2]), (u[2], u[0])];
    let t1_edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];
    for &(a, b) in &t0_edges {
        for &(c, d) in &t1_edges {
            if segments_intersect_2d(a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1) {
                return true;
            }
        }
    }

    // Vertex containment tests in both directions.
    for &(vx, vy) in &v {
        if point_in_triangle_2d(vx, vy, u[0].0, u[0].1, u[1].0, u[1].1, u[2].0, u[2].1) {
            return true;
        }
    }
    for &(ux, uy) in &u {
        if point_in_triangle_2d(ux, uy, v[0].0, v[0].1, v[1].0, v[1].1, v[2].0, v[2].1) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Per-edge BVH traversal.

#[inline]
fn find_edge_tri_intersections(
    ei: usize,
    verts: &[f64],
    edges: &[i32],
    tris: &[i32],
    edge_to_tri: &[i32],
    edge_bb_min: &[[f64; 3]],
    edge_bb_max: &[[f64; 3]],
    tri_bvh: &Bvh,
    is_collider_tri: &[bool],
    tri_body_id: &[i32],
) -> Vec<(i32, i32)> {
    let edge = [edges[2 * ei], edges[2 * ei + 1]];
    let e0 = vert3(verts, edge[0]);
    let e1 = vert3(verts, edge[1]);
    let bb_min = &edge_bb_min[ei];
    let bb_max = &edge_bb_max[ei];

    let parent_tri0 = edge_to_tri[2 * ei];
    let parent_tri1 = edge_to_tri[2 * ei + 1];
    let has_parent = parent_tri0 >= 0;

    // Parent triangle vertices (only used by the coplanar fallback and
    // only when has_parent). Init with degenerate values so the
    // fallback short-circuits if has_parent is false.
    let parent_tri = if has_parent {
        [
            tris[3 * parent_tri0 as usize],
            tris[3 * parent_tri0 as usize + 1],
            tris[3 * parent_tri0 as usize + 2],
        ]
    } else {
        [0, 0, 0]
    };
    let p_v = if has_parent {
        [
            vert3(verts, parent_tri[0]),
            vert3(verts, parent_tri[1]),
            vert3(verts, parent_tri[2]),
        ]
    } else {
        [e0, e1, e0]
    };

    // The edge is "collider" iff either of its parent triangles is.
    let edge_is_collider = (parent_tri0 >= 0 && is_collider_tri[parent_tri0 as usize])
        || (parent_tri1 >= 0 && is_collider_tri[parent_tri1 as usize]);

    // The edge's PDRD rigid-body id (0 = not a member of any body). Every
    // triangle of a body carries the same id, so either parent reports it;
    // a rod edge (parent < 0) reports 0. Used to skip intra-body pairs.
    let edge_body_id = if parent_tri0 >= 0 {
        tri_body_id[parent_tri0 as usize]
    } else if parent_tri1 >= 0 {
        tri_body_id[parent_tri1 as usize]
    } else {
        0
    };

    let mut out = Vec::new();
    traverse_overlap(tri_bvh, bb_min, bb_max, |ti| {
        if ti == parent_tri0 || ti == parent_tri1 {
            return;
        }
        if edge_is_collider && is_collider_tri[ti as usize] {
            return;
        }
        // Skip intra-PDRD-body edge/triangle pairs: a rigid body never
        // deforms, so an edge piercing a triangle of the SAME body is a
        // fixed, physically meaningless self-intersection that must be
        // tolerated even when the body starts self-tangled. Mirrors the
        // device-side same_pdrd_body filter in contact.cu.
        if edge_body_id != 0 && edge_body_id == tri_body_id[ti as usize] {
            return;
        }
        let tri = [
            tris[3 * ti as usize],
            tris[3 * ti as usize + 1],
            tris[3 * ti as usize + 2],
        ];
        if elements_share_vertex_2_3(edge, tri) {
            return;
        }
        if !bbox_overlap(
            bb_min,
            bb_max,
            &tri_bvh.elem_bboxes_min[ti as usize],
            &tri_bvh.elem_bboxes_max[ti as usize],
        ) {
            return;
        }

        let v0 = vert3(verts, tri[0]);
        let v1 = vert3(verts, tri[1]);
        let v2 = vert3(verts, tri[2]);
        let mut intersects = edge_triangle_intersect(e0, e1, v0, v1, v2);

        // Coplanar fallback only when there's a real parent
        // triangle (i.e., this isn't a rod edge) AND the
        // parent doesn't already share vertices with the
        // candidate (those would be filtered topologically
        // and never count as a real coplanar overlap).
        if !intersects && has_parent && !elements_share_vertex_3_3(parent_tri, tri) {
            intersects = triangles_coplanar_overlap(p_v, [v0, v1, v2]);
        }

        if intersects {
            out.push((ei as i32, ti));
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Public API

pub struct IntersectionInput<'a> {
    pub verts: &'a [f64],
    pub tris: &'a [i32],
    pub is_collider: Option<&'a [bool]>,
    pub rod_edges: Option<&'a [i32]>,
    /// Optional per-triangle 1-based PDRD rigid-body id (0 = not a member
    /// of any rigid body). Parallel to `tris`. When two triangles share
    /// the same nonzero id they belong to the same rigid body, which
    /// never deforms; an edge of one piercing the other is a fixed,
    /// physically meaningless self-intersection and is skipped. Mirrors
    /// the device-side `same_pdrd_body` filter in contact.cu. `None`
    /// disables the rule (every triangle treated as body 0).
    pub tri_body_id: Option<&'a [i32]>,
}

/// Check a triangle mesh for self-intersections, optionally including
/// rod edges. Output is sorted, deduplicated `(i, j)` pairs with `i <
/// j`. Rod-triangle hits are appended at the end as `(-1, tri_idx)`.
pub fn check_self_intersection(input: IntersectionInput<'_>) -> Vec<(i32, i32)> {
    let n_tris = input.tris.len() / 3;
    let n_rod_edges = input.rod_edges.map(|r| r.len() / 2).unwrap_or(0);
    if n_tris < 2 && n_rod_edges == 0 {
        return vec![];
    }

    // Step 1: derive mesh edges + per-edge parents.
    let (mesh_edges, mesh_e2t) = extract_edges_with_tri_map(input.tris);
    let n_mesh_edges = mesh_edges.len() / 2;

    // Step 2: append rod edges with parent (-1, -1). When there are
    // rod edges, consume the mesh-only Vecs by value and extend in
    // place; this avoids allocating ~2 * n_mesh_edges * size_of::<i32>()
    // of intermediate state per call (~1MB at 100k mesh edges).
    let (all_edges, all_e2t): (Vec<i32>, Vec<i32>) = if n_rod_edges > 0 {
        let mut e = mesh_edges;
        let mut t = mesh_e2t;
        e.extend_from_slice(input.rod_edges.unwrap());
        for _ in 0..n_rod_edges {
            t.push(-1);
            t.push(-1);
        }
        (e, t)
    } else {
        (mesh_edges, mesh_e2t)
    };

    // Step 3: build tri-BVH; per-edge bboxes only (no edge-BVH needed,
    // the algorithm scans tri-BVH for each edge).
    let tri_bvh = build_tri_bvh(input.verts, input.tris);
    let (edge_bb_min, edge_bb_max) = compute_edge_bboxes(input.verts, &all_edges);

    // Step 4: default is_collider and per-tri PDRD body id.
    let zero_collider: Vec<bool>;
    let is_collider = match input.is_collider {
        Some(s) => s,
        None => {
            zero_collider = vec![false; n_tris];
            &zero_collider
        }
    };
    let zero_body_id: Vec<i32>;
    let tri_body_id = match input.tri_body_id {
        Some(s) => s,
        None => {
            zero_body_id = vec![0; n_tris];
            &zero_body_id
        }
    };

    // Step 5: per-edge parallel scan into (edge_idx, tri_idx) hits.
    let n_edges = all_edges.len() / 2;
    let edge_tri_pairs: Vec<(i32, i32)> = (0..n_edges)
        .into_par_iter()
        .flat_map(|ei| {
            find_edge_tri_intersections(
                ei,
                input.verts,
                &all_edges,
                input.tris,
                &all_e2t,
                &edge_bb_min,
                &edge_bb_max,
                &tri_bvh,
                is_collider,
                tri_body_id,
            )
        })
        .collect();

    // Step 6: convert (edge, tri) → (parent, tri); dedup; sort. Rod
    // hits collected separately and appended after triangle-triangle
    // pairs.
    let mut tri_pairs: BTreeSet<(i32, i32)> = BTreeSet::new();
    let mut rod_pairs: Vec<(i32, i32)> = Vec::new();
    for (ei, ti) in edge_tri_pairs {
        let ei_us = ei as usize;
        if ei_us >= n_mesh_edges {
            // Rod edge.
            rod_pairs.push((-1, ti));
        } else {
            let p0 = all_e2t[2 * ei_us];
            let p1 = all_e2t[2 * ei_us + 1];
            if p0 >= 0 && p0 != ti {
                tri_pairs.insert((p0.min(ti), p0.max(ti)));
            }
            if p1 >= 0 && p1 != ti {
                tri_pairs.insert((p1.min(ti), p1.max(ti)));
            }
        }
    }

    let mut out: Vec<(i32, i32)> = tri_pairs.into_iter().collect();
    out.extend(rod_pairs);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat3(rows: &[[f64; 3]]) -> Vec<f64> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }
    fn flat3i(rows: &[[i32; 3]]) -> Vec<i32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    #[test]
    fn crossing_triangles_detected() {
        // Two triangles whose interiors cross.
        let verts = flat3(&[
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: None,
        });
        assert_eq!(r, vec![(0, 1)]);
    }

    #[test]
    fn coplanar_overlapping_triangles_detected() {
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0],
            [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 2.0, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: None,
        });
        assert_eq!(r, vec![(0, 1)]);
    }

    #[test]
    fn coplanar_non_overlapping_skipped() {
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0],
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 0.5, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: None,
        });
        assert!(r.is_empty());
    }

    #[test]
    fn adjacent_triangles_skipped() {
        // Sharing edge 0-1: must be skipped by the share-vertex filter.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [0, 1, 3]]);
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: None,
        });
        assert!(r.is_empty());
    }

    #[test]
    fn near_touching_not_intersecting() {
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.01], [1.0, 0.0, 0.01], [0.5, 1.0, 0.01],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: None,
        });
        assert!(r.is_empty());
    }

    #[test]
    fn collider_collider_skipped() {
        let verts = flat3(&[
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        let coll_both = vec![true, true];
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: Some(&coll_both),
            rod_edges: None,
            tri_body_id: None,
        });
        assert!(r.is_empty(), "collider × collider must be skipped");

        let coll_one = vec![true, false];
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: Some(&coll_one),
            rod_edges: None,
            tri_body_id: None,
        });
        assert_eq!(r.len(), 1, "mixed collider/dynamic must still report");
    }

    #[test]
    fn same_pdrd_body_skipped() {
        // Two interior-crossing triangles. When both belong to the same
        // nonzero PDRD body, the pair is a tolerated intra-rigid-body
        // self-intersection and must be skipped; distinct bodies (or a
        // body vs. body-0) must still report.
        let verts = flat3(&[
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 2.0, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);

        // Same body (id 1 on both): skipped.
        let same_body = vec![1, 1];
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: Some(&same_body),
        });
        assert!(r.is_empty(), "same PDRD body must be skipped");

        // Distinct bodies: still reported.
        let diff_body = vec![1, 2];
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: Some(&diff_body),
        });
        assert_eq!(r.len(), 1, "distinct PDRD bodies must still report");

        // Body vs. non-PDRD (id 0): still reported.
        let one_body = vec![1, 0];
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: None,
            tri_body_id: Some(&one_body),
        });
        assert_eq!(r.len(), 1, "rigid body vs. non-rigid must still report");
    }

    #[test]
    fn rod_edge_detected() {
        // One triangle in xy-plane; rod edge piercing from below to above.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [0.25, 0.25, -1.0], [0.25, 0.25, 1.0],
        ]);
        let tris = flat3i(&[[0, 1, 2]]);
        let rod = vec![3i32, 4];
        let r = check_self_intersection(IntersectionInput {
            verts: &verts,
            tris: &tris,
            is_collider: None,
            rod_edges: Some(&rod),
            tri_body_id: None,
        });
        assert_eq!(r, vec![(-1, 0)]);
    }

    // ----- Direct primitive tests -----

    #[test]
    fn edge_triangle_intersect_pierce_and_above() {
        // A triangle in the xy-plane. A vertical edge piercing it
        // through the centroid must intersect.
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let centroid = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let e0 = [centroid[0], centroid[1], -0.5];
        let e1 = [centroid[0], centroid[1], 0.5];
        assert!(edge_triangle_intersect(e0, e1, v0, v1, v2));
        // An edge entirely above the plane must not intersect.
        let e0 = [centroid[0], centroid[1], 0.1];
        let e1 = [centroid[0], centroid[1], 0.5];
        assert!(!edge_triangle_intersect(e0, e1, v0, v1, v2));
        // An edge that misses the triangle laterally.
        let e0 = [5.0, 5.0, -1.0];
        let e1 = [5.0, 5.0, 1.0];
        assert!(!edge_triangle_intersect(e0, e1, v0, v1, v2));
    }

    #[test]
    fn segments_intersect_2d_cross_and_parallel() {
        // Crossing segments at origin.
        assert!(segments_intersect_2d(-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0));
        // Parallel segments never intersect.
        assert!(!segments_intersect_2d(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        // Endpoint-touching does NOT count (open interval).
        assert!(!segments_intersect_2d(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn point_in_triangle_2d_basic() {
        assert!(point_in_triangle_2d(0.25, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0));
        assert!(!point_in_triangle_2d(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn triangles_coplanar_overlap_detects_partial_overlap() {
        // Two coplanar overlapping triangles (z = 0).
        let t0 = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let t1 = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 2.0, 0.0]];
        assert!(triangles_coplanar_overlap(t0, t1));
        // Disjoint coplanar triangles ⇒ no overlap.
        let t1 = [[10.0, 10.0, 0.0], [11.0, 10.0, 0.0], [10.0, 11.0, 0.0]];
        assert!(!triangles_coplanar_overlap(t0, t1));
        // Non-coplanar (normal-direction offset of 0.1 ≫ epsilon)
        // ⇒ helper rejects.
        let t1 = [[0.5, 0.5, 0.1], [2.5, 0.5, 0.1], [1.5, 2.5, 0.1]];
        assert!(!triangles_coplanar_overlap(t0, t1));
    }

    // Property-based tests for `triangles_coplanar_overlap`. We force a
    // common z = 0 plane so the coplanarity gate doesn't reject the
    // sample, and reject degenerate (near-collinear) triangles where the
    // normal length falls below COPLANAR_EPS.
    proptest::proptest! {
        #[test]
        fn prop_triangles_coplanar_overlap_self(
            ax in -100.0_f64..100.0, ay in -100.0_f64..100.0,
            bx in -100.0_f64..100.0, by in -100.0_f64..100.0,
            cx in -100.0_f64..100.0, cy in -100.0_f64..100.0,
        ) {
            let t = [[ax, ay, 0.0], [bx, by, 0.0], [cx, cy, 0.0]];
            // Skip near-degenerate triangles (signed area ~ 0); the
            // helper rejects those by design and the property is vacuous.
            let area2 = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)).abs();
            proptest::prop_assume!(area2 > 1e-3);
            proptest::prop_assert!(triangles_coplanar_overlap(t, t));
        }

        #[test]
        fn prop_triangles_coplanar_overlap_symmetric(
            ax in -100.0_f64..100.0, ay in -100.0_f64..100.0,
            bx in -100.0_f64..100.0, by in -100.0_f64..100.0,
            cx in -100.0_f64..100.0, cy in -100.0_f64..100.0,
            dx in -100.0_f64..100.0, dy in -100.0_f64..100.0,
            ex in -100.0_f64..100.0, ey in -100.0_f64..100.0,
            fx in -100.0_f64..100.0, fy in -100.0_f64..100.0,
        ) {
            let t0 = [[ax, ay, 0.0], [bx, by, 0.0], [cx, cy, 0.0]];
            let t1 = [[dx, dy, 0.0], [ex, ey, 0.0], [fx, fy, 0.0]];
            // Reject degenerate triangles for both inputs so the
            // coplanarity gate accepts both orderings symmetrically.
            let a0 = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)).abs();
            let a1 = ((ex - dx) * (fy - dy) - (ey - dy) * (fx - dx)).abs();
            proptest::prop_assume!(a0 > 1e-3 && a1 > 1e-3);
            proptest::prop_assert_eq!(
                triangles_coplanar_overlap(t0, t1),
                triangles_coplanar_overlap(t1, t0),
            );
        }
    }
}
