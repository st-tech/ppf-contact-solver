// File: crates/ppf-cts-core/src/kernels/proximity.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Contact-offset proximity scan. 1:1 functional port of
// frontend/_proximity_.py's `check_contact_offset_violation` plus the
// distance primitives it leans on (`point_triangle_dist_sq`,
// `point_edge_dist_sq`, `edge_edge_dist_sq`, `_tri_tri_distance_sq`,
// `_tri_edge_distance_sq`).
//
// Pipeline:
//   1. Build tri-BVH over input triangles (if F provided).
//   2. Build edge-BVH over input edges (if E provided).
//   3. Three rayon-parallel scans:
//      * Tri-Tri  (n_tris >= 2)
//      * Tri-Edge (n_tris >= 1 && n_edges >= 1)
//      * Edge-Edge (n_edges >= 2)
//   Each scan emits `(elem_i, elem_j)` pairs with a unified index
//   namespace: triangles 0..n_tris, edges n_tris..n_tris+n_edges.
//
// Skip rules (preserved from Python source):
//   * collider × collider pairs are skipped,
//   * pairs sharing any vertex are skipped,
//   * tri-tri / edge-edge use `j > i` ordering so each unordered pair
//     is reported once.

use rayon::prelude::*;

use super::bvh::{
    build_edge_bvh, build_tri_bvh, closest_point_on_triangle, traverse_overlap, Bvh,
};
use super::geom_util::{
    bbox_overlap, dot3, elements_share_vertex_2_3, elements_share_vertex_3_3, expand_bbox, sub3,
    vert3,
};

// Degeneracy guards. Split by physical dimension so the threshold's meaning is
// explicit even though both values are intentionally identical (no numeric change).
const EPS_LEN_SQ: f64 = 1e-14; // squared edge length (L^2) degeneracy guard
const EPS_DENOM: f64 = 1e-14; // squared parallelogram area |d1 x d2|^2 (L^4); near-parallel-edge guard

// ---------------------------------------------------------------------------
// Distance primitives

#[inline]
fn point_edge_dist_sq(p: [f64; 3], e0: [f64; 3], e1: [f64; 3]) -> f64 {
    let edge = sub3(e1, e0);
    let edge_len_sq = dot3(edge, edge);
    if edge_len_sq < EPS_LEN_SQ {
        let d = sub3(p, e0);
        return dot3(d, d);
    }
    let t = (dot3(sub3(p, e0), edge) / edge_len_sq).clamp(0.0, 1.0);
    let closest = [
        e0[0] + t * edge[0],
        e0[1] + t * edge[1],
        e0[2] + t * edge[2],
    ];
    let d = sub3(p, closest);
    dot3(d, d)
}

#[inline]
fn edge_edge_dist_sq(a0: [f64; 3], a1: [f64; 3], b0: [f64; 3], b1: [f64; 3]) -> f64 {
    let d1 = sub3(a1, a0);
    let d2 = sub3(b1, b0);
    let r = sub3(a0, b0);

    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);

    let (s, t);
    if a < EPS_LEN_SQ && e < EPS_LEN_SQ {
        let d = sub3(a0, b0);
        return dot3(d, d);
    } else if a < EPS_LEN_SQ {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else if e < EPS_LEN_SQ {
        t = 0.0;
        s = (-dot3(d1, r) / a).clamp(0.0, 1.0);
    } else {
        let b_val = dot3(d1, d2);
        let c = dot3(d1, r);
        // Gram determinant |d1 x d2|^2, a quartic (L^4) quantity, not a length^2.
        let denom = a * e - b_val * b_val;
        let s_init = if denom.abs() > EPS_DENOM {
            ((b_val * f - c * e) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let t_init = (b_val * s_init + f) / e;

        if t_init < 0.0 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else if t_init > 1.0 {
            t = 1.0;
            s = ((b_val - c) / a).clamp(0.0, 1.0);
        } else {
            s = s_init;
            t = t_init;
        }
    }

    let closest_a = [a0[0] + s * d1[0], a0[1] + s * d1[1], a0[2] + s * d1[2]];
    let closest_b = [b0[0] + t * d2[0], b0[1] + t * d2[1], b0[2] + t * d2[2]];
    let diff = sub3(closest_a, closest_b);
    dot3(diff, diff)
}

#[inline]
fn point_triangle_dist_sq(
    p: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> f64 {
    let (closest, _) = closest_point_on_triangle(p, v0, v1, v2);
    let d = sub3(p, closest);
    dot3(d, d)
}

// ---------------------------------------------------------------------------
// Pair-distance primitives with early-exit.

/// Min squared distance between two triangles, returning early once
/// the threshold is breached. 6 point-triangle + 9 edge-edge tests.
#[inline]
fn tri_tri_distance_sq(
    verts: &[f64],
    tri_i: [i32; 3],
    tri_j: [i32; 3],
    threshold_sq: f64,
) -> f64 {
    let ti = [
        vert3(verts, tri_i[0]),
        vert3(verts, tri_i[1]),
        vert3(verts, tri_i[2]),
    ];
    let tj = [
        vert3(verts, tri_j[0]),
        vert3(verts, tri_j[1]),
        vert3(verts, tri_j[2]),
    ];

    let mut min_sq = f64::INFINITY;
    // 3 verts of i to triangle j
    for vi in 0..3 {
        let d = point_triangle_dist_sq(ti[vi], tj[0], tj[1], tj[2]);
        if d < min_sq {
            min_sq = d;
            if min_sq < threshold_sq {
                return min_sq;
            }
        }
    }
    // 3 verts of j to triangle i
    for vj in 0..3 {
        let d = point_triangle_dist_sq(tj[vj], ti[0], ti[1], ti[2]);
        if d < min_sq {
            min_sq = d;
            if min_sq < threshold_sq {
                return min_sq;
            }
        }
    }
    // 9 edge-edge combinations
    let edges = [(0usize, 1usize), (1, 2), (2, 0)];
    for ei in 0..3 {
        let (a0, a1) = (ti[edges[ei].0], ti[edges[ei].1]);
        for ej in 0..3 {
            let (b0, b1) = (tj[edges[ej].0], tj[edges[ej].1]);
            let d = edge_edge_dist_sq(a0, a1, b0, b1);
            if d < min_sq {
                min_sq = d;
                if min_sq < threshold_sq {
                    return min_sq;
                }
            }
        }
    }
    min_sq
}

#[inline]
fn tri_edge_distance_sq(
    verts: &[f64],
    tri: [i32; 3],
    edge: [i32; 2],
    threshold_sq: f64,
) -> f64 {
    let t = [
        vert3(verts, tri[0]),
        vert3(verts, tri[1]),
        vert3(verts, tri[2]),
    ];
    let e = [vert3(verts, edge[0]), vert3(verts, edge[1])];

    let mut min_sq = f64::INFINITY;
    // Edge endpoints to triangle (2)
    for k in 0..2 {
        let d = point_triangle_dist_sq(e[k], t[0], t[1], t[2]);
        if d < min_sq {
            min_sq = d;
            if min_sq < threshold_sq {
                return min_sq;
            }
        }
    }
    // Triangle vertices to edge (3)
    for k in 0..3 {
        let d = point_edge_dist_sq(t[k], e[0], e[1]);
        if d < min_sq {
            min_sq = d;
            if min_sq < threshold_sq {
                return min_sq;
            }
        }
    }
    // Triangle edges to edge (3)
    let edges = [(0usize, 1usize), (1, 2), (2, 0)];
    for ek in 0..3 {
        let (a0, a1) = (t[edges[ek].0], t[edges[ek].1]);
        let d = edge_edge_dist_sq(a0, a1, e[0], e[1]);
        if d < min_sq {
            min_sq = d;
            if min_sq < threshold_sq {
                return min_sq;
            }
        }
    }
    min_sq
}

// ---------------------------------------------------------------------------
// Helpers

#[inline]
fn elements_share_vertex_2_2(a: [i32; 2], b: [i32; 2]) -> bool {
    a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1]
}

// ---------------------------------------------------------------------------
// Per-source-element scan kernels. Each emits a Vec<(i32, i32)> with the
// pairs found for the source element. Caller flat-maps these in
// vertex-index order to keep results deterministic.

#[inline]
fn find_close_tri_tri(
    ti: usize,
    verts: &[f64],
    tris: &[i32],
    bvh: &Bvh,
    contact_offset: &[f64],
    is_collider: &[bool],
    max_cand_offset: f64,
) -> Vec<(i32, i32)> {
    let tri_i = [tris[3 * ti], tris[3 * ti + 1], tris[3 * ti + 2]];
    let offset_i = contact_offset[ti];
    // Tight box (this element expanded by its own offset) drives the precise
    // per-candidate Minkowski-overlap prune below.
    let (bb_i_min, bb_i_max) =
        expand_bbox(&bvh.elem_bboxes_min[ti], &bvh.elem_bboxes_max[ti], offset_i);
    // Traversal box expands by offset_i + the max candidate offset. BVH node
    // bboxes are bare (no per-element offset baked in), so culling with the
    // tight box alone drops a candidate that sits within offset_i + offset_j
    // whenever offset_i is the smaller term, e.g. a zero-offset shell triangle
    // whose nearby rod carries the whole offset. The precise per-candidate
    // test inside still uses the tight box, so this only widens what the BVH
    // visits, never what it reports.
    let (q_min, q_max) = expand_bbox(
        &bvh.elem_bboxes_min[ti],
        &bvh.elem_bboxes_max[ti],
        offset_i + max_cand_offset,
    );

    let mut out = Vec::new();
    traverse_overlap(bvh, &q_min, &q_max, |tj_i| {
        let tj = tj_i as usize;
        if tj <= ti {
            return;
        }
        if is_collider[ti] && is_collider[tj] {
            return;
        }
        let tri_j = [tris[3 * tj], tris[3 * tj + 1], tris[3 * tj + 2]];
        if elements_share_vertex_3_3(tri_i, tri_j) {
            return;
        }

        let offset_j = contact_offset[tj];
        let required_sq = (offset_i + offset_j).powi(2);

        let (exp_min, exp_max) =
            expand_bbox(&bvh.elem_bboxes_min[tj], &bvh.elem_bboxes_max[tj], offset_j);
        if !bbox_overlap(&bb_i_min, &bb_i_max, &exp_min, &exp_max) {
            return;
        }

        let d_sq = tri_tri_distance_sq(verts, tri_i, tri_j, required_sq);
        if d_sq < required_sq {
            out.push((ti as i32, tj_i));
        }
    });
    out
}

#[inline]
fn find_close_tri_edge(
    ti: usize,
    verts: &[f64],
    tris: &[i32],
    edges: &[i32],
    edge_bvh: &Bvh,
    tri_offset: &[f64],
    edge_offset: &[f64],
    tri_is_collider: &[bool],
    edge_is_collider: &[bool],
    n_tris: usize,
    tri_bboxes_min: &[[f64; 3]],
    tri_bboxes_max: &[[f64; 3]],
    max_cand_offset: f64,
) -> Vec<(i32, i32)> {
    let tri = [tris[3 * ti], tris[3 * ti + 1], tris[3 * ti + 2]];
    let offset_i = tri_offset[ti];
    // Tight box for the precise per-candidate prune; loose box (offset_i + max
    // candidate offset) for the BVH traversal. See `find_close_tri_tri` for why
    // the bare node bboxes force the wider traversal box.
    let (bb_i_min, bb_i_max) = expand_bbox(&tri_bboxes_min[ti], &tri_bboxes_max[ti], offset_i);
    let (q_min, q_max) =
        expand_bbox(&tri_bboxes_min[ti], &tri_bboxes_max[ti], offset_i + max_cand_offset);

    let mut out = Vec::new();
    traverse_overlap(edge_bvh, &q_min, &q_max, |ej_i| {
        let ej = ej_i as usize;
        if tri_is_collider[ti] && edge_is_collider[ej] {
            return;
        }
        let edge = [edges[2 * ej], edges[2 * ej + 1]];
        if elements_share_vertex_2_3(edge, tri) {
            return;
        }

        let offset_j = edge_offset[ej];
        let required_sq = (offset_i + offset_j).powi(2);
        let (exp_min, exp_max) = expand_bbox(
            &edge_bvh.elem_bboxes_min[ej],
            &edge_bvh.elem_bboxes_max[ej],
            offset_j,
        );
        if !bbox_overlap(&bb_i_min, &bb_i_max, &exp_min, &exp_max) {
            return;
        }
        let d_sq = tri_edge_distance_sq(verts, tri, edge, required_sq);
        if d_sq < required_sq {
            out.push((ti as i32, (n_tris as i32) + ej_i));
        }
    });
    out
}

#[inline]
fn find_close_edge_edge(
    ei: usize,
    verts: &[f64],
    edges: &[i32],
    edge_bvh: &Bvh,
    contact_offset: &[f64],
    is_collider: &[bool],
    n_tris: usize,
    max_cand_offset: f64,
) -> Vec<(i32, i32)> {
    let edge_i = [edges[2 * ei], edges[2 * ei + 1]];
    let offset_i = contact_offset[ei];
    // Tight box for the precise per-candidate prune; loose box (offset_i + max
    // candidate offset) for the BVH traversal. See `find_close_tri_tri`. This
    // also matters for rod-vs-rod: two rods up to offset_i + offset_j apart
    // must be visited, not just those within offset_i.
    let (bb_i_min, bb_i_max) = expand_bbox(
        &edge_bvh.elem_bboxes_min[ei],
        &edge_bvh.elem_bboxes_max[ei],
        offset_i,
    );
    let (q_min, q_max) = expand_bbox(
        &edge_bvh.elem_bboxes_min[ei],
        &edge_bvh.elem_bboxes_max[ei],
        offset_i + max_cand_offset,
    );

    let e0_i = vert3(verts, edge_i[0]);
    let e1_i = vert3(verts, edge_i[1]);

    let mut out = Vec::new();
    traverse_overlap(edge_bvh, &q_min, &q_max, |ej_i| {
        let ej = ej_i as usize;
        if ej <= ei {
            return;
        }
        if is_collider[ei] && is_collider[ej] {
            return;
        }
        let edge_j = [edges[2 * ej], edges[2 * ej + 1]];
        if elements_share_vertex_2_2(edge_i, edge_j) {
            return;
        }

        let offset_j = contact_offset[ej];
        let required_sq = (offset_i + offset_j).powi(2);
        let (exp_min, exp_max) = expand_bbox(
            &edge_bvh.elem_bboxes_min[ej],
            &edge_bvh.elem_bboxes_max[ej],
            offset_j,
        );
        if !bbox_overlap(&bb_i_min, &bb_i_max, &exp_min, &exp_max) {
            return;
        }
        let e0_j = vert3(verts, edge_j[0]);
        let e1_j = vert3(verts, edge_j[1]);
        let d_sq = edge_edge_dist_sq(e0_i, e1_i, e0_j, e1_j);
        if d_sq < required_sq {
            out.push((
                (n_tris as i32) + (ei as i32),
                (n_tris as i32) + ej_i,
            ));
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Public API

/// Optional input slices for `check_contact_offset_violation`. Either or
/// both of `tris` and `edges` may be `None` (matching the Python F / E
/// semantics). `is_collider` and `contact_offset` are length n_tris +
/// n_edges; pass `None` for default (all false / all zero).
pub struct ProximityInput<'a> {
    pub verts: &'a [f64],
    pub tris: Option<&'a [i32]>,
    pub edges: Option<&'a [i32]>,
    pub is_collider: Option<&'a [bool]>,
    pub contact_offset: Option<&'a [f64]>,
}

/// Check for contact-offset violations between mesh elements. Output
/// pairs are `(elem_i, elem_j)` with the unified namespace described
/// above.
pub fn check_contact_offset_violation(input: ProximityInput<'_>) -> Vec<(i32, i32)> {
    let n_tris = input.tris.map(|t| t.len() / 3).unwrap_or(0);
    let n_edges = input.edges.map(|e| e.len() / 2).unwrap_or(0);
    if n_tris + n_edges < 2 {
        return vec![];
    }

    // Default the optional inputs.
    let n_elems = n_tris + n_edges;
    let zero_collider: Vec<bool>;
    let is_collider = match input.is_collider {
        Some(s) => s,
        None => {
            zero_collider = vec![false; n_elems];
            &zero_collider
        }
    };
    let zero_offset: Vec<f64>;
    let offset = match input.contact_offset {
        Some(s) => s,
        None => {
            zero_offset = vec![0.0; n_elems];
            &zero_offset
        }
    };

    let tri_is_collider = &is_collider[..n_tris];
    let edge_is_collider = &is_collider[n_tris..];
    let tri_offset = &offset[..n_tris];
    let edge_offset = &offset[n_tris..];

    // Max offset per candidate kind: the BVH traversal box for each source
    // element is widened by source_offset + this, so no candidate within the
    // summed contact offset is culled before the precise distance test (the
    // BVH node bboxes are bare). Without this an asymmetric pair (a zero-offset
    // triangle next to an offset-carrying rod) is silently dropped.
    let max_tri_offset = tri_offset.iter().copied().fold(0.0_f64, f64::max);
    let max_edge_offset = edge_offset.iter().copied().fold(0.0_f64, f64::max);

    let tri_bvh = input
        .tris
        .filter(|_| n_tris > 0)
        .map(|tris| build_tri_bvh(input.verts, tris));
    let edge_bvh = input
        .edges
        .filter(|_| n_edges > 0)
        .map(|edges| build_edge_bvh(input.verts, edges));

    let mut all_pairs: Vec<(i32, i32)> = Vec::new();

    // Tri-Tri
    if n_tris >= 2 {
        let tris = input.tris.unwrap();
        let bvh = tri_bvh.as_ref().unwrap();
        let mut pairs: Vec<(i32, i32)> = (0..n_tris)
            .into_par_iter()
            .flat_map(|ti| {
                find_close_tri_tri(
                    ti,
                    input.verts,
                    tris,
                    bvh,
                    tri_offset,
                    tri_is_collider,
                    max_tri_offset,
                )
            })
            .collect();
        all_pairs.append(&mut pairs);
    }

    // Tri-Edge
    if n_tris > 0 && n_edges > 0 {
        let tris = input.tris.unwrap();
        let edges = input.edges.unwrap();
        let tri_bvh_ref = tri_bvh.as_ref().unwrap();
        let edge_bvh_ref = edge_bvh.as_ref().unwrap();
        let tri_bb_min = &tri_bvh_ref.elem_bboxes_min;
        let tri_bb_max = &tri_bvh_ref.elem_bboxes_max;
        let mut pairs: Vec<(i32, i32)> = (0..n_tris)
            .into_par_iter()
            .flat_map(|ti| {
                find_close_tri_edge(
                    ti,
                    input.verts,
                    tris,
                    edges,
                    edge_bvh_ref,
                    tri_offset,
                    edge_offset,
                    tri_is_collider,
                    edge_is_collider,
                    n_tris,
                    tri_bb_min,
                    tri_bb_max,
                    max_edge_offset,
                )
            })
            .collect();
        all_pairs.append(&mut pairs);
    }

    // Edge-Edge
    if n_edges >= 2 {
        let edges = input.edges.unwrap();
        let bvh = edge_bvh.as_ref().unwrap();
        let mut pairs: Vec<(i32, i32)> = (0..n_edges)
            .into_par_iter()
            .flat_map(|ei| {
                find_close_edge_edge(
                    ei,
                    input.verts,
                    edges,
                    bvh,
                    edge_offset,
                    edge_is_collider,
                    n_tris,
                    max_edge_offset,
                )
            })
            .collect();
        all_pairs.append(&mut pairs);
    }

    all_pairs
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
    fn flat2i(rows: &[[i32; 2]]) -> Vec<i32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    #[test]
    fn close_tris_violate_when_offsets_sum_exceeds_distance() {
        // Two parallel triangles 0.05 apart in z.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.05], [1.0, 0.0, 0.05], [0.5, 1.0, 0.05],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        // offsets sum to 0.1 > distance 0.05 ⇒ violation
        let off = vec![0.05, 0.05];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: Some(&tris),
            edges: None,
            is_collider: None,
            contact_offset: Some(&off),
        });
        assert_eq!(r, vec![(0, 1)]);

        // Tighter offsets ⇒ no violation
        let off = vec![0.02, 0.02];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: Some(&tris),
            edges: None,
            is_collider: None,
            contact_offset: Some(&off),
        });
        assert!(r.is_empty());
    }

    #[test]
    fn asymmetric_offset_tri_edge_not_culled() {
        // Regression for the BVH offset-cull bug: a zero-offset triangle and a
        // rod edge carrying the whole contact offset. The triangle's bare bbox
        // (expanded by its own offset 0) never reaches the rod, so traversing
        // with the tight box alone dropped the pair even though the rod's offset
        // brings it into range. Mirrors the real rod-on-shell scene (shell
        // contact-offset 0, rod 0.014, edge gap ~0.009).
        let verts = flat3(&[
            // Triangle in the z = 0 plane.
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            // Rod edge hovering 0.009 above the triangle interior.
            [0.25, 0.25, 0.009], [0.30, 0.25, 0.009],
        ]);
        let tris = flat3i(&[[0, 1, 2]]);
        let edges = flat2i(&[[3, 4]]);
        // tri offset 0, rod offset 0.014 ⇒ sum 0.014 > gap 0.009 ⇒ violation.
        let off = vec![0.0, 0.014];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: Some(&tris),
            edges: Some(&edges),
            is_collider: None,
            contact_offset: Some(&off),
        });
        // namespace: tri 0, rod edge at n_tris + 0 = 1.
        assert_eq!(r, vec![(0, 1)]);

        // Pull the rod above the summed offset ⇒ no violation.
        let verts_far = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.02], [0.30, 0.25, 0.02],
        ]);
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts_far,
            tris: Some(&tris),
            edges: Some(&edges),
            is_collider: None,
            contact_offset: Some(&off),
        });
        assert!(r.is_empty());
    }

    #[test]
    fn edge_edge_distance_between_max_and_sum_not_culled() {
        // Two parallel rod edges 0.015 apart, each offset 0.01. The gap exceeds
        // either single offset (0.01) but is under their sum (0.02), so the
        // tight-box traversal (expand by offset_i = 0.01) missed it.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 0.015, 0.0], [1.0, 0.015, 0.0],
        ]);
        let edges = flat2i(&[[0, 1], [2, 3]]);
        let off = vec![0.01, 0.01];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: None,
            edges: Some(&edges),
            is_collider: None,
            contact_offset: Some(&off),
        });
        assert_eq!(r, vec![(0, 1)]);
    }

    #[test]
    fn collider_pair_skipped() {
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.05], [1.0, 0.0, 0.05], [0.5, 1.0, 0.05],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        let off = vec![0.05, 0.05];
        let coll_both = vec![true, true];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: Some(&tris),
            edges: None,
            is_collider: Some(&coll_both),
            contact_offset: Some(&off),
        });
        assert!(r.is_empty(), "collider×collider must be skipped");

        let coll_one = vec![true, false];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: Some(&tris),
            edges: None,
            is_collider: Some(&coll_one),
            contact_offset: Some(&off),
        });
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn shared_vertex_skipped() {
        // Two triangles sharing edge 0-1.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [0, 1, 3]]);
        let off = vec![0.5, 0.5];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: Some(&tris),
            edges: None,
            is_collider: None,
            contact_offset: Some(&off),
        });
        assert!(r.is_empty(), "adjacent triangles must be skipped");
    }

    #[test]
    fn close_edges_violate() {
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.05], [1.0, 0.0, 0.05],
        ]);
        let edges = flat2i(&[[0, 1], [2, 3]]);
        let off = vec![0.05, 0.05];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: None,
            edges: Some(&edges),
            is_collider: None,
            contact_offset: Some(&off),
        });
        assert_eq!(r, vec![(0, 1)]);
    }

    #[test]
    fn tri_edge_unified_index_namespace() {
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [0.25, 0.25, 0.05], [0.75, 0.25, 0.05],
        ]);
        let tris = flat3i(&[[0, 1, 2]]);
        let edges = flat2i(&[[3, 4]]);
        // n_tris = 1, n_edges = 1; offset array layout: [tri_off, edge_off]
        let off = vec![0.05, 0.05];
        let r = check_contact_offset_violation(ProximityInput {
            verts: &verts,
            tris: Some(&tris),
            edges: Some(&edges),
            is_collider: None,
            contact_offset: Some(&off),
        });
        // Pair: (tri 0, edge 0 + n_tris=1 ⇒ 1)
        assert_eq!(r, vec![(0, 1)]);
    }

    // ----- Direct primitive tests -----

    #[test]
    fn point_edge_dist_sq_endpoints_and_interior() {
        let e0 = [0.0, 0.0, 0.0];
        let e1 = [1.0, 0.0, 0.0];
        // On the segment midpoint -> 0 distance.
        assert!(point_edge_dist_sq([0.5, 0.0, 0.0], e0, e1).abs() < 1e-14);
        // Past the endpoint, projection clamps to e1 ⇒ dist = 1 along x.
        assert!((point_edge_dist_sq([2.0, 0.0, 0.0], e0, e1) - 1.0).abs() < 1e-14);
        // Off the segment perpendicularly at midpoint, dist = 0.5.
        let d = point_edge_dist_sq([0.5, 0.5, 0.0], e0, e1);
        assert!((d - 0.25).abs() < 1e-14, "got {d}");
    }

    #[test]
    fn edge_edge_dist_sq_parallel_skew_and_crossing() {
        // Parallel segments offset by 1 in y: closest pair has dist 1.
        let d = edge_edge_dist_sq(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        );
        assert!((d - 1.0).abs() < 1e-14);
        // Skew lines that "cross" in projection but offset by z = 0.5.
        let d = edge_edge_dist_sq(
            [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.5], [0.0, 1.0, 0.5],
        );
        assert!((d - 0.25).abs() < 1e-14, "got {d}");
    }

    #[test]
    fn tri_tri_distance_sq_parallel_offset_in_normal() {
        // Two coplanar-stacked triangles separated by 0.05 in z;
        // distance² should be 0.05² = 2.5e-3.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.05], [1.0, 0.0, 0.05], [0.0, 1.0, 0.05],
        ]);
        let d = tri_tri_distance_sq(&verts, [0, 1, 2], [3, 4, 5], -1.0);
        assert!((d - 0.0025).abs() < 1e-14, "got {d}");
    }

    #[test]
    fn tri_edge_distance_sq_above_triangle() {
        // Edge floats 0.1 above an xy-triangle; tri-edge distance² = 0.01.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.1], [0.5, 0.25, 0.1],
        ]);
        let d = tri_edge_distance_sq(&verts, [0, 1, 2], [3, 4], -1.0);
        assert!((d - 0.01).abs() < 1e-14, "got {d}");
    }

    #[test]
    fn elements_share_vertex_2_2_basic() {
        assert!(elements_share_vertex_2_2([0, 1], [1, 2]));
        assert!(elements_share_vertex_2_2([3, 4], [4, 5]));
        assert!(!elements_share_vertex_2_2([0, 1], [2, 3]));
        // Same edge counts as sharing.
        assert!(elements_share_vertex_2_2([0, 1], [0, 1]));
    }

    // Property-based tests over `tri_tri_distance_sq`. Use a sentinel
    // `threshold_sq = -1.0` so the early-exit branch is dormant and we
    // always observe the true minimum; bounded coords keep each case
    // numerically tame and below 100 ms.
    proptest::proptest! {
        #[test]
        fn prop_tri_tri_distance_sq_symmetric(
            a in proptest::array::uniform3(-100.0_f64..100.0),
            b in proptest::array::uniform3(-100.0_f64..100.0),
            c in proptest::array::uniform3(-100.0_f64..100.0),
            d in proptest::array::uniform3(-100.0_f64..100.0),
            e in proptest::array::uniform3(-100.0_f64..100.0),
            f in proptest::array::uniform3(-100.0_f64..100.0),
        ) {
            let verts = flat3(&[a, b, c, d, e, f]);
            let dij = tri_tri_distance_sq(&verts, [0, 1, 2], [3, 4, 5], -1.0);
            let dji = tri_tri_distance_sq(&verts, [3, 4, 5], [0, 1, 2], -1.0);
            // Both finite, swap order is exact symmetric (same arithmetic
            // on the same edge/vertex pairs, just iterated reverse).
            let tol = 1e-9 * (1.0 + dij.abs());
            proptest::prop_assert!((dij - dji).abs() <= tol, "dij={dij} dji={dji}");
        }

        #[test]
        fn prop_tri_tri_distance_sq_translation_invariant(
            a in proptest::array::uniform3(-100.0_f64..100.0),
            b in proptest::array::uniform3(-100.0_f64..100.0),
            c in proptest::array::uniform3(-100.0_f64..100.0),
            d in proptest::array::uniform3(-100.0_f64..100.0),
            e in proptest::array::uniform3(-100.0_f64..100.0),
            f in proptest::array::uniform3(-100.0_f64..100.0),
            t in proptest::array::uniform3(-100.0_f64..100.0),
        ) {
            let verts = flat3(&[a, b, c, d, e, f]);
            let shift = |p: [f64; 3]| [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
            let verts_t = flat3(&[shift(a), shift(b), shift(c), shift(d), shift(e), shift(f)]);
            let d0 = tri_tri_distance_sq(&verts, [0, 1, 2], [3, 4, 5], -1.0);
            let d1 = tri_tri_distance_sq(&verts_t, [0, 1, 2], [3, 4, 5], -1.0);
            // Translation is exact in real arithmetic; in f64 the
            // shifted dots round differently, so allow a small relative
            // slack scaled by post-shift coordinate magnitudes.
            let coord_mag = 100.0 + t[0].abs() + t[1].abs() + t[2].abs();
            let tol = 1e-10 * coord_mag * coord_mag * (1.0 + d0.abs());
            proptest::prop_assert!((d0 - d1).abs() <= tol, "d0={d0} d1={d1}");
        }
    }
}
