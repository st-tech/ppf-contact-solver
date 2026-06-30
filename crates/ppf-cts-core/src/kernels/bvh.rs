// File: crates/ppf-cts-core/src/kernels/bvh.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Morton-code BVH + frame-embedding kernels. 1:1 functional port of
// frontend/_bvh_.py's `frame_mapping` and `interpolate_surface`
// public functions, plus the inner machinery they need:
//
//   * `closest_point_on_triangle`  (Ericson Voronoi-region method).
//   * `build_bvh`                  (Morton sort + midpoint split).
//   * `closest_triangle_index`     (BVH traversal with bbox early-exit).
//
// Numerics policy: f64 throughout; tolerances in the Python tests run
// 1e-9 down to 1e-12, so the math must match exactly. Frame-degenerate
// epsilon comes straight from the Python source (1e-20).
//
// Parallelism: rayon over input points in `frame_mapping` and
// `interpolate_surface`. BVH build itself is mostly serial; the inner
// phases (Morton codes, leaf bboxes) parallelize cleanly but the
// midpoint split + bottom-up propagation are intrinsically sequential
// the same way the Python `_build_bvh_structure` is.

use rayon::prelude::*;

use super::constants::BVH_STACK_CAP;
use super::geom_util::{bbox_overlap, cross3, dot3, point_to_bbox_dist_sq, sub3};

const FRAME_DEGEN_EPS: f64 = 1e-20;
const DEFAULT_MAX_LEAF: usize = 8;

// ---------------------------------------------------------------------------
// closest_point_on_triangle
//
// Voronoi-region method from Ericson, "Real-Time Collision Detection".
// Returns (closest_point, barycentric_coords) where `bary[i]` is the
// weight on vertex i ∈ {a, b, c}. On-edge / on-vertex tie-breaking
// follows the Ericson Voronoi-region branching.

#[inline]
pub(crate) fn closest_point_on_triangle(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(p, a);

    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);

    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }

    let bp = sub3(p, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);

    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom != 0.0 { d1 / denom } else { 0.0 };
        let pt = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
        return (pt, [1.0 - v, v, 0.0]);
    }

    let cp = sub3(p, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);

    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom != 0.0 { d2 / denom } else { 0.0 };
        let pt = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
        return (pt, [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    let d4_d3 = d4 - d3;
    let d5_d6 = d5 - d6;
    if va <= 0.0 && d4_d3 >= 0.0 && d5_d6 >= 0.0 {
        let denom = d4_d3 + d5_d6;
        let w = if denom != 0.0 { d4_d3 / denom } else { 0.0 };
        let bc = sub3(c, b);
        let pt = [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
        return (pt, [0.0, 1.0 - w, w]);
    }

    let denom = va + vb + vc;
    if denom == 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let v = vb / denom;
    let w = vc / denom;
    let u = 1.0 - v - w;
    let pt = [
        a[0] + v * ab[0] + w * ac[0],
        a[1] + v * ab[1] + w * ac[1],
        a[2] + v * ab[2] + w * ac[2],
    ];
    (pt, [u, v, w])
}

// ---------------------------------------------------------------------------
// Morton-code BVH

#[derive(Debug)]
pub struct Bvh {
    pub node_bbox_min: Vec<[f64; 3]>,
    pub node_bbox_max: Vec<[f64; 3]>,
    /// `-1` for leaves.
    pub node_left: Vec<i32>,
    pub node_right: Vec<i32>,
    /// `>= 0` for leaves; `-1` for internal nodes.
    pub node_elem_start: Vec<i32>,
    pub node_elem_count: Vec<i32>,
    /// Morton-sorted element indices; leaf nodes index into this.
    pub sorted_indices: Vec<i32>,
    /// Per-element bboxes indexed by *original* (unsorted) element
    /// index. The Python ElementBVH carries these as siblings of the
    /// node arrays; proximity / intersection traversals use them at
    /// leaf hits to expand by the per-element contact offset before
    /// the per-pair distance test.
    pub elem_bboxes_min: Vec<[f64; 3]>,
    pub elem_bboxes_max: Vec<[f64; 3]>,
}

#[inline]
fn morton_3d_10bit(ix: u32, iy: u32, iz: u32) -> u32 {
    let mut code = 0u32;
    for bit in 0..10 {
        code |= ((ix >> bit) & 1) << (3 * bit);
        code |= ((iy >> bit) & 1) << (3 * bit + 1);
        code |= ((iz >> bit) & 1) << (3 * bit + 2);
    }
    code
}

/// Build a flat BVH over `n_elems` elements ordered by 30-bit Morton
/// codes. `centroids` and `bboxes_min` / `bboxes_max` are flat (N*3)
/// row-major slices indexed by *original* element index (not sorted).
pub(crate) fn build_bvh(
    centroids: &[f64],
    bboxes_min: &[f64],
    bboxes_max: &[f64],
    max_leaf_size: usize,
) -> Bvh {
    let n_elems = centroids.len() / 3;
    debug_assert_eq!(centroids.len(), n_elems * 3);
    debug_assert_eq!(bboxes_min.len(), n_elems * 3);
    debug_assert_eq!(bboxes_max.len(), n_elems * 3);

    if n_elems == 0 {
        return Bvh {
            node_bbox_min: vec![],
            node_bbox_max: vec![],
            node_left: vec![],
            node_right: vec![],
            node_elem_start: vec![],
            node_elem_count: vec![],
            sorted_indices: vec![],
            elem_bboxes_min: vec![],
            elem_bboxes_max: vec![],
        };
    }

    // Step 0: scene bounds over centroids.
    let mut scene_min = [f64::INFINITY; 3];
    let mut scene_max = [f64::NEG_INFINITY; 3];
    for i in 0..n_elems {
        for d in 0..3 {
            let v = centroids[3 * i + d];
            if v < scene_min[d] {
                scene_min[d] = v;
            }
            if v > scene_max[d] {
                scene_max[d] = v;
            }
        }
    }
    let mut scale = [
        scene_max[0] - scene_min[0],
        scene_max[1] - scene_min[1],
        scene_max[2] - scene_min[2],
    ];
    for d in 0..3 {
        if scale[d] < 1e-10 {
            scale[d] = 1.0;
        }
    }

    // Step 1: per-element Morton codes (parallel).
    let morton_codes: Vec<u32> = (0..n_elems)
        .into_par_iter()
        .map(|i| {
            let mut nx = (centroids[3 * i] - scene_min[0]) / scale[0];
            let mut ny = (centroids[3 * i + 1] - scene_min[1]) / scale[1];
            let mut nz = (centroids[3 * i + 2] - scene_min[2]) / scale[2];
            nx = nx.clamp(0.0, 1.0);
            ny = ny.clamp(0.0, 1.0);
            nz = nz.clamp(0.0, 1.0);
            let ix = ((nx * 1023.0) as u32).min(1023);
            let iy = ((ny * 1023.0) as u32).min(1023);
            let iz = ((nz * 1023.0) as u32).min(1023);
            morton_3d_10bit(ix, iy, iz)
        })
        .collect();

    // Step 2: argsort by Morton code, stable.
    let mut sorted_indices: Vec<i32> = (0..n_elems as i32).collect();
    sorted_indices.par_sort_by_key(|&i| morton_codes[i as usize]);

    // Step 3: tree topology by midpoint split (sequential, single stack).
    let max_nodes = 2 * n_elems;
    let mut node_left = vec![-1i32; max_nodes];
    let mut node_right = vec![-1i32; max_nodes];
    let mut node_elem_start = vec![-1i32; max_nodes];
    let mut node_elem_count = vec![0i32; max_nodes];
    let mut node_depth = vec![0i32; max_nodes];

    // Stack entry: (node_idx, start, end, depth).
    let mut stack: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(128);
    stack.push((0, 0, n_elems as i32, 0));
    let mut node_count: i32 = 1;

    while let Some((node_idx, start, end, depth)) = stack.pop() {
        let count = end - start;
        node_depth[node_idx as usize] = depth;
        if count <= max_leaf_size as i32 {
            node_elem_start[node_idx as usize] = start;
            node_elem_count[node_idx as usize] = count;
        } else {
            let mid = (start + end) / 2;
            let left_idx = node_count;
            let right_idx = node_count + 1;
            node_count += 2;

            node_left[node_idx as usize] = left_idx;
            node_right[node_idx as usize] = right_idx;

            // Match the Python ordering: right pushed first → left
            // popped first → tree explored in left-to-right order.
            stack.push((right_idx, mid, end, depth + 1));
            stack.push((left_idx, start, mid, depth + 1));
        }
    }

    // Step 4: leaf bboxes (parallel over node_count).
    let nc = node_count as usize;
    let mut node_bbox_min = vec![[0.0; 3]; nc];
    let mut node_bbox_max = vec![[0.0; 3]; nc];

    let leaf_bboxes: Vec<Option<([f64; 3], [f64; 3])>> = (0..nc)
        .into_par_iter()
        .map(|node_idx| {
            if node_elem_start[node_idx] < 0 {
                return None;
            }
            let start = node_elem_start[node_idx] as usize;
            let count = node_elem_count[node_idx] as usize;
            let first_elem = sorted_indices[start] as usize;
            let mut bmin = [
                bboxes_min[3 * first_elem],
                bboxes_min[3 * first_elem + 1],
                bboxes_min[3 * first_elem + 2],
            ];
            let mut bmax = [
                bboxes_max[3 * first_elem],
                bboxes_max[3 * first_elem + 1],
                bboxes_max[3 * first_elem + 2],
            ];
            for k in 1..count {
                let e = sorted_indices[start + k] as usize;
                for d in 0..3 {
                    let lo = bboxes_min[3 * e + d];
                    let hi = bboxes_max[3 * e + d];
                    if lo < bmin[d] {
                        bmin[d] = lo;
                    }
                    if hi > bmax[d] {
                        bmax[d] = hi;
                    }
                }
            }
            Some((bmin, bmax))
        })
        .collect();
    for (i, lb) in leaf_bboxes.into_iter().enumerate() {
        if let Some((bmin, bmax)) = lb {
            node_bbox_min[i] = bmin;
            node_bbox_max[i] = bmax;
        }
    }

    // Step 5: bottom-up bbox propagation.
    let max_depth = node_depth[..nc].iter().copied().max().unwrap_or(0) + 1;
    for depth in (0..max_depth).rev() {
        for node_idx in 0..nc {
            if node_depth[node_idx] == depth && node_left[node_idx] >= 0 {
                let l = node_left[node_idx] as usize;
                let r = node_right[node_idx] as usize;
                for d in 0..3 {
                    node_bbox_min[node_idx][d] = node_bbox_min[l][d].min(node_bbox_min[r][d]);
                    node_bbox_max[node_idx][d] = node_bbox_max[l][d].max(node_bbox_max[r][d]);
                }
            }
        }
    }

    // Truncate to actual node count.
    node_bbox_min.truncate(nc);
    node_bbox_max.truncate(nc);
    node_left.truncate(nc);
    node_right.truncate(nc);
    node_elem_start.truncate(nc);
    node_elem_count.truncate(nc);

    // Stash per-element bboxes for downstream queries (proximity /
    // intersection). Indexed by *original* element index, not sorted.
    let elem_bboxes_min: Vec<[f64; 3]> = (0..n_elems)
        .map(|i| {
            [
                bboxes_min[3 * i],
                bboxes_min[3 * i + 1],
                bboxes_min[3 * i + 2],
            ]
        })
        .collect();
    let elem_bboxes_max: Vec<[f64; 3]> = (0..n_elems)
        .map(|i| {
            [
                bboxes_max[3 * i],
                bboxes_max[3 * i + 1],
                bboxes_max[3 * i + 2],
            ]
        })
        .collect();

    Bvh {
        node_bbox_min,
        node_bbox_max,
        node_left,
        node_right,
        node_elem_start,
        node_elem_count,
        sorted_indices,
        elem_bboxes_min,
        elem_bboxes_max,
    }
}

// ---------------------------------------------------------------------------
// Element-BVH builders shared by proximity / intersection.

/// Build a BVH over `tris` (M, 3 indices into `verts`). Triangles are
/// the elements; per-element bboxes + centroids are computed in
/// parallel before the Morton-code build.
pub(crate) fn build_tri_bvh(verts: &[f64], tris: &[i32]) -> Bvh {
    let n = tris.len() / 3;
    let mut centroids = vec![0.0f64; n * 3];
    let mut bb_min = vec![0.0f64; n * 3];
    let mut bb_max = vec![0.0f64; n * 3];

    centroids
        .par_chunks_mut(3)
        .zip(bb_min.par_chunks_mut(3))
        .zip(bb_max.par_chunks_mut(3))
        .enumerate()
        .for_each(|(i, ((c, lo), hi))| {
            let t0 = tris[3 * i] as usize;
            let t1 = tris[3 * i + 1] as usize;
            let t2 = tris[3 * i + 2] as usize;
            for d in 0..3 {
                let v0 = verts[3 * t0 + d];
                let v1 = verts[3 * t1 + d];
                let v2 = verts[3 * t2 + d];
                c[d] = (v0 + v1 + v2) / 3.0;
                lo[d] = v0.min(v1).min(v2);
                hi[d] = v0.max(v1).max(v2);
            }
        });

    build_bvh(&centroids, &bb_min, &bb_max, DEFAULT_MAX_LEAF)
}

/// Build a BVH over `edges` (K, 2 indices into `verts`).
pub(crate) fn build_edge_bvh(verts: &[f64], edges: &[i32]) -> Bvh {
    let n = edges.len() / 2;
    let mut centroids = vec![0.0f64; n * 3];
    let mut bb_min = vec![0.0f64; n * 3];
    let mut bb_max = vec![0.0f64; n * 3];

    centroids
        .par_chunks_mut(3)
        .zip(bb_min.par_chunks_mut(3))
        .zip(bb_max.par_chunks_mut(3))
        .enumerate()
        .for_each(|(i, ((c, lo), hi))| {
            let e0 = edges[2 * i] as usize;
            let e1 = edges[2 * i + 1] as usize;
            for d in 0..3 {
                let v0 = verts[3 * e0 + d];
                let v1 = verts[3 * e1 + d];
                c[d] = (v0 + v1) / 2.0;
                lo[d] = v0.min(v1);
                hi[d] = v0.max(v1);
            }
        });

    build_bvh(&centroids, &bb_min, &bb_max, DEFAULT_MAX_LEAF)
}

/// Per-edge bboxes only (no BVH). Used to pre-filter edge / triangle
/// pairs.
pub(crate) fn compute_edge_bboxes(verts: &[f64], edges: &[i32]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let n = edges.len() / 2;
    let mut bb_min = vec![[0.0f64; 3]; n];
    let mut bb_max = vec![[0.0f64; 3]; n];
    bb_min
        .par_iter_mut()
        .zip(bb_max.par_iter_mut())
        .enumerate()
        .for_each(|(i, (lo, hi))| {
            let e0 = edges[2 * i] as usize;
            let e1 = edges[2 * i + 1] as usize;
            for d in 0..3 {
                let v0 = verts[3 * e0 + d];
                let v1 = verts[3 * e1 + d];
                lo[d] = v0.min(v1);
                hi[d] = v0.max(v1);
            }
        });
    (bb_min, bb_max)
}

/// Extract unique undirected edges from a triangle list, plus a
/// per-edge map to the (up to two) parent triangles.
///
/// Returns flat slices: `edges` is a Vec<i32> of length `2 * n_edges`;
/// `edge_to_tri` is a Vec<i32> of length `2 * n_edges`. The second
/// parent slot is `-1` for boundary edges. For non-manifold edges
/// shared by 3+ triangles, only the first two are recorded.
pub(crate) fn extract_edges_with_tri_map(tris: &[i32]) -> (Vec<i32>, Vec<i32>) {
    use std::collections::HashMap;
    let n_tris = tris.len() / 3;
    if n_tris == 0 {
        return (vec![], vec![]);
    }
    let mut map: HashMap<(i32, i32), [i32; 2]> = HashMap::new();
    let pairs = [(0usize, 1usize), (1, 2), (2, 0)];
    for ti in 0..n_tris {
        let t = [tris[3 * ti], tris[3 * ti + 1], tris[3 * ti + 2]];
        for &(a, b) in &pairs {
            let v0 = t[a];
            let v1 = t[b];
            let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            let entry = map.entry(key).or_insert([-1, -1]);
            if entry[0] < 0 {
                entry[0] = ti as i32;
            } else if entry[1] < 0 {
                entry[1] = ti as i32;
            }
        }
    }

    // Stable order: sort by (v_lo, v_hi) so edge indices are
    // deterministic across runs (the Python source argsorts on a
    // packed key, same ordering).
    let mut entries: Vec<((i32, i32), [i32; 2])> = map.into_iter().collect();
    entries.sort_unstable_by_key(|((a, b), _)| (*a, *b));

    let n_edges = entries.len();
    let mut edges_flat = Vec::with_capacity(n_edges * 2);
    let mut e2t_flat = Vec::with_capacity(n_edges * 2);
    for ((a, b), parents) in entries {
        edges_flat.push(a);
        edges_flat.push(b);
        e2t_flat.push(parents[0]);
        e2t_flat.push(parents[1]);
    }
    (edges_flat, e2t_flat)
}

/// Stack-DFS over `bvh`, calling `on_leaf_elem` once per leaf element
/// whose enclosing node bbox overlaps `query_bbox`. Used by the
/// proximity / intersection per-element scan kernels: each had a hand
/// rolled copy of this scaffold (≈25 lines) that differed only in
/// what's done at the leaf.
///
/// `on_leaf_elem` receives the unsorted element index (post-`sorted_indices`
/// mapping), matching what each call site reads anyway. The stack is
/// stack-allocated at `BVH_STACK_CAP`, same as the originals.
#[inline]
pub(crate) fn traverse_overlap<F: FnMut(i32)>(
    bvh: &Bvh,
    query_bbox_min: &[f64; 3],
    query_bbox_max: &[f64; 3],
    mut on_leaf_elem: F,
) {
    let mut stack = [0i32; BVH_STACK_CAP];
    let mut sp: usize = 0;
    stack[sp] = 0;
    sp += 1;

    while sp > 0 {
        sp -= 1;
        let node_idx = stack[sp];
        let ni = node_idx as usize;

        if !bbox_overlap(
            query_bbox_min,
            query_bbox_max,
            &bvh.node_bbox_min[ni],
            &bvh.node_bbox_max[ni],
        ) {
            continue;
        }

        if bvh.node_elem_start[ni] >= 0 {
            let start = bvh.node_elem_start[ni] as usize;
            let cnt = bvh.node_elem_count[ni] as usize;
            for k in 0..cnt {
                on_leaf_elem(bvh.sorted_indices[start + k]);
            }
        } else {
            stack[sp] = bvh.node_left[ni];
            sp += 1;
            stack[sp] = bvh.node_right[ni];
            sp += 1;
        }
    }
}

/// Find the index of the triangle on the BVH closest to `point`. Used
/// by `frame_mapping`; the frame solve happens inline at the
/// call site, so this only returns the index (no bary, no dist).
///
/// Returns `-1` for an empty BVH (zero elements). Callers receiving
/// `-1` must not index `tris_flat` / `verts_flat` with it.
#[inline]
pub(crate) fn closest_triangle_index(
    point: [f64; 3],
    bvh: &Bvh,
    verts_flat: &[f64],
    tris_flat: &[i32],
) -> i32 {
    // Empty BVH: no nodes to traverse. Bail before touching `node_bbox_min[0]`.
    if bvh.node_bbox_min.is_empty() {
        return -1;
    }

    let mut best_dist_sq = f64::INFINITY;
    let mut best_tri: i32 = 0;

    // Stack-allocated traversal stack: removes ~1M Vec heap allocs at
    // 1M queries. We never overflow in practice (see constants::BVH_STACK_CAP).
    let mut stack = [0i32; BVH_STACK_CAP];
    let mut sp: usize = 0;
    stack[sp] = 0;
    sp += 1;

    while sp > 0 {
        sp -= 1;
        let node_idx = stack[sp];
        let ni = node_idx as usize;
        let bbox_d_sq =
            point_to_bbox_dist_sq(point, &bvh.node_bbox_min[ni], &bvh.node_bbox_max[ni]);
        if bbox_d_sq >= best_dist_sq {
            continue;
        }
        if bvh.node_elem_start[ni] >= 0 {
            let start = bvh.node_elem_start[ni] as usize;
            let count = bvh.node_elem_count[ni] as usize;
            for k in 0..count {
                let ti = bvh.sorted_indices[start + k];
                let tu = ti as usize;
                let i0 = tris_flat[3 * tu] as usize;
                let i1 = tris_flat[3 * tu + 1] as usize;
                let i2 = tris_flat[3 * tu + 2] as usize;
                let a = [
                    verts_flat[3 * i0],
                    verts_flat[3 * i0 + 1],
                    verts_flat[3 * i0 + 2],
                ];
                let b = [
                    verts_flat[3 * i1],
                    verts_flat[3 * i1 + 1],
                    verts_flat[3 * i1 + 2],
                ];
                let c = [
                    verts_flat[3 * i2],
                    verts_flat[3 * i2 + 1],
                    verts_flat[3 * i2 + 2],
                ];
                let (closest, _bary) = closest_point_on_triangle(point, a, b, c);
                let diff = sub3(point, closest);
                let d_sq = dot3(diff, diff);
                if d_sq < best_dist_sq {
                    best_dist_sq = d_sq;
                    best_tri = ti;
                }
            }
        } else {
            let li = bvh.node_left[ni];
            let ri = bvh.node_right[ni];
            let ld = point_to_bbox_dist_sq(
                point,
                &bvh.node_bbox_min[li as usize],
                &bvh.node_bbox_max[li as usize],
            );
            let rd = point_to_bbox_dist_sq(
                point,
                &bvh.node_bbox_min[ri as usize],
                &bvh.node_bbox_max[ri as usize],
            );
            // Push farther child first so the closer one is popped first.
            if ld < rd {
                stack[sp] = ri;
                sp += 1;
                stack[sp] = li;
                sp += 1;
            } else {
                stack[sp] = li;
                sp += 1;
                stack[sp] = ri;
                sp += 1;
            }
        }
    }
    best_tri
}

// ---------------------------------------------------------------------------
// frame_mapping + interpolate_surface
//
// Frame B0 = [b1 | b2 | b3] where b1 = x1-x0, b2 = x2-x0, b3 = b1×b2 / ‖b1×b2‖.
// The 3×3 solve decouples:
//   c3 = (p - x0) · b3        (signed normal offset, world units)
//   [c1, c2]ᵀ = G⁻¹ [(p-x0)·b1, (p-x0)·b2]ᵀ   with Gram G = [[b1·b1, b1·b2], [b1·b2, b2·b2]]
// Cramer's rule gives det(G) = ‖b1 × b2‖² = n_sq.

/// Solve the triangle frame at (x0, x1, x2): the two edge vectors
/// b1 = x1-x0, b2 = x2-x0, their cross product n = b1×b2, and
/// n_sq = ‖n‖² (which is also det(G) for the Gram solve). Shared by
/// frame_mapping (projection) and interpolate_surface (reconstruction)
/// so the normal sign and the n_sq used by the FRAME_DEGEN_EPS guard
/// stay identical in both directions. The degenerate branch itself is
/// left to each caller because the two differ: frame_mapping zeroes all
/// coefs while interpolate_surface keeps the in-plane terms.
#[inline]
fn tri_frame(x0: [f64; 3], x1: [f64; 3], x2: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
    let b1 = sub3(x1, x0);
    let b2 = sub3(x2, x0);
    let n = cross3(b1, b2);
    let n_sq = dot3(n, n);
    (b1, b2, n, n_sq)
}

/// Compute the BVH then, for each input point, find its closest
/// triangle and return (tri_index, frame_coefs). Coefs are flat
/// row-major (n_orig * 3); the caller reshapes to (n_orig, 3) numpy.
pub fn frame_mapping(
    orig_vert_flat: &[f64],
    new_vert_flat: &[f64],
    new_tri_flat: &[i32],
) -> (Vec<i32>, Vec<f64>) {
    let n_orig = orig_vert_flat.len() / 3;
    let n_new_tri = new_tri_flat.len() / 3;
    debug_assert_eq!(orig_vert_flat.len(), n_orig * 3);
    debug_assert_eq!(new_tri_flat.len(), n_new_tri * 3);

    // Empty target surface: no triangles to map onto. Return `-1` per
    // point and zero coefs so callers can detect the no-mapping case
    // (used by `_mesh_.py:tetrahedralize` to raise a clean user-facing
    // error when fTetWild produces zero usable tets, e.g. for a flat
    // / coplanar input mesh assigned to a SOLID group).
    if n_new_tri == 0 {
        return (vec![-1; n_orig], vec![0.0; n_orig * 3]);
    }

    // Build the BVH over the target triangles (per-tri centroids +
    // bboxes are computed inside build_tri_bvh).
    let bvh = build_tri_bvh(new_vert_flat, new_tri_flat);

    // Per-point: closest-tri + frame coefs. Output is (tri_idx, c1, c2, c3).
    let mut tri_indices = vec![0i32; n_orig];
    let mut coefs = vec![0.0f64; n_orig * 3];

    tri_indices
        .par_iter_mut()
        .zip(coefs.par_chunks_mut(3))
        .enumerate()
        .for_each(|(i, (tri_idx, coef_row))| {
            let p = [
                orig_vert_flat[3 * i],
                orig_vert_flat[3 * i + 1],
                orig_vert_flat[3 * i + 2],
            ];
            let best_tri = closest_triangle_index(p, &bvh, new_vert_flat, new_tri_flat);
            *tri_idx = best_tri;

            let bt = best_tri as usize;
            let t0 = new_tri_flat[3 * bt] as usize;
            let t1 = new_tri_flat[3 * bt + 1] as usize;
            let t2 = new_tri_flat[3 * bt + 2] as usize;
            let x0 = [
                new_vert_flat[3 * t0],
                new_vert_flat[3 * t0 + 1],
                new_vert_flat[3 * t0 + 2],
            ];
            let x1 = [
                new_vert_flat[3 * t1],
                new_vert_flat[3 * t1 + 1],
                new_vert_flat[3 * t1 + 2],
            ];
            let x2 = [
                new_vert_flat[3 * t2],
                new_vert_flat[3 * t2 + 1],
                new_vert_flat[3 * t2 + 2],
            ];
            let (b1, b2, n, n_sq) = tri_frame(x0, x1, x2);

            if n_sq < FRAME_DEGEN_EPS {
                coef_row[0] = 0.0;
                coef_row[1] = 0.0;
                coef_row[2] = 0.0;
                return;
            }

            let d = sub3(p, x0);
            let db1 = dot3(d, b1);
            let db2 = dot3(d, b2);
            let b1b1 = dot3(b1, b1);
            let b2b2 = dot3(b2, b2);
            let b1b2 = dot3(b1, b2);

            coef_row[0] = (b2b2 * db1 - b1b2 * db2) / n_sq;
            coef_row[1] = (b1b1 * db2 - b1b2 * db1) / n_sq;
            // c3 = (p - x0) · n̂ = ((p - x0) · n) / ‖n‖
            let inv_nlen = 1.0 / n_sq.sqrt();
            coef_row[2] = dot3(d, n) * inv_nlen;
        });

    (tri_indices, coefs)
}

/// Reconstruct world positions from the frame embedding. Includes
/// the degenerate-deformed-triangle fallback that drops the c3 term.
///
/// Callers must filter the `frame_mapping` empty-target sentinel:
/// rows whose `tri_indices[i] < 0` have no source triangle (see
/// `frame_mapping`'s `n_new_tri == 0` path) and are left at the
/// zero output position here rather than indexing with a wrapped
/// `usize`.
pub fn interpolate_surface(
    deformed_vert_flat: &[f64],
    surf_tri_flat: &[i32],
    tri_indices: &[i32],
    coefs_flat: &[f64],
) -> Vec<f64> {
    let n = tri_indices.len();
    debug_assert_eq!(coefs_flat.len(), n * 3);

    let mut out = vec![0.0f64; n * 3];
    out.par_chunks_mut(3).enumerate().for_each(|(i, row)| {
        // Guard the frame_mapping `-1` sentinel (empty target surface)
        // before indexing: `-1 as usize` would wrap to usize::MAX and
        // panic on surf_tri_flat. Leave `row` at its zero init, which
        // matches frame_mapping's zero coefs for the no-mapping case.
        let ti_signed = tri_indices[i];
        if ti_signed < 0 {
            return;
        }
        let ti = ti_signed as usize;
        let t0 = surf_tri_flat[3 * ti] as usize;
        let t1 = surf_tri_flat[3 * ti + 1] as usize;
        let t2 = surf_tri_flat[3 * ti + 2] as usize;
        let c0 = coefs_flat[3 * i];
        let c1 = coefs_flat[3 * i + 1];
        let c2 = coefs_flat[3 * i + 2];

        let x0 = [
            deformed_vert_flat[3 * t0],
            deformed_vert_flat[3 * t0 + 1],
            deformed_vert_flat[3 * t0 + 2],
        ];
        let x1 = [
            deformed_vert_flat[3 * t1],
            deformed_vert_flat[3 * t1 + 1],
            deformed_vert_flat[3 * t1 + 2],
        ];
        let x2 = [
            deformed_vert_flat[3 * t2],
            deformed_vert_flat[3 * t2 + 1],
            deformed_vert_flat[3 * t2 + 2],
        ];
        let (b1, b2, n, n_sq) = tri_frame(x0, x1, x2);

        if n_sq < FRAME_DEGEN_EPS {
            // Degenerate deformed triangle: drop the normal term to
            // avoid NaN. c2 was relative to a normal that no longer
            // exists.
            row[0] = x0[0] + c0 * b1[0] + c1 * b2[0];
            row[1] = x0[1] + c0 * b1[1] + c1 * b2[1];
            row[2] = x0[2] + c0 * b1[2] + c1 * b2[2];
        } else {
            let inv_nlen = 1.0 / n_sq.sqrt();
            row[0] = x0[0] + c0 * b1[0] + c1 * b2[0] + c2 * n[0] * inv_nlen;
            row[1] = x0[1] + c0 * b1[1] + c1 * b2[1] + c2 * n[1] * inv_nlen;
            row[2] = x0[2] + c0 * b1[2] + c1 * b2[2] + c2 * n[2] * inv_nlen;
        }
    });
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
    fn closest_point_inside_triangle() {
        // Triangle in the xy-plane; query point straight above the
        // centroid → projects to centroid with c=(1/3,1/3,1/3).
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [1.0 / 3.0, 1.0 / 3.0, 0.5];
        let (closest, bary) = closest_point_on_triangle(p, a, b, c);
        assert!((closest[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((closest[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!(closest[2].abs() < 1e-12);
        for k in 0..3 {
            assert!((bary[k] - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn closest_point_on_vertex_a() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [-1.0, -1.0, 0.0];
        let (closest, bary) = closest_point_on_triangle(p, a, b, c);
        assert_eq!(closest, a);
        assert_eq!(bary, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn frame_mapping_on_surface_recovers_zero_normal() {
        // 3-vertex flat triangle in xy-plane. Query points = vertices.
        let verts = flat3(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let tris = flat3i(&[[0, 1, 2]]);
        let q = verts.clone();
        let (tri_idx, coefs) = frame_mapping(&q, &verts, &tris);
        assert_eq!(tri_idx, vec![0, 0, 0]);
        // c3 (normal offset) should be ~0 for on-surface points.
        for i in 0..3 {
            assert!(coefs[3 * i + 2].abs() < 1e-12, "c3[{i}] = {}", coefs[3 * i + 2]);
        }
        let back = interpolate_surface(&verts, &tris, &tri_idx, &coefs);
        for k in 0..verts.len() {
            assert!((back[k] - verts[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn frame_normal_offset_preserved_under_inplane_scale() {
        // Flat xy mesh, point sits 0.01 above the surface. Shrink mesh
        // 50% in xy: c3 ⇒ 0.01 unchanged, in-plane coords scale.
        let verts = flat3(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let tris = flat3i(&[[0, 1, 2]]);
        let q = vec![0.25, 0.25, 0.01];
        let (tri_idx, coefs) = frame_mapping(&q, &verts, &tris);
        assert!((coefs[2] - 0.01).abs() < 1e-12, "c3 = {}", coefs[2]);

        let deformed: Vec<f64> = verts.iter().enumerate().map(|(i, v)| {
            if i % 3 == 0 || i % 3 == 1 { v * 0.5 } else { *v }
        }).collect();
        let back = interpolate_surface(&deformed, &tris, &tri_idx, &coefs);
        // y-position lifts to 0.01 unchanged; x/y in-plane halve.
        assert!((back[2] - 0.01).abs() < 1e-12);
        assert!((back[0] - 0.125).abs() < 1e-12, "x = {}", back[0]);
        assert!((back[1] - 0.125).abs() < 1e-12);
    }

    #[test]
    fn frame_degenerate_triangle_zeros_coefs() {
        // Three collinear vertices ⇒ zero-area triangle.
        let verts = flat3(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let tris = flat3i(&[[0, 1, 2]]);
        let q = vec![0.5, 0.3, 0.1];
        let (_tri, coefs) = frame_mapping(&q, &verts, &tris);
        for c in &coefs {
            assert!(c.is_finite(), "coef NaN/inf: {c}");
            assert_eq!(*c, 0.0);
        }
    }

    #[test]
    fn bvh_picks_correct_triangle() {
        // Two well-separated triangles. Query closer to triangle 1
        // must return tri index 1.
        let verts = flat3(&[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0],
        ]);
        let tris = flat3i(&[[0, 1, 2], [3, 4, 5]]);
        // Build the same way frame_mapping does:
        let mut centroids = vec![0.0; 6];
        let mut bmin = vec![0.0; 6];
        let mut bmax = vec![0.0; 6];
        for i in 0..2 {
            let t = [tris[3*i], tris[3*i+1], tris[3*i+2]].map(|x| x as usize);
            for d in 0..3 {
                let v0 = verts[3*t[0]+d]; let v1 = verts[3*t[1]+d]; let v2 = verts[3*t[2]+d];
                centroids[3*i+d] = (v0+v1+v2)/3.0;
                bmin[3*i+d] = v0.min(v1).min(v2);
                bmax[3*i+d] = v0.max(v1).max(v2);
            }
        }
        let bvh = build_bvh(&centroids, &bmin, &bmax, DEFAULT_MAX_LEAF);
        let near_tri1 = closest_triangle_index([10.5, 0.3, 0.0], &bvh, &verts, &tris);
        assert_eq!(near_tri1, 1);
        let near_tri0 = closest_triangle_index([0.3, 0.3, 0.0], &bvh, &verts, &tris);
        assert_eq!(near_tri0, 0);
    }

    #[test]
    fn bvh_empty_input() {
        let bvh = build_bvh(&[], &[], &[], DEFAULT_MAX_LEAF);
        assert!(bvh.node_bbox_min.is_empty());
        assert!(bvh.sorted_indices.is_empty());
        // The traversal helper must also accept an empty BVH and return
        // a sentinel, not index `node_bbox_min[0]` (the panic that hit
        // user issue #18 when a Plane was assigned to a SOLID group).
        let idx = closest_triangle_index([0.0, 0.0, 0.0], &bvh, &[], &[]);
        assert_eq!(idx, -1);
    }

    #[test]
    fn frame_mapping_empty_target_returns_minus_one() {
        // Non-empty source verts, zero target triangles: the post-
        // `tet_extract_surface` shape when fTetWild produces no usable
        // tets. `frame_mapping` must not panic; it returns `-1` per
        // input point so the Python wrapper can detect the no-mapping
        // case and raise a clear error.
        let orig: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (tri_idx, coefs) = frame_mapping(&orig, &[], &[]);
        assert_eq!(tri_idx, vec![-1; 4]);
        assert_eq!(coefs, vec![0.0; 12]);
    }
}
