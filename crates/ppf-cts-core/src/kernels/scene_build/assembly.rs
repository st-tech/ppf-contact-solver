// File: crates/ppf-cts-core/src/kernels/scene_build/assembly.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Scene.build core assembly: rod/tri proximity check, merge-pair
// grouping, and the dyn/static scene assembly kernels that scatter
// per-object data into the global FixedScene buffers.
//
// Pin / Operation construction stays Python-side because PinData and
// the Operation subclasses are Python dataclasses. This kernel does
// the bulk numeric work and returns the assembled arrays + counters
// per object so the Python wrapper just builds Python objects with
// the prepared inputs.

use super::super::geom_util::barycentric_clamp_project;
use std::collections::BTreeSet;

/// Typed errors emitted by the dyn/static scene-assembly kernels.
/// Replaces the previous `Result<_, String>` signatures so the
/// PyO3 layer can map each variant to the right `PyErr` class.
#[derive(Debug, thiserror::Error)]
pub enum SceneAssemblyError {
    #[error("missing map_by_name entry for {name:?}")]
    MissingMapEntry { name: String },
    #[error("missing displacement index for {name:?}")]
    MissingDisplacementIndex { name: String },
    #[error("stitch ind/w row mismatch for {name:?}: ind rows {ind_rows}, w rows {w_rows}")]
    StitchRowMismatch {
        name: String,
        ind_rows: usize,
        w_rows: usize,
    },
    #[error("cross-stitch missing source map: {name:?}")]
    CrossStitchMissingSource { name: String },
    #[error("cross-stitch missing target map: {name:?}")]
    CrossStitchMissingTarget { name: String },
    #[error("static {name:?}: vertex buffer not multiple of 3")]
    StaticVertexNotMultipleOf3 { name: String },
    #[error("missing displacement entry for static {name:?}")]
    MissingStaticDisplacement { name: String },
}

// ---------------------------------------------------------------------------
// Rod / tri contact-offset proximity check.

/// Rod-tri proximity loop. Walks every rod whose `rod_offset > 0`.
/// Returns `Err(RodTriOffsetViolation::EdgeShorterThanOffset { offset,
/// edge_len })` for the first rod whose edge is no longer than its
/// offset, or `Err(RodTriOffsetViolation::VertexInsideOffset { rod_idx,
/// tri_idx, dist, required })` for the first rod endpoint lying within
/// `rod_offset + tri_offset` of a triangle.
///
/// Inputs:
///   * `verts`: world-space `(N, 3)` flat f64.
///   * `rods`: `(M, 2)` indices into `verts`.
///   * `tris`: `(K, 3)` indices into `verts`.
///   * `tri_offset`: `(K,)`. Empty means all-zero.
///   * `rod_offset`: `(M,)`. Empty means all-zero (and the function
///      becomes a no-op).
///
/// Returns `Ok(())` when there is no violation.
#[derive(Debug, thiserror::Error)]
pub enum RodTriOffsetViolation {
    #[error(
        "Contact offset ({offset:.4}) exceeds rod edge length ({edge_len:.4}). \
         Reduce the offset or increase mesh resolution."
    )]
    EdgeShorterThanOffset { offset: f64, edge_len: f64 },
    #[error(
        "Rod vertex within contact offset of triangle (rod {rod_idx}, tri {tri_idx}, \
         dist={dist:.4} < offset={required:.4}). Reduce contact-offset or move the curve away."
    )]
    VertexInsideOffset {
        rod_idx: usize,
        tri_idx: usize,
        dist: f64,
        required: f64,
    },
}

pub fn rod_tri_contact_offset_check(
    verts: &[f64],
    rods: &[[u32; 2]],
    tris: &[[u32; 3]],
    tri_offset: &[f64],
    rod_offset: &[f64],
) -> Result<(), RodTriOffsetViolation> {
    let n_rods = rods.len();
    if rod_offset.is_empty() || n_rods == 0 {
        return Ok(());
    }
    let v = |idx: u32| -> [f64; 3] {
        let i = 3 * idx as usize;
        [verts[i], verts[i + 1], verts[i + 2]]
    };
    for (ri, rod) in rods.iter().enumerate() {
        let offset = if ri < rod_offset.len() { rod_offset[ri] } else { 0.0 };
        if offset <= 0.0 {
            continue;
        }
        let p0 = v(rod[0]);
        let p1 = v(rod[1]);
        let edge = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edge_len = (edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]).sqrt();
        if edge_len <= offset {
            return Err(RodTriOffsetViolation::EdgeShorterThanOffset {
                offset,
                edge_len,
            });
        }
        for &endpoint in &[rod[0], rod[1]] {
            let p = v(endpoint);
            for (ti, tri) in tris.iter().enumerate() {
                let t_off = if ti < tri_offset.len() { tri_offset[ti] } else { 0.0 };
                let required = offset + t_off;
                let v0 = v(tri[0]);
                let v1 = v(tri[1]);
                let v2 = v(tri[2]);
                let v0p = [p[0] - v0[0], p[1] - v0[1], p[2] - v0[2]];
                let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let d00 = e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2];
                let d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
                let d11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
                let d20 = v0p[0] * e0[0] + v0p[1] * e0[1] + v0p[2] * e0[2];
                let d21 = v0p[0] * e1[0] + v0p[1] * e1[1] + v0p[2] * e1[2];
                let denom = d00 * d11 - d01 * d01;
                if denom.abs() <= 1e-30 {
                    continue;
                }
                let inv_denom = 1.0 / denom;
                let b = (d11 * d20 - d01 * d21) * inv_denom;
                let c = (d00 * d21 - d01 * d20) * inv_denom;
                let a = 1.0 - b - c;
                // Clamp to the triangle and project; this keeps its own
                // 1e-30 degeneracy gate above so the shared helper only
                // runs the convex-projection tail.
                let (_bary, _proj, dist) = barycentric_clamp_project(a, b, c, p, v0, v1, v2);
                if dist < required {
                    return Err(RodTriOffsetViolation::VertexInsideOffset {
                        rod_idx: ri,
                        tri_idx: ti,
                        dist,
                        required,
                    });
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scene.build core assembly: walks every dynamic object, scatters
// vertex/color/velocity into the global buffer, concatenates rod/shell/
// tet topology with merge-pair filtering, builds UV / dyn-color arrays
// per face, gathers cross-stitch indices, computes the global pinned-
// vertex set, and packages static geometry too.

/// One dynamic object as input to the assembly kernel. Topology
/// indices are local; the kernel maps them through `map_by_name` to
/// global slots. `vertex` / `color` are shape `(n_verts, 3)`; `uv`
/// is `(n_faces, 6)` flattened (per-face 3x2). `param_keys` /
/// `param_count_hint` carry the param schema so the caller can
/// tell which keys to extend without reading them in Rust (values
/// stay Python objects).
#[derive(Debug)]
pub struct AssembleObject<'a> {
    pub name: &'a str,
    pub obj_type: &'a str, // "rod" | "tri" | "tet"
    pub vertex: &'a [f64], // (n_verts, 3) row-major
    pub color: &'a [f64],  // (n_verts, 3) row-major
    pub velocity: [f64; 3],
    pub edges: Option<&'a [[u32; 2]]>,
    pub faces: Option<&'a [[u32; 3]]>,
    pub tets: Option<&'a [[u32; 4]]>,
    pub uv: Option<&'a [f64]>, // (n_faces, 6) row-major or None
    pub dynamic_color: u8,
    pub dynamic_intensity: f64,
    pub pinned_indices: &'a [i64], // local indices of pinned verts
    pub stitch_ind: Option<&'a [i64]>, // flat (M, 3), (M, 4), or (M, 6)
    pub stitch_ind_cols: usize,        // 3, 4, or 6
    pub stitch_w: Option<&'a [f64]>,   // flat (M, K) where K = 2, 4, or 6
    pub stitch_w_cols: usize,
    /// Per-object stitch stiffness applied to every stitch row this
    /// object owns (its loose-edge / intra-object stitches).
    pub stitch_stiffness: f64,
    pub position: [f64; 3], // displacement entry
}

/// One static object as input to the assembly kernel. `vertex` is
/// the *pre-transformed* world-space vertex buffer (already through
/// `Object.apply_transform(..., translate=False)` Python-side, since
/// the normalize branch needs object-private state).
#[derive(Debug)]
pub struct AssembleStaticObject<'a> {
    pub name: &'a str,
    pub vertex_world: &'a [f64], // (n_verts, 3) row-major, already transformed
    pub color: [f64; 3],
    pub faces: Option<&'a [[u32; 3]]>,
    pub n_faces: usize,
    pub position: [f64; 3], // displacement entry
}

/// Cross-object stitch entry. `ind` is `(K, 6)` row-major flat where
/// indices 0..2 (source barycentric) belong to `source_name` and
/// indices 3..5 (target barycentric) belong to `target_name`.
/// `weights` is `(K, 6)`, source weights in 0..2 and target weights in
/// 3..5 (each triple sums to 1; a non-SOLID source degenerates to
/// `[s, s, s]` / `[1, 0, 0]`).
#[derive(Debug)]
pub struct CrossStitch<'a> {
    pub source_name: &'a str,
    pub target_name: &'a str,
    pub ind: &'a [i64],   // K * 6
    pub weights: &'a [f64], // K * 6
    pub k: usize,
    /// Per-pair stitch stiffness applied to every row of this cross-stitch.
    pub stitch_stiffness: f64,
}

/// Per-object output produced by `assemble_dyn_scene`. Used by the
/// Python wrapper to know how many param entries to extend per
/// (object, element-kind) pair, and which face indices survived
/// merge filtering for UV / dyn-color rebuild.
#[derive(Debug, Default, Clone)]
pub struct AssembleObjectStats {
    /// Count of rod edges contributed by this object after filtering.
    pub rod_added: usize,
    /// Count of tri (shell-or-tet-surface) entries after filtering.
    pub tri_added: usize,
    /// Count of tet entries after filtering.
    pub tet_added: usize,
    /// True if this object is a pure shell (tri but no tet) so the
    /// caller knows whether it contributed UVs to the shell list.
    pub is_pure_shell: bool,
    /// Indices of rods that survived (relative to local rod list);
    /// used by the caller for collider flag building.
    pub rod_keep_mask: Vec<bool>,
    /// Indices of tris that survived (relative to local tri list);
    /// used by the caller for UV / collider flag rebuild.
    pub tri_keep_mask: Vec<bool>,
}

/// Output of the dyn-scene assembly. Numeric arrays are flat
/// row-major; the caller reshapes when wrapping into numpy.
#[derive(Debug, Default)]
pub struct AssembleResult {
    pub concat_count: usize,
    pub concat_vert: Vec<f64>,        // (concat_count, 3)
    pub concat_vert_dmap: Vec<u32>,   // (concat_count,)
    pub concat_color: Vec<f64>,       // (concat_count, 3)
    pub concat_vel: Vec<f64>,         // (concat_count, 3)
    pub concat_displacement: Vec<f64>, // (n_objects, 3) flat: one row per object (dyn + static) in insertion order
    pub concat_rod: Vec<u32>,         // (rod_count, 2) flat
    pub concat_tri: Vec<u32>,         // (n_tri, 3) flat
    pub concat_tet: Vec<u32>,         // (n_tet, 4) flat
    pub concat_uv: Vec<f64>,          // (shell_count, 6) flat (per-face (3,2))
    pub concat_dyn_tri_color: Vec<u8>,
    pub concat_dyn_tri_intensity: Vec<f64>,
    pub concat_rod_is_collider: Vec<u8>,
    pub concat_tri_is_collider: Vec<u8>,
    pub concat_stitch_ind: Vec<i64>,  // (S, 6) flat
    pub concat_stitch_w: Vec<f64>,    // (S, 6) flat
    pub concat_stitch_stiffness: Vec<f32>, // (S,) one per stitch row
    pub rod_count: usize,
    pub shell_count: usize,
    pub per_object: Vec<(String, AssembleObjectStats)>,
}

/// Walk every dynamic object and assemble the global numeric arrays
/// the FixedScene constructor consumes. `map_by_name` is the per-
/// object local->global map (from `build_index_map`). `concat_count`
/// is the total global vertex count. `displacement_index` carries the
/// `dmap` ordering: `(name, index_in_concat_displacement)` for every
/// object (dyn + static) in the original insertion order. The kernel
/// emits a flat `concat_displacement` aligned with that ordering.
#[allow(clippy::too_many_arguments)]
pub fn assemble_dyn_scene(
    objects: &[AssembleObject<'_>],
    map_by_name: &std::collections::HashMap<String, Vec<i64>>,
    concat_count: usize,
    displacement_index: &[(String, [f64; 3])],
    cross_stitches: &[CrossStitch<'_>],
) -> Result<AssembleResult, SceneAssemblyError> {
    let mut out = AssembleResult {
        concat_count,
        concat_vert: vec![0.0; 3 * concat_count],
        concat_vert_dmap: vec![0u32; concat_count],
        concat_color: vec![0.0; 3 * concat_count],
        concat_vel: vec![0.0; 3 * concat_count],
        ..Default::default()
    };

    // ------------------------------------------------------------------
    // Build dmap (name -> idx into concat_displacement) and the
    // displacement vector itself. Mirrors the
    // `for name, obj in self._object.items(): dmap[name] = ...; concat_displacement.append(obj.position)`
    // loop. Caller passes the *insertion order* (dyn + static).
    let mut dmap: std::collections::HashMap<String, usize> =
        std::collections::HashMap::with_capacity(displacement_index.len());
    let mut disp_buf: Vec<f64> = Vec::with_capacity(3 * displacement_index.len());
    for (i, (name, pos)) in displacement_index.iter().enumerate() {
        dmap.insert(name.clone(), i);
        disp_buf.extend_from_slice(pos);
    }
    out.concat_displacement = disp_buf;

    // ------------------------------------------------------------------
    // Step 1: scatter vertex / color / velocity / dmap for every dyn
    // object's vertices (concat_vert[map] = vert; concat_vel[map] =
    // velocity; ...).
    for obj in objects {
        let map = map_by_name
            .get(obj.name)
            .ok_or_else(|| SceneAssemblyError::MissingMapEntry {
                name: obj.name.to_string(),
            })?;
        let dmap_idx = *dmap
            .get(obj.name)
            .ok_or_else(|| SceneAssemblyError::MissingDisplacementIndex {
                name: obj.name.to_string(),
            })? as u32;
        let n = map.len();
        for i in 0..n {
            let g = map[i] as usize;
            // Vertex / color / velocity scatter (last writer wins, like Python).
            out.concat_vert[3 * g] = obj.vertex[3 * i];
            out.concat_vert[3 * g + 1] = obj.vertex[3 * i + 1];
            out.concat_vert[3 * g + 2] = obj.vertex[3 * i + 2];
            out.concat_color[3 * g] = obj.color[3 * i];
            out.concat_color[3 * g + 1] = obj.color[3 * i + 1];
            out.concat_color[3 * g + 2] = obj.color[3 * i + 2];
            out.concat_vel[3 * g] = obj.velocity[0];
            out.concat_vel[3 * g + 1] = obj.velocity[1];
            out.concat_vel[3 * g + 2] = obj.velocity[2];
            out.concat_vert_dmap[g] = dmap_idx;
        }
    }

    // Per-object stats placeholder list, in the input order, plus a
    // name -> index map so per-object stat lookups are O(1). The
    // Steps 2-5 below each iterate `for obj in objects` and look up
    // the stats slot four times per iteration; without the map that
    // would be O(n_objects^2) work just to thread per-object stats.
    let mut stats_by_name: Vec<(String, AssembleObjectStats)> = objects
        .iter()
        .map(|o| (o.name.to_string(), AssembleObjectStats::default()))
        .collect();
    let stat_index_by_name: std::collections::HashMap<String, usize> = stats_by_name
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (n.clone(), i))
        .collect();
    let stat_index = |name: &str| -> Option<usize> {
        stat_index_by_name.get(name).copied()
    };

    // ------------------------------------------------------------------
    // Step 2: rod assembly. Walk every rod-typed object's edges,
    // remap through map, and compute the per-edge collider flag (all
    // endpoints pinned).
    for obj in objects {
        if obj.obj_type != "rod" {
            continue;
        }
        let edges = match obj.edges {
            Some(e) => e,
            None => continue,
        };
        let map = map_by_name.get(obj.name).unwrap();
        let pinned: BTreeSet<i64> = obj.pinned_indices.iter().copied().collect();
        let mut keep_mask = Vec::with_capacity(edges.len());
        let mut added = 0usize;
        for e in edges {
            let a = map[e[0] as usize];
            let b = map[e[1] as usize];
            keep_mask.push(true);
            out.concat_rod.push(a as u32);
            out.concat_rod.push(b as u32);
            // Collider flag.
            let is_coll = pinned.contains(&(e[0] as i64))
                && pinned.contains(&(e[1] as i64));
            out.concat_rod_is_collider.push(if is_coll { 1 } else { 0 });
            added += 1;
        }
        let si = stat_index(obj.name).unwrap();
        stats_by_name[si].1.rod_added = added;
        stats_by_name[si].1.rod_keep_mask = keep_mask;
    }
    out.rod_count = out.concat_rod.len() / 2;

    // ------------------------------------------------------------------
    // Step 3: shell triangles (tri-only objects, no tet). UVs +
    // dyn-color get appended only here.
    for obj in objects {
        if obj.tets.is_some() || obj.faces.is_none() {
            continue;
        }
        let faces = obj.faces.unwrap();
        let map = map_by_name.get(obj.name).unwrap();
        let pinned: BTreeSet<i64> = obj.pinned_indices.iter().copied().collect();
        let mut keep_mask = Vec::with_capacity(faces.len());
        let mut added = 0usize;
        for (fi, face) in faces.iter().enumerate() {
            let a = map[face[0] as usize];
            let b = map[face[1] as usize];
            let c = map[face[2] as usize];
            keep_mask.push(true);
            out.concat_tri.push(a as u32);
            out.concat_tri.push(b as u32);
            out.concat_tri.push(c as u32);
            // UV row (6 floats), pad with zeros when the object has no UVs.
            if let Some(uv) = obj.uv {
                let off = 6 * fi;
                out.concat_uv.extend_from_slice(&uv[off..off + 6]);
            } else {
                out.concat_uv.extend_from_slice(&[0.0; 6]);
            }
            out.concat_dyn_tri_color.push(obj.dynamic_color);
            out.concat_dyn_tri_intensity.push(obj.dynamic_intensity);
            // Collider flag (all original endpoints pinned).
            let is_coll = pinned.contains(&(face[0] as i64))
                && pinned.contains(&(face[1] as i64))
                && pinned.contains(&(face[2] as i64));
            out.concat_tri_is_collider.push(if is_coll { 1 } else { 0 });
            added += 1;
        }
        let si = stat_index(obj.name).unwrap();
        stats_by_name[si].1.tri_added = added;
        stats_by_name[si].1.tri_keep_mask = keep_mask;
        stats_by_name[si].1.is_pure_shell = true;
    }
    out.shell_count = out.concat_tri.len() / 3;

    // ------------------------------------------------------------------
    // Step 4: tet surface triangles (objects with both tris and tets).
    // Mirrors the "assembling solid surfaces" loop. No UV; dyn-color
    // still appended.
    for obj in objects {
        if obj.tets.is_none() || obj.faces.is_none() {
            continue;
        }
        let faces = obj.faces.unwrap();
        let map = map_by_name.get(obj.name).unwrap();
        let pinned: BTreeSet<i64> = obj.pinned_indices.iter().copied().collect();
        let mut keep_mask = Vec::with_capacity(faces.len());
        let mut added = 0usize;
        for face in faces.iter() {
            let a = map[face[0] as usize];
            let b = map[face[1] as usize];
            let c = map[face[2] as usize];
            keep_mask.push(true);
            out.concat_tri.push(a as u32);
            out.concat_tri.push(b as u32);
            out.concat_tri.push(c as u32);
            out.concat_dyn_tri_color.push(obj.dynamic_color);
            out.concat_dyn_tri_intensity.push(obj.dynamic_intensity);
            let is_coll = pinned.contains(&(face[0] as i64))
                && pinned.contains(&(face[1] as i64))
                && pinned.contains(&(face[2] as i64));
            out.concat_tri_is_collider.push(if is_coll { 1 } else { 0 });
            added += 1;
        }
        let si = stat_index(obj.name).unwrap();
        stats_by_name[si].1.tri_added = added;
        stats_by_name[si].1.tri_keep_mask = keep_mask;
        stats_by_name[si].1.is_pure_shell = false;
    }

    // ------------------------------------------------------------------
    // Step 5: tet assembly.
    for obj in objects {
        if obj.tets.is_none() {
            continue;
        }
        let tets = obj.tets.unwrap();
        let map = map_by_name.get(obj.name).unwrap();
        let mut added = 0usize;
        for tet in tets.iter() {
            let a = map[tet[0] as usize];
            let b = map[tet[1] as usize];
            let c = map[tet[2] as usize];
            let d = map[tet[3] as usize];
            out.concat_tet.push(a as u32);
            out.concat_tet.push(b as u32);
            out.concat_tet.push(c as u32);
            out.concat_tet.push(d as u32);
            added += 1;
        }
        let si = stat_index(obj.name).unwrap();
        stats_by_name[si].1.tet_added = added;
    }

    // ------------------------------------------------------------------
    // Step 6: per-object stitches. Map local indices through map and
    // pad legacy 3-tuple / 4-tuple / 2-weight entries to the 6-column
    // barycentric-barycentric layout. Legacy per-object stitches treat
    // the source as a single vertex, so the source side degenerates to
    // [src, src, src] / [1, 0, 0]; the legacy target slots move into the
    // 3..5 target columns (the weight-shift).
    for obj in objects {
        let (ind, w) = match (obj.stitch_ind, obj.stitch_w) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        if obj.stitch_ind_cols == 0 || obj.stitch_w_cols == 0 {
            continue;
        }
        let m = ind.len() / obj.stitch_ind_cols;
        if w.len() / obj.stitch_w_cols != m {
            return Err(SceneAssemblyError::StitchRowMismatch {
                name: obj.name.to_string(),
                ind_rows: m,
                w_rows: w.len() / obj.stitch_w_cols,
            });
        }
        let map = map_by_name.get(obj.name).unwrap();
        for r in 0..m {
            // ind row -> 6-wide [s, s, s, t0, t1, t2]
            let row_ind = &ind[r * obj.stitch_ind_cols..(r + 1) * obj.stitch_ind_cols];
            let mapped: Vec<i64> = row_ind.iter().map(|&v| map[v as usize]).collect();
            let padded: [i64; 6] = if obj.stitch_ind_cols == 6 {
                [mapped[0], mapped[1], mapped[2], mapped[3], mapped[4], mapped[5]]
            } else if obj.stitch_ind_cols == 3 {
                // legacy [src, t0, t1] -> [src, src, src, t0, t1, t1]
                [mapped[0], mapped[0], mapped[0], mapped[1], mapped[2], mapped[2]]
            } else {
                // legacy [src, t0, t1, t2] -> [src, src, src, t0, t1, t2]
                [mapped[0], mapped[0], mapped[0], mapped[1], mapped[2], mapped[3]]
            };
            out.concat_stitch_ind.extend_from_slice(&padded);
            // w row -> 6-wide [1, 0, 0, wt0, wt1, wt2] (source degenerate)
            let row_w = &w[r * obj.stitch_w_cols..(r + 1) * obj.stitch_w_cols];
            let mut padded_w = [0.0f64; 6];
            padded_w[0] = 1.0;
            if obj.stitch_w_cols == 6 {
                padded_w.copy_from_slice(row_w);
            } else if obj.stitch_w_cols == 4 {
                // legacy [ws, wt0, wt1, wt2] -> [1, 0, 0, wt0, wt1, wt2]
                padded_w[3] = row_w[1];
                padded_w[4] = row_w[2];
                padded_w[5] = row_w[3];
            } else {
                // legacy 1/2-weight target bary -> [1, 0, 0, w0, w1, 0]
                if !row_w.is_empty() {
                    padded_w[3] = row_w[0];
                }
                if row_w.len() >= 2 {
                    padded_w[4] = row_w[1];
                }
            }
            out.concat_stitch_w.extend_from_slice(&padded_w);
            // One stiffness per stitch row, from the owning object.
            out.concat_stitch_stiffness.push(obj.stitch_stiffness as f32);
        }
    }

    // ------------------------------------------------------------------
    // Step 7: cross-object stitches. Translate per-object indices.
    for cs in cross_stitches {
        let src_map = map_by_name.get(cs.source_name).ok_or_else(|| {
            SceneAssemblyError::CrossStitchMissingSource {
                name: cs.source_name.to_string(),
            }
        })?;
        let tgt_map = map_by_name.get(cs.target_name).ok_or_else(|| {
            SceneAssemblyError::CrossStitchMissingTarget {
                name: cs.target_name.to_string(),
            }
        })?;
        for r in 0..cs.k {
            let off = 6 * r;
            // Source barycentric (slots 0..2) -> source map; target
            // barycentric (slots 3..5) -> target map.
            out.concat_stitch_ind.push(src_map[cs.ind[off] as usize]);
            out.concat_stitch_ind.push(src_map[cs.ind[off + 1] as usize]);
            out.concat_stitch_ind.push(src_map[cs.ind[off + 2] as usize]);
            out.concat_stitch_ind.push(tgt_map[cs.ind[off + 3] as usize]);
            out.concat_stitch_ind.push(tgt_map[cs.ind[off + 4] as usize]);
            out.concat_stitch_ind.push(tgt_map[cs.ind[off + 5] as usize]);
            out.concat_stitch_w
                .extend_from_slice(&cs.weights[off..off + 6]);
            // Per-pair stiffness for this cross-stitch row.
            out.concat_stitch_stiffness.push(cs.stitch_stiffness as f32);
        }
    }

    out.per_object = stats_by_name;
    Ok(out)
}

/// Output of the static-scene assembly.
#[derive(Debug, Default)]
pub struct AssembleStaticResult {
    pub static_vert: Vec<f64>,         // (N, 3) flat
    pub static_vert_dmap: Vec<u32>,    // (N,)
    pub static_tri: Vec<u32>,          // (M, 3) flat
    pub static_color: Vec<f64>,        // (N, 3)
    /// Per-object (`name`, vert_offset, n_face) for the caller's param
    /// extension and transform-animation offset bookkeeping.
    pub per_object: Vec<(String, usize, usize)>,
}

/// Walk every static object, transform-and-concatenate its vertices
/// (already pre-transformed), copy its color out, and append its
/// face indices with the running offset. `displacement_index` carries
/// the dmap ordering so static_vert_dmap aligns with concat_displacement.
pub fn assemble_static_scene(
    objects: &[AssembleStaticObject<'_>],
    displacement_index: &[(String, [f64; 3])],
) -> Result<AssembleStaticResult, SceneAssemblyError> {
    let mut out = AssembleStaticResult::default();
    // dmap lookup.
    let mut dmap: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::with_capacity(displacement_index.len());
    for (i, (name, _)) in displacement_index.iter().enumerate() {
        dmap.insert(name.as_str(), i);
    }
    for so in objects {
        let faces = match so.faces {
            Some(f) => f,
            None => continue,
        };
        if so.vertex_world.len() % 3 != 0 {
            return Err(SceneAssemblyError::StaticVertexNotMultipleOf3 {
                name: so.name.to_string(),
            });
        }
        let n_v = so.vertex_world.len() / 3;
        if n_v == 0 {
            continue;
        }
        let dmap_idx = *dmap
            .get(so.name)
            .ok_or_else(|| SceneAssemblyError::MissingStaticDisplacement {
                name: so.name.to_string(),
            })? as u32;
        let offset = (out.static_vert.len() / 3) as u32;
        out.static_vert.extend_from_slice(so.vertex_world);
        for _ in 0..n_v {
            out.static_color.push(so.color[0]);
            out.static_color.push(so.color[1]);
            out.static_color.push(so.color[2]);
            out.static_vert_dmap.push(dmap_idx);
        }
        for face in faces {
            out.static_tri.push(face[0] + offset);
            out.static_tri.push(face[1] + offset);
            out.static_tri.push(face[2] + offset);
        }
        out.per_object
            .push((so.name.to_string(), offset as usize, so.n_faces));
    }
    Ok(out)
}

/// Single-face shell shrink/strain-limit conflict predicate. Shrink/extend
/// and a strain-limit cannot be combined on the same face: each rewrites the
/// rest shape independently, so the strain bound becomes ill-defined when both
/// are active. This is the one place the rule lives, shared by the batch check
/// below and the solver's per-face assertion.
pub fn is_shell_shrink_strain_limit_conflict(x: f64, y: f64, s: f64) -> bool {
    (x != 1.0 || y != 1.0) && s > 0.0
}

/// Shell shrink/strain-limit conflict check. Returns the offending
/// face index when the conflict is present so callers can emit the
/// matching ValueError.
pub fn check_shell_shrink_strain_limit_conflict(
    shrink_x: &[f64],
    shrink_y: &[f64],
    strain_limit: &[f64],
) -> Option<(usize, f64, f64, f64)> {
    if shrink_x.is_empty() || shrink_y.is_empty() || strain_limit.is_empty() {
        return None;
    }
    let n = shrink_x.len().min(shrink_y.len()).min(strain_limit.len());
    for i in 0..n {
        let x = shrink_x[i];
        let y = shrink_y[i];
        let s = strain_limit[i];
        if is_shell_shrink_strain_limit_conflict(x, y, s) {
            return Some((i, x, y, s));
        }
    }
    None
}

