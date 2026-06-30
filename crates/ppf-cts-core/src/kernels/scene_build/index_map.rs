// File: crates/ppf-cts-core/src/kernels/scene_build/index_map.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `build_index_map`: the per-object `local -> global` index assignment
// (rod-first, shell-after, finalize). Mirrors `add_entry` + finalize
// loops in `Scene.build`.

use std::collections::BTreeMap;

/// One dynamic object as seen by the index-map builder. The index-map
/// stage only needs topology plus the vertex count; the vertex /
/// transform payload is consumed by later stages of the assembly
/// kernel.
#[derive(Debug, Clone)]
pub struct IndexMapObject<'a> {
    pub name: &'a str,
    pub n_verts: usize,
    pub edges: Option<&'a [[u32; 2]]>,
    pub faces: Option<&'a [[u32; 3]]>,
    pub tets: Option<&'a [[u32; 4]]>,
}

/// Result of `build_index_map`. `concat_count` is the global vertex
/// count after the build and finalize passes.
#[derive(Debug, Default)]
pub struct IndexMapResult {
    pub map_by_name: BTreeMap<String, Vec<i64>>,
    pub rod_vert_range: (usize, usize),
    pub shell_vert_range: (usize, usize),
    pub concat_count: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum SceneBuildError {
    #[error("vertex map for {name:?} has unassigned slot {i} after finalization")]
    UnassignedSlot { name: String, i: usize },
}

/// Walk a topology entry (rows of vertex indices) and assign global
/// slots to anything not yet assigned.
#[inline]
fn add_entry<I: Iterator<Item = u32>>(
    obj_name: &str,
    iter: I,
    map_by_name: &mut BTreeMap<String, Vec<i64>>,
    concat_count: &mut usize,
) {
    for vi_u in iter {
        let vi = vi_u as i64;
        // Skip if already assigned.
        let already = map_by_name
            .get(obj_name)
            .and_then(|m| m.get(vi as usize).copied())
            .unwrap_or(-1);
        if already >= 0 {
            continue;
        }
        let new_idx = *concat_count as i64;
        *concat_count += 1;
        if let Some(m) = map_by_name.get_mut(obj_name) {
            m[vi as usize] = new_idx;
        }
    }
}

/// Run the rod-first, shell-after, finalize index-map construction
/// from `Scene.build`.
pub fn build_index_map(
    objects: &[IndexMapObject<'_>],
) -> Result<IndexMapResult, SceneBuildError> {
    // 1. Initialize per-object map (-1 sentinel).
    let mut map_by_name: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for obj in objects {
        map_by_name.insert(obj.name.to_string(), vec![-1i64; obj.n_verts]);
    }

    let mut concat_count: usize = 0;

    // 2. Rod indexing (objects with edges, no tets).
    for obj in objects {
        if obj.tets.is_some() {
            continue;
        }
        if let Some(edges) = obj.edges {
            add_entry(
                obj.name,
                edges.iter().flat_map(|e| e.iter().copied()),
                &mut map_by_name,
                &mut concat_count,
            );
        }
    }
    let rod_vert_range = (0, concat_count);

    // 3. Shell indexing for non-tet objects (mirrors Python's first
    //    pass over `obj.get("F")` for non-tets).
    for obj in objects {
        if obj.tets.is_some() {
            continue;
        }
        if let Some(faces) = obj.faces {
            add_entry(
                obj.name,
                faces.iter().flat_map(|f| f.iter().copied()),
                &mut map_by_name,
                &mut concat_count,
            );
        }
    }
    let shell_vert_range = (rod_vert_range.1, concat_count);

    // 4. Surface pass for everyone with faces (incl. tet surfaces).
    for obj in objects {
        if let Some(faces) = obj.faces {
            add_entry(
                obj.name,
                faces.iter().flat_map(|f| f.iter().copied()),
                &mut map_by_name,
                &mut concat_count,
            );
        }
    }

    // 4.5 Loose pass: a vertex of a NON-tet object that is referenced by no
    //     edge or face (a SAND grain, or a detached vertex) is a
    //     contact-participating vertex, not an interior one. Assign these now,
    //     before the interior finalize, so the global order is
    //     `[ ..element surface.. | loose/grains | tet-interior ]`. The only
    //     non-contact vertices are tet-interior-only verts (handled by the
    //     finalize below), which keeps the contact-vertex range contiguous from
    //     0 for every mix of object types (solid/shell/rod/PDRD/sand).
    for obj in objects {
        if obj.tets.is_some() {
            continue;
        }
        add_entry(
            obj.name,
            0..obj.n_verts as u32,
            &mut map_by_name,
            &mut concat_count,
        );
    }

    // 5. Finalize: assign any remaining -1 vertices a fresh global slot. After
    //    the loose pass these are exactly the tet-interior-only verts of tet
    //    objects, so they land at the tail (>= surface_vert_count).
    for obj in objects {
        let n = obj.n_verts;
        for i in 0..n {
            let cur = map_by_name
                .get(obj.name)
                .and_then(|m| m.get(i).copied())
                .unwrap_or(-1);
            if cur >= 0 {
                continue;
            }
            let new_idx = concat_count as i64;
            concat_count += 1;
            if let Some(m) = map_by_name.get_mut(obj.name) {
                m[i] = new_idx;
            }
        }
    }

    // Sanity: every entry should be non-negative now.
    for (name, m) in &map_by_name {
        for (i, &v) in m.iter().enumerate() {
            if v < 0 {
                return Err(SceneBuildError::UnassignedSlot {
                    name: name.clone(),
                    i,
                });
            }
        }
    }

    Ok(IndexMapResult {
        map_by_name,
        rod_vert_range,
        shell_vert_range,
        concat_count,
    })
}
