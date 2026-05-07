// File: crates/ppf-cts-core/src/kernels/scene_build/index_map.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `build_index_map`: the per-object `local -> global` index assignment
// (rod-first, shell-after, finalize) with merge-pair alias resolution.
// Mirrors `add_entry` + `resolve_alias` + finalize loops in `Scene.build`.

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
    #[error("merge pair [{i}] missing required source/target name")]
    MergePairMissingName { i: usize },
    #[error("merge pair references unknown object: source={source_name:?} target={target_name:?}")]
    MergePairUnknownObject { source_name: String, target_name: String },
    #[error("vertex map for {name:?} has unassigned slot {i} after finalization")]
    UnassignedSlot { name: String, i: usize },
}

/// Resolve `(name, vi)` through the alias map, following chains.
/// Bounded by `alias.len()` to be safe against malformed cycles.
fn resolve(
    name: &str,
    vi: i64,
    alias: &BTreeMap<(String, i64), (String, i64)>,
) -> (String, i64) {
    let mut cur_name = name.to_string();
    let mut cur_vi = vi;
    let cap = alias.len();
    for _ in 0..=cap {
        match alias.get(&(cur_name.clone(), cur_vi)) {
            Some((nn, nv)) => {
                cur_name = nn.clone();
                cur_vi = *nv;
            }
            None => break,
        }
    }
    (cur_name, cur_vi)
}

/// Walk a topology entry (rows of vertex indices) and assign
/// global slots to anything not yet assigned.
#[inline]
fn add_entry<I: Iterator<Item = u32>>(
    obj_name: &str,
    iter: I,
    map_by_name: &mut BTreeMap<String, Vec<i64>>,
    alias: &BTreeMap<(String, i64), (String, i64)>,
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
        let (target_name, target_vi) = resolve(obj_name, vi, alias);
        if target_name != obj_name || target_vi != vi {
            // Aliased target. Assign to whichever slot the target has,
            // or claim a new global slot if neither is assigned.
            let target_already = map_by_name
                .get(&target_name)
                .and_then(|m| m.get(target_vi as usize).copied())
                .unwrap_or(-1);
            if target_already >= 0 {
                if let Some(m) = map_by_name.get_mut(obj_name) {
                    m[vi as usize] = target_already;
                }
            } else {
                let new_idx = *concat_count as i64;
                *concat_count += 1;
                if let Some(m) = map_by_name.get_mut(&target_name) {
                    m[target_vi as usize] = new_idx;
                }
                if let Some(m) = map_by_name.get_mut(obj_name) {
                    m[vi as usize] = new_idx;
                }
            }
        } else {
            let new_idx = *concat_count as i64;
            *concat_count += 1;
            if let Some(m) = map_by_name.get_mut(obj_name) {
                m[vi as usize] = new_idx;
            }
        }
    }
}

/// Run the rod-first, shell-after, finalize index-map construction
/// from `Scene.build`. `merge_pairs` is a slice of
/// `(source_name, target_name, [(source_vi, target_vi)...])`.
///
/// The Python source uses `(target_name, target_vi) -> (source_name,
/// source_vi)` direction so we mirror that exactly.
pub fn build_index_map(
    objects: &[IndexMapObject<'_>],
    merge_pairs: &[(String, String, Vec<(u32, u32)>)],
) -> Result<IndexMapResult, SceneBuildError> {
    // 1. Build alias map.
    let mut alias: BTreeMap<(String, i64), (String, i64)> = BTreeMap::new();
    let names: std::collections::HashSet<&str> = objects.iter().map(|o| o.name).collect();
    for (i, (src, tgt, pairs)) in merge_pairs.iter().enumerate() {
        if src.is_empty() || tgt.is_empty() {
            return Err(SceneBuildError::MergePairMissingName { i });
        }
        if !names.contains(src.as_str()) || !names.contains(tgt.as_str()) {
            return Err(SceneBuildError::MergePairUnknownObject {
                source_name: src.clone(),
                target_name: tgt.clone(),
            });
        }
        for &(src_vi, tgt_vi) in pairs {
            alias.insert((tgt.clone(), tgt_vi as i64), (src.clone(), src_vi as i64));
        }
    }

    // 2. Initialize per-object map (-1 sentinel).
    let mut map_by_name: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for obj in objects {
        map_by_name.insert(obj.name.to_string(), vec![-1i64; obj.n_verts]);
    }

    let mut concat_count: usize = 0;

    // 3. Rod indexing (objects with edges, no tets).
    for obj in objects {
        if obj.tets.is_some() {
            continue;
        }
        if let Some(edges) = obj.edges {
            add_entry(
                obj.name,
                edges.iter().flat_map(|e| e.iter().copied()),
                &mut map_by_name,
                &alias,
                &mut concat_count,
            );
        }
    }
    let rod_vert_range = (0, concat_count);

    // 4. Shell indexing for non-tet objects (mirrors Python's first
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
                &alias,
                &mut concat_count,
            );
        }
    }
    let shell_vert_range = (rod_vert_range.1, concat_count);

    // 5. Surface pass for everyone with faces (incl. tet surfaces).
    for obj in objects {
        if let Some(faces) = obj.faces {
            add_entry(
                obj.name,
                faces.iter().flat_map(|f| f.iter().copied()),
                &mut map_by_name,
                &alias,
                &mut concat_count,
            );
        }
    }

    // 6. Finalize: any remaining -1 vertices walk through alias too.
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
            let vi = i as i64;
            let (target_name, target_vi) = resolve(obj.name, vi, &alias);
            if target_name != obj.name || target_vi != vi {
                let target_already = map_by_name
                    .get(&target_name)
                    .and_then(|m| m.get(target_vi as usize).copied())
                    .unwrap_or(-1);
                if target_already >= 0 {
                    if let Some(m) = map_by_name.get_mut(obj.name) {
                        m[i] = target_already;
                    }
                } else {
                    let new_idx = concat_count as i64;
                    concat_count += 1;
                    if let Some(m) = map_by_name.get_mut(&target_name) {
                        m[target_vi as usize] = new_idx;
                    }
                    if let Some(m) = map_by_name.get_mut(obj.name) {
                        m[i] = new_idx;
                    }
                }
            } else {
                let new_idx = concat_count as i64;
                concat_count += 1;
                if let Some(m) = map_by_name.get_mut(obj.name) {
                    m[i] = new_idx;
                }
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
