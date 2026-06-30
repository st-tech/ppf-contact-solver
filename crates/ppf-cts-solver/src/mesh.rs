// File: mesh.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use na::{Matrix2xX, Matrix3xX, Matrix4xX};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Serialize, Deserialize)]
pub struct MeshInfo {
    pub face: Matrix3xX<usize>,
    pub tet: Matrix4xX<usize>,
    pub edge: Matrix2xX<usize>,
    pub hinge: Matrix4xX<usize>,
    pub vertex_count: usize,
    pub surface_vert_count: usize,
    pub shell_face_count: usize,
    pub rod_count: usize,
}

#[derive(Serialize, Deserialize)]
pub struct VertexNeighbor {
    pub face: Vec<Vec<usize>>,
    pub hinge: Vec<Vec<usize>>,
    pub edge: Vec<Vec<usize>>,
    pub rod: Vec<Vec<usize>>,
    pub tet: Vec<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
pub struct HingeNeighbor {
    pub face: Vec<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
pub struct EdgeNeighbor {
    pub face: Vec<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
pub struct Neighbor {
    pub vertex: VertexNeighbor,
    pub hinge: HingeNeighbor,
    pub edge: EdgeNeighbor,
}

#[derive(Serialize, Deserialize)]
pub struct Mesh {
    pub mesh: MeshInfo,
    pub neighbor: Neighbor,
}

impl Mesh {
    /// Builds the topology and connectivity tables from element index
    /// buffers.
    ///
    /// Precondition: every vertex index in `rod`, `face`, and `tet` must lie
    /// in the contiguous range `[0, vertex_count)`, i.e. the vertices passed
    /// to the solver must be densely packed with no gaps. `vertex_count` is
    /// the count of distinct referenced indices and is assumed to equal the
    /// vertex-buffer length the backend slices against; that equality only
    /// holds when indices are contiguous. compute_vertex_neighbors indexes a
    /// `vertex_count`-sized array with the raw global index, so a sparse
    /// index set would otherwise panic out of bounds.
    pub fn new(
        rod: Matrix2xX<usize>,
        face: Matrix3xX<usize>,
        tet: Matrix4xX<usize>,
        shell_face_count: usize,
        n_vertices: usize,
    ) -> Mesh {
        let rod_count = rod.ncols();
        // Element indices only count vertices REFERENCED by topology. A
        // faceless particle cloud (SAND) references none, so fall back to the
        // explicit vertex-buffer length `n_vertices` so free vertices size the
        // neighbor tables consistently with the vertex buffer. For meshes whose
        // vertices are all element-referenced the two are equal, so behavior is
        // unchanged.
        let vertex_count = get_vertex_count(&rod, &face, &tet).max(n_vertices);
        // Contact-participating vertices vs interior vertices. The contact
        // engine dispatches over `surface_vert_count` (every
        // `DISPATCH_START(surface_vert_count)` site and the vertex BVH). The
        // index map (build_index_map) orders the global vertex buffer as
        // `[ ..contact verts.. | tet-interior-only verts ]`. The ONLY
        // non-contact vertices are those referenced solely by tets (a solid's
        // interior Steiner points); element-surface verts, rod verts, and SAND
        // grains are all contact verts. So the boundary is the vertex count
        // minus the tet-interior-only count, which keeps grains and every
        // surface vertex in the contact range for any mix of object types
        // (solid / shell / rod / PDRD / sand) with no per-type special-casing.
        let surface_vert_count = vertex_count - count_interior_verts(&rod, &face, &tet);
        // Every element index must lie within the vertex range so the
        // `vertex_count`-sized neighbor tables are not indexed out of bounds.
        // Free vertices may extend the range beyond the largest referenced
        // index (a particle cloud, or a cloth with detached verts), so this is
        // a bound, not the old strict `== vertex_count - 1` equality.
        debug_assert!(
            {
                let max_index = rod
                    .iter()
                    .chain(face.iter())
                    .chain(tet.iter())
                    .copied()
                    .max();
                match max_index {
                    Some(m) => m < vertex_count,
                    None => true,
                }
            },
            "element vertex index out of range (vertex_count {vertex_count}); \
             vertices passed to the solver must be densely packed with no gaps"
        );
        for &e in rod.iter() {
            assert!(
                e < surface_vert_count,
                "rod vertex index {e} out of range (surface_vert_count \
                 {surface_vert_count}); rod references a vertex outside the \
                 rod/face surface set"
            );
        }
        for &e in face.iter() {
            assert!(
                e < surface_vert_count,
                "face vertex index {e} out of range (surface_vert_count \
                 {surface_vert_count}); face references a vertex outside the \
                 rod/face surface set"
            );
        }
        for &e in tet.iter() {
            // Interior tet vertices legitimately occupy
            // surface_vert_count..vertex_count, so they are bounded by
            // vertex_count rather than surface_vert_count. This fires before
            // compute_vertex_neighbors indexes a vertex_count-sized array with
            // the raw tet index, turning an opaque out-of-bounds panic into an
            // actionable message.
            assert!(
                e < vertex_count,
                "tet vertex index {e} out of range (vertex_count {vertex_count})"
            );
        }
        let (hinge, hinge_face) = compute_hinge(&face);
        let (edge, edge_face) = compute_edge(&rod, &face);
        let mesh = MeshInfo {
            face,
            tet,
            edge,
            hinge,
            vertex_count,
            surface_vert_count,
            shell_face_count,
            rod_count,
        };
        let vertex_neighbor = VertexNeighbor {
            face: compute_vertex_neighbors::<na::U3>(vertex_count, &mesh.face),
            hinge: compute_vertex_neighbors::<na::U4>(vertex_count, &mesh.hinge),
            edge: compute_vertex_neighbors::<na::U2>(vertex_count, &mesh.edge),
            rod: compute_vertex_neighbors::<na::U2>(vertex_count, &rod),
            tet: compute_vertex_neighbors::<na::U4>(vertex_count, &mesh.tet),
        };
        let edge_neighbor = EdgeNeighbor { face: edge_face };
        let hinge_neighbor = HingeNeighbor { face: hinge_face };
        let neighbor = Neighbor {
            vertex: vertex_neighbor,
            edge: edge_neighbor,
            hinge: hinge_neighbor,
        };
        Mesh { mesh, neighbor }
    }
}

fn compute_vertex_neighbors<D: na::Dim>(
    vertex_count: usize,
    element: &na::Matrix<usize, D, na::Dyn, na::VecStorage<usize, D, na::Dyn>>,
) -> Vec<Vec<usize>>
where
    na::VecStorage<usize, D, na::Dyn>: na::RawStorage<usize, D, na::Dyn>,
{
    let mut result = vec![Vec::new(); vertex_count];
    for (i, face) in element.column_iter().enumerate() {
        for &j in face.iter() {
            result[j].push(i);
        }
    }
    result
}

fn compute_edge(
    rod: &Matrix2xX<usize>,
    face: &Matrix3xX<usize>,
) -> (Matrix2xX<usize>, Vec<Vec<usize>>) {
    // Tuple key (idx_min, idx_max): face indices are global into a
    // vertex array shared with rod/tet groups, so they can sit
    // arbitrarily far past any packed-integer radix derived from
    // face count alone.
    let mut hash = HashMap::<(usize, usize), (usize, usize, usize, Option<usize>)>::new();
    for (i, column) in face.column_iter().enumerate() {
        for j in 0..3 {
            let e = (column[j], column[(j + 1) % 3], i, None);
            let key = (e.0.min(e.1), e.0.max(e.1));
            if let Some(b) = hash.get_mut(&key) {
                // Record the second face exactly once. A non-manifold edge
                // (>2 incident faces) keeps a deterministic first-two-face
                // pair instead of silently overwriting the second slot with
                // an arbitrary later face. The edge stays in the hash so it
                // survives into mesh.edge and the collision BVH, unlike
                // compute_hinge which can drop non-manifold edges.
                if b.3.is_none() {
                    b.3 = Some(i);
                }
            } else {
                hash.insert(key, e);
            }
        }
    }
    let mut edge = Vec::new();
    edge.extend(rod.iter());
    edge.extend(hash.values().flat_map(|e| [e.0, e.1]));
    let mut edge_face = Vec::new();
    edge_face.extend(vec![Vec::new(); rod.shape().1]);
    edge_face.extend(hash.values().map(|e| {
        if let Some(i) = e.3 {
            vec![e.2, i]
        } else {
            vec![e.2]
        }
    }));
    (Matrix2xX::from_vec(edge), edge_face)
}

fn compute_hinge(face: &Matrix3xX<usize>) -> (Matrix4xX<usize>, Vec<Vec<usize>>) {
    // Tuple keys for both the canonical-face dedup and the
    // shared-edge hash: face indices are global into a vertex array
    // shared with rod/tet groups, so packing them into one integer
    // would need a radix the face count cannot bound.
    //
    // Two coincident triangles (an identical vertex set) would register
    // the same edge twice with the same opposite vertex and build a
    // degenerate hinge [a, b, c, c], so they are rejected here with an
    // actionable message instead of being silently dropped: a doubled
    // face may be unintended geometry the user wants to know about. The
    // frontend encoder catches this first and names the offending
    // object; this assert is the backstop for the collision mesh and
    // any direct caller fed duplicated geometry (e.g. doubled airbag
    // faces welded with Merge by Distance).
    let mut keys = HashSet::<[usize; 3]>::new();
    for f in face.column_iter() {
        let mut e = [f[0], f[1], f[2]];
        e.sort();
        assert!(
            keys.insert(e),
            "duplicate shell face {e:?}: two triangles share the same three \
             vertices. Remove the doubled geometry (Mesh > Merge by Distance, \
             or delete the duplicate faces) before running the solver."
        );
    }
    let mut hash = HashMap::<
        (usize, usize),
        (usize, usize, usize, Option<usize>, usize, Option<usize>),
    >::new();
    let mut excludes = Vec::new();
    for (i, f) in face.column_iter().enumerate() {
        for j in 0..3 {
            let e = (f[j], f[(j + 1) % 3], f[(j + 2) % 3]);
            let key = (e.0.min(e.1), e.0.max(e.1));
            if let Some(b) = hash.get_mut(&key) {
                if b.3.is_some() {
                    excludes.push(key);
                } else {
                    b.3 = Some(e.2);
                    b.5 = Some(i);
                }
            } else {
                hash.insert(key, (e.0, e.1, e.2, None, i, None));
            }
        }
    }
    for key in excludes {
        hash.remove(&key);
    }
    let hinge = hash
        .iter()
        .filter_map(|(_, &(i0, i1, i2, op, _, _))| op.map(|i3| [i0, i1, i2, i3]))
        .flatten()
        .collect::<Vec<_>>();
    let face_neighbors = hash
        .iter()
        .filter_map(|(_, &(_, _, _, _, i, op))| op.map(|j| vec![i, j]))
        .collect::<Vec<_>>();
    (
        Matrix4xX::from_column_slice(hinge.as_slice()),
        face_neighbors,
    )
}

fn get_vertex_count(
    rod: &Matrix2xX<usize>,
    face: &Matrix3xX<usize>,
    tet: &Matrix4xX<usize>,
) -> usize {
    let mut unique_set = HashSet::new();
    for &i in rod.iter() {
        unique_set.insert(i);
    }
    for &i in face.iter() {
        unique_set.insert(i);
    }
    for &i in tet.iter() {
        unique_set.insert(i);
    }
    unique_set.len()
}

/// Count the tet-interior-only vertices: those referenced by a tet but by no
/// rod or face. These are the ONLY non-contact vertices (a solid's interior
/// Steiner points). Everything else (element surface, rods, SAND grains) is a
/// contact vertex, so `surface_vert_count = vertex_count - this`. The index map
/// orders these interior verts last, so the contact range stays `[0,
/// surface_vert_count)` for any mix of object types.
fn count_interior_verts(
    rod: &Matrix2xX<usize>,
    face: &Matrix3xX<usize>,
    tet: &Matrix4xX<usize>,
) -> usize {
    if tet.ncols() == 0 {
        return 0;
    }
    let mut surface_set = HashSet::new();
    for &i in rod.iter() {
        surface_set.insert(i);
    }
    for &i in face.iter() {
        surface_set.insert(i);
    }
    let mut interior_set = HashSet::new();
    for &i in tet.iter() {
        if !surface_set.contains(&i) {
            interior_set.insert(i);
        }
    }
    interior_set.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_hinge_valid_quad_has_one_hinge() {
        // Quad split into [0,1,2] + [0,2,3]: a single hinge across the
        // shared diagonal edge (0,2).
        let face = Matrix3xX::<usize>::from_column_slice(&[0, 1, 2, 0, 2, 3]);
        let (hinge, neighbors) = compute_hinge(&face);
        assert_eq!(hinge.ncols(), 1);
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    #[should_panic(expected = "duplicate shell face")]
    fn compute_hinge_rejects_duplicate_faces() {
        // A coincident copy of the first triangle (same vertex set)
        // must be rejected with an actionable message, not silently
        // dropped. The frontend encoder is expected to catch this first.
        let face = Matrix3xX::<usize>::from_column_slice(&[0, 1, 2, 0, 2, 3, 0, 1, 2]);
        let _ = compute_hinge(&face);
    }
}
