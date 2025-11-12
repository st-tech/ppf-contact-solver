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
    pub fn new(
        rod: Matrix2xX<usize>,
        face: Matrix3xX<usize>,
        tet: Matrix4xX<usize>,
        shell_face_count: usize,
    ) -> Mesh {
        let rod_count = rod.ncols();
        let vertex_count = get_vertex_count(&rod, &face, &tet);
        let surface_vert_count = get_surface_vert_count(&rod, &face);
        for &e in rod.iter() {
            assert!(e < surface_vert_count);
        }
        for &e in face.iter() {
            assert!(e < surface_vert_count);
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
    let mut hash = HashMap::<usize, (usize, usize, usize, Option<usize>)>::new();
    let n_face = face.shape().1;
    for (i, column) in face.column_iter().enumerate() {
        for j in 0..3 {
            let e = (column[j], column[(j + 1) % 3], i, None);
            let key = e.0.min(e.1) + 3 * n_face * e.0.max(e.1);
            if let Some(b) = hash.get_mut(&key) {
                b.3 = Some(i);
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

fn has_duplicates(numbers: &[usize]) -> bool {
    let mut seen = std::collections::HashSet::new();
    for number in numbers {
        if !seen.insert(number) {
            return true;
        }
    }
    false
}

fn compute_hinge(face: &Matrix3xX<usize>) -> (Matrix4xX<usize>, Vec<Vec<usize>>) {
    let mut keys = Vec::new();
    for f in face.column_iter() {
        let mut e = [f[0], f[1], f[2]];
        e.sort();
        let key = e[0] + 3 * face.shape().1 * e[1] + 9 * face.shape().1 * face.shape().1 * e[2];
        keys.push(key);
    }
    assert!(!has_duplicates(&keys));
    let mut hash =
        HashMap::<usize, (usize, usize, usize, Option<usize>, usize, Option<usize>)>::new();
    let mut excludes = Vec::new();
    for (i, f) in face.column_iter().enumerate() {
        for j in 0..3 {
            let e = (f[j], f[(j + 1) % 3], f[(j + 2) % 3]);
            let (idx0, idx1) = (e.0.min(e.1), e.0.max(e.1));
            let w = 3 * face.shape().1;
            assert!(idx0 < w);
            assert!(idx1 < w);
            let key = idx0 + w * idx1;
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

fn get_surface_vert_count(rod: &Matrix2xX<usize>, face: &Matrix3xX<usize>) -> usize {
    let mut unique_set = HashSet::new();
    for &i in rod.iter() {
        unique_set.insert(i);
    }
    for &i in face.iter() {
        unique_set.insert(i);
    }
    unique_set.len()
}
