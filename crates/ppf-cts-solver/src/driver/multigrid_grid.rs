// File: crates/ppf-cts-solver/src/driver/multigrid_grid.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The geometric multigrid's hierarchy, built once on the host.
//!
//! The finest level is the scene's vertices. Every level below it is a set of
//! structured grids, one per connected piece of mesh, laid over that piece's
//! authored positions: a vertex is interpolated linearly from the nodes of the
//! grid cell holding it, and each grid is then halved, a node of one level
//! interpolated from the nodes of the next. An interpolation acts on the three
//! unknowns of a vertex alike and reproduces every field that is linear over
//! those positions, so a coarse level carries a piece's translations, its
//! rotations and its smooth deformations rather than its translations alone,
//! which is what a piecewise-constant coarse space carries and why one is weak
//! on cloth that turns and bends.
//!
//! # A grid per piece, in the piece's own frame
//!
//! A scene is any mix of shells, solids and rods of any topology, so there is no
//! one parameter domain to lay a grid over. Each connected piece of mesh gets a
//! grid in the frame of its own principal axes, and an axis along which the piece
//! is thinner than half a cell carries one layer of nodes and no interpolation.
//! A flat sheet is therefore given a two-dimensional grid in its own plane
//! wherever it stands, a straight rod a line of nodes, and a solid, a tube, a
//! closed surface or a Mobius strip a three-dimensional one of which only the
//! nodes with a vertex in an adjoining cell are kept. Two pieces never share a
//! node, so nothing at a coarse level ties one object to another.
//!
//! # Every piece ends in one node
//!
//! A piece's grid is halved until it is one node, which then carries the piece's
//! translation, and that node belongs to no level below. The level counts of
//! the pieces differ, and a level holds the nodes of every piece that reaches
//! it. A vertex that belongs to no element has no grid at all.
//!
//! # The first spacing follows the cost of the level it makes
//!
//! A grid's spacing starts at twice the piece's mean edge length. A surface
//! curved through a three-dimensional grid keeps more nodes at that spacing
//! than it has cells to justify, each coupled to every node around it, so its
//! first coarse operator can hold more blocks than the mesh's own. The spacing
//! of a piece is therefore doubled again, at any level, until that piece's
//! operator holds at most [`BLOCK_RATIO`] of the blocks of the one above it,
//! which bounds what a cycle costs whatever the topology.

use rayon::prelude::*;

/// Nodes a row is interpolated from, at most: the corners of one grid cell.
pub const WIDTH: usize = 8;

/// A level's operator may hold at most this fraction of the blocks of the
/// operator it coarsens, piece by piece.
///
/// Two thirds, so the operators of all levels together hold at most three times
/// the finest one's blocks. A flat sheet at twice its edge length is well
/// inside it and keeps that spacing; a curved surface is not and takes four
/// times.
pub const BLOCK_RATIO: f64 = 2.0 / 3.0;

/// An axis along which a piece spans less than this fraction of the grid's
/// spacing carries one layer of nodes.
///
/// One half: two layers a full cell apart would then both lie farther from the
/// piece than the piece is thick.
pub const THIN: f64 = 0.5;

/// A weight below this is dropped and the rest of its row rescaled.
const DROP: f64 = 1.0e-6;

/// The mesh the hierarchy is laid over.
pub struct Mesh<'a> {
    /// The positions the grids are laid over, one per vertex.
    pub positions: &'a [[f64; 3]],
    pub faces: &'a [[u32; 3]],
    pub edges: &'a [[u32; 2]],
    pub tets: &'a [[u32; 4]],
}

/// A block pattern, both triangles and the diagonal, columns ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    pub offset: Vec<u32>,
    pub column: Vec<u32>,
}

impl Pattern {
    pub fn rows(&self) -> usize {
        self.offset.len().saturating_sub(1)
    }

    pub fn row(&self, row: usize) -> &[u32] {
        &self.column[self.offset[row] as usize..self.offset[row + 1] as usize]
    }

    pub fn blocks(&self) -> usize {
        self.column.len()
    }

    /// The pattern of a matrix stored as its upper triangle, `upper[i]` the
    /// columns `j >= i` of row `i`.
    #[cfg(test)]
    pub fn from_upper(upper: &[Vec<u32>]) -> Pattern {
        let rows = upper.len();
        let mut full: Vec<Vec<u32>> = vec![Vec::new(); rows];
        for (i, columns) in upper.iter().enumerate() {
            full[i].push(i as u32);
            for &j in columns {
                full[i].push(j);
                full[j as usize].push(i as u32);
            }
        }
        Pattern::from_rows(full)
    }

    /// The same from the flat form the fixed matrix is stored in: row `i`'s
    /// columns `j >= i` are `index[offset[i] .. offset[i + 1]]`.
    pub fn from_upper_csr(offset: &[u32], index: &[u32]) -> Pattern {
        let rows = offset.len().saturating_sub(1);
        let mut full: Vec<Vec<u32>> = vec![Vec::new(); rows];
        for i in 0..rows {
            full[i].push(i as u32);
            for &j in &index[offset[i] as usize..offset[i + 1] as usize] {
                full[i].push(j);
                full[j as usize].push(i as u32);
            }
        }
        Pattern::from_rows(full)
    }

    fn from_rows(mut rows: Vec<Vec<u32>>) -> Pattern {
        rows.par_iter_mut().for_each(|row| {
            row.sort_unstable();
            row.dedup();
        });
        let mut offset = Vec::with_capacity(rows.len() + 1);
        offset.push(0u32);
        let mut column = Vec::with_capacity(rows.iter().map(Vec::len).sum());
        for row in &rows {
            column.extend_from_slice(row);
            offset.push(column.len() as u32);
        }
        Pattern { offset, column }
    }
}

/// The interpolation between a level and the finer one above it.
#[derive(Clone, Debug, Default)]
pub struct Transfer {
    /// Rows of the finer level.
    pub fine: u32,
    /// Nodes of this level.
    pub nodes: u32,
    /// Per finer row, [`WIDTH`] nodes of this level, unused slots naming node
    /// zero at weight zero.
    pub node: Vec<u32>,
    pub weight: Vec<f32>,
    /// The same weights listed per node: node `n` gathers the finer rows
    /// `gather_row[k]` at `gather_weight[k]` for `k` in
    /// `gather_offset[n] .. gather_offset[n + 1]`.
    pub gather_offset: Vec<u32>,
    pub gather_row: Vec<u32>,
    pub gather_weight: Vec<f32>,
}

/// One coarse level.
#[derive(Clone, Debug, Default)]
pub struct Level {
    pub transfer: Transfer,
    /// The finer operator times the interpolation, `A P`: one row per finer
    /// row, its columns this level's nodes. The product is formed through it.
    pub reach: Pattern,
    /// The level's operator, the Galerkin product `P^T A P` of the finer one.
    pub pattern: Pattern,
}

/// The hierarchy below the finest level.
#[derive(Clone, Debug, Default)]
pub struct Hierarchy {
    pub levels: Vec<Level>,
    /// Connected pieces of mesh that were given a grid.
    pub pieces: usize,
    /// Vertices that belong to no element and so to no grid.
    pub loose: usize,
}

struct Piece {
    /// Rows of the current finest level this piece still owns.
    rows: Vec<u32>,
    /// Their coordinates in the piece's frame, measured from its lower corner.
    points: Vec<[f64; 3]>,
    extent: [f64; 3],
    /// The spacing of the next grid.
    spacing: f64,
}

/// One piece's interpolation onto a grid: per row its nodes and weights, and
/// the kept nodes' own coordinates.
struct NodeGrid {
    /// Per row, up to [`WIDTH`] `(node, weight)`, the node an index into `nodes`.
    stencil: Vec<Vec<(u32, f64)>>,
    nodes: Vec<[f64; 3]>,
}

fn find(parent: &mut [u32], mut v: u32) -> u32 {
    while parent[v as usize] != v {
        let up = parent[parent[v as usize] as usize];
        parent[v as usize] = up;
        v = up;
    }
    v
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        // the smaller index roots, so a piece is named by its first vertex
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

/// The principal axes of a point set, widest first, as rows.
fn principal_axes(points: &[[f64; 3]], mean: [f64; 3]) -> [[f64; 3]; 3] {
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    for p in points {
        let d = nalgebra::Vector3::new(p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]);
        cov += d * d.transpose();
    }
    let eigen = nalgebra::SymmetricEigen::new(cov);
    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| {
        eigen.eigenvalues[*b]
            .partial_cmp(&eigen.eigenvalues[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut axes = [[0.0f64; 3]; 3];
    for (k, &e) in order.iter().enumerate() {
        let v = eigen.eigenvectors.column(e);
        axes[k] = [v[0], v[1], v[2]];
    }
    if axes.iter().flatten().any(|c| !c.is_finite()) {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    axes
}

/// Interpolate a piece's rows from the grid of the given spacing.
fn node_grid(piece: &Piece, spacing: f64) -> NodeGrid {
    let mut dims = [1usize; 3];
    for a in 0..3 {
        if piece.extent[a] > THIN * spacing {
            dims[a] = (piece.extent[a] / spacing - 1.0e-9).ceil().max(1.0) as usize + 1;
        }
    }
    let linear = |i: [usize; 3]| (i[0] * dims[1] + i[1]) * dims[2] + i[2];
    let mut raw: Vec<Vec<(usize, f64)>> = Vec::with_capacity(piece.points.len());
    for p in &piece.points {
        let mut low = [0usize; 3];
        let mut frac = [0.0f64; 3];
        for a in 0..3 {
            if dims[a] == 1 {
                continue;
            }
            let s = (p[a] / spacing).max(0.0);
            let cell = (s.floor() as usize).min(dims[a] - 2);
            low[a] = cell;
            frac[a] = (s - cell as f64).clamp(0.0, 1.0);
        }
        let mut row: Vec<(usize, f64)> = Vec::with_capacity(WIDTH);
        for corner in 0..8usize {
            let bit = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
            if (0..3).any(|a| bit[a] == 1 && dims[a] == 1) {
                continue;
            }
            let mut w = 1.0;
            let mut index = low;
            for a in 0..3 {
                if dims[a] == 1 {
                    continue;
                }
                w *= if bit[a] == 1 { frac[a] } else { 1.0 - frac[a] };
                index[a] += bit[a];
            }
            if w > DROP {
                row.push((linear(index), w));
            }
        }
        let total: f64 = row.iter().map(|e| e.1).sum();
        for e in &mut row {
            e.1 /= total;
        }
        raw.push(row);
    }
    // keep the nodes some row reaches, in ascending grid order
    let mut used: Vec<usize> = raw.iter().flatten().map(|e| e.0).collect();
    used.sort_unstable();
    used.dedup();
    let stencil = raw
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(node, w)| (used.binary_search(&node).expect("a used node") as u32, w))
                .collect()
        })
        .collect();
    let nodes = used
        .iter()
        .map(|&n| {
            let i = [n / (dims[1] * dims[2]), (n / dims[2]) % dims[1], n % dims[2]];
            let mut at = [0.0f64; 3];
            for a in 0..3 {
                // a single layer stands at the middle of the piece's extent
                at[a] = if dims[a] == 1 { 0.5 * piece.extent[a] } else { i[a] as f64 * spacing };
            }
            at
        })
        .collect();
    NodeGrid { stencil, nodes }
}

/// The patterns of `A P` and of `P^T A P`, `stencil[row]` the nodes each fine
/// row is interpolated from and `gather[node]` the rows each node interpolates.
fn galerkin_pattern(
    fine: &Pattern,
    stencil: &[Vec<(u32, f32)>],
    gather: &[Vec<u32>],
) -> (Pattern, Pattern) {
    // A P first: the nodes each fine row reaches through its own columns
    let reach: Vec<Vec<u32>> = (0..fine.rows())
        .into_par_iter()
        .map(|row| {
            let mut nodes: Vec<u32> = Vec::new();
            if stencil[row].is_empty() {
                return nodes;
            }
            for &column in fine.row(row) {
                nodes.extend(stencil[column as usize].iter().map(|e| e.0));
            }
            nodes.sort_unstable();
            nodes.dedup();
            nodes
        })
        .collect();
    let rows: Vec<Vec<u32>> = gather
        .par_iter()
        .map(|members| {
            let mut columns: Vec<u32> = Vec::new();
            for &row in members {
                columns.extend_from_slice(&reach[row as usize]);
            }
            columns
        })
        .collect();
    (Pattern::from_rows(reach), Pattern::from_rows(rows))
}

/// Lay the hierarchy over a mesh whose finest operator has the pattern `fine`.
///
/// # Panics
/// If `fine` does not have one row per vertex, or an element names a vertex
/// past the end.
pub fn build(mesh: &Mesh, fine: &Pattern) -> Hierarchy {
    let vertices = mesh.positions.len();
    assert_eq!(fine.rows(), vertices, "the pattern has one row per vertex");
    let mut parent: Vec<u32> = (0..vertices as u32).collect();
    let mut in_element = vec![false; vertices];
    let mut length_sum = vec![0.0f64; vertices];
    let mut length_count = vec![0u64; vertices];
    let distance = |a: u32, b: u32| {
        let (p, q) = (mesh.positions[a as usize], mesh.positions[b as usize]);
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    {
        let mut visit = |element: &[u32]| {
            for (k, &a) in element.iter().enumerate() {
                assert!((a as usize) < vertices, "an element names vertex {a} of {vertices}");
                in_element[a as usize] = true;
                for &b in &element[k + 1..] {
                    union(&mut parent, a, b);
                    // charged to the lower vertex, and moved to the piece's root below
                    length_sum[a.min(b) as usize] += distance(a, b);
                    length_count[a.min(b) as usize] += 1;
                }
            }
        };
        for face in mesh.faces {
            visit(face);
        }
        for edge in mesh.edges {
            visit(edge);
        }
        for tet in mesh.tets {
            visit(tet);
        }
    }
    // the pieces, in the order of their first vertex
    let mut members: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    let mut loose = 0usize;
    for v in 0..vertices as u32 {
        if !in_element[v as usize] {
            loose += 1;
            continue;
        }
        members.entry(find(&mut parent, v)).or_default().push(v);
    }
    let mut pieces: Vec<Piece> = Vec::with_capacity(members.len());
    for rows in members.into_values() {
        let (mut sum, mut count) = (0.0f64, 0u64);
        for &v in &rows {
            sum += length_sum[v as usize];
            count += length_count[v as usize];
        }
        let edge = if count > 0 { sum / count as f64 } else { 0.0 };
        let world: Vec<[f64; 3]> = rows.iter().map(|&v| mesh.positions[v as usize]).collect();
        let mut mean = [0.0f64; 3];
        for p in &world {
            for a in 0..3 {
                mean[a] += p[a] / world.len() as f64;
            }
        }
        let axes = principal_axes(&world, mean);
        let mut points: Vec<[f64; 3]> = world
            .iter()
            .map(|p| {
                let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
                [
                    d[0] * axes[0][0] + d[1] * axes[0][1] + d[2] * axes[0][2],
                    d[0] * axes[1][0] + d[1] * axes[1][1] + d[2] * axes[1][2],
                    d[0] * axes[2][0] + d[1] * axes[2][1] + d[2] * axes[2][2],
                ]
            })
            .collect();
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        for p in &points {
            for a in 0..3 {
                low[a] = low[a].min(p[a]);
                high[a] = high[a].max(p[a]);
            }
        }
        for p in &mut points {
            for a in 0..3 {
                p[a] -= low[a];
            }
        }
        let extent = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
        let widest = extent.iter().cloned().fold(0.0f64, f64::max);
        // A piece of coincident vertices has no length to space a grid by; any
        // spacing gives it one node, which is all it can carry.
        let spacing = if edge > 0.0 { 2.0 * edge } else { (2.0 * widest).max(1.0) };
        pieces.push(Piece { rows, points, extent, spacing });
    }
    let piece_count = pieces.len();

    let mut levels: Vec<Level> = Vec::new();
    let mut pattern = fine.clone();
    let mut fine_rows = vertices;
    while !pieces.is_empty() {
        // each piece's grid, its spacing doubled until its operator is small enough
        let mut grids: Vec<NodeGrid> = pieces.iter().map(|p| node_grid(p, p.spacing)).collect();
        let coarse = loop {
            let mut base = Vec::with_capacity(pieces.len());
            let mut nodes = 0u32;
            for grid in &grids {
                base.push(nodes);
                nodes += grid.nodes.len() as u32;
            }
            let mut stencil: Vec<Vec<(u32, f32)>> = vec![Vec::new(); fine_rows];
            let mut gather: Vec<Vec<u32>> = vec![Vec::new(); nodes as usize];
            for ((piece, grid), &first) in pieces.iter().zip(&grids).zip(&base) {
                for (k, &row) in piece.rows.iter().enumerate() {
                    stencil[row as usize] =
                        grid.stencil[k].iter().map(|&(n, w)| (first + n, w as f32)).collect();
                    for &(n, _) in &grid.stencil[k] {
                        gather[(first + n) as usize].push(row);
                    }
                }
            }
            let (reach, candidate) = galerkin_pattern(&pattern, &stencil, &gather);
            let mut doubled = false;
            for (k, piece) in pieces.iter_mut().enumerate() {
                let count = grids[k].nodes.len();
                if count <= 1 {
                    continue;
                }
                let above: usize = piece.rows.iter().map(|&r| pattern.row(r as usize).len()).sum();
                let here: usize = (0..count).map(|n| candidate.row(base[k] as usize + n).len()).sum();
                if here as f64 > BLOCK_RATIO * above as f64 {
                    piece.spacing *= 2.0;
                    grids[k] = node_grid(piece, piece.spacing);
                    doubled = true;
                }
            }
            if !doubled {
                break (reach, candidate, stencil, gather, base, nodes);
            }
        };
        let (reach, candidate, stencil, gather, base, nodes) = coarse;
        let mut transfer = Transfer {
            fine: fine_rows as u32,
            nodes,
            node: vec![0u32; WIDTH * fine_rows],
            weight: vec![0.0f32; WIDTH * fine_rows],
            gather_offset: Vec::with_capacity(nodes as usize + 1),
            gather_row: Vec::new(),
            gather_weight: Vec::new(),
        };
        for (row, entries) in stencil.iter().enumerate() {
            for (k, &(n, w)) in entries.iter().enumerate() {
                transfer.node[WIDTH * row + k] = n;
                transfer.weight[WIDTH * row + k] = w;
            }
        }
        transfer.gather_offset.push(0);
        for (n, rows) in gather.iter().enumerate() {
            for &row in rows {
                let w = stencil[row as usize]
                    .iter()
                    .find(|e| e.0 == n as u32)
                    .expect("a gathered row names its node")
                    .1;
                transfer.gather_row.push(row);
                transfer.gather_weight.push(w);
            }
            transfer.gather_offset.push(transfer.gather_row.len() as u32);
        }
        levels.push(Level { transfer, reach, pattern: candidate.clone() });

        // the next level's pieces: every piece still wider than one node
        let mut next: Vec<Piece> = Vec::with_capacity(pieces.len());
        for ((piece, grid), &first) in pieces.into_iter().zip(grids).zip(&base) {
            if grid.nodes.len() <= 1 {
                continue;
            }
            next.push(Piece {
                rows: (0..grid.nodes.len() as u32).map(|n| first + n).collect(),
                points: grid.nodes,
                extent: piece.extent,
                spacing: 2.0 * piece.spacing,
            });
        }
        pieces = next;
        pattern = candidate;
        fine_rows = nodes as usize;
    }
    Hierarchy { levels, pieces: piece_count, loose }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheet(n: usize, tilt: bool) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64);
                positions.push(if tilt {
                    [0.6 * x + 0.3, 0.8 * x + 0.5 * y, 0.87 * y - 0.2]
                } else {
                    [x, y, 0.0]
                });
            }
        }
        let mut faces = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let a = (j * n + i) as u32;
                let (b, c, d) = (a + 1, a + n as u32, a + n as u32 + 1);
                faces.push([a, b, c]);
                faces.push([b, d, c]);
            }
        }
        (positions, faces)
    }

    fn block(n: usize) -> (Vec<[f64; 3]>, Vec<[u32; 4]>) {
        let mut positions = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    positions.push([i as f64 * 0.1, j as f64 * 0.1, k as f64 * 0.1]);
                }
            }
        }
        let at = |i: usize, j: usize, k: usize| ((i * (n + 1) + j) * (n + 1) + k) as u32;
        let mut tets = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let c = |a: usize, b: usize, d: usize| at(i + a, j + b, k + d);
                    for (p, q) in [
                        (c(1, 0, 0), c(1, 1, 0)),
                        (c(1, 1, 0), c(0, 1, 0)),
                        (c(0, 1, 0), c(0, 1, 1)),
                        (c(0, 1, 1), c(0, 0, 1)),
                        (c(0, 0, 1), c(1, 0, 1)),
                        (c(1, 0, 1), c(1, 0, 0)),
                    ] {
                        tets.push([c(0, 0, 0), p, q, c(1, 1, 1)]);
                    }
                }
            }
        }
        (positions, tets)
    }

    fn torus(n: usize, m: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        for i in 0..n {
            let u = std::f64::consts::TAU * i as f64 / n as f64;
            for j in 0..m {
                let v = std::f64::consts::TAU * j as f64 / m as f64;
                let r = 1.0 + 0.3 * v.cos();
                positions.push([r * u.cos(), 0.3 * v.sin(), r * u.sin()]);
            }
        }
        let mut faces = Vec::new();
        for i in 0..n {
            for j in 0..m {
                let a = (i * m + j) as u32;
                let b = (i * m + (j + 1) % m) as u32;
                let c = (((i + 1) % n) * m + j) as u32;
                let d = (((i + 1) % n) * m + (j + 1) % m) as u32;
                faces.push([a, c, b]);
                faces.push([b, c, d]);
            }
        }
        (positions, faces)
    }

    /// The pattern a mesh's elements give: every pair of vertices sharing one.
    fn element_pattern(vertices: usize, faces: &[[u32; 3]], edges: &[[u32; 2]], tets: &[[u32; 4]]) -> Pattern {
        let mut upper: Vec<Vec<u32>> = vec![Vec::new(); vertices];
        let mut pair = |a: u32, b: u32| upper[a.min(b) as usize].push(a.max(b));
        for f in faces {
            for a in 0..3 {
                for b in a + 1..3 {
                    pair(f[a], f[b]);
                }
            }
        }
        for e in edges {
            pair(e[0], e[1]);
        }
        for t in tets {
            for a in 0..4 {
                for b in a + 1..4 {
                    pair(t[a], t[b]);
                }
            }
        }
        Pattern::from_upper(&upper)
    }

    /// The properties every hierarchy must have, whatever it was laid over.
    fn check(h: &Hierarchy, fine: &Pattern) {
        assert!(!h.levels.is_empty(), "a mesh with an element has a coarse level");
        let mut above = fine.clone();
        for (l, level) in h.levels.iter().enumerate() {
            let t = &level.transfer;
            assert_eq!(t.fine as usize, above.rows(), "level {l} follows the one above");
            assert_eq!(level.pattern.rows(), t.nodes as usize);
            let mut touched = vec![false; t.nodes as usize];
            for row in 0..t.fine as usize {
                let w = &t.weight[WIDTH * row..WIDTH * (row + 1)];
                let total: f32 = w.iter().sum();
                assert!(
                    total == 0.0 || (total - 1.0).abs() < 1.0e-5,
                    "level {l} row {row}: the weights sum to {total}"
                );
                for k in 0..WIDTH {
                    assert!(w[k] >= 0.0);
                    if w[k] > 0.0 {
                        touched[t.node[WIDTH * row + k] as usize] = true;
                    }
                }
            }
            assert!(touched.iter().all(|t| *t), "level {l} has a node no row reaches");
            // the gather lists are the same weights, transposed
            let mut seen = 0usize;
            for n in 0..t.nodes as usize {
                for k in t.gather_offset[n] as usize..t.gather_offset[n + 1] as usize {
                    let row = t.gather_row[k] as usize;
                    let slot = (0..WIDTH)
                        .find(|s| t.node[WIDTH * row + s] == n as u32 && t.weight[WIDTH * row + s] > 0.0)
                        .expect("a gathered row interpolates from its node");
                    assert_eq!(t.weight[WIDTH * row + slot], t.gather_weight[k]);
                    seen += 1;
                }
            }
            assert_eq!(seen, t.weight.iter().filter(|w| **w > 0.0).count());
            // `A P` holds, per finer row, the nodes its own columns reach
            assert_eq!(level.reach.rows(), t.fine as usize);
            for row in 0..t.fine as usize {
                let mut want: Vec<u32> = Vec::new();
                if t.weight[WIDTH * row..WIDTH * (row + 1)].iter().any(|w| *w > 0.0) {
                    for &column in above.row(row) {
                        for s in 0..WIDTH {
                            if t.weight[WIDTH * column as usize + s] > 0.0 {
                                want.push(t.node[WIDTH * column as usize + s]);
                            }
                        }
                    }
                }
                want.sort_unstable();
                want.dedup();
                assert_eq!(level.reach.row(row), &want[..], "level {l} finer row {row}");
            }
            // the pattern is exactly the product's support
            for n in 0..t.nodes as usize {
                let mut want: Vec<u32> = Vec::new();
                for k in t.gather_offset[n] as usize..t.gather_offset[n + 1] as usize {
                    for &column in above.row(t.gather_row[k] as usize) {
                        for s in 0..WIDTH {
                            if t.weight[WIDTH * column as usize + s] > 0.0 {
                                want.push(t.node[WIDTH * column as usize + s]);
                            }
                        }
                    }
                }
                want.sort_unstable();
                want.dedup();
                assert_eq!(level.pattern.row(n), &want[..], "level {l} node {n}");
            }
            above = level.pattern.clone();
        }
        let last = h.levels.last().unwrap();
        assert!(
            last.transfer.nodes as usize <= h.pieces,
            "the last level holds at most a node a piece, got {} for {} pieces",
            last.transfer.nodes,
            h.pieces
        );
        let blocks: usize = h.levels.iter().map(|l| l.pattern.blocks()).sum();
        assert!(
            blocks as f64 <= 2.0 * fine.blocks() as f64,
            "the coarse operators hold {blocks} blocks against the finest one's {}",
            fine.blocks()
        );
    }

    #[test]
    fn a_flat_sheet_gets_a_grid_in_its_own_plane_wherever_it_stands() {
        for tilt in [false, true] {
            let (positions, faces) = sheet(33, tilt);
            let fine = element_pattern(positions.len(), &faces, &[], &[]);
            let mesh = Mesh { positions: &positions, faces: &faces, edges: &[], tets: &[] };
            let h = build(&mesh, &fine);
            check(&h, &fine);
            assert_eq!(h.pieces, 1);
            // one layer of nodes: a row interpolates from the four corners of a
            // cell of its plane and never from eight
            let t = &h.levels[0].transfer;
            let widest = (0..t.fine as usize)
                .map(|r| t.weight[WIDTH * r..WIDTH * (r + 1)].iter().filter(|w| **w > 0.0).count())
                .max()
                .unwrap();
            assert!(widest <= 4, "tilt {tilt}: a row interpolates from {widest} nodes");
            assert!(
                (t.nodes as usize) < positions.len() / 3,
                "tilt {tilt}: {} nodes for {} vertices",
                t.nodes,
                positions.len()
            );
        }
    }

    #[test]
    fn an_interpolation_reproduces_the_positions_it_was_laid_over() {
        // The field "position" is linear over the grid, so interpolating the
        // nodes' own coordinates must hand every row its coordinates back along
        // each axis that carries more than one layer. A solid has three such
        // axes and a sheet two.
        let (solid, tets) = block(6);
        let (flat, _) = sheet(21, true);
        for (points, layered_axes) in [(solid, 3usize), (flat, 2usize)] {
            let mut low = [f64::INFINITY; 3];
            let mut high = [f64::NEG_INFINITY; 3];
            let mean = {
                let mut m = [0.0f64; 3];
                for p in &points {
                    for a in 0..3 {
                        m[a] += p[a] / points.len() as f64;
                    }
                }
                m
            };
            let axes = principal_axes(&points, mean);
            let mut local: Vec<[f64; 3]> = points
                .iter()
                .map(|p| {
                    let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
                    let mut q = [0.0f64; 3];
                    for a in 0..3 {
                        q[a] = d[0] * axes[a][0] + d[1] * axes[a][1] + d[2] * axes[a][2];
                    }
                    q
                })
                .collect();
            for p in &local {
                for a in 0..3 {
                    low[a] = low[a].min(p[a]);
                    high[a] = high[a].max(p[a]);
                }
            }
            for p in &mut local {
                for a in 0..3 {
                    p[a] -= low[a];
                }
            }
            let extent = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
            let piece = Piece {
                rows: (0..local.len() as u32).collect(),
                points: local.clone(),
                extent,
                spacing: 0.23,
            };
            let grid = node_grid(&piece, piece.spacing);
            let layered = (0..3).filter(|a| extent[*a] > THIN * piece.spacing).count();
            assert_eq!(layered, layered_axes);
            for (row, entries) in grid.stencil.iter().enumerate() {
                let total: f64 = entries.iter().map(|e| e.1).sum();
                assert!((total - 1.0).abs() < 1.0e-12);
                for a in 0..layered {
                    let back: f64 = entries.iter().map(|e| e.1 * grid.nodes[e.0 as usize][a]).sum();
                    assert!(
                        (back - local[row][a]).abs() < 1.0e-5,
                        "row {row} axis {a}: {back} against {}",
                        local[row][a]
                    );
                }
            }
        }
        let _ = tets;
    }

    #[test]
    fn a_closed_surface_takes_a_wider_first_spacing_than_a_flat_one() {
        let (positions, faces) = torus(96, 32);
        let fine = element_pattern(positions.len(), &faces, &[], &[]);
        let mesh = Mesh { positions: &positions, faces: &faces, edges: &[], tets: &[] };
        let h = build(&mesh, &fine);
        check(&h, &fine);
        let first = &h.levels[0];
        assert!(
            first.pattern.blocks() as f64 <= BLOCK_RATIO * fine.blocks() as f64,
            "the first level holds {} blocks against {}",
            first.pattern.blocks(),
            fine.blocks()
        );
    }

    #[test]
    fn every_piece_has_its_own_grid_and_a_loose_vertex_has_none() {
        // two sheets side by side, a rod, and two vertices in no element
        let (a, fa) = sheet(9, false);
        let (b, fb) = sheet(17, true);
        let mut positions = a.clone();
        positions.extend(b.iter().map(|p| [p[0] + 5.0, p[1], p[2]]));
        let mut faces = fa.clone();
        faces.extend(fb.iter().map(|f| [f[0] + a.len() as u32, f[1] + a.len() as u32, f[2] + a.len() as u32]));
        let rod_first = positions.len() as u32;
        let mut edges = Vec::new();
        for k in 0..20u32 {
            positions.push([-3.0, 0.05 * k as f64, 1.0]);
            if k > 0 {
                edges.push([rod_first + k - 1, rod_first + k]);
            }
        }
        positions.push([9.0, 9.0, 9.0]);
        positions.push([9.5, 9.0, 9.0]);
        let fine = element_pattern(positions.len(), &faces, &edges, &[]);
        let mesh = Mesh { positions: &positions, faces: &faces, edges: &edges, tets: &[] };
        let h = build(&mesh, &fine);
        check(&h, &fine);
        assert_eq!(h.pieces, 3);
        assert_eq!(h.loose, 2);
        let t = &h.levels[0].transfer;
        // no node gathers rows of two pieces
        let piece_of = |row: u32| {
            if (row as usize) < a.len() {
                0
            } else if row < rod_first {
                1
            } else {
                2
            }
        };
        for n in 0..t.nodes as usize {
            let rows = &t.gather_row[t.gather_offset[n] as usize..t.gather_offset[n + 1] as usize];
            assert!(rows.iter().all(|r| piece_of(*r) == piece_of(rows[0])), "node {n} spans pieces");
        }
        // the loose vertices interpolate from nothing
        for row in positions.len() - 2..positions.len() {
            assert!(t.weight[WIDTH * row..WIDTH * (row + 1)].iter().all(|w| *w == 0.0));
        }
        // the rod is a line of nodes: two per row at most
        for row in rod_first as usize..rod_first as usize + 20 {
            let used = t.weight[WIDTH * row..WIDTH * (row + 1)].iter().filter(|w| **w > 0.0).count();
            assert!(used <= 2, "rod row {row} interpolates from {used} nodes");
        }
    }

    #[test]
    fn a_mesh_with_no_element_has_no_coarse_level() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let fine = element_pattern(2, &[], &[], &[]);
        let mesh = Mesh { positions: &positions, faces: &[], edges: &[], tets: &[] };
        let h = build(&mesh, &fine);
        assert!(h.levels.is_empty());
        assert_eq!((h.pieces, h.loose), (0, 2));
    }
}
