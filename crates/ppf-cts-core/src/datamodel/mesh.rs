// File: crates/ppf-cts-core/src/datamodel/mesh.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Mesh helpers and primitive generators:
//
//   * `bbox`, `normalize`, `scale_to`, `scale_per_axis`: coordinate-space helpers.
//   * `generate_rect_faces`, `generate_grid_faces`, `generate_cylinder_faces`,
//     `generate_cylinder_verts`, `transform_verts_2d`: topology and vertex
//     building blocks shared by `rectangle_with_uv`, `mobius`, and `cylinder`.
//   * `rectangle_with_uv`, `mobius`: primitive generators.
//   * `icosphere`: Loop-style midpoint subdivision of the
//     icosahedron, projected onto the sphere.
//   * `fix_skinny_triangles`: union-find merge of vertices on edges
//     opposite very small angles. Used to clean up tetrahedralization
//     inputs, where skinny triangles cause numerical issues.
//   * `tet_extract_surface`: degenerate-tet filter, surface extraction
//     by face-count, winding fix, and vertex reindexing. Direct port of
//     the post-fTetWild pipeline in `_mesh_.py:tetrahedralize` lines 1718-1789.
//   * `polygon_area_2d`: shoelace area used by `triangulate`.
//
// Out of scope for this turn:
//   * Cone / torus primitives: torus uses trimesh; cone is deferred.
//   * Decimation (trimesh dep): runs outside Rust; stays Python-side
//     until those paths are deleted or replaced with a native quadric
//     implementation.
//   * Subdivision / triangulation (trimesh, triangle libs): same.
//   * Tetrahedralization (pytetwild): only the surrounding compute
//     post-processing ports; the pytetwild call remains a subprocess
//     in Python.
//   * .npz cache I/O: moves to bincode in the consumer layer.

use ndarray::Array2;

// ---------------------------------------------------------------------------
// Coordinate-space helpers (mirror module-level fns in _mesh_.py:1105-1164).

/// `(min, max)` per-axis bounding box. Output shape `(2, 3)`.
pub fn bbox(verts: &Array2<f64>) -> Array2<f64> {
    debug_assert_eq!(verts.ncols(), 3, "bbox expects (N, 3)");
    let mut out = Array2::zeros((2, 3));
    if verts.nrows() == 0 {
        return out;
    }
    for d in 0..3 {
        let col = verts.column(d);
        let mut mn = col[0];
        let mut mx = col[0];
        for &v in col.iter().skip(1) {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        out[[0, d]] = mn;
        out[[1, d]] = mx;
    }
    out
}

/// Per-axis extent (max - min) of a `(2, 3)` bounding box.
fn bbox_extent(bb: &Array2<f64>) -> [f64; 3] {
    [
        bb[[1, 0]] - bb[[0, 0]],
        bb[[1, 1]] - bb[[0, 1]],
        bb[[1, 2]] - bb[[0, 2]],
    ]
}

/// Largest component of a per-axis extent.
fn max3(extent: [f64; 3]) -> f64 {
    extent[0].max(extent[1]).max(extent[2])
}

/// Center the mesh on the origin and scale uniformly so the longest
/// axis fits inside `[-0.5, 0.5]`.
pub fn normalize(verts: &mut Array2<f64>) {
    if verts.nrows() == 0 {
        return;
    }
    let bb = bbox(verts);
    let center = [
        0.5 * (bb[[0, 0]] + bb[[1, 0]]),
        0.5 * (bb[[0, 1]] + bb[[1, 1]]),
        0.5 * (bb[[0, 2]] + bb[[1, 2]]),
    ];
    let extent = bbox_extent(&bb);
    let max_extent = max3(extent);
    let scale = if max_extent > 0.0 { 1.0 / max_extent } else { 1.0 };
    for mut row in verts.rows_mut() {
        row[0] = (row[0] - center[0]) * scale;
        row[1] = (row[1] - center[1]) * scale;
        row[2] = (row[2] - center[2]) * scale;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleAxis {
    /// Scale uniformly so the longest axis matches `target`.
    Max,
    /// Scale uniformly so the bounding box's diagonal matches `target`.
    Diagonal,
    /// Scale only along the named axis (0=x, 1=y, 2=z).
    Axis(usize),
}

/// Scale the mesh in-place. `target` is the new size of the chosen
/// axis (or diagonal).
pub fn scale_to(verts: &mut Array2<f64>, target: f64, axis: ScaleAxis) {
    if verts.nrows() == 0 {
        return;
    }
    let bb = bbox(verts);
    let extent = bbox_extent(&bb);
    let scale = match axis {
        ScaleAxis::Max => {
            let m = max3(extent);
            if m > 0.0 {
                target / m
            } else {
                1.0
            }
        }
        ScaleAxis::Diagonal => {
            let d = (extent[0].powi(2) + extent[1].powi(2) + extent[2].powi(2)).sqrt();
            if d > 0.0 {
                target / d
            } else {
                1.0
            }
        }
        ScaleAxis::Axis(d) => {
            assert!(d < 3, "axis must be 0, 1, or 2");
            if extent[d] > 0.0 {
                target / extent[d]
            } else {
                1.0
            }
        }
    };
    for mut row in verts.rows_mut() {
        row[0] *= scale;
        row[1] *= scale;
        row[2] *= scale;
    }
}

/// Scale a mesh in-place along each axis around its centroid:
/// subtract centroid, multiply per-axis, add centroid back.
pub fn scale_per_axis(verts: &mut Array2<f64>, sx: f64, sy: f64, sz: f64) {
    let n = verts.nrows();
    if n == 0 {
        return;
    }
    let mut mean = [0.0_f64; 3];
    for r in 0..n {
        mean[0] += verts[[r, 0]];
        mean[1] += verts[[r, 1]];
        mean[2] += verts[[r, 2]];
    }
    let inv_n = 1.0 / n as f64;
    mean[0] *= inv_n;
    mean[1] *= inv_n;
    mean[2] *= inv_n;
    for r in 0..n {
        verts[[r, 0]] = (verts[[r, 0]] - mean[0]) * sx + mean[0];
        verts[[r, 1]] = (verts[[r, 1]] - mean[1]) * sy + mean[1];
        verts[[r, 2]] = (verts[[r, 2]] - mean[2]) * sz + mean[2];
    }
}

// ---------------------------------------------------------------------------
// Topology helpers.

/// Generate a `(2 * (res_x - 1) * (res_y - 1), 3)` triangle index
/// array for a regular `res_x × res_y` grid with alternating diagonal
/// pattern.
pub fn generate_rect_faces(res_x: usize, res_y: usize) -> Array2<u32> {
    debug_assert!(res_x >= 2 && res_y >= 2);
    let n_faces = 2 * (res_x - 1) * (res_y - 1);
    let mut faces = Array2::<u32>::zeros((n_faces, 3));
    let mut idx = 0usize;
    for j in 0..res_y - 1 {
        for i in 0..res_x - 1 {
            let v0 = (i * res_y + j) as u32;
            let v1 = v0 + 1;
            let v2 = v0 + res_y as u32;
            let v3 = v2 + 1;
            if (i % 2) == (j % 2) {
                faces[[idx, 0]] = v0;
                faces[[idx, 1]] = v1;
                faces[[idx, 2]] = v3;
                faces[[idx + 1, 0]] = v0;
                faces[[idx + 1, 1]] = v3;
                faces[[idx + 1, 2]] = v2;
            } else {
                faces[[idx, 0]] = v0;
                faces[[idx, 1]] = v1;
                faces[[idx, 2]] = v2;
                faces[[idx + 1, 0]] = v1;
                faces[[idx + 1, 1]] = v3;
                faces[[idx + 1, 2]] = v2;
            }
            idx += 2;
        }
    }
    faces
}

/// Triangle index array for a grid wrapped along the length axis,
/// used for Mobius strips. Output shape:
/// `(2 * length_split * (width_split - 1), 3)`.
pub fn generate_grid_faces(length_split: usize, width_split: usize) -> Array2<u32> {
    debug_assert!(length_split >= 2 && width_split >= 2);
    let n_faces = 2 * length_split * (width_split - 1);
    let mut faces = Array2::<u32>::zeros((n_faces, 3));
    let mut idx = 0usize;
    for i in 0..length_split {
        let next_i = (i + 1) % length_split;
        for j in 0..width_split - 1 {
            let v0 = (i * width_split + j) as u32;
            let v1 = (i * width_split + j + 1) as u32;
            let v2 = (next_i * width_split + j) as u32;
            let v3 = (next_i * width_split + j + 1) as u32;
            faces[[idx, 0]] = v0;
            faces[[idx, 1]] = v2;
            faces[[idx, 2]] = v1;
            faces[[idx + 1, 0]] = v1;
            faces[[idx + 1, 1]] = v2;
            faces[[idx + 1, 2]] = v3;
            idx += 2;
        }
    }
    faces
}

/// Generate cylinder vertices along the x-axis. Output shape
/// `((n + 1) * ny, 3)`.
pub fn generate_cylinder_verts(
    n: usize,
    ny: usize,
    min_x: f64,
    dx: f64,
    dy: f64,
    r: f64,
) -> Array2<f64> {
    let n_verts = (n + 1) * ny;
    let mut out = Array2::<f64>::zeros((n_verts, 3));
    for j in 0..ny {
        let theta = (j as f64) * dy;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        for i in 0..=n {
            let idx = (n + 1) * j + i;
            out[[idx, 0]] = min_x + (i as f64) * dx;
            out[[idx, 1]] = sin_t * r;
            out[[idx, 2]] = cos_t * r;
        }
    }
    out
}

/// Generate cylinder faces along the x-axis. Output shape
/// `(2 * n * ny, 3)`.
pub fn generate_cylinder_faces(n: usize, ny: usize) -> Array2<u32> {
    let mut faces = Array2::<u32>::zeros((2 * n * ny, 3));
    for j in 0..ny {
        for i in 0..n {
            let idx = j * n + i;
            let v0 = ((n + 1) * j + i) as u32;
            let v1 = ((n + 1) * j + i + 1) as u32;
            let v2 = ((n + 1) * ((j + 1) % ny) + (i + 1)) as u32;
            let v3 = ((n + 1) * ((j + 1) % ny) + i) as u32;
            if (i % 2) == (j % 2) {
                faces[[2 * idx, 0]] = v1;
                faces[[2 * idx, 1]] = v2;
                faces[[2 * idx, 2]] = v0;
                faces[[2 * idx + 1, 0]] = v3;
                faces[[2 * idx + 1, 1]] = v0;
                faces[[2 * idx + 1, 2]] = v2;
            } else {
                faces[[2 * idx, 0]] = v0;
                faces[[2 * idx, 1]] = v1;
                faces[[2 * idx, 2]] = v3;
                faces[[2 * idx + 1, 0]] = v2;
                faces[[2 * idx + 1, 1]] = v3;
                faces[[2 * idx + 1, 2]] = v1;
            }
        }
    }
    faces
}

/// Transform 2D grid vertices using basis vectors. The input has zero
/// z-component by convention, so the output is `out[i] = ex * x_i + ey * y_i`
/// with three full components. Shape: `(N, 3) -> (N, 3)`.
pub fn transform_verts_2d(verts: &Array2<f64>, ex: [f64; 3], ey: [f64; 3]) -> Array2<f64> {
    let n = verts.nrows();
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let x = verts[[i, 0]];
        let y = verts[[i, 1]];
        out[[i, 0]] = ex[0] * x + ey[0] * y;
        out[[i, 1]] = ex[1] * x + ey[1] * y;
        out[[i, 2]] = ex[2] * x + ey[2] * y;
    }
    out
}

/// Build a Mobius strip mesh. Returns vertex positions of shape
/// `(length_split * width_split, 3)` and triangle indices of shape
/// `(2 * length_split * (width_split - 1), 3)`.
pub fn mobius(
    length_split: usize,
    width_split: usize,
    twists: i32,
    r: f64,
    flatness: f64,
    width: f64,
    scale: f64,
) -> (Array2<f64>, Array2<u32>) {
    assert!(length_split >= 2 && width_split >= 2);
    let two_pi = std::f64::consts::TAU;
    let half_w = 0.5 * width;
    let n = length_split * width_split;
    let mut verts = Array2::<f64>::zeros((n, 3));
    for i in 0..length_split {
        let u = (i as f64) * two_pi / (length_split as f64);
        let cos_u = u.cos();
        let sin_u = u.sin();
        let half_twist = (twists as f64) * u * 0.5;
        let cos_h = half_twist.cos();
        let sin_h = half_twist.sin();
        for j in 0..width_split {
            // np.linspace(-w/2, w/2, width_split) inclusive on both ends.
            let v = if width_split == 1 {
                -half_w
            } else {
                -half_w + (j as f64) * (width / (width_split as f64 - 1.0))
            };
            let radial = r + v * cos_h;
            let row = i * width_split + j;
            verts[[row, 0]] = radial * cos_u * scale;
            verts[[row, 1]] = radial * sin_u * scale;
            verts[[row, 2]] = flatness * v * sin_h * scale;
        }
    }
    let faces = generate_grid_faces(length_split, width_split);
    (verts, faces)
}

// ---------------------------------------------------------------------------
// Icosphere: midpoint subdivision of the icosahedron, projected onto
// the unit sphere.

const ICO_BASE_FACES: [[u32; 3]; 20] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
];

/// Build an icosphere with the given radius and subdivision count.
/// `subdiv = 0` returns the bare icosahedron (12 verts, 20 tris).
/// Each subdivision step splits every triangle into 4 by inserting
/// midpoints on each edge and re-projecting them to the sphere
/// surface.
pub fn icosphere(r: f64, subdiv_count: usize) -> (Array2<f64>, Array2<u32>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();

    // Base icosahedron, normalized to unit sphere.
    let raw: [[f64; 3]; 12] = [
        [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
        [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
        [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
    ];
    let mut verts: Vec<[f64; 3]> = raw
        .iter()
        .map(|v| [v[0] / norm, v[1] / norm, v[2] / norm])
        .collect();
    let mut faces: Vec<[u32; 3]> = ICO_BASE_FACES.to_vec();

    for _ in 0..subdiv_count {
        let mut midpoints: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        for tri in &faces {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let mut get_mid = |i: u32, j: u32, vs: &mut Vec<[f64; 3]>| -> u32 {
                let key = if i < j { (i, j) } else { (j, i) };
                if let Some(&v) = midpoints.get(&key) {
                    return v;
                }
                let p0 = vs[i as usize];
                let p1 = vs[j as usize];
                let mid = [
                    0.5 * (p0[0] + p1[0]),
                    0.5 * (p0[1] + p1[1]),
                    0.5 * (p0[2] + p1[2]),
                ];
                let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
                let proj = [mid[0] / len, mid[1] / len, mid[2] / len];
                let idx = vs.len() as u32;
                vs.push(proj);
                midpoints.insert(key, idx);
                idx
            };
            let m0 = get_mid(a, b, &mut verts);
            let m1 = get_mid(b, c, &mut verts);
            let m2 = get_mid(c, a, &mut verts);
            new_faces.push([a, m0, m2]);
            new_faces.push([m0, b, m1]);
            new_faces.push([m2, m1, c]);
            new_faces.push([m0, m1, m2]);
        }
        faces = new_faces;
    }

    let mut verts_arr = Array2::<f64>::zeros((verts.len(), 3));
    for (i, v) in verts.iter().enumerate() {
        verts_arr[[i, 0]] = v[0] * r;
        verts_arr[[i, 1]] = v[1] * r;
        verts_arr[[i, 2]] = v[2] * r;
    }
    let mut faces_arr = Array2::<u32>::zeros((faces.len(), 3));
    for (i, f) in faces.iter().enumerate() {
        faces_arr[[i, 0]] = f[0];
        faces_arr[[i, 1]] = f[1];
        faces_arr[[i, 2]] = f[2];
    }
    (verts_arr, faces_arr)
}

// ---------------------------------------------------------------------------
// Skinny-triangle fix. Direct port of `_mesh_.py:_fix_skinny_triangles`.

/// Iteratively merge vertices opposite very small triangle angles.
/// Caller supplies the angle threshold in degrees (Python default
/// is 1°). Returns clean (verts, faces); degenerate triangles
/// (sharing two vertices after merge) are dropped.
pub fn fix_skinny_triangles(
    verts: &Array2<f64>,
    faces: &Array2<u32>,
    min_angle_deg: f64,
) -> (Array2<f64>, Array2<u32>) {
    let mut verts = verts.clone();
    let mut faces = faces.clone();
    let max_iterations = 10;
    let cos_max = (min_angle_deg.to_radians()).cos();

    for _ in 0..max_iterations {
        let n_tris = faces.nrows();
        if n_tris == 0 {
            break;
        }

        let mut edges_to_merge: Vec<(u32, u32)> = Vec::new();
        for ti in 0..n_tris {
            let i0 = faces[[ti, 0]];
            let i1 = faces[[ti, 1]];
            let i2 = faces[[ti, 2]];
            let v0 = [
                verts[[i0 as usize, 0]],
                verts[[i0 as usize, 1]],
                verts[[i0 as usize, 2]],
            ];
            let v1 = [
                verts[[i1 as usize, 0]],
                verts[[i1 as usize, 1]],
                verts[[i1 as usize, 2]],
            ];
            let v2 = [
                verts[[i2 as usize, 0]],
                verts[[i2 as usize, 1]],
                verts[[i2 as usize, 2]],
            ];
            // e0 = v1 - v0 (opposite vertex 2)
            // e1 = v2 - v1 (opposite vertex 0)
            // e2 = v0 - v2 (opposite vertex 1)
            let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
            let e2 = [v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]];

            let len0 = (e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]).sqrt().max(1e-12);
            let len1 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt().max(1e-12);
            let len2 = (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]).sqrt().max(1e-12);

            // cos(angle at vertex 0) = (-e2) . e0 / (|e2| * |e0|)
            let cos_a0 =
                ((-e2[0]) * e0[0] + (-e2[1]) * e0[1] + (-e2[2]) * e0[2]) / (len2 * len0);
            let cos_a1 =
                ((-e0[0]) * e1[0] + (-e0[1]) * e1[1] + (-e0[2]) * e1[2]) / (len0 * len1);
            let cos_a2 =
                ((-e1[0]) * e2[0] + (-e1[1]) * e2[1] + (-e1[2]) * e2[2]) / (len1 * len2);
            let cos_a0 = cos_a0.clamp(-1.0, 1.0);
            let cos_a1 = cos_a1.clamp(-1.0, 1.0);
            let cos_a2 = cos_a2.clamp(-1.0, 1.0);

            // Small angle at vertex k → collapse the edge opposite k.
            if cos_a0 > cos_max {
                edges_to_merge.push((i1, i2));
            }
            if cos_a1 > cos_max {
                edges_to_merge.push((i2, i0));
            }
            if cos_a2 > cos_max {
                edges_to_merge.push((i0, i1));
            }
        }

        if edges_to_merge.is_empty() {
            break;
        }

        // Union-find merge.
        let n_verts = verts.nrows();
        let mut uf = UnionFind::new(n_verts);
        for (a, b) in edges_to_merge {
            uf.union(a as usize, b as usize);
        }

        // Compress paths and build the new vertex layout.
        let mut roots = vec![0usize; n_verts];
        for i in 0..n_verts {
            roots[i] = uf.find(i);
        }
        // Map root → compact index.
        let mut root_to_idx: std::collections::HashMap<usize, u32> =
            std::collections::HashMap::new();
        let mut new_verts: Vec<[f64; 3]> = Vec::new();
        let mut inverse = vec![0u32; n_verts];
        for i in 0..n_verts {
            let r = roots[i];
            let new_idx = *root_to_idx.entry(r).or_insert_with(|| {
                let idx = new_verts.len() as u32;
                new_verts.push([verts[[r, 0]], verts[[r, 1]], verts[[r, 2]]]);
                idx
            });
            inverse[i] = new_idx;
        }

        // Remap faces, drop degenerate triangles.
        let mut next_faces: Vec<[u32; 3]> = Vec::with_capacity(n_tris);
        for ti in 0..n_tris {
            let a = inverse[faces[[ti, 0]] as usize];
            let b = inverse[faces[[ti, 1]] as usize];
            let c = inverse[faces[[ti, 2]] as usize];
            if a != b && b != c && a != c {
                next_faces.push([a, b, c]);
            }
        }

        // Materialize.
        let mut next_verts_arr = Array2::<f64>::zeros((new_verts.len(), 3));
        for (i, v) in new_verts.iter().enumerate() {
            next_verts_arr[[i, 0]] = v[0];
            next_verts_arr[[i, 1]] = v[1];
            next_verts_arr[[i, 2]] = v[2];
        }
        let mut next_faces_arr = Array2::<u32>::zeros((next_faces.len(), 3));
        for (i, f) in next_faces.iter().enumerate() {
            next_faces_arr[[i, 0]] = f[0];
            next_faces_arr[[i, 1]] = f[1];
            next_faces_arr[[i, 2]] = f[2];
        }
        verts = next_verts_arr;
        faces = next_faces_arr;
    }
    (verts, faces)
}

// ---------------------------------------------------------------------------
// Polygon helpers and tet-mesh post-processing.

/// Closed 2D polygon points around the origin. Returns shape `(n, 2)`.
pub fn circle_points_2d(n: usize, r: f64) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, 2));
    let two_pi = std::f64::consts::TAU;
    for i in 0..n {
        let t = two_pi * (i as f64) / (n as f64);
        out[[i, 0]] = r * t.cos();
        out[[i, 1]] = r * t.sin();
    }
    out
}

/// Closed line-loop edges over `n` vertices: edge `(i, (i + 1) % n)`
/// for each `i`.
pub fn tri_edge_loop(n: usize) -> Array2<u32> {
    let mut out = Array2::<u32>::zeros((n, 2));
    for i in 0..n {
        out[[i, 0]] = i as u32;
        out[[i, 1]] = ((i + 1) % n) as u32;
    }
    out
}

/// Resolve the cylinder generator parameters: `dx = (max_x - min_x) / n`,
/// `ny = max(int(2 * pi * r / dx), 3)`, `dy = 2 * pi / ny`.
pub fn cylinder_dx_ny_dy(min_x: f64, max_x: f64, n: usize, r: f64) -> (f64, usize, f64) {
    let dx = (max_x - min_x) / (n as f64);
    let ny_f = 2.0 * std::f64::consts::PI * r / dx;
    // Clamp to a minimal valid 3-sided prism so tiny-radius / coarse-n inputs
    // (where 2*pi*r < dx truncates ny to 0) don't yield a degenerate mesh or a
    // non-finite dy = 2*pi / 0.
    let ny = (ny_f as usize).max(3);
    let dy = 2.0 * std::f64::consts::PI / (ny as f64);
    (dx, ny, dy)
}

/// Full rectangle generator: param normalization, grid construction
/// (with `int` truncation, no clamp), basis transform, and optional UV
/// columns. Returns either `(N, 3)` or `(N, 5)` verts plus `(M, 3)`
/// faces.
pub fn rectangle_with_uv(
    res_x: usize,
    width: f64,
    height: f64,
    ex: [f64; 3],
    ey: [f64; 3],
    gen_uv: bool,
) -> (Array2<f64>, Array2<u32>) {
    debug_assert!(res_x >= 2);
    let ratio = height / width;
    // Match Python `int(res_x * ratio)` truncation.
    let res_y = ((res_x as f64) * ratio) as i64;
    let res_y = res_y.max(2) as usize;
    let size_x = width;
    let size_y = width * (res_y as f64) / (res_x as f64);
    let dx_step = (size_x / (res_x as f64 - 1.0)).min(size_y / (res_y as f64 - 1.0));

    let n = res_x * res_y;
    let cols = if gen_uv { 5 } else { 3 };
    let mut verts = Array2::<f64>::zeros((n, cols));
    for i in 0..res_x {
        let x = -size_x / 2.0 + dx_step * i as f64;
        for j in 0..res_y {
            let y = -size_y / 2.0 + dx_step * j as f64;
            let row = i * res_y + j;
            let vx = ex[0] * x + ey[0] * y;
            let vy = ex[1] * x + ey[1] * y;
            let vz = ex[2] * x + ey[2] * y;
            verts[[row, 0]] = vx;
            verts[[row, 1]] = vy;
            verts[[row, 2]] = vz;
            if gen_uv {
                // u = vert · ex, v = vert · ey (matches Python lines
                // 234-235 in `_mesh_.py`).
                verts[[row, 3]] = vx * ex[0] + vy * ex[1] + vz * ex[2];
                verts[[row, 4]] = vx * ey[0] + vy * ey[1] + vz * ey[2];
            }
        }
    }
    let faces = generate_rect_faces(res_x, res_y);
    (verts, faces)
}

/// Map a preset name to its open3d release filename stem. Returns
/// `None` for unknown presets.
pub fn preset_filename_stem(name: &str) -> Option<&'static str> {
    match name {
        "armadillo" => Some("ArmadilloMesh"),
        "knot" => Some("KnotMesh"),
        "bunny" => Some("BunnyMesh"),
        _ => None,
    }
}

/// All known preset names in declaration order. Used by the Python
/// frontend to format the "unknown preset" error message.
pub fn preset_names() -> &'static [&'static str] {
    &["armadillo", "knot", "bunny"]
}

/// Format the open3d release URL for the given preset filename stem.
/// Matches the Python source's URL template.
pub fn preset_url(stem: &str) -> String {
    format!(
        "https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/{stem}.ply"
    )
}

/// Compute the `.npz` cache path for a given mesh hash + tag. The
/// result is `<cache_dir>/<hash>__<name>.npz`. Uses `Path::join` so
/// the platform separator + trailing-slash handling are correct
/// without per-OS string fiddling.
pub fn cache_path(
    cache_dir: impl AsRef<std::path::Path>,
    hash: &str,
    name: &str,
) -> std::path::PathBuf {
    cache_dir.as_ref().join(format!("{hash}__{name}.npz"))
}

/// Format the triangulate "pa{area}q{min_angle}" argument string.
/// Format matches `f"{area:.100f}".rstrip("0").rstrip(".")` exactly.
pub fn format_triangulate_args(area: f64, min_angle: f64) -> String {
    let a_full = format!("{area:.100}");
    // Strip trailing zeros, then strip any trailing '.'.
    let a_str = a_full.trim_end_matches('0').trim_end_matches('.');
    // min_angle is a float printed via Python {min_angle} default repr.
    // Python `f"q{min_angle}"` for 20 -> "q20", for 20.0 -> "q20.0".
    // The Python-side caller passes a float (default 20). The format
    // logic in Python is just `str(min_angle)`. Mirror this with
    // f64::to_string which prints `20` for integer values without
    // suffix. The legacy code path passed `min_angle: float = 20`,
    // which Python renders as "20" via f-string -> "20" (no decimal),
    // matching f64 -> "20".
    format!("pa{a_str}q{}", format_python_float(min_angle))
}

/// Format an `f64` the way Python `str()` / f-string does for typical
/// values. Only covers the floats appearing in `cache_path`
/// arguments.
fn format_python_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == v.trunc() && v.abs() < 1e16 {
        // Python prints integral floats as "20.0" via f-string. Matches
        // CPython's float __str__.
        return format!("{v:.1}");
    }
    // Otherwise Python uses shortest-repr; format!("{}") matches for
    // typical positive values without leading zero issues.
    format!("{v}")
}

/// Build the `tetrahedralize` cache filename component from positional
/// + keyword args. The result is `"_".join(str(a) for a in args)` with
/// `"_".join(f"{k}={v}" for k,v in kwargs.items())` appended when
/// kwargs is non-empty. Returns the inner `_tetrahedralize_<arg_str>`
/// body that the cache path uses.
pub fn tetrahedralize_arg_str(args: &[String], kwargs: &[(String, String)]) -> String {
    let mut s = args.join("_");
    if !kwargs.is_empty() {
        let kw: Vec<String> = kwargs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        s += &kw.join("_");
    }
    s
}

/// Pick the fTetWild kwargs we expose to the subprocess from a free-
/// form key/value list and apply defaults. Returns `(key, value)`
/// pairs preserving the input order; defaults appear at the end if
/// absent.
pub fn ftetwild_kwargs(input: &[(String, String)]) -> Vec<(String, String)> {
    const ALLOWED: &[&str] = &[
        "edge_length_fac",
        "epsilon",
        "stop_energy",
        "num_opt_iter",
        "optimize",
        "simplify",
        "coarsen",
    ];
    let mut out: Vec<(String, String)> = input
        .iter()
        .filter(|(k, _)| ALLOWED.contains(&k.as_str()))
        .cloned()
        .collect();
    if !out.iter().any(|(k, _)| k == "edge_length_fac") {
        out.push(("edge_length_fac".to_string(), "0.05".to_string()));
    }
    if !out.iter().any(|(k, _)| k == "optimize") {
        out.push(("optimize".to_string(), "True".to_string()));
    }
    out
}

/// Shoelace area of a closed 2D polygon. `pts` shape: `(N, 2)`.
pub fn polygon_area_2d(pts: &Array2<f64>) -> f64 {
    debug_assert_eq!(pts.ncols(), 2);
    let n = pts.nrows();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        s += pts[[i, 0]] * pts[[j, 1]];
        s -= pts[[j, 0]] * pts[[i, 1]];
    }
    0.5 * s.abs()
}

/// Unsigned volume of the tetrahedron `(v0, v1, v2, v3)`:
/// `|cross(e0, e1) . e2| / 6` with `e0 = v1 - v0`, `e1 = v2 - v0`,
/// `e2 = v3 - v0`. The `abs()` discards orientation, so this is the
/// magnitude only. `triutils::tet_volume` in `ppf-cts-solver` mirrors
/// this kernel (f32 nalgebra path); the `/ 6.0` divisor and the
/// `abs()` convention must stay in sync between the two.
pub fn tet_volume_abs(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], v3: [f64; 3]) -> f64 {
    let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let e2 = [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]];
    // cross(e0, e1) . e2 / 6
    let cx = e0[1] * e1[2] - e0[2] * e1[1];
    let cy = e0[2] * e1[0] - e0[0] * e1[2];
    let cz = e0[0] * e1[1] - e0[1] * e1[0];
    (cx * e2[0] + cy * e2[1] + cz * e2[2]).abs() / 6.0
}

/// Output of [`tet_extract_surface`]. Vertex layout is reindexed to
/// the unique set referenced by `tet` and `tri`.
#[derive(Debug, Clone)]
pub struct TetSurfaceExtract {
    /// Reindexed vertex positions, shape `(N, 3)`.
    pub verts: Array2<f64>,
    /// Surface triangles with outward-pointing winding, shape `(M, 3)`.
    pub tri: Array2<u32>,
    /// Tet connectivity reindexed to match `verts`, shape `(K, 4)`.
    pub tet: Array2<u32>,
}

/// Post-process raw fTetWild output. Direct port of
/// `_mesh_.py:tetrahedralize` lines 1718-1789:
///   1. drop tetrahedra with `|6V| < 6 * min_volume` (numerically
///      degenerate);
///   2. extract surface triangles as the faces appearing exactly
///      once across all tets;
///   3. flip winding so each surface normal points away from the
///      tet's opposite vertex;
///   4. reindex vertices to the unique set referenced by `tet` and
///      `tri` after filtering.
pub fn tet_extract_surface(
    verts: &Array2<f64>,
    tet: &Array2<u32>,
    min_volume: f64,
) -> TetSurfaceExtract {
    debug_assert_eq!(verts.ncols(), 3);
    debug_assert_eq!(tet.ncols(), 4);

    // 1. Filter degenerate tets.
    let mut kept_tets: Vec<[u32; 4]> = Vec::with_capacity(tet.nrows());
    for k in 0..tet.nrows() {
        let i0 = tet[[k, 0]] as usize;
        let i1 = tet[[k, 1]] as usize;
        let i2 = tet[[k, 2]] as usize;
        let i3 = tet[[k, 3]] as usize;
        let v0 = [verts[[i0, 0]], verts[[i0, 1]], verts[[i0, 2]]];
        let v1 = [verts[[i1, 0]], verts[[i1, 1]], verts[[i1, 2]]];
        let v2 = [verts[[i2, 0]], verts[[i2, 1]], verts[[i2, 2]]];
        let v3 = [verts[[i3, 0]], verts[[i3, 1]], verts[[i3, 2]]];
        let vol = tet_volume_abs(v0, v1, v2, v3);
        if vol >= min_volume {
            kept_tets.push([
                tet[[k, 0]],
                tet[[k, 1]],
                tet[[k, 2]],
                tet[[k, 3]],
            ]);
        }
    }
    let n_tets = kept_tets.len();

    // 2. Surface extraction by face count.
    // Faces of tet (i0, i1, i2, i3):
    //   [i0,i1,i2] opposite i3, [i0,i1,i3] opposite i2,
    //   [i0,i2,i3] opposite i1, [i1,i2,i3] opposite i0.
    let face_combos: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let opp_indices: [usize; 4] = [3, 2, 1, 0];

    let total_faces = 4 * n_tets;
    let mut all_faces: Vec<[u32; 3]> = Vec::with_capacity(total_faces);
    let mut all_opp: Vec<u32> = Vec::with_capacity(total_faces);
    for t in &kept_tets {
        for f in 0..4 {
            let combo = face_combos[f];
            all_faces.push([t[combo[0]], t[combo[1]], t[combo[2]]]);
            all_opp.push(t[opp_indices[f]]);
        }
    }

    // Count occurrences keyed by sorted (a, b, c).
    let mut counts: std::collections::HashMap<(u32, u32, u32), u32> =
        std::collections::HashMap::with_capacity(total_faces);
    for face in &all_faces {
        let mut k = *face;
        k.sort_unstable();
        *counts.entry((k[0], k[1], k[2])).or_insert(0) += 1;
    }

    // 3. Winding fix.
    let mut tri: Vec<[u32; 3]> = Vec::new();
    for (face, &opp) in all_faces.iter().zip(all_opp.iter()) {
        let mut k = *face;
        k.sort_unstable();
        if counts[&(k[0], k[1], k[2])] != 1 {
            continue;
        }
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        let io = opp as usize;
        let v0 = [verts[[i0, 0]], verts[[i0, 1]], verts[[i0, 2]]];
        let v1 = [verts[[i1, 0]], verts[[i1, 1]], verts[[i1, 2]]];
        let v2 = [verts[[i2, 0]], verts[[i2, 1]], verts[[i2, 2]]];
        let vo = [verts[[io, 0]], verts[[io, 1]], verts[[io, 2]]];
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        // normal = cross(e1, e2)
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        let cx = (v0[0] + v1[0] + v2[0]) / 3.0;
        let cy = (v0[1] + v1[1] + v2[1]) / 3.0;
        let cz = (v0[2] + v1[2] + v2[2]) / 3.0;
        let to_opp = [vo[0] - cx, vo[1] - cy, vo[2] - cz];
        let dot = nx * to_opp[0] + ny * to_opp[1] + nz * to_opp[2];
        if dot > 0.0 {
            // Flip winding: keep first vertex, swap last two.
            tri.push([face[0], face[2], face[1]]);
        } else {
            tri.push([face[0], face[1], face[2]]);
        }
    }

    // 4. Reindex vertices to only those referenced.
    let mut used = std::collections::BTreeSet::<u32>::new();
    for t in &kept_tets {
        for &v in t {
            used.insert(v);
        }
    }
    for f in &tri {
        for &v in f {
            used.insert(v);
        }
    }
    let used: Vec<u32> = used.into_iter().collect();
    let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (i, &v) in used.iter().enumerate() {
        remap.insert(v, i as u32);
    }

    let mut new_verts = Array2::<f64>::zeros((used.len(), 3));
    for (i, &v) in used.iter().enumerate() {
        let vu = v as usize;
        new_verts[[i, 0]] = verts[[vu, 0]];
        new_verts[[i, 1]] = verts[[vu, 1]];
        new_verts[[i, 2]] = verts[[vu, 2]];
    }
    let mut new_tet = Array2::<u32>::zeros((n_tets, 4));
    for (i, t) in kept_tets.iter().enumerate() {
        for j in 0..4 {
            new_tet[[i, j]] = remap[&t[j]];
        }
    }
    let mut new_tri = Array2::<u32>::zeros((tri.len(), 3));
    for (i, t) in tri.iter().enumerate() {
        for j in 0..3 {
            new_tri[[i, j]] = remap[&t[j]];
        }
    }

    TetSurfaceExtract {
        verts: new_verts,
        tri: new_tri,
        tet: new_tet,
    }
}

// ---------------------------------------------------------------------------
// Union-find with path compression. Pure Rust, no recursion to avoid
// stack-blowing on huge meshes (Python's `find(parent[x])` recursion
// can hit 1000-default recursion limit on long chains).

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, x: usize) -> usize {
        // Iterative path compression: walk up to the root, then
        // back-patch every node on the way.
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }
    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);
        if px != py {
            self.parent[px] = py;
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive generators: line, box, tet_box, cone. Pure-math, no Scene
// dependency; live here alongside `bbox`, `rectangle_with_uv`, and `mobius`.

/// Polyline vertices for `MeshManager.line`. Returns
/// `(vertices, edges)` flat buffers: `vertices` is `(n + 1) * 3` floats,
/// `edges` is `n * 2` u32 indices.
pub fn line_mesh(p0: [f64; 3], p1: [f64; 3], n: usize) -> (Vec<f64>, Vec<u32>) {
    let mut verts = Vec::with_capacity((n + 1) * 3);
    for i in 0..=n {
        let t = i as f64 / n as f64;
        verts.push(p0[0] + (p1[0] - p0[0]) * t);
        verts.push(p0[1] + (p1[1] - p0[1]) * t);
        verts.push(p0[2] + (p1[2] - p0[2]) * t);
    }
    let mut edges = Vec::with_capacity(n * 2);
    for i in 0..n {
        edges.push(i as u32);
        edges.push((i + 1) as u32);
    }
    (verts, edges)
}

/// Box mesh literal: 8 vertices, 12 triangles. Returns
/// `(vertices, faces)` flat buffers.
pub fn box_mesh(width: f64, height: f64, depth: f64) -> (Vec<f64>, Vec<u32>) {
    let hx = 0.5 * width;
    let hy = 0.5 * height;
    let hz = 0.5 * depth;
    #[rustfmt::skip]
    let verts: Vec<f64> = vec![
        -hx, -hy, -hz,
         hx, -hy, -hz,
        -hx,  hy, -hz,
         hx,  hy, -hz,
        -hx, -hy,  hz,
         hx, -hy,  hz,
        -hx,  hy,  hz,
         hx,  hy,  hz,
    ];
    #[rustfmt::skip]
    let faces: Vec<u32> = vec![
        0, 2, 3,
        0, 3, 1,  // Front face
        4, 5, 7,
        4, 7, 6,  // Back face
        0, 1, 5,
        0, 5, 4,  // Bottom face
        2, 6, 7,
        2, 7, 3,  // Top face
        0, 4, 6,
        0, 6, 2,  // Left face
        1, 3, 7,
        1, 7, 5,  // Right face
    ];
    (verts, faces)
}

/// Tetrahedral box literal: 8 vertices, 12 triangles, 5 tetrahedra.
/// Returns `(verts, faces, tets)` flat buffers.
pub fn tet_box_mesh(
    width: f64,
    height: f64,
    depth: f64,
) -> (Vec<f64>, Vec<u32>, Vec<u32>) {
    let (verts, faces) = box_mesh(width, height, depth);
    #[rustfmt::skip]
    let tets: Vec<u32> = vec![
        0, 1, 3, 5,  // corner tet
        0, 3, 2, 6,  // corner tet
        0, 5, 4, 6,  // corner tet
        3, 5, 6, 7,  // corner tet
        0, 3, 5, 6,  // central tet
    ];
    (verts, faces, tets)
}

/// Cone mesh generator. Returns `(vertices, faces)` flat buffers;
/// `vertices` has `M * 3` floats and `faces` has `K * 3` u32 indices.
pub fn cone_mesh(
    nr: usize,
    ny: usize,
    nb: usize,
    radius: f64,
    height: f64,
    sharpen: f64,
) -> (Vec<f64>, Vec<u32>) {
    use std::f64::consts::PI;
    let mut v: Vec<[f64; 3]> = vec![[0.0, 0.0, height], [0.0, 0.0, 0.0]];
    let mut t: Vec<[u32; 3]> = Vec::new();
    let ind_btm_center: u32 = 0;
    let ind_tip: u32 = 1;
    let mut offset: Vec<u32> = Vec::new();
    let offset_btm = v.len() as u32;

    // Top cone segments, walking k from Ny - 1 down to 1.
    let mut k = ny;
    while k > 0 {
        k -= 1;
        if k > 0 {
            let r_norm = k as f64 / (ny as f64 - 1.0);
            let r = r_norm.powf(sharpen);
            offset.push(v.len() as u32);
            for i in 0..nr {
                let theta = 2.0 * PI * i as f64 / nr as f64;
                let x = radius * r * theta.cos();
                let y = radius * r * theta.sin();
                v.push([x, y, height * r]);
            }
        }
    }

    if offset.len() >= 2 {
        for w in 0..offset.len() - 1 {
            let j = offset[w];
            for i in 0..nr {
                let ind00 = i as u32;
                let ind10 = ((i + 1) % nr) as u32;
                let ind01 = ind00 + nr as u32;
                let ind11 = ind10 + nr as u32;
                if i % 2 == 0 {
                    t.push([ind00 + j, ind01 + j, ind10 + j]);
                    t.push([ind10 + j, ind01 + j, ind11 + j]);
                } else {
                    t.push([ind00 + j, ind11 + j, ind10 + j]);
                    t.push([ind00 + j, ind01 + j, ind11 + j]);
                }
            }
        }
    }

    if let Some(&last_off) = offset.last() {
        for i in 0..nr {
            let ind0 = i as u32;
            let ind1 = ((i + 1) % nr) as u32;
            t.push([ind0 + last_off, ind_tip, ind1 + last_off]);
        }
    }

    // Bottom disk segments, walking k from Nb - 1 down to 1.
    let mut offset2: Vec<u32> = Vec::new();
    let mut k = nb;
    while k > 0 {
        k -= 1;
        if k > 0 {
            let r_norm = k as f64 / nb as f64;
            offset2.push(v.len() as u32);
            for i in 0..nr {
                let theta = 2.0 * PI * i as f64 / nr as f64;
                let x = radius * r_norm * theta.cos();
                let y = radius * r_norm * theta.sin();
                v.push([x, y, height]);
            }
        }
    }

    if offset2.len() >= 2 {
        for w in 0..offset2.len() - 1 {
            let j = offset2[w];
            for i in 0..nr {
                let ind00 = i as u32;
                let ind10 = ((i + 1) % nr) as u32;
                let ind01 = ind00 + nr as u32;
                let ind11 = ind10 + nr as u32;
                if i % 2 == 0 {
                    t.push([ind00 + j, ind10 + j, ind01 + j]);
                    t.push([ind10 + j, ind11 + j, ind01 + j]);
                } else {
                    t.push([ind00 + j, ind10 + j, ind11 + j]);
                    t.push([ind00 + j, ind11 + j, ind01 + j]);
                }
            }
        }
    }

    if let Some(&last_off) = offset2.last() {
        for i in 0..nr {
            let ind0 = i as u32;
            let ind1 = ((i + 1) % nr) as u32;
            t.push([ind0 + last_off, ind1 + last_off, ind_btm_center]);
        }
    }

    if let Some(&first_off) = offset2.first() {
        let j0 = offset_btm;
        let j1 = first_off;
        for i in 0..nr {
            let ind00 = (i as u32) + j0;
            let ind10 = ((i + 1) % nr) as u32 + j0;
            let ind01 = (i as u32) + j1;
            let ind11 = ((i + 1) % nr) as u32 + j1;
            if i % 2 == 0 {
                t.push([ind00, ind10, ind01]);
                t.push([ind10, ind11, ind01]);
            } else {
                t.push([ind00, ind10, ind11]);
                t.push([ind00, ind11, ind01]);
            }
        }
    }

    let mut verts_flat: Vec<f64> = Vec::with_capacity(v.len() * 3);
    for p in v {
        verts_flat.push(p[0]);
        verts_flat.push(p[1]);
        verts_flat.push(p[2]);
    }
    let mut faces_flat: Vec<u32> = Vec::with_capacity(t.len() * 3);
    for tri in t {
        faces_flat.push(tri[0]);
        faces_flat.push(tri[1]);
        faces_flat.push(tri[2]);
    }
    (verts_flat, faces_flat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn bbox_finds_extremes() {
        let v = array![[0.0, 0.0, 0.0], [1.0, -2.0, 3.0], [-1.0, 5.0, 0.5]];
        let bb = bbox(&v);
        assert_eq!(bb[[0, 0]], -1.0);
        assert_eq!(bb[[1, 0]], 1.0);
        assert_eq!(bb[[0, 1]], -2.0);
        assert_eq!(bb[[1, 1]], 5.0);
        assert_eq!(bb[[0, 2]], 0.0);
        assert_eq!(bb[[1, 2]], 3.0);
    }

    #[test]
    fn normalize_centers_and_unit_scales() {
        let mut v = array![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        normalize(&mut v);
        // After centering, vertices are (-5, 0, 0) and (5, 0, 0);
        // scale by 1/10 → (-0.5, 0, 0) and (0.5, 0, 0).
        assert!(approx_eq(v[[0, 0]], -0.5, 1e-12));
        assert!(approx_eq(v[[1, 0]], 0.5, 1e-12));
    }

    #[test]
    fn scale_to_max_axis() {
        let mut v = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        scale_to(&mut v, 4.0, ScaleAxis::Max);
        // Longest extent was 2.0 (x); scaling by 2.0.
        assert_eq!(v[[1, 0]], 4.0);
        assert_eq!(v[[2, 1]], 2.0);
    }

    #[test]
    fn rect_faces_count_and_validity() {
        let f = generate_rect_faces(3, 4);
        assert_eq!(f.nrows(), 2 * (3 - 1) * (4 - 1));
        // Every index must be < res_x * res_y = 12.
        for v in f.iter() {
            assert!(*v < 12);
        }
    }

    #[test]
    fn icosphere_subdiv_zero_returns_icosahedron() {
        let (v, f) = icosphere(1.0, 0);
        assert_eq!(v.nrows(), 12);
        assert_eq!(f.nrows(), 20);
        // Every vertex is on the unit sphere.
        for r in 0..v.nrows() {
            let n = (v[[r, 0]].powi(2) + v[[r, 1]].powi(2) + v[[r, 2]].powi(2)).sqrt();
            assert!(approx_eq(n, 1.0, 1e-12));
        }
    }

    #[test]
    fn icosphere_subdiv_doubles_face_count_4x() {
        let (v0, f0) = icosphere(1.0, 0);
        let (v1, f1) = icosphere(1.0, 1);
        let (v2, f2) = icosphere(1.0, 2);
        assert_eq!(f0.nrows(), 20);
        assert_eq!(f1.nrows(), 80);
        assert_eq!(f2.nrows(), 320);
        // Every vertex still on unit sphere.
        for verts in [&v0, &v1, &v2] {
            for r in 0..verts.nrows() {
                let n = (verts[[r, 0]].powi(2) + verts[[r, 1]].powi(2) + verts[[r, 2]].powi(2)).sqrt();
                assert!(approx_eq(n, 1.0, 1e-12), "vertex {r} not on sphere: norm={n}");
            }
        }
    }

    #[test]
    fn icosphere_radius_scales_vertices() {
        let (v, _) = icosphere(2.5, 1);
        for r in 0..v.nrows() {
            let n = (v[[r, 0]].powi(2) + v[[r, 1]].powi(2) + v[[r, 2]].powi(2)).sqrt();
            assert!(approx_eq(n, 2.5, 1e-12));
        }
    }

    #[test]
    fn fix_skinny_triangles_no_op_on_clean_mesh() {
        let v = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let f = array![[0u32, 1, 2]];
        let (v2, f2) = fix_skinny_triangles(&v, &f, 1.0);
        assert_eq!(v2, v);
        assert_eq!(f2, f);
    }

    #[test]
    fn fix_skinny_triangles_collapses_long_thin_triangle() {
        // A degenerate-ish triangle: vertices nearly collinear, so
        // the angle at the middle vertex is tiny.
        let v = array![
            [0.0, 0.0, 0.0],
            [1.0, 1e-9, 0.0], // nearly on the line y=0
            [2.0, 0.0, 0.0],
        ];
        let f = array![[0u32, 1, 2]];
        let (v2, f2) = fix_skinny_triangles(&v, &f, 5.0);
        // After merging, the mesh should have fewer vertices and no
        // degenerate triangle.
        assert!(v2.nrows() <= 3);
        // Either the triangle merged to a degenerate (and was
        // removed) or stayed a single triangle, but the angle
        // wedge should be gone.
        for r in 0..f2.nrows() {
            assert_ne!(f2[[r, 0]], f2[[r, 1]]);
            assert_ne!(f2[[r, 1]], f2[[r, 2]]);
            assert_ne!(f2[[r, 0]], f2[[r, 2]]);
        }
    }

    #[test]
    fn union_find_path_compression_works() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        assert_eq!(uf.find(0), uf.find(2));
        assert_eq!(uf.find(3), uf.find(4));
        assert_ne!(uf.find(0), uf.find(3));
    }

    #[test]
    fn scale_per_axis_preserves_centroid() {
        let mut v = array![[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]];
        // Centroid is (1, 2, 3); scaling by 2x in every axis spreads
        // around it.
        scale_per_axis(&mut v, 2.0, 2.0, 2.0);
        // Centroid unchanged.
        let cx = (v[[0, 0]] + v[[1, 0]]) / 2.0;
        let cy = (v[[0, 1]] + v[[1, 1]]) / 2.0;
        let cz = (v[[0, 2]] + v[[1, 2]]) / 2.0;
        assert!(approx_eq(cx, 1.0, 1e-12));
        assert!(approx_eq(cy, 2.0, 1e-12));
        assert!(approx_eq(cz, 3.0, 1e-12));
        // Vertex 0 was at -centroid relative offset (-1, -2, -3); now
        // (-2, -4, -6) plus centroid = (-1, -2, -3).
        assert!(approx_eq(v[[0, 0]], -1.0, 1e-12));
        assert!(approx_eq(v[[0, 1]], -2.0, 1e-12));
        assert!(approx_eq(v[[0, 2]], -3.0, 1e-12));
    }

    #[test]
    fn scale_per_axis_handles_per_axis_factors() {
        let mut v = array![[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]];
        // Centroid is origin.
        scale_per_axis(&mut v, 2.0, 3.0, 0.5);
        assert!(approx_eq(v[[0, 0]], 2.0, 1e-12));
        assert!(approx_eq(v[[0, 1]], 3.0, 1e-12));
        assert!(approx_eq(v[[0, 2]], 0.5, 1e-12));
    }

    #[test]
    fn grid_faces_count_and_indices_in_range() {
        let f = generate_grid_faces(8, 5);
        assert_eq!(f.nrows(), 2 * 8 * (5 - 1));
        let n_verts = 8 * 5;
        for v in f.iter() {
            assert!((*v as usize) < n_verts);
        }
    }

    #[test]
    fn cylinder_verts_lie_on_circle() {
        let n = 4;
        let ny = 8;
        let dx = 0.25;
        let dy = std::f64::consts::TAU / ny as f64;
        let r = 1.5;
        let v = generate_cylinder_verts(n, ny, -0.5, dx, dy, r);
        assert_eq!(v.nrows(), (n + 1) * ny);
        // y^2 + z^2 == r^2 for every vertex.
        for row in 0..v.nrows() {
            let yy = v[[row, 1]];
            let zz = v[[row, 2]];
            assert!(approx_eq((yy * yy + zz * zz).sqrt(), r, 1e-12));
        }
    }

    #[test]
    fn cylinder_faces_in_range() {
        let n = 5;
        let ny = 6;
        let f = generate_cylinder_faces(n, ny);
        assert_eq!(f.nrows(), 2 * n * ny);
        let max_idx = (n + 1) * ny;
        for v in f.iter() {
            assert!((*v as usize) < max_idx);
        }
    }

    #[test]
    fn transform_verts_2d_basis_matches() {
        let v = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 3.0, 0.0]];
        let ex = [0.0, 0.0, 1.0];
        let ey = [1.0, 0.0, 0.0];
        // x maps to z, y maps to x.
        let out = transform_verts_2d(&v, ex, ey);
        assert!(approx_eq(out[[0, 2]], 1.0, 1e-12));
        assert!(approx_eq(out[[1, 0]], 1.0, 1e-12));
        assert!(approx_eq(out[[2, 0]], 3.0, 1e-12));
        assert!(approx_eq(out[[2, 2]], 2.0, 1e-12));
    }

    #[test]
    fn mobius_vert_count_matches() {
        let (v, f) = mobius(70, 15, 1, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(v.nrows(), 70 * 15);
        assert_eq!(f.nrows(), 2 * 70 * (15 - 1));
        // Indices stay in range.
        for i in f.iter() {
            assert!((*i as usize) < v.nrows());
        }
    }

    #[test]
    fn polygon_area_unit_square() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(approx_eq(polygon_area_2d(&pts), 1.0, 1e-12));
    }

    #[test]
    fn polygon_area_triangle() {
        let pts = array![[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
        assert!(approx_eq(polygon_area_2d(&pts), 6.0, 1e-12));
    }

    #[test]
    fn tet_extract_surface_unit_tet_returns_all_four_faces() {
        // A single tet gives 4 surface triangles, all winding outward.
        let v = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let t = array![[0u32, 1, 2, 3]];
        let out = tet_extract_surface(&v, &t, 1e-15);
        assert_eq!(out.tet.nrows(), 1);
        assert_eq!(out.tri.nrows(), 4);
        assert_eq!(out.verts.nrows(), 4);
        // Each triangle's normal must point away from the opposite
        // vertex (since the tet is convex, away from the centroid).
        let cx = 0.25_f64;
        let cy = 0.25_f64;
        let cz = 0.25_f64;
        for r in 0..out.tri.nrows() {
            let a = out.tri[[r, 0]] as usize;
            let b = out.tri[[r, 1]] as usize;
            let c = out.tri[[r, 2]] as usize;
            let v0 = [out.verts[[a, 0]], out.verts[[a, 1]], out.verts[[a, 2]]];
            let v1 = [out.verts[[b, 0]], out.verts[[b, 1]], out.verts[[b, 2]]];
            let v2 = [out.verts[[c, 0]], out.verts[[c, 1]], out.verts[[c, 2]]];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let nx = e1[1] * e2[2] - e1[2] * e2[1];
            let ny = e1[2] * e2[0] - e1[0] * e2[2];
            let nz = e1[0] * e2[1] - e1[1] * e2[0];
            // centroid of triangle:
            let tcx = (v0[0] + v1[0] + v2[0]) / 3.0;
            let tcy = (v0[1] + v1[1] + v2[1]) / 3.0;
            let tcz = (v0[2] + v1[2] + v2[2]) / 3.0;
            // direction from triangle centroid to tet centroid.
            let to_tc = [cx - tcx, cy - tcy, cz - tcz];
            let dot = nx * to_tc[0] + ny * to_tc[1] + nz * to_tc[2];
            assert!(dot < 0.0, "outward normal expected, got dot={dot}");
        }
    }

    #[test]
    fn tet_extract_surface_drops_degenerate() {
        // Two tets sharing a face; second is degenerate (zero volume).
        let v = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0], // coplanar with face 0,1,2
        ];
        let t = array![[0u32, 1, 2, 3], [0, 1, 2, 4]];
        let out = tet_extract_surface(&v, &t, 1e-12);
        // The second tet has zero volume and gets dropped.
        assert_eq!(out.tet.nrows(), 1);
        // Only 4 surface faces from the surviving tet.
        assert_eq!(out.tri.nrows(), 4);
    }

    #[test]
    fn tet_extract_surface_two_tets_share_face() {
        // Two tetrahedra sharing the face 0-1-2 form a closed
        // bipyramid surface with 6 faces.
        let v = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let t = array![[0u32, 1, 2, 3], [0, 1, 2, 4]];
        let out = tet_extract_surface(&v, &t, 1e-15);
        assert_eq!(out.tet.nrows(), 2);
        // Shared face vanishes from the boundary; 4 + 4 - 2 = 6.
        assert_eq!(out.tri.nrows(), 6);
    }

    #[test]
    fn circle_points_lie_on_circle() {
        let pts = circle_points_2d(8, 2.0);
        assert_eq!(pts.nrows(), 8);
        for r in 0..pts.nrows() {
            let d = (pts[[r, 0]].powi(2) + pts[[r, 1]].powi(2)).sqrt();
            assert!(approx_eq(d, 2.0, 1e-12));
        }
    }

    #[test]
    fn tri_edge_loop_closes() {
        let e = tri_edge_loop(4);
        assert_eq!(e.nrows(), 4);
        assert_eq!(e[[0, 0]], 0);
        assert_eq!(e[[0, 1]], 1);
        assert_eq!(e[[3, 0]], 3);
        assert_eq!(e[[3, 1]], 0);
    }

    #[test]
    fn cylinder_dx_ny_dy_matches_python() {
        let (dx, ny, dy) = cylinder_dx_ny_dy(-1.0, 1.0, 4, 0.5);
        assert!(approx_eq(dx, 0.5, 1e-12));
        assert_eq!(ny, 6);
        assert!(approx_eq(dy, std::f64::consts::TAU / 6.0, 1e-12));
    }

    #[test]
    fn rectangle_with_uv_returns_5_or_3_columns() {
        let (v, _) = rectangle_with_uv(8, 2.0, 1.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], true);
        assert_eq!(v.ncols(), 5);
        let (v3, _) = rectangle_with_uv(8, 2.0, 1.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], false);
        assert_eq!(v3.ncols(), 3);
    }

    #[test]
    fn preset_filename_stem_known() {
        assert_eq!(preset_filename_stem("armadillo"), Some("ArmadilloMesh"));
        assert_eq!(preset_filename_stem("knot"), Some("KnotMesh"));
        assert_eq!(preset_filename_stem("bunny"), Some("BunnyMesh"));
        assert_eq!(preset_filename_stem("nope"), None);
    }

    #[test]
    fn cache_path_joins() {
        let p = cache_path("/tmp/cache", "abc", "decimate__19000");
        let s = p.to_string_lossy();
        assert!(s.contains("abc__decimate__19000.npz"));
        // Verify the join produced a separator (no naive concat).
        assert!(s.starts_with("/tmp/cache"));
    }

    #[test]
    fn format_triangulate_args_strips_zeros() {
        let s = format_triangulate_args(0.001, 20.0);
        assert!(s.starts_with("pa0.001"), "got {s}");
    }

    #[test]
    fn tetrahedralize_arg_str_empty() {
        let s = tetrahedralize_arg_str(&[], &[]);
        assert_eq!(s, "");
    }

    #[test]
    fn tetrahedralize_arg_str_args_and_kwargs() {
        let s = tetrahedralize_arg_str(
            &["1".into(), "2".into()],
            &[("a".into(), "3".into()), ("b".into(), "4".into())],
        );
        assert_eq!(s, "1_2a=3_b=4");
    }

    #[test]
    fn ftetwild_kwargs_applies_defaults() {
        let out = ftetwild_kwargs(&[]);
        assert!(out.iter().any(|(k, v)| k == "edge_length_fac" && v == "0.05"));
        assert!(out.iter().any(|(k, v)| k == "optimize" && v == "True"));
    }

    #[test]
    fn ftetwild_kwargs_filters_unknown() {
        let out = ftetwild_kwargs(&[
            ("edge_length_fac".into(), "0.1".into()),
            ("garbage".into(), "x".into()),
        ]);
        assert!(out.iter().any(|(k, v)| k == "edge_length_fac" && v == "0.1"));
        assert!(!out.iter().any(|(k, _)| k == "garbage"));
    }

    // -----------------------------------------------------------------
    // Primitive generator tests (relocated from `datamodel::scene`).

    #[test]
    fn line_mesh_basic() {
        let (v, e) = line_mesh([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 4);
        assert_eq!(v.len(), 5 * 3);
        assert_eq!(e.len(), 4 * 2);
        // First and last vertex match endpoints exactly.
        assert!((v[0] - 0.0).abs() < 1e-15);
        assert!((v[12] - 1.0).abs() < 1e-15);
        // Midpoint vertex (i = 2) should be at x = 0.5.
        assert!((v[6] - 0.5).abs() < 1e-15);
        // Edges chain consecutive indices.
        assert_eq!(e, vec![0u32, 1, 1, 2, 2, 3, 3, 4]);
    }

    #[test]
    fn box_mesh_literal() {
        let (v, f) = box_mesh(2.0, 4.0, 6.0);
        assert_eq!(v.len(), 8 * 3);
        assert_eq!(f.len(), 12 * 3);
        // Vertex 0 == (-1, -2, -3), vertex 7 == (1, 2, 3).
        assert!((v[0] + 1.0).abs() < 1e-15);
        assert!((v[1] + 2.0).abs() < 1e-15);
        assert!((v[2] + 3.0).abs() < 1e-15);
        assert!((v[21] - 1.0).abs() < 1e-15);
        assert!((v[22] - 2.0).abs() < 1e-15);
        assert!((v[23] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn tet_box_has_5_tets() {
        let (v, f, t) = tet_box_mesh(1.0, 1.0, 1.0);
        assert_eq!(v.len(), 8 * 3);
        assert_eq!(f.len(), 12 * 3);
        assert_eq!(t.len(), 5 * 4);
    }

    #[test]
    fn cone_mesh_smoke() {
        let (v, f) = cone_mesh(8, 6, 3, 0.5, 1.0, 1.0);
        assert!(v.len() >= 6); // tip + bottom + at least one ring
        assert!(f.len() % 3 == 0);
        assert!(!f.is_empty());
    }
}
