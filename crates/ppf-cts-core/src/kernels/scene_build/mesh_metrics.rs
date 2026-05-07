// File: crates/ppf-cts-core/src/kernels/scene_build/mesh_metrics.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Mesh metrics: area-weighted centroid, per-triangle areas,
// face-to-vertex weights, and the average-tri-area helper.

// ---------------------------------------------------------------------------
// FixedScene.center: area-weighted centroid of a triangle mesh.

/// Returns `(cx, cy, cz, total_area)`. Total area of zero indicates a
/// degenerate mesh; the caller is responsible for reacting (Python
/// raises in that case).
pub fn area_weighted_center(verts: &[f64], tris: &[[u32; 3]]) -> ([f64; 3], f64) {
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut cz = 0.0f64;
    let mut total_area = 0.0f64;
    for tri in tris {
        let a = tri[0] as usize;
        let b = tri[1] as usize;
        let c = tri[2] as usize;
        let ax = verts[3 * a];
        let ay = verts[3 * a + 1];
        let az = verts[3 * a + 2];
        let bx = verts[3 * b];
        let by = verts[3 * b + 1];
        let bz = verts[3 * b + 2];
        let cxv = verts[3 * c];
        let cyv = verts[3 * c + 1];
        let czv = verts[3 * c + 2];
        let e1x = bx - ax;
        let e1y = by - ay;
        let e1z = bz - az;
        let e2x = cxv - ax;
        let e2y = cyv - ay;
        let e2z = czv - az;
        // Cross product e1 × e2.
        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;
        let area = 0.5 * (nx * nx + ny * ny + nz * nz).sqrt();
        let centroid_x = (ax + bx + cxv) / 3.0;
        let centroid_y = (ay + by + cyv) / 3.0;
        let centroid_z = (az + bz + czv) / 3.0;
        cx += area * centroid_x;
        cy += area * centroid_y;
        cz += area * centroid_z;
        total_area += area;
    }
    if total_area > 0.0 {
        ([cx / total_area, cy / total_area, cz / total_area], total_area)
    } else {
        ([0.0, 0.0, 0.0], 0.0)
    }
}

// ---------------------------------------------------------------------------
// Triangle-area + face-to-vertex weight helpers.
//
// `triangle_areas` is the vectorized triangle-area kernel;
// `face_to_vert_weights` collapses the count + weight pass into one.

/// `area[i] = 0.5 * |edge1 x edge2|` for each triangle.
pub fn triangle_areas(verts: &[f64], tris: &[[u32; 3]]) -> Vec<f64> {
    let mut out = vec![0.0f64; tris.len()];
    for (i, tri) in tris.iter().enumerate() {
        let a = tri[0] as usize;
        let b = tri[1] as usize;
        let c = tri[2] as usize;
        let ax = verts[3 * a];
        let ay = verts[3 * a + 1];
        let az = verts[3 * a + 2];
        let bx = verts[3 * b];
        let by = verts[3 * b + 1];
        let bz = verts[3 * b + 2];
        let cx = verts[3 * c];
        let cy = verts[3 * c + 1];
        let cz = verts[3 * c + 2];
        let e1x = bx - ax;
        let e1y = by - ay;
        let e1z = bz - az;
        let e2x = cx - ax;
        let e2y = cy - ay;
        let e2z = cz - az;
        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;
        out[i] = 0.5 * (nx * nx + ny * ny + nz * nz).sqrt();
    }
    out
}

/// Per-vertex `1.0 / (count + epsilon)` where `count` is the number of
/// incident triangles.
pub fn face_to_vert_weights(
    n_verts: usize,
    tris: &[[u32; 3]],
    epsilon: f64,
) -> Vec<f64> {
    let mut counts = vec![0.0f64; n_verts];
    for tri in tris {
        counts[tri[0] as usize] += 1.0;
        counts[tri[1] as usize] += 1.0;
        counts[tri[2] as usize] += 1.0;
    }
    counts
        .iter()
        .map(|c| 1.0 / (c + epsilon))
        .collect()
}

/// Average of a non-negative scalar buffer; returns `0.0` for empty.
pub fn average_tri_area(area: &[f64]) -> f64 {
    if area.is_empty() {
        0.0
    } else {
        let s: f64 = area.iter().sum();
        s / area.len() as f64
    }
}
