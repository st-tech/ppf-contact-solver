// File: crates/ppf-cts-core/src/kernels/scene_build/bbox.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Bounding box / axis-extent helpers used by Scene/Object queries.

// ---------------------------------------------------------------------------
// Bounding boxes / axis extents.

/// Bounding box of a flat `(N, 3)` vertex buffer. Returns `(hi, lo)`.
/// Returns all-zero arrays when the buffer is empty.
pub fn bbox(verts: &[f64]) -> ([f64; 3], [f64; 3]) {
    if verts.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut lo = [f64::INFINITY; 3];
    let n = verts.len() / 3;
    for i in 0..n {
        for k in 0..3 {
            let x = verts[3 * i + k];
            if x > hi[k] { hi[k] = x; }
            if x < lo[k] { lo[k] = x; }
        }
    }
    (hi, lo)
}

/// Bounding box of `vert[1] + displacement[vert[0]]`. `vert_idx` is the
/// per-vertex object index; `local_vert` is the flat `(N, 3)` local
/// position buffer. `displacement` is flat `(M, 3)` per-object delta.
pub fn bbox_displaced(
    local_vert: &[f64],
    vert_idx: &[i64],
    displacement: &[f64],
) -> ([f64; 3], [f64; 3]) {
    let n = vert_idx.len();
    debug_assert_eq!(local_vert.len(), 3 * n, "local_vert len/idx mismatch");
    if n == 0 {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut lo = [f64::INFINITY; 3];
    for i in 0..n {
        let oi = vert_idx[i] as usize;
        for k in 0..3 {
            let x = local_vert[3 * i + k] + displacement[3 * oi + k];
            if x > hi[k] { hi[k] = x; }
            if x < lo[k] { lo[k] = x; }
        }
    }
    (hi, lo)
}

/// World-space min, max along a single axis (0/1/2). Useful when only
/// one component is needed; saves a full bbox scan when iterated across
/// many objects.
pub fn axis_min_max(verts: &[f64], axis: usize) -> (f64, f64) {
    debug_assert!(axis < 3);
    if verts.is_empty() {
        return (f64::INFINITY, f64::NEG_INFINITY);
    }
    let n = verts.len() / 3;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let x = verts[3 * i + axis];
        if x < lo { lo = x; }
        if x > hi { hi = x; }
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Object.bbox (post-transform). Applies rotation+scale (no translate)
// to local verts, then returns `(size, center)`.

/// `(size, center)` of the bounding box after applying the
/// rotation/scale block of a 4x4 matrix to local vertices. Translation
/// column is *not* added. Used for the non-normalize branch.
pub fn object_bbox_no_translate(
    local_vert: &[f64],
    matrix: &[[f64; 4]; 4],
) -> ([f64; 3], [f64; 3]) {
    if local_vert.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let m = matrix;
    let n = local_vert.len() / 3;
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut lo = [f64::INFINITY; 3];
    for i in 0..n {
        let x = local_vert[3 * i];
        let y = local_vert[3 * i + 1];
        let z = local_vert[3 * i + 2];
        let nx = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        let ny = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        let nz = m[2][0] * x + m[2][1] * y + m[2][2] * z;
        let r = [nx, ny, nz];
        for k in 0..3 {
            if r[k] > hi[k] { hi[k] = r[k]; }
            if r[k] < lo[k] { lo[k] = r[k]; }
        }
    }
    let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let center = [
        0.5 * (hi[0] + lo[0]),
        0.5 * (hi[1] + lo[1]),
        0.5 * (hi[2] + lo[2]),
    ];
    (size, center)
}
