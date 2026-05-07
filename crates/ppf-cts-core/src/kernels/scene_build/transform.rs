// File: crates/ppf-cts-core/src/kernels/scene_build/transform.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Transform-related kernels: `Object.apply_transform`, `Object.grab`,
// 4x4 matrix composition for `Object.scale` / `Object.rotate`, and the
// `FixedScene.has_violations` message helper.

/// Typed errors from the transform kernels. Today only the axis
/// validator can fail; pre-validating via
/// `validators::validate_object_rotate_axis` is the recommended path
/// but kernels still emit a typed error so a misspelled axis cannot
/// reach the matrix code unchecked.
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("invalid axis: {axis}")]
    InvalidAxis { axis: char },
}

// ---------------------------------------------------------------------------
// Object.apply_transform: 3x3 rotation/scale + optional translation.
//
// Supports the optional `Object.normalize()` pre-step. When the object
// has been normalized, Python pre-stores `_bbox` (per-axis extent) and
// `_center` (axis-aligned bbox center) and applies
//
//     v' = (v - center) / max(bbox)
//
// before the 3x3 multiply. We accept those two as an optional
// `Normalize { bbox_max, center }` so the kernel is the single
// authority on the math.

#[derive(Debug, Clone, Copy)]
pub struct Normalize {
    pub bbox_max: f64,
    pub center: [f64; 3],
}

/// Apply the 3x3 block of a row-major 4x4 transform to a flat
/// `(N*3,)` vertex buffer in place. If `translate` is true, also adds
/// the translation column. When `normalize` is `Some`, first replace
/// each vertex `v` with `(v - center) / bbox_max`.
pub fn apply_transform_batch(
    matrix: &[[f64; 4]; 4],
    verts: &mut [f64],
    translate: bool,
    normalize: Option<Normalize>,
) {
    assert!(verts.len().is_multiple_of(3), "verts len must be multiple of 3");
    let m = matrix;
    let m00 = m[0][0]; let m01 = m[0][1]; let m02 = m[0][2]; let m03 = m[0][3];
    let m10 = m[1][0]; let m11 = m[1][1]; let m12 = m[1][2]; let m13 = m[1][3];
    let m20 = m[2][0]; let m21 = m[2][1]; let m22 = m[2][2]; let m23 = m[2][3];
    let n = verts.len() / 3;
    let (cx, cy, cz, inv_b) = match normalize {
        Some(nrm) => (
            nrm.center[0],
            nrm.center[1],
            nrm.center[2],
            // Match Python's `np.max(self._bbox)` denominator. We don't
            // guard against zero here: Python doesn't either, so a zero
            // bbox propagates as inf/NaN consistently.
            1.0 / nrm.bbox_max,
        ),
        None => (0.0, 0.0, 0.0, 1.0),
    };
    let pre_scale = normalize.is_some();
    for i in 0..n {
        let mut x = verts[3 * i];
        let mut y = verts[3 * i + 1];
        let mut z = verts[3 * i + 2];
        if pre_scale {
            x = (x - cx) * inv_b;
            y = (y - cy) * inv_b;
            z = (z - cz) * inv_b;
        }
        let mut nx = m00 * x + m01 * y + m02 * z;
        let mut ny = m10 * x + m11 * y + m12 * z;
        let mut nz = m20 * x + m21 * y + m22 * z;
        if translate {
            nx += m03;
            ny += m13;
            nz += m23;
        }
        verts[3 * i] = nx;
        verts[3 * i + 1] = ny;
        verts[3 * i + 2] = nz;
    }
}

// ---------------------------------------------------------------------------
// Object.grab. Pick the vertices with maximum dot product against a
// direction, within `eps` of the maximum. Mirrors `Object.grab` (lines
// 4235-4253). Operates on already-transformed `(N, 3)` verts.

pub fn grab_indices(verts: &[f64], direction: [f64; 3], eps: f64) -> Vec<i64> {
    if verts.is_empty() {
        return Vec::new();
    }
    let n = verts.len() / 3;
    let mut max_dot = f64::NEG_INFINITY;
    let mut dots = Vec::with_capacity(n);
    for i in 0..n {
        let d = verts[3 * i] * direction[0]
            + verts[3 * i + 1] * direction[1]
            + verts[3 * i + 2] * direction[2];
        if d > max_dot { max_dot = d; }
        dots.push(d);
    }
    let cutoff = max_dot - eps;
    let mut out = Vec::new();
    for (i, d) in dots.iter().enumerate() {
        if *d > cutoff {
            out.push(i as i64);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 4x4 transform composition for Object.scale / Object.rotate.
//
// Note: the Python `Object.rotate` applies `R @ self._transform` and
// then *re-pastes* the original translation column, so the rotation
// effectively only re-rotates the linear block (around its current
// origin) without moving the object. We mirror that exact semantics.

pub fn mat4_apply_uniform_scale(m: &[[f64; 4]; 4], s: f64) -> [[f64; 4]; 4] {
    // M = M @ diag(s, s, s, 1).
    let mut out = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let factor = if c < 3 { s } else { 1.0 };
            out[r][c] = m[r][c] * factor;
        }
    }
    out
}

/// Build a 4x4 axis-aligned rotation matrix. `axis` is one of `'x'`,
/// `'y'`, `'z'`. `angle_deg` is in degrees.
fn mat4_axis_rotation(axis: char, angle_deg: f64) -> Result<[[f64; 4]; 4], TransformError> {
    let theta = angle_deg.to_radians();
    let c = theta.cos();
    let s = theta.sin();
    let mut r = [[0.0f64; 4]; 4];
    for i in 0..4 { r[i][i] = 1.0; }
    match axis.to_ascii_lowercase() {
        'x' => {
            r[1][1] =  c; r[1][2] = -s;
            r[2][1] =  s; r[2][2] =  c;
        }
        'y' => {
            r[0][0] =  c; r[0][2] =  s;
            r[2][0] = -s; r[2][2] =  c;
        }
        'z' => {
            r[0][0] =  c; r[0][1] = -s;
            r[1][0] =  s; r[1][1] =  c;
        }
        _ => return Err(TransformError::InvalidAxis { axis }),
    }
    Ok(r)
}

#[inline]
fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Apply `R @ M` then restore `M`'s translation column.
pub fn mat4_apply_axis_rotation_keep_translation(
    m: &[[f64; 4]; 4],
    axis: char,
    angle_deg: f64,
) -> Result<[[f64; 4]; 4], TransformError> {
    let r = mat4_axis_rotation(axis, angle_deg)?;
    let mut out = mat4_mul(&r, m);
    // Restore translation column from M (R doesn't translate, but
    // matching Python behavior exactly).
    out[0][3] = m[0][3];
    out[1][3] = m[1][3];
    out[2][3] = m[2][3];
    Ok(out)
}

// ---------------------------------------------------------------------------
// FixedScene.has_violations / get_violation_messages.

/// Convert four boolean violation flags into the user-facing message
/// list, in the same order as `FixedScene.get_violation_messages`.
pub fn violation_messages(
    has_self_intersection: bool,
    has_contact_offset_violation: bool,
    has_wall_violation: bool,
    has_sphere_violation: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    if has_self_intersection {
        out.push("Scene has self-intersections".to_string());
    }
    if has_contact_offset_violation {
        out.push("Scene has contact-offset violations".to_string());
    }
    if has_wall_violation {
        out.push("Scene has wall constraint violations".to_string());
    }
    if has_sphere_violation {
        out.push("Scene has sphere constraint violations".to_string());
    }
    out
}
