// File: crates/ppf-cts-core/src/kernels/scene_build/quaternion.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Quaternion + transform helpers used by static-object animation paths
// (`TransformAnimation.evaluate`, `TransformKeyframeOperation.apply`,
// `Object._ensure_transform_animation`, and friends).
//
// All quaternions are stored as `(w, x, y, z)` to match the Python
// dataclasses; that ordering is what `_quat_*` callers in
// `frontend/_scene_.py` produce and consume.

/// Hamilton product `q1 * q2`.
pub fn quat_multiply(q1: [f64; 4], q2: [f64; 4]) -> [f64; 4] {
    let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

/// Convert a unit quaternion `(w, x, y, z)` to a row-major 3x3 rotation
/// matrix (laid out as `[[r00, r01, r02], [r10, ...], ...]`).
pub fn quat_to_mat3(q: [f64; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Build a unit quaternion from `axis` and `angle_deg`. The axis is
/// normalized inside.
pub fn axis_angle_to_quat(axis: [f64; 3], angle_deg: f64) -> [f64; 4] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let inv = if n > 0.0 { 1.0 / n } else { 0.0 };
    let half = angle_deg.to_radians() * 0.5;
    let s = half.sin();
    [
        half.cos(),
        axis[0] * inv * s,
        axis[1] * inv * s,
        axis[2] * inv * s,
    ]
}

/// Convert a row-major 3x3 rotation matrix to a unit quaternion
/// `(w, x, y, z)`.
pub fn mat3_to_quat(m: [[f64; 3]; 3]) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let (w, x, y, z) = if tr > 0.0 {
        let s = 0.5 / (tr + 1.0).sqrt();
        (
            0.25 / s,
            (m[2][1] - m[1][2]) * s,
            (m[0][2] - m[2][0]) * s,
            (m[1][0] - m[0][1]) * s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        (
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    };
    let n = (w * w + x * x + y * y + z * z).sqrt();
    let inv = if n > 0.0 { 1.0 / n } else { 0.0 };
    [w * inv, x * inv, y * inv, z * inv]
}

/// Spherical linear interpolation `(1-t)q0 + t q1`-style with the
/// shortest-arc + small-angle short-circuit.
pub fn quat_slerp(q0: [f64; 4], q1: [f64; 4], t: f64) -> [f64; 4] {
    let mut q1 = q1;
    let mut dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if dot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        dot = -dot;
    }
    if dot > 0.9995 {
        // Linear blend + renormalize.
        let r = [
            q0[0] + t * (q1[0] - q0[0]),
            q0[1] + t * (q1[1] - q0[1]),
            q0[2] + t * (q1[2] - q0[2]),
            q0[3] + t * (q1[3] - q0[3]),
        ];
        let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
        let inv = if n > 0.0 { 1.0 / n } else { 0.0 };
        return [r[0] * inv, r[1] * inv, r[2] * inv, r[3] * inv];
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_t = theta.sin();
    let a = ((1.0 - t) * theta).sin() / sin_t;
    let b = (t * theta).sin() / sin_t;
    [
        a * q0[0] + b * q1[0],
        a * q0[1] + b * q1[1],
        a * q0[2] + b * q1[2],
        a * q0[3] + b * q1[3],
    ]
}

/// Apply a `T * R(quat) * S(scale)` transform to a flat `(N, 3)` local
/// vertex buffer. Returns a fresh `Vec<f64>` of length `verts.len()`.
/// Computes `(local @ (R @ S).T) + translation`.
pub fn apply_trs_to_verts(
    local_vert: &[f64],
    translation: [f64; 3],
    quaternion: [f64; 4],
    scale: [f64; 3],
) -> Vec<f64> {
    let r = quat_to_mat3(quaternion);
    // RS = R @ diag(scale). Each column of RS is r[:,k] * scale[k].
    let rs = [
        [r[0][0] * scale[0], r[0][1] * scale[1], r[0][2] * scale[2]],
        [r[1][0] * scale[0], r[1][1] * scale[1], r[1][2] * scale[2]],
        [r[2][0] * scale[0], r[2][1] * scale[1], r[2][2] * scale[2]],
    ];
    let n = local_vert.len() / 3;
    let mut out = vec![0.0f64; local_vert.len()];
    for i in 0..n {
        let x = local_vert[3 * i];
        let y = local_vert[3 * i + 1];
        let z = local_vert[3 * i + 2];
        out[3 * i] = rs[0][0] * x + rs[0][1] * y + rs[0][2] * z + translation[0];
        out[3 * i + 1] = rs[1][0] * x + rs[1][1] * y + rs[1][2] * z + translation[1];
        out[3 * i + 2] = rs[2][0] * x + rs[2][1] * y + rs[2][2] * z + translation[2];
    }
    out
}

/// Decompose a row-major 4x4 transform `M = T * R * diag(scale)` into
/// `(translation, quaternion, scale)`. Each scale component is the
/// column norm of `M[:3,:3]`; the rotation block is `M[:3,:3]` with
/// each column divided by the matching scale (when non-zero).
pub fn decompose_trs(matrix: &[[f64; 4]; 4]) -> ([f64; 3], [f64; 4], [f64; 3]) {
    let r = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [matrix[1][0], matrix[1][1], matrix[1][2]],
        [matrix[2][0], matrix[2][1], matrix[2][2]],
    ];
    let translation = [matrix[0][3], matrix[1][3], matrix[2][3]];
    let scale_x = (r[0][0] * r[0][0] + r[1][0] * r[1][0] + r[2][0] * r[2][0]).sqrt();
    let scale_y = (r[0][1] * r[0][1] + r[1][1] * r[1][1] + r[2][1] * r[2][1]).sqrt();
    let scale_z = (r[0][2] * r[0][2] + r[1][2] * r[1][2] + r[2][2] * r[2][2]).sqrt();
    let mut r_norm = r;
    if scale_x > 0.0 {
        r_norm[0][0] /= scale_x;
        r_norm[1][0] /= scale_x;
        r_norm[2][0] /= scale_x;
    }
    if scale_y > 0.0 {
        r_norm[0][1] /= scale_y;
        r_norm[1][1] /= scale_y;
        r_norm[2][1] /= scale_y;
    }
    if scale_z > 0.0 {
        r_norm[0][2] /= scale_z;
        r_norm[1][2] /= scale_z;
        r_norm[2][2] /= scale_z;
    }
    let quat = mat3_to_quat(r_norm);
    (translation, quat, [scale_x, scale_y, scale_z])
}
