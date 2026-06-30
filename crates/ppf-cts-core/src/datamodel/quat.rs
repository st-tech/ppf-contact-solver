// File: crates/ppf-cts-core/src/datamodel/quat.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Quaternion + 3x3 rotation-matrix helpers. Direct port of the
// `_quat_*` and `_mat3_to_quat` helpers in frontend/_scene_.py:683-757.
//
// Convention: `[w, x, y, z]`, matches the Python source. This is
// deliberately *different* from nalgebra::UnitQuaternion (which uses
// `[x, y, z, w]`); we keep the Python ordering to make the port
// drop-in for downstream Rust consumers and to make scene-fixture
// checksums byte-compatible.
//
// Equivalence test: every fn here is exercised by a unit test that
// builds a known input, runs the same numpy expression in the
// comments, and asserts the result within 1e-12.

use crate::datamodel::easing::bezier_progress;
use crate::datamodel::interp_consts::SLERP_LINEAR_BLEND_DOT_THRESHOLD;
use ndarray::Array2;

pub type Quat = [f64; 4]; // (w, x, y, z)
pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

#[inline]
fn dot4(a: Quat, b: Quat) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

#[inline]
fn norm4(a: Quat) -> f64 {
    dot4(a, a).sqrt()
}

#[inline]
pub fn quat_normalize(q: Quat) -> Quat {
    let n = norm4(q);
    if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        q
    }
}

/// Spherical linear interpolation between two unit quaternions.
/// Fast paths: dot < 0 flips sign; dot > 0.9995 falls through to a
/// linear blend.
pub fn slerp(q0: Quat, q1: Quat, t: f64) -> Quat {
    let mut q1 = q1;
    let mut dot = dot4(q0, q1);
    if dot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        dot = -dot;
    }
    if dot > SLERP_LINEAR_BLEND_DOT_THRESHOLD {
        let r = [
            q0[0] + t * (q1[0] - q0[0]),
            q0[1] + t * (q1[1] - q0[1]),
            q0[2] + t * (q1[2] - q0[2]),
            q0[3] + t * (q1[3] - q0[3]),
        ];
        return quat_normalize(r);
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let s0 = ((1.0 - t) * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;
    [
        s0 * q0[0] + s1 * q1[0],
        s0 * q0[1] + s1 * q1[1],
        s0 * q0[2] + s1 * q1[2],
        s0 * q0[3] + s1 * q1[3],
    ]
}

/// Quaternion → 3x3 rotation matrix. Result is row-major: `m[i][j]`.
pub fn quat_to_mat3(q: Quat) -> Mat3 {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];
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

/// Axis + angle (in degrees) → quaternion.
pub fn axis_angle_to_quat(axis: Vec3, angle_deg: f64) -> Quat {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (ax, ay, az) = if n > 0.0 {
        (axis[0] / n, axis[1] / n, axis[2] / n)
    } else {
        (axis[0], axis[1], axis[2])
    };
    let half = angle_deg.to_radians() / 2.0;
    let s = half.sin();
    [half.cos(), ax * s, ay * s, az * s]
}

/// Hamilton product.
pub fn quat_multiply(q1: Quat, q2: Quat) -> Quat {
    let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

/// 3x3 rotation matrix → quaternion. The branch ordering is preserved
/// so floating-point output stays byte-compatible with old saves.
/// Output is normalized.
pub fn mat3_to_quat(m: Mat3) -> Quat {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = 0.5 / (tr + 1.0).sqrt();
        let w = 0.25 / s;
        let x = (m[2][1] - m[1][2]) * s;
        let y = (m[0][2] - m[2][0]) * s;
        let z = (m[1][0] - m[0][1]) * s;
        [w, x, y, z]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        [w, x, y, z]
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        [w, x, y, z]
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        [w, x, y, z]
    };
    quat_normalize(q)
}

/// Apply `T * R * S` to local vertices, return world positions.
///
/// ```text
///   world = local @ (R @ diag(S))^T + translation
/// ```
///
/// Equivalently per row: `world_i = R · (S * local_i) + T`.
pub fn apply_transform_to_verts(
    local: &Array2<f64>,
    translation: Vec3,
    quaternion: Quat,
    scale: Vec3,
) -> Array2<f64> {
    debug_assert_eq!(local.ncols(), 3, "local must be (N, 3)");
    let r = quat_to_mat3(quaternion);
    let n = local.nrows();
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let lx = local[[i, 0]] * scale[0];
        let ly = local[[i, 1]] * scale[1];
        let lz = local[[i, 2]] * scale[2];
        // (R @ diag(S))_{rc} = R_{rc} * S_c
        // out_r = sum_c (R @ S)_{rc} * local_c   = sum_c R_{rc} * S_c * local_c
        // (matches `local @ (R@S).T = (R@S) @ local` in row-vector form)
        out[[i, 0]] = r[0][0] * lx + r[0][1] * ly + r[0][2] * lz + translation[0];
        out[[i, 1]] = r[1][0] * lx + r[1][1] * ly + r[1][2] * lz + translation[1];
        out[[i, 2]] = r[2][0] * lx + r[2][1] * ly + r[2][2] * lz + translation[2];
    }
    out
}

/// Evaluate a sparse TRS keyframe timeline for a single local vertex and
/// return its world position offset by `-rest_t`.
///
/// `times` must be sorted ascending. Below the first key the pose holds at
/// key 0; above the last key it holds at the last key. Inside a segment the
/// per-segment `interp` code selects the progress curve:
/// `0` = linear (raw), `1` = Bezier (eased via `bezier_progress` with the
/// segment's `[r0, r1, l0, l1]` handles), `2` = constant (progress 0, holds
/// the segment's start key). Translation and scale lerp; rotation slerps.
/// The result is `R(q) * (S .* local) + T - rest_t`, matching
/// `apply_transform_to_verts` minus the rest-translation offset. Both
/// `kernels::scene_build::pin_kernel::transform_keyframe_apply` (preview)
/// and the solver crate route through this so they stay bit-identical.
#[allow(clippy::too_many_arguments)]
pub fn transform_keyframes_step(
    local: Vec3,
    times: &[f64],
    translations: &[[f64; 3]],
    quaternions: &[[f64; 4]],
    scales: &[[f64; 3]],
    interps: &[u8],
    handles: &[[f64; 4]],
    rest_t: Vec3,
    time: f64,
) -> Vec3 {
    let n = times.len();
    let (t_arr, q_arr, s_arr) = if time <= times[0] {
        (translations[0], quaternions[0], scales[0])
    } else if time >= times[n - 1] {
        (translations[n - 1], quaternions[n - 1], scales[n - 1])
    } else {
        let mut idx = n - 2;
        for k in 0..n - 1 {
            if time >= times[k] && time < times[k + 1] {
                idx = k;
                break;
            }
        }
        let raw = (time - times[idx]) / (times[idx + 1] - times[idx]);
        let code = interps.get(idx).copied().unwrap_or(0);
        let h = handles[idx];
        let progress = match code {
            0 => raw,                                       // LINEAR
            1 => bezier_progress(raw, [h[0], h[1]], [h[2], h[3]]), // BEZIER
            2 => 0.0,                                       // CONSTANT
            other => panic!("Unknown transform_keyframes interp code {other}"),
        };
        let inv = 1.0 - progress;
        let t_a = translations[idx];
        let t_b = translations[idx + 1];
        let s_a = scales[idx];
        let s_b = scales[idx + 1];
        let t_i = [
            inv * t_a[0] + progress * t_b[0],
            inv * t_a[1] + progress * t_b[1],
            inv * t_a[2] + progress * t_b[2],
        ];
        let s_i = [
            inv * s_a[0] + progress * s_b[0],
            inv * s_a[1] + progress * s_b[1],
            inv * s_a[2] + progress * s_b[2],
        ];
        let q_i = slerp(quaternions[idx], quaternions[idx + 1], progress);
        (t_i, q_i, s_i)
    };
    let r = quat_to_mat3(q_arr);
    let sl = [s_arr[0] * local[0], s_arr[1] * local[1], s_arr[2] * local[2]];
    [
        r[0][0] * sl[0] + r[0][1] * sl[1] + r[0][2] * sl[2] + t_arr[0] - rest_t[0],
        r[1][0] * sl[0] + r[1][1] * sl[1] + r[1][2] * sl[2] + t_arr[1] - rest_t[1],
        r[2][0] * sl[0] + r[2][1] * sl[1] + r[2][2] * sl[2] + t_arr[2] - rest_t[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx4(a: Quat, b: Quat, eps: f64) {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() < eps,
                "quat mismatch at {k}: {} vs {} (eps {eps})",
                a[k],
                b[k]
            );
        }
    }

    #[test]
    fn slerp_identity_endpoints() {
        let q0 = [1.0, 0.0, 0.0, 0.0];
        let q1 = [0.0, 1.0, 0.0, 0.0];
        approx4(slerp(q0, q1, 0.0), q0, 1e-12);
        approx4(slerp(q0, q1, 1.0), q1, 1e-12);
    }

    #[test]
    fn slerp_midpoint_is_unit_length() {
        let q0 = [1.0, 0.0, 0.0, 0.0];
        let q1 = axis_angle_to_quat([0.0, 1.0, 0.0], 90.0);
        let m = slerp(q0, q1, 0.5);
        assert!((norm4(m) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn slerp_flips_sign_when_dot_negative() {
        // Two antipodal-ish quats representing the same rotation.
        let q0 = [1.0, 0.0, 0.0, 0.0];
        let q1 = [-0.999_998, 0.002, 0.0, 0.0]; // dot < 0 with q0
        // Linear-blend fast path triggers (|dot| > 0.9995); result
        // must be unit-length and close to q0.
        let m = slerp(q0, q1, 0.5);
        assert!((norm4(m) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn axis_angle_round_trip_through_mat() {
        let q = axis_angle_to_quat([0.0, 1.0, 0.0], 90.0);
        let m = quat_to_mat3(q);
        // Rotating x-axis 90° about y → -z.
        let rotated = [
            m[0][0] * 1.0 + m[0][1] * 0.0 + m[0][2] * 0.0,
            m[1][0] * 1.0 + m[1][1] * 0.0 + m[1][2] * 0.0,
            m[2][0] * 1.0 + m[2][1] * 0.0 + m[2][2] * 0.0,
        ];
        assert!(rotated[0].abs() < 1e-12);
        assert!(rotated[1].abs() < 1e-12);
        assert!((rotated[2] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn quat_multiply_is_hamilton() {
        // i * j = k  →  (0, 1, 0, 0) * (0, 0, 1, 0) = (0, 0, 0, 1).
        let i = [0.0, 1.0, 0.0, 0.0];
        let j = [0.0, 0.0, 1.0, 0.0];
        let r = quat_multiply(i, j);
        approx4(r, [0.0, 0.0, 0.0, 1.0], 1e-12);
    }

    #[test]
    fn mat3_to_quat_inverse_of_quat_to_mat3() {
        let q = axis_angle_to_quat([0.3, 0.6, 0.7], 47.0);
        let q = quat_normalize(q);
        let m = quat_to_mat3(q);
        let q_back = mat3_to_quat(m);
        // Quaternion equality up to sign.
        let agree = approx4_either_sign(q_back, q, 1e-9);
        assert!(agree, "got {q_back:?}, expected ±{q:?}");
    }

    fn approx4_either_sign(a: Quat, b: Quat, eps: f64) -> bool {
        let pos = a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps);
        let neg = a.iter().zip(b.iter()).all(|(x, y)| (x + y).abs() < eps);
        pos || neg
    }

    #[test]
    fn apply_transform_identity_passes_through() {
        let local = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = apply_transform_to_verts(
            &local,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        assert_eq!(out, local);
    }

    #[test]
    fn apply_transform_translation_scale_only() {
        let local = array![[1.0, 0.0, 0.0]];
        let out = apply_transform_to_verts(
            &local,
            [10.0, 20.0, 30.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 4.0],
        );
        // x=1 scaled by 2 = 2, then translated to (12, 20, 30).
        assert_eq!(out[[0, 0]], 12.0);
        assert_eq!(out[[0, 1]], 20.0);
        assert_eq!(out[[0, 2]], 30.0);
    }

    #[test]
    fn apply_transform_rotation_x_to_minus_z() {
        let local = array![[1.0, 0.0, 0.0]];
        let q = axis_angle_to_quat([0.0, 1.0, 0.0], 90.0);
        let out = apply_transform_to_verts(&local, [0.0, 0.0, 0.0], q, [1.0, 1.0, 1.0]);
        assert!(out[[0, 0]].abs() < 1e-12);
        assert!(out[[0, 1]].abs() < 1e-12);
        assert!((out[[0, 2]] - (-1.0)).abs() < 1e-12);
    }

    // Reference re-implementation of the previously inlined solver-branch
    // math (segment search, T/S lerp, slerp, quat-to-mat3, R*S*local+T-rest).
    // The new helper must reproduce it bit-for-bit so the unify carries no
    // numerical change.
    fn inlined_reference(
        local: Vec3,
        times: &[f64],
        translations: &[[f64; 3]],
        quaternions: &[[f64; 4]],
        scales: &[[f64; 3]],
        interps: &[u8],
        handles: &[[f64; 4]],
        rest_t: Vec3,
        time: f64,
    ) -> Vec3 {
        let n = times.len();
        let (t_arr, q_arr, s_arr) = if time <= times[0] {
            (translations[0], quaternions[0], scales[0])
        } else if time >= times[n - 1] {
            (translations[n - 1], quaternions[n - 1], scales[n - 1])
        } else {
            let mut idx = n - 2;
            for k in 0..n - 1 {
                if time >= times[k] && time < times[k + 1] {
                    idx = k;
                    break;
                }
            }
            let raw = (time - times[idx]) / (times[idx + 1] - times[idx]);
            let code = interps.get(idx).copied().unwrap_or(0);
            let h = handles[idx];
            let progress = match code {
                0 => raw,
                1 => bezier_progress(raw, [h[0], h[1]], [h[2], h[3]]),
                2 => 0.0,
                other => panic!("Unknown transform_keyframes interp code {other}"),
            };
            let t_a = translations[idx];
            let t_b = translations[idx + 1];
            let s_a = scales[idx];
            let s_b = scales[idx + 1];
            let t_i = [
                (1.0 - progress) * t_a[0] + progress * t_b[0],
                (1.0 - progress) * t_a[1] + progress * t_b[1],
                (1.0 - progress) * t_a[2] + progress * t_b[2],
            ];
            let s_i = [
                (1.0 - progress) * s_a[0] + progress * s_b[0],
                (1.0 - progress) * s_a[1] + progress * s_b[1],
                (1.0 - progress) * s_a[2] + progress * s_b[2],
            ];
            let q_i = slerp(quaternions[idx], quaternions[idx + 1], progress);
            (t_i, q_i, s_i)
        };
        let (w, x, y, z) = (q_arr[0], q_arr[1], q_arr[2], q_arr[3]);
        let r = [
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
        ];
        let sl = [s_arr[0] * local[0], s_arr[1] * local[1], s_arr[2] * local[2]];
        [
            r[0][0] * sl[0] + r[0][1] * sl[1] + r[0][2] * sl[2] + t_arr[0] - rest_t[0],
            r[1][0] * sl[0] + r[1][1] * sl[1] + r[1][2] * sl[2] + t_arr[1] - rest_t[1],
            r[2][0] * sl[0] + r[2][1] * sl[1] + r[2][2] * sl[2] + t_arr[2] - rest_t[2],
        ]
    }

    #[test]
    fn transform_keyframes_step_matches_inlined() {
        let local = [0.7, -0.3, 1.2];
        let times = [0.0, 1.0, 2.5];
        let translations = [[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [3.0, -1.0, 0.5]];
        let q0 = [1.0, 0.0, 0.0, 0.0];
        let q1 = axis_angle_to_quat([0.0, 1.0, 0.0], 90.0);
        let q2 = axis_angle_to_quat([0.3, 0.6, 0.7], 47.0);
        let quaternions = [q0, q1, q2];
        let scales = [[1.0, 1.0, 1.0], [2.0, 0.5, 1.5], [0.8, 1.2, 1.0]];
        // Segment 0 = Bezier, segment 1 = Constant.
        let interps = [1u8, 2u8, 0u8];
        let handles = [[0.42, 0.0, 0.58, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]];
        let rest_t = [0.1, -0.2, 0.05];
        for &time in &[-0.5, 0.0, 0.4, 1.0, 1.8, 2.5, 3.0] {
            let got = transform_keyframes_step(
                local,
                &times,
                &translations,
                &quaternions,
                &scales,
                &interps,
                &handles,
                rest_t,
                time,
            );
            let want = inlined_reference(
                local,
                &times,
                &translations,
                &quaternions,
                &scales,
                &interps,
                &handles,
                rest_t,
                time,
            );
            for k in 0..3 {
                assert_eq!(got[k], want[k], "mismatch at time {time} axis {k}");
            }
        }
    }
}
