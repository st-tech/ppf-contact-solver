// File: crates/ppf-cts-core/src/datamodel/pin_apply.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Centralized per-vertex pin-operation kernels. Both
// `kernels::scene_build` (driving frontend preview) and the standalone
// solver crate (`src/scene.rs::make_constraint`) call into these so
// preview and simulation produce bit-identical positions.
//
// Primitive types only ([f64; 3], &[f64], etc.). The two callers wrap
// with their own vector library: scene_build uses ndarray slices,
// the solver crate uses nalgebra Vector3<f32>. Their nalgebra
// majors differ (workspace 0.33 vs root 0.32), so this module keeps
// the API free of nalgebra types.
//
// The `transition` argument is a string (`"linear"`, `"smooth"`,
// `"bezier"`) for wire compatibility with the on-disk TOML; a `None`
// `bezier_handles` with `transition == "bezier"` falls back to linear,
// which mirrors the Python source's defensive default.

use super::easing::bezier_progress;

/// Compute eased progress over `[t_start, t_end]` at `time`. Outside
/// the window the caller is expected to short-circuit; inside the
/// window we return the eased ramp.
pub fn progress_at(
    time: f64,
    t_start: f64,
    t_end: f64,
    transition: &str,
    bezier_handles: Option<[f64; 4]>,
) -> f64 {
    let denom = t_end - t_start;
    let raw = if denom != 0.0 {
        (time - t_start) / denom
    } else {
        0.0
    };
    match transition {
        "smooth" => raw * raw * (3.0 - 2.0 * raw),
        "bezier" => match bezier_handles {
            Some([hr0, hr1, hl0, hl1]) => bezier_progress(raw, [hr0, hr1], [hl0, hl1]),
            None => raw,
        },
        _ => raw,
    }
}

/// `position + delta * progress` per axis. Both inputs are 3-vecs;
/// `progress` is the eased ramp produced by `progress_at`.
#[inline]
pub fn move_by_step(position: [f64; 3], delta: [f64; 3], progress: f64) -> [f64; 3] {
    [
        position[0] + delta[0] * progress,
        position[1] + delta[1] * progress,
        position[2] + delta[2] * progress,
    ]
}

/// `lerp(position, target, progress)` per axis.
#[inline]
pub fn move_to_step(position: [f64; 3], target: [f64; 3], progress: f64) -> [f64; 3] {
    let inv = 1.0 - progress;
    [
        position[0] * inv + target[0] * progress,
        position[1] * inv + target[1] * progress,
        position[2] * inv + target[2] * progress,
    ]
}

/// Rodrigues rotation of `position` around an axis through `center`.
/// `axis` is normalized internally; pass it un-normalized. `angle_rad`
/// is the rotation angle in radians (positive = right-hand-rule about
/// the supplied axis).
pub fn spin_step(
    position: [f64; 3],
    center: [f64; 3],
    axis: [f64; 3],
    angle_rad: f64,
) -> [f64; 3] {
    let an = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let inv = if an > 0.0 { 1.0 / an } else { 0.0 };
    let ax = [axis[0] * inv, axis[1] * inv, axis[2] * inv];
    let cos_t = angle_rad.cos();
    let sin_t = angle_rad.sin();
    let one_minus_cos = 1.0 - cos_t;
    let px = position[0] - center[0];
    let py = position[1] - center[1];
    let pz = position[2] - center[2];
    // axis × p
    let cx = ax[1] * pz - ax[2] * py;
    let cy = ax[2] * px - ax[0] * pz;
    let cz = ax[0] * py - ax[1] * px;
    // dot(p, axis)
    let dp = px * ax[0] + py * ax[1] + pz * ax[2];
    [
        px * cos_t + cx * sin_t + ax[0] * dp * one_minus_cos + center[0],
        py * cos_t + cy * sin_t + ax[1] * dp * one_minus_cos + center[1],
        pz * cos_t + cz * sin_t + ax[2] * dp * one_minus_cos + center[2],
    ]
}

/// Convert spin angular velocity (degrees / second) over `[t_start,
/// time.min(t_end)]` into a swept angle in radians. Returns the angle
/// **including its sign**; the spin op is a no-op when the result is
/// non-positive (caller short-circuits before calling `spin_step`).
#[inline]
pub fn spin_angle_rad(
    angular_velocity_deg_per_s: f64,
    t_start: f64,
    t_end: f64,
    time: f64,
) -> f64 {
    let t_eff = time.min(t_end) - t_start;
    angular_velocity_deg_per_s.to_radians() * t_eff
}

/// `(position - center) * factor + center` per axis. Caller derives
/// `factor` from `scale_factor_at`.
#[inline]
pub fn scale_step(position: [f64; 3], center: [f64; 3], factor: f64) -> [f64; 3] {
    [
        (position[0] - center[0]) * factor + center[0],
        (position[1] - center[1]) * factor + center[1],
        (position[2] - center[2]) * factor + center[2],
    ]
}

/// Compute the time-blended scale factor between `1.0` (at `t_start`)
/// and `target_factor` (at or past `t_end`). Below `t_start` returns
/// `1.0`; above `t_end` returns `target_factor`. Inside the window
/// applies the eased ramp.
pub fn scale_factor_at(
    time: f64,
    t_start: f64,
    t_end: f64,
    target_factor: f64,
    transition: &str,
    bezier_handles: Option<[f64; 4]>,
) -> f64 {
    if time < t_start {
        1.0
    } else if time >= t_end {
        target_factor
    } else {
        let progress = progress_at(time, t_start, t_end, transition, bezier_handles);
        1.0 + (target_factor - 1.0) * progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn move_by_step_linear() {
        let out = move_by_step([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 0.5);
        assert!(approx(out[0], 6.0));
        assert!(approx(out[1], 12.0));
        assert!(approx(out[2], 18.0));
    }

    #[test]
    fn move_to_step_endpoints() {
        let p = [1.0, 2.0, 3.0];
        let t = [10.0, 20.0, 30.0];
        let mid = move_to_step(p, t, 0.5);
        assert!(approx(mid[0], 5.5));
        assert!(approx(mid[1], 11.0));
        assert!(approx(mid[2], 16.5));
        let zero = move_to_step(p, t, 0.0);
        assert_eq!(zero, p);
        let one = move_to_step(p, t, 1.0);
        assert_eq!(one, t);
    }

    #[test]
    fn spin_step_quarter_turn_around_y() {
        // Rotate (1, 0, 0) by 90 deg around y axis (through origin) ->
        // (0, 0, -1).
        let out = spin_step(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            std::f64::consts::FRAC_PI_2,
        );
        assert!(approx(out[0], 0.0));
        assert!(approx(out[1], 0.0));
        assert!(approx(out[2], -1.0));
    }

    #[test]
    fn spin_angle_rad_clamps_at_t_end() {
        // 90 deg/s for 0.5s = 45 deg = pi/4. Past t_end stays at 1s.
        let a = spin_angle_rad(90.0, 0.0, 1.0, 0.5);
        assert!(approx(a, std::f64::consts::FRAC_PI_4));
        let a = spin_angle_rad(90.0, 0.0, 1.0, 5.0);
        assert!(approx(a, std::f64::consts::FRAC_PI_2));
    }

    #[test]
    fn scale_step_about_origin() {
        let out = scale_step([2.0, 3.0, 4.0], [0.0, 0.0, 0.0], 2.0);
        assert_eq!(out, [4.0, 6.0, 8.0]);
    }

    #[test]
    fn scale_factor_at_window() {
        // Linear 1.0 -> 3.0 over [0, 1]: at 0.5 it should be 2.0.
        let f = scale_factor_at(0.5, 0.0, 1.0, 3.0, "linear", None);
        assert!(approx(f, 2.0));
        let f = scale_factor_at(-0.1, 0.0, 1.0, 3.0, "linear", None);
        assert!(approx(f, 1.0));
        let f = scale_factor_at(2.0, 0.0, 1.0, 3.0, "linear", None);
        assert!(approx(f, 3.0));
    }

    #[test]
    fn progress_at_smooth_matches_smoothstep() {
        let p = progress_at(0.5, 0.0, 1.0, "smooth", None);
        assert!(approx(p, 0.5));
        let p = progress_at(0.25, 0.0, 1.0, "smooth", None);
        assert!(approx(p, 0.15625));
    }

    #[test]
    fn progress_at_bezier_falls_back_to_linear_without_handles() {
        // No handles + transition="bezier" should match linear so the
        // solver-side path stays well-defined when older saves don't
        // carry handles.
        let p = progress_at(0.42, 0.0, 1.0, "bezier", None);
        assert!(approx(p, 0.42));
    }
}
