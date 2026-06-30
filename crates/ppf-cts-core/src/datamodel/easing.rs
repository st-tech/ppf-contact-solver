// File: crates/ppf-cts-core/src/datamodel/easing.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Direct port of `_eased_progress` and `_bezier_progress` from
// frontend/_scene_.py:881-919. Used by every PinOperation that
// wants non-linear time mapping over its [t_start, t_end] window.
//
// Bezier easing: cubic Bezier control points (0,0), P1, P2, (1,1).
// Newton-Raphson on `B_x(u) = t` for 8 iterations matches the
// Python reference exactly; the `_bezier_progress` function in the
// production code uses the same iteration count and stop criterion.

use crate::datamodel::interp_consts::{BEZIER_NEWTON_DERIV_EPS, BEZIER_NEWTON_ITERS};

#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub enum TransitionKind {
    #[default]
    Linear,
    Smooth,
    /// Cubic-Bezier easing. `right_handle` is `P1`, `left_handle` is `P2`.
    /// Both are normalized to `[0, 1]^2`.
    Bezier {
        right_handle: [f64; 2],
        left_handle: [f64; 2],
    },
}


/// Bezier easing helper. Newton-iterated solve of `B_x(u) = t` on the
/// 4-point cubic Bezier `(0,0), P1, P2, (1,1)`.
pub fn bezier_progress(t: f64, right_handle: [f64; 2], left_handle: [f64; 2]) -> f64 {
    let (p1x, p1y) = (right_handle[0], right_handle[1]);
    let (p2x, p2y) = (left_handle[0], left_handle[1]);
    let mut u = t;
    for _ in 0..BEZIER_NEWTON_ITERS {
        let omu = 1.0 - u;
        let omu2 = omu * omu;
        let u2 = u * u;
        let u3 = u2 * u;
        let bx = 3.0 * omu2 * u * p1x + 3.0 * omu * u2 * p2x + u3;
        let dbx = 3.0 * omu2 * p1x + 6.0 * omu * u * (p2x - p1x) + 3.0 * u2 * (1.0 - p2x);
        if dbx.abs() < BEZIER_NEWTON_DERIV_EPS {
            break;
        }
        u -= (bx - t) / dbx;
        u = u.clamp(0.0, 1.0);
    }
    let omu = 1.0 - u;
    let omu2 = omu * omu;
    let u2 = u * u;
    let u3 = u2 * u;
    (3.0 * omu2 * u * p1y + 3.0 * omu * u2 * p2y + u3).clamp(0.0, 1.0)
}

/// Map `time` ∈ `[t_start, t_end]` to a [0, 1] easing progress.
/// Linear is the bare ramp; Smooth is the
/// `progress² · (3 - 2·progress)` smoothstep; Bezier is the Newton
/// solve above.
pub fn eased_progress(time: f64, t_start: f64, t_end: f64, kind: &TransitionKind) -> f64 {
    let denom = t_end - t_start;
    let progress = if denom != 0.0 {
        (time - t_start) / denom
    } else {
        0.0
    };
    match kind {
        TransitionKind::Linear => progress,
        TransitionKind::Smooth => progress * progress * (3.0 - 2.0 * progress),
        TransitionKind::Bezier {
            right_handle,
            left_handle,
        } => bezier_progress(progress, *right_handle, *left_handle),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn linear_passthrough() {
        let p = eased_progress(0.5, 0.0, 1.0, &TransitionKind::Linear);
        assert!(approx(p, 0.5, 1e-15));
    }

    #[test]
    fn smooth_is_zero_at_zero_one_at_one_half_at_half() {
        // Smoothstep(0) = 0, smoothstep(1) = 1, smoothstep(0.5) = 0.5.
        for (t, expected) in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)] {
            let p = eased_progress(t, 0.0, 1.0, &TransitionKind::Smooth);
            assert!(approx(p, expected, 1e-15), "smooth({t}) = {p}, want {expected}");
        }
        // Smoothstep(0.25) = 0.15625.
        let p = eased_progress(0.25, 0.0, 1.0, &TransitionKind::Smooth);
        assert!(approx(p, 0.15625, 1e-15));
    }

    #[test]
    fn bezier_endpoints_are_zero_one() {
        let kind = TransitionKind::Bezier {
            right_handle: [0.42, 0.0],
            left_handle: [0.58, 1.0],
        };
        assert!(approx(eased_progress(0.0, 0.0, 1.0, &kind), 0.0, 1e-9));
        assert!(approx(eased_progress(1.0, 0.0, 1.0, &kind), 1.0, 1e-9));
    }

    #[test]
    fn bezier_diagonal_handles_match_linear() {
        // Handles on the diagonal P1=(1/3, 1/3), P2=(2/3, 2/3) →
        // identity easing.
        let kind = TransitionKind::Bezier {
            right_handle: [1.0 / 3.0, 1.0 / 3.0],
            left_handle: [2.0 / 3.0, 2.0 / 3.0],
        };
        for t in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let p = eased_progress(t, 0.0, 1.0, &kind);
            assert!(approx(p, t, 1e-6), "bezier({t}) = {p}");
        }
    }

    #[test]
    fn eased_progress_offsets_window() {
        // Time window starts at 1.5 and ends at 2.5; midpoint = 2.0.
        let p = eased_progress(2.0, 1.5, 2.5, &TransitionKind::Linear);
        assert!(approx(p, 0.5, 1e-15));
    }

    #[test]
    fn eased_progress_handles_zero_window() {
        // Zero-duration window: Python falls back to a 0/0 nan;
        // we explicitly substitute 0 so downstream applies stay
        // well-defined.
        let p = eased_progress(1.0, 1.0, 1.0, &TransitionKind::Linear);
        assert!(p.is_finite());
    }
}
