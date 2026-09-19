// File: crates/ppf-cts-solver/src/driver/cadence.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! How often the driver looks at a device scalar during a replayed loop.
//!
//! IT LIVES IN THE DRIVER RATHER THAN IN `ppf-cts-compute`, and that is the one
//! test that crate is held to applied to a concrete case. A residual check
//! cadence decides which iterate a linear solve returns, so it is a convergence
//! policy, and a crate that could be published on its own and used by a program
//! that is not a physics solver has no residual to check. Everything a backend
//! contributes to the same loop, recording it and replaying it without a host
//! round trip, is on the seam; how often the host looks is not.

// Declared and not yet applied, so it has no caller outside its own test. The
// allow is removed by the change that reconciles the three backends' cadences.
#![allow(dead_code)]

/// How often the host looks at a device scalar during a replayed loop.
///
/// CUDA's policy, the constants `src/driver/pcg.rs` spells as
/// `RESID_CHECK_STRIDE = 4` and `NEAR_TOL_FACTOR = 8.0`, tightening to every
/// iteration once the relative residual is within the factor of the tolerance,
/// so a sub-tolerance crossing is never sampled past.
///
/// **Declared here and not yet applied**, and that is a divergence rather than
/// an omission: the three backends test the residual at three cadences today
/// (CUDA every fourth iteration, Metal and the CPU every iteration), so the same
/// recurrence stops at three different instants and publishes three different
/// `advance.iter.out` values. One cadence is the fix and it is CUDA's, because
/// CUDA is the calibrated one; adopting it on this backend changes which iterate
/// a solve returns, so it belongs to the change that reconciles the three rather
/// than to the change that builds the seam.
#[derive(Clone, Copy, Debug)]
pub struct CheckCadence {
    pub stride: u32,
    pub near_factor: f64,
}

impl CheckCadence {
    pub const PCG: CheckCadence = CheckCadence {
        stride: 4,
        near_factor: 8.0,
    };

    pub fn next(self, relative_residual: f64, tolerance: f64) -> u32 {
        if relative_residual < self.near_factor * tolerance {
            1
        } else {
            self.stride
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cadence_tightens_inside_the_near_band() {
        let cadence = CheckCadence::PCG;
        assert_eq!(cadence.next(1.0, 1.0e-3), 4);
        assert_eq!(cadence.next(1.0e-3, 1.0e-3), 1);
    }
}
