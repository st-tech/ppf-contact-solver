# File: scenarios/bl_driven_pin_frame_exact.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A driven (kinematic) fix pin must land EXACTLY on its prescribed path in
# every OUTPUT frame, even when output frames fall strictly between the
# solver's substeps. This is the regression gate for the driven-collider
# jitter: a moving STATIC collider decodes to a shell of kinematic fix pins,
# and if the frame writer interpolates their absolute positions between the
# two substeps bracketing a frame, the chord cuts across the prescribed path
# and the collider wobbles off it. Against a camera that shares the
# collider's motion (the camera samples the exact keyframe path, the collider
# is displayed from the interpolated PC2) that wobble is the visible jiggle.
#
# The fix places each fix pin at its exact pose for the frame time and
# interpolates only the (zero, for an exact Dirichlet pin) residual, so the
# output tracks the path to round-off. This scenario asserts exactly that,
# to an extremely tight epsilon.
#
# Why this catches what the existing pin-fidelity matrix does not: every
# scenario in `_pin_fidelity_common` runs at frame_rate == 1/step_size
# (100 fps, dt 0.01), so every output frame lands ON a substep and the frame
# writer's interpolation is never exercised (alpha is 0). Here frame_rate
# (24) and step_size (0.01) are deliberately COPRIME in period, so output
# frames fall between substeps and the interpolation is active on nearly
# every frame. A SPIN gives a constant, large path curvature on every step,
# so the chord-cut of a regression is uniform and large (~1e-3 at this
# angular velocity and radius) while the fixed output sits at ~6e-8 (the
# fp32 output floor): a five-order-of-magnitude contrast against the
# tolerance below.
#
# The diff is the shared fidelity comparison: PC2 output per frame vs
# `frontend.FixedScene.time(t)` at the exact frame time Rust recorded in
# frame_to_time.out, which is the analytic prescribed pose. Emulated advance
# is a no-op, so this isolates the frame writer; the substep-rewind half of
# the same fix (kinematic pins walked back on a TOI-truncated step) needs a
# real CCD line search and is covered by the GPU driven-collider run.

from __future__ import annotations

from . import _pin_fidelity_common as _common

NEEDS_BLENDER = True

# Extremely tight: the fixed output equals the analytic pose to within
# fixed-point quantization (Q = 2^27), the fp32 output cast, and the
# solver->Blender axis remap. Measured residual is ~6e-8 (the fp32 floor);
# this bound sits just above it. A regression (chord interpolation) misses
# by ~1e-3, three orders of magnitude above this bound.
_TOLERANCE = 1e-6

CASE = {
    "name": "driven_pin_frame_exact_spin",
    "frame_count": 16,
    # Coprime-in-period with step_size so output frames fall BETWEEN substeps
    # (1/24 is not a multiple of 0.01), activating the frame-writer interp.
    "frame_rate": 24,
    "step_size": 0.01,
    "ops": [
        {
            "type": "SPIN",
            "axis": (0.0, 0.0, 1.0),
            "angular_velocity": 540.0,  # deg/s: 1.5 rev/s, large per-step arc
            "center_mode": "CENTROID",
            "frame_start": 1,
            "frame_end": 16,
            "transition": "LINEAR",
        },
    ],
    "tolerance": _TOLERANCE,
}


def build_driver(ctx):
    return _common.build_driver(CASE, ctx)


def run(ctx):
    return _common.run(ctx, CASE)
