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
# frame_to_time.out, which is the analytic prescribed pose. A fix pin is an
# exact Dirichlet condition, so that pose is what the solver owes to
# round-off whatever else the step does, which is what isolates the frame
# writer here. The substep-rewind half of the same fix (kinematic pins walked
# back on a TOI-truncated step) needs the CCD line search and is covered by
# the GPU driven-collider run.

from __future__ import annotations

from . import _pin_fidelity_common as _common

NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

# NOT ON WINDOWS, WHICH RUNS THE RIG HEADLESS AND SO HAS NO MODAL LOOP.
#
# This scenario needs a Blender that owns a window: the PC2 it asserts on is
# written by `PPF_OT_FramePump.modal` AFTER the driver's exec returns, and a
# modal operator needs an event loop to run in. Measured on the Windows leg of
# Blender CI: the driver reached `fetched queued=9 total=9`, the probe recorded
# `modal_seen: []`, and the scenario finished with ZERO checks and no error,
# because nothing it asserts on had been written yet. The drawing scenarios in
# the same set fail one step earlier and say so outright, with "GPU functions
# for drawing requires the gpu module to be initialized".
#
# Two requirements collide here: a full build/run/fetch scenario must NOT be
# run with `--background`, because the modal operator above needs an event
# loop, and the Windows leg of CI has no window server, so it runs headless.
# There is no configuration in which both hold, so this declares where it can
# run rather than failing there every time. Linux gives the rig its own Xvfb
# and macOS has a real window server.
PLATFORMS = ("linux", "darwin")

# Extremely tight: the fixed output equals the analytic pose to within the
# fp32 output cast and the solver->Blender axis remap. Measured residual is
# ~6e-8 (the fp32 floor); this bound sits just above it. A regression (chord
# interpolation) misses by ~1e-3, three orders of magnitude above this bound.
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
