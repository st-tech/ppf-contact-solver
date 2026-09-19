# File: scenarios/bl_world_scaling_static_pin.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling round-trip for STATIC pins and INITIAL GEOMETRY,
# world_scaling=10. A fully pinned plane holds every vertex at its
# authored rest position for the first frames, then a delayed MOVE_BY
# (frame 5..9) moves it. The early static-hold frames are the purest
# probe of the geometry round-trip: each rest position is scaled by 10
# on ingest (scene.rs) and divided back by 10 on write (backend.rs); if
# either leg were missing the held positions would land 10x off. The
# PC2 must equal the (un-scaled) frontend trajectory across
# both the static and the moving phases.
#
# (A literally op-less pin emits no output frames under the kinematic
# solver, so a delayed op is used to drive frame production while
# still exercising a genuine static-hold phase.)

from __future__ import annotations

from . import _world_scaling_kinematic as _ws
from . import _runner as r


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


CASE = {
    "name": "ws_static_pin_x10",
    "frame_count": 12, "frame_rate": 100, "step_size": 0.01,
    "world_scaling": 10.0,
    "tolerance": 1e-3,
    # Frames 1..4 hold the authored (scaled) geometry static; the op
    # then moves it. Both phases must match the frontend reference.
    "ops": [
        {"type": "MOVE_BY", "delta": (0.3, 0.0, 0.0),
         "frame_start": 5, "frame_end": 9, "transition": "LINEAR"},
    ],
}


def build_driver(ctx: r.ScenarioContext) -> str:
    return _ws.build_driver(CASE, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return _ws.run(ctx, CASE)
