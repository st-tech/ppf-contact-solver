# SPIN with VERTEX center: pivot is a specific Blender mesh vertex
# (deforms with the mesh).

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

CASE = {
    "name": "fidelity_spin_vertex_v0",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "SPIN",
         "axis": (0.0, 0.0, 1.0), "angular_velocity": 90.0,
         "center_mode": "VERTEX", "center_vertex": 0,
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
