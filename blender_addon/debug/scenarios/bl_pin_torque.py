# TORQUE: a force op, not kinematic. In emulated Rust mode (no
# dynamics, only kinematic constraints) the torque-pinned vertices
# remain at rest. ``frontend.FixedScene.time(t)`` likewise returns
# rest positions for torque pins (TorqueOperation.apply is a no-op
# at the frontend layer because torque is force-driven), so PC2 and
# fixed.time agree at zero displacement.

from __future__ import annotations
from . import _pin_fidelity_common as _common
from . import _runner as r

NEEDS_BLENDER = True

CASE = {
    "name": "fidelity_torque",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "TORQUE", "magnitude": 1.0, "axis_component": "PC3",
         "frame_start": 1, "frame_end": 4},
    ],
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
