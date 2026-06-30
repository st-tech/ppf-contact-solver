# File: scenarios/bl_world_scaling_spin.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling kinematic round-trip for a SPIN pin op about the pin
# centroid, world_scaling=10. A rotation is scale-invariant in
# magnitude (it carries no length), but the rotated vertex positions are
# lengths and so pass through the scale-in / scale-out. The emulated
# trajectory must still match the scale-agnostic frontend reference,
# confirming spin survives the round-trip.

from __future__ import annotations

from . import _world_scaling_kinematic as _ws
from . import _runner as r


NEEDS_BLENDER = True


CASE = {
    "name": "ws_spin_centroid_x10",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "world_scaling": 10.0,
    "tolerance": 1e-3,
    "ops": [
        {"type": "SPIN",
         "axis": (0.0, 0.0, 1.0), "angular_velocity": 180.0,
         "center_mode": "CENTROID",
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx: r.ScenarioContext) -> str:
    return _ws.build_driver(CASE, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return _ws.run(ctx, CASE)
