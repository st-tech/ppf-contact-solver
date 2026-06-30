# File: scenarios/bl_world_scaling_scale_op.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling kinematic round-trip for a SCALE pin op about an
# ABSOLUTE pivot, world_scaling=10. The pin SCALE factor is
# dimensionless (a ratio), but its pivot is a world-space length and is
# scaled on ingest (scene.rs). A vertex scaled about that pivot is a
# length and round-trips through scale-in / scale-out. The emulated
# trajectory must match the scale-agnostic frontend reference,
# confirming the pin-scale pivot scales with the geometry.

from __future__ import annotations

from . import _world_scaling_kinematic as _ws
from . import _runner as r


NEEDS_BLENDER = True


CASE = {
    "name": "ws_scale_op_x10",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "world_scaling": 10.0,
    "tolerance": 1e-3,
    "ops": [
        {"type": "SCALE", "factor": 1.5,
         "center_mode": "ABSOLUTE", "center": (0.25, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx: r.ScenarioContext) -> str:
    return _ws.build_driver(CASE, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return _ws.run(ctx, CASE)
