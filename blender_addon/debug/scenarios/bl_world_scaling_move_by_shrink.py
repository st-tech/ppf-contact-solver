# File: scenarios/bl_world_scaling_move_by_shrink.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling kinematic round-trip, SHRINK direction. Same MOVE_BY
# pin op as bl_world_scaling_move_by but with world_scaling=0.1 (an
# under-sized scene simulated 10x larger). Exercising the factor below
# 1.0 confirms the scale-in / scale-out is symmetric: the emulated
# trajectory still matches the scale-agnostic frontend reference.

from __future__ import annotations

from . import _world_scaling_kinematic as _ws
from . import _runner as r


NEEDS_BLENDER = True


CASE = {
    "name": "ws_move_by_x0p1",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "world_scaling": 0.1,
    "tolerance": 1e-3,
    "ops": [
        {"type": "MOVE_BY", "delta": (0.5, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx: r.ScenarioContext) -> str:
    return _ws.build_driver(CASE, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return _ws.run(ctx, CASE)
