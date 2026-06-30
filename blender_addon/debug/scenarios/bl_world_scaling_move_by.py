# File: scenarios/bl_world_scaling_move_by.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling kinematic round-trip: a fully pinned plane driven by a
# MOVE_BY pin op, simulated by the emulated solver with world_scaling=10
# (the scene is over-sized and simulated at 1/10 scale). The emulated
# per-frame PC2 is diffed against the scale-agnostic frontend
# ``fixed.time(t)`` reference. The pin delta is multiplied by the factor
# on ingest (scene.rs) and the output divided back by it (backend.rs),
# so the trajectory must land at the authored scale and match the
# reference -- i.e. world_scaling is a transparent round-trip for a
# kinematically driven mesh.

from __future__ import annotations

from . import _world_scaling_kinematic as _ws
from . import _runner as r


NEEDS_BLENDER = True


CASE = {
    "name": "ws_move_by_x10",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "world_scaling": 10.0,
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
