# File: scenarios/bl_world_scaling_spin_absolute.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling kinematic round-trip for a SPIN pin op about an
# ABSOLUTE world-space pivot, world_scaling=10. The pivot is a
# world-space length and is scaled by the factor on ingest (scene.rs
# scales the Spin center). If that center scaling were missing, the
# rotation would pivot about the wrong (un-scaled) point in sim space
# and the un-scaled output would diverge from the reference. This rig
# locks in that the spin pivot scales with the geometry.

from __future__ import annotations

from . import _world_scaling_kinematic as _ws
from . import _runner as r


NEEDS_BLENDER = True


CASE = {
    "name": "ws_spin_absolute_x10",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "world_scaling": 10.0,
    "tolerance": 1e-3,
    "ops": [
        {"type": "SPIN",
         "axis": (0.0, 0.0, 1.0), "angular_velocity": 180.0,
         "center_mode": "ABSOLUTE", "center": (1.0, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx: r.ScenarioContext) -> str:
    return _ws.build_driver(CASE, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return _ws.run(ctx, CASE)
