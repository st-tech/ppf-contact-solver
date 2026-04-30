# File: scenarios/bl_pin_animation_fidelity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MOVE_BY pin op fidelity. The flagship case from the original
# fidelity test; kept under this name for backwards compatibility.

from __future__ import annotations

from . import _pin_fidelity_common as _common
from . import _runner as r


NEEDS_BLENDER = True


CASE = {
    "name": "fidelity_move_by",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "MOVE_BY", "delta": (0.5, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx: r.ScenarioContext) -> str:
    return _common.build_driver(CASE, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return _common.run(ctx, CASE)
