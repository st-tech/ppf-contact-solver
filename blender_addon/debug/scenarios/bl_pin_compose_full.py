# Composed pin op: MOVE_BY + SPIN + SCALE in that order, with mixed
# center modes. Exercises the full kinematic stack.

from __future__ import annotations
from . import _pin_fidelity_common as _common
from . import _runner as r

NEEDS_BLENDER = True

CASE = {
    "name": "fidelity_compose_full",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "MOVE_BY", "delta": (0.3, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
        {"type": "SPIN",
         "axis": (0.0, 0.0, 1.0), "angular_velocity": 60.0,
         "center_mode": "CENTROID",
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
        {"type": "SCALE", "factor": 1.2,
         "center_mode": "ABSOLUTE", "center": (0.0, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
