# Composed pin op: MOVE_BY then SPIN. Order matters because the spin's
# CENTROID is recomputed from the moved positions.

from __future__ import annotations
from . import _pin_fidelity_common as _common
from . import _runner as r

NEEDS_BLENDER = True

CASE = {
    "name": "fidelity_compose_move_spin",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "MOVE_BY", "delta": (0.5, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
        {"type": "SPIN",
         "axis": (0.0, 0.0, 1.0), "angular_velocity": 90.0,
         "center_mode": "CENTROID",
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
