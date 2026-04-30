# SCALE with MAX_TOWARDS center.

from __future__ import annotations
from . import _pin_fidelity_common as _common
from . import _runner as r

NEEDS_BLENDER = True

CASE = {
    "name": "fidelity_scale_max_towards_x",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "SCALE", "factor": 0.7,
         "center_mode": "MAX_TOWARDS",
         "center_direction": (1.0, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
