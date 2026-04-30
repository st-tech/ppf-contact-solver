# SCALE around the pinned-vertex CENTROID. Shrink to 0.5 over the range.

from __future__ import annotations
from . import _pin_fidelity_common as _common
from . import _runner as r

NEEDS_BLENDER = True

CASE = {
    "name": "fidelity_scale_centroid",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "SCALE", "factor": 0.5, "center_mode": "CENTROID",
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
