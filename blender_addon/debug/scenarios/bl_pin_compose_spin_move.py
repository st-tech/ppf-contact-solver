# Same ops as bl_pin_compose_move_spin but in REVERSE order: SPIN then
# MOVE_BY. The spin pivot is the original centroid; then everything is
# translated. Different result from move-then-spin, exercising op-order
# fidelity end-to-end.

from __future__ import annotations
from . import _pin_fidelity_common as _common
from . import _runner as r

NEEDS_BLENDER = True

CASE = {
    "name": "fidelity_compose_spin_move",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "SPIN",
         "axis": (0.0, 0.0, 1.0), "angular_velocity": 90.0,
         "center_mode": "CENTROID",
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
        {"type": "MOVE_BY", "delta": (0.5, 0.0, 0.0),
         "frame_start": 1, "frame_end": 4, "transition": "LINEAR"},
    ],
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
