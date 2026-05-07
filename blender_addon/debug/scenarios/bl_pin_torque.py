# TORQUE: a force op, not kinematic. In emulated Rust mode (no
# dynamics, only kinematic constraints) the torque-pinned vertices
# remain at rest. ``frontend.FixedScene.time(t)`` likewise returns
# rest positions for torque pins (TorqueOperation.apply is a no-op
# at the frontend layer because torque is force-driven), so PC2 and
# fixed.time agree at zero displacement.
#
# On a real CUDA build the torque actually induces motion (the
# integrator applies the force; the implicit step solves it through),
# so PC2 diverges from fixed.time by however far the torque moved
# the verts. The frontend has no force-side reference to compare
# against, so we just bound the divergence: a magnitude=1.0 torque
# applied for 0.04 s on a 1x1 plane drifts on the order of 1e-3 m,
# well under 1e-2. Tightening the tolerance here would amount to
# asserting a specific solver-side numerical scheme, which we don't
# want to lock in.

from __future__ import annotations
from . import _pin_fidelity_common as _common

NEEDS_BLENDER = True

CASE = {
    "name": "fidelity_torque",
    "frame_count": 10, "frame_rate": 100, "step_size": 0.01,
    "ops": [
        {"type": "TORQUE", "magnitude": 1.0, "axis_component": "PC3",
         "frame_start": 1, "frame_end": 4},
    ],
    "tolerance": 1e-2,
}


def build_driver(ctx): return _common.build_driver(CASE, ctx)
def run(ctx): return _common.run(ctx, CASE)
