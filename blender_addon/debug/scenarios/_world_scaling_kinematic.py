# File: scenarios/_world_scaling_kinematic.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared driver + host-side diff for the kinematic world_scaling
# round-trip scenarios (Design 1).
#
# These reuse the per-pin-op fidelity harness in
# ``_pin_fidelity_common``: a fully pinned plane is driven by one
# kinematic pin op, the emulated solver writes the per-frame PC2, and
# the host diff compares the pinned-vertex trajectory against
# ``frontend.FixedScene.time(t)`` -- the source of truth behind
# ``frontend.preview()``.
#
# The ONLY thing these add on top of the fidelity harness is setting
# ``state.world_scaling`` before encode. world_scaling is implemented
# entirely inside the Rust solver: ``scene.rs`` multiplies all input
# geometry (and the pin op deltas / centers) by the factor on ingest,
# and ``backend.rs`` divides the per-frame output positions back by it
# on write. The Python ``frontend`` reconstruction is scale-agnostic
# (it never reads "world-scaling"), so ``fixed.time(t)`` already yields
# the authored-scale trajectory. Because the solver round-trips back to
# authored scale too, the emulated PC2 must match ``fixed.time(t)``
# directly with NO rescaling of either side -- to within the float
# scale/unscale round-off. That equality, holding for world_scaling
# both above (10) and below (0.1) 1.0, is exactly what these rigs lock
# in across the pin-op matrix (static, move_by, spin, scale).
#
# A CASE here is a plain ``_pin_fidelity_common`` CASE plus an extra
# ``"world_scaling"`` key (read by the injected driver line). Extra keys
# are ignored by the fidelity driver, so the rest of the schema and the
# host diff are reused unchanged.

from __future__ import annotations

from . import _pin_fidelity_common as _common
from . import _runner as r


NEEDS_BLENDER = True

# The fidelity driver sets the standard rig defaults right before encode;
# this is the last of those lines and a unique anchor. We splice the
# world_scaling assignment in directly after it so the encoder picks the
# factor up. ``CASE`` is already in scope inside the driver (the fidelity
# template inlines it as JSON), so we read the factor straight from it.
_ANCHOR = "    root.state.wind_strength = 0.0\n"
_INJECT = (
    _ANCHOR
    + '    root.state.world_scaling = float(CASE.get("world_scaling", 1.0))\n'
    + '    log(f"world_scaling={root.state.world_scaling}")\n'
)


def build_driver(case: dict, ctx: r.ScenarioContext) -> str:
    src = _common.build_driver(case, ctx)
    if _ANCHOR not in src:
        raise RuntimeError(
            "world_scaling injection anchor missing from fidelity driver "
            "template; _pin_fidelity_common may have changed"
        )
    return src.replace(_ANCHOR, _INJECT, 1)


def run(ctx: r.ScenarioContext, case: dict) -> dict:
    # The host diff is identical to the fidelity path: it reconstructs the
    # expected trajectory from the (scale-agnostic) frontend and compares
    # it to the emulated PC2. No world_scaling handling is needed here.
    return _common.run(ctx, case)
