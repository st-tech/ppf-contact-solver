# File: scenarios/rig_existing_intersection_build.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Which scenes Allow Existing Intersections lets build and which it refuses,
# with no solve. The cases and their controls are listed in
# `_existing_intersection.py`.

from __future__ import annotations

from . import _existing_intersection as ei
from . import _runner as r


# RUNS ON THE REAL BACKEND. The link test is in the neutral contact kernels,
# which every backend renders from the same body.
BACKENDS = ("real",)
# Runs a solver process, so it should not share a worker with another solver.
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return ei.run_section(ctx, "build", "existing intersection build cases")
