# File: scenarios/rig_existing_intersection_run.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What the solver does with a scene that starts tangled under Allow Existing
# Intersections, in one solve: the tangled pair runs with no NEW penetration
# (every intersecting pair at every sampled frame is exempt by the links the
# build made) and the part of the sheet outside the linked region still rests
# on the sheet it would otherwise fall through, against a held sheet and
# against a STATIC collision mesh. The reasoning is in
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
    return ei.run_section(ctx, "run", "existing intersection run cases")
