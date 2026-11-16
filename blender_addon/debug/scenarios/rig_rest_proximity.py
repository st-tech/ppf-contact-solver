# File: scenarios/rig_rest_proximity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Allow Rest Proximity through the frontend: which distances link which
# starting pairs at the scene build, and, in one solve, that a linked sheet
# falls through its held sheet while an unlinked one comes to rest on it. The
# cases are listed in the "rest" section of `_existing_intersection.py`.

from __future__ import annotations

from . import _existing_intersection as ei
from . import _runner as r


# RUNS ON THE REAL BACKEND, like the other existing-intersection scenarios.
BACKENDS = ("real",)
# Runs a solver process, so it should not share a worker with another solver.
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return ei.run_section(ctx, "rest", "allow rest proximity cases")
