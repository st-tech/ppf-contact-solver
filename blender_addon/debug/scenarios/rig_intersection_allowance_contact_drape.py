# File: scenarios/rig_intersection_allowance_contact_drape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The report that prompted the allowance change: a cloth hung vertically over
# an invisible floor and let fall. With `allow-self-intersection` its folds
# cross one another; without it no sampled frame has a self-intersecting
# triangle pair. The probe, and why the allowed case reads the peak over the
# sampled frames, are in `_allowance_contact.py`.
#
# Subtests:
#   drape_allowed_crosses_itself       the dropped cloth with allow-self:
#                                      self-intersections in some sampled
#                                      frame, and nothing below the floor.
#   drape_unflagged_never_intersects   the dropped cloth, no flag: no
#                                      self-intersection in any sampled frame.

from __future__ import annotations

from . import _allowance_contact as ac
from . import _runner as r


# RUNS ON THE REAL BACKEND. What it asserts is the contact filter in the
# neutral kernels, which every backend renders from the same body.
BACKENDS = ("real",)
# Runs a solver process per case, so it should not share a worker with another
# solver.
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return ac.run_section(ctx, "drape", "allowance drape cases")
