# File: scenarios/rig_force_field_spatial.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A grid acts by POSITION and by TIME: two halves of a box push two sheets
# apart, a sheet outside the box is unforced and the solver logs it, two time
# samples reverse the motion, and an air-velocity grid drives the drag only
# when the air has density.
# The cases and the reasoning behind each are in `_force_field_probe.py`.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "spatial", "force field in space and time")
