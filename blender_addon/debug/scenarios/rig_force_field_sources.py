# File: scenarios/rig_force_field_sources.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The external force field's two sources against plain gravity: a uniform
# acceleration grid and a constant script each reproduce it, and a
# `force-field-weight` of 0 opts an object out.
# The cases and the reasoning behind each are in `_force_field_probe.py`.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "sources", "force field sources")
