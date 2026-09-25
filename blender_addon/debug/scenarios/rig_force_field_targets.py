# File: scenarios/rig_force_field_targets.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A force field source reaches every object, or only the groups it names:
# group-targeted grids and scripts move only their groups, an ungrouped
# object gets only the untargeted source, and an unknown group is refused.
# The cases and the reasoning behind each are in `_force_field_probe.py`.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "targets", "force field targets")
