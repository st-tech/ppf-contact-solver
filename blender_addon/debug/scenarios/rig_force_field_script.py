# File: scenarios/rig_force_field_script.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The exact script: a vortex turns a sheet, the compiler refuses unsupported
# constructs by line, the solver's loader refuses bytecode that did not come
# from the compiler, and a script that yields NaN fails the run.
# The cases and the reasoning behind each are in `_force_field_probe.py`.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "script", "force field script")
