# File: scenarios/rig_force_field_spatial.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A field object acts by POSITION and by TIME, exactly: the kernel matches the
# reference evaluator for four laws, for a field far from the scene and
# under World Scaling; a moving origin reverses the motion; a Wind drives the
# drag only when the air has density; a falloff that does not reach leaves a
# sheet still.
# The cases and the reasoning behind each are in `_force_field_probe.py`.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "spatial", "force field objects")
