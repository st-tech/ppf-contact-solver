# File: scenarios/rig_force_field_laws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The field-object laws `rig_force_field_spatial` does not reach, held to the
# reference evaluator in the kernel: Point Vortex, Tube, Z Direction Negative,
# Turbulence in Global Coordinates, the image Texture in its three modes and
# both samplings and in Blender's axes, Magnetic from a known velocity, the blend
# of two track samples (origin and rotation), a Wind's gusts drifting with
# the flow, and Gravitation failing the run at its origin.
# The cases and the reasoning behind each are in `_force_field_probe.py`.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "laws", "force field laws")
