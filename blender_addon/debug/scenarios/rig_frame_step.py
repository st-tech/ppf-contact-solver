# File: scenarios/rig_frame_step.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The frame-step API (issue #151): run_until_frame holds a live solver,
# step_frame advances it, a field updated while held takes effect, and
# save_and_quit while held checkpoints a run that resumes with the update.
# The cases and the reasoning behind each are in `_force_field_probe.py`.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "frame_step", "frame stepping")
