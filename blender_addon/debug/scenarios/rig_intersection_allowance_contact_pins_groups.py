# File: scenarios/rig_intersection_allowance_contact_pins_groups.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# An allowed pair is NOT A CONTACT PAIR, for a pin's allowance and the
# inter-group one. Same drop shape as `rig_intersection_allowance_contact`;
# the probe is in `_allowance_contact.py`.
#
# Subtests (each case also requires the run to reach its last frame):
#   pin_allowance_same_object_passes       one object holding both sheets, its
#                                          pin carries allow_intersection and
#                                          no flag is set.
#   pull_pin_allowance_passes              the held sheet hangs on a PULL pin
#                                          that carries allow_intersection.
#   pull_pin_unflagged_is_held             the same pull pin, no allowance.
#   inter_group_falling_side_passes        two objects in DIFFERENT groups,
#                                          the inter-group flag on the falling
#                                          sheet.
#   inter_group_held_side_passes           the same, the flag on the held
#                                          sheet only: either side opts in.
#   inter_group_same_group_is_held         two objects in ONE group, both
#                                          flagged: they still collide.
#   inter_group_static_collider_passes     the held sheet is a STATIC
#                                          collision mesh, which is another
#                                          group from every object.
#   pin_allowance_covers_only_held_faces   the held sheet's left half is
#                                          pinned by an allowing pin and its
#                                          right half by a plain one; a sheet
#                                          dropped over the left passes, one
#                                          over the right is held.
#   pin_split_control_is_held              the same two pins, neither
#                                          allowing: both dropped sheets held.

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
    return ac.run_section(ctx, "pins_groups", "allowance pin and group cases")
