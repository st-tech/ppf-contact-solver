# File: scenarios/rig_intersection_allowance_contact.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# An allowed pair is NOT A CONTACT PAIR, for the self and inter-object
# allowances and the static collision mesh: a small sheet dropped onto a held
# one falls through it to an invisible floor where the pair is allowed, and
# rests on it where it is not. The probe and the reasoning behind its shape are
# in `_allowance_contact.py`; `rig_intersection_allowance_contact_pins_groups`
# and `rig_intersection_allowance_contact_drape` run its other two sections.
#
# Subtests (each case also requires the run to reach its last frame):
#   self_allowed_passes_through            one object holding both sheets,
#                                          allow-self on it.
#   self_unflagged_is_held                 the same object, no flag.
#   self_inter_object_flag_does_not_cover  the same object, only the
#                                          inter-object flag.
#   inter_object_falling_side_passes       two objects, the flag on the
#                                          falling sheet.
#   inter_object_held_side_passes          two objects, the flag on the held
#                                          sheet only: either side opts in.
#   inter_object_unflagged_is_held         two objects, no flag.
#   inter_object_self_flags_do_not_cover   two objects, allow-self on both.
#   pin_allowance_passes                   the held sheet's pin carries
#                                          allow_intersection.
#   static_collider_flagged_passes         the held sheet is a STATIC
#                                          collision mesh, the falling sheet
#                                          carries the inter-object flag.
#   static_collider_unflagged_is_held      the same collider, no flag.

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
    return ac.run_section(ctx, "pairs", "allowance contact cases")
