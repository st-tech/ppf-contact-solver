# File: scenarios/bl_intersection_allowance_matrix_static_anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The intersection allowance matrix, HELD object an ANIMATED STATIC collider:
# a 1 m grid lying flat at z = 0 in a STATIC group, moved 1 mm over the clip by
# a MOVE_BY static operation, under a SHELL, SOLID, ROD, SAND or PDRD object
# dropped from 0.1 m above it, with an invisible floor at z = -0.3.
#
# A moving STATIC decodes to a pin shell in the solved namespace, so its OWN
# flags do apply, unlike a rest-pose one. For each falling type (see
# `_allowance_matrix.pair_cells`):
#
#   a  no flag                              -> held
#   b  inter-object on the falling group    -> passes
#   c  inter-object on the STATIC group     -> passes
#   d  inter-group on the falling group     -> passes
#   e  inter-group on the STATIC group      -> passes
#   f  self on both groups                  -> held (the wrong flag)
#
# There is no g cell: a STATIC group carries Transform operations, not pins,
# and the pin encoder skips STATIC groups, so there is no pin item to flag.
#
# The scene, the verdict and the one-cell knob are described in
# `_allowance_matrix.py`.

from __future__ import annotations

from . import _allowance_matrix as am
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it on the CPU backend. It
# drives real builds and real solves and reads the simulated positions.
BACKENDS = ("real",)

# Every cell starts a solver process, so it should not share a host with
# another scenario's solver: `frontend.Utils.busy()` scans every process.
NOT_PARALLELIZABLE = True
# Runs only when named (see `_on_demand` in `__init__.py`): the full matrix
# costs about 16 minutes on the CPU backend and about half an hour across the
# Windows shards, and the default set samples it through
# `bl_intersection_allowance_matrix_same_group` and the
# `rig_intersection_allowance_contact*` scenarios. Run the whole family before
# changing the allowance rule or the contact filter.
ON_DEMAND = True

CELLS = am.pair_cells("STATIC_ANIM")


def build_driver(ctx: r.ScenarioContext) -> str:
    return am.build_driver(CELLS, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return am.run(ctx, CELLS)
