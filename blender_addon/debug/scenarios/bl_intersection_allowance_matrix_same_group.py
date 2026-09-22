# File: scenarios/bl_intersection_allowance_matrix_same_group.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The intersection allowance matrix, SAME-GROUP and NARROWED layouts.
#
# Same group: a held object and a falling object of one type in ONE group, for
# every type that can be both held and dropped (SHELL, SOLID, ROD, SAND,
# PDRD). Inter-group covers two objects of DIFFERENT groups only, so it must
# leave this pair colliding, while inter-object lets it through:
#
#   no_flag                     -> held
#   inter_group_still_collides  -> held
#   inter_object_passes         -> passes
#
# Narrowed: a held SHELL and a falling SHELL in two groups, one of which also
# holds a far-away bystander. That group's "Allow Inter-Group Intersections"
# is ON with "Apply to All Objects" OFF, and its list names either the object
# in the pair (-> passes) or the bystander (-> held), on the falling side and
# on the held side.
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

CELLS = (
    am.same_group_cells("SHELL")
    + am.same_group_cells("SOLID")
    + am.same_group_cells("ROD")
    + am.same_group_cells("SAND")
    + am.same_group_cells("PDRD")
    + am.narrowed_cells()
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return am.build_driver(CELLS, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return am.run(ctx, CELLS)
