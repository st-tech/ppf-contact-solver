# File: scenarios/bl_intersection_allowance_matrix_self.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The intersection allowance matrix, SELF layout: ONE object holding a pinned
# lower part (the held shape, top at z = 0) and a free upper part (the falling
# shape, 0.1 m above), for SHELL, SOLID, ROD and SAND, with an invisible floor
# at z = -0.3. Only "Allow Self-Intersections" names a pair inside one object:
#
#   allow_self_passes           -> the upper part passes through the lower
#   no_flag                     -> held
#   inter_object_only_is_held   -> held
#   inter_group_only_is_held    -> held
#
# PDRD is not here: a PDRD body is rigid and its self-contact is always
# excluded, so a self allowance has nothing to change.
#
# The two parts are disconnected components of one mesh. Each cell records
# the PC2's vertex count against the mesh's, so a component dropped on the way
# to the solver fails the cell rather than reading as a pass.
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

CELLS = (
    am.self_cells("SHELL")
    + am.self_cells("SOLID")
    + am.self_cells("ROD")
    + am.self_cells("SAND")
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return am.build_driver(CELLS, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return am.run(ctx, CELLS)
