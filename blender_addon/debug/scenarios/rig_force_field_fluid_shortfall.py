# File: scenarios/rig_force_field_fluid_shortfall.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# THE OPEN DEFECT OF A NEGATIVE FLUID COUPLING, AS A KNOWN FAILURE. This
# scenario FAILS until the defect is fixed, and is ON DEMAND so the default
# run stays a statement about regressions rather than about known debt; run
# it by name to read the current shortfall.
#
# The defect: a Fluid Flow with a negative Strength or Flow has a negative
# damping coefficient `s W rho flow`, whose Hessian the solver leaves out
# (an indefinite block), applying the gradient alone.
# The Newton loop ends a step on its TOI target after two linear solves, not
# on the residual, and two solves without that Hessian leave about
# 0.54 |s flow dt| of the push unsolved (the constant depends on the scene:
# the miss is a^2 of the step's whole displacement, a = s flow dt). The
# cases `rig_force_field_fluid` runs at a negative coefficient keep
# |s flow dt| at 0.01, where the miss is under its 2e-2 tolerance; this one
# runs where users go.
#
# Cases, each one solver step of the probe's soft sheet through the fluid
# section's rotated volume (`_force_field_probe.py`, section
# "fluid_shortfall"), against the backward Euler reference at the 2e-2 of
# the push every other fluid case is held to:
#   * fluid_strength_minus_8_lands_on_backward_euler: Strength -8, Flow 1
#     (s flow dt = -0.08). FAILS while the defect is open.
#   * fluid_flow_minus_1_at_strength_8_lands_on_backward_euler: Strength 8,
#     Flow -1. FAILS while the defect is open.
#   * each `..._converged_control_...` twin: the same step with
#     `min-newton-steps` 6, which must land within 2e-2. It passes today, and
#     it is what says the gradient is right and the miss is the early stop.
#
# MEASURED on dev-head (CPU backend, 2026-10-05), relative error of the
# step against the reference: Strength -8 4.3e-2 (2.1x the tolerance),
# Flow -1 at Strength 8 3.0e-2 (1.5x); their converged controls 3.1e-5 and
# 1.9e-5, 650x and 1000x under it. The same mechanism measured elsewhere:
# Strength -4 2.2e-2, Strength -1 5.5e-3, Flow -1 at Strength 1 3.7e-3.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True
ON_DEMAND = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "fluid_shortfall", "force field fluid shortfall")
