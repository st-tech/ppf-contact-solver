# File: scenarios/rig_force_field_fluid.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The Fluid field held to the reference evaluator in the
# kernel, through the "fluid" section of `_force_field_probe.py`, which
# states the reasoning behind each case:
#
#   * one step against `step_reference`: the relative coupling over a
#     rotated, offset, partly occupied volume with density and falloff;
#     Flow 0; a cell per axis; a negative Strength and a negative Flow
#     (gradient only); Playback Speed 0.5 and 2, the fluid's velocity
#     divided by it; World Scaling 2 from rest and 0.5 moving; a sparse
#     volume whose kept bricks are renumbered beside FLUID_EMPTY ones; two
#     fluids beside a Force field, with a twin lacking the second; a blend
#     into an empty frame;
#   * every step of a run, from the solver's own state at its start: a
#     per-frame layout (brick counts, corner, axes, cell and density all
#     differing) read in both slot orders; a volume that starts after the
#     run does; a volume finer than the step, so a step replaces both
#     resident frames; an empty frame in the middle of a sequence;
#   * the volume read at the step's end, against the implicit step reading
#     it at the start; a strong coupling streamed through the two slots;
#   * a held run against an unheld one; the field replaced while held by a
#     fluid of eight times the bricks; a save and resume; the session saved
#     at build, recovered by name after the update, rebuilding with the
#     fluid it was built with;
#   * controls: a sheet outside the domain and one 0.02 outside the scaled
#     domain coast; a malformed frame, and a frame file grown under a held
#     run, each stop the run by name.
#
# MEASURED MARGINS (dev-head, CPU backend, 2026-10-05), each against the
# tolerance the probe applies:
#
#   * A step against the reference passes within 2e-2 of the push the
#     reference gives it. Worst measured: 0.55% (Strength -1 at Flow 1),
#     0.38% (Playback Speed 2, a step twice as long in motion), 0.37%
#     (Strength 1 at Flow -1), 0.33% (sparse), 0.27% (cell per axis),
#     0.24% (per-frame layout, worst of six steps), 0.20% (relative
#     coupling, and World Scaling 0.5), 0.18% (after resume); every other
#     case below 0.13%. The push is at least 2.4e-4 against its 1e-5 floor.
#     What fills the margin: the f32 positions and output (about 1e-7), the
#     1e-3 sheet's elasticity, the solver's step at 0.9999 of a frame with
#     frames interpolated from inside the next step (about 1e-4 of a step's
#     motion per frame), and at a negative coefficient the two linear
#     solves without its Hessian, 0.54 |strength flow dt| of the push
#     (4.3% at Strength -8, which is why the negative cases keep
#     |strength flow dt| at 0.01; the on-demand known failure
#     `rig_force_field_fluid_shortfall` runs Strength -8).
#   * A coasting step, with no fluid present, moves by its velocity to
#     1e-5: measured 4.8e-7, mostly the next step's push entering the
#     interpolated frame; a fluid present moves it by about 1e-3.
#   * A sheet outside a domain drifts less than 1e-6: measured 1.7e-7 (the
#     scaled domain) and 2.8e-7.
#   * Separation guards, the distance from the reference the case must
#     match to the one a defect would give, over the push: the start read
#     5.0 (needs 0.5; the solver lands 1e5 times nearer the end read than
#     the start read, needs 10); a start ignored 1.25 (needs 0.3); the
#     second fluid's share 0.81 (needs 0.2); the old field after the update
#     0.84 (needs 0.5); a velocity left undivided by Playback Speed 0.55 at
#     0.5 and 0.93 at 2 (needs 0.3).
#   * The held run differs from the unheld one by at most 1e-5 of the
#     displacement: measured 0, the CPU backend repeating the run exactly
#     (a 1% change of Strength measures 2.8e-4 of it).
#   * The strong coupling's velocity stays within 0.05 of the fluid's:
#     measured 0.010.
#   * The cases of several steps keep every vertex 0.25 cells or more from
#     every face they read: measured 1.32 and 1.48.
#
# Each check was seen to fail on dev-head under a mutation of what it
# covers: in the solver, the corner or the velocity left unscaled by World
# Scaling, every fluid's slots at one base, a frame's words taken from the
# slot of its place in the bracket, the volume's start ignored, a held
# run's update ignored, the re-read brick count check dropped, the volume
# read at the step's start, the FLUID_EMPTY test dropped, the table's y
# stride taken from the z brick count, a negative damping applied as
# positive, the velocity left undivided by Playback Speed (both playback
# cases, the solver landing 1.2e-4 from the undivided reference at 0.5); in
# the frontend, an update of a held run that does not save the built scene
# again (the rebuild raises, naming the frame and its overwritten file); in
# the probe, the two frames of the start-read case made equal,
# the sparse kept bricks moved off the second row of y bricks, a layout
# frame shrunk onto the sheet, and the held run's Strength changed by 1%.

from __future__ import annotations

from . import _force_field_probe as probe
from . import _runner as r


BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


def run(ctx: r.ScenarioContext) -> dict:
    return probe.run_section(ctx, "fluid", "force field fluid")
