# File: scenarios/bl_contact_gap_unsolvable.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A contact gap the step size cannot solve is refused at Transfer, with the
# two things that fix it, instead of ending the run in its first frames with
# `cg failed`.
#
# The add-on's default takes the gap as a fraction of each object's own
# bounding-box diagonal, so a small object, or a ratio lowered while hunting
# for some other failure, puts the gap in micrometers or below (public issue
# #156 arrived with 86 nanometers). The session build compares the smallest
# gap a pair can engage at with `step / 2^12` (`frontend/_contact_gap_.py`).
#
# Two rigid cubes 2 cm on a side, diagonal 3.46 cm. Three builds:
#   `A_small_ratio_refused_by_name`: ratio 1e-5 is a gap of 3.5e-7 against a
#       bound of 2.4e-6 at dt = 0.01. Refused, naming both objects, the gap,
#       and both remedies.
#   `B_smaller_step_builds`: the same gap at dt = 0.001, where the bound is
#       2.4e-7. READY.
#   `C_default_ratio_builds`: dt back at 0.01 and the ratio at its default
#       1e-3, a gap of 3.5e-5. READY.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it against a CPU build.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT = "contact_gap_unsolvable"
SIDE = 0.02


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for name, x in (("Left", -0.05), ("Right", 0.05)):
        bpy.ops.mesh.primitive_cube_add(size=SIDE, location=(x, 0.0, 0.0))
        bpy.context.active_object.name = name

    root = dh.configure_state(project_name=PROJECT, frame_count=10,
                              frame_rate=60, step_size=0.01)
    root.state.disable_contact = False
    cubes = dh.api.solver.create_group("Cubes", "PDRD")
    cubes.add("Left", "Right")
    cubes.param.use_group_bounding_box_diagonal = True
    diagonal = SIDE * 3 ** 0.5

    def build(message, first=False):
        if first:
            dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                       project_name=PROJECT)
        return dh.build_verdict(message, timeout=120.0)

    # ----- A: a ratio that puts the gap below the bound ---------------
    cubes.param.contact_gap_rat = 1e-5
    gap = 1e-5 * diagonal
    verdict, error = build("small ratio", first=True)
    dh.record("A_small_ratio_refused_by_name",
              verdict == "FAILED"
              and "The contact gap between 'Left' and 'Right' is" in error
              and "Raise the contact gap to at least 2.441e-06" in error
              and "lower dt to at most" in error,
              {"solver": verdict, "gap": gap, "bound": 0.01 / 4096.0,
               "error": error[:700]})

    # ----- B: the same gap at a step that can solve it -----------------
    root.state.step_size = 0.001
    verdict, error = build("smaller step")
    dh.record("B_smaller_step_builds", verdict == "READY",
              {"solver": verdict, "gap": gap, "bound": 0.001 / 4096.0,
               "error": error[:700]})

    # ----- C: the default ratio at the default step --------------------
    root.state.step_size = 0.01
    cubes.param.contact_gap_rat = 1e-3
    verdict, error = build("default ratio")
    dh.record("C_default_ratio_builds", verdict == "READY",
              {"solver": verdict, "gap": 1e-3 * diagonal,
               "bound": 0.01 / 4096.0, "error": error[:700]})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
