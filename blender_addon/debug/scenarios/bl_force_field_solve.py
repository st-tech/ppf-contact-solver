# File: scenarios/bl_force_field_solve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Blender force fields and the exact script move simulated cloth end to end
# (issue #114): encode -> build -> solve -> fetch through a real server and
# solver, judged by the motion in the fetched PC2s.
#
# Three sheets side by side, gravity off:
#
#   * LEFT, over a Force field (Point, pushing away) that reaches it only:
#     it must rise.
#   * RIGHT, over a Wind field blowing up (+Z) that reaches it only, with Air
#     Density 1: the drag must lift it.
#   * MIDDLE, in a group whose Force Field Weight is 0: it must not move at
#     all, which is also what separates "the field acts" from "something
#     moved".
#
# An exact script adds +2 m/s^2 along Blender +Y everywhere. It reaches the
# solver through the server's compile with z_up, so the two outer sheets
# drifting along Blender +Y (not solver +Y, which is Blender +Z) is what
# proves the script's axes were converted.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_FRAME_COUNT = 20

_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>

try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    def sheet(name, x):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6,
                                        size=1.0, location=(x, 0.0, 0.0))
        o = bpy.context.object
        o.name = name
        return o

    left = sheet("FFLeft", -1.6)
    middle = sheet("FFMiddle", 0.0)
    right = sheet("FFRight", 1.6)
    dh.save_blend(PROBE_DIR, "force_field_solve.blend")
    root = dh.configure_state(project_name="force_field_solve",
                              frame_count=FRAME_COUNT)
    state = root.state
    state.air_density = 1.0

    moving = dh.api.solver.create_group("Moving", "SHELL")
    moving.add(left.name)
    moving.add(right.name)
    still = dh.api.solver.create_group("Still", "SHELL")
    still.add(middle.name)
    groups = __import__(pkg + ".models.groups", fromlist=["iterate_object_groups"])
    for g in groups.iterate_object_groups(bpy.context.scene):
        if g.name == "Still":
            g.force_field_weight = 0.0

    def field(kind, name, loc):
        bpy.ops.object.effector_add(type=kind, location=loc)
        o = bpy.context.object
        o.name = name
        o.field.use_max_distance = True
        o.field.distance_max = 1.0
        return o

    push = field("FORCE", "FFPush", (-1.6, 0.0, -0.5))
    push.field.strength = 6.0
    wind = field("WIND", "FFUpdraft", (1.6, 0.0, -0.5))
    wind.field.strength = 6.0

    text = bpy.data.texts.new("ff_drift.py")
    text.from_string("def eval(x, y, z, t):\n    return (0.0, 2.0, 0.0)\n")
    state.force_field_script = text
    state.force_field_spacing = 0.1
    state.force_field_time_samples = 2

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="force_field_solve:build",
                      timeout=300.0)
    dh.run_and_wait(timeout=300.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    def travel(obj):
        path = dh.find_pc2_for(obj)
        arr = dh.read_pc2(path) if path else None
        if arr is None or arr.shape[0] < 3 or not np.all(np.isfinite(arr)):
            return None
        # Row 0 is the display gap-fill, so travel is measured from the first
        # solver row.
        return (arr[-1] - arr[1]).mean(axis=0), float(np.abs(arr[-1] - arr[1]).max())

    got = {o.name: travel(o) for o in (left, middle, right)}
    detail = {k: (None if v is None else {"mean": [round(float(x), 5) for x in v[0]],
                                          "max_abs": round(v[1], 7)})
              for k, v in got.items()}
    detail["solver_state"] = solver_state
    detail["error"] = dh.facade.engine.state.error
    L, M, R = got["FFLeft"], got["FFMiddle"], got["FFRight"]
    dh.record("A_force_field_pushes_its_sheet_up",
              L is not None and L[0][2] > 0.01, detail)
    dh.record("B_wind_field_lifts_through_the_drag",
              R is not None and R[0][2] > 0.005, detail)
    dh.record("C_script_drifts_along_blender_y",
              L is not None and R is not None and L[0][1] > 0.005
              and R[0][1] > 0.005, detail)
    dh.record("D_weight_zero_group_stays_put",
              M is not None and M[1] < 1e-6, detail)

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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
