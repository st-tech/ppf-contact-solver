# File: scenarios/bl_world_scaling_velocity_schedule.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling SCALE-INVARIANCE for a VELOCITY SCHEDULE (frame > 1
# keyframes). A free shell is static until a mid-run keyframe (frame 4)
# sets a translational velocity, then a second (frame 8) overwrites it.
# The encoder emits these as "velocity-schedule" entries, scaling each
# authored speed by world_scaling; the solver consumes the schedule as
# a dyn_param that is NOT re-scaled in scene.rs, so the schedule is
# correctly scaled exactly once. We run the scene at base size + base
# speeds (world_scaling=1) and at 10x size + 10x speeds
# (world_scaling=0.1) and assert the 10x run reproduces 10x the base
# trajectory -- a time-varying velocity schedule survives the round-trip
# at the authored scale.
#
# NOTE: this deliberately uses ONLY frame > 1 keyframes. The frame-1
# INITIAL velocity (vel.bin) is double-scaled by the current feature
# (encoder * ws AND scene.rs * ws); bl_world_scaling_velocity is the
# dedicated detector for that bug.

from __future__ import annotations

from . import _world_scaling_invariance as _inv
from . import _runner as r


NEEDS_BLENDER = True

KNOBS = {"PPF_EMULATED_ELASTIC": "1", "PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 16
SPEED_1 = 2.0
SPEED_2 = 3.0


def build(scale):
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=6, y_subdivisions=6, size=2.0 * scale,
        location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "WsVelSchedSheet"
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    # Two mid-run keyframes (frame > 1) -> "velocity-schedule" only; no
    # frame-1 initial velocity. Both speeds scale with the scene.
    cloth.set_velocity(sheet.name, direction=(1.0, 0.0, 0.0),
                       speed=SPEED_1 * scale, frame=4)
    cloth.set_velocity(sheet.name, direction=(0.0, 1.0, 0.0),
                       speed=SPEED_2 * scale, frame=9)
    return sheet.name


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_velocity_schedule",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, 0.0),
        contact=False,
        base_scale=1.0,
        ratio=10.0,
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        rel_tol=1e-2,
        result=result,
    )
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _inv.build_driver(_DRIVER_BODY, ctx)


def run(ctx: r.ScenarioContext) -> dict:
    return _inv.run(ctx)
