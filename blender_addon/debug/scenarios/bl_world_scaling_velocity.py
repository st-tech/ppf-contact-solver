# File: scenarios/bl_world_scaling_velocity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling round-trip for an INITIAL VELOCITY launch -- and a
# DETECTOR for a confirmed double-scaling bug in the feature.
#
# A free (unpinned) shell is given a frame-1 initial translational
# velocity and drifts with zero gravity. Velocity has units length/time,
# so for the authored motion to survive the scale-out it must be scaled
# by world_scaling exactly ONCE. We run the scene at base size + base
# speed (world_scaling=1) and at 10x size + 10x speed (world_scaling=0.1)
# and assert the 10x run reproduces 10x the base run.
#
# This currently FAILS its C_scale_invariant check: the frame-1 initial
# velocity is scaled TWICE -- once in the addon encoder
# (core/encoder/params.py: ``speed * state.world_scaling`` in the
# "velocity" dict) and again in the solver (scene.rs scales the vel.bin
# matrix by ws). The net ws^2 factor shrinks the seeded motion of the
# 10x run by 1/ws, so it does not reproduce 10x the base drift. The fix
# is to scale the initial velocity in exactly one place. (The velocity
# SCHEDULE path is scaled only once -- see
# bl_world_scaling_velocity_schedule, which passes.)
#
# Kept RED on purpose as a regression guard: it must turn GREEN once the
# double-scaling is removed.
#
# (The emulator integrates injected velocity only with its implicit
# ARAP step enabled, hence PPF_EMULATED_ELASTIC=1.)

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
FRAME_COUNT = 12
BASE_SPEED = 2.0


def build(scale):
    # A free (unpinned) sheet launched along +x. Every authored length
    # scales by ``scale``: the sheet size AND the launch speed (a
    # length/time, so it scales like a length for the analogous scene).
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=6, y_subdivisions=6, size=2.0 * scale,
        location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "WsVelSheet"
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.set_velocity(sheet.name, direction=(1.0, 0.0, 0.0),
                       speed=BASE_SPEED * scale, frame=1)
    return sheet.name


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_velocity",
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
