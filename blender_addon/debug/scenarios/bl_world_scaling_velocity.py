# File: scenarios/bl_world_scaling_velocity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling round-trip for an INITIAL VELOCITY launch.
#
# A free (unpinned) shell is given a frame-1 initial translational
# velocity and drifts with zero gravity. Velocity has units length/time
# and is authored in Blender units, so world_scaling must stay transparent
# to the look: the seeded motion has to be scaled by world_scaling exactly
# ONCE (the solver scales vel.bin on ingest, like geometry, and divides the
# output back out). We run the scene at base size + base speed
# (world_scaling=1) and at 10x size + 10x speed (world_scaling=0.1) and
# assert the 10x run reproduces 10x the base run.
#
# History: this once double-scaled the frame-1 velocity (encoder * ws AND
# solver * ws, a net ws^2), fixed in c8a61856 by scaling only in the solver
# so the encoder now passes the raw Blender-unit speed. (The velocity
# SCHEDULE path is instead scaled once in the encoder and left alone by the
# solver; see bl_world_scaling_velocity_schedule.) Because this is a single
# frame-1 impulse followed by pure drift, the test also guards that the
# emulator carries momentum across steps: a7ffd916's frame-time
# interpolation zeroed it until the emulator curr->prev snapshot was gated
# off the elastic path.
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
