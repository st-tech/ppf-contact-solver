# File: scenarios/bl_world_scaling_multi_group.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling SCALE-INVARIANCE with MULTIPLE GROUPS in one scene. Two
# independent sheets in two separate SHELL groups drape under gravity in
# the same build. We run the scene at base size (world_scaling=1) and at
# 10x size (world_scaling=0.1) and assert the tracked sheet's 10x run
# reproduces 10x its base run. This confirms world_scaling applies
# uniformly across a multi-group scene (every group's geometry scaled by
# the same factor), not just a single solved object.

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
FRAME_COUNT = 14


def _sheet(name, cx, scale, group_name):
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=6, y_subdivisions=6, size=2.0 * scale,
        location=(cx * scale, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = name
    pinned = [i for i, v in enumerate(sheet.data.vertices)
              if (sheet.matrix_world @ v.co).y > 0.99 * scale]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    grp = dh.api.solver.create_group(group_name, "SHELL")
    grp.add(sheet.name)
    grp.create_pin(sheet.name, "TopEdge")
    return sheet.name


def build(scale):
    # Two sheets, two groups, offset in x so they don't overlap. We track
    # the first one; the second's presence exercises multi-group encode +
    # build under world_scaling.
    tracked = _sheet("WsMultiSheetA", -2.0, scale, "ClothA")
    _sheet("WsMultiSheetB", 2.0, scale, "ClothB")
    return tracked


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_multi_group",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
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
