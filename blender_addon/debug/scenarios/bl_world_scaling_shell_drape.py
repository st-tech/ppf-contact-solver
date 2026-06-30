# File: scenarios/bl_world_scaling_shell_drape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling SCALE-INVARIANCE for a free-vertex SHELL drape. A grid
# pinned along its +y edge sags under gravity (free verts deform, the
# pinned edge holds) -- exactly the bl_emulated_elastic_drape scene. We
# run it once at base size (world_scaling=1) and once at 10x size
# (world_scaling=0.1) and assert the 10x run reproduces 10x the base
# run's per-frame positions. This is the core promise of world_scaling
# for deformable geometry: an over-sized cloth simulated at a sane scale
# reproduces, size-scaled, the physically authored drape -- including
# the relative bbox-diagonal contact gaps, which scale with the mesh.

from __future__ import annotations

from . import _world_scaling_invariance as _inv
from . import _runner as r


NEEDS_BLENDER = True

# Free vertices need the emulator's implicit ARAP step to actually
# deform; step pacing 0 ms keeps the unit run fast.
KNOBS = {"PPF_EMULATED_ELASTIC": "1", "PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 15


def build(scale):
    # A grid pinned along its +y edge; size=2*scale spans [-scale, scale]
    # in x and y, so every authored length is exactly ``scale``-scaled.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=8, y_subdivisions=8, size=2.0 * scale,
        location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "WsDrapeSheet"
    edge = 0.99 * scale
    pinned = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > edge]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")
    return sheet.name


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_shell_drape",
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
