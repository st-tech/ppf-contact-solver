# File: scenarios/bl_world_scaling_colliders.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling with INVISIBLE COLLIDERS present (an invisible wall and
# an invisible sphere), contact enabled. A pinned sheet drapes under
# gravity above a wall and a sphere; every collider length (wall
# position, sphere position + radius, and the sphere/wall thickness) is
# authored at the cycle's scale. We run the scene at base size
# (world_scaling=1) and at 10x size (world_scaling=0.1) and assert the
# 10x run reproduces 10x the base run's per-frame positions.
#
# WHAT THIS LOCKS IN is the scene.rs ingest that scales wall position,
# sphere position, sphere radius and collider thickness: the two runs differ
# in world_scaling alone, so any length the ingest scales by the wrong power
# breaks the 10x correspondence rather than merely moving the cloth. The
# assertion is that correspondence and not a particular deflection, which is
# what makes it a check of the scaling and not of the contact response. The
# encoder-side relative-vs-absolute gap scaling is checked separately, in
# bl_world_scaling_encoder_scales.

from __future__ import annotations

from . import _world_scaling_invariance as _inv
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it: it passes a
# real-backend run unchanged.
BACKENDS = ("real",)

# This scenario carries no pacing or elasticity knobs: a real backend has
# no artificial per-step sleep and always computes real elasticity, so the
# intent is preserved by asking for neither.


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 14


def build(scale):
    # Pinned sheet that drapes toward a wall + sphere below it. Every
    # collider length scales with the scene.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=8, y_subdivisions=8, size=2.0 * scale,
        location=(0, 0, 1.0 * scale),
    )
    sheet = bpy.context.object
    sheet.name = "WsColliderSheet"
    pinned = [i for i, v in enumerate(sheet.data.vertices)
              if (sheet.matrix_world @ v.co).y > 0.99 * scale]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    # A floor wall and a sphere obstacle, both authored at the scene scale.
    dh.api.solver.add_wall(position=(0.0, 0.0, -1.0 * scale),
                           normal=(0.0, 0.0, 1.0))
    dh.api.solver.add_sphere(position=(0.0, 0.0, -0.5 * scale),
                             radius=0.4 * scale)
    return sheet.name


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_colliders",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
        contact=True,
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
