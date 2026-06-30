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
# IMPORTANT (emulated-solver limitation): the CUDA-free emulator has NO
# contact / BVH pipeline (cpp_emul/main.cpp: "No contact assembly
# happens in the emulator"), so the colliders do not deflect the cloth
# here. What this rig DOES lock in is that world_scaling != 1.0 with
# colliders present builds and runs without panic -- it drives the
# scene.rs ingest that scales wall position, sphere position, sphere
# radius, and collider thickness -- and that the solved geometry still
# round-trips at the authored scale. The contact RESPONSE to a scaled
# collider is GPU-only and is verified there (commit 2ada05b5); the
# encoder-side relative-vs-absolute gap scaling is checked in
# bl_world_scaling_encoder_scales.

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
