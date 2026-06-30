# File: scenarios/bl_world_scaling_sand.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling SCALE-INVARIANCE for the SAND granular type. A closed
# mesh is seeded with grains, assigned to a SAND group, and the cloud
# falls under gravity (the emulator auto-drifts a faceless point cloud).
# We run the same grain cloud at base size (world_scaling=1) and at 10x
# size (world_scaling=0.1) and assert the 10x run reproduces 10x the
# base run's per-frame grain positions. This covers the grain seed
# position round-trip (the grain radius / contact skin scaling lives in
# the encoder and is checked separately; the emulator has no contact
# pipeline so radius does not affect the drift output here).
#
# Grains are ALWAYS seeded from the same base geometry with a fixed RNG
# seed, then their coordinates are scaled by the cycle's factor, so both
# cycles share an identical grain count and arrangement (scaling the
# seeding domain directly is not guaranteed to be RNG-equivariant).

from __future__ import annotations

from . import _world_scaling_invariance as _inv
from . import _runner as r


NEEDS_BLENDER = True

KNOBS = {"PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 12
GRAIN_RADIUS = 0.3   # coarse fill -> a few dozen grains, fast


def build(scale):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0,
                                           location=(0, 0, 0))
    src = bpy.context.object
    src.name = "WsSandBall"
    sand_ops = __import__(pkg + ".ui.dynamics.sand_ops",
                          fromlist=["build_and_commit_particle_mesh"])
    # Seed from the fixed base geometry (radius 1) with a fixed RNG seed so
    # the grain set is identical across cycles, then scale the committed
    # particle positions by ``scale`` for a clean scale-equivariant cloud.
    sand_ops.build_and_commit_particle_mesh(src, GRAIN_RADIUS, 0.0, rng_seed=0)
    if scale != 1.0:
        for v in src.data.vertices:
            v.co = (v.co.x * scale, v.co.y * scale, v.co.z * scale)
        src.data.update()
    sand = dh.api.solver.create_group("Sand", "SAND")
    sand.add(src.name)
    return src.name


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_sand",
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
