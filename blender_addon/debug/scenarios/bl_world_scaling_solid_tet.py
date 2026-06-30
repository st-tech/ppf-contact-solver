# File: scenarios/bl_world_scaling_solid_tet.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling SCALE-INVARIANCE for a SOLID (tetrahedralized) body. A
# closed cube is assigned to a SOLID group, tetrahedralized by the
# frontend build, fully pinned, and driven by a MOVE_BY pin op. We run
# it at base size (world_scaling=1) and at 10x size (world_scaling=0.1,
# with a 10x-larger authored delta) and assert the 10x run reproduces
# 10x the base run's per-frame positions. This exercises the solid mesh
# + tet rest geometry round-trip (scene.rs scales the rest vertices and
# the pin delta) on top of the surface mesh round-trip.
#
# A kinematic (fully pinned) drive is used rather than a gravity drop
# because the CUDA-free emulator's implicit ARAP step does not deform
# SOLID tets under gravity (only kinematic pins move them), so a free
# solid would not evolve and the invariance check would be trivial.

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


def build(scale):
    # A unit cube scaled to edge length 2*scale, fully pinned and driven
    # by a MOVE_BY whose delta scales with the scene.
    bpy.ops.mesh.primitive_cube_add(size=2.0 * scale, location=(0, 0, 0))
    cube = bpy.context.object
    cube.name = "WsSolidCube"
    n = len(cube.data.vertices)
    vg = cube.vertex_groups.new(name="AllPin")
    vg.add(list(range(n)), 1.0, "REPLACE")
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)
    pin = solid.create_pin(cube.name, "AllPin")
    pin.move_by(delta=(0.3 * scale, 0.1 * scale, 0.0),
                frame_start=1, frame_end=8, transition="LINEAR")
    return cube.name


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_solid_tet",
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
