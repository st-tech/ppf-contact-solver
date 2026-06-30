# File: scenarios/bl_world_scaling_rod.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling SCALE-INVARIANCE for a ROD (edge-chain). A straight
# edge-chain rod is fully pinned and driven by a MOVE_BY pin op. We run
# it at base size (world_scaling=1) and at 10x size (world_scaling=0.1,
# 10x-larger authored delta) and assert the 10x run reproduces 10x the
# base run's per-frame positions. This covers the 1D rod geometry
# round-trip (vertex chain + rest lengths + pin delta) under
# world_scaling.
#
# A kinematic (fully pinned) drive is used rather than a gravity droop
# because the CUDA-free emulator's implicit ARAP step does not deform
# ROD chains under gravity (only kinematic pins move them).

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
N_VERT = 11


def build(scale):
    # A straight rod of N_VERT vertices along +x, length 1*scale, fully
    # pinned and driven by a MOVE_BY whose delta scales with the scene.
    m = bpy.data.meshes.new("WsRodData")
    step = (1.0 * scale) / (N_VERT - 1)
    verts = [(i * step, 0.0, 0.0) for i in range(N_VERT)]
    edges = [(i, i + 1) for i in range(N_VERT - 1)]
    m.from_pydata(verts, edges, [])
    m.update()
    rod = bpy.data.objects.new("WsRod", m)
    bpy.context.collection.objects.link(rod)
    vg = rod.vertex_groups.new(name="AllPin")
    vg.add(list(range(N_VERT)), 1.0, "REPLACE")
    grp = dh.api.solver.create_group("Rod", "ROD")
    grp.add(rod.name)
    pin = grp.create_pin(rod.name, "AllPin")
    pin.move_by(delta=(0.0, 0.0, 0.3 * scale),
                frame_start=1, frame_end=8, transition="LINEAR")
    return rod.name


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    ws_invariance(
        dh, build,
        project_name="ws_rod",
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
