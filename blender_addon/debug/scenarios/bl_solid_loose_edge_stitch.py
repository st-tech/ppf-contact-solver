# File: scenarios/bl_solid_loose_edge_stitch.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A SOLID's own loose-edge stitch reaches the solver, placed on the
# tetrahedralized surface, and pulls.
#
# The encoder ships a loose edge (an edge on no face) as a stitch row naming
# BLENDER vertices. A SOLID is re-tetrahedralized at build, so those indices
# name unrelated tetrahedral vertices, or none: used verbatim, the seam
# pulled on the wrong vertices, and an index past the tet mesh panicked the
# scene assembly. The decoder places both ends of each edge on the tet
# surface by position (``SceneDecoder._solid_stitch_rows``, covered
# point-for-point by ``frontend/tests/_decoder_cross_stitch_.py``); this is
# the end-to-end counterpart through a real fTetWild resample and a run.
#
# The object is two cubes, one pinned, joined by four loose edges across a
# gap, with gravity on. A third cube, unstitched and unpinned, falls beside
# them as the control. A stitch pulls its two points together, so the free
# cube closes the gap toward the pinned one while the control only falls.
#
# Subtests:
#   A. loose_edges_encoded: the DATA payload carries the object's four loose
#      edges as 4-column stitch rows.
#   B. stitch_builds_and_runs: the tetrahedralized object builds with its
#      stitch and the solve runs to the end with finite output.
#   C. stitch_pulls_the_islands_together: the free cube moves toward the
#      pinned one by more than half the gap, the control cube does not move
#      sideways, and the pinned cube stays put.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it: a full build and solve.
BACKENDS = ("real",)


_FRAME_COUNT = 12


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>

H = 0.25    # half the cube size
GAP = 0.1   # gap the loose edges span


def cube_verts(cx):
    return [(cx + sx * H, sy * H, sz * H)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def cube_faces(base):
    # Vertex k = (sx, sy, sz) bits (4*ix + 2*iy + iz); outward windings.
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [tuple(base + i for i in q) for q in quads]


try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Twin: cube A at x = 0 (pinned) and cube B at x = 2H + GAP, joined by
    # a loose edge from each vertex of A's +X face to the matching vertex of
    # B's -X face.
    verts = cube_verts(0.0) + cube_verts(2.0 * H + GAP)
    faces = cube_faces(0) + cube_faces(8)
    a_plus_x = [k for k in range(8) if verts[k][0] > 0.0]
    b_minus_x = [8 + k for k in range(8) if verts[8 + k][0] < 2.0 * H + GAP]
    edges = [(a, b) for a in a_plus_x for b in b_minus_x
             if abs(verts[a][1] - verts[b][1]) < 1e-9
             and abs(verts[a][2] - verts[b][2]) < 1e-9]
    mesh = bpy.data.meshes.new("TwinMesh")
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    twin = bpy.data.objects.new("Twin", mesh)
    bpy.context.collection.objects.link(twin)
    anchor = twin.vertex_groups.new(name="Anchor")
    anchor.add(list(range(8)), 1.0, "REPLACE")

    control_mesh = bpy.data.meshes.new("ControlMesh")
    control_mesh.from_pydata(cube_verts(0.0), [], cube_faces(0))
    control_mesh.update()
    control = bpy.data.objects.new("Control", control_mesh)
    control.location = (0.0, 3.0, 0.0)
    bpy.context.collection.objects.link(control)

    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(os.path.dirname(PROBE_DIR), "solid_loose_edge_stitch.blend"))
    root = dh.configure_state(
        project_name="solid_loose_edge_stitch", frame_count=FRAME_COUNT,
        frame_rate=100, step_size=0.01, gravity=(0.0, 0.0, -9.8),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(twin.name)
    solid.add(control.name)
    solid.create_pin(twin.name, "Anchor")
    # A stitch is a raw force factor that saturates at its length cap, about
    # nine times the stiffness per row; 50 closes the gap within the run.
    # Contact is off in this rig state, so the free cube may pass the pinned
    # one's face: only the closing is asserted.
    solid.param.stitch_stiffness = 50.0

    # ----- A: the loose edges ship as stitch rows ---------------------------
    data_bytes, param_bytes = dh.encode_payload()
    data = dh.decode_addon_blob(data_bytes)
    rows = None
    for group in data:
        for info in group.get("object", []):
            if info.get("name") == "Twin":
                rows = info.get("stitch")
    ind = np.asarray(rows[0]) if rows else np.zeros((0, 4))
    dh.record(
        "A_loose_edges_encoded",
        len(edges) == 4 and ind.shape == (4, 4)
        and sorted(map(tuple, ind[:, :2].tolist())) == sorted(edges),
        {"edges": edges, "ind": ind.tolist()},
    )

    # ----- B: build and run -------------------------------------------------
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes,
                      message="solid_loose_edge_stitch:build", timeout=360.0)
    dh.run_and_wait(timeout=300.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    twin_pc2 = dh.find_pc2_for(twin)
    control_pc2 = dh.find_pc2_for(control)
    twin_pos = dh.read_pc2(twin_pc2) if twin_pc2 else None
    control_pos = dh.read_pc2(control_pc2) if control_pc2 else None
    ran = (
        solver_state != "FAILED"
        and twin_pos is not None and control_pos is not None
        and twin_pos.shape[0] >= FRAME_COUNT - 1
        and bool(np.all(np.isfinite(twin_pos)))
        and bool(np.all(np.isfinite(control_pos)))
    )
    dh.record(
        "B_stitch_builds_and_runs", ran,
        {"solver_state": solver_state, "twin_pc2": twin_pc2,
         "control_pc2": control_pc2,
         "frames": None if twin_pos is None else int(twin_pos.shape[0]),
         "error": dh.facade.engine.state.error},
    )

    # ----- C: the stitch closes the gap -------------------------------------
    details = {}
    holds = False
    if ran:
        def drop(pos, idx):
            return float(pos[0, idx, 2].mean() - pos[-1, idx, 2].mean())
        def closing(pos, idx):
            return float(pos[0, idx, 0].mean() - pos[-1, idx, 0].mean())
        pinned_drop = drop(twin_pos, list(range(8)))
        hanging_drop = drop(twin_pos, list(range(8, 16)))
        control_drop = drop(control_pos, list(range(8)))
        hanging_closing = closing(twin_pos, list(range(8, 16)))
        control_closing = closing(control_pos, list(range(8)))
        details = {"pinned_drop": pinned_drop, "hanging_drop": hanging_drop,
                   "control_drop": control_drop,
                   "hanging_closing": hanging_closing,
                   "control_closing": control_closing}
        holds = (
            control_drop > 0.01
            and hanging_closing > 0.5 * GAP
            and abs(control_closing) < 1e-4
            and abs(pinned_drop) < 5e-3
        )
    dh.record("C_stitch_pulls_the_islands_together", holds, details)
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
