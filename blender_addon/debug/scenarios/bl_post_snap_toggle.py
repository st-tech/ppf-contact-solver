# File: scenarios/bl_post_snap_toggle.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# "Post Snap Exactly" toggle, verified through the addon fetch path.
#
# Both live preview and Fetch All Animation go through one closure point:
# client.apply_animation -> _apply_single_frame -> _apply_post_snap_closure,
# gated on state.post_snap_exactly (client.py: "One call handles both live
# simulation and batch fetch"). So driving the fetch path with the toggle
# on and then off exercises exactly the gating both surfaces share.
#
# Two SHELL strips are stitched A.v0 -> B.v0 with a rest gap of 0.5m:
#
#     A.v3 (-1,1,0)        A.v1 (0,1,0)  B.v1 (0.5,1,0)        B.v3 (1.5,1,0)
#         +------------------+  gap=0.5   +----------------------+
#         |  MeshA           |            |  MeshB               |
#         +------------------+            +----------------------+
#     A.v2 (-1,0,0)        A.v0 (0,0,0)  B.v0 (0.5,0,0)        B.v2 (1.5,0,0)
#                            ^---STITCH---^
#
# The closure is an addon-side reconstruction applied on top of whatever
# positions the solver returns (it reads the cross_stitch ind/w the same
# way for every endpoint type), so it needs no real stitch physics from
# the emulated solver: the toggle is the only thing that can move A.v0
# onto B.v0. We fetch once with the toggle on, read the PC2 it wrote,
# then clear + re-fetch with the toggle off and read again.
#
# Subtests:
#   A. post_snap_on_closes_seam: with post_snap_exactly on, every
#      simulated PC2 frame has |A[0] - B[0]| < TOL_ON (seam joined).
#   B. post_snap_off_keeps_gap: with post_snap_exactly off, the re-fetched
#      PC2 keeps the raw seam, |A[0] - B[0]| > TOL_OFF on every frame
#      (closure skipped, so the ~0.5m rest gap survives).
#   C. simulation_completes: solver did not fail and both PC2 files were
#      written with matching, multi-frame sample counts.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import json
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>

FRAME_COUNT = 5
REST_GAP = 0.5


def _make_strip(name, inner_x, outer_x):
    # 1m x 1m quad (two triangles) with a stable vertex order:
    #   v0 = (inner_x, 0, 0)  inner edge, stitched to partner
    #   v1 = (inner_x, 1, 0)  inner edge (top)
    #   v2 = (outer_x, 0, 0)  outer edge
    #   v3 = (outer_x, 1, 0)  outer edge (top)
    import bmesh
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    coords = [
        (inner_x, 0.0, 0.0),
        (inner_x, 1.0, 0.0),
        (outer_x, 0.0, 0.0),
        (outer_x, 1.0, 0.0),
    ]
    verts = [bm.verts.new(c) for c in coords]
    bm.verts.ensure_lookup_table()
    bm.faces.new((verts[0], verts[2], verts[3]))
    bm.faces.new((verts[0], verts[3], verts[1]))
    bm.to_mesh(mesh)
    bm.free()
    return obj


def _gap_series(arr_a, arr_b):
    # World distance between A[0] and B[0] per simulated frame. Sample 0
    # is the Blender rest geometry written before any closure, so we skip
    # it and compare from the first simulated frame onward.
    out = []
    n = min(arr_a.shape[0], arr_b.shape[0])
    for f in range(1, n):
        d = float(np.linalg.norm(
            np.asarray(arr_a[f, 0], dtype=np.float64)
            - np.asarray(arr_b[f, 0], dtype=np.float64)))
        out.append(round(d, 6))
    return out


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    mesh_a = _make_strip("MeshA", inner_x=0.0, outer_x=-1.0)
    mesh_b = _make_strip("MeshB", inner_x=0.5, outer_x=1.5)
    dh.save_blend(PROBE_DIR, "post_snap_toggle.blend")

    root = dh.configure_state(project_name="post_snap_toggle",
                              frame_count=FRAME_COUNT)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(mesh_a.name)
    cloth.add(mesh_b.name)

    # Resolve UUIDs (the encoder rejects empty source/target uuids).
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    uuid_a = uuid_mod.get_or_create_object_uuid(mesh_a)
    uuid_b = uuid_mod.get_or_create_object_uuid(mesh_b)
    if not uuid_a or not uuid_b:
        raise RuntimeError("could not allocate UUIDs for stitch meshes")

    # One shell-shell stitch A.v0 -> B.v0. 6-wide barycentric-barycentric:
    # degenerate shell source [0, 0, 0] / [1, 0, 0] (A.v0) and target
    # weights [1, 0, 0] over B's triangle (verts 0, 2, 3) place the
    # barycentric target on B.v0.
    state = dh.groups.get_addon_data(bpy.context.scene).state
    pair = state.merge_pairs.add()
    pair.object_a = mesh_a.name
    pair.object_b = mesh_b.name
    pair.object_a_uuid = uuid_a
    pair.object_b_uuid = uuid_b
    pair.stitch_stiffness = 50.0
    cs_payload = {
        "source_uuid": uuid_a,
        "target_uuid": uuid_b,
        "ind": [[0, 0, 0, 0, 2, 3]],
        "w": [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        "source_points": [[0.0, 0.0, 0.0]],
        "target_points": [[0.5, 0.0, 0.0]],
        "a_vert_count": len(mesh_a.data.vertices),
        "b_vert_count": len(mesh_b.data.vertices),
    }
    pair.cross_stitch_json = json.dumps(cs_payload, separators=(",", ":"))
    state.merge_pairs_index = len(state.merge_pairs) - 1
    dh.log(f"authored_stitch pairs={len(state.merge_pairs)}")

    # Build + run once; the solver frames are reused by both fetches.
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="post_snap:build",
                      timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=root.state.frame_count - 1,
                         timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.log(f"ran solver={dh.facade.engine.state.solver.name}")

    animation_mod = __import__(pkg + ".core.animation",
                               fromlist=["clear_animation_data"])

    # ----- Fetch with post_snap_exactly ON -----
    state.post_snap_exactly = True
    dh.fetch_and_drain()
    pc2_a = dh.find_pc2_for(mesh_a)
    pc2_b = dh.find_pc2_for(mesh_b)
    arr_a_on = dh.read_pc2(pc2_a) if pc2_a and os.path.isfile(pc2_a) else None
    arr_b_on = dh.read_pc2(pc2_b) if pc2_b and os.path.isfile(pc2_b) else None
    gaps_on = (_gap_series(arr_a_on, arr_b_on)
               if arr_a_on is not None and arr_b_on is not None else [])
    samples_on = (
        (arr_a_on.shape[0] if arr_a_on is not None else 0),
        (arr_b_on.shape[0] if arr_b_on is not None else 0),
    )
    dh.log(f"fetch_on gaps={gaps_on}")

    # ----- Clear + re-fetch with post_snap_exactly OFF -----
    animation_mod.clear_animation_data(bpy.context)
    state.post_snap_exactly = False
    dh.fetch_and_drain()
    pc2_a2 = dh.find_pc2_for(mesh_a)
    pc2_b2 = dh.find_pc2_for(mesh_b)
    arr_a_off = dh.read_pc2(pc2_a2) if pc2_a2 and os.path.isfile(pc2_a2) else None
    arr_b_off = dh.read_pc2(pc2_b2) if pc2_b2 and os.path.isfile(pc2_b2) else None
    gaps_off = (_gap_series(arr_a_off, arr_b_off)
                if arr_a_off is not None and arr_b_off is not None else [])
    dh.log(f"fetch_off gaps={gaps_off}")

    TOL_ON = 0.01
    # Rest gap is 0.5m; closure-skipped must be unmistakably large.
    TOL_OFF = 0.1

    dh.record(
        "A_post_snap_on_closes_seam",
        len(gaps_on) >= 2 and max(gaps_on) < TOL_ON,
        {
            "tolerance": TOL_ON,
            "max_gap_on": max(gaps_on) if gaps_on else None,
            "n_frames": len(gaps_on),
            "per_frame": gaps_on,
        },
    )
    dh.record(
        "B_post_snap_off_keeps_gap",
        len(gaps_off) >= 2 and min(gaps_off) > TOL_OFF,
        {
            "tolerance": TOL_OFF,
            "min_gap_off": min(gaps_off) if gaps_off else None,
            "rest_gap": REST_GAP,
            "n_frames": len(gaps_off),
            "per_frame": gaps_off,
        },
    )
    dh.record(
        "C_simulation_completes",
        dh.facade.engine.state.solver.name in ("READY", "RESUMABLE")
        and samples_on[0] >= 2 and samples_on[0] == samples_on[1],
        {
            "solver": dh.facade.engine.state.solver.name,
            "samples_on": samples_on,
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = REPO_ROOT_POSIX
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
