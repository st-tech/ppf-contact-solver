# File: scenarios/bl_solid_solid_stitch.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# SOLID-to-SOLID cross-stitch end to end.
#
# Two SOLID cubes sit face to face with a 0.02m gap. Each is
# re-tetrahedralized by fTetWild at build, so its Blender surface vertex
# indices do NOT map to the solver tet mesh. The cross-stitch is authored
# with the snap-time 6-wide barycentric-barycentric layout plus per-row
# source_points / target_points (the world anchors). The PyO3 decoder
# (cross_stitch_apply_batch) must re-project BOTH endpoints onto their own
# tet surfaces: source -> slots 0..2, target -> slots 3..5. Before the
# 6-slot migration the source side was left pinned to a mis-mapped tet
# vertex and the solver generated ghost forces; the gap-correctness of the
# projection is covered deterministically by
# tests/test_cross_stitch_apply_batch.py.
#
# This scenario is the integration counterpart: it drives a REAL fTetWild
# resample through encode -> tetrahedralize -> cross_stitch_apply_batch ->
# assemble -> 6-wide bin write -> solver load, which exercises the full ABI
# path (the 6-wide stitch_ind.bin / stitch_w.bin must match the solver's
# 6-wide reader, or the build/run fails here).
#
# Subtests:
#   A. solid_stitch_encoded:
#         encode_param -> decode the blob; the cross_stitch section holds a
#         single entry for the authored SOLID->SOLID pair whose ``ind``
#         rows are 6-wide and that carries ``source_points`` (proves the
#         new format flows snap_ops -> _encode_cross_stitch, including the
#         source-anchor channel the source projection needs).
#   B. solid_stitch_build_runs:
#         the two tetrahedralized SOLIDs build (the source/target
#         projection runs against real tet surfaces and writes 6-wide
#         stitch bins), the solver loads the 6-wide stitch and runs without
#         FAILED, and both cubes produce finite PC2s. A 4-vs-6 ABI stride
#         bug would surface here as a build/run failure or non-finite PC2.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 4
_GAP = 0.02  # face-to-face gap between the two cubes (meters)


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
FRAME_COUNT = <<FRAME_COUNT>>
GAP = <<GAP>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Fresh scene: two unit cubes face to face on +X with a small gap.
    # Cube A spans x in [-1, 1]; cube B spans x in [1+GAP, 3+GAP].
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube_a = bpy.context.active_object
    cube_a.name = "SolidA"
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(2.0 + GAP, 0.0, 0.0))
    cube_b = bpy.context.active_object
    cube_b.name = "SolidB"
    dh.log(f"cubes a={len(cube_a.data.vertices)} b={len(cube_b.data.vertices)}")

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "solid_solid_stitch.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    root = dh.configure_state(
        project_name="solid_solid_stitch",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, 0.0),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    # Both cubes in a single SOLID group.
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube_a.name)
    solid.add(cube_b.name)

    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    uuid_a = uuid_mod.get_or_create_object_uuid(cube_a)
    uuid_b = uuid_mod.get_or_create_object_uuid(cube_b)
    if not uuid_a or not uuid_b:
        raise RuntimeError("could not allocate UUIDs for solid cubes")

    # Author a SOLID->SOLID cross-stitch: each vertex on A's +X face
    # (world x == 1) ties to the matching point on B's -X face (world
    # x == 1+GAP). The 6-wide baseline ind/w are overwritten by the PyO3
    # source/target re-projection; source_points / target_points are the
    # world anchors that projection consumes.
    a_face = [cube_a.matrix_world @ v.co
              for v in cube_a.data.vertices if abs((cube_a.matrix_world @ v.co).x - 1.0) < 1e-4]
    source_points = [[float(p.x), float(p.y), float(p.z)] for p in a_face]
    target_points = [[1.0 + GAP, float(p.y), float(p.z)] for p in a_face]
    n_pair = len(source_points)
    ind = [[0, 0, 0, 0, 1, 2] for _ in range(n_pair)]
    w = [[1.0, 0.0, 0.0, 0.34, 0.33, 0.33] for _ in range(n_pair)]
    dh.log(f"stitch_pairs={n_pair}")

    state = dh.groups.get_addon_data(bpy.context.scene).state
    pair = state.merge_pairs.add()
    pair.object_a = cube_a.name
    pair.object_b = cube_b.name
    pair.object_a_uuid = uuid_a
    pair.object_b_uuid = uuid_b
    pair.stitch_stiffness = 1.0
    cs_payload = {
        "source_uuid": uuid_a,
        "target_uuid": uuid_b,
        "ind": ind,
        "w": w,
        "source_points": source_points,
        "target_points": target_points,
        "a_vert_count": len(cube_a.data.vertices),
        "b_vert_count": len(cube_b.data.vertices),
    }
    pair.cross_stitch_json = json.dumps(cs_payload, separators=(",", ":"))
    state.merge_pairs_index = len(state.merge_pairs) - 1
    dh.log(f"authored_stitch pairs={len(state.merge_pairs)}")

    # ----- A: SOLID->SOLID stitch encodes 6-wide with source_points -----
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    decoded_cs = decoded.get("cross_stitch", [])
    entry = decoded_cs[0] if decoded_cs else None
    ind_widths = []
    if entry is not None:
        ind_widths = sorted({len(row) for row in entry.get("ind", [])})
    encoder_ok = (
        len(decoded_cs) == 1
        and entry.get("source_uuid") == uuid_a
        and entry.get("target_uuid") == uuid_b
        and ind_widths == [6]
        and bool(entry.get("source_points"))
        and len(entry.get("source_points", [])) == n_pair
    )
    dh.record(
        "A_solid_stitch_encoded",
        encoder_ok,
        {
            "n_entries": len(decoded_cs),
            "ind_widths": ind_widths,
            "has_source_points": bool(entry.get("source_points")) if entry else False,
            "n_source_points": len(entry.get("source_points", [])) if entry else 0,
            "n_pair": n_pair,
        },
    )

    # ----- B: build + run completes through the 6-wide ABI path ----------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    # fTetWild resamples both cubes during build; cross_stitch_apply_batch
    # re-projects both stitch endpoints onto the resampled tet surfaces.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="solid_solid_stitch:build", timeout=360.0)
    dh.log("built")
    dh.run_and_wait(timeout=180.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.log(f"ran solver={solver_state}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    pc2_a = dh.find_pc2_for(cube_a)
    pc2_b = dh.find_pc2_for(cube_b)
    arr_a = dh.read_pc2(pc2_a) if pc2_a else None
    arr_b = dh.read_pc2(pc2_b) if pc2_b else None
    finite_a = bool(arr_a is not None and np.all(np.isfinite(arr_a)))
    finite_b = bool(arr_b is not None and np.all(np.isfinite(arr_b)))
    samples_a = int(arr_a.shape[0]) if arr_a is not None else 0
    samples_b = int(arr_b.shape[0]) if arr_b is not None else 0
    dh.record(
        "B_solid_stitch_build_runs",
        solver_state != "FAILED"
        and finite_a and finite_b
        and samples_a >= FRAME_COUNT - 1
        and samples_b >= FRAME_COUNT - 1,
        {
            "solver_state": solver_state,
            "pc2_a": pc2_a,
            "pc2_b": pc2_b,
            "samples_a": samples_a,
            "samples_b": samples_b,
            "finite_a": finite_a,
            "finite_b": finite_b,
            "error": dh.facade.engine.state.error,
        },
    )

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
        .replace("<<GAP>>", repr(_GAP))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 480.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
