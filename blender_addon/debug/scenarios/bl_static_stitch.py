# File: scenarios/bl_static_stitch.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Dynamic-to-STATIC cross-stitch end to end (the {SOLID,SHELL,ROD}-STATIC
# pipeline). This headline scenario exercises the hardest path: a dynamic
# SOLID stitched to a REST-POSE STATIC collider.
#
# A STATIC has two internal forms. An animated static is already a dynamic
# all-pinned pin-shell. A rest-pose static (no animation) is otherwise a
# DISJOINT contact-only collision mesh that the stitch index space (which
# addresses only the dynamic vertex namespace) cannot reach. So a rest-pose
# STATIC that appears as a stitch endpoint is PROMOTED at decode time into
# the dynamic all-pinned namespace (frontend _populate_static, gated on the
# pre-scanned cross_stitch endpoint UUIDs), with every vertex held by an
# immovable fixed pin so it stays kinematically frozen at its rest pose.
#
# The SOLID source is re-tetrahedralized by fTetWild, so cross_stitch_apply_batch
# re-projects its source endpoint onto its tet surface (source -> slots 0..2)
# via source_points. The STATIC target is SHELL-like (never re-tetrahedralized,
# 1:1 indices), so type="STATIC" != "SOLID" makes the decoder keep its target
# slots (3..5) verbatim. The unit-level projection/passthrough is covered by
# tests/test_cross_stitch_apply_batch.py; this is the integration counterpart.
#
# Subtests:
#   A. static_stitch_encoded:
#         encode_param -> decode the blob; the cross_stitch section holds a
#         single entry for the authored SOLID->STATIC pair whose ``ind`` rows
#         are 6-wide and that carries ``source_points`` (the source-anchor
#         channel the SOLID source re-projection needs).
#   B. static_stitch_build_runs_and_freezes:
#         the SOLID tetrahedralizes and the rest-pose STATIC is promoted into
#         the dynamic namespace; the solver loads the 6-wide stitch and runs
#         without FAILED; BOTH objects produce finite PC2s (the STATIC having
#         an output PC2 proves it was promoted, not left as a disjoint
#         collider); and the promoted STATIC stays ~frozen (immovable fixed
#         pins) while the SOLID is free to be pulled toward it.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 4
_GAP = 0.02  # face-to-face gap between the SOLID and the STATIC (meters)
_FREEZE_TOL = 0.05  # max per-frame motion allowed for the frozen STATIC (m)


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
FREEZE_TOL = <<FREEZE_TOL>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Fresh scene: a SOLID cube and a STATIC cube face to face on +X with a
    # small gap. SOLID A spans x in [-1, 1]; STATIC B spans x in [1+GAP, 3+GAP].
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube_a = bpy.context.active_object
    cube_a.name = "SolidA"
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(2.0 + GAP, 0.0, 0.0))
    cube_b = bpy.context.active_object
    cube_b.name = "StaticB"
    dh.log(f"cubes a={len(cube_a.data.vertices)} b={len(cube_b.data.vertices)}")

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "static_stitch.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    root = dh.configure_state(
        project_name="static_stitch",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, 0.0),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    # cube_a is a dynamic SOLID; cube_b is a rest-pose STATIC collider (no
    # animation, no ops) -> Case 4. Because cube_b is a stitch endpoint it is
    # promoted into the dynamic namespace at decode time.
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube_a.name)
    static = dh.api.solver.create_group("Static", "STATIC")
    static.add(cube_b.name)

    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    uuid_a = uuid_mod.get_or_create_object_uuid(cube_a)
    uuid_b = uuid_mod.get_or_create_object_uuid(cube_b)
    if not uuid_a or not uuid_b:
        raise RuntimeError("could not allocate UUIDs for the cubes")

    # Author a SOLID->STATIC cross-stitch: each vertex on the SOLID's +X face
    # (world x == 1) ties to a point on the STATIC's -X face (world
    # x == 1+GAP). The SOLID source slots are re-projected onto the SOLID tet
    # surface (using source_points); the STATIC target slots are kept
    # verbatim (STATIC is SHELL-like, 1:1, never re-projected), so they must
    # name a real triangle on the STATIC mesh. We pick the three STATIC
    # vertices on its -X face that the snap finder would land near.
    a_face = [cube_a.matrix_world @ v.co
              for v in cube_a.data.vertices
              if abs((cube_a.matrix_world @ v.co).x - 1.0) < 1e-4]
    b_face_idx = [i for i, v in enumerate(cube_b.data.vertices)
                  if abs((cube_b.matrix_world @ v.co).x - (1.0 + GAP)) < 1e-4]
    if len(a_face) < 1 or len(b_face_idx) < 3:
        raise RuntimeError(
            f"face pick failed: a_face={len(a_face)} b_face_idx={len(b_face_idx)}"
        )
    tgt_tri = [int(b_face_idx[0]), int(b_face_idx[1]), int(b_face_idx[2])]
    b_world = [cube_b.matrix_world @ cube_b.data.vertices[i].co for i in tgt_tri]
    cx = sum(p.x for p in b_world) / 3.0
    cy = sum(p.y for p in b_world) / 3.0
    cz = sum(p.z for p in b_world) / 3.0
    source_points = [[float(p.x), float(p.y), float(p.z)] for p in a_face]
    # target_points is unused by the decoder for a STATIC target (no
    # re-projection), but is part of the authored payload for completeness.
    target_points = [[float(cx), float(cy), float(cz)] for _ in a_face]
    n_pair = len(source_points)
    # 6-wide rows: degenerate SOLID source (re-projected via source_points)
    # and the STATIC target triangle barycentric (kept verbatim).
    ind = [[0, 0, 0, tgt_tri[0], tgt_tri[1], tgt_tri[2]] for _ in range(n_pair)]
    w = [[1.0, 0.0, 0.0, 0.34, 0.33, 0.33] for _ in range(n_pair)]
    dh.log(f"stitch_pairs={n_pair} tgt_tri={tgt_tri}")

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

    # ----- A: SOLID->STATIC stitch encodes 6-wide with source_points -----
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
        "A_static_stitch_encoded",
        encoder_ok,
        {
            "n_entries": len(decoded_cs),
            "ind_widths": ind_widths,
            "has_source_points": bool(entry.get("source_points")) if entry else False,
            "n_source_points": len(entry.get("source_points", [])) if entry else 0,
            "n_pair": n_pair,
        },
    )

    # ----- B: build + run; STATIC promoted, finite PC2s, STATIC frozen ----
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    # fTetWild resamples the SOLID; the rest-pose STATIC is promoted into the
    # dynamic all-pinned namespace because it is a stitch endpoint;
    # cross_stitch_apply_batch re-projects the SOLID source and keeps the
    # STATIC target verbatim.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="static_stitch:build", timeout=360.0)
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
    # The promoted STATIC must stay ~frozen across frames (immovable fixed
    # pins). Measure its max per-frame displacement from frame 0.
    static_max_motion = None
    if arr_b is not None and arr_b.shape[0] >= 1:
        ref = arr_b[0]
        static_max_motion = float(np.max(np.abs(arr_b - ref)))
    static_frozen = bool(
        static_max_motion is not None and static_max_motion < FREEZE_TOL
    )
    dh.record(
        "B_static_stitch_build_runs_and_freezes",
        solver_state != "FAILED"
        and finite_a and finite_b
        and samples_a >= FRAME_COUNT - 1
        and samples_b >= FRAME_COUNT - 1
        and static_frozen,
        {
            "solver_state": solver_state,
            "pc2_a": pc2_a,
            "pc2_b": pc2_b,
            "samples_a": samples_a,
            "samples_b": samples_b,
            "finite_a": finite_a,
            "finite_b": finite_b,
            "static_promoted_has_pc2": bool(pc2_b),
            "static_max_motion": static_max_motion,
            "static_frozen": static_frozen,
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
        .replace("<<FREEZE_TOL>>", repr(_FREEZE_TOL))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 480.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
