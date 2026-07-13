# File: scenarios/bl_shell_static_stitch.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# SHELL-to-STATIC cross-stitch end to end (the canonical "cloth stitched to
# a static collider" case). Complements bl_static_stitch.py (SOLID source)
# by exercising a PASSTHROUGH source against a promoted rest-pose STATIC.
#
# A SHELL is not re-tetrahedralized, so its source slots map 1:1 to the
# solver and cross_stitch_apply_batch keeps them verbatim (no re-projection;
# the SOLID-only re-projection branch is not taken). The rest-pose STATIC is
# promoted into the dynamic all-pinned namespace because it is a stitch
# endpoint, so the stitch index can reach its (1:1) surface. This is the
# build/run counterpart to the snap-side bl_static_snap_guard C-case and the
# decoder-level tests/test_cross_stitch_apply_batch.py SHELL/STATIC case.
#
# Subtests:
#   A. shell_static_encoded: the cross_stitch section holds one SHELL->STATIC
#      entry whose ind rows are 6-wide.
#   B. shell_static_build_runs_and_freezes: the STATIC is promoted (it gets an
#      output PC2), both objects produce finite PC2s, the solver runs without
#      FAILED, and the promoted STATIC stays frozen.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Backend-agnostic: the assertions below are robust invariants (finite PC2,
# pinned/anchored region tracks its prescribed motion, free region lags,
# body not FAILED) that hold on BOTH the emulated CPU stub and the real
# CUDA solver, so this runs on the free-runner macOS suite AND the real-GPU
# AWS jobs selected by ``runtests --backend real``.
BACKENDS = ("emulated", "real")

_FRAME_COUNT = 4
_GAP = 0.05
_FREEZE_TOL = 0.05


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

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Cloth grid in the z=0 plane (spans [-1,1]^2); STATIC cube just above it
    # (bottom face at z = 1 + GAP).
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3,
                                    size=2.0, location=(0.0, 0.0, 0.0))
    cloth = bpy.context.active_object
    cloth.name = "Cloth"
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 2.0 + GAP))
    cube = bpy.context.active_object
    cube.name = "StaticB"
    dh.log(f"cloth={len(cloth.data.vertices)} static={len(cube.data.vertices)}")

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "shell_static_stitch.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    root = dh.configure_state(
        project_name="shell_static_stitch",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, 0.0),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    shell = dh.api.solver.create_group("Cloth", "SHELL")
    shell.add(cloth.name)
    static = dh.api.solver.create_group("Static", "STATIC")
    static.add(cube.name)

    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    uuid_cloth = uuid_mod.get_or_create_object_uuid(cloth)
    uuid_static = uuid_mod.get_or_create_object_uuid(cube)
    if not uuid_cloth or not uuid_static:
        raise RuntimeError("could not allocate UUIDs")

    # Author a SHELL->STATIC cross-stitch: every cloth vertex ties to a point
    # on the STATIC's bottom (-Z) face. The SHELL source slots are degenerate
    # [si, si, si] / [1, 0, 0] (1:1 indices, no re-projection); the STATIC
    # target slots name a real bottom-face triangle (kept verbatim).
    bottom = [i for i, v in enumerate(cube.data.vertices)
              if abs((cube.matrix_world @ v.co).z - (1.0 + GAP)) < 1e-4]
    if len(bottom) < 3:
        raise RuntimeError(f"static bottom face pick failed: {len(bottom)}")
    tgt_tri = [int(bottom[0]), int(bottom[1]), int(bottom[2])]
    n_src = len(cloth.data.vertices)
    ind = [[si, si, si, tgt_tri[0], tgt_tri[1], tgt_tri[2]] for si in range(n_src)]
    w = [[1.0, 0.0, 0.0, 0.34, 0.33, 0.33] for _ in range(n_src)]
    source_points = [[float((cloth.matrix_world @ v.co).x),
                      float((cloth.matrix_world @ v.co).y),
                      float((cloth.matrix_world @ v.co).z)]
                     for v in cloth.data.vertices]
    bw = [cube.matrix_world @ cube.data.vertices[i].co for i in tgt_tri]
    cz = [sum(p[k] for p in bw) / 3.0 for k in range(3)]
    target_points = [[float(cz[0]), float(cz[1]), float(cz[2])] for _ in range(n_src)]

    state = dh.groups.get_addon_data(bpy.context.scene).state
    pair = state.merge_pairs.add()
    pair.object_a = cloth.name
    pair.object_b = cube.name
    pair.object_a_uuid = uuid_cloth
    pair.object_b_uuid = uuid_static
    pair.stitch_stiffness = 1.0
    pair.cross_stitch_json = json.dumps({
        "source_uuid": uuid_cloth,
        "target_uuid": uuid_static,
        "ind": ind,
        "w": w,
        "source_points": source_points,
        "target_points": target_points,
        "a_vert_count": len(cloth.data.vertices),
        "b_vert_count": len(cube.data.vertices),
    }, separators=(",", ":"))
    state.merge_pairs_index = len(state.merge_pairs) - 1
    dh.log(f"authored_stitch pairs={n_src}")

    # ----- A: SHELL->STATIC stitch encodes 6-wide -----
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    decoded = dh.decode_addon_blob(param_bytes)
    decoded_cs = decoded.get("cross_stitch", [])
    entry = decoded_cs[0] if decoded_cs else None
    ind_widths = sorted({len(row) for row in entry.get("ind", [])}) if entry else []
    dh.record(
        "A_shell_static_encoded",
        len(decoded_cs) == 1
        and entry.get("source_uuid") == uuid_cloth
        and entry.get("target_uuid") == uuid_static
        and ind_widths == [6],
        {"n_entries": len(decoded_cs), "ind_widths": ind_widths},
    )

    # ----- B: build + run; STATIC promoted, finite PC2s, STATIC frozen -----
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes,
                      message="shell_static_stitch:build", timeout=240.0)
    dh.run_and_wait(timeout=120.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    pc2_cloth = dh.find_pc2_for(cloth)
    pc2_static = dh.find_pc2_for(cube)
    arr_cloth = dh.read_pc2(pc2_cloth) if pc2_cloth else None
    arr_static = dh.read_pc2(pc2_static) if pc2_static else None
    finite_cloth = bool(arr_cloth is not None and np.all(np.isfinite(arr_cloth)))
    finite_static = bool(arr_static is not None and np.all(np.isfinite(arr_static)))
    static_max_motion = None
    if arr_static is not None and arr_static.shape[0] >= 1:
        static_max_motion = float(np.max(np.abs(arr_static - arr_static[0])))
    static_frozen = bool(static_max_motion is not None and static_max_motion < FREEZE_TOL)
    dh.record(
        "B_shell_static_build_runs_and_freezes",
        solver_state != "FAILED"
        and finite_cloth and finite_static
        and (arr_cloth is not None and arr_cloth.shape[0] >= FRAME_COUNT - 1)
        and (arr_static is not None and arr_static.shape[0] >= FRAME_COUNT - 1)
        and static_frozen,
        {
            "solver_state": solver_state,
            "pc2_cloth": pc2_cloth,
            "pc2_static": pc2_static,
            "static_promoted_has_pc2": bool(pc2_static),
            "finite_cloth": finite_cloth,
            "finite_static": finite_static,
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
