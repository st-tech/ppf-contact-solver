# File: scenarios/bl_static_deform_first_frame.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Frame 1 of a moving STATIC collider must land on its captured pose.
#
# The captured-deformation cache is stored in solver WORLD space, while the
# ContactSolverCache (MESH_CACHE) modifier reads object-LOCAL positions and
# Blender re-applies the object's own animated transform on top. So each frame
# written to the PC2 has to be divided by the matrix_world of ITS OWN frame.
#
# PC2 index 0 (Blender frame 1) is written by the leading gap-fill, which drew
# the pose from the cache but projected it with ONE matrix_world: the object's,
# as of whatever frame the scene happened to be parked on when the frames
# arrived. Frame 1 then showed the collider displaced by exactly how far it
# travels between the parked frame and frame 1, while every later frame (which
# the real-frame path projects per frame) landed on its captured pose exactly.
# Park the scene on frame 1 and the bug vanishes, which is what made it look
# intermittent.
#
# So this scenario deliberately parks the scene AWAY from frame 1 before
# fetching, on a collider whose matrix_world genuinely varies per frame (an
# object-level location fcurve) and whose mesh genuinely deforms (a keyed shape
# key, which is what puts it on the captured-cache path at all).
#
# Subtests:
#   A. parked_away_from_frame_one: the scene really is parked elsewhere when
#         the frames arrive, or the test proves nothing.
#   B. frame_one_matches_capture: PC2 index 0, mapped back to solver world
#         through matrix_world AT FRAME 1, equals the captured cache's frame 0.
#         Tight tolerance: this frame is a copy of the cache, not a solve.
#   C. later_frames_match_capture: the frames the real path writes still land
#         on their captured poses, so a fix to the gap-fill has not disturbed
#         them.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("emulated", "real")

_FRAME_COUNT = 10
# The collider travels far enough per frame that projecting frame 1 with the
# wrong frame's matrix is unmissable (~0.1 units/frame).
_TRAVEL_Y = -1.0
# Parked well away from frame 1 when the frames arrive. The bug's size is the
# object's travel between here and frame 1.
_PARK_FRAME = 5
# PC2 index 0 is a copy of the captured cache, not a solve, so it must match
# to float32 round-off. The bug produces ~0.4 units of error here.
_TOL = 1e-4


_DRIVER_BODY = r"""
import traceback
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
TRAVEL_Y = <<TRAVEL_Y>>
PARK_FRAME = <<PARK_FRAME>>
TOL = <<TOL>>


# Depsgraph-evaluate obj per frame -> (n_frames, n_verts, 3) in solver world
# space. This is exactly what Capture Deformation records.
def capture_world_solver_frames(obj, frame_start, frame_end):
    transform_mod = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    scene = bpy.context.scene
    saved = scene.frame_current
    n_frames = frame_end - frame_start + 1
    n_verts = len(obj.data.vertices)
    out = np.empty((n_frames, n_verts, 3), dtype=np.float32)
    z2y = np.array(transform_mod.zup_to_yup(), dtype=np.float64).reshape(4, 4)
    try:
        for i, f in enumerate(range(frame_start, frame_end + 1)):
            scene.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            eo = obj.evaluated_get(dg)
            em = eo.to_mesh()
            try:
                co = np.empty((n_verts, 3), dtype=np.float64)
                em.vertices.foreach_get("co", co.ravel())
                mw = np.array(eo.matrix_world, dtype=np.float64).reshape(4, 4)
                m = z2y @ mw
                h = np.concatenate([co, np.ones((n_verts, 1))], axis=1)
                out[i] = (h @ m.T)[:, :3].astype(np.float32, copy=False)
            finally:
                eo.to_mesh_clear()
    finally:
        scene.frame_set(saved)
    return out


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "MovingStatic"

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT

    # A keyed shape key makes the MESH deform, which is what routes this
    # collider onto the captured-cache path in the first place.
    cube.shape_key_add(name="Basis", from_mix=False)
    key = cube.shape_key_add(name="Deform", from_mix=False)
    for v in key.data:
        v.co.z *= 1.6
    key.value = 0.0
    key.keyframe_insert(data_path="value", frame=1)
    key.value = 1.0
    key.keyframe_insert(data_path="value", frame=FRAME_COUNT)

    # An object-level location fcurve makes matrix_world vary per frame, which
    # is what the gap-fill has to divide out frame by frame.
    cube.location = (0.0, 0.0, 0.0)
    cube.keyframe_insert(data_path="location", frame=1)
    cube.location = (0.0, TRAVEL_Y, 0.0)
    cube.keyframe_insert(data_path="location", frame=FRAME_COUNT)

    dh.save_blend(PROBE_DIR, "static_deform_first_frame.blend")
    root = dh.configure_state(project_name="static_deform_first_frame",
                              frame_count=FRAME_COUNT,
                              frame_rate=24,
                              step_size=1.0 / 24.0)

    group = dh.api.solver.create_group("Stat", "STATIC")
    group.add(cube.name)

    # Capture straight into the cache: the modal Capture Deformation operator
    # needs event-loop ticks the rig's Blender does not run.
    captured = capture_world_solver_frames(cube, 1, FRAME_COUNT)
    pc2 = __import__(pkg + ".core.pc2", fromlist=["write_static_deform_pc2"])
    pc2.write_static_deform_pc2(cube, captured)
    dh.log(f"captured {captured.shape}")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="static_deform_first_frame:build")
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)

    # THE POINT: park away from frame 1 before the frames land, so the
    # gap-fill's matrix_world is the wrong one unless it re-reads per frame.
    scene.frame_set(PARK_FRAME)
    parked = int(scene.frame_current)
    dh.fetch_and_drain()

    dh.record(
        "A_parked_away_from_frame_one",
        parked != 1,
        {"parked_frame": parked},
    )

    # ---- B / C: every PC2 sample must equal its captured pose ---------
    transform_mod = __import__(pkg + ".core.transform", fromlist=["world_matrix"])
    n_verts = len(cube.data.vertices)
    arr = dh.read_pc2(dh.find_pc2_for(cube))
    n_samples = int(arr.shape[0]) if arr is not None else 0

    # PC2 sample -> solver world via matrix_world of ITS OWN frame, compared
    # against the captured cache for that frame.
    def world_err(sample_idx):
        saved = scene.frame_current
        try:
            # PC2 index i is Blender frame i + start; this scenario leaves
            # Starting Frame at its default of 1, so the offset is +1 here.
            scene.frame_set(sample_idx + 1)
            wm = np.array(transform_mod.world_matrix(cube),
                          dtype=np.float64).reshape(4, 4)
        finally:
            scene.frame_set(saved)
        local = arr[sample_idx].astype(np.float64)
        h = np.concatenate([local, np.ones((n_verts, 1))], axis=1)
        world = (h @ wm.T)[:, :3]
        return float(np.max(np.abs(world - captured[sample_idx].astype(np.float64))))

    err0 = world_err(0) if n_samples >= 1 else -1.0
    dh.record(
        "B_frame_one_matches_capture",
        n_samples >= 1 and 0.0 <= err0 < TOL,
        {"pc2_samples": n_samples, "frame1_max_err": err0, "tol": TOL,
         "parked_frame": parked,
         "note": "the bug puts the object's travel between the parked frame "
                 "and frame 1 into this number"},
    )

    # Later frames go through the real-frame path, which already projects per
    # frame. They carry the solver's soft-pin residual, so the bound is looser
    # than B's, but far tighter than the bug's displacement.
    later = [world_err(i) for i in range(1, min(n_samples, FRAME_COUNT - 1))]
    worst_later = max(later) if later else -1.0
    dh.record(
        "C_later_frames_match_capture",
        len(later) > 0 and worst_later < 0.02,
        {"n_checked": len(later), "worst_err": worst_later, "bound": 0.02},
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
        .replace("<<TRAVEL_Y>>", repr(_TRAVEL_Y))
        .replace("<<PARK_FRAME>>", str(_PARK_FRAME))
        .replace("<<TOL>>", repr(_TOL))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
