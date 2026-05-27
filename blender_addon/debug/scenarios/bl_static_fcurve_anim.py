# File: scenarios/bl_static_fcurve_anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# STATIC group with object-level transform keyframes (Case 1) drives
# the soft-pin shell. The simulator's projected positions stream back
# as a PC2 + ContactSolverCache modifier on the static object so the
# displayed collider sits where the cloth's contact resolution actually
# placed it, not at the raw keyframed pose.
#
# Pipeline exercised end-to-end:
#
#   Blender object fcurves
#       -> core/utils.py:get_transform_keyframes  (sparse T/R/S samples)
#       -> encoder/mesh.py STATIC branch          (info["transform_animation"])
#       -> frontend _populate_static Case 1       (pin.transform_keyframes)
#       -> Rust solver (--features emulated)
#       -> client.apply_animation                 (per-frame scene.frame_set,
#                                                  matrix_world inverse,
#                                                  PC2 local-space write)
#       -> output PC2 + MESH_CACHE on the cube
#
# Subtests:
#   A. encoder_emits_transform_animation
#         The encoded data tree's STATIC entry carries
#         transform_animation with the sampled keyframes and no
#         static_ops / static_deform_animation alongside.
#   B. static_pc2_exists_and_has_expected_shape
#         After fetch + drain, the cube has a ContactSolverCache
#         MESH_CACHE modifier whose PC2 has (>=FRAME_COUNT-1, n_verts,
#         3) shape.
#   C. pc2_local_close_to_rest_at_frame_one
#         The pin target at keyframe frame 1 is matrix_world(1) @
#         local_vert. The apply pipeline writes
#         PC2_local = matrix_world(1)^-1 @ world_sim_pos. With
#         soft-pin residual << 1, PC2_local at the first sample
#         must be close to the cube's rest mesh in local space.
#   D. pc2_local_close_to_rest_at_keyframe_end
#         Same check at the trailing keyframe sample: even though
#         matrix_world has changed (the cube translated), the
#         object-local positions in PC2 stay close to rest because
#         the playback math reconstructs the world pose from
#         matrix_world(N) @ PC2_local(N).
#   E. mesh_cache_present_on_cube
#         A ContactSolverCache MESH_CACHE modifier was created on
#         the cube (the smart-insert path leaves it at index 0
#         when the cube has no other modifiers, but the test
#         doesn't assert position because Case 1 cubes typically
#         have no deformer stack to position around).

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 11
_KEY_FRAME_START = 1
_KEY_FRAME_END = 10
# Cube descends along Z over the keyframe window. Picked big enough
# that a regression that ignored the keyframes (e.g. recorded rest
# pose instead of per-frame matrix_world) would land orders of
# magnitude beyond the soft-pin tolerance.
_MOVE_DELTA_Z = -1.0
# Soft-pin residual tolerance. The simulator enforces the pin as a
# strong-but-finite constraint, so PC2_local sits within a small
# neighborhood of rest_co. 0.1 in solver units is generous; a clean
# pin in this scene settles well under 0.01.
_TOLERANCE = 0.1


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
KEY_FRAME_START = <<KEY_FRAME_START>>
KEY_FRAME_END = <<KEY_FRAME_END>>
MOVE_DELTA_Z = <<MOVE_DELTA_Z>>
TOL = <<TOLERANCE>>


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Cube at (0, 0, 2) initial pose. Object-level location.z is
    # keyframed from z=2 down to z=2+MOVE_DELTA_Z over the keyframe
    # window. No modifiers; no captured-deformation cache. This is
    # the canonical Case 1 input the encoder serializes as
    # transform_animation.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 2.0))
    cube = bpy.context.active_object
    cube.name = "FcurveCube"
    rest_local = np.empty((len(cube.data.vertices), 3), dtype=np.float64)
    cube.data.vertices.foreach_get("co", rest_local.ravel())
    dh.log(f"cube rest local-space bounding box: "
           f"min={rest_local.min(axis=0).tolist()} "
           f"max={rest_local.max(axis=0).tolist()}")

    # Insert keyframes on location.z. Linear handles keep the pin
    # target's interpolation predictable; the apply pipeline uses
    # Blender's depsgraph eval at each frame so the per-channel
    # bezier vs simulator-side single-curve mismatch is handled
    # automatically, but linear here keeps the test focused on the
    # soft-pin residual. fcurves live behind the Blender 5.x layered
    # action API; the addon's _get_fcurves helper walks layers/strips/
    # channelbags to surface them.
    cube.location = (0.0, 0.0, 2.0)
    cube.keyframe_insert(data_path="location", frame=KEY_FRAME_START)
    cube.location = (0.0, 0.0, 2.0 + MOVE_DELTA_Z)
    cube.keyframe_insert(data_path="location", frame=KEY_FRAME_END)
    utils_mod = __import__(pkg + ".core.utils", fromlist=["_get_fcurves"])
    if cube.animation_data and cube.animation_data.action:
        for fc in utils_mod._get_fcurves(cube.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    dh.save_blend(PROBE_DIR, "static_fcurve_anim.blend")
    root = dh.configure_state(
        project_name="static_fcurve_anim",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
    )

    static_group = dh.api.solver.create_group("Stat", "STATIC")
    static_group.add(cube.name)

    # ----- A: encoder emits transform_animation ------------------
    mesh_enc = __import__(pkg + ".core.encoder.mesh", fromlist=["_build_obj_data"])
    decoded = mesh_enc._build_obj_data(bpy.context, persist_topology_hash=False)
    static_obj = None
    for group_entry in decoded:
        if group_entry.get("type") == "STATIC":
            for o in group_entry.get("object", []):
                if o.get("name") == cube.name:
                    static_obj = o
                    break
    has_xform = static_obj is not None and "transform_animation" in static_obj
    has_sd = static_obj is not None and "static_deform_animation" in static_obj
    has_ops = bool((static_obj or {}).get("static_ops") or [])
    n_keyframes = (
        len(static_obj["transform_animation"]["time"]) if has_xform else 0
    )
    dh.record(
        "A_encoder_emits_transform_animation",
        has_xform and not has_sd and not has_ops and n_keyframes == 2,
        {
            "has_transform_animation": has_xform,
            "has_static_deform_animation": has_sd,
            "has_static_ops": has_ops,
            "n_keyframes": n_keyframes,
        },
    )

    # ----- Build, run, fetch -------------------------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="static_fcurve_anim:build")
    dh.log("built")
    dh.run_and_wait(timeout=90.0)
    dh.log(f"ran solver={dh.facade.engine.state.solver.name}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    # ----- B: PC2 exists with expected shape ---------------------
    pc2_path = dh.find_pc2_for(cube)
    pc2_arr = None
    if pc2_path and os.path.isfile(pc2_path):
        pc2_arr = dh.read_pc2(pc2_path)
    expected_n_verts = len(cube.data.vertices)
    shape_ok = (
        pc2_arr is not None
        and pc2_arr.ndim == 3
        and pc2_arr.shape[1] == expected_n_verts
        and pc2_arr.shape[2] == 3
        and pc2_arr.shape[0] >= FRAME_COUNT - 1
    )
    dh.record(
        "B_static_pc2_exists_and_has_expected_shape",
        shape_ok,
        {
            "pc2_path": pc2_path,
            "shape": list(pc2_arr.shape) if pc2_arr is not None else None,
            "expected_n_verts": expected_n_verts,
            "expected_min_samples": FRAME_COUNT - 1,
        },
    )

    # ----- C, D: PC2_local close to rest_local at keyframe frames -
    # The pipeline writes PC2_local(N) = matrix_world(N)^-1 @
    # world_sim_pos. With a tight soft pin and unscaled cube
    # transforms, the simulator's world_sim_pos ~= matrix_world(N) @
    # rest_local plus a tiny residual, so PC2_local(N) - rest_local
    # is exactly the residual in object-local space. We sample the
    # first and last keyframe-coincident PC2 samples; interior
    # frames are simulator-interpolated and the apply pipeline still
    # makes the playback display match (the test in C/D would also
    # pass at any sample, but keyframe samples are the cleanest).
    if not shape_ok:
        for name in (
            "C_pc2_local_close_to_rest_at_frame_one",
            "D_pc2_local_close_to_rest_at_keyframe_end",
            "E_mesh_cache_present_on_cube",
        ):
            dh.record(name, False, {"reason": "PC2 unavailable or wrong shape"})
    else:
        # PC2 sample i corresponds to Blender frame i+1 (the apply
        # path writes at blender_frame=n+1 with frame_idx=n).
        sample_first = KEY_FRAME_START - 1  # blender_frame=1 -> sample 0
        sample_last = KEY_FRAME_END - 1     # blender_frame=10 -> sample 9
        sample_last = min(sample_last, pc2_arr.shape[0] - 1)

        local_first = pc2_arr[sample_first].astype(np.float64)
        local_last = pc2_arr[sample_last].astype(np.float64)
        max_err_first = float(np.max(np.abs(local_first - rest_local)))
        max_err_last = float(np.max(np.abs(local_last - rest_local)))

        dh.record(
            "C_pc2_local_close_to_rest_at_frame_one",
            max_err_first < TOL,
            {
                "sample_index": sample_first,
                "blender_frame": KEY_FRAME_START,
                "max_abs_error_local": max_err_first,
                "tolerance": TOL,
            },
        )

        dh.record(
            "D_pc2_local_close_to_rest_at_keyframe_end",
            max_err_last < TOL,
            {
                "sample_index": sample_last,
                "blender_frame": sample_last + 1,
                "max_abs_error_local": max_err_last,
                "tolerance": TOL,
            },
        )

        # ----- E: MESH_CACHE modifier present ---------------------
        cache_mod = cube.modifiers.get("ContactSolverCache")
        cache_ok = (
            cache_mod is not None
            and cache_mod.cache_format == "PC2"
            and bool(cache_mod.filepath)
        )
        dh.record(
            "E_mesh_cache_present_on_cube",
            cache_ok,
            {
                "has_modifier": cache_mod is not None,
                "cache_format": (
                    cache_mod.cache_format if cache_mod is not None else None
                ),
                "filepath_set": (
                    bool(cache_mod.filepath) if cache_mod is not None else False
                ),
                "modifier_order": [m.name for m in cube.modifiers],
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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
        .replace("<<KEY_FRAME_START>>", str(_KEY_FRAME_START))
        .replace("<<KEY_FRAME_END>>", str(_KEY_FRAME_END))
        .replace("<<MOVE_DELTA_Z>>", repr(_MOVE_DELTA_Z))
        .replace("<<TOLERANCE>>", repr(_TOLERANCE))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
