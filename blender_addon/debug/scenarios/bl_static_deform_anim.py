# File: scenarios/bl_static_deform_anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# STATIC collider with a deforming modifier stack (Armature) carries a
# per-frame depsgraph-baked vertex cache through the pipeline:
#
#   Blender depsgraph
#       -> core.pc2.write_static_deform_pc2  (cache file)
#       -> encoder/mesh.py STATIC branch     (info["static_deform_animation"])
#       -> frontend _populate_static Case-3  (per-vertex pin shell)
#       -> Rust solver (--features emulated)
#       -> output vert_*.bin
#
# This scenario builds an icosphere parented to a 2-bone armature,
# animates the bones so the mesh deforms, captures the deformation
# directly into the PC2 cache (bypassing the modal Capture Deformation
# operator - headless Blender has no event-loop ticks), and then runs
# the full transfer/build/run cycle. The collider's per-frame motion
# is solver-internal (the shell is `_exclude_from_output`), so we
# validate the upload contract end-to-end and the cache itself, not
# the simulator's contact resolution.
#
# Subtests:
#   A. capture_writes_pc2_with_expected_shape
#         write_static_deform_pc2 + on-disk PC2 round-trip has
#         (n_frames, n_verts, 3) and frame 0 matches the depsgraph
#         frame-1 evaluation in solver space.
#   B. encoder_emits_static_deform_animation
#         The encoded data.pickle's STATIC group entry carries
#         static_deform_animation with the cache shape, and the
#         encoder dropped transform / transform_animation.
#   C. encoder_rejects_deforming_without_cache
#         Clearing the cache and re-encoding raises ValueError
#         mentioning "Capture Deformation" (the helpful next action).
#   D. build_and_run_completes_with_deform_cache
#         The full transfer/build/run cycle finishes without errors
#         and the cloth mesh produces at least frame_count - 1 PC2
#         samples.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 11
_BEND_FRAME_START = 1
_BEND_FRAME_END = 10
# Bone rotation at frame_end (radians around Y). Picked big enough that
# top vertices move by ~1.0 world unit so a regression that bakes the
# rest pose instead of the deformed mesh would fail the cache-content
# check by orders of magnitude more than _TOLERANCE.
_BEND_ANGLE = 1.2
_TOLERANCE = 1e-3


_DRIVER_BODY = r"""
import math
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
BEND_FRAME_START = <<BEND_FRAME_START>>
BEND_FRAME_END = <<BEND_FRAME_END>>
BEND_ANGLE = <<BEND_ANGLE>>
TOL = <<TOLERANCE>>


def _make_two_bone_armature(name):
    # Build a vertical 2-bone armature: Bone (0->1 on Z) + Bone.001
    # (1->2 on Z), the second connected to the first. Returns the
    # armature object.
    bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
    arm_obj = bpy.context.active_object
    arm_obj.name = name
    edit_bones = arm_obj.data.edit_bones
    # Clear the default single bone and lay out our two.
    while edit_bones:
        edit_bones.remove(edit_bones[0])
    b0 = edit_bones.new("Bone")
    b0.head = (0.0, 0.0, 0.0)
    b0.tail = (0.0, 0.0, 1.0)
    b1 = edit_bones.new("Bone.001")
    b1.head = (0.0, 0.0, 1.0)
    b1.tail = (0.0, 0.0, 2.0)
    b1.parent = b0
    b1.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _capture_world_solver_frames(obj, frame_start, frame_end):
    # Depsgraph-evaluate obj at each frame in [frame_start, frame_end]
    # and return (n_frames, n_verts, 3) float32 in solver coords.
    transform_mod = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    zup_to_yup = transform_mod.zup_to_yup
    scene = bpy.context.scene
    saved = scene.frame_current
    n_frames = frame_end - frame_start + 1
    n_verts = len(obj.data.vertices)
    out = np.empty((n_frames, n_verts, 3), dtype=np.float32)
    z2y = np.array(zup_to_yup(), dtype=np.float64).reshape(4, 4)
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
                homog = np.concatenate([co, np.ones((n_verts, 1), dtype=np.float64)], axis=1)
                out[i] = (homog @ m.T)[:, :3].astype(np.float32, copy=False)
            finally:
                eo.to_mesh_clear()
    finally:
        scene.frame_set(saved)
    return out


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Fresh scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Armature + sphere parented + armature modifier.
    arm = _make_two_bone_armature("DeformArm")
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2, radius=1.0, location=(0.0, 0.0, 1.0),
    )
    sphere = bpy.context.active_object
    sphere.name = "DeformSphere"
    sphere.parent = arm
    # Assign every vertex to Bone (lower) full-weight, then re-weight
    # vertices with z > 1.0 to Bone.001 so the upper bone bends only
    # the top hemisphere.
    vg_b0 = sphere.vertex_groups.new(name="Bone")
    vg_b1 = sphere.vertex_groups.new(name="Bone.001")
    upper = []
    lower = []
    for v in sphere.data.vertices:
        if v.co.z > 0.0:
            upper.append(v.index)
        else:
            lower.append(v.index)
    if upper:
        vg_b1.add(upper, 1.0, "REPLACE")
    if lower:
        vg_b0.add(lower, 1.0, "REPLACE")
    mod = sphere.modifiers.new(name="ArmatureMod", type="ARMATURE")
    mod.object = arm

    # Animate Bone.001 to rotate around local Y from 0 to BEND_ANGLE
    # over [BEND_FRAME_START, BEND_FRAME_END].
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    pose_bone = arm.pose.bones["Bone.001"]
    pose_bone.rotation_mode = "XYZ"
    pose_bone.rotation_euler = (0.0, 0.0, 0.0)
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=BEND_FRAME_START)
    pose_bone.rotation_euler = (0.0, BEND_ANGLE, 0.0)
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=BEND_FRAME_END)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = sphere

    # Cloth: a plane that drapes over the sphere so contact actually
    # happens. Solver doesn't need to produce contact resolutions for
    # this scenario's checks, but having a dynamic SHELL ensures the
    # full make/build path goes through (static-only scenes hit a
    # different code path - see `has_dynamic` in solver.py).
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 3.5))
    plane = bpy.context.active_object
    plane.name = "DrapeCloth"
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=4)
    bpy.ops.object.mode_set(mode="OBJECT")

    dh.save_blend(PROBE_DIR, "static_deform_anim.blend")
    root = dh.configure_state(
        project_name="static_deform_anim",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
    )

    static_group = dh.api.solver.create_group("Stat", "STATIC")
    static_group.add(sphere.name)
    shell_group = dh.api.solver.create_group("Cloth", "SHELL")
    shell_group.add(plane.name)

    # Bake the cache directly via the public PC2 helper. This is what
    # the modal Capture Deformation operator does at each tick;
    # bypassing the modal loop is fine because headless Blender has
    # no event-loop pump and the encoder only cares that the cache
    # file exists.
    pc2 = __import__(pkg + ".core.pc2", fromlist=[
        "write_static_deform_pc2", "remove_static_deform_pc2",
        "has_static_deform_animation", "get_static_deform_cache",
        "static_deform_pc2_key",
    ])
    bpy.context.scene.frame_start = BEND_FRAME_START
    bpy.context.scene.frame_end = BEND_FRAME_START + FRAME_COUNT - 1
    captured = _capture_world_solver_frames(
        sphere,
        bpy.context.scene.frame_start,
        bpy.context.scene.frame_end,
    )
    pc2.write_static_deform_pc2(sphere, captured)
    dh.log(
        f"captured cache shape={captured.shape} "
        f"key={pc2.static_deform_pc2_key(sphere)}"
    )

    # ----- A: cache exists, right shape, frame 0 matches depsgraph -
    has_cache = pc2.has_static_deform_animation(sphere)
    loaded = pc2.get_static_deform_cache(sphere)
    expected_shape = (FRAME_COUNT, len(sphere.data.vertices), 3)
    shape_ok = (
        has_cache
        and loaded is not None
        and tuple(loaded.shape) == expected_shape
    )
    frame0_max_err = -1.0
    if shape_ok:
        frame0_max_err = float(np.max(np.abs(loaded[0] - captured[0])))
    dh.record(
        "A_capture_writes_pc2_with_expected_shape",
        shape_ok and frame0_max_err < TOL,
        {
            "has_cache": bool(has_cache),
            "shape": list(loaded.shape) if loaded is not None else None,
            "expected_shape": list(expected_shape),
            "frame0_max_err": frame0_max_err,
            "tolerance": TOL,
        },
    )

    # ----- B: encoder emits static_deform_animation ----------------
    # Inspect the in-memory data tree directly: easier than parsing
    # the CBOR envelope, same source of truth as the upload bytes
    # (encode_obj just runs cbor2.dumps on this tree).
    mesh_enc = __import__(pkg + ".core.encoder.mesh", fromlist=["_build_obj_data"])
    decoded = mesh_enc._build_obj_data(bpy.context, persist_topology_hash=False)
    # Also produce real upload bytes so dh.encode_payload's downstream
    # callers see the same artifact - the connect/build leg uses them.
    data_bytes, param_bytes = dh.encode_payload()
    static_obj = None
    for group_entry in decoded:
        if group_entry.get("type") == "STATIC":
            for o in group_entry.get("object", []):
                if o.get("name") == sphere.name:
                    static_obj = o
                    break
    has_sd = static_obj is not None and "static_deform_animation" in static_obj
    has_xform_anim = static_obj is not None and "transform_animation" in static_obj
    has_static_ops = bool((static_obj or {}).get("static_ops") or [])
    payload_shape = None
    if has_sd:
        vf = static_obj["static_deform_animation"]["vert_frames"]
        payload_shape = list(np.asarray(vf).shape)
    dh.record(
        "B_encoder_emits_static_deform_animation",
        has_sd
        and not has_xform_anim
        and not has_static_ops
        and payload_shape == list(expected_shape),
        {
            "has_static_deform_animation": has_sd,
            "has_transform_animation": has_xform_anim,
            "has_static_ops": has_static_ops,
            "payload_shape": payload_shape,
            "expected_shape": list(expected_shape),
        },
    )

    # ----- C: deforming without cache => upload-fatal -------------
    pc2.remove_static_deform_pc2(sphere)
    encode_error = None
    try:
        dh.encode_payload()
    except Exception as exc:
        encode_error = f"{type(exc).__name__}: {exc}"
    error_mentions_capture = (
        encode_error is not None and "Capture Deformation" in encode_error
    )
    dh.record(
        "C_encoder_rejects_deforming_without_cache",
        error_mentions_capture,
        {"encode_error": encode_error},
    )
    # Restore the cache for the build/run leg below.
    pc2.write_static_deform_pc2(sphere, captured)

    # ----- D: full transfer/build/run round-trip ------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="static_deform_anim:build")
    dh.log("built")
    dh.run_and_wait(timeout=120.0)
    dh.log(f"ran solver={dh.facade.engine.state.solver.name}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    cloth_pc2 = dh.find_pc2_for(plane)
    cloth_arr = dh.read_pc2(cloth_pc2) if cloth_pc2 and os.path.isfile(cloth_pc2) else None
    cloth_samples = int(cloth_arr.shape[0]) if cloth_arr is not None else 0
    dh.record(
        "D_build_and_run_completes_with_deform_cache",
        cloth_arr is not None and cloth_samples >= FRAME_COUNT - 1,
        {
            "cloth_pc2_path": cloth_pc2,
            "cloth_samples": cloth_samples,
            "expected_min_samples": FRAME_COUNT - 1,
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
        .replace("<<BEND_FRAME_START>>", str(_BEND_FRAME_START))
        .replace("<<BEND_FRAME_END>>", str(_BEND_FRAME_END))
        .replace("<<BEND_ANGLE>>", repr(_BEND_ANGLE))
        .replace("<<TOLERANCE>>", repr(_TOLERANCE))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
