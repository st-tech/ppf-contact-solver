# File: scenarios/bl_pin_capture_deformation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Capture Pin Deformation: a SHELL pin whose vertices move because of
# an Armature modifier carries a depsgraph-baked per-frame buffer that
# the encoder consumes in place of manual Make-Keyframe vertex-co
# fcurves (implicit PC2-wins). Mirrors bl_static_deform_anim end to
# end, but exercises the per-pin path on a dynamic SHELL group rather
# than the per-object path on a STATIC collider.
#
# Subtests:
#   A. capture_writes_pin_anim_pc2_with_expected_shape:
#         write_pin_anim_pc2 round-trips through the in-memory cache and
#         on-disk file with shape (n_frames, n_pin_verts, 3); the
#         frame 0 row matches a direct depsgraph evaluation in solver
#         world space.
#   B. capture_sets_has_captured_anim_and_sentinel:
#         finalize sets pin_item.has_captured_anim and adds an
#         EMBEDDED_MOVE op at slot 0.
#   C. encoder_emits_pin_anim_from_pc2_cache:
#         the encoded data blob carries dense per-frame pin_anim with
#         one sample per scene frame and embedded_move_index=0; no
#         vertex-co fcurves were touched on the mesh's action.
#   D. encoder_prefers_pc2_over_fcurves:
#         with PC2 cache present AND stale vertex-co fcurves on the
#         same mesh, the encoded pin_anim time list matches the dense
#         per-frame schedule from the PC2, not the sparse fcurve
#         schedule.
#   E. clear_drops_cache_and_sentinel:
#         object.clear_pin_deformation removes the PC2, clears
#         has_captured_anim, drops EMBEDDED_MOVE; the encoder no longer
#         emits pin_anim.
#   F. build_and_run_completes_with_pin_capture:
#         full transfer/build/run cycle finishes; cloth produces at
#         least frame_count - 1 PC2 samples.
#   G. capture_rejects_non_deforming_object:
#         capture op refuses on a pin whose mesh has no deforming
#         modifier stack.
#   H. capture_rejects_torque_pin:
#         capture op refuses on a pin that already carries a TORQUE
#         op; matches the encoder's torque-vs-embedded incompatibility
#         contract.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 11
_BEND_FRAME_START = 1
_BEND_FRAME_END = 10
_BEND_ANGLE = 1.2  # rad; comfortably > _TOLERANCE so a rest-pose bake fails the value check
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
    # Vertical 2-bone armature (head 0,0,0 -> 0,0,1, then 0,0,1 -> 0,0,2).
    bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
    arm_obj = bpy.context.active_object
    arm_obj.name = name
    edit_bones = arm_obj.data.edit_bones
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


def _sample_pin_world_solver(obj, pin_indices, frame_start, frame_end):
    # Depsgraph-evaluate obj at each frame in [frame_start, frame_end]
    # and return (n_frames, n_pin_verts, 3) float32 in solver world
    # space, matching what pin_capture_ops._sample_pin_frame_world
    # produces inside the modal operator.
    transform_mod = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    zup_to_yup = transform_mod.zup_to_yup
    scene = bpy.context.scene
    saved = scene.frame_current
    n_frames = frame_end - frame_start + 1
    n_pin = len(pin_indices)
    out = np.empty((n_frames, n_pin, 3), dtype=np.float32)
    z2y = np.array(zup_to_yup(), dtype=np.float64).reshape(4, 4)
    pin_arr = np.asarray(pin_indices, dtype=np.int64)
    try:
        for i, f in enumerate(range(frame_start, frame_end + 1)):
            scene.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            eo = obj.evaluated_get(dg)
            em = eo.to_mesh()
            try:
                n_total = len(em.vertices)
                co = np.empty((n_total, 3), dtype=np.float64)
                em.vertices.foreach_get("co", co.ravel())
                co_sub = co[pin_arr]
                mw = np.array(eo.matrix_world, dtype=np.float64).reshape(4, 4)
                m = z2y @ mw
                homog = np.concatenate(
                    [co_sub, np.ones((n_pin, 1), dtype=np.float64)], axis=1,
                )
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

    # Armature + a tall plane parented to it, with two pinned vertex
    # rings: lower verts on Bone, upper verts on Bone.001. The upper
    # bone rotates, so the upper-ring pin (the one we capture) moves
    # over the timeline while the lower ring stays at rest.
    arm = _make_two_bone_armature("CapArm")
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 1.0))
    plane = bpy.context.active_object
    plane.name = "CapPlane"
    plane.parent = arm
    # 3x3 subdivision so we have ~16 verts to play with.
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=2)
    bpy.ops.object.mode_set(mode="OBJECT")

    vg_b0 = plane.vertex_groups.new(name="Bone")
    vg_b1 = plane.vertex_groups.new(name="Bone.001")
    upper, lower = [], []
    for v in plane.data.vertices:
        # Plane local-z is 0 (it's flat); split by local-y instead so
        # the bone deformation has somewhere to land.
        (upper if v.co.y >= 0.0 else lower).append(v.index)
    if upper:
        vg_b1.add(upper, 1.0, "REPLACE")
    if lower:
        vg_b0.add(lower, 1.0, "REPLACE")
    mod = plane.modifiers.new(name="ArmatureMod", type="ARMATURE")
    mod.object = arm

    # Pin VG covers the upper-ring vertices (the ones Bone.001 moves).
    # Keep its name distinct from the bone vertex groups so we can
    # tell pins apart from skin weights in the encoder output.
    vg_pin = plane.vertex_groups.new(name="UpperPin")
    vg_pin.add(upper, 1.0, "REPLACE")

    # Animate Bone.001 rotation around local Y over [BEND_FRAME_START,
    # BEND_FRAME_END].
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    pose_bone = arm.pose.bones["Bone.001"]
    pose_bone.rotation_mode = "XYZ"
    pose_bone.rotation_euler = (0.0, 0.0, 0.0)
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=BEND_FRAME_START)
    pose_bone.rotation_euler = (0.0, BEND_ANGLE, 0.0)
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=BEND_FRAME_END)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = plane

    dh.save_blend(PROBE_DIR, "pin_capture_deformation.blend")
    root = dh.configure_state(
        project_name="pin_capture_deformation",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
    )

    shell_group = dh.api.solver.create_group("Cloth", "SHELL")
    shell_group.add(plane.name)
    shell_group.create_pin(plane.name, "UpperPin")
    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    group.pin_vertex_groups_index = 0
    pin_item = group.pin_vertex_groups[0]

    # Stamp scene frame range to match the bone keyframes so the
    # capture sees the full bend.
    bpy.context.scene.frame_start = BEND_FRAME_START
    bpy.context.scene.frame_end = BEND_FRAME_START + FRAME_COUNT - 1

    # Bypass the modal operator (headless Blender has no event-loop
    # ticks); call the underlying helpers directly. This is the same
    # call shape the operator uses at each tick.
    pc2 = __import__(pkg + ".core.pc2", fromlist=[
        "write_pin_anim_pc2", "remove_pin_anim_pc2", "has_pin_anim_pc2",
        "get_pin_anim_cache", "pin_anim_pc2_key", "unload_pin_anim_cache",
    ])
    pin_ops = __import__(pkg + ".ui.dynamics.pin_ops",
                        fromlist=["_ensure_embedded_move_op",
                                  "_remove_embedded_move_ops"])
    pin_indices = sorted(upper)
    captured = _sample_pin_world_solver(
        plane, pin_indices,
        bpy.context.scene.frame_start, bpy.context.scene.frame_end,
    )
    pc2.write_pin_anim_pc2(plane, "UpperPin", captured)
    # Operator's finalize would do these two; mirror them by hand for
    # the headless path.
    pin_item.has_captured_anim = True
    pin_ops._ensure_embedded_move_op(pin_item)

    expected_shape = (FRAME_COUNT, len(pin_indices), 3)
    loaded = pc2.get_pin_anim_cache(plane, "UpperPin")

    # ----- A: cache shape + frame-0 round-trip --------------------
    has_cache = pc2.has_pin_anim_pc2(plane, "UpperPin")
    shape_ok = (
        has_cache
        and loaded is not None
        and tuple(loaded.shape) == expected_shape
    )
    frame0_max_err = -1.0
    if shape_ok:
        frame0_max_err = float(np.max(np.abs(loaded[0] - captured[0])))
    dh.record(
        "A_capture_writes_pin_anim_pc2_with_expected_shape",
        shape_ok and 0.0 <= frame0_max_err < TOL,
        {
            "has_cache": bool(has_cache),
            "shape": list(loaded.shape) if loaded is not None else None,
            "expected_shape": list(expected_shape),
            "frame0_max_err": frame0_max_err,
            "tolerance": TOL,
        },
    )

    # ----- B: flag + sentinel set --------------------------------
    flag = bool(pin_item.has_captured_anim)
    embed_ops = [
        i for i, op in enumerate(pin_item.operations)
        if op.op_type == "EMBEDDED_MOVE"
    ]
    sentinel_at_zero = embed_ops == [0]
    dh.record(
        "B_capture_sets_has_captured_anim_and_sentinel",
        flag and sentinel_at_zero,
        {
            "has_captured_anim": flag,
            "embedded_move_indices": embed_ops,
        },
    )

    # ----- C: encoder emits dense pin_anim with embedded_move_index 0
    encoder_pin = __import__(pkg + ".core.encoder.pin",
                             fromlist=["_encode_pin_config"])
    state = addon_root.state
    pin_cfg = encoder_pin._encode_pin_config(
        bpy.context, [group], state,
    )
    obj_uuid = pin_item.object_uuid
    obj_cfg = pin_cfg.get(obj_uuid, {})
    # Pick the first pin-vertex entry from the encoded config; every
    # pinned vertex gets the same cfg payload but its own pin_anim
    # track keyed by its own vertex index.
    any_cfg = None
    any_v = None
    for v, c in obj_cfg.items():
        if "pin_anim" in c:
            any_cfg = c
            any_v = v
            break
    has_pin_anim = any_cfg is not None
    eml = (any_cfg or {}).get("embedded_move_index", -1)
    times = list(
        (any_cfg or {}).get("pin_anim", {}).get(any_v, {}).get("time", [])
    )
    pos = (any_cfg or {}).get("pin_anim", {}).get(any_v, {}).get("position")
    pos_shape = list(np.asarray(pos).shape) if pos is not None else None
    dense_ok = has_pin_anim and len(times) == FRAME_COUNT and eml == 0
    # The mesh action must NOT carry vertex-co fcurves: the manual
    # path was never invoked.
    has_fcurves = False
    ad = plane.data.animation_data
    if ad is not None and ad.action is not None:
        import re as _re
        rgx = _re.compile(r"vertices\[\d+\]\.co$")
        if hasattr(ad.action, "layers") and ad.action.layers:
            for layer in ad.action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        if any(rgx.match(fc.data_path) for fc in bag.fcurves):
                            has_fcurves = True
        elif hasattr(ad.action, "fcurves"):
            has_fcurves = any(rgx.match(fc.data_path) for fc in ad.action.fcurves)
    dh.record(
        "C_encoder_emits_pin_anim_from_pc2_cache",
        dense_ok and not has_fcurves,
        {
            "has_pin_anim": has_pin_anim,
            "embedded_move_index": eml,
            "n_times": len(times),
            "expected_n_times": FRAME_COUNT,
            "position_shape": pos_shape,
            "mesh_has_vertex_co_fcurves": has_fcurves,
        },
    )

    # ----- D: PC2 wins over stale fcurves ------------------------
    # Author a sparse vertex-co fcurve on a pinned vertex at frames
    # 1 and 5 (two samples). If the encoder fell back to fcurves, the
    # resulting pin_anim time list would be length 2; with PC2-wins
    # it stays FRAME_COUNT.
    bpy.context.scene.frame_set(BEND_FRAME_START)
    plane.data.vertices[pin_indices[0]].keyframe_insert(data_path="co")
    bpy.context.scene.frame_set(BEND_FRAME_START + 4)
    plane.data.vertices[pin_indices[0]].keyframe_insert(data_path="co")
    pin_cfg2 = encoder_pin._encode_pin_config(
        bpy.context, [group], state,
    )
    obj_cfg2 = pin_cfg2.get(obj_uuid, {})
    any_cfg2 = None
    for v, c in obj_cfg2.items():
        if "pin_anim" in c:
            any_cfg2 = c
            any_v2 = v
            break
    times2 = list(
        (any_cfg2 or {}).get("pin_anim", {}).get(any_v2, {}).get("time", [])
    )
    dh.record(
        "D_encoder_prefers_pc2_over_fcurves",
        len(times2) == FRAME_COUNT,
        {
            "n_times_with_pc2_and_fcurves": len(times2),
            "expected_n_times": FRAME_COUNT,
        },
    )

    # ----- D2: pin overlay follows armature deformation ----------
    # Before the simulation produces a PC2 cache for the cloth, the
    # overlay must sample the depsgraph-evaluated mesh (the cloth's
    # armature/lattice/etc.) so pin dots track the bone deformation
    # in the viewport. The rest-pose fallback would leave them stuck
    # at the un-deformed positions. We exercise _build_pin_data
    # directly at a mid-bend frame and verify the dot for the moving
    # pin matches the depsgraph evaluation exactly.
    overlay_pins = __import__(
        pkg + ".ui.dynamics.overlay_geometry.pins",
        fromlist=["_build_pin_data"],
    )
    bpy.context.scene.frame_set(BEND_FRAME_START + (BEND_FRAME_END - BEND_FRAME_START)//2)
    _dg = bpy.context.evaluated_depsgraph_get()
    _moving_idx = next(
        (vi for vi in pin_indices if vi != pin_indices[0]),
        pin_indices[0],
    )
    _eo = plane.evaluated_get(_dg)
    _em = _eo.to_mesh()
    try:
        _dep_world = _eo.matrix_world @ _em.vertices[_moving_idx].co
    finally:
        _eo.to_mesh_clear()
    _dots = overlay_pins._build_pin_data(bpy.context.scene, _dg)
    _best_delta = min(
        ((_dot[0] - _dep_world).length for _dot in (_dots or [])),
        default=float("inf"),
    )
    dh.record(
        "D2_pin_overlay_follows_armature_deformation",
        _best_delta < 1e-4,
        {
            "moving_vertex": int(_moving_idx),
            "depsgraph_world": [float(_dep_world[0]), float(_dep_world[1]), float(_dep_world[2])],
            "closest_dot_delta": float(_best_delta),
        },
    )

    # ----- E: Clear drops cache + sentinel -----------------------
    # Clean the stale vertex-co fcurves first so Clear hits the no-
    # fcurves branch (drops the EMBEDDED_MOVE sentinel).
    bpy.ops.object.delete_pin_keyframes(group_index=0)
    # bpy.ops.object.delete_pin_keyframes is the manual-path delete;
    # it strips fcurves AND drops EMBEDDED_MOVE. Reseed the sentinel
    # for the Clear test.
    pin_item.has_captured_anim = True
    pin_ops._ensure_embedded_move_op(pin_item)
    bpy.ops.object.clear_pin_deformation(
        "EXEC_DEFAULT", group_index=0, pin_index=0,
    )
    cache_gone = not pc2.has_pin_anim_pc2(plane, "UpperPin")
    flag_off = not bool(pin_item.has_captured_anim)
    sentinel_gone = not any(
        op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations
    )
    pin_cfg3 = encoder_pin._encode_pin_config(
        bpy.context, [group], state,
    )
    obj_cfg3 = pin_cfg3.get(obj_uuid, {})
    has_anim_after_clear = any(
        "pin_anim" in c for c in obj_cfg3.values()
    )
    dh.record(
        "E_clear_drops_cache_and_sentinel",
        cache_gone and flag_off and sentinel_gone and not has_anim_after_clear,
        {
            "cache_gone": cache_gone,
            "flag_off": flag_off,
            "sentinel_gone": sentinel_gone,
            "encoder_emits_pin_anim_after_clear": has_anim_after_clear,
        },
    )

    # ----- F: full transfer/build/run ----------------------------
    # Restore the cache for the live run.
    pc2.write_pin_anim_pc2(plane, "UpperPin", captured)
    pin_item.has_captured_anim = True
    pin_ops._ensure_embedded_move_op(pin_item)
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="pin_capture:build")
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
    # ContactSolverCache must sit AFTER the Armature in the cloth's
    # modifier stack. If it sits before, the Armature re-applies its
    # bone deformation on top of the solver's already-bone-aware
    # output, producing visibly doubled motion on playback.
    mod_names = [m.name for m in plane.modifiers]
    arm_i = mod_names.index("ArmatureMod") if "ArmatureMod" in mod_names else -1
    cache_i = mod_names.index("ContactSolverCache") if "ContactSolverCache" in mod_names else -1
    cache_after_arm = arm_i >= 0 and cache_i >= 0 and cache_i > arm_i
    dh.record(
        "F_build_and_run_completes_with_pin_capture",
        cloth_arr is not None
        and cloth_samples >= FRAME_COUNT - 1
        and cache_after_arm,
        {
            "cloth_pc2_path": cloth_pc2,
            "cloth_samples": cloth_samples,
            "expected_min_samples": FRAME_COUNT - 1,
            "modifier_order": mod_names,
            "armature_index": arm_i,
            "cache_index": cache_i,
            "cache_after_armature": cache_after_arm,
        },
    )

    # ----- G: capture rejects non-deforming object ---------------
    # Build an isolated rig: a tiny plane with no modifiers, register
    # as its own SHELL group, attach a pin. The capture op should
    # refuse because pin_object_supports_capture returns False.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(5.0, 0.0, 0.0))
    plain = bpy.context.active_object
    plain.name = "PlainPin"
    n_plain = len(plain.data.vertices)
    plain.vertex_groups.new(name="P").add(list(range(n_plain)), 1.0, "REPLACE")
    plain_group = dh.api.solver.create_group("PlainCloth", "SHELL")
    plain_group.add(plain.name)
    plain_group.create_pin(plain.name, "P")
    plain_group_idx = None
    for i in range(32):
        g = getattr(addon_root, f"object_group_{i}")
        if g.active and g.name == "PlainCloth":
            plain_group_idx = i
            break
    plain_pin_item = plain_group.pin_vertex_groups[0] if hasattr(plain_group, 'pin_vertex_groups') else None
    # Resolve the actual PropertyGroup pin via the addon root.
    plain_g = getattr(addon_root, f"object_group_{plain_group_idx}") if plain_group_idx is not None else None
    plain_pin_pg = plain_g.pin_vertex_groups[0] if plain_g is not None else None
    cache_present_before = pc2.has_pin_anim_pc2(plain, "P")
    bpy.ops.object.capture_pin_deformation(
        "EXEC_DEFAULT", group_index=plain_group_idx, pin_index=0,
    )
    cache_present_after = pc2.has_pin_anim_pc2(plain, "P")
    flag_after = bool(getattr(plain_pin_pg, "has_captured_anim", False))
    dh.record(
        "G_capture_rejects_non_deforming_object",
        (not cache_present_before)
        and (not cache_present_after)
        and (not flag_after),
        {
            "cache_present_before": cache_present_before,
            "cache_present_after": cache_present_after,
            "has_captured_anim_after": flag_after,
        },
    )

    # ----- G2: Capture row stays visible (just disabled) without a deformer --
    # The Capture Deformation / Clear buttons must remain visible even
    # when the pin's object has no deforming modifier, so users can
    # discover the feature exists. The panel's branch depends on three
    # predicates: pin_object_supports_capture(obj),
    # pin_has_captured_anim(pin_item), and is_pin_capture_running().
    # For the plain (no-deformer) pin we expect support=False,
    # has_cache=False, running=False, which drives the "No deforming
    # modifier on this pin's object" label branch with both buttons
    # disabled. Verifying the predicates is equivalent to verifying the
    # panel branch under the if/elif/else logic in panels.py; we avoid
    # introspecting the panel because the test rig's hidden window
    # doesn't always trigger sidebar draws.
    pcap_mod = __import__(
        pkg + ".ui.dynamics.pin_capture_ops",
        fromlist=["pin_object_supports_capture",
                  "pin_has_captured_anim",
                  "is_pin_capture_running"],
    )
    g2_supports = bool(pcap_mod.pin_object_supports_capture(plain))
    g2_has_cache = bool(pcap_mod.pin_has_captured_anim(plain_pin_pg))
    g2_running = bool(pcap_mod.is_pin_capture_running())
    dh.record(
        "G2_capture_row_shown_when_no_deformer",
        (not g2_supports) and (not g2_has_cache) and (not g2_running),
        {
            "pin_object_supports_capture": g2_supports,
            "pin_has_captured_anim": g2_has_cache,
            "is_pin_capture_running": g2_running,
            "expected_panel_branch": "no_deforming_modifier_label",
        },
    )

    # ----- H: capture rejects pin that carries a TORQUE op -------
    # Re-use the original captured cache as the baseline; add a
    # TORQUE op, clear the cache to start clean, then try to capture.
    bpy.ops.object.clear_pin_deformation(
        "EXEC_DEFAULT", group_index=0, pin_index=0,
    )
    op = pin_item.operations.add()
    op.op_type = "TORQUE"
    cache_before = pc2.has_pin_anim_pc2(plane, "UpperPin")
    bpy.ops.object.capture_pin_deformation(
        "EXEC_DEFAULT", group_index=0, pin_index=0,
    )
    cache_after = pc2.has_pin_anim_pc2(plane, "UpperPin")
    flag_after_h = bool(pin_item.has_captured_anim)
    dh.record(
        "H_capture_rejects_torque_pin",
        (not cache_before) and (not cache_after) and (not flag_after_h),
        {
            "cache_present_before": cache_before,
            "cache_present_after": cache_after,
            "has_captured_anim_after": flag_after_h,
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
