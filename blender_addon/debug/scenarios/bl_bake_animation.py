# File: scenarios/bl_bake_animation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Bake Animation coverage for both MESH and CURVE paths.
#
# The ``solver.bake_all_animation`` operator transforms a fetched PC2
# cache into native Blender keyframes so the user can render or export
# the simulation without the addon. The two paths produce very
# different artifacts:
#
#   * MESH: one shape key per fetched PC2 frame (named
#     ``ContactSolverBake_<scene_frame:05d>``). Positions are written
#     via ``foreach_set`` on the shape key's ``co`` property, then the
#     shape key's ``value`` is keyed 0, 1, 0 across three consecutive
#     scene frames with CONSTANT interpolation so each shape key is
#     "on" for exactly its frame. The fcurves live on
#     ``obj.data.shape_keys.animation_data.action``; vertex coords are
#     never keyframed directly. The MESH_CACHE modifier and the PC2
#     file are removed so playback uses the baked keys instead.
#   * CURVE (ROD): per-control-point ``co`` / ``handle_left`` /
#     ``handle_right`` keyframes via ``bp.keyframe_insert`` because
#     shape keys do not apply to curves. The fcurves land on
#     ``obj.data.animation_data.action`` (the curve's data block,
#     since the property paths are rooted at
#     ``splines[i].bezier_points[j]`` which lives on the data block).
#
# The driver bypasses the modal operator (modal TIMER events cannot
# fire while the driver thread is held) and calls the bake_ops
# helpers directly: ``_build_queue`` -> ``_check_mesh_shape_keys`` ->
# ``_start_job`` -> loop ``_tick_job`` -> ``_finalize_job``. This is
# the same code path the modal exercises on each TIMER tick, just
# driven from the driver thread.
#
# Subtests:
#   A. mesh_bake_uses_shape_keys
#         Standard pinned-plane setup, run + fetch + drain, then
#         drive the bake. Assert ``key_blocks`` has one entry per
#         fetched PC2 frame named ``ContactSolverBake_<frame>``,
#         the shape-keys action carries a ``key_blocks["<sk>"].value``
#         fcurve per shape key with CONSTANT interpolation, the
#         MESH_CACHE modifier and PC2 file are gone, and stepping
#         the timeline to a non-rest frame moves vertex positions.
#   B. curve_bake_keyframes_cvs
#         Bezier curve in a ROD group, run + fetch + drain, then
#         drive the bake. Assert
#         ``obj.data.animation_data.action.fcurves`` carries
#         ``splines[i].bezier_points[j].co`` / ``handle_left`` /
#         ``handle_right`` entries, one keyframe per fetched frame,
#         and stepping the timeline animates the CVs.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import json
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


def _wipe_scene_state():
    # Remove every connection / group / assigned object so the second
    # subtest starts from a clean slate. Also disconnect the addon
    # client so the second connect_local doesn't see a stale session.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _drive_bake_to_completion(bake_ops, *, timeout=60.0):
    # The modal operator cannot tick while we hold the driver thread,
    # so call the same helpers it would call. This is exactly the
    # production code path, just without the TIMER plumbing.
    queue = bake_ops._build_queue(bpy.context, group_index=None)
    blockers = bake_ops._check_mesh_shape_keys(queue)
    if blockers:
        raise RuntimeError(f"shape-key blockers before bake: {blockers}")
    ok, err = bake_ops._start_job(queue, kind="all")
    if not ok:
        raise RuntimeError(f"_start_job rejected the queue: {err!r}")
    deadline = time.time() + timeout
    while bake_ops._tick_job(budget_ms=40):
        if time.time() > deadline:
            raise TimeoutError("bake did not finish within timeout")
    n_objs, n_frames = bake_ops._finalize_job(bpy.context)
    bake_ops._reset_job()
    return queue, n_objs, n_frames


try:
    dh = DriverHelpers(pkg, result)
    bake_ops = __import__(pkg + ".ui.dynamics.bake_ops",
                          fromlist=["_build_queue", "_check_mesh_shape_keys",
                                    "_start_job", "_tick_job",
                                    "_finalize_job", "_reset_job"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    pc2_mod = __import__(pkg + ".core.pc2",
                         fromlist=["object_pc2_key", "get_pc2_path",
                                   "MODIFIER_NAME", "_curve_cache"])
    api_mod = __import__(pkg + ".ops.api",
                         fromlist=["_raw_create_pin"])

    # =========================================================
    # Subtest A: MESH bake uses shape keys
    # =========================================================
    dh.log("subtest_A_setup")
    plane = dh.reset_scene_to_pinned_plane(name="BakeMesh")
    blend_path_a = os.path.join(os.path.dirname(PROBE_DIR), "bake_mesh.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path_a)
    root = dh.configure_state(project_name="bake_animation_mesh",
                              frame_count=6)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="bake_animation:mesh:build",
    ))
    deadline = time.time() + 90.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)
    if dh.facade.engine.state.solver.name == "FAILED":
        raise RuntimeError(
            f"mesh build failed: {dh.facade.engine.state.error!r}"
        )

    dh.run_and_wait(timeout=90.0)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("subtest_A_fetched")

    # Capture pre-bake state for comparison + assertions.
    pc2_path_pre_a = dh.find_pc2_for(plane)
    rest_v0_x = float(plane.data.vertices[0].co.x)
    expected_n_frames_a = 0
    pc2_frame_count = __import__(pkg + ".core.pc2",
                                 fromlist=["read_pc2_frame_count"]
                                ).read_pc2_frame_count
    if pc2_path_pre_a and os.path.isfile(pc2_path_pre_a):
        expected_n_frames_a = pc2_frame_count(pc2_path_pre_a)

    # Drive the bake.
    _drive_bake_to_completion(bake_ops, timeout=60.0)
    dh.log(f"subtest_A_baked frames={expected_n_frames_a}")

    sk_block = plane.data.shape_keys
    sk_names = [kb.name for kb in sk_block.key_blocks] if sk_block else []
    expected_sk_names = [
        f"ContactSolverBake_{i:05d}" for i in range(1, expected_n_frames_a + 1)
    ]
    # Expected: Basis (auto-created on first shape_key_add) + one per frame.
    sk_names_match = all(name in sk_names for name in expected_sk_names)

    # Inspect the shape-keys action for value fcurves with CONSTANT
    # interpolation. The action lives on shape_keys.animation_data, not
    # on the object's animation_data.
    sk_action = None
    if sk_block is not None and sk_block.animation_data is not None:
        sk_action = sk_block.animation_data.action
    fcurve_paths = []
    constant_ok = True
    if sk_action is not None:
        utils_mod = __import__(pkg + ".core.utils",
                               fromlist=["_get_fcurves"])
        for fc in utils_mod._get_fcurves(sk_action):
            fcurve_paths.append(fc.data_path)
            for kp in fc.keyframe_points:
                if kp.interpolation != "CONSTANT":
                    constant_ok = False
                    break
            if not constant_ok:
                break
    expected_fc_paths = {
        f'key_blocks["{name}"].value' for name in expected_sk_names
    }
    fc_paths_match = expected_fc_paths.issubset(set(fcurve_paths))

    # Verify the MESH_CACHE modifier was removed and the PC2 file is gone.
    mod_removed = (
        plane.modifiers.get(pc2_mod.MODIFIER_NAME) is None
    )
    pc2_removed = not (pc2_path_pre_a and os.path.isfile(pc2_path_pre_a))

    # Verify timeline playback uses the baked shape keys, not a modifier.
    # Step to a non-rest frame and inspect the evaluated mesh's vertex 0.
    bpy.context.scene.frame_set(2)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = plane.evaluated_get(depsgraph)
    coords = [0.0] * (len(eval_obj.data.vertices) * 3)
    eval_obj.data.vertices.foreach_get("co", coords)
    v0_x_at_frame = coords[0]
    bpy.context.scene.frame_set(1)

    dh.record(
        "A_mesh_bake_uses_shape_keys",
        expected_n_frames_a > 0
        and sk_names_match
        and len(expected_sk_names) > 0
        and fc_paths_match
        and constant_ok
        and mod_removed
        and pc2_removed
        and abs(v0_x_at_frame - rest_v0_x) > 1e-6,
        {
            "expected_n_frames": int(expected_n_frames_a),
            "sk_names": sk_names,
            "expected_sk_names": expected_sk_names,
            "fcurve_paths": sorted(fcurve_paths),
            "constant_ok": constant_ok,
            "mod_removed": mod_removed,
            "pc2_removed": pc2_removed,
            "rest_v0_x": rest_v0_x,
            "v0_x_at_frame": float(v0_x_at_frame),
            "pc2_path_pre": pc2_path_pre_a,
        },
    )

    # =========================================================
    # Subtest B: CURVE bake keyframes per-CV co / handles
    # =========================================================
    dh.log("subtest_B_setup")
    # Disconnect the addon client + wipe the scene before standing up
    # the curve setup. The state machine retains build / project info
    # across scene resets unless we explicitly reconnect.
    try:
        dh.com.disconnect()
    except Exception:
        pass
    _wipe_scene_state()

    # Build a 4-CV bezier curve along +X.
    n_cv = 4
    curve_data = bpy.data.curves.new(name="BakeRod", type="CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(count=n_cv - 1)  # default add gives one CV
    for i, bp in enumerate(spline.bezier_points):
        bp.co = (i * 0.1, 0.0, 0.0)
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    curve_obj = bpy.data.objects.new("BakeRodObj", curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Pin control point 0 via the curve's _pin_<name> custom property
    # (curves have no native vertex_groups). Then register the pin
    # entry through _raw_create_pin.
    pinned_cv = 0
    curve_obj["_pin_AllPin"] = json.dumps([pinned_cv])

    blend_path_b = os.path.join(os.path.dirname(PROBE_DIR), "bake_curve.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path_b)

    root_b = dh.configure_state(project_name="bake_animation_curve",
                                frame_count=6)
    rod_group = dh.api.solver.create_group("Rod", "ROD")

    # Resolve the new group's slot index. The Cloth group from subtest
    # A is still active in slot 0 (its assigned objects were stripped
    # by Bake but the group wrapper persists), so the Rod group lands
    # at the next available slot.
    mcp_groups_mod = __import__(pkg + ".mcp.handlers.group",
                                fromlist=["get_group_index_by_uuid"])
    rod_group_index = mcp_groups_mod.get_group_index_by_uuid(rod_group._uuid)

    # Curves can only be added through the UI op (the API path's
    # MCP wrapper rejects non-mesh).
    bpy.ops.object.select_all(action="DESELECT")
    curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.add_objects_to_group(group_index=rod_group_index)

    api_mod._raw_create_pin(rod_group._uuid, curve_obj.name, "AllPin")

    data_bytes_b, param_bytes_b = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root_b.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes_b, param=param_bytes_b,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="bake_animation:curve:build",
    ))
    deadline = time.time() + 180.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)
    if dh.facade.engine.state.solver.name == "FAILED":
        raise RuntimeError(
            f"curve build failed: {dh.facade.engine.state.error!r}"
        )

    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("subtest_B_fetched")

    # Pin the timeline at the rest frame before capturing rest CV
    # positions. After fetch_and_drain populates the curve cache, the
    # frame_change_pre handler animates CVs whenever the current frame
    # changes, so reading bezier_points[].co straight after fetch may
    # return a displaced pose rather than the rest pose.
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    rest_cv0_co = tuple(curve_obj.data.splines[0].bezier_points[pinned_cv].co)
    rest_cv1_co = tuple(curve_obj.data.splines[0].bezier_points[1].co)

    # Resolve PC2 path + frame count via curve helpers (curves have no
    # MESH_CACHE modifier; they use _curve_cache and the on-disk PC2).
    curve_key = pc2_mod.object_pc2_key(curve_obj)
    pc2_path_b = pc2_mod.get_pc2_path(curve_key)
    expected_n_frames_b = (
        pc2_frame_count(pc2_path_b) if os.path.isfile(pc2_path_b) else 0
    )
    # Snapshot PC2 contents BEFORE bake -- ``_finalize_job`` deletes the
    # PC2 file as part of swapping playback over to the baked keys, so
    # later subtests cannot read it.
    pc2_arr_pre_bake = (
        dh.read_pc2(pc2_path_b) if os.path.isfile(pc2_path_b) else None
    )

    _drive_bake_to_completion(bake_ops, timeout=60.0)
    dh.log(f"subtest_B_baked frames={expected_n_frames_b}")

    # Inspect the curve's data-block action for per-CV fcurves. The
    # bake's bp.keyframe_insert("co") writes fcurves rooted at
    # ``obj.data.splines[i].bezier_points[j]``, so the action lives
    # on ``obj.data.animation_data.action`` rather than
    # ``obj.animation_data.action`` (object-level transforms only).
    # ``_finalize_job`` confirms this: it explicitly calls
    # ``set_linear_interpolation(obj.data.animation_data.action)``.
    obj_action = None
    if curve_obj.data.animation_data is not None:
        obj_action = curve_obj.data.animation_data.action
    cv_fc_paths = set()
    if obj_action is not None:
        utils_mod = __import__(pkg + ".core.utils",
                               fromlist=["_get_fcurves"])
        for fc in utils_mod._get_fcurves(obj_action):
            cv_fc_paths.add(fc.data_path)

    # We expect one of each (co, handle_left, handle_right) per
    # bezier_point. The bake loops over every CV, so every spline
    # index 0 / point indices 0..n_cv-1 should appear.
    expected_curve_paths = set()
    for j in range(n_cv):
        prefix = f"splines[0].bezier_points[{j}]"
        expected_curve_paths.add(f"{prefix}.co")
        expected_curve_paths.add(f"{prefix}.handle_left")
        expected_curve_paths.add(f"{prefix}.handle_right")
    cv_paths_match = expected_curve_paths.issubset(cv_fc_paths)

    # Verify each fcurve has the expected number of keyframes (one
    # per fetched frame) by sampling a representative fcurve.
    sample_path = f"splines[0].bezier_points[1].co"
    sample_n_keys = 0
    if obj_action is not None:
        utils_mod = __import__(pkg + ".core.utils",
                               fromlist=["_get_fcurves"])
        for fc in utils_mod._get_fcurves(obj_action):
            if fc.data_path == sample_path:
                sample_n_keys = max(sample_n_keys, len(fc.keyframe_points))
    keys_per_frame_ok = (
        expected_n_frames_b > 0 and sample_n_keys >= expected_n_frames_b
    )

    # Verify timeline playback drives CV positions through the baked
    # fcurves. With no gravity / wind / contact and a pinned root CV,
    # the rod barely moves during the sim, so checking that CVs
    # deviate from rest is unreliable. Instead, confirm playback by
    # comparing the post-bake CV pose at a later scene frame against
    # the corresponding PC2 frame's CV layout: the bake's source
    # of truth. CV layout per CP is [handle_left, co, handle_right],
    # so CP j's "co" lives at slot ``j*3 + 1``.
    sample_frame = max(2, min(expected_n_frames_b, 4))
    pc2_arr = pc2_arr_pre_bake
    bpy.context.scene.frame_set(sample_frame)
    bpy.context.view_layer.update()
    cv1_now = tuple(curve_obj.data.splines[0].bezier_points[1].co)
    cv1_expected = (
        tuple(float(x) for x in pc2_arr[sample_frame - 1, 1 * 3 + 1])
        if pc2_arr is not None and pc2_arr.shape[0] >= sample_frame
        else None
    )
    cv1_matches_pc2 = (
        cv1_expected is not None
        and all(abs(cv1_now[i] - cv1_expected[i]) < 1e-3 for i in range(3))
    )
    bpy.context.scene.frame_set(1)

    dh.record(
        "B_curve_bake_keyframes_cvs",
        expected_n_frames_b > 0
        and obj_action is not None
        and cv_paths_match
        and keys_per_frame_ok
        and cv1_matches_pc2,
        {
            "expected_n_frames": int(expected_n_frames_b),
            "obj_action_name": obj_action.name if obj_action else None,
            "cv_fc_paths": sorted(cv_fc_paths),
            "expected_curve_paths": sorted(expected_curve_paths),
            "sample_n_keys": int(sample_n_keys),
            "sample_frame": int(sample_frame),
            "rest_cv1_co": [float(x) for x in rest_cv1_co],
            "cv1_now": [float(x) for x in cv1_now],
            "cv1_expected": (
                list(cv1_expected) if cv1_expected is not None else None
            ),
            "rest_cv0_co": [float(x) for x in rest_cv0_co],
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    # Two pipeline cycles (mesh + curve) plus two bakes, so allow a
    # generous timeout. Match the tier of bl_pin_rod_curve which does
    # one curve cycle with similar wall-clock cost.
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
