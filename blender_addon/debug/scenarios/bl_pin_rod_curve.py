# File: scenarios/bl_pin_rod_curve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# ROD curve pinning end-to-end. Verifies that a Bezier curve
# assigned to a ROD group encodes as edges (rod connectivity, no
# faces), produces a PC2 whose vertex count matches the curve's
# CV layout, and that a pinned control point holds its rest pose
# across every fetched frame.
#
# Coverage targets:
#   A. encoded_as_edges:        encode_obj output for the curve
#                               carries an "edge" array and no
#                               "face" array.
#   B. pc2_has_cv_positions:    read_pc2 returns shape
#                               (n_samples, n_cvs, 3) where n_cvs
#                               equals get_curve_cv_count(curve).
#   C. pinned_cv_stays_at_rest: across all sampled frames, the
#                               pinned control point's "co" slot
#                               in the CV layout equals the rest
#                               pose within tolerance.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


# Bezier curve with N control points laid out along +X. With N=4
# and t = {0, 1/3, 2/3, 1} per segment, sample_curve produces
# 1 + 3*(N-1) = 10 sampled vertices, and CV layout has 3*N = 12
# slots ([hl, co, hr] per control point). The first CV (index 0)
# is pinned, which after map_cp_pins_to_sampled clamps sampled
# vertex 0 in the solver.
_N_CV = 4


_DRIVER_BODY = r"""
import os
import pickle
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
N_CV = <<N_CV>>


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Wipe scene -- start from a clean slate; the helper's
    # reset_scene_to_pinned_plane is mesh-only, so we do the wipe
    # by hand and build a curve instead.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    curve_data = bpy.data.curves.new(name="Rod", type="CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(count=N_CV - 1)  # default add gives 1 cv
    for i, bp in enumerate(spline.bezier_points):
        bp.co = (i * 0.1, 0.0, 0.0)
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    curve_obj = bpy.data.objects.new("RodObj", curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Capture rest CV layout before any solver activity.
    curve_rod = __import__(pkg + ".core.curve_rod",
                           fromlist=["get_curve_rest_cvs",
                                     "get_curve_cv_count"])
    rest_cvs = curve_rod.get_curve_rest_cvs(curve_obj)
    n_cvs_expected = curve_rod.get_curve_cv_count(curve_obj)
    # Pinned CV's co slot in modifier layout: cvs[i*3 + 1] for
    # control point i. We pin control point 0, so its co lives
    # at slot 1 in each PC2 sample.
    pinned_cv = 0
    pinned_co_slot = pinned_cv * 3 + 1
    rest_co = list(rest_cvs[pinned_co_slot])

    # Install the pin via the curve's _pin_<name> custom property
    # (curves have no native vertex_groups), then register the
    # group entry through _raw_create_pin which matches the API
    # path's bookkeeping (UUID + vg_hash) without the MESH guard.
    import json as _json
    curve_obj["_pin_AllPin"] = _json.dumps([pinned_cv])

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "rod_curve.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    dh.log(f"saved_blend filepath={bpy.data.filepath!r}")

    root = dh.configure_state(project_name="pin_rod_curve", frame_count=6)

    rod_group = dh.api.solver.create_group("Rod", "ROD")

    # Add via the UI op (selection-driven); the MCP-wrapped api
    # add() rejects non-mesh objects, but the underlying op accepts
    # curves when group.object_type == "ROD".
    bpy.ops.object.select_all(action="DESELECT")
    curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.add_objects_to_group(group_index=0)

    # Curve pin registration: bypass MutationService (it explicitly
    # rejects non-mesh) and call the underlying writer directly.
    api_mod = __import__(pkg + ".ops.api", fromlist=["_raw_create_pin"])
    api_mod._raw_create_pin(rod_group._uuid, curve_obj.name, "AllPin")

    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    # ----- A: inspect encode_obj output before transferring -------
    # encode_obj returns pickled bytes; unpickle to introspect the
    # group's object_info dict and confirm rod connectivity (edge
    # array present, face key absent) for the CURVE in the ROD group.
    raw_data = encoder_mesh.encode_obj(bpy.context)
    decoded = pickle.loads(raw_data)
    rod_entry = None
    for entry in decoded:
        if entry.get("type") == "ROD":
            rod_entry = entry
            break
    rod_obj_info = None
    if rod_entry is not None:
        for info in rod_entry.get("object", []):
            if info.get("name") == curve_obj.name:
                rod_obj_info = info
                break
    edge_arr = rod_obj_info.get("edge") if rod_obj_info else None
    has_face_key = (rod_obj_info is not None and "face" in rod_obj_info)
    edge_count = int(edge_arr.shape[0]) if edge_arr is not None else 0
    pin_arr = rod_obj_info.get("pin", []) if rod_obj_info else []
    dh.record(
        "A_encoded_as_edges",
        rod_obj_info is not None
        and edge_arr is not None
        and edge_count > 0
        and not has_face_key
        and len(pin_arr) > 0,
        {
            "edge_count": edge_count,
            "has_face_key": has_face_key,
            "pin_indices": list(pin_arr) if pin_arr is not None else [],
            "vertex_count": (
                int(rod_obj_info["vert"].shape[0])
                if rod_obj_info and "vert" in rod_obj_info else 0
            ),
        },
    )

    # ----- transfer + run + fetch ---------------------------------
    # We already encoded once for assertion A; reuse those bytes
    # so encode is a single pass and the build sees the same blob
    # we asserted on.
    data_bytes = raw_data
    encode_param_mod = __import__(pkg + ".core.encoder.params",
                                  fromlist=["encode_param"])
    param_bytes = encode_param_mod.encode_param(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="pin_rod_curve:build",
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
            f"build failed: {dh.facade.engine.state.error!r}"
        )
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("fetched")

    # ----- B: PC2 has CV-shaped samples ---------------------------
    # Curves don't get a MESH_CACHE modifier (Blender doesn't honor it
    # on curves); the addon's persistent handlers stream CV positions
    # from the PC2 file directly. So look the path up via UUID instead
    # of `find_pc2_for` (which scans for MESH_CACHE).
    client_mod_local = __import__(pkg + ".core.client",
                                  fromlist=["object_pc2_key"])
    pc2_mod = __import__(pkg + ".core.pc2", fromlist=["get_pc2_path"])
    pc2_path = pc2_mod.get_pc2_path(client_mod_local.object_pc2_key(curve_obj))
    pc2_arr = None
    pc2_shape = None
    if pc2_path and os.path.isfile(pc2_path):
        pc2_arr = dh.read_pc2(pc2_path)
        pc2_shape = list(pc2_arr.shape)
    dh.record(
        "B_pc2_has_cv_positions",
        pc2_arr is not None
        and pc2_arr.ndim == 3
        and pc2_arr.shape[1] == n_cvs_expected
        and pc2_arr.shape[0] >= 2,
        {
            "pc2_path": pc2_path,
            "pc2_shape": pc2_shape,
            "n_cvs_expected": int(n_cvs_expected),
        },
    )

    # ----- C: pinned CV stays at rest across all samples ----------
    # Skip sample 0 (rest pose, trivially equal) and inspect every
    # post-rest frame. The fit is least-squares, so we use a small
    # but non-zero tolerance.
    tol = 1e-3
    if pc2_arr is not None and pc2_arr.shape[1] >= pinned_co_slot + 1:
        per_frame = []
        max_err = 0.0
        rest_np = np.asarray(rest_co, dtype=np.float32)
        for n in range(1, pc2_arr.shape[0]):
            actual = pc2_arr[n, pinned_co_slot]
            err = float(np.max(np.abs(actual - rest_np)))
            per_frame.append({"frame": n, "err": err,
                              "actual": [float(x) for x in actual]})
            if err > max_err:
                max_err = err
        stays = max_err <= tol
    else:
        per_frame = []
        max_err = float("inf")
        stays = False
    dh.record(
        "C_pinned_cv_stays_at_rest",
        stays,
        {
            "pinned_cv": pinned_cv,
            "pinned_co_slot": pinned_co_slot,
            "rest_co": [float(x) for x in rest_co],
            "max_err": max_err,
            "tolerance": tol,
            "per_frame": per_frame[:6],
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
        .replace("<<N_CV>>", str(_N_CV))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
