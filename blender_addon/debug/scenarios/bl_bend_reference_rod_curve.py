# File: scenarios/bl_bend_reference_rod_curve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Exercises the per-object "reference rest angle" feature for CURVE-based
# ROD groups end-to-end on a no-GPU host (macOS), using the CUDA-free
# emulated solver.
#
# Curve rods ship their rod vertices by sampling the curve at the
# control-point level (curve modifiers are not sampled), so the bending
# reference is also a curve, sampled the same way. This scenario builds a
# straight Bezier rod plus a copy with moved control points, points the
# rod's Reference Rest Angle at the copy, and asserts:
#
#   A. validate_accepts_copy        - the curve validator accepts a
#                                     same-structure copy.
#   B. validate_rejects_count       - it rejects a copy with a different
#                                     control-point count.
#   C. encoder_ships_reference      - the encoder samples the reference
#                                     curve into `bend_rest_vert`, which
#                                     matches sample_curve(reference) and
#                                     differs from the straight source.
#   D. encoder_rejects_bad_reference- a deviating reference fails upload.
#   E. build_succeeds_with_reference- the solver loads the reference and
#                                     builds without panic.
#   F. run_completes                - the emulated solver advances every
#                                     frame with the reference in play.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

KNOBS = {"PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import json as _json
import traceback

import numpy as np
from mathutils import Matrix

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 5
N_CV = 6


def _make_curve(name, n_cv, bend):
    data = bpy.data.curves.new(name=name + "Data", type="CURVE")
    data.dimensions = "3D"
    spline = data.splines.new(type="BEZIER")
    spline.bezier_points.add(count=n_cv - 1)  # default gives 1 point
    for i, bp in enumerate(spline.bezier_points):
        x = i * 0.1
        bp.co = (x, 0.0, bend * x * x)
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def _find_rod_entry(decoded, uuid, name):
    for grp in decoded:
        if grp.get("type") != "ROD":
            continue
        for o in grp.get("object", []):
            if o.get("uuid") == uuid or o.get("name") == name:
                return o
    return None


def _raw_group(dh, name):
    for g in dh.groups.iterate_active_object_groups(bpy.context.scene):
        if g.name == name:
            return g
    return None


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    uuidreg = __import__(pkg + ".core.uuid_registry",
                         fromlist=["get_or_create_object_uuid"])
    utils_pkg = __import__(pkg + ".core.utils",
                           fromlist=["validate_bend_reference"])
    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    curve_rod = __import__(pkg + ".core.curve_rod", fromlist=["sample_curve"])
    mutation_mod = __import__(pkg + ".core.mutation", fromlist=["_raw_create_pin"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    rod = _make_curve("BendRefCurve", N_CV, 0.0)        # straight source
    ref = _make_curve("BendRefCurve_Reference", N_CV, 0.6)  # bent copy
    badref = _make_curve("BendRefCurve_Bad", N_CV - 2, 0.6)  # wrong CV count

    # Pin control point 0 (curves have no vertex groups; use _pin_<name>).
    rod["_pin_AllPin"] = _json.dumps([0])

    rod_uuid = uuidreg.get_or_create_object_uuid(rod)
    ref_uuid = uuidreg.get_or_create_object_uuid(ref)
    badref_uuid = uuidreg.get_or_create_object_uuid(badref)

    blend_path = __import__("os").path.join(
        __import__("os").path.dirname(PROBE_DIR), "bend_reference_rod_curve.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    root = dh.configure_state(
        project_name="bend_reference_rod_curve",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )

    rod_group = dh.api.solver.create_group("Rod", "ROD")
    # The API add() rejects non-mesh; the UI op accepts curves for ROD.
    bpy.ops.object.select_all(action="DESELECT")
    rod.select_set(True)
    bpy.context.view_layer.objects.active = rod
    bpy.ops.object.add_objects_to_group(group_index=0)
    mutation_mod._raw_create_pin(rod_group._uuid, rod.name, "AllPin")

    grp = _raw_group(dh, "Rod")
    grp.bend_rest_from_reference = True
    assigned = None
    for a in grp.assigned_objects:
        if a.uuid == rod_uuid:
            assigned = a
            break
    if assigned is None:
        raise RuntimeError("assigned object for curve rod not found in group")
    assigned.bend_ref_enable = True
    assigned.bend_ref_uuid = ref_uuid
    dh.log("reference_configured")

    # --- A/B: curve validator (sampled count + edges) ---
    validate = utils_pkg.validate_bend_reference
    ok_a, msg_a = validate(rod, ref, bpy.context, "ROD")
    dh.record("A_validate_accepts_copy", ok_a, {"msg": msg_a})
    ok_b, msg_b = validate(rod, badref, bpy.context, "ROD")
    dh.record("B_validate_rejects_count", not ok_b, {"msg": msg_b})

    # --- C: encoder samples the reference curve into bend_rest_vert ---
    data_bytes, _param, _dh, _ph = encoder_pkg.prepare_upload(bpy.context)
    decoded = DriverHelpers.decode_addon_blob(data_bytes)
    entry = _find_rod_entry(decoded, rod_uuid, rod.name)
    if entry is None:
        raise RuntimeError("curve rod object not found in encoded scene")
    vert = np.asarray(entry.get("vert"), dtype=np.float64)
    has_brv = entry.get("bend_rest_vert") is not None
    brv = np.asarray(entry.get("bend_rest_vert"), dtype=np.float64) if has_brv else None
    ref_sampled, _e, _m = curve_rod.sample_curve(ref, Matrix.Identity(4))
    ref_sampled = np.asarray(ref_sampled, dtype=np.float64)
    shape_ok = bool(has_brv and brv.shape == vert.shape == ref_sampled.shape)
    matches_ref = bool(shape_ok and np.allclose(brv, ref_sampled, atol=1e-4))
    differs_src = bool(shape_ok and float(np.max(np.abs(brv - vert))) > 0.1)
    dh.record(
        "C_encoder_ships_reference",
        shape_ok and matches_ref and differs_src,
        {
            "has_bend_rest_vert": has_brv,
            "shape_ok": shape_ok,
            "matches_ref": matches_ref,
            "differs_src": differs_src,
        },
    )

    # --- D: deviating reference fails the upload ---
    assigned.bend_ref_uuid = badref_uuid
    raised = False
    emsg = ""
    try:
        encoder_pkg.prepare_upload(bpy.context)
    except ValueError as exc:
        raised = True
        emsg = str(exc)
    dh.record("D_encoder_rejects_bad_reference", raised, {"msg": emsg[:200]})
    assigned.bend_ref_uuid = ref_uuid  # restore good reference

    # --- E/F: end-to-end build + run ---
    data_bytes, param_bytes, _dh2, _ph2 = encoder_pkg.prepare_upload(bpy.context)
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, "bend-ref-rod-curve:build",
                      timeout=120.0)
    solver_name = dh.facade.engine.state.solver.name
    dh.record(
        "E_build_succeeds_with_reference",
        solver_name in ("READY", "RESUMABLE"),
        {"solver": solver_name},
    )

    saw_running = dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    frame = dh.facade.engine.state.frame
    dh.record(
        "F_run_completes",
        bool(saw_running) and frame >= FRAME_COUNT,
        {"saw_running": bool(saw_running), "frame": int(frame)},
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
