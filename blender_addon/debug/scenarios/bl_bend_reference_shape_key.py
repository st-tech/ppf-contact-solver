# File: scenarios/bl_bend_reference_shape_key.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Exercises the shape-key source of the per-object "reference rest angle"
# end-to-end through the production pipeline, using the CPU backend.
#
# A shell whose Reference Source is Shape Key takes its hinge rest angles
# from one of its OWN shape keys, read as the key's stored shape (not its
# Value slider, and before modifiers), instead of from a separate reference
# object. The scene is a flat grid with a flat Basis and a curled "Curled"
# key, and the checks assert:
#
#   A. validate_accepts_key          - an existing key of a mesh is accepted.
#   B. validate_rejects_missing_key  - a name the mesh does not carry is
#                                      refused, naming it.
#   C. validate_rejects_curve        - a curve object is refused: a
#                                      shape-key rest angle needs a mesh.
#   D. encoder_ships_shape_key       - with "Curled" at Value 0 the object
#                                      ships flat, and `bend_rest_vert` is
#                                      the curled key's stored shape.
#   E. basis_rest_under_deformed_mix - the issue's workflow: "Curled" at
#                                      Value 1 ships the object curled, and
#                                      picking Basis ships the flat Basis
#                                      as the rest shape.
#   F. encoder_rejects_renamed_key   - a stored key that was renamed fails
#                                      the upload with a ValueError naming
#                                      the stored key.
#   G. object_without_reference_refused - with the source back on Reference
#                                      Object and no object picked, the
#                                      upload is refused naming the object:
#                                      a From Reference Geometry group has no
#                                      other rest angle to fall back on.
#   J. panel_draws_every_state       - the reference box draws the
#                                      shape-key field, an ERROR line for an
#                                      unset and a missing key, the object
#                                      picker for Reference Object, and no
#                                      per-object enable; the group box draws
#                                      the reference controls under From
#                                      Reference Geometry and Bend Plasticity
#                                      under Flat, never both.
#                                      Recorded before H so the build runs
#                                      last.
#   K. retired_checkbox_migrates     - a file saved with the retired group
#                                      checkbox on opens as From Reference
#                                      Geometry, one saved with it off keeps
#                                      its Rest Angle, and both retired keys
#                                      are dropped. Recorded before H.
#   H. build_succeeds_with_shape_key - the solver loads the buffer and
#                                      builds.
#   I. run_completes                 - the solver advances every frame.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it against a CPU build.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 5


def _key_co(obj, name):
    block = obj.data.shape_keys.key_blocks[name]
    n = len(block.data)
    co = np.empty(n * 3, dtype=np.float64)
    block.data.foreach_get("co", co)
    return co.reshape(n, 3)


def _find_obj_entry(decoded, uuid, name):
    for grp in decoded:
        for o in grp.get("object", []):
            if o.get("uuid") == uuid or o.get("name") == name:
                return o
    return None


def _raw_group(dh, name):
    for g in dh.groups.iterate_active_object_groups(bpy.context.scene):
        if g.name == name:
            return g
    return None


def _encode_entry(encoder_pkg, uuid, name):
    data_bytes, _param, _dh, _ph = encoder_pkg.prepare_upload(bpy.context)
    entry = _find_obj_entry(DriverHelpers.decode_addon_blob(data_bytes), uuid, name)
    if entry is None:
        raise RuntimeError("sheet object not found in encoded scene")
    vert = np.asarray(entry.get("vert"), dtype=np.float64)
    brv = entry.get("bend_rest_vert")
    brv = None if brv is None else np.asarray(brv, dtype=np.float64)
    return vert, brv


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    uuidreg = __import__(pkg + ".core.uuid_registry",
                         fromlist=["get_or_create_object_uuid"])
    utils_pkg = __import__(pkg + ".core.utils",
                           fromlist=["validate_bend_shape_key"])
    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])

    # --- scene: flat grid with a flat Basis and a curled key ---
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=6, y_subdivisions=6, size=2, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "BendKeySheet"
    pinned = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    sheet.shape_key_add(name="Basis", from_mix=False)
    curled = sheet.shape_key_add(name="Curled", from_mix=False)
    for point in curled.data:
        point.co.z = 0.3 * point.co.x * point.co.x
    curled.value = 0.0

    bpy.ops.curve.primitive_bezier_curve_add(location=(5, 0, 0))
    curve = bpy.context.object
    curve.name = "BendKeyCurve"

    sheet_uuid = uuidreg.get_or_create_object_uuid(sheet)
    basis_co = _key_co(sheet, "Basis")
    curled_co = _key_co(sheet, "Curled")

    dh.save_blend(PROBE_DIR, "bend_reference_shape_key.blend")
    root = dh.configure_state(
        project_name="bend_reference_shape_key",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    grp = _raw_group(dh, "Cloth")
    grp.bend_rest_angle_source = "FROM_REFERENCE"
    assigned = None
    for a in grp.assigned_objects:
        if a.uuid == sheet_uuid:
            assigned = a
            break
    if assigned is None:
        raise RuntimeError("assigned object for sheet not found in group")
    assigned.bend_ref_source = "SHAPE_KEY"
    assigned.bend_ref_shape_key = "Curled"
    dh.log("reference_configured")

    # --- A/B/C: validator ---
    validate = utils_pkg.validate_bend_shape_key
    ok_a, msg_a = validate(sheet, "Curled")
    dh.record("A_validate_accepts_key", ok_a, {"msg": msg_a})
    ok_b, msg_b = validate(sheet, "NoSuchKey")
    dh.record(
        "B_validate_rejects_missing_key",
        (not ok_b) and "NoSuchKey" in msg_b,
        {"msg": msg_b},
    )
    ok_c, msg_c = validate(curve, "Basis")
    dh.record(
        "C_validate_rejects_curve",
        (not ok_c) and "mesh" in msg_c.lower(),
        {"msg": msg_c},
    )

    # --- D: flat pose shipped, curled key as the rest shape ---
    vert, brv = _encode_entry(encoder_pkg, sheet_uuid, sheet.name)
    shape_ok = bool(brv is not None and brv.shape == vert.shape == curled_co.shape)
    ships_flat = bool(np.allclose(vert, basis_co, atol=1e-4))
    matches_key = bool(shape_ok and np.allclose(brv, curled_co, atol=1e-4))
    dh.record(
        "D_encoder_ships_shape_key",
        shape_ok and ships_flat and matches_key,
        {"shape_ok": shape_ok, "ships_flat": ships_flat,
         "matches_key": matches_key},
    )

    # --- E: deformed mix shipped, Basis as the rest shape ---
    curled.value = 1.0
    assigned.bend_ref_shape_key = "Basis"
    bpy.context.view_layer.update()
    vert, brv = _encode_entry(encoder_pkg, sheet_uuid, sheet.name)
    shape_ok = bool(brv is not None and brv.shape == vert.shape)
    ships_curled = bool(np.allclose(vert, curled_co, atol=1e-4))
    rest_is_basis = bool(shape_ok and np.allclose(brv, basis_co, atol=1e-4))
    dh.record(
        "E_basis_rest_under_deformed_mix",
        shape_ok and ships_curled and rest_is_basis,
        {"shape_ok": shape_ok, "ships_curled": ships_curled,
         "rest_is_basis": rest_is_basis},
    )
    curled.value = 0.0
    assigned.bend_ref_shape_key = "Curled"
    bpy.context.view_layer.update()

    # --- F: a renamed key fails the upload by name ---
    curled.name = "CurledRenamed"
    raised = False
    emsg = ""
    try:
        encoder_pkg.prepare_upload(bpy.context)
    except ValueError as exc:
        raised = True
        emsg = str(exc)
    dh.record(
        "F_encoder_rejects_renamed_key",
        raised and "'Curled'" in emsg and sheet.name in emsg,
        {"msg": emsg[:300]},
    )
    curled.name = "Curled"

    # --- G: the source decides what is read, and an object of a From
    # Reference Geometry group that names nothing is refused ---
    assigned.bend_ref_source = "OBJECT"
    msg_g = ""
    try:
        encoder_pkg.prepare_upload(bpy.context)
    except ValueError as exc:
        msg_g = str(exc)
    dh.record(
        "G_object_without_reference_refused",
        "no bending reference" in msg_g and sheet.name in msg_g,
        {"msg": msg_g[:300]},
    )
    assigned.bend_ref_source = "SHAPE_KEY"

    # --- J: the panel draws every state, rows disabled, never hidden ---
    # The rig's hidden window does not reliably draw the sidebar, so the
    # draw function runs against a layout that records what it was asked
    # to draw, which is how a panel's predicates are verified in a subtest.
    panels = __import__(pkg + ".ui.dynamics.panels",
                        fromlist=["_draw_bend_reference"])

    class _Layout:
        def __init__(self, log):
            self.log = log
            self.enabled = True

        def box(self):
            return _Layout(self.log)

        def row(self, **_k):
            return _Layout(self.log)

        def column(self, **_k):
            return _Layout(self.log)

        def label(self, text="", icon="NONE"):
            self.log.append(("label", text, icon))

        def prop(self, _data, attr, **_k):
            self.log.append(("prop", attr, self.enabled))

        def prop_search(self, _data, attr, _sdata, _sattr, **_k):
            self.log.append(("prop_search", attr, self.enabled))

        def operator(self, idname, **_k):
            self.log.append(("op", idname, self.enabled))

            class _Op:
                pass
            return _Op()

    def _draw():
        log = []
        panels._draw_bend_reference(_Layout(log), grp, 0)
        return log

    grp.bend_ref_object_selection = sheet_uuid
    saved_key = assigned.bend_ref_shape_key
    states = {}
    assigned.bend_ref_shape_key = "Curled"
    states["ok"] = _draw()
    assigned.bend_ref_shape_key = "Gone"
    states["missing"] = _draw()
    assigned.bend_ref_shape_key = ""
    states["unset"] = _draw()
    assigned.bend_ref_shape_key = "Curled"
    assigned.bend_ref_source = "OBJECT"
    states["object"] = _draw()
    assigned.bend_ref_source = "SHAPE_KEY"
    assigned.bend_ref_shape_key = saved_key

    # The group box: the Rest Angle dropdown decides between the reference
    # controls and Bend Plasticity.
    def _draw_group():
        log = []
        panels._draw_shell_bend_rest(_Layout(log), grp, 0)
        return log

    states["group_reference"] = _draw_group()
    grp.bend_rest_angle_source = "FLAT"
    states["group_flat"] = _draw_group()
    grp.bend_rest_angle_source = "FROM_REFERENCE"

    def _has(log, entry):
        return entry in log

    def _props(log):
        return [e[1] for e in log if e[0] in ("prop", "prop_search")]

    panel_ok = (
        _has(states["ok"], ("prop_search", "bend_ref_shape_key", True))
        and _has(states["missing"], ("label", "Missing shape key: Gone", "ERROR"))
        and _has(states["unset"], (
            "label", "No shape key set: Transfer refuses this object", "ERROR"))
        # Object source: the picker row, an ERROR for the missing object,
        # and no shape-key field.
        and _has(states["object"], ("op", "object.pick_bend_reference", True))
        and _has(states["object"], (
            "label", "No reference set: Transfer refuses this object", "ERROR"))
        and not any(e[1] == "bend_ref_shape_key" for e in states["object"])
        # No per-object enable checkbox anywhere.
        and not any("bend_ref_enable" == n for v in states.values() for n in _props(v))
        # From Reference Geometry: reference controls, no Bend Plasticity.
        and "bend_ref_source" in _props(states["group_reference"])
        and "enable_bend_plasticity" not in _props(states["group_reference"])
        # Flat: Bend Plasticity, no reference controls.
        and "enable_bend_plasticity" in _props(states["group_flat"])
        and "bend_ref_source" not in _props(states["group_flat"])
    )
    dh.record(
        "J_panel_draws_every_state",
        panel_ok,
        {k: [list(e) for e in v] for k, v in states.items()},
    )

    # --- K: a file saved with the retired checkboxes migrates ---
    # Written as the raw ID-properties an older .blend carries.
    mig = __import__(pkg + ".core.migrate_renames",
                     fromlist=["migrate_bend_rest_reference"])
    grp.bend_rest_angle_source = "FLAT"
    grp["bend_rest_from_reference"] = 1
    assigned["bend_ref_enable"] = 0
    summary_on = mig.migrate_bend_rest_reference(bpy.context.scene)
    on_ok = (
        grp.bend_rest_angle_source == "FROM_REFERENCE"
        and "bend_rest_from_reference" not in grp.keys()
        and "bend_ref_enable" not in assigned.keys()
    )
    grp.bend_rest_angle_source = "FROM_GEOMETRY"
    grp["bend_rest_from_reference"] = 0
    summary_off = mig.migrate_bend_rest_reference(bpy.context.scene)
    off_ok = (
        grp.bend_rest_angle_source == "FROM_GEOMETRY"
        and "bend_rest_from_reference" not in grp.keys()
        and summary_off == ""
    )
    grp.bend_rest_angle_source = "FROM_REFERENCE"
    dh.record("K_retired_checkbox_migrates", on_ok and off_ok,
              {"summary_on": summary_on, "on_ok": on_ok, "off_ok": off_ok})

    # --- H/I: end-to-end build + run with the shape key active ---
    data_bytes, param_bytes, _dh2, _ph2 = encoder_pkg.prepare_upload(bpy.context)
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, "bend-reference-shape-key:build",
                      timeout=120.0)
    solver_name = dh.facade.engine.state.solver.name
    dh.record(
        "H_build_succeeds_with_shape_key",
        solver_name in ("READY", "RESUMABLE"),
        {"solver": solver_name},
    )

    saw_running = dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    frame = dh.facade.engine.state.frame
    dh.record(
        "I_run_completes",
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
