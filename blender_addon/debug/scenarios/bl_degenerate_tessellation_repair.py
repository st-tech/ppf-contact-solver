# File: scenarios/bl_degenerate_tessellation_repair.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The repair offered when a Transfer is refused for a degenerate tessellation.
#
# `bl_degenerate_tessellation_rejection` covers the refusal itself. This covers
# what the artist can do about it: the encoder raises a TYPED error carrying the
# object, the offending polygons and whether triangulating would help, the
# failing Transfer opens a dialog rather than leaving a paragraph in the status
# bar, and the dialog's button repairs the scene in one click.
#
# The repair is scoped to the offending polygons, and that is the property worth
# guarding. Face > Triangulate Faces over a selection converts every quad in it;
# on the mesh from issue #144 that is 2064 faces changed to fix 16. The button
# splits the 16 and leaves the rest as quads, so the artist's mesh comes back
# with the defect gone and nothing else touched.
#
# The dialog's DRAW is not exercised here. `invoke_props_dialog` attaches a
# modal handler and waits for a human, which a driver cannot answer, so this
# asserts the decision to open one (`wants_transfer_dialog`) and the operator's
# registration and properties instead, and leaves the pixels to a human.
#
# Subtests:
#   A. encoder_raises_a_typed_error
#         The refusal carries object, group, polygons and repairable, not just
#         a sentence. Without the fields the dialog cannot offer the repair.
#   B. dialog_opens_only_for_this_error
#         `wants_transfer_dialog` says yes to it and no to a plain ValueError,
#         so no other encode failure gets a dialog it cannot act on.
#   C. repair_splits_only_the_offenders
#         The polygon count rises by exactly the number of flagged quads, the
#         flagged ones are gone, and every other face is still a quad.
#   D. encode_succeeds_after_repair
#         The whole point: the same scene transfers afterwards.
#   E. unrepairable_faces_are_left_alone
#         A quad with two coincident corners has no sound triangulation, so
#         `repairable` is False (the dialog draws no button) and the operator
#         reports it rather than splitting it into more degenerate triangles.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def make_quad(name, coords):
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata([tuple(c) for c in coords], [], [tuple(range(len(coords)))])
    mesh.update()
    return obj


def make_sliver_and_plain(name, h):
    # One offending quad plus one sound quad in the same mesh. The offender is
    # the shape from issue #144: vertex 1 sits `h` above the straight line from
    # vertex 0 to vertex 2, so the 0-2 diagonal puts three near-collinear
    # vertices in one triangle and the 1-3 diagonal does not. Its conditioning
    # is 0.4 * h. The plain quad is the control: it must come back a quad.
    co = [
        (0.0, 0.0, 0.0), (1.0, h, 0.0), (2.0, 0.0, 0.0), (1.0, -1.0, 0.0),
        (3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 1.0, 0.0), (3.0, 1.0, 0.0),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(co, [], faces)
    mesh.update()
    return obj


COINCIDENT_QUAD = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    utils = __import__(pkg + ".core.utils",
                       fromlist=["DegenerateTessellationError"])
    cleanup = __import__(pkg + ".ui.geometry_cleanup_ops",
                         fromlist=["wants_transfer_dialog"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    dh.log("setup_start")
    dh.reset_scene_to_pinned_plane(name="RepairBaseMesh")
    root = dh.configure_state(project_name="degenerate_repair", frame_count=6)
    dh.api.solver.create_group("Cloth", "SHELL")

    def assign(obj):
        item = root.object_group_0.assigned_objects.add()
        item.name = obj.name
        item.uuid = uuid_mod.get_or_create_object_uuid(obj)
        return item

    def clear_assigned():
        while len(root.object_group_0.assigned_objects):
            root.object_group_0.assigned_objects.remove(
                len(root.object_group_0.assigned_objects) - 1
            )

    def encode_error():
        try:
            encoder_mesh.compute_data_hash(bpy.context)
        except Exception as exc:
            return exc
        return None

    # ----- A: the error carries structure ------------------------------
    # 4.0e-07 conditioning: the scale of the worst face of the reporter's mesh.
    strip = make_sliver_and_plain("SliverStrip", 1.0e-06)
    assign(strip)
    err_a = encode_error()
    typed = isinstance(err_a, utils.DegenerateTessellationError)
    dh.record(
        "A_encoder_raises_a_typed_error",
        typed
        and err_a.object_name == "SliverStrip"
        and err_a.group_name == "Cloth"
        and len(err_a.polygons) == 1
        and err_a.repairable is True
        # Still a ValueError, so every existing handler keeps catching it.
        and isinstance(err_a, ValueError),
        {"type": type(err_a).__name__ if err_a else None,
         "object": getattr(err_a, "object_name", None),
         "group": getattr(err_a, "group_name", None),
         "polygons": getattr(err_a, "polygons", None),
         "repairable": getattr(err_a, "repairable", None)},
    )

    # ----- B: only this error gets a dialog ----------------------------
    dh.record(
        "B_dialog_opens_only_for_this_error",
        cleanup.wants_transfer_dialog(err_a) is True
        and cleanup.wants_transfer_dialog(ValueError("something else")) is False
        and hasattr(bpy.ops.ssh, "degenerate_tessellation_dialog")
        and hasattr(bpy.ops.ssh, "triangulate_degenerate_faces"),
        {"typed": cleanup.wants_transfer_dialog(err_a),
         "plain": cleanup.wants_transfer_dialog(ValueError("something else"))},
    )

    # ----- C: the repair is scoped to the offenders --------------------
    before_polys = len(strip.data.polygons)
    before_quads = sum(1 for p in strip.data.polygons if len(p.vertices) == 4)
    op_result = bpy.ops.ssh.triangulate_degenerate_faces()
    after_polys = len(strip.data.polygons)
    after_quads = sum(1 for p in strip.data.polygons if len(p.vertices) == 4)
    found_c = utils.find_degenerate_tessellation(strip)
    dh.record(
        "C_repair_splits_only_the_offenders",
        op_result == {"FINISHED"}
        # One quad became two triangles: +1 face, -1 quad.
        and after_polys == before_polys + 1
        and after_quads == before_quads - 1
        and found_c["count"] == 0,
        {"before_polys": before_polys, "after_polys": after_polys,
         "before_quads": before_quads, "after_quads": after_quads,
         "found": found_c, "op": list(op_result)},
    )

    # ----- D: and the scene transfers afterwards ------------------------
    err_d = encode_error()
    dh.record(
        "D_encode_succeeds_after_repair",
        err_d is None,
        {"err": f"{type(err_d).__name__}: {err_d}"[:300] if err_d else ""},
    )

    # ----- E: a face no triangulation rescues is left alone -------------
    clear_assigned()
    coincident = make_quad("CoincidentQuad", COINCIDENT_QUAD)
    assign(coincident)
    err_e = encode_error()
    polys_before_e = len(coincident.data.polygons)
    op_e = bpy.ops.ssh.triangulate_degenerate_faces()
    polys_after_e = len(coincident.data.polygons)
    dh.record(
        "E_unrepairable_faces_are_left_alone",
        isinstance(err_e, utils.DegenerateTessellationError)
        and err_e.repairable is False
        and op_e == {"CANCELLED"}
        and polys_after_e == polys_before_e,
        {"repairable": getattr(err_e, "repairable", None),
         "op": list(op_e), "before": polys_before_e, "after": polys_after_e},
    )

    # ----- H: a partial repair keeps the refusal standing ----------------
    # The panel's error block, repair row included, draws off `com.error`.
    # Clearing it because the button ran, rather than because the scene came
    # back clean, hides a refusal the Transfer will still make.
    # `com.set_error` dispatches an event the engine applies only once
    # connected, and this scenario runs no server, so the CALL is what is
    # observed: clearing on intent passes "" here, and the fix passes the
    # refusal text naming the object that is still bad.
    clear_assigned()
    fixable = make_sliver_and_plain("PartialFixable", 3.75e-07)
    stubborn = make_quad("PartialStubborn", COINCIDENT_QUAD)
    assign(fixable)
    assign(stubborn)
    set_error_calls = []
    real_set_error = cleanup.com.set_error
    cleanup.com.set_error = set_error_calls.append
    try:
        bpy.ops.ssh.triangulate_degenerate_faces()
    finally:
        cleanup.com.set_error = real_set_error
    partial_error = set_error_calls[-1] if set_error_calls else ""
    dh.record(
        "H_a_partial_repair_keeps_the_refusal",
        bool(set_error_calls)
        and "PartialStubborn" in partial_error
        and "no usable rest shape" in partial_error,
        {"calls": [c[:120] for c in set_error_calls]},
    )

    # ----- I: one datablock is repaired once, not once per group ---------
    # An object in two active groups, or two objects sharing a mesh, would
    # otherwise be split twice, the second time through stale pre-repair
    # polygon indices, and counted twice in the report.
    clear_assigned()
    twice = make_sliver_and_plain("TwiceAssigned", 3.75e-07)
    assign(twice)
    assign(twice)
    before_polys = len(twice.data.polygons)
    offenders = cleanup._degenerate_tessellation_offenders(bpy.context)
    dh.record(
        "I_one_datablock_is_repaired_once",
        len(offenders) == 1,
        {"n_offenders": len(offenders), "n_polys_before": before_polys},
    )
    clear_assigned()

    # ----- G: the RUN path leaves the panel a repair to offer -------------
    # The panel's repair branch draws off `com.error`. Transfer set it, Run and
    # Resume did not, so a dialog the artist dismissed took the button with it
    # and the only way back was another Transfer.
    import types as _types
    # A fresh offender, so the stage reaches the geometry refusal rather than
    # the later "geometry has changed since the last transfer" abort.
    clear_assigned()
    run_fixture = make_sliver_and_plain("RunPathSliver", 3.75e-07)
    assign(run_fixture)
    fixture_err = encode_error()
    solver_mod = __import__(pkg + ".ui.solver", fromlist=["SOLVER_OT_Run"])
    # `com.set_error` dispatches an event the engine applies only once
    # connected, and this scenario runs no server, so the CALL is what is
    # observed. That is the regression exactly: Transfer made it and the other
    # two paths did not.
    recorded = {}

    def recorder(label):
        def _set_error(msg):
            recorded[label] = msg
        return _set_error

    stages = (
        ("transfer", "SOLVER_OT_Transfer", "_stage_encode_geometry"),
        ("run", "SOLVER_OT_Run", "_stage_check_geometry"),
    )
    real_set_error = solver_mod.com.set_error
    try:
        for label, cls_name, method in stages:
            solver_mod.com.set_error = recorder(label)
            try:
                getattr(getattr(solver_mod, cls_name), method)(
                    _types.SimpleNamespace(), bpy.context
                )
            except Exception:
                pass
    finally:
        solver_mod.com.set_error = real_set_error
    dh.record(
        "G_every_path_sets_the_error_the_panel_reads",
        set(recorded) == {label for label, _c, _m in stages}
        and all("no usable rest shape" in msg for msg in recorded.values()),
        {"recorded": {k: v[:120] for k, v in recorded.items()},
         "fixture_encode_error": str(fixture_err)[:160]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
