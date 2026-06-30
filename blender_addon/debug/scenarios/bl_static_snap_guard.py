# File: scenarios/bl_static_snap_guard.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Snap-side coverage for {SOLID,SHELL,ROD}-STATIC cross-stitch. This is a
# Blender-only scenario (no solver build / server run), so it is fast and
# runs in the Blender CI on macOS + Linux.
#
# bl_static_stitch.py authors the cross_stitch payload by hand and so does
# NOT exercise the snap authoring path. This scenario drives the shared
# snap core (_snap_pair, which OBJECT_OT_SnapToVertices delegates to) for a
# dynamic-to-STATIC pair and locks in the three things that path must
# guarantee:
#   1. the STATIC collider is NEVER translated (the encoder ships its
#      rest-pose geometry; moving it would desync geometry from the stitch);
#   2. the dynamic object IS moved to the small keep gap;
#   3. the authored 6-wide cross_stitch names the dynamic object as SOURCE
#      and the STATIC as TARGET, for all three dynamic types and for either
#      snap slot order (STATIC in slot A or slot B), since _snap_pair always
#      translates obj_a by convention but must override that for a STATIC.
#
# Subtests:
#   A. solid_static_solid_in_a:   SOLID(a) + STATIC(b) -> SOLID moves, STATIC frozen.
#   B. solid_static_static_in_a:  STATIC(a) + SOLID(b) -> SOLID moves, STATIC frozen
#                                 (the move-guard's overridden-slot case).
#   C. shell_static:              SHELL(a) + STATIC(b) -> SHELL moves, STATIC frozen.
#   D. rod_static:                ROD(a)   + STATIC(b) -> ROD moves, STATIC frozen.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

import bpy

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    snap_mod = __import__(pkg + ".mesh_ops.snap_ops",
                          fromlist=["_snap_pair"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    class _FakeOp:
        # _snap_pair only needs an object with a .report(level, msg) method.
        def __init__(self):
            self.reports = []

        def report(self, level, msg):
            self.reports.append((tuple(level), msg))

    def _fresh_scene():
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        dh.configure_state(
            project_name="static_snap_guard",
            frame_count=2,
            gravity=(0.0, 0.0, 0.0),
        )

    def _add_solid(name, location):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
        o = bpy.context.active_object
        o.name = name
        return o

    def _add_shell(name, location):
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
        o = bpy.context.active_object
        o.name = name
        return o

    def _add_static(name, location):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
        o = bpy.context.active_object
        o.name = name
        return o

    def _add_rod(name, location):
        curve_data = bpy.data.curves.new(name=name, type="CURVE")
        curve_data.dimensions = "3D"
        spline = curve_data.splines.new(type="BEZIER")
        spline.bezier_points.add(count=2)  # 3 CVs total
        for i, bp in enumerate(spline.bezier_points):
            bp.co = (i * 0.5, 0.0, 0.0)
            bp.handle_left_type = "AUTO"
            bp.handle_right_type = "AUTO"
        o = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(o)
        o.location = location
        return o

    def _run_case(check_name, dyn_type, dyn_factory, dyn_loc, static_loc,
                  static_in_slot_a):
        _fresh_scene()
        dyn = dyn_factory("Dyn", dyn_loc)
        stat = _add_static("Static", static_loc)
        dyn_group = dh.api.solver.create_group("Dyn", dyn_type)
        dyn_group.add(dyn.name)
        static_group = dh.api.solver.create_group("Static", "STATIC")
        static_group.add(stat.name)
        dyn_uuid = uuid_mod.get_or_create_object_uuid(dyn)
        stat_uuid = uuid_mod.get_or_create_object_uuid(stat)
        bpy.context.view_layer.update()

        stat_before = tuple(stat.matrix_world.translation)
        dyn_before = tuple(dyn.matrix_world.translation)

        obj_a, obj_b = (stat, dyn) if static_in_slot_a else (dyn, stat)
        op = _FakeOp()
        status = snap_mod._snap_pair(op, bpy.context, obj_a, obj_b)

        bpy.context.view_layer.update()
        stat_after = tuple(stat.matrix_world.translation)
        dyn_after = tuple(dyn.matrix_world.translation)

        static_moved = max(abs(a - b) for a, b in zip(stat_before, stat_after))
        dyn_moved = max(abs(a - b) for a, b in zip(dyn_before, dyn_after))

        # Find the merge pair + its authored cross_stitch.
        import json as _json
        state = dh.groups.get_addon_data(bpy.context.scene).state
        cs = None
        for pair in state.merge_pairs:
            uset = {pair.object_a_uuid, pair.object_b_uuid}
            if uset == {dyn_uuid, stat_uuid} and pair.cross_stitch_json:
                cs = _json.loads(pair.cross_stitch_json)
                break
        ind_widths = sorted({len(row) for row in cs.get("ind", [])}) if cs else []
        source_is_dyn = bool(cs and cs.get("source_uuid") == dyn_uuid)
        target_is_static = bool(cs and cs.get("target_uuid") == stat_uuid)

        ok = (
            "FINISHED" in status
            and static_moved < 1e-6        # STATIC never translated
            and dyn_moved > 1e-3           # dynamic repositioned to keep gap
            and cs is not None
            and ind_widths == [6]
            and source_is_dyn
            and target_is_static
        )
        dh.record(
            check_name,
            ok,
            {
                "status": sorted(status),
                "static_moved": static_moved,
                "dyn_moved": dyn_moved,
                "has_cross_stitch": cs is not None,
                "ind_widths": ind_widths,
                "source_is_dynamic": source_is_dyn,
                "target_is_static": target_is_static,
                "op_reports": [m for _, m in op.reports],
            },
        )

    # A/B: SOLID + STATIC, both slot orders (the move-guard's two branches).
    _run_case("A_solid_static_solid_in_a", "SOLID", _add_solid,
              (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), static_in_slot_a=False)
    _run_case("B_solid_static_static_in_a", "SOLID", _add_solid,
              (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), static_in_slot_a=True)
    # C: SHELL (cloth) stitched up to a STATIC above it.
    _run_case("C_shell_static", "SHELL", _add_shell,
              (0.0, 0.0, 0.0), (0.0, 0.0, 3.0), static_in_slot_a=False)
    # D: ROD (curve) stitched to a STATIC off its end. The rod spans
    # x in [0, 1] (3 collinear CVs at x=0,0.5,1.0); place the STATIC cube so
    # its -X face (x=2) sits ~1.0 beyond the rod end, leaving a gap for the
    # snap to close (a coincident start gives nothing to move).
    _run_case("D_rod_static", "ROD", _add_rod,
              (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), static_in_slot_a=False)

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
