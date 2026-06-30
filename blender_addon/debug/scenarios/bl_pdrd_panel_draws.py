# File: scenarios/bl_pdrd_panel_draws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# PDRD specialized pin panel draws without error.
#
# The encoding scenarios validate the data the rigid-body panel produces but
# never render it. This drives ``_draw_pdrd_pins`` with a recording mock
# layout (no real UI region needed) for every branch: a held pin with a
# release frame (no motion steps), a pin with a Translate step active, and a
# pin with a Rotate-about-vertex step active. A wrong property name or
# operator id would raise during draw; this catches it.
#
# Assertion-only: no server / build / run.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import types
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


class MockLayout:
    # Answers every layout call so the panel draw runs end to end.
    # ``operator`` returns a namespace so callers can set .group_index /
    # .op_type / .direction / .target freely.

    def __init__(self):
        self.enabled = True
        self.alignment = "EXPAND"

    def box(self, *a, **k):
        return MockLayout()

    def row(self, *a, **k):
        return MockLayout()

    def column(self, *a, **k):
        return MockLayout()

    def label(self, *a, **k):
        pass

    def separator(self, *a, **k):
        pass

    def prop(self, *a, **k):
        pass

    def prop_enum(self, *a, **k):
        pass

    def template_list(self, *a, **k):
        pass

    def operator(self, *a, **k):
        return types.SimpleNamespace()


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "Rigid"
    cube.vertex_groups.new(name="corner").add([0], 1.0, "REPLACE")
    cube.vertex_groups.new(name="handle").add([4, 5, 6, 7], 1.0, "REPLACE")

    root = dh.configure_state(project_name="pdrd_panel_draws", frame_count=4)
    grp_api = dh.api.solver.create_group("Rigid", "PDRD")
    grp_api.add("Rigid")
    grp_api.create_pin("Rigid", "corner")
    grp_api.create_pin("Rigid", "handle")

    group = root.object_group_0
    anchor = group.pin_vertex_groups[0]   # no motion steps -> holds fixed
    driven = group.pin_vertex_groups[1]   # motion steps added below
    anchor.use_pin_duration = True
    anchor.pin_duration = 30

    # Give the driven pin one Translate and one Rotate-about-vertex step.
    group.pin_vertex_groups_index = 1
    bpy.ops.object.add_pin_operation(group_index=0, op_type="SPIN")
    bpy.ops.object.add_pin_operation(group_index=0, op_type="MOVE_BY")
    for op in driven.operations:
        if op.op_type == "SPIN":
            op.spin_center_mode = "VERTEX"
            op.spin_center_vertex = 0

    panels = __import__(pkg + ".ui.dynamics.panels", fromlist=["_draw_pdrd_pins"])
    draw = panels._draw_pdrd_pins

    # ----- A: a held pin (no motion steps) draws ---------------------
    group.pin_vertex_groups_index = 0
    draw(MockLayout(), group, 0, bpy.context)
    dh.record("A_held_pin_draws", True, {})

    # ----- B: a pin with a Translate step active draws ---------------
    group.pin_vertex_groups_index = 1
    for i, op in enumerate(driven.operations):
        if op.op_type == "MOVE_BY":
            driven.operations_index = i
    draw(MockLayout(), group, 0, bpy.context)
    dh.record("B_translate_step_draws", True, {})

    # ----- C: a pin with a Rotate(vertex) step active draws ----------
    for i, op in enumerate(driven.operations):
        if op.op_type == "SPIN":
            driven.operations_index = i
    draw(MockLayout(), group, 0, bpy.context)
    dh.record("C_rotate_vertex_step_draws", True, {})

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
