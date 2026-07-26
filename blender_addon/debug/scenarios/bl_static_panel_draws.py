# File: scenarios/bl_static_panel_draws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The STATIC group's Transform panel draws without error, and each op
# editor draws its own fields.
#
# The static-op encoding scenarios validate the data the Transform box
# produces but never render it, and the rig's hidden window does not
# reliably trigger a sidebar draw. This drives ``_draw_static_ops`` with
# a recording mock layout (no real UI region needed) across all four
# branches: no active assigned-object row, and an active row whose
# selected op is MOVE_BY / SPIN / SCALE. A wrong property name or
# operator id would raise during draw; this catches it.
#
# The mock records every ``prop()`` call, so each op subtest asserts the
# editor's own fields actually reached the layout rather than merely that
# the draw did not raise. With no active row the panel early-outs to a
# label, so that branch asserts it draws no properties at all.
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
    # Answers every layout call so the panel draw runs end to end, and
    # records the property names drawn into a shared sink (child layouts
    # returned by box/row/column share the parent's sink).

    def __init__(self, sink=None):
        self.sink = [] if sink is None else sink
        self.enabled = True
        self.alignment = "EXPAND"

    def box(self, *a, **k):
        return MockLayout(self.sink)

    def row(self, *a, **k):
        return MockLayout(self.sink)

    def column(self, *a, **k):
        return MockLayout(self.sink)

    def label(self, *a, **k):
        pass

    def separator(self, *a, **k):
        pass

    def prop(self, data, prop_name, *a, **k):
        self.sink.append(prop_name)

    def prop_enum(self, *a, **k):
        pass

    def template_list(self, *a, **k):
        pass

    def operator(self, *a, **k):
        return types.SimpleNamespace()

    def operator_menu_enum(self, *a, **k):
        return types.SimpleNamespace()


def draw_and_collect(draw, group, index):
    layout = MockLayout()
    draw(layout, group, index)
    return layout.sink


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "Collider"

    root = dh.configure_state(project_name="static_panel_draws", frame_count=4)
    grp_api = dh.api.solver.create_group("Stat", "STATIC")
    grp_api.add("Collider")

    group = root.object_group_0
    panels = __import__(pkg + ".ui.dynamics.panels", fromlist=["_draw_static_ops"])
    draw = panels._draw_static_ops

    # ----- A: no active assigned-object row --------------------------
    # The panel early-outs to an informational label, drawing no fields.
    group.assigned_objects_index = -1
    props = draw_and_collect(draw, group, 0)
    dh.record(
        "A_no_props_without_active_row",
        props == [],
        {"props_drawn": props},
    )

    # An active row with one op we retype per subtest below. The row must be
    # made active explicitly: _draw_static_ops early-outs on an out-of-range
    # assigned_objects_index, so without this the per-op editor never draws
    # and B/C/D would pass vacuously against an empty layout.
    group.assigned_objects_index = 0
    assigned = group.assigned_objects[0]
    op = assigned.static_ops.add()
    op.frame_start = 1
    op.frame_end = 4
    op.transition = "LINEAR"
    assigned.static_ops_index = 0

    # ----- B/C/D: each op kind's editor draws its own fields ---------
    # Every editor draws the shared timing/transition fields plus the
    # fields specific to its op kind.
    common = ("frame_start", "frame_end", "transition")
    for name, op_type, expected in (
        ("B_move_by_op_draws", "MOVE_BY", common + ("delta",)),
        ("C_spin_op_draws", "SPIN",
         common + ("spin_axis", "spin_angular_velocity")),
        ("D_scale_op_draws", "SCALE", common + ("scale_factor",)),
    ):
        op.op_type = op_type
        props = draw_and_collect(draw, group, 0)
        missing = [f for f in expected if f not in props]
        dh.record(
            name,
            not missing,
            {"op_type": op_type, "missing": missing, "props_drawn": props},
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
