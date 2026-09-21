# File: scenarios/bl_intersection_allowance_panel_draws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The object-subset UI of the two intersection allowances, DRAWN.
#
# `bl_intersection_allowance_objects` proves the subset reaches the right
# per-vertex policy byte, and `bl_intersection_allowance_objects_run` proves it
# decides whether a scene builds and runs. Neither touches the panel, and a
# subset the artist cannot see or edit is not a feature. This scenario draws
# it.
#
# A DRAW IS NOT DECORATION HERE. The list is a `template_list`, which resolves
# the names of the collection and the active-index property at DRAW time, not
# at registration: rename either and the add-on still registers, every other
# check in the set still passes, and the panel throws the moment a user opens
# the group. Nothing but a draw catches that.
#
# Subtests:
#   A. panel_reaches_the_allowance_box  - the real group panel draws and the
#                                         allowance box is reached for BOTH
#                                         allowances on each draw, so a panel
#                                         that stopped calling it cannot pass.
#   B. list_row_draws                   - the UIList's `draw_item` runs on a
#                                         real layout for a real entry.
#   C. draw_raised_nothing              - no exception escaped either, which is
#                                         where a renamed collection or index
#                                         property lands.
#
# Pure UI scenario: no server, no solver, no transfer.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It draws a panel and never asks the solver to step, so nothing in it is
# backend-specific.
BACKENDS = ("real",)

# NOT ON WINDOWS, WHICH RUNS THE RIG HEADLESS.
#
# `bpy.ops.wm.redraw_timer` is how a scenario makes a panel draw, and its poll
# fails with "context is incorrect" on a Blender that owns no window. Measured
# on the Windows leg of run 35490498574, where this probe still lived inside
# `bl_intersection_allowance_objects`: the driver raised before the first
# build, so the eight checks in that file that have nothing to do with drawing
# were lost with it. Splitting the probe out is what keeps a platform that
# cannot draw from costing the coverage that does not need to.
#
# `bl_material_map_panel_draws` declares the same pair for the same reason.
# Linux gives the rig its own Xvfb and macOS has a real window server.
PLATFORMS = ("linux", "darwin")


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

NAMES = ["Listed", "Unlisted"]

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    for i, name in enumerate(NAMES):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3,
                                        size=1.0, location=(3.0 * i, 0.0, 0.0))
        bpy.context.object.name = name

    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    ia_mod = __import__(pkg + ".models.intersection_allowances",
                        fromlist=["INTERSECTION_ALLOWANCES"])
    addon_root = dh.groups.get_addon_data(scene)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    for name in NAMES:
        cloth.add(name)
    slot = groups_mod.get_group_slot_index(scene, cloth.uuid)
    group = getattr(addon_root, "object_group_%d" % slot)

    # The state worth drawing: one allowance narrowed with an entry in its
    # list, the other covering the whole group so its list and buttons are
    # drawn disabled in the same panel.
    group.show_group = True
    group.show_parameters = True
    group.allow_self_intersection = True
    group.allow_self_intersection_all_objects = False
    entry = group.allow_self_intersection_objects.add()
    entry.name = "Listed"
    entry.uuid = uuid_mod.get_or_create_object_uuid(bpy.data.objects["Listed"])
    group.allow_inter_object_intersection = True
    dh.log("group ready slot=%d listed=%r"
           % (slot, [i.name for i in group.allow_self_intersection_objects]))

    panels = __import__(pkg + ".ui.dynamics.panels", fromlist=["x"])
    ui_lists = __import__(pkg + ".ui.dynamics.ui_lists", fromlist=["x"])
    DREW = {"panel": 0, "allowance_box": 0, "list_row": 0, "errors": []}

    # Count the allowance box separately from the panel: a panel that stopped
    # calling it would otherwise leave this green.
    _real_draw = panels._draw_intersection_allowance

    def _counting_draw(box, grp, spec, actual_index):
        DREW["allowance_box"] += 1
        return _real_draw(box, grp, spec, actual_index)

    panels._draw_intersection_allowance = _counting_draw

    class PPF_PT_AllowanceProbe(bpy.types.Panel):
        bl_label = "Allowance Draw Probe"
        bl_idname = "PPF_PT_allowance_probe"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "object"

        def draw(self, context):
            try:
                # The REAL panel, so the allowance box is reached through the
                # same type chain a user's redraw takes.
                panels.DYNAMICS_PT_Groups.draw(self, context)
                DREW["panel"] += 1
                item = group.allow_self_intersection_objects[0]
                # A UIList's draw_item wants a `layout_type`; the probe panel
                # stands in for the list instance so the row runs on a real
                # layout.
                ui_lists.OBJECT_UL_IntersectionAllowanceObjectsList.draw_item(
                    self, context, self.layout, group, item, 0, group,
                    "allow_self_intersection_objects_index", 0,
                )
                DREW["list_row"] += 1
            except Exception as exc:
                DREW["errors"].append("%s: %s" % (type(exc).__name__, exc))

    PPF_PT_AllowanceProbe.layout_type = "DEFAULT"
    bpy.utils.register_class(PPF_PT_AllowanceProbe)

    areas = 0
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "PROPERTIES":
                continue
            for space in area.spaces:
                if space.type == "PROPERTIES":
                    space.context = "OBJECT"
                    areas += 1
            area.tag_redraw()
    if not areas:
        raise RuntimeError(
            "no Properties editor in this screen; the probe has no region to "
            "draw in and a pass would mean nothing"
        )

    # A HEADLESS BLENDER HAS AREAS AND STILL CANNOT DRAW, so the check above
    # is not the one that catches it: `redraw_timer` fails its poll instead,
    # and the raw RuntimeError reads like a scenario defect. PLATFORMS keeps
    # this file off such a leg, and naming the reason here is what a reader
    # gets when the scenario is run past that gate by being named on the
    # command line.
    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=3)
    except RuntimeError as exc:
        raise RuntimeError(
            "this Blender owns no window, so no panel can be drawn (%s). "
            "A draw probe needs the rig's Xvfb or a real window server; this "
            "scenario declares PLATFORMS = ('linux', 'darwin') and a "
            "hand-written scenario list skips that gate." % exc
        )

    dh.record(
        "A_panel_reaches_the_allowance_box",
        DREW["panel"] > 0 and DREW["allowance_box"] >= 2 * DREW["panel"],
        {"draws": {k: v for k, v in DREW.items() if k != "errors"},
         "note": "allowance_box counts BOTH allowances reaching the box on "
                 "every panel draw"},
    )
    dh.record(
        "B_list_row_draws", DREW["list_row"] > 0,
        {"draws": {k: v for k, v in DREW.items() if k != "errors"},
         "listed": [i.name for i in group.allow_self_intersection_objects]},
    )
    dh.record(
        "C_draw_raised_nothing", not DREW["errors"],
        {"errors": DREW["errors"][:4],
         "note": "template_list resolves its collection and index property "
                 "names at draw time, so a rename lands here and nowhere else"},
    )

    bpy.utils.unregister_class(PPF_PT_AllowanceProbe)
    panels._draw_intersection_allowance = _real_draw

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
