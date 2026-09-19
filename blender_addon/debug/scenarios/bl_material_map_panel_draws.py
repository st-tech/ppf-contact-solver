# File: scenarios/bl_material_map_panel_draws.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The spatial material map UI draws through a REAL Blender layout.
#
# The encoding scenarios validate the data a map produces and never render it,
# and a registered panel has not necessarily DRAWN. A mock layout would prove
# the draw code runs, but not that Blender's own `template_list`, operator
# property assignment and nested box layout accept what the panel hands them,
# which is where a wrong argument actually surfaces.
#
# The sidebar's category tab cannot be activated from a script
# (`Region.active_panel_category` is read-only), so the draw is probed from a
# temporary Panel in the Properties editor's Object tab, which draws whenever
# that tab is active. Each probe increments a counter as its LAST statement, so
# a draw that raised part way through leaves the counter behind: Blender sends
# a panel exception to stderr rather than raising it into the caller, and a
# scenario that only watched for an exception would pass on a broken draw.
#
#   A. shell_map_panel_drew: the map box, its list, and the selected row's
#      detail column, on the group type that carries every map parameter.
#   B. both_uilists_drew: the map row and the sample row.
#   C. solid_group_drew: the SOLID branch, which adds the note that interior
#      values are extended from the painted surface.
#   D. rod_group_drew: a group whose object_type was changed after the map was
#      authored. The rows survive the change and still reach the encoder, so
#      they have to stay drawable and therefore removable.
#   E. sample_ops: Add seeds the playhead's frame, a second Add at the same
#      frame is refused rather than raising into the UI, and Remove leaves the
#      index on a real row.
#   F. redrew_after_ops: the panel still draws once the operators have run.
#
# Assertion-only: no server / build / run.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

# NOT ON WINDOWS, WHICH RUNS THE RIG HEADLESS AND SO HAS NO MODAL LOOP.
#
# This scenario needs a Blender that owns a window: the PC2 it asserts on is
# written by `PPF_OT_FramePump.modal` AFTER the driver's exec returns, and a
# modal operator needs an event loop to run in. Measured on the Windows leg of
# Blender CI: the driver reached `fetched queued=9 total=9`, the probe recorded
# `modal_seen: []`, and the scenario finished with ZERO checks and no error,
# because nothing it asserts on had been written yet. The drawing scenarios in
# the same set fail one step earlier and say so outright, with "GPU functions
# for drawing requires the gpu module to be initialized".
#
# Two requirements collide here: a full build/run/fetch scenario must NOT be
# run with `--background`, because the modal operator above needs an event
# loop, and the Windows leg of CI has no window server, so it runs headless.
# There is no configuration in which both hold, so this declares where it can
# run rather than failing there every time. Linux gives the rig its own Xvfb
# and macOS has a real window server.
PLATFORMS = ("linux", "darwin")


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

DREW = {"maps": 0, "list_row": 0, "sample_row": 0, "map_box": 0}

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="MapMesh")
    root = dh.configure_state(project_name="map_panel_draws", frame_count=10)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    group = root.object_group_0

    vg = plane.vertex_groups.new(name="Stiffen")
    for i, v in enumerate(plane.data.vertices):
        vg.add([i], 0.5, "REPLACE")

    entry = group.material_maps.add()
    entry.parameter = "bend"
    entry.source_type = "VERTEX_GROUP"
    entry.source_name = "Stiffen"
    entry.target_value = 500.0
    group.material_maps_index = 0
    sample = entry.samples.add()
    sample.frame = 7
    sample.source_type = "VERTEX_GROUP"
    sample.source_name = "Stiffen"
    entry.samples_index = 0
    group.show_parameters = True

    panels = __import__(pkg + ".ui.dynamics.panels", fromlist=["x"])
    ui_lists = __import__(pkg + ".ui.dynamics.ui_lists", fromlist=["x"])

    # Count the map box itself, not just the panel: the real panel draws for
    # every group type, and what has to be proved is that the ROD case still
    # reaches the map list. Deleting the post-chain fallback would otherwise
    # leave this green while a retyped group's rows became unreachable.
    _real_draw_maps = panels._draw_material_maps

    def _counting_draw_maps(*args, **kwargs):
        DREW["map_box"] += 1
        return _real_draw_maps(*args, **kwargs)

    panels._draw_material_maps = _counting_draw_maps

    class PPF_PT_LiveMapProbe(bpy.types.Panel):
        bl_label = "Map Draw Probe"
        bl_idname = "PPF_PT_live_map_probe"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "object"

        def draw(self, context):
            g = bpy.context.scene.zozo_contact_solver.object_group_0
            # The REAL panel, so the `if/elif group.object_type` chain and the
            # post-chain fallback are traversed. Calling `_draw_material_maps`
            # directly would leave the ROD case green even with the fallback
            # deleted, which is the branch that keeps a retyped group's rows
            # reachable.
            panels.DYNAMICS_PT_Groups.draw(self, context)
            DREW["maps"] += 1
            item = g.material_maps[0]
            # A UIList's draw_item wants a `layout_type`; the probe panel
            # stands in for the list instance so the row runs on a real layout.
            ui_lists.OBJECT_UL_MaterialMapsList.draw_item(
                self, context, self.layout, g, item, 0, g,
                "material_maps_index", 0,
            )
            DREW["list_row"] += 1
            ui_lists.OBJECT_UL_MaterialMapSamplesList.draw_item(
                self, context, self.layout, item, item.samples[0], 0, item,
                "samples_index", 0,
            )
            DREW["sample_row"] += 1

    PPF_PT_LiveMapProbe.layout_type = "DEFAULT"
    bpy.utils.register_class(PPF_PT_LiveMapProbe)

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

    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=3)
    dh.record("A_shell_map_panel_drew", DREW["maps"] > 0, {"draws": dict(DREW)})
    dh.record(
        "B_both_uilists_drew",
        DREW["list_row"] > 0 and DREW["sample_row"] > 0,
        {"draws": dict(DREW)},
    )

    for object_type, name in (("SOLID", "C_solid"), ("ROD", "D_rod")):
        group.object_type = object_type
        before = dict(DREW)
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
        dh.record(
            f"{name}_group_drew",
            DREW["maps"] > before["maps"]
            and DREW["map_box"] > before["map_box"]
            and DREW["sample_row"] > before["sample_row"],
            {"object_type": object_type, "draws": dict(DREW),
             "n_rows": len(group.material_maps),
             "note": "map_box counts the map list reaching this group type"},
        )
    group.object_type = "SHELL"

    bpy.context.scene.frame_set(9)
    before_n = len(entry.samples)
    bpy.ops.object.ppf_add_material_map_sample(slot=0)
    added = len(entry.samples) == before_n + 1
    duplicate = bpy.ops.object.ppf_add_material_map_sample(slot=0)
    frames_after_add = [s.frame for s in entry.samples]
    entry.samples_index = 0
    bpy.ops.object.ppf_remove_material_map_sample(slot=0)
    dh.record(
        "E_sample_ops",
        added and "CANCELLED" in duplicate and frames_after_add == [7, 9]
        and entry.samples_index == 0 and len(entry.samples) == 1,
        {"frames_after_add": frames_after_add, "duplicate": list(duplicate),
         "index_after_remove": entry.samples_index,
         "n_after_remove": len(entry.samples)},
    )

    before = dict(DREW)
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
    dh.record(
        "F_redrew_after_ops",
        DREW["maps"] > before["maps"],
        {"draws": dict(DREW)},
    )

    animatable = {
        name: entry.bl_rna.properties[name].is_animatable
        for name in ("parameter", "source_type", "source_name", "target_value",
                     "enabled")
    }
    dh.record(
        "G_no_map_field_offers_a_keyframe",
        not any(animatable.values()),
        {"is_animatable": animatable},
    )

    panels._draw_material_maps = _real_draw_maps
    bpy.utils.unregister_class(PPF_PT_LiveMapProbe)

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
