# File: scenarios/bl_force_field_panel.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The Force Fields panel's rows, WITHOUT DRAWING: the panel's own draw
# function (`ui/dynamics/panels.py` `_draw_force_fields`) is called with a
# recording stand-in for Blender's layout, and the rows it asked for are read
# back. Nothing reaches the GPU or a window, so every platform runs it,
# Windows included, whose rig runs Blender headless. What a row SAYS comes from
# `ui/dynamics/force_field_status.py`, the module the MCP listing and the
# Python API read too, so these checks also hold the three to one wording.
#
# The stand-in has no region, so `draw_wrapped` takes its no-region branch
# and wraps at `force_field_status._FALLBACK_CHARS` characters; the
# region-width branch measures with `blf`, which a background Blender does
# not size (measured: "hello world" reads 6 x 1 pixels), so no headless check
# can hold it.
#
# Subtests:
#   A. type_names_are_blenders: every FieldSettings type (13 on Blender 5.2)
#      heads its row as "<object>: <the name Blender's Physics tab shows>",
#      checked against the names measured on Blender 5.2 ("Magnetic" for
#      MAGNET, "Fluid Flow" for FLUID_FLOW, "Curve Guide" for GUIDE), and the
#      MCP tool's documentation lists the same name for each identifier.
#   B. a_refusal_wraps_whole: a Wind in a scene with Air Density 0 is refused
#      with a sentence longer than one row; the rows under its heading are
#      each within the wrap width, and joined they are the whole sentence, its
#      actionable end ("use a Force field") included. `wrap_lines` cuts a text
#      written without spaces, and a word wider than a row, inside the word.
#   C. a_fluid_flow_row_names_its_baked_domain: a Fluid Flow on a gas domain
#      baked (Cache Type All) over the solve frames shows the domain and its
#      kind, "Cache: All, baked, frames 1 to 11", its resolution, and a size
#      bound that is the brick count of that resolution (an independent
#      count here) and is never below what an encode actually sends for any
#      frame (measured: equal to the largest frame, which keeps all 8 of its
#      bricks, so the bound is tight here). After the encode, "Last read"
#      carries the encode's own report.
#   D. an_unbaked_domain_says_so: a domain that is not baked shows "Cache:
#      All, not baked" and the refusal, and no group rows (a refused field
#      has nothing to narrow).
#   E. a_volume_row_names_its_file_and_grid: a Fluid Flow on a Volume shows
#      the Volume, the file's name, the velocity grid its Volume names, and
#      the resolution of the voxels this scenario wrote (9 x 2 x 18 from a
#      known bounding box); a sequence says so.
#   F. pushes_no_object: a field narrowed to a group with no objects shows
#      "Pushes no object: assign objects to the groups it reaches" with an
#      error icon, and a field that reaches the sheet does not.
#   G. the_whole_box_draws_headless: the panel's Force Fields box draws one
#      row box per field object, in the order the encoder sends them, and the
#      Script and Visualize boxes after them.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
# RUN AGAINST A CPU BUILD on dev-head; it starts no solve, so nothing in it is
# backend-specific.
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import os
import re
import traceback
import types

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
result.setdefault("measured", {})


class Layout:
    # A recording stand-in for a bpy UILayout: every row the panel asks for
    # is kept, in order, and nothing is drawn.

    def __init__(self):
        self.items = []
        self.enabled = True
        self.active = True
        self.alert = False
        self.alignment = "EXPAND"
        self.scale_y = 1.0

    def _sub(self, kind):
        child = Layout()
        self.items.append((kind, child))
        return child

    def box(self):
        return self._sub("box")

    def row(self, **kw):
        return self._sub("row")

    def column(self, **kw):
        return self._sub("column")

    def split(self, **kw):
        return self._sub("split")

    def label(self, text="", icon="NONE", translate=True, **kw):
        self.items.append(("label", text, icon))

    def prop(self, owner, name, **kw):
        self.items.append(("prop", name, kw.get("text")))

    def operator(self, idname, **kw):
        self.items.append(("operator", idname, kw.get("text")))
        return types.SimpleNamespace()

    def operator_menu_enum(self, idname, prop, **kw):
        self.items.append(("operator", idname, kw.get("text")))
        return types.SimpleNamespace()

    def template_list(self, *args, **kw):
        self.items.append(("template_list", args[0], None))

    def separator(self, **kw):
        pass


def flat_labels(layout):
    # Every label under ``layout`` as (text, icon), depth first.
    out = []
    for item in layout.items:
        if item[0] == "label":
            out.append((item[1], item[2]))
        elif isinstance(item[1], Layout):
            out.extend(flat_labels(item[1]))
    return out


def has_item(layout, kind):
    for item in layout.items:
        if item[0] == kind:
            return True
        if isinstance(item[1], Layout) and has_item(item[1], kind):
            return True
    return False


try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["encode"])
    status = __import__(pkg + ".ui.dynamics.force_field_status", fromlist=["type_name"])
    panels = __import__(pkg + ".ui.dynamics.panels", fromlist=["_draw_force_fields"])
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    enc = __import__(pkg + ".core.encoder", fromlist=["resolve_start_frame"])
    mcp_ff = __import__(pkg + ".mcp.handlers.force_field",
                        fromlist=["get_force_field_settings"])
    scene = bpy.context.scene

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0)
    sheet = bpy.context.object
    sheet.name = "FFSheet"
    dh.save_blend(PROBE_DIR, "force_field_panel.blend")
    root = dh.configure_state(project_name="force_field_panel", frame_count=11)
    state = root.state
    state.show_force_field = True
    dh.api.solver.create_group("Cloth", "SHELL").add(sheet.name)

    ctx = types.SimpleNamespace(scene=scene, view_layer=bpy.context.view_layer,
                                region=None, preferences=bpy.context.preferences)

    def draw():
        # The panel's Force Fields box, and each field's row box by name.
        top = Layout()
        panels._draw_force_fields(top, ctx, state)
        box = top.items[0][1]
        rows = {}
        order = []
        for item in box.items:
            if item[0] != "box":
                continue
            labels = flat_labels(item[1])
            head = labels[0][0] if labels else ""
            name = head.split(": ", 1)[0] if ": " in head else None
            if name is not None and name in bpy.data.objects:
                rows[name] = item[1]
                order.append(name)
            else:
                order.append(head)
        return rows, order

    def in_collection(name, objs):
        coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        if coll.name not in scene.collection.children:
            scene.collection.children.link(coll)
        for o in objs:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
        return coll

    def field(kind, name, loc=(0.0, 0.0, 0.0)):
        bpy.ops.object.effector_add(type=kind, location=loc)
        o = bpy.context.object
        o.name = name
        return o

    # ----- A --------------------------------------------------------------
    # Blender 5.2's own names, as its Physics tab shows them (measured).
    BLENDER_NAMES = {
        "BOID": "Boid", "CHARGE": "Charge", "GUIDE": "Curve Guide", "DRAG": "Drag",
        "FLUID_FLOW": "Fluid Flow", "FORCE": "Force", "HARMONIC": "Harmonic",
        "LENNARDJ": "Lennard-Jones", "MAGNET": "Magnetic", "TEXTURE": "Texture",
        "TURBULENCE": "Turbulence", "VORTEX": "Vortex", "WIND": "Wind",
    }
    kinds = [i.identifier for i in bpy.types.FieldSettings.bl_rna.properties["type"].enum_items
             if i.identifier != "NONE"]
    made = []
    for kind in kinds:
        made.append(field("FLUID" if kind == "FLUID_FLOW" else kind, "FFAll_" + kind))
    state.force_field_collection = in_collection("FFAll", made)
    rows, _ = draw()
    heads = {}
    for o in made:
        box = rows.get(o.name)
        heads[o.field.type] = flat_labels(box)[0][0] if box is not None else None
    wrong = {k: v for k, v in heads.items()
             if v != "FFAll_%s: %s" % (k, BLENDER_NAMES.get(k, "?"))}
    # The MCP tool documents the same pairs in its `type` bullet.
    doc = mcp_ff.get_force_field_settings.__doc__ or ""
    bullet = doc.split("- `type`:", 1)[-1].split("- `refused`:", 1)[0]
    documented = {m.group(1): m.group(2).split(",")[0].strip()
                  for m in re.finditer(r"\b([A-Z][A-Z_]+)\s+\(([^)]*)\)", bullet)}
    doc_wrong = {k: documented.get(k) for k in BLENDER_NAMES
                 if documented.get(k) != BLENDER_NAMES[k]}
    dh.record("A_type_names_are_blenders",
              sorted(kinds) == sorted(BLENDER_NAMES) and not wrong and not doc_wrong,
              {"kinds": kinds, "wrong": wrong, "documented_wrong": doc_wrong})

    # ----- B --------------------------------------------------------------
    # configure_state leaves Air Density at 0, so the Wind is refused for the
    # scene around it, in one sentence longer than a row.
    wind_box = rows.get("FFAll_WIND")
    wind_labels = flat_labels(wind_box) if wind_box is not None else []
    why = status.scene_problem(scene, state, bpy.data.objects["FFAll_WIND"])
    body = []
    for text, icon in wind_labels[1:]:
        if icon != "NONE":
            break
        body.append(text)
    width = status._FALLBACK_CHARS
    no_spaces = "力場はこのシーンでは" * 4
    cut = status.wrap_lines(no_spaces, 10, len)
    long_word = "turn " + "x" * 30 + " off"
    cut_word = status.wrap_lines(long_word, 12, len)
    ok = (why is not None and len(why) > width and wind_labels
          and wind_labels[0][1] == "ERROR"
          and len(body) >= 2 and all(len(t) <= width for t in body)
          and " ".join(body) == why and why.endswith("use a Force field")
          and all(len(t) <= 10 for t in cut) and "".join(cut) == no_spaces
          and all(len(t) <= 12 for t in cut_word)
          and "".join(cut_word).replace(" ", "") == long_word.replace(" ", "")
          and status.wrap_lines("", width, len) == [""])
    dh.record("B_a_refusal_wraps_whole", ok,
              {"refusal": why, "rows": body, "width": width, "cut": cut,
               "cut_word": cut_word})

    # ----- C --------------------------------------------------------------
    start = enc.resolve_start_frame(state)
    last = start + state.frame_count - 1

    def bake(dom):
        bake_fluid_domain(scene, dom, result.setdefault("bake_notes", []))

    def gas_domain(name, baked):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
        dom = bpy.context.object
        dom.name = name
        mod = dom.modifiers.new("fluid", "FLUID")
        mod.fluid_type = "DOMAIN"
        ds = mod.domain_settings
        ds.domain_type = "GAS"
        ds.resolution_max = 16
        ds.alpha = ds.beta = 0.0
        ds.gravity = (0.0, 0.0, 0.0)
        ds.use_adaptive_timesteps = False
        ds.cache_type = "ALL"
        ds.cache_directory = os.path.join(PROBE_DIR, "ff_panel_cache_" + name)
        ds.cache_frame_start, ds.cache_frame_end = start, last
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-0.6, 0.0, 0.0))
        src = bpy.context.object
        src.name = name + "Inflow"
        fm = src.modifiers.new("fluid", "FLUID")
        fm.fluid_type = "FLOW"
        fl = fm.flow_settings
        fl.flow_type = "SMOKE"
        fl.flow_behavior = "INFLOW"
        fl.use_initial_velocity = True
        fl.velocity_coord = (1.5, 0.0, 0.0)
        if baked:
            bake(dom)
        return dom

    import openvdb

    vdb_path = os.path.join(PROBE_DIR, "ff_panel_vel.vdb")
    g = openvdb.Vec3SGrid()
    g.name = "vel"
    g.transform = openvdb.createLinearTransform(voxelSize=0.2)
    acc = g.getAccessor()
    # Active voxels spanning index (1, 2, 3) .. (9, 3, 20): 9 x 2 x 18 cells.
    voxels = {(1, 2, 3): (1.0, 2.0, 3.0), (9, 2, 3): (4.0, 0.0, 0.0),
              (1, 3, 20): (0.0, 0.0, 5.0)}
    for ijk, value in voxels.items():
        acc.setValueOn(ijk, value)
    openvdb.write(vdb_path, grids=[g])
    vol_data = bpy.data.volumes.new("FFPanelVol")
    vol_data.filepath = vdb_path
    vol_data.grids.load()
    vol_data.velocity_unit = "SECOND"
    vol_obj = bpy.data.objects.new("FFPanelVol", vol_data)
    scene.collection.objects.link(vol_obj)

    dom = gas_domain("FFPanelDomain", baked=True)
    on_domain = field("FLUID", "FFPanelFluid")
    on_domain.field.source_object = dom
    on_volume = field("FLUID", "FFPanelFluidVol")
    on_volume.field.source_object = vol_obj
    state.force_field_collection = in_collection("FFFluids", [on_domain, on_volume])
    bpy.context.view_layer.update()

    def fluid_rows(name):
        rows, _ = draw()
        box = rows.get(name)
        return (flat_labels(box) if box is not None else []), box

    before, _ = fluid_rows("FFPanelFluid")
    dg = bpy.context.evaluated_depsgraph_get()
    ds_eval = dom.evaluated_get(dg).modifiers["fluid"].domain_settings
    n = [int(v) for v in ds_eval.domain_resolution]
    # The size bound counted here independently: one uint32 table entry per
    # 8^3 brick of the box, and every brick kept, four float32 channels.
    slots = 1
    for v in n:
        slots *= -(-v // 8)
    bound = slots * 4 + slots * 8 * 8 * 8 * 4 * 4
    want_c = [
        ("Domain Object: FFPanelDomain, a gas domain", "MOD_FLUIDSIM"),
        ("Cache: All, baked, frames %d to %d" % (start, last), "BLANK1"),
        ("Resolution at frame %d: %d x %d x %d" % (scene.frame_current, n[0], n[1], n[2]),
         "BLANK1"),
    ]
    # The size row, wrapped across as many labels as its length needs.
    size_text = " ".join(t for t, icon in before[4:] if icon == "BLANK1")
    want_size = ("Up to %s a frame at this resolution, %s over %d solve frames"
                 % (status.format_size(bound), status.format_size(bound * state.frame_count),
                    state.frame_count))
    # What an encode actually sends per frame, which the bound must cover.
    per_frame = []
    payload = None
    for f in params_mod._build_param_dict(bpy.context)["force_field"]["fields"]:
        if f["type"] == "fluid" and f["fluid"]["resolution"][0] == n:
            payload = f["fluid"]
    if payload is not None:
        for t, b in zip(payload["tables"], payload["bricks"]):
            per_frame.append(len(b"".join(t)) + len(b"".join(b)))
    report = {e["field"]: e for e in ff.volume_report()}
    after, _ = fluid_rows("FFPanelFluid")
    rep = report.get("FFPanelFluid", {})
    want_last = ("Last read: %d frames, %s, %s of bricks occupied"
                 % (rep.get("frames", -1), status.format_size(rep.get("bytes", -1)),
                    "{:.0%}".format(rep.get("occupancy", -1.0))))
    # A cube at Resolution Divisions 16 is 16 cells along each axis.
    ok = (n == [16, 16, 16] and before[:1] == [("FFPanelFluid: Fluid Flow", "CHECKMARK")]
          and before[1:4] == want_c and size_text == want_size
          and not any(t.startswith("Last read") for t, _ in before)
          and len(per_frame) == state.frame_count and max(per_frame) <= bound
          and sum(per_frame) == rep.get("bytes")
          and any(t == want_last for t, _ in after))
    result["measured"]["C_bound_over_largest_frame"] = (
        round(bound / max(per_frame), 4) if per_frame else None)
    dh.record("C_a_fluid_flow_row_names_its_baked_domain", ok,
              {"rows_before": before, "rows_after": after, "want": want_c,
               "want_size": want_size, "bound": bound, "per_frame": per_frame,
               "want_last": want_last})

    # ----- E (before D, whose unbaked domain the encode would refuse) ------
    vrows, _ = fluid_rows("FFPanelFluidVol")
    want_e = [
        ("FFPanelFluidVol: Fluid Flow", "CHECKMARK"),
        ("Domain Object: FFPanelVol, a Volume", "VOLUME_DATA"),
        ("File: ff_panel_vel.vdb", "BLANK1"),
        ("Velocity grid: vel", "BLANK1"),
        ("Resolution at frame %d: 9 x 2 x 18" % scene.frame_current, "BLANK1"),
    ]
    vol_data.is_sequence = True
    bpy.context.view_layer.update()
    seq_rows, _ = fluid_rows("FFPanelFluidVol")
    vol_data.is_sequence = False
    bpy.context.view_layer.update()
    dh.record("E_a_volume_row_names_its_file_and_grid",
              vrows[:5] == want_e
              and ("File: ff_panel_vel.vdb, one of a sequence", "BLANK1") in seq_rows,
              {"rows": vrows, "want": want_e, "sequence_rows": seq_rows})

    # ----- D --------------------------------------------------------------
    unbaked = gas_domain("FFPanelUnbaked", baked=False)
    on_unbaked = field("FLUID", "FFPanelFluidUnbaked")
    on_unbaked.field.source_object = unbaked
    in_collection("FFFluids", [on_unbaked])
    bpy.context.view_layer.update()
    urows, ubox = fluid_rows("FFPanelFluidUnbaked")
    why = status.static_refusal(on_unbaked) or ""
    body = []
    for text, icon in urows[1:]:
        if icon != "NONE":
            break
        body.append(text)
    ok = (urows[:1] == [("FFPanelFluidUnbaked: Fluid Flow", "ERROR")]
          and "is not baked" in why and " ".join(body) == why
          and ("Domain Object: FFPanelUnbaked, a gas domain", "MOD_FLUIDSIM") in urows
          and ("Cache: All, not baked", "BLANK1") in urows
          and ubox is not None and not has_item(ubox, "operator"))
    dh.record("D_an_unbaked_domain_says_so", ok,
              {"rows": urows, "refusal": why})

    # ----- F --------------------------------------------------------------
    nobody = dh.api.solver.create_group("Nobody", "SHELL")
    lonely = field("FORCE", "FFPanelLonely")
    in_collection("FFFluids", [lonely])
    dh.api.solver.set_force_field_targets(lonely.name, [nobody])
    rows, order = draw()
    lonely_rows = flat_labels(rows["FFPanelLonely"]) if "FFPanelLonely" in rows else []
    fluid_rows_now = flat_labels(rows["FFPanelFluid"]) if "FFPanelFluid" in rows else []
    pushes_none = ("Pushes no object: assign objects to the groups it reaches", "ERROR")
    dh.record("F_pushes_no_object",
              pushes_none in lonely_rows and pushes_none not in fluid_rows_now
              and ff.pushed_objects(scene, state, lonely) == [],
              {"lonely": lonely_rows})

    # ----- G --------------------------------------------------------------
    names = [o.name for o in ff.field_objects(scene, state)]
    ok = (order[:len(names)] == names and order[len(names):] == ["Script", ""]
          and len(order) == len(names) + 2)
    dh.record("G_the_whole_box_draws_headless", ok, {"order": order, "fields": names})

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
    verdict = r.report_named_checks(result.get("checks", {}))
    verdict.setdefault("notes", []).append(f"measured: {result.get('measured')}")
    return verdict
