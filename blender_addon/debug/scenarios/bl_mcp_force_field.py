# File: scenarios/bl_mcp_force_field.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force field's MCP surface: get_force_field_settings,
# set_force_field_settings, check_force_field_script, and the group's
# force_field_weight through set_group_material_properties.
#
# Subtests:
#   A. set_then_get_round_trips: settings written by the set tool read back
#      from the get tool, and script_source creates the script Text.
#   B. fields_are_listed_with_refusals: a Turbulence object is listed as
#      sendable with the simulated cube it pushes, and a Charge with its
#      refusal, which says it has no effect on cloth.
#   B2. refusals_do_not_repeat_the_name: a Turbulence and a Wind with a
#      Noise Amount are sendable (the noise is this add-on's own); a new
#      Turbulence, whose Size is 0, is refused for it; and no refusal names
#      its own object, since every place that shows one names it already.
#   C. bad_values_are_refused: a preview resolution below 2 is an error and
#      nothing is written, and `spacing`, which the tool does not declare,
#      is an error too.
#   D. check_needs_a_server: check_force_field_script without a connection is
#      an error naming the connection.
#   D2. script_builtins_are_listed: get_force_field_settings lists every
#      built-in with its signature, noise and curl_noise among them.
#   E. group_weight_is_settable: set_group_material_properties takes
#      force_field_weight and the group holds it.
#   F. targets_are_settable: set_force_field_targets narrows a field object
#      and the script to a group, get_force_field_settings reports it, null
#      restores every group, and an unknown source is an error.
#   G. every_type_is_listed_under_blenders_name: one field of each of the 13
#      FieldSettings types is listed with its identifier and the name
#      Blender 5.2's Physics tab shows (measured), and the four with no effect
#      on cloth are refused for it.
#   H. a_baked_domain_is_described: a Fluid Flow on a gas domain baked with
#      Cache Type All over the solve frames lists its source, kind, domain
#      type, cache type, baked state, cache frames, the first solve frame,
#      resolution and a size bound counted here independently (8 bricks of
#      16^3 cells); last_read is null until an encode and then equals the
#      encode's own report, and there is no note.
#   I. a_volume_is_described: a Fluid Flow on a Volume lists the file, the
#      grid its Volume names, no density grid, the resolution of the voxels
#      written here (9 x 2 x 18) and a bound that keeps every brick of that
#      box (a header cannot show which bricks a frame keeps).
#   I2. last_read_belongs_to_its_source: H's field switched to I's Volume
#      reports no last read (the last encode read the domain), and switched
#      back reports H's again.
#   J. transfer_refusals_are_listed_first: what Transfer refuses for the
#      scene or the source rather than the field's settings (Apply Density on
#      a file with no density grid, a solve longer than the bake, a Wind with
#      Air Density 0) the listing says first, in the words Transfer uses. A
#      problem first reached at a later frame is the documented exception:
#      the listing reads the current frame, so it is silent at frame 1 while
#      Transfer refuses at frame 2, and says why once the timeline is there.
#   K. undeclared_arguments_are_refused_by_name: a retired argument and an
#      invented one are each refused naming the argument, with the tool's
#      own arguments listed.
#   L. a_frame_outside_the_solve_is_a_note: what the current frame shows is
#      a refusal only where Transfer reads it. At a frame past the bake and
#      past the solve, the domain's empty grid is a `note` with `refused`
#      and `problem` null, in MCP and in `solver.get_force_fields()` alike,
#      and Transfer succeeds, while the panel draws the note as an INFO row
#      naming the solve frames; at a solve frame there is no note. A Clip
#      sequence of 11 files: at frame 13, outside its clip and the solve, a
#      note; with the solve stretched over frame 13, a refusal there, in the
#      words Transfer's own error holds.
#   M. the_listing_names_the_grid_transfer_reads: a Volume whose file holds
#      the vector grids vel and wind and the float grid speed. Its Velocity
#      Grid set by script to wind is listed as wind and Transfer reads wind's
#      values; set to speed it is refused for being float in the listing and
#      by Transfer; set to a name the file lacks it is refused for naming
#      none.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _driver_lib as dl
from . import _runner as r

BACKENDS = ("real",)
NEEDS_BLENDER = True

# The macOS rig cannot bind the in-process MCP server, as every other bl_mcp_*
# scenario records, so the rig does not select this one there.
PLATFORMS = ("linux", "win32")

_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    state = groups.get_addon_data(bpy.context.scene).state
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port

    ids = iter(range(1, 1000))

    def tool(name, args=None):
        return mcp_tool(pkg, url, name, args or {}, request_id=next(ids))[0]

    def tool_error(name, args):
        envelope, _ = mcp_call(pkg, url, "tools/call",
                               {"name": name, "arguments": args},
                               request_id=next(ids))
        res = envelope.get("result") or {}
        if "error" in envelope:
            return str(envelope["error"])
        if res.get("isError"):
            return str(res.get("content"))
        return None

    bpy.ops.object.effector_add(type="TURBULENCE")
    bpy.context.object.name = "FFTurb"
    # A new Turbulence has Size 0, which is refused.
    bpy.context.object.field.size = 1.0
    bpy.ops.object.effector_add(type="CHARGE")
    bpy.context.object.name = "FFCharge"
    bpy.ops.mesh.primitive_cube_add()
    bpy.context.object.name = "FFBox"
    # A simulated object, which is what a field pushes.
    api = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
    api.create_group("FFSolid", "SOLID").add("FFBox")

    src = "def eval(x, y, z, t):\n    return (0.0, 0.0, -1.0)\n"
    tool("set_force_field_settings", {
        "script_source": src, "visualize": True, "preview_resolution": [3, 3, 2]})
    got = tool("get_force_field_settings")
    s = got.get("settings", {})
    text = state.force_field_script
    mcp_check(result, "A_set_then_get_round_trips",
              s.get("visualize") is True
              and s.get("preview_resolution") == [3, 3, 2]
              and text is not None and text.as_string() == src
              and s.get("script_text") == text.name,
              got)

    fields = {f["object"]: f for f in got.get("fields", [])}
    mcp_check(result, "B_fields_are_listed_with_refusals",
              fields.get("FFTurb", {}).get("refused") is None
              and fields.get("FFTurb", {}).get("pushes") == ["FFBox"]
              and "no effect on cloth" in (fields.get("FFCharge", {}).get("refused") or "")
              and "grids" not in got and "estimate" not in got,
              got)

    bpy.ops.object.effector_add(type="TURBULENCE")
    noisy_turb = bpy.context.object
    noisy_turb.name = "FFTurbNoisy"
    noisy_turb.field.noise = 1.0
    noisy_turb.field.size = 1.0
    bpy.ops.object.effector_add(type="TURBULENCE")
    sizeless = bpy.context.object
    sizeless.name = "FFTurbSizeless"
    bpy.ops.object.effector_add(type="WIND")
    noisy_wind = bpy.context.object
    noisy_wind.name = "FFWindNoisy"
    noisy_wind.field.noise = 1.0
    listed = {f["object"]: f.get("refused")
              for f in tool("get_force_field_settings").get("fields", [])}
    mcp_check(result, "B2_refusals_do_not_repeat_the_name",
              "FFTurbNoisy" in listed and listed["FFTurbNoisy"] is None
              and "FFWindNoisy" in listed and listed["FFWindNoisy"] is None
              # A new Turbulence keeps Blender's default Size of 0.
              and "Size is 0" in (listed.get("FFTurbSizeless") or "")
              and all(why is None or name not in why for name, why in listed.items()),
              listed)
    # Out of the scene again, so the subtests below see the fields they set up.
    bpy.data.objects.remove(noisy_turb)
    bpy.data.objects.remove(noisy_wind)
    bpy.data.objects.remove(sizeless)

    e1 = tool_error("set_force_field_settings", {"spacing": 0.25})
    e2 = tool_error("set_force_field_settings", {"preview_resolution": [1, 4, 4]})
    mcp_check(result, "C_bad_values_are_refused",
              e1 is not None and e2 is not None
              and list(state.force_field_preview_resolution) == [3, 3, 2],
              {"spacing_error": e1, "preview_error": e2})

    rows = got.get("script_builtins") or []
    names = {r.get("name") for r in rows}
    mcp_check(result, "D2_script_builtins_are_listed",
              {"noise", "curl_noise", "sqrt", "atan2", "min", "range"} <= names
              and all(r.get("signature") and r.get("description") for r in rows),
              {"names": sorted(names)})

    e3 = tool_error("check_force_field_script", {})
    mcp_check(result, "D_check_needs_a_server",
              e3 is not None and "Connect" in e3, {"error": e3})

    created = tool("create_group", {"name": "FFCloth", "type": "SHELL"})
    uuid = created.get("group_uuid")
    tool("set_group_material_properties",
         {"group_uuid": uuid, "properties": {"force_field_weight": 0.25}})
    weight = None
    for g in groups.iterate_object_groups(bpy.context.scene):
        if g.name == "FFCloth":
            weight = g.force_field_weight
    mcp_check(result, "E_group_weight_is_settable",
              weight is not None and abs(weight - 0.25) < 1e-6, {"weight": weight})

    tool("set_force_field_targets", {"source": "FFTurb", "group_uuids": [uuid]})
    tool("set_force_field_targets", {"source": "SCRIPT", "group_uuids": [uuid]})
    got = tool("get_force_field_settings")
    narrowed = {f["object"]: f.get("groups") for f in got.get("fields", [])}
    tool("set_force_field_targets", {"source": "SCRIPT", "group_uuids": None})
    restored = tool("get_force_field_settings").get("script_groups")
    e4 = tool_error("set_force_field_targets", {"source": "NoSuchField"})
    mcp_check(result, "F_targets_are_settable",
              narrowed.get("FFTurb") == [uuid] and got.get("script_groups") == [uuid]
              and restored is None and e4 is not None,
              {"narrowed": narrowed, "script": got.get("script_groups"),
               "restored": restored, "unknown": e4})

    # ----- G: every type, under Blender's name ----------------------------
    import os

    import numpy as np
    import openvdb

    scene = bpy.context.scene
    ff = __import__(pkg + ".core.force_field", fromlist=["volume_report"])
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    enc = __import__(pkg + ".core.encoder", fromlist=["resolve_start_frame"])

    def collect(name, objs):
        coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        if coll.name not in scene.collection.children:
            scene.collection.children.link(coll)
        for o in objs:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
        return coll

    def fields_now():
        return {f["object"]: f for f in tool("get_force_field_settings").get("fields", [])}

    def transfer_error():
        try:
            params_mod._build_param_dict(bpy.context)
        except ValueError as e:
            return str(e)
        return None

    # Blender 5.2's own names, as its Physics tab shows them (measured).
    BLENDER_NAMES = {
        "BOID": "Boid", "CHARGE": "Charge", "GUIDE": "Curve Guide", "DRAG": "Drag",
        "FLUID_FLOW": "Fluid Flow", "FORCE": "Force", "HARMONIC": "Harmonic",
        "LENNARDJ": "Lennard-Jones", "MAGNET": "Magnetic", "TEXTURE": "Texture",
        "TURBULENCE": "Turbulence", "VORTEX": "Vortex", "WIND": "Wind",
    }
    every = []
    for kind in sorted(BLENDER_NAMES):
        bpy.ops.object.effector_add(type="FLUID" if kind == "FLUID_FLOW" else kind)
        bpy.context.object.name = "FFEvery_" + kind
        every.append(bpy.context.object)
    collect("FFEveryType", every)
    tool("set_force_field_settings", {"collection": "FFEveryType"})
    listed = fields_now()
    tool("set_force_field_settings", {"collection": ""})
    wrong = {}
    for kind, label in BLENDER_NAMES.items():
        row = listed.get("FFEvery_" + kind, {})
        if row.get("type") != kind or row.get("type_name") != label:
            wrong[kind] = [row.get("type"), row.get("type_name")]
    no_effect = {k: listed.get("FFEvery_" + k, {}).get("refused")
                 for k in ("BOID", "CHARGE", "LENNARDJ", "GUIDE")}
    mcp_check(result, "G_every_type_is_listed_under_blenders_name",
              len(listed) == len(BLENDER_NAMES) and not wrong
              and all("no effect on cloth" in (why or "") for why in no_effect.values()),
              {"wrong": wrong, "no_effect": no_effect, "listed": sorted(listed)})
    for o in every:
        bpy.data.objects.remove(o)

    # ----- H: a Fluid Flow on a baked gas domain ---------------------------
    state.frame_count = 11
    start = enc.resolve_start_frame(state)
    last = start + state.frame_count - 1
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    dom = bpy.context.object
    dom.name = "FFMcpDomain"
    mod = dom.modifiers.new("fluid", "FLUID")
    mod.fluid_type = "DOMAIN"
    ds = mod.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = 16
    ds.alpha = ds.beta = 0.0
    ds.gravity = (0.0, 0.0, 0.0)
    ds.use_adaptive_timesteps = False
    ds.cache_type = "ALL"
    # The scenario's earlier fields (a Turbulence and a Charge) would push
    # the smoke during the bake; this domain is a plain jet.
    ds.effector_weights.all = 0.0
    ds.cache_directory = os.path.join(PROBE_DIR, "ff_mcp_cache")
    ds.cache_frame_start, ds.cache_frame_end = start, last
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-0.6, 0.0, 0.0))
    inflow = bpy.context.object
    inflow.name = "FFMcpInflow"
    fm = inflow.modifiers.new("fluid", "FLUID")
    fm.fluid_type = "FLOW"
    fm.flow_settings.flow_type = "SMOKE"
    fm.flow_settings.flow_behavior = "INFLOW"
    fm.flow_settings.use_initial_velocity = True
    fm.flow_settings.velocity_coord = (1.5, 0.0, 0.0)
    bake_fluid_domain(scene, dom, result.setdefault("bake_notes", []))
    bpy.ops.object.effector_add(type="FLUID")
    on_domain = bpy.context.object
    on_domain.name = "FFMcpFluid"
    on_domain.field.source_object = dom
    collect("FFMcpFluids", [on_domain])
    tool("set_force_field_settings", {"collection": "FFMcpFluids"})
    row = fields_now().get("FFMcpFluid", {})
    fluid = row.get("fluid") or {}
    # The size bound counted independently: one uint32 table entry per 8^3
    # brick of 16^3 cells, and every brick kept, four float32 channels.
    bound = 8 * 4 + 8 * 8 * 8 * 8 * 4 * 4
    want = {"source": "FFMcpDomain", "kind": "domain", "frame": scene.frame_current,
            "solve_start": start, "solve_frames": 11, "resolution": [16, 16, 16],
            "bytes_per_frame_max": bound, "bytes_max": 11 * bound, "last_read": None,
            "problem": None, "note": None,
            "domain_type": "GAS", "cache_type": "ALL", "baked": True,
            "cache_frames": [start, last]}
    sent_error = transfer_error()
    rep = (ff.volume_report() or [{}])[0]
    after = (fields_now().get("FFMcpFluid", {}).get("fluid") or {}).get("last_read")
    want_last = {k: rep.get(k) for k in ("frames", "resolution", "bricks", "brick_slots",
                                         "occupancy", "bytes")}
    mcp_check(result, "H_a_baked_domain_is_described",
              fluid == want and row.get("refused") is None and row.get("tracked") is True
              and row.get("type_name") == "Fluid Flow" and sent_error is None
              and after == want_last and rep.get("frames") == 11,
              {"fluid": fluid, "want": want, "transfer": sent_error, "last_read": after,
               "report": rep})

    # ----- I: a Fluid Flow on a Volume ------------------------------------
    vdb_path = os.path.join(PROBE_DIR, "ff_mcp_vel.vdb")
    g = openvdb.Vec3SGrid()
    g.name = "vel"
    g.transform = openvdb.createLinearTransform(voxelSize=0.2)
    acc = g.getAccessor()
    # Active voxels spanning index (1, 2, 3) .. (9, 3, 20): 9 x 2 x 18 cells.
    for ijk, value in {(1, 2, 3): (1.0, 2.0, 3.0), (9, 2, 3): (4.0, 0.0, 0.0),
                       (1, 3, 20): (0.0, 0.0, 5.0)}.items():
        acc.setValueOn(ijk, value)
    openvdb.write(vdb_path, grids=[g])
    vol_data = bpy.data.volumes.new("FFMcpVol")
    vol_data.filepath = vdb_path
    vol_data.grids.load()
    vol_data.velocity_unit = "SECOND"
    vol_obj = bpy.data.objects.new("FFMcpVol", vol_data)
    scene.collection.objects.link(vol_obj)
    bpy.ops.object.effector_add(type="FLUID")
    on_volume = bpy.context.object
    on_volume.name = "FFMcpFluidVol"
    on_volume.field.source_object = vol_obj
    collect("FFMcpFluids", [on_volume])
    bpy.context.view_layer.update()
    vrow = fields_now().get("FFMcpFluidVol", {})
    vfluid = vrow.get("fluid") or {}
    same_file = bool(vfluid.get("file")) and (
        os.path.normcase(os.path.realpath(vfluid["file"]))
        == os.path.normcase(os.path.realpath(vdb_path)))
    # Six brick slots, every one of them kept: a header cannot say which
    # bricks a frame keeps (a staggered grid, Apply Density's density or a
    # nonzero background each fill bricks with no active velocity voxel).
    vbound = 6 * (4 + 8 * 8 * 8 * 4 * 4)
    mcp_check(result, "I_a_volume_is_described",
              vfluid.get("kind") == "volume" and vfluid.get("source") == "FFMcpVol"
              and same_file and vfluid.get("sequence") is False
              and vfluid.get("velocity_grids") == ["vel"]
              and vfluid.get("density_grid") is None
              and vfluid.get("resolution") == [9, 2, 18]
              and vfluid.get("bytes_per_frame_max") == vbound
              and vrow.get("refused") is None,
              {"fluid": vfluid, "bound": vbound})

    # ----- I2: last_read belongs to the source it read --------------------
    on_domain.field.source_object = vol_obj
    switched = fields_now().get("FFMcpFluid", {}).get("fluid") or {}
    on_domain.field.source_object = dom
    back = (fields_now().get("FFMcpFluid", {}).get("fluid") or {}).get("last_read")
    mcp_check(result, "I2_last_read_belongs_to_its_source",
              switched.get("source") == "FFMcpVol" and switched.get("last_read") is None
              and back == want_last,
              {"switched": switched, "back": back, "want_back": want_last})

    # ----- J: what Transfer refuses, the listing says first ---------------
    cases = {}
    # Apply Density on a file holding no density grid.
    on_volume.field.use_smoke_density = True
    cases["no_density_grid"] = (fields_now()["FFMcpFluidVol"]["refused"], transfer_error())
    on_volume.field.use_smoke_density = False
    # A solve longer than the bake.
    state.frame_count = 20
    cases["baked_range"] = (fields_now()["FFMcpFluid"]["refused"], transfer_error())
    state.frame_count = 11
    # A Wind with no air to blow.
    bpy.ops.object.effector_add(type="WIND")
    wind = bpy.context.object
    wind.name = "FFMcpWind"
    collect("FFMcpFluids", [wind])
    air = state.air_density
    state.air_density = 0.0
    cases["no_air"] = (fields_now()["FFMcpWind"]["refused"], transfer_error())
    state.air_density = air
    bpy.data.objects.remove(wind)
    fragments = {"no_density_grid": "holds no density grid",
                 "baked_range": "is baked over frames %d..%d and the solve reads frames "
                                "%d..%d" % (start, last, start, start + 19),
                 "no_air": "Air Density is 0"}
    agree = {k: bool(listed_why) and fragments[k] in listed_why
             and bool(sent) and listed_why in sent
             for k, (listed_why, sent) in cases.items()}
    # A sequence whose second file holds no velocity grid: the listing reads
    # the current frame only, so it says nothing at frame 1, and Transfer,
    # which reads every solve frame, refuses at frame 2, where the listing
    # then says why too.
    seq_dir = os.path.join(PROBE_DIR, "ff_mcp_seq")
    os.makedirs(seq_dir, exist_ok=True)
    d = openvdb.FloatGrid()
    d.name = "density"
    d.getAccessor().setValueOn((0, 0, 0), 1.0)
    for k in range(1, 12):
        openvdb.write(os.path.join(seq_dir, "s_%04d.vdb" % k), grids=[d if k == 2 else g])
    seq = bpy.data.volumes.new("FFMcpSeq")
    seq.filepath = os.path.join(seq_dir, "s_0001.vdb")
    seq.is_sequence = True
    seq.frame_start = 1
    seq.frame_duration = 11
    seq.grids.load()
    seq.velocity_unit = "SECOND"
    seq_obj = bpy.data.objects.new("FFMcpSeq", seq)
    scene.collection.objects.link(seq_obj)
    on_volume.field.source_object = seq_obj
    scene.frame_set(start)
    at_first = fields_now()["FFMcpFluidVol"]["refused"]
    seq_sent = transfer_error()
    scene.frame_set(start + 1)
    at_second = fields_now()["FFMcpFluidVol"]["refused"]
    scene.frame_set(start)
    cases["later_frame"] = (at_first, seq_sent, at_second)
    agree["later_frame"] = (at_first is None and bool(seq_sent)
                            and ("at frame %d" % (start + 1)) in seq_sent
                            and "names no velocity grid" in (at_second or "")
                            and at_second in seq_sent)
    on_volume.field.source_object = vol_obj
    tool("set_force_field_settings", {"collection": ""})
    mcp_check(result, "J_transfer_refusals_are_listed_first", all(agree.values()),
              {"agree": agree, "cases": cases})

    # ----- K: an argument a tool does not take is refused by name ---------
    e5 = tool_error("set_force_field_settings", {"spacing": 0.25})
    e6 = tool_error("get_force_field_settings", {"zz_not_an_argument": 1})
    mcp_check(result, "K_undeclared_arguments_are_refused_by_name",
              "spacing" in (e5 or "") and "its arguments are" in (e5 or "")
              and "preview_resolution" in (e5 or "")
              and "zz_not_an_argument" in (e6 or ""),
              {"set": e5, "get": e6})

    # ----- L: a finding at a frame Transfer does not read is a note -------
    api_solver = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
    tool("set_force_field_settings", {"collection": "FFMcpFluids"})

    def described(name):
        mcp_row = fields_now().get(name, {})
        api_row = next((r_ for r_ in api_solver.get_force_fields() if r_["object"] == name), {})
        return mcp_row, api_row

    rows = {}
    past = last + 3
    scene.frame_set(past)
    mcp_row, api_row = described("FFMcpFluid")
    note = "its Domain Object 'FFMcpDomain' holds no fluid at frame %d" % past
    rows["domain_past_the_bake"] = {
        "refused": [mcp_row.get("refused"), api_row.get("refused")],
        "problem": [(mcp_row.get("fluid") or {}).get("problem"),
                    (api_row.get("fluid") or {}).get("problem")],
        "note": [(mcp_row.get("fluid") or {}).get("note"),
                 (api_row.get("fluid") or {}).get("note")],
        "transfer": transfer_error()}
    panel = __import__(pkg + ".ui.dynamics.force_field_status", fromlist=["fluid_lines"])
    lines = panel.fluid_lines(panel.fluid_info(on_domain, scene, state,
                                               bpy.context.evaluated_depsgraph_get()))
    info_row = "Not a solve frame (the solve reads frames %d to %d): %s" % (start, last, note)
    rows["domain_past_the_bake"]["panel_info_row"] = [text for text, icon in lines
                                                      if icon == "INFO"]
    scene.frame_set(start)
    rows["domain_at_a_solve_frame"] = {"note": (fields_now()["FFMcpFluid"].get("fluid")
                                                 or {}).get("note")}
    clip_dir = os.path.join(PROBE_DIR, "ff_mcp_clip")
    os.makedirs(clip_dir, exist_ok=True)
    for k in range(1, 12):
        openvdb.write(os.path.join(clip_dir, "c_%04d.vdb" % k), grids=[g])
    clip = bpy.data.volumes.new("FFMcpClip")
    clip.filepath = os.path.join(clip_dir, "c_0001.vdb")
    clip.is_sequence = True
    clip.frame_start = 1
    clip.frame_duration = 11
    clip.sequence_mode = "CLIP"
    clip.grids.load()
    clip.velocity_unit = "SECOND"
    clip_obj = bpy.data.objects.new("FFMcpClip", clip)
    scene.collection.objects.link(clip_obj)
    # Alone in the listing, so the solve stretched past the domain's bake
    # below is refused for the sequence rather than for the domain.
    collect("FFMcpClipOnly", [on_volume])
    tool("set_force_field_settings", {"collection": "FFMcpClipOnly"})
    on_volume.field.source_object = clip_obj
    scene.frame_set(13)
    mcp_row, api_row = described("FFMcpFluidVol")
    rows["clip_outside_the_solve"] = {
        "refused": [mcp_row.get("refused"), api_row.get("refused")],
        "note": [(mcp_row.get("fluid") or {}).get("note"),
                 (api_row.get("fluid") or {}).get("note")],
        "transfer": transfer_error()}
    state.frame_count = 13
    mcp_row, api_row = described("FFMcpFluidVol")
    rows["clip_at_a_solve_frame"] = {
        "refused": [mcp_row.get("refused"), api_row.get("refused")],
        "note": (mcp_row.get("fluid") or {}).get("note"), "transfer": transfer_error()}
    state.frame_count = 11
    scene.frame_set(start)
    on_volume.field.source_object = vol_obj
    collect("FFMcpFluids", [on_volume])
    tool("set_force_field_settings", {"collection": "FFMcpFluids"})
    d_ = rows["domain_past_the_bake"]
    c_ = rows["clip_outside_the_solve"]
    s_ = rows["clip_at_a_solve_frame"]
    clip_note = "its Volume 'FFMcpClip' resolves no file at frame 13"
    mcp_check(result, "L_a_frame_outside_the_solve_is_a_note",
              start + 10 < 13 and d_["refused"] == [None, None]
              and d_["problem"] == [None, None] and d_["note"] == [note, note]
              and d_["transfer"] is None and d_["panel_info_row"] == [info_row]
              and rows["domain_at_a_solve_frame"]["note"] is None
              and c_["refused"] == [None, None]
              and all((n_ or "").startswith(clip_note) for n_ in c_["note"])
              and c_["transfer"] is None
              and s_["refused"][0] is not None and s_["refused"][0] == s_["refused"][1]
              and "outside its sequence" in s_["refused"][0] and s_["note"] is None
              and bool(s_["transfer"]) and s_["refused"][0] in s_["transfer"],
              rows)

    # ----- M: the grid listed is the grid Transfer reads ------------------
    three_path = os.path.join(PROBE_DIR, "ff_mcp_three.vdb")
    wind_grid = openvdb.Vec3SGrid()
    wind_grid.name = "wind"
    wind_grid.transform = openvdb.createLinearTransform(voxelSize=0.2)
    wind_grid.getAccessor().setValueOn((1, 2, 3), (5.0, 0.0, 0.0))
    speed_grid = openvdb.FloatGrid()
    speed_grid.name = "speed"
    speed_grid.transform = openvdb.createLinearTransform(voxelSize=0.2)
    speed_grid.getAccessor().setValueOn((1, 2, 3), 1.0)
    openvdb.write(three_path, grids=[g, wind_grid, speed_grid])
    three = bpy.data.volumes.new("FFMcpThree")
    three.filepath = three_path
    three.grids.load()
    three.velocity_unit = "SECOND"
    three_obj = bpy.data.objects.new("FFMcpThree", three)
    scene.collection.objects.link(three_obj)
    on_volume.field.source_object = three_obj
    bpy.context.view_layer.update()

    def listed_grid():
        row_ = fields_now()["FFMcpFluidVol"]
        return (row_.get("fluid") or {}).get("velocity_grids"), row_.get("refused")

    grids_rows = {"first": listed_grid()}
    three.velocity_grid = "wind"
    grids_rows["wind"] = listed_grid()
    sent = ff.fluid_frame(on_volume.field, on_volume.name, rate=1.0)
    grids_rows["wind_sent_speed"] = float(np.abs(sent.bricks[..., :3]).max())
    three.velocity_grid = "speed"
    grids_rows["speed"] = listed_grid()
    grids_rows["speed_transfer"] = transfer_error()
    three.velocity_grid = "zz_no_such_grid"
    grids_rows["missing"] = listed_grid()
    three.velocity_grid = "vel"
    on_volume.field.source_object = vol_obj
    tool("set_force_field_settings", {"collection": ""})
    sp = grids_rows["speed"][1] or ""
    # The listed reason is Transfer's, word for word: both ask
    # `core.force_field.vdb_grid_problem`.
    mcp_check(result, "M_the_listing_names_the_grid_transfer_reads",
              grids_rows["first"] == (["vel"], None) and grids_rows["wind"] == (["wind"], None)
              and abs(grids_rows["wind_sent_speed"] - 5.0) < 1e-6
              and "the velocity grid 'speed'" in sp and "is float, not vec3s or vec3d" in sp
              and bool(grids_rows["speed_transfer"]) and sp in grids_rows["speed_transfer"]
              and "its Volume 'FFMcpThree' names no velocity grid"
              in (grids_rows["missing"][1] or ""),
              grids_rows)

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""
_DRIVER_TEMPLATE = ml.MCP_LIB + dl.FLUID_BAKE_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
