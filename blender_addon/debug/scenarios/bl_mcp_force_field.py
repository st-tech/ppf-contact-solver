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
#      sendable and a Magnet with its refusal, beside the grid Transfer would
#      send (a box around the simulated cube grown by the padding, points
#      Spacing apart) and its estimate line.
#   C. bad_values_are_refused: a spacing below a millimeter and a preview
#      resolution below 2 are errors, and nothing is written.
#   D. check_needs_a_server: check_force_field_script without a connection is
#      an error naming the connection.
#   D2. script_builtins_are_listed: get_force_field_settings lists every
#      built-in with its signature, noise and curl_noise among them.
#   E. group_weight_is_settable: set_group_material_properties takes
#      force_field_weight and the group holds it.
#   F. targets_are_settable: set_force_field_targets narrows a field object
#      and the script to a group, get_force_field_settings reports it, null
#      restores every group, and an unknown source is an error.

from __future__ import annotations

from . import _mcp_lib as ml
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
    bpy.ops.object.effector_add(type="MAGNET")
    bpy.context.object.name = "FFMagnet"
    bpy.ops.mesh.primitive_cube_add()
    bpy.context.object.name = "FFBox"
    # A simulated object, which is what a field's sampled box is built around.
    api = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
    api.create_group("FFSolid", "SOLID").add("FFBox")

    src = "def eval(x, y, z, t):\n    return (0.0, 0.0, -1.0)\n"
    tool("set_force_field_settings", {
        "padding": 0.5, "spacing": 0.25, "time_samples": 3,
        "script_source": src, "visualize": True, "preview_resolution": [3, 3, 2]})
    got = tool("get_force_field_settings")
    s = got.get("settings", {})
    text = state.force_field_script
    mcp_check(result, "A_set_then_get_round_trips",
              abs(s.get("padding", 0) - 0.5) < 1e-6 and abs(s.get("spacing", 0) - 0.25) < 1e-6
              and s.get("time_samples") == 3 and s.get("visualize") is True
              and s.get("preview_resolution") == [3, 3, 2]
              and text is not None and text.as_string() == src
              and s.get("script_text") == text.name,
              got)

    fields = {f["object"]: f for f in got.get("fields", [])}
    mcp_check(result, "B_fields_are_listed_with_refusals",
              fields.get("FFTurb", {}).get("refused") is None
              and "FFMagnet" in (fields.get("FFMagnet", {}).get("refused") or "")
              # The cube spans 2 m; grown by 0.5 on each side it is 3 m, so
              # points 0.25 m apart are 13 along every axis.
              and [g.get("shape") for g in got.get("grids", [])] == [[13, 13, 13]]
              and got.get("estimate", "").startswith("[Info] Force field 13x13x13x3:"),
              got)

    e1 = tool_error("set_force_field_settings", {"spacing": 0.0})
    e2 = tool_error("set_force_field_settings", {"preview_resolution": [1, 4, 4]})
    mcp_check(result, "C_bad_values_are_refused",
              e1 is not None and e2 is not None
              and abs(state.force_field_spacing - 0.25) < 1e-6
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

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""
_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
