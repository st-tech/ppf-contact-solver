# File: scenarios/bl_mcp_object_settings.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The remaining per-object and per-group settings over MCP: pin settings, the
# tetrahedralizer overrides a SOLID object carries, the PDRD hinge, the group
# overlay color, and the particle-mesh conversion.
#
# Assertions:
#   A. ``pin_settings_round_trip`` -- set_pin_settings writes the fields it
#      was given and list_pins reports them back.
#   B. ``unknown_pin_is_refused`` -- a pin identifier no group holds is
#      refused, naming the identifier and the group.
#   C. ``tet_backend_is_validated`` -- a real backend is accepted and reported
#      under updated, and an unknown one names the two that exist.
#   D. ``pdrd_hinge_requires_a_pdrd_group`` -- the hinge sets on a PDRD group
#      and is refused on a SHELL one, because a hinge is a property of the
#      rigid body model that only PDRD uses.
#   E. ``overlay_color_round_trips`` -- set_group_overlay_color reports the
#      four components it stored.
#   F. ``particle_conversion_refuses_when_nothing_fits`` -- a grain radius no
#      grain fits into leaves the object unchanged and says so, and an object
#      that is not in the scene is named.
#   G. ``pin_removal_is_idempotent_in_its_reporting`` -- removing a pin
#      succeeds and reports the new count, and removing it again is refused
#      by name rather than reporting a second success.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

# macOS runners block loopback HTTP to Blender's in-process MCP server, so the
# rig does not select this scenario there.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

_rid = [1100]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2,
                                    location=(0, 0, 0))
    bpy.context.object.name = "Sheet"
    bpy.ops.mesh.primitive_cube_add(size=2, location=(4, 0, 0))
    bpy.context.object.name = "Block"

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    call("clear_solver")

    shell = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": shell, "type": "SHELL"})
    call("add_objects_to_group", {"group_uuid": shell, "object_names": ["Sheet"]})
    call("create_vertex_group",
         {"object_name": "Sheet", "name": "Edge", "indices": [0, 1, 2]})
    call("add_pin_vertex_group",
         {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge"})
    pdrd = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": pdrd, "type": "PDRD"})
    call("add_objects_to_group", {"group_uuid": pdrd, "object_names": ["Block"]})
    result["phases"].append((time.time(), "scene_built"))

    # ----- A --------------------------------------------------------
    settings = call("set_pin_settings",
                    {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge",
                     "use_pin_duration": True, "pin_duration": 30})
    pins = call("list_pins", {"group_uuid": shell}).get("pins") or []
    pin = pins[0] if pins else {}
    mcp_check(result, "A_pin_settings_round_trip",
              settings.get("status") == "success"
              and (settings.get("updates") or {}) == {"use_pin_duration": True,
                                                      "pin_duration": 30}
              and pin.get("use_pin_duration") is True
              and pin.get("pin_duration") == 30,
              {"updates": settings.get("updates"), "pin": pin})

    # ----- B --------------------------------------------------------
    unknown_pin = call("set_pin_settings",
                       {"group_uuid": shell,
                        "vertex_group_identifier": "Sheet::NoSuch",
                        "included": True})
    mcp_check(result, "B_unknown_pin_is_refused",
              unknown_pin.get("status") == "error"
              and "Sheet::NoSuch" in unknown_pin.get("message", "")
              and shell in unknown_pin.get("message", ""),
              {"reply": unknown_pin})

    # ----- C --------------------------------------------------------
    tet = call("set_object_tet_settings",
               {"group_uuid": shell, "object_name": "Sheet", "tet_backend": "TETGEN"})
    bad_tet = call("set_object_tet_settings",
                   {"group_uuid": shell, "object_name": "Sheet",
                    "tet_backend": "BOGUS"})
    mcp_check(result, "C_tet_backend_is_validated",
              tet.get("status") == "success"
              and (tet.get("updated") or {}).get("tet_backend") == "TETGEN"
              and bad_tet.get("status") == "error"
              and "FTETWILD" in bad_tet.get("message", "")
              and "TETGEN" in bad_tet.get("message", ""),
              {"accepted": tet.get("updated"), "refused": bad_tet.get("message")})

    # ----- D --------------------------------------------------------
    hinge = call("set_pdrd_hinge",
                 {"group_uuid": pdrd, "object_name": "Block", "enable": True})
    wrong_group = call("set_pdrd_hinge",
                       {"group_uuid": shell, "object_name": "Sheet"})
    mcp_check(result, "D_pdrd_hinge_requires_a_pdrd_group",
              hinge.get("status") == "success"
              and hinge.get("pdrd_hinge_enable") is True
              and wrong_group.get("status") == "error"
              and "PDRD" in wrong_group.get("message", ""),
              {"hinge": hinge, "refused": wrong_group.get("message")})

    # ----- E --------------------------------------------------------
    color = call("set_group_overlay_color",
                 {"group_uuid": shell, "r": 0.2, "g": 0.4, "b": 0.6, "a": 0.8})
    stored = color.get("color") or []
    mcp_check(result, "E_overlay_color_round_trips",
              color.get("status") == "success"
              and len(stored) == 4
              and all(abs(stored[i] - v) < 1e-6
                      for i, v in enumerate((0.2, 0.4, 0.6, 0.8))),
              {"color": stored})

    # ----- F --------------------------------------------------------
    verts_before = len(bpy.data.objects["Sheet"].data.vertices)
    too_fine = call("convert_to_particle_mesh",
                    {"object_name": "Sheet", "grain_radius": 0.05})
    verts_after = len(bpy.data.objects["Sheet"].data.vertices)
    ghost = call("convert_to_particle_mesh",
                 {"object_name": "Ghost", "grain_radius": 0.2})
    mcp_check(result, "F_particle_conversion_refuses_when_nothing_fits",
              too_fine.get("status") == "error"
              and "unchanged" in too_fine.get("message", "")
              and verts_before == verts_after
              and ghost.get("status") == "error"
              and "Ghost" in ghost.get("message", ""),
              {"refusal": too_fine.get("message"),
               "verts_before": verts_before, "verts_after": verts_after,
               "ghost": ghost.get("message")})

    # ----- G --------------------------------------------------------
    removed = call("remove_pin_vertex_group",
                   {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge"})
    again = call("remove_pin_vertex_group",
                 {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge"})
    mcp_check(result, "G_pin_removal_is_idempotent_in_its_reporting",
              removed.get("status") == "success"
              and removed.get("pin_count") == 0
              and again.get("status") == "error"
              and "Sheet::Edge" in again.get("message", ""),
              {"removed": removed, "again": again.get("message")})

    call("clear_solver")
    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
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
