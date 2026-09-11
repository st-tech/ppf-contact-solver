# File: scenarios/bl_mcp_deformation_and_bake.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Captured deformation and baking over MCP.
#
# A captured deformation records the motion a Blender modifier or shape key
# already produces, so the solver can treat it as a driven boundary. Capture
# therefore only applies to an object whose vertices actually move, and the
# refusal on a plain mesh is the behavior an agent has to be able to read: it
# names the modifier kinds that qualify instead of recording an empty cache.
#
# Assertions:
#   A. ``status_reads_before_any_capture`` -- both status tools report no
#      cache, no frames and not deforming, for a pin and for a STATIC object.
#   B. ``capture_refuses_a_mesh_that_does_not_move`` -- both capture tools
#      refuse a plain mesh, naming the object and the modifier kinds that
#      would qualify, rather than writing an empty cache.
#   C. ``clear_reports_honestly_with_nothing_captured`` -- the two per-target
#      clears succeed (clearing a target that holds nothing is a no-op that
#      leaves it empty either way), while the scene-wide clear refuses,
#      because it reports on the scene as a whole and there is nothing in it
#      to clear.
#   D. ``recapture_refuses_with_nothing_to_do`` -- recapture_all_deformations
#      says there is nothing to re-capture and names what it applies to.
#   E. ``bake_group_animation_round_trips`` -- baking one object succeeds and
#      an object that is not in the scene is refused by name.
#   F. ``bake_all_needs_a_ui_context`` -- bake_all_animation reports the
#      operator's poll failure in a headless Blender rather than claiming a
#      bake that did not happen.

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

_rid = [1000]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for name, loc in (("Sheet", (0, 0, 0)), ("Ground", (4, 0, 0))):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4,
                                        size=2, location=loc)
        bpy.context.object.name = name

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
    static = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": static, "type": "STATIC"})
    call("add_objects_to_group", {"group_uuid": static, "object_names": ["Ground"]})
    result["phases"].append((time.time(), "scene_built"))

    # ----- A --------------------------------------------------------
    pin_status = call("get_pin_deformation_status",
                      {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge"})
    static_status = call("get_static_deformation_status",
                         {"group_uuid": static, "object_name": "Ground"})
    mcp_check(result, "A_status_reads_before_any_capture",
              pin_status.get("status") == "success"
              and pin_status.get("is_deforming") is False
              and pin_status.get("has_cache") is False
              and pin_status.get("frame_count") == 0
              and pin_status.get("vertex_group") == "Edge"
              and static_status.get("status") == "success"
              and static_status.get("is_deforming") is False
              and static_status.get("has_cache") is False
              and static_status.get("frame_count") == 0,
              {"pin": pin_status, "static": static_status})

    # ----- B --------------------------------------------------------
    pin_capture = call("capture_pin_deformation",
                       {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge"})
    static_capture = call("capture_static_deformation",
                          {"group_uuid": static, "object_name": "Ground"})
    after_b = call("get_pin_deformation_status",
                   {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge"})
    mcp_check(result, "B_capture_refuses_a_mesh_that_does_not_move",
              pin_capture.get("status") == "error"
              and "Sheet" in pin_capture.get("message", "")
              and "Armature" in pin_capture.get("message", "")
              and static_capture.get("status") == "error"
              and "Ground" in static_capture.get("message", "")
              and "Armature" in static_capture.get("message", "")
              # Refused, so no empty cache was recorded.
              and after_b.get("has_cache") is False
              and after_b.get("frame_count") == 0,
              {"pin": pin_capture.get("message"),
               "static": static_capture.get("message"),
               "after": after_b})

    # ----- C --------------------------------------------------------
    clears = {
        "pin": call("clear_pin_deformation",
                    {"group_uuid": shell, "vertex_group_identifier": "Sheet::Edge"}),
        "static": call("clear_static_deformation",
                       {"group_uuid": static, "object_name": "Ground"}),
        "all": call("clear_all_deformations"),
    }
    mcp_check(result, "C_clear_reports_honestly_with_nothing_captured",
              clears["pin"].get("status") == "success"
              and "Sheet::Edge" in clears["pin"].get("message", "")
              and clears["static"].get("status") == "success"
              and "Ground" in clears["static"].get("message", "")
              and clears["all"].get("status") == "error"
              and "No captured deformation cache" in clears["all"].get("message", ""),
              {k: v.get("message") for k, v in clears.items()})

    # ----- D --------------------------------------------------------
    recapture = call("recapture_all_deformations")
    mcp_check(result, "D_recapture_refuses_with_nothing_to_do",
              recapture.get("status") == "error"
              and "re-capture" in recapture.get("message", "")
              and "STATIC" in recapture.get("message", ""),
              {"reply": recapture})

    # ----- E --------------------------------------------------------
    baked = call("bake_group_animation",
                 {"group_uuid": shell, "object_name": "Sheet"})
    missing = call("bake_group_animation",
                   {"group_uuid": shell, "object_name": "Ghost"})
    mcp_check(result, "E_bake_group_animation_round_trips",
              baked.get("status") == "success"
              and baked.get("object_name") == "Sheet"
              and bool(baked.get("object_uuid"))
              and missing.get("status") == "error"
              and "Ghost" in missing.get("message", ""),
              {"baked": baked, "missing": missing.get("message")})

    # ----- F --------------------------------------------------------
    bake_all = call("bake_all_animation")
    mcp_check(result, "F_bake_all_needs_a_ui_context",
              bake_all.get("status") == "error"
              and "poll" in bake_all.get("message", ""),
              {"reply": bake_all})

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
