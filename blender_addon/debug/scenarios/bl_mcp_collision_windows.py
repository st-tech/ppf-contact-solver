# File: scenarios/bl_mcp_collision_windows.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The per-object frame-range families over MCP: collision windows, which say
# when an object takes part in contact, and velocity keyframes and static ops,
# which are the other two ordered per-object lists with the same
# add / list / remove / clear shape.
#
# Assertions:
#   A. ``window_toggle_round_trips`` -- set_use_collision_windows turns the
#      feature on for the group and reports the state it set.
#   B. ``windows_add_and_list_in_order`` -- two windows are listed in the
#      order they were added, with the bounds given.
#   C. ``frame_range_rules_are_enforced`` -- an end before its start, and a
#      bound below frame 1, are each refused with the rule named.
#   D. ``window_remove_and_clear`` -- an out-of-range index names the valid
#      range, removing index 0 shifts the second window down, and clear
#      reports how many it removed.
#   E. ``velocity_keyframe_cycle`` -- a velocity keyframe round-trips through
#      the listing, removing a frame that is not keyed is refused by name,
#      and clear empties the list.
#   F. ``static_op_cycle`` -- a static op on a STATIC group round-trips
#      through list_static_ops and clear_static_ops empties it.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives the addon's in-process MCP server and never asks the solver to
# step, so nothing in it is backend-specific. A rig run against a CPU build
# passed it, and that run is the evidence this line rests on.
BACKENDS = ("real",)

NEEDS_BLENDER = True

# macOS runners block loopback HTTP to Blender's in-process MCP server, so the
# rig does not select this scenario there.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

_rid = [300]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    # A dynamic mesh and a static one, each in a group of the matching type.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=2,
                                    location=(0, 0, 0))
    bpy.context.object.name = "Sheet"
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=2,
                                    location=(4, 0, 0))
    bpy.context.object.name = "Floor"

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port

    shell = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": shell, "type": "SHELL"})
    call("add_objects_to_group", {"group_uuid": shell, "object_names": ["Sheet"]})
    static = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": static, "type": "STATIC"})
    call("add_objects_to_group", {"group_uuid": static, "object_names": ["Floor"]})
    result["phases"].append((time.time(), "groups_built"))

    # ----- A --------------------------------------------------------
    toggled = call("set_use_collision_windows", {"group_uuid": shell, "enable": True})
    mcp_check(result, "A_window_toggle_round_trips",
              toggled.get("status") == "success"
              and toggled.get("use_collision_windows") is True
              and toggled.get("group_uuid") == shell,
              {"reply": toggled})

    # ----- B --------------------------------------------------------
    w1 = call("add_collision_window",
              {"group_uuid": shell, "object_name": "Sheet", "frame_start": 1, "frame_end": 10})
    w2 = call("add_collision_window",
              {"group_uuid": shell, "object_name": "Sheet", "frame_start": 20, "frame_end": 30})
    listed = call("list_collision_windows",
                  {"group_uuid": shell, "object_name": "Sheet"}).get("windows") or []
    mcp_check(result, "B_windows_add_and_list_in_order",
              w1.get("status") == "success" and w1.get("window_count") == 1
              and w2.get("window_count") == 2
              and listed == [{"frame_start": 1, "frame_end": 10},
                             {"frame_start": 20, "frame_end": 30}],
              {"first": w1, "second": w2, "listed": listed})

    # ----- C --------------------------------------------------------
    inverted = call("add_collision_window",
                    {"group_uuid": shell, "object_name": "Sheet",
                     "frame_start": 5, "frame_end": 2})
    below_one = call("add_collision_window",
                     {"group_uuid": shell, "object_name": "Sheet",
                      "frame_start": 0, "frame_end": 2})
    unchanged = call("list_collision_windows",
                     {"group_uuid": shell, "object_name": "Sheet"}).get("windows") or []
    mcp_check(result, "C_frame_range_rules_are_enforced",
              inverted.get("status") == "error"
              and "frame_end must be >= frame_start" in inverted.get("message", "")
              and below_one.get("status") == "error"
              and ">= 1" in below_one.get("message", "")
              # Both refused, so neither reached the list.
              and len(unchanged) == 2,
              {"inverted": inverted, "below_one": below_one, "count": len(unchanged)})

    # ----- D --------------------------------------------------------
    oor = call("remove_collision_window",
               {"group_uuid": shell, "object_name": "Sheet", "index": 5})
    rm = call("remove_collision_window",
              {"group_uuid": shell, "object_name": "Sheet", "index": 0})
    shifted = call("list_collision_windows",
                   {"group_uuid": shell, "object_name": "Sheet"}).get("windows") or []
    cleared = call("clear_collision_windows",
                   {"group_uuid": shell, "object_name": "Sheet"})
    final = call("list_collision_windows",
                 {"group_uuid": shell, "object_name": "Sheet"}).get("windows")
    mcp_check(result, "D_window_remove_and_clear",
              oor.get("status") == "error" and "out of range" in oor.get("message", "")
              and rm.get("status") == "success" and rm.get("window_count") == 1
              and shifted == [{"frame_start": 20, "frame_end": 30}]
              and cleared.get("status") == "success" and "1" in cleared.get("message", "")
              and final == [],
              {"out_of_range": oor, "removed": rm, "shifted": shifted,
               "cleared": cleared, "final": final})

    # ----- E --------------------------------------------------------
    vk = call("add_velocity_keyframe",
              {"group_uuid": shell, "object_name": "Sheet", "frame": 3,
               "direction": [1, 0, 0], "speed": 2.0})
    vlist = call("list_velocity_keyframes",
                 {"group_uuid": shell, "object_name": "Sheet"}).get("keyframes") or []
    vrm = call("remove_velocity_keyframe",
               {"group_uuid": shell, "object_name": "Sheet", "frame": 3})
    vmissing = call("remove_velocity_keyframe",
                    {"group_uuid": shell, "object_name": "Sheet", "frame": 3})
    vclear = call("clear_velocity_keyframes",
                  {"group_uuid": shell, "object_name": "Sheet"})
    mcp_check(result, "E_velocity_keyframe_cycle",
              vk.get("status") == "success" and vk.get("keyframe_count") == 1
              and len(vlist) == 1 and vlist[0].get("frame") == 3
              and vlist[0].get("direction") == [1.0, 0.0, 0.0]
              and abs(vlist[0].get("speed", 0) - 2.0) < 1e-6
              and vrm.get("status") == "success" and vrm.get("keyframe_count") == 0
              and vmissing.get("status") == "error" and "frame 3" in vmissing.get("message", "")
              and vclear.get("status") == "success",
              {"added": vk, "listed": vlist, "removed": vrm, "missing": vmissing})

    # ----- F --------------------------------------------------------
    op = call("add_static_op",
              {"group_uuid": static, "object_name": "Floor", "op_type": "MOVE_BY",
               "frame_start": 1, "frame_end": 5, "delta": [1, 0, 0]})
    ops = call("list_static_ops",
               {"group_uuid": static, "object_name": "Floor"}).get("static_ops") or []
    ops_clear = call("clear_static_ops", {"group_uuid": static, "object_name": "Floor"})
    ops_after = call("list_static_ops",
                     {"group_uuid": static, "object_name": "Floor"}).get("static_ops")
    mcp_check(result, "F_static_op_cycle",
              op.get("status") == "success" and op.get("operation_count") == 1
              and len(ops) == 1 and ops[0].get("op_type") == "MOVE_BY"
              and ops[0].get("frame_start") == 1 and ops[0].get("frame_end") == 5
              and ops[0].get("delta") == [1.0, 0.0, 0.0]
              and ops_clear.get("status") == "success" and ops_after == [],
              {"added": op, "listed": ops, "cleared": ops_clear, "after": ops_after})

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
