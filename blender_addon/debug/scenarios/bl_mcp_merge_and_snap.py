# File: scenarios/bl_mcp_merge_and_snap.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Merge pairs and snapping over MCP. A merge pair stitches two objects
# together where they meet, and carries its own stitch stiffness, which is a
# per-pair solver input distinct from the group-level one.
#
# Assertions:
#   A. ``list_starts_empty`` -- no pairs on a fresh scene.
#   B. ``pair_is_added_once`` -- adding a pair lists it with both names and
#      both UUIDs, and adding the same pair again does not produce a second
#      row (the pair is identified by its two objects).
#   C. ``unknown_object_is_refused`` -- a pair naming an object that is not in
#      the scene is refused, naming that object.
#   D. ``stitch_stiffness_round_trips`` -- set_merge_pair_properties writes
#      the pair's own stiffness and the listing reports it.
#   E. ``negative_stiffness_is_refused`` -- a value below the property's own
#      floor is refused and the stored value does not change.
#   F. ``resnap_reports_the_rows_it_made`` -- resnap_merge_pair stitches and
#      reports how many rows resulted, and snap_to_vertices succeeds.
#   G. ``remove_and_clear`` -- removing the pair succeeds, removing it again
#      is refused by name, and clear empties the list.

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

_rid = [400]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    # Two sheets a unit apart, so a stitch has somewhere to close.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2,
                                    location=(0, 0, 0))
    bpy.context.object.name = "Upper"
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2,
                                    location=(0, 0, -1))
    bpy.context.object.name = "Lower"

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port

    group = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": group, "type": "SHELL"})
    call("add_objects_to_group", {"group_uuid": group, "object_names": ["Upper", "Lower"]})
    call("clear_merge_pairs")
    result["phases"].append((time.time(), "scene_built"))

    # ----- A --------------------------------------------------------
    empty = call("list_merge_pairs")
    mcp_check(result, "A_list_starts_empty",
              empty.get("status") == "success" and empty.get("pairs") == [],
              {"listing": empty})

    # ----- B --------------------------------------------------------
    first = call("add_merge_pair", {"object_a": "Upper", "object_b": "Lower"})
    again = call("add_merge_pair", {"object_a": "Upper", "object_b": "Lower"})
    pairs = call("list_merge_pairs").get("pairs") or []
    p = pairs[0] if pairs else {}
    mcp_check(result, "B_pair_is_added_once",
              first.get("status") == "success"
              and again.get("status") == "success"
              and len(pairs) == 1
              and p.get("object_a") == "Upper" and p.get("object_b") == "Lower"
              and bool(p.get("object_a_uuid")) and bool(p.get("object_b_uuid")),
              {"first": first, "again": again, "pairs": pairs})

    # ----- C --------------------------------------------------------
    unknown = call("add_merge_pair", {"object_a": "Upper", "object_b": "NoSuch"})
    mcp_check(result, "C_unknown_object_is_refused",
              unknown.get("status") == "error" and "NoSuch" in unknown.get("message", "")
              and len(call("list_merge_pairs").get("pairs") or []) == 1,
              {"reply": unknown})

    # ----- D --------------------------------------------------------
    upd = call("set_merge_pair_properties",
               {"object_a": "Upper", "object_b": "Lower", "stitch_stiffness": 0.5})
    after = (call("list_merge_pairs").get("pairs") or [{}])[0]
    mcp_check(result, "D_stitch_stiffness_round_trips",
              upd.get("status") == "success"
              and abs(after.get("stitch_stiffness", 0) - 0.5) < 1e-6,
              {"update": upd, "after": after})

    # ----- E --------------------------------------------------------
    negative = call("set_merge_pair_properties",
                    {"object_a": "Upper", "object_b": "Lower", "stitch_stiffness": -5.0})
    held = (call("list_merge_pairs").get("pairs") or [{}])[0]
    mcp_check(result, "E_negative_stiffness_is_refused",
              negative.get("status") == "error"
              and "0 or greater" in negative.get("message", "")
              # Refused, so the pair still holds what D wrote.
              and abs(held.get("stitch_stiffness", 0) - 0.5) < 1e-6,
              {"reply": negative, "held": held})

    # ----- F --------------------------------------------------------
    resnap = call("resnap_merge_pair", {"object_a": "Upper", "object_b": "Lower"})
    snap = call("snap_to_vertices", {"object_a": "Upper", "object_b": "Lower"})
    mcp_check(result, "F_resnap_reports_the_rows_it_made",
              resnap.get("status") == "success"
              and isinstance(resnap.get("stitch_row_count"), int)
              and resnap.get("stitch_row_count") > 0
              and snap.get("status") == "success",
              {"resnap": resnap, "snap": snap})

    # ----- G --------------------------------------------------------
    rm = call("remove_merge_pair", {"object_a": "Upper", "object_b": "Lower"})
    again_rm = call("remove_merge_pair", {"object_a": "Upper", "object_b": "Lower"})
    cleared = call("clear_merge_pairs")
    final = call("list_merge_pairs").get("pairs")
    mcp_check(result, "G_remove_and_clear",
              rm.get("status") == "success"
              and again_rm.get("status") == "error"
              and "no merge pair" in again_rm.get("message", "")
              and cleared.get("status") == "success" and final == [],
              {"removed": rm, "again": again_rm, "final": final})

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
