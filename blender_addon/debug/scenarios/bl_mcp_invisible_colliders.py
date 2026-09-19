# File: scenarios/bl_mcp_invisible_colliders.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The invisible collider family over MCP: walls and spheres that shape the
# motion without appearing in the scene, their properties, and their per-frame
# keyframes.
#
# Assertions:
#   A. ``list_starts_empty`` -- a fresh scene lists no colliders.
#   B. ``wall_and_sphere_are_listed`` -- after add_invisible_wall and
#      add_invisible_sphere the listing carries both, each with its type,
#      position, and the wall's normal / the sphere's radius as given.
#   C. ``properties_round_trip`` -- set_collider_properties changes the
#      sphere's radius and friction and the listing reports the new values.
#   D. ``out_of_range_index_is_refused`` -- set_collider_properties on an
#      index past the end names the valid range.
#   E. ``keyframe_cycle`` -- a keyframe added at frame 10 is listed with the
#      radius given, a duplicate frame is refused by name, and removing it
#      leaves only the implicit first-frame key.
#   F. ``missing_keyframe_is_refused`` -- removing a frame that was never
#      keyed is refused, naming the frame and the collider.
#   G. ``remove_reindexes_and_clear_empties`` -- removing index 0 shifts the
#      sphere to index 0, and clear_invisible_colliders leaves nothing.

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

_rid = [100]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    call("clear_invisible_colliders")

    # ----- A --------------------------------------------------------
    empty = call("list_invisible_colliders")
    mcp_check(result, "A_list_starts_empty",
              empty.get("status") == "success" and empty.get("colliders") == [],
              {"listing": empty})

    # ----- B --------------------------------------------------------
    wall = call("add_invisible_wall", {"position": [0, 0, -1], "normal": [0, 0, 1]})
    sphere = call("add_invisible_sphere", {"position": [1, 2, 3], "radius": 0.5})
    listed = call("list_invisible_colliders").get("colliders") or []
    types = [c.get("type") for c in listed]
    w = listed[0] if listed else {}
    s = listed[1] if len(listed) > 1 else {}
    mcp_check(result, "B_wall_and_sphere_are_listed",
              wall.get("status") == "success" and sphere.get("status") == "success"
              and types == ["WALL", "SPHERE"]
              and w.get("position") == [0.0, 0.0, -1.0] and w.get("normal") == [0.0, 0.0, 1.0]
              and s.get("position") == [1.0, 2.0, 3.0] and abs(s.get("radius", 0) - 0.5) < 1e-6
              and [c.get("index") for c in listed] == [0, 1],
              {"types": types, "wall": w, "sphere": s})

    # ----- C --------------------------------------------------------
    upd = call("set_collider_properties", {"index": 1, "radius": 0.9, "friction": 0.3})
    after = (call("list_invisible_colliders").get("colliders") or [{}, {}])[1]
    mcp_check(result, "C_properties_round_trip",
              upd.get("status") == "success" and upd.get("collider_type") == "SPHERE"
              and abs(after.get("radius", 0) - 0.9) < 1e-6
              and abs(after.get("friction", 0) - 0.3) < 1e-6,
              {"update": upd, "after": after})

    # ----- D --------------------------------------------------------
    bad = call("set_collider_properties", {"index": 7, "radius": 1.0})
    mcp_check(result, "D_out_of_range_index_is_refused",
              bad.get("status") == "error" and "7" in bad.get("message", "")
              and "out of range" in bad.get("message", ""),
              {"reply": bad})

    # ----- E --------------------------------------------------------
    kf = call("add_collider_keyframe", {"index": 1, "frame": 10, "radius": 0.7})
    dup = call("add_collider_keyframe", {"index": 1, "frame": 10, "radius": 0.7})
    listed_kf = call("list_collider_keyframes", {"index": 1}).get("keyframes") or []
    frames = [k.get("frame") for k in listed_kf]
    at10 = [k for k in listed_kf if k.get("frame") == 10]
    rm = call("remove_collider_keyframe", {"index": 1, "frame": 10})
    frames_after = [k.get("frame") for k in (call("list_collider_keyframes", {"index": 1}).get("keyframes") or [])]
    mcp_check(result, "E_keyframe_cycle",
              kf.get("status") == "success" and kf.get("keyframe_count") == 2
              and dup.get("status") == "error" and "already exists" in dup.get("message", "")
              and frames == [1, 10] and at10 and abs(at10[0].get("radius", 0) - 0.7) < 1e-6
              and rm.get("status") == "success" and frames_after == [1],
              {"added": kf, "duplicate": dup, "frames": frames, "after_remove": frames_after})

    # ----- F --------------------------------------------------------
    gone = call("remove_collider_keyframe", {"index": 1, "frame": 99})
    mcp_check(result, "F_missing_keyframe_is_refused",
              gone.get("status") == "error" and "99" in gone.get("message", "")
              and "collider 1" in gone.get("message", ""),
              {"reply": gone})

    # ----- G --------------------------------------------------------
    rm0 = call("remove_invisible_collider", {"index": 0})
    shifted = call("list_invisible_colliders").get("colliders") or []
    cleared = call("clear_invisible_colliders")
    final = call("list_invisible_colliders").get("colliders")
    mcp_check(result, "G_remove_reindexes_and_clear_empties",
              rm0.get("status") == "success"
              and len(shifted) == 1 and shifted[0].get("type") == "SPHERE" and shifted[0].get("index") == 0
              and cleared.get("status") == "success" and final == [],
              {"after_remove": shifted, "final": final})

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
