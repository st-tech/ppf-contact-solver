# File: scenarios/bl_mcp_dynamic_parameters.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Dynamic parameters over MCP: a scene parameter (gravity, wind, air) that is
# animated with per-frame keyframes rather than held constant.
#
# Assertions:
#   A. ``list_starts_empty`` -- a fresh scene has no dynamic parameters.
#   B. ``add_reports_type_and_seed_keyframe`` -- add_dynamic_param GRAVITY
#      succeeds and seeds one keyframe at frame 1 carrying the default gravity.
#   C. ``duplicate_and_unknown_types_are_refused`` -- adding GRAVITY again is
#      refused, and an unknown type is refused with the valid types named.
#   D. ``keyframe_round_trips`` -- a keyframe at frame 5 with a custom gravity
#      appears in the listing with that vector; a duplicate frame is refused.
#   E. ``keyframe_on_missing_param_is_refused`` -- keying WIND before adding
#      it is refused and tells the caller to add_dynamic_param first.
#   F. ``remove_keyframe_and_param`` -- removing frame 5 leaves the seed key,
#      removing a frame that is not keyed is refused by name, and removing
#      the parameter empties the list.

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

_rid = [200]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port

    # ----- A --------------------------------------------------------
    empty = call("list_dynamic_params")
    mcp_check(result, "A_list_starts_empty",
              empty.get("status") == "success" and empty.get("dynamic_params") == [],
              {"listing": empty})

    # ----- B --------------------------------------------------------
    added = call("add_dynamic_param", {"param_type": "GRAVITY"})
    params = call("list_dynamic_params").get("dynamic_params") or []
    seed = (params[0].get("keyframes") or [{}])[0] if params else {}
    mcp_check(result, "B_add_reports_type_and_seed_keyframe",
              added.get("status") == "success" and added.get("param_type") == "GRAVITY"
              and added.get("keyframe_count") == 1
              and len(params) == 1 and params[0].get("param_type") == "GRAVITY"
              and seed.get("frame") == 1 and isinstance(seed.get("gravity"), list)
              and len(seed["gravity"]) == 3 and seed["gravity"][2] < 0,
              {"added": added, "seed": seed})

    # ----- C --------------------------------------------------------
    dup = call("add_dynamic_param", {"param_type": "GRAVITY"})
    bogus = call("add_dynamic_param", {"param_type": "BOGUS"})
    mcp_check(result, "C_duplicate_and_unknown_types_are_refused",
              dup.get("status") == "error" and "already" in dup.get("message", "")
              and bogus.get("status") == "error"
              and all(t in bogus.get("message", "") for t in ("GRAVITY", "WIND", "AIR_DENSITY")),
              {"duplicate": dup, "unknown": bogus})

    # ----- D --------------------------------------------------------
    kf = call("add_dynamic_param_keyframe", {"param_type": "GRAVITY", "frame": 5, "gravity": [0, 0, -5]})
    dupkf = call("add_dynamic_param_keyframe", {"param_type": "GRAVITY", "frame": 5, "gravity": [0, 0, -5]})
    keys = (call("list_dynamic_params").get("dynamic_params") or [{}])[0].get("keyframes") or []
    at5 = [k for k in keys if k.get("frame") == 5]
    mcp_check(result, "D_keyframe_round_trips",
              kf.get("status") == "success" and kf.get("keyframe_count") == 2
              and [k.get("frame") for k in keys] == [1, 5]
              and at5 and at5[0].get("gravity") == [0.0, 0.0, -5.0]
              and dupkf.get("status") == "error" and "already exists" in dupkf.get("message", ""),
              {"keyframe": kf, "duplicate": dupkf, "keys": keys})

    # ----- E --------------------------------------------------------
    wind = call("add_dynamic_param_keyframe", {"param_type": "WIND", "frame": 5})
    mcp_check(result, "E_keyframe_on_missing_param_is_refused",
              wind.get("status") == "error" and "WIND" in wind.get("message", "")
              and "add_dynamic_param" in wind.get("message", ""),
              {"reply": wind})

    # ----- F --------------------------------------------------------
    rmk = call("remove_dynamic_param_keyframe", {"param_type": "GRAVITY", "frame": 5})
    missing = call("remove_dynamic_param_keyframe", {"param_type": "GRAVITY", "frame": 77})
    after = (call("list_dynamic_params").get("dynamic_params") or [{}])[0].get("keyframes") or []
    rmp = call("remove_dynamic_param", {"param_type": "GRAVITY"})
    final = call("list_dynamic_params").get("dynamic_params")
    mcp_check(result, "F_remove_keyframe_and_param",
              rmk.get("status") == "success" and rmk.get("keyframe_count") == 1
              and missing.get("status") == "error" and "77" in missing.get("message", "")
              and [k.get("frame") for k in after] == [1]
              and rmp.get("status") == "success" and final == [],
              {"remove_key": rmk, "missing": missing, "after": after, "final": final})

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
