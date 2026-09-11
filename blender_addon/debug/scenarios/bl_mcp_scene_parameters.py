# File: scenarios/bl_mcp_scene_parameters.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The scene-parameter tools over MCP, against a real Blender.
#
# ``set_scene_parameters`` and ``get_scene_parameters`` are the only way an
# MCP client reaches the global solver settings, so a field the encoder sends
# to the solver but the tools do not carry is invisible to that client: it
# cannot be read, and setting it is impossible rather than refused. Four of
# them are in that class, and this scenario holds the pair to them:
# ``world_scaling``, ``friction_mode``, ``fix_xz`` and ``post_snap_exactly``.
#
# The refusal paths matter as much as the round trip. Both enum and float
# fields land on RNA properties that clamp or reject an assignment silently,
# so a handler that forwarded a bad value would report success for a setting
# the solver never receives. The two checks below pin the loud failure.
#
# Every valid identifier this scenario compares against is read from the RNA
# property itself, never spelled out here, so adding a friction mode does not
# turn this scenario red.
#
# Assertions:
#   A. ``get_reports_encoder_fields`` -- get_scene_parameters carries all four
#      fields, each at its declared JSON type, and friction_mode reports one
#      of the property's own identifiers.
#   B. ``encoder_fields_round_trip`` -- setting each of the four individually
#      reports success and comes back changed from get_scene_parameters.
#   C. ``friction_mode_rejects_unknown`` -- an identifier outside the enum is
#      an isError result whose message names the parameter and every valid
#      mode, and the stored value is left alone.
#   D. ``float_field_rejects_non_numeric`` -- a non-numeric world_scaling is
#      an isError result naming the parameter, and the stored value is left
#      alone.
#   E. ``existing_field_still_round_trips`` -- frame_count, a field that
#      predates the four above, still round-trips through the same pair.
#   F. ``set_schema_declares_encoder_fields`` -- tools/list advertises all
#      four on set_scene_parameters, so a client can discover them.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

# field -> the JSON type get_scene_parameters must report it as.
ENCODER_FIELDS = (
    ("world_scaling", "number"),
    ("friction_mode", "string"),
    ("fix_xz", "number"),
    ("post_snap_exactly", "boolean"),
)


def json_kind(value):
    # bool before number: bool is a subclass of int in Python, and the two
    # are distinct types on the wire.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def same_value(sent, got):
    if isinstance(sent, bool) or isinstance(sent, str):
        return got == sent
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        return False
    return abs(float(got) - float(sent)) <= 1e-6


try:
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    state = groups.get_addon_data(bpy.context.scene).state
    # The same source the handler validates against, so a new mode added to
    # the property is covered here without editing this file.
    valid_modes = [
        item.identifier
        for item in state.bl_rna.properties["friction_mode"].enum_items
    ]

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    def read_params(request_id):
        payload, _ = mcp_tool(
            pkg, url, "get_scene_parameters", {}, request_id=request_id
        )
        params = payload.get("parameters")
        if not isinstance(params, dict):
            raise RuntimeError("get_scene_parameters returned %r" % (payload,))
        return params

    baseline = read_params(1)

    # ----- A. the four fields are reported, at their declared types
    type_report = {}
    for field, expected_kind in ENCODER_FIELDS:
        present = field in baseline
        value = baseline.get(field)
        type_report[field] = {
            "present": present,
            "value": value,
            "kind": json_kind(value) if present else None,
            "expected_kind": expected_kind,
        }
    mode_reported = baseline.get("friction_mode")
    mcp_check(
        result, "A_get_reports_encoder_fields",
        all(
            entry["present"] and entry["kind"] == entry["expected_kind"]
            for entry in type_report.values()
        )
        and mode_reported in valid_modes,
        {
            "fields": type_report,
            "valid_modes": valid_modes,
            "friction_mode_reported": mode_reported,
        },
    )

    # ----- B. each field round-trips on its own -------------------
    other_modes = [m for m in valid_modes if m != baseline.get("friction_mode")]
    if not other_modes:
        raise RuntimeError(
            "friction_mode offers no second identifier: %r" % (valid_modes,)
        )
    # Each target differs from what the field already holds, so a passing
    # round trip is evidence of a write and not of an unchanged default.
    # The floats are exactly representable in single precision, so what the
    # RNA property stores is what the read-back reports.
    scaling_held = same_value(0.5, baseline.get("world_scaling"))
    fix_xz_held = same_value(2.5, baseline.get("fix_xz"))
    targets = {
        "world_scaling": 0.25 if scaling_held else 0.5,
        "fix_xz": 1.5 if fix_xz_held else 2.5,
        "post_snap_exactly": not baseline.get("post_snap_exactly"),
        "friction_mode": other_modes[0],
    }

    round_trip = {}
    request_id = 10
    for field in ("world_scaling", "fix_xz", "post_snap_exactly", "friction_mode"):
        sent = targets[field]
        request_id += 1
        set_payload, set_raw = mcp_tool(
            pkg, url, "set_scene_parameters", {field: sent}, request_id=request_id
        )
        request_id += 1
        after = read_params(request_id)
        got = after.get(field)
        round_trip[field] = {
            "sent": sent,
            "read_back": got,
            "present_after": field in after,
            "was": baseline.get(field),
            "changed": not same_value(sent, baseline.get(field)),
            "matched": bool(field in after and same_value(sent, got)),
            "set_status": set_payload.get("status"),
            "set_is_error": set_raw.get("isError"),
            "updated_parameters": set_payload.get("updated_parameters"),
        }
    mcp_check(
        result, "B_encoder_fields_round_trip",
        all(
            entry["matched"]
            and entry["changed"]
            and entry["set_status"] == "success"
            for entry in round_trip.values()
        ),
        round_trip,
    )

    # ----- C. an identifier outside the enum ----------------------
    # Derived from a real identifier so it cannot accidentally BE one.
    bogus_mode = "NOT_" + valid_modes[0]
    mode_payload, mode_raw = mcp_tool(
        pkg, url, "set_scene_parameters",
        {"friction_mode": bogus_mode}, request_id=30,
    )
    mode_message = mode_payload.get("message") or ""
    after_bogus_mode = read_params(31)
    named_modes = [m for m in valid_modes if m in mode_message]
    mcp_check(
        result, "C_friction_mode_rejects_unknown",
        mode_payload.get("status") == "error"
        and mode_raw.get("isError") is True
        and "friction_mode" in mode_message
        and named_modes == valid_modes
        and after_bogus_mode.get("friction_mode") == targets["friction_mode"],
        {
            "sent": bogus_mode,
            "status": mode_payload.get("status"),
            "isError": mode_raw.get("isError"),
            "message": mode_message,
            "valid_modes": valid_modes,
            "modes_named_in_message": named_modes,
            "friction_mode_after": after_bogus_mode.get("friction_mode"),
            "friction_mode_expected": targets["friction_mode"],
        },
    )

    # ----- D. a float field handed something non-numeric ----------
    bad_float = "not-a-number"
    float_payload, float_raw = mcp_tool(
        pkg, url, "set_scene_parameters",
        {"world_scaling": bad_float}, request_id=40,
    )
    float_message = float_payload.get("message") or ""
    after_bad_float = read_params(41)
    mcp_check(
        result, "D_float_field_rejects_non_numeric",
        float_payload.get("status") == "error"
        and float_raw.get("isError") is True
        and "world_scaling" in float_message
        and same_value(targets["world_scaling"], after_bad_float.get("world_scaling")),
        {
            "sent": bad_float,
            "status": float_payload.get("status"),
            "isError": float_raw.get("isError"),
            "message": float_message,
            "world_scaling_after": after_bad_float.get("world_scaling"),
            "world_scaling_expected": targets["world_scaling"],
        },
    )

    # ----- E. a field that predates the four above ----------------
    frame_count_before = baseline.get("frame_count")
    frame_count_target = (
        int(frame_count_before) + 7 if isinstance(frame_count_before, int) else 97
    )
    fc_payload, _ = mcp_tool(
        pkg, url, "set_scene_parameters",
        {"frame_count": frame_count_target}, request_id=50,
    )
    after_frame_count = read_params(51)
    mcp_check(
        result, "E_existing_field_still_round_trips",
        fc_payload.get("status") == "success"
        and after_frame_count.get("frame_count") == frame_count_target,
        {
            "before": frame_count_before,
            "sent": frame_count_target,
            "read_back": after_frame_count.get("frame_count"),
            "set_status": fc_payload.get("status"),
            "updated_parameters": fc_payload.get("updated_parameters"),
        },
    )

    # ----- F. the setter advertises the four ----------------------
    env, _resp = mcp_call(pkg, url, "tools/list", request_id=60)
    tools = ((env.get("result") or {}).get("tools")) or []
    setter = next(
        (t for t in tools if t.get("name") == "set_scene_parameters"), None
    )
    properties = ((setter or {}).get("inputSchema") or {}).get("properties") or {}
    schema_report = {}
    for field, expected_kind in ENCODER_FIELDS:
        declared = properties.get(field)
        schema_report[field] = {
            "declared": declared is not None,
            "type": (declared or {}).get("type"),
            "expected_type": expected_kind,
        }
    mcp_check(
        result, "F_set_schema_declares_encoder_fields",
        setter is not None
        and all(
            entry["declared"] and entry["type"] == entry["expected_type"]
            for entry in schema_report.values()
        ),
        {
            "setter_found": setter is not None,
            "fields": schema_report,
            "tool_count": len(tools),
        },
    )

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
