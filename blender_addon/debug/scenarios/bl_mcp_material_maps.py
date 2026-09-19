# File: scenarios/bl_mcp_material_maps.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The spatial material map tools over MCP, against a real Blender.
#
# A map varies one material parameter across a surface, and the six tools that
# author one (add, set, remove, list, and the two that manage a map's later
# weight sources) are the only way an MCP client reaches that surface. Each one
# validates at authoring time so a caller is told which field to change, rather
# than receiving a build failure from a scene that was already written. This
# scenario drives the whole cycle through the transport and asserts both halves:
# what a map stores, and what the tools refuse.
#
# The valid parameter list is read from ``models/material_maps.py`` rather than
# written out here. That table is the addon's own record of which parameters a
# map may drive, and a scenario holding a second copy would pass while the two
# disagreed.
#
# Assertions:
#   A. ``list_starts_empty`` -- a freshly created SHELL group has no maps, and
#      list_material_maps reports the mappable parameters for its object type,
#      which excludes 'pressure' because that key carries no base property.
#   B. ``add_stores_the_row_it_reports`` -- add_material_map returns index 0 and
#      list reports the parameter, source and target the caller gave, plus the
#      base property the blend runs from and an empty sample list.
#   C. ``set_edits_named_fields_only`` -- set_material_map changes the two
#      fields it was passed and leaves parameter, source type and enabled alone.
#   D. ``sample_is_stored_and_listed`` -- add_material_map_sample stores a later
#      weight source at its own frame, inheriting the map's source type, and
#      list reports it.
#   E. ``pressure_is_refused_by_name`` -- 'pressure' is in the enum but is
#      refused with the documented reason naming the uniform slider to set
#      instead, and nothing is added.
#   F. ``unknown_parameter_is_refused`` -- a parameter outside the enum is
#      refused with a message listing every key a caller can use, and only
#      those: 'pressure' is not offered by a refusal that would refuse it again.
#   G. ``duplicate_sample_frame_is_refused`` -- a second sample on a frame that
#      already has one is refused and the stored sample is untouched.
#   H. ``removal_empties_the_group`` -- remove_material_map_sample drops the
#      sample and remove_material_map drops the map, leaving the group as
#      assertion A found it.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives the addon's in-process MCP server and never asks the solver to
# step, so nothing in it is backend-specific. A rig run against a CPU build
# passed it, and that run is the evidence this line rests on.
BACKENDS = ("real",)

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


def refusal_message(payload):
    # A handler that ran and refused answers with a well-formed result whose
    # payload carries status "error", so the reason is read from the payload
    # and not from a JSON-RPC error frame.
    if not isinstance(payload, dict) or payload.get("status") != "error":
        return ""
    return payload.get("message") or ""


try:
    models_mm = __import__(
        pkg + ".models.material_maps", fromlist=["enum_items", "base_property"]
    )
    handlers_mm = __import__(
        pkg + ".mcp.handlers.material_maps", fromlist=["_PRESSURE_REFUSAL"]
    )
    enum_keys = [key for key, _label, _desc, _num in models_mm.enum_items()]
    usable_keys = [key for key in enum_keys if key != "pressure"]
    shell_keys = [
        key
        for key in enum_keys
        if models_mm.base_property(key, "SHELL") is not None
    ]
    young_base = models_mm.base_property("young-mod", "SHELL")

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    next_id = [10]

    def tool(name, arguments):
        next_id[0] += 1
        payload, _raw = mcp_tool(pkg, url, name, arguments, request_id=next_id[0])
        return payload

    def maps_now():
        return tool("list_material_maps", {"group_uuid": group_uuid})

    # A group of this scenario's own, so a group an earlier scenario left in
    # the same Blender cannot be the one under test.
    tool("delete_all_groups", {})
    created = tool("create_group", {"name": "MapCloth", "type": "SHELL"})
    group_uuid = created.get("group_uuid") or ""
    if not group_uuid:
        raise RuntimeError("create_group returned no group_uuid: %r" % (created,))
    result["phases"].append((time.time(), "group %s" % group_uuid))

    # ----- A. a group carries no maps until one is added ----------
    listed = maps_now()
    start_frame = listed.get("start_frame")
    mcp_check(
        result, "A_list_starts_empty",
        listed.get("status") == "success"
        and listed.get("maps") == []
        and listed.get("map_count") == 0
        and listed.get("object_type") == "SHELL"
        and listed.get("available_parameters") == shell_keys
        and "pressure" not in (listed.get("available_parameters") or [])
        and isinstance(start_frame, int),
        {
            "status": listed.get("status"),
            "maps": listed.get("maps"),
            "map_count": listed.get("map_count"),
            "object_type": listed.get("object_type"),
            "available_parameters": listed.get("available_parameters"),
            "expected_available": shell_keys,
            "start_frame": start_frame,
        },
    )

    # ----- B. what add stores is what list reports ----------------
    added = tool(
        "add_material_map",
        {
            "group_uuid": group_uuid,
            "parameter": "young-mod",
            "source_type": "VERTEX_GROUP",
            "source_name": "StiffPaint",
            "target_value": 2000.0,
        },
    )
    listed = maps_now()
    rows = listed.get("maps") or []
    row = rows[0] if rows else {}
    mcp_check(
        result, "B_add_stores_the_row_it_reports",
        added.get("status") == "success"
        and added.get("index") == 0
        and added.get("map_count") == 1
        and listed.get("map_count") == 1
        and len(rows) == 1
        and row.get("index") == 0
        and row.get("parameter") == "young-mod"
        and row.get("source_type") == "VERTEX_GROUP"
        and row.get("source_name") == "StiffPaint"
        and row.get("target_value") == 2000.0
        and row.get("enabled") is True
        and row.get("base_property") == young_base
        and row.get("gate_closed_reason") is None
        and row.get("samples") == [],
        {
            "added": added,
            "listed_row": row,
            "map_count": listed.get("map_count"),
            "expected_base_property": young_base,
        },
    )

    # ----- C. an edit touches only the fields it names ------------
    updated = tool(
        "set_material_map",
        {
            "group_uuid": group_uuid,
            "index": 0,
            "source_name": "StiffPaintB",
            "target_value": 3500.0,
        },
    )
    listed = maps_now()
    rows = listed.get("maps") or []
    row = rows[0] if rows else {}
    mcp_check(
        result, "C_set_edits_named_fields_only",
        updated.get("status") == "success"
        and listed.get("map_count") == 1
        and row.get("source_name") == "StiffPaintB"
        and row.get("target_value") == 3500.0
        and row.get("parameter") == "young-mod"
        and row.get("source_type") == "VERTEX_GROUP"
        and row.get("enabled") is True,
        {"updated": updated, "listed_row": row},
    )

    # ----- D. a later weight source -------------------------------
    sample_frame = int(start_frame) + 10
    sampled = tool(
        "add_material_map_sample",
        {
            "group_uuid": group_uuid,
            "index": 0,
            "frame": sample_frame,
            "source_name": "StiffPaintLate",
        },
    )
    listed = maps_now()
    rows = listed.get("maps") or []
    row = rows[0] if rows else {}
    # The sample was posted without a source_type, so it takes the map's own.
    expected_samples = [
        {
            "frame": sample_frame,
            "source_type": "VERTEX_GROUP",
            "source_name": "StiffPaintLate",
        }
    ]
    mcp_check(
        result, "D_sample_is_stored_and_listed",
        sampled.get("status") == "success"
        and sampled.get("frame") == sample_frame
        and sampled.get("sample_count") == 1
        and row.get("samples") == expected_samples,
        {
            "sampled": sampled,
            "listed_samples": row.get("samples"),
            "expected_samples": expected_samples,
            "start_frame": start_frame,
        },
    )

    # ----- E. 'pressure' is in the enum and refused by name -------
    refused = tool(
        "add_material_map",
        {
            "group_uuid": group_uuid,
            "parameter": "pressure",
            "source_type": "VERTEX_GROUP",
            "source_name": "PressurePaint",
            "target_value": 50.0,
        },
    )
    message = refusal_message(refused)
    listed = maps_now()
    mcp_check(
        result, "E_pressure_is_refused_by_name",
        message == handlers_mm._PRESSURE_REFUSAL
        and "inflate_pressure" in message
        and "pressure" in enum_keys
        and "pressure" not in shell_keys
        and listed.get("map_count") == 1,
        {
            "message": message,
            "pressure_in_enum": "pressure" in enum_keys,
            "pressure_in_available": "pressure" in shell_keys,
            "map_count_after": listed.get("map_count"),
        },
    )

    # ----- F. a parameter outside the enum ------------------------
    refused = tool(
        "add_material_map",
        {
            "group_uuid": group_uuid,
            "parameter": "not-a-parameter",
            "source_type": "VERTEX_GROUP",
            "source_name": "SomePaint",
            "target_value": 1.0,
        },
    )
    message = refusal_message(refused)
    missing_keys = [key for key in usable_keys if key not in message]
    listed = maps_now()
    mcp_check(
        result, "F_unknown_parameter_is_refused",
        message.startswith("Unknown parameter 'not-a-parameter'")
        and not missing_keys
        and "pressure" not in message
        and listed.get("map_count") == 1,
        {
            "message": message,
            "keys_missing_from_message": missing_keys,
            "expected_keys": usable_keys,
            "map_count_after": listed.get("map_count"),
        },
    )

    # ----- G. a second sample on an occupied frame ----------------
    refused = tool(
        "add_material_map_sample",
        {
            "group_uuid": group_uuid,
            "index": 0,
            "frame": sample_frame,
            "source_name": "StiffPaintLater",
        },
    )
    message = refusal_message(refused)
    listed = maps_now()
    rows = listed.get("maps") or []
    row = rows[0] if rows else {}
    mcp_check(
        result, "G_duplicate_sample_frame_is_refused",
        "already exists" in message
        and str(sample_frame) in message
        and row.get("samples") == expected_samples,
        {
            "message": message,
            "frame": sample_frame,
            "listed_samples": row.get("samples"),
        },
    )

    # ----- H. removal, back to the state A found ------------------
    dropped_sample = tool(
        "remove_material_map_sample",
        {"group_uuid": group_uuid, "index": 0, "frame": sample_frame},
    )
    after_sample = maps_now()
    after_rows = after_sample.get("maps") or []
    dropped_map = tool(
        "remove_material_map", {"group_uuid": group_uuid, "index": 0}
    )
    final = maps_now()
    mcp_check(
        result, "H_removal_empties_the_group",
        dropped_sample.get("status") == "success"
        and dropped_sample.get("sample_count") == 0
        and len(after_rows) == 1
        and after_rows[0].get("samples") == []
        and dropped_map.get("status") == "success"
        and dropped_map.get("map_count") == 0
        and final.get("maps") == []
        and final.get("map_count") == 0,
        {
            "dropped_sample": dropped_sample,
            "rows_after_sample_removal": after_rows,
            "dropped_map": dropped_map,
            "final_maps": final.get("maps"),
            "final_map_count": final.get("map_count"),
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
