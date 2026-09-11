# File: scenarios/bl_mcp_ui_element_status.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# get_ui_element_status, the tool that reports the add-on's operator and
# property surface, against a real Blender.
#
# The tool advertises four categories (solver, dynamics, client, debug) and
# maps each one onto the UI modules registering its operators. A category
# whose entry in that table names no module answers with an empty list, and an
# empty list is indistinguishable from a surface that genuinely holds nothing,
# so a caller learns nothing from it. This scenario is the gate on that: all
# four categories stay populated with operators Blender has actually
# registered, and a category outside the four is refused by name. Only a real
# Blender can show it, because the answer comes from importing the UI modules
# and polling every operator class they register.
#
# Assertions:
#   A. ``every_category_reports_live_operators`` -- each of the four
#      advertised categories answers with a non-empty operator list, and each
#      list holds at least one bl_idname Blender has registered.
#   B. ``unfiltered_covers_every_category`` -- the unfiltered call reports at
#      least as many operators as the four categories summed, and every one of
#      the four is represented among them.
#   C. ``operator_entries_are_identified`` -- every entry carries a name, a
#      bl_idname and a category, and the category is the one requested.
#   D. ``unknown_category_is_refused`` -- an unrecognized category is an error
#      whose message names the four valid ones, not an empty list.
#   E. ``property_surface_is_reported`` -- element_type "property" reports
#      properties, reports no operators, and types and categorizes each entry.
#   F. ``invalid_element_type_is_refused`` -- an element_type outside the three
#      accepted spellings is an error naming them.
#   G. ``dynamics_properties_follow_the_group`` -- the dynamics property
#      surface is read from the scene: after a group is created, the properties
#      filtered to that category are the new group's, keyed by its UUID.

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

CATEGORIES = ("solver", "dynamics", "client", "debug")
TOOL = "get_ui_element_status"


def count_registered(entries):
    # How many reported bl_idnames Blender actually answers to. An entry the
    # tool could not name carries the placeholder "unknown", which is a string
    # like any other, so counting registered operators is what separates a
    # populated surface from a list of classes.
    live = 0
    for entry in entries:
        idname = entry.get("bl_idname") or ""
        head, _, tail = idname.partition(".")
        if tail and hasattr(getattr(bpy.ops, head, None), tail):
            live += 1
    return live


def operator_defects(entries, expected_category):
    # Up to five entries failing the documented operator shape, so a failure
    # names the offenders instead of only counting them.
    defects = []
    for entry in entries:
        if not isinstance(entry, dict):
            defects.append({"entry": repr(entry)[:120], "why": "not an object"})
        elif not isinstance(entry.get("name"), str) or not entry.get("name"):
            defects.append({"entry": entry, "why": "no name"})
        elif not isinstance(entry.get("bl_idname"), str) or not entry.get("bl_idname"):
            defects.append({"entry": entry, "why": "no bl_idname"})
        elif entry.get("category") not in CATEGORIES:
            defects.append({"entry": entry, "why": "category outside the four"})
        elif expected_category and entry.get("category") != expected_category:
            defects.append({"entry": entry, "why": "category is not the one asked for"})
        if len(defects) >= 5:
            break
    return defects


def property_defects(entries):
    defects = []
    for entry in entries:
        if not isinstance(entry, dict):
            defects.append({"entry": repr(entry)[:120], "why": "not an object"})
        elif not isinstance(entry.get("name"), str) or not entry.get("name"):
            defects.append({"entry": entry, "why": "no name"})
        elif "value" not in entry:
            defects.append({"entry": entry, "why": "no value"})
        elif not isinstance(entry.get("type"), str) or not entry.get("type"):
            defects.append({"entry": entry, "why": "no type"})
        elif entry.get("category") not in CATEGORIES:
            defects.append({"entry": entry, "why": "category outside the four"})
        if len(defects) >= 5:
            break
    return defects


try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. every advertised category names live operators ------
    request_id = 1
    listed = {}
    totals = {}
    statuses = {}
    registered = {}
    for category in CATEGORIES:
        payload, _raw = mcp_tool(
            pkg, url, TOOL,
            {"element_type": "operator", "category": category},
            request_id=request_id,
        )
        request_id += 1
        statuses[category] = payload.get("status")
        totals[category] = payload.get("total_operators")
        listed[category] = (payload.get("elements") or {}).get("operators") or []
        registered[category] = count_registered(listed[category])

    mcp_check(
        result, "A_every_category_reports_live_operators",
        all(statuses[c] == "success" for c in CATEGORIES)
        and all(totals[c] == len(listed[c]) for c in CATEGORIES)
        and all(len(listed[c]) > 0 for c in CATEGORIES)
        and all(registered[c] > 0 for c in CATEGORIES),
        {
            "statuses": statuses,
            "reported_totals": totals,
            "listed_lengths": {c: len(listed[c]) for c in CATEGORIES},
            "registered_idnames": registered,
            "first_registered": {
                c: next(
                    (
                        e.get("bl_idname")
                        for e in listed[c]
                        if "." in (e.get("bl_idname") or "")
                    ),
                    None,
                )
                for c in CATEGORIES
            },
        },
    )

    # ----- B. the unfiltered call covers all four -----------------
    all_payload, _raw = mcp_tool(
        pkg, url, TOOL, {"element_type": "operator"}, request_id=request_id,
    )
    request_id += 1
    all_operators = (all_payload.get("elements") or {}).get("operators") or []
    summed = sum(len(listed[c]) for c in CATEGORIES)
    categories_seen = sorted({op.get("category") for op in all_operators})
    mcp_check(
        result, "B_unfiltered_covers_every_category",
        all_payload.get("status") == "success"
        and all_payload.get("total_operators") == len(all_operators)
        and len(all_operators) >= summed
        and categories_seen == sorted(CATEGORIES),
        {
            "status": all_payload.get("status"),
            "unfiltered_total": all_payload.get("total_operators"),
            "unfiltered_listed": len(all_operators),
            "per_category_sum": summed,
            "categories_seen": categories_seen,
        },
    )

    # ----- C. every entry is identified and grouped as asked ------
    defects = {}
    for category in CATEGORIES:
        found = operator_defects(listed[category], category)
        if found:
            defects[category] = found
    unfiltered_defects = operator_defects(all_operators, None)
    missing_keys = sorted(
        {
            key
            for op in all_operators
            if isinstance(op, dict)
            for key in ("name", "bl_idname", "category")
            if key not in op
        }
    )
    mcp_check(
        result, "C_operator_entries_are_identified",
        not defects and not unfiltered_defects and not missing_keys,
        {
            "per_category_defects": defects,
            "unfiltered_defects": unfiltered_defects,
            "missing_keys": missing_keys,
            "inspected": len(all_operators),
            "unnamed_bl_idnames": sum(
                1 for op in all_operators if op.get("bl_idname") == "unknown"
            ),
            "sample": next(
                (op for op in all_operators if "." in (op.get("bl_idname") or "")),
                None,
            ),
        },
    )

    # ----- D. an unrecognized category is refused by name ---------
    bogus_payload, bogus_result = mcp_tool(
        pkg, url, TOOL,
        {"element_type": "operator", "category": "physics"},
        request_id=request_id,
    )
    request_id += 1
    bogus_message = bogus_payload.get("message") or ""
    mcp_check(
        result, "D_unknown_category_is_refused",
        bogus_payload.get("status") == "error"
        and bogus_result.get("isError") is True
        and "elements" not in bogus_payload
        and "physics" in bogus_message
        and all(c in bogus_message for c in CATEGORIES),
        {
            "status": bogus_payload.get("status"),
            "isError": bogus_result.get("isError"),
            "message": bogus_message,
            "payload_keys": sorted(bogus_payload),
        },
    )

    # ----- E. the property surface --------------------------------
    property_payload, _raw = mcp_tool(
        pkg, url, TOOL, {"element_type": "property"}, request_id=request_id,
    )
    request_id += 1
    elements = property_payload.get("elements") or {}
    properties = elements.get("properties") or []
    prop_defects = property_defects(properties)
    mcp_check(
        result, "E_property_surface_is_reported",
        property_payload.get("status") == "success"
        and property_payload.get("total_properties") == len(properties)
        and len(properties) > 0
        and not (elements.get("operators") or [])
        and property_payload.get("total_operators") == 0
        and not prop_defects,
        {
            "status": property_payload.get("status"),
            "total_properties": property_payload.get("total_properties"),
            "total_operators": property_payload.get("total_operators"),
            "listed_properties": len(properties),
            "categories_seen": sorted({p.get("category") for p in properties}),
            "defects": prop_defects,
            "sample": properties[0] if properties else None,
        },
    )

    # ----- F. an element_type outside the three spellings ---------
    bad_type_payload, bad_type_result = mcp_tool(
        pkg, url, TOOL, {"element_type": "widget"}, request_id=request_id,
    )
    request_id += 1
    bad_type_message = bad_type_payload.get("message") or ""
    mcp_check(
        result, "F_invalid_element_type_is_refused",
        bad_type_payload.get("status") == "error"
        and bad_type_result.get("isError") is True
        and "elements" not in bad_type_payload
        and all(word in bad_type_message for word in ("operator", "property", "all")),
        {
            "status": bad_type_payload.get("status"),
            "isError": bad_type_result.get("isError"),
            "message": bad_type_message,
            "payload_keys": sorted(bad_type_payload),
        },
    )

    # ----- G. the dynamics properties are read from the scene -----
    create_payload, _raw = mcp_tool(
        pkg, url, "create_group", {}, request_id=request_id,
    )
    request_id += 1
    group_uuid = create_payload.get("group_uuid") or ""
    dynamics_payload, _raw = mcp_tool(
        pkg, url, TOOL,
        {"element_type": "property", "category": "dynamics"},
        request_id=request_id,
    )
    request_id += 1
    dynamics_properties = (
        (dynamics_payload.get("elements") or {}).get("properties") or []
    )
    group_prefix = "group_%s_" % group_uuid
    for_this_group = [
        p for p in dynamics_properties if (p.get("name") or "").startswith(group_prefix)
    ]
    mcp_check(
        result, "G_dynamics_properties_follow_the_group",
        create_payload.get("status") == "success"
        and bool(group_uuid)
        and dynamics_payload.get("status") == "success"
        and len(for_this_group) > 0
        and all(p.get("category") == "dynamics" for p in dynamics_properties)
        and not property_defects(dynamics_properties),
        {
            "create_status": create_payload.get("status"),
            "group_uuid": group_uuid or None,
            "status": dynamics_payload.get("status"),
            "total_properties": dynamics_payload.get("total_properties"),
            "for_this_group": len(for_this_group),
            "categories_seen": sorted({p.get("category") for p in dynamics_properties}),
            "sample": for_this_group[0] if for_this_group else None,
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
