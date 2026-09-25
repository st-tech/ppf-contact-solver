# File: scenarios/bl_mcp_tool_schema_invariants.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Schema invariants that must hold across the WHOLE MCP tool list.
#
# ``bl_mcp_transport_conformance`` covers the envelope a ``tools/list`` reply
# travels in. This scenario covers the payload. Every tool schema is generated
# by ``mcp/decorators.py`` out of a handler's signature and docstring, so a
# handler ships a malformed schema without anyone writing one: an empty
# docstring gives a tool with no description, a parameter renamed in the
# signature alone leaves a ``required`` entry naming nothing, and a name in the
# ``get_`` / ``remove_`` families that carries an explicit ``annotations=``
# override silently drops the hint the family implies. None of that is a
# transport error and none of it fails a ``tools/call``, so the list is the
# only place it can be caught.
#
# The assertions read the list as it arrives over the wire rather than the
# in-process registry, because the name ordering is applied on the way out
# and is not visible in the registry.
#
# Assertions:
#   A. ``names_are_present_and_unique`` -- every entry is an object with a
#      non-empty name, and no name is served twice. A duplicate name makes the
#      losing handler unreachable, since a call resolves by name.
#   B. ``titles_are_present`` -- every tool carries a non-empty title, which is
#      what a client shows in place of the identifier.
#   C. ``descriptions_are_present`` -- every description is non-empty: the
#      description and the input schema are a tool's whole reference.
#   D. ``descriptions_point_at_no_document`` -- no description sends a client
#      to an ``llm://`` document: the add-on ships none, and
#      ``resources/list`` serves none, so such a pointer would dangle.
#   E. ``input_schemas_are_object_schemas`` -- every ``inputSchema`` is an
#      object schema whose ``properties`` is an object.
#   F. ``required_is_backed_by_properties`` -- ``required``, where present, is
#      a list of strings that all appear in ``properties``.
#   G. ``list_is_sorted_by_name`` -- the order is deterministic so a client can
#      cache the list and a model's prompt stays stable between calls.
#   H. ``read_only_family_is_annotated`` -- every ``get_*`` / ``list_*`` tool
#      carries ``annotations.readOnlyHint`` true.
#   I. ``destructive_family_is_annotated`` -- every ``remove_*`` / ``delete_*``
#      / ``clear_*`` tool carries ``annotations.destructiveHint`` true.
#   J. ``escape_hatches_are_destructive_and_open_world`` --
#      ``run_python_script`` and ``execute_shell_command`` carry both
#      ``destructiveHint`` and ``openWorldHint`` true. Neither name falls in a
#      family, and no naming rule can imply what they do: they run unsandboxed
#      code and reach outside Blender, so the hints have to be declared.

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

# A regression that hits the generator hits every tool at once, so each
# offender list is reported truncated with its full count beside it.
MAX_REPORTED = 10

READ_ONLY_PREFIXES = ("get_", "list_")
DESTRUCTIVE_PREFIXES = ("remove_", "delete_", "clear_")
ESCAPE_HATCHES = ("execute_shell_command", "run_python_script")

def offenders(items):
    # Report how many failed and enough of them to recognize the pattern.
    items = list(items)
    return {"count": len(items), "sample": items[:MAX_REPORTED]}


try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    env, resp = mcp_call(pkg, url, "tools/list", request_id=1)
    env_ok, env_why = mcp_envelope_ok(env, 1)
    tools = (env.get("result") or {}).get("tools") or []
    entries = [tool for tool in tools if isinstance(tool, dict)]
    names = [tool.get("name") for tool in entries]
    by_name = {}
    for name, tool in zip(names, entries):
        if isinstance(name, str):
            by_name.setdefault(name, tool)
    result["phases"].append((time.time(), "tools_listed n=%d" % len(tools)))

    # Every assertion below is vacuously true on an empty list, so the list
    # having arrived is part of each verdict rather than a separate one.
    listed = env_ok and len(tools) > 0

    # ----- A. one reachable name per tool -------------------------
    non_dict = [i for i, tool in enumerate(tools) if not isinstance(tool, dict)]
    unnamed = [
        repr(name) for name in names
        if not isinstance(name, str) or not name.strip()
    ]
    counts = {}
    for name in names:
        if isinstance(name, str):
            counts[name] = counts.get(name, 0) + 1
    duplicates = sorted(name for name, n in counts.items() if n > 1)
    mcp_check(
        result, "A_names_are_present_and_unique",
        listed and not non_dict and not unnamed and not duplicates,
        {
            "tool_count": len(tools),
            "envelope_ok": env_ok,
            "why": env_why,
            "non_dict_entries": non_dict,
            "unnamed": offenders(unnamed),
            "duplicates": duplicates,
        },
    )

    # ----- B. a display label for every tool ----------------------
    untitled = [
        name for name, tool in zip(names, entries)
        if not isinstance(tool.get("title"), str) or not tool["title"].strip()
    ]
    mcp_check(
        result, "B_titles_are_present",
        listed and not untitled,
        {"tool_count": len(entries), "untitled": offenders(untitled)},
    )

    # ----- C. a description on every tool --------------------------
    undescribed = []
    for name, tool in zip(names, entries):
        text = tool.get("description")
        if not isinstance(text, str) or not text.strip():
            undescribed.append(name)
    mcp_check(
        result, "C_descriptions_are_present",
        listed and not undescribed,
        {
            "tool_count": len(entries),
            "empty_description": offenders(undescribed),
        },
    )

    # ----- D. no pointer to a document the add-on does not ship ---
    env_res, resp_res = mcp_call(pkg, url, "resources/list", request_id=2)
    res_ok, res_why = mcp_envelope_ok(env_res, 2)
    served = sorted(
        entry["uri"]
        for entry in (env_res.get("result") or {}).get("resources") or []
        if isinstance(entry, dict) and isinstance(entry.get("uri"), str)
    )
    pointing = [
        name for name, tool in zip(names, entries)
        if "llm://" in (tool.get("description") or "")
    ]
    mcp_check(
        result, "D_descriptions_point_at_no_document",
        listed and res_ok and not pointing
        and not any(uri.startswith("llm://") for uri in served),
        {
            "served": served,
            "resources_envelope_ok": res_ok,
            "why": res_why,
            "pointing": offenders(pointing),
        },
    )

    # ----- E. the input schema shape a client validates against ---
    malformed = []
    for name, tool in zip(names, entries):
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            malformed.append([name, "inputSchema is %s" % type(schema).__name__])
        elif schema.get("type") != "object":
            malformed.append([name, "type is %r" % (schema.get("type"),)])
        elif not isinstance(schema.get("properties"), dict):
            malformed.append(
                [name, "properties is %s" % type(schema.get("properties")).__name__]
            )
    mcp_check(
        result, "E_input_schemas_are_object_schemas",
        listed and not malformed,
        {"tool_count": len(entries), "malformed": offenders(malformed)},
    )

    # ----- F. every required key is a key a caller can supply -----
    unbacked = []
    for name, tool in zip(names, entries):
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict) or "required" not in schema:
            continue
        required = schema.get("required")
        if not isinstance(required, list) or not all(
            isinstance(key, str) for key in required
        ):
            unbacked.append([name, "required is %r" % (required,)])
            continue
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        missing = [key for key in required if key not in properties]
        if missing:
            unbacked.append([name, missing])
    mcp_check(
        result, "F_required_is_backed_by_properties",
        listed and not unbacked,
        {"tool_count": len(entries), "unbacked": offenders(unbacked)},
    )

    # ----- G. a deterministic order a client can cache ------------
    ordered = [name for name in names if isinstance(name, str)]
    first_descent = None
    for i in range(1, len(ordered)):
        if ordered[i] < ordered[i - 1]:
            first_descent = [ordered[i - 1], ordered[i]]
            break
    mcp_check(
        result, "G_list_is_sorted_by_name",
        listed and first_descent is None,
        {
            "tool_count": len(ordered),
            "first_descent": first_descent,
            "head": ordered[:3],
            "tail": ordered[-3:],
        },
    )

    # ----- H. the read-only family --------------------------------
    read_only = [name for name in ordered if name.startswith(READ_ONLY_PREFIXES)]
    read_only_unhinted = [
        name for name in read_only
        if (by_name[name].get("annotations") or {}).get("readOnlyHint") is not True
    ]
    mcp_check(
        result, "H_read_only_family_is_annotated",
        listed and bool(read_only) and not read_only_unhinted,
        {
            "family_size": len(read_only),
            "prefixes": list(READ_ONLY_PREFIXES),
            "unhinted": offenders(read_only_unhinted),
        },
    )

    # ----- I. the destructive family ------------------------------
    destructive = [name for name in ordered if name.startswith(DESTRUCTIVE_PREFIXES)]
    destructive_unhinted = [
        name for name in destructive
        if (by_name[name].get("annotations") or {}).get("destructiveHint") is not True
    ]
    mcp_check(
        result, "I_destructive_family_is_annotated",
        listed and bool(destructive) and not destructive_unhinted,
        {
            "family_size": len(destructive),
            "prefixes": list(DESTRUCTIVE_PREFIXES),
            "unhinted": offenders(destructive_unhinted),
        },
    )

    # ----- J. the two unsandboxed escape hatches ------------------
    escape = {}
    for name in ESCAPE_HATCHES:
        annotations = (by_name.get(name) or {}).get("annotations") or {}
        escape[name] = {
            "present": name in by_name,
            "destructiveHint": annotations.get("destructiveHint"),
            "openWorldHint": annotations.get("openWorldHint"),
        }
    mcp_check(
        result, "J_escape_hatches_are_destructive_and_open_world",
        listed and all(
            entry["present"]
            and entry["destructiveHint"] is True
            and entry["openWorldHint"] is True
            for entry in escape.values()
        ),
        escape,
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
