# File: scenarios/bl_mcp_resources.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP resources surface, against a real Blender.
#
# The add-on serves one resource: the live scene at
# ``blender://scene/current``, read through the main-thread task queue. It
# ships no documentation bundle: a tool's description and input schema are its
# reference. Only a real Blender can show the scene branch: with ``bpy``
# stubbed there is no scene to enumerate.
#
# Assertions:
#   A. ``A_list_is_cacheable`` -- resources/list is a complete result carrying
#      the ttlMs and cacheScope hints a cacheable operation must attach.
#   B. ``B_list_names_the_scene_alone`` -- the list is exactly
#      blender://scene/current, as application/json, with a name.
#   D. ``D_scene_resource_returns_real_scene_data`` -- resources/read of
#      blender://scene/current succeeds and its JSON body carries the scene
#      enumeration keys, not a handler error rendered as content.
#   E. ``E_unknown_uris_are_refused`` -- a URI the server does not serve (a
#      documentation URI included) is -32602, naming the URI.
#   F. ``F_templates_list_is_empty_and_cacheable`` -- this server templates no
#      URI, so resources/templates/list answers with an empty list rather than
#      an error, and carries the caching hints too.

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

try:
    SCENE_URI = "blender://scene/current"

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. the listing is a cacheable, complete result ---------
    env, resp = mcp_call(pkg, url, "resources/list", request_id=1)
    res = env.get("result") or {}
    listed = res.get("resources") or []
    ok, why = mcp_envelope_ok(env, 1)
    mcp_check(
        result, "A_list_is_cacheable",
        ok
        and isinstance(res.get("ttlMs"), int)
        and res.get("ttlMs") >= 0
        and res.get("cacheScope") in ("public", "private")
        and len(listed) > 0,
        {
            "status": resp["status"],
            "envelope_ok": ok,
            "why": why,
            "ttlMs": res.get("ttlMs"),
            "cacheScope": res.get("cacheScope"),
            "resource_count": len(listed),
        },
    )

    # ----- B. the scene is the one resource -----------------------
    by_uri = {}
    for entry in listed:
        if isinstance(entry, dict) and isinstance(entry.get("uri"), str):
            by_uri[entry["uri"]] = entry
    scene_entry = by_uri.get(SCENE_URI) or {}
    mcp_check(
        result, "B_list_names_the_scene_alone",
        sorted(by_uri) == [SCENE_URI]
        and scene_entry.get("mimeType") == "application/json"
        and bool(scene_entry.get("name")),
        {
            "uris": sorted(by_uri),
            "mimeType": scene_entry.get("mimeType"),
            "name": scene_entry.get("name"),
        },
    )

    # ----- D. the scene resource carries scene data, not an error -
    env, resp = mcp_call(
        pkg, url, "resources/read", {"uri": SCENE_URI}, request_id=3, timeout=30.0
    )
    res = env.get("result") or {}
    contents = res.get("contents") or []
    first = contents[0] if contents and isinstance(contents[0], dict) else {}
    ok, why = mcp_envelope_ok(env, 3)
    parse_error = ""
    body = {}
    try:
        parsed = json.loads(first.get("text") or "")
        body = parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        parse_error = "%s: %s" % (type(exc).__name__, exc)
    enumeration_keys = (
        "scene", "simulation", "groups", "group_count", "objects", "object_count",
    )
    missing_keys = [key for key in enumeration_keys if key not in body]
    scene_block = body.get("scene") if isinstance(body.get("scene"), dict) else {}
    objects = body.get("objects") if isinstance(body.get("objects"), list) else None
    mcp_check(
        result, "D_scene_resource_returns_real_scene_data",
        ok
        and first.get("uri") == SCENE_URI
        and first.get("mimeType") == "application/json"
        and not parse_error
        and not missing_keys
        and body.get("status") == "success"
        and isinstance(scene_block.get("name"), str)
        and bool(scene_block.get("name"))
        and objects is not None
        and body.get("object_count") == len(objects)
        and res.get("cacheScope") == "private",
        {
            "status": resp["status"],
            "envelope_ok": ok,
            "why": why,
            "error": env.get("error"),
            "mimeType": first.get("mimeType"),
            "parse_error": parse_error,
            "body_status": body.get("status"),
            "body_message": body.get("message"),
            "missing_keys": missing_keys,
            "scene_name": scene_block.get("name"),
            "object_count": body.get("object_count"),
            "objects_len": None if objects is None else len(objects),
            "cacheScope": res.get("cacheScope"),
            "ttlMs": res.get("ttlMs"),
        },
    )

    # ----- E. a URI the server does not serve --------------------
    refusals = {}
    for request_id, bad_uri in ((4, "llm://index"), (5, "blender://scene/other")):
        env, resp = mcp_call(
            pkg, url, "resources/read", {"uri": bad_uri}, request_id=request_id
        )
        err = env.get("error") or {}
        refusals[bad_uri] = {
            "status": resp["status"],
            "code": err.get("code"),
            "message": err.get("message"),
            "names_uri": bad_uri in (err.get("message") or ""),
        }
    mcp_check(
        result, "E_unknown_uris_are_refused",
        all(
            info["status"] == 400 and info["code"] == -32602 and info["names_uri"]
            for info in refusals.values()
        ),
        refusals,
    )

    # ----- F. no templated URIs, answered rather than refused -----
    env, resp = mcp_call(pkg, url, "resources/templates/list", request_id=6)
    res = env.get("result") or {}
    ok, why = mcp_envelope_ok(env, 6)
    mcp_check(
        result, "F_templates_list_is_empty_and_cacheable",
        ok
        and res.get("resourceTemplates") == []
        and isinstance(res.get("ttlMs"), int)
        and res.get("ttlMs") >= 0
        and res.get("cacheScope") == "public",
        {
            "status": resp["status"],
            "envelope_ok": ok,
            "why": why,
            "resourceTemplates": res.get("resourceTemplates"),
            "ttlMs": res.get("ttlMs"),
            "cacheScope": res.get("cacheScope"),
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
