# File: scenarios/bl_mcp_resources.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP resources surface, against a real Blender.
#
# The add-on serves two families of resource from one namespace: the live
# scene at ``blender://scene/current``, which is read through the main-thread
# task queue, and the ``llm://`` documentation bundle, which is read straight
# off the add-on tree. Only a real Blender can show both: with ``bpy`` stubbed
# the scene branch has no scene to enumerate, and the doc branch has no
# installed tree to resolve a URI against.
#
# Assertions:
#   A. ``A_list_is_cacheable`` -- resources/list is a complete result carrying
#      the ttlMs and cacheScope hints a cacheable operation must attach.
#   B. ``B_list_names_scene_and_every_shipped_doc`` -- the list names
#      blender://scene/current as application/json, and the llm:// entries are
#      exactly the markdown files on the installed tree, each with a name.
#   C. ``C_index_reads_as_public_markdown`` -- resources/read of llm://index
#      returns one text/markdown content block with non-empty text, scoped
#      public because the bundle is identical for every caller.
#   D. ``D_scene_resource_returns_real_scene_data`` -- resources/read of
#      blender://scene/current succeeds and its JSON body carries the scene
#      enumeration keys, not a handler error rendered as content.
#   E. ``E_unknown_and_escaping_uris_are_refused`` -- a doc URI that names no
#      file, and one that tries to climb out of the bundle, are both -32602.
#   F. ``F_templates_list_is_empty_and_cacheable`` -- this server templates no
#      URI, so resources/templates/list answers with an empty list rather than
#      an error, and carries the caching hints too.
#   G. ``G_every_advertised_doc_reads_non_empty`` -- every llm:// URI the list
#      advertises reads back as non-empty markdown, so a doc file that was
#      moved, emptied or left unreadable cannot ship.

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
    import pathlib

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

    # ----- B. both families are advertised ------------------------
    by_uri = {}
    for entry in listed:
        if isinstance(entry, dict) and isinstance(entry.get("uri"), str):
            by_uri[entry["uri"]] = entry
    scene_entry = by_uri.get(SCENE_URI) or {}
    doc_uris = sorted(uri for uri in by_uri if uri.startswith("llm://"))

    # Derive the shipped doc set from the installed tree rather than from the
    # enumerator under test, so a file that is present but never advertised is
    # a failure instead of an agreement between the code and itself.
    llm_mod = __import__(
        pkg + ".mcp.llm_resources", fromlist=["list_llm_resources"]
    )
    addon_root = pathlib.Path(llm_mod.__file__).resolve().parent.parent
    llm_dir = addon_root / "LLM"
    on_disk = set()
    if (addon_root / "LLM.md").is_file():
        on_disk.add("llm://index")
    if llm_dir.is_dir():
        for path in sorted(llm_dir.rglob("*.md")):
            parts = path.relative_to(llm_dir).with_suffix("").parts
            if parts and parts[0] == "blender_addon":
                parts = parts[1:]
            on_disk.add("llm://" + "/".join(parts))
    doc_meta_faults = sorted(
        uri for uri in doc_uris
        if by_uri[uri].get("mimeType") != "text/markdown"
        or not by_uri[uri].get("name")
    )
    mcp_check(
        result, "B_list_names_scene_and_every_shipped_doc",
        scene_entry.get("mimeType") == "application/json"
        and bool(scene_entry.get("name"))
        and "llm://index" in by_uri
        and len(on_disk) > 1
        and set(doc_uris) == on_disk
        and not doc_meta_faults,
        {
            "scene_entry": {
                "present": bool(scene_entry),
                "mimeType": scene_entry.get("mimeType"),
                "name": scene_entry.get("name"),
            },
            "doc_count": len(doc_uris),
            "advertised_not_on_disk": sorted(set(doc_uris) - on_disk),
            "on_disk_not_advertised": sorted(on_disk - set(doc_uris)),
            "doc_meta_faults": doc_meta_faults,
        },
    )

    # ----- C. the doc index reads back as public markdown ---------
    env, resp = mcp_call(
        pkg, url, "resources/read", {"uri": "llm://index"}, request_id=2
    )
    res = env.get("result") or {}
    contents = res.get("contents") or []
    first = contents[0] if contents and isinstance(contents[0], dict) else {}
    index_text = first.get("text") or ""
    ok, why = mcp_envelope_ok(env, 2)
    mcp_check(
        result, "C_index_reads_as_public_markdown",
        ok
        and len(contents) == 1
        and first.get("uri") == "llm://index"
        and first.get("mimeType") == "text/markdown"
        and bool(index_text.strip())
        and res.get("cacheScope") == "public"
        and isinstance(res.get("ttlMs"), int)
        and res.get("ttlMs") >= 0,
        {
            "status": resp["status"],
            "envelope_ok": ok,
            "why": why,
            "content_blocks": len(contents),
            "uri": first.get("uri"),
            "mimeType": first.get("mimeType"),
            "chars": len(index_text.strip()),
            "first_line": index_text.splitlines()[0][:80] if index_text else "",
            "cacheScope": res.get("cacheScope"),
            "ttlMs": res.get("ttlMs"),
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

    # ----- E. a URI naming nothing, and one climbing out ----------
    refusals = {}
    for request_id, bad_uri in ((4, "llm://no-such-topic"), (5, "llm://../../etc/passwd")):
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
        result, "E_unknown_and_escaping_uris_are_refused",
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

    # ----- G. every advertised doc reads back non-empty -----------
    doc_reports = {}
    request_id = 100
    for uri in doc_uris:
        request_id += 1
        env, resp = mcp_call(
            pkg, url, "resources/read", {"uri": uri}, request_id=request_id
        )
        res = env.get("result") or {}
        contents = res.get("contents") or []
        first = contents[0] if contents and isinstance(contents[0], dict) else {}
        text = first.get("text") or ""
        ok, why = mcp_envelope_ok(env, request_id)
        doc_reports[uri] = {
            "ok": bool(
                ok
                and first.get("uri") == uri
                and first.get("mimeType") == "text/markdown"
                and bool(text.strip())
                and res.get("cacheScope") == "public"
            ),
            "status": resp["status"],
            "chars": len(text.strip()),
            "mimeType": first.get("mimeType"),
            "cacheScope": res.get("cacheScope"),
            "why": why or None,
        }
    doc_faults = {
        uri: info for uri, info in doc_reports.items() if not info["ok"]
    }
    mcp_check(
        result, "G_every_advertised_doc_reads_non_empty",
        bool(doc_uris) and not doc_faults,
        {
            "doc_count": len(doc_uris),
            "faults": doc_faults,
            "chars_by_uri": {
                uri: info["chars"] for uri, info in sorted(doc_reports.items())
            },
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
