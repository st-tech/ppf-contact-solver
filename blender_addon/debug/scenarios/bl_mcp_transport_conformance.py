# File: scenarios/bl_mcp_transport_conformance.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP Streamable HTTP transport conformance, against a real Blender.
#
# The host-side gate in ``addon_host_tests/_mcp_protocol_.py`` drives the same
# transport with ``bpy`` stubbed, so it covers envelopes and headers but sees
# an empty tool registry and no scene. This scenario covers what only a real
# Blender can show: that the rules still hold with all handlers registered and
# the task queue draining through Blender's main thread.
#
# Assertions:
#   A. ``discover_reports_both_eras`` -- server/discover needs no session and
#      names both served protocol versions plus the three capabilities.
#   B. ``modern_results_are_typed_and_cacheable`` -- tools/list carries
#      resultType, ttlMs, cacheScope and the server identity under _meta.
#   C. ``header_body_disagreement_is_refused`` -- an Mcp-Method that does not
#      match the body is 400 with -32020, and a missing one likewise.
#   D. ``unsupported_version_names_supported`` -- 400 with -32022 whose
#      data.supported lists what to retry with.
#   E. ``retired_verbs_are_405`` -- GET and DELETE without a session.
#   F. ``batching_and_response_frames_refused`` -- an array body and a client
#      response frame are both -32600.
#   G. ``unknown_method_is_404`` -- HTTP 404 carrying a -32601 body, which is
#      what distinguishes a live endpoint from an absent one.
#   H. ``cursor_is_refused`` -- this server mints no cursor, so it rejects one
#      rather than silently re-serving page one.

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

try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. discover, with no session and no handshake ----------
    env, resp = mcp_call(pkg, url, "server/discover", request_id=1)
    versions = (env.get("result") or {}).get("supportedVersions") or []
    caps = (env.get("result") or {}).get("capabilities") or {}
    minted = resp["headers"].get("Mcp-Session-Id", "")
    mcp_check(
        result, "A_discover_reports_both_eras",
        resp["status"] == 200
        and MCP_PROTOCOL_VERSION in versions
        and MCP_LEGACY_VERSION in versions
        and {"tools", "resources", "prompts"} <= set(caps)
        and not minted,
        {
            "status": resp["status"],
            "versions": versions,
            "capabilities": sorted(caps),
            "session_id_minted": minted or None,
        },
    )

    # ----- B. a modern result is typed, identified and cacheable --
    env, resp = mcp_call(pkg, url, "tools/list", request_id=2)
    res = env.get("result") or {}
    info = (res.get("_meta") or {}).get("io.modelcontextprotocol/serverInfo") or {}
    ok, why = mcp_envelope_ok(env, 2)
    mcp_check(
        result, "B_modern_results_are_typed_and_cacheable",
        ok
        and isinstance(res.get("ttlMs"), int)
        and res.get("ttlMs") >= 0
        and res.get("cacheScope") in ("public", "private")
        and bool(info.get("name"))
        and len(res.get("tools") or []) > 0,
        {
            "envelope_ok": ok,
            "why": why,
            "ttlMs": res.get("ttlMs"),
            "cacheScope": res.get("cacheScope"),
            "server_name": info.get("name"),
            "tool_count": len(res.get("tools") or []),
        },
    )

    # ----- C. a header that disagrees with the body is refused ----
    env_mismatch, resp_mismatch = mcp_call(
        pkg, url, "tools/list", request_id=3,
        extra_headers={"Mcp-Method": "prompts/list"},
    )
    env_missing, resp_missing = mcp_call(
        pkg, url, "tools/list", request_id=4,
        extra_headers={"Mcp-Method": None},
    )
    mcp_check(
        result, "C_header_body_disagreement_is_refused",
        resp_mismatch["status"] == 400
        and (env_mismatch.get("error") or {}).get("code") == -32020
        and resp_missing["status"] == 400
        and (env_missing.get("error") or {}).get("code") == -32020,
        {
            "mismatch": [resp_mismatch["status"], env_mismatch.get("error")],
            "missing": [resp_missing["status"], env_missing.get("error")],
        },
    )

    # ----- D. an unserved version is refused by name --------------
    env, resp = mcp_call(pkg, url, "tools/list", request_id=5, version="1900-01-01")
    err = env.get("error") or {}
    data = err.get("data") or {}
    mcp_check(
        result, "D_unsupported_version_names_supported",
        resp["status"] == 400
        and err.get("code") == -32022
        and data.get("requested") == "1900-01-01"
        and MCP_PROTOCOL_VERSION in (data.get("supported") or []),
        {"status": resp["status"], "error": err},
    )

    # ----- E. the verbs this revision retired ---------------------
    get_probe = mcp_verb(url, "GET", {"Accept": "text/event-stream"})
    del_probe = mcp_verb(url, "DELETE")
    mcp_check(
        result, "E_retired_verbs_are_405",
        get_probe["status"] == 405 and del_probe["status"] == 405,
        {"GET": get_probe["status"], "DELETE": del_probe["status"]},
    )

    # ----- F. batching and client response frames -----------------
    batch = mcp_post(
        url,
        [{"jsonrpc": "2.0", "id": 6, "method": "tools/list"}],
        {"Content-Type": "application/json", "Accept": "application/json"},
    )
    frame = mcp_post(
        url,
        {"jsonrpc": "2.0", "id": 7, "result": {}},
        {"Content-Type": "application/json", "Accept": "application/json"},
    )
    batch_env = mcp_parse(batch)
    frame_env = mcp_parse(frame)
    mcp_check(
        result, "F_batching_and_response_frames_refused",
        batch["status"] == 400
        and (batch_env.get("error") or {}).get("code") == -32600
        and frame["status"] == 400
        and (frame_env.get("error") or {}).get("code") == -32600,
        {
            "batch": [batch["status"], batch_env.get("error")],
            "response_frame": [frame["status"], frame_env.get("error")],
        },
    )

    # ----- G. an unimplemented method -----------------------------
    env, resp = mcp_call(pkg, url, "no/such/method", request_id=8)
    mcp_check(
        result, "G_unknown_method_is_404",
        resp["status"] == 404 and (env.get("error") or {}).get("code") == -32601,
        {"status": resp["status"], "error": env.get("error")},
    )

    # ----- H. a cursor this server never issued -------------------
    cursor_codes = {}
    for method in ("tools/list", "resources/list", "prompts/list"):
        env, resp = mcp_call(
            pkg, url, method, {"cursor": "not-from-here"}, request_id=9
        )
        cursor_codes[method] = [resp["status"], (env.get("error") or {}).get("code")]
    mcp_check(
        result, "H_cursor_is_refused",
        all(v == [400, -32602] for v in cursor_codes.values()),
        cursor_codes,
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
