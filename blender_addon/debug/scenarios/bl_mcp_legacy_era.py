# File: scenarios/bl_mcp_legacy_era.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The 2025-06-18 protocol era, against a real Blender.
#
# One endpoint answers two eras, and a request selects its era by its own
# shape. A 2025-06-18 client sends no ``params._meta``: it opens with an
# ``initialize`` request, is handed an ``Mcp-Session-Id``, and names that
# session on every later request, on a GET event stream and on the DELETE
# that ends it. That whole path is invisible to the modern-era scenarios,
# which never mint a session and are answered 405 on both verbs.
#
# The host-side gate in ``addon_host_tests/_mcp_protocol_.py`` drives the
# handshake, the session-bound follow-up and the DELETE with ``bpy`` stubbed.
# This scenario covers what only a real Blender can show: the same rules with
# every handler registered and a populated tool registry, plus the GET event
# stream, which the gate does not open at all.
#
# The event stream is read on a raw socket rather than through urllib. The
# server pushes nothing and paces its keep-alive comments a full interval
# apart, so the first body byte is one keep-alive interval away while the
# response head arrives at once. Reading the head with a short socket timeout
# and closing establishes that the stream opened, and cannot stall the driver.
#
# Assertions:
#   A. ``initialize_mints_a_session`` -- the handshake answers 200, echoes its
#      id, reports protocolVersion 2025-06-18 with the three capabilities and
#      the server identity, and mints an Mcp-Session-Id header.
#   B. ``session_header_admits_a_follow_up`` -- tools/list carrying only that
#      header, with no _meta and no version header, is served the real tool
#      registry.
#   C. ``legacy_results_carry_no_modern_fields`` -- no legacy result carries
#      resultType, ttlMs or cacheScope, which the 2025-06-18 schema has no
#      place for, including on the two methods that are cacheable in the
#      modern era.
#   D. ``get_opens_a_legacy_event_stream`` -- a GET naming the session with
#      Accept: text/event-stream answers 200 with a text/event-stream body.
#   E. ``delete_ends_the_session`` -- DELETE naming the session answers 204,
#      after which the same session id is no longer accepted.
#   F. ``request_without_session_or_meta_is_refused`` -- a request that is
#      neither session-bound nor self-describing is refused by name, and the
#      message says which of the two to send.
#   G. ``discover_answers_without_a_session`` -- server/discover is served
#      with no session, names both eras, and mints nothing. Unlike every other
#      method here its result IS a modern result, carrying resultType and the
#      caching hints: the method exists only in 2026-07-28, so there is no
#      legacy shape for it to take.

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


def legacy_header(resp, name):
    # RFC 9110 field names are case-insensitive, so a response header is
    # never matched by its exact spelling.
    lowered = name.lower()
    for key, value in (resp.get("headers") or {}).items():
        if key.lower() == lowered:
            return value
    return ""


def legacy_post(url, body, session_id=None):
    # The 2025-06-18 request shape: no params._meta, no MCP-Protocol-Version
    # header, and the session id once the handshake has minted one.
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    resp = mcp_post(url, body, headers)
    return mcp_parse(resp), resp


def legacy_modern_fields(payload):
    # The three members a 2025-06-18 client's schema does not admit.
    return sorted(
        key for key in ("resultType", "ttlMs", "cacheScope") if key in (payload or {})
    )


def legacy_stream_head(host, port, session_id, timeout=5.0):
    # Open the session event stream and read only the response head. The
    # stream stays open by design, so it is read on a raw socket with a
    # bounded timeout and closed immediately.
    request = (
        "GET /mcp HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Accept: text/event-stream\r\n"
        "Mcp-Session-Id: %s\r\n"
        "Connection: close\r\n"
        "\r\n"
    ) % (host, port, session_id)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    head = b""
    error = None
    try:
        sock.connect((host, port))
        sock.sendall(request.encode("ascii"))
        deadline = time.time() + timeout
        while b"\r\n\r\n" not in head and time.time() < deadline:
            try:
                chunk = sock.recv(4096)
            except Exception as exc:
                error = "recv: %s" % (exc,)
                break
            if not chunk:
                break
            head += chunk
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        sock.close()

    text = head.decode("latin-1").split("\r\n\r\n", 1)[0]
    lines = text.split("\r\n")
    status = -1
    if lines and lines[0].startswith("HTTP/"):
        parts = lines[0].split(" ")
        if len(parts) > 1 and parts[1].isdigit():
            status = int(parts[1])
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return {"status": status, "headers": headers, "error": error}


try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. the handshake mints a session -----------------------
    init_env, init_resp = legacy_post(
        url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_LEGACY_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "bl_mcp_legacy_era", "version": "0.1"},
            },
        },
    )
    session_id = legacy_header(init_resp, "Mcp-Session-Id")
    init_res = init_env.get("result") or {}
    init_ok, init_why = mcp_envelope_ok(init_env, 1, require_result_type=False)
    init_info = init_res.get("serverInfo") or {}
    mcp_check(
        result, "A_initialize_mints_a_session",
        init_resp["status"] == 200
        and init_ok
        and init_res.get("protocolVersion") == MCP_LEGACY_VERSION
        and {"tools", "resources", "prompts"} <= set(init_res.get("capabilities") or {})
        and bool(init_info.get("name"))
        and bool(init_info.get("version"))
        and bool(session_id),
        {
            "status": init_resp["status"],
            "envelope_ok": init_ok,
            "why": init_why,
            "protocolVersion": init_res.get("protocolVersion"),
            "capabilities": sorted(init_res.get("capabilities") or {}),
            "serverInfo": init_info,
            "session_id_minted": bool(session_id),
        },
    )

    # ----- B. the session id is what admits the next request ------
    tools_env, tools_resp = legacy_post(
        url,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        session_id=session_id,
    )
    tools_res = tools_env.get("result") or {}
    tools_ok, tools_why = mcp_envelope_ok(tools_env, 2, require_result_type=False)
    tool_names = [
        tool.get("name")
        for tool in (tools_res.get("tools") or [])
        if isinstance(tool, dict)
    ]
    mcp_check(
        result, "B_session_header_admits_a_follow_up",
        tools_resp["status"] == 200
        and tools_ok
        and len(tool_names) > 0
        and all(isinstance(name, str) and name for name in tool_names),
        {
            "status": tools_resp["status"],
            "envelope_ok": tools_ok,
            "why": tools_why,
            "tool_count": len(tool_names),
            "first_tools": sorted(name for name in tool_names if name)[:5],
        },
    )

    # ----- C. nothing modern rides along in this era --------------
    prompts_env, prompts_resp = legacy_post(
        url,
        {"jsonrpc": "2.0", "id": 3, "method": "prompts/list"},
        session_id=session_id,
    )
    prompts_res = prompts_env.get("result") or {}
    stray = {
        "initialize": legacy_modern_fields(init_res),
        "tools/list": legacy_modern_fields(tools_res),
        "prompts/list": legacy_modern_fields(prompts_res),
    }
    mcp_check(
        result, "C_legacy_results_carry_no_modern_fields",
        prompts_resp["status"] == 200
        and len(prompts_res.get("prompts") or []) > 0
        and not any(stray.values()),
        {
            "prompts_status": prompts_resp["status"],
            "prompt_count": len(prompts_res.get("prompts") or []),
            "stray_modern_fields": stray,
        },
    )

    # ----- D. the session event stream ----------------------------
    stream = legacy_stream_head("127.0.0.1", actual_port, session_id)
    content_type = stream["headers"].get("content-type", "")
    mcp_check(
        result, "D_get_opens_a_legacy_event_stream",
        stream["status"] == 200
        and content_type.split(";", 1)[0].strip() == "text/event-stream"
        and "no-cache" in stream["headers"].get("cache-control", ""),
        {
            "status": stream["status"],
            "content_type": content_type,
            "cache_control": stream["headers"].get("cache-control"),
            "read_error": stream["error"],
        },
    )

    # ----- E. DELETE ends it, and the id stops working ------------
    deleted = mcp_verb(url, "DELETE", {"Mcp-Session-Id": session_id})
    after_env, after_resp = legacy_post(
        url,
        {"jsonrpc": "2.0", "id": 4, "method": "tools/list"},
        session_id=session_id,
    )
    after_error = after_env.get("error") or {}
    mcp_check(
        result, "E_delete_ends_the_session",
        deleted["status"] == 204
        and after_resp["status"] == 400
        and after_error.get("code") == -32600,
        {
            "delete_status": deleted["status"],
            "reuse_status": after_resp["status"],
            "reuse_error": after_error,
        },
    )

    # ----- F. neither session-bound nor self-describing -----------
    bare_env, bare_resp = legacy_post(
        url, {"jsonrpc": "2.0", "id": 5, "method": "tools/list"}
    )
    bare_error = bare_env.get("error") or {}
    bare_message = bare_error.get("message") or ""
    mcp_check(
        result, "F_request_without_session_or_meta_is_refused",
        bare_resp["status"] == 400
        and bare_error.get("code") == -32600
        and "Mcp-Session-Id" in bare_message
        and MCP_LEGACY_VERSION in bare_message
        and MCP_PROTOCOL_VERSION in bare_message,
        {
            "status": bare_resp["status"],
            "code": bare_error.get("code"),
            "message": bare_message,
        },
    )

    # ----- G. discover, before the client knows the era -----------
    disc_env, disc_resp = legacy_post(
        url, {"jsonrpc": "2.0", "id": 6, "method": "server/discover"}
    )
    disc_res = disc_env.get("result") or {}
    versions = disc_res.get("supportedVersions") or []
    caps = disc_res.get("capabilities") or {}
    # server/discover is defined only by 2026-07-28, so unlike every other
    # method in this scenario its result is a modern result whatever era the
    # request was written in: there is no legacy shape for it to take. It must
    # still answer without a session, because a client calls it precisely when
    # it does not yet know what to send.
    mcp_check(
        result, "G_discover_answers_without_a_session",
        disc_resp["status"] == 200
        and MCP_LEGACY_VERSION in versions
        and MCP_PROTOCOL_VERSION in versions
        and {"tools", "resources", "prompts"} <= set(caps)
        and disc_res.get("resultType") == "complete"
        and isinstance(disc_res.get("ttlMs"), int)
        and disc_res.get("cacheScope") in ("public", "private")
        and not legacy_header(disc_resp, "Mcp-Session-Id"),
        {
            "status": disc_resp["status"],
            "versions": versions,
            "capabilities": sorted(caps),
            "resultType": disc_res.get("resultType"),
            "ttlMs": disc_res.get("ttlMs"),
            "cacheScope": disc_res.get("cacheScope"),
            "session_id_minted": bool(legacy_header(disc_resp, "Mcp-Session-Id")),
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
