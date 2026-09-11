# File: scenarios/_mcp_lib.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Source-string library for Blender-side MCP scenario drivers.
#
# Every MCP scenario has to do the same four things before it can assert
# anything: allocate a free port, start the add-on's in-process MCP server on
# it, issue requests without deadlocking, and parse the reply. This module
# exposes :data:`MCP_LIB`, a Python source fragment a driver template prepends
# the same way it prepends ``DRIVER_LIB``.
#
# TWO CONSTRAINTS SHAPE THE HELPERS, and both are easy to get wrong:
#
#   1. A tools/call is enqueued onto ``mcp.task_system`` and drains only when
#      ``process_mcp_tasks`` runs on Blender's main thread. A driver holds the
#      main thread for the whole of its ``exec()``, so a synchronous request
#      would wait for a pump that cannot run. Every request therefore goes out
#      on a worker thread while the driver pumps from the main thread.
#
#   2. The transport requires ``MCP-Protocol-Version``, ``Mcp-Method`` and, for
#      the three methods that name a target, ``Mcp-Name``, and it rejects any
#      header that disagrees with the body. ``mcp_request`` derives all of them
#      from the body so the two cannot drift apart in a scenario.

from __future__ import annotations

MCP_LIB = r"""
import base64
import json
import socket
import threading
import time
import urllib.error
import urllib.request

MCP_PROTOCOL_VERSION = "2026-07-28"
MCP_LEGACY_VERSION = "2025-06-18"

# The params member each method sources its Mcp-Name header from.
_MCP_NAME_SOURCE = {
    "tools/call": "name",
    "resources/read": "uri",
    "prompts/get": "name",
}


def mcp_alloc_free_port():
    # Bind ephemeral, capture the port, close. The MCP server's start path
    # rebinds with SO_REUSEADDR, so reuse after TIME_WAIT is safe in the small
    # window between close() and start().
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def mcp_start_server(pkg, port):
    # Returns the port the server actually bound, which is not necessarily the
    # one requested: start() falls forward when the port is taken.
    mcp_mod = __import__(
        pkg + ".mcp.mcp_server",
        fromlist=["start_mcp_server", "stop_mcp_server", "get_mcp_server"],
    )
    mcp_mod.start_mcp_server(port)
    actual = mcp_mod.get_mcp_server().port

    server_utils = __import__(
        pkg + ".mcp.server_utils", fromlist=["is_port_available"]
    )
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not server_utils.is_port_available(actual):
            break
        time.sleep(0.1)
    if server_utils.is_port_available(actual):
        raise RuntimeError("MCP server never bound to port %d" % actual)
    return actual, mcp_mod


def mcp_encode_header_value(value):
    # HTTP field values admit visible ASCII, space and horizontal tab, with no
    # leading or trailing whitespace. Anything else travels Base64-wrapped.
    plain = value == value.strip(" \t") and all(
        ch == "\t" or 0x20 <= ord(ch) <= 0x7E for ch in value
    )
    looks_encoded = value.startswith("=?base64?") and value.endswith("?=")
    if plain and not looks_encoded:
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return "=?base64?%s?=" % encoded


def mcp_meta(version=None, client_name="bl_mcp_scenario"):
    return {
        "io.modelcontextprotocol/protocolVersion": version or MCP_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": client_name,
            "version": "0.1",
        },
    }


def mcp_build(method, params=None, request_id=1, version=None, meta=None):
    # Returns (body, headers) with every mirrored header derived from the body.
    params = dict(params or {})
    if meta is not False:
        params["_meta"] = meta if meta else mcp_meta(version)
    body = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        body["id"] = request_id
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": version or MCP_PROTOCOL_VERSION,
        "Mcp-Method": method,
    }
    source = _MCP_NAME_SOURCE.get(method)
    if source and isinstance(params.get(source), str):
        headers["Mcp-Name"] = mcp_encode_header_value(params[source])
    return body, headers


def mcp_post(url, body, headers, timeout=10.0):
    # Never raises: an HTTP error status still carries the JSON-RPC error body,
    # which is the part a scenario asserts on.
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                # The transport answers any tool call that outruns its inline
                # window on a response stream. Reading that with a plain
                # read() would hand the caller SSE framing where it expects a
                # JSON-RPC object, so the frames are decoded here and the
                # final response is returned as the body. Notifications seen
                # along the way are kept under "events" for a scenario that
                # asserts on progress.
                events = []
                final = ""
                buffer = ""
                deadline = time.time() + timeout
                while time.time() < deadline:
                    chunk = resp.read(1)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", "replace")
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        for line in block.splitlines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            events.append(payload)
                            try:
                                parsed = json.loads(payload)
                            except Exception:
                                continue
                            if "result" in parsed or "error" in parsed:
                                final = payload
                    if final:
                        break
                return {
                    "status": resp.status,
                    "headers": dict(resp.headers.items()),
                    "body": final,
                    "events": events,
                    "error": None if final else "stream ended with no response",
                }
            return {
                "status": resp.status,
                "headers": dict(resp.headers.items()),
                "body": resp.read().decode("utf-8"),
                "events": [],
                "error": None,
            }
    except urllib.error.HTTPError as e:
        try:
            body_text = e.read().decode("utf-8")
        except Exception:
            body_text = ""
        return {
            "status": e.code,
            "headers": dict(e.headers.items()),
            "body": body_text,
            "error": str(e),
        }
    except Exception as e:
        return {"status": -1, "headers": {}, "body": "", "error": str(e)}


def mcp_verb(url, method, headers=None, timeout=5.0):
    # A bare GET/DELETE/OPTIONS probe, for the verbs the transport retired.
    req = urllib.request.Request(url, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "headers": dict(resp.headers.items())}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "headers": dict(e.headers.items())}
    except Exception as e:
        return {"status": -1, "headers": {}, "error": str(e)}


def mcp_drive(pkg, url, body, headers, timeout=20.0, pump_interval=0.02):
    # Issue the POST on a worker thread while pumping the task queue from this
    # (main) thread, which is the only thread a handler may touch bpy from.
    process_mcp_tasks = __import__(
        pkg + ".mcp.task_system", fromlist=["process_mcp_tasks"]
    ).process_mcp_tasks
    box = {}

    def worker():
        box["resp"] = mcp_post(url, body, headers, timeout=timeout - 1.0)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    deadline = time.time() + timeout
    pump_errors = []
    while t.is_alive() and time.time() < deadline:
        try:
            process_mcp_tasks()
        except Exception as exc:
            # A pump that raises stops every tool call in the scenario, and
            # each one then fails as a timeout, which names the wrong cause.
            # Record it so the scenario reports what actually happened.
            pump_errors.append("%s: %s" % (type(exc).__name__, exc))
        time.sleep(pump_interval)
    if pump_errors:
        result.setdefault("errors", []).extend(pump_errors[:5])
    t.join(timeout=2.0)
    return box.get("resp") or {
        "status": -1,
        "headers": {},
        "body": "",
        "error": "thread timeout",
    }


def mcp_call(pkg, url, method, params=None, request_id=1, version=None,
             meta=None, timeout=20.0, extra_headers=None):
    # The usual path: build, drive, parse. Returns (envelope, raw response).
    body, headers = mcp_build(method, params, request_id, version, meta)
    if extra_headers:
        for key, value in extra_headers.items():
            if value is None:
                headers.pop(key, None)
            else:
                headers[key] = value
    resp = mcp_drive(pkg, url, body, headers, timeout=timeout)
    return mcp_parse(resp), resp


def mcp_parse(resp):
    if not resp or not resp.get("body"):
        return {}
    try:
        return json.loads(resp["body"])
    except Exception:
        return {}


def mcp_tool(pkg, url, name, arguments=None, request_id=1, timeout=20.0):
    # Call a tool and return its decoded payload, or raise with the reason.
    envelope, resp = mcp_call(
        pkg, url, "tools/call",
        {"name": name, "arguments": arguments or {}},
        request_id=request_id, timeout=timeout,
    )
    if "error" in envelope:
        raise RuntimeError("tool %s failed: %r" % (name, envelope["error"]))
    if "result" not in envelope:
        raise RuntimeError(
            "tool %s: no result (status=%s body=%.200r)"
            % (name, resp.get("status"), resp.get("body"))
        )
    return mcp_tool_payload(envelope), envelope["result"]


def mcp_tool_payload(envelope):
    # A tool result carries the payload twice: rendered in a text block for a
    # model, and parsed in structuredContent for a program. Prefer the parsed
    # one, and fall back so a scenario still works against the text block.
    result = envelope.get("result") or {}
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    content = result.get("content") or []
    if content and isinstance(content[0], dict) and "text" in content[0]:
        try:
            return json.loads(content[0]["text"])
        except Exception:
            return {"_unparsed": content[0]["text"]}
    return {}


def mcp_envelope_ok(envelope, expected_id, require_result_type=True):
    # The documented success shape. Returns (ok, reason).
    if envelope.get("jsonrpc") != "2.0":
        return False, "jsonrpc != 2.0"
    if envelope.get("id") != expected_id:
        return False, "id mismatch: %r != %r" % (envelope.get("id"), expected_id)
    if "error" in envelope:
        return False, "error frame: %r" % (envelope["error"],)
    if "result" not in envelope:
        return False, "missing result"
    if require_result_type:
        kind = envelope["result"].get("resultType")
        if kind != "complete":
            return False, "resultType is %r, expected 'complete'" % (kind,)
    return True, ""


def mcp_check(result, name, ok, details):
    # Record one assertion in the shape report_named_checks expects.
    result.setdefault("checks", {})[name] = {"ok": bool(ok), "details": details}
    return bool(ok)
"""
