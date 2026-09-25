# File: addon_host_tests/_mcp_protocol_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Wire-level conformance gate for the add-on's MCP server.
#
# The transport is pure protocol: envelopes, headers, status codes and result
# shapes. None of that needs Blender, so it is gated here on a plain
# interpreter rather than in the Blender rig, where a protocol regression
# would only surface as a scenario that fails for an unrelated-looking
# reason.
#
# Both protocol eras are covered, because the server answers both from one
# endpoint and the whole point of the dual-era design is that neither client
# can break the other:
#   - 2026-07-28: stateless, per-request `_meta`, mirrored headers.
#   - 2025-06-18: initialize handshake, minted Mcp-Session-Id.

from __future__ import annotations

import base64
import http.client
import json
import socket
import threading
import time

from http.server import ThreadingHTTPServer

import pytest

MODERN = "2026-07-28"
LEGACY = "2025-06-18"

# JSON-RPC codes this gate asserts on, spelled out so a renumbering in the
# implementation cannot quietly satisfy the test.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
HEADER_MISMATCH = -32020
UNSUPPORTED_PROTOCOL_VERSION = -32022


def _meta(version: str = MODERN) -> dict:
    return {
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "gate", "version": "1.0"},
    }


@pytest.fixture(scope="session")
def mcp_module(request):
    """``blender_addon.mcp.http_handler``, loaded from source.

    The handler modules underneath reach real Blender data and fail to import
    against the stub. ``integration.initialize_integrated_system`` already
    tolerates that, so the tool registry comes up empty; the tool-dispatch
    tests register their own handler rather than depending on the add-on's.
    """
    load_addon_module = request.getfixturevalue("load_addon_module_fn")
    return load_addon_module("mcp.http_handler")


@pytest.fixture(scope="session")
def load_addon_module_fn():
    from conftest import load_addon_module  # noqa: PLC0415

    return load_addon_module


@pytest.fixture(scope="session")
def server(mcp_module):
    """A live endpoint, with the main-thread task pump standing in for Blender.

    Inside Blender a timer drains the task queue on the main thread, which is
    what makes a tool call return. Nothing drains it here, so the fixture runs
    the same drain function on a thread; without it every tool call would sit
    out its timeout.
    """
    from blender_addon.mcp.task_system import process_mcp_tasks  # noqa: PLC0415

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mcp_module.MCPRequestHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    pumping = True

    def pump():
        while pumping:
            process_mcp_tasks()
            time.sleep(0.005)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()

    yield httpd.server_address[1]

    pumping = False
    pump_thread.join(timeout=2.0)
    httpd.shutdown()
    httpd.server_close()


class Reply:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = status
        self.headers = headers
        self.raw = body

    @property
    def json(self) -> dict:
        return json.loads(self.raw.decode("utf-8"))

    @property
    def result(self) -> dict:
        payload = self.json
        assert "error" not in payload, f"expected a result, got {payload['error']}"
        return payload["result"]

    @property
    def error(self) -> dict:
        payload = self.json
        assert "error" in payload, f"expected an error, got {payload.get('result')}"
        return payload["error"]


def _request(port: int, method: str, path: str, body, headers: dict) -> Reply:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = None
        if body is not None:
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        return Reply(response.status, dict(response.getheaders()), response.read())
    finally:
        conn.close()


def post_modern(port: int, message: dict, *, extra: dict | None = None) -> Reply:
    """POST *message* with the headers a conformant modern client sends."""
    params = message.setdefault("params", {})
    params.setdefault("_meta", _meta())
    version = params["_meta"].get("io.modelcontextprotocol/protocolVersion", MODERN)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": version,
        "Mcp-Method": message["method"],
    }
    source = {"tools/call": "name", "resources/read": "uri", "prompts/get": "name"}
    key = source.get(message["method"])
    if key and isinstance(params.get(key), str):
        headers["Mcp-Name"] = params[key]
    headers.update(extra or {})
    headers = {k: v for k, v in headers.items() if v is not None}
    return _request(port, "POST", "/mcp", message, headers)


def post_raw(port: int, body, *, headers: dict | None = None) -> Reply:
    base = {"Content-Type": "application/json", "Accept": "application/json"}
    base.update(headers or {})
    return _request(port, "POST", "/mcp", body, base)


# ------------------------------------------------------------ modern era


def test_modern_request_needs_no_session(server):
    """A stateless client gets a result without any handshake."""
    reply = post_modern(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert reply.status == 200
    assert "Mcp-Session-Id" not in reply.headers
    assert reply.result["resultType"] == "complete"


def test_modern_results_identify_the_server(server):
    reply = post_modern(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    info = reply.result["_meta"]["io.modelcontextprotocol/serverInfo"]
    assert info["name"] and info["version"]


@pytest.mark.parametrize(
    ("method", "params", "scope"),
    [
        ("tools/list", {}, "private"),
        ("resources/list", {}, "private"),
        ("prompts/list", {}, "public"),
        ("resources/templates/list", {}, "public"),
        ("server/discover", {}, "private"),
    ],
)
def test_cacheable_results_carry_hints(server, method, params, scope):
    """Every operation on the cacheable list carries ttlMs and cacheScope."""
    reply = post_modern(
        server, {"jsonrpc": "2.0", "id": 1, "method": method, "params": dict(params)}
    )
    result = reply.result
    assert isinstance(result["ttlMs"], int) and result["ttlMs"] >= 0
    assert result["cacheScope"] == scope


@pytest.fixture(scope="session")
def echo_tool(mcp_module):
    """A real registered handler, so tool dispatch is exercised end to end.

    Registering through the decorator rather than writing into the registry
    is deliberate: ``get_handler_registry`` hands out a copy, so a test that
    mutated its return value would silently register nothing.
    """
    from blender_addon.mcp.decorators import mcp_handler  # noqa: PLC0415

    @mcp_handler
    def _gate_echo(value: int = 0):
        """Return the value it was given, for transport gating."""
        return {"echoed": value}

    return "_gate_echo"


def test_tools_call_is_not_cacheable(server, echo_tool):
    """tools/call depends on live scene state, so it carries no cache hints."""
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "_gate_echo", "arguments": {"value": 7}},
        },
    )
    result = reply.result
    assert "ttlMs" not in result and "cacheScope" not in result
    assert result["resultType"] == "complete"
    # The same payload reaches the client parsed as well as rendered.
    assert result["structuredContent"]["echoed"] == 7
    assert not result.get("isError")


def test_discover_lists_every_served_version(server):
    reply = post_modern(
        server, {"jsonrpc": "2.0", "id": 1, "method": "server/discover"}
    )
    result = reply.result
    assert MODERN in result["supportedVersions"]
    assert LEGACY in result["supportedVersions"]
    assert set(result["capabilities"]) >= {"tools", "resources", "prompts"}


@pytest.fixture(scope="session")
def unsorted_tools(mcp_module):
    """Three handlers registered in an order that is not their sorted order.

    Under the stub the add-on's own registry comes up empty, so without these
    the ordering assertion would be comparing a one-element list against
    itself and could not fail for the behavior it names.
    """
    from blender_addon.mcp.decorators import mcp_handler  # noqa: PLC0415

    @mcp_handler
    def _gate_order_zulu():
        """Registered first, sorts last."""
        return {}

    @mcp_handler
    def _gate_order_alpha():
        """Registered second, sorts first."""
        return {}

    @mcp_handler
    def _gate_order_mike():
        """Registered third, sorts in the middle."""
        return {}

    return ["_gate_order_zulu", "_gate_order_alpha", "_gate_order_mike"]


def test_tools_list_order_is_deterministic(server, unsorted_tools):
    """The list comes back sorted, whatever order the handlers registered in.

    A client caches the tool list and feeds it to a model, so a stable order
    is what keeps a prompt cache warm across calls.
    """
    first = post_modern(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    second = post_modern(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [tool["name"] for tool in first.result["tools"]]
    assert names == [tool["name"] for tool in second.result["tools"]]
    assert names == sorted(names)
    # The three above registered in an order the sort has to undo, so the
    # assertion above is doing real work.
    positions = [names.index(n) for n in unsorted_tools]
    assert positions != sorted(positions), (
        "registration order already matched sorted order, so this test proves nothing"
    )


# ------------------------------------------------ per-request `_meta` rules


def test_missing_protocol_version_is_rejected(server):
    """A modern request without its declared version is refused by name."""
    reply = post_raw(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": {}}},
        headers={"MCP-Protocol-Version": MODERN, "Mcp-Method": "tools/list"},
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS


def test_missing_client_capabilities_is_rejected(server):
    meta = _meta()
    del meta["io.modelcontextprotocol/clientCapabilities"]
    reply = post_raw(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": meta},
        },
        headers={"MCP-Protocol-Version": MODERN, "Mcp-Method": "tools/list"},
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS


def test_unsupported_version_names_what_is_supported(server):
    reply = post_raw(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": _meta("1900-01-01")},
        },
        headers={"MCP-Protocol-Version": "1900-01-01", "Mcp-Method": "tools/list"},
    )
    assert reply.status == 400
    error = reply.error
    assert error["code"] == UNSUPPORTED_PROTOCOL_VERSION
    assert error["data"]["requested"] == "1900-01-01"
    assert MODERN in error["data"]["supported"]


# --------------------------------------------------------- header mirroring


def test_method_header_must_match_body(server):
    reply = post_modern(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        extra={"Mcp-Method": "prompts/list"},
    )
    assert reply.status == 400
    assert reply.error["code"] == HEADER_MISMATCH


def test_protocol_version_header_must_match_body(server):
    reply = post_modern(
        server,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        extra={"MCP-Protocol-Version": LEGACY},
    )
    assert reply.status == 400
    assert reply.error["code"] == HEADER_MISMATCH


def test_missing_method_header_is_rejected(server):
    reply = post_raw(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": _meta()},
        },
        headers={"MCP-Protocol-Version": MODERN},
    )
    assert reply.status == 400
    assert reply.error["code"] == HEADER_MISMATCH


def test_name_header_required_for_resources_read(server):
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "blender://scene/current", "_meta": _meta()},
        },
        extra={"Mcp-Name": None},
    )
    assert reply.status == 400
    assert reply.error["code"] == HEADER_MISMATCH


def test_name_header_accepts_the_base64_sentinel(server):
    """A name that cannot travel as ASCII arrives Base64-wrapped.

    The URI names no resource, so the request gets past the header check only
    to be refused as an unknown resource: that refusal, rather than a header
    mismatch, is what shows the wrapped name was accepted.
    """
    uri = "blender://scene/\u00e9"
    encoded = base64.b64encode(uri.encode()).decode()
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": uri, "_meta": _meta()},
        },
        extra={"Mcp-Name": f"=?base64?{encoded}?="},
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS
    assert uri in reply.error["message"]


# ------------------------------------------------------- envelope handling


def test_batch_arrays_are_refused(server):
    reply = post_raw(server, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
    assert reply.status == 400
    assert reply.error["code"] == INVALID_REQUEST


def test_client_response_frames_are_refused(server):
    reply = post_raw(server, {"jsonrpc": "2.0", "id": 1, "result": {}})
    assert reply.status == 400
    assert reply.error["code"] == INVALID_REQUEST


def test_null_id_is_refused(server):
    reply = post_raw(server, {"jsonrpc": "2.0", "id": None, "method": "tools/list"})
    assert reply.status == 400
    assert reply.error["code"] == INVALID_REQUEST


def test_parse_error_omits_the_id(server):
    reply = post_raw(server, b"{not json")
    assert reply.status == 400
    payload = reply.json
    assert payload["error"]["code"] == PARSE_ERROR
    assert "id" not in payload


def test_unknown_method_is_404_with_a_json_rpc_body(server):
    """The status and the body together say the endpoint is live."""
    reply = post_modern(server, {"jsonrpc": "2.0", "id": 1, "method": "no/such/method"})
    assert reply.status == 404
    assert reply.error["code"] == METHOD_NOT_FOUND


def test_notification_is_accepted_with_no_body(server):
    reply = post_modern(server, {"jsonrpc": "2.0", "method": "tools/list"})
    assert reply.status == 202
    assert reply.raw == b""


def test_chunked_body_is_read(server):
    """A streamed body carries no Content-Length and must still be read."""
    message = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": _meta()},
        }
    ).encode()
    request = (
        b"POST /mcp HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        b"Accept: application/json\r\n"
        b"MCP-Protocol-Version: " + MODERN.encode() + b"\r\n"
        b"Mcp-Method: tools/list\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n"
        + f"{len(message):x}".encode()
        + b"\r\n"
        + message
        + b"\r\n0\r\n\r\n"
    )
    with socket.create_connection(("127.0.0.1", server), timeout=10) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks)
    assert b"200 OK" in raw.split(b"\r\n", 1)[0]
    body = raw.split(b"\r\n\r\n", 1)[1]
    assert json.loads(body)["result"]["resultType"] == "complete"


# --------------------------------------------------------- retired verbs


def test_get_without_a_session_is_405(server):
    reply = _request(server, "GET", "/mcp", None, {"Accept": "text/event-stream"})
    assert reply.status == 405
    assert "POST" in reply.headers.get("Allow", "")


def test_delete_without_a_session_is_405(server):
    reply = _request(server, "DELETE", "/mcp", None, {})
    assert reply.status == 405
    assert "POST" in reply.headers.get("Allow", "")


def test_only_one_endpoint_path_is_served(server):
    reply = post_raw(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert reply.status in (200, 400)  # /mcp is served
    other = _request(server, "POST", "/", b"{}", {"Content-Type": "application/json"})
    assert other.status == 404


# --------------------------------------------------------- pagination


@pytest.mark.parametrize("method", ["tools/list", "resources/list", "prompts/list"])
def test_cursor_is_refused_because_none_is_minted(server, method):
    """Serving page one for a cursor the server never issued repeats results."""
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": {"cursor": "whatever", "_meta": _meta()},
        },
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS


@pytest.mark.parametrize("method", ["tools/list", "resources/list", "prompts/list"])
def test_single_page_lists_carry_no_next_cursor(server, method):
    reply = post_modern(server, {"jsonrpc": "2.0", "id": 1, "method": method})
    assert "nextCursor" not in reply.result


# ------------------------------------------------------------- prompts


def test_prompts_capability_is_backed_by_an_implementation(server):
    """A declared capability must answer, never reply -32601 to its own list."""
    listed = post_modern(server, {"jsonrpc": "2.0", "id": 1, "method": "prompts/list"})
    prompts = listed.result["prompts"]
    assert prompts, (
        "the prompts capability is advertised, so the list must not be empty"
    )
    for prompt in prompts:
        assert prompt["name"] and prompt["title"] and prompt["description"]


def test_prompts_get_renders_messages(server):
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {
                "name": "diagnose_failure",
                "arguments": {"symptom": "zero frames written"},
                "_meta": _meta(),
            },
        },
    )
    messages = reply.result["messages"]
    assert messages[0]["role"] == "user"
    assert "zero frames written" in messages[0]["content"]["text"]


def test_prompts_get_refuses_a_missing_required_argument(server):
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {"name": "diagnose_failure", "arguments": {}, "_meta": _meta()},
        },
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS


def test_prompts_get_refuses_an_unknown_name(server):
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {"name": "no_such_prompt", "_meta": _meta()},
        },
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS


# ------------------------------------------------------------ resources


@pytest.mark.parametrize("uri", ["llm://index", "blender://scene/other"])
def test_unknown_resource_uri_is_invalid_params(server, uri):
    # The add-on ships no documentation, so a documentation URI names nothing.
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": uri, "_meta": _meta()},
        },
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS


def test_non_string_resource_uri_does_not_crash_the_connection(server):
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": 42, "_meta": _meta()},
        },
        extra={"Mcp-Name": "42"},
    )
    assert reply.status == 400
    assert reply.error["code"] in (INVALID_PARAMS, HEADER_MISMATCH)


# ---------------------------------------------------------------- tools


def test_unknown_tool_is_a_protocol_error_not_an_iserror_result(server):
    """Failing to FIND a tool is a protocol error; a tool that RAN and failed is not."""
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "no_such_tool", "arguments": {}, "_meta": _meta()},
        },
    )
    assert reply.status == 400
    assert reply.error["code"] == INVALID_PARAMS


def test_every_tool_carries_a_title(server):
    reply = post_modern(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    for tool in reply.result["tools"]:
        assert tool.get("title"), f"{tool['name']} has no title"


# ------------------------------------------------------------ legacy era


def _initialize(port: int) -> tuple[Reply, str]:
    reply = post_raw(
        port,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LEGACY,
                "capabilities": {},
                "clientInfo": {"name": "gate", "version": "1.0"},
            },
        },
    )
    return reply, reply.headers.get("Mcp-Session-Id", "")


def test_legacy_handshake_still_mints_a_session(server):
    reply, session_id = _initialize(server)
    assert reply.status == 200
    assert session_id
    assert reply.result["protocolVersion"] == LEGACY


def test_legacy_results_carry_no_modern_fields(server):
    """A 2025-06-18 client validates against a schema without these fields."""
    _, session_id = _initialize(server)
    reply = post_raw(
        server,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        headers={"Mcp-Session-Id": session_id},
    )
    result = reply.result
    assert "resultType" not in result
    assert "ttlMs" not in result
    assert "cacheScope" not in result


def test_legacy_request_without_a_session_is_refused(server):
    reply = post_raw(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert reply.error["code"] == INVALID_REQUEST


def test_legacy_session_keeps_its_delete(server):
    _, session_id = _initialize(server)
    reply = _request(server, "DELETE", "/mcp", None, {"Mcp-Session-Id": session_id})
    assert reply.status == 204


def test_discover_answers_without_a_session_in_either_era(server):
    """A client calls discover before it knows what the server speaks."""
    reply = post_raw(server, {"jsonrpc": "2.0", "id": 1, "method": "server/discover"})
    assert reply.status == 200
    assert MODERN in reply.result["supportedVersions"]


# ------------------------------------------------- slow tool calls


@pytest.fixture(scope="session")
def slow_tool(mcp_module):
    """A handler slower than the inline window, to exercise the stream path."""
    from blender_addon.mcp.decorators import mcp_handler  # noqa: PLC0415

    @mcp_handler
    def _gate_slow(seconds: float = 2.5):
        """Sleep, so the transport has to answer a call it cannot bound."""
        time.sleep(seconds)
        return {"slept": seconds}

    return "_gate_slow"


def _read_sse_events(port, message, headers, *, limit_seconds=30.0):
    """Collect SSE payloads from a streamed POST until the response arrives."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=limit_seconds)
    conn.request("POST", "/mcp", body=json.dumps(message).encode(), headers=headers)
    response = conn.getresponse()
    content_type = response.getheader("Content-Type", "")
    events = []
    buffer = ""
    deadline = time.monotonic() + limit_seconds
    while time.monotonic() < deadline:
        chunk = response.read(1)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", "replace")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            for line in block.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        if events and "result" in events[-1]:
            break
    conn.close()
    return content_type, events


def test_slow_tool_streams_instead_of_reporting_failure(server, slow_tool):
    """A call slower than the inline window is still a successful call."""
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "_gate_slow",
            "arguments": {"seconds": 2.0},
            "_meta": _meta(),
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MODERN,
        "Mcp-Method": "tools/call",
        "Mcp-Name": "_gate_slow",
    }
    content_type, events = _read_sse_events(server, message, headers)
    assert "text/event-stream" in content_type
    final = events[-1]
    assert "result" in final, f"no response arrived: {events}"
    assert not final["result"].get("isError")
    assert final["result"]["structuredContent"]["slept"] == 2.0


def test_slow_tool_emits_progress_when_a_token_is_supplied(server, slow_tool):
    """notifications/progress flows on the response stream of its own request."""
    meta = _meta()
    meta["progressToken"] = "tok-1"
    message = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "_gate_slow",
            "arguments": {"seconds": 5.0},
            "_meta": meta,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MODERN,
        "Mcp-Method": "tools/call",
        "Mcp-Name": "_gate_slow",
    }
    _, events = _read_sse_events(server, message, headers)
    progress = [e for e in events if e.get("method") == "notifications/progress"]
    assert progress, f"expected progress notifications, got {events}"
    assert progress[0]["params"]["progressToken"] == "tok-1"
    assert events[-1]["result"]["structuredContent"]["slept"] == 5.0


def test_fast_tool_still_answers_with_plain_json(server, echo_tool):
    """The stream is for calls that need it; a quick call stays simple."""
    reply = post_modern(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "_gate_echo", "arguments": {"value": 1}},
        },
    )
    assert "application/json" in reply.headers.get("Content-Type", "")
    assert reply.result["structuredContent"]["echoed"] == 1


def test_slow_tool_without_sse_still_returns_valid_json(server, slow_tool):
    """A client that will not read a stream must still get a parseable response.

    The wait for a slow handler must not write anything to the socket on this
    path: no response headers have been sent yet, so a keep-alive comment
    would arrive ahead of the status line and corrupt the response.
    """
    reply = _request(
        server,
        "POST",
        "/mcp",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "_gate_slow",
                "arguments": {"seconds": 4.0},
                "_meta": _meta(),
            },
        },
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "MCP-Protocol-Version": MODERN,
            "Mcp-Method": "tools/call",
            "Mcp-Name": "_gate_slow",
        },
    )
    assert reply.status == 200, f"status {reply.status}, raw={reply.raw[:200]!r}"
    assert not reply.raw.startswith(b":"), (
        f"response begins with an SSE comment: {reply.raw[:80]!r}"
    )
    assert reply.result["structuredContent"]["slept"] == 4.0


def test_cancelling_a_queued_task_records_nothing(mcp_module):
    """A task that never runs leaves no bookkeeping behind.

    Cancellation is recorded so a task already in flight cannot store a result
    for a reader that has gone. A task still on the queue is simply removed,
    so there is no later result to suppress; remembering it anyway would grow
    the map by one entry for every cancelled call, for the life of the process.
    """
    from blender_addon.mcp import task_system as ts  # noqa: PLC0415

    before = len(ts._cancelled_tasks)
    task_id = ts.post_mcp_task("_gate_echo", {"value": 1})
    ts.cancel_mcp_task(task_id)
    assert len(ts._cancelled_tasks) == before, (
        "a queued-then-cancelled task should leave no cancellation entry"
    )
    # And it must not run when the queue is next drained.
    ts.process_mcp_tasks()
    found, _ = ts.try_get_mcp_result(task_id)
    assert not found, "a cancelled task must not produce a result"


@pytest.fixture(scope="session")
def slow_unserializable_tool(mcp_module):
    """A handler that outruns the inline window and then cannot be encoded."""
    from blender_addon.mcp.decorators import mcp_handler  # noqa: PLC0415

    @mcp_handler
    def _gate_slow_broken(seconds: float = 1.6):
        """Sleep, then return something json cannot encode."""
        time.sleep(seconds)
        return {"obj": object()}

    return "_gate_slow_broken"


def test_failure_after_the_stream_opens_stays_on_the_stream(
    server, slow_unserializable_tool
):
    """A committed response stream must not carry a second HTTP response.

    Once the status line says text/event-stream the socket is SSE framed. A
    failure discovered afterwards has to be written as an event; sending a
    fresh status line would splice a whole HTTP response into the body the
    client is already parsing as events.
    """
    message = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "_gate_slow_broken",
            "arguments": {"seconds": 1.6},
            "_meta": _meta(),
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MODERN,
        "Mcp-Method": "tools/call",
        "Mcp-Name": "_gate_slow_broken",
    }
    content_type, events = _read_sse_events(server, message, headers)
    assert "text/event-stream" in content_type
    assert events, "the failure produced no event at all"
    final = events[-1]
    assert "error" in final, f"expected a JSON-RPC error event, got {final}"
    assert final["id"] == 7


def test_discover_without_meta_is_still_a_modern_result(server):
    """server/discover exists only in 2026-07-28, so its result is always typed.

    A client calls it precisely when it does not yet know what to send, so it
    may arrive with no _meta at all. Answering that with the required fields
    stripped out would produce a shape no revision defines.
    """
    reply = post_raw(server, {"jsonrpc": "2.0", "id": 1, "method": "server/discover"})
    result = reply.result
    assert result["resultType"] == "complete"
    assert isinstance(result["ttlMs"], int)
    assert result["cacheScope"] in ("public", "private")


def test_stateless_request_may_not_declare_a_legacy_version(server):
    """`_meta` does not exist in 2025-06-18, so declaring it there is incoherent.

    Serving it statelessly would hand a client validating against the legacy
    schema the very fields that schema does not define.
    """
    reply = post_raw(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": _meta(LEGACY)},
        },
        headers={"MCP-Protocol-Version": LEGACY, "Mcp-Method": "tools/list"},
    )
    assert reply.status == 400
    error = reply.error
    assert error["code"] == UNSUPPORTED_PROTOCOL_VERSION
    # The retry this invites must be one that can actually succeed.
    assert error["data"]["supported"] == [MODERN]
    assert LEGACY not in error["data"]["supported"]


def test_envelope_error_echoes_the_request_id(server):
    """A malformed envelope is still answered against the id the client sent."""
    reply = post_raw(server, {"jsonrpc": "2.0", "id": 4242, "method": "tools/list",
                              "result": {}})
    assert reply.status == 400
    payload = reply.json
    assert payload["error"]["code"] == INVALID_REQUEST
    assert payload.get("id") == 4242


def test_boolean_request_id_is_refused(server):
    """bool is a subclass of int, so it slips past a plain number check."""
    reply = post_raw(server, {"jsonrpc": "2.0", "id": True, "method": "tools/list"})
    assert reply.status == 400
    assert reply.error["code"] == INVALID_REQUEST


