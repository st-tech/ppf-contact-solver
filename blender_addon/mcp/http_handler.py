"""HTTP handler implementing the MCP Streamable HTTP transport.

Spec: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports

The server answers two protocol eras on one endpoint, and a request selects
its era by its own shape (``protocol.era_of_request``):

- **2026-07-28, stateless.** Each request is a standalone ``POST /mcp``
  carrying its protocol version, client capabilities and client identity in
  ``params._meta``, mirrored into the ``MCP-Protocol-Version``, ``Mcp-Method``
  and ``Mcp-Name`` headers. There is no handshake and no session. ``GET`` and
  ``DELETE`` are answered ``405``.

- **2025-06-18, session-bound.** The client opens with ``initialize``, which
  mints an ``Mcp-Session-Id`` echoed on every later request. ``GET`` opens a
  keep-alive event stream for that session and ``DELETE`` ends it. The server
  never pushes events, so the stream carries only keep-alive comments.

``server/discover`` is answered in both eras and needs no session, because a
client calls it precisely when it does not yet know what the server speaks.

Every failure leaves through ``ProtocolError``, which carries the JSON-RPC
code out to one place that maps it to an HTTP status. No failure is returned
as a successful result.
"""

import contextlib
import json
import time

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, urlsplit

from . import prompts as prompt_registry
from .integration import get_integrated_handlers
from .protocol import (
    ERA_LEGACY,
    ERA_MODERN,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    LATEST_PROTOCOL_VERSION,
    LEGACY_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SUPPORTED_PROTOCOL_VERSIONS,
    ProtocolError,
    cacheable,
    complete,
    era_of_request,
    http_status_for_error,
    reject_unknown_cursor,
    request_meta,
    rpc_error,
    rpc_result,
    server_info,
    validate_envelope,
    validate_modern_meta,
    validate_request_headers,
)
from .sessions import create_session, delete_session, get_session
from .task_system import (
    cancel_mcp_task,
    get_mcp_result,
    post_mcp_task,
    try_get_mcp_result,
)
from .tool_schemas import get_tools_list

# Kept as module attributes because the add-on's own tooling reads them.
PROTOCOL_VERSION = LATEST_PROTOCOL_VERSION
SERVER_NAME = server_info()["name"]
SERVER_VERSION = server_info()["version"]

_MAX_POST_BYTES = 10 * 1024 * 1024
_SSE_KEEPALIVE_SECONDS = 15.0

# A tool call that finishes inside this window is answered with a single JSON
# object, which is the simplest thing for a client to consume. A slower one is
# answered on an event stream instead, so the client sees it is still running
# rather than being told it failed.
_TOOL_CALL_INLINE_SECONDS = 1.0
_TOOL_CALL_PROGRESS_SECONDS = 2.0

# An absolute ceiling on one tool call. This is a bound, not a heuristic: a
# handler that never returns has to surface as an error the caller can read,
# not as a stream that stays open forever.
_TOOL_CALL_MAX_SECONDS = 900.0

# The single endpoint path. A request to any other path is not this server's.
_ENDPOINT_PATH = "/mcp"

# Caching hints. The tool and prompt lists are fixed for the life of the
# add-on process, so they tolerate a long freshness window; the resource list
# and every resource body are read off disk or out of the live scene, so they
# get a short one. Everything that reflects this Blender session is "private":
# only the documentation bundle is identical for every caller.
_TTL_TOOLS_MS = 600_000
_TTL_PROMPTS_MS = 600_000
_TTL_RESOURCE_LIST_MS = 30_000
_TTL_SCENE_MS = 0
_TTL_TEMPLATES_MS = 3_600_000
_TTL_DISCOVER_MS = 3_600_000

_SCENE_RESOURCE_URI = "blender://scene/current"

_CAPABILITIES = {
    "tools": {"listChanged": False},
    "resources": {"subscribe": False, "listChanged": False},
    "prompts": {"listChanged": False},
}

_INSTRUCTIONS = (
    "Drives ZOZO's Contact Solver add-on inside a running Blender. Each "
    "tool's description and input schema state what it needs and what it "
    "refuses; read them before calling it. Scene construction has a required "
    "order: connect, group, assign objects, constrain, set parameters, "
    "build, solve, fetch."
)


def _accepts_json(accept: str) -> bool:
    """Return True when an Accept header permits a JSON response.

    An absent or wildcard Accept is treated as acceptable. Otherwise the
    media-type tokens are compared exactly (after stripping ``;`` parameters)
    so that e.g. ``application/json-seq`` is not mistaken for ``application/json``.
    """
    if not accept:
        return True
    for token in accept.split(","):
        media_type = token.split(";", 1)[0].strip().lower()
        if media_type in ("application/json", "application/*", "*/*"):
            return True
    return False


def _is_local_origin(origin: str) -> bool:
    if not origin:
        # No Origin header: non-browser clients (rmcp, curl).
        return True
    try:
        host = (urlsplit(origin).hostname or "").lower()
    except ValueError:
        return False
    # Only explicit loopback hostnames count as local. An empty/unparseable
    # host (e.g. Origin: null, file://) is rejected.
    return host in ("localhost", "127.0.0.1", "::1")


class _ClientGoneError(Exception):
    """The client closed the connection, which cancels its request."""


class MCPRequestHandler(BaseHTTPRequestHandler):
    """Request handler for the MCP Streamable HTTP transport."""

    # Set once the response line and headers have gone out as an event stream.
    # After that point the socket carries SSE framing, so a failure has to be
    # written as an event; a second status line would be spliced into the body
    # of the stream the client is already reading.
    _stream_committed = False

    def log_message(self, format, *args):
        return

    def _server_info(self):
        return server_info()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, MCP-Protocol-Version, Mcp-Method, Mcp-Name, "
            "Mcp-Session-Id",
        )
        self.send_header("Access-Control-Expose-Headers", "Mcp-Session-Id")

    def _send_json(self, data, *, status=200, session_id=None):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if session_id:
            self.send_header("Mcp-Session-Id", session_id)
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_status(self, status, message="", *, allow=None):
        body = message.encode("utf-8") if message else b""
        self.send_response(status)
        if body:
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if allow:
            self.send_header("Allow", allow)
        self._cors_headers()
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_rpc_error(self, request_id, exc: ProtocolError):
        """Answer with the JSON-RPC error and the HTTP status it requires.

        Once a response stream is open there is no status left to send, so the
        error goes out as an event on that stream instead.
        """
        payload = rpc_error(request_id, exc.code, exc.message, exc.data)
        if self._stream_committed:
            self._write_event(payload)
            return
        self._send_json(payload, status=http_status_for_error(exc.code))

    def _reject_non_local_origin(self) -> bool:
        origin = self.headers.get("Origin", "")
        if not _is_local_origin(origin):
            self._send_status(403, f"Origin not allowed: {origin}")
            return True
        return False

    def _reject_non_mcp_path(self) -> bool:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path != _ENDPOINT_PATH:
            self._send_status(404)
            return True
        return False

    def _headers_dict(self) -> dict:
        return dict(self.headers.items())

    # ------------------------------------------------------------ verbs

    def do_OPTIONS(self):
        if self._reject_non_local_origin():
            return
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        """Legacy session event stream, or 405.

        The 2026-07-28 transport has no GET endpoint. A GET is served only
        when it names a live legacy session, which a stateless client never
        has.
        """
        if self._reject_non_local_origin() or self._reject_non_mcp_path():
            return

        session = get_session(self.headers.get("Mcp-Session-Id", ""))
        if session is None:
            self._send_status(
                405,
                "Method Not Allowed: this endpoint accepts POST. A GET event "
                "stream exists only for a session opened with the "
                f"{LEGACY_PROTOCOL_VERSION} initialize handshake.",
                allow="POST, OPTIONS",
            )
            return

        accept = self.headers.get("Accept", "")
        if "text/event-stream" not in accept:
            self._send_status(405, "Method Not Allowed", allow="POST, OPTIONS")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self._cors_headers()
        self.end_headers()

        # The server never pushes events, so this stream only emits keep-alive
        # comments. event_queue carries a single None sentinel from close() to
        # unblock get() promptly when the session ends.
        while not session.closed:
            # The queue only ever carries the close() sentinel, so a timeout
            # here is the normal case: it paces the keep-alive comments.
            with contextlib.suppress(Exception):
                session.event_queue.get(timeout=_SSE_KEEPALIVE_SECONDS)
            try:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    def do_DELETE(self):
        """End a legacy session, or 405."""
        if self._reject_non_local_origin() or self._reject_non_mcp_path():
            return
        session_id = self.headers.get("Mcp-Session-Id", "")
        if session_id and delete_session(session_id):
            self._send_status(204)
            return
        self._send_status(
            405,
            "Method Not Allowed: this endpoint accepts POST. Session "
            "termination applies only to a session opened with the "
            f"{LEGACY_PROTOCOL_VERSION} initialize handshake.",
            allow="POST, OPTIONS",
        )

    def _read_body(self) -> bytes:
        """Read the request body, honoring chunked transfer encoding.

        A client that streams the body sends no Content-Length. Reading only
        Content-Length would take such a request as empty and answer a parse
        error for a body that was in fact sent.
        """
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            chunks = bytearray()
            while True:
                line = self.rfile.readline(65_536).strip()
                if not line:
                    break
                try:
                    size = int(line.split(b";", 1)[0], 16)
                except ValueError as exc:
                    raise ProtocolError(PARSE_ERROR, "Malformed chunked body") from exc
                if size == 0:
                    self.rfile.readline(65_536)
                    break
                chunks += self.rfile.read(size)
                self.rfile.readline(65_536)
                if len(chunks) > _MAX_POST_BYTES:
                    raise ProtocolError(
                        INVALID_REQUEST,
                        f"Request body too large (limit: {_MAX_POST_BYTES} bytes)",
                    )
            return bytes(chunks)

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError as exc:
            raise ProtocolError(PARSE_ERROR, "Invalid Content-Length") from exc
        if length < 0 or length > _MAX_POST_BYTES:
            raise ProtocolError(
                INVALID_REQUEST,
                f"Request body too large (limit: {_MAX_POST_BYTES} bytes)",
            )
        return self.rfile.read(length)

    def do_POST(self):
        if self._reject_non_local_origin() or self._reject_non_mcp_path():
            return

        accept = self.headers.get("Accept", "")
        if not _accepts_json(accept):
            self._send_status(406, "Accept must include application/json")
            return

        request_id = None
        try:
            raw = self._read_body()
            try:
                message = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ProtocolError(PARSE_ERROR, "Parse error") from exc

            # Read the id before validating, so a malformed envelope is still
            # answered against the request the client believes it sent.
            if isinstance(message, dict) and isinstance(
                message.get("id"), (str, int, float)
            ) and not isinstance(message.get("id"), bool):
                request_id = message["id"]
            validate_envelope(message)

            if message.get("method") == "tools/call" and request_id is not None:
                era, _ = self._prepare(message)
                self._serve_tool_call(message, era)
                return

            response, session_id = self._serve(message)

            if response is None:
                # A notification is accepted and answered with no body.
                self._send_status(202)
                return
            self._send_json(response, session_id=session_id)

        except _ClientGoneError:
            # Nothing to report and nowhere to report it. The task is already
            # cancelled; the connection is simply gone.
            return
        except ProtocolError as exc:
            self._send_rpc_error(request_id, exc)
        except Exception as exc:  # noqa: BLE001 - the transport's last resort
            # An unexpected condition anywhere in handling is reported as an
            # internal error. Letting it escape would drop the connection and
            # give the client a transport failure to explain instead of a
            # server error.
            self._send_rpc_error(
                request_id, ProtocolError(INTERNAL_ERROR, f"Internal error: {exc}")
            )

    # --------------------------------------------------------- dispatch

    def _prepare(self, message: dict) -> tuple:
        """Validate one message and settle its era. Returns (era, session id).

        The session id is non-empty only when this request mints one, which
        only a legacy ``initialize`` does.
        """
        headers = self._headers_dict()
        era = era_of_request(message, headers)
        method = message["method"]
        meta = request_meta(message)

        if era == ERA_MODERN:
            validate_modern_meta(meta)
            validate_request_headers(headers, message, meta)

        # A notification carries no id and is answered with 202, but it is
        # still validated above so a malformed one is not silently accepted.
        if message.get("id") is None:
            return era, None

        session_id = None
        if era == ERA_LEGACY:
            if method == "initialize":
                session_id = create_session().id
            elif (
                method != "server/discover"
                and get_session(headers.get("Mcp-Session-Id", "")) is None
            ):
                # Every legacy request after the handshake names its session.
                raise ProtocolError(
                    INVALID_REQUEST,
                    "Unknown or missing Mcp-Session-Id. Open a session "
                    f"with a {LEGACY_PROTOCOL_VERSION} initialize "
                    "request, or send a stateless "
                    f"{LATEST_PROTOCOL_VERSION} request carrying "
                    "params._meta.",
                )

        return era, session_id

    def _serve(self, message: dict):
        """Route one message. Returns (response or None, session id or None)."""
        era, session_id = self._prepare(message)
        request_id = message.get("id")
        if request_id is None:
            return None, None
        result = self._call_method(message["method"], message.get("params") or {}, era)
        return rpc_result(request_id, result), session_id

    def _call_method(self, method: str, params: dict, era: str) -> dict:
        if method == "server/discover":
            return self._discover(era)
        if method == "initialize" and era == ERA_LEGACY:
            return self._initialize(params)
        if method == "tools/list":
            return self._tools_list(params, era)
        if method == "tools/call":
            # Every tools/call with an id is served by _serve_tool_call, which
            # owns the socket because it may answer on a stream. Only a
            # notification reaches here, and a notification wants no result.
            raise ProtocolError(
                INVALID_REQUEST,
                "tools/call must carry an id: a tool result has nowhere to go "
                "on a notification",
            )
        if method == "resources/list":
            return self._resources_list(params, era)
        if method == "resources/templates/list":
            return self._resource_templates_list(params, era)
        if method == "resources/read":
            return self._resources_read(params, era)
        if method == "prompts/list":
            return self._prompts_list(params, era)
        if method == "prompts/get":
            return self._prompts_get(params, era)
        raise ProtocolError(METHOD_NOT_FOUND, f"Method not found: {method}")

    # ---------------------------------------------------------- methods

    def _discover(self, era: str) -> dict:
        # `server/discover` exists only in 2026-07-28, so there is no era in
        # which a bare result is the right shape: it is stamped as a modern
        # result even when the request that asked for it carried no `_meta`.
        # A client calls this precisely when it does not yet know what to send.
        era = ERA_MODERN
        return cacheable(
            complete(
                {
                    "supportedVersions": list(SUPPORTED_PROTOCOL_VERSIONS),
                    "capabilities": _CAPABILITIES,
                    "instructions": _INSTRUCTIONS,
                    "_meta": {"io.modelcontextprotocol/serverInfo": server_info()},
                },
                era=era,
            ),
            era=era,
            ttl_ms=_TTL_DISCOVER_MS,
            scope="private",
        )

    def _initialize(self, params: dict) -> dict:
        """The 2025-06-18 handshake result.

        The version echoed back is the one this server serves in that era, not
        the one the client asked for, so a client asking for something else
        learns immediately which era it is in.
        """
        return {
            "protocolVersion": LEGACY_PROTOCOL_VERSION,
            "capabilities": _CAPABILITIES,
            "serverInfo": self._server_info(),
        }

    def _tools_list(self, params: dict, era: str) -> dict:
        reject_unknown_cursor(params)
        # A deterministic order lets a client cache the list and keeps a
        # model's prompt stable across calls.
        tools = sorted(get_tools_list(), key=lambda tool: tool.get("name", ""))
        return cacheable(
            complete({"tools": tools}, era=era),
            era=era,
            ttl_ms=_TTL_TOOLS_MS,
            scope="private",
        )

    def _serve_tool_call(self, message: dict, era: str) -> None:
        """Run one tool call and write its response to the socket.

        A tool call is the only request whose duration this server cannot
        bound: it runs add-on code on Blender's main thread, and building a
        scene or starting a solve legitimately takes longer than any fixed
        deadline. So the wait is not fixed. The call is answered inline when
        it finishes promptly, and on an event stream when it does not, which
        is what lets a slow tool report as running rather than as failed.
        """
        request_id = message["id"]
        params = message.get("params") or {}
        meta = request_meta(message)

        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ProtocolError(
                INVALID_PARAMS, "Invalid params: 'name' must be a non-empty string"
            )
        if tool_name not in get_integrated_handlers():
            # An unknown tool is a malformed request, not a tool that ran and
            # failed, so it is a protocol error rather than an isError result.
            raise ProtocolError(INVALID_PARAMS, f"Unknown tool: {tool_name}")
        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ProtocolError(INVALID_PARAMS, "'arguments' must be an object")

        task_id = post_mcp_task(tool_name, arguments)

        deadline = time.monotonic() + _TOOL_CALL_INLINE_SECONDS
        while time.monotonic() < deadline:
            found, result = try_get_mcp_result(task_id)
            if found:
                self._send_json(
                    rpc_result(request_id, self._tool_result(result, era))
                )
                return
            time.sleep(0.01)

        if not self._client_accepts_sse():
            # The client will not read a stream, so the only honest options are
            # to keep waiting on this connection or to fail. Keep waiting: the
            # call is running, and the bound below still applies.
            result = self._await_task(
                task_id, None, tool_name, streaming=False
            )
            self._send_json(rpc_result(request_id, self._tool_result(result, era)))
            return

        self._open_event_stream()
        progress_token = meta.get("progressToken")
        result = self._await_task(
            task_id, progress_token, tool_name, streaming=True
        )
        self._write_event(rpc_result(request_id, self._tool_result(result, era)))

    def _await_task(self, task_id, progress_token, tool_name, *, streaming):
        """Wait for *task_id*, emitting progress if the client asked for it.

        Returns the handler's result. Closing the connection cancels the call:
        that is the transport's cancellation signal, and a cancelled task must
        not leave a result behind for a reader that has gone.

        ``streaming`` says whether a response stream is already open. When it
        is not, this writes nothing at all: no status line has been sent yet,
        so any byte written here would arrive ahead of it and corrupt the
        response. A disconnect on that path surfaces when the result is sent.
        """
        started = time.monotonic()
        next_progress = started + _TOOL_CALL_PROGRESS_SECONDS
        while True:
            found, result = try_get_mcp_result(task_id)
            if found:
                return result

            elapsed = time.monotonic() - started
            if elapsed > _TOOL_CALL_MAX_SECONDS:
                cancel_mcp_task(task_id)
                return {
                    "status": "error",
                    "message": (
                        f"Tool {tool_name!r} did not finish within "
                        f"{_TOOL_CALL_MAX_SECONDS:.0f} seconds. The call was "
                        "abandoned and its result will be discarded; the "
                        "handler itself may still be running on Blender's "
                        "main thread, which this server cannot interrupt"
                    ),
                }

            now = time.monotonic()
            if not streaming:
                time.sleep(0.02)
                continue
            if progress_token is not None and now >= next_progress:
                next_progress = now + _TOOL_CALL_PROGRESS_SECONDS
                if not self._write_event(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/progress",
                        "params": {
                            "progressToken": progress_token,
                            "progress": round(elapsed, 1),
                            "message": f"{tool_name} still running",
                        },
                    }
                ):
                    cancel_mcp_task(task_id)
                    raise _ClientGoneError()
            elif now >= next_progress:
                next_progress = now + _TOOL_CALL_PROGRESS_SECONDS
                if not self._write_keepalive():
                    cancel_mcp_task(task_id)
                    raise _ClientGoneError()
            time.sleep(0.02)

    def _tool_result(self, result, era: str) -> dict:
        """The CallToolResult envelope for a handler's return value."""
        tool_result = {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
        }
        if isinstance(result, dict):
            # The same payload, parsed. A client that wants fields reads them
            # here instead of parsing the text block, which is written for a
            # model to read. No tool declares an outputSchema, and the schema
            # only constrains structuredContent when one is declared.
            tool_result["structuredContent"] = result
            if result.get("status") == "error":
                tool_result["isError"] = True
        return complete(tool_result, era=era)

    def _client_accepts_sse(self) -> bool:
        accept = self.headers.get("Accept", "")
        return "text/event-stream" in accept

    def _open_event_stream(self) -> None:
        self._stream_committed = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self._cors_headers()
        self.end_headers()

    def _write_event(self, payload: dict) -> bool:
        """Write one SSE event. False means the client is gone."""
        try:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _write_keepalive(self) -> bool:
        try:
            self.wfile.write(b": keep-alive\n\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _resources_list(self, params: dict, era: str) -> dict:
        reject_unknown_cursor(params)
        resources = [
            {
                "uri": _SCENE_RESOURCE_URI,
                "name": "Current Blender Scene",
                "title": "Current Blender Scene",
                "description": (
                    "Objects, frame range and solver-relevant properties of "
                    "the scene open in Blender right now."
                ),
                "mimeType": "application/json",
            }
        ]
        return cacheable(
            complete({"resources": resources}, era=era),
            era=era,
            ttl_ms=_TTL_RESOURCE_LIST_MS,
            scope="private",
        )

    def _resource_templates_list(self, params: dict, era: str) -> dict:
        """This server has no templated resource URIs.

        Every resource it serves is enumerable, so the list is empty. The
        method still answers, because a client that sees the ``resources``
        capability is entitled to ask.
        """
        reject_unknown_cursor(params)
        return cacheable(
            complete({"resourceTemplates": []}, era=era),
            era=era,
            ttl_ms=_TTL_TEMPLATES_MS,
            scope="public",
        )

    def _resources_read(self, params: dict, era: str) -> dict:
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ProtocolError(
                INVALID_PARAMS, "Invalid params: 'uri' must be a non-empty string"
            )

        if uri == _SCENE_RESOURCE_URI:
            task_id = post_mcp_task("get_scene_info", {})
            scene_info = get_mcp_result(task_id)
            if isinstance(scene_info, dict) and scene_info.get("status") == "error":
                # The scene could not be read. Reporting it as content would
                # hand the client an error message dressed as scene data.
                raise ProtocolError(
                    INTERNAL_ERROR,
                    f"Could not read {uri}: {scene_info.get('message', 'unknown error')}",
                )
            return cacheable(
                complete(
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(scene_info, indent=2),
                            }
                        ]
                    },
                    era=era,
                ),
                era=era,
                ttl_ms=_TTL_SCENE_MS,
                scope="private",
            )

        raise ProtocolError(INVALID_PARAMS, f"Unknown resource: {uri}")

    def _prompts_list(self, params: dict, era: str) -> dict:
        reject_unknown_cursor(params)
        return cacheable(
            complete({"prompts": prompt_registry.list_prompts()}, era=era),
            era=era,
            ttl_ms=_TTL_PROMPTS_MS,
            scope="public",
        )

    def _prompts_get(self, params: dict, era: str) -> dict:
        # prompts/get is not on the cacheable list: its result depends on the
        # arguments, which are not part of a cache key.
        return complete(
            prompt_registry.get_prompt(params.get("name"), params.get("arguments")),
            era=era,
        )
