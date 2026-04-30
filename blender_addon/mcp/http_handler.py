"""HTTP handler implementing the MCP Streamable HTTP transport.

Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports

All traffic flows through a single ``/mcp`` endpoint:

- ``POST /mcp`` carries JSON-RPC requests, notifications, or responses. When the
  body contains requests, the server replies with a single JSON response; when
  it only contains notifications or responses, the server replies 202 Accepted.
- ``GET /mcp`` with ``Accept: text/event-stream`` opens a server-to-client SSE
  stream scoped to ``Mcp-Session-Id``. ``Last-Event-ID`` resumes the stream.
- ``DELETE /mcp`` terminates the session.

The server supports MCP protocol version ``2025-06-18``.
"""

import json

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, urlsplit

from .llm_resources import list_llm_resources, read_llm_resource
from .sessions import create_session, delete_session, get_session
from .task_system import get_mcp_result, post_mcp_task
from .tool_schemas import get_tools_list


PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "zozo_contact_solver"
SERVER_VERSION = "0.1.0"
_MAX_POST_BYTES = 10 * 1024 * 1024
_SSE_KEEPALIVE_SECONDS = 15.0

_CAPABILITIES = {
    "tools": {"listChanged": False},
    "resources": {"subscribe": False, "listChanged": False},
    "prompts": {},
}


def _is_local_origin(origin: str) -> bool:
    if not origin:
        return True
    try:
        host = (urlsplit(origin).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "")


class MCPRequestHandler(BaseHTTPRequestHandler):
    """Request handler for the MCP Streamable HTTP transport."""

    def log_message(self, format, *args):
        return

    def _server_info(self):
        return {"name": SERVER_NAME, "version": SERVER_VERSION}

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Accept, Mcp-Session-Id, MCP-Protocol-Version, Last-Event-ID",
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

    def _send_status(self, status, message=""):
        body = message.encode("utf-8") if message else b""
        self.send_response(status)
        if body:
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _reject_non_local_origin(self) -> bool:
        origin = self.headers.get("Origin", "")
        if not _is_local_origin(origin):
            self._send_status(403, f"Origin not allowed: {origin}")
            return True
        return False

    def _reject_non_mcp_path(self) -> bool:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in ("/mcp", "/"):
            self._send_status(404)
            return True
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self._reject_non_local_origin() or self._reject_non_mcp_path():
            return

        accept = self.headers.get("Accept", "")
        if "text/event-stream" not in accept:
            self._send_status(405, "Method Not Allowed")
            return

        session = get_session(self.headers.get("Mcp-Session-Id", ""))
        if session is None:
            self._send_status(404, "Unknown or missing Mcp-Session-Id")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self._cors_headers()
        self.end_headers()

        last_event_id = self.headers.get("Last-Event-ID", "")
        if last_event_id:
            for eid, payload in session.replay_after(last_event_id):
                if not self._emit_sse(eid, payload):
                    return

        while not session.closed:
            try:
                item = session.event_queue.get(timeout=_SSE_KEEPALIVE_SECONDS)
            except Exception:
                item = None
            if item is None:
                try:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                continue
            eid, payload = item
            if not self._emit_sse(eid, payload):
                return

    def _emit_sse(self, event_id: str, payload: str) -> bool:
        frame = f"id: {event_id}\ndata: {payload}\n\n".encode("utf-8")
        try:
            self.wfile.write(frame)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def do_POST(self):
        if self._reject_non_local_origin() or self._reject_non_mcp_path():
            return

        accept = self.headers.get("Accept", "")
        if "application/json" not in accept or "text/event-stream" not in accept:
            self._send_status(
                406,
                "Accept must include both application/json and text/event-stream",
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_status(400, "Invalid Content-Length")
            return
        if content_length < 0 or content_length > _MAX_POST_BYTES:
            self._send_status(
                413, f"Request body too large (limit: {_MAX_POST_BYTES} bytes)"
            )
            return
        raw = self.rfile.read(content_length).decode("utf-8")

        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "Parse error"},
                    "id": None,
                }
            )
            return

        batch = message if isinstance(message, list) else [message]
        batch = [m for m in batch if isinstance(m, dict)]

        has_request = any(
            ("method" in m and m.get("id") is not None) for m in batch
        )
        initialize_msg = next(
            (m for m in batch if m.get("method") == "initialize"), None
        )

        if initialize_msg is None:
            # Spec says clients SHOULD send MCP-Protocol-Version on every request
            # after initialize, but real clients (e.g. rmcp) often omit it. Only
            # reject when the client sends an explicitly wrong version.
            version = self.headers.get("MCP-Protocol-Version")
            if version is not None and version != PROTOCOL_VERSION:
                self._send_status(
                    400,
                    f"MCP-Protocol-Version must be {PROTOCOL_VERSION}",
                )
                return

        session = (
            create_session()
            if initialize_msg is not None
            else get_session(self.headers.get("Mcp-Session-Id", ""))
        )
        if session is None:
            self._send_status(404, "Unknown or missing Mcp-Session-Id")
            return

        responses = [r for r in (self._handle_message(m) for m in batch) if r is not None]

        if not has_request:
            self._send_status(202)
            return

        payload = responses[0] if len(responses) == 1 else responses
        self._send_json(
            payload,
            session_id=session.id if initialize_msg is not None else None,
        )

    def do_DELETE(self):
        if self._reject_non_local_origin() or self._reject_non_mcp_path():
            return
        session_id = self.headers.get("Mcp-Session-Id", "")
        if session_id and delete_session(session_id):
            self._send_status(204)
        else:
            self._send_status(404, "Unknown or missing Mcp-Session-Id")

    def _handle_message(self, msg: dict):
        method = msg.get("method")
        request_id = msg.get("id")
        params = msg.get("params") or {}

        # Pure response frame from client, or a notification: no reply.
        if method is None or request_id is None:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": _CAPABILITIES,
                    "serverInfo": self._server_info(),
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": get_tools_list()},
            }

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            task_id = post_mcp_task(tool_name, arguments)
            result = get_mcp_result(task_id)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(result, indent=2)}
                    ]
                },
            }

        if method == "resources/list":
            resources = [
                {
                    "uri": "blender://scene/current",
                    "name": "Current Blender Scene",
                    "description": "Information about the current Blender scene including objects, frame range, and properties",
                    "mimeType": "application/json",
                }
            ]
            resources.extend(list_llm_resources())
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resources": resources},
            }

        if method == "resources/read":
            uri = params.get("uri")
            if uri == "blender://scene/current":
                task_id = post_mcp_task("get_scene_info", {})
                scene_info = get_mcp_result(task_id)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(scene_info, indent=2),
                            }
                        ]
                    },
                }
            text = read_llm_resource(uri) if uri else None
            if text is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": text,
                            }
                        ]
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"Unknown resource: {uri}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
