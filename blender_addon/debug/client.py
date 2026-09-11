# File: client.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Transport + control primitives — talk to a running Blender addon over:

  * TCP debug/reload port (default 8765) — reload, exec, start-mcp.
  * MCP Streamable HTTP server (default 9633) — tools/call, resources/read.

Pure stdlib. No Blender dependency — runs on any host Python.
"""

import base64
import importlib.util
import json
import os
import socket
import sys
import urllib.error
import urllib.request


def _load_port_defaults():
    """Resolve the canonical port constants from the addon's
    ``models/defaults.py`` by file path, so this standalone debug client
    stays the single source of truth alongside the addon package.

    The debug CLIs run with only ``debug/`` on ``sys.path`` and the addon
    is installed under a mangled package name, so a normal import of
    ``models.defaults`` is not available here. ``models/defaults.py`` has
    no imports, so loading it by path is safe and pulls in no Blender
    dependency. If the file cannot be found (debug client run detached
    from the addon tree), fall back to the known literals below.
    """
    fallback = {"DEFAULT_MCP_PORT": 9633, "DEFAULT_RELOAD_PORT": 8765}
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "models",
        "defaults.py",
    )
    try:
        spec = importlib.util.spec_from_file_location("_ppf_cts_defaults", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            "DEFAULT_MCP_PORT": module.DEFAULT_MCP_PORT,
            "DEFAULT_RELOAD_PORT": module.DEFAULT_RELOAD_PORT,
        }
    except (OSError, AttributeError, ImportError, SyntaxError):
        return fallback


_PORT_DEFAULTS = _load_port_defaults()

# The debug/reload port. Same value as ``DEFAULT_RELOAD_PORT`` in
# models/defaults.py (the canonical source of truth).
DEBUG_PORT = _PORT_DEFAULTS["DEFAULT_RELOAD_PORT"]
DEFAULT_MCP_PORT = _PORT_DEFAULTS["DEFAULT_MCP_PORT"]
HOST = "localhost"

MCP_PROTOCOL_VERSION = "2026-07-28"
_MCP_CLIENT_INFO = {"name": "ppf-cts-debug", "version": "0.1.0"}

# The params member each method sources its Mcp-Name header from.
_NAME_HEADER_SOURCE = {
    "tools/call": "name",
    "resources/read": "uri",
    "prompts/get": "name",
}


# ---------------------------------------------------------------------------
# MCP Streamable HTTP client
# ---------------------------------------------------------------------------


class MCPClient:
    """MCP Streamable HTTP client.

    The transport is stateless: every request is a standalone POST that
    carries its protocol version, client capabilities and client identity in
    ``params._meta``, mirrored into the ``MCP-Protocol-Version``,
    ``Mcp-Method`` and ``Mcp-Name`` headers. There is no handshake to perform
    and no session to tear down, so the context manager exists only to keep
    call sites uniform::

        with MCPClient(port) as c:
            tools = c.call("tools/list")["result"]["tools"]
    """

    def __init__(self, port=DEFAULT_MCP_PORT, host=HOST, timeout=10.0):
        self._url = f"http://{host}:{port}/mcp"
        self._timeout = timeout
        self._next_id = 0
        self.discover_reply = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def _meta(self):
        return {
            "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": _MCP_CLIENT_INFO,
        }

    def _headers(self, method, params):
        """Standard request headers, mirroring the values in the body.

        The server rejects any disagreement between these and the body, so
        they are derived from the body here rather than passed in.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": method,
        }
        source = _NAME_HEADER_SOURCE.get(method)
        if source:
            value = params.get(source)
            if isinstance(value, str):
                headers["Mcp-Name"] = _encode_header_value(value)
        return headers

    def _post(self, payload, headers, *, timeout=None):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
            if resp.status == 202:
                return None
            raw = resp.read()
            return json.loads(raw) if raw else None

    def _rpc_id(self):
        self._next_id += 1
        return self._next_id

    def call(self, method, params=None, *, timeout=None):
        """Send one JSON-RPC request and return the reply.

        An HTTP error status still carries a JSON-RPC error body, which is
        the useful part, so it is parsed and returned rather than raised.
        """
        params = dict(params or {})
        params["_meta"] = self._meta()
        payload = {
            "jsonrpc": "2.0",
            "id": self._rpc_id(),
            "method": method,
            "params": params,
        }
        headers = self._headers(method, params)
        try:
            return self._post(payload, headers, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            if raw:
                try:
                    return json.loads(raw)
                except ValueError:
                    pass
            raise

    def discover(self):
        """Ask the server for its versions, capabilities and identity."""
        self.discover_reply = self.call("server/discover")
        return self.discover_reply

    def close(self):
        """No-op: the transport holds nothing to release."""


def _encode_header_value(value):
    """Wrap *value* in the Base64 sentinel when it cannot travel as ASCII.

    HTTP field values admit visible ASCII, space and horizontal tab, with no
    leading or trailing whitespace. A resource URI or tool name outside that
    set, or one that happens to look like the sentinel itself, is encoded.
    """
    plain = value == value.strip(" \t") and all(
        ch == "\t" or 0x20 <= ord(ch) <= 0x7E for ch in value
    )
    looks_encoded = value.startswith("=?base64?") and value.endswith("?=")
    if plain and not looks_encoded:
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"=?base64?{encoded}?="


# ---------------------------------------------------------------------------
# Connection checks
# ---------------------------------------------------------------------------


def is_mcp_reachable(port=DEFAULT_MCP_PORT, host=HOST):
    """True if the MCP server answers server/discover within 2s."""
    try:
        with MCPClient(port, host, timeout=2.0) as c:
            reply = c.discover()
            return bool(reply and "result" in reply)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def is_debug_port_open(host=HOST):
    """True if the debug/reload TCP server answers ``ping`` within 2s."""
    try:
        resp = debug_request({"command": "ping"}, host=host, timeout=2.0)
        return resp.get("status") == "ok"
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Debug TCP transport
# ---------------------------------------------------------------------------


def debug_request(packet, host=HOST, timeout=30.0):
    """Send a JSON command over the debug TCP port, read until EOF, and
    return the parsed reply. The server half-closes after the response."""
    data = json.dumps(packet).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect((host, DEBUG_PORT))
        sock.sendall(data)
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    return json.loads(raw) if raw else {}


# ---------------------------------------------------------------------------
# MCP convenience wrappers
# ---------------------------------------------------------------------------


def mcp_discover(port=DEFAULT_MCP_PORT, host=HOST):
    """Ask the server what it speaks. Returns the JSON-RPC reply."""
    with MCPClient(port, host) as c:
        return c.discover()


def mcp_list_tools(port=DEFAULT_MCP_PORT, host=HOST):
    """List all registered MCP tools. Returns a list of tool dicts."""
    with MCPClient(port, host) as c:
        resp = c.call("tools/list")
        return resp.get("result", {}).get("tools", [])


def mcp_call_tool(port, tool_name, arguments=None, host=HOST, timeout=30.0):
    """Call an MCP tool by name with optional arguments."""
    with MCPClient(port, host, timeout=timeout) as c:
        return c.call(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
            timeout=timeout,
        )


def mcp_list_resources(port=DEFAULT_MCP_PORT, host=HOST):
    """List available MCP resources. Returns a list of resource dicts."""
    with MCPClient(port, host) as c:
        resp = c.call("resources/list")
        return resp.get("result", {}).get("resources", [])


def mcp_read_resource(port, uri, host=HOST):
    """Read a single MCP resource by URI."""
    with MCPClient(port, host) as c:
        return c.call("resources/read", {"uri": uri})


# ---------------------------------------------------------------------------
# Debug-port control — reload, exec, start-mcp
# ---------------------------------------------------------------------------


def debug_reload(host=HOST):
    """Trigger addon hot-reload via debug port.

    The server runs the reload synchronously on Blender's main thread
    before responding, so a longer timeout is required.
    """
    return debug_request({"command": "reload"}, host=host, timeout=45.0)


def debug_full_reload(host=HOST):
    """Full addon reload that splits disable and enable across two
    event-loop ticks. Use when PropertyGroup schema changes (adding or
    removing fields on classes referenced via CollectionProperty) don't
    show up after ``reload``."""
    return debug_request({"command": "full_reload"}, host=host, timeout=70.0)


def debug_exec(code, host=HOST, timeout=30.0):
    """Execute Python code inside Blender via the debug port and return the
    result payload."""
    return debug_request(
        {"command": "execute", "code": code, "timeout": timeout},
        host=host,
        timeout=timeout,
    )


def debug_start_mcp(port=DEFAULT_MCP_PORT, host=HOST):
    """Ask the debug server to start the MCP server on `port`."""
    return debug_request(
        {"command": "start_mcp", "port": port}, host=host, timeout=10.0
    )


# ---------------------------------------------------------------------------
# Convenience / composite helpers
# ---------------------------------------------------------------------------


def check_mcp(port=DEFAULT_MCP_PORT, host=HOST):
    """Verify MCP server is reachable, or exit with a hint.

    Returns `port` for chaining in callers that want ``port = check_mcp(...)``.
    """
    if not is_mcp_reachable(port, host):
        print(f"Error: MCP server not reachable on {host}:{port}.", file=sys.stderr)
        print(
            "Hint: start it from Blender UI or run: "
            "python blender_addon/debug/main.py start-mcp",
            file=sys.stderr,
        )
        sys.exit(1)
    return port


def get_scene(port=DEFAULT_MCP_PORT, host=HOST):
    """Fetch the current Blender scene via MCP resource endpoint."""
    check_mcp(port, host)
    resp = mcp_read_resource(port, "blender://scene/current", host=host)
    contents = resp.get("result", {}).get("contents", [])
    if contents:
        return json.loads(contents[0].get("text", "{}"))
    return {}


def run_in_blender(code, port=DEFAULT_MCP_PORT, host=HOST):
    """Execute Python code in Blender and return the result.

    Prefers the MCP ``run_python_script`` tool (richer response shape);
    falls back to the debug TCP port if MCP isn't running.
    """
    if is_mcp_reachable(port, host):
        return mcp_call_tool(port, "run_python_script", {"code": code}, host=host)
    return debug_exec(code, host=host)
