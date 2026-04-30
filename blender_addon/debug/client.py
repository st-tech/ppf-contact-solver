# File: client.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Transport + control primitives — talk to a running Blender addon over:

  * TCP debug/reload port (default 8765) — reload, exec, start-mcp.
  * MCP Streamable HTTP server (default 9633) — tools/call, resources/read.

Pure stdlib. No Blender dependency — runs on any host Python.
"""

import json
import socket
import sys
import urllib.error
import urllib.request


DEBUG_PORT = 8765
DEFAULT_MCP_PORT = 9633
HOST = "localhost"

MCP_PROTOCOL_VERSION = "2025-06-18"
_MCP_CLIENT_INFO = {"name": "ppf-cts-debug", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# MCP Streamable HTTP client
# ---------------------------------------------------------------------------

class MCPClient:
    """Short-lived MCP Streamable HTTP client. One session per instance.

    Use as a context manager to get automatic initialize + session teardown::

        with MCPClient(port) as c:
            tools = c.call("tools/list")["result"]["tools"]
    """

    def __init__(self, port=DEFAULT_MCP_PORT, host=HOST, timeout=10.0):
        self._url = f"http://{host}:{port}/mcp"
        self._timeout = timeout
        self._session_id = None
        self._next_id = 0
        self.initialize_reply = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *_):
        self.close()

    def _headers(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
            headers["MCP-Protocol-Version"] = MCP_PROTOCOL_VERSION
        return headers

    def _post(self, payload, *, timeout=None):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers=self._headers(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
            if self._session_id is None:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid
            if resp.status == 202:
                return None
            raw = resp.read()
            return json.loads(raw) if raw else None

    def _rpc_id(self):
        self._next_id += 1
        return self._next_id

    def initialize(self):
        """Perform the initialize handshake and cache the reply."""
        self.initialize_reply = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._rpc_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _MCP_CLIENT_INFO,
                },
            }
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self.initialize_reply

    def call(self, method, params=None, *, timeout=None):
        """Send a JSON-RPC request after initialize and return the reply."""
        if self._session_id is None:
            self.initialize()
        return self._post(
            {
                "jsonrpc": "2.0",
                "id": self._rpc_id(),
                "method": method,
                "params": params or {},
            },
            timeout=timeout,
        )

    def close(self):
        if not self._session_id:
            return
        req = urllib.request.Request(
            self._url,
            method="DELETE",
            headers={"Mcp-Session-Id": self._session_id},
        )
        try:
            urllib.request.urlopen(req, timeout=self._timeout)
        except (urllib.error.URLError, OSError):
            pass
        self._session_id = None


# ---------------------------------------------------------------------------
# Connection checks
# ---------------------------------------------------------------------------

def is_mcp_reachable(port=DEFAULT_MCP_PORT, host=HOST):
    """True if the MCP server accepts an initialize within 2s."""
    try:
        with MCPClient(port, host, timeout=2.0):
            return True
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

def mcp_initialize(port=DEFAULT_MCP_PORT, host=HOST):
    """MCP initialize handshake, returns the JSON-RPC reply."""
    with MCPClient(port, host) as c:
        return c.initialize_reply


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
        host=host, timeout=timeout,
    )


def debug_start_mcp(port=DEFAULT_MCP_PORT, host=HOST):
    """Ask the debug server to start the MCP server on `port`."""
    return debug_request({"command": "start_mcp", "port": port}, host=host, timeout=10.0)


def restart_remote_server(host=HOST, timeout=60.0):
    """Invoke the remote ``$HOME/dev/server/restart.sh`` script via the
    addon's currently-connected backend (SSH / Docker / local / win_native).

    The script does stop + start atomically on the remote, which is more
    reliable than coordinating two MCP calls across the still-responding
    old server and the freshly-launched one.

    Requires the Blender addon to be connected. Returns the script's
    exit code and output.
    """
    # Resolve addon root inside Blender so this works for both legacy
    # and bl_ext.user_default.<id> layouts.
    code = (
        "import sys\n"
        "_pkg = next(n.removesuffix('.core.facade') for n in sys.modules\n"
        "            if n.endswith('.core.facade'))\n"
        "_facade = __import__(_pkg + '.core.facade', fromlist=['runner'])\n"
        "runner = _facade.runner\n"
        "backend = runner._backend\n"
        "if backend is None:\n"
        "    raise RuntimeError('Addon is not connected; cannot reach remote.')\n"
        "host_env = 'HOST=0.0.0.0 ' if getattr(backend, '_container', '') else ''\n"
        "result = backend.exec_command(f'{host_env}bash $HOME/dev/server/restart.sh', shell=True)\n"
        "print('exit_code:', result.get('exit_code'))\n"
        "for line in result.get('stdout', []) or []:\n"
        "    print('[stdout]', line.rstrip())\n"
        "for line in result.get('stderr', []) or []:\n"
        "    print('[stderr]', line.rstrip())\n"
    )
    return debug_exec(code, host=host, timeout=timeout)


# ---------------------------------------------------------------------------
# Convenience / composite helpers
# ---------------------------------------------------------------------------

def check_mcp(port=DEFAULT_MCP_PORT, host=HOST):
    """Verify MCP server is reachable, or exit with a hint.

    Returns `port` for chaining in callers that want ``port = check_mcp(...)``.
    """
    if not is_mcp_reachable(port, host):
        print(f"Error: MCP server not reachable on {host}:{port}.", file=sys.stderr)
        print("Hint: start it from Blender UI or run: "
              "python blender_addon/debug/main.py start-mcp", file=sys.stderr)
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
