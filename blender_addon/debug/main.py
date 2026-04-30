#!/usr/bin/env python3
# File: main.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""General CLI — status, tools, call, exec, reload, scene, resources.

MCP tools are discovered dynamically from the running server.
MCP port defaults to 9633 but can be overridden with ``--mcp-port``.
The debug/reload port (TCP 8765) is hardcoded.

    python blender_addon/debug/main.py status
    python blender_addon/debug/main.py tools
    python blender_addon/debug/main.py call <tool> [json]
    python blender_addon/debug/main.py exec <code>
    python blender_addon/debug/main.py reload
    python blender_addon/debug/main.py start-mcp
    python blender_addon/debug/main.py scene
    python blender_addon/debug/main.py resources
    python blender_addon/debug/main.py read <uri>
    python blender_addon/debug/main.py --mcp-port 9635 tools
"""

import argparse
import json
import os
import sys
import textwrap

# Script-mode imports: adding our directory to sys.path lets us import
# sibling modules (``client``, ``output``) without the package having to
# be on PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import (  # noqa: E402
    HOST,
    DEBUG_PORT,
    DEFAULT_MCP_PORT,
    check_mcp,
    debug_full_reload,
    debug_reload,
    debug_start_mcp,
    get_scene,
    is_debug_port_open,
    is_mcp_reachable,
    mcp_call_tool,
    mcp_initialize,
    mcp_list_resources,
    mcp_list_tools,
    mcp_read_resource,
    restart_remote_server,
    run_in_blender,
)
from output import print_json  # noqa: E402


def cmd_status(args):
    port = args.mcp_port
    debug_ok = is_debug_port_open(args.host)
    mcp_ok = is_mcp_reachable(port, args.host)
    print(f"Debug/reload port (TCP {DEBUG_PORT}): "
          f"{'reachable' if debug_ok else 'unreachable'}")
    if mcp_ok:
        print(f"MCP server (HTTP): running on port {port}")
        info = mcp_initialize(port, args.host)
        server_info = info.get("result", {}).get("serverInfo", {})
        if server_info:
            print(f"  Server: {server_info.get('name', '?')} "
                  f"v{server_info.get('version', '?')}")
    else:
        print(f"MCP server (HTTP): not reachable on port {port}")


def cmd_tools(args):
    port = check_mcp(args.mcp_port, args.host)
    tools = mcp_list_tools(port, args.host)
    if args.json:
        print_json(tools)
        return
    if not tools:
        print("No tools registered.")
        return
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "").split("\n")[0][:80]
        print(f"  {name:40s} {desc}")
    print(f"\n{len(tools)} tool(s) available.")


def cmd_call(args):
    port = check_mcp(args.mcp_port, args.host)
    arguments = {}
    if args.arguments:
        arguments = json.loads(args.arguments)
    resp = mcp_call_tool(port, args.tool, arguments, args.host,
                         timeout=args.timeout)
    print_json(resp)


def cmd_exec(args):
    code = args.code
    if code == "-":
        code = sys.stdin.read()
    resp = run_in_blender(code, args.mcp_port, args.host)
    print_json(resp)


def cmd_reload(args):
    resp = debug_reload(args.host)
    status = resp.get("status", "unknown")
    print(f"Reload: {status}")
    if status != "ok" and "error" in resp:
        print(f"Error: {resp['error']}", file=sys.stderr)
        sys.exit(1)


def cmd_full_reload(args):
    resp = debug_full_reload(args.host)
    status = resp.get("status", "unknown")
    print(f"Full reload: {status}")
    if status != "ok" and "error" in resp:
        print(f"Error: {resp['error']}", file=sys.stderr)
        sys.exit(1)


def cmd_start_mcp(args):
    resp = debug_start_mcp(args.port, args.host)
    print(resp.get("message", resp.get("error", "unknown")))


def cmd_restart_server(args):
    """Run server/restart.sh on the addon's currently-connected remote."""
    resp = restart_remote_server(host=args.host, timeout=args.timeout)
    # debug_exec returns {"status": "ok" | "error", "output": "...", "error": "..."}
    status = resp.get("status")
    output = resp.get("output", "")
    if output:
        print(output, end="")
    if status != "ok":
        err = resp.get("error") or resp.get("message") or "unknown"
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    if "exit_code: 0" not in output:
        sys.exit(1)


def cmd_scene(args):
    print_json(get_scene(args.mcp_port, args.host))


def cmd_resources(args):
    port = check_mcp(args.mcp_port, args.host)
    resources = mcp_list_resources(port, args.host)
    if args.json:
        print_json(resources)
        return
    if not resources:
        print("No resources registered.")
        return
    for r in resources:
        uri = r.get("uri", "?")
        name = r.get("name", "")
        desc = (r.get("description", "") or "").split("\n")[0][:80]
        print(f"  {uri:45s} {name}")
        if desc:
            print(f"    {desc}")
    print(f"\n{len(resources)} resource(s) available.")


def cmd_runtests(args):
    """Spawn the Phase-1 orchestrator and forward results.

    Imported lazily so ``main.py`` keeps working even when the scenarios
    package fails to import (e.g. during local development of new
    scenarios with broken syntax)."""
    import orchestrator  # noqa: WPS433 — lazy by design
    import scenarios

    if args.list:
        for name in scenarios.all_names():
            print(name)
        return

    knobs = {}
    for kv in args.knob or []:
        if "=" not in kv:
            print(f"--knob expects KEY=value, got {kv!r}", file=sys.stderr)
            sys.exit(2)
        k, v = kv.split("=", 1)
        knobs[k] = v

    names = args.scenarios or scenarios.all_names()
    kwargs = dict(
        knobs=knobs,
        keep_on_fail=not args.no_keep,
        keep_all=args.keep_all,
        timeout=args.timeout,
        parallel=args.parallel,
        repeat=args.repeat,
        report_path=args.report,
    )
    if args.python is not None:
        kwargs["python"] = args.python
    summary = orchestrator.run_many(names, **kwargs)
    print(json.dumps({
        "run_id": summary["run_id"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "total": summary["total"],
    }, indent=2))
    if summary["failed"]:
        sys.exit(1)


def cmd_read(args):
    port = check_mcp(args.mcp_port, args.host)
    resp = mcp_read_resource(port, args.uri, args.host)
    if args.json:
        print_json(resp)
        return
    if "error" in resp:
        err = resp["error"]
        print(f"Error ({err.get('code', '?')}): {err.get('message', '')}",
              file=sys.stderr)
        sys.exit(1)
    contents = resp.get("result", {}).get("contents", [])
    if not contents:
        print("(empty response)", file=sys.stderr)
        sys.exit(1)
    # Concatenate every text chunk so multi-part resources round-trip cleanly.
    sys.stdout.write("".join(c.get("text", "") for c in contents))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Interact with the Blender addon via MCP / debug ports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s status
              %(prog)s tools
              %(prog)s tools --json
              %(prog)s --mcp-port 9635 tools
              %(prog)s call run_python_script '{"code": "print(bpy.app.version_string)"}'
              %(prog)s exec "print(bpy.app.version_string)"
              %(prog)s exec -                          # read code from stdin
              %(prog)s reload
              %(prog)s start-mcp
              %(prog)s scene
              %(prog)s resources
              %(prog)s read llm://overview
              %(prog)s read llm://index --json
        """),
    )
    parser.add_argument("--host", default=HOST, help="Target host (default: localhost)")
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT,
                        help=f"MCP server port (default: {DEFAULT_MCP_PORT})")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show server status")

    p_tools = sub.add_parser("tools", help="List available MCP tools")
    p_tools.add_argument("--json", action="store_true", help="Output raw JSON")

    p_call = sub.add_parser("call", help="Call an MCP tool")
    p_call.add_argument("tool", help="Tool name")
    p_call.add_argument("arguments", nargs="?", default=None,
                        help="JSON object of arguments")
    p_call.add_argument("--timeout", type=float, default=30.0,
                        help="Request timeout in seconds (default: 30)")

    p_exec = sub.add_parser("exec", help="Run Python code in Blender")
    p_exec.add_argument("code", help="Python code (use '-' to read from stdin)")

    sub.add_parser("reload", help="Hot-reload the addon via debug port")

    sub.add_parser(
        "full-reload",
        help="Full addon reload (two-phase, splits disable/enable across "
        "event-loop ticks). Use this when a PropertyGroup schema change "
        "doesn't appear after `reload`.",
    )

    p_start = sub.add_parser("start-mcp", help="Start MCP server via debug port")
    p_start.add_argument("--port", type=int, default=DEFAULT_MCP_PORT,
                         help=f"MCP port to use (default: {DEFAULT_MCP_PORT})")

    p_restart = sub.add_parser(
        "restart-server",
        help="Run server/restart.sh on the addon's connected remote "
             "(stop + start the solver server atomically).",
    )
    p_restart.add_argument("--timeout", type=float, default=60.0,
                           help="Script timeout in seconds (default: 60)")

    sub.add_parser("scene", help="Get current Blender scene info")

    p_resources = sub.add_parser("resources", help="List MCP resources")
    p_resources.add_argument("--json", action="store_true", help="Output raw JSON")

    p_read = sub.add_parser("read", help="Read an MCP resource by URI")
    p_read.add_argument("uri", help="Resource URI (e.g. llm://overview)")
    p_read.add_argument("--json", action="store_true",
                        help="Output raw JSON envelope instead of body text")

    p_run = sub.add_parser(
        "runtests",
        help="Run Phase-1 debug scenarios against an isolated emulated server.",
    )
    p_run.add_argument(
        "scenarios", nargs="*",
        help="Scenario names (default: all). Use --list to enumerate.",
    )
    p_run.add_argument("--list", action="store_true",
                      help="List registered scenarios and exit.")
    p_run.add_argument(
        "--python",
        default=None,
        help="Python interpreter for spawned servers (default: orchestrator's choice, "
             "typically project .venv).",
    )
    p_run.add_argument("--timeout", type=float, default=60.0,
                      help="Per-scenario timeout (s).")
    p_run.add_argument("--parallel", type=int, default=1,
                      help="Worker pool size. 1 = sequential (default).")
    p_run.add_argument("--repeat", type=int, default=1,
                      help="Run the scenario list this many times.")
    p_run.add_argument("--keep-all", action="store_true",
                      help="Keep every worker dir, even passing ones.")
    p_run.add_argument("--no-keep", action="store_true",
                      help="Delete worker dirs even on failure.")
    p_run.add_argument("--report", default=None,
                      help="Write the aggregated report to this path.")
    p_run.add_argument("--knob", action="append", default=None,
                      help='Extra env knob, "KEY=value". Repeatable.')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    {
        "status": cmd_status,
        "tools": cmd_tools,
        "call": cmd_call,
        "exec": cmd_exec,
        "reload": cmd_reload,
        "full-reload": cmd_full_reload,
        "start-mcp": cmd_start_mcp,
        "restart-server": cmd_restart_server,
        "scene": cmd_scene,
        "resources": cmd_resources,
        "read": cmd_read,
        "runtests": cmd_runtests,
    }[args.command](args)


if __name__ == "__main__":
    main()
