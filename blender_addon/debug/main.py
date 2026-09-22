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
    mcp_discover,
    mcp_list_resources,
    mcp_list_tools,
    mcp_read_resource,
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
        info = mcp_discover(port, args.host)
        result = info.get("result", {}) if info else {}
        server_info = result.get("_meta", {}).get(
            "io.modelcontextprotocol/serverInfo", {}
        )
        if server_info:
            print(f"  Server: {server_info.get('name', '?')} "
                  f"v{server_info.get('version', '?')}")
        versions = result.get("supportedVersions", [])
        if versions:
            print(f"  Protocol versions: {', '.join(versions)}")
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
    """Spawn the orchestrator and forward results.

    Imported lazily so ``main.py`` keeps working even when the scenarios
    package fails to import (e.g. during local development of new
    scenarios with broken syntax)."""
    import orchestrator  # noqa: WPS433 — lazy by design
    import scenarios

    # One gate on the backend NAME, before anything is selected. A removed
    # backend is refused here with the removal spelled out; it must never
    # fall through to a selection that quietly comes back empty.
    try:
        backend = scenarios.resolve_backend(args.backend)
    except scenarios.BackendUnavailable as exc:
        print(f"runtests: {exc}", file=sys.stderr)
        sys.exit(2)

    unrunnable = scenarios.unrunnable_names(backend)

    knobs = {}
    for kv in args.knob or []:
        if "=" not in kv:
            print(f"--knob expects KEY=value, got {kv!r}", file=sys.stderr)
            sys.exit(2)
        k, v = kv.split("=", 1)
        knobs[k] = v

    if args.scenarios:
        # An explicitly named scenario BYPASSES the selection filter, so
        # this is the only place it can be caught. Refuse the whole run
        # rather than dropping the name: a caller that asked for a
        # scenario by name gets an answer about that scenario.
        named_dead = {n: unrunnable[n] for n in args.scenarios
                      if n in unrunnable}
        if named_dead:
            print(f"runtests: {len(named_dead)} named scenario(s) cannot run "
                  f"on backend {backend!r}:", file=sys.stderr)
            for name, reason in named_dead.items():
                print(f"  {name}: {reason}", file=sys.stderr)
            sys.exit(2)
        names = list(args.scenarios)
    else:
        names = scenarios.all_names(backend)
        _report_unrunnable(backend, unrunnable, stream=sys.stdout)
        _report_on_demand(scenarios.on_demand_names(backend),
                          stream=sys.stderr if args.list else sys.stdout)

    if args.shard:
        try:
            names = orchestrator.select_shard(names, args.shard)
        except ValueError as exc:
            print(f"runtests: --shard: {exc}", file=sys.stderr)
            sys.exit(2)

    if args.list:
        # THE LIST IS THE SELECTION THAT WOULD RUN: the named scenarios or
        # the whole set, after the shard. That makes `--list --shard I/N` a
        # dispatch check that needs no build, no Blender and no packages: a
        # rig instance handed only the source and an interpreter prints
        # its share, and the shares can be checked to partition the set.
        for name in names:
            print(name)
        _report_unrunnable(backend, unrunnable, stream=sys.stderr)
        return

    kwargs = dict(
        knobs=knobs,
        keep_on_fail=not args.no_keep,
        keep_all=args.keep_all,
        timeout=args.timeout,
        parallel=args.parallel,
        repeat=args.repeat,
        report_path=args.report,
        backend=backend,
        unrunnable=unrunnable,
    )
    if args.python is not None:
        kwargs["python"] = args.python
    summary = orchestrator.run_many(names, **kwargs)
    print(json.dumps({
        "run_id": summary["run_id"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "total": summary["total"],
        # Carried into the printed summary on purpose. "total" counts what
        # was SELECTED, so a suite that lost its backend would otherwise
        # print a smaller, greener number with nothing to explain it.
        "unrunnable": summary["unrunnable_count"],
    }, indent=2))
    if summary["failed"]:
        sys.exit(1)


def _report_on_demand(names: list, *, stream) -> None:
    """Name the on-demand scenarios the default selection left out."""
    if not names:
        return
    print(f"\n[rig] {len(names)} on-demand scenario(s) left out of the default "
          f"selection; name them to run them:", file=stream)
    for name in names:
        print(f"    {name}", file=stream)
    print("", file=stream)


def _report_unrunnable(backend: str, unrunnable: dict, *, stream) -> None:
    """Name every scenario this backend cannot host, and why.

    Printed on every selection, not stashed in a report file: a scenario
    that vanished with its backend has to be visible in the log the reader
    is already looking at."""
    if not unrunnable:
        return
    print(f"\n[rig] {len(unrunnable)} registered scenario(s) CANNOT RUN on "
          f"backend {backend!r} and were not selected. This is lost "
          f"coverage, not a pass:", file=stream)
    by_reason: dict[str, list[str]] = {}
    for name, reason in unrunnable.items():
        by_reason.setdefault(reason, []).append(name)
    for reason, names in by_reason.items():
        print(f"  reason: {reason}", file=stream)
        for name in sorted(names):
            print(f"    {name}", file=stream)
    print("", file=stream)


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

    sub.add_parser("scene", help="Get current Blender scene info")

    p_resources = sub.add_parser("resources", help="List MCP resources")
    p_resources.add_argument("--json", action="store_true", help="Output raw JSON")

    p_read = sub.add_parser("read", help="Read an MCP resource by URI")
    p_read.add_argument("uri", help="Resource URI (e.g. llm://overview)")
    p_read.add_argument("--json", action="store_true",
                        help="Output raw JSON envelope instead of body text")

    p_run = sub.add_parser(
        "runtests",
        help="Run debug scenarios against an isolated solver server.",
    )
    p_run.add_argument(
        "scenarios", nargs="*",
        help="Scenario names (default: all supported by --backend). "
             "Use --list to enumerate.",
    )
    p_run.add_argument("--list", action="store_true",
                      help="List registered scenarios and exit.")
    p_run.add_argument("--shard", default="",
                      help="I/N: run only every N-th scenario of the selection, "
                           "starting at the I-th (0-based), in the registry's "
                           "order; N hosts given 0/N .. N-1/N run it all once.")
    # REQUIRED, and deliberately not `choices=`. There is no default
    # backend, because a default would silently label every invocation
    # that omitted it as having targeted something it did not. `choices=`
    # is avoided so an unknown name answers through
    # `scenarios.resolve_backend`, which says what this rig can target,
    # rather than with argparse's "invalid choice", which reads as a typo.
    p_run.add_argument(
        "--backend", required=True,
        help="Solver backend the run targets. 'real' is a backend that "
             "computes real physics, which is CUDA, Metal or the Rust CPU "
             "backend depending on what the tree was built for (see the "
             "BACKENDS scenario gate).",
    )
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
        "scene": cmd_scene,
        "resources": cmd_resources,
        "read": cmd_read,
        "runtests": cmd_runtests,
    }[args.command](args)


if __name__ == "__main__":
    main()
