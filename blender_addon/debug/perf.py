#!/usr/bin/env python3
# File: perf.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Draw-time profiler CLI — drives ``blender_addon/ui/perf.py`` inside
Blender over the debug/MCP transport provided by ``client.py``.

    python blender_addon/debug/perf.py enable           # start collecting timings
    python blender_addon/debug/perf.py sample           # enable + force redraws + report
    python blender_addon/debug/perf.py report           # print current stats
    python blender_addon/debug/perf.py report --json    # raw JSON
    python blender_addon/debug/perf.py reset
    python blender_addon/debug/perf.py disable

After ``enable``, mouse over the Blender sidebar / resize the 3D view to
produce samples, then run ``report``. ``sample`` forces redraws from code
and is the fastest way to get a first readout, but covers only the
currently-visible panels.
"""

import argparse
import os
import sys

# Script-mode imports: see note in main.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import HOST, DEFAULT_MCP_PORT, run_in_blender  # noqa: E402
from output import print_response  # noqa: E402


# Snippet prefix that every perf command runs first. Locates the addon's
# root package by scanning sys.modules and stripping a known leaf, then
# imports ``<pkg>.ui.perf`` as ``_perf``. ``.ui.dynamics.overlay`` is a
# stable leaf module unlikely to conflict with other addons. The
# removesuffix recovers the addon root under both the extension layout
# (bl_ext.user_default.ppf_contact_solver) and any legacy layout.
_PERF_LOADER = (
    "import sys, importlib\n"
    "_top = next((n.removesuffix('.ui.dynamics.overlay') for n in sys.modules\n"
    "             if n.endswith('.ui.dynamics.overlay')), None)\n"
    "if _top is None:\n"
    "    raise RuntimeError('ppf_contact_solver addon does not appear to be loaded')\n"
    "_perf = importlib.import_module(_top + '.ui.perf')\n"
)


def _exec(snippet: str, host: str, port: int) -> dict:
    """Prepend the loader and ship to Blender."""
    return run_in_blender(_PERF_LOADER + snippet, port, host)


def cmd_enable(args):
    snippet = (
        f"print(_perf.enable("
        f"panels={not args.no_panels}, "
        f"uilists={not args.no_uilists}, "
        f"overlay={not args.no_overlay}))\n"
        "_perf.reset()\n"
        "_perf.tag_redraw_all()\n"
        "print('profiler enabled; interact with Blender, then: report')\n"
    )
    print_response(_exec(snippet, args.host, args.mcp_port))


def cmd_disable(args):
    print_response(_exec(
        "_perf.disable(); print('profiler disabled')\n",
        args.host, args.mcp_port,
    ))


def cmd_reset(args):
    print_response(_exec(
        "_perf.reset(); _perf.tag_redraw_all(); print('stats reset')\n",
        args.host, args.mcp_port,
    ))


def cmd_report(args):
    if args.json:
        snippet = (
            f"import json; "
            f"print(json.dumps(_perf.report_json({args.top}), indent=2))\n"
        )
    else:
        snippet = f"print(_perf.report({args.top}))\n"
    print_response(_exec(snippet, args.host, args.mcp_port))


def cmd_sample(args):
    # ``bpy.ops.wm.redraw_timer`` forces synchronous draws on the main thread,
    # so timings actually accumulate inside a single run_python_script call.
    # ``time.sleep`` would just block the main thread and prevent redraws.
    snippet = (
        "import bpy\n"
        "_perf.enable(); _perf.reset(); _perf.tag_redraw_all()\n"
        f"bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations={args.iterations})\n"
        f"print(_perf.report({args.top}))\n"
    )
    print_response(_exec(snippet, args.host, args.mcp_port))


def main():
    parser = argparse.ArgumentParser(
        description="UI draw-time profiler CLI (drives blender_addon/ui/perf.py)",
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT)

    sub = parser.add_subparsers(dest="command")

    p_en = sub.add_parser("enable", help="Install timing wrappers")
    p_en.add_argument("--no-panels", action="store_true")
    p_en.add_argument("--no-uilists", action="store_true")
    p_en.add_argument("--no-overlay", action="store_true")

    sub.add_parser("disable", help="Restore original draw methods")
    sub.add_parser("reset", help="Clear collected stats")

    p_rep = sub.add_parser("report", help="Print draw-time report")
    p_rep.add_argument("--top", type=int, default=30)
    p_rep.add_argument("--json", action="store_true")

    p_samp = sub.add_parser(
        "sample",
        help="Enable → force N redraws → report",
    )
    p_samp.add_argument("--iterations", type=int, default=5)
    p_samp.add_argument("--top", type=int, default=30)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    {
        "enable": cmd_enable,
        "disable": cmd_disable,
        "reset": cmd_reset,
        "report": cmd_report,
        "sample": cmd_sample,
    }[args.command](args)


if __name__ == "__main__":
    main()
