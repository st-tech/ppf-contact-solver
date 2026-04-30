# File: scenarios/bl_connect_win_native.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Phase-2: Windows-only counterpart to bl_connect_local. Sets the
# addon's backend to WIN_NATIVE and points it at the worker's debug
# server. Requires ``PPF_WIN_NATIVE_NO_SPAWN=1`` in the environment so
# the addon's connect_win_native skips its embedded-Python detection
# and server.py Popen; the rig has already started the server.

from __future__ import annotations

import os

from . import _runner as r


NEEDS_BLENDER = True
# WIN_NATIVE backend is the Windows path; on Linux/macOS this connect
# call has no production analogue. See bl_connect_local for the Linux
# counterpart.
PLATFORMS = ("win32",)


_DRIVER_TEMPLATE = """
import sys, time, traceback
try:
    facade = __import__(pkg + ".core.facade", fromlist=["engine", "tick"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    root = groups.get_addon_data(bpy.context.scene)
    root.ssh_state.server_type = "WIN_NATIVE"
    # repr() so Windows backslashes survive intact -- e.g. on a GitHub
    # Actions runner the repo lives at D:\\a\\<name>\\<name>, and a raw
    # substitution would let \\a be reparsed as the BEL escape (\\x07).
    root.ssh_state.win_native_path = <<LOCAL_PATH_REPR>>
    # WIN_NATIVE shares ``docker_port`` for the server port, same as
    # LOCAL/DOCKER, because the panel binds all backends to the same row.
    root.ssh_state.docker_port = <<SERVER_PORT>>

    com = client.communicator
    com.connect_win_native(root.ssh_state.win_native_path,
                           root.ssh_state.docker_port)

    # Wait for the worker thread to dispatch Connected, then drain via
    # tick() to apply the queued event into the state.
    deadline = time.time() + 20.0
    while time.time() < deadline:
        facade.tick()
        s = facade.engine.state
        if s.phase.name == "ONLINE":
            break
        time.sleep(0.2)

    s = facade.engine.state
    result["phase"] = s.phase.name
    result["server"] = s.server.name
    result["solver"] = s.solver.name
    result["connected"] = bool(com.is_connected())
except Exception as exc:
    result["errors"].append(f"driver: {type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH_REPR>>", repr(repo_root))
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    import blender_harness as bh

    bspec = ctx.artifacts.get("blender_spec")
    proc = ctx.artifacts.get("blender_proc")
    if bspec is None or proc is None:
        return r.failed(["no Blender process attached to context"])

    try:
        result = bh.wait_for_result(bspec, proc, timeout=max(ctx.timeout, 90.0))
    except TimeoutError as e:
        return r.failed(
            [str(e)],
            notes=[
                f"stdout (tail): {open(bspec.stdout_path).read()[-1500:]!r}",
                f"stderr (tail): {open(bspec.stderr_path).read()[-1500:]!r}",
            ],
        )

    violations: list[str] = list(result.get("errors") or [])
    if not result.get("scenario_done"):
        violations.append("driver did not run to completion")
    if not result.get("connected"):
        violations.append("addon never reached connected=True")
    if result.get("phase") != "ONLINE":
        violations.append(
            f"phase did not reach ONLINE: {result.get('phase')!r}"
        )

    summary = result.get("probe_summary") or {}
    if summary.get("assertions"):
        violations.extend(
            f"probe: {a.get('kind', '?')}: {a.get('message', '')}"
            for a in summary["assertions"]
        )

    notes = [
        f"phase={result.get('phase')}, server={result.get('server')}, "
        f"solver={result.get('solver')}",
        f"probe events={summary.get('event_count', 0)}, "
        f"assertions={summary.get('assertion_count', 0)}",
    ]
    if violations:
        return r.failed(violations, notes=notes)
    return r.passed(notes=notes)
