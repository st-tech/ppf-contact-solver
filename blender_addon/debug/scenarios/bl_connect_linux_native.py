# File: scenarios/bl_connect_linux_native.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch Blender with the addon loaded, point its Linux Native backend at the
# worker's debug server, and assert the addon's state machine reaches ONLINE.
# The scenario provides a driver script that the bootstrap exec()s on the first
# event-loop tick. Result is written to disk; the orchestrator collects it
# after Blender exits.
#
# The Linux sibling of bl_connect_win_native and bl_mac_native_root_resolve:
# one per platform, each asserting that ITS native connection reaches ONLINE
# against a server the rig already owns.

from __future__ import annotations


from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Backend-agnostic connection handshake; runs on the real-GPU Linux job.
BACKENDS = ("real",)
# The Linux Native connection resolves a root on THIS machine, so it is
# exercised where that machine is a Linux one; Windows has
# bl_connect_win_native and macOS bl_mac_native_real_solve.
PLATFORMS = ("linux",)


_DRIVER_TEMPLATE = """
import sys, time, traceback
try:
    facade = __import__(pkg + ".core.facade", fromlist=["engine", "tick"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    conn = __import__(pkg + ".core.connection", fromlist=["native_resolvers"])
    root = groups.get_addon_data(bpy.context.scene)
    root.ssh_state.server_type = "LINUX_NATIVE"
    # repr() so a backslash in the path survives intact.
    root.ssh_state.linux_native_path = <<LOCAL_PATH_REPR>>
    # Every backend reuses ``docker_port`` as the server port; the field is
    # shared because the panel binds all of them to the same port row.
    root.ssh_state.docker_port = <<SERVER_PORT>>

    # WHICH DEVICE THIS TREE HOLDS A BUILD FOR, asked rather than assumed. The
    # property defaults to GPU and a native connect refuses a root holding only
    # the other device's build, by name, so a leg that built the CPU backend
    # would be refused by a scenario that hard-coded GPU.
    resolver = conn.native_resolvers("linux_native")[0]
    device = "GPU" if resolver(root.ssh_state.linux_native_path, "GPU") else "CPU"
    root.ssh_state.native_device = device
    result["device"] = device

    com = client.communicator
    com.connect_linux_native(root.ssh_state.linux_native_path,
                             root.ssh_state.docker_port, device)

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
    repo_root = REPO_ROOT_POSIX
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
                f"stdout (tail): {r.log_tail(bspec.stdout_path)!r}",
                f"stderr (tail): {r.log_tail(bspec.stderr_path)!r}",
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
        f"solver={result.get('solver')}, device={result.get('device')}",
        f"probe events={summary.get('event_count', 0)}, "
        f"assertions={summary.get('assertion_count', 0)}",
    ]
    if violations:
        return r.failed(violations, notes=notes)
    return r.passed(notes=notes)
