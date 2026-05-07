# File: scenarios/bl_server_unknown_recovery.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression test for the "Status stuck at Initializing / Waiting for
# Server Start..." bug. Reproduces the failure mode and asserts the
# heartbeat probe in transitions.py:PollTick recovers the engine state.
#
# Setup:
#   1. Connect via LOCAL backend (rig server is already up).
#   2. Manually dispatch ServerLost() to simulate the transient query
#      failure that flips state.server to UNKNOWN.
#   3. Assert state.server == UNKNOWN at this point.
#   4. Dispatch PollTick() and tick the runner. The new transition
#      branch should fire DoQuery() because phase==ONLINE and
#      server==UNKNOWN, the live server responds, _interpret_response
#      sets server=RUNNING.
#   5. Assert state.server == RUNNING within a bounded time.
#
# Without the fix, step 5 fails: state.server stays UNKNOWN forever
# because PollTick only re-queried when solver was already in an
# active simulation/build state.

from __future__ import annotations


from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
PLATFORMS = ("linux",)


_DRIVER_TEMPLATE = """
import sys, time, traceback
try:
    facade = __import__(pkg + ".core.facade", fromlist=["engine", "tick", "runner"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    events = __import__(pkg + ".core.events", fromlist=["ServerLost", "PollTick"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    root = groups.get_addon_data(bpy.context.scene)
    root.ssh_state.server_type = "LOCAL"
    root.ssh_state.local_path = <<LOCAL_PATH_REPR>>
    root.ssh_state.docker_port = <<SERVER_PORT>>

    com = client.communicator
    com.set_project_name("server_unknown_recovery")
    com.connect_local(root.ssh_state.local_path,
                      server_port=root.ssh_state.docker_port)

    # Mirrors _driver_lib.DriverHelpers.connect_local: dispatching
    # PollTick in the wait loop is what queues DoQuery on the worker.
    # Without it, the engine has no reason to talk to the server, so
    # state.server stays at the default UNKNOWN even with a live rig.
    deadline = time.time() + 20.0
    while time.time() < deadline:
        facade.engine.dispatch(events.PollTick())
        facade.tick()
        s = facade.engine.state
        if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
            break
        time.sleep(0.2)

    pre = facade.engine.state
    result["phase_pre"] = pre.phase.name
    result["server_pre"] = pre.server.name
    can_continue = (pre.phase.name == "ONLINE" and pre.server.name == "RUNNING")
    if not can_continue:
        result["errors"].append(
            f"connect did not stabilize: phase={pre.phase.name!r}, "
            f"server={pre.server.name!r}"
        )

    if can_continue:
        # Simulate the transient query failure that ServerLost models.
        facade.engine.dispatch(events.ServerLost())
        facade.tick()
        lost = facade.engine.state
        result["server_after_lost"] = lost.server.name
        if lost.server.name != "UNKNOWN":
            result["errors"].append(
                f"ServerLost did not flip server to UNKNOWN; "
                f"got {lost.server.name!r}"
            )
            can_continue = False

    if can_continue:
        # The heartbeat branch in PollTick must re-query and the live
        # server must answer, flipping server back to RUNNING.
        deadline = time.time() + 10.0
        recovered = False
        while time.time() < deadline:
            facade.engine.dispatch(events.PollTick())
            facade.tick()
            s = facade.engine.state
            if s.server.name == "RUNNING":
                recovered = True
                break
            time.sleep(0.2)

        final = facade.engine.state
        result["server_post"] = final.server.name
        result["recovered"] = recovered
        if not recovered:
            result["errors"].append(
                f"heartbeat did not recover; server stayed at "
                f"{final.server.name!r}"
            )
except Exception as exc:
    result["errors"].append(f"driver: {type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
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
                f"stdout (tail): {open(bspec.stdout_path).read()[-1500:]!r}",
                f"stderr (tail): {open(bspec.stderr_path).read()[-1500:]!r}",
            ],
        )

    violations: list[str] = list(result.get("errors") or [])
    if not result.get("scenario_done"):
        violations.append("driver did not run to completion")
    if not result.get("recovered"):
        violations.append(
            f"server did not recover from UNKNOWN: post={result.get('server_post')!r}"
        )

    summary = result.get("probe_summary") or {}
    if summary.get("assertions"):
        violations.extend(
            f"probe: {a.get('kind', '?')}: {a.get('message', '')}"
            for a in summary["assertions"]
        )

    notes = [
        f"server pre={result.get('server_pre')}, "
        f"after_lost={result.get('server_after_lost')}, "
        f"post={result.get('server_post')}, "
        f"recovered={result.get('recovered')}",
    ]
    if violations:
        return r.failed(violations, notes=notes)
    return r.passed(notes=notes)
