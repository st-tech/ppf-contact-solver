# File: scenarios/bl_launch_error_unsticks.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression test for the "Status stuck at Server Launching..." bug.
#
# When "Start Server" fires, the reducer sets server=LAUNCHING and the
# DoLaunchServer effect runs on the worker. If that launch fails, the
# worker turns the exception into an ErrorOccurred event. The failure
# mode this guards: the ErrorOccurred reducer cleared activity/progress
# but left server untouched, so a launch that errored out (e.g. the
# server bound its port on the remote but the client could not reach it
# because the port was not forwarded) left the panel pinned at
# "Server Launching..." forever, with no way back short of a reconnect.
#
# Checks:
#   A. launch_error_unsticks_launching: from server=LAUNCHING, an
#      ErrorOccurred drops the server to UNKNOWN so the status reads
#      "Waiting for Server Start..." (not "Server Launching...") and the
#      error text is recorded for display.
#   B. running_error_preserves_running: a RUNNING server that hits an
#      unrelated background error (a failed data send, say) is NOT knocked
#      out of RUNNING -- only the transient LAUNCHING state resets.
#
# Pure state-machine scenario: no server, no solver, no transfer, so it
# runs on every platform (macOS included).

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, dataclasses, traceback
result.setdefault("checks", {})
result.setdefault("errors", [])


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    transitions = __import__(pkg + ".core.transitions", fromlist=["transition"])
    state_mod = __import__(pkg + ".core.state",
                           fromlist=["AppState", "Phase", "Server"])
    events = __import__(pkg + ".core.events", fromlist=["ErrorOccurred"])

    transition = transitions.transition
    AppState = state_mod.AppState
    Phase = state_mod.Phase
    Server = state_mod.Server
    ErrorOccurred = events.ErrorOccurred

    # A launch in flight: "Start Server" set server=LAUNCHING on an ONLINE
    # connection and the DoLaunchServer effect is running on the worker.
    launching = dataclasses.replace(
        AppState(), phase=Phase.ONLINE, server=Server.LAUNCHING,
    )
    pre_status = launching.to_remote_status().value

    # The launch fails: the server came up on the remote but the client
    # cannot reach it (port not forwarded), so _do_launch_server raises and
    # the worker converts the exception into ErrorOccurred.
    launch_err = ErrorOccurred(
        error=("Server reached SERVER_READY but the client cannot reach it. "
               "Check that the server port is forwarded to the client."),
        source="_do_launch_server",
    )
    after_err, _effects = transition(launching, launch_err)
    post_status = after_err.to_remote_status().value

    # ----- A: a launch error must unstick LAUNCHING -----
    record(
        "A_launch_error_unsticks_launching",
        (pre_status == "Server Launching..."
         and after_err.server is Server.UNKNOWN
         and post_status != "Server Launching..."
         and bool(after_err.error)),
        {"pre_status": pre_status,
         "post_server": after_err.server.name,
         "post_status": post_status,
         "error_recorded": bool(after_err.error)},
    )

    # ----- B: an unrelated error on a RUNNING server preserves RUNNING -----
    running = dataclasses.replace(
        AppState(), phase=Phase.ONLINE, server=Server.RUNNING,
    )
    running_err = ErrorOccurred(error="data send failed", source="_do_send_data")
    after_running, _ = transition(running, running_err)
    record(
        "B_running_error_preserves_running",
        after_running.server is Server.RUNNING,
        {"post_server": after_running.server.name},
    )

except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
