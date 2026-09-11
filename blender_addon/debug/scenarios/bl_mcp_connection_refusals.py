# File: scenarios/bl_mcp_connection_refusals.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Every tool that needs a solver host, driven with no connection.
#
# This is real coverage rather than a stand-in for it. An agent driving the
# add-on has to be able to tell "there is no connection" from "there was
# nothing to do", and the difference is only visible in the refusal. A tool
# that answered these with a plausible success would send an agent on to the
# next step of a workflow that cannot run.
#
# The rig has no solver host for the refusals themselves. The connect entry
# points are still driven, in the LAST two checks, because they move the
# session out of the disconnected state that every check before them reads as
# its precondition.
#
# Each connect is issued from a session OBSERVED offline, which is the only
# phase the state machine accepts a connect request from. Driven back to back
# instead, the family reports a suite of initiations where only the first one
# reaches the state machine: a LOCAL connect needs no handshake, so it can be
# online before the next request arrives, and a request from any other phase
# is refused (check K). The phase is settled here by the one connection the
# rig can complete, a LOCAL connect to the worker's own solver server, so the
# online case is exercised against a host that answers rather than left to
# whichever phase the session happens to be passing through.
#
# Assertions:
#   A. ``connection_info_reports_disconnected`` -- get_connection_info answers
#      without raising and says the session is not connected.
#   B. ``remote_status_tools_say_connect_first`` -- the four status and
#      lifecycle tools guarded by the connection decorator refuse with the
#      same wording, which names connecting as the fix.
#   C. ``pipeline_tools_name_the_disconnected_state`` -- transfer, run,
#      resume, fetch and update_params each refuse and carry the solver
#      status in the message, so the reason is legible without another call.
#   D. ``server_side_tools_refuse`` -- compile_project, git_pull_remote,
#      execute_server_command and the two data-path probes all refuse.
#   E. ``disconnect_without_a_connection_is_refused`` -- disconnect says so
#      rather than reporting a successful teardown of nothing.
#   F. ``export_names_the_preflight_reason`` -- export_usd and export_alembic
#      refuse with the export preflight's own reason, which is the same one
#      get_fetch_status reports, so the two agree.
#   G. ``local_only_tools_still_work`` -- git_pull_local, abort_operation and
#      refresh_ui do not need the connection and succeed, which is what makes
#      the refusals above meaningful rather than a blanket failure.
#   H. ``session_tools_report_the_disconnected_state`` -- update_remote_status
#      refuses, resume_simulation_from names the status it saw, the two
#      teardown tools report an initiation even with nothing running (they
#      signal, they do not wait), and show_console answers either way: it
#      opens a window under the rig's Xvfb UI, and names the UI context it
#      needs in a --background Blender.
#   I. ``one_install_at_a_time`` -- an install is initiated and a second one
#      is refused while the first is in flight, which is the only thing that
#      stops two package installs racing each other.
#   J. ``connect_family_initiates_from_offline`` -- each connect entry point,
#      driven from a session observed offline, reports that it started a
#      connection with the settings it was handed. These run LAST because they
#      move the session out of the disconnected state every check above
#      depends on.
#   K. ``connect_refuses_while_a_connection_is_up`` -- with a connection
#      established, both the specific and the generic connect entry point
#      refuse and name the state, rather than reporting an initiation the
#      state machine dropped, and disconnect then returns the session to
#      offline.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r
from . import REPO_ROOT_POSIX

NEEDS_BLENDER = True

# macOS runners block loopback HTTP to Blender's in-process MCP server, so the
# rig does not select this scenario there.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

# The worker's own solver server, which is the one host this scenario can
# reach: checks J and K connect to it to settle the session's phase.
RIG_SERVER_PORT = <<SERVER_PORT>>
RIG_SOLVER_PATH = <<SOLVER_PATH_REPR>>

_rid = [800]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    groups_mod = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    facade_mod = __import__(pkg + ".core.facade", fromlist=["tick"])
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port

    # ----- A --------------------------------------------------------
    info = call("get_connection_info")
    status = info.get("connection_status") or {}
    mcp_check(result, "A_connection_info_reports_disconnected",
              info.get("status") == "success"
              and status.get("connected") is False
              and status.get("server_running") is False,
              {"connection_status": status})

    # ----- B --------------------------------------------------------
    guarded = {}
    for name in ("get_remote_status", "is_remote_server_running",
                 "start_remote_server", "stop_remote_server"):
        guarded[name] = call(name)
    mcp_check(result, "B_remote_status_tools_say_connect_first",
              all(v.get("status") == "error"
                  and "Not connected" in v.get("message", "")
                  and "connection" in v.get("message", "")
                  for v in guarded.values()),
              {k: v.get("message") for k, v in guarded.items()})

    # ----- C --------------------------------------------------------
    pipeline = {
        "transfer_data": call("transfer_data"),
        "run_simulation": call("run_simulation"),
        "resume_simulation": call("resume_simulation"),
        "fetch_animation": call("fetch_animation"),
        "update_params": call("update_params"),
        "clear_local_animation": call("clear_local_animation"),
    }
    mcp_check(result, "C_pipeline_tools_name_the_disconnected_state",
              all(v.get("status") == "error"
                  and "Disconnected" in v.get("message", "")
                  for v in pipeline.values()),
              {k: v.get("message") for k, v in pipeline.items()})

    # ----- D --------------------------------------------------------
    server_side = {
        "compile_project": call("compile_project"),
        "git_pull_remote": call("git_pull_remote"),
        "execute_server_command": call("execute_server_command",
                                       {"server_script": "echo probe"}),
        "execute_shell_command": call("execute_shell_command",
                                      {"shell_command": "echo probe"}),
        "debug_data_send": call("debug_data_send", {"data_size_mb": 1}),
        "delete_remote_data": call("delete_remote_data"),
    }
    mcp_check(result, "D_server_side_tools_refuse",
              all(v.get("status") == "error" and v.get("message")
                  for v in server_side.values()),
              {k: v.get("message") for k, v in server_side.items()})

    # ----- E --------------------------------------------------------
    bye = call("disconnect")
    mcp_check(result, "E_disconnect_without_a_connection_is_refused",
              bye.get("status") == "error" and "Not connected" in bye.get("message", ""),
              {"reply": bye})

    # ----- F --------------------------------------------------------
    usd = call("export_usd", {"filepath": "/tmp/bl_mcp_probe.usd"})
    abc = call("export_alembic", {"filepath": "/tmp/bl_mcp_probe.abc"})
    fetch = call("get_fetch_status")
    reason = fetch.get("export_blocked_reason") or ""
    mcp_check(result, "F_export_names_the_preflight_reason",
              usd.get("status") == "error" and abc.get("status") == "error"
              and reason and reason in usd.get("message", "")
              and reason in abc.get("message", "")
              and fetch.get("export_ready") is False
              and fetch.get("fetched_frames") == []
              # Refused, so neither exporter wrote anything.
              and not __import__("os").path.exists("/tmp/bl_mcp_probe.usd")
              and not __import__("os").path.exists("/tmp/bl_mcp_probe.abc"),
              {"usd": usd.get("message"), "alembic": abc.get("message"),
               "preflight_reason": reason})

    # ----- G --------------------------------------------------------
    local = {
        "git_pull_local": call("git_pull_local"),
        "abort_operation": call("abort_operation"),
        "refresh_ui": call("refresh_ui"),
    }
    mcp_check(result, "G_local_only_tools_still_work",
              all(v.get("status") == "success" for v in local.values()),
              {k: v.get("message") for k, v in local.items()})

    # ----- H --------------------------------------------------------
    update = call("update_remote_status")
    resume_from = call("resume_simulation_from", {"frame": 3})
    terminate = call("terminate_simulation")
    save_quit = call("save_and_quit_simulation")
    receive = call("debug_data_receive")
    console = call("show_console")
    mcp_check(result, "H_session_tools_report_the_disconnected_state",
              update.get("status") == "error"
              and "Not connected" in update.get("message", "")
              and resume_from.get("status") == "error"
              and "Disconnected" in resume_from.get("message", "")
              # These two raise a flag for a run to notice; with no run they
              # still report the initiation and carry the status they saw.
              and terminate.get("status") == "success"
              and terminate.get("current_status") == "Disconnected"
              and save_quit.get("status") == "success"
              and save_quit.get("current_status") == "Disconnected"
              and receive.get("status") == "error"
              # show_console needs a window. The rig runs Blender with a UI, a
              # --background one has none, so only the answered-either-way
              # invariant holds in both.
              and console.get("status") in ("success", "error")
              and bool(console.get("message")),
              {"update": update.get("message"),
               "resume_from": resume_from.get("message"),
               "terminate": terminate.get("current_status"),
               "save_quit": save_quit.get("current_status"),
               "receive": receive.get("message"),
               "console": console.get("message")})

    # ----- I --------------------------------------------------------
    first_install = call("install_paramiko")
    second_install = call("install_docker")
    mcp_check(result, "I_one_install_at_a_time",
              first_install.get("status") == "success"
              and "initiated" in first_install.get("message", "")
              and second_install.get("status") == "error"
              and "in progress" in second_install.get("message", ""),
              {"first": first_install.get("message"),
               "second": second_install.get("message")})

    # ----- J --------------------------------------------------------
    # Last: these leave the session connected or connecting, which every check
    # above reads as its precondition.
    # The engine applies a worker thread's event on a tick, and its own tick
    # is a Blender timer that cannot fire while this driver holds the main
    # thread. So a loop that waits on a connection has to pump the engine the
    # way that timer would, or it reads one phase forever.
    def connection_status():
        facade_mod.tick()
        return call("get_connection_info").get("connection_status") or {}

    # Offline is every phase a connect can be issued from: not connected, and
    # not still handshaking. A failed attempt can leave its reason in the
    # status line, so the test is the phase rather than one status string.
    def is_offline(status):
        return (not status.get("connected")
                and status.get("status") != "Connecting...")

    # Return the session to the offline phase, and report the status it
    # settled on. A connection that reached its host holds the session until
    # it is torn down, and one still handshaking against a host that never
    # answers holds it until the attempt is called off; disconnect covers
    # both, so the phase the next connect starts from is observed rather than
    # assumed.
    def settle_offline(timeout=60.0):
        deadline = time.time() + timeout
        while True:
            status = connection_status()
            if is_offline(status):
                return status
            call("disconnect")
            if time.time() > deadline:
                return status
            time.sleep(0.1)

    def wait_connected(timeout=60.0):
        deadline = time.time() + timeout
        while True:
            status = connection_status()
            if status.get("connected") or time.time() > deadline:
                return status
            time.sleep(0.1)

    # connect_local carries no port of its own: it reads the panel's shared
    # port field, which is where the worker's solver server has to be named
    # for the LOCAL connect to reach it.
    ssh_state = groups_mod.get_addon_data(bpy.context.scene).ssh_state
    ssh_state.docker_port = RIG_SERVER_PORT

    entry_points = (
        ("connect_local", {"path": RIG_SOLVER_PATH}),
        ("connect_ssh", {"host": "rig.invalid", "username": "nobody",
                         "key_path": "/nonexistent/key", "remote_path": "/tmp"}),
        ("connect_win_native", {"path": "/tmp", "port": 9091}),
        ("connect", None),
    )
    started_from = {}
    reports = {}
    for name, args in entry_points:
        started_from[name] = settle_offline()
        reports[name] = call(name, args)
    local_conn = reports["connect_local"]
    ssh_conn = reports["connect_ssh"]
    win_conn = reports["connect_win_native"]
    generic = reports["connect"]
    mcp_check(result, "J_connect_family_initiates_from_offline",
              all(is_offline(v) for v in started_from.values())
              and local_conn.get("status") == "success"
              and local_conn.get("connection_type") == "local"
              and local_conn.get("path") == RIG_SOLVER_PATH
              and ssh_conn.get("status") == "success"
              and ssh_conn.get("connection_type") == "ssh"
              and ssh_conn.get("host") == "rig.invalid" and ssh_conn.get("port") == 22
              and win_conn.get("status") == "success"
              and win_conn.get("connection_type") == "win_native"
              and win_conn.get("port") == 9091
              # The generic entry point reads the settings the others wrote,
              # so it reports connecting rather than success.
              and generic.get("status") == "connecting"
              and bool(generic.get("connection_type")),
              {"started_from": {k: v.get("status")
                                for k, v in started_from.items()},
               "local": local_conn,
               "ssh": ssh_conn, "win_native": win_conn, "generic": generic})

    # ----- K --------------------------------------------------------
    # The refusal the connect family owes an agent: with a connection up, a
    # second connect says so instead of reporting an initiation that changed
    # nothing. The worker's own solver server is the host, so the session
    # stays online for the whole check.
    settle_offline()
    ssh_state.docker_port = RIG_SERVER_PORT
    held = call("connect_local", {"path": RIG_SOLVER_PATH})
    online = wait_connected()
    second = call("connect_local", {"path": RIG_SOLVER_PATH})
    second_generic = call("connect")
    released = call("disconnect")
    after = settle_offline()
    mcp_check(result, "K_connect_refuses_while_a_connection_is_up",
              held.get("status") == "success"
              and online.get("connected") is True
              and second.get("status") == "error"
              and "already connected" in second.get("message", "")
              and second_generic.get("status") == "error"
              and "already connected" in second_generic.get("message", "")
              and released.get("status") == "success"
              # A teardown that ran leaves the session reset, so this one is
              # held to the disconnected status itself, not just the phase.
              and after.get("status") == "Disconnected",
              {"held": held.get("message"), "online": online,
               "second": second.get("message"),
               "second_generic": second_generic.get("message"),
               "released": released.get("message"), "after": after})

    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    # repr() for the path so a Windows separator survives the substitution
    # instead of being reparsed as an escape.
    return (
        _DRIVER_TEMPLATE
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<SOLVER_PATH_REPR>>", repr(REPO_ROOT_POSIX))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
