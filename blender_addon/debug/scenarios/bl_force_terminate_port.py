# File: scenarios/bl_force_terminate_port.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The Force Terminate Process button ends the local solver server from the state the
# add-on cannot otherwise leave.
#
# WHY. A native Connect (Windows Native, macOS Native) is refused while a
# ppf-cts-server from an earlier session still holds the port: "Port N is in
# use" when the holder is not one of ours, "A solver server is already
# running on port N" when it is ours but runs another build. A protocol
# version mismatch is the third shape: a server orphaned from an earlier
# binary keeps serving its old PROTOCOL_VERSION across the artist's update.
# Stop Server is reachable only once CONNECTED, and in the refused state the
# add-on is not, so the artist's only way out was a terminal. Force Terminate
# Process is that way out: it ends whatever listens on the configured port
# without a connection, and resets the connection state so a fresh Connect
# starts clean.
#
# THE ROW IS DRAWN ONLY WHILE A CONNECTION FAILURE STANDS, next to the
# refusal that names it, and the button and its status line are drawn
# together. Outside that state the server is under Start Server and Stop
# Server, and a standing kill button there is an invitation to end a healthy
# server; a build refusal (isolated vertices, no usable rest shape) is not a
# connection failure and takes its own repair button instead. C, D and E
# below are that rule, in its three states.
#
# The rig owns this worker's server on its own port, which is exactly the
# shape of the refused state: a server the add-on did not start, listening
# where the add-on wants to connect. The scenario asks the button to end it
# and reads the socket, not the state machine, for the verdict; the
# orchestrator tears the server down from a ``finally`` however this ends,
# so a kill that fails leaves nothing behind.
#
# Subtests:
#   A. status_says_server_listening: the panel predicate reports a server of
#      ours on the worker's port before anything is killed.
#   B. poll_true_disconnected_local: the operator's poll is True for the
#      local server type while disconnected and idle.
#   C. row_hidden_without_error: with no connection error the panel draws
#      neither the button nor its status line, however pollable the operator
#      is. This is the state a healthy session spends all its time in.
#   D. row_drawn_on_connection_refusal: with the "already running on port N"
#      refusal standing while disconnected, both are drawn.
#   E. row_drawn_on_protocol_mismatch: both are drawn for a mismatch carrying
#      NO error text, which is what the malformed-response paths leave, so
#      the panel is reading RemoteStatus rather than the message.
#   F. poll_false_remote_disconnected: for every SSH and Docker type the poll
#      is False while disconnected and the panel draws no button, with the
#      refusal staged so it is the missing transport that hides it.
#   G. remote_button_drawn_when_connected: with the engine ONLINE and a
#      remote Start Server refused over a held port, the panel draws the
#      button for a remote type. An error naming a held port is the one that
#      is offered while CONNECTED too, since the connected kill clears it
#      through the live backend. The state is staged, since the rig has no
#      SSH host on this leg; what the connected remote kill DOES is covered
#      by rig_remote_kill_port_scope, against stand-in processes.
#   H. operator_runs: invoked through a VIEW_3D temp_override, the operator
#      returns FINISHED.
#   I. port_closed_after_kill: the worker's port stops accepting connections
#      within the deadline.
#   J. state_reset_after_kill: the engine is OFFLINE with no error and no
#      backend attached, which is what a fresh Connect needs.
#   K. status_says_nothing_listening: the same predicate now reports nothing
#      on the port.
#   L. second_kill_reports_nothing: with the port free, the operator's own
#      kill reports no listener and no survivors rather than failing.
#   M. port_closed_from_host: the same question as I, asked from the
#      orchestrator process, so the verdict does not rest on code running
#      inside the process under test.
#
# Platform: the native server types are Windows and macOS. Linux runs the
# LOCAL type, whose stop_server is the same local kill, so the scenario
# covers all three legs and selects the type by platform, the way
# ``DriverHelpers.connect`` does.

from __future__ import annotations

import socket
import time

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Server lifecycle only: no scene, no build, no solve, so the verdict is the
# same against either solver binary.
BACKENDS = ("real",)

# How long the port may take to go quiet after the operator returns. The
# kill lands on another process, so the listener disappears when that
# process exits rather than when the command returns.
_PORT_DEATH_DEADLINE_S = 20.0


_DRIVER_BODY = r'''
import sys
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

REPO = <<REPO_REPR>>
SERVER_PORT = <<SERVER_PORT>>
PORT_DEATH_DEADLINE_S = <<PORT_DEATH_DEADLINE_S>>
REMOTE_TYPES = ("CUSTOM", "COMMAND", "DOCKER", "DOCKER_SSH", "DOCKER_SSH_COMMAND")


class _FakeLayout:
    # Records operator() and label() calls so the Connection box can be
    # inspected without a real Blender UILayout.
    def __init__(self):
        self.operators = []
        self.labels = []
        self.enabled = True

    def prop(self, *a, **kw):
        pass

    def label(self, text="", icon="", **kw):
        self.labels.append((text, icon))

    def operator(self, idname, *a, **kw):
        self.operators.append(idname)
        return self

    def row(self, **kw):
        return self

    def column(self, **kw):
        return self

    def box(self):
        return self


class _PanelSelf:
    def __init__(self, layout):
        self.layout = layout


def _v3d_override():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                for rgn in a.regions:
                    if rgn.type == "WINDOW":
                        return {"window": w, "area": a, "region": rgn}
    return {}


def _port_open():
    import socket as _socket
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect(("127.0.0.1", SERVER_PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


try:
    dh = DriverHelpers(pkg, result)
    conn_ops = __import__(pkg + ".ui.connection_ops",
                          fromlist=["SOLVER_OT_ForceTerminatePort"])
    main_panel = __import__(pkg + ".ui.main_panel",
                            fromlist=["MAIN_PT_RemotePanel"])
    server_kill = __import__(pkg + ".core.server_kill",
                             fromlist=["kill_local_server"])
    OpCls = conn_ops.SOLVER_OT_ForceTerminatePort
    kill_idname = OpCls.bl_idname

    if sys.platform.startswith("win"):
        server_type, path_field = "WIN_NATIVE", "win_native_path"
    elif sys.platform == "darwin":
        server_type, path_field = "MAC_NATIVE", "mac_native_path"
    else:
        server_type, path_field = "LINUX_NATIVE", "linux_native_path"

    root = dh.groups.get_addon_data(bpy.context.scene)
    props = root.ssh_state
    root.state.project_name = "kill_server"
    root.state.show_connection = True
    props.server_type = server_type
    setattr(props, path_field, REPO)
    props.docker_port = SERVER_PORT
    dh.com.set_project_name("kill_server")

    def draw_box():
        fl = _FakeLayout()
        main_panel.MAIN_PT_RemotePanel.draw(_PanelSelf(fl), bpy.context)
        return fl

    # The engine snapshot is written directly for the staged states below:
    # this leg has no SSH host to connect to and no refused Connect to
    # provoke, so no event is dispatched and no backend exists. The snapshot
    # is put back however the block ends, so each check starts from the real
    # state.
    from dataclasses import replace as _replace
    engine = dh.facade.engine

    class _staged:
        def __init__(self, **fields):
            self.fields = fields

        def __enter__(self):
            self.saved = engine.state
            with engine._lock:
                engine._state = _replace(self.saved, **self.fields)
            return self

        def __exit__(self, *exc):
            with engine._lock:
                engine._state = self.saved
            return False

    ONLINE = type(engine.state.phase).ONLINE
    # The NativeServerMismatch refusal, in the shape the rig's own server
    # gives it: ours is listening on the worker's port, from another build.
    REFUSAL = (
        "A solver server is already running on port %d, and its runs use "
        "another build." % SERVER_PORT
    )

    def _row_drawn():
        """Whether the panel drew the button AND its status line.

        The expected line is computed FIRST, which primes the listener cache
        the draw then reads, so the two cannot disagree about the port across
        a probe taken a moment apart.
        """
        conn_ops._listener_cache = None
        text, _icon = conn_ops.force_terminate_status(props)
        fl = draw_box()
        return (
            kill_idname in fl.operators,
            any(t == text for t, _ in fl.labels),
        )

    # ----- A: the predicate sees the rig's server ------------------------
    conn_ops._listener_cache = None
    text, icon = conn_ops.force_terminate_status(props)
    dh.record(
        "A_status_says_server_listening",
        icon == "CHECKMARK" and str(SERVER_PORT) in text,
        {"text": text, "icon": icon, "server_type": server_type},
    )

    # ----- B: poll is True for the local type while disconnected ---------
    poll_local = bool(OpCls.poll(bpy.context))
    dh.record(
        "B_poll_true_disconnected_local",
        poll_local and not dh.com.is_connected(),
        {"poll": poll_local, "phase": dh.facade.engine.state.phase.name},
    )

    # ----- C: no connection error, no row -------------------------------
    # The state the artist is in for all of a healthy session. The server is
    # under Start Server and Stop Server, so neither the button nor its
    # status line is drawn, however pollable the operator is.
    with _staged(error="", version_ok=True):
        btn_idle, line_idle = _row_drawn()
    dh.record(
        "C_row_hidden_without_error",
        not btn_idle and not line_idle,
        {"button": btn_idle, "status_line": line_idle,
         "poll": bool(OpCls.poll(bpy.context))},
    )

    # ----- D: a connection refusal draws it -----------------------------
    with _staged(error=REFUSAL):
        btn_refused, line_refused = _row_drawn()
    dh.record(
        "D_row_drawn_on_connection_refusal",
        btn_refused and line_refused,
        {"button": btn_refused, "status_line": line_refused,
         "error": REFUSAL},
    )

    # ----- E: a protocol version mismatch draws it ----------------------
    # Read off RemoteStatus, not off the message: the malformed-response
    # paths set version_ok False with no error text, and the row has to
    # appear there too. Staged ONLINE, which is where a mismatch is seen.
    with _staged(phase=ONLINE, version_ok=False, error=""):
        status_name = dh.com.info.status.name
        btn_mismatch, line_mismatch = _row_drawn()
    dh.record(
        "E_row_drawn_on_protocol_mismatch",
        btn_mismatch and line_mismatch
        and status_name == "PROTOCOL_VERSION_MISMATCH",
        {"button": btn_mismatch, "status_line": line_mismatch,
         "status": status_name},
    )

    # ----- F: remote types refuse it and hide it while disconnected ------
    # The refusal is staged here too, so what hides the row is the missing
    # transport rather than the missing error.
    remote = {}
    with _staged(error=REFUSAL):
        for t in REMOTE_TYPES:
            props.server_type = t
            remote[t] = {
                "poll": bool(OpCls.poll(bpy.context)),
                "drawn": kill_idname in draw_box().operators,
            }
    props.server_type = server_type
    dh.record(
        "F_poll_false_remote_disconnected",
        all(not v["poll"] and not v["drawn"] for v in remote.values()),
        remote,
    )

    # ----- G: a connected remote type draws the button ------------------
    # Staged with the refusal a CONNECTED Start Server raises when something
    # on the remote holds the port, which the connected kill clears through
    # the live backend: an error naming a held port is offered whether or not
    # the add-on is connected, unlike every other error. A port the rig does
    # not own keeps the staleness probe out of this check whatever the
    # message is capitalized like. The connected remote kill itself is not
    # exercised here; what it DOES is covered by rig_remote_kill_port_scope.
    REMOTE_REFUSAL = (
        "Server port %d is already in use on the remote host." % (SERVER_PORT + 1)
    )
    try:
        props.server_type = "CUSTOM"
        with _staged(phase=ONLINE, error=REMOTE_REFUSAL):
            drawn_remote = kill_idname in draw_box().operators
            poll_remote = bool(OpCls.poll(bpy.context))
    finally:
        props.server_type = server_type
    dh.record(
        "G_remote_button_drawn_when_connected",
        drawn_remote and poll_remote,
        {"drawn": drawn_remote, "poll": poll_remote, "error": REMOTE_REFUSAL},
    )

    # ----- H: the operator runs -----------------------------------------
    with bpy.context.temp_override(**_v3d_override()):
        res = bpy.ops.solver.force_terminate_port("EXEC_DEFAULT")
    dh.facade.tick()
    dh.record("H_operator_runs", res == {"FINISHED"}, {"result": list(res)})

    # ----- I: the listen socket is gone ---------------------------------
    started = time.time()
    open_after = _port_open()
    while open_after and time.time() - started < PORT_DEATH_DEADLINE_S:
        time.sleep(0.5)
        open_after = _port_open()
    dh.record(
        "I_port_closed_after_kill",
        not open_after,
        {"port_open": open_after, "waited_s": round(time.time() - started, 2)},
    )

    # ----- J: the state machine is reset for a fresh Connect ------------
    for _ in range(5):
        dh.facade.tick()
        time.sleep(0.1)
    s = dh.facade.engine.state
    dh.record(
        "J_state_reset_after_kill",
        s.phase.name == "OFFLINE" and s.server.name == "UNKNOWN"
        and not s.error and dh.facade.runner.backend is None,
        {"phase": s.phase.name, "server": s.server.name, "error": s.error,
         "backend": repr(dh.facade.runner.backend)},
    )

    # ----- K: the predicate now says nothing is there -------------------
    conn_ops._listener_cache = None
    text, icon = conn_ops.force_terminate_status(props)
    dh.record(
        "K_status_says_nothing_listening",
        icon == "INFO" and str(SERVER_PORT) in text,
        {"text": text, "icon": icon},
    )

    # ----- L: a kill on a free port reports nothing, and raises nothing --
    report = server_kill.kill_local_server(SERVER_PORT)
    dh.record(
        "L_second_kill_reports_nothing",
        report.checked and report.nothing_found,
        {"checked": report.checked, "killed": list(report.killed),
         "survivors": list(report.survivors), "error": report.error},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
'''


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<REPO_REPR>>", repr(REPO_ROOT_POSIX))
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PORT_DEATH_DEADLINE_S>>", repr(_PORT_DEATH_DEADLINE_S))
    )


def _port_open_from_host(ctx: r.ScenarioContext) -> bool:
    """Whether the worker's port still accepts a connection, asked from the
    orchestrator process so the verdict does not rest on code running
    inside the process under test."""
    try:
        sock = socket.create_connection((ctx.host, ctx.server_port), timeout=3.0)
    except OSError:
        return False
    sock.close()
    return True


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    checks = dict(result.get("checks", {}))
    # M: the same question from outside Blender, on the driver's deadline.
    deadline = time.time() + _PORT_DEATH_DEADLINE_S
    open_now = _port_open_from_host(ctx)
    while open_now and time.time() < deadline:
        time.sleep(0.5)
        open_now = _port_open_from_host(ctx)
    checks["M_port_closed_from_host"] = {
        "ok": not open_now, "details": {"port_open": open_now},
    }
    return r.report_named_checks(checks)
