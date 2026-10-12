# File: scenarios/bl_server_ping_keeps_it_up.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The add-on's ping is what keeps a server up while Blender is connected and
# idle, and the server's idle stop is what ends one an artist forgot.
#
# A server stops itself `Stop Idle Server After` seconds after its last
# request when nothing runs on it (`rig_server_idle_stop` times that on the
# binary). A connected add-on that is not building or running sends nothing on
# its own, so it pings the server every `Ping Interval` seconds, from a thread
# of its own rather than Blender's timer. This scenario
# connects through this platform's native backend and lets the add-on start
# its own server, through the Start Server operator, with Stop Idle Server
# After 3 s, on a port of its own (the rig's server runs with no idle stop and
# sets the NO_SPAWN variable this scenario clears).
#
# Liveness is read off the server's own process handle and its log, never by
# asking the server, because a question is a request and would keep it up.
#
# The driver blocks Blender's main thread for its whole run, which is the
# condition the ping runs off the main thread for: while it waits, the
# scenario only sleeps, and nothing but the ping thread can reach the server.
#
# Subtests:
#   A. start_server_refuses_a_ping_slower_than_the_idle_stop: Ping Interval 5
#      with Stop Idle Server After 3 makes Start Server cancel with an error
#      naming both (raised, as an operator's ERROR report is when Python calls
#      it), and start nothing.
#   B. the_launch_carries_the_idle_stop: the operator's server runs with
#      `--idle-timeout 3` and reports idle_timeout_s 3.
#   C. a_ping_every_second_keeps_it_up: connected for 8 s with Ping Interval 1
#      and Blender's main thread blocked throughout, the server is still
#      running and logged no stop.
#   D. a_ping_slower_than_the_idle_stop_lets_it_stop: Ping Interval raised to
#      6 while connected (the panel warns; nothing refuses a running
#      connection), the server stops 1.5 to 8 s later (its last 1 s ping went
#      out up to a second before the switch): what kept it up in C is the
#      ping, and nothing else.
#   E. after_a_disconnect_it_stops: a fresh server, Ping Interval 1, then
#      Disconnect without Stop Server; it stops within 3 to 6 s, which is the
#      server an artist forgot.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Server lifecycle only: no scene, no build, no solve.
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import os
import socket
import sys
import time
import traceback

REPO = "<<REPO>>"
IDLE_S = 3

NATIVE = {
    "win32": ("WIN_NATIVE", "win_native_path", "win_native", "PPF_WIN_NATIVE_NO_SPAWN"),
    "darwin": ("MAC_NATIVE", "mac_native_path", "mac_native", "PPF_MAC_NATIVE_NO_SPAWN"),
}.get(sys.platform, ("LINUX_NATIVE", "linux_native_path", "linux_native",
                     "PPF_LINUX_NATIVE_NO_SPAWN"))
SERVER_TYPE, PATH_FIELD, BACKEND_TYPE, NO_SPAWN = NATIVE
saved_env = {key: os.environ.get(key) for key in (NO_SPAWN, "PPF_CTS_DATA_ROOT")}
dh = None
spawned = []


def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def wait_for(seconds, until=None):
    # Sleep on the main thread, as a busy Blender would hold it, and return
    # the seconds waited, early when `until()` holds.
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        if until is not None and until():
            break
        time.sleep(0.05)
    return time.monotonic() - start


def pump(done, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if done(dh.facade.engine.state):
            return True
        time.sleep(0.2)
    return False


def window_override():
    win = bpy.context.window_manager.windows[0]
    area = next((a for a in win.screen.areas if a.type == "VIEW_3D"), win.screen.areas[0])
    return bpy.context.temp_override(window=win, area=area)


def start_server():
    with window_override():
        return bpy.ops.ssh.start_server()


def connect_and_start(port, ping):
    ssh.ping_interval = ping
    ssh.docker_port = port
    dh.com.set_project_name(addon.state.project_name)
    connect(REPO, port, ssh.native_device)
    if not pump(lambda s: s.phase.name == "ONLINE"
                or (s.phase.name == "OFFLINE" and bool(s.error)), 30.0):
        raise RuntimeError("never came online")
    done = start_server()
    if done == {"CANCELLED"}:
        raise RuntimeError("Start Server cancelled")
    if not pump(lambda s: s.server.name == "RUNNING" or bool(s.error), 120.0):
        raise RuntimeError(f"server never ran: {dh.facade.engine.state.error!r}")
    backend = facade.runner.backend
    proc = getattr(backend, "_process", None)
    if proc is None:
        raise RuntimeError("the add-on attached to a server rather than starting one")
    spawned.append(proc)
    return proc


def server_log():
    path = os.path.join(REPO, "server.log")
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def stops_logged(before):
    return server_log()[len(before):].count("idle stop: no request for")


def disconnect():
    dh.com.disconnect()
    for _ in range(10):
        dh.facade.tick()
        time.sleep(0.1)


try:
    dh = DriverHelpers(pkg, result)
    conn = __import__(pkg + ".core.connection", fromlist=["native_resolvers"])
    facade = __import__(pkg + ".core.facade", fromlist=["runner"])
    connect = getattr(dh.com, f"connect_{BACKEND_TYPE}")
    os.environ.pop(NO_SPAWN, None)
    os.environ["PPF_CTS_DATA_ROOT"] = os.path.join(PROBE_DIR, "idle_data")

    addon = dh.configure_state(project_name="server_ping", frame_count=2)
    ssh = addon.ssh_state
    ssh.server_type = SERVER_TYPE
    setattr(ssh, PATH_FIELD, REPO)
    ssh.native_device = dh.resolve_native_device(BACKEND_TYPE, REPO)
    ssh.server_idle_timeout = IDLE_S

    # ----- A: Start Server refuses a ping slower than the idle stop -------
    port = free_port()
    ssh.docker_port = port
    ssh.ping_interval = 5
    dh.com.set_project_name(addon.state.project_name)
    connect(REPO, port, ssh.native_device)
    pump(lambda s: s.phase.name == "ONLINE", 30.0)
    # An operator called from Python raises the ERROR it reports.
    try:
        refused, message = sorted(start_server()), ""
    except RuntimeError as e:
        refused, message = ["CANCELLED"], str(e)
    started = facade.runner.backend is not None and getattr(
        facade.runner.backend, "_process", None) is not None
    dh.record("A_start_server_refuses_a_ping_slower_than_the_idle_stop",
              refused == ["CANCELLED"] and not started
              and "Ping Interval (5 s)" in message and "(3 s)" in message,
              {"operator": refused, "message": message, "started": started})
    disconnect()

    # ----- B, C: the operator's server, kept up by a ping every second ----
    log_before = server_log()
    proc = connect_and_start(free_port(), 1)
    args = list(proc.args)
    response = conn._query_ppf_cts_server(ssh.docker_port) or {}
    dh.record("B_the_launch_carries_the_idle_stop",
              args[-2:] == ["--idle-timeout", str(IDLE_S)]
              and response.get("idle_timeout_s") == IDLE_S,
              {"args": args, "idle_timeout_s": response.get("idle_timeout_s")})
    ran = wait_for(8.0, until=lambda: proc.poll() is not None)
    dh.record("C_a_ping_every_second_keeps_it_up",
              proc.poll() is None and stops_logged(log_before) == 0,
              {"waited_s": round(ran, 2), "exit": proc.poll(),
               "idle_stops_logged": stops_logged(log_before)})

    # ----- D: the same connection, its ping slowed past the idle stop -----
    ssh.ping_interval = 6
    slowed = time.monotonic()
    wait_for(IDLE_S + 7.0, until=lambda: proc.poll() is not None)
    stopped_after = time.monotonic() - slowed if proc.poll() is not None else None
    dh.record("D_a_ping_slower_than_the_idle_stop_lets_it_stop",
              # The last 1 s ping went out up to a second before the switch.
              stopped_after is not None and IDLE_S - 1.5 <= stopped_after <= IDLE_S + 5.0
              and stops_logged(log_before) == 1,
              {"stopped_after_s": None if stopped_after is None else round(stopped_after, 2),
               "idle_stops_logged": stops_logged(log_before),
               "phase": dh.facade.engine.state.phase.name})
    disconnect()

    # ----- E: a forgotten server, after a disconnect ----------------------
    log_before = server_log()
    proc = connect_and_start(free_port(), 1)
    wait_for(2.0)
    disconnect()
    gone_from = time.monotonic()
    wait_for(IDLE_S + 6.0, until=lambda: proc.poll() is not None)
    stopped_after = time.monotonic() - gone_from if proc.poll() is not None else None
    dh.record("E_after_a_disconnect_it_stops",
              stopped_after is not None and stopped_after <= IDLE_S + 3.0
              and stops_logged(log_before) == 1,
              {"stopped_after_s": None if stopped_after is None else round(stopped_after, 2),
               "idle_stops_logged": stops_logged(log_before)})
except Exception as exc:
    result.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    for proc in spawned:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return dl.DRIVER_LIB + _DRIVER_BODY.replace("<<REPO>>", REPO_ROOT_POSIX)


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    checks = dict(result.get("checks", {}))
    for name in ("A_start_server_refuses_a_ping_slower_than_the_idle_stop",
                 "B_the_launch_carries_the_idle_stop",
                 "C_a_ping_every_second_keeps_it_up",
                 "D_a_ping_slower_than_the_idle_stop_lets_it_stop",
                 "E_after_a_disconnect_it_stops"):
        checks.setdefault(name, {"ok": False, "details": "not recorded"})
    verdict = r.report_named_checks(checks)
    if result.get("errors"):
        verdict["status"] = "fail"
        verdict.setdefault("violations", []).extend(result["errors"][:2])
    return verdict
