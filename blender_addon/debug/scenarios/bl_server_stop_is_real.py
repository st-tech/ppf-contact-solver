# File: scenarios/bl_server_stop_is_real.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Stop Server has to stop the server, and only the socket can say so.
#
# ``LinuxNativeBackend.stop_server`` and its macOS twin end the co-located
# ``ppf-cts-server``. When the addon spawned it they terminate the Popen
# handle; when the addon ADOPTED one (a Blender restart, an addon reload, or
# the rig, which owns the server and sets ``PPF_<PLATFORM>_NATIVE_NO_SPAWN``)
# they hand the port to ``core/server_kill.py``, which finds the listener with
# ``lsof -ti tcp:<port>``, sends SIGTERM, then SIGKILL to a survivor. Every
# step of that is best effort and its outcome is discarded, and
# ``effect_runner._do_stop_server`` clears the response cache and
# dispatches ``ServerStopped`` unconditionally right after, which is the
# single event ``transitions.py`` turns into ``server=UNKNOWN``. So every
# piece of state after a stop is derived from the addon having ASKED, never
# from anything having exited: a stop that killed nothing produces the
# same state, the same "Server stopped." log line, and the same cleared
# cache as a stop that worked. What the user then sees is a port that
# keeps serving and a Start Server that lands on an address already
# bound.
#
# This scenario reads the socket instead of the state machine:
#
#   A. server_answers_before_stop: a raw TCMD status query on this
#      worker's port returns the addon's own ``PROTOCOL_VERSION``. That
#      establishes the server is genuinely serving without asking the
#      engine what it believes, and it is what makes check C mean
#      something: a port that was never up would go quiet for free.
#   B. stop_request_settled: the engine leaves STOPPING for UNKNOWN
#      within its deadline. A PRECONDITION, not the verdict. It is
#      precisely the assertion a stop that kills nothing still passes,
#      and it is here so a stop that never returns reports as a wedged
#      request rather than as a port that stayed alive.
#   C. port_closed_after_stop: the same raw probe, repeated to a
#      deadline, has to stop connecting. A refused connect is the only
#      available evidence that the process owning the listen socket is
#      gone; the kernel drops a listener the moment its process exits,
#      so there is no lingering-LISTEN window to tolerate.
#   D. port_closed_from_host: run() repeats the probe from the
#      orchestrator worker, so the verdict does not rest on code running
#      inside the process under test.
#
# Why a cheaper test does not reach this: the claim is the EFFECT of a
# shell command on another process, so any in-process double (a fake
# backend, a patched ``exec_command``, a unit test asserting the command
# string) can only restate the command the code already contains. The
# rig is where a real ``ppf-cts-server`` is listening on a real port, so
# it is the only place the kill is observable end to end.
#
# Orphan safety, which matters because the rig runs workers in parallel
# and a stray listener on a worker port poisons later runs: this
# scenario spawns nothing of its own, and every socket it opens is
# closed on every path (a live fd on that port would put Blender itself
# on the ``lsof -ti tcp:<port>`` list the kill path targets). The server
# it asks the addon to kill is the orchestrator's own child on this
# worker's private port, and ``orchestrator.run_one`` tears it down from
# a ``finally`` however the scenario ends, so a stop that fails leaves
# nothing behind. Both kill patterns in ``stop_server`` carry that port,
# so neither can reach another worker's server.

from __future__ import annotations

import socket
import time

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Server lifecycle only: no scene, no build, no solve, so the verdict is
# the same against either solver binary.
BACKENDS = ("real",)
# ``LinuxNativeBackend`` and ``MacNativeBackend`` are the co-located path on
# Linux and macOS, and their stop_server is what this pins. Windows runs
# WIN_NATIVE, whose
# stop_server terminates a Popen handle or falls back to ``taskkill``,
# and neither ``lsof`` nor ``pkill`` exists there.
PLATFORMS = ("linux", "darwin")

# How long the port may take to go quiet after the addon reports the
# stop complete. The kill lands on another process, so the listener
# disappears when that process exits rather than when the command
# returns.
_PORT_DEATH_DEADLINE_S = 20.0


_DRIVER_BODY = r'''
import json
import socket
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "server_stop_is_real"
PROBE_TIMEOUT_S = 5.0
PORT_DEATH_DEADLINE_S = <<PORT_DEATH_DEADLINE_S>>
STOP_SETTLE_TIMEOUT_S = 30.0

try:
    dh = DriverHelpers(pkg, result)
    protocol = __import__(pkg + ".core.protocol",
                          fromlist=["HEADER_TEXT_CMD", "PROTOCOL_VERSION"])

    def probe_port():
        # One short-lived TCMD round trip aimed straight at the port,
        # over the addon's own wire constants. Nothing here consults
        # engine state, so what comes back is what the socket did.
        # The socket is closed on every path: an open fd on this port
        # would list Blender under `lsof -ti tcp:<port>`, which is the
        # set the native backend's stop_server kills.
        info = {"connected": False, "answered": False,
                "protocol_version": "", "status": "", "error": ""}
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(PROBE_TIMEOUT_S)
        try:
            sock.connect(("127.0.0.1", SERVER_PORT))
            info["connected"] = True
            payload = ("--name " + PROJECT_NAME + " ").encode()
            sock.sendall(protocol.HEADER_TEXT_CMD)
            sock.sendall(len(payload).to_bytes(4, "big"))
            sock.sendall(payload)
            buf = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
            if buf:
                reply = json.loads(buf.decode())
                info["answered"] = True
                info["protocol_version"] = reply.get("protocol_version", "")
                info["status"] = reply.get("status", "")
        except Exception as exc:
            info["error"] = type(exc).__name__ + ": " + str(exc)
        finally:
            sock.close()
        return info

    def port_holders():
        # Diagnostics for a stop that left the port serving: who still
        # holds it, and whether the two tools the kill path depends on
        # are installed at all. Best effort, and only worth paying for
        # on the failing branch.
        import shutil
        import subprocess
        info = {"lsof": shutil.which("lsof") or "",
                "pkill": shutil.which("pkill") or "",
                "pids": []}
        if info["lsof"]:
            try:
                proc = subprocess.run(
                    [info["lsof"], "-ti", "tcp:" + str(SERVER_PORT)],
                    capture_output=True, timeout=15.0,
                )
                info["pids"] = proc.stdout.decode(errors="replace").split()
            except Exception as exc:
                info["pids"] = [type(exc).__name__ + ": " + str(exc)]
        return info

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=PROJECT_NAME, timeout=30.0)
    dh.log("connected on port " + str(SERVER_PORT))

    # A: the server answers a real query. Established on the socket, so
    # the rest of the scenario measures a server that was demonstrably
    # up rather than one the engine merely called RUNNING.
    before = probe_port()
    dh.record(
        "A_server_answers_before_stop",
        (before["answered"]
         and before["protocol_version"] == protocol.PROTOCOL_VERSION),
        {"probe": before, "expected_protocol": protocol.PROTOCOL_VERSION},
    )

    # B: drive Stop the way the button does, and let the request settle.
    # tick() only: a PollTick here would query the port this scenario is
    # about to read, and the heartbeat branch would flip server back to
    # RUNNING off a server that is supposed to be gone.
    dh.com.stop_server()
    settled = False
    deadline = time.time() + STOP_SETTLE_TIMEOUT_S
    while time.time() < deadline:
        dh.facade.tick()
        if dh.facade.engine.state.server.name == "UNKNOWN":
            settled = True
            break
        time.sleep(0.2)
    state_after = dh.facade.engine.state
    dh.record(
        "B_stop_request_settled",
        settled,
        {"server": state_after.server.name,
         "phase": state_after.phase.name,
         "server_error": getattr(state_after, "server_error", "")},
    )
    dh.log("stop settled=" + str(settled)
           + " server=" + state_after.server.name)

    # C: the verdict. The listen socket has to be gone.
    started_at = time.time()
    deadline = started_at + PORT_DEATH_DEADLINE_S
    after = probe_port()
    while after["connected"] and time.time() < deadline:
        time.sleep(0.5)
        after = probe_port()
    details = {"probe": after,
               "waited_s": round(time.time() - started_at, 2)}
    if after["connected"]:
        details["holders"] = port_holders()
    dh.record("C_port_closed_after_stop", not after["connected"], details)
    dh.log("port connected after stop=" + str(after["connected"]))

except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
'''


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PORT_DEATH_DEADLINE_S>>", repr(_PORT_DEATH_DEADLINE_S))
    )


def _probe_port(ctx: r.ScenarioContext) -> dict:
    """Ask the worker's port, from the orchestrator process, whether it
    is still serving.

    ``connected`` is the load-bearing field: a listener exists only while
    the process holding it does. ``answered`` says the listener is a
    ppf-cts-server that still speaks the wire, which separates "the stop
    killed nothing" from "something else is squatting the port". Both
    sockets are closed before returning.
    """
    info: dict = {"connected": False, "answered": False,
                  "status": "", "error": ""}
    try:
        sock = socket.create_connection((ctx.host, ctx.server_port),
                                        timeout=3.0)
    except OSError as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    sock.close()
    info["connected"] = True
    try:
        reply = r.ProtoClient(ctx.host, ctx.server_port,
                              timeout=5.0).text_cmd({"name": ctx.project_name})
        info["answered"] = True
        info["status"] = reply.get("status", "")
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _wait_for_port_closed(ctx: r.ScenarioContext, *,
                          deadline_s: float) -> dict:
    """Poll ``_probe_port`` until the port refuses, or the deadline."""
    deadline = time.monotonic() + deadline_s
    info = _probe_port(ctx)
    while info["connected"] and time.monotonic() < deadline:
        time.sleep(0.5)
        info = _probe_port(ctx)
    return info


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 150.0))
    if err is not None:
        return err

    checks = dict(result.get("checks", {}))
    # D: the same fact, observed from outside Blender. The driver's own
    # verdict runs inside the process whose addon issued the stop; this
    # one does not, so a driver that mis-reads its socket cannot carry
    # the scenario green.
    probe = _wait_for_port_closed(ctx, deadline_s=_PORT_DEATH_DEADLINE_S)
    checks["D_port_closed_from_host"] = {
        "ok": not probe["connected"],
        "details": {"probe": probe, "port": ctx.server_port},
    }
    return r.report_named_checks(checks)
