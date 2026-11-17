# File: scenarios/bl_protocol_handshake.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The add-on and the solver server speak the same wire protocol, and the
# add-on accepts the connection because of it.
#
# `blender_addon/protocol_version.toml` is the one source of the number: the
# add-on reads it at runtime (core/protocol.py) and the server bakes it in at
# build time (crates/ppf-cts-server/build.rs). A bump that reached one half and
# not the other, or a server binary left over from before a bump, would stop
# every connection at the handshake, which is exactly the point of the check.
# So this scenario asserts each link of the chain separately, against the
# rig's own server:
#
#   A_addon_reads_the_toml          the add-on's PROTOCOL_VERSION is the toml's.
#   B_server_reports_the_same       a raw TCMD ping to the rig server answers
#                                   with that same protocol_version, so the
#                                   running binary was built from this toml.
#   C_connection_is_accepted        a real connect reaches ONLINE with the
#                                   server RUNNING, version_ok set and no
#                                   error, so the add-on can operate.
#   D_mismatch_would_be_refused     the add-on's response interpreter, fed the
#                                   server's own status with the version
#                                   altered, refuses it naming both versions
#                                   and asks to stop that server. Interpreted
#                                   only: the effect is not executed, so the
#                                   rig's server keeps running. Without it, A
#                                   to C would also pass if the check were gone.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import os
import time
import traceback
import tomllib

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>

try:
    dh = DriverHelpers(pkg, result)
    protocol = __import__(pkg + ".core.protocol", fromlist=["PROTOCOL_VERSION"])
    connection = __import__(pkg + ".core.connection",
                            fromlist=["_query_ppf_cts_server"])
    transitions = __import__(pkg + ".core.transitions",
                             fromlist=["_interpret_response"])
    effects = __import__(pkg + ".core.effects", fromlist=["DoStopServer"])
    addon_version = str(protocol.PROTOCOL_VERSION)

    # A: the add-on's constant is the toml's.
    toml_path = os.path.join(LOCAL_PATH, "blender_addon", "protocol_version.toml")
    with open(toml_path, "rb") as f:
        toml_version = str(tomllib.load(f)["protocol"])
    dh.record("A_addon_reads_the_toml", addon_version == toml_version,
              {"addon": addon_version, "toml": toml_version, "path": toml_path})

    # B: the running server's own answer.
    status = None
    deadline = time.time() + 30.0
    while status is None and time.time() < deadline:
        status = connection._query_ppf_cts_server(SERVER_PORT, timeout=3.0)
        if status is None:
            time.sleep(0.5)
    server_version = None if status is None else str(status.get("protocol_version"))
    dh.record("B_server_reports_the_same",
              server_version == addon_version,
              {"server": server_version, "addon": addon_version,
               "port": SERVER_PORT, "answered": status is not None})

    # C: the add-on connects and accepts the server.
    dh.reset_scene_to_pinned_plane(name="HandshakeMesh")
    root = dh.configure_state(project_name="protocol_handshake", frame_count=2)
    err = ""
    try:
        dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                   project_name=root.state.project_name)
    except Exception as exc:
        err = str(exc)
    state = None
    deadline = time.time() + 30.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        state = dh.facade.engine.state
        if state.can_operate:
            break
        time.sleep(0.2)
    dh.record("C_connection_is_accepted",
              err == "" and state is not None and state.can_operate
              and state.version_ok and not state.error,
              {"connect_error": err[:300],
               "phase": None if state is None else state.phase.name,
               "server": None if state is None else state.server.name,
               "version_ok": None if state is None else state.version_ok,
               "error": None if state is None else state.error})

    # D: the same status with another version is refused, by interpretation.
    refused = {}
    if status is not None and state is not None:
        stale = dict(status)
        stale["protocol_version"] = "0.0-stale"
        new_state, out = transitions._interpret_response(state, stale)
        refused = {
            "version_ok": new_state.version_ok,
            "error": new_state.error,
            "stops_server": any(isinstance(e, effects.DoStopServer) for e in out),
        }
    dh.record("D_mismatch_would_be_refused",
              bool(refused) and refused["version_ok"] is False
              and refused["stops_server"]
              and "0.0-stale" in (refused["error"] or "")
              and addon_version in (refused["error"] or ""),
              refused)
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
