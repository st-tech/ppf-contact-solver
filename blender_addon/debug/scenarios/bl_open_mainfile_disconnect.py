# File: scenarios/bl_open_mainfile_disconnect.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Companion to bl_load_disconnect: that scenario only exercises a
# single ``wm.read_homefile`` (File > New) and only fires load_pre
# once. The user-reported bug surfaces on the *second* file load:
# ``_disconnect_on_load`` is appended to ``bpy.app.handlers.load_pre``
# without the ``@bpy.app.handlers.persistent`` decorator, so Blender
# wipes it after the first file load. The next File > Open then
# silently inherits the previous project's live socket and ONLINE
# phase.
#
# Reproduction: connect, open .blend (first time), reconnect, open
# .blend (second time). The first open should disconnect; the second
# should also disconnect, but doesn't.
#
# Subtests:
#   A. connected_before_first_open: sanity: connect succeeds.
#   B. offline_after_first_open: load_pre handler still alive; OK.
#   C. connected_before_second_open: sanity: reconnect succeeds.
#   D. offline_after_second_open: this is the bug. Handler was
#      wiped by the first load, so the disconnect never runs and the
#      addon stays ONLINE pointing at a stale socket.

from __future__ import annotations

import os

from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, os, time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details):
    result["checks"][name] = {"ok": bool(ok), "details": details}


def connect_and_wait(com, root, facade, events_mod, timeout=30.0):
    com.set_project_name(root.state.project_name)
    com.connect_local(root.ssh_state.local_path,
                      server_port=root.ssh_state.docker_port)
    deadline = time.time() + timeout
    while time.time() < deadline:
        facade.engine.dispatch(events_mod.PollTick())
        facade.tick()
        s = facade.engine.state
        if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
            return True
        time.sleep(0.2)
    return False


try:
    facade = __import__(pkg + ".core.facade", fromlist=["engine", "tick"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_addon_data"])
    events_mod = __import__(pkg + ".core.events", fromlist=["PollTick"])
    com = client.communicator

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "open_target.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    log(f"saved blend at {blend_path}")

    root = groups_mod.get_addon_data(bpy.context.scene)
    root.state.project_name = "open_mainfile_disconnect"
    root.ssh_state.server_type = "LOCAL"
    root.ssh_state.local_path = LOCAL_PATH
    root.ssh_state.docker_port = SERVER_PORT

    # ----- First open ---------------------------------------------
    ok = connect_and_wait(com, root, facade, events_mod)
    record(
        "A_connected_before_first_open",
        ok and bool(com.is_connected()) and facade.runner._backend is not None,
        {
            "phase": facade.engine.state.phase.name,
            "server": facade.engine.state.server.name,
            "is_connected": bool(com.is_connected()),
            "backend_alive": facade.runner._backend is not None,
            "load_pre_handlers": [
                getattr(h, "__name__", str(h))
                for h in bpy.app.handlers.load_pre
            ],
        },
    )
    log("first connect ok")

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    for _ in range(3):
        facade.tick()
        time.sleep(0.05)
    log(f"after_first_open phase={facade.engine.state.phase.name}")
    record(
        "B_offline_after_first_open",
        facade.engine.state.phase.name == "OFFLINE"
        and not com.is_connected()
        and facade.runner._backend is None,
        {
            "phase": facade.engine.state.phase.name,
            "is_connected": bool(com.is_connected()),
            "backend_is_none": facade.runner._backend is None,
            "load_pre_handlers_after_open": [
                getattr(h, "__name__", str(h))
                for h in bpy.app.handlers.load_pre
            ],
        },
    )

    # ----- Second open: re-establish state and try again ----------
    # The opened .blend just deserialized fresh scene props, so
    # re-set the addon connection knobs and project name on the new
    # scene before reconnecting.
    root = groups_mod.get_addon_data(bpy.context.scene)
    root.state.project_name = "open_mainfile_disconnect"
    root.ssh_state.server_type = "LOCAL"
    root.ssh_state.local_path = LOCAL_PATH
    root.ssh_state.docker_port = SERVER_PORT

    ok2 = connect_and_wait(com, root, facade, events_mod)
    record(
        "C_connected_before_second_open",
        ok2 and bool(com.is_connected())
        and facade.runner._backend is not None,
        {
            "phase": facade.engine.state.phase.name,
            "server": facade.engine.state.server.name,
            "is_connected": bool(com.is_connected()),
            "backend_alive": facade.runner._backend is not None,
            "load_pre_handlers": [
                getattr(h, "__name__", str(h))
                for h in bpy.app.handlers.load_pre
            ],
        },
    )
    log("second connect ok")

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    for _ in range(3):
        facade.tick()
        time.sleep(0.05)
    log(f"after_second_open phase={facade.engine.state.phase.name}")
    record(
        "D_offline_after_second_open",
        facade.engine.state.phase.name == "OFFLINE"
        and not com.is_connected()
        and facade.runner._backend is None,
        {
            "phase": facade.engine.state.phase.name,
            "is_connected": bool(com.is_connected()),
            "backend_is_none": facade.runner._backend is None,
            "load_pre_handlers_after_open": [
                getattr(h, "__name__", str(h))
                for h in bpy.app.handlers.load_pre
            ],
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
