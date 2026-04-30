# File: scenarios/bl_load_disconnect.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Loading a different .blend (or starting a new file) while the addon
# is connected must drop the connection. The communicator and engine
# are module-level singletons that survive ``wm.read_homefile`` /
# ``wm.read_factory_settings``, so without an explicit disconnect on
# load_pre the new scene inherits a stale "connected" state pointing
# at the previous project's server.
#
# Subtests:
#   A. connected_before_load: sanity: connect succeeds.
#   B. phase_offline_after_load: engine.state.phase reset to OFFLINE.
#   C. backend_torn_down_after_load: runner._backend is None and
#      runner.project_name is empty after the load.

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

    root = groups_mod.get_addon_data(bpy.context.scene)
    root.state.project_name = "load_disconnect"

    root.ssh_state.server_type = "LOCAL"
    root.ssh_state.local_path = LOCAL_PATH
    root.ssh_state.docker_port = SERVER_PORT
    com.set_project_name(root.state.project_name)
    com.connect_local(root.ssh_state.local_path,
                      server_port=root.ssh_state.docker_port)

    deadline = time.time() + 30.0
    while time.time() < deadline:
        facade.engine.dispatch(events_mod.PollTick())
        facade.tick()
        s = facade.engine.state
        if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
            break
        time.sleep(0.2)
    record(
        "A_connected_before_load",
        facade.engine.state.phase.name == "ONLINE"
        and facade.engine.state.server.name == "RUNNING"
        and bool(com.is_connected())
        and facade.runner._backend is not None,
        {
            "phase": facade.engine.state.phase.name,
            "server": facade.engine.state.server.name,
            "is_connected": bool(com.is_connected()),
            "backend_alive": facade.runner._backend is not None,
        },
    )
    log("connected")

    # ----- The act under test --------------------------------------
    # Replace the current scene with the factory startup file. This is
    # what "File > New > General" hits. Any persistent app singletons
    # (engine, runner) survive; the addon must drop its connection on
    # load_pre/post so the new scene starts clean.
    bpy.ops.wm.read_homefile(use_factory_startup=True)
    # Drain any queued events from disconnect transition.
    for _ in range(3):
        facade.tick()
        time.sleep(0.05)
    log(f"after_load phase={facade.engine.state.phase.name} "
        f"server={facade.engine.state.server.name}")

    record(
        "B_phase_offline_after_load",
        facade.engine.state.phase.name == "OFFLINE"
        and not com.is_connected(),
        {
            "phase": facade.engine.state.phase.name,
            "is_connected": bool(com.is_connected()),
        },
    )
    record(
        "C_backend_torn_down_after_load",
        facade.runner._backend is None
        and not facade.runner.project_name,
        {
            "backend_is_none": facade.runner._backend is None,
            "project_name": facade.runner.project_name,
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
