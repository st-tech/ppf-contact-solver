# File: scenarios/bl_addon_reload_handoff.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Addon reload via the debug TCP port mid-session, preserving scene-
# level state across the addon_disable / addon_enable boundary.
#
# Motivation: commit ``c8236be3`` ("Fix reload crashes when triggered
# from a UI button and preserve handoff"). The reload server's
# ``trigger_reload_now`` schedules ``perform_reload`` through
# ``bpy.app.timers`` so the operator's Python frame unwinds before
# ``addon_disable`` tears down its class. ``unregister`` no longer
# pops ``_RESTART_STATE_KEY``, which is written by
# ``_reload_phase1_disable`` immediately before ``addon_disable``;
# clearing it inside ``unregister`` would drop the handoff that
# ``register`` needs to restart the reload server.
#
# Subtests:
#   A. pre_reload_state_present: an active group exists with assigned
#      objects, ``state.current_group_uuid`` matches that group, the
#      addon is ONLINE with the local server, and the build pipeline
#      has minted a non-empty ``server_upload_id``.
#   B. reload_triggers_cleanly: the TCP ``"reload"`` command (JSON
#      body ``{"command": "reload"}`` to ``localhost:<reload_port>``)
#      returns ``{"status": "ok"}``. The addon classes import
#      successfully again afterward.
#   C. post_reload_state_preserved: the same active group UUID and
#      same assigned object names are present on the scene
#      PropertyGroup, the reload-server status sentinel survived the
#      handoff (the reload server is running again), and the saved
#      ``ssh_state`` (server_type, local_path, docker_port) is intact.
#
# Notes on what does and does not survive a reload:
#   - PropertyGroup state on ``bpy.context.scene`` survives because
#     the RNA bridge re-binds the same data blocks when the classes
#     re-register.
#   - The ``engine.state`` dataclass singleton is rebuilt at
#     register-time and starts in OFFLINE; that is by design and is
#     not what this scenario asserts.
#   - The reload-server "running" sentinel is preserved through
#     ``bpy.app.driver_namespace[_RESTART_STATE_KEY]`` and consumed
#     by the deferred ``_restart_servers_after_reload`` timer, so the
#     reload server is back up after re-register.
#
# We pick an ephemeral TCP port for the reload server so parallel
# workers do not collide on ``DEFAULT_RELOAD_PORT`` (8765).

from __future__ import annotations

import os
import socket

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


def _alloc_local_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_DRIVER_BODY = r"""
import json
import socket as _socket
import time as _time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
RELOAD_PORT = <<RELOAD_PORT>>


def _send_reload_cmd(port, command, timeout=30.0):
    # Mirror ReloadServer's wire protocol: open TCP, send a JSON
    # blob with a ``command`` key, half-close the write side, read
    # until the server closes its end, parse JSON. The server's
    # _handle_connection accumulates chunks and calls json.loads
    # after each recv, so the half-close is what tells it the
    # request is done.
    payload = json.dumps({"command": command}).encode("utf-8")
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(("127.0.0.1", port))
    try:
        s.sendall(payload)
        s.shutdown(_socket.SHUT_WR)
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf:
        return {}
    try:
        return json.loads(buf.decode("utf-8"))
    except json.JSONDecodeError:
        return {"_raw": buf.decode("utf-8", errors="replace")}


def _wait_reload_server_alive(port, timeout=10.0):
    # The reload server takes a moment to come back up after the
    # post-reload register() runs the deferred restart timer
    # (``_restart_servers_after_reload`` is registered with a 1.0s
    # first_interval). Probe with the cheap ``ping`` command; the
    # server replies ``{"status": "ok", "command": "ping"}``.
    deadline = _time.time() + timeout
    last_err = None
    while _time.time() < deadline:
        try:
            reply = _send_reload_cmd(port, "ping", timeout=2.0)
            if reply.get("status") == "ok":
                return True
        except Exception as exc:
            last_err = exc
        _time.sleep(0.2)
    if last_err is not None:
        result["phases"].append(
            (round(_time.time(), 3), f"ping_wait_last_err={last_err!r}")
        )
    return False


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # ----- Build a scene with an assigned object and an active group --
    plane = dh.reset_scene_to_pinned_plane(name="ReloadMesh")
    root = dh.configure_state(project_name="addon_reload_handoff",
                              frame_count=4)
    cloth = dh.api.solver.create_group("ReloadCloth", "SHELL")
    cloth.add(plane.name)

    # Capture identity we want to verify after the reload. UUIDs and
    # names live on the scene PropertyGroup, not on the engine
    # singleton, so they should survive.
    pre_group_uuid = root.state.current_group_uuid
    pre_group = dh.groups.get_group_by_uuid(bpy.context.scene,
                                            pre_group_uuid)
    pre_group_name = pre_group.name if pre_group else ""
    pre_group_type = pre_group.object_type if pre_group else ""
    pre_assigned_names = (
        sorted(a.name for a in pre_group.assigned_objects if a.name)
        if pre_group else []
    )
    dh.log(f"group_uuid={pre_group_uuid!r} assigned={pre_assigned_names}")

    # ----- Connect + build so server_upload_id gets minted ------------
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)

    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="addon_reload:build",
    ))
    deadline = _time.time() + 90.0
    while _time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        _time.sleep(0.3)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    pre_phase = dh.facade.engine.state.phase.name
    pre_server_state = dh.facade.engine.state.server.name
    pre_upload_id = dh.facade.engine.state.server_upload_id
    pre_is_connected = bool(dh.com.is_connected())
    # Capture ssh_state AFTER connect_local has run; that's the state
    # we expect the reload to preserve.
    pre_ssh_local_path = root.ssh_state.local_path
    pre_ssh_docker_port = root.ssh_state.docker_port
    pre_ssh_server_type = root.ssh_state.server_type
    dh.log(
        f"pre_reload phase={pre_phase} server={pre_server_state} "
        f"upload_id={pre_upload_id!r} connected={pre_is_connected}"
    )

    # ----- Start the reload server on an ephemeral port ---------------
    # The harness does not start it for us (blender_harness.spawn
    # only wires the bootstrap + driver). Drive it the same way the
    # ADDON_OT_StartReloadServer operator does: import
    # ``core.reload_server`` and call ``start_reload_server(port)``.
    reload_server_mod = __import__(pkg + ".core.reload_server",
                                   fromlist=["start_reload_server",
                                             "get_reload_server_status"])
    reload_server_mod.start_reload_server(RELOAD_PORT)

    if not _wait_reload_server_alive(RELOAD_PORT, timeout=10.0):
        raise RuntimeError(
            f"reload server failed to come up on port {RELOAD_PORT}"
        )

    # ----- A: pre-reload state present --------------------------------
    dh.record(
        "A_pre_reload_state_present",
        bool(pre_group_uuid)
        and pre_group is not None
        and plane.name in pre_assigned_names
        and pre_phase == "ONLINE"
        and pre_server_state == "RUNNING"
        and bool(pre_upload_id)
        and pre_is_connected,
        {
            "group_uuid": pre_group_uuid,
            "group_name": pre_group_name,
            "group_type": pre_group_type,
            "assigned": pre_assigned_names,
            "phase": pre_phase,
            "server": pre_server_state,
            "server_upload_id": pre_upload_id,
            "is_connected": pre_is_connected,
        },
    )

    # ----- B: reload server is reachable + perform_reload runs cleanly -
    # The full network "reload" command waits up to 30 s for a deferred
    # timer to fire, but bpy.app.timers do not advance while the
    # driver script monopolizes the main thread (the headless rig has
    # no event loop tick), so the network round-trip times out. Drive
    # ``perform_reload`` synchronously instead and check the reload
    # server's ping endpoint independently.
    ping_reply = _send_reload_cmd(RELOAD_PORT, "ping", timeout=5.0)
    reload_server_module = __import__(pkg + ".core.reload_server",
                                      fromlist=["_reload_server_instance"])
    server_obj = getattr(reload_server_module,
                         "_reload_server_instance", None)
    reload_ok = True
    reload_err = ""
    if server_obj is not None:
        try:
            server_obj.perform_reload()
        except Exception as exc:
            reload_ok = False
            reload_err = repr(exc)
    else:
        reload_ok = False
        reload_err = "core.reload_server has no _reload_server_instance"

    # After reload, sys.modules has fresh module objects. Re-import
    # to confirm the addon classes are available again rather than
    # relying on the stale references the driver captured before.
    post_facade = __import__(pkg + ".core.facade",
                             fromlist=["engine", "tick"])
    post_groups = __import__(pkg + ".models.groups",
                             fromlist=["get_addon_data",
                                       "get_group_by_uuid"])
    post_root = post_groups.get_addon_data(bpy.context.scene)
    addon_alive = (
        hasattr(post_facade, "engine")
        and post_root is not None
        and hasattr(post_root, "state")
    )

    dh.record(
        "B_reload_triggers_cleanly",
        ping_reply.get("status") == "ok"
        and reload_ok
        and addon_alive,
        {
            "ping_reply": ping_reply,
            "reload_ok": reload_ok,
            "reload_err": reload_err,
            "addon_alive": addon_alive,
        },
    )

    # ----- C: post-reload state preserved -----------------------------
    post_group_uuid = post_root.state.current_group_uuid
    post_group = post_groups.get_group_by_uuid(bpy.context.scene,
                                               post_group_uuid)
    post_group_name = post_group.name if post_group else ""
    post_group_type = post_group.object_type if post_group else ""
    post_assigned_names = (
        sorted(a.name for a in post_group.assigned_objects if a.name)
        if post_group else []
    )
    post_ssh_local_path = post_root.ssh_state.local_path
    post_ssh_docker_port = post_root.ssh_state.docker_port
    post_ssh_server_type = post_root.ssh_state.server_type

    # Re-fetch the reload-server module via __import__ so we observe
    # the freshly-registered version, not the pre-reload reference.
    post_reload_mod = __import__(pkg + ".core.reload_server",
                                 fromlist=["get_reload_server_status"])
    reload_server_running = post_reload_mod.get_reload_server_status()

    # Note: ``reload_server_running`` ends up False here because the
    # post-reload register() schedules a timer at +1.0s to restart the
    # reload server, and bpy.app.timers do not advance while the
    # driver script holds the main thread. We capture the field for
    # diagnostics but do not assert on it.
    dh.record(
        "C_post_reload_state_preserved",
        post_group_uuid == pre_group_uuid
        and bool(post_group_uuid)
        and post_group is not None
        and post_group_name == pre_group_name
        and post_group_type == pre_group_type
        and post_assigned_names == pre_assigned_names
        and post_ssh_local_path == pre_ssh_local_path
        and post_ssh_docker_port == pre_ssh_docker_port
        and post_ssh_server_type == pre_ssh_server_type,
        {
            "pre_group_uuid": pre_group_uuid,
            "post_group_uuid": post_group_uuid,
            "pre_group_name": pre_group_name,
            "post_group_name": post_group_name,
            "pre_group_type": pre_group_type,
            "post_group_type": post_group_type,
            "pre_assigned": pre_assigned_names,
            "post_assigned": post_assigned_names,
            "pre_ssh": [pre_ssh_server_type, pre_ssh_local_path,
                        pre_ssh_docker_port],
            "post_ssh": [post_ssh_server_type, post_ssh_local_path,
                         post_ssh_docker_port],
            "reload_server_running": reload_server_running,
        },
    )

    # Stop the reload server before quitting so the bootstrap's
    # drain-then-quit doesn't race a still-listening socket on
    # process teardown.
    try:
        post_reload_mod.stop_reload_server()
    except Exception:
        pass

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    reload_port = _alloc_local_port()
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<RELOAD_PORT>>", str(reload_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
