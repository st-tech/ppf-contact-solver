# File: scenarios/bl_mcp_connection_surface.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP connection and solver-GPU tools, driven while not connected.
#
# Not connected is the state a headless rig can reach, and it is also the
# state every session starts in, so it is the one the tools are asked about
# most. What matters here is that each tool answers about it rather than
# raising, refusing when it has no source of truth, and reporting what the
# add-on actually holds. The GPU list in particular is a cache filled from
# the solver host, so with no host it must report "not probed" rather than an
# empty machine.
#
# Assertions:
#   A. ``connection_info_reports_not_connected`` -- get_connection_info
#      answers with its four sections, and connection_status says connected
#      false, server_running false and type "unknown".
#   B. ``solver_gpus_report_no_probe`` -- list_solver_gpus answers with no
#      connection, reporting probed false, an empty device list and no probe
#      error, rather than raising or naming a device.
#   C. ``refresh_solver_gpus_is_refused`` -- refresh_solver_gpus reads the
#      list from the solver host, so with no connection it refuses with
#      isError and a message telling the caller to connect first, and leaves
#      the cache unprobed.
#   D. ``clearing_the_gpu_selection_reports_none`` -- set_solver_gpu with
#      neither uuid nor index clears the selection, and list_solver_gpus then
#      names no selected device and no selected uuid.
#   E. ``gpu_index_round_trips`` -- set_solver_gpu with an index reports that
#      index back, and list_solver_gpus reads the same one from the scene.
#   F. ``connect_docker_accepts_a_port`` -- the tool's inputSchema carries an
#      integer "port" property with a default, so a caller can place the
#      container's solver port, and container and path stay required.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives the addon's in-process MCP server and never asks the solver to
# step, so nothing in it is backend-specific. A rig run against a CPU build
# passed it, and that run is the evidence this line rests on.
BACKENDS = ("real",)

NEEDS_BLENDER = True

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    gpu_devices = __import__(pkg + ".core.gpu_devices", fromlist=["has_probed"])
    props = groups.get_addon_data(bpy.context.scene).ssh_state
    saved_index = props.solver_gpu_index
    saved_uuid = props.solver_gpu_uuid

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. the connection surface, with nothing connected ------
    info, _info_raw = mcp_tool(pkg, url, "get_connection_info", request_id=1)
    status = info.get("connection_status") or {}
    project = info.get("project_info") or {}
    mcp_check(
        result, "A_connection_info_reports_not_connected",
        info.get("status") == "success"
        and status.get("connected") is False
        and status.get("server_running") is False
        and status.get("type") == "unknown"
        and isinstance(status.get("status"), str)
        and isinstance(info.get("ssh_config"), dict)
        and isinstance(info.get("docker_config"), dict)
        and isinstance(project.get("project_name"), str),
        {
            "status": info.get("status"),
            "connection_status": status,
            "project_info": project,
            "ssh_config_type": type(info.get("ssh_config")).__name__,
            "docker_config_type": type(info.get("docker_config")).__name__,
        },
    )

    # ----- B. the GPU list before any host has been read ----------
    gpus, _gpus_raw = mcp_tool(pkg, url, "list_solver_gpus", request_id=2)
    mcp_check(
        result, "B_solver_gpus_report_no_probe",
        gpus.get("status") == "success"
        and gpus.get("probed") is False
        and gpus.get("devices") == []
        and gpus.get("probe_error") is None
        and gpus.get("selected_name") is None,
        {"payload": gpus},
    )

    # ----- C. refreshing needs a host to read from ----------------
    refused, refused_raw = mcp_tool(pkg, url, "refresh_solver_gpus", request_id=3)
    refusal = (refused.get("message") or "").lower()
    mcp_check(
        result, "C_refresh_solver_gpus_is_refused",
        refused.get("status") == "error"
        and refused_raw.get("isError") is True
        and "not connected" in refusal
        and "connection" in refusal
        and gpu_devices.has_probed() is False
        and gpu_devices.cached_gpu_devices() == [],
        {
            "payload": refused,
            "isError": refused_raw.get("isError"),
            "probed_after": gpu_devices.has_probed(),
            "devices_after": len(gpu_devices.cached_gpu_devices()),
        },
    )

    # ----- D. clearing the selection ------------------------------
    cleared, _cleared_raw = mcp_tool(pkg, url, "set_solver_gpu", {}, request_id=4)
    after_clear, _after_clear_raw = mcp_tool(
        pkg, url, "list_solver_gpus", request_id=5
    )
    mcp_check(
        result, "D_clearing_the_gpu_selection_reports_none",
        cleared.get("status") == "success"
        and "automatic" in (cleared.get("message") or "").lower()
        and after_clear.get("status") == "success"
        and after_clear.get("selected_uuid") is None
        and after_clear.get("selected_name") is None,
        {
            "cleared": cleared,
            "selected_index": after_clear.get("selected_index"),
            "selected_uuid": after_clear.get("selected_uuid"),
            "selected_name": after_clear.get("selected_name"),
        },
    )

    # ----- E. an index survives the round trip through the scene --
    picked, _picked_raw = mcp_tool(
        pkg, url, "set_solver_gpu", {"index": 1}, request_id=6
    )
    after_pick, _after_pick_raw = mcp_tool(
        pkg, url, "list_solver_gpus", request_id=7
    )
    mcp_check(
        result, "E_gpu_index_round_trips",
        picked.get("status") == "success"
        and picked.get("selected_index") == 1
        and picked.get("selected_uuid") is None
        and after_pick.get("selected_index") == 1
        and after_pick.get("selected_uuid") is None
        and props.solver_gpu_index == 1,
        {
            "set": picked,
            "listed_index": after_pick.get("selected_index"),
            "listed_uuid": after_pick.get("selected_uuid"),
            "scene_index": props.solver_gpu_index,
        },
    )

    # ----- F. the container port is reachable from the tool -------
    env, _tools_resp = mcp_call(pkg, url, "tools/list", request_id=8)
    tools = ((env.get("result") or {}).get("tools")) or []
    by_name = {}
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            by_name[tool["name"]] = tool
    schema = (by_name.get("connect_docker") or {}).get("inputSchema") or {}
    port_schema = (schema.get("properties") or {}).get("port") or {}
    required = schema.get("required") or []
    mcp_check(
        result, "F_connect_docker_accepts_a_port",
        port_schema.get("type") == "integer"
        and isinstance(port_schema.get("default"), int)
        and "port" not in required
        and {"container", "path"} <= set(required),
        {
            "connect_docker_listed": "connect_docker" in by_name,
            "port_schema": port_schema,
            "required": required,
            "tool_count": len(by_name),
        },
    )

    # Leave the scene's GPU selection as it was found: the checks above
    # wrote to it, and a later scenario in this Blender would read it.
    props.solver_gpu_index = saved_index
    props.solver_gpu_uuid = saved_uuid

    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
