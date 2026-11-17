# File: scenarios/bl_servers_answer_outside_calls.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The Debug Server and the MCP Server start from their buttons and Blender
# answers calls made FROM ANOTHER PROCESS, which is how both are used: an agent
# or an IDE outside Blender drives the add-on through them.
#
# Why this is its own scenario: every other server check calls from inside
# Blender's own process while the driver holds the main thread, so it can only
# see a listener's immediate reply (`bl_local_servers_start` E and H) or has to
# pump the MCP task queue by hand (`bl_mcp_roundtrip`). The real path is
# different: a request arrives from outside and the server hands it to a
# main-thread timer, which runs only while Blender's event loop is free. So the
# driver here starts both servers, launches a separate client process, and
# returns the main thread through `DEFERRED` (see blender_harness) while the
# client calls, exactly as an outside caller would meet it. Headless Blender
# (the Windows leg) has no event loop, so there the driver fires the queued
# main-thread callbacks itself; the phases name which path ran.
#
#   A_debug_server_starts           Start (addon.start_reload_server) FINISHED.
#   B_mcp_server_starts             Start (mcp.start_server) FINISHED.
#   C_debug_ping_answered           the client's `ping` gets status ok.
#   D_debug_execute_runs_in_blender the client's `execute` runs Python in
#                                   Blender: its printed output comes back and
#                                   the scene property it set is there after.
#   E_mcp_tools_list_answered       the client's `tools/list` lists tools.
#   F_mcp_tool_call_changes_blender the client's `create_group` tools/call
#                                   returns a complete result and Blender holds
#                                   one more group afterwards.
#   G_servers_stop                  both Stop buttons FINISHED, neither running.

from __future__ import annotations

from . import _driver_lib as dl
from . import _mcp_lib as ml
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import json
import os
import socket
import subprocess
import sys
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

CLIENT = r'''
import json, socket, sys, urllib.request

spec = json.load(open(sys.argv[1]))
out = {}


def debug(packet):
    with socket.create_connection(("127.0.0.1", spec["debug_port"]), timeout=60) as c:
        c.sendall(json.dumps(packet).encode())
        c.shutdown(socket.SHUT_WR)
        buf = b""
        while True:
            chunk = c.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf) if buf else {}


def mcp(request):
    req = urllib.request.Request(spec["mcp_url"], data=json.dumps(request["body"]).encode(),
                                 method="POST")
    for k, v in request["headers"].items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode()
        if "text/event-stream" in resp.headers.get("Content-Type", ""):
            frames = [json.loads(line[5:]) for line in text.splitlines()
                      if line.startswith("data:") and line[5:].strip()]
            frames = [f for f in frames if "id" in f]
            return frames[-1] if frames else {}
        return json.loads(text)


for name, call in (("ping", lambda: debug({"command": "ping"})),
                   ("execute", lambda: debug(spec["execute"])),
                   ("tools_list", lambda: mcp(spec["tools_list"])),
                   ("create_group", lambda: mcp(spec["create_group"]))):
    try:
        out[name] = call()
    except Exception as exc:
        out[name] = {"client_error": repr(exc)}
json.dump(out, open(spec["out"], "w"))
'''


def check(name, ok, details):
    result["checks"][name] = {"ok": bool(ok), "details": details}


def op(fn):
    try:
        return set(fn()), ""
    except RuntimeError as exc:
        return {"CANCELLED"}, str(exc)


try:
    dh = DriverHelpers(pkg, result)
    rs = __import__(pkg + ".core.reload_server",
                    fromlist=["get_reload_server_status", "stop_reload_server"])
    mcp_mod = __import__(pkg + ".mcp.mcp_server",
                         fromlist=["is_mcp_running", "get_mcp_server"])
    scene = bpy.context.scene
    state = dh.groups.get_addon_data(scene).state

    rs.stop_reload_server()
    debug_port = mcp_alloc_free_port()
    state.reload_port = debug_port
    outcome, text = op(bpy.ops.addon.start_reload_server)
    check("A_debug_server_starts",
          outcome == {"FINISHED"} and bool(rs.get_reload_server_status()),
          {"outcome": sorted(outcome), "port": debug_port, "error": text[-300:]})

    state.mcp_port = mcp_alloc_free_port()
    outcome, text = op(bpy.ops.mcp.start_server)
    mcp_port = getattr(mcp_mod.get_mcp_server(), "port", None)
    check("B_mcp_server_starts",
          outcome == {"FINISHED"} and bool(mcp_mod.is_mcp_running()) and mcp_port,
          {"outcome": sorted(outcome), "port": mcp_port, "error": text[-300:]})

    groups_before = len(list(dh.groups.iterate_active_object_groups(scene)))
    work = os.path.join(PROBE_DIR or os.path.dirname(RESULT_PATH), "outside_calls")
    os.makedirs(work, exist_ok=True)
    out_path = os.path.join(work, "client_out.json")
    tools_list = mcp_build("tools/list", {}, request_id=11)
    create = mcp_build("tools/call", {"name": "create_group", "arguments": {}},
                       request_id=12)
    spec = {
        "debug_port": debug_port,
        "mcp_url": "http://127.0.0.1:%d/mcp" % (mcp_port or 0),
        "execute": {"command": "execute", "timeout": 30.0, "code":
                    "bpy.context.scene['outside_call'] = 42\n"
                    "print('scene', bpy.context.scene.name)"},
        "tools_list": {"body": tools_list[0], "headers": tools_list[1]},
        "create_group": {"body": create[0], "headers": create[1]},
        "out": out_path,
    }
    spec_path = os.path.join(work, "client_spec.json")
    client_path = os.path.join(work, "client.py")
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    with open(client_path, "w") as f:
        f.write(CLIENT)
    # Blender's own interpreter, isolated: a separate OS process, with no
    # add-on and no bpy, which is all an outside caller has.
    client = subprocess.Popen([sys.executable, "-I", client_path, spec_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    started = time.time()

    def finish():
        answers = {}
        if os.path.isfile(out_path):
            with open(out_path) as f:
                answers = json.load(f)
        log = client.stdout.read().decode(errors="replace")[-600:] if client.stdout else ""
        ping = answers.get("ping") or {}
        check("C_debug_ping_answered", ping.get("status") == "ok",
              {"reply": ping, "client_log": log})
        execute = answers.get("execute") or {}
        check("D_debug_execute_runs_in_blender",
              execute.get("status") == "ok"
              and ("scene %s" % scene.name) in (execute.get("output") or "")
              and scene.get("outside_call") == 42,
              {"reply": execute, "scene_property": scene.get("outside_call")})
        listed = answers.get("tools_list") or {}
        ok, why = mcp_envelope_ok(listed, 11)
        tools = (listed.get("result") or {}).get("tools") or []
        check("E_mcp_tools_list_answered", ok and len(tools) > 0,
              {"why": why, "tools": len(tools)})
        created = answers.get("create_group") or {}
        ok, why = mcp_envelope_ok(created, 12)
        groups_after = len(list(dh.groups.iterate_active_object_groups(scene)))
        check("F_mcp_tool_call_changes_blender",
              ok and groups_after == groups_before + 1,
              {"why": why, "groups_before": groups_before,
               "groups_after": groups_after,
               "reply": str(created)[:400]})
        out1, text1 = op(bpy.ops.addon.stop_reload_server)
        out2, text2 = op(bpy.ops.mcp.stop_server)
        check("G_servers_stop",
              out1 == {"FINISHED"} and out2 == {"FINISHED"}
              and not rs.get_reload_server_status() and not mcp_mod.is_mcp_running(),
              {"debug": sorted(out1), "mcp": sorted(out2),
               "errors": [text1[-200:], text2[-200:]]})

    def settle():
        if client.poll() is None:
            client.kill()
            result["errors"].append("the outside client did not finish in 150 s")
        finish()

    if bpy.app.background:
        # Headless (the Windows leg runs PPF_BLENDER_HEADLESS=1): Blender has
        # no event loop, so no timer ever fires and DEFERRED cannot run. The
        # driver fires the callbacks the two servers queued for the main
        # thread itself, which is the one step the event loop would have
        # taken; the client is still a separate process calling over the
        # network.
        result["phases"].append(["main thread pumped by the driver (headless)", 0])
        process_mcp_tasks = __import__(
            pkg + ".mcp.task_system", fromlist=["process_mcp_tasks"]).process_mcp_tasks
        while client.poll() is None and time.time() - started < 150.0:
            server = rs._reload_server_instance
            for callback in list(getattr(server, "_pending_timers", [])):
                callback()
            process_mcp_tasks()
            time.sleep(0.02)
        settle()
    else:
        result["phases"].append(["main thread returned to the event loop", 0])

        def DEFERRED():
            # Runs as a timer, so the event loop is free between calls and
            # the servers' main-thread handlers run while the client waits.
            if client.poll() is None and time.time() - started < 150.0:
                return 0.1
            settle()
            return None

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + ml.MCP_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
