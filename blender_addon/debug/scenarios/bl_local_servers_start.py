# File: scenarios/bl_local_servers_start.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The panel's two local servers start from their Start buttons and answer.
#
# Both are driven through their OPERATORS, the buttons an artist presses, and
# each is judged by a request it must answer, never by its own flag. That is
# the gap this closes: the Add-on Local Debug Server caught its own bind
# failure and its Start button reported "Reload server started on port N"
# anyway, while nothing listened and "Reload Add-on Now" failed with "Reload
# server is not running". Seen on a host whose idle monitor holds the default
# 8765.
#
# Debug server (`addon.start_reload_server`, `addon.stop_reload_server`):
#   A_debug_reload_refused_while_stopped  "Reload Add-on Now" says the server
#                                         is not running, before any start.
#   B_debug_busy_port_is_refused          a port held here: Start returns
#                                         CANCELLED and nothing runs.
#   C_debug_refusal_names_the_port        and its error names that port.
#   D_debug_starts_on_a_free_port         the port released: FINISHED, running.
#   E_debug_answers_a_request             a packet with no command gets the
#                                         server's own JSON error back, which
#                                         only its listener can send.
#   F_debug_stops                         Stop: not running, port free again.
#
# MCP server (`mcp.start_server`, `mcp.stop_server`):
#   G_mcp_starts_on_a_free_port           FINISHED, running, on the panel port.
#   H_mcp_answers_tools_list              a real `tools/list` over HTTP gets a
#                                         complete result listing tools.
#   I_mcp_busy_port_moves_on              a port held here: by design it starts
#                                         on another port, the panel's Port
#                                         follows, and that port answers.
#   J_mcp_stops                           Stop: not running.

from __future__ import annotations

from . import _driver_lib as dl
from . import _mcp_lib as ml
from . import _runner as r


NEEDS_BLENDER = True
# Add-on code alone, run against the real backend like the rest of the add-on
# scenarios.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import socket
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def check(name, ok, details):
    result["checks"][name] = {"ok": bool(ok), "details": details}


def op(fn):
    # (outcome, error text): bpy.ops raises an operator's ERROR report.
    try:
        return set(fn()), ""
    except RuntimeError as exc:
        return {"CANCELLED"}, str(exc)


def hold_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("localhost", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


try:
    dh = DriverHelpers(pkg, result)
    rs = __import__(pkg + ".core.reload_server",
                    fromlist=["get_reload_server_status", "stop_reload_server"])
    state = dh.groups.get_addon_data(bpy.context.scene).state

    # ---- Debug server ----------------------------------------------------
    rs.stop_reload_server()
    outcome, text = op(bpy.ops.addon.trigger_reload)
    check("A_debug_reload_refused_while_stopped",
          outcome == {"CANCELLED"} and "not running" in text,
          {"outcome": sorted(outcome), "error": text[-300:]})

    holder, port = hold_port()
    state.reload_port = port
    outcome, text = op(bpy.ops.addon.start_reload_server)
    running = bool(rs.get_reload_server_status())
    check("B_debug_busy_port_is_refused",
          outcome == {"CANCELLED"} and not running,
          {"outcome": sorted(outcome), "running": running, "port": port})
    check("C_debug_refusal_names_the_port", str(port) in text,
          {"error": text[-400:]})

    holder.close()
    outcome, text = op(bpy.ops.addon.start_reload_server)
    running = bool(rs.get_reload_server_status())
    check("D_debug_starts_on_a_free_port",
          outcome == {"FINISHED"} and running,
          {"outcome": sorted(outcome), "running": running, "error": text[-300:]})

    reply = b""
    try:
        with socket.create_connection(("localhost", port), timeout=5.0) as c:
            c.sendall(json.dumps({"no_command": True}).encode("utf-8"))
            c.shutdown(socket.SHUT_WR)
            while True:
                chunk = c.recv(65536)
                if not chunk:
                    break
                reply += chunk
        answer = json.loads(reply) if reply else {}
    except Exception as exc:
        answer = {"client_error": repr(exc)}
    check("E_debug_answers_a_request",
          "command" in str(answer.get("error", "")),
          {"reply": answer})

    outcome, text = op(bpy.ops.addon.stop_reload_server)
    running = bool(rs.get_reload_server_status())
    freed = False
    for _ in range(50):
        if port_free(port):
            freed = True
            break
        time.sleep(0.1)
    check("F_debug_stops",
          outcome == {"FINISHED"} and not running and freed,
          {"outcome": sorted(outcome), "running": running, "port_free": freed})

    # ---- MCP server ------------------------------------------------------
    mcp_mod = __import__(pkg + ".mcp.mcp_server",
                         fromlist=["is_mcp_running", "get_mcp_server",
                                   "stop_mcp_server"])

    def tools_listed(port):
        url = "http://127.0.0.1:%d/mcp" % port
        envelope, resp = mcp_call(pkg, url, "tools/list", {}, request_id=7)
        ok, why = mcp_envelope_ok(envelope, 7)
        tools = (envelope.get("result") or {}).get("tools") or []
        return ok and len(tools) > 0, {"why": why, "tools": len(tools),
                                       "status": (resp or {}).get("status")}

    port = mcp_alloc_free_port()
    state.mcp_port = port
    outcome, text = op(bpy.ops.mcp.start_server)
    running = bool(mcp_mod.is_mcp_running())
    actual = getattr(mcp_mod.get_mcp_server(), "port", None)
    check("G_mcp_starts_on_a_free_port",
          outcome == {"FINISHED"} and running and actual == port,
          {"outcome": sorted(outcome), "running": running,
           "requested": port, "actual": actual, "error": text[-300:]})
    ok, details = tools_listed(actual) if running else (False, {})
    check("H_mcp_answers_tools_list", ok, details)
    op(bpy.ops.mcp.stop_server)

    holder, port = hold_port()
    state.mcp_port = port
    outcome, text = op(bpy.ops.mcp.start_server)
    running = bool(mcp_mod.is_mcp_running())
    actual = getattr(mcp_mod.get_mcp_server(), "port", None)
    ok, details = tools_listed(actual) if running and actual else (False, {})
    details.update({"outcome": sorted(outcome), "held": port, "actual": actual,
                    "panel_port": state.mcp_port, "error": text[-300:]})
    check("I_mcp_busy_port_moves_on",
          outcome == {"FINISHED"} and running and actual != port
          and state.mcp_port == actual and ok, details)
    holder.close()

    outcome, text = op(bpy.ops.mcp.stop_server)
    running = bool(mcp_mod.is_mcp_running())
    check("J_mcp_stops", outcome == {"FINISHED"} and not running,
          {"outcome": sorted(outcome), "running": running, "error": text[-300:]})

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
