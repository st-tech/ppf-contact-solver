# File: scenarios/bl_mcp_console_and_diagnostics.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The diagnostic surface over MCP: the escape hatch that runs Python inside
# Blender, the console and error readers, the UI element report, and the mesh
# hygiene checks an agent runs before a build.
#
# Assertions:
#   A. ``script_runs_and_returns_its_output`` -- run_python_script executes in
#      Blender and hands back what the code printed.
#   B. ``script_failure_is_a_tool_error_not_a_protocol_error`` -- a script
#      that raises comes back as a well-formed result carrying isError and the
#      exception text, so the model can see it and correct, and the transport
#      still answers 200.
#   C. ``escape_hatch_is_annotated`` -- run_python_script and
#      execute_shell_command both carry destructiveHint and openWorldHint.
#      Neither name implies its risk, so the annotation is the only warning a
#      client gets before running unsandboxed code.
#   D. ``console_and_error_readers_answer`` -- get_console_lines returns a
#      line list with a matching count, and get_latest_error reports both
#      error slots and the has_errors flag.
#   E. ``ui_properties_are_reported_and_bad_type_refused`` -- element_type
#      "property" returns properties carrying name, value, type and category,
#      and an unknown element_type is refused with the accepted values named.
#   F. ``mesh_hygiene_checks_answer_on_a_clean_scene`` -- the isolated-vertex
#      detector and its repair both report zero on a scene with no STATIC
#      collider, rather than failing.
#   G. ``viewport_capture_is_all_or_nothing`` -- capture_viewport_image
#      either writes the file it names or refuses with a reason, and never
#      reports a success that left no file. Which of the two happens depends
#      on the Blender the scenario runs in: the rig gives it a UI on Xvfb and
#      the capture succeeds, while a --background Blender has no GL context
#      and it refuses. The invariant holds either way, so this is what the
#      check asserts. delete_log_file names a path that is not there.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

# macOS runners block loopback HTTP to Blender's in-process MCP server, so the
# rig does not select this scenario there.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

_rid = [700]


def call(name, args=None):
    _rid[0] += 1
    payload, raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload, raw


try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port

    # ----- A --------------------------------------------------------
    ok_script, _ = call("run_python_script",
                        {"code": "print('marker-8842')"})
    mcp_check(result, "A_script_runs_and_returns_its_output",
              ok_script.get("status") == "success"
              and "marker-8842" in (ok_script.get("output") or ""),
              {"reply": ok_script})

    # ----- B --------------------------------------------------------
    _rid[0] += 1
    envelope_b, resp_b = mcp_call(
        pkg, url, "tools/call",
        {"name": "run_python_script", "arguments": {"code": "1 / 0"}},
        request_id=_rid[0],
    )
    payload_b = mcp_tool_payload(envelope_b)
    result_b = envelope_b.get("result") or {}
    mcp_check(result, "B_script_failure_is_a_tool_error_not_a_protocol_error",
              resp_b.get("status") == 200
              and "error" not in envelope_b
              and result_b.get("isError") is True
              and payload_b.get("status") == "error"
              and "division by zero" in payload_b.get("message", ""),
              {"http_status": resp_b.get("status"),
               "is_error": result_b.get("isError"),
               "message": payload_b.get("message")})

    # ----- C --------------------------------------------------------
    _rid[0] += 1
    tools_env, _ = mcp_call(pkg, url, "tools/list", request_id=_rid[0])
    by_name = {t["name"]: t for t in ((tools_env.get("result") or {}).get("tools") or [])}
    hatches = {}
    for name in ("run_python_script", "execute_shell_command"):
        hatches[name] = (by_name.get(name) or {}).get("annotations") or {}
    mcp_check(result, "C_escape_hatch_is_annotated",
              all(a.get("destructiveHint") is True and a.get("openWorldHint") is True
                  for a in hatches.values()) and len(hatches) == 2,
              {"annotations": hatches})

    # ----- D --------------------------------------------------------
    console, _ = call("get_console_lines")
    latest, _ = call("get_latest_error")
    lines = console.get("console_lines")
    mcp_check(result, "D_console_and_error_readers_answer",
              console.get("status") == "success"
              and isinstance(lines, list)
              and console.get("line_count") == len(lines)
              and latest.get("status") == "success"
              and "local_error" in latest and "remote_error" in latest
              and latest.get("has_errors") is False,
              {"line_count": console.get("line_count"), "latest": latest})

    # ----- E --------------------------------------------------------
    props, _ = call("get_ui_element_status",
                    {"element_type": "property", "category": "solver"})
    listed = (props.get("elements") or {}).get("properties") or []
    shapes_ok = all(
        set(("name", "value", "type", "category")) <= set(p)
        and p.get("category") == "solver"
        for p in listed
    )
    bad_type, _ = call("get_ui_element_status", {"element_type": "bogus"})
    mcp_check(result, "E_ui_properties_are_reported_and_bad_type_refused",
              props.get("status") == "success"
              and len(listed) > 0 and shapes_ok
              and props.get("total_properties") == len(listed)
              and bad_type.get("status") == "error"
              and all(t in bad_type.get("message", "")
                      for t in ("operator", "property", "all")),
              {"property_count": len(listed), "shapes_ok": shapes_ok,
               "sample": listed[0] if listed else None,
               "bad_type": bad_type.get("message")})

    # ----- F --------------------------------------------------------
    detect, _ = call("detect_isolated_static_vertices")
    repair, _ = call("remove_isolated_static_vertices")
    mcp_check(result, "F_mesh_hygiene_checks_answer_on_a_clean_scene",
              detect.get("status") == "success"
              and detect.get("total_isolated") == 0 and detect.get("objects") == []
              and repair.get("status") == "success"
              and repair.get("removed_total") == 0,
              {"detect": detect, "repair": repair})

    # ----- G --------------------------------------------------------
    os_mod = __import__("os")
    shot_path = "/tmp/bl_mcp_probe_shot.png"
    if os_mod.path.exists(shot_path):
        os_mod.remove(shot_path)
    shot, _ = call("capture_viewport_image", {"filepath": shot_path})
    wrote_file = os_mod.path.exists(shot_path)
    missing_log, _ = call("delete_log_file",
                          {"log_file_path": "/tmp/bl_mcp_no_such_file.log"})
    mcp_check(result, "G_viewport_capture_is_all_or_nothing",
              bool(shot.get("message"))
              and ((shot.get("status") == "success" and wrote_file)
                   or (shot.get("status") == "error" and not wrote_file))
              and missing_log.get("status") == "error"
              and "does not exist" in missing_log.get("message", ""),
              {"capture_status": shot.get("status"),
               "capture_message": shot.get("message"),
               "wrote_file": wrote_file,
               "log": missing_log.get("message")})
    if os_mod.path.exists(shot_path):
        os_mod.remove(shot_path)

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
