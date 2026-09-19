# File: scenarios/bl_mcp_error_semantics.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The line between a protocol error and a tool that ran and failed.
#
# A client reads that line to decide what to do next. A protocol error means
# the request itself was malformed, so retrying it unchanged cannot succeed
# and the model should never be shown the failure as tool output. An isError
# result means the call reached the tool and the tool refused, which is
# content the model has to read to correct itself. The two therefore differ
# in every observable: HTTP status, which envelope member is populated, and
# whether a payload is present at all.
#
# The transport keeps them apart in one place. ``ProtocolError`` carries a
# JSON-RPC code out to ``_send_rpc_error``, which maps the code to an HTTP
# status and writes an error frame with no result; a handler's own failure
# comes back through ``_tool_result``, which wraps the payload in a normal
# result and sets ``isError`` when the payload's status is ``"error"``.
# Failing to FIND a tool belongs to the first group, not the second.
#
# Both tool calls here fail or succeed without a solver, a connection or a
# frame, so the scenario is deterministic in a headless rig: ``delete_group``
# on a UUID no scene holds raises before it touches anything, and
# ``get_active_groups`` reads the group list a fresh scene starts with.
#
# Assertions:
#   A. ``unknown_tool_is_protocol_error`` -- a name no handler is registered
#      under is 400 with -32602 naming the tool, and the envelope carries no
#      result, so it is never mistaken for a tool that ran.
#   B. ``failed_tool_is_result_with_is_error`` -- a real tool given arguments
#      it must refuse answers 200 with a well-formed result carrying
#      isError, structuredContent status "error" and a message that names
#      what was not found, and no error member.
#   C. ``successful_tool_has_no_is_error`` -- a successful call carries
#      structuredContent with status "success", no isError, and a text block
#      that parses back to the same payload.
#   D. ``non_object_arguments_refused`` -- "arguments" that is not an object
#      is 400 with -32602 for a string, a list and a number alike.
#   E. ``missing_name_is_refused`` -- a tools/call with no "name" is refused
#      in both eras. The legacy era mirrors no name header, so it reaches the
#      params check and answers -32602; the modern era requires Mcp-Name to
#      mirror the body, so the header contract refuses it first with -32020.
#      Neither carries a result.
#   F. ``empty_name_is_refused`` -- an empty name does mirror into Mcp-Name,
#      so it reaches the params check in the modern era and is refused with
#      -32602, which is the "non-empty string" half of the same rule.

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

# A name no handler is registered under, spelled so it cannot collide with a
# tool added later.
ABSENT_TOOL = "no_such_tool_bl_mcp_error_semantics"

# A syntactically valid UUID that no group in a fresh scene carries.
ABSENT_GROUP_UUID = "00000000-0000-4000-8000-00000000dead"


def text_payload(res):
    # The text block a model reads, parsed back. Returns None when there is
    # no block or it does not parse, both of which fail the check that uses
    # it rather than raising here.
    content = res.get("content") or []
    if not content or not isinstance(content[0], dict):
        return None
    try:
        return json.loads(content[0].get("text") or "")
    except Exception:
        return None


try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. a tool that is not there never ran ------------------
    env, resp = mcp_call(
        pkg, url, "tools/call",
        {"name": ABSENT_TOOL, "arguments": {}},
        request_id=1,
    )
    err = env.get("error") or {}
    mcp_check(
        result, "A_unknown_tool_is_protocol_error",
        resp["status"] == 400
        and err.get("code") == -32602
        and ABSENT_TOOL in (err.get("message") or "")
        and "result" not in env,
        {
            "status": resp["status"],
            "error": err,
            "carries_result": "result" in env,
        },
    )

    # ----- B. a tool that ran and refused ------------------------
    env, resp = mcp_call(
        pkg, url, "tools/call",
        {
            "name": "delete_group",
            "arguments": {"group_uuid": ABSENT_GROUP_UUID},
        },
        request_id=2,
    )
    res = env.get("result") or {}
    structured = res.get("structuredContent")
    parsed = text_payload(res)
    message = (structured or {}).get("message") or ""
    env_ok, why = mcp_envelope_ok(env, 2)
    mcp_check(
        result, "B_failed_tool_is_result_with_is_error",
        resp["status"] == 200
        and env_ok
        and "error" not in env
        and res.get("isError") is True
        and isinstance(structured, dict)
        and structured.get("status") == "error"
        and bool(message.strip())
        and ABSENT_GROUP_UUID in message
        and parsed == structured,
        {
            "status": resp["status"],
            "envelope_ok": env_ok,
            "why": why,
            "carries_error": "error" in env,
            "isError": res.get("isError"),
            "structured": structured,
            "text_matches_structured": parsed == structured,
        },
    )

    # ----- C. a tool that ran and succeeded ----------------------
    env, resp = mcp_call(
        pkg, url, "tools/call",
        {"name": "get_active_groups", "arguments": {}},
        request_id=3,
    )
    res = env.get("result") or {}
    structured = res.get("structuredContent")
    parsed = text_payload(res)
    env_ok, why = mcp_envelope_ok(env, 3)
    mcp_check(
        result, "C_successful_tool_has_no_is_error",
        resp["status"] == 200
        and env_ok
        and "isError" not in res
        and isinstance(structured, dict)
        and structured.get("status") == "success"
        and isinstance(structured.get("groups"), list)
        and parsed == structured,
        {
            "status": resp["status"],
            "envelope_ok": env_ok,
            "why": why,
            "isError_present": "isError" in res,
            "structured_keys": sorted(structured) if isinstance(structured, dict)
            else None,
            "structured_status": (structured or {}).get("status"),
            "text_matches_structured": parsed == structured,
        },
    )

    # ----- D. arguments that are not an object -------------------
    argument_probes = {}
    for label, bad in (("string", "not-an-object"), ("list", [1, 2]), ("number", 7)):
        env, resp = mcp_call(
            pkg, url, "tools/call",
            {"name": "get_active_groups", "arguments": bad},
            request_id=4,
        )
        argument_probes[label] = [
            resp["status"],
            (env.get("error") or {}).get("code"),
            "result" in env,
        ]
    mcp_check(
        result, "D_non_object_arguments_refused",
        all(probe == [400, -32602, False] for probe in argument_probes.values()),
        argument_probes,
    )

    # ----- E. a call that names no tool at all -------------------
    # Modern: Mcp-Name is required for tools/call and must mirror
    # params.name, so an absent name has no header to send and the header
    # contract refuses the request before the params are read.
    env_modern, resp_modern = mcp_call(
        pkg, url, "tools/call", {"arguments": {}}, request_id=5,
    )
    # Legacy: no mirrored headers, so the same request reaches the params
    # check. The era needs a session, which only an initialize mints.
    env_init, resp_init = mcp_call(
        pkg, url, "initialize", {}, request_id=6,
        version=MCP_LEGACY_VERSION, meta=False,
    )
    session_id = resp_init["headers"].get("Mcp-Session-Id", "")
    env_legacy, resp_legacy = mcp_call(
        pkg, url, "tools/call", {"arguments": {}}, request_id=7,
        version=MCP_LEGACY_VERSION, meta=False,
        extra_headers={"Mcp-Session-Id": session_id},
    )
    mcp_check(
        result, "E_missing_name_is_refused",
        bool(session_id)
        and resp_legacy["status"] == 400
        and (env_legacy.get("error") or {}).get("code") == -32602
        and "result" not in env_legacy
        and resp_modern["status"] == 400
        and (env_modern.get("error") or {}).get("code") == -32020
        and "result" not in env_modern,
        {
            "session_minted": bool(session_id),
            "legacy": [resp_legacy["status"], env_legacy.get("error")],
            "modern": [resp_modern["status"], env_modern.get("error")],
            "legacy_carries_result": "result" in env_legacy,
            "modern_carries_result": "result" in env_modern,
        },
    )

    # ----- F. a name that is present but empty -------------------
    env, resp = mcp_call(
        pkg, url, "tools/call",
        {"name": "", "arguments": {}},
        request_id=8,
    )
    err = env.get("error") or {}
    mcp_check(
        result, "F_empty_name_is_refused",
        resp["status"] == 400
        and err.get("code") == -32602
        and "result" not in env,
        {
            "status": resp["status"],
            "error": err,
            "carries_result": "result" in env,
        },
    )

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
