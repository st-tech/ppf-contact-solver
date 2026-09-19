# File: scenarios/bl_mcp_modal_jobs.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP modal job control surface, against a real Blender.
#
# Three jobs run on a Blender timer and outlive the tool call that started
# them: the keyframe bake, the STATIC collider deformation capture and the
# pin deformation capture. get_modal_job_status reports all three, and one
# abort tool per job raises that job's abort flag.
#
# A rig scenario has no bake and no capture in flight, and starting one would
# need a fetched solve, so what this scenario covers is the other half of the
# surface: the IDLE report and the REFUSAL. Both are load-bearing. An idle
# entry has to name its job and its abort tool while reporting no progress
# counters at all, because those counters are cleared when a job ends and a
# zero read from a finished job would be indistinguishable from a job that
# processed nothing. A refusal has to be a refusal: an abort tool that
# reported success with nothing running would leave a caller polling
# get_modal_job_status for a stop that was never requested.
#
# The tool descriptions are asserted against the runtime report rather than
# quoted. get_modal_job_status's description claims, per job, which tools
# start it and which tool stops it; the checks below parse those claims out
# of the served description and require the stopper to be the abort_tool the
# report itself carries, and every starter to be a registered tool.
#
# Assertions:
#   A. ``status_names_three_idle_jobs`` -- get_modal_job_status answers with
#      the three job entries in order, an empty running_jobs and any_running
#      false.
#   B. ``idle_entry_carries_no_progress`` -- each idle entry carries exactly
#      the eight documented fields, with running and abort_requested false
#      and all four progress fields null.
#   C. ``abort_bake_refused_when_idle`` -- abort_bake with no bake running is
#      a tools/call result carrying isError, not a protocol error, and its
#      message says no bake is running and names get_modal_job_status.
#   D. ``abort_static_capture_refused_when_idle`` -- the same for the STATIC
#      deformation capture, named as such in the message.
#   E. ``abort_pin_capture_refused_when_idle`` -- and for the pin capture.
#   F. ``refusals_changed_no_job_state`` -- the status read after the three
#      refusals equals the one before, so a refused abort neither started a
#      job nor raised an abort flag.
#   G. ``status_description_matches_reported_jobs`` -- the served description
#      names, for every reported job, the abort tool that job's own entry
#      carries, plus starter tools that are registered.
#   H. ``abort_descriptions_name_the_same_starters`` -- each abort tool's own
#      description names those same starters and points back at
#      get_modal_job_status, and both descriptions agree that the two
#      single-frame bake tools start none of the three jobs.
#   I. ``fetch_status_is_a_different_question`` -- get_fetch_status answers
#      with zero fetched frames and an export preflight verdict, shares no
#      field with the modal job report, and the two descriptions name each
#      other as the separate questions they are.
#   J. ``read_tools_are_annotated_read_only`` -- the two reporting tools carry
#      readOnlyHint and idempotentHint; the three abort tools, which mutate a
#      job, carry no read-only hint.

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

# The job keys the handler reports, and the tool that stops each one.
JOB_KEYS = ["bake", "static_deformation_capture", "pin_deformation_capture"]
ABORT_TOOLS = {
    "bake": "abort_bake",
    "static_deformation_capture": "abort_static_deformation_capture",
    "pin_deformation_capture": "abort_pin_deformation_capture",
}
ENTRY_KEYS = [
    "abort_requested",
    "abort_tool",
    "frames_done",
    "frames_total",
    "item_count",
    "job",
    "running",
    "status_line",
]
PROGRESS_KEYS = ["frames_done", "frames_total", "item_count", "status_line"]
SINGLE_FRAME_TOOLS = ["bake_group_single_frame", "bake_all_single_frame"]

try:
    import re

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    def flat(text):
        # The served description is wrapped, so a claim spanning two lines is
        # one sentence only after the line breaks are folded away.
        return " ".join((text or "").split())

    def refuse(request_id, name):
        # A handler that ran and refused answers with a normal tools/call
        # result carrying isError, and mcp_tool raises on exactly that, so
        # the refusal path goes through mcp_call and reads the payload out of
        # the envelope. The envelope is returned too: a refusal must not be a
        # protocol error, which is the distinction the checks assert.
        envelope, _resp = mcp_call(
            pkg, url, "tools/call",
            {"name": name, "arguments": {}},
            request_id=request_id,
        )
        return (
            mcp_tool_payload(envelope),
            envelope.get("result") or {},
            envelope,
        )

    # ----- the served tool surface, read once ---------------------
    env, _resp = mcp_call(pkg, url, "tools/list", request_id=1)
    served = (env.get("result") or {}).get("tools") or []
    tool_names = set()
    descriptions = {}
    annotations = {}
    for tool in served:
        name = tool.get("name")
        if not isinstance(name, str):
            continue
        tool_names.add(name)
        descriptions[name] = flat(tool.get("description"))
        annotations[name] = tool.get("annotations") or {}
    if not tool_names:
        raise RuntimeError("tools/list served no tools")

    # ----- A. the idle report -------------------------------------
    status, _raw = mcp_tool(pkg, url, "get_modal_job_status", {}, request_id=2)
    jobs = status.get("jobs") or []
    reported_keys = [job.get("job") for job in jobs]
    mcp_check(
        result, "A_status_names_three_idle_jobs",
        status.get("status") == "success"
        and isinstance(status.get("jobs"), list)
        and reported_keys == JOB_KEYS
        and status.get("running_jobs") == []
        and status.get("any_running") is False,
        {
            "status": status.get("status"),
            "reported_jobs": reported_keys,
            "expected_jobs": JOB_KEYS,
            "running_jobs": status.get("running_jobs"),
            "any_running": status.get("any_running"),
        },
    )

    # ----- B. what one idle entry does and does not carry ---------
    entry_report = {}
    entries_ok = len(jobs) == len(JOB_KEYS)
    for job in jobs:
        key = job.get("job")
        wrong_keys = sorted(job) != ENTRY_KEYS
        non_null = [k for k in PROGRESS_KEYS if job.get(k) is not None]
        entry_report[str(key)] = {
            "keys": sorted(job),
            "running": job.get("running"),
            "abort_requested": job.get("abort_requested"),
            "abort_tool": job.get("abort_tool"),
            "non_null_progress": non_null,
        }
        if (
            wrong_keys
            or job.get("running") is not False
            or job.get("abort_requested") is not False
            or job.get("abort_tool") != ABORT_TOOLS.get(key)
            or non_null
        ):
            entries_ok = False
    mcp_check(
        result, "B_idle_entry_carries_no_progress",
        entries_ok,
        {
            "entries": entry_report,
            "expected_keys": ENTRY_KEYS,
            "expected_abort_tools": ABORT_TOOLS,
        },
    )

    # ----- C, D, E. each abort tool with nothing to abort ---------
    refusals = {}
    refusal_names = {
        "abort_bake": "No bake is running",
        "abort_static_deformation_capture": (
            "No STATIC deformation capture is running"
        ),
        "abort_pin_deformation_capture": "No pin deformation capture is running",
    }
    for index, tool_name in enumerate(sorted(refusal_names)):
        payload, raw, envelope = refuse(3 + index, tool_name)
        refusals[tool_name] = {
            "payload": payload,
            "is_error": raw.get("isError"),
            "protocol_error": envelope.get("error"),
        }

    def refusal_ok(tool_name):
        seen = refusals[tool_name]
        message = (seen["payload"] or {}).get("message") or ""
        return (
            (seen["payload"] or {}).get("status") == "error"
            and seen["is_error"] is True
            # A refusal is a result, not a transport failure: the handler ran.
            and seen["protocol_error"] is None
            and refusal_names[tool_name] in message
            and "nothing to abort" in message
            and "get_modal_job_status" in message
        )

    def refusal_details(tool_name):
        seen = refusals[tool_name]
        return {
            "status": (seen["payload"] or {}).get("status"),
            "message": (seen["payload"] or {}).get("message"),
            "is_error": seen["is_error"],
            "protocol_error": seen["protocol_error"],
            "expected_phrase": refusal_names[tool_name],
        }

    mcp_check(
        result, "C_abort_bake_refused_when_idle",
        refusal_ok("abort_bake"),
        refusal_details("abort_bake"),
    )
    mcp_check(
        result, "D_abort_static_capture_refused_when_idle",
        refusal_ok("abort_static_deformation_capture"),
        refusal_details("abort_static_deformation_capture"),
    )
    mcp_check(
        result, "E_abort_pin_capture_refused_when_idle",
        refusal_ok("abort_pin_deformation_capture"),
        refusal_details("abort_pin_deformation_capture"),
    )

    # ----- F. and the refusals moved nothing ----------------------
    status_after, _raw = mcp_tool(
        pkg, url, "get_modal_job_status", {}, request_id=6
    )
    mcp_check(
        result, "F_refusals_changed_no_job_state",
        status_after == status
        and status_after.get("any_running") is False,
        {
            "before": status,
            "after": status_after,
            "differing_fields": sorted(
                k for k in set(status) | set(status_after)
                if status.get(k) != status_after.get(k)
            ),
        },
    )

    # ----- G. the description's claims, against the report --------
    claims = {}
    claims_ok = bool(descriptions.get("get_modal_job_status"))
    status_description = descriptions.get("get_modal_job_status", "")
    for job in jobs:
        key = str(job.get("job"))
        match = re.search(
            "``" + key + "`` is started by ([a-z_]+) or ([a-z_]+) "
            "and stopped by ([a-z_]+)",
            status_description,
        )
        if match is None:
            claims[key] = {"claim": None}
            claims_ok = False
            continue
        starters = [match.group(1), match.group(2)]
        stopper = match.group(3)
        unregistered = [s for s in starters if s not in tool_names]
        claims[key] = {
            "starters": starters,
            "stopper": stopper,
            "reported_abort_tool": job.get("abort_tool"),
            "unregistered_starters": unregistered,
        }
        if unregistered or stopper != job.get("abort_tool"):
            claims_ok = False
    mcp_check(
        result, "G_status_description_matches_reported_jobs",
        claims_ok and sorted(claims) == sorted(JOB_KEYS),
        {
            "claims": claims,
            "description_present": bool(status_description),
        },
    )

    # ----- H. and the abort tools tell the same story -------------
    abort_desc_report = {}
    abort_desc_ok = True
    for key in JOB_KEYS:
        tool_name = ABORT_TOOLS[key]
        text = descriptions.get(tool_name, "")
        starters = (claims.get(key) or {}).get("starters") or []
        missing = [s for s in starters if s not in text]
        points_back = "get_modal_job_status" in text
        abort_desc_report[tool_name] = {
            "starters": starters,
            "missing_starters": missing,
            "points_at_status_tool": points_back,
        }
        if not text or missing or not points_back:
            abort_desc_ok = False
    # A single-frame bake finishes inline, so neither tool starts a job and
    # neither is named as a starter of one.
    single_frame_registered = [
        t for t in SINGLE_FRAME_TOOLS if t in tool_names
    ]
    named_starters = set()
    for claim in claims.values():
        named_starters.update(claim.get("starters") or [])
    single_frame_claimed = sorted(
        set(SINGLE_FRAME_TOOLS) & named_starters
    )
    single_frame_phrase = (
        "bake_group_single_frame and bake_all_single_frame start none of "
        "the three" in status_description
        and "bake_group_single_frame and bake_all_single_frame start no such "
        "job" in descriptions.get("abort_bake", "")
    )
    mcp_check(
        result, "H_abort_descriptions_name_the_same_starters",
        abort_desc_ok
        and single_frame_registered == SINGLE_FRAME_TOOLS
        and not single_frame_claimed
        and single_frame_phrase,
        {
            "abort_descriptions": abort_desc_report,
            "single_frame_registered": single_frame_registered,
            "single_frame_claimed_as_starter": single_frame_claimed,
            "single_frame_phrase_present": single_frame_phrase,
        },
    )

    # ----- I. fetch progress is the other question ----------------
    fetch, _raw = mcp_tool(pkg, url, "get_fetch_status", {}, request_id=7)
    shared_fields = sorted(
        (set(fetch) & set(status)) - {"status"}
    )
    fetch_text = descriptions.get("get_fetch_status", "")
    mcp_check(
        result, "I_fetch_status_is_a_different_question",
        fetch.get("status") == "success"
        and fetch.get("fetched_frames") == []
        and fetch.get("fetched_count") == 0
        and isinstance(fetch.get("expected_frame_count"), int)
        and fetch.get("export_ready") is False
        and bool(fetch.get("export_blocked_reason"))
        and not shared_fields
        and "get_modal_job_status" in fetch_text
        and "get_fetch_status" in status_description,
        {
            "fetch_status": fetch,
            "shared_fields_with_job_status": shared_fields,
            "fetch_names_job_status": "get_modal_job_status" in fetch_text,
            "job_status_names_fetch": "get_fetch_status" in status_description,
        },
    )

    # ----- J. reporting versus mutating, in the annotations -------
    read_tools = ["get_modal_job_status", "get_fetch_status"]
    read_ok = all(
        annotations.get(t, {}).get("readOnlyHint") is True
        and annotations.get(t, {}).get("idempotentHint") is True
        for t in read_tools
    )
    abort_ok = all(
        not annotations.get(ABORT_TOOLS[k], {}).get("readOnlyHint")
        for k in JOB_KEYS
    )
    mcp_check(
        result, "J_read_tools_are_annotated_read_only",
        read_ok and abort_ok,
        {
            "read_tool_annotations": {
                t: annotations.get(t) for t in read_tools
            },
            "abort_tool_annotations": {
                ABORT_TOOLS[k]: annotations.get(ABORT_TOOLS[k])
                for k in JOB_KEYS
            },
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
