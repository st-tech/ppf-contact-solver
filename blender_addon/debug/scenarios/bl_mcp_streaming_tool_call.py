# File: scenarios/bl_mcp_streaming_tool_call.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The tools/call response-stream path, against a real Blender.
#
# ``tools/call`` is the only request whose duration the transport cannot
# bound, so it is the only one answered two different ways. A call that
# finishes inside ``_TOOL_CALL_INLINE_SECONDS`` is answered with one
# ``application/json`` object; a call still running when that window closes is
# answered on a ``text/event-stream`` response instead, and the result arrives
# as the last event on it. A client that asked for progress by putting a
# ``progressToken`` in ``params._meta`` receives ``notifications/progress``
# events while it waits; one that did not receives keep-alive comments, which
# hold the connection open without inventing a notification the client never
# subscribed to.
#
# Only a real Blender can show this. A tool runs on Blender's main thread and
# reaches it through the ``mcp.task_system`` queue, which drains when
# ``process_mcp_tasks`` is called from that thread, so how long a call takes
# is a property of the running application and not of the transport code.
#
# THREE CONSTRAINTS SHAPE THE DRIVER:
#
#   1. The driver holds the main thread for the whole of its ``exec()``, so
#      every request goes out on a worker thread while the driver pumps the
#      queue. This is the same rule ``_mcp_lib.mcp_drive`` follows.
#
#   2. ``mcp_drive`` cannot be reused here. It pumps immediately, which makes
#      every call fast, and it reads the response only after the worker has
#      consumed all of it, which loses the order the events arrived in. The
#      driver below reads the response line by line as it arrives and records
#      each frame with its arrival time.
#
#   3. The slow tool sleeps on the main thread, which is the thread the driver
#      pumps from, so one ``process_mcp_tasks()`` call blocks for the whole
#      sleep. Reading has to be on the worker for that reason as well. To
#      select the streaming branch without depending on how fast the machine
#      is, the driver holds the pump past the inline window before draining:
#      the tool cannot start before the pump runs, so the window is closed by
#      the clock rather than by a race.
#
# Assertions:
#   A. ``A_fast_call_answers_inline_json`` -- a get_ tool finishes inside the
#      inline window and is answered with one application/json response, even
#      though the client offered to read an event stream.
#   B. ``B_slow_call_streams_its_result`` -- a call still running when the
#      window closes is answered on text/event-stream, and the final event is
#      the JSON-RPC result for that id carrying the tool's own output, with
#      resultType complete and isError absent.
#   C. ``C_progress_precedes_result`` -- with a progressToken in params._meta,
#      at least one notifications/progress event carrying that same token
#      arrives before the result event.
#   D. ``D_no_token_keeps_alive_without_progress`` -- the same slow call with
#      no progressToken carries keep-alive comments instead, emits no
#      notifications/progress, and still delivers the result.
#   E. ``E_client_without_sse_stays_on_json`` -- a client whose Accept omits
#      text/event-stream is answered with application/json even when the call
#      outlives the inline window.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

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

# The transport's own constants, which the holds below are chosen against:
# a call is answered inline for 1.0 s, and _await_task writes its first frame
# 2.0 s after the stream opens. Each hold straddles the boundary it is aimed
# at by at least 0.3 s, and the slow tool finishes 1.5 s after the first
# progress frame is due, so no check depends on winning a race.
INLINE_SECONDS = 1.0
PROGRESS_SECONDS = 2.0
HOLD_PAST_INLINE = 1.5
HOLD_UNDER_KEEPALIVE = 1.3
TOOL_SLEEP_SECONDS = 3.0
PROGRESS_TOKEN = "bl-mcp-streaming-1"

# Printed by the slow tool and carried back in its payload, which is what
# distinguishes the real handler's result from a transport-shaped placeholder.
SLOW_MARKER = "slow tool finished"
SLOW_CODE = (
    "import time\n"
    "time.sleep(%.1f)\n"
    "print(%r)\n" % (TOOL_SLEEP_SECONDS, SLOW_MARKER)
)


def header_value(headers, name):
    # RFC 9110 field names are case-insensitive.
    lowered = name.lower()
    for key, value in (headers or {}).items():
        if key.lower() == lowered:
            return value
    return ""


def media_type(headers):
    return header_value(headers, "Content-Type").split(";", 1)[0].strip().lower()


def read_incrementally(url, body, headers, out, timeout):
    # Issue the POST and read the response frame by frame, recording arrival
    # order. urlopen returns once the status line and headers are in, which
    # for a streamed answer is well before the result exists, so the reads
    # below are what observe the stream while the tool is still running.
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    out["events"] = []
    out["comments"] = []
    out["frames"] = []
    out["body"] = ""
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        out["status"] = exc.code
        out["headers"] = dict(exc.headers.items())
        out["body"] = exc.read().decode("utf-8", "replace")
        return
    except Exception as exc:
        out["status"] = -1
        out["headers"] = {}
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return

    out["status"] = resp.status
    out["headers"] = dict(resp.headers.items())
    started = time.time()
    chunks = []
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace")
            chunks.append(text)
            stripped = text.rstrip("\r\n")
            if not stripped:
                continue
            out["frames"].append([round(time.time() - started, 2), stripped[:120]])
            if stripped.startswith("data:"):
                try:
                    out["events"].append(json.loads(stripped[5:].strip()))
                except ValueError:
                    out["events"].append({"_unparsed": stripped[5:]})
            elif stripped.startswith(":"):
                out["comments"].append(stripped)
    except Exception as exc:
        out["error"] = "read: %s: %s" % (type(exc).__name__, exc)
    finally:
        resp.close()
    out["body"] = "".join(chunks)


def exchange(url, body, headers, hold_seconds, timeout=60.0):
    # Pumping the queue is what runs a handler, so holding the pump is what
    # keeps a call outstanding. Hold for hold_seconds, then drain until the
    # reader returns.
    out = {}
    worker = threading.Thread(
        target=read_incrementally,
        args=(url, body, headers, out, timeout),
        daemon=True,
    )
    started = time.time()
    worker.start()
    hold_until = started + hold_seconds
    while worker.is_alive() and time.time() < hold_until:
        time.sleep(0.02)
    deadline = time.time() + timeout
    while worker.is_alive() and time.time() < deadline:
        try:
            process_mcp_tasks()
        except Exception:
            pass
        time.sleep(0.02)
    worker.join(timeout=5.0)
    out["elapsed"] = round(time.time() - started, 2)
    if worker.is_alive():
        out.setdefault("error", "reader thread did not finish")
    return out


def slow_request(request_id, progress_token=None):
    body, headers = mcp_build(
        "tools/call",
        {"name": "run_python_script", "arguments": {"code": SLOW_CODE}},
        request_id=request_id,
    )
    if progress_token is not None:
        body["params"]["_meta"]["progressToken"] = progress_token
    return body, headers


def split_stream(events, request_id):
    # Returns (progress frames with their index, result envelope, its index).
    progress = []
    final = None
    final_index = -1
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if event.get("method") == "notifications/progress":
            progress.append((index, event))
        elif event.get("id") == request_id and "result" in event:
            final = event
            final_index = index
    return progress, final, final_index


def stream_summary(out, progress, final_index):
    # Enough to diagnose a failure without re-running: what came back, when,
    # and in what order.
    return {
        "status": out.get("status"),
        "content_type": media_type(out.get("headers")),
        "elapsed": out.get("elapsed"),
        "frames": out.get("frames"),
        "comment_count": len(out.get("comments") or []),
        "progress_indices": [index for index, _ in progress],
        "result_index": final_index,
        "read_error": out.get("error"),
    }


try:
    process_mcp_tasks = __import__(
        pkg + ".mcp.task_system", fromlist=["process_mcp_tasks"]
    ).process_mcp_tasks

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. a call that finishes inside the inline window -------
    env, resp = mcp_call(
        pkg, url, "tools/call",
        {"name": "get_scene_info", "arguments": {}},
        request_id=101,
    )
    ok, why = mcp_envelope_ok(env, 101)
    payload = mcp_tool_payload(env)
    accept_offered = "text/event-stream"
    mcp_check(
        result, "A_fast_call_answers_inline_json",
        resp["status"] == 200
        and media_type(resp.get("headers")) == "application/json"
        and ok
        and payload.get("status") == "success"
        and "isError" not in (env.get("result") or {}),
        {
            "status": resp["status"],
            "content_type": media_type(resp.get("headers")),
            "accept_offered": accept_offered,
            "envelope_ok": ok,
            "why": why,
            "payload_status": payload.get("status"),
            "is_error": (env.get("result") or {}).get("isError"),
        },
    )
    result["phases"].append((time.time(), "A inline call answered"))

    # ----- B and C. one slow call, asking for progress ------------
    body, headers = slow_request(102, PROGRESS_TOKEN)
    tracked = exchange(url, body, headers, HOLD_PAST_INLINE)
    progress, final, final_index = split_stream(tracked["events"], 102)
    final_result = (final or {}).get("result") or {}
    final_payload = mcp_tool_payload(final or {})
    summary = stream_summary(tracked, progress, final_index)
    result["phases"].append((time.time(), "B/C streamed call answered"))

    mcp_check(
        result, "B_slow_call_streams_its_result",
        tracked.get("status") == 200
        and media_type(tracked.get("headers")) == "text/event-stream"
        and final is not None
        and final.get("jsonrpc") == "2.0"
        and final_result.get("resultType") == "complete"
        and "isError" not in final_result
        and final_payload.get("status") == "success"
        and final_payload.get("success") is True
        and SLOW_MARKER in (final_payload.get("output") or ""),
        dict(
            summary,
            result_type=final_result.get("resultType"),
            is_error=final_result.get("isError"),
            payload_status=final_payload.get("status"),
            payload_output=(final_payload.get("output") or "")[:120],
        ),
    )

    progress_tokens = [
        ((event.get("params") or {}).get("progressToken")) for _, event in progress
    ]
    mcp_check(
        result, "C_progress_precedes_result",
        bool(progress)
        and final is not None
        and progress[0][0] < final_index
        and all(token == PROGRESS_TOKEN for token in progress_tokens),
        dict(
            summary,
            expected_token=PROGRESS_TOKEN,
            progress_tokens=progress_tokens,
            first_progress=(progress[0][1] if progress else None),
        ),
    )

    # ----- D. the same wait, with no progress subscription --------
    body, headers = slow_request(103)
    untracked = exchange(url, body, headers, HOLD_PAST_INLINE)
    d_progress, d_final, d_final_index = split_stream(untracked["events"], 103)
    d_payload = mcp_tool_payload(d_final or {})
    d_summary = stream_summary(untracked, d_progress, d_final_index)
    result["phases"].append((time.time(), "D untracked streamed call answered"))

    mcp_check(
        result, "D_no_token_keeps_alive_without_progress",
        untracked.get("status") == 200
        and media_type(untracked.get("headers")) == "text/event-stream"
        and not d_progress
        and any(c.startswith(":") for c in (untracked.get("comments") or []))
        and d_final is not None
        and d_payload.get("status") == "success"
        and SLOW_MARKER in (d_payload.get("output") or ""),
        dict(
            d_summary,
            comments=(untracked.get("comments") or [])[:4],
            payload_status=d_payload.get("status"),
        ),
    )

    # ----- E. a client that will not read a stream ----------------
    # Held past the inline window, so the answer comes from the branch that
    # keeps waiting on this connection rather than from the inline return.
    # The hold stays under INLINE_SECONDS + PROGRESS_SECONDS on purpose: past
    # that point _await_task writes a keep-alive comment, and on this branch
    # no status line has been sent yet, so the bytes land ahead of the
    # response instead of inside a stream.
    body, headers = mcp_build(
        "tools/call",
        {"name": "get_scene_info", "arguments": {}},
        request_id=104,
    )
    headers["Accept"] = "application/json"
    json_only = exchange(url, body, headers, HOLD_UNDER_KEEPALIVE)
    json_env = mcp_parse(json_only)
    json_ok, json_why = mcp_envelope_ok(json_env, 104)
    json_payload = mcp_tool_payload(json_env)
    result["phases"].append((time.time(), "E json-only call answered"))

    mcp_check(
        result, "E_client_without_sse_stays_on_json",
        json_only.get("status") == 200
        and media_type(json_only.get("headers")) == "application/json"
        and json_only.get("elapsed", 0.0) >= INLINE_SECONDS
        and json_ok
        and json_payload.get("status") == "success"
        and not (json_only.get("events") or []),
        {
            "status": json_only.get("status"),
            "content_type": media_type(json_only.get("headers")),
            "accept_sent": "application/json",
            "elapsed": json_only.get("elapsed"),
            "inline_seconds": INLINE_SECONDS,
            "envelope_ok": json_ok,
            "why": json_why,
            "payload_status": json_payload.get("status"),
            "event_count": len(json_only.get("events") or []),
            "read_error": json_only.get("error"),
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
