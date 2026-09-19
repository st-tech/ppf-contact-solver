# File: scenarios/bl_mcp_roundtrip.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP HTTP server round-trip.
#
# The addon ships an in-process HTTP server that exposes its dynamics
# group surface to MCP clients (LLMs, IDE plug-ins, etc.) via the
# Streamable HTTP transport (spec 2026-07-28). This scenario spins up
# that server on a free port, drives it as a stateless client, and
# asserts that ``tools/call`` invocations mutate addon state the same
# way the UI operators would.
#
# Two threading constraints shape the driver:
#
#   1. The MCP server thread enqueues every ``tools/call`` into the
#      ``mcp.task_system`` queue. Tasks drain only when
#      ``process_mcp_tasks`` is called from Blender's main thread (the
#      production path is a 0.25s timer registered in
#      ``ui/console.py``). Our driver holds the main thread for the
#      duration of ``exec()``, so the timer cannot fire and the HTTP
#      handler would block its 5s ``get_mcp_result`` wait if we issued
#      requests synchronously. Solution: each request runs on a
#      background thread; the driver pumps ``process_mcp_tasks`` from
#      the main thread until the thread returns.
#
#   2. The transport is stateless. Every request is a standalone POST
#      carrying its protocol version, client capabilities and client
#      identity in ``params._meta``, mirrored into the
#      ``MCP-Protocol-Version``, ``Mcp-Method`` and ``Mcp-Name``
#      headers. The server rejects any disagreement between a header
#      and the body, so the driver derives the headers from the body.
#
# Assertions:
#   A. ``mcp_create_group_creates_group`` -- after ``create_group``,
#      ``rename_group``, and ``set_group_type`` the addon exposes one
#      active group named "Cloth" with ``object_type == "SHELL"``.
#   B. ``mcp_add_objects_assigns`` -- after ``add_objects_to_group`` the
#      group's ``assigned_objects`` list carries the Plane with a
#      non-empty UUID.
#   C. ``mcp_responses_match_schema`` -- every JSON-RPC response carries
#      the documented success shape (``jsonrpc == "2.0"``, ``id`` echoed,
#      ``result.content[0].text`` parses as JSON whose ``status`` is
#      ``"success"``), and every result carries ``resultType``.
#   D. ``D_mcp_transport_is_stateless`` -- ``server/discover`` names the
#      served protocol versions, no response mints an ``Mcp-Session-Id``,
#      and the retired ``GET`` and ``DELETE`` verbs answer 405.

from __future__ import annotations


from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
import json
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def _alloc_free_port():
    # Bind ephemeral, capture the port, close. The MCP server's start
    # path will rebind with SO_REUSEADDR so reuse-after-TIME_WAIT is
    # safe in the small window between close() and start().
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _post_mcp(url, body, headers, timeout=10.0):
    # Wrap urllib.request so the threaded caller can collect status,
    # body, and response headers without raising.
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    base_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    for k, v in {**base_headers, **(headers or {})}.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {
                "status": resp.status,
                "headers": dict(resp.headers.items()),
                "body": resp.read().decode("utf-8"),
                "error": None,
            }
    except urllib.error.HTTPError as e:
        try:
            body_text = e.read().decode("utf-8")
        except Exception:
            body_text = ""
        return {"status": e.code, "headers": dict(e.headers.items()),
                "body": body_text, "error": str(e)}
    except Exception as e:
        return {"status": -1, "headers": {}, "body": "", "error": str(e)}


_PROTOCOL_VERSION = "2026-07-28"
_NAME_HEADER_SOURCE = {
    "tools/call": "name",
    "resources/read": "uri",
    "prompts/get": "name",
}


def _modernize(body, headers):
    # Attach the request metadata and the headers that must mirror it.
    # Derived from the body rather than supplied by the caller: the server
    # rejects a header that disagrees with the body, so deriving them here
    # is the only way the two cannot drift.
    body = dict(body)
    method = body.get("method", "")
    params = dict(body.get("params") or {})
    params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": _PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "bl_mcp_roundtrip",
            "version": "0.1",
        },
    }
    body["params"] = params
    merged = {
        "MCP-Protocol-Version": _PROTOCOL_VERSION,
        "Mcp-Method": method,
    }
    source = _NAME_HEADER_SOURCE.get(method)
    if source and isinstance(params.get(source), str):
        merged["Mcp-Name"] = params[source]
    merged.update(headers or {})
    return body, merged


def _drive_request_in_thread(url, body, headers, *, timeout=15.0,
                             pump_interval=0.05):
    # Issue the POST on a background thread so that this driver (running
    # on Blender's main thread) can keep pumping ``process_mcp_tasks``
    # while the MCP HTTP handler waits on ``get_mcp_result``.
    process_mcp_tasks = __import__(
        pkg + ".mcp.task_system", fromlist=["process_mcp_tasks"]
    ).process_mcp_tasks
    box = {}

    modern_body, modern_headers = _modernize(body, headers)

    def worker():
        box["resp"] = _post_mcp(
            url, modern_body, modern_headers, timeout=timeout - 1.0
        )

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while t.is_alive() and time.time() < deadline:
        try:
            process_mcp_tasks()
        except Exception as exc:
            result["errors"].append(f"pump: {type(exc).__name__}: {exc}")
        time.sleep(pump_interval)
    t.join(timeout=2.0)
    return box.get("resp") or {"status": -1, "headers": {}, "body": "",
                               "error": "thread timeout"}


def _parse_jsonrpc(resp):
    # Return the parsed JSON-RPC envelope or raise on protocol error.
    if resp.get("status") != 200:
        raise RuntimeError(
            f"HTTP {resp.get('status')}: {resp.get('body')!r} "
            f"(error={resp.get('error')!r})"
        )
    try:
        envelope = json.loads(resp["body"])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"non-JSON response: {resp['body']!r}") from e
    return envelope


def _extract_tool_payload(envelope):
    # ``tools/call`` wraps the handler return inside
    # ``result.content[0].text`` (JSON encoded). Decode it.
    try:
        text = envelope["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"missing result.content[0].text: {envelope!r}"
        ) from e
    return json.loads(text)


def _envelope_is_well_formed(envelope, expected_id):
    # Documented success shape from http_handler.py: jsonrpc=="2.0",
    # ``id`` echoed, ``result`` present (no ``error`` key). For
    # ``tools/call`` the result must carry the ``content`` list.
    if envelope.get("jsonrpc") != "2.0":
        return False, "jsonrpc != 2.0"
    if envelope.get("id") != expected_id:
        return False, f"id mismatch: {envelope.get('id')!r} != {expected_id!r}"
    if "error" in envelope:
        return False, f"error frame: {envelope['error']!r}"
    if "result" not in envelope:
        return False, "missing result"
    # Every result a 2026-07-28 server emits is typed, so the client can
    # tell a finished result from an interim one asking for more input.
    result_type = envelope["result"].get("resultType")
    if result_type != "complete":
        return False, f"resultType is {result_type!r}, expected 'complete'"
    return True, ""


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Wipe scene, drop a Plane that we'll later assign to the group.
    plane = dh.reset_scene_to_pinned_plane(name="Plane")

    # ----- start the MCP HTTP server on a free port --------------
    port = _alloc_free_port()
    mcp_mod = __import__(pkg + ".mcp.mcp_server",
                         fromlist=["start_mcp_server", "stop_mcp_server",
                                   "is_mcp_running", "get_mcp_server"])
    mcp_mod.start_mcp_server(port)
    actual_port = mcp_mod.get_mcp_server().port
    dh.log(f"mcp_started port={actual_port}")

    # Wait until the port stops accepting fresh binds (i.e. the listener
    # is up). ``is_port_available`` returns False when the listener is
    # holding the socket; we want that.
    server_utils = __import__(pkg + ".mcp.server_utils",
                              fromlist=["is_port_available"])
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not server_utils.is_port_available(actual_port):
            break
        time.sleep(0.1)
    if server_utils.is_port_available(actual_port):
        raise RuntimeError(f"MCP server never bound to {actual_port}")

    base_url = f"http://127.0.0.1:{actual_port}/mcp"
    schema_violations = []  # collected per-response so check C is precise

    # ----- 1. server/discover (no handshake, no session) ---------
    disc_resp = _drive_request_in_thread(
        base_url,
        {"jsonrpc": "2.0", "id": 1, "method": "server/discover"},
        headers={},
    )
    disc_env = _parse_jsonrpc(disc_resp)
    ok, why = _envelope_is_well_formed(disc_env, expected_id=1)
    if not ok:
        schema_violations.append(f"server/discover: {why}")
    disc_result = (disc_env or {}).get("result") or {}
    versions = disc_result.get("supportedVersions") or []
    if _PROTOCOL_VERSION not in versions:
        raise RuntimeError(
            f"server/discover does not serve {_PROTOCOL_VERSION}; "
            f"supportedVersions={versions!r}"
        )
    # A stateless transport mints nothing to carry between requests.
    minted = disc_resp["headers"].get("Mcp-Session-Id", "")
    retired_verb_status = {}
    for verb in ("GET", "DELETE"):
        req = urllib.request.Request(base_url, method=verb)
        req.add_header("Accept", "text/event-stream")
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                retired_verb_status[verb] = resp.status
        except urllib.error.HTTPError as e:
            retired_verb_status[verb] = e.code
        except Exception as e:
            raise RuntimeError(f"{verb} probe failed: {e}")
    verbs_retired = all(
        status == 405 for status in retired_verb_status.values()
    )
    result["checks"]["D_mcp_transport_is_stateless"] = {
        "ok": bool(versions) and not minted and verbs_retired,
        "details": {
            "supported_versions": versions,
            "session_id_minted": minted or None,
            "retired_verb_status": retired_verb_status,
        },
    }
    auth_headers = {}
    dh.log(f"discovered versions={versions}")

    # ----- 2. tools/call -> create_group -------------------------
    create_resp = _drive_request_in_thread(
        base_url,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "create_group", "arguments": {}},
        },
        headers=auth_headers,
    )
    create_env = _parse_jsonrpc(create_resp)
    ok, why = _envelope_is_well_formed(create_env, expected_id=2)
    if not ok:
        schema_violations.append(f"create_group: {why}")
    create_payload = _extract_tool_payload(create_env)
    if create_payload.get("status") != "success":
        raise RuntimeError(f"create_group failed: {create_payload!r}")
    group_uuid = create_payload.get("group_uuid", "")
    if not group_uuid:
        raise RuntimeError(
            f"create_group did not return group_uuid: {create_payload!r}"
        )
    dh.log(f"created group_uuid={group_uuid}")

    # ----- 3. tools/call -> rename_group("Cloth") ----------------
    rename_resp = _drive_request_in_thread(
        base_url,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "rename_group",
                "arguments": {"group_uuid": group_uuid, "name": "Cloth"},
            },
        },
        headers=auth_headers,
    )
    rename_env = _parse_jsonrpc(rename_resp)
    ok, why = _envelope_is_well_formed(rename_env, expected_id=3)
    if not ok:
        schema_violations.append(f"rename_group: {why}")
    rename_payload = _extract_tool_payload(rename_env)
    if rename_payload.get("status") != "success":
        raise RuntimeError(f"rename_group failed: {rename_payload!r}")

    # ----- 4. tools/call -> set_group_type("SHELL") --------------
    type_resp = _drive_request_in_thread(
        base_url,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "set_group_type",
                "arguments": {"group_uuid": group_uuid, "type": "SHELL"},
            },
        },
        headers=auth_headers,
    )
    type_env = _parse_jsonrpc(type_resp)
    ok, why = _envelope_is_well_formed(type_env, expected_id=4)
    if not ok:
        schema_violations.append(f"set_group_type: {why}")
    type_payload = _extract_tool_payload(type_env)
    if type_payload.get("status") != "success":
        raise RuntimeError(f"set_group_type failed: {type_payload!r}")

    # ----- A: addon state has one SHELL group named Cloth --------
    root = dh.groups.get_addon_data(bpy.context.scene)
    active_groups = []
    for grp in dh.groups.iterate_object_groups(bpy.context.scene) \
            if hasattr(dh.groups, "iterate_object_groups") else []:
        if grp.active:
            active_groups.append(grp)
    if not active_groups:
        # Fallback for a slimmer groups module: scan slot 0..N_MAX_GROUPS-1.
        N = getattr(dh.groups, "N_MAX_GROUPS", 16)
        for i in range(N):
            grp = getattr(root, f"object_group_{i}", None)
            if grp and grp.active:
                active_groups.append(grp)
    cloth_groups = [g for g in active_groups
                    if g.name == "Cloth" and g.object_type == "SHELL"]
    dh.record(
        "A_mcp_create_group_creates_group",
        len(active_groups) == 1
        and len(cloth_groups) == 1
        and cloth_groups[0].uuid == group_uuid,
        {
            "active_count": len(active_groups),
            "names": [g.name for g in active_groups],
            "types": [g.object_type for g in active_groups],
            "matched_uuid": cloth_groups[0].uuid if cloth_groups else None,
            "expected_uuid": group_uuid,
        },
    )

    # ----- 5. tools/call -> add_objects_to_group([Plane]) --------
    add_resp = _drive_request_in_thread(
        base_url,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "add_objects_to_group",
                "arguments": {
                    "group_uuid": group_uuid,
                    "object_names": [plane.name],
                },
            },
        },
        headers=auth_headers,
    )
    add_env = _parse_jsonrpc(add_resp)
    ok, why = _envelope_is_well_formed(add_env, expected_id=5)
    if not ok:
        schema_violations.append(f"add_objects_to_group: {why}")
    add_payload = _extract_tool_payload(add_env)
    if add_payload.get("status") != "success":
        raise RuntimeError(f"add_objects_to_group failed: {add_payload!r}")
    dh.log(f"added objects payload={add_payload!r}")

    # ----- B: group's assigned_objects carries Plane with UUID ---
    target = cloth_groups[0] if cloth_groups else None
    assigned = list(target.assigned_objects) if target else []
    plane_entries = [a for a in assigned if a.name == plane.name]
    plane_uuid = plane_entries[0].uuid if plane_entries else ""
    dh.record(
        "B_mcp_add_objects_assigns",
        len(assigned) == 1
        and len(plane_entries) == 1
        and bool(plane_uuid)
        and len(plane_uuid) > 0,
        {
            "assigned_names": [a.name for a in assigned],
            "assigned_uuids": [a.uuid for a in assigned],
            "plane_uuid": plane_uuid,
            "added_objects": add_payload.get("added_objects"),
        },
    )

    # ----- C: every response matched the documented schema -------
    dh.record(
        "C_mcp_responses_match_schema",
        not schema_violations
        and create_payload.get("status") == "success"
        and rename_payload.get("status") == "success"
        and type_payload.get("status") == "success"
        and add_payload.get("status") == "success",
        {
            "schema_violations": schema_violations,
            "create_status": create_payload.get("status"),
            "rename_status": rename_payload.get("status"),
            "type_status": type_payload.get("status"),
            "add_status": add_payload.get("status"),
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    # Stop the MCP server even if assertions or transport raised, so a
    # follow-up scenario in the same Blender process can rebind.
    try:
        mcp_mod = __import__(pkg + ".mcp.mcp_server",
                             fromlist=["stop_mcp_server", "is_mcp_running"])
        if mcp_mod.is_mcp_running():
            mcp_mod.stop_mcp_server()
    except Exception as exc:
        result["errors"].append(
            f"mcp_stop: {type(exc).__name__}: {exc}"
        )
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
