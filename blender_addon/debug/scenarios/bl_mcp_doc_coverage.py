# File: scenarios/bl_mcp_doc_coverage.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The bundled tool reference documents exactly the tools the server serves.
#
# ``llm://mcp_tools_reference`` is the authoritative tool list an agent reads,
# and it is maintained by hand while the tool surface is generated from the
# handler registry. The two drift in both directions: a new handler ships
# undocumented, or an entry outlives the handler it describes. An agent that
# trusts the document then calls a tool that does not exist, or never learns
# about one that does.
#
# This scenario reads the document over MCP, the same way an agent would, and
# compares it against ``tools/list`` from the same server, so neither kind of
# drift can ship.
#
# Assertions:
#   A. ``every_tool_is_documented`` -- every name in tools/list has a heading
#      in the reference.
#   B. ``no_phantom_entries`` -- every heading in the reference names a tool
#      that tools/list actually serves.
#   C. ``reference_describes_the_current_transport`` -- the document does not
#      instruct a reader to perform the removed initialize handshake or to
#      echo a session id.
#   D. ``every_llm_document_is_readable`` -- every llm:// URI that
#      resources/list advertises reads back as non-empty markdown, so a
#      pointer in a tool description cannot lead nowhere.

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
import re

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # The tool surface, from the server.
    env, _resp = mcp_call(pkg, url, "tools/list", request_id=1)
    tools = (env.get("result") or {}).get("tools") or []
    served = sorted(t["name"] for t in tools)

    # The reference, read the way an agent reads it.
    env, _resp = mcp_call(
        pkg, url, "resources/read",
        {"uri": "llm://mcp_tools_reference"}, request_id=2,
    )
    contents = (env.get("result") or {}).get("contents") or []
    doc = contents[0].get("text", "") if contents else ""
    documented = set(re.findall(r"^### `?([a-z_0-9]+)`?", doc, re.M))

    missing = sorted(set(served) - documented)
    phantom = sorted(documented - set(served))

    mcp_check(
        result, "A_every_tool_is_documented",
        bool(served) and bool(doc) and not missing,
        {
            "served_count": len(served),
            "documented_count": len(documented),
            "missing_from_reference": missing[:20],
            "missing_total": len(missing),
        },
    )
    mcp_check(
        result, "B_no_phantom_entries",
        bool(doc) and not phantom,
        {"phantom_entries": phantom[:20], "phantom_total": len(phantom)},
    )

    # The transport has no handshake, so the reference must not instruct the
    # reader through one. Anything that matches here is an instruction an
    # agent would follow into a rejected request.
    stale = []
    for phrase in ("initialize` handshake", "Mcp-Session-Id` to echo",
                   "echo it on every subsequent request"):
        if phrase in doc:
            stale.append(phrase)
    mcp_check(
        result, "C_reference_describes_the_current_transport",
        not stale,
        {"stale_phrases": stale},
    )

    # Every llm:// pointer a tool description hands out must resolve.
    env, _resp = mcp_call(pkg, url, "resources/list", request_id=3)
    resources = (env.get("result") or {}).get("resources") or []
    llm_uris = [x["uri"] for x in resources if x.get("uri", "").startswith("llm://")]
    unreadable = []
    for uri in llm_uris:
        env, _resp = mcp_call(
            pkg, url, "resources/read", {"uri": uri}, request_id=4
        )
        body = (env.get("result") or {}).get("contents") or []
        text = body[0].get("text", "") if body else ""
        mime = body[0].get("mimeType", "") if body else ""
        if not text.strip() or mime != "text/markdown":
            unreadable.append({"uri": uri, "mime": mime, "chars": len(text)})
    mcp_check(
        result, "D_every_llm_document_is_readable",
        bool(llm_uris) and not unreadable,
        {"llm_uri_count": len(llm_uris), "unreadable": unreadable},
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
