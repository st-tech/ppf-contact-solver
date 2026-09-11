# File: scenarios/bl_mcp_prompts.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP prompts capability, against a real Blender.
#
# The server advertises a ``prompts`` capability, which obliges it to answer
# ``prompts/list`` and ``prompts/get``. This scenario drives both against the
# add-on running inside Blender, with the registry loaded exactly as a client
# finds it.
#
# The coverage is DERIVED from ``prompts/list``: every prompt the server
# advertises is fetched, rendered and probed, so a prompt added to
# ``mcp/prompts.py`` is covered without editing this file. Each argument is
# supplied a sentinel value that no template contains on its own, which is what
# makes "the caller's value reached the rendered text" checkable rather than
# assumed.
#
# Assertions:
#   A. ``prompts_list_is_populated_and_cacheable`` -- prompts/list answers with
#      a non-empty list in a typed result carrying ttlMs and cacheScope
#      "public", the scope for content identical for every caller.
#   B. ``entries_declare_name_title_description_and_arguments`` -- every entry
#      carries a non-empty name, title and description plus an arguments array
#      whose members each name themselves and describe themselves, and the
#      names are unique and in sorted order.
#   C. ``every_prompt_renders_a_user_message`` -- prompts/get with every
#      declared argument supplied returns messages whose first entry is a
#      "user" role text message with non-empty text, and echoes the same
#      description the list advertised.
#   D. ``supplied_arguments_reach_the_rendered_text`` -- every required
#      argument's value appears verbatim in the rendered text, and at least one
#      supplied value does for a prompt whose arguments are all optional.
#   E. ``missing_required_argument_is_refused`` -- prompts/get without a
#      required argument is 400 with -32602 whose message names the argument,
#      rather than a message rendered with a hole in it.
#   F. ``unknown_prompt_name_is_refused`` -- an unlisted name is 400 with
#      -32602 whose message quotes the name asked for.
#   G. ``non_object_arguments_are_refused`` -- ``arguments`` that is not an
#      object is 400 with -32602, checked before any rendering.

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

# One sentinel per argument name. The suffix keeps the value out of any
# template's own prose, so a match in a rendered message can only come from
# the argument this driver supplied.
SENTINEL = "zz-probe-%s-4f19"
UNLISTED_NAME = "no_such_prompt_zz4f19"

try:
    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. the advertised list ---------------------------------
    env, resp = mcp_call(pkg, url, "prompts/list", request_id=1)
    res = env.get("result") or {}
    listed = res.get("prompts")
    ok, why = mcp_envelope_ok(env, 1)
    mcp_check(
        result, "A_prompts_list_is_populated_and_cacheable",
        ok
        and isinstance(listed, list)
        and len(listed) > 0
        and isinstance(res.get("ttlMs"), int)
        and res.get("ttlMs") >= 0
        and res.get("cacheScope") == "public",
        {
            "status": resp["status"],
            "envelope_ok": ok,
            "why": why,
            "prompt_count": len(listed) if isinstance(listed, list) else None,
            "ttlMs": res.get("ttlMs"),
            "cacheScope": res.get("cacheScope"),
        },
    )
    if not isinstance(listed, list):
        listed = []
    result["phases"].append((time.time(), "listed %d prompts" % len(listed)))

    # ----- B. the shape of every advertised entry -----------------
    shape_faults = []
    names = []
    for entry in listed:
        if not isinstance(entry, dict):
            shape_faults.append(["<not-an-object>", repr(entry)[:100]])
            continue
        name = entry.get("name")
        names.append(name)
        for field in ("name", "title", "description"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                shape_faults.append([name, "%s is %r" % (field, value)])
        arguments = entry.get("arguments")
        if not isinstance(arguments, list):
            shape_faults.append([name, "arguments is %r" % (arguments,)])
            continue
        for argument in arguments:
            if not isinstance(argument, dict):
                shape_faults.append([name, "argument is %r" % (argument,)])
                continue
            arg_name = argument.get("name")
            if not isinstance(arg_name, str) or not arg_name.strip():
                shape_faults.append([name, "argument name is %r" % (arg_name,)])
                continue
            description = argument.get("description")
            if not isinstance(description, str) or not description.strip():
                shape_faults.append(
                    [name, "%s: description is %r" % (arg_name, description)]
                )
            if not isinstance(argument.get("required", False), bool):
                shape_faults.append(
                    [name, "%s: required is %r" % (arg_name, argument.get("required"))]
                )
    mcp_check(
        result, "B_entries_declare_name_title_description_and_arguments",
        not shape_faults
        and len(names) == len(listed)
        and len(set(names)) == len(names)
        and names == sorted(names),
        {
            "names": names,
            "unique": len(set(names)) == len(names),
            "sorted": names == sorted(names),
            "faults": shape_faults[:10],
        },
    )

    # ----- C and D. render every advertised prompt ----------------
    render_faults = []
    reach_faults = []
    rendered = {}
    reached = {}
    request_id = 10
    for entry in listed:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        name = entry["name"]
        arguments = entry.get("arguments")
        if not isinstance(arguments, list):
            arguments = []
        supplied = {}
        required = []
        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            arg_name = argument.get("name")
            if not isinstance(arg_name, str) or not arg_name:
                continue
            supplied[arg_name] = SENTINEL % arg_name
            if argument.get("required"):
                required.append(arg_name)

        request_id += 1
        env, resp = mcp_call(
            pkg, url, "prompts/get",
            {"name": name, "arguments": supplied},
            request_id=request_id,
        )
        ok, why = mcp_envelope_ok(env, request_id)
        if not ok:
            render_faults.append([name, why, resp["status"]])
            continue
        res = env["result"]
        if res.get("description") != entry.get("description"):
            render_faults.append(
                [name, "description differs from the listed one: %r"
                 % (res.get("description"),)]
            )
        messages = res.get("messages")
        if not isinstance(messages, list) or not messages:
            render_faults.append([name, "messages is %r" % (messages,)])
            continue
        first = messages[0]
        if not isinstance(first, dict):
            render_faults.append([name, "first message is %r" % (first,)])
            continue
        content = first.get("content")
        if not isinstance(content, dict):
            content = {}
        text = content.get("text")
        if first.get("role") != "user":
            render_faults.append([name, "role is %r" % (first.get("role"),)])
        if content.get("type") != "text":
            render_faults.append([name, "content type is %r" % (content.get("type"),)])
        if not isinstance(text, str) or not text.strip():
            render_faults.append([name, "text is %r" % (text,)])
            continue
        rendered[name] = {"messages": len(messages), "text_chars": len(text)}

        # A required argument the render ignores would leave the caller no way
        # to tell the value was dropped, so every one has to appear; a prompt
        # whose arguments are all optional has to spend at least one of them.
        absent = [arg for arg in required if supplied[arg] not in text]
        spent = [arg for arg, value in supplied.items() if value in text]
        reached[name] = sorted(spent)
        if absent or (supplied and not spent):
            reach_faults.append(
                [name, {"absent_required": absent, "spent": sorted(spent),
                        "supplied": sorted(supplied)}]
            )
    mcp_check(
        result, "C_every_prompt_renders_a_user_message",
        not render_faults and len(rendered) == len(listed) and len(rendered) > 0,
        {"rendered": rendered, "faults": render_faults[:10]},
    )
    mcp_check(
        result, "D_supplied_arguments_reach_the_rendered_text",
        not reach_faults and len(rendered) > 0,
        {"spent_values": reached, "faults": reach_faults[:10]},
    )

    # ----- E. a required argument left out ------------------------
    refusal_faults = []
    probed = []
    for entry in listed:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        arguments = entry.get("arguments")
        if not isinstance(arguments, list):
            continue
        required = [
            argument.get("name")
            for argument in arguments
            if isinstance(argument, dict) and argument.get("required")
        ]
        if not required:
            continue
        name = entry["name"]
        probed.append(name)
        request_id += 1
        env, resp = mcp_call(
            pkg, url, "prompts/get",
            {"name": name, "arguments": {}},
            request_id=request_id,
        )
        err = env.get("error") or {}
        message = err.get("message") or ""
        if resp["status"] != 400 or err.get("code") != -32602:
            refusal_faults.append([name, resp["status"], err])
        elif [arg for arg in required if arg not in message]:
            refusal_faults.append([name, "message names no missing argument", message])
    mcp_check(
        result, "E_missing_required_argument_is_refused",
        not refusal_faults and len(probed) > 0,
        {"probed": probed, "faults": refusal_faults[:10]},
    )

    # ----- F. a name the server never advertised ------------------
    request_id += 1
    env, resp = mcp_call(
        pkg, url, "prompts/get",
        {"name": UNLISTED_NAME, "arguments": {}},
        request_id=request_id,
    )
    err = env.get("error") or {}
    mcp_check(
        result, "F_unknown_prompt_name_is_refused",
        resp["status"] == 400
        and err.get("code") == -32602
        and UNLISTED_NAME in (err.get("message") or ""),
        {"status": resp["status"], "error": err},
    )

    # ----- G. arguments that are not an object --------------------
    probe_name = names[0] if names and isinstance(names[0], str) else UNLISTED_NAME
    request_id += 1
    env, resp = mcp_call(
        pkg, url, "prompts/get",
        {"name": probe_name, "arguments": ["not", "an", "object"]},
        request_id=request_id,
    )
    err = env.get("error") or {}
    mcp_check(
        result, "G_non_object_arguments_are_refused",
        resp["status"] == 400 and err.get("code") == -32602,
        {"name": probe_name, "status": resp["status"], "error": err},
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
