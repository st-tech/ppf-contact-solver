# File: scenarios/bl_solver_crash_kind_localized.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The server names a crash cause with a stable snake_case tag (`crash_kind`,
# `CrashKind::tag()` on the Rust side) and sends the full multi-line report
# separately in `error`. The addon turns the tag into a localized one-line
# panel headline and leaves the report to the Console.
#
# Two ways that arrangement breaks silently, and one subtest for each:
#
#   A. tag_reaches_app_state: `_interpret_response` must read `crash_kind`
#      off the same response it reads `error` off. A response that carries a
#      cause and an addon that drops it produces a panel showing a truncated
#      blob again, which is the reported symptom.
#   B. tag_cleared_with_error: a later clean response must clear the tag. A
#      tag outliving its message would name a failure the panel can no
#      longer show.
#   C. every_tag_has_a_catalog_key: every value of CRASH_CAUSE_SUMMARY, plus
#      the fallback, is a key in i18n/en.json. A summary with no key renders
#      as untranslated English in every locale, and the gap is invisible in
#      an English UI, so only a check can catch it.
#   D. unknown_tag_falls_back: a tag this build does not recognize (a newer
#      server naming a newer cause) still produces a headline instead of a
#      blank one.
#   E. missing_field_is_empty: a server that predates the field leaves the
#      tag empty, which is what makes the panel keep its raw-error path.
#
# Pure UI scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import json, os, time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    transitions = __import__(pkg + ".core.transitions",
                             fromlist=["_interpret_response"])
    state_mod = __import__(pkg + ".core.state", fromlist=["AppState"])
    status_mod = __import__(pkg + ".core.status",
                            fromlist=["CRASH_CAUSE_SUMMARY"])
    protocol_mod = __import__(pkg + ".core.protocol",
                              fromlist=["PROTOCOL_VERSION"])
    i18n_mod = __import__(pkg + ".i18n", fromlist=["__file__"])

    AppState = state_mod.AppState
    interpret = transitions._interpret_response

    def response(status, error, **extra):
        base = {
            "status": status,
            "error": error,
            # Read from the addon's own single source, so a version bump
            # cannot silently turn every case below into the
            # version-mismatch early-out and pass for the wrong reason.
            "protocol_version": protocol_mod.PROTOCOL_VERSION,
            "upload_id": "",
            "violations": [],
            "frame": 0,
        }
        base.update(extra)
        return base

    # ----- A: the tag reaches AppState alongside the report ------------
    report = (
        "Solver stopped on a failed internal check: PPF FATAL: aggregate "
        "lock Gram eigensolve failed.\n"
        "--- Solver Errors (last 32 lines) ---\n"
        "PPF FATAL: aggregate lock Gram eigensolve failed."
    )
    crashed, _ = interpret(
        AppState(),
        response("FAILED", report, crash_kind="solver_invariant"),
    )
    record(
        "A_tag_reaches_app_state",
        crashed.crash_kind == "solver_invariant"
        and crashed.server_error == report,
        {"crash_kind": crashed.crash_kind},
    )

    # ----- B: a later clean response clears the tag --------------------
    cleared, _ = interpret(crashed, response("READY", ""))
    record(
        "B_tag_cleared_with_error",
        cleared.crash_kind == "" and cleared.server_error == "",
        {"crash_kind": cleared.crash_kind,
         "server_error": cleared.server_error},
    )

    # ----- C: every summary is a translation key -----------------------
    en_path = os.path.join(os.path.dirname(i18n_mod.__file__), "en.json")
    with open(en_path, encoding="utf-8") as f:
        catalog = json.load(f)
    keys = set(k for k in catalog if k != "_meta")
    summaries = list(status_mod.CRASH_CAUSE_SUMMARY.values())
    summaries.append(status_mod.CRASH_CAUSE_FALLBACK)
    summaries.append("Solver failed: {cause}")
    missing = sorted(s for s in summaries if s not in keys)
    record(
        "C_every_tag_has_a_catalog_key",
        not missing,
        {"checked": len(summaries), "missing": missing},
    )

    # ----- D: an unrecognized tag still yields a headline ---------------
    fallback = status_mod.crash_cause_summary("a_cause_from_a_newer_server")
    record(
        "D_unknown_tag_falls_back",
        fallback == status_mod.CRASH_CAUSE_FALLBACK and bool(fallback),
        {"fallback": fallback},
    )

    # ----- E: a server without the field leaves the tag empty ----------
    legacy, _ = interpret(AppState(), response("FAILED", "some failure"))
    record(
        "E_missing_field_is_empty",
        legacy.crash_kind == "" and legacy.server_error == "some failure",
        {"crash_kind": legacy.crash_kind},
    )

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
