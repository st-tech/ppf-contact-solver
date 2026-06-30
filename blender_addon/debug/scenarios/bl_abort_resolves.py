# File: scenarios/bl_abort_resolves.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard for the stuck-"Aborting..." deadlock.
#
# ``com.abort()`` sets ``activity = ABORTING`` and the ServerPolled handler
# clears it back to IDLE once a status response shows the solver in a
# terminal phase (READY / RESUMABLE / FAILED / NO_DATA / NO_BUILD). The bug:
# the PollTick handler only re-queried while the SOLVER was non-terminal
# (RUNNING / STARTING / SAVING / BUILDING). Aborting a fetch / send / apply
# (where the solver stays READY the whole time), or a build that finished
# just before the abort, therefore left ABORTING with no reason to poll
# again -- so the engine sat in ABORTING forever and ``com.busy()`` gated
# every button (Fetch All Animation, Clear Local Animation, ...) off.
#
# The fix makes PollTick re-query whenever ``activity == ABORTING``. This
# scenario locks that behavior on the pure (side-effect-free) transition
# function:
#   A. aborting_terminal_solver_repolls: ABORTING + solver READY -> PollTick
#      emits a DoQuery (the regression; it emitted nothing before).
#   B. aborting_running_solver_repolls: ABORTING + solver RUNNING -> DoQuery
#      (unchanged path, still polls).
#   C. idle_terminal_solver_no_spurious_poll: IDLE + solver READY -> PollTick
#      emits NO DoQuery (the fix is scoped to ABORTING; idle does not spam).
#   D. aborting_resolves_on_ready_poll: feeding the ABORTING state a
#      ServerPolled whose status is READY clears activity back to IDLE.
#
# Assertion-only: exercises the pure reducer, so no server or solve.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    from dataclasses import replace
    tr = __import__(pkg + ".core.transitions", fromlist=["transition"])
    events = __import__(pkg + ".core.events",
                        fromlist=["PollTick", "ServerPolled"])
    stm = __import__(pkg + ".core.state",
                     fromlist=["Activity", "Solver", "Server", "Phase",
                               "AppState"])
    eff = __import__(pkg + ".core.effects", fromlist=["DoQuery"])
    proto = __import__(pkg + ".core.protocol",
                       fromlist=["PROTOCOL_VERSION", "STATUS_READY"])
    facade = __import__(pkg + ".core.facade", fromlist=["engine"])

    base = facade.engine.state

    def has_query(effects):
        return any(type(e).__name__ == "DoQuery" for e in effects)

    online = dict(server=stm.Server.RUNNING, phase=stm.Phase.ONLINE)

    # A. ABORTING + terminal solver must re-poll (the regression).
    s_abort_ready = replace(base, activity=stm.Activity.ABORTING,
                            solver=stm.Solver.READY, **online)
    _, eff_a = tr.transition(s_abort_ready, events.PollTick())
    record("A_aborting_terminal_solver_repolls", has_query(eff_a),
           {"effects": [type(e).__name__ for e in eff_a]})

    # B. ABORTING + running solver still re-polls (unchanged path).
    s_abort_run = replace(base, activity=stm.Activity.ABORTING,
                          solver=stm.Solver.RUNNING, **online)
    _, eff_b = tr.transition(s_abort_run, events.PollTick())
    record("B_aborting_running_solver_repolls", has_query(eff_b),
           {"effects": [type(e).__name__ for e in eff_b]})

    # C. IDLE + terminal solver must NOT re-poll (fix scoped to ABORTING).
    s_idle_ready = replace(base, activity=stm.Activity.IDLE,
                           solver=stm.Solver.READY, **online)
    _, eff_c = tr.transition(s_idle_ready, events.PollTick())
    record("C_idle_terminal_solver_no_spurious_poll", not has_query(eff_c),
           {"effects": [type(e).__name__ for e in eff_c]})

    # D. A READY status response clears ABORTING back to IDLE.
    ready_resp = {
        "status": proto.STATUS_READY,
        "protocol_version": proto.PROTOCOL_VERSION,
        "error": "",
        "initialized": True,
        "frame": 0,
        "root": "/tmp/_rig_abort",
        "upload_id": "rig-abort",
    }
    s_after, _ = tr.transition(s_abort_ready,
                               events.ServerPolled(response=ready_resp))
    record("D_aborting_resolves_on_ready_poll",
           s_after.activity == stm.Activity.IDLE,
           {"activity_after": s_after.activity.name})

    result["phases"].append((round(time.time(), 3),
                             "checks=" + str(len(result["checks"]))))
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
