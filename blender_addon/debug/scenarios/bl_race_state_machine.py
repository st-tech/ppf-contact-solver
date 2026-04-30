# File: scenarios/bl_race_state_machine.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Race / guard-rejection coverage for the addon's state machine.
# Each subtest dispatches an event sequence the guards are supposed
# to reject (or coalesce) and asserts the state didn't slip into an
# inconsistent shape. These are the bug classes behind commits like
# "Redesign upload+build pipeline around upload_id to kill stuck-state
# bug" and the live-fetch / explicit-fetch race the pin fidelity
# suite exposed.
#
# Subtests:
#   A. fetch_while_busy_rejected
#   B. build_while_busy_rejected
#   C. abort_during_build_clears_pending
#   D. spam_fetch_coalesced

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import time as _time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("subtests", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


def drain_to_solver_terminal(dh, *, timeout):
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if dh.facade.engine.state.solver.name in ("READY", "RESUMABLE", "FAILED"):
            return
        _time.sleep(0.3)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="RaceMesh")
    dh.save_blend(PROBE_DIR, "race.blend")
    root = dh.configure_state(project_name="race_state_machine", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log(f"connected phase={dh.facade.engine.state.phase.name}")

    # ----- Subtest A: fetch-while-busy rejected ------------------
    # Queue BuildPipelineRequested then FetchRequested; tick once to
    # process them in order. The fetch's ``not state.busy`` guard
    # rejects, so final activity is whatever the build settled on,
    # never FETCHING.
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes, message="race-A"))
    dh.facade.engine.dispatch(dh.events.FetchRequested())
    dh.facade.tick()
    final_activity_a = dh.facade.engine.state.activity.name
    dh.record_subtest(
        "A_fetch_while_busy_rejected",
        final_activity_a in ("SENDING", "BUILDING"),
        {"final_activity": final_activity_a},
    )
    drain_to_solver_terminal(dh, timeout=90.0)

    # ----- Subtest B: build-while-busy rejected -----------------
    # Second BuildPipelineRequested must be rejected by
    # ``state.can_operate`` since the first set activity != IDLE.
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes, message="race-B-1"))
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes, message="race-B-2"))
    dh.facade.tick()
    final_activity_b = dh.facade.engine.state.activity.name
    dh.record_subtest(
        "B_build_while_busy_rejected",
        final_activity_b in ("SENDING", "BUILDING"),
        {
            "final_activity": final_activity_b,
            "final_pending": dh.facade.engine.state.pending_build,
        },
    )
    drain_to_solver_terminal(dh, timeout=90.0)

    # ----- Subtest C: abort_during_build_clears_pending ----------
    # AbortRequested has no guard and runs second; final state must
    # reflect ABORTING with pending_build=False (the build's
    # pending_build=True is overwritten by abort).
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes, message="race-C"))
    dh.facade.engine.dispatch(dh.events.AbortRequested())
    dh.facade.tick()
    dh.record_subtest(
        "C_abort_during_build_clears_pending",
        dh.facade.engine.state.activity.name == "ABORTING"
        and dh.facade.engine.state.pending_build is False,
        {
            "post_activity": dh.facade.engine.state.activity.name,
            "post_pending": dh.facade.engine.state.pending_build,
        },
    )
    dh.settle_idle(timeout=30.0)

    # ----- Subtest D: spam_fetch_coalesced -----------------------
    # Get into a state with frames available, then fire FetchRequested
    # 5 times back-to-back. Only the first should escape the busy
    # guard; the rest are silently dropped.
    if dh.facade.engine.state.solver.name not in ("READY", "RESUMABLE"):
        dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
            data=data_bytes, param=param_bytes, message="race-D-build"))
        drain_to_solver_terminal(dh, timeout=90.0)
    dh.run_and_wait(timeout=60.0)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)

    dh.facade.runner.clear_fetched_frames()
    for _ in range(5):
        dh.facade.engine.dispatch(dh.events.FetchRequested())
    dh.facade.tick()
    final_activity_d = dh.facade.engine.state.activity.name
    dh.record_subtest(
        "D_spam_fetch_coalesced",
        final_activity_d == "FETCHING",
        {"final_activity": final_activity_d},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("subtests", {}), label="subtests")
