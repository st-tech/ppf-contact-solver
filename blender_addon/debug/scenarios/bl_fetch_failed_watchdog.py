# File: scenarios/bl_fetch_failed_watchdog.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Fetch failure + watchdog recovery (commit a8766a08).
#
# The fetch pipeline guarantees a terminal event on every code path:
# ``_do_fetch_map`` and ``_do_fetch_frames`` dispatch ``FetchFailed`` on
# precondition miss or exception, the apply side dispatches it on
# exception, and ``facade._watchdog_check`` dispatches it after
# ``_WATCHDOG_TIMEOUT_S`` of unchanged (activity, progress) while in
# FETCHING or APPLYING. The transition handler resets activity to IDLE,
# zeroes progress, sets ``state.error`` to the reason, clears the
# animation buffer, and redraws the UI so the user can recover.
#
# Subtests:
#   A. direct_fetchfailed_resets_to_idle
#         Inject activity=FETCHING by mutating the engine state directly,
#         dispatch ``FetchFailed`` with a known reason, tick the engine,
#         and assert: activity returns to IDLE, ``state.error`` carries
#         the reason, progress is zeroed, the server connection is still
#         alive (server=RUNNING).
#   B. recovery_after_fetchfailed
#         After A, run the standard build + run + fetch + drain pipeline
#         end to end. Assert applied frames equal total > 0, the PC2 file
#         exists, and ``state.error`` is empty (a clean fetch clears the
#         leftover error string from A).
#   C. watchdog_window_recovery
#         Inject activity=FETCHING and prime ``facade._stuck_since`` to a
#         monotonic timestamp older than ``_WATCHDOG_TIMEOUT_S``, then
#         call ``facade._watchdog_check`` directly and tick. Assert the
#         watchdog dispatched FetchFailed with ``"watchdog timeout"``,
#         activity is back to IDLE, and ``state.error`` reflects the
#         watchdog reason. This avoids a 30 s wall-clock wait while
#         exercising the same code path the persistent timer drives.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import time
import traceback
from dataclasses import replace

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


try:
    dh = DriverHelpers(pkg, result)
    state_mod = __import__(pkg + ".core.state",
                           fromlist=["AppState", "Activity", "Solver",
                                     "Server", "Phase"])
    Activity = state_mod.Activity
    Solver = state_mod.Solver
    Server = state_mod.Server
    facade_mod = __import__(pkg + ".core.facade",
                            fromlist=["_watchdog_check", "_watchdog_reset",
                                      "_WATCHDOG_TIMEOUT_S"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="FetchFailMesh")
    dh.save_blend(PROBE_DIR, "fetchfail.blend")
    root = dh.configure_state(project_name="fetch_failed_watchdog",
                              frame_count=6)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    s_pre = dh.facade.engine.state
    assert s_pre.server.name == "RUNNING", s_pre.server.name
    assert s_pre.activity.name == "IDLE", s_pre.activity.name

    # ----- A: direct FetchFailed resets activity to IDLE ----------
    # The transition handler is guarded on activity in (FETCHING, APPLYING),
    # so flip activity to FETCHING first via direct state mutation. This
    # mirrors what ``case FetchRequested`` produces, minus the spawned
    # DoFetchMap effect that we don't want firing during this subtest.
    direct_reason = "test:direct-fetchfailed"
    with dh.facade.engine._lock:
        dh.facade.engine._state = replace(
            dh.facade.engine._state,
            activity=Activity.FETCHING,
            progress=0.42,
            error="",
        )
    dh.facade.engine.dispatch(dh.events.FetchFailed(reason=direct_reason))
    dh.facade.tick()
    s_after_a = dh.facade.engine.state
    dh.record(
        "A_direct_fetchfailed_resets_to_idle",
        s_after_a.activity.name == "IDLE"
        and s_after_a.error == direct_reason
        and s_after_a.progress == 0.0
        and s_after_a.server.name == "RUNNING",
        {
            "activity": s_after_a.activity.name,
            "error": s_after_a.error,
            "progress": s_after_a.progress,
            "server": s_after_a.server.name,
        },
    )
    dh.log("A.done")

    # ----- B: recovery after FetchFailed --------------------------
    # A real build + run + fetch + drain must succeed cleanly. The
    # leftover ``state.error`` from A should be cleared by the regular
    # transition machinery (the connect/build success paths reset it).
    data_bytes, param_bytes = dh.encode_payload()
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="fetchfail:build",
    ))
    deadline = time.time() + 90.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")
    assert dh.facade.engine.state.solver.name in ("READY", "RESUMABLE")

    dh.run_and_wait(timeout=90.0)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.settle_idle(timeout=10.0)
    pc2_path = dh.find_pc2_for(plane)
    pc2_size = (
        os.path.getsize(pc2_path)
        if pc2_path and os.path.isfile(pc2_path) else 0
    )
    s_rec = dh.facade.engine.state
    dh.record(
        "B_recovery_after_fetchfailed",
        applied == total
        and total > 0
        and pc2_path is not None
        and pc2_size > 0
        and dh.has_mesh_cache(plane)
        and s_rec.error == ""
        and s_rec.activity.name == "IDLE",
        {
            "applied": applied,
            "total": total,
            "pc2_path": pc2_path,
            "pc2_size": pc2_size,
            "has_mesh_cache": dh.has_mesh_cache(plane),
            "error": s_rec.error,
            "activity": s_rec.activity.name,
        },
    )
    dh.log(f"B.done applied={applied}/{total} pc2_size={pc2_size}")

    # ----- C: watchdog dispatches FetchFailed on a stalled fetch --
    # Inject activity=FETCHING with a non-zero progress, prime the
    # watchdog's bookkeeping so the next check sees the snapshot as
    # "stuck for longer than the timeout", then drive ``_watchdog_check``
    # directly. The check should dispatch FetchFailed("watchdog timeout"),
    # which the next tick converts into activity=IDLE plus error.
    facade_mod._watchdog_reset()
    with dh.facade.engine._lock:
        dh.facade.engine._state = replace(
            dh.facade.engine._state,
            activity=Activity.FETCHING,
            progress=0.5,
            error="",
        )
    # Seed _last_progress_key so the second check sees an unchanged
    # snapshot, then backdate _stuck_since past the timeout window.
    facade_mod._watchdog_check()
    timeout_s = float(facade_mod._WATCHDOG_TIMEOUT_S)
    facade_mod._stuck_since = time.monotonic() - (timeout_s + 1.0)
    facade_mod._watchdog_check()
    dh.facade.tick()
    s_after_c = dh.facade.engine.state
    dh.record(
        "C_watchdog_window_recovery",
        s_after_c.activity.name == "IDLE"
        and s_after_c.error == "watchdog timeout"
        and s_after_c.progress == 0.0
        and s_after_c.server.name == "RUNNING",
        {
            "activity": s_after_c.activity.name,
            "error": s_after_c.error,
            "progress": s_after_c.progress,
            "server": s_after_c.server.name,
            "watchdog_timeout_s": timeout_s,
        },
    )
    dh.log("C.done")

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
    return r.report_named_checks(result.get("checks", {}))
