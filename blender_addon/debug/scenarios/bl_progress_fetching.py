# File: scenarios/bl_progress_fetching.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# UX progress: ``Fetching Animation`` progress bar.
#
# After a run completes, the user clicks "Fetch All Animation" and the
# panel shows a progress bar that fills as frames stream down. The bar
# is driven by ``ProgressUpdated`` events dispatched from the I/O
# worker as each chunk lands while activity == FETCHING.
#
# This scenario asserts that during the FETCHING activity at least one
# ``ProgressUpdated`` event with progress > 0 is dispatched and that
# the sequence is non-decreasing. The regression we want to catch is a
# silent UX freeze where the bar stays stuck at 0 the whole download
# even though frames are landing.
#
# The follow-up APPLYING phase isn't asserted: in a batch fetch the
# main-thread ``apply_animation`` call drains every queued frame in a
# single while-loop, so ``last_applied >= last_total`` on its first
# tick and the partial-progress dispatch path inside
# ``client.apply_animation`` is skipped (``AllFramesApplied`` fires
# instead). The APPLYING bar is only meaningfully observable during
# live-fetch streaming, which ``bl_live_frame_end_tracking`` covers.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Same parallel-mode race as bl_live_frame_end_tracking: under load
# the modal can't drain mid-driver and we can miss progress samples.
NOT_PARALLELIZABLE = True


_DRIVER_BODY = r"""
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 12


try:
    dh = DriverHelpers(pkg, result)
    client_mod = __import__(pkg + ".core.client",
                            fromlist=["apply_animation"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    plane = dh.reset_scene_to_pinned_plane(name="ProgressFetchMesh")
    dh.save_blend(PROBE_DIR, "progress_fetching.blend")
    root = dh.configure_state(project_name="progress_fetching",
                              frame_count=FRAME_COUNT)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=FRAME_COUNT - 1,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="progress_fetch:build",
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
    assert dh.facade.engine.state.solver.name in ("READY", "RESUMABLE")

    # Run the simulation to completion. We don't sample progress here
    # (bl_progress_simulating covers that path) -- we just need frames
    # on the server's filesystem so the fetch path has something to
    # download.
    dh.com.run()
    deadline = time.time() + 120.0
    saw_running = False
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        if saw_running and s.solver.name in ("READY", "RESUMABLE", "FAILED"):
            break
        time.sleep(0.2)
    assert dh.facade.engine.state.solver.name in ("READY", "RESUMABLE")

    # Reset the runner's fetched-frame list so DoFetchFrames downloads
    # all FRAME_COUNT-1 frames from scratch (otherwise it short-circuits
    # because the live fetch already populated the in-memory list).
    dh.facade.runner.clear_fetched_frames()

    # Hook ``engine.dispatch`` to record every ``ProgressUpdated`` plus
    # the activity that produced it. Polling state.progress at the
    # FETCHING cadence misses every meaningful value when the fetch
    # completes between two ticks (FRAME_COUNT=12 over loopback is
    # near-instantaneous), so we instead capture each event off the
    # dispatch path. The activity tag distinguishes the FETCHING
    # download bar from the APPLYING write-back bar.
    captured = []  # list[(activity_name, progress, traffic)]
    engine = dh.facade.engine
    original_dispatch = engine.dispatch

    def hook_dispatch(event):
        if type(event).__name__ == "ProgressUpdated":
            captured.append((
                engine.state.activity.name,
                float(getattr(event, "progress", 0.0) or 0.0),
                str(getattr(event, "traffic", "") or ""),
            ))
        return original_dispatch(event)

    engine.dispatch = hook_dispatch

    try:
        dh.com.fetch()
        saw_fetching = False
        deadline = time.time() + 60.0
        while time.time() < deadline:
            engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            # Apply queued frames so the FETCHING -> APPLYING transition
            # eventually drains and we can break out (the FramePump
            # TIMER can't fire while we hold the main thread).
            try:
                client_mod.apply_animation()
            except Exception:
                pass
            act = engine.state.activity.name
            if act == "FETCHING":
                saw_fetching = True
            elif saw_fetching and act == "IDLE":
                break
            time.sleep(0.05)
    finally:
        engine.dispatch = original_dispatch

    fetch_progresses = [p for (a, p, _t) in captured if a == "FETCHING"]
    fetch_nonzero = [p for p in fetch_progresses if p > 0.0]
    fetch_monotonic = all(fetch_progresses[i] <= fetch_progresses[i + 1]
                          for i in range(len(fetch_progresses) - 1))
    fetch_grew = bool(fetch_nonzero) and (fetch_nonzero[-1] > fetch_nonzero[0]
                                          if len(fetch_nonzero) >= 2
                                          else fetch_nonzero[-1] > 0.0)

    dh.record(
        "A_fetch_progress_advances_during_download",
        saw_fetching and bool(fetch_nonzero) and fetch_monotonic and fetch_grew,
        {
            "saw_fetching": saw_fetching,
            "fetch_event_count": len(fetch_progresses),
            "fetch_first_nonzero": fetch_nonzero[0] if fetch_nonzero else None,
            "fetch_last": fetch_progresses[-1] if fetch_progresses else None,
            "fetch_monotonic": fetch_monotonic,
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = REPO_ROOT_POSIX
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
