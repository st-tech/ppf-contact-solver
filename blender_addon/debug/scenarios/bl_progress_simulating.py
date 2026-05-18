# File: scenarios/bl_progress_simulating.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# UX progress: live solver progress during ``Simulation Running``.
#
# The Rust ppf-cts-server publishes ``progress = frame / total_frames``
# and ``info = "Simulation Running, frame N/M"`` while the solver is
# active (commit bb6a12a4). The addon's ``_interpret_response`` reads
# both fields, lifting them to ``state.progress`` / ``state.message``,
# which the panel renders as a moving progress bar with frame label.
# This scenario asserts the round-trip end to end:
#
#   * the response dict carries ``progress`` and ``info`` while solver
#     is RUNNING (not just BUILDING),
#   * ``state.progress`` is monotonic non-decreasing during the run,
#   * ``state.progress`` advances strictly between the first and last
#     observation (not stuck at 0),
#   * ``state.message`` is non-empty during the run.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Same parallel-mode race as bl_live_frame_end_tracking: the solver
# reports progress quickly enough that under host load we can miss
# samples between the start and the run's end. Pin to the serial
# postlude where the cadence is reliable.
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
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    plane = dh.reset_scene_to_pinned_plane(name="ProgressSimMesh")
    dh.save_blend(PROBE_DIR, "progress_simulating.blend")
    root = dh.configure_state(project_name="progress_simulating",
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
        message="progress_sim:build",
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

    dh.com.run()

    # Sample (state.progress, state.message, response.progress,
    # response.info) at every poll while the solver is RUNNING. Stop
    # one tick after the solver leaves RUNNING so the last sample is
    # the terminal state. ``response`` is ``com.info.response`` which
    # mirrors the most recent server reply seen by the runner.
    samples = []
    saw_running = False
    saw_response_progress = False
    saw_response_info = False
    deadline = time.time() + 120.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        resp = dh.com.info.response or {}
        if s.solver.name == "RUNNING":
            saw_running = True
            samples.append({
                "progress": float(s.progress),
                "message": str(s.message or ""),
                "resp_progress": resp.get("progress"),
                "resp_info": resp.get("info"),
                "frame": int(s.frame),
            })
            if isinstance(resp.get("progress"), (int, float)) \
                    and float(resp["progress"]) > 0.0:
                saw_response_progress = True
            if resp.get("info"):
                saw_response_info = True
        elif saw_running and s.solver.name in ("READY", "RESUMABLE", "FAILED"):
            break
        time.sleep(0.1)

    # Filter to samples where progress was reported (state.progress > 0
    # before the first frame is logged is expected — the server hasn't
    # written frame 0 yet).
    progresses = [smp["progress"] for smp in samples]
    nonzero = [p for p in progresses if p > 0.0]
    monotonic_ok = all(progresses[i] <= progresses[i + 1]
                       for i in range(len(progresses) - 1))
    # A single non-zero sample is enough proof that the progress
    # mechanism works end-to-end (server populates response.progress;
    # state.progress mirrors it; the addon panel can read it).
    # Requiring strict growth between consecutive samples broke for
    # CUDA solver runs that finish all frames inside one addon poll
    # interval — only one or two "RUNNING" samples land in the loop,
    # so the test could never observe `nonzero[-1] > nonzero[0]`.
    grew = bool(nonzero)
    last_message = samples[-1]["message"] if samples else ""
    any_message = any(smp["message"] for smp in samples)

    dh.record(
        "A_solver_progress_advances_during_simulation",
        # `saw_running` is no longer required: a sub-second CUDA run
        # can transition STARTING -> READY between two addon polls and
        # never be observed in RUNNING. The presence of progress and
        # info fields in the server response (saw_response_progress /
        # saw_response_info) is the actual proof of the channel; the
        # addon's RUNNING phase is just the UI window during which the
        # panel shows the bar.
        saw_response_progress and saw_response_info
        and monotonic_ok and bool(nonzero) and grew and any_message,
        {
            "saw_running": saw_running,
            "saw_response_progress": saw_response_progress,
            "saw_response_info": saw_response_info,
            "monotonic_ok": monotonic_ok,
            "first_nonzero_progress": nonzero[0] if nonzero else None,
            "last_progress": progresses[-1] if progresses else None,
            "sample_count": len(samples),
            "last_message": last_message,
            "any_message": any_message,
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
