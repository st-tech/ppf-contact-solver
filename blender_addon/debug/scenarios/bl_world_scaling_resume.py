# File: scenarios/bl_world_scaling_resume.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling across a SAVE / RESUME checkpoint. A world-scaled sim is
# run two ways on one build: once start-to-finish (the reference), and
# once interrupted mid-run by save_and_quit and then resumed to
# completion. We assert the resumed run reproduces the uninterrupted run
# exactly -- i.e. the checkpoint does NOT double-scale.
#
# Why this matters: the solver saves its checkpoint state at SIM scale
# (geometry x world_scaling) and, on resume, re-reads world_scaling
# fresh from param.toml and divides the per-frame output back by it. If
# resume re-applied the scale-in to the already-scaled checkpoint (or
# assumed world_scaling = 1.0 because it forgot to re-read it), the
# frames after the checkpoint would land at the wrong scale. The fully
# pinned, kinematic scene makes the per-frame target a deterministic
# function of time, so a correct resume reproduces the reference to
# round-off regardless of when the checkpoint happened to land.
#
# Subtests:
#   A. reference_all_frames  - the uninterrupted run produced every frame.
#   B. checkpoint_reached    - the interrupted run hit RESUMABLE.
#   C. resume_matches_run    - resumed frames == uninterrupted frames
#                              (no double-scaling across the checkpoint).
#   D. authored_scale        - resumed positions are at the authored
#                              scale, not the (x world_scaling) sim scale.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Cross-cycle solver/checkpoint state; keep it off the parallel batch.
NOT_PARALLELIZABLE = True
# Pace the steps so save_and_quit reliably lands mid-run (not after the
# emulated sim has already raced to completion).
KNOBS = {"PPF_EMULATED_STEP_MS": "60"}

WORLD_SCALING = 10.0


_DRIVER_BODY = r"""
import os
import time
import traceback
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 12
WORLD_SCALING = <<WORLD_SCALING>>


def _run_and_capture(dh, plane):
    dh.run_and_wait(timeout=180.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.settle_idle(timeout=10.0)
    dh.fetch_and_drain()
    pc2 = dh.find_pc2_for(plane)
    if not pc2 or not os.path.isfile(pc2):
        raise RuntimeError(f"no PC2 (path={pc2!r})")
    return dh.read_pc2(pc2).copy()


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="WsResumeMesh")
    root = dh.configure_state(project_name="ws_resume",
                              frame_count=FRAME_COUNT, frame_rate=100)
    root.state.world_scaling = WORLD_SCALING
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.4, 0.0, 0.0), frame_start=1, frame_end=10,
                transition="LINEAR")

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _d, _p = encoder_pkg.prepare_upload(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "ws-resume:build", timeout=180.0)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    # ---- Reference: uninterrupted run ----
    ref = _run_and_capture(dh, plane)
    dh.record("A_reference_all_frames", ref.shape[0] == FRAME_COUNT,
              {"samples": int(ref.shape[0]), "expected": FRAME_COUNT})

    # ---- Interrupted run: save_and_quit mid-run, then resume ----
    dh.com.run()
    dh.facade.engine.dispatch(dh.events.SaveAndQuitRequested())
    saw_resumable = False
    deadline = time.time() + 120.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name == "RESUMABLE":
            saw_resumable = True
            break
        if s.solver.name == "FAILED":
            break
        time.sleep(0.15)
    dh.record("B_checkpoint_reached", saw_resumable,
              {"solver": dh.facade.engine.state.solver.name})

    dh.settle_idle(timeout=10.0)
    dh.resume_and_wait(timeout=180.0)
    resumed = _run_and_capture(dh, plane)

    # C: resumed run reproduces the uninterrupted run (no double-scaling).
    same_shape = resumed.shape == ref.shape
    if same_shape:
        max_diff = float(np.max(np.abs(resumed - ref)))
    else:
        max_diff = float("inf")
    dh.record("C_resume_matches_run", same_shape and max_diff < 1e-3,
              {"max_abs_diff": round(max_diff, 8) if same_shape else None,
               "shape_ref": list(ref.shape), "shape_resumed": list(resumed.shape)})

    # D: resumed positions are at the AUTHORED scale (plane size 1 moved by
    # 0.4 -> |coords| ~ 1.0), not the x10 sim scale that a double-scale
    # would produce (|coords| ~ 10).
    max_abs = float(np.max(np.abs(resumed)))
    dh.record("D_authored_scale", max_abs < 3.0,
              {"max_abs_pos": round(max_abs, 5),
               "sim_scale_would_be": round(max_abs * WORLD_SCALING, 3)})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<WORLD_SCALING>>", repr(WORLD_SCALING))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
