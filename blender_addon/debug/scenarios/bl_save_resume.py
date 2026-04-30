# File: scenarios/bl_save_resume.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Save-and-quit / Resume round-trip.
#
# Sequence:
#   1. Build a pinned plane with MOVE_BY(delta=(D, 0, 0)) over frames
#      [1..frame_end] and run it.
#   2. Once the sim has produced at least one frame, dispatch
#      SaveAndQuitRequested. The Rust binary detects the ``save_and_quit``
#      marker on its next loop iteration, calls ``save_state``, and exits.
#   3. Assert: state_*.bin.gz lands on disk, addon settles at RESUMABLE,
#      state.frame matches the saved frame.
#   4. Dispatch ResumeRequested. The Rust binary re-launches with
#      ``--load N``, picks up from the saved frame, and runs to completion.
#   5. Fetch + drain so PC2 captures every frame.
#   6. Assert: PC2 has frame_count samples and the last sample has the
#      fully-applied delta (so the resume continuation actually advanced
#      past the save point and produced the right end state).

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import glob
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


try:
    DELTA = 0.5
    # The sim must outlast the driver's poll loop so save_and_quit can
    # actually land mid-run. Under parallel=4 host load the driver's
    # 0.2s poll cadence routinely misses the save window if the sim is
    # too short. 20 frames at PPF_EMULATED_STEP_MS=100 gives ~2s.
    FRAME_COUNT = 20
    FRAME_END = 18

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="SaveResumeMesh")
    dh.save_blend(PROBE_DIR, "saveresume.blend")
    root = dh.configure_state(project_name="save_resume",
                              frame_count=FRAME_COUNT)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(DELTA, 0.0, 0.0), frame_start=1, frame_end=FRAME_END,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="save-resume:build")
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    # Run, then dispatch save_and_quit while solver is still RUNNING.
    dh.com.run()
    saw_running = False
    save_dispatched = False
    deadline = __import__('time').time() + 90.0
    while __import__('time').time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        # Threshold of 1 leaves at least 17 frames of work for the
        # resume leg to complete, well clear of the natural-completion
        # boundary where save_state would be a no-op.
        if saw_running and not save_dispatched and s.frame >= 1:
            dh.facade.engine.dispatch(dh.events.SaveAndQuitRequested())
            save_dispatched = True
            dh.log(f"save_and_quit dispatched at frame={s.frame}")
        if save_dispatched and s.solver.name in ("RESUMABLE", "FAILED"):
            break
        if s.solver.name == "FAILED":
            break
        __import__('time').sleep(0.2)
    dh.log(f"after_save solver={dh.facade.engine.state.solver.name} "
           f"frame={dh.facade.engine.state.frame}")

    dh.force_frame_query(expected_frames=1, timeout=10.0)
    saved_frame = dh.facade.engine.state.frame

    output_dir = os.path.join(
        dh.facade.engine.state.remote_root, "session", "output")
    state_files = sorted(glob.glob(os.path.join(output_dir, "state_*.bin.gz")))
    dh.record(
        "save_state_file_written",
        len(state_files) > 0
        and any(os.path.getsize(p) > 0 for p in state_files),
        {"state_files": [os.path.basename(p) for p in state_files],
         "output_dir": output_dir, "saved_frame": saved_frame},
    )
    dh.record(
        "solver_resumable_after_save",
        dh.facade.engine.state.solver.name == "RESUMABLE",
        {"solver": dh.facade.engine.state.solver.name},
    )

    dh.settle_idle(timeout=10.0)
    saw_running_resume = dh.resume_and_wait(timeout=90.0)
    dh.record(
        "resume_completed",
        saw_running_resume
        and dh.facade.engine.state.solver.name in ("READY", "RESUMABLE"),
        {"solver": dh.facade.engine.state.solver.name,
         "saw_running": saw_running_resume},
    )
    dh.log(f"resumed solver={dh.facade.engine.state.solver.name}")

    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    pc2_path = dh.find_pc2_for(plane)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced after resume: {pc2_path}")
    arr = dh.read_pc2(pc2_path)
    dh.record(
        "pc2_has_all_frames",
        arr.shape[0] == FRAME_COUNT,
        {"pc2_samples": arr.shape[0], "expected": FRAME_COUNT},
    )

    last = arr[-1]
    rest = arr[0]
    final_shift_x = float(last[0][0] - rest[0][0])
    dh.record(
        "trajectory_completed_after_resume",
        abs(final_shift_x - DELTA) < 1e-3,
        {"final_shift_x": final_shift_x, "expected": DELTA,
         "rest_v0": rest[0].tolist(), "last_v0": last[0].tolist()},
    )
    dh.log(f"final_shift={final_shift_x:.4f}")

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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
