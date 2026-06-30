# File: scenarios/bl_save_resume.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Save-and-quit / Resume round-trip with checkpoint-load verification.
#
# Sequence:
#   1. Build a pinned plane with MOVE_BY(delta=(D, 0, 0)) over frames
#      [1..frame_end] and run it.
#   2. Wait until state.frame >= 1 (a real advance lands), THEN
#      dispatch SaveAndQuitRequested. The Rust binary detects the
#      ``save_and_quit`` marker on its next loop iteration, calls
#      ``save_state``, and exits. The gate guarantees ``saved_frame > 0``
#      so the resume-correctness checks below can actually
#      distinguish a real load-from-state from a silent fresh-start
#      (the saved_frame=0 case falls into ``setup(load=0)`` ->
#      ``remove_files_in_dir`` -> fresh-start path, which produces
#      indistinguishable artifacts).
#   3. Assert: state_*.bin.gz lands on disk, addon settles at RESUMABLE,
#      state.frame matches the saved frame, the server response's
#      scene_info["Last Saved"] reflects the saved frame index
#      (catches the regression where scene_info.rs scanned save_*.bin
#      instead of state_<N>.bin.gz and returned "None" forever).
#   4. Capture vert_<saved_frame>.bin's mtime, then dispatch
#      ResumeRequested. The Rust binary re-launches with ``--load -1``
#      and the solver-side fix resolves the sentinel to the actual
#      latest checkpoint; setup() then runs remove_files() starting at
#      saved_frame + 1 instead of 0, preserving vert_<saved_frame>.bin.
#   5. Fetch + drain so PC2 captures every frame.
#   6. Assert: vert_<saved_frame>.bin's mtime is unchanged across the
#      resume (catches the --load=-1 fresh-start regression that
#      would re-write the file), PC2 has frame_count samples, and
#      the last sample has the fully-applied delta.
#
# Bug 3 (ProjectSelected reset wiping total_frames -> progress bar
# stuck at 0 after server restart) is not exercised here because the
# LOCAL backend's reconnect doesn't probe with `--name __probe__` and
# a same-project ProjectSelected on reconnect preserves the cached
# total_frames. That regression is covered directly at the transition
# layer in `crates/ppf-cts-core/src/transitions/tests.rs::
# project_selected_carries_total_frames_for_new_project` and
# `project_selected_same_project_rehydrates_total_frames_when_lost`.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


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

    # Run, then wait for at least one frame advance before dispatching
    # save_and_quit. The gate guarantees saved_frame > 0; saved_frame=0
    # would make the resume-correctness checks below indistinguishable
    # between the bug and the fix (setup(load=0) wipes everything and
    # falls through to fresh-start, so the artifact diff disappears).
    # Wall-clock to first frame varies a lot between backends: the
    # CPU emulator at PPF_EMULATED_STEP_MS=100 advances in well under
    # a second, but the real CUDA solver on a GPU host needs several
    # seconds of cold start (kernel JIT, GPU mem allocation, dataset
    # build) before the first frame lands. The gate is long enough
    # to absorb that cold start; a truly hung solver is still caught
    # well below the scenario's overall 240s timeout.
    GATE_TIMEOUT_S = 60.0
    dh.com.run()
    saw_running = False
    gate_deadline = __import__('time').time() + GATE_TIMEOUT_S
    while __import__('time').time() < gate_deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if dh.facade.engine.state.solver.name == "RUNNING":
            saw_running = True
        if dh.facade.engine.state.frame >= 1:
            break
        __import__('time').sleep(0.05)
    if dh.facade.engine.state.frame < 1:
        raise RuntimeError(
            f"save-resume gate: solver never advanced past frame 0 "
            f"within {GATE_TIMEOUT_S:.0f}s "
            f"(frame={dh.facade.engine.state.frame}, "
            f"solver={dh.facade.engine.state.solver.name})"
        )
    dh.facade.engine.dispatch(dh.events.SaveAndQuitRequested())
    save_dispatched = True
    dh.log(f"save_and_quit dispatched at frame={dh.facade.engine.state.frame}")
    deadline = __import__('time').time() + 90.0
    while __import__('time').time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        if s.solver.name in ("RESUMABLE", "FAILED"):
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

    # Catches the response/scene_info.rs regression where
    # scan_output_progress scanned save_*.bin (a filename pattern
    # nothing on disk uses) instead of state_<N>.bin.gz.
    scene_info = dh.com.response.get("scene_info", {})
    last_saved_str = str(scene_info.get("Last Saved", ""))
    # The server comma-groups "Last Saved" (matching the other count
    # rows), so strip separators before comparing against the raw
    # saved_frame to stay correct once a saved frame crosses 1000.
    dh.record(
        "scene_info_last_saved_matches",
        last_saved_str.replace(",", "") == str(saved_frame),
        {"reported": last_saved_str, "expected": str(saved_frame),
         "scene_info_keys": sorted(scene_info.keys())},
    )

    # Capture vert_<saved_frame>.bin's mtime so we can assert across
    # the resume that the solver loaded the checkpoint instead of
    # wiping it and starting fresh. With the --load=-1 regression in
    # solver/main.rs, setup() runs remove_files starting at index 0
    # and the fresh restart re-writes vert_0..N -- the mtime bumps.
    saved_vert_path = os.path.join(output_dir, f"vert_{saved_frame}.bin")
    saved_vert_mtime_before = (
        os.path.getmtime(saved_vert_path)
        if os.path.isfile(saved_vert_path) else None
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

    # Catches the --load=-1 fresh-start regression: a real resume
    # leaves vert_<saved_frame>.bin untouched (setup() only deletes
    # frames > saved_frame); a fresh restart wipes and rewrites it,
    # bumping mtime.
    saved_vert_mtime_after = (
        os.path.getmtime(saved_vert_path)
        if os.path.isfile(saved_vert_path) else None
    )
    dh.record(
        "saved_vert_preserved_across_resume",
        (saved_vert_mtime_before is not None
         and saved_vert_mtime_after is not None
         and abs(saved_vert_mtime_after - saved_vert_mtime_before) < 0.001),
        {"path": saved_vert_path,
         "before": saved_vert_mtime_before,
         "after": saved_vert_mtime_after,
         "saved_frame": saved_frame},
    )

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
