# File: scenarios/bl_resume_from_frame.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Resume-from-an-explicit-frame round-trip. Covers the "resume a simulation
# from an arbitrary already-simulated frame" feature end to end:
#
#   Phase 1: the server response now carries a full saved_states list and the
#            facade exposes com.saved_state_frames(). After a checkpoint lands
#            we assert the saved frame appears in that list.
#   Phase 2: com.resume(from_frame=N) threads an explicit resume_from frame
#            through facade -> transitions -> wire -> the solver's --load N
#            (vs the historical latest-checkpoint sentinel that resume() sends).
#
# Mechanics mirror bl_save_resume: build a pinned plane with MOVE_BY, run, gate
# on a real frame advance, save_and_quit to land a checkpoint, then resume from
# that exact frame. With a single checkpoint at saved_frame, from_frame=saved_frame
# resolves to the same state the latest sentinel would, so vert_<saved_frame>.bin
# must be preserved (the solver only prunes frames > N on --load N) and the
# trajectory must complete. Resuming from an earlier-of-many checkpoints is
# covered by the unit tests transitions::tests::resume_from_specific_frame and
# wire::tests::resume_request_carries_from_frame.

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
    FRAME_COUNT = 20
    FRAME_END = 18

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="ResumeFromFrameMesh")
    dh.save_blend(PROBE_DIR, "resume_from_frame.blend")
    root = dh.configure_state(project_name="resume_from_frame",
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
    dh.build_and_wait(data_bytes, param_bytes, message="resume-from-frame:build")
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    # Run, then wait for at least one real frame advance before save_and_quit
    # so the saved checkpoint lands at a frame > 0 (a frame-0 save degenerates
    # to the fresh-start path and is indistinguishable).
    GATE_TIMEOUT_S = 60.0
    dh.com.run()
    gate_deadline = __import__('time').time() + GATE_TIMEOUT_S
    while __import__('time').time() < gate_deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if dh.facade.engine.state.frame >= 1:
            break
        __import__('time').sleep(0.05)
    if dh.facade.engine.state.frame < 1:
        raise RuntimeError(
            f"resume-from-frame gate: solver never advanced past frame 0 "
            f"within {GATE_TIMEOUT_S:.0f}s "
            f"(frame={dh.facade.engine.state.frame}, "
            f"solver={dh.facade.engine.state.solver.name})"
        )
    dh.facade.engine.dispatch(dh.events.SaveAndQuitRequested())
    dh.log(f"save_and_quit dispatched at frame={dh.facade.engine.state.frame}")
    deadline = __import__('time').time() + 90.0
    while __import__('time').time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
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
         "saved_frame": saved_frame},
    )
    dh.record(
        "solver_resumable_after_save",
        dh.facade.engine.state.solver.name == "RESUMABLE",
        {"solver": dh.facade.engine.state.solver.name},
    )

    # Phase 1: the saved checkpoint must appear in the server-reported
    # saved_states list surfaced through facade.saved_state_frames().
    saved_frames = dh.com.saved_state_frames()
    dh.record(
        "saved_states_lists_checkpoint",
        bool(saved_frames)
        and saved_frame in saved_frames
        and saved_frames == sorted(saved_frames),
        {"saved_state_frames": saved_frames, "saved_frame": saved_frame},
    )

    # Capture vert_<saved_frame>.bin's mtime so we can assert across the resume
    # that the solver loaded the checkpoint (--load N keeps frames 0..N) instead
    # of wiping it and starting fresh (which would bump the mtime).
    saved_vert_path = os.path.join(output_dir, f"vert_{saved_frame}.bin")
    saved_vert_mtime_before = (
        os.path.getmtime(saved_vert_path)
        if os.path.isfile(saved_vert_path) else None
    )

    dh.settle_idle(timeout=10.0)

    # Phase 2: resume from the EXPLICIT saved frame (resume_from wire path),
    # not the latest-checkpoint sentinel that dh.resume_and_wait() sends.
    dh.com.resume(from_frame=saved_frame)
    saw_running_resume = False
    rdeadline = __import__('time').time() + 90.0
    while __import__('time').time() < rdeadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running_resume = True
        if s.solver.name in ("READY", "RESUMABLE"):
            break
        __import__('time').sleep(0.2)
    dh.record(
        "resume_from_frame_completed",
        saw_running_resume
        and dh.facade.engine.state.solver.name in ("READY", "RESUMABLE"),
        {"solver": dh.facade.engine.state.solver.name,
         "saw_running": saw_running_resume,
         "from_frame": saved_frame},
    )
    dh.log(f"resumed_from={saved_frame} solver={dh.facade.engine.state.solver.name}")

    # A real resume from frame N leaves vert_<saved_frame>.bin untouched
    # (setup() only deletes frames > N); a fresh restart wipes and rewrites it.
    saved_vert_mtime_after = (
        os.path.getmtime(saved_vert_path)
        if os.path.isfile(saved_vert_path) else None
    )
    dh.record(
        "saved_vert_preserved_across_resume_from_frame",
        (saved_vert_mtime_before is not None
         and saved_vert_mtime_after is not None
         and abs(saved_vert_mtime_after - saved_vert_mtime_before) < 0.001),
        {"path": os.path.basename(saved_vert_path),
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
        "trajectory_completed_after_resume_from_frame",
        abs(final_shift_x - DELTA) < 1e-3,
        {"final_shift_x": final_shift_x, "expected": DELTA},
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
