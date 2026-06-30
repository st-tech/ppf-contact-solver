# File: scenarios/bl_save_state_on_finish.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# "Save State on Finish" round-trip. Verifies the save_state_on_finish param
# flows addon state -> encoder ("save-state-on-finish") -> param.toml
# (save_state_on_finish) -> solver SimArgs, and that the solver writes a
# checkpoint on the final frame before exiting EVEN WITH auto-save off.
#
# Without the flag and with auto-save off, the solver's finish branch logs
# "simulation finished, not saving state" and leaves no state_<N>.bin.gz. With
# the flag on, the finish branch saves state, so a checkpoint lands on disk and
# the server's saved_states list reports it. The assertions are meaningful only
# because auto-save is explicitly disabled here.

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
    FRAME_COUNT = 8
    FRAME_END = 6

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="SaveOnFinishMesh")
    root = dh.configure_state(project_name="save_state_on_finish",
                             frame_count=FRAME_COUNT)
    # Enable save-on-finish; keep auto-save OFF so the finish-save is the only
    # thing that can produce a checkpoint (the assertion would pass trivially
    # otherwise).
    root.state.save_state_on_finish = True
    root.state.auto_save = False
    dh.record(
        "save_state_on_finish_prop_set",
        bool(root.state.save_state_on_finish) and not bool(root.state.auto_save),
        {"save_state_on_finish": bool(root.state.save_state_on_finish),
         "auto_save": bool(root.state.auto_save)},
    )

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(DELTA, 0.0, 0.0), frame_start=1, frame_end=FRAME_END,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                    project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="save-on-finish:build")
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    # Run to completion (all FRAME_COUNT frames), then the solver hits its
    # finish branch and, because save_state_on_finish is set, writes a state.
    dh.com.run()
    deadline = __import__('time').time() + 120.0
    while __import__('time').time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name in ("RESUMABLE", "READY", "FAILED"):
            break
        __import__('time').sleep(0.1)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.log(f"after_run solver={dh.facade.engine.state.solver.name} "
           f"frame={dh.facade.engine.state.frame}")

    # The solver wrote a checkpoint on finish (state_<N>.bin.gz on disk).
    output_dir = os.path.join(
        dh.facade.engine.state.remote_root, "session", "output")
    state_files = sorted(glob.glob(os.path.join(output_dir, "state_*.bin.gz")))
    dh.record(
        "state_saved_on_finish",
        len(state_files) > 0
        and any(os.path.getsize(p) > 0 for p in state_files),
        {"state_files": [os.path.basename(p) for p in state_files],
         "output_dir": output_dir},
    )

    # The server's saved_states list (Phase 1 wire path) reports the
    # finish checkpoint, so it is resumable.
    saved_frames = dh.com.saved_state_frames()
    dh.record(
        "saved_states_reports_finish_checkpoint",
        bool(saved_frames) and saved_frames == sorted(saved_frames),
        {"saved_state_frames": saved_frames},
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
