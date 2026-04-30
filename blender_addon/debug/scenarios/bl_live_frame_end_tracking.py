# File: scenarios/bl_live_frame_end_tracking.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Live scene.frame_end tracking (commit 92546e18).
#
# During a live simulation, frames flow in via DoFetchFrames and
# apply_animation writes them to PC2 + advances scene.frame_end to the
# latest fetched frame. The timeline grows in lockstep with the sim.
#
# Subtests:
#   A. live_frame_end_grows_monotonically
#         Start a run with frame_count=12. While the solver is RUNNING,
#         loop PollTick + tick + apply_animation, sampling
#         scene.frame_end. Assert the sequence is non-decreasing, ends
#         at exactly the final blender frame, and never exceeds the
#         configured frame_count.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


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

    plane = dh.reset_scene_to_pinned_plane(name="LiveFrameEndMesh")
    dh.save_blend(PROBE_DIR, "live_frame_end.blend")
    root = dh.configure_state(project_name="live_frame_end_tracking",
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
        message="live_frame_end:build",
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

    # Pre-run sentinel: scene.frame_end should be Blender's default
    # (250). After the run we expect it to be at most FRAME_COUNT.
    pre_frame_end = bpy.context.scene.frame_end

    dh.com.run()
    samples = []
    saw_running = False
    saw_terminal = False
    deadline = time.time() + 120.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        # Drive apply_animation manually (the FramePump modal's TIMER
        # cannot fire while the driver holds the main thread).
        try:
            client_mod.apply_animation()
        except Exception:
            pass
        samples.append(int(bpy.context.scene.frame_end))
        s = dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        if saw_running and s.solver.name in ("READY", "RESUMABLE", "FAILED"):
            # Settle: drain any final fetches before sampling.
            # Timeout sized for slow CI runners; the local rig finishes
            # this drain in 1-2 seconds, but a free GitHub Linux runner
            # can lag behind the emulated solver's wall-clock pace and
            # need much more headroom.
            dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=60.0)
            # Inner drain has to be long enough for every queued frame
            # to actually flow through apply_animation: force_frame_query
            # returns when the **state.frame** counter (server-reported)
            # hits the threshold, but scene.frame_end only advances when
            # apply_animation actually pulls the vertex bytes out of the
            # in-memory queue and writes them to PC2 + the timeline.
            # 200 * 0.05s = 10s, with early exit once frame_end is at
            # the expected last frame.
            for _ in range(200):
                try:
                    client_mod.apply_animation()
                except Exception:
                    pass
                samples.append(int(bpy.context.scene.frame_end))
                dh.facade.tick()
                if int(bpy.context.scene.frame_end) >= FRAME_COUNT:
                    break
                time.sleep(0.05)
            saw_terminal = True
            break
        time.sleep(0.15)

    final_frame_end = int(bpy.context.scene.frame_end)
    # Filter out the pre-fetch default (250). We care about the
    # post-first-frame trajectory.
    advanced = [v for v in samples if v != pre_frame_end]
    monotonic_ok = all(advanced[i] <= advanced[i + 1]
                       for i in range(len(advanced) - 1))
    cap_ok = max(advanced) <= FRAME_COUNT if advanced else False
    # ``_apply_single_frame`` sets ``blender_frame = n + 1`` (Blender
    # uses 1-indexed timeline frames), and ``apply_animation`` writes
    # ``scene.frame_end = max(blender_frame)`` for applied frames. With
    # ``frame_count = FRAME_COUNT`` solver frames numbered n=0..N-1,
    # max blender_frame = N = FRAME_COUNT. (The earlier
    # ``FRAME_COUNT - 1`` expectation was masked by an off-by-one fetch
    # bug that hid the trailing frame; once the trailing-frame fix
    # landed the production-correct value is FRAME_COUNT.)
    final_ok = final_frame_end == FRAME_COUNT
    grew = bool(advanced) and advanced[-1] > advanced[0]
    dh.record(
        "A_live_frame_end_grows_monotonically",
        saw_running and saw_terminal and monotonic_ok and cap_ok
        and final_ok and grew,
        {
            "saw_running": saw_running,
            "saw_terminal": saw_terminal,
            "pre_frame_end": pre_frame_end,
            "final_frame_end": final_frame_end,
            "frame_count": FRAME_COUNT,
            "advanced_first10": advanced[:10],
            "advanced_last5": advanced[-5:],
            "monotonic_ok": monotonic_ok,
            "cap_ok": cap_ok,
        },
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
