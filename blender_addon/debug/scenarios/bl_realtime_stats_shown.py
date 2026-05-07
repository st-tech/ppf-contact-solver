# File: scenarios/bl_realtime_stats_shown.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Realtime Statistics: live ``summary`` dict during simulation.
#
# The Blender addon's main panel reads ``com.response["summary"]``
# while the solver is BUSY/SAVE_AND_QUIT and renders each entry as a
# row inside the "Realtime Statistics" box (ui/main_panel.py). The
# Rust ppf-cts-server populates this field from
# `<state.root>/session/output/data/*.out` via
# ``response::build_response``; the legacy ``server.py:make_response``
# did the equivalent via ``session.get.log.summary()``.
#
# Asserts the round-trip end to end:
#
#   * the response dict carries a non-empty ``summary`` map while
#     ``solver.name == "RUNNING"`` (not just BUILDING),
#   * the ``frame`` seed is present so the addon's "frame" remap path
#     has a row to render even before the first per-frame log line is
#     flushed,
#   * the standard log channels are present in the dict (``time-per-frame``,
#     ``time-per-step``, ``num-contact``, ``newton-steps``, ``pcg-iter``).
#     They may be zero-valued when the log files haven't been written
#     yet — that's fine; the panel just shows "0ns" / "0", which is
#     still a Realtime Statistics row.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Same parallel-mode race that bl_progress_simulating documents: the
# poll cadence under host load can miss the narrow RUNNING window.
# Pin to the serial postlude so the cadence is reliable.
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

# Channels the server's `format_log_summary` always emits (mirrors
# `crates/ppf-cts-core/src/datamodel/session/log.rs`). Stretch is
# conditional on max-sigma > 0 so it's not in the required set.
EXPECTED_CHANNELS = (
    "time-per-frame",
    "time-per-step",
    "num-contact",
    "newton-steps",
    "pcg-iter",
)


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    plane = dh.reset_scene_to_pinned_plane(name="RealtimeStatsMesh")
    dh.save_blend(PROBE_DIR, "realtime_stats_shown.blend")
    root = dh.configure_state(project_name="realtime_stats_shown",
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
        message="realtime_stats:build",
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

    # Sample the response.summary dict at every poll while the solver
    # is RUNNING. ``response`` is ``com.info.response`` which mirrors
    # the most recent server reply seen by the runner.
    samples = []
    saw_running = False
    deadline = time.time() + 120.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        resp = dh.com.info.response or {}
        if s.solver.name == "RUNNING":
            saw_running = True
            summary = resp.get("summary")
            samples.append({
                "summary_is_dict": isinstance(summary, dict),
                "summary_keys": (sorted(summary.keys())
                                 if isinstance(summary, dict) else []),
                "frame_value": (summary.get("frame")
                                if isinstance(summary, dict) else None),
                "resp_status": resp.get("status"),
            })
        elif saw_running and s.solver.name in ("READY", "RESUMABLE", "FAILED"):
            break
        time.sleep(0.1)

    populated = [smp for smp in samples if smp["summary_is_dict"]]
    has_frame_seed = bool(populated) and all(
        isinstance(smp["frame_value"], str) and smp["frame_value"].lstrip("-").isdigit()
        for smp in populated
    )
    has_expected_channels = bool(populated) and all(
        all(ch in smp["summary_keys"] for ch in EXPECTED_CHANNELS)
        for smp in populated
    )

    dh.record(
        "A_summary_present_during_simulation",
        saw_running and bool(populated),
        {
            "saw_running": saw_running,
            "populated_samples": len(populated),
            "total_samples": len(samples),
        },
    )
    dh.record(
        "B_summary_has_frame_seed",
        has_frame_seed,
        {
            "first_summary_keys": populated[0]["summary_keys"] if populated else [],
            "first_frame_value": populated[0]["frame_value"] if populated else None,
        },
    )
    dh.record(
        "C_summary_has_log_channels",
        has_expected_channels,
        {
            "expected": list(EXPECTED_CHANNELS),
            "first_summary_keys": populated[0]["summary_keys"] if populated else [],
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
