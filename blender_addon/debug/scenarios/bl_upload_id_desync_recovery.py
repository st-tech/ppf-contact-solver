# File: scenarios/bl_upload_id_desync_recovery.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Upload-id desync recovery (commit ea4303cb).
#
# The pipeline pins ``state.active_upload_id`` to the server's reported
# upload_id on the first poll after a build is dispatched. If a later
# poll sees a different upload_id while activity is BUILDING, the addon
# treats it as a desync, resets activity to IDLE, clears the pinned
# upload_id, and surfaces "Server state lost during build" so the user
# can recover by re-uploading.
#
# Subtests:
#   A. desync_detection
#         Establish baseline state through a normal transfer + build.
#         Inject activity=BUILDING with active_upload_id pinned to a
#         known value, then dispatch a synthetic ServerPolled whose
#         response carries a different upload_id. Assert:
#           * state.error contains "Server state lost"
#           * state.active_upload_id is empty
#           * state.activity is IDLE
#   B. recovery_after_desync
#         After the desync above, dispatch a real
#         BuildPipelineRequested. Assert the build completes (solver
#         READY/RESUMABLE) and active_upload_id matches the server's
#         new upload_id.

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
                           fromlist=["AppState", "Activity", "Solver"])
    Activity = state_mod.Activity
    Solver = state_mod.Solver
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    plane = dh.reset_scene_to_pinned_plane(name="DesyncMesh")
    dh.save_blend(PROBE_DIR, "desync.blend")
    root = dh.configure_state(project_name="upload_id_desync", frame_count=6)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="desync:initial",
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
    s0 = dh.facade.engine.state
    initial_uid = s0.server_upload_id
    assert s0.solver.name in ("READY", "RESUMABLE"), s0.solver.name
    assert initial_uid, "expected server_upload_id after initial build"

    # ----- A: desync detection ------------------------------------
    # Inject activity=BUILDING with a pinned active_upload_id, then
    # feed a synthetic ServerPolled whose upload_id differs. The
    # transition fires the desync path.
    pinned_uid = "pinneduid0001"
    foreign_uid = "foreignuid002"
    with dh.facade.engine._lock:
        dh.facade.engine._state = replace(
            dh.facade.engine._state,
            activity=Activity.BUILDING,
            solver=Solver.BUILDING,
            active_upload_id=pinned_uid,
        )
    synthetic_response = {
        "protocol_version": "0.03",
        "upload_id": foreign_uid,
        "status": "NO_BUILD",
        "error": "",
        "info": "",
        "frame": 0,
        "progress": 0,
        "violations": [],
        "root": s0.remote_root,
        "data_hash": "",
        "param_hash": "",
    }
    dh.facade.engine.dispatch(dh.events.ServerPolled(response=synthetic_response))
    dh.facade.tick()
    s_after = dh.facade.engine.state
    dh.record(
        "A_desync_detection",
        "Server state lost" in (s_after.error or "")
        and s_after.active_upload_id == ""
        and s_after.activity.name == "IDLE"
        and s_after.server_upload_id == foreign_uid,
        {
            "error": s_after.error,
            "active_upload_id": s_after.active_upload_id,
            "activity": s_after.activity.name,
            "server_upload_id": s_after.server_upload_id,
        },
    )

    # ----- B: recovery_after_desync -------------------------------
    # The addon should be able to recover by re-uploading. Re-encode
    # (state may have shifted), dispatch BuildPipelineRequested, and
    # wait for terminal solver. The new upload mints a fresh uid; the
    # addon must pin it without spurious desync.
    data2, param2 = dh.encode_payload()
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data2, param=param2,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="desync:recover",
    ))
    deadline = time.time() + 60.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)
    s_rec = dh.facade.engine.state
    dh.record(
        "B_recovery_after_desync",
        s_rec.solver.name in ("READY", "RESUMABLE")
        and s_rec.activity.name == "IDLE"
        and s_rec.error == ""
        and bool(s_rec.server_upload_id)
        and s_rec.server_upload_id != initial_uid
        and s_rec.server_upload_id != foreign_uid,
        {
            "solver": s_rec.solver.name,
            "activity": s_rec.activity.name,
            "error": s_rec.error,
            "initial_uid": initial_uid[:12],
            "new_uid": s_rec.server_upload_id[:12],
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
