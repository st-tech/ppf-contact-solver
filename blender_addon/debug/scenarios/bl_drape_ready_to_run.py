# File: scenarios/bl_drape_ready_to_run.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# After a successful Transfer the Run button must be clickable and
# the solver must be in READY state. There is no client-side hash
# cache anymore: ``Run.poll`` only checks connection / status / busy
# guards, and the click-time hash compare lives in ``Run.execute``.
# Drift between scene state and what the server holds becomes a click-
# time error, not a poll-time grey-out.
#
# What this scenario does:
#   - Builds a 16x16 grid sheet pinned at its +x corners (the same
#     shape as the project README's drape example, just smaller for
#     the emulated runner).
#   - Transfers + builds via the production ``prepare_upload`` path,
#     waits for the emulated build to settle.
#   - Asserts ``SOLVER_OT_Run.poll`` returns True (Run is clickable),
#     solver state is READY, and the server has echoed back the
#     param + data hashes that ``prepare_upload`` shipped.

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


def _solver_run_op():
    return __import__(pkg + ".ui.solver",
                      fromlist=["SOLVER_OT_Run"]).SOLVER_OT_Run


def _build_drape_scene():
    # Wipe the default scene and create a minimal drape: a 16x16 grid
    # sheet pinned at the two +x corners, exactly the structure the
    # README's Blender Python example builds (just smaller for the
    # emulated runner).
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=16, y_subdivisions=16, size=2,
        location=(0, 0, 0.6),
    )
    sheet = bpy.context.object
    sheet.name = "Sheet"

    vg = sheet.vertex_groups.new(name="Corners")
    corner_indices = [
        i for i, v in enumerate(sheet.data.vertices)
        if v.co.x < -0.99 and abs(abs(v.co.y) - 1.0) < 0.01
    ]
    vg.add(corner_indices, 1.0, "REPLACE")
    return sheet, corner_indices


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    sheet, corner_indices = _build_drape_scene()
    dh.save_blend(PROBE_DIR, "drape_ready_to_run.blend")
    root = dh.configure_state(project_name="drape_ready_to_run", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.param.enable_strain_limit = True
    cloth.param.strain_limit = 0.05
    cloth.param.bend = 1
    cloth.create_pin(sheet.name, "Corners")

    # Drive the upload through the same single-source-of-truth helper
    # the production Transfer operator uses. ``prepare_upload`` encodes
    # the payloads and computes the matching hashes; the server stores
    # them and echoes them on every status response, which is what
    # the click-time check in ``SOLVER_OT_Run.execute`` later compares
    # a fresh recompute against.
    encoder_pkg = __import__(pkg + ".core.encoder",
                             fromlist=["prepare_upload"])
    data_bytes, param_bytes, pre_data_hash, pre_param_hash = (
        encoder_pkg.prepare_upload(bpy.context)
    )
    dh.log(f"prepare_upload data_hash={pre_data_hash[:12]} "
           f"param_hash={pre_param_hash[:12]}")

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=pre_data_hash, param_hash=pre_param_hash,
        message="drape:transfer-and-build",
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
    s = dh.facade.engine.state
    dh.log(f"built solver={s.solver.name} "
           f"server_data={s.server_data_hash[:12]!r} "
           f"server_param={s.server_param_hash[:12]!r}")

    Run = _solver_run_op()
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])

    # ----- A: SOLVER_OT_Run.poll returns True after a fresh transfer
    # poll is base-only now (no hash gate); the click-time check in
    # execute() handles drift. After a successful build the base
    # conditions (connected, server running, status READY, not busy,
    # ClearAnimation not pollable) all line up.
    poll_result = bool(Run.poll(bpy.context))
    dh.record(
        "A_run_button_clickable_after_transfer",
        poll_result,
        {
            "poll": poll_result,
            "solver": dh.facade.engine.state.solver.name,
            "activity": dh.facade.engine.state.activity.name,
        },
    )

    # ----- B: solver state is READY ("Ready to Run" in the UI) ----
    s = dh.facade.engine.state
    dh.record(
        "B_solver_state_ready_to_run",
        s.solver.name == "READY"
        and s.activity.name == "IDLE",
        {
            "solver": s.solver.name,
            "activity": s.activity.name,
        },
    )

    # ----- C: server echoed the param hash we shipped --------------
    # The server stores ``upload_id.txt`` next to ``data.pickle`` /
    # ``param.pickle`` and replays the recorded hashes on every status
    # response. A fresh ``compute_param_hash(bpy.context)`` should
    # match that echo because nothing on the scene changed since
    # ``prepare_upload`` was called.
    s = dh.facade.engine.state
    fresh_param = encoder_params.compute_param_hash(bpy.context)
    dh.record(
        "C_param_hash_matches_no_discrepancy",
        s.server_param_hash != ""
        and s.server_param_hash == fresh_param == pre_param_hash,
        {
            "fresh_param": fresh_param[:12],
            "server_param": s.server_param_hash[:12],
            "encoded_param": pre_param_hash[:12],
        },
    )

    # ----- D: server echoed the data hash we shipped ---------------
    fresh_data = encoder_mesh.compute_data_hash(bpy.context)
    dh.record(
        "D_data_hash_matches_no_discrepancy",
        s.server_data_hash != ""
        and s.server_data_hash == fresh_data == pre_data_hash,
        {
            "fresh_data": fresh_data[:12],
            "server_data": s.server_data_hash[:12],
            "encoded_data": pre_data_hash[:12],
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
