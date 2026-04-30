# File: scenarios/bl_transfer_disabled_during_run.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression: clicking "Run" used to leave "Transfer" clickable for a
# window because ``SOLVER_OT_Transfer.poll`` only inspected
# ``status.ready()`` (a protocol-version check) plus the cached server
# response. Right after ``Run.execute`` the engine has flipped to
# STARTING_SOLVER locally but the next ``PollTick`` hasn't round-
# tripped with the server yet, so ``is_running(response)`` reads the
# stale READY response and returns False. The fix adds
# ``not com.info.status.in_progress()`` to Transfer's poll, which
# rejects every active operation including STARTING_SOLVER.
#
# Subtests:
#   A. transfer_clickable_at_ready: after a successful Transfer +
#      Build, the system is at READY and Transfer is clickable
#      (re-transfer is a legitimate action).
#   B. transfer_disabled_immediately_after_run_click: invoking
#      ``Run.execute`` flips the engine to STARTING_SOLVER and
#      Transfer.poll must return False before any further ticks.
#   C. transfer_disabled_during_simulation_in_progress: after the
#      engine has caught up via PollTick, the status reaches
#      SIMULATION_IN_PROGRESS and Transfer.poll must remain False.
#
# Run.poll, UpdateParams.poll, and Transfer.poll are all snapshotted
# at each phase to make the regression locus explicit if the fix
# regresses.

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


class _StubSelf:
    def __init__(self):
        self.captured = []
        self.modal_set_up = False
    def report(self, kind, msg):
        self.captured.append((tuple(kind), msg))
    def setup_modal(self, ctx):
        self.modal_set_up = True


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="TransferDuringRunMesh")
    dh.save_blend(PROBE_DIR, "transfer_during_run.blend")
    root = dh.configure_state(
        project_name="transfer_disabled_during_run", frame_count=6,
    )

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    encoder_pkg = __import__(pkg + ".core.encoder",
                             fromlist=["prepare_upload"])
    solver_mod = __import__(pkg + ".ui.solver",
                            fromlist=["SOLVER_OT_Run",
                                      "SOLVER_OT_Transfer",
                                      "SOLVER_OT_UpdateParams"])
    Run = solver_mod.SOLVER_OT_Run
    Transfer = solver_mod.SOLVER_OT_Transfer
    UpdateParams = solver_mod.SOLVER_OT_UpdateParams
    com = dh.com

    data, param, dhash, phash = encoder_pkg.prepare_upload(bpy.context)
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data, param=param, data_hash=dhash, param_hash=phash,
        message="transfer-during-run:build",
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

    # ----- A: at READY, Transfer is clickable ----------------------
    s = dh.facade.engine.state
    transfer_a = bool(Transfer.poll(bpy.context))
    run_a = bool(Run.poll(bpy.context))
    update_a = bool(UpdateParams.poll(bpy.context))
    dh.record(
        "A_transfer_clickable_at_ready",
        transfer_a and run_a and update_a
        and s.solver.name == "READY"
        and com.info.status.name == "READY",
        {
            "transfer_poll": transfer_a,
            "run_poll": run_a,
            "update_params_poll": update_a,
            "solver": s.solver.name,
            "status": com.info.status.name,
        },
    )

    # ----- B: immediately after Run.execute, Transfer.poll is False
    # The engine flips to STARTING_SOLVER synchronously; no PollTick
    # has rounded-tripped with the server yet, so the cached response
    # still echoes READY. The fix asserts on
    # ``status.in_progress()`` which knows STARTING_SOLVER counts.
    stub = _StubSelf()
    verdict = Run.execute(stub, bpy.context)
    s = dh.facade.engine.state
    transfer_b = bool(Transfer.poll(bpy.context))
    run_b = bool(Run.poll(bpy.context))
    update_b = bool(UpdateParams.poll(bpy.context))
    dh.record(
        "B_transfer_disabled_immediately_after_run_click",
        verdict == {"RUNNING_MODAL"}
        and stub.modal_set_up
        and not transfer_b
        and not run_b
        and not update_b
        and s.solver.name == "STARTING"
        and com.info.status.name == "STARTING_SOLVER",
        {
            "verdict": list(verdict),
            "transfer_poll": transfer_b,
            "run_poll": run_b,
            "update_params_poll": update_b,
            "solver": s.solver.name,
            "status": com.info.status.name,
            "captured": stub.captured[:2],
        },
    )

    # ----- C: drive ticks until SIMULATION_IN_PROGRESS -------------
    saw_running = False
    deadline = time.time() + 30.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if com.info.status.name == "SIMULATION_IN_PROGRESS":
            saw_running = True
            break
        time.sleep(0.05)
    transfer_c = bool(Transfer.poll(bpy.context))
    run_c = bool(Run.poll(bpy.context))
    update_c = bool(UpdateParams.poll(bpy.context))
    dh.record(
        "C_transfer_disabled_during_simulation_in_progress",
        saw_running
        and not transfer_c
        and not run_c
        and not update_c,
        {
            "saw_running": saw_running,
            "transfer_poll": transfer_c,
            "run_poll": run_c,
            "update_params_poll": update_c,
            "status": com.info.status.name,
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
