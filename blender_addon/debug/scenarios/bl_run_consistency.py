# File: scenarios/bl_run_consistency.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end of the Run-button click-time consistency check.
#
# There is no client-side hash cache and ``Run.poll`` never compares
# hashes; the only drift gate is the click-time recompute, now staged
# inside ``SOLVER_OT_Run.execute`` -> ``start_stages``. The geometry and
# parameter checks each run as a modal stage and raise ``StageAbort`` on
# drift, which the modal turns into a ``self.report`` ERROR + CANCELLED.
# Each subtest drives ``execute`` with a staged stub ``self`` and then
# ``drain_stages`` to pump those ticks, so we can capture the
# ``self.report`` payload Blender would surface to the user.
#
# Subtests:
#   A. fresh_transfer_run_enabled
#         After a successful Transfer + Build the live param hash
#         and data hash match what the server echoes; ``Run.poll``
#         is True and ``execute`` would proceed (we don't actually
#         start the solver in this subtest -- A is the "no drift"
#         baseline).
#
#   B. param_drift_run_click_reports_error
#         Mutating ``gravity_3d`` makes ``compute_param_hash`` diverge
#         from ``engine.state.server_param_hash``. ``execute`` returns
#         RUNNING_MODAL; the drained staged param check ends CANCELLED
#         with an ERROR pointing at "Update Params", the solver never
#         starts, and ``Run.poll`` is True again (the button stays
#         actionable; poll deliberately ignores hashes).
#
#   C. update_params_re_enables_run
#         After ``solver.update_params``, the server stores the new
#         param hash and echoes it on the next status response. The
#         live ``compute_param_hash`` lines up with the new echo, so a
#         fresh ``execute`` + ``drain_stages`` runs every stage (no
#         param error) and the final stage starts the solver.
#
#   D. data_drift_run_click_reports_error
#         A bmesh-driven topology change makes ``compute_data_hash``
#         diverge. The staged geometry check aborts first, so the drained
#         run ends CANCELLED with a "Transfer" error (the data path is the
#         louder of the two click-time errors because re-transferring is
#         the only fix).

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import bmesh
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


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="RunConsistMesh")
    dh.save_blend(PROBE_DIR, "runconsist.blend")
    root = dh.configure_state(project_name="run_consistency", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    data_bytes, param_bytes = dh.encode_payload()
    pre_data_hash = encoder_mesh.compute_data_hash(bpy.context)
    pre_param_hash = encoder_params.compute_param_hash(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=pre_data_hash, param_hash=pre_param_hash,
        message="run-consistency:build",
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
    dh.log(f"built solver={dh.facade.engine.state.solver.name} "
           f"data={dh.facade.engine.state.server_data_hash[:12]!r} "
           f"param={dh.facade.engine.state.server_param_hash[:12]!r}")

    Run = _solver_run_op()

    # ----- A: fresh transfer, no drift -----------------------------
    s = dh.facade.engine.state
    fresh_data_a = encoder_mesh.compute_data_hash(bpy.context)
    fresh_param_a = encoder_params.compute_param_hash(bpy.context)
    dh.record(
        "A_fresh_transfer_run_enabled",
        bool(Run.poll(bpy.context))
        and fresh_data_a == s.server_data_hash
        and fresh_param_a == s.server_param_hash,
        {
            "poll": bool(Run.poll(bpy.context)),
            "fresh_data": fresh_data_a[:12],
            "server_data": s.server_data_hash[:12],
            "fresh_param": fresh_param_a[:12],
            "server_param": s.server_param_hash[:12],
        },
    )

    # ----- B: param drift, click-time error ------------------------
    # execute() returns RUNNING_MODAL and stages the drift checks; the
    # param-check stage aborts the drained run with an "Update Params"
    # ERROR. The solver must not start, and the button re-enables (poll
    # ignores hashes) once the aborted run clears the encode-progress gate.
    root.state.gravity_3d = (0.0, 0.0, -4.9)
    s = dh.facade.engine.state
    fresh_param_b = encoder_params.compute_param_hash(bpy.context)
    stub_b = dh.staged_stub(Run)
    verdict_b = Run.execute(stub_b, bpy.context)
    final_b = stub_b.drain_stages(bpy.context)
    error_b = stub_b.error("Update Params")
    s = dh.facade.engine.state
    poll_after_b = bool(Run.poll(bpy.context))
    dh.record(
        "B_param_drift_run_click_reports_error",
        verdict_b == {"RUNNING_MODAL"}
        and final_b == {"CANCELLED"}
        and fresh_param_b != s.server_param_hash
        and bool(error_b)
        and s.solver.name != "STARTING"
        and poll_after_b,
        {
            "verdict": list(verdict_b),
            "final": list(final_b) if final_b else None,
            "poll_after": poll_after_b,
            "fresh_param": fresh_param_b[:12],
            "server_param": s.server_param_hash[:12],
            "solver": s.solver.name,
            "error": error_b[:120],
        },
    )

    # ----- C: update_params re-aligns ------------------------------
    bpy.ops.solver.update_params()
    deadline = time.time() + 60.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)
    s = dh.facade.engine.state
    fresh_param_c = encoder_params.compute_param_hash(bpy.context)
    # ``Run.poll`` snapshotted *before* execute -- once the drained stages
    # pass the drift gates the final stage calls ``com.run``, which starts
    # the solver and (legitimately) flips poll False thereafter.
    poll_pre_run_c = bool(Run.poll(bpy.context))
    stub_c = dh.staged_stub(Run)
    verdict_c = Run.execute(stub_c, bpy.context)
    final_c = stub_c.drain_stages(bpy.context)
    param_error_c = stub_c.error("Update Params")
    s = dh.facade.engine.state
    dh.record(
        "C_update_params_re_enables_run",
        fresh_param_c == s.server_param_hash
        and not param_error_c
        and poll_pre_run_c
        and verdict_c == {"RUNNING_MODAL"}
        and final_c is None
        and stub_c.modal_set_up
        and s.solver.name in ("STARTING", "RUNNING"),
        {
            "fresh_param": fresh_param_c[:12],
            "server_param": s.server_param_hash[:12],
            "poll_pre_run": poll_pre_run_c,
            "verdict": list(verdict_c),
            "final": list(final_c) if final_c else None,
            "modal_set_up": stub_c.modal_set_up,
            "solver": s.solver.name,
            "captured": stub_c.captured[:3],
        },
    )

    # ----- D: data drift, click-time error -------------------------
    bm = bmesh.new()
    bm.from_mesh(plane.data)
    bm.verts.new((1.5, 0.0, 0.0))
    bm.to_mesh(plane.data)
    bm.free()
    plane.data.update()
    s = dh.facade.engine.state
    fresh_data_d = encoder_mesh.compute_data_hash(bpy.context)
    stub_d = dh.staged_stub(Run)
    verdict_d = Run.execute(stub_d, bpy.context)
    final_d = stub_d.drain_stages(bpy.context)
    error_d = stub_d.error("Transfer")
    s = dh.facade.engine.state
    dh.record(
        "D_data_drift_run_click_reports_error",
        verdict_d == {"RUNNING_MODAL"}
        and final_d == {"CANCELLED"}
        and bool(error_d)
        and fresh_data_d != s.server_data_hash,
        {
            "verdict": list(verdict_d),
            "final": list(final_d) if final_d else None,
            "error": error_d[:120],
            "fresh_data": fresh_data_d[:12],
            "server_data": s.server_data_hash[:12],
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
