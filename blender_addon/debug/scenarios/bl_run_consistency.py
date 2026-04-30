# File: scenarios/bl_run_consistency.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end of the Run-button click-time consistency check.
#
# There is no client-side hash cache and ``Run.poll`` never compares
# hashes; the only drift gate is the click-time recompute inside
# ``SOLVER_OT_Run.execute``. Both subtests below drive ``execute``
# with a stub ``self`` so we can capture the ``self.report`` payload
# Blender would surface to the user.
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
#         from ``engine.state.server_param_hash``. ``Run.poll`` still
#         returns True (poll deliberately ignores hashes), and
#         ``execute`` reports an ERROR pointing at "Update Params"
#         and returns CANCELLED.
#
#   C. update_params_re_enables_run
#         After ``solver.update_params``, the server stores the new
#         param hash and echoes it on the next status response. The
#         live ``compute_param_hash`` lines up with the new echo and
#         a fresh ``execute`` no longer trips the param error.
#
#   D. data_drift_run_click_reports_error
#         A bmesh-driven topology change makes ``compute_data_hash``
#         diverge. ``Run.execute`` reports a "Transfer" error and
#         returns CANCELLED (the data path is the louder of the two
#         click-time errors because re-transferring is the only fix).

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


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


class _StubSelf:
    # Run.execute calls only self.report and self.setup_modal; both
    # are easy to fake so we can capture what the user would see in
    # the UI without actually starting a solver run.
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
    root.state.gravity_3d = (0.0, 0.0, -4.9)
    s = dh.facade.engine.state
    fresh_param_b = encoder_params.compute_param_hash(bpy.context)
    stub_b = _StubSelf()
    verdict_b = Run.execute(stub_b, bpy.context)
    error_b = next((m for k, m in stub_b.captured if "ERROR" in k), "")
    dh.record(
        "B_param_drift_run_click_reports_error",
        bool(Run.poll(bpy.context))
        and fresh_param_b != s.server_param_hash
        and verdict_b == {"CANCELLED"}
        and "Update Params" in error_b
        and not stub_b.modal_set_up,
        {
            "poll": bool(Run.poll(bpy.context)),
            "fresh_param": fresh_param_b[:12],
            "server_param": s.server_param_hash[:12],
            "verdict": list(verdict_b),
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
    poll_pre_run_c = bool(Run.poll(bpy.context))
    stub_c = _StubSelf()
    verdict_c = Run.execute(stub_c, bpy.context)
    param_error_c = next(
        (m for k, m in stub_c.captured
         if "ERROR" in k and "Update Params" in m), "")
    # ``Run.poll`` is checked *before* ``execute`` -- once execute
    # passes the drift gates it calls ``com.run`` which kicks off
    # the solver, after which poll legitimately returns False
    # because the base "not is_running" guard fires.
    dh.record(
        "C_update_params_re_enables_run",
        fresh_param_c == s.server_param_hash
        and not param_error_c
        and poll_pre_run_c
        and stub_c.modal_set_up,
        {
            "fresh_param": fresh_param_c[:12],
            "server_param": s.server_param_hash[:12],
            "poll_pre_run": poll_pre_run_c,
            "modal_set_up": stub_c.modal_set_up,
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
    stub_d = _StubSelf()
    verdict_d = Run.execute(stub_d, bpy.context)
    error_d = next((m for k, m in stub_d.captured if "ERROR" in k), "")
    dh.record(
        "D_data_drift_run_click_reports_error",
        verdict_d == {"CANCELLED"}
        and "Transfer" in error_d
        and fresh_data_d != s.server_data_hash
        and not stub_d.modal_set_up,
        {
            "verdict": list(verdict_d),
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
