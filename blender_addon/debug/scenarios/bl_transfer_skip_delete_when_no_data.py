# File: scenarios/bl_transfer_skip_delete_when_no_data.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression: clicking Transfer on a fresh project (server reports
# ``data="NO_DATA"``) used to dispatch a "Deleting Remote Data..."
# query first and then upload. The delete is a no-op the user paid
# for in latency and a misleading status banner. The fix in
# ``SOLVER_OT_Transfer.execute`` short-circuits to the upload path
# when the cached server response already says NO_DATA and we have a
# remote_root.
#
# Subtests:
#   A. fresh_transfer_skips_delete: invoking ``SOLVER_OT_Transfer
#      .execute`` on a fresh connection sets ``self._mode = "pipeline"``
#      directly (the delete-cycle path would set ``self._mode =
#      "delete"`` and only flip to "pipeline" after the delete reply
#      lands).
#   B. no_delete_request_dispatched: the engine's recent_events ring
#      buffer holds no QueryRequested with ``request="delete"``
#      between the operator's execute() call and the SENDING activity
#      that the upload kicked off.
#   C. upload_landed: the modal completes with the project at
#      READY/RESUMABLE — proves the skip didn't break the upload
#      pipeline downstream.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


# Lightweight stand-in for the operator instance. Captures
# ``self.report`` calls and accepts attribute assignment so
# ``self._mode = 'pipeline'`` lands somewhere we can inspect.
class _StubSelf:
    def __init__(self):
        self.captured = []
        self.modal_set_up = False
        self._mode = None
    def report(self, kind, msg):
        self.captured.append((tuple(kind), msg))
    def setup_modal(self, ctx):
        self.modal_set_up = True


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="TransferSkipDeleteMesh")
    dh.save_blend(PROBE_DIR, "transfer_skip_delete.blend")
    root = dh.configure_state(
        project_name="transfer_skip_delete_when_no_data", frame_count=4,
    )
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=3,
                transition="LINEAR")

    solver_mod = __import__(pkg + ".ui.solver",
                            fromlist=["SOLVER_OT_Transfer"])
    Transfer = solver_mod.SOLVER_OT_Transfer
    com = dh.com

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    # Pump a few ticks so the first ServerPolled response lands and
    # populates ``com.info.response['data'] == 'NO_DATA'`` plus
    # ``state.remote_root``. Without this the cached response is empty
    # and the skip-condition wouldn't fire.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if (com.info.response.get("data") == "NO_DATA"
                and dh.facade.engine.state.remote_root):
            break
        time.sleep(0.1)
    dh.log("connected; pre-Transfer state ready")

    pre_data = com.info.response.get("data")
    pre_remote_root = dh.facade.engine.state.remote_root
    pre_event_count = len(dh.facade.engine.recent_events)

    # ----- Invoke Transfer.execute on a fresh project -----------------
    stub = _StubSelf()
    verdict = Transfer.execute(stub, bpy.context)
    dh.facade.tick()
    post_event_count = len(dh.facade.engine.recent_events)
    new_events = dh.facade.engine.recent_events[
        max(0, pre_event_count - len(dh.facade.engine.recent_events)):
    ]
    # The stable signal: between the execute() and the next tick that
    # processes its dispatched event, we should see exactly ONE new
    # event -- BuildPipelineRequested. The skipped path would have
    # dispatched a QueryRequested(request={'request':'delete'}) first.
    new_event_names = [name for _, name, _ in new_events]
    saw_build_pipeline = "BuildPipelineRequested" in new_event_names
    saw_delete_query = any(
        name == "QueryRequested" and "'request': 'delete'" in repr_
        for _, name, repr_ in new_events
    )

    dh.record(
        "A_fresh_transfer_skips_delete",
        verdict == {"RUNNING_MODAL"}
        and stub.modal_set_up
        and stub._mode == "pipeline"
        and pre_data == "NO_DATA",
        {
            "verdict": list(verdict),
            "modal_set_up": stub.modal_set_up,
            "stub_mode": stub._mode,
            "pre_data": pre_data,
            "pre_remote_root": pre_remote_root,
            "captured_reports": stub.captured[:3],
        },
    )

    dh.record(
        "B_no_delete_request_dispatched",
        saw_build_pipeline and not saw_delete_query,
        {
            "new_event_names": new_event_names,
            "saw_build_pipeline": saw_build_pipeline,
            "saw_delete_query": saw_delete_query,
        },
    )

    # ----- Drain the upload pipeline to terminal --------------------
    deadline = time.time() + 90.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.2)
    s = dh.facade.engine.state
    dh.record(
        "C_upload_landed",
        s.solver.name in ("READY", "RESUMABLE")
        and bool(s.server_upload_id),
        {
            "solver": s.solver.name,
            "activity": s.activity.name,
            "server_upload_id": s.server_upload_id[:12] if s.server_upload_id else "",
            "error": s.error,
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
