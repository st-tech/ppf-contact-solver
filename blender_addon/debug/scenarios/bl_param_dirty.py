# File: scenarios/bl_param_dirty.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Verifies the click-time parameter-drift contract that replaced the
# debounced ``param_hash_cache``.
#
# Mechanism under test:
#   - ``encoder.params.compute_param_hash`` produces a stable
#     fingerprint of the live param dict.
#   - The wire protocol carries ``param_hash`` on every upload and
#     echoes it back in every status response (server stores it next
#     to ``upload_id.txt``).
#   - ``SOLVER_OT_UpdateParams.poll`` is *always* True when the base
#     connection / status guards are satisfied -- the button does not
#     compare hashes itself, so a stale client-side cache cannot grey
#     it out (the source of the bug fixed in this commit set).
#   - ``SOLVER_OT_Run.execute`` recomputes ``compute_param_hash`` at
#     click time and refuses with an ERROR ("Click Update Params
#     before running") whenever the fresh hash differs from
#     ``engine.state.server_param_hash``.
#
# Subtests:
#   A. fresh_upload_update_params_clickable: after a successful
#      Transfer, Update Params poll is True (always-clickable), Run
#      poll is True, and a fresh ``compute_param_hash`` matches what
#      the server echoed.
#   B. param_edit_keeps_update_params_clickable: mutating a tracked
#      parameter (gravity_3d) still leaves Update Params clickable
#      and Run.poll True; the drift is *only* visible via the fresh
#      hash compare, exactly what the click-time check uses.
#   C. update_params_realigns: invoking ``solver.update_params``
#      pushes the new hash; once the server echoes it, the fresh
#      hash matches the server hash again.

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


def _ui_solver():
    return __import__(pkg + ".ui.solver",
                      fromlist=["SOLVER_OT_UpdateParams", "SOLVER_OT_Run"])


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="ParamDirtyMesh")
    dh.save_blend(PROBE_DIR, "paramdirty.blend")
    root = dh.configure_state(project_name="param_dirty", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    encoder_pkg = __import__(pkg + ".core.encoder",
                             fromlist=["prepare_upload"])

    data_bytes, param_bytes, _, pre_param_hash = encoder_pkg.prepare_upload(
        bpy.context,
    )
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        param_hash=pre_param_hash, message="param-dirty:build",
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
           f"server_hash={dh.facade.engine.state.server_param_hash[:12]!r}")

    UpdateParams = _ui_solver().SOLVER_OT_UpdateParams
    Run = _ui_solver().SOLVER_OT_Run

    # ----- A: fresh upload, Update Params is always-clickable ------
    fresh_a = encoder_params.compute_param_hash(bpy.context)
    server_a = dh.facade.engine.state.server_param_hash
    dh.record(
        "A_fresh_upload_update_params_clickable",
        bool(UpdateParams.poll(bpy.context))
        and bool(Run.poll(bpy.context))
        and fresh_a == server_a
        and fresh_a == pre_param_hash,
        {
            "update_params_poll": bool(UpdateParams.poll(bpy.context)),
            "run_poll": bool(Run.poll(bpy.context)),
            "fresh": fresh_a[:12],
            "server": server_a[:12],
        },
    )

    # ----- B: param edit, Update Params still clickable ------------
    # Drift is detectable via fresh-vs-server hash compare (which is
    # exactly what Run.execute does at click time), but neither poll
    # gates on it. Run.poll stays True because the "Click Update
    # Params" message is delivered as an execute-time error report,
    # not a poll-time grey-out.
    root.state.gravity_3d = (0.0, 0.0, -4.9)
    fresh_b = encoder_params.compute_param_hash(bpy.context)
    server_b = dh.facade.engine.state.server_param_hash
    dh.record(
        "B_param_edit_keeps_update_params_clickable",
        bool(UpdateParams.poll(bpy.context))
        and bool(Run.poll(bpy.context))
        and fresh_b != server_b
        and fresh_b != fresh_a,
        {
            "update_params_poll": bool(UpdateParams.poll(bpy.context)),
            "run_poll": bool(Run.poll(bpy.context)),
            "fresh": fresh_b[:12],
            "server": server_b[:12],
            "drifted_from_a": fresh_b != fresh_a,
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
    fresh_c = encoder_params.compute_param_hash(bpy.context)
    server_c = dh.facade.engine.state.server_param_hash
    dh.record(
        "C_update_params_realigns",
        fresh_c == server_c
        and server_c != server_a
        and bool(Run.poll(bpy.context)),
        {
            "fresh": fresh_c[:12],
            "server": server_c[:12],
            "server_changed_from_a": server_c != server_a,
            "run_poll": bool(Run.poll(bpy.context)),
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
