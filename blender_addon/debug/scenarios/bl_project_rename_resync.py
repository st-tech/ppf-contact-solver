# File: scenarios/bl_project_rename_resync.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Project rename mid-session: ``com.set_project_name`` flips the
# runner's project_name so that subsequent server requests carry the
# new name on the wire. Note that it does NOT, on its own, retarget
# the on-disk upload location: ``state.remote_root`` is captured at
# connect time. The supported "switch projects" flow is rename +
# disconnect + fresh ``connect_local`` under the new name.
#
# This scenario covers all three pieces of that contract end-to-end:
# the first transfer under proj_A, the runner-side rename, and a
# disconnect + reconnect + build under proj_B that lands a real
# upload at proj_B's on-disk path while leaving proj_A's tree intact.
#
# Subtests:
#   A. ``proj_a_initial_upload``: after the first transfer + build
#      under ``project_name="proj_A"``, the worker's project root for
#      ``proj_A`` carries ``data.pickle`` + ``param.pickle`` +
#      ``upload_id.txt``.
#   B. ``runner_resyncs_on_rename``: after
#      ``state.project_name = "proj_B"`` plus
#      ``com.set_project_name("proj_B")``, the runner's stored
#      project_name flips to ``"proj_B"``. Outgoing requests now
#      carry the new name as the server-side ``name`` arg.
#   C. ``proj_b_reconnect_relocates_upload``: after
#      ``com.disconnect()`` + ``connect_local(project_name="proj_B")``
#      + a fresh ``BuildPipelineRequested``, the worker's proj_B
#      project root carries its own ``data.pickle`` + ``param.pickle``
#      + a distinct ``upload_id.txt``, while proj_A's tree (data
#      pickle and original upload_id) is preserved untouched.

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


def _project_root_for(probe_dir, name):
    # PROBE_DIR is ``<workspace>/probe``; the worker's
    # PPF_CTS_DATA_ROOT shadow lives at ``<workspace>/project`` and
    # uploads land under ``<shadow>/git-debug/<project_name>/`` (see
    # server/emulator.py's BlenderApp patch).
    workspace = os.path.dirname(probe_dir)
    return os.path.join(workspace, "project", "git-debug", name)


def _read_upload_id(root):
    path = os.path.join(root, "upload_id.txt")
    if not os.path.isfile(path):
        return ""
    with open(path) as f:
        return f.read().strip()


def _build_under(dh, *, message, timeout=90.0):
    encoder_mesh = __import__(dh.pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(dh.pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    data_bytes, param_bytes = dh.encode_payload()
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message=message,
    ))
    deadline = time.time() + timeout
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            return s.solver.name
        time.sleep(0.3)
    return dh.facade.engine.state.solver.name


def _disconnect_and_wait(dh, *, timeout=10.0):
    dh.com.disconnect()
    deadline = time.time() + timeout
    while time.time() < deadline:
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.phase.name == "OFFLINE":
            return True
        time.sleep(0.05)
    return False


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="RenameMesh")
    dh.save_blend(PROBE_DIR, "rename.blend")
    root = dh.configure_state(project_name="proj_A", frame_count=4)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.05, 0.0, 0.0), frame_start=1, frame_end=2,
                transition="LINEAR")

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected_as_proj_A")

    solver_a = _build_under(dh, message="rename:proj_A:build")
    dh.log(f"proj_A built solver={solver_a}")

    proj_root_a = _project_root_for(PROBE_DIR, "proj_A")
    data_a = os.path.join(proj_root_a, "data.pickle")
    param_a = os.path.join(proj_root_a, "param.pickle")
    upload_id_a = _read_upload_id(proj_root_a)
    data_a_mtime = os.path.getmtime(data_a) if os.path.isfile(data_a) else 0.0

    # ----- A: proj_A's first upload landed --------------------------
    dh.record(
        "A_proj_a_initial_upload",
        os.path.isfile(data_a)
        and os.path.isfile(param_a)
        and bool(upload_id_a)
        and dh.facade.runner.project_name == "proj_A",
        {
            "proj_root_a": proj_root_a,
            "data_exists": os.path.isfile(data_a),
            "param_exists": os.path.isfile(param_a),
            "upload_id_a": upload_id_a,
            "runner_project": dh.facade.runner.project_name,
        },
    )

    # ----- B: rename + set_project_name flips the runner ------------
    root.state.project_name = "proj_B"
    dh.client.communicator.set_project_name("proj_B")
    # One facade tick lets any side-effects of the property write
    # settle before we sample state. set_project_name itself is a
    # direct attribute assignment on the runner, so a simple read is
    # enough; the tick is for any future indirection.
    dh.facade.tick()
    runner_after_rename = dh.facade.runner.project_name
    dh.record(
        "B_runner_resyncs_on_rename",
        runner_after_rename == "proj_B"
        and root.state.project_name == "proj_B",
        {
            "runner_project": runner_after_rename,
            "state_project": root.state.project_name,
        },
    )

    # ----- C: rename + reconnect retargets the on-disk upload ------
    # Disconnect drops the backend and clears runner._project_name;
    # _reset_state also wipes state.remote_root and state.project_name.
    # Reconnecting via dh.connect_local re-arms the runner with the new
    # project_name and re-binds state.remote_root to the server-reported
    # proj_B root on the next poll. A fresh BuildPipelineRequested then
    # writes data.pickle + param.pickle under proj_B's tree without
    # touching proj_A's.
    disconnected = _disconnect_and_wait(dh)
    phase_after_disconnect = dh.facade.engine.state.phase.name
    runner_after_disconnect = dh.facade.runner.project_name

    # The state reset clears state.project_name. Re-seat it for proj_B
    # before reconnect so the addon's UI bindings stay coherent. The
    # connect helper itself sets the runner-side project_name through
    # com.set_project_name.
    root.state.project_name = "proj_B"
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name="proj_B")
    dh.log("reconnected_as_proj_B")

    solver_b = _build_under(dh, message="rename:proj_B:build")
    dh.log(f"proj_B built solver={solver_b}")

    proj_root_b = _project_root_for(PROBE_DIR, "proj_B")
    data_b = os.path.join(proj_root_b, "data.pickle")
    param_b = os.path.join(proj_root_b, "param.pickle")
    upload_id_b = _read_upload_id(proj_root_b)

    proj_a_data_still_present = os.path.isfile(data_a)
    proj_a_upload_id_unchanged = _read_upload_id(proj_root_a) == upload_id_a
    proj_a_data_mtime_unchanged = (
        os.path.isfile(data_a)
        and os.path.getmtime(data_a) == data_a_mtime
    )

    dh.record(
        "C_proj_b_reconnect_relocates_upload",
        disconnected
        and phase_after_disconnect == "OFFLINE"
        and not runner_after_disconnect
        and solver_b in ("READY", "RESUMABLE")
        and os.path.isfile(data_b)
        and os.path.isfile(param_b)
        and bool(upload_id_b)
        and upload_id_b != upload_id_a
        and dh.facade.runner.project_name == "proj_B"
        and proj_a_data_still_present
        and proj_a_upload_id_unchanged
        and proj_a_data_mtime_unchanged,
        {
            "disconnected": disconnected,
            "phase_after_disconnect": phase_after_disconnect,
            "runner_after_disconnect": runner_after_disconnect,
            "proj_root_b": proj_root_b,
            "data_b_exists": os.path.isfile(data_b),
            "param_b_exists": os.path.isfile(param_b),
            "upload_id_a": upload_id_a,
            "upload_id_b": upload_id_b,
            "solver_b": solver_b,
            "runner_project": dh.facade.runner.project_name,
            "proj_a_data_still_present": proj_a_data_still_present,
            "proj_a_upload_id_unchanged": proj_a_upload_id_unchanged,
            "proj_a_data_mtime_unchanged": proj_a_data_mtime_unchanged,
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
