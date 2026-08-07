# File: scenarios/bl_solver_crash_open_session_folder.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A crash report is a summary plus a detail plus two log tails, and only the
# first two fit in a panel. `Open Session Folder` is how a user reaches the
# rest: the solver's `stdout.log`, `error.log` and `status.cbor` all live in
# `<remote_root>/session`.
#
# Subtests:
#   A. operator_registered: the operator class is registered, so the panel
#      row can reference it. A new operator class is new RNA and needs a full
#      Blender restart, not a soft reload, so this also catches a stale
#      registration.
#   B. path_is_the_session_dir: the computed path is `<remote_root>/session`,
#      the directory the server writes those three files into.
#   C. poll_requires_failed_and_root: the operator is enabled only while a
#      failure is being reported AND a root is known. It must not offer to
#      open a folder that does not belong to a failed run.
#   D. no_root_yields_no_path: with no remote root the path is empty, which
#      is what the panel's disabled branch keys off.
#
# The operator is never invoked here: `wm.path_open` hands the path to the
# desktop environment, and opening a window in the rig would leave it behind.
#
# Pure UI scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import time, traceback
import bpy
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    ops_mod = __import__(pkg + ".ui.solver_control_ops",
                         fromlist=["SOLVER_OT_OpenSessionFolder"])
    status_mod = __import__(pkg + ".core.status", fromlist=["RemoteStatus"])
    client_mod = __import__(pkg + ".core.client", fromlist=["communicator"])

    OpCls = ops_mod.SOLVER_OT_OpenSessionFolder
    RemoteStatus = status_mod.RemoteStatus
    com = client_mod.communicator

    # ----- A: the operator is registered --------------------------------
    # Probe `bpy.ops`, not `bpy.types`. A registered operator is reachable
    # under its bl_idname and is NOT exposed on bpy.types by its class name,
    # so a bpy.types probe reports False for every operator in this add-on
    # and would fail here no matter what the operator did.
    registered = hasattr(bpy.ops.solver, "open_session_folder")
    record(
        "A_operator_registered",
        registered and OpCls.bl_idname == "solver.open_session_folder",
        {"bl_idname": OpCls.bl_idname, "registered": registered},
    )

    # The operator reads the live communicator, so drive it by stubbing the
    # two values it consults and restoring them afterwards.
    class _Info:
        def __init__(self, status):
            self.status = status

    saved_info = type(com).info

    def stub(status, root):
        type(com).info = property(lambda self, s=status: _Info(s))
        # Shadows the bound method with an instance attribute; the finally
        # below deletes the attribute rather than reassigning the method,
        # which would leave the shadow in place for the rest of the session.
        com.normalized_remote_root = lambda r=root: r

    try:
        # ----- B: the path is the session directory ---------------------
        stub(RemoteStatus.SIMULATION_FAILED, "/tmp/ppf-project")
        record(
            "B_path_is_the_session_dir",
            OpCls.session_path() == "/tmp/ppf-project/session",
            {"path": OpCls.session_path()},
        )

        # ----- C: poll requires a failure AND a known root --------------
        failed_with_root = OpCls.poll(bpy.context)
        stub(RemoteStatus.READY, "/tmp/ppf-project")
        ready_with_root = OpCls.poll(bpy.context)
        stub(RemoteStatus.SIMULATION_IN_PROGRESS, "/tmp/ppf-project")
        running_with_root = OpCls.poll(bpy.context)
        record(
            "C_poll_requires_failed_and_root",
            failed_with_root and not ready_with_root and not running_with_root,
            {
                "failed": bool(failed_with_root),
                "ready": bool(ready_with_root),
                "running": bool(running_with_root),
            },
        )

        # ----- D: no root means no path, and no offer to open one -------
        stub(RemoteStatus.SIMULATION_FAILED, "")
        record(
            "D_no_root_yields_no_path",
            OpCls.session_path() == "" and not OpCls.poll(bpy.context),
            {"path": OpCls.session_path()},
        )
    finally:
        type(com).info = saved_info
        com.__dict__.pop("normalized_remote_root", None)

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
