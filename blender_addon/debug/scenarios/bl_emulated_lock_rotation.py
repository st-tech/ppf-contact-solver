# File: scenarios/bl_emulated_lock_rotation.py
# License: Apache v2.0
#
# Emulator physics coverage for coexisting Translation and Rotation locks.
# Gravity may translate the sheet along X. An injected rigid spin about X is
# forbidden because Rotation Lock permits only rotation about world Z.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
KNOBS = {"PPF_EMULATED_ELASTIC": "1", "PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 16

try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=7, y_subdivisions=7, size=2.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "MotionLockedSheet"

    root = dh.configure_state(
        project_name="emulated_lock_rotation",
        frame_count=FRAME_COUNT,
        gravity=(4.0, 0.0, -9.8),
    )
    group_api = dh.api.solver.create_group("Sheet", "SHELL")
    group_api.add(sheet.name)
    group_api.set_velocity(
        sheet.name,
        direction=(0.0, 0.0, 0.0),
        speed=0.0,
        frame=1,
        angular_axis="X",
        angular_speed=360.0,
        enable_translational=False,
        enable_angular=True,
    )
    group = root.object_group_0
    assigned = next(a for a in group.assigned_objects if a.name == sheet.name)
    assigned.lock_translation_enable = True
    assigned.lock_translation_axis = (1.0, 0.0, 0.0)
    assigned.lock_rotation_enable = True
    assigned.lock_rotation_axis = (0.0, 0.0, 1.0)

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH, server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.build_and_wait(data_bytes, param_bytes, "motion-lock:build", timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.fetch_and_drain()

    pc2_path = dh.find_pc2_for(sheet)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced (path={pc2_path!r})")
    arr = dh.read_pc2(pc2_path).astype(np.float64)
    rest = arr[0]
    last = arr[-1]
    c0 = rest.mean(axis=0)
    c1 = last.mean(axis=0)
    com_delta = c1 - c0
    r0 = rest - c0
    r1 = last - c1
    relative_change = float(np.max(np.linalg.norm(r1 - r0, axis=1)))

    dh.record(
        "A_translation_lock_coexists",
        abs(com_delta[1]) < 2e-5 and abs(com_delta[2]) < 2e-5,
        {"com_delta": com_delta.tolist()},
    )
    dh.record(
        "B_allowed_translation_preserved",
        com_delta[0] > 1e-3,
        {"x_displacement": float(com_delta[0])},
    )
    dh.record(
        "C_forbidden_rotation_removed",
        relative_change < 2e-3,
        {"max_relative_position_change": relative_change},
    )
    dh.record(
        "D_output_finite",
        bool(np.all(np.isfinite(arr))),
        {"max_abs": float(np.max(np.abs(arr)))},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        (dl.DRIVER_LIB + _DRIVER_BODY)
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
