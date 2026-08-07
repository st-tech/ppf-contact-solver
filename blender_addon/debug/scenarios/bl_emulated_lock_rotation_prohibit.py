# File: scenarios/bl_emulated_lock_rotation_prohibit.py
# License: Apache v2.0
#
# Prohibit-axis mode keeps rotation perpendicular to the selected axis while
# a coexisting Translation Lock independently constrains center-of-mass motion.

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
FRAME_COUNT = 12

try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=7, y_subdivisions=7, size=2.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "ProhibitAxisSheet"

    root = dh.configure_state(
        project_name="emulated_lock_rotation_prohibit",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -4.0),
    )
    group_api = dh.api.solver.create_group("Sheet", "SHELL")
    group_api.add(sheet.name)
    group_api.set_velocity(
        sheet.name,
        direction=(0.0, 0.0, 0.0),
        speed=0.0,
        frame=1,
        angular_axis="CUSTOM",
        angular_axis_custom=(1.0, 0.0, 1.0),
        angular_speed=180.0,
        enable_translational=False,
        enable_angular=True,
    )
    group = root.object_group_0
    assigned = next(a for a in group.assigned_objects if a.name == sheet.name)
    assigned.lock_translation_enable = True
    assigned.lock_translation_axis = (0.0, 0.0, 1.0)
    assigned.lock_rotation_enable = True
    assigned.lock_rotation_prohibit_axis = True
    assigned.lock_rotation_axis = (1.0, 0.0, 0.0)

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH, server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.build_and_wait(data_bytes, param_bytes, "rotation-prohibit:build", timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.fetch_and_drain()

    pc2_path = dh.find_pc2_for(sheet)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced (path={pc2_path!r})")
    arr = dh.read_pc2(pc2_path).astype(np.float64)
    com_delta = arr[-1].mean(axis=0) - arr[0].mean(axis=0)
    a = arr[0] - arr[0].mean(axis=0)
    b = arr[-1] - arr[-1].mean(axis=0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    vector = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    axis = vector / (2.0 * np.sin(angle)) if abs(np.sin(angle)) > 1e-8 else vector

    dh.record(
        "A_translation_lock_removes_perpendicular_com_motion",
        abs(float(com_delta[0])) < 2e-5 and abs(float(com_delta[1])) < 2e-5,
        {"com_delta": com_delta.tolist()},
    )
    dh.record(
        "B_translation_lock_preserves_axis_motion",
        float(com_delta[2]) < -1e-3,
        {"com_delta": com_delta.tolist()},
    )
    dh.record(
        "C_prohibited_x_component_removed",
        abs(float(axis[0])) < 0.05,
        {"axis": axis.tolist(), "angle_deg": float(np.degrees(angle))},
    )
    dh.record(
        "D_perpendicular_rotation_preserved",
        angle > np.radians(5.0) and abs(float(axis[2])) > 0.9,
        {"axis": axis.tolist(), "angle_deg": float(np.degrees(angle))},
    )
    dh.record(
        "E_output_finite",
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
