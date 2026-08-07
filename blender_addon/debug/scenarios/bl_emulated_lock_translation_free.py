# File: scenarios/bl_emulated_lock_translation_free.py
# License: Apache v2.0
#
# Exact Lock Translation in the emulator's implicit ARAP global solve.
# A free sheet receives gravity with allowed X and forbidden Z components.
# Its physical mass-weighted center of mass must move along X only.

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
FRAME_COUNT = 20


def _shell_mass_weights(obj):
    obj.data.calc_loop_triangles()
    co = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
    weights = np.zeros(len(co), dtype=np.float64)
    for tri in obj.data.loop_triangles:
        i, j, k = tri.vertices
        area = 0.5 * np.linalg.norm(np.cross(co[j] - co[i], co[k] - co[i]))
        weights[[i, j, k]] += area / 3.0
    return weights


try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=7, y_subdivisions=7, size=2.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "LockedFreeSheet"
    weights = _shell_mass_weights(sheet)

    root = dh.configure_state(
        project_name="emulated_lock_translation_free",
        frame_count=FRAME_COUNT,
        gravity=(4.0, 0.0, -9.8),
    )
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    group = root.object_group_0
    assigned = next(a for a in group.assigned_objects if a.name == sheet.name)
    assigned.lock_translation_enable = True
    assigned.lock_translation_axis = (1.0, 0.0, 0.0)

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH, server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.build_and_wait(data_bytes, param_bytes, "lock-free:build", timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.fetch_and_drain()

    pc2_path = dh.find_pc2_for(sheet)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced (path={pc2_path!r})")
    arr = dh.read_pc2(pc2_path).astype(np.float64)
    com = np.einsum("fvc,v->fc", arr, weights) / np.sum(weights)
    drift = com - com[0]

    perpendicular = float(np.max(np.linalg.norm(drift[:, 1:3], axis=1)))
    allowed = float(drift[-1, 0])
    dh.record(
        "A_forbidden_com_motion_removed",
        perpendicular < 2e-5,
        {"max_perpendicular_com_drift": perpendicular},
    )
    dh.record(
        "B_allowed_axis_motion_preserved",
        allowed > 1e-3,
        {"final_x_com_displacement": allowed},
    )
    dh.record(
        "C_output_finite",
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
