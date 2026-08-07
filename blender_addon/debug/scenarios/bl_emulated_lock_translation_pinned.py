# File: scenarios/bl_emulated_lock_translation_pinned.py
# License: Apache v2.0
#
# Affine Lock Translation composition with exact moving pins in the emulator.
# A pinned edge moves perpendicular to the permitted X line. The free vertices
# must compensate while the physical mass-weighted center of mass stays fixed
# in the perpendicular plane.

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
FRAME_COUNT = 18


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
        x_subdivisions=8, y_subdivisions=8, size=2.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "LockedPinnedSheet"
    pinned = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    free = [i for i, v in enumerate(sheet.data.vertices) if v.co.y <= 0.99]
    vg = sheet.vertex_groups.new(name="DrivenEdge")
    vg.add(pinned, 1.0, "REPLACE")
    weights = _shell_mass_weights(sheet)

    root = dh.configure_state(
        project_name="emulated_lock_translation_pinned",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, 0.0),
    )
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    pin = cloth.create_pin(sheet.name, "DrivenEdge")
    pin.move_by(
        delta=(0.0, 0.0, 0.20), frame_start=1, frame_end=10,
        transition="LINEAR",
    )
    group = root.object_group_0
    assigned = next(a for a in group.assigned_objects if a.name == sheet.name)
    assigned.lock_translation_enable = True
    assigned.lock_translation_axis = (1.0, 0.0, 0.0)

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH, server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.build_and_wait(data_bytes, param_bytes, "lock-pinned:build", timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.fetch_and_drain()

    pc2_path = dh.find_pc2_for(sheet)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced (path={pc2_path!r})")
    arr = dh.read_pc2(pc2_path).astype(np.float64)
    com = np.einsum("fvc,v->fc", arr, weights) / np.sum(weights)
    drift = com - com[0]
    pinned_idx = np.asarray(pinned, dtype=int)
    free_idx = np.asarray(free, dtype=int)

    perpendicular = float(np.linalg.norm(drift[-1, 1:3]))
    pin_rise = float(np.mean(arr[-1, pinned_idx, 2] - arr[0, pinned_idx, 2]))
    free_fall = float(np.mean(arr[-1, free_idx, 2] - arr[0, free_idx, 2]))
    dh.record(
        "A_affine_pin_composition_exact",
        perpendicular < 2e-5,
        {"final_perpendicular_com_drift": perpendicular},
    )
    dh.record(
        "B_exact_pins_follow_target",
        pin_rise > 0.15,
        {"mean_pinned_z_displacement": pin_rise},
    )
    dh.record(
        "C_free_vertices_compensate",
        free_fall < -1e-3,
        {"mean_free_z_displacement": free_fall},
    )
    dh.record(
        "D_internal_deformation_preserved",
        float(np.max(np.ptp(arr[-1, :, 2]))) > 0.05,
        {"final_z_range": float(np.ptp(arr[-1, :, 2]))},
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
