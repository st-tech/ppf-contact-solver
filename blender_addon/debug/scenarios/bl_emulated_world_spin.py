# File: scenarios/bl_emulated_world_spin.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Fixed WORLD-axis angular velocity overwrite, through the CUDA-free
# emulator's implicit ARAP elastic solver on a no-GPU host.
#
# Companion to bl_emulated_angular_spin.py (which spins about a dynamic
# principal axis). Here the keyframe spins a flat shell grid about the fixed
# **World X** axis, which is IN-PLANE for the grid, so the sheet tumbles out
# of its rest plane. This exercises the separate world-axis solver path
# (scene.get_angular_velocity_world_overrides + the backend world loop): the
# ω vector is given directly (no principal-axis solve), only the live centroid
# is gathered for the pivot.
#
# Asserts:
#   A. pc2_has_all_frames   - the run produced every output frame.
#   B. body_rotated         - a corner vertex's angle about the centroid
#                             advanced clearly.
#   C. rotation_about_x     - the rotation is about World X: every vertex's
#                             X coordinate relative to the centroid is
#                             preserved (the defining property of a spin about
#                             X), while the motion is in Y/Z. This proves the
#                             chosen world axis is honored (not, say, PC3).
#   D. simulation_stable    - every position stays finite and bounded.

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
FRAME_COUNT = 24
SPIN_DEG_PER_S = 360.0


def _build_spinner_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=8, y_subdivisions=8, size=2, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "WorldSpinSheet"
    return sheet


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    sheet = _build_spinner_scene()
    dh.save_blend(PROBE_DIR, "emulated_world_spin.blend")
    root = dh.configure_state(
        project_name="emulated_world_spin",
        frame_count=FRAME_COUNT,
    )

    spinner = dh.api.solver.create_group("Spinner", "SHELL")
    spinner.add(sheet.name)
    # Pure spin about the FIXED World X axis (in-plane for this flat grid, so
    # it tumbles out of plane). enable_translational=False -> pure spin.
    spinner.set_velocity(
        sheet.name,
        direction=(0.0, 0.0, 0.0),
        speed=0.0,
        frame=1,
        angular_axis="X",
        angular_speed=SPIN_DEG_PER_S,
        enable_translational=False,
        enable_angular=True,
    )

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _h1, _h2 = encoder_pkg.prepare_upload(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "world-spin:build", timeout=120.0)
    saw_running = dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    pc2_path = dh.find_pc2_for(sheet)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced (path={pc2_path!r})")
    arr = dh.read_pc2(pc2_path)
    dh.log(f"pc2 shape={arr.shape}")

    rest = arr[0]
    last = arr[-1]
    c_rest = rest.mean(axis=0)
    c_last = last.mean(axis=0)

    r_rest = rest - c_rest
    r_last = last - c_last
    corner = int(np.argmax(np.linalg.norm(r_rest, axis=1)))
    u0 = r_rest[corner]
    u1 = r_last[corner]
    n0 = float(np.linalg.norm(u0))
    n1 = float(np.linalg.norm(u1))
    if n0 > 1e-6 and n1 > 1e-6:
        cosang = max(-1.0, min(1.0, float(np.dot(u0, u1) / (n0 * n1))))
        angle_deg = float(np.degrees(np.arccos(cosang)))
    else:
        angle_deg = 0.0

    # A: every frame present.
    dh.record(
        "A_pc2_has_all_frames",
        arr.shape[0] == FRAME_COUNT,
        {"pc2_samples": int(arr.shape[0]), "expected": FRAME_COUNT},
    )

    # B: clearly rotated.
    dh.record(
        "B_body_rotated",
        angle_deg > 10.0,
        {"corner_angle_deg": round(angle_deg, 3), "corner_index": corner},
    )

    # C: rotation is about World X -> the X component (relative to centroid)
    # is preserved for every vertex, while the body actually moved in Y/Z.
    extent = float(np.max(np.linalg.norm(r_rest, axis=1))) or 1.0
    max_dx_rel = float(np.max(np.abs(r_last[:, 0] - r_rest[:, 0])))
    yz_motion = float(np.max(np.abs(r_last[:, 1:] - r_rest[:, 1:])))
    radius_ok = n0 > 1e-6 and abs(n1 - n0) / n0 < 0.05
    centroid_disp = float(np.linalg.norm(c_last - c_rest))
    about_x_ok = (
        (max_dx_rel / extent) < 0.05    # X relative coord preserved
        and yz_motion > 0.1 * extent    # but it really moved in Y/Z
        and radius_ok
        and centroid_disp < 0.05
    )
    dh.record(
        "C_rotation_about_x",
        about_x_ok,
        {
            "max_dx_rel_over_extent": round(max_dx_rel / extent, 5),
            "yz_motion_over_extent": round(yz_motion / extent, 5),
            "radius_rest": round(n0, 5),
            "radius_last": round(n1, 5),
            "centroid_disp": round(centroid_disp, 5),
        },
    )

    # D: stable.
    finite = bool(np.all(np.isfinite(arr)))
    bounded = float(np.max(np.abs(arr))) < 100.0
    dh.record(
        "D_simulation_stable",
        finite and bounded,
        {"finite": finite, "max_abs": round(float(np.max(np.abs(arr))), 4)},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
