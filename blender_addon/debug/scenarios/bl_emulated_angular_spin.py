# File: scenarios/bl_emulated_angular_spin.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Angular velocity overwrite, exercised end-to-end through the CUDA-free
# emulator's implicit ARAP elastic solver
# (crates/ppf-cts-solver/src/cpp_emul/pd_arap.hpp) on a no-GPU host.
#
# A flat, UNPINNED shell grid is given a single angular-velocity-overwrite
# keyframe at frame 1: spin about its 3rd principal axis (PC3 = the sheet
# normal) at 360 deg/s. The solver step loop (backend.rs) resolves that
# principal axis from the LIVE geometry, gathers the body centroid, and
# injects a rigid spin field ω × (x - c) into vertex.prev; the emulated
# elastic step then reads that as the incoming velocity and integrates it,
# so the sheet rotates in its own plane and carries the spin by momentum.
#
# Asserts:
#   A. pc2_has_all_frames   - the run produced every output frame.
#   B. body_rotated         - a corner vertex's angle about the centroid
#                             advanced clearly (the override actually spun
#                             the body; impossible with the old no-op
#                             override or the kinematic-only emulator).
#   C. rigid_and_centered   - the spin is rigid (corner radius preserved)
#                             and pure (centroid barely moved: ω × r carries
#                             no net linear momentum).
#   D. simulation_stable    - every position stays finite and bounded.
#
# This is the emulator-side companion to bl_velocity_keyframes.py (which
# checks the encoder payload only). Together they cover the angular path
# from the Blender keyframe through to a solved rotation.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Turn on the emulator's implicit ARAP solver and pace steps as fast as
# possible (0 ms/step) since this is a unit-style test.
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
# 360 deg/s about PC3; configure_state defaults dt=0.01, fps=100, so 24
# frames is ~0.24 s -> ~86 deg of ideal rotation. The threshold below
# leaves a wide margin for the implicit integrator's numerical damping.
SPIN_DEG_PER_S = 360.0


def _build_spinner_scene():
    # A free flat grid centered at the origin; no pins, no gravity, so the
    # only motion is the injected spin. size=2 -> coords span [-1, 1].
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=8, y_subdivisions=8, size=2, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "SpinSheet"
    return sheet


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    sheet = _build_spinner_scene()
    dh.save_blend(PROBE_DIR, "emulated_angular_spin.blend")
    # configure_state defaults: dt=0.01, fps=100, gravity=(0,0,0).
    root = dh.configure_state(
        project_name="emulated_angular_spin",
        frame_count=FRAME_COUNT,
    )

    spinner = dh.api.solver.create_group("Spinner", "SHELL")
    spinner.add(sheet.name)
    # Pure spin: translational overwrite DISABLED, angular ENABLED about PC3
    # (the sheet normal), authored at frame 1 so it fires at t=0 on the first
    # solver step. With enable_translational=False the linear override never
    # runs, so the translation is left untouched (here it stays at rest) and
    # only the spin is injected.
    spinner.set_velocity(
        sheet.name,
        direction=(0.0, 0.0, 0.0),
        speed=0.0,
        frame=1,
        angular_axis=2,
        angular_speed=SPIN_DEG_PER_S,
        enable_translational=False,
        enable_angular=True,
    )

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _pre_data_hash, _pre_param_hash = (
        encoder_pkg.prepare_upload(bpy.context)
    )

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    dh.build_and_wait(data_bytes, param_bytes, "angular-spin:build",
                      timeout=120.0)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    saw_running = dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.log(f"ran saw_running={saw_running} frame={dh.facade.engine.state.frame}")

    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    pc2_path = dh.find_pc2_for(sheet)
    if not pc2_path or not os.path.isfile(pc2_path):
        raise RuntimeError(f"no PC2 produced (path={pc2_path!r})")
    arr = dh.read_pc2(pc2_path)  # (n_samples, n_verts, 3)
    dh.log(f"pc2 shape={arr.shape}")

    rest = arr[0]
    last = arr[-1]
    c_rest = rest.mean(axis=0)
    c_last = last.mean(axis=0)

    # Track the vertex farthest from the centroid (a corner) and measure how
    # far its centroid-relative direction rotated. The angle is taken about
    # each frame's own centroid so any translation is removed; for a corner
    # (perpendicular to the spin axis) this angle is the rotation angle.
    r_rest = rest - c_rest
    corner = int(np.argmax(np.linalg.norm(r_rest, axis=1)))
    u0 = rest[corner] - c_rest
    u1 = last[corner] - c_last
    n0 = float(np.linalg.norm(u0))
    n1 = float(np.linalg.norm(u1))
    if n0 > 1e-6 and n1 > 1e-6:
        cosang = float(np.dot(u0, u1) / (n0 * n1))
        cosang = max(-1.0, min(1.0, cosang))
        angle_deg = float(np.degrees(np.arccos(cosang)))
    else:
        angle_deg = 0.0

    # A: every frame present.
    dh.record(
        "A_pc2_has_all_frames",
        arr.shape[0] == FRAME_COUNT,
        {"pc2_samples": int(arr.shape[0]), "expected": FRAME_COUNT},
    )

    # B: the body clearly rotated. A single override step alone is ~3.6 deg;
    # > 10 deg proves the spin was injected AND carried by momentum across
    # steps (the whole point of the feature).
    dh.record(
        "B_body_rotated",
        angle_deg > 10.0,
        {"corner_angle_deg": round(angle_deg, 3), "corner_index": corner},
    )

    # C: the motion is a rigid, centered spin: corner radius preserved
    # (ARAP held the shape), and the centroid barely moved (ω × r carries no
    # net linear momentum, so a pure spin does not translate the body).
    radius_ok = n0 > 1e-6 and abs(n1 - n0) / n0 < 0.05
    centroid_disp = float(np.linalg.norm(c_last - c_rest))
    dh.record(
        "C_rigid_and_centered",
        radius_ok and centroid_disp < 0.05,
        {
            "radius_rest": round(n0, 5),
            "radius_last": round(n1, 5),
            "centroid_disp": round(centroid_disp, 5),
        },
    )

    # D: stable - all coordinates finite and bounded across all frames.
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
    repo_root = REPO_ROOT_POSIX
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
