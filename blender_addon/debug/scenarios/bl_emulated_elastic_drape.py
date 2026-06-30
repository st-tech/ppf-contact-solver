# File: scenarios/bl_emulated_elastic_drape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Exercises the CUDA-free emulator's implicit ARAP elastic solver
# (crates/ppf-cts-solver/src/cpp_emul/pd_arap.hpp) end-to-end through
# the production pipeline on a no-GPU host (macOS).
#
# The historical emulator only applied kinematic pins, so unpinned
# vertices never moved. With PPF_EMULATED_ELASTIC=1 the emulator runs a
# Projective-Dynamics ARAP step, so a sheet pinned along one edge and
# released under gravity actually DRAPES: free vertices sag while the
# pinned edge holds. This scenario builds exactly that scene, runs the
# emulated solver, fetches the per-frame PC2, and asserts:
#
#   A. pc2_has_all_frames        - the run produced every output frame.
#   B. pinned_edge_held          - pinned-edge vertices stay put.
#   C. free_vertices_draped      - free vertices moved downward (-z),
#                                  i.e. the elastic solver deformed them
#                                  (the whole point; impossible with the
#                                  old kinematic-only emulator).
#   D. simulation_stable         - every position stays finite and
#                                  bounded across all frames (implicit
#                                  integrator stability).
#
# This locks in the macOS elastic test path that the mesh-tearing
# physics scenarios (FEM-strain-driven) will build on.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Turn on the emulator's implicit ARAP solver for this run and pace
# steps as fast as possible (0 ms/step) since this is a unit-style test.
KNOBS = {"PPF_EMULATED_ELASTIC": "1", "PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import os
import time
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 20


def _build_drape_scene():
    # A grid pinned along its +y edge; the rest of the sheet is free to
    # drape under gravity. size=2 -> coords span [-1, 1] in x and y.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=8, y_subdivisions=8, size=2, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "DrapeSheet"

    pinned = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    free = [i for i, v in enumerate(sheet.data.vertices) if v.co.y <= 0.99]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    return sheet, pinned, free


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    sheet, pinned_idx, free_idx = _build_drape_scene()
    dh.save_blend(PROBE_DIR, "emulated_elastic_drape.blend")
    root = dh.configure_state(
        project_name="emulated_elastic_drape",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, pre_data_hash, pre_param_hash = (
        encoder_pkg.prepare_upload(bpy.context)
    )

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    dh.build_and_wait(data_bytes, param_bytes, "elastic-drape:build",
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
    pinned = np.array(pinned_idx, dtype=int)
    free = np.array(free_idx, dtype=int)

    # A: every frame present.
    dh.record(
        "A_pc2_has_all_frames",
        arr.shape[0] == FRAME_COUNT,
        {"pc2_samples": int(arr.shape[0]), "expected": FRAME_COUNT},
    )

    # B: pinned edge holds (max displacement from rest is tiny).
    pin_disp = float(np.max(np.linalg.norm(last[pinned] - rest[pinned], axis=1)))
    dh.record(
        "B_pinned_edge_held",
        pin_disp < 0.05,
        {"max_pinned_disp": round(pin_disp, 5)},
    )

    # C: free vertices draped downward (mean -z displacement is clearly
    # negative). With the kinematic-only emulator this would be ~0.
    free_dz = (last[free] - rest[free])[:, 2]
    mean_drop = float(np.mean(free_dz))
    max_drop = float(np.min(free_dz))  # most-negative = furthest droop
    dh.record(
        "C_free_vertices_draped",
        mean_drop < -0.02 and max_drop < -0.02,
        {"mean_dz": round(mean_drop, 5), "min_dz": round(max_drop, 5)},
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
