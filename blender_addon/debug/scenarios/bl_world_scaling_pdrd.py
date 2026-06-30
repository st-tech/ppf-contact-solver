# File: scenarios/bl_world_scaling_pdrd.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# world_scaling with a PDRD rigid body. The original world_scaling
# commit (2ada05b5) HARD-ERRORED when world_scaling != 1.0 was combined
# with PDRD bodies ("world-scaling != 1.0 is not yet supported with
# PDRD rigid bodies"), pending per-quantity inertia scaling. That gate
# has since been replaced with real PDRD scaling (scene.rs scales the
# rest centroid, volume, inverse Gram / inertia, mass, and joint pivot
# by the appropriate powers of world_scaling). This rig confirms the
# combination is now ACCEPTED and round-trips the geometry.
#
# The CUDA-free emulator has NO rigid-body physics (PDRD dynamics run on
# a real CUDA host only), so a PDRD body does not move here and an
# output scale-invariance / motion test is not possible. What IS
# observable in emulated mode, and what this rig locks in, is:
#
#   A. build_succeeds   - a PDRD body with world_scaling=0.1 builds
#                         WITHOUT the old hard error (the gate is gone).
#   B. frames_produced  - the solver advances every output frame.
#   C. authored_scale   - the written surface geometry is at the authored
#                         scale (cube edge ~2.0), i.e. the x0.1 scale-in
#                         (scene.rs) and the /0.1 scale-out (backend.rs)
#                         cancel. A broken round-trip would leave the
#                         output at the x0.1 sim scale (edge ~0.2).
#   D. stable           - all positions finite and bounded.
#
# The full PDRD rigid-body inertia / centroid scaling is GPU-only and is
# verified there; this rig is the emulated-host guard that the gate was
# lifted cleanly and the surface geometry survives world_scaling.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

KNOBS = {"PPF_EMULATED_STEP_MS": "0"}

WORLD_SCALING = 0.1


_DRIVER_BODY = r"""
import os
import traceback
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 8
WORLD_SCALING = <<WORLD_SCALING>>
CUBE_SIZE = 2.0   # authored edge length; surface coords span [-1, 1]


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "WsPdrdBody"

    root = dh.configure_state(project_name="ws_pdrd",
                              frame_count=FRAME_COUNT, frame_rate=100,
                              gravity=(0.0, 0.0, -9.8))
    root.state.world_scaling = WORLD_SCALING

    grp = dh.api.solver.create_group("Rigid", "PDRD")
    grp.add(cube.name)

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _d, _p = encoder_pkg.prepare_upload(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "ws-pdrd:build", timeout=180.0)
    solver_name = dh.facade.engine.state.solver.name
    dh.log(f"built solver={solver_name} error={dh.facade.engine.state.error!r}")

    # A: the build did NOT fail. The old gate panicked the solver build
    # for PDRD + world_scaling != 1.0; a clean READY/RESUMABLE proves it
    # is gone.
    dh.record("A_build_succeeds", solver_name in ("READY", "RESUMABLE"),
              {"solver": solver_name, "error": dh.facade.engine.state.error})

    if solver_name in ("READY", "RESUMABLE"):
        dh.run_and_wait(timeout=180.0)
        dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
        dh.fetch_and_drain()
        pc2 = dh.find_pc2_for(cube)
        if not pc2 or not os.path.isfile(pc2):
            raise RuntimeError(f"no PC2 produced (path={pc2!r})")
        arr = dh.read_pc2(pc2)
        dh.log(f"pc2 shape={arr.shape}")

        # B: the run advanced and produced at least the requested frames
        # (the emulated PDRD path can emit a couple of extra settle frames;
        # the exact count is not a world_scaling concern).
        dh.record("B_frames_produced", arr.shape[0] >= FRAME_COUNT,
                  {"samples": int(arr.shape[0]), "expected_min": FRAME_COUNT})

        # C: the surface geometry is written at the AUTHORED scale. The
        # cube spans [-1, 1] per axis (extent 2.0); the scale-in/out must
        # cancel so the output extent is ~2.0, not ~0.2 (sim scale).
        rest = arr[0]
        extent = float(np.max(rest, axis=0).max() - np.min(rest, axis=0).min())
        dh.record("C_authored_scale",
                  abs(extent - CUBE_SIZE) < 0.1 * CUBE_SIZE,
                  {"output_extent": round(extent, 5),
                   "authored_extent": CUBE_SIZE,
                   "sim_scale_extent": round(CUBE_SIZE * WORLD_SCALING, 5)})

        # D: finite and bounded.
        finite = bool(np.all(np.isfinite(arr)))
        bounded = float(np.max(np.abs(arr))) < 1e4
        dh.record("D_stable", finite and bounded,
                  {"finite": finite, "max_abs": round(float(np.max(np.abs(arr))), 4)})

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
        .replace("<<WORLD_SCALING>>", repr(WORLD_SCALING))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
