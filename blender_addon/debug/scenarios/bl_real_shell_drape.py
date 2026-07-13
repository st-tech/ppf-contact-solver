# File: scenarios/bl_real_shell_drape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Real-backend SHELL gravity drape. A square cloth is pinned along its top
# edge and released under gravity: on the real solver the free region sags
# downward while the pinned edge holds. This is genuine elastic dynamics
# that the emulated CPU stub cannot produce (its advance() is a no-op, so
# the free vertices would stay frozen), which is exactly why it is
# real-only: it exercises SHELL bending/gravity on the real CUDA solver.
#
# bl_ssh_remote_solve runs the same drape over the SSH backend from macOS;
# this one uses the platform-appropriate local connection (dh.connect =>
# LOCAL on Linux, WIN_NATIVE on Windows) so the real-GPU Linux and Windows
# jobs get a direct SHELL-dynamics check without the SSH tunnel.
#
# Subtests:
#   A. build_run_fetch: encode -> build -> solve -> fetch completed (solver
#      not FAILED) and produced a finite PC2 with >= frame_count-1 samples.
#   B. pinned_holds_free_sags: the pinned top edge stayed put and the free
#      vertices moved down under gravity (real drape, not a frozen sheet).

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Real-only: the drape is genuine gravity dynamics; the kinematic emulator
# freezes the free vertices, so this asserts motion only the real solver
# produces. Selected by the AWS Linux / Windows jobs via
# ``runtests --backend real``.
BACKENDS = ("real",)


_FRAME_COUNT = 24


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=10, y_subdivisions=10, size=2.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "DrapeSheet"
    # SHELL meshes are not remeshed, so PC2 vertex order matches the input;
    # we can index the pinned/free sets directly into the fetched array.
    pinned_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    free_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y <= 0.99]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned_idx, 1.0, "REPLACE")
    dh.log(f"grid verts={len(sheet.data.vertices)} pinned={len(pinned_idx)}")

    dh.save_blend(PROBE_DIR, "real_shell_drape.blend")
    root = dh.configure_state(
        project_name="real_shell_drape",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="real_shell_drape:build", timeout=300.0)
    dh.run_and_wait(timeout=300.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total} solver={solver_state}")

    pc2_path = dh.find_pc2_for(sheet)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    samples = int(arr.shape[0]) if arr is not None else 0
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    # ----- A: build + run + fetch completed, PC2 finite ---------------
    dh.record(
        "A_build_run_fetch",
        solver_state != "FAILED"
        and arr is not None
        and samples >= FRAME_COUNT - 1
        and finite,
        {
            "solver_state": solver_state,
            "pc2_path": pc2_path,
            "samples": samples,
            "expected_min_samples": FRAME_COUNT - 1,
            "all_finite": finite,
            "error": dh.facade.engine.state.error,
        },
    )

    # ----- B: pinned edge held, free region sagged under gravity ------
    pin_disp = -1.0
    mean_free_dz = 1.0
    if arr is not None and samples >= 2 and finite:
        rest = arr[0]
        last = arr[-1]
        pinned = np.asarray(pinned_idx, dtype=np.int64)
        free = np.asarray(free_idx, dtype=np.int64)
        pin_disp = float(np.max(np.linalg.norm(last[pinned] - rest[pinned], axis=1)))
        mean_free_dz = float(np.mean((last[free] - rest[free])[:, 2]))
    dh.record(
        "B_pinned_holds_free_sags",
        pin_disp >= 0.0 and pin_disp < 0.1 and mean_free_dz < -0.05,
        {
            "max_pinned_disp": round(pin_disp, 5),
            "mean_free_dz": round(mean_free_dz, 5),
        },
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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
