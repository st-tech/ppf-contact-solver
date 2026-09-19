# File: scenarios/bl_mac_native_real_solve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Real-backend SHELL gravity drape over the macOS Native connection. A square
# cloth is pinned along its top edge and released under gravity: the free
# region sags while the pinned edge holds. What separates this from
# bl_real_shell_drape is the connection: it selects MAC_NATIVE and drives the
# solve through the local macOS native build (Metal), so the whole path from
# the panel's backend choice to a moved mesh is exercised on one host.
#
# The rig owns the ppf-cts-server for this worker (PPF_MAC_NATIVE_NO_SPAWN=1
# is set in the harness env), so the addon attaches to the server already
# listening on the worker's port and drives it to RUNNING rather than
# spawning a second one against a bound port.
#
# Subtests:
#   A. mac_native_backend_online: the connection is ONLINE with the server
#      RUNNING, the live backend reports type "mac_native", and the solver
#      root the addon resolved actually holds target/release/ppf-cts-server.
#   B. build_run_fetch: encode -> build -> solve -> fetch completed (solver
#      not FAILED) and produced a finite PC2 with >= frame_count-1 samples.
#   C. pinned_holds_free_sags: the pinned top edge stayed put and the free
#      vertices moved down under gravity (real drape, not a frozen sheet).

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# The MAC_NATIVE backend is the macOS path; on Linux and Windows this connect
# call has no production analogue.
PLATFORMS = ("darwin",)
# Real-only: the drape is genuine gravity dynamics on the Metal backend, and
# the assertion is that the mesh moved.
BACKENDS = ("real",)


_FRAME_COUNT = 24


_DRIVER_BODY = r"""
import os
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
    sheet.name = "MacDrapeSheet"
    # SHELL meshes are not remeshed, so PC2 vertex order matches the input;
    # we can index the pinned/free sets directly into the fetched array.
    pinned_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    free_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y <= 0.99]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned_idx, 1.0, "REPLACE")
    dh.log(f"grid verts={len(sheet.data.vertices)} pinned={len(pinned_idx)}")

    dh.save_blend(PROBE_DIR, "mac_native_real_solve.blend")
    root = dh.configure_state(
        project_name="mac_native_real_solve",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_mac_native(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")

    # ----- A: the connection really is the macOS native backend ------
    conn_state = dh.facade.engine.state
    backend_type = getattr(dh.com.connection, "type", "")
    resolved_root = dh.com.connection.current_directory
    server_bin = os.path.join(
        resolved_root, "target", "release", "ppf-cts-server"
    )
    dh.record(
        "A_mac_native_backend_online",
        conn_state.phase.name == "ONLINE"
        and conn_state.server.name == "RUNNING"
        and backend_type == "mac_native"
        and root.ssh_state.server_type == "MAC_NATIVE"
        and os.path.exists(server_bin),
        {
            "phase": conn_state.phase.name,
            "server": conn_state.server.name,
            "backend_type": backend_type,
            "resolved_root": resolved_root,
            "server_bin_exists": os.path.exists(server_bin),
        },
    )

    dh.build_and_wait(data_bytes, param_bytes,
                      message="mac_native_real_solve:build", timeout=300.0)
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

    # ----- B: build + run + fetch completed, PC2 finite ---------------
    dh.record(
        "B_build_run_fetch",
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

    # ----- C: pinned edge held, free region sagged under gravity ------
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
        "C_pinned_holds_free_sags",
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
