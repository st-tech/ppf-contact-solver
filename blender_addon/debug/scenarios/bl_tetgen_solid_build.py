# File: scenarios/bl_tetgen_solid_build.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end check for the per-object TetGen tetrahedralizer backend.
#
# A SOLID cube is assigned the TetGen backend (assigned.tet_backend =
# "TETGEN") instead of the default fTetWild. TetGen preserves the input
# surface exactly: every Blender vertex maps 1-1 to a tet-surface vertex.
# The frontend asserts this inside the build (see
# _mesh_.py:_assert_tetgen_surface_unchanged); if TetGen had resampled
# the surface the build would raise and fail here. So a clean build that
# runs to a finite PC2 is the end-to-end proof that the 1-1 map held
# through the real encode -> tetrahedralize(TETGEN) -> build -> solve ->
# fetch cycle.
#
# The top face is pinned with a prescribed MOVE_BY so the solve is
# deterministic (no dependence on gravity / stiffness).
#
# Subtests:
#   A. tetgen_build_run_completes:
#         the TETGEN-backed SOLID builds (1-1 assertion passes), runs to
#         completion (solver not FAILED), and produces a finite PC2 with
#         at least frame_count - 1 samples whose vertex count equals the
#         input cube's surface vertex count (the surface reconstructs 1-1
#         onto the original Blender mesh).
#   B. tetgen_pin_tracks_move:
#         the pinned top face follows the prescribed MOVE_BY delta and
#         the free bottom face lags, so the TetGen mesh actually
#         simulated as a deformable body.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Backend-agnostic: the assertions below are robust invariants (finite PC2,
# pinned/anchored region tracks its prescribed motion, free region lags,
# body not FAILED) that hold on BOTH the emulated CPU stub and the real
# CUDA solver, so this runs on the free-runner macOS suite AND the real-GPU
# AWS jobs selected by ``runtests --backend real``.
BACKENDS = ("emulated", "real")


_FRAME_COUNT = 11
_SUBDIV = 2          # cube edge cuts; surface verts = (cuts+2)^3 - cuts^3
_MOVE_DZ = 0.5       # prescribed pin move along +Z (cube spans [-1, 1])


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
SUBDIV = <<SUBDIV>>
MOVE_DZ = <<MOVE_DZ>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Fresh scene: a subdivided cube (a closed, manifold SOLID surface).
    # TetGen requires exactly this kind of clean input.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "TetGenBox"
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")

    n_verts = len(cube.data.vertices)
    zmax = max(v.co.z for v in cube.data.vertices)
    zmin = min(v.co.z for v in cube.data.vertices)
    anchor = [v.index for v in cube.data.vertices if abs(v.co.z - zmax) < 1e-4]
    bottom = [v.index for v in cube.data.vertices if abs(v.co.z - zmin) < 1e-4]
    vg = cube.vertex_groups.new(name="Anchor")
    vg.add(anchor, 1.0, "REPLACE")
    dh.log(f"cube n_verts={n_verts} anchor={len(anchor)} bottom={len(bottom)}")

    dh.save_blend(PROBE_DIR, "tetgen_solid_build.blend")
    root = dh.configure_state(
        project_name="tetgen_solid_build",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, 0.0),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)
    pin = solid.create_pin(cube.name, "Anchor")
    pin.move_by(delta=(0.0, 0.0, MOVE_DZ), frame_start=1,
                frame_end=FRAME_COUNT, transition="LINEAR")

    # Select the TetGen backend on the assigned object (per-object,
    # like Velocity Overwrite).
    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    assigned = group.assigned_objects[0]
    assigned.tet_backend = "TETGEN"
    dh.log(f"tet_backend={assigned.tet_backend}")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    # TetGen runs during build; the 1-1 surface assertion is enforced there.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="tetgen_solid:build", timeout=240.0)
    dh.log("built")
    dh.run_and_wait(timeout=120.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.log(f"ran solver={solver_state}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    pc2_path = dh.find_pc2_for(cube)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    samples = int(arr.shape[0]) if arr is not None else 0
    pc2_verts = int(arr.shape[1]) if arr is not None and arr.ndim == 3 else -1
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    # ----- A: TetGen build + run completes, surface reconstructs 1-1 ---
    dh.record(
        "A_tetgen_build_run_completes",
        solver_state != "FAILED"
        and arr is not None
        and samples >= FRAME_COUNT - 1
        and finite
        and pc2_verts == n_verts,
        {
            "tet_backend": assigned.tet_backend,
            "solver_state": solver_state,
            "pc2_path": pc2_path,
            "samples": samples,
            "expected_min_samples": FRAME_COUNT - 1,
            "pc2_verts": pc2_verts,
            "input_surface_verts": n_verts,
            "all_finite": finite,
            "error": dh.facade.engine.state.error,
        },
    )

    # ----- B: pinned face tracks the move; far side lags ---------------
    anchor_dz = -1.0
    bottom_dz = -1.0
    anchor_lateral = -1.0
    if arr is not None and samples >= 2 and finite and pc2_verts == n_verts:
        delta = arr[-1] - arr[0]                      # per-vertex displacement
        anchor_arr = np.asarray(anchor, dtype=np.int64)
        bottom_arr = np.asarray(bottom, dtype=np.int64)
        anchor_dz = float(np.median(delta[anchor_arr, 2]))
        anchor_lateral = float(np.max(np.abs(delta[anchor_arr, :2])))
        bottom_dz = float(np.median(delta[bottom_arr, 2]))
    pin_tracks = 0.4 * MOVE_DZ < anchor_dz < 1.2 * MOVE_DZ and anchor_lateral < 0.2
    far_lags = bottom_dz < anchor_dz          # deformable body, not a rigid block
    dh.record(
        "B_tetgen_pin_tracks_move",
        pin_tracks and far_lags,
        {
            "move_dz": MOVE_DZ,
            "anchor_dz_median": anchor_dz,
            "anchor_lateral_max": anchor_lateral,
            "bottom_dz_median": bottom_dz,
            "pin_tracks": bool(pin_tracks),
            "far_lags": bool(far_lags),
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
        .replace("<<SUBDIV>>", str(_SUBDIV))
        .replace("<<MOVE_DZ>>", repr(_MOVE_DZ))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
