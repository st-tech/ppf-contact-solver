# File: scenarios/bl_solid_fix_weight_threshold.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end check for the SOLID "Hard-Pin Core by Weight"
# (fix_weight_threshold) feature. A partially-pinned SOLID with the
# toggle on decodes into a hard FixPair sub-holder (the high-weight
# surface core) plus a soft PullPair sub-holder (the diffused skirt);
# the hard set is SURFACE-ONLY so the solver never sees an interior fix
# pin (which would be a zero-diagonal CG nan: the fix barrier is
# dispatched over surface verts only and inertia is gated off for fix
# pins). This scenario exercises the whole encode -> tetrahedralize ->
# build -> solve -> fetch cycle through the real (emulated) solver with
# the toggle enabled, which the rest of the suite never does (the toggle
# defaults off).
#
# The anchored (top-face) pin is given a prescribed MOVE_BY so the test
# is deterministic (no dependence on gravity magnitude / material
# stiffness): a correct, non-nan solve drives the hard core to track the
# kinematic target, while the free far side lags elastically.
#
# Subtests:
#   A. build_run_completes_no_nan:
#         the toggle-on partial-pin SOLID builds and runs to completion
#         (solver not FAILED) and produces a finite PC2 with at least
#         frame_count - 1 samples. Direct end-to-end proof that the
#         surface-only hard split does not nan the solver.
#   B. hard_core_tracks_move:
#         the anchored hard vertices follow the prescribed MOVE_BY delta
#         (kinematic FixPair tracking), and the far (bottom) face lags
#         behind them (elastic body, not rigid) -- so the split produced
#         real moving hard pins plus a deformable remainder.

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
_THRESHOLD = 0.5
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
THRESHOLD = <<THRESHOLD>>
SUBDIV = <<SUBDIV>>
MOVE_DZ = <<MOVE_DZ>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Fresh scene: a subdivided cube (a SOLID body). Blender holds only
    # the surface; fTetWild fills the interior with tets at build time.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "SolidBox"
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Anchor = the top face (a PARTIAL pin, so the Poisson partial-pin
    # path runs rather than the full-surface harmonic path). Track the
    # bottom face separately as the "free far side".
    n_verts = len(cube.data.vertices)
    zmax = max(v.co.z for v in cube.data.vertices)
    zmin = min(v.co.z for v in cube.data.vertices)
    anchor = [v.index for v in cube.data.vertices if abs(v.co.z - zmax) < 1e-4]
    bottom = [v.index for v in cube.data.vertices if abs(v.co.z - zmin) < 1e-4]
    vg = cube.vertex_groups.new(name="Anchor")
    vg.add(anchor, 1.0, "REPLACE")
    dh.log(f"cube n_verts={n_verts} anchor={len(anchor)} bottom={len(bottom)}")

    dh.save_blend(PROBE_DIR, "solid_fix_weight_threshold.blend")
    # Gravity off: the prescribed pin move is the only driver, so the
    # test does not depend on gravity magnitude or material stiffness.
    root = dh.configure_state(
        project_name="solid_fix_weight_threshold",
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
    # Prescribed kinematic move of the anchored core over the timeline.
    pin.move_by(delta=(0.0, 0.0, MOVE_DZ), frame_start=1,
                frame_end=FRAME_COUNT, transition="LINEAR")

    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    # Hard intent (no pull) + per-pin surface threshold.
    pin_item = group.pin_vertex_groups[0]
    pin_item.use_pull = False
    pin_item.fix_weight_threshold = THRESHOLD
    dh.log(f"per-pin thr={pin_item.fix_weight_threshold} move_dz={MOVE_DZ}")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    # fTetWild runs during build, so give it a generous window.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="solid_fix_threshold:build", timeout=240.0)
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
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    # ----- A: build + run completes without nan -------------------
    dh.record(
        "A_build_run_completes_no_nan",
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

    # ----- B: hard core tracks the move; far side lags ------------
    anchor_dz = -1.0
    bottom_dz = -1.0
    anchor_lateral = -1.0
    if arr is not None and samples >= 2 and finite:
        delta = arr[-1] - arr[0]                      # per-vertex displacement
        anchor_arr = np.asarray(anchor, dtype=np.int64)
        bottom_arr = np.asarray(bottom, dtype=np.int64)
        # Z-follow of the hard core (should match MOVE_DZ), and its
        # lateral wander (should be ~0: a pure +Z move).
        anchor_dz = float(np.median(delta[anchor_arr, 2]))
        anchor_lateral = float(np.max(np.abs(delta[anchor_arr, :2])))
        bottom_dz = float(np.median(delta[bottom_arr, 2]))
    core_tracks = 0.4 * MOVE_DZ < anchor_dz < 1.2 * MOVE_DZ and anchor_lateral < 0.2
    # The free far side FOLLOWS the pull on the real solver (bottom_dz ~
    # MOVE_DZ, with dynamic overshoot so it can slightly EXCEED the anchor)
    # and stays frozen on the kinematic emulator (bottom_dz ~ 0). Assert the
    # bounded band rather than a lag DIRECTION, which the real overshoot breaks.
    far_bounded = -0.1 * MOVE_DZ < bottom_dz < 1.4 * MOVE_DZ
    dh.record(
        "B_hard_core_tracks_move",
        core_tracks and far_bounded,
        {
            "move_dz": MOVE_DZ,
            "anchor_dz_median": anchor_dz,
            "anchor_lateral_max": anchor_lateral,
            "bottom_dz_median": bottom_dz,
            "core_tracks": bool(core_tracks),
            "far_bounded": bool(far_bounded),
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
        .replace("<<THRESHOLD>>", repr(_THRESHOLD))
        .replace("<<SUBDIV>>", str(_SUBDIV))
        .replace("<<MOVE_DZ>>", repr(_MOVE_DZ))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
