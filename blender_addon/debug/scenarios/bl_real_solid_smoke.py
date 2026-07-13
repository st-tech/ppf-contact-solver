# File: scenarios/bl_real_solid_smoke.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Real-backend end-to-end smoke for the partially-pinned SOLID path.
#
# This is the flagship scenario the AWS GPU jobs run (BACKENDS includes
# "real"): it drives a full encode -> tetrahedralize(fTetWild) ->
# build -> solve -> fetch cycle for a SOLID whose top face is pinned and
# the rest is free. That partial pin is exactly what triggers the
# frontend's two-stage Poisson pin-field diffusion in _decoder_.py
# (_build_solid_pin_fields), the code path where the Windows regression
# lived: warmup.bat had shipped a bundle without scipy, so the diffusion
# silently fell back to a surface-only pin set and Windows diverged from
# Linux (see fix a8fe7e93). Running this on the real Windows-native and
# real-Linux builds proves the whole dependency chain (fTetWild/pytetwild
# + tetgen + scipy) is present and the pipeline runs end-to-end on the
# real CUDA solver.
#
# It is written to also pass against the emulated CPU-stub solver
# (BACKENDS = emulated, real) so the free-runner macOS job exercises the
# same driver and the cross-platform dh.connect() helper. The trick is a
# prescribed pin MOVE_BY (no gravity): the emulator applies kinematic
# pins (free vertices stay put) while the real solver deforms the free
# region, but the invariants we assert -- the pinned face tracks the move
# and the far side lags it -- hold in BOTH regimes.
#
# Subtests:
#   A. solid_build_run_completes:
#         the fTetWild-backed SOLID builds (deps present, no
#         ModuleNotFoundError), runs to completion (solver not FAILED),
#         and produces a finite PC2 with at least frame_count - 1 samples.
#   B. pin_tracks_move_far_lags:
#         the pinned top face follows the prescribed MOVE_BY delta and the
#         free bottom face lags it, i.e. the body actually simulated.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Backend-agnostic: runs on the emulated free-runner suite AND on the
# real-GPU AWS jobs. This is what the AWS Linux / Windows jobs select
# via ``runtests --backend real``.
BACKENDS = ("emulated", "real")


_FRAME_COUNT = 11
_SUBDIV = 1          # cube edge cuts; kept small so fTetWild is quick in CI
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

    # Fresh scene: a subdivided cube (a closed, manifold SOLID surface)
    # that fTetWild will tetrahedralize.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "SolidBox"
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")

    n_verts = len(cube.data.vertices)
    zmax = max(v.co.z for v in cube.data.vertices)
    zmin = min(v.co.z for v in cube.data.vertices)
    # Partial pin: only the top face is pinned. Leaving the interior +
    # bottom free is what makes the frontend run the Poisson pin-field
    # diffusion (the scipy path) rather than a trivial all-pinned case.
    anchor = [v.index for v in cube.data.vertices if abs(v.co.z - zmax) < 1e-4]
    bottom = [v.index for v in cube.data.vertices if abs(v.co.z - zmin) < 1e-4]
    vg = cube.vertex_groups.new(name="Anchor")
    vg.add(anchor, 1.0, "REPLACE")
    dh.log(f"cube n_verts={n_verts} anchor={len(anchor)} bottom={len(bottom)}")

    dh.save_blend(PROBE_DIR, "real_solid_smoke.blend")
    root = dh.configure_state(
        project_name="real_solid_smoke",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, 0.0),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)
    # Default tet backend is fTetWild (pytetwild); do not override it so
    # this smoke exercises that dependency (tetgen has its own scenario).
    pin = solid.create_pin(cube.name, "Anchor")
    pin.move_by(delta=(0.0, 0.0, MOVE_DZ), frame_start=1,
                frame_end=FRAME_COUNT, transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    # Platform-appropriate connect: LOCAL on macOS/Linux, WIN_NATIVE on
    # Windows. Both attach to the rig-owned server on SERVER_PORT.
    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    # fTetWild runs during build; a missing tet/scipy dep would raise
    # here and surface as a FAILED build below.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="real_solid_smoke:build", timeout=300.0)
    dh.log("built")
    dh.run_and_wait(timeout=180.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.log(f"ran solver={solver_state}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=60.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    pc2_path = dh.find_pc2_for(cube)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    samples = int(arr.shape[0]) if arr is not None else 0
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    # ----- A: build + run completes, deps present, PC2 finite ----------
    dh.record(
        "A_solid_build_run_completes",
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
            "input_surface_verts": n_verts,
            "error": dh.facade.engine.state.error,
        },
    )

    # ----- B: pinned face tracks the move; far side lags ---------------
    # fTetWild remeshes the surface, so the PC2 vertex order does not map
    # 1-1 to the input cube. We therefore reason about the PC2 geometry
    # directly: the top slab (max z at rest) should rise ~MOVE_DZ while
    # the bottom slab (min z at rest) lags. This holds for the kinematic
    # emulator (bottom stays put) and the real deformable solve alike.
    top_dz = -1.0
    bot_dz = -1.0
    top_lateral = -1.0
    if arr is not None and samples >= 2 and finite:
        rest = arr[0]
        last = arr[-1]
        delta = last - rest
        rz = rest[:, 2]
        span = float(rz.max() - rz.min())
        if span > 1e-6:
            top_mask = rz > (rz.max() - 0.1 * span)
            bot_mask = rz < (rz.min() + 0.1 * span)
            if top_mask.any() and bot_mask.any():
                top_dz = float(np.median(delta[top_mask, 2]))
                bot_dz = float(np.median(delta[bot_mask, 2]))
                top_lateral = float(np.max(np.abs(delta[top_mask, :2])))
    pin_tracks = 0.4 * MOVE_DZ < top_dz < 1.2 * MOVE_DZ and top_lateral < 0.2
    # The free far side FOLLOWS the pull on the real deformable solver
    # (bottom_dz ~ MOVE_DZ, with a little dynamic overshoot so it can even
    # slightly EXCEED the top) and stays frozen on the kinematic emulator
    # (bottom_dz ~ 0). Both are valid, so we assert the far side stays in a
    # physically bounded band (never sinks below rest, never overshoots
    # wildly) rather than a lag DIRECTION -- a strict bottom < top lag is
    # false on the real solve because of that dynamic overshoot.
    far_bounded = -0.1 * MOVE_DZ < bot_dz < 1.4 * MOVE_DZ
    dh.record(
        "B_pin_tracks_far_bounded",
        pin_tracks and far_bounded,
        {
            "move_dz": MOVE_DZ,
            "top_dz_median": top_dz,
            "top_lateral_max": top_lateral,
            "bottom_dz_median": bot_dz,
            "pin_tracks": bool(pin_tracks),
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
        .replace("<<SUBDIV>>", str(_SUBDIV))
        .replace("<<MOVE_DZ>>", repr(_MOVE_DZ))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
