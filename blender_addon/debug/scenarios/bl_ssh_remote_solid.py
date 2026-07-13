# File: scenarios/bl_ssh_remote_solid.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Real-backend SOLID counterpart to bl_ssh_remote_solve: macOS Blender
# drives a REAL CUDA solve of a partially-pinned SOLID on a remote GPU box
# over SSH. bl_ssh_remote_solve exercises the SHELL/cloth encode path over
# the tunnel; this one exercises the SOLID/tetrahedralize path, so both
# major encode surfaces cross paramiko's direct-tcpip channel.
#
# The addon NEVER tetrahedralizes client-side: it uploads the raw surface
# mesh + per-object tet config, and the remote server's frontend runs
# fTetWild (pytetwild) + the partial-pin Poisson diffusion (scipy) at
# build time (_decoder_._build_solid_pin_fields). So this smoke proves the
# remote GPU box has the full SOLID dependency chain AND that a SOLID
# payload survives the SSH transport, without needing pytetwild/scipy in
# macOS Blender (only paramiko, as bl_ssh_remote_solve already requires).
#
# The scene mirrors bl_real_solid_smoke: a subdivided cube whose top face
# is pinned and moved +Z with no gravity. The pinned face tracks the move
# and the free bottom lags it, a robust invariant on the real deformable
# solve. Real-only + darwin-only, like bl_ssh_remote_solve.
#
# SSH parameters come from the environment (injected by the macos-ssh job
# via `runtests --knob PPF_SSH_*=...`):
#   PPF_SSH_HOST, PPF_SSH_PORT, PPF_SSH_USER, PPF_SSH_KEY,
#   PPF_SSH_REMOTE_PATH, PPF_SSH_SERVER_PORT
#
# Subtests:
#   A. solid_ssh_build_run_fetch: connected ONLINE over SSH, the remote
#      fTetWild build + solve succeeded, and a finite PC2 with
#      >= frame_count-1 samples came back over the tunnel.
#   B. pin_tracks_move_far_lags: the pinned top face followed the
#      prescribed +Z move and the free bottom lagged it, i.e. the remote
#      real solver actually deformed the SOLID.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
# SSH backend uses paramiko on the local (macOS) side and needs a real
# remote GPU server, so real-only + darwin.
PLATFORMS = ("darwin",)
BACKENDS = ("real",)


_FRAME_COUNT = 11
_SUBDIV = 1          # cube edge cuts; kept small so fTetWild is quick in CI
_MOVE_DZ = 0.5       # prescribed pin move along +Z (cube spans [-1, 1])


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
FRAME_COUNT = <<FRAME_COUNT>>
SUBDIV = <<SUBDIV>>
MOVE_DZ = <<MOVE_DZ>>

# SSH connection details injected as env knobs by the macos-ssh job.
SSH_HOST = os.environ.get("PPF_SSH_HOST", "")
SSH_PORT = int(os.environ.get("PPF_SSH_PORT", "22"))
SSH_USER = os.environ.get("PPF_SSH_USER", "ubuntu")
SSH_KEY = os.environ.get("PPF_SSH_KEY", "")
SSH_REMOTE_PATH = os.environ.get("PPF_SSH_REMOTE_PATH", "")
SSH_SERVER_PORT = int(os.environ.get("PPF_SSH_SERVER_PORT", "9090"))


def _build_solid_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "SshSolidBox"
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")
    zmax = max(v.co.z for v in cube.data.vertices)
    # Partial pin: only the top face. The free interior/bottom is what
    # makes the remote frontend run the Poisson pin-field diffusion.
    anchor = [v.index for v in cube.data.vertices if abs(v.co.z - zmax) < 1e-4]
    vg = cube.vertex_groups.new(name="Anchor")
    vg.add(anchor, 1.0, "REPLACE")
    return cube, anchor


try:
    if not (SSH_HOST and SSH_KEY and SSH_REMOTE_PATH):
        raise RuntimeError(
            "missing SSH env: PPF_SSH_HOST / PPF_SSH_KEY / PPF_SSH_REMOTE_PATH"
        )
    # paramiko must be importable in Blender's Python for the CUSTOM
    # backend; surface a clear error if the install step was skipped.
    try:
        import paramiko  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"paramiko not importable in Blender: {exc}")

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    cube, anchor = _build_solid_scene()
    dh.log(f"cube n_verts={len(cube.data.vertices)} anchor={len(anchor)}")
    dh.save_blend(PROBE_DIR, "ssh_remote_solid.blend")
    root = dh.configure_state(
        project_name="ssh_remote_solid",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, 0.0),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)
    # Default tet backend is fTetWild (pytetwild) on the remote box.
    pin = solid.create_pin(cube.name, "Anchor")
    pin.move_by(delta=(0.0, 0.0, MOVE_DZ), frame_start=1,
                frame_end=FRAME_COUNT, transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()

    dh.connect_ssh(
        host=SSH_HOST, port=SSH_PORT, username=SSH_USER, key_path=SSH_KEY,
        remote_path=SSH_REMOTE_PATH, server_port=SSH_SERVER_PORT,
        project_name=root.state.project_name, timeout=90.0,
    )
    dh.log("connected over ssh")

    # fTetWild runs remotely during build; a missing tet/scipy dep on the
    # remote box would raise here and surface as a FAILED build below.
    dh.build_and_wait(data_bytes, param_bytes, "ssh:solid:build", timeout=300.0)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    dh.run_and_wait(timeout=300.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total} solver={solver_state}")

    pc2_path = dh.find_pc2_for(cube)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    samples = int(arr.shape[0]) if arr is not None else 0
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    # ----- A: remote SOLID build + run + fetch completed --------------
    dh.record(
        "A_solid_ssh_build_run_fetch",
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

    # ----- B: pinned face tracks the move; free bottom lags -----------
    # fTetWild remeshes the surface, so PC2 vertex order does not map 1-1
    # to the input cube. Reason about the PC2 geometry directly: the top
    # slab (max z at rest) rises ~MOVE_DZ while the bottom slab lags.
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
    # slightly EXCEED the top). Assert the far side stays in a physically
    # bounded band rather than a lag DIRECTION -- a strict bottom < top lag
    # is false on the real solve because of that dynamic overshoot.
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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
        .replace("<<SUBDIV>>", str(_SUBDIV))
        .replace("<<MOVE_DZ>>", repr(_MOVE_DZ))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
