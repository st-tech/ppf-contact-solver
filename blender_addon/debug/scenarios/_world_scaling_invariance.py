# File: scenarios/_world_scaling_invariance.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Shared driver library for the world_scaling SCALE-INVARIANCE
# scenarios (Design 2).
#
# Free-vertex deformation (drape, solid, rod, sand) and contact
# (colliders) have no closed-form reference -- the emulated solver is
# the only oracle. So instead of diffing against ``frontend``, these
# rigs run the SAME authored scene at two physical sizes and assert the
# outputs differ by exactly that size ratio:
#
#   * Cycle A: build the scene at the base size, world_scaling = 1.0.
#   * Cycle B: build the SAME scene scaled up by ``ratio`` (every
#     authored length -- vertex coords, collider positions / radii /
#     thickness, velocities -- multiplied by ``ratio``), world_scaling
#     = 1 / ratio.
#
# In sim space both collapse to the identical scene: cycle B's geometry
# (ratio x bigger) is divided by ratio on ingest, landing exactly on
# cycle A's geometry, and gravity (absolute, NOT scaled) is the same in
# both. A deterministic emulator therefore runs the identical sim, and
# because the per-frame output is divided back by world_scaling, cycle
# B's written positions are exactly ``ratio`` x cycle A's. We assert
#
#   max |B - ratio * A| / max|B|  <  rel_tol
#
# This is the scale-invariance the feature promises: an over- or under-
# sized scene simulated at a sane scale reproduces (size-scaled) the
# physically authored result. Relative contact gaps participate because
# the encoder scales the bbox-diagonal by world_scaling; absolute gaps
# do NOT scale and so are deliberately avoided here (they are covered at
# the encode level in bl_world_scaling_encoder_scales).
#
# Both build / run cycles happen in ONE Blender session on ONE server
# connection. The rebuild between cycles dispatches a fresh
# BuildPipelineRequested and waits until the server echoes the NEW data
# + param hash (not just solver=READY, which a prior run leaves stale),
# so cycle B never reads cycle A's result.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


# Appended after DRIVER_LIB; runs inside Blender. ``build(scale)`` is
# provided by each rig's body and must wipe the scene, author it at the
# given linear scale, and return the tracked object's name.
INVARIANCE_LIB = r"""
import os
import time
import numpy as np


def ws_reset_scene(dh):
    # Clean slate between cycles: drop solver groups + colliders (they
    # live in addon state and would otherwise accumulate across the two
    # builds), then delete every object.
    try:
        dh.api.solver.delete_all_groups()
    except Exception:
        pass
    try:
        dh.api.solver.clear_invisible_colliders()
    except Exception:
        pass
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def ws_run_cycle(dh, tracked_name, *, project_name, frame_count, gravity,
                 contact, world_scaling, local_path, server_port, first):
    # Configure state, encode, (connect on the first cycle), build with a
    # hash-confirmed wait, run, fetch, and return the tracked object's
    # per-frame PC2 as (n_samples, n_verts, 3).
    root = dh.configure_state(project_name=project_name,
                              frame_count=frame_count, gravity=gravity)
    root.state.world_scaling = float(world_scaling)
    if contact:
        root.state.disable_contact = False
    dh.log(f"cycle world_scaling={world_scaling} contact={contact}")

    data_hash = dh.encoder_mesh.compute_data_hash(bpy.context)
    param_hash = dh.encoder_param.compute_param_hash(bpy.context)
    data_bytes = dh.encoder_mesh.encode_obj(bpy.context)
    param_bytes = dh.encoder_param.encode_param(bpy.context)

    if first:
        dh.connect_local(local_path=local_path, server_port=server_port,
                         project_name=project_name)
        dh.log("connected")

    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=data_hash, param_hash=param_hash,
        message=f"ws-inv-build:{world_scaling}",
    ))
    deadline = time.time() + 240.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        ready = s.solver.name in ("READY", "RESUMABLE", "FAILED")
        hashes_echoed = (s.server_data_hash == data_hash
                         and s.server_param_hash == param_hash)
        if s.activity.name == "IDLE" and ready and hashes_echoed:
            break
        time.sleep(0.2)
    s = dh.facade.engine.state
    if s.solver.name == "FAILED":
        raise RuntimeError(f"build failed: {s.error!r}")
    if s.server_data_hash != data_hash:
        raise RuntimeError(
            f"server never echoed new data hash (ws={world_scaling}); "
            f"have {s.server_data_hash[:12]!r} want {data_hash[:12]!r}"
        )
    dh.log(f"built solver={s.solver.name}")

    dh.run_and_wait(timeout=240.0)
    dh.force_frame_query(expected_frames=frame_count, timeout=60.0)
    dh.fetch_and_drain()
    obj = bpy.data.objects[tracked_name]
    pc2 = dh.find_pc2_for(obj)
    if not pc2 or not os.path.isfile(pc2):
        raise RuntimeError(f"no PC2 for {tracked_name!r} (path={pc2!r})")
    arr = dh.read_pc2(pc2).copy()
    dh.log(f"cycle pc2 shape={arr.shape}")
    return arr


def ws_invariance(dh, build, *, project_name, frame_count, gravity, contact,
                  base_scale, ratio, local_path, server_port, rel_tol, result):
    # Run cycle A at base_scale (ws=1) and cycle B at base_scale*ratio
    # (ws=1/ratio), then record the scale-invariance checks. ``build`` is
    # the rig's scene factory: build(linear_scale) -> tracked object name.
    big_scale = base_scale * ratio
    ws_reset_scene(dh)
    name_a = build(base_scale)
    A = ws_run_cycle(dh, name_a, project_name=project_name,
                     frame_count=frame_count, gravity=gravity, contact=contact,
                     world_scaling=1.0, local_path=local_path,
                     server_port=server_port, first=True)

    ws_reset_scene(dh)
    name_b = build(big_scale)
    B = ws_run_cycle(dh, name_b, project_name=project_name,
                     frame_count=frame_count, gravity=gravity, contact=contact,
                     world_scaling=1.0 / ratio, local_path=local_path,
                     server_port=server_port, first=False)

    # A: shapes line up (same topology, every frame produced).
    shapes_ok = (A.shape == B.shape and A.shape[0] == frame_count)
    dh.record("A_shapes_match", shapes_ok,
              {"shape_a": list(A.shape), "shape_b": list(B.shape),
               "expected_frames": frame_count})
    if not shapes_ok:
        return

    # B: the scene actually moved in cycle A (otherwise invariance is
    # trivially true and proves nothing).
    motion = float(np.max(np.abs(A[-1] - A[0])))
    dh.record("B_scene_evolved", motion > 1e-4 * base_scale,
              {"max_displacement": round(motion, 8),
               "base_scale": base_scale})

    # C: scale invariance -- B == ratio * A to rel_tol.
    expected = ratio * A
    denom = max(float(np.max(np.abs(B))), 1e-9)
    rel = float(np.max(np.abs(B - expected))) / denom
    dh.record("C_scale_invariant", rel < rel_tol,
              {"max_rel_dev": round(rel, 8), "rel_tol": rel_tol,
               "ratio": ratio, "denom": round(denom, 6)})

    # D: both runs finite and bounded.
    finite = bool(np.all(np.isfinite(A)) and np.all(np.isfinite(B)))
    bounded = float(max(np.max(np.abs(A)), np.max(np.abs(B)))) < 1e6
    dh.record("D_stable", finite and bounded,
              {"finite": finite,
               "max_abs": round(float(max(np.max(np.abs(A)),
                                          np.max(np.abs(B)))), 4)})
"""


def build_driver(body: str, ctx: r.ScenarioContext) -> str:
    """Assemble a full invariance driver: DRIVER_LIB + INVARIANCE_LIB +
    the rig-specific ``body`` (which defines ``build(scale)`` and calls
    ``ws_invariance``). The body uses the same <<LOCAL_PATH>> /
    <<SERVER_PORT>> placeholders as the other rigs."""
    return (
        dl.DRIVER_LIB + INVARIANCE_LIB + body
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
