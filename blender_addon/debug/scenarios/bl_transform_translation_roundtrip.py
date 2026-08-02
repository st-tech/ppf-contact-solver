# File: scenarios/bl_transform_translation_roundtrip.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# An object's world transform must survive the encode -> solver-input ->
# output round-trip with its TRANSLATION intact, even when the transform mixes
# a small scale with a large local-coordinate mesh.
#
# The addon ships each object as LOCAL-space vertices plus a 4x4 "transform"
# (world_matrix), and the frontend decodes world = transform @ local. A decode
# that applies only the 3x3 (rotation*scale) and drops the translation column
# silently misplaces the object. This is latent for objects near the origin
# with unit scale, but a game asset modeled at ~100x and object-parented to a
# rig scaled ~0.01 produces a transform with a SMALL 3x3 scale and a LARGE
# translation, so dropping the translation shifts the mesh by roughly
# inv(scale) * translation, i.e. hundreds of units. The batman-cape scene hit
# exactly this: the cape decoded ~3.6 m out of place while Blender showed it
# conforming, so its stitches were stretched and the solve diverged.
#
# This scenario reproduces that transform structure with a single SHELL plane
# (local verts ~+/-100, object scale 0.01, non-origin location) and checks that
# the round-tripped geometry matches the input local verts. The emulated
# advance is a no-op, so the solver output must equal the decoded input; a
# correct decode round-trips to numeric noise, the translation-drop bug is off
# by hundreds of units.
#
# Blender + emulated solver end to end (build / run / fetch), so it exercises
# the real frontend decode that writes the session vertex buffers.
#
# Subtests:
#   A. transform_translation_survives_roundtrip: the fetched (round-tripped)
#      local verts equal the original local verts within a tight tolerance.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>

FRAME_COUNT = 3

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Reproduce the exact transform structure of a game asset modeled at large
    # local scale and object-parented to a rig scaled ~0.01 that is later
    # rotated. Parenting happens BEFORE the rig is rotated, so the parent
    # inverse captures only the 0.01 scale (matrix_parent_inverse ~ 100x, no
    # rotation); the rig rotation then survives into the plane's matrix_world.
    # The resulting world_matrix (the "transform" shipped to the solver) has a
    # small 3x3 scale, a real rotation, AND a real translation, over large
    # local coords -- exactly the batman-cape configuration that decoded out of
    # place while Blender showed it in the right spot.
    rig = bpy.data.objects.new("Rig", None)
    bpy.context.collection.objects.link(rig)
    rig.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()

    bpy.ops.mesh.primitive_plane_add(size=534.0)  # local verts at +/-267
    plane = bpy.context.active_object
    plane.name = "XformShell"
    plane.parent = rig
    plane.matrix_parent_inverse = rig.matrix_world.inverted()  # ~100x, no rotation
    plane.scale = (0.01, 0.01, 0.01)
    plane.location = (3.0, 2.0, 1.0)
    rig.rotation_euler = (0.0, 0.0, -1.5707963)  # -90 deg Z, not captured by mpi
    bpy.context.view_layer.update()

    # Log the world bbox so a failure can be attributed to the transform
    # round-trip rather than to the scene sitting at a coordinate magnitude
    # where the positions themselves lose resolution.
    import numpy as _np
    _mw = _np.array(plane.matrix_world)
    _co = _np.empty(len(plane.data.vertices) * 3)
    plane.data.vertices.foreach_get("co", _co)
    _w = (_co.reshape(-1, 3) @ _mw[:3, :3].T) + _mw[:3, 3]
    dh.log("plane world bbox=%s..%s transform_t=%s"
           % (_w.min(0).round(2).tolist(), _w.max(0).round(2).tolist(),
              _mw[:3, 3].round(3).tolist()))

    n = len(plane.data.vertices)
    orig_local = np.empty(n * 3, dtype=np.float64)
    plane.data.vertices.foreach_get("co", orig_local)
    orig_local = orig_local.reshape(n, 3)

    root = dh.configure_state(
        project_name="xform_translation_roundtrip",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, 0.0),
    )
    shell = dh.api.solver.create_group("Shell", "SHELL")
    shell.add(plane.name)

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="xform:build",
                      timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=root.state.frame_count - 1,
                         timeout=15.0)
    dh.settle_idle(timeout=15.0)

    dh.fetch_and_drain()
    pc2 = dh.find_pc2_for(plane)
    arr = dh.read_pc2(pc2) if pc2 and os.path.isfile(pc2) else None

    ok = False
    max_err = None
    n_samples = 0
    if arr is not None and arr.shape[0] >= 2 and arr.shape[1] == n:
        n_samples = int(arr.shape[0])
        # Sample 0 is the addon rest geometry; sample 1 is the round-tripped
        # solver output (emulated advance is a no-op, so it must equal the
        # decoded input). Compare the solver output back to the original local
        # verts: a translation-drop leaves it off by ~inv(0.01)*translation
        # (hundreds of units), a correct decode round-trips to numeric noise.
        solver_local = np.asarray(arr[1], dtype=np.float64)
        max_err = float(np.abs(solver_local - orig_local).max())
        ok = max_err < 1.0

    dh.record(
        "A_transform_translation_survives_roundtrip",
        ok,
        {
            "max_local_error": max_err,
            "n_samples": n_samples,
            "n_verts": n,
            "orig_local_bbox": [orig_local.min(0).tolist(),
                                orig_local.max(0).tolist()],
        },
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
