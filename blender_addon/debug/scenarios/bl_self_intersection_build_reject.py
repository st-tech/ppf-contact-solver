# File: scenarios/bl_self_intersection_build_reject.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build-time self-intersection rejection.
#
# The frontend's ``_scene_.py`` runs ``check_self_intersection`` over
# every shell-group mesh during ``populate``; if any edge of the mesh
# pierces another triangle of the same mesh at rest, the build raises
# ``Cannot create session: ... self-intersections (...)`` and the
# solver state lands at FAILED. This scenario authors a SHELL mesh
# that's deliberately self-intersecting at rest (a vertical triangle
# cuts through a horizontal triangle), drives a real build, and
# asserts the rejection lands cleanly.
#
# Subtests:
#   A. ``intersection_authored``: the mesh has two triangles whose
#      geometry forces an edge-vs-triangle hit (verified before
#      transfer, no server round-trip needed).
#   B. ``build_fails``: ``BuildPipelineRequested`` lands at
#      ``Solver.FAILED`` with ``Activity.IDLE``. The server's
#      ``status_string`` maps ``Build.FAILED -> "NO_BUILD"`` (no
#      usable artifact, same string as "never built") and the addon's
#      ``_interpret_response`` promotes that to ``Solver.FAILED``
#      when the response carries an ``error`` while ``state.activity``
#      is ``BUILDING`` -- distinguishing it from the inflight
#      race-window ping that also returns ``"NO_BUILD"`` without an
#      error. The terminal-set check then clears activity so the user
#      can fix the scene and re-upload.
#   C. ``failure_surfaces_self_intersection``: the addon's surfaced
#      error (state.error / state.server_error / console messages)
#      mentions "self-intersect" so a user can act on it.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


def _make_self_intersecting_mesh(name):
    import bmesh
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    # Two triangles sharing vertex v2 so the mesh is connected (the
    # encoder discards disconnected components in some shell paths;
    # sharing a vertex keeps both faces in the FixedScene).
    # Triangle A is horizontal in the z=0 plane: (v0, v1, v2).
    # Triangle B is vertical and shares v2: (v3, v4, v2). The edge
    # (v3 -> v4) is x=0.5, y=0.5, z in [-1, 1] -- it crosses z=0
    # at (0.5, 0.5, 0), which is inside triangle A.
    coords = [
        (0.0, 0.0, 0.0),   # v0
        (2.0, 0.0, 0.0),   # v1
        (0.0, 2.0, 0.0),   # v2  shared vertex
        (0.5, 0.5, -1.0),  # v3
        (0.5, 0.5, 1.0),   # v4  edge (v3, v4) pierces triangle A
    ]
    verts = [bm.verts.new(c) for c in coords]
    bm.verts.ensure_lookup_table()
    bm.faces.new((verts[0], verts[1], verts[2]))
    bm.faces.new((verts[3], verts[4], verts[2]))
    bm.to_mesh(mesh)
    bm.free()
    return obj


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Wipe the default scene so the only mesh in this run is our
    # crossing-triangle pair.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    cross = _make_self_intersecting_mesh("CrossingTris")
    n_verts = len(cross.data.vertices)
    # Pin ONE vertex only. Pinning every vertex marks the object as
    # ``static`` in the frontend (FixedScene._tri ends up empty), and
    # ``check_self_intersection`` is gated on ``len(self._tri) > 0``,
    # so we'd silently skip the check we're trying to exercise.
    pin_vg = cross.vertex_groups.new(name="OnePin")
    pin_vg.add([0], 1.0, "REPLACE")

    blend_path = os.path.join(os.path.dirname(PROBE_DIR),
                              "self_intersect.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    root = dh.configure_state(project_name="self_intersection_reject",
                              frame_count=4)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(cross.name)
    cloth.create_pin(cross.name, "OnePin")

    # ----- A: intersection authored at rest -----------------------
    # Triangle A: v0, v1, v2 in z=0. Triangle B: v3, v4, v5 with v3
    # below z=0 and v4 above. Edge (v3, v4) is x=1, y=1, z in
    # [-1, 1] -- so it crosses z=0 at (1, 1, 0). That point lies
    # inside triangle A (barycentric weights all in [0, 1]: along
    # (v0->v1) we're at x=0.5 of the way, along (v0->v2) we're at
    # y=0.5).
    pos = [list(v.co) for v in cross.data.vertices]
    n_faces = len(cross.data.polygons)
    intersection_authored = (
        n_verts == 5
        and n_faces == 2
        and abs(pos[3][2] - (-1.0)) < 1e-6
        and abs(pos[4][2] - 1.0) < 1e-6
        and abs(pos[3][0] - 0.5) < 1e-6
        and abs(pos[3][1] - 0.5) < 1e-6
    )
    dh.record(
        "A_intersection_authored",
        intersection_authored,
        {
            "n_verts": n_verts,
            "n_faces": n_faces,
            "pos_v3": pos[3],
            "pos_v4": pos[4],
        },
    )

    # ----- Build pipeline; expect FAILED ------------------------------
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="self_intersection_reject:build",
    ))
    deadline = time.time() + 60.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)

    final = dh.facade.engine.state
    final_solver = final.solver.name
    final_state_error = final.error or ""
    final_server_error = getattr(final, "server_error", "") or ""

    # ----- B: build fails terminally ---------------------------------
    dh.record(
        "B_build_fails",
        final_solver == "FAILED" and final.activity.name == "IDLE",
        {
            "solver": final_solver,
            "activity": final.activity.name,
            "server_error_present": bool(final_server_error),
        },
    )

    # ----- C: failure surfaces "self-intersect" -----------------------
    console_mod = __import__(pkg + ".models.console", fromlist=["console"])
    console_msgs = [
        getattr(m, "text", str(m))
        for m in getattr(console_mod.console, "messages", [])
    ]
    haystack = "\n".join(
        [final_state_error, final_server_error] + console_msgs
    ).lower()
    mentions = "self-intersect" in haystack or "self_intersect" in haystack
    dh.record(
        "C_failure_surfaces_self_intersection",
        mentions,
        {
            "state_error_tail": final_state_error[-200:],
            "server_error_tail": final_server_error[-200:],
            "console_msg_count": len(console_msgs),
            "haystack_tail": haystack[-300:],
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
