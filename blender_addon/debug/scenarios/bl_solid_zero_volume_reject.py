# File: scenarios/bl_solid_zero_volume_reject.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build-time rejection of zero-volume meshes assigned to SOLID groups.
#
# A Blender default Plane has 4 coplanar vertices and no enclosed
# volume. Handing it to fTetWild via a SOLID group yields zero usable
# tetrahedra; the post-process ``tet_extract_surface`` returns empty
# arrays, and (pre-fix) ``frame_mapping`` would panic with
# ``PanicException: index out of bounds: the len is 0 but the index is
# 0`` (community issue #18). The frontend now raises a clear
# ``ValueError`` from ``_mesh_.py:tetrahedralize`` and the decoder
# prepends the object name, so the addon's transfer log surfaces a
# single actionable line naming the object and pointing the user at
# SHELL.
#
# Subtests:
#   A. ``solid_plane_authored``: the scene has a SOLID group holding a
#      Plane primitive (4 vertices, all at z=0), verified before
#      transfer so a failure here points at scene authoring rather than
#      the build path.
#   B. ``build_fails``: ``BuildPipelineRequested`` lands at
#      ``Solver.FAILED`` with ``Activity.IDLE``, the same terminal-set
#      promotion path used by ``bl_self_intersection_build_reject``.
#   C. ``failure_names_plane_and_suggests_shell``: the surfaced error
#      mentions the object name ``Plane`` and contains both
#      ``enclosed volume`` and ``SHELL`` so a user can act on it.
#   D. ``failure_is_not_a_panic``: the surfaced error does NOT contain
#      ``PanicException`` or ``index out of bounds``. This is the
#      regression guard for issue #18: if the empty-BVH guards in
#      ``closest_triangle_index`` / ``frame_mapping`` ever regress, the
#      raw Rust panic will resurface here and this subtest will fail.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


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


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    # Wipe the default scene so the only mesh under build is our plane.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Default-size plane: 4 coplanar verts at z=0. Renamed so the
    # decoder's name-prefixed error reads "Plane: ..." (matches what a
    # user following the addon's video tutorial would see).
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "Plane"
    n_verts = len(plane.data.vertices)
    z_coords = [v.co[2] for v in plane.data.vertices]
    all_coplanar = all(abs(z) < 1e-9 for z in z_coords)

    blend_path = os.path.join(os.path.dirname(PROBE_DIR),
                              "solid_zero_volume.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    root = dh.configure_state(project_name="solid_zero_volume_reject",
                              frame_count=4)
    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(plane.name)

    group = root.object_group_0
    plane_assigned = (
        group.object_type == "SOLID"
        and len(group.assigned_objects) == 1
        and group.assigned_objects[0].name == plane.name
    )

    # ----- A: scene authored as SOLID + zero-volume plane -------------
    dh.record(
        "A_solid_plane_authored",
        n_verts == 4 and all_coplanar and plane_assigned,
        {
            "n_verts": n_verts,
            "z_coords": z_coords,
            "object_type": group.object_type,
            "assigned_count": len(group.assigned_objects),
            "assigned_name": (
                group.assigned_objects[0].name
                if group.assigned_objects else None
            ),
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
        message="solid_zero_volume_reject:build",
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

    # ----- C: error names Plane + suggests SHELL ----------------------
    console_mod = __import__(pkg + ".models.console", fromlist=["console"])
    console_msgs = [
        getattr(m, "text", str(m))
        for m in getattr(console_mod.console, "messages", [])
    ]
    haystack = "\n".join(
        [final_state_error, final_server_error] + console_msgs
    )
    haystack_lc = haystack.lower()
    mentions_plane = "plane" in haystack_lc
    mentions_enclosed_volume = "enclosed volume" in haystack_lc
    mentions_shell = "shell" in haystack_lc
    dh.record(
        "C_failure_names_plane_and_suggests_shell",
        mentions_plane and mentions_enclosed_volume and mentions_shell,
        {
            "mentions_plane": mentions_plane,
            "mentions_enclosed_volume": mentions_enclosed_volume,
            "mentions_shell": mentions_shell,
            "state_error_tail": final_state_error[-300:],
            "server_error_tail": final_server_error[-300:],
            "console_msg_count": len(console_msgs),
            "haystack_tail": haystack[-400:],
        },
    )

    # ----- D: not a Rust panic (regression guard for issue #18) -------
    is_panic = (
        "panicexception" in haystack_lc
        or "index out of bounds" in haystack_lc
    )
    dh.record(
        "D_failure_is_not_a_panic",
        not is_panic,
        {
            "is_panic": is_panic,
            "haystack_tail": haystack[-400:],
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
