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
#   E. ``headline_is_a_single_line``: the first line of the surfaced
#      error is a complete headline: it opens with ``<Type>: `` and it
#      still carries the segments the worker appends after that, down to
#      the interpreter identity it appends last. The worker's stdout
#      protocol is line-oriented, so a break in the payload costs the
#      server everything after it, and everything after it is where the
#      stage, the frame and the interpreter live.
#   F. ``failure_carries_traceback``: the surfaced error also carries the
#      worker's traceback, so a user can name the failing call without
#      opening ``server.log``.
#   G. ``transfer_reports_failure_not_success``: with the engine at
#      FAILED, ``SOLVER_OT_Transfer.on_complete`` reports an ERROR.
#      ``is_complete()`` is satisfied as soon as the status leaves
#      BUILDING, which includes the failure exit, so an unconditional
#      report makes the last thing Blender says about a failed build be
#      that it succeeded.
#   H. ``every_build_operator_reports_failure``: the same holds for the
#      other two operators that start a remote build,
#      ``SOLVER_OT_UpdateParams`` and ``DEBUG_OT_Build``. All three
#      complete on the same predicate, so all three have to read the
#      outcome before reporting it.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import re
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

    # ----- E: the headline is one complete line ----------------------
    # The build error is delivered as a headline plus the worker's
    # traceback below it, so the panel and the status bar show the first
    # line and it has to stand on its own.
    #
    # The head shape alone cannot see a break in the payload: the server
    # keeps whatever precedes the break and drops the rest, and what it
    # keeps still opens with "<Type>: ". So the two tail segments are
    # asserted as well. The interpreter segment is assembled last and
    # unconditionally, which makes its arrival the proof that the whole
    # physical line arrived; the stage segment is the field that says
    # which object the build died on.
    headline = final_server_error.partition("\n")[0]
    head_ok = bool(re.match(r"^[A-Za-z_][A-Za-z0-9_.]*: ", headline))
    names_stage = "| while: " in headline
    names_interpreter = bool(re.search(r"\| python \S+ \S+ at ", headline))
    dh.record(
        "E_headline_is_a_single_line",
        (bool(headline) and "\r" not in headline
         and head_ok and names_stage and names_interpreter),
        {
            "headline": headline[:600],
            "head_ok": head_ok,
            "names_stage": names_stage,
            "names_interpreter": names_interpreter,
        },
    )

    # ----- F: the traceback travels with the failure ------------------
    has_traceback = "Traceback (most recent call last):" in haystack
    dh.record(
        "F_failure_carries_traceback",
        has_traceback,
        {
            "has_traceback": has_traceback,
            "server_error_lines": len(final_server_error.split("\n")),
        },
    )

    # ----- G: Transfer reports the failure, not success ---------------
    # Recorded last: on_complete writes to the console, and the haystack
    # above must reflect the build alone.
    solver_ui = __import__(pkg + ".ui.solver", fromlist=["SOLVER_OT_Transfer"])

    class _ReportShim:
        def __init__(self):
            self.reports = []

        def report(self, level, message):
            self.reports.append((set(level), message))

    shim = _ReportShim()
    solver_ui.SOLVER_OT_Transfer.on_complete(shim, bpy.context)
    reports_error = any("ERROR" in lvl for lvl, _ in shim.reports)
    claims_success = any(
        "Build completed successfully." in msg for _, msg in shim.reports
    )
    dh.record(
        "G_transfer_reports_failure_not_success",
        reports_error and not claims_success,
        {
            "reports": [[sorted(lvl), msg[:200]] for lvl, msg in shim.reports],
            "remote_status": dh.com.info.status.value,
        },
    )

    # ----- H: every build button reports the failure ------------------
    # Three operators start a remote build and all three complete on the
    # same "status left BUILDING" predicate, so all three have to read
    # the outcome. They share one reporting path; this checks that each
    # of them reaches it.
    debug_ui = __import__(pkg + ".ui.debug_ops", fromlist=["DEBUG_OT_Build"])
    build_ops = [
        ("SOLVER_OT_UpdateParams", solver_ui.SOLVER_OT_UpdateParams),
        ("DEBUG_OT_Build", debug_ui.DEBUG_OT_Build),
    ]
    op_reports = {}
    ops_ok = True
    for op_name, op_cls in build_ops:
        op_shim = _ReportShim()
        op_cls.on_complete(op_shim, bpy.context)
        op_error = any("ERROR" in lvl for lvl, _ in op_shim.reports)
        op_success = any(
            "Build completed successfully." in msg
            for _, msg in op_shim.reports
        )
        ops_ok = ops_ok and op_error and not op_success
        op_reports[op_name] = [
            [sorted(lvl), msg[:200]] for lvl, msg in op_shim.reports
        ]
    dh.record(
        "H_every_build_operator_reports_failure",
        ops_ok,
        {
            "reports": op_reports,
            "remote_status": dh.com.info.status.value,
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
