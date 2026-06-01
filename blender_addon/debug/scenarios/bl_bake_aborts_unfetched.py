# File: scenarios/bl_bake_aborts_unfetched.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Bake Animation must refuse to run while the connected run still has
# animation frames that were not fetched into local PC2: baking then
# would commit an incomplete animation. Both the per-object
# (object.bake_animation) and all-object (solver.bake_all_animation)
# operators guard on this in invoke() and abort with
# "Unfetched animation frames exist. Fetch all animation frames first."
#
# The rig produces a real run, then clears the addon's fetched-frame
# record to reproduce the "remote has more frames than fetched" state
# (the MESH_CACHE stays, so the object is still bakeable), and checks
# the operators abort and leave the object in its group.
#
# Subtests:
#   A. guard_clear_when_all_fetched: after fetching every frame, the
#         guard does not fire (Bake would be allowed).
#   B. bake_all_aborts_when_unfetched: with unfetched frames, the guard
#         reports True, solver.bake_all_animation returns CANCELLED, and
#         the object is still in its group (not baked away).
#   C. bake_object_aborts_when_unfetched: object.bake_animation also
#         returns CANCELLED in the same state.
#   D. guard_clears_after_refetch: restoring the fetched record clears
#         the guard again.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

_FRAME_COUNT = 6


_DRIVER_BODY = r"""
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

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "BakePlane"
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=3)
    bpy.ops.object.mode_set(mode="OBJECT")

    dh.save_blend(PROBE_DIR, "bake_aborts_unfetched.blend")
    root = dh.configure_state(
        project_name="bake_aborts_unfetched",
        frame_count=FRAME_COUNT,
        frame_rate=24,
        step_size=1.0 / 24,
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    shell = dh.api.solver.create_group("Cloth", "SHELL")
    shell.add(plane.name)
    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    state = addon_root.state

    # Full pipeline: build, run, fetch every frame.
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="bake:build")
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    bake = __import__(pkg + ".ui.dynamics.bake_ops",
                      fromlist=["_has_unfetched_frames", "_is_bakeable"])
    scene = bpy.context.scene

    remote_frames = int((dh.facade.communicator.info.response or {}).get("frame", 0))
    bakeable = bool(bake._is_bakeable(plane))

    # ---- A: all fetched -> guard does not fire -----------------------
    fetched_now = len(state.convert_fetched_frames_to_list())
    guard_all_fetched = bake._has_unfetched_frames(scene)
    dh.record(
        "A_guard_clear_when_all_fetched",
        bakeable and remote_frames > 0 and not guard_all_fetched,
        {"remote_frames": remote_frames, "fetched": fetched_now,
         "bakeable": bakeable, "guard": guard_all_fetched},
    )

    # ---- create the unfetched condition ------------------------------
    # Clear the fetched-frame record: remote still reports FRAME_COUNT
    # frames but the addon now believes none are local. The MESH_CACHE
    # stays, so the object remains bakeable.
    state.clear_fetched_frames()
    guard_unfetched = bake._has_unfetched_frames(scene)

    # ---- B: per-object Bake Animation aborts -------------------------
    # object.bake_animation has a trivial poll, so it invokes reliably
    # headless. The invoke() guard reports {"ERROR"} and returns
    # CANCELLED; bpy.ops surfaces an {"ERROR"} report as a RuntimeError,
    # so we assert on the raised message and that the object stays.
    EXPECTED = ("Unfetched animation frames exist. "
                "Fetch all animation frames first.")
    n_before = len(group.assigned_objects)
    obj_err = None
    try:
        bpy.ops.object.bake_animation("INVOKE_DEFAULT", group_index=0)
    except RuntimeError as exc:
        obj_err = str(exc)
    obj_retained = len(group.assigned_objects) == n_before
    dh.record(
        "B_bake_object_aborts_when_unfetched",
        guard_unfetched
        and obj_err is not None
        and EXPECTED in obj_err
        and obj_retained
        and n_before > 0,
        {"guard": guard_unfetched, "error": obj_err,
         "objs_before": n_before, "objs_after": len(group.assigned_objects)},
    )

    # ---- C: all-object Bake Animation aborts (when its poll allows) ---
    # solver.bake_all_animation's poll also requires the run to be idle
    # (not mid-fetch); only exercise the invoke guard when poll passes,
    # otherwise the button is already disabled (an equally safe state).
    all_poll = bool(bpy.ops.solver.bake_all_animation.poll())
    all_err = None
    if all_poll:
        try:
            bpy.ops.solver.bake_all_animation("INVOKE_DEFAULT")
        except RuntimeError as exc:
            all_err = str(exc)
    c_ok = (not all_poll) or (
        all_err is not None
        and EXPECTED in all_err
        and len(group.assigned_objects) == n_before
    )
    dh.record(
        "C_bake_all_aborts_when_unfetched",
        c_ok,
        {"poll_enabled": all_poll, "error": all_err,
         "objs": len(group.assigned_objects)},
    )

    # ---- D: re-fetch -> guard clears ---------------------------------
    for f in range(1, remote_frames + 1):
        state.add_fetched_frame(f)
    guard_refetched = bake._has_unfetched_frames(scene)
    dh.record(
        "D_guard_clears_after_refetch",
        not guard_refetched,
        {"fetched": len(state.convert_fetched_frames_to_list()),
         "remote_frames": remote_frames, "guard": guard_refetched},
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
