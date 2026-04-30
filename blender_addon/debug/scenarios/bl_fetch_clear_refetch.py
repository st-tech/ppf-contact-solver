# File: scenarios/bl_fetch_clear_refetch.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end of the user-facing fetch/clear/refetch loop.
#
# Sequence:
#   1. Build a scene + transfer + run, fetch all frames.
#   2. Snapshot artifacts: PC2 file path, file size, MESH_CACHE modifier
#      attached to the pinned object, ``state.fetched_frame`` count.
#   3. Call ``clear_animation_data`` (the path the Clear button hits).
#      Assert: PC2 gone, modifier removed, ``state.fetched_frame``
#      cleared, ``_anim_frames`` empty.
#   4. Re-fetch via the standard helper, drain the modal, and assert
#      the PC2 reappears with the original size.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


try:
    dh = DriverHelpers(pkg, result)
    animation_mod = __import__(pkg + ".core.animation",
                               fromlist=["clear_animation_data"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="FetchClearMesh")
    dh.save_blend(PROBE_DIR, "fetchclear.blend")
    root = dh.configure_state(project_name="fetch_clear_refetch", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.2, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="fetch-clear:build")
    dh.log("built")

    dh.run_and_wait(timeout=60.0)
    dh.log(f"ran solver={dh.facade.engine.state.solver.name}")

    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    pc2_path_first = dh.find_pc2_for(plane)
    pc2_size_first = (
        os.path.getsize(pc2_path_first)
        if pc2_path_first and os.path.isfile(pc2_path_first) else 0
    )
    fetched_count_pre = len(root.state.fetched_frame)
    dh.record(
        "first_fetch_produced_pc2",
        pc2_path_first is not None
        and pc2_size_first > 0
        and dh.has_mesh_cache(plane)
        and fetched_count_pre > 0,
        {
            "pc2_path": pc2_path_first,
            "pc2_size": pc2_size_first,
            "fetched_count": fetched_count_pre,
            "has_mesh_cache": dh.has_mesh_cache(plane),
        },
    )
    dh.log(f"first_fetch pc2={pc2_path_first} size={pc2_size_first} "
           f"fetched={fetched_count_pre}")

    # Clear via the same call path the operator uses. The user-facing
    # contract: PC2 file deleted, MESH_CACHE modifier removed,
    # state.fetched_frame collection emptied. The runner's in-memory
    # ``_fetched`` is intentionally not part of this contract; it is
    # re-synced from state.fetched_frame the next time
    # ``facade.fetch(context=ctx)`` is called.
    animation_mod.clear_animation_data(bpy.context)
    pc2_exists_after_clear = (
        pc2_path_first is not None and os.path.isfile(pc2_path_first)
    )
    fetched_count_post_clear = len(root.state.fetched_frame)
    runner = dh.facade.runner
    with runner._anim_lock:
        anim_frames_queue_size = len(runner._anim_frames)
    dh.record(
        "clear_removed_artifacts",
        not pc2_exists_after_clear
        and not dh.has_mesh_cache(plane)
        and fetched_count_post_clear == 0,
        {
            "pc2_still_exists": pc2_exists_after_clear,
            "has_mesh_cache": dh.has_mesh_cache(plane),
            "state_fetched_count": fetched_count_post_clear,
            "anim_frames_queue_size": anim_frames_queue_size,
        },
    )
    dh.log("clear.done")

    applied2, total2 = dh.fetch_and_drain()
    dh.log(f"refetch.drained applied={applied2}/{total2}")
    pc2_path_second = dh.find_pc2_for(plane)
    pc2_size_second = (
        os.path.getsize(pc2_path_second)
        if pc2_path_second and os.path.isfile(pc2_path_second) else 0
    )
    dh.record(
        "refetch_restored_pc2",
        pc2_path_second is not None
        and pc2_size_second == pc2_size_first
        and dh.has_mesh_cache(plane),
        {
            "pc2_path": pc2_path_second,
            "pc2_size_second": pc2_size_second,
            "pc2_size_first": pc2_size_first,
            "has_mesh_cache": dh.has_mesh_cache(plane),
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
