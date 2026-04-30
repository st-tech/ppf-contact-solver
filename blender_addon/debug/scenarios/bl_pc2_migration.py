# File: scenarios/bl_pc2_migration.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# PC2 cache migration on first save.
#
# When the user fetches an animation BEFORE saving the .blend, the
# PC2 file lands in ``$TMPDIR/data/<key>.pc2`` (per ``core.pc2.
# get_pc2_dir``) and the MESH_CACHE modifier's ``filepath`` points
# there. ``migrate_pc2_on_save`` is a save_post handler that moves
# the file to ``<blend_dir>/data/<basename>/<key>.pc2`` and rewrites
# the modifier filepath in lockstep.
#
# This scenario walks the user-visible flow:
#   1. Setup scene WITHOUT saving the .blend.
#   2. Connect → transfer → run → fetch + drain.
#   3. Assert: PC2 exists, modifier exists, both point under
#      ``$TMPDIR/data/`` (the unsaved-state location).
#   4. ``bpy.ops.wm.save_as_mainfile(filepath=blend_path)`` to a path
#      under PROBE_DIR.
#   5. Assert: the temp PC2 is gone, a new PC2 lives at
#      ``<blend_dir>/data/<basename>/<key>.pc2``, the modifier
#      filepath now resolves to that new location, and the file has
#      the same byte-size as before (so the move was lossless).

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import tempfile
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    # Wipe the scene and create the pinned plane, but DO NOT save the
    # .blend yet -- that's the entire point: PC2 should land in the
    # temp tree, then migrate on save.
    plane = dh.reset_scene_to_pinned_plane(name="MigMesh")
    root = dh.configure_state(project_name="pc2_migration", frame_count=6)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")
    assert bpy.data.filepath == "", "expected .blend to be unsaved at setup"

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
        message="pc2_migration:build",
    ))
    deadline = __import__('time').time() + 90.0
    while __import__('time').time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        __import__('time').sleep(0.3)
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    dh.run_and_wait(timeout=90.0)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("fetched")

    # ----- A: PC2 lands under $TMPDIR/data/ pre-save -------------
    tmp_data_dir = os.path.realpath(os.path.join(tempfile.gettempdir(), "data"))
    pc2_pre = dh.find_pc2_for(plane)
    pc2_pre_real = os.path.realpath(pc2_pre) if pc2_pre else ""
    pre_size = (
        os.path.getsize(pc2_pre_real)
        if pc2_pre_real and os.path.isfile(pc2_pre_real) else 0
    )
    dh.record(
        "A_unsaved_blend_writes_pc2_to_tmp",
        pc2_pre is not None
        and pc2_pre_real.startswith(tmp_data_dir + os.sep)
        and pre_size > 0
        and dh.has_mesh_cache(plane)
        and bpy.data.filepath == "",
        {
            "pc2_pre": pc2_pre,
            "pc2_pre_real": pc2_pre_real,
            "tmp_data_dir": tmp_data_dir,
            "pre_size": pre_size,
            "blend_filepath": bpy.data.filepath,
        },
    )

    # ----- B: save_as_mainfile triggers migration ----------------
    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "migrated.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    expected_target_dir = os.path.join(
        os.path.dirname(blend_path), "data",
        os.path.splitext(os.path.basename(blend_path))[0],
    )
    pc2_post = dh.find_pc2_for(plane)
    pc2_post_real = os.path.realpath(pc2_post) if pc2_post else ""
    post_size = (
        os.path.getsize(pc2_post_real)
        if pc2_post_real and os.path.isfile(pc2_post_real) else 0
    )
    pre_still_there = bool(pc2_pre_real) and os.path.exists(pc2_pre_real)
    dh.record(
        "B_save_migrates_pc2_alongside_blend",
        pc2_post is not None
        and pc2_post_real.startswith(os.path.realpath(expected_target_dir) + os.sep)
        and post_size == pre_size
        and not pre_still_there
        and dh.has_mesh_cache(plane),
        {
            "blend_path": blend_path,
            "expected_target_dir": expected_target_dir,
            "pc2_post": pc2_post,
            "pc2_post_real": pc2_post_real,
            "post_size": post_size,
            "pre_still_there": pre_still_there,
        },
    )

    # ----- C: post-migration playback works (modifier rebound) ---
    # MESH_CACHE modifiers cache their internal file handle; if the
    # filepath update didn't take effect, advancing the scene frame
    # would silently fall back to the rest pose. Sanity-check by
    # frame_set + foreach_get on the evaluated mesh.
    if pc2_post and os.path.isfile(pc2_post):
        bpy.context.scene.frame_set(root.state.frame_count - 1)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = plane.evaluated_get(depsgraph)
        verts = eval_obj.data.vertices
        n = len(verts)
        coords = [0.0] * (n * 3)
        verts.foreach_get("co", coords)
        # Vertex 0 starts at x=-0.5; with delta=(0.1, 0, 0) fully
        # applied it should be at x=-0.4. Anything not in
        # [-0.5, -0.4] tells us the modifier rebound to nothing.
        v0_x = coords[0]
        playback_ok = abs(v0_x - (-0.4)) < 1e-2
    else:
        playback_ok = False
        v0_x = None
    dh.record(
        "C_playback_works_after_migration",
        playback_ok,
        {"v0_x": v0_x},
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
