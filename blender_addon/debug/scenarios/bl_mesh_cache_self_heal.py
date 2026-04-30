# File: scenarios/bl_mesh_cache_self_heal.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MESH_CACHE modifier self-heal coverage.
#
# ``client.heal_mesh_caches_if_stale`` rebinds the ContactSolverCache
# MESH_CACHE modifier whenever it finds the cache_format wrong or the
# filepath empty. This guards against scenes where a user (or another
# script) flipped the modifier to MDD or cleared the path. The PC2
# file on disk plus the assigned-object UUID are the source of truth;
# the modifier is reconstructed from them.
#
# Subtests:
#   A. cache_format_corruption_heals
#         Run + fetch + drain so the modifier is real. Force
#         ``mod.cache_format = "MDD"``. Call apply_animation (which
#         calls heal first). Assert cache_format == "PC2" and the
#         filepath still resolves to the original PC2.
#   B. empty_filepath_heals
#         Same setup. Force ``mod.filepath = ""``. Call apply_animation.
#         Assert filepath is non-empty and resolves to the same PC2
#         as before corruption.

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


try:
    dh = DriverHelpers(pkg, result)
    client_mod = __import__(pkg + ".core.client",
                            fromlist=["heal_mesh_caches_if_stale",
                                      "apply_animation"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    plane = dh.reset_scene_to_pinned_plane(name="HealMesh")
    dh.save_blend(PROBE_DIR, "heal.blend")
    root = dh.configure_state(project_name="mesh_cache_self_heal",
                              frame_count=6)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=encoder_params.compute_param_hash(bpy.context),
        message="heal:build",
    ))
    deadline = time.time() + 90.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.3)
    assert dh.facade.engine.state.solver.name in ("READY", "RESUMABLE")

    dh.run_and_wait(timeout=90.0)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    assert applied == total and total > 0

    pc2_original = dh.find_pc2_for(plane)
    assert pc2_original and os.path.isfile(pc2_original)

    def get_modifier():
        return plane.modifiers.get("ContactSolverCache")

    # ----- A: cache_format corruption heals ----------------------
    mod = get_modifier()
    assert mod is not None and mod.type == "MESH_CACHE"
    mod.cache_format = "MDD"
    pre_a_format = mod.cache_format
    client_mod.heal_mesh_caches_if_stale()
    mod_a = get_modifier()
    pc2_a = bpy.path.abspath(mod_a.filepath) if mod_a and mod_a.filepath else ""
    dh.record(
        "A_cache_format_corruption_heals",
        mod_a is not None
        and pre_a_format == "MDD"
        and mod_a.cache_format == "PC2"
        and os.path.realpath(pc2_a) == os.path.realpath(pc2_original),
        {
            "pre_format": pre_a_format,
            "post_format": mod_a.cache_format if mod_a else None,
            "pc2_pre": pc2_original,
            "pc2_post": pc2_a,
        },
    )

    # ----- B: empty filepath heals -------------------------------
    mod = get_modifier()
    mod.filepath = ""
    pre_b_filepath = mod.filepath
    client_mod.heal_mesh_caches_if_stale()
    mod_b = get_modifier()
    pc2_b = bpy.path.abspath(mod_b.filepath) if mod_b and mod_b.filepath else ""
    dh.record(
        "B_empty_filepath_heals",
        mod_b is not None
        and pre_b_filepath == ""
        and bool(mod_b.filepath)
        and mod_b.cache_format == "PC2"
        and os.path.realpath(pc2_b) == os.path.realpath(pc2_original),
        {
            "pre_filepath": pre_b_filepath,
            "post_filepath": mod_b.filepath if mod_b else None,
            "pc2_pre": pc2_original,
            "pc2_post": pc2_b,
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
