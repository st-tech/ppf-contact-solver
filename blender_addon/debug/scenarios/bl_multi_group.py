# File: scenarios/bl_multi_group.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Two simultaneous SHELL groups, two distinct cloth meshes.
#
# Covers the multi-group, multi-mesh fan-out: the encoder must
# carve up the scene into per-group payloads, the runner must
# generate one PC2 file per assigned object (UUID-keyed under
# ``<blend_dir>/data/<basename>/``), and ``apply_animation`` must
# drop one MESH_CACHE modifier per object pointing at its own PC2.
# A regression here would either share one PC2 across both meshes
# (so the trajectories collapse) or skip the second modifier (so
# only one mesh plays back).
#
# Subtests:
#   A. each_plane_has_mesh_cache
#         Both planes carry a ContactSolverCache MESH_CACHE
#         modifier after fetch + drain.
#   B. each_plane_has_distinct_pc2
#         The PC2 paths the two modifiers point at differ, and
#         both files exist on disk.
#   C. pc2_paths_under_data_basename
#         Both PC2 files live under
#         ``<blend_dir>/data/<basename>/`` (the post-save layout
#         from ``core.pc2.get_pc2_dir``).
#   D. trajectories_independent
#         Reading the last frame back from each PC2 confirms plane
#         A's vertex 0 advanced in +X (matching its MOVE_BY) while
#         plane B's vertex 0 advanced in +Y. Tolerance ~1e-2.

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
    dh.log("setup_start")

    # Wipe the scene; build two distinct planes at +X / -X with their
    # own all-vertex pin groups. Different mesh names guarantee distinct
    # UUIDs (and therefore distinct PC2 filenames), and offset locations
    # keep their rest poses separable.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(2.0, 0.0, 0.0))
    plane_a = bpy.context.active_object
    plane_a.name = "MultiMeshA"
    n_a = len(plane_a.data.vertices)
    vg_a = plane_a.vertex_groups.new(name="PinA")
    vg_a.add(list(range(n_a)), 1.0, "REPLACE")

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-2.0, 0.0, 0.0))
    plane_b = bpy.context.active_object
    plane_b.name = "MultiMeshB"
    n_b = len(plane_b.data.vertices)
    vg_b = plane_b.vertex_groups.new(name="PinB")
    vg_b.add(list(range(n_b)), 1.0, "REPLACE")

    # Save the .blend BEFORE running so PC2 lands directly under
    # ``<blend_dir>/data/<basename>/`` rather than the temp tree.
    # The migrate_pc2_on_save path is exercised by bl_pc2_migration;
    # here we want to assert the post-save layout cleanly.
    blend_basename = "multigroup.blend"
    dh.save_blend(PROBE_DIR, blend_basename)

    root = dh.configure_state(project_name="multi_group", frame_count=6)

    # Two SHELL groups, one plane each, with distinct MOVE_BY deltas
    # so their trajectories are linearly independent.
    cloth_a = dh.api.solver.create_group("ClothA", "SHELL")
    cloth_a.add(plane_a.name)
    pin_a = cloth_a.create_pin(plane_a.name, "PinA")
    pin_a.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                  transition="LINEAR")

    cloth_b = dh.api.solver.create_group("ClothB", "SHELL")
    cloth_b.add(plane_b.name)
    pin_b = cloth_b.create_pin(plane_b.name, "PinB")
    pin_b.move_by(delta=(0.0, 0.1, 0.0), frame_start=1, frame_end=4,
                  transition="LINEAR")

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
        message="multi_group:build",
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

    # ----- A: both planes have a MESH_CACHE modifier ---------------
    has_a = dh.has_mesh_cache(plane_a)
    has_b = dh.has_mesh_cache(plane_b)
    dh.record(
        "A_each_plane_has_mesh_cache",
        has_a and has_b,
        {"has_a": has_a, "has_b": has_b},
    )

    # ----- B: each plane has its own PC2 file --------------------
    pc2_a = dh.find_pc2_for(plane_a)
    pc2_b = dh.find_pc2_for(plane_b)
    pc2_a_real = os.path.realpath(pc2_a) if pc2_a else ""
    pc2_b_real = os.path.realpath(pc2_b) if pc2_b else ""
    a_exists = bool(pc2_a_real) and os.path.isfile(pc2_a_real)
    b_exists = bool(pc2_b_real) and os.path.isfile(pc2_b_real)
    distinct = bool(pc2_a_real) and bool(pc2_b_real) and pc2_a_real != pc2_b_real
    dh.record(
        "B_each_plane_has_distinct_pc2",
        distinct and a_exists and b_exists,
        {
            "pc2_a": pc2_a_real,
            "pc2_b": pc2_b_real,
            "a_exists": a_exists,
            "b_exists": b_exists,
        },
    )

    # ----- C: both PC2 paths under <blend_dir>/data/<basename>/ ---
    blend_path = bpy.data.filepath
    expected_dir = os.path.realpath(os.path.join(
        os.path.dirname(blend_path), "data",
        os.path.splitext(os.path.basename(blend_path))[0],
    ))
    sep = os.sep
    a_under = bool(pc2_a_real) and pc2_a_real.startswith(expected_dir + sep)
    b_under = bool(pc2_b_real) and pc2_b_real.startswith(expected_dir + sep)
    dh.record(
        "C_pc2_paths_under_data_basename",
        a_under and b_under,
        {
            "expected_dir": expected_dir,
            "pc2_a": pc2_a_real,
            "pc2_b": pc2_b_real,
            "a_under": a_under,
            "b_under": b_under,
        },
    )

    # ----- D: the two PC2 trajectories are independent -----------
    # PC2 stores LOCAL vertex coordinates, not world. The plane-add
    # location offset (+/- 2 X) is part of the object transform, not
    # the vertex data, so it does not appear in the PC2 stream.
    # Plane A's local v0 rest is (-0.5, -0.5, 0); MOVE_BY (0.1, 0, 0)
    # over frames 1..4 carries it to ~ (-0.4, -0.5, 0).
    # Plane B's local v0 rest is (-0.5, -0.5, 0); MOVE_BY (0, 0.1, 0)
    # carries it to ~ (-0.5, -0.4, 0).
    if a_exists and b_exists:
        traj_a = dh.read_pc2(pc2_a_real)
        traj_b = dh.read_pc2(pc2_b_real)
        last_a = traj_a[-1, 0]
        last_b = traj_b[-1, 0]
        a_x_ok = abs(last_a[0] - (-0.4)) < 1e-2
        a_y_unchanged = abs(last_a[1] - (-0.5)) < 1e-2
        b_y_ok = abs(last_b[1] - (-0.4)) < 1e-2
        b_x_unchanged = abs(last_b[0] - (-0.5)) < 1e-2
        traj_ok = a_x_ok and a_y_unchanged and b_y_ok and b_x_unchanged
        details = {
            "last_a": [float(last_a[0]), float(last_a[1]), float(last_a[2])],
            "last_b": [float(last_b[0]), float(last_b[1]), float(last_b[2])],
            "a_x_ok": a_x_ok,
            "a_y_unchanged": a_y_unchanged,
            "b_y_ok": b_y_ok,
            "b_x_unchanged": b_x_unchanged,
        }
    else:
        traj_ok = False
        details = {"a_exists": a_exists, "b_exists": b_exists}
    dh.record("D_trajectories_independent", traj_ok, details)

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
