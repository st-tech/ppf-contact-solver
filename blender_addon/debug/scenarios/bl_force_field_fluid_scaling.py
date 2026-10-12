# File: scenarios/bl_force_field_fluid_scaling.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The cloth rides a fluid PER BLENDER FRAME, whatever the clocks: under World
# Scaling, Time Scale and an explicit Frame Rate a sheet carried by a Fluid
# Flow covers, in scene units per Blender frame, exactly what the fluid
# itself covers per Blender frame.
#
# The fluid is a VDB sequence read through a Volume object (rotated and
# placed, unscaled), one file per Blender frame 1 to 12: a uniform velocity
# 'vel' over a 40 x 20 x 20 box of 0.1 voxels, in Velocity Unit FRAME (a
# displacement per scene frame) with Velocity Scale 2, so the file holds half
# the displacement, and a 'density' blob (a Gaussian of 0.2) whose center
# advances by exactly that displacement each frame: the smoke the cloth must
# ride. The displacement is 1/24 per frame along the file's x, 1 scene unit
# per second at the scene's 24 fps. Apply Density is off; the blob is only
# the yardstick, its centroid read back from the files through the object's
# matrix.
#
# The scene's own frame rate is 24 throughout, the step 0.01 s, 12 frames,
# gravity and air off. A Fluid Flow of Strength 200 and Flow 1 relaxes the
# sheet's velocity onto the fluid's by a factor of 3 per step. Five runs:
#
#   baseline         Use Scene FPS, World Scaling 1, Time Scale 1;
#   world_half       World Scaling 0.5;
#   time_half        Time Scale 0.5 (a Blender frame lasts 1/12 s);
#   frame_rate_48    Use Scene FPS off, Frame Rate 48;
#   world_2_time_2   World Scaling 2 and Time Scale 2.
#
# What makes all five the same per Blender frame: the add-on turns the
# file's per-frame displacement into a velocity per SCENE second (times the
# scene's 24 fps and Velocity Scale) and then per SOLVER second by
# `fluid_rate`, the solver fps over the scene's (Time Scale with Use Scene
# FPS on, 2 at Frame Rate 48), and the solver multiplies it by World Scaling
# together with the domain's corner and cells.
#
# Each tolerance below sits between what the tree measures and what a defect
# measures. The defects were made by replacing one add-on function
# in-process for one run each: `fluid_rate` returning 1 (RATE_ONE), or Time
# Scale alone with no frame-rate factor (RATE_TS); the payload's velocity
# divided by World Scaling, which is what a solver skipping World Scaling in
# its slot copy applies (WS_DROPPED); Velocity Scale left out (VSCALE); and
# every cell 5% larger (CELL_WIDE).
#
# Subtests:
#   O. this_blender_ships_openvdb: the scenario writes the sequence with
#      Blender's openvdb module; without it every other check is absent, and
#      this one names why.
#   A. the_payload_is_the_displacement_per_frame_times_the_solver_fps: in
#      every run the encoded volume at the blob's center reads the blob's
#      world displacement per frame times the run's solver fps (24, 24, 12,
#      48, 48), in scene units per solver second, within 1e-4 relative. This
#      separates the add-on's half (Velocity Unit and Scale, Time Scale, Frame
#      Rate) from the solver's (World Scaling). Measured: 1.0e-6 (the
#      centroid of the float32 blob). RATE_ONE reads 0.5 to 1.0 off in the
#      three runs whose solver fps is not 24, RATE_TS 0.5 at Frame Rate 48,
#      WS_DROPPED 0.5 to 1.0 where World Scaling is not 1.
#   B to F. <run>_rides_the_blob, one per run: after three frames of
#      relaxation (at least 6 steps, a residual of 3^-6 = 0.14%), the sheet's
#      mean displacement over each Blender frame equals the blob's over the
#      same frame within 2% of its length, and at every frame the sheet's
#      center is within 0.1 of the blob's. Measured: at most 2.5e-4, and the
#      center within 0.01 (the relaxation lag). RATE_ONE puts time_half 100%
#      off and both 48 fps runs 50% off, RATE_TS frame_rate_48 50%,
#      WS_DROPPED world_half 100% and world_2_time_2 50%, VSCALE every run
#      50%; the center then drifts 0.23 to 0.45 from the blob's.
#   G. a_sheet_outside_stays_put_in_every_run: a second sheet 0.05 outside
#      the volume's +y face (half a voxel) moves less than 1e-6 in every run:
#      the presence ends at the face under every World Scaling. Measured:
#      exactly 0. CELL_WIDE moves it 0.39 in every run.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_FRAME_COUNT = 12

_DRIVER_BODY = r"""
import os
import time
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
NF = <<FRAME_COUNT>>

SCENE_FPS = 24
VOX = 0.1
BOX = (40, 20, 20)
STEP = 1.0 / 24.0  # the displacement per Blender frame along the file's x
VELOCITY_SCALE = 2.0
SIGMA = 0.2
BLOB0 = np.array([1.0, 1.0, 1.0])  # the blob's center at frame 1, file coordinates
RELAX = 3
RUNS = (
    ("B", "baseline", dict(world_scaling=1.0, time_scale=1.0, scene_fps=True, frame_rate=24)),
    ("C", "world_half", dict(world_scaling=0.5, time_scale=1.0, scene_fps=True, frame_rate=24)),
    ("D", "time_half", dict(world_scaling=1.0, time_scale=0.5, scene_fps=True, frame_rate=24)),
    ("E", "frame_rate_48", dict(world_scaling=1.0, time_scale=1.0, scene_fps=False,
                                frame_rate=48)),
    ("F", "world_2_time_2", dict(world_scaling=2.0, time_scale=2.0, scene_fps=True,
                                 frame_rate=24)),
)


def solver(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


class NoOpenVDB(Exception):
    pass


try:
    try:
        import openvdb
    except ImportError as exc:
        DriverHelpers(pkg, result).record(
            "O_this_blender_ships_openvdb", False,
            {"error": "this Blender has no openvdb module (%s); the scenario writes "
                      "its VDB sequence with it" % exc})
        raise NoOpenVDB()
    dh = DriverHelpers(pkg, result)
    dh.record("O_this_blender_ships_openvdb", True, {})
    scene = bpy.context.scene
    scene.render.fps = SCENE_FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = NF

    # ----- the sequence, written once ---------------------------------------
    vdb_dir = os.path.join(PROBE_DIR, "fluid_scaling_vdb")
    os.makedirs(vdb_dir, exist_ok=True)
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in BOX], indexing="ij")
    voxel_xyz = np.stack([ii, jj, kk], axis=-1) * VOX
    paths = []
    for f in range(1, NF + 1):
        vel = openvdb.Vec3SGrid()
        vel.name = "vel"
        vel.transform = openvdb.createLinearTransform(voxelSize=VOX)
        stored = np.zeros(BOX + (3,), dtype=np.float32)
        stored[..., 0] = STEP / VELOCITY_SCALE
        vel.copyFromArray(stored)
        rho = openvdb.FloatGrid()
        rho.name = "density"
        rho.transform = openvdb.createLinearTransform(voxelSize=VOX)
        c = BLOB0 + np.array([(f - 1) * STEP, 0.0, 0.0])
        blob = np.exp(-np.sum((voxel_xyz - c) ** 2, axis=-1) / (2.0 * SIGMA ** 2))
        rho.copyFromArray(np.ascontiguousarray(blob, dtype=np.float32))
        path = os.path.join(vdb_dir, "stream_%04d.vdb" % f)
        openvdb.write(path, grids=[vel, rho])
        paths.append(path)

    def file_blob_centers():
        # The blob's centroid at each Blender frame, read back from the files,
        # in file coordinates.
        out = []
        for path in paths:
            g = openvdb.read(path, "density")
            lo, hi = g.evalActiveVoxelBoundingBox()
            shape = tuple(int(b - a + 1) for a, b in zip(lo, hi))
            a = np.zeros(shape, dtype=np.float32)
            g.copyToArray(a, ijk=tuple(lo))
            idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
            w = a.astype(np.float64)
            centroid = (idx * w[..., None]).reshape(-1, 3).sum(axis=0) / w.sum()
            out.append(np.array(g.transform.indexToWorld(tuple(float(v) for v in centroid + lo))))
        return np.array(out)

    blob_file = file_blob_centers()

    LOCATION = (-1.3, -0.4, -0.8)
    ROTATION = (0.15, -0.1, 0.5)

    def build_scene():
        # Wipe the scene and author the Volume object and the two sheets.
        # Returns (volume object, riding sheet, outside sheet, the volume
        # object's matrix).
        try:
            dh.api.solver.delete_all_groups()
        except Exception:
            pass
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        vol = bpy.data.volumes.new("FFStream")
        vol.filepath = paths[0]
        vol.is_sequence = True
        vol.frame_start = 1
        vol.frame_duration = NF
        # Read after the load, which names the velocity grid it finds.
        vol.grids.load()
        vol.velocity_unit = "FRAME"
        vol.velocity_scale = VELOCITY_SCALE
        vol_obj = bpy.data.objects.new("FFStream", vol)
        scene.collection.objects.link(vol_obj)
        vol_obj.location = LOCATION
        vol_obj.rotation_euler = ROTATION
        bpy.context.view_layer.update()
        m = np.array(vol_obj.matrix_world, dtype=np.float64)

        def world(p):
            return np.asarray(p) @ m[:3, :3].T + m[:3, 3]

        def sheet(name, origin, ex, ey):
            ticks = np.linspace(0.0, 0.3, 5)
            verts = [world(origin + a * np.array(ex) + b * np.array(ey))
                     for b in ticks for a in ticks]
            faces = [(j * 5 + i, j * 5 + i + 1, (j + 1) * 5 + i + 1, (j + 1) * 5 + i)
                     for j in range(4) for i in range(4)]
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(float(c) for c in v) for v in verts], [], faces)
            me.update()
            o = bpy.data.objects.new(name, me)
            scene.collection.objects.link(o)
            return o

        # Across the stream, centered on the blob at frame 1.
        riding = sheet("FFRiding", BLOB0 - np.array([0.0, 0.15, 0.15]), (0, 1, 0), (0, 0, 1))
        # Half a voxel past the +y face, whose plane is the outer voxel centers
        # plus half a voxel.
        face_y = (BOX[1] - 0.5) * VOX
        outside = sheet("FFOutside", np.array([2.0, face_y + 0.05, 0.6]), (1, 0, 0), (0, 0, 1))
        return vol_obj, riding, outside, m

    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    enc = __import__(pkg + ".core.encoder", fromlist=["resolve_solver_fps"])
    fe = __import__(pkg + ".core.field_eval", fromlist=["fluid_at"])

    payload = {}
    outside_travel = {}
    first = True
    for letter, name, cfg in RUNS:
        vol_obj, riding, outside, m = build_scene()
        scene.frame_set(1)
        dh.save_blend(PROBE_DIR, "force_field_fluid_scaling.blend")
        root = dh.configure_state(project_name="force_field_fluid_scaling", frame_count=NF,
                                  frame_rate=cfg["frame_rate"], step_size=0.01)
        state = root.state
        state.use_scene_fps = cfg["scene_fps"]
        state.world_scaling = cfg["world_scaling"]
        state.time_scale = cfg["time_scale"]
        cloth = dh.api.solver.create_group("Cloth", "SHELL")
        cloth.add(riding.name)
        cloth.add(outside.name)
        cloth.param.shell_young_modulus = 0.01
        cloth.param.bend = 0.0
        bpy.ops.object.effector_add(type="FLUID", location=(0.0, 0.0, 0.0))
        flow = bpy.context.object
        flow.name = "FFFlow"
        flow.field.source_object = vol_obj
        flow.field.strength = 200.0
        flow.field.flow = 1.0
        solver_fps = float(enc.resolve_solver_fps(state))
        blob_world = blob_file @ m[:3, :3].T + m[:3, 3]
        per_frame = blob_world[1:] - blob_world[:-1]

        # A: the payload's velocity at the blob's center, before the solve.
        (sent,) = [f for f in params_mod._build_param_dict(bpy.context)["force_field"]["fields"]
                   if f["type"] == "fluid"]
        v = sent["fluid"]
        volume = {"resolution": [tuple(x) for x in v["resolution"]],
                  "cell": np.asarray(v["cell"]), "corner": np.asarray(v["corner"]),
                  "axes": np.asarray(v["axes"]), "start": v["start"], "step": v["step"],
                  "tables": [np.frombuffer(b"".join(t), dtype="<u4") for t in v["tables"]],
                  "bricks": [np.frombuffer(b"".join(b), dtype="<f4").reshape(-1, 512, 4)
                             for b in v["bricks"]]}
        got = fe.fluid_frame(volume, 0, solver(blob_world[:1]))[0][0]
        want = solver(per_frame[0]) * solver_fps
        payload[name] = {"solver_fps": solver_fps, "got": got.tolist(), "want": want.tolist(),
                         "relative": float(np.linalg.norm(got - want) / np.linalg.norm(want)),
                         "frames": len(v["tables"])}

        # The solve, with a build wait keyed on this payload's hashes: a
        # rebuild leaves the previous run's terminal state on the engine.
        data_hash = dh.encoder_mesh.compute_data_hash(bpy.context)
        param_hash = dh.encoder_param.compute_param_hash(bpy.context)
        data_bytes, param_bytes = dh.encode_payload()
        if first:
            dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                       project_name=state.project_name)
            first = False
        dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
            data=data_bytes, param=param_bytes, data_hash=data_hash,
            param_hash=param_hash, message="force_field_fluid_scaling:" + name))
        deadline = time.time() + 300.0
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            s = dh.facade.engine.state
            if (s.activity.name == "IDLE" and s.solver.name in ("READY", "RESUMABLE", "FAILED")
                    and s.server_data_hash == data_hash and s.server_param_hash == param_hash):
                break
            time.sleep(0.2)
        s = dh.facade.engine.state
        if s.solver.name == "FAILED":
            raise RuntimeError(name + ": " + build_failure_message(dh.facade, dh.com))
        if s.server_param_hash != param_hash or s.server_data_hash != data_hash:
            raise RuntimeError("%s: the server never held this build's payload (solver %s)"
                               % (name, s.solver.name))
        dh.run_and_wait(timeout=600.0)
        dh.force_frame_query(expected_frames=NF - 1, timeout=120.0)
        dh.settle_idle(timeout=15.0)
        dh.fetch_and_drain()

        def rows(obj):
            path = dh.find_pc2_for(obj)
            arr = dh.read_pc2(path) if path else None
            if arr is None or arr.shape[0] != NF or not np.all(np.isfinite(arr)):
                return None
            # PC2 row i is Blender frame 1 + i, row 0 the rest pose; the meshes
            # are authored in world coordinates under an identity matrix.
            mo = np.array(obj.matrix_world, dtype=np.float64)
            return arr.astype(np.float64) @ mo[:3, :3].T + mo[:3, 3]

        P = rows(riding)
        O = rows(outside)
        detail = {"solver_fps": solver_fps, "solver_state": dh.facade.engine.state.solver.name,
                  "error": dh.facade.engine.state.error, "rows": None if P is None else len(P)}
        ok = False
        if P is not None:
            moved = P.mean(axis=1)[1:] - P.mean(axis=1)[:-1]
            rel = [float(np.linalg.norm(moved[r_] - per_frame[r_]) / np.linalg.norm(per_frame[r_]))
                   for r_ in range(RELAX, NF - 1)]
            lag = float(np.linalg.norm(P.mean(axis=1) - blob_world, axis=1).max())
            detail.update({"per_frame_relative": [round(x, 5) for x in rel],
                           "blob_per_frame": [round(float(x), 5) for x in per_frame[RELAX]],
                           "sheet_per_frame": [round(float(x), 5) for x in moved[RELAX]],
                           "center_distance_max": round(lag, 5)})
            ok = max(rel) < 0.02 and lag < 0.1
        dh.record("%s_%s_rides_the_blob" % (letter, name), ok, detail)
        outside_travel[name] = None if O is None else float(np.abs(O - O[0]).max())

    worst = max(p["relative"] for p in payload.values())
    dh.record("A_the_payload_is_the_displacement_per_frame_times_the_solver_fps",
              worst < 1e-4 and all(p["frames"] == NF for p in payload.values()),
              {"worst_relative": worst, "runs": payload})
    dh.record("G_a_sheet_outside_stays_put_in_every_run",
              all(t is not None and t < 1e-6 for t in outside_travel.values()),
              {"max_travel": outside_travel})

except NoOpenVDB:
    pass
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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
