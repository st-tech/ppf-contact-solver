# File: scenarios/bl_force_field_fluid_solve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A fluid simulation carries cloth end to end, one way (issue #155):
# a Mantaflow gas sim is run, its velocity written out as a
# VDB sequence the way another program would export one, the sim deleted,
# and the sequence loaded as a Volume object that a Fluid Flow field reads.
# Then encode -> build -> solve -> fetch through a real server and solver,
# judged by the motion in the fetched PC2s against the SIM'S OWN GRID.
#
# That comparison is what makes this more than a smoke test: the grid the
# sheet is judged against never went through the VDB, the Volume object,
# the encoder, the bricks or the kernel, so a wrong unit (the grid's 2.5
# cells per unit), a wrong axis (Blender against solver), a misplaced voxel
# (the VDB transform and the object's) or a fluid read at the wrong time all
# show up as the sheet moving at a velocity the fluid did not have.
#
# Two sheets, gravity and air off:
#
#   * PLUME, in the inflow's jet: at Flow 1 and Strength 20 (a relaxation
#     time of 0.05 s) its velocity must track the sim's own velocity at its
#     vertices, frame by frame, along the jet.
#   * OUTSIDE, above the domain: the field does nothing outside the domain,
#     the damping included (measured), so it must not move at all.
#
# Subtests:
#   A. the_plume_sheet_is_carried_downstream: it travels along Blender +X.
#   B. its_velocity_tracks_the_sims_own_grid: over the middle frames, the
#      sheet's velocity is within 10% of the sim's grid velocity at its
#      vertices. Measured 2 to 5%, the relaxation lag (1 / Strength = 0.05 s
#      behind a jet slowing about 1 m/s^2 along the sheet's path); a frame
#      read one off, which is the slip this check caught in its own row
#      bookkeeping, reads 12 to 15%.
#   C. a_sheet_outside_the_domain_stays_put.
#   D. the_payload_carries_the_sims_velocity: before any solve, the volume
#      the encoder sends reads the sim's own grid at cell centers in the jet,
#      frame by frame, which separates an export or encode fault from a
#      solver one.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_FRAME_COUNT = 24

_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>

try:
    import openvdb

    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT + 1

    # ----- the fluid sim: a jet along Blender +X in a 2 m box ---------------
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    dom = bpy.context.object
    dom.name = "FFSimDomain"
    mod = dom.modifiers.new("fluid", "FLUID")
    mod.fluid_type = "DOMAIN"
    ds = mod.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = 16
    ds.alpha = ds.beta = 0.0
    ds.gravity = (0.0, 0.0, 0.0)
    ds.use_adaptive_timesteps = False
    ds.cache_type = "REPLAY"
    ds.cache_directory = os.path.join(PROBE_DIR, "fluid_solve_cache")
    ds.cache_frame_start, ds.cache_frame_end = 1, FRAME_COUNT + 1
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-0.6, 0.0, 0.0))
    jet = bpy.context.object
    jet.name = "FFSimJet"
    fm = jet.modifiers.new("fluid", "FLUID")
    fm.fluid_type = "FLOW"
    fl = fm.flow_settings
    fl.flow_type = "SMOKE"
    fl.flow_behavior = "INFLOW"
    fl.use_initial_velocity = True
    fl.velocity_coord = (1.5, 0.0, 0.0)

    # Run it, keep each frame's velocity in m/s (the grid's unit is 2.5 cells
    # per second, measured), and write it out as one VDB per frame, voxel
    # centers on the sim's cell centers, the way an exporter would.
    n = 16
    cell = 2.0 / n
    corner = np.array([-1.0, -1.0, -1.0])
    sim = {}
    vdb_dir = os.path.join(PROBE_DIR, "fluid_solve_vdb")
    os.makedirs(vdb_dir, exist_ok=True)
    for frame in range(1, FRAME_COUNT + 2):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        d = dom.evaluated_get(dg).modifiers["fluid"].domain_settings
        res = tuple(d.domain_resolution)
        u = np.array(d.velocity_grid, dtype=np.float64).reshape(res[2], res[1], res[0], 3)
        u = u.transpose(2, 1, 0, 3) * 2.5 * cell
        sim[frame] = u
        g = openvdb.Vec3SGrid()
        g.name = "vel"
        g.transform = openvdb.createLinearTransform(voxelSize=cell)
        # C ORDER, explicitly: `u` is a transposed view, and copyFromArray
        # reads the raw memory without its strides, so a strided array
        # arrives scrambled with no error.
        g.copyFromArray(np.ascontiguousarray(u, dtype=np.float32))
        openvdb.write(os.path.join(vdb_dir, "vel_%04d.vdb" % frame), grids=[g])
    bpy.data.objects.remove(jet)
    bpy.data.objects.remove(dom)

    def sim_velocity(frame, points):
        # The sim's own grid, trilinear between cell centers, in Blender axes.
        u = sim[frame]
        g = (np.asarray(points) - corner) / cell - 0.5
        i0 = np.floor(g).astype(int)
        t = g - i0
        out = np.zeros((len(g), 3))
        for c in range(8):
            o = np.array([(c >> k) & 1 for k in range(3)])
            idx = np.clip(i0 + o, 0, n - 1)
            w = np.prod(np.where(o == 1, t, 1.0 - t), axis=1)
            out += w[:, None] * u[idx[:, 0], idx[:, 1], idx[:, 2]]
        return out

    # ----- the exported sequence as a Volume object ------------------------
    vol = bpy.data.volumes.new("FFWind")
    vol.filepath = os.path.join(vdb_dir, "vel_0001.vdb")
    vol.is_sequence = True
    vol.frame_start = 1
    vol.frame_duration = FRAME_COUNT + 1
    # The files hold m/s. Set the unit after the load, which names the grids.
    vol.grids.load()
    vol.velocity_unit = "SECOND"
    vol_obj = bpy.data.objects.new("FFWind", vol)
    scene.collection.objects.link(vol_obj)
    # Voxel (0, 0, 0)'s center is the sim's first cell center.
    vol_obj.location = tuple(corner + 0.5 * cell)

    def sheet(name, location):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=0.3,
                                        location=location, rotation=(0.0, 1.5708, 0.0))
        o = bpy.context.object
        o.name = name
        return o

    plume = sheet("FFPlume", (-0.3, 0.0, 0.0))
    outside = sheet("FFOutside", (0.0, 0.0, 1.6))
    scene.frame_set(1)
    dh.save_blend(PROBE_DIR, "force_field_fluid_solve.blend")
    root = dh.configure_state(project_name="force_field_fluid_solve",
                              frame_count=FRAME_COUNT, frame_rate=24)
    state = root.state
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plume.name)
    cloth.add(outside.name)

    bpy.ops.object.effector_add(type="FLUID", location=(0.0, 0.0, 0.0))
    flow = bpy.context.object
    flow.name = "FFFlow"
    flow.field.source_object = vol_obj
    flow.field.strength = 20.0
    flow.field.flow = 1.0

    # The payload's volume against the sim's own grid, at cell centers in the
    # jet, frame by frame: what reaches the solver before any solve.
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    fe = __import__(pkg + ".core.field_eval", fromlist=["fluid_at"])
    (sent,) = [f for f in params_mod._build_param_dict(bpy.context)["force_field"]["fields"]
               if f["type"] == "fluid"]
    v = sent["fluid"]
    # Each frame's table and bricks travel as a list of byte chunks.
    volume = {"resolution": [tuple(x) for x in v["resolution"]], "cell": np.asarray(v["cell"]),
              "corner": np.asarray(v["corner"]), "axes": np.asarray(v["axes"]),
              "start": v["start"], "step": v["step"],
              "tables": [np.frombuffer(b"".join(t), dtype="<u4") for t in v["tables"]],
              "bricks": [np.frombuffer(b"".join(b), dtype="<f4").reshape(-1, 512, 4)
                         for b in v["bricks"]]}
    probe_points = np.array([[-0.3125, 0.0625, 0.0625], [0.0625, 0.0625, -0.0625],
                             [0.4375, -0.0625, 0.0625]])
    sent_gap = []
    for frame in (2, 8, 14, 20):
        want = sim[frame][tuple(((probe_points - corner) / cell).astype(int).T)]
        got = fe.fluid_at(volume, probe_points[:, [0, 2, 1]] * [1, 1, -1],
                          (frame - 1) * v["step"])[0]
        got = got[:, [0, 2, 1]] * [1, -1, 1]
        sent_gap.append({"frame": frame, "error": round(float(np.abs(got - want).max()), 5),
                         "scale": round(float(np.abs(want).max()), 4)})
    dh.record("D_the_payload_carries_the_sims_velocity",
              all(g["error"] < 2e-3 * max(g["scale"], 1.0) for g in sent_gap)
              and max(g["scale"] for g in sent_gap) > 0.5,
              {"frames": sent_gap, "frames_sent": len(v["tables"])})

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="force_field_fluid_solve:build",
                      timeout=300.0)
    dh.run_and_wait(timeout=600.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    def rows(obj):
        path = dh.find_pc2_for(obj)
        arr = dh.read_pc2(path) if path else None
        if arr is None or arr.shape[0] < 3 or not np.all(np.isfinite(arr)):
            return None
        # PC2 rows are in the object's local frame; the sheets are only
        # rotated and placed, so the world positions are the matrix times
        # them. PC2 row i is Blender frame start + i (core/client.py): row 0
        # is the rest pose, filled in, and row r >= 1 is solver frame r, at
        # r / fps, which the encoder sampled from Blender frame 1 + r.
        m = np.array(obj.matrix_world, dtype=np.float64)
        return arr @ m[:3, :3].T + m[:3, 3]

    detail = {"solver_state": solver_state, "error": dh.facade.engine.state.error}
    P = rows(plume)
    O = rows(outside)
    detail["rows"] = None if P is None else int(P.shape[0])
    travel = None if P is None else (P[-1] - P[1]).mean(axis=0)
    detail["plume_travel"] = None if travel is None else [round(float(v), 4) for v in travel]
    dh.record("A_the_plume_sheet_is_carried_downstream",
              travel is not None and travel[0] > 0.1
              and abs(travel[1]) < 0.3 * travel[0] and abs(travel[2]) < 0.3 * travel[0],
              detail)

    gaps = []
    if P is not None:
        fps = 24.0
        for r_ in range(6, min(P.shape[0] - 1, 18)):
            v = ((P[r_ + 1] - P[r_ - 1]) * fps / 2.0).mean(axis=0)
            U = sim_velocity(r_ + 1, P[r_]).mean(axis=0)
            gaps.append({"frame": r_ + 1, "sheet": [round(float(x), 3) for x in v],
                         "sim": [round(float(x), 3) for x in U],
                         "relative": round(float(np.linalg.norm(v - U) / np.linalg.norm(U)), 3)})
    detail["velocity"] = gaps
    worst = max((g["relative"] for g in gaps), default=float("inf"))
    dh.record("B_its_velocity_tracks_the_sims_own_grid",
              len(gaps) >= 8 and worst < 0.1, {"worst_relative": worst, **detail})

    still = None if O is None else float(np.abs(O[-1] - O[1]).max())
    dh.record("C_a_sheet_outside_the_domain_stays_put",
              still is not None and still < 1e-6, {"max_travel": still})

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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
