# File: scenarios/bl_pdrd_surface_winding.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What a PDRD body's surface has to be for Transfer to accept it, driven from
# real Blender meshes.
#
# The mass of a PDRD body is its density times the volume its surface
# encloses (`frontend/_pdrd_mass_.py`). That needs a surface that is closed
# and wound consistently. It does not need the faces to point outward, and it
# does not need the body to be any particular size: a chain converted from
# Geometry Nodes arrives with every link inside-out, which the Mesh Cleaning
# panel reports as clean, because no two faces of such a mesh disagree (public
# issue #156).
#
# Five checks, each its own build:
#   `A_inside_out_builds`: a torus with every face reversed reaches READY. Its
#       signed volume is asserted negative and its inconsistently wound edge
#       count zero first, so the fixture is what the check says it is.
#   `B_open_refused_by_name`: a cube missing a face is refused, and the
#       refusal names the Blender object and says the surface is open.
#   `C_mixed_winding_refused_by_name`: a cube with one face reversed is
#       refused, naming the object and the winding.
#   `D_speck_builds`: a cube 0.4 mm on a side (6.4e-11 m^3) reaches READY.
#   `E_inside_out_link_hangs`: an inside-out link threaded through a static
#       one is released under gravity. It comes to rest hanging on the static
#       link, far short of where free fall would have taken it, and stays
#       rigid. Contact reads no face direction, and this is the run that says
#       so for an inside-out body.
#
# B AND C ARE TWO REFUSALS IN A ROW ON PURPOSE. C's refusal has to name
# `Dented` and the winding; while the server kept a refused build's error
# across the next upload, C came back at once carrying B's error about `Bowl`.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it against a CPU build.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import math
import os
import time
import traceback

import bmesh
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT = "pdrd_surface_winding"
BUILD_TIMEOUT = 120.0

# The hanging pair: major and minor radius of both links, and the run.
RING = 0.05
TUBE = 0.01
FPS = 60
FRAMES = 30
STEP = 0.005


def fresh():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    dh.api.solver.clear()


def signed_volume(obj):
    # Divergence theorem over the object's world-space loop triangles.
    mesh = obj.data
    mesh.calc_loop_triangles()
    world = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
    tri = np.array([t.vertices[:] for t in mesh.loop_triangles])
    a, b, c = world[tri[:, 0]], world[tri[:, 1]], world[tri[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def edit(obj, change):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    change(bm)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def add_cube(name, size, location=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def add_torus(name, location, rotation):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=RING, minor_radius=TUBE, major_segments=24,
        minor_segments=8, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def build(message, first):
    if first:
        dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                   project_name=PROJECT)
    return dh.build_verdict(message, timeout=BUILD_TIMEOUT)


def one_body(make):
    # A scene holding one PDRD body, built by `make`.
    fresh()
    obj = make()
    dh.configure_state(project_name=PROJECT, frame_count=FRAMES)
    group = dh.api.solver.create_group("Bodies", "PDRD")
    group.add(obj.name)
    return obj


try:
    dh = DriverHelpers(pkg, result)
    cleaning = __import__(pkg + ".mesh_ops.cleaning_ops",
                          fromlist=["find_surface_defects"])
    dh.log("setup_start")

    # ----- A: every face reversed ------------------------------------
    def inside_out_torus():
        torus = add_torus("InsideOut", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        torus.data.flip_normals()
        torus.data.update()
        return torus

    torus = one_body(inside_out_torus)
    volume = signed_volume(torus)
    bad_winding = cleaning.find_surface_defects(torus.data)["bad_winding"]
    verdict, error = build("inside-out", first=True)
    dh.record("A_inside_out_builds",
              volume < 0.0 and bad_winding == 0 and verdict == "READY",
              {"signed_volume": volume, "bad_winding": bad_winding,
               "solver": verdict, "error": error[:600]})

    # ----- B: a missing face -----------------------------------------
    def bowl():
        cube = add_cube("Bowl", 0.4)
        edit(cube, lambda bm: bmesh.ops.delete(
            bm, geom=[bm.faces[0]], context="FACES_ONLY"))
        return cube

    one_body(bowl)
    verdict, error = build("open", first=False)
    dh.record("B_open_refused_by_name",
              verdict == "FAILED"
              and "PDRD body 'Bowl': its surface is open" in error,
              {"solver": verdict, "error": error[:600]})

    # ----- C: one face against its neighbors -------------------------
    def dented():
        cube = add_cube("Dented", 0.4)
        edit(cube, lambda bm: bm.faces[0].normal_flip())
        return cube

    dent = one_body(dented)
    bad_winding = cleaning.find_surface_defects(dent.data)["bad_winding"]
    verdict, error = build("mixed winding", first=False)
    dh.record("C_mixed_winding_refused_by_name",
              bad_winding > 0 and verdict == "FAILED"
              and "PDRD body 'Dented': its faces are not wound consistently"
              in error,
              {"bad_winding": bad_winding, "solver": verdict,
               "error": error[:600]})

    # ----- D: far below a cubic millimeter ---------------------------
    speck = one_body(lambda: add_cube("Speck", 4e-4))
    verdict, error = build("speck", first=False)
    dh.record("D_speck_builds", verdict == "READY",
              {"volume": signed_volume(speck), "solver": verdict,
               "error": error[:600]})

    # ----- E: an inside-out link hangs on a static one ----------------
    # The anchor's ring lies in the XZ plane and the free link's in the YZ
    # plane, one ring radius lower, so each tube passes through the other
    # ring's hole with RING - 2 * TUBE of clearance. Released, the free link
    # can fall that far and no farther.
    fresh()
    anchor = add_torus("Anchor", (0.0, 0.0, 1.0), (math.pi / 2, 0.0, 0.0))
    link = add_torus("Link", (0.0, 0.0, 1.0 - RING), (0.0, math.pi / 2, 0.0))
    link.data.flip_normals()
    link.data.update()
    root = dh.configure_state(project_name=PROJECT, frame_count=FRAMES,
                              frame_rate=FPS, step_size=STEP,
                              gravity=(0.0, 0.0, -9.8))
    root.state.disable_contact = False
    dh.api.solver.create_group("Anchor", "STATIC").add(anchor.name)
    links = dh.api.solver.create_group("Links", "PDRD")
    links.add(link.name)
    dh.save_blend(PROBE_DIR, "hanging_link.blend")
    link_volume = signed_volume(link)
    verdict, error = build("hanging link", first=False)
    details = {"signed_volume": link_volume, "solver": verdict,
               "error": error[:600]}
    ok = link_volume < 0.0 and verdict == "READY"
    if ok:
        dh.run_and_wait(timeout=600.0)
        dh.force_frame_query(expected_frames=FRAMES - 1, timeout=60.0)
        dh.settle_idle(timeout=30.0)
        dh.fetch_and_drain(fetch_timeout=120.0, drain_timeout=120.0)
        state = dh.facade.engine.state
        pc2 = dh.find_pc2_for(link)
        samples = 0
        drop = stretch = float("nan")
        if pc2 and os.path.isfile(pc2):
            arr = dh.read_pc2(pc2)
            samples = int(arr.shape[0])
            # The cache is in the object's own frame, which is a rotation of
            # the world's, so lengths are the world's.
            drop = float(np.linalg.norm(arr[-1].mean(axis=0) - arr[0].mean(axis=0)))
            first, last = arr[0], arr[-1]
            d0 = np.linalg.norm(first[:, None] - first[None], axis=2)
            d1 = np.linalg.norm(last[:, None] - last[None], axis=2)
            stretch = float(np.abs(d1 - d0).max() / d0.max())
        free_fall = 0.5 * 9.8 * (FRAMES / FPS) ** 2
        details.update({
            "run_solver": state.solver.name, "frame": state.frame,
            "samples": samples, "drop": drop, "stretch": stretch,
            "clearance": RING - 2 * TUBE, "free_fall": free_fall,
            "run_error": (state.server_error or "")[:600],
        })
        ok = (state.solver.name in ("READY", "RESUMABLE")
              and state.frame >= FRAMES - 1
              and samples >= 2
              # It fell onto the anchor, so it moved at least half the
              # clearance, and it was caught, so it is within one ring
              # diameter of where it started.
              and 0.5 * (RING - 2 * TUBE) < drop < 2 * RING
              and stretch < 1e-3)
    dh.record("E_inside_out_link_hangs", ok, details)

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
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
