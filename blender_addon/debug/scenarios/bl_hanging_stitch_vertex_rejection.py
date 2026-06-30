# File: scenarios/bl_hanging_stitch_vertex_rejection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Hanging-stitch-vertex rejection coverage.
#
# A sewing seam is a "stitch edge": a loose edge (an edge shared by no
# polygon) the solver pulls closed (see ``encoder.mesh.detect_stitch_
# edges``). A normal seam runs between two boundary vertices that are
# themselves corners of cloth triangles, so both endpoints pick up
# surface mass. Subdividing a seam (or otherwise leaving a vertex
# partway along it) inserts a vertex that touches only loose edges and
# belongs to no face. The solver aggregates vertex mass from faces, rod
# edges and tets only, so that midway vertex ends up massless: its
# momentum (inertia) Hessian block is zero, the linear solve goes
# singular, and the simulation aborts before the first frame with NO
# error. Dissolving the midway vertices makes it run again. This was the
# silent "sewing + subdivided seam never simulates" community report.
#
# Rather than fail silently, the encoder now rejects such a mesh at
# Transfer time (``encoder.mesh._build_obj_data``) with an explicit
# ValueError naming the object and the offending vertices.
#
# Subtests:
#   A. encoder_rejects_hanging_seam_vertex
#         Build two triangles joined by a sewing seam, then subdivide
#         the seam so a midway vertex hangs off two loose edges with no
#         face. Assign as SHELL, call ``encode_obj`` / ``compute_data_
#         hash``; both must raise ValueError mentioning "sewing".
#   B. clean_seam_passes
#         The same two triangles with an un-subdivided single-edge seam
#         (both endpoints are face corners) must encode and hash without
#         raising.
#   C. pinned_hanging_seam_accepted
#         The same subdivided seam as A, but with the face-less midway
#         vertex pinned. A pin is a Dirichlet point (the curtain-hook
#         case): the vertex is fixed, carries no free DOF, and never goes
#         singular, so the encoder must NOT reject it. Topology alone still
#         flags the vertex; the pin exemption clears it.

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


def make_seam_mesh(name, *, subdivide_seam):
    # Two separated triangles joined by a single loose seam edge between
    # vertex 2 (corner of tri A) and vertex 3 (corner of tri B). With
    # subdivide_seam the seam is split into 2-mid and mid-3, leaving the
    # inserted midpoint touching only loose edges (no face) -> massless.
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    coords = [
        (0.0, 0.0, 0.0),   # 0
        (1.0, 0.0, 0.0),   # 1
        (1.0, 1.0, 0.0),   # 2  seam end A (face corner)
        (2.0, 0.0, 0.0),   # 3  seam end B (face corner)
        (3.0, 0.0, 0.0),   # 4
        (3.0, 1.0, 0.0),   # 5
    ]
    faces = [(0, 1, 2), (3, 4, 5)]
    if subdivide_seam:
        coords.append((1.5, 0.5, 0.0))  # 6  midway seam vertex (no face)
        edges = [(2, 6), (6, 3)]
    else:
        edges = [(2, 3)]
    mesh.from_pydata(coords, edges, faces)
    mesh.update()
    return obj


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash",
                                        "detect_hanging_stitch_vertices"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="SeamBaseMesh")
    dh.save_blend(PROBE_DIR, "hanging_stitch.blend")
    root = dh.configure_state(project_name="hanging_stitch_vertex_rejection",
                              frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)

    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    # ----- A: subdivided seam (hanging midpoint) rejected at encode ----
    bad = make_seam_mesh("HangingSeam", subdivide_seam=True)
    hanging = encoder_mesh.detect_hanging_stitch_vertices(bad.data)
    item = root.object_group_0.assigned_objects.add()
    item.name = bad.name
    item.uuid = uuid_mod.get_or_create_object_uuid(bad)
    encode_err = ""
    hash_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
    except ValueError as e:
        encode_err = str(e)
    try:
        encoder_mesh.compute_data_hash(bpy.context)
    except ValueError as e:
        hash_err = str(e)
    dh.record(
        "A_encoder_rejects_hanging_seam_vertex",
        hanging == [6]
        and bool(encode_err) and "sewing" in encode_err
        and bool(hash_err) and "sewing" in hash_err,
        {
            "hanging": hanging,
            "encode_err": encode_err[:200],
            "hash_err": hash_err[:200],
        },
    )

    # ----- B: clean single-edge seam passes ---------------------------
    # Pop the bad assignment off, swap in the un-subdivided seam, and
    # confirm it encodes / hashes cleanly with no hanging vertex.
    root.object_group_0.assigned_objects.remove(
        len(root.object_group_0.assigned_objects) - 1
    )
    good = make_seam_mesh("CleanSeam", subdivide_seam=False)
    good_hanging = encoder_mesh.detect_hanging_stitch_vertices(good.data)
    item2 = root.object_group_0.assigned_objects.add()
    item2.name = good.name
    item2.uuid = uuid_mod.get_or_create_object_uuid(good)
    clean_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
        encoder_mesh.compute_data_hash(bpy.context)
    except Exception as e:
        clean_err = f"{type(e).__name__}: {e}"
    dh.record(
        "B_clean_seam_passes",
        good_hanging == [] and clean_err == "",
        {"good_hanging": good_hanging, "err": clean_err[:200]},
    )

    # ----- C: pinned hanging seam vertex is accepted ------------------
    # Same subdivided seam as A (hanging midpoint vertex 6), but pin that
    # vertex. A pinned vertex is fixed by the solver, so it is exempt from
    # the rejection even though topology alone still flags it.
    while len(root.object_group_0.assigned_objects):
        root.object_group_0.assigned_objects.remove(0)
    pinned_mesh = make_seam_mesh("PinnedHangingSeam", subdivide_seam=True)
    hook_vg = pinned_mesh.vertex_groups.new(name="hook")
    hook_vg.add([6], 1.0, "REPLACE")
    cloth.add(pinned_mesh.name)
    cloth.create_pin(pinned_mesh.name, "hook")
    raw_hanging = encoder_mesh.detect_hanging_stitch_vertices(pinned_mesh.data)
    pin_aware = encoder_mesh.detect_hanging_stitch_vertices(
        pinned_mesh.data, {6}
    )
    pinned_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
        encoder_mesh.compute_data_hash(bpy.context)
    except Exception as e:
        pinned_err = f"{type(e).__name__}: {e}"
    dh.record(
        "C_pinned_hanging_seam_accepted",
        raw_hanging == [6] and pin_aware == [] and pinned_err == "",
        {
            "raw_hanging": raw_hanging,
            "pin_aware": pin_aware,
            "err": pinned_err[:200],
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
