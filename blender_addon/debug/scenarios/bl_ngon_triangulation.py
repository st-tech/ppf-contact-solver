# File: scenarios/bl_ngon_triangulation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# N-gon triangulation coverage.
#
# The solver runs on triangles. Rather than reject quads / N-gons, the
# encoder triangulates every polygon through Blender's own
# ``mesh.loop_triangles`` tessellation (``numpy_mesh_utils`` ->
# ``loop_triangulate_mesh``), so the triangles the solver receives are
# exactly the triangles Blender draws in the viewport. The previous
# encoder fan-triangulated from the first vertex, which diverges from the
# viewport on concave / non-planar N-gons (overlapping or back-facing
# triangles) and silently mismatched UV / vertex-group / pin
# correspondence.
#
# Subtests:
#   A. assignment_accepts_ngon_mesh
#         Build a concave 5-gon (a "dart") via bmesh, select it, hit
#         ``object.add_objects_to_group``: the op must succeed and the
#         object must land in assigned_objects (N-gons are no longer
#         rejected).
#   B. encoded_face_matches_loop_triangles
#         Encode the scene, decode the CBOR payload, and assert the
#         pentagon's ``face`` array is the same triangle set Blender's
#         ``loop_triangles`` produces, with one ``uv`` entry per face.
#   C. triangulation_differs_from_naive_fan
#         For the concave dart, the encoded triangle set must differ
#         from a naive fan-from-first-vertex split: proof the encoder
#         honors Blender's tessellation, not the old fan.
#   D. encode_and_hash_succeed
#         ``encode_obj`` and ``compute_data_hash`` both run without
#         raising on the N-gon scene.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import bmesh
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def make_concave_pentagon_mesh(name):
    # A concave 5-gon ("dart"): vertex 3 dips inward, so a naive fan from
    # vertex 0 emits an overlapping / back-facing triangle while Blender's
    # loop_triangles produces a clean tessellation. Carries a UV layer so
    # the per-triangle UV alignment is exercised too. Returns the object.
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    coords = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.0, 2.0, 0.0),
    ]
    verts = [bm.verts.new(c) for c in coords]
    face = bm.faces.new(verts)
    uv = bm.loops.layers.uv.new("UVMap")
    for loop in face.loops:
        loop[uv].uv = (loop.vert.co.x, loop.vert.co.y)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def tri_set(rows):
    # Order-independent set of triangles (each a frozenset of 3 vert ids).
    return {frozenset(int(v) for v in row) for row in rows}


def naive_fan(poly_verts):
    v = list(poly_verts)
    return [[v[0], v[i], v[i + 1]] for i in range(1, len(v) - 1)]


def find_object_info(payload, obj_name):
    for group in payload:
        for info in group.get("object", []):
            if info.get("name") == obj_name:
                return info
    return None


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                             fromlist=["encode_obj", "compute_data_hash"])
    nmu = __import__(pkg + ".core.numpy_mesh_utils",
                     fromlist=["loop_triangle_indices"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="NgonBaseMesh")
    dh.save_blend(PROBE_DIR, "ngon.blend")
    root = dh.configure_state(project_name="ngon_triangulation", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)

    pentagon = make_concave_pentagon_mesh("PentagonNgon")
    # Ground-truth Blender tessellation for the pentagon.
    bl_tri = nmu.loop_triangle_indices(pentagon.data)
    poly_verts = list(pentagon.data.polygons[0].vertices)
    dh.log("pentagon loop_tris=" + str(bl_tri.tolist())
           + " poly=" + str(poly_verts))

    # ----- A: assignment ACCEPTS the N-gon ------------------------
    bpy.ops.object.select_all(action="DESELECT")
    pentagon.select_set(True)
    bpy.context.view_layer.objects.active = pentagon
    pre_count = len(root.object_group_0.assigned_objects)
    op_raised = ""
    op_verdict = None
    try:
        op_verdict = bpy.ops.object.add_objects_to_group(group_index=0)
    except RuntimeError as e:
        op_raised = str(e)
    assigned_names = [a.name for a in root.object_group_0.assigned_objects]
    dh.record(
        "A_assignment_accepts_ngon_mesh",
        op_raised == ""
        and (op_verdict is None or op_verdict == {"FINISHED"})
        and pentagon.name in assigned_names,
        {
            "pre_count": pre_count,
            "post_count": len(assigned_names),
            "assigned": assigned_names,
            "op_raised": op_raised,
            "op_verdict": list(op_verdict) if op_verdict else None,
        },
    )

    # ----- B: encoded face matches Blender loop_triangles ---------
    blob = encoder_mesh.encode_obj(bpy.context)
    payload = DriverHelpers.decode_addon_blob(blob)
    info = find_object_info(payload, pentagon.name)
    enc_face = info.get("face") if info else None
    enc_uv = info.get("uv") if info else None
    face_matches = (
        enc_face is not None
        and tri_set(enc_face) == tri_set(bl_tri.tolist())
        and len(enc_face) == len(bl_tri)
    )
    uv_aligned = enc_uv is not None and len(enc_uv) == len(enc_face)
    dh.record(
        "B_encoded_face_matches_loop_triangles",
        bool(face_matches) and bool(uv_aligned),
        {
            "enc_face": enc_face,
            "bl_tri": bl_tri.tolist(),
            "n_uv": (len(enc_uv) if enc_uv is not None else None),
            "n_face": (len(enc_face) if enc_face is not None else None),
        },
    )

    # ----- C: triangulation differs from naive fan ----------------
    fan = naive_fan(poly_verts)
    differs = (
        enc_face is not None
        and tri_set(enc_face) != tri_set(fan)
    )
    dh.record(
        "C_triangulation_differs_from_naive_fan",
        bool(differs),
        {
            "encoded": [sorted(int(v) for v in t) for t in (enc_face or [])],
            "naive_fan": [sorted(t) for t in fan],
        },
    )

    # ----- D: encode + hash succeed -------------------------------
    enc_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
        encoder_mesh.compute_data_hash(bpy.context)
    except Exception as e:
        enc_err = type(e).__name__ + ": " + str(e)
    dh.record(
        "D_encode_and_hash_succeed",
        enc_err == "",
        {"err": enc_err},
    )

except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
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
