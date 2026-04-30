# File: scenarios/bl_ngon_rejection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# N-gon rejection coverage.
#
# The solver supports only triangles and quads. The legacy encoder
# silently fan-triangulated N-gons in
# ``numpy_mesh_utils.triangulate_numpy_mesh``, which corrupted UV /
# vertex-group / pin correspondence between what the user saw in
# Blender and what the solver received. The addon now rejects N-gons
# at two layers:
#
#   - Assignment (``OBJECT_OT_AddObjectsToGroup``): refuses to take a
#     mesh containing any face with > 4 vertices, reports ERROR, and
#     leaves the group unchanged.
#   - Encoding (``encoder.mesh._build_obj_data``): raises ValueError
#     for any assigned object with N-gons. Catches out-of-band
#     assignments (older saves, MCP scripts that touch the
#     PropertyGroup directly) at Transfer time.
#
# Subtests:
#   A. assignment_rejects_ngon_mesh
#         Build a 5-gon via bmesh, select it, hit
#         ``object.add_objects_to_group``, assert RuntimeError
#         (Blender re-raises ``self.report({"ERROR"}, …)`` from
#         inside ``bpy.ops.*``) and the assigned_objects list is
#         unchanged.
#   B. encoder_rejects_out_of_band_ngon
#         Force-assign the 5-gon mesh via raw PropertyGroup mutation,
#         then call ``encode_obj`` and ``compute_data_hash``. Both
#         must raise ValueError mentioning "N-gon".
#   C. tri_quad_only_passes
#         Sanity: a clean tri+quad-only mesh encodes and hashes
#         without raising.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import bmesh
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def make_pentagon_mesh(name):
    # Five-sided mesh: a single 5-gon face. Returns the new object.
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    coords = [
        (0.0, 1.0, 0.0),
        (0.95, 0.31, 0.0),
        (0.59, -0.81, 0.0),
        (-0.59, -0.81, 0.0),
        (-0.95, 0.31, 0.0),
    ]
    verts = [bm.verts.new(c) for c in coords]
    bm.faces.new(verts)
    bm.to_mesh(mesh)
    bm.free()
    return obj


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash"])
    utils_mod = __import__(pkg + ".core.utils",
                           fromlist=["count_ngon_faces"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="NgonBaseMesh")
    dh.save_blend(PROBE_DIR, "ngon.blend")
    root = dh.configure_state(project_name="ngon_rejection", frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)

    # Build the bad-geometry object: one 5-gon face.
    pentagon = make_pentagon_mesh("PentagonNgon")
    ngon_count = utils_mod.count_ngon_faces(pentagon)
    plane_ngon_count = utils_mod.count_ngon_faces(plane)
    dh.log(f"pentagon ngon_count={ngon_count} plane_ngon_count={plane_ngon_count}")

    # ----- A: assignment refuses ----------------------------------
    bpy.ops.object.select_all(action="DESELECT")
    pentagon.select_set(True)
    pre_count = len(root.object_group_0.assigned_objects)
    op_raised = False
    op_verdict = None
    try:
        op_verdict = bpy.ops.object.add_objects_to_group(group_index=0)
    except RuntimeError:
        op_raised = True
    post_count = len(root.object_group_0.assigned_objects)
    dh.record(
        "A_assignment_rejects_ngon_mesh",
        post_count == pre_count
        and not any(a.name == pentagon.name
                    for a in root.object_group_0.assigned_objects)
        and (op_raised or op_verdict == {"CANCELLED"})
        and ngon_count == 1,
        {
            "pre_count": pre_count, "post_count": post_count,
            "pentagon_ngons": ngon_count,
            "plane_ngons": plane_ngon_count,
            "op_raised": op_raised,
            "op_verdict": list(op_verdict) if op_verdict else None,
        },
    )

    # ----- B: out-of-band assignment fails at encode --------------
    item = root.object_group_0.assigned_objects.add()
    item.name = pentagon.name
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    item.uuid = uuid_mod.get_or_create_object_uuid(pentagon)
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
        "B_encoder_rejects_out_of_band_ngon",
        bool(encode_err) and "N-gon" in encode_err
        and bool(hash_err) and "N-gon" in hash_err,
        {
            "encode_err": encode_err[:160],
            "hash_err": hash_err[:160],
        },
    )

    # ----- C: tri+quad-only mesh passes ---------------------------
    # Pop the bad assignment back off and verify the originally-
    # assigned plane still encodes / hashes cleanly.
    root.object_group_0.assigned_objects.remove(
        len(root.object_group_0.assigned_objects) - 1
    )
    clean_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
        encoder_mesh.compute_data_hash(bpy.context)
    except Exception as e:
        clean_err = f"{type(e).__name__}: {e}"
    dh.record(
        "C_tri_quad_only_passes",
        clean_err == "",
        {"err": clean_err},
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
