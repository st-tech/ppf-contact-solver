# File: scenarios/bl_duplicate_face_rejection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Duplicate-face rejection coverage.
#
# Two triangles that share the same three vertices (coincident faces)
# make the solver's bending-hinge builder produce a degenerate element
# and abort at startup (mesh.rs ``compute_hinge``). This is the
# airbag / inflate "doubled geometry welded with Merge by Distance"
# case reported from the community. Rather than silently drop faces the
# user may have placed intentionally, the addon rejects them at two
# layers:
#
#   - Assignment (``OBJECT_OT_AddObjectsToGroup``): refuses to take a
#     mesh with any duplicate face, reports ERROR, leaves the group
#     unchanged.
#   - Encoding (``encoder.mesh._build_obj_data``): raises ValueError for
#     any assigned object with duplicate faces. Catches out-of-band
#     assignments (older saves, MCP scripts) at Transfer time.
#
# Subtests:
#   A. assignment_rejects_doubled_mesh
#         Build a mesh of two coincident quads (same four vertices),
#         select it, hit ``object.add_objects_to_group``, assert the op
#         refused (RuntimeError / CANCELLED) and the assigned_objects
#         list is unchanged.
#   B. encoder_rejects_out_of_band_doubled
#         Force-assign the doubled mesh via raw PropertyGroup mutation,
#         then call ``encode_obj`` and ``compute_data_hash``. Both must
#         raise ValueError mentioning "duplicate".
#   C. clean_mesh_passes
#         Sanity: a clean tri+quad-only mesh encodes and hashes without
#         raising.

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


def make_doubled_mesh(name):
    # Two coincident quads referencing the same four vertices. Built via
    # from_pydata (which does not validate) so the duplicate faces
    # survive, reproducing the post-Merge-by-Distance leftover topology.
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    faces = [(0, 1, 2, 3), (0, 1, 2, 3)]
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    return obj


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash"])
    utils_mod = __import__(pkg + ".core.utils",
                           fromlist=["count_duplicate_faces"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="DupBaseMesh")
    dh.save_blend(PROBE_DIR, "duplicate_face.blend")
    root = dh.configure_state(project_name="duplicate_face_rejection",
                              frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)

    # Build the bad-geometry object: two coincident quads.
    doubled = make_doubled_mesh("DoubledQuad")
    dup_count = utils_mod.count_duplicate_faces(doubled)
    plane_dup_count = utils_mod.count_duplicate_faces(plane)
    poly_count = len(doubled.data.polygons)
    dh.log(f"doubled dup_count={dup_count} poly_count={poly_count} "
           f"plane_dup_count={plane_dup_count}")

    # ----- A: assignment refuses ----------------------------------
    bpy.ops.object.select_all(action="DESELECT")
    doubled.select_set(True)
    pre_count = len(root.object_group_0.assigned_objects)
    op_raised = False
    op_verdict = None
    try:
        op_verdict = bpy.ops.object.add_objects_to_group(group_index=0)
    except RuntimeError:
        op_raised = True
    post_count = len(root.object_group_0.assigned_objects)
    dh.record(
        "A_assignment_rejects_doubled_mesh",
        post_count == pre_count
        and not any(a.name == doubled.name
                    for a in root.object_group_0.assigned_objects)
        and (op_raised or op_verdict == {"CANCELLED"})
        and dup_count == 2,
        {
            "pre_count": pre_count, "post_count": post_count,
            "doubled_dups": dup_count,
            "doubled_polys": poly_count,
            "plane_dups": plane_dup_count,
            "op_raised": op_raised,
            "op_verdict": list(op_verdict) if op_verdict else None,
        },
    )

    # ----- B: out-of-band assignment fails at encode --------------
    item = root.object_group_0.assigned_objects.add()
    item.name = doubled.name
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])
    item.uuid = uuid_mod.get_or_create_object_uuid(doubled)
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
        "B_encoder_rejects_out_of_band_doubled",
        bool(encode_err) and "duplicate" in encode_err
        and bool(hash_err) and "duplicate" in hash_err,
        {
            "encode_err": encode_err[:160],
            "hash_err": hash_err[:160],
        },
    )

    # ----- C: clean tri+quad-only mesh passes ---------------------
    # Pop the bad assignment back off and verify the originally-assigned
    # plane still encodes / hashes cleanly.
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
        "C_clean_mesh_passes",
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
