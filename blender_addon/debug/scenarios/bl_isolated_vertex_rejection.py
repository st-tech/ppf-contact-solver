# File: scenarios/bl_isolated_vertex_rejection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Isolated-vertex rejection + cleanup coverage (STATIC colliders).
#
# A STATIC collider's contact parameters are averaged per vertex over its
# incident faces (encoder.mesh / builder::make_collision_mesh). A vertex in
# no triangle (a stray point left in an imported model) has no incident
# face, so the area-weighted average divides by zero and the solver aborts
# the build with a bare assertion. The encoder now rejects such a mesh at
# Transfer time (encoder.mesh._build_obj_data) with an explicit ValueError
# naming the object and offending vertices, and the Blender panel offers a
# "Remove Isolated Vertices" button (geometry_cleanup_ops) that deletes the
# stray points so the scene can be transferred.
#
# Subtests:
#   A. encoder_rejects_isolated_collider_verts
#         A STATIC collider quad with 3 stray (faceless) vertices; encode
#         must raise ValueError mentioning "isolated vert".
#   B. clean_collider_passes
#         The same quad with no stray verts must encode/hash without raising.
#   C. cleanup_removes_and_passes
#         The cleanup helper deletes the stray verts (faces intact); the
#         cleaned mesh then encodes without raising.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def make_collider_mesh(name, *, with_stray):
    # A single quad (verts 0..3, one face). With with_stray, append 3
    # vertices (4,5,6) that belong to no face -> isolated/faceless.
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    faces = [(0, 1, 2, 3)]
    if with_stray:
        coords += [(5.0, 5.0, 5.0), (6.0, 6.0, 6.0), (7.0, 7.0, 7.0)]
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    return obj


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash",
                                        "detect_isolated_vertices"])
    cleanup = __import__(pkg + ".ui.geometry_cleanup_ops",
                         fromlist=["_static_isolated_offenders",
                                   "_delete_vertices"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    dh.log("setup_start")
    dh.reset_scene_to_pinned_plane(name="IsoBaseMesh")
    dh.save_blend(PROBE_DIR, "isolated_vertex.blend")
    root = dh.configure_state(project_name="isolated_vertex_rejection",
                              frame_count=6)
    collider = dh.api.solver.create_group("Collider", "STATIC")

    def assign(obj):
        item = root.object_group_0.assigned_objects.add()
        item.name = obj.name
        item.uuid = uuid_mod.get_or_create_object_uuid(obj)
        return item

    def clear_assigned():
        while len(root.object_group_0.assigned_objects):
            root.object_group_0.assigned_objects.remove(
                len(root.object_group_0.assigned_objects) - 1
            )

    # ----- A: stray collider verts rejected at encode -------------------
    bad = make_collider_mesh("StrayCollider", with_stray=True)
    iso = encoder_mesh.detect_isolated_vertices(bad.data)
    assign(bad)
    encode_err = ""
    try:
        encoder_mesh.compute_data_hash(bpy.context)
    except ValueError as e:
        encode_err = str(e)
    dh.record(
        "A_encoder_rejects_isolated_collider_verts",
        iso == [4, 5, 6]
        and bool(encode_err) and "isolated vert" in encode_err.lower(),
        {"iso": iso, "err": encode_err[:200]},
    )

    # ----- B: clean collider passes -------------------------------------
    clear_assigned()
    good = make_collider_mesh("CleanCollider", with_stray=False)
    good_iso = encoder_mesh.detect_isolated_vertices(good.data)
    assign(good)
    clean_err = ""
    try:
        encoder_mesh.encode_obj(bpy.context)
        encoder_mesh.compute_data_hash(bpy.context)
    except Exception as e:
        clean_err = f"{type(e).__name__}: {e}"
    dh.record(
        "B_clean_collider_passes",
        good_iso == [] and clean_err == "",
        {"good_iso": good_iso, "err": clean_err[:200]},
    )

    # ----- C: cleanup removes stray verts, then encode passes -----------
    clear_assigned()
    bad2 = make_collider_mesh("StrayCollider2", with_stray=True)
    assign(bad2)
    offenders = cleanup._static_isolated_offenders(bpy.context)
    removed = 0
    for obj, indices in offenders.items():
        removed += cleanup._delete_vertices(obj, indices)
    after_iso = encoder_mesh.detect_isolated_vertices(bad2.data)
    op_err = ""
    try:
        encoder_mesh.compute_data_hash(bpy.context)
    except Exception as e:
        op_err = f"{type(e).__name__}: {e}"
    dh.record(
        "C_cleanup_removes_and_passes",
        removed == 3 and after_iso == []
        and len(bad2.data.vertices) == 4 and op_err == "",
        {"removed": removed, "after_iso": after_iso,
         "verts": len(bad2.data.vertices), "err": op_err[:200]},
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
