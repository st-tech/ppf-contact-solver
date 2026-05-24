# File: scenarios/bl_pin_overlay_follows_edit_mode.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Pin overlay tracks live BMesh vertex positions while the user is
# in Edit Mode.
#
# Background: ``_build_pin_data`` in
# ``ui/dynamics/overlay_geometry/pins.py`` used to read pin positions
# from ``obj.data.vertices[i].co``. Blender's Edit Mode owns vertex
# positions via a BMesh layer that is only flushed back to
# ``obj.data.vertices`` on mode exit, so a user dragging a pinned
# vertex saw the underlying mesh update visually but the pin overlay
# dot stayed at the pre-edit position until they tabbed out.
#
# The fix reads live positions from ``bmesh.from_edit_mesh(obj.data)``
# when the pinned object is in edit mode, and forces the overlay
# cache rebuild every redraw while any pinned mesh is being edited.
#
# This scenario covers the data layer (the bit that's deterministic
# without a viewport draw):
#
#   A. edit_mode_returns_live_positions: enter edit mode, displace
#      a pinned vert via BMesh, call ``_build_pin_data``, assert
#      the overlay point for that vert is at the displaced world
#      position instead of the pre-edit one.
#   B. object_mode_falls_back_to_base_mesh: after exiting edit mode
#      the same call returns the post-flush mesh positions
#      (Blender flushes BMesh to ``obj.data.vertices`` on tab-out).
#
# Pure UI scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback
import bmesh
from mathutils import Vector
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_addon_data"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    pins_mod = __import__(pkg + ".ui.dynamics.overlay_geometry.pins",
                          fromlist=["_build_pin_data"])
    solver_api = api_mod.solver

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Build a SHELL pin scene with the default plane. Pin vert 0
    # (which is at one corner of the plane).
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "EditPlane"
    vg = plane.vertex_groups.new(name="OnePin")
    vg.add([0], 1.0, "REPLACE")

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "OnePin")

    root = groups_mod.get_addon_data(bpy.context.scene)
    group = root.object_group_0
    group.show_pin_overlay = True
    pre_edit_world_co = plane.matrix_world @ plane.data.vertices[0].co
    log("pre_edit_world_co=" + repr(tuple(pre_edit_world_co)))

    # ---- A: edit mode returns live BMesh positions ----------------
    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(plane.data)
    bm.verts.ensure_lookup_table()
    # Displace vert 0 by +2.5 on X. With matrix_world identity this
    # puts the world position at the original + 2.5 on X.
    bm.verts[0].co.x += 2.5
    bmesh.update_edit_mesh(plane.data, loop_triangles=False, destructive=False)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    pin_data_edit = pins_mod._build_pin_data(bpy.context.scene, depsgraph)
    edit_world_cos = [tuple(p[0]) for p in pin_data_edit]
    expected_edit = (
        pre_edit_world_co.x + 2.5,
        pre_edit_world_co.y,
        pre_edit_world_co.z,
    )
    edit_hit = any(
        abs(c[0] - expected_edit[0]) < 1e-4
        and abs(c[1] - expected_edit[1]) < 1e-4
        and abs(c[2] - expected_edit[2]) < 1e-4
        for c in edit_world_cos
    )
    # ALSO assert no overlay point is still at the pre-edit position:
    # if the fix regresses to reading obj.data.vertices, the pin would
    # be at the old corner.
    pre_edit_lingers = any(
        abs(c[0] - pre_edit_world_co.x) < 1e-4
        and abs(c[1] - pre_edit_world_co.y) < 1e-4
        and abs(c[2] - pre_edit_world_co.z) < 1e-4
        for c in edit_world_cos
    )
    record(
        "A_edit_mode_returns_live_positions",
        edit_hit and not pre_edit_lingers,
        {
            "edit_world_cos": edit_world_cos,
            "expected_edit": list(expected_edit),
            "pre_edit_lingers": pre_edit_lingers,
        },
    )

    # ---- B: object-mode fallback after tab-out --------------------
    bpy.ops.object.mode_set(mode="OBJECT")
    # Blender flushed the BMesh on mode-exit, so the post-edit world
    # position is now persistent on obj.data.vertices.
    flushed_world_co = plane.matrix_world @ plane.data.vertices[0].co
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pin_data_obj = pins_mod._build_pin_data(bpy.context.scene, depsgraph)
    obj_world_cos = [tuple(p[0]) for p in pin_data_obj]
    obj_hit = any(
        abs(c[0] - flushed_world_co.x) < 1e-4
        and abs(c[1] - flushed_world_co.y) < 1e-4
        and abs(c[2] - flushed_world_co.z) < 1e-4
        for c in obj_world_cos
    )
    record(
        "B_object_mode_falls_back_to_base_mesh",
        obj_hit,
        {
            "obj_world_cos": obj_world_cos,
            "flushed_world_co": [
                flushed_world_co.x, flushed_world_co.y, flushed_world_co.z,
            ],
        },
    )

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
