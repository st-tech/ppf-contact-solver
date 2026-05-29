# File: scenarios/bl_pin_create_clears_stale_membership.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression: Create Pin against a same-named pre-existing vertex group
# must produce exactly the current edit-mode selection, not a union
# with the prior membership. The Remove operator leaves the underlying
# vertex group on the mesh (by design); re-Creating with that name
# used to inherit the old verts because ``vg.add(..., "REPLACE")`` only
# overwrites the listed indices' weights, leaving stale members intact.
#
# Subtests:
#   A. first_create_produces_exact_selection: Create with name "X" and
#      verts {0,1,2} selected ends with vg "X" membership == {0,1,2}.
#   B. remove_keeps_underlying_vg: pressing Remove on the pin entry
#      drops the pin list slot but leaves vg "X" on the mesh with its
#      original members.
#   C. second_create_clears_stale: with vg "X" still present and a
#      disjoint selection {4,5}, a second Create with name "X"
#      produces membership == {4,5}. Before the fix this returned
#      {0,1,2,4,5}.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


def vg_members(obj, vg_name):
    vg = obj.vertex_groups.get(vg_name)
    if vg is None:
        return None
    out = set()
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                out.add(v.index)
                break
    return out


def select_verts(obj, indices):
    import bmesh
    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != "EDIT_MESH":
        bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    target = set(indices)
    for v in bm.verts:
        v.select = v.index in target
    bmesh.update_edit_mesh(obj.data)


try:
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_addon_data"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    solver_api = api_mod.solver

    log("setup_start")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "StaleCube"

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(cube.name)
    group_index = 0

    # ---- A: first Create produces exactly the selected verts --------
    select_verts(cube, [0, 1, 2])
    first_result = bpy.ops.object.create_pin_vertex_group(
        "EXEC_DEFAULT", group_index=group_index, vg_name="X",
    )
    first_members = vg_members(cube, "X")
    record(
        "A_first_create_produces_exact_selection",
        first_result == {"FINISHED"} and first_members == {0, 1, 2},
        {
            "op_result": list(first_result),
            "members": sorted(first_members) if first_members else None,
        },
    )

    # ---- B: Remove drops the pin list slot but keeps the vg ---------
    root = groups_mod.get_addon_data(bpy.context.scene)
    group = root.object_group_0
    group.pin_vertex_groups_index = 0
    pre_remove_pin_count = len(group.pin_vertex_groups)
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.remove_pin_vertex_group(
        "EXEC_DEFAULT", group_index=group_index,
    )
    post_remove_pin_count = len(group.pin_vertex_groups)
    persisted_members = vg_members(cube, "X")
    record(
        "B_remove_keeps_underlying_vg",
        pre_remove_pin_count == 1
        and post_remove_pin_count == 0
        and persisted_members == {0, 1, 2},
        {
            "pre_remove_pin_count": pre_remove_pin_count,
            "post_remove_pin_count": post_remove_pin_count,
            "persisted_members": (
                sorted(persisted_members) if persisted_members else None
            ),
        },
    )

    # ---- C: second Create with same name clears stale members -------
    select_verts(cube, [4, 5])
    second_result = bpy.ops.object.create_pin_vertex_group(
        "EXEC_DEFAULT", group_index=group_index, vg_name="X",
    )
    second_members = vg_members(cube, "X")
    record(
        "C_second_create_clears_stale",
        second_result == {"FINISHED"} and second_members == {4, 5},
        {
            "op_result": list(second_result),
            "members": sorted(second_members) if second_members else None,
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
