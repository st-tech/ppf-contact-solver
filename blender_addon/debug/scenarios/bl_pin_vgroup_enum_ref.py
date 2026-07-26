# File: scenarios/bl_pin_vgroup_enum_ref.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Behavior guard for the reported bug: a pin vertex-group dropdown garbled
# when the vertex group had a non-ASCII (Japanese) name.
#
# Blender does NOT copy the item strings a dynamic EnumProperty items callback
# returns; its C side holds pointers into the Python str objects. If the list
# is not kept referenced past the callback return, the dropdown draws freed
# memory: garbled characters and many bogus entries. A Japanese name like
# "固定" makes it deterministic (a separate per-str UTF-8 cache buffer is freed
# with the object). The retention is now guaranteed by the @dynamic_enum_items
# decorator (models/enum_props.py), which keeps the last returned list alive.
#
#   A. vgroup_single_clean_entry: a non-ASCII vertex group yields exactly one
#      dropdown item with the name intact, not many garbled ones.
#   B. vgroup_ref_retained_after_gc: the decorator's retention cell still holds
#      the item strings after a forced GC plus allocation churn (the exact
#      free-and-reuse that garbled the UI).
#
# The structural enforcement (a forgotten decorator hard-errors at load) is
# covered separately by bl_enum_props_guard.
#
# Pure UI scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, gc, time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


# Non-ASCII vertex group name from the bug report.
JP = "固定"  # "固定"


try:
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_vertex_group_items"])
    enum_mod = __import__(pkg + ".models.enum_props",
                          fromlist=["dynamic_enum_items"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    solver_api = api_mod.solver

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "JPCube"
    vg = cube.vertex_groups.new(name=JP)
    vg.add([0, 1, 2, 3], 1.0, "REPLACE")

    grp = solver_api.create_group("JPGroup", "SHELL")
    grp.add(cube.name)

    # Relocate the ObjectGroup PropertyGroup by its stable UUID.
    group_uuid = grp.uuid
    root = groups_mod.get_addon_data(bpy.context.scene)
    group = None
    for i in range(32):
        g = getattr(root, "object_group_" + str(i), None)
        if g is not None and getattr(g, "uuid", "") == group_uuid:
            group = g
            break
    assert group is not None, "freshly-created group not found by uuid"

    # ----- A: the dropdown resolves to one clean entry -----------------
    items = groups_mod.get_vertex_group_items(group, None)
    ident = groups_mod.encode_vertex_group_identifier(cube.name, JP)
    record(
        "A_vgroup_single_clean_entry",
        len(items) == 1
        and items[0][0] == ident
        and JP in items[0][0]
        and JP in items[0][1],
        {"items": items, "expected_ident": ident},
    )

    # ----- B: the decorator's retention cell is GC-durable -------------
    # @dynamic_enum_items exposes its retention cell; a missing decorator
    # (the regression) means the callback would fail EnumProperty at load,
    # and here the holder attribute would be absent.
    holder = getattr(groups_mod.get_vertex_group_items, "_ppf_enum_items_holder", None)
    holder_ok = isinstance(holder, list) and len(holder) == 1 and holder[0] is items
    del items
    # Churn the allocator so any freed string buffer would be reused; this
    # is what turned the dangling pointers into garbled glyphs in the UI.
    junk = ["gc-fill-" + str(i) * 9 for i in range(200000)]
    del junk
    gc.collect()
    retained = getattr(groups_mod.get_vertex_group_items, "_ppf_enum_items_holder", None)
    survived = (
        isinstance(retained, list)
        and len(retained) == 1
        and len(retained[0]) == 1
        and retained[0][0][0] == ident
        and JP in retained[0][0][1]
    )
    record(
        "B_vgroup_ref_retained_after_gc",
        holder_ok and survived,
        {"holder_is_return": holder_ok, "retained_after_gc": retained},
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
