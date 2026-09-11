# File: scenarios/bl_mcp_vertex_groups.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP vertex-group surface, which is what makes a mesh pin reachable.
#
# A mesh pin does not carry its own vertices: it names a Blender vertex group
# that already holds them, and ``core.mutation.create_pin`` refuses the pin
# unless the object carries a group of that name. An MCP caller therefore has
# to be able to read a mesh's vertex groups and to create one, and until the
# two tools below existed the whole pinning path was reachable from the UI and
# from the scripting API but not over MCP.
#
# This scenario drives the finished chain end to end against a real Blender:
# read the groups, create the one the pin will name, put the object in a
# dynamics group, pin the vertex group, and read the pin back. The mesh is
# built with plain ``bpy`` datablock calls, so nothing here needs a solver, a
# connection or a build.
#
# The refusals are the other half, and they are asserted on the tool payload
# rather than on the JSON-RPC envelope. A handler that ran and rejected its
# arguments is a well-formed result carrying ``isError``, whose payload has
# ``status`` "error" and a message; a protocol error is reserved for a call
# that never reached a tool. Each refusal also has to leave the scene as it
# was found, so each one re-reads the state it could have damaged.
#
# Assertions:
#   A. ``vertex_group_tools_are_registered`` -- tools/list carries
#      list_vertex_groups, create_vertex_group and add_pin_vertex_group, each
#      requiring the arguments the chain below passes.
#   B. ``fresh_mesh_reports_no_vertex_groups`` -- a mesh that has never been
#      given one reports an empty list, not an absent key or an error.
#   C. ``create_vertex_group_assigns_named_vertices`` -- the group is created
#      holding exactly the named vertices, Blender's own membership agrees,
#      and list_vertex_groups then reports it with that vertex count.
#   D. ``pinning_a_named_vertex_group_succeeds`` -- create_group,
#      add_objects_to_group and add_pin_vertex_group take the vertex group to
#      a pin, and list_pins reports it by its "object::group" identifier.
#   E. ``unknown_vertex_group_is_refused`` -- pinning a name the object does
#      not carry is refused with a message naming it, and the pin list is
#      unchanged.
#   F. ``non_mesh_object_is_refused`` -- create_vertex_group on a curve is
#      refused by type, and the curve gains nothing.
#   G. ``out_of_range_index_is_refused`` -- an index the mesh does not have is
#      refused with the valid range, and no group is created.
#   H. ``duplicate_name_is_refused`` -- a second group of the same name is
#      refused, and the existing membership is not overwritten.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")
# No solver, no build and no connection: bpy datablock calls and the add-on's
# own HTTP server, so the assertions hold identically on either backend.
BACKENDS = ("emulated", "real")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

MESH_NAME = "PinMesh"
CURVE_NAME = "PinCurve"
VG_NAME = "PinnedRow"
PIN_INDICES = [6, 7, 8]
GRID_VERTS = 9


def build_grid(name):
    # A 3 by 3 vertex sheet of four quads, built from the datablock API so the
    # vertex indices this scenario names are the ones from_pydata assigned.
    coords = [(col * 0.5, row * 0.5, 0.0) for row in range(3) for col in range(3)]
    faces = [
        (row * 3 + col, row * 3 + col + 1, row * 3 + col + 4, row * 3 + col + 3)
        for row in range(2)
        for col in range(2)
    ]
    mesh = bpy.data.meshes.new(name + "Data")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    return obj


def build_curve(name):
    # A curve object, for the type refusal. A curve has no vertex groups; its
    # pinned control points live in a _pin_<name> custom property instead.
    curve = bpy.data.curves.new(name + "Data", type="CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("POLY")
    spline.points.add(2)
    for index, point in enumerate(spline.points):
        point.co = (index * 0.5, 0.0, 0.0, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return obj


def group_names(payload):
    return [entry.get("name") for entry in payload.get("vertex_groups") or []]


def blender_membership(obj, vg_name):
    # The membership Blender itself recorded, read from the mesh rather than
    # from the tool's own report of what it wrote.
    vertex_group = obj.vertex_groups.get(vg_name)
    if vertex_group is None:
        return None
    return sorted(
        vertex.index
        for vertex in obj.data.vertices
        for element in vertex.groups
        if element.group == vertex_group.index
    )


def only_entry(entries):
    # The single element, or an empty dict so a wrong count fails the check
    # instead of raising out of it.
    return entries[0] if len(entries) == 1 else {}


try:
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing, do_unlink=True)
    mesh_obj = build_grid(MESH_NAME)
    curve_obj = build_curve(CURVE_NAME)
    result["phases"].append((time.time(), "scene_built"))

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    # ----- A. the three tools reach the registry ------------------
    env, resp = mcp_call(pkg, url, "tools/list", request_id=1)
    schemas = {
        tool.get("name"): tool
        for tool in ((env.get("result") or {}).get("tools") or [])
    }
    required = {}
    for tool_name in (
        "list_vertex_groups",
        "create_vertex_group",
        "add_pin_vertex_group",
    ):
        schema = (schemas.get(tool_name) or {}).get("inputSchema") or {}
        required[tool_name] = sorted(schema.get("required") or [])
    mcp_check(
        result, "A_vertex_group_tools_are_registered",
        required["list_vertex_groups"] == ["object_name"]
        and required["create_vertex_group"] == ["indices", "name", "object_name"]
        and required["add_pin_vertex_group"]
        == ["group_uuid", "vertex_group_identifier"],
        {"required": required, "tool_count": len(schemas)},
    )

    # ----- B. a mesh that carries no vertex group -----------------
    fresh, _ = mcp_tool(
        pkg, url, "list_vertex_groups", {"object_name": MESH_NAME}, request_id=2
    )
    mcp_check(
        result, "B_fresh_mesh_reports_no_vertex_groups",
        fresh.get("status") == "success"
        and fresh.get("vertex_groups") == []
        and fresh.get("vertex_group_count") == 0
        and fresh.get("mesh_vertex_count") == GRID_VERTS
        and fresh.get("object_name") == MESH_NAME,
        fresh,
    )

    # ----- C. creating the membership a pin will name -------------
    created, _ = mcp_tool(
        pkg, url, "create_vertex_group",
        {"object_name": MESH_NAME, "name": VG_NAME, "indices": PIN_INDICES},
        request_id=3,
    )
    listed, _ = mcp_tool(
        pkg, url, "list_vertex_groups", {"object_name": MESH_NAME}, request_id=4
    )
    membership = blender_membership(mesh_obj, VG_NAME)
    entry = only_entry(listed.get("vertex_groups") or [])
    mcp_check(
        result, "C_create_vertex_group_assigns_named_vertices",
        created.get("status") == "success"
        and created.get("vertex_group_name") == VG_NAME
        and created.get("vertex_count") == len(PIN_INDICES)
        and created.get("weight") == 1.0
        and membership == sorted(PIN_INDICES)
        and listed.get("vertex_group_count") == 1
        and entry.get("name") == VG_NAME
        and entry.get("vertex_count") == len(PIN_INDICES),
        {"created": created, "listed": listed, "blender_membership": membership},
    )

    # ----- D. the pin itself --------------------------------------
    group_res, _ = mcp_tool(
        pkg, url, "create_group", {"name": "Cloth", "type": "SHELL"}, request_id=5
    )
    group_uuid = group_res.get("group_uuid") or ""
    assigned, _ = mcp_tool(
        pkg, url, "add_objects_to_group",
        {"group_uuid": group_uuid, "object_names": [MESH_NAME]},
        request_id=6,
    )
    identifier = MESH_NAME + "::" + VG_NAME
    pinned, _ = mcp_tool(
        pkg, url, "add_pin_vertex_group",
        {"group_uuid": group_uuid, "vertex_group_identifier": identifier},
        request_id=7,
    )
    pins, _ = mcp_tool(
        pkg, url, "list_pins", {"group_uuid": group_uuid}, request_id=8
    )
    pin_entry = only_entry(pins.get("pins") or [])
    mcp_check(
        result, "D_pinning_a_named_vertex_group_succeeds",
        group_res.get("status") == "success"
        and bool(group_uuid)
        and assigned.get("status") == "success"
        and len(assigned.get("added_objects") or []) == 1
        and (assigned.get("warnings") or []) == []
        and pinned.get("status") == "success"
        and pinned.get("vertex_group_name") == VG_NAME
        and pinned.get("object_name") == MESH_NAME
        and pins.get("status") == "success"
        and pins.get("pin_count") == 1
        and pin_entry.get("vertex_group_identifier") == identifier
        and pin_entry.get("object_name") == MESH_NAME
        and bool(pin_entry.get("object_uuid")),
        {
            "group_uuid": group_uuid,
            "assigned": assigned,
            "pinned": pinned,
            "pins": pins,
        },
    )

    # ----- E. pinning a vertex group that does not exist ----------
    absent_name = "NoSuchVertexGroup"
    refused_pin, raw_pin = mcp_tool(
        pkg, url, "add_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": MESH_NAME + "::" + absent_name,
        },
        request_id=9,
    )
    pins_after, _ = mcp_tool(
        pkg, url, "list_pins", {"group_uuid": group_uuid}, request_id=10
    )
    pin_message = refused_pin.get("message") or ""
    mcp_check(
        result, "E_unknown_vertex_group_is_refused",
        refused_pin.get("status") == "error"
        and raw_pin.get("isError") is True
        and absent_name in pin_message
        and MESH_NAME in pin_message
        and pins_after.get("pin_count") == 1,
        {
            "payload": refused_pin,
            "isError": raw_pin.get("isError"),
            "pin_count_after": pins_after.get("pin_count"),
        },
    )

    # ----- F. a vertex group on something that is not a mesh ------
    refused_curve, raw_curve = mcp_tool(
        pkg, url, "create_vertex_group",
        {"object_name": CURVE_NAME, "name": "CurveGroup", "indices": [0]},
        request_id=11,
    )
    curve_message = refused_curve.get("message") or ""
    mcp_check(
        result, "F_non_mesh_object_is_refused",
        refused_curve.get("status") == "error"
        and raw_curve.get("isError") is True
        and CURVE_NAME in curve_message
        and "CURVE" in curve_message
        and len(curve_obj.vertex_groups) == 0,
        {
            "payload": refused_curve,
            "isError": raw_curve.get("isError"),
            "curve_vertex_groups": len(curve_obj.vertex_groups),
        },
    )

    # ----- G. an index the mesh does not have ---------------------
    refused_range, raw_range = mcp_tool(
        pkg, url, "create_vertex_group",
        {
            "object_name": MESH_NAME,
            "name": "OutOfRange",
            "indices": [0, GRID_VERTS],
        },
        request_id=12,
    )
    after_range, _ = mcp_tool(
        pkg, url, "list_vertex_groups", {"object_name": MESH_NAME}, request_id=13
    )
    range_message = refused_range.get("message") or ""
    mcp_check(
        result, "G_out_of_range_index_is_refused",
        refused_range.get("status") == "error"
        and raw_range.get("isError") is True
        and "out of range" in range_message
        and str(GRID_VERTS) in range_message
        and group_names(after_range) == [VG_NAME],
        {
            "payload": refused_range,
            "isError": raw_range.get("isError"),
            "vertex_groups_after": group_names(after_range),
        },
    )

    # ----- H. a name the object already carries -------------------
    refused_dup, raw_dup = mcp_tool(
        pkg, url, "create_vertex_group",
        {"object_name": MESH_NAME, "name": VG_NAME, "indices": [0]},
        request_id=14,
    )
    after_dup, _ = mcp_tool(
        pkg, url, "list_vertex_groups", {"object_name": MESH_NAME}, request_id=15
    )
    dup_message = refused_dup.get("message") or ""
    dup_entry = only_entry(after_dup.get("vertex_groups") or [])
    mcp_check(
        result, "H_duplicate_name_is_refused",
        refused_dup.get("status") == "error"
        and raw_dup.get("isError") is True
        and VG_NAME in dup_message
        and "already has a vertex group" in dup_message
        and group_names(after_dup) == [VG_NAME]
        and dup_entry.get("vertex_count") == len(PIN_INDICES)
        and blender_membership(mesh_obj, VG_NAME) == sorted(PIN_INDICES),
        {
            "payload": refused_dup,
            "isError": raw_dup.get("isError"),
            "vertex_groups_after": after_dup.get("vertex_groups"),
            "membership_after": blender_membership(mesh_obj, VG_NAME),
        },
    )

    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
