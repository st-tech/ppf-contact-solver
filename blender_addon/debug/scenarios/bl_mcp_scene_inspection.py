# File: scenarios/bl_mcp_scene_inspection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The scene enumeration an MCP agent starts from, against a real Blender.
#
# ``get_scene_info`` is the first call an agent makes against a file it did
# not author. It is the only place the caller learns which objects exist,
# what geometry each one carries, and which of them a dynamics group already
# claims. Every later tool addresses an object or a group by name, so a wrong
# answer here does not surface as a reporting defect: it sends the rest of
# the session to the wrong target.
#
# The report is assembled from two sources that can disagree. The frame range
# and the object list come from Blender, while the group list and the
# ``in_group`` flag come from addon state, and the simulation fps comes from
# ``resolve_fps`` rather than from either. This scenario builds a two-object
# scene with plain ``bpy`` so every count is known exactly, reads the report
# over the MCP transport, then creates a group and assigns one of the two
# objects through the same transport and reads it again. The second reading
# is what shows the report following addon state rather than restating the
# Blender scene.
#
# Assertions:
#   A. ``scene_header_matches_blender`` -- the scene name, frame_start,
#      frame_end, frame_current and blender_fps equal what Blender holds,
#      measured against a frame range set to values no default supplies.
#   B. ``objects_carry_type_and_geometry`` -- both meshes are reported with
#      type MESH and the exact vertex and face counts they were built with.
#   C. ``object_count_and_sort_order`` -- object_count equals the list length
#      and the list is sorted by name, though the scene holds the two objects
#      in the opposite (creation) order.
#   D. ``group_count_is_zero_before_setup`` -- with no groups the group list
#      is empty, group_count is 0, and every object reports in_group false.
#   E. ``simulation_fps_is_positive`` -- the fps the solver would run at is
#      reported as a positive number, separately from blender_fps.
#   F. ``created_group_is_reported`` -- after create_group and
#      add_objects_to_group the report carries one group with its name, type,
#      uuid and member list.
#   G. ``in_group_marks_only_the_assigned_object`` -- the assigned object
#      flips to in_group true while the unassigned one stays false, and the
#      object list itself is unchanged.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives the addon's in-process MCP server and never asks the solver to
# step, so nothing in it is backend-specific. A rig run against a CPU build
# passed it, and that run is the evidence this line rests on.
BACKENDS = ("real",)

NEEDS_BLENDER = True

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    # ----- a two-object scene whose counts are known exactly ------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    def new_mesh(name, coords, faces):
        mesh = bpy.data.meshes.new(name + "Mesh")
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        mesh.from_pydata([tuple(c) for c in coords], [], [tuple(f) for f in faces])
        mesh.update()
        return obj

    # Built in reverse alphabetical order on purpose: a report that echoed
    # the scene's own iteration order would still satisfy every other check
    # in C, so the sort is only measurable against a scene that is unsorted.
    sheet = new_mesh(
        "Zeta_Sheet",
        [(float(x), float(y), 0.0) for y in (0, 1, 2) for x in (0, 1, 2)],
        [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)],
    )
    block = new_mesh(
        "Alpha_Block",
        [(0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, 2.0), (0.0, 1.0, 2.0)],
        [(0, 1, 2, 3)],
    )

    # A frame range no default supplies, so a report that returned constants
    # or read a different scene cannot match.
    scene = bpy.context.scene
    scene.frame_start = 3
    scene.frame_end = 11
    scene.frame_set(5)

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    next_id = [0]

    def scene_info():
        # get_scene_info takes no arguments; a failed status is a transport
        # or handler failure and stops the scenario rather than becoming a
        # check whose details no longer describe anything.
        next_id[0] += 1
        payload, _ = mcp_tool(
            pkg, url, "get_scene_info", {}, request_id=next_id[0]
        )
        if payload.get("status") != "success":
            raise RuntimeError("get_scene_info failed: %r" % (payload,))
        return payload

    before = scene_info()

    # ----- A. the header restates Blender, not a default ----------
    header = before.get("scene") or {}
    live = {
        "name": scene.name,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "frame_current": scene.frame_current,
        "blender_fps": scene.render.fps,
    }
    mcp_check(
        result, "A_scene_header_matches_blender",
        header.get("name") == live["name"]
        and header.get("frame_start") == live["frame_start"] == 3
        and header.get("frame_end") == live["frame_end"] == 11
        and header.get("frame_current") == live["frame_current"] == 5
        and header.get("blender_fps") == live["blender_fps"],
        {"reported": header, "blender": live},
    )

    # ----- B. one entry per object, with its geometry -------------
    objects = before.get("objects") or []
    by_name = {entry.get("name"): entry for entry in objects}
    sheet_entry = by_name.get(sheet.name) or {}
    block_entry = by_name.get(block.name) or {}
    mcp_check(
        result, "B_objects_carry_type_and_geometry",
        sheet_entry.get("type") == "MESH"
        and sheet_entry.get("vertex_count") == 9
        and sheet_entry.get("face_count") == 4
        and block_entry.get("type") == "MESH"
        and block_entry.get("vertex_count") == 4
        and block_entry.get("face_count") == 1,
        {
            "sheet": sheet_entry,
            "block": block_entry,
            "expected": {
                sheet.name: {"vertex_count": 9, "face_count": 4},
                block.name: {"vertex_count": 4, "face_count": 1},
            },
        },
    )

    # ----- C. the count matches the list, and the list is sorted --
    reported_names = [entry.get("name") for entry in objects]
    scene_order = [obj.name for obj in scene.objects]
    mcp_check(
        result, "C_object_count_and_sort_order",
        before.get("object_count") == len(objects)
        and len(objects) == 2
        and reported_names == sorted(reported_names)
        and reported_names == [block.name, sheet.name],
        {
            "object_count": before.get("object_count"),
            "reported_names": reported_names,
            "scene_iteration_order": scene_order,
        },
    )

    # ----- D. nothing is claimed by a group yet -------------------
    mcp_check(
        result, "D_group_count_is_zero_before_setup",
        before.get("group_count") == 0
        and (before.get("groups") or []) == []
        and sheet_entry.get("in_group") is False
        and block_entry.get("in_group") is False,
        {
            "group_count": before.get("group_count"),
            "groups": before.get("groups"),
            "in_group": {
                name: entry.get("in_group") for name, entry in by_name.items()
            },
        },
    )

    # ----- E. the fps the solver would use ------------------------
    simulation = before.get("simulation") or {}
    fps = simulation.get("fps")
    mcp_check(
        result, "E_simulation_fps_is_positive",
        isinstance(fps, int | float)
        and not isinstance(fps, bool)
        and fps > 0,
        {
            "simulation": simulation,
            "fps_type": type(fps).__name__,
            "blender_fps": header.get("blender_fps"),
        },
    )

    # ----- set a group up over the same transport -----------------
    created, _ = mcp_tool(
        pkg, url, "create_group",
        {"name": "Cloth", "type": "SHELL"}, request_id=20,
    )
    if created.get("status") != "success":
        raise RuntimeError("create_group failed: %r" % (created,))
    group_uuid = created.get("group_uuid") or ""
    if not group_uuid:
        raise RuntimeError("create_group returned no group_uuid: %r" % (created,))

    added, _ = mcp_tool(
        pkg, url, "add_objects_to_group",
        {"group_uuid": group_uuid, "object_names": [block.name]},
        request_id=21,
    )
    if added.get("status") != "success":
        raise RuntimeError("add_objects_to_group failed: %r" % (added,))
    if added.get("warnings"):
        raise RuntimeError(
            "add_objects_to_group warned: %r" % (added.get("warnings"),)
        )
    result["phases"].append((time.time(), "group_uuid=%s" % group_uuid))

    after = scene_info()

    # ----- F. the group reaches the report ------------------------
    groups = after.get("groups") or []
    group = groups[0] if groups else {}
    mcp_check(
        result, "F_created_group_is_reported",
        after.get("group_count") == 1
        and len(groups) == 1
        and group.get("name") == "Cloth"
        and group.get("object_type") == "SHELL"
        and group.get("uuid") == group_uuid
        and group.get("object_names") == [block.name],
        {
            "group_count": after.get("group_count"),
            "groups": groups,
            "expected_uuid": group_uuid,
            "expected_member": block.name,
        },
    )

    # ----- G. only the assigned object is marked ------------------
    after_objects = after.get("objects") or []
    after_by_name = {entry.get("name"): entry for entry in after_objects}
    after_block = after_by_name.get(block.name) or {}
    after_sheet = after_by_name.get(sheet.name) or {}
    mcp_check(
        result, "G_in_group_marks_only_the_assigned_object",
        after_block.get("in_group") is True
        and after_sheet.get("in_group") is False
        and after.get("object_count") == 2
        and [entry.get("name") for entry in after_objects]
        == [block.name, sheet.name],
        {
            "assigned": {
                "name": block.name,
                "in_group": after_block.get("in_group"),
            },
            "unassigned": {
                "name": sheet.name,
                "in_group": after_sheet.get("in_group"),
            },
            "object_count": after.get("object_count"),
            "reported_names": [entry.get("name") for entry in after_objects],
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
