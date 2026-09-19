# File: scenarios/bl_mcp_group_lifecycle.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The dynamics group lifecycle over MCP, from create to clear_solver. A group
# is the unit the solver simulates, so this is the arc every agent-driven
# scene walks before anything else can happen.
#
# Assertions:
#   A. ``create_rename_and_type`` -- a new group is SOLID, renaming reports
#      both names, and set_group_type accepts a valid type.
#   B. ``invalid_type_names_the_valid_ones`` -- an unknown type is refused
#      with every accepted type listed.
#   C. ``assignment_and_inclusion`` -- objects assign to the group, get_group
#      reports them, and set_object_included flips one without unassigning it.
#   D. ``unknown_group_is_refused_on_every_path`` -- get_group, rename_group
#      and duplicate_group each refuse an unknown uuid with the same wording.
#   E. ``geometry_queries_answer_and_refuse`` -- get_average_edge_length and
#      get_object_bounding_box_diagonal measure a real object, and each names
#      an object that is not in the scene.
#   F. ``duplicate_copies_settings_not_membership`` -- duplicate_group makes a
#      group with its own uuid, a derived name and the source's type, and
#      leaves the objects where they were: an object belongs to one group, so
#      a copy that took the membership would empty the original.
#   G. ``get_active_groups_does_not_write`` -- the listing is annotated
#      readOnlyHint, which is a factual claim, so a call must leave the
#      groups' stored display indices exactly as it found them.
#   H. ``removal_and_clear_solver`` -- removing an object twice refuses the
#      second time, remove_all reports the count, and clear_solver leaves no
#      active group.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives the addon's in-process MCP server and never asks the solver to
# step, so nothing in it is backend-specific. A rig run against a CPU build
# passed it, and that run is the evidence this line rests on.
BACKENDS = ("real",)

NEEDS_BLENDER = True

# macOS runners block loopback HTTP to Blender's in-process MCP server, so the
# rig does not select this scenario there.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

_rid = [500]


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


def stored_indices():
    # Read the groups' own display indices straight off the RNA, which is what
    # a readOnly claim is measured against.
    groups_mod = __import__(pkg + ".models.groups",
                           fromlist=["iterate_active_object_groups"])
    return [g.index for g in groups_mod.iterate_active_object_groups(bpy.context.scene)]


try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for name, loc in (("Alpha", (0, 0, 0)), ("Beta", (3, 0, 0))):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4,
                                        size=2, location=loc)
        bpy.context.object.name = name

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    call("clear_solver")
    result["phases"].append((time.time(), "scene_built"))

    # ----- A --------------------------------------------------------
    created = call("create_group")
    group = created["group_uuid"]
    renamed = call("rename_group", {"group_uuid": group, "name": "Cloth"})
    typed = call("set_group_type", {"group_uuid": group, "type": "SHELL"})
    mcp_check(result, "A_create_rename_and_type",
              created.get("status") == "success"
              and (created.get("group") or {}).get("object_type") == "SOLID"
              and renamed.get("status") == "success" and renamed.get("name") == "Cloth"
              and typed.get("status") == "success" and typed.get("type") == "SHELL",
              {"created": created.get("group"), "renamed": renamed, "typed": typed})

    # ----- B --------------------------------------------------------
    bad_type = call("set_group_type", {"group_uuid": group, "type": "BOGUS"})
    message_b = bad_type.get("message", "")
    mcp_check(result, "B_invalid_type_names_the_valid_ones",
              bad_type.get("status") == "error" and "BOGUS" in message_b
              and all(t in message_b for t in
                      ("SOLID", "SHELL", "ROD", "STATIC", "PDRD", "SAND")),
              {"reply": bad_type})

    # ----- C --------------------------------------------------------
    assigned = call("add_objects_to_group",
                    {"group_uuid": group, "object_names": ["Alpha", "Beta"]})
    fetched = call("get_group", {"group_uuid": group}).get("group") or {}
    excluded = call("set_object_included",
                    {"group_uuid": group, "object_name": "Beta", "included": False})
    after_c = call("get_group", {"group_uuid": group}).get("group") or {}
    flags = {o.get("name"): o.get("included") for o in (after_c.get("assigned_objects") or [])}
    mcp_check(result, "C_assignment_and_inclusion",
              assigned.get("status") == "success" and len(assigned.get("added_objects") or []) == 2
              and fetched.get("object_count") == 2
              and excluded.get("status") == "success" and excluded.get("included") is False
              # Excluding does not unassign: the object stays in the group.
              and after_c.get("object_count") == 2
              and flags == {"Alpha": True, "Beta": False},
              {"assigned": assigned, "flags": flags, "count": after_c.get("object_count")})

    # ----- D --------------------------------------------------------
    refusals = {
        "get_group": call("get_group", {"group_uuid": "nope"}),
        "rename_group": call("rename_group", {"group_uuid": "nope", "name": "x"}),
        "duplicate_group": call("duplicate_group", {"group_uuid": "nope"}),
    }
    mcp_check(result, "D_unknown_group_is_refused_on_every_path",
              all(v.get("status") == "error" and "nope" in v.get("message", "")
                  and "not found" in v.get("message", "")
                  for v in refusals.values()),
              {k: v.get("message") for k, v in refusals.items()})

    # ----- E --------------------------------------------------------
    edge = call("get_average_edge_length", {"object_name": "Alpha"})
    diag = call("get_object_bounding_box_diagonal", {"object_name": "Alpha"})
    edge_missing = call("get_average_edge_length", {"object_name": "Ghost"})
    diag_missing = call("get_object_bounding_box_diagonal", {"object_name": "Ghost"})
    mcp_check(result, "E_geometry_queries_answer_and_refuse",
              edge.get("status") == "success" and edge.get("average_edge_length", 0) > 0
              and edge.get("total_vertices", 0) > 0
              and diag.get("status") == "success"
              and diag.get("largest_diagonal_distance", 0) > 0
              and edge_missing.get("status") == "error" and "Ghost" in edge_missing.get("message", "")
              and diag_missing.get("status") == "error" and "Ghost" in diag_missing.get("message", ""),
              {"edge": edge.get("average_edge_length"),
               "diagonal": diag.get("largest_diagonal_distance"),
               "edge_missing": edge_missing.get("message"),
               "diag_missing": diag_missing.get("message")})

    # ----- F --------------------------------------------------------
    copy = call("duplicate_group", {"group_uuid": group})
    copy_uuid = copy.get("group_uuid")
    copy_group = call("get_group", {"group_uuid": copy_uuid}).get("group") or {}
    source_after = call("get_group", {"group_uuid": group}).get("group") or {}
    mcp_check(result, "F_duplicate_copies_settings_not_membership",
              copy.get("status") == "success"
              and copy_uuid and copy_uuid != group
              and copy.get("group_name") != "Cloth"
              and copy_group.get("object_type") == "SHELL"
              # An object belongs to one group, so the copy starts empty and
              # the source keeps everything it had.
              and copy_group.get("object_count") == 0
              and source_after.get("object_count") == 2,
              {"copy": copy, "copy_group": copy_group,
               "source_after": source_after.get("object_count")})

    # ----- G --------------------------------------------------------
    # Scramble the stored indices first: if the listing renumbers them, this
    # is where it shows.
    groups_mod = __import__(pkg + ".models.groups",
                           fromlist=["iterate_active_object_groups"])
    for g in groups_mod.iterate_active_object_groups(bpy.context.scene):
        g.index = 99
    before_g = stored_indices()
    listing = call("get_active_groups")
    after_g = stored_indices()
    mcp_check(result, "G_get_active_groups_does_not_write",
              listing.get("status") == "success"
              and listing.get("group_count") == 2
              and before_g == after_g and set(after_g) == {99},
              {"before": before_g, "after": after_g,
               "group_count": listing.get("group_count")})

    # ----- H --------------------------------------------------------
    rm = call("remove_object_from_group", {"group_uuid": group, "object_name": "Alpha"})
    rm_again = call("remove_object_from_group", {"group_uuid": group, "object_name": "Alpha"})
    rm_all = call("remove_all_objects_from_group", {"group_uuid": group})
    call("clear_solver")
    final = call("get_active_groups")
    mcp_check(result, "H_removal_and_clear_solver",
              rm.get("status") == "success"
              and rm_again.get("status") == "error" and "Alpha" in rm_again.get("message", "")
              and rm_all.get("status") == "success" and rm_all.get("objects_removed") == 1
              and final.get("group_count") == 0 and final.get("groups") == [],
              {"removed": rm, "again": rm_again, "removed_all": rm_all, "final": final})

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
