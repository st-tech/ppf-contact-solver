# File: scenarios/bl_mcp_bend_reference.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP set_bend_reference surface, against a real Blender.
#
# A bending reference is a topological copy of an assigned object whose
# vertices were moved: the group takes its hinge rest angles from that copy
# instead of from the object's own initial pose. Two switches have to line up
# before the scene build reads one, and they live on different records. The
# GROUP carries bend_rest_from_reference, which only a SHELL or ROD group
# offers as a material parameter, and the assigned OBJECT carries the
# reference itself plus its own enable bit. set_bend_reference writes the
# object half, and refuses every write that would come to rest where nothing
# reads it, so each gate below is driven from both sides.
#
# The reference object is stored by UUID and the name beside it is only a
# label, so the checks read all three back: what the call reported, what
# get_group_objects reports, and what the PropertyGroup holds.
#
# A reference is compared against, never simulated, so it is NOT a member of
# the group that reads it. The reference here is assigned to no group at all,
# which is what check C exercises, and the two by-name refusals cover the
# names that must resolve: the reference has to be an object in the scene,
# and the target has to be a member of the named group.
#
# The scene holds a flat grid assigned to a SHELL group, a bent
# positions-only copy of it, a coarser grid whose vertex count does not
# match, and a cube assigned to a SOLID group, which is a type that carries
# no bending term and so cannot use a reference at all.
#
# Assertions:
#   A. ``reference_needs_group_flag`` -- while the SHELL group's
#      bend_rest_from_reference is off, a reference is refused, and the
#      message names both the flag and set_group_material_properties as the
#      fix. Nothing is stored.
#   B. ``unsupported_group_type_is_refused`` -- a SOLID group is refused by
#      type, naming the two types that do take a reference.
#   C. ``reference_assigned_and_reported`` -- with the flag on, the reference
#      is assigned, and the call, get_group_objects and the stored
#      PropertyGroup agree on its UUID, its name and its enable bit. The
#      group still holds one object, so the reference is accepted while
#      belonging to no group.
#   D. ``unknown_reference_refused_by_name`` -- a reference naming no object
#      in the scene is refused, the message names it, and the assignment
#      from C is untouched.
#   E. ``unassigned_object_refused_by_name`` -- a target that is in the scene
#      but not a member of the named group is refused, naming the object and
#      the group.
#   F. ``topology_refusal_passes_message_through`` -- a reference whose
#      vertex count does not match is refused with exactly the message
#      validate_bend_reference produced for that pair, and the assignment
#      from C is untouched.
#   G. ``empty_name_clears_and_disables`` -- an empty reference name clears
#      the UUID and the label and turns the per-object enable bit off, which
#      get_group_objects reports as a null object name.
#   H. ``enable_without_reference_is_refused`` -- enable=True alongside an
#      empty name is refused, since there would be nothing to enable, and
#      the cleared state stays cleared.
#   I. ``enable_false_records_without_using`` -- a reference passed with
#      enable=False is stored and reported while the enable bit stays off,
#      which is how a reference is recorded before it is used.

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

SHEET = "BendSheet"
REFERENCE = "BendSheetReference"
COARSE = "BendSheetCoarse"
BLOCK = "BendBlock"

EMPTY_REFERENCE = {
    "bend_ref_uuid": "",
    "bend_ref_name": "",
    "bend_ref_enable": False,
}

try:
    groups = __import__(
        pkg + ".models.groups", fromlist=["get_active_group_by_uuid"]
    )
    utils = __import__(
        pkg + ".core.utils", fromlist=["validate_bend_reference"]
    )

    # ----- scene: a sheet, its reference, a mismatch, a block -----
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=6, y_subdivisions=6, size=2.0, location=(0.0, 0.0, 0.0)
    )
    sheet = bpy.context.active_object
    sheet.name = SHEET

    # A positions-only copy of the sheet: the same vertices and the same
    # faces, curled in +z, which is what a reference has to be.
    reference = sheet.copy()
    reference.data = sheet.data.copy()
    reference.name = REFERENCE
    bpy.context.collection.objects.link(reference)
    for vertex in reference.data.vertices:
        vertex.co.z = 0.3 * vertex.co.x * vertex.co.x
    reference.data.update()

    # A grid of a different resolution, so its vertex count cannot match.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=3, y_subdivisions=3, size=2.0, location=(4.0, 0.0, 0.0)
    )
    coarse = bpy.context.active_object
    coarse.name = COARSE

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 4.0, 0.0))
    block = bpy.context.active_object
    block.name = BLOCK

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    def call(request_id, name, arguments):
        # A tool call that is expected to succeed.
        payload, raw = mcp_tool(pkg, url, name, arguments, request_id=request_id)
        return payload, raw

    def refuse(request_id, name, arguments):
        # A handler that ran and rejected its input answers with an ordinary
        # tools/call result carrying isError, not with a transport failure,
        # so the refusal is read off the envelope rather than from a raise.
        envelope, _resp = mcp_call(
            pkg, url, "tools/call",
            {"name": name, "arguments": arguments},
            request_id=request_id,
        )
        return mcp_tool_payload(envelope), (envelope.get("result") or {})

    def bend_args(group_uuid, object_name, reference_object_name, enable=None):
        arguments = {
            "group_uuid": group_uuid,
            "object_name": object_name,
            "reference_object_name": reference_object_name,
        }
        if enable is not None:
            arguments["enable"] = enable
        return arguments

    def rna_reference(group_uuid, object_name):
        # The add-on's own state, read from the PropertyGroup the handler
        # writes, so a round-trip is measured against the stored value and
        # not only against what the reply repeated back.
        group = groups.get_active_group_by_uuid(bpy.context.scene, group_uuid)
        if group is None:
            raise RuntimeError("group %s is not active" % group_uuid)
        for assigned in group.assigned_objects:
            if assigned.name != object_name:
                continue
            return {
                "bend_ref_uuid": assigned.bend_ref_uuid,
                "bend_ref_name": assigned.bend_ref_name,
                "bend_ref_enable": bool(assigned.bend_ref_enable),
            }
        raise RuntimeError(
            "'%s' is not assigned to group %s" % (object_name, group_uuid)
        )

    def group_objects(request_id, group_uuid):
        listing, _ = call(
            request_id, "get_group_objects", {"group_uuid": group_uuid}
        )
        if listing.get("status") != "success":
            raise RuntimeError("get_group_objects: %r" % (listing,))
        return listing

    def reported_reference(listing, object_name):
        for entry in listing.get("objects") or []:
            if entry.get("name") == object_name:
                return entry.get("bend_reference") or {}
        raise RuntimeError(
            "get_group_objects did not report '%s': %r" % (object_name, listing)
        )

    # ----- a SHELL group holding the sheet, a SOLID one the block -
    shell_created, _ = call(1, "create_group", {"name": "Sheets", "type": "SHELL"})
    shell_uuid = shell_created.get("group_uuid") or ""
    if shell_created.get("status") != "success" or not shell_uuid:
        raise RuntimeError("create_group SHELL: %r" % (shell_created,))

    solid_created, _ = call(2, "create_group", {"name": "Blocks", "type": "SOLID"})
    solid_uuid = solid_created.get("group_uuid") or ""
    if solid_created.get("status") != "success" or not solid_uuid:
        raise RuntimeError("create_group SOLID: %r" % (solid_created,))

    added_sheet, _ = call(
        3,
        "add_objects_to_group",
        {"group_uuid": shell_uuid, "object_names": [SHEET]},
    )
    if added_sheet.get("status") != "success" or not added_sheet.get(
        "added_objects"
    ):
        raise RuntimeError("add_objects_to_group sheet: %r" % (added_sheet,))

    added_block, _ = call(
        4,
        "add_objects_to_group",
        {"group_uuid": solid_uuid, "object_names": [BLOCK]},
    )
    if added_block.get("status") != "success" or not added_block.get(
        "added_objects"
    ):
        raise RuntimeError("add_objects_to_group block: %r" % (added_block,))

    # ----- A. the group flag has to be on first --------------------
    props_before, _ = call(
        5, "get_group_material_properties", {"group_uuid": shell_uuid}
    )
    flag_before = (
        (props_before.get("properties") or {}).get("bend_rest_from_reference")
        or {}
    ).get("value")

    refuse_a, raw_a = refuse(
        6, "set_bend_reference", bend_args(shell_uuid, SHEET, REFERENCE)
    )
    message_a = refuse_a.get("message") or ""
    stored_a = rna_reference(shell_uuid, SHEET)
    mcp_check(
        result, "A_reference_needs_group_flag",
        flag_before is False
        and refuse_a.get("status") == "error"
        and raw_a.get("isError") is True
        and "bend_rest_from_reference" in message_a
        and "set_group_material_properties" in message_a
        and SHEET in message_a
        and stored_a == EMPTY_REFERENCE,
        {
            "flag_before": flag_before,
            "status": refuse_a.get("status"),
            "is_error": raw_a.get("isError"),
            "message": message_a,
            "stored": stored_a,
        },
    )

    # ----- B. a type that carries no bending term ------------------
    refuse_b, raw_b = refuse(
        7, "set_bend_reference", bend_args(solid_uuid, BLOCK, REFERENCE)
    )
    message_b = refuse_b.get("message") or ""
    stored_b = rna_reference(solid_uuid, BLOCK)
    mcp_check(
        result, "B_unsupported_group_type_is_refused",
        refuse_b.get("status") == "error"
        and raw_b.get("isError") is True
        and "SHELL or ROD" in message_b
        and "SOLID" in message_b
        and solid_uuid in message_b
        and stored_b == EMPTY_REFERENCE,
        {
            "status": refuse_b.get("status"),
            "is_error": raw_b.get("isError"),
            "message": message_b,
            "stored": stored_b,
        },
    )

    # ----- C. the flag on, and the reference assigned --------------
    enabled, _ = call(
        8,
        "set_group_material_properties",
        {
            "group_uuid": shell_uuid,
            "properties": {"bend_rest_from_reference": True},
        },
    )
    if enabled.get("status") != "success":
        raise RuntimeError("set_group_material_properties: %r" % (enabled,))

    set_c, _ = call(
        9, "set_bend_reference", bend_args(shell_uuid, SHEET, REFERENCE)
    )
    ref_uuid = set_c.get("reference_object_uuid") or ""
    stored_c = rna_reference(shell_uuid, SHEET)
    listing_c = group_objects(10, shell_uuid)
    reported_c = reported_reference(listing_c, SHEET)
    expected_stored_c = {
        "bend_ref_uuid": ref_uuid,
        "bend_ref_name": REFERENCE,
        "bend_ref_enable": True,
    }
    expected_reported_c = {
        "enabled": True,
        "object_uuid": ref_uuid,
        "object_name": REFERENCE,
        "stored_name": REFERENCE,
    }
    mcp_check(
        result, "C_reference_assigned_and_reported",
        set_c.get("status") == "success"
        and bool(ref_uuid)
        and set_c.get("reference_object_name") == REFERENCE
        and set_c.get("bend_ref_enable") is True
        and set_c.get("object_name") == SHEET
        and stored_c == expected_stored_c
        and reported_c == expected_reported_c
        # The reference belongs to no group: the SHELL group still holds
        # only the sheet, so an unassigned object is a legal reference.
        and listing_c.get("object_count") == 1
        and [o.get("name") for o in listing_c.get("objects") or []] == [SHEET],
        {
            "status": set_c.get("status"),
            "message": set_c.get("message"),
            "reference_object_name": set_c.get("reference_object_name"),
            "reference_object_uuid": ref_uuid,
            "bend_ref_enable": set_c.get("bend_ref_enable"),
            "stored": stored_c,
            "expected_stored": expected_stored_c,
            "reported": reported_c,
            "expected_reported": expected_reported_c,
            "group_members": [
                o.get("name") for o in listing_c.get("objects") or []
            ],
        },
    )

    # ----- D. a reference that names no object in the scene --------
    refuse_d, raw_d = refuse(
        11,
        "set_bend_reference",
        bend_args(shell_uuid, SHEET, "NoSuchReference"),
    )
    message_d = refuse_d.get("message") or ""
    stored_d = rna_reference(shell_uuid, SHEET)
    mcp_check(
        result, "D_unknown_reference_refused_by_name",
        refuse_d.get("status") == "error"
        and raw_d.get("isError") is True
        and "NoSuchReference" in message_d
        and "not found in scene" in message_d
        and stored_d == expected_stored_c,
        {
            "status": refuse_d.get("status"),
            "is_error": raw_d.get("isError"),
            "message": message_d,
            "stored": stored_d,
            "expected_stored": expected_stored_c,
        },
    )

    # ----- E. a target that is not a member of the group -----------
    refuse_e, raw_e = refuse(
        12, "set_bend_reference", bend_args(shell_uuid, BLOCK, REFERENCE)
    )
    message_e = refuse_e.get("message") or ""
    mcp_check(
        result, "E_unassigned_object_refused_by_name",
        refuse_e.get("status") == "error"
        and raw_e.get("isError") is True
        and BLOCK in message_e
        and "not in group" in message_e
        and shell_uuid in message_e,
        {
            "status": refuse_e.get("status"),
            "is_error": raw_e.get("isError"),
            "message": message_e,
            "group_uuid": shell_uuid,
        },
    )

    # ----- F. the topology validator's own message, passed through -
    validator_ok, validator_message = utils.validate_bend_reference(
        sheet, coarse, bpy.context, "SHELL"
    )
    refuse_f, raw_f = refuse(
        13, "set_bend_reference", bend_args(shell_uuid, SHEET, COARSE)
    )
    message_f = refuse_f.get("message") or ""
    stored_f = rna_reference(shell_uuid, SHEET)
    mcp_check(
        result, "F_topology_refusal_passes_message_through",
        validator_ok is False
        and refuse_f.get("status") == "error"
        and raw_f.get("isError") is True
        and message_f == validator_message
        and COARSE in message_f
        and stored_f == expected_stored_c,
        {
            "status": refuse_f.get("status"),
            "is_error": raw_f.get("isError"),
            "message": message_f,
            "validator_ok": validator_ok,
            "validator_message": validator_message,
            "source_vertices": len(sheet.data.vertices),
            "reference_vertices": len(coarse.data.vertices),
            "stored": stored_f,
            "expected_stored": expected_stored_c,
        },
    )

    # ----- G. an empty name clears the reference -------------------
    clear_g, _ = call(14, "set_bend_reference", bend_args(shell_uuid, SHEET, ""))
    stored_g = rna_reference(shell_uuid, SHEET)
    listing_g = group_objects(15, shell_uuid)
    reported_g = reported_reference(listing_g, SHEET)
    expected_reported_g = {
        "enabled": False,
        "object_uuid": "",
        "object_name": None,
        "stored_name": "",
    }
    mcp_check(
        result, "G_empty_name_clears_and_disables",
        clear_g.get("status") == "success"
        and clear_g.get("reference_object_name") == ""
        and clear_g.get("bend_ref_enable") is False
        and stored_g == EMPTY_REFERENCE
        and reported_g == expected_reported_g,
        {
            "status": clear_g.get("status"),
            "message": clear_g.get("message"),
            "reference_object_name": clear_g.get("reference_object_name"),
            "bend_ref_enable": clear_g.get("bend_ref_enable"),
            "stored": stored_g,
            "reported": reported_g,
            "expected_reported": expected_reported_g,
        },
    )

    # ----- H. enable=True with nothing to enable -------------------
    refuse_h, raw_h = refuse(
        16,
        "set_bend_reference",
        bend_args(shell_uuid, SHEET, "", enable=True),
    )
    message_h = refuse_h.get("message") or ""
    stored_h = rna_reference(shell_uuid, SHEET)
    mcp_check(
        result, "H_enable_without_reference_is_refused",
        refuse_h.get("status") == "error"
        and raw_h.get("isError") is True
        and "reference_object_name" in message_h
        and "enable" in message_h
        and stored_h == EMPTY_REFERENCE,
        {
            "status": refuse_h.get("status"),
            "is_error": raw_h.get("isError"),
            "message": message_h,
            "stored": stored_h,
        },
    )

    # ----- I. a reference recorded without being used --------------
    set_i, _ = call(
        17,
        "set_bend_reference",
        bend_args(shell_uuid, SHEET, REFERENCE, enable=False),
    )
    stored_i = rna_reference(shell_uuid, SHEET)
    listing_i = group_objects(18, shell_uuid)
    reported_i = reported_reference(listing_i, SHEET)
    expected_stored_i = {
        "bend_ref_uuid": ref_uuid,
        "bend_ref_name": REFERENCE,
        "bend_ref_enable": False,
    }
    expected_reported_i = {
        "enabled": False,
        "object_uuid": ref_uuid,
        "object_name": REFERENCE,
        "stored_name": REFERENCE,
    }
    mcp_check(
        result, "I_enable_false_records_without_using",
        set_i.get("status") == "success"
        and set_i.get("reference_object_name") == REFERENCE
        and set_i.get("bend_ref_enable") is False
        and stored_i == expected_stored_i
        and reported_i == expected_reported_i,
        {
            "status": set_i.get("status"),
            "message": set_i.get("message"),
            "bend_ref_enable": set_i.get("bend_ref_enable"),
            "stored": stored_i,
            "expected_stored": expected_stored_i,
            "reported": reported_i,
            "expected_reported": expected_reported_i,
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
