# File: scenarios/bl_mcp_bend_reference.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The MCP set_bend_reference surface, against a real Blender.
#
# A bending reference is a topological copy of an assigned object whose
# vertices were moved: the group takes its hinge rest angles from that copy
# instead of from the object's own initial pose. Two records have to line up
# before the scene build reads one. The GROUP's bend_rest_angle_source, a
# material parameter only a SHELL or ROD group offers, must be
# FROM_REFERENCE, and the assigned OBJECT carries the reference itself.
# set_bend_reference writes the object half, and refuses every write that
# would come to rest where nothing reads it, so each gate below is driven
# from both sides.
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
#   A. ``reference_needs_group_source`` -- while the SHELL group's
#      bend_rest_angle_source is FLAT, a reference is refused, and the message
#      names the property, FROM_REFERENCE and set_group_material_properties
#      as the fix. Nothing is stored.
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
#   G. ``empty_name_clears`` -- an empty reference name clears the UUID and
#      the label, which get_group_objects reports as a null object name.
#   H. ``retired_enable_argument_refused`` -- the per-object enable bit is
#      gone (every object of a From Reference Geometry group reads its
#      reference), so an `enable` argument is refused by name and nothing is
#      stored.
#   I. ``retired_group_flag_refused`` -- the old bend_rest_from_reference
#      material property is refused by name, and the reference is then set
#      again for the checks that follow.

# The shape-key source (set_bend_reference_shape_key) is the second reference
# kind: the sheet carries a Basis and a "Curled" shape key, and the tool
# switches the object over to reading a key of its own. The stored reference
# object from I stays stored, unread, which is what the source field says.
#
#   J. ``shape_key_assigned_and_reported`` -- naming "Curled" switches the
#      source to SHAPE_KEY; the call, get_group_objects and
#      the PropertyGroup agree, and the reference object from I is kept.
#   K. ``unknown_shape_key_refused_by_name`` -- a key the sheet does not
#      have is refused naming it, and J's state is untouched.
#   L. ``clearing_object_keeps_shape_key`` -- clearing the reference OBJECT
#      while the shape key is the source leaves the shape key in use.
#   M. ``empty_shape_key_clears`` -- an empty shape key name clears it.
#   N. ``reference_object_switches_source_back`` -- set_bend_reference makes
#      OBJECT the source again.
#   O. ``shape_key_mix_assigned_and_reported`` --
#      set_bend_reference_shape_key_mix makes the whole shape-key mix the
#      source, and get_group_objects reports it.

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

    # The sheet's own shape keys: a flat Basis and a curled key, for the
    # shape-key source (checks J to N).
    sheet.shape_key_add(name="Basis", from_mix=False)
    curled = sheet.shape_key_add(name="Curled", from_mix=False)
    for point in curled.data:
        point.co.z = 0.3 * point.co.x * point.co.x

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
        (props_before.get("properties") or {}).get("bend_rest_angle_source")
        or {}
    ).get("value")

    refuse_a, raw_a = refuse(
        6, "set_bend_reference", bend_args(shell_uuid, SHEET, REFERENCE)
    )
    message_a = refuse_a.get("message") or ""
    stored_a = rna_reference(shell_uuid, SHEET)
    mcp_check(
        result, "A_reference_needs_group_source",
        flag_before == "FLAT"
        and refuse_a.get("status") == "error"
        and raw_a.get("isError") is True
        and "bend_rest_angle_source" in message_a
        and "FROM_REFERENCE" in message_a
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
            "properties": {"bend_rest_angle_source": "FROM_REFERENCE"},
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
    }
    expected_reported_c = {
        "source": "OBJECT",
        "shape_key_name": "",
        "object_uuid": ref_uuid,
        "object_name": REFERENCE,
        "stored_name": REFERENCE,
    }
    mcp_check(
        result, "C_reference_assigned_and_reported",
        set_c.get("status") == "success"
        and bool(ref_uuid)
        and set_c.get("reference_object_name") == REFERENCE
        and set_c.get("bend_ref_source") == "OBJECT"
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
        "source": "OBJECT",
        "shape_key_name": "",
        "object_uuid": "",
        "object_name": None,
        "stored_name": "",
    }
    mcp_check(
        result, "G_empty_name_clears",
        clear_g.get("status") == "success"
        and clear_g.get("reference_object_name") == ""
        and stored_g == EMPTY_REFERENCE
        and reported_g == expected_reported_g,
        {
            "status": clear_g.get("status"),
            "message": clear_g.get("message"),
            "reference_object_name": clear_g.get("reference_object_name"),
            "stored": stored_g,
            "reported": reported_g,
            "expected_reported": expected_reported_g,
        },
    )

    # ----- H. the retired enable argument --------------------------
    refuse_h, raw_h = refuse(
        16,
        "set_bend_reference",
        bend_args(shell_uuid, SHEET, REFERENCE, enable=True),
    )
    message_h = refuse_h.get("message") or ""
    stored_h = rna_reference(shell_uuid, SHEET)
    mcp_check(
        result, "H_retired_enable_argument_refused",
        refuse_h.get("status") == "error"
        and raw_h.get("isError") is True
        and "enable" in message_h
        and stored_h == EMPTY_REFERENCE,
        {
            "status": refuse_h.get("status"),
            "is_error": raw_h.get("isError"),
            "message": message_h,
            "stored": stored_h,
        },
    )

    # ----- I. the retired group flag --------------------------------
    refuse_i, raw_i = refuse(
        17,
        "set_group_material_properties",
        {
            "group_uuid": shell_uuid,
            "properties": {"bend_rest_from_reference": True},
        },
    )
    message_i = refuse_i.get("message") or ""
    mcp_check(
        result, "I_retired_group_flag_refused",
        refuse_i.get("status") == "error"
        and raw_i.get("isError") is True
        and "bend_rest_from_reference" in message_i,
        {
            "status": refuse_i.get("status"),
            "is_error": raw_i.get("isError"),
            "message": message_i,
        },
    )
    reset_i, _ = call(
        18, "set_bend_reference", bend_args(shell_uuid, SHEET, REFERENCE)
    )
    if reset_i.get("status") != "success":
        raise RuntimeError("set_bend_reference again: %r" % (reset_i,))

    def rna_shape_key(group_uuid, object_name):
        group = groups.get_active_group_by_uuid(bpy.context.scene, group_uuid)
        for assigned in group.assigned_objects:
            if assigned.name == object_name:
                return {
                    "bend_ref_source": assigned.bend_ref_source,
                    "bend_ref_shape_key": assigned.bend_ref_shape_key,
                    "bend_ref_uuid": assigned.bend_ref_uuid,
                }
        raise RuntimeError(
            "'%s' is not assigned to group %s" % (object_name, group_uuid)
        )

    def sk_args(group_uuid, object_name, shape_key_name, enable=None):
        arguments = {
            "group_uuid": group_uuid,
            "object_name": object_name,
            "shape_key_name": shape_key_name,
        }
        if enable is not None:
            arguments["enable"] = enable
        return arguments

    # ----- J. a shape key of the object itself ---------------------
    set_j, _ = call(
        19, "set_bend_reference_shape_key", sk_args(shell_uuid, SHEET, "Curled")
    )
    stored_j = rna_shape_key(shell_uuid, SHEET)
    reported_j = reported_reference(group_objects(20, shell_uuid), SHEET)
    expected_stored_j = {
        "bend_ref_source": "SHAPE_KEY",
        "bend_ref_shape_key": "Curled",
        "bend_ref_uuid": ref_uuid,
    }
    expected_reported_j = {
        "source": "SHAPE_KEY",
        "object_uuid": ref_uuid,
        "object_name": REFERENCE,
        "stored_name": REFERENCE,
        "shape_key_name": "Curled",
    }
    mcp_check(
        result, "J_shape_key_assigned_and_reported",
        set_j.get("status") == "success"
        and set_j.get("shape_key_name") == "Curled"
        and set_j.get("bend_ref_source") == "SHAPE_KEY"
        and stored_j == expected_stored_j
        and reported_j == expected_reported_j,
        {
            "reply": set_j,
            "stored": stored_j,
            "reported": reported_j,
            "expected_reported": expected_reported_j,
        },
    )

    # ----- K. a shape key the object does not have -----------------
    refuse_k, raw_k = refuse(
        21, "set_bend_reference_shape_key", sk_args(shell_uuid, SHEET, "NoSuchKey")
    )
    message_k = refuse_k.get("message") or ""
    stored_k = rna_shape_key(shell_uuid, SHEET)
    mcp_check(
        result, "K_unknown_shape_key_refused_by_name",
        refuse_k.get("status") == "error"
        and raw_k.get("isError") is True
        and "NoSuchKey" in message_k
        and SHEET in message_k
        and stored_k == expected_stored_j,
        {"message": message_k, "stored": stored_k},
    )

    # ----- L. clearing the reference object keeps the shape key ----
    clear_l, _ = call(22, "set_bend_reference", bend_args(shell_uuid, SHEET, ""))
    stored_l = rna_shape_key(shell_uuid, SHEET)
    expected_stored_l = dict(expected_stored_j, bend_ref_uuid="")
    mcp_check(
        result, "L_clearing_object_keeps_shape_key",
        clear_l.get("status") == "success"
        and clear_l.get("bend_ref_source") == "SHAPE_KEY"
        and stored_l == expected_stored_l,
        {"reply": clear_l, "stored": stored_l},
    )

    # ----- M. an empty shape key name clears it ---------------------
    clear_m, _ = call(
        23, "set_bend_reference_shape_key", sk_args(shell_uuid, SHEET, "")
    )
    stored_m = rna_shape_key(shell_uuid, SHEET)
    expected_stored_m = {
        "bend_ref_source": "SHAPE_KEY",
        "bend_ref_shape_key": "",
        "bend_ref_uuid": "",
    }
    mcp_check(
        result, "M_empty_shape_key_clears",
        clear_m.get("status") == "success"
        and stored_m == expected_stored_m,
        {"reply": clear_m, "stored": stored_m},
    )

    # ----- N. a reference object switches the source back ----------
    set_n, _ = call(24, "set_bend_reference", bend_args(shell_uuid, SHEET, REFERENCE))
    stored_n = rna_shape_key(shell_uuid, SHEET)
    mcp_check(
        result, "N_reference_object_switches_source_back",
        set_n.get("status") == "success"
        and set_n.get("bend_ref_source") == "OBJECT"
        and stored_n.get("bend_ref_source") == "OBJECT"
        and bool(stored_n.get("bend_ref_uuid")),
        {"reply": set_n, "stored": stored_n},
    )

    # ----- O. the whole shape-key mix as the source -----------------
    set_o, _ = call(
        25, "set_bend_reference_shape_key_mix",
        {"group_uuid": shell_uuid, "object_name": SHEET},
    )
    stored_o = rna_shape_key(shell_uuid, SHEET)
    reported_o = reported_reference(group_objects(26, shell_uuid), SHEET)
    mcp_check(
        result, "O_shape_key_mix_assigned_and_reported",
        set_o.get("status") == "success"
        and set_o.get("bend_ref_source") == "SHAPE_KEY_MIX"
        and stored_o.get("bend_ref_source") == "SHAPE_KEY_MIX"
        and reported_o.get("source") == "SHAPE_KEY_MIX",
        {"reply": set_o, "stored": stored_o, "reported": reported_o},
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
