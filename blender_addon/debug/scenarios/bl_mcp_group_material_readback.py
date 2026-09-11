# File: scenarios/bl_mcp_group_material_readback.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP group read-back and pin list-management surface, against a real
# Blender.
#
# get_group_material_properties is the read half of a pair: the parameters it
# reports are the parameters set_group_material_properties accepts for the
# group's object_type, and the limits it reports are the limits that tool
# holds a write to. This scenario drives both halves against one SHELL group
# so each is measured against the other rather than against a parameter table
# copied into the scenario: the accepted set comes out of the writer's own
# refusal of an unknown name, every reported value is compared with the
# property it is stored in, and a value outside a property's own range is
# read back after the refusal to show the group did not take a clamp of it.
#
# An enum is the one parameter kind whose report can offer something the
# group cannot be asked to run. shell_model carries STABLE_NEOHOOKEAN with an
# explicit item number and an empty UI name, so a .blend holding it still
# loads as that identifier while the panel's picker does not offer it. The
# report lists the offered identifiers only, and the writer refuses the
# withdrawn one, which is what keeps a caller from asking for a model the
# scene build would substitute for.
#
# The pin half covers list order. A pin's position decides which of two pins
# holding one vertex that vertex takes its settings from, so a rename has to
# leave the order alone and a move has to change it by exactly one place.
# Both handlers run a UI operator that acts on the row the group has
# SELECTED, so each writes that row before running it. That row is the
# artist's own state: the rename saves it and puts it back, and the move
# leaves the selection on the pin it moved, which is where the panel's arrow
# buttons leave it. Checks L and M measure both rather than assuming either.
#
# Assertions:
#   A. ``report_covers_the_writable_set`` -- the reported parameter names are
#      exactly the names set_group_material_properties lists as valid for the
#      group's type when it refuses an unknown one.
#   B. ``every_parameter_carries_its_current_value`` -- every entry carries a
#      type, a description, a value and a default, a number carries min and
#      max holding its value, an enum carries options and value_withdrawn,
#      and every reported value equals the property it is read from.
#   C. ``written_value_reads_back`` -- a float, a boolean and an enum written
#      in one call come back from the report, and from the properties
#      themselves, as the values written.
#   D. ``out_of_range_write_is_refused`` -- a friction of 2.5 against a range
#      of 0.0 to 1.0 is refused by name and by range, and the parameter still
#      reads back as the value C wrote rather than as the clamp.
#   E. ``refused_write_applies_nothing`` -- the whole report is identical
#      before and after that refusal, so the valid second parameter the same
#      call carried did not land either.
#   F. ``enum_options_are_the_selectable_items`` -- every reported enum lists
#      exactly the identifiers whose UI name is non-empty, and shell_model's
#      withdrawn identifier is registered on the property yet absent from the
#      options.
#   G. ``withdrawn_enum_identifier_is_refused`` -- writing that identifier is
#      refused, naming the accepted ones and why the property still holds it,
#      and the parameter keeps the value C wrote.
#   H. ``pins_list_in_the_order_they_were_added`` -- the three pins the setup
#      made are reported in that order.
#   I. ``rename_changes_the_listed_name`` -- the pin reports its new name, is
#      listed under it at the same position, and the object's vertex group
#      was renamed with it.
#   J. ``renamed_pin_resolves_under_the_new_name`` -- the new identifier
#      reaches the handler's own no-op refusal while the old one is refused
#      as absent from the group, and neither refusal reorders the list.
#   K. ``rename_onto_a_taken_name_is_refused`` -- refused naming the object
#      and the suffix Blender would append, with the list and the object's
#      vertex groups unchanged.
#   L. ``rename_restores_the_selected_row`` -- renaming a pin other than the
#      selected one leaves the selected row where the artist left it.
#   M. ``move_reorders_the_list`` -- one place per call, reported as a
#      previous and a new index, with the selection following the moved pin,
#      and moving back restores the original order.
#   N. ``move_past_the_end_is_refused`` -- the first pin cannot move up, and
#      the refusal names the edge it is already at.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

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

import ast

OBJ_NAME = "PinnedSheet"
PIN_VERTEX_GROUPS = (("Hem", [0, 1]), ("Collar", [2]), ("Cuff", [3]))
UNKNOWN_PARAM = "no_such_material_parameter"
# Registered on shell_model with an explicit item number and an empty UI
# name, which is what suppresses it from the panel's picker.
WITHDRAWN_MODEL = "STABLE_NEOHOOKEAN"

try:
    groups = __import__(
        pkg + ".models.groups", fromlist=["get_active_group_by_uuid"]
    )

    # ----- scene: one plane, and nothing else ---------------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    sheet = bpy.context.active_object
    sheet.name = OBJ_NAME

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    request_ids = [0]

    def next_id():
        request_ids[0] += 1
        return request_ids[0]

    def call(name, arguments):
        # A tool call whose payload is returned whatever it says: a handler
        # that ran and refused answers with a normal result carrying isError,
        # so a refusal is data here rather than a transport failure.
        payload, raw = mcp_tool(pkg, url, name, arguments, request_id=next_id())
        return payload, raw

    def refusal(name, arguments):
        # The refusal path, driven through mcp_call so the isError flag on the
        # result is available beside the payload.
        envelope, _resp = mcp_call(
            pkg, url, "tools/call",
            {"name": name, "arguments": arguments},
            request_id=next_id(),
        )
        return mcp_tool_payload(envelope), (envelope.get("result") or {})

    def expect(name, arguments):
        # A setup step: anything other than success is a scenario failure
        # rather than an assertion, so it stops the driver where it happened.
        payload, _raw = call(name, arguments)
        if payload.get("status") != "success":
            raise RuntimeError("%s: %r" % (name, payload))
        return payload

    def group_rna():
        # The add-on's own state, so a reported value is measured against the
        # property it is stored in and not only against what the reply said.
        group = groups.get_active_group_by_uuid(bpy.context.scene, group_uuid)
        if group is None:
            raise RuntimeError("group %s is not active" % group_uuid)
        return group

    def rna_value(name):
        # The stored value in the shape the report uses for that property
        # type, so the two are comparable without a tolerance.
        group = group_rna()
        prop = group.bl_rna.properties[name]
        value = getattr(group, name)
        if prop.type == "BOOLEAN":
            return bool(value)
        if prop.type == "ENUM":
            return str(value)
        if prop.type == "INT":
            return int(value)
        return float(value)

    def material_report():
        return expect(
            "get_group_material_properties", {"group_uuid": group_uuid}
        )

    def ident(vg_name):
        return "%s::%s" % (OBJ_NAME, vg_name)

    def pin_order():
        listing = expect("list_pins", {"group_uuid": group_uuid})
        return [
            pin.get("vertex_group_identifier") for pin in listing.get("pins") or []
        ]

    def vertex_group_names():
        return sorted(vg.name for vg in bpy.data.objects[OBJ_NAME].vertex_groups)

    # ----- a SHELL group holding the plane, with three pins --------
    created = expect("create_group", {"name": "Readback", "type": "SHELL"})
    group_uuid = created.get("group_uuid") or ""
    if not group_uuid:
        raise RuntimeError("create_group returned no uuid: %r" % (created,))
    expect(
        "add_objects_to_group",
        {"group_uuid": group_uuid, "object_names": [OBJ_NAME]},
    )
    for vg_name, indices in PIN_VERTEX_GROUPS:
        expect(
            "create_vertex_group",
            {"object_name": OBJ_NAME, "name": vg_name, "indices": indices},
        )
        expect(
            "add_pin_vertex_group",
            {
                "group_uuid": group_uuid,
                "vertex_group_identifier": ident(vg_name),
            },
        )

    # ----- A. the report covers exactly what the writer accepts ----
    report_a = material_report()
    reported_names = sorted(report_a.get("properties") or {})
    refuse_a, raw_a = refusal(
        "set_group_material_properties",
        {"group_uuid": group_uuid, "properties": {UNKNOWN_PARAM: 1.0}},
    )
    message_a = refuse_a.get("message") or ""
    marker = "Valid properties: "
    writer_names = []
    parse_error = None
    if marker in message_a:
        try:
            writer_names = sorted(ast.literal_eval(message_a.split(marker, 1)[1]))
        except Exception as exc:
            parse_error = "%s: %s" % (type(exc).__name__, exc)
    mcp_check(
        result, "A_report_covers_the_writable_set",
        report_a.get("group_type") == "SHELL"
        and report_a.get("property_count") == len(reported_names)
        and refuse_a.get("status") == "error"
        and raw_a.get("isError") is True
        and UNKNOWN_PARAM in message_a
        and bool(writer_names)
        and reported_names == writer_names,
        {
            "group_type": report_a.get("group_type"),
            "property_count": report_a.get("property_count"),
            "reported_count": len(reported_names),
            "reported_only": [n for n in reported_names if n not in writer_names],
            "writer_only": [n for n in writer_names if n not in reported_names],
            "refusal_is_error": raw_a.get("isError"),
            "refusal_message": message_a,
            "parse_error": parse_error,
        },
    )

    # ----- B. every entry, against the property it comes from ------
    entries_b = report_a.get("properties") or {}
    shape_faults = []
    value_faults = []
    for name in sorted(entries_b):
        entry = entries_b[name]
        kind = entry.get("type")
        description = entry.get("description")
        if not isinstance(description, str) or not description:
            shape_faults.append("%s: description is %r" % (name, description))
        if "value" not in entry or "default" not in entry:
            shape_faults.append("%s: keys are %s" % (name, sorted(entry)))
        elif kind in ("float", "int"):
            if "min" not in entry or "max" not in entry:
                shape_faults.append("%s: no min/max, keys %s" % (name, sorted(entry)))
            elif not entry["min"] <= entry["value"] <= entry["max"]:
                shape_faults.append(
                    "%s: value %r outside its own %r to %r"
                    % (name, entry["value"], entry["min"], entry["max"])
                )
        elif kind == "enum":
            if not isinstance(entry.get("options"), list):
                shape_faults.append("%s: options are %r" % (name, entry.get("options")))
            if not isinstance(entry.get("value_withdrawn"), bool):
                shape_faults.append(
                    "%s: value_withdrawn is %r" % (name, entry.get("value_withdrawn"))
                )
        elif kind == "boolean":
            if not isinstance(entry.get("value"), bool):
                shape_faults.append("%s: value is %r" % (name, entry.get("value")))
        else:
            shape_faults.append("%s: unexpected type %r" % (name, kind))
        stored = rna_value(name)
        if entry.get("value") != stored:
            value_faults.append(
                "%s: reported %r, stored %r" % (name, entry.get("value"), stored)
            )
    mcp_check(
        result, "B_every_parameter_carries_its_current_value",
        len(entries_b) > 0 and not shape_faults and not value_faults,
        {
            "entry_count": len(entries_b),
            "kinds": sorted({e.get("type") for e in entries_b.values()}),
            "shape_faults": shape_faults,
            "value_faults": value_faults,
            "friction": entries_b.get("friction"),
            "shell_model": entries_b.get("shell_model"),
        },
    )

    # ----- C. a write, read back through the report ----------------
    # Exactly representable in a C float, so the value written is the value
    # the property holds and the comparison needs no tolerance.
    written_c = {"friction": 0.375, "enable_inflate": True, "shell_model": "ARAP"}
    set_c, _raw_c = call(
        "set_group_material_properties",
        {"group_uuid": group_uuid, "properties": dict(written_c)},
    )
    report_c = material_report()
    entries_c = report_c.get("properties") or {}
    reported_c = {k: (entries_c.get(k) or {}).get("value") for k in written_c}
    stored_c = {k: rna_value(k) for k in written_c}
    mcp_check(
        result, "C_written_value_reads_back",
        set_c.get("status") == "success"
        and sorted(set_c.get("properties_set") or []) == sorted(written_c)
        and reported_c == written_c
        and stored_c == written_c,
        {
            "set_status": set_c.get("status"),
            "set_message": set_c.get("message"),
            "properties_set": sorted(set_c.get("properties_set") or []),
            "updates": set_c.get("updates"),
            "written": written_c,
            "reported": reported_c,
            "stored": stored_c,
        },
    )

    # ----- D. a value the property's own range excludes ------------
    # The call carries a second, valid parameter, which check E reads back.
    over_max = 2.5
    refuse_d, raw_d = refusal(
        "set_group_material_properties",
        {
            "group_uuid": group_uuid,
            "properties": {"friction": over_max, "bend": 12.0},
        },
    )
    message_d = refuse_d.get("message") or ""
    report_d = material_report()
    entries_d = report_d.get("properties") or {}
    friction_d = entries_d.get("friction") or {}
    mcp_check(
        result, "D_out_of_range_write_is_refused",
        refuse_d.get("status") == "error"
        and raw_d.get("isError") is True
        and "'friction'" in message_d
        and "0.0 to 1.0" in message_d
        and "2.5 is outside it" in message_d
        # Refused, so the property holds what C wrote and not the clamp the
        # range would have produced.
        and friction_d.get("value") == written_c["friction"]
        and rna_value("friction") == written_c["friction"]
        and friction_d.get("min") == 0.0
        and friction_d.get("max") == 1.0,
        {
            "status": refuse_d.get("status"),
            "is_error": raw_d.get("isError"),
            "message": message_d,
            "requested": over_max,
            "reported_after": friction_d.get("value"),
            "stored_after": rna_value("friction"),
            "clamp_would_be": friction_d.get("max"),
        },
    )

    # ----- E. and it applied none of the call ---------------------
    changed_e = sorted(
        name
        for name in set(entries_c) | set(entries_d)
        if entries_c.get(name) != entries_d.get(name)
    )
    mcp_check(
        result, "E_refused_write_applies_nothing",
        entries_d == entries_c,
        {
            "changed": changed_e,
            "before": {name: entries_c.get(name) for name in changed_e},
            "after": {name: entries_d.get(name) for name in changed_e},
            "bend_before": (entries_c.get("bend") or {}).get("value"),
            "bend_after": (entries_d.get("bend") or {}).get("value"),
        },
    )

    # ----- F. an enum offers only what its picker offers -----------
    enum_faults = []
    enum_names = []
    for name in sorted(entries_d):
        entry = entries_d[name]
        if entry.get("type") != "enum":
            continue
        enum_names.append(name)
        prop = group_rna().bl_rna.properties[name]
        selectable = [item.identifier for item in prop.enum_items if item.name]
        if entry.get("options") != selectable:
            enum_faults.append(
                "%s: options %r, selectable %r"
                % (name, entry.get("options"), selectable)
            )
        hidden = [item.identifier for item in prop.enum_items if not item.name]
        offered_hidden = [
            identifier
            for identifier in hidden
            if identifier in (entry.get("options") or [])
        ]
        if offered_hidden:
            enum_faults.append("%s: offers hidden %r" % (name, offered_hidden))
    shell_entry = entries_d.get("shell_model") or {}
    shell_prop = group_rna().bl_rna.properties["shell_model"]
    registered = [item.identifier for item in shell_prop.enum_items]
    mcp_check(
        result, "F_enum_options_are_the_selectable_items",
        bool(enum_names)
        and not enum_faults
        and WITHDRAWN_MODEL in registered
        and WITHDRAWN_MODEL not in (shell_entry.get("options") or [])
        and shell_entry.get("value") == "ARAP"
        and shell_entry.get("value_withdrawn") is False,
        {
            "enum_parameters": enum_names,
            "enum_faults": enum_faults,
            "shell_model_registered": registered,
            "shell_model_options": shell_entry.get("options"),
            "shell_model_value": shell_entry.get("value"),
            "value_withdrawn": shell_entry.get("value_withdrawn"),
        },
    )

    # ----- G. and refuses the identifier it does not offer ---------
    refuse_g, raw_g = refusal(
        "set_group_material_properties",
        {"group_uuid": group_uuid, "properties": {"shell_model": WITHDRAWN_MODEL}},
    )
    message_g = refuse_g.get("message") or ""
    mcp_check(
        result, "G_withdrawn_enum_identifier_is_refused",
        refuse_g.get("status") == "error"
        and raw_g.get("isError") is True
        and "'shell_model'" in message_g
        and "'%s' is not one of them" % WITHDRAWN_MODEL in message_g
        and "still stores that identifier" in message_g
        and "ARAP" in message_g
        and rna_value("shell_model") == "ARAP",
        {
            "status": refuse_g.get("status"),
            "is_error": raw_g.get("isError"),
            "message": message_g,
            "stored_after": rna_value("shell_model"),
        },
    )

    # ----- H. the pin list the setup built -------------------------
    listing_h = expect("list_pins", {"group_uuid": group_uuid})
    order_h = [
        pin.get("vertex_group_identifier") for pin in listing_h.get("pins") or []
    ]
    expected_h = [ident(name) for name, _indices in PIN_VERTEX_GROUPS]
    mcp_check(
        result, "H_pins_list_in_the_order_they_were_added",
        listing_h.get("pin_count") == len(expected_h)
        and order_h == expected_h
        and all(
            pin.get("object_name") == OBJ_NAME and bool(pin.get("object_uuid"))
            for pin in listing_h.get("pins") or []
        ),
        {
            "pin_count": listing_h.get("pin_count"),
            "order": order_h,
            "expected": expected_h,
            "selected_row": group_rna().pin_vertex_groups_index,
        },
    )

    # ----- I. a rename, in the list and on the object --------------
    renamed_i, _raw_i = call(
        "rename_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("Collar"),
            "new_name": "CollarInner",
        },
    )
    order_i = pin_order()
    names_i = vertex_group_names()
    mcp_check(
        result, "I_rename_changes_the_listed_name",
        renamed_i.get("status") == "success"
        and renamed_i.get("previous_vertex_group_name") == "Collar"
        and renamed_i.get("vertex_group_name") == "CollarInner"
        and renamed_i.get("vertex_group_identifier") == ident("CollarInner")
        and order_i == [ident("Hem"), ident("CollarInner"), ident("Cuff")]
        # The pin entry and the membership are written separately, so the
        # object has to answer to the new name as well, and only to it.
        and "CollarInner" in names_i
        and "Collar" not in names_i,
        {
            "status": renamed_i.get("status"),
            "message": renamed_i.get("message"),
            "reported": {
                "previous": renamed_i.get("previous_vertex_group_name"),
                "current": renamed_i.get("vertex_group_name"),
                "identifier": renamed_i.get("vertex_group_identifier"),
            },
            "order": order_i,
            "vertex_groups": names_i,
        },
    )

    # ----- J. which identifier the pin now answers to --------------
    stale_j, raw_stale_j = refusal(
        "rename_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("Collar"),
            "new_name": "Reachable",
        },
    )
    # A no-op rename is refused by the handler itself, which is reached only
    # after the identifier resolved to a pin in the list.
    noop_j, raw_noop_j = refusal(
        "rename_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("CollarInner"),
            "new_name": "CollarInner",
        },
    )
    message_stale_j = stale_j.get("message") or ""
    message_noop_j = noop_j.get("message") or ""
    order_j = pin_order()
    mcp_check(
        result, "J_renamed_pin_resolves_under_the_new_name",
        stale_j.get("status") == "error"
        and raw_stale_j.get("isError") is True
        and "not found in group" in message_stale_j
        and ident("Collar") in message_stale_j
        and noop_j.get("status") == "error"
        and raw_noop_j.get("isError") is True
        and "already carries that name" in message_noop_j
        and ident("CollarInner") in message_noop_j
        and order_j == order_i
        and vertex_group_names() == names_i,
        {
            "stale_status": stale_j.get("status"),
            "stale_message": message_stale_j,
            "resolved_status": noop_j.get("status"),
            "resolved_message": message_noop_j,
            "order": order_j,
            "vertex_groups": vertex_group_names(),
        },
    )

    # ----- K. a name the object already carries --------------------
    taken_k, raw_k = refusal(
        "rename_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("Hem"),
            "new_name": "Cuff",
        },
    )
    message_k = taken_k.get("message") or ""
    order_k = pin_order()
    names_k = vertex_group_names()
    mcp_check(
        result, "K_rename_onto_a_taken_name_is_refused",
        taken_k.get("status") == "error"
        and raw_k.get("isError") is True
        and "already has a vertex group named" in message_k
        and "'Cuff'" in message_k
        and "numbered suffix" in message_k
        and order_k == order_i
        # Refused before Blender was asked, so no suffixed group was made.
        and names_k == names_i,
        {
            "status": taken_k.get("status"),
            "is_error": raw_k.get("isError"),
            "message": message_k,
            "order": order_k,
            "vertex_groups": names_k,
        },
    )

    # ----- L. the artist's selected row, across a rename -----------
    # Row 0 stands for a row the artist selected; the rename below is of the
    # pin at row 2, which the handler selects for the operator and then puts
    # the saved row back.
    group_rna().pin_vertex_groups_index = 0
    selected_before_l = int(group_rna().pin_vertex_groups_index)
    renamed_l, _raw_l = call(
        "rename_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("Cuff"),
            "new_name": "CuffOuter",
        },
    )
    selected_after_l = int(group_rna().pin_vertex_groups_index)
    order_l = pin_order()
    mcp_check(
        result, "L_rename_restores_the_selected_row",
        renamed_l.get("status") == "success"
        and selected_before_l == 0
        and selected_after_l == selected_before_l
        and order_l == [ident("Hem"), ident("CollarInner"), ident("CuffOuter")],
        {
            "renamed_row": 2,
            "selected_before": selected_before_l,
            "selected_after": selected_after_l,
            "restored": selected_after_l == selected_before_l,
            "order": order_l,
        },
    )

    # ----- M. one place per call, and back again -------------------
    moved_m, _raw_m = call(
        "move_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("Hem"),
            "direction": "DOWN",
        },
    )
    order_m = pin_order()
    selected_m = int(group_rna().pin_vertex_groups_index)
    back_m, _raw_back_m = call(
        "move_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("Hem"),
            "direction": "UP",
        },
    )
    order_back_m = pin_order()
    mcp_check(
        result, "M_move_reorders_the_list",
        moved_m.get("status") == "success"
        and moved_m.get("previous_index") == 0
        and moved_m.get("index") == 1
        and moved_m.get("pin_order") == order_m
        and order_m == [ident("CollarInner"), ident("Hem"), ident("CuffOuter")]
        and back_m.get("status") == "success"
        and back_m.get("previous_index") == 1
        and back_m.get("index") == 0
        and order_back_m == order_l,
        {
            "down": {
                "status": moved_m.get("status"),
                "previous_index": moved_m.get("previous_index"),
                "index": moved_m.get("index"),
                "pin_order": moved_m.get("pin_order"),
            },
            "order_after_down": order_m,
            # The move leaves the selection on the pin it moved, which is
            # where the panel's arrow buttons leave it.
            "selected_after_down": selected_m,
            "up": {
                "status": back_m.get("status"),
                "previous_index": back_m.get("previous_index"),
                "index": back_m.get("index"),
            },
            "order_after_up": order_back_m,
        },
    )

    # ----- N. the edge of the list ---------------------------------
    edge_n, raw_n = refusal(
        "move_pin_vertex_group",
        {
            "group_uuid": group_uuid,
            "vertex_group_identifier": ident("Hem"),
            "direction": "UP",
        },
    )
    message_n = edge_n.get("message") or ""
    order_n = pin_order()
    mcp_check(
        result, "N_move_past_the_end_is_refused",
        edge_n.get("status") == "error"
        and raw_n.get("isError") is True
        and "already the first" in message_n
        and "cannot move UP" in message_n
        and order_n == order_back_m,
        {
            "status": edge_n.get("status"),
            "is_error": raw_n.get("isError"),
            "message": message_n,
            "order": order_n,
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
