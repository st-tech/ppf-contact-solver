# File: scenarios/bl_mcp_ordered_collections.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# In-place editing and reordering of the ordered per-object collections,
# against a real Blender.
#
# Four collections on this surface carry meaning in their ORDER. Pin
# operations and static ops are shipped to the solver in list order and
# compose in that order; velocity keyframes are held in frame order; a
# collision window is addressed by the index its listing reports. An add puts
# an op at the HEAD of its list, so changing one field by removing the entry
# and adding it back moves that entry to the front and rewrites the motion.
# set_pin_operation, set_static_op, set_velocity_keyframe and
# set_collision_window exist to edit an entry where it sits, and
# move_pin_operation / move_static_op are the only way to change the order on
# purpose. The two halves are what this scenario measures against each other:
# an edit must leave the order alone, and a move must be what changes it.
#
# Order and edited field are kept independent so neither assertion can pass by
# accident: each list holds one MOVE_BY, one SPIN and one SCALE, so the order
# is read off the op types, while every edit names a field no other entry
# carries. A listing taken before and after each call is compared field by
# field, so "changed only the named field" is measured over every field of
# every entry rather than over the one the call reported.
#
# A refused call is measured the same way. Each refusal here also carries a
# second, valid field, and the listing is compared again afterwards: the
# handlers check the whole set before writing any of it, so a refused call
# has to leave the entry holding exactly what it held.
#
# Assertions:
#   A. ``pin_ops_add_at_the_head`` -- three adds report index 0 and a growing
#      count, and the listing comes back in the reverse of the add order,
#      which is the property the in-place edit exists to protect.
#   B. ``pin_edit_keeps_order_and_one_field`` -- editing the MIDDLE entry
#      moves nothing and changes exactly one field of exactly that entry.
#   C. ``pin_move_reorders_the_listing`` -- a move rotates the entries as
#      asked, reports the moved entry's new index and the whole new order,
#      and carries each entry's fields with it.
#   D. ``pin_out_of_range_index_is_refused`` -- an index past the end, a
#      negative one, and either half of a move are each refused by the name
#      of the argument at fault, and nothing moves.
#   E. ``pin_call_naming_no_field_is_refused`` -- a call that names no field
#      is refused, naming the entry's op type and the remove handler, rather
#      than reported as an edit that changed nothing.
#   F. ``pin_foreign_field_is_refused`` -- a field belonging to another op
#      type is refused by name, and the valid field in the same call is
#      dropped with it.
#   G. ``static_ops_add_at_the_head`` -- the static-op list behaves the same
#      way.
#   H. ``static_edit_keeps_order_and_one_field``
#   I. ``static_move_reorders_the_listing``
#   J. ``static_out_of_range_index_is_refused``
#   K. ``static_call_naming_no_field_is_refused``
#   L. ``static_foreign_field_is_refused``
#   M. ``velocity_edit_keeps_frame_and_one_field`` -- editing a keyframe's
#      speed retimes nothing: the index it reports back is the one it was
#      given, and the stored frames are untouched.
#   N. ``velocity_retime_onto_taken_frame_is_refused`` -- a frame another
#      keyframe already holds is refused by number and by object, and the
#      speed the same call carried is dropped with it.
#   O. ``velocity_retime_reports_the_new_index`` -- a retime that reorders
#      the list reports where the entry landed, and the entry that moved is
#      the one that carries its own speed to the new position.
#   P. ``window_edit_keeps_its_index`` -- a window edited in place stays at
#      the index its listing reports, with the other windows untouched.
#   Q. ``window_inverted_by_both_bounds_is_refused`` -- a call whose two
#      bounds cross is refused, quoting the window it would have left.
#   R. ``window_inverted_by_one_bound_is_refused`` -- and so is one bound
#      moved past the bound already stored, from either side, which is the
#      case a per-argument check would miss.

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

SHEET_NAME = "OrderedSheet"
BLOCK_NAME = "OrderedBlock"
VG_NAME = "PinRow"
PIN_ID = SHEET_NAME + "::" + VG_NAME
PIN_INDICES = [6, 7, 8]

_next_id = [100]


def new_id():
    _next_id[0] += 1
    return _next_id[0]


def tool(name, arguments):
    # One tools/call, whatever its outcome. A handler that ran and refused
    # answers with an ordinary result carrying isError and a payload whose
    # status is "error", so a refusal is returned here rather than raised;
    # only a transport or dispatch failure raises.
    envelope, resp = mcp_call(
        pkg, url, "tools/call",
        {"name": name, "arguments": arguments},
        request_id=new_id(),
    )
    if "result" not in envelope:
        raise RuntimeError(
            "%s: no result (status=%s error=%r)"
            % (name, resp.get("status"), envelope.get("error"))
        )
    return mcp_tool_payload(envelope), envelope["result"]


def must(name, arguments):
    # A call whose failure is a broken scenario rather than an assertion:
    # it stops the driver instead of being recorded as a failed check.
    payload, _ = tool(name, arguments)
    if payload.get("status") != "success":
        raise RuntimeError("%s refused: %r" % (name, payload))
    return payload


def refused(name, arguments):
    # The three things a refusal is measured by, in one record.
    payload, raw = tool(name, arguments)
    return {
        "status": payload.get("status"),
        "is_error": raw.get("isError"),
        "message": payload.get("message") or "",
    }


def is_refused(record, *needles):
    return (
        record["status"] == "error"
        and record["is_error"] is True
        and all(needle in record["message"] for needle in needles)
    )


def entry_diff(before, after):
    # Every field of every entry that differs between two listings, as
    # [index, field, old, new]. An empty list is the assertion that nothing
    # moved and nothing changed; a length change is reported on its own so a
    # dropped entry cannot read as an unchanged list.
    if len(before) != len(after):
        return [["length", len(before), len(after)]]
    diffs = []
    for index in range(len(before)):
        old, new = before[index], after[index]
        for field in sorted(set(old) | set(new)):
            if old.get(field) != new.get(field):
                diffs.append([index, field, old.get(field), new.get(field)])
    return diffs


def op_types(entries):
    return [entry.get("op_type") for entry in entries]


def build_grid(name):
    # A 3 by 3 vertex sheet of four quads, built from the datablock API so
    # the vertex indices the pin names are the ones from_pydata assigned.
    coords = [
        (col * 0.5, row * 0.5, 0.0) for row in range(3) for col in range(3)
    ]
    faces = [
        (row * 3 + col, row * 3 + col + 1, row * 3 + col + 4, row * 3 + col + 3)
        for row in range(2)
        for col in range(2)
    ]
    return build_mesh(name, coords, faces)


def build_cube(name, height):
    coords = [
        (-0.5, -0.5, height - 0.5), (0.5, -0.5, height - 0.5),
        (0.5, 0.5, height - 0.5), (-0.5, 0.5, height - 0.5),
        (-0.5, -0.5, height + 0.5), (0.5, -0.5, height + 0.5),
        (0.5, 0.5, height + 0.5), (-0.5, 0.5, height + 0.5),
    ]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    return build_mesh(name, coords, faces)


def build_mesh(name, coords, faces):
    mesh = bpy.data.meshes.new(name + "Data")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    return obj


try:
    groups = __import__(
        pkg + ".models.groups", fromlist=["get_active_group_by_uuid"]
    )

    # ----- scene: one sheet to pin, one block to move --------------
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing, do_unlink=True)
    build_grid(SHEET_NAME)
    build_cube(BLOCK_NAME, 3.0)
    result["phases"].append((time.time(), "scene_built"))

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    shell_uuid = must(
        "create_group", {"name": "Ordered", "type": "SHELL"}
    ).get("group_uuid") or ""
    must(
        "add_objects_to_group",
        {"group_uuid": shell_uuid, "object_names": [SHEET_NAME]},
    )
    must(
        "create_vertex_group",
        {"object_name": SHEET_NAME, "name": VG_NAME, "indices": PIN_INDICES},
    )
    must(
        "add_pin_vertex_group",
        {"group_uuid": shell_uuid, "vertex_group_identifier": PIN_ID},
    )

    static_uuid = must(
        "create_group", {"name": "Colliders", "type": "STATIC"}
    ).get("group_uuid") or ""
    must(
        "add_objects_to_group",
        {"group_uuid": static_uuid, "object_names": [BLOCK_NAME]},
    )
    if not shell_uuid or not static_uuid:
        raise RuntimeError(
            "group setup: shell=%r static=%r" % (shell_uuid, static_uuid)
        )
    result["phases"].append((time.time(), "groups_ready"))

    def pin_args(arguments):
        arguments = dict(arguments)
        arguments["group_uuid"] = shell_uuid
        arguments["vertex_group_identifier"] = PIN_ID
        return arguments

    def pin_ops():
        return must(
            "list_pin_operations",
            {"group_uuid": shell_uuid, "vertex_group_identifier": PIN_ID},
        ).get("operations") or []

    def pin_tool(name, arguments):
        return must(name, pin_args(arguments))

    def pin_refused(name, arguments):
        return refused(name, pin_args(arguments))

    def static_args(arguments):
        arguments = dict(arguments)
        arguments["group_uuid"] = static_uuid
        arguments["object_name"] = BLOCK_NAME
        return arguments

    def static_ops():
        return must(
            "list_static_ops",
            {"group_uuid": static_uuid, "object_name": BLOCK_NAME},
        ).get("static_ops") or []

    def static_tool(name, arguments):
        return must(name, static_args(arguments))

    def static_refused(name, arguments):
        return refused(name, static_args(arguments))

    def sheet_args(arguments):
        arguments = dict(arguments)
        arguments["group_uuid"] = shell_uuid
        arguments["object_name"] = SHEET_NAME
        return arguments

    def velocity_rna():
        # The add-on's own state. list_velocity_keyframes reports frame,
        # direction and speed only, so the fields an edit must leave alone
        # are read from the PropertyGroup the handler writes.
        group = groups.get_active_group_by_uuid(bpy.context.scene, shell_uuid)
        if group is None:
            raise RuntimeError("group %s is not active" % shell_uuid)
        for assigned in group.assigned_objects:
            if assigned.name != SHEET_NAME:
                continue
            return [
                {
                    "frame": int(kf.frame),
                    "direction": [float(c) for c in kf.direction],
                    "speed": float(kf.speed),
                    "angular_axis": str(kf.angular_axis),
                    "angular_speed": float(kf.angular_speed),
                    "angular_axis_custom": [
                        float(c) for c in kf.angular_axis_custom
                    ],
                    "enable_translational": bool(kf.enable_translational),
                    "enable_angular": bool(kf.enable_angular),
                }
                for kf in assigned.velocity_keyframes
            ]
        raise RuntimeError(
            "'%s' is not assigned to group %s" % (SHEET_NAME, shell_uuid)
        )

    def velocity_listed():
        return must(
            "list_velocity_keyframes",
            {"group_uuid": shell_uuid, "object_name": SHEET_NAME},
        ).get("keyframes") or []

    def windows():
        return must(
            "list_collision_windows",
            {"group_uuid": shell_uuid, "object_name": SHEET_NAME},
        ).get("windows") or []

    def window_pairs(entries):
        return [[e.get("frame_start"), e.get("frame_end")] for e in entries]

    # ================= pin operations =============================
    # Added oldest first, so the head-insertion the add performs leaves the
    # listing in the reverse of this order.
    pin_add_1 = pin_tool(
        "add_pin_operation",
        {
            "op_type": "MOVE_BY", "frame_start": 1, "frame_end": 10,
            "delta": [1.0, 0.0, 0.0],
        },
    )
    pin_add_2 = pin_tool(
        "add_pin_operation",
        {
            "op_type": "SPIN", "frame_start": 11, "frame_end": 20,
            "spin_axis": [0.0, 0.0, 1.0], "spin_angular_velocity": 90.0,
        },
    )
    pin_add_3 = pin_tool(
        "add_pin_operation",
        {
            "op_type": "SCALE", "frame_start": 21, "frame_end": 30,
            "scale_factor": 2.0,
        },
    )
    ops_a = pin_ops()

    # ----- A. an add goes to the head of the list -----------------
    mcp_check(
        result, "A_pin_ops_add_at_the_head",
        [add.get("operation_index") for add in (pin_add_1, pin_add_2, pin_add_3)]
        == [0, 0, 0]
        and [add.get("operation_count") for add in (pin_add_1, pin_add_2, pin_add_3)]
        == [1, 2, 3]
        and op_types(ops_a) == ["SCALE", "SPIN", "MOVE_BY"]
        and [entry.get("frame_start") for entry in ops_a] == [21, 11, 1]
        and [entry.get("frame_end") for entry in ops_a] == [30, 20, 10]
        and ops_a[0].get("scale_factor") == 2.0
        and ops_a[1].get("spin_angular_velocity") == 90.0
        and ops_a[2].get("delta") == [1.0, 0.0, 0.0],
        {
            "add_indices": [
                add.get("operation_index")
                for add in (pin_add_1, pin_add_2, pin_add_3)
            ],
            "add_counts": [
                add.get("operation_count")
                for add in (pin_add_1, pin_add_2, pin_add_3)
            ],
            "listed": ops_a,
        },
    )

    # ----- B. editing the middle entry ----------------------------
    edit_b, _ = tool(
        "set_pin_operation",
        pin_args({"index": 1, "spin_angular_velocity": 45.0}),
    )
    ops_b = pin_ops()
    diff_b = entry_diff(ops_a, ops_b)
    mcp_check(
        result, "B_pin_edit_keeps_order_and_one_field",
        edit_b.get("status") == "success"
        and edit_b.get("index") == 1
        and edit_b.get("op_type") == "SPIN"
        and edit_b.get("updates") == {"spin_angular_velocity": 45.0}
        and edit_b.get("operation_count") == 3
        and op_types(ops_b) == ["SCALE", "SPIN", "MOVE_BY"]
        and diff_b == [[1, "spin_angular_velocity", 90.0, 45.0]],
        {
            "reply": edit_b,
            "order": op_types(ops_b),
            "diff": diff_b,
            "listed": ops_b,
        },
    )

    # ----- C. a move is what changes the order --------------------
    move_c = pin_tool("move_pin_operation", {"index": 0, "new_index": 2})
    ops_c = pin_ops()
    mcp_check(
        result, "C_pin_move_reorders_the_listing",
        move_c.get("index") == 2
        and move_c.get("op_type") == "SCALE"
        and move_c.get("operation_order") == ["SPIN", "MOVE_BY", "SCALE"]
        and op_types(ops_c) == ["SPIN", "MOVE_BY", "SCALE"]
        and ops_c == [ops_b[1], ops_b[2], ops_b[0]],
        {
            "reply": move_c,
            "order_before": op_types(ops_b),
            "order_after": op_types(ops_c),
            "carried_fields": ops_c == [ops_b[1], ops_b[2], ops_b[0]],
            "listed": ops_c,
        },
    )

    # ----- D. an index that addresses no entry --------------------
    pin_bad = {
        "set_past_end": pin_refused(
            "set_pin_operation", {"index": 3, "frame_end": 40}
        ),
        "set_negative": pin_refused(
            "set_pin_operation", {"index": -1, "frame_end": 40}
        ),
        "move_index": pin_refused(
            "move_pin_operation", {"index": 9, "new_index": 0}
        ),
        "move_new_index": pin_refused(
            "move_pin_operation", {"index": 0, "new_index": 3}
        ),
    }
    ops_d = pin_ops()
    mcp_check(
        result, "D_pin_out_of_range_index_is_refused",
        is_refused(pin_bad["set_past_end"], "Index 3 out of range", "0..2")
        and is_refused(pin_bad["set_negative"], "Index -1 out of range", "0..2")
        and is_refused(pin_bad["move_index"], "index 9 out of range", "0..2")
        and is_refused(
            pin_bad["move_new_index"], "new_index 3 out of range", "0..2"
        )
        and entry_diff(ops_c, ops_d) == [],
        {"refusals": pin_bad, "diff": entry_diff(ops_c, ops_d)},
    )

    # ----- E. a call that names no field at all -------------------
    refuse_e = pin_refused("set_pin_operation", {"index": 0})
    ops_e = pin_ops()
    mcp_check(
        result, "E_pin_call_naming_no_field_is_refused",
        is_refused(
            refuse_e, "No field named", "op_type=SPIN", "remove_pin_operation"
        )
        and entry_diff(ops_c, ops_e) == [],
        {"refusal": refuse_e, "diff": entry_diff(ops_c, ops_e)},
    )

    # ----- F. a field belonging to another op type ----------------
    # The call also carries a field the entry does hold, so the empty diff
    # says the refusal dropped the whole call rather than half of it.
    refuse_f = pin_refused(
        "set_pin_operation",
        {"index": 1, "delta": [9.0, 9.0, 9.0], "scale_factor": 3.0},
    )
    ops_f = pin_ops()
    mcp_check(
        result, "F_pin_foreign_field_is_refused",
        is_refused(
            refuse_f,
            "Field 'scale_factor' is not valid for op_type=MOVE_BY",
            "allowed:",
            "'delta'",
        )
        and entry_diff(ops_c, ops_f) == [],
        {
            "refusal": refuse_f,
            "diff": entry_diff(ops_c, ops_f),
            "listed": ops_f,
        },
    )

    # ================= static ops =================================
    static_add_1 = static_tool(
        "add_static_op",
        {
            "op_type": "MOVE_BY", "frame_start": 1, "frame_end": 10,
            "delta": [1.0, 0.0, 0.0],
        },
    )
    static_add_2 = static_tool(
        "add_static_op",
        {
            "op_type": "SPIN", "frame_start": 11, "frame_end": 20,
            "spin_axis": [0.0, 0.0, 1.0], "spin_angular_velocity": 90.0,
        },
    )
    static_add_3 = static_tool(
        "add_static_op",
        {
            "op_type": "SCALE", "frame_start": 21, "frame_end": 30,
            "scale_factor": 2.0,
        },
    )
    ops_g = static_ops()

    # ----- G. the same head insertion -----------------------------
    mcp_check(
        result, "G_static_ops_add_at_the_head",
        [
            add.get("operation_count")
            for add in (static_add_1, static_add_2, static_add_3)
        ] == [1, 2, 3]
        and op_types(ops_g) == ["SCALE", "SPIN", "MOVE_BY"]
        and [entry.get("frame_start") for entry in ops_g] == [21, 11, 1]
        and [entry.get("frame_end") for entry in ops_g] == [30, 20, 10]
        and ops_g[0].get("scale_factor") == 2.0
        and ops_g[1].get("spin_angular_velocity") == 90.0
        and ops_g[2].get("delta") == [1.0, 0.0, 0.0],
        {
            "add_counts": [
                add.get("operation_count")
                for add in (static_add_1, static_add_2, static_add_3)
            ],
            "listed": ops_g,
        },
    )

    # ----- H. editing the middle entry ----------------------------
    edit_h, _ = tool(
        "set_static_op",
        static_args({"index": 1, "spin_angular_velocity": 45.0}),
    )
    ops_h = static_ops()
    diff_h = entry_diff(ops_g, ops_h)
    mcp_check(
        result, "H_static_edit_keeps_order_and_one_field",
        edit_h.get("status") == "success"
        and edit_h.get("index") == 1
        and edit_h.get("op_type") == "SPIN"
        and edit_h.get("updates") == {"spin_angular_velocity": 45.0}
        and edit_h.get("operation_count") == 3
        and op_types(ops_h) == ["SCALE", "SPIN", "MOVE_BY"]
        and diff_h == [[1, "spin_angular_velocity", 90.0, 45.0]],
        {
            "reply": edit_h,
            "order": op_types(ops_h),
            "diff": diff_h,
            "listed": ops_h,
        },
    )

    # ----- I. and the move that does change the order -------------
    move_i = static_tool("move_static_op", {"index": 0, "new_index": 2})
    ops_i = static_ops()
    mcp_check(
        result, "I_static_move_reorders_the_listing",
        move_i.get("index") == 2
        and move_i.get("op_type") == "SCALE"
        and move_i.get("operation_order") == ["SPIN", "MOVE_BY", "SCALE"]
        and op_types(ops_i) == ["SPIN", "MOVE_BY", "SCALE"]
        and ops_i == [ops_h[1], ops_h[2], ops_h[0]],
        {
            "reply": move_i,
            "order_before": op_types(ops_h),
            "order_after": op_types(ops_i),
            "carried_fields": ops_i == [ops_h[1], ops_h[2], ops_h[0]],
            "listed": ops_i,
        },
    )

    # ----- J. an index that addresses no entry --------------------
    static_bad = {
        "set_past_end": static_refused(
            "set_static_op", {"index": 3, "frame_end": 40}
        ),
        "set_negative": static_refused(
            "set_static_op", {"index": -1, "frame_end": 40}
        ),
        "move_index": static_refused(
            "move_static_op", {"index": 9, "new_index": 0}
        ),
        "move_new_index": static_refused(
            "move_static_op", {"index": 0, "new_index": 3}
        ),
    }
    ops_j = static_ops()
    mcp_check(
        result, "J_static_out_of_range_index_is_refused",
        is_refused(static_bad["set_past_end"], "Index 3 out of range", "0..2")
        and is_refused(
            static_bad["set_negative"], "Index -1 out of range", "0..2"
        )
        and is_refused(static_bad["move_index"], "index 9 out of range", "0..2")
        and is_refused(
            static_bad["move_new_index"], "new_index 3 out of range", "0..2"
        )
        and entry_diff(ops_i, ops_j) == [],
        {"refusals": static_bad, "diff": entry_diff(ops_i, ops_j)},
    )

    # ----- K. a call that names no field at all -------------------
    refuse_k = static_refused("set_static_op", {"index": 0})
    ops_k = static_ops()
    mcp_check(
        result, "K_static_call_naming_no_field_is_refused",
        is_refused(
            refuse_k, "No field named", "op_type=SPIN", "remove_static_op"
        )
        and entry_diff(ops_i, ops_k) == [],
        {"refusal": refuse_k, "diff": entry_diff(ops_i, ops_k)},
    )

    # ----- L. a field belonging to another op type ----------------
    refuse_l = static_refused(
        "set_static_op",
        {"index": 1, "delta": [9.0, 9.0, 9.0], "spin_angular_velocity": 30.0},
    )
    ops_l = static_ops()
    mcp_check(
        result, "L_static_foreign_field_is_refused",
        is_refused(
            refuse_l,
            "Field 'spin_angular_velocity' is not valid for "
            "static op_type=MOVE_BY",
            "allowed:",
            "'delta'",
        )
        and entry_diff(ops_i, ops_l) == [],
        {
            "refusal": refuse_l,
            "diff": entry_diff(ops_i, ops_l),
            "listed": ops_l,
        },
    )

    # ================= velocity keyframes =========================
    # Added out of frame order, so the list the edits address is one the
    # handler sorted rather than one the caller happened to build in order.
    for frame, speed in ((30, 3.0), (10, 1.0), (20, 2.0)):
        must(
            "add_velocity_keyframe",
            sheet_args(
                {
                    "frame": frame,
                    "direction": [1.0, 0.0, 0.0],
                    "speed": speed,
                }
            ),
        )
    kf_before_m = velocity_rna()

    # ----- M. an edit that retimes nothing ------------------------
    edit_m, _ = tool(
        "set_velocity_keyframe", sheet_args({"index": 1, "speed": 2.5})
    )
    kf_m = velocity_rna()
    diff_m = entry_diff(kf_before_m, kf_m)
    mcp_check(
        result, "M_velocity_edit_keeps_frame_and_one_field",
        [kf["frame"] for kf in kf_before_m] == [10, 20, 30]
        and edit_m.get("status") == "success"
        and edit_m.get("index") == 1
        and edit_m.get("new_index") == 1
        and edit_m.get("frame") == 20
        and edit_m.get("updates") == {"speed": 2.5}
        and edit_m.get("keyframe_count") == 3
        and diff_m == [[1, "speed", 2.0, 2.5]],
        {
            "reply": edit_m,
            "frames_before": [kf["frame"] for kf in kf_before_m],
            "frames_after": [kf["frame"] for kf in kf_m],
            "diff": diff_m,
        },
    )

    # ----- N. retiming onto a frame already taken -----------------
    refuse_n = refused(
        "set_velocity_keyframe",
        sheet_args({"index": 1, "frame": 30, "speed": 9.0}),
    )
    kf_n = velocity_rna()
    mcp_check(
        result, "N_velocity_retime_onto_taken_frame_is_refused",
        is_refused(
            refuse_n,
            "Frame 30 already has a velocity keyframe",
            SHEET_NAME,
        )
        and entry_diff(kf_m, kf_n) == [],
        {"refusal": refuse_n, "diff": entry_diff(kf_m, kf_n)},
    )

    # ----- O. a retime that reorders the list ---------------------
    edit_o, _ = tool(
        "set_velocity_keyframe", sheet_args({"index": 0, "frame": 25})
    )
    kf_o = velocity_rna()
    listed_o = velocity_listed()
    mcp_check(
        result, "O_velocity_retime_reports_the_new_index",
        edit_o.get("status") == "success"
        and edit_o.get("index") == 0
        and edit_o.get("new_index") == 1
        and edit_o.get("frame") == 25
        and edit_o.get("updates") == {"frame": 25}
        and [kf["frame"] for kf in kf_o] == [20, 25, 30]
        # The entry that moved carried its own speed to the new index.
        and [kf["speed"] for kf in kf_o] == [2.5, 1.0, 3.0]
        and [entry.get("frame") for entry in listed_o] == [20, 25, 30],
        {
            "reply": edit_o,
            "frames_before": [kf["frame"] for kf in kf_n],
            "frames_after": [kf["frame"] for kf in kf_o],
            "speeds_after": [kf["speed"] for kf in kf_o],
            "listed": listed_o,
        },
    )

    # ================= collision windows ==========================
    for start, end in ((10, 20), (30, 40), (50, 60)):
        must(
            "add_collision_window",
            sheet_args({"frame_start": start, "frame_end": end}),
        )
    windows_before_p = windows()

    # ----- P. an edit that keeps the window at its index ----------
    edit_p, _ = tool(
        "set_collision_window", sheet_args({"index": 1, "frame_end": 45})
    )
    windows_p = windows()
    mcp_check(
        result, "P_window_edit_keeps_its_index",
        window_pairs(windows_before_p) == [[10, 20], [30, 40], [50, 60]]
        and edit_p.get("status") == "success"
        and edit_p.get("index") == 1
        and edit_p.get("frame_start") == 30
        and edit_p.get("frame_end") == 45
        and edit_p.get("window_count") == 3
        and window_pairs(windows_p) == [[10, 20], [30, 45], [50, 60]],
        {
            "reply": edit_p,
            "before": window_pairs(windows_before_p),
            "after": window_pairs(windows_p),
        },
    )

    # ----- Q. two bounds that cross ------------------------------
    refuse_q = refused(
        "set_collision_window",
        sheet_args({"index": 1, "frame_start": 50, "frame_end": 45}),
    )
    windows_q = windows()
    mcp_check(
        result, "Q_window_inverted_by_both_bounds_is_refused",
        is_refused(
            refuse_q, "frame_end must be >= frame_start", "[50-45]"
        )
        and window_pairs(windows_q) == [[10, 20], [30, 45], [50, 60]],
        {"refusal": refuse_q, "after": window_pairs(windows_q)},
    )

    # ----- R. one bound moved past the one already stored ---------
    # The window that RESULTS is what is measured, so each of these is legal
    # on its own and illegal against the bound the entry already carries.
    refuse_r_start = refused(
        "set_collision_window", sheet_args({"index": 1, "frame_start": 99})
    )
    refuse_r_end = refused(
        "set_collision_window", sheet_args({"index": 1, "frame_end": 5})
    )
    windows_r = windows()
    mcp_check(
        result, "R_window_inverted_by_one_bound_is_refused",
        is_refused(
            refuse_r_start, "frame_end must be >= frame_start", "[99-45]"
        )
        and is_refused(
            refuse_r_end, "frame_end must be >= frame_start", "[30-5]"
        )
        and window_pairs(windows_r) == [[10, 20], [30, 45], [50, 60]],
        {
            "refusal_start": refuse_r_start,
            "refusal_end": refuse_r_end,
            "after": window_pairs(windows_r),
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
