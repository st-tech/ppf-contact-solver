# File: scenarios/bl_mcp_object_locks.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP set_object_locks surface, against a real Blender.
#
# Lock Translation and Lock Rotation are per-object exact constraints on the
# Newton direction, and each is carried by three fields plus, for rotation, a
# mode switch: an enable bit, an all-axes mode, a world-space axis, and
# lock_rotation_prohibit_axis. The MODE carries the enable bit, not the axis,
# so an axis of exactly zero is a legal stored value and says nothing about
# whether the lock is on. That is the property this scenario is written
# around: an axis that a per-axis lock would read has to be non-zero, and the
# same axis under a disabled lock, or under the all-axes mode that reads no
# axis at all, has to be accepted.
#
# The refusal is decided on the state the call RESULTS in rather than on its
# arguments, since the enable bit, the mode and the axis can each come from
# the call or from what the object already carries. Two of the checks below
# reach that path from opposite sides: one passes a zero axis to a lock that
# is already enabled, the other enables a lock over an axis that is already
# zero.
#
# One reading note: get_group_objects reports a group's membership together
# with each object's mutable per-object state, including a "locks" object
# carrying all seven lock fields. Check A asserts that shape, and the
# per-lock checks read the state back through an argument-free
# set_object_locks call, which applies nothing and returns the whole lock
# state, so the two read paths are cross-checked against each other.
#
# Assertions:
#   A. ``group_holds_object`` -- the cube is assigned to a SOLID group, and
#      the membership listing names it with a UUID and no lock field.
#   B. ``translation_lock_round_trips`` -- an enable plus an axis in one call
#      lands in the add-on's own state, is echoed by the call, and comes back
#      unchanged from an argument-free read.
#   C. ``partial_call_leaves_others_unchanged`` -- a call naming one field
#      applies exactly that field and leaves the other six as they were.
#   D. ``rotation_lock_round_trips`` -- the rotation half behaves the same
#      way, and the translation lock set in B is untouched by it.
#   E. ``zero_axis_enabled_is_refused`` -- a zero axis on an enabled per-axis
#      lock is refused, naming the lock, the axis argument and the all-axes
#      mode that would make the axis unnecessary.
#   F. ``refused_call_applies_nothing`` -- the refusal in E also drops the
#      second, valid field that call carried, so a refused call is atomic.
#   G. ``zero_axis_accepted_while_disabled`` -- the same zero axis is stored
#      without complaint once the lock is disabled.
#   H. ``zero_axis_accepted_under_all_mode`` -- and while the lock is still
#      ENABLED, provided the all-axes mode is on, which reads no axis.
#   I. ``reenable_over_zero_axis_is_refused`` -- enabling a per-axis lock
#      whose stored axis is zero is refused by the same message, which is
#      what makes the check one on the resulting state.

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

OBJ_NAME = "LockedCube"

try:
    groups = __import__(
        pkg + ".models.groups", fromlist=["get_active_group_by_uuid"]
    )

    # ----- scene: one cube, and nothing else ----------------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = OBJ_NAME

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    def call(request_id, name, arguments):
        # A tool call whose refusal is data rather than a transport failure:
        # a handler that ran and refused answers with isError and a payload
        # whose status is "error", so both are returned to the caller.
        payload, raw = mcp_tool(
            pkg, url, name, arguments, request_id=request_id
        )
        return payload, raw

    def locks_call(request_id, **arguments):
        # set_object_locks against the one assigned object, with every lock
        # argument left to the caller.
        arguments["group_uuid"] = group_uuid
        arguments["object_name"] = OBJ_NAME
        return call(request_id, "set_object_locks", arguments)

    def rna_locks():
        # The add-on's own state, read from the PropertyGroup the handler
        # writes, so a round-trip is measured against the stored value and
        # not only against what the reply repeated back.
        group = groups.get_active_group_by_uuid(bpy.context.scene, group_uuid)
        if group is None:
            raise RuntimeError("group %s is not active" % group_uuid)
        for assigned in group.assigned_objects:
            if assigned.name != OBJ_NAME:
                continue
            return {
                "lock_translation_enable": bool(
                    assigned.lock_translation_enable
                ),
                "lock_translation_all": bool(assigned.lock_translation_all),
                "lock_translation_axis": [
                    float(c) for c in assigned.lock_translation_axis
                ],
                "lock_rotation_enable": bool(assigned.lock_rotation_enable),
                "lock_rotation_all": bool(assigned.lock_rotation_all),
                "lock_rotation_axis": [
                    float(c) for c in assigned.lock_rotation_axis
                ],
                "lock_rotation_prohibit_axis": bool(
                    assigned.lock_rotation_prohibit_axis
                ),
            }
        raise RuntimeError(
            "'%s' is not assigned to group %s" % (OBJ_NAME, group_uuid)
        )

    # ----- a SOLID group holding the cube -------------------------
    created, _ = call(1, "create_group", {"name": "Locks", "type": "SOLID"})
    if created.get("status") != "success":
        raise RuntimeError("create_group: %r" % (created,))
    group_uuid = created.get("group_uuid") or ""
    if not group_uuid:
        raise RuntimeError("create_group returned no uuid: %r" % (created,))

    added, _ = call(
        2,
        "add_objects_to_group",
        {"group_uuid": group_uuid, "object_names": [OBJ_NAME]},
    )
    if added.get("status") != "success" or not added.get("added_objects"):
        raise RuntimeError("add_objects_to_group: %r" % (added,))

    # ----- A. the membership listing, and the lock state it carries
    LOCK_FIELDS = [
        "lock_translation_enable",
        "lock_translation_all",
        "lock_translation_axis",
        "lock_rotation_enable",
        "lock_rotation_all",
        "lock_rotation_axis",
        "lock_rotation_prohibit_axis",
    ]
    listing, _ = call(3, "get_group_objects", {"group_uuid": group_uuid})
    objects = listing.get("objects") or []
    entry = objects[0] if objects else {}
    reported_locks = entry.get("locks") or {}
    missing_lock_fields = [f for f in LOCK_FIELDS if f not in reported_locks]
    mcp_check(
        result, "A_group_holds_object",
        listing.get("status") == "success"
        and listing.get("group_type") == "SOLID"
        and len(objects) == 1
        and entry.get("name") == OBJ_NAME
        and bool(entry.get("uuid"))
        and entry.get("type") == "MESH"
        and isinstance(entry.get("locks"), dict)
        and not missing_lock_fields
        # The listing and the RNA are two views of one state, so they agree.
        and reported_locks.get("lock_translation_enable") is (
            rna_locks()["lock_translation_enable"]
        ),
        {
            "status": listing.get("status"),
            "group_type": listing.get("group_type"),
            "object_count": len(objects),
            "reported_keys": sorted(entry),
            "reported_locks": reported_locks,
            "missing_lock_fields": missing_lock_fields,
            "stored_locks": rna_locks(),
        },
    )

    # ----- B. enable plus axis, in one call -----------------------
    expected_b = {
        "lock_translation_enable": True,
        "lock_translation_all": False,
        "lock_translation_axis": [0.0, 0.0, 1.0],
        "lock_rotation_enable": False,
        "lock_rotation_all": False,
        "lock_rotation_axis": [1.0, 0.0, 0.0],
        "lock_rotation_prohibit_axis": False,
    }
    set_b, _ = locks_call(
        4,
        lock_translation_enable=True,
        lock_translation_axis=[0.0, 0.0, 1.0],
    )
    stored_b = rna_locks()
    # An argument-free call applies nothing and reports the whole state,
    # which is the only read-back this surface offers.
    read_b, _ = locks_call(5)
    mcp_check(
        result, "B_translation_lock_round_trips",
        set_b.get("status") == "success"
        and sorted(set_b.get("updates") or {})
        == ["lock_translation_axis", "lock_translation_enable"]
        and set_b.get("locks") == expected_b
        and stored_b == expected_b
        and read_b.get("status") == "success"
        and read_b.get("updates") == {}
        and read_b.get("locks") == expected_b,
        {
            "set_status": set_b.get("status"),
            "set_message": set_b.get("message"),
            "set_updates": set_b.get("updates"),
            "set_locks": set_b.get("locks"),
            "stored": stored_b,
            "read_back_updates": read_b.get("updates"),
            "read_back_locks": read_b.get("locks"),
            "expected": expected_b,
        },
    )

    # ----- C. a call naming one field moves one field -------------
    expected_c = dict(expected_b)
    expected_c["lock_rotation_prohibit_axis"] = True
    set_c, _ = locks_call(6, lock_rotation_prohibit_axis=True)
    stored_c = rna_locks()
    untouched_c = {
        k: v
        for k, v in stored_c.items()
        if k != "lock_rotation_prohibit_axis"
    }
    mcp_check(
        result, "C_partial_call_leaves_others_unchanged",
        set_c.get("status") == "success"
        and set_c.get("updates") == {"lock_rotation_prohibit_axis": True}
        and stored_c == expected_c,
        {
            "set_status": set_c.get("status"),
            "set_updates": set_c.get("updates"),
            "stored": stored_c,
            "expected": expected_c,
            "untouched": untouched_c,
        },
    )

    # ----- D. the rotation half, independent of the other ---------
    expected_d = dict(expected_c)
    expected_d["lock_rotation_enable"] = True
    expected_d["lock_rotation_axis"] = [0.0, 1.0, 0.0]
    set_d, _ = locks_call(
        7,
        lock_rotation_enable=True,
        lock_rotation_axis=[0.0, 1.0, 0.0],
    )
    stored_d = rna_locks()
    mcp_check(
        result, "D_rotation_lock_round_trips",
        set_d.get("status") == "success"
        and sorted(set_d.get("updates") or {})
        == ["lock_rotation_axis", "lock_rotation_enable"]
        and set_d.get("locks") == expected_d
        and stored_d == expected_d
        and stored_d["lock_translation_enable"] is True
        and stored_d["lock_translation_axis"] == [0.0, 0.0, 1.0],
        {
            "set_status": set_d.get("status"),
            "set_updates": set_d.get("updates"),
            "set_locks": set_d.get("locks"),
            "stored": stored_d,
            "expected": expected_d,
        },
    )

    # ----- E. a zero axis a per-axis lock would read --------------
    # The call also carries a second, valid field, which check F reads back.
    refuse_e, raw_e = locks_call(
        8,
        lock_rotation_axis=[0.0, 0.0, 0.0],
        lock_translation_all=True,
    )
    message_e = refuse_e.get("message") or ""
    mcp_check(
        result, "E_zero_axis_enabled_is_refused",
        refuse_e.get("status") == "error"
        and raw_e.get("isError") is True
        and "Lock Rotation" in message_e
        and "lock_rotation_axis" in message_e
        and "lock_rotation_all" in message_e,
        {
            "status": refuse_e.get("status"),
            "is_error": raw_e.get("isError"),
            "message": message_e,
        },
    )

    # ----- F. and it applied none of the call ---------------------
    stored_f = rna_locks()
    mcp_check(
        result, "F_refused_call_applies_nothing",
        stored_f == expected_d,
        {
            "stored": stored_f,
            "expected": expected_d,
            "differing": sorted(
                k for k in expected_d if stored_f.get(k) != expected_d[k]
            ),
        },
    )

    # ----- G. the same zero axis under a disabled lock ------------
    expected_g = dict(expected_d)
    expected_g["lock_translation_enable"] = False
    expected_g["lock_translation_axis"] = [0.0, 0.0, 0.0]
    set_g, _ = locks_call(
        9,
        lock_translation_enable=False,
        lock_translation_axis=[0.0, 0.0, 0.0],
    )
    stored_g = rna_locks()
    mcp_check(
        result, "G_zero_axis_accepted_while_disabled",
        set_g.get("status") == "success"
        and set_g.get("locks") == expected_g
        and stored_g == expected_g,
        {
            "set_status": set_g.get("status"),
            "set_message": set_g.get("message"),
            "set_updates": set_g.get("updates"),
            "stored": stored_g,
            "expected": expected_g,
        },
    )

    # ----- H. and under the all-axes mode, still enabled ----------
    expected_h = dict(expected_g)
    expected_h["lock_rotation_all"] = True
    expected_h["lock_rotation_axis"] = [0.0, 0.0, 0.0]
    set_h, _ = locks_call(
        10,
        lock_rotation_all=True,
        lock_rotation_axis=[0.0, 0.0, 0.0],
    )
    stored_h = rna_locks()
    mcp_check(
        result, "H_zero_axis_accepted_under_all_mode",
        set_h.get("status") == "success"
        and set_h.get("locks") == expected_h
        and stored_h == expected_h
        and stored_h["lock_rotation_enable"] is True,
        {
            "set_status": set_h.get("status"),
            "set_message": set_h.get("message"),
            "set_updates": set_h.get("updates"),
            "stored": stored_h,
            "expected": expected_h,
        },
    )

    # ----- I. enabling a lock over an axis already zero -----------
    refuse_i, raw_i = locks_call(11, lock_translation_enable=True)
    message_i = refuse_i.get("message") or ""
    stored_i = rna_locks()
    mcp_check(
        result, "I_reenable_over_zero_axis_is_refused",
        refuse_i.get("status") == "error"
        and raw_i.get("isError") is True
        and "Lock Translation" in message_i
        and "lock_translation_axis" in message_i
        and "lock_translation_all" in message_i
        and stored_i == expected_h,
        {
            "status": refuse_i.get("status"),
            "is_error": raw_i.get("isError"),
            "message": message_i,
            "stored": stored_i,
            "expected": expected_h,
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
