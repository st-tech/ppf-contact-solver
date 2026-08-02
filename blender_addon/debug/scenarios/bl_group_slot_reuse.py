# File: scenarios/bl_group_slot_reuse.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Group creation and deletion addressed by slot, not by scan position.
#
# A group lives in one of the fixed ``object_group_N`` pointer slots.
# ``OBJECT_OT_CreateGroup`` takes the LOWEST FREE slot, and
# ``object.delete_group`` addresses a group by that same slot index. Two
# things follow, and both are invisible until a scene has deleted a group:
# the lowest free slot stops being the highest active one, so identifying a
# freshly created group by scanning for the last active slot returns some
# other group; and ``ObjectGroup.index`` numbers only the ACTIVE groups
# consecutively for display, so it stops agreeing with the slot and cannot
# be used to address one.
#
# This matters most to scripted authoring, where the standard re-run guard
# is "delete the group with my name, then create it again"
# (``examples/blender/*.py``). Under a scan-based identity that guard
# renames and retypes a bystander group and configures it in place of the
# one it just made.
#
# Subtests:
#   A. delete_api_succeeds
#         ``Group.delete()`` completes. It reaches ``object.delete_group``,
#         which has a ``group_index`` property and no ``group_uuid`` one, so
#         addressing it by UUID raises and takes every re-run guard with it.
#   B. delete_removes_only_its_target
#         Deleting the middle of three groups leaves the other two active
#         with their names and UUIDs unchanged.
#   C. create_fills_the_freed_slot
#         With a lower slot free, the created group lands in THAT slot and
#         carries the requested name and type.
#   D. create_returns_the_group_it_made
#         The returned proxy resolves to the newly created group, so the
#         name, type and params the caller sets land on it.
#   E. create_leaves_other_groups_intact
#         Every other group keeps its name, UUID, slot and type across the
#         create. This is the bystander-clobber regression.
#   F. authoring_rerun_is_idempotent
#         Running a delete-then-create authoring pass twice leaves exactly
#         the same two groups, each holding its own params.
#   G. delete_all_groups_clears_the_current_handle
#         ``object.delete_all_groups`` reports FINISHED, frees every slot,
#         AND clears ``State.current_group_uuid``, the handle
#         ``OBJECT_OT_CreateGroup`` publishes and every caller reads a
#         newly created group back through. The handle is declared on
#         ``State``, one level below the scene root that carries the group
#         slots. Addressing it at the wrong level is not an error a caller
#         can see: assigning a name that is not an RNA property on a
#         PropertyGroup falls through to a plain Python attribute on the
#         wrapper, so the operator still reports FINISHED and still empties
#         every slot while the handle keeps naming a group that is gone.
#         Only reading the handle back afterward catches it, which is what
#         this subtest does.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    groups_mod = __import__(pkg + ".models.groups",
                           fromlist=["iterate_object_groups"])
    solver = dh.api.solver

    def slots():
        # (slot, name, uuid, type) for every ACTIVE group, slot-ordered.
        return [
            (i, g.name, g.uuid, g.object_type)
            for i, g in enumerate(groups_mod.iterate_object_groups(bpy.context.scene))
            if g.active
        ]

    def slot_of(uuid):
        for i, name, u, t in slots():
            if u == uuid:
                return i
        return None

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    solver.clear()

    # ----- three groups in slots 0, 1, 2 --------------------------
    a = solver.create_group("SlotA", "SHELL")
    b = solver.create_group("SlotB", "ROD")
    c = solver.create_group("SlotC", "SOLID")
    before = slots()
    uuid_a, uuid_b, uuid_c = a.uuid, b.uuid, c.uuid

    # ----- A: Group.delete() reaches the operator at all ----------
    delete_err = ""
    try:
        b.delete()
    except Exception as exc:
        delete_err = f"{type(exc).__name__}: {exc}"
    dh.record(
        "A_delete_api_succeeds",
        delete_err == "",
        {"error": delete_err, "before": before},
    )

    # ----- B: only the target went ---------------------------------
    after_delete = slots()
    names_after = {n for _, n, _, _ in after_delete}
    uuids_after = {u for _, _, u, _ in after_delete}
    dh.record(
        "B_delete_removes_only_its_target",
        names_after == {"SlotA", "SlotC"}
        and uuids_after == {uuid_a, uuid_c}
        and slot_of(uuid_a) == 0 and slot_of(uuid_c) == 2,
        {"after_delete": after_delete},
    )

    # ----- C/D/E: create with slot 1 free --------------------------
    d = solver.create_group("SlotD", "ROD")
    after_create = slots()
    uuid_d = d.uuid

    dh.record(
        "C_create_fills_the_freed_slot",
        slot_of(uuid_d) == 1,
        {"slot_of_new": slot_of(uuid_d), "after_create": after_create},
    )

    # The proxy must resolve to the group that was actually created: read
    # the name back through it, and write a param through it and confirm
    # the write landed on that same group.
    d.param.rod_young_modulus = 4321.0
    live_d = groups_mod.get_group_by_uuid(bpy.context.scene, uuid_d)
    dh.record(
        "D_create_returns_the_group_it_made",
        d.name == "SlotD"
        and live_d is not None
        and live_d.name == "SlotD"
        and live_d.object_type == "ROD"
        and abs(live_d.rod_young_modulus - 4321.0) < 1e-3,
        {
            "proxy_name": d.name,
            "live_name": live_d.name if live_d else None,
            "live_type": live_d.object_type if live_d else None,
            "young": live_d.rod_young_modulus if live_d else None,
        },
    )

    kept = {
        u: (i, n, t) for i, n, u, t in after_create if u in (uuid_a, uuid_c)
    }
    dh.record(
        "E_create_leaves_other_groups_intact",
        kept.get(uuid_a) == (0, "SlotA", "SHELL")
        and kept.get(uuid_c) == (2, "SlotC", "SOLID")
        and len(after_create) == 3,
        {"kept": {k[:6]: v for k, v in kept.items()},
         "after_create": after_create},
    )

    # ----- F: the authoring re-run pattern -------------------------
    # Mirrors examples/blender/*.py: delete any group carrying my name,
    # then create it fresh. Two passes must land on the same state.
    def authoring_pass():
        for g in solver.get_groups():
            if g.name in ("Warp", "Weft"):
                g.delete()
        warp = solver.create_group("Warp", "ROD")
        warp.param.length_factor = 0.83
        weft = solver.create_group("Weft", "ROD")
        weft.param.length_factor = 1.0
        return warp.uuid, weft.uuid

    solver.clear()
    authoring_pass()
    pass1 = [(n, t) for _, n, _, t in slots()]
    w2_uuid, f2_uuid = authoring_pass()
    pass2 = [(n, t) for _, n, _, t in slots()]
    live_w = groups_mod.get_group_by_uuid(bpy.context.scene, w2_uuid)
    live_f = groups_mod.get_group_by_uuid(bpy.context.scene, f2_uuid)
    dh.record(
        "F_authoring_rerun_is_idempotent",
        pass1 == pass2
        and sorted(n for n, _ in pass2) == ["Warp", "Weft"]
        and live_w is not None and live_f is not None
        and abs(live_w.length_factor - 0.83) < 1e-4
        and abs(live_f.length_factor - 1.0) < 1e-4,
        {
            "pass1": pass1, "pass2": pass2,
            "warp_lf": live_w.length_factor if live_w else None,
            "weft_lf": live_f.length_factor if live_f else None,
        },
    )

    # ----- G: delete_all_groups clears the current-group handle ----
    # Pass F leaves two groups and a handle naming the last one created,
    # so the handle is non-empty going in and the clear is observable.
    # Status and slot emptiness are asserted alongside it because a
    # misdirected write to the handle disturbs neither.
    state = groups_mod.get_addon_data(bpy.context.scene).state
    handle_before = state.current_group_uuid
    delete_all_status = []
    delete_all_err = ""
    try:
        delete_all_status = sorted(bpy.ops.object.delete_all_groups())
    except Exception as exc:
        delete_all_err = f"{type(exc).__name__}: {exc}"
    handle_after = state.current_group_uuid
    dh.record(
        "G_delete_all_groups_clears_the_current_handle",
        delete_all_err == ""
        and delete_all_status == ["FINISHED"]
        and handle_before != ""
        and handle_after == ""
        and slots() == [],
        {
            "status": delete_all_status,
            "error": delete_all_err,
            "handle_before": handle_before,
            "handle_after": handle_after,
            "remaining": slots(),
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
