# File: scenarios/bl_cleanup_respects_user_obj_color.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression: the deferred cleanup that runs off depsgraph updates
# (``group_ops._apply_cleanup``) must not write obj.color on any object.
# It used to sweep every MESH and CURVE in bpy.data.objects and reset the
# color of anything not in an active group, which made the Object
# Properties > Viewport Display > Color field unsettable on every mesh in
# the file: with the add-on enabled the user's pick reverted to white on
# the next event-loop tick, including on a fresh file with no groups.
#
# The add-on now writes a display color only on an object that is in a
# group, and takes it back at the point membership ends. So the cleanup
# pass has nothing it is allowed to write.
#
# This is a deliberate sibling of bl_overlay_respects_user_obj_color,
# which gates the OTHER writer (overlay.apply_object_overlays). Neither
# scenario covers the other's path.
#
# The driver calls _apply_cleanup() directly. It cannot wait for the real
# 0.0s timer: the harness runs the whole driver body in one exec() on the
# main thread and writes scenario_result.json as soon as it returns, so a
# queued bpy.app.timers callback can never run before an assertion. To
# prove the direct call is not exercising a dead path, subtest W first
# asserts the handler is registered and that a depsgraph tick re-arms it.
#
# Subtests:
#   W. wiring: _cleanup_deleted_objects is in depsgraph_update_post and a
#      depsgraph tick sets the _cleanup_scheduled latch.
#   A. mesh bystander, never in any group, keeps its color.
#   B. curve bystander keeps its color (the old sweep covered CURVE too).
#   C. removing an object from a group clears the tint at removal, and a
#      color the user sets afterwards survives cleanup.
#   D. a member keeps its group tint through cleanup.
#   E. a copy of a member keeps its inherited color; the add-on does not
#      reach outside the group to normalize it.
#   F. preserved behavior: cleanup still prunes an assignment whose
#      object was deleted from the scene.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def _close(a, b, eps=1e-5):
    return all(abs(a[i] - b[i]) < eps for i in range(4))


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    group_ops = __import__(
        pkg + ".ui.dynamics.group_ops",
        fromlist=["_apply_cleanup", "_cleanup_deleted_objects"],
    )
    groups_mod = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    apply_cleanup = group_ops._apply_cleanup

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # ---- W: the path under test is really wired to the depsgraph ----
    handler_names = [
        getattr(h, "__name__", "") for h in bpy.app.handlers.depsgraph_update_post
    ]
    registered = "_cleanup_deleted_objects" in handler_names
    # The latch is a module global cleared only inside _apply_cleanup, so
    # within one driver it stays True after the first tick. Reset it, then
    # cause a depsgraph update and check it re-arms.
    group_ops._cleanup_scheduled = False
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(3.0, 3.0, 0.0))
    bystander = bpy.context.active_object
    bystander.name = "Bystander"
    bpy.context.view_layer.update()
    rearmed = bool(group_ops._cleanup_scheduled)
    group_ops._cleanup_scheduled = False
    dh.record(
        "W_cleanup_is_wired_to_depsgraph",
        registered and rearmed,
        {"handlers": handler_names, "latch_rearmed": rearmed},
    )

    # ---- A: mesh bystander keeps the color the user picked ----------
    bystander.color = (1.0, 0.0, 0.0, 1.0)
    apply_cleanup()
    a_after = tuple(bystander.color)
    dh.record(
        "A_mesh_bystander_color_survives_cleanup",
        _close(a_after, (1.0, 0.0, 0.0, 1.0)),
        {"before": [1.0, 0.0, 0.0, 1.0], "after": list(a_after)},
    )

    # ---- B: curve bystander keeps its color -------------------------
    bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(5.0, 3.0, 0.0))
    curve_obj = bpy.context.active_object
    curve_obj.name = "BystanderCurve"
    curve_obj.color = (0.0, 0.0, 1.0, 1.0)
    apply_cleanup()
    b_after = tuple(curve_obj.color)
    dh.record(
        "B_curve_bystander_color_survives_cleanup",
        _close(b_after, (0.0, 0.0, 1.0, 1.0)),
        {"after": list(b_after)},
    )

    # ---- set up a real group member ---------------------------------
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    member = bpy.context.active_object
    member.name = "MemberPlane"

    api = dh.api.solver
    cloth = api.create_group("Cloth", "SHELL")
    cloth.add(member.name)
    root = groups_mod.get_addon_data(bpy.context.scene)
    slot = groups_mod.get_group_slot_index(
        bpy.context.scene, root.state.current_group_uuid
    )
    group_pg = getattr(root, "object_group_%d" % slot)
    group_pg.show_overlay_color = True
    group_pg.color = (0.2, 0.7, 0.3, 1.0)
    expected_tint = (0.2, 0.7, 0.3, 1.0)
    overlay_mod = __import__(
        pkg + ".ui.dynamics.overlay", fromlist=["apply_object_overlays"]
    )
    overlay_mod.apply_object_overlays()

    # ---- D: a member keeps its tint through cleanup ------------------
    apply_cleanup()
    d_after = tuple(member.color)
    dh.record(
        "D_member_keeps_group_tint_through_cleanup",
        _close(d_after, expected_tint),
        {"after": list(d_after), "expected": list(expected_tint)},
    )

    # ---- E: a copy of a member is left alone -------------------------
    for o in bpy.data.objects:
        o.select_set(o is member)
    bpy.context.view_layer.objects.active = member
    bpy.ops.object.duplicate()
    copy_obj = bpy.context.view_layer.objects.active
    copy_inherited = tuple(copy_obj.color)
    apply_cleanup()
    e_after = tuple(copy_obj.color)
    dh.record(
        "E_copy_of_member_is_not_normalized",
        _close(e_after, copy_inherited),
        {
            "inherited": list(copy_inherited),
            "after": list(e_after),
            "copy_uuid": copy_obj.get("_solver_uuid"),
        },
    )
    bpy.data.objects.remove(copy_obj, do_unlink=True)

    # ---- C: removal clears the tint, then the user's pick sticks -----
    cloth.remove(member.name)
    c_at_removal = tuple(member.color)
    member.color = (1.0, 0.5, 0.0, 1.0)
    apply_cleanup()
    c_after = tuple(member.color)
    dh.record(
        "C_ex_member_is_cleared_at_removal_then_left_alone",
        _close(c_at_removal, (1.0, 1.0, 1.0, 1.0))
        and _close(c_after, (1.0, 0.5, 0.0, 1.0)),
        {
            "color_at_removal": list(c_at_removal),
            "user_pick": [1.0, 0.5, 0.0, 1.0],
            "after_cleanup": list(c_after),
            "still_has_uuid": member.get("_solver_uuid") is not None,
        },
    )

    # ---- F: stale-assignment pruning still works --------------------
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 2.0, 0.0))
    doomed = bpy.context.active_object
    doomed.name = "DoomedPlane"
    cloth2 = api.create_group("Cloth2", "SHELL")
    cloth2.add(doomed.name)
    slot2 = groups_mod.get_group_slot_index(
        bpy.context.scene, root.state.current_group_uuid
    )
    group2_pg = getattr(root, "object_group_%d" % slot2)
    before_count = len(group2_pg.assigned_objects)
    bpy.data.objects.remove(doomed, do_unlink=True)
    apply_cleanup()
    after_count = len(group2_pg.assigned_objects)
    dh.record(
        "F_stale_assignment_still_pruned",
        before_count == 1 and after_count == 0,
        {"before": before_count, "after": after_count},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 60.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
