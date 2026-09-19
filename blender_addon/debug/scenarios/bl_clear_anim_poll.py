# File: scenarios/bl_clear_anim_poll.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard for the "Clear Local Animation" enable/disable signal.
#
# The button's poll() (and the bake polls, and the Run drift gate) must
# reflect, on EVERY redraw with no memoized state, whether solver animation
# exists on an object that button can actually clear. The check is
# ``pc2.scene_has_solver_cache()``: it scans object modifiers for a
# ContactSolverCache (and the in-memory curve cache) and keeps the ones whose
# UUID is assigned to an ACTIVE group, which is the set
# ``clear_animation_data`` walks. It stays stateless (an earlier
# generation-memoized cache went stale after fetch and is gone).
#
#   A. no_cache_false: with no ContactSolverCache anywhere, the scan is False.
#   B. modifier_enables_stateless: adding a ContactSolverCache modifier to an
#      assigned object flips the scan to True immediately, with NO depsgraph
#      update in between (the whole point of being stateless).
#   C. remove_disables: removing it flips the scan back to False at once.
#   D. curve_cache_enables: an in-memory curve cache (rods, which use no
#      MESH_CACHE modifier) also reads as True, and clears to False.
#   E. poll_reflects_cache: SOLVER_OT_ClearAnimation.poll returns True with a
#      cache present and False without (it routes through the same scan).
#   F. unassigned_copy_ignored: a ContactSolverCache on an object that belongs
#      to no group is NOT counted. Duplicating a simulated object copies the
#      modifier, and the copy's inherited UUID is stripped, so no group
#      references it and Clear Local Animation cannot reach it. Counting it
#      left Run greyed out for good, with nothing the user could press to
#      re-enable it.
#   G. clear_then_run_gate: the reported symptom end to end. With the original
#      cleared and only the copy still carrying a modifier, the Clear poll
#      reads False, which is what re-opens SOLVER_OT_Run's gate.
#   H. inactive_group_ignored: a cached object in a DEACTIVATED group is not
#      counted either. Clear Local Animation walks active groups only, so a
#      parallel setup the user is not simulating must not block this one.
#   I. missing_pc2_warning_ignores_unassigned_copy: the panel's "Data path:
#      ... does not exist." warning (_find_missing_pc2_paths) takes the same
#      scope. Pointing an unassigned copy's modifier at a file that never
#      existed is enough to raise it, and no button clears it.
#   J. missing_pc2_warning_clears_with_the_cache: once the assigned cache is
#      gone the warning goes with it, even though the copy keeps its broken
#      modifier. The second half of the same report: the warning outlived every
#      action available, sitting beside a greyed-out Clear button.
#   K. orphan_cache_is_named: what the gates ignore, the panel still names.
#      find_orphan_solver_caches reports the copy and never the assigned
#      object, so an addon-created modifier cannot sit in the scene with the
#      addon refusing to admit it exists.
#   L. inactive_group_cache_is_not_an_orphan: membership is tested against
#      EVERY group, not the active ones. Deactivating a group holds a second
#      setup aside; it must not offer to strip that setup's caches.
#   M. remove_stale_cache_spares_the_rest: SOLVER_OT_RemoveStaleCaches drops
#      the orphan's modifier, leaves the assigned object's in place, and
#      leaves the PC2 on disk (a copy's modifier normally points at the
#      ORIGINAL's cache, so deleting the file would take the original's
#      animation with it).
#
# Assertion-only: no server connection or solve.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it. The sweep that had
# failed it loaded a DIFFERENT tree's addon through the shared extension
# symlink, so that verdict was about other code; against this tree it
# passes unchanged.
BACKENDS = ("real",)


_DRIVER_TEMPLATE = r"""
import bpy, os, time, traceback
import numpy as np
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    pc2 = __import__(pkg + ".core.pc2",
                     fromlist=["scene_has_solver_cache", "MODIFIER_NAME",
                               "_curve_cache",
                               "assigned_uuids_in_active_groups"])
    solver = __import__(pkg + ".ui.solver",
                        fromlist=["SOLVER_OT_ClearAnimation"])
    ClearAnim = solver.SOLVER_OT_ClearAnimation
    api = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
    uuidreg = __import__(pkg + ".core.uuid_registry",
                         fromlist=["get_object_uuid",
                                   "get_or_create_object_uuid"])
    groups = __import__(pkg + ".models.groups",
                        fromlist=["iterate_active_object_groups"])
    ctx = bpy.context

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    p1 = ctx.active_object
    p1.name = "ClearPollA"
    grp = api.create_group("Cloth", "SHELL")
    grp.add(p1.name)
    ctx.view_layer.update()

    # A: nothing cached.
    record("A_no_cache_false", pc2.scene_has_solver_cache() is False,
           {"scan": pc2.scene_has_solver_cache()})

    # B: a fresh ContactSolverCache modifier reads True with NO depsgraph
    #    update between the mutation and the scan -- proves it is stateless.
    p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    b_val = pc2.scene_has_solver_cache()
    record("B_modifier_enables_stateless", b_val is True, {"scan": b_val})

    # C: removing it reads False immediately.
    p1.modifiers.remove(p1.modifiers.get(pc2.MODIFIER_NAME))
    record("C_remove_disables", pc2.scene_has_solver_cache() is False,
           {"scan": pc2.scene_has_solver_cache()})

    # D: in-memory curve cache (rods) also counts, under the same scope:
    #    keyed by an assigned object's UUID it reads True, keyed by anything
    #    else it does not.
    uid = uuidreg.get_object_uuid(p1)
    pc2._curve_cache[uid] = np.zeros((1, 1, 3), dtype=np.float32)
    d_on = pc2.scene_has_solver_cache()
    pc2._curve_cache.pop(uid, None)
    pc2._curve_cache["_rig_unassigned_key"] = np.zeros((1, 1, 3), dtype=np.float32)
    d_stray = pc2.scene_has_solver_cache()
    pc2._curve_cache.pop("_rig_unassigned_key", None)
    d_off = pc2.scene_has_solver_cache()
    record("D_curve_cache_enables",
           bool(uid) and d_on is True and d_stray is False and d_off is False,
           {"uuid": uid, "with_curve": d_on, "with_stray_curve": d_stray,
            "without_curve": d_off})

    # E: the operator poll routes through the scan.
    p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    poll_on = ClearAnim.poll(ctx)
    p1.modifiers.remove(p1.modifiers.get(pc2.MODIFIER_NAME))
    poll_off = ClearAnim.poll(ctx)
    record("E_poll_reflects_cache",
           poll_on is True and poll_off is False,
           {"poll_with_cache": poll_on, "poll_without_cache": poll_off})

    # F: duplicate the simulated object, which copies the modifier. The
    #    depsgraph handler strips the copy's inherited UUID (a fresh one is
    #    assigned lazily, the first time a write context needs it), so no
    #    group references the copy either way; run the resolver directly as
    #    well so the check does not depend on handler timing.
    p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    bpy.ops.object.select_all(action="DESELECT")
    p1.select_set(True)
    ctx.view_layer.objects.active = p1
    bpy.ops.object.duplicate(linked=False)
    dup = ctx.active_object
    ctx.view_layer.update()
    if uuidreg.get_object_uuid(dup) == uid:
        uuidreg.get_or_create_object_uuid(dup)
    dup_uid = uuidreg.get_object_uuid(dup)
    assigned = pc2.assigned_uuids_in_active_groups(ctx.scene)
    # Only the copy carries a modifier now: this is the state Clear Local
    # Animation leaves behind once it has cleared the assigned original.
    p1.modifiers.remove(p1.modifiers.get(pc2.MODIFIER_NAME))
    f_scan = pc2.scene_has_solver_cache()
    record("F_unassigned_copy_ignored",
           dup is not p1
           and dup_uid != uid
           and dup.modifiers.get(pc2.MODIFIER_NAME) is not None
           and dup_uid not in assigned
           and f_scan is False,
           {"orig_uuid": uid, "copy_uuid": dup_uid,
            "copy_has_modifier":
                dup.modifiers.get(pc2.MODIFIER_NAME) is not None,
            "copy_assigned": dup_uid in assigned, "scan": f_scan})

    # G: the reported symptom end to end. With both cached the poll is True;
    #    once the assigned original is cleared it reads False even though the
    #    copy keeps its modifier, which is what re-opens SOLVER_OT_Run's gate
    #    (Run polls False while ClearAnim polls True).
    p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    g_before = ClearAnim.poll(ctx)
    p1.modifiers.remove(p1.modifiers.get(pc2.MODIFIER_NAME))
    g_after = ClearAnim.poll(ctx)
    record("G_clear_then_run_gate",
           g_before is True and g_after is False,
           {"poll_before_clear": g_before, "poll_after_clear": g_after,
            "run_gate_open": not g_after})

    # H: a cached object in a deactivated group is out of scope too, since
    #    Clear Local Animation walks active groups only.
    p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    grp_pg = next(iter(groups.iterate_active_object_groups(ctx.scene)), None)
    h_active = pc2.scene_has_solver_cache()
    grp_pg.active = False
    h_inactive = pc2.scene_has_solver_cache()
    grp_pg.active = True
    h_reactivated = pc2.scene_has_solver_cache()
    p1.modifiers.remove(p1.modifiers.get(pc2.MODIFIER_NAME))
    record("H_inactive_group_ignored",
           h_active is True and h_inactive is False
           and h_reactivated is True,
           {"active": h_active, "inactive": h_inactive,
            "reactivated": h_reactivated})

    # I: the panel's missing-data warning reads the same scope. It needs a
    #    saved .blend, since it resolves relative modifier paths.
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(os.path.dirname(PROBE_DIR), "clear_anim_poll.blend")
    )
    assigned_mod = p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    assigned_mod.filepath = "//data/clear_anim_poll/assigned_missing.pc2"
    # The copy points at the path from the report: a file that never existed.
    dup.modifiers.get(pc2.MODIFIER_NAME).filepath = "//aa.pc2"
    missing = solver._find_missing_pc2_paths(ctx)
    record("I_missing_pc2_warning_ignores_unassigned_copy",
           any(m.endswith("assigned_missing.pc2") for m in missing)
           and not any("aa.pc2" in m for m in missing),
           {"missing": missing})

    # J: and it goes quiet once the assigned cache is cleared, even though
    #    the copy keeps its broken modifier. Warning about a path that Clear
    #    Local Animation cannot reach left it on screen for the life of the
    #    file, with the button greyed out beside it.
    p1.modifiers.remove(p1.modifiers.get(pc2.MODIFIER_NAME))
    after_clear = solver._find_missing_pc2_paths(ctx)
    record("J_missing_pc2_warning_clears_with_the_cache",
           after_clear == []
           and dup.modifiers.get(pc2.MODIFIER_NAME) is not None
           and ClearAnim.poll(ctx) is False,
           {"missing_after_clear": after_clear,
            "copy_still_has_modifier":
                dup.modifiers.get(pc2.MODIFIER_NAME) is not None})

    # K: the orphan is NAMED rather than silently ignored. The assigned
    #    object's own cache is never an orphan.
    p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    orphans = pc2.find_orphan_solver_caches(ctx.scene)
    record("K_orphan_cache_is_named",
           [o.name for o in orphans] == [dup.name],
           {"orphans": [o.name for o in orphans], "assigned": p1.name})

    # L: a DEACTIVATED group's cache is not an orphan. Holding a second
    #    setup aside must not offer to strip its caches.
    grp_pg.active = False
    l_orphans = [o.name for o in pc2.find_orphan_solver_caches(ctx.scene)]
    grp_pg.active = True
    record("L_inactive_group_cache_is_not_an_orphan",
           l_orphans == [dup.name],
           {"orphans_while_inactive": l_orphans})

    # M: the operator removes the orphan's modifier, leaves the assigned
    #    object's alone, and leaves every PC2 on disk. A copy's modifier
    #    normally points at the ORIGINAL's cache, so deleting the file
    #    would take the original's animation with it.
    kept_pc2 = pc2.get_pc2_path(uid)
    os.makedirs(os.path.dirname(kept_pc2), exist_ok=True)
    with open(kept_pc2, "wb") as fh:
        fh.write(b"rig-sentinel")
    op_result = bpy.ops.solver.remove_stale_caches()
    record("M_remove_stale_cache_spares_the_rest",
           dup.modifiers.get(pc2.MODIFIER_NAME) is None
           and p1.modifiers.get(pc2.MODIFIER_NAME) is not None
           and os.path.exists(kept_pc2)
           and pc2.find_orphan_solver_caches(ctx.scene) == [],
           {"op": str(op_result),
            "copy_modifier": dup.modifiers.get(pc2.MODIFIER_NAME) is not None,
            "assigned_modifier": p1.modifiers.get(pc2.MODIFIER_NAME) is not None,
            "pc2_kept": os.path.exists(kept_pc2)})

    result["phases"].append((round(time.time(), 3),
                             "checks=" + str(len(result["checks"]))))
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
