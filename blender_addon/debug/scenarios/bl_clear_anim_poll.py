# File: scenarios/bl_clear_anim_poll.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard for the "Clear Local Animation" enable/disable signal.
#
# The button's poll() (and the bake polls, and the Run drift gate) must
# reflect, on EVERY redraw with no memoized state, whether any object carries
# solver animation. The check is ``pc2.scene_has_solver_cache()``: it scans
# object modifiers for a ContactSolverCache (and the in-memory curve cache)
# directly, which is ~100x cheaper than resolving every assigned object by
# UUID, so it stays sub-millisecond on large scenes while remaining stateless
# (an earlier generation-memoized cache went stale after fetch and is gone).
#
#   A. no_cache_false: with no ContactSolverCache anywhere, the scan is False.
#   B. modifier_enables_stateless: adding a ContactSolverCache modifier flips
#      the scan to True immediately, with NO depsgraph update in between (the
#      whole point of being stateless).
#   C. remove_disables: removing it flips the scan back to False at once.
#   D. curve_cache_enables: an in-memory curve cache (rods, which use no
#      MESH_CACHE modifier) also reads as True, and clears to False.
#   E. poll_reflects_cache: SOLVER_OT_ClearAnimation.poll returns True with a
#      cache present and False without (it routes through the same scan).
#
# Assertion-only: no server connection or solve.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback
import numpy as np
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    pc2 = __import__(pkg + ".core.pc2",
                     fromlist=["scene_has_solver_cache", "MODIFIER_NAME",
                               "_curve_cache"])
    solver = __import__(pkg + ".ui.solver",
                        fromlist=["SOLVER_OT_ClearAnimation"])
    ClearAnim = solver.SOLVER_OT_ClearAnimation
    api = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
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

    # D: in-memory curve cache (rods) also counts.
    pc2._curve_cache["_rig_clear_poll_key"] = np.zeros((1, 1, 3), dtype=np.float32)
    d_on = pc2.scene_has_solver_cache()
    pc2._curve_cache.pop("_rig_clear_poll_key", None)
    d_off = pc2.scene_has_solver_cache()
    record("D_curve_cache_enables", d_on is True and d_off is False,
           {"with_curve": d_on, "without_curve": d_off})

    # E: the operator poll routes through the scan.
    p1.modifiers.new(pc2.MODIFIER_NAME, "MESH_CACHE")
    poll_on = ClearAnim.poll(ctx)
    p1.modifiers.remove(p1.modifiers.get(pc2.MODIFIER_NAME))
    poll_off = ClearAnim.poll(ctx)
    record("E_poll_reflects_cache",
           poll_on is True and poll_off is False,
           {"poll_with_cache": poll_on, "poll_without_cache": poll_off})

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
