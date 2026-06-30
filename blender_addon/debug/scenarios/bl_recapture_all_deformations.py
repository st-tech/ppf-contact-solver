# File: scenarios/bl_recapture_all_deformations.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# "Re-capture All Deformations": one Solver-panel button that rebuilds every
# deforming STATIC collider's deformation cache AND every animated pin's
# capture in a single pass (the per-item Capture Deformation buttons,
# batched). The button is always shown but its poll() greys it out when
# nothing across the active groups needs a capture.
#
# This scenario verifies, with no server connection (fast + deterministic on
# every host):
#
#   A. enumerate_static_and_pins:
#         collect_capturable_static_objects finds the deforming STATIC
#         collider, and collect_capturable_pins finds the one animated pin.
#   B. poll_true_with_work:
#         SOLVER_OT_RecaptureAllDeformations.poll() is True once capturable
#         work exists.
#   C. static_capture_writes_cache:
#         driving the STATIC phase through the new public wrappers
#         (start_capture_for_objects / advance_capture / finalize_capture /
#         cleanup_capture) writes a static-deform cache of the expected
#         frame count.
#   D. pin_capture_writes_cache:
#         the same for the PIN phase wrappers (start_capture_for_pins /
#         advance_pin_capture / finalize_pin_capture / cleanup_pin_capture),
#         setting the pin's has_captured_anim flag.
#   E. poll_false_when_nothing_capturable:
#         on a clean scene with no groups, poll() is False and both
#         enumerators return empty (the "disabled when no capturable
#         objects" requirement).
#   F. operator_registered:
#         the operator is registered under bpy.ops.solver.
#   G. clear_all_removes_caches:
#         solver.clear_all_deformations drops both the static-deform cache
#         and the pin capture (and clears the pin's has_captured_anim flag)
#         in one pass.
#   H. clear_all_poll_false_when_empty:
#         once nothing is cached, the Clear All button's poll() is False
#         (the "disabled when nothing to clear" requirement).
#
# The new public wrappers are exactly what the modal coordinator calls each
# timer tick, so driving them directly here exercises the real capture path
# without the event loop the headless rig can't pump.

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


def _new_geo_group(name):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT",
                            socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT",
                            socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    return ng, gin, gout


def _set_position_group(name, offset):
    # A genuine deformer: writes vertex position with a constant offset, so
    # has_deforming_modifier_stack / is_deforming_static_object both classify
    # the object as deforming (declarative tier, no keyframes needed).
    ng, gin, gout = _new_geo_group(name)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    sp.inputs["Offset"].default_value = offset
    ng.links.new(gin.outputs[0], sp.inputs["Geometry"])
    ng.links.new(sp.outputs["Geometry"], gout.inputs[0])
    return ng


def _drive_to_completion(advance):
    # Pump a capture job's tick wrapper until it reports done or aborted,
    # mirroring what the modal coordinator does each timer tick.
    guard = 0
    aborted, err = False, ""
    while True:
        more, aborted, err = advance(bpy.context)
        guard += 1
        if aborted or not more or guard > 100000:
            break
    return aborted, err


try:
    sd = __import__(
        pkg + ".ui.dynamics.static_deform_ops",
        fromlist=["collect_capturable_static_objects",
                  "start_capture_for_objects", "advance_capture",
                  "finalize_capture", "cleanup_capture",
                  "object_deformation_frame_count"],
    )
    pc = __import__(
        pkg + ".ui.dynamics.pin_capture_ops",
        fromlist=["collect_capturable_pins", "start_capture_for_pins",
                  "advance_pin_capture", "finalize_pin_capture",
                  "cleanup_pin_capture", "pin_captured_frame_count"],
    )
    solver_mod = __import__(
        pkg + ".ui.solver", fromlist=["SOLVER_OT_RecaptureAllDeformations"]
    )
    pc2 = __import__(
        pkg + ".core.pc2",
        fromlist=["has_static_deform_animation", "has_pin_anim_pc2"],
    )
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    solver_api = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
    Recap = solver_mod.SOLVER_OT_RecaptureAllDeformations

    ctx = bpy.context
    scene = ctx.scene

    # ---- Fresh scene with no objects and no groups -------------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    try:
        bpy.ops.object.delete_all_groups()
    except Exception:
        pass
    scene.frame_start = 1
    scene.frame_end = 12

    # ---- F: operator is registered ----------------------------------
    record(
        "F_operator_registered",
        hasattr(bpy.ops.solver, "recapture_all_deformations"),
        {"has_op": hasattr(bpy.ops.solver, "recapture_all_deformations")},
    )

    # ---- E: nothing capturable -> poll False, enumerators empty ------
    e_poll = Recap.poll(ctx)
    e_static = sd.collect_capturable_static_objects(ctx)
    e_pins = pc.collect_capturable_pins(ctx, cheap=True)
    record(
        "E_poll_false_when_nothing_capturable",
        e_poll is False and e_static == [] and e_pins == [],
        {"poll": e_poll, "n_static": len(e_static), "n_pins": len(e_pins)},
    )

    # ---- Build a deforming STATIC collider --------------------------
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    s_obj = ctx.active_object
    s_obj.name = "DeformStatic"
    s_mod = s_obj.modifiers.new("Deform", "NODES")
    s_mod.node_group = _set_position_group("_recap_static_sp", (0.0, 0.0, 0.3))
    ctx.view_layer.update()
    static_group = solver_api.create_group("StaticG", "STATIC")
    static_group.add(s_obj.name)

    # ---- Build a deforming SHELL object with a full-cover pin --------
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(3.0, 0.0, 0.0))
    c_obj = ctx.active_object
    c_obj.name = "DeformCloth"
    c_mod = c_obj.modifiers.new("Deform", "NODES")
    c_mod.node_group = _set_position_group("_recap_cloth_sp", (0.0, 0.0, 0.5))
    n_c = len(c_obj.data.vertices)
    c_obj.vertex_groups.new(name="PinVG").add(
        list(range(n_c)), 1.0, "REPLACE"
    )
    ctx.view_layer.update()
    cloth_group = solver_api.create_group("ClothG", "SHELL")
    cloth_group.add(c_obj.name)
    cloth_group.create_pin(c_obj.name, "PinVG")

    # The deformers carry no keyframes, so _effective_frame_range bounds
    # each capture by the simulation frame count (state.frame_count). Read
    # it at runtime so the expected cache length tracks the real range logic
    # rather than a hardcoded guess.
    EXPECTED = int(groups.get_addon_data(scene).state.frame_count)

    # ---- A: enumeration finds the static object and the pin ----------
    stat_objs = sd.collect_capturable_static_objects(ctx, allow_eval=True)
    stat_names = [o.name for o in stat_objs]
    pin_specs = pc.collect_capturable_pins(ctx)
    record(
        "A_enumerate_static_and_pins",
        "DeformStatic" in stat_names and len(pin_specs) == 1,
        {"static_objects": stat_names, "pin_specs": pin_specs},
    )

    # ---- B: coordinator poll True now that work exists --------------
    b_poll = Recap.poll(ctx)
    record("B_poll_true_with_work", b_poll is True, {"poll": b_poll})

    # ---- C: STATIC phase wrappers write a deform cache --------------
    ok_s, err_s = sd.start_capture_for_objects(ctx, stat_objs)
    s_aborted = False
    if ok_s:
        s_aborted, _ = _drive_to_completion(sd.advance_capture)
    n_objs, n_frames_s = sd.finalize_capture(ctx) if ok_s else (0, 0)
    sd.cleanup_capture(ctx)
    s_has = pc2.has_static_deform_animation(s_obj)
    s_count = sd.object_deformation_frame_count(s_obj)
    record(
        "C_static_capture_writes_cache",
        ok_s and not s_aborted and n_objs == 1 and s_has
        and EXPECTED >= 1 and s_count == EXPECTED,
        {"ok": ok_s, "err": err_s, "aborted": s_aborted, "objects": n_objs,
         "frames_written": n_frames_s, "has_cache": s_has,
         "frame_count": s_count, "expected": EXPECTED},
    )

    # ---- D: PIN phase wrappers write a captured-anim cache ----------
    addon = groups.get_addon_data(scene)
    gi, pidx = pin_specs[0]
    pin_item = getattr(addon, "object_group_%d" % gi).pin_vertex_groups[pidx]
    ok_p, err_p = pc.start_capture_for_pins(ctx, pin_specs)
    p_aborted = False
    if ok_p:
        p_aborted, _ = _drive_to_completion(pc.advance_pin_capture)
    n_pins, n_frames_p = pc.finalize_pin_capture(ctx) if ok_p else (0, 0)
    pc.cleanup_pin_capture(ctx)
    p_has = pc2.has_pin_anim_pc2(c_obj, "PinVG")
    p_count = pc.pin_captured_frame_count(pin_item)
    p_flag = bool(pin_item.has_captured_anim)
    record(
        "D_pin_capture_writes_cache",
        ok_p and not p_aborted and n_pins == 1 and p_has and p_flag
        and EXPECTED >= 1 and p_count == EXPECTED,
        {"ok": ok_p, "err": err_p, "aborted": p_aborted, "pins": n_pins,
         "frames_written": n_frames_p, "has_cache": p_has,
         "has_captured_anim": p_flag, "frame_count": p_count,
         "expected": EXPECTED},
    )

    # ---- G: Clear All Deformations removes every cache --------------
    # Caches from C (static) and D (pin) are now on disk. The new
    # solver.clear_all_deformations operator must drop both in one pass.
    pre_s = pc2.has_static_deform_animation(s_obj)
    pre_p = pc2.has_pin_anim_pc2(c_obj, "PinVG")
    clr_poll = bool(bpy.ops.solver.clear_all_deformations.poll())
    bpy.ops.solver.clear_all_deformations("EXEC_DEFAULT")
    post_s = pc2.has_static_deform_animation(s_obj)
    post_p = pc2.has_pin_anim_pc2(c_obj, "PinVG")
    post_flag = bool(pin_item.has_captured_anim)
    record(
        "G_clear_all_removes_caches",
        clr_poll and pre_s and pre_p
        and not post_s and not post_p and not post_flag,
        {"poll": clr_poll, "pre_static": pre_s, "pre_pin": pre_p,
         "post_static": post_s, "post_pin": post_p,
         "post_pin_flag": post_flag},
    )

    # ---- H: Clear All poll False once nothing is cached -------------
    record(
        "H_clear_all_poll_false_when_empty",
        bpy.ops.solver.clear_all_deformations.poll() is False,
        {"poll": bool(bpy.ops.solver.clear_all_deformations.poll())},
    )

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
