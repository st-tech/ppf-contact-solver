# File: scenarios/bl_pin_make_keyframe_writes_fcurves.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Pin keyframe animation is authored as native vertex-co fcurves on
# the mesh action, signaled by an ``EMBEDDED_MOVE`` op pinned to
# slot 0 of ``pin_item.operations``. The encoder reads the fcurves
# directly at Transfer time and emits a sparse ``pin_anim`` track
# spliced at ``embedded_move_index``.
#
# There is no dense-PC2 path for SHELL/SOLID pins anymore: the old
# ``_pininput.pc2`` cache and the ``set_animation`` API are gone.
# STATIC mesh colliders have their own separate dense path
# (``_staticdeform.pc2`` via Capture Deformation, consumed by
# ``encoder/mesh.py``), exercised by ``bl_static_deform_anim``.
#
# Subtests:
#   A. make_inserts_vertex_co_fcurves: Make Keyframe creates three
#      vertex-co fcurves per pinned vertex on the mesh's action and
#      adds the EMBEDDED_MOVE sentinel at slot 0 of operations.
#   B. delete_all_removes_fcurves: Delete All Keyframes wipes every
#      ``vertices[N].co`` fcurve AND removes the EMBEDDED_MOVE op.
#   C. encoder_emits_sparse_pin_anim_from_fcurves: with fcurves
#      authored, the encoder emits a per-vertex ``pin_anim`` with one
#      (time, position) sample per authored keyframe and sets
#      ``embedded_move_index = 0``.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, os, time, traceback, re
import numpy as np
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


_RGX = re.compile(r"vertices\[(\d+)\]\.co$")


def collect_vertex_co_fcurves(obj):
    ad = obj.data.animation_data
    action = ad.action if ad else None
    out = []
    if action is None:
        return out
    if hasattr(action, "layers") and len(action.layers) > 0:
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    cb = strip.channelbag(slot)
                    if cb is None:
                        continue
                    for fc in cb.fcurves:
                        m = _RGX.match(fc.data_path)
                        if m is not None and 0 <= fc.array_index < 3:
                            out.append((int(m.group(1)), fc.array_index,
                                        len(fc.keyframe_points)))
    elif hasattr(action, "fcurves"):
        for fc in action.fcurves:
            m = _RGX.match(fc.data_path)
            if m is not None and 0 <= fc.array_index < 3:
                out.append((int(m.group(1)), fc.array_index,
                            len(fc.keyframe_points)))
    return out


try:
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_addon_data"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    encoder_pin = __import__(pkg + ".core.encoder.pin",
                             fromlist=["_encode_pin_config"])
    solver_api = api_mod.solver

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.frame_end = 30

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "FcPlane"
    n_verts = len(plane.data.vertices)
    vg = plane.vertex_groups.new(name="AllPin")
    vg.add(list(range(n_verts)), 1.0, "REPLACE")

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")

    root = groups_mod.get_addon_data(bpy.context.scene)
    group = root.object_group_0
    state = root.state
    group.pin_vertex_groups_index = 0
    pin_item = group.pin_vertex_groups[0]

    # ---- A: Make Keyframe writes vertex-co fcurves + EMBEDDED_MOVE
    bpy.context.scene.frame_set(5)
    for v in plane.data.vertices:
        v.co.x += 0.3
    plane.data.update()
    bpy.ops.object.make_pin_keyframe(group_index=0)

    bpy.context.scene.frame_set(15)
    for v in plane.data.vertices:
        v.co.x += 0.4
    plane.data.update()
    bpy.ops.object.make_pin_keyframe(group_index=0)

    fcs_after_make = collect_vertex_co_fcurves(plane)
    n_axes = {a for (_, a, _) in fcs_after_make}
    n_verts_covered = {v for (v, _, _) in fcs_after_make}
    keyframes_per_curve = sorted({k for (_, _, k) in fcs_after_make})
    op_types = [op.op_type for op in pin_item.operations]
    record(
        "A_make_inserts_vertex_co_fcurves",
        len(fcs_after_make) == 3 * n_verts
        and n_axes == {0, 1, 2}
        and n_verts_covered == set(range(n_verts))
        and keyframes_per_curve == [2]
        and op_types == ["EMBEDDED_MOVE"],
        {
            "n_fcurves": len(fcs_after_make),
            "axes": sorted(n_axes),
            "n_verts_covered": len(n_verts_covered),
            "keyframes_per_curve": keyframes_per_curve,
            "op_types": op_types,
        },
    )

    # ---- C: encoder emits sparse pin_anim from fcurves ------------
    try:
        cfg_dict = encoder_pin._encode_pin_config(
            bpy.context, [group], state,
        )
        encode_err = None
    except Exception as e:
        cfg_dict = {}
        encode_err = type(e).__name__ + ": " + str(e)
    any_vert_cfg = None
    for obj_cfg in cfg_dict.values():
        for vert_cfg in obj_cfg.values():
            any_vert_cfg = vert_cfg
            break
        if any_vert_cfg is not None:
            break
    pin_anim_dict = (
        any_vert_cfg.get("pin_anim", {}) if any_vert_cfg else {}
    )
    first_track = next(iter(pin_anim_dict.values()), None) if pin_anim_dict else None
    sample_times = list(first_track.get("time", [])) if first_track else []
    record(
        "C_encoder_emits_sparse_pin_anim_from_fcurves",
        encode_err is None
        and any_vert_cfg is not None
        and any_vert_cfg.get("embedded_move_index") == 0
        and len(sample_times) == 2,
        {
            "encode_err": encode_err,
            "embedded_move_index": (
                any_vert_cfg.get("embedded_move_index")
                if any_vert_cfg else None
            ),
            "pin_anim_vertex_count": len(pin_anim_dict),
            "sample_count": len(sample_times),
            "sample_times": sample_times,
        },
    )

    # ---- B: Delete All Keyframes removes the fcurves + EMBEDDED_MOVE
    bpy.ops.object.delete_pin_keyframes(group_index=0)
    fcs_after_delete = collect_vertex_co_fcurves(plane)
    op_types_after_delete = [op.op_type for op in pin_item.operations]
    record(
        "B_delete_all_removes_fcurves",
        fcs_after_delete == []
        and "EMBEDDED_MOVE" not in op_types_after_delete,
        {
            "n_remaining_fcurves": len(fcs_after_delete),
            "op_types": op_types_after_delete,
        },
    )

    log("checks=" + str(len(result["checks"])) + " done")
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
