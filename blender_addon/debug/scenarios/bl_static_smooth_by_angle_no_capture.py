# File: scenarios/bl_static_smooth_by_angle_no_capture.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard: a STATIC mesh collider whose only Geometry Nodes
# modifier is Blender 4.1+'s auto-added "Smooth by Angle" (it sets
# shade-smooth flags and moves NO vertices) must NOT be classified as a
# deforming static object, and must encode cleanly without a Capture
# Deformation pass.
#
# This invariant has flip-flopped twice. ``has_deforming_modifier_stack``
# originally listed ``NODES``; commit "drop NODES from declarative
# deforming-modifier set" removed it (relying on depsgraph sampling);
# commit "honor frame-1 geometry-nodes deform as solver input" re-added
# it, which re-broke the auto-added Smooth-by-Angle case (the encoder
# hard-failed with "has a deforming modifier stack ... but no
# deformation cache" on a static collider the artist never animated).
# The current rule keeps ``NODES`` in the set but only counts a GN
# modifier whose node group actually writes vertex positions
# (``_nodes_modifier_can_deform``). This scenario locks both directions:
#
#   A. smooth_by_angle_not_declarative_deformer:
#         ``has_deforming_modifier_stack`` is False for a shade-smooth-
#         only GN group.
#   B. smooth_by_angle_not_deforming_static:
#         ``is_deforming_static_object`` is False for the same object
#         (the depsgraph backstop agrees: the mesh shape is constant).
#   C. static_encodes_without_capture:
#         ``encode_obj`` succeeds with the object in a STATIC group and
#         emits it as a plain rest mesh (no ``static_deform_animation``),
#         instead of raising the Capture-Deformation ValueError.
#   D. set_position_is_still_a_deformer:
#         positive control. Swapping the modifier to a Set-Position node
#         group flips both classifiers back to True, so the fix can't be
#         "solved" by declaring every GN non-deforming.
#
# Assertion-only: no server connection or solve, so it is fast and
# deterministic on every host.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback
import numpy as np
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


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


def _shade_smooth_group():
    # Mimics the auto-added "Smooth by Angle": sets the shade-smooth
    # flag, never touches position. Must NOT count as a deformer.
    ng, gin, gout = _new_geo_group("_pcs_guard_shade_smooth")
    ss = ng.nodes.new("GeometryNodeSetShadeSmooth")
    ng.links.new(gin.outputs[0], ss.inputs["Geometry"])
    ng.links.new(ss.outputs["Geometry"], gout.inputs[0])
    return ng


def _set_position_group():
    # Writes vertex position with a constant offset. A genuine deformer.
    ng, gin, gout = _new_geo_group("_pcs_guard_set_position")
    sp = ng.nodes.new("GeometryNodeSetPosition")
    sp.inputs["Offset"].default_value = (0.0, 0.0, 0.25)
    ng.links.new(gin.outputs[0], sp.inputs["Geometry"])
    ng.links.new(sp.outputs["Geometry"], gout.inputs[0])
    return ng


try:
    utils = __import__(pkg + ".core.utils",
                       fromlist=["has_deforming_modifier_stack",
                                 "is_deforming_static_object"])
    enc = __import__(pkg + ".core.encoder.mesh",
                     fromlist=["encode_obj", "_build_obj_data"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    solver_api = api_mod.solver
    ctx = bpy.context

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    ctx.scene.frame_start = 1
    ctx.scene.frame_end = 12

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    plane = ctx.active_object
    plane.name = "GuardBase"

    mod = plane.modifiers.new("Smooth by Angle", "NODES")
    mod.node_group = _shade_smooth_group()
    ctx.view_layer.update()

    # ---- A: shade-smooth GN is not a declarative deformer ------------
    decl = utils.has_deforming_modifier_stack(plane)
    record(
        "A_smooth_by_angle_not_declarative_deformer",
        decl is False,
        {"has_deforming_modifier_stack": decl,
         "modifier_type": mod.type,
         "node_group": mod.node_group.name},
    )

    # ---- B: ... and not a deforming static object --------------------
    deforming = utils.is_deforming_static_object(plane, ctx)
    record(
        "B_smooth_by_angle_not_deforming_static",
        deforming is False,
        {"is_deforming_static_object": deforming},
    )

    # ---- C: encodes as a plain STATIC mesh, no capture required ------
    static = solver_api.create_group("Static", "STATIC")
    static.add(plane.name)
    encode_err = None
    emitted = None
    try:
        tree = enc._build_obj_data(ctx, persist_topology_hash=False)
        for g in tree:
            for o in g.get("object", []):
                if o.get("name") == plane.name:
                    emitted = o
        # encode the full payload too, to exercise the CBOR path that
        # the upload uses.
        enc.encode_obj(ctx)
    except Exception as e:
        encode_err = type(e).__name__ + ": " + str(e)
    has_sd = bool(emitted is not None
                  and "static_deform_animation" in emitted)
    has_verts = bool(emitted is not None
                     and emitted.get("vert") is not None
                     and len(emitted.get("vert")) > 0)
    record(
        "C_static_encodes_without_capture",
        encode_err is None
        and emitted is not None
        and not has_sd
        and has_verts,
        {"encode_err": encode_err,
         "object_emitted": emitted is not None,
         "has_static_deform_animation": has_sd,
         "has_rest_verts": has_verts},
    )

    # ---- D: positive control - Set Position IS a deformer -----------
    mod.node_group = _set_position_group()
    ctx.view_layer.update()
    decl_pos = utils.has_deforming_modifier_stack(plane)
    deforming_pos = utils.is_deforming_static_object(plane, ctx)
    record(
        "D_set_position_is_still_a_deformer",
        decl_pos is True and deforming_pos is True,
        {"has_deforming_modifier_stack": decl_pos,
         "is_deforming_static_object": deforming_pos},
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
