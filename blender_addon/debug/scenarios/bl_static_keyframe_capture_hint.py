# File: scenarios/bl_static_keyframe_capture_hint.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A STATIC collider with only rigid loc/rot/scale keyframes does NOT need a
# Capture Deformation pass: the encoder samples those keyframes via the
# transform-animation path and the solver moves the collider per frame. So
# the per-object "Capture Deformation" button is intentionally disabled for
# it. To stop the greyed-out button from reading as broken, the panel shows
# an explanatory hint ("Keyframe animation transfers automatically; capture
# is for deformers").
#
# The hint branch in panels.py fires when, for the selected STATIC object:
#   not has_deform_cache AND not object_needs_deformation_capture(obj)
#   AND has_transform_fcurves(obj).
# The rig's hidden window doesn't reliably trigger sidebar draws, so this
# verifies those gating predicates directly (equivalent to verifying the
# branch, matching bl_pin_capture_deformation's G2 approach):
#
#   A. keyframed_static_has_transform_fcurves:
#         has_transform_fcurves is True for a loc-keyframed cube.
#   B. keyframed_static_not_capturable:
#         object_needs_deformation_capture is False for it (button disabled
#         -> the hint branch is what renders).
#   C. deformer_static_is_capturable:
#         a cube with a position-writing Geometry Nodes modifier flips
#         object_needs_deformation_capture back to True (button enabled,
#         the "Deforming modifier detected" branch, NOT the hint).
#   D. plain_static_no_hint:
#         a plain cube (no keyframes, no deformer) has neither transform
#         fcurves nor capture need, so neither the button-enabled branch nor
#         the hint fires.
#
# Assertion-only: no server connection or solve, fast and deterministic.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback
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
    sp = ng.nodes.new("GeometryNodeSetPosition")
    sp.inputs["Offset"].default_value = (0.0, 0.0, 0.25)
    ng.links.new(gin.outputs[0], sp.inputs["Geometry"])
    ng.links.new(sp.outputs["Geometry"], gout.inputs[0])
    return ng


try:
    utils = __import__(pkg + ".core.utils",
                       fromlist=["has_transform_fcurves"])
    sd = __import__(pkg + ".ui.dynamics.static_deform_ops",
                    fromlist=["object_needs_deformation_capture"])
    solver_api = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
    has_xform = utils.has_transform_fcurves
    needs_cap = sd.object_needs_deformation_capture
    ctx = bpy.context

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    try:
        bpy.ops.object.delete_all_groups()
    except Exception:
        pass
    ctx.scene.frame_start = 1
    ctx.scene.frame_end = 12

    static_group = solver_api.create_group("StaticG", "STATIC")

    # --- Cube with rigid location keyframes (the reported case) -------
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    kf = ctx.active_object
    kf.name = "KeyframedCube"
    kf.location = (0.0, 0.0, 0.0)
    kf.keyframe_insert(data_path="location", frame=1)
    kf.location = (0.0, 0.0, 2.0)
    kf.keyframe_insert(data_path="location", frame=5)
    static_group.add(kf.name)

    # --- Cube with a position-writing Geometry Nodes deformer --------
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 0.0, 0.0))
    dfm = ctx.active_object
    dfm.name = "DeformerCube"
    dfm.modifiers.new("Deform", "NODES").node_group = _new_geo_group(
        "_hint_set_position"
    )
    static_group.add(dfm.name)

    # --- Plain static cube: no animation, no deformer ----------------
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(6.0, 0.0, 0.0))
    plain = ctx.active_object
    plain.name = "PlainCube"
    static_group.add(plain.name)

    ctx.view_layer.update()

    kf_xform = has_xform(kf)
    kf_needs = needs_cap(kf, ctx)
    record(
        "A_keyframed_static_has_transform_fcurves",
        kf_xform is True,
        {"has_transform_fcurves": kf_xform},
    )
    record(
        "B_keyframed_static_not_capturable",
        kf_needs is False,
        {"object_needs_deformation_capture": kf_needs,
         "expected_panel_branch": "keyframe_auto_transfer_hint"},
    )

    dfm_needs = needs_cap(dfm, ctx)
    dfm_xform = has_xform(dfm)
    record(
        "C_deformer_static_is_capturable",
        dfm_needs is True and dfm_xform is False,
        {"object_needs_deformation_capture": dfm_needs,
         "has_transform_fcurves": dfm_xform,
         "expected_panel_branch": "deforming_modifier_detected"},
    )

    plain_xform = has_xform(plain)
    plain_needs = needs_cap(plain, ctx)
    record(
        "D_plain_static_no_hint",
        plain_xform is False and plain_needs is False,
        {"has_transform_fcurves": plain_xform,
         "object_needs_deformation_capture": plain_needs,
         "expected_panel_branch": "none"},
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
