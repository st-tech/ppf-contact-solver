# File: scenarios/bl_geonode_capture_range_frame_count.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Capture Deformation frame-range selection for the Geometry Nodes path.
#
# ``static_deform_ops._effective_frame_range`` (shared by both the
# STATIC-collider capture and the per-pin capture) caps the captured
# span at the last keyframe across every influencing action. Armature /
# Lattice / shape-key deformation settle at a keyframe, so that bound is
# exact. A Geometry Nodes modifier driven by Scene Time has no keyframes
# the action walker can see, so it falls through to a fallback. That
# fallback honors the solver's global ``state.frame_count`` (the number
# of frames the simulation actually consumes from the cache), not
# ``scene.frame_end`` (the viewport timeline, which the user may have
# shrunk independently of the simulation length).
#
# This scenario locks that behavior in three directions:
#
#   A. geonode_range_uses_frame_count:
#         a Set-Position GN deformer with NO keyframes, a short viewport
#         timeline (frame_end=12), and a longer simulation
#         (frame_count=50) -> the capture range is [1, 50], NOT [1, 12].
#   B. geonode_range_tracks_frame_count:
#         changing frame_count to 30 moves the upper bound to 30. Rules
#         out a coincidental match with any other scene quantity.
#   D. keyframe_overrides_frame_count:
#         positive control. Inserting a keyframe at frame 7 caps the
#         range at the last keyframe (7), proving frame_count is only the
#         no-keyframe fallback and the keyframed path is unchanged.
#
# Assertion-only: it calls ``_effective_frame_range`` directly with no
# server connection or solve, so it is fast and deterministic on every
# host.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


def _set_position_group():
    # A genuine deformer: writes vertex position with a constant offset,
    # carries no keyframes. Mirrors a Scene-Time GN setup as far as
    # ``_effective_frame_range`` can see (it walks actions, not nodes).
    ng = bpy.data.node_groups.new(
        "_pcs_capture_range_set_position", "GeometryNodeTree")
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
    sd = __import__(pkg + ".ui.dynamics.static_deform_ops",
                    fromlist=["_effective_frame_range"])
    utils = __import__(pkg + ".core.utils",
                       fromlist=["has_deforming_modifier_stack"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    ctx = bpy.context
    scene = ctx.scene

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Viewport timeline deliberately short; the simulation runs longer.
    # The geonode capture range must follow frame_count, not frame_end.
    scene.frame_start = 1
    scene.frame_end = 12
    state = groups.get_addon_data(scene).state
    state.frame_count = 50

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    plane = ctx.active_object
    plane.name = "CaptureRangePlane"
    mod = plane.modifiers.new("WaveGN", "NODES")
    mod.node_group = _set_position_group()
    ctx.view_layer.update()

    is_def = utils.has_deforming_modifier_stack(plane)
    has_anim = bool(plane.animation_data is not None
                    and plane.animation_data.action is not None)

    # ---- A: geonode (no keyframes) -> range from frame_count ----------
    fs, fe, src = sd._effective_frame_range(scene, plane)
    record(
        "A_geonode_range_uses_frame_count",
        is_def and not has_anim
        and fs == 1 and fe == 50
        and fe != scene.frame_end
        and "simulation frame count" in src,
        {"frame_start": fs, "frame_end": fe, "range_source": src,
         "scene_frame_end": scene.frame_end,
         "state_frame_count": state.frame_count,
         "is_deformer": is_def, "has_action": has_anim},
    )

    # ---- B: upper bound tracks frame_count (rules out coincidence) ----
    state.frame_count = 30
    fs2, fe2, src2 = sd._effective_frame_range(scene, plane)
    record(
        "B_geonode_range_tracks_frame_count",
        fs2 == 1 and fe2 == 30 and "simulation frame count (30)" in src2,
        {"frame_start": fs2, "frame_end": fe2, "range_source": src2,
         "state_frame_count": state.frame_count},
    )

    # ---- D: keyframed deformer still wins (last keyframe) -------------
    # A keyframe on any influencing action caps the capture at that
    # frame; frame_count is only the no-keyframe fallback. Set the frame
    # first, then insert (avoids the Blender 5.x per-index keyframe form).
    scene.frame_set(7)
    plane.keyframe_insert(data_path="location")
    scene.frame_set(1)
    fs3, fe3, src3 = sd._effective_frame_range(scene, plane)
    record(
        "D_keyframe_overrides_frame_count",
        fs3 == 1 and fe3 == 7 and "last keyframe (7)" in src3,
        {"frame_start": fs3, "frame_end": fe3, "range_source": src3,
         "state_frame_count": state.frame_count},
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
