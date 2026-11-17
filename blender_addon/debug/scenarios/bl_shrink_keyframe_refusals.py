# File: scenarios/bl_shrink_keyframe_refusals.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# What a keyframed shrink refuses to be set together with, at encode.
#
# A keyframed shrink rebuilds the rest shape every step. A strain limit is
# measured against the unshrunk rest shape, so a Shell keyframing its shrink
# may not enable one. Plasticity commits its creep into the rest shape, and the
# solver rebuilds every element's rest shape at once, so plasticity on ANY
# group beside a keyframed shrink would be erased every step. Each refusal has
# to name the groups and the setting, so the artist knows what to change.
#
#   A_keyframed_shrink_encodes      a Shell keying Shrink X, alone, encodes and
#                                   carries the schedule.
#   B_strain_limit_refused          the same Shell with Enable Strain Limit is
#                                   refused, naming the group, the property and
#                                   the strain limit.
#   C_plasticity_elsewhere_refused  a Solid with Plasticity beside the keyed
#                                   Shell is refused, naming both groups.
#   D_solid_shrink_encodes          the Solid keying its own Shrink, with
#                                   Plasticity off, encodes and carries it.
#   E_own_plasticity_refused        that Solid with Plasticity on is refused.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def encode():
    try:
        return dh.encode_payload()[1], ""
    except Exception as exc:
        return None, str(exc)


def series_of(param_bytes, object_name, key):
    if param_bytes is None:
        return None
    for params, objects, _uuids in dh.decode_addon_blob(param_bytes)["group"]:
        if object_name in objects:
            return (params.get("param-anim") or {}).get(key)
    return None


try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=0.4, location=(-0.5, 0.0, 0.5))
    plane = bpy.context.object
    plane.name = "Cloth"
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.1,
                                          location=(0.5, 0.0, 0.5))
    ball = bpy.context.object
    ball.name = "Body"
    root = dh.configure_state(project_name="shrink_refusals", frame_count=10,
                              gravity=(0.0, 0.0, 0.0))
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    addon_root = dh.groups.get_addon_data(scene)
    rnas = {}
    for label, name, kind in (("Shrinking", "Cloth", "SHELL"),
                              ("Plastic", "Body", "SOLID")):
        group = dh.api.solver.create_group(label, kind)
        group.add(name)
        rnas[name] = getattr(addon_root, "object_group_%d"
                             % groups_mod.get_group_slot_index(scene, group.uuid))
    cloth, body = rnas["Cloth"], rnas["Body"]

    # A: a keyed Shrink X alone encodes.
    cloth.shrink_x = 1.0
    cloth.keyframe_insert("shrink_x", frame=1)
    cloth.shrink_x = 0.5
    cloth.keyframe_insert("shrink_x", frame=10)
    scene.frame_set(1)
    blob, err = encode()
    sx = series_of(blob, "Cloth", "shrink-x")
    dh.record("A_keyframed_shrink_encodes",
              err == "" and sx is not None and abs(float(sx[-1]) - 0.5) < 1e-6,
              {"error": err[:300], "series": None if sx is None else [float(v) for v in sx]})

    # B: with a strain limit on the same Shell.
    cloth.enable_strain_limit = True
    _blob, err = encode()
    cloth.enable_strain_limit = False
    dh.record("B_strain_limit_refused",
              "Shrinking" in err and "shrink_x" in err and "Strain Limit" in err,
              {"error": err[:400]})

    # C: with plasticity on another group.
    body.enable_plasticity = True
    _blob, err = encode()
    body.enable_plasticity = False
    dh.record("C_plasticity_elsewhere_refused",
              "Plastic" in err and "Shrinking" in err and "Plasticity" in err,
              {"error": err[:400]})

    # D: the Solid keys its own Shrink.
    body.shrink = 1.0
    body.keyframe_insert("shrink", frame=1)
    body.shrink = 0.8
    body.keyframe_insert("shrink", frame=10)
    scene.frame_set(1)
    blob, err = encode()
    sb = series_of(blob, "Body", "shrink")
    dh.record("D_solid_shrink_encodes",
              err == "" and sb is not None and abs(float(sb[-1]) - 0.8) < 1e-6,
              {"error": err[:300], "series": None if sb is None else [float(v) for v in sb]})

    # E: and its own plasticity.
    body.enable_plasticity = True
    _blob, err = encode()
    body.enable_plasticity = False
    dh.record("E_own_plasticity_refused",
              "Plastic" in err and "Plasticity" in err and "shrink" in err,
              {"error": err[:400]})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
