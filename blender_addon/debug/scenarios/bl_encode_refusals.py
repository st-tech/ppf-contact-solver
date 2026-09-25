# File: scenarios/bl_encode_refusals.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Settings the solve cannot honor are REFUSED at encode, by name, rather than
# shipped as something the artist did not author.
#
# Each check covers a path that shipped a plausible result instead of an
# error: a keyframe the encoder sampled and the solver never read, a
# magnitude dropped because its direction was zero, a rest shape two features
# both rewrote, a group solved at a radius some of its grains were not seeded
# for. Encode-only: no build and no run.
#
# Subtests:
#   A. inactive_momentum_curve_refused: Inactive Momentum Frames is not
#      animatable, and a curve a saved file already carries on it is refused
#      naming the path, since the solver reads one duration.
#   B. zero_wind_direction_refused: a wind strength with a zero direction is
#      refused naming the wind; a zero strength with it encodes.
#   C. velocity_keyframes_encode_and_refuse: a directed translational velocity
#      keyframe encodes into the velocity schedule, and a zero direction with
#      a nonzero speed is refused naming the frame and the object.
#   D. solid_elastic_keyframe_refused: keyframing a SOLID group's Young's
#      modulus is refused (its elastic material lives on tetrahedra, which
#      carry no per-frame table), while keyframing its friction encodes.
#   E. plasticity_with_tracked_rest_shape_refused: a SOLID group with
#      Plasticity on and a pin tracking its captured rest shape is refused
#      naming the pin, since both rewrite the rest shape.
#   F. partial_tracking_pin_refused: Track Rest-Pose Deformation on a pin
#      that holds only some vertices is refused naming the pin.
#   G. sand_mixed_grain_radii_refused: a SAND group whose objects were
#      converted at different grain radii is refused naming both, since the
#      group is solved at one radius.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It encodes and never asks the solver to step.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def encode_error():
    try:
        params_mod.encode_param(bpy.context)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def group_of(api_group):
    return dh.groups.get_active_group_by_uuid(bpy.context.scene, api_group.uuid)


def point_cloud(name, radius, x):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata([(x, 0.0, 0.0), (x + 0.1, 0.0, 0.0)], [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    # What Convert to Particle Mesh stamps; a SAND group takes nothing else.
    obj["particle_mesh"] = True
    obj["grain_radius"] = radius
    return obj


try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["encode_param", "_build_param_dict"])
    plane = dh.reset_scene_to_pinned_plane(name="Sheet")
    dh.save_blend(PROBE_DIR, "encode_refusals.blend")
    root = dh.configure_state(project_name="encode_refusals", frame_count=6)
    state = root.state
    scene = bpy.context.scene

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")
    baseline = encode_error()
    dh.log(f"baseline error: {baseline!r}")

    # ---- A: a stored curve on a setting that cannot animate ---------------
    rna = state.bl_rna.properties["inactive_momentum_frames"]
    not_animatable = not rna.is_animatable
    action = bpy.data.actions.new("LegacyMomentum")
    slot = action.slots.new(id_type="SCENE", name="Scene")
    layer = action.layers.new("Layer")
    strip = layer.strips.new(type="KEYFRAME")
    bag = strip.channelbag(slot, ensure=True)
    curve = bag.fcurves.new("zozo_contact_solver.state.inactive_momentum_frames")
    curve.keyframe_points.insert(1, 5.0)
    curve.keyframe_points.insert(4, 10.0)
    scene.animation_data_create()
    scene.animation_data.action = action
    scene.animation_data.action_slot = slot
    err_a = encode_error()
    scene.animation_data_clear()
    dh.record(
        "A_inactive_momentum_curve_refused",
        baseline == "" and not_animatable
        and "inactive_momentum_frames" in err_a
        and "cannot change over it" in err_a
        and encode_error() == "",
        {"baseline": baseline, "not_animatable": not_animatable,
         "error": err_a[:300]},
    )

    # ---- B: a wind strength with no direction -----------------------------
    state.wind_direction = (0.0, 0.0, 0.0)
    state.wind_strength = 3.0
    err_b = encode_error()
    state.wind_strength = 0.0
    err_b_zero = encode_error()
    state.wind_direction = (1.0, 0.0, 0.0)
    dh.record(
        "B_zero_wind_direction_refused",
        err_b.startswith("ValueError: Wind has a zero-length direction")
        and "3.0" in err_b and err_b_zero == "",
        {"error": err_b[:300], "zero_strength_error": err_b_zero[:300]},
    )

    # ---- C: velocity keyframes, directed and not --------------------------
    assigned = group_of(cloth).assigned_objects[0]
    kf = assigned.velocity_keyframes.add()
    kf.frame = 3
    kf.direction = (0.0, 0.0, 1.0)
    kf.speed = 2.0
    kf.enable_translational = True
    err_c_ok = encode_error()
    schedule = []
    if not err_c_ok:
        built = params_mod._build_param_dict(bpy.context)
        for params, _names, uuids in built["group"]:
            schedule = (params.get("velocity-schedule") or {}).get(assigned.uuid, [])
    kf.direction = (0.0, 0.0, 0.0)
    err_c = encode_error()
    assigned.velocity_keyframes.clear()
    dh.record(
        "C_velocity_keyframes_encode_and_refuse",
        err_c_ok == "" and len(schedule) == 1
        and "The velocity keyframe at frame 3 of 'Sheet'" in err_c
        and "zero-length direction" in err_c,
        {"ok_error": err_c_ok[:300], "schedule": str(schedule)[:200],
         "error": err_c[:300]},
    )

    # ---- D: SOLID elastic keyframes are refused, friction is not ----------
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(3.0, 0.0, 0.0))
    block = bpy.context.active_object
    block.name = "Block"
    everything = block.vertex_groups.new(name="Everything")
    everything.add(list(range(len(block.data.vertices))), 1.0, "REPLACE")
    some = block.vertex_groups.new(name="Some")
    some.add([0, 1], 1.0, "REPLACE")
    solid_api = dh.api.solver.create_group("Solid", "SOLID")
    solid_api.add(block.name)
    solid = group_of(solid_api)
    solid.solid_young_modulus = 1000.0
    solid.keyframe_insert(data_path="solid_young_modulus", frame=1)
    solid.solid_young_modulus = 5000.0
    solid.keyframe_insert(data_path="solid_young_modulus", frame=4)
    err_d = encode_error()
    scene.animation_data_clear()
    solid.friction = 0.2
    solid.keyframe_insert(data_path="friction", frame=1)
    solid.friction = 0.6
    solid.keyframe_insert(data_path="friction", frame=4)
    err_d_friction = encode_error()
    scene.animation_data_clear()
    dh.record(
        "D_solid_elastic_keyframe_refused",
        "solid_young_modulus" in err_d and "not animated" in err_d
        and "Solid" in err_d and err_d_friction == "",
        {"error": err_d[:300], "friction_error": err_d_friction[:300]},
    )

    # ---- E: plasticity and a tracked rest shape ---------------------------
    solid_api.create_pin(block.name, "Everything")
    tracked = solid.pin_vertex_groups[len(solid.pin_vertex_groups) - 1]
    tracked.track_rest_pose_deformation = True
    tracked.has_captured_anim = True
    solid.enable_plasticity = True
    err_e = encode_error()
    solid.enable_plasticity = False
    dh.record(
        "E_plasticity_with_tracked_rest_shape_refused",
        "Everything" in err_e and "Plasticity" in err_e
        and "Track Rest-Pose Deformation" in err_e,
        {"error": err_e[:300]},
    )
    tracked.track_rest_pose_deformation = False
    tracked.has_captured_anim = False

    # ---- F: tracking on a pin that holds only some vertices ---------------
    solid_api.create_pin(block.name, "Some")
    partial = solid.pin_vertex_groups[len(solid.pin_vertex_groups) - 1]
    partial.track_rest_pose_deformation = True
    partial.has_captured_anim = True
    err_f = encode_error()
    partial.track_rest_pose_deformation = False
    partial.has_captured_anim = False
    dh.record(
        "F_partial_tracking_pin_refused",
        "Pin 'Some' on 'Block'" in err_f
        and "hold every vertex" in err_f,
        {"error": err_f[:300]},
    )

    # ---- G: grains converted at different radii ---------------------------
    fine = point_cloud("FineGrains", 0.01, 6.0)
    coarse = point_cloud("CoarseGrains", 0.02, 8.0)
    sand_api = dh.api.solver.create_group("Sand", "SAND")
    sand_api.add(fine.name)
    err_g_one = encode_error()
    sand_api.add(coarse.name)
    err_g = encode_error()
    dh.record(
        "G_sand_mixed_grain_radii_refused",
        err_g_one == ""
        and "different radii" in err_g
        and "'FineGrains' at 0.01" in err_g
        and "'CoarseGrains' at 0.02" in err_g,
        {"one_radius_error": err_g_one[:300], "error": err_g[:300]},
    )
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
