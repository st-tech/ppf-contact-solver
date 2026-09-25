# File: scenarios/bl_static_transform_sampling.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A moving STATIC collider reaches the solver as its world transform sampled
# at EVERY frame of the solve, and the samples are Blender's own evaluation.
#
# The solver interpolates one shared, shortest-path curve between the samples
# it is sent. Sent only the keys, it drove every channel with one curve's
# interpolation, sampled a key between frames at the frame below it, clamped a
# key before the start onto t=0, lost a full turn between two keys (both keys
# are the same rotation), and never saw motion a parent, constraint or driver
# gave the collider. Sampled at every frame, every one of those is whatever
# Blender says it is at that frame. Encode-only: no build and no run.
#
# Subtests:
#   A. every_frame_is_blenders_pose: a cube keyed with a LINEAR location, an
#      auto-clamped 3-key BEZIER rotation, an ELASTIC-eased scale, a key at a
#      fractional frame and a key before the starting frame ships exactly one
#      sample per solve frame, each equal to Blender's world matrix at that
#      frame.
#   B. full_turn_between_two_keys_turns: two keys a full turn apart ship
#      intermediate samples that are rotated, where the two keys alone are the
#      same rotation.
#   C. parent_motion_is_motion: a cube with no curves of its own, parented to
#      a keyed empty, ships its parent's motion, and is not sent to Capture
#      Deformation: its mesh keeps its shape.
#   D. keyed_but_motionless_ships_nothing: a cube keyed twice at one place
#      ships no animation.
#   E. shared_action_reads_its_own_slot: two cubes share one action, the
#      first animated through its slot and the second assigned an empty slot;
#      the second has no transform curves and ships no animation.
#   F. the_payload_carries_the_samples: the encoded DATA payload holds, for
#      the cube of A, one transform sample per solve frame.
#   G. sheared_motion_is_refused: a moving cube rotated under a parent with a
#      non-uniform scale is sheared, which a location, rotation and scale
#      cannot carry; the encode refuses it naming the object and the frame.
#   H. ops_under_parent_motion_are_refused: Static ops on the cube of C,
#      whose motion comes from its parent and so raises no "ops will be
#      ignored" label, are refused rather than dropped.
#   I. panel_says_which_motion_needs_a_capture: the STATIC panel's line under
#      Capture Deformation says a capture is needed only for a mesh that
#      changes shape (a Displace modifier), that parent motion transfers on its
#      own with a capture optional (the cube of C), and that keyframe motion
#      transfers on its own (the cube of A).

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It encodes and never asks the solver to step.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import math
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

FRAME_COUNT = 24
START = 3


def _pose(obj):
    loc, quat, scale = transform.world_matrix(obj).decompose()
    return ([float(loc.x), float(loc.y), float(loc.z)],
            [float(quat.w), float(quat.x), float(quat.y), float(quat.z)],
            [float(scale.x), float(scale.y), float(scale.z)])


def _quat_angle(a, b):
    # The rotation angle between two unit quaternions, in a form that stays
    # accurate near zero: acos of the dot product turns the float32 rounding
    # of a unit quaternion's norm (about 1e-7) into an angle near 1e-3.
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    a = [x / na for x in a]
    b = [x / nb for x in b]
    sign = 1.0 if sum(x * y for x, y in zip(a, b)) >= 0.0 else -1.0
    diff = math.sqrt(sum((x - sign * y) ** 2 for x, y in zip(a, b)))
    total = math.sqrt(sum((x + sign * y) ** 2 for x, y in zip(a, b)))
    # |a - b| = 2 sin(angle / 4) and |a + b| = 2 cos(angle / 4).
    return 4.0 * math.atan2(diff, total)


def _close(a, b, tol=1e-5):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


try:
    dh = DriverHelpers(pkg, result)
    utils = __import__(pkg + ".core.utils", fromlist=["get_transform_keyframes"])
    transform = __import__(pkg + ".core.transform", fromlist=["world_matrix"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    root = dh.configure_state(project_name="static_transform_sampling",
                              frame_count=FRAME_COUNT, frame_rate=24)
    root.state.frame_start = START
    scene = bpy.context.scene

    # ---- A: mixed channels, easing, a fractional key, a key before START --
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 0.0, 0.0))
    mixed = bpy.context.active_object
    mixed.name = "MixedKeys"
    mixed.keyframe_insert("location", index=0, frame=START)
    mixed.location.x = 3.0
    mixed.keyframe_insert("location", index=0, frame=START + 20)
    mixed.location.y = -1.0
    mixed.keyframe_insert("location", index=1, frame=1)
    mixed.location.y = 1.0
    mixed.keyframe_insert("location", index=1, frame=START + 9.5)
    for frame, angle in ((START, 0.0), (START + 8, 1.2), (START + 20, 0.4)):
        mixed.rotation_euler.z = angle
        mixed.keyframe_insert("rotation_euler", index=2, frame=frame)
    for frame, value in ((START, 1.0), (START + 20, 2.0)):
        mixed.scale = (value, value, value)
        mixed.keyframe_insert("scale", frame=frame)
    for fc in utils.get_id_fcurves(mixed):
        for kp in fc.keyframe_points:
            if fc.data_path == "location":
                kp.interpolation = "LINEAR"
            elif fc.data_path == "scale":
                kp.interpolation = "ELASTIC"
                kp.easing = "EASE_OUT"
            else:
                kp.interpolation = "BEZIER"
                kp.handle_left_type = "AUTO_CLAMPED"
                kp.handle_right_type = "AUTO_CLAMPED"
    g_static = dh.api.solver.create_group("Colliders", "STATIC")
    g_static.add(mixed.name)

    kf = utils.get_transform_keyframes(mixed, bpy.context, START, FRAME_COUNT) or {}
    expected = []
    for k in range(FRAME_COUNT):
        scene.frame_set(START + k)
        expected.append(_pose(mixed))
    scene.frame_set(START)
    offsets = kf.get("frame_offset", [])
    worst = 0.0
    matches = len(offsets) == FRAME_COUNT
    for k, (loc, quat, scale) in enumerate(expected[:len(offsets)]):
        got_loc = kf["translation"][k]
        got_quat = kf["quaternion"][k]
        got_scale = kf["scale"][k]
        worst = max(worst,
                    max(abs(a - b) for a, b in zip(got_loc, loc)),
                    _quat_angle(got_quat, quat),
                    max(abs(a - b) for a, b in zip(got_scale, scale)))
    matches = matches and worst <= 1e-5 and offsets == [float(k) for k in range(FRAME_COUNT)]
    dh.record("A_every_frame_is_blenders_pose", matches,
              {"n_samples": len(offsets), "worst_error": worst,
               "segments": sorted({s["interpolation"] for s in kf.get("segments", [])})})

    # ---- B: a full turn between two keys ----------------------------------
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(5.0, 0.0, 0.0))
    turn = bpy.context.active_object
    turn.name = "FullTurn"
    turn.rotation_euler.z = 0.0
    turn.keyframe_insert("rotation_euler", index=2, frame=START)
    turn.rotation_euler.z = 2.0 * math.pi
    turn.keyframe_insert("rotation_euler", index=2, frame=START + 20)
    g_static.add(turn.name)
    kb = utils.get_transform_keyframes(turn, bpy.context, START, FRAME_COUNT) or {}
    quats = kb.get("quaternion", [])
    turned = max((_quat_angle(q, quats[0]) for q in quats), default=0.0)
    dh.record("B_full_turn_between_two_keys_turns",
              len(quats) == FRAME_COUNT and turned > math.radians(90.0)
              and _quat_angle(quats[0], quats[min(20, len(quats) - 1)]) < 1e-4,
              {"max_turn_deg": math.degrees(turned), "n": len(quats)})

    # ---- C: motion from a parent ------------------------------------------
    bpy.ops.object.empty_add(location=(0.0, 5.0, 0.0))
    carrier = bpy.context.active_object
    carrier.name = "Carrier"
    carrier.keyframe_insert("location", frame=START)
    carrier.location.z = 2.0
    carrier.keyframe_insert("location", frame=START + 10)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 5.0, 0.0))
    child = bpy.context.active_object
    child.name = "Carried"
    child.parent = carrier
    child.matrix_parent_inverse = carrier.matrix_world.inverted()
    g_static.add(child.name)
    kc = utils.get_transform_keyframes(child, bpy.context, START, FRAME_COUNT)
    heights = [t[1] for t in (kc or {}).get("translation", [])]
    deforms = utils.static_mesh_deforms(child, bpy.context)
    dh.record("C_parent_motion_is_motion",
              kc is not None and len(heights) == FRAME_COUNT
              and abs((heights[10] - heights[0]) - 2.0) < 1e-5
              and not deforms,
              {"animated": kc is not None, "heights": heights[:12],
               "static_mesh_deforms": deforms})

    # ---- D: keyed but motionless ------------------------------------------
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, -5.0, 0.0))
    still = bpy.context.active_object
    still.name = "KeyedStill"
    still.keyframe_insert("location", frame=START)
    still.keyframe_insert("location", frame=START + 10)
    g_static.add(still.name)
    kd = utils.get_transform_keyframes(still, bpy.context, START, FRAME_COUNT)
    dh.record("D_keyed_but_motionless_ships_nothing", kd is None,
              {"animation": None if kd is None else len(kd.get("frame_offset", []))})

    # ---- E: a shared action read through each object's own slot -----------
    action = bpy.data.actions.new("SharedSlots")
    slot_a = action.slots.new(id_type="OBJECT", name="SlotA")
    slot_b = action.slots.new(id_type="OBJECT", name="SlotB")
    layer = action.layers.new("Layer")
    strip = layer.strips.new(type="KEYFRAME")
    bag = strip.channelbag(slot_a, ensure=True)
    curve = bag.fcurves.new("location", index=0)
    curve.keyframe_points.insert(START, -5.0)
    curve.keyframe_points.insert(START + 10, -3.0)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(-5.0, 0.0, 0.0))
    slotted = bpy.context.active_object
    slotted.name = "SlotAnimated"
    slotted.animation_data_create()
    slotted.animation_data.action = action
    slotted.animation_data.action_slot = slot_a
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(-5.0, 3.0, 0.0))
    bystander = bpy.context.active_object
    bystander.name = "SlotBystander"
    bystander.animation_data_create()
    bystander.animation_data.action = action
    bystander.animation_data.action_slot = slot_b
    g_static.add(slotted.name)
    g_static.add(bystander.name)
    ke_a = utils.get_transform_keyframes(slotted, bpy.context, START, FRAME_COUNT)
    ke_b = utils.get_transform_keyframes(bystander, bpy.context, START, FRAME_COUNT)
    dh.record("E_shared_action_reads_its_own_slot",
              ke_a is not None and ke_b is None
              and not utils.has_transform_fcurves(bystander)
              and utils.has_transform_fcurves(slotted),
              {"animated_a": ke_a is not None, "animated_b": ke_b is not None,
               "curves_b": len(utils.get_id_fcurves(bystander))})

    # ---- F: the encoded payload -------------------------------------------
    data_bytes, _param = dh.encode_payload()
    data = dh.decode_addon_blob(data_bytes)
    shipped = {}
    for group in data:
        for info in group.get("object", []):
            anim = info.get("transform_animation")
            if anim is not None:
                shipped[info.get("name")] = len(anim.get("frame_offset", []))
    dh.record("F_the_payload_carries_the_samples",
              shipped.get("MixedKeys") == FRAME_COUNT
              and shipped.get("FullTurn") == FRAME_COUNT
              and shipped.get("Carried") == FRAME_COUNT
              and "KeyedStill" not in shipped
              and "SlotBystander" not in shipped,
              {"shipped": shipped})

    # ---- G: a sheared moving collider --------------------------------------
    bpy.ops.object.empty_add(location=(0.0, -9.0, 0.0))
    stretcher = bpy.context.active_object
    stretcher.name = "Stretcher"
    stretcher.scale = (2.0, 1.0, 1.0)
    stretcher.keyframe_insert("location", frame=START)
    stretcher.location.z = 1.0
    stretcher.keyframe_insert("location", frame=START + 10)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, -9.0, 0.0))
    skewed = bpy.context.active_object
    skewed.name = "Skewed"
    skewed.parent = stretcher
    skewed.matrix_parent_inverse.identity()
    skewed.location = (0.0, 0.0, 0.0)
    skewed.rotation_euler.z = math.radians(30.0)
    g_static.add(skewed.name)
    sheared_error = ""
    try:
        dh.encode_payload()
    except ValueError as exc:
        sheared_error = str(exc)
    dh.record("G_sheared_motion_is_refused",
              "STATIC object 'Skewed' is sheared at frame" in sheared_error
              and "Capture Deformation" in sheared_error,
              {"error": sheared_error[:300]})

    # ---- H: Static ops displaced by a parent's motion ---------------------
    g_static.remove(skewed.name)
    group_pg = dh.groups.get_active_group_by_uuid(scene, g_static.uuid)
    carried = next(a for a in group_pg.assigned_objects if a.name == "Carried")
    op = carried.static_ops.add()
    op.op_type = "MOVE_BY"
    ops_error = ""
    try:
        dh.encode_payload()
    except ValueError as exc:
        ops_error = str(exc)
    carried.static_ops.clear()
    cleared_error = ""
    try:
        dh.encode_payload()
    except ValueError as exc:
        cleared_error = str(exc)
    dh.record("H_ops_under_parent_motion_are_refused",
              "STATIC object 'Carried'" in ops_error
              and "has Static ops and is also moved by its parent" in ops_error
              and cleared_error == "",
              {"error": ops_error[:300], "after_clear": cleared_error[:300]})

    # ---- I: the panel's capture line ---------------------------------------
    sd_ops = __import__(pkg + ".ui.dynamics.static_deform_ops",
                        fromlist=["static_capture_hint"])
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 12.0, 0.0))
    displaced = bpy.context.active_object
    displaced.name = "Displaced"
    displaced.modifiers.new("Displace", "DISPLACE")
    hints = {
        name: sd_ops.static_capture_hint(bpy.data.objects[name], bpy.context)
        for name in ("Displaced", "Carried", "MixedKeys")
    }
    dh.record("I_panel_says_which_motion_needs_a_capture",
              hints["Displaced"] == "Deforming modifier detected; capture to encode"
              and (hints["Carried"] or "").startswith("Parent or constraint motion")
              and "capture is optional" in (hints["Carried"] or "")
              and (hints["MixedKeys"] or "").startswith("Keyframe animation"),
              {"hints": hints})
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
