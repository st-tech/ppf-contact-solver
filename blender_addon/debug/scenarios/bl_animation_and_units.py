# File: scenarios/bl_animation_and_units.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What the encoder samples over time, what it reads once, and which units the
# payload carries, checked at encode. Encode-only: no build and no run.
#
# Four contracts meet here. A keyframe button is offered exactly where the
# encoder samples a curve, and a curve anywhere else under the add-on's
# namespace is refused by path. The param payload, and so its hash, is the
# same wherever the playhead sits. Time Scale reaches every velocity keyframe,
# the one in effect at the starting frame included. World Scaling multiplies
# every length-based setting the solver reads raw.
#
# Subtests:
#   A. keyframe_offer_matches_sampling: across every add-on PropertyGroup the
#      animatable properties are exactly the sampled set (the SCENE_ANIM_KEYS
#      settings on State and the material sliders on a group), and a
#      keyframe on a formerly animatable read-once setting is rejected by
#      Blender.
#   B. unsampled_curve_refused_by_path: a stored curve on a read-once State
#      setting, a curve reaching into a group's pin collection, a driver on a
#      sampled setting and an NLA strip on a sampled setting are each refused
#      naming the data path; with them gone the scene encodes.
#   C. sampled_keys_encode: keyframed step size, gravity and group friction
#      still encode into their schedules.
#   D. param_hash_playhead_independent: with keyframed settings, a velocity
#      key, an invisible collider, a Time Scale and a non-default starting
#      frame, the param hash is identical with the playhead at the starting
#      frame and elsewhere, and the playhead is put back.
#   E. start_frame_velocity_time_scaled: the translational velocity key at the
#      starting frame is multiplied by Time Scale like a later key.
#   F. world_scaling_scales_lengths: constraint-ghat, a scene-unit length,
#      scales with World Scaling, while gravity and wind, physical constants
#      of the world the resized scene is simulated in, do not, static,
#      sampled and from the legacy keyframe list alike; neither do the vel.bin
#      velocity (scaled by the solver), air density or the step size.
#   G. legacy_first_keyframe_is_starting_frame: the first keyframe of a
#      legacy scene-parameter entry stands for the scene setting at the
#      starting frame, whatever frame it stores, in the viewport preview, in
#      the encoded schedule and in the conversion to F-curves; an entry whose
#      later keyframe sits at or before the starting frame is left
#      unconverted.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND: it encodes and never asks the solver to step, so
# it holds on any backend a rig host can start.
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


def scene_curve(path, keys, index=0):
    # A curve in the scene's ACTIVE action, in the slotted layout Blender 5
    # writes, which is the layout the encoder's samplers read.
    scene = bpy.context.scene
    ad = scene.animation_data or scene.animation_data_create()
    if ad.action is None:
        action = bpy.data.actions.new("ProbeAction")
        slot = action.slots.new(id_type="SCENE", name="Scene")
        layer = action.layers.new("Layer")
        layer.strips.new(type="KEYFRAME")
        ad.action = action
        ad.action_slot = slot
    bag = ad.action.layers[0].strips[0].channelbag(ad.action_slot, ensure=True)
    curve = bag.fcurves.new(path, index=index)
    for frame, value in keys:
        curve.keyframe_points.insert(frame, value)
    return curve


def clear_animation():
    bpy.context.scene.animation_data_clear()


def close(a, b, tol=1e-6):
    a = [float(x) for x in (a if hasattr(a, "__len__") else [a])]
    b = [float(x) for x in (b if hasattr(b, "__len__") else [b])]
    return len(a) == len(b) and all(
        abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b)
    )


def group_params(built, uuid):
    for params, _names, uuids in built["group"]:
        if uuid in uuids:
            return params
    return {}


def animatable_props():
    # (struct, property) for every animatable property of every add-on
    # PropertyGroup reachable from the scene namespace. Only PropertyGroup
    # structs are walked, so a pointer to a Blender ID would not pull in that
    # ID's own animatable properties.
    root = bpy.context.scene.zozo_contact_solver.bl_rna
    seen, offered = set(), set()

    def walk(struct):
        if struct.identifier in seen:
            return
        seen.add(struct.identifier)
        for prop in struct.properties:
            if prop.identifier == "rna_type":
                continue
            if prop.is_animatable:
                offered.add((struct.identifier, prop.identifier))
            if prop.type in ("POINTER", "COLLECTION"):
                fixed = prop.fixed_type
                if getattr(fixed.base, "identifier", "") == "PropertyGroup":
                    walk(fixed)

    walk(root)
    return offered, seen


try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["encode_param", "_build_param_dict",
                                      "compute_param_hash"])
    refusal = __import__(pkg + ".core.encoder.curve_refusal",
                         fromlist=["SAMPLED_STATE_PROPS", "SAMPLED_GROUP_PROPS"])
    plane = dh.reset_scene_to_pinned_plane(name="Sheet")
    dh.save_blend(PROBE_DIR, "animation_and_units.blend")
    root = dh.configure_state(project_name="animation_and_units", frame_count=6)
    state = root.state
    scene = bpy.context.scene

    cloth_api = dh.api.solver.create_group("Cloth", "SHELL")
    cloth_api.add(plane.name)
    cloth_api.create_pin(plane.name, "AllPin")
    cloth = group_of(cloth_api)
    slot = dh.groups.get_group_slot_index(scene, cloth.uuid)
    assigned = cloth.assigned_objects[0]
    baseline = encode_error()
    dh.log(f"baseline error: {baseline!r}")

    # ---- A: the keyframe offer is exactly the sampled set ------------------
    offered, walked = animatable_props()
    sampled = (
        {("State", p) for p in refusal.SAMPLED_STATE_PROPS}
        | {("ObjectGroup", p) for p in refusal.SAMPLED_GROUP_PROPS}
    )
    collider = state.invisible_colliders.add()
    vkf = assigned.velocity_keyframes.add()
    formerly = [
        (state, "time_scale"), (state, "frame_start"), (state, "frame_rate"),
        (state, "world_scaling"), (state, "constraint_ghat"),
        (state, "frame_count"), (collider, "position"), (collider, "contact_gap"),
        (vkf, "speed"), (cloth.pin_vertex_groups[0], "pull_strength"),
    ]
    accepted = []
    for owner, prop in formerly:
        try:
            owner.keyframe_insert(prop, frame=1)
            accepted.append(prop)
        except Exception:
            pass
    clear_animation()
    state.invisible_colliders.remove(len(state.invisible_colliders) - 1)
    assigned.velocity_keyframes.clear()
    dh.record(
        "A_keyframe_offer_matches_sampling",
        baseline == "" and offered == sampled and not accepted
        and {"State", "ObjectGroup", "PinOperation", "VelocityKeyframe",
             "InvisibleColliderItem"} <= walked,
        {"baseline": baseline,
         "offered_not_sampled": sorted(f"{s}.{p}" for s, p in offered - sampled),
         "sampled_not_offered": sorted(f"{s}.{p}" for s, p in sampled - offered),
         "keyframe_accepted_on": accepted, "structs_walked": len(walked)},
    )

    # ---- B: every unsampled curve is refused, naming its path --------------
    ts_path = "zozo_contact_solver.state.time_scale"
    scene_curve(ts_path, [(1, 1.0), (4, 0.5)])
    err_state = encode_error()
    clear_animation()

    pin_path = (f"zozo_contact_solver.object_group_{slot}"
                ".pin_vertex_groups[0].pull_strength")
    scene_curve(pin_path, [(1, 0.0), (4, 1.0)])
    err_nested = encode_error()
    clear_animation()

    gravity_path = "zozo_contact_solver.state.gravity_3d"
    scene.animation_data_create()
    scene.driver_add(gravity_path, 2)
    err_driver = encode_error()
    clear_animation()

    nla_action = bpy.data.actions.new("NlaProbe")
    nla_slot = nla_action.slots.new(id_type="SCENE", name="Scene")
    nla_layer = nla_action.layers.new("Layer")
    nla_strip_src = nla_layer.strips.new(type="KEYFRAME")
    nla_bag = nla_strip_src.channelbag(nla_slot, ensure=True)
    dt_path = "zozo_contact_solver.state.step_size"
    nla_curve = nla_bag.fcurves.new(dt_path)
    nla_curve.keyframe_points.insert(1, 0.01)
    nla_curve.keyframe_points.insert(4, 0.005)
    ad = scene.animation_data_create()
    track = ad.nla_tracks.new()
    track.name = "ProbeTrack"
    nla_strip = track.strips.new("ProbeStrip", 1, nla_action)
    if nla_strip.action_slot is None:
        nla_strip.action_slot = nla_slot
    err_nla = encode_error()
    clear_animation()
    err_after = encode_error()
    dh.record(
        "B_unsampled_curve_refused_by_path",
        f"'{ts_path}'" in err_state and "nothing samples this setting" in err_state
        and f"'{pin_path}'" in err_nested
        and "nothing samples this setting" in err_nested
        and f"'{gravity_path}'" in err_driver and "driver" in err_driver
        and f"'{dt_path}'" in err_nla and "NLA strip 'ProbeStrip'" in err_nla
        and err_after == "",
        {"state_error": err_state[:300], "nested_error": err_nested[:300],
         "driver_error": err_driver[:300], "nla_error": err_nla[:300],
         "after_clear_error": err_after[:300]},
    )

    # ---- C: sampled keys still encode --------------------------------------
    scene_curve(dt_path, [(1, 0.01), (6, 0.005)])
    scene_curve(gravity_path, [(1, 0.0), (6, -9.8)], index=2)
    cloth.friction = 0.2
    cloth.keyframe_insert(data_path="friction", frame=1)
    cloth.friction = 0.6
    cloth.keyframe_insert(data_path="friction", frame=6)
    err_c = encode_error()
    built_c = params_mod._build_param_dict(bpy.context) if not err_c else {}
    dyn_c = built_c.get("dyn_param", {})
    anim_c = group_params(built_c, assigned.uuid).get("param-anim", {})
    dh.record(
        "C_sampled_keys_encode",
        err_c == "" and len(dyn_c.get("dt", [])) >= 2
        and len(dyn_c.get("gravity", [])) >= 2 and "friction" in anim_c,
        {"error": err_c[:300], "dyn_keys": sorted(dyn_c),
         "dt_samples": len(dyn_c.get("dt", [])),
         "param_anim_keys": sorted(anim_c)},
    )

    # ---- D: the param hash does not move with the playhead -----------------
    # Keep C's curves, and add everything else the payload reads: a velocity
    # key, an invisible collider, a Time Scale, a frame rate and a starting
    # frame that is not the scene's.
    cloth_api.set_velocity(plane.name, direction=(1.0, 0.0, 0.0), speed=2.0,
                           frame=3)
    dh.api.solver.add_wall(position=(0.0, 0.0, -1.0), normal=(0.0, 0.0, 1.0))
    state.time_scale = 0.5
    state.use_scene_fps = False
    state.frame_rate = 50
    state.use_scene_frame_start = False
    state.frame_start = 2
    hashes = {}
    restored = True
    for playhead in (2, 5, 1, 30):
        scene.frame_set(playhead)
        hashes[playhead] = params_mod.compute_param_hash(bpy.context)
        restored = restored and scene.frame_current == playhead
    distinct = sorted(set(hashes.values()))
    dh.record(
        "D_param_hash_playhead_independent",
        len(distinct) == 1 and restored,
        {"hashes": {str(k): v[:12] for k, v in hashes.items()},
         "playhead_restored": restored},
    )
    clear_animation()
    assigned.velocity_keyframes.clear()
    state.invisible_colliders.clear()
    state.use_scene_frame_start = True
    state.time_scale = 1.0
    scene.frame_set(1)

    # ---- E: the starting-frame velocity key takes Time Scale ---------------
    cloth_api.set_velocity(plane.name, direction=(1.0, 0.0, 0.0), speed=2.0,
                           frame=1)
    cloth_api.set_velocity(plane.name, direction=(1.0, 0.0, 0.0), speed=2.0,
                           frame=4)
    readings = {}
    for scale in (1.0, 0.5):
        state.time_scale = scale
        params = group_params(params_mod._build_param_dict(bpy.context),
                              assigned.uuid)
        start = [float(x) for x in params["velocity"][assigned.uuid]]
        later = [float(x) for x in params["velocity-schedule"][assigned.uuid][0][1]]
        readings[scale] = {"start": start, "later": later}
    state.time_scale = 1.0
    assigned.velocity_keyframes.clear()
    dh.record(
        "E_start_frame_velocity_time_scaled",
        close(readings[1.0]["start"], [2.0, 0.0, 0.0])
        and close(readings[0.5]["start"], [1.0, 0.0, 0.0])
        and close(readings[0.5]["later"], readings[0.5]["start"])
        and close(readings[1.0]["later"], readings[1.0]["start"]),
        {"readings": {str(k): v for k, v in readings.items()}},
    )

    # ---- F: World Scaling reaches scene-unit lengths only -------------------
    state.gravity_3d = (0.0, 0.0, -9.8)
    state.wind_direction = (1.0, 0.0, 0.0)
    state.wind_strength = 3.0
    state.constraint_ghat = 0.002
    state.air_density = 0.001
    cloth_api.set_velocity(plane.name, direction=(0.0, 0.0, 1.0), speed=1.5,
                           frame=1)
    # Gravity sampled from its curve, wind from the legacy keyframe list, so
    # both animated paths are read.
    scene_curve(gravity_path, [(1, -9.8), (6, -4.9)], index=2)
    legacy = state.dyn_params.add()
    legacy.param_type = "WIND"
    first = legacy.keyframes.add()
    first.frame = 1
    second = legacy.keyframes.add()
    second.frame = 4
    second.wind_direction_value = (0.0, 1.0, 0.0)
    second.wind_strength_value = 2.0
    per_scale = {}
    for ws in (1.0, 0.1):
        state.world_scaling = ws
        built = params_mod._build_param_dict(bpy.context)
        sp = built["scene"]
        dyn = built.get("dyn_param", {})
        per_scale[ws] = {
            "gravity": [float(x) for x in sp["gravity"]],
            "wind": [float(x) for x in sp["wind"]],
            "ghat": float(sp["constraint-ghat"]),
            "air_density": float(sp["air-density"]),
            "dt": float(sp["dt"]),
            "anim_gravity": [float(x) for x in dyn["gravity"][-1][1]],
            "legacy_wind": [float(x) for x in dyn["wind"][-1][1]],
            "velocity": [float(x) for x in
                         group_params(built, assigned.uuid)["velocity"][assigned.uuid]],
        }
    state.world_scaling = 1.0
    clear_animation()
    state.dyn_params.clear()
    assigned.velocity_keyframes.clear()
    one, tenth = per_scale[1.0], per_scale[0.1]
    scaled = close([tenth["ghat"]], [0.1 * one["ghat"]])
    fixed = all(close(tenth[key], one[key])
                for key in ("gravity", "wind", "anim_gravity", "legacy_wind",
                            "air_density", "dt", "velocity"))
    dh.record(
        "F_world_scaling_scales_lengths",
        scaled and fixed
        and close(one["gravity"], [0.0, -9.8, 0.0])
        and close(one["wind"], [3.0, 0.0, 0.0])
        and close(one["legacy_wind"], [0.0, 0.0, -2.0]),
        {"ws_1": one, "ws_0.1": tenth},
    )

    # ---- G: a legacy entry's first keyframe is the starting frame ----------
    overlay_mod = __import__(pkg + ".ui.dynamics.overlay_geometry.colliders",
                             fromlist=["_resolve_scene_dyn_params"])
    migrate_mod = __import__(pkg + ".core.migrate_dyn_params",
                             fromlist=["convert_legacy_dyn_params"])
    utils_mod = __import__(pkg + ".core.utils", fromlist=["get_id_fcurves"])
    state.use_scene_frame_start = False
    state.frame_start = 3
    state.gravity_3d = (0.0, 0.0, -10.0)
    entry = state.dyn_params.add()
    entry.param_type = "GRAVITY"
    stored_first = entry.keyframes.add()
    stored_first.frame = 1
    later_key = entry.keyframes.add()
    later_key.frame = 6
    later_key.gravity_value = (0.0, 0.0, -2.0)
    preview_start = [float(x) for x in
                     overlay_mod._resolve_scene_dyn_params(state, 3)[0]]
    preview_mid = [float(x) for x in
                   overlay_mod._resolve_scene_dyn_params(state, 4)[0]]
    encoded_first = params_mod._build_param_dict(bpy.context)["dyn_param"]["gravity"][0]
    # A second entry whose later keyframe sits before the starting frame has
    # no faithful curve, so the conversion must leave it for the encoder.
    stale = state.dyn_params.add()
    stale.param_type = "WIND"
    stale_first = stale.keyframes.add()
    stale_first.frame = 1
    stale_later = stale.keyframes.add()
    stale_later.frame = 2
    summary = migrate_mod.convert_legacy_dyn_params(scene)
    gravity_z = next(
        (c for c in utils_mod.get_id_fcurves(scene)
         if c.data_path == gravity_path and c.array_index == 2), None)
    keys = sorted((float(p.co[0]), float(p.co[1]))
                  for p in (gravity_z.keyframe_points if gravity_z else []))
    left = [item.param_type for item in state.dyn_params]
    clear_animation()
    state.dyn_params.clear()
    state.use_scene_frame_start = True
    dh.record(
        "G_legacy_first_keyframe_is_starting_frame",
        close(preview_start, [0.0, 0.0, -10.0])
        and close(preview_mid, [0.0, 0.0, -10.0 + 8.0 / 3.0], tol=1e-5)
        and close(encoded_first[0], 0.0)
        and close(encoded_first[1], [0.0, -10.0, 0.0])
        and len(keys) == 2 and close(keys[0], [3.0, -10.0], tol=1e-5)
        and close(keys[1], [6.0, -2.0], tol=1e-5)
        and left == ["WIND"] and "left 1" in summary,
        {"preview_at_start": preview_start, "preview_at_4": preview_mid,
         "encoded_first": [float(encoded_first[0]),
                           [float(x) for x in encoded_first[1]]],
         "converted_keys": keys, "left_in_list": left, "summary": summary},
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
