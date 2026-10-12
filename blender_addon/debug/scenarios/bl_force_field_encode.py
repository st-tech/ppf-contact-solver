# File: scenarios/bl_force_field_encode.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The add-on's force-field ENCODE (issue #114): Blender field objects sent as
# FIELD OBJECTS, their settings and a track of per-frame samples in solver
# axes, as the keyword arguments of the frontend's `ForceField.add`; and the
# exact script's text.
#
# What makes the encode worth a scenario is the AXIS BOOKKEEPING, which fails
# silently: the solver's axes are Blender's (x, z, -y), and a mistake there
# ships a plausible field rotated a quarter turn. Subtest D therefore checks
# the payload against closed forms written in BLENDER's axes from the object
# itself, and checks the evaluator Visualize draws with against the same.
#
# Subtests:
#   A. no_effect_on_cloth_refused: a Charge field fails the encode, naming it
#      and saying it has no effect on cloth.
#   B. wind_needs_air_density: a Wind field in a scene with Air Density 0 fails
#      the encode, naming Air Density.
#   C. payload_records: a Turbulence and a Wind that nothing moves travel as
#      one sample each, with their settings in the frontend's names and a
#      track step of one solver frame.
#   D. axes_match_closed_forms_in_blender_axes: see above.
#   E. animated_field_is_sampled_every_frame: a Force field keyframed from
#      x = -1 to x = +1 travels as one sample per solve frame, from solver
#      x = -1 to +1.
#   F. a_discrete_change_mid_solve_is_refused: Use Max keyframed off half way
#      fails the encode naming it.
#   G. script_text_travels: the chosen Text's source is in the payload.
#   H. targets_travel_and_a_field_pushing_nothing_is_refused: a field
#      narrowed to a group carries that group's payload position; narrowed to
#      a group with no objects, the encode fails naming the field.
#   I. object_scale_is_the_noise_scale: a field scaled (2, 1, 0.5) carries
#      that noise scale and unscaled axes; a mirrored one is refused.
#   J. a_field_parented_to_a_moving_object_follows_it: a Wind with no keys of
#      its own, parented to an Empty keyframed from x = 0 to x = 4 (a walking
#      character, issue #151), travels as one sample per solve frame from
#      solver x = 0 to 4, its own offset from the parent kept.
#   K. implicit_types_travel: a Drag, a Harmonic and a Vortex with Inflow are
#      sendable, and their coefficients reach the payload in the frontend's
#      names.
#   L. an_image_texture_travels_linear: an sRGB byte image's 0.5 (stored as
#      128/255) reaches the payload through the sRGB curve Blender itself
#      reads it by (measured), a Non-Color image's unchanged, in Blender's
#      axes (`z_up`); a procedural texture and the Checker extension are
#      refused by name.
#   M. a_gas_domain_travels_per_frame: a Fluid Flow whose source is a
#      rotated, scaled Mantaflow gas domain, baked (Cache Type All) over the
#      solve frames, travels as one volume frame per solve frame, Flow as
#      `flow`, and the payload's volume read at a cell center is the domain's
#      own grid there, in m/s (2.5 cells of the longest axis per grid unit,
#      measured) and turned to world axes, in solver axes, with the cell along
#      each of the domain's axes; Visualize draws Strength times the same.
#   N. a_volume_object_travels_through_openvdb: a Fluid Flow whose source is a
#      Volume object, its file written here with Blender's openvdb in m/s
#      (Velocity Unit Second), carries the file's velocity turned and scaled
#      by the object, at voxel centers; outside the file's active voxels
#      nothing.
#   O. fluid_sources_refused_by_name: no Domain Object, a liquid domain, an
#      Adaptive Domain, a Replay cache, Noise Amount on a Fluid Flow, and a
#      file with no velocity grid are each refused naming the field.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import math
import traceback

import numpy as np
from mathutils import Vector

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["encode"])
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    enc = __import__(pkg + ".core.encoder", fromlist=["resolve_solver_fps"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0,
                                    location=(0, 0, 0))
    sheet = bpy.context.object
    sheet.name = "FFSheet"
    dh.save_blend(PROBE_DIR, "force_field_encode.blend")
    root = dh.configure_state(project_name="force_field_encode", frame_count=11)
    state = root.state
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)

    def field(kind, name, loc=(0.0, 0.0, 0.0)):
        bpy.ops.object.effector_add(type=kind, location=loc)
        o = bpy.context.object
        o.name = name
        return o

    def build():
        return params_mod._build_param_dict(bpy.context)

    def refused(fragment):
        try:
            build()
            return False, "no error"
        except ValueError as e:
            return fragment in str(e), str(e)

    def solver(v):
        return [v[0], v[2], -v[1]]

    def by_type(ffp):
        return {f["type"]: f for f in ffp.get("fields", [])}

    # ----- A --------------------------------------------------------------
    charge = field("CHARGE", "FFCharge")
    ok, msg = refused("FFCharge")
    dh.record("A_no_effect_on_cloth_refused", ok and "no effect on cloth" in msg,
              {"error": msg})
    bpy.data.objects.remove(charge)

    # ----- B --------------------------------------------------------------
    turb = field("TURBULENCE", "FFTurb", (0.1, 0.2, 0.0))
    turb.field.strength = 4.0
    turb.field.size = 0.7
    turb.field.seed = 21
    wind = field("WIND", "FFWind", (0.0, 0.0, -1.0))
    wind.rotation_euler = (0.4, 0.0, 0.3)
    wind.field.strength = 3.0
    ok, msg = refused("Air Density")
    dh.record("B_wind_needs_air_density", ok, {"error": msg})
    state.air_density = 1.0

    # ----- C --------------------------------------------------------------
    fps = float(enc.resolve_solver_fps(state))
    fields = by_type(build().get("force_field") or {})
    t, w = fields.get("turbulence", {}), fields.get("wind", {})
    ok = (set(fields) == {"turbulence", "wind"}
          and np.allclose(t.get("position"), solver((0.1, 0.2, 0.0)))
          # Blender holds the settings in single precision.
          and abs(t.get("size", 0.0) - 0.7) < 1e-6 and t.get("seed") == 21
          and t.get("strength") == 4.0
          and t.get("shape") == "point" and t.get("falloff") == "sphere"
          and w.get("shape") == "plane" and w.get("strength") == 3.0
          and abs(t.get("step", 0.0) - 1.0 / fps) < 1e-12 and t.get("start") == 0.0
          and t.get("min_distance") is None and t.get("groups") is None)
    dh.record("C_payload_records", ok, {"turbulence": {k: v for k, v in t.items()},
                                        "wind_keys": sorted(w), "fps": fps})

    # ----- D --------------------------------------------------------------
    # The Wind's local Z in Blender's axes, read off the object, and where it
    # must land: the payload's third axis column, in solver axes.
    bpy.context.view_layer.update()
    z_blender = np.array(wind.matrix_world.to_3x3().col[2])
    axes = np.array(w.get("axes", np.zeros((3, 3))))
    axes_ok = np.allclose(axes[:, 2], solver(z_blender), atol=1e-9) \
        and np.allclose(np.array(w["position"]), solver((0.0, 0.0, -1.0)))
    # And the evaluator Visualize draws with, at a Blender point: a Plane
    # Wind of strength 3 with no falloff is an air velocity of 3 along its
    # local Z, in Blender's axes, wherever the point is.
    point = np.array([[0.3, -0.2, 0.5]])
    acc, air = ff.evaluate([wind], point)
    eval_ok = np.allclose(air[0], 3.0 * z_blender, atol=1e-9) and np.allclose(acc, 0.0)
    dh.record("D_axes_match_closed_forms_in_blender_axes", axes_ok and eval_ok,
              {"payload_z": axes[:, 2].tolist(), "want_z": solver(z_blender),
               "drawn_air": air[0].tolist(), "want_air": (3.0 * z_blender).tolist()})

    # ----- E --------------------------------------------------------------
    bpy.data.objects.remove(turb)
    bpy.data.objects.remove(wind)
    state.air_density = 0.0
    push = field("FORCE", "FFPush", (-1.0, 0.0, 0.3))
    push.field.strength = 5.0
    push.keyframe_insert("location", frame=1)
    push.location = (1.0, 0.0, 0.3)
    push.keyframe_insert("location", frame=11)
    still = field("FORCE", "FFStill", (0.0, 0.0, 2.0))
    still.field.strength = 1.0
    # Keyed by the position's rank: a track is (T, 3), a single sample (3,).
    fields = {len(np.shape(f["position"])): f
              for f in (build().get("force_field") or {}).get("fields", [])}
    moving = fields.get(2, {})
    track = np.array(moving.get("position", np.zeros((0, 3))))
    ok = (len(track) == 11 and np.allclose(track[0], solver((-1.0, 0.0, 0.3)))
          and np.allclose(track[-1], solver((1.0, 0.0, 0.3)))
          and np.shape(fields.get(1, {}).get("position")) == (3,)
          and len(moving.get("strength", [])) == 11)
    dh.record("E_animated_field_is_sampled_every_frame", ok,
              {"samples": len(track), "first": track[0].tolist() if len(track) else None,
               "last": track[-1].tolist() if len(track) else None,
               "still_position": fields.get(1, {}).get("position")})
    bpy.data.objects.remove(still)

    # ----- F --------------------------------------------------------------
    push.field.use_max_distance = True
    push.field.distance_max = 3.0
    push.keyframe_insert("field.use_max_distance", frame=1)
    push.field.use_max_distance = False
    push.keyframe_insert("field.use_max_distance", frame=6)
    ok, msg = refused("use_max")
    dh.record("F_a_discrete_change_mid_solve_is_refused", ok and "FFPush" in msg,
              {"error": msg})
    bpy.data.objects.remove(push)

    # ----- G --------------------------------------------------------------
    text = bpy.data.texts.new("ff_script.py")
    text.from_string("def eval(x, y, z, t):\n    return (0.0, 0.0, 1.0)\n")
    state.force_field_script = text
    ffp = build().get("force_field") or {}
    (sc,) = ffp.get("scripts") or [{}]
    dh.record("G_script_text_travels",
              sc.get("source") == text.as_string() and sc.get("name") == "ff_script.py"
              and sc.get("groups") is None,
              {"script": sc})

    # ----- H --------------------------------------------------------------
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=0.5,
                                    location=(3.0, 0.0, 0.0))
    far = bpy.context.object
    far.name = "FFFar"
    far_group = dh.api.solver.create_group("Far", "SHELL")
    far_group.add(far.name)
    near = field("FORCE", "FFNear", (3.0, 0.0, 0.3))
    near.field.strength = 2.0
    dh.api.solver.set_force_field_targets("FFNear", [far_group])
    (f,) = (build().get("force_field") or {}).get("fields", [{}])
    pushed = [o.name for o in ff.pushed_objects(bpy.context.scene, state, near)]
    travels = f.get("groups") is not None and len(f["groups"]) == 1 and pushed == ["FFFar"]
    empty_group = dh.api.solver.create_group("Empty", "SHELL")
    dh.api.solver.set_force_field_targets("FFNear", [empty_group])
    ok, msg = refused("pushes no object")
    dh.record("H_targets_travel_and_a_field_pushing_nothing_is_refused",
              travels and ok and "FFNear" in msg,
              {"groups": f.get("groups"), "pushed": pushed, "error": msg})
    bpy.data.objects.remove(near)

    # ----- I --------------------------------------------------------------
    scaled = field("FORCE", "FFScaled", (0.0, 0.0, 1.0))
    scaled.rotation_euler = (0.0, 0.3, 0.0)
    scaled.scale = (2.0, 1.0, 0.5)
    scaled.field.noise = 0.5
    bpy.context.view_layer.update()
    (f,) = (build().get("force_field") or {}).get("fields", [{}])
    axes = np.array(f.get("axes", np.zeros((3, 3))))
    scale_ok = (np.allclose(f.get("noise_scale"), [2.0, 1.0, 0.5])
                and np.allclose(axes.T @ axes, np.eye(3), atol=1e-9)
                and f.get("noise") == 0.5)
    scaled.scale = (1.0, -1.0, 1.0)
    bpy.context.view_layer.update()
    ok, msg = refused("mirrored")
    dh.record("I_object_scale_is_the_noise_scale", scale_ok and ok,
              {"noise_scale": f.get("noise_scale"), "error": msg})
    bpy.data.objects.remove(scaled)

    # ----- J --------------------------------------------------------------
    state.air_density = 1.0
    walker = bpy.data.objects.new("FFWalker", None)
    bpy.context.scene.collection.objects.link(walker)
    walker.location = (0.0, 0.0, 0.0)
    walker.keyframe_insert("location", frame=1)
    walker.location = (4.0, 0.0, 0.0)
    walker.keyframe_insert("location", frame=11)
    bpy.context.scene.frame_set(1)
    carried = field("WIND", "FFCarried", (0.0, 0.5, 1.0))
    carried.parent = walker
    carried.field.strength = 2.0
    bpy.context.view_layer.update()
    (f,) = (build().get("force_field") or {}).get("fields", [{}])
    track = np.array(f.get("position", np.zeros((0, 3))))
    want_first = solver((0.0, 0.5, 1.0))
    want_last = solver((4.0, 0.5, 1.0))
    ok = (carried.animation_data is None and len(track) == 11
          and np.allclose(track[0], want_first, atol=1e-6)
          and np.allclose(track[-1], want_last, atol=1e-6))
    record_j = (ok, track)
    dh.record("J_a_field_parented_to_a_moving_object_follows_it", ok,
              {"samples": len(track),
               "first": track[0].tolist() if len(track) else None,
               "last": track[-1].tolist() if len(track) else None,
               "want_first": want_first, "want_last": want_last})

    bpy.data.objects.remove(carried)

    # ----- K --------------------------------------------------------------
    drag = field("DRAG", "FFDrag", (0.0, 0.0, 0.5))
    drag.field.linear_drag = 0.25
    drag.field.quadratic_drag = 0.75
    spring = field("HARMONIC", "FFSpring", (0.0, 0.0, 0.5))
    spring.field.rest_length = 0.3
    spring.field.harmonic_damping = 0.4
    swirl = field("VORTEX", "FFSwirl", (0.0, 0.0, -0.5))
    swirl.field.inflow = 0.5
    refusals = {o.name: ff.refusal(o) for o in (drag, spring, swirl)}
    fields = by_type(build().get("force_field") or {})
    d, h, v = fields.get("drag", {}), fields.get("harmonic", {}), fields.get("vortex", {})
    ok = (all(why is None for why in refusals.values())
          and abs(d.get("linear_drag", 0) - 0.25) < 1e-6
          and abs(d.get("quadratic_drag", 0) - 0.75) < 1e-6
          and abs(h.get("rest_length", 0) - 0.3) < 1e-6
          and abs(h.get("harmonic_damping", 0) - 0.4) < 1e-6
          and abs(v.get("inflow", 0) - 0.5) < 1e-6)
    dh.record("K_implicit_types_travel", ok,
              {"refusals": refusals, "drag": {k: d.get(k) for k in ("linear_drag", "quadratic_drag")},
               "harmonic": {k: h.get(k) for k in ("rest_length", "harmonic_damping")},
               "vortex_inflow": v.get("inflow")})

    for o in (drag, spring, swirl):
        bpy.data.objects.remove(o)

    # ----- L --------------------------------------------------------------
    def image_texture(name, space):
        img = bpy.data.images.new(name, 2, 2)
        # Colorspace FIRST: changing it regenerates a generated image, blank.
        img.colorspace_settings.name = space
        img.pixels = [0.5, 0.5, 0.5, 1.0] * 4
        tex = bpy.data.textures.new(name, "IMAGE")
        tex.image = img
        tex.extension = "EXTEND"
        return tex

    textured = field("TEXTURE", "FFTex", (0.0, 0.0, 0.5))
    payload_values = {}
    z_up = []
    for space in ("sRGB", "Non-Color"):
        textured.field.texture = image_texture("ff_" + space, space)
        (f,) = (build().get("force_field") or {}).get("fields", [{}])
        # The image travels as a list of byte chunks.
        img = np.frombuffer(b"".join(f.get("image", [])), dtype=np.float32)
        payload_values[space] = float(img[0]) if len(img) else None
        z_up.append(f.get("z_up"))
    textured.field.texture = bpy.data.textures.new("ff_clouds", "CLOUDS")
    ok_proc, msg_proc = refused("not supported yet")
    tex = image_texture("ff_checker", "sRGB")
    tex.extension = "CHECKER"
    textured.field.texture = tex
    ok_check, msg_check = refused("Checker")
    # A byte image stores 0.5 as 128 / 255; the sRGB curve maps that.
    stored = 128.0 / 255.0
    srgb_half = ((stored + 0.055) / 1.055) ** 2.4
    ok = (payload_values.get("sRGB") is not None
          and abs(payload_values["sRGB"] - srgb_half) < 1e-5
          and abs((payload_values.get("Non-Color") or 0.0) - stored) < 1e-6
          and z_up == [True, True] and ok_proc and ok_check)
    dh.record("L_an_image_texture_travels_linear", ok,
              {"payload": payload_values, "want_srgb": srgb_half, "z_up": z_up,
               "procedural": msg_proc, "checker": msg_check})

    # ----- M --------------------------------------------------------------
    import os

    for o in [o for o in bpy.data.objects if o.field is not None and o.field.type != "NONE"]:
        bpy.data.objects.remove(o)
    fe = __import__(pkg + ".core.field_eval", fromlist=["fluid_at"])
    scene = bpy.context.scene

    def payload_volume(v):
        # Each frame's table and bricks travel as a list of byte chunks.
        return {"resolution": tuple(v["resolution"]), "cell": np.asarray(v["cell"]),
                "corner": np.asarray(v["corner"]), "axes": np.asarray(v["axes"]),
                "start": v["start"], "step": v["step"],
                "tables": [np.frombuffer(b"".join(t), dtype="<u4") for t in v["tables"]],
                "bricks": [np.frombuffer(b"".join(b), dtype="<f4").reshape(-1, 512, 4)
                           for b in v["bricks"]]}

    def bake(dom):
        # A gas domain is read only once baked over the solve frames.
        bake_fluid_domain(scene, dom, result.setdefault("bake_notes", []))

    def gas_domain(name, location, scale, angle):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
        dom = bpy.context.object
        dom.name = name
        dom.scale = (scale, scale, scale)
        dom.rotation_euler = (0.0, 0.0, angle)
        mod = dom.modifiers.new("fluid", "FLUID")
        mod.fluid_type = "DOMAIN"
        ds = mod.domain_settings
        ds.domain_type = "GAS"
        ds.resolution_max = 16
        ds.alpha = ds.beta = 0.0
        ds.gravity = (0.0, 0.0, 0.0)
        ds.use_adaptive_timesteps = False
        ds.cache_type = "ALL"
        ds.cache_directory = os.path.join(PROBE_DIR, "ff_fluid_cache_" + name)
        start = enc.resolve_start_frame(state)
        ds.cache_frame_start, ds.cache_frame_end = start, start + state.frame_count - 1
        bpy.context.view_layer.update()
        inflow_at = dom.matrix_world @ Vector((-0.6, 0.0, 0.0))
        bpy.ops.mesh.primitive_cube_add(size=0.3 * scale, location=inflow_at)
        src = bpy.context.object
        src.name = name + "Inflow"
        fm = src.modifiers.new("fluid", "FLUID")
        fm.fluid_type = "FLOW"
        fl = fm.flow_settings
        fl.flow_type = "SMOKE"
        fl.flow_behavior = "INFLOW"
        fl.use_initial_velocity = True
        fl.velocity_coord = (1.5, 0.0, 0.0)
        bake(dom)
        return dom, src

    dom, inflow = gas_domain("FFDomain", (0.3, -0.2, 0.5), 0.75, 0.5)
    # The operator names it FLUID; the field it makes is type FLUID_FLOW.
    flow_field = field("FLUID", "FFFluid", (0.0, 0.0, 2.0))
    flow_field.field.source_object = dom
    flow_field.field.strength = 2.0
    flow_field.field.flow = 1.0
    first = scene.frame_current
    f = by_type(build().get("force_field") or {}).get("fluid", {})
    vol = f.get("fluid", {})
    details = {"keys": sorted(f), "frames": len(vol.get("tables", [])),
               "flow": f.get("flow"), "use_density": f.get("use_density")}
    ok = (len(vol.get("tables", [])) == 11 and f.get("flow") == 1.0
          and f.get("use_density") is False and f.get("strength") == 2.0
          and abs(vol.get("step", 0.0) - 1.0 / fps) < 1e-12)
    if ok:
        # The domain's own grid at the last solve frame, at its densest cell.
        scene.frame_set(first + 10)
        dg = bpy.context.evaluated_depsgraph_get()
        ds = dom.evaluated_get(dg).modifiers["fluid"].domain_settings
        n = tuple(ds.domain_resolution)
        grid = np.array(ds.velocity_grid, dtype=np.float64).reshape(n[2], n[1], n[0], 3)
        dens = np.array(ds.density_grid, dtype=np.float64).reshape(n[2], n[1], n[0])
        kz, jy, ix = np.unravel_index(int(np.argmax(dens)), dens.shape)
        cell = 1.5 / max(n)
        local = Vector([-1.0 + (i + 0.5) * 2.0 / max(n) for i in (ix, jy, kz)])
        world = dom.matrix_world @ local
        rot = np.array(dom.matrix_world.to_3x3().normalized(), dtype=np.float64)
        # Per solver second: the solver plays a Blender frame in 1 / fps
        # seconds, and this scene's own rate differs (fluid_rate).
        rate = fps / (scene.render.fps / scene.render.fps_base)
        details["rate"] = rate
        want = rot @ (grid[kz, jy, ix] * 2.5 * cell) * rate
        got = fe.fluid_at(payload_volume(vol), [solver(world)], 10.0 / fps)[0][0]
        # And Visualize, which draws at the current frame through the same
        # reference, at rest: Strength 2 times the fluid, in Blender's axes.
        drawn = ff.evaluate([flow_field], [world])[0][0]
        scene.frame_set(first)
        details["visualize_error"] = float(np.abs(drawn - 2.0 * want).max())
        err = float(np.abs(got - solver(want)).max())
        # The cell along each of the domain's axes; the cube's are equal.
        cells = np.asarray(vol["cell"], dtype=np.float64)[-1]
        details.update({"cell": [int(ix), int(jy), int(kz)], "want": solver(want),
                        "got": got.tolist(), "error": err,
                        "cell_size": cells.tolist()})
        ok = (err < 2e-3 * float(np.abs(want).max()) and float(np.abs(want).max()) > 0.1
              and details["visualize_error"] < 4e-3 * float(np.abs(want).max())
              and cells.shape == (3,) and np.allclose(cells, cell, rtol=0.0, atol=1e-9))
    dh.record("M_a_gas_domain_travels_per_frame", ok, details)

    # ----- N --------------------------------------------------------------
    import openvdb

    vdb_path = os.path.join(PROBE_DIR, "ff_velocity.vdb")
    g = openvdb.Vec3SGrid()
    g.name = "vel"
    g.transform = openvdb.createLinearTransform(voxelSize=0.2)
    acc = g.getAccessor()
    voxels = {(1, 2, 3): (1.0, 2.0, 3.0), (2, 2, 3): (4.0, 0.0, 0.0), (1, 3, 4): (0.0, 0.0, 5.0)}
    for ijk, value in voxels.items():
        acc.setValueOn(ijk, value)
    openvdb.write(vdb_path, grids=[g])
    volume_data = bpy.data.volumes.new("FFVelocity")
    volume_data.filepath = vdb_path
    # The file holds m/s. Set the unit after the load, which names the grids.
    volume_data.grids.load()
    volume_data.velocity_unit = "SECOND"
    volume_obj = bpy.data.objects.new("FFVelocity", volume_data)
    scene.collection.objects.link(volume_obj)
    volume_obj.location = (1.0, 0.0, 0.0)
    volume_obj.rotation_euler = (0.0, 0.0, math.pi / 2)
    volume_obj.scale = (1.5, 1.5, 1.5)
    bpy.context.view_layer.update()
    flow_field.field.source_object = volume_obj
    flow_field.field.flow = 0.0
    flow_field.field.strength = 1.0
    f = by_type(build().get("force_field") or {}).get("fluid", {})
    vol = f.get("fluid", {})
    m = volume_obj.matrix_world
    checks = []
    for ijk, value in voxels.items():
        world = m @ Vector([0.2 * c for c in ijk])
        want = np.array(m.to_3x3() @ Vector(value)) * fps / (scene.render.fps / scene.render.fps_base)
        got = fe.fluid_at(payload_volume(vol), [solver(world)], 0.0)[0][0]
        checks.append(float(np.abs(got - solver(want)).max()))
    outside = fe.fluid_at(payload_volume(vol), [solver(m @ Vector((2.0, 2.0, 2.0)))], 0.0)
    cells = np.asarray(vol.get("cell", np.zeros((1, 3))), dtype=np.float64)[0]
    ok = (len(vol.get("tables", [])) == 11 and max(checks) < 1e-5
          and cells.shape == (3,) and np.allclose(cells, 0.3, rtol=0.0, atol=1e-9)
          and float(outside[2][0]) == 0.0)
    dh.record("N_a_volume_object_travels_through_openvdb", ok,
              {"errors": checks, "cell": cells.tolist(),
               "outside_presence": float(outside[2][0])})

    # ----- O --------------------------------------------------------------
    reasons = {}
    flow_field.field.source_object = None
    reasons["no_source"] = refused("no Domain Object")
    flow_field.field.source_object = dom
    flow_field.field.noise = 0.5
    reasons["noise"] = refused("Noise Amount")
    flow_field.field.noise = 0.0
    # Changing a domain setting frees its bake, so this goes after every
    # refusal that needs the baked domain.
    dom.modifiers["fluid"].domain_settings.cache_type = "REPLAY"
    reasons["replay"] = refused("Replay cache")
    dom.modifiers["fluid"].domain_settings.cache_type = "ALL"
    dom.modifiers["fluid"].domain_settings.use_adaptive_domain = True
    reasons["adaptive"] = refused("Adaptive Domain")
    dom.modifiers["fluid"].domain_settings.use_adaptive_domain = False
    dom.modifiers["fluid"].domain_settings.domain_type = "LIQUID"
    reasons["liquid"] = refused("liquid")
    dom.modifiers["fluid"].domain_settings.domain_type = "GAS"
    density_only = os.path.join(PROBE_DIR, "ff_density.vdb")
    d = openvdb.FloatGrid()
    d.name = "density"
    d.getAccessor().setValueOn((0, 0, 0), 1.0)
    openvdb.write(density_only, grids=[d])
    volume_data.filepath = density_only
    flow_field.field.source_object = volume_obj
    reasons["no_velocity_grid"] = refused("no velocity grid")
    ok = all(r[0] and "FFFluid" in r[1] for r in reasons.values())
    dh.record("O_fluid_sources_refused_by_name", ok,
              {k: v[1] for k, v in reasons.items()})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
