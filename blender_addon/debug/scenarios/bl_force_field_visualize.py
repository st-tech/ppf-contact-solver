# File: scenarios/bl_force_field_visualize.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force field's Visualize overlay (owner's request): arrows at a sparse
# PREVIEW grid, for drawing only (the solver evaluates a field at every vertex
# exactly), for the field at the timeline's current frame.
#
# The checks are on `preview_fields`, WHAT is drawn, and on the arrow geometry,
# neither of which needs a GPU, so every platform runs them. Building the GPU
# batches does, and Windows runs the rig headless, so that check is
# `bl_force_field_visualize_draws`, a scenario of its own that the Windows
# leg does not select.
#
# Subtests:
#   A. off_draws_nothing: with Visualize off there is nothing to draw.
#   B. preview_grid_is_its_own: Preview Resolution 4 x 3 x 2 gives 24 arrow
#      sites.
#   C. arrows_are_the_evaluator: the acceleration arrows equal the solver's
#      reference laws at the same points, and the script's arrows equal the
#      script.
#   D. arrows_follow_the_timeline: a Force whose strength is keyed 0 -> 5 has
#      no arrows at the timeline's first frame and arrows at its last, and the
#      overlay's cache key changes with the frame, so the drawing rebuilds.
#   E. key_follows_a_moved_field: moving the field object changes the overlay's
#      cache key, so the drawn arrows rebuild.
#   F. arrow_heads_are_sized_by_their_arrow: a short arrow's head is a quarter
#      of its length, and a long one's is capped by the shaft thickness, so a
#      weak spot draws a small arrow rather than a full head on a stub.
#
# The overlay keeps each field's drawn record under `core.force_field.
# field_key` and rebuilds it only when the key changes, so a key that misses
# something the record is read from draws a stale field. G to K make each such
# input change, alone, and require the field's key AND the overlay's key to
# change and the record to be rebuilt with different content:
#   G. a_fluid_follows_apply_density: Apply Density on a Fluid Flow whose
#      Volume holds a density grid.
#   H. a_fluid_follows_its_domain_object: the Domain Object switched to a
#      second Volume holding a different velocity.
#   I. a_fluid_follows_a_moved_source: the Volume object moved 0.5 along x;
#      the record's corner moves that much along solver x and nowhere else,
#      to 1e-9 (measured: exactly, an error of 0.0; a record left stale
#      reads 0.5 off).
#   J. a_fluid_follows_a_rebake: a gas domain's cache files (Cache Type All)
#      replaced by a bake with a different inflow, so that nothing about the
#      domain object changes but its cache files.
#   K. a_texture_follows_its_settings_and_image: a Texture field's mode,
#      the texture's extension, its interpolation and its image, each in turn.
#   M. a_texture_follows_its_image_content: an image's CONTENT, which the
#      image's flags do not show: a generated image's Color and then its
#      Type (it stays clean), a first, second and third pixel edit (it stays
#      dirty from the first), and its alpha: half transparent under Use
#      Alpha (read straight, so the drawn colors stay as they were while the
#      keys change), the Alpha mode None and back, and Use Alpha off (each a
#      premultiplied or straight reading, so the colors change). Each edit
#      rebuilds the record, and the drawn image equals what Transfer sends
#      (`texture_image`) exactly; an unchanged image reuses its record.
#   L. a_fluid_is_read_once_at_a_frame: the reverse of G to M. A Volume whose
#      grids were never loaded has its Velocity Grid renamed by the first
#      read ("velocity" to the file's "vel"), and a second draw at the same
#      frame must reuse the record that read built.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["evaluate"])
    viz = __import__(pkg + ".ui.dynamics.overlay_geometry.force_field",
                     fromlist=["preview_fields"])
    scene = bpy.context.scene

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0)
    sheet = bpy.context.object
    dh.save_blend(PROBE_DIR, "force_field_visualize.blend")
    root = dh.configure_state(project_name="force_field_visualize", frame_count=11)
    state = root.state
    dh.api.solver.create_group("Cloth", "SHELL").add(sheet.name)
    bpy.ops.object.effector_add(type="FORCE", location=(0.0, 0.0, -0.4))
    push = bpy.context.object
    push.field.strength = 0.0
    push.field.keyframe_insert("strength", frame=1)
    push.field.strength = 5.0
    push.field.keyframe_insert("strength", frame=11)
    scene.frame_set(11)

    dh.record("A_off_draws_nothing",
              viz.preview_fields(scene, state) is None
              and viz.build_force_field_batches(scene, state, 10.0) == [], {})

    state.force_field_visualize = True
    state.force_field_preview_resolution = (4, 3, 2)
    pts, cell, fields = viz.preview_fields(scene, state)
    dh.record("B_preview_grid_is_its_own",
              pts.shape == (24, 3) and len(fields) == 2,
              {"points": list(pts.shape), "fields": len(fields)})

    text = bpy.data.texts.new("ff_viz.py")
    # `import math` and `curl_noise` on purpose: the template uses both, and
    # the drawing path once ran scripts with no import machinery at all. The
    # noise is multiplied by zero so the expected arrows stay a closed form.
    text.from_string("import math\n\ndef eval(x, y, z, t):\n"
                     "    cx, cy, cz = curl_noise(x, y, z, octaves=2, seed=5)\n"
                     "    return (y + 0.0 * cx, -x, 0.5 * math.cos(0.0) + 0.0 * cz)\n")
    state.force_field_script = text
    pts, cell, fields = viz.preview_fields(scene, state)
    acc, _air = ff.evaluate(ff.field_objects(scene, state), pts)
    script = fields[2][0] if len(fields) == 3 else None
    want = np.stack([pts[:, 1], -pts[:, 0], np.full(len(pts), 0.5)], axis=1)
    dh.record("C_arrows_are_the_evaluator",
              len(fields) == 3 and np.allclose(fields[0][0], acc)
              and script is not None and np.allclose(script, want),
              {"fields": len(fields)})
    state.force_field_script = None

    scene.frame_set(1)
    first = viz.preview_fields(scene, state)[2][0][0]
    key_first = viz.force_field_key(scene, state)
    scene.frame_set(11)
    last = viz.preview_fields(scene, state)[2][0][0]
    key_last = viz.force_field_key(scene, state)
    dh.record("D_arrows_follow_the_timeline",
              float(np.abs(first).max()) < 1e-9
              and float(np.abs(last).max()) > 1.0
              and key_first != key_last,
              {"max_first": float(np.abs(first).max()),
               "max_last": float(np.abs(last).max())})

    key0 = viz.force_field_key(scene, state)
    push.location = (0.3, 0.0, -0.4)
    bpy.context.view_layer.update()
    key1 = viz.force_field_key(scene, state)
    dh.record("E_key_follows_a_moved_field", key0 != key1, {})

    from mathutils import Vector

    def head_length(length, thickness):
        tris = viz._field_arrow(Vector((0.0, 0.0, 0.0)), Vector((length, 0.0, 0.0)),
                                thickness)
        # The head's ring sits where the shaft ends: the smallest x among the
        # cone's vertices, which are the last 18 (6 triangles of 3).
        return length - min(v.x for v in tris[-18:])

    short = head_length(0.02, 0.002)
    long_ = head_length(1.0, 0.002)
    dh.record("F_arrow_heads_are_sized_by_their_arrow",
              abs(short - 0.25 * 0.02) < 1e-7 and abs(long_ - 12.0 * 0.002) < 1e-6,
              {"short_head": short, "long_head": long_})

    # ----- G .. K: a field's drawn record follows what it is read from -----
    import os

    import openvdb

    def snapshot(obj):
        # The field's key, its drawn record and the overlay's key, as the
        # overlay sees them at the current frame.
        return ff.field_key(obj), ff.record_at_frame(obj), viz.force_field_key(scene, state)

    def follows(before, after, differs):
        # The edit changes both keys, and the record is rebuilt with
        # different content rather than reused.
        (k0, r0, v0), (k1, r1, v1) = before, after
        return {"field_key": k0 != k1, "overlay_key": v0 != v1,
                "record_rebuilt": r1 is not r0, "content": bool(differs(r0, r1))}

    def bricks(rec):
        return np.asarray(rec["fluid"]["bricks"][0], dtype=np.float64)

    def write_volume(name, velocity, density, load=True):
        path = os.path.join(PROBE_DIR, name + ".vdb")
        g = openvdb.Vec3SGrid()
        g.name = "vel"
        g.transform = openvdb.createLinearTransform(voxelSize=0.1)
        d = openvdb.FloatGrid()
        d.name = "density"
        d.transform = openvdb.createLinearTransform(voxelSize=0.1)
        for ijk in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
            g.getAccessor().setValueOn(ijk, velocity)
            d.getAccessor().setValueOn(ijk, density)
        openvdb.write(path, grids=[g, d])
        data = bpy.data.volumes.new(name)
        data.filepath = path
        if load:
            data.grids.load()
        data.velocity_unit = "SECOND"
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        return obj

    vol_a = write_volume("ff_viz_vol_a", (1.0, 0.0, 0.0), 0.5)
    vol_b = write_volume("ff_viz_vol_b", (0.0, 2.0, 0.0), 0.25)
    bpy.ops.object.effector_add(type="FLUID", location=(0.0, 0.0, 0.0))
    fluid = bpy.context.object
    fluid.name = "FFVizFluid"
    fluid.field.source_object = vol_a
    bpy.context.view_layer.update()

    s0 = snapshot(fluid)
    fluid.field.use_smoke_density = True
    s1 = snapshot(fluid)
    got = follows(s0, s1, lambda a, b: a["use_density"] != b["use_density"]
                  and not np.array_equal(bricks(a)[..., 3], bricks(b)[..., 3]))
    dh.record("G_a_fluid_follows_apply_density", all(got.values()), got)

    s2 = snapshot(fluid)
    fluid.field.source_object = vol_b
    bpy.context.view_layer.update()
    s3 = snapshot(fluid)
    got = follows(s2, s3, lambda a, b: not np.array_equal(bricks(a)[..., :3],
                                                          bricks(b)[..., :3]))
    dh.record("H_a_fluid_follows_its_domain_object", all(got.values()), got)

    s4 = snapshot(fluid)
    vol_b.location = (vol_b.location[0] + 0.5, vol_b.location[1], vol_b.location[2])
    bpy.context.view_layer.update()
    s5 = snapshot(fluid)
    # Blender +x is solver +x: the corner moves 0.5 along it and nowhere else.
    shift = (np.asarray(s5[1]["fluid"]["corner"][0], dtype=np.float64)
             - np.asarray(s4[1]["fluid"]["corner"][0], dtype=np.float64))
    corner_error = float(np.abs(shift - [0.5, 0.0, 0.0]).max())
    result.setdefault("measured", {})["I_corner_error"] = corner_error
    got = follows(s4, s5, lambda a, b: corner_error <= 1e-9)
    dh.record("I_a_fluid_follows_a_moved_source", all(got.values()), got)

    # A Volume whose grids were never loaded names the default Velocity Grid
    # "velocity", which Blender renames to the file's "vel" at the first
    # read (measured); the record built by that read must still be found at
    # the same frame rather than read again.
    vol_c = write_volume("ff_viz_vol_c", (0.0, 0.0, 3.0), 0.5, load=False)
    fluid.field.source_object = vol_c
    bpy.context.view_layer.update()
    named_before = vol_c.data.velocity_grid
    first = ff.record_at_frame(fluid)
    named_after = vol_c.data.velocity_grid
    again = ff.record_at_frame(fluid)
    dh.record("L_a_fluid_is_read_once_at_a_frame",
              named_before == "velocity" and named_after == "vel" and again is first,
              {"named_before": named_before, "named_after": named_after,
               "same_record": again is first})
    bpy.data.objects.remove(fluid)

    # A gas domain baked over the drawn frame, then its cache files replaced
    # by a bake with a different inflow: nothing about the domain object
    # changes, only its files. The new files are a second domain's first bake,
    # copied over the first's, because a bake that follows a free crashed
    # Blender 5.1.2 on Windows (measured: in this scenario under the rig in
    # every run, and in a fresh Blender with no add-on loaded in 21 of 24
    # runs; Mantaflow's initialization reads a null pointer,
    # `MANTA::getRealValue`).
    import shutil

    start = scene.frame_current

    def gas_domain(name, location, velocity):
        # A size-2 gas domain baked over 1 .. start, its inflow jet inside it
        # pushing at `velocity`.
        x, y, z = location
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
        dom = bpy.context.object
        dom.name = name
        mod = dom.modifiers.new("fluid", "FLUID")
        mod.fluid_type = "DOMAIN"
        ds = mod.domain_settings
        ds.domain_type = "GAS"
        ds.resolution_max = 16
        ds.alpha = ds.beta = 0.0
        ds.gravity = (0.0, 0.0, 0.0)
        ds.use_adaptive_timesteps = False
        ds.cache_type = "ALL"
        ds.cache_directory = os.path.join(PROBE_DIR, name.lower() + "_cache")
        ds.cache_frame_start, ds.cache_frame_end = 1, start
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=(x - 0.6, y, z))
        jet = bpy.context.object
        jet.name = name + "Jet"
        fm = jet.modifiers.new("fluid", "FLUID")
        fm.fluid_type = "FLOW"
        fm.flow_settings.flow_type = "SMOKE"
        fm.flow_settings.flow_behavior = "INFLOW"
        fm.flow_settings.use_initial_velocity = True
        fm.flow_settings.velocity_coord = velocity
        bake_fluid_domain(scene, dom, result.setdefault("bake_notes", []))
        return dom, jet, bpy.path.abspath(ds.cache_directory)

    # Far enough apart that neither jet lies in the other's domain.
    dom, jet, cache = gas_domain("FFVizDomain", (3.0, 0.0, 0.0), (1.5, 0.0, 0.0))
    other, other_jet, other_cache = gas_domain("FFVizOther", (3.0, 6.0, 0.0), (0.0, 1.5, 0.0))
    scene.frame_set(start)
    bpy.ops.object.effector_add(type="FLUID", location=(3.0, 0.0, 0.0))
    on_domain = bpy.context.object
    on_domain.name = "FFVizDomainFluid"
    on_domain.field.source_object = dom
    s6 = snapshot(on_domain)
    copied = []
    for sub in sorted(os.listdir(other_cache)):
        for nm in sorted(os.listdir(os.path.join(other_cache, sub))):
            os.makedirs(os.path.join(cache, sub), exist_ok=True)
            shutil.copyfile(os.path.join(other_cache, sub, nm), os.path.join(cache, sub, nm))
            copied.append(f"{sub}/{nm}")
    result.setdefault("measured", {})["J_files_replaced"] = len(copied)
    # Away and back, so the domain reads the drawn frame from the new files.
    scene.frame_set(start - 1)
    scene.frame_set(start)
    s7 = snapshot(on_domain)
    got = follows(s6, s7, lambda a, b: not np.array_equal(bricks(a)[..., :3],
                                                          bricks(b)[..., :3]))
    dh.record("J_a_fluid_follows_a_rebake", all(got.values()) and len(copied) > 0, got)
    for o in (on_domain, other, other_jet):
        bpy.data.objects.remove(o)

    # A Texture field: the texture's mode, extension, interpolation and image
    # each reach the drawn record.
    def image(name, value):
        img = bpy.data.images.new(name, 2, 2)
        img.colorspace_settings.name = "Non-Color"
        img.pixels = [value, 0.2, 0.9, 1.0, 0.1, value, 0.4, 1.0,
                      0.3, 0.6, value, 1.0, 0.8, 0.0, 0.5, 1.0]
        return img

    tex = bpy.data.textures.new("ff_viz_tex", "IMAGE")
    tex.image = image("ff_viz_img_a", 0.7)
    tex.extension = "EXTEND"
    bpy.ops.object.effector_add(type="TEXTURE", location=(0.0, 0.0, 0.0))
    textured = bpy.context.object
    textured.name = "FFVizTex"
    textured.field.texture = tex
    edits = {
        "mode": (lambda: setattr(textured.field, "texture_mode", "GRADIENT"),
                 lambda a, b: a["texture_mode"] == "rgb" and b["texture_mode"] == "gradient"),
        "extension": (lambda: setattr(tex, "extension", "CLIP"),
                      lambda a, b: a["extension"] == "extend" and b["extension"] == "clip"),
        "interpolation": (lambda: setattr(tex, "use_interpolation",
                                          not tex.use_interpolation),
                          lambda a, b: a["interpolation"] != b["interpolation"]),
        "image": (lambda: setattr(tex, "image", image("ff_viz_img_b", 0.05)),
                  lambda a, b: not np.array_equal(a["image"], b["image"])),
    }
    texture_got = {}
    for what, (edit, differs) in edits.items():
        before = snapshot(textured)
        edit()
        texture_got[what] = follows(before, snapshot(textured), differs)
    dh.record("K_a_texture_follows_its_settings_and_image",
              all(all(v.values()) for v in texture_got.values()), texture_got)

    # An image's content: generated settings, repeated edits, and its alpha.
    gen = bpy.data.images.new("ff_viz_img_gen", 8, 8, alpha=True)
    gen.generated_color = (0.2, 0.2, 0.2, 1.0)
    tex.image = gen
    tex.use_alpha = True
    pixels = np.empty(8 * 8 * 4, dtype=np.float32)

    def paint(channel, value):
        gen.pixels.foreach_get(pixels)
        pixels[channel::4] = value
        gen.pixels.foreach_set(pixels)
        gen.update()

    # (what, edit, whether the colors Blender reads change)
    content = [
        ("generated_color", lambda: setattr(gen, "generated_color", (0.8, 0.6, 0.4, 1.0)), True),
        ("generated_type", lambda: setattr(gen, "generated_type", "COLOR_GRID"), True),
        ("first_edit", lambda: paint(0, 0.9), True),
        ("second_edit", lambda: paint(0, 0.1), True),
        ("third_edit", lambda: paint(1, 0.5), True),
        ("half_transparent_under_use_alpha", lambda: paint(3, 0.5), False),
        ("alpha_mode_none", lambda: setattr(gen, "alpha_mode", "NONE"), True),
        ("alpha_mode_straight", lambda: setattr(gen, "alpha_mode", "STRAIGHT"), True),
        ("use_alpha_off", lambda: setattr(tex, "use_alpha", False), True),
    ]
    content_got = {}
    for what, edit, changes in content:
        before = snapshot(textured)
        edit()
        after = snapshot(textured)
        row = follows(before, after, lambda a, b: changes != np.array_equal(a["image"],
                                                                           b["image"]))
        row["drawn_is_sent"] = bool(np.array_equal(
            after[1]["image"], ff.texture_image(textured.field, textured.name)))
        row["dirty"] = bool(gen.is_dirty)
        content_got[what] = row
    again = ff.record_at_frame(textured)
    content_got["unchanged_reuses_its_record"] = again is ff.record_at_frame(textured)
    dh.record("M_a_texture_follows_its_image_content",
              all(all(v for k, v in row.items() if k != "dirty") for row in content_got.values()
                  if isinstance(row, dict))
              and content_got["unchanged_reuses_its_record"], content_got)

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
    verdict = r.report_named_checks(result.get("checks", {}))
    verdict.setdefault("notes", []).append(f"measured: {result.get('measured')}")
    return verdict
