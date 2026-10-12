# File: scenarios/bl_force_field_blender_parity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# BLENDER'S OWN FIELDS AGAINST THIS PROJECT'S, for every emulated non-fluid
# type, and the solver in the same chain.
#
# The oracle is Blender itself, black box: for each case a field object,
# rotated, offset and scaled (1.3, 0.8, 1.1), in a collection of its own, and
# two particle systems (mass 1, gravity 0, Euler, timestep 0.01) emitted at
# frame 1 from non-uniformly placed points, one at rest and one at V_REF,
# each limited to that collection by `effector_weights.collection`, so
# every case is measured in one walk of frames 1 and 2 with no other field
# reaching it. With the Euler integrator `(v2 - v1) / timestep` is Blender's
# acceleration at the starting state, which is how the force-field spec's
# laws were measured. The reference is the add-on's own record of the same
# object, `core.force_field.record_at_frame`, evaluated by
# `core.field_eval` at the particles' own positions and velocities, in
# solver axes and back: so an axis or plane mistake in the add-on's sampling
# (Blender's (x, y, z) against the solver's (x, z, -y)), a Texture read in
# the wrong axes (`z_up`), or a law that drifted from Blender's fails here
# against Blender rather than against the project's own formula.
#
# Cases: Force Point and Plane, each also with Gravitation; Tube and Cone
# falloff; Z Direction Positive and Negative; Vortex Point and Plane with
# Inflow; Magnetic Point and Plane; Harmonic Point and Plane with damping;
# Drag Point and Plane; Wind; an image Texture (8 x 8, float, Non-Color, no
# interpolation, so Blender reads the nearest pixel exactly) whose red ramps
# along Blender X, green along Blender Y and blue mixes both, in RGB,
# Gradient and Curl modes with World and Object Coordinates, plus an sRGB
# byte image and a Clip Cube extension, where Blender Z enters the lookup;
# and the image readings Blender changes by a pixel's alpha or by Clamp,
# each over pixels whose alpha cycles through 0, 1/4, ..., 1: an 8-bit sRGB
# image with Use Alpha off (read premultiplied, the sRGB curve after the
# product), an 8-bit Non-Color one with Use Alpha on (read straight, 0 where
# the alpha is 0), a float one with Use Alpha on (divided by its alpha), and
# a float one with negative channels under Clamp (zeroed).
#
# NOT COMPARED, by design: Turbulence and Noise Amount are this project's own
# noise (the pattern is not Blender's), and Fluid Flow has
# its own scenarios. Wind is compared ONLY in its spatial law: on a particle
# Blender's Wind is a push `s W z_hat`, while this project's Wind is an AIR
# VELOCITY of the same `s W z_hat` that acts through the drag (a deliberate
# difference), so the check holds Blender's push against
# the add-on's air velocity, number for number, which pins the axis, the
# shape and the falloff and nothing about the coupling; no Wind sheet is
# solved.
#
# AN OBJECT COORDINATES TEXTURE IS READ UNDER THE OBJECT'S SCALE. Every field
# object, an Object Coordinates Texture's included, is scaled (1.3, 0.8,
# 1.1): Blender reads such an image at the point's coordinates under the
# object's FULL inverse matrix, scale included, and so do the record and the
# solver, at the local point divided by `noise_scale` with the Nabla
# differences taken there. Two of the three Object
# Coordinates cases are solved, RGB and Gradient, so the solver's lookup and
# its Nabla steps are both in the chain.
#
# EACH CASE WHOSE LAW IS IN QUESTION CARRIES A GUARD on its own oracle: the
# record with that law taken out must miss Blender by more than 0.05 of |a|,
# so the scene's points read pixels and places where the law matters.
# Measured: an Object Coordinates Texture's record read unscaled
# (`noise_scale` 1) misses by 1.4 to 1.7 of |a|; the transparency and Clamp
# cases' records with each pixel's color as stored (an sRGB 8-bit image
# through the sRGB curve, no alpha and no Clamp applied) by 0.92 to 2.0.
# Seen to fail on dev-head: the add-on's evaluator reading unscaled fails B
# and S of the Object Coordinates cases (S at 1.0 to 2.4 of |a|); the image
# read with its colors as stored fails the four transparency and Clamp B
# checks and D of the solved one (0.81 of |a|).
#
# Subtests (margins measured on dev-head, Blender 5.2, CPU backend):
#   B_<case>_matches_blender: for both particle systems, the reference at
#      each particle's (x, v) is Blender's acceleration, and so is
#      Visualize's own `force_field.evaluate` at the rest system's points:
#      each component within 1e-4 of the case's largest |a| (5e-4 for an
#      8-bit image) plus Blender's own float32 rounding of the velocity it
#      reads back, one spacing of |v2| over the timestep (1.2e-5 m/s^2 at
#      |V_REF| = 1.14). Measured past that rounding: at most 9.1e-7 of |a|
#      here and 1.3e-6 over twelve placements (110x and 77x under 1e-4); the
#      sRGB image 2.9e-5 here and 4.8e-5 at worst (10x under 5e-4), the
#      add-on linearizing its bytes by the sRGB formula where Blender goes
#      through its color management (the premultiplied 8-bit sRGB image
#      2.9e-5). The record sampled with its origin in Blender's axes, no
#      rotation and no `z_up` missed by 0.62 to 1.7 of
#      |a|, in every case. Each check also guards its own oracle: Blender's
#      largest |a| at least 0.05, every particle alive, a case with a cut
#      (Max Distance, Z Direction, a radial limit) reading zero at some
#      points and not at others, and a velocity-dependent type reading at
#      least 0.05 between the two systems.
#   S_<case>_solver_step_matches_the_reference: a sheet of eight small
#      triangles whose vertices are the case's first 24 particle points, in
#      a SHELL group of its own (Young's 0.01, bend 0) that the field alone
#      targets, starts at V_REF; one solver step of 0.01 s through encode ->
#      build -> run -> fetch, read back as (x1 - x0 - dt v0) / dt^2, must
#      match one backward Euler step of the add-on's record (explicit laws
#      at the start, implicit ones at the iterate, as the solver solves),
#      each component within 3e-3 of the largest |a| plus the PC2's own
#      float32 rounding of x0 and x1 over dt^2 (at most 2.4e-3 m/s^2 at the
#      sheets' |x| <= 3.2). Measured past that rounding: at most 3.3e-4 of
#      |a| (9x under), the solver's own convergence and the sheet's
#      elasticity; a payload whose origin travels in Blender's axes and
#      whose Texture has no `z_up` missed by 0.68 to 2.0 of |a|, in every
#      case. A Texture triangle reads one set of pixels, see
#      `oracle_points`.
#   D_<case>_solver_matches_blender: for the explicit-only types solved
#      (Force with Gravitation, Cone, Magnetic, Texture), that same solver
#      acceleration against Blender's own at the same points, within the
#      same bound, which puts the device in the same chain as the oracle.
#      Measured: at most 1.2e-4 of |a| past the rounding (26x under); the
#      payload above missed by 1.0 to 1.4 of |a|, in every case. The
#      implicit types (Vortex Inflow, Harmonic, Drag) differ from Blender's
#      explicit Euler by O(dt) on purpose and are held through B and S.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

# The add-on's Frame Count has a minimum of 10; only the first step is read.
_FRAME_COUNT = 10

# THE ORACLE: the cases, the particle measurement in Blender and the
# comparison against the add-on's record. A separate block, so a standalone
# measurement script can read it out of this file and run the same code under
# `blender -b`, without the rig, to re-measure the margins above.
_ORACLE = r"""
import math

import numpy as np

# One particle step; the solver step is the same length.
ORACLE_DT = 0.01
# The moving particle system's velocity and the sheets' initial velocity.
V_REF = np.array([0.8, -0.5, 0.6])
# Every field object's scale, an Object Coordinates Texture's included: it
# enters no law but an Object Coordinates lookup.
FIELD_SCALE = (1.3, 0.8, 1.1)
IMAGE_SIZE = 8
# A case's sheet lies within SHEET_RADIUS of its field (a world-coordinate
# Texture's within it of the world origin, where the image is); the grid of
# field origins is GRID apart, so no two sheets meet.
SHEET_RADIUS = 0.9
GRID = 2.2
EXTRA_RADIUS = 1.8
TRIANGLES = 8
EXTRA_POINTS = 12
EDGE = 0.04
SEED = 20261005


def blender_to_solver(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def solver_to_blender(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def oracle_cases():
    # (name, Blender type, field settings, solved through the add-on)
    cases = [
        ("force_point", "FORCE", {"shape": "POINT", "strength": 3.0, "falloff_power": 2.0,
                                  "use_max_distance": True, "distance_max": 1.0}, False),
        ("force_plane", "FORCE", {"shape": "PLANE", "strength": -2.0, "falloff_power": 1.0,
                                  "use_min_distance": True, "distance_min": 0.3}, False),
        ("force_point_gravitation", "FORCE", {"shape": "POINT", "strength": 4.0,
                                              "falloff_power": 1.0, "use_gravity_falloff": True,
                                              "use_min_distance": True, "distance_min": 0.5},
         True),
        ("force_plane_gravitation", "FORCE", {"shape": "PLANE", "strength": 1.5,
                                              "use_gravity_falloff": True}, False),
        ("force_tube", "FORCE", {"shape": "POINT", "falloff_type": "TUBE", "strength": 2.0,
                                 "falloff_power": 1.0, "radial_falloff": 2.0,
                                 "use_radial_max": True, "radial_max": 0.6,
                                 "use_radial_min": True, "radial_min": 0.15}, False),
        ("force_cone", "FORCE", {"shape": "PLANE", "falloff_type": "CONE", "strength": 20.0,
                                 "falloff_power": 1.0, "radial_falloff": 1.0,
                                 "use_radial_max": True, "radial_max": 70.0,
                                 "use_radial_min": True, "radial_min": 20.0}, True),
        ("force_zdir_positive", "FORCE", {"shape": "POINT", "strength": 2.5,
                                          "z_direction": "POSITIVE", "falloff_power": 1.0},
         False),
        ("force_zdir_negative", "FORCE", {"shape": "PLANE", "strength": 2.5,
                                          "z_direction": "NEGATIVE"}, False),
        ("wind_plane", "WIND", {"shape": "PLANE", "strength": 2.0, "falloff_power": 1.0,
                                "use_max_distance": True, "distance_max": 0.6}, False),
        ("vortex_point", "VORTEX", {"shape": "POINT", "strength": 2.0, "falloff_power": 1.0,
                                    "inflow": 0.7}, True),
        ("vortex_plane", "VORTEX", {"shape": "PLANE", "strength": 1.5, "falloff_power": 1.0,
                                    "inflow": 0.5, "use_max_distance": True,
                                    "distance_max": 0.7}, True),
        ("magnetic_point", "MAGNET", {"shape": "POINT", "strength": 3.0, "falloff_power": 1.0},
         True),
        ("magnetic_plane", "MAGNET", {"shape": "PLANE", "strength": -2.0, "falloff_power": 1.0},
         False),
        ("harmonic_point", "HARMONIC", {"shape": "POINT", "strength": 3.0, "rest_length": 0.4,
                                        "harmonic_damping": 0.3, "falloff_power": 1.0,
                                        "use_max_distance": True, "distance_max": 0.7}, True),
        ("harmonic_plane", "HARMONIC", {"shape": "PLANE", "strength": 2.0,
                                        "harmonic_damping": 0.5, "falloff_power": 1.0}, False),
        ("drag_point", "DRAG", {"shape": "POINT", "linear_drag": 0.6, "quadratic_drag": 0.4,
                                "falloff_power": 1.0, "use_max_distance": True,
                                "distance_max": 1.0}, False),
        ("drag_plane", "DRAG", {"shape": "PLANE", "linear_drag": 1.5, "quadratic_drag": 2.0,
                                "falloff_power": 2.0}, True),
    ]
    for coords in ("world", "object"):
        for mode in ("RGB", "GRADIENT", "CURL"):
            name = "texture_%s_%s" % (mode.lower(), coords)
            cases.append((name, "TEXTURE",
                          {"texture_mode": mode, "strength": 2.0,
                           "use_object_coords": coords == "object", "texture_nabla": 0.1,
                           "_image": "float", "_extension": "EXTEND"},
                          name in ("texture_curl_world", "texture_rgb_object",
                                   "texture_gradient_object")))
    cases.append(("texture_rgb_srgb", "TEXTURE",
                  {"texture_mode": "RGB", "strength": 2.0, "_image": "byte",
                   "_extension": "EXTEND"}, False))
    cases.append(("texture_gradient_clipcube", "TEXTURE",
                  {"texture_mode": "GRADIENT", "strength": 2.0, "texture_nabla": 0.1,
                   "falloff_power": 1.0, "_image": "float", "_extension": "CLIP_CUBE"}, False))
    # Transparency and Clamp (`oracle_image`), the 8-bit premultiplied case
    # solved too, in Object Coordinates so its sheet has a place of its own.
    cases.append(("texture_rgb_byte_premultiplied", "TEXTURE",
                  {"texture_mode": "RGB", "strength": 2.0, "use_object_coords": True,
                   "_image": "byte", "_alpha": True, "_use_alpha": False,
                   "_extension": "EXTEND"}, True))
    cases.append(("texture_rgb_byte_use_alpha", "TEXTURE",
                  {"texture_mode": "RGB", "strength": 2.0, "_image": "byte",
                   "_colorspace": "Non-Color", "_alpha": True, "_use_alpha": True,
                   "_extension": "EXTEND"}, False))
    cases.append(("texture_rgb_float_use_alpha", "TEXTURE",
                  {"texture_mode": "RGB", "strength": 2.0, "_image": "float", "_alpha": True,
                   "_use_alpha": True, "_extension": "EXTEND"}, False))
    cases.append(("texture_rgb_float_clamp", "TEXTURE",
                  {"texture_mode": "RGB", "strength": 2.0, "_image": "float", "_negative": True,
                   "_use_clamp": True, "_extension": "EXTEND"}, False))
    # The 26 places around the origin, then a layer further out along +X for
    # the cases past them; a case keeps its place as cases are added.
    grid = [(GRID * i, GRID * j, GRID * k) for i in (-1, 0, 1) for j in (-1, 0, 1)
            for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    grid += [(2.0 * GRID, GRID * j, GRID * k) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    out = []
    for n, (name, kind, settings, solved) in enumerate(cases):
        location = np.array(grid[n]) + np.array([0.07, -0.05, 0.03])
        rotation = (0.4 + 0.13 * n, -0.3 + 0.07 * n, 0.7 - 0.11 * n)
        world_texture = kind == "TEXTURE" and not settings.get("use_object_coords")
        out.append({"name": name, "kind": kind, "settings": settings, "solved": solved,
                    "location": location, "rotation": rotation, "scale": FIELD_SCALE,
                    "center": np.zeros(3) if world_texture else location,
                    "cut": bool(settings.get("use_max_distance")
                                or settings.get("z_direction", "BOTH") != "BOTH"
                                or settings.get("use_radial_max")),
                    "velocity": kind in ("VORTEX", "MAGNET", "HARMONIC", "DRAG")})
    return out


def oracle_image(name, settings):
    # Red ramps along the image's x (Blender X in world coordinates), green
    # along its y (Blender Y), blue mixes both, so each channel's gradient
    # and each component of the push is distinct. `_alpha` cycles the alpha
    # through 0, 1/4, 1/2, 3/4 and 1 from pixel to pixel; `_negative` runs
    # red from -0.6 and blue from -0.5, so Clamp has channels to zero. A
    # float image is Non-Color and an 8-bit one sRGB unless `_colorspace`
    # says otherwise.
    w = h = IMAGE_SIZE
    kind = settings["_image"]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=kind == "float")
    img.colorspace_settings.name = settings.get(
        "_colorspace", "Non-Color" if kind == "float" else "sRGB")
    i = np.arange(w)[None, :] + np.zeros((h, 1))
    j = np.arange(h)[:, None] + np.zeros((1, w))
    px = np.ones((h, w, 4))
    px[..., 0] = 0.1 + 0.8 * i / (w - 1)
    px[..., 1] = 0.2 + 0.5 * j / (h - 1)
    px[..., 2] = ((3 * i + 5 * j) % 8) / 7.0
    if settings.get("_alpha"):
        px[..., 3] = ((2 * i + 3 * j) % 5) / 4.0
    if settings.get("_negative"):
        px[..., 0] = -0.6 + 1.5 * i / (w - 1)
        px[..., 2] -= 0.5
    img.pixels = px.ravel().tolist()
    return img


def oracle_stored_colors(image):
    # The image's colors as stored, (H, W, 3) linear, row 0 at the bottom:
    # what a reading that applied no alpha and no Clamp would send (an sRGB
    # 8-bit image through the sRGB curve, as every reading does).
    w, h = image.size
    px = np.array(image.pixels[:], dtype=np.float64).reshape(h, w, 4)[..., :3]
    if image.colorspace_settings.name == "sRGB" and not image.is_float:
        px = np.where(px <= 0.04045, px / 12.92, ((px + 0.055) / 1.055) ** 2.4)
    return px


def oracle_guards(case):
    # The record with a law of this case taken out, (label, record -> record):
    # an Object Coordinates Texture read unscaled, and an image read with its
    # colors as stored where its alpha or Clamp changes Blender's reading.
    s = case["settings"]
    guards = []
    if case["kind"] == "TEXTURE" and s.get("use_object_coords"):
        guards.append(("read_unscaled", lambda rec: dict(
            rec, noise_scale=np.ones_like(np.asarray(rec["noise_scale"], dtype=np.float64)))))
    if s.get("_alpha") or s.get("_negative"):
        image = case["field"].field.texture.image
        guards.append(("colors_as_stored",
                       lambda rec: dict(rec, image=oracle_stored_colors(image))))
    return guards


def oracle_field(scene, case):
    coll = bpy.data.collections.new("FFP_" + case["name"])
    scene.collection.children.link(coll)
    f = bpy.data.objects.new("FFP_" + case["name"], None)
    coll.objects.link(f)
    f.location = tuple(case["location"])
    f.rotation_euler = case["rotation"]
    f.scale = case["scale"]
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    bpy.context.view_layer.objects.active = f
    f.select_set(True)
    bpy.ops.object.forcefield_toggle()
    fs = f.field
    fs.type = case["kind"]
    # Flow and Inflow are one stored value; a Vortex's Inflow comes after.
    fs.flow = 0.0
    for key, value in case["settings"].items():
        if not key.startswith("_"):
            setattr(fs, key, value)
    if case["kind"] == "TEXTURE":
        tex = bpy.data.textures.new("FFP_tex_" + case["name"], "IMAGE")
        tex.image = oracle_image("FFP_img_" + case["name"], case["settings"])
        tex.extension = case["settings"]["_extension"]
        tex.use_interpolation = False
        tex.use_alpha = case["settings"].get("_use_alpha", True)
        tex.use_clamp = case["settings"].get("_use_clamp", False)
        fs.texture = tex
    case["field"] = f
    case["collection"] = coll
    return f


def oracle_accept(case, x):
    # Keep a point clear of every place the law is discontinuous or
    # singular (the origin, the axis, the plane, each falloff limit, a pixel
    # edge), so a float32 rounding cannot put Blender and the reference on
    # two sides of it; the margins also cover the solver step's travel of
    # |V_REF| dt = 0.011.
    s = case["settings"]
    m = np.array(case["field"].matrix_world, dtype=np.float64)
    rot = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    p = rot.T @ (x - m[:3, 3])
    d = float(np.linalg.norm(p))
    r = math.hypot(p[0], p[1])
    pz = float(p[2])
    if d < 0.3 or r < 0.1 or abs(pz) < 0.08:
        return False
    point = s.get("shape") == "POINT"
    if s.get("use_gravity_falloff") and (d if point else abs(pz)) < 0.45:
        return False
    if case["kind"] == "VORTEX" and not point:
        axial = r
    elif point and s.get("falloff_type", "SPHERE") == "SPHERE":
        axial = d
    else:
        axial = abs(pz)
    for key in ("distance_min", "distance_max"):
        if key in s and abs(axial - s[key]) < 0.06:
            return False
    radial = math.degrees(math.atan2(r, pz)) if s.get("falloff_type") == "CONE" else r
    margin = 3.0 if s.get("falloff_type") == "CONE" else 0.06
    for key in ("radial_min", "radial_max"):
        if key in s and abs(radial - s[key]) < margin:
            return False
    if case["kind"] == "TEXTURE":
        for qq in oracle_lookups(case, x):
            for c in (0, 1):
                u = (qq[c] + 1.0) * 0.5 * IMAGE_SIZE
                if abs(u - round(u)) < 2e-3:
                    return False
            if s["_extension"] == "CLIP_CUBE":
                if abs(abs(qq[2]) - 1.0) < 2e-3 * 2.0 / IMAGE_SIZE:
                    return False
    return True


def oracle_lookups(case, x):
    # Where a Texture reads its image for the point x: the texture
    # coordinates (World: the point; Object: under the field's full inverse
    # matrix, Blender's own reading) and one Nabla along each axis.
    s = case["settings"]
    if s.get("use_object_coords"):
        m = np.array(case["field"].matrix_world, dtype=np.float64)
        q = (np.linalg.inv(m) @ np.append(x, 1.0))[:3]
    else:
        q = np.asarray(x, dtype=np.float64)
    nabla = s.get("texture_nabla", 0.025)
    return [q] + [q + nabla * e for e in np.eye(3)]


def oracle_texels(case, x):
    # The pixels (and the Clip Cube side) every lookup of x lands on.
    out = []
    for qq in oracle_lookups(case, x):
        out += [int(math.floor((qq[c] + 1.0) * 0.5 * IMAGE_SIZE)) for c in (0, 1)]
        out.append(abs(qq[2]) > 1.0)
    return tuple(out)


def oracle_points(case, rng):
    # TRIANGLES small triangles within SHEET_RADIUS of the case's center,
    # each vertex accepted (the sheet the solver steps), then EXTRA_POINTS
    # single points out to EXTRA_RADIUS (Blender only), drawn from an
    # anisotropic normal, so the points are not a regular grid.
    center = case["center"]
    spread = np.array([0.55, 0.35, 0.45])
    tris, centers = [], []
    tries = 0
    while len(tris) < TRIANGLES:
        tries += 1
        if tries > 200000:
            raise RuntimeError("no sheet placement for case %s" % case["name"])
        c = center + rng.normal(0.0, 1.0, 3) * spread
        if np.linalg.norm(c - center) > SHEET_RADIUS - EDGE:
            continue
        if any(np.linalg.norm(c - o) < 4.0 * EDGE for o in centers):
            continue
        u = rng.normal(0.0, 1.0, 3)
        u /= np.linalg.norm(u)
        w = np.cross(u, rng.normal(0.0, 1.0, 3))
        w /= np.linalg.norm(w)
        verts = [c + EDGE * (u * math.cos(t) + w * math.sin(t))
                 for t in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)]
        if not all(oracle_accept(case, v) for v in verts):
            continue
        # A Texture is piecewise constant over pixels: a triangle whose
        # vertices read different pixels is stretched by the push itself,
        # and its elasticity, small as it is, moves the vertices off their
        # own field (measured 1e-3 of |a| for a curl that steps by 14 across
        # one triangle), so a sheet triangle reads one set of pixels.
        if case["kind"] == "TEXTURE" and len({oracle_texels(case, v) for v in verts}) != 1:
            continue
        tris.append(verts)
        centers.append(c)
    extra = []
    while len(extra) < EXTRA_POINTS:
        tries += 1
        if tries > 400000:
            raise RuntimeError("no extra points for case %s" % case["name"])
        x = center + rng.normal(0.0, 1.0, 3) * spread * 2.0
        if np.linalg.norm(x - center) <= EXTRA_RADIUS and oracle_accept(case, x):
            extra.append(x)
    case["sheet"] = np.array(tris).reshape(-1, 3)
    case["points"] = np.concatenate([case["sheet"], np.array(extra)])
    return case["points"]


def oracle_emitter(scene, case, tag, velocity):
    pts = case["points"]
    mesh = bpy.data.meshes.new("FFPpts_%s_%s" % (case["name"], tag))
    mesh.from_pydata([tuple(float(c) for c in p) for p in pts], [], [])
    em = bpy.data.objects.new("FFPem_%s_%s" % (case["name"], tag), mesh)
    scene.collection.objects.link(em)
    em.modifiers.new("ps", "PARTICLE_SYSTEM")
    ps = em.particle_systems[0].settings
    ps.count = len(pts)
    ps.frame_start = ps.frame_end = 1
    ps.lifetime = 1000
    ps.emit_from = "VERT"
    ps.use_emit_random = False
    ps.normal_factor = 0.0
    ps.object_align_factor = tuple(float(c) for c in velocity)
    ps.factor_random = 0.0
    ps.physics_type = "NEWTON"
    ps.mass = 1.0
    ps.effector_weights.gravity = 0.0
    # Only this case's field reaches this system.
    ps.effector_weights.collection = case["collection"]
    ps.use_rotations = False
    ps.integrator = "EULER"
    ps.drag_factor = ps.brownian_factor = ps.damping = 0.0
    ps.timestep = ORACLE_DT
    ps.subframes = 0
    case.setdefault("emitters", {})[tag] = em
    return em


def oracle_build(scene, cases):
    rng = np.random.default_rng(SEED)
    scene.frame_start, scene.frame_end = 1, 10
    scene.use_gravity = False
    for case in cases:
        oracle_field(scene, case)
    bpy.context.view_layer.update()
    for case in cases:
        oracle_points(case, rng)
        oracle_emitter(scene, case, "rest", np.zeros(3))
        oracle_emitter(scene, case, "moving", V_REF)


def oracle_walk(scene, cases):
    # Frames 1 then 2, in order; the systems are emitted at frame 1.
    for frame in (1, 2):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        for case in cases:
            for tag, em in case["emitters"].items():
                ps = em.evaluated_get(dg).particle_systems[0].particles
                case.setdefault("states", {})[(tag, frame)] = (
                    np.array([tuple(q.location) for q in ps], dtype=np.float64),
                    np.array([tuple(q.velocity) for q in ps], dtype=np.float64),
                    sorted({q.alive_state for q in ps}))
    scene.frame_set(1)
    bpy.context.view_layer.update()


def oracle_compare(case, ff, fe, tolerance):
    # Blender's acceleration against the add-on's record, both systems, and
    # Visualize's own evaluate at the rest points. Each component may miss
    # by `tolerance` of the case's largest |a| plus Blender's own float32
    # rounding of the velocity it reads back: one spacing of |v2| over the
    # timestep (1.2e-5 m/s^2 at |V_REF| = 1.14), which a slow-falloff case
    # with a small |a| would otherwise charge to the law. Each guard's
    # record (`oracle_guards`) must miss the rest system by over 0.05 of
    # |a|.
    f = case["field"]
    rec = ff.record_at_frame(f, 1.0)
    wind = case["kind"] == "WIND"
    blender, rows, floors, errors = {}, {}, {}, {}
    for tag in ("rest", "moving"):
        x1, v1, alive1 = case["states"][(tag, 1)]
        x2, v2, alive2 = case["states"][(tag, 2)]
        a = (v2 - v1) / ORACLE_DT
        acc, air = fe.evaluate(rec, blender_to_solver(x1), velocities=blender_to_solver(v1),
                               normals=np.zeros((len(x1), 3)))
        ref = solver_to_blender(air if wind else acc)
        blender[tag] = a
        floors[tag] = np.spacing(np.abs(v2).astype(np.float32)).astype(np.float64) / ORACLE_DT
        errors[tag] = np.abs(a - ref)
        rows[tag] = {"error": float(errors[tag].max()),
                     "alive": alive1 == ["ALIVE"] and alive2 == ["ALIVE"],
                     "placed": float(np.abs(x1 - case["points"]).max()),
                     "worst": int(errors[tag].max(axis=1).argmax()),
                     "zero": int((np.abs(a).max(axis=1) < 1e-9).sum()),
                     "count": len(a)}
        if tag == "rest":
            acc_v, air_v = ff.evaluate([f], x1, rate=1.0)
            errors["visualize"] = np.abs((air_v if wind else acc_v) - a)
            floors["visualize"] = floors["rest"]
            rows["visualize_error"] = float(errors["visualize"].max())
    scale = max(float(np.abs(blender["rest"]).max()), float(np.abs(blender["moving"]).max()))
    velocity_effect = float(np.abs(blender["moving"] - blender["rest"]).max())
    excess = max(float(np.max(errors[k] - floors[k])) for k in errors) / max(scale, 1e-30)
    x1, v1, _ = case["states"][("rest", 1)]
    guarded = {}
    for label, without in oracle_guards(case):
        acc_g, air_g = fe.evaluate(without(rec), blender_to_solver(x1),
                                   velocities=blender_to_solver(v1),
                                   normals=np.zeros((len(x1), 3)))
        guarded[label] = float(np.abs(solver_to_blender(air_g if wind else acc_g)
                                      - blender["rest"]).max()) / max(scale, 1e-30)
    moving = rows["moving"]
    ok = (scale > 0.05 and rows["rest"]["alive"] and moving["alive"]
          and all(bool(np.all(errors[k] <= tolerance * scale + floors[k])) for k in errors))
    if case["cut"]:
        ok = ok and 0 < moving["zero"] < moving["count"]
    if case["velocity"]:
        ok = ok and velocity_effect > 0.05
    ok = ok and all(miss > 0.05 for miss in guarded.values())
    case["blender"] = blender
    case["record"] = rec
    details = {"scale": scale, "relative": max(float(e.max()) for e in errors.values())
               / max(scale, 1e-30), "excess_over_rounding": excess,
               "tolerance": tolerance, "velocity_effect": velocity_effect,
               "law_removed_misses_by": guarded, "rows": rows}
    return ok, details
"""

_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>

try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["evaluate"])
    fe = __import__(pkg + ".core.field_eval", fromlist=["evaluate"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    dh.save_blend(PROBE_DIR, "force_field_blender_parity.blend")

    # ----- B: Blender's particles against the add-on's record ---------------
    cases = oracle_cases()
    oracle_build(scene, cases)
    oracle_walk(scene, cases)
    for case in cases:
        tol = 5e-4 if case["settings"].get("_image") == "byte" else 1e-4
        ok, details = oracle_compare(case, ff, fe, tol)
        dh.record("B_%s_matches_blender" % case["name"], ok, details)

    # ----- S and D: the solver, one step through the add-on -----------------
    solved = [c for c in cases if c["solved"]]
    for case in cases:
        for em in case["emitters"].values():
            bpy.data.objects.remove(em)
        if not case["solved"]:
            bpy.data.objects.remove(case["field"])
    root = dh.configure_state(project_name="force_field_blender_parity",
                              frame_count=FRAME_COUNT, frame_rate=100, step_size=ORACLE_DT)
    state = root.state
    for case in solved:
        verts = [tuple(float(c) for c in v) for v in case["sheet"]]
        faces = [(3 * t, 3 * t + 1, 3 * t + 2) for t in range(len(verts) // 3)]
        mesh = bpy.data.meshes.new("FFPsheet_" + case["name"])
        mesh.from_pydata(verts, [], faces)
        obj = bpy.data.objects.new("FFPsheet_" + case["name"], mesh)
        scene.collection.objects.link(obj)
        group = dh.api.solver.create_group("FFP_" + case["name"], "SHELL")
        group.add(obj.name)
        group.param.shell_young_modulus = 0.01
        group.param.bend = 0.0
        speed = float(np.linalg.norm(V_REF))
        group.set_velocity(obj.name, direction=tuple(float(c) for c in V_REF / speed),
                           speed=speed, frame=1)
        dh.api.solver.set_force_field_targets(case["field"].name, [group])
        case["sheet_object"] = obj
    bpy.context.view_layer.update()

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="force_field_blender_parity:build",
                      timeout=300.0)
    dh.run_and_wait(timeout=600.0)
    run_state = {"solver_state": dh.facade.engine.state.solver.name,
                 "error": dh.facade.engine.state.error}
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    def step_reference(rec, x, v0, dt):
        # One backward Euler step of a free vertex, what the solver solves:
        # (y - x - v0 dt) / dt^2 is the explicit laws at the step's start
        # (x, v0) plus the implicit ones at the iterate (y, (y - x) / dt),
        # by Newton with a central-difference Jacobian. Solver axes.
        explicit = fe.evaluate(rec, x, v0, 0.0, normals=np.zeros_like(x), part="explicit")[0]

        def implicit_at(y):
            return fe.evaluate(rec, y, (y - x) / dt, 0.0, part="implicit")[0]

        y = x + v0 * dt + dt * dt * explicit
        h = 1e-6
        for _ in range(60):
            residual = (y - x - v0 * dt) / (dt * dt) - explicit - implicit_at(y)
            jac = np.zeros((len(x), 3, 3))
            for k in range(3):
                e = np.zeros(3)
                e[k] = h
                jac[:, :, k] = np.eye(3)[k] / (dt * dt) - (
                    implicit_at(y + e) - implicit_at(y - e)) / (2.0 * h)
            dy = np.linalg.solve(jac, -residual[..., None])[..., 0]
            y = y + dy
            if float(np.abs(dy).max()) < 1e-13:
                break
        return y

    dt = ORACLE_DT
    for case in solved:
        obj = case["sheet_object"]
        path = dh.find_pc2_for(obj)
        rows_ = dh.read_pc2(path) if path else None
        details = dict(run_state)
        details["rows"] = None if rows_ is None else int(rows_.shape[0])
        ok_s = ok_d = False
        if rows_ is not None and rows_.shape[0] >= 2 and np.all(np.isfinite(rows_[:2])):
            # The sheet sits at the identity, so PC2 rows are world positions;
            # row 0 is the rest pose, row 1 solver frame 1, one step later.
            x0 = rows_[0].astype(np.float64)
            x1 = rows_[1].astype(np.float64)
            # float32 rounding of x0 and x1, each half a spacing, over dt^2.
            quantum = (np.spacing(np.abs(rows_[0]).astype(np.float32)).astype(np.float64)
                       + np.spacing(np.abs(rows_[1]).astype(np.float32)).astype(np.float64)
                       ) * 0.5 / (dt * dt)
            got = (x1 - x0 - dt * V_REF) / (dt * dt)
            x0_s = blender_to_solver(x0)
            v0_s = np.tile(blender_to_solver(V_REF), (len(x0), 1))
            y = step_reference(case["record"], x0_s, v0_s, dt)
            want = solver_to_blender(y - x0_s - dt * v0_s) / (dt * dt)
            scale = float(np.abs(want).max())
            err = np.abs(got - want)
            excess = float(np.max(err - quantum))
            # The worst vertex, and how far the field differs across its own
            # triangle, which is what the sheet's elasticity can feel.
            worst = int(np.max(err - quantum, axis=1).argmax())
            tri = want[3 * (worst // 3):3 * (worst // 3) + 3]
            details.update({"scale": scale, "max_error": float(err.max()),
                            "relative": float(err.max()) / max(scale, 1e-30),
                            "excess_over_rounding": excess / max(scale, 1e-30),
                            "rounding": float(quantum.max()), "worst_vertex": worst,
                            "worst_got": got[worst].tolist(), "worst_want": want[worst].tolist(),
                            "worst_triangle_spread": float(np.abs(tri - tri[0]).max())})
            ok_s = scale > 0.05 and bool(np.all(err <= quantum + 3e-3 * scale))
            n = len(x0)
            blender = case["blender"]["moving"][:n]
            err_b = np.abs(got - blender)
            details["blender_relative"] = float(err_b.max()) / max(scale, 1e-30)
            details["blender_excess"] = float(np.max(err_b - quantum)) / max(scale, 1e-30)
            ok_d = scale > 0.05 and bool(np.all(err_b <= quantum + 3e-3 * scale))
        dh.record("S_%s_solver_step_matches_the_reference" % case["name"], ok_s, details)
        if not case["velocity"] or case["kind"] == "MAGNET":
            dh.record("D_%s_solver_matches_blender" % case["name"], ok_d,
                      {k: details.get(k) for k in ("scale", "blender_relative",
                                                    "blender_excess", "rounding")})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _ORACLE + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
