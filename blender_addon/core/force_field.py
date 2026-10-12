# File: force_field.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Blender force fields, and the exact force-field script, for the solver.

The solver takes an external force field as FIELD OBJECTS, each evaluated
exactly at every vertex on the device, and one exact script
(``frontend/_force_field_.py``). This module turns the scene's force-field
objects into field objects and is also what the viewport's Visualize overlay
draws from, so what is drawn is what is sent.

WHAT EACH BLENDER FIELD BECOMES was measured on Blender as a black box, law
by law, and ``core/field_eval.py`` (byte-identical to
``frontend/_field_eval_.py``) computes each law in double
precision. A field object is sent as its settings and a TRACK of per-frame
samples: its world origin and axes and every animatable value, sampled at
every solve frame when anything can move or animate it, once otherwise.
Nothing per vertex is sent, and nothing per volume but a Fluid Flow's
fluid, whose velocity IS volume data: its Domain Object,
a BAKED Mantaflow gas domain or a Volume object, is read at every solve
frame and sent as sparse bricks. Object scale reaches a field only as the
size of its Noise Amount's gusts and the scale of an Object Coordinates
texture's lookup (measured: scale enters nothing else).

EVERYTHING ELSE IS REFUSED BY NAME at Transfer rather than ignored
(:func:`refusal`): a type that has no effect on cloth (Charge, Boid,
Lennard-Jones, Curve Guide, all measured), a setting Blender honors and
this solver does not read (an image texture's Flip Axis, Crop, Repeat or
Colors, Noise Amount or Wind Factor on a Fluid Flow, Apply Density on any
other type), a Fluid Flow whose Domain Object is not a baked gas domain or
a velocity volume, the ones still to come (procedural textures,
Absorption), a Turbulence of Size 0 and Wind in a scene with no Air
Density. A refused field reported loudly is better than a field that
silently does nothing, which is issue #114.

THE TURBULENCE PATTERN AND THE NOISE AMOUNT ARE NOT BLENDER'S. Blender's
noise implementation is GPL and cannot be copied into this project, so the
noise is this project's own (`core/noise.py`, the same algorithm the solver
runs): Size, Strength, Noise Amount and Seed mean what they mean in Blender,
and the swirls are a different random pattern.

EACH SOURCE REACHES EVERY GROUP OR THE GROUPS IT NAMES
(`models/force_field_targets.py`).
"""

from __future__ import annotations

import json
import math
import os
import threading
import zlib

from typing import NamedTuple

import numpy as np

from . import field_eval as _field_eval
from . import noise as _noise
from . import script_api as _script_api

SUPPORTED_TYPES = ("FORCE", "WIND", "VORTEX", "TURBULENCE", "MAGNET", "DRAG", "HARMONIC",
                   "TEXTURE", "FLUID_FLOW")
SUPPORTED_SHAPES = ("POINT", "PLANE")
SUPPORTED_FALLOFF = ("SPHERE", "TUBE", "CONE")
# Blender's type identifiers, as the solver names them.
SOLVER_TYPE = {"FORCE": "force", "WIND": "wind", "VORTEX": "vortex",
               "TURBULENCE": "turbulence", "MAGNET": "magnetic", "DRAG": "drag",
               "HARMONIC": "harmonic", "TEXTURE": "texture", "FLUID_FLOW": "fluid"}
# Types measured to have no effect on Blender's own cloth,
# refused rather than sent as nothing.
NO_EFFECT_ON_CLOTH = {"CHARGE": "Charge", "BOID": "Boid", "LENNARDJ": "Lennard-Jones",
                      "GUIDE": "Curve Guide"}
# Types whose law is measured and whose solver term is still to come.
NOT_YET: dict = {}
# The image colorspaces whose values the add-on can turn into the LINEAR
# values the solver reads (measured: Blender reads an sRGB image through the
# sRGB curve).
LINEAR_COLORSPACES = ("Non-Color", "Linear Rec.709", "Linear", "Raw")

# An image texture's settings that change what Blender's texture returns
# (measured black box through `Texture.evaluate`) and that this solver does
# not read, each with its default and its label in Blender's panel. Repeat
# and Mirror act only under the Repeat extension (measured: Extend and Clip
# read alike at any Repeat), so they are checked only there.
TEXTURE_DEFAULTS = (("use_flip_axis", False, "Flip Axis"),
                    ("crop_min_x", 0.0, "Crop Minimum X"),
                    ("crop_min_y", 0.0, "Crop Minimum Y"),
                    ("crop_max_x", 1.0, "Crop Maximum X"),
                    ("crop_max_y", 1.0, "Crop Maximum Y"),
                    ("use_color_ramp", False, "Use Color Ramp"),
                    ("intensity", 1.0, "Brightness"),
                    ("contrast", 1.0, "Contrast"),
                    ("saturation", 1.0, "Saturation"),
                    ("factor_red", 1.0, "Factor Red"),
                    ("factor_green", 1.0, "Factor Green"),
                    ("factor_blue", 1.0, "Factor Blue"),
                    ("filter_size", 1.0, "Filter Size"),
                    ("use_calculate_alpha", False, "Calculate Alpha"))
TEXTURE_REPEAT_DEFAULTS = (("repeat_x", 1, "Repeat X"), ("repeat_y", 1, "Repeat Y"),
                           ("use_mirror_x", False, "Mirror X"),
                           ("use_mirror_y", False, "Mirror Y"))
# Image sources whose pixels are not one still picture the add-on can read.
IMAGE_SOURCES_REFUSED = {
    "SEQUENCE": "an image sequence, which would be read as one still frame",
    "MOVIE": "a movie, which would be read as one still frame",
    "TILED": "a tiled (UDIM) image",
    "VIEWER": "a Render Result or Viewer image",
}

# Arrow colors for the overlay: acceleration and air velocity.
COLOR_ACCELERATION = (1.0, 0.55, 0.15, 0.9)
COLOR_AIR = (0.3, 0.8, 1.0, 0.9)
COLOR_SCRIPT = (0.85, 0.4, 1.0, 0.9)


# --- which objects ---------------------------------------------------------


def field_objects(scene, state) -> list:
    """The objects whose force field the solver receives.

    Every object with a field in the chosen collection (its children
    included), or in the whole scene when none is chosen.
    """
    coll = getattr(state, "force_field_collection", None)
    source = coll.all_objects if coll is not None else scene.objects
    found = []
    for obj in source:
        field = getattr(obj, "field", None)
        if field is not None and field.type != "NONE":
            found.append(obj)
    found.sort(key=lambda o: o.name)
    return found


def refusal(obj, frames=None) -> str | None:
    """Why ``obj``'s field cannot be sent, or ``None`` when it can.

    ``frames`` is the solve's ``(first Blender frame, frame count)``, which a
    Fluid Flow's baked range must cover; ``None`` takes it from the scene's
    add-on settings. The reason does not name the object: every place that
    shows it (the panel's row, the MCP listing, the Transfer error) names it
    already.
    """
    f = obj.field
    if f.type in NO_EFFECT_ON_CLOTH:
        return f"{NO_EFFECT_ON_CLOTH[f.type]} fields have no effect on cloth in Blender either"
    if f.type in NOT_YET:
        return f"{NOT_YET[f.type]} fields are not supported yet"
    if f.type not in SUPPORTED_TYPES:
        return f"{f.type.title()} fields are not supported"
    if f.shape not in SUPPORTED_SHAPES:
        return f"shape {f.shape.title()} is not supported; use Point or Plane"
    if f.falloff_type not in SUPPORTED_FALLOFF:
        return f"{f.falloff_type.title()} falloff is not supported; use Sphere, Tube or Cone"
    # A Wind field IS an air flow here, which is what Blender's Flow setting
    # asks for (and Wind defaults to Flow 1), so it is read for no other type.
    # A Vortex is the exception: Blender stores its Inflow in the same
    # property (measured: setting either sets both), and Inflow is read.
    if f.type not in ("WIND", "VORTEX", "FLUID_FLOW") and abs(float(f.flow)) > 0.0:
        return f"Flow {f.flow:g} applies only to a Wind or a Fluid Flow field here; set it to 0"
    # Blender stores a Harmonic's Multiple Springs in the 2D flag (measured:
    # setting one sets the other), and a Harmonic's panel shows only the
    # former, so the reason names the setting the artist can see.
    if f.use_2d_force:
        if f.type == "HARMONIC":
            return "Multiple Springs is not supported; turn it off"
        return "2D has no effect on cloth in Blender either; turn it off"
    if f.use_absorption:
        return "Absorption is not supported yet; turn it off" + _hidden_on_fluid_flow(f)
    if not f.apply_to_location:
        return "Affect Location is off, so the field would move nothing; turn it on"
    # Apply Density weighs a field by the density of the smoke it acts in,
    # and a Fluid Flow is the one type whose fluid this solver reads.
    if f.type != "FLUID_FLOW" and f.use_smoke_density:
        return "Apply Density is read only on a Fluid Flow field here; turn it off"
    if f.type == "TEXTURE":
        why = _texture_refusal(f)
        if why:
            return why
    if f.type == "FLUID_FLOW":
        why = fluid_source_refusal(f, frames)
        if why:
            return why
        # Blender applies both on a Fluid Flow, and Absorption too (measured
        # on particles and on a cloth sheet lying along a baked flow: Noise
        # Amount 2 changed them by 0.042 of 0.091 and 1.1e-4 of 3.4e-4, Wind
        # Factor 1 took the sheet's whole push off, Absorption behind a
        # collision wall the whole push off both), and this solver's fluid
        # reads none of them, so each is refused, not dropped. Gravitation is
        # the one Blender ignores there (both moved identically), so
        # `_discrete` sends it off instead of refusing it.
        if float(f.noise) != 0.0:
            return ("Noise Amount on a Fluid Flow field is not supported; set it to 0"
                    + _hidden_on_fluid_flow(f))
        if float(f.wind_factor) != 0.0:
            return ("Wind Factor on a Fluid Flow field is not supported; set it to 0"
                    + _hidden_on_fluid_flow(f))
    # A new Turbulence field has Size 0, and the swirls are Size across, so
    # at 0 the noise would vary from one vertex to the next.
    if f.type == "TURBULENCE" and float(f.size) <= 0.0:
        return "Size is 0; set it above 0 (the size of the swirls)"
    # The object's scale is the size of its Noise Amount's gusts, and its
    # axes are the field's frame: a zero scale gives neither, and a mirrored
    # one turns the frame inside out.
    if obj.matrix_world.to_3x3().determinant() <= 0.0:
        return "its scale is zero or mirrored (negative); give it a positive scale"
    # Blender reads an Object Coordinates texture under the object's full
    # inverse world matrix (measured). The record carries a rotation and a
    # scale per axis, which is that inverse only when the matrix is not
    # sheared; under a rotated, non-uniformly scaled parent it is.
    if f.type == "TEXTURE" and f.use_object_coords and _sheared(obj.matrix_world):
        return ("its transform is sheared (a rotated child of a non-uniformly scaled "
                "parent), which Use Coordinates would read differently from Blender; "
                "apply the parent's scale, or turn off Use Coordinates")
    return None


def _sheared(matrix_world) -> bool:
    """Whether a world matrix's axes are not at right angles."""
    linear = np.array(matrix_world, dtype=np.float64)[:3, :3]
    axes = linear / np.linalg.norm(linear, axis=0)
    off = axes.T @ axes - np.eye(3)
    return bool(np.abs(off).max() > 1e-4)


def _hidden_on_fluid_flow(f) -> str:
    """How to reach a setting the Fluid Flow panel does not show, for a
    refusal of one on a Fluid Flow field; empty for another type. Measured:
    Blender's Fluid Flow panel draws no Noise Amount, Wind Factor,
    Absorption or Gravitation, a Force's draws all four, and a value
    outlives a change of type (a Wind's Wind Factor 1 stays on the Fluid
    Flow it is turned into)."""
    if f.type != "FLUID_FLOW":
        return ""
    return (" (the Fluid Flow panel does not show it: set the field's type to Force, change "
            "it there, then set the type back)")


def _texture_refusal(f) -> str | None:
    tex = f.texture
    if tex is None:
        return "a Texture field with no texture"
    if getattr(tex, "use_nodes", False):
        return "a texture driven by nodes is not supported; use an image texture"
    if tex.type != "IMAGE":
        return (f"{tex.type.title()} textures are not supported yet; an image texture is "
                "(procedural ones come in a later phase)")
    if tex.image is None:
        return "its image texture has no image"
    if tex.image.source in IMAGE_SOURCES_REFUSED:
        return (f"its image '{tex.image.name}' is {IMAGE_SOURCES_REFUSED[tex.image.source]}; "
                "use a still image")
    if tex.extension == "CHECKER":
        return "the Checker extension is not supported; use Extend, Repeat, Clip or Clip Cube"
    if f.use_root_coords:
        return "Root Texture Coordinates acts on particles only; turn it off"
    space = tex.image.colorspace_settings.name
    if space != "sRGB" and space not in LINEAR_COLORSPACES:
        return (f"its image's colorspace {space!r} is not supported; use sRGB or a linear "
                f"one ({', '.join(LINEAR_COLORSPACES)})")
    checked = TEXTURE_DEFAULTS + (TEXTURE_REPEAT_DEFAULTS if tex.extension == "REPEAT" else ())
    for prop, default, label in checked:
        value = getattr(tex, prop)
        if value != default:
            if isinstance(default, bool):
                return f"the texture's {label} is on, which this solver does not read; turn it off"
            return (f"the texture's {label} is {value:g}, which this solver does not read; "
                    f"set it to {default:g}")
    return None


# --- Fluid Flow: the fluid's own grid -----------------------------------------

# A Mantaflow grid's velocity is in units of FLUID_GRID_UNIT cells per second
# (measured), the cell being the longest axis's.
FLUID_GRID_UNIT = 2.5
# A Volume's density grid. Blender's Volume names its velocity grids
# (`velocity_grid`, `velocity_x_grid` ...) and no density grid.
VDB_DENSITY_NAME = "density"
VDB_VECTOR_TYPES = ("vec3s", "vec3d")
VDB_SCALAR_TYPES = ("float", "double")
# Metadata Mantaflow writes on every grid of its OpenVDB cache and openvdb's
# own writer does not (measured), so a cache file is recognized wherever it
# was copied to.
MANTAFLOW_CACHE_METADATA = "file_base_resolution"
# The most cells a fluid frame may have along an axis: the solver's
# FLUID_MAX_RESOLUTION (crates/ppf-cts-solver/src/force_field.rs).
# addon_host_tests/_force_field_volume_.py holds the two equal.
FLUID_MAX_RESOLUTION = 1 << 15
# A large byte string (a fluid frame's table and bricks, a texture's image)
# travels as a LIST of `bytes` chunks of at most this size, because cbor2
# 6.0.1 decodes one bytes item in time quadratic in its length. The value
# and its measurements are frontend/_cbor_bridge_.PICKLE_CHUNK_BYTES's;
# addon_host_tests/_force_field_volume_.py holds the two equal. A chunk is
# `bytes`, never a memoryview, which cbor2 would encode as integers.
PAYLOAD_CHUNK_BYTES = 1 << 20
# The most a transfer carries: the server refuses an upload whose scene
# geometry and parameters together pass MAX_PAYLOAD_BYTES before reading it
# (crates/ppf-cts-server/src/wire/mod.rs and upload.rs), and on a streamed
# connection that refusal reaches the client as a reset socket. So it is
# refused here by name instead: an encode stops once its fluid volumes and
# images pass it, and every upload is measured whole before it starts
# (`upload_refusal`). addon_host_tests/_force_field_volume_.py holds the two
# equal.
TRANSPORT_CAP_BYTES = 4 * 1024 * 1024 * 1024


def _fluid_domain_modifier(obj):
    """``obj``'s Mantaflow domain modifier, or None."""
    for mod in getattr(obj, "modifiers", ()):
        if mod.type == "FLUID" and mod.fluid_type == "DOMAIN":
            return mod
    return None


def _fluid_domain(obj):
    """``obj``'s Mantaflow domain settings, or None."""
    mod = _fluid_domain_modifier(obj)
    return None if mod is None else mod.domain_settings


def _enabled(mod) -> bool:
    return bool(mod.show_viewport or mod.show_render)


def _modifiers_after(obj, mod) -> list:
    """The names of ``obj``'s enabled modifiers stacked after ``mod``."""
    names, past = [], False
    for m in obj.modifiers:
        if past and _enabled(m):
            names.append(m.name)
        if m.name == mod.name:
            past = True
    return names


def _solve_frames():
    """The scene's solve as ``(first Blender frame, frame count)``, or None
    before the add-on's settings exist on the scene."""
    import bpy

    from ..models.groups import get_addon_data, has_addon_data
    from .encoder import resolve_start_frame

    scene = bpy.context.scene
    if not has_addon_data(scene):
        return None
    state = get_addon_data(scene).state
    return resolve_start_frame(state), max(1, int(state.frame_count))


def fluid_rate(scene, solver_fps: float) -> float:
    """The factor that turns a fluid velocity per SCENE second into one per
    SOLVER second: ``solver_fps / scene fps``.

    A baked fluid is motion indexed by Blender frame, and the encoder places
    frame k at ``k / solver_fps`` solver seconds (``resolve_solver_fps``: the
    add-on's frame rate times Time Scale). The smoke therefore covers, per
    solver second, ``solver_fps / scene fps`` times what it covers per scene
    second, and the cloth rides it frame for frame only if its velocity is
    converted by the same factor. With Use Scene FPS on this is Time Scale
    times the scene's Base (``render.fps_base``), because the add-on's frame
    rate is the scene's whole-number FPS: Time Scale itself at a rate such as
    24 or 30, and 1.001 times it at 29.97.
    """
    return float(solver_fps) / (float(scene.render.fps) / float(scene.render.fps_base))


def _scene_fluid_rate() -> float:
    """:func:`fluid_rate` for the scene's own settings, 1 before the
    add-on's settings exist on the scene."""
    import bpy

    from ..models.groups import get_addon_data, has_addon_data
    from .encoder import resolve_solver_fps

    scene = bpy.context.scene
    if not has_addon_data(scene):
        return 1.0
    return fluid_rate(scene, resolve_solver_fps(get_addon_data(scene).state))


# The folders of a domain's cache directory that hold its grids (measured:
# `data/fluid_data_####.vdb` and `noise/fluid_noise_####.vdb` are OpenVDB
# files), and `guiding`, named for the guiding velocity it caches.
MANTAFLOW_GRID_FOLDERS = ("data", "noise", "guiding")


def _fluid_cache_owner(path: str) -> str | None:
    """The name of the fluid domain whose cache holds the grid file ``path``,
    or None."""
    import bpy

    target = os.path.realpath(path)
    for ob in bpy.data.objects:
        domain = _fluid_domain(ob)
        if domain is None or not domain.cache_directory:
            continue
        root = os.path.realpath(bpy.path.abspath(domain.cache_directory))
        for folder in MANTAFLOW_GRID_FOLDERS:
            grids = os.path.join(root, folder)
            try:
                inside = os.path.commonpath([target, grids]) == grids
            except ValueError:  # on different drives
                inside = False
            if inside:
                return ob.name
    return None


def fluid_source_refusal(f, frames=None) -> str | None:
    """Why a Fluid Flow field's Domain Object cannot be read, or None.

    A gas domain must be BAKED, as Cache Type All or Modular, over every
    solve frame: a Replay cache reads the live simulation on the pass that
    simulates a frame and the clipped cache on a later one (measured), so
    one scene would send two different fluids. "Baked over a frame" is
    judged by what was baked, not by the cache's settings: scene frame
    ``F`` plays cache frame ``F - Cache Offset``, which must lie in the
    cache's frame range and have its file on disk (:func:`_unbaked_frame`).
    The domain must also be evaluated in the view layer, where its grid is
    read from (:func:`_domain_frame`). ``frames`` is the solve's ``(first
    Blender frame, frame count)``; ``None`` takes it from the scene's add-on
    settings, and skips the range check without them. Every setting is read
    off the ORIGINAL object, whose RNA is what the artist edits.
    """
    src = f.source_object
    if src is None:
        return "a Fluid Flow field with no Domain Object; pick a baked gas domain or a Volume"
    if src.type == "VOLUME":
        return _volume_source_refusal(src)
    mod = _fluid_domain_modifier(src)
    if mod is None:
        return f"its Domain Object '{src.name}' is neither a fluid domain nor a Volume"
    domain = mod.domain_settings
    if domain.domain_type != "GAS":
        return f"its Domain Object '{src.name}' is a liquid domain; use a gas domain"
    if domain.use_adaptive_domain:
        return (f"its Domain Object '{src.name}' uses Adaptive Domain, whose grid moves "
                "every frame; turn it off")
    # The grid is placed by the object's bounds, and a modifier after the
    # Fluid one moves the bounds but not the grid (measured: an Array after
    # it doubled the bounds of an unchanged grid).
    after = _modifiers_after(src, mod)
    if after:
        return (f"its Domain Object '{src.name}' has modifier '{after[0]}' after its Fluid "
                "modifier, which moves the bounds its grid is placed by but not the grid; "
                "move it above the Fluid modifier or disable it")
    # Measured: with Noise on, the density grid is the noise's finer one.
    if f.use_smoke_density and domain.use_noise:
        return (f"its Domain Object '{src.name}' has Noise on, whose density grid is finer "
                "than its velocity grid; turn off Apply Density or the domain's Noise")
    # The grid is read off the domain the view layer evaluates, and one it
    # does not evaluate holds none, baked or not (measured: Disable in
    # Viewports on the domain or on its collection, an excluded collection,
    # a domain outside the view layer, and the Fluid modifier's viewport
    # toggle each read a (0, 0, 0) grid; the eye icon and Disable in Renders
    # read the whole grid). Checked before the bake, since Blender refuses
    # to bake a domain disabled in viewports.
    if not mod.show_viewport:
        return (f"its Domain Object '{src.name}' has its Fluid modifier off in the viewport, "
                "where its fluid is read from; turn on the modifier's Show in Viewport")
    if not _evaluated_in_view_layer(src):
        return (f"its Domain Object '{src.name}' is not evaluated in this view layer, where its "
                "fluid is read from: it or its collection is disabled in viewports or "
                "excluded, or it is not in this scene; enable it in the viewport")
    if domain.cache_type == "REPLAY":
        return (f"its Domain Object '{src.name}' uses a Replay cache, whose grid differs "
                "between a frame it simulates and one it reads back; set Cache Type to All "
                "or Modular and bake it over the solve frames")
    not_baked = (f"its Domain Object '{src.name}' is not baked; bake its data over the solve "
                 "frames")
    if not domain.has_cache_baked_data:
        return not_baked
    if frames is None:
        frames = _solve_frames()
    if frames is None:
        return None
    first, count = int(frames[0]), int(frames[1])
    last = first + count - 1
    start, end = int(domain.cache_frame_start), int(domain.cache_frame_end)
    # Measured: scene frame F plays cache frame F - offset, at either sign;
    # a frame whose cache frame lies outside start..end reads a grid of
    # zeros, an empty one, or the last baked frame's, never its own.
    offset = int(domain.cache_frame_offset)
    remedy = ("bake it over the solve frames" if offset == 0 else
              f"bake it over frames {first - offset}..{last - offset}, which its Cache Offset "
              f"{offset} plays at the solve frames")
    if first - offset < start or last - offset > end:
        plays = ("" if offset == 0 else
                 f", which its Cache Offset {offset} plays at frames {start + offset}.."
                 f"{end + offset},")
        return (f"its Domain Object '{src.name}' is baked over frames {start}..{end}{plays} "
                f"and the solve reads frames {first}..{last}; {remedy}")
    missing = _unbaked_frame(domain, first - offset, last - offset)
    if missing is not None:
        frame, name, every_frame = missing
        if every_frame:
            return not_baked
        at = "" if offset == 0 else f" (cache frame {frame}, under its Cache Offset {offset})"
        return (f"its Domain Object '{src.name}' has no baked data for frame {frame + offset}"
                f"{at}: its cache holds no {name}; {remedy}")
    return None


def _evaluated_in_view_layer(obj) -> bool:
    """Whether the context's view layer evaluates ``obj``: whether its
    depsgraph, the one a fluid is read through, holds an evaluated copy.
    The depsgraph is looked up, never evaluated, because this runs in a
    panel's draw; `encode` evaluates it before asking."""
    import bpy

    return bool(obj.evaluated_get(bpy.context.view_layer.depsgraph).is_evaluated)


# The files a gas domain's baked frame must have, under its cache's `data`
# folder, for Blender to read that frame's velocity and density, per Cache
# Data Format. Measured by removing one file of a frame and reopening: with
# OpenVDB every grid is in fluid_data_000N.vdb; with Uni a frame without
# vel_000N.uni plays the frame before's velocity, and one without
# density_000N.uni plays the frame before's velocity AND density, while the
# heat, flame, shadow and config files change nothing. A format not listed
# is judged by any file of the frame.
_BAKED_FRAME_FILES = {"OPENVDB": ("fluid_data_{:04d}.vdb",),
                      "UNI": ("density_{:04d}.uni", "vel_{:04d}.uni")}


def _unbaked_frame(domain, first: int, last: int):
    """The first CACHE frame of ``first .. last`` whose baked files are not
    all on disk, as ``(frame, what is missing, whether every frame misses
    its files)``, or None when each is there. A frame inside the cache's range
    with a file missing reads as an earlier baked frame's grid once the file
    is reopened (measured), so the files, not the range or the bake flag, say
    what was baked (Blender 5.1 keeps the flag set after a free, measured)."""
    import bpy

    # A relative cache directory ("//...") is relative to the file that holds
    # the domain: the library's, for a domain linked from another .blend.
    library = getattr(domain.id_data, "library", None)
    root = (bpy.path.abspath(domain.cache_directory, library=library)
            if domain.cache_directory else "")
    folder = os.path.join(root, "data")
    try:
        present = set(os.listdir(folder))
    except OSError:
        present = set()
    patterns = _BAKED_FRAME_FILES.get(domain.cache_data_format)

    def lacking(frame):
        if patterns is None:
            tag = f"_{frame:04d}."
            if not any(tag in name for name in present):
                return f"file of that frame in {folder}"
            return None
        for pattern in patterns:
            name = pattern.format(frame)
            if name not in present:
                return f"'{name}' in {folder} (the bake did not reach it, or it was removed)"
        return None

    gaps = [(frame, lacking(frame)) for frame in range(int(first), int(last) + 1)]
    gaps = [(frame, what) for frame, what in gaps if what is not None]
    if not gaps:
        return None
    return gaps[0][0], gaps[0][1], len(gaps) == int(last) - int(first) + 1


def _volume_source_refusal(src) -> str | None:
    import bpy

    try:
        import openvdb  # noqa: F401  (Blender ships it)
    except ImportError:
        return "this Blender has no openvdb module to read the Volume's file with"
    # The add-on reads the file, so anything a modifier does to the volume
    # Blender shows would be lost.
    mods = [m.name for m in src.modifiers if _enabled(m)]
    if mods:
        return (f"its Volume '{src.name}' has modifier '{mods[0]}', which changes the volume "
                "Blender shows but not the file the solver reads; apply or remove it")
    if src.data.filepath:
        owner = _fluid_cache_owner(bpy.path.abspath(src.data.filepath))
        if owner is not None:
            return (f"its Volume '{src.name}' reads the cache of fluid domain '{owner}'; a "
                    "Mantaflow cache holds grid units, so point the field at the domain")
    return None


def _rotation_and_scales(linear):
    """A 3 x 3 linear map as a rotation and a positive scale along each of
    its axes, ``(rotation, scales)``, or None when it is not one (a shear, a
    zero or a mirroring scale)."""
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale <= 0.0):
        return None
    rot = linear / scale
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or np.linalg.det(rot) < 0.0:
        return None
    return rot, scale


class FluidFrame(NamedTuple):
    """One frame of a Fluid Flow's fluid, as the solver takes it.

    ``resolution`` is ``(nx, ny, nz)``, ``(0, 0, 0)`` for an EMPTY frame (no
    fluid: nothing acts anywhere, the damping included). ``table`` and
    ``bricks`` are ``field_eval.fluid_bricks``'s layout (uint32 and float32
    ``(count, 512, 4)``), the velocity along the domain's axes in scene
    units per second. ``corner`` is the corner of cell (0, 0, 0), ``axes``
    the domain's axes as columns, both in solver axes, and ``cell`` the cell
    size along each of those axes.
    """

    resolution: tuple
    table: np.ndarray
    bricks: np.ndarray
    corner: np.ndarray
    axes: np.ndarray
    cell: np.ndarray


class _BrickCutter:
    """Cuts one frame into the solver's bricks without densifying it: either
    a ROW of bricks at a time (``FLUID_BRICK`` cells along y and z, the whole
    x extent; :meth:`rows`, :meth:`add`), every row in table order, or,
    ``sparse``, only the bricks a sparse source can hold anything in
    (:meth:`add_brick`, in table order), the others read as zero. Either way
    the result is ``field_eval.fluid_bricks``'s, value for value."""

    def __init__(self, resolution, what: str, sparse: bool = False, remedy: str = ""):
        b = _field_eval.FLUID_BRICK
        self.sparse = sparse
        self.n = tuple(int(v) for v in resolution)
        self.nb = tuple((k + b - 1) // b for k in self.n)
        self.what = what
        # Appended to the refusal of a value that is not finite: where it can
        # come from and what to do about it, which depends on the source.
        self.remedy = remedy
        self.table = np.full(self.nb[0] * self.nb[1] * self.nb[2], _field_eval.FLUID_EMPTY,
                             dtype=np.uint32)
        self.kept = []
        self.count = 0
        self.done = 0
        self.last = -1

    def rows(self):
        """``(y0, dy, z0, dz)`` of every row, in the order :meth:`add` takes
        them."""
        b = _field_eval.FLUID_BRICK
        for bz in range(self.nb[2]):
            for by in range(self.nb[1]):
                y0, z0 = by * b, bz * b
                yield y0, min(b, self.n[1] - y0), z0, min(b, self.n[2] - z0)

    def brick(self, index: int):
        """Table entry ``index``'s first cell ``(x0, y0, z0)`` and its extent
        ``(dx, dy, dz)``, short at the far faces."""
        b = _field_eval.FLUID_BRICK
        nbx, nby = self.nb[0], self.nb[1]
        at = (index % nbx * b, index // nbx % nby * b, index // (nbx * nby) * b)
        return at, tuple(min(b, k - a) for k, a in zip(self.n, at, strict=True))

    def _check_shape(self, block, velocity, density, shape):
        if np.shape(velocity) != shape + (3,):
            raise ValueError(f"{self.what}: a {block}'s velocity has shape "
                             f"{np.shape(velocity)}, not {shape + (3,)}")
        if density is not None and np.shape(density) != shape:
            raise ValueError(f"{self.what}: a {block}'s density has shape "
                             f"{np.shape(density)}, not {shape}")

    def add(self, velocity, density=None):
        """The next row: ``velocity`` ``(nx, dy, dz, 3)`` and ``density``
        ``(nx, dy, dz)`` or None (zero)."""
        if self.sparse:
            raise ValueError(f"{self.what}: a row given to a sparse cut")
        b = _field_eval.FLUID_BRICK
        nbx, nby = self.nb[0], self.nb[1]
        by, bz = self.done % nby, self.done // nby
        dy, dz = min(b, self.n[1] - by * b), min(b, self.n[2] - bz * b)
        self._check_shape("brick row", velocity, density, (self.n[0], dy, dz))
        row = np.zeros((nbx * b, b, b, 4), dtype=np.float32)
        row[:self.n[0], :dy, :dz, :3] = velocity
        if density is not None:
            row[:self.n[0], :dy, :dz, 3] = density
        if not np.all(np.isfinite(row)):
            raise ValueError(f"{self.what}: the fluid holds a value that is not finite"
                             f"{self.remedy}")
        # (bx, x, y, z, c) -> (bx, z, y, x, c): cells x fastest in a brick.
        cells = row.reshape(nbx, b, b, b, 4).transpose(0, 3, 2, 1, 4).reshape(nbx, b ** 3, 4)
        keep = np.any(cells != 0.0, axis=(1, 2))
        k = int(keep.sum())
        base = (bz * nby + by) * nbx
        slots = self.table[base:base + nbx]
        slots[keep] = np.arange(self.count, self.count + k, dtype=np.uint32)
        if k:
            self.kept.append(np.ascontiguousarray(cells[keep]))
        self.count += k
        self.done += 1

    def add_brick(self, index: int, velocity, density=None):
        """Table entry ``index``'s brick, ``velocity`` and ``density`` over
        :meth:`brick`'s extent; bricks are given in ascending table order, and
        one never given reads as zero."""
        if not self.sparse:
            raise ValueError(f"{self.what}: a single brick given to a cut by rows")
        if not self.last < index < len(self.table):
            raise ValueError(f"{self.what}: brick {index} after brick {self.last}, of "
                             f"{len(self.table)}")
        b = _field_eval.FLUID_BRICK
        _, (dx, dy, dz) = self.brick(index)
        self._check_shape("brick", velocity, density, (dx, dy, dz))
        cell = np.zeros((b, b, b, 4), dtype=np.float32)
        cell[:dx, :dy, :dz, :3] = velocity
        if density is not None:
            cell[:dx, :dy, :dz, 3] = density
        if not np.all(np.isfinite(cell)):
            raise ValueError(f"{self.what}: the fluid holds a value that is not finite"
                             f"{self.remedy}")
        self.last = index
        if np.any(cell != 0.0):
            # (x, y, z, c) -> (z, y, x, c): cells x fastest.
            self.kept.append(cell.transpose(2, 1, 0, 3).reshape(1, b ** 3, 4))
            self.table[index] = self.count
            self.count += 1

    def finish(self):
        """``(table, bricks)``, once every row (or, sparse, every brick to
        give) is in."""
        if not self.sparse and self.done != self.nb[1] * self.nb[2]:
            raise ValueError(f"{self.what}: {self.done} of {self.nb[1] * self.nb[2]} brick "
                             "rows were cut")
        b3 = _field_eval.FLUID_BRICK ** 3
        bricks = (np.concatenate(self.kept) if self.kept
                  else np.zeros((0, b3, 4), dtype=np.float32))
        self.kept = []
        return self.table, bricks


def _destaggered(block, components):
    """Cell-centered velocity from a STAGGERED block (OpenVDB's class
    ``staggered``, a MAC grid): component c of voxel i lives on the face at
    i - 1/2 along c, so the value at the center of voxel i is the mean of
    the faces of voxels i and i + 1 along c. ``block`` holds one more voxel
    along every axis than the result; ``components`` names which of the
    three are staggered."""
    a = np.asarray(block, dtype=np.float64)
    u = a[:-1, :-1, :-1].copy()
    step = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for c in range(3):
        if components[c]:
            i, j, k = step[c]
            upper = a[i:a.shape[0] - 1 + i, j:a.shape[1] - 1 + j, k:a.shape[2] - 1 + k, c]
            u[..., c] = 0.5 * (u[..., c] + upper)
    return u


def fluid_frame(f, name: str, rate: float | None = None) -> FluidFrame:
    """The Fluid Flow field ``f``'s fluid at the current frame, cut into
    bricks: a :class:`FluidFrame`.

    The velocity is in scene units per second of the SOLVER's clock: the
    source's own velocity per scene second times ``rate``
    (:func:`fluid_rate`; ``None`` reads it off the scene), so the cloth rides
    the fluid at the pace its frames are placed. World Scaling is the
    solver's to apply. Raises ``ValueError`` naming the field.
    """
    if rate is None:
        rate = _scene_fluid_rate()
    src = f.source_object
    if src.type == "VOLUME":
        return _volume_frame(src, bool(f.use_smoke_density), name, float(rate))
    return _domain_frame(f, src, name, float(rate))


def _domain_frame(f, src, name: str, rate: float) -> FluidFrame:
    """A baked Mantaflow gas domain at the current frame.

    The grid spans the object's bounding box, ``n_i`` cells of
    ``extent_i / n_i`` along each of its axes (measured: Blender rounds a
    short axis to whole cells and stretches them over the box). A grid unit
    is ``FLUID_GRID_UNIT`` cells of the LONGEST axis per simulation second,
    one uniform unit (measured: Blender's speed). The domain's own Time
    Scale is how many simulation seconds pass in one of Blender's, so the
    velocity is multiplied by it too: the smoke's speed as it plays, where
    Blender's own Fluid Flow field reads the grid in simulation seconds (a
    deliberate difference).
    """
    import bpy

    scene = bpy.context.scene
    where = f"Force field '{name}' at frame {scene.frame_current}"
    dg = bpy.context.evaluated_depsgraph_get()
    # The ORIGINAL's settings; the evaluated copy holds the grids and their
    # resolution (measured: the original's read empty).
    settings = _fluid_domain(src)
    live = _fluid_domain(src.evaluated_get(dg))
    n = tuple(int(v) for v in live.domain_resolution)
    count = n[0] * n[1] * n[2]
    if count == 0 or len(live.velocity_grid) != 3 * count:
        raise ValueError(
            f"{where}: its Domain Object '{src.name}' holds no fluid ({len(live.velocity_grid)} "
            f"velocity values for {n} cells); bake it over the solve frames")
    m = np.array(src.matrix_world, dtype=np.float64)
    box = np.array(src.bound_box, dtype=np.float64)
    lo, hi = box.min(axis=0), box.max(axis=0)
    # The bounds must hold the grid exactly, cell by cell along each axis.
    grid_cell = np.array(live.cell_size, dtype=np.float64)
    if not np.allclose((hi - lo) / np.array(n), grid_cell, rtol=1e-4, atol=0.0):
        raise ValueError(
            f"{where}: its Domain Object '{src.name}' has bounds {(hi - lo).tolist()} that do "
            f"not hold its grid of {list(n)} cells of {grid_cell.tolist()}; something moves "
            "its mesh after the Fluid modifier")
    split = _rotation_and_scales(m[:3, :3])
    if split is None:
        raise ValueError(f"{where}: its Domain Object '{src.name}' is sheared, mirrored or "
                         "of zero scale; it must be rotated and scaled only")
    rot, scale = split
    extent = (hi - lo) * scale
    cell = extent / np.array(n, dtype=np.float64)
    unit = (FLUID_GRID_UNIT * float(extent.max()) / max(n) * float(settings.time_scale)
            * rate)
    vel = np.empty(3 * count, dtype=np.float32)
    live.velocity_grid.foreach_get(vel)
    # x fastest in Blender's flat grid (measured).
    vel = vel.reshape(n[2], n[1], n[0], 3)
    rho = None
    if f.use_smoke_density:
        if len(live.density_grid) != count:
            raise ValueError(
                f"{where}: Apply Density reads its Domain Object '{src.name}''s density, which "
                f"holds {len(live.density_grid)} values for {n} cells (its Noise grid); turn "
                "off the domain's Noise or Apply Density")
        d = np.empty(count, dtype=np.float32)
        live.density_grid.foreach_get(d)
        rho = d.reshape(n[2], n[1], n[0])
    # Blender's bake writes a value that is not finite now and then (measured:
    # 52 velocities in one frame of a 16^3 gas bake, once in about thirty
    # bakes, never in the same bake run again), and the same bake usually
    # comes out whole the next time.
    cutter = _BrickCutter(n, f"{where}: its Domain Object '{src.name}'",
                          remedy="; Blender's bake wrote it: free the bake and bake again")
    for y0, dy, z0, dz in cutter.rows():
        u = vel[z0:z0 + dz, y0:y0 + dy].transpose(2, 1, 0, 3).astype(np.float64) * unit
        r = None if rho is None else rho[z0:z0 + dz, y0:y0 + dy].transpose(2, 1, 0)
        cutter.add(u, r)
    table, bricks = cutter.finish()
    corner = (m @ np.append(lo, 1.0))[:3]
    return FluidFrame(n, table, bricks, _to_solver(corner), _to_solver(rot.T).T, cell)


def _no_file_reason(data, live) -> str:
    """Why a Volume resolved no file at the current frame, as far as Blender
    says."""
    import bpy

    reasons = []
    if live.grids.error_message:
        reasons.append(live.grids.error_message)
    elif data.is_sequence and data.sequence_mode == "CLIP":
        # Measured: a Clip sequence resolves no file outside its frames.
        reasons.append(
            f"the frame is outside its sequence (Sequence Mode Clip, Start Frame "
            f"{data.frame_start}, Frames {data.frame_duration}, Offset {data.frame_offset}); "
            "shorten the solve or set Sequence Mode to Extend")
    if not data.filepath:
        reasons.append("it names no file")
    elif data.filepath.startswith("//") and not bpy.data.filepath:
        reasons.append("its path is relative to the .blend file, which is not saved")
    return "; ".join(reasons) or "Blender resolved no file for it"


def volume_velocity_names(data) -> list[str]:
    """The grids a Volume datablock's Velocity settings name, the ones
    Transfer reads: Velocity X, Y and Z when all three are set, else
    Velocity Grid alone (``[""]`` when it names none)."""
    split = (data.velocity_x_grid, data.velocity_y_grid, data.velocity_z_grid)
    return list(split) if all(split) else [data.velocity_grid]


def vdb_grid_problem(grids: dict, names: list[str], use_density: bool, volume: str,
                     path: str) -> str | None:
    """Why a Volume's file cannot serve as a Fluid Flow's fluid, as far as its
    grids' metadata shows, or None.

    ``grids`` maps every grid the file holds to ``(value type name, whether
    Mantaflow wrote it)``, ``names`` is :func:`volume_velocity_names` of the
    Volume named ``volume``, and ``path`` is its file. Transfer
    (:func:`_volume_frame`) and the status listing
    (``ui/dynamics/force_field_status.py``, from the file's header) both ask
    this one function, so the reason the panel and MCP list is the one
    Transfer refuses with, word for word.
    """
    at = f"its Volume '{volume}''s file {path}"
    held = ", ".join(sorted(grids)) or "none"
    if names == [""]:
        return (f"its Volume '{volume}' names no velocity grid ({path} holds {held}); "
                "set Velocity Grid in its Volume properties")
    kinds = VDB_SCALAR_TYPES if len(names) == 3 else VDB_VECTOR_TYPES
    for gname in names:
        grid = grids.get(gname)
        if grid is None:
            return (f"{at} holds no grid named '{gname}' (it holds {held}); set its "
                    "Velocity Grid")
        value_type, mantaflow = grid
        if value_type not in kinds:
            return (f"the velocity grid '{gname}' in {at} is {value_type}, "
                    f"not {' or '.join(kinds)}")
        if mantaflow:
            return (f"{at} is a Mantaflow cache file, whose velocity is in grid units; "
                    "point the field at its domain")
    if use_density:
        grid = grids.get(VDB_DENSITY_NAME)
        if grid is None:
            return f"Apply Density is on and {at} holds no density grid (it holds {held})"
        if grid[0] not in VDB_SCALAR_TYPES:
            return (f"the density grid in {at} is {grid[0]}, "
                    f"not {' or '.join(VDB_SCALAR_TYPES)}")
    return None


def _same_transform(a, b) -> bool:
    """Whether two openvdb transforms map every index to the same place, to
    a millionth of a voxel."""
    steps = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    pa = np.array([a.indexToWorld(s) for s in steps], dtype=np.float64)
    pb = np.array([b.indexToWorld(s) for s in steps], dtype=np.float64)
    voxel = float(np.linalg.norm(pa[1:] - pa[0], axis=1).min())
    return float(np.abs(pa - pb).max()) <= 1e-6 * voxel


def _grid_array(grid, shape):
    """A zeroed array of ``shape`` in ``grid``'s own precision, for
    ``copyToArray``."""
    double = grid.valueTypeName in ("vec3d", "double")
    return np.zeros(shape, dtype=np.float64 if double else np.float32)


def _velocity_block(grids, staggered, origin, shape):
    """The velocity of voxels ``origin`` .. ``origin + shape - 1``, cell
    centered, ``shape + (3,)`` in the file's units: one vector grid or three
    component grids, each read one voxel further along an axis it is
    staggered on."""
    nx, ny, nz = shape
    grow = 1 if any(staggered) else 0
    big = (nx + grow, ny + grow, nz + grow)
    if len(grids) == 1:
        a = _grid_array(grids[0], big + (3,))
        grids[0].copyToArray(a, ijk=origin)
    else:
        a = np.zeros(big + (3,), dtype=np.float64)
        for c, g in enumerate(grids):
            part = _grid_array(g, big)
            g.copyToArray(part, ijk=origin)
            a[..., c] = part
    if not grow:
        return a.astype(np.float64)
    return _destaggered(a, staggered)


# A Volume is read brick by brick, only where its active values are, or row
# by row over its whole active bounding box, whichever costs less. The two
# read the same values. The costs, measured with Blender's openvdb: visiting
# one active value from Python (`_touched_bricks`), reading one brick, and
# reading one voxel of a row. The values are visited only when that costs at
# most a quarter of the row read, which bounds what an unused visit wastes;
# a stray voxel far from a small plume then costs a few bricks instead of
# its whole bounding box.
SPARSE_VISIT_SECONDS = 0.7e-6
SPARSE_BRICK_SECONDS = 24e-6
ROW_VOXEL_SECONDS = 18e-9


def _touched_bricks(grids, offsets, lo, n) -> list | None:
    """The table entries of the bricks of the box ``lo .. lo + n - 1`` the
    active values of ``grids`` reach, ascending: a value's own voxels and,
    through ``offsets`` (per grid, the voxel offsets at which a value is
    also read, as de-staggering reads a face from the voxel below it), the
    voxels that read it. None when a grid's background is not zero, since
    every voxel then reads something."""
    b = _field_eval.FLUID_BRICK
    nbx, nby = ((k + b - 1) // b for k in n[:2])
    lx, ly, lz = (int(v) for v in lo)
    nx, ny, nz = (int(v) for v in n)
    found = set()
    for grid, shifts in zip(grids, offsets, strict=True):
        if np.any(np.asarray(grid.background, dtype=np.float64) != 0.0):
            return None
        for item in grid.citerOnValues():
            (x0, y0, z0), (x1, y1, z1) = item.min, item.max
            for sx, sy, sz in shifts:
                ax, ay, az = max(x0 + sx - lx, 0), max(y0 + sy - ly, 0), max(z0 + sz - lz, 0)
                cx = min(x1 + sx - lx, nx - 1)
                cy = min(y1 + sy - ly, ny - 1)
                cz = min(z1 + sz - lz, nz - 1)
                if ax > cx or ay > cy or az > cz:
                    continue
                for bz in range(az // b, cz // b + 1):
                    for by in range(ay // b, cy // b + 1):
                        for bx in range(ax // b, cx // b + 1):
                            found.add(bx + nbx * (by + nby * bz))
    return sorted(found)


def _volume_frame(src, use_density: bool, name: str, rate: float) -> FluidFrame:
    """A Volume object's file at the current frame, read with Blender's
    openvdb module (``bpy`` exposes a grid's metadata, never its voxels).

    The velocity grids are the Volume's own settings: Velocity Grid (one
    vector grid), or Velocity X, Y and Z (three float grids) when all three
    are set; Velocity Unit Frame means a displacement per scene frame, and
    the result is multiplied by Velocity Scale. A ``staggered`` grid is
    de-staggered to cell centers. Settings are read off the ORIGINAL
    datablock (measured: the evaluated copy keeps a stale Velocity Unit and
    Scale after a scripted edit), with its grids loaded, which is when
    Blender names the velocity grid it finds; the file is the evaluated
    copy's, the one the current frame resolves to. The active voxels are
    read one row of bricks at a time.
    """
    import bpy
    import openvdb

    scene = bpy.context.scene
    where = f"Force field '{name}' at frame {scene.frame_current}"
    data = src.data
    dg = bpy.context.evaluated_depsgraph_get()
    live = src.evaluated_get(dg).data
    path = live.grids.frame_filepath
    if not path:
        raise ValueError(f"{where}: its Volume '{src.name}' resolves no file: "
                         f"{_no_file_reason(data, live)}")
    path = bpy.path.abspath(path)
    # The file named with the object it belongs to, in every message below.
    at = f"its Volume '{src.name}''s file {path}"
    if not data.grids.is_loaded:
        data.grids.load()
    try:
        metas = {g.name: g for g in openvdb.readAllGridMetadata(path)}
    except Exception as e:  # openvdb raises its own types
        raise ValueError(f"{where}: cannot read {at}: {e}") from e
    names = volume_velocity_names(data)
    why = vdb_grid_problem(
        {g: (m.valueTypeName, MANTAFLOW_CACHE_METADATA in m.metadata) for g, m in metas.items()},
        names, use_density, src.name, path)
    if why is not None:
        raise ValueError(f"{where}: {why}")
    try:
        grids = [openvdb.read(path, g) for g in names]
        density = openvdb.read(path, VDB_DENSITY_NAME) if use_density else None
    except Exception as e:
        raise ValueError(f"{where}: cannot read {at}: {e}") from e
    reference = grids[0].transform
    for g in grids[1:] + ([density] if density is not None else []):
        if not _same_transform(reference, g.transform):
            raise ValueError(
                f"{where}: in {at} the grid '{g.name}' is not on the voxels of '{names[0]}' "
                f"(voxel {g.transform.voxelSize()} at {g.transform.indexToWorld((0, 0, 0))}, "
                f"against {reference.voxelSize()} at {reference.indexToWorld((0, 0, 0))}); "
                "export them on one transform")
    if data.velocity_unit == "FRAME":
        per_second = scene.render.fps / scene.render.fps_base
    elif data.velocity_unit == "SECOND":
        per_second = 1.0
    else:
        raise ValueError(f"{where}: Velocity Unit {data.velocity_unit!r} is not supported")
    unit = float(per_second) * float(data.velocity_scale) * rate
    # The index-to-world map is affine; its linear part from the images of
    # the unit steps.
    origin = np.array(reference.indexToWorld((0, 0, 0)), dtype=np.float64)
    vdb_linear = np.stack([np.array(reference.indexToWorld(tuple(e)), dtype=np.float64)
                           - origin for e in np.eye(3, dtype=int)], axis=1)
    m = np.array(src.matrix_world, dtype=np.float64)
    split_map = _rotation_and_scales(m[:3, :3] @ vdb_linear)
    if split_map is None:
        raise ValueError(f"{where}: its Volume '{src.name}''s voxels are sheared, mirrored or "
                         "of zero size in the world; give it a rotation and a positive scale")
    rot, cell = split_map
    boxes = [g.evalActiveVoxelBoundingBox() for g in grids]
    lo = np.min([np.array(b[0], dtype=np.int64) for b in boxes], axis=0)
    hi = np.max([np.array(b[1], dtype=np.int64) for b in boxes], axis=0)
    empty = bool(np.any(hi < lo))
    if empty:
        # No active voxel: no fluid at this frame. The frame keeps the
        # file's own placement, which the solver verifies like any other.
        lo = np.zeros(3, dtype=np.int64)
    corner = (m @ np.append(origin + vdb_linear @ (lo.astype(np.float64) - 0.5), 1.0))[:3]
    axes = _to_solver(rot.T).T
    if empty:
        b3 = _field_eval.FLUID_BRICK ** 3
        return FluidFrame((0, 0, 0), np.zeros(0, dtype=np.uint32),
                          np.zeros((0, b3, 4), dtype=np.float32), _to_solver(corner), axes, cell)
    n = tuple(int(v) for v in hi - lo + 1)
    active = sum(int(g.activeVoxelCount()) for g in grids)
    if max(n) > FLUID_MAX_RESOLUTION:
        raise ValueError(
            f"{where}: in {at} the active voxels span {list(n)} voxels ({active} active), past "
            f"the {FLUID_MAX_RESOLUTION} a fluid may have along an axis; prune or clip the "
            "file's stray voxels")
    b = _field_eval.FLUID_BRICK
    table_bytes = 4 * math.prod((k + b - 1) // b for k in n)
    if table_bytes > TRANSPORT_CAP_BYTES:
        raise ValueError(
            f"{where}: in {at} the active voxels span {list(n)} voxels ({active} active), whose "
            f"brick table alone is {_size(table_bytes)}, past the {_size(TRANSPORT_CAP_BYTES)} "
            "a transfer carries; prune or clip the file's stray voxels")
    # One vector grid is staggered in all three components or none.
    classes = [g.gridClass == "staggered" for g in grids]
    staggered = classes * 3 if len(grids) == 1 else classes
    # The file's velocity in its world axes, then the object's transform, then
    # the domain's own axes, in scene units per solver second.
    to_domain = (m[:3, :3].T @ rot) * unit

    def read(at, shape):
        u = _velocity_block(grids, staggered, at, shape)
        r = None
        if density is not None:
            r = _grid_array(density, shape)
            density.copyToArray(r, ijk=at)
        return u @ to_domain, r

    # Where each grid's values are read: a staggered component also from the
    # voxel below along its axis.
    steps = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    if len(grids) == 1:
        offsets = [((0, 0, 0),) + (steps if classes[0] else ())]
    else:
        offsets = [((0, 0, 0),) + ((steps[c],) if staggered[c] else ()) for c in range(3)]
    sources = grids + ([density] if density is not None else [])
    offsets += [((0, 0, 0),)] * (len(sources) - len(offsets))
    touched = None
    rows_cost = ROW_VOXEL_SECONDS * math.prod(n)
    visits = sum(int(g.activeVoxelCount()) for g in sources)
    if SPARSE_VISIT_SECONDS * visits <= 0.25 * rows_cost:
        touched = _touched_bricks(sources, offsets, lo, n)
        if touched is not None and SPARSE_BRICK_SECONDS * len(touched) > rows_cost:
            touched = None
    cutter = _BrickCutter(n, f"{where}: {at}", sparse=touched is not None)
    if touched is not None:
        for index in touched:
            first, shape = cutter.brick(index)
            cutter.add_brick(index, *read(tuple(int(a + f) for a, f in zip(lo, first, strict=True)), shape))
    else:
        for y0, dy, z0, dz in cutter.rows():
            cutter.add(*read((int(lo[0]), int(lo[1]) + y0, int(lo[2]) + z0), (n[0], dy, dz)))
    table, bricks = cutter.finish()
    return FluidFrame(n, table, bricks, _to_solver(corner), axes, cell)


def _size(nbytes: int) -> str:
    """A byte count for a message, in the largest unit it fills: a part of a
    few kilobytes beside one of a gigabyte reads as kilobytes, never as
    0.0 MiB."""
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024 ** 3:.2f} GiB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / 1024 ** 2:.1f} MiB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KiB"
    return f"{nbytes:,} bytes"


def _past_cap(nbytes: int) -> str:
    """``nbytes`` against the transfer cap, for a refusal: the two sizes as
    :func:`_size` gives them, or in bytes where those would read alike."""
    total, cap = _size(nbytes), _size(TRANSPORT_CAP_BYTES)
    if total == cap:
        total, cap = f"{nbytes:,} bytes", f"{TRANSPORT_CAP_BYTES:,} bytes"
    return f"{total}, past the {cap} a transfer carries"


def _parts_line(parts) -> str:
    """``(label, bytes)`` pairs, largest first, for a refusal."""
    return "; ".join(f"{label}: {_size(n)}" for label, n in sorted(parts, key=lambda p: -p[1]))


def _chunks(array) -> list:
    """``array``'s bytes as a list of ``bytes`` chunks of at most
    ``PAYLOAD_CHUNK_BYTES``, empty for an empty array."""
    raw = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
    return [raw[i:i + PAYLOAD_CHUNK_BYTES].tobytes()
            for i in range(0, raw.size, PAYLOAD_CHUNK_BYTES)]


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


# Under Use Alpha a float image's color is divided by its alpha only where
# the alpha is above this float32 value (measured: alpha 1e-4 reads the color
# as stored, the next float32 above it reads it divided, and so does every
# larger alpha tried; a byte image's smallest alpha, 1 / 255, is far above).
TEXTURE_ALPHA_MIN = np.float32(1e-4)


def texture_image(f, name: str) -> np.ndarray:
    """A Texture field's image as (H, W, 3) LINEAR RGB, row 0 at the bottom:
    at every pixel, the color Blender's own texture reads at that pixel's
    center.

    What Blender reads, measured black box through ``Texture.evaluate`` at
    pixel centers on byte and float images, made in memory and loaded from
    PNG and OpenEXR files, in every Alpha mode, and on particles in a
    Texture field (``image.pixels`` gives ``c`` and the alpha ``a``):

    * A BYTE image is read PREMULTIPLIED, ``c * a``, and an sRGB one then
      through the sRGB curve (R 0.6 at alpha 0.5 reads 0.301 in Non-Color,
      0.0738 in sRGB). Under Use Alpha it is read straight instead, ``c``,
      wherever the alpha is above 0, and 0 where it is 0.
    * A FLOAT image is read as stored; under Use Alpha its color is divided
      by its alpha wherever the alpha is above ``TEXTURE_ALPHA_MIN`` (R 0.6
      at alpha 0.5 reads 1.2), and read as stored elsewhere, a negative
      alpha included.
    * Use Alpha acts only while the image's Alpha mode is not None.
    * Clamp sets a negative channel to 0 and keeps one above 1.

    Between pixels, with Interpolation, Blender filters the premultiplied
    colors and divides by the filtered alpha, and applies a byte image's
    sRGB curve after filtering, while this solver interpolates the colors
    sent here bilinearly, a deliberate difference in sampling.

    Raises ``ValueError`` naming the field for an image with no pixels, and
    for a reading float32 cannot hold.
    """
    tex = f.texture
    image = tex.image
    w, h = image.size
    if w * h == 0:
        raise ValueError(f"Force field '{name}': its image '{image.name}' has no pixels")
    pixels = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(h, w, 4).astype(np.float64)
    rgb, alpha = pixels[..., :3], pixels[..., 3:]
    straight = bool(tex.use_alpha) and image.alpha_mode != "NONE"
    if image.is_float:
        if straight:
            divide = alpha > float(TEXTURE_ALPHA_MIN)
            rgb = np.where(divide, rgb / np.where(divide, alpha, 1.0), rgb)
    else:
        rgb = np.where(alpha > 0.0, rgb, 0.0) if straight else rgb * alpha
        if image.colorspace_settings.name == "sRGB":
            rgb = _srgb_to_linear(rgb)
    if tex.use_clamp:
        rgb = np.maximum(rgb, 0.0)
    if not np.all(np.isfinite(rgb.astype(np.float32))):
        raise ValueError(
            f"Force field '{name}': its image '{image.name}' reads a color float32 cannot hold "
            "(a value that is not finite, or under Use Alpha a float image's color divided "
            "by its alpha); make every color finite, or turn off the texture's Use Alpha")
    return rgb


# --- one field at one frame ------------------------------------------------

# The settings a track sample holds besides the frame, in `frontend`'s names,
# and the Blender property each is read from.
_SCALARS = (("strength", "strength"), ("power", "falloff_power"),
            ("min_distance", "distance_min"), ("max_distance", "distance_max"),
            ("radial_power", "radial_falloff"), ("radial_min", "radial_min"),
            ("radial_max", "radial_max"), ("size", "size"), ("noise", "noise"),
            ("linear_drag", "linear_drag"), ("quadratic_drag", "quadratic_drag"),
            ("harmonic_damping", "harmonic_damping"), ("rest_length", "rest_length"),
            ("inflow", "inflow"), ("nabla", "texture_nabla"))


def _to_solver(v):
    """Blender (x, y, z) vectors to the solver's (x, z, -y)."""
    v = np.asarray(v)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def _from_solver(v):
    """The solver's (x, y, z) vectors back to Blender's (x, -z, y)."""
    v = np.asarray(v)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def _discrete(obj) -> dict:
    """What cannot change during a solve, in `frontend`'s names."""
    f = obj.field
    return {
        "type": SOLVER_TYPE[f.type],
        "shape": f.shape.lower(),
        "falloff": f.falloff_type.lower(),
        "z_direction": {"BOTH": "both", "POSITIVE": "positive",
                        "NEGATIVE": "negative"}[f.z_direction],
        "use_min": bool(f.use_min_distance),
        "use_max": bool(f.use_max_distance),
        "use_radial_min": bool(f.use_radial_min),
        "use_radial_max": bool(f.use_radial_max),
        # Blender applies no Gravitation to a Fluid Flow (measured on
        # particles and cloth: on and off moved them identically, while it
        # weighs a Force by 1 / d^2), and the Fluid Flow panel does not show
        # it, so a flag left on from another type is sent off, not refused.
        "gravitation": bool(f.use_gravity_falloff) and f.type != "FLUID_FLOW",
        "global_coordinates": bool(f.use_global_coords),
        "seed": int(f.seed),
    }


def _sample(obj) -> dict:
    """One track sample at the current frame, in solver axes: the origin, the
    unscaled axes as columns, the scale along each (the noise scale) and the
    animatable values."""
    m = np.array(obj.matrix_world, dtype=np.float64)
    rot = m[:3, :3]
    # Positive along every axis: refusal() turns away a zero or mirrored one.
    scale = np.linalg.norm(rot, axis=0)
    axes_blender = rot / scale
    # Re-orthonormalized: a sheared world matrix (a rotated parent with a
    # non-uniform scale) is not a frame, and the solver refuses one that is
    # not a rotation.
    u, _, vt = np.linalg.svd(axes_blender)
    axes_blender = u @ vt
    sample = {
        "position": _to_solver(m[:3, 3]),
        "axes": _to_solver(axes_blender.T).T,
        "noise_scale": scale,
    }
    f = obj.field
    for name, prop in _SCALARS:
        sample[name] = float(getattr(f, prop))
    # A Wind acts through the drag, which already weighs a surface by how it
    # faces the flow, so its own Wind Factor (1 on a new one) is not read.
    sample["wind_factor"] = 0.0 if f.type == "WIND" else float(f.wind_factor)
    return sample


def _could_move(obj) -> bool:
    """Whether anything can change ``obj``'s transform or field settings
    during the solve: its own animation data (keys, drivers, NLA, including
    on its field's settings), a parent or a constraint. A field with none is
    sent as one sample."""
    return bool(obj.animation_data or obj.parent or len(obj.constraints))


def _source_could_move(src) -> bool:
    """Whether anything but its files can change what a Fluid Flow reads
    from ``src`` between solve frames: :func:`_could_move`, and for a Volume
    its datablock's own animation (a keyed Velocity Scale, say)."""
    if _could_move(src):
        return True
    return src.type == "VOLUME" and bool(getattr(src.data, "animation_data", None))


# --- what a field is drawn and sent from, as a key ----------------------------


def _hashable(value):
    """An RNA value as a hashable one: arrays as nested tuples, an enum
    flag's set sorted, so equal settings give equal values."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(str(v) for v in value))
    try:
        return tuple(_hashable(v) for v in value)
    except TypeError:
        return repr(value)


def _rna_state(struct) -> tuple:
    """Every plain setting of an RNA struct, as ``(identifier, value)``
    pairs: pointers and collections left out, and so are a fluid domain's
    read-only grids (``velocity_grid`` ...), which are data, not settings,
    and a datablock's own bookkeeping (its name, user count, tags), which
    every datablock has and nothing here reads."""
    import bpy

    shared = (frozenset(p.identifier for p in bpy.types.ID.bl_rna.properties)
              if isinstance(struct, bpy.types.ID) else frozenset())
    out = []
    for p in struct.bl_rna.properties:
        ident = p.identifier
        if ident == "rna_type" or p.type in ("POINTER", "COLLECTION") or ident in shared:
            continue
        if p.is_readonly and p.type == "FLOAT" and ident.endswith("_grid"):
            continue
        out.append((ident, _hashable(getattr(struct, ident))))
    return tuple(out)


def _stat(path: str):
    """``(size, mtime_ns)`` of ``path``, or None when it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def _files_state(root: str, frame: int | None = None, recursive: bool = True) -> tuple:
    """``(relative path, size, mtime_ns)`` of the files under ``root``, sorted;
    given ``frame``, only the files whose name carries that frame's
    four-digit number, as Mantaflow names its cache files. Empty when
    ``root`` cannot be listed."""
    tag = None if frame is None else f"_{int(frame):04d}."
    out = []
    for dirpath, dirs, names in os.walk(root):
        if not recursive:
            dirs.clear()
        for nm in names:
            if tag is not None and tag not in nm:
                continue
            p = os.path.join(dirpath, nm)
            st = _stat(p)
            if st is not None:
                out.append((os.path.relpath(p, root),) + st)
    return tuple(sorted(out))


def _source_state(src, frame: int | None = None) -> tuple:
    """What a Fluid Flow reads from its Domain Object besides the frame, as
    a hashable value: the object's transform and bounds, its modifiers, every
    setting of the domain or the Volume, and the size and modification time
    of the files the fluid is read from (every file of the cache or the
    Volume's folder; given the scene ``frame``, a cache's files of the cache
    frame it plays, ``frame - Cache Offset``)."""
    import bpy

    if src is None:
        return (None,)
    head = (src.name, src.type, tuple(tuple(r) for r in src.matrix_world),
            tuple((m.name, m.type, bool(m.show_viewport), bool(m.show_render))
                  for m in src.modifiers))
    if src.type == "VOLUME":
        data = src.data
        path = bpy.path.abspath(data.filepath) if data.filepath else ""
        folder = os.path.dirname(path) if path else ""
        return head + (_rna_state(data), _files_state(folder, recursive=False) if folder else ())
    domain = _fluid_domain(src)
    if domain is None:
        return head
    root = bpy.path.abspath(domain.cache_directory) if domain.cache_directory else ""
    cache_frame = None if frame is None else int(frame) - int(domain.cache_frame_offset)
    return head + (tuple(tuple(c) for c in src.bound_box), _rna_state(domain),
                   _files_state(root, cache_frame) if root else ())


def _pixel_digest(image) -> int:
    """A checksum of ``image``'s pixels as ``image.pixels`` gives them."""
    w, h = image.size
    pixels = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return zlib.crc32(memoryview(pixels).cast("B"))


def _image_state(image) -> tuple:
    """Everything :func:`texture_image` reads off ``image``, as a hashable
    value: its identity, size, colorspace and Alpha mode, and its CONTENT.
    An image with unsaved edits is described by a checksum of its pixels,
    because a second paint stroke or a scripted edit leaves every flag as
    the first one left it (measured: ``is_dirty`` stays True); a generated
    image by its Generated settings, which change its pixels and leave it
    clean (measured: a new Color or Type); a packed one by a checksum of
    the packed file; and any other by its file's path, size and time. The
    checksum is taken only while edits are unsaved, so a clean image costs
    no read of its pixels."""
    import bpy

    head = (image.name, tuple(image.size), image.source, image.colorspace_settings.name,
            bool(image.is_float), image.alpha_mode)
    if image.is_dirty:
        return head + ("edited", _pixel_digest(image))
    if image.packed_file is not None:
        return head + ("packed", image.packed_file.size, zlib.crc32(image.packed_file.data))
    if image.source == "GENERATED":
        return head + ("generated", image.generated_type, tuple(image.generated_color),
                       int(image.generated_width), int(image.generated_height),
                       bool(image.use_generated_float))
    path = bpy.path.abspath(image.filepath) if image.filepath else ""
    return head + ("file", path, _stat(path) if path else None)


def _texture_state(f) -> tuple:
    """Everything a Texture field's image reading depends on, as a hashable
    value: the texture's settings and the image's (:func:`_image_state`)."""
    tex = f.texture
    if tex is None:
        return (None,)
    image = getattr(tex, "image", None)
    picture = None if image is None else _image_state(image)
    return (f.texture_mode, bool(f.use_object_coords), bool(f.use_root_coords), tex.name,
            tex.type, _rna_state(tex), picture)


def field_key(obj) -> tuple:
    """Every setting and the transform ``obj``'s field is drawn from, as a
    hashable value, so a cache keyed on it rebuilds exactly when one
    changes: a Texture's texture and image (:func:`_image_state`, its
    content included), and a Fluid Flow's frame, Apply Density and Domain
    Object (:func:`_source_state`, a cache's files of the cache frame the
    current frame plays)."""
    import bpy

    f = obj.field
    # A fluid changes with the frame whatever its field does.
    frame = bpy.context.scene.frame_current if f.type == "FLUID_FLOW" else None
    key = (obj.name, frame, tuple(tuple(r) for r in obj.matrix_world),
           tuple(sorted(_discrete(obj).items())) if f.type in SOLVER_TYPE else f.type,
           tuple(float(getattr(f, prop)) for _, prop in _SCALARS), float(f.wind_factor),
           bool(f.use_smoke_density))
    if f.type == "TEXTURE":
        key += (_texture_state(f),)
    if f.type == "FLUID_FLOW":
        key += (_source_state(f.source_object, frame),)
    return key


# The last record drawn per field object, keyed on what it was built from, so
# a redraw that changes only the view (a zoom) does not re-read a fluid or an
# image.
_RECORDS: dict = {}


def record_at_frame(obj, rate: float | None = None) -> dict:
    """``obj``'s field as a one-sample record at the current frame, what
    ``field_eval.evaluate`` takes; for drawing. ``rate`` is
    :func:`fluid_rate` (``None`` reads it off the scene), which a fluid's
    velocity is sent multiplied by. The record is reused while
    :func:`field_key` and the rate stay the same; it is shared, so a caller
    must not change it."""
    if rate is None:
        rate = _scene_fluid_rate()
    key = field_key(obj) + (float(rate),)
    known = _RECORDS.get(obj.name)
    if known is not None and known[0] == key:
        return known[1]
    rec = _discrete(obj)
    for name, value in _sample(obj).items():
        rec[name] = np.asarray(value, dtype=np.float64)[None, ...]
    rec["start"], rec["step"] = 0.0, 1.0
    if obj.field.type == "TEXTURE":
        tex = obj.field.texture
        rec["image"] = texture_image(obj.field, obj.name)
        rec["texture_mode"] = obj.field.texture_mode.lower()
        rec["extension"] = tex.extension.lower()
        rec["interpolation"] = bool(tex.use_interpolation)
        rec["object_coordinates"] = bool(obj.field.use_object_coords)
        rec["z_up"] = True
    if obj.field.type == "FLUID_FLOW":
        fr = fluid_frame(obj.field, obj.name, rate)
        rec["fluid"] = {"resolution": [list(fr.resolution)], "cell": fr.cell[None],
                        "corner": fr.corner[None], "axes": fr.axes[None], "start": 0.0,
                        "step": 1.0, "tables": [fr.table], "bricks": [fr.bricks]}
        rec["use_density"] = bool(obj.field.use_smoke_density)
        # Keyed after the read, which can rename a Volume's Velocity Grid
        # (see `encode`), so the next draw at this frame finds the record.
        key = field_key(obj) + (float(rate),)
    _RECORDS[obj.name] = (key, rec)
    return rec


def evaluate(objs, points, time: float = 0.0, rate: float | None = None,
             errors: list | None = None):
    """Sum the fields of ``objs`` at the current frame at Blender-world
    ``points`` (N x 3), through the solver's own reference laws.

    Returns ``(acceleration, air_velocity)``, each N x 3 in Blender axes.
    Magnetic acts on velocity, so it draws nothing here, at rest, and a point
    in empty space presents no surface for a Wind Factor to weigh by, so the
    push is drawn whole. A field whose source cannot be read raises
    ``ValueError`` naming it; given an ``errors`` list, that field is left
    out and its message appended there instead, so the others still draw.
    """
    if rate is None:
        rate = _scene_fluid_rate()
    names = {o.name for o in objs}
    for stale in [k for k in _RECORDS if k not in names]:
        del _RECORDS[stale]
    p = _to_solver(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    accel = np.zeros_like(p)
    air = np.zeros_like(p)
    for obj in objs:
        try:
            rec = record_at_frame(obj, rate)
        except ValueError as e:
            if errors is None:
                raise
            errors.append(str(e))
            continue
        a, w = _field_eval.evaluate(rec, p, np.zeros_like(p), time, normals=np.zeros_like(p))
        accel += a
        air += w
    return _from_solver(accel), _from_solver(air)


def preflight(obj) -> str | None:
    """Why Transfer would refuse ``obj``'s field as far as the CURRENT frame
    shows, or None: :func:`refusal`, then, for a field that reads a source (a
    Texture's image, a Fluid Flow's fluid), that reading itself, whose error
    names the field. The reading is :func:`record_at_frame`'s, reused while
    nothing it depends on changes; a first call reads the source, so this is
    for a query (MCP, the Python API), never for a panel's draw."""
    why = refusal(obj)
    if why is not None or obj.field.type not in ("TEXTURE", "FLUID_FLOW"):
        return why
    try:
        record_at_frame(obj)
    except ValueError as e:
        return str(e)
    return None


# --- the preview box --------------------------------------------------------


def objects_box(objs) -> tuple[np.ndarray, np.ndarray]:
    """The world bounds of ``objs`` at the current frame, grown by a tenth of
    their largest extent on every side, so arrows also show where the
    objects are about to go."""
    pts = []
    for o in objs:
        pts.extend(o.matrix_world @ _vec(c) for c in o.bound_box)
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    margin = 0.1 * max(float((hi - lo).max()), 1e-3)
    return lo - margin, hi + margin


def _vec(c):
    from mathutils import Vector  # pyright: ignore

    return Vector(c)


def grid_points(lo, hi, resolution) -> np.ndarray:
    """(D, H, W, 3) Blender-world corner points of a W x H x D grid."""
    w, h, d = (int(v) for v in resolution)
    xs = np.linspace(lo[0], hi[0], w)
    ys = np.linspace(lo[1], hi[1], h)
    zs = np.linspace(lo[2], hi[2], d)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


# --- transfer ---------------------------------------------------------------


def dynamic_objects(scene) -> list:
    """Every object of an active non-Static group that the solve includes."""
    from ..models.groups import iterate_active_object_groups
    from .uuid_registry import resolve_assigned

    out = []
    for group in iterate_active_object_groups(scene):
        if str(group.object_type) == "STATIC":
            continue
        for assigned in group.assigned_objects:
            if assigned.included:
                obj = resolve_assigned(assigned)
                if obj is not None:
                    out.append(obj)
    return out


def _group_objects(scene, uuids) -> list:
    from ..models.groups import get_group_by_uuid
    from .uuid_registry import resolve_assigned

    out = []
    for uid in uuids:
        group = get_group_by_uuid(scene, uid)
        if group is None:
            continue
        for assigned in group.assigned_objects:
            if assigned.included:
                obj = resolve_assigned(assigned)
                if obj is not None:
                    out.append(obj)
    return out


def pushed_objects(scene, state, obj) -> list:
    """The simulated objects ``obj``'s field reaches: its target groups'
    objects, or every simulated object."""
    from ..models.force_field_targets import target_group_uuids

    key = target_group_uuids(state, obj)
    return dynamic_objects(scene) if key is None else _group_objects(scene, sorted(key))


# --- a Fluid Flow's volume in the payload -------------------------------------


class _VolumePayload:
    """One Fluid Flow's frames as the payload carries them: per frame its
    resolution, table and bricks (lists of ``bytes`` chunks), corner, axes
    and cell, with the byte and brick counts the size report gives."""

    def __init__(self, obj):
        self.field = obj.name
        self.source = obj.field.source_object.name
        self.resolution, self.tables, self.bricks = [], [], []
        self.corner, self.axes, self.cell = [], [], []
        self.nbytes = 0
        self.kept = 0
        self.slots = 0

    def add(self, fr: FluidFrame) -> None:
        table = fr.table.astype("<u4", copy=False)
        bricks = fr.bricks.astype("<f4", copy=False)
        self.resolution.append([int(v) for v in fr.resolution])
        self.tables.append(_chunks(table))
        self.bricks.append(_chunks(bricks))
        self.corner.append([float(v) for v in fr.corner])
        self.axes.append(np.asarray(fr.axes, dtype=np.float64).tolist())
        self.cell.append([float(v) for v in fr.cell])
        self.nbytes += table.nbytes + bricks.nbytes
        self.kept += int(len(bricks))
        self.slots += int(len(table))

    def entry(self, step: float) -> dict:
        """The ``fluid`` keyword the frontend's ``FluidVolume.from_bricks``
        is built from."""
        return {"resolution": self.resolution, "tables": self.tables, "bricks": self.bricks,
                "corner": self.corner, "axes": self.axes, "cell": self.cell,
                "start": 0.0, "step": step}

    def report(self) -> dict:
        frames = len(self.resolution)
        largest = (np.max(np.array(self.resolution), axis=0).tolist() if frames
                   else [0, 0, 0])
        return {"field": self.field, "source": self.source, "frames": frames,
                "resolution": [int(v) for v in largest], "bricks": self.kept,
                "brick_slots": self.slots,
                "occupancy": (self.kept / self.slots) if self.slots else 0.0,
                "bytes": int(self.nbytes), "reused": False}


# Each Fluid Flow's last encoded volume, keyed on everything it was read from
# (`_volume_key`), so a drift check (Run, Resume, the MCP hash) on an
# unchanged fluid hashes the same bytes without walking the frames and
# re-reading the source. One entry per field object; an entry ends when its
# field is no longer sent or `release_encode_cache` is called.
_ENCODED: dict = {}
# The size of what the last encode sent per Fluid Flow (`volume_report`).
_REPORT: list = []
# What makes the last encode's payload large: ``(label, bytes)`` of each
# Fluid Flow volume and Texture image it sent, which `upload_refusal` names.
_PARTS: list = []


def release_encode_cache() -> None:
    """Drop every remembered volume, drawn record and size report: the
    volumes and records hold as much memory as the payload, and the report
    describes sources of the file that was open, so a file opened next
    reports no read of its own sources until it encodes them. The next
    encode or draw reads its sources again."""
    _ENCODED.clear()
    _RECORDS.clear()
    _REPORT.clear()
    _PARTS.clear()


def volume_report() -> list:
    """The size of each Fluid Flow volume the last encode sent: dicts of
    ``field``, ``source``, ``frames``, ``resolution`` (the largest per axis),
    ``bricks`` (kept), ``brick_slots`` (the tables' entries), ``occupancy``
    (their ratio), ``bytes`` and ``reused`` (sent from the last encode's
    reading, the source unchanged)."""
    return [dict(r) for r in _REPORT]


def volume_report_line(entry: dict) -> str:
    """One :func:`volume_report` entry, for a report or a console."""
    res = "x".join(str(v) for v in entry["resolution"])
    return (f"Fluid Flow '{entry['field']}': {entry['frames']} frames of '{entry['source']}' "
            f"(up to {res} cells), {_size(entry['bytes'])}, "
            f"{entry['occupancy']:.0%} of bricks occupied")


def _volume_key(scene, obj, frames, solver_fps: float, rate: float) -> tuple:
    """Everything a Fluid Flow's encoded volume is a function of, when its
    Domain Object cannot move (:func:`_source_could_move`): the file, the
    scene, the field's Apply Density, the solve's frames and clock, the
    rate the velocity is scaled by (:func:`fluid_rate`), and
    :func:`_source_state`."""
    import bpy

    return (bpy.data.filepath, scene.name, obj.name, int(frames[0]), int(frames[1]),
            float(solver_fps), float(rate),
            bool(obj.field.use_smoke_density), _source_state(obj.field.source_object))


def _over_cap(total: int, payload: _VolumePayload, obj, frame: int, count: int) -> None:
    """Refuse by name once the volumes and images read so far pass what a
    transfer carries."""
    if total <= TRANSPORT_CAP_BYTES:
        return
    done = len(payload.resolution)
    projected = payload.nbytes * count / max(done, 1)
    summary = payload.report()
    res = "x".join(str(v) for v in summary["resolution"])
    raise ValueError(
        f"Force field '{obj.name}': at frame {frame} the fluid volumes and images read so "
        f"far come to {_size(total)}, past the {_size(TRANSPORT_CAP_BYTES)} a transfer "
        f"carries; this field's fluid from '{payload.source}' (up to {res} cells, "
        f"{summary['occupancy']:.0%} of bricks occupied) projects to "
        f"{_size(int(projected))} over its {count} frames. Lower the domain's resolution "
        "or shorten the solve")


def _payload_refusal(total: int, parts: list) -> None:
    """Refuse by name once the volumes and images an encode sends, read or
    reused, pass what a transfer carries; ``parts`` are ``(label, bytes)``."""
    if total <= TRANSPORT_CAP_BYTES:
        return
    raise ValueError(
        f"Force field: the fluid volumes and images come to {_past_cap(total)} "
        f"({_parts_line(parts)}). Lower a domain's resolution, shorten the solve or use a "
        "smaller image")


def upload_refusal(data_bytes: int, param_bytes: int) -> str | None:
    """Why the server would refuse an upload of ``data_bytes`` of scene
    geometry and ``param_bytes`` of parameters, or None.

    The server refuses an upload whose two payloads TOGETHER pass
    ``MAX_PAYLOAD_BYTES`` before reading it (crates/ppf-cts-server/src/wire/
    upload.rs), and on a streamed connection that refusal reaches the client
    as a reset socket, so every upload is measured here, whole, before it
    starts. The reason names what the last encode put in the parameters
    (each Fluid Flow volume and Texture image), which is what makes them
    large.
    """
    data_bytes, param_bytes = int(data_bytes), int(param_bytes)
    total = data_bytes + param_bytes
    if total <= TRANSPORT_CAP_BYTES:
        return None
    carry = f" ({_parts_line(_PARTS)})" if _PARTS else ""
    remedy = ("Lower a fluid domain's resolution, shorten the solve or use a smaller image"
              if _PARTS else "Send a smaller scene")
    return (f"The upload comes to {_past_cap(total)}: scene geometry {_size(data_bytes)} and "
            f"parameters {_size(param_bytes)}{carry}. {remedy}")


def encode(context, state, start_frame: int, frame_count: int, solver_fps: float,
           position_by_uuid: dict) -> dict | None:
    """The PARAM payload's ``force_field`` entry, or ``None`` when the scene
    carries neither a supported field nor a script.

    Each field object is sent as the keyword arguments of the frontend's
    ``ForceField.add``: its settings, and a track of its origin, axes and
    animatable values in solver axes, sampled at every solve frame (frame
    offsets ``0 .. frame_count - 1`` from ``start_frame``, one ``frame_set``
    per frame for every field, the current frame restored afterward) when
    anything can move it, and once otherwise. A Fluid Flow also carries its
    fluid, read at every solve frame (:func:`fluid_frame`), unless the last
    encode read it from the same unchanged source, whose volume is sent
    again as it was. Large byte strings travel as lists of chunks
    (:func:`_chunks`), and the volumes and images together are refused by
    name past what a transfer carries. ``position_by_uuid`` maps an active
    group's uuid to its position in the payload's group list. Raises
    ``ValueError`` naming the object for anything it cannot send.
    """
    from ..models.force_field_targets import SCRIPT, encode_positions
    scene = context.scene
    count = max(1, int(frame_count))
    frames = (int(start_frame), count)
    objs = field_objects(scene, state)
    # A refusal asks the view layer's depsgraph which domains it evaluates,
    # so it is brought up to date with the scene first.
    context.evaluated_depsgraph_get()
    for obj in objs:
        why = refusal(obj, frames)
        if why:
            raise ValueError(f"Force field '{obj.name}': {why}")
    text = getattr(state, "force_field_script", None)
    source = text.as_string() if text is not None else ""
    if not objs and not source.strip():
        _ENCODED.clear()
        _REPORT.clear()
        _PARTS.clear()
        return None
    if any(o.field.type == "WIND" for o in objs) and float(state.air_density) <= 0.0:
        raise ValueError(
            "Force field: a Wind field blows the air, and this scene's Air Density "
            "is 0, so it would move nothing. Raise Air Density, or use a Force field"
        )
    out: dict = {"fields": [], "scripts": []}
    if source.strip():
        out["scripts"].append({
            "source": source, "name": text.name,
            "groups": encode_positions(scene, state, SCRIPT, position_by_uuid),
        })
    if not objs:
        _ENCODED.clear()
        _REPORT.clear()
        _PARTS.clear()
        return out

    groups = {}
    for obj in objs:
        # Refuses a stale or Static target reference by the field's name.
        groups[obj.name] = encode_positions(scene, state, obj, position_by_uuid)
        if not pushed_objects(scene, state, obj):
            raise ValueError(
                f"Force field '{obj.name}' pushes no object: assign objects to the "
                "groups it reaches"
            )
    rate = fluid_rate(scene, solver_fps)
    step = 1.0 / float(solver_fps)
    # Field-level data, read once at the starting frame: an image as float32
    # and its shape, which the decoder turns back into an array.
    images, image_names = {}, {}
    for obj in objs:
        if obj.field.type == "TEXTURE":
            images[obj.name] = texture_image(obj.field, obj.name).astype("<f4")
            image_names[obj.name] = obj.field.texture.image.name
    total = sum(int(image.nbytes) for image in images.values())
    moving = [o for o in objs if _could_move(o)]
    fluids = [o for o in objs if o.field.type == "FLUID_FLOW"]
    for stale in [k for k in _ENCODED if k not in {o.name for o in fluids}]:
        del _ENCODED[stale]
    keys, reused = {}, {}
    for obj in fluids:
        if _source_could_move(obj.field.source_object):
            _ENCODED.pop(obj.name, None)
            continue
        keys[obj.name] = _volume_key(scene, obj, frames, solver_fps, rate)
        known = _ENCODED.get(obj.name)
        if known is not None and known[0] == keys[obj.name]:
            reused[obj.name] = known
            total += known[2]["bytes"]
    read = [o for o in fluids if o.name not in reused]
    # A fluid changes every frame, so a volume not reused is read at every
    # solve frame whether or not its field object moves.
    n = count if moving or read else 1
    samples = {o.name: [] for o in objs}
    volumes = {o.name: _VolumePayload(o) for o in read}
    discrete = {}
    current = scene.frame_current, scene.frame_subframe
    try:
        for k in range(n):
            frame = int(start_frame) + k
            scene.frame_set(frame)
            for obj in (objs if k == 0 else moving):
                settings = _discrete(obj)
                first = discrete.setdefault(obj.name, settings)
                if settings != first:
                    changed = sorted(key for key in settings if settings[key] != first[key])
                    raise ValueError(
                        f"Force field '{obj.name}': {', '.join(changed)} changes during "
                        f"the solve (at frame {frame}); only continuous "
                        "settings and the transform may be animated"
                    )
                why = refusal(obj, frames)
                if why is not None:
                    raise ValueError(f"Force field '{obj.name}' at frame {frame}: {why}")
                samples[obj.name].append(_sample(obj))
            for obj in read:
                payload = volumes[obj.name]
                before = payload.nbytes
                payload.add(fluid_frame(obj.field, obj.name, rate))
                total += payload.nbytes - before
                _over_cap(total, payload, obj, frame, count)
    finally:
        scene.frame_set(current[0], subframe=current[1])

    report = []
    fluid_entries = {}
    for obj in fluids:
        if obj.name in reused:
            _, entry, summary = reused[obj.name]
            summary = dict(summary, reused=True)
        else:
            entry = volumes[obj.name].entry(step)
            summary = volumes[obj.name].report()
            if obj.name in keys:
                # Keyed on the source as the walk left it, not as it was
                # before: loading a Volume sequence's grids at a frame lets
                # Blender rename its Velocity Grid to the grid that file holds
                # (measured: the default "velocity" becomes "vel"), so a key
                # taken before the walk would never match the next encode's.
                _ENCODED[obj.name] = (_volume_key(scene, obj, frames, solver_fps, rate),
                                      entry, summary)
        fluid_entries[obj.name] = entry
        report.append(summary)
        print(f"[ppf] {volume_report_line(summary)}"
              + (" (unchanged, sent from the last reading)" if summary["reused"] else ""))
    # The walk checks the running total as it reads, to stop early; this
    # checks what is sent whole, the images and the volumes reused from the
    # last encode included, which a walk that reads nothing never sees.
    parts = [(f"Texture '{name}', image '{tex_image}'", int(images[name].nbytes))
             for name, tex_image in image_names.items()]
    parts += [(f"Fluid Flow '{r['field']}', fluid of '{r['source']}'", int(r["bytes"]))
              for r in report]
    _payload_refusal(sum(n for _, n in parts), parts)
    _REPORT[:] = report
    _PARTS[:] = parts

    for obj in objs:
        track = samples[obj.name]
        args = dict(discrete[obj.name])
        for key in ("use_min", "use_max", "use_radial_min", "use_radial_max"):
            args.pop(key)
        for key in ("position", "axes", "noise_scale", "strength", "power",
                    "radial_power", "size", "noise", "wind_factor", "linear_drag",
                    "quadratic_drag", "harmonic_damping", "rest_length", "inflow",
                    "nabla"):
            values = np.array([s[key] for s in track])
            args[key] = values.tolist() if len(track) > 1 else values[0].tolist()
        # A distance is passed only when it is used, which is how the
        # frontend spells "off".
        for key, use in (("min_distance", "use_min"), ("max_distance", "use_max"),
                         ("radial_min", "use_radial_min"),
                         ("radial_max", "use_radial_max")):
            values = [s[key] for s in track]
            on = discrete[obj.name][use]
            args[key] = (values if len(track) > 1 else values[0]) if on else None
        args["start"] = 0.0
        args["step"] = step
        args["groups"] = groups[obj.name]
        if obj.field.type == "TEXTURE":
            tex = obj.field.texture
            image = images[obj.name]
            args["image"] = _chunks(image)
            args["image_shape"] = list(image.shape)
            args["texture_mode"] = obj.field.texture_mode.lower()
            args["extension"] = tex.extension.lower()
            args["interpolation"] = bool(tex.use_interpolation)
            args["object_coordinates"] = bool(obj.field.use_object_coords)
            # The image is read in Blender's axes and its texels are
            # Blender-world vectors (measured).
            args["z_up"] = True
        if obj.field.type == "FLUID_FLOW":
            # Blender stores Flow and Inflow as one value; the frontend calls
            # a fluid's `flow`. The volume travels as its bricks, per solve
            # frame, which the decoder hands to `FluidVolume.from_bricks`.
            args["flow"] = args.pop("inflow")
            args.pop("noise")
            args.pop("wind_factor")
            args["use_density"] = bool(obj.field.use_smoke_density)
            args["fluid"] = fluid_entries[obj.name]
        out["fields"].append(args)
    return out


# --- the exact script, locally, for drawing only -----------------------------

def _import_math_only(name, *args, **kwargs):
    # `import math` is how a script names the math functions, and the
    # compiler admits that one import; anything else is refused here too.
    if name != "math":
        raise ImportError(f"a force field script may import only math, not {name!r}")
    return math


_SCRIPT_BUILTINS = {"abs": abs, "min": min, "max": max, "float": float, "range": range,
                    "__import__": _import_math_only}
# The builtins the compiler admits beyond `math`, the same algorithm the
# solver runs.
_SCRIPT_GLOBALS = {"noise": _noise.noise, "curl_noise": _noise.curl_noise}
# The list the panel shows (`core/script_api.py`) names exactly these.
assert set(_SCRIPT_BUILTINS) - {"__import__"} == {n for n, _, _ in _script_api.PYTHON}
assert set(_SCRIPT_GLOBALS) == {n for n, _, _ in _script_api.NOISE}


def builtins_reference() -> list[dict]:
    """Every function and constant a script may use, as
    ``{"section", "name", "signature", "description"}`` rows, for MCP and the
    Python API."""
    return [{"section": title, "name": name, "signature": sig, "description": what}
            for title, entries in _script_api.SECTIONS for name, sig, what in entries]


def local_script(source: str):
    """The script's function, for the Visualize overlay only.

    Run with ``math`` and five builtins in reach, the same subset the compiler
    admits. What reaches the solver is the SERVER's compile of the text, never
    this.
    """
    namespace = {"math": math, "__builtins__": _SCRIPT_BUILTINS, **_SCRIPT_GLOBALS}
    exec(compile(source, "<force field script>", "exec"), namespace)
    fn = namespace.get("eval")
    if fn is None:
        fns = [v for k, v in namespace.items()
               if callable(v) and k not in _SCRIPT_BUILTINS and k != "__builtins__"
               and k not in _SCRIPT_GLOBALS and not isinstance(v, type(math))]
        fn = fns[0] if len(fns) == 1 else None
    if fn is None:
        raise ValueError("define def eval(x, y, z, t)")
    return fn


# --- Compile and Check --------------------------------------------------------

_check_lock = threading.Lock()
_check_state: dict = {"running": False, "result": None, "text": "", "digest": None}


def check_state() -> dict:
    with _check_lock:
        return dict(_check_state)


def _digest(source: str) -> str:
    import hashlib

    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def check_result_for(source: str) -> dict | None:
    """The last answer, when it was for exactly this text."""
    with _check_lock:
        if _check_state["digest"] == _digest(source):
            return _check_state["result"]
    return None


def request_check(open_channel, source: str, text_name: str) -> None:
    """Ask the server to compile ``source``, on a thread of its own.

    ``open_channel`` is the live connection's channel opener. The answer lands
    in :func:`check_state`; the caller polls it. A connect attempt or a Run
    never waits on this, since it holds no I/O worker.
    """
    with _check_lock:
        if _check_state["running"]:
            return
        _check_state.update(running=True, result=None, text=text_name, digest=_digest(source))

    def work():
        try:
            answer = json_request(open_channel, {
                "request": "force_field_check", "source": source, "z_up": True,
            })
            if "error" in answer:
                result = {"ok": False, "error": str(answer["error"]), "line": None,
                          "transport": True}
            else:
                result = dict(answer.get("result") or {})
        except Exception as e:  # the connection itself failed
            result = {"ok": False, "error": f"the server could not be asked: {e}",
                      "line": None, "transport": True}
        with _check_lock:
            _check_state.update(running=False, result=result)

    threading.Thread(target=work, name="ppf-force-field-check", daemon=True).start()


def json_request(open_channel, request: dict, timeout: float = 60.0) -> dict:
    """Send one JSON request and read the one-line JSON answer."""
    from .protocol import _send_json_header

    channel = open_channel()
    try:
        channel.settimeout(timeout)
        _send_json_header(channel, request)
        response = b""
        while b"\n" not in response:
            chunk = channel.recv(65536)
            if not chunk:
                break
            response += chunk
    finally:
        channel.close()
    line = response.split(b"\n", 1)[0]
    if not line:
        raise RuntimeError("the server closed the connection without answering")
    return json.loads(line.decode("utf-8"))
