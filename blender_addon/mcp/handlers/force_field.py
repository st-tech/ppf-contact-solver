# File: force_field.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP tools for the scene's force fields (issues #151 and #114): the settings
# the Force Fields panel shows, and the server's Compile and Check.

from typing import Optional

import bpy  # pyright: ignore

from ..decorators import MCPError, mcp_handler
from ...models.groups import get_addon_data


def _settings(state) -> dict:
    return {
        "collection": state.force_field_collection.name if state.force_field_collection else None,
        "script_text": state.force_field_script.name if state.force_field_script else None,
        "visualize": state.force_field_visualize,
        "preview_resolution": list(state.force_field_preview_resolution),
    }


@mcp_handler
def get_force_field_settings():
    """Get the scene's force field settings, the field objects found, and the
    last Compile and Check answer.

    `fields` lists every object with a force field that Transfer considers
    (in the chosen collection, or the whole scene). Each entry holds:

    - `object`: the object's name.
    - `type`: Blender's `field.type` identifier, what bpy reads and sets.
      `type_name` is the name Blender's Physics tab shows for it. Sent:
      FORCE (Force), WIND (Wind), VORTEX (Vortex), TURBULENCE (Turbulence),
      MAGNET (Magnetic), DRAG (Drag), HARMONIC (Harmonic), TEXTURE
      (Texture, an image texture) and FLUID_FLOW (Fluid Flow). Refused by
      name, having no effect on cloth: CHARGE (Charge), BOID (Boid),
      LENNARDJ (Lennard-Jones) and GUIDE (Curve Guide).
    - `refused`: why Transfer refuses the field, or null. It covers the
      field's settings, the scene around it (a Wind with Air Density 0, a
      target group that cannot be used) and a Fluid Flow's source as far as
      its metadata at the current frame shows (no file, no grid by the name
      the Volume's Velocity settings give, no density grid with Apply
      Density, a domain holding no fluid). Transfer reads the source at the
      solve frames only, so a finding at a current frame outside them is no
      refusal: it is `fluid.note` instead, except for a Volume that is not a
      sequence, which reads its one file at every frame. Transfer reads
      every solve frame and every voxel, so it can still refuse what this
      cannot see: a problem first reached at another solve frame, a setting
      keyed to change during the solve, or a voxel-level problem in a file.
    - `groups`: the group uuids the field is narrowed to, or null when it
      pushes every group. `script_groups` is the same for the script.
    - `pushes`: the simulated objects it reaches; Transfer refuses a field
      that pushes none.
    - `tracked`: true when Transfer samples the field at every solve frame
      (anything can move or animate it, or it is a Fluid Flow), so a
      Transfer steps the timeline through every solve frame.
    - `fluid` (Fluid Flow only): its source at the current frame, read from
      metadata, never from the grid. `source` (the name of the field's
      Domain Object, Blender's `source_object`) and `kind` ("domain" for a
      Mantaflow gas domain, "volume" for a Volume object, null when it is
      neither or unset), `frame` (the current frame), `solve_start` and
      `solve_frames` (the first solve frame and how many there are, the
      frames Transfer reads), `resolution` (cells along the source's x, y, z
      at this frame, or null when unknown), `bytes_per_frame_max` and
      `bytes_max` (an upper bound on what the volume adds to the parameter
      payload per frame and over every solve frame at this frame's
      resolution: every 8 x 8 x 8 brick of the box, while a frame sends only
      its non-empty bricks, so a sparse one is smaller), `last_read` (what
      the last parameter encode read of it: `frames`, `resolution`,
      `bricks`, `brick_slots`, `occupancy` and `bytes`, or null when no
      encode has read this source), `problem` (the source's part of
      `refused`, or null) and `note` (what the metadata at the current frame
      shows wrong when Transfer does not read that frame, so it refuses
      nothing for it; null otherwise).
      A domain adds `domain_type` (GAS or LIQUID), `cache_type` (REPLAY,
      MODULAR or ALL), `baked` and `cache_frames`; a Volume adds `file`
      (this frame's file), `sequence`, `velocity_grids` (the grids its
      Velocity settings name, as Transfer reads them) and `density_grid`.

    `check` is the last server answer for the current script text, or null
    when the text changed since, or was never checked. `script_builtins`
    lists every function and constant a script may use, each with its
    section, name, signature and description.
    """
    from ...core import force_field as ff
    from ...ui.dynamics import force_field_status as status

    scene = bpy.context.scene
    state = get_addon_data(scene).state
    text = state.force_field_script
    from ...models import force_field_targets as targets

    return {
        "settings": _settings(state),
        "fields": status.field_rows(scene, state),
        "script_groups": targets.target_group_uuids(state, targets.SCRIPT),
        "check": ff.check_result_for(text.as_string()) if text is not None else None,
        "check_running": ff.check_state()["running"],
        "script_builtins": ff.builtins_reference(),
    }


@mcp_handler
def set_force_field_settings(
    collection: Optional[str] = None,
    script_text: Optional[str] = None,
    script_source: Optional[str] = None,
    visualize: Optional[bool] = None,
    preview_resolution: Optional[list[int]] = None,
):
    """Set the scene's force field settings.

    Blender's force field objects are sent with Transfer as field objects,
    which the solver evaluates exactly at every vertex: no resolution and no
    domain. A field's world transform and animatable settings are sampled at
    every solve frame when anything can move or animate it (keys, drivers, a
    parent, a constraint), once otherwise; its other settings must not change
    during the solve. Blender's own settings mean what they mean in its
    Physics tab, with these differences and units:

    - Strength is per unit mass, the same push on a light or a heavy cloth;
      Blender's Strength is a force, so the two agree at mass 1. Lengths are
      scene units and t is seconds.
    - Shape Point or Plane (other shapes are refused); falloff Sphere, Tube
      or Cone, its weight W from Falloff Power and Min and Max Distance. A
      Cone's Radial Min and Max are degrees. Z Direction keeps one side of
      the field's XY plane. Gravitation multiplies by 1/d^2, exactly and
      unbounded: Min Distance does not bound it, and a vertex exactly at the
      field's center (on a Plane, on its plane) stops the run with an error
      naming the field.
    - Force: s W toward or away from the center (a Plane pushes away from
      its plane), Strength in m/s^2.
    - Wind: an air velocity of s W m/s along the field's Z, acting through
      the air drag, so it acts on a surface by how it faces the flow and
      needs a positive Air Density. A Wind is the air flow Blender's Flow
      asks for, so its Flow adds nothing, and its Wind Factor is not read.
    - Vortex: a Point swirls at `s W |p|` around Z, Strength in 1/s^2, p the
      point in the field's frame; Inflow damps the velocity by Inflow W. A
      Plane drives the velocity toward the swirl s F(r) r around Z
      (Strength in 1/s) with its centripetal part, always damping toward it.
    - Turbulence: a noise pattern whose swirls are Size across, Strength in
      m/s^2. The pattern is this add-on's own, not Blender's; a Size of 0 is
      refused.
    - Magnetic: s W (v x t) on a Point, t the tangent around Z, and
      s W (v x z) on a Plane, Strength in 1/s; nothing acts on a cloth at
      rest.
    - Harmonic: a spring `-s W (|p| - Rest Length)` toward the center (a Plane
      pulls toward its plane), Strength in 1/s^2, critically damped by
      Harmonic Damping; a negative Strength pushes away. Multiple Springs is
      refused.
    - Drag: `-W (Linear Drag |v| + Quadratic Drag) v`, which is how Blender's
      field behaves (its labels read swapped); Strength is not read.
    - Texture, an image texture: RGB mode pushes s W (0.5 - color), Strength
      in m/s^2; Gradient and Curl take forward differences Nabla apart. The
      image spans x and y from -1 to 1 in scene units of WORLD coordinates
      (the field's own with Object Coordinates), and its Extension is Extend,
      Repeat, Clip or Clip Cube. Pixels are linearized by the image's
      colorspace; with Interpolation the solver reads bilinearly between
      pixels where Blender filters. Refused by name: a procedural texture,
      the Checker extension, Root Texture Coordinates, an image sequence,
      movie, tiled (UDIM) or Render Result image, a colorspace other than
      sRGB, Non-Color or Linear Rec.709, Use Coordinates on a field object
      whose transform is sheared (a rotated child of a non-uniformly scaled
      parent), and any texture setting the solver does not read when it is
      not at its default (Flip Axis, Crop, Repeat and Mirror
      under the Repeat extension, Brightness, Contrast, Saturation and the
      color factors, a color ramp, Filter Size, Calculate Alpha).
    - Fluid Flow: pushes by `s W rho (U - Flow v)`, Strength in 1/s, U the
      fluid's velocity and v the vertex's, so at Flow 1 the cloth relaxes
      toward the fluid at rate Strength, solved implicitly. rho is the smoke
      density with Apply Density, otherwise 1 inside the fluid's box and
      0 outside: outside it nothing acts, damping included. U is read
      trilinearly between cell centers and follows the scene's World Scaling
      and Time Scale as every length and rate does. The Domain Object is
      either of two:

      * A Mantaflow gas domain baked with Cache Type All or Modular over
        every solve frame. Its velocity is converted from grid units at 2.5
        times its longest axis's cell per second and multiplied by the
        domain's Time Scale, so the cloth moves with the smoke as it is
        simulated; Blender's own field does not apply Time Scale and scales
        a non-cubic domain per axis. A Replay cache, an unbaked domain, an
        Adaptive Domain, a modifier after the domain's Fluid modifier, and
        Apply Density on a domain with Noise are refused.
      * A Volume object: the grid its Volume data's Velocity settings name
        (one vector grid, or three component grids), in its Velocity Unit
        (per frame, at the scene's frame rate, or per second) times Velocity
        Scale. A staggered grid is read at cell centers, and a grid named
        `density` on the velocity grid's voxels serves Apply Density.
        The velocity turns and scales with the object, a different scale
        along each of the voxels' axes included; a modifier, and a transform
        that shears, mirrors or flattens the voxels (a non-uniform scale
        across voxels the file or a parent rotates), are refused. A file
        inside a fluid domain's cache directory is refused, since a
        Mantaflow cache holds grid units: point the field at the domain
        instead.

      The source is read at every solve frame, and the solver blends the two
      frames around each step's end, where Blender holds the next frame's
      grid for the whole frame. Noise Amount, Wind Factor and Absorption,
      which Blender applies to a Fluid Flow though its panel hides them, are
      refused on one; Gravitation, which Blender ignores there, is sent
      off. The volume travels in the parameter
      payload, so Transfer and Update Params send it again;
      get_force_field_settings bounds its size.
    - Noise Amount adds gusts as a fraction of the strength, as large as the
      object's scale (a Turbulence's Size on a Turbulence), and on a Wind
      they drift with the flow; the noise is this add-on's own. Wind Factor
      weighs a push by how the cloth faces it, `(1 - wf) a + wf (a . n) n`.
    - Drag, Harmonic, a Vortex's damping and a Fluid Flow are solved
      implicitly, so strong values stay stable.
    - Refused by name, with the reason get_force_field_settings lists per
      field: Charge, Boid, Lennard-Jones and Curve Guide (no effect on
      cloth), 2D, Absorption, Affect Location off, Apply Density on a
      field that is not a Fluid Flow, Flow on a type other than Wind, Vortex
      and Fluid Flow, and a zero or mirrored object scale.

    The script is a Text holding `def eval(x, y, z, t):` returning
    `(ax, ay, az)` in m/s^2, in Blender's axes and scene units, t in seconds.
    It is evaluated exactly at every vertex on the solver. Only arithmetic,
    comparisons, if/else, local variables, `for i in range(<number>)`, abs,
    min, max, float, math functions, `noise(x, y, z, octaves=1, seed=0,
    time=0.0, frequency=1.0, decay=0.0)` (a scalar in about [-1, 1]) and
    `curl_noise(...)` with the same arguments (a swirling divergence-free
    vector: return it, or unpack it into three names) are allowed; octaves is
    a whole number 1 to 8 written in the script. With time=t the pattern
    evolves in place `frequency` times per second and fades as
    exp(-decay * t). get_force_field_settings' `script_builtins` has the
    complete list. Check
    it with check_force_field_script. Each source reaches every group unless
    narrowed with set_force_field_targets.

    An empty string for `collection` or `script_text` clears it.

    Args:
        collection: Take field objects only from this collection
        script_text: Name of the Text to use as the script
        script_source: Replace the chosen script's text with this source, or
            create a Text named force_field.py holding it when none is chosen
        visualize: Draw the field as arrows in the viewport
        preview_resolution: Arrows drawn along X, Y and Z (drawing only), at
            the timeline's current frame
    """
    state = get_addon_data(bpy.context.scene).state

    def lookup(kind, name):
        if name == "":
            return None
        found = getattr(bpy.data, kind).get(name)
        if found is None:
            raise MCPError(f"no {kind[:-1]} named {name!r}")
        return found

    if collection is not None:
        state.force_field_collection = lookup("collections", collection)
    if preview_resolution is not None:
        if len(preview_resolution) != 3 or any(int(v) < 2 for v in preview_resolution):
            raise MCPError("preview_resolution takes three counts, each at least 2")
        state.force_field_preview_resolution = [int(v) for v in preview_resolution]
    if script_text is not None:
        state.force_field_script = lookup("texts", script_text)
    if script_source is not None:
        text = state.force_field_script
        if text is None:
            text = bpy.data.texts.new("force_field.py")
            state.force_field_script = text
        text.from_string(script_source)
    if visualize is not None:
        state.force_field_visualize = visualize
    return {"settings": _settings(state)}


@mcp_handler
def check_force_field_script():
    """Ask the running server to compile the force field script, as Compile
    and Check does, without a Transfer.

    Returns at once; the answer arrives in get_force_field_settings under
    `check` (ok, and either a summary or an error with its line). Needs a
    connection to a running server and a chosen script.
    """
    from ...core import force_field as ff
    from ...core.facade import communicator as com
    from ...ui.dynamics.force_field_ops import check_unavailable_reason

    reason = check_unavailable_reason(bpy.context)
    if reason:
        raise MCPError(reason)
    text = get_addon_data(bpy.context.scene).state.force_field_script
    ff.request_check(com.channel_opener(), text.as_string(), text.name)
    return {"message": "check started"}


@mcp_handler
def set_force_field_targets(source: str, group_uuids: Optional[list[str]] = None):
    """Point a force field source at every simulated group, or at chosen groups.

    A source reaches every group by default. Narrowed, it pushes only the
    objects of the groups named; Static groups are refused, since colliders
    ignore force fields.

    Args:
        source: A force field object's name, or "SCRIPT" for the script
        group_uuids: The groups to push, by uuid; null or omitted for every group
    """
    from ...models import force_field_targets as targets

    scene = bpy.context.scene
    state = get_addon_data(scene).state
    try:
        resolved = targets.resolve_source(state, source)
        targets.set_targets(scene, state, resolved, group_uuids)
    except ValueError as e:
        raise MCPError(str(e)) from None
    return {"source": source,
            "groups": targets.target_group_uuids(state, resolved)}
