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
        "padding": state.force_field_padding,
        "spacing": state.force_field_spacing,
        "time_samples": state.force_field_time_samples,
        "max_mb": state.force_field_max_mb,
        "script_text": state.force_field_script.name if state.force_field_script else None,
        "visualize": state.force_field_visualize,
        "preview_resolution": list(state.force_field_preview_resolution),
    }


@mcp_handler
def get_force_field_settings():
    """Get the scene's force field settings, the field objects found, and the
    last Compile and Check answer.

    `script_builtins` lists every function and constant a script may use,
    each with its section, name, signature and description.

    Each field lists `groups`, the group uuids it is narrowed to, or null when
    it pushes every group; `script_groups` is the same for the script.

    Each field object is listed with its type and, when Transfer would refuse
    it, the reason. `grids` lists the grids Transfer would send, each with its
    fields, its sample counts along X, Y and Z and its box in Blender world
    space; `estimate` is the `[Info] ... MB estimated` line for them, or the
    reason there are none. `check` is the last server answer for the current script
    text, or null when the text changed since, or was never checked.
    """
    from ...core import force_field as ff

    scene = bpy.context.scene
    state = get_addon_data(scene).state
    objs = ff.field_objects(scene, state)
    try:
        plan = ff.grid_plan(scene, state)
        estimate = ff.estimate_line(ff.plan_shapes(plan), state.force_field_time_samples)
        grids = [{"fields": [o.name for o in e["fields"]], "shape": list(e["shape"]),
                  "min": [float(v) for v in e["lo"]], "max": [float(v) for v in e["hi"]]}
                 for e in plan]
    except ValueError as e:
        estimate, grids = str(e), []
    text = state.force_field_script
    from ...models import force_field_targets as targets

    return {
        "settings": _settings(state),
        "fields": [
            {"object": o.name, "type": o.field.type, "refused": ff.refusal(o),
             "groups": targets.target_group_uuids(state, o)}
            for o in objs
        ],
        "script_groups": targets.target_group_uuids(state, targets.SCRIPT),
        "estimate": estimate,
        "grids": grids,
        "check": ff.check_result_for(text.as_string()) if text is not None else None,
        "check_running": ff.check_state()["running"],
        "script_builtins": ff.builtins_reference(),
    }


@mcp_handler
def set_force_field_settings(
    collection: Optional[str] = None,
    padding: Optional[float] = None,
    spacing: Optional[float] = None,
    time_samples: Optional[int] = None,
    max_mb: Optional[float] = None,
    script_text: Optional[str] = None,
    script_source: Optional[str] = None,
    visualize: Optional[bool] = None,
    preview_resolution: Optional[list[int]] = None,
):
    """Set the scene's force field settings.

    Blender Force, Wind, Vortex and Turbulence objects (shape Point or Plane,
    falloff Sphere or Tube) are sampled on a grid of points `spacing` apart
    over the box around the objects each field pushes, grown by `padding`, at
    `time_samples` instants spread over the simulation, and sent with
    Transfer. Force, Vortex and Turbulence strengths are accelerations in
    m/s^2; Wind is an air velocity in m/s acting through the air drag, so it
    needs a positive air density. Outside the box the sampled fields are zero.

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
        padding: How far each field's box extends past the objects it pushes
        spacing: Distance between sample points, the same along X, Y and Z
        time_samples: Instants sampled over the simulation, at least 1
        max_mb: Transfer refuses sampled grids larger than this
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
    if padding is not None:
        if padding < 0.0:
            raise MCPError("padding must be 0 or more")
        state.force_field_padding = padding
    if spacing is not None:
        if spacing < 1e-3:
            raise MCPError("spacing must be at least 0.001")
        state.force_field_spacing = spacing
    if preview_resolution is not None:
        if len(preview_resolution) != 3 or any(int(v) < 2 for v in preview_resolution):
            raise MCPError("preview_resolution takes three counts, each at least 2")
        state.force_field_preview_resolution = [int(v) for v in preview_resolution]
    if time_samples is not None:
        if time_samples < 1:
            raise MCPError("time_samples must be at least 1")
        state.force_field_time_samples = time_samples
    if max_mb is not None:
        state.force_field_max_mb = max_mb
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
