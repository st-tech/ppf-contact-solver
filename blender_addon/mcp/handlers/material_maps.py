# File: handlers/material_maps.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP handlers for a group's spatial material maps.
#
# A map varies one material parameter across the surface: the value at a vertex
# is lerp(base, target, weight), with `base` the group's own slider, so a weight
# of 0 reproduces the unmapped result exactly. Every rule enforced here belongs
# to the addon: `models/material_maps.py` holds which parameters a map may
# drive, and `core/encoder/material_maps.py` holds what a row must carry to
# reach the solver. Each check applies one of those rules at authoring time, so
# a caller is told which field to change instead of receiving a build failure
# from a scene that was already written.

import math

import bpy  # pyright: ignore

from ...models.collection_utils import (
    safe_update_index,
    sort_keyframes_by_frame,
    validate_no_duplicate_frame,
)
from ...models.material_maps import (
    SOURCE_TYPE_ITEMS,
    base_property,
    enum_items,
    gate_reason,
)
from ..decorators import MCPError, group_handler
from .group import get_active_group_by_uuid_helper

_PARAMETERS = tuple(key for key, _label, _desc, _num in enum_items())
_SOURCE_TYPES = tuple(ident for ident, _label, _desc, _num in SOURCE_TYPE_ITEMS)

# `pressure` is offered by the enum and deliberately carries no base property,
# so it is refused by name here rather than by the object-type check, which
# would report it as unavailable on every type and name no reason. The physics
# is recorded in full in models/material_maps.py: the per-face pressure
# potential has a translation-variant gradient, and the cancellation that makes
# the assembled force origin-independent holds only while the pressure is
# uniform. Measured on a 5 cm icosphere with the pressure painted 20 to 100,
# per-vertex forces change by 275% of their peak when the object is moved 1 m,
# and by 2202% at 8 m, while a uniform pressure is exactly invariant.
_PRESSURE_REFUSAL = (
    "'pressure' cannot be driven by a spatial map. Its per-face potential is "
    "translation-variant, and the cancellation that makes the assembled force "
    "independent of the origin holds only while the pressure is uniform, so "
    "the same painted weights would mean a different force depending on where "
    "the object sits in the scene. Set the group's uniform 'inflate_pressure' "
    "instead."
)


# ---------------------------------------------------------------------------
# Resolvers and validation
# ---------------------------------------------------------------------------


def _resolve_map(group_uuid: str, index: int):
    """Return (group, map) for the map at `index`, or raise MCPError."""
    group = get_active_group_by_uuid_helper(group_uuid)
    count = len(group.material_maps)
    if not 0 <= index < count:
        raise MCPError(
            f"Group {group_uuid} has no material map at index {index}; it has "
            f"{count}. Call list_material_maps for the current indices."
        )
    return group, group.material_maps[index]


def _check_parameter(group, parameter) -> str:
    """The solver key `parameter` names, or raise MCPError.

    A map is reduced to one coefficient per element by averaging that element's
    own vertices, so the parameter has to be one this group's elements read.
    `base_property` is the addon's own table of which those are.
    """
    if parameter not in _PARAMETERS:
        # List only what a caller can actually use. Naming a parameter here
        # that the next branch refuses would send the caller straight into a
        # second refusal.
        usable = [name for name in _PARAMETERS if name != "pressure"]
        raise MCPError(
            f"Unknown parameter '{parameter}'. Valid parameters are "
            f"{', '.join(usable)}."
        )
    if parameter == "pressure":
        raise MCPError(_PRESSURE_REFUSAL)
    if base_property(parameter, group.object_type) is None:
        raise MCPError(
            f"'{parameter}' is not available as a map on a "
            f"{group.object_type} group. A map is reduced to one coefficient "
            "per element, so the parameter has to be one this group's elements "
            "read."
        )
    return parameter


def _check_source(source_type, source_name) -> tuple[str, str]:
    """The (source_type, source_name) pair to store, or raise MCPError.

    The named vertex group or attribute is resolved when the scene is built,
    not here: an attribute a Geometry Nodes modifier writes exists only on the
    evaluated mesh, which a caller is entitled to set up after the map.
    """
    if source_type not in _SOURCE_TYPES:
        raise MCPError(
            f"Unknown source_type '{source_type}'. Valid source types are "
            f"{', '.join(_SOURCE_TYPES)}."
        )
    if not isinstance(source_name, str) or not source_name:
        raise MCPError(
            "source_name must be a non-empty string naming the vertex group or "
            "float attribute that holds the weights. A map with no source name "
            "is refused when the scene is built."
        )
    return source_type, source_name


def _check_target(group, parameter: str, target_value) -> float:
    """`target_value` as a float the mapped slider itself would accept.

    Both bounds are read from the base property's own RNA rather than assumed:
    `young-mod` has a positive minimum, so a target of 0.0 would clear a
    non-negative test here and then abort the build, and several parameters
    carry a hard ceiling the artist's slider enforces. `target_value` is a
    plain float with no range of its own, so a value outside the slider's range
    is stored intact and only fails later, at encode.
    """
    try:
        value = float(target_value)
    except (TypeError, ValueError) as exc:
        raise MCPError(f"target_value must be a number: {exc}") from exc
    if not math.isfinite(value):
        raise MCPError(
            f"target_value {value} is not a finite number, so it names no "
            "point between the group's own value and the map target."
        )
    prop_name = base_property(parameter, group.object_type)
    prop = group.bl_rna.properties[prop_name]
    floor = float(prop.hard_min)
    ceiling = float(prop.hard_max)
    if value < floor:
        raise MCPError(
            f"target_value {value} is below the {floor} minimum the "
            f"'{prop_name}' slider itself enforces. The blend runs from that "
            "slider to the target, so both ends carry the same units and the "
            "same range."
        )
    if value > ceiling:
        raise MCPError(
            f"target_value {value} is above the {ceiling} maximum the "
            f"'{prop_name}' slider itself enforces. The blend runs from that "
            "slider to the target, so both ends carry the same units and the "
            "same range."
        )
    return value


def _check_enabled(enabled):
    """`enabled` as a bool, or raise MCPError.

    An optional boolean reaches the handler exactly as the client sent it, so a
    string is checked rather than coerced: `bool("false")` is True, which would
    turn a request to disable a map into a request to enable it.
    """
    if not isinstance(enabled, bool):
        raise MCPError(
            "enabled must be the JSON literal true or false, not "
            f"{type(enabled).__name__}."
        )
    return enabled


def _check_one_map_per_parameter(group, parameter: str, enabled: bool, skip: int):
    """Refuse a second enabled map on `parameter`, ignoring row `skip`.

    Two maps on one parameter have no defined composition: the blend runs from
    the group's own slider to the target, so a second target is a different
    answer for the same value rather than a refinement of it. A disabled row is
    ignored, because the encoder ignores it too.
    """
    if not enabled:
        return
    for i, other in enumerate(group.material_maps):
        if i == skip or not other.enabled or other.parameter != parameter:
            continue
        raise MCPError(
            f"Group '{group.name}' already has an enabled map at index {i} "
            f"driving '{parameter}'; each parameter takes at most one. Disable "
            "or remove that map, or edit it with set_material_map."
        )


def _start_frame() -> int:
    """The Blender frame the simulation begins on."""
    from ...core.encoder import resolve_start_frame
    from ...models.groups import get_addon_data

    return resolve_start_frame(get_addon_data(bpy.context.scene).state)


def _serialize_map(group, entry, index: int) -> dict:
    """One map as an MCP caller sees it."""
    return {
        "index": index,
        "parameter": entry.parameter,
        "source_type": entry.source_type,
        "source_name": entry.source_name,
        "target_value": float(entry.target_value),
        "enabled": bool(entry.enabled),
        # The slider the map blends away from, so a caller can read or set the
        # weight-0 end with the material property tools.
        "base_property": base_property(entry.parameter, group.object_type),
        # Not a refusal: a closed gate zeroes the parameter for the whole solve
        # and a map target cannot reintroduce it, so the build refuses the map
        # while the gate stays closed.
        "gate_closed_reason": gate_reason(group, entry.parameter),
        "samples": [
            {
                "frame": int(sample.frame),
                "source_type": sample.source_type,
                "source_name": sample.source_name,
            }
            for sample in entry.samples
        ],
    }


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


@group_handler
def add_material_map(
    group_uuid: str,
    parameter: str,
    source_type: str,
    source_name: str,
    target_value: float,
    enabled: bool = True,
):
    """Add a spatial material map, varying one parameter across the surface.

    The value at a vertex is lerp(base, target, weight), where base is the
    group's own slider for that parameter and weight is read per vertex from
    the named source, clamped to [0, 1]. A weight of 0 therefore reproduces the
    unmapped result exactly. Each element takes the mean of its own vertices'
    weights.

    Only SHELL and SOLID groups carry the element tables a map is reduced over,
    and each object type reads a different set of parameters, so 'parameter' is
    checked against this group's type. 'pressure' is never mappable. A group
    takes at most one enabled map per parameter.

    The source is resolved when the scene is built, so the vertex group or
    attribute does not have to exist yet. A vertex group is read by name from
    the object; an attribute is read from the evaluated mesh on the POINT
    domain, which is where a Store Named Attribute node writes one.

    Args:
        group_uuid: UUID of the group to add the map to.
        parameter: Solver key to vary. One of young-mod, bend, friction,
            deformation-damping, bending-damping, strain-limit, plasticity,
            bend-plasticity, bend-warp, bend-weft.
        source_type: VERTEX_GROUP to read weight paint, ATTRIBUTE to read a
            float attribute off the evaluated mesh.
        source_name: Name of the vertex group or float attribute holding the
            weights at the start frame.
        target_value: Value reached where the weight is 1, in the same units as
            the group's own slider for this parameter.
        enabled: Whether the map is included in the simulation.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    parameter = _check_parameter(group, parameter)
    source_type, source_name = _check_source(source_type, source_name)
    target = _check_target(group, parameter, target_value)
    enabled = _check_enabled(enabled)
    _check_one_map_per_parameter(group, parameter, enabled, skip=-1)

    entry = group.material_maps.add()
    entry.parameter = parameter
    entry.source_type = source_type
    entry.source_name = source_name
    entry.target_value = target
    entry.enabled = enabled
    index = len(group.material_maps) - 1
    # Select the new row, as the add operator does: the panel draws the sample
    # list of the selected map only.
    group.material_maps_index = index
    return {
        "message": f"Added a '{parameter}' material map to group '{group.name}'",
        "group_uuid": group_uuid,
        "index": index,
        "map": _serialize_map(group, entry, index),
        "map_count": len(group.material_maps),
    }


@group_handler
def set_material_map(
    group_uuid: str,
    index: int,
    parameter: str | None = None,
    source_type: str | None = None,
    source_name: str | None = None,
    target_value: float | None = None,
    enabled: bool | None = None,
):
    """Edit fields of an existing spatial material map.

    Every field left out keeps its current value. The whole resulting row is
    validated before anything is written, so a refusal leaves the map exactly
    as it was. That means changing 'parameter' alone can be refused because the
    target already stored is below the new parameter's own minimum; pass both
    in one call.

    Args:
        group_uuid: UUID of the group that owns the map.
        index: Zero-based index as reported by list_material_maps.
        parameter: New solver key to vary, or omit to keep the current one.
        source_type: VERTEX_GROUP or ATTRIBUTE, or omit to keep the current one.
        source_name: New vertex group or attribute name, or omit to keep it.
        target_value: New value reached where the weight is 1, or omit to keep
            it.
        enabled: Whether the map is included in the simulation, or omit to keep
            the current setting.
    """
    group, entry = _resolve_map(group_uuid, index)
    if all(
        field is None
        for field in (parameter, source_type, source_name, target_value, enabled)
    ):
        raise MCPError(
            "set_material_map was called with no field to change. Pass at "
            "least one of parameter, source_type, source_name, target_value, "
            "enabled."
        )

    new_parameter = _check_parameter(
        group, entry.parameter if parameter is None else parameter
    )
    new_source_type, new_source_name = _check_source(
        entry.source_type if source_type is None else source_type,
        entry.source_name if source_name is None else source_name,
    )
    new_target = _check_target(
        group,
        new_parameter,
        entry.target_value if target_value is None else target_value,
    )
    new_enabled = _check_enabled(
        bool(entry.enabled) if enabled is None else enabled
    )
    _check_one_map_per_parameter(group, new_parameter, new_enabled, skip=index)

    entry.parameter = new_parameter
    entry.source_type = new_source_type
    entry.source_name = new_source_name
    entry.target_value = new_target
    entry.enabled = new_enabled
    return {
        "message": f"Updated material map {index} on group '{group.name}'",
        "group_uuid": group_uuid,
        "index": index,
        "map": _serialize_map(group, entry, index),
    }


@group_handler
def remove_material_map(group_uuid: str, index: int):
    """Remove a spatial material map and every weight source on it.

    Args:
        group_uuid: UUID of the group that owns the map.
        index: Zero-based index as reported by list_material_maps. Removing a
            map renumbers the ones after it.
    """
    group, entry = _resolve_map(group_uuid, index)
    parameter = entry.parameter
    group.material_maps.remove(index)
    # Keep the selection on a real row rather than one past the end, and on row
    # 0 rather than -1 once the last row is gone.
    group.material_maps_index = safe_update_index(index, len(group.material_maps))
    return {
        "message": (
            f"Removed the '{parameter}' material map from group '{group.name}'"
        ),
        "group_uuid": group_uuid,
        "index": index,
        "map_count": len(group.material_maps),
    }


@group_handler
def list_material_maps(group_uuid: str):
    """List a group's spatial material maps and its mappable parameters.

    Each map reports the slider it blends away from as 'base_property', and
    'gate_closed_reason' whenever the parameter is switched off for the whole
    solve, in which case the build refuses the map: a map target cannot
    reintroduce a value the group turned off.

    'available_parameters' is what this group's object type can map, which is
    what add_material_map accepts. 'start_frame' is the frame the map's own
    source describes, and every weight sample has to sit after it.

    Args:
        group_uuid: UUID of the group to report.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    return {
        "group_uuid": group_uuid,
        "group_name": group.name,
        "object_type": group.object_type,
        "start_frame": _start_frame(),
        "available_parameters": [
            key
            for key in _PARAMETERS
            if base_property(key, group.object_type) is not None
        ],
        "maps": [
            _serialize_map(group, entry, i)
            for i, entry in enumerate(group.material_maps)
        ],
        "map_count": len(group.material_maps),
    }


# ---------------------------------------------------------------------------
# Weight sources over time
# ---------------------------------------------------------------------------


@group_handler
def add_material_map_sample(
    group_uuid: str,
    index: int,
    frame: int,
    source_name: str,
    source_type: str | None = None,
):
    """Add a later weight source to a spatial material map.

    The map's own source is the weights at the simulation start frame, and each
    sample names a different source reached at its own frame. Between two
    consecutive samples the weights are the linear interpolation of the two, so
    a constant hold is two samples naming one source.

    A frame at or before the start frame is refused, because the map's own
    source already describes that frame. Only a SHELL group carries a
    per-element material schedule, so a map on any other type takes a single
    source and no samples.

    Args:
        group_uuid: UUID of the group that owns the map.
        index: Zero-based index of the map, as reported by list_material_maps.
        frame: Blender frame at which the weights are exactly this source.
        source_name: Vertex group or float attribute holding this sample's
            weights.
        source_type: VERTEX_GROUP or ATTRIBUTE. Omit to use the map's own
            source type.
    """
    group, entry = _resolve_map(group_uuid, index)
    if group.object_type != "SHELL":
        raise MCPError(
            f"A {group.object_type} group carries no per-element material "
            "schedule, so its maps take a single source and no samples. The "
            "map's own source stays in effect for the whole solve."
        )
    source_type, source_name = _check_source(
        entry.source_type if source_type is None else source_type,
        source_name,
    )
    start_frame = _start_frame()
    if frame <= start_frame:
        raise MCPError(
            f"Frame {frame} is at or before the start frame {start_frame}. The "
            "map's own source is already the weights at the start frame, so a "
            "sample there would name the same time twice. Use a later frame, "
            "or change the map's own source_name."
        )
    try:
        validate_no_duplicate_frame(entry.samples, frame)
    except ValueError as exc:
        raise MCPError(str(exc)) from exc

    sample = entry.samples.add()
    sample.frame = frame
    sample.source_type = source_type
    sample.source_name = source_name
    entry.samples_index = sort_keyframes_by_frame(entry.samples)
    return {
        "message": (
            f"Added a '{entry.parameter}' map sample at frame {frame} reading "
            f"'{source_name}'"
        ),
        "group_uuid": group_uuid,
        "index": index,
        "frame": frame,
        "sample_count": len(entry.samples),
        "map": _serialize_map(group, entry, index),
    }


@group_handler
def remove_material_map_sample(group_uuid: str, index: int, frame: int):
    """Remove the weight source at a given frame from a material map.

    The map's own source is not a sample and cannot be removed here; change it
    with set_material_map instead.

    Args:
        group_uuid: UUID of the group that owns the map.
        index: Zero-based index of the map, as reported by list_material_maps.
        frame: Frame of the sample to remove, as reported by
            list_material_maps.
    """
    group, entry = _resolve_map(group_uuid, index)
    for i, sample in enumerate(entry.samples):
        if int(sample.frame) != frame:
            continue
        entry.samples.remove(i)
        entry.samples_index = safe_update_index(i, len(entry.samples))
        return {
            "message": (
                f"Removed the '{entry.parameter}' map sample at frame {frame}"
            ),
            "group_uuid": group_uuid,
            "index": index,
            "frame": frame,
            "sample_count": len(entry.samples),
        }
    raise MCPError(
        f"Material map {index} on group '{group.name}' has no sample at frame "
        f"{frame}."
    )
