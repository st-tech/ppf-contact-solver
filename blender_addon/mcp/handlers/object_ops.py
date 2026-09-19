"""Per-assigned-object and per-pin operation handlers.

Covers surfaces that all hang off an AssignedObject or PinVertexGroupItem:
pin settings, pin operations, static ops, velocity keyframes, collision
windows. Identifies groups by UUID and objects by name (name is resolved
to UUID up front so later lookups are rename-safe).
"""

import math

from typing import Optional

import bpy  # pyright: ignore

from ...models.collection_utils import safe_update_index, sort_keyframes_by_frame
from ...models.defaults import MAX_COLLISION_WINDOWS
from ..decorators import (
    MCPError,
    ValidationError,
    group_handler,
    mcp_handler,
)
from .group import (
    get_active_group_by_uuid_helper,
    resolve_assigned_with_index,
    _parse_pin_identifier,
)


# ---------------------------------------------------------------------------
# Shared resolvers
# ---------------------------------------------------------------------------


def _resolve_assigned(group_uuid: str, object_name: str):
    """Return (group, assigned, object_uuid) for an object in a group.

    Raises MCPError if the group is unknown, the object is missing, or the
    object is not a member of the group.
    """
    group, assigned, _, obj_uuid = resolve_assigned_with_index(
        group_uuid, object_name
    )
    return group, assigned, obj_uuid


def _resolve_pin(group, vertex_group_identifier: str):
    """Locate a pin in a group by identifier. Returns (pin_item, obj_uuid, vg_name)."""
    from ...models.groups import decode_vertex_group_identifier
    from ...core.uuid_registry import get_object_uuid

    obj_name, vg_name = _parse_pin_identifier(vertex_group_identifier)
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        raise MCPError(f"Object '{obj_name}' not found in scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(f"Object '{obj_name}' has no UUID")

    for item in group.pin_vertex_groups:
        if item.object_uuid != obj_uuid:
            continue
        _, item_vg = decode_vertex_group_identifier(item.name)
        if item_vg == vg_name:
            return item, obj_uuid, vg_name
    raise MCPError(
        f"Pin '{obj_name}::{vg_name}' not found in group {group.uuid}"
    )


from ._helpers import check_vec3_validation as _check_vec3


# ---------------------------------------------------------------------------
# Pin settings
# ---------------------------------------------------------------------------


def _check_rest_pose_tracking(group, pin, vertex_group_identifier: str):
    """Verify the preconditions the encoder reads before rest-pose tracking.

    ``track_rest_pose_deformation`` reaches the solver only for a SOLID group
    whose pin covers every vertex of its mesh and carries a captured
    deformation (``core/encoder/pin.py`` gates on all three), so enabling it
    without them stores a flag nothing reads. Raises MCPError naming the
    precondition that is unmet.
    """
    from ...core.utils import pin_covers_all_vertices
    from ...core.uuid_registry import resolve_pin
    from ...models.groups import decode_vertex_group_identifier

    if group.object_type != "SOLID":
        raise MCPError(
            "track_rest_pose_deformation requires a SOLID group; group "
            f"{group.uuid} is {group.object_type}"
        )
    try:
        obj = resolve_pin(pin)
    except ValueError as e:
        raise MCPError(str(e)) from e
    if obj is None:
        raise MCPError(
            f"Pin '{vertex_group_identifier}' resolves to no object in the scene"
        )
    _, vg_name = decode_vertex_group_identifier(pin.name)
    is_full = bool(obj.type == "MESH" and pin_covers_all_vertices(obj, vg_name))
    # Coverage is an O(N) vertex scan, so the panel reads it from this cache
    # instead of recomputing it per redraw. Record what was measured, full or
    # not, so the panel shows the same answer the caller gets here.
    pin.full_pin_checked = True
    pin.full_pin_cached = is_full
    if not is_full:
        raise MCPError(
            "track_rest_pose_deformation requires a full pin: "
            f"'{obj.name}::{vg_name}' does not cover every vertex of the mesh"
        )
    if not getattr(pin, "has_captured_anim", False):
        raise MCPError(
            "track_rest_pose_deformation requires a captured deformation on "
            f"pin '{obj.name}::{vg_name}'; run capture_pin_deformation first"
        )


@group_handler
def set_pin_settings(
    group_uuid: str,
    vertex_group_identifier: str,
    included: Optional[bool] = None,
    use_pin_duration: Optional[bool] = None,
    pin_duration: Optional[int] = None,
    use_pull: Optional[bool] = None,
    pull_strength: Optional[float] = None,
    fix_weight_threshold: Optional[float] = None,
    track_rest_pose_deformation: Optional[bool] = None,
    allow_intersection: Optional[bool] = None,
):
    """Set per-pin runtime settings: inclusion, duration, pull, and three
    conditional fields.

    Every argument is optional and an omitted one leaves that field as it is.

    Three of the fields are read only under a condition, and the handler
    refuses a write that would land where nothing reads it:

    - ``fix_weight_threshold`` is SOLID only, and splits a hard pin's diffused
      weights into a hard kinematic shell and a soft-pulled skirt. A non-SOLID
      group is refused. A pull pin ignores it (the pin holds only as hard as
      its own force), so setting it alongside ``use_pull`` is accepted and
      takes effect if the pin is switched off pull later.
    - ``track_rest_pose_deformation`` needs a SOLID group, a pin covering every
      vertex of its mesh, and a captured deformation on that pin. Turning it on
      without all three is refused, naming the one that is missing. Turning it
      off is always accepted.
    - ``allow_intersection`` suppresses the overlap REPORT for geometry this
      pin holds entirely (every corner of a face, both ends of a rod segment).
      Contact, CCD and the line search are unchanged, so an overlap is
      tolerated, not resolved.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
        included: Include this pin in the simulation
        use_pin_duration: Enable per-pin active duration
        pin_duration: Number of frames the pin is active
        use_pull: Use pull force instead of hard constraint
        pull_strength: Pull force strength
        fix_weight_threshold: SOLID hard pin only, 0.0 to 1.0. Tet vertices
            whose diffused pin weight reaches this are held as hard kinematic
            fixes and lower-weight ones stay soft-pulled, so a lower value
            holds more of the pinned region rigidly
        track_rest_pose_deformation: Drive a time-varying rest pose from the
            captured deformation, so the body settles into the captured shape
            instead of straining against it
        allow_intersection: Accept an overlap of the geometry this pin holds
            instead of stopping the simulation
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)

    if fix_weight_threshold is not None:
        if group.object_type != "SOLID":
            raise MCPError(
                "fix_weight_threshold applies to a SOLID group only; group "
                f"{group_uuid} is {group.object_type}"
            )
        # The property clamps to [0, 1] on assignment, which would store a
        # different number than the caller asked for without saying so.
        if not 0.0 <= float(fix_weight_threshold) <= 1.0:
            raise ValidationError(
                "fix_weight_threshold must be between 0.0 and 1.0, got "
                f"{fix_weight_threshold!r}"
            )
    if track_rest_pose_deformation:
        _check_rest_pose_tracking(group, pin, vertex_group_identifier)

    updates = {
        "included": included,
        "use_pin_duration": use_pin_duration,
        "pin_duration": pin_duration,
        "use_pull": use_pull,
        "pull_strength": pull_strength,
        "fix_weight_threshold": fix_weight_threshold,
        "track_rest_pose_deformation": track_rest_pose_deformation,
        "allow_intersection": allow_intersection,
    }
    applied = {}
    for k, v in updates.items():
        if v is None:
            continue
        setattr(pin, k, v)
        applied[k] = v

    return {
        "message": f"Updated {len(applied)} pin settings",
        "group_uuid": group_uuid,
        "object_uuid": obj_uuid,
        "vertex_group_name": vg_name,
        "updates": applied,
    }


# ---------------------------------------------------------------------------
# Pin operations (per pin: Move/Spin/Scale/Torque)
# ---------------------------------------------------------------------------


_PIN_OP_TYPES = {"MOVE_BY", "SPIN", "SCALE", "TORQUE"}


# Lower bound of the RNA property behind each numeric op field (PinOperation
# and StaticOpItem in ui/state_types.py). Blender clamps an assignment below
# the bound without reporting it, so a call that named a smaller number would
# be answered with success while a different one was stored. Every value that
# reaches one of these fields is measured against its bound first.
_OP_FIELD_MINIMUM = {
    "frame_start": 1,
    "frame_end": 1,
    "scale_factor": 0.01,
}


def _check_op_field_minimum(field: str, value) -> None:
    """Refuse a value the field's own RNA range would clamp."""
    minimum = _OP_FIELD_MINIMUM.get(field)
    if minimum is not None and value < minimum:
        raise ValidationError(
            f"{field} must be at least {minimum}: the property clamps to that "
            f"range, so {value!r} would be stored as {minimum} and this call "
            "would report a value it did not keep"
        )


def _check_move_indices(index: int, new_index: int, count: int, label: str) -> None:
    """Validate a reorder request against the list it addresses."""
    if count == 0:
        raise MCPError(f"There is no {label} entry to move")
    upper = count - 1
    for name, value in (("index", index), ("new_index", new_index)):
        if value < 0 or value > upper:
            raise ValidationError(f"{name} {value} out of range (0..{upper})")
    if index == new_index:
        raise ValidationError(
            f"index and new_index are both {index}; a move has to name a "
            "different position"
        )


# Fields carrying a three-component vector, on either op kind.
_VEC3_OP_FIELDS = frozenset({
    "delta",
    "spin_axis",
    "spin_center",
    "spin_center_direction",
    "scale_center",
    "scale_center_direction",
})


def _check_op_enum_identifier(op, field: str, value) -> None:
    """Refuse an identifier the field's own enum does not offer.

    Assigning an identifier an enum does not carry raises a TypeError out of
    Blender, so the value is measured against ``enum_items`` here and the
    assignment is never attempted.
    """
    prop = op.bl_rna.properties.get(field)
    if prop is None or prop.type != "ENUM":
        return
    identifiers = [item.identifier for item in prop.enum_items]
    if value not in identifiers:
        raise ValidationError(
            f"{field} must be one of {identifiers}, got {value!r}"
        )


def _checked_op_fields(op, allowed: set, kwargs: dict, type_label: str) -> dict:
    """Measure every named field against the entry it addresses and return the
    values to write, in the order they were named.

    Nothing is written here. The whole set is checked before the caller stores
    any of it, so a field refused at the end of the set cannot leave the ones
    before it on the entry while the caller is answered with an error.
    """
    checked = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if k not in allowed:
            raise ValidationError(
                f"Field '{k}' is not valid for {type_label}; "
                f"allowed: {sorted(allowed)}"
            )
        _check_op_field_minimum(k, v)
        if k in _VEC3_OP_FIELDS:
            v = _check_vec3(k, v)
        else:
            _check_op_enum_identifier(op, k, v)
        checked[k] = v
    return checked


def _apply_pin_op_fields(op, op_type: str, kwargs: dict):
    """Write op-type-specific fields from kwargs onto the operation.

    Either every named field lands or none does: the values are all checked
    before the first one is stored, so a refused field leaves the operation
    holding exactly what it held before the call.
    """
    common = {"frame_start", "frame_end", "transition", "show_overlay"}
    if op_type == "MOVE_BY":
        allowed = common | {"delta"}
    elif op_type == "SPIN":
        allowed = common | {
            "spin_axis",
            "spin_angular_velocity",
            "spin_flip",
            "spin_center",
            "spin_center_mode",
            "spin_center_vertex",
            "spin_center_direction",
        }
    elif op_type == "SCALE":
        allowed = common | {
            "scale_factor",
            "scale_center",
            "scale_center_mode",
            "scale_center_vertex",
            "scale_center_direction",
        }
    elif op_type == "TORQUE":
        allowed = common | {
            "torque_axis_component",
            "torque_magnitude",
            "torque_flip",
        }
    else:
        raise ValidationError(f"Unknown pin op_type '{op_type}'")

    checked = _checked_op_fields(op, allowed, kwargs, f"op_type={op_type}")
    for k, v in checked.items():
        setattr(op, k, v)


@group_handler
def add_pin_operation(
    group_uuid: str,
    vertex_group_identifier: str,
    op_type: str,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    transition: Optional[str] = None,
    delta: Optional[list[float]] = None,
    spin_axis: Optional[list[float]] = None,
    spin_angular_velocity: Optional[float] = None,
    spin_flip: Optional[bool] = None,
    spin_center: Optional[list[float]] = None,
    spin_center_mode: Optional[str] = None,
    spin_center_vertex: Optional[int] = None,
    spin_center_direction: Optional[list[float]] = None,
    scale_factor: Optional[float] = None,
    scale_center: Optional[list[float]] = None,
    scale_center_mode: Optional[str] = None,
    scale_center_vertex: Optional[int] = None,
    scale_center_direction: Optional[list[float]] = None,
    torque_axis_component: Optional[str] = None,
    torque_magnitude: Optional[float] = None,
    torque_flip: Optional[bool] = None,
):
    """Append an operation to a pin's operation list.

    TORQUE cannot coexist with other op types on the same pin.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
        op_type: One of MOVE_BY, SPIN, SCALE, TORQUE
        frame_start: First frame the op is active
        frame_end: Last frame the op is active
        transition: LINEAR or SMOOTH
        delta: [x, y, z] translation for MOVE_BY (meters)
        spin_axis: [x, y, z] rotation axis for SPIN
        spin_angular_velocity: Degrees per second (SPIN)
        spin_flip: Reverse spin direction
        spin_center: [x, y, z] fixed center for SPIN (ABSOLUTE mode only)
        spin_center_mode: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
        spin_center_vertex: Vertex index for SPIN VERTEX mode
        spin_center_direction: [x, y, z] direction vector for SPIN MAX_TOWARDS mode
        scale_factor: Scale multiplier for SCALE
        scale_center: [x, y, z] fixed center for SCALE (ABSOLUTE mode only)
        scale_center_mode: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
        scale_center_vertex: Vertex index for SCALE VERTEX mode
        scale_center_direction: [x, y, z] direction vector for SCALE MAX_TOWARDS mode
        torque_axis_component: PC1, PC2, or PC3 (principal axis)
        torque_magnitude: Torque in newton-meters
        torque_flip: Reverse torque direction
    """
    if op_type not in _PIN_OP_TYPES:
        raise ValidationError(
            f"Invalid op_type '{op_type}'. Must be one of {sorted(_PIN_OP_TYPES)}"
        )

    group = get_active_group_by_uuid_helper(group_uuid)
    pin, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)

    if any(op.op_type == "EMBEDDED_MOVE" for op in pin.operations):
        raise MCPError(
            "Pin is keyframed; remove its keyframed animation before "
            "adding Move/Spin/Scale/Torque ops"
        )
    existing_types = {o.op_type for o in pin.operations}
    if op_type == "TORQUE" and (existing_types - {"TORQUE"}):
        raise MCPError("Torque cannot be mixed with Move/Spin/Scale ops")
    if op_type != "TORQUE" and "TORQUE" in existing_types:
        raise MCPError("Cannot add Move/Spin/Scale to a pin that has Torque")

    op = pin.operations.add()
    op.op_type = op_type
    if op_type == "SCALE":
        op.scale_center_mode = "CENTROID"
    elif op_type == "SPIN":
        op.spin_center_mode = "CENTROID"
    elif op_type == "TORQUE":
        op.torque_axis_component = "PC3"

    # The entry has to exist before its fields can be written, so a refused
    # field is undone here rather than left as a half-built op the caller was
    # told nothing about.
    try:
        _apply_pin_op_fields(
            op,
            op_type,
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "transition": transition,
                "delta": delta,
                "spin_axis": spin_axis,
                "spin_angular_velocity": spin_angular_velocity,
                "spin_flip": spin_flip,
                "spin_center": spin_center,
                "spin_center_mode": spin_center_mode,
                "spin_center_vertex": spin_center_vertex,
                "spin_center_direction": spin_center_direction,
                "scale_factor": scale_factor,
                "scale_center": scale_center,
                "scale_center_mode": scale_center_mode,
                "scale_center_vertex": scale_center_vertex,
                "scale_center_direction": scale_center_direction,
                "torque_axis_component": torque_axis_component,
                "torque_magnitude": torque_magnitude,
                "torque_flip": torque_flip,
            },
        )
    except Exception:
        pin.operations.remove(len(pin.operations) - 1)
        raise

    new_index = len(pin.operations) - 1
    pin.operations.move(new_index, 0)
    pin.operations_index = 0

    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Added {op_type} op to pin '{vg_name}'",
        "group_uuid": group_uuid,
        "vertex_group_name": vg_name,
        "operation_index": 0,
        "op_type": op_type,
        "operation_count": len(pin.operations),
    }


@group_handler
def remove_pin_operation(
    group_uuid: str, vertex_group_identifier: str, index: int
):
    """Remove a pin operation by index.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
        index: Zero-based index into the pin's operations list
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin, _, vg_name = _resolve_pin(group, vertex_group_identifier)
    if index < 0 or index >= len(pin.operations):
        raise ValidationError(
            f"Index {index} out of range (0..{len(pin.operations) - 1})"
        )
    pin.operations.remove(index)
    pin.operations_index = safe_update_index(index, len(pin.operations))
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Removed pin op index {index}",
        "group_uuid": group_uuid,
        "vertex_group_name": vg_name,
        "operation_count": len(pin.operations),
    }


@group_handler
def set_pin_operation(
    group_uuid: str,
    vertex_group_identifier: str,
    index: int,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    transition: Optional[str] = None,
    delta: Optional[list[float]] = None,
    spin_axis: Optional[list[float]] = None,
    spin_angular_velocity: Optional[float] = None,
    spin_flip: Optional[bool] = None,
    spin_center: Optional[list[float]] = None,
    spin_center_mode: Optional[str] = None,
    spin_center_vertex: Optional[int] = None,
    spin_center_direction: Optional[list[float]] = None,
    scale_factor: Optional[float] = None,
    scale_center: Optional[list[float]] = None,
    scale_center_mode: Optional[str] = None,
    scale_center_vertex: Optional[int] = None,
    scale_center_direction: Optional[list[float]] = None,
    torque_axis_component: Optional[str] = None,
    torque_magnitude: Optional[float] = None,
    torque_flip: Optional[bool] = None,
):
    """Change fields on one operation a pin already carries, addressed by its
    index.

    Every field argument is optional and an omitted one is left as it is, so
    one number can be changed without restating the rest of the entry. Editing
    in place is also what preserves the LIST ORDER: the operations are shipped
    to the solver in list order and compose in that order, while adding one
    puts it at the head, so removing an entry and adding it back to change a
    field moves it to the front and changes the motion the pin performs.

    The op type is fixed when the entry is added. A field belonging to another
    op type is refused rather than written where nothing reads it, so turn a
    MOVE_BY into a SPIN by removing it and adding the SPIN in its place. An
    entry on a keyframed pin holds no editable field and is refused as well.

    Every named field is checked before any of them is written, so a call
    answered with an error leaves the entry holding what it held before.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
        index: Zero-based index into the pin's operations list, in the order
            list_pin_operations reports
        frame_start: First frame the op is active
        frame_end: Last frame the op is active
        transition: LINEAR or SMOOTH
        delta: [x, y, z] translation for MOVE_BY (meters)
        spin_axis: [x, y, z] rotation axis for SPIN
        spin_angular_velocity: Degrees per second (SPIN)
        spin_flip: Reverse spin direction
        spin_center: [x, y, z] fixed center for SPIN (ABSOLUTE mode only)
        spin_center_mode: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
        spin_center_vertex: Vertex index for SPIN VERTEX mode
        spin_center_direction: [x, y, z] direction vector for SPIN MAX_TOWARDS mode
        scale_factor: Scale multiplier for SCALE
        scale_center: [x, y, z] fixed center for SCALE (ABSOLUTE mode only)
        scale_center_mode: CENTROID, ABSOLUTE, MAX_TOWARDS, or VERTEX
        scale_center_vertex: Vertex index for SCALE VERTEX mode
        scale_center_direction: [x, y, z] direction vector for SCALE MAX_TOWARDS mode
        torque_axis_component: PC1, PC2, or PC3 (principal axis)
        torque_magnitude: Torque in newton-meters
        torque_flip: Reverse torque direction
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)
    if index < 0 or index >= len(pin.operations):
        raise ValidationError(
            f"Index {index} out of range (0..{len(pin.operations) - 1})"
        )

    op = pin.operations[index]
    if op.op_type not in _PIN_OP_TYPES:
        raise MCPError(
            f"Operation {index} on pin '{vg_name}' is the keyframed-animation "
            f"marker ({op.op_type}) and carries no field to set. Author that "
            "motion with add_pin_keyframe, or drop it with "
            "delete_pin_keyframes."
        )

    updates = {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "transition": transition,
        "delta": delta,
        "spin_axis": spin_axis,
        "spin_angular_velocity": spin_angular_velocity,
        "spin_flip": spin_flip,
        "spin_center": spin_center,
        "spin_center_mode": spin_center_mode,
        "spin_center_vertex": spin_center_vertex,
        "spin_center_direction": spin_center_direction,
        "scale_factor": scale_factor,
        "scale_center": scale_center,
        "scale_center_mode": scale_center_mode,
        "scale_center_vertex": scale_center_vertex,
        "scale_center_direction": scale_center_direction,
        "torque_axis_component": torque_axis_component,
        "torque_magnitude": torque_magnitude,
        "torque_flip": torque_flip,
    }
    applied = {k: v for k, v in updates.items() if v is not None}
    if not applied:
        raise ValidationError(
            "No field named. Give at least one field valid for op_type="
            f"{op.op_type}, or use remove_pin_operation to drop the entry."
        )
    _apply_pin_op_fields(op, op.op_type, updates)

    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": (
            f"Updated {len(applied)} field(s) on {op.op_type} op {index} of "
            f"pin '{vg_name}'"
        ),
        "group_uuid": group_uuid,
        "object_uuid": obj_uuid,
        "vertex_group_name": vg_name,
        "index": index,
        "op_type": op.op_type,
        "updates": applied,
        "operation_count": len(pin.operations),
    }


@group_handler
def move_pin_operation(
    group_uuid: str,
    vertex_group_identifier: str,
    index: int,
    new_index: int,
):
    """Move one of a pin's operations to another position in its list.

    The order is semantic rather than presentational: the operations are
    shipped to the solver in list order and compose in that order, so moving
    an entry changes the motion the pin performs. Index 0 is the head of the
    list, and the entry that follows composes on top of what precedes it.

    Both positions must address an entry that exists, and they must differ.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
        index: Zero-based index of the operation to move
        new_index: Zero-based position to move it to
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)
    _check_move_indices(index, new_index, len(pin.operations), "pin operation")

    pin.operations.move(index, new_index)
    pin.operations_index = new_index

    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": (
            f"Moved pin op {index} to {new_index} on pin '{vg_name}'"
        ),
        "group_uuid": group_uuid,
        "object_uuid": obj_uuid,
        "vertex_group_name": vg_name,
        "index": new_index,
        "op_type": pin.operations[new_index].op_type,
        "operation_order": [op.op_type for op in pin.operations],
    }


@group_handler
def list_pin_operations(group_uuid: str, vertex_group_identifier: str):
    """List operations attached to a pin.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)
    ops = []
    for op in pin.operations:
        entry = {
            "op_type": op.op_type,
            "frame_start": op.frame_start,
            "frame_end": op.frame_end,
            "transition": op.transition,
        }
        if op.op_type == "MOVE_BY":
            entry["delta"] = list(op.delta)
        elif op.op_type == "SPIN":
            entry.update({
                "spin_axis": list(op.spin_axis),
                "spin_angular_velocity": op.spin_angular_velocity,
                "spin_flip": op.spin_flip,
                "spin_center": list(op.spin_center),
                "spin_center_mode": op.spin_center_mode,
                "spin_center_vertex": op.spin_center_vertex,
                "spin_center_direction": list(op.spin_center_direction),
            })
        elif op.op_type == "SCALE":
            entry.update({
                "scale_factor": op.scale_factor,
                "scale_center": list(op.scale_center),
                "scale_center_mode": op.scale_center_mode,
                "scale_center_vertex": op.scale_center_vertex,
                "scale_center_direction": list(op.scale_center_direction),
            })
        elif op.op_type == "TORQUE":
            entry.update({
                "torque_axis_component": op.torque_axis_component,
                "torque_magnitude": op.torque_magnitude,
                "torque_flip": op.torque_flip,
            })
        ops.append(entry)
    return {
        "group_uuid": group_uuid,
        "vertex_group_name": vg_name,
        "object_uuid": obj_uuid,
        "operations": ops,
    }


@group_handler
def clear_pin_operations(group_uuid: str, vertex_group_identifier: str):
    """Remove every operation from a pin.

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin, _, vg_name = _resolve_pin(group, vertex_group_identifier)
    removed = len(pin.operations)
    pin.operations.clear()
    pin.operations_index = safe_update_index(-1, len(pin.operations))
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Cleared {removed} pin ops",
        "group_uuid": group_uuid,
        "vertex_group_name": vg_name,
        "operation_count": len(pin.operations),
    }


# ---------------------------------------------------------------------------
# Keyframed pin animation (per pin: position keys on the pinned vertices).
# A pin draws its motion from EITHER the parametric operations above OR
# vertex-co keyframes, never both, so each surface refuses a pin the other
# one owns. Both handlers wrap the panel's Make Keyframe / Delete All
# Keyframes operators, which own the fcurve layout and the marker op that
# tells the encoder to splice the per-vertex track in.
# ---------------------------------------------------------------------------


def _resolve_keyframable_pin(group, vertex_group_identifier: str):
    """Return (pin_item, obj, vg_name) for a pin whose mesh can carry keys.

    Both operators report and return without touching anything when the pin
    is not on a mesh, so that case is refused here: a caller is told why
    nothing would happen instead of reading a success off a call that did
    nothing.
    """
    pin_item, _obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)
    obj_name, _ = _parse_pin_identifier(vertex_group_identifier)
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise MCPError(f"Object '{obj_name}' not found in scene")
    if obj.type != "MESH":
        raise MCPError(
            f"Pin '{vertex_group_identifier}' is on a {obj.type.title()} "
            "object. Keyframed pin motion is stored as vertex position curves "
            "on a mesh, so only a mesh pin can carry it."
        )
    return pin_item, obj, vg_name


@group_handler
def add_pin_keyframe(group_uuid: str, vertex_group_identifier: str):
    """Key the positions of a pin's vertices at the scene's current frame.

    The key records the positions the mesh holds right now, at the frame the
    scene is on, so move the timeline and pose the mesh before calling; the
    frame that was keyed comes back in the result. That frame must be 1 or
    later, since a key below frame 1 is dropped when the pin track is read.
    Call it once per pose to build the track. The keys are ordinary Blender
    keyframes on the mesh, set to LINEAR interpolation to match how the solver
    reads a sparse pin track, and the Dope Sheet retimes or deletes them like
    any other key.

    A pin takes its motion from EITHER the parametric operations
    (add_pin_operation) OR keyframes, never both. A pin that already carries
    Move/Spin/Scale/Torque operations is therefore refused here, the mirror of
    add_pin_operation refusing a keyframed pin. A pin holding a captured
    deformation is refused too: the capture wins at encode time, so keys
    written on top of it would never be read.

    Args:
        group_uuid: UUID of the group containing the pin
        vertex_group_identifier: Pin id in 'object::vertex_group' form
    """
    from ...core.encoder.pin import _collect_pin_vertex_fcurve_frames
    from ...core.utils import get_vertices_in_group
    from .group import get_group_index_by_uuid

    group = get_active_group_by_uuid_helper(group_uuid)
    pin_item, obj, vg_name = _resolve_keyframable_pin(
        group, vertex_group_identifier
    )
    vertex_group = obj.vertex_groups.get(vg_name)
    if vertex_group is None:
        raise MCPError(f"Vertex group '{vg_name}' is not on object '{obj.name}'")
    # A key is written per pinned vertex, so an empty pin would leave the
    # operator writing nothing and marking the pin animated all the same.
    if not get_vertices_in_group(obj, vertex_group):
        raise MCPError(
            f"Pin '{vertex_group_identifier}' covers no vertex, so there is "
            "nothing to key. Put vertices in the vertex group first."
        )

    # The two rules the Make Keyframe operator enforces, stated the way it
    # states them so an agent and an artist are told the same thing, each with
    # the tool that clears the blocking state named.
    if any(op.op_type in _PIN_OP_TYPES for op in pin_item.operations):
        raise MCPError(
            "Pin has Move/Spin/Scale/Torque operations; keyframed animation "
            "cannot be combined with them. Drop them with "
            "remove_pin_operation or clear_pin_operations first."
        )
    if getattr(pin_item, "has_captured_anim", False):
        raise MCPError(
            "Pin is captured from the depsgraph; clear the captured "
            "deformation with clear_pin_deformation before adding manual "
            "keyframes."
        )

    scene = bpy.context.scene
    if scene is None:
        raise MCPError("No active Blender scene")
    frame = int(scene.frame_current)
    # The encoder collects a pin's vertex-co keys from frame 1 upward
    # (``_collect_pin_vertex_fcurve_frames``), so a key written below that is
    # never read. Refuse before the operator writes one and marks the pin
    # keyframed.
    if frame < 1:
        raise MCPError(
            f"The scene is on frame {frame}, and a pin key below frame 1 is "
            "dropped when the pin track is read. Move the timeline to frame 1 "
            "or later and call again."
        )

    # The operator reads the pin off the group's selected row, so point the
    # row at this pin the way capture_static_deformation does.
    group.pin_vertex_groups_index = _resolve_pin_index(group, pin_item)
    result = bpy.ops.object.make_pin_keyframe(
        "EXEC_DEFAULT", group_index=get_group_index_by_uuid(group_uuid),
    )
    if "FINISHED" not in result:
        raise MCPError(
            f"object.make_pin_keyframe returned {sorted(result)} for pin "
            f"'{vertex_group_identifier}'; no keyframe was written"
        )

    frames, lookup = _collect_pin_vertex_fcurve_frames(obj, vg_name)
    if frame not in frames:
        raise MCPError(
            f"object.make_pin_keyframe reported success, but frame {frame} "
            f"carries no key on pin '{vertex_group_identifier}'"
        )
    keyed_vertices = len({vertex_index for vertex_index, _axis in lookup})
    return {
        "message": (
            f"Keyed {keyed_vertices} vertices of pin '{vg_name}' at frame "
            f"{frame}"
        ),
        "group_uuid": group_uuid,
        "vertex_group_identifier": vertex_group_identifier,
        "vertex_group_name": vg_name,
        "object_name": obj.name,
        "frame": frame,
        "keyframed_frames": frames,
        "keyframed_vertex_count": keyed_vertices,
        "operator_result": sorted(result),
    }


@group_handler
def delete_pin_keyframes(group_uuid: str, vertex_group_identifier: str):
    """Remove the keyframed motion of a pin, at every frame it was keyed on.

    This deletes the vertex position curves add_pin_keyframe wrote and drops
    the marker that records the pin as keyframed, which is what frees the pin
    to take parametric operations again. There is no per-frame form: the whole
    track goes, so retime or delete single keys in the Dope Sheet instead when
    that is what you want.

    The curves are addressed by mesh, not by pin, so a second pin on the same
    OBJECT loses its keys in the same call.

    A pin holding a captured deformation is refused. The marker op this drops
    is the same one a capture relies on to be encoded at all, so removing it
    would leave the capture stored on the pin and no longer driving it.
    clear_pin_deformation drops the cache and the marker together, which is
    how a captured pin is freed.

    Args:
        group_uuid: UUID of the group containing the pin
        vertex_group_identifier: Pin id in 'object::vertex_group' form
    """
    from ...core.encoder.pin import _collect_pin_vertex_fcurve_frames
    from .group import get_group_index_by_uuid

    group = get_active_group_by_uuid_helper(group_uuid)
    pin_item, obj, vg_name = _resolve_keyframable_pin(
        group, vertex_group_identifier
    )
    # The operator drops every EMBEDDED_MOVE op on the pin, and that op is
    # what makes a capture-only pin emit a cfg at all (``core/encoder/pin.py``
    # skips a non-SOLID pin with no duration, no pull, no ops and no
    # allowance). Nothing re-adds it, so a captured pin is refused here rather
    # than left holding a capture the encoder no longer reads.
    if getattr(pin_item, "has_captured_anim", False):
        raise MCPError(
            f"Pin '{vertex_group_identifier}' holds a captured deformation, "
            "which shares the keyframed-motion marker this would remove. "
            "Use clear_pin_deformation, which drops the captured cache and "
            "the marker together."
        )

    frames_before, _lookup = _collect_pin_vertex_fcurve_frames(obj, vg_name)
    markers_before = sum(
        1 for op in pin_item.operations if op.op_type not in _PIN_OP_TYPES
    )

    group.pin_vertex_groups_index = _resolve_pin_index(group, pin_item)
    result = bpy.ops.object.delete_pin_keyframes(
        "EXEC_DEFAULT", group_index=get_group_index_by_uuid(group_uuid),
    )
    if "FINISHED" not in result:
        raise MCPError(
            f"object.delete_pin_keyframes returned {sorted(result)} for pin "
            f"'{vertex_group_identifier}'; the keyframes are still there"
        )

    frames_after, _lookup_after = _collect_pin_vertex_fcurve_frames(obj, vg_name)
    if frames_after:
        raise MCPError(
            f"object.delete_pin_keyframes reported success, but pin "
            f"'{vertex_group_identifier}' still carries keys at "
            f"{frames_after}"
        )
    markers_after = sum(
        1 for op in pin_item.operations if op.op_type not in _PIN_OP_TYPES
    )
    return {
        "message": (
            f"Cleared the keyframed motion of pin '{vg_name}': "
            f"{len(frames_before)} keyed frame(s) removed"
        ),
        "group_uuid": group_uuid,
        "vertex_group_identifier": vertex_group_identifier,
        "vertex_group_name": vg_name,
        "object_name": obj.name,
        "removed_frames": frames_before,
        "removed_marker_ops": markers_before - markers_after,
        "operation_count": len(pin_item.operations),
        "operator_result": sorted(result),
    }


# ---------------------------------------------------------------------------
# Static ops (per static object: Move/Spin/Scale of the whole object)
# ---------------------------------------------------------------------------


_STATIC_OP_TYPES = {"MOVE_BY", "SPIN", "SCALE"}


def _apply_static_op_fields(op, op_type: str, kwargs: dict):
    """Write op-type-specific fields from kwargs onto the static op.

    Either every named field lands or none does: the values are all checked
    before the first one is stored, so a refused field leaves the op holding
    exactly what it held before the call.
    """
    common = {"frame_start", "frame_end", "transition", "show_overlay"}
    if op_type == "MOVE_BY":
        allowed = common | {"delta"}
    elif op_type == "SPIN":
        allowed = common | {"spin_axis", "spin_angular_velocity"}
    elif op_type == "SCALE":
        allowed = common | {"scale_factor"}
    else:
        raise ValidationError(f"Unknown static op_type '{op_type}'")

    checked = _checked_op_fields(
        op, allowed, kwargs, f"static op_type={op_type}"
    )
    for k, v in checked.items():
        setattr(op, k, v)


@group_handler
def add_static_op(
    group_uuid: str,
    object_name: str,
    op_type: str,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    transition: Optional[str] = None,
    delta: Optional[list[float]] = None,
    spin_axis: Optional[list[float]] = None,
    spin_angular_velocity: Optional[float] = None,
    scale_factor: Optional[float] = None,
):
    """Add a move/spin/scale op to a static-moving object.

    Only valid on groups of type STATIC.

    Args:
        group_uuid: UUID of STATIC group
        object_name: Name of the assigned object
        op_type: One of MOVE_BY, SPIN, SCALE
        frame_start: First frame the op is active
        frame_end: Last frame the op is active
        transition: LINEAR or SMOOTH
        delta: [x, y, z] translation (MOVE_BY)
        spin_axis: [x, y, z] rotation axis (SPIN)
        spin_angular_velocity: Degrees per second (SPIN)
        scale_factor: Scale multiplier (SCALE)
    """
    if op_type not in _STATIC_OP_TYPES:
        raise ValidationError(
            f"Invalid op_type '{op_type}'. Must be one of {sorted(_STATIC_OP_TYPES)}"
        )
    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if group.object_type != "STATIC":
        raise MCPError(
            f"Group {group_uuid} is {group.object_type}; static ops require STATIC"
        )

    op = assigned.static_ops.add()
    op.op_type = op_type
    # The entry has to exist before its fields can be written, so a refused
    # field is undone here rather than left as a half-built op the caller was
    # told nothing about.
    try:
        _apply_static_op_fields(
            op,
            op_type,
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "transition": transition,
                "delta": delta,
                "spin_axis": spin_axis,
                "spin_angular_velocity": spin_angular_velocity,
                "scale_factor": scale_factor,
            },
        )
    except Exception:
        assigned.static_ops.remove(len(assigned.static_ops) - 1)
        raise
    assigned.static_ops.move(len(assigned.static_ops) - 1, 0)
    assigned.static_ops_index = 0
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Added static {op_type} op to '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "op_type": op_type,
        "operation_count": len(assigned.static_ops),
    }


@group_handler
def remove_static_op(group_uuid: str, object_name: str, index: int):
    """Remove a static op by index.

    Args:
        group_uuid: UUID of STATIC group
        object_name: Name of the assigned object
        index: Zero-based index into the object's static_ops list
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if index < 0 or index >= len(assigned.static_ops):
        raise ValidationError(
            f"Index {index} out of range (0..{len(assigned.static_ops) - 1})"
        )
    assigned.static_ops.remove(index)
    assigned.static_ops_index = safe_update_index(index, len(assigned.static_ops))
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Removed static op index {index} from '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "operation_count": len(assigned.static_ops),
    }


@group_handler
def set_static_op(
    group_uuid: str,
    object_name: str,
    index: int,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    transition: Optional[str] = None,
    delta: Optional[list[float]] = None,
    spin_axis: Optional[list[float]] = None,
    spin_angular_velocity: Optional[float] = None,
    scale_factor: Optional[float] = None,
):
    """Change fields on one static op an object already carries, addressed by
    its index.

    Every field argument is optional and an omitted one is left as it is, so
    one number can be changed without restating the rest of the entry. Editing
    in place is also what preserves the LIST ORDER: the ops are shipped to the
    solver in list order and compose in that order, while adding one puts it
    at the head, so removing an entry and adding it back to change a field
    moves it to the front and changes the motion of the object.

    The op type is fixed when the entry is added, and a field belonging to
    another op type is refused rather than written where nothing reads it.
    Every named field is checked before any of them is written, so a call
    answered with an error leaves the entry holding what it held before.

    Args:
        group_uuid: UUID of STATIC group
        object_name: Name of the assigned object
        index: Zero-based index into the object's static_ops list, in the
            order list_static_ops reports
        frame_start: First frame the op is active
        frame_end: Last frame the op is active
        transition: LINEAR or SMOOTH
        delta: [x, y, z] translation (MOVE_BY)
        spin_axis: [x, y, z] rotation axis (SPIN)
        spin_angular_velocity: Degrees per second (SPIN)
        scale_factor: Scale multiplier (SCALE)
    """
    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if group.object_type != "STATIC":
        raise MCPError(
            f"Group {group_uuid} is {group.object_type}; static ops require STATIC"
        )
    if index < 0 or index >= len(assigned.static_ops):
        raise ValidationError(
            f"Index {index} out of range (0..{len(assigned.static_ops) - 1})"
        )

    op = assigned.static_ops[index]
    updates = {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "transition": transition,
        "delta": delta,
        "spin_axis": spin_axis,
        "spin_angular_velocity": spin_angular_velocity,
        "scale_factor": scale_factor,
    }
    applied = {k: v for k, v in updates.items() if v is not None}
    if not applied:
        raise ValidationError(
            "No field named. Give at least one field valid for op_type="
            f"{op.op_type}, or use remove_static_op to drop the entry."
        )
    _apply_static_op_fields(op, op.op_type, updates)

    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": (
            f"Updated {len(applied)} field(s) on {op.op_type} op {index} of "
            f"'{object_name}'"
        ),
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "index": index,
        "op_type": op.op_type,
        "updates": applied,
        "operation_count": len(assigned.static_ops),
    }


@group_handler
def move_static_op(
    group_uuid: str,
    object_name: str,
    index: int,
    new_index: int,
):
    """Move one of an object's static ops to another position in its list.

    The order is semantic rather than presentational: the ops are shipped to
    the solver in list order and compose in that order, so moving an entry
    changes the motion the object performs. Index 0 is the head of the list,
    and the entry that follows composes on top of what precedes it.

    Both positions must address an entry that exists, and they must differ.

    Args:
        group_uuid: UUID of STATIC group
        object_name: Name of the assigned object
        index: Zero-based index of the static op to move
        new_index: Zero-based position to move it to
    """
    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if group.object_type != "STATIC":
        raise MCPError(
            f"Group {group_uuid} is {group.object_type}; static ops require STATIC"
        )
    _check_move_indices(index, new_index, len(assigned.static_ops), "static op")

    assigned.static_ops.move(index, new_index)
    assigned.static_ops_index = new_index

    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Moved static op {index} to {new_index} on '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "index": new_index,
        "op_type": assigned.static_ops[new_index].op_type,
        "operation_order": [op.op_type for op in assigned.static_ops],
    }


@group_handler
def list_static_ops(group_uuid: str, object_name: str):
    """List static ops attached to an assigned object.

    Args:
        group_uuid: UUID of STATIC group
        object_name: Name of the assigned object
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    ops = []
    for op in assigned.static_ops:
        entry = {
            "op_type": op.op_type,
            "frame_start": op.frame_start,
            "frame_end": op.frame_end,
            "transition": op.transition,
        }
        if op.op_type == "MOVE_BY":
            entry["delta"] = list(op.delta)
        elif op.op_type == "SPIN":
            entry["spin_axis"] = list(op.spin_axis)
            entry["spin_angular_velocity"] = op.spin_angular_velocity
        elif op.op_type == "SCALE":
            entry["scale_factor"] = op.scale_factor
        ops.append(entry)
    return {
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "static_ops": ops,
    }


@group_handler
def clear_static_ops(group_uuid: str, object_name: str):
    """Remove all static ops from an assigned object.

    Args:
        group_uuid: UUID of STATIC group
        object_name: Name of the assigned object
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    removed = len(assigned.static_ops)
    assigned.static_ops.clear()
    assigned.static_ops_index = -1
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Cleared {removed} static ops on '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
    }


# ---------------------------------------------------------------------------
# PDRD hinge joint (per object)
# ---------------------------------------------------------------------------


@group_handler
def set_pdrd_hinge(
    group_uuid: str,
    object_name: str,
    enable: bool = True,
    pca_axis: int = 2,
):
    """Pin a PDRD body as a hinge (per object).

    Locks the body's position and restricts its rotation to one principal
    (PCA) axis of its rest shape, the building block for gears. The group must
    be of type PDRD. Per-object, so each body in a group can be hinged on its
    own axle.

    Args:
        group_uuid: UUID of the PDRD group
        object_name: Name of the assigned object
        enable: Pin the body (True) or release it so it moves freely (False)
        pca_axis: Free axle: 0 (largest extent), 1 (middle), 2 (thinnest, the
            usual axle for a flat gear or disk)
    """
    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if group.object_type != "PDRD":
        raise ValidationError("set_pdrd_hinge requires a PDRD group")
    if int(pca_axis) not in (0, 1, 2):
        raise ValidationError("pca_axis must be 0, 1 or 2")
    assigned.pdrd_hinge_enable = bool(enable)
    assigned.pdrd_hinge_axis = str(int(pca_axis))
    return {
        "message": (
            f"{'Hinged' if enable else 'Released'} '{object_name}'"
            f" (axle PC{int(pca_axis) + 1})"
        ),
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "pdrd_hinge_enable": bool(enable),
        "pdrd_hinge_axis": str(int(pca_axis)),
    }


# ---------------------------------------------------------------------------
# Lock Translation / Lock Rotation (per object)
# ---------------------------------------------------------------------------


# (label, enable field, all-axes field, axis field) for the two locks. The
# all-axes field is the MODE: with it set the lock covers all three axes and
# the axis is not encoded at all, so the axis matters only for the per-axis
# mode and "axis != 0" is never the test for whether a lock is on.
_LOCK_TRIPLES = (
    (
        "Lock Translation",
        "lock_translation_enable",
        "lock_translation_all",
        "lock_translation_axis",
    ),
    (
        "Lock Rotation",
        "lock_rotation_enable",
        "lock_rotation_all",
        "lock_rotation_axis",
    ),
)


def _check_lock_axis(label: str, axis_param: str, all_param: str, axis) -> None:
    """Refuse an axis a per-axis lock cannot use.

    Called only for a lock whose resulting state is enabled and not all-axes,
    the one combination whose axis reaches the solver. The encoder normalizes
    the direction and raises on a zero-length or non-finite one, so refuse it
    at the call that would store it.
    """
    if not all(math.isfinite(c) for c in axis):
        raise MCPError(f"{label}: {axis_param} must be finite, got {axis!r}")
    if axis == (0.0, 0.0, 0.0):
        raise MCPError(
            f"{label}: axis is zero; scene build will fail until it is "
            f"non-zero. Pass a non-zero {axis_param}, or set {all_param}, "
            "which locks every axis and reads no axis at all."
        )


@group_handler
def set_object_locks(
    group_uuid: str,
    object_name: str,
    lock_translation_enable: Optional[bool] = None,
    lock_translation_all: Optional[bool] = None,
    lock_translation_axis: Optional[list[float]] = None,
    lock_rotation_enable: Optional[bool] = None,
    lock_rotation_all: Optional[bool] = None,
    lock_rotation_axis: Optional[list[float]] = None,
    lock_rotation_prohibit_axis: Optional[bool] = None,
):
    """Lock an object's rigid translation, its rigid rotation, or both.

    Lock Translation constrains the object's mass-weighted center of mass to a
    fixed world-space line through its initial position; Lock Rotation
    restricts its mass-weighted best-fit rigid rotation to a fixed world-space
    axis. Deformation stays free under either, and the two are independent
    booleans on the same object: either, both or neither may be enabled. Both
    are exact constraints on the Newton direction rather than penalty springs,
    so there is no stiffness to tune.

    Per object, and available on the dynamic group types (SOLID, SHELL, ROD,
    PDRD, SAND). A STATIC group is refused, since the encoder ships no lock for
    one. A lock also reaches the solver only for an object that is included in
    its group.

    Every argument is optional and an omitted one leaves that field as it is.
    The MODE carries the enable bit, not the axis: lock_translation_all and
    lock_rotation_all saturate their lock to all three axes and stop the axis
    being read, so a zero axis is correct under either. For the per-axis mode
    the axis must be non-zero and finite, and a call that would leave an
    enabled per-axis lock with a zero axis is refused. That is decided on the
    state the call results in, so an axis and its mode can be set together in
    one call in either order.

    Args:
        group_uuid: UUID of the group containing the object
        object_name: Name of the assigned object in the group
        lock_translation_enable: Constrain the center of mass (True) or let it
            move freely (False)
        lock_translation_all: Pin the center of mass to its initial point
            instead of letting it slide along the translation axis
        lock_translation_axis: World-space direction [x, y, z] of the line the
            center of mass may move along. Direction only, normalized by the
            encoder
        lock_rotation_enable: Restrict the best-fit rigid rotation (True) or
            leave it free (False)
        lock_rotation_all: Forbid net rotation about every axis instead of
            about the rotation axis alone
        lock_rotation_axis: World-space rotation axis [x, y, z]. Direction
            only, normalized by the encoder
        lock_rotation_prohibit_axis: False: rotation about the rotation axis is
            the object's only rotational freedom. True: rotation about that
            axis is the one thing forbidden, and the perpendicular plane stays
            free
    """
    from ...models.groups import DYNAMIC_OBJECT_TYPES

    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if group.object_type not in DYNAMIC_OBJECT_TYPES:
        raise MCPError(
            "set_object_locks requires a dynamic group "
            f"({', '.join(DYNAMIC_OBJECT_TYPES)}); group {group_uuid} is "
            f"{group.object_type}"
        )

    updates = {
        "lock_translation_enable": lock_translation_enable,
        "lock_translation_all": lock_translation_all,
        "lock_rotation_enable": lock_rotation_enable,
        "lock_rotation_all": lock_rotation_all,
        "lock_rotation_prohibit_axis": lock_rotation_prohibit_axis,
    }
    if lock_translation_axis is not None:
        updates["lock_translation_axis"] = _check_vec3(
            "lock_translation_axis", lock_translation_axis
        )
    if lock_rotation_axis is not None:
        updates["lock_rotation_axis"] = _check_vec3(
            "lock_rotation_axis", lock_rotation_axis
        )

    def _resulting(field):
        """The value this field will hold once the call is applied."""
        value = updates.get(field)
        return getattr(assigned, field) if value is None else value

    # Validate the state the call results in, not its arguments alone: the
    # enable bit, the mode and the axis of one lock can each come from the
    # call or from what the object already carries.
    for label, enable_field, all_field, axis_field in _LOCK_TRIPLES:
        if not _resulting(enable_field) or _resulting(all_field):
            continue
        _check_lock_axis(
            label, axis_field, all_field, tuple(_resulting(axis_field))
        )

    applied = {}
    for field, value in updates.items():
        if value is None:
            continue
        setattr(assigned, field, value)
        applied[field] = list(value) if isinstance(value, tuple) else value

    return {
        "message": f"Updated {len(applied)} lock field(s) on '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "updates": applied,
        "locks": {
            "lock_translation_enable": bool(assigned.lock_translation_enable),
            "lock_translation_all": bool(assigned.lock_translation_all),
            "lock_translation_axis": [
                float(c) for c in assigned.lock_translation_axis
            ],
            "lock_rotation_enable": bool(assigned.lock_rotation_enable),
            "lock_rotation_all": bool(assigned.lock_rotation_all),
            "lock_rotation_axis": [float(c) for c in assigned.lock_rotation_axis],
            "lock_rotation_prohibit_axis": bool(
                assigned.lock_rotation_prohibit_axis
            ),
        },
    }


# ---------------------------------------------------------------------------
# Bending reference rest angle (per object; SHELL and ROD). The group flag
# bend_rest_from_reference switches the group's bending rest angle over to
# reference geometry, and the reference object itself lives on the assigned
# object, so every shell or rod in the group points at its own copy.
# ---------------------------------------------------------------------------


@group_handler
def set_bend_reference(
    group_uuid: str,
    object_name: str,
    reference_object_name: str,
    enable: Optional[bool] = None,
):
    """Point one assigned object's bending rest angle at a reference object.

    A reference is a topological COPY of the object whose vertices were moved:
    the same vertex count and the same connectivity (faces for a SHELL, edges
    for a ROD), with only positions differing. Its modifiers and geometry
    nodes are evaluated before the comparison, so a copy shaped by a modifier
    is a valid reference. A curve rod is compared at control-point level
    instead, which is how a curve rod is shipped, and a curve modifier is not
    sampled there. Anything that fails the comparison is refused here, naming
    the mismatch, rather than at scene build.

    The group's own bend_rest_from_reference flag is what makes the group read
    a reference at all, so it has to be on before a reference can be set; turn
    it on with set_group_material_properties. Only SHELL and ROD groups carry
    that flag.

    Pass an empty reference_object_name to clear the reference, which also
    stops this object reading one. Clearing is accepted whatever the group
    flag holds, so a stale reference can always be taken off.

    Args:
        group_uuid: UUID of the SHELL or ROD group
        object_name: Name of the assigned object whose rest angle comes from
            the reference
        reference_object_name: Name of the reference object, or "" to clear
            the reference this object holds
        enable: Whether this object reads its reference. Defaults to True when
            a reference is given and False when one is cleared, so it is worth
            naming only to record a reference without using it yet
    """
    from ...core.utils import validate_bend_reference
    from ...core.uuid_registry import get_or_create_object_uuid

    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    reference_object_name = (reference_object_name or "").strip()

    if not reference_object_name:
        if enable:
            raise ValidationError(
                "enable=True needs a reference object: name one in "
                "reference_object_name, or leave enable out to clear"
            )
        assigned.bend_ref_uuid = ""
        assigned.bend_ref_name = ""
        assigned.bend_ref_enable = False
        return {
            "message": f"Cleared the bending reference of '{object_name}'",
            "group_uuid": group_uuid,
            "object_name": object_name,
            "object_uuid": obj_uuid,
            "reference_object_name": "",
            "bend_ref_enable": False,
        }

    if group.object_type not in ("SHELL", "ROD"):
        raise MCPError(
            "A bending reference applies to a SHELL or ROD group; group "
            f"{group_uuid} is {group.object_type}"
        )
    if not group.bend_rest_from_reference:
        raise MCPError(
            f"Group {group_uuid} does not take its bending rest angle from "
            f"reference geometry, so a reference on '{object_name}' would sit "
            "where nothing reads it. Set bend_rest_from_reference on the "
            "group with set_group_material_properties first."
        )

    source_obj = bpy.data.objects.get(object_name)
    ref_obj = bpy.data.objects.get(reference_object_name)
    if ref_obj is None:
        raise MCPError(
            f"Reference object '{reference_object_name}' not found in scene"
        )
    # The reference is stored by UUID so it survives a rename, and a
    # library-linked object cannot be given one.
    ref_uuid = get_or_create_object_uuid(ref_obj)
    if not ref_uuid:
        raise MCPError(
            f"Reference object '{ref_obj.name}' is not writable "
            "(library-linked), so it cannot be recorded as a reference"
        )
    ok, message = validate_bend_reference(
        source_obj, ref_obj, bpy.context, group.object_type
    )
    if not ok:
        raise MCPError(message)

    assigned.bend_ref_uuid = ref_uuid
    assigned.bend_ref_name = ref_obj.name
    assigned.bend_ref_enable = True if enable is None else bool(enable)
    return {
        "message": (
            f"Bending rest angle of '{object_name}' now comes from "
            f"'{ref_obj.name}'"
        ),
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "reference_object_name": ref_obj.name,
        "reference_object_uuid": ref_uuid,
        "bend_ref_enable": bool(assigned.bend_ref_enable),
    }


# ---------------------------------------------------------------------------
# Velocity keyframes (initial velocity driven by keyframed vectors)
# ---------------------------------------------------------------------------


# Axis identifiers VelocityKeyframe.angular_axis accepts (ui/state_types.py),
# plus the integer spelling of the three principal axes the tools take as a
# convenience.
_ANGULAR_AXES = ("PC1", "PC2", "PC3", "X", "Y", "Z", "CUSTOM")
_ANGULAR_AXIS_BY_INDEX = {0: "PC1", 1: "PC2", 2: "PC3"}


def _resolve_angular_axis(angular_axis) -> str:
    """Return the enum identifier for a caller's angular-axis argument."""
    if isinstance(angular_axis, bool):
        raise ValidationError(
            "angular_axis must be PC1/PC2/PC3, X/Y/Z, CUSTOM, or 0/1/2, got "
            f"{angular_axis!r}"
        )
    if isinstance(angular_axis, int):
        if angular_axis not in _ANGULAR_AXIS_BY_INDEX:
            raise ValidationError(
                f"angular_axis {angular_axis} names no principal axis; use "
                "0, 1 or 2, or a PC1/PC2/PC3, X/Y/Z, CUSTOM identifier"
            )
        return _ANGULAR_AXIS_BY_INDEX[angular_axis]
    identifier = str(angular_axis).upper()
    if identifier not in _ANGULAR_AXES:
        raise ValidationError(
            "angular_axis must be PC1/PC2/PC3, X/Y/Z, CUSTOM, or 0/1/2"
        )
    return identifier


def _check_velocity_speed(speed) -> float:
    """Refuse a speed the RNA property's own range would clamp.

    ``VelocityKeyframe.speed`` carries ``min=0.0``, and the direction vector
    is what carries the sign, so a negative magnitude would be stored as zero
    and the keyframe would silently stop the object instead.
    """
    value = float(speed)
    if value < 0.0:
        raise ValidationError(
            f"speed must be zero or greater: the property clamps at 0.0, so "
            f"{speed!r} would be stored as 0.0. Point 'direction' the other "
            "way to reverse the motion."
        )
    return value


def _restore_frame_order(keyframes, index: int) -> int:
    """Slide the keyframe at *index* back into frame order and report where it
    landed.

    The collection is held in frame order, and editing a frame in place is the
    one operation that can break that in either direction, so the entry is
    walked both ways rather than only toward the head.
    """
    position = index
    while position > 0 and (
        keyframes[position].frame < keyframes[position - 1].frame
    ):
        keyframes.move(position, position - 1)
        position -= 1
    while position + 1 < len(keyframes) and (
        keyframes[position].frame > keyframes[position + 1].frame
    ):
        keyframes.move(position, position + 1)
        position += 1
    return position


@group_handler
def add_velocity_keyframe(
    group_uuid: str,
    object_name: str,
    frame: int,
    direction: list[float],
    speed: float,
    angular_axis: "int | str" = "PC3",
    angular_speed: float = 0.0,
    angular_axis_custom: list[float] | None = None,
    enable_translational: bool = True,
    enable_angular: bool | None = None,
):
    """Add a velocity keyframe at the given frame for an assigned object.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
        frame: Blender frame number (>= 1)
        direction: [x, y, z] direction vector (normalized at runtime)
        speed: Velocity magnitude (m/s)
        angular_axis: Axis to spin about (solid/shell/PDRD). One of
            "PC1"/"PC2"/"PC3" (principal axes, resolved dynamically from the
            geometry), "X"/"Y"/"Z" (fixed world axes), or "CUSTOM" (the
            angular_axis_custom vector). Ints 0/1/2 map to PC1/PC2/PC3.
            Ignored when angular_speed == 0.
        angular_speed: Signed spin speed in degrees per second (0 = no spin).
        angular_axis_custom: World [x, y, z] axis used when angular_axis ==
            "CUSTOM" (normalized before use). Defaults to [0, 0, 1].
        enable_translational: Overwrite the translational velocity at this
            frame (False = leave translation alone, e.g. a pure spin).
        enable_angular: Overwrite the angular velocity at this frame. Defaults
            to True when angular_speed is non-zero, else False.
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if frame < 1:
        raise ValidationError("frame must be >= 1")
    direction = _check_vec3("direction", direction)
    speed = _check_velocity_speed(speed)
    angular_axis = _resolve_angular_axis(angular_axis)
    if angular_axis_custom is None:
        angular_axis_custom = [0.0, 0.0, 1.0]
    angular_axis_custom = _check_vec3("angular_axis_custom", angular_axis_custom)
    if enable_angular is None:
        enable_angular = float(angular_speed) != 0.0

    for kf in assigned.velocity_keyframes:
        if kf.frame == frame:
            raise MCPError(f"Frame {frame} already has a velocity keyframe")

    kf = assigned.velocity_keyframes.add()
    kf.frame = frame
    kf.direction = direction
    kf.speed = float(speed)
    kf.angular_axis = angular_axis
    kf.angular_speed = float(angular_speed)
    kf.angular_axis_custom = tuple(float(c) for c in angular_axis_custom)
    kf.enable_translational = bool(enable_translational)
    kf.enable_angular = bool(enable_angular)
    assigned.velocity_keyframes_index = sort_keyframes_by_frame(
        assigned.velocity_keyframes
    )
    return {
        "message": f"Added velocity keyframe at frame {frame} for '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "frame": frame,
        "keyframe_count": len(assigned.velocity_keyframes),
    }


@group_handler
def remove_velocity_keyframe(group_uuid: str, object_name: str, frame: int):
    """Remove the velocity keyframe at the given frame.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
        frame: Frame number of the keyframe to remove
    """
    _, assigned, _ = _resolve_assigned(group_uuid, object_name)
    for i, kf in enumerate(assigned.velocity_keyframes):
        if kf.frame == frame:
            assigned.velocity_keyframes.remove(i)
            assigned.velocity_keyframes_index = safe_update_index(
                i, len(assigned.velocity_keyframes)
            )
            from ...models.groups import invalidate_overlays
            invalidate_overlays()
            return {
                "message": f"Removed velocity keyframe at frame {frame}",
                "group_uuid": group_uuid,
                "object_name": object_name,
                "keyframe_count": len(assigned.velocity_keyframes),
            }
    raise MCPError(f"No velocity keyframe at frame {frame} for '{object_name}'")


@group_handler
def set_velocity_keyframe(
    group_uuid: str,
    object_name: str,
    index: int,
    frame: Optional[int] = None,
    direction: Optional[list[float]] = None,
    speed: Optional[float] = None,
    angular_axis: "int | str | None" = None,
    angular_speed: Optional[float] = None,
    angular_axis_custom: Optional[list[float]] = None,
    enable_translational: Optional[bool] = None,
    enable_angular: Optional[bool] = None,
):
    """Change fields on one velocity keyframe an object already carries,
    addressed by its index.

    Every field argument is optional and an omitted one is left as it is, so a
    keyframe's speed can be changed without restating its direction and its
    two enable gates.

    The frame may be changed as well, which retimes the keyframe in place. The
    list is held in frame order, so the entry can land at a different index,
    and the index it ends up at comes back as new_index. A frame another
    keyframe on the same object already occupies is refused, since a frame
    carries at most one velocity keyframe.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
        index: Zero-based index into the object's velocity keyframe list, in
            the frame order list_velocity_keyframes reports
        frame: Blender frame number (>= 1) to retime this keyframe to
        direction: [x, y, z] direction vector (normalized at runtime)
        speed: Velocity magnitude in m/s, zero or greater
        angular_axis: Axis to spin about (solid/shell/PDRD). One of
            "PC1"/"PC2"/"PC3" (principal axes, resolved dynamically from the
            geometry), "X"/"Y"/"Z" (fixed world axes), or "CUSTOM" (the
            angular_axis_custom vector). Ints 0/1/2 map to PC1/PC2/PC3
        angular_speed: Signed spin speed in degrees per second (0 = no spin)
        angular_axis_custom: World [x, y, z] axis used when angular_axis is
            "CUSTOM" (normalized before use)
        enable_translational: Overwrite the translational velocity at this
            frame (False leaves translation alone, for a pure spin)
        enable_angular: Overwrite the angular velocity at this frame
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    keyframes = assigned.velocity_keyframes
    if index < 0 or index >= len(keyframes):
        raise ValidationError(
            f"Index {index} out of range (0..{len(keyframes) - 1})"
        )

    updates: dict = {}
    if frame is not None:
        if frame < 1:
            raise ValidationError("frame must be >= 1")
        for position, other in enumerate(keyframes):
            if position != index and other.frame == frame:
                raise MCPError(
                    f"Frame {frame} already has a velocity keyframe on "
                    f"'{object_name}'"
                )
        updates["frame"] = int(frame)
    if direction is not None:
        updates["direction"] = _check_vec3("direction", direction)
    if speed is not None:
        updates["speed"] = _check_velocity_speed(speed)
    if angular_axis is not None:
        updates["angular_axis"] = _resolve_angular_axis(angular_axis)
    if angular_speed is not None:
        updates["angular_speed"] = float(angular_speed)
    if angular_axis_custom is not None:
        updates["angular_axis_custom"] = _check_vec3(
            "angular_axis_custom", angular_axis_custom
        )
    if enable_translational is not None:
        updates["enable_translational"] = bool(enable_translational)
    if enable_angular is not None:
        updates["enable_angular"] = bool(enable_angular)
    if not updates:
        raise ValidationError(
            "No field named. Give at least one of frame, direction, speed, "
            "angular_axis, angular_speed, angular_axis_custom, "
            "enable_translational or enable_angular."
        )

    keyframe = keyframes[index]
    for field, value in updates.items():
        setattr(keyframe, field, value)
    # A collection item reference addresses a SLOT, not the entry that was in
    # it: the reorder below moves the entries, leaving this reference on
    # whichever keyframe ends up at *index*. Read the frame out while the
    # reference still names this one.
    stored_frame = int(keyframe.frame)
    new_index = _restore_frame_order(keyframes, index)
    assigned.velocity_keyframes_index = new_index

    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": (
            f"Updated {len(updates)} field(s) on velocity keyframe {index} of "
            f"'{object_name}' (frame {stored_frame})"
        ),
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "index": index,
        "new_index": new_index,
        "frame": stored_frame,
        "updates": {
            field: list(value) if isinstance(value, tuple) else value
            for field, value in updates.items()
        },
        "keyframe_count": len(keyframes),
    }


@group_handler
def list_velocity_keyframes(group_uuid: str, object_name: str):
    """List velocity keyframes for an assigned object.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    keyframes = [
        {
            "frame": kf.frame,
            "direction": list(kf.direction),
            "speed": kf.speed,
        }
        for kf in assigned.velocity_keyframes
    ]
    return {
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "keyframes": keyframes,
    }


@group_handler
def clear_velocity_keyframes(group_uuid: str, object_name: str):
    """Clear all velocity keyframes on an assigned object.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
    """
    _, assigned, _ = _resolve_assigned(group_uuid, object_name)
    removed = len(assigned.velocity_keyframes)
    assigned.velocity_keyframes.clear()
    assigned.velocity_keyframes_index = safe_update_index(
        -1, len(assigned.velocity_keyframes)
    )
    from ...models.groups import invalidate_overlays
    invalidate_overlays()
    return {
        "message": f"Cleared {removed} velocity keyframes on '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
    }


# ---------------------------------------------------------------------------
# Collision windows (per-object intervals where contact is active)
# ---------------------------------------------------------------------------


@group_handler
def set_use_collision_windows(group_uuid: str, enable: bool):
    """Toggle the per-object collision-window feature for a group.

    Args:
        group_uuid: UUID of group
        enable: True to enable, False to disable
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    group.use_collision_windows = bool(enable)
    return {
        "message": f"Set use_collision_windows={enable} on group {group_uuid}",
        "group_uuid": group_uuid,
        "use_collision_windows": bool(enable),
    }


@group_handler
def add_collision_window(
    group_uuid: str, object_name: str, frame_start: int, frame_end: int
):
    """Add a collision-active window on an assigned object.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
        frame_start: First frame of the window
        frame_end: Last frame of the window
    """
    if frame_start < 1 or frame_end < 1:
        raise ValidationError("frame_start and frame_end must be >= 1")
    if frame_end < frame_start:
        raise ValidationError("frame_end must be >= frame_start")
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if len(assigned.collision_windows) >= MAX_COLLISION_WINDOWS:
        raise MCPError(
            f"Maximum {MAX_COLLISION_WINDOWS} collision windows per object"
        )
    item = assigned.collision_windows.add()
    item.frame_start = frame_start
    item.frame_end = frame_end
    assigned.collision_windows_index = len(assigned.collision_windows) - 1
    return {
        "message": f"Added collision window [{frame_start}-{frame_end}] for '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "window_count": len(assigned.collision_windows),
    }


@group_handler
def remove_collision_window(group_uuid: str, object_name: str, index: int):
    """Remove a collision window by index.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
        index: Zero-based index into the object's collision_windows list
    """
    _, assigned, _ = _resolve_assigned(group_uuid, object_name)
    if index < 0 or index >= len(assigned.collision_windows):
        raise ValidationError(
            f"Index {index} out of range (0..{len(assigned.collision_windows) - 1})"
        )
    assigned.collision_windows.remove(index)
    assigned.collision_windows_index = safe_update_index(
        index, len(assigned.collision_windows)
    )
    return {
        "message": f"Removed collision window index {index}",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "window_count": len(assigned.collision_windows),
    }


@group_handler
def set_collision_window(
    group_uuid: str,
    object_name: str,
    index: int,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
):
    """Change the bounds of one collision window an object already carries,
    addressed by its index.

    Either bound may be given on its own and the other is left as it is. The
    window that results is what gets validated, so moving frame_start past the
    frame_end already stored is refused instead of being kept as an inverted
    window that turns contact off for the whole run.

    Editing in place also keeps the window at its index, which is how
    list_collision_windows and remove_collision_window address it.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
        index: Zero-based index into the object's collision_windows list
        frame_start: First frame of the window (>= 1)
        frame_end: Last frame of the window (>= frame_start)
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    windows = assigned.collision_windows
    if index < 0 or index >= len(windows):
        raise ValidationError(
            f"Index {index} out of range (0..{len(windows) - 1})"
        )
    if frame_start is None and frame_end is None:
        raise ValidationError(
            "No field named. Give frame_start, frame_end, or both."
        )

    window = windows[index]
    start = int(window.frame_start if frame_start is None else frame_start)
    end = int(window.frame_end if frame_end is None else frame_end)
    if start < 1 or end < 1:
        raise ValidationError("frame_start and frame_end must be >= 1")
    if end < start:
        raise ValidationError(
            f"frame_end must be >= frame_start; this call would leave the "
            f"window [{start}-{end}]"
        )

    window.frame_start = start
    window.frame_end = end
    return {
        "message": (
            f"Set collision window {index} on '{object_name}' to "
            f"[{start}-{end}]"
        ),
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "index": index,
        "frame_start": start,
        "frame_end": end,
        "window_count": len(windows),
    }


@group_handler
def list_collision_windows(group_uuid: str, object_name: str):
    """List collision windows on an assigned object.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    windows = [
        {"frame_start": w.frame_start, "frame_end": w.frame_end}
        for w in assigned.collision_windows
    ]
    return {
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "windows": windows,
    }


@group_handler
def clear_collision_windows(group_uuid: str, object_name: str):
    """Clear every collision window on an assigned object.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
    """
    _, assigned, _ = _resolve_assigned(group_uuid, object_name)
    removed = len(assigned.collision_windows)
    assigned.collision_windows.clear()
    assigned.collision_windows_index = safe_update_index(
        -1, len(assigned.collision_windows)
    )
    return {
        "message": f"Cleared {removed} collision windows on '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
    }


# ---------------------------------------------------------------------------
# Static-deformation capture (Armature / Lattice / Shape Key driven STATIC
# colliders). Mirrors the UI's Capture Deformation / Clear Deformation Cache
# buttons. The capture is modal and runs after this handler returns; poll
# get_static_deformation_status to detect completion.
# ---------------------------------------------------------------------------


@group_handler
def capture_static_deformation(group_uuid: str, object_name: str):
    """Record the per-frame shape of an animated STATIC mesh onto the collider.

    Use this for STATIC objects whose vertices move because of an Armature
    modifier, a Lattice or Mesh Deform cage, animated Shape Keys, or a
    driver that pokes vertex coordinates. The recording runs as a modal
    operator and continues after this call returns; poll
    ``get_static_deformation_status`` to detect completion.

    Press again any time the underlying animation changes (a new pose,
    edited action keyframes, a modifier swap). The recording does NOT
    update on its own.

    Args:
        group_uuid: UUID of STATIC group containing the object
        object_name: Name of the assigned mesh to capture
    """
    from .group import get_group_index_by_uuid
    group, _, idx, obj_uuid = resolve_assigned_with_index(group_uuid, object_name)
    if group.object_type != "STATIC":
        raise MCPError(
            f"Group {group_uuid} is {group.object_type}; capture requires STATIC"
        )

    obj = bpy.data.objects.get(object_name)
    from ...core.utils import is_deforming_static_object
    if not is_deforming_static_object(obj, bpy.context):
        raise MCPError(
            f"Object '{object_name}' has no deforming modifier or shape-key "
            "animation. Capture Deformation only applies to STATIC meshes "
            "whose vertices move (Armature, Lattice, Mesh Deform, Shape Keys, "
            "or drivers)."
        )

    # Select the assigned-object row in the UI so the modal operator
    # picks up the right object via group.assigned_objects_index.
    group.assigned_objects_index = idx

    group_index = get_group_index_by_uuid(group_uuid)
    bpy.ops.object.capture_static_deformation(
        "EXEC_DEFAULT", group_index=group_index,
    )
    return {
        "message": (
            f"Started Capture Deformation for '{object_name}'. The recording "
            "runs in the background; poll get_static_deformation_status until "
            "it reports a non-zero frame_count."
        ),
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
    }


@group_handler
def clear_static_deformation(group_uuid: str, object_name: str):
    """Discard the recorded deformation cache for one STATIC object.

    The object returns to the pre-capture state: Capture Deformation
    becomes the only enabled button on the row, and the next Transfer
    will refuse to upload the object until a fresh capture is taken.

    Args:
        group_uuid: UUID of STATIC group containing the object
        object_name: Name of the assigned mesh
    """
    from .group import get_group_index_by_uuid
    group, _, idx, obj_uuid = resolve_assigned_with_index(group_uuid, object_name)
    if group.object_type != "STATIC":
        raise MCPError(
            f"Group {group_uuid} is {group.object_type}; clear requires STATIC"
        )

    group.assigned_objects_index = idx

    group_index = get_group_index_by_uuid(group_uuid)
    bpy.ops.object.clear_static_deformation(
        "EXEC_DEFAULT", group_index=group_index,
    )
    return {
        "message": f"Cleared deformation cache for '{object_name}'",
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
    }


@group_handler
def get_static_deformation_status(group_uuid: str, object_name: str):
    """Report the deformation-capture state of one STATIC object.

    Returns three fields:

    - ``is_deforming``: True if the object's modifier stack or shape-key
      animation actually moves vertices over the timeline. When False,
      Capture Deformation is not needed and the button is grayed out.
    - ``has_cache``: True if a deformation cache exists for the object.
    - ``frame_count``: Number of frames in the cache, or 0 when absent.

    Args:
        group_uuid: UUID of STATIC group containing the object
        object_name: Name of the assigned mesh
    """
    group, _, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if group.object_type != "STATIC":
        raise MCPError(
            f"Group {group_uuid} is {group.object_type}; status requires STATIC"
        )
    obj = bpy.data.objects.get(object_name)
    from ...core.utils import is_deforming_static_object
    from ...ui.dynamics.static_deform_ops import (
        object_deformation_frame_count,
        object_has_deformation_cache,
    )
    return {
        "group_uuid": group_uuid,
        "object_name": object_name,
        "object_uuid": obj_uuid,
        "is_deforming": bool(is_deforming_static_object(obj, bpy.context)),
        "has_cache": bool(object_has_deformation_cache(obj)),
        "frame_count": int(object_deformation_frame_count(obj)),
    }


# ---------------------------------------------------------------------------
# Pin-deformation capture (Armature / Lattice / Mesh Deform / Shape Key
# driven SHELL / SOLID / ROD pins). Mirrors the UI's Capture Deformation /
# Clear Deformation Cache buttons on the pin details panel. The capture is
# modal and runs after this handler returns; poll get_pin_deformation_status
# to detect completion.
# ---------------------------------------------------------------------------


def _resolve_pin_index(group, pin_item) -> int:
    """Return the index of ``pin_item`` in its group's pin collection."""
    for i, item in enumerate(group.pin_vertex_groups):
        if item.as_pointer() == pin_item.as_pointer():
            return i
    raise MCPError(
        f"Pin item not found in group {group.uuid}; collection may have "
        "mutated mid-call"
    )


@group_handler
def capture_pin_deformation(group_uuid: str, vertex_group_identifier: str):
    """Record the per-frame shape of a deformable pin onto the cloth mesh.

    Use this for pins whose vertices ride along with an Armature, Lattice,
    Mesh Deform cage, animated Shape Keys, or a driver. The recording
    runs as a modal operator and continues after this call returns; poll
    ``get_pin_deformation_status`` until ``frame_count`` is non-zero.

    Press again any time the underlying animation changes. The recording
    does NOT update on its own. Refuses to start if the pin already
    carries manual Make-Keyframe vertex-co fcurves; clear those first.

    Args:
        group_uuid: UUID of the SHELL/SOLID/ROD group containing the pin
        vertex_group_identifier: Pin id in 'object::vertex_group' form
    """
    from .group import get_group_index_by_uuid

    group = get_active_group_by_uuid_helper(group_uuid)
    if group.object_type == "STATIC":
        raise MCPError(
            f"Group {group_uuid} is STATIC; use capture_static_deformation instead"
        )
    pin_item, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)

    obj = bpy.data.objects.get(_parse_pin_identifier(vertex_group_identifier)[0])
    if obj is None or obj.type != "MESH":
        raise MCPError(
            f"Pin '{vertex_group_identifier}': capture only applies to mesh pins"
        )
    from ...core.utils import has_deforming_modifier_stack
    if not has_deforming_modifier_stack(obj):
        raise MCPError(
            f"Object '{obj.name}' has no deforming modifier stack. "
            "Capture Deformation applies to pins whose vertices move "
            "(Armature, Lattice, Mesh Deform, Shape Keys, or drivers)."
        )

    pin_index = _resolve_pin_index(group, pin_item)
    group_index = get_group_index_by_uuid(group_uuid)
    bpy.ops.object.capture_pin_deformation(
        "EXEC_DEFAULT",
        group_index=group_index,
        pin_index=pin_index,
    )
    return {
        "message": (
            f"Started Capture Deformation for pin "
            f"'{vertex_group_identifier}'. The recording runs in the "
            "background; poll get_pin_deformation_status until it reports "
            "a non-zero frame_count."
        ),
        "group_uuid": group_uuid,
        "vertex_group_identifier": vertex_group_identifier,
        "object_uuid": obj_uuid,
        "vertex_group": vg_name,
    }


@group_handler
def clear_pin_deformation(group_uuid: str, vertex_group_identifier: str):
    """Discard the captured deformation cache for one pin.

    The pin returns to whatever motion source it had before (none, or
    manual Make-Keyframe fcurves if any).  If no manual fcurves exist
    the EMBEDDED_MOVE sentinel is also removed so the pin no longer
    appears animated.

    Args:
        group_uuid: UUID of the group containing the pin
        vertex_group_identifier: Pin id in 'object::vertex_group' form
    """
    from .group import get_group_index_by_uuid

    group = get_active_group_by_uuid_helper(group_uuid)
    if group.object_type == "STATIC":
        raise MCPError(
            f"Group {group_uuid} is STATIC; use clear_static_deformation instead"
        )
    pin_item, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)
    pin_index = _resolve_pin_index(group, pin_item)
    group_index = get_group_index_by_uuid(group_uuid)
    bpy.ops.object.clear_pin_deformation(
        "EXEC_DEFAULT",
        group_index=group_index,
        pin_index=pin_index,
    )
    return {
        "message": (
            f"Cleared captured deformation for pin "
            f"'{vertex_group_identifier}'"
        ),
        "group_uuid": group_uuid,
        "vertex_group_identifier": vertex_group_identifier,
        "object_uuid": obj_uuid,
        "vertex_group": vg_name,
    }


@group_handler
def get_pin_deformation_status(
    group_uuid: str, vertex_group_identifier: str,
):
    """Report the captured-deformation state of one pin.

    Returns four fields:

    - ``is_deforming``: True if the pin object's modifier stack will move
      vertices over the timeline (Armature, Lattice, ...).
    - ``has_cache``: True if a captured-deformation cache exists for the
      pin (in memory or on disk).
    - ``frame_count``: Number of frames in the cache, or 0 when absent.
    - ``has_captured_anim_flag``: The pin item's ``has_captured_anim``
      bool; should match ``has_cache`` after the load_post reconciler runs.

    Args:
        group_uuid: UUID of the group containing the pin
        vertex_group_identifier: Pin id in 'object::vertex_group' form
    """
    from ...core.pc2 import has_pin_anim_pc2
    from ...core.utils import has_deforming_modifier_stack
    from ...ui.dynamics.pin_capture_ops import pin_captured_frame_count

    group = get_active_group_by_uuid_helper(group_uuid)
    if group.object_type == "STATIC":
        raise MCPError(
            f"Group {group_uuid} is STATIC; use get_static_deformation_status instead"
        )
    pin_item, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)
    obj_name, _ = _parse_pin_identifier(vertex_group_identifier)
    obj = bpy.data.objects.get(obj_name)
    is_deforming = bool(obj is not None and has_deforming_modifier_stack(obj))
    has_cache = bool(
        obj is not None and obj.type == "MESH" and has_pin_anim_pc2(obj, vg_name)
    )
    return {
        "group_uuid": group_uuid,
        "vertex_group_identifier": vertex_group_identifier,
        "object_uuid": obj_uuid,
        "vertex_group": vg_name,
        "is_deforming": is_deforming,
        "has_cache": has_cache,
        "frame_count": int(pin_captured_frame_count(pin_item)),
        "has_captured_anim_flag": bool(
            getattr(pin_item, "has_captured_anim", False)
        ),
    }


# Map each tet override kwarg to its (value field, override flag). Setting a
# value also flips the override flag on; the encoder only forwards a field
# when its flag is set (see core/encoder/params.py:_encode_obj_tet_kwargs),
# otherwise the backend default applies.
_TET_OVERRIDE_FIELDS = {
    "ftetwild_edge_length_fac": "ftetwild_override_edge_length_fac",
    "ftetwild_epsilon": "ftetwild_override_epsilon",
    "ftetwild_stop_energy": "ftetwild_override_stop_energy",
    "ftetwild_num_opt_iter": "ftetwild_override_num_opt_iter",
    "ftetwild_optimize": "ftetwild_override_optimize",
    "ftetwild_simplify": "ftetwild_override_simplify",
    "ftetwild_coarsen": "ftetwild_override_coarsen",
    "tetgen_min_ratio": "tetgen_override_min_ratio",
    "tetgen_max_volume": "tetgen_override_max_volume",
}


@group_handler
def set_object_tet_settings(
    group_uuid: str,
    object_name: str,
    tet_backend: Optional[str] = None,
    ftetwild_edge_length_fac: Optional[float] = None,
    ftetwild_epsilon: Optional[float] = None,
    ftetwild_stop_energy: Optional[float] = None,
    ftetwild_num_opt_iter: Optional[int] = None,
    ftetwild_optimize: Optional[bool] = None,
    ftetwild_simplify: Optional[bool] = None,
    ftetwild_coarsen: Optional[bool] = None,
    tetgen_min_ratio: Optional[float] = None,
    tetgen_max_volume: Optional[float] = None,
    clear_overrides: Optional[list[str]] = None,
):
    """Set the per-object tetrahedralizer backend and overrides.

    SOLID meshes are tetrahedralized at build time, and each object in a group
    picks its backend and overrides independently (ignored for non-SOLID
    objects). Passing any override value also enables that override, and an
    override the call does not mention keeps whatever it already holds.

    An enabled override stays enabled until it is cleared: name it in
    ``clear_overrides`` to switch it off and hand the field back to the
    backend default. Clearing needs its own argument because a value argument
    left out already means "leave this one alone", so it has no spelling to
    spare for "turn this one off". A field named in ``clear_overrides`` and
    given a value in the same call is refused rather than resolved by
    argument order.

    Args:
        group_uuid: UUID of the group containing the object.
        object_name: Name of the assigned object in the group.
        tet_backend: "FTETWILD" (tolerant remesher, default) or "TETGEN"
            (preserves the input surface exactly, needs a clean closed manifold).
        ftetwild_edge_length_fac: fTetWild ideal tet edge length as a fraction
            of the bounding-box diagonal.
        ftetwild_epsilon: fTetWild envelope size as a fraction of the bbox diagonal.
        ftetwild_stop_energy: fTetWild AMIPS energy threshold (larger is faster).
        ftetwild_num_opt_iter: fTetWild maximum optimization passes.
        ftetwild_optimize: Improve cell quality (slower).
        ftetwild_simplify: Simplify the input surface before tetrahedralization.
        ftetwild_coarsen: Coarsen the input surface.
        tetgen_min_ratio: TetGen minimum radius-edge ratio.
        tetgen_max_volume: TetGen maximum tet volume (0 = uncapped).
        clear_overrides: Override fields to switch back off, so the backend
            default applies again, for example ["ftetwild_epsilon"]. Any name
            that is not an override field is refused.
    """
    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)

    updated: dict = {}
    if tet_backend is not None:
        backend = tet_backend.upper()
        if backend not in {"FTETWILD", "TETGEN"}:
            raise MCPError("tet_backend must be 'FTETWILD' or 'TETGEN'")
        assigned.tet_backend = backend
        updated["tet_backend"] = backend

    values = {
        "ftetwild_edge_length_fac": ftetwild_edge_length_fac,
        "ftetwild_epsilon": ftetwild_epsilon,
        "ftetwild_stop_energy": ftetwild_stop_energy,
        "ftetwild_num_opt_iter": ftetwild_num_opt_iter,
        "ftetwild_optimize": ftetwild_optimize,
        "ftetwild_simplify": ftetwild_simplify,
        "ftetwild_coarsen": ftetwild_coarsen,
        "tetgen_min_ratio": tetgen_min_ratio,
        "tetgen_max_volume": tetgen_max_volume,
    }
    cleared: list[str] = []
    if clear_overrides is not None:
        if not isinstance(clear_overrides, (list, tuple)):
            raise MCPError(
                "clear_overrides must be a list of override field names, got "
                f"{clear_overrides!r}"
            )
        unknown = sorted(
            set(clear_overrides) - set(_TET_OVERRIDE_FIELDS)
        )
        if unknown:
            raise MCPError(
                f"clear_overrides names no such override field: "
                f"{', '.join(unknown)}. Override fields are "
                f"{', '.join(sorted(_TET_OVERRIDE_FIELDS))}"
            )
        conflicting = sorted(
            field for field in set(clear_overrides) if values[field] is not None
        )
        if conflicting:
            raise MCPError(
                "clear_overrides and a value were both given for "
                f"{', '.join(conflicting)}; a value enables the override and "
                "clearing switches it off, so pass one or the other"
            )
        cleared = sorted(set(clear_overrides))

    for field, value in values.items():
        if value is None:
            continue
        setattr(assigned, field, value)
        setattr(assigned, _TET_OVERRIDE_FIELDS[field], True)
        updated[field] = value

    for field in cleared:
        setattr(assigned, _TET_OVERRIDE_FIELDS[field], False)

    return {
        "message": f"Updated tetrahedralizer settings for '{object_name}'",
        "group_uuid": group_uuid,
        "object_uuid": obj_uuid,
        "updated": updated,
        "cleared_overrides": cleared,
        "overrides": {
            field: bool(getattr(assigned, flag))
            for field, flag in _TET_OVERRIDE_FIELDS.items()
        },
    }


# ---------------------------------------------------------------------------
# Isolated (faceless / stray) vertex cleanup on STATIC colliders.
# A STATIC collider vertex in no face has no incident faces to average contact
# parameters over, so the solver aborts the build (Transfer raises a ValueError
# whose message contains "isolated vert"). These mirror the "Remove Isolated
# Vertices" panel button: detect previews, remove deletes. Scene-wide over all
# active STATIC colliders, matching the button. Light bmesh edit, synchronous.
# ---------------------------------------------------------------------------


@mcp_handler
def detect_isolated_static_vertices():
    """Report stray faceless vertices on active STATIC colliders that block Transfer.

    Scans every included, active STATIC collider mesh for vertices that
    belong to no triangle (no face). The solver build aborts on these, and
    Transfer reports a ValueError naming the object and the vertex indices.
    Read-only; pair with remove_isolated_static_vertices to delete them.
    """
    from ...ui.geometry_cleanup_ops import _static_isolated_offenders

    offenders = _static_isolated_offenders(bpy.context)
    objects = [
        {
            "object_name": obj.name,
            "isolated_count": len(idx),
            "isolated_indices": idx,
        }
        for obj, idx in offenders.items()
    ]
    return {
        "message": (
            f"Found isolated vertices on {len(objects)} STATIC collider(s)"
            if objects
            else "No isolated vertices on STATIC colliders"
        ),
        "objects": objects,
        "object_count": len(objects),
        "total_isolated": sum(o["isolated_count"] for o in objects),
    }


@mcp_handler
def remove_isolated_static_vertices():
    """Delete stray faceless vertices from active STATIC colliders so the scene transfers.

    Removes only vertices that belong to no triangle (with their loose
    edges); faces are untouched. Mirrors the Remove Isolated Vertices panel
    button and scans every included, active STATIC collider. Run
    detect_isolated_static_vertices first to preview what will be deleted.
    """
    from ...ui.geometry_cleanup_ops import (
        _delete_vertices,
        _static_isolated_offenders,
    )

    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass

    offenders = _static_isolated_offenders(bpy.context)
    if not offenders:
        return {
            "message": "No isolated vertices found on STATIC colliders",
            "removed_total": 0,
            "objects": [],
        }

    results, total = [], 0
    for obj, indices in offenders.items():
        removed = _delete_vertices(obj, indices)
        total += removed
        results.append({"object_name": obj.name, "removed": removed})

    from ...core.client import communicator as com

    com.set_error("")
    return {
        "message": f"Removed {total} isolated vertex(es). Transfer again.",
        "removed_total": total,
        "objects": results,
    }


@mcp_handler
def convert_to_particle_mesh(
    object_name: str,
    grain_radius: float,
    extra_spacing: float = 0.0,
    rng_seed: int = 0,
):
    """Replace a solid mesh with a cloud of grain centers for a Sand group.

    Destructive: the faces are discarded and the object becomes a faceless
    mesh of loose vertices carrying a render-only Particle Mesh modifier. The
    grain count is not chosen, it is whatever fills the volume at the given
    separation, and it comes back in the result.

    grain_radius is locked after conversion, since the non-overlapping spacing
    is derived from it, so pick it before converting rather than adjusting it
    afterward. A radius or spacing that fits no grain at all is refused with
    the object untouched.

    Args:
        object_name: Solid mesh object with faces, not already a particle mesh
        grain_radius: Physical grain radius, which is also the contact skin
        extra_spacing: Gap added between grains beyond touching. 0 packs them
            as densely as non-overlap allows
        rng_seed: Seed for the Poisson-disk seeding, for a repeatable cloud
    """
    from ...ui.dynamics.sand_ops import (
        build_and_commit_particle_mesh,
        seed_inside_eroded,
    )

    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise ValidationError(f"No such object: {object_name}")
    if obj.type != "MESH" or obj.data is None:
        raise ValidationError(f"Object '{object_name}' is {obj.type}, not a MESH")
    if not len(obj.data.polygons):
        raise ValidationError(
            f"Object '{object_name}' has no faces. Conversion seeds grains "
            "inside a closed solid mesh."
        )
    if obj.get("particle_mesh"):
        raise ValidationError(f"Object '{object_name}' is already a particle mesh")
    if grain_radius <= 0.0:
        raise ValidationError("grain_radius must be positive")

    # Seed here first, while the mesh is still intact, so a radius that fits
    # no grain is refused with nothing consumed and the caller can retry at a
    # smaller one. The commit replaces obj.data and removes the original
    # datablock once nothing else uses it, so a count read after it names a
    # remedy the caller no longer has the mesh to apply. seed_inside_eroded
    # writes nothing (it reads the evaluated mesh through a BVH and returns
    # arrays) and draws every random number from
    # np.random.default_rng(rng_seed), so it reaches the same count the
    # commit's own seeding pass will. The price is that a conversion which
    # succeeds seeds twice.
    cloud, _stats = seed_inside_eroded(obj, grain_radius, extra_spacing, rng_seed)
    if len(cloud) < 1:
        raise MCPError(
            f"No grains of radius {grain_radius} fit inside '{object_name}'. "
            "The object is unchanged. Reduce grain_radius or extra_spacing, "
            "or use a larger mesh."
        )

    n = build_and_commit_particle_mesh(
        obj, grain_radius, extra_spacing, rng_seed=rng_seed
    )
    if n < 1:
        raise MCPError(
            f"Converting '{object_name}' placed no grain, although the check "
            f"before it placed {len(cloud)}. The object is now an empty "
            "particle mesh and its original mesh is gone; rebuild the mesh "
            "before converting again."
        )
    return {
        "message": f"Converted '{object_name}' to {n} grain(s)",
        "object_name": object_name,
        "grain_count": n,
        "grain_radius": grain_radius,
        "extra_spacing": extra_spacing,
        "rng_seed": rng_seed,
    }


@mcp_handler
def recapture_all_deformations():
    """Re-capture every deforming STATIC collider and every animated pin.

    One pass over the whole scene, instead of calling
    capture_static_deformation and capture_pin_deformation per object. The
    statics are captured first and the pins after, since the two share the
    depsgraph and cannot run at once.

    The captures run in the background after this returns; poll
    get_static_deformation_status and get_pin_deformation_status until they
    report the frame counts you expect.
    """
    from ...ui.solver import SOLVER_OT_RecaptureAllDeformations

    if not SOLVER_OT_RecaptureAllDeformations.poll(bpy.context):
        raise MCPError(
            "Nothing to re-capture, or a capture or bake is already running. "
            "Re-capture applies to deforming STATIC colliders and animated "
            "pins in active groups."
        )
    bpy.ops.solver.recapture_all_deformations()
    return {
        "message": (
            "Started Re-capture All Deformations. The captures run in the "
            "background; poll get_static_deformation_status and "
            "get_pin_deformation_status until they report a frame count."
        ),
    }


@mcp_handler
def clear_all_deformations():
    """Delete every captured deformation cache in the scene.

    Covers all STATIC-collider deform caches and all animated-pin captures
    across the active groups, plus any cache orphaned by an object that was
    deleted or taken out of its group. The objects keep their deformers, so
    recapture_all_deformations rebuilds what this removes.
    """
    from ...ui.solver import SOLVER_OT_ClearAllDeformations

    if not SOLVER_OT_ClearAllDeformations.poll(bpy.context):
        raise MCPError(
            "No captured deformation cache to clear, or a capture or bake is "
            "already running."
        )
    bpy.ops.solver.clear_all_deformations()
    return {"message": "Cleared every captured deformation cache in the scene"}
