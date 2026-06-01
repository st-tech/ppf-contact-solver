"""Per-assigned-object and per-pin operation handlers.

Covers surfaces that all hang off an AssignedObject or PinVertexGroupItem:
pin settings, pin operations, static ops, velocity keyframes, collision
windows. Identifies groups by UUID and objects by name (name is resolved
to UUID up front so later lookups are rename-safe).
"""

from typing import Optional

import bpy  # pyright: ignore

from ...models.collection_utils import safe_update_index, sort_keyframes_by_frame
from ..decorators import (
    MCPError,
    ValidationError,
    group_handler,
)
from .group import get_active_group_by_uuid_helper


# ---------------------------------------------------------------------------
# Shared resolvers
# ---------------------------------------------------------------------------


def _resolve_assigned(group_uuid: str, object_name: str):
    """Return (group, assigned, object_uuid) for an object in a group.

    Raises MCPError if the group is unknown, the object is missing, or the
    object is not a member of the group.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    from ...core.uuid_registry import get_object_uuid

    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise MCPError(f"Object '{object_name}' not found in scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(f"Object '{object_name}' has no UUID")
    for assigned in group.assigned_objects:
        if assigned.uuid == obj_uuid:
            return group, assigned, obj_uuid
    raise MCPError(f"Object '{object_name}' not in group {group_uuid}")


def _parse_pin_identifier(vertex_group_identifier: str) -> tuple[str, str]:
    from ...models.groups import parse_pin_identifier
    return parse_pin_identifier(vertex_group_identifier, ValidationError)


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


def _check_vec3(name: str, v):
    from ...core.utils import check_vec3
    return check_vec3(name, v, ValidationError)


# ---------------------------------------------------------------------------
# Pin settings
# ---------------------------------------------------------------------------


_PIN_SETTABLE = {
    "included",
    "use_pin_duration",
    "pin_duration",
    "use_pull",
    "pull_strength",
    "pin_stiffness",
}


@group_handler
def set_pin_settings(
    group_uuid: str,
    vertex_group_identifier: str,
    included: Optional[bool] = None,
    use_pin_duration: Optional[bool] = None,
    pin_duration: Optional[int] = None,
    use_pull: Optional[bool] = None,
    pull_strength: Optional[float] = None,
    pin_stiffness: Optional[float] = None,
):
    """Set per-pin runtime settings (include/duration/pull/stiffness).

    Args:
        group_uuid: UUID of group
        vertex_group_identifier: Pin id in 'object::vertex_group' form
        included: Include this pin in the simulation
        use_pin_duration: Enable per-pin active duration
        pin_duration: Number of frames the pin is active
        use_pull: Use pull force instead of hard constraint
        pull_strength: Pull force strength
        pin_stiffness: Scale on this pin's moving (kinematic) constraint
            force; 1.0 default, only affects animated pins
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin, obj_uuid, vg_name = _resolve_pin(group, vertex_group_identifier)

    updates = {
        "included": included,
        "use_pin_duration": use_pin_duration,
        "pin_duration": pin_duration,
        "use_pull": use_pull,
        "pull_strength": pull_strength,
        "pin_stiffness": pin_stiffness,
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


def _apply_pin_op_fields(op, op_type: str, kwargs: dict):
    """Copy op-type-specific fields from kwargs onto the operation."""
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

    for k, v in kwargs.items():
        if v is None:
            continue
        if k not in allowed:
            raise ValidationError(
                f"Field '{k}' is not valid for op_type={op_type}; "
                f"allowed: {sorted(allowed)}"
            )
        if k in ("delta", "spin_axis", "spin_center", "spin_center_direction",
                 "scale_center", "scale_center_direction"):
            v = _check_vec3(k, v)
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
        delta: [x, y, z] translation for MOVE_BY (metres)
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
        torque_magnitude: Torque in newton-metres
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
# Static ops (per static object: Move/Spin/Scale of the whole object)
# ---------------------------------------------------------------------------


_STATIC_OP_TYPES = {"MOVE_BY", "SPIN", "SCALE"}


def _apply_static_op_fields(op, op_type: str, kwargs: dict):
    common = {"frame_start", "frame_end", "transition", "show_overlay"}
    if op_type == "MOVE_BY":
        allowed = common | {"delta"}
    elif op_type == "SPIN":
        allowed = common | {"spin_axis", "spin_angular_velocity"}
    elif op_type == "SCALE":
        allowed = common | {"scale_factor"}
    else:
        raise ValidationError(f"Unknown static op_type '{op_type}'")

    for k, v in kwargs.items():
        if v is None:
            continue
        if k not in allowed:
            raise ValidationError(
                f"Field '{k}' is not valid for static op_type={op_type}; "
                f"allowed: {sorted(allowed)}"
            )
        if k in ("delta", "spin_axis"):
            v = _check_vec3(k, v)
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
# Velocity keyframes (initial velocity driven by keyframed vectors)
# ---------------------------------------------------------------------------


@group_handler
def add_velocity_keyframe(
    group_uuid: str,
    object_name: str,
    frame: int,
    direction: list[float],
    speed: float,
):
    """Add a velocity keyframe at the given frame for an assigned object.

    Args:
        group_uuid: UUID of group
        object_name: Name of the assigned object
        frame: Blender frame number (>= 1)
        direction: [x, y, z] direction vector (normalized at runtime)
        speed: Velocity magnitude (m/s)
    """
    _, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if frame < 1:
        raise ValidationError("frame must be >= 1")
    direction = _check_vec3("direction", direction)

    for kf in assigned.velocity_keyframes:
        if kf.frame == frame:
            raise MCPError(f"Frame {frame} already has a velocity keyframe")

    kf = assigned.velocity_keyframes.add()
    kf.frame = frame
    kf.direction = direction
    kf.speed = float(speed)
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
    assigned.velocity_keyframes_index = 0
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


MAX_COLLISION_WINDOWS = 8


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
    assigned.collision_windows_index = 0
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
    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
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
    idx = -1
    for i, a in enumerate(group.assigned_objects):
        if a.uuid == obj_uuid:
            idx = i
            break
    if idx < 0:
        raise MCPError(f"Object '{object_name}' not in group {group_uuid}")
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
    group, assigned, obj_uuid = _resolve_assigned(group_uuid, object_name)
    if group.object_type != "STATIC":
        raise MCPError(
            f"Group {group_uuid} is {group.object_type}; clear requires STATIC"
        )

    idx = -1
    for i, a in enumerate(group.assigned_objects):
        if a.uuid == obj_uuid:
            idx = i
            break
    if idx < 0:
        raise MCPError(f"Object '{object_name}' not in group {group_uuid}")
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
